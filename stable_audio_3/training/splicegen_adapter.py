"""Training wrapper for SpliceGen-conditioned adapters on Stable Audio 3.

Trains, alongside the LoRA parameters:

- ``conditioner.conditioners.{cl_siso, chroma, bpm}`` projections
- the DiT ``to_prepend_embed`` head (prepend routing is new for medium-base)
- the per-block ``modular_local_embeds`` projections (zero-initialized, carry
  the bpm cycle signal)

These modules are unfrozen, kept in fp32, added to the optimizer as a separate
parameter group, and saved in checkpoints together with the LoRA tensors.
"""

import typing as tp

import torch

from ..models.lora import get_lora_params, get_lora_state_dict
from ..models.lora.utils import name_is_lora
from .diffusion import DiffusionCondTrainingWrapper
from .utils import create_optimizer_from_config, create_scheduler_from_config

# Substring patterns identifying the from-scratch adapter modules
# (matched against parameter names of the ConditionedDiffusionModelWrapper).
ADAPTER_EXTRA_PATTERNS = (
    "to_prepend_embed",
    "modular_local_embeds",
    "conditioners.cl_siso",
    "conditioners.chroma",
    "conditioners.bpm",
)

# Module-name patterns to exclude from LoRA parametrization: the from-scratch
# modules (LoRA on freshly initialized layers is pointless) and the pretrained
# seconds_total conditioner (see docs/workflows/lora.md on conditioner hijacking).
ADAPTER_LORA_EXCLUDE = (
    "to_prepend_embed",
    "modular_local_embeds",
    "conditioners",
)


def name_is_adapter_extra(name: str) -> bool:
    return any(p in name for p in ADAPTER_EXTRA_PATTERNS) and not name_is_lora(name)


def get_adapter_extra_state_dict(model) -> tp.Dict[str, torch.Tensor]:
    """State dict of the from-scratch adapter modules (keys relative to *model*)."""
    return {k: v for k, v in model.state_dict().items() if name_is_adapter_extra(k)}


class SpliceGenAdapterTrainingWrapper(DiffusionCondTrainingWrapper):
    """LoRA training wrapper that also trains the SpliceGen conditioning modules.

    Args:
        extra_lr: Learning rate for the from-scratch modules. Defaults to the
            LoRA optimizer learning rate.

    All other arguments are forwarded to ``DiffusionCondTrainingWrapper``.
    """

    def __init__(self, *args, extra_lr: tp.Optional[float] = None, **kwargs):
        lora_config = kwargs.get("lora_config")
        if lora_config is not None:
            # Never LoRA-parametrize the from-scratch modules or the conditioner.
            exclude = list(lora_config.get("exclude") or [])
            for pattern in ADAPTER_LORA_EXCLUDE:
                if pattern not in exclude:
                    exclude.append(pattern)
            lora_config = {**lora_config, "exclude": exclude}
            kwargs["lora_config"] = lora_config

        super().__init__(*args, **kwargs)

        self.extra_lr = extra_lr
        # Checkpoints slim `state_dict` down to adapter tensors (see
        # on_save_checkpoint); Lightning's restore must not require full coverage.
        self.strict_loading = False

        if self.lora_config is None:
            raise ValueError("SpliceGenAdapterTrainingWrapper requires a lora_config")

        # Unfreeze the from-scratch modules and keep their params in fp32
        # (cast_base_to_precision downcast them together with the frozen base).
        self._extra_param_names = []
        for name, param in self.diffusion.named_parameters():
            if name_is_adapter_extra(name):
                param.data = param.data.to(torch.float32)
                param.requires_grad_(True)
                self._extra_param_names.append(name)

        if len(self._extra_param_names) == 0:
            raise ValueError(
                "No from-scratch adapter parameters found. Is the model built from "
                "a SpliceGen conditioning config (prepend + modular local cond)?"
            )

        # New conditioners run in train mode so stochastic augmentation
        # (chroma top-K dropout) is active during training.
        for cond_id in ("cl_siso", "chroma", "bpm"):
            if cond_id in self.diffusion.conditioner.conditioners:
                self.diffusion.conditioner.conditioners[cond_id].train()

        n_extra = sum(
            p.numel() for n, p in self.diffusion.named_parameters() if n in set(self._extra_param_names)
        )
        print(f"SpliceGen adapter: {len(self._extra_param_names)} from-scratch tensors ({n_extra / 1e6:.2f}M params) trained alongside LoRA")

    def _extra_params(self):
        names = set(self._extra_param_names)
        return [p for n, p in self.diffusion.named_parameters() if n in names]

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs["diffusion"]

        lora_params = [
            *get_lora_params(self.diffusion.model),
            *get_lora_params(self.diffusion.conditioner),
        ]
        extra_params = self._extra_params()

        base_lr = diffusion_opt_config["optimizer"]["config"].get("lr")
        param_groups = [
            {"params": lora_params},
            {"params": extra_params, "lr": self.extra_lr if self.extra_lr is not None else base_lr},
        ]

        opt_diff = create_optimizer_from_config(diffusion_opt_config["optimizer"], param_groups)

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config["scheduler"], opt_diff)
            return [opt_diff], [{"scheduler": sched_diff, "interval": "step"}]

        return [opt_diff]

    def on_save_checkpoint(self, checkpoint):
        # Slim `state_dict` down to LoRA tensors + from-scratch module weights
        # (key spaces follow the parent convention: keys relative to
        # diffusion.model / diffusion.conditioner so load_state_dict(strict=False)
        # on both modules restores everything). Unlike the original
        # weights-only implementation, everything else Lightning saved
        # (loops, optimizer_states, lr_schedulers, ...) is kept, so
        # trainer.fit(ckpt_path=...) restores the global step and optimizer —
        # spot-preemption resume no longer restarts the step budget from 0.
        # The optimizer state only covers trainable params, so checkpoints grow
        # by ~2x the adapter size, not by the full model.
        checkpoint["state_dict"] = {
            **get_lora_state_dict(self.diffusion.model),
            **get_lora_state_dict(self.diffusion.conditioner),
            **get_adapter_extra_state_dict(self.diffusion.model),
            **get_adapter_extra_state_dict(self.diffusion.conditioner),
        }
        checkpoint["lora_config"] = self.lora_config

    def on_load_checkpoint(self, checkpoint):
        # Adapter weights live in a module-relative key space that Lightning's
        # own state_dict load (strict_loading=False) cannot map; apply them
        # here. Weight-only checkpoints (pre trainer-state fix) take the
        # manual pre-fit load path in the train script instead.
        state_dict = checkpoint.get("state_dict") or {}
        if state_dict:
            load_adapter_into_model(self.diffusion, dict(state_dict))

    def export_adapter_safetensors(self, path):
        """Export LoRA + from-scratch tensors as safetensors with embedded config."""
        from ..models.lora import save_lora_safetensors

        state_dict = {
            **get_lora_state_dict(self.diffusion.model),
            **get_lora_state_dict(self.diffusion.conditioner),
            **get_adapter_extra_state_dict(self.diffusion.model),
            **get_adapter_extra_state_dict(self.diffusion.conditioner),
        }
        save_lora_safetensors(state_dict, self.lora_config, path)


class SpliceGenFullFTTrainingWrapper(DiffusionCondTrainingWrapper):
    """Full fine-tuning wrapper for the SpliceGen-conditioned model (no adapters).

    Trains every DiT and conditioner parameter in fp32 (pretransform stays
    frozen). The from-scratch conditioning modules get their own optimizer
    parameter group so they can use a higher learning rate than the pretrained
    weights. Checkpointing is the default Lightning full checkpoint (weights +
    optimizer + trainer state), so resume restores the global step.

    Args:
        extra_lr: Learning rate for the from-scratch conditioning modules.
            Defaults to the base optimizer learning rate.
    """

    def __init__(self, *args, extra_lr: tp.Optional[float] = None, **kwargs):
        if kwargs.get("lora_config") is not None:
            raise ValueError("SpliceGenFullFTTrainingWrapper does not take a lora_config")
        super().__init__(*args, **kwargs)
        self.extra_lr = extra_lr

        for module in (self.diffusion.model, self.diffusion.conditioner):
            for param in module.parameters():
                param.data = param.data.to(torch.float32)
                param.requires_grad_(True)
            module.train()
        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.requires_grad_(False)

        n_total = sum(
            p.numel()
            for m in (self.diffusion.model, self.diffusion.conditioner)
            for p in m.parameters()
            if p.requires_grad
        )
        print(f"SpliceGen full FT: {n_total / 1e6:.2f}M trainable params")

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs["diffusion"]
        base_lr = diffusion_opt_config["optimizer"]["config"].get("lr")

        base_params, extra_params = [], []
        for prefix, module in (("model", self.diffusion.model), ("conditioner", self.diffusion.conditioner)):
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{prefix}.{name}"
                (extra_params if name_is_adapter_extra(full_name) else base_params).append(param)

        param_groups = [
            {"params": base_params},
            {"params": extra_params, "lr": self.extra_lr if self.extra_lr is not None else base_lr},
        ]
        opt_diff = create_optimizer_from_config(diffusion_opt_config["optimizer"], param_groups)

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config["scheduler"], opt_diff)
            return [opt_diff], [{"scheduler": sched_diff, "interval": "step"}]

        return [opt_diff]

    def export_model_safetensors(self, path):
        """Export the full fine-tuned model (DiT + conditioner) as safetensors."""
        from safetensors.torch import save_file

        state_dict = {
            **{f"model.{k}": v.contiguous() for k, v in self.diffusion.model.state_dict().items()},
            **{f"conditioner.{k}": v.contiguous() for k, v in self.diffusion.conditioner.state_dict().items()},
        }
        save_file(state_dict, path)


def load_adapter_into_model(model, state_dict):
    """Load a SpliceGen adapter checkpoint state dict into a diffusion wrapper.

    Both LoRA tensors and from-scratch module weights use keys relative to
    ``model.model`` (DiT) and ``model.conditioner``; strict=False loading on
    each covers the merged key space.
    """
    from ..models.lora import prepare_dora_state_dict

    prepare_dora_state_dict(state_dict)
    model.model.load_state_dict(state_dict, strict=False)
    model.conditioner.load_state_dict(state_dict, strict=False)
