"""LoRA + adapter fine-tuning of Stable Audio 3 with SpliceGen 3 conditioning.

Usage:
  uv run python scripts/train_lora_splicegen.py \
      --dataset_dir /opt/dlami/nvme/hfds/splicegen_prod_v1_SAME_L_latents_noncommercial_license \
      --pad_latent /path/to/SAME-L/silence_pad_embed.pt \
      --rank 16 --logger wandb --project sa3_sg_adapters --run_name prepend-r16
"""

# Disable HuggingFace progress bars BEFORE any imports
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import itertools
import json
import re
import subprocess
from pathlib import Path

import torch
import pytorch_lightning as pl

from safetensors.torch import load_file
from stable_audio_3.data.hf_latent_dataset import create_hf_latent_dataloader
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.loading_utils import copy_state_dict
from stable_audio_3.model_configs import base_models
from stable_audio_3.models.lora.utils import load_lora_checkpoint
from stable_audio_3.training.diffusion import DiffusionCondInpaintDemoCallback
from stable_audio_3.training.splicegen_adapter import (
    SpliceGenAdapterTrainingWrapper,
    SpliceGenFullFTTrainingWrapper,
)

DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "splicegen_prepend_medium.json"

# State-dict key prefixes of modules intentionally absent from / new to the
# checkpoint (removed prompt conditioner, new SpliceGen conditioning modules).
EXPECTED_MISSING_PREFIXES = (
    "conditioner.conditioners.prompt",       # removed t5gemma conditioner
)
EXPECTED_NEW_PATTERNS = (
    "conditioner.conditioners.cl_siso",
    "conditioner.conditioners.chroma",
    "conditioner.conditioners.bpm",
    "to_prepend_embed",
    "modular_local_embeds",
)


def load_model(model_name: str, model_config_path: str, device: torch.device,
               dtype: torch.dtype = torch.bfloat16, from_scratch: bool = False):
    """Build the SpliceGen-conditioned model and load pretrained base weights.

    The model architecture comes from the local model config; the weights
    from the HF base checkpoint. Verifies that the only key mismatches are the
    removed prompt conditioner and the new conditioning modules.

    With ``from_scratch=True``, only the ``pretransform.*`` weights (the SAME-L
    autoencoder the dataset latents were encoded with — needed for demo
    decoding) are loaded from the base checkpoint; the DiT and all conditioners
    keep their fresh random initialization.
    """
    if model_name not in base_models:
        raise ValueError(f"Requires a base model. Got '{model_name}', valid: {list(base_models)}")
    _, local_ckpt = base_models[model_name].resolve()

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_diffusion_cond_from_config(model_config)

    ckpt_sd = load_file(local_ckpt)

    if from_scratch:
        pretransform_sd = {k: v for k, v in ckpt_sd.items() if k.startswith("pretransform.")}
        if not pretransform_sd:
            raise RuntimeError(
                f"No pretransform.* keys found in the base checkpoint {local_ckpt}; "
                "cannot load the autoencoder for a from-scratch run"
            )
        print(
            f"From-scratch init: loading only {len(pretransform_sd)} pretransform tensors "
            f"from the base checkpoint; DiT + conditioners stay randomly initialized"
        )
        copy_state_dict(model, pretransform_sd)
        model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        if model.pretransform is not None:
            model.pretransform.enable_grad = False
        return model, model_config

    model_sd_keys = set(model.state_dict().keys())
    model_sd = model.state_dict()

    # Checkpoint keys that won't load (must all belong to the removed prompt conditioner)
    skipped = [
        k for k in ckpt_sd
        if k not in model_sd_keys or ckpt_sd[k].shape != model_sd[k].shape
    ]
    unexpected_skipped = [k for k in skipped if not k.startswith(EXPECTED_MISSING_PREFIXES)]
    if unexpected_skipped:
        raise RuntimeError(
            "Unexpected checkpoint keys skipped while loading base weights "
            f"(architecture drift?): {unexpected_skipped[:10]}"
        )
    print(f"Skipping {len(skipped)} checkpoint keys from removed prompt conditioner")

    # Model keys that stay freshly initialized (must all be new SpliceGen modules)
    fresh = [k for k in model_sd_keys if k not in ckpt_sd]
    unexpected_fresh = [k for k in fresh if not any(p in k for p in EXPECTED_NEW_PATTERNS)]
    if unexpected_fresh:
        raise RuntimeError(
            "Unexpected freshly-initialized model keys (missing from checkpoint): "
            f"{unexpected_fresh[:10]}"
        )
    print(f"{len(fresh)} model keys are new SpliceGen conditioning modules (trained from scratch)")

    copy_state_dict(model, ckpt_sd)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class InitialStepCallback(pl.Callback):
    """Fast-forward the step counters when resuming from a weights-only checkpoint.

    Adapter checkpoints saved before the trainer-state fix carry no loop state,
    so Lightning would restart from step 0 — retraining the full step budget
    and breaking the wandb step axis. This sets the optimizer-step and
    logging-step counters to the checkpoint's step so max_steps and wandb
    continue from where the previous run stopped. (Optimizer moment estimates
    are still fresh; they re-warm within a few hundred steps.)
    """

    def __init__(self, initial_step: int):
        self.initial_step = initial_step

    def on_train_start(self, trainer, module):
        epoch_loop = trainer.fit_loop.epoch_loop
        epoch_loop._batches_that_stepped = self.initial_step
        step_progress = epoch_loop.automatic_optimization.optim_progress.optimizer.step
        step_progress.total.completed = self.initial_step
        print(f"Fast-forwarded global step to {self.initial_step}")


class S3SyncCallback(pl.Callback):
    """Sync the checkpoint directory to S3 after each checkpoint save."""

    def __init__(self, checkpoint_dir: str, s3_uri: str, every_n_train_steps: int):
        self.checkpoint_dir = checkpoint_dir
        self.s3_uri = s3_uri.rstrip("/")
        self.every_n_train_steps = every_n_train_steps

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.every_n_train_steps != 0:
            return
        if trainer.global_rank != 0:
            return
        if not os.path.isdir(self.checkpoint_dir):
            return
        try:
            subprocess.run(
                ["aws", "s3", "sync", self.checkpoint_dir, self.s3_uri, "--only-show-errors"],
                check=True,
                timeout=1800,
            )
            print(f"Synced checkpoints to {self.s3_uri}")
        except Exception as e:
            print(f"S3 checkpoint sync failed (continuing): {e}")


def find_latest_s3_checkpoint(s3_uri: str):
    """Return the S3 URI of the newest .ckpt under s3_uri (by modification time), or None."""
    try:
        out = subprocess.run(
            ["aws", "s3", "ls", "--recursive", s3_uri.rstrip("/") + "/"],
            capture_output=True, text=True, check=True, timeout=300,
        ).stdout
    except Exception as e:
        print(f"Could not list {s3_uri}: {e}")
        return None
    # Lines look like: "2026-07-11 01:23:45   123456789 path/to/epoch=0-step=1000.ckpt".
    # Sort by the leading timestamp — lexicographic sort on the key would rank
    # step=2000 above step=10000.
    entries = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".ckpt"):
            entries.append(((parts[0], parts[1]), parts[-1]))
    if not entries:
        return None
    bucket = s3_uri.split("/")[2]
    latest = max(entries)[1]
    return f"s3://{bucket}/{latest}"


def train(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(args.seed, workers=True)

    # Pin each DDP worker to its own GPU: this runs before Lightning assigns
    # devices, and defaulting to "cuda" would stack all workers' model builds
    # (base + fp32 adapter params) on GPU 0 — OOM at high LoRA ranks.
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if args.from_scratch and not args.full_finetune:
        raise ValueError("--from_scratch requires --full_finetune (adapters on random weights make no sense)")

    model, model_config = load_model(
        args.model, args.model_config, device,
        dtype=torch.float32 if args.full_finetune else torch.bfloat16,
        from_scratch=args.from_scratch,
    )

    latent_rate = model_config["sample_rate"] / model.pretransform.downsampling_ratio

    dataloader = create_hf_latent_dataloader(
        dataset_path=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pad_to_n_frames=args.latent_frames,
        sample_rate=latent_rate,
        pad_latent_path=args.pad_latent,
        random_crop=True,
        sample_on_duration=True,
        sample_on_proba=True,
        min_bpm=20,
        max_bpm=250,
        seed=args.seed,
    )

    # Resume: explicit checkpoint > latest checkpoint found in S3
    resume_checkpoint = args.lora_checkpoint
    resume_source_name = args.lora_checkpoint  # keeps the step-bearing filename
    if resume_checkpoint is None and args.s3_checkpoint_uri:
        latest = find_latest_s3_checkpoint(args.s3_checkpoint_uri)
        if latest is not None:
            local_resume = os.path.join(args.save_dir, "resume.ckpt")
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Resuming from {latest}")
            subprocess.run(["aws", "s3", "cp", latest, local_resume, "--only-show-errors"], check=True)
            resume_checkpoint = local_resume
            resume_source_name = latest

    lora_state_dict = None
    resume_has_trainer_state = args.full_finetune  # full-FT ckpts always carry it
    initial_step = args.initial_step
    if resume_checkpoint and not args.full_finetune:
        # Adapter checkpoints saved after the trainer-state fix are full
        # Lightning checkpoints (with a slimmed state_dict) and go through
        # trainer.fit(ckpt_path=...), which restores loops + optimizer and the
        # adapter weights (wrapper's on_load_checkpoint). Weights-only adapter
        # checkpoints (pre-fix) are loaded into the model here, and the step
        # counter is fast-forwarded from the filename via InitialStepCallback.
        if resume_checkpoint.endswith(".ckpt"):
            peek = torch.load(resume_checkpoint, map_location="cpu", weights_only=False, mmap=True)
            resume_has_trainer_state = "loops" in peek
            del peek
        if not resume_has_trainer_state:
            lora_state_dict, _ = load_lora_checkpoint(resume_checkpoint)
            if initial_step is None:
                m = re.search(r"step=(\d+)", resume_source_name or "")
                if m:
                    initial_step = int(m.group(1))
                else:
                    print(
                        "WARNING: resuming from a weights-only checkpoint without a "
                        "step=N filename or --initial_step; step counter restarts at 0"
                    )

    lora_config = None
    if not args.full_finetune:
        lora_config = {
            "rank": args.rank,
            "alpha": args.lora_alpha if args.lora_alpha is not None else args.rank,
            "adapter_type": args.adapter_type,
            "dropout": args.dropout,
            "include": args.include,
            "exclude": args.exclude,
        }
    optimizer_config = {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": args.lr,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                },
            }
        }
    }

    training_config = model_config.get("training", {})

    wrapper_cls = SpliceGenFullFTTrainingWrapper if args.full_finetune else SpliceGenAdapterTrainingWrapper
    training_wrapper = wrapper_cls(
        model,
        extra_lr=args.extra_lr,
        mask_loss_weight=training_config.get("mask_loss_weight", 1.0),
        mask_padding_attention=True,
        silence_extension_scale_seconds=training_config.get("silence_extension_scale_seconds", 4.0),
        use_ema=False,
        log_loss_info=False,
        optimizer_configs=optimizer_config,
        pre_encoded=True,
        timestep_sampler=training_config.get("timestep_sampler", "trunc_logit_normal"),
        timestep_sampler_options={},
        inpainting_config=training_config.get(
            "inpainting", {"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}}
        ),
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config,
        lora_state_dict=lora_state_dict,
        log_every_n_steps=args.log_every,
        ot_coupling=training_config.get("ot_coupling", True),
        base_precision=args.base_precision,
    )

    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(
            project=args.project,
            name=args.run_name,
            group=args.group,
            id=args.run_id,
            resume="allow" if args.run_id else None,
        )
        logger.watch(training_wrapper, log_freq=1000)
        run_dir = args.run_name or (logger.experiment.id if isinstance(logger.experiment.id, str) else "run")
        checkpoint_dir = os.path.join(args.save_dir, args.project, run_dir, "checkpoints")
    elif args.logger == "csv":
        logger = pl.loggers.CSVLogger(args.save_dir)
        checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    else:
        logger = None
        checkpoint_dir = os.path.join(args.save_dir, "checkpoints")

    if logger is not None:
        args_dict = vars(args).copy()
        args_dict["model_config_dict"] = model_config
        try:
            logger.log_hyperparams(args_dict)
        except Exception as e:
            print(f"Could not log hyperparams: {e}")

    # save_top_k=1: keep only the newest checkpoint on local disk (large-rank
    # checkpoints are multi-GB and filled the VM disk at save_top_k=-1, hanging
    # rank 0 mid-save and timing out NCCL). The S3 history still accumulates
    # every checkpoint, since `aws s3 sync` never deletes already-synced files.
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=1
    )

    # Fixed demo batch: full-mask generation conditioned on batch metadata
    # (SpliceGen's demo_cond_from_batch), decoded through the SAME-L decoder.
    demo_dl = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=args.num_demos,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=dataloader.collate_fn,
    )
    demo_batch = next(iter(demo_dl))
    _, metadata = demo_batch
    for j, md in enumerate(metadata[: args.num_demos]):
        print(
            f"Demo sample {j}: bpm={md.get('bpm')} seconds_start={md.get('seconds_start'):.2f} "
            f"seconds_total={md.get('seconds_total'):.2f}"
        )
    demo_dl = itertools.cycle([demo_batch])

    demo_callback = DiffusionCondInpaintDemoCallback(
        demo_every=args.demo_every,
        sample_size=model_config.get("sample_size"),
        sample_rate=model_config.get("sample_rate"),
        demo_steps=50,
        demo_cfg_scales=[2, 4, 7],
        inpaint_demo_config={"num_full_mask": args.num_demos},
        demo_dl=demo_dl,
    )

    callbacks = [ckpt_callback, ExceptionCallback(), demo_callback, pl.callbacks.ModelSummary(max_depth=2)]

    if resume_checkpoint and not resume_has_trainer_state and initial_step:
        callbacks.append(InitialStepCallback(initial_step))

    if args.s3_checkpoint_uri:
        callbacks.append(
            S3SyncCallback(checkpoint_dir, args.s3_checkpoint_uri, every_n_train_steps=args.checkpoint_every)
        )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=args.accum_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_steps=args.steps,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    ckpt_path = resume_checkpoint if resume_has_trainer_state else None
    trainer.fit(training_wrapper, dataloader, ckpt_path=ckpt_path)

    # Final export
    if trainer.global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.full_finetune:
            export_path = os.path.join(checkpoint_dir, "model_final.safetensors")
            training_wrapper.export_model_safetensors(export_path)
        else:
            export_path = os.path.join(checkpoint_dir, "adapter_final.safetensors")
            training_wrapper.export_adapter_safetensors(export_path)
        print(f"Exported final weights to {export_path}")


def main():
    p = argparse.ArgumentParser(description="SpliceGen-conditioned LoRA fine-tuning for Stable Audio 3")
    p.add_argument("--model", choices=list(base_models), default="medium-base")
    p.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG),
                   help="Path to the SpliceGen-conditioned model config JSON")
    p.add_argument("--dataset_dir", required=True,
                   help="Path to the HF save_to_disk dataset of pre-encoded SAME-L latents")
    p.add_argument("--pad_latent", default=None,
                   help="Path to SAME-L silence_pad_embed.pt (latent silence padding). Zeros if omitted.")
    p.add_argument("--latent_frames", type=int, default=173,
                   help="Latent crop/pad length in frames (173 = ~16s at SAME-L rate)")
    # Adapter (ignored with --full_finetune)
    p.add_argument("--full_finetune", action="store_true",
                   help="Train all model weights (no adapters); checkpoints are full Lightning checkpoints")
    p.add_argument("--from_scratch", action="store_true",
                   help="Randomly initialize the DiT + conditioners (only the pretransform is loaded "
                        "from the base checkpoint). Requires --full_finetune.")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=None)
    p.add_argument("--adapter_type",
                   choices=["lora", "dora", "dora-rows", "dora-cols", "bora"],
                   default="dora-rows")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=None)
    p.add_argument("--base_precision", choices=["bf16", "bfloat16", "fp16", "float16"], default="bf16")
    p.add_argument("--lora_checkpoint", default=None, help="Adapter checkpoint to resume from")
    p.add_argument("--initial_step", type=int, default=None,
                   help="Fast-forward the step counter when resuming a weights-only adapter "
                        "checkpoint (default: parsed from the checkpoint's step=N filename)")
    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--extra_lr", type=float, default=None,
                   help="LR for the from-scratch conditioning modules (default: same as --lr)")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accum_batches", type=int, default=1)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # Logging / checkpoints
    p.add_argument("--logger", choices=["wandb", "csv", "none"], default="wandb")
    p.add_argument("--project", type=str, default="sa3_sg_adapters")
    p.add_argument("--group", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None, help="Stable wandb run id for resume")
    p.add_argument("--save_dir", type=str, default="./adapter_checkpoints")
    p.add_argument("--s3_checkpoint_uri", type=str, default=None,
                   help="S3 prefix to sync checkpoints to and auto-resume from (spot recovery)")
    p.add_argument("--checkpoint_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--demo_every", type=int, default=1000)
    p.add_argument("--num_demos", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
