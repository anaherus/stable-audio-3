"""Package a trained SpliceGen-conditioned SA3 checkpoint as an eval "clean package".

Produces a directory with `config.json` (SA3 model config) + `model.ckpt`
(full merged state dict, keys as in ConditionedDiffusionModelWrapper.state_dict()),
the layout the audio-science-genai evaluation framework expects for
`--checkpoint <s3://... | local dir>`.

Adapter checkpoints are merged into the base weights (no LoRA machinery needed
at eval time); full-FT Lightning checkpoints are converted by stripping the
`diffusion.` prefix.

Usage:
  python scripts/package_eval_checkpoint.py \
      --checkpoint /tmp/sa3_final_ckpts/r64.ckpt --kind adapter \
      --out_dir /tmp/sa3_eval_packages/sa3-sg-prepend-r64 \
      [--s3_uri s3://dit-melodic-loops/clean_checkpoints_genai/sa3_sg/sa3-sg-prepend-r64/]
"""

import argparse
import json
import shutil
import subprocess
from functools import partial
from pathlib import Path

import torch

from stable_audio_3.models.lora import add_lora, merge_lora, LoRAParametrization
from stable_audio_3.training.splicegen_adapter import load_adapter_into_model

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora_splicegen import DEFAULT_MODEL_CONFIG, load_model


def package_adapter(ckpt_path: str, model_config_path: str) -> tuple[dict, dict]:
    """Base + adapter -> merged full state dict."""
    device = torch.device("cpu")
    model, model_config = load_model("medium-base", model_config_path, device, dtype=torch.float32)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_config = ckpt["lora_config"]
    rank = lora_config["rank"]
    alpha = lora_config.get("alpha", rank)
    adapter_type = lora_config.get("adapter_type", "lora")
    parametrize_cfg = {
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=alpha, adapter_type=adapter_type)
        },
        torch.nn.Conv1d: {
            "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=alpha, adapter_type=adapter_type)
        },
    }
    add_lora(model.model, parametrize_cfg, include=lora_config.get("include"), exclude=lora_config.get("exclude"))
    add_lora(model.conditioner, parametrize_cfg, include=lora_config.get("include"), exclude=lora_config.get("exclude"))
    load_adapter_into_model(model, ckpt["state_dict"])

    merge_lora(model.model)
    merge_lora(model.conditioner)

    sd = model.state_dict()
    assert not any("lora" in k or "parametrizations" in k for k in sd), "merge left parametrization keys"
    return sd, model_config


def package_fullft(ckpt_path: str, model_config_path: str) -> tuple[dict, dict]:
    """Full-FT Lightning checkpoint -> plain model state dict."""
    with open(model_config_path) as f:
        model_config = json.load(f)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = {k[len("diffusion."):]: v.clone() for k, v in ckpt["state_dict"].items() if k.startswith("diffusion.")}
    if not sd:
        raise ValueError("No diffusion.* keys found; is this a full-FT Lightning checkpoint?")
    return sd, model_config


def verify(sd: dict, model_config: dict) -> None:
    """The packaged state dict must exactly cover a freshly built model."""
    from stable_audio_3.factory import create_diffusion_cond_from_config

    model = create_diffusion_cond_from_config(model_config)
    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    missing = sorted(model_keys - sd_keys)
    unexpected = sorted(sd_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"Key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    print(f"verified: {len(sd_keys)} tensors cover the model exactly")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Trained checkpoint (.ckpt)")
    p.add_argument("--kind", choices=["adapter", "fullft"], required=True)
    p.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    p.add_argument("--out_dir", required=True, help="Local package dir to create")
    p.add_argument("--s3_uri", default=None, help="Optional S3 prefix to upload the package to")
    args = p.parse_args()

    if args.kind == "adapter":
        sd, model_config = package_adapter(args.checkpoint, args.model_config)
    else:
        sd, model_config = package_fullft(args.checkpoint, args.model_config)

    verify(sd, model_config)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    torch.save({"state_dict": sd}, out_dir / "model.ckpt")
    size_gb = (out_dir / "model.ckpt").stat().st_size / 1e9
    print(f"wrote {out_dir} (model.ckpt {size_gb:.2f} GB)")

    if args.s3_uri:
        subprocess.run(
            ["aws", "s3", "sync", str(out_dir), args.s3_uri.rstrip("/") + "/", "--only-show-errors"],
            check=True,
        )
        print(f"uploaded to {args.s3_uri}")


if __name__ == "__main__":
    main()
