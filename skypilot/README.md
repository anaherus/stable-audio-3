# SkyPilot: SpliceGen-conditioned adapter sweep

Launches parallel managed jobs training SpliceGen-conditioned LoRA adapters for
`stable-audio-3-medium-base` (one job per LoRA rank), with wandb logging and
S3 checkpoint sync for spot recovery.

## Code delivery

By default the launcher **syncs your local working tree** to the VMs (respecting
`.gitignore`), so no git push is needed — just commit or not, and launch.
For reproducible launches pin a pushed commit instead:
`--git-url https://github.com/<you>/stable-audio-3.git [--git-ref <sha>]`.

## Prerequisites

1. **Credentials** — read automatically when available, or export in the shell:
   - wandb API key: from `WANDB_API_KEY` or `~/.netrc` (written by `wandb login`)
   - HF token: from `HF_TOKEN` or the local huggingface_hub cache (needed on the
     VM to download `stabilityai/stable-audio-3-medium-base`)
   - `GIT_TOKEN`: only for `--git-url` with a private fork
2. **A python with skypilot[aws]** installed.
3. AWS credentials with access to the `audio-science-hub` / `dit-melodic-loops`
   buckets (the VM itself uses the `instance_role_audioscience_ec2_skypilot`
   instance role).

## Launch

```bash
cd stable-audio-3
AWS_PROFILE=saml python skypilot/launch_sweep.py \
    sa3-sg-prepend --ranks 8 16 64 --mode spot_h100
```

This creates managed jobs `sa3-sg-prepend-r8`, `-r16`, `-r64`, each running
`scripts/train_lora_splicegen.py` with:

- dataset: `s3://audio-science-hub/datasets/splicegen_prod_v1_SAME_L_latents_noncommercial_license`
  synced to nvme scratch
- SAME-L `silence_pad_embed.pt` for latent silence padding
- checkpoints synced to `s3://audio-science-hub/checkpoints/sa3_sg_adapters/<job>/`;
  on spot preemption the restarted job auto-resumes from the latest S3 checkpoint
  and the same wandb run id.

Useful flags: `--dry-run`, `--detach`, `--mode demand_l40s` (cheap single-GPU
debug), `--extra_args "--extra_lr 5e-4"`.

## Monitor

```bash
sky jobs queue
sky jobs logs <job-id>
```

and the wandb project `sa3_sg_adapters` (runs are grouped by sweep name).
