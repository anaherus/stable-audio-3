#!/usr/bin/env python3
"""Launch SkyPilot managed jobs for the SpliceGen-conditioned SA3 adapter sweep.

Usage (from the repo root; wandb key is read from the shell or ~/.netrc):

    python skypilot/launch_sweep.py sa3-sg-prepend --ranks 8 16 64

Requires a python with `skypilot[aws]` installed. Mirrors the conventions of
audio-science-genai/skypilot.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

import sky
from sky import skypilot_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------- infra conventions (mirrors audio-science-genai/skypilot) ----------

AWS_REMOTE_IDENTITY = "instance_role_audioscience_ec2_skypilot"
JOBS_BUCKET_BASE = "s3://audio-science-hub/skypilot/managed-jobs"
CHECKPOINTS_BUCKET_BASE = "s3://audio-science-hub/checkpoints/sa3_sg_adapters"

IMAGE_IDS = {
    "us-west-2": "ami-082f1d05bd5419df0",
    "us-east-2": "ami-0a8a98cb53617e30d",
}


def _base(**kw) -> sky.Resources:
    # Dataset is ~170GB of arrow shards synced to local disk; leave headroom.
    # image_id defaults to None (SkyPilot default GPU image with NVIDIA drivers);
    # the audio-science AMIs are only registered for the 8-GPU p5 fleet.
    return sky.Resources(
        disk_size=kw.pop("disk_size", 500),
        disk_tier=kw.pop("disk_tier", "medium"),
        image_id=kw.pop("image_id", None),
        **kw,
    )


RESOURCES_BY_MODE: dict[str, list[sky.Resources]] = {
    # Single-GPU modes: adapter training (~165M trainable params, 240-token
    # sequences) fits comfortably on one card; 8-GPU nodes are overkill.
    # H100:1 spot in any AWS region (no machine-type or on-demand fallback).
    "spot_h100_1": [_base(infra="aws", accelerators="H100:1", use_spot=True)],
    "spot_l40s": [
        _base(infra="aws/us-east-2", accelerators="L40S:1", use_spot=True),
        _base(infra="aws/us-west-2", accelerators="L40S:1", use_spot=True),
    ],
    "demand_l40s": [_base(infra="aws/us-west-2", accelerators="L40S:1", use_spot=False)],
    # Multi-GPU modes (DDP via lightning devices=auto). SkyPilot default GPU
    # image: the audio-science AMIs are not accessible from this account.
    "spot_east_h100": [_base(infra="aws/us-east-2", accelerators="H100:8", use_spot=True)],
    "spot_west_h100": [_base(infra="aws/us-west-2", accelerators="H100:8", use_spot=True)],
    "spot_h100": [_base(infra="aws", accelerators="H100:8", use_spot=True)],
    "spot_east_a100": [_base(infra="aws/us-east-2", accelerators="A100:8", use_spot=True)],
    "demand_a100": [_base(infra="aws/us-west-2", accelerators="A100:8", use_spot=False)],
}

# ---------- task body ----------

ENVS: dict[str, str] = {
    "AWS_REGION": "us-west-2",
    # s3 sources
    "DATASET_S3_PATH": "s3://audio-science-hub/datasets/splicegen_prod_v1_SAME_L_latents_noncommercial_license/",
    "SILENCE_PAD_EMBED_S3_PATH": "s3://dit-melodic-loops/clean_checkpoints_genai/SAME-L/silence_pad_embed.pt",
    # local destinations (nvme scratch on DLAMI)
    "DATASET_LOCAL_PATH": "/opt/dlami/nvme/hfds/splicegen_prod_v1_SAME_L_latents_noncommercial_license",
    "SILENCE_PAD_EMBED_LOCAL_PATH": "/opt/dlami/nvme/checkpoints/SAME-L/silence_pad_embed.pt",
    "SAVE_DIR": "/opt/dlami/nvme/outputs/sa3_sg_adapters",
}

SETUP: list[str] = [
    "set -e",
    # uv + project env (torch cu126 via pyproject index)
    "command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh",
    'export PATH="$HOME/.local/bin:$PATH"',
    "uv sync --extra lora --extra splicegen",
    # Ensure AWS CLI v2 (with CRT) — v1 lacks it
    "if ! aws --version 2>&1 | grep -q 'aws-cli/2'; then",
    '    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip',
    "    sudo apt-get update -qq && sudo apt-get install -y -qq unzip",
    "    unzip -q -o /tmp/awscli.zip -d /tmp",
    "    sudo /tmp/aws/install --update",
    "    rm -rf /tmp/aws /tmp/awscli.zip",
    "    hash -r",
    "fi",
    # CRT transfer tuning
    "aws configure set s3.preferred_transfer_client crt",
    "aws configure set s3.max_concurrent_requests   256",
    "aws configure set s3.multipart_chunksize        32MB",
    "aws configure set s3.multipart_threshold        32MB",
    "aws configure set s3.target_bandwidth           100Gb/s",
    # HGX nodes (p5 8xH100 NVSwitch) need NVIDIA Fabric Manager or CUDA init
    # fails with 'Error 802: system not yet initialized'. The SkyPilot default
    # image doesn't run it; install the version matching the driver and start it.
    # The FM version must match the driver version EXACTLY (e.g. 535.183.01),
    # otherwise the service refuses to start.
    "if [ \"$(nvidia-smi -L | grep -c H100)\" -ge 2 ] && ! systemctl is-active --quiet nvidia-fabricmanager; then",
    "    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
    "    DRIVER_BRANCH=${DRIVER_VER%%.*}",
    "    sudo apt-get update -qq",
    "    sudo apt-get install -y -qq --allow-downgrades nvidia-fabricmanager-${DRIVER_BRANCH}=${DRIVER_VER}-1 || {",
    "        echo 'Exact FM version not in apt; fetching .deb from the NVIDIA CUDA repo';",
    "        curl -fsSL -o /tmp/fm.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-${DRIVER_BRANCH}_${DRIVER_VER}-1_amd64.deb;",
    "        sudo dpkg -i /tmp/fm.deb; }",
    "    sudo systemctl enable nvidia-fabricmanager && sudo systemctl restart nvidia-fabricmanager",
    "    for i in $(seq 1 30); do systemctl is-active --quiet nvidia-fabricmanager && break; sleep 2; done",
    "    systemctl is-active --quiet nvidia-fabricmanager || { sudo journalctl -u nvidia-fabricmanager --no-pager | tail -20; exit 1; }",
    "fi",
    # Fail setup early (and visibly) if CUDA still can't initialize.
    ".venv/bin/python -c \"import torch; torch.cuda.init(); print('CUDA OK:', torch.cuda.device_count(), 'GPUs')\"",
    # Dataset + silence pad. /opt/dlami/nvme exists only on the audio-science
    # DLAMI; on the SkyPilot default image create it on the root disk.
    "sudo mkdir -p /opt/dlami/nvme && sudo chown -R $(whoami) /opt/dlami",
    "mkdir -p $(dirname $SILENCE_PAD_EMBED_LOCAL_PATH) $DATASET_LOCAL_PATH $SAVE_DIR",
    "aws s3 sync $DATASET_S3_PATH $DATASET_LOCAL_PATH --only-show-errors",
    "aws s3 cp $SILENCE_PAD_EMBED_S3_PATH $SILENCE_PAD_EMBED_LOCAL_PATH --only-show-errors",
]

RUN: list[str] = [
    "set -e",
    'export PATH="$HOME/.local/bin:$PATH"',
    ".venv/bin/python scripts/train_lora_splicegen.py"
    " --dataset_dir $DATASET_LOCAL_PATH"
    " --pad_latent $SILENCE_PAD_EMBED_LOCAL_PATH"
    " --rank $LORA_RANK"
    " --adapter_type $ADAPTER_TYPE"
    " --lr $LR"
    " --steps $STEPS"
    " --batch_size $BATCH_SIZE"
    " --project $WANDB_PROJECT"
    " --group $WANDB_GROUP"
    " --run_name $RUN_NAME"
    " --run_id $RUN_ID"
    " --save_dir $SAVE_DIR"
    " --s3_checkpoint_uri $S3_CHECKPOINT_URI"
    " --checkpoint_every $CHECKPOINT_EVERY"
    " --demo_every $DEMO_EVERY"
    " $EXTRA_ARGS",
]


def _require_env(name: str, hint: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        sys.exit(f"error: {name} is not set in the shell. {hint}")
    return val


def _wandb_api_key() -> str:
    """WANDB_API_KEY from the shell, falling back to ~/.netrc (written by `wandb login`)."""
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        return key
    try:
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth and auth[2]:
            print("[launch] using wandb API key from ~/.netrc", file=sys.stderr)
            return auth[2]
    except Exception:
        pass
    sys.exit("error: WANDB_API_KEY is not set and no api.wandb.ai entry in ~/.netrc. Run `wandb login`.")


def _hf_token() -> str:
    """HF token from the shell or the huggingface_hub token cache."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        token = get_token() or ""
        if token:
            print("[launch] using HF token from local huggingface_hub cache", file=sys.stderr)
            return token
    except Exception:
        pass
    print(
        "[launch] WARNING: no HF_TOKEN found; the VM will download "
        "stabilityai/stable-audio-3-medium-base anonymously (fails if the repo is gated).",
        file=sys.stderr,
    )
    return ""


def _git_head_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _apply_git_workdir_envs(task: sky.Task) -> None:
    """Resolve a git workdir into GIT_URL / GIT_* envs + secrets on the task.

    Newer SkyPilot exposes Task.update_envs_and_secrets_from_workdir(); older
    versions only do this inside the CLI. Replicate the CLI helper here so the
    on-VM git_clone.sh can actually clone.
    """
    if hasattr(task, "update_envs_and_secrets_from_workdir"):
        task.update_envs_and_secrets_from_workdir()
        return

    from sky.client.cli import git as git_utils_mod
    from sky.utils import git as git_utils

    url = task.workdir["url"]
    ref = task.workdir.get("ref", "")
    token = os.environ.get(git_utils.GIT_TOKEN_ENV_VAR)
    ssh_key_path = os.environ.get(git_utils.GIT_SSH_KEY_PATH_ENV_VAR)

    git_repo = git_utils_mod.GitRepo(url, ref, token, ssh_key_path)
    clone_info = git_repo.get_repo_clone_info()
    if clone_info is None:
        return
    task.envs[git_utils.GIT_URL_ENV_VAR] = clone_info.url
    if ref:
        ref_type = git_repo.get_ref_type()
        if ref_type == git_utils_mod.GitRefType.COMMIT:
            task.envs[git_utils.GIT_COMMIT_HASH_ENV_VAR] = ref
        elif ref_type == git_utils_mod.GitRefType.BRANCH:
            task.envs[git_utils.GIT_BRANCH_ENV_VAR] = ref
        elif ref_type == git_utils_mod.GitRefType.TAG:
            task.envs[git_utils.GIT_TAG_ENV_VAR] = ref
    if clone_info.token is not None:
        task.secrets[git_utils.GIT_TOKEN_ENV_VAR] = clone_info.token
    if clone_info.ssh_key is not None:
        task.secrets[git_utils.GIT_SSH_KEY_ENV_VAR] = clone_info.ssh_key


def _jobs_bucket(name: str) -> str:
    user = os.environ.get("USER") or getpass.getuser()
    return f"{JOBS_BUCKET_BASE}/{user}/{name}/"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="base job name; per-rank jobs are named <name>-r<rank>")
    p.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 64],
                   help="LoRA ranks to sweep (one managed job each)")
    p.add_argument("--adapter_type", default="dora-rows")
    p.add_argument("--full_finetune", action="store_true",
                   help="Full fine-tuning (no adapters): launches a single job named <name>-fullft")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=32, help="per-GPU batch size")
    p.add_argument("--mode", choices=list(RESOURCES_BY_MODE), default="spot_h100_1")
    p.add_argument("--project", default="sa3_sg_adapters")
    p.add_argument("--group", default=None, help="wandb group (default: <name>)")
    p.add_argument("--checkpoint_every", type=int, default=1000)
    p.add_argument("--demo_every", type=int, default=1000)
    p.add_argument("--extra_args", default="", help="extra CLI args appended to the train command")
    p.add_argument("--run_id", default=None,
                   help="wandb run id override (single-job launches only; default: job name). "
                        "Use when continuing a run whose id differs from its name.")
    p.add_argument("--git-url", dest="git_url",
                   default=os.environ.get("SA3_FORK_URL", ""),
                   help="git URL of a stable-audio-3 fork to clone on the VM. "
                        "If omitted, the local working tree is synced instead (default).")
    p.add_argument("--git-ref", dest="git_ref", default=None,
                   help="commit SHA / branch to clone on the VM (default: local HEAD). Only with --git-url.")
    p.add_argument("--detach", action="store_true", help="submit all jobs and exit")
    p.add_argument("--dry-run", action="store_true", help="print tasks without launching")
    args = p.parse_args()

    wandb_key = _wandb_api_key()
    hf_token = _hf_token()

    if args.git_url:
        # Reproducible mode: clone the fork at a pinned ref on the VM.
        if not (os.environ.get("GIT_TOKEN") or os.environ.get("GIT_SSH_KEY_PATH")):
            print(
                "[launch] note: neither GIT_TOKEN nor GIT_SSH_KEY_PATH is set; "
                "the on-VM clone will fail if the fork is private.",
                file=sys.stderr,
            )
        git_ref = args.git_ref or _git_head_commit()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True,
        ).stdout.strip()
        if args.git_ref is None and dirty:
            print(
                f"[launch] WARNING: local tree is dirty; uncommitted changes are NOT in ref {git_ref[:8]}.",
                file=sys.stderr,
            )
        workdir: str | dict = {"url": args.git_url, "ref": git_ref}
        src_desc = f"git@{git_ref[:8]}"
    else:
        # Default: sync the local working tree to the VM (no push required).
        # Note: runs are not reproducible from a commit hash in this mode.
        workdir = str(REPO_ROOT)
        src_desc = "local workdir"
        print(
            "[launch] syncing local working tree to the VMs (no git clone). "
            "Pass --git-url for reproducible clone-from-fork launches.",
            file=sys.stderr,
        )

    group = args.group or args.name
    request_ids = []

    if args.full_finetune:
        # Single job; --rank/--adapter_type are passed but ignored by the script.
        jobs = [(f"{args.name}-fullft", args.ranks[0])]
        extra_args = (args.extra_args + " --full_finetune").strip()
    else:
        jobs = [(f"{args.name}-r{rank}", rank) for rank in args.ranks]
        extra_args = args.extra_args

    if args.run_id and len(jobs) > 1:
        sys.exit("error: --run_id only makes sense for single-job launches")

    for job_name, rank in jobs:
        # Stable wandb id => resume across preemptions
        run_id = args.run_id or job_name.replace("_", "-")
        envs = {
            **ENVS,
            "LORA_RANK": str(rank),
            "ADAPTER_TYPE": args.adapter_type,
            "LR": str(args.lr),
            "STEPS": str(args.steps),
            "BATCH_SIZE": str(args.batch_size),
            "WANDB_PROJECT": args.project,
            "WANDB_GROUP": group,
            "RUN_NAME": job_name,
            "RUN_ID": run_id,
            "S3_CHECKPOINT_URI": f"{CHECKPOINTS_BUCKET_BASE}/{job_name}/",
            "CHECKPOINT_EVERY": str(args.checkpoint_every),
            "DEMO_EVERY": str(args.demo_every),
            "EXTRA_ARGS": extra_args,
        }

        task = sky.Task(
            name=job_name,
            setup="\n".join(SETUP),
            run="\n".join(RUN),
            workdir=workdir,
            num_nodes=1,
            envs=envs,
            secrets={"WANDB_API_KEY": wandb_key, **({"HF_TOKEN": hf_token} if hf_token else {})},
        )
        task.set_resources(RESOURCES_BY_MODE[args.mode])
        if isinstance(workdir, dict):
            try:
                _apply_git_workdir_envs(task)
            except Exception as e:
                sys.exit(f"error: failed to validate git workdir against remote: {e}")

        print(
            f"[launch] job={job_name} rank={rank} mode={args.mode} src={src_desc} "
            f"ckpts={envs['S3_CHECKPOINT_URI']}",
            file=sys.stderr,
        )
        if args.dry_run:
            continue

        overrides = {
            "aws": {"remote_identity": AWS_REMOTE_IDENTITY},
            "jobs": {"bucket": _jobs_bucket(job_name)},
        }
        with skypilot_config.override_skypilot_config(overrides):
            request_id = sky.jobs.launch(task)
        request_ids.append((job_name, request_id))

    if args.dry_run:
        return 0

    for job_name, request_id in request_ids:
        result = sky.stream_and_get(request_id)
        # Depending on the sky version, jobs.launch returns (job_id, handle)
        # or ([job_id, ...], handle).
        managed_job_id = result[0] if result else None
        if isinstance(managed_job_id, (list, tuple)):
            managed_job_id = managed_job_id[0] if managed_job_id else None
        print(f"[launch] submitted {job_name}: managed job id {managed_job_id}", file=sys.stderr)

    print(
        "[launch] all jobs submitted. Monitor with `sky jobs queue` / "
        "`sky jobs logs <id>` and the wandb project page.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
