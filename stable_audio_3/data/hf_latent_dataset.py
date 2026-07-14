"""HuggingFace-datasets-backed pre-encoded latent dataset.

Port of ``audio_science_genai/data/simplified_pre_encoded_dataset.py`` (and the
``PadCrop_Normalized_T_PC`` transform from ``audio_science_genai/data/utils.py``).
"""

import math
import random
import typing as tp

import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset


class LatentPadCrop(nn.Module):
    """Pad/crop a (C, T) latent to ``n_frames``, tracking crop timing.

    Port of ``PadCrop_Normalized_T_PC`` with ``do_vae_sample=False``: short
    inputs are padded with a fixed "latent silence" vector (the latent the
    autoencoder assigns to silence) instead of zeros.
    """

    def __init__(
        self,
        n_frames: int,
        sample_rate: float,
        randomize: tp.Union[bool, float] = True,
        pad_latent: tp.Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.randomize = float(randomize)
        self.pad_latent = pad_latent

    def __call__(self, source: torch.Tensor, offset=None, pad_zeros=True):
        n_channels, n_samples = source.shape

        upper_bound = max(0, n_samples - self.n_frames)

        if offset is None:
            offset = 0
            if torch.rand(1).item() < self.randomize and n_samples > self.n_frames:
                offset = random.randint(0, upper_bound)

        t_start = offset / (upper_bound + self.n_frames)
        t_end = (offset + self.n_frames) / (upper_bound + self.n_frames)

        if pad_zeros:
            chunk = source.new_zeros([n_channels, self.n_frames])
        else:
            assert self.pad_latent is not None, (
                "Must provide a latent silence vector to pad with (pad_latent)"
            )
            chunk = self.pad_latent.to(source.dtype).unsqueeze(-1).repeat(1, self.n_frames).clone()

        chunk[:, : min(n_samples, self.n_frames)] = source[:, offset : offset + self.n_frames]

        seconds_start = offset / self.sample_rate
        seconds_total = n_samples / self.sample_rate

        padding_mask = torch.zeros([self.n_frames])
        padding_mask[: min(n_samples, self.n_frames)] = 1

        return (chunk, t_start, t_end, seconds_start, seconds_total, padding_mask, offset)


def rearrange_from_map(ex: tp.Dict[str, tp.Any], rearrange_map: tp.Dict[str, str]):
    for k, pattern in rearrange_map.items():
        if k in ex:
            ex[k] = rearrange(ex[k], pattern)
        else:
            raise RuntimeError(f"Key {k} not found in example for rearranging")
    return ex


class HFLatentDataset(Dataset):
    """Map-style dataset over a HuggingFace dataset of pre-encoded latents.

    Args:
        dataset: HuggingFace dataset (already loaded from disk).
        reals_column: Column with the (C, T) latent used as diffusion target.
        temporal_columns: Frame-aligned feature columns cropped alongside the
            latent (e.g. chroma). Expected (C, T) after ``rearrange_in_map``.
        other_columns: Non-temporal columns (e.g. embedding, bpm).
        pad_to_n_frames: Target latent length.
        sample_rate: Latent frame rate in Hz.
        randomize: Random-crop probability.
        pad_latent: (C,) latent silence vector for padding, or None for zeros.
        limit_seconds_total: Cap seconds_total at seconds_start + window length.
        rearrange_in_map / rearrange_out_map: einops patterns applied to columns
            before / after processing (e.g. chroma "t c -> c t" / "c t -> t c").
        min_bpm / max_bpm: BPM values outside this range are replaced by None.
    """

    def __init__(
        self,
        dataset,
        reals_column: str = "latent",
        temporal_columns: tp.Optional[tp.List[str]] = None,
        other_columns: tp.Optional[tp.List[str]] = None,
        pad_to_n_frames: int = 173,
        sample_rate: float = 44100 / 4096,
        randomize: bool = True,
        pad_latent: tp.Optional[torch.Tensor] = None,
        limit_seconds_total: bool = True,
        rearrange_in_map: tp.Optional[tp.Dict[str, str]] = None,
        rearrange_out_map: tp.Optional[tp.Dict[str, str]] = None,
        min_bpm: tp.Optional[float] = None,
        max_bpm: tp.Optional[float] = None,
        skip_frame_diff_check: bool = False,
    ):
        self.ds = dataset
        self.reals_column = reals_column
        self.temporal_columns = temporal_columns or []
        self.other_columns = other_columns or []
        self.pad_to_n_frames = pad_to_n_frames
        self.sample_rate = sample_rate
        self.pad_zeros = pad_latent is None
        self.limit_seconds_total = limit_seconds_total
        self.rearrange_in_map = rearrange_in_map
        self.rearrange_out_map = rearrange_out_map
        if min_bpm is not None and max_bpm is not None and min_bpm > max_bpm:
            raise ValueError(f"min_bpm ({min_bpm}) must be <= max_bpm ({max_bpm}).")
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.skip_frame_diff_check = skip_frame_diff_check

        self.pad_crop = LatentPadCrop(
            pad_to_n_frames, sample_rate, randomize=randomize, pad_latent=pad_latent
        )
        self.all_columns = [self.reals_column] + self.temporal_columns + self.other_columns
        self.ds.set_format(type="torch", columns=self.all_columns)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        info = {}
        ex = self.ds[idx]

        if self.rearrange_in_map is not None:
            ex = rearrange_from_map(ex, self.rearrange_in_map)

        # Process latent
        latent = ex[self.reals_column]  # [C, T]
        latent_chunk, t_start, t_end, seconds_start, seconds_end, padding_mask, offset = self.pad_crop(
            latent, pad_zeros=self.pad_zeros
        )

        if self.limit_seconds_total:
            seconds_end = min(seconds_start + self.pad_to_n_frames / self.sample_rate, seconds_end)

        info["t_start"] = t_start
        info["t_end"] = t_end
        info["seconds_start"] = seconds_start
        info["seconds_total"] = seconds_end
        # (1, T) so that training code can index md["padding_mask"][0] -> (T,)
        info["padding_mask"] = padding_mask.unsqueeze(0)

        # Process temporal columns (frame-aligned with the latent)
        for col in self.temporal_columns:
            data = ex[col]  # [C, T] at this point
            frame_diff = latent.shape[1] - data.shape[1]
            if not abs(frame_diff) < 2 and not self.skip_frame_diff_check:
                raise RuntimeError(
                    f"Frame mismatch between latent and {col}: {latent.shape[1]} vs {data.shape[1]}"
                )
            if frame_diff > 0:
                data = torch.nn.functional.pad(data, (0, frame_diff))
            elif frame_diff < 0:
                data = data[:, 0 : latent.shape[-1]]
            data, _, _, _, _, _, _ = self.pad_crop(data, offset=offset, pad_zeros=True)
            info[col] = data.float()

        # Process other feature columns
        for col in self.other_columns:
            data = ex[col]
            if isinstance(data, torch.Tensor) and data.ndim == 1:
                data = data.unsqueeze(0)
            if col == "bpm":
                if data is None or (isinstance(data, torch.Tensor) and torch.isnan(data).any()):
                    data = None
                else:
                    bpm_int = int(data)
                    if bpm_int <= 0:
                        data = None
                    elif self.min_bpm is not None and bpm_int < self.min_bpm:
                        data = None
                    elif self.max_bpm is not None and bpm_int > self.max_bpm:
                        data = None
                    else:
                        data = bpm_int
            info[col] = data

        if self.rearrange_out_map is not None:
            info = rearrange_from_map(info, self.rearrange_out_map)

        # Give conditioning tensors a leading batch dim of 1 (ArrayConditioner
        # concatenates per-sample tensors along dim 0). padding_mask stays (1, T).
        for k, v in info.items():
            if k == "padding_mask":
                continue
            if isinstance(v, torch.Tensor):
                info[k] = v.view(*(1,) * max(0, 3 - v.ndim), *v.shape)

        return latent_chunk, info


def latent_collation_fn(batch):
    """Collate (latent, info) pairs into (stacked latents, list of info dicts)."""
    latents = []
    infos = []
    for latent, info in batch:
        latents.append(latent)
        infos.append(info)
    return torch.stack(latents), infos


def get_weighted_sampler(
    ds,
    sample_on_duration: bool = False,
    duration_column: str = "duration",
    sample_on_proba: bool = False,
    proba_column: str = "proba",
):
    """WeightedRandomSampler over log-duration and/or proba columns (SpliceGen recipe)."""
    if not (sample_on_duration or sample_on_proba):
        return None

    sample_proba = torch.ones(len(ds))
    if sample_on_duration:
        durations = torch.as_tensor(ds[duration_column][:], dtype=torch.float32)
        print(f"Duration sampling: mean={durations.mean().item():.2f}s std={durations.std().item():.2f}s")
        sample_proba *= torch.log(durations + 0.00001)
    if sample_on_proba:
        probas = torch.as_tensor(ds[proba_column][:], dtype=torch.float32)
        print(f"Proba sampling: mean={probas.mean().item():.4f} std={probas.std().item():.4f}")
        sample_proba *= probas
        sample_proba += 0.1
    sample_proba /= sample_proba.sum()

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_proba, num_samples=len(sample_proba), replacement=True
    )


def create_hf_latent_dataloader(
    dataset_path: str,
    batch_size: int,
    reals_column: str = "latent",
    temporal_columns: tp.Optional[tp.List[str]] = ("chroma",),
    other_columns: tp.Optional[tp.List[str]] = ("embedding", "bpm"),
    num_workers: int = 8,
    random_crop: bool = True,
    sample_on_duration: bool = True,
    sample_on_proba: bool = True,
    duration_column: str = "duration",
    proba_column: str = "proba",
    pad_to_n_frames: int = 173,
    sample_rate: float = 44100 / 4096,
    pad_latent_path: tp.Optional[str] = None,
    limit_seconds_total: bool = True,
    rearrange_in_map: tp.Optional[tp.Dict[str, str]] = None,
    rearrange_out_map: tp.Optional[tp.Dict[str, str]] = None,
    min_bpm: tp.Optional[float] = 20,
    max_bpm: tp.Optional[float] = 250,
    skip_frame_diff_check: bool = True,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    shuffle: tp.Optional[bool] = None,
    drop_last: bool = True,
    seed: tp.Optional[int] = None,
):
    """Build a DataLoader over a saved-to-disk HuggingFace latent dataset.

    Defaults match the SpliceGen prod v1 SAME-L dataset config
    (``splicegen_prod_v1_vae_SAME_L.json`` in audio-science-genai).
    """
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)

    sampler = get_weighted_sampler(
        ds,
        sample_on_duration=sample_on_duration,
        duration_column=duration_column,
        sample_on_proba=sample_on_proba,
        proba_column=proba_column,
    )

    pad_latent = None
    if pad_latent_path is not None:
        pad_latent = torch.load(pad_latent_path, map_location="cpu", weights_only=True)
        pad_latent = pad_latent.squeeze()
        # SAME-L silence_pad_embed.pt stores the latent directly (do_vae_sample=False
        # in the SpliceGen SAME-L config). A (2*C,) tensor would be a (mean, scale)
        # VAE parameterization, which SAME-L does not use.
        assert pad_latent.ndim == 1, f"Expected 1D pad latent, got shape {tuple(pad_latent.shape)}"

    pytorch_dataset = HFLatentDataset(
        dataset=ds,
        reals_column=reals_column,
        temporal_columns=list(temporal_columns) if temporal_columns else [],
        other_columns=list(other_columns) if other_columns else [],
        pad_to_n_frames=pad_to_n_frames,
        sample_rate=sample_rate,
        randomize=random_crop,
        pad_latent=pad_latent,
        limit_seconds_total=limit_seconds_total,
        rearrange_in_map=rearrange_in_map if rearrange_in_map is not None else {"chroma": "t c -> c t"},
        rearrange_out_map=rearrange_out_map if rearrange_out_map is not None else {"chroma": "c t -> t c"},
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        skip_frame_diff_check=skip_frame_diff_check,
    )

    shuffle = shuffle if shuffle is not None else (sampler is None)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=latent_collation_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=(lambda worker_id: torch.manual_seed(seed + worker_id)) if seed is not None else None,
    )

    return loader
