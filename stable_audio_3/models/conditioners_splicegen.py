"""SpliceGen 3 conditioners ported from audio-science-genai.

- ``ArrayConditioner`` / ``FeatureConditioner``: consume pre-computed feature
  arrays from the dataset (CL-SiSo ``embedding`` 512-d vectors and ``chroma``
  12-d chromagrams) and project them to the model conditioning dimension.
- ``CycleConditioner``: an analytic multi-frequency cyclic encoder over musical
  time driven by ``bpm`` / ``seconds_start`` / ``seconds_total``.

Ported faithfully from ``audio_science_genai/models/conditioners.py`` and
``audio_science_genai/models/transforms.py`` (only the features exercised by
the SpliceGen v3 configs are kept: scale/shift, filter threshold, pre/post
normalization, mean/first reduction, expansion, and top-K dropout).
"""

import math
import typing as tp

import torch
from einops import rearrange
from torch import nn

from .conditioners import Conditioner


class TopKDropout(nn.Module):
    """Top-K dropout over the channel dimension of a (B, T, C) tensor.

    When ``topk`` is a float it is interpreted as the fraction of cumulative
    magnitude to preserve (cumsum mode); when an int, as the number of top
    values to keep. During training the amount kept is randomly varied
    according to ``width``/``p``; during eval the maximum amount is kept.

    Port of ``audio_science_genai.models.transforms.TopKDropout``.
    """

    def __init__(
        self,
        width: float = 1.0,
        p: float = 1.0,
        topk: tp.Union[int, float, None] = None,
        time_invariant: bool = False,
    ):
        super().__init__()
        self.width = width
        self.p = p
        self.topk = topk
        self.time_invariant = time_invariant

    def _bernoulli_dropout(self, x: torch.Tensor, C_max: int) -> torch.Tensor:
        B = x.shape[0]
        bernoulli_mask = torch.bernoulli(torch.full((B,), self.p, device=x.device)).bool()
        low = C_max - max(1, int(self.width * C_max))
        dropout = torch.where(
            bernoulli_mask,
            torch.randint(low, C_max, (B,), device=x.device),
            torch.full((B,), C_max - 1, device=x.device),
        ).view(B, 1, 1)
        return dropout

    def forward(self, x: torch.Tensor, C_max: tp.Union[int, float, None] = None) -> torch.Tensor:
        B, T, C = x.shape
        topk = self.topk if C_max is None else C_max
        C_max = C if topk is None else topk

        x_topk = x.mean(1, keepdims=True).repeat(1, T, 1) if self.time_invariant else x.clone()
        do_cumsum = isinstance(C_max, float)
        if do_cumsum:
            x_topk = x_topk.abs()
            T_max = C_max
            C_max = C
        threshold, _ = torch.topk(x_topk, C_max, dim=-1)

        if do_cumsum:
            cumsum = torch.cumsum(threshold, dim=-1)
            cumsum_pct = cumsum / cumsum[..., -1:]

            T_min = (1.0 - self.width) * T_max
            C_min = torch.sum(cumsum_pct <= T_min, dim=-1, keepdims=True)
            C_min = torch.clamp(C_min, max=C - 1)

            C_max = torch.sum(cumsum_pct <= T_max, dim=-1, keepdims=True)
            C_max = torch.clamp(C_max, max=C - 1)

        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1)
        time_indices = torch.arange(T, device=x.device).view(1, T, 1)

        if do_cumsum:
            if self.training:
                alpha = torch.rand(B, device=x.device).view(B, 1, 1)
                dropout = ((1.0 - alpha) * C_min + alpha * C_max).long()

                bernoulli_mask = torch.bernoulli(torch.full((B,), self.p, device=x.device)).view(B, 1, 1).long()
                dropout = (1 - bernoulli_mask) * (C - 1) * torch.ones_like(dropout) + bernoulli_mask * dropout
            else:
                dropout = C_max.clone()

            threshold = threshold[batch_indices.expand(B, T, 1), time_indices.expand(B, T, 1), dropout]
        else:
            if self.training:
                dropout = self._bernoulli_dropout(x, C_max)
                threshold = threshold[
                    batch_indices.expand(B, T, 1), time_indices.expand(B, T, 1), dropout.expand(B, T, 1)
                ]
            else:
                threshold = threshold[..., -1:]

        mask = x_topk >= threshold
        return x * mask


class Reducer(nn.Module):
    """Reduce a (B, T, C) feature along the time dimension to (B, 1, C)."""

    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ["mean", "first"], f"Unsupported reduction mode: {mode}"
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return torch.mean(x, dim=1, keepdim=True)
        return x[:, 0:1, :]


class Expander(nn.Module):
    """Expand a (B, T, C) tensor along time to T0 frames via zero-order hold."""

    def forward(self, x: torch.Tensor, T0: int = None) -> torch.Tensor:
        B, T, C = x.shape
        if T0 is None or T0 <= T:
            return x
        upsampling_factor = T0 // T + 1
        return torch.repeat_interleave(x, upsampling_factor, dim=1)[:, :T0]


class ArrayConditioner(Conditioner):
    """Conditioner for pre-computed array-like inputs (port of SpliceGen's ArrayConditioner).

    Args:
        dim: Channel dimension of the input array.
        output_dim: Output (conditioning) dimension.
        project_out: Force a projection even when dim == output_dim.
        channels_last: If True, outputs (B, T, C); if False, (B, C, T).
        filter_threshold: Zero out values below this threshold.
        shift: Shift subtracted before projection.
        scale: Scale applied before projection.
        prenorm: p-normalization order applied before aggregation (None = off).
        postnorm: p-normalization order applied after aggregation (None = off).
        reduce: Optional time reduction ("mean" or "first").
        expand: If True, expand output back to the input number of time frames.
        topk_dropout_factor: Top-K amount (int count or float cumulative fraction).
        topk_dropout_width: Randomization width for top-K dropout.
        topk_dropout_p: Probability of randomizing the top-K amount.
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        project_out: bool = False,
        channels_last: bool = True,
        filter_threshold: float = 0.0,
        shift: float = 0.0,
        scale: float = 1.0,
        prenorm: float = None,
        postnorm: float = None,
        reduce: str = None,
        expand: bool = True,
        topk_dropout_factor: tp.Union[int, float, None] = None,
        topk_dropout_width: float = 0.0,
        topk_dropout_p: float = 0.0,
    ):
        super().__init__(dim, output_dim, project_out=project_out)
        self.channels_last = channels_last
        self.filter_threshold = filter_threshold
        self.shift = shift
        self.scale = scale
        self.prenorm = prenorm
        self.postnorm = postnorm

        self.reduce = None if reduce is None else Reducer(reduce)
        self.expand = Expander() if expand else None

        self.topk_dropout = None
        if topk_dropout_factor is not None or topk_dropout_p > 0:
            self.topk_dropout = TopKDropout(
                width=topk_dropout_width,
                p=topk_dropout_p,
                topk=topk_dropout_factor,
            )

    def forward(
        self, x: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Any = "cuda"
    ) -> tp.Any:
        """Project input array(s) to the output dimension.

        Accepts a (B, T, C) tensor or a list of per-sample tensors each carrying
        a leading batch dim of 1 (concatenated along the batch dimension).
        Returns ``[embeddings, masks]`` with embeddings (B, T, output_dim) when
        channels_last else (B, output_dim, T), and masks (B, T).
        """
        self.proj_out.to(device)

        if isinstance(x, tp.List):
            x = torch.cat([xi.to(device) for xi in x], dim=0)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # B, C -> B, 1, C

        x = x.to(device=device, dtype=torch.float32)

        B, T0, C = x.shape

        # Apply reduction (if enabled)
        x = self.reduce(x) if self.reduce else x

        # Apply top-K dropout (if enabled)
        x = self.topk_dropout(x) if self.topk_dropout else x

        if self.filter_threshold > 0:
            x = (x >= self.filter_threshold).long() * x

        # Normalize features before any aggregation
        if self.prenorm is not None:
            x = torch.nn.functional.normalize(x, p=self.prenorm, dim=-1)

        # Normalize features after any aggregation
        if self.postnorm is not None:
            if self.postnorm > 0:
                x = torch.nn.functional.normalize(x, p=self.postnorm, dim=-1)
            else:
                # Normalize such that the time-averaged signal would be p-normalized
                x_mean = x.mean(1, keepdims=True)
                x_lp = x_mean.norm(p=-self.postnorm, dim=-1, keepdim=True)
                denom = x_lp.clamp_min(1e-12).expand_as(x)
                x = x / denom

        x = (x - self.shift) * self.scale

        if not isinstance(self.proj_out, nn.Identity):
            x = x.to(next(self.proj_out.parameters()).dtype)
        embeddings = self.proj_out(x)
        B, T, C = embeddings.shape

        # Apply expansion (if enabled)
        embeddings = self.expand(embeddings, T0=T0) if self.expand else embeddings

        if not self.channels_last:
            embeddings = embeddings.permute(0, 2, 1)  # B, T, C -> B, C, T

        masks = torch.ones(B, T).to(device)

        return [embeddings, masks]


class FeatureConditioner(ArrayConditioner):
    """ArrayConditioner with a feature extractor in front.

    In this port only pre-encoded features are supported, so the extractor is
    an identity: the dataset supplies the pre-computed feature arrays directly.
    """

    def __init__(self, feature_extractor: nn.Module = None, **array_config):
        super().__init__(**array_config)
        self._feat_extractor = feature_extractor if feature_extractor is not None else nn.Identity()

    def forward(self, x, device="cuda"):
        feats = self._feat_extractor(x)
        return super().forward(feats, device=device)


class CycleConditioner(Conditioner):
    """Multi-frequency cyclic conditioner: a positional-embedding-style encoder over musical time.

    Anchors on a base frequency of 1 cycle per beat and emits one (sin, cos)
    pair for each entry in ``ratios``: for ``n`` ratios the raw output has
    ``2*n`` channels ordered ``[sin(r0), cos(r0), sin(r1), cos(r1), ...]``,
    optionally projected to ``output_dim``.

    Port of ``audio_science_genai.models.conditioners.CycleConditioner``.

    Attributes:
        min_fold_bpm: If set, fold each (post-clip) BPM into
            ``[min_fold_bpm, 2 * min_fold_bpm)`` by integer powers of 2.
        none_imputation: Value substituted for ``None`` BPM entries. Applied
            after clipping and folding, so the imputed value is used verbatim.
    """

    def __init__(
        self,
        bpm_key: str,
        phase_key: str,
        len_key: str,
        output_dim: int,
        ratios: tp.Optional[tp.List[float]] = None,
        project_out: bool = False,
        min_bpm: float = 0.0,
        max_bpm: float = 1.0,
        n_frames: int = 345,
        sr: float = 21.533203125,
        channels_last: bool = True,
        min_fold_bpm: tp.Optional[float] = None,
        none_imputation: float = 0.0,
    ):
        if min_fold_bpm is not None and min_fold_bpm <= 0:
            raise ValueError("CycleConditioner: `min_fold_bpm` must be positive.")
        if ratios is None:
            ratios = [0.25]
        ratios = [float(r) for r in ratios]

        super().__init__(2 * len(ratios), output_dim, project_out)
        self.bpm_key = bpm_key
        self.phase_key = phase_key
        self.len_key = len_key
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.n_frames = n_frames
        self.sr = sr
        self.ratios = ratios
        self.channels_last = channels_last
        self.min_fold_bpm = min_fold_bpm
        self.none_imputation = float(none_imputation)

    def forward(self, conds: tp.List[dict], device=None) -> tp.Any:
        """Generate multi-frequency cyclic embeddings from BPM, phase, and length metadata.

        Args:
            conds: List of dicts each containing ``bpm_key``, ``phase_key`` and ``len_key``.
            device: Device to place tensors on.

        Returns:
            ``[embeddings, mask]`` where embeddings is (B, T, C) or (B, C, T)
            depending on ``channels_last`` and mask is (B, 1).
        """
        self.proj_out.to(device)

        raw_bpms = [c[self.bpm_key] for c in conds]
        none_mask = torch.tensor([v is None for v in raw_bpms], device=device)
        bpms = torch.tensor([float(v) if v is not None else 1.0 for v in raw_bpms]).to(device)
        phases = torch.tensor([math.floor(float(c[self.phase_key]) * self.sr) for c in conds]).to(device)
        lens = torch.tensor([math.ceil(float(c[self.len_key]) * self.sr) for c in conds]).to(device)

        bpms = bpms.clamp(self.min_bpm, self.max_bpm)

        if self.min_fold_bpm is not None:
            # Fold every BPM into [min_fold_bpm, 2 * min_fold_bpm) via integer powers of 2.
            safe = bpms.clamp_min(torch.finfo(bpms.dtype).tiny)
            k = torch.floor(torch.log2(safe / self.min_fold_bpm))
            bpms = bpms * torch.pow(torch.tensor(2.0, dtype=bpms.dtype, device=bpms.device), -k)

        # Imputation runs after clipping/folding so the imputed value is used verbatim.
        if none_mask.any():
            bpms = torch.where(none_mask, torch.full_like(bpms, self.none_imputation), bpms)

        # Base cycle = 1 cycle per beat; per-ratio frequencies scale this base.
        sec_per_beat = 60.0 / bpms
        sam_per_beat = sec_per_beat * self.sr
        base_freq_per_sample = 2 * math.pi / sam_per_beat  # [B]
        ratios_t = torch.tensor(self.ratios, device=device, dtype=base_freq_per_sample.dtype)  # [n]
        freqs = base_freq_per_sample.unsqueeze(1) * ratios_t.unsqueeze(0)  # [B, n]

        lens = torch.clamp(lens, torch.zeros_like(lens), phases + self.n_frames)
        base = torch.arange(self.n_frames, device=device)
        ranges = base.unsqueeze(0) + phases.unsqueeze(1)  # [B, T]
        mask = base.unsqueeze(0) < (lens - phases).unsqueeze(1)  # [B, T]

        args = freqs.unsqueeze(-1) * ranges.unsqueeze(1)  # [B, n, T]
        # Interleave sin/cos within each ratio: channel order is [sin(r0), cos(r0), sin(r1), cos(r1), ...]
        cycles = rearrange(
            torch.stack([torch.sin(args), torch.cos(args)], dim=-1),
            "b n t c -> b t (n c)",
        )

        cycles = cycles * mask.unsqueeze(-1).to(cycles.dtype)

        if not isinstance(self.proj_out, nn.Identity):
            cycles = cycles.to(device, dtype=next(self.proj_out.parameters()).dtype)
        cycles = self.proj_out(cycles)

        if not self.channels_last:
            cycles = rearrange(cycles, "b t c -> b c t")

        return [cycles, torch.ones(cycles.shape[0], 1).to(device)]
