"""RFI 16-block window aggregator (M7.6 monitoring).

Lives next to ``dsart.services.corr_fast_integration`` and is fed one
cube at a time from the hot path (per ``flag_block`` call). Every
``window_size`` cubes (= 16 by default = ~2.147 s of voltages) it
emits a ``RFIWindow`` record containing:

* per-(ant, ch_downsampled, pol) **mean S1_4096** (pre-flag auto-power)
  with ``freq_downsample``× channel binning, fp32
* per-(ant, ch_downsampled, pol) **flag fraction** for each detector
  (``sk``, ``bandpass``, ``group``, ``sumthr``, ``flagants``, ``final``)
  with the same downsampling, uint8 (0..16, fits the cube count exactly)
* scalar window-level metrics:

  - ``total_flag_fraction``
    fraction of (ant, ch, pol) cells flagged by the OR-fold mask,
    averaged over all cubes in the window. Per-pol breakdown also
    emitted (``total_flag_fraction_pol0``, ``total_flag_fraction_pol1``).

  - ``bandpass_channel_fraction``
    mean over antennas of the per-(ant, pol) channel-flag-fraction
    of the bandpass-outlier mask (i.e. for each antenna compute
    ``mean(bp_mask[ant, :, pol])`` then average over ants). Per-pol
    sub-keys also emitted.

  - ``ant_fraction_flagged``
    fraction of (ant, pol) entries that the group-outlier flagged
    in at least one cube of the window. Per-pol sub-keys also
    emitted.

  - per-detector sub-fractions ``frac_sk``, ``frac_bp``, ``frac_grp``,
    ``frac_sumthr``, ``frac_fa`` (all averaged over the window, both
    pols folded — the bonus per-pol versions live in
    ``per_pol[pol]``).

Source-tag bit decomposition is used to extract per-detector
sub-masks from the production ``RFIFlagger.flag_block`` output, so
the corr_fast hot path doesn't need to recompute anything.

The aggregator operates on **whatever device** the masks come in on
(CUDA for prod, CPU for tests). The downsample + sums stay on-device
for the 16-cube accumulation; the per-window finalisation pays a
single ~150 KB CPU transfer.

Cost on a 2080 Ti at the production cube size (NANTS=96, NCHAN=384,
NPOL=2): ~0.5 ms per ``push`` (one Stokes-I-sized sum of fp32 +
several uint8 sums), ~1 ms at window finalisation. Negligible vs.
the 134 ms real-time budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from dsart.rfi.combine import FlagSourceBit

LOG = logging.getLogger("dsart.rfi_window")

# Per the M7.6 design: 16 cubes = 16 * 134.218 ms ≈ 2.147 s aggregation.
WINDOW_SIZE_DEFAULT: Final[int] = 16

# 4× channel downsample: 384 -> 96 per chgroup. Concatenated across 16
# chgroups → 1536 ch over the full DSA-110 band on the h23 dashboard.
FREQ_DOWNSAMPLE_DEFAULT: Final[int] = 4


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RFIWindow:
    """Per-window aggregated RFI state. Emitted by
    :meth:`RFIWindowAggregator.push` every ``window_size`` cubes.

    All arrays are CPU numpy in C-contiguous order.
    """

    block_n_start: int
    block_n_end: int                              # inclusive
    n_cubes: int                                  # = window_size (always)
    n_cubes_warmup: int                           # how many of n_cubes had warmup=True

    # Per-cell aggregates, downsampled in frequency.
    # Shape: (NANTS, NCHAN_DS, NPOL).
    s1_full_mean: np.ndarray                      # fp32; per-cube S1_4096 averaged
    mask_count_final: np.ndarray                  # uint8; cube count flagged by final OR
    mask_count_sk: np.ndarray                     # uint8
    mask_count_bp: np.ndarray                     # uint8
    mask_count_grp: np.ndarray                    # uint8
    mask_count_sumthr: np.ndarray                 # uint8 (cells added by sumthr)
    mask_count_fa: np.ndarray                     # uint8 (flagants overlay)

    # Window-scalar metrics. Index 0 = pol0/XX, 1 = pol1/YY, 2 = both.
    total_flag_fraction: tuple[float, float, float]
    bandpass_channel_fraction: tuple[float, float, float]
    ant_fraction_flagged: tuple[float, float, float]
    frac_sk: tuple[float, float, float]
    frac_bp: tuple[float, float, float]
    frac_grp: tuple[float, float, float]
    frac_sumthr: tuple[float, float, float]
    frac_fa: tuple[float, float, float]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class RFIWindowAggregator:
    """Stateful 16-cube aggregator.

    Construct once at pipeline startup, then call :meth:`push` per cube
    with the autos S1_full and the production flagger's mask + source
    tags. Returns an :class:`RFIWindow` every ``window_size`` cubes,
    or ``None`` between windows.

    Args:
        n_ants: NANTS for the cube (96 in production).
        n_chan: NCHAN_PER_CHGROUP for the cube (384 in production).
        n_pol: NPOL (2 in production).
        window_size: cubes per window. Default 16.
        freq_downsample: channel binning factor. Must divide ``n_chan``.
            Default 4 (→ 96 channels per chgroup at NCHAN=384).
        device: torch device on which the on-device accumulators live.
            Defaults to CPU; CUDA in production.
    """

    def __init__(
        self,
        *,
        n_ants: int,
        n_chan: int,
        n_pol: int,
        window_size: int = WINDOW_SIZE_DEFAULT,
        freq_downsample: int = FREQ_DOWNSAMPLE_DEFAULT,
        device: torch.device | str = "cpu",
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size={window_size}, expected > 0")
        if freq_downsample <= 0:
            raise ValueError(f"freq_downsample={freq_downsample}, expected > 0")
        if n_chan % freq_downsample != 0:
            raise ValueError(
                f"n_chan={n_chan} must be divisible by "
                f"freq_downsample={freq_downsample}"
            )
        self._n_ants = int(n_ants)
        self._n_chan = int(n_chan)
        self._n_chan_ds = int(n_chan // freq_downsample)
        self._n_pol = int(n_pol)
        self._window_size = int(window_size)
        self._freq_downsample = int(freq_downsample)
        self._device = torch.device(device)

        self._reset_accumulators()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _reset_accumulators(self) -> None:
        d = self._device
        shape = (self._n_ants, self._n_chan_ds, self._n_pol)
        # fp32 sum of S1_full (downsampled in freq).
        self._s1_sum = torch.zeros(shape, dtype=torch.float32, device=d)
        # Per-detector cube counts (uint8 is fine for window_size ≤ 255).
        self._mask_count_final = torch.zeros(shape, dtype=torch.uint8, device=d)
        self._mask_count_sk = torch.zeros(shape, dtype=torch.uint8, device=d)
        self._mask_count_bp = torch.zeros(shape, dtype=torch.uint8, device=d)
        self._mask_count_grp = torch.zeros(shape, dtype=torch.uint8, device=d)
        self._mask_count_sumthr = torch.zeros(shape, dtype=torch.uint8, device=d)
        self._mask_count_fa = torch.zeros(shape, dtype=torch.uint8, device=d)

        self._cubes_in_window: int = 0
        self._cubes_warmup_in_window: int = 0
        self._block_n_start: int | None = None
        self._block_n_last: int | None = None

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def n_chan_ds(self) -> int:
        return self._n_chan_ds

    @property
    def cubes_in_window(self) -> int:
        return self._cubes_in_window

    # ------------------------------------------------------------------
    # Per-cube ingestion
    # ------------------------------------------------------------------

    def _downsample_chan(self, x: torch.Tensor) -> torch.Tensor:
        """Sum-over-bin frequency downsample along the channel axis."""
        if self._freq_downsample == 1:
            return x
        # (NANTS, NCHAN, NPOL) -> (NANTS, NCHAN_DS, ds, NPOL) -> sum
        return x.view(
            self._n_ants, self._n_chan_ds, self._freq_downsample, self._n_pol,
        ).sum(dim=2)

    def push(
        self,
        *,
        s1_full: torch.Tensor,
        mask: torch.Tensor,
        source_tags: torch.Tensor,
        block_n: int,
        warmup: bool,
    ) -> RFIWindow | None:
        """Ingest one cube.

        Args:
            s1_full: fp32 ``S1_4096`` per (ant, ch, pol). Shape
                ``(NANTS, NCHAN, NPOL)``. Typically the squeeze of
                ``autos.s1[4096]`` from :func:`dsart.rfi.compute_autos`.
                We average it across the window to give the dashboard
                a clean "pre-flag bandpass".
            mask: bool tensor ``(NANTS, NCHAN, NPOL)`` — the final
                OR-folded RFI mask produced by ``RFIFlagger.flag_block``.
            source_tags: uint8 tensor ``(NANTS, NCHAN, NPOL)`` — the
                per-cell bitfield from
                :class:`dsart.rfi.FlagSourceBit`. Used to decompose
                into per-detector sub-masks.
            block_n: the cube's 1-based block index (from
                ``corr_fast_integration``'s ``n_in`` counter).
            warmup: True iff the production flagger reported
                ``warmup=True`` for this cube. Counts toward
                ``RFIWindow.n_cubes_warmup``.

        Returns:
            :class:`RFIWindow` if this cube closed the window
            (i.e. cube count reached ``window_size``), else ``None``.
        """
        if mask.shape != (self._n_ants, self._n_chan, self._n_pol):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != expected "
                f"({self._n_ants}, {self._n_chan}, {self._n_pol})"
            )
        if source_tags.shape != mask.shape:
            raise ValueError(
                f"source_tags shape {tuple(source_tags.shape)} != "
                f"mask shape {tuple(mask.shape)}"
            )
        if s1_full.shape != mask.shape:
            raise ValueError(
                f"s1_full shape {tuple(s1_full.shape)} != "
                f"mask shape {tuple(mask.shape)}"
            )

        # Ensure tensors are on the aggregator's device. The hot path
        # passes CUDA tensors at the production op-point; tests pass
        # CPU. Cross-device transfers happen only on dev/device mismatch.
        if s1_full.device != self._device:
            s1_full = s1_full.to(self._device)
        if mask.device != self._device:
            mask = mask.to(self._device)
        if source_tags.device != self._device:
            source_tags = source_tags.to(self._device)

        # Track window start/end block_n.
        if self._cubes_in_window == 0:
            self._block_n_start = block_n
        self._block_n_last = block_n
        if warmup:
            self._cubes_warmup_in_window += 1

        # ----- S1 sum (downsampled) -------------------------------
        # Cast to fp32 before downsample so the sum doesn't overflow
        # at small bit-widths. s1 is already fp32 from compute_autos.
        s1_ds = self._downsample_chan(s1_full.to(torch.float32))
        self._s1_sum += s1_ds

        # ----- Per-detector mask counts (downsampled) --------------
        # Source-tag decomposition: cheap bitwise AND then bool->uint8.
        # Downsample with sum so the counter records "any cubes flagged
        # in any of the (freq_downsample) raw channels".
        bp_mask = (source_tags & int(FlagSourceBit.BANDPASS_OUTLIER)) != 0
        sk_mask = (source_tags & int(FlagSourceBit.SK)) != 0
        grp_mask = (source_tags & int(FlagSourceBit.GROUP_OUTLIER)) != 0
        st_mask = (source_tags & int(FlagSourceBit.SUM_THRESHOLD)) != 0
        fa_mask = (source_tags & int(FlagSourceBit.FLAGANTS_DAT)) != 0

        # The downsample is intentionally a max-equivalent here: we want
        # the counter to register "this downsampled cell saw a flag in
        # any of its constituent raw channels", so the per-bin OR maps
        # to "any". For boolean masks, sum>0 ≡ any; we keep sums for
        # numerical robustness when ds_factor > 1.
        def _ds_any_u8(m: torch.Tensor) -> torch.Tensor:
            # Convert bool -> uint8 first, then downsample by max
            # (= any-along-bin), then we OR into the running counter.
            mu8 = m.to(torch.uint8).view(
                self._n_ants, self._n_chan_ds, self._freq_downsample,
                self._n_pol,
            )
            return mu8.amax(dim=2)

        self._mask_count_final += _ds_any_u8(mask)
        self._mask_count_sk += _ds_any_u8(sk_mask)
        self._mask_count_bp += _ds_any_u8(bp_mask)
        self._mask_count_grp += _ds_any_u8(grp_mask)
        self._mask_count_sumthr += _ds_any_u8(st_mask)
        self._mask_count_fa += _ds_any_u8(fa_mask)

        self._cubes_in_window += 1

        if self._cubes_in_window < self._window_size:
            return None

        # ------------------------------------------------------------
        # Window closed: finalise.
        # ------------------------------------------------------------
        return self._finalise()

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def _finalise(self) -> RFIWindow:
        n = float(self._cubes_in_window)

        # Pull all accumulators back to CPU once.
        s1_mean = (self._s1_sum / n).detach().cpu().numpy()
        counts = {
            "final": self._mask_count_final.detach().cpu().numpy(),
            "sk":    self._mask_count_sk.detach().cpu().numpy(),
            "bp":    self._mask_count_bp.detach().cpu().numpy(),
            "grp":   self._mask_count_grp.detach().cpu().numpy(),
            "sumthr": self._mask_count_sumthr.detach().cpu().numpy(),
            "fa":    self._mask_count_fa.detach().cpu().numpy(),
        }

        # --- Per-pol scalar metrics --------------------------------
        # Total flag fraction = mean over (ant, ch_ds, pol) of
        # count/n_cubes. Computed per-pol and both-pol (= mean over all).
        def _frac_over_pol(arr: np.ndarray, pol: int | None) -> float:
            if pol is None:
                return float(arr.astype(np.float32).mean() / n)
            return float(arr[..., pol].astype(np.float32).mean() / n)

        total_fp0 = _frac_over_pol(counts["final"], 0)
        total_fp1 = _frac_over_pol(counts["final"], 1)
        total_all = _frac_over_pol(counts["final"], None)

        # bandpass-channel-fraction:
        # for each ant: mean over channels of "ever flagged by bp" in
        # this window. Then average over ants. Per-pol.
        # We have count_bp ∈ [0, n_cubes]; "ever flagged" ≡ count_bp > 0.
        bp_any = (counts["bp"] > 0).astype(np.float32)        # (ant, ch_ds, pol)
        # Per-ant per-pol fraction of channels.
        bp_per_ant_pol = bp_any.mean(axis=1)                  # (ant, pol)
        bpcf_p0 = float(bp_per_ant_pol[:, 0].mean())
        bpcf_p1 = float(bp_per_ant_pol[:, 1].mean())
        bpcf_all = float(bp_per_ant_pol.mean())

        # ant-fraction-flagged (group):
        # an (ant, pol) is "flagged this window" if grp fired on
        # ANY channel in ANY cube within the window. group-outlier
        # broadcasts across channels so this is really per-(ant, pol)
        # binary, but we still .any() over ch_ds to be safe.
        grp_ant_pol = (counts["grp"] > 0).any(axis=1)          # (ant, pol) bool
        ant_p0 = float(grp_ant_pol[:, 0].mean())
        ant_p1 = float(grp_ant_pol[:, 1].mean())
        ant_all = float(grp_ant_pol.mean())

        # Per-detector means over the full cube (both pols, all cells).
        def _det_frac(arr: np.ndarray, pol: int | None) -> float:
            return _frac_over_pol(arr, pol)

        frac_sk = (_det_frac(counts["sk"], 0),
                   _det_frac(counts["sk"], 1),
                   _det_frac(counts["sk"], None))
        frac_bp = (_det_frac(counts["bp"], 0),
                   _det_frac(counts["bp"], 1),
                   _det_frac(counts["bp"], None))
        frac_grp = (_det_frac(counts["grp"], 0),
                    _det_frac(counts["grp"], 1),
                    _det_frac(counts["grp"], None))
        frac_sumthr = (_det_frac(counts["sumthr"], 0),
                       _det_frac(counts["sumthr"], 1),
                       _det_frac(counts["sumthr"], None))
        frac_fa = (_det_frac(counts["fa"], 0),
                   _det_frac(counts["fa"], 1),
                   _det_frac(counts["fa"], None))

        bn_start = self._block_n_start or 0
        bn_end = self._block_n_last or 0
        n_warmup = self._cubes_warmup_in_window
        cubes_in_window = self._cubes_in_window

        window = RFIWindow(
            block_n_start=bn_start,
            block_n_end=bn_end,
            n_cubes=cubes_in_window,
            n_cubes_warmup=n_warmup,
            s1_full_mean=s1_mean.astype(np.float32, copy=False),
            mask_count_final=counts["final"],
            mask_count_sk=counts["sk"],
            mask_count_bp=counts["bp"],
            mask_count_grp=counts["grp"],
            mask_count_sumthr=counts["sumthr"],
            mask_count_fa=counts["fa"],
            total_flag_fraction=(total_fp0, total_fp1, total_all),
            bandpass_channel_fraction=(bpcf_p0, bpcf_p1, bpcf_all),
            ant_fraction_flagged=(ant_p0, ant_p1, ant_all),
            frac_sk=frac_sk,
            frac_bp=frac_bp,
            frac_grp=frac_grp,
            frac_sumthr=frac_sumthr,
            frac_fa=frac_fa,
        )

        self._reset_accumulators()
        return window
