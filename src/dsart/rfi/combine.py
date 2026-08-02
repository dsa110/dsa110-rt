"""Top-level RFI flagger integrator (M3 chunk 3c; plan §4.2 steps 6-7).

Pulls together the four per-cube detectors plus the static
``flagants.dat`` overlay, OR-folds them into one
``[NANTS, NCHAN, NPOL]`` boolean mask, and emits a
``[NANTS, NCHAN, NPOL] uint8`` "flag-source" tag tensor where each bit
records which detector fired in that cell:

.. table:: Source-tag bit layout

    ===============   =====
    bit               1<<n
    ===============   =====
    SK                0
    bandpass-outlier  1
    group-outlier     2
    sum-threshold     3
    flagants.dat      4
    persistence       5
    ===============   =====

The numeric values are exposed as :class:`FlagSourceBit` for
downstream diagnostic logging.

Time persistence
================

The four detectors are memoryless — each cube is judged on its own
statistics. :mod:`dsart.rfi.persistence` adds an optional latch on top:
a cell the detectors flag for a whole 30 s window stays flagged for
15 min afterwards (bit 5). The latch is fed the *detector* OR only, so
it can expire; see that module's docstring for the recurrences and the
cost analysis. Off unless ``persistence=`` is passed to
:class:`RFIFlagger`.

Cold-start state machine
========================

Plan §4.2 step "Cold start" requires that during the first ``5·τ_B``
cubes after pipeline start (``τ_B`` = bandpass-outlier MAD warmup
window, default 30 s ≈ ~224 cubes at 134 ms cadence), the
bandpass-outlier detector be **bypassed** while SK + group-outlier
remain active, and that the cube's transport header carry
``flags.bit4 = rfi_warming_up = 1``.

For chunk 3c we expose this behaviour via a counter-driven state
machine inside :class:`RFIFlagger` plus a small
:class:`MockTransportHeader` adapter for the
:mod:`bench.rfi_warmup` test. The parent M3 agent wires the live
``corr_fast_compute`` service to call ``flag_block`` and propagate
``result.warmup`` into the real transport header (see plan §4.3
``flags.bit4``).

Usage::

    flagger = RFIFlagger(
        flagants_path="/.../flagants.dat",
        sk_far=1e-4, bandpass_k=5.0, group_k=5.0,
    )
    result = flagger.flag_block(real, imag)
    mask, tags, warmup = result.mask, result.source_tags, result.warmup
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dsart.common.constants import (
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    RFI_BANDPASS_WARMUP_CUBES_DEFAULT,
)
from dsart.rfi.autos import DEFAULT_M_VALUES, AutoSpectra, compute_autos
from dsart.rfi.bandpass_outlier import (
    DEFAULT_BANDPASS_K,
    bandpass_outlier_mask,
)
from dsart.rfi.flagants_loader import load_flagants_torch
from dsart.rfi.group_outlier import DEFAULT_GROUP_K, group_outlier_mask
from dsart.rfi.persistence import FlagPersistence
from dsart.rfi.sk import DEFAULT_SK_FAR, sk_combined_mask
from dsart.rfi.sum_threshold import (
    DEFAULT_ETA,
    DEFAULT_MAX_M,
    sum_threshold_1d,
)


# ---------------------------------------------------------------------------
# Source-tag bit layout
# ---------------------------------------------------------------------------


class FlagSourceBit(enum.IntFlag):
    """Per-cell flag-source tag bits (uint8)."""

    NONE = 0
    SK = 1 << 0                  # value 1
    BANDPASS_OUTLIER = 1 << 1    # value 2
    GROUP_OUTLIER = 1 << 2       # value 4
    SUM_THRESHOLD = 1 << 3       # value 8
    FLAGANTS_DAT = 1 << 4        # value 16
    PERSISTENCE = 1 << 5         # value 32


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlagBlockResult:
    """Per-cube output of :meth:`RFIFlagger.flag_block`.

    Args:
        mask: bool tensor ``[NANTS, NCHAN, NPOL]`` — final OR-folded
            cube mask.
        source_tags: uint8 tensor ``[NANTS, NCHAN, NPOL]`` —
            per-cell bitfield of :class:`FlagSourceBit` flags.
        warmup: True iff this cube is inside the cold-start window
            (``5·τ_B`` cubes since RFIFlagger init); transports
            should set ``flags.bit4 = rfi_warming_up`` for these
            cubes.
        flag_fraction_total: scalar fraction of cells flagged
            (``mask.float().mean()``). Convenience for monitoring.
        s1_full: fp32 ``S1_4096`` per (ant, ch, pol). Same shape as
            ``mask``. The full-cube auto-power that the flagger
            computed internally from
            :func:`dsart.rfi.compute_autos` (and that bandpass-outlier
            / group-outlier consumed). Exposed here so monitoring
            (M7.6 :class:`dsart.services.rfi_window.RFIWindowAggregator`)
            can build a per-ant pre-flag bandpass without
            recomputing the autos. ``None`` for callers that pass an
            ``autos_override`` whose ``s1`` does not contain the
            full-block M; only set on the production code path that
            calls ``compute_autos`` itself.
        n_persist_latched: number of cells currently held flagged by
            the :mod:`dsart.rfi.persistence` latch (0 when persistence
            is disabled). Read off the same host sync that produces
            ``flag_fraction_total``, so it costs nothing extra.
        n_persist_new: number of cells that latched on *this* cube.
    """

    mask: torch.Tensor
    source_tags: torch.Tensor
    warmup: bool
    flag_fraction_total: float
    s1_full: torch.Tensor | None = None
    n_persist_latched: int = 0
    n_persist_new: int = 0


# ---------------------------------------------------------------------------
# Mock transport header (chunk-3c testing convenience; bench/rfi_warmup.py
# uses this to exercise the warmup state machine)
# ---------------------------------------------------------------------------


@dataclass
class MockTransportHeader:
    """Stand-in for the live transport header used by chunk 3c benches.

    Mirrors the relevant subset of plan §4.3 ``flags`` semantics:
    ``bit4 = rfi_warming_up``. The parent M3 agent's transport spec
    (M4a) replaces this with the real header dataclass.
    """

    flags: int = 0

    BIT_RFI_WARMING_UP: int = field(default=1 << 4, init=False, repr=False)

    def set_rfi_warmup(self, warming_up: bool) -> None:
        """Set / clear the ``rfi_warming_up`` bit (bit 4)."""
        if warming_up:
            self.flags |= self.BIT_RFI_WARMING_UP
        else:
            self.flags &= ~self.BIT_RFI_WARMING_UP

    def is_rfi_warming_up(self) -> bool:
        return bool(self.flags & self.BIT_RFI_WARMING_UP)


# ---------------------------------------------------------------------------
# Top-level flagger
# ---------------------------------------------------------------------------


def _flagants_to_cube(
    flagants_mask: torch.Tensor,
    *,
    n_ant: int = NANTS,
    n_ch: int = NCHAN_PER_CHGROUP,
    n_pol: int = NPOL,
) -> torch.Tensor:
    """Broadcast a flagants ant mask to ``[n_ant, n_ch, n_pol]``.

    Args:
        flagants_mask: bool tensor of length ≥ ``n_ant``. Production
            uses :data:`NANTS`; tests pass a shorter reduced mask.
        n_ant, n_ch, n_pol: target cube dims. Defaults are the
            canonical voltage cube dims.
    """
    if flagants_mask.shape[0] < n_ant:
        raise ValueError(
            f"flagants_mask length {flagants_mask.shape[0]} < n_ant {n_ant}"
        )
    return (
        flagants_mask[:n_ant].view(n_ant, 1, 1)
        .expand(n_ant, n_ch, n_pol)
        .contiguous()
    )


class RFIFlagger:
    """Top-level cube-cadence RFI flagger.

    Wires SK + bandpass-outlier + group-outlier + sum-threshold +
    flagants.dat into a single ``flag_block(real, imag)`` call. Holds
    the warmup-counter state used to suppress bandpass-outlier during
    the cold-start window.

    Args:
        flagants_path: path to legacy ``flagants.dat`` (or ``None``
            to skip the static-ant overlay; useful for tests).
        device: torch device for the static flagants mask. Detector
            tensors are placed on the same device as the input
            voltages at flag time.
        sk_far: per-(ant, ch, pol, M) two-sided false-alarm rate for
            SK.
        bandpass_k: outlier-σ threshold for the bandpass-outlier
            detector.
        group_k: outlier-σ threshold for the group-outlier detector.
        sum_threshold_max_m: max sliding-window length for the
            SumThreshold post-pass.
        sum_threshold_eta: threshold-shape parameter for SumThreshold.
        m_values: SK accumulation depths. Default
            :data:`dsart.rfi.autos.DEFAULT_M_VALUES`.
        warmup_cubes: number of cubes during which to bypass the
            bandpass-outlier and assert ``flags.bit4 = rfi_warming_up``.
            Default :data:`RFI_BANDPASS_WARMUP_CUBES_DEFAULT`. Tests
            override to small integers (1-5).
        run_sum_threshold: include the SumThreshold post-pass
            (default True). Tests can disable to inspect raw
            per-detector flags.
        persistence: optional :class:`dsart.rfi.FlagPersistence` latch.
            When given, cells the detectors flag for a whole trailing
            window stay flagged for the configured hold time and carry
            :attr:`FlagSourceBit.PERSISTENCE`. ``None`` (default) keeps
            the pre-existing memoryless behaviour exactly.

    Notes
    -----
    The flagger is *stateful* across cubes via ``self._cubes_seen`` (the
    warmup counter) and, when configured, the persistence latch. All
    other state — thresholds, configs — is immutable after
    construction. The instance is safe to reuse across a long run; call
    :meth:`reset_warmup` to re-arm the warmup window (e.g. after a
    re-cal or pipeline restart) and :meth:`reset_persistence` to drop
    every latch.
    """

    def __init__(
        self,
        *,
        flagants_path: str | Path | None,
        device: torch.device | str = "cpu",
        sk_far: float = DEFAULT_SK_FAR,
        bandpass_k: float = DEFAULT_BANDPASS_K,
        group_k: float = DEFAULT_GROUP_K,
        sum_threshold_max_m: int = DEFAULT_MAX_M,
        sum_threshold_eta: float = DEFAULT_ETA,
        m_values: tuple[int, ...] = DEFAULT_M_VALUES,
        warmup_cubes: int = RFI_BANDPASS_WARMUP_CUBES_DEFAULT,
        run_sum_threshold: bool = True,
        persistence: FlagPersistence | None = None,
    ) -> None:
        if warmup_cubes < 0:
            raise ValueError(
                f"warmup_cubes={warmup_cubes}, expected >= 0"
            )
        self._device = torch.device(device)
        self._sk_far = sk_far
        self._bandpass_k = bandpass_k
        self._group_k = group_k
        self._st_max_m = sum_threshold_max_m
        self._st_eta = sum_threshold_eta
        self._m_values = tuple(m_values)
        self._warmup_cubes = warmup_cubes
        self._run_sum_threshold = run_sum_threshold
        self._persistence = persistence

        if flagants_path is None:
            self._flagants_mask = torch.zeros(
                NANTS, dtype=torch.bool, device=self._device,
            )
        else:
            self._flagants_mask = load_flagants_torch(
                flagants_path, device=self._device,
            )

        self._cubes_seen = 0

    # ------------------------------------------------------------------
    # Properties / state-machine controls
    # ------------------------------------------------------------------

    @property
    def cubes_seen(self) -> int:
        """Number of cubes processed since construction / last reset."""
        return self._cubes_seen

    @property
    def in_warmup(self) -> bool:
        """``True`` during the cold-start ``warmup_cubes`` window."""
        return self._cubes_seen < self._warmup_cubes

    @property
    def warmup_cubes(self) -> int:
        return self._warmup_cubes

    @property
    def flagants_mask(self) -> torch.Tensor:
        return self._flagants_mask

    @property
    def persistence(self) -> FlagPersistence | None:
        """The configured time-persistence latch, or ``None``."""
        return self._persistence

    def reset_warmup(self) -> None:
        """Reset the warmup counter (e.g. after pipeline restart)."""
        self._cubes_seen = 0

    def reset_persistence(self) -> None:
        """Drop every persistence latch (no-op when not configured)."""
        if self._persistence is not None:
            self._persistence.reset()

    # ------------------------------------------------------------------
    # Top-level call
    # ------------------------------------------------------------------

    def flag_block(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        *,
        n_packets: int = 2048,
        n_times_per_packet: int = 2,
        autos_override: AutoSpectra | None = None,
        update_header: MockTransportHeader | None = None,
    ) -> FlagBlockResult:
        """Run the full per-cube flagging pipeline on one voltage block.

        Args:
            real, imag: voltage tensors in M2 GEMM layout
                (see :func:`dsart.rfi.autos.compute_autos`). Ignored
                if ``autos_override`` is provided.
            n_packets, n_times_per_packet: passed through to
                :func:`compute_autos`. Tests use smaller blocks; the
                production path uses the canonical
                ``(2048, 2)``.
            autos_override: pre-computed :class:`AutoSpectra` (e.g.
                from :func:`compute_autos_from_complex` for synthetic
                tests). When set, ``real``/``imag`` may be ``None``.
            update_header: optional :class:`MockTransportHeader` to
                update with the warmup bit. Passed by reference; the
                method calls ``set_rfi_warmup(self.in_warmup)`` on it
                **before** advancing the cube counter.

        Returns:
            :class:`FlagBlockResult` with the OR-folded mask, per-cell
            source tags, warmup indicator, and total flag fraction.

        Raises:
            ValueError: invalid input shapes (delegated to underlying
                detectors).
        """
        if autos_override is None:
            autos = compute_autos(
                real, imag,
                m_values=self._m_values,
                n_packets=n_packets,
                n_times_per_packet=n_times_per_packet,
            )
        else:
            autos = autos_override

        warmup_flag = self.in_warmup

        # ---- SK over all M's --------------------------------------
        sk_m = sk_combined_mask(
            autos.s1, autos.s2, far=self._sk_far,
        )                                                   # (NANTS, NCHAN, NPOL) bool

        # ---- Bandpass-outlier (bypassed during cold-start) --------
        # The full-cube auto-power is the M = 4096 entry's only
        # accumulation; squeeze its leading axis.
        full_m = max(self._m_values)
        s1_full = autos.s1[full_m].squeeze(0)               # (NANTS, NCHAN, NPOL)
        if warmup_flag:
            bp_m = torch.zeros_like(sk_m)
        else:
            bp_m = bandpass_outlier_mask(s1_full, k=self._bandpass_k)

        # ---- Group-outlier (always active) ------------------------
        gr_m = group_outlier_mask(s1_full, k=self._group_k)

        # ---- Sum-threshold post-pass (along channel axis only at
        #      this cube-level granularity; the (ch, t) 2D form lives
        #      on the SK n_acc plane and is exposed separately for
        #      tests / future per-sub-cube extension). The OR'd
        #      sk + bandpass mask is the natural input — group-outlier
        #      flags are already cube-wide so dilation is a no-op).
        if self._run_sum_threshold:
            base = sk_m | bp_m
            # Move ch to last axis: (NANTS, NPOL, NCHAN). The detector
            # operates per (ant, pol).
            base_t = base.permute(0, 2, 1).contiguous()
            dilated_t = sum_threshold_1d(
                base_t, max_m=self._st_max_m, eta=self._st_eta,
            )
            dilated = dilated_t.permute(0, 2, 1).contiguous()
            # Sum-threshold's "added" cells are the difference.
            st_added = dilated & ~base
            sum_m = st_added
        else:
            sum_m = torch.zeros_like(sk_m)

        # ---- flagants.dat (broadcast [NANTS] → cube shape) -------
        device = sk_m.device
        if self._flagants_mask.device != device:
            self._flagants_mask = self._flagants_mask.to(device)
        n_ant_actual, n_ch_actual, n_pol_actual = sk_m.shape
        fa_m = _flagants_to_cube(
            self._flagants_mask,
            n_ant=n_ant_actual,
            n_ch=n_ch_actual,
            n_pol=n_pol_actual,
        )

        # ---- Detector OR-fold -------------------------------------
        detector_m = sk_m | bp_m | gr_m | sum_m

        # ---- Time persistence (M8.1) ------------------------------
        # Fed the DETECTOR mask only: including the latch's own output
        # would keep every latched cell's run alive forever, and
        # including flagants would latch antennas that are already
        # unconditionally flagged.
        if self._persistence is not None:
            pe_m, pe_stats = self._persistence.update(detector_m)
        else:
            pe_m, pe_stats = None, None

        final = detector_m | fa_m
        if pe_m is not None:
            final = final | pe_m

        # ---- Source-tag uint8 -------------------------------------
        tags = torch.zeros_like(final, dtype=torch.uint8)
        tags |= sk_m.to(torch.uint8) * int(FlagSourceBit.SK)
        tags |= bp_m.to(torch.uint8) * int(FlagSourceBit.BANDPASS_OUTLIER)
        tags |= gr_m.to(torch.uint8) * int(FlagSourceBit.GROUP_OUTLIER)
        tags |= sum_m.to(torch.uint8) * int(FlagSourceBit.SUM_THRESHOLD)
        tags |= fa_m.to(torch.uint8) * int(FlagSourceBit.FLAGANTS_DAT)
        if pe_m is not None:
            tags |= pe_m.to(torch.uint8) * int(FlagSourceBit.PERSISTENCE)

        # One host sync per cube, shared by the flag fraction and the
        # persistence counters (the latch itself never syncs).
        if pe_stats is None:
            flag_frac = float(final.float().mean().item())
            n_latched = n_new = 0
        else:
            frac_t = final.float().mean().to(torch.float64)
            packed = torch.stack((
                frac_t,
                pe_stats.n_latched.to(torch.float64),
                pe_stats.n_new_latched.to(torch.float64),
            )).tolist()
            flag_frac = float(packed[0])
            n_latched, n_new = int(packed[1]), int(packed[2])

        if update_header is not None:
            update_header.set_rfi_warmup(warmup_flag)

        # Advance the cube counter AFTER computing the mask but
        # BEFORE returning — so the caller's `result.warmup` reflects
        # the state DURING this cube, while subsequent cubes count up.
        self._cubes_seen += 1

        return FlagBlockResult(
            mask=final,
            source_tags=tags,
            warmup=warmup_flag,
            flag_fraction_total=flag_frac,
            s1_full=s1_full,
            n_persist_latched=n_latched,
            n_persist_new=n_new,
        )


# ---------------------------------------------------------------------------
# Functional one-shot entry (no warmup state)
# ---------------------------------------------------------------------------


def flag_block(
    real: torch.Tensor,
    imag: torch.Tensor,
    flagants_dat_path: str | Path | None,
    *,
    sk_far: float = DEFAULT_SK_FAR,
    bandpass_k: float = DEFAULT_BANDPASS_K,
    group_k: float = DEFAULT_GROUP_K,
    n_packets: int = 2048,
    n_times_per_packet: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless one-shot flag pass.

    Convenience wrapper for code that wants to flag a single cube
    without instantiating :class:`RFIFlagger` (e.g. unit tests). Every
    call starts with a fresh warmup counter so the bandpass-outlier
    is **bypassed**; callers that want bandpass-outlier active should
    use :class:`RFIFlagger` with ``warmup_cubes=0``.

    Args:
        real, imag: voltage tensors in M2 GEMM layout.
        flagants_dat_path: path to legacy ``flagants.dat`` (or
            ``None`` for no static overlay).
        sk_far, bandpass_k, group_k: detector thresholds. See
            :class:`RFIFlagger` for details.
        n_packets, n_times_per_packet: voltage block dimensions.

    Returns:
        ``(mask, source_tags)`` — same dtypes / shapes as
        :class:`FlagBlockResult` fields of the same name.
    """
    flagger = RFIFlagger(
        flagants_path=flagants_dat_path,
        device=real.device,
        sk_far=sk_far,
        bandpass_k=bandpass_k,
        group_k=group_k,
        warmup_cubes=0,
    )
    result = flagger.flag_block(
        real, imag,
        n_packets=n_packets,
        n_times_per_packet=n_times_per_packet,
    )
    return result.mask, result.source_tags
