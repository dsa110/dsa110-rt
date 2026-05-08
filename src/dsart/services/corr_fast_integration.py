"""Fast correlator FULL integration service (M3 chunk 4; plan §4.2 step 1-7).

Where the chunk-2b spine (:mod:`dsart.services.corr_fast_compute`) wires
**unpack → cal → GEMM → Stokes I** end-to-end, this module assembles the
PRODUCTION pipeline:

  1. ``unpack_int4_split``  (M2 D15 / D16)
  2. ``RFIFlagger.flag_block`` → ``FlagBlockResult`` (M3 chunk 3c)
  3. **Voltage zero-fill** of flagged ``(ant, ch, pol)`` cells   (chunk 4)
     — drops flagged channels OUT of the GEMM so the resulting fast vis
     have zero contribution from RFI-contaminated voltages, instead of
     contaminating downstream cells. The zeroed voltage tensor is then
     fed forward; the fast-vis tile already reflects the F18 sign
     convention so no further correction is needed.
  4. ``apply_cal_split`` (with F21 DEC-phase cal; M3 chunk 1)
  5. ``FastCorrKernel.compute_split`` (M3 chunk 2a)
  6. ``stokes_i_pol_sum``                                          (chunk 2a)
  7. ``FastVisGridder.compute``                                    (chunk 3a)
  8. **Static-sky EMA subtraction** of the sparse-COO cube         (chunk 4)
  9. *(pluggable)* coarse-DM dedispersion              (chunk 3b lands here)
 10. *(pluggable)* stage-2 FIFO push                  (chunk 3b lands here)
 11. *(pluggable)* transport TX                          (M4a / chunk 8)

Chunks 9-11 are gated on incoming work and are exposed as Protocol-
shaped hooks (``CoarseDMStage``, ``Stage2FifoStage``, ``TransportTxStage``)
so chunk 3b's coarse-DM module + chunk 8's transport TX can plug in
without touching this orchestrator (Class-A boundary; see
``PARALLEL_AGENTS.md`` §3).

# Why a separate module from corr_fast_compute

``corr_fast_compute`` (chunk 2b) is the **spine** that proves chunks
1-2a compose end-to-end. It writes raw fast-vis tiles to disk for
inspection — it intentionally does not include the RFI flagger,
gridder, or static-sky stages. ``corr_fast_integration`` is the
PRODUCTION service shell that adds those + the pluggable hooks. The
two services share the per-block compute scaffolding but differ in
their pipeline graph + on-disk artefacts.

Per PARALLEL_AGENTS.md §4: this service runs on h01 GPU 0 with
``DSART_BUFFER_KEY_PREFIX=m3`` (which maps the legacy ``fada`` key to
``fa3a`` per the buffer-key prefix rule).

CLI:

    python -m dsart.services.corr_fast_integration \\
        [--fada-key fada]
        [--device auto]
        [--max-blocks N]
        [--t-int-fast-native 8]
        [--obs-dec-deg 53.848986]
        [--apply-cal /path/to/beamformer_weights_*.dat]
        [--cal-mode phase_only|full]
        [--cal-pol-swap]
        [--flagants /path/to/flagants.dat]
        [--n-grid 256]
        [--kernel-support 1]
        [--static-sky-alpha 0.001]
        [--static-sky-disabled]
        [--n-fv-chunk N]               # F31b: bound peak GPU memory
        [--output-dir /tmp/dsart-fast-grid]
        [--blocks-output-mode full|first_tile_only|none]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import torch

from dsart.cal.cal_loader import (
    CalMode,
    FastCorrCalTensors,
    load_cal_with_dec_phase,
)
from dsart.common.config_loader import load
from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    N_CHGROUP,
    NANTS,
    NATIVE_SAMPLE_US,
    NBASE,
    NCHAN_PER_CHGROUP,
    NPOL,
    PHI_LAT_OVRO_DEG,
    T_INT_FAST_NATIVE,
)
from dsart.grid import (
    FastVisGridder,
    SparsityPattern,
    build_pattern,
    compute_top_of_band_cell_lambda,
)
from dsart.rfi import (
    FlagBlockResult,
    RFIFlagger,
)
from dsart.services.corr_fast_kernel import (
    FastCorrKernel,
    stokes_i_pol_sum,
)
from dsart.services.slow_corr_kernel import (
    NTIMES_PER_PACKET,
    apply_cal_split,
    unpack_int4_split,
)
from dsart.coarse_dm.dm_plan import DMPlan
from dsart.coarse_dm.stage1 import (
    apply_stage1_shifts,
    max_t_dedisp_for_plan,
)


LOG = logging.getLogger("corr_fast_integration")

DEFAULT_CONFIG_PATH = Path("/home/ubuntu/proj/dsa110-rt/configs/config_corr.yaml")


# ---------------------------------------------------------------------------
# Pluggable stage protocols
# ---------------------------------------------------------------------------


class CoarseDMStage(Protocol):
    """Coarse-DM dedispersion hook (chunk 3b plug-in surface).

    Chunk 3b's ``CoarseDedisperser.dedisperse_cube`` will satisfy this
    protocol once it lands. Until then, a no-op stub is wired by
    ``IntegrationContext.coarse_dm = NoOpCoarseDM()``.
    """

    def dedisperse(
        self,
        gridded: torch.Tensor,
        *,
        block_n: int,
        chgroup: int,
    ) -> torch.Tensor:
        """Return dedispersed cube of shape ``(N_DM, n_fast_vis, N_filled)``.

        For the no-op stub, returns the input wrapped in a
        ``[1, ...]`` axis (single trivial DM trial).
        """
        ...


class Stage2FifoStage(Protocol):
    """Stage-2 FIFO push hook (chunk 3b plug-in surface)."""

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        """Push ``dedispersed`` into the FIFO; return any cubes that
        were evicted by this push (ready for transport TX).
        """
        ...


class TransportTxStage(Protocol):
    """Transport TX hook (chunk 8 plug-in surface)."""

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        """Transmit ``cubes_for_tx``; return count of cubes actually
        sent (== ``len(cubes_for_tx)`` for the production case;
        ``0`` for the no-op stub).
        """
        ...


# ---------------------------------------------------------------------------
# No-op stubs (Chunk 4 placeholder; chunk 3b / chunk 8 swap these out)
# ---------------------------------------------------------------------------


@dataclass
class NoOpCoarseDM:
    """Identity dedisperser: emits the gridded cube untouched on a
    single ``DM=0`` trial axis. Used until chunk 3b plugs in.

    Production replacement: ``dsart.coarse_dm.CoarseDedisperser``
    (Class A; chunk 3b owns it).
    """

    def dedisperse(
        self,
        gridded: torch.Tensor,
        *,
        block_n: int,
        chgroup: int,
    ) -> torch.Tensor:
        return gridded.unsqueeze(0)


@dataclass
class NoOpStage2Fifo:
    """Identity FIFO: every push immediately evicts the same cube
    (depth = 0). Used until chunk 3b plugs in.

    Production replacement: ``dsart.coarse_dm.Stage2Fifo`` (Class A;
    chunk 3b owns it).
    """

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        return [dedispersed]


@dataclass
class NoOpTransportTx:
    """Drop-on-the-floor TX. Used until chunk 8 plugs in.

    Production replacement: ``dsart.transport.tx.TransportTx``
    (Class A; chunk 8 owns it).
    """

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        return 0


# ---------------------------------------------------------------------------
# Stage-1 multi-DM-trial coarse-DM (F25 production path)
# ---------------------------------------------------------------------------


@dataclass
class Stage1MultiDMCoarseDM:
    """Production coarse-DM stage: vis-domain stage-1 shifts → per-trial grid.

    Per F25 (M3_PLAN_FIXES.md), the production multi-DM-trial
    integration applies per-channel integer-bin shifts on the
    visibility tensor before the gridder, once per coarse-DM trial.
    The :func:`apply_stage1_shifts` primitive provides the math; this
    class wraps it in the chunk-4 ``CoarseDMStage`` Protocol shape so
    the orchestrator's :func:`process_block_multi_dm` can call it
    uniformly across the DM-axis loop.

    Constructed once per chgroup at service startup; held in
    ``IntegrationContext.coarse_dm`` for the lifetime of the run.
    The ``plan`` is read-only; ``gridder`` is the same one used for
    the single-DM legacy path.

    Args:
        plan: :class:`dsart.coarse_dm.dm_plan.DMPlan` whose
            ``t_int_fast_us`` matches the runtime
            ``t_int_fast_native``. Construction-time pin: chunk-9
            tests + the chunk-4 orchestrator both check that
            ``plan.t_int_fast_native`` equals
            ``ctx.cfg.t_int_fast_native``.
        gridder: chunk-3a :class:`dsart.grid.FastVisGridder`.
        chgroup: chgroup index this dedisperser is bound to (matches
            ``ctx.cfg.chgroup``).
        dm_indices: optional ``(N_DM_subset,)`` int subset of the
            plan's DM trials to compute (default: all
            ``plan.n_coarse``).
    """

    plan: DMPlan
    gridder: "FastVisGridder"
    chgroup: int
    dm_indices: np.ndarray | None = None
    sliding_window: bool = False
    dm_chunk_size: int = 2
    """RT Phase 2: number of coarse-DM trials per gridder.compute call.
    See :attr:`IntegrationCfg.dm_chunk_size` for memory + perf rationale.
    Default 2 matches the production cfg default (chunk=4 OOMs on a
    2080 Ti at the joined-window op-point)."""
    """F34: 2-block sliding-window stage-1.

    When ``True``, :meth:`dedisperse_from_vis` joins the previous
    block's ``vis_stokes_i`` with the current one along the time
    axis before applying stage-1 shifts and gridding, then emits
    the dedispersed slice corresponding to the **previous** block
    (now fully resolved against any pulse whose lower-frequency
    tail crosses into the current block). The first call emits an
    all-zero output (no previous block to resolve); thereafter the
    pipeline runs at one-block-in / one-block-out cadence with
    one block of latency.

    Required at the M3 production op-point: at DM = 3000 pc/cc and
    ``t_int_fast_native = 8`` (262.144 µs cadence), the max
    intra-chgroup delay reaches ~480 fast-vis bins (~ block size
    of 512 bins), so a pulse landing near the end of any block
    has its lower-frequency tail dispersed into the next block.
    Single-block stage-1 silently truncates that tail; the
    sliding window recovers it.

    K = 2 (= ring-buffer depth) suffices because the max delay
    is bounded by the DM = 3000 ceiling per the M3 production
    review."""

    _t_dedisp_cache: dict[int, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    _dm_idx_iter: np.ndarray = field(init=False, repr=False)
    _prev_vis_stokes_i: torch.Tensor | None = field(
        default=None, init=False, repr=False,
    )
    _prev_block_n: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.chgroup < N_CHGROUP:
            raise ValueError(
                f"Stage1MultiDMCoarseDM.chgroup={self.chgroup}, "
                f"expected 0..{N_CHGROUP - 1}"
            )
        if self.dm_indices is None:
            self._dm_idx_iter = np.arange(
                self.plan.n_coarse, dtype=np.int64,
            )
        else:
            arr = np.asarray(self.dm_indices, dtype=np.int64).reshape(-1)
            if arr.size == 0:
                raise ValueError("dm_indices is empty")
            if int(arr.min()) < 0 or int(arr.max()) >= self.plan.n_coarse:
                raise IndexError(
                    f"dm_indices contain out-of-range trials "
                    f"(plan.n_coarse={self.plan.n_coarse})"
                )
            self._dm_idx_iter = arr

    @property
    def n_dm(self) -> int:
        return int(self._dm_idx_iter.shape[0])

    def t_dedisp_for(self, n_fast_vis: int) -> int:
        """Uniform output time-axis length across all DM trials.

        Equals ``n_fast_vis - max_bin_shift_over_all_selected_DMs``.
        Cached per ``n_fast_vis`` value.

        Returns 0 if the cube is too short for any DM trial — caller
        should treat 0 as "skip emit" (matches the chunk-3b stage-2
        FIFO warm-up no-emit policy).
        """
        cache_key = int(n_fast_vis)
        cached = self._t_dedisp_cache.get(cache_key)
        if cached is not None:
            return cached
        t = max_t_dedisp_for_plan(
            cache_key, self.plan,
            chgroup=self.chgroup,
            dm_indices=self._dm_idx_iter,
        )
        self._t_dedisp_cache[cache_key] = int(t)
        return int(t)

    def _dedisperse_one_window(
        self, vis_stokes_i: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-DM stage-1 shift + grid to a single vis tensor.

        Inner primitive shared by the legacy (single-block) and
        F34-sliding-window paths. Returns
        ``(N_DM, T_dedisp(n_fv), N_filled) complex64`` where
        ``T_dedisp(n_fv) = n_fv - max_bin_shift`` over the selected
        DM subset.

        RT Phase 2: groups DM trials into chunks of
        :attr:`dm_chunk_size` (default 2) and fuses each chunk's
        gridder.compute calls into a single (``chunk * t_dedisp``,
        NBASE, NCHAN_eff) scatter. At the production K=1 op-point the
        per-DM scatter is atomic-bound (~500 src contributions / cell);
        widening the scatter's batch axis 2x doubles the number of
        in-flight (t_row, cell) atomic queues, recovering SM
        utilisation that was being left on the table by the 24
        single-DM scatters. Stage-1 gathers write directly into the
        chunk buffer via ``apply_stage1_shifts(out=...)`` so there is
        no extra ~1.6 GB/DM copy.

        The caller is responsible for shape validation (this private
        method is hot-path; the public wrappers do the upfront
        shape-check + max-shift sanity errors).
        """
        n_fv = int(vis_stokes_i.shape[0])
        t_dedisp = self.t_dedisp_for(n_fv)
        n_filled = int(self.gridder.pattern.n_filled)
        nb = int(vis_stokes_i.shape[1])
        nch = int(vis_stokes_i.shape[2])
        device = vis_stokes_i.device
        out = torch.empty(
            (self.n_dm, t_dedisp, n_filled),
            dtype=torch.complex64,
            device=device,
        )
        dm_chunk = max(1, int(getattr(self, "dm_chunk_size", 2)))
        dm_chunk = min(dm_chunk, self.n_dm)

        # Pre-cache the full per-(ch, dm) bin-shift table on the device
        # (sliced to the active vis NCHAN). One small int64 tensor —
        # avoids a per-DM CPU→GPU shuffle inside the chunk loop.
        bin_shifts_full = self.plan.delay_bins_per_chgroup(self.chgroup)
        bin_shifts = bin_shifts_full[:nch, self._dm_idx_iter]            # (NCHAN_v, n_dm)
        bin_shifts_dev = torch.as_tensor(
            bin_shifts, dtype=torch.int64, device=device,
        )                                                                # (NCHAN_v, n_dm)
        t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)

        for c0 in range(0, self.n_dm, dm_chunk):
            c1 = min(c0 + dm_chunk, self.n_dm)
            chunk = c1 - c0
            # Build a (chunk * t_dedisp, 1, NCHAN_v) int64 time-index
            # tensor that, broadcast across NBASE on the gather, lifts
            # `chunk` DM trials' worth of stage-1-shifted vis out of
            # ``vis_stokes_i`` in one CUDA kernel launch (vs `chunk`
            # separate single-DM gathers). Index size: 64 KB per chunk
            # at chunk=2 — negligible.
            bs_chunk = bin_shifts_dev[:, c0:c1]                        # (NCHAN_v, chunk)
            # (chunk, t_dedisp, NCHAN_v) = bin_shifts.T[c, None, ch] + arange[None, t, None]
            t_idx = (
                bs_chunk.t()[:, None, :] + t_arange[None, :, None]
            )                                                          # (chunk, t_dedisp, NCHAN_v)
            t_idx_flat = t_idx.reshape(chunk * t_dedisp, 1, nch)
            t_idx_b = t_idx_flat.expand(chunk * t_dedisp, nb, nch)
            buf = vis_stokes_i.gather(0, t_idx_b)                      # (chunk*t_dedisp, NB, NCH)
            grid_chunk = self.gridder.compute(buf)                     # (chunk*t_dedisp, n_filled)
            out[c0:c1] = grid_chunk.reshape(chunk, t_dedisp, n_filled)
            del buf, grid_chunk, t_idx, t_idx_flat, t_idx_b
        return out

    def dedisperse_from_vis(
        self,
        vis_stokes_i: torch.Tensor,
        *,
        block_n: int,
    ) -> torch.Tensor:
        """Run the per-DM-trial stage-1 shift + gridder loop.

        Parameters
        ----------
        vis_stokes_i : torch.Tensor
            Shape ``(n_fast_vis, NBASE, NCHAN)`` complex64 — output
            of :func:`stokes_i_pol_sum`.
        block_n : int
            Block counter (mirrors PSRDADA page sequence). Used for
            logging only; does NOT affect the math.

        Returns
        -------
        torch.Tensor
            Shape ``(N_DM, T_dedisp, N_filled)`` complex64. Each DM
            trial's slice ``[c]`` is
            ``gridder.compute(stage1_shift(vis_stokes_i, plan, dm_idx_iter[c]))``
            truncated to the uniform ``T_dedisp`` time axis.

            With ``sliding_window=True`` (F34): the FIRST call seeds
            the ring buffer and returns an all-zero
            ``(N_DM, n_fast_vis, N_filled)`` cube. Subsequent calls
            join the previous and current ``vis_stokes_i`` along the
            time axis, dedisperse the join, and emit the slice
            ``[:, :n_fast_vis, :]`` corresponding to the PREVIOUS
            block (now fully resolved). One-block latency.

        Raises
        ------
        ValueError
            If ``vis_stokes_i`` has the wrong shape or the cube is
            too short for any DM trial.
        """
        if vis_stokes_i.ndim != 3:
            raise ValueError(
                f"vis_stokes_i must be 3-D (n_fv, NBASE, NCHAN); "
                f"got {vis_stokes_i.ndim}-D shape "
                f"{tuple(vis_stokes_i.shape)}"
            )
        n_fv = int(vis_stokes_i.shape[0])

        if not self.sliding_window:
            # Legacy (pre-F34) single-block path. Bit-identical.
            t_dedisp = self.t_dedisp_for(n_fv)
            if t_dedisp <= 0:
                raise ValueError(
                    f"n_fast_vis={n_fv} too short for plan max bin "
                    f"shift on chgroup={self.chgroup}; selected DMs "
                    f"span {self.plan.dm_pc_cc[self._dm_idx_iter[0]]:.1f}.."
                    f"{self.plan.dm_pc_cc[self._dm_idx_iter[-1]]:.1f} pc/cc"
                )
            return self._dedisperse_one_window(vis_stokes_i)

        # F34 sliding-window path: join prev + current, dedisp the
        # join, emit the prev block's slice. One-block latency.
        n_filled = int(self.gridder.pattern.n_filled)
        if self._prev_vis_stokes_i is None:
            # Cold start: nothing to emit yet. Save current; return
            # all-zeros at the prev-block shape so downstream stages
            # (static-sky EMA / FIFO / TX) see a uniformly-shaped
            # cube every block. The static-sky EMA's warmup window
            # absorbs this no-emit (it's already gated to ignore
            # all-zeros during cold-start).
            self._prev_vis_stokes_i = vis_stokes_i.clone()
            self._prev_block_n = int(block_n)
            return torch.zeros(
                (self.n_dm, n_fv, n_filled),
                dtype=torch.complex64,
                device=vis_stokes_i.device,
            )

        prev_n_fv = int(self._prev_vis_stokes_i.shape[0])
        if prev_n_fv != n_fv:
            # Variable-block-size streams aren't supported (the static
            # FADA buffer guarantees uniform blocks; warn loudly so
            # any non-static-FADA caller catches the mismatch).
            raise ValueError(
                f"sliding-window: prev block n_fv={prev_n_fv} != "
                f"current n_fv={n_fv}; F34 assumes uniform-block "
                f"streams (PSRDADA fada page or dada_junkdb)."
            )

        joined = torch.cat(
            [self._prev_vis_stokes_i, vis_stokes_i], dim=0,
        )                                                              # (2*n_fv, NBASE, NCHAN_eff)
        joined_n_fv = int(joined.shape[0])
        t_dedisp_joined = self.t_dedisp_for(joined_n_fv)
        if t_dedisp_joined < n_fv:
            # The plan's max bin shift exceeds the prev-block length —
            # i.e. a pulse at top freq in the prev block's first tile
            # has its bot-freq tail FURTHER than (n_fv + n_fv) bins
            # later. K=2 ring buffer can't recover that — would need
            # K≥3. At DM=3000 pc/cc + t_int_fast_native=8 the max
            # shift is ~480 bins ≪ n_fv = 512, so this should never
            # trigger in production; raising loudly here so a future
            # high-DM extension is forced to bump K explicitly.
            raise ValueError(
                f"sliding-window: max plan shift "
                f"({joined_n_fv - t_dedisp_joined}) exceeds "
                f"prev-block n_fv={n_fv}; bump ring-buffer K above 2."
            )

        dedisp_joined = self._dedisperse_one_window(joined)            # (n_dm, t_dedisp_joined, n_filled)

        # Emit the slice corresponding to the PREVIOUS block: the
        # first n_fv tiles of the joined output. Per Convention A
        # (stage-1 reference = chgroup TOP), output time index 0
        # aligns with the prev block's input time 0 at the top
        # channel; lower channels' shifts are absorbed by the join
        # so output[:, :n_fv] is the FULLY-resolved prev block.
        out = dedisp_joined[:, :n_fv, :].clone()

        # Slide: current becomes the new prev for the next call.
        self._prev_vis_stokes_i = vis_stokes_i.clone()
        self._prev_block_n = int(block_n)

        del joined, dedisp_joined
        return out


# ---------------------------------------------------------------------------
# Static-sky EMA subtraction
# ---------------------------------------------------------------------------


@dataclass
class StaticSkyEMA:
    """Per-cell exponential moving average of the gridded cube.

    Subtracts the running mean from each new cube before forwarding,
    then updates the running mean with the (post-subtraction) cube
    blended in at rate ``alpha``.

    The EMA is computed in **complex** to preserve the F20-aligned
    visibility phase; the subtraction is therefore complex too.
    Continuum sources at fixed sky positions decorrelate to a
    near-constant per-cell value across cubes (modulo Earth-rotation
    fringe winding, which the F21 DEC-phase cal already partially
    de-rotates), so the EMA learns + subtracts them.

    Args:
        alpha: EMA smoothing factor in (0, 1]. The EMA half-life is
            ``ln(0.5) / ln(1-alpha) ≈ 0.69 / alpha`` cubes. Default
            ``0.001`` → ~700-cube half-life (~7 s at 134 ms cube
            cadence).
        warmup_cubes: number of cubes at the start during which we
            BUILD the EMA but do NOT subtract (so the first few
            cubes are not artificially zeroed by the cold EMA).
    """

    alpha: float = 0.001
    warmup_cubes: int = 8

    _running_mean: torch.Tensor | None = field(
        default=None, init=False, repr=False,
    )
    _cubes_seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(
                f"StaticSkyEMA.alpha={self.alpha}, expected (0, 1]"
            )
        if self.warmup_cubes < 0:
            raise ValueError(
                f"StaticSkyEMA.warmup_cubes={self.warmup_cubes}, "
                f"expected >= 0"
            )

    @property
    def cubes_seen(self) -> int:
        return self._cubes_seen

    @property
    def in_warmup(self) -> bool:
        return self._cubes_seen < self.warmup_cubes

    def reset(self) -> None:
        self._running_mean = None
        self._cubes_seen = 0

    def apply(self, gridded: torch.Tensor) -> torch.Tensor:
        """Subtract the running mean from ``gridded`` and update the EMA.

        ``gridded`` is expected to be ``(n_fast_vis, N_filled)`` complex64
        (the output of :meth:`FastVisGridder.compute`). The EMA is
        kept at ``(N_filled,)`` complex64 — averaged over the
        ``n_fast_vis`` axis on the way in, broadcast back on the way
        out.
        """
        if not gridded.is_complex():
            raise TypeError(
                f"StaticSkyEMA.apply: gridded must be complex; got "
                f"{gridded.dtype}"
            )
        if gridded.ndim != 2:
            raise ValueError(
                f"StaticSkyEMA.apply: gridded must be 2D "
                f"(n_fast_vis, N_filled); got "
                f"{tuple(gridded.shape)}"
            )

        per_cell_mean = gridded.mean(dim=0)                              # (N_filled,)

        if self._running_mean is None:
            self._running_mean = per_cell_mean.clone().detach()
            out = gridded.clone()                                        # cold start: pass through
        elif self.in_warmup:
            out = gridded.clone()                                        # build EMA, don't subtract
            self._running_mean = (
                (1.0 - self.alpha) * self._running_mean
                + self.alpha * per_cell_mean
            )
        else:
            out = gridded - self._running_mean.unsqueeze(0)              # subtract, then update
            self._running_mean = (
                (1.0 - self.alpha) * self._running_mean
                + self.alpha * per_cell_mean
            )

        self._cubes_seen += 1
        return out


# ---------------------------------------------------------------------------
# RFI mask → voltage-cube zero-fill
# ---------------------------------------------------------------------------


def apply_rfi_mask_to_voltages(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    rfi_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero ``(ant, ch, pol)`` cells flagged by the RFI flagger.

    Args:
        real_v, imag_v: voltage tensors in M2 GEMM layout
            ``(NCHAN, NTIMES_PER_PACKET, NPOL, NPACKETS_PER_BLOCK,
            NANTS)`` (any float dtype).
        rfi_mask: bool tensor ``(NANTS, NCHAN, NPOL)`` from
            :class:`FlagBlockResult.mask`. **True == flagged**.

    Returns:
        Pair of voltage tensors with flagged cells set to zero
        in-place. The mask is broadcast to the GEMM layout via
        ``(NCHAN, 1, NPOL, 1, NANTS)``.

    Notes
    -----
    Zero-filling at the voltage level (rather than masking the
    visibilities) means the GEMM never sees the bad samples — so any
    baseline that touches a flagged antenna also has zero contribution
    on those (ch, pol) cells, which is the physically correct
    treatment. The cost is one ``mul_`` over the voltage cube: ~2 GB
    of fp16, ~150 µs on a 2080 Ti.
    """
    if rfi_mask.dtype != torch.bool:
        raise TypeError(
            f"rfi_mask must be bool; got {rfi_mask.dtype}"
        )
    n_ant_m, n_ch_m, n_pol_m = rfi_mask.shape
    n_ch_v, _ntp_v, n_pol_v, _np_v, n_ant_v = real_v.shape
    if (n_ant_m, n_ch_m, n_pol_m) != (n_ant_v, n_ch_v, n_pol_v):
        raise ValueError(
            f"rfi_mask shape {tuple(rfi_mask.shape)} != voltage layout "
            f"(NANTS, NCHAN, NPOL) = ({n_ant_v}, {n_ch_v}, {n_pol_v})"
        )

    # Mask reshape → (NCHAN, 1, NPOL, 1, NANTS) — broadcast against
    # voltage GEMM layout (NCHAN, NTIMES_PER_PACKET, NPOL,
    # NPACKETS_PER_BLOCK, NANTS).
    mask_bcast = (
        rfi_mask.permute(1, 2, 0)                                        # (NCHAN, NPOL, NANTS)
        .unsqueeze(1).unsqueeze(3)                                       # (NCHAN, 1, NPOL, 1, NANTS)
        .to(real_v.device)
    )
    keep = (~mask_bcast).to(real_v.dtype)                                # 0/1 fp{16,32}
    real_v.mul_(keep)
    imag_v.mul_(keep)
    return real_v, imag_v


# ---------------------------------------------------------------------------
# Per-block integration context (held across the run loop)
# ---------------------------------------------------------------------------


@dataclass
class FastIntegrationConfig:
    """All knobs for one fast-corr integration run. Held by reference;
    mutate fields between calls only when not in flight.

    Args:
        chgroup: corr-node chgroup index (0-15).
        obs_dec_rad: observing-source declination (rad). Drives F21
            cal-loader and the gridder's pattern lookup.
        n_grid: side length of the gridded cube. Default 256.
        kernel_support: gridding kernel half-width (cells). Default
            1 (3x3 pillbox).
        t_int_fast_native: fast-corr integration depth in NATIVE
            samples per fast-vis tile. Default 8 (262.144 µs); burst
            test override is 32 (1048.576 µs / 4× cadence).
        cal_path / cal_mode / cal_pol_swap: F21 cal options.
        flagants_path: optional flagants.dat path.
        rfi_enabled: kill-switch for the whole RFI flagger stage
            (default True). Useful for synth-data tests where the RFI
            flagger would mistake injected signals for narrowband CW.
        static_sky_alpha / static_sky_warmup_cubes / static_sky_disabled:
            EMA controls. ``static_sky_disabled=True`` is a kill-
            switch (useful for tests + the chunk-5 0319 continuum
            bench where the brightest source IS the static sky).
        rfi_mask_voltage_zero_fill: route RFI mask through the
            voltage zero-fill (default True). Set to False for tests
            that want to verify the RFI flagger fired without
            actually altering the GEMM result.
    """

    chgroup: int
    obs_dec_rad: float
    n_grid: int = 256
    kernel_support: int = 1
    cell_lambda_mode: str = "common"
    """F28: how :func:`build_pattern` picks the per-cell (u, v) λ-extent.

    * ``"common"`` (F28 default): all chgroups share a single
      ``cell_lambda`` derived from the top-of-band reference frequency
      via :func:`compute_top_of_band_cell_lambda`. A fixed (l, m)
      source then lands at the SAME image pixel in every chgroup, so
      per-chgroup gridded image cubes are stackable by the
      fine-dedisperser/imager without resampling. Top of band is
      critically sampled; lower-frequency chgroups are oversampled
      (their ``n_filled`` shrinks).
    * ``"per_chgroup"`` (legacy pre-F28): each chgroup auto-fits its
      own ``cell_lambda`` from this chgroup's longest baseline-in-λ.
      The chunk-6 burst bench observes this as a 5-pixel column drift
      across the 16 chgroups — the artefact F28 fixes.

    Test code that needs the legacy path bit-identically (e.g.
    grid-pixel parity asserts pinned against the pre-F28 outputs)
    should pass ``"per_chgroup"`` explicitly. New benches and
    production should use ``"common"`` and stack chgroup cubes
    pixel-for-pixel."""

    t_int_fast_native: int = T_INT_FAST_NATIVE
    cal_path: Path | None = None
    cal_mode: str = CalMode.PHASE_ONLY
    cal_pol_swap: bool = False
    flagants_path: Path | None = None
    rfi_enabled: bool = True
    rfi_mask_voltage_zero_fill: bool = True
    static_sky_alpha: float = 0.001
    static_sky_warmup_cubes: int = 8
    static_sky_disabled: bool = False
    dm_plan_path: Path | None = None
    """Optional path to a DMPlan ``.npz`` (canonical or chunk-3b slim
    schema). When set, the orchestrator uses the F25 production
    multi-DM-trial path: stage-1 vis-domain shifts → per-trial
    gridder → per-trial static-sky → per-trial stage-2 FIFO →
    transport. When unset (default), the orchestrator uses the
    legacy single-DM path with the post-grid ``CoarseDMStage`` Protocol
    plug-in (chunk 4 behaviour preserved for backward compatibility
    + single-DM benches such as chunk 5 / chunk 6)."""

    dm_indices_subset: tuple[int, ...] | None = None
    """Optional subset of the plan's DM trials to evaluate. Useful
    for the chunk-6 single-DM burst replay (``--single-dm`` is
    equivalent to ``dm_indices_subset=(<burst_dm_idx>,)``) and for
    chunk-9 throughput benches that evaluate a sparse subset of
    trials. ``None`` means all ``plan.n_coarse`` trials."""

    n_fv_chunk: int | None = None
    """F31b: per-block streaming chunk size for the kernel + Stokes-I
    pipeline. When set, ``process_block`` slices the voltage block
    into ``n_fv_chunk`` fast-vis-tile slabs, runs
    ``FastCorrKernel.compute_split`` on each slab, immediately
    pol-sums to Stokes I, and writes the slab into the
    pre-allocated ``(n_fv_total, NBASE, NCHAN)`` Stokes-I cube. This
    bounds the peak transient cfp32 ``(n_fv_slab, NBASE, NCHAN, NPOL)``
    intermediate to ~MB instead of the ~14 GB full-block size at
    ``t_int_fast_native=8`` on a 2080Ti — required by F31 for
    production fit on the 11 GB production GPU.

    None ⇒ auto-pick the largest power-of-two slab whose cfp32
    output ``vis_2pol`` slab tensor stays under
    :data:`_F31B_CHUNK_TARGET_BYTES` (= 256 MB). Cap at
    ``n_fv_total`` for the block. The Stokes-I cube itself
    (``vis_stokes_i``) is still a single allocation; F33's
    8-channel pre-dedispersion sum reduces THAT to ~MB."""

    sliding_window: bool = False
    """F34: 2-block sliding-window stage-1 dedispersion. When True,
    :class:`Stage1MultiDMCoarseDM` keeps a ``K = 2`` ring buffer of
    the previous block's ``vis_stokes_i`` and joins it with the
    current block before applying stage-1 shifts; the dedispersed
    output corresponds to the PREVIOUS block (now fully resolved
    against any pulse whose lower-frequency tail crosses into the
    current block).

    Required at the M3 production op-point: at DM = 3000 pc/cc and
    ``t_int_fast_native = 8`` the max intra-chgroup delay reaches
    ~480 fast-vis bins, comparable to the ~512-bin block size, so
    cross-block pulses are otherwise truncated. K = 2 ring depth
    is sufficient (DM = 3000 ceiling per the M3 production review).

    Cost: one-block (~134 ms) latency, +1 vis_stokes_i clone in
    GPU memory (~900 MB at chan_sum_factor=8), and one extra
    stage-1 + grid pass per block (since the join is 2× the
    block size). The latency is acceptable for the search; the
    memory + compute fit on the 11 GB 2080Ti budget."""

    chan_sum_factor: int = 1
    """F33: number of fine channels collapsed into one effective
    channel before dedispersion / gridding. ``1`` (default) keeps
    the legacy per-fine-channel pipeline (NCHAN = 384 per chgroup).
    ``8`` (M3 production op-point per the F33 review) sums each
    8-channel block of ``vis_stokes_i`` into one channel BEFORE
    coarse-DM dedispersion, reducing the post-Stokes-I cfp32 cube
    from ~7 GB → ~900 MB on a 2080Ti at ``t_int_fast_native = 8``.
    The DMPlan is rebuilt against the summed-channel band-CENTER
    frequencies (see :meth:`DMPlan.from_summed_canonical`), and the
    sparsity pattern + gridder use the same band-CENTER frequencies
    (folded into ``pattern_id``).

    Constraint: must divide :data:`NCHAN_PER_CHGROUP` (= 384). The
    DM smearing inside one summed channel at DM = 3000 pc/cc,
    ν = 1.31 GHz is ~ 2.7 ms — well below the search-side fine-DM
    step (per the M3 production review)."""

    dm_chunk_size: int = 2
    """RT Phase 2: number of coarse-DM trials whose stage-1-shifted
    vis is concatenated along the time axis into one
    :meth:`FastVisGridder.compute` call.

    The gridder.compute hot path is atomic-bound (each output cell
    receives ~500 contributions per (t, cell) row), and at
    ``n_fv_per_block=512`` × ``sliding_window=2`` block join the per-
    DM scatter is ~1024 t-rows wide — under-utilising a 2080 Ti's
    ~136 simultaneous warps. Stacking ``dm_chunk_size`` DM trials
    into one ``(dm_chunk_size * t_dedisp, NBASE, NCHAN_eff)``
    scatter widens the (t_row, cell) parallelism dim with no
    extra work, recovering throughput.

    Memory cost per chunk: ``dm_chunk_size * t_dedisp * NBASE *
    NCHAN_eff * 8 bytes`` cfp32 ≈ ``dm_chunk_size * 1.6 GB`` at the
    production op-point. The gridder internally splits cfp32 → 2×
    fp32 (real+imag) for the scatter, doubling the transient peak;
    chunk=2 fits the 11 GB 2080 Ti envelope (joined vis 1.7 GB +
    ring-buffer prev vis 0.9 GB + chunk buf 3.2 GB cfp32 + 2×
    1.6 GB fp32 split ≈ 9 GB), chunk=4 OOMs (~15.5 GB peak).
    ``dm_chunk_size = 1`` recovers the legacy (pre-Phase-2) one-DM-
    per-call path."""


@dataclass
class IntegrationContext:
    """Mutable per-run state: kernel, cal tensors, RFI flagger,
    gridder, static-sky EMA, pluggable stages.

    Constructed once at service startup (after the first fada header
    is read so the device is committed); held for the lifetime of the
    run.
    """

    cfg: FastIntegrationConfig
    device: torch.device
    voltage_dtype: torch.dtype

    kernel: FastCorrKernel
    cal: FastCorrCalTensors | None
    rfi_flagger: RFIFlagger | None
    gridder: FastVisGridder
    static_sky: StaticSkyEMA | None

    coarse_dm: CoarseDMStage = field(default_factory=NoOpCoarseDM)
    stage2_fifo: Stage2FifoStage = field(default_factory=NoOpStage2Fifo)
    transport_tx: TransportTxStage = field(default_factory=NoOpTransportTx)

    multi_dm_coarse_dm: Stage1MultiDMCoarseDM | None = None
    """When set (chunk-9 production path), :func:`process_block` uses
    the F25 multi-DM-trial integration: vis-domain stage-1 shifts +
    per-trial gridder + per-trial static-sky. When ``None`` (chunk-4
    legacy path), the post-grid ``coarse_dm`` Protocol stub is used
    instead."""


@dataclass
class IntegrationOutput:
    """Per-block output of :func:`process_block`.

    Args:
        gridded_minus_sky: ``(n_fast_vis, N_filled)`` complex64 — the
            gridded cube AFTER static-sky subtraction. None if the
            block was dropped (malformed / empty).
        rfi: :class:`FlagBlockResult` from the RFI flagger this
            block, or ``None`` if RFI was disabled.
        n_tx: number of cubes the transport stage actually sent this
            block.
        block_n: 1-indexed block counter (mirrors fada page sequence).
    """

    gridded_minus_sky: torch.Tensor | None
    rfi: FlagBlockResult | None
    n_tx: int
    block_n: int


# ---------------------------------------------------------------------------
# Per-block compute (pure; the orchestrator + tests both call this)
# ---------------------------------------------------------------------------


#: F31b: target peak transient-byte budget for the per-slab
#: ``vis_2pol`` cfp32 tensor returned by ``FastCorrKernel.compute_split``
#: when ``process_block`` streams chunks. 256 MB leaves headroom for
#: the surviving full-block ``vis_stokes_i`` (still cfp32-large at
#: ``NCHAN=384``; F33's 8-channel sum reduces THAT to ~MB).
_F31B_CHUNK_TARGET_BYTES: Final[int] = 256 << 20


def _auto_n_fv_chunk_for_streaming(
    n_fv_total: int, *, nchan: int, nbada_pol: int = 2,
    nbase: int = NBASE, target_bytes: int = _F31B_CHUNK_TARGET_BYTES,
) -> int:
    """F31b: largest power-of-two ``n_fv_chunk`` whose per-slab cfp32
    ``vis_2pol`` ≤ ``target_bytes``. Capped at ``n_fv_total``.

    Per-slab bytes = ``n_fv_chunk * nbase * nchan * nbada_pol * 8``
    (cfp32 = 8 bytes). Round-down to power-of-two so all slabs but the
    last are equal-sized + the matmul tile shape stays GEMM-friendly.
    """
    bytes_per_fv = nbase * nchan * nbada_pol * 8
    if bytes_per_fv == 0:
        return max(1, n_fv_total)
    raw = max(1, target_bytes // bytes_per_fv)
    pow2 = 1
    while pow2 * 2 <= raw:
        pow2 *= 2
    return min(pow2, n_fv_total)


def process_block(
    raw: np.ndarray,
    *,
    ctx: IntegrationContext,
    block_n: int,
) -> IntegrationOutput:
    """Run the chunk-4 pipeline on ONE fada-block-worth of raw bytes.

    This is the entire fast-corr integration graph in one function.
    The PSRDADA loop in :func:`run` calls this per block; tests
    construct a synthetic ``raw`` byte array and call this directly.

    Pipeline (mirrors module docstring step list):

        1. unpack int4 → (real_v, imag_v) GEMM-layout
        2. RFI flag_block(real, imag) → FlagBlockResult (if cfg.rfi_enabled)
        3. Apply RFI mask to voltages (if cfg.rfi_mask_voltage_zero_fill)
        4. apply_cal_split (if ctx.cal is not None)
        5. + 6. **Streamed** per-``n_fv_chunk`` slab: FastCorrKernel.compute_split
              → stokes_i_pol_sum → write into the pre-allocated
              ``(n_fv_total, NBASE, NCHAN)`` Stokes-I cube. F31b
              ensures the cfp32 ``(n_fv_slab, NBASE, NCHAN, NPOL)``
              intermediate is bounded to ~256 MB (vs ~14 GB
              full-block at ``t_int_fast_native=8``) so the pipeline
              fits on the 11 GB 2080Ti production GPU.
        7. FastVisGridder.compute → (n_fv, N_filled) complex64
        8. StaticSkyEMA.apply (if cfg.static_sky_disabled is False)
        9. coarse_dm.dedisperse → (N_DM, n_fv, N_filled)
       10. stage2_fifo.push → list[evictees]
       11. transport_tx.transmit → n_sent

    Returns
    -------
    :class:`IntegrationOutput`
        Per-block diagnostic + output bundle.
    """
    # 1. Unpack
    real_v, imag_v = unpack_int4_split(
        raw, device=ctx.device, out_dtype=ctx.voltage_dtype,
    )

    # 2. RFI flag (before cal — flags should be on raw data + DC pol
    # so that any cal-induced dynamic range shifts can't mask them).
    rfi_result: FlagBlockResult | None = None
    if ctx.rfi_flagger is not None and ctx.cfg.rfi_enabled:
        rfi_result = ctx.rfi_flagger.flag_block(real_v, imag_v)

        # 3. Voltage zero-fill of flagged cells (full-cube; bandpass
        # outliers and SK are on per-cube SCALAR statistics so the
        # mask is constant-in-time across the cube — we apply it to
        # all (NTIMES, NPACKETS) samples uniformly).
        if ctx.cfg.rfi_mask_voltage_zero_fill and rfi_result.mask.any():
            real_v, imag_v = apply_rfi_mask_to_voltages(
                real_v, imag_v, rfi_result.mask,
            )

    # 4. Cal apply with F21 DEC-phase fold
    if ctx.cal is not None:
        real_v_c, imag_v_c = apply_cal_split(
            real_v, imag_v, ctx.cal.cal_real, ctx.cal.cal_imag,
        )
        del real_v, imag_v
        real_v, imag_v = real_v_c, imag_v_c

    # 5. + 6. Streamed kernel + Stokes-I.
    #
    # Two paths:
    #
    # - **RT Phase 3 fused fast path** (default whenever
    #   ``chan_sum_factor > 1`` and ``cfg.n_fv_chunk`` is ``None``):
    #   one ``compute_split(real_v, imag_v, fuse_stokes_i=True)`` call
    #   that returns the full ``(n_fv_total, NBASE, NCHAN_eff)`` cfp32
    #   Stokes-I cube directly. F31a's internal slab chunking still
    #   bounds the fp16 V_real / V_imag footprint, but the F31b OUTER
    #   chunk loop, the cfp32 vis_2pol slab transient, and the separate
    #   ``stokes_i_pol_sum`` + chan-sum pass are all eliminated.
    #
    # - **F31b LEGACY PATH** (``chan_sum_factor == 1`` OR the bench /
    #   test pinned ``cfg.n_fv_chunk``): per-slab kernel + pol-sum +
    #   optional chan-sum into a pre-allocated summed-channel cube.
    #   Kept intact so the existing test fixtures + the M3 single-
    #   chgroup CPU benches that exercise vis_2pol behavior still work.
    ppfv = ctx.cfg.t_int_fast_native // NTIMES_PER_PACKET
    n_packets_in = real_v.shape[3]
    if n_packets_in % ppfv != 0:
        raise ValueError(
            f"process_block: n_packets_in={n_packets_in} not a "
            f"multiple of packets_per_fast_vis={ppfv} "
            f"(t_int_fast_native={ctx.cfg.t_int_fast_native}); the "
            f"voltage block cannot be tiled into integer fast-vis "
            f"slabs."
        )
    n_fv_total = n_packets_in // ppfv
    chan_sum_factor = int(ctx.cfg.chan_sum_factor)
    full_nchan = ctx.kernel.nchan
    if chan_sum_factor > 1 and full_nchan % chan_sum_factor != 0:
        raise ValueError(
            f"chan_sum_factor={chan_sum_factor} does not divide "
            f"nchan={full_nchan}; configure cfg.chan_sum_factor "
            f"so that NCHAN_PER_CHGROUP is divisible."
        )
    nchan_eff = (
        full_nchan // chan_sum_factor if chan_sum_factor > 1 else full_nchan
    )

    use_fused_path = (chan_sum_factor > 1 and ctx.cfg.n_fv_chunk is None)
    if use_fused_path:
        vis_stokes_i = ctx.kernel.compute_split(
            real_v, imag_v,
            fuse_stokes_i=True, chan_sum_factor=chan_sum_factor,
        )
        del real_v, imag_v
    else:
        if ctx.cfg.n_fv_chunk is not None:
            n_fv_chunk = int(ctx.cfg.n_fv_chunk)
            if n_fv_chunk <= 0:
                raise ValueError(
                    f"cfg.n_fv_chunk={n_fv_chunk}, expected > 0"
                )
            n_fv_chunk = min(n_fv_chunk, n_fv_total)
        else:
            n_fv_chunk = _auto_n_fv_chunk_for_streaming(
                n_fv_total, nchan=ctx.kernel.nchan,
                nbada_pol=ctx.kernel.nbada_pol,
            )
        vis_stokes_i = torch.empty(
            (n_fv_total, NBASE, nchan_eff),
            dtype=torch.complex64, device=ctx.device,
        )
        for fv0 in range(0, n_fv_total, n_fv_chunk):
            fv1 = min(fv0 + n_fv_chunk, n_fv_total)
            real_v_slab = real_v[:, :, :, fv0 * ppfv : fv1 * ppfv, :].contiguous()
            imag_v_slab = imag_v[:, :, :, fv0 * ppfv : fv1 * ppfv, :].contiguous()
            vis_2pol_slab = ctx.kernel.compute_split(real_v_slab, imag_v_slab)
            stokes_i_slab = stokes_i_pol_sum(vis_2pol_slab)
            if chan_sum_factor > 1:
                stokes_i_slab = stokes_i_slab.reshape(
                    fv1 - fv0, NBASE, nchan_eff, chan_sum_factor,
                ).sum(dim=-1)
            vis_stokes_i[fv0:fv1] = stokes_i_slab
            del real_v_slab, imag_v_slab, vis_2pol_slab, stokes_i_slab
        del real_v, imag_v

    if ctx.multi_dm_coarse_dm is not None:
        # ----- Chunk-9 / F25 production path: multi-DM-trial vis-domain
        # stage-1 shifts → per-trial grid → per-trial static-sky.
        # Returns (N_DM, T_dedisp, N_filled) complex64.
        dedispersed = ctx.multi_dm_coarse_dm.dedisperse_from_vis(
            vis_stokes_i, block_n=block_n,
        )
        del vis_stokes_i

        # 8. Per-trial static-sky EMA subtraction (collapses the
        # T_dedisp axis → N_filled internally; replicates back to
        # the full (T_dedisp, N_filled) shape).
        if ctx.static_sky is not None and not ctx.cfg.static_sky_disabled:
            n_dm, t_dedisp, n_filled = dedispersed.shape
            for c in range(n_dm):
                dedispersed[c] = ctx.static_sky.apply(dedispersed[c])

        # The "headline" pre-stage2 cube is the multi-DM cube itself.
        # Tests + benches read this from IntegrationOutput.gridded_minus_sky.
        gridded_minus_sky = dedispersed
    else:
        # ----- Chunk-4 legacy path (single-DM, post-grid coarse-DM stub)
        # 7. Gridder (sparse-COO)
        gridded = ctx.gridder.compute(vis_stokes_i)                      # (n_fv, N_filled) complex64
        del vis_stokes_i

        # 8. Static-sky EMA subtraction
        if ctx.static_sky is not None and not ctx.cfg.static_sky_disabled:
            gridded_minus_sky = ctx.static_sky.apply(gridded)
        else:
            gridded_minus_sky = gridded

        # 9. Coarse-DM dedispersion (no-op stub today; chunk 3b's
        # primitive lives in a separate module — chunk 9 introduced
        # ``Stage1MultiDMCoarseDM`` for the production multi-DM
        # path, which is the branch above).
        dedispersed = ctx.coarse_dm.dedisperse(
            gridded_minus_sky, block_n=block_n, chgroup=ctx.cfg.chgroup,
        )

    # 10. Stage-2 FIFO push (no-op stub today)
    cubes_for_tx = ctx.stage2_fifo.push(dedispersed, block_n=block_n)

    # 11. Transport TX (no-op stub today)
    rfi_warmup_flag = bool(rfi_result.warmup) if rfi_result is not None else False
    n_tx = ctx.transport_tx.transmit(
        cubes_for_tx, block_n=block_n, rfi_warming_up=rfi_warmup_flag,
    )

    return IntegrationOutput(
        gridded_minus_sky=gridded_minus_sky,
        rfi=rfi_result,
        n_tx=n_tx,
        block_n=block_n,
    )


# ---------------------------------------------------------------------------
# Construction helpers (also called by tests + benches directly)
# ---------------------------------------------------------------------------


def _key_to_int(key_str: str) -> int:
    if len(key_str) != 4:
        raise ValueError(f"buffer key must be 4 chars, got {key_str!r}")
    return int(f"0x{key_str}", 16)


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_fada_buffer_size(cfg_path: Path) -> int:
    cfg = load(cfg_path)
    return int(cfg["buffers"]["fada"]["bytes_per_block"])


class _StopRequested(Exception):
    """SIGTERM / SIGINT → request main-loop stop."""


def _install_signals(state: dict[str, Any]) -> None:
    def handle(signum, _frame):
        LOG.info("received signal %d, requesting stop", signum)
        state["stop"] = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def _build_cal(
    cfg: FastIntegrationConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FastCorrCalTensors | None:
    """Load + log the F21 cal tensors. Returns None if no cal_path set."""
    if cfg.cal_path is None:
        return None
    out = load_cal_with_dec_phase(
        cfg.cal_path,
        chgroup=cfg.chgroup,
        obs_dec_rad=cfg.obs_dec_rad,
        cal_mode=cfg.cal_mode,
        pol_swap=cfg.cal_pol_swap,
        device=device,
        dtype=dtype,
    )
    LOG.info(
        "cal: %s mode=%s pol_swap=%s n_flagged=%d mag_p50=%.3g mag_p99=%.3g",
        out.info["cal_path"], out.info["cal_mode"], out.info["pol_swap"],
        out.info["n_flagged"],
        out.info.get("cal_mag_p50", float("nan")),
        out.info.get("cal_mag_p99", float("nan")),
    )
    LOG.info(
        "F21 DEC-phase: chgroup=%d obs_dec_deg=%.4f phi_lat_deg=%.3f "
        "(delta_dec_deg=%+.4f)",
        cfg.chgroup, out.info["obs_dec_deg"], PHI_LAT_OVRO_DEG,
        out.info["obs_dec_deg"] - PHI_LAT_OVRO_DEG,
    )
    return out


def _build_gridder(
    cfg: FastIntegrationConfig,
    *,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    is_core_baseline_mask: np.ndarray | None,
    device: torch.device,
) -> tuple[SparsityPattern, FastVisGridder]:
    # F28: resolve cell_lambda per ``cfg.cell_lambda_mode``.
    if cfg.cell_lambda_mode == "common":
        cell_lambda_used = compute_top_of_band_cell_lambda(
            antpos_e, antpos_n,
            n_grid=cfg.n_grid,
            is_core_baseline_mask=is_core_baseline_mask,
        )
    elif cfg.cell_lambda_mode == "per_chgroup":
        cell_lambda_used = None                                       # legacy auto-fit in build_pattern
    else:
        raise ValueError(
            f"cfg.cell_lambda_mode={cfg.cell_lambda_mode!r}; expected "
            f"'common' (F28 default) or 'per_chgroup' (legacy)."
        )
    pattern = build_pattern(
        antpos_e, antpos_n,
        chgroup=cfg.chgroup,
        dec_deg=math.degrees(cfg.obs_dec_rad),
        n_grid=cfg.n_grid,
        kernel_support=cfg.kernel_support,
        chan_sum_factor=cfg.chan_sum_factor,
        cell_lambda=cell_lambda_used,
        is_core_baseline_mask=is_core_baseline_mask,
    )
    LOG.info(
        "sparsity pattern: chgroup=%d obs_dec_deg=%.4f n_grid=%d "
        "kernel_support=%d chan_sum_factor=%d cell_lambda_mode=%s "
        "cell_lambda=%.4g → n_filled=%d (id=0x%016x)",
        cfg.chgroup, math.degrees(cfg.obs_dec_rad), cfg.n_grid,
        cfg.kernel_support, cfg.chan_sum_factor,
        cfg.cell_lambda_mode, float(pattern.cell_lambda),
        pattern.n_filled, int(pattern.pattern_id),
    )
    gridder = FastVisGridder.from_pattern(
        pattern, antpos_e, antpos_n,
        is_core_baseline_mask=is_core_baseline_mask,
        device=device,
    )
    return pattern, gridder


def build_context(
    cfg: FastIntegrationConfig,
    *,
    device: torch.device,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    is_core_baseline_mask: np.ndarray | None = None,
    coarse_dm: CoarseDMStage | None = None,
    stage2_fifo: Stage2FifoStage | None = None,
    transport_tx: TransportTxStage | None = None,
    dm_plan: DMPlan | None = None,
) -> IntegrationContext:
    """One-shot context builder. Tests call this directly with synthetic
    antpos arrays; the service shell calls this after reading the fada
    header to commit the device.

    The ``coarse_dm`` / ``stage2_fifo`` / ``transport_tx`` parameters
    default to no-op stubs (chunk 4 placeholder); chunk 3b / chunk 8
    will pass real implementations in.
    """
    voltage_dtype: torch.dtype = (
        torch.float32 if (
            cfg.cal_path is not None and cfg.cal_mode == CalMode.FULL
        ) else torch.float16
    )

    kernel = FastCorrKernel(
        device=device,
        t_int_fast_native=cfg.t_int_fast_native,
    )
    LOG.info(
        "FastCorrKernel ready: t_int_fast_native=%d (%.3f µs cadence) "
        "→ n_fast_vis_per_block=%d",
        cfg.t_int_fast_native,
        cfg.t_int_fast_native * NATIVE_SAMPLE_US,
        kernel.n_fast_vis_per_full_block,
    )

    cal = _build_cal(cfg, device=device, dtype=voltage_dtype)

    rfi_flagger: RFIFlagger | None = None
    if cfg.rfi_enabled:
        rfi_flagger = RFIFlagger(
            flagants_path=cfg.flagants_path,
            device=device,
        )
        LOG.info(
            "RFIFlagger ready: warmup_cubes=%d sk_far=%.3g "
            "bandpass_k=%.2f group_k=%.2f",
            rfi_flagger.warmup_cubes, rfi_flagger._sk_far,
            rfi_flagger._bandpass_k, rfi_flagger._group_k,
        )
    else:
        LOG.info("RFIFlagger DISABLED (cfg.rfi_enabled=False)")

    _pattern, gridder = _build_gridder(
        cfg,
        antpos_e=antpos_e, antpos_n=antpos_n,
        is_core_baseline_mask=is_core_baseline_mask,
        device=device,
    )

    static_sky: StaticSkyEMA | None = None
    if not cfg.static_sky_disabled:
        static_sky = StaticSkyEMA(
            alpha=cfg.static_sky_alpha,
            warmup_cubes=cfg.static_sky_warmup_cubes,
        )
        LOG.info(
            "StaticSkyEMA ready: alpha=%.4g warmup_cubes=%d",
            static_sky.alpha, static_sky.warmup_cubes,
        )
    else:
        LOG.info("StaticSkyEMA DISABLED (cfg.static_sky_disabled=True)")

    # Chunk-9 / F25 production multi-DM path.
    # ``dm_plan`` arg overrides ``cfg.dm_plan_path`` — used by tests +
    # benches that synthesise a custom plan (e.g. chunk-6 single-DM
    # burst, chunk-9 throughput).
    multi_dm: Stage1MultiDMCoarseDM | None = None
    plan: DMPlan | None = dm_plan
    if plan is None and cfg.dm_plan_path is not None:
        # F33-aware NPZ load. Always go via the canonical DmPlan so the
        # summed-channel branch can rebuild delay tables against the
        # band-CENTER frequencies; the chan_sum_factor=1 path is
        # bit-identical to the legacy from_npz route.
        from dsart.common.contracts import DmPlan as _CanonicalDmPlan
        canonical = _CanonicalDmPlan.from_npz(str(cfg.dm_plan_path))
        if cfg.chan_sum_factor == 1:
            plan = DMPlan.from_canonical(canonical)
        else:
            plan = DMPlan.from_summed_canonical(
                canonical, chan_sum_factor=cfg.chan_sum_factor,
            )
    if plan is not None:
        # F33: chan_sum_factor pin — if the caller built a plan
        # externally (e.g. tests passing a synthetic DMPlan), it must
        # match the integration cfg or the gridder pattern + DMPlan
        # delay-table channel grids will silently drift.
        if int(plan.chan_sum_factor) != int(cfg.chan_sum_factor):
            raise ValueError(
                f"DMPlan.chan_sum_factor={plan.chan_sum_factor} does "
                f"not match cfg.chan_sum_factor={cfg.chan_sum_factor}; "
                f"rebuild the plan via DMPlan.from_summed_canonical "
                f"with chan_sum_factor={cfg.chan_sum_factor}."
            )
        # Sanity: t_int_fast_native pin (within fp tolerance for
        # non-integer cadences, which the schema allows).
        plan_native = plan.t_int_fast_native
        if abs(plan_native - cfg.t_int_fast_native) > 1e-6:
            raise ValueError(
                f"DMPlan.t_int_fast_native={plan_native} does not match "
                f"cfg.t_int_fast_native={cfg.t_int_fast_native} — "
                f"rebuild the plan or change the config"
            )
        dm_indices_arr: np.ndarray | None = None
        if cfg.dm_indices_subset is not None:
            dm_indices_arr = np.asarray(
                cfg.dm_indices_subset, dtype=np.int64,
            )
        multi_dm = Stage1MultiDMCoarseDM(
            plan=plan,
            gridder=gridder,
            chgroup=cfg.chgroup,
            dm_indices=dm_indices_arr,
            sliding_window=bool(cfg.sliding_window),
            dm_chunk_size=int(cfg.dm_chunk_size),
        )
        LOG.info(
            "Stage1MultiDMCoarseDM ready: chgroup=%d n_dm=%d t_int_fast_us=%.3f "
            "(plan_n_coarse=%d, dm_subset=%s, sliding_window=%s, dm_chunk_size=%d)",
            cfg.chgroup, multi_dm.n_dm, plan.t_int_fast_us,
            plan.n_coarse,
            "all" if cfg.dm_indices_subset is None else
            str(cfg.dm_indices_subset),
            cfg.sliding_window, multi_dm.dm_chunk_size,
        )

    return IntegrationContext(
        cfg=cfg,
        device=device,
        voltage_dtype=voltage_dtype,
        kernel=kernel,
        cal=cal,
        rfi_flagger=rfi_flagger,
        gridder=gridder,
        static_sky=static_sky,
        coarse_dm=coarse_dm if coarse_dm is not None else NoOpCoarseDM(),
        stage2_fifo=(
            stage2_fifo if stage2_fifo is not None else NoOpStage2Fifo()
        ),
        transport_tx=(
            transport_tx if transport_tx is not None else NoOpTransportTx()
        ),
        multi_dm_coarse_dm=multi_dm,
    )


# ---------------------------------------------------------------------------
# Antpos / core-baseline accessor (reads the same source the gridder
# pattern build uses; centralised here so the service + tests + benches
# don't drift).
# ---------------------------------------------------------------------------


def load_antpos_from_cal_blob(
    cal_path: Path,
    *,
    cal_yaml_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load antenna positions + the core-baseline mask from a cal blob.

    The legacy ``beamformer_weights_*.dat`` blob carries (E, N) antpos
    in its header — the gridder needs the same arrays the cal was
    derived against, so reading both from the same blob guarantees
    the antpos hash on the SparsityPattern matches.

    The CAL YAML (sibling of the .dat) carries ``antenna_order`` —
    the per-fada-slot DSA-110 station numbers. F32 (M3 carryover)
    requires station-number-based core/outrigger discrimination; if
    the yaml is missing this loader falls back to the legacy
    radius-based mask (which is unreliable for DSA-110 — see
    :func:`dsart.grid.sparsity_pattern.core_baseline_mask_from_station_numbers`).

    Parameters
    ----------
    cal_path : Path
        Path to ``beamformer_weights_*.dat``.
    cal_yaml_path : Path, optional
        Path to the sibling cal yaml. If ``None``, looks for any
        ``beamformer_weights_*.yaml`` in the same directory.

    Returns
    -------
    (antpos_e, antpos_n, is_core_baseline_mask) : tuple
        - antpos_e, antpos_n : (NANTS,) float32 arrays
        - is_core_baseline_mask : (NBASE,) bool — True for cross-
          baselines where both antennas have station ≤ 102 (the
          canonical core). False for autos or any baseline involving
          a station 103-116 outrigger.
    """
    from dsart.cal.bf_weights import load_bf_weights
    from dsart.grid.sparsity_pattern import (
        N_CORE_DEFAULT,
        core_baseline_mask_from_antpos,
        core_baseline_mask_from_station_numbers,
    )

    bf = load_bf_weights(cal_path)
    antpos_e = np.asarray(bf.antpos_e, dtype=np.float32)
    antpos_n = np.asarray(bf.antpos_n, dtype=np.float32)

    cal_path = Path(cal_path)
    yaml_path = cal_yaml_path
    if yaml_path is None:
        candidates = sorted(cal_path.parent.glob("beamformer_weights_*.yaml"))
        if candidates:
            yaml_path = candidates[0]

    if yaml_path is not None and yaml_path.is_file():
        import yaml as _yaml
        with open(yaml_path, "r") as f:
            ydoc = _yaml.safe_load(f)
        antenna_order = ydoc["cal_solutions"]["antenna_order"]
        mask = core_baseline_mask_from_station_numbers(antenna_order)
    else:
        # Legacy fallback (radius-based; F32 notes this is unreliable
        # for DSA-110 — outriggers 103-115 overlap in radius with core
        # 99-102). Kept here only for tests that don't have a yaml.
        mask = core_baseline_mask_from_antpos(
            antpos_e, antpos_n, n_core=N_CORE_DEFAULT,
        )
    return (antpos_e, antpos_n, mask)


def _build_core_baseline_mask(
    antpos_e: np.ndarray | None = None,
    antpos_n: np.ndarray | None = None,
    *,
    n_core: int = 82,
) -> np.ndarray:
    """``(NBASE,) bool`` mask: True iff both antennas are core.

    When ``antpos_e`` / ``antpos_n`` are provided, this delegates to
    :func:`dsart.grid.sparsity_pattern.core_baseline_mask_from_antpos`
    to select the ``n_core`` smallest-radius antennas — the correct
    behavior for real DSA-110 antpos (per F27 in
    ``M3_PLAN_FIXES.md``).

    When ``antpos_e`` / ``antpos_n`` are ``None``, falls back to the
    legacy positional definition (ants 0..n_core-1 are core). This is
    only correct for SYNTHETIC antpos arrays where the test author
    placed core ants at the start; real cal-blob antpos has e.g. ant
    48 at r ≈ 1008 m (an outrigger) inside the first 82 indices.
    Existing test files synthesise antpos that way and continue to
    pass; production code paths in :func:`build_context` and
    :func:`load_antpos_from_cal_blob` route through the antpos-based
    path.
    """
    if antpos_e is not None and antpos_n is not None:
        from dsart.grid.sparsity_pattern import core_baseline_mask_from_antpos
        return core_baseline_mask_from_antpos(
            antpos_e, antpos_n, n_core=n_core,
        )
    nbase = NANTS * (NANTS + 1) // 2
    mask = np.zeros(nbase, dtype=bool)
    k = 0
    for a in range(NANTS):
        for b in range(a + 1):
            mask[k] = (a < n_core) and (b < n_core)
            k += 1
    return mask


# ---------------------------------------------------------------------------
# Service shell (PSRDADA fada → in-memory pipeline → on-disk artefacts)
# ---------------------------------------------------------------------------


def _serialise_block(
    output_dir: Path,
    block_n: int,
    *,
    out: IntegrationOutput,
    blocks_output_mode: str,
) -> None:
    """Write per-block artefacts (gridded cube + meta json) to disk.

    Modes:
      * ``full`` — torch.save the entire ``(n_fv, N_filled)`` tensor.
      * ``first_tile_only`` — only fast-vis tile 0.
      * ``none`` — meta json only (gridded cube discarded).
    """
    if out.gridded_minus_sky is None:
        return

    block_dir = output_dir / f"block_{block_n:06d}"
    block_dir.mkdir(exist_ok=True)

    if blocks_output_mode == "first_tile_only":
        out_tensor = out.gridded_minus_sky[0:1].cpu().contiguous()
        torch.save(out_tensor, block_dir / "gridded_minus_sky.pt")
    elif blocks_output_mode == "full":
        out_tensor = out.gridded_minus_sky.cpu().contiguous()
        torch.save(out_tensor, block_dir / "gridded_minus_sky.pt")
    elif blocks_output_mode == "none":
        out_tensor = None
    else:
        raise ValueError(
            f"blocks_output_mode={blocks_output_mode!r} (expected "
            f"full | first_tile_only | none)"
        )

    meta = {
        "block_n": block_n,
        "n_fast_vis_total": int(out.gridded_minus_sky.shape[0]),
        "n_filled": int(out.gridded_minus_sky.shape[1]),
        "n_fast_vis_written": (
            int(out_tensor.shape[0]) if out_tensor is not None else 0
        ),
        "rfi_flag_fraction_total": (
            float(out.rfi.flag_fraction_total) if out.rfi is not None else None
        ),
        "rfi_warmup": (
            bool(out.rfi.warmup) if out.rfi is not None else None
        ),
        "n_tx": out.n_tx,
        "blocks_output_mode": blocks_output_mode,
    }
    (block_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def run(
    fada_key: int,
    output_dir: Path,
    device: torch.device,
    cfg: FastIntegrationConfig,
    *,
    max_blocks: int | None = None,
    expected_fada_bytes: int = FADA_BYTES_PER_BLOCK,
    blocks_output_mode: str = "first_tile_only",
    coarse_dm: CoarseDMStage | None = None,
    stage2_fifo: Stage2FifoStage | None = None,
    transport_tx: TransportTxStage | None = None,
) -> dict[str, Any]:
    """Connect to PSRDADA fada, run the integration pipeline per block,
    optionally serialise per-block artefacts.

    Returns a summary dict including:
        {n_blocks_in, n_blocks_processed, n_dropped, elapsed_s,
         ms_per_block_p50, ms_per_block_p99, output_dir,
         t_int_fast_native, n_fast_vis_per_block,
         rfi_flag_fraction_p50, rfi_flag_fraction_p99,
         n_tx_total, n_grid, n_filled}
    """
    from psrdada import Reader

    if blocks_output_mode not in ("full", "first_tile_only", "none"):
        raise ValueError(
            f"blocks_output_mode must be full|first_tile_only|none, "
            f"got {blocks_output_mode!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {"stop": False}
    _install_signals(state)

    LOG.info(
        "connecting fada=0x%x device=%s chgroup=%d obs_dec_deg=%.4f",
        fada_key, device, cfg.chgroup, math.degrees(cfg.obs_dec_rad),
    )
    LOG.info("output: %s mode=%s", output_dir, blocks_output_mode)
    reader = Reader(fada_key)

    n_in = 0
    n_processed = 0
    n_drop = 0
    n_tx_total = 0

    try:
        fada_header = reader.getHeader()
        LOG.info("fada header: %d keys (UTC_START=%s)",
                 len(fada_header), fada_header.get("UTC_START", "?"))

        if cfg.cal_path is None:
            raise RuntimeError(
                "corr_fast_integration.run() requires --apply-cal: the "
                "antpos arrays + core-baseline mask are sourced from "
                "the cal blob (load_antpos_from_cal_blob) so the "
                "gridder pattern matches the cal."
            )
        antpos_e, antpos_n, core_mask = load_antpos_from_cal_blob(cfg.cal_path)
        ctx = build_context(
            cfg, device=device,
            antpos_e=antpos_e, antpos_n=antpos_n,
            is_core_baseline_mask=core_mask,
            coarse_dm=coarse_dm,
            stage2_fifo=stage2_fifo,
            transport_tx=transport_tx,
        )

        per_block_ms: list[float] = []
        per_block_flag_frac: list[float] = []
        t_start = time.monotonic()

        while not state["stop"]:
            try:
                page = reader.getNextPage()
            except StopIteration:
                LOG.info("fada reader StopIteration (EOD)")
                break
            if reader.isEndOfData:
                LOG.info("fada EOD flag set; draining final block")
            n_in += 1

            t_block_start = time.monotonic()

            page_arr = np.asarray(page)
            if page_arr.nbytes != expected_fada_bytes:
                LOG.error(
                    "fada block #%d wrong size: got=%d expected=%d; skipping",
                    n_in, page_arr.nbytes, expected_fada_bytes,
                )
                reader.markCleared()
                n_drop += 1
                if max_blocks is not None and n_in >= max_blocks:
                    break
                continue

            out = process_block(page_arr, ctx=ctx, block_n=n_in)
            n_tx_total += out.n_tx

            if blocks_output_mode != "none":
                _serialise_block(
                    output_dir, n_in,
                    out=out, blocks_output_mode=blocks_output_mode,
                )

            n_processed += 1
            if out.rfi is not None:
                per_block_flag_frac.append(float(out.rfi.flag_fraction_total))
            reader.markCleared()
            del out

            t_block_end = time.monotonic()
            per_block_ms.append((t_block_end - t_block_start) * 1000.0)

            if n_in % 16 == 0:
                LOG.info(
                    "processed n_in=%d n_processed=%d n_drop=%d n_tx=%d "
                    "last_block=%.1fms",
                    n_in, n_processed, n_drop, n_tx_total, per_block_ms[-1],
                )

            if max_blocks is not None and n_in >= max_blocks:
                LOG.info("hit --max-blocks=%d; stopping", max_blocks)
                break
            if reader.isEndOfData:
                LOG.info("fada EOD; loop done")
                break

        elapsed = time.monotonic() - t_start
        ms_p50 = float(np.median(per_block_ms)) if per_block_ms else float("nan")
        ms_p99 = float(np.percentile(per_block_ms, 99)) if per_block_ms else float("nan")
        ff_p50 = float(np.median(per_block_flag_frac)) if per_block_flag_frac else float("nan")
        ff_p99 = float(np.percentile(per_block_flag_frac, 99)) if per_block_flag_frac else float("nan")
        LOG.info(
            "summary: n_in=%d n_processed=%d n_drop=%d n_tx=%d elapsed=%.1fs "
            "p50=%.1fms p99=%.1fms ff_p50=%.3f ff_p99=%.3f",
            n_in, n_processed, n_drop, n_tx_total, elapsed,
            ms_p50, ms_p99, ff_p50, ff_p99,
        )
        return {
            "n_blocks_in": n_in,
            "n_blocks_processed": n_processed,
            "n_dropped": n_drop,
            "n_tx_total": n_tx_total,
            "elapsed_s": elapsed,
            "ms_per_block_p50": ms_p50,
            "ms_per_block_p99": ms_p99,
            "rfi_flag_fraction_p50": ff_p50,
            "rfi_flag_fraction_p99": ff_p99,
            "output_dir": str(output_dir),
            "t_int_fast_native": cfg.t_int_fast_native,
            "n_fast_vis_per_block": ctx.kernel.n_fast_vis_per_full_block,
            "n_grid": cfg.n_grid,
            "n_filled": int(ctx.gridder.pattern.n_filled),
        }
    finally:
        try:
            reader.disconnect()
        except Exception:
            LOG.exception("reader.disconnect failed (non-fatal)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fada-key", default="fada",
                   help="4-char fada buffer key (default: fada). "
                        "PARALLEL_AGENTS.md §4.1 maps via "
                        "$DSART_BUFFER_KEY_PREFIX so e.g. m3 → 'fa3a'.")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cuda:N / cpu (default: auto). "
                        "PARALLEL_AGENTS.md §4.2 pins M3 to GPU 0 on h01.")
    p.add_argument("--config", type=Path,
                   default=Path("configs/config_corr.yaml"),
                   help="path to config_corr.yaml for fada buffer-size validation")
    p.add_argument("--max-blocks", type=int, default=1,
                   help="stop after N fada blocks (default: 1 for "
                        "smoke runs; production: omit for unlimited)")
    p.add_argument("--t-int-fast-native", type=int, default=T_INT_FAST_NATIVE,
                   help="fast-corr integration depth in NATIVE samples per "
                        "fast-vis tile. Default: %d (= %d µs cadence). "
                        "Burst-test override: 32 (= 1048.576 µs, 4× cadence)."
                        % (T_INT_FAST_NATIVE, int(T_INT_FAST_NATIVE * NATIVE_SAMPLE_US)))
    p.add_argument("--obs-dec-deg", type=float, required=True,
                   help="observing source declination (deg) — F21 + gridder "
                        "pattern lookup")
    p.add_argument("--chgroup", type=int, default=0,
                   help="corr-node chgroup index 0..15")
    p.add_argument("--apply-cal", type=Path, default=None,
                   help="legacy beamformer_weights_*.dat blob path "
                        "(F21 fold lives in cal_loader)")
    p.add_argument("--cal-mode", default=CalMode.PHASE_ONLY,
                   choices=(CalMode.PHASE_ONLY, CalMode.FULL),
                   help="phase_only = divide by |G| (default; fp16 stable). "
                        "full = preserve gain magnitude (routes through fp32).")
    p.add_argument("--cal-pol-swap", action="store_true",
                   help="swap cal pol axis")
    p.add_argument("--flagants", type=Path, default=None,
                   help="path to legacy flagants.dat (default: no static flagants)")
    p.add_argument("--rfi-disabled", action="store_true",
                   help="disable the RFI flagger entirely (e.g. for "
                        "synth-data tests with injected narrowband signals "
                        "that would trigger the SK detector)")
    p.add_argument("--n-grid", type=int, default=256,
                   help="grid side length (default: 256)")
    p.add_argument("--kernel-support", type=int, default=1,
                   help="gridding kernel half-width in cells (default: 1 → 3x3)")
    p.add_argument("--static-sky-alpha", type=float, default=0.001,
                   help="static-sky EMA smoothing factor (default: 0.001 → "
                        "~700-cube half-life)")
    p.add_argument("--static-sky-warmup-cubes", type=int, default=8,
                   help="cubes during which to BUILD the EMA but not subtract")
    p.add_argument("--static-sky-disabled", action="store_true",
                   help="disable static-sky subtraction entirely (useful for "
                        "the 0319 continuum bench where the brightest source "
                        "IS the static sky)")
    p.add_argument("--n-fv-chunk", type=int, default=None,
                   help="F31b: per-block streaming chunk size for the "
                        "kernel + Stokes-I pipeline. Default: auto "
                        "(picks largest pow-2 slab whose cfp32 vis_2pol "
                        "stays under 256 MB; required for the 11 GB "
                        "2080Ti production GPU at t_int_fast_native=8). "
                        "Pass an explicit value to override (e.g. 8 for "
                        "deterministic memory profiling).")
    p.add_argument("--chan-sum-factor", type=int, default=1,
                   help="F33: collapse this many adjacent fine channels "
                        "into one effective channel before dedispersion. "
                        "Default: 1 (per-fine-channel pipeline; legacy). "
                        "Production op-point: 8 (NCHAN 384 → 48; reduces "
                        "post-Stokes-I cfp32 cube ~7 GB → ~900 MB). Must "
                        "divide NCHAN_PER_CHGROUP (= 384). The DMPlan and "
                        "gridder pattern are rebuilt against summed-"
                        "channel band-CENTER frequencies.")
    p.add_argument("--sliding-window", action="store_true",
                   help="F34: 2-block sliding-window stage-1 dedispersion. "
                        "Keeps a K=2 ring buffer of vis_stokes_i so that "
                        "pulses crossing block boundaries (intra-chgroup "
                        "delay ~480 fast-vis bins at DM=3000 pc/cc, "
                        "t_int_fast_native=8) are fully resolved. Adds one "
                        "block (~134 ms) of latency.")
    p.add_argument("--cell-lambda-mode", default="common",
                   choices=("common", "per_chgroup"),
                   help="F28: per-cell (u, v) λ-extent selection. "
                        "'common' (default; F28): a single cell_lambda "
                        "from the top-of-band frequency is shared across "
                        "all chgroups so a fixed (l, m) source lands at "
                        "the same image pixel in every chgroup, enabling "
                        "pixel-wise stacking by the fine-dedisperser/imager. "
                        "'per_chgroup' (legacy pre-F28): each chgroup "
                        "auto-fits its own cell scale; the burst bench "
                        "exhibits a 5-pixel column drift across the band "
                        "in this mode.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/tmp/dsart-fast-grid"),
                   help="per-block output dir (default: %(default)s)")
    p.add_argument("--blocks-output-mode", default="first_tile_only",
                   choices=("full", "first_tile_only", "none"),
                   help="full: write the entire (n_fv, N_filled) tensor. "
                        "first_tile_only: write only fast-vis tile 0. "
                        "none: meta json only.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.config.exists():
        fada_b = _load_fada_buffer_size(args.config)
        if fada_b != FADA_BYTES_PER_BLOCK:
            LOG.error("config %s fada bytes_per_block=%d != %d",
                      args.config, fada_b, FADA_BYTES_PER_BLOCK)
            return 2
    else:
        LOG.warning("config %s not found; skipping buffer-size validation", args.config)

    fada_int = _key_to_int(args.fada_key)
    device = _pick_device(args.device)

    if args.apply_cal is not None and not args.apply_cal.is_file():
        LOG.error("--apply-cal path %s not found", args.apply_cal)
        return 2

    cfg = FastIntegrationConfig(
        chgroup=args.chgroup,
        obs_dec_rad=math.radians(args.obs_dec_deg),
        n_grid=args.n_grid,
        kernel_support=args.kernel_support,
        t_int_fast_native=args.t_int_fast_native,
        cal_path=args.apply_cal,
        cal_mode=args.cal_mode,
        cal_pol_swap=args.cal_pol_swap,
        flagants_path=args.flagants,
        rfi_enabled=not args.rfi_disabled,
        static_sky_alpha=args.static_sky_alpha,
        static_sky_warmup_cubes=args.static_sky_warmup_cubes,
        static_sky_disabled=args.static_sky_disabled,
        n_fv_chunk=args.n_fv_chunk,
        chan_sum_factor=args.chan_sum_factor,
        sliding_window=args.sliding_window,
        cell_lambda_mode=args.cell_lambda_mode,
    )

    try:
        run(
            fada_int,
            args.output_dir,
            device,
            cfg,
            max_blocks=args.max_blocks,
            blocks_output_mode=args.blocks_output_mode,
        )
    except _StopRequested:
        LOG.info("clean stop")
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt; clean stop")
    except Exception:
        LOG.exception("fatal error in run()")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
