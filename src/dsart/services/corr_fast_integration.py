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
  4b. ``OnlineInjector.apply_block`` — voltage-domain injection AFTER
      RFI + cal (M7.4 Phase 8; no-op unless an injector is configured)
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
import contextlib
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional, Protocol

import numpy as np
import torch

from dsart.cal.cal_loader import (
    CalMode,
    FastCorrCalTensors,
    load_cal_with_dec_phase,
)
from dsart.common.config_loader import load
from dsart.common.constants import (
    BLOCK_SAMPLES_NATIVE,
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
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    apply_cal_split,
    unpack_int4_split,
)
from dsart.inject.online import InjectionConfig, OnlineInjector
from dsart.inject.runtime_watch import RuntimeInjectWatch
from dsart.coarse_dm.dm_plan import DMPlan
from dsart.coarse_dm.stage1 import (
    apply_stage1_shifts,
    max_t_dedisp_for_plan,
)
from dsart.coarse_dm.stage2_fifo import Stage2FIFO
from dsart.common.constants import COARSE_DM_FIFO_DEPTH_DEFAULT
from dsart.transport.frame import (
    DTYPE_CFP16,
    DTYPE_CINT8,
)
from dsart.transport.tx import TransportTx
from dsart.transport.async_tx import (
    AsyncTransportTx,
    AsyncTransportTxConfig,
)
from dsart.transport.tx_ring import CubeShmRingDims


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


class _Stage2FIFOAdapter:
    """Protocol-compatible wrapper around :class:`Stage2FIFO`.

    The orchestrator calls ``stage2_fifo.push(dedispersed, *, block_n)
    -> list[Tensor]`` per the :class:`Stage2FifoStage` Protocol; the
    real :class:`Stage2FIFO` has ``push(cube) -> Tensor | None`` and a
    separate ``push_for_protocol`` adapter method. This thin wrapper
    routes through ``push_for_protocol`` so the real FIFO drops in
    with no protocol churn.

    Used when the operator passes ``--transport-tx-host``: the
    Stage-2 FIFO becomes a real K-deep cube ring (M7.2 overlap path)
    instead of the M3 :class:`NoOpStage2Fifo` identity adapter.
    """

    __slots__ = ("fifo",)

    def __init__(self, depth: int) -> None:
        self.fifo = Stage2FIFO(depth=depth)

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        return self.fifo.push_for_protocol(dedispersed, block_n=block_n)


class _Stage2InterChgroupShiftAdapter:
    """Lazy wrapper around :class:`Stage2InterChgroupShiftFifo`.

    Option A wire-in: applies per-(coarse-DM) inter-chgroup time
    alignment at the corr_fast TX boundary, so the search side can
    revert to ``compute_time_shift_search(include_coarse_offset=False)``
    and the search-side rx-ring buffer shrinks back to the
    stage-3-only minimum (~16x smaller at the M7.4 op point).

    The inner :class:`Stage2InterChgroupShiftFifo` requires
    ``t_dedisp`` at construction, but that value is only known once
    the first dedispersed cube is produced (it depends on the corr-
    fast cadence + block size + multi-DM stage-1 history). This
    adapter constructs the inner FIFO on the FIRST push, using the
    incoming cube's ``shape[1]`` as the ground truth.
    """

    __slots__ = ("_chgroup", "_coarse_dm", "_t_int_corr_us", "_inner")

    def __init__(
        self,
        *,
        chgroup: int,
        coarse_dm_pc_cm3: np.ndarray,
        t_int_corr_us: float,
    ) -> None:
        self._chgroup = int(chgroup)
        self._coarse_dm = np.ascontiguousarray(
            coarse_dm_pc_cm3, dtype=np.float64
        )
        self._t_int_corr_us = float(t_int_corr_us)
        self._inner: Optional["Stage2InterChgroupShiftFifo"] = None

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        if self._inner is None:
            from dsart.coarse_dm.stage2_chgroup_alignment import (
                Stage2InterChgroupShiftFifo,
            )
            if dedispersed.ndim != 3:
                raise ValueError(
                    "Stage2InterChgroupShiftFifo expects (N_DM, T, N_filled); "
                    f"got ndim={dedispersed.ndim}"
                )
            n_dm_in = int(dedispersed.shape[0])
            t_dedisp = int(dedispersed.shape[1])
            if n_dm_in != self._coarse_dm.size:
                raise ValueError(
                    f"Stage2InterChgroupShiftFifo n_dm mismatch: "
                    f"cube.shape[0]={n_dm_in} != "
                    f"len(coarse_dm)={self._coarse_dm.size}"
                )
            self._inner = Stage2InterChgroupShiftFifo(
                chgroup=self._chgroup,
                coarse_dm_pc_cm3=self._coarse_dm,
                t_dedisp=t_dedisp,
                t_int_corr_us=self._t_int_corr_us,
            )
            LOG.info(
                "Stage2InterChgroupShiftFifo built: chgroup=%d n_dm=%d "
                "t_dedisp=%d max_shift_samples=%d "
                "max_ring_depth_cubes=%d",
                self._chgroup,
                self._inner.n_dm,
                self._inner.t_dedisp,
                int(self._inner.shifts_samples.max()),
                self._inner.max_ring_depth_in_cubes,
            )
        return self._inner.push(dedispersed, block_n=block_n)


class _TransportTxAdapter:
    """Protocol-compatible wrapper around :class:`TransportTx`.

    Bridges two signature gaps so the M7.2 production TX wires into
    the chunk-4 orchestrator without modifying the call site:

    1. :class:`TransportTx.transmit` supports an optional
       ``specnum`` kwarg that the M4a prod-frame path REQUIRES
       (raises ``NotImplementedError`` otherwise). The chunk-4
       :class:`TransportTxStage` Protocol does not pass ``specnum``.
       This adapter supplies ``specnum=block_n`` automatically when
       the underlying TX uses ``use_prod_frame=True``.
    2. Decouples the orchestrator from import-time knowledge of the
       transport stack (only the adapter holds the concrete TX
       instance; the orchestrator only sees the Protocol surface).

    Production wiring (M7.2.8 corner-turn) will source ``specnum``
    from the SNAP F-engine packet counter at the start of each block
    (plan §4.3 line 1421); for the M7.2 Phase A corr-side benchmark,
    ``block_n`` is a monotone proxy good enough to satisfy the
    receiver's seq-gap accounting (which the loopback drain ignores).
    """

    __slots__ = ("tx", "_pass_specnum")

    def __init__(self, tx: TransportTx) -> None:
        self.tx = tx
        self._pass_specnum: bool = bool(tx.use_prod_frame)

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        if self._pass_specnum:
            return self.tx.transmit(
                cubes_for_tx,
                block_n=block_n,
                rfi_warming_up=rfi_warming_up,
                specnum=int(block_n),
            )
        return self.tx.transmit(
            cubes_for_tx,
            block_n=block_n,
            rfi_warming_up=rfi_warming_up,
        )


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


class _AsyncTransportTxAdapter:
    """Protocol-compatible wrapper around :class:`AsyncTransportTx`.

    M7.2 production path: off-loads the encode + ``sendto`` work to
    worker subprocesses so the main GPU-pipeline thread is never
    blocked on TX. The hot-path :meth:`transmit` only does:

      1. D2H of the cube (once; ~10 ms PCIe at the N=8 op-point).
      2. ``numpy.copyto`` per worker into the worker's shm slot
         (~1 ms each at production dims).
      3. ``mp.Queue.put`` per worker to signal the slot is ready.

    Total main-thread cost: ~10 ms D2H + ~4 × 1 ms slot copies + ~4 ×
    Queue.put overhead. The remaining ~64 ms of cint8 quantisation +
    ProdFrame packing + ``sendto`` syscalls runs concurrently in the
    worker subprocesses, overlapping with the *next* block's GPU
    compute.

    Required by the user's "no shortcuts" production-only directive
    for M7.2: the corr-side egress must hit the wire at the production
    rate while the corr-side block budget stays at its M7.1 ceiling
    (~110 ms p50 at the N=8 prod settings).

    Args:
        async_tx: a constructed :class:`AsyncTransportTx`. Owns its
            worker subprocesses + shm rings for the lifetime of the
            ``corr_fast`` service.
    """

    __slots__ = ("async_tx",)

    def __init__(self, async_tx: AsyncTransportTx) -> None:
        self.async_tx = async_tx

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        return self.async_tx.transmit(
            cubes_for_tx,
            block_n=block_n,
            rfi_warming_up=rfi_warming_up,
            specnum=int(block_n),
        )

    def close(self) -> None:
        self.async_tx.close()


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

    def _dedisperse_one_window_phase9_cpu(
        self, vis_stokes_i: torch.Tensor,
    ) -> torch.Tensor:
        """Phase-9 PyTorch fallback for the dedisp inner kernel.

        Used on CPU where the Triton fused kernel can't run. Bit-
        equivalent to the GPU Triton path modulo fp32 reduction-order
        rounding. Mirrors the pre-Phase-10 implementation: coalesced
        gather (T, B, C) → (T, C, B), single complex64 ``index_add_``
        scatter, dm-chunked.
        """
        n_fv = int(vis_stokes_i.shape[0])
        t_dedisp = self.t_dedisp_for(n_fv)
        n_filled = int(self.gridder.pattern.n_filled)
        nb = int(vis_stokes_i.shape[1])
        nch = int(vis_stokes_i.shape[2])
        device = vis_stokes_i.device

        vis_T = vis_stokes_i.permute(0, 2, 1).contiguous()
        bin_shifts_full = self.plan.delay_bins_per_chgroup(self.chgroup)
        bin_shifts = bin_shifts_full[:nch, self._dm_idx_iter]
        bin_shifts_dev = torch.as_tensor(
            bin_shifts, dtype=torch.int64, device=device,
        )
        t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)
        cim_bc = self.gridder.cell_index_map.reshape(nb, nch)
        cim_cb = cim_bc.t().contiguous().reshape(-1)
        out = torch.empty(
            (self.n_dm, t_dedisp, n_filled),
            dtype=torch.complex64, device=device,
        )
        dm_chunk = max(1, int(getattr(self, "dm_chunk_size", 2)))
        dm_chunk = min(dm_chunk, self.n_dm)
        for c0 in range(0, self.n_dm, dm_chunk):
            c1 = min(c0 + dm_chunk, self.n_dm)
            chunk = c1 - c0
            t_chunk = chunk * t_dedisp
            bs_chunk = bin_shifts_dev[:, c0:c1]
            t_idx_2d = (
                bs_chunk.t()[:, None, :] + t_arange[None, :, None]
            ).reshape(t_chunk, nch)
            t_idx_3d = t_idx_2d[:, :, None].expand(t_chunk, nch, nb)
            gathered = torch.gather(vis_T, 0, t_idx_3d)
            src = gathered.reshape(t_chunk, nch * nb)
            out_c = torch.zeros(
                (t_chunk, n_filled + 1),
                dtype=torch.complex64, device=device,
            )
            out_c.index_add_(1, cim_cb, src)
            out[c0:c1] = out_c[:, :n_filled].reshape(chunk, t_dedisp, n_filled)
        return out

    def _get_dedisp_csr(
        self, *, nb: int, nch: int, n_filled: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lazily construct + cache the per-cell CSR map for the
        Phase-10 fused Triton dedisp kernel.

        See :func:`triton_dedisp.build_cell_csr` for the layout.
        Cached on the instance keyed by ``(nb, nch, n_filled, device)``
        because the gridder pattern (and hence cell_index_map) is fixed
        per-context, so this only ever runs once per Stage1MultiDMCoarseDM.
        """
        cache_key = (nb, nch, n_filled, str(device))
        cached = getattr(self, "_dedisp_csr_cache", None)
        if cached is None:
            cached = {}
            self._dedisp_csr_cache = cached
        hit = cached.get(cache_key)
        if hit is not None:
            return hit
        from dsart.services.triton_dedisp import build_cell_csr  # noqa: PLC0415
        cim_bc = self.gridder.cell_index_map.to(device)                # (NSRC,) int64
        csr_offs, csr_b, csr_c = build_cell_csr(
            cim_bc, n_filled=n_filled, nchan_eff=nch, nbase=nb,
        )
        triple = (csr_offs, csr_b, csr_c)
        cached[cache_key] = triple
        return triple

    def _dedisperse_one_window(
        self, vis_stokes_i: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-DM stage-1 shift + grid to a single vis tensor.

        Inner primitive shared by the legacy (single-block) and
        F34-sliding-window paths. Returns
        ``(N_DM, T_dedisp(n_fv), N_filled) complex64`` where
        ``T_dedisp(n_fv) = n_fv - max_bin_shift`` over the selected
        DM subset.

        RT Phase 5 — layout-aware coalesced gather, no transpose
        ==========================================================

        Replaces the Phase-2 implementation's pattern of:

          1. Materialise a ``(chunk*t_dedisp, NBASE, NCHAN_eff)``
             gather buffer via ``vis_stokes_i.gather(0, t_idx_b)``
             whose memory access pattern is *uncoalesced* — for
             fixed ``(t', b)`` varying the innermost ``c``,
             consecutive output cells read from time slices
             ``Δshift × NBASE × NCHAN_eff`` bytes apart in the
             contiguous ``(T, B, C)`` layout. Measured 45 GB/s
             effective on a 2080 Ti at the production op-point
             (peak DRAM is 616 GB/s — ~7% of peak, the canonical
             signature of a strided gather).
          2. Pass the buffer to ``gridder.compute`` which scatter-
             adds into a ``(T, n_filled+1)`` complex output.

        With a two-step layout-aware path that does NOT require a
        transpose-back of the gather output:

          1. Permute ``vis_stokes_i`` once per call to
             ``(n_fv, NCHAN_eff, NBASE)`` (just swap the inner two
             dims). ``B`` becomes innermost (stride 1) so the
             per-(t, c) gather reads ``NBASE`` contiguous bytes per
             memory transaction → fully coalesced. (~48 ms / call @
             production op-point; ~1.83 GB transient.)
          2. Per chunk: one ``torch.gather`` along the time axis
             with a ``(chunk*t_dedisp, NCHAN_eff, NBASE)`` index
             (broadcast stride-0 on ``B``) → output naturally lands
             in ``(chunk*t_dedisp, NCHAN_eff, NBASE)`` contiguous.
             No transpose needed: ``reshape(T_chunk, C*B)`` is a
             free view.
          3. Per chunk: ``.real.contiguous() / .imag.contiguous()``
             split + two 2-D ``index_add_`` calls (Re + Im) into a
             ``(T_chunk, n_filled+1)`` fp32 accumulator, using a
             *swapped* cell-index map ``cim_cb[c*NBASE + b] =
             cell_index_map[b*NCHAN_eff + c]``. This is the same
             fast scatter the legacy ``gridder.compute`` K=1 fast
             path uses; we only re-order the index to match our
             (T, C, B) source layout.

        The big wins vs the Phase-4 baseline:

          * Coalesced gather (~9× faster on the gather step).
          * No (T, B, C) transpose-back materialisation (~150 ms +
            ~1 GB transient saved per call vs an earlier attempt).
          * Same fast 2-D ``index_add_`` scatter (~33 ms / chunk).

        Phase 5 measured @ production op-point on h01 GPU::

            phase                Phase 4 ms    Phase 5 ms    Δ
            -----------------    ----------    ----------    -------
            permute_vis               0             48        +48
            gather (12 chunks)       977           110       -867
            src_re/im split           0             45        +45
            scatter (Re + Im)        394           394          0
            -----------------    ----------    ----------    -------
            dedisp_total           1371           597       -774

        Equivalence with the legacy path is bit-identical on CPU
        (gather is mathematically identical — only the source
        layout differs — and the scatter is the same Re/Im
        ``index_add_`` with a re-ordered index map) and
        ULP-tolerant on GPU (atomic-add reduction order is
        non-deterministic across kernel-launch boundaries).

        Caller is responsible for shape validation; the public
        wrappers (``dedisperse_from_vis``) do the upfront
        shape-check + max-shift sanity errors.
        """
        n_fv = int(vis_stokes_i.shape[0])
        t_dedisp = self.t_dedisp_for(n_fv)
        n_filled = int(self.gridder.pattern.n_filled)
        nb = int(vis_stokes_i.shape[1])
        nch = int(vis_stokes_i.shape[2])
        device = vis_stokes_i.device

        # CPU fallback — Triton requires CUDA tensors. Tests that run
        # purely on CPU still need a working dedisp path; route to the
        # legacy Phase-9 PyTorch implementation in that case. Production
        # path (and the GPU integration / RT bench) always takes the
        # Triton branch below.
        if device.type != "cuda":
            return self._dedisperse_one_window_phase9_cpu(vis_stokes_i)

        # RT Phase 10 — fused gather+scatter Triton kernel.
        # Replaces the (gather → 1.78 GB cfp32 intermediate → 130 ms
        # complex64 atomicAdd scatter) pair with one kernel that walks
        # each grid cell's CSR source list directly. Microbench at
        # production op-point: 196 → 12 ms (16.4×). End-to-end win
        # validated below in the in-tree integration tests.
        from dsart.services.triton_dedisp import fused_dedisp_triton  # noqa: PLC0415

        # One-time permute (T, B, C) cfp32 → (B, C, T) cfp32. The
        # kernel needs T to be the innermost dim so 32-thread warps
        # coalesce. Same memory cost as the legacy (T, B, C) → (T, C, B)
        # permute (one full-tensor copy of ~870 MB at the production
        # op-point).
        vis_BCT = vis_stokes_i.permute(1, 2, 0).contiguous()         # (B, C, T_full) cfp32

        # Split into (re, im) fp32 views. The kernel reads one fp32 at
        # a time; we avoid the fp32 cast since the input is already
        # cfp32 (from the Phase-6/9 stages of compute_split).
        vis_re = vis_BCT.real.contiguous()                            # (B, C, T_full) fp32
        vis_im = vis_BCT.imag.contiguous()
        del vis_BCT

        # Per-(c, dm) bin-shift table → device int32. n_dm * nch * 4
        # bytes — under 5 KB at the production op-point.
        bin_shifts_full = self.plan.delay_bins_per_chgroup(self.chgroup)
        bin_shifts = bin_shifts_full[:nch, self._dm_idx_iter]
        bin_shifts_dev = torch.as_tensor(
            bin_shifts, dtype=torch.int32, device=device,
        ).contiguous()                                                # (C, n_dm) int32

        # CSR map of grid cell → source (b, c) list. Cached per context.
        csr_offs, csr_b, csr_c = self._get_dedisp_csr(
            nb=nb, nch=nch, n_filled=n_filled, device=device,
        )

        out_re, out_im = fused_dedisp_triton(
            vis_re, vis_im,
            bin_shifts=bin_shifts_dev,
            csr_offs=csr_offs, csr_b=csr_b, csr_c=csr_c,
            n_filled=n_filled, t_dedisp=t_dedisp,
            BLOCK_T=128,
        )
        del vis_re, vis_im, bin_shifts_dev
        return torch.complex(out_re, out_im)

    def _dedisperse_one_window_legacy(
        self, vis_stokes_i: torch.Tensor,
    ) -> torch.Tensor:
        """RT Phase 2 dedisperse path — kept for regression / A-B testing.

        See :meth:`_dedisperse_one_window` for the production path.
        This implementation builds the (chunk*t_dedisp, NBASE,
        NCHAN_eff) gather buffer via the legacy uncoalesced gather
        and passes it to ``gridder.compute`` for the scatter.
        Bit-identical to Phase 4 / Phase 3 / Phase 2 on the CPU
        reduction-order branch.
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

        bin_shifts_full = self.plan.delay_bins_per_chgroup(self.chgroup)
        bin_shifts = bin_shifts_full[:nch, self._dm_idx_iter]
        bin_shifts_dev = torch.as_tensor(
            bin_shifts, dtype=torch.int64, device=device,
        )
        t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)

        for c0 in range(0, self.n_dm, dm_chunk):
            c1 = min(c0 + dm_chunk, self.n_dm)
            chunk = c1 - c0
            bs_chunk = bin_shifts_dev[:, c0:c1]
            t_idx = (
                bs_chunk.t()[:, None, :] + t_arange[None, :, None]
            )
            t_idx_flat = t_idx.reshape(chunk * t_dedisp, 1, nch)
            t_idx_b = t_idx_flat.expand(chunk * t_dedisp, nb, nch)
            buf = vis_stokes_i.gather(0, t_idx_b)
            grid_chunk = self.gridder.compute(buf)
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

    Multi-DM mode (M7.4 fix): set ``n_dm > 1`` to maintain ``n_dm``
    independent EMA states. The :meth:`apply` ``dm_slot`` argument
    selects which state to read+update. This is required when the
    same EMA is fed multiple dedispersion trials in one block — a
    single shared state would have each trial's residuals leak into
    the others, leaving comparable burst energy in every trial.

    Args:
        alpha: EMA smoothing factor in (0, 1]. The EMA half-life is
            ``ln(0.5) / ln(1-alpha) ≈ 0.69 / alpha`` cubes. Default
            ``0.001`` → ~700-cube half-life (~7 s at 134 ms cube
            cadence).
        warmup_cubes: number of cubes at the start during which we
            BUILD the EMA but do NOT subtract (so the first few
            cubes are not artificially zeroed by the cold EMA).
        n_dm: number of independent EMA states to keep. Default 1
            (legacy single-DM path). Set to ``plan.n_coarse`` (or
            ``multi_dm.n_dm``) to make each dedispersion trial own
            its own running mean. Backward-compatible: ``dm_slot=0``
            with ``n_dm=1`` reproduces the legacy single-slot
            behaviour bit-for-bit.
    """

    alpha: float = 0.001
    warmup_cubes: int = 8
    n_dm: int = 1

    _running_mean_per_dm: list = field(
        default_factory=list, init=False, repr=False,
    )
    _cubes_seen_per_dm: list = field(
        default_factory=list, init=False, repr=False,
    )

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
        if self.n_dm < 1:
            raise ValueError(
                f"StaticSkyEMA.n_dm={self.n_dm}, expected >= 1"
            )
        self._running_mean_per_dm = [None] * int(self.n_dm)
        self._cubes_seen_per_dm = [0] * int(self.n_dm)

    @property
    def cubes_seen(self) -> int:
        """Cubes seen in slot 0 (backwards-compat alias for single-DM use)."""
        return int(self._cubes_seen_per_dm[0]) if self._cubes_seen_per_dm else 0

    def cubes_seen_for(self, dm_slot: int = 0) -> int:
        return int(self._cubes_seen_per_dm[int(dm_slot)])

    @property
    def in_warmup(self) -> bool:
        """Slot 0 warmup status (backwards-compat alias)."""
        return self.cubes_seen < self.warmup_cubes

    def in_warmup_for(self, dm_slot: int = 0) -> bool:
        return self.cubes_seen_for(dm_slot) < self.warmup_cubes

    def reset(self) -> None:
        self._running_mean_per_dm = [None] * int(self.n_dm)
        self._cubes_seen_per_dm = [0] * int(self.n_dm)

    def apply(
        self, gridded: torch.Tensor, dm_slot: int = 0,
    ) -> torch.Tensor:
        """Subtract the running mean from ``gridded`` and update the EMA.

        ``gridded`` is expected to be ``(n_fast_vis, N_filled)`` complex64
        (the output of :meth:`FastVisGridder.compute`). The EMA is
        kept at ``(N_filled,)`` complex64 — averaged over the
        ``n_fast_vis`` axis on the way in, broadcast back on the way
        out.

        ``dm_slot``: which of ``n_dm`` independent EMA states to
        read+update. Default 0 for backwards-compat with the
        legacy single-DM call sites.
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
        slot = int(dm_slot)
        if slot < 0 or slot >= self.n_dm:
            raise IndexError(
                f"StaticSkyEMA.apply: dm_slot={slot} out of range "
                f"[0, {self.n_dm})"
            )

        per_cell_mean = gridded.mean(dim=0)                              # (N_filled,)
        running_mean = self._running_mean_per_dm[slot]
        cubes_seen = int(self._cubes_seen_per_dm[slot])

        if running_mean is None:
            self._running_mean_per_dm[slot] = per_cell_mean.clone().detach()
            out = gridded.clone()                                        # cold start: pass through
        elif cubes_seen < self.warmup_cubes:
            out = gridded.clone()                                        # build EMA, don't subtract
            self._running_mean_per_dm[slot] = (
                (1.0 - self.alpha) * running_mean
                + self.alpha * per_cell_mean
            )
        else:
            out = gridded - running_mean.unsqueeze(0)                    # subtract, then update
            self._running_mean_per_dm[slot] = (
                (1.0 - self.alpha) * running_mean
                + self.alpha * per_cell_mean
            )

        self._cubes_seen_per_dm[slot] = cubes_seen + 1
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
    # ---- RFI flagger tuning knobs (M7.6) ----
    # All default to the library defaults baked into dsart.rfi.*; override
    # via the CLI flags (--sk-far, --bandpass-k, --group-k, --sumthr-max-m,
    # --sumthr-eta, --m-values, --warmup-cubes, --sumthr-disabled) so the
    # production pipeline can be retuned without code changes.
    rfi_sk_far: float | None = None
    rfi_bandpass_k: float | None = None
    rfi_group_k: float | None = None
    rfi_sumthr_max_m: int | None = None
    rfi_sumthr_eta: float | None = None
    rfi_m_values: tuple[int, ...] | None = None
    rfi_warmup_cubes: int | None = None
    rfi_sumthr_enabled: bool = True
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

    inject_watch_enabled: bool = False
    """M7.4 Phase 6 runtime: when True, ``build_context`` always
    constructs an :class:`OnlineInjector` (even when
    :attr:`inject_configs` is empty) so :class:`RuntimeInjectWatch`
    can ``add_pending`` configs delivered via etcd at runtime. The
    Control-tab "Send injection" button in dsa_monitor produces these
    runtime writes; the dsart_rt orchestrator forwards them to the
    per-chgroup ``/cmd/dsart/corr/<chgroup>/inject`` key the watch
    listens on. Default ``False`` = startup-only injection (CLI
    ``--inject-spec`` is still honoured)."""

    inject_configs: tuple[InjectionConfig, ...] = ()
    """M7.4 Phase 6: live voltage-domain signal injection. When
    non-empty, ``build_context`` constructs an :class:`OnlineInjector`
    for this chgroup and queues each :class:`InjectionConfig` via
    :meth:`OnlineInjector.add_pending`. The injector is then called
    via :func:`_apply_online_injection` **after** RFI excision and
    calibration (M7.4 Phase 8; was pre-RFI through Phase 6/7), so
    injected pulses model a calibrated-frame point source and exercise
    the full grid → coarse-DM → static-sky → transport → search chain
    end-to-end. Default ``()`` = injection disabled (zero hot-path
    overhead — the runtime check is a single ``is None`` test in
    ``_apply_online_injection``).

    Operator workflow (Phase 6):

    * CLI: ``--inject-spec '{...}'`` (repeatable) appends one
      :class:`InjectionConfig`-shaped JSON dict per occurrence.
    * Python API / tests: pass a tuple of :class:`InjectionConfig`
      instances directly.

    The native-sample arrival time is referenced to
    ``apply_at_specnum``; the orchestrator maps the per-block counter
    ``block_n`` to ``block_specnum_start = block_n * NPACKETS_PER_BLOCK``
    (block counter is 1-indexed from service start since SNAP
    specnum sourcing from packet headers is the M7.2.8 corner-turn
    work). For Phase 6 bench / soak runs this is enough: operators
    pick ``apply_at_specnum`` a few blocks ahead of the launch
    (default = ``5 * NPACKETS_PER_BLOCK`` ⇒ pulse lands at block 5)."""


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

    injector: OnlineInjector | None = None
    """M7.4 Phase 6: voltage-domain online injector. When set,
    :func:`_apply_online_injection` calls
    :meth:`OnlineInjector.apply_block` AFTER RFI excision and
    calibration (M7.4 Phase 8). ``None`` = injection disabled (no
    runtime cost). Built in :func:`build_context` from
    :attr:`FastIntegrationConfig.inject_configs`."""

    profiler: "StageProfiler | None" = None
    """Optional per-stage CUDA-event timing profiler (M7.2 Phase 0
    diagnostic). When set, :func:`process_block` brackets each major
    stage with a paired ``torch.cuda.Event`` and the profiler logs a
    rolling mean per stage every ``profiler.every`` blocks. When
    ``None`` (default), the profiler hooks are skipped entirely so
    the hot path has zero overhead. See :class:`StageProfiler`."""


class StageProfiler:
    """Per-stage CUDA-event timing for the corr-fast pipeline (M7.2
    Phase 0 diagnostic).

    Wrap a section of code with::

        with ctx.profiler.bracket("multi_dm"):
            ...

    The profiler records a paired ``torch.cuda.Event`` around each
    section, then on :meth:`commit_block` syncs the last event and
    accumulates the elapsed time. Every ``every`` blocks it emits a
    summary log line of mean per-stage ms over the window.

    Overhead: 2 event-records per bracketed stage (~10 µs each) plus
    one ``synchronize`` per block. The synchronize is comparable to
    the wall-clock the orchestrator already pays at the end of each
    block (the next iteration blocks on PSRDADA fada writability),
    so the net measurement perturbation is small. The profiler is
    OFF by default — pass ``--profile-stages-every N`` to enable.
    """

    def __init__(self, every: int = 64) -> None:
        if every < 1:
            raise ValueError(f"StageProfiler.every={every}, expected >= 1")
        self.every = int(every)
        self._block_n: int = 0
        self._totals: dict[str, list[float]] = {}
        # Active brackets for the current block (cleared on commit_block).
        # Stored as (name, start_event, end_event).
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextlib.contextmanager
    def bracket(self, name: str):
        """Bracket a code section with a paired CUDA event."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))

    def commit_block(self) -> None:
        """Sync the last event, accumulate per-stage ms, emit if due."""
        if not self._pending:
            self._block_n += 1
            return
        # Syncing the final event ensures all stages' end events have
        # actually completed on the GPU (events recorded earlier on
        # the same stream are guaranteed-done by the time a later
        # event syncs). elapsed_time() then returns a well-defined
        # number for every bracket.
        self._pending[-1][2].synchronize()
        for name, start, end in self._pending:
            self._totals.setdefault(name, []).append(start.elapsed_time(end))
        self._pending.clear()
        self._block_n += 1
        if self._block_n >= self.every:
            self._emit()
            self._block_n = 0

    def _emit(self) -> None:
        if not self._totals:
            return
        parts: list[str] = []
        for name, ms_list in self._totals.items():
            if not ms_list:
                continue
            mean = sum(ms_list) / len(ms_list)
            ms_sorted = sorted(ms_list)
            p50 = ms_sorted[len(ms_sorted) // 2]
            p99 = ms_sorted[min(len(ms_sorted) - 1, int(0.99 * len(ms_sorted)))]
            parts.append(f"{name}=mean:{mean:.2f}ms p50:{p50:.2f}ms p99:{p99:.2f}ms")
        LOG.info(
            "stage-profile (over %d blocks): %s",
            sum(len(v) for v in self._totals.values()) // max(1, len(self._totals)),
            " | ".join(parts),
        )
        # Reset windows for next emission.
        for ms_list in self._totals.values():
            ms_list.clear()


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
        4b. _apply_online_injection — post-cal voltage injection
            (M7.4 Phase 8; no-op unless ctx.injector is set)
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
    # RT Phase 4: process_block is now a thin wrapper that calls the
    # corr-side and consume-side phases sequentially on the default
    # stream. The phase split is what BlockPipeliner uses to overlap
    # the two halves on separate CUDA streams.
    if ctx.profiler is not None:
        # M7.2 Phase 0 diagnostic: bracket the two halves so we can
        # decompose per-block ms into corr-half (unpack + RFI + cal +
        # compute_split) vs consume-half (multi-DM dedisp + static-sky
        # + stage2_fifo + transport_tx). The two halves are the
        # natural boundary for N-scaling questions: the corr-half is
        # N-independent; the consume-half is where ``n_dm`` shows up.
        with ctx.profiler.bracket("corr_phase"):
            vis_stokes_i, rfi_result = _process_block_corr_phase(
                raw, ctx=ctx, block_n=block_n,
            )
        with ctx.profiler.bracket("consume_phase"):
            out = _process_block_consume_phase(
                vis_stokes_i, rfi_result, ctx=ctx, block_n=block_n,
            )
        return out

    vis_stokes_i, rfi_result = _process_block_corr_phase(
        raw, ctx=ctx, block_n=block_n,
    )
    return _process_block_consume_phase(
        vis_stokes_i, rfi_result,
        ctx=ctx, block_n=block_n,
    )


def _process_block_unpack_phase(
    raw: np.ndarray,
    *,
    ctx: IntegrationContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RT Phase 13 (3-stream BlockPipeliner): unpack-only sub-phase.

    Splits Step 1 of :func:`_process_block_corr_phase` out into its
    own callable so the 3-stream :class:`BlockPipeliner3S` can issue
    it on a dedicated unpack stream — overlapping the H2D copy +
    fluff with the previous block's compute_split on the compute
    stream. Returns the (real_v, imag_v) fp16 voltage pair in GEMM
    layout.
    """
    return unpack_int4_split(
        raw, device=ctx.device, out_dtype=ctx.voltage_dtype,
    )


def _apply_online_injection(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    *,
    ctx: IntegrationContext,
    block_n: int,
) -> None:
    """M7.4 Phase 8 (2026-05-28): voltage-domain online injection,
    applied **AFTER** RFI excision and calibration.

    Physics rationale for the post-cal placement: an injected signal
    models an astrophysical point source whose per-antenna fringe is
    defined in the *calibrated* array frame (the (l, m) the operator
    requests is where the source should land in the calibrated image).
    Applying the injection to pre-cal voltages — as the pipeline did
    through Phase 6/7 — meant ``apply_cal_split`` then rotated each
    antenna's injected phase by that antenna's calibration solution,
    smearing the intended fringe and shifting / decohering the source
    away from the requested (l, m). Injecting after cal makes the
    injected source appear exactly where intended, independent of the
    calibration table. (RFI excision is also upstream now, so injected
    bursts are never mistaken for narrowband CW and flagged out — the
    desired behaviour for an injection validation signal.)

    The call mutates ``real_v`` / ``imag_v`` in place via
    ``OnlineInjector.apply_block`` (``torch.Tensor.add_``); no tensor
    is returned and no extra allocation/copy is introduced, so the
    relocation is RT-cost-neutral vs the pre-cal placement (it is the
    same single ``apply_block`` call, just sequenced after cal).

    No-op (single ``is None`` test) when ``ctx.injector is None``.

    The native-sample reference for ``apply_at_specnum`` is the block
    counter ``block_n`` × ``NPACKETS_PER_BLOCK`` (1-indexed from
    service start).
    """
    if ctx.injector is None:
        return
    block_specnum_start = int(block_n) * NPACKETS_PER_BLOCK
    inject_log = ctx.injector.apply_block(
        real_v, imag_v, block_specnum_start,
    )
    if inject_log["active_inj_ids"]:
        LOG.info(
            "M7.4 inject (post-cal): block_n=%d specnum_start=%d "
            "active=%s n_purged=%d",
            block_n, block_specnum_start,
            inject_log["active_inj_ids"], inject_log["n_purged"],
        )


def _process_block_corr_phase(
    raw: np.ndarray,
    *,
    ctx: IntegrationContext,
    block_n: int,
) -> tuple[torch.Tensor, FlagBlockResult | None]:
    """RT Phase 4: corr-side half of :func:`process_block`.

    Steps 1-6: unpack int4 → RFI flag → RFI mask → cal apply → fused
    ``compute_split`` → returns a ready-to-grid Stokes-I cube.

    All ops in this function run on the current CUDA stream (either
    the default stream when called from :func:`process_block`, or the
    pipeliner's ``_corr_stream``). Pure GPU work after the host-side
    int4 unpack; the returned ``vis_stokes_i`` may still be in flight
    on the calling stream when this function returns.
    """
    # 1. Unpack
    real_v, imag_v = _process_block_unpack_phase(raw, ctx=ctx)

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

    # 4b. M7.4 Phase 8: voltage-domain online injection, AFTER RFI +
    # cal (was step 1b / pre-RFI through Phase 6/7 — see
    # _apply_online_injection docstring for the post-cal rationale).
    _apply_online_injection(real_v, imag_v, ctx=ctx, block_n=block_n)

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

    return vis_stokes_i, rfi_result


def _process_block_compute_phase(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    *,
    ctx: IntegrationContext,
    block_n: int,
) -> tuple[torch.Tensor, FlagBlockResult | None]:
    """RT Phase 13 (3-stream BlockPipeliner): RFI + cal + compute_split.

    Steps 2-6 of :func:`_process_block_corr_phase`, taking the
    pre-unpacked (real_v, imag_v) voltages instead of raw bytes.
    Caller is responsible for stream-syncing the inputs from the
    upstream unpack stream before issuing this on a different stream.
    """
    # 2. RFI flag (before cal — flags should be on raw data + DC pol
    # so that any cal-induced dynamic range shifts can't mask them).
    rfi_result: FlagBlockResult | None = None
    if ctx.rfi_flagger is not None and ctx.cfg.rfi_enabled:
        rfi_result = ctx.rfi_flagger.flag_block(real_v, imag_v)
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

    # 4b. M7.4 Phase 8: voltage-domain online injection, AFTER RFI +
    # cal. The 3-stream pipeliner path (--pipeliner-3s) previously did
    # NOT inject at all (apply_block lived only in the non-pipeliner
    # _process_block_corr_phase); this adds parity so injection works
    # identically regardless of the streaming mode, and always lands
    # post-cal. See _apply_online_injection docstring.
    _apply_online_injection(real_v, imag_v, ctx=ctx, block_n=block_n)

    # 5. + 6. Streamed kernel + Stokes-I (production fused fast path).
    ppfv = ctx.cfg.t_int_fast_native // NTIMES_PER_PACKET
    n_packets_in = real_v.shape[3]
    if n_packets_in % ppfv != 0:
        raise ValueError(
            f"compute_phase: n_packets_in={n_packets_in} not a multiple "
            f"of packets_per_fast_vis={ppfv} (t_int_fast_native="
            f"{ctx.cfg.t_int_fast_native})"
        )
    n_fv_total = n_packets_in // ppfv
    chan_sum_factor = int(ctx.cfg.chan_sum_factor)
    full_nchan = ctx.kernel.nchan
    if chan_sum_factor > 1 and full_nchan % chan_sum_factor != 0:
        raise ValueError(
            f"chan_sum_factor={chan_sum_factor} does not divide "
            f"nchan={full_nchan}"
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

    return vis_stokes_i, rfi_result


def _process_block_consume_phase(
    vis_stokes_i: torch.Tensor,
    rfi_result: FlagBlockResult | None,
    *,
    ctx: IntegrationContext,
    block_n: int,
) -> IntegrationOutput:
    """RT Phase 4: consume-side half of :func:`process_block`.

    Steps 7-11: dedispersion → static-sky subtract → stage2 fifo push
    → transport tx. All ops run on the current CUDA stream (either
    the default stream when called from :func:`process_block`, or the
    pipeliner's ``_dedisp_stream``).

    Caller must ensure ``vis_stokes_i`` is fully written by the time
    work on the current stream starts (handled by ``BlockPipeliner``
    via cross-stream events; trivially satisfied for the default-
    stream path).
    """
    if ctx.multi_dm_coarse_dm is not None:
        # ----- Chunk-9 / F25 production path: multi-DM-trial vis-domain
        # stage-1 shifts → per-trial grid → per-trial static-sky.
        # Returns (N_DM, T_dedisp, N_filled) complex64.
        if ctx.profiler is not None:
            with ctx.profiler.bracket("multi_dm"):
                dedispersed = ctx.multi_dm_coarse_dm.dedisperse_from_vis(
                    vis_stokes_i, block_n=block_n,
                )
        else:
            dedispersed = ctx.multi_dm_coarse_dm.dedisperse_from_vis(
                vis_stokes_i, block_n=block_n,
            )
        del vis_stokes_i

        # 8. Per-trial static-sky EMA subtraction (collapses the
        # T_dedisp axis → N_filled internally; replicates back to
        # the full (T_dedisp, N_filled) shape).
        #
        # M7.4 fix: pass ``dm_slot=c`` so each dedispersion trial has
        # its own EMA state (built by ``_build_integration_context``
        # with ``n_dm=plan.n_coarse``). A shared single-slot EMA
        # leaks each trial's residuals into every subsequent trial,
        # which empirically made the burst show up at comparable
        # SNR in every coarse-DM bin on the 250924mptq replay.
        if ctx.static_sky is not None and not ctx.cfg.static_sky_disabled:
            if ctx.profiler is not None:
                with ctx.profiler.bracket("static_sky"):
                    n_dm, t_dedisp, n_filled = dedispersed.shape
                    for c in range(n_dm):
                        dedispersed[c] = ctx.static_sky.apply(
                            dedispersed[c], dm_slot=c,
                        )
            else:
                n_dm, t_dedisp, n_filled = dedispersed.shape
                for c in range(n_dm):
                    dedispersed[c] = ctx.static_sky.apply(
                        dedispersed[c], dm_slot=c,
                    )

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

    # 10. Stage-2 FIFO push (NoOp stub or real Stage2FIFO ring;
    #     see ``--transport-tx-host`` wiring in :func:`run`)
    # 11. Transport TX (NoOp stub, chunk-8 cfp16/cint8, or M4a prod
    #     ProdFrame + cint8 + pacer; see ``--transport-tx-host``)
    rfi_warmup_flag = bool(rfi_result.warmup) if rfi_result is not None else False
    if ctx.profiler is not None:
        with ctx.profiler.bracket("stage2_tx"):
            cubes_for_tx = ctx.stage2_fifo.push(dedispersed, block_n=block_n)
            n_tx = ctx.transport_tx.transmit(
                cubes_for_tx, block_n=block_n,
                rfi_warming_up=rfi_warmup_flag,
            )
    else:
        cubes_for_tx = ctx.stage2_fifo.push(dedispersed, block_n=block_n)
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
# RT Phase 4: 2-stream / 2-slot ring-buffer pipeliner
# ---------------------------------------------------------------------------


class BlockPipeliner:
    """RT Phase 4: overlap the corr-side and consume-side halves of
    :func:`process_block` on two CUDA streams with a 2-slot ring buffer.

    Why
    ===

    After Phase 3, the K=1 production op-point per-block GPU breakdown
    on a 2080Ti is roughly::

        unpack_int4_split        ~107 ms   (4%)
        compute_split (fused)    ~817 ms   (36%)   ── corr stream A
        dedisperse_one_window   ~1370 ms   (60%)   ── consume stream B
                                ─────────
        wall (sequential)       ~2306 ms   (= corr + dedisp)

    Stream A (corr) and stream B (dedisp + gridder) are independent —
    they only communicate through ``vis_stokes_i``, which is at most
    ~88 MB at the production op-point — so they can run concurrently
    on separate CUDA streams. Steady-state overlapped wall =
    ``max(corr_time, dedisp_time)`` ≈ 1370 ms per block, a 1.7×
    speedup over the sequential Phase 3 path.

    Trade-off: 2-block result latency. The output of block ``N`` is
    returned by the ``push(N+2)`` call (since slot ``N % 2`` is reused
    by block ``N+2`` and we sync on it then). Real-time FRB search
    cares about per-block throughput, not the 268 ms (= 2 × 134 ms)
    extra wall latency.

    Pipeline
    ========

    Per ``push(raw, block_n)`` call::

        slot = self._n_pushed % n_buffers              # 0 or 1

        # ── Stream A: corr-side (independent of consume) ──
        # (No explicit wait_event needed: stream A serializes through
        # its own queue; vis_stokes_i is stream-A-allocated and only
        # consumed on stream B after the cross-stream record_stream.)
        with cuda.stream(corr_stream):
            vis_stokes_i, rfi_result = _corr_phase(raw, ctx)
            corr_done.record(corr_stream)

        # ── Stream B: consume-side (waits for corr-side) ──
        with cuda.stream(dedisp_stream):
            dedisp_stream.wait_event(corr_done)
            # CRITICAL: tell allocator stream B is also using
            # vis_stokes_i so it isn't freed before B is done.
            vis_stokes_i.record_stream(dedisp_stream)
            output = _consume_phase(vis_stokes_i, rfi_result, ctx, n)
            dedisp_done.record(dedisp_stream)

    The output for block ``N`` is held until ``push(N+n_buffers)``,
    at which point we synchronize on ``dedisp_done[N % n_buffers]``
    and return it. Use :meth:`flush` after the last ``push`` to
    drain the in-flight blocks.

    Bit-identity
    ============

    The pipelined path produces bit-identical
    ``IntegrationOutput.gridded_minus_sky`` to a sequential loop
    over :func:`process_block`, because the per-block GPU graph is
    unchanged — only the host-side issue order and stream
    assignment differ. The cross-stream events ensure stream B
    reads the same bytes from ``vis_stokes_i`` it would have on
    the default stream.

    Notes
    =====

    * Requires ``ctx.device.type == "cuda"``. CPU contexts can't
      use multiple streams; on CPU just call :func:`process_block`
      sequentially.
    * Stateful subsystems (``static_sky`` EMA, ``stage2_fifo``,
      ``transport_tx``, ``Stage1MultiDMCoarseDM`` sliding window)
      are updated in block order on stream B's queue, which
      preserves the same in-order semantics as the sequential
      path. The static-sky EMA is float-state on the GPU; reads
      and writes are serialized by stream B.
    * The host-side ``rfi_result`` (a Python dataclass returned
      by stream A) is captured synchronously in stream A's
      issue-time slice and handed off to stream B at issue
      time too — its scalar fields ``flag_fraction_total``
      and ``warmup`` are bool / float, not GPU tensors, so no
      cross-stream synchronization is required for them.
    """

    def __init__(
        self,
        ctx: IntegrationContext,
        *,
        n_buffers: int = 2,
    ) -> None:
        if ctx.device.type != "cuda":
            raise ValueError(
                f"BlockPipeliner requires a CUDA context; got "
                f"ctx.device={ctx.device}. Use process_block() "
                f"directly for CPU contexts."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "BlockPipeliner needs torch.cuda.is_available()"
            )
        if n_buffers < 2:
            raise ValueError(
                f"n_buffers={n_buffers}; pipelining requires "
                f"≥2 buffers (got 1 ⇒ no overlap)."
            )

        self.ctx = ctx
        self.n_buffers = n_buffers
        self._corr_stream = torch.cuda.Stream(device=ctx.device)
        self._dedisp_stream = torch.cuda.Stream(device=ctx.device)

        # Per-slot in-flight state. Only the dedisp_done event needs to
        # survive across push() calls (we sync on it to free up the slot
        # for re-use). vis_stokes_i is not pre-allocated; it's produced
        # fresh by stream A's compute_split each block (the allocator
        # caches the underlying memory after the first n_buffers blocks).
        self._dedisp_done_events: list[torch.cuda.Event | None] = (
            [None] * n_buffers
        )
        self._inflight_outputs: list[IntegrationOutput | None] = (
            [None] * n_buffers
        )

        self._n_pushed = 0
        self._closed = False

    def push(
        self,
        raw: np.ndarray,
        *,
        block_n: int,
    ) -> IntegrationOutput | None:
        """Submit one fada block to the pipeline.

        Returns the :class:`IntegrationOutput` for the block submitted
        ``n_buffers`` calls ago, or ``None`` while the pipeline is
        still warming up (first ``n_buffers`` calls).
        """
        if self._closed:
            raise RuntimeError("BlockPipeliner.push() after close()")

        slot = self._n_pushed % self.n_buffers

        # If this slot was previously written, sync on its dedisp_done
        # event and return its output now (= block_n - n_buffers).
        prior_output: IntegrationOutput | None = None
        if self._n_pushed >= self.n_buffers:
            ev = self._dedisp_done_events[slot]
            assert ev is not None, (
                f"BlockPipeliner: slot {slot} reused but no "
                f"dedisp_done event recorded"
            )
            ev.synchronize()
            prior_output = self._inflight_outputs[slot]
            self._inflight_outputs[slot] = None
            self._dedisp_done_events[slot] = None

        # ── Stream A: corr-side ──
        # Note: the corr stream serializes through its own queue, so
        # block N's corr is naturally ordered after block N-1's corr.
        # We do NOT need wait_event on the consume side here: the
        # vis_stokes_i is stream-A-allocated, so until we cross-stream
        # it to stream B (via the record_stream + wait_event below),
        # the allocator won't reclaim its memory.
        with torch.cuda.stream(self._corr_stream):
            vis_stokes_i, rfi_result = _process_block_corr_phase(
                raw, ctx=self.ctx, block_n=block_n,
            )
            corr_done = torch.cuda.Event()
            corr_done.record(self._corr_stream)

        # ── Stream B: consume-side ──
        # 1) wait_event(corr_done): stream B waits for stream A's
        #    compute_split to fully populate vis_stokes_i.
        # 2) vis_stokes_i.record_stream(dedisp_stream): tells the
        #    caching allocator that the dedisp stream is also using
        #    this tensor. Without this, when block N+1's corr starts
        #    on stream A and the allocator looks for free buffers, it
        #    might reclaim vis_stokes_i before stream B has finished
        #    reading from it (since stream A has already finished).
        with torch.cuda.stream(self._dedisp_stream):
            self._dedisp_stream.wait_event(corr_done)
            vis_stokes_i.record_stream(self._dedisp_stream)
            output = _process_block_consume_phase(
                vis_stokes_i, rfi_result,
                ctx=self.ctx, block_n=block_n,
            )
            dedisp_done = torch.cuda.Event()
            dedisp_done.record(self._dedisp_stream)

        # Stash output + event for retrieval ``n_buffers`` calls from now.
        self._inflight_outputs[slot] = output
        self._dedisp_done_events[slot] = dedisp_done
        self._n_pushed += 1
        return prior_output

    def flush(self) -> list[IntegrationOutput]:
        """Drain in-flight blocks. Returns outputs in submission order.

        Call after the last :meth:`push`. After ``flush``, the
        pipeliner cannot accept more pushes (call :meth:`close`).
        """
        outs: list[IntegrationOutput] = []
        # The slots still holding in-flight blocks are the LAST
        # ``min(n_pushed, n_buffers)`` blocks submitted.
        n_pending = min(self._n_pushed, self.n_buffers)
        first_pending = self._n_pushed - n_pending
        for i in range(n_pending):
            slot = (first_pending + i) % self.n_buffers
            ev = self._dedisp_done_events[slot]
            if ev is None:
                continue
            ev.synchronize()
            outs.append(self._inflight_outputs[slot])
            self._inflight_outputs[slot] = None
            self._dedisp_done_events[slot] = None
        return outs

    def close(self) -> None:
        """Mark the pipeliner closed; no further pushes accepted."""
        self._closed = True


class BlockPipeliner3S:
    """RT Phase 13: 3-stream variant of :class:`BlockPipeliner`.

    Splits the pipeline into THREE phases on three CUDA streams:

      * **Stream U** (unpack): :func:`_process_block_unpack_phase` —
        H2D + Stage-1 transpose + Triton fluff (~53 ms / block at the
        production op-point post-Phase-12).
      * **Stream C** (compute): :func:`_process_block_compute_phase` —
        RFI + cal + fused HMMA Triton ``compute_split`` (~91 ms).
      * **Stream D** (dedisp): :func:`_process_block_consume_phase` —
        Phase-10 fused gather+scatter Triton dedisp + downstream
        consumers (~60 ms).

    Steady-state wall is ``max(U_time, C_time, D_time)`` plus a small
    cross-stream serialisation overhead. At the production op-point
    post-Phases 11+12 that's ``max(53, 91, 60) ≈ 91 ms`` theoretical
    (actual measured under SM contention TBD). Equivalently a 3-block
    output latency: the result of block ``N`` is returned by
    ``push(N+3)``.

    Why three streams?
    ==================
    The 2-stream :class:`BlockPipeliner` overlaps the corr-side wall
    (unpack + compute_split = 144 ms) with the consume-side wall
    (dedisp = 60 ms) → max = 144 ms. Now that Phases 8/11 cratered
    the GEMM tail, the unpack stage (53 ms) is comparable in cost to
    the dedisp stage (60 ms), and pulling it out into its own stream
    can shift the long pole from compute_split (91 ms) to itself
    instead of the corr_phase sum.

    The (real_v, imag_v) intermediate (~1.17 GB / block at production
    op-point) is held per-slot — with ``n_buffers=3`` this peaks at
    ~3.5 GB on the 11 GB 2080Ti.
    """

    def __init__(
        self,
        ctx: IntegrationContext,
        *,
        n_buffers: int = 3,
    ) -> None:
        if ctx.device.type != "cuda":
            raise ValueError(
                f"BlockPipeliner3S requires CUDA; got ctx.device={ctx.device}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("BlockPipeliner3S needs torch.cuda.is_available()")
        if n_buffers < 3:
            raise ValueError(
                f"n_buffers={n_buffers}; 3-stream pipelining requires ≥3 buffers"
            )
        self.ctx = ctx
        self.n_buffers = n_buffers
        self._unpack_stream  = torch.cuda.Stream(device=ctx.device)
        self._compute_stream = torch.cuda.Stream(device=ctx.device)
        self._dedisp_stream  = torch.cuda.Stream(device=ctx.device)

        self._dedisp_done_events: list[torch.cuda.Event | None] = (
            [None] * n_buffers
        )
        self._inflight_outputs: list[IntegrationOutput | None] = (
            [None] * n_buffers
        )
        self._n_pushed = 0
        self._closed = False

    def push(
        self,
        raw: np.ndarray,
        *,
        block_n: int,
    ) -> IntegrationOutput | None:
        """Submit one fada block to the 3-stream pipeline.

        Returns the :class:`IntegrationOutput` for the block submitted
        ``n_buffers`` calls ago, or ``None`` while the pipeline is
        warming up.
        """
        if self._closed:
            raise RuntimeError("BlockPipeliner3S.push() after close()")

        slot = self._n_pushed % self.n_buffers

        prior_output: IntegrationOutput | None = None
        if self._n_pushed >= self.n_buffers:
            ev = self._dedisp_done_events[slot]
            assert ev is not None
            ev.synchronize()
            prior_output = self._inflight_outputs[slot]
            self._inflight_outputs[slot] = None
            self._dedisp_done_events[slot] = None

        # ── Stream U: unpack ──
        with torch.cuda.stream(self._unpack_stream):
            real_v, imag_v = _process_block_unpack_phase(raw, ctx=self.ctx)
            unpack_done = torch.cuda.Event()
            unpack_done.record(self._unpack_stream)

        # ── Stream C: RFI + cal + compute_split ──
        with torch.cuda.stream(self._compute_stream):
            self._compute_stream.wait_event(unpack_done)
            real_v.record_stream(self._compute_stream)
            imag_v.record_stream(self._compute_stream)
            vis_stokes_i, rfi_result = _process_block_compute_phase(
                real_v, imag_v, ctx=self.ctx, block_n=block_n,
            )
            compute_done = torch.cuda.Event()
            compute_done.record(self._compute_stream)

        # ── Stream D: dedisp + downstream ──
        with torch.cuda.stream(self._dedisp_stream):
            self._dedisp_stream.wait_event(compute_done)
            vis_stokes_i.record_stream(self._dedisp_stream)
            output = _process_block_consume_phase(
                vis_stokes_i, rfi_result,
                ctx=self.ctx, block_n=block_n,
            )
            dedisp_done = torch.cuda.Event()
            dedisp_done.record(self._dedisp_stream)

        self._inflight_outputs[slot] = output
        self._dedisp_done_events[slot] = dedisp_done
        self._n_pushed += 1
        return prior_output

    def flush(self) -> list[IntegrationOutput]:
        """Drain in-flight blocks. Returns outputs in submission order."""
        outs: list[IntegrationOutput] = []
        n_pending = min(self._n_pushed, self.n_buffers)
        first_pending = self._n_pushed - n_pending
        for i in range(n_pending):
            slot = (first_pending + i) % self.n_buffers
            ev = self._dedisp_done_events[slot]
            if ev is None:
                continue
            ev.synchronize()
            outs.append(self._inflight_outputs[slot])
            self._inflight_outputs[slot] = None
            self._dedisp_done_events[slot] = None
        return outs

    def close(self) -> None:
        self._closed = True


def process_blocks_pipelined(
    raws: list[np.ndarray],
    *,
    ctx: IntegrationContext,
    n_buffers: int = 2,
    block_n_start: int = 1,
) -> list[IntegrationOutput]:
    """Convenience: run :class:`BlockPipeliner` over a list of raw blocks.

    Drains the pipeline at the end so the caller gets one
    :class:`IntegrationOutput` per input ``raw``, in submission order.

    Used by the Phase-4 bit-identity test + the perf bench. Production
    code should construct a :class:`BlockPipeliner` and call ``push``
    inside the fada read loop so the host can issue the next read while
    the GPU pipeline drains the prior block.
    """
    pipeliner = BlockPipeliner(ctx, n_buffers=n_buffers)
    outs: list[IntegrationOutput | None] = [None] * len(raws)
    push_idx = 0
    for raw in raws:
        prior = pipeliner.push(raw, block_n=block_n_start + push_idx)
        if prior is not None:
            outs[push_idx - n_buffers] = prior
        push_idx += 1
    drained = pipeliner.flush()
    pipeliner.close()
    # The last n_buffers entries should now be filled by ``drained``,
    # in submission order.
    for i, out in enumerate(drained):
        outs[len(raws) - len(drained) + i] = out
    # Sanity: every output slot should be populated.
    for i, out in enumerate(outs):
        if out is None:
            raise RuntimeError(
                f"process_blocks_pipelined: output slot {i} not "
                f"populated (n_pushed={push_idx}, n_drained="
                f"{len(drained)}, n_buffers={n_buffers})"
            )
    return outs  # type: ignore[return-value]


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
        rfi_kwargs: dict[str, Any] = {
            "flagants_path": cfg.flagants_path,
            "device": device,
            "run_sum_threshold": cfg.rfi_sumthr_enabled,
        }
        if cfg.rfi_sk_far is not None:
            rfi_kwargs["sk_far"] = cfg.rfi_sk_far
        if cfg.rfi_bandpass_k is not None:
            rfi_kwargs["bandpass_k"] = cfg.rfi_bandpass_k
        if cfg.rfi_group_k is not None:
            rfi_kwargs["group_k"] = cfg.rfi_group_k
        if cfg.rfi_sumthr_max_m is not None:
            rfi_kwargs["sum_threshold_max_m"] = cfg.rfi_sumthr_max_m
        if cfg.rfi_sumthr_eta is not None:
            rfi_kwargs["sum_threshold_eta"] = cfg.rfi_sumthr_eta
        if cfg.rfi_m_values is not None:
            rfi_kwargs["m_values"] = cfg.rfi_m_values
        if cfg.rfi_warmup_cubes is not None:
            rfi_kwargs["warmup_cubes"] = cfg.rfi_warmup_cubes
        rfi_flagger = RFIFlagger(**rfi_kwargs)
        LOG.info(
            "RFIFlagger ready: warmup_cubes=%d sk_far=%.3g "
            "bandpass_k=%.2f group_k=%.2f sumthr=%s m_values=%s",
            rfi_flagger.warmup_cubes, rfi_flagger._sk_far,
            rfi_flagger._bandpass_k, rfi_flagger._group_k,
            "on" if cfg.rfi_sumthr_enabled else "OFF",
            rfi_flagger._m_values,
        )
    else:
        LOG.info("RFIFlagger DISABLED (cfg.rfi_enabled=False)")

    _pattern, gridder = _build_gridder(
        cfg,
        antpos_e=antpos_e, antpos_n=antpos_n,
        is_core_baseline_mask=is_core_baseline_mask,
        device=device,
    )

    # Chunk-9 / F25 production multi-DM path.
    # ``dm_plan`` arg overrides ``cfg.dm_plan_path`` — used by tests +
    # benches that synthesise a custom plan (e.g. chunk-6 single-DM
    # burst, chunk-9 throughput).
    #
    # M7.4 fix: we load the plan *before* the StaticSkyEMA so the EMA
    # can be sized with one independent state per coarse-DM trial
    # (``n_dm=plan.n_coarse``). The pre-M7.4 single-slot EMA leaked
    # residuals across DM trials and made bursts appear at comparable
    # SNR in every coarse-DM bin (see the per-(sid, half) ``dm_p50``
    # table in the 250924mptq postmortem).
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

    # StaticSkyEMA — one independent state per coarse-DM trial when
    # the multi-DM path is active; legacy single-DM path uses one
    # slot (the default). See class docstring for the M7.4 motivation.
    n_static_sky_slots = int(multi_dm.n_dm) if multi_dm is not None else 1
    static_sky: StaticSkyEMA | None = None
    if not cfg.static_sky_disabled:
        static_sky = StaticSkyEMA(
            alpha=cfg.static_sky_alpha,
            warmup_cubes=cfg.static_sky_warmup_cubes,
            n_dm=n_static_sky_slots,
        )
        LOG.info(
            "StaticSkyEMA ready: alpha=%.4g warmup_cubes=%d n_dm=%d",
            static_sky.alpha, static_sky.warmup_cubes, static_sky.n_dm,
        )
    else:
        LOG.info("StaticSkyEMA DISABLED (cfg.static_sky_disabled=True)")

    injector: OnlineInjector | None = None
    if cfg.inject_configs or cfg.inject_watch_enabled:
        injector = OnlineInjector(
            antpos_e=antpos_e,
            antpos_n=antpos_n,
            chgroup=int(cfg.chgroup),
            device=device,
            dtype=voltage_dtype,
        )
        for inj_cfg in cfg.inject_configs:
            injector.add_pending(inj_cfg)
            LOG.info(
                "OnlineInjector queued: id=%s dm=%.2f pc/cc l=%.4f m=%.4f "
                "fluence=%.3g Jy·ms width=%d samples profile=%s "
                "apply_at_specnum=%d",
                inj_cfg.inj_id, inj_cfg.dm_pc_cm3,
                inj_cfg.l_rad, inj_cfg.m_rad,
                inj_cfg.fluence_jy_ms, inj_cfg.width_samples,
                inj_cfg.profile, inj_cfg.apply_at_specnum,
            )
        LOG.info(
            "OnlineInjector ready: chgroup=%d device=%s dtype=%s "
            "n_pending=%d watch_enabled=%s (hot path: in-place add "
            "after unpack, before RFI)",
            cfg.chgroup, device, voltage_dtype, len(injector.pending),
            cfg.inject_watch_enabled,
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
        injector=injector,
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


def _publish_corr_fast_ready(
    cn_id: "int | None",
    *,
    ready: bool,
    warmup_s: "float | None" = None,
    n_blocks: "int | None" = None,
) -> None:
    """Publish corr_fast warmup/readiness state to etcd.

    The dashboard system-state banner uses this to tell operators when
    it is safe to arm (utc_start). It MUST NOT arm until every corr_fast
    has finished its kernel warmup -- arming earlier lets capture data
    flow into fada while the consumer is still JIT-compiling, which pins
    fada at 70/70 and drops packets on sky (see _warmup_pipeline_jit).

    The orchestrator's ``state`` flips to "running" as soon as the
    routines are spawned -- i.e. *before* this warmup completes -- so it
    is not a usable "ready to arm" signal. This per-process record is.

    Keyed at ``/mon/corr_rt/<cn>/corr_fast_ready`` and stamped with the
    current PID so the dashboard can reject a stale record left by a
    previous run (it only trusts a record whose pid matches the live
    ``routines["corr_fast"].pid`` the orchestrator publishes).

    Two one-shot writes per process (warmup start + completion); never on
    the hot path. Best-effort: any etcd error is logged and swallowed so
    a monitoring hiccup can never sink the search pipeline.
    """
    if cn_id is None:
        return
    try:
        from dsautils.dsa_store import DsaStore

        payload = {
            "ready": bool(ready),
            "pid": os.getpid(),
            "time_mjd": time.time() / 86400.0 + 40587.0,
            "warmup_blocks": n_blocks,
            "warmup_s": round(warmup_s, 2) if warmup_s is not None else None,
        }
        DsaStore().put_dict(
            f"/mon/corr_rt/{int(cn_id)}/corr_fast_ready", payload
        )
    except Exception:  # noqa: BLE001 — monitoring must never sink the pipe
        LOG.warning(
            "corr_fast readiness publish failed (cn=%s ready=%s)",
            cn_id, ready, exc_info=True,
        )


def _warmup_pipeline_jit(
    ctx: IntegrationContext,
    *,
    expected_fada_bytes: int,
    n_blocks: int,
) -> None:
    """Pre-compile every Triton / Inductor kernel in the fast path BEFORE
    the ready-sentinel is touched.

    Why this exists (M7.4 Phase 8b, 2026-05-28 — fada data-loss fix):
    --------------------------------------------------------------------
    The fast path JIT-compiles its GPU kernels (unpack_int4_split,
    FastCorrKernel.compute_split, the fused dedispersion kernels, the
    sparse gridder, static-sky subtract, RFI SK) on their *first* call.
    On a cold 2080 Ti that first block costs multiple seconds (Inductor
    cal-apply compile alone can be 10-60 s). Previously the ready
    sentinel was touched immediately *before* the read loop, so the
    orchestrator started the SNAP capture binaries while those first
    blocks were still compiling.

    ``fada`` is an ``r=2`` multi-reader ring (corr_slow + corr_fast),
    written by ``dsaX_merge`` which is in turn fed by the *non-blockable*
    SNAP UDP capture. While corr_fast (or corr_slow) is stalled in JIT,
    merge cannot retire fada pages, fada fills to 70/70, merge blocks on
    the write side, dada/eada back-fill, and the capture binary's UDP
    socket overflows -> **silent packet loss on sky**. Once the ring is
    pinned near full it only drains if the consumer runs *faster* than
    the SNAP cadence, which is why operators saw a ~20 min crawl back to
    a low-occupancy steady state (or never).

    Running a handful of dummy blocks here forces all compiles to happen
    *before* the sentinel, so when capture finally starts the consumer is
    already at steady-state speed and fada stays near empty.

    The pass is side-effect-free: transport TX and the stage-2 FIFO are
    swapped for no-ops (no zero-cubes hit the search nodes, no FIFO
    accumulation), the injector is detached, and every stateful detector
    (static-sky EMA, RFI warmup counter, multi-DM sliding window) is
    reset afterwards so real block #1 starts from a pristine cold state.
    """
    if n_blocks <= 0:
        return
    LOG.info(
        "kernel warmup: compiling fast-path kernels with %d dummy block(s) "
        "before ready-sentinel (avoids fada fill / UDP loss on cold start)",
        n_blocks,
    )
    t0 = time.monotonic()
    # Seeded pseudo-random fill: unpacks to varied non-zero int4 voltages
    # so the RFI spectral-kurtosis / variance kernels see non-degenerate
    # statistics (a constant fill can drive SK to 0/0). The values are
    # irrelevant -- all outputs are discarded -- we only want the kernels
    # to compile.
    dummy = np.random.default_rng(0xDADA).integers(
        0, 256, size=int(expected_fada_bytes), dtype=np.uint8,
    )

    saved_tx = ctx.transport_tx
    saved_fifo = ctx.stage2_fifo
    saved_injector = ctx.injector
    ctx.transport_tx = NoOpTransportTx()
    ctx.stage2_fifo = NoOpStage2Fifo()
    ctx.injector = None
    try:
        for i in range(int(n_blocks)):
            try:
                _ = process_block(dummy, ctx=ctx, block_n=i)
            except Exception:
                # Warmup must never sink startup: log and bail out of the
                # pass (the first real block will pay the JIT cost as it
                # did before this hook existed).
                LOG.exception(
                    "kernel warmup: dummy block %d raised; aborting warmup "
                    "(falling back to in-loop JIT)", i,
                )
                break
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
    finally:
        ctx.transport_tx = saved_tx
        ctx.stage2_fifo = saved_fifo
        ctx.injector = saved_injector
        # Reset all per-cube state so the real stream starts cold-clean.
        if ctx.static_sky is not None:
            ctx.static_sky.reset()
        if ctx.rfi_flagger is not None:
            ctx.rfi_flagger.reset_warmup()
        if ctx.multi_dm_coarse_dm is not None:
            ctx.multi_dm_coarse_dm._prev_vis_stokes_i = None
            ctx.multi_dm_coarse_dm._prev_block_n = -1
    LOG.info(
        "kernel warmup: done in %.1fs; fast-path kernels hot, detector "
        "state reset", time.monotonic() - t0,
    )


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
    dm_plan: DMPlan | None = None,
    use_pipeliner_3s: bool = False,
    profile_stages_every: int = 0,
    stage2_fifo_depth: int = COARSE_DM_FIFO_DEPTH_DEFAULT,
    stage2_mode: str = "uniform",
    transport_tx_host: str = "",
    transport_tx_port: int = 9000,
    transport_tx_mode: str = "chunk8",
    transport_tx_dtype: str = "cfp16",
    transport_tx_target_gbps_per_flow: float = 0.073,
    transport_tx_workers: int = 0,
    transport_tx_ring_slots: int = 8,
    transport_tx_coarse_dm_mask: int = 0xFF,
    transport_tx_worker_hosts: str = "",
    ready_sentinel_path: Path | None = None,
    # M7.6 RFI monitoring (None on rfi_mon_cn_id disables the path)
    rfi_mon_cn_id: int | None = None,
    rfi_mon_window_size: int | None = None,
    rfi_mon_freq_downsample: int | None = None,
    rfi_mon_shm_slots: int = 64,
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

        # ── M7.2 overlap path: real Stage2FIFO + TransportTx ─────────
        # When --transport-tx-host is set, replace the NoOp stubs with
        # the real :class:`Stage2FIFO` (M3 chunk 3b) + :class:`TransportTx`
        # (M3 chunk 8 + M4a chunk 2) so the per-block budget includes
        # the cross-chgroup timing FIFO and the cube-encode + sendto
        # syscall costs. Phase-0 measurements with the NoOp stubs
        # missed both of these N-scaling components.
        #
        # M7.2 production async path (--transport-tx-workers > 0):
        # the inline ``TransportTx.transmit`` blocks the GPU pipeline
        # for ~74 ms per cube at the N=8 op-point (Python encode +
        # sendto). Off-loading to N worker subprocesses keeps the
        # corr-side block budget at its M7.1 ceiling (~110 ms p50)
        # while still hitting the wire at production rate. Construction
        # happens AFTER build_context (n_grid + n_filled are needed
        # from gridder.pattern). Below we install Stage2FIFO eagerly
        # and replace ctx.transport_tx after build_context.
        use_async_tx = (
            bool(transport_tx_host)
            and transport_tx_workers > 0
            and transport_tx_mode == "prod"
        )
        # Option A (M7.4 follow-up): if the operator opts in via
        # ``--stage2-mode per_coarse_dm``, install the per-(chgroup,
        # coarse-DM) shift FIFO instead of the uniform-depth one. The
        # adapter constructs lazily on the first push so we don't need
        # to know ``t_dedisp`` here.
        def _install_per_coarse_dm_fifo() -> "_Stage2InterChgroupShiftAdapter":
            if dm_plan is None and cfg.dm_plan_path is None:
                raise ValueError(
                    "--stage2-mode=per_coarse_dm requires a DM plan "
                    "(set --dm-plan-path or pass dm_plan=... to run())"
                )
            local_plan = dm_plan
            if local_plan is None:
                from dsart.coarse_dm.dm_plan import load_dm_plan
                local_plan = load_dm_plan(str(cfg.dm_plan_path))
            from dsart.common.constants import NATIVE_SAMPLE_US
            return _Stage2InterChgroupShiftAdapter(
                chgroup=int(cfg.chgroup),
                coarse_dm_pc_cm3=local_plan.dm_pc_cc,
                t_int_corr_us=float(
                    cfg.t_int_fast_native * NATIVE_SAMPLE_US
                ),
            )

        use_per_coarse_dm_stage2 = stage2_mode == "per_coarse_dm"
        if stage2_mode not in ("uniform", "per_coarse_dm"):
            raise ValueError(
                f"stage2_mode={stage2_mode!r}; expected "
                f"'uniform' or 'per_coarse_dm'"
            )

        if use_async_tx:
            if stage2_fifo is None:
                if use_per_coarse_dm_stage2:
                    stage2_fifo = _install_per_coarse_dm_fifo()
                    LOG.info(
                        "M7.2 async TX + Option A: per-coarse-DM stage-2 "
                        "FIFO installed (chgroup=%d)",
                        cfg.chgroup,
                    )
                else:
                    stage2_fifo = _Stage2FIFOAdapter(depth=stage2_fifo_depth)
                    LOG.info(
                        "M7.2 async TX: real Stage2FIFO depth=%d (was NoOp)",
                        stage2_fifo_depth,
                    )
            if transport_tx is None:
                # Placeholder; real AsyncTransportTx built after
                # build_context() once gridder.pattern is finalised.
                transport_tx = NoOpTransportTx()
        elif transport_tx_host:
            if stage2_fifo is None:
                if use_per_coarse_dm_stage2:
                    stage2_fifo = _install_per_coarse_dm_fifo()
                    LOG.info(
                        "M7.2 overlap + Option A: per-coarse-DM stage-2 "
                        "FIFO installed (chgroup=%d)",
                        cfg.chgroup,
                    )
                else:
                    stage2_fifo = _Stage2FIFOAdapter(depth=stage2_fifo_depth)
                    LOG.info(
                        "M7.2 overlap: real Stage2FIFO ring depth=%d (was NoOp)",
                        stage2_fifo_depth,
                    )
            if transport_tx is None:
                if transport_tx_mode == "chunk8":
                    dtype_code = (
                        DTYPE_CFP16 if transport_tx_dtype == "cfp16"
                        else DTYPE_CINT8
                    )
                    real_tx = TransportTx(
                        host=transport_tx_host,
                        port=transport_tx_port,
                        chgroup=int(cfg.chgroup),
                        dtype_code=dtype_code,
                        use_prod_frame=False,
                    )
                    LOG.info(
                        "M7.2 overlap: real TransportTx (chunk8) → "
                        "%s:%d chgroup=%d dtype=%s (was NoOp)",
                        transport_tx_host, transport_tx_port,
                        cfg.chgroup, transport_tx_dtype,
                    )
                else:
                    # prod mode: 72-byte ProdFrame with cint8 + pacer.
                    # Requires prepare_prod() before any transmit; we
                    # call it after build_context where gridder.pattern
                    # is finalised so n_grid + pattern_id are known.
                    from dsart.transport.prod_frame import (
                        BITS_CINT8_COMPLEX,
                    )
                    from dsart.transport.tx import TransportTxProdConfig
                    prod_cfg = TransportTxProdConfig(
                        target_gbps_per_flow=(
                            transport_tx_target_gbps_per_flow
                        ),
                        bits_per_cell=BITS_CINT8_COMPLEX,
                        t_int_factor=1,
                        corr_idx=int(cfg.chgroup),
                    )
                    real_tx = TransportTx(
                        host=transport_tx_host,
                        port=transport_tx_port,
                        chgroup=int(cfg.chgroup),
                        use_prod_frame=True,
                        prod_config=prod_cfg,
                    )
                    LOG.info(
                        "M7.2 overlap: real TransportTx (prod) → "
                        "%s:%d chgroup=%d gbps/flow=%.3f (was NoOp)",
                        transport_tx_host, transport_tx_port,
                        cfg.chgroup, transport_tx_target_gbps_per_flow,
                    )
                transport_tx = _TransportTxAdapter(real_tx)

        ctx = build_context(
            cfg, device=device,
            antpos_e=antpos_e, antpos_n=antpos_n,
            is_core_baseline_mask=core_mask,
            coarse_dm=coarse_dm,
            stage2_fifo=stage2_fifo,
            transport_tx=transport_tx,
            dm_plan=dm_plan,
        )

        # ── M7.4 Phase 6 runtime: per-chgroup inject watch ──────────────
        # When ``--inject-watch`` was passed (cfg.inject_watch_enabled),
        # build_context guarantees ctx.injector is a real OnlineInjector;
        # we open a DsaStore watch on /cmd/dsart/corr/<chgroup>/inject
        # so the dashboard's Control tab (or any operator) can push
        # ``{cmd: "inject", val: <InjectionConfig dict>}`` PUTs into
        # the running service without a restart.
        inject_watch: RuntimeInjectWatch | None = None
        mon_publisher: "CorrFastMonPublisher | None" = None
        if cfg.inject_watch_enabled:
            if ctx.injector is None:
                raise RuntimeError(
                    "inject_watch_enabled is True but build_context did "
                    "not build an OnlineInjector (this is a bug — both "
                    "should imply the other)"
                )
            inject_watch = RuntimeInjectWatch(
                injector=ctx.injector, chgroup=int(cfg.chgroup),
            )
            inject_watch.start()
            # M7.4 Phase 6c: publish corr_fast service-start epoch to
            # /mon/corr_rt/<chgroup>/corr_fast so the dashboard's Control
            # tab "Send injection" can derive apply_at_specnum in the
            # SAME epoch the hot path uses (block_n × NPACKETS_PER_BLOCK).
            # Fixes Bug 1 from the 2026-05-28 E2E.
            from dsart.services.corr_fast_mon import CorrFastMonPublisher
            mon_publisher = CorrFastMonPublisher(
                chgroup=int(cfg.chgroup),
                npackets_per_block=NPACKETS_PER_BLOCK,
            )

        # ── M7.2 production async TX path ─────────────────────────────
        # Spawn AsyncTransportTx workers now that gridder.pattern is
        # finalised. The async path is mandatory for production: the
        # inline TransportTx blocks the GPU pipeline by ~74 ms per cube
        # at the N=8 op-point (Python encode + sendto). Off-loading to
        # subprocesses keeps the corr-side block budget at its M7.1
        # ceiling (~110 ms p50) while still hitting the wire at
        # production rate; TX latency overlaps into the *next* block's
        # GPU compute.
        async_tx: AsyncTransportTx | None = None
        if use_async_tx:
            from dsart.grid.sparsity_pattern import predict_pattern_id
            pat = ctx.gridder.pattern  # SparsityPattern
            n_filled = int(pat.n_filled)
            n_grid_eff = int(pat.n_grid)
            # n_dm_total: prefer the runtime Stage1MultiDMCoarseDM,
            # since the DMPlan may have come from cfg.dm_plan_path
            # (loaded inside build_context, not via the dm_plan arg).
            if ctx.multi_dm_coarse_dm is not None:
                n_dm_total = int(ctx.multi_dm_coarse_dm.n_dm)
            elif dm_plan is not None:
                n_dm_total = int(dm_plan.n_coarse)
            else:
                raise RuntimeError(
                    "AsyncTransportTx: no DMPlan available "
                    "(neither cfg.dm_plan_path nor dm_plan arg)"
                )
            if n_dm_total < transport_tx_workers:
                raise ValueError(
                    f"--transport-tx-workers={transport_tx_workers} > "
                    f"n_dm_total={n_dm_total}; each worker needs >= 1 DM trial"
                )
            n_dm_per_worker = (
                n_dm_total + transport_tx_workers - 1
            ) // transport_tx_workers
            # Upper bound on n_fast_vis for the ring slot. Worst case
            # is no dedispersion-shift loss; t_int_fast_native=32 →
            # n_fv_max = BLOCK_SAMPLES_NATIVE / t_int_fast_native = 128.
            n_fast_vis_max = (
                BLOCK_SAMPLES_NATIVE // max(1, cfg.t_int_fast_native)
            )
            ring_dims = CubeShmRingDims(
                n_slots=transport_tx_ring_slots,
                shape=(n_dm_per_worker, n_fast_vis_max, n_filled),
                dtype=np.dtype("complex64"),
            )
            # Compute pattern_id from the actual pattern. For loopback
            # bench the search side is a drain, but for end-to-end
            # M7.2 the search node validates the pattern_id against
            # its own SparsityPattern. The predict_pattern_id call is
            # cheap (no full pattern build).
            try:
                pattern_id = int(predict_pattern_id(
                    chgroup=int(cfg.chgroup),
                    dec_deg=float(np.degrees(cfg.obs_dec_rad)),
                    n_grid=n_grid_eff,
                    kernel_support=int(cfg.kernel_support),
                    chan_sum_factor=int(cfg.chan_sum_factor),
                    antpos_e=antpos_e,
                    antpos_n=antpos_n,
                    is_core_baseline_mask=core_mask,
                )) & 0xFFFF_FFFF_FFFF_FFFF
            except Exception:
                LOG.warning(
                    "predict_pattern_id failed; falling back to 0 "
                    "(loopback-only; production e2e requires matching "
                    "pattern_id on the search side)",
                    exc_info=True,
                )
                pattern_id = 0

            # M7.3 (2026-05-20): parse comma-separated per-worker
            # destination hosts. Empty string keeps the legacy
            # single-host behaviour (host:port + w per worker).
            worker_hosts_list: list[str] | None = None
            if transport_tx_worker_hosts:
                worker_hosts_list = [
                    h.strip() for h in transport_tx_worker_hosts.split(",")
                    if h.strip()
                ]
                if len(worker_hosts_list) != int(transport_tx_workers):
                    raise ValueError(
                        f"--transport-tx-worker-hosts has "
                        f"{len(worker_hosts_list)} entries; expected "
                        f"--transport-tx-workers={transport_tx_workers}"
                    )
            async_cfg = AsyncTransportTxConfig(
                host=transport_tx_host,
                port=transport_tx_port,
                worker_hosts=worker_hosts_list,
                chgroup=int(cfg.chgroup),
                n_workers=int(transport_tx_workers),
                n_dm_total=int(n_dm_total),
                ring_dims=ring_dims,
                pattern_id=pattern_id,
                n_grid=n_grid_eff,
                target_gbps_per_flow=float(transport_tx_target_gbps_per_flow),
                corr_idx=int(cfg.chgroup),
                log_level="INFO",
                shm_name_prefix=f"dsart-corr-tx-{cfg.chgroup}",
                coarse_dm_mask=int(transport_tx_coarse_dm_mask) & 0xFF,
            )
            async_tx = AsyncTransportTx.spawn(async_cfg)
            ctx.transport_tx = _AsyncTransportTxAdapter(async_tx)
            LOG.info(
                "M7.2 async TX spawned: n_workers=%d n_dm_total=%d "
                "dm_per_worker=%d ring_slots=%d n_fv_max=%d n_filled=%d "
                "n_grid=%d pattern_id=0x%x host=%s:%d gbps/flow=%.3f",
                transport_tx_workers, n_dm_total, n_dm_per_worker,
                transport_tx_ring_slots, n_fast_vis_max, n_filled,
                n_grid_eff, pattern_id, transport_tx_host,
                transport_tx_port, transport_tx_target_gbps_per_flow,
            )

        # M7.2 prod-mode prepare_prod: needs gridder.sparsity_pattern
        # which is finalised inside build_context. Call here after
        # context build so n_grid + pattern_id are available.
        if (
            transport_tx_host
            and transport_tx_mode == "prod"
            and isinstance(transport_tx, _TransportTxAdapter)
            and transport_tx.tx.use_prod_frame
        ):
            # M3 sparsity_pattern.n_grid is the n_grid of the cached
            # pattern; the pattern_id is a stable hash that the search
            # side will validate against its own cached pattern. For
            # the M7.2 corr-only benchmark we don't have a search-side
            # validator, so the pattern_id is fixed at 0 (the cube is
            # opaque on loopback). Production wiring (M7.2.8 corner-
            # turn) will source pattern_id from the search node via
            # /cnf/search/* — out of scope for the Phase 0/A bench.
            pat = ctx.gridder.pattern  # SparsityPattern
            transport_tx.tx.prepare_prod(
                pattern_id_by_chgroup={int(cfg.chgroup): 0},
                n_grid=int(pat.n_grid),
            )
            LOG.info(
                "M7.2 overlap: TransportTx.prepare_prod(n_grid=%d, "
                "pattern_id=0)",
                int(pat.n_grid),
            )
        # M7.2 Phase 0 diagnostic: optional per-stage CUDA-event timing.
        # Disabled by default (zero hot-path overhead). Incompatible
        # with the 3-stream pipeliner (which would record events on
        # different streams; the elapsed_time math would be wrong) —
        # mutual exclusion enforced below.
        if profile_stages_every > 0:
            if use_pipeliner_3s:
                raise ValueError(
                    "--profile-stages-every is incompatible with "
                    "--pipeliner-3s (events on different streams have "
                    "no meaningful elapsed_time)."
                )
            ctx.profiler = StageProfiler(every=profile_stages_every)
            LOG.info(
                "StageProfiler enabled: emitting per-stage means every "
                "%d blocks", profile_stages_every,
            )

        per_block_ms: list[float] = []
        per_block_flag_frac: list[float] = []
        t_start = time.monotonic()

        # ── M7.6 RFI window aggregator + shm writer ───────────────────
        # Only constructed when RFI is enabled AND a cn_id was provided
        # (set when dsart_rt spawns the routine; in standalone bench
        # mode rfi_mon_cn_id may be None and we skip the export path).
        # The aggregator accumulates one 16-cube window at the
        # production cadence (≈ 2.147 s of voltages); on each window
        # close it publishes a record to /dev/shm/dsart-rfi-window-<cn>
        # which the rfi_monitor_export sidecar reads and serves over
        # HTTP for the h23 monitoring dashboard. Negligible hot-path
        # cost (~0.5 ms / cube).
        rfi_aggregator: "RFIWindowAggregator | None" = None
        rfi_shm_writer: "RFIMonShmWriter | None" = None
        if ctx.rfi_flagger is not None and rfi_mon_cn_id is not None:
            from dsart.services.rfi_window import (
                FREQ_DOWNSAMPLE_DEFAULT,
                RFIWindowAggregator,
                WINDOW_SIZE_DEFAULT,
            )
            from dsart.services.rfi_mon_shm import RFIMonShmWriter

            agg_window = int(rfi_mon_window_size or WINDOW_SIZE_DEFAULT)
            agg_ds = int(rfi_mon_freq_downsample or FREQ_DOWNSAMPLE_DEFAULT)
            rfi_aggregator = RFIWindowAggregator(
                n_ants=NANTS,
                n_chan=NCHAN_PER_CHGROUP,
                n_pol=NPOL,
                window_size=agg_window,
                freq_downsample=agg_ds,
                device=device,
            )
            rfi_shm_writer = RFIMonShmWriter(
                cn_id=int(rfi_mon_cn_id),
                n_ants=NANTS,
                n_chan_ds=rfi_aggregator.n_chan_ds,
                n_pol=NPOL,
                window_size=agg_window,
                freq_downsample=agg_ds,
                n_slots=int(rfi_mon_shm_slots),
            )
            LOG.info(
                "M7.6 RFI monitor: aggregator window=%d cubes (~%.2f s), "
                "freq_downsample=%d × (NCHAN=%d -> %d), shm=%s slots=%d",
                agg_window,
                agg_window * (NATIVE_SAMPLE_US * 2048 * 1e-6),
                agg_ds, NCHAN_PER_CHGROUP, rfi_aggregator.n_chan_ds,
                f"/dev/shm/dsart-rfi-window-{rfi_mon_cn_id}",
                int(rfi_mon_shm_slots),
            )
        elif ctx.rfi_flagger is not None:
            LOG.info(
                "M7.6 RFI monitor: disabled (no --rfi-mon-cn-id supplied)"
            )

        # ── 3-stream pipeliner (M7.1 RT path) ─────────────────────────
        # When enabled, overlaps unpack (Stream U) || compute_split +
        # RFI + cal (Stream C) || dedisp + grid + TX (Stream D). The
        # steady-state wall per push() is max(U, C, D) instead of the
        # sum. Per-block latency increases by 2 × block_period (since
        # push(N) returns the result for block N-3), but throughput
        # drops to the long pole — the relevant metric for RT cadence.
        # See :class:`BlockPipeliner3S` for the implementation details.
        pipeliner: BlockPipeliner3S | None = None
        if use_pipeliner_3s:
            if device.type != "cuda":
                raise ValueError(
                    f"use_pipeliner_3s requires CUDA; got device={device}"
                )
            pipeliner = BlockPipeliner3S(ctx, n_buffers=3)
            LOG.info(
                "BlockPipeliner3S enabled (n_buffers=3); 3 streams "
                "U||C||D with 3-block result latency"
            )

        def _consume_completed(
            n_out: int, out: "IntegrationOutput | None",
        ) -> None:
            """Bookkeeping for a block whose pipeline pass just finished.

            n_out is the 1-based block number (mirroring n_in for the
            sequential path). out is the IntegrationOutput returned by
            either ``process_block`` or ``pipeliner.push``.
            """
            nonlocal n_tx_total, n_processed
            if out is None:
                return
            n_tx_total += out.n_tx
            if blocks_output_mode != "none":
                _serialise_block(
                    output_dir, n_out,
                    out=out, blocks_output_mode=blocks_output_mode,
                )
            n_processed += 1
            if out.rfi is not None:
                per_block_flag_frac.append(
                    float(out.rfi.flag_fraction_total)
                )

            # M7.6: feed the RFI window aggregator + shm publisher.
            # Only active when both --rfi-mon-cn-id was supplied (so
            # rfi_aggregator + rfi_shm_writer are non-None) and the
            # flagger actually returned an s1_full (it does on the
            # production code path; tests with autos_override that
            # omit the full-block M may set None — skip then).
            if (
                rfi_aggregator is not None
                and rfi_shm_writer is not None
                and out.rfi is not None
                and out.rfi.s1_full is not None
            ):
                try:
                    window = rfi_aggregator.push(
                        s1_full=out.rfi.s1_full,
                        mask=out.rfi.mask,
                        source_tags=out.rfi.source_tags,
                        block_n=int(n_out),
                        warmup=bool(out.rfi.warmup),
                    )
                    if window is not None:
                        rfi_shm_writer.publish(window)
                except Exception:
                    # Monitoring should never sink the hot path. Log
                    # once-per-cube and continue; ops can drain the
                    # shm independently.
                    LOG.exception(
                        "M7.6 rfi monitor publish failed (continuing)"
                    )

        # M7.4 Phase 8b (2026-05-28): pre-compile the fast-path Triton /
        # Inductor kernels on dummy blocks BEFORE touching the ready
        # sentinel. The orchestrator gates the SNAP capture binaries on
        # this sentinel, so by warming the kernels here the consumer is
        # already at steady-state speed when capture starts -> fada never
        # fills to 70/70 -> no merge back-pressure -> no UDP packet loss.
        # See :func:`_warmup_pipeline_jit` for the full rationale.
        # Tunable via DSART_FAST_WARMUP_BLOCKS (0 disables; default 4).
        try:
            _warmup_n = int(os.environ.get("DSART_FAST_WARMUP_BLOCKS", "4"))
        except (TypeError, ValueError):
            _warmup_n = 4
        # Publish "warming" so the dashboard shows PREPARING (not safe to
        # arm yet) for the full JIT window, then "ready" once kernels are
        # hot so the operator knows it is safe to utc_start.
        _publish_corr_fast_ready(rfi_mon_cn_id, ready=False, n_blocks=_warmup_n)
        _warmup_t0 = time.time()
        _warmup_pipeline_jit(
            ctx, expected_fada_bytes=expected_fada_bytes, n_blocks=_warmup_n,
        )
        _publish_corr_fast_ready(
            rfi_mon_cn_id, ready=True,
            warmup_s=time.time() - _warmup_t0, n_blocks=_warmup_n,
        )

        # M7.2 (2026-05-19) ready-sentinel hook: signal the orchestrator
        # that Python imports + GPU init + Triton modules import + cal
        # loading + DM plan loading + pipeliner construction are all
        # complete. As of Phase 8b the fast-path kernels are also already
        # JIT-compiled (see _warmup_pipeline_jit above), so the first real
        # block runs at steady-state speed instead of paying the compile.
        # The orchestrator gates capture routines (cap_a_real/cap_b_real,
        # dada_junkdb) on this file so they don't pre-fill dada/eada (and
        # hence fada) during this process's cold start.
        if ready_sentinel_path is not None:
            try:
                ready_sentinel_path.parent.mkdir(parents=True, exist_ok=True)
                ready_sentinel_path.touch()
                LOG.info("ready sentinel touched: %s", ready_sentinel_path)
            except OSError as e:
                LOG.warning(
                    "failed to touch ready sentinel %s: %s "
                    "(continuing without gate)",
                    ready_sentinel_path, e,
                )

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

            if pipeliner is not None:
                # push() returns the IntegrationOutput for the block
                # submitted ``n_buffers=3`` calls ago, or None during
                # the first 3 pushes (warmup). The block number it
                # corresponds to is ``n_in - 3`` for an in-order
                # submission stream.
                prior_out = pipeliner.push(page_arr, block_n=n_in)
                # We must hold the page_arr buffer at least until
                # Stream U's H2D copy completes; the pipeliner takes
                # care of stream-event ordering, but PSRDADA's
                # markCleared() releases the underlying ring page,
                # which is fine because torch.as_tensor(raw, device=GPU)
                # always issues a copy (verified upstream).
                reader.markCleared()
                _consume_completed(n_in - 3, prior_out)
                del prior_out
            else:
                out = process_block(page_arr, ctx=ctx, block_n=n_in)
                reader.markCleared()
                _consume_completed(n_in, out)
                del out

            # M7.2 Phase 0 diagnostic: flush this block's per-stage
            # CUDA events into the rolling histogram. Disabled (None)
            # in the default config; under --pipeliner-3s the profiler
            # is explicitly forbidden (see context build).
            if ctx.profiler is not None:
                ctx.profiler.commit_block()

            t_block_end = time.monotonic()
            per_block_ms.append((t_block_end - t_block_start) * 1000.0)

            if n_in % 16 == 0:
                LOG.info(
                    "processed n_in=%d n_processed=%d n_drop=%d n_tx=%d "
                    "last_block=%.1fms",
                    n_in, n_processed, n_drop, n_tx_total, per_block_ms[-1],
                )
                if mon_publisher is not None:
                    mon_publisher.publish(
                        block_n=n_in,
                        n_processed=n_processed,
                        n_drop=n_drop,
                        n_tx=n_tx_total,
                        last_block_ms=per_block_ms[-1],
                    )

            if max_blocks is not None and n_in >= max_blocks:
                LOG.info("hit --max-blocks=%d; stopping", max_blocks)
                break
            if reader.isEndOfData:
                LOG.info("fada EOD; loop done")
                break

        if pipeliner is not None:
            # Drain the 3-stream pipeline: this returns up to ``n_buffers``
            # blocks still in flight (the last 3 pushed).
            drained = pipeliner.flush()
            pipeliner.close()
            LOG.info(
                "BlockPipeliner3S drained %d in-flight blocks", len(drained),
            )
            # The drained outputs correspond to blocks (n_in - 2) .. n_in
            # in submission order; emit them through _consume_completed
            # so per-block bookkeeping stays consistent.
            first_drained_n = n_in - len(drained) + 1
            for i, out in enumerate(drained):
                _consume_completed(first_drained_n + i, out)

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
        # M7.4 Phase 6: cancel the runtime inject watch (idempotent).
        _inject_watch = locals().get("inject_watch")
        if _inject_watch is not None:
            try:
                _inject_watch.stop()
            except Exception:
                LOG.exception(
                    "inject_watch.stop failed (non-fatal)"
                )
        # M7.2 async TX: clean shutdown of worker subprocesses + shm.
        # ``async_tx`` is defined only inside the ``try`` block above
        # (after build_context); guard with locals() since the
        # ``finally`` clause runs even if we never reached the wiring.
        _async_tx = locals().get("async_tx")
        if _async_tx is not None:
            try:
                _async_tx.close()
            except Exception:
                LOG.exception("async_tx.close failed (non-fatal)")
        # M7.6 RFI monitor: close + unlink the shm. Leaving the segment
        # behind would confuse the sidecar + h23 dashboard on the next
        # spawn (stale writer pid, stale magic if ABI ever bumps).
        _shm_writer = locals().get("rfi_shm_writer")
        if _shm_writer is not None:
            try:
                _shm_writer.close()
                _shm_writer.unlink()
            except Exception:
                LOG.exception(
                    "rfi_shm_writer close/unlink failed (non-fatal)"
                )


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
                        "smoke runs; production: pass 0 = unlimited, "
                        "run until fada EOD).")
    p.add_argument("--t-int-fast-native", type=int, default=T_INT_FAST_NATIVE,
                   help="fast-corr integration depth in NATIVE samples per "
                        "fast-vis tile. Default: %d (= %d µs cadence). "
                        "Burst-test override: 32 (= 1048.576 µs, 4× cadence)."
                        % (T_INT_FAST_NATIVE, int(T_INT_FAST_NATIVE * NATIVE_SAMPLE_US)))
    p.add_argument("--n-coarse-dm", type=int, default=0,
                   help="If > 0, build a SYNTHETIC summed-channel DMPlan "
                        "with this many coarse trials linearly spaced from "
                        "0 to --dm-max and enable the F25 multi-DM-trial "
                        "stage1 → gridder → dedispersion path. Use 5 with "
                        "--t-int-fast-native 32 for the realtime-feasible "
                        "x32 op-point; 24 with --t-int-fast-native 8 for "
                        "the production O-4 op-point. Default: 0 (legacy "
                        "single-DM path; no dedispersion).")
    p.add_argument("--dm-max", type=float, default=1000.0,
                   help="dm_max (pc/cc) for the synthetic --n-coarse-dm "
                        "plan. Ignored when --n-coarse-dm = 0. Default: "
                        "1000 (covers the M3 FRB injection range without "
                        "the extreme-DM smearing the x8 path was sized "
                        "against).")
    p.add_argument("--dm-plan-path", type=Path, default=None,
                   help="Path to a canonical DmPlan ``.npz`` built by "
                        "tools/build_dm_plan.py. When set, the F25 multi-"
                        "DM-trial path is enabled and the plan supersedes "
                        "any --n-coarse-dm synthetic plan (which is the "
                        "M7.1 shortcut). Use this for M7.2+ where the "
                        "coarse-DM trial list must come from the legacy "
                        "gen_dmtrials recursion (not a linear stand-in). "
                        "The plan's t_int_fast_us is pinned to "
                        "--t-int-fast-native × NATIVE_SAMPLE_US at "
                        "load-time; mismatch raises.")
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
    # ---- M7.4 Phase 6: voltage-domain online signal injection ----
    p.add_argument("--inject-spec",
                   action="append", default=[],
                   metavar="JSON",
                   help=("M7.4 Phase 6: queue a voltage-domain "
                         "dispersed-pulse injection. Pass a JSON dict "
                         "with the six InjectionConfig fields: "
                         "{\"inj_id\": str, \"l_rad\": float, "
                         "\"m_rad\": float, \"dm_pc_cm3\": float, "
                         "\"fluence_jy_ms\": float, \"width_samples\": int, "
                         "\"profile\": \"gaussian\"|\"boxcar\", "
                         "\"apply_at_specnum\": int}. Repeatable: pass "
                         "--inject-spec multiple times to queue "
                         "multiple injections. The native-sample peak "
                         "lands at block_n=apply_at_specnum/2048 from "
                         "service start (1 block = 2048 specnums = "
                         "4096 native samples). Default: no injection."))
    p.add_argument("--inject-watch", action="store_true",
                   help=("M7.4 Phase 6 runtime: subscribe to the etcd "
                         "key /cmd/dsart/corr/<chgroup>/inject (DsaStore "
                         "watch) and route any "
                         "{cmd: \"inject\", val: <InjectionConfig>} "
                         "writes into the live OnlineInjector via "
                         "add_pending. This is what the dashboard's "
                         "Control-tab \"Send injection\" button drives. "
                         "Always builds the injector even when "
                         "--inject-spec is empty. Default off."))
    # ---- RFI tuning knobs (M7.6) ----
    # All optional; when omitted the library defaults from dsart.rfi.*
    # are used. The defaults live in dsart.rfi.{sk, bandpass_outlier,
    # group_outlier, sum_threshold, autos} and combine.RFIFlagger.
    p.add_argument("--sk-far", type=float, default=None,
                   help="SK two-sided per-(M, cell) false-alarm rate "
                        "(library default: 1e-4)")
    p.add_argument("--bandpass-k", type=float, default=None,
                   help="bandpass-outlier MAD-sigma threshold "
                        "(library default: 5.0)")
    p.add_argument("--group-k", type=float, default=None,
                   help="group-outlier MAD-sigma threshold across ants "
                        "(library default: 5.0)")
    p.add_argument("--sumthr-max-m", type=int, default=None,
                   help="SumThreshold maximum dilation window (must be a "
                        "power of two; library default: 8)")
    p.add_argument("--sumthr-eta", type=float, default=None,
                   help="SumThreshold shape parameter eta "
                        "(library default: 1.5)")
    p.add_argument("--sumthr-disabled", action="store_true",
                   help="disable the SumThreshold post-pass (SK + "
                        "bandpass + group + flagants still active)")
    p.add_argument("--rfi-m-values", type=str, default=None,
                   help="comma-separated SK accumulation depths (must be "
                        "divisors of 4096; library default: 64,256,1024,4096)")
    p.add_argument("--rfi-warmup-cubes", type=int, default=None,
                   help="cold-start window length (cubes) during which "
                        "bandpass-outlier is bypassed and rfi_warming_up "
                        "is asserted in the transport header. Library "
                        "default ~1118 cubes (~150 s).")
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
    p.add_argument("--pipeliner-3s", action="store_true",
                   help="enable BlockPipeliner3S: 3-stream overlap of "
                        "unpack || compute || dedisp. Reduces per-block "
                        "wall to max(U,C,D) instead of U+C+D. Adds a "
                        "3-block result latency (output for block N "
                        "returned by push(N+3)). Requires CUDA.")
    p.add_argument("--profile-stages-every", type=int, default=0,
                   help="M7.2 Phase 0 diagnostic. When > 0, brackets "
                        "corr_phase / consume_phase / multi_dm / "
                        "static_sky / stage2_tx with paired CUDA "
                        "events and emits a mean+p50+p99 ms per stage "
                        "every N blocks. Default 0 = OFF (zero hot-"
                        "path overhead). Use 64 to get a per-stage "
                        "breakdown every ~8.6s at the current op-point.")
    # ── M7.2 overlap path: real Stage2FIFO + TransportTx ─────────────
    p.add_argument("--stage2-fifo-depth", type=int,
                   default=COARSE_DM_FIFO_DEPTH_DEFAULT,
                   help=("M7.2: depth (in cubes) of the real "
                         "Stage2FIFO ring. Only used when "
                         "--transport-tx-host is set and "
                         "--stage2-mode=uniform (default). "
                         f"Default {COARSE_DM_FIFO_DEPTH_DEFAULT} "
                         "(COARSE_DM_FIFO_DEPTH_DEFAULT). The FIFO "
                         "is the timing buffer for cross-chgroup "
                         "alignment per plan §3.6.2; the per-(g, c) "
                         "delay budget at the current DM plan must "
                         "fit in `depth` cubes."))
    p.add_argument(
        "--stage2-mode",
        type=str,
        default="uniform",
        choices=("uniform", "per_coarse_dm"),
        help=(
            "M7.4 Option A: select the stage-2 inter-chgroup time-"
            "alignment FIFO implementation.\n"
            "  uniform        : legacy K-deep ring (Stage2FIFO); the "
            "search side must absorb the per-chgroup offset via "
            "compute_time_shift_search(include_coarse_offset=True). "
            "Default.\n"
            "  per_coarse_dm  : per-(chgroup, coarse-DM) sample-exact "
            "shift FIFO (Stage2InterChgroupShiftFifo). The search "
            "side must run with include_coarse_offset=False; "
            "reclaims ~50%% of the search-side rx-ring t_buf. "
            "Requires --dm-plan-path."
        ),
    )
    p.add_argument("--transport-tx-host", type=str, default="",
                   help=("M7.2: when set, enables the real Stage2FIFO + "
                         "TransportTx path (replaces the NoOp stubs). "
                         "Empty (default) keeps the M3 / Phase-0 "
                         "behaviour (no TX). Use 127.0.0.1 for "
                         "loopback benchmarks (the kernel drops the "
                         "UDP packets if no listener is bound, but "
                         "the sendto syscall + encoding cost is "
                         "still paid)."))
    p.add_argument("--transport-tx-port",
                   "--transport-tx-base-port",
                   dest="transport_tx_port",
                   type=int, default=9000,
                   help="M7.2: BASE UDP destination port for the worker "
                        "pool. Worker w sends to host:(port + w). The "
                        "per-worker offset lets the search side identify "
                        "which corr-worker each frame originates from "
                        "via the L4 source port. Default 9000. "
                        "Production (M7.3 + M7.2 plan): 6625 (workers "
                        "0..3 → 6625..6628). The `--transport-tx-base-port` "
                        "spelling is the canonical form; the legacy "
                        "`--transport-tx-port` spelling is preserved.")
    p.add_argument("--transport-tx-mode",
                   choices=("chunk8", "prod"), default="chunk8",
                   help=("M7.2: TransportTx wire format. "
                         "'chunk8' = simple 32-byte FastVisFrame, "
                         "one sendto per (dm_idx, t_idx), no pacing "
                         "(loopback bench path). 'prod' = M4a 72-byte "
                         "ProdFrame with cint8 quantisation + MTU "
                         "fragmentation + per-dm_idx token bucket "
                         "(production path). 'prod' requires "
                         "--transport-tx-target-gbps-per-flow > 0 "
                         "and uses block_n as specnum."))
    p.add_argument("--transport-tx-dtype",
                   choices=("cfp16", "cint8"), default="cfp16",
                   help=("M7.2: payload encoding for chunk8 mode. "
                         "Ignored in prod mode (which uses cint8 "
                         "always per plan §9). Default cfp16."))
    p.add_argument("--transport-tx-target-gbps-per-flow",
                   type=float, default=0.073,
                   help=("M7.2: per-dm_idx target rate for the prod "
                         "token-bucket pacer (Gbps). Default 0.073 "
                         "per plan §4.3 line 1447 (6 DM trials per "
                         "corr-search pair × 0.073 ≈ 0.44 Gbps per "
                         "pair). Ignored in chunk8 mode."))
    p.add_argument("--transport-tx-workers", type=int, default=0,
                   help=("M7.2 PRODUCTION async TX: off-load encode + "
                         "sendto to N worker subprocesses so the GPU "
                         "pipeline thread is never blocked on TX. "
                         "Each worker handles a contiguous DM-axis "
                         "slice of the cube. Production default = 4 "
                         "(4-way DM split, forward-compatible to "
                         "M7.3 fan-out across 4 search nodes). 0 = "
                         "inline synchronous TX (M3 chunk-8 + M4a "
                         "chunk-2 behaviour, blows the RT budget at "
                         "N=8 — debug only). Requires --transport-tx-mode "
                         "= prod."))
    p.add_argument("--transport-tx-ring-slots", type=int, default=8,
                   help=("M7.2: depth of each per-worker TX cube ring "
                         "(POSIX shm). 8 slots × 134 ms cadence ≈ "
                         "1 s of cube buffering; covers worker "
                         "startup jitter and burstiness. Only used "
                         "when --transport-tx-workers > 0."))
    p.add_argument("--transport-tx-coarse-dm-mask",
                   type=lambda s: int(s, 0), default=0xFF,
                   help=("M7.2 selective TX mask over coarse-DM "
                         "indices (LSB = coarse_dm[0]). Workers whose "
                         "DM-slice is ENTIRELY OUT of the mask drain "
                         "without transmitting; workers whose slice is "
                         "ENTIRELY IN transmit normally. Default 0xFF "
                         "(M7.3 production = all 8 coarse DMs sent). "
                         "M7.2-low: 0x03 (coarse 0,1 to n01). "
                         "M7.2-high: 0xC0 (coarse 6,7 to n01). Mask "
                         "must align with worker DM-slice boundaries "
                         "(typically n_workers=4, N=8 ⇒ workers cover "
                         "DM[0:2),[2:4),[4:6),[6:8) so 0x03 / 0x0C / "
                         "0x30 / 0xC0 are valid; 0x05 etc. would be "
                         "rejected). Accepts 0x / 0o / decimal."))
    p.add_argument("--transport-tx-worker-hosts", type=str, default="",
                   help=("M7.3: comma-separated list of destination "
                         "IPs/hostnames — one per --transport-tx-workers "
                         "worker. When set, worker w sends to "
                         "worker_hosts[w]:<port> (no per-worker port "
                         "offset; each search node binds to the same "
                         "base port). Length MUST equal "
                         "--transport-tx-workers. Empty string (the "
                         "default) keeps the M7.2 16x1 behaviour where "
                         "all workers share --transport-tx-host and the "
                         "port offset (host:port + w) disambiguates. "
                         "Production 16x4 example: "
                         "'10.41.0.205,10.41.0.222,10.41.0.253,10.41.0.238' "
                         "with --transport-tx-workers 4."))
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--ready-sentinel-path", type=Path, default=None,
                   help="M7.2: touch this file once Python imports, GPU "
                        "init, cal loading, DM plan loading, and pipeliner "
                        "construction are complete (just before the main "
                        "loop). The dsart_rt orchestrator gates capture "
                        "routines (dada_junkdb) on this file so they do "
                        "not stuff the dada/eada rings during this "
                        "process's multi-second cold start (~90 s on a "
                        "2080Ti including Triton module imports).")
    # ---- M7.6 RFI monitoring (window aggregator + shm publisher) ----
    p.add_argument("--rfi-mon-cn-id", type=int, default=None,
                   help="When set, enable the M7.6 16-cube RFI window "
                        "aggregator and publish records to "
                        "/dev/shm/dsart-rfi-window-<cn-id>. The sidecar "
                        "rfi_monitor_export reads this shm to serve "
                        "the h23 dashboard. Omit (default) to disable.")
    p.add_argument("--rfi-mon-window-size", type=int, default=None,
                   help="cubes per RFI monitor window (default: 16 = "
                        "~2.147 s of voltages)")
    p.add_argument("--rfi-mon-freq-downsample", type=int, default=None,
                   help="channel downsample factor in the RFI monitor "
                        "shm (default: 4 -> 96 ch/chgroup at "
                        "NCHAN_PER_CHGROUP=384). Must divide "
                        "NCHAN_PER_CHGROUP.")
    p.add_argument("--rfi-mon-shm-slots", type=int, default=64,
                   help="ring depth (number of window slots) in the "
                        "RFI monitor shm. Default 64 = ~137 s of "
                        "history; the h23 service polls fast enough "
                        "that this is just the consumer's lag budget.")
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

    if args.dm_plan_path is not None and args.n_coarse_dm > 0:
        LOG.error(
            "--dm-plan-path %s and --n-coarse-dm %d are mutually exclusive "
            "(the canonical plan supersedes the synthetic linear shortcut). "
            "Drop one.",
            args.dm_plan_path, args.n_coarse_dm,
        )
        return 2
    if args.dm_plan_path is not None and not args.dm_plan_path.is_file():
        LOG.error("--dm-plan-path %s not found", args.dm_plan_path)
        return 2

    # ---- RFI knob parsing (M7.6) ----
    rfi_m_values_parsed: tuple[int, ...] | None = None
    if getattr(args, "rfi_m_values", None) is not None:
        rfi_m_values_parsed = tuple(
            int(s) for s in args.rfi_m_values.split(",") if s
        )
        if not rfi_m_values_parsed:
            LOG.error("--rfi-m-values must be non-empty after parsing")
            return 2

    # ---- M7.4 Phase 6: --inject-spec parsing ----
    inject_configs: tuple[InjectionConfig, ...] = ()
    if getattr(args, "inject_spec", None):
        parsed = []
        for spec in args.inject_spec:
            try:
                parsed.append(InjectionConfig.from_json(spec))
            except ValueError as exc:
                LOG.error("--inject-spec %r: %s", spec, exc)
                return 2
        inject_configs = tuple(parsed)
        LOG.info(
            "M7.4 Phase 6: %d injection(s) queued from --inject-spec",
            len(inject_configs),
        )

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
        rfi_sk_far=args.sk_far,
        rfi_bandpass_k=args.bandpass_k,
        rfi_group_k=args.group_k,
        rfi_sumthr_max_m=args.sumthr_max_m,
        rfi_sumthr_eta=args.sumthr_eta,
        rfi_m_values=rfi_m_values_parsed,
        rfi_warmup_cubes=args.rfi_warmup_cubes,
        rfi_sumthr_enabled=not args.sumthr_disabled,
        static_sky_alpha=args.static_sky_alpha,
        static_sky_warmup_cubes=args.static_sky_warmup_cubes,
        static_sky_disabled=args.static_sky_disabled,
        n_fv_chunk=args.n_fv_chunk,
        chan_sum_factor=args.chan_sum_factor,
        sliding_window=args.sliding_window,
        cell_lambda_mode=args.cell_lambda_mode,
        dm_plan_path=args.dm_plan_path,
        inject_configs=inject_configs,
        inject_watch_enabled=bool(getattr(args, "inject_watch", False)),
    )

    dm_plan: DMPlan | None = None
    if args.n_coarse_dm > 0:
        from dsart.coarse_dm.synthetic_plan import build_synthetic_summed_plan
        dm_plan = build_synthetic_summed_plan(
            n_coarse=int(args.n_coarse_dm),
            dm_max=float(args.dm_max),
            chan_sum_factor=int(args.chan_sum_factor),
            t_int_fast_us=float(args.t_int_fast_native * NATIVE_SAMPLE_US),
        )
        LOG.info(
            "synthetic DMPlan: n_coarse=%d dm_max=%.1f chan_sum_factor=%d "
            "t_int_fast_us=%.3f (degenerate-but-shape-valid; F25 path active)",
            args.n_coarse_dm, args.dm_max, args.chan_sum_factor,
            args.t_int_fast_native * NATIVE_SAMPLE_US,
        )
    elif args.dm_plan_path is not None:
        LOG.info(
            "canonical DMPlan: loading %s (chan_sum_factor=%d, "
            "t_int_fast_us=%.3f)",
            args.dm_plan_path, args.chan_sum_factor,
            args.t_int_fast_native * NATIVE_SAMPLE_US,
        )

    # --max-blocks 0 sentinel = unlimited (production / soak runs).
    max_blocks: int | None = (
        None if args.max_blocks == 0 else int(args.max_blocks)
    )
    try:
        run(
            fada_int,
            args.output_dir,
            device,
            cfg,
            max_blocks=max_blocks,
            blocks_output_mode=args.blocks_output_mode,
            dm_plan=dm_plan,
            use_pipeliner_3s=args.pipeliner_3s,
            profile_stages_every=args.profile_stages_every,
            stage2_fifo_depth=args.stage2_fifo_depth,
            stage2_mode=args.stage2_mode,
            transport_tx_host=args.transport_tx_host,
            transport_tx_port=args.transport_tx_port,
            transport_tx_mode=args.transport_tx_mode,
            transport_tx_dtype=args.transport_tx_dtype,
            transport_tx_target_gbps_per_flow=(
                args.transport_tx_target_gbps_per_flow
            ),
            transport_tx_workers=args.transport_tx_workers,
            transport_tx_ring_slots=args.transport_tx_ring_slots,
            transport_tx_coarse_dm_mask=args.transport_tx_coarse_dm_mask,
            transport_tx_worker_hosts=args.transport_tx_worker_hosts,
            ready_sentinel_path=args.ready_sentinel_path,
            rfi_mon_cn_id=args.rfi_mon_cn_id,
            rfi_mon_window_size=args.rfi_mon_window_size,
            rfi_mon_freq_downsample=args.rfi_mon_freq_downsample,
            rfi_mon_shm_slots=args.rfi_mon_shm_slots,
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
