"""1x-compat restructure of the corr_fast block loop (x2 WS-C deliverable 1).

``corr_fast_integration.run`` today calls :func:`process_block` once per
fada page, serially. Measured on the live fleet (plan rev 4, "Corr"
table) that loop leaves the GPU idle or busy with avoidable copies for
roughly a third of the 134.218 ms block:

* the 288 MiB H2D of the page is synchronous (``torch.as_tensor`` on the
  page) -- 24.5 ms with nothing else queued;
* the F34 sliding window does ``torch.cat`` (prev, cur), a full
  ``(T,B,C)->(B,C,T)`` permute, a re/im split and two ``clone`` per
  block, and the stage-1 kernel computes ``t_dedisp_joined`` rows of
  which only the first ``n_fv`` are ever emitted;
* the stage-1 bin-shift table is re-uploaded from pageable memory every
  block (an implicit device synchronisation);
* the RFI cube mask is applied with a separate ``mul_`` pass over the
  ~1.2 GB of voltages (10.2 ms measured) behind a ``mask.any()`` sync;
* the TX hand-off D2H blocks the host at the tail of every block.

:class:`PipelinedCorrFast` keeps every arithmetic operation of
:func:`process_block` and changes only scheduling and buffering:

1. **Double-buffered async H2D.** Page ``k`` is copied on a dedicated copy
   stream into one of two preallocated device slots while block ``k-1``
   computes; the fada page is released (``markCleared``) as soon as its
   copy completes. One fada reader, as today.
2. **2-block ring.** The fused fast-GEMM kernel writes its Stokes-I output
   straight into the current half of a preallocated time-minor
   ``(NBASE, NCHAN_eff, 2*n_fv)`` fp32 ring (same kernel, same fp32
   values, only the store strides differ); stage 1 reads ``[prev | cur]``
   in place and computes only the ``n_fv`` emitted rows; one device copy
   slides ``cur -> prev``. No cat / permute / split / clone.
3. **Cube mask folded into the cal gains.** ``v * keep`` followed by the
   cal rotation equals the cal rotation with ``cal * keep``: identical
   values (only the sign of an exact zero can differ, which ``==`` and
   the cint8 codes do not see), without the ~10 ms voltage pass or the
   ``mask.any()`` sync. Blocks on which the array-burst time mask fires
   (~1%) take today's ``apply_rfi_mask_to_voltages`` path unchanged.
4. **Deferred host reads.** The TX hand-off of block ``k`` runs during
   block ``k+1`` on a D2H stream, overlapping the next block's GPU work;
   outputs are returned with that one-block lag. The bin-shift table
   lives on the device.
5. **Preallocation + ``gc.freeze``.** Device raw slots and the vis ring
   are allocated once; :func:`freeze_gc` moves start-up objects out of
   the collector's reach after warm-up.

Everything runs on the current (default) CUDA stream except the H2D copy
and the TX D2H, so every unchanged consumer -- static sky, the sky
exporter, the RFI monitor aggregator -- sees the same stream order as
today.

Bit-identity contract (plan §Correctness 1): for the same pages, the
sequence of ``IntegrationOutput.gridded_minus_sky`` tensors and of cubes
handed to ``transport_tx.transmit`` (with the same ``block_n`` and
``rfi_warming_up``) is ``torch.equal`` to a sequential loop over
:func:`process_block`. CPU: ``tests/x2/test_corr_pipeline1x.py``; GPU:
the ``x2_gpu`` tests in the same file plus the two kernel-level identities
(GEMM store strides, stage-1 row count) checked under ``TRITON_INTERPRET``
in ``tests/x2/test_corr_pipeline1x_triton.py``.

Selected with ``corr_fast_integration --orchestrator pipelined``; the
default stays ``legacy`` (today's loop).
"""
from __future__ import annotations

import gc
import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    LEGACY_FLUFF_SCALE,
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services import corr_fast_integration as cfi
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    apply_cal_split,
    unpack_int4_split,
)

LOG = logging.getLogger("dsart.x2.corr.pipeline1x")

__all__ = [
    "PipelinedCorrFast",
    "freeze_gc",
    "fused_corr_no_cat_into",
]


def freeze_gc() -> int:
    """Collect once, then move every live object to the permanent
    generation so the cyclic GC never re-walks the start-up heap (torch,
    triton, numpy module graphs) inside a block. Returns the number of
    objects frozen."""
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


def fused_corr_no_cat_into(
    R: torch.Tensor,
    I: torch.Tensor,
    out_re: torch.Tensor,
    out_im: torch.Tensor,
    *,
    n_fv: int,
    nchan: int,
    nchan_eff: int,
    nvp: int,
    csf: int,
    nbase: int = NBASE,
    BLOCK_M: int = 16,
    BLOCK_N: int = 16,
    num_warps: int = 4,
) -> None:
    """Launch today's RT-Phase-15 fused fast-GEMM kernel with caller-owned
    output views of logical shape ``(nbase, nchan_eff, n_fv)``.

    Identical to :func:`dsart.services.triton_corr_fused.
    fused_corr_post_no_cat_triton` (same kernel, same grid, same
    constexprs, same fp32 accumulation) except that the kernel stores
    through the strides of ``out_re`` / ``out_im`` -- here a slab of the
    time-minor stage-1 ring -- instead of a freshly allocated
    ``(n_fv, nbase, nchan_eff)`` tensor. The stored fp32 values are the
    same bits.
    """
    from dsart.services.triton_corr_fused import (  # noqa: PLC0415
        _fused_corr_post_no_cat_kernel,
    )

    assert R.is_contiguous() and I.is_contiguous()
    assert R.dtype == torch.float16 and I.dtype == torch.float16
    assert R.shape == I.shape
    assert tuple(out_re.shape) == (nbase, nchan_eff, n_fv), out_re.shape
    assert out_re.stride() == out_im.stride()
    nants = R.shape[2]
    K = R.shape[1]
    assert nants % BLOCK_M == 0 and nants % BLOCK_N == 0
    m_tiles = nants // BLOCK_M
    n_tiles = nants // BLOCK_N
    s_bls, s_c, s_fv = out_re.stride()
    grid = (n_fv, nchan_eff, m_tiles * n_tiles)
    _fused_corr_post_no_cat_kernel[grid](
        R, I,
        out_re, out_im,
        n_fv,
        nchan, nchan_eff, nants, nbase, nvp, csf, K,
        R.stride(0), R.stride(1), R.stride(2),
        s_fv, s_bls, s_c,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )


@dataclass
class _Staged:
    block_n: int
    slot: int
    h2d_done: Any  # torch.cuda.Event | None


@dataclass
class _Done:
    out: "cfi.IntegrationOutput"
    cubes: list
    rfi_warmup: bool
    ready: Any  # torch.cuda.Event | None


class _CaptureTx:
    """Stand-in TX used when a legacy consume phase must run: records the
    cubes so the real transmit can be deferred."""

    __slots__ = ("cubes",)

    def __init__(self) -> None:
        self.cubes: list = []

    def transmit(self, cubes_for_tx, *, block_n: int, rfi_warming_up: bool) -> int:
        self.cubes = list(cubes_for_tx)
        return 0


class PipelinedCorrFast:
    """Scheduling-only restructure of :func:`process_block` (see module doc).

    Usage (the fada loop)::

        orch = PipelinedCorrFast(ctx)
        for page in pages:
            for out in orch.push(page, block_n=n, release=reader.markCleared):
                consume(out)
        for out in orch.flush():
            consume(out)

    :meth:`push` stages page ``k``, issues block ``k-1`` and hands block
    ``k-2`` to the TX (``defer_tx=True``), so it returns at most one
    finished :class:`IntegrationOutput` -- block ``k-2`` -- whose
    ``n_tx`` is final. With ``defer_tx=False`` the TX runs right after the
    block is issued and the lag is one block.
    """

    N_SLOTS = 2

    def __init__(
        self,
        ctx: "cfi.IntegrationContext",
        *,
        expected_fada_bytes: int = FADA_BYTES_PER_BLOCK,
        defer_tx: bool = True,
        fold_mask_into_cal: bool = True,
        register_pages: bool = True,
    ) -> None:
        if ctx.profiler is not None:
            raise ValueError(
                "PipelinedCorrFast: the StageProfiler synchronises every "
                "block; use the always-on StageTimer instead"
            )
        self.ctx = ctx
        self.cuda = ctx.device.type == "cuda"
        self.expected_fada_bytes = int(expected_fada_bytes)
        self.defer_tx = bool(defer_tx)
        self.fold_mask_into_cal = bool(fold_mask_into_cal)
        self.register_pages = bool(register_pages) and self.cuda
        self._raw = [
            torch.empty(self.expected_fada_bytes, dtype=torch.uint8, device=ctx.device)
            for _ in range(self.N_SLOTS)
        ]
        if self.cuda:
            self._copy_stream = torch.cuda.Stream(device=ctx.device)
            self._d2h_stream = torch.cuda.Stream(device=ctx.device)
        else:
            self._copy_stream = self._d2h_stream = None
        self._slot_free: list[Any] = [None] * self.N_SLOTS
        #: (h2d_done, host page) of copies still in flight: a pinned page is read by the DMA
        #: after ``_stage`` returns, so it stays referenced until its copy has landed (a
        #: caller without a ``release`` callback may drop its own reference at once)
        self._inflight: deque = deque()
        self._staged: _Staged | None = None
        self._done: deque[_Done] = deque()
        self._n_staged = 0
        # stage-1 ring
        self._ring_re: torch.Tensor | None = None
        self._ring_im: torch.Tensor | None = None
        self._ring_n_fv = 0
        self._ring_primed = False
        self._bin_shifts_dev: torch.Tensor | None = None
        self.n_blocks_out = 0
        #: optional ``tap(name, block_n, re, im)`` -- CP1 compat taps see the
        #: fast visibilities in the ring (``vis``, (NBASE, NCHAN_eff, n_fv)).
        self.tap: Callable[..., Any] | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def push(
        self,
        page: np.ndarray,
        *,
        block_n: int,
        release: Callable[[], Any] | None = None,
    ) -> list["cfi.IntegrationOutput"]:
        """Stage ``page`` (block ``block_n``), issue the previously staged
        block, run any due TX hand-off, release ``page`` once its copy has
        landed, and return the finished outputs (oldest first)."""
        timer = self.ctx.stage_timer
        prev = self._staged
        t0 = time.perf_counter()
        if prev is not None and timer is not None:
            timer.begin_block(prev.block_n)
        try:
            staged = self._stage(page, block_n=block_n)
            self._staged = staged
            t1 = time.perf_counter()
            if prev is not None:
                self._issue(prev)
            t2 = time.perf_counter()
            ready = self._emit(keep=1 if self.defer_tx else 0)
            t3 = time.perf_counter()
        finally:
            if prev is not None and timer is not None:
                timer.end_block()
        if release is not None:
            if staged.h2d_done is not None:
                staged.h2d_done.synchronize()
            release()
        t4 = time.perf_counter()
        #: host-side phase times of the last push (ms): stage (H2D issue + page pin), issue
        #: (launching the block body), emit (wait for the previous block + TX), release
        self.last_timing = {"stage": (t1 - t0) * 1e3, "issue": (t2 - t1) * 1e3,
                            "emit": (t3 - t2) * 1e3, "release": (t4 - t3) * 1e3}
        return ready

    def flush(self) -> list["cfi.IntegrationOutput"]:
        """Issue the last staged block and complete every pending TX."""
        timer = self.ctx.stage_timer
        prev, self._staged = self._staged, None
        if prev is not None:
            if timer is not None:
                timer.begin_block(prev.block_n)
            try:
                self._issue(prev)
            finally:
                if timer is not None:
                    timer.end_block()
        outs = self._emit(keep=0)
        if self.cuda:
            torch.cuda.synchronize(self.ctx.device)
        return outs

    def reset(self) -> None:
        """Drop all in-flight state (staged page, pending TX, stage-1 ring).
        Used after warm-up so real block 1 starts cold, exactly like
        :func:`_warmup_pipeline_jit` resets the legacy loop."""
        if self.cuda:
            torch.cuda.synchronize(self.ctx.device)
        self._staged = None
        self._done.clear()
        self._inflight.clear()
        self._slot_free = [None] * self.N_SLOTS
        self._n_staged = 0
        self._ring_primed = False

    def warmup(self, n_blocks: int) -> None:
        """Pre-compile every kernel this orchestrator launches (the GEMM
        with ring strides and the ``n_fv``-row stage-1 are new Triton
        specialisations) on seeded dummy pages, with TX / FIFO / injector
        detached, then reset all per-cube state -- the pipelined twin of
        :func:`corr_fast_integration._warmup_pipeline_jit`."""
        if n_blocks <= 0:
            return
        ctx = self.ctx
        t0 = time.monotonic()
        dummy = np.random.default_rng(0xDADA).integers(
            0, 256, size=self.expected_fada_bytes, dtype=np.uint8,
        )
        saved = (ctx.transport_tx, ctx.stage2_fifo, ctx.injector)
        ctx.transport_tx = cfi.NoOpTransportTx()
        ctx.stage2_fifo = cfi.NoOpStage2Fifo()
        ctx.injector = None
        try:
            # n_blocks + 1 pushes so the ring path (needs a primed prev
            # half) runs at least n_blocks - 1 times.
            for i in range(int(n_blocks) + 1):
                self.push(dummy, block_n=i)
            self.flush()
        except Exception:
            LOG.exception("pipelined warmup raised; falling back to in-loop JIT")
        finally:
            ctx.transport_tx, ctx.stage2_fifo, ctx.injector = saved
            self.reset()
            if ctx.static_sky is not None:
                ctx.static_sky.reset()
            if ctx.rfi_flagger is not None:
                ctx.rfi_flagger.reset_warmup()
            if ctx.multi_dm_coarse_dm is not None:
                ctx.multi_dm_coarse_dm._prev_vis_stokes_i = None
                ctx.multi_dm_coarse_dm._prev_block_n = -1
        LOG.info("pipelined warmup: %d blocks in %.1fs", n_blocks, time.monotonic() - t0)

    # ------------------------------------------------------------------
    # staging (copy stream)
    # ------------------------------------------------------------------

    def _stage(self, page: np.ndarray, *, block_n: int) -> _Staged:
        arr = np.asarray(page)
        if arr.dtype != np.uint8:
            arr = arr.view(np.uint8)
        arr = arr.reshape(-1)
        if arr.size != self.expected_fada_bytes:
            raise ValueError(
                f"PipelinedCorrFast: page has {arr.size} B, expected "
                f"{self.expected_fada_bytes}"
            )
        slot = self._n_staged % self.N_SLOTS
        self._n_staged += 1
        if arr.flags["WRITEABLE"]:
            src = torch.as_tensor(arr)
        else:
            # PSRDADA reader pages (and replayed bytes) are read-only views; the tensor is only
            # ever the source of the H2D copy, so torch's "not writable" warning does not apply
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                src = torch.as_tensor(arr)
        dst = self._raw[slot]
        timer = self.ctx.stage_timer
        if not self.cuda:
            if timer is not None:
                timer.span_begin("h2d")
            dst.copy_(src)
            if timer is not None:
                timer.span_end("h2d")
            return _Staged(block_n=int(block_n), slot=slot, h2d_done=None)
        if self.register_pages:
            from dsart.services.host_pin import (  # noqa: PLC0415
                maybe_register_host_buffer,
            )
            maybe_register_host_buffer(arr)
        cs = self._copy_stream
        with torch.cuda.stream(cs):
            free = self._slot_free[slot]
            if free is not None:
                cs.wait_event(free)
            if timer is not None:
                timer.span_begin("h2d", cs)
            dst.copy_(src, non_blocking=True)
            if timer is not None:
                timer.span_end("h2d", cs)
            ev = torch.cuda.Event()
            ev.record(cs)
        while self._inflight and self._inflight[0][0].query():
            self._inflight.popleft()
        self._inflight.append((ev, arr))
        return _Staged(block_n=int(block_n), slot=slot, h2d_done=ev)

    # ------------------------------------------------------------------
    # one block (current stream)
    # ------------------------------------------------------------------

    def _unpack(self, raw_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.ctx
        if raw_t.is_cuda and ctx.voltage_dtype == torch.float16:
            # The exact kernel unpack_int4_split dispatches to after its H2D.
            from dsart.services.triton_unpack_onwire import (  # noqa: PLC0415
                fused_int4_unpack_onwire_triton,
            )
            return fused_int4_unpack_onwire_triton(
                raw_t,
                NPACKETS=NPACKETS_PER_BLOCK, NANTS=NANTS,
                NCHAN=NCHAN_PER_CHGROUP, NTIMES=NTIMES_PER_PACKET,
                NPOL=NPOL, scale=float(LEGACY_FLUFF_SCALE),
                out_dtype=torch.float16,
            )
        host = raw_t.cpu().numpy() if raw_t.is_cuda else raw_t.numpy()
        return unpack_int4_split(host, device=ctx.device, out_dtype=ctx.voltage_dtype)

    def _mask_and_cal(
        self,
        real_v: torch.Tensor,
        imag_v: torch.Tensor,
        rfi_result: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.ctx
        cal = ctx.cal
        cal_r = cal.cal_real if cal is not None else None
        cal_i = cal.cal_imag if cal is not None else None
        if rfi_result is not None:
            ab_time_mask = cfi._array_burst_time_mask(ctx, rfi_result)
            if ctx.cfg.rfi_mask_voltage_zero_fill:
                if ab_time_mask is not None:
                    # rare (~1% of blocks): today's single broadcast pass
                    real_v, imag_v = cfi.apply_rfi_mask_to_voltages(
                        real_v, imag_v, rfi_result.mask,
                        time_chan_mask=ab_time_mask,
                    )
                elif cal is not None and self.fold_mask_into_cal:
                    keep = (
                        (~rfi_result.mask).permute(1, 2, 0)        # (C, P, A)
                        .unsqueeze(1).unsqueeze(3)                # (C, 1, P, 1, A)
                        .to(device=cal_r.device, dtype=cal_r.dtype)
                    )
                    cal_r = cal_r * keep
                    cal_i = cal_i * keep
                elif bool(rfi_result.mask.any()):
                    real_v, imag_v = cfi.apply_rfi_mask_to_voltages(
                        real_v, imag_v, rfi_result.mask,
                    )
            cfi._tmark(ctx, "zero_fill")
        if cal is not None:
            real_v, imag_v = apply_cal_split(real_v, imag_v, cal_r, cal_i)
            cfi._tmark(ctx, "cal")
        return real_v, imag_v

    def _ring_mode(self) -> bool:
        ctx = self.ctx
        m = ctx.multi_dm_coarse_dm
        return (
            m is not None
            and bool(m.sliding_window)
            and int(ctx.cfg.chan_sum_factor) > 1
            and ctx.cfg.n_fv_chunk is None
        )

    def _issue(self, st: _Staged) -> None:
        ctx = self.ctx
        if self.cuda:
            torch.cuda.current_stream(ctx.device).wait_event(st.h2d_done)
        real_v, imag_v = self._unpack(self._raw[st.slot])
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record()
            self._slot_free[st.slot] = ev
        cfi._tmark(ctx, "unpack")

        rfi_result = None
        if ctx.rfi_flagger is not None and ctx.cfg.rfi_enabled:
            rfi_result = ctx.rfi_flagger.flag_block(real_v, imag_v)
            cfi._tmark(ctx, "rfi")
        real_v, imag_v = self._mask_and_cal(real_v, imag_v, rfi_result)
        cfi._apply_online_injection(real_v, imag_v, ctx=ctx, block_n=st.block_n)

        if self._ring_mode():
            out, cubes = self._gemm_stage1_consume(real_v, imag_v, rfi_result, st.block_n)
        else:
            out, cubes = self._legacy_tail(real_v, imag_v, rfi_result, st.block_n)
        ready = None
        if self.cuda:
            ready = torch.cuda.Event()
            ready.record()
        rfi_warm = bool(rfi_result.warmup) if rfi_result is not None else False
        self._done.append(_Done(out=out, cubes=cubes, rfi_warmup=rfi_warm, ready=ready))

    def _legacy_tail(self, real_v, imag_v, rfi_result, block_n):
        """GEMM + consume through today's functions (configs without the
        sliding-window fused path). TX is captured and deferred."""
        ctx = self.ctx
        vis = self._legacy_gemm(real_v, imag_v)
        cfi._tmark(ctx, "gemm")
        cap = _CaptureTx()
        saved = ctx.transport_tx
        ctx.transport_tx = cap
        try:
            out = cfi._process_block_consume_phase(vis, rfi_result, ctx=ctx, block_n=block_n)
        finally:
            ctx.transport_tx = saved
        return out, cap.cubes

    def _legacy_gemm(self, real_v, imag_v) -> torch.Tensor:
        """Steps 5+6 of ``_process_block_corr_phase`` verbatim."""
        ctx = self.ctx
        ppfv = ctx.cfg.t_int_fast_native // NTIMES_PER_PACKET
        n_packets_in = real_v.shape[3]
        if n_packets_in % ppfv != 0:
            raise ValueError("PipelinedCorrFast: packets not a multiple of ppfv")
        n_fv_total = n_packets_in // ppfv
        csf = int(ctx.cfg.chan_sum_factor)
        full_nchan = ctx.kernel.nchan
        nchan_eff = full_nchan // csf if csf > 1 else full_nchan
        if csf > 1 and ctx.cfg.n_fv_chunk is None:
            return ctx.kernel.compute_split(
                real_v, imag_v, fuse_stokes_i=True, chan_sum_factor=csf,
            )
        if ctx.cfg.n_fv_chunk is not None:
            n_fv_chunk = min(int(ctx.cfg.n_fv_chunk), n_fv_total)
        else:
            n_fv_chunk = cfi._auto_n_fv_chunk_for_streaming(
                n_fv_total, nchan=ctx.kernel.nchan, nbada_pol=ctx.kernel.nbada_pol,
            )
        vis = torch.empty((n_fv_total, NBASE, nchan_eff), dtype=torch.complex64,
                          device=ctx.device)
        for fv0 in range(0, n_fv_total, n_fv_chunk):
            fv1 = min(fv0 + n_fv_chunk, n_fv_total)
            r = real_v[:, :, :, fv0 * ppfv: fv1 * ppfv, :].contiguous()
            i = imag_v[:, :, :, fv0 * ppfv: fv1 * ppfv, :].contiguous()
            v2 = ctx.kernel.compute_split(r, i)
            si = cfi.stokes_i_pol_sum(v2)
            if csf > 1:
                si = si.reshape(fv1 - fv0, NBASE, nchan_eff, csf).sum(dim=-1)
            vis[fv0:fv1] = si
        return vis

    # -- the ring path ----------------------------------------------------

    def _ensure_ring(self, n_fv: int, nchan_eff: int) -> None:
        if self._ring_re is not None and self._ring_n_fv == n_fv:
            return
        shape = (NBASE, nchan_eff, 2 * n_fv)
        self._ring_re = torch.zeros(shape, dtype=torch.float32, device=self.ctx.device)
        self._ring_im = torch.zeros(shape, dtype=torch.float32, device=self.ctx.device)
        self._ring_n_fv = n_fv
        self._ring_primed = False

    def _gemm_into_ring(self, real_v: torch.Tensor, imag_v: torch.Tensor) -> int:
        """Fast GEMM + Stokes-I + chan-sum into the ring's current half.
        Returns ``n_fv``."""
        ctx = self.ctx
        kern = ctx.kernel
        ppfv = ctx.cfg.t_int_fast_native // NTIMES_PER_PACKET
        n_pk = int(real_v.shape[3])
        if n_pk % ppfv != 0:
            raise ValueError("PipelinedCorrFast: packets not a multiple of ppfv")
        n_fv = n_pk // ppfv
        csf = int(ctx.cfg.chan_sum_factor)
        if kern.nchan % csf != 0:
            raise ValueError(f"chan_sum_factor={csf} does not divide nchan={kern.nchan}")
        nchan_eff = kern.nchan // csf
        self._ensure_ring(n_fv, nchan_eff)
        cur_re = self._ring_re[:, :, n_fv:]
        cur_im = self._ring_im[:, :, n_fv:]
        K = NTIMES_PER_PACKET * ppfv
        triton_ok = (
            real_v.is_cuda
            and kern.nbada_pol == kern.nvolt_pol
            and real_v.dtype == torch.float16
            and K >= 16
        )
        if not triton_ok:
            vis = kern.compute_split(real_v, imag_v, fuse_stokes_i=True, chan_sum_factor=csf)
            cur_re.copy_(vis.real.permute(1, 2, 0))
            cur_im.copy_(vis.imag.permute(1, 2, 0))
            return n_fv
        # compute_split's validation + F31a slab loop, then the same
        # Stage 1-3 reshapes as FastCorrKernel._compute_one_slab.
        from dsart.services.corr_fast_kernel import validate_fast_voltage_shape  # noqa: PLC0415
        validate_fast_voltage_shape(
            real_v, imag_v, t_int_fast_native=kern.t_int_fast_native,
            nants=kern.nants, nchan=kern.nchan, nvolt_pol=kern.nvolt_pol,
        )
        n_fv_chunk = kern._auto_n_fv_chunk(n_fv)
        for fv0 in range(0, n_fv, n_fv_chunk):
            fv1 = min(fv0 + n_fv_chunk, n_fv)
            n_s = fv1 - fv0
            p0, p1 = fv0 * ppfv, fv1 * ppfv
            r_slab = real_v[:, :, :, p0:p1, :].contiguous()
            i_slab = imag_v[:, :, :, p0:p1, :].contiguous()
            R5 = r_slab.view(kern.nchan, NTIMES_PER_PACKET, kern.nvolt_pol, n_s, ppfv, kern.nants)
            I5 = i_slab.view(kern.nchan, NTIMES_PER_PACKET, kern.nvolt_pol, n_s, ppfv, kern.nants)
            R6 = R5.permute(3, 0, 2, 1, 4, 5).contiguous()
            I6 = I5.permute(3, 0, 2, 1, 4, 5).contiguous()
            del R5, I5, r_slab, i_slab
            nb = n_s * kern.nchan * kern.nvolt_pol
            R = R6.reshape(nb, K, kern.nants)
            I = I6.reshape(nb, K, kern.nants)
            fused_corr_no_cat_into(
                R, I, cur_re[:, :, fv0:fv1], cur_im[:, :, fv0:fv1],
                n_fv=n_s, nchan=kern.nchan, nchan_eff=nchan_eff,
                nvp=kern.nvolt_pol, csf=csf, nbase=NBASE,
            )
            del R6, I6, R, I
        return n_fv

    def _stage1_from_ring(self, n_fv: int) -> torch.Tensor:
        """F34 stage 1 on ``[prev | cur]``: the emitted ``n_fv`` rows only."""
        ctx = self.ctx
        m = ctx.multi_dm_coarse_dm
        n_filled = m.n_filled_out
        if not self._ring_primed:
            # cold start: exactly what dedisperse_from_vis returns first
            return torch.zeros((m.n_dm, n_fv, n_filled), dtype=torch.complex64,
                               device=ctx.device)
        joined = 2 * n_fv
        t_dedisp_joined = m.t_dedisp_for(joined)
        if t_dedisp_joined < n_fv:
            raise ValueError(
                f"sliding-window: max plan shift ({joined - t_dedisp_joined}) "
                f"exceeds prev-block n_fv={n_fv}; bump ring-buffer K above 2."
            )
        nb, nch = int(self._ring_re.shape[0]), int(self._ring_re.shape[1])
        if self._ring_re.is_cuda:
            from dsart.services.triton_dedisp import fused_dedisp_triton  # noqa: PLC0415
            if self._bin_shifts_dev is None:
                bs = m._stage1_bin_shifts()[:nch, m._dm_idx_iter]
                self._bin_shifts_dev = torch.as_tensor(
                    bs, dtype=torch.int32, device=ctx.device,
                ).contiguous()
            csr_offs, csr_b, csr_c = m._get_dedisp_csr(
                nb=nb, nch=nch, n_filled=n_filled, device=ctx.device,
            )
            out_re, out_im = fused_dedisp_triton(
                self._ring_re, self._ring_im,
                bin_shifts=self._bin_shifts_dev,
                csr_offs=csr_offs, csr_b=csr_b, csr_c=csr_c,
                n_filled=n_filled, t_dedisp=n_fv,
                BLOCK_T=128,
            )
            return torch.complex(out_re, out_im)
        # CPU: today's Phase-9 path on the same joined values, then the
        # emitted slice (bit-identical to dedisperse_from_vis on CPU).
        joined_vis = torch.complex(self._ring_re, self._ring_im).permute(2, 0, 1)
        return m._dedisperse_one_window(joined_vis)[:, :n_fv, :].clone()

    def _gemm_stage1_consume(self, real_v, imag_v, rfi_result, block_n):
        ctx = self.ctx
        n_fv = self._gemm_into_ring(real_v, imag_v)
        del real_v, imag_v
        cfi._tmark(ctx, "gemm")
        if self.tap is not None:
            self.tap("vis", block_n, self._ring_re[:, :, n_fv:], self._ring_im[:, :, n_fv:])
        dedispersed = self._stage1_from_ring(n_fv)
        # slide: cur -> prev (the cold-start block primes the ring)
        self._ring_re[:, :, :n_fv].copy_(self._ring_re[:, :, n_fv:])
        self._ring_im[:, :, :n_fv].copy_(self._ring_im[:, :, n_fv:])
        self._ring_primed = True
        cfi._tmark(ctx, "stage1")
        if ctx.static_sky is not None and not ctx.cfg.static_sky_disabled:
            for c in range(int(dedispersed.shape[0])):
                dedispersed[c] = ctx.static_sky.apply(dedispersed[c], dm_slot=c)
            cfi._tmark(ctx, "static_sky")
        # dedispersed is a fresh tensor every block: the per-coarse-DM
        # stage-2 FIFO keeps VIEWS of it across blocks.
        cubes = ctx.stage2_fifo.push(dedispersed, block_n=block_n)
        cfi._tmark(ctx, "stage2")
        out = cfi.IntegrationOutput(
            gridded_minus_sky=dedispersed, rfi=rfi_result, n_tx=0, block_n=int(block_n),
        )
        return out, list(cubes)

    # ------------------------------------------------------------------
    # deferred TX (D2H stream)
    # ------------------------------------------------------------------

    def _emit(self, *, keep: int) -> list["cfi.IntegrationOutput"]:
        ctx = self.ctx
        timer = ctx.stage_timer
        outs: list[cfi.IntegrationOutput] = []
        while len(self._done) > keep:
            d = self._done.popleft()
            if self.cuda:
                ds = self._d2h_stream
                with torch.cuda.stream(ds):
                    ds.wait_event(d.ready)
                    if timer is not None:
                        timer.span_begin("tx", ds)
                    n_tx = ctx.transport_tx.transmit(
                        d.cubes, block_n=d.out.block_n, rfi_warming_up=d.rfi_warmup,
                    )
                    if timer is not None:
                        timer.span_end("tx", ds)
                    # the TX hand-off copies synchronously; this event also
                    # orders any later reuse of the cubes' memory.
                    ds.synchronize()
            else:
                if timer is not None:
                    timer.span_begin("tx")
                n_tx = ctx.transport_tx.transmit(
                    d.cubes, block_n=d.out.block_n, rfi_warming_up=d.rfi_warmup,
                )
                if timer is not None:
                    timer.span_end("tx")
            d.out.n_tx = int(n_tx)
            d.cubes = []
            outs.append(d.out)
            self.n_blocks_out += 1
        return outs
