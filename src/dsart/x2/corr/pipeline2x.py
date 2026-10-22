"""2x corr_fast orchestrator (x2 WS-C deliverable 4), behind the op-point.

Selected only by ``corr_fast_integration --opoint 2x-...`` (default ``1x``:
today's loop, untouched). Scheduling is the 1x-compat restructure's
(:class:`~dsart.x2.corr.pipeline1x.PipelinedCorrFast`: double-buffered
async H2D, one-block output lag, deferred TX on a D2H stream); the block
body is the 2x chain:

    unpack (today's kernel)
      -> RFI flag (today's flagger, voltage domain, unchanged)
      -> 2x fast GEMM: core cross-baselines only, cal + cube mask folded
         into the per-(ch, pol, ant) gains and applied on operand load,
         array-burst time mask on load, cfp16 time-minor ring
         (+ fixed-order class reduce for per-baseline formats)
      -> per tier: CSR stage 1 (all coarse DMs, x2 / x4 decimation),
         batched static sky, row-rule packer
      -> TX hand-off (WS-T's sender; :class:`Tx2xStage`)

**Stage-2 FIFO is off in 2x** -- the search applies the single stage-2
rounding (SPEC §1/§2); nothing here touches ``ctx.stage2_fifo``.

Injection blocks (an injector footprint overlaps the block, rare) take an
explicit path that reproduces 1x's order exactly: zero-fill both masks,
apply cal to the voltages, add the injection, then the GEMM with unit
gains -- so an injected burst is the same signal it is today.

Consumers of the fast visibilities (inventory, see
``docs/x2/corr_fastvis_consumers.md``): the corner-turn streams (this
module), the static-sky snapshot exporter (:meth:`Corr2xPipeline.
sky_cells` + the tier-A :class:`BatchedStaticSky` duck-type the 1x
objects; owner cells from the value map, so per-baseline values grid at
the owner cell exactly as cells do), and nothing else: the RFI monitor
and array-burst detector read the voltages (unchanged), corr_slow /
voltage_retention read fada themselves. Dropping non-core baselines from
the fast GEMM therefore removes no consumer.

The CPU path (tests, I1) runs the same chain with the kernels' twins: a
complex64 GEMM, the numpy stage-1 twin over the ring rows the CSR needs,
the torch static sky and packer. ``value_subset`` / ``dm_subset`` shrink
the work for CPU tests and CP1 benches; production uses neither.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from dsart.services import corr_fast_integration as cfi
from dsart.services.slow_corr_kernel import NPACKETS_PER_BLOCK, NTIMES_PER_PACKET, apply_cal_split
from dsart.x2.corr.gemm2x import (
    build_gemm_tables,
    compact_cal,
    gemm2x_into_ring,
    gemm2x_reference,
    reduce_classes,
    ring_write_reference,
)
from dsart.x2.corr.packer2x import PackerRule, RowRulePacker, heavy_step_phase0
from dsart.x2.corr.pipeline1x import PipelinedCorrFast, _Done
from dsart.x2.corr.plan2x import CorrPlan2x
from dsart.x2.corr.stage1_cuda import Stage1Csr, geometry_for, stage1_csr, stage1_twin
from dsart.x2.corr.static_sky2x import BatchedStaticSky

LOG = logging.getLogger("dsart.x2.corr.pipeline2x")

__all__ = ["Corr2xPipeline", "NoOpTx2x", "RecordingTx2x", "Tx2xStage", "injection_active"]

BLOCK_NATIVE = NPACKETS_PER_BLOCK * NTIMES_PER_PACKET          # 4096


class Tx2xStage(Protocol):
    """TX hand-off of one tier of one block (WS-T's sender implements it)."""

    def transmit_tier(
        self, *, block_n: int, tier: str, dms: np.ndarray, t0_native: int, T_native: int,
        stream_ids: list[int], packets: list[dict], rfi_warming_up: bool,
    ) -> int:
        """``packets[s]``: ``payload`` uint8 ``(n_d, n_t, bytes_per_sample)``,
        ``step_main`` / ``step_heavy`` ``(n_d,)``. ``dms`` the global coarse
        indices of axis 0. Returns datagrams queued."""
        ...


class NoOpTx2x:
    def transmit_tier(self, **kw) -> int:
        return 0


@dataclass
class RecordingTx2x:
    """Test/I1 sink: keeps host copies of everything handed to TX."""

    log: list = field(default_factory=list)

    def transmit_tier(self, *, block_n, tier, dms, t0_native, T_native, stream_ids, packets,
                      rfi_warming_up) -> int:
        self.log.append(dict(
            block_n=int(block_n), tier=tier, dms=np.asarray(dms), t0_native=int(t0_native),
            T_native=int(T_native), stream_ids=list(stream_ids), rfi_warming_up=bool(rfi_warming_up),
            packets=[{k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                      for k, v in p.items()} for p in packets],
        ))
        return sum(int(p["payload"].shape[0] * p["payload"].shape[1]) for p in packets)


def default_rule_2x(vm, *, bits_main: int, hold_blocks: int) -> PackerRule:
    """The row rule the 2x corr runs: WS-T's ``quant.RowRule`` defaults when
    that module is present (the packer is bit-equal to it), else the Phase-0
    recommendation (tau 3 s, no |y| exclusion, skip blocks < 50% non-zero);
    the heavy step from Phase 0 §2 for this map; sigma held until the
    static-sky warm-up is over."""
    n_max = max(int(st.n_contrib.max()) for st in vm.streams)
    s_h = heavy_step_phase0(n_max, int(vm.n_image))
    try:
        from dsart.x2.quant import RowRule  # noqa: PLC0415
    except ImportError:
        return PackerRule(comp_bits=bits_main // 2, s_heavy=s_h, tau_s=3.0,
                          excl_sigma=float("inf"), min_frac=0.5, hold_blocks=hold_blocks)
    return PackerRule.from_quant(RowRule(comp_bits=bits_main // 2), s_heavy=s_h,
                                 hold_blocks=hold_blocks)


def injection_active(injector, block_n: int) -> bool:
    """True if any pending injection's footprint overlaps block ``block_n``
    (same window arithmetic as ``OnlineInjector.apply_block``)."""
    if injector is None or not getattr(injector, "pending", None):
        return False
    lo = int(block_n) * NPACKETS_PER_BLOCK * 2
    hi = lo + BLOCK_NATIVE
    for active in list(injector.pending.values()):
        if active.last_native_in_window >= lo and active.first_native_in_window < hi:
            return True
    return False


@dataclass
class _TierState:
    ti: int
    name: str
    dms: np.ndarray                 # global coarse indices processed
    shifts: np.ndarray              # (n_d, C) int32 q1 units
    geo: Any
    D: int
    T_native: int
    static_sky: BatchedStaticSky | None
    packer: RowRulePacker
    shifts_dev: torch.Tensor | None = None
    smin_dev: torch.Tensor | None = None


class Corr2xPipeline(PipelinedCorrFast):
    """2x block chain on the pipelined scheduler (see module doc)."""

    def __init__(
        self,
        ctx: "cfi.IntegrationContext",
        *,
        plan: CorrPlan2x,
        vm,
        is_core_baseline_mask: np.ndarray,
        bits_main: int = 8,
        rule: PackerRule | None = None,
        class_reduce: bool | None = None,
        tx2x: Tx2xStage | None = None,
        static_sky_window_blocks: int = 8,
        static_sky_warmup: int = 8,
        fv_chunk: int = 128,
        value_subset: int | None = None,
        dm_subset: list[int] | None = None,
        **kw,
    ) -> None:
        super().__init__(ctx, **kw)
        self.plan = plan
        self.vm = vm
        g = int(ctx.cfg.chgroup)
        self.chgroup = g
        nants = int(ctx.kernel.nants)
        n_pkt = self.expected_fada_bytes // (nants * int(ctx.kernel.nchan) * 4)
        self.n_q = n_pkt * NTIMES_PER_PACKET // 8                # q1 samples per block (t_int 8)
        self.T_ring = 2 * self.n_q
        self.fv_chunk = int(fv_chunk)
        if self.n_q % self.fv_chunk:
            raise ValueError(f"fv_chunk={fv_chunk} must divide {self.n_q}")
        self.tx2x = tx2x if tx2x is not None else NoOpTx2x()
        if class_reduce is None:
            class_reduce = vm.kind == "baselines"
        self.class_reduce = bool(class_reduce)
        self.tables = build_gemm_tables(
            is_core_baseline_mask, nants=nants,
            classes=vm.classes if self.class_reduce else None,
        )
        csr = stage1_csr(vm, g, ring_rows="classes" if self.class_reduce else "baselines")
        self.csr, self.voff, self.n_heavy = self._subset(csr, value_subset)
        self.stream_ids = [g * vm.n_sub + s for s in range(vm.n_sub)]
        # ring rows actually read by stage 1 (the CPU path only builds these)
        self.needed_rows = np.unique(self.csr.row)
        self.rule = rule if rule is not None else default_rule_2x(
            vm, bits_main=bits_main, hold_blocks=static_sky_warmup)
        dms_keep = None if dm_subset is None else set(int(d) for d in dm_subset)
        self.tiers: list[_TierState] = []
        for ti, t in enumerate(plan.tiers):
            local = [c for c in range(t.n_coarse) if dms_keep is None or t.d0 + c in dms_keep]
            if not local:
                continue
            shifts = np.ascontiguousarray(plan.stage1_bins(ti)[g][:, local].T.astype(np.int32))
            geo = geometry_for(shifts, D=t.decim)
            ss = BatchedStaticSky(n_d=len(local), n_values=self.csr.n_values,
                                  window_blocks=static_sky_window_blocks,
                                  warmup_cubes=static_sky_warmup, device=ctx.device)
            pk = RowRulePacker(n_d=len(local), value_offsets=self.voff,
                               n_contrib=self.csr.n_contrib, n_heavy=self.n_heavy,
                               rule=self.rule, device=ctx.device)
            ts = _TierState(ti=ti, name=t.name, dms=np.asarray([t.d0 + c for c in local]),
                            shifts=shifts, geo=geo, D=t.decim, T_native=t.T_native,
                            static_sky=ss, packer=pk)
            if self.cuda:
                ts.shifts_dev = torch.from_numpy(shifts).to(ctx.device)
                ts.smin_dev = torch.from_numpy(geo.smin).to(ctx.device)
            self.tiers.append(ts)
        self._ring: torch.Tensor | None = None
        self._scratch: torch.Tensor | None = None
        self._csr_dev = None
        self._tables_dev = None
        self._q_prev: int | None = None
        self.n_blocks_emitted = 0
        LOG.info(
            "Corr2xPipeline: chgroup=%d kind=%s n_sub=%d values=%d (streams %s) ring_rows=%d "
            "(%s) tiers=%s class_reduce=%s",
            g, vm.kind, vm.n_sub, self.csr.n_values, self.stream_ids, self.tables.n_ring_rows,
            "classes" if self.class_reduce else "core baselines",
            [(t.name, len(t.dms), t.D, t.geo.TT, t.geo.W, t.geo.chunk) for t in self.tiers],
            self.class_reduce,
        )

    # ------------------------------------------------------------------

    def _subset(self, csr: Stage1Csr, value_subset):
        vm = self.vm
        g = self.chgroup
        n_heavy = [int(vm.streams[g * vm.n_sub + s].n_heavy) for s in range(vm.n_sub)]
        self.value_sel = []
        if value_subset is None:
            for s in range(vm.n_sub):
                self.value_sel.append(np.arange(int(csr.value_offsets[s + 1] - csr.value_offsets[s])))
            return csr, [int(v) for v in csr.value_offsets], n_heavy
        keep, voff, nh = [], [0], []
        for s in range(vm.n_sub):
            lo, hi = int(csr.value_offsets[s]), int(csr.value_offsets[s + 1])
            nv = hi - lo
            m = min(int(value_subset), nv)
            # half from the front (heavy values), half from the back (main)
            sel = np.unique(np.concatenate([np.arange(m // 2), np.arange(nv - (m - m // 2), nv)]))
            self.value_sel.append(sel)
            keep.extend((lo + sel).tolist())
            voff.append(voff[-1] + sel.size)
            nh.append(int(np.count_nonzero(sel < n_heavy[s])))
        ptr, rows, ks, cj = [0], [], [], []
        for v in keep:
            a, b = int(csr.ptr[v]), int(csr.ptr[v + 1])
            rows.append(csr.row[a:b])
            ks.append(csr.k[a:b])
            cj.append(csr.conj[a:b])
            ptr.append(ptr[-1] + (b - a))
        sub = Stage1Csr(ptr=np.asarray(ptr, np.int32), row=np.concatenate(rows),
                        k=np.concatenate(ks), conj=np.concatenate(cj),
                        value_offsets=np.asarray(voff, np.int32),
                        n_contrib=csr.n_contrib[keep].astype(np.int32))
        return sub, voff, nh

    def sky_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Owner cells of this chgroup's values (wire order, streams
        concatenated) for :class:`SkySnapshotExporter` -- per-baseline values
        land on their class's owner cell; h23 ``np.add.at``s duplicates."""
        rows, cols = [], []
        for s in range(self.vm.n_sub):
            st = self.vm.streams[self.chgroup * self.vm.n_sub + s]
            rows.append(st.cell_row[self.value_sel[s]])
            cols.append(st.cell_col[self.value_sel[s]])
        return np.concatenate(rows).astype(np.uint16), np.concatenate(cols).astype(np.uint16)

    @property
    def sky_static(self) -> BatchedStaticSky | None:
        """Tier-A static sky (slot 0 = lowest coarse DM) for the exporter."""
        return self.tiers[0].static_sky if self.tiers else None

    def warmup(self, n_blocks: int) -> None:
        """Compile every kernel on dummy pages with the 2x TX detached."""
        saved, self.tx2x = self.tx2x, NoOpTx2x()
        try:
            super().warmup(n_blocks)
        finally:
            self.tx2x = saved

    def reset(self) -> None:
        super().reset()
        self._q_prev = None
        for t in self.tiers:
            if t.static_sky is not None:
                t.static_sky.reset()
            t.packer.reset()

    # ------------------------------------------------------------------
    # the 2x block body
    # ------------------------------------------------------------------

    def _ensure_buffers(self) -> None:
        if self._ring is not None:
            return
        dev = self.ctx.device
        nsum = self.vm.nchan_eff
        rows = self.tables.n_ring_rows if self.cuda else int(self.needed_rows.size)
        self._ring = torch.zeros((rows, nsum, self.T_ring, 2), dtype=torch.float16, device=dev)
        if self.cuda:
            if self.tables.n_scratch_rows:
                self._scratch = torch.empty((self.tables.n_scratch_rows, nsum, self.fv_chunk, 2),
                                            dtype=torch.float32, device=dev)
            self._tables_dev = self.tables.to(dev)
            self._csr_dev = {k: torch.from_numpy(np.ascontiguousarray(getattr(self.csr, k))).to(dev)
                             for k in ("ptr", "row", "k", "conj")}

    def _gains_and_voltages(self, real_v, imag_v, rfi_result, block_n):
        """Compact gains with the cube mask folded in + optional time keep;
        on injection blocks, today's explicit zero-fill -> cal -> inject."""
        ctx = self.ctx
        cal = ctx.cal
        nchan, _, npol, _, nants = real_v.shape
        ab = None
        mask = None
        if rfi_result is not None:
            ab = cfi._array_burst_time_mask(ctx, rfi_result)
            if ctx.cfg.rfi_mask_voltage_zero_fill:
                mask = rfi_result.mask
            else:
                ab = None
        if injection_active(ctx.injector, block_n):
            if mask is not None and (ab is not None or bool(mask.any())):
                real_v, imag_v = cfi.apply_rfi_mask_to_voltages(real_v, imag_v, mask,
                                                                 time_chan_mask=ab)
            if cal is not None:
                real_v, imag_v = apply_cal_split(real_v, imag_v, cal.cal_real, cal.cal_imag)
            cfi._apply_online_injection(real_v, imag_v, ctx=ctx, block_n=block_n)
            cr = torch.ones((nchan, npol, nants), dtype=torch.float32, device=real_v.device)
            ci = torch.zeros_like(cr)
            return real_v, imag_v, cr, ci, None
        if cal is not None:
            cr, ci = compact_cal(cal.cal_real, cal.cal_imag, mask)
        else:
            cr = torch.ones((nchan, npol, nants), dtype=torch.float32, device=real_v.device)
            ci = torch.zeros_like(cr)
            if mask is not None:
                keep = (~mask).permute(1, 2, 0).to(cr)
                cr = cr * keep
        tk = None if ab is None else (~ab).to(device=real_v.device, dtype=torch.float32)
        return real_v, imag_v, cr, ci, tk

    def _gemm(self, real_v, imag_v, cr, ci, tk, q_block: int) -> None:
        csf = self.vm.chan_sum_factor
        if self.cuda:
            for f0 in range(0, self.n_q, self.fv_chunk):
                gemm2x_into_ring(real_v, imag_v, cr, ci, self._tables_dev, self._ring,
                                 fv0=f0, n_fv=self.fv_chunk, t_ring0=q_block + f0, csf=csf,
                                 t_int=8, scratch=self._scratch, time_keep=tk)
                if self.tables.n_scratch_rows:
                    reduce_classes(self._scratch, self._ring, self._tables_dev,
                                   t_ring0=q_block + f0, n_fv=self.fv_chunk)
            return
        # CPU twin: complex64 GEMM of every core row (fv chunks bound the
        # (fv, k, A, A) intermediate), then only the rows stage 1 reads.
        need = torch.from_numpy(self.needed_rows)
        rv, iv = real_v.half(), imag_v.half()
        for f0 in range(0, self.n_q, 8):
            rows = gemm2x_reference(rv, iv, cr, ci, self.tables, fv0=f0, n_fv=8, csf=csf,
                                    t_int=8, time_keep=tk, accum=torch.float32)
            if self.class_reduce:
                c = self.vm.classes
                rows = ring_write_reference(rows, self.tables, c.bl_class, c.bl_conj)
            rows = rows[:, :, need]                                        # (8, C, Rn)
            pos = (q_block + f0 + torch.arange(8)) % self.T_ring
            self._ring[:, :, pos, 0] = rows.real.permute(2, 1, 0).half()
            self._ring[:, :, pos, 1] = rows.imag.permute(2, 1, 0).half()

    def _stage1(self, ts: _TierState, q_emit: int) -> torch.Tensor:
        n_out = self.n_q // ts.D
        V = self.csr.n_values
        if self.cuda:
            from dsart.x2.corr.stage1_cuda import stage1_tier_cuda  # noqa: PLC0415
            out = torch.empty((len(ts.dms), n_out, V, 2), dtype=torch.float32,
                              device=self.ctx.device)
            stage1_tier_cuda(self._ring, self._csr_dev, ts.shifts_dev, ts.smin_dev, out, ts.geo,
                             q_base=q_emit, n_out=n_out)
            return torch.view_as_complex(out)
        remap = np.searchsorted(self.needed_rows, self.csr.row).astype(np.int32)
        csr_local = Stage1Csr(ptr=self.csr.ptr, row=remap, k=self.csr.k, conj=self.csr.conj,
                              value_offsets=self.csr.value_offsets, n_contrib=self.csr.n_contrib)
        ring_np = self._ring.numpy()
        x = stage1_twin(ring_np, csr_local, ts.shifts, ts.geo, q_base=q_emit, n_out=n_out)
        return torch.from_numpy(x.astype(np.complex64))

    def _issue(self, st) -> None:
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
        real_v, imag_v, cr, ci, tk = self._gains_and_voltages(real_v, imag_v, rfi_result, st.block_n)
        cfi._tmark(ctx, "gains")
        self._ensure_buffers()
        q_block = int(st.block_n) * self.n_q
        self._gemm(real_v, imag_v, cr, ci, tk, q_block)
        del real_v, imag_v
        cfi._tmark(ctx, "gemm")
        packets = []
        q_emit = self._q_prev
        if q_emit is not None and q_emit + self.n_q == q_block:
            for ts in self.tiers:
                x = self._stage1(ts, q_emit)
                cfi._tmark(ctx, f"stage1_{ts.name}")
                if ts.static_sky is not None:
                    ts.static_sky.apply_(x)
                pk = ts.packer.pack(x)
                cfi._tmark(ctx, f"pack_{ts.name}")
                packets.append((ts, q_emit, pk, x))
        elif q_emit is not None:
            LOG.warning("block discontinuity (%s -> %s): stage-1 window restarts", q_emit, q_block)
        self._q_prev = q_block
        ready = None
        if self.cuda:
            ready = torch.cuda.Event()
            ready.record()
        out = cfi.IntegrationOutput(gridded_minus_sky=None, rfi=rfi_result, n_tx=0,
                                    block_n=int(st.block_n))
        out.x2 = packets                                     # type: ignore[attr-defined]
        rfi_warm = bool(rfi_result.warmup) if rfi_result is not None else False
        self._done.append(_Done(out=out, cubes=packets, rfi_warmup=rfi_warm, ready=ready))

    def _emit(self, *, keep: int):
        outs = []
        timer = self.ctx.stage_timer
        while len(self._done) > keep:
            d = self._done.popleft()
            n_tx = 0
            ctxm = torch.cuda.stream(self._d2h_stream) if self.cuda else _null()
            with ctxm:
                if self.cuda:
                    self._d2h_stream.wait_event(d.ready)
                if timer is not None:
                    timer.span_begin("tx", self._d2h_stream if self.cuda else None)
                for ts, q_emit, pk, _x in d.cubes:
                    n_tx += int(self.tx2x.transmit_tier(
                        block_n=d.out.block_n - 1, tier=ts.name, dms=ts.dms,
                        t0_native=q_emit * 8, T_native=ts.T_native, stream_ids=self.stream_ids,
                        packets=pk, rfi_warming_up=d.rfi_warmup,
                    ))
                if timer is not None:
                    timer.span_end("tx", self._d2h_stream if self.cuda else None)
                if self.cuda:
                    self._d2h_stream.synchronize()
            d.out.n_tx = n_tx
            d.cubes = []
            outs.append(d.out)
            self.n_blocks_out += 1
        return outs


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
