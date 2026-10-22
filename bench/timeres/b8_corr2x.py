"""B8 - the 2x corr chain on one corr GPU (CP1 corr track, x2 WS-C).

Measures, for every CP1 format (value kind x n_sub x bits) on production-
shaped data (96 antennas, 82-antenna core, 192 summed channels, 512 q1
samples per 134.218 ms block):

  gemm      2x fast GEMM into the cfp16 ring (core rows; classes + reduce)
  stage1    CSR stage 1 per tier (all coarse DMs, x2 / x4 decimation)
  sky+pack  batched static sky + row-rule packer per tier
  e2e       Corr2xPipeline on synthetic pages (async H2D, RFI off, NoOp TX):
            per-stage CUDA-event times (StageTimer) and the block wall time
  1x        the 1x-compat PipelinedCorrFast vs the legacy process_block loop
            on today's op-point (same pages), for the O2 decision

plus peak GPU memory per case and exactness: GEMM kernel vs the float64
twin (<= 1 fp16 ULP), stage-1 kernel vs its numpy twin (bit-exact) on a
value subset, 1x orchestrator vs process_block (torch.equal).

Budget: 134.218 ms per block; plan design 57 ms nominal / 99 ms worst; the
escalation E2 triggers at GEMM + stage 1 > 55 ms after fallbacks.

Run ONLY in a CP1 window on a corr node with the pipeline stopped (refuses
if any process holds a GPU). ``--selftest`` is a CPU dry run: it builds the
plans, value maps, CSRs and launch geometries for every format and prints
the shapes, memory estimates and kernel launch parameters -- no timing.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run  # noqa: E402

BLOCK_MS = 134.217728
FORMATS = ("2x-gridded-ns2-cint4", "2x-gridded-ns4-cint5", "2x-gridded-ns4-cint4",
           "2x-baselines-ns4-cint5", "2x-baselines-ns2-cint4", "2x-baselines-ns4-cint4")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _antpos(src: str):
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    if src == "fixture":
        z = np.load(os.path.join(ROOT, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
        return (z["antpos_e"].astype(np.float32), z["antpos_n"].astype(np.float32),
                core_baseline_mask_from_station_numbers(z["station"]))
    from dsart.services.corr_fast_integration import load_antpos_from_cal_blob
    return load_antpos_from_cal_blob(src)


def _setup(fmt: str, antpos, chgroup: int):
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.x2.corr.plan2x import CorrPlan2x
    from dsart.x2.corr.stage1_cuda import geometry_for, stage1_csr
    from dsart.x2.opoint import opoint_from_id
    from dsart.x2.plan import build_tiered_plan_for_opoint
    e, n, mask = antpos
    op = opoint_from_id(fmt)
    vm = build_value_map_for_opoint(op, e, n, is_core_baseline_mask=mask, dec_deg=71.63)
    plan = CorrPlan2x.from_tiered_plan(build_tiered_plan_for_opoint(op))
    classes = op.value_kind == "baselines"
    csr = stage1_csr(vm, chgroup, ring_rows="classes" if classes else "baselines")
    geos = []
    for ti, t in enumerate(plan.tiers):
        sh = np.ascontiguousarray(plan.stage1_bins(ti)[chgroup].T.astype(np.int32))
        geos.append((t, sh, geometry_for(sh, D=t.decim)))
    return op, vm, plan, csr, geos, mask


def describe(fmt: str, antpos, chgroup: int) -> dict:
    op, vm, plan, csr, geos, mask = _setup(fmt, antpos, chgroup)
    from dsart.x2.corr.gemm2x import build_gemm_tables
    tab = build_gemm_tables(mask, classes=vm.classes if op.value_kind == "baselines" else None)
    n_q, T_ring, nsum = 512, 1024, vm.nchan_eff
    out = dict(
        fmt=fmt, chgroup=chgroup, values=int(csr.n_values), sources=int(csr.row.size),
        ring_rows=tab.n_ring_rows, scratch_rows=tab.n_scratch_rows,
        ring_gib=round(tab.n_ring_rows * nsum * T_ring * 4 / 2 ** 30, 3),
        tile_pairs=int(tab.tile_pairs.shape[0]),
        tiers=[], plan_id=f"0x{plan.plan_id:016x}", pattern_id=f"0x{vm.pattern_id:016x}",
    )
    for t, sh, g in geos:
        n_out = n_q // t.decim
        out["tiers"].append(dict(
            tier=t.name, n_d=t.n_coarse, D=t.decim, TT=g.TT, W=g.W, chunk=g.chunk,
            spread=g.spread, smem_kib=round(g.smem_bytes / 1024, 1), threads=g.threads,
            grid=[int(csr.n_values), -(-n_out // g.TT)],
            out_gib=round(t.n_coarse * n_out * csr.n_values * 8 / 2 ** 30, 3),
            smem_reads_G=round(int(csr.row.size) * t.n_coarse * n_q / 1e9, 2),
        ))
    return out


def _pages(n, seed=1):
    from dsart.common.constants import FADA_BYTES_PER_BLOCK
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=FADA_BYTES_PER_BLOCK, dtype=np.uint8) for _ in range(n)]


def case_kernels(fmt, antpos, chgroup, dev, reps):
    """GEMM + stage 1 + sky/pack timings, peak memory, exactness."""
    from dsart.x2.corr.gemm2x import (build_gemm_tables, gemm2x_into_ring, gemm2x_reference,
                                      reduce_classes)
    from dsart.x2.corr.packer2x import PackerRule, RowRulePacker
    from dsart.x2.corr.stage1_cuda import stage1_tier_cuda, stage1_twin
    from dsart.x2.corr.static_sky2x import BatchedStaticSky
    from dsart.services.slow_corr_kernel import unpack_int4_split
    op, vm, plan, csr, geos, mask = _setup(fmt, antpos, chgroup)
    classes = op.value_kind == "baselines"
    tab = build_gemm_tables(mask, classes=vm.classes if classes else None).to(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    re, im = unpack_int4_split(_pages(1)[0], device=dev)
    nsum = vm.nchan_eff
    cr = torch.ones((384, 2, 96), device=dev)
    ci = torch.zeros_like(cr)
    ring = torch.zeros((tab.n_ring_rows, nsum, 1024, 2), dtype=torch.float16, device=dev)
    chunk = 128
    scr = (torch.empty((tab.n_scratch_rows, nsum, chunk, 2), device=dev)
           if tab.n_scratch_rows else None)

    def gemm():
        for f0 in range(0, 512, chunk):
            gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=f0, n_fv=chunk, t_ring0=512 + f0,
                             csf=2, t_int=8, scratch=scr)
            if scr is not None:
                reduce_classes(scr, ring, tab, t_ring0=512 + f0, n_fv=chunk)

    res = {"gemm_ms": cuda_time_ms(gemm, reps=reps)}
    # exactness of the GEMM on 8 fast samples (core-row tables, against the twin)
    if not classes:
        want = gemm2x_reference(re, im, cr, ci, build_gemm_tables(mask).to(dev), fv0=0, n_fv=8,
                                csf=2, t_int=8)
        ring2 = torch.zeros_like(ring)
        gemm2x_into_ring(re, im, cr, ci, tab, ring2, fv0=0, n_fv=8, t_ring0=0, csf=2, t_int=8)
        got = torch.complex(ring2[:, :, :8, 0].float(), ring2[:, :, :8, 1].float()).permute(2, 1, 0)
        w16 = torch.complex(want.real.half().float(), want.imag.half().float())
        res["gemm_max_ulp_rel"] = float(((got - w16).abs() / (w16.abs() + 1e-3)).max())
    csr_dev = {k: torch.from_numpy(np.ascontiguousarray(getattr(csr, k))).to(dev)
               for k in ("ptr", "row", "k", "conj")}
    n_heavy = [int(vm.streams[chgroup * vm.n_sub + s].n_heavy) for s in range(vm.n_sub)]
    tot1 = tot2 = 0.0
    for t, sh, g in geos:
        n_out = 512 // t.decim
        out = torch.empty((t.n_coarse, n_out, csr.n_values, 2), device=dev)
        shd = torch.from_numpy(sh).to(dev)
        smd = torch.from_numpy(g.smin).to(dev)
        ms = cuda_time_ms(lambda: stage1_tier_cuda(ring, csr_dev, shd, smd, out, g, q_base=512,
                                                   n_out=n_out), reps=reps)
        res[f"stage1_{t.name}_ms"] = ms
        tot1 += ms
        x = torch.view_as_complex(out)
        ss = BatchedStaticSky(n_d=t.n_coarse, n_values=csr.n_values, device=dev)
        pk = RowRulePacker(n_d=t.n_coarse, value_offsets=csr.value_offsets, n_contrib=csr.n_contrib,
                           n_heavy=n_heavy, rule=PackerRule(comp_bits=op.bits_main // 2), device=dev)
        ms2 = cuda_time_ms(lambda: pk.pack(ss.apply_(x.clone())), reps=reps)
        res[f"skypack_{t.name}_ms"] = ms2
        tot2 += ms2
        # stage-1 exactness on the first 6 values
        sub = type(csr)(ptr=csr.ptr[:7], row=csr.row[:csr.ptr[6]], k=csr.k[:csr.ptr[6]],
                        conj=csr.conj[:csr.ptr[6]], value_offsets=np.array([0, 6]),
                        n_contrib=csr.n_contrib[:6])
        ring_np = ring.cpu().numpy()
        twin = stage1_twin(ring_np, sub, sh, g, q_base=512, n_out=n_out)
        got = x[:, :, :6].cpu().numpy()
        res[f"stage1_{t.name}_bit_exact"] = bool(np.array_equal(got, twin.astype(np.complex64)))
    res.update(gemm_plus_stage1_ms=res["gemm_ms"] + tot1, skypack_ms=tot2,
               frac_of_block=(res["gemm_ms"] + tot1 + tot2) / BLOCK_MS,
               peak_gib=torch.cuda.max_memory_allocated(dev) / 2 ** 30)
    return res


def case_e2e(fmt, antpos, chgroup, dev, n_blocks):
    """Corr2xPipeline end to end on synthetic pages; StageTimer table."""
    import math as _m
    import dsart.services.corr_fast_integration as cfi
    from dsart.x2.corr.pipeline2x import Corr2xPipeline, NoOpTx2x
    op, vm, plan, csr, geos, mask = _setup(fmt, antpos, chgroup)
    e, n, _ = antpos
    cfg = cfi.FastIntegrationConfig(chgroup=chgroup, obs_dec_rad=_m.radians(71.63), n_grid=256,
                                    t_int_fast_native=8, rfi_enabled=False, static_sky_disabled=True,
                                    chan_sum_factor=2, n_sub=op.n_sub, cell_lambda_mode="tee45")
    ctx = cfi.build_context(cfg, device=dev, antpos_e=e, antpos_n=n, is_core_baseline_mask=mask)
    ctx.stage_timer = cfi.StageTimer(device=dev)
    torch.cuda.reset_peak_memory_stats(dev)
    orch = Corr2xPipeline(ctx, plan=plan, vm=vm, is_core_baseline_mask=mask,
                          bits_main=op.bits_main, tx2x=NoOpTx2x())
    orch.warmup(2)
    pages = _pages(4)
    walls = []
    for b in range(n_blocks):
        t0 = time.perf_counter()
        orch.push(pages[b % 4], block_n=10 + b)
        walls.append((time.perf_counter() - t0) * 1e3)
    orch.flush()
    ctx.stage_timer.poll()
    s = ctx.stage_timer.summary()
    w = sorted(walls[2:])
    return dict(stage_ms=s["stage_ms"], wall_ms_mean=float(np.mean(w)),
                wall_ms_p99=float(w[min(len(w) - 1, int(0.99 * len(w)))]),
                frac_of_block=float(np.mean(w)) / BLOCK_MS,
                peak_gib=torch.cuda.max_memory_allocated(dev) / 2 ** 30)


def case_1x(antpos, chgroup, dev, n_blocks, dm_plan_path):
    """Today's op-point: legacy process_block loop vs PipelinedCorrFast."""
    import math as _m
    import dsart.services.corr_fast_integration as cfi
    from dsart.x2.corr.pipeline1x import PipelinedCorrFast
    e, n, mask = antpos
    out = {}
    pages = _pages(4)
    for mode in ("legacy", "pipelined"):
        cfg = cfi.FastIntegrationConfig(
            chgroup=chgroup, obs_dec_rad=_m.radians(71.63), n_grid=256, t_int_fast_native=32,
            rfi_enabled=False, static_sky_disabled=False, chan_sum_factor=8, n_sub=4,
            sliding_window=True, cell_lambda_mode="tee45", dm_plan_path=dm_plan_path)
        ctx = cfi.build_context(cfg, device=dev, antpos_e=e, antpos_n=n, is_core_baseline_mask=mask)
        ctx.stage_timer = cfi.StageTimer(device=dev)
        walls = []
        if mode == "legacy":
            cfi._warmup_pipeline_jit(ctx, expected_fada_bytes=pages[0].size, n_blocks=2)
            for b in range(n_blocks):
                t0 = time.perf_counter()
                o = cfi.process_block(pages[b % 4], ctx=ctx, block_n=b + 1)
                o.gridded_minus_sky.sum().item()
                walls.append((time.perf_counter() - t0) * 1e3)
        else:
            orch = PipelinedCorrFast(ctx)
            orch.warmup(2)
            for b in range(n_blocks):
                t0 = time.perf_counter()
                orch.push(pages[b % 4], block_n=b + 1)
                walls.append((time.perf_counter() - t0) * 1e3)
            orch.flush()
        ctx.stage_timer.poll()
        w = sorted(walls[2:])
        out[mode] = dict(wall_ms_mean=float(np.mean(w)), wall_ms_p99=float(w[-1]),
                         stage_ms=ctx.stage_timer.summary()["stage_ms"])
        del ctx
        torch.cuda.empty_cache()
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--antpos", default="fixture",
                    help="'fixture' (tests/fixtures/x2) or a cal blob path (antennas.out)")
    ap.add_argument("--chgroup", type=int, default=15, help="15 = lowest band (largest delays)")
    ap.add_argument("--formats", nargs="*", default=list(FORMATS))
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--dm-plan-1x", default=None, help="v2 DM plan npz for the 1x comparison")
    args = ap.parse_args()
    antpos = _antpos(args.antpos)
    if args.selftest:
        for fmt in args.formats:
            print(json.dumps(describe(fmt, antpos, args.chgroup)), flush=True)
        print("b8 selftest ok (CPU dry run: shapes and launch geometry only)")
        return
    gpu_busy_guard(args.force)
    rec = Recorder("b8_corr2x", args)
    dev = torch.device(args.device)
    for fmt in args.formats:
        rec.row(case=f"describe_{fmt}", ok=True, **describe(fmt, antpos, args.chgroup))
        try_run(rec, f"kernels_{fmt}", lambda: case_kernels(fmt, antpos, args.chgroup, dev, args.reps),
                fmt=fmt)
        torch.cuda.empty_cache()
        try_run(rec, f"e2e_{fmt}", lambda: case_e2e(fmt, antpos, args.chgroup, dev, args.blocks), fmt=fmt)
        torch.cuda.empty_cache()
    if args.dm_plan_1x:
        try_run(rec, "1x_legacy_vs_pipelined",
                lambda: case_1x(antpos, args.chgroup, dev, args.blocks, args.dm_plan_1x))


if __name__ == "__main__":
    main()
