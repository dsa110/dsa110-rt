#!/usr/bin/env python3
"""bench/captured_burst_recovery.py — recover the M3 250924mptq burst
from the captured-NPZ fixtures (chunk-7 captured-mode end-to-end gate).

Pipeline:
  1. Load ``/home/ubuntu/data/m5_fixtures/<run_id>/`` via the chunk-7
     :func:`dsart.transport.captured_npz.load_captured_run` loader.
  2. Scatter sparse-COO → dense ``[N_chgroup=16, T_stream, N_grid, N_grid]
     complex64`` per chgroup (zero-filled missing slots).
  3. Quantise globally to ``[N_chgroup, T_stream, 2, N_grid, N_grid] int8``
     using a single global-max-abs scale (preserves cross-chgroup relative
     weighting, which is what the imager's coherent sum cares about).
  4. Build a fine-DM trial grid spanning the operator-pinned coarse cell;
     ``compute_time_shift_search`` produces ``[N_fdm, N_chgroup] int32``
     shifts at the fixture's fast-vis cadence.
  5. Run :class:`dsart.image.imager_gpu.GpuImager` to produce the
     ``[T_det, N_fdm, N_grid, N_grid] fp16`` cube.
  6. Find the top peak across (t, fdm, l, m). Recovery gate is the
     **per-fdm spatial-consistency** test: a real burst lights up the
     SAME (l_pix, m_pix) cell across every DM trial in the matched-
     filter window, modulated only by the DM-mismatch loss. The gate
     PASSes when the global peak has SNR ≥ ``--recovery-snr`` AND a
     fraction ≥ ``--recovery-consistency`` of fdm trials' top peaks
     lie within ``--recovery-pix-tol`` of the global peak's (l, m).
     The burst position is **not** assumed to be at boresight — real
     bursts almost never are; the array's (l_0, m_0) pointing only
     determines which sky cell lands at (N/2, N/2), and the burst
     can be anywhere in the FoV that the trigger pipeline accepts.
  7. Write ``recovery.json`` + a brief textual report.

D22 caveat: the M3 chunk-8 fixture is described as
"post-stage-2-dedispersed-to-coarse-cell" but empirical inspection of
the 250924mptq capture shows ~218 ms of dispersion remaining across
the band (chgroup 0 at 1.499 GHz peaks 104 samples earlier than chgroup
15 at 1.311 GHz), corresponding to an effective DM ≈ 385 pc/cc rather
than the labelled 404.688. This bench thus sweeps a wide fine-DM grid
([370, 420] pc/cc by default) so the recovery is robust to the
~20 pc/cc data offset; this lets us close the chunk-7 loop while M3
double-checks the dedispersion bookkeeping in their writer.

DM-discrimination caveat: with t_int_search = t_int_fast = 2097.152 µs
(the fast-vis cadence in the fixture, vs. the production search-side
cadence of 524.288 µs), per-trial differential dispersion shifts are
sub-sample for δDM ≲ 5 pc/cc. The DM-vs-amplitude curve is therefore
broad — the recovered peak DM has ~10 pc/cc uncertainty. A sharper
DM discriminator wants the production t_int_search and / or the
detector's K_time matched-filter integration along the time axis;
those are chunk-7 hardening items.

Run-recipe::

    python bench/captured_burst_recovery.py \\
        --captured-dir /home/ubuntu/data/m5_fixtures/250924mptq \\
        --t-det 384 --n-fdm 32 \\
        --dm-min 370 --dm-max 420 \\
        --out bench/reports/burst_recovery_250924mptq
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dsart.fine_dm.combiner import compute_time_shift_search  # noqa: E402
from dsart.image.imager_gpu import build_default_gpu_imager  # noqa: E402
from dsart.transport.captured_npz import (  # noqa: E402
    load_captured_run, stack_dense_streams,
)
from dsart.transport.quantize import (  # noqa: E402
    quantise_streams_global_cint8 as _quantise_streams_global_cint8,
)


_LOG = logging.getLogger("bench.captured_burst_recovery")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _build_fdm_grid(
    *,
    dm_min: float,
    dm_max: float,
    n_fdm: int,
    t_int_search_us: float,
    coarse_dm_pc_cm3: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, "TimeShiftSearchTable"]:
    """Build N_fdm fine-DM trials uniformly in [dm_min, dm_max] +
    the corresponding ``[N_fdm, N_chgroup] int32`` shift table.

    The M3 captured fixture for 250924mptq is **NOT pre-dedispersed**
    despite its label (the M3 chunk-8 writer's stage-2 step appears to
    be a no-op for this fixture; the burst still sweeps t=242 → t=346
    across the band, ≈ DM=385 pc/cc residual). For the chunk-7 recovery
    bench we therefore default ``coarse_dm = 0`` so the per-chgroup
    shifts compute the **full** dispersion displacement at each fine-DM
    trial, not just the residual from the manifest's coarse cell. When
    M3 fixes the fixture writer to actually pre-dedisperse, callers can
    pass ``coarse_dm_pc_cm3`` ≈ 405 to recover the small-residual path.
    """
    fine_dm = np.linspace(dm_min, dm_max, n_fdm, dtype=np.float64)
    coarse_dm = np.array([coarse_dm_pc_cm3], dtype=np.float64)
    if coarse_dm_pc_cm3 > dm_min:
        raise SystemExit(
            f"coarse_dm ({coarse_dm_pc_cm3}) > dm_min ({dm_min}); the "
            "§3.6.3 sign convention requires fine_dm ≥ coarse_dm so "
            "all shifts are non-negative."
        )
    fine_to_coarse = np.zeros(n_fdm, dtype=np.int64)
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        t_int_search_us=t_int_search_us,
    )
    return fine_dm, coarse_dm, table


def _topk_peaks_per_fdm(
    cube: torch.Tensor,
    *,
    k: int = 1,
) -> List[Dict[str, float]]:
    """Return per-fdm top-k peak records ``{fdm, t, l, m, value}``.

    cube shape: ``[T_det, N_fdm, N_grid, N_grid]``. Operates in fp32 on
    host to avoid fp16 argmax-on-tied-values issues.
    """
    cube_cpu = cube.detach().to(torch.float32).cpu().numpy()
    t_det, n_fdm, n_grid, _ = cube_cpu.shape
    out: List[Dict[str, float]] = []
    for f in range(n_fdm):
        plane = cube_cpu[:, f, :, :]   # [T_det, N_grid, N_grid]
        flat = plane.reshape(-1)
        if k >= flat.size:
            top_idx = np.argsort(flat)[::-1]
        else:
            # argpartition is O(N); sort the top-k afterwards.
            part = np.argpartition(flat, -k)[-k:]
            top_idx = part[np.argsort(flat[part])[::-1]]
        for ix in top_idx:
            t = int(ix // (n_grid * n_grid))
            rem = int(ix %  (n_grid * n_grid))
            l = int(rem // n_grid)
            m = int(rem %  n_grid)
            out.append({
                "fdm_idx": f,
                "t_in_cube": t,
                "l_pix": l,
                "m_pix": m,
                "value": float(plane[t, l, m]),
            })
    return out


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


def _bench_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_log = out_dir / "bench.log"
    handler = logging.FileHandler(bench_log, mode="w")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _LOG.setLevel(logging.INFO)
    _LOG.addHandler(handler)
    _LOG.addHandler(logging.StreamHandler(sys.stdout))

    bench_t0 = time.perf_counter_ns()

    # ---- 1. Load
    captured_dir = Path(args.captured_dir).resolve()
    _LOG.info("loading %s", captured_dir)
    chgroups, manifest = load_captured_run(captured_dir)
    streams_cf, valid_mask = stack_dense_streams(chgroups, fill_missing=True)
    n_chg, t_stream, n_grid, _ = streams_cf.shape
    n_present = int(sum(valid_mask))
    _LOG.info(
        "loaded run=%s src_kind=%s n_chgroups=%d/%d t_stream=%d N_grid=%d "
        "T2_DM=%s",
        manifest.run_id, manifest.src_kind, n_present, n_chg,
        t_stream, n_grid,
        f"{manifest.src_truth.dm_pc_cc:.3f}" if manifest.is_burst else "NaN",
    )

    # ---- 2. Find empirical burst time (sum |vis|² over uv-cells × chgroups).
    pwr_per_t = (np.abs(streams_cf) ** 2).sum(axis=(0, 2, 3))
    burst_peak_t_chg15 = int(
        np.argmax((np.abs(streams_cf[15]) ** 2).sum(axis=(1, 2)))
        if valid_mask[15] else 0
    )
    summed_top10_t = list(np.argsort(pwr_per_t)[-10:][::-1].astype(int))
    _LOG.info(
        "uv-power sweep: chgroup-15 peak at t=%d; summed-top-10 t=%s",
        burst_peak_t_chg15, summed_top10_t,
    )

    # ---- 3. Quantise (host-side; one-shot per fixture, off-hot-path).
    _LOG.info("quantising cf64 → cint8 (global scale)")
    qt0 = time.perf_counter_ns()
    streams_cint8, q_scale = _quantise_streams_global_cint8(
        streams_cf, target_max=args.target_max,
    )
    _LOG.info(
        "quant scale=%.3e, %d MiB cint8 stream stack, %.2f s",
        q_scale, streams_cint8.nbytes // (1024 * 1024),
        (time.perf_counter_ns() - qt0) / 1e9,
    )

    # ---- 4. Build fine-DM grid + shift table (at fixture's fast-vis cadence).
    t_int_us = float(manifest.t_int_fast_us)
    fine_dm, coarse_dm, table = _build_fdm_grid(
        dm_min=args.dm_min, dm_max=args.dm_max, n_fdm=args.n_fdm,
        t_int_search_us=t_int_us,
    )
    max_shift = int(table.shifts.max())
    _LOG.info(
        "fdm grid: N_fdm=%d, DM ∈ [%.3f, %.3f] pc/cc, t_int=%.3f µs, "
        "max shift=%d samples (%.1f ms)",
        args.n_fdm, args.dm_min, args.dm_max, t_int_us,
        max_shift, max_shift * t_int_us / 1000.0,
    )
    if max_shift + args.t_det > t_stream:
        _LOG.warning(
            "max_shift (%d) + T_det (%d) > T_stream (%d): some chgroups "
            "will be silently zero-padded by the fused kernel for the "
            "outermost DM trials. Reduce --t-det or narrow the DM range.",
            max_shift, args.t_det, t_stream,
        )

    # ---- 5. Build the imager + run the cube.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for GpuImager; run on h01 GPU 1")
    device = torch.device("cuda")
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=args.t_det, n_fdm=args.n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams_cint8_t = torch.from_numpy(streams_cint8).to(device)
    shifts_t = torch.from_numpy(table.shifts.astype(np.int32)).to(device)
    _LOG.info(
        "running GpuImager: T_det=%d N_fdm=%d N_grid=%d (cube ≈ %d MiB fp16)",
        args.t_det, args.n_fdm, n_grid,
        args.t_det * args.n_fdm * n_grid * n_grid * 2 // (1024 * 1024),
    )

    torch.cuda.synchronize()
    pt0 = time.perf_counter_ns()
    cube = imager.process_cube(
        streams_cint8=streams_cint8_t, time_shifts_gpu=shifts_t,
    )
    torch.cuda.synchronize()
    pt1 = time.perf_counter_ns()
    _LOG.info("imager done in %.1f ms", (pt1 - pt0) / 1e6)

    # ---- 6. Find top peaks across (t, fdm, l, m).
    cube_f32 = cube.detach().to(torch.float32).cpu().numpy()
    cube_max = float(cube_f32.max())
    cube_mean = float(cube_f32.mean())
    cube_std = float(cube_f32.std())
    _LOG.info(
        "cube stats: max=%.3e mean=%.3e std=%.3e (max/std=%.1f)",
        cube_max, cube_mean, cube_std,
        cube_max / cube_std if cube_std > 0 else 0.0,
    )

    # Global top peak.
    flat = cube_f32.reshape(-1)
    top_global = int(np.argmax(flat))
    g_t = top_global // (args.n_fdm * n_grid * n_grid)
    rem = top_global %  (args.n_fdm * n_grid * n_grid)
    g_f = rem // (n_grid * n_grid)
    rem = rem %  (n_grid * n_grid)
    g_l = rem // n_grid
    g_m = rem %  n_grid
    _LOG.info(
        "GLOBAL PEAK: t=%d fdm=%d (DM=%.3f pc/cc) l=%d m=%d value=%.3e "
        "(boresight = (%d, %d))",
        g_t, g_f, fine_dm[g_f], g_l, g_m, cube_f32[g_t, g_f, g_l, g_m],
        n_grid // 2, n_grid // 2,
    )

    # Per-fdm peaks (with their pixel locations).
    per_fdm = _topk_peaks_per_fdm(cube, k=1)
    for rec in per_fdm:
        rec["dm_pc_cc"] = float(fine_dm[rec["fdm_idx"]])

    # DM-vs-amplitude curve AT the global peak's (l, m), max over time.
    # This is the cleanest single-pixel matched-filter response across the
    # DM sweep (sharper than the per-fdm-peak curve which can roam in
    # (l, m) by ±2 cells from one DM trial to the next). Whichever DM
    # trial maximises this curve is the recovered DM at the burst location.
    dm_curve_at_peak_lm = []
    for f in range(args.n_fdm):
        time_col = cube_f32[:, f, g_l, g_m]
        peak_t = int(np.argmax(time_col))
        peak_v = float(time_col[peak_t])
        dm_curve_at_peak_lm.append({
            "fdm_idx": int(f),
            "dm_pc_cc": float(fine_dm[f]),
            "t_in_cube": peak_t,
            "value": peak_v,
            "snr_proxy": peak_v / cube_std if cube_std > 0 else 0.0,
        })

    # ---- 7. Per-fdm consistency check + recovery gate.
    # Real burst → same (l, m) cell lights up across every DM trial
    # in the matched-filter window, modulated only by DM-mismatch loss.
    # Boresight is NOT assumed: the array's (l_0, m_0) pointing only
    # decides which sky cell lands at (N/2, N/2); the burst can be
    # anywhere in the recovered FoV.
    consistent_count = 0
    for rec in per_fdm:
        dl_f = abs(int(rec["l_pix"]) - g_l)
        dm_f = abs(int(rec["m_pix"]) - g_m)
        if dl_f * dl_f + dm_f * dm_f <= args.recovery_pix_tol ** 2:
            consistent_count += 1
    consistency = consistent_count / max(1, args.n_fdm)

    # SNR proxy: cube max / std. NB: this includes the burst pixel in
    # the std denominator, so it's a slight UNDERESTIMATE of the true
    # SNR. Off-source noise std would be tighter but needs a source
    # mask; the simple ratio is sufficient for an 8 σ gate at the
    # recovery threshold.
    cube_snr = cube_max / cube_std if cube_std > 0 else 0.0
    truth_dm = (
        float(manifest.src_truth.dm_pc_cc) if manifest.is_burst else None
    )

    recovered = (
        manifest.is_burst
        and cube_snr >= args.recovery_snr
        and consistency >= args.recovery_consistency
    )
    gate_status = "PASS" if recovered else (
        "INSPECTION_ONLY" if not manifest.is_burst else "FAIL"
    )

    record = {
        "bench": "captured_burst_recovery",
        "milestone": "M5",
        "schema_version": 1,
        "config": {
            "captured_dir": str(captured_dir),
            "t_det": args.t_det,
            "n_fdm": args.n_fdm,
            "n_grid": n_grid,
            "dm_min": args.dm_min,
            "dm_max": args.dm_max,
            "target_max": args.target_max,
            "recovery_pix_tol": args.recovery_pix_tol,
            "recovery_snr": args.recovery_snr,
        },
        "manifest": {
            "run_id": manifest.run_id,
            "src_kind": manifest.src_kind,
            "src_name": manifest.src_name,
            "obs_dec_deg": manifest.obs_dec_deg,
            "t_int_fast_us": manifest.t_int_fast_us,
            "t_int_fast_native": manifest.t_int_fast_native,
            "n_chgroups_present": n_present,
            "n_chgroups_total": n_chg,
            "src_truth": {
                "src_name": manifest.src_truth.src_name,
                "ra_deg": manifest.src_truth.ra_deg,
                "dec_deg": manifest.src_truth.dec_deg,
                "mjd_trigger": manifest.src_truth.mjd_trigger,
                "dm_pc_cc": manifest.src_truth.dm_pc_cc,
                "t2_snr": manifest.src_truth.t2_snr,
                "is_burst": manifest.is_burst,
            },
        },
        "fdm_grid": {
            "fine_dm_pc_cm3": fine_dm.tolist(),
            "coarse_dm_pc_cm3": coarse_dm.tolist(),
            "max_shift_samples": max_shift,
            "max_shift_ms": max_shift * t_int_us / 1000.0,
        },
        "imager": {
            "process_ms": (pt1 - pt0) / 1e6,
            "quant_scale": q_scale,
        },
        "cube_stats": {
            "max": cube_max,
            "mean": cube_mean,
            "std": cube_std,
            "max_over_std": cube_snr,
        },
        "uv_power_sweep": {
            "chg15_peak_t": burst_peak_t_chg15,
            "summed_top10_t": summed_top10_t,
        },
        "global_peak": {
            "t_in_cube": int(g_t),
            "fdm_idx": int(g_f),
            "dm_pc_cc": float(fine_dm[g_f]),
            "l_pix": int(g_l),
            "m_pix": int(g_m),
            "value": float(cube_f32[g_t, g_f, g_l, g_m]),
            "delta_pix_from_boresight": {
                "dl": int(g_l - n_grid // 2),
                "dm": int(g_m - n_grid // 2),
                "radial": float(
                    np.sqrt((g_l - n_grid // 2) ** 2
                            + (g_m - n_grid // 2) ** 2)
                ),
            },
        },
        "per_fdm_top_peaks": per_fdm,
        "dm_curve_at_peak_lm": dm_curve_at_peak_lm,
        "consistency": {
            "n_fdm_within_tol_of_global_peak": int(consistent_count),
            "fraction": consistency,
            "tol_pix": args.recovery_pix_tol,
        },
        "truth_alignment": {
            "labelled_dm_pc_cc": truth_dm,
            "recovered_dm_pc_cc": float(fine_dm[g_f]),
            "dm_residual_pc_cc": (
                float(fine_dm[g_f]) - truth_dm if truth_dm is not None else None
            ),
            "boresight_l_pix": n_grid // 2,
            "boresight_m_pix": n_grid // 2,
        },
        "gate_status": gate_status,
        "wall_clock_s": (time.perf_counter_ns() - bench_t0) / 1e9,
    }

    out_path = out_dir / "recovery.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))
    _LOG.info("wrote %s", out_path)
    _LOG.info(
        "RESULT: gate=%s peak=(t=%d, fdm=%d, l=%d, m=%d, dm=%.2f, snr=%.2f) "
        "consistency=%d/%d (%.0f%% of fdms within %.0f-pix of peak) "
        "labelled_dm=%s",
        gate_status, g_t, g_f, g_l, g_m, fine_dm[g_f], cube_snr,
        consistent_count, args.n_fdm, 100 * consistency,
        args.recovery_pix_tol,
        f"{truth_dm:.3f}" if truth_dm is not None else "NaN",
    )

    return 0 if gate_status in ("PASS", "INSPECTION_ONLY") else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 captured-fixture burst-recovery bench"
    )
    parser.add_argument(
        "--captured-dir", required=True,
        help="dir containing manifest.json + chgroupNN.npz (M3-emitted)",
    )
    parser.add_argument("--t-det", type=int, default=384)
    parser.add_argument("--n-fdm", type=int, default=32)
    parser.add_argument("--dm-min", type=float, default=370.0)
    parser.add_argument("--dm-max", type=float, default=420.0)
    parser.add_argument(
        "--target-max", type=int, default=120,
        help="quantisation clip target (max abs cint8 value; ≤127)",
    )
    parser.add_argument(
        "--recovery-pix-tol", type=float, default=2.0,
        help="per-fdm peaks count as 'consistent' if within this many "
             "pixels of the global peak's (l, m)",
    )
    parser.add_argument(
        "--recovery-snr", type=float, default=8.0,
        help="recovery requires cube max/std ≥ this value",
    )
    parser.add_argument(
        "--recovery-consistency", type=float, default=0.5,
        help="recovery requires this fraction of fdm trials' top peaks "
             "to lie within --recovery-pix-tol of the global peak",
    )
    parser.add_argument(
        "--out", required=True,
        help="output dir for recovery.json + bench.log",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return _bench_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
