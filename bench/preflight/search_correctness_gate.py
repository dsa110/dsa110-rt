#!/usr/bin/env python3
"""bench/preflight/search_correctness_gate.py — operator-driven search-
side CORRECTNESS gate. Run this BEFORE any fleet restart that touches
detector / Layer-1 / cube-pipeline code, alongside the speed gate.

What it does
============

Drives the full search pipeline (CubePipeline + Detector + Layer-1)
against a deterministic synthetic input::

    chgroup-15 stream = N(0, 1) per (t, l, m) + a delta injection
                        at the cube cell (t_in_cube, l_pix, m_pix)
                        of amplitude `inject_amplitude`.

The injection is placed at the **reference** chgroup (g=15, zero
shift), so for every fine-DM trial the combiner reads the injection
through the same (l, m) cell at the same cube-time. After the imager
this lights up cell ``(t_in_cube, fine_dm_with_zero_shift_at_g15,
l_pix, m_pix)`` on the detector's cube. We then assert:

  1. The detector emits AT LEAST one Candidate near (l_pix, m_pix,
     t_in_cube), within the tolerance window (±2 grid cells, ±64
     spectra-samples).

  2. That Candidate's SNR is above the operator-configurable floor
     (default 30 σ; with `inject_amplitude=200` the analytic peak
     SNR is ~tens of σ after 16-chgroup sum + 256² FFT — well above
     the production threshold).

  3. The noise-only segments (cubes 0..warmup_cubes that don't
     contain the injection) DO NOT emit any Candidates above the
     production threshold (12 σ). This catches false-positive
     regressions like the M7.4.2 coverage-correction bias.

PASS = all three. FAIL = any one fails; the failing assertion is
printed with the recovered Candidate details so the operator can
attribute the regression.

Why this exists
===============

Same motivation as the speed gate, but for CORRECTNESS instead of
latency. The live-fleet "fire an injection and grep journalctl"
loop is slow and confounded by real RFI / calibration drift. This
gate gives the operator a single command that says "did the M7.7
(or any other) change break detection on a clean signal?".

The gate is independent of:
  * etcd / control plane
  * real SNAP packets / SHM transport
  * C1 / C2 / dump-uploader / clusterer

So a regression here = a search-side compute bug; conversely a PASS
here + a missed live injection = a transport / config / fleet bug
(NOT a search-compute bug).

Production op-point pinning
===========================

Mirrors ``configs/dsart_search_rt.yaml`` exactly: n_grid=256, n_fdm=
34, t_det=192, M7.7 symmetric-shift padding on, fp16/cuda/gpu-imager,
real v2 DM plan, all detector knobs. Bump :data:`PROD_*` constants
when the yaml flips.

CLI
===

::

  python -m bench.preflight.search_correctness_gate \\
      [--n-cubes 15]                                    \\
      [--inject-cube-idx 8]                             \\
      [--inject-t-in-cube 96]                           \\
      [--inject-l-pix 128] [--inject-m-pix 128]         \\
      [--inject-amplitude 200.0]                        \\
      [--snr-floor 30.0]                                \\
      [--false-positive-threshold-sigma 12.0]           \\
      [--owner-idx 0]                                   \\
      [--dm-plan-path ...]                              \\
      [--out /tmp/m77_correctness_gate]

Typical run on a search node:

::

  CUDA_VISIBLE_DEVICES=0 python -m bench.preflight.search_correctness_gate

Exit code conventions
=====================

* 0 — all three assertions PASS. Detector correctness is healthy on
      a clean signal at the production op-point. Safe to fleet-push.
* 1 — at least one assertion FAILED. DO NOT fleet-push; the printed
      failure indicates which sub-assertion failed (recovery /
      SNR / false-positive).
* 2 — bench setup itself failed (DM plan missing, GPU missing, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# DSART_TEST gates the host-pin helpers that the SyntheticRxRingSource
# uses with --prequantise; we explicitly drive the GPU path on a real
# node, so DON'T set DSART_TEST here.

import torch  # noqa: E402

from dsart.common.contracts import Candidate  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import (  # noqa: E402
    SyntheticInjection,
    SyntheticRxRingSource,
)
from dsart.services.search_compute import (  # noqa: E402
    _dm_grids_from_npz,
)


_LOG = logging.getLogger("bench.preflight.search_correctness_gate")


# Production op-point — keep in sync with bench/preflight/search_speed_gate.py
# AND configs/dsart_search_rt.yaml.
PROD_N_GRID = 256
PROD_N_FDM = 34
PROD_T_DET = 192
PROD_T_INT_SEARCH_US = 1048.576
PROD_THRESHOLD_SIGMA = 12.0  # c1.snr_min, the false-positive floor
PROD_BANK_MASK_IMG = ("unit",)
PROD_BANK_MASK_DM = ("d1",)
PROD_BANK_MASK_TIME = ("b1", "b2", "b4", "b8", "b16", "b32", "b64")
PROD_DM_PLAN_PATH = "/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz"
PROD_LAYER1_MAX_SAMPLES = 10000
PROD_LAYER2_MAX_SAMPLES = 100000
PROD_DET_TILE_SIZE = 256
PROD_DET_N_TOP = 24

# Tolerance windows for the recovered candidate. The injection is a
# single delta at chgroup 15; the imager smears it over a 5-cell edge
# mask + the K_image kernel support, so the recovered (l, m) drifts
# by ≤ 2 cells. Δt depends on the box-car width of the matched kernel
# (max K_time = 64, half-window = 32). dm tolerance is wider because
# the burst is a chgroup-15-only delta which lights up many fine-DMs.
PROD_LM_TOL_CELLS = 2
PROD_T_TOL_SAMPLES = 32
PROD_DM_TOL_FINE = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dm_grids(
    dm_plan_path: Path, owner_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the production v2 plan + slice to one coarse-DM owner.
    Mirrors the bench/search_node_throughput.py --dm-plan-path +
    --coarse-dm-owner-idx logic exactly.
    """
    coarse, fine, f2c = _dm_grids_from_npz(dm_plan_path, n_coarse=8)
    mask = f2c == owner_idx
    if not mask.any():
        raise SystemExit(
            f"--owner-idx {owner_idx} not in fine_to_coarse range "
            f"{int(f2c.min())}..{int(f2c.max())}"
        )
    fine = fine[mask]
    coarse = coarse[owner_idx : owner_idx + 1]
    f2c_local = np.zeros(len(fine), dtype=np.int32)
    return coarse, fine, f2c_local


def _build_pipeline(
    *,
    device: torch.device,
    n_fdm: int,
    cube_dtype: torch.dtype = torch.float16,
    threshold_sigma: float = PROD_THRESHOLD_SIGMA,
) -> Tuple[CubePipeline, DeterministicDetector]:
    """Build a production-pinned CubePipeline + Detector tuple. M7.7 is
    auto-detected on the first slot whose stream_origin_offset_samples
    > 0, so just constructing this object is sufficient — the M7.7 path
    flips on at cube 0 in the run loop.
    """
    bank = build_kernel_bank(
        image_tokens=list(PROD_BANK_MASK_IMG),
        dm_tokens=list(PROD_BANK_MASK_DM),
        time_tokens=list(PROD_BANK_MASK_TIME),
        dtype=cube_dtype,
    )
    detector = DeterministicDetector(
        kernel_bank=bank,
        threshold_sigma=float(threshold_sigma),
        detector_version="v1.preflight",
        search_node_id=1,
        gpu_half=0,
        dtype=cube_dtype,
        device=device,
        streaming=True,
        streaming_tile_size=PROD_DET_TILE_SIZE,
        streaming_decoder_n_top=PROD_DET_N_TOP,
        boxcar_accum_dtype=torch.float16,
        layer2_sigma_max_samples=PROD_LAYER2_MAX_SAMPLES,
    )
    cfg = CubePipelineConfig(
        n_grid=PROD_N_GRID,
        edge_mask_kernel_support=5,
        cube_dtype=cube_dtype,
        device=str(device),
        image_backend="gpu",
        gpu_t_det=PROD_T_DET,
        gpu_n_fdm=n_fdm,
        gpu_complex_dtype=torch.complex32,
    )
    pipeline = CubePipeline(
        config=cfg,
        detector=detector,
        layer1_state=Layer1State(
            n_fdm=n_fdm, n_burnin_cubes=5,
            max_samples=PROD_LAYER1_MAX_SAMPLES,
        ),
    )
    return pipeline, detector


def _expected_fine_dm_idx(time_shift_table) -> int:
    """The fine-DM whose chgroup-15 shift is zero is the "best" trial
    for a chgroup-15-only delta injection (no time-smear across fdms).
    With Option A, shifts[:, 15] spans -pad_right..+pad_left and the
    fine-DM with zero g15 shift is the closest-to-coarse-DM one.
    """
    shifts_g15 = time_shift_table.shifts[:, 15]
    return int(np.argmin(np.abs(shifts_g15)))


def _classify_candidate_recovery(
    cands: Sequence[Candidate],
    *,
    inject_cube_idx: int,
    inject_t_in_cube: int,
    inject_l_pix: int,
    inject_m_pix: int,
    expected_fine_dm_idx: int,
    cube_specnum_start: int,
    lm_tol: int = PROD_LM_TOL_CELLS,
    t_tol: int = PROD_T_TOL_SAMPLES,
    dm_tol: int = PROD_DM_TOL_FINE,
) -> Optional[Candidate]:
    """Return the closest-to-truth in-window candidate, or None.

    A candidate "matches" the injection when its (l, m, dm_idx,
    event_specnum) is within the tolerance window around the injection
    truth. ``event_specnum`` is the absolute spectra-sample number;
    the injection lives at ``cube_specnum_start + t_in_cube``.
    """
    if not cands:
        return None
    truth_specnum = int(cube_specnum_start) + int(inject_t_in_cube)
    in_window: List[Candidate] = []
    for c in cands:
        if abs(int(c.l) - int(inject_l_pix)) > lm_tol:
            continue
        if abs(int(c.m) - int(inject_m_pix)) > lm_tol:
            continue
        if abs(int(c.dm_idx) - int(expected_fine_dm_idx)) > dm_tol:
            continue
        if abs(int(c.event_specnum) - int(truth_specnum)) > t_tol:
            continue
        in_window.append(c)
    if not in_window:
        return None
    return max(in_window, key=lambda c: float(c.snr))


def _candidates_above_threshold_noise_only(
    all_cands_per_cube: Sequence[Sequence[Candidate]],
    *,
    inject_cube_idx: int,
    threshold_sigma: float,
) -> List[Candidate]:
    """Return any Candidate from NON-injection cubes whose SNR is above
    the false-positive threshold. Used to catch detector regressions
    that turn pure noise into spurious bursts.
    """
    out: List[Candidate] = []
    for cube_idx, cands in enumerate(all_cands_per_cube):
        if int(cube_idx) == int(inject_cube_idx):
            continue
        for c in cands:
            if float(c.snr) >= float(threshold_sigma):
                out.append(c)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run_gate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.dm_plan_path).exists():
        _LOG.error(
            "--dm-plan-path %s not found; production search nodes "
            "ship the v2 plan at %s", args.dm_plan_path, PROD_DM_PLAN_PATH,
        )
        return 2

    coarse, fine, f2c = _resolve_dm_grids(
        Path(args.dm_plan_path), int(args.owner_idx)
    )
    n_fdm = int(fine.shape[0])
    _LOG.info(
        "DM plan: %s | owner_idx=%d -> n_fdm=%d coarse_dm=%.3f",
        args.dm_plan_path, int(args.owner_idx), n_fdm, float(coarse[0]),
    )

    device_str = str(args.device)
    if device_str == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    _LOG.info("device=%s", device)
    if device.type != "cuda":
        _LOG.error(
            "This gate requires CUDA (production op-point pins the "
            "GPU imager). Run on a search node with --device cuda:0."
        )
        return 2

    inject_cube_idx = int(args.inject_cube_idx)
    if inject_cube_idx <= 0 or inject_cube_idx >= int(args.n_cubes):
        _LOG.error(
            "--inject-cube-idx=%d must be > 0 (Layer-1 burn-in) AND "
            "< --n-cubes=%d (so it gets emitted)",
            inject_cube_idx, int(args.n_cubes),
        )
        return 2

    injection = SyntheticInjection(
        cube_idx=inject_cube_idx,
        t_in_cube=int(args.inject_t_in_cube),
        l_pix=int(args.inject_l_pix),
        m_pix=int(args.inject_m_pix),
        amplitude=float(args.inject_amplitude),
    )
    src = SyntheticRxRingSource(
        n_cubes=int(args.n_cubes),
        t_det=PROD_T_DET,
        n_fdm=n_fdm,
        n_grid=PROD_N_GRID,
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c,
        rng=np.random.default_rng(int(args.rng_seed)),
        cube_cadence_s=0.0,
        t_int_search_us=PROD_T_INT_SEARCH_US,
        pre_quantise=False,  # Per-cube fresh noise (so noise-only cubes are independent).
        symmetric_shift_padding=True,
        injections=(injection,),
    )
    _LOG.info(
        "synthetic source: n_cubes=%d injection=(cube=%d t=%d l=%d m=%d "
        "amp=%.1f) | M7.7 padding pad_left=%d pad_right=%d T_stream=%d",
        int(args.n_cubes), inject_cube_idx, int(args.inject_t_in_cube),
        int(args.inject_l_pix), int(args.inject_m_pix),
        float(args.inject_amplitude),
        src._pad_left, src._pad_right, src._t_stream,
    )

    expected_fdm = _expected_fine_dm_idx(src.time_shift_table)
    _LOG.info(
        "expected_fine_dm_idx for chgroup-15 delta injection: %d "
        "(fine_dm[%d]=%.3f pc cm^-3; shift_g15=%d samples)",
        expected_fdm, expected_fdm,
        float(fine[expected_fdm]),
        int(src.time_shift_table.shifts[expected_fdm, 15]),
    )

    pipeline, detector = _build_pipeline(
        device=device,
        n_fdm=n_fdm,
        cube_dtype=torch.float16,
        threshold_sigma=float(args.false_positive_threshold_sigma),
    )

    all_cands_per_cube: List[List[Candidate]] = []
    async with src:
        async for slot in src:
            res = pipeline.process(slot)
            cands_list = [c for c in res.candidates]
            all_cands_per_cube.append(cands_list)
            cands_above = [c for c in cands_list if c.snr >= float(args.false_positive_threshold_sigma)]
            _LOG.info(
                "cube=%d  total_cands=%d  cands>=%.1fσ=%d  max_snr=%.2f",
                slot.cube_id, len(cands_list),
                float(args.false_positive_threshold_sigma),
                len(cands_above),
                max((c.snr for c in cands_list), default=0.0),
            )
            await src.release(slot.cube_id)

    # Dump all candidates to disk for offline analysis.
    cands_path = out_dir / "candidates.ndjson"
    n_cands_total = 0
    with cands_path.open("w") as fh:
        for cube_idx, cands in enumerate(all_cands_per_cube):
            for c in cands:
                rec = dataclasses.asdict(c)
                rec["bench_cube_idx"] = int(cube_idx)
                fh.write(json.dumps(rec, default=str) + "\n")
                n_cands_total += 1
    _LOG.info(
        "wrote %s (%d candidates across %d cubes)",
        cands_path, n_cands_total, len(all_cands_per_cube),
    )

    # ----- Assertion 1: detector recovered the injection -----
    injection_cube_cands = (
        all_cands_per_cube[inject_cube_idx]
        if inject_cube_idx < len(all_cands_per_cube) else []
    )
    cube_specnum_start = inject_cube_idx * PROD_T_DET
    matched = _classify_candidate_recovery(
        injection_cube_cands,
        inject_cube_idx=inject_cube_idx,
        inject_t_in_cube=int(args.inject_t_in_cube),
        inject_l_pix=int(args.inject_l_pix),
        inject_m_pix=int(args.inject_m_pix),
        expected_fine_dm_idx=expected_fdm,
        cube_specnum_start=cube_specnum_start,
    )

    # ----- Assertion 3: no false positives in noise-only cubes -----
    false_positives = _candidates_above_threshold_noise_only(
        all_cands_per_cube,
        inject_cube_idx=inject_cube_idx,
        threshold_sigma=float(args.false_positive_threshold_sigma),
    )

    # Build run summary first so we can persist even on FAIL.
    run = {
        "schema_version": 1,
        "bench": "preflight.search_correctness_gate",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "n_cubes": int(args.n_cubes),
            "owner_idx": int(args.owner_idx),
            "n_fdm": n_fdm,
            "t_det": PROD_T_DET,
            "n_grid": PROD_N_GRID,
            "threshold_sigma": float(args.false_positive_threshold_sigma),
            "device": str(device),
            "rng_seed": int(args.rng_seed),
        },
        "injection": {
            "cube_idx": inject_cube_idx,
            "t_in_cube": int(args.inject_t_in_cube),
            "l_pix": int(args.inject_l_pix),
            "m_pix": int(args.inject_m_pix),
            "amplitude": float(args.inject_amplitude),
            "expected_fine_dm_idx": expected_fdm,
            "fine_dm_pc_cm3": float(fine[expected_fdm]),
        },
        "n_candidates_total": n_cands_total,
        "n_candidates_per_cube": [len(c) for c in all_cands_per_cube],
        "recovered": dataclasses.asdict(matched) if matched is not None else None,
        "false_positives": [dataclasses.asdict(c) for c in false_positives],
        "snr_floor_required": float(args.snr_floor),
    }

    # Print human-readable summary + PASS/FAIL banner.
    print()
    print("=" * 72)
    print("Search-side correctness preflight gate — summary")
    print("=" * 72)
    print(
        f"  injection truth: cube={inject_cube_idx} t={args.inject_t_in_cube} "
        f"l={args.inject_l_pix} m={args.inject_m_pix} "
        f"amp={args.inject_amplitude} expected_fdm={expected_fdm}"
    )
    if matched is not None:
        print(
            f"  recovered:       cube={inject_cube_idx} "
            f"event_specnum={int(matched.event_specnum)} (offset "
            f"{int(matched.event_specnum) - cube_specnum_start - int(args.inject_t_in_cube):+d}) "
            f"l={int(matched.l):>3d} (Δ{int(matched.l) - int(args.inject_l_pix):+d}) "
            f"m={int(matched.m):>3d} (Δ{int(matched.m) - int(args.inject_m_pix):+d}) "
            f"fdm={int(matched.dm_idx):>3d} (Δ{int(matched.dm_idx) - expected_fdm:+d}) "
            f"snr={float(matched.snr):.2f} kernel={matched.kernel_id}"
        )
    else:
        print("  recovered:       <NO IN-WINDOW CANDIDATE>")

    print()
    print("  noise-only cubes:")
    for cube_idx, cands in enumerate(all_cands_per_cube):
        if cube_idx == inject_cube_idx:
            continue
        max_snr = max((float(c.snr) for c in cands), default=0.0)
        n_above = sum(
            1 for c in cands
            if float(c.snr) >= float(args.false_positive_threshold_sigma)
        )
        print(
            f"    cube={cube_idx:>3d}  n_cands={len(cands):>4d}  "
            f"max_snr={max_snr:>6.2f}  n_above_{args.false_positive_threshold_sigma:.0f}σ={n_above}"
        )

    print()
    pass_recovery = matched is not None
    pass_snr = (matched is not None) and (float(matched.snr) >= float(args.snr_floor))
    pass_no_fp = (len(false_positives) == 0)

    print("Assertions:")
    print(
        f"  [{'PASS' if pass_recovery else 'FAIL'}] recovery: candidate in "
        f"window (Δl,Δm ≤ {PROD_LM_TOL_CELLS}, Δfdm ≤ {PROD_DM_TOL_FINE}, "
        f"Δt ≤ {PROD_T_TOL_SAMPLES})"
    )
    print(
        f"  [{'PASS' if pass_snr else 'FAIL'}] SNR floor:    recovered "
        f"SNR ≥ {float(args.snr_floor):.1f} σ"
    )
    print(
        f"  [{'PASS' if pass_no_fp else 'FAIL'}] no false +:   noise-only "
        f"cubes (excluding injection cube {inject_cube_idx}) have "
        f"no Candidate ≥ {float(args.false_positive_threshold_sigma):.1f} σ "
        f"(saw {len(false_positives)} false positive(s))"
    )

    overall_pass = pass_recovery and pass_snr and pass_no_fp
    run["gate_status"] = "PASS" if overall_pass else "FAIL"
    run["assertions"] = {
        "recovery": pass_recovery,
        "snr_floor": pass_snr,
        "no_false_positives": pass_no_fp,
    }

    summary_path = out_dir / "run.json"
    summary_path.write_text(json.dumps(run, indent=2, default=str))
    print(f"\nwrote {summary_path}")

    print()
    if overall_pass:
        print(
            f"  ✓ PASS — search-side correctness healthy at production "
            f"op-point. Safe to fleet-push."
        )
        return 0
    print(
        f"  ✗ FAIL — at least one assertion failed. DO NOT fleet-push; "
        f"see the failed assertion(s) above and the per-cube candidates "
        f"in {cands_path}."
    )
    return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search-side correctness preflight gate (M7.7+; "
                    "run on a GPU node before any fleet-push)."
    )
    p.add_argument(
        "--n-cubes", type=int, default=15,
        help="Number of cubes to drive through. 15 = 5 warmup + 1 "
             "injection + 9 noise-only (plenty for false-positive "
             "statistics). Default 15.",
    )
    p.add_argument(
        "--inject-cube-idx", type=int, default=8,
        help="Cube index where the injection lands. Must be > 5 "
             "(Layer-1 burn-in) and < --n-cubes. Default 8.",
    )
    p.add_argument(
        "--inject-t-in-cube", type=int, default=PROD_T_DET // 2,
        help="Cube-time bin (0..t_det-1) of the injection. Default "
             "t_det/2 = 96 (mid-cube; far from boundaries).",
    )
    p.add_argument(
        "--inject-l-pix", type=int, default=PROD_N_GRID // 2,
        help=f"l grid cell of the injection. Default {PROD_N_GRID//2} "
             "(image centre; far from edge mask).",
    )
    p.add_argument(
        "--inject-m-pix", type=int, default=PROD_N_GRID // 2,
        help=f"m grid cell of the injection. Default {PROD_N_GRID//2} "
             "(image centre; far from edge mask).",
    )
    p.add_argument(
        "--inject-amplitude", type=float, default=200.0,
        help="Injection delta-amplitude in noise-sigma units. Default "
             "200 produces a ~tens-of-σ Candidate after the imager + "
             "matched-filter chain, well above the SNR floor.",
    )
    p.add_argument(
        "--snr-floor", type=float, default=30.0,
        help="Recovered-candidate SNR must be >= this. Default 30 σ; "
             "tune if you adjust --inject-amplitude.",
    )
    p.add_argument(
        "--false-positive-threshold-sigma", type=float,
        default=PROD_THRESHOLD_SIGMA,
        help=f"False-positive threshold (σ). Default {PROD_THRESHOLD_SIGMA} "
             "matches production c1.snr_min. Any Candidate from a "
             "noise-only cube above this triggers the no-false-+ FAIL.",
    )
    p.add_argument(
        "--owner-idx", type=int, default=0,
        help="Coarse-DM owner index (0..7) for the slice. Default 0.",
    )
    p.add_argument(
        "--dm-plan-path", type=str, default=PROD_DM_PLAN_PATH,
        help=f"Production DM plan NPZ. Default {PROD_DM_PLAN_PATH}.",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="GPU device. Default cuda:0. Combine with "
             "CUDA_VISIBLE_DEVICES=<id> to pin a physical GPU.",
    )
    p.add_argument(
        "--rng-seed", type=int, default=0,
        help="RNG seed for the per-cube noise. Default 0.",
    )
    p.add_argument(
        "--out", type=str, default="/tmp/m77_correctness_gate",
        help="Output directory for candidates.ndjson + run.json. "
             "Default /tmp/m77_correctness_gate.",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root log level. Default INFO.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    print(
        f"[gate] dsa110-rt search correctness preflight "
        f"(UTC {datetime.now(timezone.utc).isoformat(timespec='seconds')})"
    )
    print(
        f"[gate] op-point: n_grid={PROD_N_GRID} t_det={PROD_T_DET} "
        f"owner_idx={args.owner_idx} M7.7=ON inject_amp={args.inject_amplitude}"
    )
    return asyncio.run(_run_gate(args))


if __name__ == "__main__":
    sys.exit(main())
