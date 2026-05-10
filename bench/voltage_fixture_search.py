#!/usr/bin/env python3
"""bench/voltage_fixture_search.py — M5 Chunk 7 voltage-fixture search
gate (plan §8 line 2330).

End-to-end M5 closure gate: feed the M3-captured transport-TX ``.npz``
set (per chunk 6 `bench/corr_fast_burst_250924mptq.py` output) into
the search-compute pipeline and verify the burst is recovered as a
``Candidate`` at the expected (l, m, fine_dm, t) cell.

Plan §8 line 2330 specifies the fixture as the 250924mptq burst
(DM ≈ 404.7 pc cm⁻³, RA = 307.78°, Dec = 53.85°, MJD ≈ 60942.172,
SNR ≈ 30) — see ``M5_PLAN_FIXES.md`` D4. The bench is the **only**
M3 → M5 coupling point per ``PARALLEL_AGENTS.md`` §1.

Two CLI modes:

  * ``--mode synthetic`` (chunk-7 default; M3-independent):
        Bench runs against a synthetic burst injected through the
        chunk-6b synthetic RX-ring + cube pipeline. Verifies the
        wiring catches a strong burst at known (l_pix, m_pix,
        fine_dm_idx, t_in_cube). PASS gate fires automatically.

  * ``--mode captured`` (M5 closure path; needs M3 captures):
        Loads the M3-emitted ``.npz`` set from
        ``--captured-dir <path>`` via
        :func:`dsart.transport.captured_npz.load_captured_run` (F6
        resolved 2026-05-06; M3 chunk-8 fixture writer is at
        ``bench/m3_emit_m5_fixtures.py``), rebuilds the per-chgroup
        ``[N_chgroup=16, n_fv_total, N_grid, N_grid]`` uv-grid stream
        stack, and currently emits an INSPECTION_ONLY record (cube
        shape + T2 truth + valid_mask). The detector sweep against
        the captured stack is the next chunk-7 hardening item — once
        the fused-combine GPU path supports cf64-input cubes (or the
        cf64 → cint8 quant step is wired), the bench will close the
        loop and stamp ``bench/reports/M5/m_operator_approved.yaml``
        with PASS only after operator inspection of the recovered
        (l, m, fine_dm) cell against the fixture's T2 truth.

Outputs (under ``--out``):

  * ``run.json``        — the full run record:
        ``{config, mode, fixture: {dm, ra_deg, dec_deg, mjd, snr},
           recovered: {l_pix, m_pix, fine_dm_idx, t_in_cube, snr,
                       kernel_id} or null,
           gate_status: PASS|FAIL|NEEDS_OPERATOR}``
  * ``candidates.ndjson`` — per-emitted ``Candidate`` (one JSON line
        each, from ``dataclasses.asdict``). Synthetic mode only;
        captured mode does not surface per-Candidate output yet.
  * ``bench.log``       — operator progress.

Operator gate semantics:
  * synthetic: PASS/FAIL automatic (recovery_window check).
  * captured: INSPECTION_ONLY (loader smoke-test) → NEEDS_OPERATOR
    once the detector sweep is wired (chunk-7 hardening). PASS lands
    when the operator approves a recovered burst against fixture truth.
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.common.contracts import Candidate  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import (  # noqa: E402
    SyntheticInjection,
    SyntheticRxRingSource,
)


_LOG = logging.getLogger("bench.voltage_fixture_search")


# ---------------------------------------------------------------------------
# Plan §8 line 2330 + D4 fixture truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixtureTruth:
    """Known sky truth for the M5 voltage-fixture (250924mptq) per D4."""

    name: str = "250924mptq"
    dm_pc_cm3: float = 404.7
    ra_deg: float = 307.78
    dec_deg: float = 53.85
    mjd: float = 60942.172
    snr: float = 30.0


DEFAULT_FIXTURE = FixtureTruth()


# ---------------------------------------------------------------------------
# Mode-synthetic helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticBurstSpec:
    """Synthetic burst placed into the chunk-6b RX-ring source for the
    chunk-7 chunk-7 mode-synthetic run. The bench bypasses the M3
    transport entirely; the imager + detector see a unit-σ noise cube
    with a strong δ injection at the requested cell. Distinct from the
    chunk-5 ``cube_injection_detector`` bench because chunk-7 still
    drives the **full search_compute pipeline** (combiner + imager +
    Layer-1 + detector). M6 chunk 0 retired the trigger emitter +
    listener; per-Candidate output now flows directly from the
    pipeline result.
    """
    cube_idx: int
    t_in_cube: int
    l_pix: int
    m_pix: int
    amplitude: float = 200.0


def _build_dm_grids(n_fdm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic coarse/fine DM grid; the synthetic burst lands at
    fine_dm_idx = n_fdm // 2 (we don't gate on dispersion correctness
    in mode-synthetic — that's chunk 5's job)."""
    n_coarse = max(2, n_fdm // 2)
    n_fine_per_coarse = max(1, n_fdm // n_coarse)
    coarse = np.linspace(50.0, 200.0, n_coarse, dtype=np.float64)
    spacing = (
        (coarse[1] - coarse[0]) / n_fine_per_coarse if n_coarse > 1 else 1.0
    )
    fine = np.concatenate(
        [coarse[c] + np.arange(n_fine_per_coarse) * spacing for c in range(n_coarse)]
    )
    fine = fine[:n_fdm]
    fine_to_coarse = np.repeat(
        np.arange(n_coarse, dtype=np.int64), n_fine_per_coarse
    )[:n_fdm]
    return coarse, fine, fine_to_coarse


# ---------------------------------------------------------------------------
# Mode-captured stub (M3 → M5 integration point)
# ---------------------------------------------------------------------------


def _load_captured_npz_set(
    captured_dir: Path,
) -> Tuple[np.ndarray, List[bool], "CapturedManifest"]:
    """Load the M3-emitted captured transport-TX ``.npz`` set.

    F6 (M3 → M5 coupling point) was resolved 2026-05-06: the schema
    + on-disk layout are documented in
    ``src/dsart/transport/captured_npz.py`` and produced by the M3
    bench ``bench/m3_emit_m5_fixtures.py`` (M3 worktree, commit
    f9981f8). M3 emits per-chgroup F26 sparse-COO NPZ + a manifest
    under ``/home/ubuntu/data/m5_fixtures/<run_id>/``.

    This wrapper:
    1. Loads + validates the manifest + every chgroup NPZ (strict
       schema check; missing fields raise ValueError).
    2. Scatters each chgroup's sparse-COO vis cube to a dense
       ``[n_fv_total, N_grid, N_grid] complex64`` slab.
    3. Stacks them into the production-shape
       ``[N_chgroup=16, n_fv_total, N_grid, N_grid]`` array, with
       missing chgroups (eg sb12 in 0319) zero-filled (consult
       ``valid_mask`` for which slots are real).

    Args:
        captured_dir: directory containing ``manifest.json`` plus one
            ``chgroupNN.npz`` per channel group (typically 16; 0319
            is 15 due to the known sb12 data gap).

    Returns:
        Tuple of:
        - ``streams`` ``[N_chgroup=16, n_fv_total, N_grid, N_grid]``
          complex64 dense stack (zero-filled missing slots).
        - ``valid_mask`` ``List[bool]`` of length 16, True at slots
          that came from real data + False at zero-filled slots.
        - ``manifest`` ``CapturedManifest`` with cross-chgroup metadata
          + T2 truth (burst fixtures only).

    Raises:
        FileNotFoundError if manifest or any chgroup NPZ missing.
        ValueError on schema violation or cross-chgroup invariant
            disagreement.
    """
    from dsart.transport.captured_npz import (
        load_captured_run, stack_dense_streams,
    )
    chgroups, manifest = load_captured_run(captured_dir)
    streams, valid_mask = stack_dense_streams(
        chgroups, fill_missing=True, n_chgroup_total=16,
    )
    return streams, valid_mask, manifest


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


async def _bench_synthetic(args: argparse.Namespace) -> Dict[str, object]:
    """End-to-end mode-synthetic run: synthetic RX-ring → CubePipeline.

    Per-cube ``Candidate`` lists are collected directly from
    ``CubePipelineResult`` (M6 chunk 0 retired the M5 trigger emitter
    + listener). Returns the run record dict.
    """
    t_det = int(args.t_det)
    n_fdm = int(args.n_fdm)
    n_grid = int(args.n_grid)
    out_dir = Path(args.out).resolve()

    # Place the burst at the cube centre (well within the time-edge
    # gate's K_time/2 buffer; lm at phase centre for unambiguous
    # recovery; fine_dm at the centre of the trial grid).
    burst = SyntheticBurstSpec(
        cube_idx=0,
        t_in_cube=t_det // 2,
        l_pix=n_grid // 2,
        m_pix=n_grid // 2,
        amplitude=float(args.synthetic_amplitude),
    )
    expected_fine_dm_idx = n_fdm // 2

    coarse_dm, fine_dm, fine_to_coarse = _build_dm_grids(n_fdm)
    src = SyntheticRxRingSource(
        n_cubes=1,
        t_det=t_det,
        n_fdm=n_fdm,
        n_grid=n_grid,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        rng=np.random.default_rng(int(args.rng_seed)),
        injections=(SyntheticInjection(
            cube_idx=burst.cube_idx,
            t_in_cube=burst.t_in_cube,
            l_pix=burst.l_pix,
            m_pix=burst.m_pix,
            amplitude=burst.amplitude,
        ),),
    )
    pipeline = CubePipeline(
        config=CubePipelineConfig(
            n_grid=n_grid, edge_mask_kernel_support=5,
            cube_dtype=torch.float32, device="cpu",
        ),
        detector=DeterministicDetector(
            threshold_sigma=float(args.threshold_sigma),
            detector_version="v1.M5",
            search_node_id=1,
            gpu_half=1,
            dtype=torch.float32,
        ),
        layer1_state=Layer1State(n_fdm=n_fdm, n_burnin_cubes=1),
    )

    all_cands: List[Candidate] = []
    async with src:
        async for slot in src:
            result = pipeline.process(slot)
            all_cands.extend(result.candidates)
            await src.release(slot.cube_id)

    candidates_path = out_dir / "candidates.ndjson"
    with candidates_path.open("w") as fh:
        for c in all_cands:
            fh.write(json.dumps(dataclasses.asdict(c)) + "\n")
    _LOG.info("wrote %s (%d candidates)", candidates_path, len(all_cands))

    # Find the candidate closest to the expected (l, m, fine_dm, t).
    recovered: Optional[Dict[str, object]] = None
    if all_cands:
        def dist(c: Candidate) -> float:
            return math.sqrt(
                (c.l - burst.l_pix) ** 2
                + (c.m - burst.m_pix) ** 2
                + (c.dm_idx - expected_fine_dm_idx) ** 2
                + ((c.event_specnum - burst.t_in_cube) / max(1, t_det)) ** 2
            )
        best = min(all_cands, key=dist)
        recovered = {
            "l_pix": float(best.l),
            "m_pix": float(best.m),
            "dm_idx": int(best.dm_idx),
            "event_specnum": int(best.event_specnum),
            "snr": float(best.snr),
            "kernel_id": str(best.kernel_id),
        }
    # PASS if recovered AND within match window (lm ≤ 2, dm ≤ 2,
    # t window scales with K_time/2).
    gate_status = "FAIL"
    if recovered is not None:
        lm_ok = (
            abs(recovered["l_pix"] - burst.l_pix) <= 2
            and abs(recovered["m_pix"] - burst.m_pix) <= 2
        )
        dm_ok = abs(recovered["dm_idx"] - expected_fine_dm_idx) <= 2
        t_ok = abs(recovered["event_specnum"] - burst.t_in_cube) <= 64
        if lm_ok and dm_ok and t_ok:
            gate_status = "PASS"

    return {
        "mode": "synthetic",
        "config": {
            "t_det": t_det, "n_fdm": n_fdm, "n_grid": n_grid,
            "threshold_sigma": float(args.threshold_sigma),
            "synthetic_amplitude": float(args.synthetic_amplitude),
            "rng_seed": int(args.rng_seed),
        },
        "fixture": {
            "name": DEFAULT_FIXTURE.name,
            "dm_pc_cm3": DEFAULT_FIXTURE.dm_pc_cm3,
            "ra_deg": DEFAULT_FIXTURE.ra_deg,
            "dec_deg": DEFAULT_FIXTURE.dec_deg,
            "mjd": DEFAULT_FIXTURE.mjd,
            "snr": DEFAULT_FIXTURE.snr,
        },
        "synthetic_burst": {
            "cube_idx": burst.cube_idx,
            "t_in_cube": burst.t_in_cube,
            "l_pix": burst.l_pix,
            "m_pix": burst.m_pix,
            "amplitude": burst.amplitude,
            "expected_fine_dm_idx": expected_fine_dm_idx,
        },
        "recovered": recovered,
        "n_candidates": len(all_cands),
        "gate_status": gate_status,
    }


async def _bench_captured(args: argparse.Namespace) -> Dict[str, object]:
    """End-to-end mode-captured run: M3-captured NPZ → loader → cube
    summary, optionally followed by the v1 detector sweep.

    Two sub-modes (operator-controlled via ``--detector-sweep``):

      ``--detector-sweep`` ON (chunk-7 closure default for is_burst
        captures): delegates to
        :mod:`bench.captured_burst_detector` (D24) which loads the cf
        streams, quantises to cint8, runs the production GpuImager
        + Layer-1 σ-clip + 8-kernel collapsed-bank streaming detector,
        and stamps PASS / FAIL based on burst recovery + matched-
        filter K_time SNR boost. The captured_burst_detector's
        ``detector.json`` is folded into this bench's ``run.json``
        envelope.

      ``--detector-sweep`` OFF: legacy INSPECTION_ONLY behaviour
        (loader smoke-test only); useful when the operator wants to
        verify the M3 → M5 hand-off shape without burning the GPU.

    The voltage_fixture_search.run.json envelope keeps the M5
    operator-approval marker stable: ``m_operator_approved.yaml``
    references this bench's ``voltage_run_id`` (= manifest.run_id),
    not the captured_burst_detector's report directly.
    """
    if not args.captured_dir:
        raise SystemExit(
            "--mode captured requires --captured-dir <path>"
        )
    captured_dir = Path(args.captured_dir).resolve()
    streams, valid_mask, manifest = _load_captured_npz_set(captured_dir)

    n_chg, n_fv, n_grid, n_grid_y = streams.shape
    assert n_grid == n_grid_y, "streams must be square"
    n_valid = sum(valid_mask)
    abs_max = float(np.abs(streams).max()) if streams.size else 0.0
    abs_mean = float(np.abs(streams).mean()) if streams.size else 0.0

    _LOG.info(
        "captured loader: run=%s src_kind=%s n_chgroup_present=%d/%d "
        "n_fv_total=%d N_grid=%d streams_bytes=%d (abs max=%.3g mean=%.3g)",
        manifest.run_id, manifest.src_kind, n_valid, n_chg,
        n_fv, n_grid, streams.nbytes, abs_max, abs_mean,
    )

    record: Dict[str, object] = {
        "mode": "captured",
        "captured_dir": str(captured_dir),
        "manifest": {
            "run_id": manifest.run_id,
            "src_kind": manifest.src_kind,
            "src_name": manifest.src_name,
            "obs_dec_deg": manifest.obs_dec_deg,
            "t_int_fast_us": manifest.t_int_fast_us,
            "t_int_fast_native": manifest.t_int_fast_native,
            "n_chgroups_in_manifest": manifest.n_chgroups,
            "chgroups": list(manifest.chgroups),
            "git_sha": manifest.git_sha,
            "utc_iso": manifest.utc_iso,
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
        "streams_shape": {
            "n_chgroup_total": int(n_chg),
            "n_chgroup_present": int(n_valid),
            "n_fv_total": int(n_fv),
            "n_grid": int(n_grid),
            "valid_mask": list(valid_mask),
            "streams_dtype": str(streams.dtype),
            "streams_bytes": int(streams.nbytes),
            "abs_max": abs_max,
            "abs_mean": abs_mean,
        },
    }

    # Free the cf64 streams: captured_burst_detector reloads from
    # disk in its own process-managed memory budget. Holding the
    # full [16, n_fv_total, 256, 256] cf64 stack here (~1 GiB) on
    # top of the imager's GPU workspace would risk an OOM during
    # the detector sweep.
    del streams

    if not getattr(args, "detector_sweep", False):
        record["gate_status"] = "INSPECTION_ONLY"
        record["detector_sweep"] = None
        return record

    # ---- Detector sweep delegated to captured_burst_detector.
    # We import here (lazily) to avoid pulling torch/cupy in the
    # synthetic-mode code path (the synthetic mode uses the CPU
    # cube_pipeline + a small mock listener and runs in CI).
    import bench.captured_burst_detector as cbd  # noqa: E402

    detector_out = Path(args.out).resolve() / "detector"
    detector_out.mkdir(parents=True, exist_ok=True)
    cbd_args = argparse.Namespace(
        captured_dir=str(captured_dir),
        out=str(detector_out),
        detector_t_det=int(args.detector_t_det),
        n_fdm=int(args.detector_n_fdm),
        dm_min=float(args.detector_dm_min),
        dm_max=float(args.detector_dm_max),
        coarse_dm=float(args.detector_coarse_dm),
        shift_offset=int(args.detector_shift_offset),
        threshold_sigma=float(args.threshold_sigma),
        target_max=int(args.detector_target_max),
        merge_radius_lm=int(args.detector_merge_radius_lm),
        merge_radius_fdm=int(args.detector_merge_radius_fdm),
        merge_radius_t=int(args.detector_merge_radius_t),
    )
    _LOG.info(
        "captured detector sweep: T_det=%d N_fdm=%d threshold=%.1fσ "
        "DM∈[%.1f, %.1f] (writing %s)",
        cbd_args.detector_t_det, cbd_args.n_fdm, cbd_args.threshold_sigma,
        cbd_args.dm_min, cbd_args.dm_max, detector_out,
    )
    rc = cbd._bench_main(cbd_args)
    detector_record_path = detector_out / "detector.json"
    if not detector_record_path.exists():
        record["gate_status"] = "FAIL"
        record["detector_sweep"] = {
            "rc": int(rc),
            "error": "captured_burst_detector did not write detector.json",
        }
        return record
    detector_record = json.loads(detector_record_path.read_text())
    record["detector_sweep"] = {
        "rc": int(rc),
        "detector_json_path": str(detector_record_path),
        "n_candidates_post_merge": detector_record.get(
            "n_candidates_post_merge"
        ),
        "top_candidate": detector_record.get("top_candidate"),
        "burst_match": detector_record.get("burst_match"),
        "rfi_contamination": detector_record.get("rfi_contamination"),
        "timings_ms": detector_record.get("timings_ms"),
    }
    # captured_burst_detector returns rc=0 on PASS, rc=1 on FAIL.
    # The detector.json itself does NOT carry a gate_status field
    # (its PASS/FAIL is derived from burst_match on emit), so we
    # re-derive it here for the run.json envelope using the same
    # gate semantics (plan §8 line 2330):
    #   * is_burst fixture → PASS iff burst_match present + matched
    #     SNR ≥ threshold + matched_filter_snr_boost ≥ 1.0.
    #   * negative-control fixture → PASS iff NO candidates emitted.
    if manifest.is_burst:
        bm = detector_record.get("burst_match")
        record["gate_status"] = (
            "PASS"
            if (
                bm is not None
                and float(bm.get("matched_snr", 0.0))
                >= float(args.threshold_sigma)
                and float(bm.get("matched_filter_snr_boost", 0.0)) >= 1.0
            )
            else "FAIL"
        )
        if bm is not None:
            record["recovered"] = {
                "kernel_id": bm.get("matched_kernel_id"),
                "snr": bm.get("matched_snr"),
                "l_pix": bm.get("l_pix"),
                "m_pix": bm.get("m_pix"),
                "dm_idx": bm.get("dm_idx"),
                "dm_fine_pc_cc": bm.get("dm_fine_pc_cc"),
                "t_in_cube": bm.get("t_in_cube"),
                "matched_filter_snr_boost": bm.get(
                    "matched_filter_snr_boost"
                ),
                "labelled_dm_pc_cc": bm.get("labelled_dm_pc_cc"),
                "dm_residual_frac": bm.get("dm_residual_frac"),
            }
        else:
            record["recovered"] = None
    else:
        n_cand = int(detector_record.get("n_candidates_post_merge") or 0)
        record["gate_status"] = "PASS" if n_cand == 0 else "FAIL"
        record["recovered"] = None
    return record


async def _bench_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_log_path = out_dir / "bench.log"
    bench_log_handler = logging.FileHandler(bench_log_path, mode="w")
    bench_log_handler.setLevel(logging.INFO)
    bench_log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _LOG.setLevel(logging.INFO)
    _LOG.addHandler(bench_log_handler)
    _LOG.addHandler(logging.StreamHandler(sys.stdout))

    bench_start_ns = time.perf_counter_ns()
    if args.mode == "synthetic":
        run_record = await _bench_synthetic(args)
    elif args.mode == "captured":
        run_record = await _bench_captured(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode!r}")
    wall_s = (time.perf_counter_ns() - bench_start_ns) / 1.0e9

    run_record["wall_clock_s"] = wall_s
    run_record["utc"] = datetime.now(timezone.utc).isoformat()
    run_record["bench"] = "voltage_fixture_search"
    run_record["milestone"] = "M5"
    run_record["schema_version"] = 1

    run_path = out_dir / "run.json"
    with run_path.open("w") as fh:
        json.dump(run_record, fh, indent=2, sort_keys=True)
    _LOG.info("wrote %s", run_path)
    _LOG.info(
        "voltage_fixture_search [%s]: gate=%s recovered=%s",
        run_record.get("mode"),
        run_record.get("gate_status"),
        run_record.get("recovered") is not None,
    )
    return 0 if run_record.get("gate_status") in (
        "PASS", "NEEDS_OPERATOR", "INSPECTION_ONLY",
    ) else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 chunk-7 voltage-fixture search gate"
    )
    parser.add_argument(
        "--mode", choices=("synthetic", "captured"), default="synthetic",
        help="synthetic: M3-independent injection through full pipeline; "
             "captured: M3 → M5 integration via "
             "dsart.transport.captured_npz (F6 resolved 2026-05-06; "
             "currently INSPECTION_ONLY — detector sweep is the next "
             "chunk-7 hardening item).",
    )
    parser.add_argument(
        "--captured-dir", type=str, default=None,
        help="dir containing M3-emitted 16 .npz captured-TX files + "
             "manifest.json (mode=captured only).",
    )
    parser.add_argument("--t-det", type=int, default=64)
    parser.add_argument("--n-fdm", type=int, default=8)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument(
        "--threshold-sigma", type=float, default=8.0,
    )
    parser.add_argument("--synthetic-amplitude", type=float, default=300.0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "voltage_fixture" / "M5"),
    )

    # ----- captured-mode detector-sweep knobs (plan §8 line 2330).
    parser.add_argument(
        "--detector-sweep", action="store_true",
        help="captured-mode only: run the v1 detector sweep "
             "(D24 captured_burst_detector logic) and stamp PASS/FAIL "
             "based on burst recovery + matched-filter K_time SNR boost. "
             "Without this flag, captured-mode is INSPECTION_ONLY.",
    )
    parser.add_argument("--detector-t-det", type=int, default=256,
                        help="captured-mode detector cube T_det (default "
                             "256, the operator-pinned v1 deployment "
                             "integration time per D26).")
    parser.add_argument("--detector-n-fdm", type=int, default=32,
                        help="captured-mode detector N_fdm (default 32).")
    parser.add_argument("--detector-dm-min", type=float, default=370.0,
                        help="captured-mode fdm grid lower bound in pc/cc.")
    parser.add_argument("--detector-dm-max", type=float, default=420.0,
                        help="captured-mode fdm grid upper bound in pc/cc.")
    parser.add_argument("--detector-coarse-dm", type=float, default=0.0,
                        help="coarse DM the captured streams are claimed "
                             "to be dedispersed to. Default 0.0 since the "
                             "250924mptq fixture is empirically NOT pre-"
                             "stage-2 dedispersed (D23 caveat).")
    parser.add_argument("--detector-shift-offset", type=int, default=125,
                        help="add this many samples to all per-(fdm, "
                             "chgroup) shifts so the burst lands inside "
                             "the canonical zone of the cube (default 125 "
                             "puts the 250924mptq burst at cube_t ≈ 128, "
                             "the centre of T_det=256).")
    parser.add_argument("--detector-target-max", type=int, default=120,
                        help="cint8 quantisation target absolute max.")
    # Defaults pulled directly from dsart.detector.merger so the
    # bench tracks production NMS radii without manual sync.
    from dsart.detector import (  # noqa: E402
        DEFAULT_MERGE_RADIUS_LM, DEFAULT_MERGE_RADIUS_FDM,
        DEFAULT_MERGE_RADIUS_T,
    )
    parser.add_argument(
        "--detector-merge-radius-lm", type=int,
        default=DEFAULT_MERGE_RADIUS_LM,
    )
    parser.add_argument(
        "--detector-merge-radius-fdm", type=int,
        default=DEFAULT_MERGE_RADIUS_FDM,
    )
    parser.add_argument(
        "--detector-merge-radius-t", type=int,
        default=DEFAULT_MERGE_RADIUS_T,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())
