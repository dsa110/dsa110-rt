#!/usr/bin/env python3
"""bench/voltage_fixture_search.py — M5 Chunk 7 voltage-fixture search
gate (plan §8 line 2330).

End-to-end M5 closure gate: feed the M3-captured transport-TX ``.npz``
set (per chunk 6 `bench/corr_fast_burst_250924mptq.py` output) into
the search-compute pipeline and verify the burst is recovered as a
trigger packet at the expected (l, m, fine_dm, t) cell.

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
        ``--captured-dir <path>`` (per F6 schema; the loader will
        adapt to M3's published schema during chunk-7 hardening),
        rebuilds the per-chgroup uv-grid streams, drives the search-
        compute pipeline at production cube geometry, and stamps
        ``bench/reports/M5/m_operator_approved.yaml`` with PASS only
        after operator inspection of the recovered (l, m, fine_dm)
        cell against the fixture's known truth.

Outputs (under ``--out``):

  * ``run.json``       — the full run record:
        ``{config, mode, fixture: {dm, ra_deg, dec_deg, mjd, snr},
           recovered: {l_pix, m_pix, fine_dm_idx, t_in_cube, snr,
                       kernel_id} or null,
           gate_status: PASS|FAIL|NEEDS_OPERATOR}``
  * ``triggers.ndjson`` — per-emitted ``TriggerPacket``.
  * ``bench.log``      — operator progress.

Operator gate semantics:
  * synthetic: PASS/FAIL automatic (recovery_window check).
  * captured: NEEDS_OPERATOR until the bench is wired against M3's
    real npz schema + the operator inspects the report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
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
from dsart.trigger.conditions import (  # noqa: E402
    PerCubePerKernelCap,
    PerCubeTotalCap,
    RateLimitTokenBucket,
    SnrThreshold,
)
from dsart.trigger.emitter import (  # noqa: E402
    ConnectionEndpoint,
    TriggerEmitter,
    TriggerEmitterConfig,
)
from dsart.trigger.holdoff import HoldoffStateMachine  # noqa: E402
from dsart.trigger.mock_listener import (  # noqa: E402
    MockListenerConfig,
    MockTriggerListener,
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
    Layer-1 + detector + emitter + listener).
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
) -> Tuple[Dict[int, np.ndarray], Dict[str, object]]:
    """Load the M3-emitted captured transport-TX ``.npz`` set.

    F6 documents the M3 → M5 coupling point: M3 owns the schema
    (the chunk-6 burst bench produces 16 ``.npz`` files, one per
    chgroup, each containing the post-stage-2 dedispersed dense
    uv-grid stream + a sidecar manifest with the cube geometry).
    Per F6 the schema is intended to land in
    ``src/dsart/transport/captured_npz.py`` once M3 chunk-8 hardens.

    Until the schema is locked, this loader raises a clear error
    pointing at the chunk-7 hardening TODO. The mode-synthetic path
    (default) does NOT call this function and so the chunk-7 chunk-7
    skeleton lands without an M3 dependency.

    Args:
        captured_dir: directory containing 16 ``.npz`` files +
            ``manifest.json``.

    Returns:
        Tuple of (per-chgroup dense uv-streams dict, manifest dict).

    Raises:
        NotImplementedError until M3 publishes the schema.
    """
    raise NotImplementedError(
        "voltage_fixture_search.py mode=captured needs the M3-published "
        "captured-npz schema (see M5_PLAN_FIXES.md F6). The chunk-7 "
        "synthetic path is the immediate gate; chunk-7 hardening swaps "
        "in the real loader once `src/dsart/transport/captured_npz.py` "
        "is published by M3 (transport-TX → search-compute coupling "
        "point per PARALLEL_AGENTS.md §1)."
    )


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


async def _bench_synthetic(args: argparse.Namespace) -> Dict[str, object]:
    """End-to-end mode-synthetic run: synthetic RX-ring → CubePipeline
    → emitter → mock listener. Returns the run record dict.
    """
    t_det = int(args.t_det)
    n_fdm = int(args.n_fdm)
    n_grid = int(args.n_grid)

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

    # ---- Listener + emitter ----
    listener = MockTriggerListener(
        host="127.0.0.1", port=int(args.listener_port),
        config=MockListenerConfig(
            accept_rate=1.0, accept_delay_ms=0.0,
            completed_delay_ms=0.5, send_completed=True,
        ),
    )
    await listener.start()
    endpoint = ConnectionEndpoint(host=listener.host, port=listener.port)
    emitter_cfg = TriggerEmitterConfig(
        search_node_id=1,
        gpu_half=1,
        endpoints=[endpoint],
        conditions=[
            SnrThreshold(min_snr=float(args.threshold_sigma)),
            PerCubePerKernelCap(max_per_kernel=128),
            PerCubeTotalCap(max_total=1024),
            RateLimitTokenBucket(rate_per_s=1e6, burst=1_000_000),
        ],
        holdoff=HoldoffStateMachine(holdoff_ms=0.0),
    )
    emitter = TriggerEmitter(emitter_cfg)
    await emitter.start()

    all_records = []
    all_cands: List[Candidate] = []
    try:
        async with src:
            async for slot in src:
                result = pipeline.process(slot)
                all_cands.extend(result.candidates)
                records = await emitter.process_candidates(
                    slot.cube_id, result.candidates,
                )
                all_records.extend(records)
                await src.release(slot.cube_id)
    finally:
        await emitter.stop()
        await listener.stop()

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
        "n_records_dispatched": len(all_records),
        "gate_status": gate_status,
    }


async def _bench_captured(args: argparse.Namespace) -> Dict[str, object]:
    """End-to-end mode-captured run: M3-captured npz → CubePipeline →
    emitter → listener. Currently raises NotImplementedError pending
    M3's npz schema (F6).
    """
    if not args.captured_dir:
        raise SystemExit(
            "--mode captured requires --captured-dir <path>"
        )
    captured_dir = Path(args.captured_dir).resolve()
    _load_captured_npz_set(captured_dir)  # raises until F6 lands
    return {}  # unreachable; here for type completeness


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
    return 0 if run_record.get("gate_status") in ("PASS", "NEEDS_OPERATOR") else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 chunk-7 voltage-fixture search gate"
    )
    parser.add_argument(
        "--mode", choices=("synthetic", "captured"), default="synthetic",
        help="synthetic: M3-independent injection through full pipeline; "
             "captured: M3 → M5 integration (needs M3 npz schema, F6).",
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
    parser.add_argument("--listener-port", type=int, default=11227)
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "voltage_fixture" / "M5"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())
