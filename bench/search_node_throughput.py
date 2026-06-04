#!/usr/bin/env python3
"""bench/search_node_throughput.py — M5 Chunk 6b-β throughput +
per-stage latency bench (plan §8 lines 2316-2317).

Drives the ``CubePipeline`` against a ``SyntheticRxRingSource`` at the
configured cube cadence, instruments per-stage wall-clock timing
(combiner+imager, Layer-1 normalisation, detector forward), and writes
a compact NDJSON / JSON result set the operator inspects via
``tools/viz/search_detector_check.py --mode throughput`` (the viz mode
lands in chunk-7 hardening; chunk-6b-β ships the producer + a CLI
summary).

Bench rate budget (production, plan §8 line 2317):
    cube cadence at default ops ........ 134 ms (7.45 cubes/s)
    end-to-end per-cube budget ......... ~30 ms

The bench runs h01 alone and is the per-stage perf gate for the M5
detector pipeline. It does NOT depend on M3 / M4a — the synthetic
RX-ring fills the role of M4a's POSIX-shm ring.

CLI surface (see ``--help`` for the full grid):

  python -m bench.search_node_throughput \\
      [--n-cubes 100]                                \\
      [--cube-cadence-s 0.0]                         \\
      [--t-det 64] [--n-fdm 8] [--n-grid 32]         \\
      [--threshold-sigma 8.0]                        \\
      [--out bench/reports/<UTC>/throughput/M5/]     \\
      [--quick-smoke]

Outputs (under ``--out``):

  * ``stage_timings.ndjson`` — one record per cube
        ``{cube_id, build_cube_ns, layer1_norm_ns, detector_forward_ns,
           total_pipeline_ns, n_candidates}``
  * ``summary.json``         — config + percentile summary
        ``{config: {...}, n_cubes, percentiles: {p50, p90, p99} per
           stage}``
  * ``bench.log``            — human-readable progress log

Operator gate: per-stage p99 < 30 ms total at default ops geometry
(``T_det=512, N_fdm=32, N_grid=256`` on cuda). The chunk-6b-β default
geometry (T_det=64) is sized for h01-CPU smoke; the operator runs the
full geometry on h01-GPU during M5 hardening. (M6 chunk 0 retired the
M5 trigger emitter; this bench no longer measures emitter-dispatch
latency — the corr-side dsa110-xengine framework owns voltage-trigger
fan-out in M6.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from bench._bank_mask import parse_bank_mask  # noqa: E402

from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import (  # noqa: E402
    SyntheticRxRingSource,
)
from dsart.services.search_compute import (  # noqa: E402
    _dm_grids_from_npz,
)


_LOG = logging.getLogger("bench.search_node_throughput")


# ---------------------------------------------------------------------------
# Bench config
# ---------------------------------------------------------------------------


# Default geometry: small enough for CPU h01 smoke; the GPU-heavy
# operator-facing run uses the plan-pinned T_det=512, N_fdm=32 (per-GPU),
# N_grid=256 via CLI overrides during M5 hardening.
DEFAULT_T_DET: int = 64
DEFAULT_N_FDM: int = 8
DEFAULT_N_GRID: int = 32
DEFAULT_N_CUBES: int = 50
DEFAULT_CUBE_CADENCE_S: float = 0.0
DEFAULT_THRESHOLD_SIGMA: float = 8.0

# --quick-smoke: minimal pass (5 cubes) for the M5.sh DoD path; full
# perf characterisation lives in the operator-facing runs.
QUICK_SMOKE_N_CUBES: int = 5
QUICK_SMOKE_T_DET: int = 32
QUICK_SMOKE_N_FDM: int = 4
QUICK_SMOKE_N_GRID: int = 16


@dataclass(frozen=True, slots=True)
class StageTimingRecord:
    """One cube's stage-timing record (NDJSON record)."""
    cube_id: int
    n_candidates: int
    build_cube_ns: int
    layer1_norm_ns: int
    detector_forward_ns: int
    total_pipeline_ns: int

    def to_json(self) -> Dict[str, int]:
        return {
            "cube_id": self.cube_id,
            "n_candidates": self.n_candidates,
            "build_cube_ns": self.build_cube_ns,
            "layer1_norm_ns": self.layer1_norm_ns,
            "detector_forward_ns": self.detector_forward_ns,
            "total_pipeline_ns": self.total_pipeline_ns,
        }


# ---------------------------------------------------------------------------
# Helpers: percentile rollups
# ---------------------------------------------------------------------------


def percentiles(values_ns: Sequence[int]) -> Dict[str, float]:
    """Return {p50, p90, p99, mean, max} in milliseconds."""
    if not values_ns:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values_ns, dtype=np.int64)
    arr_ms = arr.astype(np.float64) / 1.0e6
    return {
        "p50": float(np.percentile(arr_ms, 50)),
        "p90": float(np.percentile(arr_ms, 90)),
        "p99": float(np.percentile(arr_ms, 99)),
        "mean": float(arr_ms.mean()),
        "max": float(arr_ms.max()),
    }


# ---------------------------------------------------------------------------
# DM grid (synthetic; the bench doesn't gate on dispersion correctness)
# ---------------------------------------------------------------------------


def _build_dm_grids(n_fdm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic coarse/fine DM grid for the synthetic source."""
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
# Bench main
# ---------------------------------------------------------------------------


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

    if args.quick_smoke:
        n_cubes = QUICK_SMOKE_N_CUBES
        t_det = QUICK_SMOKE_T_DET
        n_fdm = QUICK_SMOKE_N_FDM
        n_grid = QUICK_SMOKE_N_GRID
    else:
        n_cubes = int(args.n_cubes)
        t_det = int(args.t_det)
        n_fdm = int(args.n_fdm)
        n_grid = int(args.n_grid)

    threshold_sigma = float(args.threshold_sigma)
    cube_cadence_s = float(args.cube_cadence_s)

    image_tokens, dm_tokens, time_tokens = parse_bank_mask(args.bank_mask)
    n_kernels_total = (
        len(image_tokens) * len(dm_tokens) * len(time_tokens)
    )

    _LOG.info(
        "bench config: n_cubes=%d cadence=%.3fs T_det=%d N_fdm=%d N_grid=%d "
        "threshold=%.2fσ",
        n_cubes, cube_cadence_s, t_det, n_fdm, n_grid, threshold_sigma,
    )
    _LOG.info(
        "bank-mask: k_img=%s k_dm=%s k_time=%s (total %d kernel triples)",
        list(image_tokens), list(dm_tokens), list(time_tokens),
        n_kernels_total,
    )

    # M7.7 (2026-06-04): if --dm-plan-path is given, load the EXACT
    # production v2 plan + optionally select one coarse-DM owner so the
    # bench sees the same shifts table (±83 samples for Option A) as
    # the live search_compute. Without this, the synthetic linear DM
    # grid produces all-positive shifts ≤ ~30 samples, which understates
    # the H2D cost and doesn't exercise pad_right > 0.
    dm_plan_path: Optional[str] = getattr(args, "dm_plan_path", None)
    coarse_dm_owner_idx = int(getattr(args, "coarse_dm_owner_idx", -1))
    if dm_plan_path is not None and Path(dm_plan_path).exists():
        coarse_dm, fine_dm, fine_to_coarse = _dm_grids_from_npz(
            Path(dm_plan_path), n_coarse=8
        )
        if coarse_dm_owner_idx >= 0:
            mask = fine_to_coarse == coarse_dm_owner_idx
            if not mask.any():
                raise SystemExit(
                    f"--coarse-dm-owner-idx {coarse_dm_owner_idx} not in "
                    f"fine_to_coarse range "
                    f"{int(fine_to_coarse.min())}..{int(fine_to_coarse.max())}"
                )
            fine_dm = fine_dm[mask]
            fine_to_coarse = fine_to_coarse[mask]
            # Remap surviving fine_to_coarse to local 0-based index, and
            # keep only the owner's coarse_dm row so shifts compute
            # cleanly. Mirrors `_select_dm_owner_half`'s local convention.
            coarse_dm = coarse_dm[coarse_dm_owner_idx : coarse_dm_owner_idx + 1]
            fine_to_coarse = np.zeros_like(fine_to_coarse, dtype=np.int32)
        n_fdm = int(fine_dm.shape[0])
        _LOG.info(
            "DM plan loaded from %s: n_coarse=%d n_fdm=%d "
            "(coarse_dm_owner_idx=%d)",
            dm_plan_path, len(coarse_dm), n_fdm, coarse_dm_owner_idx,
        )
    else:
        if dm_plan_path is not None:
            _LOG.warning(
                "--dm-plan-path %s does not exist; falling back to "
                "synthetic linear DM grid (shifts will NOT match "
                "production Option A geometry).",
                dm_plan_path,
            )
        coarse_dm, fine_dm, fine_to_coarse = _build_dm_grids(n_fdm)

    src = SyntheticRxRingSource(
        n_cubes=n_cubes,
        t_det=t_det,
        n_fdm=n_fdm,
        n_grid=n_grid,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        rng=np.random.default_rng(int(args.rng_seed)),
        cube_cadence_s=cube_cadence_s,
        t_int_search_us=float(getattr(args, "t_int_search_us", 1048.576)),
        pre_quantise=bool(args.prequantise),
        symmetric_shift_padding=bool(getattr(args, "symmetric_shift_padding", False)),
    )
    if bool(getattr(args, "symmetric_shift_padding", False)):
        _LOG.info(
            "SyntheticRxRingSource: --symmetric-shift-padding on (M7.7: "
            "pad_left=%d pad_right=%d T_stream=%d; slots stamped with "
            "stream_origin_offset_samples=%d so CubePipeline takes the "
            "M7.7 fused-L1 fast path).",
            src._pad_left, src._pad_right, src._t_stream, src._pad_left,
        )
    if args.prequantise:
        _LOG.info(
            "SyntheticRxRingSource: --prequantise on (M3 RX-ring "
            "chunk-8b emulation: one cf32 cube quantised once and "
            "yielded as cint8 every iteration; isolates GPU pipeline "
            "from host-side bench scaffolding)."
        )
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cube_dtype = torch.float16 if (
        args.cube_dtype == "fp16" and device != "cpu"
    ) else torch.float32
    detector_dtype = torch.float16 if (
        args.cube_dtype == "fp16" and device != "cpu"
    ) else torch.float32
    _LOG.info("device=%s cube_dtype=%s", device, cube_dtype)
    bank = build_kernel_bank(
        image_tokens=image_tokens,
        dm_tokens=dm_tokens,
        time_tokens=time_tokens,
        dtype=detector_dtype,
    )
    boxcar_accum_arg = str(getattr(args, "detector_boxcar_accum_dtype", "default"))
    if boxcar_accum_arg == "default":
        boxcar_accum_dtype: Optional[torch.dtype] = None
    elif boxcar_accum_arg == "fp32":
        boxcar_accum_dtype = torch.float32
    elif boxcar_accum_arg == "fp16":
        boxcar_accum_dtype = torch.float16
    elif boxcar_accum_arg == "bf16":
        boxcar_accum_dtype = torch.bfloat16
    else:
        raise SystemExit(
            f"--detector-boxcar-accum-dtype={boxcar_accum_arg!r}; expected "
            "one of default/fp32/fp16/bf16"
        )
    detector = DeterministicDetector(
        kernel_bank=bank,
        threshold_sigma=threshold_sigma,
        detector_version="v1.M5",
        search_node_id=1,
        gpu_half=1,
        dtype=detector_dtype,
        device=torch.device(device),
        streaming=bool(args.detector_streaming),
        streaming_tile_size=int(args.detector_streaming_tile_size),
        streaming_decoder_n_top=int(args.detector_streaming_decoder_n_top),
        boxcar_accum_dtype=boxcar_accum_dtype,
        layer2_sigma_max_samples=(
            int(args.detector_layer2_max_samples)
            if int(args.detector_layer2_max_samples) > 0
            else None
        ),
    )
    _LOG.info(
        "detector: streaming=%s tile_size=%d boxcar_accum_dtype=%s "
        "layer2_sigma_max_samples=%s",
        bool(args.detector_streaming), int(args.detector_streaming_tile_size),
        boxcar_accum_dtype,
        (
            int(args.detector_layer2_max_samples)
            if int(args.detector_layer2_max_samples) > 0
            else None
        ),
    )
    image_backend = str(args.image_backend)
    if image_backend == "gpu":
        if not device.startswith("cuda"):
            raise SystemExit(
                f"--image-backend gpu requires --device cuda*, got "
                f"--device {device!r}"
            )
        if cube_dtype != torch.float16:
            raise SystemExit(
                "--image-backend gpu currently pins cube_dtype=fp16 "
                "(GpuImager runs cuFFT-cfp16). Use --cube-dtype fp16."
            )
        gpu_complex_dtype = torch.complex32
    else:
        gpu_complex_dtype = torch.complex64

    pipeline_cfg = CubePipelineConfig(
        n_grid=n_grid,
        edge_mask_kernel_support=5,
        cube_dtype=cube_dtype,
        device=device,
        image_backend=image_backend,
        gpu_t_det=t_det if image_backend == "gpu" else None,
        gpu_n_fdm=n_fdm if image_backend == "gpu" else None,
        gpu_complex_dtype=gpu_complex_dtype,
    )
    _LOG.info(
        "image_backend=%s gpu_complex_dtype=%s",
        image_backend, gpu_complex_dtype,
    )
    layer1_max_samples = (
        int(args.layer1_max_samples)
        if args.layer1_max_samples is not None and int(args.layer1_max_samples) > 0
        else None
    )
    pipeline = CubePipeline(
        config=pipeline_cfg,
        detector=detector,
        layer1_state=Layer1State(
            n_fdm=n_fdm,
            n_burnin_cubes=5,
            max_samples=layer1_max_samples,
        ),
    )
    if layer1_max_samples is not None:
        _LOG.info(
            "Layer1State: max_samples=%d (per-fdm σ-clip subsample cap)",
            layer1_max_samples,
        )

    # ---- Drain the source, collecting stage timings ----
    records: List[StageTimingRecord] = []
    skip_detector = bool(args.skip_detector)
    enable_overlap = bool(args.pipeline_overlap)
    if skip_detector:
        _LOG.info(
            "--skip-detector: bypassing Layer-1 norm + Detector.forward(); "
            "measuring build_cube alone."
        )
    bench_start_ns = time.perf_counter_ns()
    full_prefetch = bool(getattr(args, "full_prefetch", False))
    async with src:
        if (
            full_prefetch
            and (not skip_detector)
            and image_backend == "gpu"
        ):
            # M7.2 cube-cadence experiment: full prefetch — runs the
            # NEXT cube's imager on a dedicated stream concurrent with
            # the CURRENT cube's Layer-1 + detector. With the imager
            # collapsed to the new cube_cadence samples (carry-over),
            # the cross-cube parallelism becomes worth the SM contention
            # because imager is now ~50% of layer1+det's cost.
            aiter = src.__aiter__()
            try:
                slot = await aiter.__anext__()
            except StopAsyncIteration:
                slot = None
            pending = (
                pipeline.prefetch_build(slot) if slot is not None else None
            )
            while slot is not None and pending is not None:
                t_dispatch_start = time.perf_counter_ns()
                try:
                    next_slot = await aiter.__anext__()
                except StopAsyncIteration:
                    next_slot = None
                next_pending = (
                    pipeline.prefetch_build(next_slot)
                    if next_slot is not None
                    else None
                )
                result = pipeline.process_prefetched(pending)
                t_done = time.perf_counter_ns()
                rec = StageTimingRecord(
                    cube_id=slot.cube_id,
                    n_candidates=len(result.candidates),
                    build_cube_ns=int(result.stage_timings_ns["build_cube"]),
                    layer1_norm_ns=int(result.stage_timings_ns["layer1_norm"]),
                    detector_forward_ns=int(
                        result.stage_timings_ns["detector_forward"]
                    ),
                    total_pipeline_ns=int(t_done - t_dispatch_start),
                )
                records.append(rec)
                if (slot.cube_id + 1) % max(1, n_cubes // 10) == 0:
                    _LOG.info(
                        "cube=%d/%d total=%.2fms detector=%.2fms cands=%d",
                        slot.cube_id + 1, n_cubes,
                        rec.total_pipeline_ns / 1.0e6,
                        rec.detector_forward_ns / 1.0e6,
                        rec.n_candidates,
                    )
                await src.release(slot.cube_id)
                slot, pending = next_slot, next_pending
        elif (
            enable_overlap
            and (not skip_detector)
            and image_backend == "gpu"
        ):
            # Chunk-8d narrow overlap: prefetch only the cint8 H2D for
            # cube N+1 on a dedicated H2D stream while the main stream
            # runs imager + Layer-1 + detector for cube N. Avoids the
            # SM contention that regressed the prior full-prefetch
            # overlap path; the H2D engine is independent of the SMs.
            aiter = src.__aiter__()
            try:
                slot = await aiter.__anext__()
            except StopAsyncIteration:
                slot = None
            pending = pipeline.prefetch_h2d(slot) if slot is not None else None
            while slot is not None and pending is not None:
                t_dispatch_start = time.perf_counter_ns()
                try:
                    next_slot = await aiter.__anext__()
                except StopAsyncIteration:
                    next_slot = None
                next_pending = (
                    pipeline.prefetch_h2d(next_slot)
                    if next_slot is not None
                    else None
                )
                result = pipeline.process_h2d_prefetched(pending)
                t_done = time.perf_counter_ns()
                rec = StageTimingRecord(
                    cube_id=slot.cube_id,
                    n_candidates=len(result.candidates),
                    build_cube_ns=int(result.stage_timings_ns["build_cube"]),
                    layer1_norm_ns=int(result.stage_timings_ns["layer1_norm"]),
                    detector_forward_ns=int(
                        result.stage_timings_ns["detector_forward"]
                    ),
                    total_pipeline_ns=int(t_done - t_dispatch_start),
                )
                records.append(rec)
                if (slot.cube_id + 1) % max(1, n_cubes // 10) == 0:
                    _LOG.info(
                        "cube=%d/%d total=%.2fms detector=%.2fms cands=%d",
                        slot.cube_id + 1, n_cubes,
                        rec.total_pipeline_ns / 1.0e6,
                        rec.detector_forward_ns / 1.0e6,
                        rec.n_candidates,
                    )
                await src.release(slot.cube_id)
                slot, pending = next_slot, next_pending
        else:
            async for slot in src:
                t_dispatch_start = time.perf_counter_ns()
                if skip_detector:
                    cube, validity_mask = pipeline._build_cube(slot)
                    t_build_done = time.perf_counter_ns()
                    rec = StageTimingRecord(
                        cube_id=slot.cube_id,
                        n_candidates=0,
                        build_cube_ns=int(t_build_done - t_dispatch_start),
                        layer1_norm_ns=0,
                        detector_forward_ns=0,
                        total_pipeline_ns=int(
                            t_build_done - t_dispatch_start
                        ),
                    )
                else:
                    result = pipeline.process(slot)
                    t_done = time.perf_counter_ns()
                    rec = StageTimingRecord(
                        cube_id=slot.cube_id,
                        n_candidates=len(result.candidates),
                        build_cube_ns=int(
                            result.stage_timings_ns["build_cube"]
                        ),
                        layer1_norm_ns=int(
                            result.stage_timings_ns["layer1_norm"]
                        ),
                        detector_forward_ns=int(
                            result.stage_timings_ns["detector_forward"]
                        ),
                        total_pipeline_ns=int(t_done - t_dispatch_start),
                    )
                records.append(rec)
                if (slot.cube_id + 1) % max(1, n_cubes // 10) == 0:
                    _LOG.info(
                        "cube=%d/%d total=%.2fms detector=%.2fms cands=%d",
                        slot.cube_id + 1, n_cubes,
                        rec.total_pipeline_ns / 1.0e6,
                        rec.detector_forward_ns / 1.0e6,
                        rec.n_candidates,
                    )
                await src.release(slot.cube_id)
    bench_wall_s = (time.perf_counter_ns() - bench_start_ns) / 1.0e9

    # M7.7.1 Phase A.2 (2026-06-04): drain the overlap-path GPU
    # substage event ring. The events were recorded on the main
    # stream during ``process_h2d_prefetched`` and have long since
    # completed; we read mean per-substage GPU ms here and add them
    # to the summary so the speed gate can show wall vs GPU.
    gpu_substage_ms: Dict[str, float] = {}
    try:
        gpu_substage_ms = pipeline.get_overlap_substage_event_timing_and_reset()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "overlap-path GPU substage drain failed: %s "
            "(continuing without GPU breakdown)",
            exc,
        )

    # Also drain the imager's per-cube substage event ring (combine /
    # fft / mask). Lets the speed gate break down the imager's GPU
    # cost — the biggest single contributor on the production
    # op-point at M7.7.
    imager_substage_ms: Dict[str, float] = {}
    try:
        gpu_imager = pipeline.gpu_imager
        if gpu_imager is not None and hasattr(gpu_imager, "pop_substage_timings"):
            imager_substage_ms = gpu_imager.pop_substage_timings()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "imager substage drain failed: %s "
            "(continuing without combine/fft/mask breakdown)",
            exc,
        )

    # ---- Write outputs ----
    ndjson_path = out_dir / "stage_timings.ndjson"
    with ndjson_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json()) + "\n")
    _LOG.info("wrote %s (%d records)", ndjson_path, len(records))

    summary = {
        "schema_version": 1,
        "bench": "search_node_throughput",
        "milestone": "M5",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_cubes": n_cubes,
            "cube_cadence_s": cube_cadence_s,
            "t_det": t_det,
            "n_fdm": n_fdm,
            "n_grid": n_grid,
            "threshold_sigma": threshold_sigma,
            "rng_seed": int(args.rng_seed),
            "device": device,
            "cube_dtype": str(cube_dtype).rsplit(".", 1)[-1],
            "bank_mask": args.bank_mask,
            "bank_mask_resolved": {
                "k_img": list(image_tokens),
                "k_dm": list(dm_tokens),
                "k_time": list(time_tokens),
                "n_kernels": n_kernels_total,
            },
            "skip_detector": skip_detector,
            "pipeline_overlap": bool(enable_overlap),
        },
        "wall_clock_s": bench_wall_s,
        "achieved_cubes_per_s": (
            len(records) / bench_wall_s if bench_wall_s > 0 else 0.0
        ),
        "n_cubes_processed": len(records),
        "n_candidates_total": int(sum(r.n_candidates for r in records)),
        "percentiles_ms": {
            "build_cube": percentiles([r.build_cube_ns for r in records]),
            "layer1_norm": percentiles([r.layer1_norm_ns for r in records]),
            "detector_forward": percentiles(
                [r.detector_forward_ns for r in records]
            ),
            "total_pipeline": percentiles(
                [r.total_pipeline_ns for r in records]
            ),
        },
        # M7.7.1 Phase A.2 (2026-06-04): per-substage GPU-time
        # breakdown from cuda Events recorded on the main stream
        # during ``process_h2d_prefetched``. Reported as cube-mean
        # (the events are too cheap to also percentile per-cube;
        # cube-to-cube variation is dominated by the wall-clock
        # stage timings already present above). Keys:
        #   imager_ms    — _run_imager_from_staged GPU time
        #   validity_ms  — _build_validity_mask GPU time
        #   layer1_ms    — _layer1_normalise GPU time
        #   detector_ms  — detector.forward GPU time
        #   total_gpu_ms — sum of the four (= GPU busy time / cube)
        #   n            — number of cubes contributing (= cubes in
        #                  the overlap path; 0 for sync / skip-detector
        #                  / CPU runs).
        # Compare ``total_gpu_ms`` with ``percentiles_ms.total_pipeline.p50``
        # to gauge how much of per-cube wall-clock is GPU SM time vs
        # CPU sync / cross-iteration overhead.
        "gpu_substage_ms": gpu_substage_ms,
        # Per-cube imager substage GPU ms breakdown (combine /
        # fft / mask). Same semantics as gpu_substage_ms above but
        # narrowed to the fused GpuImager's per-batch inner sections.
        # ``total_ms`` should approximately match
        # ``gpu_substage_ms.imager_ms`` (any gap = launch / wait
        # overhead between the per-batch loop iterations on the main
        # stream); the per-substage breakdown is what tells the
        # operator WHERE the imager's GPU time is going (D21 design
        # estimate: combine 45% / fft 44% / mask 11%).
        "imager_substage_ms": imager_substage_ms,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    _LOG.info("wrote %s", summary_path)
    pct = summary["percentiles_ms"]["total_pipeline"]
    _LOG.info(
        "throughput summary: %.2f cubes/s · total p50=%.2fms p99=%.2fms",
        summary["achieved_cubes_per_s"], pct["p50"], pct["p99"],
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 chunk-6b-β throughput + per-stage latency bench"
    )
    parser.add_argument("--n-cubes", type=int, default=DEFAULT_N_CUBES)
    parser.add_argument(
        "--cube-cadence-s", type=float, default=DEFAULT_CUBE_CADENCE_S,
        help="Wall-clock cube cadence (default 0 = drain as fast as the "
             "consumer can; use ~0.134 to mimic production cadence).",
    )
    parser.add_argument("--t-det", type=int, default=DEFAULT_T_DET)
    parser.add_argument("--n-fdm", type=int, default=DEFAULT_N_FDM)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument(
        "--threshold-sigma", type=float, default=DEFAULT_THRESHOLD_SIGMA,
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="torch device for the detector + cube ('cpu', 'cuda', "
             "'cuda:0', 'auto'). Combiner + imager always run on CPU "
             "in chunk-6b-α (the GPU sparse-scatter+cuFFT lands in the "
             "production hardening pass).",
    )
    parser.add_argument(
        "--cube-dtype", type=str, default="fp32", choices=("fp32", "fp16"),
        help="Cube + detector dtype. fp16 only valid with --device cuda* "
             "(plan §3.6.11 production pin).",
    )
    parser.add_argument(
        "--bank-mask", type=str, default=None,
        help="Detector kernel-bank subset, e.g. "
             "'k_img=unit;k_dm=d1;k_time=*' to keep only the unit image "
             "kernel × d1 DM kernel × all 8 time kernels. Default: full "
             "128 triple bank. Used by Chunk 6c-β perf-vs-quality sweeps.",
    )
    parser.add_argument(
        "--skip-detector", action="store_true",
        help="Skip Layer-1 normalisation + Detector.forward(). Only the "
             "combiner + 2D iFFT + edge mask (build_cube) stage runs per "
             "cube. Used to measure the imager-side upper-bound throughput "
             "in isolation; with the detector skipped, GPU memory pressure "
             "disappears and the bench can run at production geometry "
             "(T_det=512, N_fdm=32, N_grid=256) on a 2080 Ti. "
             "Chunk 6c follow-up.",
    )
    parser.add_argument(
        "--image-backend", type=str, default="cpu", choices=("cpu", "gpu"),
        help="CubePipeline image backend. 'cpu' (default) uses the "
             "chunk-6a numpy reference (combine_chgroups + "
             "dirty_image_from_uv_grid). 'gpu' uses the chunk-8 production "
             "GpuImager (host-side cf->cint8 quantise + fused dequant+combine "
             "CUDA kernel + cuFFT-cfp16 ifft2 + edge mask). 'gpu' requires "
             "--device cuda* and --cube-dtype fp16 (production pin).",
    )
    parser.add_argument(
        "--detector-streaming", action="store_true",
        help="Use the chunk-8 streaming kernel-by-kernel detector "
             "forward (DeterministicDetector(streaming=True)). The "
             "batched forward materialises [K, T, F, H, W] up front "
             "(16 GiB for K=8 fp32 at production T_det=256, N_fdm=32, "
             "N_grid=256 — OOMs on an 11 GiB 2080 Ti). The streaming "
             "forward is ~2× the batched detector compute but is the "
             "only path that fits at production geometry.",
    )
    parser.add_argument(
        "--detector-streaming-tile-size", type=int, default=64,
        help="W-axis tile size for the streaming detector's lowmem "
             "boxcar (forwarded to boxcar_via_cumsum's tile_size arg). "
             "Default 64 caps the fp32 cumsum working set at ~768 MiB "
             "at production geometry.",
    )
    parser.add_argument(
        "--full-prefetch",
        action="store_true",
        help="M7.2 cube-cadence experiment: run the full imager (H2D + "
             "combine + IFFT + edge mask) for cube N+1 on a dedicated "
             "GPU stream concurrent with cube N's Layer-1 + detector. "
             "Takes precedence over --pipeline-overlap. Mirrors the "
             "old chunk-8 'wide overlap' path; intended for use with "
             "the cube-cadence carry-over where the imager is ~50% "
             "of layer1+det.",
    )
    parser.add_argument(
        "--pipeline-overlap",
        action="store_true",
        help="Enable one-cube lookahead overlap: prebuild cube N+1 on a "
             "prefetch stream while running Layer-1 + detector on cube N "
             "(GPU backend only).",
    )
    parser.add_argument(
        "--detector-streaming-decoder-n-top", type=int, default=64,
        help="Per-kernel top-k budget used by streaming decoder "
        "(decode_topk_lowmem). Lower values reduce topk work; keep "
        "well above expected per-kernel candidate counts.",
    )
    parser.add_argument(
        "--detector-boxcar-accum-dtype", type=str, default="default",
        choices=("default", "fp32", "fp16", "bf16"),
        help="Cumsum accumulation dtype for the streaming detector's "
             "amortise-cumsum fast-path. 'default' (None to the detector) "
             "preserves chunk-8 behaviour (fp32 accum when cube is fp16). "
             "'fp16' opts into the commissioning fast-path that halves "
             "boxcar memory traffic (saves ~190 ms / cube at production "
             "geometry on a 2080 Ti). The empirically measured σ-clip "
             "error vs fp32 is ≤ 0.02% across all 7 K_time widths at "
             "T_det=256 / N_fdm=34 / N_grid=256 — 75x below the Layer-2 "
             "EMA's intrinsic 1.5% cube-to-cube noise floor (plan §1622).",
    )
    parser.add_argument(
        "--detector-layer2-max-samples", type=int, default=1_000_000,
        help="Per-kernel sample cap for Layer-2 interior sigma clipping. "
             "Set <=0 to disable subsampling (full interior). Default "
             "1,000,000 (current production-safe baseline).",
    )
    parser.add_argument(
        "--prequantise", action="store_true",
        help="Pre-quantise one cube of cf32 streams to cint8 once in "
             "the synthetic RxRing source and re-yield the cached cint8 "
             "stack on every iteration. Emulates the chunk-8b RX-ring "
             "production contract (M3 emits cint8 already; search node "
             "never re-quantises) and isolates the GPU pipeline cost "
             "from host-side bench scaffolding (synthetic source "
             "generation + cf -> cint8 quantise dominate the bench at "
             "T_det=256/N_grid=256 — together ~5 s/cube on h01).",
    )
    parser.add_argument(
        "--layer1-max-samples", type=int, default=10_000,
        help="Per-fdm cell-count cap for the Layer-1 σ-clipped std "
             "(forwarded to ``Layer1State.max_samples``). At production "
             "geometry each fdm slab has 256³ = 16.8 M cells; capping "
             "at 1 M brings ``torch.median`` per iter from ~25 ms to "
             "~1 ms with σ̂ standard error ≈ 7e-4 σ. Set ≤ 0 or omit to "
             "disable (chunk-1 behaviour: full slab). Default 1_000_000.",
    )
    parser.add_argument(
        "--symmetric-shift-padding", action="store_true",
        help="M7.7 (2026-06-04): emit slots with stream_origin_offset_"
             "samples = max(0, shifts.max()) and a buffer pre-padded by "
             "pad_left + pad_right rows. Exercises the CubePipeline "
             "fused-L1 fast path that the production search nodes run "
             "(coverage correction off, fused-L1 imager re-enabled). "
             "Without this flag the bench silently bypasses M7.7 so any "
             "perf regression in the post-M7.7 path goes unnoticed.",
    )
    parser.add_argument(
        "--dm-plan-path", type=str, default=None,
        help="Path to the production DmPlan NPZ "
             "(e.g. /home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz). "
             "When set the bench loads the SAME coarse/fine DM grids as "
             "production search_compute, so the shifts table covers the "
             "Option A ±83 sample range and the synthetic stream geometry "
             "matches live. Without this the bench uses a tight synthetic "
             "linear DM grid (shifts ~30 samples, all-positive).",
    )
    parser.add_argument(
        "--coarse-dm-owner-idx", type=int, default=-1,
        help="When --dm-plan-path is set, restrict the bench to the K "
             "fine-DM trials owned by this coarse-DM index — exactly the "
             "per-half slice that production search_compute_{0,1} run "
             "with (matches --coarse-dm-owners-half-{0,1}). -1 (default) "
             "= use the full plan's first N_fdm rows (legacy bench path).",
    )
    parser.add_argument(
        "--t-int-search-us", type=float, default=1048.576,
        help="Search-sample cadence in microseconds. Default 1048.576 "
             "matches production (--t-int-search-us 1048.576). Used by "
             "the synthetic source to compute the time_shift_table.",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "throughput" / "M5"),
    )
    parser.add_argument("--quick-smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())
