#!/usr/bin/env python3
"""bench/cube_injection_detector.py — M5 Chunk 5 primary detector
correctness gate (plan §8 line 2329).

Cube-level (post-imager) injection sweep. Synthesises a thermal-noise
``[T_det, N_fdm, N_grid, N_grid] float32`` cube, adds a parametric
boxcar injection at known
``(l_pix, m_pix, fine_dm_idx, t_in_cube, snr, width_samples)``, and
feeds the cube straight into ``Detector.forward()`` → decoder →
canonical-zone gate → cross-kernel merger. M6 chunk 0 retired the
trigger emitter; the bench now operates purely on the pipeline /
detector ``Candidate`` output (no TCP fan-out, no MockTriggerListener).

This bench does NOT depend on M3 / M4a / corr-side artefacts. It runs
h01 alone (or any laptop with the dsa110-rt env) and is the user-facing
M5 detector correctness gate per plan §8 line 2329.

CLI surface (see ``--help`` for the full grid):

  python -m bench.cube_injection_detector \\
      [--quick-sweep | --full-sweep]                 \\
      [--snrs 6 8 10 12 15]                          \\
      [--widths 2 4 8 16 32 64 128]                  \\
      [--n-trials-per-cell 3]                        \\
      [--noise-only-cubes 30]                        \\
      [--out bench/reports/<UTC>/cube_injection/M5/]

Outputs (under ``--out``):

  * ``injection_log.ndjson``   — one JSON line per emitted injection
    ``Candidate`` (``dataclasses.asdict(c)``); consumed by
    ``tools/viz/search_detector_check.py --mode cube_injection``.
  * ``noise_only_log.ndjson`` — one JSON line per noise-only
    ``Candidate`` for the FAR sub-check.
  * ``summary.json``           — aggregate metrics + bench config snapshot.
  * ``bench.log``              — human-readable progress / per-cell log.

Per plan §8 line 2329 the operator runs the viz tool on these logs to
produce the recovery heatmap + report.html. NO PASS/FAIL banner.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Imports from the M5 stack (after sys.path insert).
# ---------------------------------------------------------------------------

os.environ.setdefault("DSART_TEST", "1")  # enable contract __post_init__ checks

import torch  # noqa: E402

from dsart.common.constants import (  # noqa: E402
    DETECTOR_DM_KERNELS,
    DETECTOR_IMAGE_KERNELS,
    DETECTOR_TIME_KERNELS,
)
from dsart.common.contracts import Candidate  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.inject.cube_injection import (  # noqa: E402
    CubeInjectionConfig,
    iter_snr_width_grid,
    synthesise_cube,
)


_LOG = logging.getLogger("bench.cube_injection_detector")


# ---------------------------------------------------------------------------
# Bench config
# ---------------------------------------------------------------------------


# Default sweep grid (plan §8 line 2329).
DEFAULT_SNRS: Tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 15.0)
DEFAULT_WIDTHS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)

# --quick-sweep: minimal grid for fast smoke (M5.sh DoD path).
QUICK_SWEEP_SNRS: Tuple[float, ...] = (8.0, 12.0)
QUICK_SWEEP_WIDTHS: Tuple[int, ...] = (4, 32)

# Cube geometry — small enough for fast iteration; the plan-pinned
# T_det = 512, N_fdm = 32 (per-GPU), N_grid = 256 are exercised by the
# Chunk-6 search_node_throughput bench. The cube_injection bench keeps
# things small to spend wall-clock on the parametric sweep, not on
# rasterising one mega-cube.
DEFAULT_T_DET: int = 512
DEFAULT_N_FDM: int = 8
DEFAULT_N_GRID: int = 64

# Detector threshold for the bench. M6 chunk 0 retired the trigger
# emitter so per-cube caps + rate-limits + holdoff knobs no longer
# apply here — the bench operates directly on the detector candidate
# list, which is already SNR-cut at this threshold.
BENCH_DETECTOR_THRESHOLD_SIGMA: float = 5.0  # capture θ ∈ {6, 7, 8, 9, 10}

# Recovery tolerance per plan §8 line 2329 (cube-level injection bypasses
# imager loss → tighter than Layer-3's 0.7×):
RECOVERY_SNR_FRACTION: float = 0.85

# Recovery position tolerance: per plan §1582-1588 the local-max NMS
# radii are min(2, k_psf) for (l, m) and k_dm/2+1 for fdm, k_time/2+1
# for time. We pin a generous tolerance window so a near-miss recovery
# (e.g. one pixel offset due to even-width centring bias) still counts.
RECOVERY_LM_TOL: int = 2
RECOVERY_FDM_TOL: int = 2
RECOVERY_T_TOL: int = 64  # absorbs the K_time = 128 boxcar response width


# ---------------------------------------------------------------------------
# Bank-mask parser (Chunk 6c-α) — re-exported from bench._bank_mask so
# bench/search_node_throughput.py shares the same parser. See module
# docstring there for syntax + examples.
# ---------------------------------------------------------------------------

from bench._bank_mask import parse_bank_mask  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Bench data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellResult:
    """One ``(snr, width)`` sweep cell result."""

    injected: dict
    n_trials: int
    n_recovered: int
    recovered_snrs: List[float]
    matched_kernel_id: Optional[str]
    cube_t_det: int
    cube_n_fdm: int
    cube_n_grid: int

    @property
    def recovery_fraction(self) -> float:
        return self.n_recovered / self.n_trials if self.n_trials > 0 else 0.0

    @property
    def snr_ratio_mean(self) -> float:
        if not self.recovered_snrs:
            return float("nan")
        injected_snr = float(self.injected["snr"])
        return sum(self.recovered_snrs) / (len(self.recovered_snrs) * injected_snr)


@dataclass(slots=True)
class NoiseOnlySummary:
    n_cubes: int
    n_kernels: int
    candidate_snrs: List[float] = field(default_factory=list)
    n_candidates: int = 0


# ---------------------------------------------------------------------------
# Bench core
# ---------------------------------------------------------------------------


def _build_detector(
    *,
    t_det: int,
    n_fdm: int,
    threshold_sigma: float,
    seed: int,
    image_tokens: Sequence[str] = DETECTOR_IMAGE_KERNELS,
    dm_tokens: Sequence[str] = DETECTOR_DM_KERNELS,
    time_tokens: Sequence[str] = DETECTOR_TIME_KERNELS,
) -> DeterministicDetector:
    """Construct a fresh DeterministicDetector for the bench.

    The bench instantiates a new detector per sweep cell so the
    Layer-2 σ_k EMA warm-up (and the ``NOISE_WARMUP`` flag) does not
    leak across unrelated cells. Layer-2 burn-in is short-circuited by
    seeding ``s_k`` to the analytic √(K_dm × K_time) values.

    The kernel bank is constructed from the (image, dm, time) token
    subsets so the bench can sweep bank-mask configurations per
    Chunk 6c-α. Default subsets reproduce the full 128-triple bank.
    """
    torch.manual_seed(seed)
    bank = build_kernel_bank(
        image_tokens=tuple(image_tokens),
        dm_tokens=tuple(dm_tokens),
        time_tokens=tuple(time_tokens),
        dtype=torch.float32,
    )
    return DeterministicDetector(
        kernel_bank=bank,
        threshold_sigma=threshold_sigma,
        detector_version="v1.M5",
        dtype=torch.float32,
    )


def _detect_one_cube(
    detector: DeterministicDetector,
    cube: torch.Tensor,
    validity_mask: torch.Tensor,
    sigma_layer1: torch.Tensor,
) -> List[Candidate]:
    """Run the detector on one cube. Wraps ``detector.forward`` so the
    rest of the bench is dtype-agnostic. Returns the merged candidate
    list (canonical-zone gate is applied below at emit time)."""
    with torch.no_grad():
        cands = detector.forward(cube, validity_mask, sigma_layer1)
    return cands


def _kernel_match_radius(
    kernel_id: str,
    *,
    fdm_tol: int = RECOVERY_FDM_TOL,
    t_tol: int = RECOVERY_T_TOL,
) -> Tuple[int, int]:
    """Per-kernel position tolerances: K_dm and K_time NMS radii grow
    with the kernel's box width, so generous-but-bounded windows
    rather than fixed radii. Returns (fdm_radius, t_radius)."""
    parts = kernel_id.split(":")
    if len(parts) != 3:
        return fdm_tol, t_tol
    try:
        k_dm = int(parts[1][1:])
        k_t = int(parts[2][1:])
    except (IndexError, ValueError):
        return fdm_tol, t_tol
    return max(fdm_tol, k_dm // 2 + 1), max(t_tol, k_t // 2 + 1)


def _is_match(
    cand: Candidate,
    inj: CubeInjectionConfig,
    *,
    lm_tol: int = RECOVERY_LM_TOL,
) -> bool:
    fdm_tol, t_tol = _kernel_match_radius(cand.kernel_id)
    if abs(int(cand.l) - inj.l_pix) > lm_tol:
        return False
    if abs(int(cand.m) - inj.m_pix) > lm_tol:
        return False
    if abs(int(cand.dm_idx) - inj.fine_dm_idx) > fdm_tol:
        return False
    if abs(int(cand.event_specnum) - inj.t_in_cube) > t_tol:
        return False
    return True


# ---------------------------------------------------------------------------
# Async bench bodies
# ---------------------------------------------------------------------------


def _run_one_injection_trial(
    inj: CubeInjectionConfig,
    *,
    detector: DeterministicDetector,
    rng: np.random.Generator,
    t_det: int,
    n_fdm: int,
    n_grid: int,
) -> List[Candidate]:
    """Synthesise one (noise + injection) cube, run detector; return
    the per-cube candidate list."""
    cube, validity_mask, sigma_layer1 = synthesise_cube(
        t_det=t_det, n_fdm=n_fdm, n_grid=n_grid,
        injections=(inj,), rng=rng,
    )
    return _detect_one_cube(detector, cube, validity_mask, sigma_layer1)


def _run_one_noise_only_cube(
    *,
    detector: DeterministicDetector,
    rng: np.random.Generator,
    t_det: int,
    n_fdm: int,
    n_grid: int,
) -> List[Candidate]:
    cube, validity_mask, sigma_layer1 = synthesise_cube(
        t_det=t_det, n_fdm=n_fdm, n_grid=n_grid,
        injections=(), rng=rng,
    )
    return _detect_one_cube(detector, cube, validity_mask, sigma_layer1)


async def _bench_main(args: argparse.Namespace) -> int:
    """Async bench entry point. Wired from ``main()`` via ``asyncio.run``."""
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

    snrs: Sequence[float] = (
        QUICK_SWEEP_SNRS if args.quick_sweep else tuple(args.snrs)
    )
    widths: Sequence[int] = (
        QUICK_SWEEP_WIDTHS if args.quick_sweep else tuple(args.widths)
    )
    n_trials = int(args.n_trials_per_cell)
    n_noise_cubes = int(args.noise_only_cubes)

    t_det = int(args.t_det)
    n_fdm = int(args.n_fdm)
    n_grid = int(args.n_grid)

    image_tokens, dm_tokens, time_tokens = parse_bank_mask(args.bank_mask)
    n_kernels_total = (
        len(image_tokens) * len(dm_tokens) * len(time_tokens)
    )

    if args.quick_sweep:
        n_trials = max(min(n_trials, 1), 1)
        n_noise_cubes = max(min(n_noise_cubes, 2), 1)
        # Smaller cubes than the operator-facing default so the CPU-side
        # 128-kernel inner loop stays fast (the smoke / DoD path doesn't
        # need the full plan-pinned T_det = 512). The full-sweep path
        # honours the CLI defaults.
        # K_time = 128 / K_dm = 5, 7 boxcars are skipped when the cube
        # axis is shorter than the kernel; the detector handles this
        # gracefully (forward.py L627, L635 guards).
        if t_det == DEFAULT_T_DET:
            t_det = 64
        if n_fdm == DEFAULT_N_FDM:
            n_fdm = 4
        if n_grid == DEFAULT_N_GRID:
            n_grid = 16

    _LOG.info(
        "bench config: snrs=%s widths=%s n_trials=%d n_noise_cubes=%d "
        "T_det=%d N_fdm=%d N_grid=%d threshold=%.2fσ",
        list(snrs), list(widths), n_trials, n_noise_cubes,
        t_det, n_fdm, n_grid, BENCH_DETECTOR_THRESHOLD_SIGMA,
    )
    _LOG.info(
        "bank-mask: k_img=%s k_dm=%s k_time=%s (total %d kernel triples)",
        list(image_tokens), list(dm_tokens), list(time_tokens),
        n_kernels_total,
    )

    # Common injection geometry: phase centre, mid-DM, mid-time. Plan
    # §1592 time-edge gate masks first/last 64 samples (n_kernel_max_t=128
    # // 2 = 64) so t_in_cube = T_det // 2 is well inside.
    l_pix = n_grid // 2
    m_pix = n_grid // 2
    fine_dm_idx = n_fdm // 2
    t_in_cube = t_det // 2

    cell_results: List[CellResult] = []
    score_per_kernel_per_cell: dict = {}  # (snr, width) → kernel_id → snr value
    noise_summary = NoiseOnlySummary(
        n_cubes=0, n_kernels=n_kernels_total,
    )
    injection_candidates: List[Candidate] = []
    noise_only_candidates: List[Candidate] = []

    # ----- Sweep over (snr, width) cells -----
    for inj in iter_snr_width_grid(
        snrs=tuple(float(s) for s in snrs),
        widths=tuple(int(w) for w in widths),
        l_pix=l_pix,
        m_pix=m_pix,
        fine_dm_idx=fine_dm_idx,
        t_in_cube=t_in_cube,
    ):
        cell_seed = (
            args.seed
            + 1000 * int(inj.snr)
            + int(inj.width_samples)
        )
        detector = _build_detector(
            t_det=t_det,
            n_fdm=n_fdm,
            threshold_sigma=BENCH_DETECTOR_THRESHOLD_SIGMA,
            seed=cell_seed,
            image_tokens=image_tokens,
            dm_tokens=dm_tokens,
            time_tokens=time_tokens,
        )
        recovered_snrs: List[float] = []
        matched_kernel_id: Optional[str] = None
        kernel_score_at_match: dict = {}

        for trial in range(n_trials):
            trial_rng = np.random.default_rng(cell_seed + 17 * trial)
            cands = _run_one_injection_trial(
                inj,
                detector=detector,
                rng=trial_rng,
                t_det=t_det,
                n_fdm=n_fdm,
                n_grid=n_grid,
            )
            injection_candidates.extend(cands)
            # Find best matching candidate (highest SNR among matches).
            matches = [c for c in cands if _is_match(c, inj)]
            if matches:
                best = max(matches, key=lambda c: c.snr)
                recovered_snrs.append(float(best.snr))
                if matched_kernel_id is None:
                    matched_kernel_id = best.kernel_id
            # Snapshot per-kernel score map for the FIRST trial only
            # (the operator only needs one heatmap per cell).
            if trial == 0:
                for c in cands:
                    if not _is_match(c, inj):
                        continue
                    prev = kernel_score_at_match.get(c.kernel_id, 0.0)
                    if c.snr > prev:
                        kernel_score_at_match[c.kernel_id] = float(c.snr)

        cell = CellResult(
            injected={
                "snr": float(inj.snr),
                "width_samples": int(inj.width_samples),
                "l_pix": inj.l_pix,
                "m_pix": inj.m_pix,
                "fine_dm_idx": inj.fine_dm_idx,
                "t_in_cube": inj.t_in_cube,
                "profile": inj.profile,
            },
            n_trials=n_trials,
            n_recovered=len(recovered_snrs),
            recovered_snrs=list(recovered_snrs),
            matched_kernel_id=matched_kernel_id,
            cube_t_det=t_det,
            cube_n_fdm=n_fdm,
            cube_n_grid=n_grid,
        )
        cell_results.append(cell)
        score_per_kernel_per_cell[
            (float(inj.snr), int(inj.width_samples))
        ] = dict(kernel_score_at_match)

        _LOG.info(
            "cell snr=%.1f width=%d : recovered %d/%d (mean ratio=%.3f) "
            "matched=%s",
            inj.snr, inj.width_samples,
            cell.n_recovered, cell.n_trials,
            cell.snr_ratio_mean, cell.matched_kernel_id,
        )

    # ----- Noise-only FAR sub-check -----
    if n_noise_cubes > 0:
        far_detector = _build_detector(
            t_det=t_det,
            n_fdm=n_fdm,
            threshold_sigma=BENCH_DETECTOR_THRESHOLD_SIGMA,
            seed=args.seed + 9999,
            image_tokens=image_tokens,
            dm_tokens=dm_tokens,
            time_tokens=time_tokens,
        )
        for cube_idx in range(n_noise_cubes):
            noise_rng = np.random.default_rng(args.seed + 50_000 + cube_idx)
            cands = _run_one_noise_only_cube(
                detector=far_detector,
                rng=noise_rng,
                t_det=t_det,
                n_fdm=n_fdm,
                n_grid=n_grid,
            )
            cube_snrs = [float(c.snr) for c in cands]
            noise_summary.candidate_snrs.extend(cube_snrs)
            noise_summary.n_candidates += len(cands)
            noise_only_candidates.extend(cands)
            _LOG.info(
                "noise-only cube %d/%d : %d candidates",
                cube_idx + 1, n_noise_cubes, len(cands),
            )
        noise_summary.n_cubes = n_noise_cubes

    # ----- Aggregate FAR by θ -----
    thetas = [6.0, 7.0, 8.0, 9.0, 10.0]
    n_kernels = noise_summary.n_kernels
    far_samples: List[dict] = []
    if noise_summary.n_cubes > 0 and n_kernels > 0:
        for theta in thetas:
            n_above = sum(1 for s in noise_summary.candidate_snrs if s >= theta)
            empirical = n_above / float(noise_summary.n_cubes * n_kernels)
            far_samples.append({
                "theta": theta,
                "empirical_per_cube_per_kernel": empirical,
                "n_cubes": noise_summary.n_cubes,
                "n_kernels": n_kernels,
            })

    # ----- Persist logs + summary -----
    injection_path = out_dir / "injection_log.ndjson"
    with injection_path.open("w") as fh:
        for c in injection_candidates:
            fh.write(json.dumps(dataclasses.asdict(c)) + "\n")
    noise_only_path = out_dir / "noise_only_log.ndjson"
    with noise_only_path.open("w") as fh:
        for c in noise_only_candidates:
            fh.write(json.dumps(dataclasses.asdict(c)) + "\n")

    summary = {
        "tool": "bench/cube_injection_detector.py",
        "version": "v1.M5",
        "generated_utc_ns": time.time_ns(),
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "snrs": list(snrs),
            "widths": list(widths),
            "n_trials_per_cell": n_trials,
            "noise_only_cubes": n_noise_cubes,
            "T_det": t_det,
            "N_fdm": n_fdm,
            "N_grid": n_grid,
            "detector_threshold_sigma": BENCH_DETECTOR_THRESHOLD_SIGMA,
            "recovery_snr_fraction": RECOVERY_SNR_FRACTION,
            "recovery_lm_tol": RECOVERY_LM_TOL,
            "recovery_fdm_tol": RECOVERY_FDM_TOL,
            "recovery_t_tol": RECOVERY_T_TOL,
            "quick_sweep": bool(args.quick_sweep),
            "seed": int(args.seed),
            "bank_mask": args.bank_mask,
            "bank_mask_resolved": {
                "k_img": list(image_tokens),
                "k_dm": list(dm_tokens),
                "k_time": list(time_tokens),
                "n_kernels": n_kernels_total,
            },
        },
        "cells": [
            {
                "injected": c.injected,
                "n_trials": c.n_trials,
                "n_recovered": c.n_recovered,
                "recovery_fraction": c.recovery_fraction,
                "snr_ratio_mean": (
                    c.snr_ratio_mean
                    if math.isfinite(c.snr_ratio_mean)
                    else None
                ),
                "matched_kernel_id": c.matched_kernel_id,
            }
            for c in cell_results
        ],
        "far": far_samples,
        "noise_only_total_candidates": noise_summary.n_candidates,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _LOG.info("wrote %d injection cells + %d noise-only cubes to %s",
              len(cell_results), noise_summary.n_cubes, out_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _default_out_dir() -> Path:
    """Default output dir: bench/reports/<UTC-date>/cube_injection/M5/."""
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return REPO_ROOT / "bench" / "reports" / today / "cube_injection" / "M5"


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument(
        "--quick-sweep", action="store_true",
        help="Minimal smoke sweep (2 SNRs × 2 widths × 1 trial × 4 noise cubes); "
             "used by tools/dod/M5.sh.",
    )
    grp.add_argument(
        "--full-sweep", action="store_true",
        help="Default plan-pinned sweep (5 SNRs × 7 widths × 3 trials × "
             "30 noise cubes); ~30 s wall-clock on h01 GPU 1.",
    )
    ap.add_argument(
        "--snrs", nargs="+", type=float, default=list(DEFAULT_SNRS),
        help=f"Injection SNR sweep (default: {list(DEFAULT_SNRS)}).",
    )
    ap.add_argument(
        "--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS),
        help=f"Injection width-samples sweep (default: {list(DEFAULT_WIDTHS)}).",
    )
    ap.add_argument(
        "--n-trials-per-cell", type=int, default=3,
        help="Number of independent (noise + injection) trials per cell.",
    )
    ap.add_argument(
        "--noise-only-cubes", type=int, default=30,
        help="Number of noise-only cubes for the FAR sub-check.",
    )
    ap.add_argument(
        "--t-det", type=int, default=DEFAULT_T_DET,
        help=f"Cube time depth (default: {DEFAULT_T_DET}).",
    )
    ap.add_argument(
        "--n-fdm", type=int, default=DEFAULT_N_FDM,
        help=f"Cube fine-DM depth (default: {DEFAULT_N_FDM}).",
    )
    ap.add_argument(
        "--n-grid", type=int, default=DEFAULT_N_GRID,
        help=f"Cube spatial side length (default: {DEFAULT_N_GRID}).",
    )
    ap.add_argument(
        "--seed", type=int, default=20260506,
        help="RNG seed (Layer-2 EMA ⊕ injection ⊕ noise; deterministic).",
    )
    ap.add_argument(
        "--bank-mask", type=str, default=None,
        help="Detector kernel-bank subset, e.g. "
             "'k_img=unit;k_dm=d1;k_time=*' to keep only the unit image "
             "kernel × d1 DM kernel × all 8 time kernels. Each axis "
             "defaults to '*' (full subset). Default: None = full 128 "
             "triple bank. Used by Chunk 6c-α perf-vs-quality sweeps.",
    )
    ap.add_argument(
        "--out", type=str, default=str(_default_out_dir()),
        help="Output directory (default: bench/reports/<UTC>/cube_injection/M5/).",
    )
    args = ap.parse_args(argv)

    # `--full-sweep` is the default semantics — only the explicit
    # --quick-sweep flag changes behaviour. The flag exists primarily
    # so M5.sh can intend "run the slow path".
    if args.full_sweep and args.quick_sweep:  # pragma: no cover
        ap.error("--quick-sweep and --full-sweep are mutually exclusive")

    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
