#!/usr/bin/env python3
"""bench/preflight/search_carryover_equivalence.py — M7.7.2 numerical
equivalence gate for cube-level carry-over re-imaging.

What it does
============

For a small synthetic run (N cubes of complex-Gaussian noise with
temporally-overlapping streams), drives TWO CubePipeline instances
side-by-side at the same op-point:

  * Run A: ``cube_pipeline_carry_over_re_imaging = False`` — full
    re-image for every cube (the historical M7.7 baseline).
  * Run B: ``cube_pipeline_carry_over_re_imaging = True`` — the
    M7.7.2 carry-over path; cube N+1's first ``t_det - cadence``
    image-space rows are copied from cube N's last ``t_det - cadence``
    rows with per-fdm σ rescale, and only the new ``cadence`` rows
    are re-imaged.

Both pipelines are fed the same ``SyntheticRxRingSource(overlap_streams=
True, cube_cadence_samples=128)`` so consecutive cubes' streams share
the trailing ``t_stream - cadence`` rows of absolute time. Under that
condition, ``cube_{N+1}[0:t_det-cadence]`` computed by the full
re-image MUST equal ``cube_N[cadence:t_det]`` to fp16 precision
(modulo the σ rescale that both runs apply). Any deviation outside
the fp16 tolerance (atol=2e-3, rtol=2e-3 on cube_dtype=fp16 cells of
magnitude ~10) indicates a correctness regression in the carry-over
path.

What this gate enforces
=======================

For every cube ``N >= 1``:

  * ``cube_A[N, :t_lo, :, :, :] ≈ cube_B[N, :t_lo, :, :, :]``
    (carry-over rows match the full re-image).
  * ``cube_A[N, t_lo:, :, :, :] ≈ cube_B[N, t_lo:, :, :, :]``
    (newly-imaged rows are computed identically — these are NOT
    the carry-over region; they should be bit-equal modulo the
    different combine launch grid for the partial-row kernel).
  * No NaN / inf in either run.

Exit code conventions
=====================

* 0 — every cube agrees within tolerance. Safe to enable carry-over
      in production.
* 1 — some cube exceeds tolerance. Carry-over has a correctness
      regression; DO NOT enable in production. Script prints the
      worst offender + the per-region max-abs-diff so the operator
      knows whether the bug is in the carry-over copy / σ rescale
      (``[:t_lo]`` region) or in the partial-grid combine kernel
      (``[t_lo:]`` region).
* 2 — bench run itself failed.

Usage
=====

::

    python -m bench.preflight.search_carryover_equivalence \\
        [--n-cubes 12]                 \\
        [--n-grid 256] [--n-fdm 34] [--t-det 192] \\
        [--cube-cadence-samples 128]   \\
        [--device cuda:0]              \\
        [--atol 2e-3] [--rtol 2e-3]    \\
        [--dm-plan-path /home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz] \\
        [--coarse-dm-owner-idx 0]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import SyntheticRxRingSource  # noqa: E402
from dsart.services.search_compute import _dm_grids_from_npz  # noqa: E402

_LOG = logging.getLogger("carryover_equivalence")


# ---------------------------------------------------------------------------
# DM-plan loading (same as search_node_throughput / search_speed_gate)
# ---------------------------------------------------------------------------


def _select_owner_half(
    coarse_dm: np.ndarray,
    fine_dm: np.ndarray,
    fine_to_coarse: np.ndarray,
    *,
    owner_idx: int,
):
    """Restrict the plan to the fine-DM rows that belong to one
    coarse-DM owner. Mirrors the production owner-half slicing in
    bench.search_node_throughput.
    """
    if owner_idx < 0:
        return coarse_dm, fine_dm, fine_to_coarse
    mask = fine_to_coarse == int(owner_idx)
    if not bool(mask.any()):
        raise ValueError(
            f"--coarse-dm-owner-idx {owner_idx} matches no fine-DM "
            f"rows in the plan (n_coarse={len(coarse_dm)})"
        )
    fine_dm_sub = fine_dm[mask]
    # Owner has a single coarse-DM cell from the perspective of one
    # search-compute half.
    coarse_dm_sub = np.asarray([float(coarse_dm[int(owner_idx)])])
    fine_to_coarse_sub = np.zeros_like(fine_to_coarse[mask], dtype=np.int64)
    return coarse_dm_sub, fine_dm_sub, fine_to_coarse_sub


# ---------------------------------------------------------------------------
# Side-by-side pipeline driver
# ---------------------------------------------------------------------------


def _build_pipeline(
    *,
    n_grid: int,
    n_fdm: int,
    t_det: int,
    device: str,
    carry_over: bool,
    cube_cadence_samples: int,
    cube_dtype: torch.dtype = torch.float16,
    gpu_complex_dtype: torch.dtype = torch.complex32,
) -> Tuple[CubePipeline, "DeterministicDetector"]:
    """Construct a CubePipeline + minimal detector at the production
    op-point with the requested carry-over setting."""
    cfg = CubePipelineConfig(
        n_grid=n_grid,
        edge_mask_kernel_support=5,
        device=device,
        cube_dtype=cube_dtype,
        gpu_complex_dtype=gpu_complex_dtype,
        image_backend="gpu",
        gpu_t_det=t_det,
        gpu_n_fdm=n_fdm,
        cube_pipeline_carry_over_re_imaging=carry_over,
        cube_cadence_samples=cube_cadence_samples,
    )
    bank = build_kernel_bank(
        image_tokens=("unit",),
        dm_tokens=("d1",),
        time_tokens=("b1",),
        dtype=cube_dtype,
    )
    detector = DeterministicDetector(
        kernel_bank=bank,
        threshold_sigma=12.0,
        detector_version="v1.M5.eq",
        search_node_id=1,
        gpu_half=1,
        dtype=cube_dtype,
        device=torch.device(device),
        streaming=True,
        streaming_tile_size=256,
        streaming_decoder_n_top=24,
        boxcar_accum_dtype=cube_dtype,
        layer2_sigma_max_samples=100000,
    )
    pipe = CubePipeline(
        config=cfg,
        detector=detector,
        layer1_state=Layer1State(
            n_fdm=n_fdm,
            n_burnin_cubes=5,
            max_samples=10000,
        ),
    )
    return pipe, detector


async def _drive_runs(
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run pipeline A (no carry-over) and pipeline B (with carry-over)
    side by side on the SAME synthetic source feed (overlap_streams=True
    so the two runs' inputs are identical). Returns the per-cube
    post-imager cube (before L1 normalisation) for each run.
    """
    n_grid = int(args.n_grid)
    n_fdm_default = int(args.n_fdm)
    t_det = int(args.t_det)
    cadence = int(args.cube_cadence_samples)
    device = str(args.device)

    dm_plan_path = Path(args.dm_plan_path)
    coarse_dm, fine_dm, fine_to_coarse = _dm_grids_from_npz(
        dm_plan_path, n_coarse=8,
    )
    coarse_dm, fine_dm, fine_to_coarse = _select_owner_half(
        coarse_dm, fine_dm, fine_to_coarse,
        owner_idx=int(args.coarse_dm_owner_idx),
    )
    n_fdm = int(fine_dm.shape[0])
    if n_fdm != n_fdm_default:
        _LOG.warning(
            "DM plan supplied n_fdm=%d (--coarse-dm-owner-idx=%d); "
            "overriding --n-fdm=%d.",
            n_fdm, int(args.coarse_dm_owner_idx), n_fdm_default,
        )

    cube_dtype = (
        torch.float32 if str(args.cube_dtype) == "fp32" else torch.float16
    )
    gpu_complex_dtype = (
        torch.complex64 if cube_dtype == torch.float32 else torch.complex32
    )
    rng_seed = int(args.rng_seed)

    async def _drive_pipeline(*, carry_over: bool) -> List[np.ndarray]:
        pipe, _det = _build_pipeline(
            n_grid=n_grid, n_fdm=n_fdm, t_det=t_det, device=device,
            carry_over=carry_over, cube_cadence_samples=cadence,
            cube_dtype=cube_dtype, gpu_complex_dtype=gpu_complex_dtype,
        )
        src = SyntheticRxRingSource(
            n_cubes=int(args.n_cubes),
            t_det=t_det, n_fdm=n_fdm, n_grid=n_grid,
            coarse_dm_pc_cm3=coarse_dm, fine_dm_pc_cm3=fine_dm,
            fine_to_coarse=fine_to_coarse,
            rng=np.random.default_rng(rng_seed),
            symmetric_shift_padding=True,
            cube_cadence_samples=cadence,
            overlap_streams=True,
        )
        await src.start()
        cubes: List[np.ndarray] = []
        cube_idx = 0
        it = src.__aiter__()
        tag = "B (carry-over)" if carry_over else "A (full re-image)"
        while True:
            try:
                slot = await it.__anext__()
            except StopAsyncIteration:
                break
            res = pipe.process(slot)
            cubes.append(res.cube.detach().to(torch.float32).cpu().numpy())
            n_cands = (
                len(res.candidates) if res.candidates is not None else 0
            )
            sig_max = (
                float(res.sigma_layer1.detach().abs().max().cpu())
                if res.sigma_layer1 is not None else float("nan")
            )
            print(
                f"  [{tag}] cube {cube_idx}: cands={n_cands} "
                f"sigma_max={sig_max:.3g}",
                flush=True,
            )
            cube_idx += 1
        await src.stop()
        # Free GPU memory before constructing the second pipeline.
        del pipe, _det, src
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return cubes

    cubes_a = await _drive_pipeline(carry_over=False)
    cubes_b = await _drive_pipeline(carry_over=True)

    return cubes_a, cubes_b


def _compare(
    cubes_a: List[np.ndarray],
    cubes_b: List[np.ndarray],
    *,
    t_lo: int,
    atol: float,
    rtol: float,
) -> int:
    # Per-cube diff summary (full cube vs carry-over) so it's obvious
    # which cube first diverges and whether divergence is local to
    # the [:t_lo] copy region or also leaks into the [t_lo:] re-
    # imaged region.
    print("  per-cube diff summary:")
    for k in range(len(cubes_a)):
        a = cubes_a[k]
        b = cubes_b[k]
        co = (
            float(np.max(np.abs(a[:t_lo] - b[:t_lo])))
            if t_lo > 0 else 0.0
        )
        nr = float(np.max(np.abs(a[t_lo:] - b[t_lo:])))
        amag = float(np.max(np.abs(a)))
        bmag = float(np.max(np.abs(b)))
        print(
            f"    cube {k}: |a|_max={amag:.4g} |b|_max={bmag:.4g} "
            f"co-diff={co:.4g} new-diff={nr:.4g}"
        )
    print()
    """Compare per-cube outputs; print worst-cell diagnostics; return
    0 on PASS (all within tol), 1 on FAIL."""
    n = len(cubes_a)
    assert len(cubes_b) == n

    # Cube 0: pipeline B has no prior cube to carry from, so its
    # output is computed by the full-re-image path -- BIT-EQUAL to
    # pipeline A's cube 0. (Sanity check; if this fails, the
    # carry-over plumbing leaked into cube 0 by accident.)
    a0 = cubes_a[0]
    b0 = cubes_b[0]
    if not np.array_equal(a0, b0):
        # Tolerate fp16 launch-time noise via allclose; the carry-
        # over plumbing should not cause ANY divergence on cube 0
        # but the partial-grid kernel launches a different grid even
        # for cube 0 if t_lo != 0; cube 0 always uses t_lo = 0.
        ad = float(np.max(np.abs(a0 - b0)))
        print(
            f"  cube 0 (full re-image both runs): max-abs-diff={ad:.3g}"
        )
        if ad > atol:
            print(
                "  ✗ FAIL cube 0 already diverges -- carry-over "
                "plumbing leaked into the very first cube. Bug in "
                "CubePipeline._run_imager_from_staged guard."
            )
            return 1

    max_diff_carryover = 0.0
    max_diff_new_rows = 0.0
    worst_cube_carryover = -1
    worst_cube_new_rows = -1
    for cube_idx in range(1, n):
        a = cubes_a[cube_idx]
        b = cubes_b[cube_idx]
        if np.isnan(a).any() or np.isinf(a).any():
            print(f"  ✗ FAIL cube {cube_idx}: NaN/Inf in baseline cube")
            return 1
        if np.isnan(b).any() or np.isinf(b).any():
            print(f"  ✗ FAIL cube {cube_idx}: NaN/Inf in carry-over cube")
            return 1
        # Carry-over region: rows [0, t_lo)
        co_diff = float(np.max(np.abs(a[:t_lo] - b[:t_lo]))) if t_lo > 0 else 0.0
        # New region: rows [t_lo, t_det)
        new_diff = float(np.max(np.abs(a[t_lo:] - b[t_lo:])))
        if co_diff > max_diff_carryover:
            max_diff_carryover = co_diff
            worst_cube_carryover = cube_idx
        if new_diff > max_diff_new_rows:
            max_diff_new_rows = new_diff
            worst_cube_new_rows = cube_idx

    print()
    print("  numerical-equivalence summary (cubes 1..N-1):")
    print(f"    max diff in carry-over rows [:{t_lo}]:    {max_diff_carryover:.3g} (worst cube {worst_cube_carryover})")
    print(f"    max diff in new rows        [{t_lo}:]:    {max_diff_new_rows:.3g} (worst cube {worst_cube_new_rows})")
    print(f"    tol: atol={atol:.3g} rtol={rtol:.3g}")
    # Compute the relative tolerance on a per-region basis.
    a_max = max(float(np.max(np.abs(a))) for a in cubes_a[1:])
    pass_carry = max_diff_carryover <= atol + rtol * a_max
    pass_new = max_diff_new_rows <= atol + rtol * a_max

    if pass_carry and pass_new:
        print(
            f"  ✓ PASS -- carry-over output matches full re-image to "
            f"fp16 precision (atol={atol:.3g}, rtol*max={rtol*a_max:.3g})."
        )
        return 0
    print(
        f"  ✗ FAIL -- carry-over output DIVERGES from full re-image. "
        f"Investigate before enabling --cube-pipeline-carry-over-re-imaging "
        f"in production."
    )
    if not pass_carry:
        print(
            "    -- carry-over rows ([0:t_lo]) diverge: sigma rescale "
            "or copy bug (check _run_imager_from_staged sigma snapshot)."
        )
    if not pass_new:
        print(
            "    -- new rows ([t_lo:t_det]) diverge: partial-grid combine "
            "kernel bug (check fused_combine_cuda.py t_lo/t_out math)."
        )
    return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M7.7.2 carry-over numerical equivalence gate. "
                    "Run BEFORE enabling --cube-pipeline-carry-over-"
                    "re-imaging in production.",
    )
    p.add_argument("--n-cubes", type=int, default=8)
    # Default to n_grid=128 because the equivalence gate runs TWO
    # CubePipelines sequentially in fp32 to side-step fp16 saturation
    # in the synthetic test signal (see M77_2_CARRYOVER_DESIGN.md
    # §1a). At n_grid=256 the fp32 imager OOMs a single 2080 Ti.
    p.add_argument("--n-grid", type=int, default=128)
    p.add_argument("--n-fdm", type=int, default=34)
    p.add_argument("--t-det", type=int, default=192)
    p.add_argument("--cube-cadence-samples", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=2e-3)
    p.add_argument("--rtol", type=float, default=2e-3)
    p.add_argument(
        "--dm-plan-path", type=str,
        default="/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz",
    )
    p.add_argument("--coarse-dm-owner-idx", type=int, default=0)
    # Default to fp32 — see M77_2_CARRYOVER_DESIGN.md §1a for why
    # fp16 saturates in this synthetic test.
    p.add_argument(
        "--cube-dtype", type=str, default="fp32", choices=["fp16", "fp32"],
        help=(
            "cube dtype for the equivalence run. The synthetic source "
            "uses unit-variance complex Gaussian streams; with N_grid=256 "
            "the imager DC bin reaches ~1e6, which saturates fp16. Use "
            "fp32 to isolate the carry-over kernel logic. Production "
            "uses fp16 because the per-chgroup calibration and edge mask "
            "scale the imager output back into fp16 range."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if not Path(args.dm_plan_path).exists():
        print(
            f"[carryover-equivalence] FATAL: --dm-plan-path "
            f"{args.dm_plan_path} not found.",
            file=sys.stderr,
        )
        return 2

    t_lo = int(args.t_det) - int(args.cube_cadence_samples)
    print(
        f"[carryover-equivalence] n_cubes={args.n_cubes} t_det={args.t_det} "
        f"cadence={args.cube_cadence_samples} t_lo={t_lo} "
        f"n_grid={args.n_grid} n_fdm={args.n_fdm} device={args.device}"
    )
    print(f"[carryover-equivalence] dm_plan={args.dm_plan_path}")
    print()

    try:
        cubes_a, cubes_b = asyncio.run(_drive_runs(args))
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(
            f"[carryover-equivalence] FATAL: pipeline run failed: {exc}",
            file=sys.stderr,
        )
        return 2

    return _compare(
        cubes_a, cubes_b,
        t_lo=t_lo, atol=float(args.atol), rtol=float(args.rtol),
    )


if __name__ == "__main__":
    sys.exit(main())
