#!/usr/bin/env python3
"""Build a DM plan from an EXPLICIT coarse/fine grid (§3.2 schema).

``build_dm_plan.py`` derives its grid from the legacy tolerance-driven
Levin recursion, so it can only produce grids whose bucket widths follow
from ``(dm_min, dm_max, tol)``.  The 2026-09-21 sensitivity study wanted
a grid specified directly instead: a fixed DM ceiling, bucket widths on
a chosen ramp, and coarse DMs nudged off the bucket centres onto
favourable integer-rounding positions.  This tool writes exactly that,
re-using ``build_dm_plan``'s shift-table / partition helpers so the
on-disk schema and sign conventions are identical.

Why the ramp and the offsets are what they are
----------------------------------------------
The intra-chgroup residual dispersion at |fine - coarse| is baked into
the uv grid by the multi-frequency-synthesis gridding and cannot be
undone downstream, so the recovered S/N is set by the bucket WIDTH.
With ``n_coarse`` pinned at 8 by the corr-node budget (~3.1 ms per
coarse DM against a ~4 ms margin) the width is ``DM range / 8``, which
makes the DM ceiling the only lever.  At a FIXED ceiling a uniform grid
is near-optimal and a mild ramp is marginally better under a low-DM-
weighted burst population; heavy grading is counter-productive.

NOTE ON THE STORED SHIFT TABLES.  ``time_shift_corr_stage1/stage2/
search`` are written with the v2 contract's ν_chgroup_bot reference
(hence ``time_shift_search[:, 15] == 0``).  The RUNTIME does NOT read
them: ``coarse_dm/dm_plan.py::from_canonical`` recomputes the stage-1
table and ``transport/production_rx_ring.py`` recomputes the search
table, both against the chgroup TOP channel (the 2026-06-03 Convention-A
fix).  The metadata records this explicitly so nobody mistakes the
stored arrays for what runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dm_plan import (  # noqa: E402
    _git_sha,
    _load_t_int_search_us,
    build_partition,
    build_time_shifts,
    compute_dm_overlap_coarse,
    gen_dmtrials_step,
)
from dsart.common.constants import (  # noqa: E402
    BW_PROC_MHZ,
    DELTA_NU_CH_GHZ,
    DM_PLAN_METADATA_VERSION,
    N_CHAN_PROC_NATIVE,
    NU_BOT_PROC_GHZ,
    NU_TOP_PROC_GHZ,
    T_INT_FAST_US_DEFAULT,
)
from dsart.common.contracts import DmPlan  # noqa: E402


def effective_tol(
    step_pc_cm3: float,
    dm_at_step: float,
    t_int_search_us: float,
    chan_sum_factor: int,
) -> float:
    """The Levin tolerance this grid's WORST fine step corresponds to.

    ``tol`` is provenance-only in the schema (validated > 0, never
    consumed), but writing a fabricated value would be misleading, so
    invert ``build_dm_plan.gen_dmtrials_step`` -- the very function that
    turns a tolerance into a step -- on the largest fine step in the
    grid.  The result is the smearing tolerance the grid actually meets
    in its worst place.
    """
    nu = (NU_TOP_PROC_GHZ + NU_BOT_PROC_GHZ) / 2.0
    dnu = (DELTA_NU_CH_GHZ * 1e3) * float(chan_sum_factor)
    n_chan = N_CHAN_PROC_NATIVE // int(chan_sum_factor)
    lo, hi = 1.0 + 1e-12, 4.0
    target = float(dm_at_step) + float(step_pc_cm3)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        got = gen_dmtrials_step(float(dm_at_step), nu, dnu, n_chan,
                                float(t_int_search_us), mid)
        if got < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ramp_widths(dm_min: float, dm_max: float, n_coarse: int,
                beta: float) -> np.ndarray:
    """Bucket widths on a linear ramp ``1 + beta*k``, summing to the span."""
    w = 1.0 + float(beta) * np.arange(int(n_coarse), dtype="float64")
    return w / w.sum() * (float(dm_max) - float(dm_min))


def build_explicit(
    dm_min: float,
    dm_max: float,
    n_coarse: int,
    n_fine_per_coarse: int,
    beta: float,
    coarse_offsets: np.ndarray | None,
    t_int_fast_us: float,
    t_int_search_us: float,
    chan_sum_factor: int,
    widths: np.ndarray | None = None,
) -> DmPlan:
    w = ramp_widths(dm_min, dm_max, n_coarse, beta) if widths is None \
        else np.asarray(widths, dtype="float64")
    if w.shape != (n_coarse,):
        raise ValueError(f"widths shape {w.shape} != ({n_coarse},)")
    edges = float(dm_min) + np.concatenate([[0.0], np.cumsum(w)])
    centres = 0.5 * (edges[1:] + edges[:-1])
    offs = (np.zeros(n_coarse, dtype="float64") if coarse_offsets is None
            else np.asarray(coarse_offsets, dtype="float64"))
    if offs.shape != (n_coarse,):
        raise ValueError(f"coarse_offsets shape {offs.shape} != ({n_coarse},)")
    coarse_dm = (centres + offs).astype("float64")
    if not np.all(np.diff(coarse_dm) > 0):
        raise ValueError("coarse_dm must be strictly increasing after offsets")
    # Fine trials: uniform inside each bucket.  Uniform is the right
    # choice -- a dynamic-programming placement that keeps the integer-
    # rounding structure gains nothing at plan level and costs the tail.
    fine, f2c = [], []
    for k in range(n_coarse):
        step = w[k] / n_fine_per_coarse
        fine.extend(edges[k] + step * (np.arange(n_fine_per_coarse) + 0.5))
        f2c.extend([k] * n_fine_per_coarse)
    fine_dm = np.asarray(fine, dtype="float64")
    # fine_to_coarse is the STRUCTURAL block map (f // K), not a
    # nearest-coarse search: the per-GPU partition assumes exactly
    # K fines per coarse and dithered coarse DMs would otherwise let a
    # boundary trial migrate to its neighbour.
    fine_to_coarse = np.asarray(f2c, dtype="int32")
    fine_offsets_idx = (np.arange(n_coarse + 1, dtype="int32")
                        * int(n_fine_per_coarse))
    fine_offsets_flat = (fine_dm - coarse_dm[fine_to_coarse]).astype("float64")

    ts1, ts2, tss = build_time_shifts(
        fine_dm, coarse_dm, fine_to_coarse, t_int_fast_us, t_int_search_us
    )
    overlap = compute_dm_overlap_coarse(fine_to_coarse, n_coarse)
    canon, cons, canon_g, cons_g = build_partition(n_coarse)

    steps = w / float(n_fine_per_coarse)
    i_worst = int(np.argmax(steps))
    tol_eff = effective_tol(
        float(steps[i_worst]), float(centres[i_worst]),
        t_int_search_us, chan_sum_factor,
    )
    metadata = {
        "band_top_GHz": float(NU_TOP_PROC_GHZ),
        "band_bot_GHz": float(NU_BOT_PROC_GHZ),
        "BW_MHz": float(BW_PROC_MHZ),
        "N_chan_proc_native": int(N_CHAN_PROC_NATIVE),
        "t_int_fast_us": float(t_int_fast_us),
        "t_int_search_us": float(t_int_search_us),
        "tol": float(tol_eff),
        "coarse_seed_tol": float(tol_eff),
        "build_utc_ns": int(time.time_ns()),
        "git_sha": _git_sha(REPO_ROOT),
        "version": DM_PLAN_METADATA_VERSION,
        "chan_sum_factor": int(chan_sum_factor),
        "fine_chan_sum_factor": int(chan_sum_factor),
        "fine_n_chan_effective": int(N_CHAN_PROC_NATIVE // chan_sum_factor),
        "fine_dnu_mhz_effective": float(
            (DELTA_NU_CH_GHZ * 1e3) * chan_sum_factor),
        "dm_plan_version_tag": "explicit-v1",
        "builder": "tools/build_dm_plan_explicit.py",
        "explicit_beta_ramp": float(beta),
        "explicit_bucket_widths_pc_cm3": [float(x) for x in w],
        "explicit_bucket_edges_pc_cm3": [float(x) for x in edges],
        "explicit_coarse_centres_pc_cm3": [float(x) for x in centres],
        "explicit_coarse_offsets_pc_cm3": [float(x) for x in offs],
        "explicit_n_fine_per_coarse": int(n_fine_per_coarse),
        "explicit_tol_effective_worst_step": float(tol_eff),
        "explicit_worst_fine_step_pc_cm3": float(steps[i_worst]),
        # Make the stored-vs-runtime shift-table divergence explicit.
        "stored_shift_tables_reference": "nu_chgroup_bot (v2 contract)",
        "runtime_shift_tables": (
            "RECOMPUTED at runtime against the chgroup TOP channel "
            "(coarse_dm/dm_plan.py::from_canonical and "
            "transport/production_rx_ring.py); the stored "
            "time_shift_* arrays are NOT what runs and must not be used "
            "as a reference."
        ),
        "provenance": (
            "2026-09-21 sensitivity study; see "
            "_inspect/sensitivity/DM_SCHEME_FINAL.md. Coarse offsets are "
            "the per-bucket integer-rounding optimum found offline."
        ),
    }
    return DmPlan(
        dm_min=float(dm_min), dm_max=float(edges[-1]), tol=float(tol_eff),
        fine_dm=fine_dm, coarse_dm=coarse_dm, fine_to_coarse=fine_to_coarse,
        fine_offsets_idx=fine_offsets_idx,
        fine_offsets_flat=fine_offsets_flat,
        time_shift_corr_stage1=ts1, time_shift_corr_stage2=ts2,
        time_shift_search=tss,
        dm_idx_range_canonical=canon, dm_idx_range_consumed=cons,
        dm_idx_range_canonical_per_gpu=canon_g,
        dm_idx_range_consumed_per_gpu=cons_g,
        dm_overlap_coarse=overlap, metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--dm-min", type=float, required=True)
    p.add_argument("--dm-max", type=float, required=True)
    p.add_argument("--n-coarse", type=int, default=8)
    p.add_argument("--n-fine-per-coarse", type=int, default=34)
    p.add_argument("--beta", type=float, default=0.10,
                   help="bucket-width ramp: width_k ~ 1 + beta*k")
    p.add_argument("--coarse-offsets", type=str, default=None,
                   help="comma-separated per-bucket offsets (pc/cm3) applied "
                        "to the bucket centres; default all zero")
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--t-int-fast-us", type=float,
                   default=T_INT_FAST_US_DEFAULT)
    p.add_argument("--t-int-search-us", type=float, default=None)
    a = p.parse_args(argv)
    t_search = (a.t_int_search_us if a.t_int_search_us is not None
                else _load_t_int_search_us(REPO_ROOT))
    offs = (None if a.coarse_offsets is None
            else np.array([float(x) for x in a.coarse_offsets.split(",")]))
    plan = build_explicit(
        a.dm_min, a.dm_max, a.n_coarse, a.n_fine_per_coarse, a.beta, offs,
        a.t_int_fast_us, t_search, a.chan_sum_factor,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_npz(str(out))
    print(f"wrote {out}")
    print(f"  dm_min={plan.dm_min:.2f} dm_max={plan.dm_max:.2f} "
          f"n_coarse={plan.coarse_dm.size} n_fine={plan.fine_dm.size}")
    print(f"  coarse_dm = {np.round(plan.coarse_dm, 2)}")
    print(f"  fine step per bucket = "
          f"{np.round(np.array(plan.metadata['explicit_bucket_widths_pc_cm3'])/a.n_fine_per_coarse, 3)}")
    print(f"  effective tol (worst fine step) = {plan.tol:.4f}")
    print(f"  |time_shift_search| max = {int(np.abs(plan.time_shift_search).max())} samples"
          f"  (STORED nu_chgroup_bot convention; runtime recomputes with TOP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
