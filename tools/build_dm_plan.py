#!/usr/bin/env python3
"""Build the dsa110-rt DM plan and write it to ``configs/dm_plan.npz``.

Implements plan §3.2 verbatim:

  Step 1  fine_dm: gen_dmtrials_step recursion over the processed band.
  Step 2  coarse_dm: gen_dmtrials_step recursion over chgroup 0's local
          band (= worst-case = highest ν → smallest Δdm; SHARED across
          all 16 corr nodes).
  Step 3  fine_to_coarse + CSR-flat fine_offsets (each fine assigned to
          the largest coarse_dm[c] ≤ fine_dm[f] ⇒ δdm ≥ 0 by construction,
          which is what enables `time_shift_search ≥ 0`).
  Step 4  DEDISP three-table set:
            time_shift_corr_stage1[N_chgroup, N_chan, N_coarse] int32
            time_shift_corr_stage2[N_chgroup, N_coarse]         int32
            time_shift_search[N_fine, N_chgroup]                int32
  Step 5  Per-search-node + per-GPU partitioning with halo overlap.

DoD (plan §8 line 2141 second half):
  pytest tests/test_numerical_conventions.py::test_dm_plan_time_shift_tables
  passes against the produced .npz.

Usage:
  python tools/build_dm_plan.py [--out PATH] [--dm-min 0] [--dm-max 3000]
                                [--tol 1.5] [--t-int-fast-us 262.144]
                                [--t-int-search-us FROM_OPS_YAML]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BLOCK_SAMPLES_NATIVE,
    DELTA_NU_CH_GHZ,
    DETECTOR_K_DM_WIDEST,
    DM_MAX_DEFAULT,
    DM_MIN_DEFAULT,
    DM_PLAN_METADATA_VERSION,
    DM_TOL_DEFAULT,
    K_DM_MS_GHZ2_PC,
    NCHAN_PER_CHGROUP,
    NU_BOT_PROC_GHZ,
    NU_CHGROUP_BOT_GHZ,
    NU_CHGROUP_TOP_GHZ,
    NU_TOP_PROC_GHZ,
    BW_PROC_MHZ,
    N_CHAN_PROC_NATIVE,
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    T_INT_FAST_US_DEFAULT,
    freq_GHz,
)
from dsart.common.contracts import DmPlan  # noqa: E402


# ---------------------------------------------------------------------------
# §3.2 step 0: legacy gen_dmtrials_step recursion (verbatim from
# /media/ubuntu/ssd/vikram/scratch/gen_dmtrials.py, parameterised)
# ---------------------------------------------------------------------------


def gen_dmtrials_step(
    dm_prev: float,
    nu_GHz: float,
    dnu_MHz: float,
    n_chan: int,
    dt_us: float,
    tol: float,
) -> float:
    """One step of the legacy gen_dmtrials.py recursion (plan §3.2 line 488).

    The literal `8.3` in the discriminant carries the standard
    pulsar-astronomy `2 · K_DM_ms_GHz2_pc ≈ 8.3 ms · MHz · GHz⁻³ · pc⁻¹ · cm³`
    parameterisation; kept verbatim from legacy for byte-equivalence with
    `gen_dmtrials.py` outputs.
    """
    n2 = float(n_chan) ** 2
    alp = 1.0 / (16.0 + n2)
    bet = dt_us ** 2
    return n2 * alp * dm_prev + math.sqrt(
        16.0 * alp * (tol ** 2 - n2 * alp) * dm_prev ** 2
        + 16.0 * alp * bet * (tol ** 2 - 1.0) * (nu_GHz ** 3 / 8.3 / dnu_MHz) ** 2
    )


def gen_dm_list(
    dm_min: float,
    dm_max: float,
    nu_GHz: float,
    dnu_MHz: float,
    n_chan: int,
    dt_us: float,
    tol: float,
    n_cap: int | None = None,
) -> np.ndarray:
    """Iterate gen_dmtrials_step from dm_min until exceeding dm_max.

    If ``n_cap`` is given, the iteration also stops once the list reaches
    ``n_cap`` entries (whichever exit triggers first). The natural exit
    (``dms[-1] >= dm_max``) is preserved when ``n_cap is None``; M7.2
    uses the n_cap path to lock the coarse-DM trial count exactly (and
    lets the resulting ``coarse_dm[-1]`` define the effective dm_max).
    """
    dms = [float(dm_min)]
    while dms[-1] < dm_max:
        if n_cap is not None and len(dms) >= n_cap:
            break
        dms.append(
            gen_dmtrials_step(dms[-1], nu_GHz, dnu_MHz, n_chan, dt_us, tol)
        )
    return np.asarray(dms, dtype="float64")


# ---------------------------------------------------------------------------
# §3.2 steps 1-2: fine + coarse trial lists
# ---------------------------------------------------------------------------


def build_fine_list(
    dm_min: float,
    dm_max: float,
    t_int_search_us: float,
    tol: float,
    n_cap: int | None = None,
) -> np.ndarray:
    """Step 1: fine list parameterised over the processed band (§3.2 line 511).

    nu_center = (NU_TOP_PROC_GHZ + NU_BOT_PROC_GHZ) / 2  (≈ 1.405 GHz)
    Δν_MHz    = DELTA_NU_CH_GHZ * 1e3                    (≈ 0.030518 MHz)
    N_chan    = N_CHAN_PROC_NATIVE                       (= 6144)
    Δt_us     = t_int_search_us                          (from operating point)

    M7.2: optional ``n_cap`` truncates the recursion at a fixed trial count;
    used by the n-coarse-cap path so fine_dm stops at the effective dm_max
    (= coarse_dm[-1]) rather than over-running into uncovered DM space.
    """
    nu_center = (NU_TOP_PROC_GHZ + NU_BOT_PROC_GHZ) / 2.0
    return gen_dm_list(
        dm_min=dm_min,
        dm_max=dm_max,
        nu_GHz=nu_center,
        dnu_MHz=DELTA_NU_CH_GHZ * 1e3,
        n_chan=N_CHAN_PROC_NATIVE,
        dt_us=t_int_search_us,
        tol=tol,
        n_cap=n_cap,
    )


def build_coarse_list(
    dm_min: float,
    dm_max: float,
    t_int_fast_us: float,
    tol: float,
    n_cap: int | None = None,
) -> np.ndarray:
    """Step 2: coarse list per-corr (shared across all 16 corrs; §3.2 line 525).

    Worst-case = chgroup 0 (highest ν_chgroup_center → smallest band-local
    Δdm). Use the recursion at chgroup-0's center frequency + chgroup-0's
    local bandwidth (≈ 11.72 MHz = 384 channels × Δν_ch); ``n_chan = 1``
    treats the chgroup as a single wide channel for intra-band smearing
    accounting (the corr-side stage-1 pre-grid integrates 384 → 1).
    Δt_us is t_int_fast_us (corr-side cadence).

    M7.2: ``n_cap`` locks the coarse-DM trial count exactly. The
    effective DM coverage is ``coarse_dm[-1]`` (the last computed
    trial), NOT the requested ``dm_max``; callers using ``n_cap``
    should rebuild the fine list with the effective dm_max so no fine
    trials orphan past the last coarse cell.
    """
    nu_center_chgroup0 = (NU_CHGROUP_TOP_GHZ[0] + NU_CHGROUP_BOT_GHZ[0]) / 2.0
    bw_chgroup_MHz = (NU_CHGROUP_TOP_GHZ[0] - NU_CHGROUP_BOT_GHZ[0]) * 1000.0
    return gen_dm_list(
        dm_min=dm_min,
        dm_max=dm_max,
        nu_GHz=nu_center_chgroup0,
        dnu_MHz=bw_chgroup_MHz,
        n_chan=1,
        dt_us=t_int_fast_us,
        tol=tol,
        n_cap=n_cap,
    )


# ---------------------------------------------------------------------------
# §3.2 step 3: fine_to_coarse + CSR-flat fine_offsets
# ---------------------------------------------------------------------------


def build_fine_to_coarse(
    fine_dm: np.ndarray, coarse_dm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each fine trial to the largest coarse cell ≤ fine_dm.

    By construction this gives δdm = fine_dm[f] - coarse_dm[c] ≥ 0,
    which is the precondition for `time_shift_search ≥ 0` (§3.2 line 579).

    Returns:
        fine_to_coarse  int32[N_fine]: c index per fine.
        fine_offsets_idx int32[N_coarse + 1]: CSR row pointers.
        fine_offsets_flat float64[N_fine]: CSR-grouped (fine_dm - coarse_dm[c])
            in coarse-major, fine-ascending order.
    """
    n_fine = fine_dm.shape[0]
    n_coarse = coarse_dm.shape[0]
    fine_to_coarse = np.searchsorted(coarse_dm, fine_dm, side="right") - 1
    fine_to_coarse = np.clip(fine_to_coarse, 0, n_coarse - 1).astype("int32")

    # CSR row pointers from per-coarse counts.
    counts = np.bincount(fine_to_coarse, minlength=n_coarse)
    fine_offsets_idx = np.zeros(n_coarse + 1, dtype="int32")
    fine_offsets_idx[1:] = np.cumsum(counts).astype("int32")

    # Pack offsets in coarse-major order. Stable sort preserves fine-ascending
    # order within each coarse cell (since fine_dm is itself ascending).
    order = np.argsort(fine_to_coarse, kind="stable")
    fine_offsets_flat = (
        fine_dm[order] - coarse_dm[fine_to_coarse[order]]
    ).astype("float64")

    if n_fine != fine_offsets_flat.shape[0]:
        raise RuntimeError(
            f"CSR length {fine_offsets_flat.shape[0]} != N_fine {n_fine}"
        )
    if int(fine_offsets_idx[-1]) != n_fine:
        raise RuntimeError(
            f"CSR sentinel {fine_offsets_idx[-1]} != N_fine {n_fine}"
        )

    return fine_to_coarse, fine_offsets_idx, fine_offsets_flat


# ---------------------------------------------------------------------------
# §3.2 step 4: DEDISP three-table set
# ---------------------------------------------------------------------------


def _delta_tau_us_array(
    nu_a_GHz: np.ndarray, nu_b_GHz: np.ndarray, dm: np.ndarray
) -> np.ndarray:
    """Vectorised τ_us(nu_a) - τ_us(nu_b) per plan §3.6.1.

    All shapes broadcast. Returns µs.
    """
    return K_DM_MS_GHZ2_PC * dm * (1.0 / nu_a_GHz ** 2 - 1.0 / nu_b_GHz ** 2) * 1e3


def build_time_shifts(
    fine_dm: np.ndarray,
    coarse_dm: np.ndarray,
    fine_to_coarse: np.ndarray,
    t_int_fast_us: float,
    t_int_search_us: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three integer-sample shift tables per §3.6.2 + §3.6.3."""
    n_fine = fine_dm.shape[0]
    n_coarse = coarse_dm.shape[0]

    # Per-channel per-chgroup frequency table  [N_chgroup, N_chan]
    nu_ch = np.array(
        [[freq_GHz(g, ch) for ch in range(NCHAN_PER_CHGROUP)] for g in range(N_CHGROUP)],
        dtype="float64",
    )
    nu_chgroup_bot_arr = np.asarray(NU_CHGROUP_BOT_GHZ, dtype="float64")

    # Stage 1: time_shift_corr_stage1[g, ch, c]
    #   = rint( (τ(ν_chgroup_bot[g], c) - τ(freq(g, ch), c)) / t_int_fast_us )
    #   ≥ 0 (chgroup_bot has the largest τ; ν_chan < ν_chgroup_bot is impossible)
    delta_us_s1 = _delta_tau_us_array(
        nu_chgroup_bot_arr[:, None, None],         # [G, 1, 1]
        nu_ch[:, :, None],                          # [G, C, 1]
        coarse_dm[None, None, :],                   # [1, 1, K]
    )
    time_shift_corr_stage1 = np.rint(delta_us_s1 / t_int_fast_us).astype("int32")

    # Stage 2: time_shift_corr_stage2[g, c]
    #   = rint( (τ(ν_bot_proc, c) - τ(ν_chgroup_bot[g], c)) / t_int_fast_us )
    #   ≥ 0 (ν_bot_proc has the largest τ; chgroup-15 ≈ ν_bot_proc → 0)
    delta_us_s2 = _delta_tau_us_array(
        np.asarray(NU_BOT_PROC_GHZ, dtype="float64"),   # scalar
        nu_chgroup_bot_arr[:, None],                    # [G, 1]
        coarse_dm[None, :],                              # [1, K]
    )
    time_shift_corr_stage2 = np.rint(delta_us_s2 / t_int_fast_us).astype("int32")

    # Search residual: time_shift_search[f, g]
    #   = rint( (τ(ν_bot_proc, δdm) - τ(ν_chgroup_bot[g], δdm)) / t_int_search_us )
    #   = rint( Δτ_us(ν_chgroup_bot[g], ν_bot_proc, δdm) / t_int_search_us )
    delta_dm = fine_dm - coarse_dm[fine_to_coarse]   # [N_fine]; ≥ 0 by construction
    if (delta_dm < 0).any():
        raise RuntimeError(
            "δdm contains negatives — fine_to_coarse binning is broken"
        )
    delta_us_search = _delta_tau_us_array(
        np.asarray(NU_BOT_PROC_GHZ, dtype="float64"),
        nu_chgroup_bot_arr[None, :],                    # [1, G]
        delta_dm[:, None],                               # [N_fine, 1]
    )
    time_shift_search = np.rint(delta_us_search / t_int_search_us).astype("int32")

    if n_fine != time_shift_search.shape[0] or N_CHGROUP != time_shift_search.shape[1]:
        raise RuntimeError(
            f"time_shift_search shape {time_shift_search.shape} != "
            f"({n_fine}, {N_CHGROUP})"
        )
    if time_shift_corr_stage1.shape != (N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse):
        raise RuntimeError(
            f"time_shift_corr_stage1 shape {time_shift_corr_stage1.shape} != "
            f"({N_CHGROUP}, {NCHAN_PER_CHGROUP}, {n_coarse})"
        )
    if time_shift_corr_stage2.shape != (N_CHGROUP, n_coarse):
        raise RuntimeError(
            f"time_shift_corr_stage2 shape {time_shift_corr_stage2.shape} != "
            f"({N_CHGROUP}, {n_coarse})"
        )

    return time_shift_corr_stage1, time_shift_corr_stage2, time_shift_search


# ---------------------------------------------------------------------------
# §3.2 step 5: per-search-node + per-GPU partition with halo overlap
# ---------------------------------------------------------------------------


def _balance_split(
    cum_weight: np.ndarray, n_buckets: int
) -> np.ndarray:
    """Find ``n_buckets - 1`` split indices that balance ``cum_weight`` evenly.

    Returns indices ``s_0, s_1, ..., s_{n_buckets-2}`` such that bucket k
    covers ``[s_{k-1}+1 .. s_k]`` (with ``s_{-1} = -1`` and an implicit
    ``s_{n_buckets-1} = N - 1``).
    """
    total = float(cum_weight[-1])
    targets = [total * (i + 1) / n_buckets for i in range(n_buckets - 1)]
    split = np.searchsorted(cum_weight, targets, side="right")
    # Each split index is the first cell whose cumulative weight EXCEEDS the
    # target; the bucket boundary is the index BEFORE that.
    return np.maximum(split - 1, 0).astype("int32")


def build_partition(
    fine_to_coarse: np.ndarray, n_coarse: int, dm_overlap_coarse: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 5: per-(search, GPU) canonical + consumed coarse-DM ranges.

    Balanced on fine-trials-per-range (not coarse-trials), per §3.2 line 532.
    Halo of ``dm_overlap_coarse`` coarse cells on each side of every range.
    """
    fine_per_coarse = np.bincount(fine_to_coarse, minlength=n_coarse)
    cum = np.cumsum(fine_per_coarse).astype("float64")

    # Search-node boundaries: split at N_SEARCH-1 internal points.
    sn_split = _balance_split(cum, N_SEARCH)
    canonical = np.zeros((N_SEARCH, 2), dtype="int32")
    consumed = np.zeros((N_SEARCH, 2), dtype="int32")
    prev = 0
    for s in range(N_SEARCH):
        hi = int(sn_split[s]) if s < N_SEARCH - 1 else n_coarse - 1
        canonical[s] = (prev, hi)
        consumed[s] = (
            max(0, prev - dm_overlap_coarse),
            min(n_coarse - 1, hi + dm_overlap_coarse),
        )
        prev = hi + 1

    # Per-GPU: split each search node's canonical range into N_SEARCH_GPU halves.
    canonical_per_gpu = np.zeros((N_SEARCH, N_SEARCH_GPU, 2), dtype="int32")
    consumed_per_gpu = np.zeros((N_SEARCH, N_SEARCH_GPU, 2), dtype="int32")
    for s in range(N_SEARCH):
        lo, hi = int(canonical[s, 0]), int(canonical[s, 1])
        # Sub-cumulative within this search node (zero-based)
        sub = cum[lo : hi + 1] - (cum[lo - 1] if lo > 0 else 0.0)
        gpu_split = _balance_split(sub, N_SEARCH_GPU)
        prev_g = lo
        for g in range(N_SEARCH_GPU):
            hi_g = lo + int(gpu_split[g]) if g < N_SEARCH_GPU - 1 else hi
            canonical_per_gpu[s, g] = (prev_g, hi_g)
            consumed_per_gpu[s, g] = (
                max(0, prev_g - dm_overlap_coarse),
                min(n_coarse - 1, hi_g + dm_overlap_coarse),
            )
            prev_g = hi_g + 1

    return canonical, consumed, canonical_per_gpu, consumed_per_gpu


def compute_dm_overlap_coarse(fine_to_coarse: np.ndarray, n_coarse: int) -> int:
    """``ceil(K_dm_widest / (2 · mean_fine_per_coarse))``; ≥ 1 (§3.2 line 535)."""
    fine_per_coarse = np.bincount(fine_to_coarse, minlength=n_coarse)
    mean_fpc = max(1.0, float(fine_per_coarse.mean()))
    return max(1, int(math.ceil(DETECTOR_K_DM_WIDEST / (2.0 * mean_fpc))))


# ---------------------------------------------------------------------------
# Top-level build + CLI
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_t_int_search_us(repo_root: Path) -> float:
    """Read default t_int_search_us from ``configs/operating_points.yaml``.

    Source of truth per plan §6 line ~668; M1 defaults to operating point O-4
    (``t_int_search_us = 524.288``). Schema: ``{default: <id>, rows: {<id>:
    {...}}}`` (M0-pinned). The build_dm_plan output schema captures the value
    used at build time in ``metadata.t_int_search_us`` for reproducibility
    audits.
    """
    ops_path = repo_root / "configs" / "operating_points.yaml"
    with ops_path.open() as f:
        ops = yaml.safe_load(f)
    default_id = ops["default"]
    row = ops["rows"][default_id]
    return float(row["t_int_search_us"])


def build_dm_plan(
    *,
    dm_min: float,
    dm_max: float,
    tol: float,
    t_int_fast_us: float,
    t_int_search_us: float,
    n_coarse_cap: int | None = None,
    repo_root: Path = REPO_ROOT,
) -> DmPlan:
    """Build a complete ``DmPlan`` per plan §3.2 (steps 1-5).

    Args:
        n_coarse_cap: optional cap on the coarse-DM trial count. When set,
            ``coarse_dm`` is truncated to exactly ``n_coarse_cap`` entries
            (assuming the unbounded recursion would have produced at least
            that many; otherwise the natural dm_max exit triggers first
            and a ValueError is raised). The fine list is also rebuilt
            against the effective dm_max = ``coarse_dm[-1]`` so no fine
            trial lands past the last coarse cell.
    """
    coarse_dm = build_coarse_list(
        dm_min, dm_max, t_int_fast_us, tol, n_cap=n_coarse_cap
    )
    if n_coarse_cap is not None and coarse_dm.shape[0] < n_coarse_cap:
        raise ValueError(
            f"n_coarse_cap={n_coarse_cap} but the recursion only produced "
            f"{coarse_dm.shape[0]} trials before exceeding dm_max={dm_max}. "
            f"Raise --dm-max or lower --tol/--dm-min to extend the natural list."
        )
    effective_dm_max = float(coarse_dm[-1])
    # Fine list covers [dm_min, effective_dm_max]. The natural recursion
    # exit produces one trial past effective_dm_max; we KEEP it so the
    # last coarse cell is populated. ``build_fine_to_coarse`` maps that
    # tail trial to the last coarse cell via searchsorted+clip with
    # δdm = (tail - coarse_dm[-1]) ≥ 0 — bounded by one fine recursion
    # step, well below any other cell's δdm spread. Clipping it out
    # leaves the last coarse cell empty (degenerate; wastes one DM
    # trial worth of corr-side compute), so we deliberately let it
    # overshoot.
    fine_dm = build_fine_list(
        dm_min, effective_dm_max, t_int_search_us, tol
    )
    fine_to_coarse, fine_offsets_idx, fine_offsets_flat = build_fine_to_coarse(
        fine_dm, coarse_dm
    )
    ts_s1, ts_s2, ts_search = build_time_shifts(
        fine_dm, coarse_dm, fine_to_coarse, t_int_fast_us, t_int_search_us
    )
    n_coarse = coarse_dm.shape[0]
    overlap = compute_dm_overlap_coarse(fine_to_coarse, n_coarse)
    canon, cons, canon_g, cons_g = build_partition(fine_to_coarse, n_coarse, overlap)

    metadata = {
        "band_top_GHz": float(NU_TOP_PROC_GHZ),
        "band_bot_GHz": float(NU_BOT_PROC_GHZ),
        "BW_MHz": float(BW_PROC_MHZ),
        "N_chan_proc_native": int(N_CHAN_PROC_NATIVE),
        "t_int_fast_us": float(t_int_fast_us),
        "t_int_search_us": float(t_int_search_us),
        "tol": float(tol),
        "build_utc_ns": int(time.time_ns()),
        "git_sha": _git_sha(repo_root),
        "version": DM_PLAN_METADATA_VERSION,
    }

    return DmPlan(
        dm_min=float(dm_min),
        dm_max=float(effective_dm_max),
        tol=float(tol),
        fine_dm=fine_dm,
        coarse_dm=coarse_dm,
        fine_to_coarse=fine_to_coarse,
        fine_offsets_idx=fine_offsets_idx,
        fine_offsets_flat=fine_offsets_flat,
        time_shift_corr_stage1=ts_s1,
        time_shift_corr_stage2=ts_s2,
        time_shift_search=ts_search,
        dm_idx_range_canonical=canon,
        dm_idx_range_consumed=cons,
        dm_idx_range_canonical_per_gpu=canon_g,
        dm_idx_range_consumed_per_gpu=cons_g,
        dm_overlap_coarse=overlap,
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the dsa110-rt DM plan (§3.2).")
    p.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "configs" / "dm_plan.npz"),
        help="output .npz path (default: configs/dm_plan.npz)",
    )
    p.add_argument("--dm-min", type=float, default=DM_MIN_DEFAULT)
    p.add_argument("--dm-max", type=float, default=DM_MAX_DEFAULT)
    p.add_argument("--tol", type=float, default=DM_TOL_DEFAULT)
    p.add_argument(
        "--n-coarse-cap",
        type=int,
        default=None,
        help=(
            "Optional cap on coarse-DM trial count. Truncates the legacy "
            "gen_dmtrials recursion at exactly N trials and uses "
            "coarse_dm[-1] as the effective dm_max (the fine list is "
            "clipped to match). Used by M7.2 to compare {6,7,8}-trial "
            "operating points while preserving (dm_min, tol)."
        ),
    )
    p.add_argument(
        "--t-int-fast-us",
        type=float,
        default=T_INT_FAST_US_DEFAULT,
        help=f"corr-side cadence (default: {T_INT_FAST_US_DEFAULT})",
    )
    p.add_argument(
        "--t-int-search-us",
        type=float,
        default=None,
        help="search-side cadence (default: read from operating_points.yaml)",
    )
    p.add_argument("--quiet", action="store_true", help="no progress prints")
    args = p.parse_args(argv)

    t_int_search_us = (
        args.t_int_search_us
        if args.t_int_search_us is not None
        else _load_t_int_search_us(REPO_ROOT)
    )

    plan = build_dm_plan(
        dm_min=args.dm_min,
        dm_max=args.dm_max,
        tol=args.tol,
        t_int_fast_us=args.t_int_fast_us,
        t_int_search_us=t_int_search_us,
        n_coarse_cap=args.n_coarse_cap,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_npz(str(out_path))

    if not args.quiet:
        print(f"build_dm_plan: wrote {out_path}")
        print(f"  N_fine            = {plan.fine_dm.shape[0]}")
        print(f"  N_coarse          = {plan.coarse_dm.shape[0]}")
        print(f"  dm_overlap_coarse = {plan.dm_overlap_coarse}")
        print(
            f"  time_shift_corr_stage1 = {plan.time_shift_corr_stage1.shape} int32 "
            f"(max={int(plan.time_shift_corr_stage1.max())} samples)"
        )
        print(
            f"  time_shift_corr_stage2 = {plan.time_shift_corr_stage2.shape} int32 "
            f"(max={int(plan.time_shift_corr_stage2.max())} samples)"
        )
        print(
            f"  time_shift_search      = {plan.time_shift_search.shape} int32 "
            f"(max={int(plan.time_shift_search.max())} samples)"
        )
        print(f"  metadata.git_sha = {plan.metadata['git_sha'][:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
