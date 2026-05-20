#!/usr/bin/env python3
"""Build the dsa110-rt v2 DM plan and write it to ``configs/dm_plan.npz``.

v2 scheme (user-clarified 2026-05-18; supersedes v1):

  Step 1  Bottom-sub-band Levin seed (chgroup 15, lowest ν → tightest
          spacing) to find ``dm_max_effective``: take exactly N_coarse=8
          steps of the recursion from ``dm_min``; ``dm_max_effective`` =
          ``coarse_seed[7]``. Discard the seed POSITIONS afterwards.
  Step 2  Full-band Levin recursion over ``[dm_min, dm_max_effective]`` →
          raw fine-DM list (length N_fine_raw).
  Step 3  Trim the tail to a multiple of N_coarse=8:
          ``N_fine = (N_fine_raw // 8) * 8``, ``K = N_fine / 8``.
  Step 4  Place coarse DMs at the K-fine bucket midpoints:
          ``coarse_dm[i] = fine_dm[i*K + K//2]``. Each GPU owns K fine
          DMs SYMMETRIC about its assigned coarse_dm[i].
  Step 5  ``fine_to_coarse[f] = f // K`` (1:1 GPU ownership); ``δdm =
          fine_dm[f] - coarse_dm[f // K]`` is SIGNED. ``time_shift_search``
          is correspondingly SIGNED; the combiner reads
          ``stream[t - shift]`` which handles both signs (PAST data
          naturally available; FUTURE data via one-sided rewind).
  Step 6  Per-(search, GPU) partition: ``[s, g] = (2s+g, 2s+g)``. No halo,
          no inter-GPU overlap; ``dm_overlap_coarse = 0``.

  Also written by Step 4:
            time_shift_corr_stage1[N_chgroup, N_chan, N_coarse] int32
            time_shift_corr_stage2[N_chgroup, N_coarse]         int32
            time_shift_search    [N_fine, N_chgroup]            int32 (SIGNED)

Usage:
  python tools/build_dm_plan.py [--out PATH] [--dm-min 100] [--dm-max 3000]
                                [--tol 1.6] [--t-int-fast-us 1048.576]
                                [--t-int-search-us FROM_OPS_YAML]
                                [--n-coarse-cap 8]
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
    DELTA_NU_CH_GHZ,
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
    chan_sum_factor: int,
    n_cap: int | None = None,
) -> np.ndarray:
    """Step 1: fine list parameterised over the processed band (§3.2 line 511).

    nu_center = (NU_TOP_PROC_GHZ + NU_BOT_PROC_GHZ) / 2  (≈ 1.405 GHz)
    Δν_MHz    = (DELTA_NU_CH_GHZ * 1e3) * chan_sum_factor
    N_chan    = N_CHAN_PROC_NATIVE // chan_sum_factor
    Δt_us     = t_int_search_us                          (from operating point)

    Production M7.3 path runs corr-side ``--chan-sum-factor 8`` before
    coarse/fine dedispersion, so the effective full-band channelisation
    for Levin spacing is:
      Δν_MHz = 0.244140625
      N_chan = 768
    (with the same ~1.405 GHz center frequency). This function takes
    ``chan_sum_factor`` explicitly so the plan builder tracks the actual
    corr-side pipeline shape instead of silently assuming native 6144 chans.

    M7.2: optional ``n_cap`` truncates the recursion at a fixed trial count;
    used by the n-coarse-cap path so fine_dm stops at the effective dm_max
    (= coarse_dm[-1]) rather than over-running into uncovered DM space.
    """
    if chan_sum_factor <= 0:
        raise ValueError(f"chan_sum_factor must be > 0, got {chan_sum_factor}")
    if N_CHAN_PROC_NATIVE % chan_sum_factor != 0:
        raise ValueError(
            f"N_CHAN_PROC_NATIVE={N_CHAN_PROC_NATIVE} is not divisible by "
            f"chan_sum_factor={chan_sum_factor}"
        )
    nu_center = (NU_TOP_PROC_GHZ + NU_BOT_PROC_GHZ) / 2.0
    dnu_mhz = (DELTA_NU_CH_GHZ * 1e3) * float(chan_sum_factor)
    n_chan_eff = N_CHAN_PROC_NATIVE // chan_sum_factor
    return gen_dm_list(
        dm_min=dm_min,
        dm_max=dm_max,
        nu_GHz=nu_center,
        dnu_MHz=dnu_mhz,
        n_chan=n_chan_eff,
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
    """Step 2: coarse list using the BOTTOM-MOST sub-band (user-clarified
    2026-05-18; was previously chgroup 0 = TOP sub-band, which is physically
    backwards — see fix below).

    Per the Levin-thesis recursion (the same gen_dmtrials formula reused
    for the fine list), the maximum allowable ΔDM step is limited by the
    DM-smearing tolerance Δt_smear ≤ tol · Δt_int. Smearing within a
    chgroup of bandwidth B at center ν scales as B · ΔDM / ν³ — so the
    LOWEST-ν chgroup (= NU_CHGROUP_BOT_GHZ[15]) gives the WORST smear and
    therefore the TIGHTEST coarse-DM spacing. Using the top chgroup
    (formerly hard-coded as chgroup[0]) gives the LOOSEST spacing, which
    under-resolves the high-DM end and is exactly what the prior builder
    was doing — the M7.2 conversation 2026-05-18 surfaced this bug. The
    docstring claim "Worst-case = chgroup 0 (highest ν → smallest Δdm)"
    was wrong on the physics: higher ν → smaller smear → LARGER allowed
    ΔDM → LOOSER spacing → smaller N_coarse for a fixed dm_max → coarser
    grid. We want the tightest spacing, hence the bottom chgroup.

    Use the recursion at chgroup-15's center frequency + chgroup-15's
    local bandwidth (≈ 11.72 MHz = 384 channels × Δν_ch); ``n_chan = 1``
    treats the chgroup as a single wide channel for intra-band smearing
    accounting (the corr-side stage-1 pre-grid integrates 384 → 1; the
    residual smear after stage-1 is the chgroup-wide bandwidth applied
    to the ΔDM step).
    Δt_us is t_int_fast_us (corr-side cadence).

    M7.2: ``n_cap`` locks the coarse-DM trial count exactly. The
    effective DM coverage is ``coarse_dm[-1]`` (the last computed
    trial), NOT the requested ``dm_max``; callers using ``n_cap``
    should rebuild the fine list with the effective dm_max so no fine
    trials orphan past the last coarse cell.
    """
    BOTTOM_CHGROUP = N_CHGROUP - 1
    nu_center_chgroup_bottom = (
        NU_CHGROUP_TOP_GHZ[BOTTOM_CHGROUP] + NU_CHGROUP_BOT_GHZ[BOTTOM_CHGROUP]
    ) / 2.0
    bw_chgroup_MHz = (
        NU_CHGROUP_TOP_GHZ[BOTTOM_CHGROUP] - NU_CHGROUP_BOT_GHZ[BOTTOM_CHGROUP]
    ) * 1000.0
    return gen_dm_list(
        dm_min=dm_min,
        dm_max=dm_max,
        nu_GHz=nu_center_chgroup_bottom,
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
    """v2 even-K partition: ``fine_to_coarse[f] = f // K`` with
    ``K = N_fine // N_coarse``.

    Pre-condition: N_fine MUST be divisible by N_coarse (the caller —
    ``build_dm_plan`` v2 — trims the fine list tail to enforce this).
    The coarse positions ``coarse_dm[i] = fine_dm[i*K + K//2]`` are the
    midpoints of the per-GPU K-fine buckets, so δdm = fine_dm[f] -
    coarse_dm[f//K] is SIGNED (K//2 fines per coarse sit BELOW with
    δdm < 0, K - K//2 - 1 sit ABOVE with δdm > 0, and exactly one is
    EQUAL with δdm = 0). The downstream ``time_shift_search`` table is
    correspondingly signed; the search-side combiner reads
    ``stream[t - shift]`` which handles both signs (negative shifts
    read PAST data from the rolling RX ring; positive shifts read
    FUTURE data delivered by one-sided rewind).

    Returns:
        fine_to_coarse  int32[N_fine]: c index per fine (= f // K).
        fine_offsets_idx int32[N_coarse + 1]: CSR row pointers (each
            coarse cell holds exactly K fines, so this is a simple
            ``[0, K, 2K, ..., N_fine]`` sequence).
        fine_offsets_flat float64[N_fine]: CSR-grouped
            ``(fine_dm[f] - coarse_dm[f // K])`` values; SIGNED.
    """
    n_fine = fine_dm.shape[0]
    n_coarse = coarse_dm.shape[0]
    if n_coarse <= 0:
        raise ValueError(f"n_coarse={n_coarse} must be > 0")
    if n_fine % n_coarse != 0:
        raise ValueError(
            f"v2 DM plan: N_fine={n_fine} must be divisible by N_coarse="
            f"{n_coarse}; caller (build_dm_plan v2) trims the fine list "
            f"tail to enforce this. Got K = {n_fine}/{n_coarse} = "
            f"{n_fine / n_coarse:.3f} (non-integer)."
        )
    k = n_fine // n_coarse
    fine_to_coarse = (np.arange(n_fine, dtype="int64") // k).astype("int32")

    # CSR row pointers from per-coarse counts (uniform K per cell under v2).
    fine_offsets_idx = (np.arange(n_coarse + 1, dtype="int32") * k)

    # No re-sort: fines are already grouped in coarse-major order under
    # f // K, and ascending within each cell (since fine_dm is ascending).
    fine_offsets_flat = (fine_dm - coarse_dm[fine_to_coarse]).astype("float64")

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
    delta_dm = fine_dm - coarse_dm[fine_to_coarse]   # [N_fine]; SIGNED under v2
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


def build_partition(
    n_coarse: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v2 per-(search, GPU) canonical + consumed coarse-DM ranges.

    Each GPU owns exactly ONE coarse DM under the v2 scheme (no halo, no
    inter-GPU overlap, 1:1 mapping). GPU (s, g) owns coarse index
    ``i = N_SEARCH_GPU * s + g``. Pre-condition:
    ``n_coarse == N_SEARCH * N_SEARCH_GPU`` (= 8 under production).

    Per-search-node spans two contiguous coarse cells (g=0 and g=1):
    ``dm_idx_range_{canonical,consumed}[s] = (2s, 2s+1)``.
    Per-GPU is degenerate point: ``[s, g] = (2s+g, 2s+g)``.

    Returns:
        canonical: ``[N_SEARCH, 2] int32`` ``(lo, hi)`` per search node.
        consumed:  ``[N_SEARCH, 2] int32`` identical (dm_overlap_coarse=0).
        canonical_per_gpu: ``[N_SEARCH, N_SEARCH_GPU, 2] int32``.
        consumed_per_gpu:  ``[N_SEARCH, N_SEARCH_GPU, 2] int32`` identical.
    """
    expected = N_SEARCH * N_SEARCH_GPU
    if n_coarse != expected:
        raise ValueError(
            f"v2 DM plan requires n_coarse == N_SEARCH * N_SEARCH_GPU = "
            f"{expected}; got {n_coarse}"
        )
    canonical = np.zeros((N_SEARCH, 2), dtype="int32")
    consumed = np.zeros((N_SEARCH, 2), dtype="int32")
    canonical_per_gpu = np.zeros((N_SEARCH, N_SEARCH_GPU, 2), dtype="int32")
    consumed_per_gpu = np.zeros((N_SEARCH, N_SEARCH_GPU, 2), dtype="int32")
    for s in range(N_SEARCH):
        lo = N_SEARCH_GPU * s
        hi = N_SEARCH_GPU * s + (N_SEARCH_GPU - 1)
        canonical[s] = (lo, hi)
        consumed[s] = (lo, hi)
        for g in range(N_SEARCH_GPU):
            i = N_SEARCH_GPU * s + g
            canonical_per_gpu[s, g] = (i, i)
            consumed_per_gpu[s, g] = (i, i)
    return canonical, consumed, canonical_per_gpu, consumed_per_gpu


def compute_dm_overlap_coarse(fine_to_coarse: np.ndarray, n_coarse: int) -> int:
    """v2: dm_overlap_coarse = 0 (no halo, no inter-GPU coarse overlap).

    Under v2 each GPU owns exactly one coarse cube with no neighbors
    consumed. The legacy ``ceil(K_dm_widest / (2 · mean_fine_per_coarse))``
    formula is replaced by a constant 0; the ``fine_to_coarse`` and
    ``n_coarse`` arguments are kept for ABI symmetry with v1 callers.
    """
    del fine_to_coarse, n_coarse  # v2: unused
    return 0


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
    chan_sum_factor: int = 8,
    tol_coarse_seed: float | None = None,
    n_coarse_cap: int | None = None,
    repo_root: Path = REPO_ROOT,
) -> DmPlan:
    """Build a complete v2 ``DmPlan`` (user-clarified scheme, 2026-05-18).

    Flow:
      1. Bottom-sub-band Levin recursion (chgroup 15) from ``dm_min`` with
         ``n_cap = N_coarse = N_SEARCH * N_SEARCH_GPU = 8``. The 8th step's
         DM value sets ``dm_max_effective``; the recursion's POSITIONS are
         then discarded (only its final-step DM is used).
      2. Full-band Levin recursion (processed band, per-channel Δν, n_chan
         = N_CHAN_PROC_NATIVE) from ``dm_min`` to ``dm_max_effective`` → a
         raw fine-DM list of length ``N_fine_raw``.
      3. Trim the tail of the raw fine list to the largest multiple of
         ``N_coarse``: ``N_fine = (N_fine_raw // N_coarse) * N_coarse`` and
         ``K = N_fine / N_coarse``.
      4. Place coarse DMs at the bucket midpoints:
         ``coarse_dm[i] = fine_dm[i*K + K//2]`` for ``i = 0..N_coarse-1``.
         This makes the coarse DMs sit naturally in the middle of each
         GPU's K-fine ownership range (K/2 fines below, ~K/2 above).
      5. ``fine_to_coarse[f] = f // K`` (1:1 even split), δdm = fine_dm[f]
         - coarse_dm[f // K] is SIGNED → ``time_shift_search`` is SIGNED.
      6. Per-(search, GPU) partition is 1:1: ``[s, g] = (2s+g, 2s+g)``.
         No halo, no inter-GPU overlap, ``dm_overlap_coarse = 0``.

    Args:
        n_coarse_cap: must equal ``N_SEARCH * N_SEARCH_GPU`` (default 8)
            under v2. Retained for ABI compatibility; non-default values
            will raise. Defaults to ``N_SEARCH * N_SEARCH_GPU`` when None.
        dm_max: provides the upper bound for the bottom-sub-band recursion
            search. dm_max_effective is derived from the recursion's 8th
            step (typically << dm_max). dm_max must be > dm_min and large
            enough that the recursion does not exit early.
    """
    n_coarse_target = N_SEARCH * N_SEARCH_GPU  # = 8 in production
    n_cap = n_coarse_cap if n_coarse_cap is not None else n_coarse_target
    if n_cap != n_coarse_target:
        raise ValueError(
            f"v2 DM plan requires n_coarse == N_SEARCH * N_SEARCH_GPU = "
            f"{n_coarse_target}; got n_coarse_cap={n_cap}"
        )

    coarse_seed_tol = float(tol if tol_coarse_seed is None else tol_coarse_seed)

    # Step 1: bottom-sub-band Levin seed to determine dm_max_effective.
    coarse_seed = build_coarse_list(
        dm_min, dm_max, t_int_fast_us, coarse_seed_tol, n_cap=n_cap
    )
    if coarse_seed.shape[0] < n_cap:
        raise ValueError(
            f"bottom-sub-band Levin produced only {coarse_seed.shape[0]} "
            f"trials before dm_max={dm_max}; need {n_cap}. Raise --dm-max "
            f"or lower --tol/--dm-min."
        )
    dm_max_effective = float(coarse_seed[-1])

    # Step 2: full-band fine list over [dm_min, dm_max_effective].
    fine_raw = build_fine_list(
        dm_min, dm_max_effective, t_int_search_us, tol, chan_sum_factor
    )
    n_fine_raw = fine_raw.shape[0]
    if n_fine_raw < n_cap:
        raise ValueError(
            f"full-band Levin produced N_fine_raw={n_fine_raw} < N_coarse="
            f"{n_cap}; cannot form even-K partition. Loosen --tol or raise "
            f"--dm-max."
        )

    # Step 3: trim to multiple of N_coarse.
    n_fine = (n_fine_raw // n_cap) * n_cap
    fine_dm = fine_raw[:n_fine].astype("float64")
    k = n_fine // n_cap
    final_dm_max = float(fine_dm[-1])

    # Step 4: coarse_dm[i] = fine_dm[i*K + K//2] (bucket midpoints).
    coarse_dm = np.array(
        [fine_dm[i * k + k // 2] for i in range(n_cap)],
        dtype="float64",
    )

    # Step 5: fine_to_coarse + CSR offsets (signed deltas).
    fine_to_coarse, fine_offsets_idx, fine_offsets_flat = build_fine_to_coarse(
        fine_dm, coarse_dm
    )
    ts_s1, ts_s2, ts_search = build_time_shifts(
        fine_dm, coarse_dm, fine_to_coarse, t_int_fast_us, t_int_search_us
    )

    # Step 6: 1:1 GPU-owns-1-coarse partition (no halo).
    overlap = compute_dm_overlap_coarse(fine_to_coarse, n_cap)  # = 0
    canon, cons, canon_g, cons_g = build_partition(n_cap)

    metadata = {
        "band_top_GHz": float(NU_TOP_PROC_GHZ),
        "band_bot_GHz": float(NU_BOT_PROC_GHZ),
        "BW_MHz": float(BW_PROC_MHZ),
        "N_chan_proc_native": int(N_CHAN_PROC_NATIVE),
        "t_int_fast_us": float(t_int_fast_us),
        "t_int_search_us": float(t_int_search_us),
        "tol": float(tol),
        "coarse_seed_tol": coarse_seed_tol,
        "build_utc_ns": int(time.time_ns()),
        "git_sha": _git_sha(repo_root),
        "version": DM_PLAN_METADATA_VERSION,
        "fine_chan_sum_factor": int(chan_sum_factor),
        "fine_n_chan_effective": int(N_CHAN_PROC_NATIVE // chan_sum_factor),
        "fine_dnu_mhz_effective": float((DELTA_NU_CH_GHZ * 1e3) * chan_sum_factor),
        # v2-specific bookkeeping (informational; not required by the
        # consumers but useful for debugging + report-generation):
        "v2_dm_max_seed_bottom_subband": float(dm_max_effective),
        "v2_dm_max_final": final_dm_max,
        "v2_K_fines_per_gpu": int(k),
        "v2_n_fine_raw_before_trim": int(n_fine_raw),
        "v2_n_fine_trimmed": int(n_fine_raw - n_fine),
        "v2_coarse_seed_pos_pc_cm3": [float(x) for x in coarse_seed.tolist()],
    }

    return DmPlan(
        dm_min=float(dm_min),
        dm_max=final_dm_max,
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
        "--tol-coarse-seed",
        type=float,
        default=None,
        help=(
            "Optional tolerance used ONLY for the bottom-sub-band coarse seed "
            "(Step 1). Default: same as --tol."
        ),
    )
    p.add_argument(
        "--chan-sum-factor",
        type=int,
        default=8,
        help=(
            "Effective channel summing factor applied before fine-DM Levin "
            "spacing. 8 => dnu=0.244140625 MHz, n_chan=768."
        ),
    )
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
        chan_sum_factor=args.chan_sum_factor,
        tol_coarse_seed=args.tol_coarse_seed,
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
            "  fine_spacing      = "
            f"{plan.metadata['fine_dnu_mhz_effective']:.9f} MHz, "
            f"n_chan={plan.metadata['fine_n_chan_effective']} "
            f"(chan_sum_factor={plan.metadata['fine_chan_sum_factor']})"
        )
        print(
            f"  tol/coarse_seed_tol = {plan.tol:.3f}/"
            f"{plan.metadata['coarse_seed_tol']:.3f}"
        )
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
