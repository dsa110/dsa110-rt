"""Synthetic-but-shape-valid DMPlan builder for throughput tests + smokes.

Use cases
=========

* ``corr_fast_integration --n-coarse-dm N --dm-max M`` (service-shell smoke,
  e.g. the M7.1 fast-path realtime soak).
* ``bench/fast_path_throughput.py`` (chunk-9 throughput bench).
* Anywhere we want to exercise the F25 multi-DM-trial Stage1 +
  gridder + dedispersion paths without committing to a real M1-derived
  DMPlan ``.npz`` blob.

What "synthetic" means here
===========================

The plan is *shape-valid* (passes :func:`build_context`'s plumbing
checks: ``chan_sum_factor`` match, ``t_int_fast_native`` match,
delay-table shapes match ``[N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse]``
etc.) but the per-channel / per-chgroup time-shift tables are all
zeros. That is correct for throughput / cadence stress tests where the
input is junkdb-generated Gaussian noise — the dedispersion math runs
through every code path but doesn't recover any real signal. For
correctness tests + the M3 captured-burst recovery suite, build a
plan via the M1 production pipeline (``dsart.coarse_dm.dm_plan``)
instead.
"""

from __future__ import annotations

import numpy as np

from dsart.coarse_dm.dm_plan import DMPlan
from dsart.common.constants import (
    BW_PROC_MHZ,
    DM_PLAN_METADATA_VERSION,
    N_CHAN_PROC_NATIVE,
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    NCHAN_PER_CHGROUP,
    NU_BOT_PROC_GHZ,
    NU_TOP_PROC_GHZ,
)
from dsart.common.contracts import DmPlan


def build_synthetic_summed_plan(
    *,
    n_coarse: int,
    dm_max: float,
    chan_sum_factor: int,
    t_int_fast_us: float,
) -> DMPlan:
    """Build a degenerate-but-shape-valid summed-channel :class:`DMPlan`.

    Parameters
    ----------
    n_coarse : int
        Number of coarse-DM trials. Linearly spaced from 0 to ``dm_max``.
    dm_max : float
        Maximum coarse DM in pc/cc.
    chan_sum_factor : int
        F33 pre-dedispersion channel-sum factor. Must divide
        :data:`NCHAN_PER_CHGROUP` (= 384).
    t_int_fast_us : float
        Fast-corr post-integration cadence (µs). Must match the
        :class:`FastIntegrationConfig.t_int_fast_native` *
        :data:`NATIVE_SAMPLE_US` of the integration cfg, or
        :func:`build_context` will raise.

    Returns
    -------
    DMPlan
        Shape-valid plan; all time-shift tables are zero (no real
        dedispersion).
    """
    n_fine = max(8, 2 * int(n_coarse))
    coarse = np.linspace(0.0, float(dm_max), int(n_coarse), dtype=np.float64)
    fine = np.linspace(coarse[0], coarse[-1], n_fine, dtype=np.float64)
    fine_offsets_idx = np.linspace(
        0, n_fine, num=int(n_coarse) + 1, dtype=np.int32,
    )
    canonical = DmPlan(
        dm_min=float(coarse[0]),
        dm_max=float(coarse[-1]) + 1.0,
        tol=1.5,
        fine_dm=fine,
        coarse_dm=coarse,
        fine_to_coarse=np.zeros(n_fine, dtype=np.int32),
        fine_offsets_idx=fine_offsets_idx,
        fine_offsets_flat=np.zeros(n_fine, dtype=np.float64),
        time_shift_corr_stage1=np.zeros(
            (N_CHGROUP, NCHAN_PER_CHGROUP, int(n_coarse)), dtype=np.int32,
        ),
        time_shift_corr_stage2=np.zeros(
            (N_CHGROUP, int(n_coarse)), dtype=np.int32,
        ),
        time_shift_search=np.zeros((n_fine, N_CHGROUP), dtype=np.int32),
        dm_idx_range_canonical=np.zeros((N_SEARCH, 2), dtype=np.int32),
        dm_idx_range_consumed=np.zeros((N_SEARCH, 2), dtype=np.int32),
        dm_idx_range_canonical_per_gpu=np.zeros(
            (N_SEARCH, N_SEARCH_GPU, 2), dtype=np.int32,
        ),
        dm_idx_range_consumed_per_gpu=np.zeros(
            (N_SEARCH, N_SEARCH_GPU, 2), dtype=np.int32,
        ),
        dm_overlap_coarse=2,
        metadata={
            "band_top_GHz": NU_TOP_PROC_GHZ,
            "band_bot_GHz": NU_BOT_PROC_GHZ,
            "BW_MHz": BW_PROC_MHZ,
            "N_chan_proc_native": N_CHAN_PROC_NATIVE,
            "t_int_fast_us": float(t_int_fast_us),
            "t_int_search_us": 524.288,
            "tol": 1.5,
            "build_utc_ns": 1_872_345_677_000_000_000,
            "git_sha": "synthetic-plan",
            "version": DM_PLAN_METADATA_VERSION,
        },
    )
    return DMPlan.from_summed_canonical(
        canonical, chan_sum_factor=int(chan_sum_factor),
    )
