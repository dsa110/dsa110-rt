"""DSA-110 fast-vis sparse uv-grid pipeline (M3 chunk 3a).

Public API for the corr-side fast-vis gridder + sparsity-pattern
modules. Consumed by:

* :mod:`dsart.services.corr_fast_compute` (M3 chunk 2b) — production
  service shell that wires the gridder into the streaming pipeline.
* ``dsart-search-rx`` (M5) — read-only import of
  :class:`SparsityPattern` + :func:`build_pattern` +
  :func:`predict_pattern_id` for the search-side ``cmd: prepare``
  pattern rebuild (plan §4.3 Option C; see ``PARALLEL_AGENTS.md`` §3
  for the M3 / M5 ownership split).

References
==========

* Plan §3 lines 305-309 — sparsity pattern + ``pattern_id`` semantics.
* Plan §4.2 lines 1283-1346 — streaming pipeline placement.
* :mod:`dsart.grid.sparsity_pattern` — Class C; corr ↔ search shared.
* :mod:`dsart.grid.kernel` — Class A; corr-side production gridder.
* :mod:`dsart.grid.pol_sum` — Class A; re-export of
  :func:`dsart.services.corr_fast_kernel.stokes_i_pol_sum`.
"""

from dsart.grid.kernel import FastVisGridder
from dsart.grid.pol_sum import stokes_i_pol_sum
from dsart.grid.sparsity_pattern import (
    CORE_RADIUS_M_DEFAULT,
    N_CORE_DEFAULT,
    SUPPORTED_KERNEL_SUPPORTS,
    SparsityPattern,
    build_pattern,
    compute_antpos_hash,
    compute_chgroup_auto_cell_lambda,
    compute_chgroup_table_hash,
    compute_top_of_band_cell_lambda,
    core_baseline_mask_from_antpos,
    gaussian_kernel_weights,
    predict_pattern_id,
    quantise_dec_deg,
)

__all__ = [
    "CORE_RADIUS_M_DEFAULT",
    "FastVisGridder",
    "N_CORE_DEFAULT",
    "SUPPORTED_KERNEL_SUPPORTS",
    "SparsityPattern",
    "build_pattern",
    "compute_antpos_hash",
    "compute_chgroup_auto_cell_lambda",
    "compute_chgroup_table_hash",
    "compute_top_of_band_cell_lambda",
    "core_baseline_mask_from_antpos",
    "gaussian_kernel_weights",
    "predict_pattern_id",
    "quantise_dec_deg",
    "stokes_i_pol_sum",
]
