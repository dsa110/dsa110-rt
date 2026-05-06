"""DSA-110 fast-vis coarse-DM dedispersion + stage-2 FIFO (M3 chunk 3b).

Public API for the corr-side coarse-DM modules. Consumed by:

* :mod:`dsart.services.corr_fast_compute` (M3 chunk 4 integration) —
  wires the dedisperser + FIFO into the streaming pipeline.
* :mod:`dsart.fine_dm.combiner` (M5 search-side, **read-only**) —
  imports :class:`DMPlan` for cross-coarse-DM combining and the
  same delay-table conventions for fine-DM residuals.

References
==========

* Plan §3.2 — DM plan (coarse + fine).
* Plan §3.6.2 — DEDISP architecture (stage-1 / stage-2).
* Plan §4.2 lines 1283-1346 — streaming pipeline placement.
* :mod:`dsart.coarse_dm.dm_plan` — :class:`DMPlan` slim view.
* :mod:`dsart.coarse_dm.dedisp` — image-cube dedispersion primitive.
* :mod:`dsart.coarse_dm.stage2_fifo` — bounded cube FIFO container.
"""

from dsart.coarse_dm.dedisp import coarse_dedisp, max_output_t_dedisp
from dsart.coarse_dm.dm_plan import (
    DMPlan,
    build_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
    load_dm_plan,
)
from dsart.coarse_dm.stage2_fifo import Stage2FIFO

__all__ = [
    "DMPlan",
    "Stage2FIFO",
    "build_chgroup_freq_table_GHz",
    "coarse_dedisp",
    "compute_delay_native_samples_table",
    "load_dm_plan",
    "max_output_t_dedisp",
]
