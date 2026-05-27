"""Cluster characterisation: ClusterStats dataclass + computer.

Time-only-clustered components arrive from :mod:`dsart.coinc.components`;
this module reduces each component to the field set consumed by the
criteria evaluator and the C2 hiplot CSV.

Reference: ``docs/c1c2/C1C2_DESIGN.md`` §3.4 (trigger criteria) and
§3.6 (rolling CSVs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .window import WindowEntry

__all__ = [
    "ClusterStats",
    "compute_stats",
]


@dataclass(frozen=True, slots=True)
class ClusterStats:
    """Per-cluster summary the criteria evaluator + CSV writer consume."""

    n_events: int
    n_search_nodes: int
    n_gpu_halves: int

    snr_max: float
    snr_sum: float
    snr_mean: float

    dm_min: float
    dm_max: float
    dm_median: float
    dm_iqr: float

    l_median: float
    m_median: float
    lm_diag_rad: float

    width_min: int
    width_max: int
    width_median: float

    t_start_mjd: float
    t_end_mjd: float
    t_peak_mjd: float

    kernel_ids_distinct: Tuple[str, ...]
    peak_event_specnum: int

    # Galactic-DM discriminant (added 2026-05-27).
    #
    # ``gal_dm_max_los`` is the NE2001 disk DM at the current pointing
    # direction, served by /mon/array/gal_dm (declination.service on
    # h23). When the C2 service has a fresh value, every cluster is
    # tagged with the ratio ``dm_median / gal_dm_max_los`` as
    # ``dm_galactic_fraction``:
    #
    #   * < 0.75  → almost certainly inside the Galactic disk
    #   * >= 0.75 → almost certainly extragalactic / extra-disk
    #
    # ``dm_galactic_fraction = nan`` means we did not have a usable
    # gal_dm_max_los at evaluation time (etcd outage or fresh boot
    # before the first poll); criteria predicates that gate on this
    # field treat nan as "do not match" so existing classes that don't
    # use the discriminant are unaffected.
    gal_dm_max_los: float = float("nan")
    dm_galactic_fraction: float = float("nan")


def compute_stats(
    members: Sequence[WindowEntry],
    *,
    gal_dm_max_los: Optional[float] = None,
) -> ClusterStats:
    """Reduce a clustered component to :class:`ClusterStats`.

    ``gal_dm_max_los`` is the NE2001 max-LOS galactic DM (pc/cc) at the
    current pointing — passed in by the coincidencer service from
    ``/mon/array/gal_dm``. When provided and finite & positive, the
    cluster is tagged with ``dm_galactic_fraction = dm_median /
    gal_dm_max_los``; otherwise both fields are NaN.
    """
    if not members:
        raise ValueError("compute_stats requires at least one member")

    snrs = np.fromiter((m.snr for m in members), dtype=np.float64,
                       count=len(members))
    dms = np.fromiter((m.dm_pc_cc for m in members), dtype=np.float64,
                      count=len(members))
    widths = np.fromiter(
        (m.width_samples for m in members), dtype=np.int64,
        count=len(members),
    )
    ls = np.fromiter((m.l_rad for m in members), dtype=np.float64,
                     count=len(members))
    ms = np.fromiter((m.m_rad for m in members), dtype=np.float64,
                     count=len(members))
    mjds = np.fromiter((m.mjd for m in members), dtype=np.float64,
                       count=len(members))

    peak_idx = int(np.argmax(snrs))
    peak = members[peak_idx]

    # Quartile helpers — numpy.quantile is well-defined for small arrays.
    q1 = float(np.quantile(dms, 0.25)) if len(dms) >= 2 else float(dms[0])
    q3 = float(np.quantile(dms, 0.75)) if len(dms) >= 2 else float(dms[0])

    kernel_ids = sorted({m.kernel_id for m in members})

    dm_med = float(np.median(dms))

    if (
        gal_dm_max_los is not None
        and math.isfinite(float(gal_dm_max_los))
        and float(gal_dm_max_los) > 0.0
    ):
        gal_dm_los_f = float(gal_dm_max_los)
        dm_galactic_fraction = dm_med / gal_dm_los_f
    else:
        gal_dm_los_f = float("nan")
        dm_galactic_fraction = float("nan")

    return ClusterStats(
        n_events=len(members),
        n_search_nodes=len({m.search_node_id for m in members}),
        n_gpu_halves=len({(m.search_node_id, m.gpu_half) for m in members}),
        snr_max=float(snrs.max()),
        snr_sum=float(snrs.sum()),
        snr_mean=float(snrs.mean()),
        dm_min=float(dms.min()),
        dm_max=float(dms.max()),
        dm_median=dm_med,
        dm_iqr=q3 - q1,
        l_median=float(np.median(ls)),
        m_median=float(np.median(ms)),
        lm_diag_rad=float(
            math.hypot(ls.max() - ls.min(), ms.max() - ms.min()),
        ),
        width_min=int(widths.min()),
        width_max=int(widths.max()),
        width_median=float(np.median(widths)),
        t_start_mjd=float(mjds.min()),
        t_end_mjd=float(mjds.max()),
        t_peak_mjd=float(peak.mjd),
        kernel_ids_distinct=tuple(kernel_ids),
        peak_event_specnum=int(peak.event_specnum),
        gal_dm_max_los=gal_dm_los_f,
        dm_galactic_fraction=dm_galactic_fraction,
    )
