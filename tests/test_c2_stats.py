"""Tests for :mod:`dsart.coinc.stats` (cluster characterisation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsart.coinc.stats import ClusterStats, compute_stats
from dsart.coinc.window import WindowEntry


def _e(
    *,
    snr: float,
    dm: float = 100.0,
    l_rad: float = 0.0,
    m_rad: float = 0.0,
    width_samples: int = 4,
    mjd: float = 60781.0,
    kernel_id: str = "unit:d1:b4",
    search_node_id: int = 1,
    gpu_half: int = 0,
    event_specnum: int = 0,
) -> WindowEntry:
    return WindowEntry(
        mjd=mjd,
        snr=snr,
        l_rad=l_rad,
        m_rad=m_rad,
        l_pix=0,
        m_pix=0,
        dm_pc_cc=dm,
        dm_idx_global=0,
        fine_dm_idx=0,
        event_specnum=event_specnum,
        width_samples=width_samples,
        kernel_id=kernel_id,
        flags=0,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        cube_id=0,
        sample_period_us=1048.576,
    )


def test_compute_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_stats([])


def test_compute_stats_singleton() -> None:
    e = _e(snr=10.0, dm=100.0, width_samples=4, event_specnum=42)
    s = compute_stats([e])
    assert s.n_events == 1
    assert s.n_search_nodes == 1
    assert s.n_gpu_halves == 1
    assert s.snr_max == 10.0
    assert s.snr_mean == 10.0
    assert s.snr_sum == 10.0
    assert s.dm_min == 100.0 == s.dm_max == s.dm_median
    assert s.dm_iqr == 0.0
    assert s.lm_diag_rad == 0.0
    assert s.width_median == 4
    assert s.peak_event_specnum == 42
    assert s.kernel_ids_distinct == ("unit:d1:b4",)


def test_compute_stats_multi_search_nodes_and_halves() -> None:
    es = [
        _e(snr=10.0, search_node_id=1, gpu_half=0),
        _e(snr=11.0, search_node_id=1, gpu_half=1),
        _e(snr=12.0, search_node_id=2, gpu_half=0),
        _e(snr=13.0, search_node_id=2, gpu_half=0),
    ]
    s = compute_stats(es)
    assert s.n_events == 4
    assert s.n_search_nodes == 2
    # (s, g) pairs: (1,0), (1,1), (2,0) → 3
    assert s.n_gpu_halves == 3
    assert s.snr_max == 13.0
    assert s.snr_sum == 46.0
    assert s.snr_mean == pytest.approx(46.0 / 4)


def test_compute_stats_dm_and_widths_stats() -> None:
    dms = [50.0, 100.0, 150.0, 200.0]
    widths = [2, 4, 8, 16]
    es = [
        _e(snr=10.0, dm=dm, width_samples=w, event_specnum=i)
        for i, (dm, w) in enumerate(zip(dms, widths))
    ]
    s = compute_stats(es)
    assert s.dm_min == 50.0
    assert s.dm_max == 200.0
    assert s.dm_median == pytest.approx(125.0)
    # IQR = Q3 - Q1
    q1, q3 = float(np.quantile(dms, 0.25)), float(np.quantile(dms, 0.75))
    assert s.dm_iqr == pytest.approx(q3 - q1)
    assert s.width_min == 2
    assert s.width_max == 16
    assert s.width_median == pytest.approx(float(np.median(widths)))


def test_compute_stats_lm_diag_rad_from_bbox() -> None:
    es = [
        _e(snr=10.0, l_rad=0.0, m_rad=0.0),
        _e(snr=10.0, l_rad=3.0e-3, m_rad=4.0e-3),  # bbox: 3e-3, 4e-3
    ]
    s = compute_stats(es)
    assert s.lm_diag_rad == pytest.approx(math.hypot(3.0e-3, 4.0e-3))


def test_compute_stats_peak_picks_max_snr_entry() -> None:
    es = [
        _e(snr=8.0, event_specnum=10, mjd=60781.0),
        _e(snr=15.0, event_specnum=20, mjd=60781.5),
        _e(snr=9.0, event_specnum=30, mjd=60782.0),
    ]
    s = compute_stats(es)
    assert s.snr_max == 15.0
    assert s.peak_event_specnum == 20
    assert s.t_peak_mjd == pytest.approx(60781.5)
    assert s.t_start_mjd == pytest.approx(60781.0)
    assert s.t_end_mjd == pytest.approx(60782.0)


def test_compute_stats_kernel_ids_distinct_sorted() -> None:
    es = [
        _e(snr=10.0, kernel_id="unit:d1:b4"),
        _e(snr=10.0, kernel_id="unit:d1:b1"),
        _e(snr=10.0, kernel_id="unit:d1:b4"),
        _e(snr=10.0, kernel_id="unit:d1:b2"),
    ]
    s = compute_stats(es)
    assert s.kernel_ids_distinct == ("unit:d1:b1", "unit:d1:b2", "unit:d1:b4")
