"""Tests for :mod:`dsart.coinc.criteria` (YAML-driven trigger
evaluator with hot-reload + per-class holdoff)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import pytest

from dsart.coinc.criteria import (
    BadCriteriaFile,
    CriteriaEvaluator,
    TriggerClass,
)
from dsart.coinc.stats import ClusterStats


def _stats(**overrides) -> ClusterStats:
    base = dict(
        n_events=5,
        n_search_nodes=2,
        n_gpu_halves=3,
        snr_max=15.0,
        snr_sum=60.0,
        snr_mean=12.0,
        dm_min=300.0,
        dm_max=400.0,
        dm_median=350.0,
        dm_iqr=10.0,
        l_median=0.0,
        m_median=0.0,
        lm_diag_rad=1.0e-3,
        width_min=2,
        width_max=8,
        width_median=4.0,
        t_start_mjd=60781.0,
        t_end_mjd=60781.0,
        t_peak_mjd=60781.0,
        kernel_ids_distinct=("unit:d1:b4",),
        peak_event_specnum=100,
    )
    base.update(overrides)
    return ClusterStats(**base)


def _write(path: Path, body: str) -> None:
    path.write_text(body)


def test_criteria_load_default_starter(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: bright_frb
    require:
      snr_max_min: 12.0
      n_events_min: 3
    action: dump_all_gpus
    holdoff_s: 30.0
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    ev = CriteriaEvaluator(p)
    names = [c.name for c in ev.classes]
    assert names == ["bright_frb", "log_only"]


def test_criteria_unknown_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: bad
    require:
      not_a_real_key: 5.0
    action: log_only
""")
    with pytest.raises(BadCriteriaFile, match="unknown require keys"):
        CriteriaEvaluator(p)


def test_criteria_bad_yaml_shapes(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, "trigger_classes: notalist\n")
    with pytest.raises(BadCriteriaFile, match="must be a list"):
        CriteriaEvaluator(p)
    _write(p, "{}\n")
    with pytest.raises(BadCriteriaFile, match="trigger_classes"):
        CriteriaEvaluator(p)
    _write(p, """
trigger_classes:
  - require:
      n_events_min: 1
    action: log_only
""")
    with pytest.raises(BadCriteriaFile, match="name"):
        CriteriaEvaluator(p)


def test_criteria_duplicate_name_raises(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: dup
    require:
      n_events_min: 1
    action: log_only
  - name: dup
    require:
      n_events_min: 1
    action: log_only
""")
    with pytest.raises(BadCriteriaFile, match="duplicate"):
        CriteriaEvaluator(p)


def test_criteria_first_match_wins(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: bright_frb
    require:
      snr_max_min: 12.0
      n_events_min: 3
    action: dump_all_gpus
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    ev = CriteriaEvaluator(p)
    matched = ev.evaluate(_stats(snr_max=15.0, n_events=5))
    assert matched is not None
    assert matched.name == "bright_frb"
    # Weaker cluster picks up log_only
    matched = ev.evaluate(_stats(snr_max=8.0, n_events=2))
    assert matched is not None
    assert matched.name == "log_only"


def test_criteria_no_match_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: bright_frb
    require:
      snr_max_min: 100.0
    action: dump_all_gpus
""")
    ev = CriteriaEvaluator(p)
    assert ev.evaluate(_stats(snr_max=15.0)) is None


def test_criteria_holdoff_blocks_repeat_fires(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: bright_frb
    require:
      snr_max_min: 12.0
    action: dump_all_gpus
    holdoff_s: 10.0
""")
    fake_now = {"t": 100.0}

    def now():
        return fake_now["t"]

    ev = CriteriaEvaluator(p, now=now)
    s = _stats(snr_max=15.0)
    assert ev.evaluate(s) is not None
    # Same cluster a moment later is suppressed by holdoff
    fake_now["t"] = 105.0
    assert ev.evaluate(s) is None
    # 10.0001 s after first fire — clears holdoff
    fake_now["t"] = 110.0001
    assert ev.evaluate(s) is not None


def test_criteria_holdoff_falls_through_to_next_class(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: bright_frb
    require:
      snr_max_min: 12.0
    action: dump_all_gpus
    holdoff_s: 10.0
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    fake_now = {"t": 100.0}
    ev = CriteriaEvaluator(p, now=lambda: fake_now["t"])
    s = _stats(snr_max=15.0)
    assert ev.evaluate(s).name == "bright_frb"
    fake_now["t"] = 102.0
    # bright_frb is held off, but the second class (log_only) still matches.
    matched = ev.evaluate(s)
    assert matched is not None
    assert matched.name == "log_only"


def test_criteria_dm_window_min_max(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: dm_window
    require:
      dm_median_min_pc_cc: 100.0
      dm_median_max_pc_cc: 1000.0
    action: dump_all_gpus
""")
    ev = CriteriaEvaluator(p)
    assert ev.evaluate(_stats(dm_median=50.0)) is None
    assert ev.evaluate(_stats(dm_median=500.0)) is not None
    assert ev.evaluate(_stats(dm_median=1500.0)) is None


def test_criteria_lm_diag_width_iqr_thresholds(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: tight
    require:
      lm_diag_max_rad: 5.0e-3
      width_median_max_samples: 16
      dm_iqr_max_pc_cc: 5.0
    action: dump_all_gpus
""")
    ev = CriteriaEvaluator(p)
    # Loose
    assert ev.evaluate(_stats(lm_diag_rad=10e-3)) is None
    # Wide pulse
    assert ev.evaluate(_stats(width_median=32.0)) is None
    # High IQR
    assert ev.evaluate(_stats(dm_iqr=10.0)) is None
    # Within all thresholds
    assert ev.evaluate(_stats(lm_diag_rad=1e-3, width_median=4.0,
                              dm_iqr=2.0)) is not None


def test_criteria_hot_reload_picks_up_changes(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: v1
    require:
      n_events_min: 1
    action: log_only
""")
    ev = CriteriaEvaluator(p)
    assert [c.name for c in ev.classes] == ["v1"]
    # Mutate the file with a different mtime (st_mtime resolution can
    # be coarse; bump it explicitly).
    import os
    old = p.stat().st_mtime
    _write(p, """
trigger_classes:
  - name: v2
    require:
      n_events_min: 1
    action: log_only
""")
    os.utime(p, (old + 5.0, old + 5.0))
    assert ev.reload_if_changed() is True
    assert [c.name for c in ev.classes] == ["v2"]
    # second call sees no further changes
    assert ev.reload_if_changed() is False


def test_criteria_starter_yaml_loads(tmp_path: Path) -> None:
    """The shipped starter YAML must parse cleanly."""
    starter = Path(__file__).resolve().parents[1] / "configs" / "c2_trigger_criteria.yaml"
    ev = CriteriaEvaluator(starter)
    names = [c.name for c in ev.classes]
    assert "bright_frb_extragalactic" in names
    assert "bright_galactic" in names
    assert "log_only" in names


# ---------------------------------------------------------------------------
# Galactic-DM discriminant (added 2026-05-27)
# ---------------------------------------------------------------------------


def test_criteria_dm_galactic_fraction_max_gates_galactic(
    tmp_path: Path,
) -> None:
    """``dm_galactic_fraction_max`` accepts clusters with DM/gal_dm < t."""
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: gal
    require:
      dm_galactic_fraction_max: 0.75
    action: dump_all_gpus
""")
    ev = CriteriaEvaluator(p)
    assert ev.evaluate(_stats(dm_galactic_fraction=0.30)) is not None
    assert ev.evaluate(_stats(dm_galactic_fraction=0.74)) is not None
    assert ev.evaluate(_stats(dm_galactic_fraction=0.76)) is None
    assert ev.evaluate(_stats(dm_galactic_fraction=15.0)) is None


def test_criteria_dm_galactic_fraction_min_gates_extragalactic(
    tmp_path: Path,
) -> None:
    """``dm_galactic_fraction_min`` accepts clusters with DM/gal_dm >= t."""
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: egal
    require:
      dm_galactic_fraction_min: 0.75
    action: dump_all_gpus
""")
    ev = CriteriaEvaluator(p)
    assert ev.evaluate(_stats(dm_galactic_fraction=0.74)) is None
    assert ev.evaluate(_stats(dm_galactic_fraction=0.75)) is not None
    assert ev.evaluate(_stats(dm_galactic_fraction=15.0)) is not None


def test_criteria_dm_galactic_fraction_nan_does_not_match(
    tmp_path: Path,
) -> None:
    """A cluster with no gal_dm (NaN fraction) never matches either gate.

    This protects existing behaviour: on cold boot before the first
    successful /mon/array/gal_dm poll, classes that gate on the
    discriminant silently fall through to log_only.
    """
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: gal
    require:
      dm_galactic_fraction_max: 0.75
    action: dump_all_gpus
  - name: egal
    require:
      dm_galactic_fraction_min: 0.75
    action: dump_all_gpus
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    ev = CriteriaEvaluator(p)
    # default _stats() has dm_galactic_fraction=nan (frozen-default)
    s = _stats()
    matched = ev.evaluate(s)
    assert matched is not None
    assert matched.name == "log_only"


def test_criteria_two_class_galactic_split_yaml_loads(
    tmp_path: Path,
) -> None:
    """End-to-end: gal/egal first-match split routes by DM/gal_dm."""
    p = tmp_path / "c.yaml"
    _write(p, """
trigger_classes:
  - name: egal
    require:
      snr_max_min: 8.0
      dm_galactic_fraction_min: 0.75
    action: dump_all_gpus
  - name: gal
    require:
      snr_max_min: 12.0
      dm_galactic_fraction_max: 0.75
    action: dump_all_gpus
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    ev = CriteriaEvaluator(p)
    # SNR=10, frac=10.0 (DM=1000/gal_dm=100) — passes egal (SNR>=8) so
    # routes to egal.
    m1 = ev.evaluate(_stats(snr_max=10.0, dm_galactic_fraction=10.0))
    assert m1 is not None
    assert m1.name == "egal"
    # SNR=10, frac=0.3 — egal predicate fails (frac<0.75), gal fails
    # SNR>=12 — falls to log_only.
    m2 = ev.evaluate(_stats(snr_max=10.0, dm_galactic_fraction=0.30))
    assert m2 is not None
    assert m2.name == "log_only"
    # SNR=15, frac=0.3 — gal passes, routes to gal.
    m3 = ev.evaluate(_stats(snr_max=15.0, dm_galactic_fraction=0.30))
    assert m3 is not None
    assert m3.name == "gal"
