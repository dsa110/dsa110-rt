"""Tests for :mod:`dsart.coinc.archive` (per-event archive layout)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsart.coinc.archive import (
    C1_WINDOW_CSV_FIELDS,
    C2_CLUSTER_CSV_FIELDS,
    EVENT_SUBDIRS,
    EventArchiveWriter,
    stats_to_l3_metadata,
)
from dsart.coinc.stats import ClusterStats
from dsart.coinc.window import WindowEntry


def _entry(
    *, mjd: float = 60781.0, snr: float = 10.0, dm: float = 100.0,
    event_specnum: int = 0, search_node_id: int = 1, gpu_half: int = 0,
) -> WindowEntry:
    return WindowEntry(
        mjd=mjd,
        snr=snr,
        l_rad=1.5e-3,
        m_rad=-2.5e-3,
        l_pix=120,
        m_pix=130,
        dm_pc_cc=dm,
        dm_idx_global=10,
        fine_dm_idx=2,
        event_specnum=event_specnum,
        width_samples=4,
        kernel_id="unit:d1:b4",
        flags=0,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        cube_id=7,
        sample_period_us=1048.576,
    )


def _stats() -> ClusterStats:
    return ClusterStats(
        n_events=3,
        n_search_nodes=2,
        n_gpu_halves=3,
        snr_max=12.5,
        snr_sum=33.0,
        snr_mean=11.0,
        dm_min=99.0,
        dm_max=101.0,
        dm_median=100.0,
        dm_iqr=1.0,
        l_median=1.5e-3,
        m_median=-2.5e-3,
        lm_diag_rad=2.0e-3,
        width_min=2,
        width_max=8,
        width_median=4.0,
        t_start_mjd=60781.0,
        t_end_mjd=60781.0 + 1.0 / 86400.0,
        t_peak_mjd=60781.0 + 0.5 / 86400.0,
        kernel_ids_distinct=("unit:d1:b4", "unit:d1:b8"),
        peak_event_specnum=42,
    )


def test_archive_create_creates_all_subdirs(tmp_path: Path) -> None:
    wr = EventArchiveWriter(tmp_path / "candidates")
    ev = wr.create("260521abcd")
    assert ev == tmp_path / "candidates" / "260521abcd"
    for sub in EVENT_SUBDIRS:
        assert (ev / sub).is_dir(), f"missing {sub}"


def test_archive_create_is_idempotent(tmp_path: Path) -> None:
    wr = EventArchiveWriter(tmp_path / "candidates")
    wr.create("260521abcd")
    # Drop something into Level2 to ensure the second create() doesn't
    # blow it away.
    (tmp_path / "candidates" / "260521abcd" / "Level2" / "marker").write_text("ok")
    wr.create("260521abcd")
    assert (tmp_path / "candidates" / "260521abcd" / "Level2" / "marker").exists()


def test_archive_create_rejects_bad_name(tmp_path: Path) -> None:
    wr = EventArchiveWriter(tmp_path / "candidates")
    with pytest.raises(ValueError):
        wr.create("")
    with pytest.raises(ValueError):
        wr.create("../escape")


def test_archive_calibration_symlink(tmp_path: Path) -> None:
    cal = tmp_path / "fixture-cal"
    cal.mkdir()
    (cal / "cal.npz").write_text("blob")
    wr = EventArchiveWriter(tmp_path / "candidates", calibration_source=cal)
    ev = wr.create("260521abcd")
    cal_link = ev / "calibration"
    assert cal_link.is_symlink()
    assert (cal_link / "cal.npz").is_file()


def test_archive_writes_c2_cluster_csv(tmp_path: Path) -> None:
    wr = EventArchiveWriter(tmp_path / "candidates")
    ev = wr.create("260521abcd")
    p = wr.write_c2_cluster_csv(
        ev, "260521abcd", _stats(),
        trigger_class="bright_frb", trigger="260521abcd",
    )
    assert p == ev / "Level2" / "C2_260521abcd.csv"
    text = p.read_text()
    # Header line equals schema.
    header = text.splitlines()[0].split(",")
    assert tuple(header) == C2_CLUSTER_CSV_FIELDS
    # Single data row.
    assert len(text.splitlines()) == 2
    assert "bright_frb" in text
    assert "260521abcd" in text


def test_archive_writes_c1_window_csv(tmp_path: Path) -> None:
    wr = EventArchiveWriter(tmp_path / "candidates")
    ev = wr.create("260521abcd")
    members = [
        _entry(event_specnum=10, search_node_id=1, gpu_half=0),
        _entry(event_specnum=20, search_node_id=2, gpu_half=1),
    ]
    p = wr.write_c1_window_csv(
        ev, "260521abcd", members, trigger="260521abcd",
    )
    assert p == ev / "Level2" / "C1_window_260521abcd.csv"
    lines = p.read_text().splitlines()
    assert tuple(lines[0].split(",")) == C1_WINDOW_CSV_FIELDS
    # Header + 2 rows.
    assert len(lines) == 3
    # Each row's trigger field reads "260521abcd"
    for ln in lines[1:]:
        assert ln.split(",")[-1] == "260521abcd"


def test_archive_writes_l3_metadata(tmp_path: Path) -> None:
    wr = EventArchiveWriter(tmp_path / "candidates")
    ev = wr.create("260521abcd")
    meta = stats_to_l3_metadata(
        event_name="260521abcd",
        stats=_stats(),
        trigger_class_name="bright_frb",
        trigger_action="dump_all_gpus",
        holdoff_s=30.0,
    )
    p = wr.write_l3_metadata(ev, "260521abcd", meta)
    assert p == ev / "Level3" / "260521abcd.json"
    parsed = json.loads(p.read_text())
    assert parsed["event_name"] == "260521abcd"
    assert parsed["trigger"]["class"] == "bright_frb"
    assert parsed["c2"]["n_events"] == 3
    assert parsed["c2"]["kernel_ids_distinct"] == ["unit:d1:b4", "unit:d1:b8"]


def test_archive_l3_metadata_includes_required_fields() -> None:
    meta = stats_to_l3_metadata(
        event_name="260521abcd",
        stats=_stats(),
        trigger_class_name="bright_frb",
        trigger_action="dump_all_gpus",
        holdoff_s=30.0,
    )
    assert meta["event_name"] == "260521abcd"
    assert meta["schema_version"] == 1
    assert "trigger" in meta
    assert "c2" in meta
    assert {"snr_max", "dm_median", "t_peak_mjd"} <= set(meta["c2"].keys())


# ---------------------------------------------------------------------------
# Galactic-DM discriminant CSV / L3 fields (added 2026-05-27)
# ---------------------------------------------------------------------------


def test_archive_c2_csv_includes_gal_dm_columns(tmp_path: Path) -> None:
    """CSV header has the two new columns and they survive a write."""
    from dsart.coinc.stats import compute_stats
    from dsart.coinc.window import WindowEntry
    wr = EventArchiveWriter(tmp_path / "candidates")
    ev = wr.create("260521abcd")
    e = WindowEntry(
        mjd=60781.0, snr=10.0, l_rad=0.0, m_rad=0.0,
        l_pix=0, m_pix=0, dm_pc_cc=30.0, dm_idx_global=0, fine_dm_idx=0,
        event_specnum=1, width_samples=4, kernel_id="unit:d1:b4",
        flags=0, search_node_id=1, gpu_half=0, cube_id=0,
        sample_period_us=1048.576,
    )
    stats = compute_stats([e], gal_dm_max_los=100.0)
    wr.write_c2_cluster_csv(
        ev, "260521abcd", stats, trigger_class="bright_galactic",
        trigger="260521abcd",
    )
    text = (ev / "Level2" / "C2_260521abcd.csv").read_text()
    header = text.splitlines()[0].split(",")
    assert "gal_dm_max_los_pc_cc" in header
    assert "dm_galactic_fraction" in header
    row = dict(zip(header, text.splitlines()[1].split(",")))
    assert row["gal_dm_max_los_pc_cc"] == "100.000000"
    # 30 / 100 = 0.3
    assert row["dm_galactic_fraction"] == "0.300000"


def test_archive_c2_csv_writes_empty_for_nan_gal_dm(tmp_path: Path) -> None:
    """NaN gal_dm renders as empty cell, not the literal "nan"."""
    wr = EventArchiveWriter(tmp_path / "candidates")
    ev = wr.create("260521abcd")
    wr.write_c2_cluster_csv(
        ev, "260521abcd", _stats(), trigger_class="log_only",
        trigger="260521abcd",
    )
    text = (ev / "Level2" / "C2_260521abcd.csv").read_text()
    header = text.splitlines()[0].split(",")
    row = dict(zip(header, text.splitlines()[1].split(",")))
    assert row["gal_dm_max_los_pc_cc"] == ""
    assert row["dm_galactic_fraction"] == ""


def test_archive_l3_metadata_includes_gal_dm_fields() -> None:
    """L3 JSON exposes the gal_dm fields; None when NaN."""
    from dsart.coinc.stats import compute_stats
    from dsart.coinc.window import WindowEntry
    e = WindowEntry(
        mjd=60781.0, snr=10.0, l_rad=0.0, m_rad=0.0,
        l_pix=0, m_pix=0, dm_pc_cc=30.0, dm_idx_global=0, fine_dm_idx=0,
        event_specnum=1, width_samples=4, kernel_id="unit:d1:b4",
        flags=0, search_node_id=1, gpu_half=0, cube_id=0,
        sample_period_us=1048.576,
    )
    # With gal_dm populated.
    meta = stats_to_l3_metadata(
        event_name="260521abcd",
        stats=compute_stats([e], gal_dm_max_los=100.0),
        trigger_class_name="bright_galactic",
        trigger_action="dump_all_gpus",
        holdoff_s=30.0,
    )
    assert meta["c2"]["gal_dm_max_los_pc_cc"] == 100.0
    assert meta["c2"]["dm_galactic_fraction"] == 0.30

    # NaN gal_dm → None.
    meta_nan = stats_to_l3_metadata(
        event_name="260521abcd",
        stats=compute_stats([e]),  # no gal_dm
        trigger_class_name="log_only",
        trigger_action="log_only",
        holdoff_s=0.0,
    )
    assert meta_nan["c2"]["gal_dm_max_los_pc_cc"] is None
    assert meta_nan["c2"]["dm_galactic_fraction"] is None
