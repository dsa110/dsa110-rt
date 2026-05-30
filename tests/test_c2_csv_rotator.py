"""Tests for :mod:`dsart.coinc.csv_rotator` (hourly rotation +
retention enforcement)."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dsart.coinc.csv_rotator import (
    RollingCsvWriter,
    concat_recent_hourly,
    rotation_key,
)


def _write_hourly(dir_path: Path, prefix: str, key: str,
                  rows: list[str]) -> Path:
    p = dir_path / f"{prefix}_{key}.csv"
    with p.open("w", encoding="utf-8") as f:
        f.write("a,b\n")
        for r in rows:
            f.write(r + "\n")
    return p


def test_concat_recent_hourly_concatenates_window(tmp_path: Path) -> None:
    now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    # Three consecutive hours present; one older hour outside the window.
    _write_hourly(tmp_path, "c2", "20260521_10", ["1,x"])
    _write_hourly(tmp_path, "c2", "20260521_11", ["2,y"])
    _write_hourly(tmp_path, "c2", "20260521_12", ["3,z"])
    _write_hourly(tmp_path, "c2", "20260521_08", ["9,old"])  # out of 3h window
    out = concat_recent_hourly(
        tmp_path, "c2", now_utc=now, window_hours=3,
        out_name="c2_last24h.csv",
    )
    assert out == tmp_path / "c2_last24h.csv"
    lines = out.read_text().splitlines()
    assert lines[0] == "a,b"            # single header
    assert lines.count("a,b") == 1      # no repeated headers
    assert lines[1:] == ["1,x", "2,y", "3,z"]  # oldest->newest, no old hour


def test_concat_recent_hourly_skips_missing_hours(tmp_path: Path) -> None:
    now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    _write_hourly(tmp_path, "c2", "20260521_12", ["3,z"])  # only newest hour
    out = concat_recent_hourly(
        tmp_path, "c2", now_utc=now, window_hours=24,
        out_name="c2_last24h.csv",
    )
    assert out is not None
    assert out.read_text().splitlines()[1:] == ["3,z"]


def test_concat_recent_hourly_no_inputs_returns_none(tmp_path: Path) -> None:
    now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    assert concat_recent_hourly(
        tmp_path, "c2", now_utc=now, window_hours=24,
        out_name="c2_last24h.csv",
    ) is None


def test_concat_recent_hourly_excludes_rolling_file_itself(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    _write_hourly(tmp_path, "c2", "20260521_12", ["3,z"])
    # A stale rolling file must never be folded back into itself.
    (tmp_path / "c2_last24h.csv").write_text("a,b\n999,stale\n")
    out = concat_recent_hourly(
        tmp_path, "c2", now_utc=now, window_hours=24,
        out_name="c2_last24h.csv",
    )
    assert out is not None
    assert "999,stale" not in out.read_text()


def test_rotation_key_format() -> None:
    t = datetime(2026, 5, 21, 18, 35, 7, tzinfo=timezone.utc)
    assert rotation_key(t) == "20260521_18"


def test_rotation_key_rejects_naive() -> None:
    with pytest.raises(ValueError):
        rotation_key(datetime(2026, 5, 21, 18, 35, 7))


def test_rotation_key_rejects_non_utc() -> None:
    other = timezone(timedelta(hours=2))
    with pytest.raises(ValueError):
        rotation_key(datetime(2026, 5, 21, 18, 35, 7, tzinfo=other))


def test_writer_creates_file_with_header(tmp_path: Path) -> None:
    wr = RollingCsvWriter(
        tmp_path, "c1", ("a", "b", "c"), retention_hours=48,
    )
    t = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    wr.append_row({"a": 1, "b": "x", "c": 3.14}, now_utc=t)
    expected = tmp_path / "c1_20260521_12.csv"
    assert expected.exists()
    text = expected.read_text()
    assert text.splitlines()[0] == "a,b,c"
    assert "1,x,3.14" in text


def test_writer_appends_multiple_rows_same_hour(tmp_path: Path) -> None:
    wr = RollingCsvWriter(tmp_path, "c1", ("a", "b"))
    t = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(3):
        wr.append_row({"a": i, "b": i * 2}, now_utc=t + timedelta(minutes=i))
    lines = (tmp_path / "c1_20260521_12.csv").read_text().splitlines()
    assert lines[0] == "a,b"
    assert lines[1:] == ["0,0", "1,2", "2,4"]


def test_writer_rotates_on_hour_boundary(tmp_path: Path) -> None:
    wr = RollingCsvWriter(tmp_path, "c1", ("a",))
    t12 = datetime(2026, 5, 21, 12, 59, 59, tzinfo=timezone.utc)
    t13 = datetime(2026, 5, 21, 13, 0, 0, tzinfo=timezone.utc)
    wr.append_row({"a": "x"}, now_utc=t12)
    wr.append_row({"a": "y"}, now_utc=t13)
    assert (tmp_path / "c1_20260521_12.csv").exists()
    assert (tmp_path / "c1_20260521_13.csv").exists()
    # Each contains a header + exactly one row
    assert (tmp_path / "c1_20260521_12.csv").read_text().splitlines() == [
        "a", "x",
    ]
    assert (tmp_path / "c1_20260521_13.csv").read_text().splitlines() == [
        "a", "y",
    ]


def test_writer_respects_existing_file_no_double_header(tmp_path: Path) -> None:
    # Pre-create a file with content (e.g. from a previous process run).
    pre = tmp_path / "c1_20260521_12.csv"
    pre.write_text("a\n1\n2\n")
    wr = RollingCsvWriter(tmp_path, "c1", ("a",))
    t = datetime(2026, 5, 21, 12, 30, 0, tzinfo=timezone.utc)
    wr.append_row({"a": 3}, now_utc=t)
    text = pre.read_text()
    # Header still appears exactly once
    assert text.count("a\n") == 1
    assert text.splitlines() == ["a", "1", "2", "3"]


def test_writer_extra_keys_in_row_are_ignored(tmp_path: Path) -> None:
    wr = RollingCsvWriter(tmp_path, "c1", ("a", "b"))
    t = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    wr.append_row({"a": 1, "b": 2, "c": 99}, now_utc=t)
    lines = (tmp_path / "c1_20260521_12.csv").read_text().splitlines()
    assert lines == ["a,b", "1,2"]


def test_housekeep_deletes_old_files(tmp_path: Path) -> None:
    wr = RollingCsvWriter(
        tmp_path, "c1", ("a",), retention_hours=2,
    )
    now = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)
    # Three files: one fresh, one just inside retention, one expired.
    fresh = tmp_path / "c1_20260521_17.csv"
    just_in = tmp_path / "c1_20260521_16.csv"
    expired = tmp_path / "c1_20260521_10.csv"
    for p in (fresh, just_in, expired):
        p.write_text("a\n1\n")
    os.utime(fresh, (now.timestamp() - 60, now.timestamp() - 60))
    os.utime(just_in, (now.timestamp() - 60 * 60, now.timestamp() - 60 * 60))
    os.utime(expired, (
        now.timestamp() - 3 * 60 * 60, now.timestamp() - 3 * 60 * 60,
    ))
    removed = wr.housekeep(now)
    assert removed == 1
    assert fresh.exists()
    assert just_in.exists()
    assert not expired.exists()


def test_housekeep_ignores_files_outside_prefix(tmp_path: Path) -> None:
    wr = RollingCsvWriter(
        tmp_path, "c1", ("a",), retention_hours=1,
    )
    # An old non-prefix file should NOT be touched even though it would
    # otherwise be old enough.
    foreign = tmp_path / "c2_20260521_10.csv"
    foreign.write_text("ignore me\n")
    old = time.time() - 3 * 60 * 60
    os.utime(foreign, (old, old))
    now = datetime.now(timezone.utc)
    wr.housekeep(now)
    assert foreign.exists()


def test_housekeep_requires_aware_datetime(tmp_path: Path) -> None:
    wr = RollingCsvWriter(tmp_path, "c1", ("a",), retention_hours=1)
    with pytest.raises(ValueError):
        wr.housekeep(datetime(2026, 5, 21, 12, 0, 0))


def test_writer_rejects_bad_init(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RollingCsvWriter(tmp_path, "", ("a",))
    with pytest.raises(ValueError):
        RollingCsvWriter(tmp_path, "c1", ("a",), retention_hours=0)
