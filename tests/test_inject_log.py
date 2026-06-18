"""Tests for :mod:`dsart.coinc.inject_log` (durable fired-injection log)."""

from __future__ import annotations

import time
from pathlib import Path

from dsart.coinc.inject_log import (
    FiredInjection,
    append_fired_injection,
    event_coincident_inj_id,
    load_fired_injections,
)


def _inj(inj_id="p1", dm=500.0, l=0.0, m=0.0, fired=None, apply_at=1):
    return FiredInjection(
        inj_id=inj_id, dm_pc_cm3=dm, l_rad=l, m_rad=m,
        apply_at_specnum=apply_at,
        fired_at_unix=fired if fired is not None else time.time(),
        ttl_s=60.0,
    )


def test_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "f.jsonl"
    assert append_fired_injection(log, _inj(inj_id="abc", dm=321.0))
    out = load_fired_injections(log)
    assert len(out) == 1
    assert out[0].inj_id == "abc"
    assert out[0].dm_pc_cm3 == 321.0


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_fired_injections(tmp_path / "nope.jsonl") == []


def test_dedupe_keeps_latest(tmp_path: Path) -> None:
    log = tmp_path / "f.jsonl"
    t0 = time.time()
    append_fired_injection(log, _inj(inj_id="x", fired=t0, apply_at=0))
    append_fired_injection(log, _inj(inj_id="x", fired=t0 + 5, apply_at=999))
    out = load_fired_injections(log)
    assert len(out) == 1
    assert out[0].apply_at_specnum == 999


def test_bad_lines_skipped(tmp_path: Path) -> None:
    log = tmp_path / "f.jsonl"
    append_fired_injection(log, _inj(inj_id="ok"))
    with log.open("a") as fh:
        fh.write("not json\n")
        fh.write('{"inj_id": ""}\n')  # invalid (empty id)
        fh.write("\n")                  # blank
    out = load_fired_injections(log)
    assert [o.inj_id for o in out] == ["ok"]


def test_since_unix_filter(tmp_path: Path) -> None:
    log = tmp_path / "f.jsonl"
    append_fired_injection(log, _inj(inj_id="old", fired=1000.0))
    append_fired_injection(log, _inj(inj_id="new", fired=9_000_000_000.0))
    out = load_fired_injections(log, since_unix=2000.0)
    assert [o.inj_id for o in out] == ["new"]


def test_coincidence_dm_lm_time() -> None:
    now = 1_700_000_000.0
    fired = [_inj(inj_id="hit", dm=500.0, l=0.01, m=0.0, fired=now)]
    mjd = 40587.0 + now / 86400.0
    assert event_coincident_inj_id(
        fired, mjd=mjd, dm_pc_cc=505.0, l_rad=0.012, m_rad=0.001,
    ) == "hit"
    # DM too far
    assert event_coincident_inj_id(
        fired, mjd=mjd, dm_pc_cc=700.0, l_rad=0.01, m_rad=0.0,
    ) is None
    # sky too far
    assert event_coincident_inj_id(
        fired, mjd=mjd, dm_pc_cc=500.0, l_rad=0.5, m_rad=0.5,
    ) is None
    # time too far (well outside +120 s window)
    assert event_coincident_inj_id(
        fired, mjd=40587.0 + (now + 9999) / 86400.0,
        dm_pc_cc=500.0, l_rad=0.01, m_rad=0.0,
    ) is None


def test_coincidence_no_mjd_skips_time_gate() -> None:
    fired = [_inj(inj_id="hit", dm=500.0, l=0.0, m=0.0, fired=1.0)]
    # mjd<=0 → time gate skipped, DM+sky still required.
    assert event_coincident_inj_id(
        fired, mjd=0.0, dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0,
    ) == "hit"


def test_coincidence_prefers_closest_sky() -> None:
    now = 1_700_000_000.0
    mjd = 40587.0 + now / 86400.0
    fired = [
        _inj(inj_id="far", dm=500.0, l=0.04, m=0.0, fired=now),
        _inj(inj_id="near", dm=500.0, l=0.001, m=0.0, fired=now),
    ]
    assert event_coincident_inj_id(
        fired, mjd=mjd, dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0,
    ) == "near"
