"""Tests for :mod:`dsart.coinc.cube_veto` decide() + event_is_injection()."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dsart.coinc.cube_veto import (
    CubeMetrics,
    CubeVetoThresholds,
    decide,
    event_is_injection,
)


def _clean_metrics(**ov) -> CubeMetrics:
    """A metrics object that passes every rule (a real-looking burst)."""
    base = dict(
        ok=True,
        snr_c1=30.0, dm_c1=500.0, fdm_trig=4, l_pix_trig=128, m_pix_trig=128,
        t_img=100, t_apex=100, fdm_apex=4, t_shift=0, dm_shift_trials=0,
        tz_trig=20.0, tz_apex=20.0, dmz=20.0, dm_edge=0,
        imgz_apex=20.0, img_off_apex=1.0, n_fdm=8,
        g_apex_cube_same=1, g_apex_val=20.0, n_cubes=8,
    )
    base.update(ov)
    return CubeMetrics(**base)


def test_clean_burst_keeps() -> None:
    d = decide(_clean_metrics(), is_injection=False)
    assert d.keep is True
    assert d.action == "KEEP"
    assert d.rules_fired == ()


def test_injection_always_kept() -> None:
    # Even a metrics object that trips every rule is kept when injection.
    bad = _clean_metrics(
        img_off_apex=999, t_shift=999, dm_shift_trials=999,
        tz_apex=0.0, imgz_apex=0.0, dm_edge=1, g_apex_cube_same=0, tz_trig=0.0,
    )
    d = decide(bad, is_injection=True)
    assert d.keep is True
    assert d.is_injection is True
    assert d.rules_fired == ()


def test_metrics_not_ok_fail_open() -> None:
    d = decide(CubeMetrics(ok=False, reason="missing npz"), is_injection=False)
    assert d.keep is True
    assert "fail-open" in d.notes


def test_r1_image_offset() -> None:
    d = decide(_clean_metrics(img_off_apex=41.0), is_injection=False)
    assert not d.keep and "R1_image_offset" in d.rules_fired


def test_r2_time_shift() -> None:
    d = decide(_clean_metrics(t_shift=31), is_injection=False)
    assert not d.keep and "R2_time_shift" in d.rules_fired


def test_r3_dm_shift() -> None:
    d = decide(_clean_metrics(dm_shift_trials=9), is_injection=False)
    assert not d.keep and "R3_dm_shift" in d.rules_fired


def test_r4_no_peak() -> None:
    d = decide(_clean_metrics(tz_apex=3.9, imgz_apex=3.9), is_injection=False)
    assert not d.keep and "R4_no_peak" in d.rules_fired


def test_r5_dm_edge_rail() -> None:
    d = decide(
        _clean_metrics(dm_edge=1, dm_shift_trials=5), is_injection=False,
    )
    assert not d.keep and "R5_dm_edge_rail" in d.rules_fired


def test_r10_cube_unconfirmed() -> None:
    d = decide(
        _clean_metrics(g_apex_cube_same=0, tz_trig=9.9), is_injection=False,
    )
    assert not d.keep and "R10_cube_unconfirmed" in d.rules_fired


def test_thresholds_respected() -> None:
    # With a stricter R1 threshold, a previously-clean offset now trips.
    th = CubeVetoThresholds(r1_img_off_px=0.5)
    d = decide(_clean_metrics(img_off_apex=1.0), is_injection=False,
               thresholds=th)
    assert not d.keep and "R1_image_offset" in d.rules_fired


# --- event_is_injection ------------------------------------------------


def _make_event(tmp: Path, name: str, *, l3_inj=None, csv_inj_id="",
                csv_row=None) -> Path:
    ev = tmp / name
    (ev / "Level2").mkdir(parents=True)
    (ev / "Level3").mkdir(parents=True)
    doc = {"event_name": name}
    if l3_inj is not None:
        doc["injection"] = l3_inj
    (ev / "Level3" / f"{name}.json").write_text(json.dumps(doc))
    row = csv_row or {"mjd": "0", "dm_pc_cc": "0", "l_rad": "0", "m_rad": "0"}
    row = dict(row)
    row["inj_id"] = csv_inj_id
    with (ev / "Level2" / f"C1_window_{name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return ev


def test_event_is_injection_l3_marker(tmp_path: Path) -> None:
    ev = _make_event(tmp_path, "aaaa",
                     l3_inj={"is_injection": True, "inj_ids": ["x"]})
    assert event_is_injection(ev, "aaaa") is True


def test_event_is_injection_csv_inj_id(tmp_path: Path) -> None:
    ev = _make_event(tmp_path, "bbbb", csv_inj_id="probe1")
    assert event_is_injection(ev, "bbbb") is True


def test_event_not_injection_when_no_marker(tmp_path: Path) -> None:
    ev = _make_event(tmp_path, "cccc")
    assert event_is_injection(ev, "cccc") is False


def test_event_is_injection_coincidence_fallback(tmp_path: Path) -> None:
    import time

    from dsart.coinc.inject_log import FiredInjection, append_fired_injection

    now = time.time()
    log = tmp_path / "fired.jsonl"
    append_fired_injection(log, FiredInjection(
        inj_id="homxlike", dm_pc_cm3=500.0, l_rad=0.01, m_rad=0.0,
        apply_at_specnum=1, fired_at_unix=now, ttl_s=60.0,
    ))
    mjd = 40587.0 + now / 86400.0
    ev = _make_event(
        tmp_path, "dddd",
        csv_row={"mjd": f"{mjd:.11f}", "dm_pc_cc": "503.0",
                 "l_rad": "0.011", "m_rad": "0.0"},
    )
    # No L3 marker, no CSV inj_id → only the durable-log coincidence flags it.
    assert event_is_injection(ev, "dddd") is False
    assert event_is_injection(ev, "dddd", fired_log_path=log) is True
