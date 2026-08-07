"""Unit tests for the hourly injection sentinel.

All collaborators are faked (etcd store, dashboard HTTP, Slack
notifier) — no network, no observatory env. Covers: health-gated
skips, K freshness (age, missing, dec-change) recalibration triggers,
target-SNR bounds, the fire path and per-injection Slack messages,
loss-stage attribution (search/C1 vs C2 vs C3), the guard-rejected
classification, summary stats, the once-per-day summary gate, and a
render smoke test for the summary PNG. A final drift check pins the
constants mirrored from the dashboard's inject_calibration module.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DSART_SRC = os.path.join(REPO_ROOT, "src")
if DSART_SRC not in sys.path:
    sys.path.insert(0, DSART_SRC)

from dsart.services import inject_sentinel as isent  # noqa: E402
from dsart.services.inject_sentinel import (  # noqa: E402
    InjectSentinel,
    Outcome,
    SentinelConfig,
    bucket_key,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStore:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})

    def get_dict(self, key):
        return self.docs.get(key)


class FakeNotifier:
    def __init__(self):
        self.texts = []
        self.files = []
        self._ts = 0

    def post_text(self, text, thread_ts=None, reply_broadcast=False):
        self._ts += 1
        self.texts.append({"text": text, "thread_ts": thread_ts})
        return {"ok": True, "ts": f"ts{self._ts}"}

    def post_file(self, path, title=None, initial_comment=None,
                  thread_ts=None):
        self.files.append({
            "path": str(path), "title": title,
            "initial_comment": initial_comment,
        })
        return {"ok": True, "ts": "fts1"}


def healthy_docs(now):
    docs = {}
    for cg in range(16):
        docs[f"/mon/corr_rt/{cg}/corr_fast"] = {"ts_wall_unix": now}
    for sid, g in isent.SEARCH_HALVES:
        docs[f"/mon/search_rt/{sid}/compute/{g}"] = {
            "ts_wall_unix": now, "c1_metering_active": 0,
        }
    docs["/mon/array/dec"] = {"dec_deg": 71.63}
    return docs


def make_config(tmp_path, **over):
    base = dict(
        enabled=True,
        channel="CTEST",
        results_path=str(tmp_path / "results.jsonl"),
        state_path=str(tmp_path / "state.json"),
        summary_dir=str(tmp_path / "plots"),
        candidates_root=str(tmp_path / "candidates"),
        recovery_timeout_s=2.0,
        recovery_poll_s=0.01,
    )
    base.update(over)
    return SentinelConfig.from_dict(base)


class FakeDash:
    """Fake dashboard: GET /control/inject_calibrations + POSTs."""

    def __init__(self, now, k_entries=None):
        self.now = now
        self.k_entries = k_entries if k_entries is not None else {}
        self.calibrate_calls = []
        self.inject_calls = []
        self.inject_response = None  # None -> default ok

    def get(self, url):
        assert url.endswith("/control/inject_calibrations")
        return 200, {"ok": True, "entries": list(self.k_entries.values())}

    def post(self, url, fields):
        if url.endswith("/control/inject_calibrate"):
            self.calibrate_calls.append(dict(fields))
            dm = float(fields["dm_pc_cm3"])
            bucket = bucket_key(dm)
            self.k_entries[bucket] = {
                "bucket": bucket, "K": 170000.0,
                "last_calibrated_at_unix": self.now,
            }
            return 200, {"ok": True, "K": 170000.0, "bucket": bucket}
        if url.endswith("/control/inject"):
            self.inject_calls.append(dict(fields))
            if self.inject_response is not None:
                return self.inject_response
            return 200, {
                "ok": True,
                "val": {"fluence_jy_ms": 2.5e-4,
                        "inj_id": fields["inj_id"]},
            }
        raise AssertionError(f"unexpected POST {url}")


def fresh_entry(now, dm, k=170000.0, age_s=0.0):
    b = bucket_key(dm)
    return b, {
        "bucket": b, "K": k, "last_calibrated_at_unix": now - age_s,
    }


def make_sentinel(cfg, store, dash, notifier=None, now=None, dm=1000.0):
    rng = random.Random(42)
    t0 = now if now is not None else time.time()
    clock = {"t": t0}

    def time_fn():
        clock["t"] += 0.05
        return clock["t"]

    s = InjectSentinel(
        cfg,
        store=store,
        notifier=notifier or FakeNotifier(),
        http_post_form=dash.post,
        http_get=dash.get,
        rng=rng,
        time_fn=time_fn,
    )
    return s, clock


def add_match_doc(store, inj_id, *, snr=19.0, dm=1000.0, l=0.011, m=-0.004):
    store.docs[isent.MATCH_EVENT_PREFIX + inj_id] = {
        "inj_id": inj_id,
        "best": {
            "observed_snr": snr,
            "observed_dm_pc_cm3": dm + 3.0,
            "observed_l_rad": l + 1e-4,
            "observed_m_rad": m,
            "observed_search_node_id": 9,
            "observed_gpu_half": 1,
        },
        "n_matches": 4,
        "active": {
            "dm_pc_cm3": dm, "l_rad": l, "m_rad": m, "target_snr": 20.0,
        },
    }


def add_event(tmp_path, cfg, name, inj_id, *, keep=True, ncubes=8,
              c3_written=True):
    ev = tmp_path / "candidates" / name
    (ev / "Level3").mkdir(parents=True)
    (ev / "Level3" / f"{name}.json").write_text(json.dumps({
        "injection": {"is_injection": True, "inj_ids": [inj_id]},
    }))
    cubes = ev / "cubes"
    cubes.mkdir()
    for i in range(ncubes):
        (cubes / f"cube_s1_g0_{i}.npz").write_bytes(b"x")
    if c3_written:
        (ev / "C3_decision.json").write_text(json.dumps({"keep": keep}))
    return ev


# ---------------------------------------------------------------------------
# Health gate
# ---------------------------------------------------------------------------


def test_unhealthy_fleet_skips_cycle_and_throttles_warning(tmp_path):
    now = time.time()
    docs = healthy_docs(now)
    docs["/mon/corr_rt/7/corr_fast"] = {"ts_wall_unix": now - 3600}
    cfg = make_config(tmp_path)
    dash = FakeDash(now)
    notifier = FakeNotifier()
    s, _ = make_sentinel(cfg, FakeStore(docs), dash, notifier, now=now)

    rec = s.run_cycle()
    assert rec["outcome"] == Outcome.SKIPPED_UNHEALTHY
    assert "chgroups=7" in rec["reason"]
    assert dash.inject_calls == []
    assert len(notifier.texts) == 1  # warn posted once

    rec2 = s.run_cycle()
    assert rec2["outcome"] == Outcome.SKIPPED_UNHEALTHY
    assert len(notifier.texts) == 1  # throttled

    # Both attempts landed in the JSONL.
    rows = s.load_results_since(0)
    assert len(rows) == 2


def test_metering_active_counts_as_unhealthy(tmp_path):
    now = time.time()
    docs = healthy_docs(now)
    docs["/mon/search_rt/1/compute/0"] = {
        "ts_wall_unix": now, "c1_metering_active": 1,
    }
    cfg = make_config(tmp_path)
    s, _ = make_sentinel(cfg, FakeStore(docs), FakeDash(now), now=now)
    ok, reason = s.check_health()
    assert not ok and "s1g0" in reason


# ---------------------------------------------------------------------------
# K freshness
# ---------------------------------------------------------------------------


def test_missing_k_triggers_recalibration(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    dash = FakeDash(now)  # no entries
    s, _ = make_sentinel(cfg, FakeStore(healthy_docs(now)), dash, now=now)
    usable, info = s.ensure_k_fresh(1000.0)
    assert usable
    assert info["recalibrated"]
    assert "no_k" in info["recal_reasons"]
    assert len(dash.calibrate_calls) == 1
    assert float(dash.calibrate_calls[0]["dm_pc_cm3"]) == 1000.0


def test_stale_k_triggers_recalibration(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    b, entry = fresh_entry(now, 500.0, age_s=90000.0)  # > 24 h
    dash = FakeDash(now, {b: entry})
    s, _ = make_sentinel(cfg, FakeStore(healthy_docs(now)), dash, now=now)
    usable, info = s.ensure_k_fresh(500.0)
    assert usable
    assert info["recalibrated"]
    assert any(r.startswith("stale") for r in info["recal_reasons"])


def test_fresh_k_is_not_recalibrated_and_dec_baseline_adopted(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    b, entry = fresh_entry(now, 1500.0, age_s=600.0)
    dash = FakeDash(now, {b: entry})
    s, _ = make_sentinel(cfg, FakeStore(healthy_docs(now)), dash, now=now)
    usable, info = s.ensure_k_fresh(1500.0)
    assert usable and not info["recalibrated"]
    assert dash.calibrate_calls == []
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["k_calibration_dec"][b] == pytest.approx(71.63)


def test_dec_change_triggers_recalibration(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    b, entry = fresh_entry(now, 2000.0, age_s=600.0)
    dash = FakeDash(now, {b: entry})
    # K was measured at dec 16.27 per the sentinel's own provenance.
    (tmp_path / "state.json").write_text(json.dumps({
        "k_calibration_dec": {b: 16.27},
    }))
    s, _ = make_sentinel(cfg, FakeStore(healthy_docs(now)), dash, now=now)
    usable, info = s.ensure_k_fresh(2000.0)
    assert usable and info["recalibrated"]
    assert any(r.startswith("dec_changed") for r in info["recal_reasons"])
    # Provenance updated to the current dec.
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["k_calibration_dec"][b] == pytest.approx(71.63)


def test_recal_failure_yields_fire_failed_cycle(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)

    class DeadDash(FakeDash):
        def post(self, url, fields):
            if url.endswith("/control/inject_calibrate"):
                self.calibrate_calls.append(dict(fields))
                return 500, {"ok": False, "error": "boom"}
            return super().post(url, fields)

    dash = DeadDash(now)
    s, _ = make_sentinel(cfg, FakeStore(healthy_docs(now)), dash, now=now)
    rec = s.run_cycle()
    assert rec["outcome"] == Outcome.FIRE_FAILED
    assert dash.inject_calls == []


# ---------------------------------------------------------------------------
# Parameter picking
# ---------------------------------------------------------------------------


def test_picked_params_respect_bounds(tmp_path):
    cfg = make_config(tmp_path)
    s = InjectSentinel(cfg, store=FakeStore(), notifier=FakeNotifier(),
                       http_post_form=lambda *a: (200, {"ok": True}),
                       http_get=lambda *a: (200, {"ok": True, "entries": []}),
                       rng=random.Random(7))
    for _ in range(200):
        p = s._pick_params()
        assert p["dm_pc_cm3"] in cfg.dm_choices
        assert cfg.target_snr_min <= p["target_snr"] <= cfg.target_snr_max
        assert abs(p["l_rad"]) <= cfg.lm_max_rad
        assert abs(p["m_rad"]) <= cfg.lm_max_rad
        assert p["width_samples"] == 4


# ---------------------------------------------------------------------------
# Full cycles: recovery + attribution
# ---------------------------------------------------------------------------


def _run_recovered_cycle(tmp_path, *, keep=True, c3_written=True):
    now = time.time()
    cfg = make_config(tmp_path)
    dash = FakeDash(now)
    for dm in cfg.dm_choices:
        bb, ee = fresh_entry(now, dm, age_s=100.0)
        dash.k_entries[bb] = ee
    store = FakeStore(healthy_docs(now))
    notifier = FakeNotifier()
    s, clock = make_sentinel(cfg, store, dash, notifier, now=now)

    fired = {}
    orig_post = dash.post

    def post_and_land(url, fields):
        status, doc = orig_post(url, fields)
        if url.endswith("/control/inject") and doc.get("ok"):
            inj_id = fields["inj_id"]
            dm = float(fields["dm_pc_cm3"])
            add_match_doc(store, inj_id, dm=dm,
                          l=float(fields["l_rad"]),
                          m=float(fields["m_rad"]))
            add_event(tmp_path, cfg, "260807test", inj_id,
                      keep=keep, c3_written=c3_written)
            # Event dir mtime must be >= fired_at - 60 (it is: just made).
            fired["inj_id"] = inj_id
        return status, doc

    s._http_post_form = post_and_land
    rec = s.run_cycle()
    return rec, notifier, dash, fired


def test_recovered_cycle_end_to_end(tmp_path):
    rec, notifier, dash, fired = _run_recovered_cycle(tmp_path)
    assert rec["outcome"] == Outcome.RECOVERED
    assert rec["event"] == "260807test"
    assert rec["cubes"] == 8
    assert rec["c3_decision"] == "KEEP"
    assert rec["inj_id"] == fired["inj_id"]
    assert rec["inj_id"].startswith("inj_")
    assert len(rec["inj_id"]) <= 14  # short id: inj_MMDD_HHMM
    # Recovered message links the event to its dashboard burst page.
    assert "/bursts/260807test|260807test>" in notifier.texts[1]["text"]
    assert rec["snr_ratio"] == pytest.approx(19.0 / 20.0)
    assert rec["offset_arcsec"] == pytest.approx(
        1e-4 * 180 / math.pi * 3600, rel=1e-6)
    # allow_bright must never be sent.
    assert all("allow_bright" not in c for c in dash.inject_calls)
    # Two Slack texts: sent + threaded recovered.
    assert len(notifier.texts) == 2
    assert notifier.texts[0]["thread_ts"] is None
    assert notifier.texts[1]["thread_ts"] == "ts1"
    assert "injection sent" in notifier.texts[0]["text"]
    body = notifier.texts[1]["text"]
    assert "recovered" in body and "260807test" in body
    assert "(l, m)" in body and "DM" in body and "SNR" in body


def test_c3_reject_is_missed_c3(tmp_path):
    rec, notifier, _, _ = _run_recovered_cycle(tmp_path, keep=False)
    assert rec["outcome"] == Outcome.MISSED_C3
    assert "ANOMALY" in notifier.texts[1]["text"]


def test_c3_pending_still_counts_recovered(tmp_path):
    rec, notifier, _, _ = _run_recovered_cycle(tmp_path, c3_written=False)
    assert rec["outcome"] == Outcome.RECOVERED
    assert "pending" in notifier.texts[1]["text"]


def test_no_match_doc_is_missed_search_or_c1(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    dash = FakeDash(now)
    for dm in cfg.dm_choices:
        b, e = fresh_entry(now, dm, age_s=100.0)
        dash.k_entries[b] = e
    notifier = FakeNotifier()
    s, _ = make_sentinel(
        cfg, FakeStore(healthy_docs(now)), dash, notifier, now=now)
    rec = s.run_cycle()
    assert rec["outcome"] == Outcome.MISSED_SEARCH_OR_C1
    assert "NOT recovered" in notifier.texts[1]["text"]
    assert "search/C1" in notifier.texts[1]["text"]


def test_match_without_event_is_missed_c2(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    dash = FakeDash(now)
    for dm in cfg.dm_choices:
        b, e = fresh_entry(now, dm, age_s=100.0)
        dash.k_entries[b] = e
    store = FakeStore(healthy_docs(now))
    notifier = FakeNotifier()
    s, _ = make_sentinel(cfg, store, dash, notifier, now=now)

    orig_post = dash.post

    def post_and_match_only(url, fields):
        status, doc = orig_post(url, fields)
        if url.endswith("/control/inject") and doc.get("ok"):
            add_match_doc(store, fields["inj_id"],
                          dm=float(fields["dm_pc_cm3"]))
        return status, doc

    s._http_post_form = post_and_match_only
    rec = s.run_cycle()
    assert rec["outcome"] == Outcome.MISSED_C2
    assert rec["observed_snr"] == pytest.approx(19.0)
    assert "lost at C2" in notifier.texts[1]["text"]


def test_guard_rejection_is_classified(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    dash = FakeDash(now)
    for dm in cfg.dm_choices:
        b, e = fresh_entry(now, dm, age_s=100.0)
        dash.k_entries[b] = e
    dash.inject_response = (400, {
        "ok": False,
        "error": "predicted SNR 41 above the imager-safe ceiling",
    })
    s, _ = make_sentinel(cfg, FakeStore(healthy_docs(now)), dash, now=now)
    rec = s.run_cycle()
    assert rec["outcome"] == Outcome.GUARD_REJECTED


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _rows_for_summary(now):
    return [
        {"ts_unix": now - 1000 * i, "outcome": o, "dm_pc_cm3": dm,
         "target_snr": 20.0, "observed_snr": 19.0, "snr_ratio": 0.95,
         "offset_arcsec": 15.0, "delta_dm_pc_cm3": 2.0}
        for i, (o, dm) in enumerate([
            (Outcome.RECOVERED, 500.0),
            (Outcome.RECOVERED, 1000.0),
            (Outcome.MISSED_SEARCH_OR_C1, 1500.0),
            (Outcome.MISSED_C2, 2000.0),
            (Outcome.SKIPPED_UNHEALTHY, None),
        ])
    ]


def test_summary_stats_counts_and_attribution(tmp_path):
    now = time.time()
    cfg = make_config(tmp_path)
    s = InjectSentinel(cfg, store=FakeStore(), notifier=FakeNotifier(),
                       http_post_form=lambda *a: (200, {}),
                       http_get=lambda *a: (200, {}))
    stats = s.compute_summary_stats(_rows_for_summary(now))
    assert stats["injected"] == 4
    assert stats["recovered"] == 2
    assert stats["by_outcome"][Outcome.MISSED_C2] == 1
    assert stats["by_dm"]["500"] == {"injected": 1, "recovered": 1}
    assert stats["by_dm"]["1500"] == {"injected": 1, "recovered": 0}
    text = s._summary_text(stats)
    assert "4 injected, 2 recovered" in text
    assert "missed_search_or_c1" in text.replace(" ", "_") or "missed" in text


def test_summary_plot_renders(tmp_path):
    pytest.importorskip("matplotlib")
    now = time.time()
    cfg = make_config(tmp_path)
    s = InjectSentinel(cfg, store=FakeStore(), notifier=FakeNotifier(),
                       http_post_form=lambda *a: (200, {}),
                       http_get=lambda *a: (200, {}))
    out = tmp_path / "plots" / "summary.png"
    assert s.render_summary_plot(_rows_for_summary(now), out)
    assert out.stat().st_size > 10_000


def test_daily_summary_once_per_day_gate(tmp_path):
    pytest.importorskip("matplotlib")
    now = time.time()
    cfg = make_config(tmp_path, summary_hour_utc=0)  # always past the hour
    notifier = FakeNotifier()
    s, _ = make_sentinel(
        cfg, FakeStore(healthy_docs(now)), FakeDash(now), notifier, now=now)
    for r in _rows_for_summary(now):
        s._append_result(r)
    assert s.summary_due()
    result = s.post_daily_summary()
    assert result["ok"]
    assert len(notifier.files) == 1
    assert "injected" in notifier.files[0]["initial_comment"]
    assert not s.summary_due()  # stamped for today


# ---------------------------------------------------------------------------
# Drift checks against the dashboard module (source-level, no import).
# ---------------------------------------------------------------------------


def test_mirrored_constants_match_dashboard_source():
    src = open(os.path.join(
        REPO_ROOT, "tools", "dashboard", "dsa_monitor",
        "inject_calibration.py")).read()
    assert f'MATCH_EVENT_PREFIX: str = "{isent.MATCH_EVENT_PREFIX}"' in src
    assert f"DM_BUCKET_PC_CC: float = {isent.DM_BUCKET_PC_CC:.1f}" in src
    assert (
        f"DEFAULT_CALIBRATION_WIDTH: int = "
        f"{isent.CALIBRATION_WIDTH_SAMPLES}" in src
    )
    assert f"DEFAULT_CORR_FAST_MAX_AGE_S" in src
    # The sentinel's ceiling assumptions: guard constant present and the
    # config default target range sits below it.
    assert "IMAGER_SAFE_OBSERVED_SNR: float = 30.0" in src
    assert SentinelConfig().target_snr_max < 30.0


def test_bucket_key_matches_dashboard_examples():
    assert bucket_key(500.0) == "dm0500"
    assert bucket_key(1000.0) == "dm1000"
    assert bucket_key(1500.0) == "dm1500"
    assert bucket_key(2000.0) == "dm2000"
    assert bucket_key(1024.0) == "dm1000"
