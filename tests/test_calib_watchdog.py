"""Decision logic of the calibration pre-processor watchdog."""

from __future__ import annotations

import calendar
import time
from dataclasses import replace
from pathlib import Path

from dsart.services.calib_watchdog import (
    Observation, State, WatchdogConfig, decide, fname_time, newest_pulled)

CFG = WatchdogConfig()
SET = 306.0  # correlator file cadence, s


def t(s: str) -> float:
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S")))


def announce(ts: float, host: str = "lxd110h22") -> dict:
    name = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))
    return {"cmd": "rsync",
            "val": {"hostname": host, "filename": f"/home/ubuntu/data/{name}_sb15.hdf5"}}


NOW = t("2026-10-08T21:00:00")


def obs(**kw) -> Observation:
    """Healthy, observing: the latest announced set has been pulled."""
    base = dict(
        now=NOW, cmd=announce(NOW - 1.5 * SET), newest_pulled=NOW - 1.5 * SET,
        preprocess_state="active", preprocess_heartbeat=NOW - 30,
        children_alive={"rsync": 1, "gather": 1, "assess": 1},
        calibration_state="active", calibration_heartbeat=NOW - 30,
        newest_ms=NOW - 3600)
    base.update(kw)
    return Observation(**base)


def stalled(now: float, frozen: float) -> Observation:
    """Nodes still announcing, nothing pulled since ``frozen``."""
    return obs(now=now, cmd=announce(now - 1.5 * SET), newest_pulled=frozen,
               preprocess_heartbeat=now - 30, calibration_heartbeat=now - 30,
               newest_ms=now - 3600)


def test_fname_time():
    assert fname_time("/x/2026-10-09T05:32:02_sb06.hdf5") == t("2026-10-09T05:32:02")
    assert fname_time("tmp") is None
    assert fname_time("notes_2026-10-09T05:32:02_sb06.hdf5") is None


def test_newest_pulled(tmp_path: Path):
    for n in ("2026-10-09T05:32:02_sb06.hdf5", "2026-10-10T20:06:10_sb00.hdf5", "tmp"):
        (tmp_path / n).write_bytes(b"")
    assert newest_pulled(tmp_path) == t("2026-10-10T20:06:10")


def test_healthy_observing_does_nothing():
    st = State()
    dec = decide(obs(), st, CFG)
    assert not dec.restart and not dec.alerts and dec.status == "ok"


def test_dead_watch_restarts_on_second_stalled_run_only():
    st = State()
    frozen = NOW - 3 * SET
    # Lag at the first run is 1.5 sets (< stall): nothing yet.
    assert decide(stalled(NOW, frozen), st, CFG).status == "ok"
    d1 = decide(stalled(NOW + 300, frozen), st, CFG)   # lag ~2.5 sets
    assert d1.status == "stall suspected" and not d1.restart
    d2 = decide(stalled(NOW + 600, frozen), st, CFG)
    assert d2.restart and st.restarts == 1
    assert "etcd watch has died" in d2.alerts[0]
    assert st.incident_from == frozen


def test_one_set_behind_is_not_a_stall():
    # Checked between a set's first announcement and its pull.
    st = State()
    for k in range(5):
        now = NOW + 300 * k
        o = obs(now=now, cmd=announce(now - SET), newest_pulled=now - 2 * SET,
                preprocess_heartbeat=now - 30, calibration_heartbeat=now - 30,
                newest_ms=now - 3600)
        assert not decide(o, st, CFG).restart


def test_not_observing_never_stalls():
    st = State()
    # Backfill / stopped array: the latest announcement is a day old.
    for k in range(4):
        now = NOW + 300 * k
        o = obs(now=now, cmd=announce(NOW - 86400), newest_pulled=NOW - 2 * 86400,
                preprocess_heartbeat=now - 30, calibration_heartbeat=now - 30)
        dec = decide(o, st, CFG)
        assert not dec.restart and dec.status == "idle (not observing)"


def test_calibrate_command_latest_gives_no_verdict():
    st = State(stalled_runs=1)
    o = obs(cmd={"cmd": "calibrate", "val": {"calname": "0204+152", "flist": []}})
    dec = decide(o, st, CFG)
    assert not dec.restart and st.stalled_runs == 1


def test_cooldown_then_second_restart_then_give_up_then_recover():
    st = State()
    frozen = NOW - 10 * SET
    now = NOW
    restarts = []
    for _ in range(40):  # 200 min of a stall a restart cannot fix
        dec = decide(stalled(now, frozen), st, CFG)
        if dec.restart:
            restarts.append(now)
        now += 300
    assert len(restarts) == CFG.max_restarts
    assert all(b - a >= CFG.cooldown_s for a, b in zip(restarts, restarts[1:]))
    assert st.gave_up
    # The give-up alert went out exactly once.
    st2 = replace(st)
    assert not decide(stalled(now, frozen), st2, CFG).alerts
    # Pulls resume: one recovery alert naming the deaf window, counters reset.
    dec = decide(obs(now=now, cmd=announce(now - SET), newest_pulled=now - SET,
                     preprocess_heartbeat=now - 30, calibration_heartbeat=now - 30,
                     newest_ms=now - 3600), st, CFG)
    assert len(dec.alerts) == 1 and "recovered" in dec.alerts[0]
    assert st.restarts == 0 and not st.gave_up


def test_failed_unit_restarts_immediately():
    st = State()
    dec = decide(obs(preprocess_state="failed"), st, CFG)
    assert dec.restart and "has failed" in dec.alerts[0]


def test_manually_stopped_unit_is_left_alone_and_alerted_once():
    st = State()
    d1 = decide(obs(preprocess_state="inactive"), st, CFG)
    d2 = decide(obs(preprocess_state="inactive"), st, CFG)
    assert not d1.restart and not d2.restart
    assert len(d1.alerts) == 1 and not d2.alerts


def test_stale_heartbeat_restarts():
    dec = decide(obs(preprocess_heartbeat=NOW - 900), State(), CFG)
    assert dec.restart and "heartbeat is stale" in dec.alerts[0]


def test_dead_child_restarts():
    dec = decide(obs(children_alive={"rsync": 0, "gather": 1, "assess": 1}), State(), CFG)
    assert dec.restart and "rsync" in dec.alerts[0]


def test_calibration_consumer_down_alerts_once_never_restarts():
    st = State()
    d1 = decide(obs(calibration_state="inactive"), st, CFG)
    d2 = decide(obs(calibration_state="inactive"), st, CFG)
    d3 = decide(obs(), st, CFG)
    assert not (d1.restart or d2.restart or d3.restart)
    assert len(d1.alerts) == 1 and not d2.alerts
    assert len(d3.alerts) == 1 and "running again" in d3.alerts[0]


def test_stale_ms_alerts_once_while_observing_only():
    st = State()
    old = NOW - 30 * 3600
    assert not decide(obs(newest_ms=old, cmd=announce(NOW - 86400)), st, CFG).alerts
    d1 = decide(obs(newest_ms=old), st, CFG)
    d2 = decide(obs(newest_ms=old), st, CFG)
    assert len(d1.alerts) == 1 and "measurement set" in d1.alerts[0] and not d2.alerts
    d3 = decide(obs(newest_ms=NOW - 60), st, CFG)
    assert len(d3.alerts) == 1 and "appeared" in d3.alerts[0]
