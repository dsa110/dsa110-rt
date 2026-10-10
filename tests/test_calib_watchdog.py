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
RUN = 300.0  # timer period, s


def t(s: str) -> float:
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S")))


def announce(ts: float, host: str = "lxd110h22") -> dict:
    name = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))
    return {"cmd": "rsync",
            "val": {"hostname": host, "filename": f"/home/ubuntu/data/{name}_sb15.hdf5"}}


NOW = t("2026-10-08T21:00:00")


def obs(now: float = NOW, **kw) -> Observation:
    """Healthy and observing at ``now``: the latest announced set is on h23."""
    base = dict(
        now=now, cmd=announce(now - 1.5 * SET), newest_pulled=now - 1.5 * SET,
        last_arrival=now - 20,
        preprocess_state="active", preprocess_heartbeat=now - 30,
        children_alive={"rsync": 1, "gather": 1, "assess": 1},
        calibration_state="active", calibration_heartbeat=now - 30,
        newest_ms=now - 3600)
    base.update(kw)
    return Observation(**base)


def deaf(now: float, frozen: float) -> Observation:
    """Nodes still announcing; nothing pulled (or arrived) since ``frozen``."""
    return obs(now, newest_pulled=frozen, last_arrival=frozen + SET)


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
    for k in range(6):
        dec = decide(obs(NOW + RUN * k), st, CFG)
        assert not dec.restart and not dec.alerts and dec.status == "ok"
    assert st.stalled_runs == 0


def test_dead_watch_restarts_on_second_unpulled_announcement():
    st = State()
    frozen = NOW - 3 * SET
    d0 = decide(deaf(NOW, frozen), st, CFG)          # remembers the announcement
    assert d0.status == "ok" and st.pending_t is not None
    d1 = decide(deaf(NOW + RUN, frozen), st, CFG)    # it was not pulled
    assert d1.status == "stall suspected" and not d1.restart
    d2 = decide(deaf(NOW + 2 * RUN, frozen), st, CFG)
    assert d2.restart and st.restarts == 1
    assert "etcd watch has died" in d2.alerts[0]
    assert st.incident_from == frozen
    assert st.pending_t is None  # pre-restart announcements are not held against it


def test_gap_in_file_production_is_not_a_stall():
    # Regression, 2026-10-10 20:47-20:57 UTC: the corr nodes wrote 20:06:10,
    # then 20:29:01 (announced ~20:52), then 20:52:26 -- no files in between.
    # Comparing filename times read the gaps as 23 min unpulled and restarted
    # the pre-processor, which dropped two queued calibrator triggers.
    st = State()
    runs = [
        # the backfill's last announcement, days old: not observing
        obs(t("2026-10-10T20:47:10"), cmd=announce(t("2026-10-09T12:08:54")),
            newest_pulled=t("2026-10-10T20:06:10"), last_arrival=t("2026-10-10T20:47:05")),
        obs(t("2026-10-10T20:52:20"), cmd=announce(t("2026-10-10T20:29:01")),
            newest_pulled=t("2026-10-10T20:06:10"), last_arrival=t("2026-10-10T20:52:15")),
        obs(t("2026-10-10T20:57:37"), cmd=announce(t("2026-10-10T20:52:26")),
            newest_pulled=t("2026-10-10T20:29:02"), last_arrival=t("2026-10-10T20:53:30")),
        obs(t("2026-10-10T21:02:40"), cmd=announce(t("2026-10-10T20:57:35")),
            newest_pulled=t("2026-10-10T20:57:35"), last_arrival=t("2026-10-10T21:02:39")),
    ]
    for o in runs:
        dec = decide(o, st, CFG)
        assert not dec.restart and dec.status != "stall suspected", dec
    assert st.stalled_runs == 0


def test_announcement_just_before_the_run_is_not_judged_yet():
    st = State()
    decide(obs(NOW), st, CFG)
    # A manual run 60 s later must not judge (or replace) the pending set.
    pending = st.pending_t
    o = obs(NOW + 60, cmd=announce(NOW + 60 - SET), newest_pulled=NOW - 3 * SET,
            last_arrival=NOW - 600)
    assert decide(o, st, CFG).status == "ok" and st.pending_t == pending


def test_one_node_failing_its_pulls_is_not_a_stall():
    # h22's DNS record vanished: its file of every set stays on the node, but
    # the other 15 subbands of the same set arrive.
    st = State()
    for k in range(5):
        now = NOW + RUN * k
        a = now - 1.5 * SET
        dec = decide(obs(now, cmd=announce(a, "lxd110h22"), newest_pulled=a - 1,
                         last_arrival=now - 300), st, CFG)
        assert not dec.restart and dec.status == "ok"


def test_backlog_with_files_arriving_is_not_a_stall():
    # A long backfill queued ahead of the live files: the announced live set is
    # not on h23 yet, but files keep arriving.
    st = State()
    frozen = NOW - 3 * SET
    for k in range(5):
        now = NOW + RUN * k
        dec = decide(obs(now, newest_pulled=frozen, last_arrival=now - 2), st, CFG)
        assert not dec.restart and dec.status == "ok"


def test_not_observing_never_stalls():
    st = State()
    for k in range(4):
        now = NOW + RUN * k
        dec = decide(obs(now, cmd=announce(NOW - 86400), newest_pulled=NOW - 2 * 86400,
                         last_arrival=NOW - 2 * 86400), st, CFG)
        assert not dec.restart and dec.status == "idle (not observing)"
    assert st.pending_t is None


def test_calibrate_command_latest_keeps_the_pending_set():
    st = State()
    decide(obs(NOW), st, CFG)
    pending = st.pending_t
    o = obs(NOW + RUN, cmd={"cmd": "calibrate", "val": {"calname": "0204+152", "flist": []}})
    assert not decide(o, st, CFG).restart and st.pending_t == pending


def test_cooldown_then_second_restart_then_give_up_then_recover():
    st = State()
    frozen = NOW - 10 * SET
    now = NOW
    restarts = []
    for _ in range(40):  # 200 min of a stall a restart cannot fix
        dec = decide(deaf(now, frozen), st, CFG)
        if dec.restart:
            restarts.append(now)
        now += RUN
    assert len(restarts) == CFG.max_restarts
    assert all(b - a >= CFG.cooldown_s for a, b in zip(restarts, restarts[1:]))
    assert st.gave_up
    # The give-up alert went out exactly once.
    assert not decide(deaf(now, frozen), replace(st), CFG).alerts
    # Pulls resume (fixed by hand): one recovery alert, counters reset.
    dec = decide(obs(now), st, CFG)
    assert len(dec.alerts) == 1 and "recovered" in dec.alerts[0]
    assert st.restarts == 0 and not st.gave_up


def test_restart_that_works_reports_recovery():
    st = State()
    frozen = NOW - 3 * SET
    for k in range(3):
        decide(deaf(NOW + RUN * k, frozen), st, CFG)
    assert st.restarts == 1
    # After the restart the new announcements are pulled again.
    d1 = decide(obs(NOW + 3 * RUN), st, CFG)
    d2 = decide(obs(NOW + 4 * RUN), st, CFG)
    assert not d1.alerts and len(d2.alerts) == 1 and "recovered" in d2.alerts[0]


def test_failed_unit_restarts_immediately():
    dec = decide(obs(preprocess_state="failed"), State(), CFG)
    assert dec.restart and "has failed" in dec.alerts[0]


def test_manually_stopped_unit_is_left_alone_and_alerted_once():
    st = State()
    d1 = decide(obs(preprocess_state="inactive"), st, CFG)
    d2 = decide(obs(NOW + RUN, preprocess_state="inactive"), st, CFG)
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
    d2 = decide(obs(NOW + RUN, calibration_state="inactive"), st, CFG)
    d3 = decide(obs(NOW + 2 * RUN), st, CFG)
    assert not (d1.restart or d2.restart or d3.restart)
    assert len(d1.alerts) == 1 and not d2.alerts
    assert len(d3.alerts) == 1 and "running again" in d3.alerts[0]


def test_stale_ms_alerts_once_while_observing_only():
    st = State()
    old = NOW - 30 * 3600
    assert not decide(obs(newest_ms=old, cmd=announce(NOW - 86400)), st, CFG).alerts
    d1 = decide(obs(newest_ms=old), st, CFG)
    d2 = decide(obs(NOW + RUN, newest_ms=old), st, CFG)
    assert len(d1.alerts) == 1 and "measurement set" in d1.alerts[0] and not d2.alerts
    d3 = decide(obs(NOW + 2 * RUN, newest_ms=NOW), st, CFG)
    assert len(d3.alerts) == 1 and "appeared" in d3.alerts[0]
