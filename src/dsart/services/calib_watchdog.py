"""Watchdog for the h23 calibration pre-processor (``dsa110-calib-preprocess``).

The pre-processor pulls each finished correlator hdf5 file off its corr node
when the node announces it on etcd ``/cmd/cal``. It does that from an etcd
WATCH callback, and python-etcd3 kills the watch thread on a transient gRPC
error without telling anyone: systemd still says "active (running)", the
three children still report ``ntasks_alive: 1`` and the heartbeat is still
fresh, but nothing is ever pulled again. Pulls are purely event-driven, so
every file announced while the watch is dead stays on its node, no calibrator
transit triggers, and the SEFD page silently stops updating. This happened
2026-09-13 and again 2026-10-08 20:16 (two days of calibrator transits lost).

This module is a one-shot check, run every few minutes by
``dsa110-calib-watchdog.timer``; being a fresh process each time, it has no
long-lived watch of its own that could die the same way. Each run:

* remembers the latest ``/cmd/cal`` announcement and, on the next run
  (>= ``pull_wait_s`` later), checks that its set was pulled: some file of
  that set or a newer one is in the hdf5 directory (judged by the timestamp in
  the filename, so one node's failing pulls do not count), or anything at all
  arrived recently (a long backlog only delays pulls). Two unpulled
  announcements in a row while the corr nodes are announcing (the array is
  observing) restart the pre-processor. It deliberately does NOT compare
  filename times of consecutive sets: the corr nodes do not write files
  continuously across a stop/start, and a 23-min gap between sets read as a
  stall caused a false restart (2026-10-10 20:57);
* restarts it immediately if the unit has failed, its heartbeat
  (``/mon/service/calpreprocess``) is stale, or a child reports
  ``ntasks_alive: 0``. A unit that is merely inactive was stopped by hand
  and is left alone (reported only);
* reports, but never touches, the calibration consumer
  (``dsa110-calib-calibration``) if it is not running;
* reports when no new calibrator measurement set has appeared for
  ``ms_stale_s`` while the array is observing. That end-to-end check also
  covers the calibration consumer, whose own ``/cmd/cal`` watch can die in
  exactly the same way and which nothing here can see directly.

Restarts are rate-limited (``cooldown_s``) and capped (``max_restarts``
without a recovery in between). Every restart, give-up and recovery is
logged, published to etcd ``/mon/cal/watchdog``, and posted to Slack.

A restart does NOT re-pull the files announced while the pre-processor was
deaf; the alert names the window so a calibrator transit inside it can be
backfilled by re-announcing its files on ``/cmd/cal``.

Run: ``python -m dsart.services.calib_watchdog [--dry-run]``.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LOG = logging.getLogger("dsart.services.calib_watchdog")

PREPROCESS_UNIT = "dsa110-calib-preprocess.service"
CALIBRATION_UNIT = "dsa110-calib-calibration.service"
CHILDREN = ("rsync", "gather", "assess")

#: Correlator file names start with their UTC start time.
_FNAME_TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_")

_MJD_UNIX_EPOCH = 40587.0


def fname_time(name: str) -> Optional[float]:
    """Unix time encoded at the start of a correlator file name, else None."""
    m = _FNAME_TS.match(os.path.basename(name))
    if m is None:
        return None
    return float(calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")))


def utc(t: Optional[float]) -> str:
    if t is None:
        return "never"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))


# ---------------------------------------------------------------------------
# Config, observation, state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchdogConfig:
    #: How long an announced set gets to arrive before it counts as unpulled
    #: (healthy pulls take seconds; a backfill can queue a couple of minutes).
    pull_wait_s: float = 240.0
    #: Files of one set carry timestamps up to a few s apart (the
    #: pre-processor gathers within 60 s).
    gather_tol_s: float = 60.0
    #: Consecutive unpulled announcements required before restarting. With
    #: the 5-min timer that is >= ~10 min of announced files not pulled.
    confirm_runs: int = 2
    #: The latest announced file must have started within this long for the
    #: array to count as observing (file length ~5.1 min + announce latency).
    observing_window_s: float = 1500.0
    #: /mon/service/<x> heartbeats are written every 60 s.
    heartbeat_stale_s: float = 300.0
    cooldown_s: float = 1800.0
    max_restarts: int = 3
    #: Longest gap between calibrator MSs before alerting. At dec +16 the
    #: longest gap between transits is 17.5 h (0521+166 -> 2253+161); other
    #: pointings can have a single calibrator per day.
    ms_stale_s: float = 26 * 3600.0


@dataclass(frozen=True)
class Observation:
    now: float
    #: Latest /cmd/cal value (decoded), None if unreadable.
    cmd: Optional[Mapping[str, Any]]
    #: Newest filename time among files in the hdf5 directory.
    newest_pulled: Optional[float]
    preprocess_state: str  # systemctl ActiveState
    preprocess_heartbeat: Optional[float]  # unix time
    #: child name -> ntasks_alive (missing child = unknown)
    children_alive: Mapping[str, int]
    calibration_state: str
    calibration_heartbeat: Optional[float]
    #: mtime of the newest *.ms in the calibration directory.
    newest_ms: Optional[float] = None
    #: mtime of the hdf5 directory = when the last file arrived (rsync renames
    #: each file into place). rsync -a preserves the files' own mtimes.
    last_arrival: Optional[float] = None


@dataclass
class State:
    #: Filename time of the announcement seen at the previous run, and when.
    pending_t: Optional[float] = None
    pending_seen: Optional[float] = None
    stalled_runs: int = 0
    restarts: int = 0  # since the last recovery
    last_restart: Optional[float] = None
    gave_up: bool = False
    #: Newest pulled file when the incident opened (start of the deaf window).
    incident_from: Optional[float] = None
    incident_reason: str = ""
    preprocess_stopped_alerted: bool = False
    calibration_alerted: bool = False
    ms_alerted: bool = False

    @classmethod
    def load(cls, path: Path) -> "State":
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        os.replace(tmp, path)


@dataclass
class Decision:
    restart: bool = False
    alerts: List[str] = field(default_factory=list)
    status: str = "ok"
    detail: str = ""


# ---------------------------------------------------------------------------
# Decision logic (pure)
# ---------------------------------------------------------------------------


def decide(obs: Observation, st: State, cfg: WatchdogConfig) -> Decision:
    """Decide what to do this run. Mutates ``st``; performs no I/O."""
    dec = Decision()

    # Calibration consumer: report only, once per incident.
    cal_hb_stale = (obs.calibration_heartbeat is None
                    or obs.now - obs.calibration_heartbeat > cfg.heartbeat_stale_s)
    cal_bad = obs.calibration_state != "active" or cal_hb_stale
    if cal_bad and not st.calibration_alerted:
        st.calibration_alerted = True
        dec.alerts.append(
            f"calibration consumer {CALIBRATION_UNIT} is {obs.calibration_state}"
            f" (heartbeat {utc(obs.calibration_heartbeat)}): calibrate commands "
            "are not being processed. Not restarted automatically.")
    elif not cal_bad and st.calibration_alerted:
        st.calibration_alerted = False
        dec.alerts.append(f"calibration consumer {CALIBRATION_UNIT} is running again.")

    # Is the array observing (the corr nodes announcing fresh files)? A
    # "calibrate" command as the latest value, or an unreadable one, gives no
    # announcement this run.
    announced = None
    if obs.cmd is not None and obs.cmd.get("cmd") == "rsync":
        announced = fname_time(str((obs.cmd.get("val") or {}).get("filename", "")))
    observing = announced is not None and obs.now - announced <= cfg.observing_window_s

    # Was the set announced at the previous run pulled? None = no verdict.
    pending_t, pending_seen = st.pending_t, st.pending_seen
    unpulled = None
    if (pending_t is not None and pending_seen is not None
            and obs.now - pending_seen >= cfg.pull_wait_s):
        pulled = (obs.newest_pulled is not None
                  and obs.newest_pulled >= pending_t - cfg.gather_tol_s)
        arriving = (obs.last_arrival is not None
                    and obs.now - obs.last_arrival < cfg.pull_wait_s)
        unpulled = not pulled and not arriving
    if not observing:
        if announced is not None:
            st.pending_t = st.pending_seen = None
    elif pending_t is None or unpulled is not None:
        st.pending_t, st.pending_seen = announced, obs.now

    # End-to-end: are calibrator transits still turning into MSs?
    ms_stale = obs.newest_ms is None or obs.now - obs.newest_ms > cfg.ms_stale_s
    if ms_stale and observing and not st.ms_alerted:
        st.ms_alerted = True
        dec.alerts.append(
            f"no new calibrator measurement set since {utc(obs.newest_ms)} "
            f"(>{cfg.ms_stale_s / 3600:.0f} h) although the array is observing. "
            "Check `journalctl --user -u dsa110-calib-preprocess` for 'Calibrating <src>' "
            f"and {CALIBRATION_UNIT} for 'Creating ...ms' (its /cmd/cal watch can die "
            "silently too). Ignore if the array was down through the transits.")
    elif not ms_stale and st.ms_alerted:
        st.ms_alerted = False
        dec.alerts.append(f"a new calibrator measurement set appeared ({utc(obs.newest_ms)}).")

    # Pre-processor faults, most specific first.
    reason = ""
    if obs.preprocess_state == "failed":
        reason = f"{PREPROCESS_UNIT} has failed"
    elif obs.preprocess_state != "active":
        # Stopped by hand (inactive) or mid-transition: leave it alone.
        dec.status = f"preprocess {obs.preprocess_state}"
        if observing and not st.preprocess_stopped_alerted:
            st.preprocess_stopped_alerted = True
            dec.alerts.append(
                f"{PREPROCESS_UNIT} is {obs.preprocess_state} while the corr nodes are "
                "announcing files: nothing is being pulled. Left alone (it looks "
                "deliberately stopped) -- start it with `systemctl --user start "
                f"{PREPROCESS_UNIT}`.")
        st.stalled_runs = 0
        return dec
    elif (obs.preprocess_heartbeat is None
          or obs.now - obs.preprocess_heartbeat > cfg.heartbeat_stale_s):
        reason = (f"pre-processor heartbeat is stale (last {utc(obs.preprocess_heartbeat)}):"
                  " its main loop is hung")
    else:
        dead = [c for c in CHILDREN if obs.children_alive.get(c, 1) < 1]
        if dead:
            reason = f"pre-processor child process(es) dead: {', '.join(dead)}"
    st.preprocess_stopped_alerted = False

    if not observing:
        st.stalled_runs = 0
    elif not reason and unpulled is not None:
        if unpulled:
            st.stalled_runs += 1
            if st.stalled_runs >= cfg.confirm_runs:
                reason = (f"the set announced as {utc(pending_t)} is still not pulled "
                          f"{(obs.now - pending_seen) / 60:.0f} min later, nothing has "
                          f"arrived since {utc(obs.last_arrival)} and the corr nodes "
                          f"keep announcing (latest {utc(announced)}): its etcd watch "
                          "has died")
            else:
                dec.status = "stall suspected"
        else:
            st.stalled_runs = 0

    dec.detail = (f"announced {utc(announced)} newest_pulled {utc(obs.newest_pulled)}"
                  f" last_arrival {utc(obs.last_arrival)} observing {observing}"
                  f" unpulled {unpulled}")

    if not reason:
        healthy = observing and unpulled is False
        if healthy and (st.restarts or st.gave_up):
            dec.alerts.append(
                f"pre-processor recovered: files are being pulled again (newest "
                f"{utc(obs.newest_pulled)}). Files announced between "
                f"{utc(st.incident_from)} and the restart were NOT pulled; any "
                "calibrator transit in that window needs its files re-announced on "
                "/cmd/cal.")
            st.restarts, st.gave_up = 0, False
            st.incident_from, st.incident_reason = None, ""
        if dec.status == "ok" and not observing:
            dec.status = "idle (not observing)"
        return dec

    # A fault: restart, subject to cooldown and the cap.
    dec.status = "fault"
    if st.restarts == 0 and not st.gave_up:
        st.incident_from = obs.newest_pulled
    st.incident_reason = reason
    if st.gave_up:
        return dec
    if st.last_restart is not None and obs.now - st.last_restart < cfg.cooldown_s:
        dec.status = "fault (cooldown)"
        return dec
    if st.restarts >= cfg.max_restarts:
        st.gave_up = True
        dec.alerts.append(
            f"giving up after {st.restarts} restarts of {PREPROCESS_UNIT} without "
            f"recovery: {reason}. Needs a human; no more automatic restarts until "
            "pulls resume.")
        return dec
    st.restarts += 1
    st.last_restart = obs.now
    st.stalled_runs = 0
    # Whatever was announced before the restart is never re-pulled; judge the
    # new process on what is announced after it.
    st.pending_t = st.pending_seen = None
    dec.restart = True
    dec.status = "restarted"
    again = " (the previous restart did not help)" if st.restarts > 1 else ""
    dec.alerts.append(
        f"restarting {PREPROCESS_UNIT}{again}, {st.restarts}/{cfg.max_restarts}: "
        f"{reason}. Files announced since {utc(st.incident_from)} are NOT re-pulled "
        "by the restart; re-announce any calibrator transit in that window on "
        "/cmd/cal.")
    return dec


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def newest_pulled(hdf5_dir: Path) -> Optional[float]:
    newest = None
    with os.scandir(hdf5_dir) as it:
        for entry in it:
            t = fname_time(entry.name)
            if t is not None and (newest is None or t > newest):
                newest = t
    return newest


def newest_ms(ms_dir: Path) -> Optional[float]:
    newest = None
    with os.scandir(ms_dir) as it:
        for entry in it:
            if entry.name.endswith(".ms"):
                try:
                    t = entry.stat().st_mtime
                except OSError:
                    continue
                if newest is None or t > newest:
                    newest = t
    return newest


def unit_state(unit: str) -> str:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "ActiveState", "--value", unit],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("systemctl show %s: %s", unit, exc)
        return "unknown"


def _get_dict(client: Any, key: str) -> Optional[Dict[str, Any]]:
    try:
        value, _ = client.get(key)
        return json.loads(value) if value else None
    except Exception as exc:  # noqa: BLE001 -- etcd/JSON hiccup = unknown
        LOG.warning("etcd get %s: %s", key, exc)
        return None


def _mjd_heartbeat(d: Optional[Mapping[str, Any]]) -> Optional[float]:
    try:
        return (float(d["time"]) - _MJD_UNIX_EPOCH) * 86400.0  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        return None


def observe(client: Any, hdf5_dir: Path, ms_dir: Path, now: float) -> Observation:
    try:
        last_arrival: Optional[float] = os.stat(hdf5_dir).st_mtime
    except OSError:
        last_arrival = None
    children = {}
    for c in CHILDREN:
        d = _get_dict(client, f"/mon/cal/{c}_process")
        if d is not None and "ntasks_alive" in d:
            children[c] = int(d["ntasks_alive"])
    return Observation(
        now=now,
        cmd=_get_dict(client, "/cmd/cal"),
        newest_pulled=newest_pulled(hdf5_dir),
        preprocess_state=unit_state(PREPROCESS_UNIT),
        preprocess_heartbeat=_mjd_heartbeat(_get_dict(client, "/mon/service/calpreprocess")),
        children_alive=children,
        calibration_state=unit_state(CALIBRATION_UNIT),
        calibration_heartbeat=_mjd_heartbeat(_get_dict(client, "/mon/service/calibration")),
        newest_ms=newest_ms(ms_dir),
        last_arrival=last_arrival,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hdf5-dir", type=Path, default=Path("/operations/correlator"))
    p.add_argument("--ms-dir", type=Path, default=Path("/operations/calibration"))
    p.add_argument("--state", type=Path,
                   default=Path.home() / ".local/state/dsa110-calib-watchdog/state.json")
    p.add_argument("--etcd-host", default="etcdv3service.pro.pvt")
    p.add_argument("--etcd-port", type=int, default=2379)
    p.add_argument("--slack-channel", default="")
    p.add_argument("--slack-token-file", default="")
    p.add_argument("--pull-wait-s", type=float, default=WatchdogConfig.pull_wait_s)
    p.add_argument("--cooldown-s", type=float, default=WatchdogConfig.cooldown_s)
    p.add_argument("--dry-run", action="store_true",
                   help="decide and log only: no restart, no Slack, no etcd/state writes")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import etcd3  # deferred so the decision logic imports without it

    cfg = WatchdogConfig(pull_wait_s=args.pull_wait_s, cooldown_s=args.cooldown_s)
    client = etcd3.client(host=args.etcd_host, port=args.etcd_port, timeout=10)
    now = time.time()
    st = State.load(args.state)
    obs = observe(client, args.hdf5_dir, args.ms_dir, now)
    dec = decide(obs, st, cfg)
    LOG.info("%s; %s; preprocess %s, children %s, calibration %s, newest MS %s",
             dec.status, dec.detail, obs.preprocess_state, dict(obs.children_alive),
             obs.calibration_state, utc(obs.newest_ms))
    for a in dec.alerts:
        LOG.warning("ALERT %s", a)
    if args.dry_run:
        if dec.restart:
            LOG.warning("dry run: would restart %s", PREPROCESS_UNIT)
        return 0

    if dec.restart:
        rc = subprocess.run(["systemctl", "--user", "restart", PREPROCESS_UNIT],
                            timeout=120).returncode
        if rc != 0:
            dec.alerts.append(f"`systemctl --user restart {PREPROCESS_UNIT}` exited {rc}")
            LOG.error("restart exited %d", rc)
    st.save(args.state)
    try:
        client.put("/mon/cal/watchdog", json.dumps({
            "time": now, "status": dec.status, "detail": dec.detail,
            "restarts": st.restarts, "last_restart": st.last_restart,
            "gave_up": st.gave_up, "incident_reason": st.incident_reason,
        }))
    except Exception as exc:  # noqa: BLE001 -- status publish is best-effort
        LOG.warning("etcd put /mon/cal/watchdog: %s", exc)
    if dec.alerts and args.slack_channel:
        from dsart.services.slack_notify import SlackNotifier, SlackNotifyConfig
        notifier = SlackNotifier(SlackNotifyConfig(
            enabled=True, channel=args.slack_channel, token_file=args.slack_token_file))
        for a in dec.alerts:
            res = notifier.post_text(f":warning: *calibration watchdog (h23)*: {a}")
            if not res.get("ok"):
                LOG.warning("slack post failed: %s", res.get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
