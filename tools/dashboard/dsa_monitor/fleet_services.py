"""M7.4 Phase 8 v2: fleet-wide service status + cold restart.

Two public entry points:

* :func:`query_all_services_status` walks :data:`SERVICE_INVENTORY`
  with a per-host ThreadPoolExecutor (max 8 in flight, 5s timeout
  per host) and returns one row per service:

  ``{tier, host, service, kind, state, age_s, raw, err}``

  where ``state`` is one of ``active`` / ``inactive`` / ``failed`` /
  ``unknown`` / ``unreachable``. The render layer colours these
  green / gray / red / yellow / orange.

* :func:`restart_all_services` is the "cold recovery from stop"
  button. The full sequence is documented in
  :func:`restart_all_services` itself; the short version is:

  1. ``stop`` broadcast on /cmd/corr_rt/0 + search fanout.
  2. ssh fanout to each corr/search host: ``pkill -f dsart`` then
     ``pkill -9 -f dsart`` (best-effort, 5s timeout per host).
  3. ssh fanout to each corr/search host: tear down PSRDADA buffers
     + ``rm -f /dev/shm/dsart-capture-* /dev/shm/dsart-rfi-window-*``
     + recreate ``/tmp/dsart-rt-children``.
  4. Local ``systemctl --user restart sefd_dashboard.service
     dsart_c2.service`` on lxd110h23. The ``dsa_monitor.service``
     restart is *deferred* via a detached ``subprocess.Popen`` that
     ``sleep 2 && systemctl --user restart dsa_monitor.service``, so
     the HTTP response can come back before this process is killed.
  5. ``lxc exec calibration23 -- sudo -u ubuntu systemctl --user
     restart hiplot.service`` (best-effort).
  6. ssh fanout to each corr/search host: re-spawn the dsart_rt
     orchestrator (NOT ``start`` / ``utc_start`` — operator drives
     those from the existing Control tab buttons).
  7. ``lxd110h20`` is excluded from every fanout (see
     :data:`H20_HOSTNAMES`).

Both helpers are stdlib-only (subprocess, concurrent.futures, json,
logging, time).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from services_inventory import (
    CORR_HOSTS,
    H20_HOSTNAMES,
    HOST_CALIBRATION23,
    HOST_H23,
    KIND_LXC_SYSTEMD_USER,
    KIND_PROCESS,
    KIND_SYSTEMD_SYSTEM,
    KIND_SYSTEMD_USER,
    SEARCH_HOSTS,
    SERVICE_INVENTORY,
    ServiceEntry,
    TIER_DSART_ORCH_CORR,
    TIER_DSART_ORCH_SEARCH,
    all_orch_hosts,
    restartable_entries,
)

LOG = logging.getLogger("dsa_monitor.fleet_services")

#: ssh options used on every fanout. ``BatchMode=yes`` makes a
#: password-prompt host fail fast rather than hang; ``ConnectTimeout``
#: keeps a dead host bounded; ``-n`` detaches stdin so the parent's
#: stdin is not consumed by ssh.
_SSH_OPTS: tuple[str, ...] = (
    "-o", "ConnectTimeout=5",
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-n",
)

#: Per-host wall-clock budget for one ssh / lxc invocation.
DEFAULT_SSH_TIMEOUT_S: float = 5.0

#: Per-host wall-clock budget for the *re-spawn* fanout — slightly
#: longer because the launch snippet sleeps 2 s to verify the child
#: stayed alive.
DEFAULT_RESPAWN_TIMEOUT_S: float = 10.0

#: ThreadPoolExecutor max workers for any host-fanout step.
DEFAULT_FANOUT_WORKERS: int = 8

#: Repo root the launch snippet bases conda + PYTHONPATH on. This
#: mirrors ``tools/ops/_m75_phaseB_16x4_launch.sh::REPO`` which uses
#: ``/home/ubuntu/proj/dsa110-rt`` on the corr / search hosts.
REPO_ON_NODES: str = "/home/ubuntu/proj/dsa110-rt"

#: The Phase-B cleanup script's per-host commands, lifted verbatim
#: into Python (so the dashboard does not shell out to the script).
#: One command per corr-node line; the search-node variant adds the
#: search_rx / search_compute patterns.
_CLEANUP_CORR_CMD = (
    "for round in 1 2; do "
    "for p in $(ps -eo pid,cmd "
    "| grep -E 'dsart|corr_fast|corr_slow|dada_drain|dada_junkdb|dsaX_merge' "
    "| grep -v grep "
    "| awk '{print $1}'); do "
    "kill -9 $p 2>/dev/null; "
    "done; "
    "sleep 1; "
    "done; "
    "for k in dada eada fada bada gada hada; do "
    "dada_db -d -k $k >/dev/null 2>&1; "
    "done; "
    "ipcs -m | awk '/ubuntu/ {print $2}' | xargs -r -I{} ipcrm -m {} 2>/dev/null; "
    "rm -f /tmp/dsart-corr-*.ready; "
    "rm -f /dev/shm/dsart-capture-* /dev/shm/dsart-rfi-window-* 2>/dev/null; "
    "rm -rf /tmp/dsart-rt-children; "
    "mkdir -p /tmp/dsart-rt-children; "
    "echo OK"
)

_CLEANUP_SEARCH_CMD = (
    "for round in 1 2; do "
    "for p in $(ps -eo pid,cmd "
    "| grep -E 'dsart|search_compute|search_rx' "
    "| grep -v grep "
    "| awk '{print $1}'); do "
    "kill -9 $p 2>/dev/null; "
    "done; "
    "sleep 1; "
    "done; "
    "rm -f /dev/shm/dsart-rxring-* /dev/shm/dsart-* 2>/dev/null; "
    "rm -rf /tmp/dsart-rt-children; "
    "mkdir -p /tmp/dsart-rt-children; "
    "echo OK"
)

#: ``pkill -f dsart`` then sigkill — the "stop verb may have been
#: missed" belt-and-braces step. Best-effort: exit code is ignored.
_PKILL_DSART_CMD = (
    "pkill -f dsart 2>/dev/null; "
    "sleep 1; "
    "pkill -9 -f dsart 2>/dev/null; "
    "echo PKILL_OK"
)

#: Start-time housekeeping: the "safe to delete on restart" cleanup,
#: fanned out to corr + search nodes when the operator issues ``start``.
#: CRITICAL: on a *start* the node processes are coming UP and own the
#: PSRDADA rings + ``/dev/shm`` segments, so these commands NEVER touch
#: shm / dada (that is the ``stop`` / restart-all path's job). They only
#: remove accumulated ephemera that no running process holds an fd on:
#: stale orchestrator/routine logs, ready sentinels, and (corr) debug
#: block dumps. h23 (the candidate/CSV archive) is never in the fanout.
#: Corr additions (2026-07-14, after n10/n11 filled their NVMe):
#:
#: * ``data/spl/*.hdf5`` — meridian_fringestop_spl UVH5 outputs
#:   (~238 MB per observation, accumulate unbounded; the
#:   ``*_spl_incomplete.hdf5`` strays too). The ``*.npz``
#:   fringestopping-table cache in the same dir is deliberately KEPT
#:   (small, and deleting it forces a ~30 s casatools regen per dec).
#: * ``data/voltage_staging/*`` — M8 staged voltage fragments
#:   (~6.5 GiB each). Anything still staged at a *start* is stale:
#:   C3 collects KEEP events promptly after the trigger, and
#:   flag-only REJECTs are strays no collector will ever pull.
_START_CLEANUP_CORR_CMD = (
    "rm -f $HOME/tmp/dsart-rt/*.log 2>/dev/null; "
    "rm -f /tmp/dsart-corr-*.ready 2>/dev/null; "
    "rm -rf /home/ubuntu/tmp/dsart-fast-grid 2>/dev/null; "
    "rm -f /home/ubuntu/data/spl/*.hdf5 2>/dev/null; "
    "rm -f /home/ubuntu/data/voltage_staging/* 2>/dev/null; "
    "echo CLEAN_OK"
)

#: Search-node variant. In addition to the stale logs it clears the
#: LOCAL cube-dump tree: every triggered NPZ under ``c1.dump_root``
#: (default ``/home/ubuntu/data/c2/cube_dump``) is rsynced to h23 by
#: ``cube_uploader`` but never deleted locally, so it accumulates
#: unbounded. h23 holds the archived copy under
#: ``/dataz/dsa110/candidates/<event>/cubes/`` — the local copy is
#: redundant once a run starts. Also drops the per-event ``upload.log``.
_START_CLEANUP_SEARCH_CMD = (
    "rm -f $HOME/tmp/dsart-rt/*.log 2>/dev/null; "
    "rm -f /home/ubuntu/data/c2/cube_dump/*/upload.log 2>/dev/null; "
    "rm -rf /home/ubuntu/data/c2/cube_dump/* 2>/dev/null; "
    "echo CLEAN_OK"
)


# ---------------------------------------------------------------------------
# Thin subprocess helpers — every caller goes through these so the
# unit tests can monkey-patch ``subprocess.run`` / ``subprocess.Popen``
# in one place.
# ---------------------------------------------------------------------------


def _run(
    argv: list[str],
    *,
    timeout: float,
    input_text: str | None = None,
) -> dict[str, Any]:
    """Run ``argv`` with ``subprocess.run`` and return a JSON-ready
    summary dict. Never raises; failures (timeout, OSError, non-zero
    exit) all return ``ok=False`` with a descriptive ``err``.
    """
    started = time.monotonic()
    try:
        cp = subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "rc": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "err": f"timeout after {timeout:.1f}s",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except (FileNotFoundError, OSError) as exc:
        return {
            "ok": False,
            "rc": None,
            "stdout": "",
            "stderr": "",
            "err": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    return {
        "ok": cp.returncode == 0,
        "rc": cp.returncode,
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
        "err": "" if cp.returncode == 0 else f"rc={cp.returncode}",
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def _ssh_run(
    host: str,
    remote_cmd: str,
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
) -> dict[str, Any]:
    """``ssh <opts> <host> <remote_cmd>`` wrapper."""
    argv = ["ssh", *_SSH_OPTS, host, remote_cmd]
    return _run(argv, timeout=timeout)


def _lxc_run(
    container: str,
    remote_cmd: str,
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
    user: str | None = None,
) -> dict[str, Any]:
    """``lxc exec <container> -- bash -lc <remote_cmd>`` wrapper.

    ``user`` (e.g. ``"ubuntu"``) is wrapped with sudo so the systemd
    --user instance for that uid is reachable.
    """
    if user is not None:
        argv = [
            "lxc", "exec", container, "--",
            "sudo", "-u", user, "bash", "-lc", remote_cmd,
        ]
    else:
        argv = [
            "lxc", "exec", container, "--", "bash", "-lc", remote_cmd,
        ]
    return _run(argv, timeout=timeout)


def _systemctl_user_local(
    *args: str,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
) -> dict[str, Any]:
    """``systemctl --user <args...>`` on the local host (no ssh)."""
    return _run(["systemctl", "--user", *args], timeout=timeout)


def _systemctl_system_local(
    *args: str,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
) -> dict[str, Any]:
    """``systemctl <args...>`` (system slice) on the local host."""
    return _run(["systemctl", *args], timeout=timeout)


# ---------------------------------------------------------------------------
# Status query
# ---------------------------------------------------------------------------


def _classify_systemctl_output(stdout: str, rc: int | None) -> str:
    """Map ``systemctl is-active`` stdout to the canonical state.

    is-active prints one of ``active`` / ``inactive`` / ``failed`` /
    ``activating`` / ``deactivating`` / ``unknown`` — we collapse the
    transition states to ``active``/``inactive`` for the table.
    """
    txt = (stdout or "").strip().lower().splitlines()[-1:]
    s = txt[0] if txt else ""
    if s == "active" or s == "activating":
        return "active"
    if s == "inactive" or s == "deactivating":
        return "inactive"
    if s == "failed":
        return "failed"
    return "unknown"


def _query_systemd_unit(
    entry: ServiceEntry,
    *,
    user_scope: bool,
    via: str,
    timeout: float,
) -> dict[str, Any]:
    """``via`` is ``"local"`` / ``"ssh:<host>"`` / ``"lxc:<container>"``.

    We make two calls: ``is-active`` to get the state, ``show
    --property=ActiveEnterTimestampMonotonic`` for the age (best
    effort — falls back to ``None`` on parse failure).
    """
    scope = ("--user",) if user_scope else ()
    is_active_args = ["systemctl", *scope, "is-active", "--no-pager", entry.service]
    show_args = [
        "systemctl", *scope,
        "show", "--no-pager",
        "--property=ActiveEnterTimestamp",
        "--property=ActiveEnterTimestampMonotonic",
        entry.service,
    ]
    if via == "local":
        a = _run(is_active_args, timeout=timeout)
        b = _run(show_args, timeout=timeout)
    elif via.startswith("ssh:"):
        host = via.split(":", 1)[1]
        a = _ssh_run(host, " ".join(is_active_args), timeout=timeout)
        b = _ssh_run(host, " ".join(show_args), timeout=timeout)
    elif via.startswith("lxc:"):
        container = via.split(":", 1)[1]
        # systemd --user inside the LXC container requires sudo -u ubuntu.
        a = _lxc_run(
            container, " ".join(is_active_args),
            timeout=timeout, user="ubuntu",
        )
        b = _lxc_run(
            container, " ".join(show_args),
            timeout=timeout, user="ubuntu",
        )
    else:
        raise ValueError(f"unknown via={via!r}")
    if not a["ok"] and "timeout" in (a.get("err") or "").lower():
        return {"state": "unreachable", "age_s": None, "raw": a, "err": a["err"]}
    if a.get("rc") is None and not a["ok"]:
        # FileNotFoundError / OSError / etc. before the command ran.
        return {"state": "unreachable", "age_s": None, "raw": a, "err": a["err"]}
    state = _classify_systemctl_output(a.get("stdout", ""), a.get("rc"))
    age_s = _parse_active_enter_age(b.get("stdout", ""))
    return {
        "state": state,
        "age_s": age_s,
        "raw": a,
        "err": "" if state != "unknown" else a.get("err") or "",
    }


def _parse_active_enter_age(stdout: str) -> float | None:
    """Parse ``ActiveEnterTimestamp=... UTC`` lines into a wall-clock
    age in seconds. Returns None if the unit never entered active
    (timestamp is ``0`` / ``n/a``).
    """
    if not stdout:
        return None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("ActiveEnterTimestamp="):
            continue
        value = line.split("=", 1)[1].strip()
        if not value or value == "n/a" or value == "0":
            return None
        # systemd default fmt: "Wed 2026-05-28 07:00:00 UTC"
        # Parse with strptime; if any field doesn't match, give up.
        import datetime as _dt
        for fmt in (
            "%a %Y-%m-%d %H:%M:%S %Z",
            "%a %Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = _dt.datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_dt.timezone.utc)
                return max(0.0, time.time() - dt.timestamp())
            except ValueError:
                continue
        return None
    return None


def _query_process(
    entry: ServiceEntry,
    *,
    timeout: float,
) -> dict[str, Any]:
    """``pgrep -af '<pattern>'`` on the entry's host. rc=0 → active,
    rc=1 → inactive, anything else → unreachable.
    """
    pattern = entry.service
    # We anchor with the instance name when present so a stray
    # binary on the same host doesn't spoof a positive match. The
    # instance tag (e.g. "pipeline_rt") is part of the dsart_rt
    # argv from _m75_phaseB_16x4_launch.sh.
    if entry.instance:
        pattern = f"{entry.service}.*-in {entry.instance}"
    remote = f"pgrep -af {pattern!r}"
    r = _ssh_run(entry.host, remote, timeout=timeout)
    if not r["ok"] and "timeout" in (r.get("err") or "").lower():
        return {"state": "unreachable", "age_s": None, "raw": r, "err": r["err"]}
    if r.get("rc") is None:
        return {"state": "unreachable", "age_s": None, "raw": r, "err": r["err"]}
    if r["rc"] == 0:
        return {"state": "active", "age_s": None, "raw": r, "err": ""}
    if r["rc"] == 1:
        return {"state": "inactive", "age_s": None, "raw": r, "err": ""}
    return {"state": "unknown", "age_s": None, "raw": r, "err": r["err"]}


def _query_one(
    entry: ServiceEntry,
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
) -> dict[str, Any]:
    """Dispatch to the right query helper based on ``entry.kind``."""
    if entry.kind == KIND_SYSTEMD_USER:
        if entry.host == HOST_H23:
            via = "local"
        else:
            via = f"ssh:{entry.host}"
        sub = _query_systemd_unit(
            entry, user_scope=True, via=via, timeout=timeout,
        )
    elif entry.kind == KIND_SYSTEMD_SYSTEM:
        if entry.host == HOST_H23:
            via = "local"
        else:
            via = f"ssh:{entry.host}"
        sub = _query_systemd_unit(
            entry, user_scope=False, via=via, timeout=timeout,
        )
    elif entry.kind == KIND_LXC_SYSTEMD_USER:
        sub = _query_systemd_unit(
            entry, user_scope=True, via=f"lxc:{entry.host}", timeout=timeout,
        )
    elif entry.kind == KIND_PROCESS:
        sub = _query_process(entry, timeout=timeout)
    else:
        sub = {
            "state": "unknown", "age_s": None, "raw": {},
            "err": f"unknown kind={entry.kind!r}",
        }
    return {
        "tier": entry.tier,
        "host": entry.host,
        "service": entry.service,
        "kind": entry.kind,
        "state": sub["state"],
        "age_s": sub["age_s"],
        "err": sub.get("err", ""),
    }


def query_all_services_status(
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
    max_workers: int = DEFAULT_FANOUT_WORKERS,
    inventory: Iterable[ServiceEntry] | None = None,
) -> dict[str, Any]:
    """Fan out :func:`_query_one` across the inventory.

    Returns a JSON-ready dict::

        {
          "ts_unix": 1.7e9,
          "rows": [<one dict per entry>],
          "n_active":   int,
          "n_inactive": int,
          "n_failed":   int,
          "n_unreachable": int,
          "n_unknown":  int,
        }

    Failure on any one host never aborts the rest.
    """
    inv = list(inventory or SERVICE_INVENTORY)
    rows: list[dict[str, Any]] = [None] * len(inv)        # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        fut_map = {
            pool.submit(_query_one, e, timeout=timeout): idx
            for idx, e in enumerate(inv)
        }
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                rows[idx] = fut.result()
            except Exception as exc:                              # noqa: BLE001
                e = inv[idx]
                LOG.exception("query_one failed on %s/%s", e.host, e.service)
                rows[idx] = {
                    "tier": e.tier,
                    "host": e.host,
                    "service": e.service,
                    "kind": e.kind,
                    "state": "unreachable",
                    "age_s": None,
                    "err": f"{type(exc).__name__}: {exc}",
                }
    counts = {
        "n_active": sum(1 for r in rows if r["state"] == "active"),
        "n_inactive": sum(1 for r in rows if r["state"] == "inactive"),
        "n_failed": sum(1 for r in rows if r["state"] == "failed"),
        "n_unreachable": sum(1 for r in rows if r["state"] == "unreachable"),
        "n_unknown": sum(1 for r in rows if r["state"] == "unknown"),
    }
    return {
        "ts_unix": time.time(),
        "rows": rows,
        **counts,
    }


# ---------------------------------------------------------------------------
# Restart-all
# ---------------------------------------------------------------------------


def _fanout(
    hosts: Iterable[str],
    cmd: str,
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
    max_workers: int = DEFAULT_FANOUT_WORKERS,
    label: str = "fanout",
) -> dict[str, dict[str, Any]]:
    """ssh-fanout ``cmd`` to every host in ``hosts``.

    ``hosts`` is filtered through :data:`H20_HOSTNAMES` — h20 is
    NEVER part of any fanout, regardless of what the caller passed.
    """
    targets = [
        h for h in hosts if h not in H20_HOSTNAMES
        and h.split(".", 1)[0] not in H20_HOSTNAMES
    ]
    out: dict[str, dict[str, Any]] = {}
    if not targets:
        return out
    with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        fut_map = {
            pool.submit(_ssh_run, h, cmd, timeout=timeout): h
            for h in targets
        }
        for fut in as_completed(fut_map):
            h = fut_map[fut]
            try:
                out[h] = fut.result()
            except Exception as exc:                              # noqa: BLE001
                LOG.exception("%s on %s failed", label, h)
                out[h] = {
                    "ok": False, "rc": None, "stdout": "", "stderr": "",
                    "err": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": 0.0,
                }
    return out


def cleanup_nodes_for_start(
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
    max_workers: int = DEFAULT_FANOUT_WORKERS,
) -> dict[str, Any]:
    """Start-time housekeeping fanned out to every corr + search host.

    Deletes the "safe to delete on restart" ephemera identified in the
    2026-06-02 disk-write audit:

      corr   : ``~/tmp/dsart-rt/*.log`` (stale routine/orchestrator
               logs), ``/tmp/dsart-corr-*.ready`` sentinels,
               ``/home/ubuntu/tmp/dsart-fast-grid`` (debug block dumps),
               ``/home/ubuntu/data/spl/*.hdf5`` (SPL UVH5 outputs —
               the 2026-07-14 n10/n11 disk-full culprit; the fstable
               ``*.npz`` cache in that dir is kept), and
               ``/home/ubuntu/data/voltage_staging/*`` (stale M8
               voltage fragments — KEEPs are collected by C3 right
               after the trigger, so anything left at start is dead
               weight).
      search : ``~/tmp/dsart-rt/*.log`` + the local
               ``/home/ubuntu/data/c2/cube_dump`` tree (NPZs already
               rsynced to h23) and its ``upload.log`` files.

    NEVER touches **h23** (the candidate / rolling-CSV archive) or
    **h20** (read-only grafana/influx — :func:`_fanout` filters it),
    and NEVER removes ``/dev/shm`` rings or PSRDADA buffers (on a
    *start* the node processes own those). Best-effort: per-host
    failures are reported in ``per_host`` but never raise. Returns a
    JSON-ready summary.
    """
    corr_res = _fanout(
        CORR_HOSTS, _START_CLEANUP_CORR_CMD,
        timeout=timeout, max_workers=max_workers, label="start_cleanup_corr",
    )
    search_res = _fanout(
        SEARCH_HOSTS, _START_CLEANUP_SEARCH_CMD,
        timeout=timeout, max_workers=max_workers, label="start_cleanup_search",
    )
    per_host = {**corr_res, **search_res}
    n_ok = sum(1 for v in per_host.values() if v.get("ok"))
    n_failed = len(per_host) - n_ok
    return {
        "ok": n_failed == 0,
        "n_hosts": len(per_host),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "per_host": per_host,
    }


def _orch_relaunch_cmd(instance: str, cn_id: int) -> str:
    """Build the per-host orchestrator re-spawn command.

    Lifts STAGE 3a from ``tools/ops/_m75_phaseB_16x4_launch.sh`` but
    DOES NOT then ``start`` / ``utc_start`` the fleet — the operator
    drives those from the existing Control tab buttons. Re-spawn only
    puts ``dsart_rt`` back on the etcd command-watch line.
    """
    repo = REPO_ON_NODES
    return (
        "source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; "
        "conda activate dsa110-rt 2>/dev/null; "
        f"cd {repo}; "
        f"export PYTHONPATH={repo}/src; "
        "export DSART_RT_LOG_DIR=/tmp/dsart-rt-children; "
        "export DSART_RT_GATE_TIMEOUT_S=300; "
        "setsid nohup python3 -u -m dsart.services.dsart_rt "
        f"-in {instance} -cn {int(cn_id)} --log-level INFO "
        f"> /tmp/dsart-rt-{instance}-{int(cn_id)}.log 2>&1 < /dev/null &"
        f" echo $! > /tmp/dsart-rt-{instance}-{int(cn_id)}.pid; "
        "disown; "
        "sleep 2; "
        f"if kill -0 $(cat /tmp/dsart-rt-{instance}-{int(cn_id)}.pid) "
        "2>/dev/null; then echo alive; else echo DEAD; fi"
    )


def _respawn_dsart_orch(
    *,
    timeout: float = DEFAULT_RESPAWN_TIMEOUT_S,
    max_workers: int = DEFAULT_FANOUT_WORKERS,
    inventory: Iterable[ServiceEntry] | None = None,
) -> dict[str, dict[str, Any]]:
    """ssh-fanout the per-host orchestrator re-spawn to every entry
    in the ``dsart_orch_corr`` + ``dsart_orch_search`` tiers, EXCLUDING
    anything on a host in :data:`H20_HOSTNAMES`.
    """
    inv = list(inventory or SERVICE_INVENTORY)
    targets = [
        e for e in inv
        if e.tier in (TIER_DSART_ORCH_CORR, TIER_DSART_ORCH_SEARCH)
        and e.is_restartable()
        and e.cn_id is not None
        and e.instance is not None
    ]
    out: dict[str, dict[str, Any]] = {}
    if not targets:
        return out
    with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        fut_map = {}
        for e in targets:
            cmd = _orch_relaunch_cmd(str(e.instance), int(e.cn_id))  # type: ignore[arg-type]
            fut_map[pool.submit(_ssh_run, e.host, cmd, timeout=timeout)] = e.host
        for fut in as_completed(fut_map):
            h = fut_map[fut]
            try:
                out[h] = fut.result()
            except Exception as exc:                              # noqa: BLE001
                LOG.exception("respawn on %s failed", h)
                out[h] = {
                    "ok": False, "rc": None, "stdout": "", "stderr": "",
                    "err": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": 0.0,
                }
    return out


def _restart_local_user_units(
    units: Iterable[str],
    *,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
) -> dict[str, dict[str, Any]]:
    """``systemctl --user restart <unit>`` for each unit on the local
    host. Does NOT include ``dsa_monitor.service`` — that one is
    deferred (see :func:`_schedule_self_restart`).
    """
    out: dict[str, dict[str, Any]] = {}
    for unit in units:
        out[unit] = _systemctl_user_local("restart", unit, timeout=timeout)
    return out


def _schedule_self_restart(
    *,
    unit: str = "dsa_monitor.service",
    delay_s: float = 2.0,
) -> dict[str, Any]:
    """Spawn a detached ``bash -lc 'sleep N && systemctl --user
    restart <unit>'`` so the HTTP response can complete before this
    process is killed.

    Uses ``subprocess.Popen`` (NOT ``subprocess.run``) so the parent
    does not block. Output is discarded — there is no one to read it
    once the dsa_monitor process is gone.
    """
    cmd = f"sleep {delay_s:g} && systemctl --user restart {unit}"
    argv = ["bash", "-lc", cmd]
    try:
        # start_new_session=True detaches the child so a parent
        # signal does not propagate to it.
        popen = subprocess.Popen(                                  # noqa: S603
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {
            "ok": True,
            "pid": popen.pid,
            "argv": argv,
            "delay_s": float(delay_s),
            "unit": unit,
        }
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("schedule_self_restart failed")
        return {
            "ok": False,
            "err": f"{type(exc).__name__}: {exc}",
            "argv": argv,
            "delay_s": float(delay_s),
            "unit": unit,
        }


def _restart_hiplot_lxc(
    *,
    container: str = HOST_CALIBRATION23,
    timeout: float = DEFAULT_SSH_TIMEOUT_S,
) -> dict[str, Any]:
    """``lxc exec calibration23 -- sudo -u ubuntu systemctl --user
    restart hiplot.service``."""
    return _lxc_run(
        container,
        "systemctl --user restart hiplot.service",
        timeout=timeout,
        user="ubuntu",
    )


def restart_all_services(
    *,
    dry_run: bool = False,
    timeout_s: float = DEFAULT_SSH_TIMEOUT_S,
    respawn_timeout_s: float = DEFAULT_RESPAWN_TIMEOUT_S,
    max_workers: int = DEFAULT_FANOUT_WORKERS,
    stop_broadcast: Callable[[], dict[str, Any]] | None = None,
    self_restart_delay_s: float = 2.0,
    inventory: Iterable[ServiceEntry] | None = None,
) -> dict[str, Any]:
    """The "Restart everything (except lxd110h20)" button.

    Executes the 7 steps documented at the top of this module. Returns
    a JSON-ready summary dict the audit-log layer / Flask response
    serialises. ``ok`` is True iff every step's sub-results were ok.

    Parameters
    ----------
    dry_run
        If True, do NOT actually invoke any ssh / systemctl / Popen
        calls. Returns a summary that records what *would* have been
        attempted. Used by the unit tests + a future "preflight"
        button.
    stop_broadcast
        Optional zero-arg callable returning the result dict for the
        etcd ``stop`` broadcast (step 1). Lets the Flask layer pass
        in ``functools.partial(control_stop_fleet, control_store,
        user=user)`` without us importing control_store here (which
        would be a circular import on the control_store side).
    self_restart_delay_s
        Seconds the deferred ``dsa_monitor.service`` restart should
        sleep before tearing the dashboard process down. Default 2 s
        is enough for Flask to ship the HTTP response.
    inventory
        Override for the unit tests — defaults to
        :data:`SERVICE_INVENTORY`.
    """
    started_at = time.time()
    summary: dict[str, Any] = {
        "ok": True,
        "ts_unix": started_at,
        "dry_run": bool(dry_run),
        "steps": {},
        "host_results": {},
    }

    inv = list(inventory or SERVICE_INVENTORY)
    corr_hosts = tuple(
        e.host for e in inv
        if e.tier == TIER_DSART_ORCH_CORR and e.is_restartable()
    )
    search_hosts = tuple(
        e.host for e in inv
        if e.tier == TIER_DSART_ORCH_SEARCH and e.is_restartable()
    )

    # Step 1: etcd stop broadcast.
    if dry_run or stop_broadcast is None:
        summary["steps"]["1_stop_broadcast"] = {
            "skipped": dry_run or stop_broadcast is None,
            "reason": (
                "dry_run" if dry_run else "no stop_broadcast callable"
            ),
        }
    else:
        try:
            res = stop_broadcast()
            summary["steps"]["1_stop_broadcast"] = {
                "ok": bool(res.get("ok", True)),
                "result": res,
            }
            if not summary["steps"]["1_stop_broadcast"]["ok"]:
                summary["ok"] = False
        except Exception as exc:                                   # noqa: BLE001
            LOG.exception("stop_broadcast failed")
            summary["steps"]["1_stop_broadcast"] = {
                "ok": False, "err": f"{type(exc).__name__}: {exc}",
            }
            summary["ok"] = False

    # Step 2: pkill -f dsart fanout (best-effort).
    if dry_run:
        summary["steps"]["2_pkill_dsart"] = {
            "skipped": True, "targets": list(corr_hosts + search_hosts),
        }
    else:
        pkill = _fanout(
            corr_hosts + search_hosts, _PKILL_DSART_CMD,
            timeout=timeout_s, max_workers=max_workers, label="pkill_dsart",
        )
        # pkill is best-effort: rc=1 means "no procs matched" which
        # is fine. Treat any successful ssh round-trip (rc=0 or 1) as
        # ok; only ssh-layer failures bubble up.
        for h, r in pkill.items():
            if r.get("rc") not in (0, 1):
                summary["ok"] = False
        summary["steps"]["2_pkill_dsart"] = {
            "ok": all(
                r.get("rc") in (0, 1) for r in pkill.values()
            ),
            "per_host": pkill,
        }
        _merge_host_results(summary["host_results"], pkill, "pkill_dsart")

    # Step 3: PSRDADA + shm cleanup fanout.
    if dry_run:
        summary["steps"]["3_cleanup"] = {
            "skipped": True, "targets": list(corr_hosts + search_hosts),
        }
    else:
        clean_corr = _fanout(
            corr_hosts, _CLEANUP_CORR_CMD,
            timeout=timeout_s, max_workers=max_workers, label="cleanup_corr",
        )
        clean_search = _fanout(
            search_hosts, _CLEANUP_SEARCH_CMD,
            timeout=timeout_s, max_workers=max_workers, label="cleanup_search",
        )
        merged = {**clean_corr, **clean_search}
        cleanup_ok = all(r.get("ok") for r in merged.values()) if merged else True
        if not cleanup_ok:
            summary["ok"] = False
        summary["steps"]["3_cleanup"] = {
            "ok": cleanup_ok,
            "per_host": merged,
        }
        _merge_host_results(summary["host_results"], merged, "cleanup")

    # Step 4: local systemctl --user restart for the non-self units.
    local_units = ("sefd_dashboard.service", "dsart_c2.service")
    if dry_run:
        summary["steps"]["4_local_units_restart"] = {
            "skipped": True, "units": list(local_units),
        }
    else:
        local_res = _restart_local_user_units(local_units, timeout=timeout_s)
        local_ok = all(r.get("ok") for r in local_res.values())
        if not local_ok:
            summary["ok"] = False
        summary["steps"]["4_local_units_restart"] = {
            "ok": local_ok,
            "per_unit": local_res,
        }

    # Step 5: lxc exec calibration23 hiplot restart.
    if dry_run:
        summary["steps"]["5_hiplot_restart"] = {"skipped": True}
    else:
        h_res = _restart_hiplot_lxc(timeout=timeout_s)
        # Best-effort: a missing container is logged but doesn't fail
        # the overall job.
        summary["steps"]["5_hiplot_restart"] = {
            "ok": bool(h_res.get("ok")),
            "result": h_res,
            "best_effort": True,
        }

    # Step 6: re-spawn dsart_rt orchestrators.
    if dry_run:
        summary["steps"]["6_orch_respawn"] = {
            "skipped": True,
            "targets": list(corr_hosts + search_hosts),
        }
    else:
        respawn = _respawn_dsart_orch(
            timeout=respawn_timeout_s, max_workers=max_workers,
            inventory=inv,
        )
        respawn_ok = all(
            r.get("ok") and "alive" in (r.get("stdout") or "")
            for r in respawn.values()
        ) if respawn else True
        if not respawn_ok:
            summary["ok"] = False
        summary["steps"]["6_orch_respawn"] = {
            "ok": respawn_ok, "per_host": respawn,
        }
        _merge_host_results(summary["host_results"], respawn, "orch_respawn")

    # Step 7 (deferred): self-restart of dsa_monitor.service.
    # IMPORTANT: this is the LAST thing we do so the prior steps'
    # results are captured in ``summary`` before the process dies.
    if dry_run:
        summary["steps"]["7_self_restart_scheduled"] = {"skipped": True}
    else:
        sched = _schedule_self_restart(delay_s=self_restart_delay_s)
        summary["steps"]["7_self_restart_scheduled"] = sched
        if not sched.get("ok"):
            summary["ok"] = False

    summary["elapsed_s"] = round(time.time() - started_at, 3)
    summary["h20_hostnames_excluded"] = sorted(H20_HOSTNAMES)
    return summary


def _merge_host_results(
    bucket: dict[str, Any],
    per_host: dict[str, dict[str, Any]],
    step_label: str,
) -> None:
    """Append a ``{step_label: <result>}`` entry into the host-keyed
    rollup so the audit row can summarise everything per host in one
    glance.
    """
    for host, res in per_host.items():
        bucket.setdefault(host, {})[step_label] = {
            "ok": bool(res.get("ok")),
            "rc": res.get("rc"),
            "err": res.get("err", ""),
        }


__all__ = [
    "DEFAULT_FANOUT_WORKERS",
    "DEFAULT_RESPAWN_TIMEOUT_S",
    "DEFAULT_SSH_TIMEOUT_S",
    "REPO_ON_NODES",
    "query_all_services_status",
    "restart_all_services",
]
