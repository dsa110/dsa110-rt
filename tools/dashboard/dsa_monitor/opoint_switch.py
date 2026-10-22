"""Op-point switch behind the web-UI Start control (x2, docs/x2/SPEC.md §10).

``control_store.control_start_fleet(..., opoint=...)`` calls
:func:`switch_opoint` before it broadcasts ``start``. A switch happens only
when the requested op-point differs from ``/cnf/opoint`` (absent = 1x) or the
operator asks to re-apply; otherwise the Start proceeds exactly as before
(after checking that the live configs carry the active op-point's stamp).

Switch sequence (every step audit-logged under namespace ``opoint``; any
failure refuses the Start with a clear message and nothing is broadcast):

1. fleet stopped: :func:`control_store.compute_system_state` is READY and no
   search orchestrator reports running/starting;
2. ``systemctl --user`` reachable from this process (:class:`SystemdUser`);
3. the target set is stored and clean: ``/cnf/opoints/<op>/{pipeline_rt,
   search_rt}`` present, stamped ``opoint: <op>``, no X2TODO placeholders or
   ``x2_todo`` entries, no references to another op-point's files; the h23
   files the restarted services will read (search_rt, C2 criteria) exist and
   are clean too;
4. backup of the live ``/cnf/pipeline_rt``, ``/cnf/search_rt``, ``/cnf/opoint``
   to ``/cnf/opoints/_previous/*``;
5. one write (an etcd transaction when available) of the two live configs and
   the ``/cnf/opoint`` record, verified by read-back;
6. ``systemctl --user restart`` of :data:`OPOINT_RESTART_UNITS`, then a wait
   until each is active/running with a new, stable MainPID;
7. on a failure in 5-6: the backup is written back and the units restarted
   again so they run the previous op-point.

The dashboard is the systemd --user unit ``dsa_monitor.service`` (verified
2026-10-22: parent is ``systemd --user``, environment has XDG_RUNTIME_DIR and
DBUS_SESSION_BUS_ADDRESS), the same manager that owns the restarted units, so
``systemctl --user`` works directly; :class:`SystemdUser` still fills both
variables from ``/run/user/<uid>`` when the process was started without them
(e.g. a manual ``python app.py`` from a non-login shell).
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

LOG = logging.getLogger("dsa_monitor.opoint_switch")

REPO_ROOT = Path(__file__).resolve().parents[3]
#: The h23 checkout's configs (what C2/C3/bot read after a restart).
H23_CONFIGS_DIR = REPO_ROOT / "configs"

try:  # the dashboard unit puts the checkout's src/ on PYTHONPATH
    from dsart.x2 import config_diff as cdiff
    from dsart.x2 import opoint_runtime as opr
except ImportError:  # pragma: no cover - fallback for ad-hoc launches
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from dsart.x2 import config_diff as cdiff
    from dsart.x2 import opoint_runtime as opr

#: h23 ``systemctl --user`` units restarted, in this order, by a switch. C2
#: first (C3 and the bot consume its products), the plotters last.
OPOINT_RESTART_UNITS: tuple[str, ...] = (
    "dsart_c2.service",
    "dsart_c3.service",
    "dsart_inject_bot.service",
    "hiplot_c1.service",
    "hiplot_c2.service",
)

OPOINT_CHOICES: tuple[str, ...] = opr.OPOINTS
DEFAULT_HEALTH_TIMEOUT_S: float = 90.0
DEFAULT_SETTLE_S: float = 8.0
_SEARCH_CN_IDS: tuple[int, ...] = (1, 2, 9, 13)

_SWITCH_LOCK = threading.Lock()


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# systemctl --user
# ---------------------------------------------------------------------------


class SystemdUser:
    """``systemctl --user`` on the local host with a robust environment.

    ``run`` / ``environ`` / ``uid`` / ``path_exists`` are injectable for tests.
    Methods never raise; failures come back as ``ok=False`` with ``err``.
    """

    def __init__(
        self,
        *,
        run: Callable[..., Any] = subprocess.run,
        environ: Mapping[str, str] | None = None,
        uid: int | None = None,
        path_exists: Callable[[str], bool] = os.path.exists,
    ) -> None:
        self._run = run
        self._environ = environ
        self._uid = os.getuid() if uid is None else int(uid)
        self._exists = path_exists

    def env(self) -> dict[str, str]:
        e = dict(os.environ if self._environ is None else self._environ)
        if not e.get("XDG_RUNTIME_DIR"):
            cand = f"/run/user/{self._uid}"
            if self._exists(cand):
                e["XDG_RUNTIME_DIR"] = cand
        if not e.get("DBUS_SESSION_BUS_ADDRESS") and e.get("XDG_RUNTIME_DIR"):
            bus = os.path.join(e["XDG_RUNTIME_DIR"], "bus")
            if self._exists(bus):
                e["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
        return e

    def systemctl(self, *args: str, timeout: float = 10.0) -> dict[str, Any]:
        argv = ["systemctl", "--user", *args]
        t0 = time.monotonic()
        try:
            cp = self._run(argv, capture_output=True, text=True, timeout=timeout,
                           env=self.env())
            ok = cp.returncode == 0
            return {"argv": argv, "ok": ok, "rc": cp.returncode,
                    "stdout": cp.stdout or "", "stderr": cp.stderr or "",
                    "err": "" if ok else f"rc={cp.returncode}: {(cp.stderr or '').strip()[:300]}",
                    "elapsed_s": round(time.monotonic() - t0, 3)}
        except subprocess.TimeoutExpired:
            return {"argv": argv, "ok": False, "rc": None, "stdout": "", "stderr": "",
                    "err": f"timeout after {timeout:.0f}s",
                    "elapsed_s": round(time.monotonic() - t0, 3)}
        except (FileNotFoundError, OSError) as exc:
            return {"argv": argv, "ok": False, "rc": None, "stdout": "", "stderr": "",
                    "err": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - t0, 3)}

    def available(self, timeout: float = 5.0) -> dict[str, Any]:
        """Is the user manager reachable from this process?"""
        r = self.systemctl("show", "--property=Version", timeout=timeout)
        if r["ok"] and "Version=" not in r["stdout"]:
            r = dict(r, ok=False, err="unexpected output from systemctl --user show")
        return r

    def restart(self, unit: str, timeout: float = 60.0) -> dict[str, Any]:
        return dict(self.systemctl("restart", unit, timeout=timeout), unit=unit)

    def show(self, unit: str, timeout: float = 5.0) -> dict[str, str]:
        r = self.systemctl(
            "show", "--no-pager",
            "--property=ActiveState", "--property=SubState",
            "--property=MainPID", "--property=NRestarts", "--property=Result",
            unit, timeout=timeout,
        )
        props: dict[str, str] = {}
        for line in (r.get("stdout") or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
        if not r["ok"]:
            props.setdefault("_err", r["err"])
        return props


def wait_units_healthy(
    sd: SystemdUser,
    units: Iterable[str],
    *,
    old_pids: Mapping[str, int] | None = None,
    timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    settle_s: float = DEFAULT_SETTLE_S,
    poll_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait until every unit is active/running on a NEW MainPID that stays the
    same (and NRestarts unchanged) for ``settle_s`` — a crash-looping unit
    (Restart=on-failure) is momentarily "active" but never settles."""
    units = list(units)
    old = dict(old_pids or {})
    t0 = clock()
    since: dict[str, tuple[float, int, str]] = {}
    status: dict[str, dict[str, Any]] = {}
    while True:
        all_ok = True
        for u in units:
            p = sd.show(u)
            try:
                pid = int(p.get("MainPID") or 0)
            except ValueError:
                pid = 0
            nr = p.get("NRestarts", "")
            running = (p.get("ActiveState") == "active" and p.get("SubState") == "running"
                       and pid > 0)
            replaced = not old.get(u) or pid != old.get(u)
            now = clock()
            if running and replaced:
                prev = since.get(u)
                if prev is None or prev[1] != pid or prev[2] != nr:
                    since[u] = (now, pid, nr)
                stable_for = now - since[u][0]
                healthy = stable_for >= settle_s
            else:
                since.pop(u, None)
                stable_for = 0.0
                healthy = False
            status[u] = {"healthy": healthy, "active": p.get("ActiveState"),
                         "sub": p.get("SubState"), "pid": pid, "nrestarts": nr,
                         "replaced": replaced, "stable_s": round(stable_for, 1),
                         "err": p.get("_err", "")}
            all_ok = all_ok and healthy
        if all_ok:
            return {"ok": True, "units": status, "elapsed_s": round(clock() - t0, 1)}
        if clock() - t0 > timeout_s:
            bad = [u for u, s in status.items() if not s["healthy"]]
            return {"ok": False, "units": status, "elapsed_s": round(clock() - t0, 1),
                    "error": f"not healthy after {timeout_s:.0f}s: {', '.join(bad)}"}
        sleep(poll_s)


# ---------------------------------------------------------------------------
# state reads
# ---------------------------------------------------------------------------


def check_fleet_stopped(
    store: Any,
    *,
    compute_state: Callable[[Any], dict[str, Any]] | None = None,
    search_cn_ids: Iterable[int] = _SEARCH_CN_IDS,
) -> dict[str, Any]:
    """READY (orchestrators up, pipeline stopped) on corr AND search.

    ``compute_system_state`` decides "running" from the corr orchestrators
    only, so the search orchestrators' own ``state`` is checked here too.
    """
    import control_store as cs_mod

    st = (compute_state or cs_mod.compute_system_state)(store)
    if st.get("state") != "ready":
        return {"ok": False, "system_state": st,
                "error": (f"fleet is {st.get('label', st.get('state'))} "
                          f"({st.get('detail', '')}); an op-point switch needs the "
                          "fleet stopped (READY) - stop it first")}
    now_mjd = time.time() / 86400.0 + 40587.0
    running = []
    for cn in search_cn_ids:
        try:
            d = store.get_dict(f"/mon/search_rt/{cn}")
        except Exception:  # noqa: BLE001
            d = None
        if cs_mod._is_fresh(d, now_mjd) and str(d.get("state")) in ("running", "starting"):
            running.append(cn)
    if running:
        return {"ok": False, "system_state": st,
                "error": f"search orchestrators still running on cn {running}; stop the fleet first"}
    return {"ok": True, "system_state": st}


def read_live(store: Any) -> dict[str, Any]:
    return {inst: store.get_dict(opr.LIVE_KEYS[inst]) for inst in opr.INSTANCES}


_BANNER_CACHE: dict[str, Any] = {"t": -1e9, "val": None}


def opoint_banner(store: Any, *, ttl_s: float = 15.0,
                  clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Active op-point for the Control banner (cached ``ttl_s``)."""
    now = clock()
    if _BANNER_CACHE["val"] is not None and now - _BANNER_CACHE["t"] < ttl_s:
        return dict(_BANNER_CACHE["val"], switching=_SWITCH_LOCK.locked())
    active = opr.resolve_active_opoint(store, retries=1, use_cache=False)
    live = read_live(store)
    stamps = {inst: opr.config_stamp(cfg) for inst, cfg in live.items()}
    consistent = all(opr.stamp_matches(cfg, active.name) for cfg in live.values()
                     if cfg is not None)
    val = {"active": active.name, "source": active.source, "warning": active.warning,
           "live_stamps": stamps, "consistent": consistent,
           "choices": list(OPOINT_CHOICES)}
    _BANNER_CACHE.update(t=now, val=val)
    return dict(val, switching=_SWITCH_LOCK.locked())


def _h23_files_report(op: str, configs_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Problems with the h23 files the restarted services read, + their parsed search_rt."""
    problems: list[str] = []
    parsed: dict[str, Any] = {}
    for kind in ("search_rt", "c2_criteria"):
        try:
            path = opr.config_path(kind, op, configs_dir)
        except FileNotFoundError as exc:
            problems.append(f"h23 {exc}")
            continue
        try:
            cfg = opr.load_yaml(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"h23 {path}: unreadable ({exc})")
            continue
        parsed[kind] = cfg
        rep = opr.validate_config(cfg, op, kind)
        if not rep.ok:
            problems.append(f"h23 {path}: {rep.summary()}")
    return problems, parsed


def opoint_status(store: Any, *, configs_dir: Path = H23_CONFIGS_DIR) -> dict[str, Any]:
    """Full status for ``GET /control/opoint``: active op-point + readiness of each set."""
    out = opoint_banner(store, ttl_s=0.0)
    sets: dict[str, Any] = {}
    for op in OPOINT_CHOICES:
        cfgs = {inst: store.get_dict(opr.set_key(op, inst)) for inst in opr.INSTANCES}
        meta = store.get_dict(opr.meta_key(op))
        pushed = all(v is not None for v in cfgs.values())
        rep = opr.validate_set(cfgs, op, require_stamp=True) if pushed else None
        h23_problems, _ = _h23_files_report(op, configs_dir)
        sets[op] = {
            "pushed": pushed,
            "meta": {k: meta.get(k) for k in ("pushed_utc", "user", "git_sha", "source")}
            if isinstance(meta, dict) else None,
            "switchable": bool(pushed and rep is not None and rep.ok and not h23_problems),
            "problems": (rep.summary() if rep is not None and not rep.ok
                         else ("" if pushed else "not pushed (push_dsart_to_etcd.py --opoint "
                               f"{op} --write)")),
            "h23_problems": h23_problems,
        }
    out["sets"] = sets
    return out


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


def _put_all(store: Any, items: list[tuple[str, Any]]) -> str:
    fn = getattr(store, "put_dicts_atomic", None)
    if callable(fn):
        return str(fn(items) or "atomic")
    for k, v in items:
        store.put_dict(k, v)
    return "sequential"


def switch_opoint(
    store: Any,
    target: str,
    *,
    user: str | None = None,
    reapply: bool = False,
    systemd: SystemdUser | None = None,
    compute_state: Callable[[Any], dict[str, Any]] | None = None,
    units: Iterable[str] = OPOINT_RESTART_UNITS,
    configs_dir: Path = H23_CONFIGS_DIR,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    settle_s: float = DEFAULT_SETTLE_S,
    poll_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Make ``target`` the active op-point (see module docstring). Never raises.

    Returns ``{ok, switched, from, to, steps, error?, warnings}``.
    """
    import control_store as cs_mod

    units = list(units)
    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    res: dict[str, Any] = {"ok": False, "switched": False, "to": target, "from": None,
                           "steps": steps, "warnings": warnings}

    def _audit(step: str, ok: bool, val: Any = None, note: str = "") -> None:
        steps.append({"step": step, "ok": ok, "val": val, "note": note})
        cs_mod.audit_log(store, namespace="opoint", cn_target="h23+fleet",
                         cmd=f"opoint_{step}", val=val, ok=ok,
                         note=f"{res['from']}->{target}: {note}"[:900], user=user)

    def _refuse(step: str, msg: str, val: Any = None) -> dict[str, Any]:
        _audit(step, False, val, msg)
        res["error"] = msg
        return res

    try:
        opr.validate_name(target)
    except opr.OpPointError as exc:
        res["error"] = str(exc)
        return res
    if not _SWITCH_LOCK.acquire(blocking=False):
        res["error"] = "another op-point switch is in progress"
        return res
    try:
        active = opr.resolve_active_opoint(store, retries=2, retry_sleep_s=0.5,
                                           use_cache=False, sleep_fn=sleep)
        if active.source == "default":
            return _refuse("read", f"cannot read {opr.ACTIVE_KEY}: {active.warning}")
        res["from"] = active.name
        live = read_live(store)
        if target == active.name and not reapply:
            bad = [inst for inst, cfg in live.items()
                   if cfg is not None and not opr.stamp_matches(cfg, target)]
            if bad:
                return _refuse(
                    "consistency",
                    f"/cnf/opoint is {target} but live "
                    + ", ".join(f"/cnf/{i} is stamped {opr.config_stamp(live[i])!r}" for i in bad)
                    + "; tick 're-apply op-point' to rewrite the live configs from "
                    f"/cnf/opoints/{target}")
            res["ok"] = True
            _audit("unchanged", True, {"opoint": target}, "no switch needed")
            return res

        # 1. fleet stopped
        fs = check_fleet_stopped(store, compute_state=compute_state)
        if not fs["ok"]:
            return _refuse("fleet_stopped", fs["error"],
                           {"state": fs["system_state"].get("state")})
        _audit("fleet_stopped", True, {"counts": fs["system_state"].get("counts")})

        # 2. systemctl --user reachable
        sd = systemd or SystemdUser()
        av = sd.available()
        if not av["ok"]:
            return _refuse("systemd", f"systemctl --user is not usable from the dashboard "
                                      f"process: {av['err']}")

        # 3. target set stored + clean; h23 files clean
        new = {inst: store.get_dict(opr.set_key(target, inst)) for inst in opr.INSTANCES}
        missing = [opr.set_key(target, i) for i, v in new.items() if v is None]
        if missing:
            return _refuse("validate", f"{', '.join(missing)} not in etcd; push it with "
                                       f"tools/ops/push_dsart_to_etcd.py --opoint {target} --write")
        rep = opr.validate_set(new, target, require_stamp=True)
        if not rep.ok:
            return _refuse("validate", f"/cnf/opoints/{target}: {rep.summary()}")
        h23_problems, h23_parsed = _h23_files_report(target, Path(configs_dir))
        if h23_problems:
            return _refuse("validate", "; ".join(h23_problems))
        meta = store.get_dict(opr.meta_key(target))
        sha_new = {inst: cdiff.canonical_sha256(cfg) for inst, cfg in new.items()}
        if "search_rt" in h23_parsed:
            try:
                h23_sha = cdiff.canonical_sha256(opr.stamp(h23_parsed["search_rt"], target))
            except opr.OpPointError:
                h23_sha = None
            if h23_sha != sha_new["search_rt"]:
                warnings.append(
                    f"h23 {opr.config_path('search_rt', target, configs_dir)} differs from "
                    f"/cnf/opoints/{target}/search_rt (C2/C3/bot read the h23 file, the "
                    "search nodes the etcd copy)")
        _audit("validate", True, {"sha256": {k: v[:16] for k, v in sha_new.items()}},
               "; ".join(warnings))

        # 4. backup
        prev_active = store.get_dict(opr.ACTIVE_KEY)
        if prev_active is None:
            prev_active = {"opoint": opr.DEFAULT_OPOINT, "note": "was unset"}
        if any(live[i] is None for i in opr.INSTANCES):
            return _refuse("backup", "a live config key is empty; refusing to switch "
                                     "without a backup")
        backup = [(opr.backup_key(i), live[i]) for i in opr.INSTANCES]
        backup.append((opr.backup_key("opoint"), prev_active))
        _put_all(store, backup)
        _audit("backup", True, {"to": opr.BACKUP_PREFIX,
                                "sha256": {i: cdiff.canonical_sha256(live[i])[:16]
                                           for i in opr.INSTANCES}})
        restore_items = [(opr.LIVE_KEYS[i], live[i]) for i in opr.INSTANCES]
        restore_items.append((opr.ACTIVE_KEY, prev_active))

        def _rollback(reason: str, *, restart: bool = True) -> dict[str, Any]:
            rb: dict[str, Any] = {}
            try:
                _put_all(store, restore_items)
                rb["etcd"] = "restored"
            except Exception as exc:  # noqa: BLE001
                rb["etcd"] = f"RESTORE FAILED: {exc!r} (backup in {opr.BACKUP_PREFIX})"
            if restart:
                pids = {u: _pid(sd, u) for u in units}
                rb["restart"] = {u: sd.restart(u)["ok"] for u in units}
                rb["health"] = wait_units_healthy(
                    sd, units, old_pids=pids, timeout_s=health_timeout_s,
                    settle_s=settle_s, poll_s=poll_s, clock=clock, sleep=sleep)["ok"]
            res["rollback"] = rb
            res["switched"] = False
            return _refuse("rollback", f"{reason}; rolled back to {res['from']} ({rb})", rb)

        # 5. write live keys
        record = {"opoint": target, "set_utc": _iso_now(), "user": user or "unknown",
                  "host": socket.gethostname(), "previous": active.name,
                  "sha256": sha_new,
                  "set_meta": ({k: meta.get(k) for k in ("pushed_utc", "git_sha", "source", "user")}
                               if isinstance(meta, dict) else None)}
        items = [(opr.LIVE_KEYS[i], new[i]) for i in opr.INSTANCES]
        items.append((opr.ACTIVE_KEY, record))
        try:
            how = _put_all(store, items)
            back = read_live(store)
            if any(cdiff.canonical_sha256(back[i]) != sha_new[i] for i in opr.INSTANCES):
                raise RuntimeError("read-back of the live configs differs from the set")
            if opr.parse_active_record(store.get_dict(opr.ACTIVE_KEY)) != target:
                raise RuntimeError(f"read-back of {opr.ACTIVE_KEY} is not {target}")
        except Exception as exc:  # noqa: BLE001
            # nothing restarted yet: the services still run the previous op-point
            return _rollback(f"live write failed: {exc!r}", restart=False)
        _audit("write", True, {"keys": [k for k, _ in items], "how": how})
        res["switched"] = True

        # 6. restart + health
        pids = {u: _pid(sd, u) for u in units}
        restarts = [sd.restart(u) for u in units]
        failed = [r["unit"] for r in restarts if not r["ok"]]
        _audit("restart_units", not failed,
               {r["unit"]: r["ok"] for r in restarts},
               "; ".join(f"{r['unit']}: {r['err']}" for r in restarts if not r["ok"]))
        if failed:
            return _rollback(f"systemctl --user restart failed for {', '.join(failed)}")
        health = wait_units_healthy(sd, units, old_pids=pids, timeout_s=health_timeout_s,
                                    settle_s=settle_s, poll_s=poll_s, clock=clock, sleep=sleep)
        res["health"] = health
        _audit("health", health["ok"],
               {u: s["healthy"] for u, s in health["units"].items()},
               health.get("error", f"healthy in {health['elapsed_s']}s"))
        if not health["ok"]:
            return _rollback(health["error"])
        _BANNER_CACHE["val"] = None
        res["ok"] = True
        return res
    except Exception as exc:  # noqa: BLE001 - surface, never raise into Flask
        LOG.exception("op-point switch failed")
        res["error"] = f"unexpected error: {exc!r}"
        try:
            _audit("error", False, None, res["error"])
        except Exception:  # noqa: BLE001
            pass
        return res
    finally:
        _SWITCH_LOCK.release()


def _pid(sd: SystemdUser, unit: str) -> int:
    try:
        return int(sd.show(unit).get("MainPID") or 0)
    except ValueError:
        return 0
