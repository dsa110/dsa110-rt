"""Web-UI Start control op-point switch (SPEC §10) with fake etcd and systemctl."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from dsart.x2 import opoint_runtime as opr
from dsart.x2.opoint import opoint_from_id
from dsart.x2.plan import build_tiered_plan_for_opoint

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
DSA_MONITOR_DIR = REPO / "tools" / "dashboard" / "dsa_monitor"
if str(DSA_MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(DSA_MONITOR_DIR))

import control_store  # noqa: E402
import opoint_switch as osw  # noqa: E402

NS2 = "2x-gridded-ns2-cint4"
UNITS = osw.OPOINT_RESTART_UNITS
K1 = {"/cnf/inject/snr_calibration/dm0500": {"k": 2890.0},
      "/cnf/inject/snr_calibration/dm0500@w0032": {"k": 1000.0}}


# ------------------------------------------------------------------ fakes


class FakeKV:
    """etcd stand-in with the ControlStore surface the switch uses."""

    def __init__(self, kv=None):
        self.kv = copy.deepcopy(kv or {})
        self.ops: list[tuple[str, str]] = []
        self.fail_apply_containing: str | None = None

    def get_dict(self, key):
        return copy.deepcopy(self.kv.get(key))

    def put_dict(self, key, value):
        self.ops.append(("put", key))
        self.kv[key] = copy.deepcopy(value)

    def delete(self, key):
        self.ops.append(("del", key))
        self.kv.pop(key, None)

    def get_prefix_dicts(self, prefix):
        return {k: copy.deepcopy(v) for k, v in self.kv.items() if k.startswith(prefix)}

    def apply_atomic(self, puts, deletes=()):
        keys = [k for k, _ in puts] + list(deletes)
        assert len(keys) == len(set(keys)), "duplicate key in one txn"
        if self.fail_apply_containing and any(self.fail_apply_containing == k for k in keys):
            self.fail_apply_containing = None
            raise RuntimeError("etcd txn failed")
        for k, v in puts:
            self.put_dict(k, v)
        for k in deletes:
            self.delete(k)
        return "txn"

    def audit_cmds(self):
        return [self.kv[k]["cmd"] for _op, k in self.ops
                if _op == "put" and k.startswith("/mon/audit/control/")]

    def live_writes(self):
        return [k for op, k in self.ops if k in ("/cnf/pipeline_rt", "/cnf/search_rt", "/cnf/opoint")]


class FakeSystemd:
    def __init__(self, *, fail_restart=(), crashloop=(), available=True, env_has_seed=True):
        self.calls: list[tuple[str, str]] = []
        self.pid = {u: 100 + i for i, u in enumerate(UNITS)}
        self.active = {u: True for u in UNITS}
        self.next_pid = 1000
        self.fail_restart = set(fail_restart)
        self.crashloop = set(crashloop)
        self._available = available
        self.env_has_seed = env_has_seed

    def available(self):
        return {"ok": self._available, "err": "" if self._available else "Failed to connect to bus"}

    def restart(self, unit):
        self.calls.append(("restart", unit))
        if unit in self.fail_restart:
            return {"unit": unit, "ok": False, "err": "rc=1: Job failed"}
        self.next_pid += 1
        self.pid[unit] = self.next_pid
        self.active[unit] = True
        return {"unit": unit, "ok": True, "err": ""}

    def stop(self, unit):
        self.calls.append(("stop", unit))
        self.active[unit] = False
        self.pid[unit] = 0
        return {"unit": unit, "ok": True, "err": ""}

    def show(self, unit):
        if unit in self.crashloop and self.active[unit]:
            self.next_pid += 1
            self.pid[unit] = self.next_pid
        if not self.active[unit]:
            return {"ActiveState": "inactive", "SubState": "dead", "MainPID": "0", "NRestarts": "0"}
        return {"ActiveState": "active", "SubState": "running",
                "MainPID": str(self.pid[unit]), "NRestarts": "0"}

    def environment(self, unit):
        return "Environment=PYTHONPATH=/x" + (" PYTHONHASHSEED=0" if self.env_has_seed else "")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(float(s), 0.01)


READY = {"state": "ready", "label": "READY", "detail": "stopped", "counts": {"corr_up": 16}}


def _ready(_store):
    return dict(READY)


# --------------------------------------------------------------- set fixtures


@pytest.fixture(scope="module")
def env2x(tmp_path_factory):
    """Clean h23 configs (2x template without placeholders), a plan file, rendered sets."""
    d = tmp_path_factory.mktemp("h23")
    cfg = d / "configs"
    shutil.copytree(CONFIGS, cfg, ignore=shutil.ignore_patterns("*.npz", "dm_plans"))
    for f in (cfg / "opoints" / "2x").glob("*.yaml"):
        doc = yaml.safe_load(f.read_text())
        doc.pop("x2_todo", None)
        f.write_text(opr.PLACEHOLDER_RE.sub("7", yaml.safe_dump(doc, sort_keys=False)))
    plan = build_tiered_plan_for_opoint(opoint_from_id(NS2))
    plan_path = d / "plan_ns2.npz"
    plan.to_npz(plan_path)
    sets = {
        NS2: opr.render_repo_set(NS2, cfg, plan=plan, plan_path=str(plan_path)),
        "1x": opr.render_repo_set("1x", cfg),
        "1x-legacy": opr.render_repo_set("1x-legacy", cfg),
    }
    for sel, s in sets.items():
        rep = opr.validate_set(s, sel)
        assert rep.ok, (sel, rep.summary())
    return {"configs": cfg, "plan": plan, "plan_path": plan_path, "sets": sets}


def _kv_with_sets(env, *, live_sel="1x", active=None, k=None, extra=None):
    kv = {}
    for sel, s in env["sets"].items():
        for inst in opr.INSTANCES:
            kv[opr.set_key(sel, inst)] = s[inst]
        kv[opr.meta_key(sel)] = {"pushed_utc": "t", "git_sha": "abc"}
    base = {inst: yaml.safe_load((env["configs"] / opr.CONFIG_FILES[inst]).read_text())
            for inst in opr.INSTANCES}
    live = base if live_sel == "unstamped" else env["sets"][live_sel]
    for inst in opr.INSTANCES:
        kv[opr.LIVE_KEYS[inst]] = live[inst]
    if active is not None:
        kv[opr.ACTIVE_KEY] = active
    kv.update(K1 if k is None else k)
    kv.update(extra or {})
    return FakeKV(kv)


def _switch(store, target, sd, env, clock=None, **kw):
    clock = clock or Clock()
    return osw.switch_opoint(store, target, user="ops", systemd=sd, compute_state=_ready,
                             configs_dir=env["configs"], health_timeout_s=30, settle_s=3,
                             poll_s=1.0, clock=clock, sleep=clock.sleep, **kw)


# ------------------------------------------------------------------ switch


def test_same_opoint_is_a_plain_start(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    sd = FakeSystemd()
    res = _switch(s, "1x", sd, env2x)
    assert res["ok"] and not res["switched"] and res["from"] == "1x"
    assert s.live_writes() == [] and sd.calls == []
    assert s.audit_cmds() == ["opoint_unchanged"]


def test_same_opoint_inconsistent_live_is_refused(env2x):
    s = _kv_with_sets(env2x, live_sel=NS2)            # /cnf/opoint unset (=1x) but live is 2x
    res = _switch(s, "1x", FakeSystemd(), env2x)
    assert not res["ok"] and "re-apply" in res["error"]


def test_switch_1x_to_2x_and_back(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    before_live = {k: s.kv[k] for k in opr.LIVE_KEYS.values()}
    sd = FakeSystemd()
    res = _switch(s, NS2, sd, env2x)
    assert res["ok"] and res["switched"], res.get("error")
    assert (res["from"], res["to"]) == ("1x", NS2)
    # live configs = the stored set; record carries id + plan
    for inst in opr.INSTANCES:
        assert s.kv[opr.LIVE_KEYS[inst]] == env2x["sets"][NS2][inst]
    rec = s.kv[opr.ACTIVE_KEY]
    assert rec["opoint"] == NS2 and rec["selection"] == NS2 and rec["config_set"] == "2x"
    assert rec["plan_id"] == f"0x{env2x['plan'].plan_id:016x}"
    assert rec["plan_path"] == str(env2x["plan_path"]) and rec["previous"] == "1x"
    assert opr.parse_active_record(rec) == NS2
    # backups of the previous live state
    for inst in opr.INSTANCES:
        assert s.kv[opr.backup_key(inst)] == before_live[opr.LIVE_KEYS[inst]]
    assert s.kv[opr.backup_key("opoint")] == {"unset": True}
    # 1x K parked, no 1x K live
    assert not any(k.startswith(opr.K_PREFIX) for k in s.kv)
    assert s.kv["/cnf/opoints/_kpark/1x/dm0500"] == {"k": 2890.0}
    assert res["k_park"] == {"from_family": "1x", "family": "2x", "parked": 2, "restored": 0}
    # bot stopped before the write, then every unit restarted in order
    assert sd.calls[0] == ("stop", "dsart_inject_bot.service")
    assert [u for c, u in sd.calls if c == "restart"] == list(UNITS)
    assert res["health"]["ok"]
    assert s.audit_cmds() == ["opoint_fleet_stopped", "opoint_validate", "opoint_backup",
                              "opoint_write", "opoint_restart_units", "opoint_health"]
    assert res["warnings"] == []

    # a 2x K appears, then switch back to 1x: 1x K restored, 2x K parked
    s.kv[opr.K_PREFIX + "dm0500@w0016"] = {"k": 77.0}
    sd2 = FakeSystemd()
    res = _switch(s, "1x", sd2, env2x)
    assert res["ok"] and res["switched"], res.get("error")
    assert s.get_prefix_dicts(opr.K_PREFIX) == K1
    assert s.get_prefix_dicts("/cnf/opoints/_kpark/2x/") == {
        "/cnf/opoints/_kpark/2x/dm0500@w0016": {"k": 77.0}}
    assert s.get_prefix_dicts("/cnf/opoints/_kpark/1x/") == {}
    assert s.kv[opr.LIVE_KEYS["search_rt"]]["opoint_selection"] == "1x"


def test_1x_to_1x_legacy_keeps_k(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    res = _switch(s, "1x-legacy", FakeSystemd(), env2x)
    assert res["ok"] and res["switched"], res.get("error")
    assert s.get_prefix_dicts(opr.K_PREFIX) == K1
    assert res["k_park"]["parked"] == 0
    assert s.kv[opr.ACTIVE_KEY]["opoint"] == "1x"
    assert s.kv[opr.ACTIVE_KEY]["selection"] == "1x-legacy"


@pytest.mark.parametrize("state", ["observing", "prepared", "preparing", "offline"])
def test_refused_unless_stopped(env2x, state):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    sd = FakeSystemd()
    res = osw.switch_opoint(s, NS2, user="ops", systemd=sd, configs_dir=env2x["configs"],
                            compute_state=lambda _s: {"state": state, "label": state.upper(),
                                                      "detail": "x"})
    assert not res["ok"] and "stopped" in res["error"]
    assert s.live_writes() == [] and sd.calls == []


def test_refused_when_a_search_node_runs(env2x):
    now_mjd = time.time() / 86400.0 + 40587.0
    s = _kv_with_sets(env2x, live_sel="unstamped",
                      extra={"/mon/search_rt/9": {"time_mjd": now_mjd, "state": "running"}})
    res = _switch(s, NS2, FakeSystemd(), env2x)
    assert not res["ok"] and "[9]" in res["error"]


def test_refused_when_set_missing_or_dirty(env2x, tmp_path):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    for inst in opr.INSTANCES:
        del s.kv[opr.set_key(NS2, inst)]
    res = _switch(s, NS2, FakeSystemd(), env2x)
    assert not res["ok"] and "push_dsart_to_etcd.py --opoint" in res["error"]
    # a set with placeholders (a template that still owes a value) is refused
    plan_path = tmp_path / "p.npz"
    env2x["plan"].to_npz(plan_path)
    dirty = opr.render_repo_set(NS2, _dirty_configs(tmp_path), plan=env2x["plan"],
                                plan_path=str(plan_path))
    for inst in opr.INSTANCES:
        s.kv[opr.set_key(NS2, inst)] = dirty[inst]
    res = _switch(s, NS2, FakeSystemd(), env2x)
    assert not res["ok"] and "placeholder" in res["error"]
    assert s.live_writes() == []


def _dirty_configs(tmp_path) -> Path:
    """Copy of configs/ whose 2x template still owes a value (X2TODO token + x2_todo entry)."""
    dst = tmp_path / "configs_dirty"
    shutil.copytree(CONFIGS, dst, ignore=shutil.ignore_patterns("*.npz", "dm_plans"))
    f = dst / "opoints" / "2x" / "dsart_search_rt.yaml"
    doc = yaml.safe_load(f.read_text())
    doc["x2_todo"] = ["WS-T: an open review"]
    doc["routines"][0]["args"] += " --rx-threads X2TODO:WS-T:threads"
    f.write_text(yaml.safe_dump(doc, sort_keys=False))
    return dst


def test_refused_when_h23_files_have_placeholders(env2x, tmp_path):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    res = osw.switch_opoint(s, NS2, user="ops", systemd=FakeSystemd(), compute_state=_ready,
                            configs_dir=_dirty_configs(tmp_path))
    assert not res["ok"] and "h23" in res["error"] and "placeholder" in res["error"]


def test_refused_when_the_h23_plan_differs(env2x, tmp_path):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    other = build_tiered_plan_for_opoint(opoint_from_id("2x-gridded-ns4-cint4"))
    p = tmp_path / "other.npz"
    other.to_npz(p)
    for inst in opr.INSTANCES:
        cfg = s.kv[opr.set_key(NS2, inst)]
        cfg["plan_path"] = str(p)
    res = _switch(s, NS2, FakeSystemd(), env2x)
    assert not res["ok"] and "plan_id" in res["error"]


def test_missing_h23_plan_is_a_warning(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    for inst in opr.INSTANCES:
        s.kv[opr.set_key(NS2, inst)]["plan_path"] = "/nonexistent/plan.npz"
    res = _switch(s, NS2, FakeSystemd(), env2x)
    assert res["ok"], res.get("error")
    assert any("not on h23" in w for w in res["warnings"])


def test_systemd_unavailable_is_refused_before_writes(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    res = _switch(s, NS2, FakeSystemd(available=False), env2x)
    assert not res["ok"] and "systemctl --user" in res["error"]
    assert s.live_writes() == []


def _assert_restored(s, before):
    for k, v in before.items():
        assert s.kv.get(k) == v, k
    assert opr.ACTIVE_KEY not in s.kv                 # was unset, is unset again


def test_restart_failure_rolls_back_exactly(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    before = {k: copy.deepcopy(s.kv[k]) for k in list(opr.LIVE_KEYS.values()) + list(K1)}
    sd = FakeSystemd(fail_restart={"dsart_c3.service"})
    res = _switch(s, NS2, sd, env2x)
    assert not res["ok"] and not res["switched"]
    assert "dsart_c3" in res["error"] and "rolled back" in res["error"]
    _assert_restored(s, before)
    assert s.get_prefix_dicts("/cnf/opoints/_kpark/") == {}
    assert res["rollback"]["etcd"] == "restored"
    assert s.audit_cmds()[-1] == "opoint_rollback"


def test_crash_loop_fails_health_and_rolls_back(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    before = {k: copy.deepcopy(s.kv[k]) for k in list(opr.LIVE_KEYS.values()) + list(K1)}
    sd = FakeSystemd(crashloop={"dsart_c2.service"})
    res = _switch(s, NS2, sd, env2x)
    assert not res["ok"] and "not healthy" in res["error"] and "dsart_c2" in res["error"]
    _assert_restored(s, before)


def test_failed_write_rolls_back_and_restarts_the_bot(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    s.fail_apply_containing = "/cnf/pipeline_rt"
    sd = FakeSystemd()
    res = _switch(s, NS2, sd, env2x)
    assert not res["ok"] and "live write failed" in res["error"]
    assert s.kv[opr.LIVE_KEYS["pipeline_rt"]].get("opoint_selection") is None
    assert ("restart", "dsart_inject_bot.service") in sd.calls     # the stopped bot comes back
    assert sd.active["dsart_inject_bot.service"]


def test_switch_lock(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    assert osw._SWITCH_LOCK.acquire(blocking=False)
    try:
        res = _switch(s, NS2, FakeSystemd(), env2x)
    finally:
        osw._SWITCH_LOCK.release()
    assert not res["ok"] and "in progress" in res["error"]


def test_pythonhashseed_warning(env2x):
    s = _kv_with_sets(env2x, live_sel="unstamped")
    res = _switch(s, NS2, FakeSystemd(env_has_seed=False), env2x)
    assert res["ok"] and any("PYTHONHASHSEED" in w for w in res["warnings"])


def test_bad_target_and_unreadable_etcd(env2x):
    assert "unknown op-point" in _switch(FakeKV(), "2x", FakeSystemd(), env2x)["error"]

    class Down(FakeKV):
        def get_dict(self, key):
            raise ConnectionError("down")

    res = _switch(Down(), NS2, FakeSystemd(), env2x)
    assert not res["ok"] and "cannot read /cnf/opoint" in res["error"]


def test_plan_k_park_ops():
    s = FakeKV({**K1, "/cnf/opoints/_kpark/2x/dm0500": {"k": 5.0},
                "/cnf/opoints/_kpark/1x/stale": {"k": 0.0}})
    kp = osw.plan_k_park(s, "1x", NS2)
    puts = dict(kp["puts"])
    assert puts["/cnf/inject/snr_calibration/dm0500"] == {"k": 5.0}          # 2x restored
    assert puts["/cnf/opoints/_kpark/1x/dm0500"] == {"k": 2890.0}            # 1x parked
    assert "/cnf/inject/snr_calibration/dm0500" not in kp["deletes"]         # overwritten, not deleted
    assert "/cnf/inject/snr_calibration/dm0500@w0032" in kp["deletes"]
    assert "/cnf/opoints/_kpark/1x/stale" in kp["deletes"]
    assert "/cnf/opoints/_kpark/2x/dm0500" in kp["deletes"]
    assert kp["prior"]["/cnf/inject/snr_calibration/dm0500"] == {"k": 2890.0}
    assert kp["prior"]["/cnf/opoints/_kpark/1x/dm0500"] is None
    assert osw.plan_k_park(s, "1x", "1x-legacy")["puts"] == []


# ------------------------------------------------------------ systemd helper


def test_systemd_env_fills_runtime_dir_and_bus():
    sd = osw.SystemdUser(environ={"PATH": "/bin"}, uid=1000,
                         path_exists=lambda p: p in ("/run/user/1000", "/run/user/1000/bus"))
    e = sd.env()
    assert e["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert e["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
    sd = osw.SystemdUser(environ={"XDG_RUNTIME_DIR": "/x", "DBUS_SESSION_BUS_ADDRESS": "y"},
                         uid=1000, path_exists=lambda p: True)
    assert sd.env()["XDG_RUNTIME_DIR"] == "/x" and sd.env()["DBUS_SESSION_BUS_ADDRESS"] == "y"
    sd = osw.SystemdUser(environ={}, uid=1000, path_exists=lambda p: False)
    assert "XDG_RUNTIME_DIR" not in sd.env()


def test_systemd_calls_and_errors():
    seen = []

    def run(argv, **kw):
        seen.append((argv, kw["env"].get("XDG_RUNTIME_DIR")))
        if argv[2] == "show" and "--property=Version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Version=237\n", "")
        if argv[2] == "show":
            return subprocess.CompletedProcess(
                argv, 0, "ActiveState=active\nSubState=running\nMainPID=42\nNRestarts=0\n", "")
        if argv[2] == "restart" and argv[3] == "bad.service":
            return subprocess.CompletedProcess(argv, 5, "", "Unit bad.service not found.")
        return subprocess.CompletedProcess(argv, 0, "", "")

    sd = osw.SystemdUser(run=run, environ={}, uid=1000, path_exists=lambda p: True)
    assert sd.available()["ok"]
    assert sd.show("dsart_c2.service")["MainPID"] == "42"
    assert sd.restart("dsart_c2.service")["ok"]
    r = sd.restart("bad.service")
    assert not r["ok"] and "not found" in r["err"]
    assert all(a[:2] == ["systemctl", "--user"] and x == "/run/user/1000" for a, x in seen)

    def boom(argv, **kw):
        raise FileNotFoundError("systemctl")

    assert not osw.SystemdUser(run=boom).available()["ok"]

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    assert "timeout" in osw.SystemdUser(run=slow).restart("x")["err"]

    def odd(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, "nothing\n", "")

    assert not osw.SystemdUser(run=odd).available()["ok"]


def test_wait_units_healthy_settles_and_detects_crash_loops():
    clock = Clock()
    sd = FakeSystemd()
    old = {u: sd.pid[u] for u in UNITS}
    for u in UNITS:
        sd.restart(u)
    r = osw.wait_units_healthy(sd, UNITS, old_pids=old, timeout_s=30, settle_s=5,
                               clock=clock, sleep=clock.sleep)
    assert r["ok"] and 5 <= r["elapsed_s"] <= 7
    clock = Clock()
    sd = FakeSystemd(crashloop={"hiplot_c1.service"})
    r = osw.wait_units_healthy(sd, UNITS, old_pids={}, timeout_s=20, settle_s=5,
                               clock=clock, sleep=clock.sleep)
    assert not r["ok"] and r["error"].endswith("hiplot_c1.service")
    # not replaced (same pid as before the restart) never counts as healthy
    clock = Clock()
    sd = FakeSystemd()
    r = osw.wait_units_healthy(sd, ["dsart_c2.service"], old_pids={"dsart_c2.service": sd.pid["dsart_c2.service"]},
                               timeout_s=10, settle_s=1, clock=clock, sleep=clock.sleep)
    assert not r["ok"]


# ---------------------------------------------------------- control_store


class _Txn:
    def put(self, k, v):
        return ("put", k, v)

    def delete(self, k):
        return ("delete", k)


class _TxnClient:
    def __init__(self):
        self.transactions = _Txn()
        self.calls = []
        self.kv = {"/a/1": b'{"x": 1}'}

    def transaction(self, compare, success, failure):
        self.calls.append(success)
        return True, []

    def get_prefix(self, prefix):
        class M:
            def __init__(self, k):
                self.key = k.encode()
        for k, v in self.kv.items():
            if k.startswith(prefix):
                yield v, M(k)


class _Dsa:
    def __init__(self, client):
        self.client = client
        self.puts, self.deletes = [], []

    def get_etcd(self):
        return self.client

    def put_dict(self, k, v):
        self.puts.append(k)

    def delete(self, k):
        self.deletes.append(k)


def test_control_store_apply_atomic_and_prefix():
    cs = control_store.ControlStore()
    client = _TxnClient()
    cs._store = _Dsa(client)
    assert cs.apply_atomic([("/a", {"v": 1})], ["/b"]) == "txn"
    assert client.calls == [[("put", "/a", json.dumps({"v": 1})), ("delete", "/b")]]
    assert cs.get_prefix_dicts("/a/") == {"/a/1": {"x": 1}}
    with pytest.raises(ValueError):
        cs.apply_atomic([("/a", {"v": float("nan")})])
    cs2 = control_store.ControlStore()
    cs2._store = _Dsa(object())
    assert cs2.apply_atomic([("/a", {"v": 1})], ["/b"]) == "sequential"
    assert cs2._store.puts == ["/a"] and cs2._store.deletes == ["/b"]


def _cs_with(kv):
    cs = control_store.ControlStore()
    cs._store = kv
    return cs


def test_control_start_fleet_refused_switch_broadcasts_nothing():
    kv = FakeKV()
    called = []
    out = control_store.control_start_fleet(
        _cs_with(kv), obs_dec_deg=None, user="ops", opoint=NS2,
        pre_start=lambda: called.append("pre") or {"ok": True},
        opoint_switcher=lambda store, op, **kw: {"ok": False, "error": "fleet is OBSERVING"})
    assert out["ok"] is False and "fleet is OBSERVING" in out["error"]
    assert called == []
    assert not any(k.startswith("/cmd/") for _op, k in kv.ops)


def test_control_start_fleet_order_switch_then_cleanup_then_start():
    kv = FakeKV()
    order = []

    def switcher(store, op, **kw):
        order.append(("switch", op, kw["reapply"]))
        return {"ok": True, "switched": True}

    def pre():
        order.append(("pre",))
        assert not any(k.startswith("/cmd/") for _op, k in kv.ops)
        return {"ok": True, "n_ok": 20}

    out = control_store.control_start_fleet(_cs_with(kv), obs_dec_deg=71.6, opoint=NS2,
                                            reapply_opoint=True, pre_start=pre,
                                            opoint_switcher=switcher)
    assert out["ok"] and out["opoint"] == NS2 and out["node_cleanup"]["n_ok"] == 20
    assert order == [("switch", NS2, True), ("pre",)]
    assert [k for _op, k in kv.ops if k.startswith("/cmd/")][0] == "/cmd/corr_rt/0"


def test_control_start_fleet_without_opoint_is_unchanged():
    kv = FakeKV()
    out = control_store.control_start_fleet(_cs_with(kv), obs_dec_deg=53.85, user="ops")
    assert out == {"ok": True, "cmd": "start", "val": 53.85, "corr_broadcast_key": "/cmd/corr_rt/0",
                   "search_fanout_keys": ["/cmd/search_rt/1", "/cmd/search_rt/2",
                                          "/cmd/search_rt/9", "/cmd/search_rt/13"]}
    assert len(kv.ops) == 6
    out = control_store.control_start_fleet(_cs_with(FakeKV()), pre_start=lambda: 1 / 0)
    assert out["ok"] and out["node_cleanup"]["ok"] is False


# ------------------------------------------------------------ banner/status


def test_banner_and_status(env2x):
    osw._BANNER_CACHE["val"] = None
    s = _kv_with_sets(env2x, live_sel=NS2, active=opr.make_active_record(NS2, plan_id="0x1"))
    b = osw.opoint_banner(s, ttl_s=0)
    assert b["active"] == NS2 and b["consistent"] and b["plan_id"] == "0x1"
    assert b["choices"][:2] == ["1x", "1x-legacy"] and NS2 in b["choices"]
    s.kv[opr.LIVE_KEYS["search_rt"]] = env2x["sets"]["1x"]["search_rt"]
    assert osw.opoint_banner(s, ttl_s=0)["consistent"] is False
    st = osw.opoint_status(s, configs_dir=env2x["configs"])
    assert st["sets"][NS2]["switchable"] and st["sets"]["1x"]["switchable"]
    assert not st["sets"]["2x-baselines-ns4-cint4"]["pushed"]
    st = osw.opoint_status(s, configs_dir=CONFIGS)
    assert not st["sets"][NS2]["switchable"] and st["sets"][NS2]["h23_problems"]
    osw._BANNER_CACHE["val"] = None


def test_k_prefix_matches_the_dashboard():
    text = (DSA_MONITOR_DIR / "inject_calibration.py").read_text()
    assert f'CALIBRATION_PREFIX: str = "{opr.K_PREFIX}"' in text


def test_dashboard_unit_runs_under_the_user_manager():
    """The dashboard is a systemd --user unit, so systemctl --user reaches the
    manager that owns C2/C3/bot/hiplot (verified live 2026-10-22)."""
    unit = (DSA_MONITOR_DIR / "dsa_monitor.service").read_text()
    assert "WantedBy=default.target" in unit
    for u in UNITS:
        if u.startswith("dsart_") or u.startswith("hiplot_"):
            assert (REPO / "systemd" / u).is_file(), u
            assert "WantedBy=default.target" in (REPO / "systemd" / u).read_text()


@pytest.mark.skipif(os.environ.get("DSART_FLASK_TESTS") != "1", reason="flask app test opt-in")
def test_start_route_rejects_unknown_opoint():  # pragma: no cover - needs flask
    pytest.importorskip("flask")
    from unittest import mock
    with mock.patch("rfi_store.RFIPoller.start", return_value=None):
        import app
    r = app.app.test_client().post("/control/start", data={"opoint": "9x"})
    assert r.status_code == 400
