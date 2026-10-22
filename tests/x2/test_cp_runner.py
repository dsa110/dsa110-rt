"""tools/x2/cp_runner.py + cp_runner_plans.py: dry run, guards, abort, checks (no ssh, no etcd)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "tools" / "x2", REPO / "tools" / "dashboard" / "dsa_monitor"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cp_runner as cr  # noqa: E402
import cp_runner_plans as cpp  # noqa: E402

CTX = {"checkpoint": "cp1", "tag": "cp1_test", "repo": str(REPO), "scratch": "/home/ubuntu/x2_scratch/cp1_test",
       "state": "/tmp/x2cp/cp1_test", "py_h23": "PYH23", "py": "PYNODE", "pyflakes": "PYFLAKES",
       "corr_node": "n04", "search_node": "n09", "live_format": "2x-gridded-ns2-cint4", "format": None,
       "corr_nodes": " ".join(cr.CORR_NODES), "search_nodes": " ".join(cr.SEARCH_NODES),
       "corr_nodes_csv": ",".join(cr.CORR_NODES), "search_nodes_csv": ",".join(cr.SEARCH_NODES),
       "all_nodes": " ".join(cr.CORR_NODES + cr.SEARCH_NODES)}


def _never(*a, **k):
    raise AssertionError("a dry run executed a command")


class FakeRun:
    """subprocess.run stand-in: records argv; scripted failures by substring."""

    def __init__(self, fail_if: tuple[str, ...] = (), stdout: str = ""):
        self.calls: list[list[str]] = []
        self.fail_if = fail_if
        self.stdout = stdout

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        text = " ".join(argv)
        rc = 1 if any(s in text for s in self.fail_if) else 0
        return subprocess.CompletedProcess(argv, rc, self.stdout, "")

    def texts(self):
        return [" ".join(c) for c in self.calls]


def _fake_repo(tmp_path: Path) -> Path:
    """A repo dir in which every placeholder prerequisite exists."""
    for rel, text in ((cpp._BENCH_CORR, "x"), (cpp._BENCH_OWNER, "x"),
                      (cpp._FAKE, "prod_frame2 frame_version send_mmsg")):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


# ------------------------------------------------------------------ the plan


def test_cp1_plan_shape():
    plan = cpp.build_cp1()
    ids = [s.id for s in plan.steps]
    assert len(ids) == len(set(ids))
    assert plan.tracks() == ["pre", "window", "corr", "search", "fleet", "restore"]
    assert plan.groups == [["window"], ["corr", "search"], ["fleet"], ["restore"]]
    for must in ("window.stop_observing", "window.stage_scratch", "corr.live_2x", "search.engine_synthetic",
                 "fleet.n01_ipfrag", "fleet.capacity_sweep", "fleet.format_choice", "fleet.search_full_load",
                 "fleet.c_sender_snap_live", "restore.etcd_unchanged", "restore.user_start",
                 "restore.search_health"):
        assert must in ids
    assert set(plan.teardown) == {"corr", "search", "fleet"}
    # plan timings: corr ~2 h, search ~1.5 h, fleet ~2 h
    b = {t: sum(s.budget_min for s in plan.track(t)) for t in plan.tracks()}
    assert 100 <= b["corr"] <= 130 and 75 <= b["search"] <= 100 and 100 <= b["fleet"] <= 130


def test_pre_track_is_read_only_and_only_ipfrag_changes_a_host():
    plan = cpp.build_cp1()
    for s in plan.track("pre"):
        assert s.touches == (), s.id                  # may run while production observes
        assert all(c.host is None for c in s.commands), s.id
    sudo = [s.id for s in plan.steps for c in s.commands if "sudo" in c.sh]
    assert sudo == ["fleet.n01_ipfrag"]
    ipf = next(s for s in plan.steps if s.id == "fleet.n01_ipfrag")
    assert ipf.confirm and "host" in ipf.touches and ipf.hosts() == ["n01"]
    # every GPU/shm/NIC/service step is inside the window and guarded on its nodes
    for s in plan.steps:
        if s.touches:
            assert s.track in ("window", "corr", "search", "fleet"), s.id
            assert s.hosts(), s.id


def test_sweep_reaches_18_gbps_and_records_pps():
    sweep = next(s for s in cpp.build_cp1().steps if s.id == "fleet.capacity_sweep")
    cmd = sweep.commands[0].sh
    assert "--rates-gbps 9,11,13,15,16,17,18" in cmd and "--record-pps" in cmd
    assert set(sweep.guard_hosts) == set(cr.CORR_NODES + cr.SEARCH_NODES)
    assert max(r["need_gbps"] for r in cpp.FORMAT_TABLE.values()) <= 18


# ------------------------------------------------------------------- dry run


def test_dry_run_prints_every_step_and_executes_nothing():
    out: list[str] = []
    r = cr.Runner(cpp.build_cp1(), CTX, dry_run=True, run_fn=_never, out=out.append)
    assert r.run(["all"]) == 0
    text = "\n".join(out)
    plan = cpp.build_cp1()
    for s in plan.steps + [t for v in plan.teardown.values() for t in v]:
        assert f"[{s.id}]" in text, s.id
    for s in plan.steps:
        if s.touches:
            block = text.split(f"[{s.id}]")[1].split("\n[")[0]
            assert "GUARD $ (h23" in block and "check fleet-stopped" in block, s.id
    assert "PLACEHOLDER" in text and "window critical path" in text
    assert "<format: set at run time>" in text            # chosen at the S1 step
    assert "sudo -n install -m 0644" in text


def test_cli_dry_run_and_abort_dry_run(capsys):
    assert cr.main(["run", "cp1", "--dry-run", "--track", "all", "--tag", "t1"]) == 0
    out = capsys.readouterr().out
    assert "[fleet.n01_ipfrag]" in out and "DRY RUN" in out
    assert cr.main(["abort", "cp1", "--dry-run", "--tag", "t1"]) == 0
    out = capsys.readouterr().out
    assert "[corr.teardown]" in out and "[restore.etcd_unchanged]" in out
    assert "[restore.user_start]" not in out              # never in an abort
    assert cr.main(["list", "cp1", "--track", "corr", "--tag", "t1"]) == 0


# --------------------------------------------------------------- real runs


def test_happy_path_order_markers_and_results(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    run = FakeRun()
    plan = cpp.build_cp1()
    yes = [s.id for s in plan.steps if s.confirm or s.manual]
    ctx = dict(CTX, format="2x-baselines-ns4-cint5")
    r = cr.Runner(plan, ctx, yes=yes, results_dir=tmp_path / "res", run_fn=run, out=lambda s: None,
                  interactive=False, repo=repo)
    assert r.run(["all"]) == 0
    doc = json.loads((tmp_path / "res" / "results.json").read_text())
    assert doc["status"] == "passed"
    main = [x for x in doc["steps"] if x["phase"] == "main"]
    assert all(x["status"] == "passed" for x in main)
    order = [x["track"] for x in main]
    assert order.index("corr") > order.index("window") and order.index("fleet") > max(
        i for i, t in enumerate(order) if t in ("corr", "search"))
    assert order[-1] == "restore"
    assert {x["track"] for x in doc["steps"] if x["phase"] == "teardown"} == {"corr", "search", "fleet"}
    texts = run.texts()
    live = [t for t in texts if "b8_corr2x.py live" in t]
    assert live and live[0].startswith("ssh -o BatchMode=yes -o ConnectTimeout=10 n04.pro.pvt export "
                                       "DSART_X2_CP=cp1_test; timeout -k 10 2700 bash -c")
    assert any("--opoint 2x-baselines-ns4-cint5" in t for t in texts)   # @{format} filled
    # the stopped-production assertion ran right before the corr live step, on h23 and on n04
    rec = next(x for x in doc["steps"] if x["id"] == "corr.live_2x")
    assert [g["host"] for g in rec["guard"]] == ["h23", "n04"]
    assert "check fleet-stopped" in rec["guard"][0]["cmd"]


def test_guard_refusal_aborts_without_touching(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    run = FakeRun(fail_if=("check fleet-stopped",))     # production is observing
    plan = cpp.build_cp1()
    r = cr.Runner(plan, CTX, yes=["window.stop_observing"], results_dir=tmp_path / "res", run_fn=run,
                  out=lambda s: None, interactive=False, repo=repo)
    assert r.run(["all"]) == 1
    doc = json.loads((tmp_path / "res" / "results.json").read_text())
    assert doc["status"].startswith("aborted")
    texts = run.texts()
    assert not any("setup.py -q build_ext" in t or "nvidia-smi -L" in t or "b8_corr2x.py live" in t
                   or "b8_owner2x.py kernels" in t or "sudo" in t for t in texts)
    phases = {x.get("phase") for x in doc["steps"]}
    assert "abort-restore" in phases
    assert not any(x["id"] == "restore.user_start" for x in doc["steps"])


def test_guard_on_a_touching_step_refuses(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    run = FakeRun(fail_if=("production process",))      # the per-node check fails (rc 1 on that cmd)
    plan = cpp.build_cp1()
    r = cr.Runner(plan, CTX, results_dir=tmp_path / "res", run_fn=run, out=lambda s: None,
                  interactive=False, repo=repo)
    assert r.run(["corr"]) == 1
    doc = json.loads((tmp_path / "res" / "results.json").read_text())
    first = doc["steps"][0]
    assert first["id"] == "corr.env_check" and first["status"] == "refused-guard"
    assert not any("nvidia-smi -L" in t for t in run.texts())
    assert any(x["id"] == "corr.teardown" for x in doc["steps"])   # single track: its own teardown


def test_placeholder_fails_a_real_run(tmp_path):
    run = FakeRun()
    r = cr.Runner(cpp.build_cp1(), CTX, results_dir=tmp_path / "res", run_fn=run, out=lambda s: None,
                  interactive=False, repo=tmp_path)             # no bench scripts here
    assert r.run(["search"]) == 1
    doc = json.loads((tmp_path / "res" / "results.json").read_text())
    st = {x["id"]: x["status"] for x in doc["steps"] if x.get("phase") == "main"}
    assert st["search.env_check"] == "passed" and st["search.kernels"] == "missing-prereq"


def test_confirmation_required_when_not_interactive(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    r = cr.Runner(cpp.build_cp1(), CTX, results_dir=tmp_path / "res", run_fn=FakeRun(),
                  out=lambda s: None, interactive=False, repo=repo, only=["fleet.n01_ipfrag"])
    assert r.run(["fleet"]) == 1
    doc = json.loads((tmp_path / "res" / "results.json").read_text())
    assert doc["steps"][0]["status"] == "declined"


def test_format_choice_validates(tmp_path):
    for fmt, ok in (("2x-gridded-ns4-cint4", True), ("2x", False), ("1x", False)):
        r = cr.Runner(cpp.build_cp1(), dict(CTX, format=fmt), run_fn=FakeRun(stdout="table"),
                      out=lambda s: None, interactive=False, repo=_fake_repo(tmp_path / fmt),
                      only=["fleet.format_choice"])
        step = next(s for s in r.plan.steps if s.id == "fleet.format_choice")
        rec = r.run_step(step)
        assert (rec["status"] == "passed") == ok, fmt


def test_interactive_format_choice(tmp_path):
    answers = iter(["2x-gridded-ns2-cint4"])
    r = cr.Runner(cpp.build_cp1(), CTX, run_fn=FakeRun(), out=lambda s: None, interactive=True,
                  input_fn=lambda prompt: next(answers), repo=tmp_path)
    step = next(s for s in r.plan.steps if s.id == "fleet.format_choice")
    assert r.run_step(step)["status"] == "passed" and r.ctx["format"] == "2x-gridded-ns2-cint4"


def test_wrap_and_prod_guard():
    r = cr.Runner(cpp.build_cp1(), CTX, run_fn=_never, out=lambda s: None)
    argv = r._wrap(cr.Command("echo hi", "@{search_node}", 30, guard=True), strict=True)
    assert argv[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    assert argv[5] == "n09.pro.pvt"
    assert argv[6] == ("export DSART_X2_CP=cp1_test; cd /home/ubuntu/x2_scratch/cp1_test && "
                       "bash tools/x2/prod_guard.sh timeout -k 10 30 bash -c 'echo hi'")
    assert r._wrap(cr.Command("ls", None, 5, mark=False), strict=True) == ["bash", "-c", "ls"]
    with pytest.raises(KeyError):
        r._wrap(cr.Command("echo @{format}"), strict=True)


def test_guard_tripped_output_fails_the_step(tmp_path):
    run = FakeRun(stdout="PRODUCTION_GUARD: tripped on n13")
    r = cr.Runner(cpp.build_cp1(), CTX, run_fn=run, out=lambda s: None, interactive=False, repo=tmp_path)
    step = cr.Step("x.guarded", "x", "t", 1, [cr.Command("true", "n13", 10, guard=True)])
    assert r.run_step(step)["status"] == "guard-tripped"


# -------------------------------------------------------------------- checks


PROG = ("2026-10-22 14:57:12,808 INFO dsart.services.search_compute: cube_progress: cubes=259910 "
        "cands=100023 clusters=0 ({rate} cubes/s last 1.9s; 4.96 cubes/s overall) "
        "stage_ms[build/l1/det/total]=218.1/36.5/32.4/287.0 build_ms[h2d/imager/valid]=15.5/37.5/0.0(n=10) "
        "scatter={scat}/55751.1 us(mean/max,n=10) sk[med/p95/max]=2.808/7.548/8.318")
SEEK = ("2026-10-22 07:26:01,000 WARNING dsart.transport.production_rx_ring: ProductionRxRingSource: "
        "lag-recovery seek from 1 to 2")


def test_search_health_signature():
    ok = cr.summarize_search_health([PROG.format(rate=5.16, scat=53861.9)] * 5)
    assert ok["ok"] and abs(ok["scatter_ms_max"] - 53.8619) < 1e-6 and ok["rate_min"] == 5.16
    one_low = [PROG.format(rate=5.1, scat=54000), PROG.format(rate=4.5, scat=54000),
               PROG.format(rate=5.1, scat=54000)]
    assert cr.summarize_search_health(one_low)["ok"]
    two_low = one_low[:2] + [PROG.format(rate=4.6, scat=54000)]
    assert not cr.summarize_search_health(two_low)["ok"]
    seek = cr.summarize_search_health([PROG.format(rate=5.1, scat=54000), SEEK])
    assert not seek["ok"] and seek["seeks"] == 1
    hot = cr.summarize_search_health([PROG.format(rate=5.1, scat=120000.0)])
    assert not hot["ok"] and "scatter" in hot["problems"][0]
    assert not cr.summarize_search_health([])["ok"]


def test_search_health_over_tail_output():
    tail = "\n".join([
        "==> /tmp/dsart-rt-children/search_rt-13-search_compute_0.log <==",
        PROG.format(rate=5.1, scat=54000), PROG.format(rate=5.0, scat=55000),
        "==> /tmp/dsart-rt-children/search_rt-13-search_compute_1.log <==",
        PROG.format(rate=5.1, scat=54000), SEEK])
    run = FakeRun(stdout=tail)
    res = cr.search_health(["n13"], seconds=1, run_fn=run, sleep=lambda s: None)
    assert not res["ok"]
    halves = res["nodes"]["n13"]["halves"]
    assert halves["search_rt-13-search_compute_0.log"]["ok"]
    assert not halves["search_rt-13-search_compute_1.log"]["ok"]
    assert run.calls[0][5] == "n13.pro.pvt" and "tail -n0 -F" in run.calls[0][6]


def _sampler(t0, pk0, disc0, n=3, pps=100000, disc_rate=0.0, iface="enp129s0f0"):
    lines = [f"# iface {iface} mtu 9000 host x"]
    for i in range(n):
        pk = pk0 + i * 10 * pps
        lines.append(f"{t0 + 10 * i} {pk * 1000} {pk} {disc0 + int(i * 10 * pps * disc_rate)} 0 0 0 0 0 0")
    return "\n".join(lines)


def test_nic_rates_and_relative_gate():
    r = cr.nic_rates(_sampler(1000.0, 0, 0, pps=122000))
    assert r["iface"] == "enp129s0f0" and abs(r["pps"] - 122000) < 1 and r["discard_frac"] == 0
    assert abs(r["gbps"] - 122000 * 8000 / 1e9) < 1e-6
    base = _sampler(0.0, 0, 0, disc_rate=3e-4)
    assert cr.nic_gate(base, _sampler(0.0, 0, 0, disc_rate=2e-4))["ok"]
    assert not cr.nic_gate(base, _sampler(0.0, 0, 0, disc_rate=1e-3))["ok"]
    assert not cr.nic_gate(base, "# iface x\n")["ok"]
    assert cr.nic_rates(_sampler(0.0, 0, 0, iface="enp129s0f1"))["iface"] == "enp129s0f1"


def test_format_table_suggestion():
    t = "\n".join(cr.format_table({"clean_gbps_per_node": {"n01": 14.5, "n02": 16.0}}))
    assert "suggested (highest total that fits; gridded on ties): 2x-baselines-ns4-cint5" in t
    t = "\n".join(cr.format_table({"clean_gbps_per_node": {"n01": 17.6, "n02": 18.0}}))
    assert t.splitlines()[-2].endswith("2x-gridded-ns4-cint5")
    t = "\n".join(cr.format_table({"clean_gbps_min": 10.0}))
    assert "NONE fits -> escalation E4" in t
    assert "fits" in cr.format_table(None)[0]


class FakeKV:
    def __init__(self, kv):
        self.kv = copy.deepcopy(kv)

    def get_dict(self, k):
        return copy.deepcopy(self.kv.get(k))

    def get_prefix_dicts(self, prefix):
        return {k: v for k, v in self.kv.items() if k.startswith(prefix)}


def test_etcd_snapshot_round_trip():
    kv = FakeKV({"/cnf/pipeline_rt": {"routines": []}, "/cnf/search_rt": {"routines": [1]},
                 "/cnf/inject/snr_calibration/dm0500": {"k": 1}})
    snap = cr.etcd_snapshot(kv)
    assert snap["active"] == "1x"
    assert cr.verify_snapshot(kv, snap) == (True, [])
    kv.kv["/cnf/inject/snr_calibration/dm0500"] = {"k": 2}
    ok, bad = cr.verify_snapshot(kv, snap)
    assert not ok and bad == ["/cnf/inject/snr_calibration/"]
    kv.kv["/cnf/inject/snr_calibration/dm0500"] = {"k": 1}
    kv.kv["/cnf/opoint"] = {"opoint": "2x-gridded-ns2-cint4"}
    ok, bad = cr.verify_snapshot(kv, snap)
    assert not ok and any("2x-gridded" in b for b in bad)


def test_check_fleet_stopped_uses_corr_and_search_state():
    now = time.time() / 86400.0 + 40587.0
    kv = FakeKV({"/mon/corr_rt/3": {"time_mjd": now, "state": "stopped"},
                 "/mon/search_rt/1": {"time_mjd": now, "state": "stopped"}})
    ok, msg = cr.check_fleet_stopped(kv)
    assert ok and msg.startswith("READY")
    kv.kv["/mon/search_rt/1"]["state"] = "running"
    ok, msg = cr.check_fleet_stopped(kv)
    assert not ok and "[1]" in msg
    kv.kv["/mon/corr_rt/3"]["state"] = "running"
    ok, msg = cr.check_fleet_stopped(kv)
    assert not ok and "stopped" in msg


@pytest.mark.parametrize("name", ["cp_runner_nic_sample.sh", "cp_runner_vm_sample.sh"])
def test_node_sampler_scripts_are_read_only(name):
    text = (REPO / "tools" / "x2" / name).read_text()
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for word in ("sudo", "sysctl -w", "ip link set", "ethtool -K", "ethtool -G", "rm ", "> /", "tee "):
        assert word not in body, word
    if name == "cp_runner_nic_sample.sh":
        assert "10\\.41\\.0\\." in body and "brif" in body


def test_vm_rates_and_e5_steps():
    lines = ["# numa_balancing 1 host n13"]
    for i in range(3):
        # t hint local migrated pgfault pgmaj user system idle
        lines.append(f"{100 + 10 * i} {i * 31000} {i * 30000} {i * 50} {i * 900000} {i} "
                     f"{1000 + i * 800} {500 + i * 100} {9000 + i * 3100}")
    r = cr.vm_rates("\n".join(lines))
    assert r["ok"] and r["numa_balancing"] == "1"
    assert abs(r["numa_hint_faults_per_s"] - 3100) < 1e-9
    assert abs(r["system_frac"] - 200 / (1600 + 200 + 6200)) < 1e-9
    assert not cr.vm_rates("# numa_balancing 1\n")["ok"]
    plan = cpp.build_cp1()
    full = next(s for s in plan.steps if s.id == "fleet.search_full_load")
    assert any("cp_runner_vm_sample.sh" in c.sh for c in full.commands)
    assert any("vm-summary" in c.sh for c in full.commands)
    eng = next(s for s in plan.steps if s.id == "search.engine_synthetic")
    assert "cp_runner_vm_sample.sh" in eng.commands[0].sh
    assert any(s.id == "pre.vm_baseline_1x" and not s.touches for s in plan.steps)
