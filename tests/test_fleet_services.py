"""Unit tests for the dsa_monitor Control tab's fleet-services
helpers (M7.4 Phase 8 v2).

We exercise:

* The pinned :data:`services_inventory.SERVICE_INVENTORY` shape +
  :data:`H20_HOSTNAMES` (the "never restart" pin).
* :func:`fleet_services.query_all_services_status` against mocked
  ``subprocess.run`` that fakes ``systemctl is-active`` + ``pgrep``.
* :func:`fleet_services.restart_all_services` end-to-end:
  - h20 is in NONE of the ssh fanout target lists,
  - per-host failure does not abort the rest of the fanout,
  - audit row is written on both success and failure paths via
    :func:`control_store.fleet_restart_all`,
  - the deferred ``systemctl --user restart dsa_monitor.service`` is
    scheduled via :class:`subprocess.Popen` (NOT
    :func:`subprocess.run`).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Any
from unittest import mock

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
DSA_MONITOR_DIR = os.path.normpath(os.path.join(
    HERE, "..", "tools", "dashboard", "dsa_monitor",
))
if DSA_MONITOR_DIR not in sys.path:
    sys.path.insert(0, DSA_MONITOR_DIR)

import control_store                                            # noqa: E402
import fleet_services                                           # noqa: E402
import services_inventory as inv                                # noqa: E402


# ---------------------------------------------------------------------------
# FakeDsaStore — mirrors the pattern in test_dsa_monitor_control_store.py
# so the audit-log writes can be inspected.
# ---------------------------------------------------------------------------


class FakeDsaStore:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.puts.append((key, dict(payload)))

    def get_dict(self, key: str) -> Any:
        return None


@pytest.fixture()
def fake_store_pair():
    fake = FakeDsaStore()
    cs = control_store.ControlStore()
    cs._store = fake
    return cs, fake


# ---------------------------------------------------------------------------
# Inventory shape — pins the dataclass + the "never restart" list
# ---------------------------------------------------------------------------


class TestInventoryShape:
    def test_inventory_is_nonempty_and_frozen_entries(self):
        assert len(inv.SERVICE_INVENTORY) > 0
        e = inv.SERVICE_INVENTORY[0]
        # Each entry is a frozen dataclass (immutable).
        with pytest.raises(Exception):
            e.tier = "mutated"                          # type: ignore[misc]

    def test_dsa_monitor_h23_tier_has_two_units(self):
        rows = inv.entries_by_tier(inv.TIER_DSA_MONITOR_H23)
        names = sorted(r.service for r in rows)
        assert names == ["dsa_monitor.service", "sefd_dashboard.service"]
        for r in rows:
            assert r.host == "lxd110h23"
            assert r.kind == inv.KIND_SYSTEMD_USER

    def test_coincidencer_h23_tier(self):
        rows = inv.entries_by_tier(inv.TIER_COINCIDENCER_H23)
        assert len(rows) == 1
        assert rows[0].service == "dsart_c2.service"
        assert rows[0].kind == inv.KIND_SYSTEMD_USER
        assert rows[0].host == "lxd110h23"

    def test_hiplot_calibration23_uses_lxc_kind(self):
        rows = inv.entries_by_tier(inv.TIER_HIPLOT_CALIBRATION23)
        assert len(rows) == 1
        assert rows[0].host == "calibration23"
        assert rows[0].kind == inv.KIND_LXC_SYSTEMD_USER
        assert rows[0].service == "hiplot.service"

    def test_grafana_h20_tier_has_three_units_and_is_not_restartable(self):
        rows = inv.entries_by_tier(inv.TIER_GRAFANA_H20)
        assert len(rows) == 3
        for r in rows:
            assert r.host == "lxd110h20"
            assert r.kind == inv.KIND_SYSTEMD_SYSTEM
            assert r.is_restartable() is False
        names = sorted(r.service for r in rows)
        assert names == [
            "grafana-server.service",
            "influxdb.service",
            "telegraf.service",
        ]

    def test_dsart_orch_corr_has_16_entries(self):
        rows = inv.entries_by_tier(inv.TIER_DSART_ORCH_CORR)
        assert len(rows) == 16
        for r in rows:
            assert r.kind == inv.KIND_PROCESS
            assert r.service == "dsart_rt"
            assert r.instance == "pipeline_rt"
            assert r.cn_id is not None
            assert r.host.endswith(".pro.pvt")

    def test_dsart_orch_search_has_four_entries(self):
        rows = inv.entries_by_tier(inv.TIER_DSART_ORCH_SEARCH)
        assert len(rows) == 4
        assert sorted(r.cn_id for r in rows) == [1, 2, 9, 13]
        for r in rows:
            assert r.kind == inv.KIND_PROCESS
            assert r.instance == "search_rt"

    def test_h20_hostnames_is_frozenset_with_only_h20(self):
        assert isinstance(inv.H20_HOSTNAMES, frozenset)
        assert inv.H20_HOSTNAMES == frozenset({"lxd110h20"})

    def test_restartable_entries_excludes_h20(self):
        all_hosts = {e.host for e in inv.SERVICE_INVENTORY}
        assert "lxd110h20" in all_hosts                # in inventory…
        restartable_hosts = {e.host for e in inv.restartable_entries()}
        assert "lxd110h20" not in restartable_hosts    # …but not restartable.

    def test_inventory_total_size_matches_components(self):
        # 2 (dsa_monitor_h23) + 1 (dsart_c2) + 1 (hiplot) + 3 (grafana)
        # + 16 (corr orch) + 4 (search orch) = 27
        assert len(inv.SERVICE_INVENTORY) == 27


# ---------------------------------------------------------------------------
# query_all_services_status
# ---------------------------------------------------------------------------


def _fake_completed(rc: int, stdout: str = "", stderr: str = "") -> mock.Mock:
    m = mock.Mock()
    m.returncode = rc
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestQueryStatus:
    def test_active_unit_classified_active(self):
        # Return active for is-active, empty for show.
        def fake_run(argv, **kw):
            if "is-active" in argv:
                return _fake_completed(0, "active\n")
            return _fake_completed(0, "")
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            inv_one = [inv.ServiceEntry(
                tier="t", host=inv.HOST_H23,
                service="dsa_monitor.service",
                kind=inv.KIND_SYSTEMD_USER,
            )]
            out = fleet_services.query_all_services_status(inventory=inv_one)
        assert out["n_active"] == 1
        assert out["rows"][0]["state"] == "active"

    def test_inactive_unit_classified_inactive(self):
        def fake_run(argv, **kw):
            if "is-active" in argv:
                # is-active returns rc=3 for inactive in real systemd,
                # but the classifier looks at stdout, not rc.
                return _fake_completed(3, "inactive\n")
            return _fake_completed(0, "")
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            inv_one = [inv.ServiceEntry(
                tier="t", host=inv.HOST_H23,
                service="dsa_monitor.service",
                kind=inv.KIND_SYSTEMD_USER,
            )]
            out = fleet_services.query_all_services_status(inventory=inv_one)
        assert out["rows"][0]["state"] == "inactive"

    def test_failed_unit_classified_failed(self):
        def fake_run(argv, **kw):
            if "is-active" in argv:
                return _fake_completed(3, "failed\n")
            return _fake_completed(0, "")
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            inv_one = [inv.ServiceEntry(
                tier="t", host=inv.HOST_H23,
                service="dsa_monitor.service",
                kind=inv.KIND_SYSTEMD_USER,
            )]
            out = fleet_services.query_all_services_status(inventory=inv_one)
        assert out["rows"][0]["state"] == "failed"

    def test_process_kind_uses_pgrep(self):
        # rc=0 means pgrep found a match → active. The ssh wraps it as
        # "ssh <opts> host <remote_cmd>". The remote_cmd should contain
        # 'pgrep -af'.
        observed_argvs = []

        def fake_run(argv, **kw):
            observed_argvs.append(argv)
            return _fake_completed(0, "1234 python dsart...\n")
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            inv_one = [inv.ServiceEntry(
                tier=inv.TIER_DSART_ORCH_CORR,
                host="n03.pro.pvt",
                service="dsart_rt",
                kind=inv.KIND_PROCESS,
                cn_id=3,
                instance="pipeline_rt",
            )]
            out = fleet_services.query_all_services_status(inventory=inv_one)
        assert out["rows"][0]["state"] == "active"
        assert any(
            "pgrep -af" in (a[-1] if isinstance(a, list) else "")
            for a in observed_argvs
        )

    def test_process_rc1_classified_inactive(self):
        def fake_run(argv, **kw):
            return _fake_completed(1, "", "")
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            inv_one = [inv.ServiceEntry(
                tier=inv.TIER_DSART_ORCH_CORR,
                host="n03.pro.pvt",
                service="dsart_rt",
                kind=inv.KIND_PROCESS,
                cn_id=3,
                instance="pipeline_rt",
            )]
            out = fleet_services.query_all_services_status(inventory=inv_one)
        assert out["rows"][0]["state"] == "inactive"

    def test_ssh_timeout_maps_to_unreachable(self):
        def fake_run(argv, **kw):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=5)
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            inv_one = [inv.ServiceEntry(
                tier=inv.TIER_DSART_ORCH_CORR,
                host="n03.pro.pvt",
                service="dsart_rt",
                kind=inv.KIND_PROCESS,
                cn_id=3,
                instance="pipeline_rt",
            )]
            out = fleet_services.query_all_services_status(inventory=inv_one)
        assert out["rows"][0]["state"] == "unreachable"
        assert out["n_unreachable"] == 1

    def test_mixed_results_counted_correctly(self):
        # 1 active local, 1 unreachable ssh.
        active = inv.ServiceEntry(
            tier="t", host=inv.HOST_H23,
            service="dsa_monitor.service", kind=inv.KIND_SYSTEMD_USER,
        )
        unreach = inv.ServiceEntry(
            tier="t", host="n99.pro.pvt", service="dsart_rt",
            kind=inv.KIND_PROCESS, cn_id=99, instance="pipeline_rt",
        )

        def fake_run(argv, **kw):
            # ssh argv contains "n99.pro.pvt" → timeout.
            if any("n99.pro.pvt" in a for a in argv):
                raise subprocess.TimeoutExpired(cmd=argv, timeout=5)
            if "is-active" in argv:
                return _fake_completed(0, "active\n")
            return _fake_completed(0, "")
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            out = fleet_services.query_all_services_status(
                inventory=[active, unreach],
            )
        assert out["n_active"] == 1
        assert out["n_unreachable"] == 1


# ---------------------------------------------------------------------------
# restart_all_services — the h20 exclusion pin + partial-failure pin
# ---------------------------------------------------------------------------


class _CallRecorder:
    """Record every subprocess.run argv + return a programmable result."""

    def __init__(self, default_rc: int = 0, fail_hosts: set | None = None):
        self.calls: list[list[str]] = []
        self.default_rc = default_rc
        self.fail_hosts = fail_hosts or set()
        self.lock = threading.Lock()

    def __call__(self, argv, **kw):
        with self.lock:
            self.calls.append(list(argv))
        # For ssh argvs, hostname is the position right after the
        # ssh options.
        for h in self.fail_hosts:
            if any(h in str(a) for a in argv):
                return _fake_completed(2, "", f"simulated ssh failure for {h}")
        # respawn check: if cmd contains "alive" pattern in fixture,
        # we want stdout="alive" so it counts as ok.
        if any("dsart_rt" in str(a) and "kill -0" in str(a) for a in argv):
            return _fake_completed(0, "alive\n")
        return _fake_completed(self.default_rc, "OK\n")


class TestRestartAllH20Exclusion:
    def test_h20_appears_in_no_ssh_argv(self):
        recorder = _CallRecorder(default_rc=0)
        popen_calls = []

        def fake_popen(argv, **kw):
            popen_calls.append(list(argv))
            m = mock.Mock()
            m.pid = 12345
            return m

        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            fleet_services.restart_all_services(
                self_restart_delay_s=2.0,
            )
        all_argv = " ".join(" ".join(a) for a in recorder.calls)
        assert "lxd110h20" not in all_argv
        # h20 hostnames also must not appear in any Popen argv.
        all_popen_argv = " ".join(" ".join(a) for a in popen_calls)
        assert "lxd110h20" not in all_popen_argv

    def test_only_corr_and_search_hosts_pkilled(self):
        recorder = _CallRecorder(default_rc=0)
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=1)):
            fleet_services.restart_all_services()
        # Find every ssh argv (it has "ssh" at position 0).
        ssh_hosts = set()
        for argv in recorder.calls:
            if argv and argv[0] == "ssh":
                # The host is the last arg before the remote cmd; the
                # easier path is to scan for any .pro.pvt token.
                for a in argv:
                    if isinstance(a, str) and a.endswith(".pro.pvt"):
                        ssh_hosts.add(a)
                        break
        # Should include corr (16) + search (4) = 20 unique hosts.
        assert len(ssh_hosts) == 20
        # None of them is h20.
        assert all("lxd110h20" not in h for h in ssh_hosts)


class TestCleanupNodesForStart:
    def test_fanout_targets_corr_and_search_not_h20_or_h23(self):
        recorder = _CallRecorder(default_rc=0)
        with mock.patch.object(subprocess, "run", side_effect=recorder):
            summary = fleet_services.cleanup_nodes_for_start()
        ssh_hosts = set()
        for argv in recorder.calls:
            if argv and argv[0] == "ssh":
                for a in argv:
                    if isinstance(a, str) and a.endswith(".pro.pvt"):
                        ssh_hosts.add(a)
                        break
        # corr (16) + search (4) = 20 unique hosts; all succeed.
        assert len(ssh_hosts) == 20
        assert summary["n_hosts"] == 20
        assert summary["n_ok"] == 20
        assert summary["ok"] is True
        all_argv = " ".join(" ".join(a) for a in recorder.calls)
        # h20 (read-only) + h23 (archive) are NEVER in the fanout.
        assert "lxd110h20" not in all_argv
        assert "lxd110h23" not in all_argv
        # On a START we never wipe shm rings / PSRDADA — the starting
        # processes own those.
        assert "/dev/shm" not in all_argv
        assert "dada_db" not in all_argv
        # corr clears ready sentinels + debug grid; search clears the
        # local cube_dump tree (already rsynced to h23).
        assert "dsart-corr-*.ready" in all_argv
        assert "dsart-fast-grid" in all_argv
        assert "/home/ubuntu/data/c2/cube_dump" in all_argv

    def test_partial_failure_reported_not_raised(self):
        recorder = _CallRecorder(default_rc=0, fail_hosts={"n03.pro.pvt"})
        with mock.patch.object(subprocess, "run", side_effect=recorder):
            summary = fleet_services.cleanup_nodes_for_start()
        # One host failed; the call still returns (never raises).
        assert summary["ok"] is False
        assert summary["n_failed"] == 1
        assert summary["per_host"]["n03.pro.pvt"]["rc"] == 2
        # The other 19 still succeeded.
        assert summary["n_ok"] == 19


class TestRestartAllPartialFailure:
    def test_one_host_failure_doesnt_abort_other_fanout(self):
        # n03.pro.pvt simulates a non-zero ssh exit; the other 19
        # corr/search hosts still get reached.
        recorder = _CallRecorder(default_rc=0, fail_hosts={"n03.pro.pvt"})
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=1)):
            summary = fleet_services.restart_all_services()
        # Overall ok flag flipped False.
        assert summary["ok"] is False
        # The cleanup step has per-host results for ~20 hosts (corr +
        # search), including n03.
        cleanup = summary["steps"]["3_cleanup"]
        assert "per_host" in cleanup
        assert "n03.pro.pvt" in cleanup["per_host"]
        # n03 has rc != 0 (we returned rc=2).
        assert cleanup["per_host"]["n03.pro.pvt"]["rc"] == 2
        # And the cleanup step has ALL other corr + search hosts attempted.
        # Total corr=16 + search=4 = 20.
        assert len(cleanup["per_host"]) == 20
        # The other 19 hosts succeeded.
        ok_count = sum(
            1 for r in cleanup["per_host"].values() if r.get("ok")
        )
        assert ok_count == 19


class TestRestartAllSelfRestartDeferred:
    def test_self_restart_uses_popen_not_run(self):
        recorder = _CallRecorder(default_rc=0)
        popen_calls = []

        def fake_popen(argv, **kw):
            popen_calls.append((list(argv), dict(kw)))
            m = mock.Mock()
            m.pid = 99999
            return m

        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            summary = fleet_services.restart_all_services(
                self_restart_delay_s=2.0,
            )
        # Exactly one Popen call (the deferred dsa_monitor restart).
        assert len(popen_calls) == 1
        argv, kw = popen_calls[0]
        # argv[0]/argv[1] are "bash"/"-lc"; argv[2] is the deferred cmd.
        assert argv[0] == "bash"
        assert argv[1] == "-lc"
        assert "sleep 2" in argv[2]
        assert "systemctl --user restart dsa_monitor.service" in argv[2]
        # start_new_session=True is required so signals to the parent
        # don't propagate to the child.
        assert kw.get("start_new_session") is True
        # The schedule step's summary records the pid.
        assert summary["steps"]["7_self_restart_scheduled"]["ok"] is True
        assert summary["steps"]["7_self_restart_scheduled"]["pid"] == 99999

    def test_dry_run_skips_popen_and_subprocess(self):
        recorder = _CallRecorder(default_rc=0)
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen") as popen_m:
            summary = fleet_services.restart_all_services(dry_run=True)
        assert recorder.calls == []
        assert popen_m.call_count == 0
        # All steps marked skipped.
        for k, v in summary["steps"].items():
            assert v.get("skipped") is True


class TestRestartAllStopBroadcastBridge:
    def test_stop_broadcast_called_first(self):
        recorder = _CallRecorder(default_rc=0)
        order = []

        def stop_bcast():
            order.append("stop")
            return {"ok": True, "cmd": "stop"}

        with mock.patch.object(subprocess, "run", side_effect=lambda a, **k: order.append("subprocess") or _fake_completed(0)), \
             mock.patch.object(subprocess, "Popen", side_effect=lambda *a, **k: order.append("popen") or mock.Mock(pid=1)):
            fleet_services.restart_all_services(stop_broadcast=stop_bcast)
        assert order[0] == "stop"
        # popen comes LAST (step 7).
        assert order[-1] == "popen"


# ---------------------------------------------------------------------------
# control_store wrappers — audit on success + failure
# ---------------------------------------------------------------------------


class TestFleetServiceStatusWrapper:
    def test_writes_audit_row_on_success(self, fake_store_pair):
        cs, fake = fake_store_pair

        def fake_run(argv, **kw):
            if "is-active" in argv:
                return _fake_completed(0, "active\n")
            return _fake_completed(0, "")
        # All process-kind ssh calls also return rc=0 with output.
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            res = control_store.fleet_service_status(cs)
        assert "rows" in res
        assert len(res["rows"]) == len(inv.SERVICE_INVENTORY)
        # Exactly one audit row written.
        audit_puts = [
            p for p in fake.puts if p[0].startswith("/mon/audit/control/")
        ]
        assert len(audit_puts) == 1
        payload = audit_puts[0][1]
        assert payload["cmd"] == "services_status"
        assert payload["namespace"] == "services"

    def test_audit_ok_false_when_unreachable(self, fake_store_pair):
        cs, fake = fake_store_pair

        def fake_run(argv, **kw):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=5)
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            res = control_store.fleet_service_status(cs)
        audit_puts = [
            p for p in fake.puts if p[0].startswith("/mon/audit/control/")
        ]
        assert len(audit_puts) == 1
        # n_unreachable > 0 → ok=False on the audit row.
        assert audit_puts[0][1]["ok"] is False
        assert res["n_unreachable"] > 0


class TestFleetRestartAllWrapper:
    def test_dry_run_writes_audit_and_no_subprocess(self, fake_store_pair):
        cs, fake = fake_store_pair
        with mock.patch.object(subprocess, "run") as run_m, \
             mock.patch.object(subprocess, "Popen") as popen_m:
            res = control_store.fleet_restart_all(cs, dry_run=True)
        # No subprocess / popen in dry_run.
        assert run_m.call_count == 0
        assert popen_m.call_count == 0
        # One audit row written.
        audit_puts = [
            p for p in fake.puts if p[0].startswith("/mon/audit/control/")
        ]
        assert len(audit_puts) == 1
        payload = audit_puts[0][1]
        assert payload["cmd"] == "restart_all"
        assert payload["namespace"] == "services"
        # ok=True (dry_run is non-destructive).
        assert payload["ok"] is True
        # Result also records dry_run.
        assert res["dry_run"] is True

    def test_partial_failure_audit_ok_false(self, fake_store_pair):
        cs, fake = fake_store_pair
        recorder = _CallRecorder(default_rc=0, fail_hosts={"n03.pro.pvt"})
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=1)):
            res = control_store.fleet_restart_all(cs)
        # Overall flag: at least one step failed → ok=False.
        assert res["ok"] is False
        audit_puts = [
            p for p in fake.puts if p[0].startswith("/mon/audit/control/")
        ]
        # control_stop_fleet also writes one audit row; the restart_all
        # wrapper writes another. Grab the one whose cmd is restart_all.
        restart_audits = [
            p for p in audit_puts if p[1]["cmd"] == "restart_all"
        ]
        assert len(restart_audits) == 1
        assert restart_audits[0][1]["ok"] is False
        # The per-host val dict has n03 → False, others → True.
        val = restart_audits[0][1]["val"]
        assert val.get("n03.pro.pvt") is False
        # At least one host succeeded → others True.
        assert any(v is True for v in val.values())

    def test_audit_records_self_restart_attempt_in_summary(self, fake_store_pair):
        cs, fake = fake_store_pair
        recorder = _CallRecorder(default_rc=0)
        popen_calls = []

        def fake_popen(argv, **kw):
            popen_calls.append(list(argv))
            return mock.Mock(pid=42)
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            res = control_store.fleet_restart_all(cs)
        # Self-restart scheduled exactly once.
        assert len(popen_calls) == 1
        assert "systemctl --user restart dsa_monitor.service" in popen_calls[0][2]
        assert res["steps"]["7_self_restart_scheduled"]["ok"] is True


# ---------------------------------------------------------------------------
# ssh option pin — every ssh argv must use the agreed timeout +
# BatchMode + StrictHostKeyChecking flags.
# ---------------------------------------------------------------------------


class TestSshOpts:
    def test_ssh_argv_carries_required_options(self):
        recorder = _CallRecorder(default_rc=0)
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=1)):
            fleet_services.restart_all_services()
        ssh_calls = [c for c in recorder.calls if c and c[0] == "ssh"]
        assert ssh_calls, "expected at least one ssh call"
        for argv in ssh_calls:
            assert "-o" in argv
            assert "ConnectTimeout=5" in argv
            assert "StrictHostKeyChecking=no" in argv
            assert "BatchMode=yes" in argv
            assert "-n" in argv


# ---------------------------------------------------------------------------
# Inventory tier coverage by restart-all — every restartable tier
# must be touched.
# ---------------------------------------------------------------------------


class TestRestartCoverage:
    def test_all_corr_and_search_hosts_get_respawn(self):
        recorder = _CallRecorder(default_rc=0)
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=1)):
            summary = fleet_services.restart_all_services()
        respawn_hosts = set(summary["steps"]["6_orch_respawn"]["per_host"].keys())
        assert len(respawn_hosts) == 20
        # corr + search hosts only.
        for h in respawn_hosts:
            assert h.endswith(".pro.pvt")
            assert "lxd110h20" not in h

    def test_local_units_restart_excludes_dsa_monitor(self):
        recorder = _CallRecorder(default_rc=0)
        with mock.patch.object(subprocess, "run", side_effect=recorder), \
             mock.patch.object(subprocess, "Popen", return_value=mock.Mock(pid=1)):
            summary = fleet_services.restart_all_services()
        units = summary["steps"]["4_local_units_restart"]["per_unit"]
        # dsa_monitor.service must NOT be in the synchronous restart
        # list — it's deferred via Popen in step 7.
        assert "dsa_monitor.service" not in units
        assert "sefd_dashboard.service" in units
        assert "dsart_c2.service" in units


# ---------------------------------------------------------------------------
# Inventory key-builder pins (cn → host)
# ---------------------------------------------------------------------------


class TestCnToHost:
    def test_corr_cn_3_to_host(self):
        assert inv.CN_TO_HOST[3] == "n03.pro.pvt"

    def test_corr_cn_22_to_host(self):
        assert inv.CN_TO_HOST[22] == "n22.pro.pvt"

    def test_search_cns_present(self):
        for cn in (1, 2, 9, 13):
            assert inv.CN_TO_HOST[cn] == f"n{cn:02d}.pro.pvt"
