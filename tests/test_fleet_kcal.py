"""Unit tests for ``tools/dashboard/dsa_monitor/fleet_kcal.py``.

The module under test fans out parallel ssh calls to the corr nodes to
delete each node's K-cal (beamformer-weights) blob. These tests never
invoke real ssh — every test patches :func:`subprocess.run` so the
call list is fully inspectable, then asserts:

* the per-node path resolves to ``beamformer_weights_sb<NN>.dat`` for
  the node's chgroup (the ``CALSB`` substitution dsart_rt does);
* ``dry_run=True`` (default) only probes (``[ -e ... ]``) and NEVER
  issues an ``rm``;
* an apply deletes the blob (``rm -rf``) and reports
  ``status="deleted"``;
* a missing blob reports ``status="not_found"`` with ``ok=True`` (not
  an error);
* ssh timeouts (``subprocess.TimeoutExpired``) surface as a per-host
  ``status="error"`` / ``error="ssh_timeout"`` without breaking the
  rest of the fan-out;
* the audit row written via ``control_store.audit_log`` carries the
  ``cmd=delete_kcal`` shape; etcd put failures during audit are
  tolerated.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Any
from unittest import mock

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
DSA_MONITOR_DIR = os.path.normpath(os.path.join(
    HERE, "..", "tools", "dashboard", "dsa_monitor",
))
if DSA_MONITOR_DIR not in sys.path:
    sys.path.insert(0, DSA_MONITOR_DIR)

import fleet_kcal                                                  # noqa: E402
import control_store                                              # noqa: E402
from corr_topology import CORR_NODES, CORR_NODES_BY_CHGROUP        # noqa: E402


# ---------------------------------------------------------------------------
# Fake DsaStore that records every put_dict (for audit-log assertions).
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
    cs._store = fake                                               # bypass DsaStore import
    return cs, fake


def _completed(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ssh"], returncode=rc, stdout=stdout, stderr=stderr,
    )


class FakeSubprocessRunner:
    """``subprocess.run`` replacement returning a canned response keyed
    by the ssh host. Records every (host, remote_cmd) call.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.host_response: dict[str, Any] = {}
        self.default: Any = _completed(0, "NOT_FOUND\n", "")
        self._lock = threading.Lock()

    def set_host(self, host: str, response: Any) -> None:
        self.host_response[host] = response

    def __call__(self, args, **kwargs):
        host = args[-2]
        remote = args[-1]
        with self._lock:
            self.calls.append((host, remote))
        r = self.host_response.get(host, self.default)
        if isinstance(r, BaseException):
            raise r
        return r

    def all_commands(self) -> list[str]:
        return [c for _, c in self.calls]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestKcalPaths:
    def test_filename_for_chgroup_zero_padded(self):
        assert fleet_kcal.kcal_filename_for_chgroup(3) == "beamformer_weights_sb03.dat"
        assert fleet_kcal.kcal_filename_for_chgroup(15) == "beamformer_weights_sb15.dat"

    def test_path_for_chgroup_uses_run_dir(self):
        p = fleet_kcal.kcal_path_for_chgroup(0)
        assert p == (
            "/home/ubuntu/data/voltages/250924mptq/cals/"
            "beamformer_weights_sb00.dat"
        )
        assert fleet_kcal.KCAL_RUN in p

    def test_chgroup_out_of_range_raises(self):
        with pytest.raises(ValueError):
            fleet_kcal.kcal_filename_for_chgroup(16)
        with pytest.raises(ValueError):
            fleet_kcal.kcal_filename_for_chgroup(-1)


# ---------------------------------------------------------------------------
# _per_host_delete
# ---------------------------------------------------------------------------


class TestPerHostDelete:
    def test_dry_run_only_probes_never_removes(self):
        node = CORR_NODES_BY_CHGROUP[0]
        fr = FakeSubprocessRunner()
        fr.set_host(node.fqdn, _completed(0, "EXISTS\n", ""))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_kcal._per_host_delete(node, dry_run=True)
        assert res["ok"] is True
        assert res["status"] == "exists"
        assert res["deleted"] is False
        cmds = fr.all_commands()
        assert len(cmds) == 1
        assert "[ -e " in cmds[0]
        assert "rm -rf" not in cmds[0]

    def test_apply_deletes_blob(self):
        node = CORR_NODES_BY_CHGROUP[5]
        fr = FakeSubprocessRunner()
        fr.set_host(node.fqdn, _completed(0, "DELETED\n", ""))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_kcal._per_host_delete(node, dry_run=False)
        assert res["ok"] is True
        assert res["status"] == "deleted"
        assert res["deleted"] is True
        assert res["path"].endswith("beamformer_weights_sb05.dat")
        cmds = fr.all_commands()
        assert any("rm -rf" in c for c in cmds)

    def test_missing_blob_is_not_found_not_error(self):
        node = CORR_NODES_BY_CHGROUP[1]
        fr = FakeSubprocessRunner()
        fr.set_host(node.fqdn, _completed(0, "NOT_FOUND\n", ""))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_kcal._per_host_delete(node, dry_run=False)
        assert res["ok"] is True
        assert res["status"] == "not_found"
        assert res["deleted"] is False
        assert res["error"] is None

    def test_ssh_timeout_surfaces_as_error(self):
        node = CORR_NODES_BY_CHGROUP[2]
        fr = FakeSubprocessRunner()
        fr.set_host(node.fqdn, subprocess.TimeoutExpired(cmd="ssh", timeout=30))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_kcal._per_host_delete(node, dry_run=False)
        assert res["ok"] is False
        assert res["status"] == "error"
        assert res["error"] == "ssh_timeout"

    def test_ssh_nonzero_rc_surfaces_as_error(self):
        node = CORR_NODES_BY_CHGROUP[4]
        fr = FakeSubprocessRunner()
        fr.set_host(node.fqdn, _completed(255, "", "ssh: connect failed"))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_kcal._per_host_delete(node, dry_run=False)
        assert res["ok"] is False
        assert res["status"] == "error"
        assert "ssh_failed" in res["error"]


# ---------------------------------------------------------------------------
# delete_kcal_fleet
# ---------------------------------------------------------------------------


class TestDeleteKcalFleet:
    def test_apply_deletes_all_and_writes_audit(self, fake_store_pair):
        cs, fake = fake_store_pair
        nodes = list(CORR_NODES[:3])
        fr = FakeSubprocessRunner()
        for n in nodes:
            fr.set_host(n.fqdn, _completed(0, "DELETED\n", ""))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_kcal.delete_kcal_fleet(
                cs, dry_run=False, nodes=nodes, user="tester",
            )
        assert out["ok"] is True
        assert out["summary"]["n_hosts"] == 3
        assert out["summary"]["n_deleted"] == 3
        assert out["summary"]["n_failed"] == 0
        assert out["summary"]["dry_run"] is False
        assert out["summary"]["run"] == fleet_kcal.KCAL_RUN
        # hosts sorted by host name.
        hosts = [h["host"] for h in out["hosts"]]
        assert hosts == sorted(hosts)
        # one audit row written.
        assert len(fake.puts) == 1
        key, payload = fake.puts[0]
        assert key.startswith("/mon/audit/control/")
        assert payload["cmd"] == "delete_kcal"
        assert payload["ok"] is True

    def test_dry_run_never_removes(self, fake_store_pair):
        cs, _fake = fake_store_pair
        nodes = list(CORR_NODES[:2])
        fr = FakeSubprocessRunner()
        for n in nodes:
            fr.set_host(n.fqdn, _completed(0, "EXISTS\n", ""))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_kcal.delete_kcal_fleet(
                cs, dry_run=True, nodes=nodes,
            )
        assert out["ok"] is True
        assert out["summary"]["n_present"] == 2
        assert out["summary"]["n_deleted"] == 0
        assert not any("rm -rf" in c for c in fr.all_commands())

    def test_mixed_results_overall_not_ok_on_failure(self, fake_store_pair):
        cs, _fake = fake_store_pair
        nodes = list(CORR_NODES[:3])
        fr = FakeSubprocessRunner()
        fr.set_host(nodes[0].fqdn, _completed(0, "DELETED\n", ""))
        fr.set_host(nodes[1].fqdn, _completed(0, "NOT_FOUND\n", ""))
        fr.set_host(nodes[2].fqdn, _completed(255, "", "boom"))
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_kcal.delete_kcal_fleet(
                cs, dry_run=False, nodes=nodes,
            )
        assert out["ok"] is False
        assert out["summary"]["n_deleted"] == 1
        assert out["summary"]["n_not_found"] == 1
        assert out["summary"]["n_failed"] == 1

    def test_audit_failure_is_tolerated(self):
        nodes = list(CORR_NODES[:1])
        fr = FakeSubprocessRunner()
        fr.set_host(nodes[0].fqdn, _completed(0, "DELETED\n", ""))

        class BadStore:
            def put_dict(self, *a, **k):
                raise RuntimeError("etcd down")

            def get_dict(self, *a, **k):
                return None

        cs = control_store.ControlStore()
        cs._store = BadStore()
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_kcal.delete_kcal_fleet(
                cs, dry_run=False, nodes=nodes,
            )
        # The verb still returns its per-host report despite audit failure.
        assert out["ok"] is True
        assert out["summary"]["n_deleted"] == 1
