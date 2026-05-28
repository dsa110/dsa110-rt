"""Unit tests for ``tools/dashboard/dsa_monitor/fleet_update.py``.

The module under test fans out parallel ssh calls to the corr +
search nodes to bring each node's ``/home/ubuntu/proj/dsa110-rt``
checkout in sync with origin. These tests never invoke real ssh —
every test patches :func:`subprocess.run` so the call list is fully
inspectable, then asserts:

* the canonical 4-step happy path runs ``rev-parse + branch +
  porcelain``, ``git fetch``, ``git pull --ff-only`` / ``git reset
  --hard``, ``rev-parse`` (post-SHA);
* ``dry_run=True`` (default) NEVER issues a ``git pull`` or ``git
  reset`` command — only step 1 + ``git fetch``;
* dirty worktrees abort the host with ``error="dirty_worktree"``
  unless ``force=True`` is passed;
* ssh timeouts (``subprocess.TimeoutExpired``) surface as a
  per-host ``error="ssh_timeout"`` without breaking the rest of the
  fan-out;
* the audit row written via ``control_store.audit_log`` carries the
  correct cmd / val / ok shape; etcd put failures during audit are
  tolerated (the verb itself still returns the per-host report);
* parallelism — total wall time of the fan-out scales with
  ``max_workers``, not with ``n_hosts``.
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

import fleet_update                                                # noqa: E402
import control_store                                               # noqa: E402


# ---------------------------------------------------------------------------
# Fake DsaStore that records every put_dict (for audit-log assertions).
# Mirrors the pattern in tests/test_dsa_monitor_control_store.py.
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


# ---------------------------------------------------------------------------
# subprocess.run fake — drives _ssh_run by inspecting the remote command.
# ---------------------------------------------------------------------------


def _completed(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ssh"], returncode=rc, stdout=stdout, stderr=stderr,
    )


def _classify_remote(remote: str) -> str:
    """Bucket a remote-command string into one of the four step kinds
    so tests can target a precise response per step.
    """
    # Order matters: 'git fetch' must be classified before 'git reset'
    # because the reset command also contains 'origin/' but never
    # 'git fetch'.
    if "git status --porcelain" in remote:
        return "step1_prestate"
    if "git fetch" in remote:
        return "step3_fetch"
    if "git reset --hard" in remote:
        return "step5_reset"
    if "git pull --ff-only" in remote:
        return "step5_pull"
    if remote.endswith("git rev-parse HEAD"):
        return "step6_postsha"
    return "unknown"


class FakeSubprocessRunner:
    """Stateful ``subprocess.run`` replacement that returns a different
    canned response per (host, step) pair. Records every invocation
    so tests can inspect the full call list.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []           # (host, remote_cmd)
        self.responses: dict[tuple[str, str], Any] = {}
        self.default_responses: dict[str, Any] = {}
        self.host_default: dict[str, dict[str, Any]] = {}
        self.delay_s: float = 0.0
        self._lock = threading.Lock()

    def set_host_step(self, host: str, step: str, response: Any) -> None:
        self.responses[(host, step)] = response

    def set_host_defaults(self, host: str, defaults: dict[str, Any]) -> None:
        self.host_default[host] = dict(defaults)

    def set_default(self, step: str, response: Any) -> None:
        self.default_responses[step] = response

    def __call__(self, args, **kwargs):
        host = args[-2]
        remote = args[-1]
        with self._lock:
            self.calls.append((host, remote))
        step = _classify_remote(remote)
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        # Per (host, step) override first.
        if (host, step) in self.responses:
            r = self.responses[(host, step)]
        elif host in self.host_default and step in self.host_default[host]:
            r = self.host_default[host][step]
        elif step in self.default_responses:
            r = self.default_responses[step]
        else:
            r = _completed(0, "", "")
        if isinstance(r, BaseException):
            raise r
        if callable(r):
            return r(args, **kwargs)
        return r

    def commands_for(self, host: str) -> list[str]:
        return [c for h, c in self.calls if h == host]

    def all_commands(self) -> list[str]:
        return [c for _, c in self.calls]


# Common "clean checkout, pre==post" canned responses ------------------


CLEAN_PRESTATE = _completed(
    0,
    "abc123abc123\nm7/c1c2-coincidencer\n",
    "",
)
CLEAN_FETCH = _completed(0, "", "")
CLEAN_PULL = _completed(0, "Already up to date.\n", "")
CLEAN_POST_SAME = _completed(0, "abc123abc123\n", "")
CLEAN_POST_NEW = _completed(0, "def456def456\n", "")


def _wire_clean_host(fr: FakeSubprocessRunner, host: str, *,
                     post_sha: str = "abc123abc123") -> None:
    fr.set_host_defaults(host, {
        "step1_prestate": CLEAN_PRESTATE,
        "step3_fetch": CLEAN_FETCH,
        "step5_pull": CLEAN_PULL,
        "step5_reset": _completed(0, "HEAD is now at " + post_sha, ""),
        "step6_postsha": _completed(0, post_sha + "\n", ""),
    })


# ---------------------------------------------------------------------------
# _parse_dirty_files
# ---------------------------------------------------------------------------


class TestParseDirtyFiles:
    def test_modified_worktree(self):
        out = fleet_update._parse_dirty_files([" M src/somefile.py"])
        assert out == ["src/somefile.py"]

    def test_untracked_and_added_mixed(self):
        out = fleet_update._parse_dirty_files([
            "?? new_file.txt",
            "A  staged.py",
            " M edited.py",
            "",
        ])
        assert out == ["new_file.txt", "staged.py", "edited.py"]

    def test_empty_input(self):
        assert fleet_update._parse_dirty_files([]) == []


# ---------------------------------------------------------------------------
# _per_host_update — happy paths
# ---------------------------------------------------------------------------


class TestPerHostHappyPath:
    def test_apply_clean_worktree_produces_new_sha(self):
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "n03.pro.pvt", post_sha="def456def456")
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=False, branch=None,
            )
        assert res["ok"] is True
        assert res["host"] == "n03.pro.pvt"
        assert res["pre_sha"] == "abc123abc123"
        assert res["post_sha"] == "def456def456"
        assert res["changed"] is True
        assert res["branch"] == "m7/c1c2-coincidencer"
        assert res["dirty_files"] == []
        assert res["error"] is None
        # 4 ssh calls in the happy apply path.
        assert len(fr.calls) == 4
        cmds = fr.all_commands()
        assert any("git pull --ff-only origin m7/c1c2-coincidencer" in c for c in cmds)
        # No reset on the apply-clean path.
        assert not any("git reset --hard" in c for c in cmds)

    def test_dry_run_skips_pull_and_keeps_pre_sha(self):
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "n03.pro.pvt")
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=True, force=False, branch=None,
            )
        assert res["ok"] is True
        assert res["dry_run"] is True
        assert res["pre_sha"] == "abc123abc123"
        assert res["post_sha"] == "abc123abc123"
        assert res["changed"] is False
        # Only 2 ssh calls on dry_run: pre-state + fetch.
        cmds = fr.all_commands()
        assert len(cmds) == 2
        assert any("git status --porcelain" in c for c in cmds)
        assert any("git fetch origin" in c for c in cmds)
        # Pin: dry_run NEVER calls git pull or git reset.
        assert not any("git pull" in c for c in cmds)
        assert not any("git reset" in c for c in cmds)

    def test_branch_detection_from_host(self):
        """branch=None → uses the branch git reports per-host."""
        fr = FakeSubprocessRunner()
        fr.set_host_defaults("n03.pro.pvt", {
            "step1_prestate": _completed(
                0, "abc123abc123\nfeature/foo\n", "",
            ),
            "step3_fetch": _completed(0, "", ""),
            "step5_pull": _completed(0, "", ""),
            "step6_postsha": _completed(0, "abc123abc123\n", ""),
        })
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=False, branch=None,
            )
        assert res["ok"] is True
        assert res["branch"] == "feature/foo"
        cmds = fr.all_commands()
        assert any("git fetch origin feature/foo" in c for c in cmds)
        assert any("git pull --ff-only origin feature/foo" in c for c in cmds)

    def test_branch_override(self):
        """branch='other' wins over the per-host current branch."""
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "n03.pro.pvt")
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=False,
                branch="release/v1",
            )
        assert res["branch"] == "release/v1"
        cmds = fr.all_commands()
        assert any("git fetch origin release/v1" in c for c in cmds)
        assert any("git pull --ff-only origin release/v1" in c for c in cmds)


# ---------------------------------------------------------------------------
# _per_host_update — dirty-worktree gate
# ---------------------------------------------------------------------------


class TestDirtyWorktreeGate:
    def test_dirty_worktree_aborts_without_force(self):
        fr = FakeSubprocessRunner()
        fr.set_host_defaults("n03.pro.pvt", {
            "step1_prestate": _completed(
                0,
                "abc123abc123\nm7/c1c2-coincidencer\n M somefile.py\n",
                "",
            ),
        })
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=False, branch=None,
            )
        assert res["ok"] is False
        assert res["error"] == "dirty_worktree"
        assert res["dirty_files"] == ["somefile.py"]
        assert res["pre_sha"] == "abc123abc123"
        # Critically: NO fetch, NO pull, NO reset.
        cmds = fr.all_commands()
        assert len(cmds) == 1
        assert not any("git fetch" in c for c in cmds)
        assert not any("git pull" in c for c in cmds)
        assert not any("git reset" in c for c in cmds)

    def test_dirty_worktree_with_force_uses_reset(self):
        fr = FakeSubprocessRunner()
        fr.set_host_defaults("n03.pro.pvt", {
            "step1_prestate": _completed(
                0,
                "abc123abc123\nm7/c1c2-coincidencer\n M somefile.py\n",
                "",
            ),
            "step3_fetch": _completed(0, "", ""),
            "step5_reset": _completed(0, "HEAD is now at def456", ""),
            "step6_postsha": _completed(0, "def456def456\n", ""),
        })
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=True, branch=None,
            )
        assert res["ok"] is True
        assert res["dirty_files"] == ["somefile.py"]
        assert res["pre_sha"] == "abc123abc123"
        assert res["post_sha"] == "def456def456"
        assert res["changed"] is True
        cmds = fr.all_commands()
        # Reset, not pull.
        assert any(
            "git reset --hard origin/m7/c1c2-coincidencer" in c for c in cmds
        )
        assert not any("git pull" in c for c in cmds)

    def test_dirty_worktree_dry_run_force_no_pull(self):
        """force=True + dry_run=True still skips reset/pull."""
        fr = FakeSubprocessRunner()
        fr.set_host_defaults("n03.pro.pvt", {
            "step1_prestate": _completed(
                0,
                "abc123abc123\nm7/c1c2-coincidencer\n M somefile.py\n",
                "",
            ),
            "step3_fetch": _completed(0, "", ""),
        })
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=True, force=True, branch=None,
            )
        assert res["ok"] is True
        assert res["dirty_files"] == ["somefile.py"]
        assert res["post_sha"] == res["pre_sha"]
        cmds = fr.all_commands()
        assert not any("git reset" in c for c in cmds)
        assert not any("git pull" in c for c in cmds)


# ---------------------------------------------------------------------------
# _per_host_update — error surfaces
# ---------------------------------------------------------------------------


class TestPerHostErrorSurfaces:
    def test_ssh_timeout_surfaces_as_ssh_timeout(self):
        fr = FakeSubprocessRunner()
        fr.set_host_step(
            "n03.pro.pvt", "step1_prestate",
            subprocess.TimeoutExpired(cmd="ssh", timeout=5.0),
        )
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=True, force=False, branch=None,
            )
        assert res["ok"] is False
        assert res["error"] == "ssh_timeout"
        # No follow-up steps should have been attempted.
        assert len(fr.calls) == 1

    def test_ssh_timeout_at_fetch_step(self):
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "n03.pro.pvt")
        fr.set_host_step(
            "n03.pro.pvt", "step3_fetch",
            subprocess.TimeoutExpired(cmd="ssh", timeout=5.0),
        )
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=False, branch=None,
            )
        assert res["ok"] is False
        assert res["error"] == "ssh_timeout"
        assert res["pre_sha"] == "abc123abc123"   # populated before fetch

    def test_ssh_nonzero_rc_surfaces_ssh_failed(self):
        fr = FakeSubprocessRunner()
        fr.set_host_step(
            "n03.pro.pvt", "step1_prestate",
            _completed(255, "", "Permission denied (publickey)."),
        )
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=True, force=False, branch=None,
            )
        assert res["ok"] is False
        assert "ssh_failed" in res["error"]

    def test_git_fetch_failure_surfaces_git_fetch_failed(self):
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "n03.pro.pvt")
        fr.set_host_step(
            "n03.pro.pvt", "step3_fetch",
            _completed(1, "", "fatal: could not read from remote"),
        )
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=True, force=False, branch=None,
            )
        assert res["ok"] is False
        assert "git_fetch_failed" in res["error"]

    def test_git_pull_failure_surfaces_git_update_failed(self):
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "n03.pro.pvt")
        fr.set_host_step(
            "n03.pro.pvt", "step5_pull",
            _completed(1, "", "fatal: not possible to fast-forward"),
        )
        with mock.patch.object(subprocess, "run", side_effect=fr):
            res = fleet_update._per_host_update(
                "n03.pro.pvt", dry_run=False, force=False, branch=None,
            )
        assert res["ok"] is False
        assert "git_update_failed" in res["error"]


# ---------------------------------------------------------------------------
# update_fleet — fan-out + summary + audit
# ---------------------------------------------------------------------------


class TestUpdateFleetFanout:
    def test_summary_counts_correct(self, fake_store_pair):
        cs, fake = fake_store_pair
        hosts = ["nA", "nB", "nC", "nD"]
        fr = FakeSubprocessRunner()
        # nA: clean apply, new sha (ok + changed)
        _wire_clean_host(fr, "nA", post_sha="def456def456")
        # nB: clean apply, same sha (ok + unchanged)
        _wire_clean_host(fr, "nB", post_sha="abc123abc123")
        # nC: dirty without force → fails as dirty_worktree (not ok + dirty)
        fr.set_host_defaults("nC", {
            "step1_prestate": _completed(
                0, "abc123abc123\nmain\n M file.py\n", "",
            ),
        })
        # nD: ssh timeout (not ok)
        fr.set_host_step(
            "nD", "step1_prestate",
            subprocess.TimeoutExpired(cmd="ssh", timeout=5.0),
        )

        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_update.update_fleet(
                cs, dry_run=False, force=False, hosts=hosts,
                branch=None, max_workers=2, user="ops",
            )
        assert out["ok"] is False                  # nC + nD failed
        s = out["summary"]
        assert s["n_hosts"] == 4
        assert s["n_ok"] == 2
        assert s["n_failed"] == 2
        assert s["n_dirty"] == 1
        assert s["n_changed"] == 1
        # Per-host list sorted by host name.
        assert [r["host"] for r in out["hosts"]] == hosts

    def test_dry_run_default_pins_no_writes_to_any_host(self, fake_store_pair):
        cs, fake = fake_store_pair
        hosts = ["nA", "nB"]
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "nA")
        _wire_clean_host(fr, "nB")
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_update.update_fleet(
                cs, hosts=hosts, max_workers=2, user="ops",
            )
        assert out["ok"] is True
        # The dashboard default is dry_run=True. No host should have
        # seen a pull or a reset.
        assert out["summary"]["dry_run"] is True
        all_cmds = fr.all_commands()
        assert not any("git pull" in c for c in all_cmds)
        assert not any("git reset" in c for c in all_cmds)

    def test_audit_row_carries_summary_and_cmd_update_dsart(self, fake_store_pair):
        cs, fake = fake_store_pair
        hosts = ["nA", "nB"]
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "nA", post_sha="def456def456")
        _wire_clean_host(fr, "nB", post_sha="def456def456")
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_update.update_fleet(
                cs, dry_run=False, force=False, hosts=hosts,
                max_workers=2, user="alice",
            )
        # Exactly one audit row.
        audit_rows = [p for p in fake.puts
                      if p[0].startswith("/mon/audit/control/")]
        assert len(audit_rows) == 1
        key, payload = audit_rows[0]
        assert key.endswith("Z")
        assert payload["cmd"] == "update_dsart"
        assert payload["namespace"] == "dsa_monitor.fleet_update"
        assert payload["user"] == "alice"
        assert payload["ok"] is True
        assert payload["val"]["n_ok"] == 2
        assert payload["val"]["n_changed"] == 2
        assert payload["val"]["dry_run"] is False
        assert payload["val"]["force"] is False
        assert "nA" in payload["cn_target"]
        assert "nB" in payload["cn_target"]
        # The function-level return matches.
        assert out["ok"] is True
        assert out["summary"]["n_ok"] == 2

    def test_audit_etcd_put_failure_tolerated(self, fake_store_pair):
        cs, fake = fake_store_pair
        hosts = ["nA"]
        # Make audit put_dict explode.
        def _explode(key, payload):
            raise RuntimeError("etcd down")
        fake.put_dict = _explode                    # type: ignore[assignment]
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "nA")
        with mock.patch.object(subprocess, "run", side_effect=fr):
            # Should NOT raise.
            out = fleet_update.update_fleet(
                cs, dry_run=True, hosts=hosts, max_workers=1, user="ops",
            )
        # The verb still returns the per-host report.
        assert out["ok"] is True
        assert out["summary"]["n_ok"] == 1

    def test_empty_hosts_returns_empty_result(self, fake_store_pair):
        cs, _ = fake_store_pair
        out = fleet_update.update_fleet(cs, hosts=[], user="ops")
        assert out["ok"] is True
        assert out["hosts"] == []
        assert out["summary"]["n_hosts"] == 0

    def test_ssh_timeout_does_not_break_fanout(self, fake_store_pair):
        cs, _ = fake_store_pair
        hosts = ["nGood", "nBad", "nGood2"]
        fr = FakeSubprocessRunner()
        _wire_clean_host(fr, "nGood")
        _wire_clean_host(fr, "nGood2")
        fr.set_host_step(
            "nBad", "step1_prestate",
            subprocess.TimeoutExpired(cmd="ssh", timeout=5.0),
        )
        with mock.patch.object(subprocess, "run", side_effect=fr):
            out = fleet_update.update_fleet(
                cs, dry_run=True, hosts=hosts, max_workers=3, user="ops",
            )
        assert out["ok"] is False
        # The two good hosts must still have ok=True; only the bad
        # one is failed.
        by_host = {r["host"]: r for r in out["hosts"]}
        assert by_host["nGood"]["ok"] is True
        assert by_host["nGood2"]["ok"] is True
        assert by_host["nBad"]["ok"] is False
        assert by_host["nBad"]["error"] == "ssh_timeout"

    def test_default_fleet_hosts_topology_is_20(self):
        # 16 corr + 4 search.
        assert len(fleet_update.DEFAULT_CORR_HOSTS) == 16
        assert len(fleet_update.DEFAULT_SEARCH_HOSTS) == 4
        assert len(fleet_update.DEFAULT_FLEET_HOSTS) == 20
        # All canonical n??.pro.pvt names.
        for h in fleet_update.DEFAULT_FLEET_HOSTS:
            assert h.startswith("n") and h.endswith(".pro.pvt")


# ---------------------------------------------------------------------------
# Parallelism — total wall time scales with max_workers
# ---------------------------------------------------------------------------


class TestParallelism:
    def test_parallel_fanout_faster_than_serial(self, fake_store_pair):
        cs, _ = fake_store_pair
        hosts = ["nA", "nB", "nC", "nD"]
        per_call_delay_s = 0.10                    # dry_run is 2 calls/host
        # Per-host time ≈ 0.20s. Serial = 0.80s, parallel(4) ≈ 0.20s.
        fr = FakeSubprocessRunner()
        for h in hosts:
            _wire_clean_host(fr, h)
        fr.delay_s = per_call_delay_s

        with mock.patch.object(subprocess, "run", side_effect=fr):
            t0 = time.monotonic()
            out_parallel = fleet_update.update_fleet(
                cs, dry_run=True, hosts=hosts, max_workers=4, user="ops",
            )
            dt_parallel = time.monotonic() - t0

        assert out_parallel["ok"] is True
        # Parallel-of-4 should finish much closer to per-host time
        # (~0.2s) than to serial (~0.8s). We assert it's <50% of the
        # serial bound, which is a robust check that still tolerates
        # the GIL + ThreadPoolExecutor scheduling jitter.
        serial_bound = len(hosts) * 2 * per_call_delay_s   # 0.80s
        assert dt_parallel < (serial_bound * 0.5), (
            f"parallel fan-out took {dt_parallel:.3f}s; expected "
            f"< {serial_bound * 0.5:.3f}s (serial would be ~{serial_bound:.3f}s)"
        )

    def test_two_slow_hosts_close_to_one_host_duration(self, fake_store_pair):
        cs, _ = fake_store_pair
        # Two hosts each blocking for ~0.30s per ssh call; with
        # max_workers=2 and dry_run (2 ssh per host), total wall
        # time should be closer to 0.60s (one host's per-host time)
        # than to 1.20s (sequential).
        hosts = ["nA", "nB"]
        per_call_delay_s = 0.30
        fr = FakeSubprocessRunner()
        for h in hosts:
            _wire_clean_host(fr, h)
        fr.delay_s = per_call_delay_s

        with mock.patch.object(subprocess, "run", side_effect=fr):
            t0 = time.monotonic()
            fleet_update.update_fleet(
                cs, dry_run=True, hosts=hosts, max_workers=2, user="ops",
            )
            dt = time.monotonic() - t0

        per_host_time = 2 * per_call_delay_s        # 0.60s
        sequential_time = 2 * per_host_time         # 1.20s
        midpoint = (per_host_time + sequential_time) / 2
        # Should be on the "closer to one host's time" side of the
        # midpoint — i.e., parallelism is actually happening.
        assert dt < midpoint, (
            f"two-host parallel took {dt:.3f}s; expected "
            f"< midpoint {midpoint:.3f}s (one host = {per_host_time:.3f}s, "
            f"sequential = {sequential_time:.3f}s)"
        )
