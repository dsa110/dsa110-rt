"""Unit tests for the dashboard fringe-stopping-table panel.

Exercises the pure helpers (snap / expected_filename / parse_filename),
the local h23 inventory, the ssh-and-rsync helpers via subprocess
monkey-patching (so the tests don't touch the real fleet), and the
current-DEC traffic-light logic across the green / amber / red /
unknown branches.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Dashboard package isn't on sys.path by default in tests; mirror the
# layout the systemd service uses (tools/dashboard/dsa_monitor/ added
# to sys.path so its sibling modules import).
_DASH_DIR = Path(__file__).resolve().parents[1] / "tools" / "dashboard" / "dsa_monitor"
sys.path.insert(0, str(_DASH_DIR))

import fstable_panel as fp                                  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_snap_dec_to_grid_default_step():
    assert fp.snap_dec_to_grid(54.5734) == pytest.approx(54.50, abs=1e-9)
    assert fp.snap_dec_to_grid(54.62) == pytest.approx(54.50, abs=1e-9)
    assert fp.snap_dec_to_grid(54.63) == pytest.approx(54.75, abs=1e-9)
    assert fp.snap_dec_to_grid(-0.13) == pytest.approx(-0.25, abs=1e-9)
    assert fp.snap_dec_to_grid(0.0) == pytest.approx(0.0, abs=1e-9)


def test_snap_dec_to_grid_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        fp.snap_dec_to_grid(0.0, step_deg=0.0)
    with pytest.raises(ValueError):
        fp.snap_dec_to_grid(0.0, step_deg=-0.25)


def test_expected_filename_matches_corr_side_layout():
    # Must match meridian_fringestop_rt.cache_table_filename
    # byte-for-byte; this is the contract that lets us detect cache
    # hits by filename alone.
    name = fp.expected_filename(54.5734, 96, 58849.0)
    assert name == "fringestopping_table_dec_+54.5000deg_96ant_refmjd58849.000000.npz"

    # Negative DEC keeps the explicit sign.
    name_neg = fp.expected_filename(-12.3, 96, 58849.0)
    assert name_neg.startswith("fringestopping_table_dec_-12.2500deg_")


def test_parse_filename_round_trip_with_expected():
    nant = 96
    refmjd = 58849.0
    for dec in (-30.0, -0.25, 0.0, 0.25, 54.5, 85.0):
        name = fp.expected_filename(dec, nant, refmjd)
        parsed = fp.parse_filename(name)
        assert parsed is not None, f"could not parse {name!r}"
        assert parsed["dec_deg"] == pytest.approx(dec, abs=1e-9)
        assert parsed["nant"] == nant
        assert parsed["refmjd"] == pytest.approx(refmjd, abs=1e-9)


def test_parse_filename_rejects_garbage():
    assert fp.parse_filename("garbage.npz") is None
    assert fp.parse_filename("") is None
    # Almost-right but wrong precision.
    assert fp.parse_filename(
        "fringestopping_table_dec_+54.500deg_96ant_refmjd58849.000000.npz"
    ) is None


# ---------------------------------------------------------------------------
# h23 master-dir inventory
# ---------------------------------------------------------------------------


def test_list_h23_tables_reads_valid_files(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)
    name1 = fp.expected_filename(54.5, 96, 58849.0)
    name2 = fp.expected_filename(0.0, 96, 58849.0)
    (tmp_path / name1).write_bytes(b"x" * 1234)
    (tmp_path / name2).write_bytes(b"y" * 4567)
    (tmp_path / "ignore_me.txt").write_text("not a cache file")
    (tmp_path / "bogus.npz").write_text("not a cache filename")

    rows = fp.list_h23_tables()
    names = {e.filename for e in rows}
    assert name1 in names and name2 in names
    assert "ignore_me.txt" not in names
    assert "bogus.npz" not in names
    for e in rows:
        assert e.host == "h23"
        assert e.size_bytes > 0
        assert e.mtime_unix > 0


def test_list_h23_tables_handles_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path / "does-not-exist")
    assert fp.list_h23_tables() == []


def test_ensure_master_dir_creates_path(tmp_path, monkeypatch):
    target = tmp_path / "a" / "b" / "fstables"
    monkeypatch.setattr(fp, "H23_MASTER_DIR", target)
    assert not target.exists()
    fp.ensure_master_dir()
    assert target.is_dir()


# ---------------------------------------------------------------------------
# ssh ls fan-out (subprocess.run monkey-patched)
# ---------------------------------------------------------------------------


class _StubCp:
    """Minimal subprocess.CompletedProcess-shaped stub."""
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_list_corr_tables_ok(monkeypatch):
    name = fp.expected_filename(54.5, 96, 58849.0)
    stdout = f"{name}\t3617827\t1717100000.0\n"

    def fake_run(args, **kwargs):
        assert args[0] == "ssh"
        assert "n03.pro.pvt" in args
        return _StubCp(0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, entries, err = fp.list_corr_tables("n03.pro.pvt", timeout_s=2.0)
    assert ok and err is None
    assert len(entries) == 1
    e = entries[0]
    assert e.host == "n03.pro.pvt"
    assert e.filename == name
    assert e.dec_deg == pytest.approx(54.5)
    assert e.nant == 96
    assert e.refmjd == pytest.approx(58849.0)


def test_list_corr_tables_dir_missing(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _StubCp(2, "", "no such dir"),
    )
    ok, entries, err = fp.list_corr_tables("n99.pro.pvt", timeout_s=2.0)
    assert ok and entries == [] and err is None


def test_list_corr_tables_ssh_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _StubCp(255, "", "ssh: connection refused"),
    )
    ok, entries, err = fp.list_corr_tables("n99.pro.pvt", timeout_s=2.0)
    assert not ok and entries == [] and err is not None
    assert "ssh rc=255" in err


def test_list_corr_tables_timeout(monkeypatch):
    def raises_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=2.0)
    monkeypatch.setattr(subprocess, "run", raises_timeout)
    ok, entries, err = fp.list_corr_tables("nXX.pro.pvt", timeout_s=2.0)
    assert not ok and entries == [] and err is not None
    assert "timeout" in err.lower()


def test_list_fleet_tables_fan_out(monkeypatch):
    name1 = fp.expected_filename(54.5, 96, 58849.0)

    def fake_run(args, **kwargs):
        host = args[args.index("ssh") + 1 + len([a for a in args[:5] if a == "ssh"])]
        # The host arg is the 1st argument after the -n flag's set of ssh opts.
        host = args[args.index("-n") + 1]
        if host == "n03.pro.pvt":
            return _StubCp(0, stdout=f"{name1}\t100\t1717.0\n", stderr="")
        if host == "n04.pro.pvt":
            return _StubCp(0, stdout="", stderr="")          # empty dir
        return _StubCp(255, stdout="", stderr="down")

    monkeypatch.setattr(subprocess, "run", fake_run)
    fleet = fp.list_fleet_tables(
        ["n03.pro.pvt", "n04.pro.pvt", "n05.pro.pvt"],
        timeout_s=1.0, max_workers=4,
    )
    assert set(fleet.keys()) == {"n03.pro.pvt", "n04.pro.pvt", "n05.pro.pvt"}
    assert fleet["n03.pro.pvt"]["ok"] and len(fleet["n03.pro.pvt"]["entries"]) == 1
    assert fleet["n04.pro.pvt"]["ok"] and fleet["n04.pro.pvt"]["entries"] == []
    assert not fleet["n05.pro.pvt"]["ok"]
    assert "ssh rc=255" in fleet["n05.pro.pvt"]["error"]


# ---------------------------------------------------------------------------
# Traffic-light composition
# ---------------------------------------------------------------------------


def _seed_h23(monkeypatch, tmp_path, filenames):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for n in filenames:
        (tmp_path / n).write_bytes(b"x")


def test_current_dec_status_unknown_when_no_dec():
    status = fp.current_dec_status(
        dec_deg=None, nant=96, refmjd=58849.0,
        corr_hosts=["n03.pro.pvt", "n04.pro.pvt"],
    )
    assert status["light"] == "unknown"
    assert status["all_ready"] is False
    assert status["expected_filename"] is None


def test_current_dec_status_red_when_no_corr_has_it(monkeypatch, tmp_path):
    """No reachable corr node has the table → RED (pipeline will crash)."""
    _seed_h23(monkeypatch, tmp_path, [])
    fleet_inv = {
        "n03.pro.pvt": {"ok": True, "entries": [], "error": None},
        "n04.pro.pvt": {"ok": True, "entries": [], "error": None},
    }
    status = fp.current_dec_status(
        dec_deg=54.5734, nant=96, refmjd=58849.0,
        corr_hosts=list(fleet_inv.keys()),
        fleet_inventory=fleet_inv,
    )
    assert status["light"] == "red"
    assert status["all_ready"] is False
    assert status["dec_deg_grid"] == pytest.approx(54.5)
    assert status["expected_filename"].startswith(
        "fringestopping_table_dec_+54.5000deg_96ant_"
    )
    # Red message should hint at the appropriate next action.
    assert "Build" in status["message"]


def test_current_dec_status_amber_when_partial_fleet(monkeypatch, tmp_path):
    name = fp.expected_filename(54.5, 96, 58849.0)
    _seed_h23(monkeypatch, tmp_path, [name])
    fleet_inv = {
        "n03.pro.pvt": {
            "ok": True,
            "entries": [{"filename": name}],
            "error": None,
        },
        "n04.pro.pvt": {
            "ok": True,
            "entries": [],
            "error": None,
        },
        "n05.pro.pvt": {
            "ok": False,
            "entries": [],
            "error": "ssh_timeout",
        },
    }
    status = fp.current_dec_status(
        dec_deg=54.5, nant=96, refmjd=58849.0,
        corr_hosts=list(fleet_inv.keys()),
        fleet_inventory=fleet_inv,
    )
    assert status["light"] == "amber"
    assert status["corr_hosts_with_table"] == ["n03.pro.pvt"]
    assert status["corr_hosts_missing_table"] == ["n04.pro.pvt"]
    assert status["corr_hosts_unreachable"] == ["n05.pro.pvt"]
    assert status["all_ready"] is False


def test_current_dec_status_green_when_all_corr_present_h23_missing(monkeypatch, tmp_path):
    """All corr nodes have it but h23 master is missing → still GREEN
    (pipeline will start). h23 master is only a redeploy convenience."""
    name = fp.expected_filename(54.5, 96, 58849.0)
    _seed_h23(monkeypatch, tmp_path, [])             # h23 master absent
    fleet_inv = {
        h: {"ok": True, "entries": [{"filename": name}], "error": None}
        for h in ("n03.pro.pvt", "n04.pro.pvt")
    }
    status = fp.current_dec_status(
        dec_deg=54.5, nant=96, refmjd=58849.0,
        corr_hosts=list(fleet_inv.keys()),
        fleet_inventory=fleet_inv,
    )
    assert status["light"] == "green"
    assert status["all_ready"] is True
    assert status["h23_has_table"] is False
    assert set(status["corr_hosts_with_table"]) == {"n03.pro.pvt", "n04.pro.pvt"}
    # Message should still mention h23 absence so operator can fix it.
    assert "h23" in status["message"]


def test_current_dec_status_green_when_all_present(monkeypatch, tmp_path):
    name = fp.expected_filename(54.5, 96, 58849.0)
    _seed_h23(monkeypatch, tmp_path, [name])
    fleet_inv = {
        h: {"ok": True, "entries": [{"filename": name}], "error": None}
        for h in ("n03.pro.pvt", "n04.pro.pvt")
    }
    status = fp.current_dec_status(
        dec_deg=54.5, nant=96, refmjd=58849.0,
        corr_hosts=list(fleet_inv.keys()),
        fleet_inventory=fleet_inv,
    )
    assert status["light"] == "green"
    assert status["all_ready"] is True
    assert set(status["corr_hosts_with_table"]) == {"n03.pro.pvt", "n04.pro.pvt"}


def test_current_dec_status_unknown_when_all_corr_unreachable(monkeypatch, tmp_path):
    name = fp.expected_filename(54.5, 96, 58849.0)
    _seed_h23(monkeypatch, tmp_path, [name])
    fleet_inv = {
        h: {"ok": False, "entries": [], "error": "ssh_timeout"}
        for h in ("n03.pro.pvt", "n04.pro.pvt")
    }
    status = fp.current_dec_status(
        dec_deg=54.5, nant=96, refmjd=58849.0,
        corr_hosts=list(fleet_inv.keys()),
        fleet_inventory=fleet_inv,
    )
    # Can't tell either way → unknown, NOT green.
    assert status["light"] == "unknown"
    assert status["all_ready"] is False
    assert set(status["corr_hosts_unreachable"]) == {"n03.pro.pvt", "n04.pro.pvt"}


# ---------------------------------------------------------------------------
# Build path: subprocess monkey-patched
# ---------------------------------------------------------------------------


def test_build_table_for_dec_subprocess_args(monkeypatch, tmp_path):
    out_dir = tmp_path
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = dict(kwargs)
        # Pretend the build script wrote the expected file.
        dec_grid = 54.5
        name = fp.expected_filename(dec_grid, 96, 58849.0)
        (out_dir / name).write_bytes(b"\x00" * 64)
        return _StubCp(0, stdout="built dec=+54.5000", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(fp, "H23_MASTER_DIR", out_dir)

    res = fp.build_table_for_dec(
        dec_deg=54.5734, nant=96, refmjd=58849.0,
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=out_dir,
        timeout_s=10.0,
    )

    assert res.ok is True
    assert res.dec_deg_grid == pytest.approx(54.5)
    assert res.expected_filename.startswith("fringestopping_table_dec_+54.5000deg_96ant_")
    assert res.error is None
    # Verify args passed to the subprocess.
    a = captured["args"]
    assert a[0].endswith("/python")
    assert "--from-etcd" in a
    assert "--dec-min" in a and "--dec-max" in a
    # the dec-min / dec-max should be the *grid-snapped* value.
    i = a.index("--dec-min")
    assert float(a[i + 1]) == pytest.approx(54.5)


def test_build_table_for_dec_marks_failure_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _StubCp(0, stdout="ok", stderr=""),
    )
    res = fp.build_table_for_dec(
        dec_deg=54.5, nant=96, refmjd=58849.0,
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=tmp_path,
        timeout_s=1.0,
    )
    assert res.ok is False
    assert "missing" in (res.error or "")


def test_build_table_for_dec_handles_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)

    def raises_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python build", timeout=1.0)
    monkeypatch.setattr(subprocess, "run", raises_timeout)

    res = fp.build_table_for_dec(
        dec_deg=54.5, nant=96, refmjd=58849.0,
        repo_root=Path(__file__).resolve().parents[1],
        output_dir=tmp_path,
        timeout_s=1.0,
    )
    assert res.ok is False
    assert "timed out" in res.error.lower()


# ---------------------------------------------------------------------------
# Deploy path
# ---------------------------------------------------------------------------


def test_deploy_table_to_fleet_rejects_missing_master(monkeypatch, tmp_path):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)
    name = fp.expected_filename(54.5, 96, 58849.0)
    # Master dir is empty.
    out = fp.deploy_table_to_fleet(name, ["n03.pro.pvt"], timeout_s=1.0)
    assert out["ok"] is False
    assert out["all_ok"] is False
    assert "missing" in out["error"]


def test_deploy_table_to_fleet_rejects_bad_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)
    bad = "../etc/passwd"
    out = fp.deploy_table_to_fleet(bad, ["n03.pro.pvt"], timeout_s=1.0)
    # parse fails first, but if the path-validation happens earlier
    # (in the route) we'd never get here. Backend defense in depth:
    # if it reaches the backend, it still won't slip out because the
    # h23 source check uses H23_MASTER_DIR / filename and rejects when
    # not a file. Either branch yields ok=False.
    assert out["ok"] is False


def test_deploy_table_to_fleet_fan_out(monkeypatch, tmp_path):
    name = fp.expected_filename(54.5, 96, 58849.0)
    monkeypatch.setattr(fp, "H23_MASTER_DIR", tmp_path)
    (tmp_path / name).write_bytes(b"x")

    calls: list = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        # Treat n05 as the failing host.
        host_arg = args[-1]
        if "n05" in host_arg:
            return _StubCp(23, stdout="", stderr="rsync transfer error")
        return _StubCp(0, stdout="sent 100 bytes", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = fp.deploy_table_to_fleet(
        name,
        ["n03.pro.pvt", "n04.pro.pvt", "n05.pro.pvt"],
        timeout_s=1.0, max_workers=4,
    )
    assert out["ok"] is True
    assert out["all_ok"] is False
    hosts = {h["host"]: h for h in out["hosts"]}
    assert hosts["n03.pro.pvt"]["ok"] and hosts["n03.pro.pvt"]["rc"] == 0
    assert hosts["n04.pro.pvt"]["ok"]
    assert not hosts["n05.pro.pvt"]["ok"]
    assert hosts["n05.pro.pvt"]["rc"] == 23
    # rsync was called once per host.
    rsync_calls = [c for c in calls if c[0] == "rsync"]
    assert len(rsync_calls) == 3
