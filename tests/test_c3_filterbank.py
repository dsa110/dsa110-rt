"""Tests for dsart/coinc/filterbank.py (C3 → bbproc integration)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from dsart.coinc.filterbank import (
    FilterbankConfig,
    newest_complete_cal_set,
    run_for_event,
    snapshot_cal,
)


def _mk_cal_set(d: Path, isot: str, n: int = 16) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for sb in range(n):
        (d / f"beamformer_weights_sb{sb:02d}_{isot}.dat").write_bytes(
            b"x" * 16)


def test_newest_complete_cal_set_skips_incomplete(tmp_path: Path) -> None:
    _mk_cal_set(tmp_path, "2026-07-10T00:00:00")
    _mk_cal_set(tmp_path, "2026-07-14T00:00:00")
    _mk_cal_set(tmp_path, "2026-07-15T00:00:00", n=7)   # incomplete, newest
    got = newest_complete_cal_set(str(tmp_path))
    assert got is not None and len(got) == 16
    assert "2026-07-14T00:00:00" in got["00"]


def test_snapshot_cal_copies_and_is_idempotent(tmp_path: Path) -> None:
    applied = tmp_path / "applied"
    _mk_cal_set(applied, "2026-07-14T00:00:00")
    dest = tmp_path / "cand" / "filterbank" / "cal"
    sb00 = snapshot_cal(str(applied), dest)
    assert sb00 is not None and sb00.is_file()
    assert len(list(dest.glob("*.dat"))) == 16
    # newer set appears — existing snapshot must be REUSED (provenance)
    _mk_cal_set(applied, "2026-07-15T12:00:00")
    sb00b = snapshot_cal(str(applied), dest)
    assert sb00b == sb00
    assert "2026-07-14" in sb00b.name


def _stub_scripts(tmp_path: Path) -> FilterbankConfig:
    """toolkit/plot stand-ins: write their output arg + record the call."""
    toolkit = tmp_path / "toolkit_stub.sh"
    toolkit.write_text(
        "#!/bin/bash\n"
        "out=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = '-P' ]; then out=$2; fi; shift\n"
        "done\n"
        "echo \"$@\" > /dev/null\n"
        "dd if=/dev/zero of=$out bs=1k count=1 2>/dev/null\n")
    toolkit.chmod(toolkit.stat().st_mode | stat.S_IEXEC)
    plot = tmp_path / "plot_stub.py"
    plot.write_text(
        "import sys\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "open(out, 'wb').write(b'PNG')\n")
    applied = tmp_path / "applied"
    _mk_cal_set(applied, "2026-07-14T00:00:00")
    return FilterbankConfig(
        enabled=True, toolkit_bin=str(toolkit), plot_script=str(plot),
        core_antennas="all", cal_applied_dir=str(applied), rfi_mode="both",
        timeout_s=30.0)


def _mk_event(tmp_path: Path, name: str, n_frags: int = 2) -> Path:
    ev = tmp_path / name
    (ev / "Level2" / "voltages").mkdir(parents=True)
    (ev / "Level3").mkdir(parents=True)
    for sb in range(n_frags):
        (ev / "Level2" / "voltages" / f"{name}_sb{sb:02d}_data.out"
         ).write_bytes(b"v")
    return ev


def test_run_for_event_produces_both_variants(tmp_path: Path) -> None:
    cfg = _stub_scripts(tmp_path)
    ev = _mk_event(tmp_path, "260715aaaa")
    rep = run_for_event(cfg, ev, "260715aaaa",
                        {"l_median": 0.001, "m_median": -0.002,
                         "dm_median": 168.8, "width_median": 4.0,
                         "t_peak_mjd": 61236.5},
                        dec_deg=16.2734)
    assert rep["ok"] is True, rep
    assert rep["dec_deg"] == 16.2734
    # the toolkit invocation must carry the F21 dec + tstart mjd
    tk = rep["runs"][0]["cmd"]
    assert "--dec-deg" in tk and "16.2734" in tk[tk.index("--dec-deg") + 1]
    assert "--mjd" in tk
    fb = ev / "filterbank"
    assert (fb / "260715aaaa.fil").is_file()
    assert (fb / "260715aaaa_rfi.fil").is_file()
    assert (fb / "260715aaaa.png").is_file()
    assert (fb / "260715aaaa_rfi.png").is_file()
    assert len(list((fb / "cal").glob("*.dat"))) == 16
    meta = json.loads((fb / "filterbank.json").read_text())
    assert meta["l"] == 0.001 and meta["dm"] == 168.8
    # width: 4 search samples x 1048.576us / (32.768us x 8) = 16
    assert meta["width_fil_samples"] == 16
    assert sorted(meta["outputs"]) == sorted([
        "260715aaaa.fil", "260715aaaa.png",
        "260715aaaa_rfi.fil", "260715aaaa_rfi.png"])


def test_run_for_event_skips_without_fragments(tmp_path: Path) -> None:
    cfg = _stub_scripts(tmp_path)
    ev = _mk_event(tmp_path, "260715bbbb", n_frags=0)
    rep = run_for_event(cfg, ev, "260715bbbb", {})
    assert rep["ok"] is True
    assert rep["skipped"] == "no voltage fragments"
    assert not (ev / "filterbank").exists()


def test_run_for_event_failure_is_contained(tmp_path: Path) -> None:
    cfg = _stub_scripts(tmp_path)
    bad = FilterbankConfig(
        enabled=True, toolkit_bin="/nonexistent/toolkit",
        plot_script=cfg.plot_script, core_antennas="all",
        cal_applied_dir=cfg.cal_applied_dir, rfi_mode="off", timeout_s=5.0)
    ev = _mk_event(tmp_path, "260715cccc")
    rep = run_for_event(bad, ev, "260715cccc", {"dm_median": 100.0})
    assert rep["ok"] is False          # reported, not raised


# ---- cal_hdf5 C3 integration helpers (2026-07-15) --------------------------


def test_cal_hdf5_config_from_dict_defaults() -> None:
    from dsart.coinc.cal_hdf5_archive import CalHdf5Config
    c = CalHdf5Config.from_dict(None)
    assert c.enabled and c.hours_each_side == 2.0
    c2 = CalHdf5Config.from_dict({"enabled": False, "hours_each_side": 1.0})
    assert not c2.enabled and c2.hours_each_side == 1.0


def test_archive_event_idempotent_and_partial(tmp_path: Path) -> None:
    """archive_event links what exists, reports incompleteness, and a
    re-run picks up newly-arrived files without touching prior links."""
    from dsart.coinc.cal_hdf5_archive import archive_event
    corr = tmp_path / "correlator"
    corr.mkdir()
    cands = tmp_path / "cands"
    ev = cands / "260715aaaa"
    (ev / "Level3").mkdir(parents=True)
    # burst at 2026-07-15T03:00:00 -> mjd
    import datetime as _dt
    t = _dt.datetime(2026, 7, 15, 3, 0, 0, tzinfo=_dt.timezone.utc)
    mjd = t.timestamp() / 86400.0 + 40587.0
    (ev / "Level3" / "260715aaaa.json").write_text(
        json.dumps({"c2": {"t_peak_mjd": mjd}}))

    def _mk(hh, mm, n_sb=16):
        for sb in range(n_sb):
            (corr / f"2026-07-15T{hh:02d}:{mm:02d}:00_sb{sb:02d}.hdf5"
             ).write_bytes(b"h5")

    # only the PRE half exists initially (01:00-03:00, 5-min cadence)
    for k in range(24):
        _mk(1 + (k * 5) // 60, (k * 5) % 60)
    rep1 = archive_event("260715aaaa", candidates_root=cands,
                         correlator_dir=corr, hours_each_side=2.0)
    assert rep1["complete"] is False
    assert rep1["n_linked"] == 24 * 16
    # POST half arrives; re-run links only the new ones
    for k in range(24):
        _mk(3 + (k * 5) // 60, (k * 5) % 60)
    rep2 = archive_event("260715aaaa", candidates_root=cands,
                         correlator_dir=corr, hours_each_side=2.0)
    assert rep2["complete"] is True
    assert rep2["n_already_present"] == 24 * 16
    assert rep2["n_linked"] == 24 * 16
    # hard links, not copies
    one = ev / "calibration" / "2026-07-15T01:00:00_sb00.hdf5"
    assert one.stat().st_nlink == 2
