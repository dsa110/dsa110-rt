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


# ---------------------------------------------------------------------------
# Timing: .fil tstart from the manifest, and the sweep-corrected --t0
# (2026-08-26 fixes; see filterbank.py N_PRE_BLOCKS / _sweep_s)
# ---------------------------------------------------------------------------


def _mk_manifests(ev: Path, name: str, block_mjd_first: float,
                  n_sb: int = 2) -> None:
    """Per-subband voltage manifests, as C3 collection writes them."""
    vd = ev / "Level2" / "voltages"
    vd.mkdir(parents=True, exist_ok=True)
    for sb in range(n_sb):
        (vd / f"{name}_sb{sb:02d}.json").write_text(json.dumps({
            "event_name": name, "subband": f"sb{sb:02d}",
            "block_mjd_first": block_mjd_first,
            "n_pre": 14, "n_post": 8, "n_blocks_written": 23,
        }))


def test_n_pre_blocks_pinned_to_fourteen() -> None:
    # Regression pin: the constant was 8 while every manifest records
    # n_pre=14, which put the .fil tstart 6 blocks (0.805 s) late.
    from dsart.coinc.filterbank import BLOCK_S, N_PRE_BLOCKS
    assert N_PRE_BLOCKS == 14
    assert abs((14 - 8) * BLOCK_S - 0.80531) < 1e-4


def test_sweep_s_matches_measured_events() -> None:
    # Values validated against the two events whose bursts are actually
    # present in the voltages: the recovered peak sits one sweep before
    # the C2 t_peak (260824utbu -0.204 s, 260822iyhi -0.356 s).
    from dsart.coinc.filterbank import _sweep_s
    assert abs(_sweep_s(367.2) - 0.2078) < 5e-4
    assert abs(_sweep_s(631.1) - 0.3571) < 5e-4
    assert _sweep_s(0.0) == 0.0


def test_tstart_and_t0_come_from_manifest(tmp_path: Path) -> None:
    cfg = _stub_scripts(tmp_path)
    name = "260826aaaa"
    ev = _mk_event(tmp_path, name)
    blk = 61277.01057247913
    t_peak = 61277.01059456718           # 1.9084 s after blk
    _mk_manifests(ev, name, blk)
    rep = run_for_event(cfg, ev, name,
                        {"l_median": 0.0, "m_median": 0.0,
                         "dm_median": 395.799866, "width_median": 1.0,
                         "t_peak_mjd": t_peak}, dec_deg=71.63)
    assert rep["ok"] is True, rep
    assert rep["tstart_source"] == "manifest"
    assert abs(rep["tstart_mjd"] - blk) < 1e-12
    # toolkit gets the manifest tstart, not a computed guess
    tk = rep["runs"][0]["cmd"]
    assert abs(float(tk[tk.index("--mjd") + 1]) - blk) < 1e-12
    # plot gets a sweep-corrected --t0
    from dsart.coinc.filterbank import _sweep_s
    want = (t_peak - blk) * 86400.0 - _sweep_s(395.799866)
    plot = rep["runs"][1]["cmd"]
    assert "--t0" in plot
    assert abs(float(plot[plot.index("--t0") + 1]) - want) < 1e-6
    # ... and that lands where the burst actually is (1.6844 s), not at
    # the header-implied 1.0737 s the old code produced.
    assert abs(want - 1.6844) < 2e-3


def test_tstart_falls_back_when_no_manifest(tmp_path: Path) -> None:
    cfg = _stub_scripts(tmp_path)
    name = "260826bbbb"
    ev = _mk_event(tmp_path, name)      # data.out only, no *_sbNN.json
    t_peak = 61277.5
    rep = run_for_event(cfg, ev, name,
                        {"l_median": 0.0, "m_median": 0.0,
                         "dm_median": 100.0, "width_median": 1.0,
                         "t_peak_mjd": t_peak}, dec_deg=71.63)
    assert rep["ok"] is True, rep
    assert rep["tstart_source"] == "computed_fallback"
    from dsart.coinc.filterbank import BLOCK_S, N_PRE_BLOCKS
    want = t_peak - N_PRE_BLOCKS * BLOCK_S / 86400.0
    assert abs(rep["tstart_mjd"] - want) < 1e-12
