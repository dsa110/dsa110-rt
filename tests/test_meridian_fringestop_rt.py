"""Unit tests for the casa38 meridian_fringestop production wrapper's pure
helpers. These run in the dsa110-rt env (numpy + stdlib); the wrapper defers
all casa38-only imports (dsamfs/astropy/dsautils) into functions, so importing
the module here does not require casa38.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

_WRAPPER = (
    Path(__file__).resolve().parents[1] / "tools" / "ops" / "meridian_fringestop_rt.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("meridian_fringestop_rt", _WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


mfs = _load_module()


# ---- snap_dec_to_grid -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,step,expected",
    [
        (25.13, 0.25, 25.25),   # closer to 25.25 than 25.0
        (25.12, 0.25, 25.0),    # closer to 25.0
        (25.0, 0.25, 25.0),     # exact grid point
        (-30.0, 0.25, -30.0),
        (71.61, 0.25, 71.5),
        (71.62, 0.25, 71.5),    # 0.12 below 71.62, 0.13 above -> nearer 71.5
        (71.63, 0.25, 71.75),   # now nearer 71.75
    ],
)
def test_snap_dec_to_grid(raw, step, expected):
    assert mfs.snap_dec_to_grid(raw, step) == pytest.approx(expected)


def test_snap_grid_values_are_exact_multiples():
    # 0.25 is exact in binary; snapped values round-trip cleanly through
    # deg2rad so they match the cache's stored dec_rad to < 1e-6.
    for raw in (12.37, -7.81, 84.99):
        g = mfs.snap_dec_to_grid(raw, 0.25)
        assert abs((g / 0.25) - round(g / 0.25)) < 1e-9


def test_snap_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        mfs.snap_dec_to_grid(25.0, 0.0)


# ---- filename helpers -----------------------------------------------------


def test_cache_table_filename_matches_build_scheme():
    # Must match tools/build_fstable_cache.py::_fstable_filename exactly.
    assert (
        mfs.cache_table_filename(25.25, 96, 58849.0)
        == "fringestopping_table_dec_+25.2500deg_96ant_refmjd58849.000000.npz"
    )
    assert (
        mfs.cache_table_filename(-30.0, 96, 58849.0)
        == "fringestopping_table_dec_-30.0000deg_96ant_refmjd58849.000000.npz"
    )


def test_legacy_table_name_matches_dsamfs():
    # dsamfs builds f"fringestopping_table_dec{(pt_dec*u.rad).to_value(u.deg):.1f}deg_{nant}ant.npz"
    assert mfs.legacy_table_name(25.25, 96) == "fringestopping_table_dec25.2deg_96ant.npz"
    assert mfs.legacy_table_name(25.0, 96) == "fringestopping_table_dec25.0deg_96ant.npz"
    assert mfs.legacy_table_name(-30.0, 96) == "fringestopping_table_dec-30.0deg_96ant.npz"


# ---- resolve_subband ------------------------------------------------------


def _ch0():
    # Order matters: subband = index in key order (parse_params).
    return {
        "lxd110h03": 1024, "lxd110h04": 1408, "lxd110h05": 1792, "lxd110h06": 2176,
        "lxd110h07": 2560, "lxd110h08": 2944, "lxd110h10": 3328, "lxd110h11": 3712,
        "lxd110h12": 4096, "lxd110h14": 4480, "lxd110h15": 4864, "lxd110h16": 5248,
        "lxd110h18": 5632, "lxd110h19": 6016, "lxd110h21": 6400, "lxd110h22": 6784,
    }


def test_resolve_subband_index_by_key_order():
    assert mfs.resolve_subband(_ch0(), "lxd110h03") == 0
    assert mfs.resolve_subband(_ch0(), "lxd110h10") == 6
    assert mfs.resolve_subband(_ch0(), "lxd110h22") == 15


def test_resolve_subband_unknown_host_raises():
    # Must fail loud, not silently fall back to subband 0 like dsamfs.
    with pytest.raises(KeyError):
        mfs.resolve_subband(_ch0(), "lxd110h99")


# ---- validate_cache_table -------------------------------------------------


def _write_fake_table(path: Path, *, nint=96, nbls=4656, dec_deg=25.25,
                      tsamp=0.134217728, refmjd=58849.0):
    np.savez(
        path,
        dec_rad=math.radians(dec_deg),
        tsamp_s=tsamp,
        bw=np.zeros((nint, nbls), dtype=np.float64),
        bwref=np.zeros((nbls, nint), dtype=np.float64),
        refmjd=refmjd,
    )


def test_validate_cache_table_ok(tmp_path):
    f = tmp_path / mfs.cache_table_filename(25.25, 96, 58849.0)
    _write_fake_table(f)
    ok, reason = mfs.validate_cache_table(
        f, nbls=4656, nint=96, dec_deg_grid=25.25, tsamp=0.134217728, refmjd=58849.0,
    )
    assert ok, reason


def test_validate_cache_table_missing(tmp_path):
    ok, reason = mfs.validate_cache_table(
        tmp_path / "nope.npz", nbls=4656, nint=96, dec_deg_grid=25.25,
        tsamp=0.134217728, refmjd=58849.0,
    )
    assert not ok and "does not exist" in reason


def test_validate_cache_table_wrong_shape(tmp_path):
    f = tmp_path / "t.npz"
    _write_fake_table(f, nint=48)
    ok, reason = mfs.validate_cache_table(
        f, nbls=4656, nint=96, dec_deg_grid=25.25, tsamp=0.134217728, refmjd=58849.0,
    )
    assert not ok and "shape" in reason


def test_validate_cache_table_dec_mismatch(tmp_path):
    f = tmp_path / "t.npz"
    _write_fake_table(f, dec_deg=25.5)
    ok, reason = mfs.validate_cache_table(
        f, nbls=4656, nint=96, dec_deg_grid=25.25, tsamp=0.134217728, refmjd=58849.0,
    )
    assert not ok and "dec_rad" in reason


def test_validate_cache_table_refmjd_mismatch(tmp_path):
    f = tmp_path / "t.npz"
    _write_fake_table(f, refmjd=59000.0)
    ok, reason = mfs.validate_cache_table(
        f, nbls=4656, nint=96, dec_deg_grid=25.25, tsamp=0.134217728, refmjd=58849.0,
    )
    assert not ok and "refmjd" in reason


# ---- stage_legacy_symlink -------------------------------------------------


def test_stage_legacy_symlink_creates_and_replaces(tmp_path):
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    cache.mkdir()
    cache_file = cache / mfs.cache_table_filename(25.25, 96, 58849.0)
    cache_file.write_bytes(b"x")

    dest = mfs.stage_legacy_symlink(cache_file, work, 25.25, 96)
    assert dest.is_symlink()
    assert dest.name == "fringestopping_table_dec25.2deg_96ant.npz"
    assert Path(os.readlink(dest)) == cache_file

    # Re-staging replaces a stale symlink without error.
    dest2 = mfs.stage_legacy_symlink(cache_file, work, 25.25, 96)
    assert dest2 == dest and dest2.is_symlink()


# ---- SPL heartbeat (distinct key + file family) ---------------------------


def test_heartbeat_key_differs_for_spl():
    hb = mfs.Heartbeat(cn_id=6, working_dir=Path("/tmp"), subband=3, dec_deg=25.25)
    hb_spl = mfs.Heartbeat(
        cn_id=6, working_dir=Path("/tmp"), subband=3, dec_deg=25.25, spl=True,
    )
    assert hb._key == "/mon/corr_rt/6/meridian_ready"
    assert hb_spl._key == "/mon/corr_rt/6/meridian_spl_ready"


def test_heartbeat_newest_hdf5_separates_spl_and_prod(tmp_path):
    # Production hdf5 + SPL hdf5 share a working dir here (in production
    # they don't, but the glob must still pick the right family).
    (tmp_path / "2026-06-16T00:00:00_sb03.hdf5").write_bytes(b"x")
    (tmp_path / "2026-06-16T00:00:00_sb03_spl.hdf5").write_bytes(b"x")

    prod = mfs.Heartbeat(cn_id=6, working_dir=tmp_path, subband=3, dec_deg=25.25)
    spl = mfs.Heartbeat(
        cn_id=6, working_dir=tmp_path, subband=3, dec_deg=25.25, spl=True,
    )
    prod_last, prod_n = prod._newest_hdf5()
    spl_last, spl_n = spl._newest_hdf5()
    assert prod_last == "2026-06-16T00:00:00_sb03.hdf5" and prod_n == 1
    assert spl_last == "2026-06-16T00:00:00_sb03_spl.hdf5" and spl_n == 1


# ---- SPL CLI override args (exercise the REAL main() parser) ---------------


def test_main_parser_accepts_spl_override_args(monkeypatch):
    """main()'s real argparse must accept the SPL override flags. We stub
    _prepare (the only etcd-touching call before --prepare-only returns)
    so the test stays offline and asserts on the parsed Namespace."""
    import shlex

    captured = {}

    def _fake_prepare(args):
        captured["args"] = args
        return {"subband": 3, "dec_grid": 25.25, "working_dir": args.working_dir,
                "cache_ok": True, "eff_nint": 96, "override_nfreq_int": 4}

    monkeypatch.setattr(mfs, "_prepare", _fake_prepare)
    argv = shlex.split(
        "--cn-id 6 --pt-dec-deg 25.25 --spl --integration-s 12.884901888 "
        "--nfreq-int-spl 4 --working-dir /home/ubuntu/data/spl --prepare-only"
    )
    rc = mfs.main(argv)
    assert rc == 0
    args = captured["args"]
    assert args.spl is True
    assert args.integration_s == pytest.approx(12.884901888)
    assert args.nfreq_int_spl == 4
    assert str(args.working_dir) == "/home/ubuntu/data/spl"
