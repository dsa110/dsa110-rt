"""x2 D4: raw-window archive, uploader subdirs, sweeper, offline re-imager."""

from __future__ import annotations

import importlib.util
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from dsart.coinc import cube_uploader as U
from dsart.dump.cube_retention import CubeRetentionConfig, sweep_once
from dsart.x2.cutout import raw_archive as R
from dsart.x2.cutout import schema as S
from dsart.x2.opoint import opoint_2x
from dsart.x2.plan import build_tiered_plan_for_opoint

REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def _plan():
    return build_tiered_plan_for_opoint(opoint_2x("gridded", 4, 8))


def test_stream_window_matches_producer_rule() -> None:
    # tier A (1 T_ext/sample) and tier B (2), with stage-2 extents
    assert R.stream_window(1000, 954, 954, ext_factor=1, max_pos=40, max_neg=3) == \
        (1000 - 954 - 40, 954 * 2 + 40 + 3)
    s0, n = R.stream_window(1001, 954, 954, ext_factor=2, max_pos=10, max_neg=0)
    assert s0 == (1001 - 954) // 2 - 10 and s0 + n == -(-(1001 + 954) // 2)


def test_synthetic_ring_and_slab_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 255, size=(32, 400, 11), dtype=np.uint8)
    ring = R.SyntheticRing({5: codes}, base=1000, oldest=1100, frontier=1350)
    with pytest.raises(TimeoutError):
        ring.wait_window(5, 1050, 10)
    c, v = ring.wait_window(5, 1300, 100)
    assert v[:, :50].all() and not v[:, 50:].any()
    assert np.array_equal(c[:, :50], codes[:, 300:350])
    shifts = rng.integers(0, 30, size=(17, 32)).astype(np.int32)
    out = tmp_path / "raw_s9_g1"
    R.write_raw_slab(out, codes=c, valid=v, row0=1300, t_native=16, d=5,
                     shifts=shifts, fine_global=np.arange(100, 117),
                     bits_main=8, pattern_id=0xABCDEF0123456789)
    assert not R.is_raw_complete(out)
    with pytest.raises(R.RawArchiveError):
        R.read_raw_dir(out)
    req = type("Q", (), dict(event_name="261022rawx", request_id=4, t_peak_ext=1350,
                             t0_ext=1094, n_t=512, raw_pre_ext=954, raw_post_ext=954,
                             flags=3, plan_id=7))()
    R.write_raw_meta(out, req, identity={"search_node_id": 9, "gpu_half": 1})
    R.mark_raw_ok(out)
    slabs = R.read_raw_dir(out)
    s = slabs[5]
    assert s.row0 == 1300 and s.t_native == 16 and s.n_rows == 100
    assert s.extra["pattern_id"] == 0xABCDEF0123456789
    rows, valid = s.rows(3, 1290, 20)
    assert not valid[:10].any() and valid[10:].all()
    assert np.array_equal(rows[10:], c[3, :10])
    with pytest.raises(R.RawArchiveError):
        R.write_raw_slab(out, codes=c.astype(np.int16), valid=v, row0=0,
                         t_native=16, d=6, shifts=shifts, fine_global=np.arange(17))


def test_uploader_dest_subdir(tmp_path: Path) -> None:
    a1 = U.build_rsync_argv("ev1", tmp_path, "u@h", "/root")
    assert a1[-1] == "u@h:/root/ev1/cubes/"                     # today's
    a2 = U.build_rsync_argv("ev1", tmp_path, "u@h", "/root",
                            dest_subdir="raw/raw_s9_g1")
    assert a2[-1] == "u@h:/root/ev1/raw/raw_s9_g1/"
    with pytest.raises(ValueError):
        U.build_rsync_argv("ev1", tmp_path, "u@h", "/root", dest_subdir="../x")
    seen = []

    def fake_upload(**kw):
        seen.append(kw)

        class _P:
            def wait(self):
                return 0
        return _P()

    for sub, expect in (("cubes", None), ("cutouts", "cutouts")):
        up = U.BoundedCubeUploader(dest_host="u@h", dest_root="/root",
                                   upload_fn=fake_upload, dest_subdir=sub)
        up.start()
        up.submit("ev1", tmp_path)
        up.stop()
        assert seen[-1].get("dest_subdir") == expect


def _touch(p: Path, n: int, age_h: float = 0.0) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)
    t = time.time() - age_h * 3600.0
    os.utime(p, (t, t))


def test_sweeper_product_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "dump"
    # 1x event dir with cubes: one entry, as today
    _touch(root / "ev1x" / "cube_s1_g0_5.npz", 500)
    # 2x event: per-half raw dirs (one uploaded, one not) + cutouts
    _touch(root / "ev2x" / "raw" / "raw_s1_g0" / "raw_d000.npz", 4000)
    (root / "ev2x" / "raw" / "raw_s1_g0" / "upload.log").write_text("x UPLOAD_OK y\n")
    _touch(root / "ev2x" / "raw" / "raw_s1_g1" / "raw_d001.npz", 4000)
    _touch(root / "ev2x" / "cutouts" / "cutout_s1_g0.npz", 100)
    cfg = CubeRetentionConfig(dump_root=root, max_age_h=96.0,
                              max_total_gb=6e-6, low_water_gb=5e-6)
    st = sweep_once(cfg)
    # nested raw bytes are counted; the UPLOADED half's raw goes first
    assert not (root / "ev2x" / "raw" / "raw_s1_g0").exists()
    assert (root / "ev2x" / "raw" / "raw_s1_g1").exists()
    assert st["n_unuploaded_sacrificed"] == 0
    assert (root / "ev1x" / "cube_s1_g0_5.npz").exists()


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "reimage_raw_window", REPO / "tools" / "x2" / "reimage_raw_window.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offline_reimager_plumbing_to_a_valid_cutout(tmp_path: Path) -> None:
    from dsart.coinc.wire import CUTOUT_FLAG_CUTOUT, CutoutRequest
    tool = _load_tool()
    plan = _plan()
    h = 2
    raw = tmp_path / "raw_s2_g0"
    rng = np.random.default_rng(3)
    owned = list(plan.dms_of_half(h))
    for d in owned[:-1]:                                     # last DM too_late
        sl = plan.fines_of(d)
        R.write_raw_slab(raw, codes=rng.integers(0, 255, (16 * 4, 50, 4), np.uint8),
                         valid=np.ones((64, 50), np.uint8), row0=900,
                         t_native=int(plan.coarse_t_native[d]), d=int(d),
                         shifts=np.zeros((sl.stop - sl.start, 64), np.int32),
                         fine_global=np.arange(sl.start, sl.stop))
    R.write_raw_meta(raw, None, identity={"search_node_id": 2, "gpu_half": 0,
                                          "half_index": h}, too_late=[owned[-1]])
    R.mark_raw_ok(raw)
    n_t, N = 16, 8

    def imager(slab, f_local, t0, n, *, ext_factor):
        img = np.zeros((n, N, N), np.float32)
        img[:, 1, 2] = slab.d + 0.01 * f_local
        return img

    r = tool.reimage(raw, plan, half=h, t0_ext=1000, n_t=n_t, imager=imager, n_grid=N)
    assert r["missing"] == [int(owned[-1])]
    ax = r["axis"]
    n_f = len(ax["fine_dm_pc_cc"])
    req = CutoutRequest(event_name="offline", request_id=0, t_peak_ext=1008,
                        t0_ext=1000, n_t=n_t, flags=CUTOUT_FLAG_CUTOUT,
                        stamp_half_px=0, n_stamp_t=0, l_pix=1, m_pix=2)
    fields = S.assemble_cutout(
        axis=ax, images=r["images"], n_t=n_t,
        sigma_applied=np.ones((n_t, n_f), np.float32),
        row_valid=np.ones((n_t, len(ax["coarse_owned"])), bool),
        input_valid_frac=np.ones(n_f, np.float32), request=req,
        identity=dict(search_node_id=2, gpu_half=0, half_index=h),
        clock=dict(t_ext_us=524.288, native_per_ext=16), geometry=dict(n_grid=N))
    npz, _ = S.write_cutout(tmp_path / "out", fields)
    c = S.read_cutout(npz)
    k0 = 0
    sl0 = c.block(k0)
    assert np.allclose(c["pix_lc"][:, sl0][0], owned[0] + 0.01 * np.arange(sl0.stop - sl0.start))
    assert np.all(c["peak_grid"][:, c.block(len(owned) - 1)] == 0.0)
