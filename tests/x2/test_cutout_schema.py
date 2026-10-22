"""x2 D3: dsart.cutout v1 schema + the event-products adapter."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from dsart.coinc import cube_veto as CV
from dsart.coinc.wire import CUTOUT_FLAG_CUTOUT, CUTOUT_FLAG_MANUAL, CutoutRequest
from dsart.common.contracts import CubeDumpManifest
from dsart.dump.cube_dump import CubeDumpWriter, CubeDumpWriterConfig
from dsart.x2.cutout import products as P
from dsart.x2.cutout import schema as S
from dsart.x2.opoint import opoint_2x
from dsart.x2.plan import build_tiered_plan_for_opoint

N = 16      # small test image side
NT = 32     # cutout rows


@lru_cache(maxsize=None)
def _plan(n_sub: int = 2):
    return build_tiered_plan_for_opoint(opoint_2x("gridded", n_sub, 8))


def _request(t0=1000, peak_off=16, f_glob=-1, coarse=-1, dm=-1.0, l=5, m=7,
             manual=False) -> CutoutRequest:
    flags = CUTOUT_FLAG_CUTOUT | (CUTOUT_FLAG_MANUAL if manual else 0)
    return CutoutRequest(
        event_name="261022abcd", request_id=3, t_peak_ext=t0 + peak_off,
        t0_ext=t0, n_t=NT, width_ext=4, dm_peak_pc_cc=dm, l_pix=l, m_pix=m,
        fine_global_peak=f_glob, coarse_global_peak=coarse,
        stamp_half_px=3, n_stamp_t=8, stamp_dm_halfwidth_pc_cc=10.0,
        flags=flags)


def _fields(h: int, req: CutoutRequest, burst_fine_local=None, seed=0):
    plan = _plan()
    axis = S.fine_axis_for_half(plan, h, t_ext_native=16)
    n_f = len(axis["fine_dm_pc_cc"])
    rng = np.random.default_rng(seed + h)
    imgs = rng.normal(size=(NT, n_f, N, N)).astype(np.float32)
    if burst_fine_local is not None:
        imgs[16, burst_fine_local, 5, 7] = 50.0
    n_d = len(axis["coarse_owned"])
    fields = S.assemble_cutout(
        axis=axis, images=imgs, n_t=NT,
        sigma_applied=np.ones((NT, n_f), np.float32),
        row_valid=np.ones((NT, n_d), bool),
        input_valid_frac=np.ones(n_f, np.float32), request=req,
        identity=dict(search_node_id=[1, 2, 9, 13][h // 2], gpu_half=h % 2,
                      half_index=h, opoint="2x-gridded-ns2-cint4",
                      plan_id=plan.plan_id, pattern_id=0x1234,
                      code_version="test"),
        clock=dict(t_ext_us=524.288, native_per_ext=16, mjd_t0=61335.5,
                   mjd_clock="test"),
        geometry=dict(n_grid=N, cell_l_rad=1e-3, cell_m_rad=1e-3),
        status_detail={"coarse": {}})
    return fields, imgs, axis


def test_fine_axis_covers_the_plan_once() -> None:
    plan = _plan()
    seen = []
    for h in range(8):
        ax = S.fine_axis_for_half(plan, h, t_ext_native=16)
        seen.extend(ax["fine_global_idx"].tolist())
        assert np.all(np.diff(ax["fine_global_idx"]) > 0)
        assert np.all(ax["coarse_owned"] % 8 == h)
        off = ax["fine_offset"]
        for k, d in enumerate(ax["coarse_owned"]):
            blk = ax["coarse_global_idx"][off[k]:off[k + 1]]
            assert np.all(blk == d)
            assert off[k + 1] - off[k] == plan.n_fdm[d]
        # tier A at T_ext, tier B at 2 T_ext
        assert set(ax["t_step_ext"][ax["tier"] == 0].tolist()) <= {1}
        assert set(ax["t_step_ext"][ax["tier"] == 1].tolist()) <= {2}
    assert sorted(seen) == list(range(plan.n_fine))


def test_reduce_images_matches_cube_max() -> None:
    rng = np.random.default_rng(1)
    imgs = rng.normal(size=(6, 3, 8, 8)).astype(np.float16)
    pg, pp, lc = S.reduce_images(imgs, 2, 3)
    assert np.array_equal(pg, imgs.max(axis=(2, 3)).astype(np.float32))
    assert np.array_equal(
        pg, np.take_along_axis(imgs.reshape(6, 3, 64), pp[..., None], 2)[..., 0])
    assert np.array_equal(lc, imgs[:, :, 2, 3].astype(np.float32))
    assert S.reduce_images(imgs)[2].shape == (0, 3)


def test_apex_matches_c3_definition() -> None:
    rng = np.random.default_rng(2)
    pg = rng.normal(size=(20, 7)).astype(np.float32)
    off = np.array([0, 3, 7])
    at, af = S.apex_per_block(pg, off)
    for k in range(2):
        blk = pg[:, off[k]:off[k + 1]]
        t_ap, f_ap = np.unravel_index(int(np.nanargmax(blk)), blk.shape)
        assert (at[k], af[k]) == (t_ap, f_ap)


def test_assemble_write_read_round_trip(tmp_path: Path) -> None:
    plan = _plan()
    h = 3
    ax = S.fine_axis_for_half(plan, h, t_ext_native=16)
    f_loc = 5
    fg = int(ax["fine_global_idx"][f_loc])
    req = _request(f_glob=fg, coarse=int(ax["coarse_global_idx"][f_loc]),
                   dm=float(ax["fine_dm_pc_cc"][f_loc]))
    fields, imgs, _ = _fields(h, req, burst_fine_local=f_loc)
    npz, man = S.write_cutout(tmp_path / "cutouts", fields)
    assert npz.name == "cutout_s2_g1.npz" and man.name == "cutout_s2_g1.json"
    assert not list((tmp_path / "cutouts").glob("*.tmp"))
    c = S.read_cutout(npz)
    assert c.sid == 2 and c.gpu_half == 1
    assert c.scalar("plan_id") == plan.plan_id
    assert c["peak_grid"][16, f_loc] == pytest.approx(50.0)
    assert int(c["peak_pix"][16, f_loc]) == 5 * N + 7
    assert c["pix_lc"][16, f_loc] == pytest.approx(50.0)
    k = int(np.searchsorted(c["fine_offset"], f_loc, side="right") - 1)
    assert int(c["apex_t"][k]) == 16
    assert int(c["fine_offset"][k] + c["apex_f"][k]) == f_loc
    assert c["apex_image"][k][5, 7] == pytest.approx(50.0, rel=1e-3)
    assert int(c.scalar("has_peak_image")) == 1
    assert c["peak_image"][5, 7] == pytest.approx(50.0, rel=1e-3)
    w_rows = imgs[14:18, f_loc, 5, 7].astype(np.float64).mean()
    assert float(c["peak_image_w"][5, 7]) == pytest.approx(w_rows, rel=2e-3)
    assert c["stamp"].shape[2:] == (7, 7) and c["stamp"].shape[0] == 8
    j = int(np.flatnonzero(c["stamp_fine"] == f_loc)[0])
    assert float(c["stamp"][4, j, 3, 3]) == pytest.approx(50.0, rel=1e-3)
    m = S.read_manifest(man)
    assert m["event_name"] == "261022abcd" and m["status"] == "complete"
    assert m["plan_id"] == f"0x{plan.plan_id:016x}"
    assert m["bytes"] == npz.stat().st_size and len(m["blake2b_128"]) == 32


def test_manual_request_has_no_pixel_content(tmp_path: Path) -> None:
    req = _request(l=-1, m=-1, manual=True)
    fields, _, _ = _fields(0, req)
    S.validate(S._coerce(fields))
    assert fields["pix_lc"].shape[0] == 0 and fields["stamp"].size == 0
    assert fields["has_peak_image"] == 1        # strongest apex of the half


def test_validate_rejects_bad_files() -> None:
    req = _request()
    fields, _, _ = _fields(1, req)
    good = S._coerce(fields)
    S.validate(good)
    bad = dict(good)
    bad["peak_grid"] = good["peak_grid"].astype(np.float64)
    with pytest.raises(S.CutoutError):
        S.validate(bad)
    bad = dict(good)
    bad["fine_offset"] = good["fine_offset"][::-1].copy()
    with pytest.raises(S.CutoutError):
        S.validate(bad)
    bad = dict(good)
    bad["t0_ext"] = np.asarray(1001, dtype=np.int64)
    with pytest.raises(S.CutoutError):
        S.validate(bad)
    bad = dict(good)
    del bad["row_valid"]
    with pytest.raises(S.CutoutError):
        S.validate(bad)
    bad = dict(good)
    bad["extra"] = np.zeros(1)
    with pytest.raises(S.CutoutError):
        S.validate(bad)


def test_filename_parse() -> None:
    assert S.parse_cutout_filename("x/cutout_s13_g1.npz") == (13, 1)
    assert S.parse_cutout_filename("cutout_s2_g0.json") == (2, 0)
    assert S.parse_cutout_filename("cube_s2_g0_5.npz") is None
    assert S.parse_cutout_filename("cutout_s2_g0.npz.tmp") is None


# ------------------------------------------------------------ products


def test_products_from_cutouts(tmp_path: Path) -> None:
    ev = tmp_path / "261022abcd"
    for h in (0, 1, 5):
        fields, _, _ = _fields(h, _request())
        S.write_cutout(ev / "cutouts", fields)
    prod = P.load_event_products(ev)
    assert prod.kind == "cutouts" and prod.halves == {(1, 0), (1, 1), (9, 1)}
    n_planes = sum(len(_plan().dms_of_half(h)) for h in (0, 1, 5))
    assert len(prod.planes) == n_planes
    fg = int(_plan().fines_of_half(5)[3])
    plane, j = prod.plane_of_fine(fg)
    assert plane.half == (9, 1) and plane.fine_global_idx[j] == fg
    p0 = prod.planes[0]
    assert p0.t_ext_us == pytest.approx(524.288) and p0.t0_ext == 1000
    img = p0.apex_image(p0.apex_t, p0.apex_f)
    assert img.shape == (N, N)
    with pytest.raises(KeyError):
        p0.apex_image(p0.apex_t + 1, p0.apex_f)
    kind, halves, n = P.halves_with_products(ev)
    assert kind == "cutouts" and n == 3
    summ = P.summarize(ev, expected=[(1, 0), (1, 1), (2, 0), (9, 1)])
    assert summ["missing"] == ["s2g0"] and not summ["complete"]
    assert summ["n_halves"] == 3


def test_products_manifest_is_the_completion_signal(tmp_path: Path) -> None:
    ev = tmp_path / "261022abcd"
    fields, _, _ = _fields(2, _request())
    npz, man = S.write_cutout(ev / "cutouts", fields)
    man.unlink()                                  # NPZ landed, manifest not yet
    assert P.halves_with_products(ev)[1] == set()


def _write_1x_cube(dump_root: Path, sid: int, g: int, spec: int, cube):
    w = CubeDumpWriter(CubeDumpWriterConfig(dump_root=dump_root,
                                            search_node_id=sid, gpu_half=g))
    m = CubeDumpManifest(
        cube_id=0, event_specnum_start=spec, mjd_start=60942.1,
        t_det=cube.shape[0], n_fdm_in_cube=cube.shape[1],
        n_grid=cube.shape[2], trigger_source="udp", cluster_record=None,
        npz_path="", search_node_id=sid, gpu_half=g)
    return w._write_one(cube, m)


def test_1x_cube_adapter_bit_identical(tmp_path: Path) -> None:
    """The adapter reproduces cube_veto's own peak_grid + apex-plane reads."""
    ev = tmp_path / "260801abcd"
    (ev / "cubes").mkdir(parents=True)
    rng = np.random.default_rng(5)
    for sid, g in ((1, 0), (2, 1), (13, 1)):
        cube = rng.normal(size=(12, 5, 8, 8)).astype(np.float16)
        _write_1x_cube(ev / "cubes", sid, g, 777, cube)
    (ev / "cubes" / "cube_sX_g0_1.npz").write_bytes(b"junk")   # C3 skips it
    prod = P.load_event_products(ev)
    assert prod.kind == "cubes" and prod.halves == {(1, 0), (2, 1), (13, 1)}
    for p in prod.planes:
        pg_ref, _, _ = CV._load_peak_grid(str(p.path))
        assert p.peak_grid.dtype == pg_ref.dtype
        assert np.array_equal(p.peak_grid, pg_ref)
        ref_cube = CV._mmap_cube(str(p.path))
        assert np.array_equal(p.apex_image(3, 2),
                              np.asarray(ref_cube[3, 2], dtype=np.float32))
        assert p.coarse == -1 and p.source == "cubes"
    kind, halves, n_files = P.halves_with_products(ev)
    assert kind == "cubes" and n_files == 4      # the raw glob count
    assert halves == {(1, 0), (2, 1), (13, 1)}


def test_no_products(tmp_path: Path) -> None:
    prod = P.load_event_products(tmp_path / "nothing")
    assert prod.kind == "none" and not prod.planes
    assert json.loads(json.dumps(P.summarize(tmp_path / "nothing")))["kind"] == "none"
