"""Cutout producer (x2/search/cutout_service.py, SPEC §9).

* request / status codecs, validation (plan_id 0 = no check), queue depth,
  MANUAL after triggered, duplicates, v1 packets counted;
* the image source: tier-B ext rows 2k and 2k+1 are EXACT duplicates (C3's
  apex lookup), images = the detection images / the sigma applied at
  detection, snapshot slicing == ring reads;
* end to end through WS-D's assemble_cutout when x2/cutout/schema.py is
  present (skipped otherwise);
* the fine axis is the ENGINE's (its owned DMs, in its plane order): an
  engine owning a subset of its half's DMs writes a consistent cutout, and
  for a whole half the axis equals WS-D's ``fine_axis_for_half``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dsart.x2.search.codec import RowFormat
from dsart.x2.search.cutout_service import (ACCEPTED, FAILED, FLAG_MANUAL, R_BAD_REQUEST,
                                            R_PLAN_MISMATCH, R_QUEUE_FULL, CutoutImages,
                                            CutoutProducer, CutoutRequest, decode_request,
                                            encode_request, encode_status)
from dsart.x2.search.engine import EngineConfig, OwnerEngine
from dsart.x2.search.geometry import owner_plan_from_tiers
from dsart.x2.search.ring import MemoryRing
from dsart.x2.search.synth import SyntheticSky, make_dm_inputs, random_value_cells

N = 32
V = 20


_CELLS = {}


def _cells_of(eng, d, s):
    return _CELLS[d][s]


def _engine():
    fa = np.array([50.0, 53.0, 56.0])
    fb = np.array([100.0, 106.0])
    tiers = [dict(name="A", t_native=16, coarse=np.array([53.0]), fine=fa, f2c=np.zeros(3, int)),
             dict(name="B", t_native=32, coarse=np.array([103.0]), fine=fb, f2c=np.zeros(2, int))]
    plan = owner_plan_from_tiers(tiers, half=0, n_sub=1, csf=2, n_halves=1)
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells, inputs = {}, {}
    for s in plan.dms:
        cells[s.d] = random_value_cells(16, V, N, radius=6, seed=s.d + 4, n_unused=2)
        _CELLS[s.d] = cells[s.d]
        inputs[s.d] = make_dm_inputs(cells[s.d], n_grid=N, fmt=fmt, step_main=0.3534,
                                     step_heavy=0.1)
    ring = MemoryRing({s.d: fmt for s in plan.dms}, 16, t_buf=4096)
    sky = SyntheticSky(plan=plan, cells=cells, formats={s.d: fmt for s in plan.dms}, n_grid=N)
    for s in plan.dms:
        sky.fill(ring, s, 0, 4 * s.hop)
    eng = OwnerEngine(plan, inputs, ring, EngineConfig(n_grid=N, families=("b",), widths=(1, 2, 4),
                                                       fft_mode="pruned", n_tap_pixels=32,
                                                       layer2_n_burnin=0))
    for k in (1, 2):
        eng.process_cycle(k)
    return eng


class _Plan:                                              # the axis the schema needs
    def __init__(self, op):
        fines = np.concatenate([s.fine_global for s in op.dms])
        self._f = fines
        self.fine_dm = np.concatenate([s.fine_dm for s in op.dms])
        self.fine_to_coarse = np.concatenate([[s.d] * s.n_fdm for s in op.dms])
        self.fine_local_idx = np.concatenate([np.arange(s.n_fdm) for s in op.dms])
        self.fine_tier = np.concatenate([[0 if s.tier == "A" else 1] * s.n_fdm
                                         for s in op.dms])
        self.coarse_t_native = np.array([s.t_native for s in op.dms])

    def fines_of_half(self, h):
        return self._f

    def dms_of_half(self, h):
        return np.array([0, 1])


def _axis_plan(eng):
    return _Plan(eng.plan)


def _req(**kw):
    base = dict(flags=1 | 2, event_name="261022abcd", request_id=7, trigger_class_id=3,
                t_peak_ext=832, t0_ext=800, n_t=64, raw_pre_ext=40, raw_post_ext=40, width_ext=4,
                stamp_half_px=3, dm_peak_pc_cc=53.0, snr_peak=20.0, l_rad=0.0, m_rad=0.0,
                fine_global_peak=1, coarse_global_peak=0, l_pix=10, m_pix=12, n_stamp_t=16,
                stamp_dm_halfwidth_pc_cc=10.0, mjd_peak=61330.5, plan_id=0, deadline_s=1)
    base.update(kw)
    return CutoutRequest(**base)


def test_request_status_codecs_and_validation():
    r = _req()
    buf = encode_request(r)
    assert len(buf) == 128 and decode_request(buf) == replace(r, dm_peak_pc_cc=np.float32(53.0))
    st = encode_status(r, ACCEPTED, 0, sid=2, g=1, n_dm=3, n_ok=0)
    assert len(st) == 64 and st[:4] == b"DSCS"
    eng = _engine()
    p = CutoutProducer(eng, search_node_id=2, gpu_half=1, out_root="/nonexistent", plan_id=0xABC)
    assert p.offer(b"\0" * 64) is None and p.counters["c2_trigger_v1_ignored"] == 1
    assert p.offer(encode_request(_req(plan_id=0)))[1] == ACCEPTED          # 0 = no check
    got = p.offer(encode_request(_req(request_id=8, plan_id=0x123)))
    assert got[1:] == (FAILED, R_PLAN_MISMATCH)
    assert p.offer(encode_request(_req(request_id=9, plan_id=0xABC)))[1] == ACCEPTED
    assert p.offer(encode_request(_req(request_id=10, n_t=63)))[1:] == (FAILED, R_BAD_REQUEST)
    assert p.offer(encode_request(_req(request_id=11, raw_pre_ext=4000, raw_post_ext=4000)))[2] \
        == R_BAD_REQUEST
    # duplicate = same state, no new work
    n = len(p._q)
    assert p.offer(encode_request(_req(plan_id=0)))[1] == ACCEPTED and len(p._q) == n
    # depth 8, MANUAL served after triggered
    p2 = CutoutProducer(eng, search_node_id=2, gpu_half=1, out_root="/nonexistent")
    p2.offer(encode_request(_req(request_id=100, flags=FLAG_MANUAL)))
    for i in range(7):
        assert p2.offer(encode_request(_req(request_id=i)))[1] == ACCEPTED
    assert p2.offer(encode_request(_req(request_id=99)))[1:] == (FAILED, R_QUEUE_FULL)
    order = [p2.next_request().request_id for _ in range(8)]
    assert order[-1] == 100 and order[:7] == list(range(7))


def test_cutout_images_tier_b_rows_are_exact_duplicates_and_match_detection():
    eng = _engine()
    t0, n_t = 800, 64                                         # ext rows (even start)
    src = CutoutImages(eng, t0_ext=t0, n_t=n_t)
    a_spec, b_spec = sorted(eng.plan.dms, key=lambda s: s.d)
    fb = a_spec.n_fdm                                         # first tier-B plane fine
    img_b = src(fb)
    assert img_b.shape == (n_t, N, N)
    for k in range(0, n_t, 2):                                # rows 2k, 2k+1 bit-identical
        assert np.array_equal(img_b[k].view(np.uint16), img_b[k + 1].view(np.uint16))
    assert not np.array_equal(img_b[0], img_b[2])
    # tier A: one row per ext row, = the detection image / the applied sigma
    img_a = src(1)
    st = eng.states[a_spec.d]
    rows = np.arange(t0, t0 + n_t)
    live = st.image[np.mod(rows, st.R_img), 1].astype(np.float32)
    sig, found = eng.sigma_at(a_spec.d, rows)
    assert found.all()
    assert np.allclose(img_a.astype(np.float32).reshape(n_t, -1),
                       (live / sig[:, 1:2]).astype(np.float16).astype(np.float32), atol=2e-3)
    sa = src.sigma_applied()
    assert np.array_equal(sa[0::2, fb], sa[1::2, fb])
    rv, ivf = src.valid()
    assert rv.all() and np.all(ivf == 1.0)


def test_snapshot_slicing_equals_ring_reads():
    eng = _engine()
    p = CutoutProducer(eng, search_node_id=0, gpu_half=0, out_root="/nonexistent")
    req = _req()
    snaps, too_late, nbytes = p.raw_snapshot(req)
    assert not too_late and nbytes > 0
    a = CutoutImages(eng, t0_ext=req.t0_ext, n_t=req.n_t, raw=snaps)
    b = CutoutImages(eng, t0_ext=req.t0_ext, n_t=req.n_t)
    for f in range(5):
        assert np.array_equal(a(f), b(f))


def test_end_to_end_with_ws_d_schema(tmp_path):
    schema = pytest.importorskip("dsart.x2.cutout.schema")
    eng = _engine()

    p = CutoutProducer(eng, search_node_id=0, gpu_half=0, out_root=tmp_path, plan=_axis_plan(eng))
    arrays, status = p.build_arrays(_req())
    assert status == "complete"
    npz_path, _manifest = schema.write_cutout(tmp_path / "cutouts", arrays)
    cut = schema.read_cutout(Path(npz_path), check=True)
    pg = cut["peak_grid"]
    fb = 3                                                    # first tier-B plane fine
    assert np.array_equal(pg[0::2, fb], pg[1::2, fb])
    assert np.array_equal(cut["sigma_applied"][0::2, fb], cut["sigma_applied"][1::2, fb])


class _SV:
    def __init__(self, cells, n_heavy):
        c = np.asarray(cells)
        self.cell_row = np.where(c >= 0, c // N, -1)
        self.cell_col = np.where(c >= 0, c % N, -1)
        self.n_values = c.size
        self.n_heavy = n_heavy
        self.n_contrib = np.ones(c.size, int)


class _VM:
    def __init__(self, cells_per_stream, n_heavy):
        self.streams = [_SV(c, n_heavy) for c in cells_per_stream]
        self.n_grid = N
        self.normalise = False
        self.heavy_threshold = 1
        self.pattern_id = 12345


def test_raw_archive_d4_and_offline_reimage_matches_the_producer(tmp_path):
    from dsart.x2.cutout.raw_archive import is_raw_complete, read_raw_dir
    from dsart.x2.ref.search2x_ref import reimage_from_raw

    eng = _engine()
    p = CutoutProducer(eng, search_node_id=0, gpu_half=0, out_root=tmp_path)
    req = _req(t_peak_ext=832, raw_pre_ext=48, raw_post_ext=48)
    out = tmp_path / "raw_s0_g0"
    snaps, too_late, nbytes = p.raw_snapshot(req, out)
    assert is_raw_complete(out) and not too_late
    slabs = read_raw_dir(out)
    assert sorted(slabs) == [0, 1] and slabs[1].t_native == 32
    src = CutoutImages(eng, t0_ext=800, n_t=64, raw=snaps)
    a_spec, b_spec = sorted(eng.plan.dms, key=lambda s: s.d)
    for spec, f_plane, f_loc in ((a_spec, 1, 1), (b_spec, a_spec.n_fdm + 1, 1)):
        vm = _VM([_cells_of(eng, spec.d, s) for s in range(spec.n_stream)], n_heavy=2)
        rows_t = (800 + np.arange(64)) // eng.plan.ext_factor(spec)
        sig, _ = eng.sigma_at(spec.d, rows_t)
        img = reimage_from_raw(slabs[spec.d], f_loc, 800, 64,
                               ext_factor=eng.plan.ext_factor(spec), value_map=vm,
                               s_main=0.3534, s_heavy=0.1, sigma=sig[:, f_loc])
        mine = src(f_plane).astype(np.float64)
        assert np.max(np.abs(img - mine)) < 4e-3 * max(1.0, np.abs(mine).max())
        if spec.tier == "B":
            assert np.array_equal(img[0::2], img[1::2])


def test_serve_writes_raw_then_cutout_and_uploads_cutout_first(tmp_path):
    pytest.importorskip("dsart.x2.cutout.schema")
    from dsart.x2.cutout.raw_archive import is_raw_complete
    from dsart.x2.search.cutout_service import DONE

    eng = _engine()
    calls = []

    class _Up:
        def __init__(self, name):
            self.name = name

        def submit(self, ev, src):
            calls.append((self.name, ev, Path(src).name))
            return True

    p = CutoutProducer(eng, search_node_id=2, gpu_half=1, out_root=tmp_path,
                       plan=_axis_plan(eng))
    p.uploaders = (_Up("cutout"), _Up("raw"))
    req = _req(raw_pre_ext=40, raw_post_ext=40)
    state, reason, info = p.serve(req, event_dir=tmp_path / req.event_name)
    assert state == DONE and reason == 0, info
    assert is_raw_complete(tmp_path / req.event_name / "raw" / "raw_s2_g1")
    assert (tmp_path / req.event_name / "cutouts" / "cutout_s2_g1.json").is_file()
    assert [c[0] for c in calls] == ["cutout", "raw"]


def _engine_two_tier_a(owned):
    """Half 0 (one half) holds tier-A DMs 0, 1 and tier-B DM 2; the engine owns ``owned``."""
    from dsart.x2.search.geometry import OwnerPlan

    tiers = [dict(name="A", t_native=16, coarse=np.array([53.0, 60.0]),
                  fine=np.array([50.0, 53.0, 56.0, 58.0, 60.0, 62.0]),
                  f2c=np.array([0, 0, 0, 1, 1, 1])),
             dict(name="B", t_native=32, coarse=np.array([103.0]), fine=np.array([100.0, 106.0]),
                  f2c=np.zeros(2, int))]
    full = owner_plan_from_tiers(tiers, half=0, n_sub=1, csf=2, n_halves=1)
    plan = OwnerPlan(half=full.half, dms=tuple(full.by_d(d) for d in owned),
                     n_stream=full.n_stream, ext_native=full.ext_native, opoint=full.opoint,
                     plan_id=full.plan_id, extra=dict(full.extra))
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells, inputs = {}, {}
    for sp in plan.dms:
        cells[sp.d] = random_value_cells(16, V, N, radius=6, seed=sp.d + 4, n_unused=2)
        inputs[sp.d] = make_dm_inputs(cells[sp.d], n_grid=N, fmt=fmt, step_main=0.3534,
                                      step_heavy=0.1)
    ring = MemoryRing({sp.d: fmt for sp in plan.dms}, 16, t_buf=4096)
    sky = SyntheticSky(plan=plan, cells=cells, formats={sp.d: fmt for sp in plan.dms}, n_grid=N)
    for sp in plan.dms:
        sky.fill(ring, sp, 0, 4 * sp.hop)
    eng = OwnerEngine(plan, inputs, ring, EngineConfig(n_grid=N, families=("b",), widths=(1, 2, 4),
                                                       fft_mode="pruned", n_tap_pixels=32,
                                                       layer2_n_burnin=0))
    for k in (1, 2):
        eng.process_cycle(k)
    return eng, full


class _TieredView:
    """The half's tiered-plan arrays (global fine index) for ALL its DMs."""

    def __init__(self, full):
        specs = sorted(full.dms, key=lambda sp: sp.d)
        n = 1 + max(int(sp.fine_global.max()) for sp in specs)
        self.fine_dm = np.zeros(n)
        self.fine_to_coarse = np.full(n, -1)
        self.fine_local_idx = np.zeros(n, int)
        self.fine_tier = np.zeros(n, int)
        for sp in specs:
            self.fine_dm[sp.fine_global] = sp.fine_dm
            self.fine_to_coarse[sp.fine_global] = sp.d
            self.fine_local_idx[sp.fine_global] = np.arange(sp.n_fdm)
            self.fine_tier[sp.fine_global] = 0 if sp.tier == "A" else 1
        self.coarse_t_native = np.array([sp.t_native for sp in specs])
        self._specs = specs

    def fines_of_half(self, h):
        return np.concatenate([sp.fine_global for sp in self._specs])

    def dms_of_half(self, h):
        return np.array([sp.d for sp in self._specs])


def test_cutout_axis_is_the_engines_owned_dms(tmp_path):
    schema = pytest.importorskip("dsart.x2.cutout.schema")
    from dsart.x2.search.cutout_service import fine_axis_for_owner

    eng, full = _engine_two_tier_a(owned=(0, 2))           # DM 1 of the half is not ours
    view = _TieredView(full)
    p = CutoutProducer(eng, search_node_id=0, gpu_half=0, out_root=tmp_path, plan=view)
    arrays, status = p.build_arrays(_req(fine_global_peak=1, coarse_global_peak=0))
    assert status == "complete"
    npz_path, _manifest = schema.write_cutout(tmp_path / "cutouts", arrays)
    cut = schema.read_cutout(Path(npz_path), check=True)
    want = np.concatenate([full.by_d(0).fine_global, full.by_d(2).fine_global])
    assert np.array_equal(np.asarray(cut["fine_global_idx"]), want)
    assert np.array_equal(np.asarray(cut["coarse_owned"]), [0, 2])
    assert cut["peak_grid"].shape[1] == want.size
    fb = full.by_d(0).n_fdm                                    # first tier-B plane fine
    pg = cut["peak_grid"]
    assert np.array_equal(pg[0::2, fb], pg[1::2, fb])
    # without a tiered plan the DmSpecs give the same axis
    a1 = fine_axis_for_owner(eng.plan, view, t_ext_native=16)
    a0 = fine_axis_for_owner(eng.plan, None, t_ext_native=16)
    for key in a1:
        assert np.array_equal(a1[key], a0[key]) and a1[key].dtype == a0[key].dtype, key
    # an engine owning its whole half: WS-D's fine_axis_for_half, field for field
    whole = fine_axis_for_owner(full, view, t_ext_native=16)
    ref = schema.fine_axis_for_half(view, 0, t_ext_native=16)
    for key in ref:
        assert np.array_equal(whole[key], ref[key]) and whole[key].dtype == ref[key].dtype, key
