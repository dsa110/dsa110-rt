"""Owner engine (x2/search/engine.py), CPU path, end to end on a synthetic ring.

Small grids only (N = 32, 3 fines per DM); full-size runs belong to CP1.

* 2x: a tier-A and a tier-B DM on one half, frame-v2 rows (2 heavy int8 +
  4-bit mid-tread main), dispersed point-source bursts -> engine == float64
  reference (x2/ref/search2x_ref.py) on chunk maxima and candidates;
  external-clock units (tier B x2) and global fine indices.
* 1x-compat: the same engine at today's parameters vs the production
  functions composed on CPU (compat1x.production_cycle_cpu).
* plan plumbing: per-DM n_fdm from the plan, owner map d % 8, fail-loud
  build, C1 merger window pinned to today's wall-clock window, re-imaging.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsart.x2.ref import search2x_ref as ref
from dsart.x2.search.codec import RowFormat
from dsart.x2.search.engine import EngineConfig, OwnerEngine
from dsart.x2.search.geometry import owner_plan_from_tiers
from dsart.x2.search.ring import MemoryRing
from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells

N = 32
V = 24


def _tiers(t_a=16):
    fa = np.array([50.0, 52.0, 54.0, 56.0, 58.0])
    fb = np.array([100.0, 104.0, 108.0])
    return [dict(name="A", t_native=t_a, coarse=np.array([53.0, 57.0]), fine=fa,
                 f2c=(fa >= 55.0).astype(int)),
            dict(name="B", t_native=32, coarse=np.array([104.0]), fine=fb, f2c=np.zeros(3, int))]


def _setup_2x(bursts, *, seed=3, families=("b",), widths=(1, 2, 4, 8, 16, 32, 64), t_a=16,
              only=None):
    plan = owner_plan_from_tiers(_tiers(t_a), half=0, n_sub=1, csf=2, n_halves=1)
    if only is not None:
        from dsart.x2.search.geometry import OwnerPlan
        plan = OwnerPlan(half=0, dms=tuple(s for s in plan.dms if s.d in only),
                         n_stream=plan.n_stream, ext_native=16)
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells, inputs, formats = {}, {}, {}
    for spec in plan.dms:
        cells[spec.d] = random_value_cells(spec.n_stream, V, N, radius=6, seed=seed + spec.d,
                                           n_unused=2)
        inputs[spec.d] = make_dm_inputs(cells[spec.d], n_grid=N, fmt=fmt, step_main=0.3534,
                                        step_heavy=0.1)
        formats[spec.d] = fmt
    ring = MemoryRing(formats, plan.n_stream, t_buf=2048)
    sky = SyntheticSky(plan=plan, cells=cells, formats=formats, n_grid=N, bursts=list(bursts),
                       seed=seed)
    cfg = EngineConfig(n_grid=N, widths=widths, families=families, layer2_n_burnin=0,
                       fft_mode="pruned", n_tap_pixels=64)
    eng = OwnerEngine(plan, inputs, ring, cfg)
    return plan, ring, sky, eng, cells


def _fill(plan, ring, sky, last_cycle):
    for spec in plan.dms:
        sky.fill(ring, spec, 0, (last_cycle + 1) * spec.hop + spec.max_neg)


def _ref_for(eng, sky, cells, d, res, decode="chunk"):
    st = eng.states[d]
    spec = st.spec
    row0 = res.t0 - st.history - spec.max_pos
    n = st.history + spec.hop + spec.max_pos + spec.max_neg
    vals = sky.decoded_block(d, row0, n, spec.n_stream, V)
    filters = ref.ref_filter_bank(eng.cfg.widths, eng.cfg.families[0])
    return ref.ref_search_cycle(
        values=vals, cells=cells[d], shifts=spec.shifts, row0=row0, t0=res.t0, hop=spec.hop,
        delay=st.delay, history=st.history, n_grid=N, mask=eng.mask, image_scale=st.image_scale,
        filters=filters, s_k=res.s_k_applied, sigma=res.sigma_applied, decode=decode,
        n_top=int(round(224 * spec.hop / 192)), threshold=eng.cfg.threshold)


def _check_vs_ref(eng, sky, cells, res):
    for d, r in res.per_dm.items():
        rf = _ref_for(eng, sky, cells, d, r)
        mx = rf.chunk_max[0].reshape(r.best.shape)
        assert np.max(np.abs(r.best - mx)) < 0.02, (d, np.max(np.abs(r.best - mx)))
        got = {(c.t_tier, c.fdm_local, c.l, c.m): c.snr for c in r.candidates}
        want = {(c["t"], c["f"], c["l"], c["m"]): c["snr"] for c in rf.candidates}
        assert set(got) == set(want), (d, sorted(got)[:5], sorted(want)[:5])
        assert all(abs(got[key] - want[key]) < 0.02 for key in got)


def test_engine_2x_matches_reference_and_finds_bursts():
    bursts = [Burst(d=0, f=2, t=384 + 200, l=10, m=21, amp=1.6),
              Burst(d=2, f=1, t=192 + 100, l=19, m=12, amp=1.8, width=2)]
    plan, ring, sky, eng, cells = _setup_2x(bursts)
    assert [s.n_fdm for s in plan.dms] == [3, 2, 3]          # variable n_fdm, from the plan
    _fill(plan, ring, sky, 1)
    res = eng.process_cycle(1, keep_maps=True)
    _check_vs_ref(eng, sky, cells, res)
    a = [c for c in res.per_dm[0].candidates if (c.l, c.m) == (10, 21)]
    assert a and a[0].event_specnum == 584 and a[0].fdm_local == 2 and a[0].snr > 15
    assert a[0].fine_global == plan.by_d(0).fine_global[2] == 2
    b = [c for c in res.per_dm[2].candidates if (c.l, c.m) == (19, 12)]
    assert b and b[0].tier == "B"
    assert b[0].event_specnum == 2 * b[0].t_tier and b[0].width_ext % 2 == 0
    assert b[0].fine_global == 5 + 1                          # tier B after the 5 tier-A fines
    leg = b[0].to_legacy(search_node_id=0, gpu_half=0)
    assert leg.dm_idx == 6 and leg.event_specnum == b[0].event_specnum
    st = eng.status()
    assert st["dms"][0]["ok"] and st["dms"][2]["n_fdm"] == 3


def test_engine_bsplines_match_reference():
    bursts = [Burst(d=1, f=0, t=192 + 90, l=16, m=17, amp=0.6, width=5)]
    plan, ring, sky, eng, cells = _setup_2x(bursts, families=("t",), widths=(1, 2, 4, 8),
                                            t_a=32, only=(1,))
    _fill(plan, ring, sky, 1)
    res = eng.process_cycle(1, keep_maps=True)
    _check_vs_ref(eng, sky, cells, res)


def test_engine_b3x_bank_matches_reference():
    """Phase 0 b3x bank (box1, box2, [1,2,1], B3 4..64): 148 history rows,
    per-filter Layer-2 seeds and floors."""
    bursts = [Burst(d=1, f=1, t=192 + 120, l=14, m=18, amp=0.35, width=12)]
    plan, ring, sky, eng, cells = _setup_2x(bursts, families=("b3x",), t_a=32, only=(1,))
    st = eng.states[1]
    assert st.history == 148 and st.delay == 74
    assert eng.bank.kernel_ids() == ["b1", "b2", "t2", "q4", "q8", "q16", "q32", "q64"]
    floors = st.layer2._sigma_floor_vec.numpy()
    assert np.allclose(floors[:2], 0.5) and np.isclose(floors[2], 0.5 * np.sqrt(6) / np.sqrt(2))
    _fill(plan, ring, sky, 1)
    res = eng.process_cycle(1, keep_maps=True)
    _check_vs_ref(eng, sky, cells, res)
    best = [c for c in res.per_dm[1].candidates if (c.l, c.m) == (14, 18)]
    assert best and best[0].filt.startswith("q")


def test_engine_cycles_are_contiguous_and_stateful():
    plan, ring, sky, eng, cells = _setup_2x([], t_a=32, only=(0,))
    _fill(plan, ring, sky, 3)
    st = eng.states[0]
    r1 = eng.process_cycle(1)
    assert st.image_hi == 2 * st.spec.hop
    r2 = eng.process_cycle(2)
    assert st.image_hi == 3 * st.spec.hop                    # only the new hop imaged
    assert r2.per_dm[0].t0 == r1.per_dm[0].t0 + st.spec.hop
    s1 = r1.per_dm[0].sigma_applied
    assert np.all(np.abs(r2.per_dm[0].sigma_applied / s1 - 1) < 0.08)
    # pure noise, 2x image scale: Layer-1 sigma ~ 1 image unit
    assert np.all(np.abs(s1 - 1.0) < 0.15), s1
    assert all(c.snr < 12.0 for c in r2.candidates)
    # re-imaging a window = the detection images / applied sigma
    rows, img, vf = next(eng.reimage_iter(0, 400, 432, batch_rows=32))
    assert vf == 1.0 and img.shape == (32, st.spec.n_fdm, N, N)
    live = st.image[np.mod(rows, st.R_img)].astype(np.float32).reshape(img.shape)
    sig, found = eng.sigma_at(0, rows)
    assert found.all()
    assert np.allclose(img, live / sig[:, :, None, None], rtol=1e-6, atol=1e-7)


def test_engine_fails_loudly_without_a_csr():
    plan = owner_plan_from_tiers(_tiers(), half=0, n_sub=1, csf=2, n_halves=1)
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells = random_value_cells(16, V, N, radius=6, seed=0)
    inputs = {plan.dms[0].d: make_dm_inputs(cells, n_grid=N, fmt=fmt)}
    with pytest.raises(RuntimeError, match="build failed"):
        OwnerEngine(plan, inputs, MemoryRing({}, 16, 16), EngineConfig(n_grid=N))


def test_owner_map_and_global_indices():
    tiers = [dict(name="A", t_native=16, coarse=np.arange(12) * 10.0 + 55,
                  fine=np.arange(50, 170, 2.0), f2c=np.repeat(np.arange(12), 5)),
             dict(name="B", t_native=32, coarse=np.array([300.0, 400.0]),
                  fine=np.arange(250, 450, 20.0), f2c=np.repeat([0, 1], 5))]
    seen = []
    for h in range(8):
        p = owner_plan_from_tiers(tiers, half=h, n_sub=2, csf=2)
        for s in p.dms:
            assert s.d % 8 == h
            seen.extend(s.fine_global.tolist())
            assert s.n_stream == 32
            assert s.hop == (384 if s.t_native == 16 else 192)
    assert sorted(seen) == list(range(70))


def test_tiered_plan_owner_view():
    """The frozen WS-P TieredDmPlan through the owner adapter; the local
    stage-2 stand-in agrees with the plan's table."""
    from dsart.x2.plan import build_tiered_plan
    from dsart.x2.search.geometry import owner_plan_from_tiered, stage2_bins_subband

    plan = build_tiered_plan(n_sub=2)
    total = 0
    for h in range(8):
        op = owner_plan_from_tiered(plan, half=h)
        assert all(s.d % 8 == h for s in op.dms)
        assert op.plan_id == f"{int(plan.plan_id):016x}"
        total += sum(s.n_fdm for s in op.dms)
        for s in op.dms:
            assert s.n_fdm == int(plan.n_fdm[s.d]) and s.max_neg == 0
            assert s.max_pos == int(plan.stage2_max[s.d])
    assert total == plan.n_fine
    op = owner_plan_from_tiered(plan, half=3)
    s = op.dms[0]
    mine = stage2_bins_subband(s.fine_dm, t_tier_us=s.t_native * 32.768, csf=2, n_sub=2)
    assert np.array_equal(mine, s.shifts)


def test_c1_merger_window_constant_wall_clock():
    """Keep sample_period_specnum = 16 in 2x: T_ext units and doubled
    widths give today's window in ms (coordinator item 5)."""
    from dsart.x2.search.candidates import Candidate2x, merge_c1

    def c(t, w, snr, fg):
        return Candidate2x(half=0, d=0, tier="A", fdm_local=fg, fine_global=fg, dm=100.0, l=10,
                           m=10, t_tier=t, event_specnum=t, width_ext=w, support_ext=w,
                           filt=f"b{w}", snr=snr)
    # 1x: widths 2+2, window 0.5*4*16 = 32 samples of 1.048576 ms = 33.55 ms
    assert len(merge_c1([c(0, 2, 20, 0), c(32, 2, 15, 1)], sample_period_specnum=16)) == 1
    # 2x: same physical widths are 4+4 T_ext; window 64 T_ext = 33.55 ms
    assert len(merge_c1([c(0, 4, 20, 0), c(64, 4, 15, 1)], sample_period_specnum=16)) == 1
    assert len(merge_c1([c(0, 4, 20, 0), c(65, 4, 15, 1)], sample_period_specnum=16)) == 2


def test_engine_1x_compat_matches_production_chain():
    """The same engine at today's parameters vs production functions on CPU."""
    from dsart.x2.search.compat1x import compare_candidates, production_cycle_cpu
    from dsart.x2.search.csr import csr_from_lut_1x
    from dsart.x2.search.engine import DmInputs
    from dsart.x2.search.geometry import owner_plan_1x

    n_val = 30
    plan = owner_plan_1x(coarse_dm=np.array([102.0]), fine_dm=np.array([96.0, 102.0, 108.0]),
                         fine_to_coarse=np.zeros(3, int), owned=[0], n_sub=1, csf=8)
    spec = plan.dms[0]
    cells = random_value_cells(16, n_val, N, radius=6, seed=11, n_unused=4)
    lut = np.stack([np.where(c >= 0, c, 0) for c in cells]).astype(np.int32)
    nf = np.array([(c >= 0).sum() for c in cells], dtype=np.int32)
    fmt = RowFormat("cint8", n_values=n_val)
    csr = csr_from_lut_1x(lut, nf, n_grid=N)
    inputs = {0: DmInputs(fmt=fmt, csr=csr, c_v=np.ones((16, n_val), np.float32), offset=0.0)}
    ring = MemoryRing({0: fmt}, 16, t_buf=1536)
    sky = SyntheticSky(plan=plan, cells={0: cells}, formats={0: fmt}, n_grid=N, seed=5,
                       bursts=[Burst(d=0, f=1, t=3 * 192 + 40, l=11, m=20, amp=1.2)])
    sky.fill(ring, spec, 0, 5 * 192 + spec.max_neg + 4)
    cfg = EngineConfig.compat_1x(n_grid=N, layer2_n_burnin=0, fft_mode="pruned",
                                 n_tap_pixels=64)
    eng = OwnerEngine(plan, inputs, ring, cfg)
    eng.process_cycle(2)
    res = eng.process_cycle(3, keep_maps=True).per_dm[0]
    S = res.t0 - 64                                          # today's cube start
    pad_l, pad_r = spec.max_pos, spec.max_neg
    raw, _ = ring.wait_window(0, S - pad_l, 256 + pad_l + pad_r)
    prod = production_cycle_cpu(cells=raw.view(np.int8), lut=lut, n_filled=nf,
                                shifts=spec.shifts, pad_left=pad_l, t_det=256, n_grid=N,
                                sigma_inv=1.0 / res.sigma_applied, s_k=res.s_k_applied,
                                specnum_start=S, fine_dm=spec.fine_dm)
    # K3b compat: chunk maxima of production's (fp16) max cube vs the engine's
    pm = prod.max_snr.astype(np.float32).reshape(3, 64, *prod.max_snr.shape[1:]).max(axis=1)
    pm = pm.reshape(res.best.shape)
    # production holds the cumsum in fp16 (an absolute error up to ~ulp of
    # the running sum, 0.016 at |cumsum| ~ 16-32) and stores the max in fp16:
    # the CPU gate is 0.03 + 3e-3 |S/N| (the GPU gate vs real taps is CP1's)
    tol = 0.03 + 3e-3 * np.abs(pm)
    assert np.all(np.abs(res.best - pm) <= tol), np.max(np.abs(res.best - pm) - tol)
    # every production candidate is found by the engine (same cell, |dS/N| <= 0.05)
    cmp = compare_candidates(prod.raw_candidates, res.candidates, dt=0, ddm=0, dlm=0, dsnr=0.05,
                             dm_key=lambda c: c.dm_idx if hasattr(c, "dm_idx") else c.fdm_local)
    assert not cmp["unmatched_a"], cmp
    assert any((c.l, c.m, c.fdm_local) == (11, 20, 1) for c in res.candidates)
