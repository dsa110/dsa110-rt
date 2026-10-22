"""Corner-turn value maps (docs/x2/SPEC.md §3): redundancy, heavy class, wire order,
1x-compat equality with today's sub-band patterns / CSRs, owner fold, natural weighting,
pattern_id.

Antenna positions: ``tests/fixtures/x2/dsa110_antpos.npz`` = the production cal-blob
E/N (float32), the corr_setup_96 station order, and station-table heights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dsart.common.constants import NBASE
from dsart.grid.baseline_grid import (
    build_value_map,
    build_value_map_for_opoint,
    heavy_threshold,
    kept_core_baselines,
    redundancy_classes,
    value_map_pattern_id,
)
from dsart.grid.sparsity_pattern import (
    IMAGE_PIXEL_ARCSEC_TEE,
    _per_baseline_uv_meters,
    cell_lambda_for_pixel_arcsec,
    core_baseline_mask_from_station_numbers,
)
from dsart.x2.opoint import OPPOINT_1X, opoint_2x

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parents[1] / "fixtures" / "x2" / "dsa110_antpos.npz"
DEC = 71.63
CELL = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, 256)
N_IMAGE_2X = 3321 * 16 * 192          # core cross-baselines x summed channels (csf 2)


@pytest.fixture(scope="module")
def ant():
    z = np.load(FIX)
    order = yaml.safe_load((REPO / "configs" / "corr_setup_96.yaml").read_text())["antenna_order"]
    order = np.array([int(order[k]) for k in sorted(order, key=int)])
    assert np.array_equal(order, z["station"]), "fixture station order != configs/corr_setup_96.yaml"
    mask = core_baseline_mask_from_station_numbers(z["station"])
    return z["antpos_e"], z["antpos_n"], z["antpos_u"], mask


_MAPS: dict = {}


def vmap(ant, kind, n_sub, **kw):
    key = (kind, n_sub, tuple(sorted(kw.items())))
    if key not in _MAPS:
        e, n, u, mask = ant
        args = dict(antpos_u=u, is_core_baseline_mask=mask, kind=kind, n_sub=n_sub, chan_sum_factor=2,
                    dec_deg=DEC, cell_lambda=CELL)
        args.update(kw)
        _MAPS[key] = build_value_map(e, n, **args)
    return _MAPS[key]


# -- baselines + redundancy --------------------------------------------------------------
def test_kept_baselines_are_todays(ant):
    e, n, _, mask = ant
    bl, a, b = kept_core_baselines(mask)
    assert bl.shape == (3321,) and np.all(a > b) and np.all(mask[bl])
    du, dv = _per_baseline_uv_meters(e, n, is_core_baseline_mask=mask)
    assert np.array_equal(du, e[a].astype(float) - e[b].astype(float))
    assert np.array_equal(dv, n[a].astype(float) - n[b].astype(float))


def test_redundancy_classes_are_exact_and_3d_equals_2d(ant):
    e, n, u, mask = ant
    c2 = redundancy_classes(e, n, None, is_core_baseline_mask=mask)
    c3 = redundancy_classes(e, n, u, is_core_baseline_mask=mask)
    assert c2.n_kept == 3321
    assert c3.n_class == c2.n_class == 1748
    assert np.array_equal(c2.bl_class, c3.bl_class) and np.array_equal(c2.bl_conj, c3.bl_conj)
    assert int(c3.multiplicity.max()) == 41 and int(c3.multiplicity.sum()) == 3321
    # classes are tight and far apart: the result does not depend on the tolerance
    assert c3.max_diameter_m < 0.012 and c3.min_gap_m > 2.8
    for tol in (0.02, 0.5, 2.0):
        ct = redundancy_classes(e, n, u, is_core_baseline_mask=mask, tol_m=tol)
        assert np.array_equal(ct.bl_class, c3.bl_class) and np.array_equal(ct.bl_conj, c3.bl_conj)
    # heights: the station table has one nominal core elevation, |dU| is curvature only
    _, a, b = kept_core_baselines(mask)
    assert np.abs(u[a] - u[b]).max() < 0.01
    # every member's oriented vector is within tol of its class mean; conj = reversed vector
    V = np.stack([e[a] - e[b], n[a] - n[b], u[a] - u[b]], 1).astype(float)
    sgn = np.where(c3.bl_conj, -1.0, 1.0)[:, None]
    assert np.abs(V * sgn - c3.class_uvw_m[c3.bl_class]).max() < 0.05
    first = c3.class_members[c3.class_ptr[:-1]]
    assert not np.any(c3.bl_conj[first])                         # canonical = first member's orientation
    assert np.array_equal(c3.bl_class[first], np.arange(c3.n_class))


def test_model_rint_binning_overcounts_classes(ant):
    """docs/timeres_study/model quantised B/0.05 m with rint -> 1773: bin edges split
    baselines <= 1.2 cm apart. Every rint class lies inside exactly one exact class."""
    e, n, _, mask = ant
    c = redundancy_classes(e, n, None, is_core_baseline_mask=mask)
    _, a, b = kept_core_baselines(mask)
    BE = e[a].astype(float) - e[b].astype(float)
    BN = n[a].astype(float) - n[b].astype(float)
    q = np.stack([np.rint(BE / 0.05), np.rint(BN / 0.05)], 1).astype(np.int64)
    flip = (q[:, 1] < 0) | ((q[:, 1] == 0) & (q[:, 0] < 0))
    q[flip] *= -1
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    assert inv.max() + 1 == 1773
    for r in range(1773):
        assert np.unique(c.bl_class[inv == r]).size == 1
    assert len({(int(cc), int(r)) for cc, r in zip(c.bl_class, inv)}) == 1773


# -- value maps ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,n_sub", [("gridded", 2), ("gridded", 4), ("baselines", 2), ("baselines", 4)])
def test_counts_heavy_class_and_wire_order(ant, kind, n_sub):
    vm = vmap(ant, kind, n_sub)
    assert vm.n_streams == 16 * n_sub and vm.n_image == N_IMAGE_2X
    assert vm.heavy_threshold == heavy_threshold(N_IMAGE_2X, 1e-4) == 1021      # n >= N/1e4
    per = 192 // n_sub
    for st in vm.streams:
        nh = st.n_heavy
        assert np.all(st.heavy[:nh]) and not np.any(st.heavy[nh:])
        assert np.array_equal(st.heavy, st.n_contrib >= 1021)
        key = (st.cell_row.astype(np.int64) << 16) | st.cell_col
        for part in (slice(0, nh), slice(nh, None)):
            k, cid = key[part], st.class_id[part]
            assert np.all((np.diff(k) > 0) | ((np.diff(k) == 0) & (np.diff(cid) > 0)))
    s = vm.summary()
    if kind == "baselines":
        mult = vm.classes.multiplicity
        assert np.all(vm.n_values == 1748) and vm.n_dropped_offgrid == 0
        assert np.all(vm.n_heavy == int(np.sum(mult * per >= 1021)))
        assert int(vm.n_heavy[0]) == (80 if n_sub == 2 else 26)     # plan's 1773-class model: 71 / 23
        for st in vm.streams:
            assert np.array_equal(st.n_contrib, mult[st.class_id] * per)
    else:
        # plan: 3.1% of cells carry 43.5% of the weight (n_sub 2); 1.2% / 21.7% (n_sub 4)
        lo, hi, wlo, whi = (0.025, 0.035, 0.40, 0.46) if n_sub == 2 else (0.009, 0.015, 0.18, 0.24)
        assert lo < s["heavy_value_frac"] < hi and wlo < s["heavy_weight_frac"] < whi
        for st in vm.streams:
            assert np.all(st.class_id == -1) and st.legacy_pattern_id is not None


def test_corr_csr_sources(ant):
    e, n, _, mask = ant
    kept = set(kept_core_baselines(mask)[0].tolist())
    for kind in ("gridded", "baselines"):
        vm = vmap(ant, kind, 2)
        per = 96
        total = 0
        for sid in (0, 1, 17, 31):
            st = vm.streams[sid]
            c = vm.corr_csr(sid)
            assert np.array_equal(np.diff(c.ptr), st.n_contrib)
            assert np.all((c.ch >= st.sub_band * per) & (c.ch < (st.sub_band + 1) * per))
            assert set(np.unique(c.bl).tolist()) <= kept
            flat = c.bl.astype(np.int64) * 192 + c.ch
            for v in range(0, st.n_values, 97):
                f = flat[c.ptr[v]:c.ptr[v + 1]]
                assert np.all(np.diff(f) > 0)                              # fixed ascending order
            total += c.n_src
            if kind == "baselines":
                cl = vm.classes
                kidx = np.searchsorted(cl.kept_bl, c.bl)
                vv = np.repeat(np.arange(st.n_values), np.diff(c.ptr))
                assert np.array_equal(cl.bl_class[kidx], st.class_id[vv])
                assert np.array_equal(c.conj, cl.bl_conj[kidx])
            else:
                assert not np.any(c.conj)
        assert total == 4 * 3321 * per                                     # every source exactly once
    cat, voff = vmap(ant, "gridded", 2).corr_csr_chgroup(3)
    assert voff[-1] == cat.ptr.shape[0] - 1 and cat.n_src == 3321 * 192


def _half_spectrum(vm, x):
    """G_h from the owner CSR for stream values x[stream] (complex)."""
    o = vm.owner
    nh = o.n_half
    G = np.zeros((vm.n_grid, nh), dtype=np.complex128)
    val = np.array([x[s][v] for s, v in zip(o.ent_stream, o.ent_value)])
    val = np.where(o.ent_conj == 1, np.conj(val), val) * o.ent_weight
    cell = np.repeat(o.cell_uw, np.diff(o.cell_ptr))
    np.add.at(G.reshape(-1), cell, 0.5 * val)
    return G


@pytest.mark.parametrize("kind", ["gridded", "baselines"])
def test_owner_fold_images_like_the_full_grid(ant, kind):
    vm = vmap(ant, kind, 2)
    rng = np.random.default_rng(7)
    x = [rng.standard_normal(st.n_values) + 1j * rng.standard_normal(st.n_values) for st in vm.streams]
    N = vm.n_grid
    G = np.zeros((N, N), dtype=np.complex128)
    for st, xv in zip(vm.streams, x):
        np.add.at(G, (st.cell_row.astype(int), st.cell_col.astype(int)), xv * st.sqrt_n)
    full = np.fft.ifft2(G).real
    half = np.fft.irfft2(_half_spectrum(vm, x), s=(N, N))
    assert np.allclose(half, full, rtol=0, atol=1e-12 * np.abs(full).max())


@pytest.mark.parametrize("kind", ["gridded", "baselines"])
def test_natural_weighting_point_source_at_phase_centre(ant, kind):
    """All visibilities 1: corr sends sum/sqrt(n) = sqrt(n), owner re-weights by sqrt(n) ->
    every (baseline, channel) contributes weight 1, the image peak is N_image / N^2."""
    vm = vmap(ant, kind, 4)
    x = [st.n_contrib / np.sqrt(st.n_contrib.astype(float)) + 0j for st in vm.streams]
    N = vm.n_grid
    img = np.fft.irfft2(_half_spectrum(vm, x), s=(N, N))
    assert np.isclose(img[0, 0], vm.n_image / N ** 2, rtol=1e-6)         # float32 sqrt(n) weights
    assert np.isclose(img.max(), img[0, 0])


def test_1x_compat_gridded_map_equals_todays_patterns_and_csrs(ant):
    """kind gridded, n_sub 4, csf 8, no heavy class, unnormalised == production today."""
    import torch

    from dsart.grid.kernel import FastVisGridder
    from dsart.grid.sparsity_pattern import build_pattern
    from dsart.grid.subband import build_subband_layout, build_subband_patterns
    from dsart.image.tiled_combine_cuda import build_inverse_lut, build_sparse_cell_csr
    from dsart.services.triton_dedisp import build_cell_csr

    e, n, _, mask = ant
    vm = build_value_map_for_opoint(OPPOINT_1X, e, n, is_core_baseline_mask=mask, dec_deg=DEC)
    assert (vm.kind, vm.n_sub, vm.chan_sum_factor, vm.heavy_threshold, vm.normalise) == ("gridded", 4, 8, None, False)
    assert vm.cell_lambda == CELL
    luts, nfs = [], []
    for g in range(16):
        pats = build_subband_patterns(e, n, chgroup=g, n_sub=4, dec_deg=DEC, n_grid=256, kernel_support=1,
                                      chan_sum_factor=8, cell_lambda=CELL, is_core_baseline_mask=mask)
        whole = build_pattern(e, n, chgroup=g, dec_deg=DEC, n_grid=256, kernel_support=1, chan_sum_factor=8,
                              cell_lambda=CELL, is_core_baseline_mask=mask)
        layout = build_subband_layout(pats, FastVisGridder.from_pattern(whole, e, n, is_core_baseline_mask=mask))
        offs, cb, cc = (t.numpy() for t in build_cell_csr(layout.cell_index_map, n_filled=layout.n_filled_total,
                                                           nchan_eff=48, nbase=NBASE))
        for s, p in enumerate(pats):
            st = vm.streams[g * 4 + s]
            assert np.array_equal(st.cell_row, p.ix_row) and np.array_equal(st.cell_col, p.ix_col)
            assert st.legacy_pattern_id == p.pattern_id
            mine = vm.corr_csr(g * 4 + s)
            lo = int(layout.offsets[s])
            ref_ptr = offs[lo:int(layout.offsets[s + 1]) + 1] - offs[lo]
            assert np.array_equal(mine.ptr, ref_ptr)
            ref_flat = cb[offs[lo]:offs[int(layout.offsets[s + 1])]].astype(np.int64) * 48 \
                + cc[offs[lo]:offs[int(layout.offsets[s + 1])]]
            mine_flat = mine.bl.astype(np.int64) * 48 + mine.ch
            for v in range(st.n_values):                 # same sources per cell (order: ours is fixed)
                a, b = mine.ptr[v], mine.ptr[v + 1]
                assert np.array_equal(np.sort(ref_flat[a:b]), mine_flat[a:b])
            luts.append(vm.cell_flat(g * 4 + s))
            nfs.append(st.n_values)
    # owner side == today's v4 half-cell CSR (per half cell: stream -> (k, k_mirror))
    nmax = max(nfs)
    lut = torch.full((64, nmax), -1, dtype=torch.int32)
    for i, l in enumerate(luts):
        lut[i, :l.shape[0]] = torch.from_numpy(l)
    v4 = build_sparse_cell_csr(build_inverse_lut(lut, torch.tensor(nfs), n_grid=256), n_grid=256)
    ref = {}
    ptr = v4.cell_ptr.numpy()
    for i, uw in enumerate(v4.cell_uw.numpy()):
        for j in range(ptr[i], ptr[i + 1]):
            ref[(int(uw), int(v4.ent_g[j]))] = (int(v4.ent_k[j]), int(v4.ent_km[j]))
    o = vm.owner
    mine = {}
    for i, uw in enumerate(o.cell_uw):
        for j in range(o.cell_ptr[i], o.cell_ptr[i + 1]):
            k = mine.setdefault((int(uw), int(o.ent_stream[j])), [-1, -1])
            assert k[int(o.ent_conj[j])] == -1
            k[int(o.ent_conj[j])] = int(o.ent_value[j])
    assert {k: tuple(v) for k, v in mine.items()} == ref
    assert np.all(o.ent_weight == 1.0)


def test_pattern_id_covers_every_input(ant):
    e, n, u, mask = ant
    base = dict(kind="gridded", antpos_e=e, antpos_n=n, antpos_u=u, is_core_baseline_mask=mask,
                chan_sum_factor=2, n_sub=2, n_grid=256, cell_lambda=CELL, dec_deg=DEC,
                redundancy_tol_m=0.05, heavy_frac=1e-4, normalise=True)
    pid = value_map_pattern_id(**base)
    assert pid == vmap(ant, "gridded", 2).pattern_id
    assert value_map_pattern_id(**dict(base, dec_deg=DEC + 0.1)) == pid          # same 0.25 deg bin
    e2 = e.copy()
    e2[5] += 0.001
    u2 = u.copy()
    u2[7] += 0.001
    m2 = mask.copy()
    m2[np.flatnonzero(mask)[0]] = False
    for change in (dict(kind="baselines"), dict(n_sub=4), dict(chan_sum_factor=4), dict(n_grid=512),
                   dict(cell_lambda=CELL * 1.001), dict(dec_deg=DEC + 0.3), dict(redundancy_tol_m=0.04),
                   dict(heavy_frac=None), dict(normalise=False), dict(antpos_e=e2), dict(antpos_u=u2),
                   dict(antpos_u=None), dict(is_core_baseline_mask=m2)):
        assert value_map_pattern_id(**dict(base, **change)) != pid, change
    ids = vmap(ant, "gridded", 2).stream_pattern_ids
    assert len(set(ids)) == 32 and pid not in ids


def test_build_for_opoint_matches_explicit_call(ant):
    e, n, u, mask = ant
    a = build_value_map_for_opoint(opoint_2x("baselines", 2, 8), e, n, antpos_u=u, is_core_baseline_mask=mask,
                                   dec_deg=DEC)
    b = vmap(ant, "baselines", 2)
    assert a.pattern_id == b.pattern_id
    for sa, sb in zip(a.streams, b.streams):
        assert np.array_equal(sa.class_id, sb.class_id) and np.array_equal(sa.n_contrib, sb.n_contrib)
