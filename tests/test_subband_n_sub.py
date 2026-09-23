"""Sub-band gridding (--n-sub): invariants of every new piece.

The whole-chgroup path (n_sub=1) must stay bit-identical; the sub-band
path must be exactly the physics described in dsart.grid.subband.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from dsart.coarse_dm.dm_plan import (  # noqa: E402
    DMPlan,
    build_summed_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
    subband_ref_freqs_table_GHz,
)
from dsart.common.constants import (  # noqa: E402
    K_DM_MS_GHZ2_PC,
    N_CHGROUP,
    NANTS,
    NATIVE_SAMPLE_US,
    NBASE,
)
from dsart.fine_dm.combiner import compute_time_shift_search  # noqa: E402
from dsart.grid.kernel import FastVisGridder  # noqa: E402
from dsart.grid.sparsity_pattern import (  # noqa: E402
    IMAGE_PIXEL_ARCSEC_TEE,
    build_pattern,
    cell_lambda_for_pixel_arcsec,
    predict_pattern_id,
)
from dsart.grid.subband import (  # noqa: E402
    build_subband_layout,
    build_subband_patterns,
    split_stream_id,
    stream_id,
)
from dsart.transport.prod_frame import BITS_CINT8_COMPLEX  # noqa: E402
from dsart.transport.tx import TransportTx  # noqa: E402

CSF = 8
NCH = 384 // CSF
T_US = 32 * NATIVE_SAMPLE_US
N_GRID = 256
CELL = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, N_GRID)
DEC = 71.63


def _antpos(seed=3):
    rng = np.random.default_rng(seed)
    # compact core so every baseline lands in-grid at the tee45 scale
    return (rng.uniform(-150, 150, NANTS).astype(np.float64),
            rng.uniform(-150, 150, NANTS).astype(np.float64))


def _plan(coarse):
    freqs = build_summed_chgroup_freq_table_GHz(CSF)
    dm = np.asarray(coarse, dtype=np.float64)
    return DMPlan(
        dm_pc_cc=dm, n_fine_per_coarse=25, t_int_fast_us=T_US,
        chgroup_freqs_GHz=freqs,
        _delay_native_samples_table=compute_delay_native_samples_table(dm, freqs),
        chan_sum_factor=CSF,
    )


# ---------------------------------------------------------------------------
# stream ids
# ---------------------------------------------------------------------------

def test_stream_id_round_trip():
    for n_sub in (1, 2, 4):
        seen = set()
        for g in range(N_CHGROUP):
            for s in range(n_sub):
                sid = stream_id(g, s, n_sub)
                assert split_stream_id(sid, n_sub) == (g, s)
                seen.add(sid)
        assert seen == set(range(N_CHGROUP * n_sub))


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

def test_subband_patterns_partition_the_whole_pattern():
    """cells(whole chgroup) == union of cells(sub-bands), and ids differ."""
    e, n = _antpos()
    whole = build_pattern(e, n, chgroup=5, dec_deg=DEC, n_grid=N_GRID,
                          chan_sum_factor=CSF, cell_lambda=CELL)
    subs = build_subband_patterns(
        e, n, chgroup=5, n_sub=4, dec_deg=DEC, n_grid=N_GRID,
        kernel_support=1, chan_sum_factor=CSF, cell_lambda=CELL,
        is_core_baseline_mask=None,
    )
    key = lambda p: set(  # noqa: E731
        (int(r) << 16) | int(c) for r, c in zip(p.ix_row, p.ix_col))
    union = set().union(*(key(p) for p in subs))
    assert union == key(whole)
    ids = {int(p.pattern_id) for p in subs} | {int(whole.pattern_id)}
    assert len(ids) == 5
    for s, p in enumerate(subs):
        assert p.sub_band == (s, 4)
        assert int(p.pattern_id) == predict_pattern_id(
            chgroup=5, dec_deg=DEC, n_grid=N_GRID, chan_sum_factor=CSF,
            cell_lambda=CELL, antpos_e=e, antpos_n=n, sub_band=(s, 4))


def test_whole_pattern_id_unchanged_by_sub_band_support():
    """sub_band=None must hash exactly as before (40-byte payload)."""
    e, n = _antpos()
    p = build_pattern(e, n, chgroup=2, dec_deg=DEC, n_grid=N_GRID,
                      chan_sum_factor=CSF, cell_lambda=CELL)
    assert p.sub_band is None
    assert int(p.pattern_id) == predict_pattern_id(
        chgroup=2, dec_deg=DEC, n_grid=N_GRID, chan_sum_factor=CSF,
        cell_lambda=CELL, antpos_e=e, antpos_n=n)


def test_sub_band_requires_explicit_cell_lambda():
    e, n = _antpos()
    with pytest.raises(ValueError, match="cell_lambda"):
        build_pattern(e, n, chgroup=0, dec_deg=DEC, n_grid=N_GRID,
                      chan_sum_factor=CSF, cell_lambda=None, sub_band=(0, 4))


# ---------------------------------------------------------------------------
# combined cell map
# ---------------------------------------------------------------------------

def test_layout_n_sub_1_is_the_gridder_map():
    e, n = _antpos()
    pats = build_subband_patterns(
        e, n, chgroup=7, n_sub=1, dec_deg=DEC, n_grid=N_GRID,
        kernel_support=1, chan_sum_factor=CSF, cell_lambda=CELL,
        is_core_baseline_mask=None,
    )
    g = FastVisGridder.from_pattern(pats[0], e, n, device="cpu")
    lay = build_subband_layout(pats, g)
    assert torch.equal(lay.cell_index_map, g.cell_index_map)
    assert lay.n_filled_total == int(pats[0].n_filled)


def test_layout_maps_each_channel_into_its_own_subband_only():
    e, n = _antpos()
    pats = build_subband_patterns(
        e, n, chgroup=11, n_sub=4, dec_deg=DEC, n_grid=N_GRID,
        kernel_support=1, chan_sum_factor=CSF, cell_lambda=CELL,
        is_core_baseline_mask=None,
    )
    whole = build_pattern(e, n, chgroup=11, dec_deg=DEC, n_grid=N_GRID,
                          chan_sum_factor=CSF, cell_lambda=CELL)
    gw = FastVisGridder.from_pattern(whole, e, n, device="cpu")
    lay = build_subband_layout(pats, gw)
    cim = lay.cell_index_map.numpy().reshape(NBASE, NCH)
    wmap = gw.cell_index_map.numpy().reshape(NBASE, NCH)
    per = NCH // 4
    total = lay.n_filled_total
    wkey = (whole.ix_row.astype(np.int64) << 16) | whole.ix_col.astype(np.int64)
    for s, p in enumerate(pats):
        skey = (p.ix_row.astype(np.int64) << 16) | p.ix_col.astype(np.int64)
        got = cim[:, s * per:(s + 1) * per]
        ref = wmap[:, s * per:(s + 1) * per]
        hit = ref < int(whole.n_filled)
        # same set of mapped sources as the whole-chgroup gridder ...
        assert np.array_equal(got < total, hit)
        # ... each landing in ITS OWN sub-band's cell range ...
        lo, hi = int(lay.offsets[s]), int(lay.offsets[s + 1])
        assert np.all((got[hit] >= lo) & (got[hit] < hi))
        # ... on exactly the same uv cell the whole-chgroup map uses
        assert np.array_equal(skey[got[hit] - lo], wkey[ref[hit]])


# ---------------------------------------------------------------------------
# stage-1 delays
# ---------------------------------------------------------------------------

def test_delay_bins_per_subband():
    plan = _plan(np.linspace(174.5, 1399.5, 8))
    for g in (0, 7, 15):
        whole = plan.delay_bins_per_chgroup(g)
        assert np.array_equal(plan.delay_bins_per_subband(g, 1), whole)
        sub = plan.delay_bins_per_subband(g, 4)
        per = NCH // 4
        refs = plan.subband_ref_freqs_GHz(g, 4)
        assert np.allclose(refs, plan.chgroup_freqs_GHz[g, ::per])
        assert np.all(sub[::per] == 0)                 # each sub-band top
        assert np.all(sub <= whole)                    # t_dedisp stays safe
        assert np.all(sub >= 0)
        nu = plan.chgroup_freqs_GHz[g]
        ref = np.repeat(refs, per)
        us = (K_DM_MS_GHZ2_PC * plan.dm_pc_cc[None, :]
              * (1 / nu[:, None] ** 2 - 1 / ref[:, None] ** 2) * 1e3)
        want = np.rint(np.rint(us / NATIVE_SAMPLE_US) / 32.0).astype(np.int64)
        assert np.array_equal(sub, want)


def test_search_side_refs_match_corr_plan():
    plan = _plan(np.linspace(174.5, 1399.5, 8))
    tab = subband_ref_freqs_table_GHz(CSF, 4)
    for g in range(N_CHGROUP):
        assert np.array_equal(tab[g], plan.subband_ref_freqs_GHz(g, 4))


# ---------------------------------------------------------------------------
# search shift table
# ---------------------------------------------------------------------------

def test_shift_table_n_sub():
    coarse = np.linspace(174.5, 1399.5, 8)
    fine = np.concatenate([np.linspace(c - 70, c + 70, 25) for c in coarse])
    f2c = np.repeat(np.arange(8), 25)
    kw = dict(coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine, fine_to_coarse=f2c,
              t_int_search_us=T_US, merge_coarse_rounding=True,
              t_int_corr_us=T_US)
    one = compute_time_shift_search(**kw)
    also_one = compute_time_shift_search(**kw, n_sub=1)
    assert np.array_equal(one.shifts, also_one.shifts)
    refs = subband_ref_freqs_table_GHz(CSF, 4)
    four = compute_time_shift_search(**kw, n_sub=4, nu_subband_ref_GHz=refs)
    assert four.shifts.shape == (200, 64)
    # sub-band 0 of each chgroup uses the summed-channel ref, the whole
    # chgroup the native top: they differ by < 1 sample
    assert np.max(np.abs(four.shifts[:, ::4] - one.shifts)) <= 1
    # lower sub-bands arrive later, so their shift is never larger
    for s in range(1, 4):
        assert np.all(four.shifts[:, s::4] <= four.shifts[:, (s - 1)::4])
    with pytest.raises(ValueError, match="merge_coarse_rounding"):
        compute_time_shift_search(
            **{**kw, "merge_coarse_rounding": False}, n_sub=4,
            nu_subband_ref_GHz=refs)


# ---------------------------------------------------------------------------
# TX encode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorised_encode_is_byte_identical(seed):
    rng = np.random.default_rng(seed)
    row = (rng.standard_normal((128, 2137)) * rng.uniform(0.1, 300)
           + 1j * rng.standard_normal((128, 2137))).astype(np.complex64)
    row[5] = 0          # all-zero tile -> scale 1.0
    row[9, :7] = 1e6    # a saturating outlier
    payloads, scales = TransportTx._encode_rows(row, BITS_CINT8_COMPLEX)
    for t in range(row.shape[0]):
        cells = row[t]
        re_im = np.stack([cells.real, cells.imag], axis=1)
        sc, off = TransportTx._compute_scale_offset(re_im)
        want = TransportTx._encode_payload(cells, BITS_CINT8_COMPLEX, sc)
        assert np.float32(scales[t]) == np.float32(sc)
        assert payloads[t].tobytes() == want
        assert float(off) == 0.0
