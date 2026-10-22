"""float64 search reference (x2/ref/search2x_ref.py) vs today's production code at 1x.

Every check compares the independent reference against the production
function that does the same job on CPU: the CPU imager path
(combine_chgroups + dirty_image_from_uv_grid + apply_edge_mask), the
inverse-LUT / v4 CSR and a literal transcription of the v4 combine kernel,
the padded-cumsum boxcars, the sigma-clip estimator, today's top-k
decoder, and a whole 1x cube composed from those (compat1x).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.x2.ref import search2x_ref as ref

N = 32


def _rand_streams(rng, n_stream, n_val, t_rows):
    cells = []
    vals = []
    for _ in range(n_stream):
        cells.append(rng.choice(N * N, size=n_val, replace=False).astype(np.int64))
        vals.append(rng.standard_normal((t_rows, n_val)) + 1j * rng.standard_normal((t_rows, n_val)))
    return cells, vals


def test_ref_image_equals_production_cpu_imager():
    from dsart.fine_dm.combiner import combine_chgroups, compute_time_shift_search
    from dsart.image.imager import apply_edge_mask, compute_edge_mask, dirty_image_from_uv_grid

    rng = np.random.default_rng(0)
    t_det = 24
    table = compute_time_shift_search(coarse_dm_pc_cm3=np.array([0.0]),
                                      fine_dm_pc_cm3=np.array([5.0, 12.0, 20.0]),
                                      fine_to_coarse=np.zeros(3, dtype=np.int64),
                                      t_int_search_us=524.288)
    t_stream = t_det + int(table.shifts.max()) + 2
    cells, vals = _rand_streams(rng, 16, 40, t_stream)
    dense = {}
    for g in range(16):
        d = np.zeros((t_stream, N * N), dtype=np.complex64)
        for t in range(t_stream):
            np.add.at(d[t], cells[g], vals[g][t].astype(np.complex64))
        dense[g] = d.reshape(t_stream, N, N)
    mask = compute_edge_mask(n_grid=N, kernel_support=1)
    G = ref.ref_grid_dedispersed(vals, cells, table.shifts, row0=0, out_rows=np.arange(t_det),
                                 n_grid=N)
    want = ref.ref_image(G, ref.ref_edge_mask(N, 1), 1.0 / 256)
    for f in range(3):
        uv = combine_chgroups(per_chgroup_streams=dense, time_shift_per_chgroup=table.shifts[f],
                              t_window=(0, t_det), n_grid=N)
        got = apply_edge_mask(dirty_image_from_uv_grid(uv, prescale=1.0 / 256), mask)
        scale = np.abs(want[f]).max()
        assert np.max(np.abs(got - want[f])) <= 2e-6 * scale


def test_fold_checkerboard_equals_fftshift_and_v4():
    """Hermitian fold x (-1)^(u+w), irfft2 == fftshift(Re ifft2); and the
    fold of the np.add.at grid == the v4 kernel transcription, bit for bit."""
    from dsart.image.tiled_combine_cuda import build_inverse_lut, build_sparse_cell_csr, pad_t
    from dsart.x2.search.combine import v4_emulate

    rng = np.random.default_rng(1)
    n_sb, nfm, nf, t_rows, t_det = 8, 40, 30, 20, 16
    lut = np.zeros((n_sb, nfm), dtype=np.int32)
    cells = []
    for g in range(n_sb):
        lut[g, :nf] = rng.choice(N * N, size=nf, replace=False)
        c = np.full(nfm, -1, dtype=np.int64)
        c[:nf] = lut[g, :nf]
        cells.append(c)
    n_filled = np.full(n_sb, nf, dtype=np.int32)
    codes = rng.integers(-127, 128, size=(n_sb, t_rows, nfm, 2)).astype(np.int8)
    codes[:, :, nf:] = 0
    shifts = rng.integers(-3, 4, size=(5, n_sb)).astype(np.int32)
    dense = np.zeros((n_sb, nfm, pad_t(t_rows), 2), dtype=np.int8)
    dense[:, :, :t_rows] = codes.transpose(0, 2, 1, 3)
    csr = build_sparse_cell_csr(build_inverse_lut(torch.from_numpy(lut),
                                                  torch.from_numpy(n_filled), n_grid=N), n_grid=N)
    got = v4_emulate(dense, csr, shifts, t_rows=t_rows, t_out_len=t_det, t_lo=0, n_grid=N)
    vals = [codes[g, :, :, 0].astype(float) + 1j * codes[g, :, :, 1] for g in range(n_sb)]
    G = ref.ref_grid_dedispersed(vals, cells, shifts, row0=0, out_rows=np.arange(t_det), n_grid=N)
    u = np.arange(N)[:, None]
    w = np.arange(N // 2 + 1)[None, :]
    Gm = G[..., (-u) % N, (-w) % N]
    Gh = 0.5 * (G[..., :, :N // 2 + 1] + np.conj(Gm)) * np.where((u + w) % 2, -1.0, 1.0)
    assert np.array_equal(Gh.real.astype(np.float16), got[..., 0])
    assert np.array_equal(Gh.imag.astype(np.float16), got[..., 1])
    img_half = np.fft.irfft2(Gh, s=(N, N))
    img_ref = np.fft.fftshift(np.fft.ifft2(G).real, axes=(-2, -1))
    assert np.max(np.abs(img_half - img_ref)) < 1e-12 * np.abs(img_ref).max()


def test_point_source_images_positive_at_its_pixel():
    rng = np.random.default_rng(2)
    cells = [rng.choice(N * N, size=60, replace=False) for _ in range(4)]
    vals = [ref.point_source_values(c, N, 9, 21, 2.0)[None, :] for c in cells]
    G = ref.ref_grid_dedispersed(vals, cells, np.zeros((1, 4), int), row0=0, out_rows=[0], n_grid=N)
    img = ref.ref_image(G, ref.ref_edge_mask(N, 1), 1.0)[0, 0]
    assert np.unravel_index(np.argmax(img), img.shape) == (9, 21)
    assert img[9, 21] == pytest.approx(2.0 * 240 / N ** 2, rel=1e-12)


def test_ref_boxcar_equals_production_padded_cumsum():
    from dsart.detector.forward import boxcar_from_padded_cumsum, precompute_padded_cumsum

    rng = np.random.default_rng(3)
    T, maxw = 160, 64
    x = rng.standard_normal((T, 2, 4, 4)).astype(np.float32)
    cs = precompute_padded_cumsum(torch.from_numpy(x), axis=0, max_width=maxw,
                                  accum_dtype=torch.float64)
    filters = ref.ref_filter_bank((1, 2, 4, 8, 16, 32, 64), "b")
    centres = np.arange(maxw // 2, T - maxw // 2)
    want = ref.ref_filter_scores(x.astype(np.float64), 0, filters, centres)
    for k, f in enumerate(filters):
        got = boxcar_from_padded_cumsum(cs, axis=0, width=f.width, max_width=maxw,
                                        n_out=centres.size, t_base=maxw // 2).numpy()
        assert np.max(np.abs(got - want[k])) < 1e-10


def test_bspline_taps_and_centring():
    for fam, order in (("b", 1), ("t", 2), ("g", 3)):
        for w in (1, 2, 3, 8):
            f = ref.ref_filter_bank([w], fam)[0]
            assert f.length == order * (w - 1) + 1
            assert f.taps.sum() == w ** order
            assert f.a == f.length // 2
    assert np.array_equal(ref.ref_filter_bank([3], "t")[0].taps, [1, 2, 3, 2, 1])


def test_ref_sigma_clip_equals_production_estimator():
    from dsart.noise_norm.layer1 import sigma_clipped_std_batched

    rng = np.random.default_rng(4)
    for m in (1000, 1001):
        x = rng.standard_normal(m)
        x[:7] = 40.0
        got = float(sigma_clipped_std_batched(torch.from_numpy(x[None].astype(np.float32)))[0])
        assert ref.ref_sigma_clipped_std(x.astype(np.float32)) == pytest.approx(got, rel=2e-6)


def test_ref_decoder_equals_production_topk_decoder():
    from dsart.detector.decoder import decode_topk_argmax_lowmem

    rng = np.random.default_rng(5)
    score = rng.standard_normal((96, 3, 8, 8)).astype(np.float32) * 3.0
    for (t, f, l_, m_, v) in [(10, 0, 2, 3, 30.0), (12, 1, 3, 3, 25.0), (60, 2, 7, 1, 12.0),
                              (61, 0, 0, 7, 11.0), (90, 1, 4, 4, 9.9)]:
        score[t, f, l_, m_] = v
    win = torch.zeros(score.shape, dtype=torch.int16)
    prod = decode_topk_argmax_lowmem(torch.from_numpy(score), win, threshold=9.55,
                                     kernel_ids=["unit:d1:b1"], kernel_time_widths=[1],
                                     merge_radius_lm=3, merge_radius_fdm=5, merge_radius_t=128,
                                     n_top=200)
    mine = ref.ref_decode_topk(score.astype(np.float64), threshold=9.55, n_top=200)
    assert [(c.event_specnum, c.dm_idx, int(c.l), int(c.m)) for c in prod] == \
        [(t, f, l_, m_) for (t, f, l_, m_, _v) in mine]
    assert [round(c.snr, 4) for c in prod] == [round(v, 4) for *_x, v in mine]


def test_ref_chunk_max_tie_order():
    snr = np.zeros((2, 128, 1))
    snr[1, 5, 0] = 3.0
    snr[0, 9, 0] = 3.0
    mx, tin, kk = ref.ref_chunk_max(snr, 64)
    assert mx[0, 0] == 3.0 and tin[0, 0] == 5 and kk[0, 0] == 1


def _onex_inputs(seed=6, burst_amp=6.0):
    """A 1x cube: 16 streams (n_sub 1), compact cint8 block, today's
    merged-rounding signed shifts, symmetric padding."""
    from dsart.fine_dm.combiner import compute_time_shift_search
    from dsart.x2.search.geometry import CYCLE_NATIVE

    rng = np.random.default_rng(seed)
    n_sb, nfm, nf = 16, 48, 40
    fine = np.array([95.0, 100.0, 105.0, 110.0])
    table = compute_time_shift_search(coarse_dm_pc_cm3=np.array([102.0]), fine_dm_pc_cm3=fine,
                                      fine_to_coarse=np.zeros(4, dtype=np.int64),
                                      t_int_search_us=1048.576, merge_coarse_rounding=True,
                                      t_int_corr_us=1048.576)
    shifts = table.shifts
    pad_l = int(max(0, shifts.max()))
    pad_r = int(max(0, -shifts.min()))
    t_det = 256
    S = 10 * (CYCLE_NATIVE // 32)                       # cube start (absolute, 1x samples)
    t_rows = t_det + pad_l + pad_r
    row0 = S - pad_l
    lut = np.zeros((n_sb, nfm), dtype=np.int32)
    cells = []
    h = N // 2
    for g in range(n_sb):
        u = rng.integers(-6, 7, size=200) + h
        w = rng.integers(-6, 7, size=200) + h
        lin = np.unique(u * N + w)[:nf]
        lut[g, :nf] = lin
        c = np.full(nfm, -1, dtype=np.int64)
        c[:nf] = lin
        cells.append(c)
    y = rng.standard_normal((n_sb, t_rows, nfm)) + 1j * rng.standard_normal((n_sb, t_rows, nfm))
    t_b, f_b = S + 120, 2
    for g in range(n_sb):
        vis = ref.point_source_values(cells[g], N, 11, 19, burst_amp)
        r = t_b - int(shifts[f_b, g]) - row0
        y[g, r] += vis
        y[g, :, nf:] = 0
    q = np.clip(np.rint(y * 20.0 / 3.0), -127, 127)
    codes = np.stack([q.real, q.imag], axis=-1).astype(np.int8)    # [n_sb, t_rows, nfm, 2]
    return dict(codes=codes, cells=cells, lut=lut, nf=nf, shifts=shifts, pad_l=pad_l,
                t_det=t_det, S=S, row0=row0, fine=fine, t_b=t_b, f_b=f_b)


def test_ref_1x_cube_equals_production_chain():
    """Whole 1x cube: reference vs production functions composed on CPU."""
    from dsart.x2.search.compat1x import compare_candidates, production_cycle_cpu

    p = _onex_inputs()
    n_sb = p["codes"].shape[0]
    vals = [p["codes"][g, :, :, 0].astype(float) + 1j * p["codes"][g, :, :, 1]
            for g in range(n_sb)]
    widths = (1, 2, 4, 8, 16, 32, 64)
    filters = ref.ref_filter_bank(widths, "b")
    s_k = np.sqrt(np.array(widths, float))
    t0 = p["S"] + 64
    mask = ref.ref_edge_mask(N, 1)
    kw = dict(values=vals, cells=p["cells"], shifts=p["shifts"], row0=p["row0"], t0=t0, hop=192,
              delay=32, history=64, n_grid=N, mask=mask, image_scale=1 / 256, filters=filters,
              s_k=s_k)
    first = ref.ref_search_cycle(decode="full", **kw)
    chunk = ref.ref_search_cycle(decode="chunk", sigma=first.sigma, **kw)
    sigma = first.sigma
    prod = production_cycle_cpu(cells=p["codes"].reshape(n_sb, -1, 2 * p["codes"].shape[2]),
                                lut=p["lut"], n_filled=np.full(n_sb, p["nf"], np.int32),
                                shifts=p["shifts"], pad_left=p["pad_l"], t_det=p["t_det"],
                                n_grid=N, sigma_inv=1.0 / sigma, s_k=s_k,
                                specnum_start=p["S"], fine_dm=p["fine"])
    # detector input: fp16 production cube vs float64 reference
    d = np.abs(prod.cube.astype(np.float64) - first.x_norm)
    assert d.max() < 4e-3 * max(1.0, np.abs(first.x_norm).max()), d.max()
    # decoded candidates (pre C1 merge on both sides): same cells, |dS/N| small
    class _C:
        def __init__(self, t, f, l_, m_, snr):
            self.event_specnum, self.dm_idx, self.l, self.m, self.snr = t, f, l_, m_, snr
    mine = [_C(c["t"], c["f"], c["l"], c["m"], c["snr"]) for c in first.candidates]
    cmp = compare_candidates(mine, prod.raw_candidates, dt=0, ddm=0, dlm=0, dsnr=0.05)
    assert not cmp["unmatched_a"] and not cmp["unmatched_b"], cmp
    assert any(c.event_specnum == p["t_b"] and c.dm_idx == p["f_b"] and (c.l, c.m) == (11, 19)
               for c in prod.raw_candidates)
    # the 2x chunk-max decoder finds every candidate today's decoder finds
    # (and more: today's top-224 cells saturate on one bright burst)
    mine_c = [_C(c["t"], c["f"], c["l"], c["m"], c["snr"]) for c in chunk.candidates]
    sup = compare_candidates(prod.raw_candidates, mine_c, dt=0, ddm=0, dlm=0, dsnr=0.05)
    assert not sup["unmatched_a"], sup
    assert len(mine_c) >= len(prod.raw_candidates)
