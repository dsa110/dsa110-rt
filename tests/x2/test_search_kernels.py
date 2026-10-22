"""Owner-engine kernel twins on CPU (x2/search: codec, combine, fft, filters,
candidates, valuemap, kernels_cuda twin logic).

Bit-exact where the kernels are exact (K0 codes, K1 at 1x vs the v4 kernel
transcription), float tolerance where they are not (FFT), and the float64
reference for everything else. Small arrays only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dsart.x2.ref import search2x_ref as ref
from dsart.x2.search.codec import RowFormat, pack_frame2_rows, unpack_frame2_rows
from dsart.x2.search.combine import StreamRing, k0_rows_to_ring, k1_twin, v4_emulate
from dsart.x2.search.csr import csr_from_lut_1x, csr_from_value_cells, csr_matches_v4
from dsart.x2.search.fft import FftPlan, image_from_uv_torch, image_from_uv_twin
from dsart.x2.search.filters import FilterBank, k3b_twin
from dsart.x2.search.kernels_cuda import k0_twin_bitgather


def test_frame2_pack_unpack_roundtrip_and_k0_bitgather():
    rng = np.random.default_rng(0)
    for bits, nh in ((4, 3), (5, 0), (6, 7)):
        fmt = RowFormat("frame2", n_values=37, n_heavy=nh, bits=bits)
        lo = -(1 << (bits - 1))
        re = rng.integers(lo, -lo, size=(9, 37))
        im = rng.integers(lo, -lo, size=(9, 37))
        re[:, :nh] = rng.integers(-128, 128, size=(9, nh))
        im[:, :nh] = rng.integers(-128, 128, size=(9, nh))
        raw = pack_frame2_rows(re, im, fmt)
        assert raw.shape == (9, fmt.row_bytes)
        r2, i2 = unpack_frame2_rows(raw, fmt)
        assert np.array_equal(r2, re) and np.array_equal(i2, im)
        r3, i3, nd = k0_twin_bitgather(raw, fmt)
        assert np.array_equal(r3, re) and np.array_equal(i3, im)
        assert np.array_equal(nd[:, nh:], (re[:, nh:] == lo) | (im[:, nh:] == lo))


def test_k0_nodata_sentinel_decodes_to_zero_and_flags_empty_rows():
    fmt = RowFormat("frame2", n_values=6, n_heavy=1, bits=4)
    re = np.array([[5, 1, -8, 2, 3, 4], [-128, -8, -8, -8, -8, -8]])
    im = np.array([[1, 1, 0, 2, 3, 4], [0, 0, 0, 0, 0, 0]])
    raw = pack_frame2_rows(re, im, fmt)[None]
    ring = StreamRing(1, 6, 8)
    k0_rows_to_ring(ring, raw, np.ones((1, 2), bool), row0=3, fmt=fmt)
    assert list(ring.codes[0, :, 3, 0]) == [5, 1, 0, 2, 3, 4]
    assert ring.valid[0, 3] and not ring.valid[0, 4]          # all-no-data row invalid
    assert ring.n_nodata == 1 + 6


def _lut_inputs(rng, n_sb=6, nfm=30, nf=24, n_grid=32):
    lut = np.zeros((n_sb, nfm), dtype=np.int32)
    for g in range(n_sb):
        lut[g, :nf] = rng.choice(n_grid * n_grid, size=nf, replace=False)
    return lut, np.full(n_sb, nf, dtype=np.int32)


def test_k1_twin_1x_bit_identical_to_v4_kernel():
    from dsart.image.tiled_combine_cuda import build_inverse_lut, build_sparse_cell_csr, pad_t

    rng = np.random.default_rng(1)
    N, t_rows, t_det = 32, 22, 16
    lut, nfl = _lut_inputs(rng)
    n_sb, nfm = lut.shape
    csr_v4 = build_sparse_cell_csr(build_inverse_lut(torch.from_numpy(lut), torch.from_numpy(nfl),
                                                     n_grid=N), n_grid=N)
    csr = csr_from_lut_1x(lut, nfl, n_grid=N)
    ok, msg = csr_matches_v4(csr, csr_v4)
    assert ok, msg
    codes = rng.integers(-127, 128, size=(n_sb, t_rows, nfm, 2)).astype(np.int8)
    codes[:, :, 24:] = 0
    shifts = rng.integers(-3, 4, size=(4, n_sb)).astype(np.int32)
    dense = np.zeros((n_sb, nfm, pad_t(t_rows), 2), dtype=np.int8)
    dense[:, :, :t_rows] = codes.transpose(0, 2, 1, 3)
    want = v4_emulate(dense, csr_v4, shifts, t_rows=t_rows, t_out_len=t_det, t_lo=0, n_grid=N)
    ring = np.zeros((n_sb, nfm, 64, 2), dtype=np.int8)          # rows outside data: zero
    ring[:, :, :t_rows] = codes.transpose(0, 2, 1, 3)
    uv = k1_twin(ring, c_v=np.ones((n_sb, nfm), np.float32), offset=0.0, csr=csr,
                 shifts=shifts, t_lo=0, n_t=t_det)
    got = np.zeros_like(want)
    got[:, :, csr.cell_u, csr.cell_w, 0] = uv.real.astype(np.float16)
    got[:, :, csr.cell_u, csr.cell_w, 1] = uv.imag.astype(np.float16)
    assert np.array_equal(got, want)


def test_k1_twin_many_to_one_with_weights_equals_reference_gridding():
    rng = np.random.default_rng(2)
    N, S, V, T = 32, 5, 40, 6
    cells = []
    for _ in range(S):
        c = rng.choice(N * N, size=V, replace=False)
        c[:8] = c[8:16]                       # many-to-one: repeated cells in a stream
        c[-3:] = -1
        cells.append(c.astype(np.int64))
    csr = csr_from_value_cells(cells, n_grid=N)
    cv = rng.uniform(0.1, 3.0, size=(S, V)).astype(np.float32)
    codes = rng.integers(-7, 8, size=(S, V, 16, 2)).astype(np.int8)
    shifts = rng.integers(0, 4, size=(3, S)).astype(np.int32)
    uv = k1_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=shifts, t_lo=6, n_t=T)
    vals = [((codes[s, :, :, 0].T.astype(float)) + 1j * codes[s, :, :, 1].T) * cv[s]
            for s in range(S)]
    G = ref.ref_grid_dedispersed(vals, cells, shifts, row0=0, out_rows=np.arange(6, 6 + T),
                                 n_grid=N)
    u = np.arange(N)[:, None]
    w = np.arange(N // 2 + 1)[None, :]
    Gh = 0.5 * (G[..., :N // 2 + 1] + np.conj(G[..., (-u) % N, (-w) % N])) \
        * np.where((u + w) % 2, -1.0, 1.0)
    want = Gh[:, :, csr.cell_u, csr.cell_w]
    assert np.max(np.abs(uv - want)) < 1e-5 * np.abs(want).max()
    # every non-zero half cell is an active CSR cell
    mask = np.zeros(Gh.shape[-2:], bool)
    mask[csr.cell_u, csr.cell_w] = True
    assert np.all(np.abs(Gh[..., ~mask]) == 0)


@pytest.mark.parametrize("n_grid,fold", [(64, "col"), (64, "row"), (256, "row")])
def test_fft_twin_and_pruned_equal_irfft2(n_grid, fold):
    rng = np.random.default_rng(3)
    S, V = 4, 60 if n_grid == 64 else 300
    cells = []
    h = n_grid // 2
    for _ in range(S):
        uu = rng.integers(-h // 3, h // 3, size=V) + h
        ww = rng.integers(-h // 3, h // 3, size=V) + h
        cells.append(uu * n_grid + ww)
    csr = csr_from_value_cells(cells, n_grid=n_grid, fold=fold)
    plan = FftPlan.build(csr)
    assert plan.fft_lines.size + plan.const_lines.size == np.unique(csr.cell_line).size
    B = 3
    # a Hermitian-consistent half spectrum: fold a random full plane
    Gf = np.zeros((B, n_grid, n_grid), dtype=np.complex128)
    for c in cells:
        Gf[:, c // n_grid, c % n_grid] += rng.standard_normal((B, c.size)) \
            + 1j * rng.standard_normal((B, c.size))
    u = csr.cell_u
    w = csr.cell_w
    uv = (0.5 * (Gf[:, u, w] + np.conj(Gf[:, (-u) % n_grid, (-w) % n_grid]))).astype(np.complex64)
    want = np.fft.ifft2(Gf).real
    mask = ref.ref_edge_mask(n_grid, 1)
    scale = 3.0
    exp = want * mask * scale
    tw = image_from_uv_twin(uv, plan, mask * scale, out_dtype=None)
    assert np.max(np.abs(tw - exp)) < 2e-6 * np.abs(exp).max()
    for mode in ("pruned", "irfft2"):
        tt = image_from_uv_torch(torch.from_numpy(uv), plan, torch.from_numpy(mask * scale),
                                 mode=mode).numpy()
        assert np.max(np.abs(tt - exp)) < 2e-6 * np.abs(exp).max()


@pytest.mark.parametrize("bank", ["box", "b3x"])
def test_k3b_twin_equals_reference_chunk_max(bank):
    rng = np.random.default_rng(4)
    fb = FilterBank.build() if bank == "box" else FilterBank.b3x()
    F, N, hop = 2, 6, 128
    hist = fb.history()
    R = hop + hist
    t0 = 1000
    x = rng.standard_normal((R, F, N * N)).astype(np.float16)
    x[hist + 40, 1, 7] = 300.0                               # exercises the 250 clip
    rows = np.arange(t0 - hist, t0 + hop)
    ring = np.zeros_like(x)
    ring[np.mod(rows, R)] = x
    sig = np.array([1.3, 0.8])
    s_k = fb.seeds * 1.05
    best, tk, taps = k3b_twin(ring, ring_row0=0, c_first=t0 - fb.max_d, n_c=hop,
                              sigma_inv=(1 / sig).astype(np.float32),
                              s_inv=(1 / s_k).astype(np.float32), bank=fb, clip=250.0,
                              tap_pixels=np.array([3, 7, 20]))
    xn = np.clip(x.astype(np.float64) / sig[None, :, None], -250, 250).reshape(R, F, N, N)
    filters = (ref.ref_filter_bank([f.width for f in fb.filters], "b") if bank == "box"
               else ref.ref_filter_bank((), "b3x"))
    centres = np.arange(t0 - fb.max_d, t0 - fb.max_d + hop)
    sc = ref.ref_filter_scores(xn, int(rows[0]), filters, centres)
    snr = sc / s_k[:, None, None, None, None]
    mx, tin, kk = ref.ref_chunk_max(snr, 64)
    assert np.max(np.abs(best - mx.reshape(best.shape))) < 1e-3
    want_tap = sc.reshape(len(fb), hop, F, -1)[:, :, :, 7].transpose(0, 2, 1)
    assert np.max(np.abs(taps[:, :, 1, :] - want_tap)) < 1e-3 + 1e-6 * np.abs(sc).max()


def test_chunkmax_decoder_matches_reference_decoder():
    from dsart.x2.search.candidates import decode_chunkmax
    from dsart.x2.search.geometry import DmSpec

    rng = np.random.default_rng(5)
    fb = FilterBank.build()
    F, N, n_ch = 3, 8, 3
    best = (rng.standard_normal((n_ch, F, N * N)) * 2).astype(np.float32)
    tk = ((rng.integers(0, 64, size=best.shape) << 8) | rng.integers(0, 7, size=best.shape))
    for (ci, f, p, v) in [(0, 1, 10, 30.0), (1, 1, 11, 20.0), (2, 0, 60, 12.0), (2, 2, 3, 9.9)]:
        best[ci, f, p] = v
    spec = DmSpec(d=5, tier="A", t_native=16, coarse_dm=100.0, fine_dm=np.array([99., 100, 101]),
                  fine_global=np.array([40, 41, 42]), shifts=np.zeros((3, 2), int))
    got = decode_chunkmax(best, tk.astype(np.int32), c_first=500, chunk=64, n_grid=N, bank=fb,
                          threshold=9.55, n_top=50, spec=spec, half=5, ext_factor=1)
    tin = tk >> 8
    want = ref.ref_decode_topk(best.reshape(n_ch, F, N, N).astype(np.float64), threshold=9.55,
                               n_top=50, t_of=lambda ci, f, l_, m_: 500 + ci * 64
                               + int(tin.reshape(n_ch, F, N, N)[ci, f, l_, m_]))
    assert [(c.t_tier, c.fdm_local, c.l, c.m) for c in got] == [w[:4] for w in want]
    assert got[0].fine_global == 41 and got[0].d == 5 and got[0].half == 5


FIX = Path(__file__).resolve().parents[1] / "fixtures" / "x2" / "dsa110_antpos.npz"


def test_valuemap_adapter_real_2x_map_images_like_reference():
    """WS-P's frozen 2x value map (gridded, n_sub 2, production antpos):
    adapter CSR == its OwnerCsr; K1 + K2/K3a twins == float64 reference."""
    import yaml
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.opoint import opoint_2x
    from dsart.x2.search.valuemap import dm_inputs_from_value_map

    z = np.load(FIX)
    mask_bl = core_baseline_mask_from_station_numbers(z["station"])
    del yaml
    vm = build_value_map_for_opoint(opoint_2x("gridded", n_sub=2, bits_main=8), z["antpos_e"],
                                    z["antpos_n"], is_core_baseline_mask=mask_bl, dec_deg=71.63)
    inp = dm_inputs_from_value_map(vm, s_main=0.3534, s_heavy=0.0773, bits_main=8, fold="col")
    csr = inp.csr
    own = vm.owner
    assert csr.n_active == own.cell_uw.size and csr.n_entries == own.ent_stream.size
    nh = 129
    key_own = set(zip(own.cell_uw.tolist()))
    assert key_own == set(zip((csr.cell_u * nh + csr.cell_w).tolist()))
    assert csr.max_entries > 0 and csr.fold == "col"
    # today's single-side DC-at-centre layout (rows 5..210, cols 17..183):
    # K2 needs 112 half columns (col fold) vs 122 half rows -- NOT the plan's
    # 57; choose_fold picks the cheaper axis (CP1 K2 cost risk)
    cells_all = [sv.cell_row.astype(np.int64) * 256 + sv.cell_col for sv in vm.streams]
    from dsart.x2.search.csr import choose_fold
    rowc = csr_from_value_cells(cells_all, n_grid=256, fold="row")
    assert csr.fft_lines.size == 112 and rowc.fft_lines.size == 122
    assert choose_fold(cells_all, 256) == "col"
    # one row, one "fine DM" with zero shifts: twin chain == reference
    rng = np.random.default_rng(6)
    S = csr.n_stream
    V = int(inp.c_v.shape[1])
    codes = np.zeros((S, V, 4, 2), dtype=np.int8)
    for s in range(S):
        nv = vm.streams[s].n_values
        codes[s, :nv, 1] = rng.integers(-7, 8, size=(nv, 2))
    uv = k1_twin(codes, c_v=inp.c_v, offset=0.0, csr=csr, shifts=np.zeros((1, S), np.int32),
                 t_lo=1, n_t=1)
    plan = FftPlan.build(csr)
    msk = ref.ref_edge_mask(256, 1)
    img = image_from_uv_twin(uv.reshape(1, -1), plan, msk, out_dtype=None)[0]
    vals, cells = [], []
    for s in range(S):
        sv = vm.streams[s]
        nv = sv.n_values
        c = codes[s, :nv, 1].astype(float)
        vals.append(((c[:, 0] + 1j * c[:, 1]) * inp.c_v[s, :nv])[None, :])
        cells.append(sv.cell_row.astype(np.int64) * 256 + sv.cell_col.astype(np.int64))
    G = ref.ref_grid_dedispersed(vals, cells, np.zeros((1, S), int), row0=0, out_rows=[0],
                                 n_grid=256)
    want = ref.ref_image(G, msk, 1.0)[0, 0]
    assert np.max(np.abs(img - want)) < 2e-5 * np.abs(want).max()
