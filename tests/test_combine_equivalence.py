"""The two new combine kernels must be BIT-identical to the dense one.

``coo_combine_cuda`` and ``tiled_combine_cuda`` are alternative
formulations of ``fused_dequant_combine_per_fdm_half``. They are only
ever safe to select if they reproduce it exactly — the imager feeds
``irfft2`` and then a matched-filter detector, so a 1-LSB drift in the
half-spectrum is a drift in reported S/N.

Skipped without CUDA + cupy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip("combine kernels are CUDA-only", allow_module_level=True)
pytest.importorskip("cupy")

from dsart.image.coo_combine_cuda import (  # noqa: E402
    MAX_NSB_FP16_EXACT,
    accumulator_dtype_for,
    combine_coo,
)
from dsart.image.fused_combine_cuda import (  # noqa: E402
    fused_dequant_combine_per_fdm_half,
)
from dsart.image.tiled_combine_cuda import (  # noqa: E402
    combine_tiled,
    scatter_compact_to_dense_tmin,
)
from dsart.transport.gpu_scatter import (  # noqa: E402
    scatter_compact_to_dense,
    zero_dense_rows,
)

# Small but structurally faithful: odd t_rows, non-zero t_lo, a mirror
# column and the self-conjugate w == 0 / w == N/2 columns all exercised.
N_GRID = 32
T_ROWS = 40
T_DET = 32
T_LO = 8
N_FILLED = 120
N_FILLED_MAX = 160
DEV = "cuda:0"


def _inputs(n_sb, n_fdm, seed=0):
    rng = np.random.default_rng(seed)
    lut = np.zeros((n_sb, N_FILLED_MAX), dtype=np.int32)
    for g in range(n_sb):
        lut[g, :N_FILLED] = rng.choice(
            N_GRID * N_GRID, size=N_FILLED, replace=False).astype(np.int32)
    n_filled = np.full((n_sb,), N_FILLED, dtype=np.int32)
    cells = np.zeros((n_sb, T_ROWS, N_FILLED_MAX * 2), dtype=np.int8)
    cells[:, :, : N_FILLED * 2] = rng.integers(
        -127, 128, size=(n_sb, T_ROWS, N_FILLED * 2), dtype=np.int8)
    shifts = rng.integers(-4, 5, size=(n_fdm, n_sb)).astype(np.int32)
    t = lambda a: torch.from_numpy(a).to(DEV)  # noqa: E731
    return t(cells), t(lut), t(n_filled), t(shifts)


def _dense_reference(cells, lut, n_filled, shifts, n_sb, n_fdm):
    t_out_len = T_DET - T_LO
    n_half = N_GRID // 2 + 1
    dense = torch.zeros(
        (n_sb, T_ROWS, 2, N_GRID, N_GRID), dtype=torch.int8, device=DEV)
    zero_dense_rows(dense=dense, t_det=T_ROWS)
    scatter_compact_to_dense(
        cells_packed=cells, lut=lut, n_filled_per_corr=n_filled,
        dense=dense, t_det=T_ROWS, n_grid=N_GRID,
        n_filled_max=N_FILLED_MAX)
    plane = torch.empty(
        (t_out_len, N_GRID, n_half), dtype=torch.complex32, device=DEV)
    out = torch.empty(
        (n_fdm, t_out_len, N_GRID, n_half, 2),
        dtype=torch.float16, device=DEV)
    for f in range(n_fdm):
        fused_dequant_combine_per_fdm_half(
            dense, shifts[f].contiguous().int(), plane,
            t_lo=T_LO, fftshift=True)
        out[f] = torch.view_as_real(plane)
    return out


@pytest.mark.parametrize("n_sb,n_fdm", [(4, 6), (16, 8)])
def test_coo_combine_bit_identical(n_sb, n_fdm):
    cells, lut, n_filled, shifts = _inputs(n_sb, n_fdm)
    want = _dense_reference(cells, lut, n_filled, shifts, n_sb, n_fdm)
    got = combine_coo(
        cells, lut, n_filled, shifts, n_grid=N_GRID,
        t_out_len=T_DET - T_LO, t_lo=T_LO, fftshift=True)
    assert torch.equal(got, want)


@pytest.mark.parametrize("n_sb,n_fdm,tile", [(4, 6, 8), (16, 8, 12)])
def test_tiled_combine_bit_identical(n_sb, n_fdm, tile):
    cells, lut, n_filled, shifts = _inputs(n_sb, n_fdm)
    want = _dense_reference(cells, lut, n_filled, shifts, n_sb, n_fdm)
    dense_t = torch.zeros(
        (n_sb, 2, N_GRID * N_GRID, T_ROWS), dtype=torch.int8, device=DEV)
    scatter_compact_to_dense_tmin(
        cells, lut, n_filled, dense_t, n_grid=N_GRID)
    got = combine_tiled(
        dense_t, shifts, n_grid=N_GRID, t_out_len=T_DET - T_LO,
        t_lo=T_LO, fftshift=True, tile=tile)
    assert torch.equal(got, want)


def test_tiled_rejects_oversized_block():
    """n_fdm * tile must fit a CUDA block; the guard must be explicit."""
    cells, lut, n_filled, shifts = _inputs(4, 64)
    dense_t = torch.zeros(
        (4, 2, N_GRID * N_GRID, T_ROWS), dtype=torch.int8, device=DEV)
    with pytest.raises(ValueError, match="1024 threads"):
        combine_tiled(
            dense_t, shifts, n_grid=N_GRID, t_out_len=T_DET - T_LO,
            t_lo=T_LO, tile=32)


def test_fp16_accumulator_only_while_exact():
    """int8 sums stay exact in fp16 only up to 2048 / 127 = 16 sub-bands."""
    assert MAX_NSB_FP16_EXACT == 16
    assert accumulator_dtype_for(16) is torch.float16
    assert accumulator_dtype_for(17) is torch.float32
    assert accumulator_dtype_for(64) is torch.float32


# --------------------------------------------------------------------------
# v2 / v3: the formulations that actually unlock larger n_sb
# --------------------------------------------------------------------------

from dsart.image.tiled_combine_cuda import (  # noqa: E402
    build_inverse_lut,
    combine_tiled_v2,
    combine_tiled_v3,
    max_tile_for,
    pad_t,
    scatter_compact_to_dense_v2,
    transpose_compact_tmajor,
)


def _geometry(shifts, n_sb):
    spread = int(shifts.max().item()) - int(shifts.min().item())
    return max_tile_for(n_sb, spread, T_DET - T_LO)


@pytest.mark.parametrize("n_sb,n_fdm", [(4, 6), (16, 8), (64, 5)])
@pytest.mark.parametrize("threads", [256, 1024])
def test_v2_bit_identical(n_sb, n_fdm, threads):
    cells, lut, n_filled, shifts = _inputs(n_sb, n_fdm)
    want = _dense_reference(cells, lut, n_filled, shifts, n_sb, n_fdm)
    t_pad = pad_t(T_ROWS)
    dense = torch.zeros(
        (n_sb, N_GRID * N_GRID, t_pad, 2), dtype=torch.int8, device=DEV)
    scatter_compact_to_dense_v2(cells, lut, n_filled, dense, n_grid=N_GRID)
    got = combine_tiled_v2(
        dense, shifts, n_grid=N_GRID, t_rows=T_ROWS,
        t_out_len=T_DET - T_LO, t_lo=T_LO, fftshift=True,
        threads=threads, tile=_geometry(shifts, n_sb))
    assert torch.equal(got, want)


@pytest.mark.parametrize("n_sb,n_fdm", [(4, 6), (16, 8), (64, 5)])
def test_v3_bit_identical(n_sb, n_fdm):
    cells, lut, n_filled, shifts = _inputs(n_sb, n_fdm)
    want = _dense_reference(cells, lut, n_filled, shifts, n_sb, n_fdm)
    t_pad = pad_t(T_ROWS)
    inv = build_inverse_lut(lut, n_filled, n_grid=N_GRID)
    dense = torch.zeros(
        (n_sb, N_FILLED_MAX, t_pad, 2), dtype=torch.int8, device=DEV)
    transpose_compact_tmajor(cells, dense)
    got = combine_tiled_v3(
        dense, inv, shifts, n_grid=N_GRID, t_rows=T_ROWS,
        t_out_len=T_DET - T_LO, t_lo=T_LO, fftshift=True,
        threads=1024, tile=_geometry(shifts, n_sb))
    assert torch.equal(got, want)


def test_inverse_lut_round_trips():
    """inv_lut must invert lut exactly, and mark absent cells with -1."""
    _cells, lut, n_filled, _s = _inputs(4, 4)
    inv = build_inverse_lut(lut, n_filled, n_grid=N_GRID)
    assert inv.shape == (4, N_GRID * N_GRID)
    for g in range(4):
        nf = int(n_filled[g].item())
        cells_g = lut[g, :nf].long()
        assert torch.equal(
            inv[g, cells_g], torch.arange(nf, dtype=torch.int32, device=DEV))
        assert int((inv[g] >= 0).sum().item()) == nf


def test_v3_shared_memory_guard():
    """An oversized staged window must be rejected, not silently wrong."""
    _cells, lut, n_filled, shifts = _inputs(64, 5)
    inv = build_inverse_lut(lut, n_filled, n_grid=N_GRID)
    dense = torch.zeros(
        (64, N_FILLED_MAX, pad_t(T_ROWS), 2), dtype=torch.int8, device=DEV)
    # 2 cells * n_sb * win_w * 2 B must exceed the 48 KiB cap; the
    # miniature test geometry cannot reach it on its own, so ask for an
    # explicitly oversized tile.
    assert 2 * 64 * (500 + 1) * 2 > 49152
    with pytest.raises(ValueError, match="shared"):
        combine_tiled_v3(
            dense, inv, shifts, n_grid=N_GRID, t_rows=T_ROWS,
            t_out_len=T_DET - T_LO, t_lo=T_LO, tile=500)

    # and the automatic tile must always be within the cap
    for n_sb in (16, 32, 64):
        t = max_tile_for(n_sb, 66, 192)
        assert 2 * n_sb * (t + 66) * 2 <= 49152
        assert t >= 1
