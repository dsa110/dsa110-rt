"""GpuImager.process_cube: compact (v3) combine == dense combine, bit for bit.

tests/test_combine_equivalence.py checks the v3 KERNEL against the dense
kernel. This checks the INTEGRATION: the imager calls v3 once per FFT
batch, writing in place into ``uv_batch[:n, t_lo:T_det]`` via
``out_row0``, then runs the same irfft2 + mask. The whole output cube
must match the dense path exactly, including the M7.7.2 carry-over
``t_lo > 0`` geometry and an FFT batch that does not divide n_fdm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("GPU imager test", allow_module_level=True)
pytest.importorskip("cupy")

from dsart.image.imager_gpu import build_default_gpu_imager  # noqa: E402
from dsart.image.tiled_combine_cuda import (  # noqa: E402
    build_inverse_lut,
    pad_t,
    transpose_compact_tmajor,
)
from dsart.transport.gpu_scatter import (  # noqa: E402
    scatter_compact_to_dense,
    zero_dense_rows,
)

DEV = torch.device("cuda:0")
N_GRID = 32
T_DET = 24
T_ROWS = 34
N_FILLED = 90
N_FILLED_MAX = 120


def _inputs(n_chg, n_fdm, seed):
    rng = np.random.default_rng(seed)
    lut = np.zeros((n_chg, N_FILLED_MAX), dtype=np.int32)
    for g in range(n_chg):
        lut[g, :N_FILLED] = rng.choice(N_GRID * N_GRID, N_FILLED, replace=False)
    nf = np.full(n_chg, N_FILLED, dtype=np.int32)
    cells = np.zeros((n_chg, T_ROWS, N_FILLED_MAX * 2), dtype=np.int8)
    cells[:, :, :N_FILLED * 2] = rng.integers(
        -60, 61, (n_chg, T_ROWS, N_FILLED * 2), dtype=np.int8)
    shifts = rng.integers(-5, 6, (n_fdm, n_chg)).astype(np.int32)
    t = lambda a: torch.from_numpy(a).to(DEV)  # noqa: E731
    return t(cells), t(lut), t(nf), t(shifts), shifts


@pytest.mark.parametrize("n_chg,n_fdm,t_lo,batch", [
    (16, 7, 0, "3"),     # batch does not divide n_fdm
    (64, 5, 8, "12"),    # 64 sub-band streams + carry-over rows
    (16, 12, 16, "5"),
])
def test_compact_imager_matches_dense(monkeypatch, n_chg, n_fdm, t_lo, batch):
    monkeypatch.setenv("DSART_IMAGER_FFT_BATCH", batch)
    cells, lut, nf, shifts, shifts_np = _inputs(n_chg, n_fdm, seed=n_chg + n_fdm)

    dense = torch.zeros((n_chg, T_ROWS, 2, N_GRID, N_GRID),
                        dtype=torch.int8, device=DEV)
    zero_dense_rows(dense=dense, t_det=T_ROWS)
    scatter_compact_to_dense(cells_packed=cells, lut=lut, n_filled_per_corr=nf,
                             dense=dense, t_det=T_ROWS, n_grid=N_GRID,
                             n_filled_max=N_FILLED_MAX)
    im_d = build_default_gpu_imager(n_grid=N_GRID, t_det=T_DET, n_fdm=n_fdm,
                                    n_chgroup=n_chg, device=DEV)
    want = im_d.process_cube(streams_cint8=dense, time_shifts_gpu=shifts,
                             t_lo=t_lo).clone()

    tmaj = torch.zeros((n_chg, N_FILLED_MAX, pad_t(T_ROWS), 2),
                       dtype=torch.int8, device=DEV)
    transpose_compact_tmajor(cells, tmaj)
    inv = build_inverse_lut(lut, nf, n_grid=N_GRID)
    im_c = build_default_gpu_imager(n_grid=N_GRID, t_det=T_DET, n_fdm=n_fdm,
                                    n_chgroup=n_chg, device=DEV)
    got = im_c.process_cube(
        streams_cint8=tmaj, time_shifts_gpu=shifts, t_lo=t_lo,
        compact_inv_lut=inv, compact_t_stream=T_ROWS,
        compact_shift_bounds=(int(shifts_np.min()), int(shifts_np.max())),
    )
    torch.cuda.synchronize()
    assert got.shape == want.shape
    rows = slice(t_lo, T_DET)
    assert torch.equal(got[rows], want[rows])


def test_compact_rejects_per_chgroup_calibration():
    cells, lut, nf, shifts, _ = _inputs(16, 4, seed=1)
    tmaj = torch.zeros((16, N_FILLED_MAX, pad_t(T_ROWS), 2),
                       dtype=torch.int8, device=DEV)
    transpose_compact_tmajor(cells, tmaj)
    inv = build_inverse_lut(lut, nf, n_grid=N_GRID)
    im = build_default_gpu_imager(n_grid=N_GRID, t_det=T_DET, n_fdm=4,
                                  n_chgroup=16, device=DEV)
    with pytest.raises(ValueError, match="unit-scale"):
        im.process_cube(
            streams_cint8=tmaj, time_shifts_gpu=shifts,
            chgroup_scales=torch.ones(16, device=DEV),
            compact_inv_lut=inv, compact_t_stream=T_ROWS)
