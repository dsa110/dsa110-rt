"""Parity tests for ``rx_ring_assemble_compact_block`` (M7.4.1 GPU-scatter).

The compact assembler emits the raw COO wire payload + sidecars in a
~30 MiB buffer (vs the dense path's ~565 MiB plane). The GPU scatter
kernel then expands the compact buffer into the same dense plane the
M7.4 imager already consumes. The end-to-end dense plane MUST be
byte-identical to ``rx_ring_assemble_dense_block`` for the same ring
state — otherwise we've changed numerics.

We test in three layers:

1. CPU-only compact↔dense parity (``test_compact_matches_dense``):
   call BOTH assemblers on the same ring state, expand the compact
   buffer in pure Python (mirrors the GPU kernel), and assert the
   dense plane matches byte-for-byte. Catches C-side scatter bugs.

2. CPU↔GPU scatter parity (``test_gpu_scatter_matches_python_ref``):
   call ``rx_ring_assemble_compact_block`` then run the GPU scatter
   kernel; the result must match a Python reference scatter on the
   same compact buffer. Catches NVRTC kernel bugs. cuda-skipped.

3. End-to-end compact+GPU vs dense parity (``test_gpu_pipeline_vs_dense_c``):
   run ``rx_ring_assemble_compact_block`` → ``gpu_scatter.zero+scatter``
   and compare to ``rx_ring_assemble_dense_block``. cuda-skipped.

Skipped automatically if the C extension or cuda is missing.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

# Skip whole module if the C lib (compact assembler) is unavailable.
try:
    from dsart.transport.recv_ring import (
        RxRing,
        RxRingDims,
        VF_DATA_PRESENT,
        VF_PATTERN_MISMATCH,
        VF_RX_OVERRUN,
        _get_lib,
    )
    _lib = _get_lib()
    _HAS_COMPACT = hasattr(_lib, "rx_ring_assemble_compact_block")
    _HAS_DENSE = hasattr(_lib, "rx_ring_assemble_dense_block")
except Exception:  # noqa: BLE001
    _HAS_COMPACT = False
    _HAS_DENSE = False

# CUDA detection: lazy. We only check inside GPU tests.
def _cuda_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import cupy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_NEEDS_C = pytest.mark.skipif(
    not (_HAS_COMPACT and _HAS_DENSE),
    reason=(
        "rx_ring_assemble_compact_block / _dense_block missing — "
        "rebuild C extension"
    ),
)
_NEEDS_GPU = pytest.mark.skipif(
    not _cuda_available(),
    reason="cuda + cupy required for GPU scatter parity",
)


def _unique_shm() -> str:
    return f"/m741_assemble_compact_{uuid.uuid4().hex[:12]}"


def _dims(
    n_corr: int = 3,
    n_coarse_dm: int = 2,
    t_buf: int = 32,
    n_filled: int = 7,
) -> RxRingDims:
    return RxRingDims(
        n_corr=n_corr,
        n_coarse_dm=n_coarse_dm,
        t_buf_samples=t_buf,
        n_filled_per_corr=n_filled,
        bytes_per_cell=2,  # cint8 complex
    )


def _open_ring(name: str, dims: RxRingDims) -> RxRing:
    try:
        RxRing.unlink_name(name)
    except Exception:
        pass
    return RxRing.open_or_create(name, dims)


def _make_lut(n_corr: int, n_filled: int, n_grid: int) -> np.ndarray:
    """Per-corr deterministic LUT: cell k → (ix=k%n_grid, iy=(k+corr)%n_grid).

    Matches the helper in test_recv_ring_assemble_dense.py so the two
    test suites operate on comparable geometry.
    """
    lut = np.zeros((n_corr, n_filled), dtype=np.int32)
    for c in range(n_corr):
        for k in range(n_filled):
            ix = k % n_grid
            iy = (k + c) % n_grid
            lut[c, k] = ix * n_grid + iy
    return lut


def _python_scatter_compact_to_dense(
    *,
    cells_packed: np.ndarray,        # int8 [N_corr, T_det, n_filled_max*2]
    lut: np.ndarray,                 # int32 [N_corr, lut_stride]
    n_filled_per_corr: np.ndarray,   # int32 [N_corr]
    n_corr: int,
    t_det: int,
    t_stream: int,
    n_grid: int,
    n_filled_max: int,
) -> np.ndarray:
    """Pure-Python reference scatter — mirrors the GPU kernel byte-by-byte.

    Returns a dense ``int8 [N_corr, T_stream, 2, N_grid, N_grid]`` plane
    with rows [0, t_det) populated from the compact buffer + LUT and
    rows [t_det, T_stream) at zero (M7.4 dense convention).
    """
    n_grid_sq = n_grid * n_grid
    out = np.zeros((n_corr, t_stream, 2, n_grid, n_grid), dtype=np.int8)
    for c in range(n_corr):
        nf = int(n_filled_per_corr[c])
        if nf <= 0:
            continue
        for t in range(t_det):
            for k in range(nf):
                lin = int(lut[c, k])
                if not (0 <= lin < n_grid_sq):
                    continue
                ix = lin // n_grid
                iy = lin % n_grid
                out[c, t, 0, ix, iy] = cells_packed[c, t, 2 * k]
                out[c, t, 1, ix, iy] = cells_packed[c, t, 2 * k + 1]
    return out


@_NEEDS_C
class TestCompactCpuParity:
    def test_compact_matches_dense_full_present(self) -> None:
        """All slots VF_DATA_PRESENT: compact + python scatter == dense C."""
        name = _unique_shm()
        dims = _dims()
        ring = _open_ring(name, dims)
        try:
            n_corr = dims.n_corr
            n_dm = dims.n_coarse_dm
            n_filled = dims.n_filled_per_corr
            t_det = 5
            n_grid = 8
            owned_dm = 1

            rng = np.random.default_rng(seed=42)
            for c in range(n_corr):
                for dm in range(n_dm):
                    for t in range(t_det):
                        raw = rng.integers(
                            -100, 100, size=(n_filled * 2,), dtype=np.int8,
                        ).tobytes()
                        sc = 0.05 + 0.001 * (c * 100 + dm * 10 + t)
                        ring.write_slot(
                            corr=c, dm=dm, t_seq=t, payload=raw,
                            validity_flags=VF_DATA_PRESENT,
                            scale=sc, offset=0.125,
                        )

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.full((n_corr,), n_filled, dtype=np.int32)

            (dense_c, scale_c, offre_c, offim_c, valid_c, *_) = (
                ring.assemble_dense_block(
                    specnum_start=0, t_det=t_det, n_grid=n_grid,
                    owned_dm=owned_dm,
                    n_filled_per_corr=nfp,
                    linear_lut_strided=lut,
                    compute_half=0,
                )
            )
            (cells, scale_k, offre_k, offim_k, valid_k, *_) = (
                ring.assemble_compact_block(
                    specnum_start=0, t_det=t_det,
                    owned_dm=owned_dm,
                    n_filled_per_corr=nfp,
                    n_filled_max=n_filled,
                    sidecar_t_stride=t_det,
                    compute_half=0,
                )
            )

            np.testing.assert_array_equal(scale_c, scale_k)
            np.testing.assert_array_equal(offre_c, offre_k)
            np.testing.assert_array_equal(offim_c, offim_k)
            np.testing.assert_array_equal(valid_c, valid_k)

            # Python reference scatter from the compact buffer.
            dense_py = _python_scatter_compact_to_dense(
                cells_packed=cells, lut=lut, n_filled_per_corr=nfp,
                n_corr=n_corr, t_det=t_det, t_stream=t_det,
                n_grid=n_grid, n_filled_max=n_filled,
            )
            np.testing.assert_array_equal(
                dense_c, dense_py,
                err_msg="compact→python_scatter must match C dense scatter",
            )
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_compact_matches_dense_with_bad_slots(self) -> None:
        """Bad slots (pattern_mismatch / no data / rx_overrun) leave their
        compact rows zero, matching the dense path's zeroed (corr, t)."""
        name = _unique_shm()
        dims = _dims()
        ring = _open_ring(name, dims)
        try:
            n_corr = dims.n_corr
            n_dm = dims.n_coarse_dm
            n_filled = dims.n_filled_per_corr
            t_det = 6
            n_grid = 8
            owned_dm = 0

            payload = (np.ones(n_filled * 2, dtype=np.int8) * 7).tobytes()
            for c in range(n_corr):
                for dm in range(n_dm):
                    for t in range(t_det):
                        ring.write_slot(
                            corr=c, dm=dm, t_seq=t, payload=payload,
                            validity_flags=VF_DATA_PRESENT,
                            scale=0.5, offset=0.0,
                        )
            # Bad slots (mirrors the dense test).
            ring.write_slot(
                corr=0, dm=owned_dm, t_seq=1, payload=payload,
                validity_flags=VF_PATTERN_MISMATCH | VF_DATA_PRESENT,
                scale=0.5, offset=0.0,
            )
            ring.write_slot(
                corr=2, dm=owned_dm, t_seq=3, payload=payload,
                validity_flags=0, scale=0.5, offset=0.0,
            )
            ring.write_slot(
                corr=1, dm=owned_dm, t_seq=4, payload=payload,
                validity_flags=VF_DATA_PRESENT | VF_RX_OVERRUN,
                scale=0.5, offset=0.0,
            )

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.full((n_corr,), n_filled, dtype=np.int32)

            (dense_c, scale_c, *_rest, valid_c, _no, _np, _nd) = (
                ring.assemble_dense_block(
                    specnum_start=0, t_det=t_det, n_grid=n_grid,
                    owned_dm=owned_dm,
                    n_filled_per_corr=nfp,
                    linear_lut_strided=lut,
                    compute_half=0,
                )
            )
            (cells, scale_k, _re_k, _im_k, valid_k, n_o, n_p, n_d) = (
                ring.assemble_compact_block(
                    specnum_start=0, t_det=t_det,
                    owned_dm=owned_dm,
                    n_filled_per_corr=nfp,
                    n_filled_max=n_filled,
                    sidecar_t_stride=t_det,
                    compute_half=0,
                )
            )

            assert (n_o, n_p, n_d) == (1, 1, 1)
            np.testing.assert_array_equal(valid_c, valid_k)
            np.testing.assert_array_equal(scale_c, scale_k)

            # The bad (corr, t) compact rows must be all-zero.
            assert (cells[0, 1] == 0).all()
            assert (cells[2, 3] == 0).all()
            assert (cells[1, 4] == 0).all()

            dense_py = _python_scatter_compact_to_dense(
                cells_packed=cells, lut=lut, n_filled_per_corr=nfp,
                n_corr=n_corr, t_det=t_det, t_stream=t_det,
                n_grid=n_grid, n_filled_max=n_filled,
            )
            np.testing.assert_array_equal(dense_c, dense_py)
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_silent_corr_zero(self) -> None:
        """n_filled_per_corr[c] = -1 → compact rows for that corr stay zero."""
        name = _unique_shm()
        dims = _dims()
        ring = _open_ring(name, dims)
        try:
            n_corr = dims.n_corr
            n_dm = dims.n_coarse_dm
            n_filled = dims.n_filled_per_corr
            t_det = 3
            n_grid = 8

            payload = (np.full(n_filled * 2, 5, dtype=np.int8)).tobytes()
            for c in range(n_corr):
                for dm in range(n_dm):
                    for t in range(t_det):
                        ring.write_slot(
                            corr=c, dm=dm, t_seq=t, payload=payload,
                            validity_flags=VF_DATA_PRESENT,
                            scale=1.0, offset=0.0,
                        )

            nfp = np.full((n_corr,), n_filled, dtype=np.int32)
            nfp[1] = -1  # corr=1 silent

            (cells, _sc, _ore, _oim, _valid, *_) = (
                ring.assemble_compact_block(
                    specnum_start=0, t_det=t_det,
                    owned_dm=0,
                    n_filled_per_corr=nfp,
                    n_filled_max=n_filled,
                    sidecar_t_stride=t_det,
                    compute_half=0,
                )
            )
            assert (cells[1] == 0).all()
            assert (cells[0] != 0).any()
            assert (cells[2] != 0).any()
        finally:
            ring.close()
            RxRing.unlink_name(name)


@_NEEDS_C
@_NEEDS_GPU
class TestGpuScatterParity:
    def test_gpu_scatter_matches_python_ref(self) -> None:
        """End-to-end compact+GPU scatter == dense C scatter."""
        import torch
        from dsart.transport.gpu_scatter import (
            scatter_compact_to_dense,
            zero_dense_rows,
        )

        name = _unique_shm()
        dims = _dims(n_corr=4, n_coarse_dm=2, t_buf=64, n_filled=11)
        ring = _open_ring(name, dims)
        try:
            n_corr = dims.n_corr
            n_dm = dims.n_coarse_dm
            n_filled = dims.n_filled_per_corr
            t_det = 7
            t_stream = 10  # lookahead rows; GPU scatter must NOT touch them
            n_grid = 16
            owned_dm = 1

            rng = np.random.default_rng(seed=7)
            for c in range(n_corr):
                for dm in range(n_dm):
                    for t in range(t_det):
                        raw = rng.integers(
                            -120, 120, size=(n_filled * 2,), dtype=np.int8,
                        ).tobytes()
                        ring.write_slot(
                            corr=c, dm=dm, t_seq=t, payload=raw,
                            validity_flags=VF_DATA_PRESENT,
                            scale=0.5, offset=0.0,
                        )

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.full((n_corr,), n_filled, dtype=np.int32)

            (dense_c, *_rest_dense) = ring.assemble_dense_block(
                specnum_start=0, t_det=t_det, n_grid=n_grid,
                owned_dm=owned_dm,
                n_filled_per_corr=nfp,
                linear_lut_strided=lut,
                out_t_stride=t_stream,
                compute_half=0,
            )
            (cells_h, *_rest_compact) = ring.assemble_compact_block(
                specnum_start=0, t_det=t_det,
                owned_dm=owned_dm,
                n_filled_per_corr=nfp,
                n_filled_max=n_filled,
                sidecar_t_stride=t_stream,
                compute_half=0,
            )

            # Put some non-zero garbage in the lookahead rows of the dense
            # GPU plane so we can assert the GPU scatter leaves them alone.
            dense_gpu = torch.full(
                (n_corr, t_stream, 2, n_grid, n_grid),
                123, dtype=torch.int8, device="cuda",
            )
            cells_gpu = torch.from_numpy(np.ascontiguousarray(cells_h)).to(
                "cuda", non_blocking=False,
            )
            lut_gpu = torch.from_numpy(np.ascontiguousarray(lut)).to(
                "cuda", non_blocking=False,
            )
            nfp_gpu = torch.from_numpy(np.ascontiguousarray(nfp)).to(
                "cuda", non_blocking=False,
            )

            zero_dense_rows(dense=dense_gpu, t_det=t_det)
            scatter_compact_to_dense(
                cells_packed=cells_gpu,
                lut=lut_gpu,
                n_filled_per_corr=nfp_gpu,
                dense=dense_gpu,
                t_det=t_det,
                n_grid=n_grid,
                n_filled_max=n_filled,
            )
            torch.cuda.synchronize()

            dense_gpu_h = dense_gpu.cpu().numpy()
            # Rows [0, t_det) must match dense C scatter byte-for-byte.
            np.testing.assert_array_equal(
                dense_gpu_h[:, :t_det], dense_c[:, :t_det],
                err_msg="GPU scatter [0,t_det) mismatch vs C dense scatter",
            )
            # Rows [t_det, t_stream) must still be the original garbage
            # (GPU scatter ONLY writes [0, t_det)).
            assert (dense_gpu_h[:, t_det:] == 123).all(), (
                "GPU scatter touched lookahead rows it should not have"
            )
        finally:
            ring.close()
            RxRing.unlink_name(name)
