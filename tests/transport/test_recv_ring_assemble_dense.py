"""Parity tests for ``rx_ring_assemble_dense_block`` (M7.4 scatter).

Mirrors the structure of ``test_recv_ring_assemble_validity.py`` (the
M7.2.9 validity-walk parity tests): writes a known cube to the ring,
calls ``RxRing.assemble_dense_block``, and verifies dense scatter +
sidecars against a Python reference.

Skipped automatically if the C extension is not built.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

# Skip whole module if the C lib is unavailable / stale.
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
    _HAS_DENSE = hasattr(_lib, "rx_ring_assemble_dense_block")
except Exception:  # noqa: BLE001
    _HAS_DENSE = False

_NEEDS = pytest.mark.skipif(
    not _HAS_DENSE,
    reason="rx_ring_assemble_dense_block missing — rebuild C extension",
)


def _unique_shm() -> str:
    return f"/m74_assemble_dense_{uuid.uuid4().hex[:12]}"


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
    """Per-corr deterministic LUT: cell k → (ix=k%n_grid, iy=(k+corr)%n_grid)."""
    lut = np.zeros((n_corr, n_filled), dtype=np.int32)
    for c in range(n_corr):
        for k in range(n_filled):
            ix = k % n_grid
            iy = (k + c) % n_grid
            lut[c, k] = ix * n_grid + iy
    return lut


@_NEEDS
class TestDenseScatterParity:
    def test_full_present_round_trip(self) -> None:
        """All slots VF_DATA_PRESENT — dense plane mirrors LUT scatter,
        sidecars carry per-slot scale/offset, validity all-True."""
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

            rng = np.random.default_rng(seed=0)
            payloads: dict[tuple[int, int, int], bytes] = {}
            scales: dict[tuple[int, int, int], float] = {}

            for c in range(n_corr):
                for dm in range(n_dm):
                    for t in range(t_det):
                        raw = rng.integers(-100, 100, size=(n_filled * 2,),
                                           dtype=np.int8).tobytes()
                        sc = 0.05 + 0.001 * (c * 100 + dm * 10 + t)
                        ring.write_slot(
                            corr=c, dm=dm, t_seq=t,
                            payload=raw,
                            validity_flags=VF_DATA_PRESENT,
                            scale=sc, offset=0.0,
                        )
                        payloads[(c, dm, t)] = raw
                        scales[(c, dm, t)] = sc

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.full((n_corr,), n_filled, dtype=np.int32)

            (cint8_out, scale_out, offre_out, offim_out,
             valid, n_over, n_pat, n_nodp) = ring.assemble_dense_block(
                specnum_start=0,
                t_det=t_det,
                n_grid=n_grid,
                owned_dm=owned_dm,
                n_filled_per_corr=nfp,
                linear_lut_strided=lut,
                compute_half=0,
            )

            assert (n_over, n_pat, n_nodp) == (0, 0, 0)
            assert valid.dtype == np.bool_
            assert valid.all()
            assert cint8_out.shape == (n_corr, t_det, 2, n_grid, n_grid)
            assert scale_out.shape == (n_corr, t_det)

            for c in range(n_corr):
                for t in range(t_det):
                    raw = np.frombuffer(payloads[(c, owned_dm, t)],
                                        dtype=np.int8)
                    re_expected = np.zeros((n_grid, n_grid), dtype=np.int8)
                    im_expected = np.zeros((n_grid, n_grid), dtype=np.int8)
                    for k in range(n_filled):
                        lin = int(lut[c, k])
                        ix, iy = lin // n_grid, lin % n_grid
                        re_expected[ix, iy] = raw[2 * k]
                        im_expected[ix, iy] = raw[2 * k + 1]
                    np.testing.assert_array_equal(
                        cint8_out[c, t, 0], re_expected,
                        err_msg=f"re plane (corr={c}, t={t})",
                    )
                    np.testing.assert_array_equal(
                        cint8_out[c, t, 1], im_expected,
                        err_msg=f"im plane (corr={c}, t={t})",
                    )
                    np.testing.assert_allclose(
                        scale_out[c, t], scales[(c, owned_dm, t)],
                        atol=1e-6,
                    )
                    assert offre_out[c, t] == 0.0
                    assert offim_out[c, t] == 0.0
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_bad_slot_zeros_and_validity_drops(self) -> None:
        """Bad slots (PATTERN_MISMATCH / no DATA_PRESENT / RX_OVERRUN
        flag) leave dense planes + scale/offset zero AND drop the
        corresponding validity row to False. Counter deltas match."""
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
                            corr=c, dm=dm, t_seq=t,
                            payload=payload,
                            validity_flags=VF_DATA_PRESENT,
                            scale=0.5, offset=0.0,
                        )

            # Inject 3 bad slots on the OWNED dm — these drop validity:
            #   t=1: PATTERN_MISMATCH on corr=0
            #   t=3: vf=0 (no DATA_PRESENT) on corr=2
            #   t=4: RX_OVERRUN flag on corr=1
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
            # Also inject a bad slot on the NON-owned dm. Must NOT
            # affect anything because the scatter only walks owned_dm.
            ring.write_slot(
                corr=0, dm=1 - owned_dm, t_seq=2, payload=payload,
                validity_flags=VF_PATTERN_MISMATCH, scale=0.5, offset=0.0,
            )

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.full((n_corr,), n_filled, dtype=np.int32)
            (cint8_out, scale_out, _offre, _offim,
             valid, n_over, n_pat, n_nodp) = ring.assemble_dense_block(
                specnum_start=0, t_det=t_det, n_grid=n_grid,
                owned_dm=owned_dm,
                n_filled_per_corr=nfp,
                linear_lut_strided=lut,
                compute_half=0,
            )

            bad_t = {1, 3, 4}
            for t in range(t_det):
                expect = t not in bad_t
                assert bool(valid[t]) is expect, (
                    f"valid[{t}]={valid[t]} expected {expect}"
                )
            assert n_pat == 1, f"n_pat={n_pat} expected 1 (t=1 corr=0)"
            assert n_nodp == 1, f"n_nodp={n_nodp} expected 1 (t=3 corr=2)"
            assert n_over == 1, f"n_over={n_over} expected 1 (t=4 corr=1)"

            # Bad row CINT8 planes for the EXACT (corr, t) of the
            # bad-flag slot are zeroed; other corrs' planes at the same
            # t row keep their (good) data even though the t-row is
            # invalid (semantics: bad ANY (corr) ⇒ t-row invalid;
            # scatter still zero-fills ONLY the bad (corr, t) cells
            # so the GPU dequant sees zeros there).
            assert cint8_out[0, 1].sum() == 0  # bad slot zeroed
            assert cint8_out[2, 3].sum() == 0
            assert cint8_out[1, 4].sum() == 0
            assert scale_out[0, 1] == 0.0
            assert scale_out[2, 3] == 0.0
            assert scale_out[1, 4] == 0.0
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_silent_corr_skip(self) -> None:
        """n_filled_per_corr[c] = -1 marks corr as intentionally silent;
        scatter leaves dense plane + sidecars zero for that corr."""
        name = _unique_shm()
        dims = _dims()
        ring = _open_ring(name, dims)
        try:
            n_corr = dims.n_corr
            n_filled = dims.n_filled_per_corr
            t_det = 3
            n_grid = 8
            owned_dm = 0
            payload = (np.ones(n_filled * 2, dtype=np.int8) * 13).tobytes()
            for c in range(n_corr):
                for t in range(t_det):
                    ring.write_slot(
                        corr=c, dm=owned_dm, t_seq=t,
                        payload=payload,
                        validity_flags=VF_DATA_PRESENT,
                        scale=1.0, offset=0.0,
                    )

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.array([n_filled, -1, n_filled], dtype=np.int32)
            (cint8_out, scale_out, _offre, _offim,
             _v, _no, _np_, _nn) = ring.assemble_dense_block(
                specnum_start=0, t_det=t_det, n_grid=n_grid,
                owned_dm=owned_dm,
                n_filled_per_corr=nfp,
                linear_lut_strided=lut,
                compute_half=0,
            )
            # Silent corr (c=1) has fully zero dense plane + scale.
            assert cint8_out[1].sum() == 0
            assert scale_out[1].sum() == 0
            # Active corrs (c=0, 2) have non-zero scatter at LUT cells.
            assert cint8_out[0].sum() != 0
            assert cint8_out[2].sum() != 0
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_reusable_output_buffers(self) -> None:
        """Caller-provided output buffers are re-zeroed by the C helper
        each call (so they can be allocated ONCE and reused per cube).
        """
        name = _unique_shm()
        dims = _dims()
        ring = _open_ring(name, dims)
        try:
            n_corr = dims.n_corr
            n_filled = dims.n_filled_per_corr
            t_det = 3
            n_grid = 8

            for c in range(n_corr):
                for t in range(t_det):
                    ring.write_slot(
                        corr=c, dm=0, t_seq=t,
                        payload=(np.full(n_filled * 2, c + 1,
                                         dtype=np.int8).tobytes()),
                        validity_flags=VF_DATA_PRESENT,
                        scale=2.0, offset=0.0,
                    )

            lut = _make_lut(n_corr, n_filled, n_grid)
            nfp = np.full((n_corr,), n_filled, dtype=np.int32)

            out_cint8 = np.zeros((n_corr, t_det, 2, n_grid, n_grid),
                                 dtype=np.int8)
            out_scale = np.zeros((n_corr, t_det), dtype=np.float32)
            out_offre = np.zeros((n_corr, t_det), dtype=np.float32)
            out_offim = np.zeros((n_corr, t_det), dtype=np.float32)
            out_valid = np.zeros((t_det,), dtype=np.uint8)

            # First call.
            ring.assemble_dense_block(
                specnum_start=0, t_det=t_det, n_grid=n_grid, owned_dm=0,
                n_filled_per_corr=nfp, linear_lut_strided=lut,
                compute_half=0,
                out_cint8=out_cint8, out_scale=out_scale,
                out_offset_re=out_offre, out_offset_im=out_offim,
                out_validity=out_valid,
            )
            first_cint8 = out_cint8.copy()
            first_scale = out_scale.copy()

            # Second call. The helper zeroes the buffers on entry; if it
            # didn't, this would accumulate. (We re-call with the same
            # ring state so the expected output is identical.)
            ring.assemble_dense_block(
                specnum_start=0, t_det=t_det, n_grid=n_grid, owned_dm=0,
                n_filled_per_corr=nfp, linear_lut_strided=lut,
                compute_half=0,
                out_cint8=out_cint8, out_scale=out_scale,
                out_offset_re=out_offre, out_offset_im=out_offim,
                out_validity=out_valid,
            )
            np.testing.assert_array_equal(out_cint8, first_cint8)
            np.testing.assert_array_equal(out_scale, first_scale)
        finally:
            ring.close()
            RxRing.unlink_name(name)
