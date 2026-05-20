"""Parity tests for the M7.2.9 batched validity-walk helper.

``rx_ring_assemble_validity_block`` (recv_ring.c) replaces the
``ProductionRxRingSource._assemble_cube`` Python loop with a single C
call. These tests verify the batched helper produces exactly the same
per-t validity verdict, the same per-counter deltas, and the same
overrun-counter bumps as the per-slot Python path that it replaces.

Run on any host that has built the C extension:

    python setup.py build_ext --inplace
    pytest tests/transport/test_recv_ring_assemble_validity.py -q
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

try:
    from dsart.transport.recv_ring import (
        BYTES_CINT8_COMPLEX,
        VF_DATA_PRESENT,
        VF_PATTERN_MISMATCH,
        VF_RX_OVERRUN,
        RxRing,
        RxRingDims,
        _get_lib,
    )
    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False


def _lib_has_assembler() -> bool:
    try:
        return _IMPORT_OK and hasattr(
            _get_lib(), "rx_ring_assemble_validity_block"
        )
    except (RuntimeError, OSError, ImportError):
        return False


_NEEDS = pytest.mark.skipif(
    not _lib_has_assembler(),
    reason="recv_ring.so missing or stale "
           "(no rx_ring_assemble_validity_block); rebuild with "
           "`python setup.py build_ext --inplace`",
)


def _unique_shm() -> str:
    return f"/dsart_test_{uuid.uuid4().hex[:12]}"


def _dims(*, n_corr=4, n_coarse_dm=3, t_buf=64, n_filled=32) -> RxRingDims:
    return RxRingDims(
        n_corr=n_corr,
        n_coarse_dm=n_coarse_dm,
        t_buf_samples=t_buf,
        n_filled_per_corr=n_filled,
        bytes_per_cell=BYTES_CINT8_COMPLEX,
    )


def _open_ring(name: str, dims: RxRingDims) -> RxRing:
    try:
        RxRing.unlink_name(name)
    except Exception:
        pass
    return RxRing.open_or_create(name, dims)


def _python_assemble_validity(
    ring: RxRing,
    *,
    specnum_start: int,
    t_det: int,
    compute_half: int,
    coarse_dm_mask: int,
) -> tuple[np.ndarray, int, int, int]:
    """Reference: per-slot Python loop matching the legacy fallback in
    ``ProductionRxRingSource._assemble_validity_python_fallback`` under
    the M7.2 search-overlap geometry (walk t_det rows; cube_cadence
    is the stride between emits, NOT the walk length)."""
    dims = ring.dims
    valid = np.ones(t_det, dtype=np.bool_)
    n_over = n_pat = n_nodp = 0
    for corr in range(dims.n_corr):
        for dm in range(dims.n_coarse_dm):
            if not ((coarse_dm_mask >> dm) & 1):
                continue
            for t in range(t_det):
                t_abs = specnum_start + t
                try:
                    _, vf = ring.read_slot(
                        corr=corr, dm=dm, t_seq=t_abs,
                        compute_half=compute_half,
                    )
                except OSError:
                    n_over += 1
                    valid[t] = False
                    continue
                if vf & VF_RX_OVERRUN:
                    n_over += 1
                    bad = True
                elif vf & VF_PATTERN_MISMATCH:
                    n_pat += 1
                    bad = True
                elif not (vf & VF_DATA_PRESENT):
                    n_nodp += 1
                    bad = True
                else:
                    bad = False
                if bad:
                    valid[t] = False
    return valid, n_over, n_pat, n_nodp


@_NEEDS
class TestParity:
    def test_all_present_no_bad_flags(self) -> None:
        """Fully-populated cube: every t-row is valid; counters all 0."""
        name = _unique_shm()
        dims = _dims()
        ring = _open_ring(name, dims)
        try:
            for corr in range(dims.n_corr):
                for dm in range(dims.n_coarse_dm):
                    for t in range(16):
                        ring.write_slot(
                            corr=corr, dm=dm, t_seq=t,
                            payload=None,
                            validity_flags=VF_DATA_PRESENT,
                        )
            v, dn_o, dn_p, dn_n = ring.assemble_validity_block(
                specnum_start=0,
                cube_cadence_samples=8,
                t_det=12,
                compute_half=0,
            )
            assert v.dtype == np.bool_
            assert v.shape == (12,)  # output sized to t_det, not cube_cadence
            assert v.all()
            assert (dn_o, dn_p, dn_n) == (0, 0, 0)
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_per_t_row_invalidation_matches_python(self) -> None:
        """Inject a mix of bad flags; batched output must match the
        per-slot Python loop in BOTH per-t validity and counter deltas.
        """
        name = _unique_shm()
        dims = _dims(n_corr=3, n_coarse_dm=2, t_buf=32, n_filled=16)
        ring = _open_ring(name, dims)
        try:
            # Build a synthetic ring state. Search-overlap geometry:
            # cube_cadence is the stride between emits; t_det is the
            # walked window per cube. M7.2 prod op-point is
            # cube_cadence=128, t_det=192.
            cube_cadence, t_det = 4, 8
            # Default: everyone DATA_PRESENT, then poke holes:
            for corr in range(dims.n_corr):
                for dm in range(dims.n_coarse_dm):
                    for t in range(t_det):
                        ring.write_slot(
                            corr=corr, dm=dm, t_seq=t,
                            payload=None,
                            validity_flags=VF_DATA_PRESENT,
                        )
            # Inject one PATTERN_MISMATCH at (corr=0,dm=1,t=2) -> row 2 bad.
            ring.write_slot(
                corr=0, dm=1, t_seq=2, payload=None,
                validity_flags=VF_DATA_PRESENT | VF_PATTERN_MISMATCH,
            )
            # Inject one VF=0 (no DATA_PRESENT) at (corr=2,dm=0,t=4)
            # -> row 4 bad, n_nodp += 1.
            ring.write_slot(
                corr=2, dm=0, t_seq=4, payload=None,
                validity_flags=0,
            )
            # Inject one RX_OVERRUN at (corr=1,dm=0,t=7) — within the
            # walked window now, so the helper must invalidate row 7
            # AND bump n_overrun.
            ring.write_slot(
                corr=1, dm=0, t_seq=7, payload=None,
                validity_flags=VF_DATA_PRESENT | VF_RX_OVERRUN,
            )

            # Reference (Python loop) over a fresh attach so the
            # overrun-counter bump from the batched call doesn't bleed
            # into the per-slot read_slot pass.
            ring_ref = RxRing.mmap_attach_readonly(name, dims)
            try:
                v_ref, n_o_ref, n_p_ref, n_n_ref = _python_assemble_validity(
                    ring_ref,
                    specnum_start=0,
                    t_det=t_det,
                    compute_half=0,
                    coarse_dm_mask=0xFFFFFFFF,
                )
            finally:
                ring_ref.close()

            # Batched C path on the original handle.
            v_c, n_o_c, n_p_c, n_n_c = ring.assemble_validity_block(
                specnum_start=0,
                cube_cadence_samples=cube_cadence,
                t_det=t_det,
                compute_half=0,
            )

            assert v_c.dtype == np.bool_
            assert v_c.shape == (t_det,)
            np.testing.assert_array_equal(v_c, v_ref)
            assert (n_o_c, n_p_c, n_n_c) == (n_o_ref, n_p_ref, n_n_ref)

            # Sanity-check the actual values we injected:
            # row 2 bad (pattern_mismatch), row 4 bad (no data present),
            # row 7 bad (RX_OVERRUN inside the t_det window).
            expected_rows = [True, True, False, True, False, True, True, False]
            assert list(v_c.tolist()) == expected_rows
            assert n_p_c == 1
            assert n_n_c == 1
            assert n_o_c == 1
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_coarse_dm_mask_skips_disabled_lanes(self) -> None:
        """A pattern_mismatch in an OFF coarse-DM must not invalidate the
        per-t mask (the mask gates which lanes the helper visits).

        t_buf must comfortably exceed (n_coarse_dm * cube_cadence) so
        the per-corr write_seq doesn't lap our query window — every
        write_slot for any (dm, t) bumps the SAME per-corr counter,
        and the helper's wseq check is "wseq > t_abs + t_buf" which
        would otherwise spuriously flag t=0 as overrun.
        """
        name = _unique_shm()
        dims = _dims(n_corr=2, n_coarse_dm=4, t_buf=64, n_filled=8)
        ring = _open_ring(name, dims)
        try:
            for corr in range(dims.n_corr):
                for dm in range(dims.n_coarse_dm):
                    for t in range(4):
                        ring.write_slot(
                            corr=corr, dm=dm, t_seq=t,
                            payload=None,
                            validity_flags=VF_DATA_PRESENT,
                        )
            # Poison dm=3 (which we'll mask off).
            ring.write_slot(
                corr=0, dm=3, t_seq=1, payload=None,
                validity_flags=VF_DATA_PRESENT | VF_PATTERN_MISMATCH,
            )
            # Mask = 0b0011 -> visit dm=0 and dm=1 only; dm=3 ignored.
            v, dn_o, dn_p, dn_n = ring.assemble_validity_block(
                specnum_start=0,
                cube_cadence_samples=4,
                t_det=4,
                compute_half=0,
                coarse_dm_mask=0b0011,
            )
            assert v.all(), (
                "mask=0b0011 must not touch dm=3 — t=1 should stay valid"
            )
            assert (dn_o, dn_p, dn_n) == (0, 0, 0)
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_overrun_counter_bumped_on_lapped_slot(self) -> None:
        """If wseq advanced past t_abs + t_buf, the slot is treated as
        an overrun: the helper bumps n_overrun AND the per-half
        overrun_count_per_compute mirror (matching
        rx_ring_read_slot's behaviour)."""
        name = _unique_shm()
        dims = _dims(n_corr=1, n_coarse_dm=1, t_buf=8, n_filled=4)
        ring = _open_ring(name, dims)
        try:
            # Advance write_seq past the lookback window by writing
            # t_buf+5 slots (wseq = 13, t_buf = 8 → slot t=0 lapped).
            for t in range(13):
                ring.write_slot(
                    corr=0, dm=0, t_seq=t, payload=None,
                    validity_flags=VF_DATA_PRESENT,
                )

            overrun_before = ring.get_overrun_count(compute_half=0)
            v, dn_o, dn_p, dn_n = ring.assemble_validity_block(
                specnum_start=0,
                cube_cadence_samples=4,
                t_det=4,  # walk t=0..3 (all lapped)
                compute_half=0,
            )
            overrun_after = ring.get_overrun_count(compute_half=0)

            assert (v == False).all()
            assert dn_o == 4
            assert overrun_after - overrun_before == 4
            assert dn_p == 0
            assert dn_n == 0
        finally:
            ring.close()
            RxRing.unlink_name(name)
