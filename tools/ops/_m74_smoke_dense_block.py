#!/usr/bin/env python3
"""M7.4 dense-scatter smoke test (C extension end-to-end).

Bypasses the heavy ``dsart`` import chain (torch, cupy) and exercises the
new ``rx_ring_assemble_dense_block`` C helper directly via ctypes. Covers:

1. ``rx_ring_write_slot`` v2 signature (scale + offset persisted).
2. Slot layout — scale/offset/vf are at the correct byte offsets.
3. ``rx_ring_assemble_dense_block`` scatter:
   * cint8 payload bytes scatter to the LUT-targeted (ix, iy) cells.
   * per-(corr, t) scale + offset sidecars come back faithful to the
     wire values stored at write time.
   * validity_per_t mirrors the per-slot vf flag.
   * Bad slots leave dense cells = 0 AND scale/offset = 0 (the GPU
     dequant kernel reads ``scale * cint8 + offset`` → 0).
4. Backward-compatibility: ``rx_ring_assemble_validity_block`` still
   returns correct results with the v2 slot layout (vf at slot end).

Run:
    PYTHONPATH=src python tools/ops/_m74_smoke_dense_block.py

Exit code 0 on full PASS; non-zero on first mismatch.
"""
from __future__ import annotations

import ctypes
import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SO_PATH = ROOT / "src/dsart/transport/_recv_ring.cpython-38-x86_64-linux-gnu.so"
if not SO_PATH.exists():
    # Fallback: find any built recv_ring.so
    cands = sorted((ROOT / "src/dsart/transport").glob("_recv_ring*.so"))
    if not cands:
        print("ERROR: _recv_ring.so not built. Run `python setup.py "
              "build_ext --inplace` first.", file=sys.stderr)
        sys.exit(2)
    SO_PATH = cands[0]

lib = ctypes.CDLL(str(SO_PATH))

# ------------------------------------------------------------------------
# ctypes signatures (mirror src/dsart/transport/recv_ring.h)
# ------------------------------------------------------------------------
lib.rx_ring_open_or_create.restype = ctypes.c_void_p
lib.rx_ring_open_or_create.argtypes = [
    ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
]
lib.rx_ring_close.argtypes = [ctypes.c_void_p]
lib.rx_ring_unlink.argtypes = [ctypes.c_char_p]
lib.rx_ring_unlink.restype = ctypes.c_int

lib.rx_ring_write_slot.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_float, ctypes.c_float, ctypes.c_uint16,
]
lib.rx_ring_write_slot.restype = ctypes.c_int

lib.rx_ring_assemble_dense_block.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64,
    ctypes.c_uint32,  # t_det
    ctypes.c_uint32,  # out_t_stride
    ctypes.c_uint32,  # n_grid
    ctypes.c_uint32,  # owned_dm
    ctypes.c_uint32,  # compute_half
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
lib.rx_ring_assemble_dense_block.restype = ctypes.c_int

lib.rx_ring_assemble_validity_block.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
lib.rx_ring_assemble_validity_block.restype = ctypes.c_int

# Validity flag bits (mirror recv_ring.h).
VF_DATA_PRESENT = 1 << 0
VF_PATTERN_MISMATCH = 1 << 1
VF_RX_OVERRUN = 1 << 4


def _shm() -> bytes:
    return f"/m74_smoke_{uuid.uuid4().hex[:12]}".encode()


def _open(name: bytes, n_corr: int, n_dm: int, t_buf: int,
          n_filled: int) -> int:
    errbuf = ctypes.create_string_buffer(256)
    h = lib.rx_ring_open_or_create(
        name, n_corr, n_dm, t_buf, n_filled, 2,
        1, errbuf, 256,
    )
    if not h:
        raise RuntimeError(f"rx_ring_open_or_create failed: {errbuf.value!r}")
    return h


def _wr(handle: int, corr: int, dm: int, t: int,
        payload: bytes, scale: float, offset: float, vf: int) -> None:
    payload_buf = (ctypes.c_char * len(payload))(*payload) if payload else None
    payload_ptr = ctypes.cast(payload_buf, ctypes.c_void_p) if payload_buf else None
    rc = lib.rx_ring_write_slot(
        ctypes.c_void_p(handle),
        ctypes.c_uint32(corr), ctypes.c_uint32(dm),
        ctypes.c_uint64(t),
        payload_ptr, ctypes.c_size_t(len(payload)),
        ctypes.c_float(scale), ctypes.c_float(offset),
        ctypes.c_uint16(vf),
    )
    if rc != 0:
        raise RuntimeError(f"write_slot rc={rc}")


def test_scatter_roundtrip() -> None:
    """Write a known cube to the ring, scatter it back via the C
    helper, and verify the dense + sidecar outputs against the
    Python-side reference."""
    name = _shm()
    n_corr, n_dm, t_buf, n_filled = 3, 2, 64, 7
    t_det, n_grid = 5, 8
    owned_dm = 1

    handle = _open(name, n_corr, n_dm, t_buf, n_filled)
    try:
        # Build a deterministic LUT: corr c, cell k -> (ix=k%n_grid,
        # iy=(k+c) % n_grid). lin = ix*n_grid + iy.
        lut = np.zeros((n_corr, n_filled), dtype=np.int32)
        for c in range(n_corr):
            for k in range(n_filled):
                ix = k % n_grid
                iy = (k + c) % n_grid
                lut[c, k] = ix * n_grid + iy
        n_filled_per_corr = np.full((n_corr,), n_filled, dtype=np.int32)

        # Generate per-(corr, dm, t) payload + scale/offset. Payload
        # is [re_0, im_0, re_1, im_1, ...] int8 (n_filled cells × 2 B).
        rng = np.random.default_rng(seed=42)
        payload_table: dict[tuple[int, int, int], bytes] = {}
        scale_table: dict[tuple[int, int, int], float] = {}
        offset_table: dict[tuple[int, int, int], float] = {}

        # Mark slot (corr=0, dm=owned, t=2) as PATTERN_MISMATCH (bad).
        # Slot (corr=2, dm=owned, t=3) gets vf=0 (no data present).
        # Both must invalidate the row in validity_per_t AND scatter 0.
        bad_slots = {(0, owned_dm, 2): VF_PATTERN_MISMATCH,
                     (2, owned_dm, 3): 0}

        for c in range(n_corr):
            for dm in range(n_dm):
                for t in range(t_det):
                    raw = rng.integers(-100, 100, size=(n_filled * 2,),
                                       dtype=np.int8).tobytes()
                    scale = 0.05 + 0.001 * (c * 10 + dm * 5 + t)
                    offset = 0.0  # symmetric cint8 wire convention
                    vf = bad_slots.get((c, dm, t), VF_DATA_PRESENT)
                    _wr(handle, c, dm, t, raw, scale, offset, vf)
                    payload_table[(c, dm, t)] = raw
                    scale_table[(c, dm, t)] = scale
                    offset_table[(c, dm, t)] = offset

        # Run dense-block scatter.
        out_cint8 = np.zeros((n_corr, t_det, 2, n_grid, n_grid), dtype=np.int8)
        out_scale = np.zeros((n_corr, t_det), dtype=np.float32)
        out_offre = np.zeros((n_corr, t_det), dtype=np.float32)
        out_offim = np.zeros((n_corr, t_det), dtype=np.float32)
        out_valid = np.zeros((t_det,), dtype=np.uint8)
        n_over = ctypes.c_uint64(0)
        n_pat = ctypes.c_uint64(0)
        n_nodp = ctypes.c_uint64(0)
        rc = lib.rx_ring_assemble_dense_block(
            ctypes.c_void_p(handle), ctypes.c_uint64(0),
            ctypes.c_uint32(t_det), ctypes.c_uint32(t_det),
            ctypes.c_uint32(n_grid),
            ctypes.c_uint32(owned_dm), ctypes.c_uint32(0),
            n_filled_per_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(n_filled),
            out_cint8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offre.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(n_over), ctypes.byref(n_pat), ctypes.byref(n_nodp),
        )
        assert rc == 0, f"assemble_dense_block rc={rc}"

        # ----- 1. Validity per t -----
        bad_t_rows = {2, 3}
        for t in range(t_det):
            expect = 0 if t in bad_t_rows else 1
            assert out_valid[t] == expect, (
                f"valid_per_t[{t}]={out_valid[t]} expected {expect}"
            )
        assert n_pat.value == 1, f"n_pat={n_pat.value} expected 1"
        assert n_nodp.value == 1, f"n_nodp={n_nodp.value} expected 1"
        assert n_over.value == 0, f"n_over={n_over.value} expected 0"

        # ----- 2. Scatter of GOOD slots: dense matches LUT scatter -----
        for c in range(n_corr):
            for t in range(t_det):
                if (c, owned_dm, t) in bad_slots:
                    # Bad: dense plane must be all zero; scale/offset = 0.
                    assert np.all(out_cint8[c, t] == 0), (
                        f"bad slot ({c}, {t}) leaked non-zero cint8"
                    )
                    assert out_scale[c, t] == 0.0
                    assert out_offre[c, t] == 0.0
                    assert out_offim[c, t] == 0.0
                    continue

                # Good: dense plane has src cells at LUT positions.
                raw = np.frombuffer(payload_table[(c, owned_dm, t)],
                                    dtype=np.int8)
                expected_re = np.zeros((n_grid, n_grid), dtype=np.int8)
                expected_im = np.zeros((n_grid, n_grid), dtype=np.int8)
                for k in range(n_filled):
                    lin = int(lut[c, k])
                    ix, iy = lin // n_grid, lin % n_grid
                    expected_re[ix, iy] = raw[2 * k]
                    expected_im[ix, iy] = raw[2 * k + 1]
                assert np.array_equal(out_cint8[c, t, 0], expected_re), (
                    f"re plane mismatch at ({c}, {t})"
                )
                assert np.array_equal(out_cint8[c, t, 1], expected_im), (
                    f"im plane mismatch at ({c}, {t})"
                )
                # Sidecar scale matches.
                assert abs(out_scale[c, t]
                           - scale_table[(c, owned_dm, t)]) < 1e-6, (
                    f"scale[{c}, {t}]={out_scale[c, t]} != "
                    f"{scale_table[(c, owned_dm, t)]}"
                )

        print(f"PASS test_scatter_roundtrip: t_det={t_det} n_corr={n_corr} "
              f"n_filled={n_filled} validity={out_valid.tolist()} "
              f"counters=(over={n_over.value}, pat={n_pat.value}, "
              f"nodp={n_nodp.value})")
    finally:
        lib.rx_ring_close(ctypes.c_void_p(handle))
        lib.rx_ring_unlink(name)


def test_validity_block_still_works() -> None:
    """The vf byte moved past the 8-byte scale/offset sidecar — make
    sure ``rx_ring_assemble_validity_block`` still finds it."""
    name = _shm()
    n_corr, n_dm, t_buf, n_filled = 2, 2, 32, 4
    handle = _open(name, n_corr, n_dm, t_buf, n_filled)
    try:
        for c in range(n_corr):
            for dm in range(n_dm):
                for t in range(8):
                    _wr(handle, c, dm, t, bytes(n_filled * 2),
                        1.5, 0.0, VF_DATA_PRESENT)
        # Inject one bad.
        _wr(handle, 1, 1, 5, bytes(n_filled * 2),
            1.5, 0.0, VF_PATTERN_MISMATCH | VF_DATA_PRESENT)

        out = np.ones((8,), dtype=np.uint8)
        n_over = ctypes.c_uint64(0)
        n_pat = ctypes.c_uint64(0)
        n_nodp = ctypes.c_uint64(0)
        rc = lib.rx_ring_assemble_validity_block(
            ctypes.c_void_p(handle), ctypes.c_uint64(0),
            ctypes.c_uint32(8), ctypes.c_uint32(8),
            ctypes.c_uint32(0), ctypes.c_uint32(0x3),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(n_over), ctypes.byref(n_pat), ctypes.byref(n_nodp),
        )
        assert rc == 0
        assert out[5] == 0, f"row 5 must be invalid; got {out[5]}"
        for t in range(8):
            if t == 5:
                continue
            assert out[t] == 1, f"row {t} must be valid; got {out[t]}"
        assert n_pat.value == 1
        print(f"PASS test_validity_block_still_works: vf at v2 offset")
    finally:
        lib.rx_ring_close(ctypes.c_void_p(handle))
        lib.rx_ring_unlink(name)


def test_zero_scale_bad_slot() -> None:
    """When scale=0 and vf=VF_DATA_PRESENT, the scatter writes the
    payload AND a 0 scale (assembler trusts vf, not scale)."""
    name = _shm()
    n_corr, n_dm, t_buf, n_filled = 1, 1, 16, 2
    handle = _open(name, n_corr, n_dm, t_buf, n_filled)
    try:
        _wr(handle, 0, 0, 0, b"\x05\xff" * 2, 0.0, 0.0, VF_DATA_PRESENT)
        lut = np.array([[0, 1]], dtype=np.int32)
        nfp = np.array([2], dtype=np.int32)
        out_cint8 = np.zeros((1, 1, 2, 4, 4), dtype=np.int8)
        out_scale = np.zeros((1, 1), dtype=np.float32)
        out_offre = np.zeros((1, 1), dtype=np.float32)
        out_offim = np.zeros((1, 1), dtype=np.float32)
        out_valid = np.zeros((1,), dtype=np.uint8)
        n_over = ctypes.c_uint64(0)
        n_pat = ctypes.c_uint64(0)
        n_nodp = ctypes.c_uint64(0)
        rc = lib.rx_ring_assemble_dense_block(
            ctypes.c_void_p(handle), ctypes.c_uint64(0),
            ctypes.c_uint32(1), ctypes.c_uint32(1),
            ctypes.c_uint32(4),
            ctypes.c_uint32(0), ctypes.c_uint32(0),
            nfp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(2),
            out_cint8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offre.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(n_over), ctypes.byref(n_pat), ctypes.byref(n_nodp),
        )
        assert rc == 0
        # Valid slot was written; payload should scatter and scale=0
        # makes the GPU dequant contribute zero — which is correct: this
        # slot's wire payload had no usable dynamic range (e.g. tx
        # _compute_scale_offset early-returned scale=1 on the all-zero
        # case, but a 0-scale wire slot would correctly null-out).
        assert out_valid[0] == 1
        # cell k=0 -> linear 0 -> ix=0,iy=0; payload re=0x05, im=-1.
        assert out_cint8[0, 0, 0, 0, 0] == 0x05
        assert out_cint8[0, 0, 1, 0, 0] == -1  # 0xff as int8
        assert out_scale[0, 0] == 0.0
        print("PASS test_zero_scale_bad_slot: scale=0 + vf=present propagates")
    finally:
        lib.rx_ring_close(ctypes.c_void_p(handle))
        lib.rx_ring_unlink(name)


def main() -> int:
    print(f"smoke: using {SO_PATH.name}")
    test_scatter_roundtrip()
    test_validity_block_still_works()
    test_zero_scale_bad_slot()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
