"""Frame v2 codec (SPEC §5): layout, sizes, packing, validation."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from dsart.transport import prod_frame
from dsart.transport import prod_frame2 as f2


def _hdr(**kw) -> f2.Frame2Header:
    base = dict(
        seq_first=1000, t0_native=16000, stream_id=5, dm_idx=33, k=3,
        n_sub=2, bits_main=8, bits_heavy=16, n_grid=256, t_int_factor=16,
        value_kind=0, n_values=10, pattern_id=0x0123456789ABCDEF, n_heavy=2,
        step_main=0.25, step_heavy=0.0625, payload_bytes=3 * (4 + 8),
        s_main=float(np.float32(0.3534)), s_heavy=float(np.float32(0.1)),
        flags=f2.FLAG_QUANTIZED | f2.FLAG_HEAVY_PRESENT,
    )
    base.update(kw)
    return f2.Frame2Header(**base)


def test_header_is_72_bytes_and_round_trips():
    h = _hdr()
    b = f2.pack_header(h)
    assert len(b) == f2.HEADER_BYTES == 72
    assert f2.unpack_header(b) == h


def test_exact_byte_offsets():
    h = _hdr(seq_first=0x0010203040506070, t0_native=0x0010203040506070 * 16,
             flags=f2.FLAG_QUANTIZED | f2.FLAG_HEAVY_PRESENT | f2.FLAG_MIDRISE
             | f2.FLAG_LAST_IN_BLOCK)
    b = f2.pack_header(h)
    o = f2.OFFSETS
    u = lambda fmt, off: struct.unpack_from("<" + fmt, b, off)[0]  # noqa: E731
    assert u("I", o["magic"]) == 0xD5A1107E
    assert u("H", o["version"]) == 2
    assert u("H", o["flags"]) == 0b1100011
    assert u("Q", o["seq_first"]) == h.seq_first
    assert u("Q", o["t0_native"]) == h.t0_native
    assert u("H", o["stream_id"]) == 5
    assert u("H", o["dm_idx"]) == 33
    assert u("B", o["k"]) == 3
    assert u("B", o["n_sub"]) == 2
    assert u("B", o["bits_main"]) == 8
    assert u("B", o["bits_heavy"]) == 16
    assert u("H", o["n_grid"]) == 256
    assert u("B", o["t_int_factor"]) == 16
    assert u("B", o["value_kind"]) == 0
    assert u("I", o["n_values"]) == 10
    assert u("Q", o["pattern_id"]) == 0x0123456789ABCDEF
    assert u("I", o["n_heavy"]) == 2
    assert u("f", o["step_main"]) == 0.25
    assert u("f", o["step_heavy"]) == 0.0625
    assert u("I", o["payload_bytes"]) == 36
    assert u("f", o["s_main"]) == np.float32(0.3534)
    assert u("f", o["s_heavy"]) == np.float32(0.1)
    # dm_idx and version sit at the v1 offsets (receivers dispatch on them)
    assert o["dm_idx"] == 26 and o["version"] == 4


def test_peek_version_v1_and_v2():
    v1 = prod_frame.pack_header(prod_frame.ProdFrameHeader(
        seq=1, specnum=2, chgroup=0, dm_idx=0, frag_idx=0, n_frags=1,
        n_grid=256, n_filled=1, pattern_id=1, bits_per_cell=16,
        t_int_factor=1, scale=1.0, offset=0.0, payload_bytes_in_frag=2,
        flags=prod_frame.FLAG_QUANTIZED))
    assert f2.peek_version(v1) == 1
    assert f2.peek_version(f2.pack_header(_hdr())) == 2
    assert f2.peek_version(b"\x00") == -1
    with pytest.raises(f2.Frame2Error):
        f2.unpack_header(v1)


@pytest.mark.parametrize("nv,nh,bm,bps,k", [
    (2759, 78, 8, 156 + 2681, 3),            # gridded n_sub 2, cint4
    (2267, 25, 10, 50 + 2803, 3),            # gridded n_sub 4, cint5
    (1773, 23, 8, 46 + 1750, 4),             # baselines n_sub 4, cint4
    (1773, 71, 12, 142 + 2553, 3),           # baselines n_sub 2, cint6
    (5, 0, 10, 7, 255),                      # tiny: capped at 255
])
def test_bytes_per_sample_and_k(nv, nh, bm, bps, k):
    bh = 16 if nh else 0
    assert f2.bytes_per_sample(nv, nh, bm, bh) == bps
    assert f2.max_k(bps) == k
    assert f2.HEADER_BYTES + k * bps <= 8972      # 9000 B on the wire


def test_max_k_rejects_oversize_sample():
    with pytest.raises(f2.Frame2Error):
        f2.max_k(8901)


@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("n_main", [1, 2, 3, 7, 101])
def test_main_codes_round_trip_full_range(bits, n_main):
    c = bits // 2
    rng = np.random.default_rng(bits * 1000 + n_main)
    codes = rng.integers(-(1 << (c - 1)), 1 << (c - 1), size=(4, n_main, 2))
    codes[0, 0] = (-(1 << (c - 1)), (1 << (c - 1)) - 1)       # extremes
    packed = f2.pack_main_codes(codes, bits)
    assert packed.shape == (4, (n_main * bits + 7) // 8)
    back = f2.unpack_main_codes(packed, n_main, bits)
    assert back.dtype == np.int8
    np.testing.assert_array_equal(back, codes)


def test_main_packing_exact_bits():
    # cint4: re low nibble, im high nibble
    assert f2.pack_main_codes(np.array([[1, -1]]), 8).tobytes() == bytes([0xF1])
    # cint5: v0 = (1, 2) -> 0b00010_00001, v1 = (-1, -16) -> 0b10000_11111,
    # stream LSB-first: 0x41 | 543 << 10 = 0x087C41
    assert f2.pack_main_codes(np.array([[1, 2], [-1, -16]]), 10).tobytes() == bytes([0x41, 0x7C, 0x08])
    # cint6: (-32, 31) -> 0x20 | 0x1F << 6 = 0x7E0 -> E0 07 (4 pad bits zero)
    assert f2.pack_main_codes(np.array([[-32, 31]]), 12).tobytes() == bytes([0xE0, 0x07])


def test_main_codes_out_of_range_rejected():
    with pytest.raises(f2.Frame2Error):
        f2.pack_main_codes(np.array([[8, 0]]), 8)
    with pytest.raises(f2.Frame2Error):
        f2.pack_main_codes(np.array([[0, -17]]), 10)


def test_samples_round_trip_heavy_first():
    rng = np.random.default_rng(7)
    k, nh, nm, bits = 3, 4, 9, 10
    ch = rng.integers(-128, 128, size=(k, nh, 2)).astype(np.int8)
    cm = rng.integers(-16, 16, size=(k, nm, 2))
    p = f2.pack_samples(ch, cm, bits)
    bps = f2.bytes_per_sample(nh + nm, nh, bits, 16)
    assert p.shape == (k, bps)
    # heavy bytes are plain interleaved int8 (re, im)
    np.testing.assert_array_equal(p[:, : 2 * nh].view(np.int8).reshape(k, nh, 2), ch)
    h2, m2 = f2.unpack_samples(p.tobytes(), k, nh + nm, nh, bits, 16)
    np.testing.assert_array_equal(h2, ch)
    np.testing.assert_array_equal(m2, cm)


def test_frame_round_trip_and_length_checks():
    h = _hdr()
    payload = bytes(range(36))
    wire = f2.pack_frame(h, payload)
    h2, p2 = f2.unpack_frame(wire)
    assert h2 == h and p2 == payload
    with pytest.raises(f2.Frame2Error):
        f2.pack_frame(h, payload[:-1])
    with pytest.raises(f2.Frame2Error):
        f2.unpack_frame(wire + b"\x00")


@pytest.mark.parametrize("kw", [
    dict(flags=f2.FLAG_HEAVY_PRESENT),                                   # not quantized
    dict(flags=f2.FLAG_QUANTIZED | f2.FLAG_HEAVY_PRESENT | f2.FLAG_RESERVED_BIT2),
    dict(flags=f2.FLAG_QUANTIZED | f2.FLAG_HEAVY_PRESENT | (1 << 9)),    # undefined
    dict(flags=f2.FLAG_QUANTIZED),                                       # heavy flag missing
    dict(t0_native=16001),                                               # t0 != seq*T
    dict(k=0, payload_bytes=0),
    dict(stream_id=32),                                                  # >= 16*n_sub
    dict(bits_main=16, payload_bytes=3 * (4 + 16)),
    dict(bits_heavy=0),                                                  # heavy without bits
    dict(n_sub=3),
    dict(value_kind=2),
    dict(t_int_factor=3, t0_native=3000),
    dict(payload_bytes=35),                                              # != k*bps
    dict(seq_first=(1 << 56) - 2, t0_native=((1 << 56) - 2) * 16),
])
def test_header_validation(kw):
    with pytest.raises(f2.Frame2Error):
        f2.pack_header(_hdr(**kw))


def test_payload_cap():
    n_values = 8901
    with pytest.raises(f2.Frame2Error):
        f2.pack_header(_hdr(n_values=n_values, n_heavy=0, bits_heavy=0, k=1,
                            payload_bytes=n_values,
                            flags=f2.FLAG_QUANTIZED))


def test_bad_magic_rejected():
    b = bytearray(f2.pack_header(_hdr()))
    b[0] ^= 1
    with pytest.raises(f2.Frame2Error):
        f2.unpack_header(bytes(b))


def test_block_datagrams_split_and_flags():
    rng = np.random.default_rng(3)
    S, nh, nm = 10, 2, 50
    ch = rng.integers(-127, 128, size=(S, nh, 2)).astype(np.int8)
    cm = rng.integers(-7, 8, size=(S, nm, 2))
    dgs = list(f2.iter_block_datagrams(
        codes_heavy=ch, codes_main=cm, seq_first=512, t_int_factor=32,
        stream_id=7, dm_idx=40, n_sub=4, bits_main=8, n_grid=256,
        value_kind=1, pattern_id=99, step_main=0.5, step_heavy=0.1,
        s_main=0.3534, s_heavy=0.1, k=4, extra_flags=f2.FLAG_RFI_WARMING_UP))
    assert len(dgs) == 3
    got_h, got_m, seqs = [], [], []
    for i, d in enumerate(dgs):
        h, p = f2.unpack_frame(d)
        seqs.append(h.seq_first)
        assert h.t0_native == h.seq_first * 32
        assert h.k == (4 if i < 2 else 2)
        assert h.last_in_block == (i == 2)
        assert h.rfi_warming_up and not h.midrise
        assert h.flags & f2.FLAG_HEAVY_PRESENT
        a, b = f2.unpack_samples(p, h.k, h.n_values, h.n_heavy, 8, 16)
        got_h.append(a)
        got_m.append(b)
    assert seqs == [512, 516, 520]
    np.testing.assert_array_equal(np.concatenate(got_h), ch)
    np.testing.assert_array_equal(np.concatenate(got_m), cm)


def test_block_datagrams_default_k_is_max_and_fits_mtu():
    S, nh, nm = 256, 78, 2681
    ch = np.zeros((S, nh, 2), np.int8)
    cm = np.zeros((S, nm, 2), np.int8)
    dgs = list(f2.iter_block_datagrams(
        codes_heavy=ch, codes_main=cm, seq_first=0, t_int_factor=16,
        stream_id=0, dm_idx=0, n_sub=2, bits_main=8, n_grid=256, value_kind=0,
        pattern_id=1, step_main=1.0, step_heavy=1.0, s_main=0.3534, s_heavy=0.1))
    assert len(dgs) == -(-256 // 3)
    assert max(len(d) for d in dgs) == 72 + 3 * 2837 <= 8972
