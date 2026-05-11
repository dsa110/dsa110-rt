"""Acceptance tests for :mod:`dsart.transport.prod_frame` (M4a chunk 1).

Pins the 72-byte wire format, the round-trip pack/unpack identity, the
fragment splitter, and the v1 protocol validators against plan §4.3
lines 1409-1444. No socket I/O — pure codec; runs on any host.

See ``docs/m4a/chunk1_prod_frame_spec.md`` §8 for the test plan.
"""

from __future__ import annotations

import math
import struct
from dataclasses import replace
from typing import Final

import pytest

from dsart.transport.prod_frame import (
    BITS_CFP16_COMPLEX,
    BITS_CINT8_COMPLEX,
    DEFAULT_MAX_FRAG_PAYLOAD_BYTES,
    FLAG_LAST_IN_BLOCK,
    FLAG_NOISE_WARMUP,
    FLAG_QUANTIZED,
    FLAG_RESERVED_BIT2,
    FLAG_RFI_WARMING_UP,
    HEADER_BYTES,
    MAGIC,
    VALID_BITS_PER_CELL,
    VALID_T_INT_FACTORS,
    VERSION,
    ProdFrameFieldRangeError,
    ProdFrameHeader,
    ProdFrameLengthError,
    ProdFrameMagicError,
    ProdFrameVersionError,
    expected_payload_bytes,
    pack_frame,
    pack_header,
    predict_pattern_id,
    split_payload_into_fragments,
    unpack_frame,
    unpack_header,
)


# ---------------------------------------------------------------------------
# Test fixtures: a "realistic" header used by several tests below.
# ---------------------------------------------------------------------------


def _f32(x: float) -> float:
    """Pre-quantise to float32 so dataclass equality after pack→unpack
    is bit-exact (the wire format stores float32, so 1.5e-3 stored as
    Python float becomes 0.001500000013... after round-trip)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _realistic_header(
    **overrides: object,
) -> ProdFrameHeader:
    """Build a header populated with realistic default-ops values, with
    individual fields override-able for parametric tests."""
    fields: dict[str, object] = dict(
        seq=12_345,
        specnum=0xCAFEBABE_DEADBEEF,
        chgroup=3,
        dm_idx=11,
        frag_idx=0,
        n_frags=1,
        n_grid=256,
        n_filled=5800,
        pattern_id=0xDEADBEEFCAFEBABE,
        bits_per_cell=BITS_CINT8_COMPLEX,
        t_int_factor=8,
        scale=_f32(1.5e-3),    # realistic cint8 dequant scale, pre-quantised to float32
        offset=_f32(-0.5),
        payload_bytes_in_frag=5800 * BITS_CINT8_COMPLEX // 8,
        flags=FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK,
        version=VERSION,
    )
    fields.update(overrides)
    return ProdFrameHeader(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8.1 Header layout pins
# ---------------------------------------------------------------------------


def test_header_struct_size_is_72_bytes() -> None:
    """The packed header MUST be exactly 72 bytes per plan §4.3 line 1409."""
    assert HEADER_BYTES == 72
    h = _realistic_header()
    blob = pack_header(h)
    assert len(blob) == HEADER_BYTES == 72


def test_header_field_offsets() -> None:
    """Pack a header with distinct sentinel values for every field and
    verify each byte-offset reads back exactly what plan §4.3 lines
    1411-1442 specify."""
    h = ProdFrameHeader(
        seq=0x1122334455667788,
        specnum=0x99AABBCCDDEEFF00,
        chgroup=0x0042,
        dm_idx=0x0017,
        frag_idx=0x0001,
        n_frags=0x0003,
        n_grid=256,
        n_filled=5800,
        pattern_id=0xFEED_FACE_DEAD_BEEF,
        bits_per_cell=BITS_CFP16_COMPLEX,
        t_int_factor=16,
        scale=0.25,
        offset=-0.125,
        payload_bytes_in_frag=2904,
        flags=FLAG_QUANTIZED | FLAG_RFI_WARMING_UP,
        version=VERSION,
    )
    blob = pack_header(h)

    assert struct.unpack_from("<I", blob, 0)[0] == MAGIC
    assert struct.unpack_from("<H", blob, 4)[0] == VERSION
    assert struct.unpack_from("<H", blob, 6)[0] == (
        FLAG_QUANTIZED | FLAG_RFI_WARMING_UP
    )
    assert struct.unpack_from("<Q", blob, 8)[0] == 0x1122334455667788
    assert struct.unpack_from("<Q", blob, 16)[0] == 0x99AABBCCDDEEFF00
    assert struct.unpack_from("<H", blob, 24)[0] == 0x0042
    assert struct.unpack_from("<H", blob, 26)[0] == 0x0017
    assert struct.unpack_from("<H", blob, 28)[0] == 0x0001
    assert struct.unpack_from("<H", blob, 30)[0] == 0x0003
    assert struct.unpack_from("<H", blob, 32)[0] == 256
    # reserved0 at offset 34 MUST be 0
    assert struct.unpack_from("<H", blob, 34)[0] == 0
    assert struct.unpack_from("<I", blob, 36)[0] == 5800
    assert struct.unpack_from("<Q", blob, 40)[0] == 0xFEED_FACE_DEAD_BEEF
    assert struct.unpack_from("<B", blob, 48)[0] == BITS_CFP16_COMPLEX
    assert struct.unpack_from("<B", blob, 49)[0] == 16
    # reserved1 at offset 50 MUST be 0
    assert struct.unpack_from("<H", blob, 50)[0] == 0
    assert struct.unpack_from("<f", blob, 52)[0] == pytest.approx(0.25)
    assert struct.unpack_from("<f", blob, 56)[0] == pytest.approx(-0.125)
    assert struct.unpack_from("<I", blob, 60)[0] == 2904
    # reserved2[8] at offset 64-71 MUST be all zero
    assert blob[64:72] == b"\x00" * 8


def test_header_magic_constant() -> None:
    """``MAGIC`` MUST equal 0xD5A1107E ("DSA110 7E") per plan §4.3 line 1412."""
    assert MAGIC == 0xD5A1107E


# ---------------------------------------------------------------------------
# 8.2 Round-trip identity
# ---------------------------------------------------------------------------


def test_round_trip_minimal() -> None:
    """Minimal header + single-byte payload survives pack→unpack
    intact."""
    h = ProdFrameHeader(
        seq=0,
        specnum=0,
        chgroup=0,
        dm_idx=0,
        frag_idx=0,
        n_frags=1,
        n_grid=0,
        n_filled=0,
        pattern_id=0,
        bits_per_cell=BITS_CINT8_COMPLEX,
        t_int_factor=1,
        scale=0.0,
        offset=0.0,
        payload_bytes_in_frag=1,
        flags=0,
    )
    payload = b"\x42"
    wire = pack_frame(h, payload)
    assert len(wire) == HEADER_BYTES + 1

    h2, p2 = unpack_frame(wire)
    assert h2 == h
    assert p2 == payload


def test_round_trip_realistic() -> None:
    """Realistic header + 5800-byte payload survives pack→unpack
    byte-identical."""
    h = _realistic_header()
    payload = bytes((i * 31 + 7) & 0xFF for i in range(h.payload_bytes_in_frag))
    wire = pack_frame(h, payload)
    assert len(wire) == HEADER_BYTES + h.payload_bytes_in_frag

    h2, p2 = unpack_frame(wire)
    assert h2 == h
    assert p2 == payload
    # Round-trip the to_dict() representation too — useful for log
    # consumers and capture-meta JSON readers.
    d = h2.to_dict()
    assert d["pattern_id"] == f"0x{h.pattern_id:016x}"
    assert d["quantized"] is True
    assert d["last_in_block"] is True


def test_round_trip_largest_payload() -> None:
    """A fragment at exactly ``DEFAULT_MAX_FRAG_PAYLOAD_BYTES`` round-
    trips."""
    h = _realistic_header(
        n_filled=DEFAULT_MAX_FRAG_PAYLOAD_BYTES // 2,    # cint8 cplx = 2 B/cell
        payload_bytes_in_frag=DEFAULT_MAX_FRAG_PAYLOAD_BYTES,
    )
    payload = bytes(range(256)) * (DEFAULT_MAX_FRAG_PAYLOAD_BYTES // 256) + bytes(
        DEFAULT_MAX_FRAG_PAYLOAD_BYTES % 256
    )
    assert len(payload) == DEFAULT_MAX_FRAG_PAYLOAD_BYTES
    wire = pack_frame(h, payload)
    h2, p2 = unpack_frame(wire)
    assert h2 == h
    assert p2 == payload


# ---------------------------------------------------------------------------
# 8.3 Fragmentation
# ---------------------------------------------------------------------------


def test_split_payload_into_fragments_exact_mtu() -> None:
    """Payload sized to a clean N × MTU produces N equal fragments."""
    mtu = 64
    payload = bytes(i & 0xFF for i in range(mtu * 3))
    frags = split_payload_into_fragments(payload, max_frag_payload_bytes=mtu)
    assert len(frags) == 3
    assert all(len(f) == mtu for f in frags)
    assert b"".join(frags) == payload


def test_split_payload_into_fragments_partial_last() -> None:
    """Payload of size ``2 * MTU + 17`` produces 3 fragments of sizes
    [MTU, MTU, 17]."""
    mtu = 100
    payload = bytes(i & 0xFF for i in range(mtu * 2 + 17))
    frags = split_payload_into_fragments(payload, max_frag_payload_bytes=mtu)
    assert [len(f) for f in frags] == [mtu, mtu, 17]
    assert b"".join(frags) == payload


def test_split_payload_into_fragments_zero_length() -> None:
    """A zero-length payload returns ``[b""]`` so (frag_idx, n_frags=1)
    accounting on the receiver remains well-defined."""
    assert split_payload_into_fragments(b"") == [b""]


# ---------------------------------------------------------------------------
# 8.4 Field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("seq", 2**64),
        ("seq", -1),
        ("n_grid", 2**16),
        ("frag_idx", 2**16),
        ("chgroup", -1),
        ("n_filled", 2**32),
        ("pattern_id", 2**64),
        ("bits_per_cell", 8),      # protocol: not in VALID_BITS_PER_CELL
        ("bits_per_cell", 4),
        ("t_int_factor", 2),       # protocol: not in VALID_T_INT_FACTORS
        ("t_int_factor", 0),
        ("payload_bytes_in_frag", 2**32),
    ],
)
def test_pack_field_range_overflow(field: str, value: int) -> None:
    """Out-of-range fields raise :class:`ProdFrameFieldRangeError` at
    pack time."""
    h = _realistic_header(**{field: value})
    with pytest.raises(ProdFrameFieldRangeError, match=field):
        pack_header(h)


def test_pack_reserved_flag_bit2_raises() -> None:
    """v1 senders MUST NOT set bit2 (DEDISP no-emit) — pack raises."""
    h = _realistic_header(flags=FLAG_RESERVED_BIT2)
    with pytest.raises(ProdFrameFieldRangeError, match="bit2"):
        pack_header(h)


def test_pack_unknown_flag_bit_raises() -> None:
    """Forward-compat: any bits outside the defined v1 set must be 0."""
    h = _realistic_header(flags=(1 << 5))
    with pytest.raises(ProdFrameFieldRangeError, match="reserved bits"):
        pack_header(h)


def test_pack_frag_idx_geq_n_frags_raises() -> None:
    """``frag_idx`` must be strictly less than ``n_frags``."""
    h = _realistic_header(frag_idx=3, n_frags=3)
    with pytest.raises(ProdFrameFieldRangeError, match="frag_idx"):
        pack_header(h)


def test_pack_nan_scale_is_permitted() -> None:
    """NaN ``scale`` is permitted (propagates through dequant as
    well-defined NaN); the unpack must round-trip the NaN bit pattern."""
    h = _realistic_header(scale=float("nan"))
    wire = pack_header(h)
    h2 = unpack_header(wire)
    assert math.isnan(h2.scale)
    assert h2.offset == h.offset


def test_pack_frame_payload_length_mismatch_raises() -> None:
    """``pack_frame`` rejects ``payload_bytes_in_frag != len(payload)``."""
    h = _realistic_header(payload_bytes_in_frag=100)
    with pytest.raises(ProdFrameLengthError, match="payload_bytes_in_frag"):
        pack_frame(h, b"too short")


# ---------------------------------------------------------------------------
# 8.5 Unpack errors
# ---------------------------------------------------------------------------


def test_unpack_buffer_too_short_raises_length_error() -> None:
    """Buffer shorter than 72 bytes raises ProdFrameLengthError."""
    with pytest.raises(ProdFrameLengthError, match="too short"):
        unpack_header(b"\x00" * (HEADER_BYTES - 1))


def test_unpack_bad_magic_raises_magic_error() -> None:
    """First 4 bytes != MAGIC raises ProdFrameMagicError."""
    h = _realistic_header()
    blob = pack_header(h)
    bad = bytearray(blob)
    struct.pack_into("<I", bad, 0, 0xDEADBEEF)
    with pytest.raises(ProdFrameMagicError, match="0xdeadbeef"):
        unpack_header(bytes(bad))


def test_unpack_bad_version_raises_version_error() -> None:
    """Version != 1 raises ProdFrameVersionError."""
    h = _realistic_header()
    blob = pack_header(h)
    bad = bytearray(blob)
    struct.pack_into("<H", bad, 4, 2)
    with pytest.raises(ProdFrameVersionError, match="version"):
        unpack_header(bytes(bad))


def test_unpack_payload_length_mismatch_raises_length_error() -> None:
    """Buffer length != HEADER_BYTES + payload_bytes_in_frag raises
    ProdFrameLengthError."""
    h = _realistic_header(payload_bytes_in_frag=100)
    blob = pack_header(h)
    # Append 50 bytes instead of 100.
    wire = blob + b"\x00" * 50
    with pytest.raises(ProdFrameLengthError, match="buffer length"):
        unpack_frame(wire)


def test_unpack_bad_bits_per_cell_raises_field_range_error() -> None:
    """Tampering ``bits_per_cell`` to 8 on the wire raises at unpack."""
    h = _realistic_header()
    blob = pack_header(h)
    bad = bytearray(blob)
    struct.pack_into("<B", bad, 48, 8)
    with pytest.raises(ProdFrameFieldRangeError, match="bits_per_cell"):
        unpack_header(bytes(bad))


def test_unpack_unknown_flag_bit_raises_field_range_error() -> None:
    """Tampering flags to set bit5 on the wire raises at unpack
    (forward-compat: v2 senders should bump version, not set v1-
    reserved bits)."""
    h = _realistic_header()
    blob = pack_header(h)
    bad = bytearray(blob)
    struct.pack_into("<H", bad, 6, h.flags | (1 << 5))
    with pytest.raises(ProdFrameFieldRangeError, match="reserved bits"):
        unpack_header(bytes(bad))


# ---------------------------------------------------------------------------
# Helpers: expected_payload_bytes + pattern_id re-export sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_filled,bits,expected",
    [
        (0, 16, 0),
        (1, 16, 2),
        (5800, 16, 11_600),
        (5800, 32, 23_200),
        (1, 32, 4),
    ],
)
def test_expected_payload_bytes(
    n_filled: int, bits: int, expected: int
) -> None:
    assert expected_payload_bytes(n_filled, bits) == expected


def test_expected_payload_bytes_bad_bits_raises() -> None:
    with pytest.raises(ProdFrameFieldRangeError):
        expected_payload_bytes(100, 8)


def test_predict_pattern_id_re_exported() -> None:
    """``predict_pattern_id`` is re-exported from sparsity_pattern so
    TX/RX callers don't need to import the gridder module directly."""
    assert callable(predict_pattern_id)
    # Sanity: re-export points to the live sparsity_pattern impl.
    from dsart.grid.sparsity_pattern import predict_pattern_id as upstream

    assert predict_pattern_id is upstream


def test_constants_match_plan_contract() -> None:
    """Pin the protocol-level constants the TX/RX agents will key on."""
    assert VERSION == 1
    assert HEADER_BYTES == 72
    assert BITS_CINT8_COMPLEX == 16
    assert BITS_CFP16_COMPLEX == 32
    assert VALID_BITS_PER_CELL == (16, 32)
    assert VALID_T_INT_FACTORS == (1, 4, 8, 16, 32, 64, 128)
    assert FLAG_QUANTIZED == 0x01
    assert FLAG_LAST_IN_BLOCK == 0x02
    assert FLAG_RESERVED_BIT2 == 0x04
    assert FLAG_NOISE_WARMUP == 0x08
    assert FLAG_RFI_WARMING_UP == 0x10
    assert DEFAULT_MAX_FRAG_PAYLOAD_BYTES == 8964


# ---------------------------------------------------------------------------
# Bonus: the dataclass is hashable so RX-side dedup caches can key on it.
# ---------------------------------------------------------------------------


def test_header_is_hashable_and_immutable() -> None:
    """``frozen=True`` makes the dataclass hashable + immutable."""
    h = _realistic_header()
    assert hash(h) == hash(_realistic_header())
    with pytest.raises(Exception):
        h.seq = 0  # type: ignore[misc]


def test_replace_creates_modified_copy() -> None:
    """``dataclasses.replace`` works on the frozen dataclass — useful
    for the TX side bumping ``frag_idx`` per fragment without
    reconstructing every field."""
    h = _realistic_header(frag_idx=0, n_frags=3)
    h1 = replace(h, frag_idx=1)
    h2 = replace(h, frag_idx=2)
    assert h1.frag_idx == 1 and h2.frag_idx == 2
    assert h.frag_idx == 0
