"""``FastVisFrame`` — wire-format for fast-vis cubes (M3 chunk 8).

This module defines the **fast-vis cube frame** that the corr-side
:class:`dsart.transport.tx.TransportTx` emits and the search-side
:class:`dsart.transport.rx.TransportRx` consumes.

The chunk-8 contract is intentionally simpler than plan §4.3's
production format (the 72-byte header with `pattern_id`, `scale`,
`offset`, `n_frags`, etc. — that lives in M4a). Chunk 8's job is to
prove the loopback **codec + sequence accounting + CRC validation +
chunk-4 ``TransportTxStage`` Protocol compliance** end-to-end on a
single host. Production fields like `pattern_id`, fragment reassembly,
and per-payload `scale`/`offset` are deferred to M4a.

# Header layout (32 bytes, little-endian, packed)

::

    offset  size    field           type     notes
    0       4       magic           u32      = 0xD5A0FA57 (DSA0FAST)
    4       4       seq             u32      strictly monotonic per (TX, chgroup)
    8       1       chgroup         u8       0..15 (corr-node chgroup)
    9       1       dm_idx          u8       coarse-DM trial index
    10      2       t_idx           u16      fast-vis tile index within block
    12      2       n_grid          u16      grid side length (256 prod)
    14      1       dtype_code      u8       0=cfp16, 1=cint8
    15      1       flags           u8       bit0=rfi_warming_up; bits1-7=reserved
    16      4       payload_bytes   u32      length of payload that follows
    20      4       crc32           u32      CRC32 of (header w/ crc=0) + payload
    24      8       _pad            bytes    rounds header to 32 B; sender MUST 0

The 8-byte trailing pad is reserved for the M4a handoff (we will likely
absorb a 64-bit ``pattern_id`` here without re-versioning the wire
format on the search-side decoder).

# Magic value

``0xD5A0FA57`` is the ASCII-leetspeak rendering "DSA0 FAST" — mirrors
plan §4.3 line 1388's ``0xD5A1107E`` ("DSA110 7E") convention. The
chunk-8 spec writes it as ``0xDSA0FA57`` for memorability; the actual
hex value is ``0xD5A0FA57`` (S → 5, T → 7).

# CRC

CRC32 (zlib's IEEE polynomial; ``zlib.crc32``) over the **whole header
with the ``crc32`` field zeroed**, concatenated with the payload bytes.
Computed by the sender; verified by the receiver before the payload is
trusted.

# Why a 32-byte header

Production §4.3 uses 72 bytes (with `scale`, `offset`, `pattern_id`,
fragment fields, etc.). For chunk-8 we deliberately keep the header
small: chunk-4's transport hand-off doesn't yet have any of those
fields, and a 32-byte header keeps the per-frame overhead under 0.05%
at production payload sizes (~65 KB). M4a will extend to the 72-byte
production header — at that point ``FastVisFrame`` becomes the
chunk-8-only wire form and a peer ``ProdFrame`` carries the production
header.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC: Final[int] = 0xD5A0FA57
"""Frame-type marker for fast-vis cubes (DSA0 FAST in leetspeak)."""

DTYPE_CFP16: Final[int] = 0
"""Complex fp16 payload: 2 × fp16 (re, im) per cell = 4 bytes/cell."""

DTYPE_CINT8: Final[int] = 1
"""Complex int8 payload: 2 × int8 (re, im) per cell = 2 bytes/cell."""

FLAG_RFI_WARMING_UP: Final[int] = 0x01
"""Bit0 of the frame.flags byte: set when the upstream RFI flagger is
in its Stat-B 150 s warm-up window. Mirrors plan §4.3 header
``flags.bit4=rfi_warming_up`` semantics (we use bit0 here because the
chunk-8 flags byte is dedicated to rfi-warmup; the production header
re-purposes bits 0..3 for other roles)."""

# struct format: little-endian, 32 bytes total
# I=u32, B=u8, H=u16
# Layout: magic, seq, chgroup, dm_idx, t_idx, n_grid, dtype_code, flags,
#         payload_bytes, crc32, then 8 bytes of pad.
_HEADER_FMT: Final[str] = "<II BB H H BB I I 8s"
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct(_HEADER_FMT)
HEADER_BYTES: Final[int] = _HEADER_STRUCT.size
assert HEADER_BYTES == 32, (
    f"FastVisFrame header layout drifted: HEADER_BYTES={HEADER_BYTES}, "
    f"expected 32"
)

# Offset of the crc32 field within the header — we zero this byte slice
# during CRC computation.
_CRC_OFFSET: Final[int] = 20
_CRC_END: Final[int] = _CRC_OFFSET + 4

# Default per-frame payload cap. Chosen to fit safely inside a single
# UDP datagram on a 64 KiB-MTU loopback socket (max UDP payload =
# 65507 = 65535 − 20 (IPv4) − 8 (UDP)). Production §4.3 with 9000 B
# jumbo MTU caps fragments at ~8964 B.
DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 65000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FrameMagicError(ValueError):
    """Raised when a buffer's first 4 bytes do not match :data:`MAGIC`."""


class FrameCRCError(ValueError):
    """Raised when CRC32 over (zeroed-CRC header + payload) does not
    match the wire ``crc32`` field. Indicates corruption (bit-flips on
    the wire OR a truncated payload OR sender bug)."""


class FramePayloadOversizeError(ValueError):
    """Raised when ``payload_bytes`` exceeds ``max_payload_bytes`` at
    pack time. The TX caller is expected to fragment OR shrink the
    cube; chunk 8 does not implement fragmentation (production uses
    ``n_frags`` per plan §4.3, deferred to M4a)."""


# ---------------------------------------------------------------------------
# Dataclass + codec
# ---------------------------------------------------------------------------


@dataclass
class FastVisFrame:
    """One fast-vis cube frame (header + payload).

    Constructed by :meth:`pack` (TX side) or :meth:`unpack` (RX side).

    Args:
        seq: per-flow strictly-monotonic sequence number (TX bumps
            this once per frame transmitted).
        chgroup: corr-node chgroup id 0..15.
        dm_idx: coarse-DM trial index (0..N_DM-1).
        t_idx: fast-vis tile index within block (0..n_fast_vis-1).
        n_grid: grid side length (256 in production).
        dtype_code: payload dtype: :data:`DTYPE_CFP16` (0) or
            :data:`DTYPE_CINT8` (1).
        flags: bitfield. bit0=rfi_warming_up. bits1-7 reserved.
        payload: raw payload bytes. The dtype interpretation is set
            by ``dtype_code``; the FastVisFrame does NOT decode the
            payload (callers handle (re, im) packing).
    """

    seq: int
    chgroup: int
    dm_idx: int
    t_idx: int
    n_grid: int
    dtype_code: int
    flags: int
    payload: bytes

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    @property
    def rfi_warming_up(self) -> bool:
        return bool(self.flags & FLAG_RFI_WARMING_UP)

    def to_dict(self) -> dict:
        """Diagnostic dict (for logs / capture-meta JSON / tests)."""
        return {
            "seq": self.seq,
            "chgroup": self.chgroup,
            "dm_idx": self.dm_idx,
            "t_idx": self.t_idx,
            "n_grid": self.n_grid,
            "dtype_code": self.dtype_code,
            "flags": self.flags,
            "payload_bytes": self.payload_bytes,
            "rfi_warming_up": self.rfi_warming_up,
        }

    # --- TX-side ---------------------------------------------------------

    def pack(self, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> bytes:
        """Serialise to wire bytes (header + payload).

        The ``crc32`` field is computed over the full header (with the
        crc32 field zeroed) concatenated with the payload, then patched
        into the header bytes before return.

        Raises:
            FramePayloadOversizeError: if ``len(payload) >
                max_payload_bytes``. (The header itself is fixed-
                size; the cap applies to the payload only.)
            ValueError: if any field overflows its u32/u16/u8 range.
        """
        if len(self.payload) > max_payload_bytes:
            raise FramePayloadOversizeError(
                f"FastVisFrame payload={len(self.payload)} bytes exceeds "
                f"max_payload_bytes={max_payload_bytes}; production "
                f"would fragment via plan §4.3 n_frags (deferred to M4a)"
            )
        _validate_field("seq", self.seq, 0, 0xFFFF_FFFF)
        _validate_field("chgroup", self.chgroup, 0, 0xFF)
        _validate_field("dm_idx", self.dm_idx, 0, 0xFF)
        _validate_field("t_idx", self.t_idx, 0, 0xFFFF)
        _validate_field("n_grid", self.n_grid, 0, 0xFFFF)
        _validate_field("dtype_code", self.dtype_code, 0, 0xFF)
        _validate_field("flags", self.flags, 0, 0xFF)

        header = _HEADER_STRUCT.pack(
            MAGIC,
            int(self.seq),
            int(self.chgroup),
            int(self.dm_idx),
            int(self.t_idx),
            int(self.n_grid),
            int(self.dtype_code),
            int(self.flags),
            len(self.payload),
            0,                                                           # crc32 placeholder
            b"\x00" * 8,                                                 # _pad
        )
        crc_buf = bytearray(header)
        crc_buf[_CRC_OFFSET:_CRC_END] = b"\x00\x00\x00\x00"
        crc = zlib.crc32(bytes(crc_buf) + self.payload) & 0xFFFF_FFFF
        out = bytearray(header)
        struct.pack_into("<I", out, _CRC_OFFSET, crc)
        return bytes(out) + self.payload

    # --- RX-side ---------------------------------------------------------

    @classmethod
    def unpack(cls, buf: bytes | bytearray | memoryview) -> "FastVisFrame":
        """Deserialise wire bytes → ``FastVisFrame``.

        Validates:
            * Buffer is at least :data:`HEADER_BYTES` long.
            * Magic == :data:`MAGIC` (else :class:`FrameMagicError`).
            * Buffer is exactly ``HEADER_BYTES + payload_bytes`` long.
            * CRC32 matches (else :class:`FrameCRCError`).

        Returns:
            Reconstructed ``FastVisFrame``.

        Raises:
            FrameMagicError: invalid magic.
            FrameCRCError: CRC mismatch (corruption / wrong sender).
            ValueError: buffer length / payload length inconsistency.
        """
        buf_b = bytes(buf)
        if len(buf_b) < HEADER_BYTES:
            raise ValueError(
                f"FastVisFrame.unpack: buffer too short: got {len(buf_b)}, "
                f"need >= {HEADER_BYTES} bytes"
            )
        (
            magic,
            seq,
            chgroup,
            dm_idx,
            t_idx,
            n_grid,
            dtype_code,
            flags,
            payload_bytes,
            wire_crc,
            _pad,
        ) = _HEADER_STRUCT.unpack_from(buf_b, 0)

        if magic != MAGIC:
            raise FrameMagicError(
                f"FastVisFrame.unpack: bad magic: got 0x{magic:08x}, "
                f"expected 0x{MAGIC:08x}"
            )
        expected_len = HEADER_BYTES + payload_bytes
        if len(buf_b) != expected_len:
            raise ValueError(
                f"FastVisFrame.unpack: buffer length {len(buf_b)} != "
                f"header {HEADER_BYTES} + payload_bytes {payload_bytes} "
                f"= {expected_len}"
            )

        # Recompute CRC over (zeroed-crc header + payload) and compare.
        crc_buf = bytearray(buf_b[:HEADER_BYTES])
        crc_buf[_CRC_OFFSET:_CRC_END] = b"\x00\x00\x00\x00"
        actual_crc = zlib.crc32(
            bytes(crc_buf) + buf_b[HEADER_BYTES:]
        ) & 0xFFFF_FFFF
        if actual_crc != wire_crc:
            raise FrameCRCError(
                f"FastVisFrame.unpack: CRC mismatch: wire=0x{wire_crc:08x}, "
                f"computed=0x{actual_crc:08x} (seq={seq}, chgroup={chgroup}, "
                f"dm_idx={dm_idx}, t_idx={t_idx})"
            )

        payload = buf_b[HEADER_BYTES:HEADER_BYTES + payload_bytes]
        return cls(
            seq=seq,
            chgroup=chgroup,
            dm_idx=dm_idx,
            t_idx=t_idx,
            n_grid=n_grid,
            dtype_code=dtype_code,
            flags=flags,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_field(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int):
        raise TypeError(
            f"FastVisFrame.{name}={value!r} must be int, got {type(value).__name__}"
        )
    if not (lo <= value <= hi):
        raise ValueError(
            f"FastVisFrame.{name}={value} out of range [{lo}, {hi}]"
        )
