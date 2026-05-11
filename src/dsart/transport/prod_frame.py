"""``ProdFrame`` — production 72-byte value-channel header (M4a chunk 1).

This module defines the **production** corr → search value-channel frame
that the M4a-extended :class:`dsart.transport.tx.TransportTx` emits and
:class:`dsart.transport.rx.TransportRx` consumes. It is the peer of the
M3 chunk-8 :mod:`dsart.transport.frame` module's ``FastVisFrame`` (32-byte
simplified header); both modules co-exist during M4a, with `prod_frame`
becoming the canonical wire form for the chunk-7 net-loopback bench and
all downstream production paths.

The wire format is pinned by plan §4.3 lines 1409-1444; the field
semantics carry over from plan §3 line 307 (``pattern_id`` is a hash of
*inputs*, not output bytes) and §3.6 (Stokes-I, V-4 convention, single
side +uv). For the full chunk-1 freeze, including module hand-off
contracts to chunks 2 (TX) and 3+4+5 (RX), see
``docs/m4a/chunk1_prod_frame_spec.md``.

# Header layout (72 bytes, little-endian, packed)

::

    offset  size    field             type     notes
    0       4       magic             u32      = 0xD5A1107E ("DSA110 7E")
    4       2       version           u16      = 1
    6       2       flags             u16      bit0=quantized,
                                                bit1=last_in_block,
                                                bit2=reserved (DEDISP no-emit; v1 senders MUST NOT set),
                                                bit3=noise_warmup (search-side; RX ignores),
                                                bit4=rfi_warming_up (soft warm-up, payload still valid)
    8       8       seq               u64      monotonic per (corr, dm_idx) flow
    16      8       specnum           u64      SNAP packet seq at block start
    24      2       chgroup           u16      0..15
    26      2       dm_idx            u16      coarse-DM idx in global DM plan
    28      2       frag_idx          u16      0..n_frags-1
    30      2       n_frags           u16      total fragments for this seq
    32      2       n_grid            u16      grid side length
    34      2       reserved0         u16      pad to 8-B align n_filled
    36      4       n_filled          u32      cells in sparsity pattern (sum of payload_cells)
    40      8       pattern_id        u64      blake2b_64(...) per §3 line 307
    48      1       bits_per_cell     u8       16 (cint8 complex) or 32 (cfp16 complex)
    49      1       t_int_factor      u8       1, 4, 8, 16, 32, 64, 128
    50      2       reserved1         u16      pad to 4-B align scale
    52      4       scale             f32      x_real = scale * q + offset
    56      4       offset            f32
    60      4       payload_bytes_in_frag  u32
    64      8       reserved2[8]      u8[8]    sender writes \\0
    total: 72 bytes

# Why no CRC

Unlike chunk-8's :class:`FastVisFrame` (which carries a CRC32), the
production header has no CRC. The plan §4.3 integrity gate is the
``pattern_id`` per-payload verify + the receiver's per-(corr, dm_idx)
seq reorder window + the ``n_filled`` consistency check (sum of
``payload_bytes_in_frag`` across fragments must equal
``n_filled * bits_per_cell // 8``). Inserting a CRC32 would consume one
of the ``reserved*`` slots and break the 8-byte alignment guarantees on
``pattern_id`` / 4-byte alignment on ``scale``/``offset`` — both of
which are required so a C receiver can ``__builtin_memcpy`` the header
into a packed struct without unaligned-load penalties.

# What this module does NOT do

* No socket I/O — pure codec module.
* No fragment reassembly — chunk 3 (``rx.py``) owns the reorder window.
* No ``scale`` / ``offset`` computation — caller computes from filled cells.
* No ``pattern_id`` computation — re-exported from
  :mod:`dsart.grid.sparsity_pattern.predict_pattern_id`.
* No mon-key emission — owned by chunks 2 (TX) and 3 (RX).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

# Re-exported for TX/RX-side callers so they don't import the gridder module
# directly.
from dsart.grid.sparsity_pattern import predict_pattern_id  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC: Final[int] = 0xD5A1107E
"""Wire-format magic (\"DSA110 7E\") per plan §4.3 line 1412."""

VERSION: Final[int] = 1
"""Current wire-format version. Future revs that add fields without
changing existing offsets keep version=1; layout-breaking revs bump to
2 and run alongside v1 senders/receivers during the transition."""

HEADER_BYTES: Final[int] = 72
"""Fixed 72-byte production header (plan §4.3 line 1409)."""

DEFAULT_MTU_BYTES: Final[int] = 9000
"""Default MTU: jumbo frames on ``nic.search``. Conservative-MTU
deployments override to 1500."""

DEFAULT_MAX_FRAG_PAYLOAD_BYTES: Final[int] = 8964
"""Default per-fragment payload cap: MTU 9000 − IPv4 (20) − UDP (8) −
8 B kernel-side slack = 8964 B. Plan §4.3 line 1409 cites 8964 B for
9000 B MTU."""

# bits_per_cell allowed values (uint8 at header byte 48).
BITS_CINT8_COMPLEX: Final[int] = 16
"""``bits_per_cell`` value for cint8 complex (re+im = 2 × int8). The
operational default per plan §9."""

BITS_CFP16_COMPLEX: Final[int] = 32
"""``bits_per_cell`` value for cfp16 complex (re+im = 2 × fp16). Debug
/ wider-dynamic-range path per plan §9."""

VALID_BITS_PER_CELL: Final[tuple[int, ...]] = (
    BITS_CINT8_COMPLEX,
    BITS_CFP16_COMPLEX,
)

VALID_T_INT_FACTORS: Final[tuple[int, ...]] = (1, 4, 8, 16, 32, 64, 128)
"""Allowed ``t_int_factor`` values per plan §9 ops table."""

# flags bitfield (uint16 LE at header bytes 6-7) — see plan §4.3 line 1414.
FLAG_QUANTIZED: Final[int] = 1 << 0
"""bit0: payload is quantized cint8 (vs cfp16). Distinct from
``bits_per_cell`` because future v2 may introduce other quantization
modes (e.g., 4-bit) that are also \"quantized\" with a different
``bits_per_cell``."""

FLAG_LAST_IN_BLOCK: Final[int] = 1 << 1
"""bit1: last fragment of a (specnum, dm_idx) block; allows the
receiver to commit the slot eagerly without waiting for the reorder
window to slide."""

FLAG_RESERVED_BIT2: Final[int] = 1 << 2
"""bit2: reserved (DEDISP no-emit). v1 senders MUST NOT set this; v1
receivers MUST drop frames that do, treating the sender as
non-conformant. Kept reserved for future DEDISP cold-start signalling."""

FLAG_NOISE_WARMUP: Final[int] = 1 << 3
"""bit3: search-side flag. v1 RX ignores; carried for round-trip
preservation when M5/M6 inspector tools snoop frames."""

FLAG_RFI_WARMING_UP: Final[int] = 1 << 4
"""bit4: RFI Stat-B 150 s burn-in window. Soft warm-up — payload is
still emitted with reduced bandpass-outlier coverage. NOT subject to
the DEDISP no-emit policy. Mirrors chunk-8 ``FLAG_RFI_WARMING_UP``
(which lives at bit0 in the chunk-8 header; production bit-position is
bit4 per plan §4.3 line 1417)."""

_FLAG_KNOWN_BITS: Final[int] = (
    FLAG_QUANTIZED
    | FLAG_LAST_IN_BLOCK
    | FLAG_RESERVED_BIT2
    | FLAG_NOISE_WARMUP
    | FLAG_RFI_WARMING_UP
)


# struct format: little-endian, no padding.
#
# Field order MUST match the plan §4.3 byte-offset table exactly. Every
# native-alignment gap in that table is filled by an explicit reserved*
# field, so this struct format string emits zero implicit padding.
#
#   <  little-endian, native sizes, no padding
#   I  uint32 magic
#   H  uint16 version
#   H  uint16 flags
#   Q  uint64 seq
#   Q  uint64 specnum
#   H  uint16 chgroup
#   H  uint16 dm_idx
#   H  uint16 frag_idx
#   H  uint16 n_frags
#   H  uint16 n_grid
#   H  uint16 reserved0
#   I  uint32 n_filled
#   Q  uint64 pattern_id
#   B  uint8  bits_per_cell
#   B  uint8  t_int_factor
#   H  uint16 reserved1
#   f  float32 scale
#   f  float32 offset
#   I  uint32 payload_bytes_in_frag
#   8s 8-byte reserved2
_HEADER_FMT: Final[str] = (
    "<I H H Q Q H H H H H H I Q B B H f f I 8s"
)
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct(_HEADER_FMT)
assert _HEADER_STRUCT.size == HEADER_BYTES, (
    f"ProdFrame header layout drifted: {_HEADER_STRUCT.size} != {HEADER_BYTES}"
)


# Range bounds reused by the pack-side validator.
_U8_MAX: Final[int] = 0xFF
_U16_MAX: Final[int] = 0xFFFF
_U32_MAX: Final[int] = 0xFFFF_FFFF
_U64_MAX: Final[int] = 0xFFFF_FFFF_FFFF_FFFF
_F32_MAX: Final[float] = 3.4028234663852886e38  # IEEE-754 single-precision max


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProdFrameMagicError(ValueError):
    """Raised when a buffer's first 4 bytes do not match :data:`MAGIC`."""


class ProdFrameVersionError(ValueError):
    """Raised when the header's ``version`` field does not match
    :data:`VERSION`. The high-rate RX hot path catches this and drops
    the datagram + bumps a mon-key; tests assert that this exception is
    raised so callers don't accidentally accept future-version frames."""


class ProdFrameLengthError(ValueError):
    """Raised when:

    * ``payload_bytes_in_frag`` from the header does not match the
      actual payload length in the buffer, OR
    * ``n_filled * bits_per_cell`` is not a whole-byte multiple, OR
    * the buffer is shorter than :data:`HEADER_BYTES`.
    """


class ProdFrameFieldRangeError(ValueError):
    """Raised at pack time when a field overflows its declared range,
    or at pack/unpack time when a protocol-level constant is violated
    (``bits_per_cell ∉ {16, 32}``, ``t_int_factor ∉ VALID_T_INT_FACTORS``,
    ``flags & FLAG_RESERVED_BIT2``, or ``flags`` has bits outside the
    defined v1 set)."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProdFrameHeader:
    """One ProdFrame header (no payload).

    ``frozen=True`` + ``slots=True`` keeps the RX hot path allocation
    cost low — no per-instance ``__dict__`` and immutable after
    unpack. Callers wanting to mutate construct a new instance.

    Field semantics follow plan §4.3 lines 1411-1442 (which is the
    single source of truth — this docstring summarises but does not
    duplicate). ``magic``, ``reserved0``, ``reserved1``, and
    ``reserved2`` are NOT carried on the dataclass: ``magic`` is a
    protocol constant (always :data:`MAGIC`), and the ``reserved*``
    fields are pure padding that the codec emits as zero on the wire
    and validates as zero on receive (with a tolerated-non-zero
    warning path for forward-compat).
    """

    seq: int
    specnum: int
    chgroup: int
    dm_idx: int
    frag_idx: int
    n_frags: int
    n_grid: int
    n_filled: int
    pattern_id: int
    bits_per_cell: int
    t_int_factor: int
    scale: float
    offset: float
    payload_bytes_in_frag: int
    flags: int = 0
    version: int = VERSION

    @property
    def quantized(self) -> bool:
        return bool(self.flags & FLAG_QUANTIZED)

    @property
    def last_in_block(self) -> bool:
        return bool(self.flags & FLAG_LAST_IN_BLOCK)

    @property
    def rfi_warming_up(self) -> bool:
        return bool(self.flags & FLAG_RFI_WARMING_UP)

    @property
    def noise_warmup(self) -> bool:
        return bool(self.flags & FLAG_NOISE_WARMUP)

    def to_dict(self) -> dict:
        """Diagnostic dict (logs / capture-meta JSON / mon-key drain).

        ``pattern_id`` is rendered as hex so JSON consumers can
        distinguish from other 64-bit fields at a glance.
        """
        return {
            "version": self.version,
            "flags": self.flags,
            "seq": self.seq,
            "specnum": self.specnum,
            "chgroup": self.chgroup,
            "dm_idx": self.dm_idx,
            "frag_idx": self.frag_idx,
            "n_frags": self.n_frags,
            "n_grid": self.n_grid,
            "n_filled": self.n_filled,
            "pattern_id": f"0x{self.pattern_id:016x}",
            "bits_per_cell": self.bits_per_cell,
            "t_int_factor": self.t_int_factor,
            "scale": self.scale,
            "offset": self.offset,
            "payload_bytes_in_frag": self.payload_bytes_in_frag,
            "rfi_warming_up": self.rfi_warming_up,
            "noise_warmup": self.noise_warmup,
            "last_in_block": self.last_in_block,
            "quantized": self.quantized,
        }


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _validate_int_range(name: str, value: int, lo: int, hi: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProdFrameFieldRangeError(
            f"ProdFrame.{name}={value!r} must be int, got {type(value).__name__}"
        )
    if not (lo <= value <= hi):
        raise ProdFrameFieldRangeError(
            f"ProdFrame.{name}={value} out of range [{lo}, {hi}]"
        )


def _validate_f32_range(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProdFrameFieldRangeError(
            f"ProdFrame.{name}={value!r} must be float, got {type(value).__name__}"
        )
    if value != value:  # NaN check
        return  # NaN is permitted: it propagates through the dequant
    if abs(float(value)) > _F32_MAX:
        raise ProdFrameFieldRangeError(
            f"ProdFrame.{name}={value!r} overflows float32 range"
        )


def _validate_protocol_constants(
    bits_per_cell: int,
    t_int_factor: int,
    flags: int,
    version: int,
) -> None:
    if bits_per_cell not in VALID_BITS_PER_CELL:
        raise ProdFrameFieldRangeError(
            f"ProdFrame.bits_per_cell={bits_per_cell!r} ∉ {VALID_BITS_PER_CELL}"
        )
    if t_int_factor not in VALID_T_INT_FACTORS:
        raise ProdFrameFieldRangeError(
            f"ProdFrame.t_int_factor={t_int_factor!r} ∉ {VALID_T_INT_FACTORS}"
        )
    if version != VERSION:
        raise ProdFrameVersionError(
            f"ProdFrame.version={version!r} != current VERSION={VERSION}"
        )
    if flags & FLAG_RESERVED_BIT2:
        raise ProdFrameFieldRangeError(
            f"ProdFrame.flags=0x{flags:04x} has reserved bit2 set; v1 senders "
            f"MUST NOT assert FLAG_RESERVED_BIT2 (DEDISP no-emit; deferred to v2)"
        )
    unknown = flags & ~_FLAG_KNOWN_BITS
    if unknown:
        raise ProdFrameFieldRangeError(
            f"ProdFrame.flags=0x{flags:04x} has unknown bits set "
            f"(0x{unknown:04x}); v1 reserved bits 5..15 must be 0"
        )


# ---------------------------------------------------------------------------
# Public codec API
# ---------------------------------------------------------------------------


def pack_header(h: ProdFrameHeader) -> bytes:
    """Serialise a :class:`ProdFrameHeader` to its 72-byte wire form.

    Validates every field against its declared u8/u16/u32/u64/f32 range
    and against the v1 protocol-constants set (``bits_per_cell``,
    ``t_int_factor``, ``flags``, ``version``). Raises
    :class:`ProdFrameFieldRangeError` for range / protocol violations
    and :class:`ProdFrameVersionError` for version mismatches.

    The returned bytes are exactly :data:`HEADER_BYTES` long.
    """
    _validate_protocol_constants(
        h.bits_per_cell, h.t_int_factor, h.flags, h.version
    )
    _validate_int_range("seq", h.seq, 0, _U64_MAX)
    _validate_int_range("specnum", h.specnum, 0, _U64_MAX)
    _validate_int_range("chgroup", h.chgroup, 0, _U16_MAX)
    _validate_int_range("dm_idx", h.dm_idx, 0, _U16_MAX)
    _validate_int_range("frag_idx", h.frag_idx, 0, _U16_MAX)
    _validate_int_range("n_frags", h.n_frags, 1, _U16_MAX)
    _validate_int_range("n_grid", h.n_grid, 0, _U16_MAX)
    _validate_int_range("n_filled", h.n_filled, 0, _U32_MAX)
    _validate_int_range("pattern_id", h.pattern_id, 0, _U64_MAX)
    _validate_int_range(
        "payload_bytes_in_frag", h.payload_bytes_in_frag, 0, _U32_MAX
    )
    _validate_int_range("flags", h.flags, 0, _U16_MAX)
    _validate_f32_range("scale", h.scale)
    _validate_f32_range("offset", h.offset)
    if h.frag_idx >= h.n_frags:
        raise ProdFrameFieldRangeError(
            f"ProdFrame.frag_idx={h.frag_idx} >= n_frags={h.n_frags}"
        )

    return _HEADER_STRUCT.pack(
        MAGIC,
        int(h.version),
        int(h.flags),
        int(h.seq),
        int(h.specnum),
        int(h.chgroup),
        int(h.dm_idx),
        int(h.frag_idx),
        int(h.n_frags),
        int(h.n_grid),
        0,                            # reserved0
        int(h.n_filled),
        int(h.pattern_id),
        int(h.bits_per_cell),
        int(h.t_int_factor),
        0,                            # reserved1
        float(h.scale),
        float(h.offset),
        int(h.payload_bytes_in_frag),
        b"\x00" * 8,                  # reserved2
    )


def unpack_header(buf: bytes | bytearray | memoryview) -> ProdFrameHeader:
    """Deserialise the first :data:`HEADER_BYTES` bytes of ``buf`` into
    a :class:`ProdFrameHeader`.

    Validates magic, version, and the v1 protocol-constants set.
    Reserved fields (``reserved0``, ``reserved1``, ``reserved2``) are
    silently dropped — a future v2 may carve fields out of those slots,
    so a non-zero v2 reserved is not a v1 protocol error.

    Raises:
        ProdFrameLengthError: ``buf`` shorter than :data:`HEADER_BYTES`.
        ProdFrameMagicError: first 4 bytes != :data:`MAGIC`.
        ProdFrameVersionError: ``version`` != :data:`VERSION`.
        ProdFrameFieldRangeError: ``bits_per_cell``, ``t_int_factor``,
            or ``flags`` violates v1 protocol constants.
    """
    buf_b = bytes(buf)
    if len(buf_b) < HEADER_BYTES:
        raise ProdFrameLengthError(
            f"ProdFrame buffer too short: got {len(buf_b)}, need >= {HEADER_BYTES}"
        )
    (
        magic,
        version,
        flags,
        seq,
        specnum,
        chgroup,
        dm_idx,
        frag_idx,
        n_frags,
        n_grid,
        _reserved0,
        n_filled,
        pattern_id,
        bits_per_cell,
        t_int_factor,
        _reserved1,
        scale,
        offset,
        payload_bytes_in_frag,
        _reserved2,
    ) = _HEADER_STRUCT.unpack_from(buf_b, 0)

    if magic != MAGIC:
        raise ProdFrameMagicError(
            f"ProdFrame bad magic: got 0x{magic:08x}, expected 0x{MAGIC:08x}"
        )
    _validate_protocol_constants(bits_per_cell, t_int_factor, flags, version)

    return ProdFrameHeader(
        seq=seq,
        specnum=specnum,
        chgroup=chgroup,
        dm_idx=dm_idx,
        frag_idx=frag_idx,
        n_frags=n_frags,
        n_grid=n_grid,
        n_filled=n_filled,
        pattern_id=pattern_id,
        bits_per_cell=bits_per_cell,
        t_int_factor=t_int_factor,
        scale=scale,
        offset=offset,
        payload_bytes_in_frag=payload_bytes_in_frag,
        flags=flags,
        version=version,
    )


def pack_frame(
    header: ProdFrameHeader,
    payload: bytes | bytearray | memoryview,
) -> bytes:
    """Pack header + payload into one fragment's wire bytes.

    The caller is responsible for splitting large payloads into
    fragments first (use :func:`split_payload_into_fragments`) and for
    setting ``header.payload_bytes_in_frag == len(payload)``. ``pack_frame``
    validates that consistency and raises
    :class:`ProdFrameLengthError` on mismatch — failing fast keeps a
    miscoded TX caller from producing on-the-wire bytes that the RX
    will only catch later.
    """
    payload_b = bytes(payload)
    if header.payload_bytes_in_frag != len(payload_b):
        raise ProdFrameLengthError(
            f"ProdFrame.payload_bytes_in_frag={header.payload_bytes_in_frag} "
            f"!= len(payload)={len(payload_b)}"
        )
    return pack_header(header) + payload_b


def unpack_frame(
    buf: bytes | bytearray | memoryview,
) -> tuple[ProdFrameHeader, bytes]:
    """Deserialise one fragment's wire bytes into (header, payload).

    Validates that the buffer length matches ``HEADER_BYTES +
    header.payload_bytes_in_frag`` exactly. A mismatch means either a
    truncated UDP packet or a misformatted sender — raises
    :class:`ProdFrameLengthError`.
    """
    header = unpack_header(buf)
    buf_b = bytes(buf)
    expected_len = HEADER_BYTES + header.payload_bytes_in_frag
    if len(buf_b) != expected_len:
        raise ProdFrameLengthError(
            f"ProdFrame buffer length {len(buf_b)} != "
            f"HEADER_BYTES ({HEADER_BYTES}) + payload_bytes_in_frag "
            f"({header.payload_bytes_in_frag}) = {expected_len}"
        )
    payload = buf_b[HEADER_BYTES:expected_len]
    return header, payload


def split_payload_into_fragments(
    payload: bytes | bytearray | memoryview,
    *,
    max_frag_payload_bytes: int = DEFAULT_MAX_FRAG_PAYLOAD_BYTES,
) -> list[bytes]:
    """Split a contiguous payload byte buffer into fragments of size
    ``<= max_frag_payload_bytes``.

    Returns a list of ``bytes`` slices where every fragment except
    possibly the last is exactly ``max_frag_payload_bytes`` long, and
    the concatenation reproduces the input byte-for-byte. A zero-length
    input returns ``[b""]`` (one empty fragment) so the (frag_idx,
    n_frags) accounting on the receiver remains well-defined.
    """
    if max_frag_payload_bytes <= 0:
        raise ValueError(
            f"max_frag_payload_bytes={max_frag_payload_bytes} must be > 0"
        )
    payload_b = bytes(payload)
    if len(payload_b) == 0:
        return [b""]
    return [
        payload_b[i : i + max_frag_payload_bytes]
        for i in range(0, len(payload_b), max_frag_payload_bytes)
    ]


def expected_payload_bytes(n_filled: int, bits_per_cell: int) -> int:
    """Whole-payload byte size for an N_filled-cell value vector at the
    given ``bits_per_cell``.

    Returns ``n_filled * bits_per_cell // 8``. Used by the RX side to
    validate ``sum(payload_bytes_in_frag for f in frags) ==
    expected_payload_bytes(n_filled, bits_per_cell)`` once all fragments
    for a seq have arrived.

    Raises:
        ProdFrameFieldRangeError: ``bits_per_cell`` is not in
            :data:`VALID_BITS_PER_CELL`, OR
            ``n_filled * bits_per_cell`` is not a whole-byte multiple.
    """
    if bits_per_cell not in VALID_BITS_PER_CELL:
        raise ProdFrameFieldRangeError(
            f"bits_per_cell={bits_per_cell!r} ∉ {VALID_BITS_PER_CELL}"
        )
    if n_filled < 0:
        raise ProdFrameFieldRangeError(f"n_filled={n_filled} must be >= 0")
    total_bits = n_filled * bits_per_cell
    if total_bits % 8 != 0:
        raise ProdFrameLengthError(
            f"n_filled={n_filled} * bits_per_cell={bits_per_cell} = {total_bits} "
            f"is not a whole-byte multiple"
        )
    return total_bits // 8


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


__all__ = [
    # constants
    "MAGIC",
    "VERSION",
    "HEADER_BYTES",
    "DEFAULT_MTU_BYTES",
    "DEFAULT_MAX_FRAG_PAYLOAD_BYTES",
    "BITS_CINT8_COMPLEX",
    "BITS_CFP16_COMPLEX",
    "VALID_BITS_PER_CELL",
    "VALID_T_INT_FACTORS",
    "FLAG_QUANTIZED",
    "FLAG_LAST_IN_BLOCK",
    "FLAG_RESERVED_BIT2",
    "FLAG_NOISE_WARMUP",
    "FLAG_RFI_WARMING_UP",
    # data
    "ProdFrameHeader",
    # codec
    "pack_header",
    "unpack_header",
    "pack_frame",
    "unpack_frame",
    "split_payload_into_fragments",
    "expected_payload_bytes",
    # re-exported from sparsity_pattern
    "predict_pattern_id",
    # errors
    "ProdFrameMagicError",
    "ProdFrameVersionError",
    "ProdFrameLengthError",
    "ProdFrameFieldRangeError",
]
