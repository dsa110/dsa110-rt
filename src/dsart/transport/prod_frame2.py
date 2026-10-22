"""Frame v2 — the 2x corner-turn datagram (``docs/x2/SPEC.md`` §5, FROZEN).

One datagram carries ``k`` consecutive stream samples of ONE (stream, coarse
DM) flow. A stream sample is the full value row of one (chgroup, sub-band)
stream at one tier-clock sample: ``n_heavy`` heavy values (cint8) followed by
``n_values - n_heavy`` main values (cint4/5/6, packed). Values and their
order come from the value map (SPEC §3); the codes come from the row rule
(SPEC §4, reference :mod:`dsart.x2.quant`).

The v1 frame (:mod:`dsart.transport.prod_frame`) is unchanged; both share
``MAGIC`` and keep ``version`` at byte offset 4, so a receiver dispatches on
:func:`peek_version` before parsing.

Header layout (72 bytes, little-endian, packed)::

    off size field          type   notes
    0   4    magic          u32    0xD5A1107E (same as v1)
    4   2    version        u16    2
    6   2    flags          u16    bit0 QUANTIZED (must be 1)
                                   bit1 LAST_IN_BLOCK (last datagram of this
                                        (stream, dm) flow in the corr block)
                                   bit2 reserved (must be 0, as in v1)
                                   bit3 NOISE_WARMUP, bit4 RFI_WARMING_UP (v1)
                                   bit5 HEAVY_PRESENT (== n_heavy > 0)
                                   bit6 ROW_GUARD (every sample here was coded
                                        at its own row scale, SPEC §4 step 3)
                                   bits 7..15 must be 0
    8   8    seq_first      u64    tier-clock row of the first sample =
                                   t0_native / t_int_factor (exact)
    16  8    t0_native      u64    absolute NATIVE-sample index (32.768 us) of
                                   the first sample = 2 x SNAP specnum (same
                                   epoch as SNAP specnum 0; C2's
                                   ``search_to_snap_specnum`` returns specnums,
                                   65.536 us); external row = t0_native // 16
                                   at 2x (// 32 at 1x), SPEC §8/§9
    24  2    stream_id      u16    chgroup * n_sub + s
    26  2    dm_idx         u16    global coarse-DM index d (same offset as v1)
    28  1    k              u8     samples in this datagram, >= 1
    29  1    n_sub          u8     1, 2 or 4
    30  1    bits_main      u8     8 / 10 / 12 = cint4 / cint5 / cint6
    31  1    bits_heavy     u8     16 (cint8) or 0 (no heavy values)
    32  2    n_grid         u16
    34  1    t_int_factor   u8     native samples per stream sample (16 | 32)
    35  1    value_kind     u8     0 gridded cells, 1 unique baselines
    36  4    n_values       u32    values per stream sample, heavy included
    40  8    pattern_id     u64    value-map id (SPEC §3)
    48  4    n_heavy        u32    leading heavy values, <= n_values
    52  4    step_main      f32    s_main  * sigma (the sigma every sample in
                                   this datagram was coded with; monitoring)
    56  4    step_heavy     f32    s_heavy * sigma
    60  4    payload_bytes  u32    k * bytes_per_sample
    64  4    s_main         f32    main decode step in sigma_hat units
    68  4    s_heavy        f32    heavy decode step in sigma_hat units
    total 72

Payload: ``k`` samples back to back; each sample is
``n_heavy x (int8 re, int8 im)`` followed by the main values as ONE packed
little-endian bit stream (value v occupies bits ``[v*W, (v+1)*W)``, re in the
low ``W/2`` bits, im in the high ``W/2`` bits, two's complement, bit i of the
stream is bit ``i % 8`` of byte ``i // 8``), zero-padded to a whole byte.
``bytes_per_sample = n_heavy * bits_heavy / 8 + ceil(n_main * bits_main / 8)``.
Codes are mid-tread two's complement; ``-2^(c-1)`` (-8 / -16 / -32 main,
-128 heavy) is the NO_DATA sentinel and decodes to 0.

Sender rules: ``k <= max_k(bytes_per_sample)`` (largest datagram 72 + 8900 =
8972 B, 9000 B on the wire at MTU 9000 — never IP-fragmented); all samples
of a datagram share ``step_main``, ``step_heavy`` and ROW_GUARD exactly, so a
datagram never straddles a corr block (sigma_hat changes per block) nor a
guarded row (each run of equal steps is framed separately); the last datagram
of a block carries LAST_IN_BLOCK.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC: Final[int] = 0xD5A1107E
VERSION: Final[int] = 2
HEADER_BYTES: Final[int] = 72
MAX_PAYLOAD_BYTES: Final[int] = 8900
"""Payload cap: MTU 9000 - IPv4 20 - UDP 8 - header 72."""

FLAG_QUANTIZED: Final[int] = 1 << 0
FLAG_LAST_IN_BLOCK: Final[int] = 1 << 1
FLAG_RESERVED_BIT2: Final[int] = 1 << 2
FLAG_NOISE_WARMUP: Final[int] = 1 << 3
FLAG_RFI_WARMING_UP: Final[int] = 1 << 4
FLAG_HEAVY_PRESENT: Final[int] = 1 << 5
FLAG_ROW_GUARD: Final[int] = 1 << 6
FLAGS_DEFINED_V2: Final[int] = (
    FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK | FLAG_RESERVED_BIT2
    | FLAG_NOISE_WARMUP | FLAG_RFI_WARMING_UP | FLAG_HEAVY_PRESENT
    | FLAG_ROW_GUARD
)

NATIVE_PER_SPECNUM: Final[int] = 2
"""Native samples (32.768 us) per SNAP specnum (65.536 us)."""
NATIVE_PER_EXT_2X: Final[int] = 16
NATIVE_PER_EXT_1X: Final[int] = 32


def t0_native_from_specnum(specnum: int) -> int:
    """Native-sample index of SNAP specnum ``specnum`` (= 2 x specnum)."""
    return NATIVE_PER_SPECNUM * int(specnum)


def specnum_from_t0_native(t0_native: int) -> int:
    """Inverse of :func:`t0_native_from_specnum` (floor)."""
    return int(t0_native) // NATIVE_PER_SPECNUM


def ext_row(t0_native: int, native_per_ext: int = NATIVE_PER_EXT_2X) -> int:
    """External-clock row (SPEC §8/§9) of a native index."""
    return int(t0_native) // int(native_per_ext)


def no_data_code(comp_bits: int) -> int:
    """NO_DATA sentinel ``-2**(comp_bits - 1)`` (decodes to 0)."""
    return -(1 << (comp_bits - 1))

VALID_BITS_MAIN: Final[tuple[int, ...]] = (8, 10, 12)
VALID_BITS_HEAVY: Final[tuple[int, ...]] = (0, 16)
VALID_N_SUB: Final[tuple[int, ...]] = (1, 2, 4)
VALID_T_INT_FACTORS: Final[tuple[int, ...]] = (1, 4, 8, 16, 32, 64, 128)
VALUE_KIND_CELLS: Final[int] = 0
VALUE_KIND_BASELINES: Final[int] = 1
MAX_K: Final[int] = 255
SEQ_LIMIT: Final[int] = (1 << 56) - 1
"""seq values must stay below 2**56 - 1 (the ring v4 row tag keeps the top
byte for row-state flags and uses all-ones as its WRITING sentinel)."""

#   <  I  H  H  Q  Q  H  H  B B B B H  B B I  Q  I  f  f  I  f  f
_HEADER_FMT: Final[str] = "<IHHQQHHBBBBHBBIQIffIff"
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct(_HEADER_FMT)
assert _HEADER_STRUCT.size == HEADER_BYTES

# Byte offsets (C receivers and the GPU packer mirror these).
OFFSETS: Final[dict[str, int]] = {
    "magic": 0, "version": 4, "flags": 6, "seq_first": 8, "t0_native": 16,
    "stream_id": 24, "dm_idx": 26, "k": 28, "n_sub": 29, "bits_main": 30,
    "bits_heavy": 31, "n_grid": 32, "t_int_factor": 34, "value_kind": 35,
    "n_values": 36, "pattern_id": 40, "n_heavy": 48, "step_main": 52,
    "step_heavy": 56, "payload_bytes": 60, "s_main": 64, "s_heavy": 68,
}

_U8 = 0xFF
_U16 = 0xFFFF
_U32 = 0xFFFF_FFFF
_U64 = 0xFFFF_FFFF_FFFF_FFFF


class Frame2Error(ValueError):
    """Any frame-v2 codec violation (magic, version, range, length)."""


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------


def bytes_per_sample(n_values: int, n_heavy: int, bits_main: int,
                     bits_heavy: int) -> int:
    """Payload bytes of one stream sample."""
    if bits_main not in VALID_BITS_MAIN:
        raise Frame2Error(f"bits_main={bits_main} not in {VALID_BITS_MAIN}")
    if bits_heavy not in VALID_BITS_HEAVY:
        raise Frame2Error(f"bits_heavy={bits_heavy} not in {VALID_BITS_HEAVY}")
    if not 0 <= n_heavy <= n_values:
        raise Frame2Error(f"n_heavy={n_heavy} not in [0, n_values={n_values}]")
    if bits_heavy == 0 and n_heavy:
        raise Frame2Error("n_heavy > 0 requires bits_heavy = 16")
    n_main = n_values - n_heavy
    return n_heavy * bits_heavy // 8 + (n_main * bits_main + 7) // 8


def max_k(bps: int, cap: int = MAX_PAYLOAD_BYTES) -> int:
    """Largest k for a sample of ``bps`` bytes: floor(cap / bps), <= 255."""
    if bps <= 0:
        raise Frame2Error(f"bytes_per_sample={bps} must be > 0")
    k = cap // bps
    if k < 1:
        raise Frame2Error(
            f"one sample ({bps} B) exceeds the {cap} B payload cap")
    return min(k, MAX_K)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Frame2Header:
    seq_first: int
    t0_native: int
    stream_id: int
    dm_idx: int
    k: int
    n_sub: int
    bits_main: int
    bits_heavy: int
    n_grid: int
    t_int_factor: int
    value_kind: int
    n_values: int
    pattern_id: int
    n_heavy: int
    step_main: float
    step_heavy: float
    payload_bytes: int
    s_main: float
    s_heavy: float
    flags: int = FLAG_QUANTIZED
    version: int = VERSION

    @property
    def n_main(self) -> int:
        return self.n_values - self.n_heavy

    @property
    def bytes_per_sample(self) -> int:
        return bytes_per_sample(self.n_values, self.n_heavy, self.bits_main,
                                self.bits_heavy)

    @property
    def row_guard(self) -> bool:
        return bool(self.flags & FLAG_ROW_GUARD)

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
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}
        d["pattern_id"] = f"0x{self.pattern_id:016x}"
        return d


def _check_int(name: str, v: int, lo: int, hi: int) -> None:
    if not isinstance(v, (int, np.integer)) or isinstance(v, bool):
        raise Frame2Error(f"{name}={v!r} must be int")
    if not lo <= int(v) <= hi:
        raise Frame2Error(f"{name}={v} out of range [{lo}, {hi}]")


def validate_header(h: Frame2Header, *, payload_len: int | None = None) -> None:
    """Raise :class:`Frame2Error` unless ``h`` is a conformant v2 header."""
    if h.version != VERSION:
        raise Frame2Error(f"version={h.version} != {VERSION}")
    _check_int("flags", h.flags, 0, _U16)
    if not h.flags & FLAG_QUANTIZED:
        raise Frame2Error("v2 frames must set FLAG_QUANTIZED")
    if h.flags & FLAG_RESERVED_BIT2:
        raise Frame2Error("reserved flag bit2 set")
    if h.flags & ~FLAGS_DEFINED_V2:
        raise Frame2Error(f"undefined flag bits 0x{h.flags & ~FLAGS_DEFINED_V2:04x}")
    if bool(h.flags & FLAG_HEAVY_PRESENT) != (h.n_heavy > 0):
        raise Frame2Error("FLAG_HEAVY_PRESENT must equal (n_heavy > 0)")
    _check_int("seq_first", h.seq_first, 0, _U64)
    _check_int("t0_native", h.t0_native, 0, _U64)
    _check_int("stream_id", h.stream_id, 0, _U16)
    _check_int("dm_idx", h.dm_idx, 0, _U16)
    _check_int("k", h.k, 1, MAX_K)
    _check_int("n_grid", h.n_grid, 0, _U16)
    _check_int("n_values", h.n_values, 1, _U32)
    _check_int("pattern_id", h.pattern_id, 0, _U64)
    _check_int("n_heavy", h.n_heavy, 0, _U32)
    _check_int("payload_bytes", h.payload_bytes, 0, _U32)
    if h.n_sub not in VALID_N_SUB:
        raise Frame2Error(f"n_sub={h.n_sub} not in {VALID_N_SUB}")
    if h.stream_id >= 16 * h.n_sub:
        raise Frame2Error(f"stream_id={h.stream_id} >= 16 * n_sub={h.n_sub}")
    if h.t_int_factor not in VALID_T_INT_FACTORS:
        raise Frame2Error(f"t_int_factor={h.t_int_factor} not in {VALID_T_INT_FACTORS}")
    if h.value_kind not in (VALUE_KIND_CELLS, VALUE_KIND_BASELINES):
        raise Frame2Error(f"value_kind={h.value_kind} not in (0, 1)")
    bps = bytes_per_sample(h.n_values, h.n_heavy, h.bits_main, h.bits_heavy)
    if h.payload_bytes != h.k * bps:
        raise Frame2Error(
            f"payload_bytes={h.payload_bytes} != k*bytes_per_sample={h.k}*{bps}")
    if h.payload_bytes > MAX_PAYLOAD_BYTES:
        raise Frame2Error(f"payload_bytes={h.payload_bytes} > {MAX_PAYLOAD_BYTES}")
    if h.t0_native != h.seq_first * h.t_int_factor:
        raise Frame2Error(
            f"t0_native={h.t0_native} != seq_first*t_int_factor="
            f"{h.seq_first}*{h.t_int_factor}")
    if h.seq_first + h.k >= SEQ_LIMIT:
        raise Frame2Error("seq_first + k >= 2**56 - 1")
    if payload_len is not None and payload_len != h.payload_bytes:
        raise Frame2Error(
            f"payload length {payload_len} != payload_bytes={h.payload_bytes}")


def pack_header(h: Frame2Header) -> bytes:
    validate_header(h)
    return _HEADER_STRUCT.pack(
        MAGIC, VERSION, int(h.flags), int(h.seq_first), int(h.t0_native),
        int(h.stream_id), int(h.dm_idx), int(h.k), int(h.n_sub),
        int(h.bits_main), int(h.bits_heavy), int(h.n_grid),
        int(h.t_int_factor), int(h.value_kind), int(h.n_values),
        int(h.pattern_id), int(h.n_heavy), float(h.step_main),
        float(h.step_heavy), int(h.payload_bytes), float(h.s_main),
        float(h.s_heavy),
    )


def peek_version(buf: bytes | bytearray | memoryview) -> int:
    """Version field of any ProdFrame (v1 or v2): u16 at byte 4, or -1."""
    if len(buf) < 6:
        return -1
    return int.from_bytes(bytes(buf[4:6]), "little")


def unpack_header(buf: bytes | bytearray | memoryview) -> Frame2Header:
    b = bytes(buf[:HEADER_BYTES])
    if len(b) < HEADER_BYTES:
        raise Frame2Error(f"buffer too short: {len(b)} < {HEADER_BYTES}")
    f = _HEADER_STRUCT.unpack(b)
    if f[0] != MAGIC:
        raise Frame2Error(f"bad magic 0x{f[0]:08x}")
    h = Frame2Header(
        version=f[1], flags=f[2], seq_first=f[3], t0_native=f[4],
        stream_id=f[5], dm_idx=f[6], k=f[7], n_sub=f[8], bits_main=f[9],
        bits_heavy=f[10], n_grid=f[11], t_int_factor=f[12], value_kind=f[13],
        n_values=f[14], pattern_id=f[15], n_heavy=f[16], step_main=f[17],
        step_heavy=f[18], payload_bytes=f[19], s_main=f[20], s_heavy=f[21],
    )
    validate_header(h)
    return h


def pack_frame(h: Frame2Header, payload: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    p = payload.tobytes() if isinstance(payload, np.ndarray) else bytes(payload)
    validate_header(h, payload_len=len(p))
    return pack_header(h) + p


def unpack_frame(buf: bytes | bytearray | memoryview) -> tuple[Frame2Header, bytes]:
    h = unpack_header(buf)
    if len(buf) != HEADER_BYTES + h.payload_bytes:
        raise Frame2Error(
            f"datagram length {len(buf)} != {HEADER_BYTES} + {h.payload_bytes}")
    return h, bytes(buf[HEADER_BYTES:])


# ---------------------------------------------------------------------------
# Payload packing
# ---------------------------------------------------------------------------


def _check_codes(codes: np.ndarray, comp_bits: int, what: str) -> np.ndarray:
    c = np.asarray(codes)
    if c.shape[-1:] != (2,):
        raise Frame2Error(f"{what} codes must have a trailing (re, im) axis of 2")
    lo, hi = -(1 << (comp_bits - 1)), (1 << (comp_bits - 1)) - 1
    if c.size and (c.min() < lo or c.max() > hi):
        raise Frame2Error(f"{what} codes outside [{lo}, {hi}] for {comp_bits}-bit")
    return c.astype(np.int64, copy=False)


def pack_main_codes(codes: np.ndarray, bits_main: int) -> np.ndarray:
    """Pack main codes ``[..., n_main, 2]`` into ``uint8 [..., nbytes]``.

    ``nbytes = ceil(n_main * bits_main / 8)``; layout per the module docstring.
    """
    if bits_main not in VALID_BITS_MAIN:
        raise Frame2Error(f"bits_main={bits_main} not in {VALID_BITS_MAIN}")
    b = bits_main // 2
    c = _check_codes(codes, b, "main")
    lead, n_main = c.shape[:-2], c.shape[-2]
    mask = (1 << b) - 1
    field = (c[..., 0] & mask) | ((c[..., 1] & mask) << b)        # [..., n_main]
    nbytes = (n_main * bits_main + 7) // 8
    if bits_main == 8:
        return field.astype(np.uint8).reshape(*lead, nbytes)
    bits = ((field[..., None] >> np.arange(bits_main)) & 1).astype(np.uint8)
    bits = bits.reshape(*lead, n_main * bits_main)
    pad = nbytes * 8 - n_main * bits_main
    if pad:
        bits = np.concatenate(
            [bits, np.zeros((*lead, pad), dtype=np.uint8)], axis=-1)
    return np.packbits(bits, axis=-1, bitorder="little")


def unpack_main_codes(packed: np.ndarray | bytes, n_main: int,
                      bits_main: int) -> np.ndarray:
    """Inverse of :func:`pack_main_codes`: ``uint8 [..., nbytes]`` ->
    ``int8 [..., n_main, 2]``. Padding bits are ignored."""
    if bits_main not in VALID_BITS_MAIN:
        raise Frame2Error(f"bits_main={bits_main} not in {VALID_BITS_MAIN}")
    p = np.frombuffer(packed, dtype=np.uint8) if isinstance(packed, (bytes, bytearray, memoryview)) else np.asarray(packed, dtype=np.uint8)
    nbytes = (n_main * bits_main + 7) // 8
    if p.shape[-1] != nbytes:
        raise Frame2Error(f"packed main length {p.shape[-1]} != {nbytes}")
    lead = p.shape[:-1]
    b = bits_main // 2
    if bits_main == 8:
        field = p.astype(np.int64)
    else:
        bits = np.unpackbits(p, axis=-1, bitorder="little")[..., : n_main * bits_main]
        bits = bits.reshape(*lead, n_main, bits_main).astype(np.int64)
        field = (bits << np.arange(bits_main)).sum(axis=-1)
    mask = (1 << b) - 1
    sign = 1 << (b - 1)
    re = field & mask
    im = (field >> b) & mask
    re = (re ^ sign) - sign
    im = (im ^ sign) - sign
    return np.stack([re, im], axis=-1).astype(np.int8)


def pack_samples(codes_heavy: np.ndarray, codes_main: np.ndarray,
                 bits_main: int) -> np.ndarray:
    """Payload of ``k`` samples: ``codes_heavy [k, n_heavy, 2]`` (int8) and
    ``codes_main [k, n_main, 2]`` -> ``uint8 [k, bytes_per_sample]``."""
    ch = np.asarray(codes_heavy)
    cm = np.asarray(codes_main)
    if ch.ndim != 3 or cm.ndim != 3 or ch.shape[0] != cm.shape[0]:
        raise Frame2Error("codes_heavy/codes_main must be [k, n, 2] with equal k")
    k = cm.shape[0]
    heavy = _check_codes(ch, 8, "heavy").astype(np.int8).view(np.uint8)
    heavy = heavy.reshape(k, ch.shape[1] * 2)
    main = pack_main_codes(cm, bits_main).reshape(k, -1)
    return np.ascontiguousarray(np.concatenate([heavy, main], axis=1))


def unpack_samples(payload: bytes | np.ndarray, k: int, n_values: int,
                   n_heavy: int, bits_main: int,
                   bits_heavy: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_samples` -> ``(codes_heavy, codes_main)``."""
    bps = bytes_per_sample(n_values, n_heavy, bits_main, bits_heavy)
    p = np.frombuffer(payload, dtype=np.uint8) if not isinstance(payload, np.ndarray) else payload.astype(np.uint8, copy=False)
    if p.size != k * bps:
        raise Frame2Error(f"payload size {p.size} != k*bps = {k}*{bps}")
    p = p.reshape(k, bps)
    hb = n_heavy * 2
    heavy = p[:, :hb].copy().view(np.int8).reshape(k, n_heavy, 2)
    main = unpack_main_codes(p[:, hb:], n_values - n_heavy, bits_main)
    return heavy, main


# ---------------------------------------------------------------------------
# Block -> datagrams (reference sender framing)
# ---------------------------------------------------------------------------


def iter_block_datagrams(
    *,
    codes_heavy: np.ndarray,
    codes_main: np.ndarray,
    seq_first: int,
    t_int_factor: int,
    stream_id: int,
    dm_idx: int,
    n_sub: int,
    bits_main: int,
    n_grid: int,
    value_kind: int,
    pattern_id: int,
    step_main,
    step_heavy,
    s_main: float,
    s_heavy: float,
    guarded=None,
    extra_flags: int = 0,
    k: int | None = None,
) -> Iterator[bytes]:
    """Frame one corr block of one (stream, DM) flow into v2 datagrams.

    ``codes_* [S, n, 2]`` hold the block's S samples; ``step_main`` /
    ``step_heavy`` are scalars or per-sample ``[S]`` arrays and ``guarded`` an
    optional per-sample bool array (SPEC §4 row guard). Samples are framed in
    runs of identical (step_main, step_heavy, guarded), at most ``k`` per
    datagram (default :func:`max_k`); the block's last datagram carries
    ``FLAG_LAST_IN_BLOCK``. This is the reference the GPU packer matches.
    """
    ch = np.asarray(codes_heavy)
    cm = np.asarray(codes_main)
    S, n_heavy = ch.shape[0], ch.shape[1]
    n_values = n_heavy + cm.shape[1]
    bits_heavy = 16 if n_heavy else 0
    bps = bytes_per_sample(n_values, n_heavy, bits_main, bits_heavy)
    kk = max_k(bps) if k is None else int(k)
    if not 1 <= kk <= max_k(bps):
        raise Frame2Error(f"k={kk} not in [1, {max_k(bps)}]")
    sm = np.broadcast_to(np.asarray(step_main, dtype=np.float32), (S,))
    sh = np.broadcast_to(np.asarray(step_heavy, dtype=np.float32), (S,))
    gd = (np.zeros(S, bool) if guarded is None
          else np.broadcast_to(np.asarray(guarded, dtype=bool), (S,)))
    payload = pack_samples(ch, cm, bits_main)                       # [S, bps]
    base_flags = FLAG_QUANTIZED | int(extra_flags)
    if n_heavy:
        base_flags |= FLAG_HEAVY_PRESENT
    j0 = 0
    while j0 < S:
        j1 = j0 + 1
        while (j1 < S and j1 - j0 < kk and gd[j1] == gd[j0]
               and sm[j1].tobytes() == sm[j0].tobytes()
               and sh[j1].tobytes() == sh[j0].tobytes()):
            j1 += 1
        flags = base_flags | (FLAG_LAST_IN_BLOCK if j1 == S else 0)
        if gd[j0]:
            flags |= FLAG_ROW_GUARD
        seq = seq_first + j0
        h = Frame2Header(
            seq_first=seq, t0_native=seq * t_int_factor, stream_id=stream_id,
            dm_idx=dm_idx, k=j1 - j0, n_sub=n_sub, bits_main=bits_main,
            bits_heavy=bits_heavy, n_grid=n_grid, t_int_factor=t_int_factor,
            value_kind=value_kind, n_values=n_values, pattern_id=pattern_id,
            n_heavy=n_heavy, step_main=float(sm[j0]), step_heavy=float(sh[j0]),
            payload_bytes=(j1 - j0) * bps, s_main=s_main, s_heavy=s_heavy,
            flags=flags,
        )
        yield pack_header(h) + payload[j0:j1].tobytes()
        j0 = j1


__all__ = [
    "MAGIC", "VERSION", "HEADER_BYTES", "MAX_PAYLOAD_BYTES", "OFFSETS",
    "FLAG_QUANTIZED", "FLAG_LAST_IN_BLOCK", "FLAG_RESERVED_BIT2",
    "FLAG_NOISE_WARMUP", "FLAG_RFI_WARMING_UP", "FLAG_HEAVY_PRESENT",
    "FLAG_ROW_GUARD", "FLAGS_DEFINED_V2", "NATIVE_PER_SPECNUM",
    "NATIVE_PER_EXT_2X", "NATIVE_PER_EXT_1X", "t0_native_from_specnum",
    "specnum_from_t0_native", "ext_row", "no_data_code", "VALID_BITS_MAIN", "VALID_BITS_HEAVY",
    "VALID_N_SUB", "VALID_T_INT_FACTORS", "VALUE_KIND_CELLS",
    "VALUE_KIND_BASELINES", "MAX_K", "SEQ_LIMIT", "Frame2Error",
    "Frame2Header", "bytes_per_sample", "max_k", "validate_header",
    "pack_header", "unpack_header", "pack_frame", "unpack_frame",
    "peek_version", "pack_main_codes", "unpack_main_codes", "pack_samples",
    "unpack_samples", "iter_block_datagrams",
]
