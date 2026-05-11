# M4a chunk 1: `prod_frame.py` — production 72-byte value-channel header

**Status**: spec; chunk-1 implementation pending.
**Owner**: M4a driver (synchronous; before TX/RX agents fan out).
**Plan refs**:
- §4.3 lines 1409-1444 — wire format
- §3 line 307 — `pattern_id` semantics
- §4.4 line 1462 — `pattern_id` mismatch handling
- §3.6 Stokes-I + V-4 — single-pol convention
- M4a_PLAN_FIXES.md "Chunk-1 wire-format freeze"

This module is the **single synchronization point between the TX and RX
agents**. Once landed, chunks 2 (TX) and 3+4+5 (RX) can proceed in
parallel without further joint commits.

---

## 1. Module layout

```
src/dsart/transport/prod_frame.py     # the codec
tests/transport/test_prod_frame.py    # the test suite
```

`prod_frame.py` is a **peer** of `frame.py` (M3 chunk 8's `FastVisFrame`),
not a replacement. Both modules live side-by-side throughout M4a. M3's
loopback bench keeps consuming `frame.FastVisFrame`; M4a's loopback bench
(chunk 7) is the first consumer of `prod_frame.ProdFrame`.

---

## 2. Constants (`prod_frame.py` top of module)

```python
from typing import Final
import struct

MAGIC: Final[int] = 0xD5A1107E  # "DSA110 7E"
VERSION: Final[int] = 1
HEADER_BYTES: Final[int] = 72
DEFAULT_MTU_BYTES: Final[int] = 9000      # jumbo on nic.search
DEFAULT_MAX_FRAG_PAYLOAD_BYTES: Final[int] = 8964  # 9000 - 20 (IPv4) - 8 (UDP) - 8 spare

# bits_per_cell values (header byte 48):
BITS_CINT8_COMPLEX: Final[int] = 16   # 2 × int8 (re, im)
BITS_CFP16_COMPLEX: Final[int] = 32   # 2 × fp16 (re, im)

# t_int_factor allowed values (header byte 49). Mirrors plan §9.
VALID_T_INT_FACTORS: Final[tuple[int, ...]] = (1, 4, 8, 16, 32, 64, 128)

# flags bitfield (header bytes 6-7, uint16 LE):
FLAG_QUANTIZED: Final[int]      = 1 << 0   # bit0; payload is quantized cint8 (vs cfp16)
FLAG_LAST_IN_BLOCK: Final[int]  = 1 << 1   # bit1; last fragment of a (specnum, dm_idx) block
FLAG_RESERVED_BIT2: Final[int]  = 1 << 2   # bit2; DEDISP no-emit; v1 senders MUST NOT set
FLAG_NOISE_WARMUP: Final[int]   = 1 << 3   # bit3; search-side flag (RX ignores)
FLAG_RFI_WARMING_UP: Final[int] = 1 << 4   # bit4; soft warm-up, payload still valid
# bits5..15 reserved; sender writes 0; receiver MAY warn on non-zero
```

---

## 3. Header byte layout

Reproduced from plan §4.3 lines 1411-1442 (the source of truth), packed
to 72 bytes, little-endian:

```
offset  size    field             type     notes
0       4       magic             u32      = 0xD5A1107E
4       2       version           u16      = 1
6       2       flags             u16      see FLAG_* constants
8       8       seq               u64      monotonic per (corr, dm_idx)
16      8       specnum           u64      SNAP packet seq at block start
24      2       chgroup           u16      0..15
26      2       dm_idx            u16      coarse-DM index in global DM plan
28      2       frag_idx          u16      0..n_frags-1
30      2       n_frags           u16      total fragments for this seq
32      2       n_grid            u16      grid side length
34      2       reserved0         u16      sender writes 0; pad to 8-B align n_filled
36      4       n_filled          u32      # cells in sparsity pattern; sum of payload_cells across frags
40      8       pattern_id        u64      blake2b_64(...) per §3 line 307
48      1       bits_per_cell     u8       16 (cint8 cplx) or 32 (cfp16 cplx)
49      1       t_int_factor      u8       1, 4, 8, 16, 32, 64, 128
50      2       reserved1         u16      sender writes 0; pad to 4-B align scale
52      4       scale             f32      dequant: x_real = scale * q + offset
56      4       offset            f32
60      4       payload_bytes_in_frag  u32  length of this frag's payload (cells * bits/8)
64      8       reserved2[8]      u8[8]    sender writes \x00\x00\x00\x00\x00\x00\x00\x00
total:  72 bytes
```

`struct` format string (little-endian, no padding):

```python
_HEADER_FMT: Final[str] = "<I H H Q Q H H H H H H I Q B B H f f I 8s"
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct(_HEADER_FMT)
assert _HEADER_STRUCT.size == 72
```

**Why this struct format produces 72 bytes**: every field is at its
natural-alignment boundary because the plan deliberately reserved the
`reserved0` and `reserved1` pads to make `n_filled` 4-aligned,
`pattern_id` 8-aligned, and `scale`/`offset` 4-aligned. `struct` in
`<` mode emits no implicit padding, so the natural-alignment-matching
layout pinned in the plan composes cleanly with `struct.pack`.

**No CRC field in the production header** (unlike chunk-8's `FastVisFrame`):
- The receiver's `pattern_id` check + `seq` reorder window + `n_filled`
  consistency check is the integrity gate. The plan §4.3 doesn't budget
  a CRC; CRC was a chunk-8 simplification to test the framing logic
  end-to-end without yet having `pattern_id` from M3 chunk-3a.
- Adding a CRC32 (4 bytes) would either eat one of the `reserved*` slots
  or break the 72-byte / 8-B alignment, so chunk-1 doesn't introduce one.
- M4b (the cross-host phase) and beyond may revisit if UDP-over-real-NIC
  corruption shows up in `bench/net_pair.py` invariants.

---

## 4. `ProdFrameHeader` dataclass

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ProdFrameHeader:
    """One ProdFrame header (no payload).

    Frozen + slots so the receiver's hot path doesn't allocate per-
    instance __dict__. Pack/unpack go through ProdFrame.* classmethods
    that take or return ``(ProdFrameHeader, bytes_payload)`` tuples to
    keep the dataclass strictly metadata.
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
```

No `magic`, `reserved0`, `reserved1`, `reserved2` on the dataclass —
those are protocol constants (magic) or padding (reserved*) and the
caller never sets them.

---

## 5. Public API

```python
def pack_header(h: ProdFrameHeader) -> bytes: ...
def unpack_header(buf: bytes | bytearray | memoryview) -> ProdFrameHeader: ...

def pack_frame(h: ProdFrameHeader, payload: bytes | memoryview) -> bytes: ...
def unpack_frame(buf: bytes | bytearray | memoryview) -> tuple[ProdFrameHeader, bytes]: ...

# Convenience for the TX side (chunk 2):
def split_payload_into_fragments(
    payload: bytes | memoryview,
    *,
    max_frag_payload_bytes: int = DEFAULT_MAX_FRAG_PAYLOAD_BYTES,
) -> list[bytes]: ...

# Convenience for the RX side (chunk 3):
def expected_payload_bytes(
    n_filled: int, bits_per_cell: int
) -> int:
    """Bytes the *whole* (un-fragmented) payload would occupy: returns
    n_filled * bits_per_cell // 8. Used by the RX-side n_filled / sum-
    of-frag consistency check (raises ValueError if bits_per_cell is
    not 16 or 32, or if n_filled * bits_per_cell is not a whole-byte
    multiple)."""
```

`pack_frame`'s contract: `payload` is **already one fragment's worth of
bytes**; `pack_frame` does not split. Callers wanting fragmentation use
`split_payload_into_fragments` first and pack each piece with the
right `(frag_idx, n_frags, payload_bytes_in_frag)`.

---

## 6. Exceptions

```python
class ProdFrameMagicError(ValueError):
    """Buffer's first 4 bytes do not match MAGIC (0xD5A1107E)."""

class ProdFrameVersionError(ValueError):
    """Header version != 1 (forward-compat sentinel; receiver should
    drop the datagram + log, not raise into the hot path — but the
    unpack helper raises so callers wrap accordingly)."""

class ProdFrameLengthError(ValueError):
    """``payload_bytes_in_frag`` from the header does not match the
    actual payload length in the buffer, OR ``n_filled * bits_per_cell``
    is not a whole-byte multiple."""

class ProdFrameFieldRangeError(ValueError):
    """A field overflowed its u8/u16/u32/u64/f32 range at pack time, OR
    a flags bit outside the defined v1 set is asserted (bit2 / bit5+)."""
```

Note: `bits_per_cell ∉ {16, 32}` and `t_int_factor ∉ VALID_T_INT_FACTORS`
also raise `ProdFrameFieldRangeError` — these are protocol-level
validations, not arithmetic-range.

---

## 7. `pattern_id` integration

`prod_frame.py` does **not** re-implement the BLAKE2b hash. Chunk 1
re-exports `predict_pattern_id` from `dsart.grid.sparsity_pattern` so
the TX side (chunk 2) doesn't need to import the gridder module
directly:

```python
# At the bottom of prod_frame.py:
from dsart.grid.sparsity_pattern import predict_pattern_id  # re-export

__all__ = [
    "MAGIC", "VERSION", "HEADER_BYTES",
    "BITS_CINT8_COMPLEX", "BITS_CFP16_COMPLEX",
    "FLAG_QUANTIZED", "FLAG_LAST_IN_BLOCK", "FLAG_RESERVED_BIT2",
    "FLAG_NOISE_WARMUP", "FLAG_RFI_WARMING_UP",
    "ProdFrameHeader",
    "pack_header", "unpack_header",
    "pack_frame", "unpack_frame",
    "split_payload_into_fragments", "expected_payload_bytes",
    "predict_pattern_id",
    "ProdFrameMagicError", "ProdFrameVersionError",
    "ProdFrameLengthError", "ProdFrameFieldRangeError",
]
```

The TX side computes `pattern_id` once per `cmd: prepare` and caches it
on a per-`chgroup` basis; it does **not** call `predict_pattern_id` per
datagram. Chunk 2's TX-state struct holds `pattern_id_by_chgroup:
dict[int, int]`.

The RX side does the same — computes `pattern_id` for its local
`SparsityPattern` cache on `cmd: prepare` and looks up per-`chgroup` for
verification. Chunk 3 carries the receiver-side lookup logic.

---

## 8. Test plan (`tests/transport/test_prod_frame.py`)

Total: 14 tests, all `pytest.mark`-free (no GPU, no h01-specific
fixtures; runs on any host). Target line+branch coverage on
`prod_frame.py` is **100%**.

### 8.1 Header layout pins (3 tests)

1. `test_header_struct_size_is_72_bytes` — assert
   `HEADER_STRUCT.size == 72`, `HEADER_BYTES == 72`.
2. `test_header_field_offsets` — pack a header with every field set to
   a known sentinel (`seq=0xAA...`, `specnum=0xBB...`, etc.) and check
   the byte at each offset matches the plan §4.3 layout. Uses
   `struct.unpack_from("<I", b, 0)` etc. to read individual fields.
3. `test_header_magic_constant` — assert `MAGIC == 0xD5A1107E`.

### 8.2 Round-trip identity (3 tests)

4. `test_round_trip_minimal` — pack a header with zero-filled scalars +
   1-byte payload; unpack; assert byte-identical reproduction.
5. `test_round_trip_realistic` — pack with realistic ops values
   (`seq=12345, specnum=0xCAFEBABE, n_grid=256, n_filled=5800,
   pattern_id=0xDEADBEEFCAFEBABE, bits=16, t_int=8, scale=1.5e-3,
   offset=-0.5, payload_bytes_in_frag=5800`) + a 5800-byte payload;
   unpack; assert dataclass field-equality and byte-identical payload.
6. `test_round_trip_largest_payload` — pack `DEFAULT_MAX_FRAG_PAYLOAD_BYTES`
   payload; unpack; assert correctness.

### 8.3 Fragmentation (2 tests)

7. `test_split_payload_into_fragments_exact_mtu` — payload that's an
   exact multiple of MTU produces N fragments each of size MTU and the
   re-concatenation byte-identical to the input.
8. `test_split_payload_into_fragments_partial_last` — payload of size
   `MTU * 2 + 17` produces 3 fragments of `[MTU, MTU, 17]`.

### 8.4 Field validation (3 tests)

9. `test_pack_field_range_overflow` — parametrized: passing
   `seq=2**64`, `n_grid=2**16`, `frag_idx=2**16`, `bits_per_cell=8`
   (invalid), `t_int_factor=2` (invalid) each raises
   `ProdFrameFieldRangeError`.
10. `test_pack_reserved_flag_bit2_raises` — setting `flags &
    FLAG_RESERVED_BIT2` at pack time raises (v1 senders MUST NOT set
    bit2).
11. `test_pack_unknown_flag_bit_raises` — setting `flags & (1 << 5)`
    raises (forward-compat: reserved bits must be 0).

### 8.5 Unpack errors (3 tests)

12. `test_unpack_bad_magic_raises_magic_error` — flip first 4 bytes to
    `0xDEADBEEF`; unpack raises `ProdFrameMagicError`.
13. `test_unpack_bad_version_raises_version_error` — patch bytes 4-5 to
    `version=2`; unpack raises `ProdFrameVersionError`.
14. `test_unpack_payload_length_mismatch_raises_length_error` — pack a
    header with `payload_bytes_in_frag=100` but supply 50 bytes of
    payload; unpack raises `ProdFrameLengthError`.

---

## 9. Verification on h01

After chunk-1 lands locally, the M4a driver:

1. `git push origin m4a/main`
2. SSH to h01: `cd ~/proj/dsa110-rt-m4a && git fetch && git checkout m4a/main && git pull`
3. `pip install -e .` (worktree already exists; just refresh)
4. `pytest tests/transport/test_prod_frame.py -q --tb=short`
5. Expect 14/14 pass; no warnings.
6. Capture pytest output to `~/dsart-m4a-chunk1-verify.log`.

---

## 10. What chunk 1 explicitly does NOT do

(Inertial pins so the TX/RX agents don't get confused about scope.)

- **No socket I/O**: zero `socket`, `recv*`, `send*`, or `select` calls.
  `prod_frame.py` is a pure codec module.
- **No fragmentation re-assembly**: `pack_frame` packs one fragment;
  the receiver's reorder-window + bitmap logic lives in chunk 3
  (`transport/rx.py`).
- **No `scale`/`offset` computation**: chunk 1 trusts the caller's
  pre-computed `scale`/`offset` floats. The TX side (chunk 2) is
  responsible for computing them over filled cells per the plan §4.2
  pin (dynamic range tracks actual data, not zeros).
- **No `pattern_id` computation**: re-exports `predict_pattern_id`
  from `sparsity_pattern.py`; does not reimplement BLAKE2b.
- **No mon-key emission**: mon-keys are owned by chunks 2 (TX) and 3
  (RX). `prod_frame.py` is mon-key-silent.

---

## 11. Hand-off to TX / RX agents

When chunk 1 is merged into `m4a/main`:

- **TX agent** (chunk 2, branch `m4a/tx-prod-header`) consumes
  `pack_frame` + `split_payload_into_fragments` + the re-exported
  `predict_pattern_id`. Extends `transport/tx.py::TransportTx`
  to emit production frames via a new method
  `_transmit_one_cube_prod(...)` peer to the chunk-8 path.
- **RX agent** (chunks 3+4+5, branch `m4a/rx-defrag-and-ring`)
  consumes `unpack_frame` + the re-exported `predict_pattern_id` +
  the consts. Extends `transport/rx.py::TransportRx` with a per-
  `(corr, dm_idx)` reorder window + per-payload `pattern_id` verify;
  adds `transport/recv_ring.{c,py}` + `transport/production_rx_ring.py`.

The TX and RX agents do **not** modify `prod_frame.py` after chunk 1
lands (the wire format is frozen). If a TX/RX issue surfaces that
requires a wire-format tweak, the M4a driver writes a chunk-1b patch
on `m4a/main` and both agents rebase.
