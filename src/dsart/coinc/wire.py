"""C1 ↔ C2 wire-format encoders / parsers (locked schema_version = 1).

The byte-level contract is in
``docs/c1c2/C1C2_WIRE_SCHEMA.md`` — this module is its sole owner.

Two surfaces:

  * ``C1BatchEncoder`` — builds the ASCII candsfile-like batch the
    C1 emitter writes onto its TCP socket.  Pure (no I/O) so unit
    tests can round-trip against ``parse_c1_batch``.
  * ``parse_c1_batch`` — async streaming parser the C2 receiver feeds
    one line at a time; returns a fully-validated ``C1Batch`` on END
    or raises ``BadBatch`` on framing error.
  * ``encode_c2_trigger`` / ``decode_c2_trigger`` — the 64-byte
    little-endian UDP trigger packet.

The module intentionally has no asyncio dependency.  The service
modules wrap these calls behind their own transport tasks.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "C2_TRIGGER_PACKET_SIZE",
    "C2_TRIGGER_MAGIC",
    "C2_TRIGGER_FLAG_DUMP_CUBE",
    "C2_TRIGGER_FLAG_DUMP_VOLTAGE",
    "C1CandidateRow",
    "C1BatchHeader",
    "C1Batch",
    "BadBatch",
    "C1BatchEncoder",
    "parse_c1_batch",
    "encode_c2_trigger",
    "decode_c2_trigger",
]


SCHEMA_VERSION: int = 1
"""Bumping requires sub-agent review (see docs/c1c2/C1C2_WIRE_SCHEMA.md §0)."""

# Magic: ASCII 'D','S','R','T' = 0x44535254 big-endian = 0x54525344 LE.
C2_TRIGGER_MAGIC: int = 0x54525344
C2_TRIGGER_PACKET_SIZE: int = 64
C2_TRIGGER_FLAG_DUMP_CUBE: int = 1 << 0
#: Voltage-dump bit (corr-node retention service). Additive to the locked
#: schema_version=1 wire struct — it merely gives meaning to a
#: previously-always-zero ``flags`` bit, so no schema bump is required.
#: The C2 cube broadcast sets DUMP_CUBE; the C2 voltage broadcast (to the
#: 16 corr nodes) sets DUMP_VOLTAGE. See docs/voltage_dumps/.
C2_TRIGGER_FLAG_DUMP_VOLTAGE: int = 1 << 1

_TRIGGER_STRUCT = struct.Struct("<IHH16sqdII16s")
assert _TRIGGER_STRUCT.size == C2_TRIGGER_PACKET_SIZE, (
    _TRIGGER_STRUCT.size,
    C2_TRIGGER_PACKET_SIZE,
)


class BadBatch(ValueError):
    """Raised by ``parse_c1_batch`` on framing / schema errors."""


@dataclass(frozen=True, slots=True)
class C1CandidateRow:
    snr: float
    l_rad: float
    m_rad: float
    l_pix: int
    m_pix: int
    dm_pc_cc: float
    dm_idx_global: int
    fine_dm_idx: int
    event_specnum: int
    width_samples: int
    kernel_id: str
    flags: int


@dataclass(frozen=True, slots=True)
class C1BatchHeader:
    schema_version: int
    cube_id: int
    event_specnum_start: int
    mjd_start: float
    sample_period_specnum: int
    sample_period_us: float
    n_grid: int
    n_fdm_in_cube: int
    search_node_id: int
    gpu_half: int
    n_candidates: int

    def candidate_mjd(self, event_specnum: int) -> float:
        """Convert a candidate's ``event_specnum`` to MJD using this header.

        Mirrors the recipe in ``docs/c1c2/C1C2_WIRE_SCHEMA.md §1.1``.

        Units contract: ``event_specnum_start`` is ``slot.specnum_start``
        (``services/search_compute.py:1776-1782``) and the row's
        ``event_specnum`` is ``specnum_start + t_idx``
        (``detector/decoder.py:216``) — **both count SEARCH samples**, so
        their difference is already a detector-sample count.
        ``sample_period_us`` is the SEARCH-sample period (1048.576 µs at
        the production op-point), so the product is straight
        microseconds. Dividing the delta by ``sample_period_specnum``
        first — as this did until 2026-08-06 — made every reported event
        MJD early by up to ~0.25 s; the same trap is called out at
        ``services/search_compute.py:1338-1345``, where it was fixed for
        the cube MJD clock.

        ``sample_period_specnum`` is native SNAP specnums per search
        sample and belongs to native-specnum conversion only; the one
        canonical converter for that is
        ``services.coincidencer.search_to_snap_specnum``.
        """
        samples_since_start = event_specnum - self.event_specnum_start
        return (
            self.mjd_start
            + samples_since_start * self.sample_period_us / 1e6 / 86400.0
        )


@dataclass(frozen=True, slots=True)
class C1Batch:
    header: C1BatchHeader
    candidates: Tuple[C1CandidateRow, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# C1 emitter — pure encoder used by the search-node side
# ---------------------------------------------------------------------------


class C1BatchEncoder:
    """Stateless encoder for the ASCII batch.

    Use:
        bytes_to_send = C1BatchEncoder.encode(header, rows)
    """

    @staticmethod
    def encode(header: C1BatchHeader, rows: Sequence[C1CandidateRow]) -> bytes:
        if header.n_candidates != len(rows):
            raise ValueError(
                f"header.n_candidates={header.n_candidates} != len(rows)={len(rows)}"
            )
        parts: List[str] = []
        parts.append(
            "# C1 "
            f"{header.schema_version} "
            f"{header.cube_id} "
            f"{header.event_specnum_start} "
            f"{header.mjd_start:.11f} "
            f"{header.sample_period_specnum} "
            f"{header.sample_period_us:.6f} "
            f"{header.n_grid} "
            f"{header.n_fdm_in_cube} "
            f"{header.search_node_id} "
            f"{header.gpu_half} "
            f"{header.n_candidates}\n"
        )
        for r in rows:
            parts.append(
                f"{r.snr:.6e} "
                f"{r.l_rad:.9e} "
                f"{r.m_rad:.9e} "
                f"{r.l_pix:d} "
                f"{r.m_pix:d} "
                f"{r.dm_pc_cc:.6f} "
                f"{r.dm_idx_global:d} "
                f"{r.fine_dm_idx:d} "
                f"{r.event_specnum:d} "
                f"{r.width_samples:d} "
                f"{r.kernel_id} "
                f"{r.flags:d}\n"
            )
        parts.append("# END\n")
        return "".join(parts).encode("utf-8")


# ---------------------------------------------------------------------------
# Parser — used by the C2 receiver
# ---------------------------------------------------------------------------


def parse_c1_batch(lines: Iterable[str]) -> C1Batch:
    """Parse a single batch from an iterable of lines (no trailing newlines).

    Raises ``BadBatch`` if framing / schema is invalid.  Used by the
    receiver after a batch's worth of lines is collected.
    """
    it = iter(lines)
    try:
        header_line = next(it)
    except StopIteration:
        raise BadBatch("empty batch")
    header = _parse_header(header_line)
    rows: List[C1CandidateRow] = []
    for _ in range(header.n_candidates):
        try:
            row_line = next(it)
        except StopIteration:
            raise BadBatch(
                f"truncated batch: expected {header.n_candidates} rows, "
                f"got {len(rows)}"
            )
        if row_line.lstrip().startswith("#"):
            # A '#'-prefixed line in a row slot means the END marker
            # arrived before all promised rows did — treat as a torn
            # batch rather than a malformed row.
            raise BadBatch(
                f"truncated batch: expected {header.n_candidates} rows, "
                f"got {len(rows)} (saw '#' line {row_line!r})"
            )
        rows.append(_parse_row(row_line))
    try:
        end_line = next(it)
    except StopIteration:
        raise BadBatch("missing END marker")
    if end_line.strip() != "# END":
        raise BadBatch(f"expected '# END', got {end_line!r}")
    return C1Batch(header=header, candidates=tuple(rows))


def _parse_header(line: str) -> C1BatchHeader:
    s = line.strip()
    if not s.startswith("# C1 "):
        raise BadBatch(f"header must start with '# C1 '; got {line!r}")
    toks = s[len("# C1 "):].split()
    if len(toks) != 11:
        raise BadBatch(
            f"header expects 11 fields after '# C1 '; got {len(toks)}: {toks}"
        )
    try:
        schema_version = int(toks[0])
        cube_id = int(toks[1])
        event_specnum_start = int(toks[2])
        mjd_start = float(toks[3])
        sample_period_specnum = int(toks[4])
        sample_period_us = float(toks[5])
        n_grid = int(toks[6])
        n_fdm_in_cube = int(toks[7])
        search_node_id = int(toks[8])
        gpu_half = int(toks[9])
        n_candidates = int(toks[10])
    except ValueError as exc:
        raise BadBatch(f"header type error: {exc}; line={line!r}") from None
    if schema_version != SCHEMA_VERSION:
        raise BadBatch(
            f"unsupported schema_version={schema_version} "
            f"(this build accepts only {SCHEMA_VERSION})"
        )
    if n_candidates < 0:
        raise BadBatch(f"n_candidates={n_candidates} must be >= 0")
    if sample_period_specnum <= 0:
        raise BadBatch(
            f"sample_period_specnum={sample_period_specnum} must be > 0"
        )
    if sample_period_us <= 0.0:
        raise BadBatch(
            f"sample_period_us={sample_period_us} must be > 0"
        )
    return C1BatchHeader(
        schema_version=schema_version,
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=sample_period_specnum,
        sample_period_us=sample_period_us,
        n_grid=n_grid,
        n_fdm_in_cube=n_fdm_in_cube,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        n_candidates=n_candidates,
    )


def _parse_row(line: str) -> C1CandidateRow:
    toks = line.split()
    if len(toks) != 12:
        raise BadBatch(
            f"row expects 12 fields; got {len(toks)}: {line!r}"
        )
    try:
        return C1CandidateRow(
            snr=float(toks[0]),
            l_rad=float(toks[1]),
            m_rad=float(toks[2]),
            l_pix=int(toks[3]),
            m_pix=int(toks[4]),
            dm_pc_cc=float(toks[5]),
            dm_idx_global=int(toks[6]),
            fine_dm_idx=int(toks[7]),
            event_specnum=int(toks[8]),
            width_samples=int(toks[9]),
            kernel_id=toks[10],
            flags=int(toks[11]),
        )
    except ValueError as exc:
        raise BadBatch(f"row type error: {exc}; line={line!r}") from None


# ---------------------------------------------------------------------------
# C2 → C1 trigger packet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class C2TriggerPacket:
    event_name: str
    event_specnum: int
    mjd_target: float
    trigger_class_id: int = 0
    flags: int = C2_TRIGGER_FLAG_DUMP_CUBE


def encode_c2_trigger(pkt: C2TriggerPacket) -> bytes:
    if len(pkt.event_name) > 16:
        raise ValueError(
            f"event_name {pkt.event_name!r} > 16 bytes; truncate or change schema"
        )
    name_bytes = pkt.event_name.encode("ascii", errors="strict")
    if len(name_bytes) < 16:
        name_bytes = name_bytes + b"\x00" * (16 - len(name_bytes))
    return _TRIGGER_STRUCT.pack(
        C2_TRIGGER_MAGIC,
        SCHEMA_VERSION,
        int(pkt.trigger_class_id),
        name_bytes,
        int(pkt.event_specnum),
        float(pkt.mjd_target),
        int(pkt.flags),
        0,
        b"\x00" * 16,
    )


def decode_c2_trigger(buf: bytes) -> C2TriggerPacket:
    if len(buf) != C2_TRIGGER_PACKET_SIZE:
        raise BadBatch(
            f"trigger packet wrong size: {len(buf)} bytes "
            f"(expected {C2_TRIGGER_PACKET_SIZE})"
        )
    (
        magic,
        version,
        trigger_class_id,
        name_bytes,
        event_specnum,
        mjd_target,
        flags,
        _reserved,
        _pad,
    ) = _TRIGGER_STRUCT.unpack(buf)
    if magic != C2_TRIGGER_MAGIC:
        raise BadBatch(
            f"trigger magic mismatch: 0x{magic:08x} (expected 0x{C2_TRIGGER_MAGIC:08x})"
        )
    if version != SCHEMA_VERSION:
        raise BadBatch(
            f"trigger schema_version={version} unsupported (build accepts {SCHEMA_VERSION})"
        )
    event_name = name_bytes.rstrip(b"\x00").decode("ascii", errors="strict")
    return C2TriggerPacket(
        event_name=event_name,
        event_specnum=int(event_specnum),
        mjd_target=float(mjd_target),
        trigger_class_id=int(trigger_class_id),
        flags=int(flags),
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def build_header(
    *,
    cube_id: int,
    event_specnum_start: int,
    mjd_start: float,
    sample_period_specnum: int,
    sample_period_us: float,
    n_grid: int,
    n_fdm_in_cube: int,
    search_node_id: int,
    gpu_half: int,
    n_candidates: int,
) -> C1BatchHeader:
    """Convenience constructor that fills schema_version from this module."""
    return C1BatchHeader(
        schema_version=SCHEMA_VERSION,
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=sample_period_specnum,
        sample_period_us=sample_period_us,
        n_grid=n_grid,
        n_fdm_in_cube=n_fdm_in_cube,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        n_candidates=n_candidates,
    )
