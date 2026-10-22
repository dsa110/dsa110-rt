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
    # x2 cutout packets (SPEC §9)
    "CUTOUT_REQUEST_MAGIC",
    "CUTOUT_REQUEST_SIZE",
    "CUTOUT_STATUS_MAGIC",
    "CUTOUT_STATUS_SIZE",
    "CUTOUT_STATUS_PORT_DEFAULT",
    "CutoutRequest",
    "CutoutStatus",
    "encode_cutout_request",
    "decode_cutout_request",
    "encode_cutout_status",
    "decode_cutout_status",
    "packet_kind",
    "class_id_for",
    "default_cutout_window",
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
    #: Matched-filter boxcar width in SEARCH samples (1048.576 µs each
    #: at the production op-point) — the detector's own boxcar index,
    #: ``Candidate.width_samples`` from ``detector/decoder.py``.
    #:
    #: CROSS-READING HAZARD: the injector's ``width_samples``
    #: (``inject/online.py``, ``InjectionRequest``) is a FWHM in NATIVE
    #: samples of 32.768 µs, so the two differ by a factor of 32 at the
    #: production op-point — a shot fired at injector w=4 is reported
    #: here at w≈2, not w=4. Never compare the two numbers directly;
    #: convert through the sample periods.
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
# x2 cutout request / status (docs/x2/SPEC.md §9, downstream_inventory §3.2-3.3)
# ---------------------------------------------------------------------------
#
# Two NEW packet types, deliberately distinct from the 64-byte DSRT trigger
# above: different magic AND size, so a 1x search listener (which dumps a
# cube on ANY 64-byte DSRT packet -- it never reads ``flags``) and the corr
# voltage listener drop them on the size check, and the DSRT schema stays
# at version 1. Frozen in SPEC §9 v1; changing a field needs integration.

#: ``b"DSCQ"`` read as a little-endian u32.
CUTOUT_REQUEST_MAGIC: int = 0x51435344
CUTOUT_REQUEST_VERSION: int = 1
CUTOUT_REQUEST_SIZE: int = 128

#: ``b"DSCS"`` read as a little-endian u32.
CUTOUT_STATUS_MAGIC: int = 0x53435344
CUTOUT_STATUS_VERSION: int = 1
CUTOUT_STATUS_SIZE: int = 64

#: Default UDP port C2 binds for half -> C2 cutout status datagrams.
CUTOUT_STATUS_PORT_DEFAULT: int = 11501

# Request flags.
CUTOUT_FLAG_CUTOUT: int = 1 << 0
CUTOUT_FLAG_RAW_ARCHIVE: int = 1 << 1
CUTOUT_FLAG_INJECTION: int = 1 << 2
CUTOUT_FLAG_MANUAL: int = 1 << 3
CUTOUT_FLAG_RESEND: int = 1 << 4
_CUTOUT_FLAGS_ALL: int = (1 << 5) - 1

# Status states.
CUTOUT_STATE_ACCEPTED: int = 1
CUTOUT_STATE_DONE: int = 2
CUTOUT_STATE_PARTIAL: int = 3
CUTOUT_STATE_FAILED: int = 4
CUTOUT_STATE_NAMES = {
    CUTOUT_STATE_ACCEPTED: "accepted",
    CUTOUT_STATE_DONE: "done",
    CUTOUT_STATE_PARTIAL: "partial",
    CUTOUT_STATE_FAILED: "failed",
}

# Status reasons.
CUTOUT_REASON_OK: int = 0
CUTOUT_REASON_TOO_LATE: int = 1
CUTOUT_REASON_WINDOW_TIMEOUT: int = 2
CUTOUT_REASON_QUEUE_FULL: int = 3
CUTOUT_REASON_PLAN_MISMATCH: int = 4
CUTOUT_REASON_BAD_REQUEST: int = 5
CUTOUT_REASON_GPU_ERROR: int = 6
CUTOUT_REASON_IO_ERROR: int = 7
CUTOUT_REASON_UPLOAD_FAILED: int = 8
CUTOUT_REASON_SHUTTING_DOWN: int = 9
CUTOUT_REASON_NAMES = {
    CUTOUT_REASON_OK: "ok",
    CUTOUT_REASON_TOO_LATE: "too_late",
    CUTOUT_REASON_WINDOW_TIMEOUT: "window_timeout",
    CUTOUT_REASON_QUEUE_FULL: "queue_full",
    CUTOUT_REASON_PLAN_MISMATCH: "plan_mismatch",
    CUTOUT_REASON_BAD_REQUEST: "bad_request",
    CUTOUT_REASON_GPU_ERROR: "gpu_error",
    CUTOUT_REASON_IO_ERROR: "io_error",
    CUTOUT_REASON_UPLOAD_FAILED: "upload_failed",
    CUTOUT_REASON_SHUTTING_DOWN: "shutting_down",
}

_CUTOUT_REQUEST_STRUCT = struct.Struct("<IHH16sIHHqqIIIHHffffihhhHfdQH14s")
_CUTOUT_STATUS_STRUCT = struct.Struct("<IHBB16sIBBBBfIQIh10s")
assert _CUTOUT_REQUEST_STRUCT.size == CUTOUT_REQUEST_SIZE
assert _CUTOUT_STATUS_STRUCT.size == CUTOUT_STATUS_SIZE


@dataclass(frozen=True, slots=True)
class CutoutRequest:
    """C2 -> every search half: re-image + archive around a candidate.

    Units (SPEC §8/§9): every ``*_ext`` value counts the op-point's external
    sample (524.288 us at 2x, 1048.576 us at 1x) on the absolute native
    epoch (``t_ext = native // native_per_ext``). ``t0_ext`` and ``n_t``
    are even (tier-B samples cover two rows). A MANUAL (Dump Now) request
    has no candidate: ``l_pix = m_pix = -1`` and ``dm_peak_pc_cc < 0``.
    """

    event_name: str
    request_id: int
    t_peak_ext: int
    t0_ext: int
    n_t: int = 512
    raw_pre_ext: int = 954
    raw_post_ext: int = 954
    width_ext: int = 1
    dm_peak_pc_cc: float = -1.0
    snr_peak: float = 0.0
    l_rad: float = 0.0
    m_rad: float = 0.0
    l_pix: int = -1
    m_pix: int = -1
    fine_global_peak: int = -1
    coarse_global_peak: int = -1
    stamp_half_px: int = 16
    n_stamp_t: int = 64
    stamp_dm_halfwidth_pc_cc: float = 150.0
    mjd_peak: float = 0.0
    plan_id: int = 0
    deadline_s: int = 15
    trigger_class_id: int = 0
    flags: int = CUTOUT_FLAG_CUTOUT

    @property
    def is_manual(self) -> bool:
        return bool(self.flags & CUTOUT_FLAG_MANUAL)

    @property
    def wants_raw(self) -> bool:
        return bool(self.flags & CUTOUT_FLAG_RAW_ARCHIVE)

    @property
    def has_pixel(self) -> bool:
        return self.l_pix >= 0 and self.m_pix >= 0

    @property
    def key(self) -> Tuple[str, int]:
        """Idempotency key: a resend carries the same key."""
        return (self.event_name, int(self.request_id))

    def validate(self) -> None:
        """Raise ``BadBatch`` if the request is not internally consistent."""
        _check_event_name(self.event_name)
        if self.n_t <= 0 or self.n_t % 2:
            raise BadBatch(f"n_t={self.n_t} must be even and > 0")
        if self.t0_ext % 2:
            raise BadBatch(f"t0_ext={self.t0_ext} must be even")
        if not (self.t0_ext <= self.t_peak_ext < self.t0_ext + self.n_t):
            raise BadBatch(
                f"t_peak_ext={self.t_peak_ext} outside "
                f"[{self.t0_ext}, {self.t0_ext + self.n_t})")
        if self.flags & ~_CUTOUT_FLAGS_ALL:
            raise BadBatch(f"unknown flag bits in 0x{self.flags:04x}")
        if (self.l_pix < 0) != (self.m_pix < 0):
            raise BadBatch("l_pix/m_pix must both be set or both be -1")
        if self.width_ext < 0 or self.stamp_half_px < 0 or self.n_stamp_t < 0:
            raise BadBatch("negative width / stamp size")


@dataclass(frozen=True, slots=True)
class CutoutStatus:
    """Search half -> C2: lossy progress report for one request."""

    event_name: str
    request_id: int
    state: int
    search_node_id: int
    gpu_half: int
    reason: int = CUTOUT_REASON_OK
    n_dm_owned: int = 0
    n_dm_ok: int = 0
    valid_frac: float = 1.0
    cutout_bytes: int = 0
    raw_bytes: int = 0
    latency_ms: int = 0
    upload_rc: int = -1

    @property
    def state_name(self) -> str:
        return CUTOUT_STATE_NAMES.get(int(self.state), f"state{self.state}")

    @property
    def reason_name(self) -> str:
        return CUTOUT_REASON_NAMES.get(int(self.reason), f"reason{self.reason}")

    @property
    def is_terminal(self) -> bool:
        return int(self.state) in (
            CUTOUT_STATE_DONE, CUTOUT_STATE_PARTIAL, CUTOUT_STATE_FAILED)


def _check_event_name(name: str) -> bytes:
    if not name:
        raise BadBatch("empty event_name")
    try:
        raw = name.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        raise BadBatch(f"event_name {name!r} is not ASCII") from None
    if len(raw) > 16:
        raise BadBatch(f"event_name {name!r} > 16 bytes")
    return raw


def _pad16(raw: bytes) -> bytes:
    return raw + b"\x00" * (16 - len(raw))


def class_id_for(name: str) -> int:
    """Stable 16-bit trigger-class id (crc32), unlike ``hash(name)``, which
    Python salts per process (PYTHONHASHSEED) so it changes on restart."""
    import zlib

    return zlib.crc32(str(name).encode("utf-8")) & 0xFFFF


def encode_cutout_request(req: CutoutRequest) -> bytes:
    req.validate()
    return _CUTOUT_REQUEST_STRUCT.pack(
        CUTOUT_REQUEST_MAGIC,
        CUTOUT_REQUEST_VERSION,
        int(req.flags),
        _pad16(_check_event_name(req.event_name)),
        int(req.request_id) & 0xFFFFFFFF,
        int(req.trigger_class_id) & 0xFFFF,
        0,
        int(req.t_peak_ext),
        int(req.t0_ext),
        int(req.n_t),
        int(req.raw_pre_ext),
        int(req.raw_post_ext),
        int(req.width_ext),
        int(req.stamp_half_px),
        float(req.dm_peak_pc_cc),
        float(req.snr_peak),
        float(req.l_rad),
        float(req.m_rad),
        int(req.fine_global_peak),
        int(req.coarse_global_peak),
        int(req.l_pix),
        int(req.m_pix),
        int(req.n_stamp_t),
        float(req.stamp_dm_halfwidth_pc_cc),
        float(req.mjd_peak),
        int(req.plan_id) & 0xFFFFFFFFFFFFFFFF,
        int(req.deadline_s),
        b"\x00" * 14,
    )


def decode_cutout_request(buf: bytes) -> CutoutRequest:
    if len(buf) != CUTOUT_REQUEST_SIZE:
        raise BadBatch(
            f"cutout request wrong size: {len(buf)} bytes "
            f"(expected {CUTOUT_REQUEST_SIZE})")
    (magic, version, flags, name_b, request_id, class_id, _r0, t_peak,
     t0, n_t, raw_pre, raw_post, width, stamp_half, dm, snr, l_rad, m_rad,
     fine_g, coarse_g, l_pix, m_pix, n_stamp_t, stamp_dm_hw, mjd_peak,
     plan_id, deadline_s, _r1) = _CUTOUT_REQUEST_STRUCT.unpack(buf)
    if magic != CUTOUT_REQUEST_MAGIC:
        raise BadBatch(f"cutout request magic mismatch: 0x{magic:08x}")
    if version != CUTOUT_REQUEST_VERSION:
        raise BadBatch(f"cutout request version={version} unsupported")
    try:
        name = name_b.rstrip(b"\x00").decode("ascii", errors="strict")
    except UnicodeDecodeError:
        raise BadBatch("cutout request event_name is not ASCII") from None
    req = CutoutRequest(
        event_name=name, request_id=int(request_id), t_peak_ext=int(t_peak),
        t0_ext=int(t0), n_t=int(n_t), raw_pre_ext=int(raw_pre),
        raw_post_ext=int(raw_post), width_ext=int(width),
        dm_peak_pc_cc=float(dm), snr_peak=float(snr), l_rad=float(l_rad),
        m_rad=float(m_rad), l_pix=int(l_pix), m_pix=int(m_pix),
        fine_global_peak=int(fine_g), coarse_global_peak=int(coarse_g),
        stamp_half_px=int(stamp_half), n_stamp_t=int(n_stamp_t),
        stamp_dm_halfwidth_pc_cc=float(stamp_dm_hw), mjd_peak=float(mjd_peak),
        plan_id=int(plan_id), deadline_s=int(deadline_s),
        trigger_class_id=int(class_id), flags=int(flags),
    )
    req.validate()
    return req


def encode_cutout_status(st: CutoutStatus) -> bytes:
    return _CUTOUT_STATUS_STRUCT.pack(
        CUTOUT_STATUS_MAGIC,
        CUTOUT_STATUS_VERSION,
        int(st.state) & 0xFF,
        int(st.reason) & 0xFF,
        _pad16(_check_event_name(st.event_name)),
        int(st.request_id) & 0xFFFFFFFF,
        int(st.search_node_id) & 0xFF,
        int(st.gpu_half) & 0xFF,
        min(int(st.n_dm_owned), 255),
        min(int(st.n_dm_ok), 255),
        float(st.valid_frac),
        min(int(st.cutout_bytes), 0xFFFFFFFF),
        int(st.raw_bytes) & 0xFFFFFFFFFFFFFFFF,
        min(int(st.latency_ms), 0xFFFFFFFF),
        max(min(int(st.upload_rc), 32767), -32768),
        b"\x00" * 10,
    )


def decode_cutout_status(buf: bytes) -> CutoutStatus:
    if len(buf) != CUTOUT_STATUS_SIZE:
        raise BadBatch(
            f"cutout status wrong size: {len(buf)} bytes "
            f"(expected {CUTOUT_STATUS_SIZE})")
    (magic, version, state, reason, name_b, request_id, sid, g, n_owned,
     n_ok, valid_frac, cutout_bytes, raw_bytes, latency_ms, upload_rc,
     _r) = _CUTOUT_STATUS_STRUCT.unpack(buf)
    if magic != CUTOUT_STATUS_MAGIC:
        raise BadBatch(f"cutout status magic mismatch: 0x{magic:08x}")
    if version != CUTOUT_STATUS_VERSION:
        raise BadBatch(f"cutout status version={version} unsupported")
    if state not in CUTOUT_STATE_NAMES:
        raise BadBatch(f"cutout status state={state} unknown")
    try:
        name = name_b.rstrip(b"\x00").decode("ascii", errors="strict")
    except UnicodeDecodeError:
        raise BadBatch("cutout status event_name is not ASCII") from None
    _check_event_name(name)
    return CutoutStatus(
        event_name=name, request_id=int(request_id), state=int(state),
        search_node_id=int(sid), gpu_half=int(g), reason=int(reason),
        n_dm_owned=int(n_owned), n_dm_ok=int(n_ok),
        valid_frac=float(valid_frac), cutout_bytes=int(cutout_bytes),
        raw_bytes=int(raw_bytes), latency_ms=int(latency_ms),
        upload_rc=int(upload_rc),
    )


def packet_kind(buf: bytes) -> str:
    """Classify a datagram on a shared port by (size, magic) without
    raising: ``"dsrt_trigger"``, ``"cutout_request"``, ``"cutout_status"``
    or ``"unknown"``."""
    if len(buf) < 4:
        return "unknown"
    magic = int.from_bytes(buf[:4], "little")
    if len(buf) == C2_TRIGGER_PACKET_SIZE and magic == C2_TRIGGER_MAGIC:
        return "dsrt_trigger"
    if len(buf) == CUTOUT_REQUEST_SIZE and magic == CUTOUT_REQUEST_MAGIC:
        return "cutout_request"
    if len(buf) == CUTOUT_STATUS_SIZE and magic == CUTOUT_STATUS_MAGIC:
        return "cutout_status"
    return "unknown"


def default_cutout_window(
    t_peak_ext: int, n_t: int = 512,
) -> Tuple[int, int]:
    """``(t0_ext, n_t)``: ``n_t`` rows (rounded up to even) with the peak
    in the middle, ``t0_ext`` rounded DOWN to even (tier-B alignment)."""
    n = int(n_t) + (int(n_t) % 2)
    t0 = int(t_peak_ext) - n // 2
    t0 -= t0 % 2
    return t0, n


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
