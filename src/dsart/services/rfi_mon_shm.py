"""POSIX-shm ring buffer for RFI window records (M7.6 monitoring).

Bridges the corr_fast hot path (producer) and the rfi_monitor_export
sidecar (consumer / HTTP exporter) without coupling the two
processes' lifetimes. Same pattern as
:mod:`dsart.capture.mon_shm`: a single named POSIX-shm segment at
``/dev/shm/dsart-rfi-window-<cn_id>``, fixed-size header followed
by a small ring of fixed-size records.

ABI:
    Header (256 B aligned):
        uint32 magic         = ``RFI_MON_MAGIC``
        uint32 version       = ``RFI_MON_VERSION``
        uint32 cn_id
        uint32 n_slots
        uint32 n_ants
        uint32 n_chan_ds
        uint32 n_pol
        uint32 window_size
        uint32 freq_downsample
        uint32 record_size_bytes
        uint64 publish_seq     (1-based; reader uses this as the
                                'most recently fully-published'
                                marker; reads (publish_seq - 1) %
                                n_slots; retries if it changed mid-
                                read = seqlock pattern)
        uint64 startup_utc_ns
        uint64 _reserved[20]

    Record (record_size_bytes, page-aligned):
        Per-record header (128 B):
            uint64 seq                (matches publish_seq snapshot
                                       at the time the record was
                                       finalised)
            uint64 publish_utc_ns
            uint64 block_n_start
            uint64 block_n_end
            uint32 n_cubes
            uint32 n_cubes_warmup
            float32 scalars[24]      (8 metric groups x 3 (pol0,
                                      pol1, both); see _SCALAR_KEYS)
            uint64 _reserved[2]

        Body (concatenated, all little-endian):
            float32 s1_full_mean[n_ants * n_chan_ds * n_pol]
            uint8   mask_count_final[n_ants * n_chan_ds * n_pol]
            uint8   mask_count_sk
            uint8   mask_count_bp
            uint8   mask_count_grp
            uint8   mask_count_sumthr
            uint8   mask_count_fa

At the production op-point (NANTS=96, NCHAN_DS=96, NPOL=2):
    s1_bytes = 96 * 96 * 2 * 4 = 73,728
    mask_bytes = 6 * 96 * 96 * 2 * 1 = 110,592
    body = 184,320 B
    header = 128 B
    record (page-aligned to 4096) = 188,416 B
    n_slots = 64  ->  segment = 64 * 188416 + 256 = ~11.5 MiB

64 slots × 2.147 s/win ≈ 137 s of history on the corr side. The h23
dashboard polls fast enough that it never sees more than ~5 s of
unread windows; the 137 s buffer is the safety margin against
restarts and transient pull stalls.

Producer (RFIMonShmWriter):
    Stages a record into a slot fully, increments ``publish_seq``,
    moves on. Single-writer assumption (one corr_fast_integration
    per shm segment).

Consumer (RFIMonShmReader):
    Reads ``publish_seq``, decodes the record at
    ``(publish_seq - 1) % n_slots``, validates the per-record
    ``seq`` matches, retries up to ``_READ_RETRIES`` if torn.
    Supports waterfall reads of the most recent N records
    (``read_recent(n)``) for the sidecar's /api/windows endpoint.
"""

from __future__ import annotations

import dataclasses
import errno
import mmap
import os
import struct
import time
from typing import Final, Optional

import numpy as np

from dsart.services.rfi_window import RFIWindow

# Bump when the layout changes.
_RFI_MON_MAGIC: Final[int] = 0xCAFE5F11
_RFI_MON_VERSION: Final[int] = 1

# Per-segment header.
# Format: <I I I I I I I I I I Q Q + 20*Q  (= 12*uint then 22*uint64)
# Equivalent struct.calcsize("<10I" + "QQ" + "20Q") = 40 + 16 + 160 = 216 B.
# Pad to 256 to align records to 256-byte boundaries.
_HEADER_STRUCT_FMT: Final[str] = "<10I" + "QQ" + "20Q"
_HEADER_BYTES: Final[int] = 256
_HEADER_BYTES_USED: Final[int] = struct.calcsize(_HEADER_STRUCT_FMT)
assert _HEADER_BYTES_USED <= _HEADER_BYTES, (
    f"header overflow {_HEADER_BYTES_USED} > {_HEADER_BYTES}"
)

# Per-record header. Format: seq Q + publish_utc_ns Q + bn_start Q +
# bn_end Q + n_cubes I + n_cubes_warmup I + 24*f + 2*Q  = 8+8+8+8+4+4+96+16 = 152
_RECORD_HDR_FMT: Final[str] = "<QQQQII" + "24f" + "2Q"
_RECORD_HDR_BYTES_USED: Final[int] = struct.calcsize(_RECORD_HDR_FMT)
# Round to 128 (so record body always page-aligned offset within record).
_RECORD_HDR_BYTES: Final[int] = 256
assert _RECORD_HDR_BYTES_USED <= _RECORD_HDR_BYTES, (
    f"record header overflow {_RECORD_HDR_BYTES_USED} > {_RECORD_HDR_BYTES}"
)

# Scalar metric layout: 8 groups × 3 components (pol0, pol1, both).
# Must match RFIWindow exactly. Used by both writer and reader.
_SCALAR_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("total_flag_fraction", "tot"),
    ("bandpass_channel_fraction", "bpc"),
    ("ant_fraction_flagged", "ant"),
    ("frac_sk", "sk"),
    ("frac_bp", "bp"),
    ("frac_grp", "grp"),
    ("frac_sumthr", "st"),
    ("frac_fa", "fa"),
)
assert len(_SCALAR_KEYS) * 3 == 24, "scalar count mismatch with format string"

# Page size for record alignment.
_PAGE_BYTES: Final[int] = 4096

# Seqlock retries on the read side. Each retry costs one record-sized
# read; 8 is plenty since the writer holds publish_seq advanced for
# at most a few microseconds per window finalise.
_READ_RETRIES: Final[int] = 8

_SHM_DIR: Final[str] = "/dev/shm"


def shm_name(cn_id: int) -> str:
    """Return the POSIX shm_open name (without leading slash) for cn_id."""
    return f"dsart-rfi-window-{cn_id}"


def shm_path(cn_id: int) -> str:
    """Return the /dev/shm path for cn_id."""
    return os.path.join(_SHM_DIR, shm_name(cn_id))


# ---------------------------------------------------------------------------
# Layout maths
# ---------------------------------------------------------------------------


def _body_bytes(n_ants: int, n_chan_ds: int, n_pol: int) -> int:
    cells = n_ants * n_chan_ds * n_pol
    return cells * 4 + 6 * cells  # 1 fp32 array + 6 uint8 arrays


def _record_bytes(n_ants: int, n_chan_ds: int, n_pol: int) -> int:
    body = _body_bytes(n_ants, n_chan_ds, n_pol)
    raw = _RECORD_HDR_BYTES + body
    # Round up to PAGE_BYTES for alignment.
    return ((raw + _PAGE_BYTES - 1) // _PAGE_BYTES) * _PAGE_BYTES


def segment_bytes(
    *, n_ants: int, n_chan_ds: int, n_pol: int, n_slots: int,
) -> int:
    return _HEADER_BYTES + n_slots * _record_bytes(n_ants, n_chan_ds, n_pol)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RFIMonShmNotPresent(RuntimeError):
    pass


class RFIMonShmAbiMismatch(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Snapshot dataclass (consumer view)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RFIMonRecord:
    """A reader's decoded view of one window record."""

    seq: int
    publish_utc_ns: int
    block_n_start: int
    block_n_end: int
    n_cubes: int
    n_cubes_warmup: int
    # Per-metric, per-pol scalars: shape (8 groups, 3 (pol0/pol1/both)).
    scalars: dict[str, tuple[float, float, float]]
    s1_full_mean: np.ndarray                       # (NANTS, NCHAN_DS, NPOL) fp32
    mask_count_final: np.ndarray                   # (NANTS, NCHAN_DS, NPOL) uint8
    mask_count_sk: np.ndarray
    mask_count_bp: np.ndarray
    mask_count_grp: np.ndarray
    mask_count_sumthr: np.ndarray
    mask_count_fa: np.ndarray


# ---------------------------------------------------------------------------
# Writer (corr_fast hot-path side)
# ---------------------------------------------------------------------------


class RFIMonShmWriter:
    """Producer-side handle to the RFI window POSIX-shm ring.

    Creates (truncates) the segment on construction. Subsequent
    :meth:`publish` calls finalise a window record into the next
    slot and bump ``publish_seq``.

    Single-writer; not thread-safe (but the corr_fast hot path is
    serialised anyway).
    """

    def __init__(
        self,
        *,
        cn_id: int,
        n_ants: int,
        n_chan_ds: int,
        n_pol: int,
        window_size: int,
        freq_downsample: int,
        n_slots: int = 64,
    ) -> None:
        if n_slots <= 0:
            raise ValueError(f"n_slots={n_slots}, expected > 0")
        self._cn_id = int(cn_id)
        self._n_ants = int(n_ants)
        self._n_chan_ds = int(n_chan_ds)
        self._n_pol = int(n_pol)
        self._window_size = int(window_size)
        self._freq_downsample = int(freq_downsample)
        self._n_slots = int(n_slots)

        self._record_bytes = _record_bytes(n_ants, n_chan_ds, n_pol)
        self._segment_bytes = segment_bytes(
            n_ants=n_ants, n_chan_ds=n_chan_ds, n_pol=n_pol, n_slots=n_slots,
        )
        self._cells = n_ants * n_chan_ds * n_pol
        self._s1_bytes = self._cells * 4
        self._mask_bytes = self._cells              # one uint8 per cell
        self._body_off = _RECORD_HDR_BYTES

        path = shm_path(cn_id)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, self._segment_bytes)
            self._mm = mmap.mmap(
                fd, self._segment_bytes,
                mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)

        self._publish_seq: int = 0
        self._write_header(startup_utc_ns=time.time_ns())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def record_bytes(self) -> int:
        return self._record_bytes

    @property
    def segment_bytes(self) -> int:
        return self._segment_bytes

    @property
    def publish_seq(self) -> int:
        return self._publish_seq

    def publish(self, window: RFIWindow) -> int:
        """Write `window` into the next ring slot and bump publish_seq.

        Returns the new ``publish_seq`` value (i.e. the seq stamp of
        the just-published record). The consumer reads the slot
        at ``(publish_seq - 1) % n_slots``.
        """
        if window.s1_full_mean.shape != (self._n_ants, self._n_chan_ds, self._n_pol):
            raise ValueError(
                f"window.s1_full_mean shape {window.s1_full_mean.shape} "
                f"!= expected ({self._n_ants}, {self._n_chan_ds}, "
                f"{self._n_pol})"
            )

        new_seq = self._publish_seq + 1
        slot = (new_seq - 1) % self._n_slots
        off = _HEADER_BYTES + slot * self._record_bytes

        # ----- Stage record header --------------------------------
        # Scalar metric block (24 fp32 in spec order).
        scalars: list[float] = []
        for attr_name, _short in _SCALAR_KEYS:
            triplet = getattr(window, attr_name)
            scalars.extend([float(triplet[0]), float(triplet[1]),
                            float(triplet[2])])
        hdr_packed = struct.pack(
            _RECORD_HDR_FMT,
            new_seq,
            time.time_ns(),
            int(window.block_n_start),
            int(window.block_n_end),
            int(window.n_cubes),
            int(window.n_cubes_warmup),
            *scalars,
            0, 0,                                  # reserved
        )
        # Pad to _RECORD_HDR_BYTES.
        hdr_packed = hdr_packed.ljust(_RECORD_HDR_BYTES, b"\x00")
        self._mm[off : off + _RECORD_HDR_BYTES] = hdr_packed

        # ----- Stage record body -----------------------------------
        body_off = off + self._body_off
        # fp32 s1
        s1 = np.ascontiguousarray(window.s1_full_mean, dtype=np.float32)
        self._mm[body_off : body_off + self._s1_bytes] = s1.tobytes()
        body_off += self._s1_bytes
        # six uint8 masks
        for arr_name in (
            "mask_count_final", "mask_count_sk", "mask_count_bp",
            "mask_count_grp", "mask_count_sumthr", "mask_count_fa",
        ):
            arr = getattr(window, arr_name)
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
            self._mm[body_off : body_off + self._mask_bytes] = arr.tobytes()
            body_off += self._mask_bytes

        # ----- Publish atomically ----------------------------------
        # Writing publish_seq is the publishing edge. mmap word writes
        # are atomic on x86-64 (8-byte aligned). Use struct.pack into
        # the publish_seq field offset.
        publish_seq_off = struct.calcsize("<10I")  # immediately after the 10 uint32 header fields
        self._mm[publish_seq_off : publish_seq_off + 8] = struct.pack(
            "<Q", new_seq,
        )

        self._publish_seq = new_seq
        return new_seq

    def close(self) -> None:
        try:
            self._mm.flush()
        except (OSError, ValueError):
            pass
        try:
            self._mm.close()
        except (OSError, ValueError):
            pass

    def unlink(self) -> None:
        """Remove the segment from the filesystem. Safe even if the
        backing file is gone (idempotent)."""
        try:
            os.unlink(shm_path(self._cn_id))
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_header(self, *, startup_utc_ns: int) -> None:
        """Lay down the segment header. publish_seq starts at 0."""
        packed = struct.pack(
            _HEADER_STRUCT_FMT,
            _RFI_MON_MAGIC,
            _RFI_MON_VERSION,
            int(self._cn_id),
            int(self._n_slots),
            int(self._n_ants),
            int(self._n_chan_ds),
            int(self._n_pol),
            int(self._window_size),
            int(self._freq_downsample),
            int(self._record_bytes),
            0,                                     # publish_seq
            int(startup_utc_ns),
            *([0] * 20),                           # reserved
        )
        self._mm[0 : len(packed)] = packed
        # Zero the rest of the header window (in case we ever shorten it).
        if len(packed) < _HEADER_BYTES:
            self._mm[len(packed) : _HEADER_BYTES] = b"\x00" * (
                _HEADER_BYTES - len(packed)
            )


# ---------------------------------------------------------------------------
# Reader (sidecar / monitor side)
# ---------------------------------------------------------------------------


class RFIMonShmReader:
    """Consumer-side handle. Read-only mmap of the same segment."""

    def __init__(self, cn_id: int) -> None:
        self._cn_id = int(cn_id)
        path = shm_path(cn_id)
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise RFIMonShmNotPresent(
                    f"shm not present: {path}"
                ) from e
            raise
        try:
            st = os.fstat(fd)
            size = st.st_size
            self._mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        self._parse_header()

    # ------------------------------------------------------------------
    # Header layout queries
    # ------------------------------------------------------------------

    def _parse_header(self) -> None:
        raw = bytes(self._mm[0:_HEADER_BYTES_USED])
        fields = struct.unpack(_HEADER_STRUCT_FMT, raw)
        (magic, version, cn_id, n_slots, n_ants, n_chan_ds, n_pol,
         window_size, freq_downsample, record_bytes,
         publish_seq, startup_utc_ns, *_reserved) = fields
        if magic != _RFI_MON_MAGIC:
            raise RFIMonShmAbiMismatch(
                f"bad magic 0x{magic:08x} (expected 0x{_RFI_MON_MAGIC:08x})"
            )
        if version != _RFI_MON_VERSION:
            raise RFIMonShmAbiMismatch(
                f"unsupported version {version} (we know {_RFI_MON_VERSION})"
            )
        self._n_slots = int(n_slots)
        self._n_ants = int(n_ants)
        self._n_chan_ds = int(n_chan_ds)
        self._n_pol = int(n_pol)
        self._window_size = int(window_size)
        self._freq_downsample = int(freq_downsample)
        self._record_bytes = int(record_bytes)
        self._startup_utc_ns = int(startup_utc_ns)
        self._publish_seq_off = struct.calcsize("<10I")

        self._cells = self._n_ants * self._n_chan_ds * self._n_pol
        self._s1_bytes = self._cells * 4
        self._mask_bytes = self._cells

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def n_ants(self) -> int:
        return self._n_ants

    @property
    def n_chan_ds(self) -> int:
        return self._n_chan_ds

    @property
    def n_pol(self) -> int:
        return self._n_pol

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def freq_downsample(self) -> int:
        return self._freq_downsample

    @property
    def startup_utc_ns(self) -> int:
        return self._startup_utc_ns

    def read_publish_seq(self) -> int:
        """Atomic read of the current publish_seq (= number of windows
        published since shm was created)."""
        raw = bytes(self._mm[self._publish_seq_off : self._publish_seq_off + 8])
        (val,) = struct.unpack("<Q", raw)
        return val

    # ------------------------------------------------------------------
    # Record decode
    # ------------------------------------------------------------------

    def _decode_record(self, slot: int) -> RFIMonRecord:
        """Decode a single slot. Does NOT do torn-write detection;
        callers wrap this in a seqlock loop. Raises if the seq within
        the record disagrees with `slot` math."""
        off = _HEADER_BYTES + slot * self._record_bytes
        hdr_raw = bytes(self._mm[off : off + _RECORD_HDR_BYTES_USED])
        fields = struct.unpack(_RECORD_HDR_FMT, hdr_raw)
        seq = int(fields[0])
        publish_utc_ns = int(fields[1])
        block_n_start = int(fields[2])
        block_n_end = int(fields[3])
        n_cubes = int(fields[4])
        n_cubes_warmup = int(fields[5])
        scalars_flat = fields[6 : 6 + 24]

        scalars: dict[str, tuple[float, float, float]] = {}
        for i, (attr_name, _short) in enumerate(_SCALAR_KEYS):
            base = 3 * i
            scalars[attr_name] = (
                float(scalars_flat[base + 0]),
                float(scalars_flat[base + 1]),
                float(scalars_flat[base + 2]),
            )

        body_off = off + _RECORD_HDR_BYTES
        s1 = np.frombuffer(
            self._mm, dtype=np.float32, count=self._cells, offset=body_off,
        ).copy().reshape(self._n_ants, self._n_chan_ds, self._n_pol)
        body_off += self._s1_bytes

        def _read_u8(name_off: int) -> np.ndarray:
            return np.frombuffer(
                self._mm, dtype=np.uint8, count=self._cells, offset=name_off,
            ).copy().reshape(self._n_ants, self._n_chan_ds, self._n_pol)

        m_final = _read_u8(body_off); body_off += self._mask_bytes
        m_sk    = _read_u8(body_off); body_off += self._mask_bytes
        m_bp    = _read_u8(body_off); body_off += self._mask_bytes
        m_grp   = _read_u8(body_off); body_off += self._mask_bytes
        m_st    = _read_u8(body_off); body_off += self._mask_bytes
        m_fa    = _read_u8(body_off); body_off += self._mask_bytes

        return RFIMonRecord(
            seq=seq,
            publish_utc_ns=publish_utc_ns,
            block_n_start=block_n_start,
            block_n_end=block_n_end,
            n_cubes=n_cubes,
            n_cubes_warmup=n_cubes_warmup,
            scalars=scalars,
            s1_full_mean=s1,
            mask_count_final=m_final,
            mask_count_sk=m_sk,
            mask_count_bp=m_bp,
            mask_count_grp=m_grp,
            mask_count_sumthr=m_st,
            mask_count_fa=m_fa,
        )

    def read_latest(self) -> Optional[RFIMonRecord]:
        """Return the most recently published window record, or
        ``None`` if no record has been published yet. Performs
        seqlock-style torn-write detection; retries up to
        ``_READ_RETRIES`` times."""
        for _ in range(_READ_RETRIES):
            seq0 = self.read_publish_seq()
            if seq0 == 0:
                return None
            slot = (seq0 - 1) % self._n_slots
            rec = self._decode_record(slot)
            if rec.seq != seq0:
                # The producer overwrote this slot mid-read; retry.
                continue
            seq1 = self.read_publish_seq()
            if seq1 == seq0:
                return rec
            # Else publish_seq advanced -- our rec is the (seq0-th)
            # one, which IS valid (just no longer the latest). Return
            # it anyway and let the next call grab the newer one.
            return rec
        return None

    def read_recent(self, n: int) -> list[RFIMonRecord]:
        """Return up to ``n`` most-recent records, oldest first.
        Records that have already been overwritten in the ring
        (i.e. ``seq + n_slots <= publish_seq``) are skipped."""
        if n <= 0:
            return []
        seq = self.read_publish_seq()
        if seq == 0:
            return []
        start = max(1, seq - n + 1)
        out: list[RFIMonRecord] = []
        for s in range(start, seq + 1):
            slot = (s - 1) % self._n_slots
            for _ in range(_READ_RETRIES):
                rec = self._decode_record(slot)
                if rec.seq == s:
                    out.append(rec)
                    break
                # Slot has been overwritten by a newer record while we
                # were reading; that's allowed for old records --
                # break and skip rather than spin forever.
                if rec.seq > s:
                    break
        return out

    def close(self) -> None:
        try:
            self._mm.close()
        except (OSError, ValueError):
            pass
