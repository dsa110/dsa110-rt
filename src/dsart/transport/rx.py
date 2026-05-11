"""``TransportRx`` — fast-vis cube receive (M3 chunk 8 + M4a chunks 3/4/5).

Loopback / unicast UDP receiver for fast-vis cubes; the chunk-8 peer
to :class:`dsart.transport.tx.TransportTx`. Validates magic + CRC,
tracks per-``chgroup`` sequence gaps for drop accounting (mirrors plan
§4.3 ``rx_seq_gap_count`` mon-key), and optionally captures payloads
to disk for the loopback-bench `.cfp16` set.

M4a chunk 3 adds the **production prod-frame path** alongside the chunk-8
path. Pass ``use_prod_frame=True`` to activate it. The prod-frame path
implements:

- Per-(corr, dm_idx) reorder window (depth W configurable via
  ``TransportRxProdConfig.reorder_window_depth``; default W=4 per plan
  §4.3 line 1473).
- Fragment bitmap reassembly: each incoming fragment updates a per-seq
  bitmap; when all ``n_frags`` bits are set the payload is complete.
- Per-datagram ``pattern_id`` verify: mismatch → drop + zero-fill +
  ``pattern_mismatch_count++`` (plan §3 line 308).
- Dequantisation at COO-store time (plan §4.4 line 1462):
  ``x_real = scale * q + offset`` for cint8; cfp16 passes through.

# Sequence-gap detection (chunk-8 path)

The transmitter keeps a per-``(host, port, chgroup)`` strictly-
monotonic ``seq``. On the receive side we track ``next_expected_seq``
per ``chgroup`` and count any incoming ``seq != next_expected``:

* ``seq == next_expected``  → in-order; bump expected.
* ``seq > next_expected``   → gap; ``n_seq_gaps += seq -
  next_expected`` (count of MISSING seq values, not skipped frames),
  then ``next_expected = seq + 1``.
* ``seq < next_expected``   → out-of-order or TX restart. Counted as
  ``n_out_of_order``; expected unchanged. (Loopback never reorders
  in the kernel; this branch is for safety.)

Per-``chgroup`` because production has 16 chgroups multiplexed onto
one search-side process; chunk-8 only operates on a single chgroup
per RX instance, but the dict-keyed counter is forward-compatible.

# Capture

:meth:`recv_into_capture` writes each incoming payload to
``capture_dir/seq_<seq:08d>_chg<g>_dm<d>_t<t>.<ext>`` and a
``meta.json`` index at the end. Used by the loopback bench + by the
M3 voltage-fixture sub-DoDs (chunks 5/6) once they wire transport-TX.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from dsart.transport.frame import (
    DTYPE_CFP16,
    DTYPE_CINT8,
    HEADER_BYTES,
    FastVisFrame,
    FrameCRCError,
    FrameMagicError,
)
from dsart.transport.prod_frame import (
    BITS_CFP16_COMPLEX,
    BITS_CINT8_COMPLEX,
    HEADER_BYTES as PROD_HEADER_BYTES,
    FLAG_RESERVED_BIT2,
    ProdFrameFieldRangeError,
    ProdFrameHeader,
    ProdFrameLengthError,
    ProdFrameMagicError,
    ProdFrameVersionError,
    unpack_frame,
)


LOG = logging.getLogger("dsart.transport.rx")


# Max UDP datagram on a 64 KiB-MTU loopback (65535 − 28).
_MAX_UDP_DATAGRAM: int = 65507


# ---------------------------------------------------------------------------
# Chunk-8 stats (unchanged)
# ---------------------------------------------------------------------------


@dataclass
class RxStats:
    """Per-RX-instance counters. Mirrors plan §4.3 mon-key set."""

    n_received: int = 0
    n_crc_fail: int = 0
    n_magic_fail: int = 0
    n_seq_gaps: int = 0
    n_out_of_order: int = 0
    bytes_received: int = 0
    # Per-chgroup state: chgroup → next_expected_seq.
    _next_seq: dict[int, int] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "n_received": self.n_received,
            "n_crc_fail": self.n_crc_fail,
            "n_magic_fail": self.n_magic_fail,
            "n_seq_gaps": self.n_seq_gaps,
            "n_out_of_order": self.n_out_of_order,
            "bytes_received": self.bytes_received,
        }


# ---------------------------------------------------------------------------
# M4a chunk-3: production config + reorder-window state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransportRxProdConfig:
    """Configuration for the M4a prod-frame receive path.

    Args:
        n_coarse_dm: number of coarse DM trials (sizes per-(corr, dm_idx) state).
        n_corr: number of correlator chgroups (default 16 per plan §4.3).
        reorder_window_depth: depth W of the per-(corr, dm_idx) reorder window
            (default 4 per plan §4.3 line 1473).
        so_rcvbuf_bytes: ``SO_RCVBUF`` socket option value (default 256 MiB per
            plan §4.3 line 1448).
        expected_pattern_id_by_chgroup: dict mapping chgroup → expected pattern_id.
            Set at ``cmd: prepare`` time. Missing entries → mismatch for all
            datagrams on that chgroup.
    """

    n_coarse_dm: int
    n_corr: int = 16
    reorder_window_depth: int = 4
    so_rcvbuf_bytes: int = 256 * 1024 * 1024
    expected_pattern_id_by_chgroup: dict[int, int] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.n_coarse_dm <= 0:
            raise ValueError(f"n_coarse_dm={self.n_coarse_dm}, expected > 0")
        if self.n_corr <= 0:
            raise ValueError(f"n_corr={self.n_corr}, expected > 0")
        if self.reorder_window_depth <= 0:
            raise ValueError(
                f"reorder_window_depth={self.reorder_window_depth}, expected > 0"
            )


@dataclass
class _PendingSeq:
    """One slot in the per-(corr, dm_idx) reorder window."""

    seq: int = 0
    n_frags_expected: int = 0
    # bit i set → fragment i has been received
    fragments_received_bitmap: int = 0
    # Partial payload buffer: list of (frag_idx, payload_bytes) pairs
    frag_buffers: list[tuple[int, bytes]] = field(default_factory=list)
    # Header from the first fragment (carries scale/offset/bits_per_cell/etc)
    header: Optional[ProdFrameHeader] = None
    # validity_flags for the ring slot when committed
    validity_flags: int = 0  # bit0 = data_present (set on commit)
    occupied: bool = False
    # committed: True after on_commit fires; stays True (does NOT reset)
    # until the window slides past this seq. Lets _slide_to distinguish a
    # successfully-committed slot from a never-received seq. See test
    # test_seq_gap_causes_zerofill (plan §4.3 line 1474).
    committed: bool = False

    def reset(self) -> None:
        self.seq = 0
        self.n_frags_expected = 0
        self.fragments_received_bitmap = 0
        self.frag_buffers = []
        self.header = None
        self.validity_flags = 0
        self.occupied = False
        self.committed = False

    @property
    def all_frags_received(self) -> bool:
        if self.n_frags_expected == 0:
            return False
        expected_mask = (1 << self.n_frags_expected) - 1
        return (self.fragments_received_bitmap & expected_mask) == expected_mask


class _ReorderWindow:
    """Fixed-depth ring of pending sequences for one (corr_idx, dm_idx) flow.

    Implements the plan §4.3 / §4.4 reorder window:
    - Depth W slots, indexed by (seq mod W).
    - On arrival of a fragment: if seq is in window → update bitmap.
      If seq > tail → slide window, zero-filling any dropped seqs.
      If seq < head → silently drop (late retransmission).
    """

    def __init__(
        self,
        depth: int,
        on_commit: Callable[[ProdFrameHeader, bytes, int], None],
        on_zerofill: Callable[[int, int], None],
        on_out_of_order: Callable[[], None],
    ) -> None:
        self._depth = depth
        self._slots: list[_PendingSeq] = [_PendingSeq() for _ in range(depth)]
        # head_seq: the seq of the oldest occupied slot (or the expected next
        # seq if the window is empty). Initialised to None before first packet.
        self._head_seq: Optional[int] = None
        self._on_commit = on_commit
        self._on_zerofill = on_zerofill
        self._on_out_of_order = on_out_of_order

    def _slot_for(self, seq: int) -> _PendingSeq:
        return self._slots[seq % self._depth]

    def _tail_seq(self) -> int:
        """seq of the newest slot = head + depth - 1."""
        assert self._head_seq is not None
        return self._head_seq + self._depth - 1

    def _slide_to(self, new_head: int) -> None:
        """Slide the window so that ``new_head`` is the new head seq.

        For each seq the window slides past:
        * If the slot belongs to that seq and was committed -> just reset
          (successful reassembly; no zero-fill).
        * If the slot belongs to that seq but reassembly was incomplete
          -> zero-fill with the known ``n_frags_expected``.
        * If the slot does NOT belong to that seq (slot is empty or
          holds a different / later seq) -> the seq was never received
          at all; zero-fill with ``n_frags_expected=0`` (unknown).

        The third case is the gap-detection path that
        ``test_seq_gap_causes_zerofill`` keys on (plan section 4.3 line
        1474: "When the window slides past a seq with missing
        fragments, the corresponding ring slot validity bit is set to
        false and the slot is zero-filled" -- "missing fragments"
        includes "never received any fragment").
        """
        assert self._head_seq is not None
        for seq_to_drop in range(self._head_seq, new_head):
            slot = self._slot_for(seq_to_drop)
            if slot.occupied and slot.seq == seq_to_drop:
                if slot.committed:
                    slot.reset()
                else:
                    self._on_zerofill(seq_to_drop, slot.n_frags_expected)
                    slot.reset()
            else:
                self._on_zerofill(seq_to_drop, 0)
        self._head_seq = new_head

    def ingest_fragment(
        self,
        header: ProdFrameHeader,
        payload: bytes,
    ) -> None:
        """Process one incoming fragment.

        On full reassembly → calls ``on_commit``.
        On window slide with drops → calls ``on_zerofill`` per dropped seq.
        On late arrival → calls ``on_out_of_order``; fragment discarded.
        """
        seq = header.seq
        frag_idx = header.frag_idx
        n_frags = header.n_frags

        # First ever packet: anchor the window.
        if self._head_seq is None:
            self._head_seq = seq

        # seq is behind the window → late re-transmission, drop silently.
        if seq < self._head_seq:
            self._on_out_of_order()
            return

        # seq is ahead of window tail → need to slide.
        tail = self._tail_seq()
        if seq > tail:
            # Slide so seq becomes the new tail (head advances accordingly).
            new_head = seq - self._depth + 1
            self._slide_to(new_head)

        slot = self._slot_for(seq)
        if not slot.occupied:
            slot.seq = seq
            slot.n_frags_expected = n_frags
            slot.occupied = True
            if slot.header is None:
                slot.header = header

        # Update bitmap and stash fragment.
        if frag_idx < 64:
            slot.fragments_received_bitmap |= (1 << frag_idx)
        slot.frag_buffers.append((frag_idx, payload))

        if slot.all_frags_received and not slot.committed:
            # Reassemble fragments in order.
            slot.frag_buffers.sort(key=lambda x: x[0])
            full_payload = b"".join(p for _, p in slot.frag_buffers)
            validity = 0b00000001  # bit0 = data_present
            validity |= slot.validity_flags
            self._on_commit(slot.header, full_payload, validity)
            # Keep slot.occupied + slot.seq so _slide_to can distinguish
            # "successfully committed" from "never received". The slot is
            # released by _slide_to when the window advances past this seq.
            slot.committed = True
            slot.frag_buffers = []  # free payload bytes eagerly


# ---------------------------------------------------------------------------
# M4a chunk-3: prod-frame stats
# ---------------------------------------------------------------------------


@dataclass
class RxProdStats:
    """Per-flow prod-frame statistics (plan §4.3 mon-keys for chunk 3)."""

    # pattern_mismatch_count: keyed by chgroup
    pattern_mismatch_count: dict[int, int] = field(default_factory=dict)
    # seq_gap_count_per_flow: keyed by (corr_idx, dm_idx)
    seq_gap_count_per_flow: dict[tuple[int, int], int] = field(default_factory=dict)
    # window_slide_zerofill_count: total window-slide zero-fill events
    window_slide_zerofill_count: int = 0
    # out_of_order_drop_count: frames dropped as late arrivals
    out_of_order_drop_count: int = 0
    # Header error counters
    bad_magic_count: int = 0
    bad_version_count: int = 0
    bad_length_count: int = 0
    bad_field_range_count: int = 0
    reserved_bit_count: int = 0
    # Total frames received (valid header, including pattern mismatches)
    n_received: int = 0
    # Total committed (fully reassembled) payloads
    n_committed: int = 0

    def to_dict(self) -> dict:
        return {
            "pattern_mismatch_count": dict(self.pattern_mismatch_count),
            "seq_gap_count_per_flow": {
                f"{k[0]}:{k[1]}": v
                for k, v in self.seq_gap_count_per_flow.items()
            },
            "window_slide_zerofill_count": self.window_slide_zerofill_count,
            "out_of_order_drop_count": self.out_of_order_drop_count,
            "bad_magic_count": self.bad_magic_count,
            "bad_version_count": self.bad_version_count,
            "bad_length_count": self.bad_length_count,
            "bad_field_range_count": self.bad_field_range_count,
            "reserved_bit_count": self.reserved_bit_count,
            "n_received": self.n_received,
            "n_committed": self.n_committed,
        }


# ---------------------------------------------------------------------------
# M4a chunk-3: assembled slot type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RxProdSlot:
    """One fully-reassembled COO payload from the production RX path.

    Carries the dequantised float32 complex values and associated metadata.
    If ``data_present`` is False the ``values`` array is zero-filled (either
    due to a window-slide drop or a pattern_id mismatch).
    """

    header: ProdFrameHeader
    # Dequantised values: [n_filled] complex64
    values: np.ndarray
    # Raw payload bytes (before dequant), kept for downstream processing
    raw_payload: bytes
    # validity_flags as per CONC-1 contract bit field
    validity_flags: int
    # Convenience accessors
    data_present: bool
    pattern_mismatch: bool


# ---------------------------------------------------------------------------
# Dequantisation helpers (plan §4.4 line 1462)
# ---------------------------------------------------------------------------


def dequantise_cint8(
    raw: bytes,
    n_filled: int,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Dequantise a cint8 complex payload to float32 complex.

    Wire layout: ``[n_filled, 2]`` int8 (re, im interleaved). Output:
    ``[n_filled]`` complex64.

    Per plan §4.4 line 1462: ``x_real = scale * q + offset``.
    """
    raw_arr = np.frombuffer(raw, dtype=np.int8)
    if raw_arr.size != n_filled * 2:
        raise ValueError(
            f"dequantise_cint8: expected {n_filled * 2} int8 bytes, "
            f"got {raw_arr.size}"
        )
    ri = raw_arr.reshape(n_filled, 2).astype(np.float32)
    return (ri[:, 0] * np.float32(scale) + np.float32(offset)) + 1j * (
        ri[:, 1] * np.float32(scale) + np.float32(offset)
    )


def dequantise_cfp16(
    raw: bytes,
    n_filled: int,
) -> np.ndarray:
    """Pass-through for cfp16 complex payload (already float-typed).

    Wire layout: ``[n_filled, 2]`` float16 (re, im interleaved). Output:
    ``[n_filled]`` complex64.
    """
    raw_arr = np.frombuffer(raw, dtype=np.float16)
    if raw_arr.size != n_filled * 2:
        raise ValueError(
            f"dequantise_cfp16: expected {n_filled * 2} fp16 values, "
            f"got {raw_arr.size}"
        )
    ri = raw_arr.reshape(n_filled, 2).astype(np.float32)
    return ri[:, 0] + 1j * ri[:, 1]


def dequantise_payload(
    raw: bytes,
    n_filled: int,
    bits_per_cell: int,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Dispatch to the correct dequantiser based on ``bits_per_cell``."""
    if bits_per_cell == BITS_CINT8_COMPLEX:
        return dequantise_cint8(raw, n_filled, scale, offset)
    elif bits_per_cell == BITS_CFP16_COMPLEX:
        return dequantise_cfp16(raw, n_filled)
    else:
        raise ValueError(
            f"dequantise_payload: unsupported bits_per_cell={bits_per_cell}"
        )


# ---------------------------------------------------------------------------
# M4a chunk-3: TransportRxProd (prod-frame receive path)
# ---------------------------------------------------------------------------


class TransportRxProd:
    """Production prod-frame receiver (M4a chunk 3).

    Manages per-(corr_idx, dm_idx) reorder windows, pattern_id verify, and
    dequantisation. Called from ``TransportRx`` when ``use_prod_frame=True``.

    Args:
        config: ``TransportRxProdConfig`` with dimensions and expected pattern IDs.
        ring_write_cb: callback invoked for each fully-reassembled + dequantised
            slot. Signature: ``(corr_idx, dm_idx, slot: RxProdSlot) -> None``.
            Used by chunk-4 to write the slot into the POSIX-shm ring.
            Tests may pass a stub that stores slots in a list.
    """

    def __init__(
        self,
        config: TransportRxProdConfig,
        ring_write_cb: Callable[[int, int, RxProdSlot], None] | None = None,
    ) -> None:
        self._cfg = config
        self._ring_write_cb = ring_write_cb
        self.prod_stats = RxProdStats()
        # per-(corr_idx, dm_idx) reorder windows
        self._windows: dict[tuple[int, int], _ReorderWindow] = {}

    def update_expected_pattern_id(
        self, chgroup: int, pattern_id: int
    ) -> None:
        """Update the expected pattern_id for a chgroup (cmd: prepare reload)."""
        # Cast to mutable dict – config is frozen but the dict value is mutable
        # because Python dicts passed as defaults are mutable containers.
        self._cfg.expected_pattern_id_by_chgroup[chgroup] = pattern_id

    def _get_window(self, corr_idx: int, dm_idx: int) -> _ReorderWindow:
        key = (corr_idx, dm_idx)
        if key not in self._windows:
            self._windows[key] = _ReorderWindow(
                depth=self._cfg.reorder_window_depth,
                on_commit=self._make_commit_cb(corr_idx, dm_idx),
                on_zerofill=self._make_zerofill_cb(corr_idx, dm_idx),
                on_out_of_order=self._make_oor_cb(corr_idx, dm_idx),
            )
        return self._windows[key]

    def _make_commit_cb(
        self, corr_idx: int, dm_idx: int
    ) -> Callable[[ProdFrameHeader, bytes, int], None]:
        def _commit(
            hdr: ProdFrameHeader, raw_payload: bytes, validity: int
        ) -> None:
            self.prod_stats.n_committed += 1
            data_present = bool(validity & 0x01)
            pattern_mismatch = bool(validity & 0x02)
            if data_present and raw_payload:
                try:
                    values = dequantise_payload(
                        raw_payload,
                        hdr.n_filled,
                        hdr.bits_per_cell,
                        hdr.scale,
                        hdr.offset,
                    )
                except (ValueError, ArithmeticError):
                    values = np.zeros(hdr.n_filled, dtype=np.complex64)
            else:
                values = np.zeros(
                    hdr.n_filled if hdr is not None else 0,
                    dtype=np.complex64,
                )
            slot = RxProdSlot(
                header=hdr,
                values=values,
                raw_payload=raw_payload,
                validity_flags=validity,
                data_present=data_present,
                pattern_mismatch=pattern_mismatch,
            )
            if self._ring_write_cb is not None:
                self._ring_write_cb(corr_idx, dm_idx, slot)

        return _commit

    def _make_zerofill_cb(
        self, corr_idx: int, dm_idx: int
    ) -> Callable[[int, int], None]:
        def _zerofill(seq: int, n_frags: int) -> None:
            self.prod_stats.window_slide_zerofill_count += 1
            key = (corr_idx, dm_idx)
            self.prod_stats.seq_gap_count_per_flow[key] = (
                self.prod_stats.seq_gap_count_per_flow.get(key, 0) + 1
            )
            LOG.debug(
                "rx_prod: zero-fill (corr=%d, dm=%d, seq=%d, n_frags=%d)",
                corr_idx,
                dm_idx,
                seq,
                n_frags,
            )

        return _zerofill

    def _make_oor_cb(
        self, corr_idx: int, dm_idx: int
    ) -> Callable[[], None]:
        def _oor() -> None:
            self.prod_stats.out_of_order_drop_count += 1
            LOG.debug(
                "rx_prod: out-of-order drop (corr=%d, dm=%d)", corr_idx, dm_idx
            )

        return _oor

    def ingest_datagram(self, buf: bytes | bytearray | memoryview) -> None:
        """Process one raw UDP datagram buffer.

        Validates the header, checks pattern_id, and routes to the correct
        reorder window. All errors are caught and counted; nothing is raised.
        """
        try:
            hdr, payload = unpack_frame(buf)
        except ProdFrameMagicError:
            self.prod_stats.bad_magic_count += 1
            return
        except ProdFrameVersionError:
            self.prod_stats.bad_version_count += 1
            return
        except ProdFrameLengthError:
            self.prod_stats.bad_length_count += 1
            return
        except ProdFrameFieldRangeError:
            self.prod_stats.bad_field_range_count += 1
            return

        self.prod_stats.n_received += 1

        # v1: reserved bit2 MUST NOT be set by sender.
        if hdr.flags & FLAG_RESERVED_BIT2:
            self.prod_stats.reserved_bit_count += 1
            LOG.warning(
                "rx_prod: datagram with FLAG_RESERVED_BIT2 set (seq=%d chg=%d) — dropped",
                hdr.seq,
                hdr.chgroup,
            )
            return

        # pattern_id verify (plan §3 line 308).
        expected_pid = self._cfg.expected_pattern_id_by_chgroup.get(
            hdr.chgroup
        )
        if expected_pid is not None and hdr.pattern_id != expected_pid:
            chg = hdr.chgroup
            self.prod_stats.pattern_mismatch_count[chg] = (
                self.prod_stats.pattern_mismatch_count.get(chg, 0) + 1
            )
            LOG.debug(
                "rx_prod: pattern_id mismatch (chg=%d seq=%d expected=%#x got=%#x)",
                chg,
                hdr.seq,
                expected_pid,
                hdr.pattern_id,
            )
            # Drop: emit a zero-fill slot with pattern_mismatch bit set.
            zero_vals = np.zeros(hdr.n_filled, dtype=np.complex64)
            validity = 0b00000010  # bit1 = pattern_mismatch; bit0 NOT set
            slot = RxProdSlot(
                header=hdr,
                values=zero_vals,
                raw_payload=b"",
                validity_flags=validity,
                data_present=False,
                pattern_mismatch=True,
            )
            if self._ring_write_cb is not None:
                self._ring_write_cb(hdr.chgroup, hdr.dm_idx, slot)
            return

        # Route to reorder window.
        # corr_idx is taken from chgroup (each corr node owns one chgroup).
        corr_idx = hdr.chgroup
        window = self._get_window(corr_idx, hdr.dm_idx)
        window.ingest_fragment(hdr, payload)


# ---------------------------------------------------------------------------
# TransportRx (chunk-8 path, unchanged + prod-frame flag)
# ---------------------------------------------------------------------------


class TransportRx:
    """UDP receiver for fast-vis cubes.

    Args:
        host: bind IP. Loopback bench: ``127.0.0.1``. ``0.0.0.0`` for
            multi-interface listeners (chunk 8 doesn't use this).
        port: bind UDP port. Pass ``0`` to let the kernel pick an
            ephemeral free port (used by acceptance tests). After
            ``__init__`` the assigned port is in :attr:`port`.
        recv_timeout_s: ``socket.SO_RCVTIMEO``. ``receive_one``
            returns ``None`` rather than raising on timeout.
        use_prod_frame: if ``True``, use the M4a prod-frame path.
            If ``False`` (default), use the chunk-8 ``FastVisFrame`` path.
        prod_config: required when ``use_prod_frame=True``.
        ring_write_cb: forwarded to :class:`TransportRxProd`; called for
            each fully-reassembled + dequantised slot.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        recv_timeout_s: float = 1.0,
        use_prod_frame: bool = False,
        prod_config: TransportRxProdConfig | None = None,
        ring_write_cb: Callable[[int, int, RxProdSlot], None] | None = None,
    ) -> None:
        self.host = host
        self.use_prod_frame = use_prod_frame
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if use_prod_frame and prod_config is not None:
            rcvbuf = prod_config.so_rcvbuf_bytes
        else:
            # Production tunes SO_RCVBUF to 256 MiB (plan §4.3); for
            # loopback / acceptance tests, 8 MiB is plenty.
            rcvbuf = 8 * 1024 * 1024

        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf,
            )
        except OSError:
            LOG.warning("could not raise SO_RCVBUF; bursts may drop")

        self._sock.bind((host, int(port)))
        bound_host, bound_port = self._sock.getsockname()
        self.port: int = int(bound_port)
        self.host_actual: str = str(bound_host)
        self.recv_timeout_s = float(recv_timeout_s)
        self._sock.settimeout(self.recv_timeout_s)
        self.stats = RxStats()

        # Prod-frame path state.
        if use_prod_frame:
            if prod_config is None:
                raise ValueError(
                    "TransportRx(use_prod_frame=True) requires prod_config"
                )
            self._prod = TransportRxProd(prod_config, ring_write_cb)
        else:
            self._prod = None  # type: ignore[assignment]

    @property
    def prod_stats(self) -> RxProdStats | None:
        if self._prod is None:
            return None
        return self._prod.prod_stats

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "TransportRx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- Public API ------------------------------------------------------

    def receive_one(self) -> FastVisFrame | None:
        """Receive ONE frame off the socket (chunk-8 path only).

        Returns:
            The decoded :class:`FastVisFrame` on success; ``None`` if
            the socket timed out (no frame within
            :attr:`recv_timeout_s`).

        Raises:
            FrameMagicError: incoming buffer doesn't carry the expected
                magic. Counters are bumped before raise so production
                can choose to swallow this in a non-strict capture loop.
            FrameCRCError: CRC mismatch. Same handling as above.
            ValueError: malformed / truncated buffer.

        On any of these errors the per-(host, port) stats are still
        updated (``n_magic_fail``, ``n_crc_fail``).
        """
        try:
            buf, _addr = self._sock.recvfrom(_MAX_UDP_DATAGRAM)
        except socket.timeout:
            return None
        if len(buf) < HEADER_BYTES:
            self.stats.n_magic_fail += 1
            raise ValueError(
                f"TransportRx.receive_one: short buffer: {len(buf)} bytes "
                f"< {HEADER_BYTES}"
            )
        try:
            frame = FastVisFrame.unpack(buf)
        except FrameMagicError:
            self.stats.n_magic_fail += 1
            raise
        except FrameCRCError:
            self.stats.n_crc_fail += 1
            raise
        # Stats / seq accounting (only for valid frames).
        self.stats.n_received += 1
        self.stats.bytes_received += len(buf)
        self._update_seq_accounting(frame)
        return frame

    def receive_one_prod(self) -> None:
        """Receive ONE prod-frame datagram off the socket (prod-frame path).

        On reassembly completion the ``ring_write_cb`` is invoked. Returns
        ``None`` on timeout. Header errors are counted but not raised.
        """
        if self._prod is None:
            raise RuntimeError(
                "TransportRx.receive_one_prod called but use_prod_frame=False"
            )
        try:
            buf, _addr = self._sock.recvfrom(_MAX_UDP_DATAGRAM)
        except socket.timeout:
            return None
        self.stats.bytes_received += len(buf)
        self._prod.ingest_datagram(buf)
        return None

    def recv_into_capture(
        self,
        capture_dir: Path,
        max_frames: int,
        *,
        progress_every: int = 0,
    ) -> dict[str, int]:
        """Receive up to ``max_frames`` frames; persist each payload
        to disk + write a ``meta.json`` index. Returns the
        :class:`RxStats` dict.

        Output layout:

            capture_dir/
              seq_<seq:08d>_chg<g>_dm<d>_t<t>.<ext>     (one per frame)
              meta.json                                 (final index)

        Where ``<ext>`` is ``cfp16`` for ``dtype_code=0`` or ``cint8``
        for ``dtype_code=1``. The capture dir is created if missing.

        Args:
            capture_dir: target dir.
            max_frames: stop after this many *valid* frames received
                (not counting CRC / magic failures).
            progress_every: log every N frames (0 = silent).
        """
        capture_dir = Path(capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        index: list[dict] = []
        t0 = time.monotonic()
        while self.stats.n_received < max_frames:
            try:
                frame = self.receive_one()
            except (FrameMagicError, FrameCRCError, ValueError) as exc:
                LOG.warning("RX frame rejected: %s", exc)
                continue
            if frame is None:
                continue                                                 # timeout; loop again
            ext = self._dtype_ext(frame.dtype_code)
            fname = (
                f"seq_{frame.seq:08d}_chg{frame.chgroup}_"
                f"dm{frame.dm_idx}_t{frame.t_idx:04d}.{ext}"
            )
            (capture_dir / fname).write_bytes(frame.payload)
            entry = frame.to_dict()
            entry["filename"] = fname
            index.append(entry)
            if progress_every and self.stats.n_received % progress_every == 0:
                LOG.info(
                    "rx progress: n=%d gaps=%d crc_fail=%d",
                    self.stats.n_received, self.stats.n_seq_gaps,
                    self.stats.n_crc_fail,
                )
        elapsed = time.monotonic() - t0
        meta = {
            "stats": self.stats.to_dict(),
            "frames": index,
            "elapsed_s": elapsed,
            "max_frames": max_frames,
        }
        (capture_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self.stats.to_dict()

    # ---- Internals -------------------------------------------------------

    def _update_seq_accounting(self, frame: FastVisFrame) -> None:
        next_seq = self.stats._next_seq.get(frame.chgroup)
        if next_seq is None:
            # First frame on this chgroup: anchor.
            self.stats._next_seq[frame.chgroup] = (frame.seq + 1) & 0xFFFF_FFFF
            return
        if frame.seq == next_seq:
            self.stats._next_seq[frame.chgroup] = (frame.seq + 1) & 0xFFFF_FFFF
        elif frame.seq > next_seq:
            self.stats.n_seq_gaps += frame.seq - next_seq
            self.stats._next_seq[frame.chgroup] = (frame.seq + 1) & 0xFFFF_FFFF
        else:
            # frame.seq < next_seq → out-of-order or TX restart.
            self.stats.n_out_of_order += 1

    @staticmethod
    def _dtype_ext(dtype_code: int) -> str:
        if dtype_code == DTYPE_CFP16:
            return "cfp16"
        if dtype_code == DTYPE_CINT8:
            return "cint8"
        return f"raw{dtype_code:02d}"
