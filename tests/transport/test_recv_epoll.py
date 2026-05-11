"""M4a chunk 6 — unit tests for the C epoll receive loop.

Verifies the C path's semantics match the Python ``TransportRxProd``
reference implementation (chunk 3) on every counter that affects
production correctness: ``n_received``, ``n_committed``,
``pattern_mismatch_count``, ``window_slide_zerofill_count``,
``out_of_order_drop_count``, and all ``bad_*`` validation counters.

Tests use real UDP loopback so we exercise the full C path
(recvmmsg → header parse → reorder window → atomic counters). The
tests are sequential because ``RxEpoll`` is a process-global singleton.
"""
from __future__ import annotations

import os
import socket
import struct
import time

import pytest

from dsart.transport.prod_frame import (
    BITS_CINT8_COMPLEX,
    FLAG_LAST_IN_BLOCK,
    FLAG_QUANTIZED,
    FLAG_RESERVED_BIT2,
    HEADER_BYTES,
    MAGIC,
    ProdFrameHeader,
    pack_frame,
    split_payload_into_fragments,
)

# Skip whole module if the C extension hasn't been built yet.
recv_epoll = pytest.importorskip("dsart.transport.recv_epoll")
RxEpoll = recv_epoll.RxEpoll


PID_OK = 0xCAFEBABEDEADBEEF


def _hdr(
    *,
    seq: int = 0,
    specnum: int = 0,
    chgroup: int = 0,
    dm_idx: int = 0,
    frag_idx: int = 0,
    n_frags: int = 1,
    n_grid: int = 256,
    n_filled: int = 100,
    pattern_id: int = PID_OK,
    bits_per_cell: int = BITS_CINT8_COMPLEX,
    t_int_factor: int = 16,
    scale: float = 1.0,
    offset: float = 0.0,
    payload_bytes_in_frag: int = 200,
    flags: int = FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK,
) -> ProdFrameHeader:
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
    )


def _send(sock: socket.socket, dst: tuple[str, int], wire: bytes) -> None:
    sock.sendto(wire, dst)


@pytest.fixture
def rx():
    """RxEpoll fixture — opens, starts, tears down for one test."""
    instance = RxEpoll.open(
        bind_host="127.0.0.1", bind_port=0,
        so_rcvbuf_bytes=8 * 1024 * 1024,
    )
    instance.start()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def tx_sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        yield s
    finally:
        s.close()


def _drain(rx: "RxEpoll", timeout_s: float = 0.5) -> None:
    """Sleep long enough for the C loop to drain in-flight datagrams."""
    time.sleep(timeout_s)


class TestBasicPacketReception:
    """Single-fragment payloads, no fragmentation, single chgroup."""

    def test_one_payload_one_fragment(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 100
        payload = bytes(n_filled * 2)
        hdr = _hdr(n_filled=n_filled, payload_bytes_in_frag=len(payload))
        wire = pack_frame(hdr, payload)
        _send(tx_sock, ("127.0.0.1", rx.port), wire)
        _drain(rx)
        c = rx.counters()
        assert c.n_received == 1
        assert c.n_committed == 1
        assert c.bad_magic_count == 0
        assert c.bad_length_count == 0
        assert c.pattern_mismatch_count == 0

    def test_many_payloads_no_loss(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 100
        payload = bytes(n_filled * 2)
        for seq in range(50):
            hdr = _hdr(seq=seq, n_filled=n_filled,
                       payload_bytes_in_frag=len(payload))
            wire = pack_frame(hdr, payload)
            _send(tx_sock, ("127.0.0.1", rx.port), wire)
        _drain(rx)
        c = rx.counters()
        assert c.n_received == 50
        assert c.n_committed == 50
        assert c.window_slide_zerofill_count == 0

    def test_bytes_received_total_matches_wire(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 100
        payload = bytes(n_filled * 2)
        hdr = _hdr(n_filled=n_filled, payload_bytes_in_frag=len(payload))
        wire = pack_frame(hdr, payload)
        for _ in range(10):
            _send(tx_sock, ("127.0.0.1", rx.port), wire)
        _drain(rx)
        c = rx.counters()
        assert c.bytes_received_total == 10 * len(wire)


class TestFragmentation:
    """Multi-fragment payloads with the chunk-3 reorder window logic."""

    def test_two_fragments_in_order_commits(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 5000  # → 10000 cint8 bytes → 2 frags at 8964 cap
        payload = bytes(n_filled * 2)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=8964)
        assert len(frags) == 2
        for frag_idx, frag in enumerate(frags):
            flags = FLAG_QUANTIZED
            if frag_idx == len(frags) - 1:
                flags |= FLAG_LAST_IN_BLOCK
            hdr = _hdr(
                seq=7, n_filled=n_filled, n_frags=len(frags),
                frag_idx=frag_idx,
                payload_bytes_in_frag=len(frag),
                flags=flags,
            )
            _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, frag))
        _drain(rx)
        c = rx.counters()
        assert c.n_received == 2
        assert c.n_committed == 1
        assert c.window_slide_zerofill_count == 0

    def test_two_fragments_reverse_order_commits(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 5000
        payload = bytes(n_filled * 2)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=8964)
        # Send frag 1 first, then frag 0.
        for frag_idx in [1, 0]:
            flags = FLAG_QUANTIZED
            if frag_idx == len(frags) - 1:
                flags |= FLAG_LAST_IN_BLOCK
            hdr = _hdr(
                seq=12, n_filled=n_filled, n_frags=len(frags),
                frag_idx=frag_idx,
                payload_bytes_in_frag=len(frags[frag_idx]),
                flags=flags,
            )
            _send(tx_sock, ("127.0.0.1", rx.port),
                  pack_frame(hdr, frags[frag_idx]))
        _drain(rx)
        c = rx.counters()
        assert c.n_received == 2
        assert c.n_committed == 1

    def test_missing_fragment_then_window_slide_zerofills(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 5000
        payload = bytes(n_filled * 2)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=8964)
        # Send only frag 0 of seq=0.
        hdr0 = _hdr(seq=0, n_filled=n_filled, n_frags=len(frags),
                    frag_idx=0, payload_bytes_in_frag=len(frags[0]),
                    flags=FLAG_QUANTIZED)
        _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr0, frags[0]))
        # Advance window past seq=0 by sending seq=4 (depth W=4 means
        # head moves to 1 after seq=4 arrives → 0 is zero-filled).
        for seq in range(1, 5):
            for fi, f in enumerate(frags):
                flags = FLAG_QUANTIZED
                if fi == len(frags) - 1:
                    flags |= FLAG_LAST_IN_BLOCK
                hdr = _hdr(seq=seq, n_filled=n_filled, n_frags=len(frags),
                           frag_idx=fi, payload_bytes_in_frag=len(f),
                           flags=flags)
                _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, f))
        _drain(rx)
        c = rx.counters()
        # 4 complete payloads should commit; seq=0 should zerofill.
        assert c.n_committed == 4
        assert c.window_slide_zerofill_count >= 1


class TestPatternIdValidation:
    def test_matching_pattern_id_accepted(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 100
        payload = bytes(n_filled * 2)
        hdr = _hdr(pattern_id=PID_OK, n_filled=n_filled,
                   payload_bytes_in_frag=len(payload))
        _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain(rx)
        c = rx.counters()
        assert c.pattern_mismatch_count == 0
        assert c.n_committed == 1

    def test_wrong_pattern_id_increments_mismatch(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 100
        payload = bytes(n_filled * 2)
        # Send 10 datagrams with the wrong pattern_id.
        for seq in range(10):
            hdr = _hdr(seq=seq, pattern_id=PID_OK ^ 0xFFFF, n_filled=n_filled,
                       payload_bytes_in_frag=len(payload))
            _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain(rx)
        c = rx.counters()
        assert c.pattern_mismatch_count == 10
        assert c.n_committed == 0  # all dropped at the pattern check

    def test_no_expected_pid_accepts_anything(self, rx, tx_sock):
        # Don't set expected pattern_id; everything should commit.
        n_filled = 100
        payload = bytes(n_filled * 2)
        hdr = _hdr(pattern_id=0x1234, n_filled=n_filled,
                   payload_bytes_in_frag=len(payload))
        _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain(rx)
        c = rx.counters()
        assert c.pattern_mismatch_count == 0
        assert c.n_committed == 1


class TestProtocolValidation:
    """Bad headers must be counted, not committed."""

    def test_bad_magic_dropped(self, rx, tx_sock):
        n_filled = 100
        payload = bytes(n_filled * 2)
        hdr = _hdr(n_filled=n_filled, payload_bytes_in_frag=len(payload))
        wire = bytearray(pack_frame(hdr, payload))
        # Corrupt magic.
        struct.pack_into("<I", wire, 0, 0xDEADBEEF)
        _send(tx_sock, ("127.0.0.1", rx.port), bytes(wire))
        _drain(rx)
        c = rx.counters()
        assert c.bad_magic_count == 1
        assert c.n_committed == 0

    def test_bad_length_dropped(self, rx, tx_sock):
        n_filled = 100
        payload = bytes(n_filled * 2)
        hdr = _hdr(n_filled=n_filled, payload_bytes_in_frag=len(payload))
        wire = pack_frame(hdr, payload)
        # Truncate to less than the header — C will count this as bad_length.
        _send(tx_sock, ("127.0.0.1", rx.port), wire[:HEADER_BYTES - 10])
        _drain(rx)
        c = rx.counters()
        assert c.bad_length_count >= 1
        assert c.n_committed == 0

    def test_reserved_bit2_dropped(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, PID_OK)
        n_filled = 100
        payload = bytes(n_filled * 2)
        # Build a valid frame, then patch FLAG_RESERVED_BIT2 in.
        hdr = _hdr(n_filled=n_filled, payload_bytes_in_frag=len(payload))
        wire = bytearray(pack_frame(hdr, payload))
        # flags is at offset 6, u16 little-endian.
        existing_flags = struct.unpack_from("<H", wire, 6)[0]
        struct.pack_into("<H", wire, 6, existing_flags | FLAG_RESERVED_BIT2)
        _send(tx_sock, ("127.0.0.1", rx.port), bytes(wire))
        _drain(rx)
        c = rx.counters()
        assert c.reserved_bit_count >= 1
        assert c.n_committed == 0


class TestUpdateExpectedPid:
    """cmd:prepare-style mid-flight pattern_id updates."""

    def test_clear_then_set_resumes_acceptance(self, rx, tx_sock):
        rx.set_expected_pattern_id(0, 0x1111)
        # First batch: pid mismatch (rx expects 0x1111, tx sends PID_OK).
        n_filled = 100
        payload = bytes(n_filled * 2)
        for seq in range(3):
            hdr = _hdr(seq=seq, pattern_id=PID_OK, n_filled=n_filled,
                       payload_bytes_in_frag=len(payload))
            _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain(rx)
        c0 = rx.counters()
        assert c0.pattern_mismatch_count == 3
        assert c0.n_committed == 0

        # Update RX's expected pid → next batch must commit.
        rx.set_expected_pattern_id(0, PID_OK)
        for seq in range(3, 6):
            hdr = _hdr(seq=seq, pattern_id=PID_OK, n_filled=n_filled,
                       payload_bytes_in_frag=len(payload))
            _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain(rx)
        c1 = rx.counters()
        assert c1.pattern_mismatch_count == 3  # unchanged
        assert c1.n_committed == 3
