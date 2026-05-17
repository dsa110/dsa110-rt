"""M7.2 Phase B — unit tests for recv_epoll ↔ RxRing write-through.

Builds on test_recv_epoll.py (Phase A counters-only tests) by also
attaching a POSIX-shm ring during the receive loop and asserting that:

  * Reassembled cube slots land in the ring at the right
    ``(corr, dm, t_seq)`` with the right payload and ``VF_DATA_PRESENT``.
  * Pattern-mismatched frames publish a zero-payload slot with
    ``VF_PATTERN_MISMATCH`` (so consumers don't stall waiting for a
    write_seq advance that would never come).
  * Slots that age out of the reorder window without all fragments
    receive a ``validity == 0`` hole slot (write_seq still advances; the
    consumer's validity_mask drops the t).
  * Multi-port bind (``add_port``) lets one process drain the 16
    production listen ports under one drainer thread, with ``hdr.chgroup``
    routing inside the C reorder window unchanged.
  * The new ``ring_*`` counter family stays consistent with what was
    published.

These tests use real UDP loopback so they exercise the full C path
(recvmmsg → header parse → reorder window → ring publish → reader
``rx_ring_read_slot``). Each test owns the RxEpoll singleton and its
own shm name so a failing test doesn't leak the segment into the next
test's namespace.
"""
from __future__ import annotations

import socket
import time
import uuid

import pytest

from dsart.transport.prod_frame import (
    BITS_CINT8_COMPLEX,
    FLAG_LAST_IN_BLOCK,
    FLAG_QUANTIZED,
    ProdFrameHeader,
    pack_frame,
    split_payload_into_fragments,
)

recv_epoll = pytest.importorskip("dsart.transport.recv_epoll")
recv_ring = pytest.importorskip("dsart.transport.recv_ring")
RxEpoll = recv_epoll.RxEpoll
RxRing = recv_ring.RxRing
RxRingDims = recv_ring.RxRingDims

VF_DATA_PRESENT = recv_ring.VF_DATA_PRESENT
VF_PATTERN_MISMATCH = recv_ring.VF_PATTERN_MISMATCH

PID_OK = 0xCAFEBABEDEADBEEF
PID_WRONG = 0xDEADC0DEBADBAD00


def _hdr(
    *,
    seq: int = 0,
    chgroup: int = 0,
    dm_idx: int = 0,
    frag_idx: int = 0,
    n_frags: int = 1,
    n_filled: int = 100,
    pattern_id: int = PID_OK,
    payload_bytes_in_frag: int = 200,
    flags: int = FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK,
) -> ProdFrameHeader:
    return ProdFrameHeader(
        seq=seq,
        specnum=seq * 256,            # arbitrary; not exercised by Phase B
        chgroup=chgroup,
        dm_idx=dm_idx,
        frag_idx=frag_idx,
        n_frags=n_frags,
        n_grid=256,
        n_filled=n_filled,
        pattern_id=pattern_id,
        bits_per_cell=BITS_CINT8_COMPLEX,
        t_int_factor=16,
        scale=1.0,
        offset=0.0,
        payload_bytes_in_frag=payload_bytes_in_frag,
        flags=flags,
    )


def _send(sock: socket.socket, dst: tuple[str, int], wire: bytes) -> None:
    sock.sendto(wire, dst)


def _drain(seconds: float = 0.5) -> None:
    """Give the C loop a window to recvmmsg + ring-publish."""
    time.sleep(seconds)


def _unique_shm_name(test_name: str) -> str:
    return f"/dsart-rx-phaseb-{test_name}-{uuid.uuid4().hex[:8]}"


# Per-test ring dims. Small enough to keep the test footprint trivial
# (~100 KiB shm), large enough that the t_seq values used in tests
# don't wrap (T_buf >> max seq).
DEFAULT_DIMS = RxRingDims(
    n_corr=4,
    n_coarse_dm=2,
    t_buf_samples=64,
    n_filled_per_corr=100,
    bytes_per_cell=2,  # cint8 complex
)
DEFAULT_PAYLOAD_BYTES = DEFAULT_DIMS.n_filled_per_corr * DEFAULT_DIMS.bytes_per_cell


@pytest.fixture
def shm_name(request):
    """Allocate a unique shm name; ensure it's unlinked after the test."""
    name = _unique_shm_name(request.node.name.replace("[", "-").replace("]", ""))
    yield name
    # Best-effort cleanup. If the test forgot to detach + close, the next
    # process to attempt the same name would get EEXIST; unlinking here
    # restores a clean namespace for the next test.
    try:
        RxRing.unlink_name(name)
    except Exception:
        pass


@pytest.fixture
def rx_with_ring(shm_name):
    """Open RxEpoll (singleton), attach a ring with DEFAULT_DIMS, start the
    loop. Tears everything down (loop → detach → close → unlink) on exit."""
    instance = RxEpoll.open(
        bind_host="127.0.0.1",
        bind_port=0,
        so_rcvbuf_bytes=8 * 1024 * 1024,
    )
    instance.attach_ring(
        shm_name=shm_name,
        owner=True,
        n_corr=DEFAULT_DIMS.n_corr,
        n_coarse_dm=DEFAULT_DIMS.n_coarse_dm,
        t_buf_samples=DEFAULT_DIMS.t_buf_samples,
        n_filled=DEFAULT_DIMS.n_filled_per_corr,
        bytes_per_cell=DEFAULT_DIMS.bytes_per_cell,
    )
    instance.start()
    try:
        yield instance, shm_name
    finally:
        instance.close()


@pytest.fixture
def tx_sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        yield s
    finally:
        s.close()


def _attach_reader(shm_name: str) -> RxRing:
    """Open a read-only attach to the same ring the RX is writing into."""
    return RxRing.mmap_attach_readonly(shm_name, DEFAULT_DIMS)


# ---------------------------------------------------------------------------
# A. Basic single-frame publish
# ---------------------------------------------------------------------------


class TestSingleFramePublish:
    def test_single_frame_lands_in_ring_with_data_present(
        self, rx_with_ring, tx_sock
    ):
        rx, name = rx_with_ring
        rx.set_expected_pattern_id(0, PID_OK)

        # Construct a payload with a distinctive byte pattern so we
        # can assert the exact bytes ended up in the ring.
        payload = bytes((i * 7) & 0xFF for i in range(DEFAULT_PAYLOAD_BYTES))
        hdr = _hdr(seq=0, chgroup=0, dm_idx=0, n_filled=100,
                   payload_bytes_in_frag=len(payload))
        _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain()

        c = rx.counters()
        assert c.n_committed == 1
        assert c.ring_slots_written == 1
        assert c.ring_data_present_count == 1
        assert c.ring_pattern_mismatch_count == 0
        assert c.ring_zerofill_slot_count == 0
        assert c.ring_write_error_count == 0

        # Reader can pull the exact bytes back at (corr=0, dm=0, t_seq=0).
        reader = _attach_reader(name)
        try:
            got_payload, got_vf = reader.read_slot(
                corr=0, dm=0, t_seq=0, compute_half=0
            )
            assert got_vf & VF_DATA_PRESENT
            assert not (got_vf & VF_PATTERN_MISMATCH)
            assert got_payload == payload
        finally:
            reader.close()

    def test_write_seq_advances_one_per_commit(
        self, rx_with_ring, tx_sock
    ):
        rx, name = rx_with_ring
        rx.set_expected_pattern_id(0, PID_OK)
        payload = bytes(DEFAULT_PAYLOAD_BYTES)
        n_send = 32
        for seq in range(n_send):
            hdr = _hdr(seq=seq, chgroup=0, dm_idx=0,
                       payload_bytes_in_frag=len(payload))
            _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain(0.7)

        c = rx.counters()
        assert c.n_committed == n_send
        assert c.ring_slots_written == n_send
        reader = _attach_reader(name)
        try:
            # write_seq is per-corr; corr=0 only.
            assert reader.get_write_seq(0) == n_send
            # Other corr should be untouched.
            assert reader.get_write_seq(1) == 0
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# B. Pattern-mismatch publishes a stub slot
# ---------------------------------------------------------------------------


class TestPatternMismatchStub:
    def test_mismatch_publishes_vf_pattern_mismatch_empty_payload(
        self, rx_with_ring, tx_sock
    ):
        rx, name = rx_with_ring
        rx.set_expected_pattern_id(0, PID_OK)
        # Send a packet with the WRONG pattern_id.
        payload = bytes((0xAA,) * DEFAULT_PAYLOAD_BYTES)
        hdr = _hdr(seq=5, chgroup=0, dm_idx=0, pattern_id=PID_WRONG,
                   payload_bytes_in_frag=len(payload))
        _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
        _drain()

        c = rx.counters()
        # Phase A counter still bumps.
        assert c.pattern_mismatch_count == 1
        # Phase B counters: one ring publish, VF_PATTERN_MISMATCH branch.
        assert c.ring_slots_written == 1
        assert c.ring_pattern_mismatch_count == 1
        assert c.ring_data_present_count == 0
        # ingest_fragment was NOT called; n_committed stays 0.
        assert c.n_committed == 0

        reader = _attach_reader(name)
        try:
            got, vf = reader.read_slot(corr=0, dm=0, t_seq=5, compute_half=0)
            assert vf & VF_PATTERN_MISMATCH
            assert not (vf & VF_DATA_PRESENT)
            # Payload was zero-filled by rx_ring_write_slot (since we
            # passed NULL from C).
            assert got == b"\x00" * DEFAULT_PAYLOAD_BYTES
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# C. Window-slide publishes a hole (validity == 0)
# ---------------------------------------------------------------------------


class TestWindowSlideHole:
    def test_missing_fragment_publishes_zerofill(self, rx_with_ring, tx_sock):
        """When seq=N's window slides out without all fragments, the C
        path now publishes validity=0 to the ring at (corr, dm, N) so
        the consumer's write_seq advances and validity_mask drops t=N.
        """
        rx, name = rx_with_ring
        rx.set_expected_pattern_id(0, PID_OK)

        # 2-fragment payload setup.
        n_filled = 5000
        payload = bytes(n_filled * 2)
        frags = split_payload_into_fragments(
            payload, max_frag_payload_bytes=8964
        )
        assert len(frags) == 2

        # Send ONLY frag 0 of seq=0 (incomplete).
        hdr0 = _hdr(seq=0, chgroup=0, dm_idx=0, n_filled=n_filled,
                    n_frags=len(frags), frag_idx=0,
                    payload_bytes_in_frag=len(frags[0]),
                    flags=FLAG_QUANTIZED)
        _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr0, frags[0]))

        # Send full payloads for seq=1..4 to slide the window past seq=0.
        for seq in range(1, 5):
            for fi, f in enumerate(frags):
                flags = FLAG_QUANTIZED
                if fi == len(frags) - 1:
                    flags |= FLAG_LAST_IN_BLOCK
                hdr = _hdr(seq=seq, chgroup=0, dm_idx=0,
                           n_filled=n_filled, n_frags=len(frags),
                           frag_idx=fi,
                           payload_bytes_in_frag=len(f),
                           flags=flags)
                _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, f))
        _drain()

        c = rx.counters()
        assert c.n_committed == 4
        # At least one zerofill from seq=0 sliding out.
        assert c.window_slide_zerofill_count >= 1
        assert c.ring_zerofill_slot_count >= 1
        # ring_slots_written = 4 commits + at least 1 zerofill.
        assert c.ring_slots_written >= 5
        # Read seq=0 back from the ring: should be validity=0
        # (a "hole" — not data-present, not pattern-mismatch).
        reader = _attach_reader(name)
        try:
            _, vf = reader.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            assert vf == 0
            # Conversely, seq=1 should be data-present.
            _, vf1 = reader.read_slot(corr=0, dm=0, t_seq=1, compute_half=0)
            assert vf1 & VF_DATA_PRESENT
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# D. Multi-fragment concat ordering
# ---------------------------------------------------------------------------


class TestMultiFragmentConcat:
    def test_two_fragments_concatenated_in_frag_idx_order(
        self, shm_name, tx_sock
    ):
        """Two-fragment payload with distinguishable bytes per fragment.
        Verifies the ring slot contains frag 0 bytes followed by frag 1
        bytes (not the wire-arrival order)."""
        # Use larger n_filled_per_corr so two frags fit cleanly.
        dims = RxRingDims(
            n_corr=2, n_coarse_dm=1, t_buf_samples=32,
            n_filled_per_corr=5000, bytes_per_cell=2,
        )
        payload_bytes = dims.n_filled_per_corr * dims.bytes_per_cell  # 10000

        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        rx.attach_ring(
            shm_name=shm_name, owner=True,
            n_corr=dims.n_corr, n_coarse_dm=dims.n_coarse_dm,
            t_buf_samples=dims.t_buf_samples,
            n_filled=dims.n_filled_per_corr,
            bytes_per_cell=dims.bytes_per_cell,
        )
        rx.start()
        try:
            rx.set_expected_pattern_id(0, PID_OK)
            # Build a payload with frag 0 = 0xAA-pattern, frag 1 = 0x55-pattern.
            payload = bytes((0xAA,)) * (payload_bytes // 2) + \
                      bytes((0x55,)) * (payload_bytes - payload_bytes // 2)
            frags = split_payload_into_fragments(
                payload, max_frag_payload_bytes=8964
            )
            assert len(frags) == 2

            # Send frags out of arrival order to verify the concat uses
            # frag_idx ordering, not arrival order.
            for fi in [1, 0]:
                flags = FLAG_QUANTIZED
                if fi == len(frags) - 1:
                    flags |= FLAG_LAST_IN_BLOCK
                hdr = _hdr(seq=42, chgroup=0, dm_idx=0,
                           n_filled=dims.n_filled_per_corr,
                           n_frags=len(frags), frag_idx=fi,
                           payload_bytes_in_frag=len(frags[fi]),
                           flags=flags)
                _send(tx_sock, ("127.0.0.1", rx.port),
                      pack_frame(hdr, frags[fi]))
            _drain()

            reader = RxRing.mmap_attach_readonly(shm_name, dims)
            try:
                got, vf = reader.read_slot(corr=0, dm=0, t_seq=42,
                                           compute_half=0)
                assert vf & VF_DATA_PRESENT
                # The ring slot stores n_filled * bytes_per_cell bytes;
                # we sent exactly that many, so the round-trip is exact.
                assert got == payload
            finally:
                reader.close()
        finally:
            rx.close()


# ---------------------------------------------------------------------------
# E. Multi-chgroup routing inside one socket
# ---------------------------------------------------------------------------


class TestMultiChgroupRouting:
    def test_two_chgroups_route_to_distinct_ring_corrs(
        self, rx_with_ring, tx_sock
    ):
        """One UDP socket receives frames with hdr.chgroup ∈ {0, 1};
        ring should have writes in both corr slices."""
        rx, name = rx_with_ring
        rx.set_expected_pattern_id(0, PID_OK)
        rx.set_expected_pattern_id(1, PID_OK)
        payload_chg0 = bytes((0x11,) * DEFAULT_PAYLOAD_BYTES)
        payload_chg1 = bytes((0x22,) * DEFAULT_PAYLOAD_BYTES)
        for seq in range(4):
            for chg, p in [(0, payload_chg0), (1, payload_chg1)]:
                hdr = _hdr(seq=seq, chgroup=chg, dm_idx=0,
                           payload_bytes_in_frag=len(p))
                _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, p))
        _drain()

        c = rx.counters()
        assert c.n_committed == 8
        assert c.ring_slots_written == 8

        reader = _attach_reader(name)
        try:
            assert reader.get_write_seq(0) == 4
            assert reader.get_write_seq(1) == 4
            for seq in range(4):
                got0, _ = reader.read_slot(corr=0, dm=0, t_seq=seq,
                                           compute_half=0)
                got1, _ = reader.read_slot(corr=1, dm=0, t_seq=seq,
                                           compute_half=0)
                assert got0 == payload_chg0
                assert got1 == payload_chg1
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# F. Multi-port (M7.2 production topology) — add_port + drain both
# ---------------------------------------------------------------------------


class TestMultiPort:
    def test_add_port_increments_n_sockets_before_start(self):
        """add_port must work before start; after start it should reject."""
        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        try:
            assert rx.n_sockets == 1
            p2 = rx.add_port(bind_host="127.0.0.1", bind_port=0)
            assert rx.n_sockets == 2
            assert p2 != rx.port
            rx.start()
            with pytest.raises(RuntimeError):
                rx.add_port(bind_host="127.0.0.1", bind_port=0)
        finally:
            rx.close()

    def test_two_ports_one_drainer_both_route_to_ring(
        self, shm_name, tx_sock
    ):
        """Bind two UDP sockets, send to each with distinct chgroups;
        both writes should land in the ring."""
        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        try:
            port_a = rx.port
            port_b = rx.add_port(bind_host="127.0.0.1", bind_port=0)
            assert port_a != port_b
            rx.attach_ring(
                shm_name=shm_name, owner=True,
                n_corr=DEFAULT_DIMS.n_corr,
                n_coarse_dm=DEFAULT_DIMS.n_coarse_dm,
                t_buf_samples=DEFAULT_DIMS.t_buf_samples,
                n_filled=DEFAULT_DIMS.n_filled_per_corr,
                bytes_per_cell=DEFAULT_DIMS.bytes_per_cell,
            )
            rx.set_expected_pattern_id(0, PID_OK)
            rx.set_expected_pattern_id(1, PID_OK)
            rx.start()

            payload_a = bytes((0xAB,) * DEFAULT_PAYLOAD_BYTES)
            payload_b = bytes((0xCD,) * DEFAULT_PAYLOAD_BYTES)
            # Send chgroup=0 frames to port A.
            for seq in range(3):
                hdr = _hdr(seq=seq, chgroup=0, dm_idx=0,
                           payload_bytes_in_frag=len(payload_a))
                _send(tx_sock, ("127.0.0.1", port_a),
                      pack_frame(hdr, payload_a))
            # Send chgroup=1 frames to port B.
            for seq in range(3):
                hdr = _hdr(seq=seq, chgroup=1, dm_idx=0,
                           payload_bytes_in_frag=len(payload_b))
                _send(tx_sock, ("127.0.0.1", port_b),
                      pack_frame(hdr, payload_b))
            _drain()

            c = rx.counters()
            assert c.n_committed == 6
            assert c.ring_slots_written == 6

            reader = _attach_reader(shm_name)
            try:
                assert reader.get_write_seq(0) == 3
                assert reader.get_write_seq(1) == 3
                got_a, _ = reader.read_slot(corr=0, dm=0, t_seq=0,
                                            compute_half=0)
                got_b, _ = reader.read_slot(corr=1, dm=0, t_seq=0,
                                            compute_half=0)
                assert got_a == payload_a
                assert got_b == payload_b
            finally:
                reader.close()
        finally:
            rx.close()


# ---------------------------------------------------------------------------
# G. Lifecycle / API
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_attach_ring_after_start_rejected(self, shm_name):
        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        try:
            rx.start()
            with pytest.raises(RuntimeError):
                rx.attach_ring(
                    shm_name=shm_name, owner=True,
                    n_corr=4, n_coarse_dm=2, t_buf_samples=32,
                    n_filled=100, bytes_per_cell=2,
                )
        finally:
            rx.close()

    def test_double_attach_ring_rejected(self, shm_name):
        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        try:
            rx.attach_ring(
                shm_name=shm_name, owner=True,
                n_corr=4, n_coarse_dm=2, t_buf_samples=32,
                n_filled=100, bytes_per_cell=2,
            )
            with pytest.raises(RuntimeError):
                rx.attach_ring(
                    shm_name=shm_name + "-other", owner=True,
                    n_corr=4, n_coarse_dm=2, t_buf_samples=32,
                    n_filled=100, bytes_per_cell=2,
                )
        finally:
            rx.close()

    def test_close_tears_down_ring(self, shm_name):
        """After close() the segment is no longer attached on the
        producer side — a reopen on the same name should succeed."""
        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        rx.attach_ring(
            shm_name=shm_name, owner=True,
            n_corr=4, n_coarse_dm=2, t_buf_samples=32,
            n_filled=100, bytes_per_cell=2,
        )
        assert rx.ring_attached
        rx.close()
        # Reopen succeeds (no stale singleton).
        rx2 = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                           so_rcvbuf_bytes=8 * 1024 * 1024)
        try:
            assert not rx2.ring_attached
        finally:
            rx2.close()

    def test_no_ring_attached_phase_a_unchanged(self, tx_sock):
        """Phase A regression: with no ring attached, every counter
        behaves exactly as before — n_committed bumps, no ring_*
        counter is non-zero, and we don't crash on slot publish."""
        rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                          so_rcvbuf_bytes=8 * 1024 * 1024)
        try:
            rx.start()
            rx.set_expected_pattern_id(0, PID_OK)
            payload = bytes(200)
            hdr = _hdr(seq=0, payload_bytes_in_frag=len(payload))
            _send(tx_sock, ("127.0.0.1", rx.port), pack_frame(hdr, payload))
            _drain()
            c = rx.counters()
            assert c.n_committed == 1
            assert c.ring_slots_written == 0
            assert c.ring_data_present_count == 0
            assert not rx.ring_attached
        finally:
            rx.close()
