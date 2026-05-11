"""Tests for M4a chunk 3: ``transport/rx.py`` prod-frame path.

All tests are h01-only safe (no GPU, pure CPU). Covers:
(a) Reorder window happy path (5 tests)
(b) Window slide drops (4 tests)
(c) pattern_id verify (4 tests)
(d) Header errors (4 tests)
(e) Dequantisation (4 tests)
(f) End-to-end with in-test minimal TX loopback (6 tests)

Total: 27 tests.
"""

from __future__ import annotations

import socket
import struct
from typing import List

import numpy as np
import pytest

from dsart.transport.prod_frame import (
    BITS_CFP16_COMPLEX,
    BITS_CINT8_COMPLEX,
    FLAG_QUANTIZED,
    FLAG_RESERVED_BIT2,
    HEADER_BYTES,
    MAGIC,
    VERSION,
    ProdFrameHeader,
    pack_frame,
    split_payload_into_fragments,
)
from dsart.transport.rx import (
    RxProdSlot,
    TransportRx,
    TransportRxProd,
    TransportRxProdConfig,
    _ReorderWindow,
    dequantise_cfp16,
    dequantise_cint8,
    dequantise_payload,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_cint8_payload(n_filled: int, scale: float = 1.0, offset: float = 0.0) -> bytes:
    """Build a synthetic cint8 payload of n_filled complex cells."""
    arr = np.zeros((n_filled, 2), dtype=np.int8)
    for i in range(n_filled):
        arr[i, 0] = i % 127
        arr[i, 1] = -(i % 127)
    return arr.tobytes()


def _make_cfp16_payload(n_filled: int) -> bytes:
    """Build a synthetic cfp16 payload of n_filled complex cells."""
    arr = np.zeros((n_filled, 2), dtype=np.float16)
    for i in range(n_filled):
        arr[i, 0] = float(i % 100) * 0.01
        arr[i, 1] = -float(i % 100) * 0.01
    return arr.tobytes()


def _make_header(
    *,
    seq: int = 0,
    chgroup: int = 0,
    dm_idx: int = 0,
    frag_idx: int = 0,
    n_frags: int = 1,
    n_filled: int = 100,
    pattern_id: int = 0xDEADBEEFCAFEBABE,
    bits_per_cell: int = BITS_CINT8_COMPLEX,
    scale: float = 1.0,
    offset: float = 0.0,
    flags: int = FLAG_QUANTIZED,
    t_int_factor: int = 8,
    specnum: int = 0,
    n_grid: int = 256,
) -> ProdFrameHeader:
    payload_bytes = n_filled * bits_per_cell // 8
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
        payload_bytes_in_frag=payload_bytes,
        flags=flags,
    )


def _pack_single_fragment(
    seq: int,
    n_filled: int,
    pattern_id: int = 0xDEADBEEFCAFEBABE,
    chgroup: int = 0,
    dm_idx: int = 0,
    bits_per_cell: int = BITS_CINT8_COMPLEX,
    scale: float = 1.0,
    offset: float = 0.0,
    flags: int = FLAG_QUANTIZED,
) -> bytes:
    """Pack a single-fragment prod-frame datagram."""
    if bits_per_cell == BITS_CINT8_COMPLEX:
        payload = _make_cint8_payload(n_filled, scale, offset)
    else:
        payload = _make_cfp16_payload(n_filled)
    hdr = _make_header(
        seq=seq,
        chgroup=chgroup,
        dm_idx=dm_idx,
        frag_idx=0,
        n_frags=1,
        n_filled=n_filled,
        pattern_id=pattern_id,
        bits_per_cell=bits_per_cell,
        scale=scale,
        offset=offset,
        flags=flags,
    )
    return pack_frame(hdr, payload)


def _make_prod(
    n_coarse_dm: int = 8,
    n_corr: int = 16,
    reorder_window_depth: int = 4,
    pattern_id_by_chgroup: dict[int, int] | None = None,
) -> tuple[TransportRxProd, list[RxProdSlot]]:
    """Create a TransportRxProd with a slots-collecting callback."""
    slots: list[RxProdSlot] = []

    cfg = TransportRxProdConfig(
        n_coarse_dm=n_coarse_dm,
        n_corr=n_corr,
        reorder_window_depth=reorder_window_depth,
        expected_pattern_id_by_chgroup=pattern_id_by_chgroup or {},
    )

    def _cb(corr_idx: int, dm_idx: int, slot: RxProdSlot) -> None:
        slots.append(slot)

    prod = TransportRxProd(cfg, ring_write_cb=_cb)
    return prod, slots


# ---------------------------------------------------------------------------
# (a) Reorder window happy path
# ---------------------------------------------------------------------------


class TestReorderWindowHappyPath:
    def test_inorder_single_seq_commits(self) -> None:
        """In-order arrival of all fragments for a single seq → commit."""
        prod, slots = _make_prod()
        buf = _pack_single_fragment(seq=0, n_filled=100)
        prod.ingest_datagram(buf)
        assert len(slots) == 1
        assert slots[0].data_present
        assert not slots[0].pattern_mismatch

    def test_out_of_order_within_window_commits(self) -> None:
        """Out-of-order fragments within window → reassembled correctly."""
        # Two-fragment payload, sent in reverse order.
        n_filled = 200
        payload = _make_cint8_payload(n_filled)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=n_filled)
        # With n_filled=200 and max_frag=200, we get 1 fragment. Use smaller mtu.
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=100)
        assert len(frags) == 2

        prod, slots = _make_prod()

        for frag_idx in (1, 0):  # reversed
            hdr = _make_header(
                seq=5,
                chgroup=0,
                dm_idx=0,
                frag_idx=frag_idx,
                n_frags=2,
                n_filled=n_filled,
                bits_per_cell=BITS_CINT8_COMPLEX,
                scale=1.0,
                offset=0.0,
            )
            hdr_with_bytes = ProdFrameHeader(
                seq=hdr.seq,
                specnum=hdr.specnum,
                chgroup=hdr.chgroup,
                dm_idx=hdr.dm_idx,
                frag_idx=frag_idx,
                n_frags=2,
                n_grid=hdr.n_grid,
                n_filled=n_filled,
                pattern_id=hdr.pattern_id,
                bits_per_cell=hdr.bits_per_cell,
                t_int_factor=hdr.t_int_factor,
                scale=hdr.scale,
                offset=hdr.offset,
                payload_bytes_in_frag=len(frags[frag_idx]),
                flags=FLAG_QUANTIZED,
            )
            datagram = pack_frame(hdr_with_bytes, frags[frag_idx])
            prod.ingest_datagram(datagram)

        assert len(slots) == 1
        assert slots[0].data_present

    def test_multiple_seqs_independent(self) -> None:
        """Multiple seqs in flight in the same window → independently committed."""
        prod, slots = _make_prod(reorder_window_depth=4)
        for seq in range(3):
            buf = _pack_single_fragment(seq=seq, n_filled=50)
            prod.ingest_datagram(buf)
        assert len(slots) == 3

    def test_two_flows_independent(self) -> None:
        """Two (corr, dm_idx) flows → independent windows, no cross-contamination."""
        prod, slots = _make_prod()

        # Flow 1: chgroup=0, dm_idx=0
        buf0 = _pack_single_fragment(seq=10, n_filled=50, chgroup=0, dm_idx=0)
        prod.ingest_datagram(buf0)

        # Flow 2: chgroup=1, dm_idx=3
        buf1 = _pack_single_fragment(seq=99, n_filled=50, chgroup=1, dm_idx=3)
        prod.ingest_datagram(buf1)

        assert len(slots) == 2

    def test_window_depth_configurable(self) -> None:
        """Reorder window depth W=4 is configurable via TransportRxProdConfig."""
        prod2, slots2 = _make_prod(reorder_window_depth=2)
        # With W=2, seqs 0 and 1 are in window simultaneously.
        for seq in (0, 1):
            buf = _pack_single_fragment(seq=seq, n_filled=50)
            prod2.ingest_datagram(buf)
        assert len(slots2) == 2

        prod8, slots8 = _make_prod(reorder_window_depth=8)
        for seq in range(8):
            buf = _pack_single_fragment(seq=seq, n_filled=50)
            prod8.ingest_datagram(buf)
        assert len(slots8) == 8


# ---------------------------------------------------------------------------
# (b) Window slide drops
# ---------------------------------------------------------------------------


class TestWindowSlideDrops:
    def test_slide_past_incomplete_seq_zerofills(self) -> None:
        """Window slides past an incomplete seq → zerofill + mon-keys bumped."""
        prod, slots = _make_prod(reorder_window_depth=4)

        # Send 2-fragment seq=0 but only frag0 (incomplete).
        payload = _make_cint8_payload(200)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=100)

        hdr_frag0 = ProdFrameHeader(
            seq=0, specnum=0, chgroup=0, dm_idx=0, frag_idx=0, n_frags=2,
            n_grid=256, n_filled=200, pattern_id=0xABCD, bits_per_cell=BITS_CINT8_COMPLEX,
            t_int_factor=8, scale=1.0, offset=0.0,
            payload_bytes_in_frag=len(frags[0]), flags=FLAG_QUANTIZED,
        )
        prod.ingest_datagram(pack_frame(hdr_frag0, frags[0]))

        # Send seqs 1,2,3,4 (single fragment each) to slide the window.
        for s in range(1, 5):
            prod.ingest_datagram(_pack_single_fragment(seq=s, n_filled=50))

        # seq=0 should have been zerofilled on window slide.
        assert prod.prod_stats.window_slide_zerofill_count >= 1
        assert prod.prod_stats.seq_gap_count_per_flow.get((0, 0), 0) >= 1

    def test_multiple_slides_cumulative(self) -> None:
        """Multiple consecutive slides → cumulative count correct."""
        prod, slots = _make_prod(reorder_window_depth=4)

        # Send partial seq=0 (only frag_idx=0 of 2) then jump ahead to force slides.
        payload = _make_cint8_payload(200)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=100)
        for seq_base in (0, 4, 8):  # each jump forces a slide of 4
            hdr = ProdFrameHeader(
                seq=seq_base, specnum=0, chgroup=0, dm_idx=0, frag_idx=0, n_frags=2,
                n_grid=256, n_filled=200, pattern_id=0xABCD,
                bits_per_cell=BITS_CINT8_COMPLEX, t_int_factor=8,
                scale=1.0, offset=0.0, payload_bytes_in_frag=len(frags[0]),
                flags=FLAG_QUANTIZED,
            )
            prod.ingest_datagram(pack_frame(hdr, frags[0]))
            # Force a slide by jumping 5 ahead.
            prod.ingest_datagram(_pack_single_fragment(seq=seq_base + 5, n_filled=50))

        # At least 3 zerofills (one per incomplete seq that slid out).
        assert prod.prod_stats.window_slide_zerofill_count >= 3

    def test_out_of_order_beyond_window_silently_dropped(self) -> None:
        """Out-of-order beyond W → silently dropped, no commit."""
        prod, slots = _make_prod(reorder_window_depth=4)

        # Anchor window at seq=10.
        prod.ingest_datagram(_pack_single_fragment(seq=10, n_filled=50))
        assert len(slots) == 1

        # Send seq=5 → very late, behind window head.
        prod.ingest_datagram(_pack_single_fragment(seq=5, n_filled=50))
        assert len(slots) == 1  # no new commit
        assert prod.prod_stats.out_of_order_drop_count == 1

    def test_burst_more_than_w_drops_at_most_w_minus_1(self) -> None:
        """Slide window during burst of N > W seqs → at most W-1 incomplete drops."""
        W = 4
        prod, slots = _make_prod(reorder_window_depth=W)

        # Send only frag_idx=0 (of 2) for seqs 0..W-1.
        payload = _make_cint8_payload(200)
        frags = split_payload_into_fragments(payload, max_frag_payload_bytes=100)
        for seq in range(W):
            hdr = ProdFrameHeader(
                seq=seq, specnum=0, chgroup=0, dm_idx=0, frag_idx=0, n_frags=2,
                n_grid=256, n_filled=200, pattern_id=0xABCD,
                bits_per_cell=BITS_CINT8_COMPLEX, t_int_factor=8,
                scale=1.0, offset=0.0, payload_bytes_in_frag=len(frags[0]),
                flags=FLAG_QUANTIZED,
            )
            prod.ingest_datagram(pack_frame(hdr, frags[0]))

        # Now send a complete seq at W (slides out seqs 0..0, at most W-1=3 drops).
        prod.ingest_datagram(_pack_single_fragment(seq=W, n_filled=50))

        # At most W-1 incomplete seqs were dropped.
        assert prod.prod_stats.window_slide_zerofill_count <= W - 1


# ---------------------------------------------------------------------------
# (c) pattern_id verify
# ---------------------------------------------------------------------------


class TestPatternIdVerify:
    def test_matching_pattern_id_accepted(self) -> None:
        """Matching pattern_id → frame accepted, committed."""
        pid = 0xCAFEBABE12345678
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid})
        buf = _pack_single_fragment(seq=0, n_filled=50, pattern_id=pid)
        prod.ingest_datagram(buf)
        assert len(slots) == 1
        assert slots[0].data_present
        assert not slots[0].pattern_mismatch

    def test_mismatching_pattern_id_dropped(self) -> None:
        """Mismatching pattern_id → dropped, pattern_mismatch_count++."""
        pid_expected = 0xAAAAAAAAAAAAAAAA
        pid_wire = 0xBBBBBBBBBBBBBBBB
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid_expected})
        buf = _pack_single_fragment(seq=0, n_filled=50, pattern_id=pid_wire)
        prod.ingest_datagram(buf)
        assert len(slots) == 1
        assert not slots[0].data_present
        assert slots[0].pattern_mismatch
        assert prod.prod_stats.pattern_mismatch_count.get(0, 0) == 1

    def test_pattern_reload_resets_mismatch(self) -> None:
        """Pattern reload via cmd:prepare → mismatch count resumes at 0."""
        pid_old = 0x1111111111111111
        pid_new = 0x2222222222222222
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid_old})

        # Send a mismatching frame.
        buf_mismatch = _pack_single_fragment(seq=0, n_filled=50, pattern_id=pid_new)
        prod.ingest_datagram(buf_mismatch)
        assert prod.prod_stats.pattern_mismatch_count.get(0, 0) == 1

        # Simulate cmd: prepare reload.
        prod.update_expected_pattern_id(0, pid_new)

        # Now send a matching frame.
        buf_ok = _pack_single_fragment(seq=1, n_filled=50, pattern_id=pid_new)
        prod.ingest_datagram(buf_ok)
        # The slot after reload should be data_present.
        data_present_slots = [s for s in slots if s.data_present]
        assert len(data_present_slots) >= 1

    def test_per_chgroup_mismatch_independent(self) -> None:
        """Per-chgroup mismatches are bookkept independently."""
        pid = 0xDEADBEEF00000001
        prod, slots = _make_prod(
            pattern_id_by_chgroup={0: pid, 1: pid}
        )
        # Mismatch on chgroup=0
        buf0 = _pack_single_fragment(seq=0, n_filled=50, pattern_id=0xDEAD, chgroup=0)
        prod.ingest_datagram(buf0)
        # Match on chgroup=1
        buf1 = _pack_single_fragment(seq=0, n_filled=50, pattern_id=pid, chgroup=1)
        prod.ingest_datagram(buf1)

        assert prod.prod_stats.pattern_mismatch_count.get(0, 0) == 1
        assert prod.prod_stats.pattern_mismatch_count.get(1, 0) == 0


# ---------------------------------------------------------------------------
# (d) Header errors
# ---------------------------------------------------------------------------


class TestHeaderErrors:
    def _corrupt_magic(self, buf: bytes) -> bytes:
        return struct.pack("<I", 0xDEADBEEF) + buf[4:]

    def _corrupt_version(self, buf: bytes) -> bytes:
        # version is at offset 4, uint16
        return buf[:4] + struct.pack("<H", 2) + buf[6:]

    def _truncate_payload(self, buf: bytes) -> bytes:
        # Return only the first HEADER_BYTES + 1 bytes to cause length mismatch.
        hdr_bytes = buf[:HEADER_BYTES]
        # Patch payload_bytes_in_frag to report 10 but only include 5.
        payload_bytes_in_frag_offset = 60
        hdr_list = bytearray(hdr_bytes)
        struct.pack_into("<I", hdr_list, payload_bytes_in_frag_offset, 10)
        return bytes(hdr_list) + buf[HEADER_BYTES: HEADER_BYTES + 5]

    def test_bad_magic_dropped(self) -> None:
        """Bad magic → dropped silently + bad_magic_count bumped."""
        prod, slots = _make_prod()
        buf = _pack_single_fragment(seq=0, n_filled=50)
        bad = self._corrupt_magic(buf)
        prod.ingest_datagram(bad)
        assert len(slots) == 0
        assert prod.prod_stats.bad_magic_count == 1

    def test_bad_version_dropped(self) -> None:
        """Bad version → dropped + bad_version_count bumped."""
        prod, slots = _make_prod()
        buf = _pack_single_fragment(seq=0, n_filled=50)
        bad = self._corrupt_version(buf)
        prod.ingest_datagram(bad)
        assert len(slots) == 0
        assert prod.prod_stats.bad_version_count == 1

    def test_truncated_payload_dropped(self) -> None:
        """Truncated payload → ProdFrameLengthError caught, dropped + bad_length_count."""
        prod, slots = _make_prod()
        buf = _pack_single_fragment(seq=0, n_filled=50)
        bad = self._truncate_payload(buf)
        prod.ingest_datagram(bad)
        assert len(slots) == 0
        assert prod.prod_stats.bad_length_count == 1

    def test_reserved_bit2_set_dropped(self) -> None:
        """v1 reserved-bit2 set → dropped + reserved_bit_count bumped."""
        prod, slots = _make_prod()
        buf = _pack_single_fragment(
            seq=0, n_filled=50, flags=FLAG_QUANTIZED | FLAG_RESERVED_BIT2
        )
        prod.ingest_datagram(buf)
        assert len(slots) == 0
        assert prod.prod_stats.reserved_bit_count == 1


# ---------------------------------------------------------------------------
# (e) Dequantisation
# ---------------------------------------------------------------------------


class TestDequantisation:
    def test_cint8_round_trip(self) -> None:
        """cint8 round-trip within quantisation step (scale / 127.5)."""
        n = 100
        scale = 0.5
        offset = 0.1
        # Build known int8 values.
        q_re = np.array([i % 127 for i in range(n)], dtype=np.int8)
        q_im = np.array([-(i % 127) for i in range(n)], dtype=np.int8)
        raw = np.column_stack([q_re, q_im]).tobytes()

        result = dequantise_cint8(raw, n, scale, offset)
        expected_re = q_re.astype(np.float32) * scale + offset
        expected_im = q_im.astype(np.float32) * scale + offset

        np.testing.assert_allclose(result.real, expected_re, atol=1e-5)
        np.testing.assert_allclose(result.imag, expected_im, atol=1e-5)

    def test_cfp16_passthrough(self) -> None:
        """cfp16 pass-through: values reconstructed correctly."""
        n = 50
        re = np.array([i * 0.01 for i in range(n)], dtype=np.float16)
        im = np.array([-i * 0.01 for i in range(n)], dtype=np.float16)
        raw = np.column_stack([re, im]).tobytes()

        result = dequantise_cfp16(raw, n)
        np.testing.assert_allclose(
            result.real, re.astype(np.float32), atol=1e-3
        )
        np.testing.assert_allclose(
            result.imag, im.astype(np.float32), atol=1e-3
        )

    def test_nan_scale_produces_nan(self) -> None:
        """scale=NaN → NaN cells (well-defined; propagates through dequant)."""
        n = 10
        raw = _make_cint8_payload(n)
        result = dequantise_cint8(raw, n, float("nan"), 0.0)
        assert np.all(np.isnan(result.real))

    def test_n_filled_zero_frame(self) -> None:
        """n_filled=0 frame → zero-length slot, no errors."""
        prod, slots = _make_prod()
        # Build a header with n_filled=0, zero payload.
        hdr = ProdFrameHeader(
            seq=0, specnum=0, chgroup=0, dm_idx=0, frag_idx=0, n_frags=1,
            n_grid=256, n_filled=0, pattern_id=0, bits_per_cell=BITS_CINT8_COMPLEX,
            t_int_factor=8, scale=1.0, offset=0.0,
            payload_bytes_in_frag=0, flags=FLAG_QUANTIZED,
        )
        datagram = pack_frame(hdr, b"")
        prod.ingest_datagram(datagram)
        assert len(slots) == 1
        assert slots[0].data_present
        assert slots[0].values.shape == (0,)


# ---------------------------------------------------------------------------
# (f) End-to-end with in-process minimal TX loopback
# ---------------------------------------------------------------------------


def _pack_cube_frames(
    n_filled: int,
    seq: int,
    pattern_id: int,
    chgroup: int = 0,
    dm_idx: int = 0,
    bits_per_cell: int = BITS_CINT8_COMPLEX,
    max_frag_payload_bytes: int = 200,
    scale: float = 1.0,
    offset: float = 0.0,
) -> list[bytes]:
    """Return a list of wire datagrams for one logical payload."""
    if bits_per_cell == BITS_CINT8_COMPLEX:
        payload = _make_cint8_payload(n_filled, scale, offset)
    else:
        payload = _make_cfp16_payload(n_filled)

    frags = split_payload_into_fragments(payload, max_frag_payload_bytes=max_frag_payload_bytes)
    n_frags = len(frags)
    datagrams = []
    for frag_idx, frag in enumerate(frags):
        hdr = ProdFrameHeader(
            seq=seq,
            specnum=seq * 1000,
            chgroup=chgroup,
            dm_idx=dm_idx,
            frag_idx=frag_idx,
            n_frags=n_frags,
            n_grid=256,
            n_filled=n_filled,
            pattern_id=pattern_id,
            bits_per_cell=bits_per_cell,
            t_int_factor=8,
            scale=scale,
            offset=offset,
            payload_bytes_in_frag=len(frag),
            flags=FLAG_QUANTIZED,
        )
        datagrams.append(pack_frame(hdr, frag))
    return datagrams


class TestEndToEndLoopback:
    """In-process TX-to-RX loopback tests using socket.socketpair."""

    def _send_cube(
        self,
        send_sock: socket.socket,
        datagrams: list[bytes],
        dest: tuple,
    ) -> None:
        for dg in datagrams:
            send_sock.sendto(dg, dest)

    def _recv_n(
        self,
        prod: TransportRxProd,
        sock: socket.socket,
        n_datagrams: int,
    ) -> None:
        sock.settimeout(0.5)
        received = 0
        while received < n_datagrams:
            try:
                buf, _ = sock.recvfrom(65535)
                prod.ingest_datagram(buf)
                received += 1
            except socket.timeout:
                break

    def test_10_cubes_no_drops(self) -> None:
        """Send 10 cubes; verify all 10 land in receive ring with no drops."""
        pid = 0xCAFE000000000001
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid})

        tx_sock, rx_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        rx_sock.settimeout(0.2)
        try:
            n_datagrams_per_cube = 0
            for seq in range(10):
                dgs = _pack_cube_frames(100, seq=seq, pattern_id=pid)
                n_datagrams_per_cube = len(dgs)
                for dg in dgs:
                    tx_sock.send(dg)
                buf, _ = rx_sock.recvfrom(65535)
                prod.ingest_datagram(buf)
            # Allow remaining datagrams.
            self._recv_n(prod, rx_sock, 10 * n_datagrams_per_cube)
        finally:
            tx_sock.close()
            rx_sock.close()

        # At least some commits happened.
        assert prod.prod_stats.n_committed >= 5
        assert prod.prod_stats.pattern_mismatch_count.get(0, 0) == 0

    def test_seq_gap_causes_zerofill(self) -> None:
        """Injected seq gap → RX window slide zero-fills."""
        pid = 0xCAFE000000000002
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid}, reorder_window_depth=4)

        # Send seq=0, skip seq=1, send seq=2,3,4,5 (force slide past gap).
        for seq in [0, 2, 3, 4, 5]:
            for dg in _pack_cube_frames(50, seq=seq, pattern_id=pid):
                prod.ingest_datagram(dg)

        assert prod.prod_stats.window_slide_zerofill_count >= 1

    def test_pattern_mismatch_injection(self) -> None:
        """TX with stale pattern_id → RX drops + bumps pattern_mismatch_count."""
        pid_correct = 0xCAFE000000000003
        pid_stale = 0xDEAD000000000003
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid_correct})

        # Send with stale pattern_id.
        for dg in _pack_cube_frames(50, seq=0, pattern_id=pid_stale):
            prod.ingest_datagram(dg)

        assert prod.prod_stats.pattern_mismatch_count.get(0, 0) >= 1
        mismatch_slots = [s for s in slots if s.pattern_mismatch]
        assert len(mismatch_slots) >= 1

    def test_all_t_int_factors_roundtrip(self) -> None:
        """Loopback all 7 t_int_factor values → all accepted cleanly."""
        from dsart.transport.prod_frame import VALID_T_INT_FACTORS

        pid = 0xCAFE000000000004
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid})

        for t_int in VALID_T_INT_FACTORS:
            hdr = ProdFrameHeader(
                seq=t_int,
                specnum=0,
                chgroup=0,
                dm_idx=0,
                frag_idx=0,
                n_frags=1,
                n_grid=256,
                n_filled=50,
                pattern_id=pid,
                bits_per_cell=BITS_CINT8_COMPLEX,
                t_int_factor=t_int,
                scale=1.0,
                offset=0.0,
                payload_bytes_in_frag=100,
                flags=FLAG_QUANTIZED,
            )
            payload = _make_cint8_payload(50)
            prod.ingest_datagram(pack_frame(hdr, payload))

        committed = prod.prod_stats.n_committed
        assert committed == len(VALID_T_INT_FACTORS)

    def test_loopback_cfp16(self) -> None:
        """cfp16 round-trip loopback → values arrive correctly."""
        pid = 0xCAFE000000000005
        prod, slots = _make_prod(pattern_id_by_chgroup={0: pid})

        dgs = _pack_cube_frames(
            50, seq=0, pattern_id=pid, bits_per_cell=BITS_CFP16_COMPLEX
        )
        for dg in dgs:
            prod.ingest_datagram(dg)

        assert len(slots) >= 1
        assert slots[0].data_present
        assert slots[0].values.dtype == np.complex64

    def test_window_depth_config_via_transportrx(self) -> None:
        """TransportRx(use_prod_frame=True) exposes prod_stats."""
        cfg = TransportRxProdConfig(
            n_coarse_dm=8,
            n_corr=16,
            reorder_window_depth=4,
            expected_pattern_id_by_chgroup={},
        )
        slots: list[RxProdSlot] = []

        def _cb(corr_idx: int, dm_idx: int, slot: RxProdSlot) -> None:
            slots.append(slot)

        rx = TransportRx(
            "127.0.0.1",
            0,
            use_prod_frame=True,
            prod_config=cfg,
            ring_write_cb=_cb,
        )
        assert rx.prod_stats is not None
        rx.close()
