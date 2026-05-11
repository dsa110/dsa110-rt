"""Tests for M4a chunk-2 production TX path.

Groups:
  (a) Token bucket (4 tests)
  (b) _transmit_one_cube_prod happy path (6 tests)
  (c) Quantisation + scale/offset (4 tests)
  (d) Headers + flags (4 tests)
  (e) Pacer + mon-keys (4 tests)
  (f) Integration with chunk-8 path (3 tests)
  (extra) Edge cases / robustness (10 tests)

All tests are pure CPU; no GPU required.
All sockets are AF_INET SOCK_DGRAM over 127.0.0.1 loopback.
"""
from __future__ import annotations

import socket
import struct
import time
import unittest.mock as mock
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from dsart.transport.frame import MAGIC as CHUNK8_MAGIC
from dsart.transport.prod_frame import (
    BITS_CINT8_COMPLEX,
    BITS_CFP16_COMPLEX,
    DEFAULT_MAX_FRAG_PAYLOAD_BYTES,
    FLAG_LAST_IN_BLOCK,
    FLAG_QUANTIZED,
    FLAG_RFI_WARMING_UP,
    HEADER_BYTES,
    MAGIC as PROD_MAGIC,
    ProdFrameHeader,
    pack_frame,
    unpack_frame,
)
from dsart.transport.tx import (
    TransportTx,
    TransportTxProdConfig,
    _TokenBucket,
)

# ---------------------------------------------------------------------------
# Constants shared by tests
# ---------------------------------------------------------------------------

_FAKE_PATTERN_ID = 0xDEADBEEFCAFEBABE
_FAKE_N_GRID = 256
_FAKE_SPECNUM = 0x0102030405060708
_RECV_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_sparse_cube(
    n_dm: int,
    n_fv: int,
    n_filled: int,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Synthetic sparse-COO cube ``(N_DM, n_fv, N_filled)`` complex64."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dm, n_fv, n_filled, 2)).astype(np.float32)
    c = (data[..., 0] + 1j * data[..., 1]).astype(np.complex64)
    return torch.from_numpy(c)


def _make_rx_sock() -> tuple[socket.socket, int]:
    """Bind a UDP recv socket to an ephemeral loopback port.

    Returns ``(rx_sock, rx_port)``.
    """
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(_RECV_TIMEOUT_S)
    return rx, rx.getsockname()[1]


def _make_prod_tx(
    rx_port: int,
    chgroup: int = 3,
    *,
    target_gbps: float = 100.0,
    bits_per_cell: int = BITS_CINT8_COMPLEX,
    t_int_factor: int = 1,
    bucket_fifo_depth: int = 16,
    pacer_headroom: float = 1.05,
) -> TransportTx:
    """Build a :class:`TransportTx` with ``use_prod_frame=True`` sending
    to ``127.0.0.1:rx_port``. Calls :meth:`prepare_prod` automatically."""
    cfg = TransportTxProdConfig(
        target_gbps_per_flow=target_gbps,
        pacer_headroom=pacer_headroom,
        bits_per_cell=bits_per_cell,
        t_int_factor=t_int_factor,
        bucket_fifo_depth=bucket_fifo_depth,
    )
    tx = TransportTx(
        "127.0.0.1",
        rx_port,
        chgroup=chgroup,
        use_prod_frame=True,
        prod_config=cfg,
    )
    tx.prepare_prod({chgroup: _FAKE_PATTERN_ID}, _FAKE_N_GRID)
    return tx


def _recv_all_frames(
    rx_sock: socket.socket,
    expected: int,
) -> list[tuple[ProdFrameHeader, bytes]]:
    """Drain up to ``expected`` ProdFrame datagrams from ``rx_sock``."""
    frames = []
    for _ in range(expected):
        try:
            data = rx_sock.recv(65536)
            frames.append(unpack_frame(data))
        except socket.timeout:
            break
    return frames


def _make_udp_send_sock() -> tuple[socket.socket, tuple[str, int]]:
    """Return an unbound UDP send socket and its getsockname address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    return sock, sock.getsockname()


# ---------------------------------------------------------------------------
# (a) Token bucket tests
# ---------------------------------------------------------------------------


class TestTokenBucket:
    """Group (a): _TokenBucket unit tests (4 tests)."""

    def test_a1_empty_bucket_drops_and_increments_counter(self):
        """Empty bucket (FIFO full) → send returns False and drop_count ++."""
        bucket = _TokenBucket(
            rate_bytes_per_sec=0.0,   # never refills
            capacity_bytes=100,
            max_fifo=2,
        )
        bucket._balance = 0.0  # drain
        send_sock, _ = _make_udp_send_sock()
        rx_sock, rx_port = _make_rx_sock()
        try:
            addr = ("127.0.0.1", rx_port)
            wire = b"x" * 20
            # Fill FIFO to max_fifo=2.
            bucket.try_send(wire, send_sock, addr)
            bucket.try_send(wire, send_sock, addr)
            assert len(bucket._fifo) == 2, "FIFO should be at capacity"

            # Next send triggers drop-oldest.
            initial_drops = bucket.drop_count
            result = bucket.try_send(wire, send_sock, addr)
            assert not result, "Expected False when bucket empty and FIFO full"
            assert bucket.drop_count == initial_drops + 1, (
                "drop_count should have incremented"
            )
        finally:
            send_sock.close()
            rx_sock.close()

    def test_a2_refill_over_time_send_succeeds(self):
        """After elapsed time tokens refill → send returns True."""
        rate = 1_000_000.0  # 1 MB/s
        capacity = 1000
        bucket = _TokenBucket(rate_bytes_per_sec=rate, capacity_bytes=capacity)
        bucket._balance = 0.0
        bucket._fifo.clear()

        send_sock, _ = _make_udp_send_sock()
        rx_sock, rx_port = _make_rx_sock()
        try:
            addr = ("127.0.0.1", rx_port)
            wire = b"x" * 100  # small payload
            start_ns = time.monotonic_ns()
            advance_ns = int(0.01 * 1e9)  # 10 ms → adds 10000 tokens

            with mock.patch("dsart.transport.tx.time") as mock_time:
                mock_time.monotonic_ns.return_value = start_ns
                bucket._last_ns = start_ns
                bucket._balance = 0.0

                # Advance time 10 ms and try again.
                mock_time.monotonic_ns.return_value = start_ns + advance_ns
                result = bucket.try_send(wire, send_sock, addr)

            assert result is True, (
                "Expected True (sent) after time elapsed and bucket refilled"
            )
        finally:
            send_sock.close()
            rx_sock.close()

    def test_a3_queue_full_drops_oldest_never_raises(self):
        """Queue-full → oldest item dropped, no exception raised."""
        bucket = _TokenBucket(
            rate_bytes_per_sec=0.0,  # never refills
            capacity_bytes=100,
            max_fifo=3,
        )
        bucket._balance = 0.0

        send_sock, _ = _make_udp_send_sock()
        rx_sock, rx_port = _make_rx_sock()
        try:
            addr = ("127.0.0.1", rx_port)
            items = [bytes([i] * 10) for i in range(5)]
            # Fill FIFO to capacity.
            for item in items[:3]:
                bucket.try_send(item, send_sock, addr)
            assert len(bucket._fifo) == 3
            oldest_in_fifo = bucket._fifo[0][0]
            assert oldest_in_fifo == items[0]

            # Submit items[3] → drops items[0] (oldest).
            initial_drops = bucket.drop_count
            try:
                bucket.try_send(items[3], send_sock, addr)
            except Exception as exc:
                pytest.fail(f"try_send raised unexpectedly: {exc}")

            assert bucket.drop_count == initial_drops + 1
            fifo_contents = [entry[0] for entry in bucket._fifo]
            assert items[0] not in fifo_contents, "Oldest item should be gone"
            assert items[3] in fifo_contents, "New item should be queued"
        finally:
            send_sock.close()
            rx_sock.close()

    def test_a4_drop_count_monotone_increments(self):
        """drop_count only ever increases (monotone)."""
        bucket = _TokenBucket(
            rate_bytes_per_sec=0.0,
            capacity_bytes=10,
            max_fifo=1,
        )
        send_sock, _ = _make_udp_send_sock()
        rx_sock, rx_port = _make_rx_sock()
        try:
            addr = ("127.0.0.1", rx_port)
            counts = []
            for _ in range(10):
                bucket._balance = 0.0
                bucket.try_send(b"x" * 20, send_sock, addr)
                counts.append(bucket.drop_count)
            for i in range(1, len(counts)):
                assert counts[i] >= counts[i - 1], (
                    f"drop_count decreased at step {i}: "
                    f"{counts[i - 1]} → {counts[i]}"
                )
            assert counts[-1] >= 1, "Should have at least one drop"
        finally:
            send_sock.close()
            rx_sock.close()


# ---------------------------------------------------------------------------
# (b) _transmit_one_cube_prod happy path
# ---------------------------------------------------------------------------


class TestTransmitOneCubeProd:
    """Group (b): _transmit_one_cube_prod happy path (6 tests)."""

    def test_b1_single_fragment_round_trip(self):
        """Single-fragment cube round-trips through UDP loopback correctly."""
        n_filled = 10  # tiny → fits in one fragment
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port, t_int_factor=8)
        try:
            cube = _make_sparse_cube(1, 1, n_filled)
            n_sent = tx._transmit_one_cube_prod(
                cube, specnum=_FAKE_SPECNUM, flags=0,
            )
            assert n_sent >= 1

            data = rx_sock.recv(65536)
            hdr, payload = unpack_frame(data)

            assert hdr.chgroup == 3
            assert hdr.n_grid == _FAKE_N_GRID
            assert hdr.n_filled == n_filled
            assert hdr.n_frags == 1
            assert hdr.frag_idx == 0
            assert hdr.specnum == _FAKE_SPECNUM
            assert hdr.pattern_id == _FAKE_PATTERN_ID
            assert hdr.t_int_factor == 8
            assert len(payload) == hdr.payload_bytes_in_frag
        finally:
            tx.close()
            rx_sock.close()

    def test_b2_multi_fragment_cube_matching_seq_ascending_frag_idx(self):
        """Multi-fragment cube: all frags share seq, frag_idx is ascending."""
        # 6000 cells × 2 B = 12000 B → 2 fragments at default MTU (8964 B)
        n_filled = 6000
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, n_filled)
            tx._transmit_one_cube_prod(cube, specnum=_FAKE_SPECNUM, flags=0)

            frames = _recv_all_frames(rx_sock, 4)
            assert len(frames) >= 2, f"Expected ≥2 fragments, got {len(frames)}"

            seq_set = {hdr.seq for hdr, _ in frames}
            assert len(seq_set) == 1, f"All frags must share one seq; got {seq_set}"

            frag_idxs = sorted(hdr.frag_idx for hdr, _ in frames)
            assert frag_idxs == list(range(len(frames)))

            n_frags_set = {hdr.n_frags for hdr, _ in frames}
            assert len(n_frags_set) == 1
            assert list(n_frags_set)[0] == len(frames)
        finally:
            tx.close()
            rx_sock.close()

    def test_b3_seq_monotone_per_dm_idx_independent_counters(self):
        """seq is monotone per dm_idx; distinct dm_idxs have independent counters."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            # Cube with 2 DM indices, 1 time bin.
            cube = _make_sparse_cube(2, 1, 5)
            tx._transmit_one_cube_prod(cube, specnum=1, flags=0)
            tx._transmit_one_cube_prod(cube, specnum=2, flags=0)

            frames = _recv_all_frames(rx_sock, 20)
            by_dm: dict[int, list[int]] = {}
            for hdr, _ in frames:
                by_dm.setdefault(hdr.dm_idx, []).append(hdr.seq)

            assert 0 in by_dm and 1 in by_dm, f"Expected dm_idx 0 and 1; got {list(by_dm)}"
            # Seq values for each dm are distinct and monotone increasing.
            for dm, seqs in by_dm.items():
                for i in range(1, len(seqs)):
                    assert seqs[i] >= seqs[i - 1], (
                        f"dm_idx={dm} seq not monotone at step {i}: "
                        f"{seqs[i - 1]} → {seqs[i]}"
                    )
        finally:
            tx.close()
            rx_sock.close()

    def test_b4_pattern_id_matches_cached(self):
        """pattern_id in every header matches the value set in prepare_prod."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, 10)
            tx._transmit_one_cube_prod(cube, specnum=99, flags=0)
            data = rx_sock.recv(65536)
            hdr, _ = unpack_frame(data)
            assert hdr.pattern_id == _FAKE_PATTERN_ID
        finally:
            tx.close()
            rx_sock.close()

    def test_b5_scale_offset_wire_bytes_match_repack(self):
        """Re-packing the received header+payload produces byte-identical output."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, 20, seed=7)
            tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
            data = rx_sock.recv(65536)
            hdr, payload = unpack_frame(data)
            repacked = pack_frame(hdr, payload)
            assert repacked == data, "Re-packed frame differs from wire bytes"
        finally:
            tx.close()
            rx_sock.close()

    def test_b6_last_in_block_only_on_final_fragment(self):
        """FLAG_LAST_IN_BLOCK is set only on the last fragment per payload."""
        n_filled = 6000  # forces ≥2 fragments
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, n_filled)
            tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
            frames = _recv_all_frames(rx_sock, 4)
            if len(frames) < 2:
                pytest.skip("Payload fits in one fragment; increase n_filled")
            for hdr, _ in frames[:-1]:
                assert not hdr.last_in_block, (
                    f"FLAG_LAST_IN_BLOCK set on non-final frag {hdr.frag_idx}"
                )
            assert frames[-1][0].last_in_block, (
                "FLAG_LAST_IN_BLOCK not set on final fragment"
            )
        finally:
            tx.close()
            rx_sock.close()


# ---------------------------------------------------------------------------
# (c) Quantisation + scale/offset
# ---------------------------------------------------------------------------


class TestQuantisationScaleOffset:
    """Group (c): Quantisation and scale/offset semantics (4 tests)."""

    def test_c1_cint8_scale_over_filled_cells_only(self):
        """scale/offset uses only actual filled-cell values, ignores zeros."""
        core = np.array([3.0 + 2j, -1.0 + 4j, 0.5 - 1.5j], dtype=np.complex64)
        with_zeros = np.concatenate([core, np.zeros(100, dtype=np.complex64)])

        def to_re_im(arr: np.ndarray) -> np.ndarray:
            return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)

        scale_core, off_core = TransportTx._compute_scale_offset(to_re_im(core))
        scale_all, off_all = TransportTx._compute_scale_offset(to_re_im(with_zeros))
        # Appending zeros does NOT change the max-abs → same scale.
        assert scale_core == scale_all, (
            f"Scale changed with appended zeros: {scale_core} vs {scale_all}"
        )
        assert off_core == off_all == np.float32(0.0)

    def test_c2_cfp16_scale_is_identity_flag_quantized_clear(self):
        """cfp16 path: scale≈1.0, offset≈0.0, FLAG_QUANTIZED not set."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port, bits_per_cell=BITS_CFP16_COMPLEX)
        try:
            cube = _make_sparse_cube(1, 1, 5)
            tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
            data = rx_sock.recv(65536)
            hdr, _ = unpack_frame(data)
            assert hdr.scale == pytest.approx(1.0, abs=1e-6), (
                f"cfp16 scale should be 1.0, got {hdr.scale}"
            )
            assert hdr.offset == pytest.approx(0.0, abs=1e-6)
            assert not (hdr.flags & FLAG_QUANTIZED), (
                "FLAG_QUANTIZED must be 0 for cfp16 (D1 decision)"
            )
        finally:
            tx.close()
            rx_sock.close()

    def test_c3_filled_cells_span_matches_n_filled(self):
        """Sum of payload_bytes_in_frag across fragments equals n_filled × bpc // 8."""
        for n_filled in (1, 50, 500):
            rx_sock, rx_port = _make_rx_sock()
            tx = _make_prod_tx(rx_port)
            try:
                cube = _make_sparse_cube(1, 1, n_filled)
                tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
                frames = _recv_all_frames(rx_sock, 10)
                total = sum(h.payload_bytes_in_frag for h, _ in frames)
                expected = n_filled * BITS_CINT8_COMPLEX // 8
                assert total == expected, (
                    f"n_filled={n_filled}: total={total}, expected={expected}"
                )
            finally:
                tx.close()
                rx_sock.close()

    def test_c4_empty_cube_zero_drops_zero_mon_key_bumps(self):
        """n_filled=0 cube: tx_dropped_payloads stays zero."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, 0)
            before = tx.tx_dropped_payloads
            tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
            assert tx.tx_dropped_payloads == before, (
                "tx_dropped_payloads should not increment for empty cube"
            )
        finally:
            tx.close()
            rx_sock.close()


# ---------------------------------------------------------------------------
# (d) Headers + flags
# ---------------------------------------------------------------------------


class TestHeadersAndFlags:
    """Group (d): Header field round-trip and flag propagation (4 tests)."""

    def _first_frame(
        self, rx_sock: socket.socket
    ) -> tuple[ProdFrameHeader, bytes]:
        data = rx_sock.recv(65536)
        return unpack_frame(data)

    def test_d1_rfi_warming_up_flag_propagates(self):
        """FLAG_RFI_WARMING_UP propagates from the flags argument."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, 5)
            tx._transmit_one_cube_prod(
                cube, specnum=0, flags=FLAG_RFI_WARMING_UP,
            )
            hdr, _ = self._first_frame(rx_sock)
            assert bool(hdr.flags & FLAG_RFI_WARMING_UP), (
                "FLAG_RFI_WARMING_UP not set in header"
            )
        finally:
            tx.close()
            rx_sock.close()

    def test_d2_t_int_factor_round_trips_via_header(self):
        """t_int_factor (header byte 49) round-trips correctly."""
        for t_int in (1, 8, 32):
            rx_sock, rx_port = _make_rx_sock()
            tx = _make_prod_tx(rx_port, t_int_factor=t_int)
            try:
                cube = _make_sparse_cube(1, 1, 5)
                tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
                hdr, _ = self._first_frame(rx_sock)
                assert hdr.t_int_factor == t_int, (
                    f"t_int_factor: expected {t_int}, got {hdr.t_int_factor}"
                )
            finally:
                tx.close()
                rx_sock.close()

    def test_d3_bits_per_cell_round_trips_via_header(self):
        """bits_per_cell (header byte 48) round-trips correctly."""
        for bits in (BITS_CINT8_COMPLEX, BITS_CFP16_COMPLEX):
            rx_sock, rx_port = _make_rx_sock()
            tx = _make_prod_tx(rx_port, bits_per_cell=bits)
            try:
                cube = _make_sparse_cube(1, 1, 5)
                tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
                hdr, _ = self._first_frame(rx_sock)
                assert hdr.bits_per_cell == bits
            finally:
                tx.close()
                rx_sock.close()

    def test_d4_specnum_round_trips(self):
        """specnum round-trips exactly for various values."""
        for specnum in (0, 0xCAFEBABE, 2**63 - 1):
            rx_sock, rx_port = _make_rx_sock()
            tx = _make_prod_tx(rx_port)
            try:
                cube = _make_sparse_cube(1, 1, 5)
                tx._transmit_one_cube_prod(cube, specnum=specnum, flags=0)
                hdr, _ = self._first_frame(rx_sock)
                assert hdr.specnum == specnum, (
                    f"specnum: expected {specnum:#x}, got {hdr.specnum:#x}"
                )
            finally:
                tx.close()
                rx_sock.close()


# ---------------------------------------------------------------------------
# (e) Pacer + mon-keys
# ---------------------------------------------------------------------------


class TestPacerAndMonKeys:
    """Group (e): Token-bucket pacer behaviour + mon-key counters (4 tests)."""

    def test_e1_tx_dropped_payloads_increments_when_pacer_drops(self):
        """tx_dropped_payloads > 0 when bucket rate=0 and FIFO fills."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(
            rx_port,
            target_gbps=0.0,        # never refills
            bucket_fifo_depth=1,    # FIFO depth 1 → fast drops
        )
        try:
            cube = _make_sparse_cube(2, 1, 20)
            # First send creates the per-dm buckets. Drain them so the
            # *next* send must queue / drop instead of debiting tokens.
            tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
            for bucket in tx._bucket_by_flow.values():
                bucket._balance = 0.0
                bucket._fifo.clear()
            # With FIFO depth=1, subsequent payloads cause drops.
            before = tx.tx_dropped_payloads
            for specnum in range(1, 10):
                tx._transmit_one_cube_prod(cube, specnum=specnum, flags=0)
                # Keep the bucket drained between sends.
                for bucket in tx._bucket_by_flow.values():
                    bucket._balance = 0.0
            assert tx.tx_dropped_payloads > before, (
                "tx_dropped_payloads should increment when bucket is drained "
                "and FIFO is at capacity"
            )
        finally:
            tx.close()
            rx_sock.close()

    def test_e2_cube_seq_emitted_advances_monotonically(self):
        """cube_seq_emitted is non-decreasing after each block."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        try:
            cube = _make_sparse_cube(1, 1, 5)
            prev = tx.cube_seq_emitted
            for specnum in range(5):
                tx._transmit_one_cube_prod(cube, specnum=specnum, flags=0)
                assert tx.cube_seq_emitted >= prev, (
                    f"cube_seq_emitted decreased at specnum={specnum}"
                )
                prev = tx.cube_seq_emitted
        finally:
            tx.close()
            rx_sock.close()

    def test_e3_mon_keys_reset_after_prepare_prod(self):
        """prepare_prod resets tx_dropped_payloads and cube_seq_emitted."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        rx_sock.close()
        tx.tx_dropped_payloads = 99
        tx.cube_seq_emitted = 42
        tx.prepare_prod({3: _FAKE_PATTERN_ID}, _FAKE_N_GRID)
        assert tx.tx_dropped_payloads == 0
        assert tx.cube_seq_emitted == 0
        tx.close()

    def test_e4_token_rate_matches_target_within_10_percent(self):
        """Token bucket refill matches target_gbps × headroom within 10%.

        Refill is verified over 0.1 s with capacity sized larger than
        the expected refill so we test the rate, not the capacity cap.
        """
        target_gbps = 0.001          # 1 Mbps = 125 KB/s
        pacer_headroom = 1.05
        expected_rate = target_gbps * 1e9 / 8.0 * pacer_headroom  # B/s
        dt_s = 0.1
        expected_refill_bytes = expected_rate * dt_s             # ~13125

        bucket = _TokenBucket(
            rate_bytes_per_sec=expected_rate,
            # Capacity big enough not to clamp the refill at this dt.
            capacity_bytes=int(expected_refill_bytes * 10),
        )
        bucket._balance = 0.0
        start_ns = time.monotonic_ns()
        bucket._last_ns = start_ns

        with mock.patch("dsart.transport.tx.time") as mock_time:
            mock_time.monotonic_ns.return_value = start_ns + int(dt_s * 1e9)
            bucket._refill()

        relative_error = (
            abs(bucket._balance - expected_refill_bytes) / expected_refill_bytes
        )
        assert relative_error < 0.10, (
            f"Token rate error > 10%: expected={expected_refill_bytes:.1f} B "
            f"over {dt_s} s, got balance={bucket._balance:.1f}"
        )


# ---------------------------------------------------------------------------
# (f) Integration with chunk-8 path
# ---------------------------------------------------------------------------


class TestIntegrationChunk8:
    """Group (f): Coexistence of prod-frame and chunk-8 paths (3 tests)."""

    def test_f1_default_path_emits_chunk8_magic(self):
        """use_prod_frame=False (default) emits chunk-8 MAGIC=0xD5A0FA57."""
        rx_sock, rx_port = _make_rx_sock()
        with TransportTx("127.0.0.1", rx_port, chgroup=0) as tx:
            cube = _make_sparse_cube(1, 1, 10)
            tx.transmit([cube], block_n=0, rfi_warming_up=False)
        try:
            data = rx_sock.recv(65536)
            magic = struct.unpack_from("<I", data, 0)[0]
            assert magic == CHUNK8_MAGIC, (
                f"Expected chunk-8 MAGIC={CHUNK8_MAGIC:#x}, got {magic:#x}"
            )
        finally:
            rx_sock.close()

    def test_f2_prod_frame_path_emits_prod_magic_72byte_header(self):
        """use_prod_frame=True emits MAGIC=0xD5A1107E and ≥72-byte frames."""
        rx_sock, rx_port = _make_rx_sock()
        cfg = TransportTxProdConfig(
            target_gbps_per_flow=100.0,
            bits_per_cell=BITS_CINT8_COMPLEX,
        )
        with TransportTx(
            "127.0.0.1", rx_port, chgroup=0,
            use_prod_frame=True, prod_config=cfg,
        ) as tx:
            tx.prepare_prod({0: _FAKE_PATTERN_ID}, _FAKE_N_GRID)
            cube = _make_sparse_cube(1, 1, 10)
            tx.transmit([cube], block_n=0, rfi_warming_up=False, specnum=42)
        try:
            data = rx_sock.recv(65536)
            magic = struct.unpack_from("<I", data, 0)[0]
            assert magic == PROD_MAGIC, (
                f"Expected prod MAGIC={PROD_MAGIC:#x}, got {magic:#x}"
            )
            assert len(data) >= HEADER_BYTES
        finally:
            rx_sock.close()

    def test_f3_both_paths_share_lifecycle_cleanly(self):
        """Both paths use __enter__/__exit__ without raising."""
        cfg = TransportTxProdConfig(target_gbps_per_flow=100.0)
        with TransportTx("127.0.0.1", 1, chgroup=0) as tx8:
            assert tx8.use_prod_frame is False
        with TransportTx(
            "127.0.0.1", 1, chgroup=0,
            use_prod_frame=True, prod_config=cfg,
        ) as tx_prod:
            assert tx_prod.use_prod_frame is True


# ---------------------------------------------------------------------------
# (extra) Edge cases and robustness (10 tests, total >= 25)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Extra tests for boundary conditions and error paths (10 tests)."""

    def test_x1_transmit_raises_if_specnum_none_with_prod_frame(self):
        """transmit() raises NotImplementedError when specnum=None in prod mode."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        rx_sock.close()
        try:
            cube = _make_sparse_cube(1, 1, 5)
            with pytest.raises(NotImplementedError, match="specnum"):
                tx.transmit([cube], block_n=0, rfi_warming_up=False)
        finally:
            tx.close()

    def test_x2_transmit_prod_raises_before_prepare(self):
        """_transmit_one_cube_prod raises NotImplementedError if n_grid=0."""
        rx_sock, rx_port = _make_rx_sock()
        rx_sock.close()
        cfg = TransportTxProdConfig(target_gbps_per_flow=100.0)
        tx = TransportTx(
            "127.0.0.1", rx_port, chgroup=0,
            use_prod_frame=True, prod_config=cfg,
        )
        # NOT calling prepare_prod.
        try:
            cube = _make_sparse_cube(1, 1, 5)
            with pytest.raises(NotImplementedError, match="prepare_prod"):
                tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
        finally:
            tx.close()

    def test_x3_prod_config_required_when_use_prod_frame_true(self):
        """Omitting prod_config with use_prod_frame=True raises ValueError."""
        with pytest.raises(ValueError, match="prod_config"):
            TransportTx(
                "127.0.0.1", 0, chgroup=0,
                use_prod_frame=True,
                prod_config=None,
            )

    def test_x4_txprodconfig_invalid_t_int_factor_raises(self):
        """TransportTxProdConfig rejects invalid t_int_factor."""
        with pytest.raises(ValueError, match="t_int_factor"):
            TransportTxProdConfig(
                target_gbps_per_flow=1.0,
                t_int_factor=3,
            )

    def test_x5_txprodconfig_invalid_bits_per_cell_raises(self):
        """TransportTxProdConfig rejects invalid bits_per_cell."""
        with pytest.raises(ValueError, match="bits_per_cell"):
            TransportTxProdConfig(
                target_gbps_per_flow=1.0,
                bits_per_cell=8,
            )

    def test_x6_image_cube_raises_not_implemented(self):
        """Image cube (ndim=4) raises NotImplementedError in prod path."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        rx_sock.close()
        try:
            image_cube = torch.zeros(1, 1, 256, 256, dtype=torch.complex64)
            with pytest.raises(NotImplementedError, match="ndim=4"):
                tx._transmit_one_cube_prod(image_cube, specnum=0, flags=0)
        finally:
            tx.close()

    def test_x7_prepare_prod_resets_seq_counters(self):
        """prepare_prod clears per-dm_idx seq counters and buckets."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        rx_sock.close()
        try:
            tx._seq_by_flow[0] = 100
            tx._bucket_by_flow[0] = tx._get_bucket(0)
            tx.prepare_prod({3: _FAKE_PATTERN_ID}, _FAKE_N_GRID)
            assert tx._seq_by_flow == {}
            assert tx._bucket_by_flow == {}
        finally:
            tx.close()

    def test_x8_cint8_flag_quantized_set_cfp16_not_set(self):
        """FLAG_QUANTIZED set iff bits_per_cell==BITS_CINT8_COMPLEX (D1 decision)."""
        for bits, expect in [(BITS_CINT8_COMPLEX, True), (BITS_CFP16_COMPLEX, False)]:
            rx_sock, rx_port = _make_rx_sock()
            tx = _make_prod_tx(rx_port, bits_per_cell=bits)
            try:
                cube = _make_sparse_cube(1, 1, 5)
                tx._transmit_one_cube_prod(cube, specnum=0, flags=0)
                data = rx_sock.recv(65536)
                hdr, _ = unpack_frame(data)
                assert bool(hdr.flags & FLAG_QUANTIZED) == expect, (
                    f"bits={bits}: FLAG_QUANTIZED={bool(hdr.flags & FLAG_QUANTIZED)}, "
                    f"expected={expect}"
                )
            finally:
                tx.close()
                rx_sock.close()

    def test_x9_token_bucket_starts_full(self):
        """Token bucket starts with balance == capacity_bytes."""
        capacity = 1000
        bucket = _TokenBucket(rate_bytes_per_sec=1000.0, capacity_bytes=capacity)
        assert bucket._balance == float(capacity)

    def test_x10_two_dm_flows_have_independent_buckets(self):
        """dm_idx=0 and dm_idx=1 get independent token buckets."""
        rx_sock, rx_port = _make_rx_sock()
        tx = _make_prod_tx(rx_port)
        rx_sock.close()
        try:
            b0 = tx._get_bucket(0)
            b1 = tx._get_bucket(1)
            assert b0 is not b1
            b0._balance = 0.0
            assert b1._balance > 0, "b1 should be unaffected by b0 drain"
        finally:
            tx.close()
