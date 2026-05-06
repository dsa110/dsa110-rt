"""Transport loopback acceptance tests (M3 chunk 8).

Pins the wire-format codec, the UDP loopback round-trip behaviour,
the per-(chgroup) sequence-gap accounting, and the chunk-4
``TransportTxStage`` Protocol compliance.

Test groups (mirroring chunk-8 substrate spec §3.3):

1. ``FastVisFrame`` codec — pack/unpack round-trip, magic/CRC/oversize
   validation, dtype-code round-trip.
2. ``TransportTx`` semantics — frame-per-tile counting, monotonic seq
   across calls.
3. ``TransportRx`` semantics — receive timeout None, magic + CRC
   validation.
4. Loopback round-trip — 100 cubes no loss; injected seq gap
   detected.
5. Chunk-4 Protocol compliance — TransportTx as TransportTxStage,
   process_block end-to-end; rfi_warming_up bit propagates to
   frame.flags bit0.
"""

from __future__ import annotations

import math
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from dsart.common.constants import FADA_BYTES_PER_BLOCK, NANTS
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    NoOpCoarseDM,
    NoOpStage2Fifo,
    _build_core_baseline_mask,
    build_context,
    process_block,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)
from dsart.transport import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    DTYPE_CFP16,
    DTYPE_CINT8,
    FLAG_RFI_WARMING_UP,
    HEADER_BYTES,
    MAGIC,
    FastVisFrame,
    FrameCRCError,
    FrameMagicError,
    FramePayloadOversizeError,
    TransportRx,
    TransportTx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind a UDP socket to ephemeral port 0 → return the port number,
    then close the probe socket. The returned port is unbound by the
    time the caller uses it (kernel won't immediately reuse for UDP);
    callers should bind ``0`` themselves and read the actual port off
    the bound socket. Provided for the few tests that need a fixed
    port string ahead of time.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_payload(n_cells: int, *, dtype: np.dtype = np.float16,
                  seed: int = 0) -> bytes:
    """A reproducible random payload of ``n_cells`` complex cells in
    ``dtype`` — re/im interleaved."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-1.0, 1.0, size=(n_cells, 2)).astype(dtype)
    return arr.tobytes()


def _make_frame(
    *,
    seq: int = 0,
    chgroup: int = 0,
    dm_idx: int = 0,
    t_idx: int = 0,
    n_grid: int = 256,
    dtype_code: int = DTYPE_CFP16,
    flags: int = 0,
    n_cells: int = 64,
) -> FastVisFrame:
    return FastVisFrame(
        seq=seq, chgroup=chgroup, dm_idx=dm_idx, t_idx=t_idx,
        n_grid=n_grid, dtype_code=dtype_code, flags=flags,
        payload=_make_payload(n_cells, seed=seq * 1024 + t_idx),
    )


def _synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Mirror tests/test_corr_fast_integration._synth_antpos."""
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _synthetic_fada_block(seed: int = 20260505) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, size=FADA_BYTES_PER_BLOCK, dtype=np.uint8)


def _make_chunk4_ctx(
    *,
    transport_tx: TransportTx | None = None,
    n_grid: int = 64,
):
    """Mirror tests/test_corr_fast_integration._build_test_context but
    with a custom ``transport_tx``. CPU-only; small grid for speed.
    """
    cfg = FastIntegrationConfig(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=n_grid,
        kernel_support=1,
        t_int_fast_native=NPACKETS_PER_BLOCK * NTIMES_PER_PACKET,
        cal_path=None,
        rfi_enabled=False,
        static_sky_disabled=True,
        static_sky_warmup_cubes=0,
    )
    e, n = _synth_antpos()
    core_mask = _build_core_baseline_mask(n_core=82)
    return build_context(
        cfg, device=torch.device("cpu"),
        antpos_e=e, antpos_n=n,
        is_core_baseline_mask=core_mask,
        transport_tx=transport_tx,
    )


# ---------------------------------------------------------------------------
# 1. FastVisFrame codec
# ---------------------------------------------------------------------------


def test_FastVisFrame_pack_unpack_round_trip() -> None:
    """pack(frame).unpack() returns identical fields + payload."""
    frame = _make_frame(
        seq=12345, chgroup=7, dm_idx=3, t_idx=42, n_grid=256,
        dtype_code=DTYPE_CFP16, flags=FLAG_RFI_WARMING_UP, n_cells=128,
    )
    wire = frame.pack()
    assert len(wire) == HEADER_BYTES + frame.payload_bytes
    out = FastVisFrame.unpack(wire)
    assert out.seq == frame.seq
    assert out.chgroup == frame.chgroup
    assert out.dm_idx == frame.dm_idx
    assert out.t_idx == frame.t_idx
    assert out.n_grid == frame.n_grid
    assert out.dtype_code == frame.dtype_code
    assert out.flags == frame.flags
    assert out.payload == frame.payload
    assert out.rfi_warming_up is True
    assert HEADER_BYTES == 32


def test_FastVisFrame_magic_validation() -> None:
    """Corrupt the magic byte → unpack raises FrameMagicError."""
    frame = _make_frame(seq=1, n_cells=8)
    wire = bytearray(frame.pack())
    # Flip a byte in the magic (offset 0..3).
    wire[0] ^= 0xFF
    with pytest.raises(FrameMagicError, match="bad magic"):
        FastVisFrame.unpack(bytes(wire))


def test_FastVisFrame_crc_validation() -> None:
    """Flip a payload byte → unpack raises FrameCRCError."""
    frame = _make_frame(seq=2, n_cells=8)
    wire = bytearray(frame.pack())
    # Flip a byte WELL inside the payload (after the 32-byte header).
    wire[HEADER_BYTES + 4] ^= 0x55
    with pytest.raises(FrameCRCError, match="CRC mismatch"):
        FastVisFrame.unpack(bytes(wire))


def test_FastVisFrame_oversize_payload_rejected() -> None:
    """payload > max_payload_bytes raises FramePayloadOversizeError."""
    big_payload = b"\x00" * (DEFAULT_MAX_PAYLOAD_BYTES + 1)
    frame = FastVisFrame(
        seq=0, chgroup=0, dm_idx=0, t_idx=0, n_grid=256,
        dtype_code=DTYPE_CFP16, flags=0, payload=big_payload,
    )
    with pytest.raises(FramePayloadOversizeError, match="exceeds max_payload_bytes"):
        frame.pack()
    # And explicitly with a tighter cap.
    small_frame = FastVisFrame(
        seq=0, chgroup=0, dm_idx=0, t_idx=0, n_grid=256,
        dtype_code=DTYPE_CFP16, flags=0, payload=b"\x00" * 100,
    )
    with pytest.raises(FramePayloadOversizeError):
        small_frame.pack(max_payload_bytes=50)


def test_FastVisFrame_dtype_code_round_trip() -> None:
    """Both DTYPE_CFP16 (0) and DTYPE_CINT8 (1) round-trip cleanly."""
    for code in (DTYPE_CFP16, DTYPE_CINT8):
        frame = _make_frame(
            seq=10 + code, dtype_code=code, n_cells=32,
        )
        wire = frame.pack()
        out = FastVisFrame.unpack(wire)
        assert out.dtype_code == code
        assert out.payload == frame.payload


# ---------------------------------------------------------------------------
# 2. TransportTx semantics
# ---------------------------------------------------------------------------


def test_TransportTx_sends_one_frame_per_tile() -> None:
    """One sparse-COO cube of (N_DM=3, n_fv=5, N_filled=16) →
    3 * 5 = 15 frames sent."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.5)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=4)
        try:
            cube = torch.complex(
                torch.randn(3, 5, 16),
                torch.randn(3, 5, 16),
            )
            n = tx.transmit([cube], block_n=1, rfi_warming_up=False)
            assert n == 15
            assert tx.n_sent == 15
        finally:
            tx.close()
        # Drain the socket to verify we actually got 15 frames.
        received: list[FastVisFrame] = []
        for _ in range(15):
            f = rx.receive_one()
            assert f is not None
            received.append(f)
        assert len(received) == 15
        # All chgroup=4, all dtype=cfp16, dm/t indices correctly assigned.
        seen = {(f.dm_idx, f.t_idx) for f in received}
        assert seen == {(d, t) for d in range(3) for t in range(5)}
        assert all(f.chgroup == 4 for f in received)
    finally:
        rx.close()


def test_TransportTx_seq_monotonic_across_transmit_calls() -> None:
    """3 transmit() calls → seq 0, 1, ..., n_total-1 strictly monotonic."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.5)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=2)
        try:
            cube_a = torch.complex(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
            cube_b = torch.complex(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
            cube_c = torch.complex(torch.randn(1, 1, 8), torch.randn(1, 1, 8))
            n_a = tx.transmit([cube_a], block_n=1, rfi_warming_up=False)
            n_b = tx.transmit([cube_b], block_n=2, rfi_warming_up=False)
            n_c = tx.transmit([cube_c], block_n=3, rfi_warming_up=False)
            assert (n_a, n_b, n_c) == (4, 6, 1)
        finally:
            tx.close()
        seqs = []
        for _ in range(11):
            f = rx.receive_one()
            assert f is not None
            seqs.append(f.seq)
        assert seqs == list(range(11))
    finally:
        rx.close()


# ---------------------------------------------------------------------------
# 3. TransportRx semantics
# ---------------------------------------------------------------------------


def test_TransportRx_receive_one_returns_None_on_timeout() -> None:
    """Empty socket → returns None within roughly recv_timeout_s."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.1)
    try:
        t0 = time.monotonic()
        out = rx.receive_one()
        elapsed = time.monotonic() - t0
        assert out is None
        # Should have returned ~recv_timeout_s after a single recvfrom.
        assert 0.05 < elapsed < 1.0
        assert rx.stats.n_received == 0
    finally:
        rx.close()


def test_TransportRx_receive_one_validates_magic() -> None:
    """Send raw garbage → receive_one raises FrameMagicError + bumps mon."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.5)
    try:
        garbage = b"\x00" * HEADER_BYTES + b"X" * 8
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(garbage, ("127.0.0.1", rx.port))
        finally:
            sock.close()
        with pytest.raises(FrameMagicError):
            rx.receive_one()
        assert rx.stats.n_magic_fail == 1
    finally:
        rx.close()


def test_TransportRx_receive_one_validates_crc() -> None:
    """Flip a payload byte after pack → receive_one raises FrameCRCError."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.5)
    try:
        frame = _make_frame(seq=99, n_cells=32)
        wire = bytearray(frame.pack())
        wire[HEADER_BYTES + 6] ^= 0xAA                                   # corrupt payload
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(bytes(wire), ("127.0.0.1", rx.port))
        finally:
            sock.close()
        with pytest.raises(FrameCRCError):
            rx.receive_one()
        assert rx.stats.n_crc_fail == 1
        assert rx.stats.n_received == 0
    finally:
        rx.close()


# ---------------------------------------------------------------------------
# 4. Loopback round-trip
# ---------------------------------------------------------------------------


def test_loopback_round_trip_100_cubes_no_loss() -> None:
    """100 cubes via loopback → RX recovers all 100, no CRC fails,
    monotonic seq, no gaps. Each cube emits exactly 1 frame."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=1.0)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=0)
        try:
            n_filled = 256
            for i in range(100):
                cube = torch.complex(
                    torch.randn(1, 1, n_filled),
                    torch.randn(1, 1, n_filled),
                )
                sent = tx.transmit([cube], block_n=i, rfi_warming_up=False)
                assert sent == 1
            assert tx.n_sent == 100
        finally:
            tx.close()
        for i in range(100):
            f = rx.receive_one()
            assert f is not None, f"missing frame at i={i}"
            assert f.seq == i
        assert rx.stats.n_received == 100
        assert rx.stats.n_crc_fail == 0
        assert rx.stats.n_magic_fail == 0
        assert rx.stats.n_seq_gaps == 0
        assert rx.stats.n_out_of_order == 0
    finally:
        rx.close()


def test_loopback_handles_seq_gap() -> None:
    """Manually inject a seq gap (skip seq=42) → RX detects exactly 1 gap."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=1.0)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for s in range(50):
                if s == 42:
                    continue                                              # skip seq=42
                f = _make_frame(seq=s, chgroup=11, n_cells=8)
                sock.sendto(f.pack(), ("127.0.0.1", rx.port))
        finally:
            sock.close()
        # 49 frames will arrive; 1 gap (seq=42) detected.
        for _ in range(49):
            f = rx.receive_one()
            assert f is not None
        assert rx.stats.n_received == 49
        assert rx.stats.n_seq_gaps == 1
        assert rx.stats.n_crc_fail == 0
    finally:
        rx.close()


# ---------------------------------------------------------------------------
# 5. Chunk-4 TransportTxStage Protocol compliance
# ---------------------------------------------------------------------------


def test_chunk4_TransportTxStage_protocol_compliance() -> None:
    """Wire TransportTx into IntegrationContext.transport_tx; call
    process_block with synthetic raw fada → no exception, ``n_tx``
    equals the number of frames TransportTx counted as sent.

    With NoOpCoarseDM (single DM trial = 1) and NoOpStage2Fifo
    (immediate eviction), one block should produce exactly
    ``n_fast_vis_per_full_block`` frames sent.
    """
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=2.0)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=0)
        try:
            ctx = _make_chunk4_ctx(transport_tx=tx, n_grid=64)
            assert ctx.transport_tx is tx
            assert isinstance(ctx.coarse_dm, NoOpCoarseDM)
            assert isinstance(ctx.stage2_fifo, NoOpStage2Fifo)

            raw = _synthetic_fada_block()
            out = process_block(raw, ctx=ctx, block_n=1)

            # Single full-block tile → n_fv == 1.
            n_fv = ctx.kernel.n_fast_vis_per_full_block
            expected_frames = 1 * n_fv                                   # N_DM=1 * n_fv
            assert out.n_tx == expected_frames
            assert tx.n_sent == expected_frames
            assert tx.n_send_fail == 0
        finally:
            tx.close()

        # Drain RX: each frame should validate cleanly.
        n_drained = 0
        while True:
            f = rx.receive_one()
            if f is None:
                break
            n_drained += 1
            assert f.chgroup == 0
            assert f.dtype_code == DTYPE_CFP16
        assert n_drained == expected_frames
        assert rx.stats.n_crc_fail == 0
        assert rx.stats.n_seq_gaps == 0
    finally:
        rx.close()


def test_chunk4_rfi_warmup_bit_propagates_to_frame_flags() -> None:
    """Set rfi_warming_up=True in transmit() → every frame's flags bit0
    is set; with False → bit0 is unset."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=1.0)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=0)
        try:
            cube = torch.complex(torch.randn(1, 3, 8), torch.randn(1, 3, 8))

            tx.transmit([cube], block_n=1, rfi_warming_up=True)
            tx.transmit([cube], block_n=2, rfi_warming_up=False)
        finally:
            tx.close()

        warm_seen = []
        for _ in range(6):
            f = rx.receive_one()
            assert f is not None
            warm_seen.append(f.rfi_warming_up)
        # First 3 frames (block 1, rfi_warming_up=True) → True.
        # Next 3 frames (block 2, rfi_warming_up=False) → False.
        assert warm_seen[:3] == [True, True, True]
        assert warm_seen[3:] == [False, False, False]
        assert (warm_seen[0] is True) and (warm_seen[3] is False)
    finally:
        rx.close()


# ---------------------------------------------------------------------------
# Bonus / regression coverage
# ---------------------------------------------------------------------------


def test_TransportTx_image_cube_shape_auto_detected() -> None:
    """4D cube (N_DM, n_fv, N_grid, N_grid) → one frame per (dm, t),
    payload = N_grid * N_grid cells (F26 spec)."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.5)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=0)
        try:
            n_grid = 32                                                  # 32*32*4 cfp16 = 4096 B
            cube = torch.complex(
                torch.randn(1, 1, n_grid, n_grid),
                torch.randn(1, 1, n_grid, n_grid),
            )
            n = tx.transmit([cube], block_n=1, rfi_warming_up=False)
            assert n == 1
        finally:
            tx.close()
        f = rx.receive_one()
        assert f is not None
        assert f.n_grid == n_grid
        # cfp16: 4 bytes per cell; n_grid*n_grid cells.
        assert f.payload_bytes == n_grid * n_grid * 4
    finally:
        rx.close()


def test_TransportTx_rejects_bad_cube_shape() -> None:
    """ndim ∉ {3, 4} or non-square trailing axes → ValueError."""
    rx = TransportRx("127.0.0.1", 0, recv_timeout_s=0.1)
    try:
        tx = TransportTx("127.0.0.1", rx.port, chgroup=0)
        try:
            with pytest.raises(ValueError, match="ndim"):
                tx.transmit(
                    [torch.complex(torch.randn(2), torch.randn(2))],     # ndim=1
                    block_n=1, rfi_warming_up=False,
                )
            with pytest.raises(ValueError, match="square"):
                tx.transmit(
                    [torch.complex(
                        torch.randn(1, 1, 4, 5),
                        torch.randn(1, 1, 4, 5),
                    )],
                    block_n=1, rfi_warming_up=False,
                )
        finally:
            tx.close()
    finally:
        rx.close()
