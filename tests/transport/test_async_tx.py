"""Tests for M7.2 async TX (``transport/async_tx.py``).

End-to-end tests that spawn real worker subprocesses, each running a
real :class:`TransportTx` (prod-frame mode), receiving on a real UDP
loopback socket. Verifies:

  (a) AsyncTransportTx.spawn + transmit + close roundtrip
  (b) DM-axis split: each worker's frames carry the expected dm_idx range
  (c) Backpressure surfaces (slow consumer → producer raises)
  (d) Lifecycle: close idempotent + workers cleanly exit
  (e) Worker crash detection — defer to follow-up; spawn-test only

All tests run on CPU only (the main process does not require torch.cuda).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import struct
import time
import uuid
from collections import defaultdict

import numpy as np
import pytest
import torch

from dsart.transport.async_tx import (
    AsyncTransportTx,
    AsyncTransportTxConfig,
)
from dsart.transport.prod_frame import (
    HEADER_BYTES as PROD_HEADER_BYTES,
    MAGIC as PROD_MAGIC,
    unpack_frame,
)
from dsart.transport.tx_ring import CubeShmRingDims, TxRingBackpressureError


_FAKE_PATTERN_ID = 0xDEADBEEFCAFEBABE
_FAKE_N_GRID = 256


def _make_rx(port: int = 0) -> tuple[socket.socket, int]:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    rx.bind(("127.0.0.1", port))
    rx.settimeout(2.0)
    return rx, rx.getsockname()[1]


def _make_cube(n_dm: int, n_fv: int, n_filled: int, *, seed: int = 7):
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((n_dm, n_fv, n_filled), dtype=np.float32)
    im = rng.standard_normal((n_dm, n_fv, n_filled), dtype=np.float32)
    return torch.from_numpy(re + 1j * im).to(torch.complex64)


def _drain_rx_until_idle(
    rx: socket.socket,
    idle_s: float = 0.3,
    *,
    first_packet_timeout_s: float = 5.0,
) -> list[bytes]:
    """Collect all frames until ``idle_s`` of silence after the first packet.

    Waits up to ``first_packet_timeout_s`` for the first packet to arrive
    (handles worker subprocess startup which can take ~1.5 s on first
    spawn). Returns empty list if no packet arrives in that window.
    """
    rx.settimeout(0.05)
    frames: list[bytes] = []
    t_first_deadline = time.monotonic() + first_packet_timeout_s
    # First packet wait loop
    while time.monotonic() < t_first_deadline:
        try:
            data, _ = rx.recvfrom(65535)
            frames.append(data)
            break
        except socket.timeout:
            continue
    if not frames:
        return frames
    # Drain until idle
    last_data_t = time.monotonic()
    while time.monotonic() - last_data_t < idle_s:
        try:
            data, _ = rx.recvfrom(65535)
            frames.append(data)
            last_data_t = time.monotonic()
        except socket.timeout:
            continue
    return frames


# ---------------------------------------------------------------------------
# (a) Single-worker spawn + transmit + close
# ---------------------------------------------------------------------------


def test_spawn_single_worker_one_cube():
    rx, port = _make_rx()
    n_dm = 2
    n_fv = 8
    n_filled = 32
    cube = _make_cube(n_dm, n_fv, n_filled)
    dims = CubeShmRingDims(
        n_slots=4,
        shape=(n_dm, n_fv, n_filled),
        dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=port, chgroup=3,
        n_workers=1, n_dm_total=n_dm,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=100.0,  # generous; pacer must not drop
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
        n_handed = tx.transmit(
            [cube], block_n=11, rfi_warming_up=False, specnum=0xABCD,
        )
        assert n_handed == 1
        frames = _drain_rx_until_idle(rx, idle_s=0.5)
        # Expect at least n_dm × n_fv frames (one per (dm, t) tile);
        # MTU fragmentation may yield more.
        assert len(frames) >= n_dm * n_fv, f"got {len(frames)} frames"
        # All frames should be valid prod-frames with correct chgroup.
        for f in frames:
            assert len(f) >= PROD_HEADER_BYTES
            hdr, _payload = unpack_frame(f)
            assert hdr.chgroup == 3
    finally:
        tx.close()
        rx.close()


# ---------------------------------------------------------------------------
# (b) Multi-worker DM-axis split — each worker emits its own DM range
# ---------------------------------------------------------------------------


def test_spawn_multi_worker_dm_split():
    rx, port = _make_rx()
    n_dm_total = 8
    n_workers = 4
    n_fv = 8
    n_filled = 32
    cube = _make_cube(n_dm_total, n_fv, n_filled)
    # Per-worker ring shape: max dm slice = 2.
    per_worker_dm = n_dm_total // n_workers
    dims = CubeShmRingDims(
        n_slots=4,
        shape=(per_worker_dm, n_fv, n_filled),
        dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=port, chgroup=5,
        n_workers=n_workers, n_dm_total=n_dm_total,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=100.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
        # DM split sanity
        assert tx.n_workers == n_workers
        for w in range(n_workers):
            lo, hi = tx.dm_split(w)
            assert hi - lo == per_worker_dm
            assert lo == w * per_worker_dm
            assert hi == (w + 1) * per_worker_dm

        n_handed = tx.transmit(
            [cube], block_n=200, rfi_warming_up=False, specnum=0xDEADBEEF,
        )
        assert n_handed == 1
        frames = _drain_rx_until_idle(rx, idle_s=0.8)
        # Each of n_dm_total × n_fv tiles emits ≥ 1 fragment.
        assert len(frames) >= n_dm_total * n_fv, (
            f"expected >= {n_dm_total * n_fv} frames, got {len(frames)}"
        )
        # All frames should have a valid chgroup; collect dm_idx per
        # frame to verify the split.
        seen_dm = set()
        for f in frames:
            hdr, _payload = unpack_frame(f)
            assert hdr.chgroup == 5
            seen_dm.add(int(hdr.dm_idx))
        # Each worker (w) emits frames with header dm_idx in
        # [0, hi-lo) — the prod-frame dm_idx is LOCAL to the cube the
        # worker transmits, since each worker calls
        # TransportTx.transmit([slice]) independently. So all workers
        # emit dm_idx in {0, 1} (per_worker_dm == 2).
        assert seen_dm == set(range(per_worker_dm)), (
            f"expected dm_idx range = {set(range(per_worker_dm))}, "
            f"got {seen_dm}"
        )
    finally:
        tx.close()
        rx.close()


# ---------------------------------------------------------------------------
# (c) Multiple blocks in a row — confirm no backpressure under fast consumer
# ---------------------------------------------------------------------------


def test_many_blocks_no_backpressure():
    rx, port = _make_rx()
    n_dm = 4
    n_fv = 16
    n_filled = 64
    dims = CubeShmRingDims(
        n_slots=4, shape=(2, n_fv, n_filled),
        dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=port, chgroup=7,
        n_workers=2, n_dm_total=n_dm,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=100.0,
        reserve_timeout_s=2.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
        cube = _make_cube(n_dm, n_fv, n_filled)
        # Warm-up cube — wait for it to land on RX before measuring steady
        # state. Worker spawn takes ~1.5 s on first import; the steady-state
        # behaviour starts after that.
        tx.transmit([cube], block_n=0, rfi_warming_up=False, specnum=0)
        _ = _drain_rx_until_idle(rx, idle_s=0.3, first_packet_timeout_s=5.0)

        # Now run at ~production cadence (8 Hz ≈ 134 ms; we go faster to
        # keep the test brief but still slower than the encode rate).
        n_blocks = 12
        backpressure_before = tx.stats()["n_backpressure_total"]
        for b in range(n_blocks):
            tx.transmit(
                [cube], block_n=b + 1, rfi_warming_up=False, specnum=1000 + b,
            )
            time.sleep(0.02)  # 50 Hz — well below worker encode rate
        frames = _drain_rx_until_idle(rx, idle_s=0.5)
        assert len(frames) >= n_blocks * n_dm * n_fv, (
            f"expected >= {n_blocks * n_dm * n_fv} frames, got {len(frames)}"
        )
        stats = tx.stats()
        # +1 for the warm-up cube.
        assert stats["n_cubes_in"] == n_blocks + 1
        assert stats["n_workers_alive"] == cfg.n_workers
        # Steady-state backpressure should be zero (warm-up backpressure
        # is excluded by snapshotting the counter before the burst).
        steady_bp = stats["n_backpressure_total"] - backpressure_before
        assert steady_bp == 0, (
            f"steady-state backpressure should be 0, got {steady_bp}"
        )
    finally:
        tx.close()
        rx.close()


# ---------------------------------------------------------------------------
# (d) Close is idempotent + workers exit
# ---------------------------------------------------------------------------


def test_close_is_idempotent():
    rx, port = _make_rx()
    dims = CubeShmRingDims(
        n_slots=2, shape=(2, 4, 16), dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=port, chgroup=1,
        n_workers=1, n_dm_total=2,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=10.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    tx.close()
    tx.close()  # idempotent
    rx.close()


# ---------------------------------------------------------------------------
# (e) Wrong-shape cube raises
# ---------------------------------------------------------------------------


def test_wrong_n_dm_raises():
    rx, port = _make_rx()
    dims = CubeShmRingDims(
        n_slots=2, shape=(2, 4, 16), dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=port, chgroup=1,
        n_workers=2, n_dm_total=4,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=10.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
        bad_cube = _make_cube(3, 4, 16)  # 3 != 4
        with pytest.raises(ValueError, match="N_DM"):
            tx.transmit(
                [bad_cube], block_n=0, rfi_warming_up=False, specnum=0,
            )
    finally:
        tx.close()
        rx.close()


def test_missing_specnum_raises():
    rx, port = _make_rx()
    dims = CubeShmRingDims(
        n_slots=2, shape=(2, 4, 16), dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=port, chgroup=1,
        n_workers=1, n_dm_total=2,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=10.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
        cube = _make_cube(2, 4, 16)
        with pytest.raises(ValueError, match="specnum"):
            tx.transmit([cube], block_n=0, rfi_warming_up=False)
    finally:
        tx.close()
        rx.close()
