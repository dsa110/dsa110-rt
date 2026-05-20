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
    """4-worker / N=8 split. Each worker uses port = base_port + worker_idx.
    All workers emit frames carrying GLOBAL dm_idx (worker_idx*per_worker_dm
    + local) via the ``dm_idx_offset`` shim in TransportTxProdConfig.
    """
    # Pick a free port range. We bind 4 sockets, one per worker.
    n_workers = 4
    rx_socks, base_port = _bind_per_worker_ports(base_port=0, n_workers=n_workers)
    n_dm_total = 8
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
        host="127.0.0.1", port=base_port, chgroup=5,
        n_workers=n_workers, n_dm_total=n_dm_total,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=100.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
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

        # Drain ALL 4 sockets (one per worker). Each worker emits
        # per_worker_dm × n_fv tiles.
        all_frames: list[tuple[int, bytes]] = []
        for w, rx in enumerate(rx_socks):
            frames = _drain_rx_until_idle(rx, idle_s=0.6)
            all_frames.extend((w, f) for f in frames)
        # Aggregate frame count: ≥ n_dm_total × n_fv (each tile emits
        # ≥ 1 fragment).
        assert len(all_frames) >= n_dm_total * n_fv, (
            f"expected >= {n_dm_total * n_fv} frames across all 4 "
            f"workers, got {len(all_frames)}"
        )
        # Frames from worker w MUST carry global dm_idx in
        # [w*per_worker_dm, (w+1)*per_worker_dm). Across all workers,
        # all 8 GLOBAL dm_idx values must appear.
        seen_dm: set[int] = set()
        for w, f in all_frames:
            hdr, _payload = unpack_frame(f)
            assert hdr.chgroup == 5
            gdm = int(hdr.dm_idx)
            seen_dm.add(gdm)
            lo = w * per_worker_dm
            assert lo <= gdm < lo + per_worker_dm, (
                f"frame on worker-w={w} port carries global dm_idx="
                f"{gdm}, outside expected range "
                f"[{lo}, {lo + per_worker_dm})"
            )
        assert seen_dm == set(range(n_dm_total)), (
            f"expected GLOBAL dm_idx range = {set(range(n_dm_total))}, "
            f"got {seen_dm}"
        )
    finally:
        tx.close()
        for rx in rx_socks:
            rx.close()


# ---------------------------------------------------------------------------
# (c) Multiple blocks in a row — confirm no backpressure under fast consumer
# ---------------------------------------------------------------------------


def test_many_blocks_no_backpressure():
    n_workers = 2
    socks, base_port = _bind_per_worker_ports(base_port=0, n_workers=n_workers)
    n_dm = 4
    n_fv = 16
    n_filled = 64
    dims = CubeShmRingDims(
        n_slots=4, shape=(2, n_fv, n_filled),
        dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=base_port, chgroup=7,
        n_workers=n_workers, n_dm_total=n_dm,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=100.0,
        reserve_timeout_s=2.0,
    )
    tx = AsyncTransportTx.spawn(cfg)
    try:
        cube = _make_cube(n_dm, n_fv, n_filled)
        # Warm-up cube — drain ALL worker ports until at least worker 0
        # has emitted (signals workers are up).
        tx.transmit([cube], block_n=0, rfi_warming_up=False, specnum=0)
        _ = _drain_rx_until_idle(socks[0], idle_s=0.3, first_packet_timeout_s=5.0)
        for s in socks[1:]:
            _ = _drain_rx_until_idle(s, idle_s=0.1, first_packet_timeout_s=1.0)

        n_blocks = 12
        backpressure_before = tx.stats()["n_backpressure_total"]
        for b in range(n_blocks):
            tx.transmit(
                [cube], block_n=b + 1, rfi_warming_up=False, specnum=1000 + b,
            )
            time.sleep(0.02)
        # Drain ALL worker ports.
        total_frames = 0
        for s in socks:
            total_frames += len(_drain_rx_until_idle(s, idle_s=0.4))
        assert total_frames >= n_blocks * n_dm * n_fv, (
            f"expected >= {n_blocks * n_dm * n_fv} frames across all "
            f"workers, got {total_frames}"
        )
        stats = tx.stats()
        # +1 for the warm-up cube.
        assert stats["n_cubes_in"] == n_blocks + 1
        assert stats["n_workers_alive"] == cfg.n_workers
        steady_bp = stats["n_backpressure_total"] - backpressure_before
        assert steady_bp == 0, (
            f"steady-state backpressure should be 0, got {steady_bp}"
        )
    finally:
        tx.close()
        for s in socks:
            s.close()


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


# ---------------------------------------------------------------------------
# M7.2 selective TX mask
# ---------------------------------------------------------------------------


def _spawn_with_mask(mask: int, base_port: int, n_workers: int = 4, n_dm_total: int = 8):
    """Helper: spawn AsyncTransportTx with given coarse_dm_mask."""
    per_worker_dm = n_dm_total // n_workers
    dims = CubeShmRingDims(
        n_slots=4, shape=(per_worker_dm, 8, 32),
        dtype=np.dtype("complex64"),
    )
    cfg = AsyncTransportTxConfig(
        host="127.0.0.1", port=base_port, chgroup=3,
        n_workers=n_workers, n_dm_total=n_dm_total,
        ring_dims=dims,
        pattern_id=_FAKE_PATTERN_ID, n_grid=_FAKE_N_GRID,
        target_gbps_per_flow=100.0,
        coarse_dm_mask=mask,
    )
    return AsyncTransportTx.spawn(cfg)


def _bind_per_worker_ports(base_port: int, n_workers: int):
    """Bind one RX socket per worker port (base_port + w).

    To handle OS-allocated ports + collision-free sequential binding,
    scan upward from a high-numbered base until ``n_workers`` consecutive
    ports succeed. The returned ``base`` is the first port of the run.
    """
    if base_port == 0:
        import random
        # Probe a high range to find n_workers consecutive free ports.
        for _ in range(64):
            cand = random.randint(20000, 60000 - n_workers)
            socks_try: list[socket.socket] = []
            try:
                for w in range(n_workers):
                    s, _p = _make_rx(port=cand + w)
                    socks_try.append(s)
                return socks_try, cand
            except (OSError, PermissionError):
                for s in socks_try:
                    s.close()
                continue
        raise RuntimeError("could not find n_workers consecutive free ports")
    socks: list[socket.socket] = []
    for w in range(n_workers):
        socks.append(_make_rx(port=base_port + w)[0])
    return socks, base_port


def test_coarse_dm_mask_m72_low_emits_only_worker0():
    """M7.2-low (mask=0x03): only worker 0 (DM[0,2)) transmits; workers
    1-3 drain. Worker 0's port (base_port + 0) receives frames with
    GLOBAL dm_idx in {0, 1}; the other 3 ports stay silent.
    """
    n_workers = 4
    socks, base = _bind_per_worker_ports(base_port=0, n_workers=n_workers)
    tx = _spawn_with_mask(mask=0x03, base_port=base, n_workers=n_workers)
    try:
        cube = _make_cube(8, 8, 32)
        tx.transmit([cube], block_n=10, rfi_warming_up=False, specnum=1)
        w0_frames = _drain_rx_until_idle(socks[0], idle_s=0.6)
        w_other = []
        for w in range(1, n_workers):
            w_other.extend(_drain_rx_until_idle(
                socks[w], idle_s=0.2, first_packet_timeout_s=0.5))
        assert len(w0_frames) > 0, "expected frames on worker-0 port"
        assert len(w_other) == 0, (
            f"expected NO frames on workers 1..3; got {len(w_other)}"
        )
        gdm = {int(unpack_frame(f)[0].dm_idx) for f in w0_frames}
        assert gdm == {0, 1}, (
            f"M7.2-low: worker 0 must emit GLOBAL dm_idx={{0,1}}; got {gdm}"
        )
    finally:
        tx.close()
        for s in socks:
            s.close()


def test_coarse_dm_mask_m72_high_emits_only_worker3():
    """M7.2-high (mask=0xC0): only worker 3 (DM[6,8)) transmits; workers
    0-2 drain. Worker 3's port (base_port + 3) receives frames with
    GLOBAL dm_idx in {6, 7}; the other 3 ports stay silent.
    """
    n_workers = 4
    socks, base = _bind_per_worker_ports(base_port=0, n_workers=n_workers)
    tx = _spawn_with_mask(mask=0xC0, base_port=base, n_workers=n_workers)
    try:
        cube = _make_cube(8, 8, 32)
        tx.transmit([cube], block_n=10, rfi_warming_up=False, specnum=1)
        w3_frames = _drain_rx_until_idle(socks[3], idle_s=0.6)
        w_other = []
        for w in (0, 1, 2):
            w_other.extend(_drain_rx_until_idle(
                socks[w], idle_s=0.2, first_packet_timeout_s=0.5))
        assert len(w3_frames) > 0, "expected frames on worker-3 port"
        assert len(w_other) == 0, (
            f"expected NO frames on workers 0..2; got {len(w_other)}"
        )
        gdm = {int(unpack_frame(f)[0].dm_idx) for f in w3_frames}
        assert gdm == {6, 7}, (
            f"M7.2-high: worker 3 must emit GLOBAL dm_idx={{6,7}}; got {gdm}"
        )
    finally:
        tx.close()
        for s in socks:
            s.close()


def test_coarse_dm_mask_full_emits_all_global_dm():
    """Default mask=0xFF (M7.3): all 4 workers transmit; union of
    GLOBAL dm_idx values across all 4 worker ports == {0..7}.
    """
    n_workers = 4
    socks, base = _bind_per_worker_ports(base_port=0, n_workers=n_workers)
    tx = _spawn_with_mask(mask=0xFF, base_port=base, n_workers=n_workers)
    try:
        cube = _make_cube(8, 8, 32)
        tx.transmit([cube], block_n=10, rfi_warming_up=False, specnum=1)
        seen_gdm: set[int] = set()
        total = 0
        for w in range(n_workers):
            for f in _drain_rx_until_idle(socks[w], idle_s=0.4):
                total += 1
                seen_gdm.add(int(unpack_frame(f)[0].dm_idx))
        assert seen_gdm == set(range(8)), (
            f"expected GLOBAL dm_idx={{0..7}} across all workers; got {seen_gdm}"
        )
        # 8 global DM × 8 fv ⇒ ≥ 64 fragments
        assert total >= 64, f"expected ≥ 64 total frames; got {total}"
    finally:
        tx.close()
        for s in socks:
            s.close()


def test_coarse_dm_mask_partial_overlap_rejected():
    """Mask 0x05 (DM 0, 2) crosses worker boundaries (worker 0 = DM[0,2),
    worker 1 = DM[2,4)) — partial overlap MUST raise at spawn."""
    rx, port = _make_rx()
    try:
        with pytest.raises(ValueError, match="align"):
            _spawn_with_mask(mask=0x05, base_port=port)
    finally:
        rx.close()


def test_coarse_dm_mask_all_disabled_rejected():
    """Mask 0x00 disables every worker — must raise at spawn."""
    rx, port = _make_rx()
    try:
        with pytest.raises(ValueError, match="disables every worker"):
            _spawn_with_mask(mask=0x00, base_port=port)
    finally:
        rx.close()
