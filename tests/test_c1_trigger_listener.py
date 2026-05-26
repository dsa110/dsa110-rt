"""Tests for the M7.4 C2 → C1 UDP trigger listener
(``src/dsart/dump/c2_trigger_listener.py``).

Covers:
  * Hit path: trigger packet → ring lookup → CubeDumpWriter submit
    with the correct ``${dump_root}/<event_name>/cube_s..._g..._N.npz``
    path in the manifest.
  * Miss paths: too-late (older than ring) + too-early (newer than
    ring) → counters incremented, no dispatch.
  * Bad-magic / wrong-size packets → counters incremented, no
    dispatch.
"""

from __future__ import annotations

import asyncio
import functools
import os
import socket
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")


def asyncio_test(func):
    """Run an async coroutine inside a fresh event loop per test
    (mirrors the helper in ``tests/test_udp_listener.py``)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper

from dsart.coinc.wire import (  # noqa: E402
    C2_TRIGGER_FLAG_DUMP_CUBE,
    C2_TRIGGER_PACKET_SIZE,
    C2TriggerPacket,
    encode_c2_trigger,
)
from dsart.common.contracts import CubeDumpManifest  # noqa: E402
from dsart.dump.c2_trigger_listener import (  # noqa: E402
    C2TriggerListener,
    C2TriggerListenerConfig,
)
from dsart.services.cube_pipeline import (  # noqa: E402
    CubeRetentionRing,
    RetainedCube,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _payload(t_det: int, n_fdm: int, n_grid: int, value: float = 1.0) -> np.ndarray:
    return np.full(
        (t_det, n_fdm, n_grid, n_grid), value, dtype=np.float16,
    )


def _stage(ring: CubeRetentionRing, *, cube_id: int, specnum: int) -> RetainedCube:
    return ring.stage_cube(
        cube_id=cube_id,
        event_specnum_start=specnum,
        mjd_start=58000.0 + 1e-6 * cube_id,
        sample_period_specnum=16,
        sample_period_us=1048.576,
        cube_tensor=_payload(4, 2, 4, value=float(cube_id)),
    )


def _send_udp(payload: bytes, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(payload, ("127.0.0.1", port))


def _trigger_packet(event_name: str, event_specnum: int) -> bytes:
    pkt = C2TriggerPacket(
        event_name=event_name,
        event_specnum=event_specnum,
        mjd_target=58000.0,
        trigger_class_id=7,
        flags=C2_TRIGGER_FLAG_DUMP_CUBE,
    )
    return encode_c2_trigger(pkt)


async def _wait_for(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


# ---------------------------------------------------------------------------
# Listener bind / lifecycle
# ---------------------------------------------------------------------------


@asyncio_test
async def test_listener_start_stop(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )
    listener = C2TriggerListener(config=cfg, ring=ring)
    await listener.start()
    assert listener.is_running
    assert listener.bound_port == port
    await listener.stop()
    assert not listener.is_running


# ---------------------------------------------------------------------------
# Hit path
# ---------------------------------------------------------------------------


@asyncio_test
async def test_hit_path_dispatches_to_dump(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=4, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    _stage(ring, cube_id=0, specnum=1000)
    _stage(ring, cube_id=1, specnum=1064)  # window [1064, 1128)
    _stage(ring, cube_id=2, specnum=1128)  # window [1128, 1192)
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=1,
        search_node_id=2,
        dump_root=tmp_path,
    )

    dispatched: List[Tuple[RetainedCube, C2TriggerPacket, CubeDumpManifest]] = []

    def _spy_dispatcher(retained, packet, manifest) -> bool:
        dispatched.append((retained, packet, manifest))
        return True

    listener = C2TriggerListener(
        config=cfg, ring=ring, dispatcher=_spy_dispatcher,
    )
    await listener.start()
    try:
        _send_udp(_trigger_packet("evt_alpha", 1150), listener.bound_port)
        ok = await _wait_for(lambda: len(dispatched) == 1, timeout_s=2.0)
        assert ok, listener.mon
        retained, packet, manifest = dispatched[0]
        assert retained.cube_id == 2
        assert packet.event_name == "evt_alpha"
        assert packet.event_specnum == 1150
        assert manifest.trigger_source == "udp"
        assert manifest.cube_id == 2
        assert manifest.event_specnum_start == 1150
        assert manifest.search_node_id == 2
        assert manifest.gpu_half == 1
        expected = (
            tmp_path / "evt_alpha" / "cube_s2_g1_1150.npz"
        )
        assert Path(manifest.npz_path) == expected
        mon = listener.mon
        assert mon["received"] == 1
        assert mon["hits"] == 1
        assert mon["dispatched"] == 1
        assert mon["too_late"] == 0
        assert mon["too_early"] == 0
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# Miss paths
# ---------------------------------------------------------------------------


@asyncio_test
async def test_too_late_miss(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    _stage(ring, cube_id=10, specnum=2000)
    _stage(ring, cube_id=11, specnum=2064)
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )
    dispatched: List = []
    listener = C2TriggerListener(
        config=cfg, ring=ring,
        dispatcher=lambda r, p, m: (dispatched.append(p), True)[1],
    )
    await listener.start()
    try:
        # specnum 1000 is older than the ring's oldest (2000).
        _send_udp(_trigger_packet("old_evt", 1000), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_late"] >= 1, timeout_s=2.0)
        assert ok, listener.mon
        assert len(dispatched) == 0
        assert listener.mon["hits"] == 0
    finally:
        await listener.stop()


@asyncio_test
async def test_too_early_miss(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    _stage(ring, cube_id=20, specnum=2000)
    _stage(ring, cube_id=21, specnum=2064)
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )
    dispatched: List = []
    listener = C2TriggerListener(
        config=cfg, ring=ring,
        dispatcher=lambda r, p, m: (dispatched.append(p), True)[1],
    )
    await listener.start()
    try:
        # specnum 1_000_000 is way beyond the newest end.
        _send_udp(_trigger_packet("future", 1_000_000), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_early"] >= 1, timeout_s=2.0)
        assert ok, listener.mon
        assert len(dispatched) == 0
    finally:
        await listener.stop()


@asyncio_test
async def test_empty_ring_miss_counts_as_too_early(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )
    listener = C2TriggerListener(config=cfg, ring=ring)
    await listener.start()
    try:
        _send_udp(_trigger_packet("evt", 12345), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_early"] >= 1, timeout_s=2.0)
        assert ok, listener.mon
        assert listener.mon["hits"] == 0
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# Bad packets
# ---------------------------------------------------------------------------


@asyncio_test
async def test_bad_magic_dropped(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    _stage(ring, cube_id=0, specnum=1000)
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )
    listener = C2TriggerListener(config=cfg, ring=ring)
    await listener.start()
    try:
        # 64 bytes of zeros — wrong magic.
        bad = b"\x00" * C2_TRIGGER_PACKET_SIZE
        _send_udp(bad, listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["bad_magic"] >= 1, timeout_s=2.0)
        assert ok, listener.mon
        assert listener.mon["hits"] == 0
    finally:
        await listener.stop()


@asyncio_test
async def test_short_packet_dropped(tmp_path: Path) -> None:
    ring = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    _stage(ring, cube_id=0, specnum=1000)
    port = _free_udp_port()
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=port,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )
    listener = C2TriggerListener(config=cfg, ring=ring)
    await listener.start()
    try:
        _send_udp(b"abc", listener.bound_port)
        ok = await _wait_for(
            lambda: listener.mon["bad_schema"] >= 1, timeout_s=2.0,
        )
        assert ok, listener.mon
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# bind_port helper
# ---------------------------------------------------------------------------


def test_bind_port_offset_property() -> None:
    cfg0 = C2TriggerListenerConfig(
        bind_host="127.0.0.1", base_port=11227, gpu_half=0,
        search_node_id=1, dump_root=Path("/tmp"),
    )
    cfg1 = C2TriggerListenerConfig(
        bind_host="127.0.0.1", base_port=11227, gpu_half=1,
        search_node_id=1, dump_root=Path("/tmp"),
    )
    assert cfg0.bind_port == 11227
    assert cfg1.bind_port == 11228
