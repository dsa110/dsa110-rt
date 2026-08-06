"""Bounded retry of ``too_early`` C2 cube requests (2026-08-06).

Context: the c1_emit dedicated-sender fix cut the C1 -> C2 ->
dump-request round trip to ~1 s, so a request now regularly arrives
before the slower halves have produced the cube holding the event
(post-restart their frontiers trail the detecting half by 1-60 s and
never catch up). The listener used to treat that as a terminal miss;
it now parks the request and fulfils it when the ring's frontier
reaches the event.

Covers:
  * park -> ring advances -> dump fires with a manifest identical to
    an in-window request's, plus the operator INFO line;
  * retry timeout -> terminal WARNING miss, no dump, and the listener
    still serves later in-window requests;
  * ``too_late`` unchanged (immediate miss, never parked);
  * two independent parks fulfilling in frontier order.

Torch-free by construction: ``dsart.services.cube_pipeline`` (which
imports torch, unavailable in the CPU test env -- see the collection
error on ``tests/test_c1_trigger_listener.py``) is replaced by a
lightweight fake ring module BEFORE the listener import below. The
fake reproduces the two ring behaviours the listener depends on:
newest-first iteration and the ``[start, start + t_det)`` containment
rule of ``find_cube_for_specnum``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import socket
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")


# ---------------------------------------------------------------------------
# Fake cube_pipeline (installed before the listener import)
# ---------------------------------------------------------------------------


@dataclass
class RetainedCube:
    """Mirror of ``cube_pipeline.RetainedCube`` (fields the listener
    reads)."""

    cube_id: int
    event_specnum_start: int
    mjd_start: float
    t_det: int
    n_fdm: int
    n_grid: int
    sample_period_specnum: int
    sample_period_us: float
    pinned_host_tensor: Any


class CubeRetentionRing:
    """Minimal stand-in for the real retention ring.

    Only what the listener touches: ``snapshot`` / ``iter_newest_first``
    (newest first) and the inflight mark/release pair.
    """

    def __init__(self, depth: int = 8, t_det: int = 4) -> None:
        self._depth = int(depth)
        self._t_det = int(t_det)
        self._lock = threading.Lock()
        self._cubes: List[RetainedCube] = []  # oldest -> newest
        self.inflight: List[Any] = []
        self.released: List[Any] = []

    def stage_cube(self, *, cube_id: int, event_specnum_start: int) -> RetainedCube:
        cube = RetainedCube(
            cube_id=int(cube_id),
            event_specnum_start=int(event_specnum_start),
            mjd_start=58000.0 + 1e-6 * int(cube_id),
            t_det=self._t_det,
            n_fdm=2,
            n_grid=4,
            sample_period_specnum=16,
            sample_period_us=1048.576,
            pinned_host_tensor=np.full(
                (self._t_det, 2, 4, 4), float(cube_id), dtype=np.float16,
            ),
        )
        with self._lock:
            self._cubes.append(cube)
            if len(self._cubes) > self._depth:
                self._cubes.pop(0)
        return cube

    def snapshot(self) -> List[RetainedCube]:
        with self._lock:
            return list(reversed(self._cubes))

    def iter_newest_first(self) -> Iterator[RetainedCube]:
        return iter(self.snapshot())

    def mark_inflight(self, buf: Any) -> None:
        self.inflight.append(buf)

    def release_inflight(self, buf: Any) -> None:
        self.released.append(buf)


def find_cube_for_specnum(
    ring: CubeRetentionRing, event_specnum: int,
) -> Optional[RetainedCube]:
    for cube in ring.iter_newest_first():
        start = int(cube.event_specnum_start)
        if start <= int(event_specnum) < start + int(cube.t_det):
            return cube
    return None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "no torch"
        return False
    return True


# Only shim when the real module genuinely cannot import, and pull the
# shim back out afterwards so no OTHER test module in the session ever
# picks up the fake instead of the real cube_pipeline.
_SHIMMED = False
if (
    not _torch_available()
    and "dsart.services.cube_pipeline" not in sys.modules
):
    _fake = types.ModuleType("dsart.services.cube_pipeline")
    _fake.RetainedCube = RetainedCube
    _fake.CubeRetentionRing = CubeRetentionRing
    _fake.find_cube_for_specnum = find_cube_for_specnum
    sys.modules["dsart.services.cube_pipeline"] = _fake
    _SHIMMED = True


from dsart.coinc.wire import (  # noqa: E402
    C2_TRIGGER_FLAG_DUMP_CUBE,
    C2TriggerPacket,
    encode_c2_trigger,
)
from dsart.dump.c2_trigger_listener import (  # noqa: E402
    C2TriggerListener,
    C2TriggerListenerConfig,
)

if _SHIMMED:
    sys.modules.pop("dsart.services.cube_pipeline", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def asyncio_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _send_udp(payload: bytes, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(payload, ("127.0.0.1", port))


def _trigger_packet(event_name: str, event_specnum: int) -> bytes:
    return encode_c2_trigger(
        C2TriggerPacket(
            event_name=event_name,
            event_specnum=event_specnum,
            mjd_target=58000.0,
            trigger_class_id=7,
            flags=C2_TRIGGER_FLAG_DUMP_CUBE,
        )
    )


async def _wait_for(predicate, timeout_s: float = 3.0, interval_s: float = 0.01) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


def _make_listener(
    tmp_path: Path,
    ring: CubeRetentionRing,
    *,
    retry_timeout_s: float,
    dispatched: List[Tuple[Any, Any, Any]],
) -> C2TriggerListener:
    cfg = C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=_free_udp_port(),
        gpu_half=1,
        search_node_id=2,
        dump_root=tmp_path,
        too_early_retry_timeout_s=retry_timeout_s,
    )

    def _spy(retained, packet, manifest) -> bool:
        dispatched.append((retained, packet, manifest))
        return True

    return C2TriggerListener(
        config=cfg,
        ring=ring,
        dispatcher=_spy,
        # Fast polling so the tests run in well under a second; the
        # production cadence is C2TriggerListener._RETRY_POLL_INTERVAL_S.
        retry_poll_interval_s=0.02,
    )


# ---------------------------------------------------------------------------
# 1. Park -> frontier arrives -> fulfilment
# ---------------------------------------------------------------------------


@asyncio_test
async def test_too_early_parks_then_fulfils_when_ring_advances(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="dsart.dump.c2_trigger_listener")
    ring = CubeRetentionRing(depth=8, t_det=4)
    ring.stage_cube(cube_id=0, event_specnum_start=1000)  # [1000, 1004)
    dispatched: List[Tuple[Any, Any, Any]] = []
    listener = _make_listener(
        tmp_path, ring, retry_timeout_s=30.0, dispatched=dispatched,
    )
    await listener.start()
    try:
        # 1064 is beyond the newest cube's end: too_early, so parked.
        _send_udp(_trigger_packet("evt_late_half", 1065), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_early_parked"] == 1)
        assert ok, listener.mon
        assert listener.n_parked == 1
        assert not dispatched
        assert listener.mon["too_early"] == 0  # not a terminal miss

        # The half keeps chewing through cubes; still short of the event.
        ring.stage_cube(cube_id=1, event_specnum_start=1004)
        await asyncio.sleep(0.05)
        assert not dispatched

        # Frontier reaches the event -> the parked request fulfils.
        ring.stage_cube(cube_id=2, event_specnum_start=1064)  # [1064, 1068)
        ok = await _wait_for(lambda: len(dispatched) == 1)
        assert ok, listener.mon
        retained, packet, manifest = dispatched[0]
        assert retained.cube_id == 2
        assert packet.event_specnum == 1065
        assert listener.n_parked == 0
        mon = listener.mon
        assert mon["hits"] == 1
        assert mon["dispatched"] == 1
        assert mon["too_early_fulfilled"] == 1
        assert mon["too_early"] == 0
        assert mon["too_late"] == 0
        assert any(
            "too_early request fulfilled after" in rec.getMessage()
            and rec.levelno == logging.INFO
            for rec in caplog.records
        ), [r.getMessage() for r in caplog.records]

        # The staged output is identical to a fresh in-window request's.
        _send_udp(_trigger_packet("evt_control", 1065), listener.bound_port)
        ok = await _wait_for(lambda: len(dispatched) == 2)
        assert ok, listener.mon
        _, _, control_manifest = dispatched[1]
        assert manifest.cube_id == control_manifest.cube_id
        assert manifest.event_specnum_start == control_manifest.event_specnum_start
        assert manifest.cube_specnum_start == control_manifest.cube_specnum_start
        assert manifest.t_det == control_manifest.t_det
        assert manifest.trigger_source == control_manifest.trigger_source
        assert manifest.search_node_id == control_manifest.search_node_id
        assert manifest.gpu_half == control_manifest.gpu_half
        # Only the event dir differs (different event names).
        assert Path(manifest.npz_path) == (
            tmp_path / "evt_late_half" / "cube_s2_g1_1065.npz"
        )
        assert Path(control_manifest.npz_path) == (
            tmp_path / "evt_control" / "cube_s2_g1_1065.npz"
        )
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 2. Retry timeout
# ---------------------------------------------------------------------------


@asyncio_test
async def test_too_early_retry_timeout_is_a_terminal_miss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="dsart.dump.c2_trigger_listener")
    ring = CubeRetentionRing(depth=8, t_det=4)
    ring.stage_cube(cube_id=0, event_specnum_start=1000)
    dispatched: List[Tuple[Any, Any, Any]] = []
    listener = _make_listener(
        tmp_path, ring, retry_timeout_s=0.2, dispatched=dispatched,
    )
    await listener.start()
    try:
        # The ring never advances past 1004, so this never fulfils.
        _send_udp(_trigger_packet("evt_stuck", 9000), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_early"] == 1)
        assert ok, listener.mon
        assert not dispatched
        assert listener.n_parked == 0
        assert listener.mon["too_early_parked"] == 1
        assert listener.mon["too_early_fulfilled"] == 0
        msgs = [
            rec.getMessage() for rec in caplog.records
            if rec.levelno == logging.WARNING
        ]
        assert any(
            "miss (too_early, retry timed out after" in m
            and "event=evt_stuck" in m
            and "specnum=9000" in m
            and "ring window" in m
            for m in msgs
        ), msgs

        # The listener kept serving: an in-window request still dumps.
        _send_udp(_trigger_packet("evt_ok", 1002), listener.bound_port)
        ok = await _wait_for(lambda: len(dispatched) == 1)
        assert ok, listener.mon
        assert dispatched[0][1].event_name == "evt_ok"
        assert listener.mon["hits"] == 1
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 3. too_late is unchanged
# ---------------------------------------------------------------------------


@asyncio_test
async def test_too_late_is_still_an_immediate_miss(tmp_path: Path) -> None:
    ring = CubeRetentionRing(depth=8, t_det=4)
    ring.stage_cube(cube_id=0, event_specnum_start=2000)
    ring.stage_cube(cube_id=1, event_specnum_start=2004)
    dispatched: List[Tuple[Any, Any, Any]] = []
    listener = _make_listener(
        # A long retry timeout must not delay (or park) a too_late miss.
        tmp_path, ring, retry_timeout_s=600.0, dispatched=dispatched,
    )
    await listener.start()
    try:
        _send_udp(_trigger_packet("evt_gone", 1000), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_late"] == 1)
        assert ok, listener.mon
        assert listener.n_parked == 0
        assert listener.mon["too_early_parked"] == 0
        assert not dispatched
        # Even once the ring rolls forward, nothing resurrects it.
        ring.stage_cube(cube_id=2, event_specnum_start=2008)
        await asyncio.sleep(0.1)
        assert not dispatched
        assert listener.mon["hits"] == 0
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 4. Independent parks
# ---------------------------------------------------------------------------


@asyncio_test
async def test_two_parked_requests_fulfil_independently_in_order(
    tmp_path: Path,
) -> None:
    ring = CubeRetentionRing(depth=8, t_det=4)
    ring.stage_cube(cube_id=0, event_specnum_start=1000)
    dispatched: List[Tuple[Any, Any, Any]] = []
    listener = _make_listener(
        tmp_path, ring, retry_timeout_s=30.0, dispatched=dispatched,
    )
    await listener.start()
    try:
        _send_udp(_trigger_packet("evt_near", 1101), listener.bound_port)
        _send_udp(_trigger_packet("evt_far", 1201), listener.bound_port)
        ok = await _wait_for(lambda: listener.mon["too_early_parked"] == 2)
        assert ok, listener.mon
        assert listener.n_parked == 2

        # A duplicate of an already-parked request must not park twice
        # (C2 re-sends would otherwise dump the same cube twice).
        _send_udp(_trigger_packet("evt_far", 1201), listener.bound_port)
        await asyncio.sleep(0.1)
        assert listener.mon["too_early_parked"] == 2
        assert listener.n_parked == 2

        # Frontier reaches the nearer event only.
        ring.stage_cube(cube_id=1, event_specnum_start=1100)  # [1100, 1104)
        ok = await _wait_for(lambda: len(dispatched) == 1)
        assert ok, listener.mon
        assert dispatched[0][1].event_name == "evt_near"
        assert listener.n_parked == 1

        # ... then the farther one.
        ring.stage_cube(cube_id=2, event_specnum_start=1200)  # [1200, 1204)
        ok = await _wait_for(lambda: len(dispatched) == 2)
        assert ok, listener.mon
        assert dispatched[1][1].event_name == "evt_far"
        assert listener.n_parked == 0
        mon = listener.mon
        assert mon["too_early_fulfilled"] == 2
        assert mon["hits"] == 2
        assert mon["dispatched"] == 2
        assert mon["too_early"] == 0
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# Knob wiring
# ---------------------------------------------------------------------------


def test_retry_timeout_zero_restores_terminal_too_early(tmp_path: Path) -> None:
    """``too_early_retry_timeout_s: 0`` is the documented opt-out."""
    ring = CubeRetentionRing(depth=8, t_det=4)
    ring.stage_cube(cube_id=0, event_specnum_start=1000)
    dispatched: List[Tuple[Any, Any, Any]] = []
    listener = _make_listener(
        tmp_path, ring, retry_timeout_s=0.0, dispatched=dispatched,
    )
    # Drive the handler directly -- no socket needed for this one.
    listener._handle_trigger(
        C2TriggerPacket(
            event_name="evt_no_retry",
            event_specnum=9999,
            mjd_target=58000.0,
            trigger_class_id=0,
            flags=C2_TRIGGER_FLAG_DUMP_CUBE,
        ),
        ("127.0.0.1", 1),
    )
    assert listener.mon["too_early"] == 1
    assert listener.mon["too_early_parked"] == 0
    assert listener.n_parked == 0
    assert not dispatched
