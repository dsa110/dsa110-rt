"""Tests for ``dsart.trigger.emitter`` end-to-end against
``MockTriggerListener`` (M5 chunk 4).

Plan §4.4 lines 1669-1718 + §8 line 2328 ``bench/trigger_emitter_wiring.py``.
Coverage:

  * Emit one Candidate; listener receives one TriggerPacket with
    matching field values.
  * Fan-out: Emit one Candidate; ALL N mock listeners receive the
    packet (mirrors the production "16 corr listeners" geometry).
  * trigger_id format: ``s<sid>-g<g>-<10-digit-counter>``.
  * Holdoff suppresses repeat emits within the holdoff window.
  * Predicate chain drops sub-threshold candidates.
  * In-flight tracker: accepts both stages, then evicts entry on
    completed.
  * Reconnect on listener restart: kill one listener mid-run; the
    surviving N-1 keep firing; conn_state for the dead one flips to
    reconnecting; after restart, packets resume.
  * EmitRecord callback fires for every Candidate that enters the
    pipeline (passed + dropped).
  * Halo-flagged candidates are logged but NOT emitted.
"""

from __future__ import annotations

import asyncio
import functools
import os
import time

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.trigger.conditions import (  # noqa: E402
    PerCubePerKernelCap,
    PerCubeTotalCap,
    RateLimitTokenBucket,
    SnrThreshold,
)
from dsart.trigger.emitter import (  # noqa: E402
    ConnectionEndpoint,
    ConnState,
    TriggerEmitter,
    TriggerEmitterConfig,
)
from dsart.trigger.holdoff import HoldoffStateMachine  # noqa: E402
from dsart.trigger.mock_listener import (  # noqa: E402
    MockListenerConfig,
    MockTriggerListener,
    MockTriggerListenerFan,
)


def asyncio_test(coro_fn):
    """Tiny adapter to let `async def` test bodies run under stock pytest
    without requiring pytest-asyncio (which isn't installed in the
    dsa110-rt conda env). Each test gets its own fresh event loop."""

    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro_fn(*args, **kwargs))

    return wrapper


def _cand(*, snr: float = 10.0, kernel_id: str = "psf:d3:b16",
          dm_idx: int = 10, l: float = 4.0, m: float = 4.0,
          flags: int = int(CandidateFlags.NONE)) -> Candidate:
    return Candidate(
        l=l, m=m, dm_fine=float(dm_idx), dm_idx=dm_idx,
        event_specnum=256, width_samples=4,
        kernel_id=kernel_id, snr=snr, detector_version="v1.M5",
        flags=flags, search_node_id=0, gpu_half=0,
    )


def _default_chain():
    return [
        SnrThreshold(min_snr=8.0),
        PerCubePerKernelCap(max_per_kernel=4),
        PerCubeTotalCap(max_total=16),
        RateLimitTokenBucket(rate_per_s=1000.0, burst=1000),
    ]


async def _wait_for(predicate, *, timeout_s: float = 2.0,
                    poll_s: float = 0.005):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        await asyncio.sleep(poll_s)
    return False


@asyncio_test
async def test_emit_one_candidate_to_one_listener() -> None:
    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=2,
            gpu_half=1,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
            holdoff=HoldoffStateMachine(holdoff_ms=50.0),
        )
        async with TriggerEmitter(cfg) as emitter:
            # Wait for the connection to be UP.
            ok = await _wait_for(
                lambda: emitter.conn_state[0] == ConnState.UP
            )
            assert ok, f"connection never came up; state={emitter.conn_state}"

            records = await emitter.process_candidates(
                cube_id=0, candidates=[_cand(snr=10.0)],
            )
            assert len(records) == 1
            assert records[0].predicate_pass
            assert records[0].trigger_id is not None

            # Wait for the listener to receive it.
            ok = await _wait_for(lambda: listener.n_received >= 1)
            assert ok, f"listener never got the packet; got {listener.n_received}"
            r = listener.received[0]
            assert r.snr == pytest.approx(10.0)
            assert r.kernel_id == "psf:d3:b16"
            assert r.accepted


@asyncio_test
async def test_trigger_id_format() -> None:
    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=3, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
        )
        async with TriggerEmitter(cfg) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            recs = await emitter.process_candidates(
                cube_id=0, candidates=[_cand(snr=10.0)],
            )
            tid = recs[0].trigger_id
            assert tid is not None
            # Format: s<sid>-g<g>-<10-digit-counter>
            assert tid.startswith("s3-g0-")
            counter_part = tid.split("-")[-1]
            assert len(counter_part) == 10
            assert counter_part.isdigit()


@asyncio_test
async def test_fan_out_to_all_listeners() -> None:
    async with MockTriggerListenerFan(n=4) as fan:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(h, p) for (h, p) in fan.addrs],
            conditions=_default_chain(),
            holdoff=HoldoffStateMachine(holdoff_ms=50.0),
        )
        async with TriggerEmitter(cfg) as emitter:
            ok = await _wait_for(
                lambda: all(s == ConnState.UP for s in emitter.conn_state),
                timeout_s=3.0,
            )
            assert ok, f"some conns never came up: {emitter.conn_state}"

            await emitter.process_candidates(
                cube_id=0, candidates=[_cand(snr=10.0)],
            )

            ok = await _wait_for(
                lambda: all(l.n_received >= 1 for l in fan.listeners),
                timeout_s=2.0,
            )
            assert ok, f"some listeners didn't receive: " \
                f"{[l.n_received for l in fan.listeners]}"


@asyncio_test
async def test_holdoff_suppresses_repeat() -> None:
    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
            holdoff=HoldoffStateMachine(holdoff_ms=50.0),
        )
        async with TriggerEmitter(cfg) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            now = time.time_ns()
            cands = [_cand(snr=10.0), _cand(snr=10.5)]  # same (l, m, kernel)
            recs = await emitter.process_candidates(
                cube_id=0, candidates=cands, now_utc_ns=now,
            )
            assert recs[0].predicate_pass
            assert not recs[1].predicate_pass
            assert recs[1].holdoff_suppressed
            assert emitter.holdoff_suppressed == 1
            assert emitter.emitted_total == 1


@asyncio_test
async def test_predicate_drops_sub_threshold() -> None:
    """Below-threshold candidates are dropped by the SnrThreshold
    condition; the EmitRecord carries predicate_pass=False + the
    condition reason."""
    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
        )
        async with TriggerEmitter(cfg) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            recs = await emitter.process_candidates(
                cube_id=0, candidates=[_cand(snr=6.0)],
            )
            assert len(recs) == 1
            assert not recs[0].predicate_pass
            assert recs[0].predicate_reason is not None
            assert emitter.dropped.get("snr_threshold", 0) == 1
            # Listener should not have received anything.
            await asyncio.sleep(0.05)
            assert listener.n_received == 0


@asyncio_test
async def test_in_flight_tracker_evicts_on_completion() -> None:
    async with MockTriggerListener(
        config=MockListenerConfig(
            accept_delay_ms=2.0, completed_delay_ms=10.0,
        ),
    ) as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
            holdoff=HoldoffStateMachine(holdoff_ms=50.0),
        )
        async with TriggerEmitter(cfg) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            await emitter.process_candidates(
                cube_id=0, candidates=[_cand(snr=10.0)],
            )
            # Wait for the completed ACK to come back and evict.
            ok = await _wait_for(
                lambda: emitter.in_flight_tracker.in_flight == 0,
                timeout_s=2.0,
            )
            assert ok, (
                f"in-flight not evicted: "
                f"{emitter.in_flight_tracker.in_flight}"
            )
            # Latency stats populated.
            assert emitter.in_flight_tracker.accepted_ack_latency_ns_p50 is not None
            assert emitter.in_flight_tracker.completed_ack_latency_ns_p50 is not None


@asyncio_test
async def test_halo_flagged_candidate_not_emitted() -> None:
    """Plan §4.4 line 1594: halo-dropped candidates are LOGGED but
    NOT emitted."""
    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
        )
        async with TriggerEmitter(cfg) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            cand = _cand(snr=10.0, flags=int(CandidateFlags.HALO_DROPPED))
            recs = await emitter.process_candidates(
                cube_id=0, candidates=[cand],
            )
            assert len(recs) == 1
            assert not recs[0].predicate_pass
            assert recs[0].halo_dropped
            assert emitter.halo_dropped == 1
            assert emitter.emitted_total == 0
            await asyncio.sleep(0.05)
            assert listener.n_received == 0


@asyncio_test
async def test_reconnect_on_listener_restart() -> None:
    """Kill one listener in the fan; verify the surviving listeners
    keep receiving packets and the dead one's conn flips to
    reconnecting. Restart it; verify packets resume."""
    async with MockTriggerListenerFan(n=3) as fan:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(h, p) for (h, p) in fan.addrs],
            conditions=_default_chain(),
            holdoff=HoldoffStateMachine(holdoff_ms=0.0),
            backoff_initial_s=0.05,
            backoff_cap_s=0.2,
        )
        async with TriggerEmitter(cfg) as emitter:
            ok = await _wait_for(
                lambda: all(s == ConnState.UP for s in emitter.conn_state),
                timeout_s=2.0,
            )
            assert ok

            # Emit one trigger per cube, before, during, and after kill.
            await emitter.process_candidates(
                cube_id=0, candidates=[_cand(snr=10.0, l=1.0)],
            )
            await asyncio.sleep(0.1)
            assert all(l.n_received >= 1 for l in fan.listeners)

            # Kill listener 1.
            await fan.stop_listener(1)
            # Trigger another emit; the live listeners should receive
            # the new packet, the dead one should not.
            await emitter.process_candidates(
                cube_id=1, candidates=[_cand(snr=10.0, l=2.0)],
            )
            # Wait for the receiver to notice the disconnect.
            ok = await _wait_for(
                lambda: emitter.conn_state[1] in (
                    ConnState.RECONNECTING, ConnState.CONNECTING,
                ),
                timeout_s=3.0,
            )
            assert ok, f"conn 1 didn't flip to reconnecting: {emitter.conn_state}"
            # Live listeners should have received both triggers.
            ok = await _wait_for(
                lambda: fan.listeners[0].n_received >= 2
                and fan.listeners[2].n_received >= 2,
                timeout_s=2.0,
            )
            assert ok, f"live listeners didn't get both: " \
                f"{[l.n_received for l in fan.listeners]}"

            # Restart listener 1; emit another trigger; verify it
            # receives.
            await fan.restart_listener(1)
            ok = await _wait_for(
                lambda: emitter.conn_state[1] == ConnState.UP,
                timeout_s=3.0,
            )
            assert ok, f"conn 1 didn't recover: {emitter.conn_state}"
            await emitter.process_candidates(
                cube_id=2, candidates=[_cand(snr=10.0, l=3.0)],
            )
            ok = await _wait_for(
                lambda: fan.listeners[1].n_received >= 1,
                timeout_s=2.0,
            )
            assert ok, (
                f"listener 1 didn't receive after restart; got "
                f"{fan.listeners[1].n_received}"
            )


@asyncio_test
async def test_per_emit_record_callback_fires_for_all_candidates() -> None:
    """The on_emit_record callback fires for every Candidate (passed,
    dropped, halo-flagged) — that's the canonical ndjson sink."""
    collected = []

    async def on_record(rec):
        collected.append(rec)

    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=_default_chain(),
        )
        async with TriggerEmitter(cfg, on_emit_record=on_record) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            cands = [
                _cand(snr=10.0),                                     # passes
                _cand(snr=6.0),                                      # snr drop
                _cand(snr=10.0, flags=int(CandidateFlags.HALO_DROPPED)),  # halo
            ]
            await emitter.process_candidates(cube_id=0, candidates=cands)
            assert len(collected) == 3
            assert collected[0].predicate_pass
            assert not collected[1].predicate_pass
            assert collected[2].halo_dropped


@asyncio_test
async def test_rate_limit_caps_emit_rate() -> None:
    """Plan §8 line 2328: fire 100 triggers/s for 30 s, confirm only
    burst + rate × T are sent and the rest counted as
    dropped_ratelimit. Scaled-down to 10 ms / 50 triggers / rate=10
    burst=2 for unit-test speed."""
    chain = [
        SnrThreshold(min_snr=8.0),
        RateLimitTokenBucket(rate_per_s=10.0, burst=2),
    ]
    async with MockTriggerListener() as listener:
        cfg = TriggerEmitterConfig(
            search_node_id=0, gpu_half=0,
            endpoints=[ConnectionEndpoint(listener.host, listener.port)],
            conditions=chain,
            holdoff=HoldoffStateMachine(holdoff_ms=0.0),  # disable holdoff
        )
        async with TriggerEmitter(cfg) as emitter:
            await _wait_for(lambda: emitter.conn_state[0] == ConnState.UP)
            now = time.time_ns()
            # Fire 50 candidates with the SAME timestamp — the bucket
            # gets exactly `burst` = 2 emits, the rest drop.
            cands = [
                _cand(snr=10.0, l=float(i), m=0.0)  # different (l, m) so
                                                    # holdoff doesn't fire
                for i in range(50)
            ]
            await emitter.process_candidates(
                cube_id=0, candidates=cands, now_utc_ns=now,
            )
            assert emitter.emitted_total == 2
            assert emitter.dropped.get("ratelimit", 0) == 48
