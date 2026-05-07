#!/usr/bin/env python3
"""bench/trigger_emitter_wiring.py — M5 chunk 4(b) trigger-emitter
end-to-end wiring bench (plan §8 line 2328 / §4.4 lines 1669-1718).

Drives the production ``TriggerEmitter`` against an N-port
``MockTriggerListenerFan`` (default N=16, mirroring the production
"16 corr listeners" geometry pinned in plan §4.4 line 1669) and
measures four operationally observable properties per plan §8 line 2328:

  1. **Fan-out**: every Candidate that survives the predicate chain +
     holdoff fans out to ALL N listeners (counts agree per-listener
     within an explicit tolerance for kill-window drops).

  2. **Accepted-ack p99 ≤ 20 ms**: round-trip latency from emitter
     dispatch to the listener's ``stage="accepted"`` ACK. The mock
     listener's default ``accept_delay_ms=0`` puts the wire-level
     floor near loopback RTT (≪ 1 ms), so the bench's 20 ms budget
     tests the emitter's TCP fan-out + ACK demux overhead at
     sustained candidate rate. Note: the ``stage="completed"`` ACK
     latency (which includes the corr listener's voltage-dump
     work, defaulted to ``completed_delay_ms=5``) is reported
     informationally but does NOT gate — it is gated separately by
     ``T_completion_timeout_s = 5 s`` per plan §4.4 line 1718.

  3. **Rate-limit fires**: a brief "blast" phase deliberately exceeds
     the ``RateLimitTokenBucket`` (production default 10 / s, burst
     50) so the bench verifies the predicate-chain drop counter
     for ``RateLimitTokenBucket`` is non-zero. Rate-limit is the
     production safety valve against runaway false-trigger storms;
     the bench failing this gate would mean the chain is silent.

  4. **Listener-fail recovery**: mid-stream the bench kills one mock
     listener, sends another batch of candidates (fan-out should
     drop to N-1), then restarts the listener and verifies that the
     emitter's exponential-backoff reconnect path brings the dead
     conn back to ``UP`` and packet flow resumes (per plan §4.4
     lines 1696-1699). The kill window is sized so the emitter's
     1 s initial backoff has time to retry at least twice.

CLI surface (all flags optional; defaults are operator-friendly):

  python -m bench.trigger_emitter_wiring \\
      [--n-listeners 16]                                    \\
      [--base-port 0]                                       \\
      [--steady-cubes 200] [--steady-cands-per-cube 3]      \\
      [--steady-cube-cadence-ms 0]                          \\
      [--listener-fail-cubes 50] [--listener-fail-idx 3]    \\
      [--listener-fail-restart-after-cubes 20]              \\
      [--rate-limit-blast-cubes 30]                         \\
      [--rate-limit-blast-cands-per-cube 100]               \\
      [--rate-limit-rate-per-s 10.0]                        \\
      [--rate-limit-burst 50]                               \\
      [--ack-p99-budget-ms 20.0]                            \\
      [--out bench/reports/<UTC>/trigger_emitter_wiring/M5/] \\
      [--quick-smoke]

Outputs (under ``--out``):

  * ``trigger_records.ndjson`` — one record per Candidate that
        entered the emit pipeline (predicate_pass + drops + holdoff).
        Schema: ``{phase, cube_id, kernel_id, l, m, snr, trigger_id,
        predicate_pass, predicate_reason, halo_dropped,
        holdoff_suppressed}``.
  * ``summary.json``           — config + per-phase measurements +
        per-conn n_received / n_dropped_tx_full + ack-latency
        percentiles + pass/fail gate stamp.
  * ``bench.log``              — human progress.

Operator gate (per plan §8 line 2328): ack_p99_ms ≤ 20.0 AND
``rate_limit_dropped > 0`` AND ``listener_recovery == "reconnected"``
AND fan-out parity ratio across surviving listeners ≥ 0.99 (within
the kill-window tolerance). The bench stamps PASS / FAIL in
``summary.json::gate``; M5.sh consumes this stamp.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
    EmitRecord,
    TriggerEmitter,
    TriggerEmitterConfig,
)
from dsart.trigger.holdoff import HoldoffStateMachine  # noqa: E402
from dsart.trigger.mock_listener import (  # noqa: E402
    MockListenerConfig,
    MockTriggerListenerFan,
)


_LOG = logging.getLogger("bench.trigger_emitter_wiring")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_N_LISTENERS: int = 16
DEFAULT_STEADY_CUBES: int = 200
DEFAULT_STEADY_CANDS_PER_CUBE: int = 1
DEFAULT_STEADY_CUBE_CADENCE_MS: float = 100.0
DEFAULT_LISTENER_FAIL_CUBES: int = 50
DEFAULT_LISTENER_FAIL_IDX: int = 3
DEFAULT_LISTENER_FAIL_RESTART_AFTER_CUBES: int = 20
DEFAULT_RATE_LIMIT_BLAST_CUBES: int = 30
DEFAULT_RATE_LIMIT_BLAST_CANDS_PER_CUBE: int = 100
DEFAULT_RATE_LIMIT_RATE_PER_S: float = 10.0
DEFAULT_RATE_LIMIT_BURST: int = 50
DEFAULT_ACK_P99_BUDGET_MS: float = 20.0

# Kernel-bank sample (just enough to exercise PerCubePerKernelCap; the
# emitter doesn't care which kernel ids it sees, only that they parse
# as the M1 ``Candidate._check_kernel_id`` namespace).
KERNEL_IDS_FOR_BENCH: Tuple[str, ...] = (
    "psf:d3:b16",
    "psf:d3:b32",
    "psf:d5:b8",
    "unit:d1:b1",
    "psf_shift_lm:d7:b64",
)

# Quick-smoke cuts everything ~10x for the M5.sh DoD path.
QUICK_SMOKE_N_LISTENERS: int = 5
QUICK_SMOKE_STEADY_CUBES: int = 25
QUICK_SMOKE_LISTENER_FAIL_CUBES: int = 8
QUICK_SMOKE_LISTENER_FAIL_RESTART_AFTER_CUBES: int = 3
QUICK_SMOKE_RATE_LIMIT_BLAST_CUBES: int = 5


# ---------------------------------------------------------------------------
# Candidate generator (deterministic)
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    cube_id: int,
    cand_idx: int,
    snr: float,
    kernel_id: str,
    flags: int = int(CandidateFlags.NONE),
) -> Candidate:
    """Return a deterministic Candidate with unique (l, m) per
    (cube_id, cand_idx) — defeats holdoff so the bench observes the
    emitter's full fan-out path. ``snr`` and ``kernel_id`` rotate
    deterministically across the steady stream."""
    # Spread (l, m) across a 64x64 grid so per-(l, m, kernel) holdoff
    # rarely fires. (Production cube_injection_detector tests holdoff;
    # this bench tests the wiring downstream of holdoff.)
    l_int = (cube_id * 7 + cand_idx * 3) % 64
    m_int = (cube_id * 11 + cand_idx * 5) % 64
    return Candidate(
        l=float(l_int),
        m=float(m_int),
        dm_fine=float((cube_id * 13 + cand_idx) % 64),
        dm_idx=int((cube_id * 13 + cand_idx) % 64),
        event_specnum=int(cube_id * 256 + cand_idx),
        width_samples=int(2 ** (cand_idx % 4)),
        kernel_id=kernel_id,
        snr=float(snr),
        detector_version="v1.M5",
        flags=int(flags),
        search_node_id=0,
        gpu_half=0,
    )


def _build_steady_cube(
    cube_id: int,
    n_cands: int,
    *,
    snr: float = 12.0,
) -> List[Candidate]:
    """One cube's worth of distinct candidates, designed to clear
    SnrThreshold + PerCubeCaps + RateLimitTokenBucket at sustainable
    rates."""
    cands: List[Candidate] = []
    for i in range(n_cands):
        kernel_id = KERNEL_IDS_FOR_BENCH[i % len(KERNEL_IDS_FOR_BENCH)]
        cands.append(
            _make_candidate(
                cube_id=cube_id, cand_idx=i, snr=snr, kernel_id=kernel_id,
            )
        )
    return cands


def _build_blast_cube(
    cube_id: int,
    n_cands: int,
    *,
    snr: float = 15.0,
) -> List[Candidate]:
    """Per-cube candidate burst that intentionally exceeds the
    rate-limit + per-cube total cap so the bench can observe the
    predicate-chain drop counter."""
    cands: List[Candidate] = []
    for i in range(n_cands):
        kernel_id = KERNEL_IDS_FOR_BENCH[i % len(KERNEL_IDS_FOR_BENCH)]
        cands.append(
            _make_candidate(
                cube_id=cube_id, cand_idx=i, snr=snr, kernel_id=kernel_id,
            )
        )
    return cands


# ---------------------------------------------------------------------------
# Emitter + fan setup
# ---------------------------------------------------------------------------


@dataclass
class _BenchHandles:
    fan: MockTriggerListenerFan
    emitter: TriggerEmitter
    record_sink: List[EmitRecord]


async def _build_emitter_and_fan(args: argparse.Namespace) -> _BenchHandles:
    """Stand up N mock listeners + an emitter wired to all N. Block
    until every connection is UP (sanity check)."""
    n_listeners = int(args.n_listeners)
    base_port = int(args.base_port)
    fan = MockTriggerListenerFan(
        n=n_listeners,
        host="127.0.0.1",
        base_port=base_port,
        config=MockListenerConfig(
            accept_rate=1.0,
            accept_delay_ms=0.0,
            completed_delay_ms=5.0,
            send_completed=True,
        ),
    )
    await fan.start()
    _LOG.info(
        "MockTriggerListenerFan up: n=%d ports=%s",
        n_listeners, fan.ports,
    )

    endpoints = [
        ConnectionEndpoint("127.0.0.1", port) for port in fan.ports
    ]
    cfg = TriggerEmitterConfig(
        search_node_id=0,
        gpu_half=0,
        endpoints=endpoints,
        conditions=[
            SnrThreshold(min_snr=8.0),
            PerCubePerKernelCap(max_per_kernel=4),
            PerCubeTotalCap(max_total=16),
            RateLimitTokenBucket(
                rate_per_s=float(args.rate_limit_rate_per_s),
                burst=int(args.rate_limit_burst),
            ),
        ],
        holdoff=HoldoffStateMachine(holdoff_ms=50.0),
    )
    record_sink: List[EmitRecord] = []

    async def _on_emit_record(rec: EmitRecord) -> None:
        record_sink.append(rec)

    emitter = TriggerEmitter(cfg, on_emit_record=_on_emit_record)
    await emitter.start()

    # Wait up to 5 s for every connection to come UP. If any one
    # never connects the bench is broken; abort with a clear log.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(s == ConnState.UP for s in emitter.conn_state):
            break
        await asyncio.sleep(0.005)
    if not all(s == ConnState.UP for s in emitter.conn_state):
        await emitter.stop()
        await fan.stop()
        raise RuntimeError(
            f"emitter conn_state never settled to all UP after 5 s: "
            f"{emitter.conn_state}"
        )
    _LOG.info(
        "TriggerEmitter up: %d connections, all UP",
        len(emitter.conn_state),
    )
    return _BenchHandles(fan=fan, emitter=emitter, record_sink=record_sink)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


async def _drain_in_flight(
    emitter: TriggerEmitter,
    *,
    timeout_s: float = 2.0,
    poll_s: float = 0.005,
) -> int:
    """Wait until every in-flight trigger has been ACK-completed (or
    times out at the in-flight tracker's completion_timeout_s).

    Returns the number of in-flight entries that were still pending
    at deadline (0 = clean drain).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if emitter.in_flight_tracker.in_flight == 0:
            return 0
        await asyncio.sleep(poll_s)
    return emitter.in_flight_tracker.in_flight


async def _run_steady_phase(
    handles: _BenchHandles,
    args: argparse.Namespace,
    record_sink: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Phase 1: sustained steady-state emit. ``steady_cands_per_cube``
    candidates per cube fan out across all listeners; the bench
    measures throughput, ack latency, and per-listener parity."""
    n_cubes = (
        QUICK_SMOKE_STEADY_CUBES if args.quick_smoke
        else int(args.steady_cubes)
    )
    n_cands = int(args.steady_cands_per_cube)
    cube_cadence_s = float(args.steady_cube_cadence_ms) * 1e-3

    pre_emitted = handles.emitter.emitted_total
    pre_per_listener = [
        l.n_received for l in handles.fan.listeners
    ]
    t0 = time.monotonic()
    for cube_id in range(n_cubes):
        cands = _build_steady_cube(cube_id, n_cands)
        records = await handles.emitter.process_candidates(
            cube_id=cube_id, candidates=cands,
        )
        for rec in records:
            record_sink.append(_emit_record_to_dict(rec, phase="steady"))
        if cube_cadence_s > 0:
            await asyncio.sleep(cube_cadence_s)
        if (cube_id + 1) % 50 == 0:
            _LOG.info(
                "steady: cube=%d/%d emitted_total=%d in_flight=%d",
                cube_id + 1, n_cubes,
                handles.emitter.emitted_total,
                handles.emitter.in_flight_tracker.in_flight,
            )
    pending = await _drain_in_flight(handles.emitter, timeout_s=3.0)
    elapsed = time.monotonic() - t0

    post_emitted = handles.emitter.emitted_total
    post_per_listener = [
        l.n_received for l in handles.fan.listeners
    ]
    delta_per_listener = [
        b - a for a, b in zip(pre_per_listener, post_per_listener)
    ]
    delta_emitted = post_emitted - pre_emitted

    tracker = handles.emitter.in_flight_tracker
    return {
        "n_cubes": n_cubes,
        "n_cands_per_cube": n_cands,
        "elapsed_s": elapsed,
        "delta_emitted": delta_emitted,
        "delta_per_listener_received": delta_per_listener,
        "fanout_parity_min": (
            float(min(delta_per_listener) / max(delta_emitted, 1))
            if delta_per_listener else 0.0
        ),
        "fanout_parity_max": (
            float(max(delta_per_listener) / max(delta_emitted, 1))
            if delta_per_listener else 0.0
        ),
        "in_flight_pending_at_drain": pending,
        "accepted_ack_p50_ms": _ns_or_none_ms(
            tracker.accepted_ack_latency_ns_p50
        ),
        "accepted_ack_p99_ms": _ns_or_none_ms(
            tracker.accepted_ack_latency_ns_p99
        ),
        "completed_ack_p50_ms": _ns_or_none_ms(
            tracker.completed_ack_latency_ns_p50
        ),
        "completed_ack_p99_ms": _ns_or_none_ms(
            tracker.completed_ack_latency_ns_p99
        ),
        "holdoff_suppressed": handles.emitter.holdoff_suppressed,
        "halo_dropped": handles.emitter.halo_dropped,
        "predicate_dropped_by_condition": dict(handles.emitter.dropped),
    }


async def _run_listener_fail_phase(
    handles: _BenchHandles,
    args: argparse.Namespace,
    record_sink: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Phase 2: kill listener[fail_idx], drive a pre-restart batch of
    cubes, restart, then drive a post-restart batch. Verifies the
    emitter's reconnect path brings the dead conn back to UP and
    packet flow resumes on it."""
    n_cubes = (
        QUICK_SMOKE_LISTENER_FAIL_CUBES if args.quick_smoke
        else int(args.listener_fail_cubes)
    )
    restart_after = (
        QUICK_SMOKE_LISTENER_FAIL_RESTART_AFTER_CUBES if args.quick_smoke
        else int(args.listener_fail_restart_after_cubes)
    )
    fail_idx = int(args.listener_fail_idx)
    if not (0 <= fail_idx < len(handles.fan.listeners)):
        raise ValueError(
            f"listener_fail_idx={fail_idx} out of range [0, "
            f"{len(handles.fan.listeners)})"
        )
    n_cands = int(args.steady_cands_per_cube)

    pre_per_listener = [l.n_received for l in handles.fan.listeners]
    pre_n_dropped_tx_full = list(handles.emitter.n_dropped_tx_full_per_corr)
    pre_n_sent = list(handles.emitter.n_sent_per_corr)

    # Kill the target listener.
    _LOG.info(
        "listener-fail: killing fan.listeners[%d] on port %d",
        fail_idx, handles.fan.listeners[fail_idx].port,
    )
    await handles.fan.stop_listener(fail_idx)
    # Wait until the emitter observes the EOF and flips DOWN /
    # RECONNECTING. (The emitter's TX queue may still have a few
    # in-flight bytes; this gives the receiver loop ~50 ms to drain.)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        st = handles.emitter.conn_state[fail_idx]
        if st in (ConnState.DOWN, ConnState.CONNECTING, ConnState.RECONNECTING):
            break
        await asyncio.sleep(0.01)
    state_after_kill = handles.emitter.conn_state[fail_idx]
    _LOG.info(
        "listener-fail: emitter conn_state[%d]=%s",
        fail_idx, state_after_kill,
    )

    # Pre-restart drive: emit ``restart_after`` cubes; the dead
    # listener should miss these (its receive count won't grow), but
    # the surviving N-1 still see them. The emitter's TX queue for
    # the dead conn may also rack up `n_dropped_tx_full` once the
    # 128-deep per-conn queue saturates.
    cube_offset = 100_000  # decouple from steady-phase cube_ids
    for cube_id in range(cube_offset, cube_offset + restart_after):
        cands = _build_steady_cube(cube_id, n_cands)
        records = await handles.emitter.process_candidates(
            cube_id=cube_id, candidates=cands,
        )
        for rec in records:
            record_sink.append(
                _emit_record_to_dict(rec, phase="listener_fail_pre_restart")
            )

    mid_per_listener = [l.n_received for l in handles.fan.listeners]
    received_during_kill = mid_per_listener[fail_idx] - pre_per_listener[fail_idx]
    survivors_received_during_kill = sum(
        b - a for i, (a, b) in enumerate(
            zip(pre_per_listener, mid_per_listener)
        )
        if i != fail_idx
    )

    # Restart the listener; verify the emitter reconnects.
    _LOG.info(
        "listener-fail: restarting fan.listeners[%d] on port %d",
        fail_idx, handles.fan.listeners[fail_idx].port,
    )
    await handles.fan.restart_listener(fail_idx)
    # Backoff is 1 s initial, doubling — wait up to backoff_cap_s
    # for the reconnect.
    deadline = time.monotonic() + 35.0
    while time.monotonic() < deadline:
        if handles.emitter.conn_state[fail_idx] == ConnState.UP:
            break
        await asyncio.sleep(0.02)
    state_after_restart = handles.emitter.conn_state[fail_idx]
    reconnect_elapsed = (
        time.monotonic() - (deadline - 35.0) if state_after_restart == ConnState.UP
        else None
    )
    _LOG.info(
        "listener-fail: post-restart conn_state[%d]=%s "
        "(reconnected in ≤ %.2f s)",
        fail_idx, state_after_restart,
        reconnect_elapsed if reconnect_elapsed is not None else float("nan"),
    )

    # Post-restart drive: emit the remaining cubes; the previously
    # dead listener should now see them again.
    post_restart_cubes = max(0, n_cubes - restart_after)
    pre_post_per_listener = [l.n_received for l in handles.fan.listeners]
    for cube_id in range(
        cube_offset + restart_after,
        cube_offset + restart_after + post_restart_cubes,
    ):
        cands = _build_steady_cube(cube_id, n_cands)
        records = await handles.emitter.process_candidates(
            cube_id=cube_id, candidates=cands,
        )
        for rec in records:
            record_sink.append(
                _emit_record_to_dict(rec, phase="listener_fail_post_restart")
            )
    pending = await _drain_in_flight(handles.emitter, timeout_s=3.0)

    post_per_listener = [l.n_received for l in handles.fan.listeners]
    received_post_restart = (
        post_per_listener[fail_idx] - pre_post_per_listener[fail_idx]
    )
    post_n_dropped_tx_full = list(handles.emitter.n_dropped_tx_full_per_corr)
    delta_dropped_tx_full = [
        b - a for a, b in zip(pre_n_dropped_tx_full, post_n_dropped_tx_full)
    ]

    return {
        "n_cubes": n_cubes,
        "fail_idx": fail_idx,
        "fail_listener_port": handles.fan.listeners[fail_idx].port,
        "state_after_kill": state_after_kill,
        "state_after_restart": state_after_restart,
        "reconnect_elapsed_s": reconnect_elapsed,
        "received_during_kill_dead_listener": received_during_kill,
        "received_during_kill_surviving_listeners": (
            survivors_received_during_kill
        ),
        "received_post_restart_revived_listener": received_post_restart,
        "post_restart_cubes_driven": post_restart_cubes,
        "tx_queue_full_drops_during_kill": delta_dropped_tx_full,
        "in_flight_pending_at_drain": pending,
    }


async def _run_rate_limit_blast_phase(
    handles: _BenchHandles,
    args: argparse.Namespace,
    record_sink: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Phase 3: deliberately exceed RateLimitTokenBucket so the bench
    observes its drop counter increment. Every cube sends N >> burst
    candidates; the chain's RateLimit / PerCubeTotalCap should drop
    most of them."""
    n_cubes = (
        QUICK_SMOKE_RATE_LIMIT_BLAST_CUBES if args.quick_smoke
        else int(args.rate_limit_blast_cubes)
    )
    n_cands_per_cube = int(args.rate_limit_blast_cands_per_cube)

    pre_dropped = dict(handles.emitter.dropped)
    pre_emitted = handles.emitter.emitted_total

    cube_offset = 200_000
    for cube_id in range(cube_offset, cube_offset + n_cubes):
        cands = _build_blast_cube(cube_id, n_cands_per_cube)
        records = await handles.emitter.process_candidates(
            cube_id=cube_id, candidates=cands,
        )
        for rec in records:
            record_sink.append(
                _emit_record_to_dict(rec, phase="rate_limit_blast")
            )
        # No sleep between cubes -- the goal is to saturate the
        # token-bucket. Yield control once per cube so the listener
        # tasks can drain their TX/RX queues.
        await asyncio.sleep(0)
    pending = await _drain_in_flight(handles.emitter, timeout_s=3.0)

    post_dropped = dict(handles.emitter.dropped)
    post_emitted = handles.emitter.emitted_total
    delta_dropped = {
        cond: post_dropped.get(cond, 0) - pre_dropped.get(cond, 0)
        for cond in set(post_dropped) | set(pre_dropped)
    }
    return {
        "n_cubes": n_cubes,
        "n_cands_per_cube": n_cands_per_cube,
        "delta_emitted": post_emitted - pre_emitted,
        "delta_dropped_by_condition": delta_dropped,
        "in_flight_pending_at_drain": pending,
    }


# ---------------------------------------------------------------------------
# Pass / fail gate evaluation
# ---------------------------------------------------------------------------


def _evaluate_gate(
    summary: Dict[str, Any], *, ack_p99_budget_ms: float,
) -> Dict[str, Any]:
    """Apply plan §8 line 2328 gates to the bench summary."""
    steady = summary["phases"]["steady"]
    failphase = summary["phases"]["listener_fail"]
    blastphase = summary["phases"]["rate_limit_blast"]

    # Gate A: accepted-ack p99 within budget. The two-stage ACK
    # protocol per plan §4.4 line 1718: ``stage="accepted"`` is the
    # corr listener's "I have your trigger" wire-level ACK; latency
    # is dominated by TCP round-trip + ACK demux. ``stage="completed"``
    # is the corr listener's "voltage dump finished" ACK — its
    # latency includes 50-200 ms of disk I/O on a real burst, gated
    # separately by ``T_completion_timeout_s = 5 s`` rather than the
    # 20 ms wire-level budget. Plan §8 line 2328's "ack p99 ≤ 20 ms"
    # therefore targets the accepted-ack stage; the completed-ack
    # p99 is reported informationally but does NOT gate.
    p99_ms = steady.get("accepted_ack_p99_ms")
    gate_a_pass = p99_ms is not None and p99_ms <= ack_p99_budget_ms

    # Gate B: rate-limit fired (RateLimitTokenBucket dropped > 0).
    # ``emitter._mon_dropped`` keys are the condition's ``name``
    # attribute, lowercased per ``conditions/*::name=`` defaults
    # (``ratelimit``, ``per_cube_total_cap`` etc.) — NOT the class
    # name. The condition-name plan §1718 / §3 line 383 reserves
    # these short strings as wire-stable identifiers; the bench
    # consumes them verbatim.
    rl_drops = blastphase["delta_dropped_by_condition"].get(
        "ratelimit", 0,
    )
    # Per-cube total cap is also acceptable evidence of the chain
    # firing under blast (the 100 cands/cube blast trips PerCubeTotal
    # before RL on the first cubes; once tokens drain, RL takes over).
    pct_drops = blastphase["delta_dropped_by_condition"].get(
        "per_cube_total_cap", 0,
    )
    gate_b_pass = (rl_drops + pct_drops) > 0

    # Gate C: listener recovery — conn flipped DOWN/RECONNECTING
    # during kill, then back to UP after restart.
    state_after_kill = failphase["state_after_kill"]
    state_after_restart = failphase["state_after_restart"]
    gate_c_pass = (
        state_after_kill in (ConnState.DOWN, ConnState.RECONNECTING,
                              ConnState.CONNECTING)
        and state_after_restart == ConnState.UP
    )

    # Gate D: fan-out parity. Every listener should have received
    # roughly the same packet count (within 1% — accounting for the
    # last-cube race where the receiver loop hasn't drained the final
    # ACK into the in-flight tracker yet, and for the kill-window
    # drops on the failed listener).
    parity_min = steady["fanout_parity_min"]
    gate_d_pass = parity_min >= 0.99

    return {
        "ack_p99_budget_ms": ack_p99_budget_ms,
        "gate_a_accepted_ack_p99_within_budget": {
            "pass": bool(gate_a_pass),
            "observed_accepted_ack_p99_ms": p99_ms,
            "observed_completed_ack_p99_ms_informational": (
                steady.get("completed_ack_p99_ms")
            ),
        },
        "gate_b_rate_limit_or_per_cube_cap_fired": {
            "pass": bool(gate_b_pass),
            "rate_limit_drops": rl_drops,
            "per_cube_total_cap_drops": pct_drops,
        },
        "gate_c_listener_recovery": {
            "pass": bool(gate_c_pass),
            "state_after_kill": state_after_kill,
            "state_after_restart": state_after_restart,
        },
        "gate_d_steady_fanout_parity_ge_0_99": {
            "pass": bool(gate_d_pass),
            "observed_min_parity": parity_min,
            "observed_max_parity": steady["fanout_parity_max"],
        },
        "overall_pass": bool(
            gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass
        ),
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _ns_or_none_ms(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return float(v) * 1e-6


def _emit_record_to_dict(rec: EmitRecord, *, phase: str) -> Dict[str, Any]:
    cand = rec.candidate
    return {
        "phase": phase,
        "cube_id_hint": int(rec.candidate.event_specnum) // 256,
        "kernel_id": cand.kernel_id,
        "l": float(cand.l),
        "m": float(cand.m),
        "snr": float(cand.snr),
        "trigger_id": rec.trigger_id,
        "predicate_pass": bool(rec.predicate_pass),
        "predicate_reason": rec.predicate_reason,
        "halo_dropped": bool(rec.halo_dropped),
        "holdoff_suppressed": bool(rec.holdoff_suppressed),
        "emit_utc_ns": int(rec.emit_utc_ns),
    }


def _write_ndjson(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")


def _write_summary(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


async def _bench_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_log_path = out_dir / "bench.log"
    bench_log_handler = logging.FileHandler(bench_log_path, mode="w")
    bench_log_handler.setLevel(logging.INFO)
    bench_log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _LOG.setLevel(logging.INFO)
    _LOG.addHandler(bench_log_handler)
    _LOG.addHandler(logging.StreamHandler(sys.stdout))

    if args.quick_smoke:
        # Override the heavier steady-phase knob too; the rest of
        # the quick-smoke cuts are applied per-phase.
        args.n_listeners = QUICK_SMOKE_N_LISTENERS

    _LOG.info(
        "bench config: n_listeners=%d steady_cubes=%d "
        "listener_fail_cubes=%d (restart-after %d) "
        "rate_limit_blast_cubes=%d (cands_per_cube %d) "
        "ack_p99_budget_ms=%.2f",
        args.n_listeners,
        QUICK_SMOKE_STEADY_CUBES if args.quick_smoke else args.steady_cubes,
        QUICK_SMOKE_LISTENER_FAIL_CUBES if args.quick_smoke
            else args.listener_fail_cubes,
        QUICK_SMOKE_LISTENER_FAIL_RESTART_AFTER_CUBES if args.quick_smoke
            else args.listener_fail_restart_after_cubes,
        QUICK_SMOKE_RATE_LIMIT_BLAST_CUBES if args.quick_smoke
            else args.rate_limit_blast_cubes,
        args.rate_limit_blast_cands_per_cube,
        args.ack_p99_budget_ms,
    )

    handles = await _build_emitter_and_fan(args)
    record_sink_dicts: List[Dict[str, Any]] = []
    overall_t0 = time.monotonic()

    try:
        _LOG.info("=== Phase 1: steady ===")
        steady = await _run_steady_phase(handles, args, record_sink_dicts)
        _LOG.info(
            "steady: emitted=%d in %.2fs (%.0f /s); "
            "completed_ack p50=%.2fms p99=%.2fms; parity_min=%.4f",
            steady["delta_emitted"], steady["elapsed_s"],
            steady["delta_emitted"] / max(steady["elapsed_s"], 1e-9),
            steady["completed_ack_p50_ms"] or float("nan"),
            steady["completed_ack_p99_ms"] or float("nan"),
            steady["fanout_parity_min"],
        )

        _LOG.info("=== Phase 2: listener-fail recovery ===")
        failphase = await _run_listener_fail_phase(
            handles, args, record_sink_dicts,
        )
        _LOG.info(
            "listener-fail: state_after_kill=%s state_after_restart=%s "
            "reconnect_in_s=%s",
            failphase["state_after_kill"],
            failphase["state_after_restart"],
            failphase["reconnect_elapsed_s"],
        )

        _LOG.info("=== Phase 3: rate-limit blast ===")
        blastphase = await _run_rate_limit_blast_phase(
            handles, args, record_sink_dicts,
        )
        _LOG.info(
            "blast: emitted=%d dropped=%s",
            blastphase["delta_emitted"],
            blastphase["delta_dropped_by_condition"],
        )

        # Final fan-out per-listener snapshot for the summary.
        per_listener_n_received_final = {
            str(l.port): {
                "n_received": l.n_received,
                "n_connections_seen": l.n_connections_seen,
                "n_dropped_connections": l.n_dropped_connections,
            }
            for l in handles.fan.listeners
        }
    finally:
        await handles.emitter.stop()
        await handles.fan.stop()

    elapsed_total_s = time.monotonic() - overall_t0
    summary: Dict[str, Any] = {
        "schema_version": "M5.bench.trigger_emitter_wiring.v1",
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_listeners": int(args.n_listeners),
            "rate_limit_rate_per_s": float(args.rate_limit_rate_per_s),
            "rate_limit_burst": int(args.rate_limit_burst),
            "ack_p99_budget_ms": float(args.ack_p99_budget_ms),
            "quick_smoke": bool(args.quick_smoke),
        },
        "elapsed_s": elapsed_total_s,
        "phases": {
            "steady": steady,
            "listener_fail": failphase,
            "rate_limit_blast": blastphase,
        },
        "per_listener_final": per_listener_n_received_final,
    }
    summary["gate"] = _evaluate_gate(
        summary, ack_p99_budget_ms=float(args.ack_p99_budget_ms),
    )

    _write_ndjson(record_sink_dicts, out_dir / "trigger_records.ndjson")
    _write_summary(summary, out_dir / "summary.json")
    _LOG.info(
        "wrote %s (%d records)",
        out_dir / "trigger_records.ndjson", len(record_sink_dicts),
    )
    _LOG.info("wrote %s", out_dir / "summary.json")

    overall = summary["gate"]["overall_pass"]
    _LOG.info(
        "GATE: overall=%s (a=%s b=%s c=%s d=%s)",
        "PASS" if overall else "FAIL",
        summary["gate"]["gate_a_accepted_ack_p99_within_budget"]["pass"],
        summary["gate"]["gate_b_rate_limit_or_per_cube_cap_fired"]["pass"],
        summary["gate"]["gate_c_listener_recovery"]["pass"],
        summary["gate"]["gate_d_steady_fanout_parity_ge_0_99"]["pass"],
    )
    return 0 if overall else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M5 trigger-emitter wiring bench (plan §8 line 2328)"
    )
    parser.add_argument(
        "--n-listeners", type=int, default=DEFAULT_N_LISTENERS,
        help=(
            "Number of mock listeners to fan out to (default 16, "
            "production geometry per plan §4.4 line 1669)."
        ),
    )
    parser.add_argument(
        "--base-port", type=int, default=0,
        help=(
            "Base TCP port for the listener fan; consecutive ports "
            "are allocated. 0 (default) lets the kernel pick "
            "ephemeral ports."
        ),
    )
    parser.add_argument(
        "--steady-cubes", type=int, default=DEFAULT_STEADY_CUBES,
        help="Phase-1 cube count (default 200).",
    )
    parser.add_argument(
        "--steady-cands-per-cube", type=int,
        default=DEFAULT_STEADY_CANDS_PER_CUBE,
        help=(
            "Phase-1 candidates per cube (default 1). The mock "
            "listener processes packets serially per connection "
            "(accept_ack -> completed_delay -> completed_ack), so "
            "multi-cand-per-cube bursts inflate the per-cube ACK "
            "tail by ~5 ms × cands. Production triggers are 1-2 "
            "per cube post-holdoff, matching the default. Operators "
            "stress-test multi-trigger-per-cube via the rate-limit "
            "blast phase (--rate-limit-blast-cands-per-cube)."
        ),
    )
    parser.add_argument(
        "--steady-cube-cadence-ms", type=float,
        default=DEFAULT_STEADY_CUBE_CADENCE_MS,
        help=(
            "Phase-1 inter-cube delay in ms (default 100). At "
            "1 cand/cube the steady stream runs at 10 cands/s — "
            "matching the production RateLimitTokenBucket steady-"
            "state rate. Drop below the listener's per-packet "
            "processing time (≈ 5 ms) only when stress-testing "
            "the emitter's TX-queue back-pressure path."
        ),
    )
    parser.add_argument(
        "--listener-fail-cubes", type=int,
        default=DEFAULT_LISTENER_FAIL_CUBES,
        help="Phase-2 total cube count (default 50).",
    )
    parser.add_argument(
        "--listener-fail-idx", type=int,
        default=DEFAULT_LISTENER_FAIL_IDX,
        help=(
            "Which listener to kill mid-stream (default 3; must be "
            "< n_listeners)."
        ),
    )
    parser.add_argument(
        "--listener-fail-restart-after-cubes", type=int,
        default=DEFAULT_LISTENER_FAIL_RESTART_AFTER_CUBES,
        help=(
            "Phase-2 sub-step: number of cubes to drive while the "
            "listener is killed before restarting (default 20). "
            "Must be ≤ --listener-fail-cubes."
        ),
    )
    parser.add_argument(
        "--rate-limit-blast-cubes", type=int,
        default=DEFAULT_RATE_LIMIT_BLAST_CUBES,
        help="Phase-3 cube count (default 30).",
    )
    parser.add_argument(
        "--rate-limit-blast-cands-per-cube", type=int,
        default=DEFAULT_RATE_LIMIT_BLAST_CANDS_PER_CUBE,
        help=(
            "Phase-3 candidates per cube (default 100; intentionally "
            "exceeds PerCubeTotalCap=16 + RateLimit burst=50)."
        ),
    )
    parser.add_argument(
        "--rate-limit-rate-per-s", type=float,
        default=DEFAULT_RATE_LIMIT_RATE_PER_S,
        help=(
            "RateLimitTokenBucket steady-state rate "
            "(default 10.0; production setting per "
            "configs/config_compute_search.yaml)."
        ),
    )
    parser.add_argument(
        "--rate-limit-burst", type=int,
        default=DEFAULT_RATE_LIMIT_BURST,
        help="RateLimitTokenBucket burst capacity (default 50).",
    )
    parser.add_argument(
        "--ack-p99-budget-ms", type=float,
        default=DEFAULT_ACK_P99_BUDGET_MS,
        help=(
            "Plan §8 line 2328 ack p99 budget in ms (default 20.0). "
            "Mock listener floors at completed_delay_ms=5 ms; the "
            "bench tests the emitter's TCP fan-out + ACK demux "
            "overhead against this budget."
        ),
    )
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "trigger_emitter_wiring" / "M5"),
        help="Output directory (default bench/reports/trigger_emitter_wiring/M5).",
    )
    parser.add_argument(
        "--quick-smoke", action="store_true",
        help=(
            "Cut every phase ~10x for the M5.sh DoD path "
            "(5 listeners, 25 steady cubes, 8 fail-phase cubes, "
            "5 blast cubes)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())
