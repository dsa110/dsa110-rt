"""Search-side trigger emitter (plan §4.4 lines 1669-1718).

Per-search-compute-process module that:

  1. Owns 1..N persistent TCP connections to corr listeners (the
     production geometry is N=16, one per chgroup; tests can wire any N).
  2. Runs every Candidate through the trigger-emit predicate chain
     (``predicate.evaluate_chain``) and the holdoff state machine
     (``HoldoffStateMachine.check_and_register``).
  3. Builds a ``TriggerPacket`` for each candidate that survives, fans
     it out to all connections (so every corr listener sees every
     trigger; the corr-side dedup cache catches the cross-search-node
     duplicates per plan §4.5).
  4. Reads back ACK records on each connection (via a per-connection
     reader task), demultiplexes by ``trigger_id``, and updates an
     in-flight tracker.
  5. Reconnects with exponential backoff on disconnect (1, 2, 4, ...
     capped at 30 s).
  6. Emits a structured per-Candidate ``EmitRecord`` to the candidate
     ndjson log via a caller-supplied callback (the search-compute
     service plugs in the on-disk writer; tests collect into a list).

The emitter is **asyncio-native** and runs as the only consumer of the
candidate stream in its parent process. Production
``services/search_compute.py`` (Chunk 6) wires it; the M5 unit-test
path (Chunk 5 ``bench/cube_injection_detector.py``) wires it against
``MockTriggerListener`` on 127.0.0.1.

Design constraints (plan §4.4):

  - Send semantics are fire-and-forget per connection; the response
    stream is read by a separate task.
  - Per-connection bounded TX queue (depth 128) so a slow / dead
    listener cannot back up the cube loop indefinitely.
  - Disconnect → exponential backoff reconnect; failed sends during
    disconnect drop the trigger on that one connection (the other
    N-1 connections still receive it).
  - In-flight tracker timeouts at ``T_completion_timeout_s = 5``;
    timed-out triggers are logged as failed and evicted.
  - ``trigger_id`` format: ``s<sid>-g<g>-<10-digit-counter>``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from ..common.constants import (
    TRIGGER_ACK_REASONS,
    TRIGGER_ACK_STAGES,
)
from ..common.contracts import (
    Candidate,
    CandidateFlags,
    TriggerAck,
    TriggerPacket,
)
from .holdoff import HoldoffStateMachine
from .ndjson_codec import (
    decode_ack,
    encode_packet,
    split_ndjson_buffer,
)
from .predicate import (
    TriggerCondition,
    TriggerContext,
    evaluate_chain,
)

__all__ = [
    "ConnState",
    "ConnectionEndpoint",
    "EmitRecord",
    "InFlightTracker",
    "TriggerEmitter",
    "TriggerEmitterConfig",
    "DEFAULT_BACKOFF_INITIAL_S",
    "DEFAULT_BACKOFF_CAP_S",
    "DEFAULT_TX_QUEUE_DEPTH",
    "DEFAULT_COMPLETION_TIMEOUT_S",
]


_LOG = logging.getLogger(__name__)


DEFAULT_BACKOFF_INITIAL_S: float = 1.0
DEFAULT_BACKOFF_CAP_S: float = 30.0
DEFAULT_TX_QUEUE_DEPTH: int = 128
DEFAULT_COMPLETION_TIMEOUT_S: float = 5.0


# ---------------------------------------------------------------------------
# Connection state machine
# ---------------------------------------------------------------------------


class ConnState:
    """Connection state values exported as a /mon/.../conn_state[<n>]
    string per plan §4.4 line 1669 / §3.7 mon-key registry."""

    DOWN = "down"
    CONNECTING = "connecting"
    UP = "up"
    RECONNECTING = "reconnecting"


@dataclass(frozen=True, slots=True)
class ConnectionEndpoint:
    """``(host, port)`` for one corr-listener connection. Indexed by the
    chgroup id in production (per plan §4.4 line 1669); test fixtures
    can wire any host/port pair."""

    host: str
    port: int


# ---------------------------------------------------------------------------
# Per-Candidate emit record (for the on-disk ndjson log)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmitRecord:
    """One record per Candidate that *enters* the emit pipeline (plan
    §4.4 line 1716). The candidate ndjson log writer / test collector
    consumes this; the wire `TriggerPacket` is a separate object."""

    candidate: Candidate
    trigger_id: Optional[str]
    predicate_pass: bool
    predicate_reason: Optional[str]
    halo_dropped: bool
    holdoff_suppressed: bool
    emit_utc_ns: int


# ---------------------------------------------------------------------------
# In-flight tracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _InFlightEntry:
    trigger_id: str
    emit_utc_ns: int
    accepted_per_endpoint: Dict[int, int] = field(default_factory=dict)
    completed_per_endpoint: Dict[int, int] = field(default_factory=dict)
    rejected_per_endpoint: Dict[int, str] = field(default_factory=dict)
    n_endpoints: int = 0


class InFlightTracker:
    """Records the per-trigger ACK state per endpoint until the trigger
    completes or times out. Plan §4.4 line 1718.

    Public state surfaced for telemetry / tests:

      - ``in_flight`` → number of triggers currently awaiting full
        completion.
      - ``failed_completion_total`` → cumulative count of triggers that
        timed out on the completed-stage ACK.
      - ``accepted_ack_latency_ns_p50/p99`` → reservoir of recent
        accepted-ACK round-trip latencies (last 1024 samples).
      - ``completed_ack_latency_ns_p50/p99`` → same for completed ACK.
    """

    def __init__(
        self,
        *,
        completion_timeout_s: float = DEFAULT_COMPLETION_TIMEOUT_S,
    ) -> None:
        self.completion_timeout_s = float(completion_timeout_s)
        self._entries: Dict[str, _InFlightEntry] = {}
        self.failed_completion_total: int = 0
        self._accepted_lat_ns: List[int] = []
        self._completed_lat_ns: List[int] = []
        self._lat_max_samples: int = 1024

    @property
    def in_flight(self) -> int:
        return len(self._entries)

    def register_emit(
        self,
        trigger_id: str,
        emit_utc_ns: int,
        n_endpoints: int,
    ) -> None:
        self._entries[trigger_id] = _InFlightEntry(
            trigger_id=trigger_id,
            emit_utc_ns=int(emit_utc_ns),
            n_endpoints=int(n_endpoints),
        )

    def on_ack(self, ack: TriggerAck, endpoint_idx: int, recv_utc_ns: int) -> None:
        entry = self._entries.get(ack.trigger_id)
        if entry is None:
            return  # stale ACK (timed out / unknown trigger)
        if ack.stage == "accepted":
            if ack.accepted is False:
                entry.rejected_per_endpoint[endpoint_idx] = ack.reason or "unknown"
            else:
                entry.accepted_per_endpoint[endpoint_idx] = recv_utc_ns
                self._record_lat(self._accepted_lat_ns, recv_utc_ns - entry.emit_utc_ns)
        elif ack.stage == "completed":
            entry.completed_per_endpoint[endpoint_idx] = recv_utc_ns
            self._record_lat(self._completed_lat_ns, recv_utc_ns - entry.emit_utc_ns)
            # If every endpoint has completed (or rejected), evict.
            if (
                len(entry.completed_per_endpoint) + len(entry.rejected_per_endpoint)
                >= entry.n_endpoints
            ):
                self._entries.pop(ack.trigger_id, None)

    def _record_lat(self, sink: List[int], lat_ns: int) -> None:
        if lat_ns < 0:
            return
        sink.append(lat_ns)
        # Bound the window so memory is O(_lat_max_samples).
        if len(sink) > self._lat_max_samples:
            del sink[0 : len(sink) - self._lat_max_samples]

    def evict_timed_out(self, now_utc_ns: int) -> int:
        """Drop in-flight entries older than ``completion_timeout_s``;
        return the count dropped. Increment ``failed_completion_total``
        per evicted entry."""
        cutoff_ns = int(now_utc_ns) - int(self.completion_timeout_s * 1e9)
        dropped: List[str] = []
        for tid, entry in self._entries.items():
            if entry.emit_utc_ns < cutoff_ns:
                dropped.append(tid)
        for tid in dropped:
            del self._entries[tid]
        self.failed_completion_total += len(dropped)
        return len(dropped)

    @staticmethod
    def _percentile(samples: List[int], p: float) -> Optional[float]:
        if not samples:
            return None
        sorted_s = sorted(samples)
        idx = max(0, min(len(sorted_s) - 1, int(round(p * (len(sorted_s) - 1)))))
        return float(sorted_s[idx])

    @property
    def accepted_ack_latency_ns_p50(self) -> Optional[float]:
        return self._percentile(self._accepted_lat_ns, 0.5)

    @property
    def accepted_ack_latency_ns_p99(self) -> Optional[float]:
        return self._percentile(self._accepted_lat_ns, 0.99)

    @property
    def completed_ack_latency_ns_p50(self) -> Optional[float]:
        return self._percentile(self._completed_lat_ns, 0.5)

    @property
    def completed_ack_latency_ns_p99(self) -> Optional[float]:
        return self._percentile(self._completed_lat_ns, 0.99)


# ---------------------------------------------------------------------------
# Per-connection async state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Connection:
    idx: int
    endpoint: ConnectionEndpoint
    state: str = ConnState.DOWN
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    tx_queue: Optional[asyncio.Queue] = None
    sender_task: Optional[asyncio.Task] = None
    receiver_task: Optional[asyncio.Task] = None
    backoff_s: float = DEFAULT_BACKOFF_INITIAL_S
    n_dropped_tx_full: int = 0
    n_sent: int = 0


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TriggerEmitterConfig:
    """Static config shared across the emitter's lifetime."""

    search_node_id: int
    gpu_half: int
    endpoints: List[ConnectionEndpoint]
    conditions: List[TriggerCondition] = field(default_factory=list)
    holdoff: Optional[HoldoffStateMachine] = None
    tx_queue_depth: int = DEFAULT_TX_QUEUE_DEPTH
    backoff_initial_s: float = DEFAULT_BACKOFF_INITIAL_S
    backoff_cap_s: float = DEFAULT_BACKOFF_CAP_S
    completion_timeout_s: float = DEFAULT_COMPLETION_TIMEOUT_S
    default_actions: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "voltage_dump": True,
            "filterbank": True,
            "n_beams": 1,
        }
    )
    default_priority: str = "normal"
    n_pre_blocks: Optional[int] = None
    n_post_blocks: Optional[int] = None
    src_name_template: str = "auto_{utc}_b{counter}"

    def __post_init__(self) -> None:
        if self.search_node_id < 0:
            raise ValueError(f"search_node_id={self.search_node_id}")
        if self.gpu_half not in (0, 1):
            raise ValueError(f"gpu_half={self.gpu_half}, expected 0 or 1")
        if not self.endpoints:
            raise ValueError("endpoints must be non-empty")


class TriggerEmitter:
    """Search-side trigger emitter — async TCP fan-out + ACK demux.

    Lifecycle:

        emitter = TriggerEmitter(config, on_emit_record=callback)
        await emitter.start()
        ...
        await emitter.process_candidates(cube_id, list_of_candidates)
        ...
        await emitter.stop()

    or use as an async context manager:

        async with TriggerEmitter(config, on_emit_record=cb) as e:
            await e.process_candidates(...)
    """

    def __init__(
        self,
        config: TriggerEmitterConfig,
        *,
        on_emit_record: Optional[Callable[[EmitRecord], Awaitable[None] | None]] = None,
    ) -> None:
        self.config = config
        self._holdoff = config.holdoff or HoldoffStateMachine()
        self._on_emit_record = on_emit_record
        self._connections: List[_Connection] = [
            _Connection(idx=i, endpoint=ep)
            for i, ep in enumerate(config.endpoints)
        ]
        self._counter: int = 0
        self._in_flight = InFlightTracker(
            completion_timeout_s=config.completion_timeout_s,
        )
        self._mon_emitted_total: int = 0
        self._mon_dropped: Dict[str, int] = {}
        self._mon_holdoff_suppressed: int = 0
        self._mon_halo_dropped: int = 0
        self._stop = False
        self._cube_emit_state: Dict[int, _CubeEmitState] = {}

    @property
    def in_flight_tracker(self) -> InFlightTracker:
        return self._in_flight

    @property
    def emitted_total(self) -> int:
        return self._mon_emitted_total

    @property
    def dropped(self) -> Mapping[str, int]:
        return dict(self._mon_dropped)

    @property
    def holdoff_suppressed(self) -> int:
        return self._mon_holdoff_suppressed

    @property
    def halo_dropped(self) -> int:
        return self._mon_halo_dropped

    @property
    def conn_state(self) -> List[str]:
        return [c.state for c in self._connections]

    @property
    def n_pending_per_corr(self) -> List[int]:
        return [c.tx_queue.qsize() if c.tx_queue is not None else 0 for c in self._connections]

    @property
    def n_dropped_tx_full_per_corr(self) -> List[int]:
        return [c.n_dropped_tx_full for c in self._connections]

    @property
    def n_sent_per_corr(self) -> List[int]:
        return [c.n_sent for c in self._connections]

    async def start(self) -> "TriggerEmitter":
        for conn in self._connections:
            conn.tx_queue = asyncio.Queue(maxsize=self.config.tx_queue_depth)
            conn.state = ConnState.CONNECTING
            conn.sender_task = asyncio.create_task(self._sender_loop(conn))
            conn.receiver_task = asyncio.create_task(self._receiver_loop(conn))
        return self

    async def stop(self) -> None:
        self._stop = True
        # Cancel all per-connection tasks; close writers.
        for conn in self._connections:
            if conn.sender_task is not None:
                conn.sender_task.cancel()
            if conn.receiver_task is not None:
                conn.receiver_task.cancel()
        for conn in self._connections:
            for task in (conn.sender_task, conn.receiver_task):
                if task is None:
                    continue
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if conn.writer is not None:
                with contextlib.suppress(Exception):
                    conn.writer.close()
                    await conn.writer.wait_closed()
            conn.state = ConnState.DOWN
            conn.writer = None
            conn.reader = None
            conn.sender_task = None
            conn.receiver_task = None

    async def __aenter__(self) -> "TriggerEmitter":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ----- candidate processing -----

    async def process_candidates(
        self,
        cube_id: int,
        candidates: List[Candidate],
        *,
        now_utc_ns: Optional[int] = None,
    ) -> List[EmitRecord]:
        """Run the candidate stream from one cube through the emitter.

        Returns the list of ``EmitRecord``s produced (one per Candidate
        that entered the pipeline). Exposed for the test path; the
        production path discards the return value (the on_emit_record
        callback is the canonical sink).
        """
        if now_utc_ns is None:
            now_utc_ns = time.time_ns()

        # Per-cube emit state — cap counters reset per cube.
        cube_state = _CubeEmitState()
        self._cube_emit_state[cube_id] = cube_state

        # Best-effort timeout sweep (cheap; bounded by in-flight count).
        self._in_flight.evict_timed_out(now_utc_ns)

        records: List[EmitRecord] = []
        for cand in candidates:
            # Halo / time-edge gate: if the candidate carries the
            # canonical-zone-gate's drop flags from filter_to_canonical,
            # log it but do not emit (plan §4.4 line 1594).
            halo_dropped = bool(
                cand.flags & (
                    int(CandidateFlags.HALO_DROPPED)
                    | int(CandidateFlags.TIME_EDGE_DROPPED)
                )
            )
            if halo_dropped:
                self._mon_halo_dropped += 1
                rec = EmitRecord(
                    candidate=cand,
                    trigger_id=None,
                    predicate_pass=False,
                    predicate_reason="halo_or_time_edge_dropped",
                    halo_dropped=True,
                    holdoff_suppressed=False,
                    emit_utc_ns=now_utc_ns,
                )
                records.append(rec)
                await self._invoke_on_emit_record(rec)
                continue

            # Holdoff (single check + register; suppresses dup peaks
            # within trigger_holdoff_ms of an earlier emit at the same
            # (l, m, kernel) cell).
            suppressed = self._holdoff.check_and_register(cand, now_utc_ns)
            if suppressed:
                self._mon_holdoff_suppressed += 1
                rec = EmitRecord(
                    candidate=cand,
                    trigger_id=None,
                    predicate_pass=False,
                    predicate_reason="holdoff_suppressed",
                    halo_dropped=False,
                    holdoff_suppressed=True,
                    emit_utc_ns=now_utc_ns,
                )
                records.append(rec)
                await self._invoke_on_emit_record(rec)
                continue

            # Predicate chain.
            ctx = TriggerContext(
                cube_id=cube_id,
                cube_emitted_in_kernel=cube_state.per_kernel,
                cube_emitted_total=cube_state.total,
                now_utc_ns=now_utc_ns,
            )
            emit, cond_name, cond_reason = evaluate_chain(
                self.config.conditions, cand, ctx,
            )
            if not emit:
                self._mon_dropped[cond_name or "unknown"] = (
                    self._mon_dropped.get(cond_name or "unknown", 0) + 1
                )
                rec = EmitRecord(
                    candidate=cand,
                    trigger_id=None,
                    predicate_pass=False,
                    predicate_reason=cond_reason or cond_name,
                    halo_dropped=False,
                    holdoff_suppressed=False,
                    emit_utc_ns=now_utc_ns,
                )
                records.append(rec)
                await self._invoke_on_emit_record(rec)
                continue

            # Survived chain → build trigger packet, fan out, register.
            self._counter += 1
            trigger_id = self._make_trigger_id(self._counter)
            packet = self._build_packet(cand, trigger_id, now_utc_ns)
            await self._fan_out(packet)
            self._in_flight.register_emit(
                trigger_id, now_utc_ns, len(self._connections),
            )
            cube_state.per_kernel[cand.kernel_id] = (
                cube_state.per_kernel.get(cand.kernel_id, 0) + 1
            )
            cube_state.total += 1
            self._mon_emitted_total += 1
            rec = EmitRecord(
                candidate=cand,
                trigger_id=trigger_id,
                predicate_pass=True,
                predicate_reason=None,
                halo_dropped=False,
                holdoff_suppressed=False,
                emit_utc_ns=now_utc_ns,
            )
            records.append(rec)
            await self._invoke_on_emit_record(rec)

        # Done with this cube; bound the cube-state dict.
        if len(self._cube_emit_state) > 64:
            # Keep only the 32 most recent.
            recent_ids = sorted(self._cube_emit_state.keys())[-32:]
            self._cube_emit_state = {
                k: self._cube_emit_state[k] for k in recent_ids
            }
        return records

    # ----- helpers -----

    def _make_trigger_id(self, counter: int) -> str:
        return (
            f"s{self.config.search_node_id}-g{self.config.gpu_half}-"
            f"{counter:010d}"
        )

    def _build_packet(
        self,
        cand: Candidate,
        trigger_id: str,
        now_utc_ns: int,
    ) -> TriggerPacket:
        # event_utc_ns := now − 0  (real specnum→UTC conversion lives
        # in M1 specnum_table; for v1 unit-test path the emitter just
        # stamps now). The Chunk-6 service path passes the real value
        # via a callback (deferred).
        return TriggerPacket(
            trigger_id=trigger_id,
            search_node_id=self.config.search_node_id,
            emit_utc_ns=int(now_utc_ns),
            event_specnum=int(cand.event_specnum),
            event_utc_ns=int(now_utc_ns),
            l=float(cand.l),
            m=float(cand.m),
            dm_fine=float(cand.dm_fine),
            dm_idx=int(cand.dm_idx),
            fine_dm_trial=int(cand.dm_idx),  # v1: fine_dm_trial == dm_idx (O-7)
            width_samples=int(cand.width_samples),
            kernel_id=cand.kernel_id,
            snr=float(cand.snr),
            actions=dict(self.config.default_actions),
            priority=self.config.default_priority,
            src_name=self._make_src_name(now_utc_ns, self._counter),
            n_pre_blocks=self.config.n_pre_blocks,
            n_post_blocks=self.config.n_post_blocks,
        )

    def _make_src_name(self, now_utc_ns: int, counter: int) -> str:
        # auto_<UTC>_b<counter>; UTC encoded as YYYYMMDDHHMMSS.
        utc_struct = time.gmtime(now_utc_ns * 1e-9)
        utc_str = time.strftime("%Y%m%d_%H%M%S", utc_struct)
        return self.config.src_name_template.format(
            utc=utc_str, counter=counter,
        )

    async def _fan_out(self, packet: TriggerPacket) -> None:
        """Push the packet onto every connection's TX queue. If a
        queue is full, increment the per-conn drop counter and skip
        (the trigger is still emitted to the surviving connections)."""
        wire = encode_packet(packet)
        for conn in self._connections:
            if conn.tx_queue is None:
                continue
            try:
                conn.tx_queue.put_nowait(wire)
            except asyncio.QueueFull:
                conn.n_dropped_tx_full += 1

    async def _invoke_on_emit_record(self, rec: EmitRecord) -> None:
        if self._on_emit_record is None:
            return
        out = self._on_emit_record(rec)
        if asyncio.iscoroutine(out):
            await out

    # ----- per-connection tasks -----

    async def _sender_loop(self, conn: _Connection) -> None:
        """Connect (with backoff), then loop: drain TX queue → write.

        The sender handles disconnects from two directions:
          - direct write/drain failure (peer reset, broken pipe);
          - receiver task observed EOF and called ``_mark_disconnected``
            while we were blocked on ``tx_queue.get()`` (writer goes
            None under our feet, and the receiver pushed a `None`
            sentinel into the queue to wake us so we can reconnect
            even when no new packet is being emitted).
        In either case we rebuild the connection on the next iteration
        with exponential backoff and drop the in-flight packet (per
        plan §4.4: fail-during-disconnect → drop on this conn only)."""
        try:
            while not self._stop:
                if conn.writer is None:
                    await self._connect_with_backoff(conn)
                    if conn.writer is None:
                        # Backoff was cancelled; exit.
                        return
                if conn.tx_queue is None:
                    return
                wire = await conn.tx_queue.get()
                # Re-check after the await — receiver may have flipped
                # us to disconnected while we were blocked, and dropped
                # a None sentinel in the queue to kick us awake.
                if wire is None or conn.writer is None:
                    continue
                try:
                    conn.writer.write(wire)
                    await conn.writer.drain()
                    conn.n_sent += 1
                except (ConnectionError, OSError, BrokenPipeError) as e:
                    _LOG.warning(
                        "sender_loop: write to %s:%s failed: %s",
                        conn.endpoint.host, conn.endpoint.port, e,
                    )
                    await self._mark_disconnected(conn)
        except asyncio.CancelledError:
            return

    async def _receiver_loop(self, conn: _Connection) -> None:
        """Read ACKs from the connection; demultiplex by trigger_id."""
        buf = b""
        try:
            while not self._stop:
                if conn.reader is None:
                    # Wait for the sender to establish the connection.
                    await asyncio.sleep(0.05)
                    continue
                try:
                    chunk = await conn.reader.read(4096)
                except (ConnectionError, OSError):
                    chunk = b""
                if not chunk:
                    # Peer closed or read failed; mark disconnected and let
                    # the sender task reconnect.
                    await self._mark_disconnected(conn)
                    buf = b""
                    continue
                buf += chunk
                lines, buf = split_ndjson_buffer(buf)
                for line in lines:
                    if not line:
                        continue
                    try:
                        ack = decode_ack(line)
                    except ValueError:
                        _LOG.warning(
                            "receiver_loop: malformed ACK from %s:%s",
                            conn.endpoint.host, conn.endpoint.port,
                        )
                        continue
                    self._in_flight.on_ack(ack, conn.idx, time.time_ns())
        except asyncio.CancelledError:
            return

    async def _connect_with_backoff(self, conn: _Connection) -> None:
        delay = conn.backoff_s
        while not self._stop:
            conn.state = ConnState.CONNECTING
            try:
                reader, writer = await asyncio.open_connection(
                    host=conn.endpoint.host, port=conn.endpoint.port,
                )
            except (ConnectionError, OSError):
                conn.state = ConnState.RECONNECTING
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
                delay = min(self.config.backoff_cap_s, delay * 2.0)
                conn.backoff_s = delay
                continue
            conn.reader = reader
            conn.writer = writer
            conn.state = ConnState.UP
            conn.backoff_s = self.config.backoff_initial_s
            return

    async def _mark_disconnected(self, conn: _Connection) -> None:
        conn.state = ConnState.RECONNECTING
        if conn.writer is not None:
            with contextlib.suppress(Exception):
                conn.writer.close()
                await conn.writer.wait_closed()
        conn.writer = None
        conn.reader = None
        # Wake the sender so it can re-enter _connect_with_backoff
        # even when no fresh packet is being emitted on this conn.
        # Best-effort: if the queue is full the sender will eventually
        # drain a real packet, see writer is None, and reconnect.
        if conn.tx_queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                conn.tx_queue.put_nowait(None)


@dataclass(slots=True)
class _CubeEmitState:
    per_kernel: Dict[str, int] = field(default_factory=dict)
    total: int = 0


# Sanity: confirm the M1 contract values are loaded (catches an import
# regression where the constants module changes shape and the emitter
# silently uses stale literals).
assert "accepted" in TRIGGER_ACK_STAGES
assert "ratelimit" in TRIGGER_ACK_REASONS
