"""Mock corr-side trigger listener for M5 unit + bench tests
(plan §8 line 2328 / §4.4 line 1718 ``bench/trigger_emitter_wiring.py``;
M5 PARALLEL_AGENTS.md Class A `mock_listener.py`).

A minimal asyncio TCP server that accepts NDJSON ``TriggerPacket``
records on a configurable port, sends back two-stage NDJSON
``TriggerAck`` records (``stage="accepted"`` then ``stage="completed"``),
and counts what it received per connection. The real corr listener
(``corr_fast_compute``'s embedded TCP listener) is M6 — this mock is
the v1 unit-test peer that lets M5 test the emitter end-to-end without
the corr-side voltage / dump infrastructure.

Designed for two test modes:

  - **Single-port** (``MockTriggerListener``): one server, accepts all
    connections; useful for the basic emitter-wiring smoke test.
  - **Multi-port fan** (``MockTriggerListenerFan``): N independent
    listeners on consecutive ports, mirroring the production "16
    persistent TCP connections to 16 corr listeners" geometry. Each
    fan-listener tracks its own counters / behaviour; tests can
    selectively kill a single listener mid-run to verify the
    emitter's reconnect path.

Knobs (per-listener, can be set from the test):

  - ``accept_rate``: probability of returning ``accepted=True`` per
    incoming trigger. Default 1.0 (always accept).
  - ``accept_reason_when_rejected``: ``TRIGGER_ACK_REASONS``-valid
    string. Default ``"ratelimit"``.
  - ``accept_delay_ms``: delay between ``recv()`` and ``accepted`` ACK
    send. Default 0.
  - ``completed_delay_ms``: delay between ``accepted`` and ``completed``
    ACK send. Default 5 (so tests can distinguish the two stages).
  - ``send_completed``: whether to send the completed ACK at all.
    Default True.
  - ``shutdown_after_n``: if not None, shut the server down after
    receiving N triggers. Default None (run until cancelled).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from typing import Any, Dict, List, Optional

from ..common.contracts import TriggerAck
from .ndjson_codec import (
    decode_packet,
    encode_ack,
    split_ndjson_buffer,
)

__all__ = [
    "MockListenerConfig",
    "MockTriggerListener",
    "MockTriggerListenerFan",
    "ReceivedRecord",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ReceivedRecord:
    """One trigger received by the mock listener (frozen so tests can
    safely snapshot the listener's `received` list)."""

    trigger_id: str
    snr: float
    kernel_id: str
    receive_utc_ns: int
    accepted: bool
    reason: Optional[str]


@dataclasses.dataclass(slots=True)
class MockListenerConfig:
    """Tunable per-listener behaviour."""

    accept_rate: float = 1.0
    accept_reason_when_rejected: str = "ratelimit"
    accept_delay_ms: float = 0.0
    completed_delay_ms: float = 5.0
    send_completed: bool = True
    shutdown_after_n: Optional[int] = None
    queue_depth_report: int = 0


def _now_utc_ns() -> int:
    return time.time_ns()


class MockTriggerListener:
    """Single-port asyncio TCP server.

    Usage:
        listener = MockTriggerListener(host="127.0.0.1", port=11227)
        await listener.start()
        ...
        await listener.stop()

    The instance is also an async context manager.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        config: Optional[MockListenerConfig] = None,
    ) -> None:
        self.host = host
        self._requested_port = int(port)
        self.port: int = -1  # populated after start()
        self.config = config or MockListenerConfig()
        self._server: Optional[asyncio.base_events.Server] = None
        self.received: List[ReceivedRecord] = []
        self._connections_seen: int = 0
        self._dropped_connections: int = 0
        self._stop_event: Optional[asyncio.Event] = None
        self._reject_counter: int = 0
        # Active per-client writers so stop() can drop existing
        # connections and force the emitter to observe the disconnect.
        # `asyncio.Server.close()` only stops accepting new connections
        # — it does NOT close already-accepted ones, which would leave
        # the emitter happily writing to a "stopped" listener.
        self._active_writers: "set[asyncio.StreamWriter]" = set()

    @property
    def n_received(self) -> int:
        return len(self.received)

    @property
    def n_connections_seen(self) -> int:
        return self._connections_seen

    @property
    def n_dropped_connections(self) -> int:
        """Connections forcibly closed by the listener (via `kill_one_connection`).
        Differs from n_connections_seen − active because the latter would
        also count clean client-side disconnects."""
        return self._dropped_connections

    async def start(self) -> "MockTriggerListener":
        if self._server is not None:
            raise RuntimeError("listener already started")
        self._stop_event = asyncio.Event()
        # On a restart we *must* re-bind to the same port the emitter
        # already knows about (self.port), otherwise the emitter's
        # exponential backoff will never find us again. On the very
        # first start we honour the caller's requested port (which
        # may be 0 → ephemeral).
        bind_port = self.port if self.port > 0 else self._requested_port
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=bind_port,
        )
        sock = self._server.sockets[0] if self._server.sockets else None
        self.port = sock.getsockname()[1] if sock is not None else bind_port
        return self

    async def stop(self) -> None:
        if self._server is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        self._server.close()
        # Force-close every still-open client connection so the emitter
        # observes EOF and flips to RECONNECTING. (`server.close()`
        # alone leaves established sockets running until the *client*
        # disconnects.)
        active = list(self._active_writers)
        self._active_writers.clear()
        for w in active:
            with contextlib.suppress(Exception):
                w.close()
        for w in active:
            with contextlib.suppress(Exception):
                await w.wait_closed()
            self._dropped_connections += 1
        with contextlib.suppress(Exception):
            await self._server.wait_closed()
        self._server = None

    async def __aenter__(self) -> "MockTriggerListener":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._connections_seen += 1
        self._active_writers.add(writer)
        buf = b""
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    return  # client EOF
                buf += chunk
                lines, buf = split_ndjson_buffer(buf)
                for line in lines:
                    if not line:
                        continue
                    await self._process_line(line, writer)
                    if (
                        self.config.shutdown_after_n is not None
                        and self.n_received >= self.config.shutdown_after_n
                    ):
                        return
        except (asyncio.CancelledError, ConnectionResetError):
            return
        finally:
            self._active_writers.discard(writer)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _process_line(
        self,
        line: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            packet = decode_packet(line)
        except ValueError:
            # Mirror the §4.5 line 1718 'bad_schema' reason (operator
            # gets a counter for malformed packets).
            ack = TriggerAck(
                trigger_id="<unknown>",
                stage="accepted",
                ack_utc_ns=_now_utc_ns(),
                accepted=False,
                reason="bad_schema",
                queue_depth=self.config.queue_depth_report,
            )
            writer.write(encode_ack(ack))
            with contextlib.suppress(Exception):
                await writer.drain()
            return

        accept = self._decide_accept()
        reason = None if accept else self.config.accept_reason_when_rejected

        if self.config.accept_delay_ms > 0:
            await asyncio.sleep(self.config.accept_delay_ms * 1e-3)

        accepted_ack = TriggerAck(
            trigger_id=packet.trigger_id,
            stage="accepted",
            ack_utc_ns=_now_utc_ns(),
            accepted=accept,
            reason=reason,
            queue_depth=self.config.queue_depth_report,
            dup_of=None,
        )
        self.received.append(
            ReceivedRecord(
                trigger_id=packet.trigger_id,
                snr=packet.snr,
                kernel_id=packet.kernel_id,
                receive_utc_ns=_now_utc_ns(),
                accepted=accept,
                reason=reason,
            )
        )
        writer.write(encode_ack(accepted_ack))
        with contextlib.suppress(Exception):
            await writer.drain()

        if accept and self.config.send_completed:
            if self.config.completed_delay_ms > 0:
                await asyncio.sleep(self.config.completed_delay_ms * 1e-3)
            completed_ack = TriggerAck(
                trigger_id=packet.trigger_id,
                stage="completed",
                ack_utc_ns=_now_utc_ns(),
                voltage_dump_path=f"/tmp/mock/fl_{packet.trigger_id}.out",
                filterbank_paths=(),
                dump_completion_utc_ns=_now_utc_ns(),
                dump_duration_ms=int(self.config.completed_delay_ms),
            )
            writer.write(encode_ack(completed_ack))
            with contextlib.suppress(Exception):
                await writer.drain()

    def _decide_accept(self) -> bool:
        if self.config.accept_rate >= 1.0:
            return True
        if self.config.accept_rate <= 0.0:
            return False
        # Deterministic round-robin: every Nth packet is rejected
        # (avoids RNG-flakiness in tests).
        self._reject_counter += 1
        rejection_period = max(1, int(round(1.0 / (1.0 - self.config.accept_rate))))
        return (self._reject_counter % rejection_period) != 0


class MockTriggerListenerFan:
    """N independent MockTriggerListeners on consecutive ports starting
    at ``base_port`` (or random ephemeral when ``base_port=0``).

    Mirrors the production "16 corr listeners" geometry. Tests can
    selectively kill a single listener mid-run via
    ``await listener_fan.stop_listener(idx)`` and observe the emitter's
    reconnect behaviour on the surviving 15.
    """

    def __init__(
        self,
        *,
        n: int = 16,
        host: str = "127.0.0.1",
        base_port: int = 0,
        config: Optional[MockListenerConfig] = None,
    ) -> None:
        self.n = int(n)
        self.host = host
        self.base_port = int(base_port)
        self.config = config or MockListenerConfig()
        self.listeners: List[MockTriggerListener] = [
            MockTriggerListener(
                host=host,
                port=(base_port + i) if base_port > 0 else 0,
                config=dataclasses.replace(self.config),
            )
            for i in range(self.n)
        ]

    @property
    def ports(self) -> List[int]:
        return [l.port for l in self.listeners]

    @property
    def addrs(self) -> List[tuple[str, int]]:
        return [(l.host, l.port) for l in self.listeners]

    @property
    def n_received_total(self) -> int:
        return sum(l.n_received for l in self.listeners)

    async def start(self) -> "MockTriggerListenerFan":
        await asyncio.gather(*(l.start() for l in self.listeners))
        return self

    async def stop(self) -> None:
        await asyncio.gather(*(l.stop() for l in self.listeners))

    async def stop_listener(self, idx: int) -> None:
        await self.listeners[idx].stop()

    async def restart_listener(self, idx: int) -> None:
        await self.listeners[idx].start()

    async def __aenter__(self) -> "MockTriggerListenerFan":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def by_port(self, port: int) -> Optional[MockTriggerListener]:
        for l in self.listeners:
            if l.port == port:
                return l
        return None

    def snapshot_per_listener(self) -> Dict[int, Dict[str, Any]]:
        """Per-port summary for test assertions."""
        return {
            l.port: {
                "n_received": l.n_received,
                "n_connections_seen": l.n_connections_seen,
                "n_dropped_connections": l.n_dropped_connections,
            }
            for l in self.listeners
        }
