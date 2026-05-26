"""Async TCP receiver for C1 → C2 batches.

The C2 service spins one :class:`C1BatchReceiver` instance bound to
``coinc.bind:port``. Each accepted TCP connection runs a per-
connection reader task that drains the socket line-by-line; a batch
is committed when ``# END`` is seen and forwarded to a supplied
``on_batch`` async callback.

A torn batch (e.g. peer closed mid-batch) increments the
``c2_torn_batch`` counter. A header with the wrong schema_version
increments ``c2_bad_schema`` and drops the batch (see
``docs/c1c2/C1C2_WIRE_SCHEMA.md`` §1.2 and §0).

This module deliberately has no dependency on the rest of the
:mod:`dsart.coinc` package beyond ``wire``; the service module wires
:class:`C1BatchReceiver` to the window + components + criteria stack.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from . import wire

__all__ = [
    "C1BatchReceiver",
    "ReceiverCounters",
    "OnBatch",
]


_LOG = logging.getLogger("dsart.coinc.receiver")


OnBatch = Callable[[wire.C1Batch, str], Awaitable[None]]
"""Async callback signature: ``async fn(batch, peer_repr) -> None``.

``peer_repr`` is a human-friendly remote peer string like
``"10.41.0.205:54321"`` — used only for logging.
"""


@dataclass
class ReceiverCounters:
    accepted: int = 0
    batches_ok: int = 0
    bad_schema: int = 0
    torn_batch: int = 0
    bad_batch: int = 0
    connections_open: int = 0
    bytes_read: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "batches_ok": self.batches_ok,
            "bad_schema": self.bad_schema,
            "torn_batch": self.torn_batch,
            "bad_batch": self.bad_batch,
            "connections_open": self.connections_open,
            "bytes_read": self.bytes_read,
        }


class C1BatchReceiver:
    """Async TCP server that decodes C1 batches and forwards them."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        on_batch: OnBatch,
        idle_timeout_s: float = 60.0,
    ) -> None:
        self._host = host
        self._port = port
        self._on_batch = on_batch
        self._idle_timeout_s = idle_timeout_s
        self._server: Optional[asyncio.base_events.Server] = None
        self._counters = ReceiverCounters()
        self._client_tasks: set[asyncio.Task] = set()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def counters(self) -> ReceiverCounters:
        return self._counters

    @property
    def is_serving(self) -> bool:
        return self._server is not None and self._server.is_serving()

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("receiver already started")
        self._server = await asyncio.start_server(
            self._handle_client, host=self._host, port=self._port,
            reuse_address=True,
        )
        addrs = ", ".join(
            str(sock.getsockname()) for sock in self._server.sockets
        )
        _LOG.info("C1BatchReceiver listening on %s", addrs)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        # Allow live client tasks to finish naturally; we don't cancel
        # them because they may still be flushing a final batch. The
        # service's overall stop() will reap them on shutdown.
        for t in list(self._client_tasks):
            t.cancel()
        if self._client_tasks:
            await asyncio.gather(*self._client_tasks, return_exceptions=True)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    # ----- per-connection handler --------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        peer_repr = f"{peer[0]}:{peer[1]}" if peer else "?"
        _LOG.info("client connected: %s", peer_repr)
        self._counters.accepted += 1
        self._counters.connections_open += 1
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            await self._read_batches(reader, peer_repr)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _LOG.exception("client %s reader crashed", peer_repr)
        finally:
            self._counters.connections_open -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass
            if task is not None:
                self._client_tasks.discard(task)
            _LOG.info("client disconnected: %s", peer_repr)

    async def _read_batches(
        self, reader: asyncio.StreamReader, peer_repr: str,
    ) -> None:
        """Drain the connection, line-by-line, committing on '# END'."""
        pending: List[str] = []
        in_batch = False
        while True:
            try:
                raw = await asyncio.wait_for(
                    reader.readline(), timeout=self._idle_timeout_s,
                )
            except asyncio.TimeoutError:
                _LOG.warning(
                    "client %s idle for %.1fs (no batches); closing",
                    peer_repr, self._idle_timeout_s,
                )
                return
            if not raw:
                # EOF
                if in_batch:
                    self._counters.torn_batch += 1
                    _LOG.warning(
                        "client %s closed mid-batch (%d lines pending)",
                        peer_repr, len(pending),
                    )
                return
            self._counters.bytes_read += len(raw)
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            stripped = line.lstrip()
            # Skip blank lines (sometimes appear at the seam between
            # back-to-back batches if the sender flushes oddly).
            if not stripped:
                continue
            # New batch starts on a "# C1 " header. If we see one mid-
            # batch, treat the previous as torn.
            if stripped.startswith("# C1 "):
                if in_batch:
                    self._counters.torn_batch += 1
                    _LOG.warning(
                        "client %s started new batch mid-batch "
                        "(prior dropped)",
                        peer_repr,
                    )
                pending = [line]
                in_batch = True
                continue
            if not in_batch:
                # Garbage outside a batch — ignore.
                continue
            pending.append(line)
            if stripped == "# END":
                await self._commit_batch(pending, peer_repr)
                pending = []
                in_batch = False

    async def _commit_batch(
        self, lines: List[str], peer_repr: str,
    ) -> None:
        try:
            batch = wire.parse_c1_batch(lines)
        except wire.BadBatch as exc:
            msg = str(exc)
            if "schema_version" in msg:
                self._counters.bad_schema += 1
                _LOG.warning("client %s bad schema: %s", peer_repr, exc)
            elif "truncated" in msg:
                self._counters.torn_batch += 1
                _LOG.warning("client %s torn batch: %s", peer_repr, exc)
            else:
                self._counters.bad_batch += 1
                _LOG.warning("client %s bad batch: %s", peer_repr, exc)
            return
        self._counters.batches_ok += 1
        try:
            await self._on_batch(batch, peer_repr)
        except Exception:  # noqa: BLE001
            _LOG.exception(
                "client %s on_batch callback raised (batch dropped)",
                peer_repr,
            )
