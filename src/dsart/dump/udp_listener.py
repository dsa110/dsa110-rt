"""Asyncio UDP listener for external "dump next cube" trigger requests
(M6 chunk 4).

The listener binds a single UDP socket on a configurable host:port (default
``127.0.0.1:11227``; D9 lock in ``M6_PLAN_FIXES.md``). Any datagram that
arrives flips a one-shot ``dump_next_cube`` flag that the per-(search_node,
gpu_half) cube driver atomically check-and-clears via
``consume_dump_next_cube_flag()``. Queued requests do NOT accumulate: one
or many datagrams between cubes still produce exactly one True from the
next consume call (D9). Datagram payload bytes are NOT parsed (D12: the
UDP trigger has no specnum / coordinates; it just dumps the next cube).

The trigger source is on-host (operator scripts, future T2/T3
re-clusterers), so the default bind is loopback only. Bind to
``0.0.0.0`` via ``UdpTriggerListenerConfig.host`` if a remote source ever
appears (not used in M6).

Concurrency model:

  * ``datagram_received`` runs on the event loop thread (asyncio's UDP
    transport), where it grabs a ``threading.Lock``, sets the flag, and
    bumps the "received" counter.
  * ``consume_dump_next_cube_flag`` is synchronous and may be called from
    any thread (in production: the per-cube driver, which in chunk 5 will
    drive cubes from a worker thread). It grabs the same lock to make
    the (read flag, clear flag, bump consumed counter) sequence atomic
    relative to the asyncio receive callback.

The lock is held for at most a few CPU instructions per datagram /
consume; with peak production datagram rates of O(10/s) the contention
is negligible.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger("dsart.dump.udp_listener")


__all__ = [
    "UdpTriggerListenerConfig",
    "UdpTriggerListener",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UdpTriggerListenerConfig:
    """Bind configuration for ``UdpTriggerListener`` (M6 D9).

    Args:
        host: Interface to bind. Default ``127.0.0.1`` per D9 (the only
            production trigger sources are on-host operator scripts and
            future T2/T3 re-clusterers).
        port: UDP port to bind. Default ``11227`` per D9 (the legacy
            ``dsaX_filTrigger_twoInput`` that previously claimed this
            port on search nodes is removed in the M6 path). Pass
            ``port=0`` in tests to bind an ephemeral free port; read
            the actual port back via ``UdpTriggerListener.bound_port``.
    """

    host: str = "127.0.0.1"
    port: int = 11227


# ---------------------------------------------------------------------------
# DatagramProtocol
# ---------------------------------------------------------------------------


class _TriggerDatagramProtocol(asyncio.DatagramProtocol):
    """Internal asyncio protocol that delegates to the parent listener."""

    def __init__(self, listener: "UdpTriggerListener") -> None:
        self._listener = listener

    def datagram_received(self, data: bytes, addr) -> None:  # noqa: ANN001
        self._listener._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        # Best-effort log; UDP error_received fires for ICMP unreachables
        # and similar transient conditions which we explicitly want to
        # ignore on the receive side (we never send).
        _LOG.warning("UDP error_received on listener: %r", exc)

    def connection_lost(self, exc: Optional[BaseException]) -> None:
        # Datagram endpoints don't strictly have "connections", but
        # asyncio invokes this when the transport closes.
        if exc is not None:
            _LOG.warning("UDP transport connection_lost with exc: %r", exc)


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


class UdpTriggerListener:
    """Asyncio UDP listener that arms a one-shot "dump next cube" flag
    (M6 chunk 4 / D9 / D12).

    The flag is intended to be polled (and cleared) by the per-cube driver
    via ``consume_dump_next_cube_flag()``. Multiple datagrams between
    consume calls still produce exactly one True from the next consume.

    The listener does NOT parse datagram payload bytes. Per D12, the UDP
    trigger carries no specnum / coordinates: it only requests "dump the
    next cube".

    Args:
        config: ``UdpTriggerListenerConfig`` with bind host + port.

    Lifecycle:
        ``await start()``  binds the socket and arms the receive task.
        ``await stop()``   closes the socket and tears down the task.
        Calling ``start()`` after ``stop()`` is supported for tests; in
        production each listener is started exactly once per process.

    Thread safety:
        ``datagram_received`` runs on the asyncio event loop thread.
        ``consume_dump_next_cube_flag`` is synchronous and safe to call
        from any thread (including the per-cube driver thread in chunk
        5). A single ``threading.Lock`` makes the (set flag) and (read
        + clear flag) sequences atomic relative to each other.
    """

    def __init__(self, config: UdpTriggerListenerConfig) -> None:
        self._config = config

        self._lock = threading.Lock()
        self._flag: bool = False
        self._n_datagrams: int = 0
        self._n_consumed: int = 0

        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_TriggerDatagramProtocol] = None
        self._bound_port: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind the UDP socket and start the asyncio receive endpoint.

        Raises:
            OSError: if the bind fails (e.g. the port is already in use).
                Per the M6 chunk-4 spec the listener does NOT silently
                fall back to a different port; the caller must catch and
                handle.
            RuntimeError: if the listener is already started.
        """
        if self._transport is not None:
            raise RuntimeError(
                "UdpTriggerListener.start() called while already started"
            )

        loop = asyncio.get_running_loop()
        # NOTE: we use create_datagram_endpoint rather than a raw socket
        # + loop.add_reader because the former gives us a well-defined
        # cross-platform error path (OSError raised inline if bind fails)
        # and a portable transport.close() shutdown path. add_reader would
        # require us to manually wrap socket.recvfrom() loops and EAGAIN
        # handling, which buys nothing here.
        try:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _TriggerDatagramProtocol(self),
                local_addr=(self._config.host, self._config.port),
                family=socket.AF_INET,
                allow_broadcast=False,
            )
        except OSError as exc:
            _LOG.error(
                "UdpTriggerListener bind failed on %s:%d (%r)",
                self._config.host,
                self._config.port,
                exc,
            )
            raise

        sock = transport.get_extra_info("socket")
        bound_port = sock.getsockname()[1] if sock is not None else self._config.port

        self._transport = transport
        self._protocol = protocol  # keep a strong ref alive for the lifetime
        self._bound_port = int(bound_port)

        _LOG.info(
            "UdpTriggerListener bound to %s:%d (configured port=%d)",
            self._config.host,
            self._bound_port,
            self._config.port,
        )

    async def stop(self) -> None:
        """Close the UDP socket and tear down the asyncio receive task.

        Idempotent: a second call after the first is a no-op.
        """
        if self._transport is None:
            return

        try:
            self._transport.close()
        finally:
            self._transport = None
            self._protocol = None
            self._bound_port = 0

        # transport.close() schedules the close callback; yield once to
        # let the event loop drive any pending close work to completion
        # so that "stop() then immediate send" tests see no further
        # datagrams arrive.
        await asyncio.sleep(0)

        _LOG.info("UdpTriggerListener stopped")

    # ------------------------------------------------------------------
    # Trigger plumbing
    # ------------------------------------------------------------------

    def _on_datagram(self, data: bytes, addr) -> None:  # noqa: ANN001
        """Receive callback (called on the asyncio event loop thread)."""
        with self._lock:
            already_armed = self._flag
            self._flag = True
            self._n_datagrams += 1

        # Logging is outside the lock (formatting + handler I/O are
        # comparatively expensive). We log payload SIZE and peer addr
        # but never the payload bytes themselves: per D12 the payload
        # is opaque to us (the spec defines no schema), and operator
        # scripts may at some point decide to send arbitrary content.
        # Logging bytes blindly risks leaking secrets and produces
        # log lines of unbounded size.
        _LOG.info(
            "udp trigger from %s, %d bytes (already_armed=%s)",
            addr,
            len(data),
            already_armed,
        )

    def consume_dump_next_cube_flag(self) -> bool:
        """Atomically check-and-clear the "dump next cube" flag.

        Returns:
            True if at least one UDP datagram arrived since the previous
            call (or since ``start()``); the flag is cleared as a side
            effect. False otherwise; subsequent calls return False until
            the next datagram arrives.

        Notes:
            One-shot semantics (D9): N datagrams between two consume
            calls still produce exactly one True. Use
            ``n_datagrams_received`` for the underlying count.

            Safe to call from any thread; in production this is invoked
            from the per-cube driver, which in chunk 5 may run on a
            worker thread separate from the asyncio event loop.
        """
        with self._lock:
            triggered = self._flag
            self._flag = False
            if triggered:
                self._n_consumed += 1
        return triggered

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_datagrams_received(self) -> int:
        """Monotonic count of datagrams received since construction.

        Includes datagrams that arrived after multiple were already
        pending (i.e. it does NOT collapse to the consumed count).
        """
        with self._lock:
            return self._n_datagrams

    @property
    def n_triggers_consumed(self) -> int:
        """Monotonic count of consume calls that returned True.

        Equal to the number of cubes the driver dumped due to a UDP
        trigger (modulo the dump-queue-full case, which is the
        cube_dump writer's problem).
        """
        with self._lock:
            return self._n_consumed

    @property
    def bound_port(self) -> int:
        """Actual port bound to.

        Equal to ``config.port`` when ``config.port != 0``; equal to the
        OS-assigned ephemeral port when ``config.port == 0`` (used by
        tests). Returns 0 if the listener is not started.
        """
        return self._bound_port

    @property
    def is_running(self) -> bool:
        """True iff the listener is currently bound and accepting datagrams."""
        return self._transport is not None
