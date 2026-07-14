"""C2 → corr-node UDP voltage-dump trigger listener.

Corr-side analogue of :mod:`dsart.dump.c2_trigger_listener` (the
search-node cube listener). Binds one UDP port on the corr-net interface
and turns each 64-byte :class:`~dsart.coinc.wire.C2TriggerPacket` into a
dump (or delete) request handed to the retention service's dump worker.

Routing:
  * ``DUMP_VOLTAGE`` flag set, ``event_specnum > 0``  → ``on_dump(name, specnum)``
  * ``DUMP_VOLTAGE`` flag set, ``event_specnum == 0`` → ``on_delete(name)``
    (the C3 REJECT path: delete this event's staged voltages)
  * flag clear / bad magic / bad size                  → counted + dropped

The handlers are expected to be cheap + non-blocking (they enqueue onto
the dump worker's thread-safe queue); all heavy work happens off the
event loop in the worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..coinc.wire import (
    BadBatch,
    C2_TRIGGER_FLAG_DUMP_VOLTAGE,
    C2_TRIGGER_MAGIC,
    C2_TRIGGER_PACKET_SIZE,
    C2TriggerPacket,
    decode_c2_trigger,
)

_LOG = logging.getLogger("dsart.dump.voltage_trigger_listener")

__all__ = [
    "VoltageTriggerListenerConfig",
    "VoltageTriggerListener",
]


@dataclass(frozen=True, slots=True)
class VoltageTriggerListenerConfig:
    bind_host: str
    bind_port: int
    cn_id: int = 0
    chgroup: int = 0


class _DatagramProtocol(asyncio.DatagramProtocol):
    def __init__(self, listener: "VoltageTriggerListener") -> None:
        self._listener = listener

    def datagram_received(self, data: bytes, addr) -> None:  # noqa: ANN001
        self._listener._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        _LOG.warning("VoltageTriggerListener UDP error_received: %r", exc)


class VoltageTriggerListener:
    """Async UDP listener for corr-node voltage dump/delete triggers.

    Parameters
    ----------
    config:
        bind host/port + node identity (for mon labelling).
    on_dump:
        ``(event_name, event_specnum, mjd_target) -> bool`` — enqueue a
        dump; return False if the queue was full (counted as
        ``queue_full``). ``mjd_target`` is the packet's event MJD,
        recorded in the staging manifest.
    on_delete:
        ``(event_name) -> bool`` — enqueue a staged-voltage delete.
    """

    def __init__(
        self,
        *,
        config: VoltageTriggerListenerConfig,
        on_dump: Callable[[str, int, float], bool],
        on_delete: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._config = config
        self._on_dump = on_dump
        self._on_delete = on_delete
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._bound_port: int = 0
        self._lock = threading.Lock()
        self._mon: Dict[str, Any] = {
            "received": 0,
            "dump_flagged": 0,
            "enqueued": 0,
            "deletes": 0,
            "queue_full": 0,
            "not_flagged": 0,
            "bad_magic": 0,
            "bad_size": 0,
            "bad_schema": 0,
            "last_event_name": "",
            "last_event_specnum": 0,
            "bind_host": str(config.bind_host),
            "bind_port": int(config.bind_port),
            "cn_id": int(config.cn_id),
            "chgroup": int(config.chgroup),
        }

    async def start(self) -> None:
        if self._transport is not None:
            raise RuntimeError("VoltageTriggerListener already started")
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _DatagramProtocol(self),
            local_addr=(self._config.bind_host, self._config.bind_port),
            family=socket.AF_INET,
            allow_broadcast=False,
        )
        sock = transport.get_extra_info("socket")
        self._bound_port = int(
            sock.getsockname()[1] if sock is not None
            else self._config.bind_port
        )
        self._transport = transport
        _LOG.info(
            "VoltageTriggerListener bound %s:%d (cn=%d chgroup=%d)",
            self._config.bind_host, self._bound_port,
            self._config.cn_id, self._config.chgroup,
        )

    async def stop(self) -> None:
        if self._transport is None:
            return
        try:
            self._transport.close()
        finally:
            self._transport = None
            self._bound_port = 0
        await asyncio.sleep(0)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._mon[key] = int(self._mon[key]) + n

    def _on_datagram(self, data: bytes, addr) -> None:  # noqa: ANN001
        self._bump("received")
        if len(data) != C2_TRIGGER_PACKET_SIZE:
            self._bump("bad_size")
            return
        if int.from_bytes(data[:4], "little") != C2_TRIGGER_MAGIC:
            self._bump("bad_magic")
            return
        try:
            pkt: C2TriggerPacket = decode_c2_trigger(data)
        except BadBatch:
            self._bump("bad_schema")
            return
        with self._lock:
            self._mon["last_event_name"] = str(pkt.event_name)
            self._mon["last_event_specnum"] = int(pkt.event_specnum)
        if not (int(pkt.flags) & C2_TRIGGER_FLAG_DUMP_VOLTAGE):
            self._bump("not_flagged")
            return
        self._bump("dump_flagged")
        if int(pkt.event_specnum) == 0:
            # Delete sentinel (C3 REJECT path).
            self._bump("deletes")
            if self._on_delete is not None:
                try:
                    self._on_delete(str(pkt.event_name))
                except Exception:  # noqa: BLE001
                    _LOG.exception("on_delete handler raised")
            return
        try:
            accepted = bool(
                self._on_dump(
                    str(pkt.event_name),
                    int(pkt.event_specnum),
                    float(pkt.mjd_target),
                )
            )
        except Exception:  # noqa: BLE001
            _LOG.exception("on_dump handler raised")
            accepted = False
        if accepted:
            self._bump("enqueued")
        else:
            self._bump("queue_full")
            _LOG.warning(
                "voltage dump queue full; dropped event=%s specnum=%d",
                pkt.event_name, int(pkt.event_specnum),
            )

    @property
    def mon(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._mon)

    @property
    def bound_port(self) -> int:
        return int(self._bound_port)

    @property
    def is_running(self) -> bool:
        return self._transport is not None
