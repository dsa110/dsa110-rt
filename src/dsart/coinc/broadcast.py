"""UDP fan-out of C2 trigger packets to the 8 C1 listeners.

The C2 service holds one :class:`TriggerBroadcaster` instance with the
``(search_node_id → ipv4)`` map from
``configs/dsart_search_rt.yaml::hostargs``. Each call to
:meth:`broadcast` sends a 64-byte :data:`wire.encode_c2_trigger`
packet to every ``(search_node_id, gpu_half)`` combination:

  * destination address: ``(hosts[sid], port_base + gpu_half)``
    (so port 11227 for gpu_half=0 and 11228 for gpu_half=1).

We use one shared connection-less UDP socket; UDP send is non-blocking
on Linux (the kernel buffers up to wmem_default) so eight sends in a
row complete in O(microseconds). The return value is a per-destination
success map so the service can record send failures into its
mon-points.

See ``docs/c1c2/C1C2_WIRE_SCHEMA.md`` §2 for the packet contract and
``docs/c1c2/C1C2_DESIGN.md`` §4 for the destination map convention.
"""

from __future__ import annotations

import logging
import socket
from typing import Dict, Mapping, Optional, Tuple

from . import wire

__all__ = [
    "TriggerBroadcaster",
    "DEFAULT_PORT_BASE",
    "GPU_HALVES",
]


_LOG = logging.getLogger("dsart.coinc.broadcast")

DEFAULT_PORT_BASE: int = 11227
GPU_HALVES: tuple[int, ...] = (0, 1)


class TriggerBroadcaster:
    """UDP fan-out to the 8 C1 dump-trigger listeners.

    Parameters
    ----------
    hosts:
        ``{search_node_id: ipv4_str}`` map.
    port_base:
        Base port; actual destination port is
        ``port_base + gpu_half`` per :data:`GPU_HALVES`.
    sock:
        Optional pre-bound ``socket.socket``; used by unit tests to
        inject a fake listener.
    """

    def __init__(
        self,
        hosts: Mapping[int, str],
        *,
        port_base: int = DEFAULT_PORT_BASE,
        sock: Optional[socket.socket] = None,
    ) -> None:
        if not hosts:
            raise ValueError("hosts map must be non-empty")
        self._hosts: Dict[int, str] = {int(k): str(v) for k, v in hosts.items()}
        self._port_base = int(port_base)
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Allow reusing the source port if the OS happens to bind it
            # to a fixed value (helpful in tests).
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock = sock

    @property
    def hosts(self) -> Mapping[int, str]:
        return self._hosts

    @property
    def port_base(self) -> int:
        return self._port_base

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "TriggerBroadcaster":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ----- public API ---------------------------------------------------

    def broadcast(
        self,
        event_name: str,
        event_specnum: int,
        mjd_target: float,
        *,
        trigger_class_id: int = 0,
        flags: int = wire.C2_TRIGGER_FLAG_DUMP_CUBE,
    ) -> Dict[Tuple[int, int], bool]:
        """Send a trigger packet to every (search_node, gpu_half).

        Returns a ``{(search_node_id, gpu_half): bool}`` map; True iff
        the send did not raise.
        """
        pkt = wire.C2TriggerPacket(
            event_name=event_name,
            event_specnum=event_specnum,
            mjd_target=mjd_target,
            trigger_class_id=trigger_class_id,
            flags=flags,
        )
        blob = wire.encode_c2_trigger(pkt)
        return self.broadcast_raw(blob)

    def broadcast_raw(
        self, blob: bytes,
    ) -> Dict[Tuple[int, int], bool]:
        out: Dict[Tuple[int, int], bool] = {}
        for sid, host in self._hosts.items():
            for g in GPU_HALVES:
                port = self._port_base + g
                try:
                    self._sock.sendto(blob, (host, port))
                    out[(sid, g)] = True
                except OSError as exc:
                    _LOG.warning(
                        "trigger send to (s=%d, g=%d) %s:%d failed: %s",
                        sid, g, host, port, exc,
                    )
                    out[(sid, g)] = False
        return out
