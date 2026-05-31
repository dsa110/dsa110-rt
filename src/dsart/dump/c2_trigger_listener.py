"""C2 → C1 UDP trigger listener (M7.4 search-node side).

Replaces the M6 :mod:`udp_listener` (the legacy ``UdpTriggerListener``
that flipped a one-shot "dump next cube" flag with no payload). The
M7.4 listener:

  * Binds ``(c1.dump_listener.bind_host, c1.dump_listener.base_port +
    gpu_half)`` on the search-net interface (NOT 127.0.0.1 — h23 is
    the trigger source).
  * Decodes each 64-byte datagram as a ``C2TriggerPacket`` via
    ``coinc.wire.decode_c2_trigger``.
  * Looks up the matching cube in the CPU-side
    :class:`CubeRetentionRing` via :func:`find_cube_for_specnum`.
  * On a hit, dispatches a ``CubeDumpWriter`` job at
    ``${c1.dump_root}/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz``.
  * On a miss (ring rotated past the cube or trigger arrived before
    the cube), increments the corresponding mon-point counter and
    logs a WARNING with the current ring window.

The wire schema + cube-lookup semantics live in
``docs/c1c2/C1C2_WIRE_SCHEMA.md`` §2.

This module is additive: the legacy ``udp_listener.py`` /
``UdpTriggerListener`` is intentionally NOT removed so M6-era benches
and historical tests still import it. The production search-compute
service (``services/search_compute.py``) is the only call site that
flips from the legacy listener to :class:`C2TriggerListener`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
import threading

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..coinc.wire import (
    BadBatch,
    C2_TRIGGER_MAGIC,
    C2_TRIGGER_PACKET_SIZE,
    C2TriggerPacket,
    decode_c2_trigger,
)
from ..common.contracts import CubeDumpManifest
from ..services.cube_pipeline import (
    CubeRetentionRing,
    RetainedCube,
    find_cube_for_specnum,
)
from .cube_dump import CubeDumpWriter

_LOG = logging.getLogger("dsart.dump.c2_trigger_listener")

__all__ = [
    "C2TriggerListenerConfig",
    "C2TriggerListener",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class C2TriggerListenerConfig:
    """Bind + dispatch config for the C2 trigger listener.

    Args:
        bind_host: search-net IPv4 to bind. Production: the
            per-host nic.search interface (filled from
            ``dsart_search_rt.yaml::hostargs``); tests pass
            ``127.0.0.1``.
        base_port: actual bind port is ``base_port + gpu_half``
            (default base 11227 → 11227 for half 0, 11228 for half 1).
        gpu_half: 0 or 1.
        search_node_id: 0..N_SEARCH-1.
        dump_root: directory under which event subdirs are created
            (one ``<event_name>/`` per fired event).
    """

    bind_host: str
    base_port: int
    gpu_half: int
    search_node_id: int
    dump_root: Path

    @property
    def bind_port(self) -> int:
        return int(self.base_port) + int(self.gpu_half)


# ---------------------------------------------------------------------------
# DatagramProtocol
# ---------------------------------------------------------------------------


class _C2TriggerDatagramProtocol(asyncio.DatagramProtocol):
    def __init__(self, listener: "C2TriggerListener") -> None:
        self._listener = listener

    def datagram_received(self, data: bytes, addr) -> None:  # noqa: ANN001
        self._listener._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        _LOG.warning("C2TriggerListener UDP error_received: %r", exc)

    def connection_lost(self, exc: Optional[BaseException]) -> None:
        if exc is not None:
            _LOG.warning(
                "C2TriggerListener UDP connection_lost: %r", exc
            )


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


class C2TriggerListener:
    """Async UDP listener that consumes ``C2TriggerPacket``s and
    dispatches matched cubes to a :class:`CubeDumpWriter` against the
    :class:`CubeRetentionRing`.

    Concurrency model:
      * ``datagram_received`` runs on the asyncio event-loop thread.
      * The ring is read-only here; ring writes happen on the cube
        driver. ``find_cube_for_specnum`` walks the snapshot list
        with no shared state mutation, so no lock is required.
      * Mon-points are protected by a ``threading.Lock`` so external
        monitor pushers can read consistent counters.

    Args:
        config: bind + dispatch config.
        ring: shared :class:`CubeRetentionRing` populated by the cube
            driver (one ring per gpu_half).
        cube_dump: :class:`CubeDumpWriter` instance (one writer per
            gpu_half).
        clock_monotonic_ns: optional clock injector (used in tests).
        dispatcher: optional override of the dispatch callback
            (``(retained, packet, manifest) -> bool``). Defaults to a
            wrapper around ``cube_dump.submit``. Tests use this to
            assert dispatch without actually writing NPZs.
    """

    def __init__(
        self,
        *,
        config: C2TriggerListenerConfig,
        ring: CubeRetentionRing,
        cube_dump: Optional[CubeDumpWriter] = None,
        dispatcher: Optional[
            Callable[[RetainedCube, C2TriggerPacket, CubeDumpManifest], bool]
        ] = None,
    ) -> None:
        self._config = config
        self._ring = ring
        self._cube_dump = cube_dump
        self._dispatcher = dispatcher or self._default_dispatcher
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_C2TriggerDatagramProtocol] = None
        self._bound_port: int = 0
        self._lock = threading.Lock()
        self._mon: Dict[str, Any] = {
            "received": 0,
            "hits": 0,
            "too_late": 0,
            "too_early": 0,
            "bad_magic": 0,
            "bad_schema": 0,
            "dispatched": 0,
            "dispatch_dropped": 0,
            "last_event_name": "",
            "last_event_specnum": 0,
            "bind_host": str(config.bind_host),
            "bind_port": int(config.bind_port),
            "search_node_id": int(config.search_node_id),
            "gpu_half": int(config.gpu_half),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._transport is not None:
            raise RuntimeError(
                "C2TriggerListener.start() called while already started"
            )
        loop = asyncio.get_running_loop()
        try:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _C2TriggerDatagramProtocol(self),
                local_addr=(self._config.bind_host, self._config.bind_port),
                family=socket.AF_INET,
                allow_broadcast=False,
            )
        except OSError as exc:
            _LOG.error(
                "C2TriggerListener bind failed on %s:%d (%r)",
                self._config.bind_host,
                self._config.bind_port,
                exc,
            )
            raise
        sock = transport.get_extra_info("socket")
        self._bound_port = int(
            sock.getsockname()[1] if sock is not None else self._config.bind_port
        )
        self._transport = transport
        self._protocol = protocol
        _LOG.info(
            "C2TriggerListener bound to %s:%d (configured %d, sid=%d gpu_half=%d)",
            self._config.bind_host,
            self._bound_port,
            self._config.bind_port,
            self._config.search_node_id,
            self._config.gpu_half,
        )

    async def stop(self) -> None:
        if self._transport is None:
            return
        try:
            self._transport.close()
        finally:
            self._transport = None
            self._protocol = None
            self._bound_port = 0
        await asyncio.sleep(0)
        _LOG.info("C2TriggerListener stopped")

    # ------------------------------------------------------------------
    # Datagram handling
    # ------------------------------------------------------------------

    def _on_datagram(self, data: bytes, addr) -> None:  # noqa: ANN001
        with self._lock:
            self._mon["received"] = int(self._mon["received"]) + 1
        if len(data) != C2_TRIGGER_PACKET_SIZE:
            with self._lock:
                self._mon["bad_schema"] = int(self._mon["bad_schema"]) + 1
            _LOG.warning(
                "C2TriggerListener: bad packet size from %s (got %d, expected %d)",
                addr, len(data), C2_TRIGGER_PACKET_SIZE,
            )
            return
        # Cheap magic prefilter before the full decode (drops random
        # noise / port scans without throwing).
        magic_le = int.from_bytes(data[:4], "little")
        if magic_le != C2_TRIGGER_MAGIC:
            with self._lock:
                self._mon["bad_magic"] = int(self._mon["bad_magic"]) + 1
            _LOG.warning(
                "C2TriggerListener: bad magic 0x%08x from %s", magic_le, addr,
            )
            return
        try:
            packet = decode_c2_trigger(data)
        except BadBatch as exc:
            with self._lock:
                self._mon["bad_schema"] = int(self._mon["bad_schema"]) + 1
            _LOG.warning(
                "C2TriggerListener: decode failed from %s: %s", addr, exc,
            )
            return
        with self._lock:
            self._mon["last_event_name"] = str(packet.event_name)
            self._mon["last_event_specnum"] = int(packet.event_specnum)
        self._handle_trigger(packet, addr)

    def _handle_trigger(self, packet: C2TriggerPacket, addr) -> None:  # noqa: ANN001
        retained = find_cube_for_specnum(self._ring, packet.event_specnum)
        if retained is None:
            # Miss: classify too_late vs too_early using the ring's
            # oldest / newest specnum window.
            snapshot = self._ring.snapshot()
            if not snapshot:
                kind = "too_early"
                window_str = "ring empty"
            else:
                newest = snapshot[0]  # newest-first iter
                oldest = snapshot[-1]
                oldest_start = int(oldest.event_specnum_start)
                newest_end_excl = (
                    int(newest.event_specnum_start)
                    + int(newest.t_det) * int(newest.sample_period_specnum)
                )
                if int(packet.event_specnum) < oldest_start:
                    kind = "too_late"
                else:
                    kind = "too_early"
                window_str = (
                    f"[{oldest_start}, {newest_end_excl})"
                )
            with self._lock:
                self._mon[kind] = int(self._mon[kind]) + 1
            _LOG.warning(
                "C2TriggerListener miss (%s): event=%s specnum=%d "
                "from %s; ring window %s",
                kind,
                packet.event_name,
                int(packet.event_specnum),
                addr,
                window_str,
            )
            return
        with self._lock:
            self._mon["hits"] = int(self._mon["hits"]) + 1
        manifest = self._build_manifest(packet, retained)
        accepted = self._dispatcher(retained, packet, manifest)
        with self._lock:
            if accepted:
                self._mon["dispatched"] = int(self._mon["dispatched"]) + 1
            else:
                self._mon["dispatch_dropped"] = (
                    int(self._mon["dispatch_dropped"]) + 1
                )

    def _build_manifest(
        self,
        packet: C2TriggerPacket,
        retained: RetainedCube,
    ) -> CubeDumpManifest:
        """Compose the ``CubeDumpManifest`` for the writer thread.

        Path layout: ``${dump_root}/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz``.
        The writer thread internally composes its own canonical path
        from the manifest's ``event_specnum_start`` and ``(sid, g)``
        config; we put the event-specnum-based name in the manifest's
        ``npz_path`` field (informational) so downstream consumers can
        find it.
        """
        event_dir = Path(self._config.dump_root) / str(packet.event_name)
        npz_path = event_dir / (
            f"cube_s{int(self._config.search_node_id)}"
            f"_g{int(self._config.gpu_half)}"
            f"_{int(packet.event_specnum)}.npz"
        )
        return CubeDumpManifest(
            cube_id=int(retained.cube_id),
            event_specnum_start=int(packet.event_specnum),
            mjd_start=float(retained.mjd_start),
            t_det=int(retained.t_det),
            n_fdm_in_cube=int(retained.n_fdm),
            n_grid=int(retained.n_grid),
            trigger_source="udp",  # C2 path = external-triggered (uses UDP transport)
            cluster_record=None,
            npz_path=str(npz_path),
            search_node_id=int(self._config.search_node_id),
            gpu_half=int(self._config.gpu_half),
        )

    def _default_dispatcher(
        self,
        retained: RetainedCube,
        packet: C2TriggerPacket,
        manifest: CubeDumpManifest,
    ) -> bool:
        if self._cube_dump is None:
            _LOG.warning(
                "C2TriggerListener: no cube_dump writer wired; "
                "dropping event=%s specnum=%d",
                packet.event_name, int(packet.event_specnum),
            )
            return False
        # The M7.4 CubeDumpWriter respects ``manifest.npz_path`` and
        # auto-creates the parent directory if it differs from the
        # canonical compose. The manifest we built in
        # :meth:`_build_manifest` points at
        # ``${dump_root}/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz``.
        #
        # SNAPSHOT the cube out of the ring slot BEFORE enqueueing.
        # ``retained.pinned_host_tensor`` ALIASES a CubeRetentionRing
        # slot buffer that the live pipeline reuses every ``depth`` cubes
        # (~1 s at production cadence). The writer thread serialises the
        # ~855 MB NPZ asynchronously, which outlives that reuse window,
        # so without this copy the slot is overwritten mid-dump: the
        # writer reduces ``peak_grid`` from the still-intact buffer but
        # the ``cube`` bytes ``np.savez`` streams out afterwards come
        # from the clobbered next-cube data — the peak_grid(~60000) /
        # cube(~noise) mismatch we observed on injection dumps. The copy
        # runs synchronously here in the listener thread while the
        # matched cube is guaranteed still live in the ring (we just
        # found it via find_cube_for_specnum microseconds ago), so it
        # captures the correct, self-consistent cube. ~855 MB memcpy
        # (~0.3 s) is acceptable on the rare C2-dump path.
        cube_snapshot = np.array(retained.pinned_host_tensor, copy=True)
        return bool(
            self._cube_dump.submit(
                cube=cube_snapshot, manifest=manifest,
            )
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

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

    @property
    def ring(self) -> CubeRetentionRing:
        return self._ring

    def set_ring(self, ring: CubeRetentionRing) -> None:
        """Swap the retention ring (used by the search-compute service
        which lazily rebuilds the ring once the first cube's geometry
        is known)."""
        self._ring = ring
