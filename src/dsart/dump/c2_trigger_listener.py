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
  * On a ``too_late`` miss (ring rotated past the cube), increments
    the corresponding mon-point counter and logs a WARNING with the
    current ring window.
  * On a ``too_early`` miss (the request names a cube this half has
    not produced yet), PARKS the request and re-checks the ring until
    the half's frontier reaches the event or
    ``too_early_retry_timeout_s`` elapses (2026-08-06, see
    :class:`C2TriggerListener`).

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        too_early_retry_timeout_s: how long a ``too_early`` request is
            parked and retried against the advancing ring before it is
            written off as a miss. Default 120 s: it covers the
            observed post-restart inter-half frontier spread of up to
            ~60 s with margin, and since the ring passes over every
            specnum exactly once a parked request either fulfils
            within that spread or the fleet is genuinely wedged.
            ``0.0`` restores the pre-2026-08-06 behaviour (a
            ``too_early`` request is an immediate terminal miss).
    """

    bind_host: str
    base_port: int
    gpu_half: int
    search_node_id: int
    dump_root: Path
    too_early_retry_timeout_s: float = 120.0

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
# Parked (too_early) requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParkedRequest:
    """A ``too_early`` request awaiting this half's frontier.

    Args:
        packet: the decoded trigger, replayed verbatim on fulfilment so
            the dump is byte-identical to an in-window request.
        addr: sender address, kept only for the log lines.
        t_park_s: ``time.monotonic()`` at park time (for the wait
            duration in the fulfilment log).
        deadline_s: ``t_park_s + too_early_retry_timeout_s``.
    """

    packet: C2TriggerPacket
    addr: Any
    t_park_s: float
    deadline_s: float

    @property
    def key(self) -> Tuple[str, int]:
        return (str(self.packet.event_name), int(self.packet.event_specnum))


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
      * ``too_early`` requests are parked on a list serviced by a
        small daemon thread (started lazily on the first park), so
        the socket-serving path never blocks and other requests keep
        being answered while parks are outstanding.

    ``too_early`` retry (2026-08-06)
    --------------------------------
    Tonight's dedicated-C1-sender fix cut the C1 → C2 → dump-request
    round trip from many seconds to ~1 s, so a request now routinely
    arrives BEFORE the slower halves have processed the cube holding
    the event: after a restart, halves' frontiers are frozen 1-60 s
    apart (every half runs at ≈realtime pace, so nothing ever catches
    up). With the old all-or-nothing behaviour those halves logged a
    terminal ``too_early`` miss ~4 s before they would have had the
    cube, events landed 4/8-5/8 cubes, and C2 DISCARDed the whole
    event dir. A ``too_early`` request is therefore parked and
    re-checked against the ring every
    :data:`_RETRY_POLL_INTERVAL_S`; when the frontier reaches the
    event it is dispatched exactly as a fresh in-window request would
    be (same manifest, same staging path). ``too_late`` is unchanged:
    the data really is gone and retrying cannot help.

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
        retry_poll_interval_s: how often the parked-request servicer
            re-checks the ring. Tests shorten it; production keeps
            the :data:`_RETRY_POLL_INTERVAL_S` default.
    """

    #: Ring re-check cadence for parked ``too_early`` requests. Cheap
    #: (a snapshot walk over ~12 cubes) and well under the ~2.5 s the
    #: ring retains, so no cube can appear and rotate out between polls.
    _RETRY_POLL_INTERVAL_S: float = 0.25

    #: Hard cap on simultaneously parked requests. C2 fires at most a
    #: handful of events per minute; anything beyond this means C2 is
    #: storming us, and parking without bound would grow the list (and
    #: each packet's retry work) forever. Overflow degrades to the old
    #: terminal-miss behaviour.
    _MAX_PARKED: int = 64

    def __init__(
        self,
        *,
        config: C2TriggerListenerConfig,
        ring: CubeRetentionRing,
        cube_dump: Optional[CubeDumpWriter] = None,
        dispatcher: Optional[
            Callable[[RetainedCube, C2TriggerPacket, CubeDumpManifest], bool]
        ] = None,
        stager: Optional[Any] = None,
        retry_poll_interval_s: Optional[float] = None,
    ) -> None:
        self._config = config
        self._ring = ring
        self._cube_dump = cube_dump
        self._dispatcher = dispatcher or self._default_dispatcher
        # Optional :class:`dsart.dump.proactive_stager.ProactiveCubeStager`.
        # On a live-ring HIT we drop its now-redundant staged copy; on a
        # ``too_late`` MISS we try to claim a proactively-staged cube for
        # this specnum (converting the miss into a rescued dump). When
        # None the listener behaves exactly as the M7.4 baseline.
        self._stager = stager
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_C2TriggerDatagramProtocol] = None
        self._bound_port: int = 0
        self._lock = threading.Lock()
        # Parked ``too_early`` requests + the daemon thread that
        # services them (started lazily on the first park).
        self._retry_poll_interval_s = float(
            self._RETRY_POLL_INTERVAL_S
            if retry_poll_interval_s is None
            else retry_poll_interval_s
        )
        self._park_lock = threading.Lock()
        self._parked: List[_ParkedRequest] = []
        self._retry_stop = threading.Event()
        self._retry_thread: Optional[threading.Thread] = None
        self._mon: Dict[str, Any] = {
            "received": 0,
            "hits": 0,
            "too_late": 0,
            # ``too_early`` counts only TERMINAL too_early misses, i.e.
            # requests whose retry timed out (pre-2026-08-06 this was
            # every too_early request).
            "too_early": 0,
            "too_early_parked": 0,
            "too_early_fulfilled": 0,
            "too_early_parked_now": 0,
            # Proactive-stage rescues: a ``too_late`` miss whose cube had
            # been staged proactively and was claimed into the event dir.
            "rescued": 0,
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
            self._stop_retry_thread()
            return
        try:
            self._transport.close()
        finally:
            self._transport = None
            self._protocol = None
            self._bound_port = 0
        self._stop_retry_thread()
        await asyncio.sleep(0)
        _LOG.info("C2TriggerListener stopped")

    def _stop_retry_thread(self) -> None:
        """Join the parked-request servicer and abandon any parks.

        Bounded join (the thread only ever sleeps ``poll_interval``
        between passes) so shutdown can't hang on it; the thread is a
        daemon anyway.
        """
        thread = self._retry_thread
        self._retry_thread = None
        self._retry_stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0 + self._retry_poll_interval_s)
        with self._park_lock:
            abandoned = list(self._parked)
            self._parked = []
        with self._lock:
            self._mon["too_early_parked_now"] = 0
        for parked in abandoned:
            _LOG.warning(
                "C2TriggerListener: abandoning parked too_early request "
                "at shutdown: event=%s specnum=%d",
                parked.packet.event_name,
                int(parked.packet.event_specnum),
            )

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

    def _classify_miss(self, event_specnum: int) -> Tuple[str, str]:
        """Classify a ring miss as ``too_late`` / ``too_early`` and
        render the ring window for the log line."""
        snapshot = self._ring.snapshot()
        if not snapshot:
            return "too_early", "ring empty"
        newest = snapshot[0]  # newest-first iter
        oldest = snapshot[-1]
        oldest_start = int(oldest.event_specnum_start)
        # SEARCH samples on both sides — a cube spans exactly
        # t_det of them, not t_det * sample_period_specnum (see
        # cube_pipeline.find_cube_for_specnum). The old form
        # reported a ring window 16x wider than the ring really
        # covers, so a genuine too_early miss could be logged
        # with a window that appeared to contain it.
        newest_end_excl = int(newest.event_specnum_start) + int(newest.t_det)
        kind = "too_late" if int(event_specnum) < oldest_start else "too_early"
        return kind, f"[{oldest_start}, {newest_end_excl})"

    def _handle_trigger(self, packet: C2TriggerPacket, addr) -> None:  # noqa: ANN001
        retained = find_cube_for_specnum(self._ring, packet.event_specnum)
        if retained is None:
            # Miss: classify too_late vs too_early using the ring's
            # oldest / newest specnum window.
            kind, window_str = self._classify_miss(packet.event_specnum)
            if (
                kind == "too_early"
                and float(self._config.too_early_retry_timeout_s) > 0.0
            ):
                # The cube isn't gone, it hasn't been produced yet:
                # park the request and let the ring come to it.
                self._park_too_early(packet, addr, window_str)
                return
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
            # Proactive-stage rescue (2026-07-21): the live ring rotated
            # past the cube (measured burst→request latency 7.8-9 s vs a
            # ~6.5 s ring), but if the detector proactively staged this
            # cube when it fired bright, claim it into the event dir now.
            # Only meaningful on ``too_late`` -- a ``too_early`` trigger
            # names a cube that hasn't been produced (hence never staged).
            if kind == "too_late" and self._stager is not None:
                try:
                    claimed = self._stager.claim(
                        event_specnum=int(packet.event_specnum),
                        event_name=str(packet.event_name),
                    )
                except Exception as exc:  # noqa: BLE001 - never poison loop
                    _LOG.warning(
                        "C2TriggerListener: proactive claim raised for "
                        "event=%s specnum=%d: %r",
                        packet.event_name, int(packet.event_specnum), exc,
                    )
                    claimed = None
                if claimed is not None:
                    with self._lock:
                        self._mon["rescued"] = int(self._mon["rescued"]) + 1
                    _LOG.info(
                        "C2TriggerListener rescued too_late event=%s "
                        "specnum=%d from proactive stage -> %s",
                        packet.event_name, int(packet.event_specnum), claimed,
                    )
            return
        self._dispatch_hit(packet, retained)

    # ------------------------------------------------------------------
    # too_early parking / retry
    # ------------------------------------------------------------------

    def _park_too_early(
        self,
        packet: C2TriggerPacket,
        addr,  # noqa: ANN001
        window_str: str,
    ) -> None:
        """Park a ``too_early`` request for retry against the ring.

        Runs on the event-loop thread; it only appends to a list and
        (once) starts the servicer thread, so the socket-serving path
        stays non-blocking.
        """
        timeout_s = float(self._config.too_early_retry_timeout_s)
        now = time.monotonic()
        parked = _ParkedRequest(
            packet=packet,
            addr=addr,
            t_park_s=now,
            deadline_s=now + timeout_s,
        )
        with self._park_lock:
            # A resent trigger for an already-parked event must not
            # park (and therefore dump) twice.
            if any(p.key == parked.key for p in self._parked):
                _LOG.debug(
                    "C2TriggerListener: too_early request already parked "
                    "(event=%s specnum=%d)",
                    packet.event_name, int(packet.event_specnum),
                )
                return
            overflow = len(self._parked) >= int(self._MAX_PARKED)
            if not overflow:
                self._parked.append(parked)
                n_parked = len(self._parked)
        if overflow:
            with self._lock:
                self._mon["too_early"] = int(self._mon["too_early"]) + 1
            _LOG.warning(
                "C2TriggerListener miss (too_early, park queue full at %d): "
                "event=%s specnum=%d from %s; ring window %s",
                int(self._MAX_PARKED),
                packet.event_name,
                int(packet.event_specnum),
                addr,
                window_str,
            )
            return
        with self._lock:
            self._mon["too_early_parked"] = (
                int(self._mon["too_early_parked"]) + 1
            )
            self._mon["too_early_parked_now"] = int(n_parked)
        _LOG.info(
            "C2TriggerListener parked too_early request: event=%s "
            "specnum=%d from %s; ring window %s; retrying up to %.0fs",
            packet.event_name,
            int(packet.event_specnum),
            addr,
            window_str,
            timeout_s,
        )
        self._ensure_retry_thread()

    def _ensure_retry_thread(self) -> None:
        thread = self._retry_thread
        if thread is not None and thread.is_alive():
            return
        self._retry_stop.clear()
        thread = threading.Thread(
            target=self._retry_loop,
            name="c2trig-retry",
            daemon=True,
        )
        self._retry_thread = thread
        thread.start()

    def _retry_loop(self) -> None:
        while not self._retry_stop.wait(self._retry_poll_interval_s):
            try:
                self._service_parked()
            except Exception:  # noqa: BLE001 - never kill the servicer
                _LOG.exception(
                    "C2TriggerListener: parked-request servicer pass failed"
                )

    def _service_parked(self) -> None:
        """One retry pass: fulfil parked requests the ring now covers,
        expire those past their deadline."""
        now = time.monotonic()
        ready: List[Tuple[_ParkedRequest, RetainedCube]] = []
        expired: List[_ParkedRequest] = []
        with self._park_lock:
            if not self._parked:
                return
            keep: List[_ParkedRequest] = []
            # Oldest park first, so parks fulfil in arrival order as the
            # frontier sweeps past them.
            for parked in self._parked:
                retained = find_cube_for_specnum(
                    self._ring, parked.packet.event_specnum,
                )
                if retained is not None:
                    ready.append((parked, retained))
                elif now >= parked.deadline_s:
                    expired.append(parked)
                else:
                    keep.append(parked)
            self._parked = keep
            n_parked = len(keep)
        with self._lock:
            self._mon["too_early_parked_now"] = int(n_parked)
        for parked, retained in ready:
            waited_s = now - parked.t_park_s
            with self._lock:
                self._mon["too_early_fulfilled"] = (
                    int(self._mon["too_early_fulfilled"]) + 1
                )
            _LOG.info(
                "C2TriggerListener too_early request fulfilled after %.1fs "
                "wait (event=%s, specnum=%d)",
                waited_s,
                parked.packet.event_name,
                int(parked.packet.event_specnum),
            )
            # Identical to the in-window path: same manifest, same
            # stager drop_pending, same dispatcher.
            self._dispatch_hit(parked.packet, retained)
        for parked in expired:
            kind, window_str = self._classify_miss(
                parked.packet.event_specnum
            )
            with self._lock:
                self._mon["too_early"] = int(self._mon["too_early"]) + 1
            _LOG.warning(
                "C2TriggerListener miss (too_early, retry timed out after "
                "%.0fs): event=%s specnum=%d from %s; ring window %s",
                float(self._config.too_early_retry_timeout_s),
                parked.packet.event_name,
                int(parked.packet.event_specnum),
                parked.addr,
                window_str,
            )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch_hit(
        self,
        packet: C2TriggerPacket,
        retained: RetainedCube,
    ) -> None:
        """Dump ``retained`` for ``packet``. Shared by the in-window
        path (event-loop thread) and the parked-retry path (servicer
        thread)."""
        with self._lock:
            self._mon["hits"] = int(self._mon["hits"]) + 1
        # Live-ring dump wins: discard any redundant proactively-staged
        # copy for this specnum so it isn't left to the TTL sweeper.
        if self._stager is not None:
            try:
                self._stager.drop_pending(int(packet.event_specnum))
            except Exception as exc:  # noqa: BLE001 - never poison loop
                _LOG.warning(
                    "C2TriggerListener: proactive drop_pending raised: %r", exc,
                )
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
            # NOTE: deliberately the TRIGGER specnum, not sample 0 --
            # the writer composes the NPZ filename from this and C3 +
            # the dashboards glob on cube_s*_g*_<trigger_specnum>.npz.
            # The real anchor goes in cube_specnum_start below.
            event_specnum_start=int(packet.event_specnum),
            mjd_start=float(retained.mjd_start),
            cube_specnum_start=int(retained.event_specnum_start),
            cube_mjd_start=float(retained.mjd_start),
            sample_period_specnum=int(retained.sample_period_specnum),
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
        # SLOT-PINNING (M7.6, 2026-05-31): ``retained.pinned_host_tensor``
        # aliases a CubeRetentionRing slot buffer the live pipeline reuses
        # every ``depth`` cubes. The writer serialises the ~855 MB NPZ
        # asynchronously; if the ring wraps back to that slot mid-dump it
        # would clobber the bytes (the peak_grid/cube mismatch). We MARK
        # the buffer in-flight so the ring allocates a fresh buffer for
        # that slot on reuse instead of overwriting this one, and RELEASE
        # it via ``on_complete`` once the writer is done. This is
        # zero-copy — earlier we copied the whole 855 MB here, which
        # OOM-killed the memory-tight 93 GiB search nodes (search_rx
        # SIGKILL, 2026-05-31). On queue-full (submit -> False) we release
        # immediately since no dump will run.
        buf = retained.pinned_host_tensor
        self._ring.mark_inflight(buf)
        accepted = bool(
            self._cube_dump.submit(
                cube=buf,
                manifest=manifest,
                on_complete=lambda b=buf: self._ring.release_inflight(b),
            )
        )
        if not accepted:
            self._ring.release_inflight(buf)
        return accepted

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def mon(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._mon)

    @property
    def n_parked(self) -> int:
        """Number of ``too_early`` requests currently awaiting the
        half's frontier."""
        with self._park_lock:
            return len(self._parked)

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
