"""C1 → C2 TCP batch emitter (M7.4 search-node side).

Per ``docs/c1c2/C1C2_DESIGN.md`` §2.6 + ``C1C2_WIRE_SCHEMA.md`` §1,
each ``(search_node_id, gpu_half)`` maintains exactly one persistent
TCP connection to ``c1.c2_endpoint`` (default ``h23:11500``). The
:class:`C1TcpEmitter` exposed here owns that connection's lifecycle
and a bounded outbound queue. The per-cube driver calls
:meth:`C1TcpEmitter.submit` synchronously with a fully-built
``C1BatchHeader`` + the cube's survivor list; the connection task
drains the queue, encodes each batch via the locked
``coinc.wire.C1BatchEncoder``, and ``sendall``s it onto the socket.

Backpressure semantics:
  * Outbound queue is bounded (default depth = 16).
  * ``submit`` is non-blocking: queue-full drops are counted as
    ``batches_dropped`` and surfaced via :attr:`mon`.
  * The connection task reconnects on socket-level failures with
    exponential backoff (250 ms → 30 s cap) per the wire schema.

The module imports asyncio + socket only (no torch / numpy). The
encoder is pure (no I/O) so unit tests round-trip through a local
in-process TCP echo server.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from ..coinc.wire import (
    C1BatchEncoder,
    C1BatchHeader,
    C1CandidateRow,
    SCHEMA_VERSION,
)
from ..common.contracts import Candidate, CubeGeometry

__all__ = [
    "C1EmitConfig",
    "C1TcpEmitter",
    "candidate_to_c1_row",
]


_LOG = logging.getLogger("dsart.services.c1_emit")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class C1EmitConfig:
    """Static config for the per-(search_node, gpu_half) C1 emitter.

    Args:
        host: C2 server hostname / IPv4.
        port: C2 server TCP port (default 11500 per wire schema).
        search_node_id: 0..N_SEARCH-1.
        gpu_half: 0 or 1.
        queue_depth: outbound asyncio queue depth (per design doc
            default 16).
        connect_backoff_initial_s: first reconnect sleep on failure.
        connect_backoff_max_s: cap on the reconnect sleep.
        send_timeout_s: per-batch ``asyncio.wait_for`` deadline for the
            ``writer.drain()`` call. None disables.
        connect_timeout_s: per-attempt ``asyncio.wait_for`` deadline for
            ``asyncio.open_connection``. Without this, the Linux TCP
            connect retransmit timer (RTO doubling: 1, 2, 4, ..., capped
            at ~2 min on stock Ubuntu) can keep the connect task hung
            for the full retry sequence when SYN packets are dropped
            silently by conntrack / a bridge / a stuck CLOSE-WAIT in the
            peer's listen queue. Concretely seen 2026-05-28 on
            ``n01 gpu_half=0``: the c1_emit run() coroutine logged
            "c1_emit connecting" but neither success nor "connection
            failed" for 6+ minutes despite the peer being reachable
            from the same host on a different source port. A short
            per-attempt deadline forces the outer retry loop to fire,
            which observably self-healed every prior backpressure /
            broken-pipe transient.
    """

    host: str
    port: int = 11500
    search_node_id: int = 0
    gpu_half: int = 0
    queue_depth: int = 16
    connect_backoff_initial_s: float = 0.25
    connect_backoff_max_s: float = 30.0
    send_timeout_s: Optional[float] = 5.0
    connect_timeout_s: Optional[float] = 10.0
    idle_heartbeat_s: Optional[float] = 20.0
    """When the outbound queue is idle for this long, the drain loop
    proactively (a) probes for a half-closed peer (``reader.at_eof()`` /
    ``transport.is_closing()``) so a connection the C2 receiver dropped is
    detected and reconnected instead of wedging in CLOSE-WAIT forever, and
    (b) ships a 0-row keepalive heartbeat so C2's ``idle_timeout_s``
    (default 60 s — see ``coinc/receiver.py``) never fires on a search half
    that is briefly not producing cubes.

    This is the fix for the recurring "7 of 8 connections" symptom (seen on
    ``n01 gpu_half=0``): the search side only writes to C2 on a per-cube
    ``submit()``; the ``_drain_loop`` otherwise blocks on ``queue.get()``
    and never reads the socket, so when C2 idle-closed a quiet half the
    search side parked in CLOSE-WAIT and never reconnected. Must be < C2's
    ``idle_timeout_s``. None disables the idle behaviour (legacy)."""
    tcp_keepalive: bool = True
    """Enable SO_KEEPALIVE + aggressive TCP keepalive timers on the
    connected socket so the kernel surfaces a dead/half-open peer as a
    socket error even when neither side has app-level traffic. Defence in
    depth behind ``idle_heartbeat_s``."""
    tcp_keepidle_s: int = 20
    tcp_keepintvl_s: int = 5
    tcp_keepcnt: int = 3
    max_width_samples: Optional[int] = None
    """Drop any candidate whose boxcar ``width_samples`` exceeds this
    before it is transmitted to C2. ``None`` disables the filter (legacy:
    every survivor is shipped). Wide boxcars (≥32) over-respond to
    correlated sky / low-level broadband RFI and dominate the on-sky
    false-positive floor (2026-05-29 analysis: 95-100 % of the spurious
    candidate volume sat at width ≥ 32), saturating the C1→C2 path and
    skewing the coincidence ``peak_event_specnum`` toward stale cubes
    (→ ``too_late`` dump misses). Real FRBs are predominantly narrow, so a
    width cap of 16 strips the false-positive floor while preserving
    genuine narrow events. Configured via ``c1.max_c1c2_width_samples`` in
    ``dsart_search_rt.yaml``."""
    max_candidates_per_block: Optional[int] = None
    """C1→C2 metering: cap the number of candidates transmitted per cube
    (block). ``None`` / ``<= 0`` disables (ship every width-survivor).

    When more than this many candidates survive the width cap in a single
    cube, only the top ``max_candidates_per_block`` are shipped, selected
    narrow-first (``width_samples`` ascending) and then bright-first
    (``snr`` descending) — i.e. if there are too many width-1/2 candidates
    we keep the highest-SNR width-2 ones. This bounds the worst-case
    C1→C2 + C2-clustering load during RFI floods (the 2026-05-28 soak saw
    a single source emit ~3300 cands/window and lag C2 ~25 s, so every
    dump trigger arrived ``too_late``). With 8 search halves × ~7.45
    cubes/s ≈ 60 batches/s into one C2, a cap of 8 bounds the fleet
    worst-case at ~480 rows/s while never biting normal load (a few
    cands/cube/half at ``snr_min``) or a real burst (a handful of narrow
    candidates near the peak).     RT-safe: selection is O(k log N) via
    ``heapq.nsmallest`` and is skipped entirely when under the cap.
    Configured via ``c1.max_candidates_per_block`` in
    ``dsart_search_rt.yaml``."""
    dm_width_floor_frac: Optional[float] = None
    """C1→C2 physical-plausibility filter: drop any candidate whose
    boxcar ``width_samples`` is below ``dm_width_floor_frac`` × the
    intra-channel dispersion-smearing floor for its DM, evaluated at the
    ACTUAL per-cube sample period (see ``search_compute.dm_smear_samples``;
    the service passes ``geom.sample_period_us``). ``None`` / ``<= 0``
    disables.

    At the production cadence (t_int_search = 1048.576 µs) the smearing
    floor at DM≈2500 is ~1.8 samples, so width-2 high-DM detections are
    *consistent* with smearing and are kept; only clearly-narrow (width-1)
    high-DM detections are rejected. ``0.6`` sheds width-1 above DM≈2330
    while never touching low-DM narrow events (the floor is sub-sample at
    low DM) or real high-DM bursts (whose matched-filter width sits at the
    floor, i.e. boxcar ≥2). Background: 2026-05-30 a single search half
    flooded C2 with field-filling, time-incoherent width-1 spikes at
    DM≈2538, which matched the ``bright_pulsar`` class and drove a dump
    storm. Width alone is not a strong discriminator at the production
    cadence — the trigger criteria + C2 dump-rate cap do the heavy lifting;
    this filter is cheap defense-in-depth that removes the unphysical
    width-1 tail at the source. Configured via ``c1.dm_width_floor_frac``
    in ``dsart_search_rt.yaml``."""


# ---------------------------------------------------------------------------
# Candidate → C1CandidateRow mapping
# ---------------------------------------------------------------------------


def candidate_to_c1_row(
    cand: Candidate,
    *,
    geom: CubeGeometry,
) -> C1CandidateRow:
    """Project a ``Candidate`` + ``CubeGeometry`` sidecar into a
    locked-schema ``C1CandidateRow``.

    The detector emits Candidates with ``l``, ``m`` as float-cast
    pixel indices and ``dm_idx`` as the global fine-DM index (the
    search-compute service threads the cube's ``DmPlan`` view to
    ``decode_local_max``, which writes the index from
    ``fine_to_coarse`` if present — see ``detector/decoder.py``). The
    ``CubeGeometry`` provides the radian-units conversion and the
    per-cube fine-DM grid; we use it to populate both ``l_rad / m_rad``
    and ``dm_pc_cc / fine_dm_idx``.

    ``fine_dm_idx`` is recovered as the local index into
    ``geom.fine_dm_pc_cc`` by matching ``dm_fine`` ≈
    ``geom.fine_dm_pc_cc[fine_dm_idx]``. If ``dm_fine`` doesn't fall
    on the per-cube grid (e.g. test fixtures that pass synthetic
    floats), we clamp ``fine_dm_idx`` to 0; the C2 receiver tolerates
    that.
    """
    l_pix = int(round(float(cand.l)))
    m_pix = int(round(float(cand.m)))
    n_grid = int(geom.n_grid)
    if l_pix < 0:
        l_pix = 0
    elif l_pix >= n_grid:
        l_pix = n_grid - 1
    if m_pix < 0:
        m_pix = 0
    elif m_pix >= n_grid:
        m_pix = n_grid - 1
    l_rad = float(geom.l0_rad) + float(geom.cell_l_rad) * float(cand.l)
    m_rad = float(geom.m0_rad) + float(geom.cell_m_rad) * float(cand.m)

    # Recover per-cube fine_dm_idx via the geometry grid. We match on
    # the float value rather than carrying it through the Candidate to
    # avoid bloating that contract — the cube grid is small and the
    # lookup is rare (a handful of survivors per cube).
    fine_dm = geom.fine_dm_pc_cc
    fine_dm_idx = 0
    if fine_dm is not None and fine_dm.size > 0:
        # Closest-grid lookup; bench / test fixtures may pass synthetic
        # ``dm_fine`` floats that don't exactly fall on the grid.
        diffs = abs(fine_dm - float(cand.dm_fine))
        fine_dm_idx = int(diffs.argmin())
    return C1CandidateRow(
        snr=float(cand.snr),
        l_rad=float(l_rad),
        m_rad=float(m_rad),
        l_pix=int(l_pix),
        m_pix=int(m_pix),
        dm_pc_cc=float(cand.dm_fine),
        dm_idx_global=int(cand.dm_idx),
        fine_dm_idx=int(fine_dm_idx),
        event_specnum=int(cand.event_specnum),
        width_samples=int(cand.width_samples),
        kernel_id=str(cand.kernel_id),
        flags=int(cand.flags),
    )


# ---------------------------------------------------------------------------
# Outbound queue payload + emitter task
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OutboundBatch:
    header: C1BatchHeader
    rows: Tuple[C1CandidateRow, ...]


class C1TcpEmitter:
    """Asyncio-driven persistent TCP client for the C1 → C2 path.

    Lifecycle:
        ``emitter = C1TcpEmitter(config=cfg)``
        ``task = asyncio.create_task(emitter.run())``
        ``... emitter.submit(header, candidates) ...``
        ``await emitter.stop()``
        ``await task``

    The :meth:`run` coroutine owns the connection lifecycle; it
    reconnects on any socket-level error with exponential backoff. The
    :meth:`submit` method is synchronous and non-blocking so the
    per-cube driver can call it directly without yielding to the
    event loop.

    Mon-points exposed via :attr:`mon`:
        * ``connected`` (bool) — True while a writer is open.
        * ``bytes_sent`` (int) — cumulative encoded-batch bytes shipped.
        * ``batches_sent`` (int) — cumulative successful sends.
        * ``batches_dropped`` (int) — cumulative queue-full drops.
        * ``batches_send_failed`` (int) — sends that raised mid-send.
        * ``reconnects`` (int) — count of socket reconnect attempts.
        * ``queue_depth`` (int) — current outbound queue depth.
        * ``last_connect_error`` (str | None) — last connect-error str
          (mostly for operator triage).
        * ``last_connected_at_ns`` (int) — monotonic-ns of last
          successful connect.
    """

    def __init__(self, *, config: C1EmitConfig) -> None:
        if config.queue_depth <= 0:
            raise ValueError(
                f"queue_depth={config.queue_depth}, expected > 0"
            )
        self._config = config
        self._queue: "asyncio.Queue[_OutboundBatch]" = asyncio.Queue(
            maxsize=config.queue_depth
        )
        self._stopping = asyncio.Event()
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader: Optional[asyncio.StreamReader] = None
        # Last real batch header seen by the drain loop, reused (with
        # n_candidates=0) as the keepalive-heartbeat template. None until
        # the first cube flows; idle keepalive then relies on at_eof +
        # TCP keepalive only.
        self._last_header: Optional[C1BatchHeader] = None
        # Rate-limiter for the queue-full drop warning. Initialise to
        # -inf so the very first drop logs immediately; subsequent
        # drops are batched per the policy in ``submit``.
        self._last_drop_log_mono: float = float("-inf")
        self._mon: Dict[str, Any] = {
            "connected": False,
            "bytes_sent": 0,
            "batches_sent": 0,
            "batches_dropped": 0,
            "batches_send_failed": 0,
            "heartbeats_sent": 0,
            "reconnects": 0,
            "queue_depth": 0,
            "last_connect_error": None,
            "last_connected_at_ns": 0,
            "host": config.host,
            "port": int(config.port),
            "search_node_id": int(config.search_node_id),
            "gpu_half": int(config.gpu_half),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        header: C1BatchHeader,
        candidates: Sequence[Candidate],
        *,
        geom: Optional[CubeGeometry] = None,
        rows: Optional[Sequence[C1CandidateRow]] = None,
    ) -> bool:
        """Non-blocking enqueue. Returns True on success, False on
        queue-full drop.

        Pass ``rows=`` when the caller has already projected
        Candidates into ``C1CandidateRow``s (saves one walk on the
        hot path). When ``rows`` is None, the caller must pass
        ``geom`` so the row mapping can fill ``l_rad / m_rad /
        fine_dm_idx``.

        The encoded header field ``n_candidates`` must match
        ``len(rows)`` after projection; this method asserts it.
        """
        if rows is None:
            if geom is None:
                raise ValueError(
                    "submit() requires either rows= or geom= to project candidates"
                )
            rows_tuple = tuple(
                candidate_to_c1_row(c, geom=geom) for c in candidates
            )
        else:
            rows_tuple = tuple(rows)
        if header.n_candidates != len(rows_tuple):
            raise ValueError(
                f"header.n_candidates={header.n_candidates} != "
                f"len(rows)={len(rows_tuple)}"
            )
        try:
            self._queue.put_nowait(_OutboundBatch(header=header, rows=rows_tuple))
            self._mon["queue_depth"] = self._queue.qsize()
            return True
        except asyncio.QueueFull:
            self._mon["batches_dropped"] = int(self._mon["batches_dropped"]) + 1
            # Rate-limited drop log: a stuck C2 emitter would otherwise
            # flood every cube (steady-state ~5–7 lines / s × 8 halves
            # = 60+ lines/s on the search-rx logger). Suppress unless
            # we've seen at least N more drops since the previous warn
            # OR M seconds elapsed — keeps the "first drop after a long
            # quiet period" observable but stops the spam during
            # continuous backpressure.
            n_dropped = int(self._mon["batches_dropped"])
            should_log = (
                n_dropped == 1
                or n_dropped % 100 == 0
                or (time.monotonic() - self._last_drop_log_mono) > 30.0
            )
            if should_log:
                _LOG.warning(
                    "c1_emit drop (queue full); cube_id=%d depth=%d "
                    "(total_dropped=%d)",
                    int(header.cube_id), self._queue.maxsize, n_dropped,
                )
                self._last_drop_log_mono = time.monotonic()
            return False

    async def stop(self) -> None:
        """Signal the run loop to drain + exit. Idempotent."""
        self._stopping.set()
        # Push a sentinel so the queue-await wakes up promptly.
        try:
            self._queue.put_nowait(_SHUTDOWN_SENTINEL)
        except asyncio.QueueFull:
            # Queue is full; loop will check _stopping after the next
            # drain.
            pass

    @property
    def mon(self) -> Dict[str, Any]:
        """Live mon-point dict. Snapshot-friendly (caller may dict-copy)."""
        self._mon["queue_depth"] = self._queue.qsize()
        return self._mon

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connection + send lifecycle. Reconnects on failure with
        exponential backoff until :meth:`stop` is called."""
        backoff = float(self._config.connect_backoff_initial_s)
        while not self._stopping.is_set():
            try:
                await self._connect_once()
                # Successful connect → reset backoff.
                backoff = float(self._config.connect_backoff_initial_s)
                await self._drain_loop()
            except asyncio.CancelledError:
                _LOG.info("c1_emit run() cancelled; tearing down")
                break
            except Exception as exc:  # noqa: BLE001
                self._mon["last_connect_error"] = repr(exc)
                self._mon["connected"] = False
                _LOG.warning(
                    "c1_emit connection failed (%s:%d): %r; sleeping %.2fs",
                    self._config.host, self._config.port, exc, backoff,
                )
            finally:
                await self._close_writer()
            if self._stopping.is_set():
                break
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=backoff
                )
            except asyncio.TimeoutError:
                pass
            backoff = min(
                backoff * 2.0,
                float(self._config.connect_backoff_max_s),
            )

    async def _connect_once(self) -> None:
        _LOG.info(
            "c1_emit connecting to %s:%d (sid=%d, gpu_half=%d, timeout=%ss)",
            self._config.host,
            self._config.port,
            self._config.search_node_id,
            self._config.gpu_half,
            self._config.connect_timeout_s,
        )
        # Wrap asyncio.open_connection in wait_for so a SYN-blackhole
        # path (silently-dropped SYNs at conntrack / bridge / peer
        # listen queue) cannot wedge run() indefinitely. The outer
        # run() loop catches asyncio.TimeoutError as part of its
        # generic Exception handler, logs "connection failed
        # (TimeoutError)", and retries with exponential backoff -- the
        # observable self-heal pattern. See C1EmitConfig docstring for
        # the 2026-05-28 n01 gpu_half=0 incident this guards.
        timeout = self._config.connect_timeout_s
        if timeout is not None and timeout > 0:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host=self._config.host, port=self._config.port,
                ),
                timeout=float(timeout),
            )
        else:
            self._reader, self._writer = await asyncio.open_connection(
                host=self._config.host, port=self._config.port,
            )
        self._apply_keepalive()
        self._mon["connected"] = True
        self._mon["reconnects"] = int(self._mon["reconnects"]) + 1
        self._mon["last_connect_error"] = None
        self._mon["last_connected_at_ns"] = int(time.monotonic_ns())
        _LOG.info(
            "c1_emit connected to %s:%d (sid=%d, gpu_half=%d, reconnects=%d)",
            self._config.host,
            self._config.port,
            self._config.search_node_id,
            self._config.gpu_half,
            int(self._mon["reconnects"]),
        )

    def _apply_keepalive(self) -> None:
        """Enable SO_KEEPALIVE + aggressive timers on the connected socket.

        Lets the kernel detect a dead / half-open peer (e.g. a C2 that went
        away without a clean FIN) and surface it as a socket error on the
        next drain, even when the application is idle. Best-effort: missing
        sockopts (non-Linux) are ignored.
        """
        if not self._config.tcp_keepalive or self._writer is None:
            return
        sock = self._writer.get_extra_info("socket")
        if sock is None:
            return
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):
                sock.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,
                    int(self._config.tcp_keepidle_s),
                )
            if hasattr(socket, "TCP_KEEPINTVL"):
                sock.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_KEEPINTVL,
                    int(self._config.tcp_keepintvl_s),
                )
            if hasattr(socket, "TCP_KEEPCNT"):
                sock.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_KEEPCNT,
                    int(self._config.tcp_keepcnt),
                )
        except OSError as exc:  # noqa: BLE001
            _LOG.debug("c1_emit keepalive sockopt failed: %r", exc)

    async def _drain_loop(self) -> None:
        assert self._writer is not None
        idle = self._config.idle_heartbeat_s
        while not self._stopping.is_set():
            if idle is not None and idle > 0:
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=float(idle),
                    )
                except asyncio.TimeoutError:
                    # No cube traffic for `idle` s. Probe for a peer that
                    # closed on us and keep the connection warm so C2's
                    # idle_timeout never idle-closes a cube-drought half.
                    # Raises on a dead/half-closed peer → outer loop
                    # reconnects (clears the CLOSE-WAIT socket).
                    await self._idle_keepalive()
                    continue
            else:
                item = await self._queue.get()
            self._mon["queue_depth"] = self._queue.qsize()
            if item is _SHUTDOWN_SENTINEL:
                return
            # Seed / refresh the heartbeat template from real traffic.
            self._last_header = item.header
            try:
                await self._send_batch(item.header, item.rows)
            finally:
                # The asyncio.Queue tracks unfinished tasks; signal done.
                self._queue.task_done()

    async def _send_batch(
        self,
        header: C1BatchHeader,
        rows: Tuple[C1CandidateRow, ...],
    ) -> None:
        """Encode + write one batch; raise on socket error so the outer
        run() loop reconnects."""
        assert self._writer is not None
        writer = self._writer
        try:
            payload = C1BatchEncoder.encode(header, rows)
            writer.write(payload)
            if self._config.send_timeout_s is not None:
                await asyncio.wait_for(
                    writer.drain(), timeout=self._config.send_timeout_s,
                )
            else:
                await writer.drain()
            self._mon["bytes_sent"] = int(self._mon["bytes_sent"]) + len(payload)
            self._mon["batches_sent"] = int(self._mon["batches_sent"]) + 1
        except (ConnectionError, asyncio.TimeoutError, OSError) as exc:
            self._mon["batches_send_failed"] = (
                int(self._mon["batches_send_failed"]) + 1
            )
            _LOG.warning(
                "c1_emit send failed (cube_id=%d): %r; tearing connection",
                int(header.cube_id), exc,
            )
            raise  # outer loop reconnects

    async def _idle_keepalive(self) -> None:
        """Idle-period connection health-check + keepalive heartbeat.

        Called when the outbound queue has been empty for
        ``idle_heartbeat_s``. First detects a peer that closed on us (the
        C2 idle_timeout path leaves us in CLOSE-WAIT); a closed peer raises
        so run() reconnects. Then ships a 0-row heartbeat (if we have a
        header template) so C2 never idle-closes a quiet-but-healthy half.
        """
        reader = self._reader
        writer = self._writer
        if reader is not None and reader.at_eof():
            raise ConnectionResetError(
                "c1_emit: peer closed connection (EOF on idle probe)"
            )
        if writer is not None and writer.transport.is_closing():
            raise ConnectionResetError(
                "c1_emit: transport closing (idle probe)"
            )
        # Always ship a 0-row heartbeat so C2's idle_timeout never reaps a
        # quiet-but-healthy half. CRITICAL: do NOT gate this on having seen
        # a real batch (_last_header). A half that owns a coarse-DM range
        # with no candidates yet (e.g. n01 gpu_half=0 at startup) would
        # otherwise never heartbeat, C2 would idle-close it every 60 s, and
        # the subsequent reconnect/dead-socket-send race can wedge the drain
        # loop (observed 2026-05-29: n01 half-0 stuck in perpetual queue-full
        # with no reconnect). We always know our sid/gpu_half, so a synthetic
        # 0-candidate header is sufficient to keep the connection warm.
        hb_header = self._heartbeat_header()
        await self._send_batch(hb_header, ())
        self._mon["heartbeats_sent"] = (
            int(self._mon["heartbeats_sent"]) + 1
        )

    def _heartbeat_header(self) -> C1BatchHeader:
        """0-candidate header for an idle keepalive.

        Reuses the last real batch header (n_candidates forced to 0) when
        available so C2 sees a consistent cube_id stream; otherwise
        synthesises a minimal header from our static (sid, gpu_half) so a
        half that has not yet emitted a single candidate can still keep the
        connection warm. cube_id=-1 marks a synthetic heartbeat (C2 returns
        early on n_candidates==0, so the geometry fields are never read)."""
        if self._last_header is not None:
            return dataclasses.replace(self._last_header, n_candidates=0)
        # sample_period_specnum / sample_period_us MUST be > 0: the C2
        # parser (_parse_header) rejects <= 0 as a BadBatch, which would
        # tear the connection on every heartbeat. For a 0-candidate batch
        # C2 returns before reading any geometry, so these are placeholders
        # (1 / 1.0); only the > 0 validity matters.
        return C1BatchHeader(
            schema_version=SCHEMA_VERSION,
            cube_id=-1,
            event_specnum_start=0,
            mjd_start=0.0,
            sample_period_specnum=1,
            sample_period_us=1.0,
            n_grid=0,
            n_fdm_in_cube=0,
            search_node_id=int(self._config.search_node_id),
            gpu_half=int(self._config.gpu_half),
            n_candidates=0,
        )

    async def _close_writer(self) -> None:
        w = self._writer
        self._writer = None
        self._reader = None
        self._mon["connected"] = False
        if w is None:
            return
        try:
            w.close()
            await w.wait_closed()
        except Exception:  # noqa: BLE001
            pass


# Sentinel used to wake the queue.get() in the run loop on stop().
_SHUTDOWN_SENTINEL: Any = object()
