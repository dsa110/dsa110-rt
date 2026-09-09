"""Tests for the M7.4 C1 → C2 TCP batch emitter
(``src/dsart/services/c1_emit.py``).

Goals:
  * Round-trip a header + a few rows through an in-process asyncio
    TCP echo server and verify the bytes parse via
    ``coinc.wire.parse_c1_batch``.
  * Submit drops on queue overflow (non-blocking, returns False).
  * ``candidate_to_c1_row`` projects (l, m) / dm to the locked
    schema.
  * Reconnect on socket error (server disappears mid-stream).
  * 2026-08-06 sender-thread regressions: a producer that floods
    ``submit()`` from the service event loop without ever awaiting must
    NOT starve the sender (``test_threaded_sender_survives_producer_
    starvation``), the threaded sender reconnects when C2 restarts, and
    drop accounting still works when C2 stops reading.

The ``run()`` coroutine is still driven directly on the test loop by
the older tests (it remains public); the new tests exercise the
production path, ``start()`` / ``stop()``.
"""

from __future__ import annotations

import asyncio
import functools
import os
import socket
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")


def asyncio_test(func):
    """Run an async coroutine inside a fresh event loop per test
    (mirrors the helper in ``tests/test_udp_listener.py``)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper

from dsart.coinc.wire import (  # noqa: E402
    SCHEMA_VERSION,
    parse_c1_batch,
)
from dsart.common.contracts import Candidate, CandidateFlags, CubeGeometry  # noqa: E402
from dsart.services.c1_emit import (  # noqa: E402
    C1EmitConfig,
    C1TcpEmitter,
    candidate_to_c1_row,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cand(
    *,
    snr: float = 12.5,
    l: float = 32.0,
    m: float = 48.0,
    dm_fine: float = 75.0,
    dm_idx: int = 3,
    event_specnum: int = 1024,
    width_samples: int = 1,
    kernel_id: str = "unit:d1:b1",
) -> Candidate:
    return Candidate(
        l=l,
        m=m,
        dm_fine=dm_fine,
        dm_idx=dm_idx,
        event_specnum=event_specnum,
        width_samples=width_samples,
        kernel_id=kernel_id,
        snr=snr,
        detector_version="v1.M5",
        flags=int(CandidateFlags.NONE),
        search_node_id=0,
        gpu_half=0,
    )


def _geom(n_grid: int = 256, n_fdm: int = 8) -> CubeGeometry:
    fine_dm = np.linspace(50.0, 800.0, n_fdm, dtype=np.float64)
    return CubeGeometry(
        cube_id=42,
        specnum_start=1000,
        sample_period_specnum=16,
        t_det=192,
        n_grid=n_grid,
        n_fdm_in_cube=n_fdm,
        sample_period_us=1048.576,
        cell_l_rad=1.5e-4,
        cell_m_rad=1.5e-4,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=fine_dm,
        mjd_start=58000.0,
    )


def _header(emitter: C1TcpEmitter, *, n_candidates: int, cube_id: int = 42):
    from dsart.coinc.wire import build_header
    geom = _geom()
    return build_header(
        cube_id=cube_id,
        event_specnum_start=int(geom.specnum_start),
        mjd_start=float(geom.mjd_start),
        sample_period_specnum=int(geom.sample_period_specnum),
        sample_period_us=float(geom.sample_period_us),
        n_grid=int(geom.n_grid),
        n_fdm_in_cube=int(geom.n_fdm_in_cube),
        search_node_id=emitter._config.search_node_id,
        gpu_half=emitter._config.gpu_half,
        n_candidates=n_candidates,
    )


class _EchoServer:
    """Tiny in-process TCP server that collects everything a client
    writes into a single bytearray."""

    def __init__(self) -> None:
        self.port: int = 0
        self.bytes_seen = bytearray()
        self._server: asyncio.AbstractServer | None = None
        self._conns: List[asyncio.StreamWriter] = []
        self._lock = asyncio.Lock()

    async def start(self, host: str = "127.0.0.1") -> int:
        async def _on_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            self._conns.append(writer)
            try:
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        break
                    async with self._lock:
                        self.bytes_seen.extend(chunk)
            finally:
                try:
                    writer.close()
                except Exception:
                    pass

        self._server = await asyncio.start_server(_on_client, host=host, port=0)
        sock = self._server.sockets[0]
        self.port = int(sock.getsockname()[1])
        return self.port

    async def stop(self) -> None:
        for w in self._conns:
            try:
                w.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


# ---------------------------------------------------------------------------
# candidate_to_c1_row
# ---------------------------------------------------------------------------


def test_candidate_to_c1_row_basic_mapping() -> None:
    geom = _geom()
    cand = _cand(l=32.4, m=48.7, snr=11.5, dm_idx=5, dm_fine=75.0)
    row = candidate_to_c1_row(cand, geom=geom)
    assert row.snr == pytest.approx(11.5)
    assert row.l_pix == 32
    assert row.m_pix == 49
    assert row.dm_pc_cc == pytest.approx(75.0)
    assert row.dm_idx_global == 5
    assert row.event_specnum == cand.event_specnum
    assert row.width_samples == cand.width_samples
    assert row.kernel_id == cand.kernel_id
    # Radian conversion: CENTRED image (origin at n_grid//2 = 128) and
    # row/col → (m, l) axis swap — Candidate.l is the cube ROW = sky
    # m-axis; Candidate.m is the COLUMN = sky l-axis (2026-06-10 live
    # injection sweep). Uses the float l/m, not the int pixel.
    assert row.l_rad == pytest.approx(geom.cell_l_rad * (48.7 - 128))
    assert row.m_rad == pytest.approx(geom.cell_m_rad * (32.4 - 128))


def test_candidate_to_c1_row_centred_swapped_axes() -> None:
    """The cube image is CENTRED (sky origin at pixel n_grid//2) and the
    cube row axis is sky m while the column axis is sky l. Confirmed
    live 2026-06-10: an injection at (l, m) = (+0.009, +0.0045) rad
    landed at cube (row, col) = (164, 200) ≈ (0.0045/cell + 128,
    0.009/cell + 128); injections at m=0 previously reported the
    notorious ±0.0192 "corner" value (= (0-128)·1.5e-4) under the old
    raw-FFT-layout interpretation."""
    geom = _geom()  # n_grid=256
    cand = _cand(l=196.0, m=60.0)
    row = candidate_to_c1_row(cand, geom=geom)
    # l_rad from the COLUMN (cand.m), m_rad from the ROW (cand.l).
    assert row.l_rad == pytest.approx(geom.cell_l_rad * (60 - 128))
    assert row.m_rad == pytest.approx(geom.cell_m_rad * (196 - 128))
    # Pixel indices stay in raw cube (row, col) layout (they index
    # dumped cubes).
    assert row.l_pix == 196
    assert row.m_pix == 60


def test_candidate_to_c1_row_clamps_pixel_indices() -> None:
    geom = _geom(n_grid=64)
    # l = -3 should clamp to 0; m = 100 should clamp to 63.
    cand = _cand(l=-3.0, m=100.0)
    row = candidate_to_c1_row(cand, geom=geom)
    assert row.l_pix == 0
    assert row.m_pix == 63


def test_candidate_to_c1_row_recovers_fine_dm_idx() -> None:
    geom = _geom(n_fdm=8)
    # Pick a dm_fine value exactly on the grid -> fine_dm_idx matches.
    target_idx = 5
    cand = _cand(dm_fine=float(geom.fine_dm_pc_cc[target_idx]))
    row = candidate_to_c1_row(cand, geom=geom)
    assert row.fine_dm_idx == target_idx


# ---------------------------------------------------------------------------
# submit() preconditions
# ---------------------------------------------------------------------------


@asyncio_test
async def test_submit_validates_count() -> None:
    cfg = C1EmitConfig(
        host="127.0.0.1", port=_free_port(),
        search_node_id=0, gpu_half=0, queue_depth=4,
    )
    emitter = C1TcpEmitter(config=cfg)
    cands = [_cand(snr=10.0), _cand(snr=11.0)]
    # Build a header with the wrong n_candidates and expect ValueError.
    header_wrong = _header(emitter, n_candidates=99)
    with pytest.raises(ValueError):
        emitter.submit(header_wrong, cands, geom=_geom())


@asyncio_test
async def test_submit_requires_rows_or_geom() -> None:
    cfg = C1EmitConfig(
        host="127.0.0.1", port=_free_port(),
        search_node_id=0, gpu_half=0, queue_depth=4,
    )
    emitter = C1TcpEmitter(config=cfg)
    cands = [_cand(snr=10.0)]
    header = _header(emitter, n_candidates=len(cands))
    with pytest.raises(ValueError):
        emitter.submit(header, cands)  # no rows= and no geom=


# ---------------------------------------------------------------------------
# Round-trip: encode → write → parse
# ---------------------------------------------------------------------------


@asyncio_test
async def test_round_trip_through_echo_server() -> None:
    srv = _EchoServer()
    port = await srv.start()
    try:
        cfg = C1EmitConfig(
            host="127.0.0.1", port=port,
            search_node_id=2, gpu_half=1, queue_depth=4,
            connect_backoff_initial_s=0.01,
            connect_backoff_max_s=0.05,
            send_timeout_s=2.0,
        )
        emitter = C1TcpEmitter(config=cfg)
        task = asyncio.create_task(emitter.run())
        try:
            # Submit one batch with two candidates.
            cands = [
                _cand(snr=15.0, l=10.0, m=20.0, dm_idx=3, dm_fine=75.0),
                _cand(snr=12.0, l=200.0, m=100.0, dm_idx=7, dm_fine=375.0,
                       event_specnum=2048, width_samples=4,
                       kernel_id="psf:d3:b4"),
            ]
            header = _header(emitter, n_candidates=len(cands))
            geom = _geom()
            assert emitter.submit(header, cands, geom=geom) is True
            # Allow the run loop to connect and drain.
            for _ in range(200):
                await asyncio.sleep(0.01)
                if emitter.mon["batches_sent"] >= 1:
                    break
            assert emitter.mon["batches_sent"] == 1
            assert emitter.mon["connected"]
        finally:
            emitter.stop()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()
        # Parse what the server saw.
        text = srv.bytes_seen.decode("utf-8")
        lines = [ln for ln in text.splitlines() if ln]
        batch = parse_c1_batch(lines)
        assert batch.header.schema_version == SCHEMA_VERSION
        assert batch.header.cube_id == 42
        assert batch.header.search_node_id == 2
        assert batch.header.gpu_half == 1
        assert batch.header.n_candidates == 2
        assert len(batch.candidates) == 2
        snrs = sorted(r.snr for r in batch.candidates)
        assert snrs == pytest.approx([12.0, 15.0])
    finally:
        await srv.stop()


# ---------------------------------------------------------------------------
# Queue-full overflow
# ---------------------------------------------------------------------------


@asyncio_test
async def test_submit_drops_on_queue_full() -> None:
    # No server: the run loop will keep retrying connect, leaving the
    # queue full so we can observe drops without racing the drain loop.
    cfg = C1EmitConfig(
        host="127.0.0.1",
        port=_free_port(),
        search_node_id=0,
        gpu_half=0,
        queue_depth=2,
        connect_backoff_initial_s=10.0,
        connect_backoff_max_s=10.0,
    )
    emitter = C1TcpEmitter(config=cfg)
    geom = _geom()
    cands = [_cand(snr=10.0)]
    header1 = _header(emitter, n_candidates=1, cube_id=1)
    header2 = _header(emitter, n_candidates=1, cube_id=2)
    header3 = _header(emitter, n_candidates=1, cube_id=3)
    assert emitter.submit(header1, cands, geom=geom) is True
    assert emitter.submit(header2, cands, geom=geom) is True
    # Third should drop.
    assert emitter.submit(header3, cands, geom=geom) is False
    assert emitter.mon["batches_dropped"] >= 1
    assert emitter.mon["queue_depth"] == 2


# ---------------------------------------------------------------------------
# Reconnect on server disappearance
# ---------------------------------------------------------------------------


@asyncio_test
async def test_reconnect_after_server_drop() -> None:
    srv = _EchoServer()
    port = await srv.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=0, gpu_half=0, queue_depth=64,
        connect_backoff_initial_s=0.05,
        connect_backoff_max_s=0.05,
        send_timeout_s=2.0,
    )
    emitter = C1TcpEmitter(config=cfg)
    task = asyncio.create_task(emitter.run())
    try:
        # Wait for first connect.
        for _ in range(100):
            await asyncio.sleep(0.02)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"]
        # Drop the server.
        await srv.stop()
        await asyncio.sleep(0.05)  # let TCP FIN/RST propagate
        # Spam batches until the TCP layer raises EPIPE or
        # ConnectionResetError on the writer. ``writer.write()`` itself
        # may buffer silently after a half-closed peer, but the next
        # ``writer.drain()`` against the dead socket will raise.
        geom = _geom()
        cands = [_cand(snr=10.0)]
        for cid in range(40):
            header = _header(emitter, n_candidates=1, cube_id=cid + 100)
            emitter.submit(header, cands, geom=geom)
            await asyncio.sleep(0.05)
            if (
                emitter.mon["batches_send_failed"] >= 1
                or emitter.mon["reconnects"] >= 2
            ):
                break
        assert (
            emitter.mon["batches_send_failed"] >= 1
            or emitter.mon["reconnects"] >= 2
        ), emitter.mon
    finally:
        emitter.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
        await srv.stop()


@asyncio_test
async def test_connect_timeout_fires_on_syn_blackhole() -> None:
    """The run() loop must self-heal when asyncio.open_connection hangs
    (e.g. SYN dropped by conntrack / firewall / a stuck peer listen
    queue). Without an upstream timeout the TCP stack retransmits SYN
    for ~2 min before giving up; the c1_emit task would silently wedge
    that entire window while the cube ring fills and total_dropped
    climbs unbounded. With connect_timeout_s the wait_for raises
    TimeoutError which the outer except logs as "connection failed"
    and the backoff retries -- the same self-heal path every other
    transient already exercises.

    Reproduces the symptom by pointing at TEST-NET-1 (RFC 5737:
    192.0.2.0/24 reserved for docs, guaranteed unrouted). The kernel
    never gets a SYN-ACK; without the wait_for the open_connection
    call would hang until the SYN retransmit timer gives up.
    """
    cfg = C1EmitConfig(
        host="192.0.2.1",  # TEST-NET-1 -- guaranteed-unrouted RFC 5737
        port=11500,
        search_node_id=0,
        gpu_half=0,
        queue_depth=8,
        connect_backoff_initial_s=0.05,
        connect_backoff_max_s=0.05,
        send_timeout_s=2.0,
        connect_timeout_s=0.3,  # short for test; production default 10s
    )
    emitter = C1TcpEmitter(config=cfg)
    task = asyncio.create_task(emitter.run())
    try:
        # The first wait_for should fire at ~0.3s; with the 0.05s
        # backoff cap, by 2.0s we should have at least 3-4 attempts
        # logged on last_connect_error.
        await asyncio.sleep(2.0)
        assert not emitter.mon["connected"], (
            "should never connect to TEST-NET-1"
        )
        # The last error should be a TimeoutError, and we should have
        # cycled through the run() loop several times. If the wait_for
        # is missing or its timeout is None, we'd be stuck on the very
        # first open_connection call with last_connect_error == None.
        last_err = emitter.mon["last_connect_error"] or ""
        assert "Timeout" in last_err or "timeout" in last_err, (
            f"expected TimeoutError-class last_connect_error; "
            f"got {last_err!r}. The connect_timeout_s watchdog is "
            f"not firing. mon={dict(emitter.mon)}"
        )
    finally:
        emitter.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()


@asyncio_test
async def test_connect_timeout_disabled_when_none() -> None:
    """Setting connect_timeout_s=None disables the wait_for wrapper.
    Used by callers that want the OS TCP RTO sequence (rare, e.g.
    bench scripts that simulate long-haul WAN connects)."""
    cfg = C1EmitConfig(
        host="127.0.0.1",
        port=_free_port(),
        search_node_id=0,
        gpu_half=0,
        queue_depth=8,
        connect_backoff_initial_s=0.05,
        connect_backoff_max_s=0.05,
        send_timeout_s=2.0,
        connect_timeout_s=None,
    )
    emitter = C1TcpEmitter(config=cfg)
    # Just verify the config accepts None and the emitter constructs
    # without raising. Full no-timeout dynamics are hard to test
    # without injecting OS-level packet drops.
    assert emitter._config.connect_timeout_s is None
    assert emitter is not None


# ---------------------------------------------------------------------------
# Idle keepalive heartbeat + peer-close (CLOSE-WAIT) self-heal
# ---------------------------------------------------------------------------


@asyncio_test
async def test_idle_heartbeat_keeps_connection_warm() -> None:
    """With no cube traffic the drain loop must ship 0-row keepalive
    heartbeats so C2's idle_timeout never idle-closes a quiet half. This
    is the proactive half of the 7/8-connections fix.

    Seeds the header template with one real batch, then idles and asserts
    a heartbeat (n_candidates=0) lands at the server within a couple of
    idle intervals.
    """
    srv = _EchoServer()
    port = await srv.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=3, gpu_half=1, queue_depth=64,
        connect_backoff_initial_s=0.05, connect_backoff_max_s=0.05,
        send_timeout_s=2.0,
        idle_heartbeat_s=0.2,  # tiny for test; production 20s
    )
    emitter = C1TcpEmitter(config=cfg)
    task = asyncio.create_task(emitter.run())
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"]
        # Seed the heartbeat template with one real cube.
        emitter.submit(_header(emitter, n_candidates=1, cube_id=7),
                       [_cand(snr=10.0)], geom=_geom())
        # Now idle: heartbeats must accrue without any further submit().
        for _ in range(100):
            await asyncio.sleep(0.05)
            if emitter.mon["heartbeats_sent"] >= 2:
                break
        assert emitter.mon["heartbeats_sent"] >= 2, emitter.mon
        # The server should have received at least one 0-candidate batch.
        await asyncio.sleep(0.1)
        text = srv.bytes_seen.decode("utf-8")
        lines = [ln for ln in text.splitlines() if ln]
        # Header lines look like "# C1 <schema> <cube> ... <n_candidates>"
        # (n_candidates is the last whitespace token). A heartbeat is a
        # header whose n_candidates == 0.
        n_zero = sum(
            1 for ln in lines
            if ln.startswith("# C1") and ln.split()[-1] == "0"
        )
        assert n_zero >= 1, f"no heartbeat batch seen; lines={lines!r}"
    finally:
        emitter.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
        await srv.stop()


def test_synthetic_heartbeat_header_parses() -> None:
    """The synthetic (no prior batch) heartbeat header must round-trip
    through the real C2 parser without BadBatch. Regression for the
    2026-05-29 bug where sample_period_specnum/us=0 made every
    candidate-less heartbeat a BadBatch -> connection torn in a loop ->
    n01 dropped to 0 connections."""
    from dsart.coinc.wire import C1BatchEncoder, parse_c1_batch
    cfg = C1EmitConfig(
        host="127.0.0.1", port=1, search_node_id=1, gpu_half=0,
    )
    emitter = C1TcpEmitter(config=cfg)
    hb = emitter._heartbeat_header()
    assert hb.n_candidates == 0
    assert hb.search_node_id == 1 and hb.gpu_half == 0
    payload = C1BatchEncoder.encode(hb, ())
    batch = parse_c1_batch(payload.decode("utf-8").splitlines())
    assert batch.header.n_candidates == 0
    assert batch.header.search_node_id == 1
    assert batch.header.gpu_half == 0


@asyncio_test
async def test_idle_heartbeat_without_prior_batch() -> None:
    """A half that has NEVER emitted a candidate (e.g. n01 gpu_half=0 in a
    cube drought at startup) must still ship 0-row heartbeats so C2 does
    not idle-close it every 60 s. Regression for the 2026-05-29 n01 half-0
    wedge: previously the heartbeat was gated on a seeded _last_header, so
    a candidate-less half never heartbeated -> repeated 60 s idle-close ->
    drain-loop wedge. The synthetic header must carry our sid/gpu_half.
    """
    srv = _EchoServer()
    port = await srv.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=1, gpu_half=0, queue_depth=64,
        connect_backoff_initial_s=0.05, connect_backoff_max_s=0.05,
        send_timeout_s=2.0,
        idle_heartbeat_s=0.2,
    )
    emitter = C1TcpEmitter(config=cfg)
    task = asyncio.create_task(emitter.run())
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"]
        # No submit() ever: heartbeats must still accrue.
        for _ in range(100):
            await asyncio.sleep(0.05)
            if emitter.mon["heartbeats_sent"] >= 2:
                break
        assert emitter.mon["heartbeats_sent"] >= 2, emitter.mon
        await asyncio.sleep(0.1)
        text = srv.bytes_seen.decode("utf-8")
        lines = [ln for ln in text.splitlines() if ln.startswith("# C1")]
        assert lines, f"no batch header seen; bytes={text!r}"
        # Every header here is a 0-candidate heartbeat carrying sid=1,
        # gpu_half=0 (header tokens: # C1 <schema> <cube> <specnum>
        # <mjd> <samp_specnum> <samp_us> <n_grid> <n_fdm> <sid> <half> <ncand>).
        for ln in lines:
            toks = ln.split()
            assert toks[-1] == "0", f"non-heartbeat header: {ln!r}"
            assert toks[-3] == "1" and toks[-2] == "0", (
                f"heartbeat header missing sid/gpu_half: {ln!r}"
            )
    finally:
        emitter.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
        await srv.stop()


@asyncio_test
async def test_idle_probe_reconnects_on_peer_close() -> None:
    """If the peer closes the connection during a cube drought (C2's
    idle_timeout path -> FIN -> our socket parks in CLOSE-WAIT), the idle
    probe must detect the EOF and force run() to reconnect, instead of
    wedging forever. This is the reactive half of the 7/8 fix.
    """
    srv = _EchoServer()
    port = await srv.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=0, gpu_half=0, queue_depth=64,
        connect_backoff_initial_s=0.05, connect_backoff_max_s=0.05,
        send_timeout_s=2.0,
        idle_heartbeat_s=0.15,
        # Disable the heartbeat-template path for this test by never
        # seeding a header: relies purely on at_eof detection.
    )
    emitter = C1TcpEmitter(config=cfg)
    task = asyncio.create_task(emitter.run())
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"]
        reconnects_before = int(emitter.mon["reconnects"])
        # Server closes its client sockets (clean FIN) -> our side EOF.
        for w in list(srv._conns):
            try:
                w.close()
            except Exception:
                pass
        srv._conns.clear()
        # Within a few idle intervals the probe must notice + reconnect.
        for _ in range(100):
            await asyncio.sleep(0.05)
            if int(emitter.mon["reconnects"]) > reconnects_before:
                break
        assert int(emitter.mon["reconnects"]) > reconnects_before, (
            f"idle probe did not reconnect after peer close; mon="
            f"{dict(emitter.mon)}"
        )
    finally:
        emitter.stop()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
        await srv.stop()


# ---------------------------------------------------------------------------
# 2026-08-06: dedicated sender thread
# ---------------------------------------------------------------------------


class _ThreadedSink:
    """Fake C2 receiver on REAL threads (no asyncio).

    The starvation regression below deliberately blocks the test's event
    loop for the whole flood, so the fake C2 must not live on that loop:
    an ``asyncio.start_server`` sink would neither accept nor read while
    the producer runs, and the test would pass/fail for the wrong
    reason. This acceptor runs one thread for ``accept`` plus one reader
    thread per connection.

    ``hang=True`` accepts connections but NEVER reads them, so the
    kernel receive window closes and the sender wedges in ``drain()``
    — the "C2 stopped reading" case.
    """

    def __init__(self, *, hang: bool = False, rcvbuf: Optional[int] = None):
        self.hang = bool(hang)
        self.rcvbuf = rcvbuf
        self.port: int = 0
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._srv: Optional[socket.socket] = None
        self._conns: List[socket.socket] = []
        self._threads: List[threading.Thread] = []
        self._closing = threading.Event()

    def start(self, port: int = 0) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.rcvbuf is not None:
            # Set on the LISTENING socket so accepted sockets inherit it
            # (must be set before the handshake for the window to shrink).
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(self.rcvbuf))
        srv.bind(("127.0.0.1", int(port)))
        srv.listen(8)
        srv.settimeout(0.2)
        self._srv = srv
        self.port = int(srv.getsockname()[1])
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        self._threads.append(t)
        return self.port

    def _accept_loop(self) -> None:
        while not self._closing.is_set():
            try:
                assert self._srv is not None
                conn, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            self._conns.append(conn)
            if self.hang:
                continue  # accepted, never read
            rt = threading.Thread(target=self._read_loop, args=(conn,),
                                  daemon=True)
            rt.start()
            self._threads.append(rt)

    def _read_loop(self, conn: socket.socket) -> None:
        conn.settimeout(0.2)
        try:
            while not self._closing.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                with self._lock:
                    self._buf.extend(chunk)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._closing.set()
        for c in list(self._conns):
            try:
                c.close()
            except OSError:
                pass
        self._conns.clear()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
            self._srv = None
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._closing.clear()

    def batches(self) -> List[int]:
        """cube_ids of every complete batch header the sink has seen."""
        with self._lock:
            text = bytes(self._buf).decode("utf-8", "replace")
        out: List[int] = []
        for ln in text.splitlines():
            if ln.startswith("# C1"):
                # "# C1 <schema> <cube_id> <specnum> ... <n_candidates>"
                toks = ln.split()
                try:
                    out.append(int(toks[3]))
                except (IndexError, ValueError):
                    pass
        return out


def _blocking_work(seconds: float) -> None:
    """Stand-in for the ~350 ms of blocking per-cube GPU work that
    ``search_compute._process_one_cube`` performs inside ``run_forever``.
    The point is not CPU burn but that the EVENT LOOP never regains
    control (no await), exactly as in production."""
    time.sleep(seconds)


@asyncio_test
async def test_threaded_sender_survives_producer_starvation() -> None:
    """REGRESSION (2026-08-06, lxd110h02 / s2g1): a producer that submits
    from the service event loop and never yields must not stop the C1
    stream.

    The old design ran the drain loop as an ``asyncio.Task`` on the
    caller's loop, so the ~350 ms of blocking per-cube work in
    ``_process_one_cube`` left it unschedulable: the 16-deep queue
    overflowed and ``submit()`` dropped every batch (47 k drops in 9 h,
    including a real injected candidate at 07:50:27). This test floods
    ``submit()`` from a coroutine containing ZERO awaits; with the
    sender on its own thread every batch must still reach C2.

    Run against the pre-fix code (``asyncio.create_task(emitter.run())``
    instead of ``start()``) this fails with batches_sent == 0 and
    batches_dropped > 0.
    """
    sink = _ThreadedSink()
    port = sink.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=2, gpu_half=1,
        queue_depth=16,  # production default -- deliberately not raised
        connect_backoff_initial_s=0.02, connect_backoff_max_s=0.05,
        send_timeout_s=5.0, idle_heartbeat_s=None,
    )
    emitter = C1TcpEmitter(config=cfg)
    emitter.start()
    try:
        # Let the sender thread connect (this is the ONLY await before
        # the flood).
        for _ in range(200):
            await asyncio.sleep(0.01)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"], dict(emitter.mon)

        n_cubes = 120
        geom = _geom()
        cands = [_cand(snr=10.0)]
        # ---- no await anywhere inside this loop ----
        for cid in range(n_cubes):
            header = _header(emitter, n_candidates=1, cube_id=cid)
            assert emitter.submit(header, cands, geom=geom) is True, (
                f"batch {cid} dropped -- sender starved; "
                f"mon={dict(emitter.mon)}"
            )
            _blocking_work(0.004)
        # -------------------------------------------

        assert emitter.mon["batches_dropped"] == 0, dict(emitter.mon)
        for _ in range(400):
            await asyncio.sleep(0.01)
            if emitter.mon["batches_sent"] >= n_cubes:
                break
        assert emitter.mon["batches_sent"] == n_cubes, dict(emitter.mon)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(sink.batches()) >= n_cubes:
                break
        assert sink.batches() == list(range(n_cubes))
    finally:
        emitter.stop()
        sink.stop()


def test_threaded_sender_reconnects_after_server_restart() -> None:
    """The threaded sender must rebuild the connection when C2 goes away
    and comes back, and resume shipping batches (same backoff lifecycle
    as the old task-based run(), now inside the thread's private loop).

    Fully synchronous test: no event loop in the test at all, which is
    the whole point of the sender owning one.
    """
    sink = _ThreadedSink()
    port = sink.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=1, gpu_half=0, queue_depth=16,
        connect_backoff_initial_s=0.02, connect_backoff_max_s=0.05,
        send_timeout_s=2.0, idle_heartbeat_s=0.2,
    )
    emitter = C1TcpEmitter(config=cfg)
    emitter.start()
    geom = _geom()
    cands = [_cand(snr=10.0)]
    try:
        for _ in range(200):
            time.sleep(0.01)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"], dict(emitter.mon)
        for cid in range(5):
            assert emitter.submit(
                _header(emitter, n_candidates=1, cube_id=cid), cands,
                geom=geom,
            ) is True
            time.sleep(0.01)
        for _ in range(200):
            time.sleep(0.01)
            if len(sink.batches()) >= 5:
                break
        assert [b for b in sink.batches() if b >= 0][:5] == list(range(5))
        reconnects_before = int(emitter.mon["reconnects"])

        # C2 dies mid-stream.
        sink.stop()
        # Keep submitting so the send path notices the dead socket; the
        # idle heartbeat would get there too, just more slowly.
        for cid in range(100, 140):
            emitter.submit(
                _header(emitter, n_candidates=1, cube_id=cid), cands,
                geom=geom,
            )
            time.sleep(0.02)
            if not emitter.mon["connected"]:
                break

        # C2 comes back on the same endpoint.
        sink2 = _ThreadedSink()
        sink2.start(port=port)
        try:
            for _ in range(400):
                time.sleep(0.01)
                if int(emitter.mon["reconnects"]) > reconnects_before:
                    break
            assert int(emitter.mon["reconnects"]) > reconnects_before, (
                f"sender never reconnected; mon={dict(emitter.mon)}"
            )
            sent_before = int(emitter.mon["batches_sent"])
            for cid in range(200, 205):
                emitter.submit(
                    _header(emitter, n_candidates=1, cube_id=cid), cands,
                    geom=geom,
                )
                time.sleep(0.01)
            for _ in range(300):
                time.sleep(0.01)
                if any(b >= 200 for b in sink2.batches()):
                    break
            assert any(b >= 200 for b in sink2.batches()), (
                f"no post-reconnect batch resumed; mon={dict(emitter.mon)}"
            )
            assert int(emitter.mon["batches_sent"]) > sent_before
        finally:
            sink2.stop()
    finally:
        emitter.stop()
        sink.stop()


def test_drop_accounting_when_server_stops_reading() -> None:
    """When C2 accepts but never reads, the sender wedges in drain(),
    the bounded queue fills, and ``submit()`` must keep returning False
    and counting drops (unchanged backpressure contract -- only the
    queue type changed, asyncio.Queue -> queue.Queue)."""
    # Tiny receive window so the socket buffers fill in a bounded number
    # of batches rather than megabytes' worth.
    sink = _ThreadedSink(hang=True, rcvbuf=2048)
    port = sink.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=0, gpu_half=0, queue_depth=4,
        connect_backoff_initial_s=0.02, connect_backoff_max_s=0.05,
        # Long send timeout: we want the sender parked in drain(), not
        # tearing the connection down and retrying.
        send_timeout_s=60.0, idle_heartbeat_s=None,
    )
    emitter = C1TcpEmitter(config=cfg)
    emitter.start()
    geom = _geom()
    cands = [_cand(snr=10.0)]
    try:
        for _ in range(200):
            time.sleep(0.01)
            if emitter.mon["connected"]:
                break
        assert emitter.mon["connected"], dict(emitter.mon)
        saw_false = False
        for cid in range(20000):
            ok = emitter.submit(
                _header(emitter, n_candidates=1, cube_id=cid), cands,
                geom=geom,
            )
            if not ok:
                saw_false = True
                break
            time.sleep(0.0005)
        assert saw_false, (
            f"submit never reported a drop against a non-reading peer; "
            f"mon={dict(emitter.mon)}"
        )
        assert int(emitter.mon["batches_dropped"]) >= 1
        assert int(emitter.mon["queue_depth"]) == cfg.queue_depth
        # And the drop counter keeps climbing while the peer stays deaf.
        before = int(emitter.mon["batches_dropped"])
        for cid in range(20000, 20010):
            emitter.submit(
                _header(emitter, n_candidates=1, cube_id=cid), cands,
                geom=geom,
            )
        assert int(emitter.mon["batches_dropped"]) > before
    finally:
        emitter.stop()
        sink.stop()


def test_stop_is_prompt_when_peer_is_dead() -> None:
    """``stop()`` must not hang the service shutdown path when the socket
    is wedged: sentinel + bounded join, thread is a daemon."""
    sink = _ThreadedSink(hang=True, rcvbuf=2048)
    port = sink.start()
    cfg = C1EmitConfig(
        host="127.0.0.1", port=port,
        search_node_id=0, gpu_half=0, queue_depth=4,
        connect_backoff_initial_s=0.02, connect_backoff_max_s=0.05,
        send_timeout_s=60.0, idle_heartbeat_s=0.2,
        stop_join_timeout_s=2.0,
    )
    emitter = C1TcpEmitter(config=cfg)
    emitter.start()
    try:
        for _ in range(200):
            time.sleep(0.01)
            if emitter.mon["connected"]:
                break
        t0 = time.monotonic()
        emitter.stop()
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"stop() took {elapsed:.2f}s"
        # Idempotent.
        emitter.stop()
    finally:
        sink.stop()


def test_start_is_idempotent() -> None:
    cfg = C1EmitConfig(
        host="127.0.0.1", port=_free_port(),
        search_node_id=0, gpu_half=0, queue_depth=4,
        connect_backoff_initial_s=0.02, connect_backoff_max_s=0.05,
    )
    emitter = C1TcpEmitter(config=cfg)
    emitter.start()
    t = emitter._thread
    emitter.start()
    assert emitter._thread is t
    emitter.stop()
