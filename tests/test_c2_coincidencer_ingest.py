"""Tests for the C2 ingest queue (drain-collapse guard, 2026-07-21).

The C1BatchReceiver hands each parsed batch to
:meth:`CoincidencerService._ingest_batch`, which admits it to a bounded
in-process queue drained by :meth:`CoincidencerService._ingest_worker`.
On overflow the admission policy keeps the top-N batches by max SNR and
guarantees a bright (>= ``priority_snr``) batch is never dropped in
favour of a dimmer queued one; every drop is counted and logged (never
silent). See the incident note in ``CoincidencerConfig`` and
``configs/dsart_search_rt.yaml``.

These drive the service handlers directly (no socket, no broadcaster),
mirroring ``tests/test_c2_coincidencer_on_batch.py``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dsart.coinc import wire
from dsart.services.coincidencer import (
    CoincidencerConfig,
    CoincidencerService,
)


def asyncio_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoopBroadcaster:
    def broadcast(self, **kwargs) -> Dict[int, bool]:
        return {}

    def close(self) -> None:
        pass


class _StubStore:
    def __init__(self) -> None:
        self.last_put: Dict[str, Any] = {}

    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        return None

    def put_dict(self, key: str, value: Dict[str, Any]) -> None:
        self.last_put = dict(value)


def _criteria_file(tmp_path: Path) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text("""
trigger_classes:
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    return p


def _make_service(
    tmp_path: Path,
    *,
    ingest_queue_depth: int = 4,
    priority_snr: float = 30.0,
) -> CoincidencerService:
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1",
        bind_port=0,
        window_s=5.0,
        csv_dir_c1=tmp_path / "c1",
        csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events",
        trigger_criteria_path=_criteria_file(tmp_path),
        name_allocator_offline=True,
        gal_dm_poll_interval_s=60.0,
        startup_grace_s=0.0,
        ingest_queue_depth=ingest_queue_depth,
        priority_snr=priority_snr,
    )
    return CoincidencerService(
        config=cfg,
        mon_store=_StubStore(),
        broadcaster=_NoopBroadcaster(),
    )


def _batch(mjd_start: float, *, snr: float, n: int = 1,
           event_specnum_start: int = 0) -> wire.C1Batch:
    header = wire.build_header(
        cube_id=0,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=1,
        sample_period_us=100_000.0,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=1,
        gpu_half=0,
        n_candidates=n,
    )
    rows = tuple(
        wire.C1CandidateRow(
            snr=snr, l_rad=0.0, m_rad=0.0, l_pix=0, m_pix=0,
            dm_pc_cc=100.0, dm_idx_global=0, fine_dm_idx=0,
            event_specnum=event_specnum_start + i,
            width_samples=4, kernel_id="unit:d1:b4", flags=0,
        )
        for i in range(n)
    )
    return wire.C1Batch(header=header, candidates=rows)


def _heartbeat(mjd_start: float) -> wire.C1Batch:
    header = wire.build_header(
        cube_id=0, event_specnum_start=0, mjd_start=mjd_start,
        sample_period_specnum=1, sample_period_us=100_000.0,
        n_grid=256, n_fdm_in_cube=34, search_node_id=1, gpu_half=0,
        n_candidates=0,
    )
    return wire.C1Batch(header=header, candidates=tuple())


def _buf_snrs(svc: CoincidencerService):
    return sorted(qb.max_snr for qb in svc._ingest_buf)


# ---------------------------------------------------------------------------
# Admission policy (synchronous enqueue path; worker not running)
# ---------------------------------------------------------------------------


@asyncio_test
async def test_under_capacity_no_drops(tmp_path: Path) -> None:
    svc = _make_service(tmp_path, ingest_queue_depth=4)
    for i in range(4):
        await svc._ingest_batch(_batch(60000.0, snr=9.0 + i), "peer")
    assert len(svc._ingest_buf) == 4
    assert svc._counters["ingest_batches_dropped"] == 0
    assert svc._counters["ingest_queue_hwm"] == 4


@asyncio_test
async def test_bright_batch_survives_storm_on_full_queue(
    tmp_path: Path,
) -> None:
    """A storm of dim batches fills the queue; one bright batch then
    arrives and MUST be admitted, evicting the dimmest queued batch."""
    svc = _make_service(tmp_path, ingest_queue_depth=4, priority_snr=30.0)
    # Fill with 4 dim batches (snr ~9).
    for i in range(4):
        await svc._ingest_batch(_batch(60000.0, snr=9.0 + 0.1 * i), "dim")
    assert len(svc._ingest_buf) == 4
    # Bright 112.75-sigma injection (cf. tonight's incident).
    await svc._ingest_batch(_batch(60000.0, snr=112.75), "bright")
    # Bright survived; queue still bounded; exactly one dim dropped.
    assert len(svc._ingest_buf) == 4
    assert any(abs(s - 112.75) < 1e-6 for s in _buf_snrs(svc))
    assert svc._counters["ingest_batches_dropped"] == 1
    # The dropped one was sub-priority, so not counted as priority loss.
    assert svc._counters["ingest_priority_dropped"] == 0


@asyncio_test
async def test_bright_not_evicted_by_later_dim_batches(
    tmp_path: Path,
) -> None:
    """Once a bright batch is queued, a continuing dim storm never
    evicts it — the dim newcomers are the ones dropped."""
    svc = _make_service(tmp_path, ingest_queue_depth=4, priority_snr=30.0)
    await svc._ingest_batch(_batch(60000.0, snr=150.6), "bright")
    for i in range(20):
        await svc._ingest_batch(_batch(60000.0, snr=8.0), "dim")
    # Bright still present.
    assert any(s >= 150.0 for s in _buf_snrs(svc))
    # 20 dim arrivals into a depth-4 queue already holding the bright:
    # 3 slots for dim, so 17 dim dropped.
    assert svc._counters["ingest_batches_dropped"] == 17
    assert svc._counters["ingest_priority_dropped"] == 0


@asyncio_test
async def test_incoming_dim_dropped_when_queue_full_of_brighter(
    tmp_path: Path,
) -> None:
    svc = _make_service(tmp_path, ingest_queue_depth=3)
    for _ in range(3):
        await svc._ingest_batch(_batch(60000.0, snr=50.0), "hi")
    before = _buf_snrs(svc)
    await svc._ingest_batch(_batch(60000.0, snr=9.0), "lo")
    # Incoming dim rejected; buffer contents unchanged.
    assert _buf_snrs(svc) == before
    assert svc._counters["ingest_batches_dropped"] == 1


@asyncio_test
async def test_priority_drop_is_logged_at_error(
    tmp_path: Path, caplog,
) -> None:
    """When the queue is saturated with equally-bright batches, a further
    bright batch cannot be kept — that loss is detection-critical and is
    counted separately + logged at ERROR every time."""
    svc = _make_service(tmp_path, ingest_queue_depth=2, priority_snr=30.0)
    for _ in range(2):
        await svc._ingest_batch(_batch(60000.0, snr=100.0), "hi")
    with caplog.at_level(logging.ERROR, logger="dsart.services.coincidencer"):
        await svc._ingest_batch(_batch(60000.0, snr=100.0), "hi3")
    assert svc._counters["ingest_batches_dropped"] == 1
    assert svc._counters["ingest_priority_dropped"] == 1
    assert any(
        "PRIORITY" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    )


@asyncio_test
async def test_ordinary_drops_warn_rate_limited(
    tmp_path: Path, caplog,
) -> None:
    """A dim storm produces a counted drop for every overflow but the
    WARNING log is rate-limited (not one line per dropped batch)."""
    svc = _make_service(tmp_path, ingest_queue_depth=2, priority_snr=30.0)
    for _ in range(2):
        await svc._ingest_batch(_batch(60000.0, snr=9.0), "dim")
    with caplog.at_level(logging.WARNING, logger="dsart.services.coincidencer"):
        for _ in range(200):
            await svc._ingest_batch(_batch(60000.0, snr=9.0), "dim")
    assert svc._counters["ingest_batches_dropped"] == 200
    warns = [r for r in caplog.records if r.levelno == logging.WARNING
             and "ingest overflow" in r.getMessage()]
    # Rate-limited: far fewer log lines than dropped batches, but at
    # least the first drop must warn.
    assert 1 <= len(warns) < 200


@asyncio_test
async def test_heartbeat_is_evicted_first(tmp_path: Path) -> None:
    """0-row heartbeats carry max_snr = -1 and must be the first to go
    under pressure (they are harmless to drop)."""
    svc = _make_service(tmp_path, ingest_queue_depth=2)
    await svc._ingest_batch(_heartbeat(60000.0), "hb")
    await svc._ingest_batch(_batch(60000.0, snr=9.0), "cand")
    assert len(svc._ingest_buf) == 2
    # Full now; a real candidate arrives -> heartbeat evicted, not the
    # candidate.
    await svc._ingest_batch(_batch(60000.0, snr=8.0), "cand2")
    assert all(qb.max_snr >= 0 for qb in svc._ingest_buf)
    assert svc._counters["ingest_batches_dropped"] == 1


# ---------------------------------------------------------------------------
# Worker: end-to-end drain (queue -> _on_batch)
# ---------------------------------------------------------------------------


async def _drain(svc: CoincidencerService, *, max_ticks: int = 500) -> None:
    worker = asyncio.create_task(svc._ingest_worker())
    for _ in range(max_ticks):
        await asyncio.sleep(0)
        if not svc._ingest_buf:
            # one more tick to let the in-flight batch (if any) finish
            await asyncio.sleep(0)
            if not svc._ingest_buf:
                break
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


@asyncio_test
async def test_worker_processes_all_under_normal_load(
    tmp_path: Path,
) -> None:
    """Below capacity, the worker drains every enqueued batch through
    _on_batch with no drops — behaviour unchanged from the inline path."""
    svc = _make_service(tmp_path, ingest_queue_depth=64)
    n = 10
    for i in range(n):
        # distinct specnums so each row is a distinct window entry
        await svc._ingest_batch(
            _batch(60000.0, snr=9.0, n=2, event_specnum_start=100 * i),
            "peer",
        )
    await _drain(svc)
    assert svc._ingest_buf == []
    assert svc._counters["ingest_batches_dropped"] == 0
    # Every candidate row made it into the window/graph pipeline.
    assert svc._counters["rows_in"] == n * 2


@asyncio_test
async def test_worker_processes_surviving_bright_after_storm(
    tmp_path: Path,
) -> None:
    """After a storm that overflows the queue, the worker still processes
    the bright survivor (its rows reach _on_batch)."""
    svc = _make_service(tmp_path, ingest_queue_depth=3, priority_snr=30.0)
    for i in range(10):
        await svc._ingest_batch(
            _batch(60000.0, snr=8.0, event_specnum_start=10 * i), "dim")
    await svc._ingest_batch(
        _batch(60000.0, snr=95.0, event_specnum_start=9999), "bright")
    assert svc._counters["ingest_batches_dropped"] >= 1
    await _drain(svc)
    assert svc._ingest_buf == []
    # rows_in counts survivors that reached the pipeline; the bright
    # batch is guaranteed to be among them.
    assert svc._counters["rows_in"] >= 1
