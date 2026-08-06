"""Tests for ``dsart.dump.cube_dump`` (M6 chunk 3).

Covers the bright-pulse predicate (D8) and the cube-dump writer
thread + bounded queue (D7) per the chunk-3 test plan:

  1.  predicate happy-path
  2.  predicate rejects on SNR
  3.  predicate rejects on DM bounds (lo / hi); ``None`` disables
  4.  predicate rejects on width
  5.  predicate holdoff: same-clock retry suppressed; advancing the
      injected clock past ``holdoff_ms`` re-arms the predicate
  6.  predicate rejects on ``min_cntc``
  7.  writer happy-path round-trip via ``np.load``
  8.  writer preserves ``cluster_record`` JSON shape (asdict round-trip)
  9.  writer UDP manifest writes ``"null"`` and round-trips
  10. queue-overflow backpressure: some submits return ``False``;
      ``n_dropped`` reflects the count and the writer drains the
      accepted backlog
  11. writer accepts torch tensors (auto float16 conversion in the
      worker thread); ``importorskip`` if torch isn't installed
  12. ``stop()`` waits for in-flight dumps -> no truncated files
  13. ``np.savez`` ``OSError`` increments ``n_failed``, doesn't crash
      the worker, and submit returns successfully
  14. canonical filename ``cube_s2_g1_1024.npz`` for sid=2, g=1, spec=1024
"""

from __future__ import annotations

import os

# CRITICAL: DSART_TEST=1 must be set before importing dsart so the
# contract dataclasses' __post_init__ asserts run.
os.environ["DSART_TEST"] = "1"

import dataclasses  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dsart.common.contracts import ClusterRecord, CubeDumpManifest  # noqa: E402
from dsart.dump.cube_dump import (  # noqa: E402
    BrightPulsePredicate,
    BrightPulsePredicateConfig,
    CubeDumpWriter,
    CubeDumpWriterConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster_record(**overrides: object) -> ClusterRecord:
    base: dict = dict(
        cluster_id=0,
        cube_id=0,
        cntc=3,
        cntb_lm=2,
        cntb_dm=2,
        peak_candidate_idx=1,
        l_rad=1.5e-4 * 132,
        m_rad=1.5e-4 * 230,
        l_pix=132,
        m_pix=230,
        dm_fine_pc_cc=397.42,
        fine_dm_idx=15,
        t_in_cube=64,
        t_seconds=64 * 131.072e-6,
        width_samples=4,
        snr=20.81,
        kernel_id="unit:d1:b4",
        event_specnum=2048,
        search_node_id=2,
        gpu_half=1,
    )
    base.update(overrides)
    return ClusterRecord(**base)  # type: ignore[arg-type]


def _make_manifest(
    *,
    trigger_source: str = "auto",
    cluster_record: ClusterRecord | None = ...,  # type: ignore[assignment]
    event_specnum_start: int = 1024,
    cube_id: int = 0,
    npz_path: str = "/tmp/dump_s2_g1_1024.npz",
    sid: int = 2,
    gpu_half: int = 1,
) -> CubeDumpManifest:
    if cluster_record is ...:
        cluster_record = (
            _make_cluster_record() if trigger_source == "auto" else None
        )
    return CubeDumpManifest(
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=60942.123456789,
        t_det=4,
        n_fdm_in_cube=2,
        n_grid=8,
        trigger_source=trigger_source,
        cluster_record=cluster_record,
        npz_path=npz_path,
        search_node_id=sid,
        gpu_half=gpu_half,
    )


def _make_cube(
    *,
    t_det: int = 4,
    n_fdm: int = 2,
    n_grid: int = 8,
    fill: float = 1.5,
    dtype: str = "float16",
) -> np.ndarray:
    return np.full((t_det, n_fdm, n_grid, n_grid), fill, dtype=dtype)


class _FakeClock:
    """Manually-advanced monotonic clock for predicate holdoff tests."""

    def __init__(self, t_ms: float = 0.0) -> None:
        self.t_ms = float(t_ms)

    def __call__(self) -> float:
        return self.t_ms

    def advance(self, ms: float) -> None:
        self.t_ms += float(ms)


def _started_writer(
    tmp_path: Path,
    *,
    sid: int = 2,
    gpu_half: int = 1,
    queue_maxsize: int = 4,
) -> CubeDumpWriter:
    cfg = CubeDumpWriterConfig(
        dump_root=tmp_path,
        search_node_id=sid,
        gpu_half=gpu_half,
        queue_maxsize=queue_maxsize,
    )
    writer = CubeDumpWriter(cfg)
    writer.start()
    return writer


def _wait_for_dumped(
    writer: CubeDumpWriter, expected: int, *, timeout_s: float = 5.0
) -> None:
    """Spin-poll ``n_dumped`` to avoid timing flakes on busy CI."""
    deadline = time.monotonic() + timeout_s
    while writer.n_dumped < expected and time.monotonic() < deadline:
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# 1. Bright-pulse predicate happy-path
# ---------------------------------------------------------------------------


def test_predicate_happy_path() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=10.0,
        dm_fine_min_pc_cc=100.0,
        dm_fine_max_pc_cc=1000.0,
        width_samples_max=16,
        min_cntc=1,
        holdoff_ms=0.0,
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    record = _make_cluster_record(
        snr=20.0, dm_fine_pc_cc=400.0, width_samples=4, cntc=3
    )
    assert pred(record) is True


# ---------------------------------------------------------------------------
# 2. Predicate rejects on SNR
# ---------------------------------------------------------------------------


def test_predicate_rejects_low_snr() -> None:
    cfg = BrightPulsePredicateConfig(min_snr=10.0, holdoff_ms=0.0)
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    record = _make_cluster_record(snr=9.999)
    assert pred(record) is False
    # Predicate that just edges the threshold passes.
    record = _make_cluster_record(snr=10.0)
    assert pred(record) is True


# ---------------------------------------------------------------------------
# 3. Predicate DM bounds (with None disabling)
# ---------------------------------------------------------------------------


def test_predicate_rejects_dm_below_min() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=0.0,
        dm_fine_min_pc_cc=100.0,
        holdoff_ms=0.0,
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred(_make_cluster_record(dm_fine_pc_cc=50.0)) is False


def test_predicate_rejects_dm_above_max() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=0.0,
        dm_fine_max_pc_cc=500.0,
        holdoff_ms=0.0,
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred(_make_cluster_record(dm_fine_pc_cc=600.0)) is False


def test_predicate_dm_none_bounds_disable_dm_check() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=0.0,
        dm_fine_min_pc_cc=None,
        dm_fine_max_pc_cc=None,
        holdoff_ms=0.0,
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    # A DM of 0 (the floor of the contract) and a very high DM both pass.
    assert pred(_make_cluster_record(dm_fine_pc_cc=0.0)) is True
    pred2 = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred2(_make_cluster_record(dm_fine_pc_cc=1.0e6)) is True


# ---------------------------------------------------------------------------
# 4. Predicate rejects on width
# ---------------------------------------------------------------------------


def test_predicate_rejects_wide_width() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=0.0, width_samples_max=8, holdoff_ms=0.0
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred(_make_cluster_record(width_samples=9)) is False
    pred2 = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred2(_make_cluster_record(width_samples=8)) is True


def test_predicate_width_none_disables() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=0.0, width_samples_max=None, holdoff_ms=0.0
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred(_make_cluster_record(width_samples=1024)) is True


# ---------------------------------------------------------------------------
# 5. Predicate holdoff with fake-clock injection
# ---------------------------------------------------------------------------


def test_predicate_holdoff_squelches_then_releases() -> None:
    cfg = BrightPulsePredicateConfig(min_snr=0.0, holdoff_ms=5000.0)
    clock = _FakeClock(0.0)
    pred = BrightPulsePredicate(cfg, time_now_ms=clock)
    rec = _make_cluster_record(snr=20.0)

    assert pred(rec) is True
    # Same instant: holdoff hasn't elapsed.
    assert pred(rec) is False
    # Just before deadline.
    clock.advance(4999.0)
    assert pred(rec) is False
    # Exactly at the deadline: holdoff has elapsed (>=).
    clock.advance(1.0)
    assert pred(rec) is True
    # Re-armed; another instant repeat is squelched.
    assert pred(rec) is False


# ---------------------------------------------------------------------------
# 6. Predicate min_cntc filter
# ---------------------------------------------------------------------------


def test_predicate_rejects_low_cntc() -> None:
    cfg = BrightPulsePredicateConfig(
        min_snr=0.0, min_cntc=2, holdoff_ms=0.0
    )
    pred = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    rec = _make_cluster_record(
        cluster_id=-1, cntc=1, cntb_lm=1, cntb_dm=1, snr=20.0
    )
    assert pred(rec) is False
    rec2 = _make_cluster_record(cntc=2, cntb_lm=2, cntb_dm=2, snr=20.0)
    pred2 = BrightPulsePredicate(cfg, time_now_ms=_FakeClock(0.0))
    assert pred2(rec2) is True


# ---------------------------------------------------------------------------
# 7. Writer happy-path round-trip
# ---------------------------------------------------------------------------


def test_writer_happy_round_trip(tmp_path: Path) -> None:
    writer = _started_writer(tmp_path)
    try:
        cube = _make_cube(fill=2.5)
        manifest = _make_manifest()
        assert writer.submit(cube=cube, manifest=manifest) is True
        _wait_for_dumped(writer, 1)
    finally:
        writer.stop()

    assert writer.n_dumped == 1
    assert writer.n_dropped == 0
    assert writer.n_failed == 0

    npz_path = tmp_path / "cube_s2_g1_1024.npz"
    assert npz_path.exists()

    with np.load(npz_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["cube"], cube)
        assert data["cube"].dtype == np.float16
        # Writer-side precomputed peak_grid (consumed by the C2 plotter
        # to skip the dominant per-cube max-reduction stage).
        assert "peak_grid" in data.files, (
            "writer must precompute peak_grid for plotter fast path"
        )
        peak_grid = data["peak_grid"]
        assert peak_grid.shape == (cube.shape[0], cube.shape[1])
        assert peak_grid.dtype == np.float16
        np.testing.assert_array_equal(
            peak_grid, cube.astype(np.float16).max(axis=(2, 3)),
        )
        assert float(data["mjd_start"]) == pytest.approx(manifest.mjd_start)
        assert int(data["event_specnum_start"]) == manifest.event_specnum_start
        assert int(data["t_det"]) == manifest.t_det
        assert int(data["n_fdm_in_cube"]) == manifest.n_fdm_in_cube
        assert int(data["n_grid"]) == manifest.n_grid
        assert str(data["trigger_source"]) == "auto"
        assert int(data["search_node_id"]) == 2
        assert int(data["gpu_half"]) == 1


# ---------------------------------------------------------------------------
# 8. Writer cluster_record JSON round-trip
# ---------------------------------------------------------------------------


def test_writer_cluster_record_json_round_trip(tmp_path: Path) -> None:
    writer = _started_writer(tmp_path)
    cluster = _make_cluster_record()
    manifest = _make_manifest(cluster_record=cluster)
    try:
        writer.submit(cube=_make_cube(), manifest=manifest)
        _wait_for_dumped(writer, 1)
    finally:
        writer.stop()

    npz_path = tmp_path / "cube_s2_g1_1024.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        decoded = json.loads(str(data["cluster_record"]))

    assert decoded == dataclasses.asdict(cluster)


# ---------------------------------------------------------------------------
# 9. Writer UDP manifest -> cluster_record == 'null'
# ---------------------------------------------------------------------------


def test_writer_udp_manifest_writes_null(tmp_path: Path) -> None:
    writer = _started_writer(tmp_path)
    manifest = _make_manifest(
        trigger_source="udp",
        cluster_record=None,
        event_specnum_start=2048,
    )
    try:
        writer.submit(cube=_make_cube(), manifest=manifest)
        _wait_for_dumped(writer, 1)
    finally:
        writer.stop()

    npz_path = tmp_path / "cube_s2_g1_2048.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        # cluster_record stored as the JSON string "null"
        assert str(data["cluster_record"]) == "null"
        assert json.loads(str(data["cluster_record"])) is None
        assert str(data["trigger_source"]) == "udp"


# ---------------------------------------------------------------------------
# 10. Queue overflow backpressure
# ---------------------------------------------------------------------------


def test_writer_queue_overflow_backpressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_savez = np.savez

    def _slow_savez(*args: object, **kwargs: object) -> None:
        time.sleep(0.05)
        real_savez(*args, **kwargs)

    monkeypatch.setattr(np, "savez", _slow_savez)

    writer = _started_writer(tmp_path, queue_maxsize=2)
    cube = _make_cube()

    accepted = 0
    rejected = 0
    n_submits = 10
    try:
        for i in range(n_submits):
            manifest = _make_manifest(
                cube_id=i,
                event_specnum_start=1024 + i,
                npz_path=f"/tmp/cube_{i}.npz",
            )
            ok = writer.submit(cube=cube, manifest=manifest)
            accepted += int(ok)
            rejected += int(not ok)
    finally:
        writer.stop()

    # At least one drop with maxsize=2 + slow writer + 10 fast submits.
    assert rejected >= 1
    assert accepted + rejected == n_submits
    # Every accepted item is eventually written (no truncation).
    assert writer.n_dumped == accepted
    assert writer.n_dropped == rejected
    assert writer.n_failed == 0


# ---------------------------------------------------------------------------
# 11. Writer accepts torch tensors
# ---------------------------------------------------------------------------


def test_writer_accepts_torch_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    writer = _started_writer(tmp_path)
    try:
        cube_t = torch.full(
            (4, 2, 8, 8), 1.25, dtype=torch.float16
        )
        manifest = _make_manifest()
        assert writer.submit(cube=cube_t, manifest=manifest) is True
        _wait_for_dumped(writer, 1)
    finally:
        writer.stop()

    assert writer.n_dumped == 1
    assert writer.n_failed == 0
    npz_path = tmp_path / "cube_s2_g1_1024.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        cube = data["cube"]
        assert cube.dtype == np.float16
        assert cube.shape == (4, 2, 8, 8)
        np.testing.assert_array_equal(
            cube, np.full((4, 2, 8, 8), 1.25, dtype="float16")
        )


# ---------------------------------------------------------------------------
# 12. stop() waits for in-flight dumps
# ---------------------------------------------------------------------------


def test_stop_waits_for_in_flight_dumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_savez = np.savez

    def _slow_savez(*args: object, **kwargs: object) -> None:
        time.sleep(0.10)
        real_savez(*args, **kwargs)

    monkeypatch.setattr(np, "savez", _slow_savez)

    writer = _started_writer(tmp_path, queue_maxsize=4)
    n = 3
    cube = _make_cube()
    accepted: list[int] = []
    for i in range(n):
        manifest = _make_manifest(
            cube_id=i,
            event_specnum_start=10_000 + i,
        )
        if writer.submit(cube=cube, manifest=manifest):
            accepted.append(10_000 + i)

    # stop() must block until the worker has processed all accepted items.
    writer.stop()

    assert writer.n_dumped == len(accepted)
    for spec in accepted:
        path = tmp_path / f"cube_s2_g1_{spec}.npz"
        assert path.exists(), f"missing dump: {path}"
        # File is fully written: np.load succeeds with no truncation error.
        with np.load(path, allow_pickle=False) as data:
            assert data["cube"].shape == (4, 2, 8, 8)


# ---------------------------------------------------------------------------
# 13. OSError from np.savez bumps n_failed without crashing the worker
# ---------------------------------------------------------------------------


def test_writer_swallows_savez_oserror(tmp_path: Path) -> None:
    bad_root = tmp_path / "does_not_exist"
    cfg = CubeDumpWriterConfig(
        dump_root=bad_root,
        search_node_id=2,
        gpu_half=1,
        queue_maxsize=4,
    )
    writer = CubeDumpWriter(cfg)
    writer.start()
    try:
        manifest = _make_manifest()
        assert writer.submit(cube=_make_cube(), manifest=manifest) is True
        # Worker eventually fails the write.
        deadline = time.monotonic() + 5.0
        while writer.n_failed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert writer.n_failed == 1
        assert writer.n_dumped == 0

        # Worker is still alive — a second submit also fails-fast on
        # the same missing dir without the executor being torn down.
        manifest2 = _make_manifest(
            cube_id=1, event_specnum_start=2048
        )
        assert writer.submit(cube=_make_cube(), manifest=manifest2) is True
        deadline = time.monotonic() + 5.0
        while writer.n_failed < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert writer.n_failed == 2
    finally:
        writer.stop()


# ---------------------------------------------------------------------------
# 14. Canonical filename template
# ---------------------------------------------------------------------------


def test_writer_canonical_filename(tmp_path: Path) -> None:
    writer = _started_writer(tmp_path, sid=2, gpu_half=1)
    manifest = _make_manifest(
        sid=2, gpu_half=1, event_specnum_start=1024
    )
    try:
        writer.submit(cube=_make_cube(), manifest=manifest)
        _wait_for_dumped(writer, 1)
    finally:
        writer.stop()

    expected = tmp_path / "cube_s2_g1_1024.npz"
    assert expected.exists()
    # And no neighbours were created.
    sibling_files = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert sibling_files == ["cube_s2_g1_1024.npz"]


# ---------------------------------------------------------------------------
# Misc — guardrails on the predicate config / writer config sanity checks.
# ---------------------------------------------------------------------------


def test_predicate_rejects_invalid_min_cntc() -> None:
    with pytest.raises(ValueError, match="min_cntc"):
        BrightPulsePredicate(BrightPulsePredicateConfig(min_cntc=0))


def test_predicate_rejects_inverted_dm_band() -> None:
    cfg = BrightPulsePredicateConfig(
        dm_fine_min_pc_cc=100.0, dm_fine_max_pc_cc=50.0
    )
    with pytest.raises(ValueError, match="dm_fine"):
        BrightPulsePredicate(cfg)


def test_writer_rejects_zero_queue() -> None:
    with pytest.raises(ValueError, match="queue_maxsize"):
        CubeDumpWriter(
            CubeDumpWriterConfig(
                dump_root=Path("/tmp"),
                search_node_id=0,
                gpu_half=0,
                queue_maxsize=0,
            )
        )


def test_writer_submit_before_start_raises(tmp_path: Path) -> None:
    cfg = CubeDumpWriterConfig(
        dump_root=tmp_path,
        search_node_id=0,
        gpu_half=0,
    )
    writer = CubeDumpWriter(cfg)
    with pytest.raises(RuntimeError, match="before start"):
        writer.submit(cube=_make_cube(), manifest=_make_manifest(sid=0, gpu_half=0))


def test_writer_submit_after_stop_raises(tmp_path: Path) -> None:
    writer = _started_writer(tmp_path)
    writer.stop()
    with pytest.raises(RuntimeError, match="after stop"):
        writer.submit(cube=_make_cube(), manifest=_make_manifest())


# ---------------------------------------------------------------------------
# Queue-full displacement (2026-08-06)
# ---------------------------------------------------------------------------
#
# The queue is shallow because each slot pins a ~1.1 GiB cube, so depth is a
# memory budget and not the lever. The lever is which request loses when a
# burst arrives: a ``udp`` dump has been through C1 -> C2 coincidence and
# survived cross-node vetoes, an ``auto`` dump is one half's local predicate
# firing with no corroboration. These tests pin that ordering, and that a
# displaced dump's ring pin is released -- if it were not, the slot would stay
# pinned for the life of the process.


def _blocked_writer(tmp_path: Path, *, queue_maxsize: int = 2):
    """A writer whose drain loop is wedged, so the queue can be filled."""
    gate = threading.Event()
    writer = _started_writer(tmp_path, queue_maxsize=queue_maxsize)
    original = writer._write_one

    def _slow(cube, manifest):            # noqa: ANN001 - test shim
        gate.wait(timeout=10.0)
        return original(cube, manifest)

    writer._write_one = _slow             # type: ignore[assignment]
    return writer, gate


def test_udp_displaces_queued_auto_when_full(tmp_path: Path) -> None:
    writer, gate = _blocked_writer(tmp_path, queue_maxsize=2)
    try:
        released: list[int] = []
        # Wedge the writer on the first item, then fill every queue slot with
        # auto dumps so the next submit must contend. Every submit carries an
        # on_complete, because whichever one ends up at the head is the one
        # that gets displaced and we assert its pin is released.
        assert writer.submit(
            cube=_make_cube(), manifest=_make_manifest(cube_id=0),
            on_complete=lambda: released.append(0),
        )
        accepted = 0
        for i in range(1, 12):
            ok = writer.submit(
                cube=_make_cube(),
                manifest=_make_manifest(trigger_source="auto", cube_id=i),
                on_complete=lambda i=i: released.append(i),
            )
            accepted += int(ok)
            if not ok:
                break
        assert writer.n_dropped >= 1, "expected the queue to fill"
        drops_before = writer.n_dropped

        # A C2-confirmed dump must now displace a queued auto rather than be
        # refused, and the displaced item's ring pin must be released.
        assert writer.submit(
            cube=_make_cube(),
            manifest=_make_manifest(
                trigger_source="udp", cluster_record=None, cube_id=99,
            ),
        ) is True
        assert writer.n_displaced == 1
        # A displaced dump still did not get written, so it counts in both.
        assert writer.n_dropped == drops_before + 1
        assert released, "displaced dump must release its ring slot pin"
    finally:
        gate.set()
        writer.stop()


def test_auto_does_not_displace_anything(tmp_path: Path) -> None:
    """An auto dump is the lowest value, so it can never evict."""
    writer, gate = _blocked_writer(tmp_path, queue_maxsize=2)
    try:
        assert writer.submit(cube=_make_cube(), manifest=_make_manifest())
        while writer.submit(
            cube=_make_cube(), manifest=_make_manifest(trigger_source="auto"),
        ):
            pass
        before = writer.n_dropped
        assert writer.submit(
            cube=_make_cube(), manifest=_make_manifest(trigger_source="auto"),
        ) is False
        assert writer.n_displaced == 0
        assert writer.n_dropped == before + 1
    finally:
        gate.set()
        writer.stop()


def test_udp_does_not_displace_a_queued_udp(tmp_path: Path) -> None:
    """Equal value must not churn the queue -- first come, first served."""
    writer, gate = _blocked_writer(tmp_path, queue_maxsize=2)
    try:
        udp = dict(trigger_source="udp", cluster_record=None)
        assert writer.submit(cube=_make_cube(), manifest=_make_manifest(**udp))
        while writer.submit(cube=_make_cube(), manifest=_make_manifest(**udp)):
            pass
        before = writer.n_dropped
        assert writer.submit(
            cube=_make_cube(), manifest=_make_manifest(**udp),
        ) is False
        assert writer.n_displaced == 0
        assert writer.n_dropped == before + 1
    finally:
        gate.set()
        writer.stop()


def test_displacement_leaves_the_queue_drainable(tmp_path: Path) -> None:
    """After a displacement the survivors and the arrival all still write."""
    writer, gate = _blocked_writer(tmp_path, queue_maxsize=2)
    try:
        assert writer.submit(cube=_make_cube(), manifest=_make_manifest())
        while writer.submit(
            cube=_make_cube(), manifest=_make_manifest(trigger_source="auto"),
        ):
            pass
        writer.submit(
            cube=_make_cube(),
            manifest=_make_manifest(
                trigger_source="udp", cluster_record=None, cube_id=7,
            ),
        )
        gate.set()
        writer.stop()          # drains everything still queued
        assert writer.n_dumped >= 2
        assert writer.n_failed == 0
    finally:
        gate.set()
