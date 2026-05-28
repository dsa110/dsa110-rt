"""Unit tests for ``dsart.services.search_ring_mon``.

Covers the M7.4 Phase 6c search-side cube_ring mon-key publisher that
fixes Bug 2 (commit 8b3f2ad-follow-up): the dashboard's "Dump Now"
button was using corr_fast's ``block_specnum_start`` as the target
event_specnum but the search-side ring keys cubes by
``slot.specnum_start`` — a different domain. This publisher exposes
the search-side ring window so cube_dump_now can pick a target that
actually lands in the live retention window.

Publisher contract:

* ``build_search_ring_mon_key(sid, g)`` returns
  ``/mon/search/<sid>/<gpu_half>/ring``.
* :meth:`SearchRingMonPublisher.publish_from_ring` snapshots a
  CubeRetentionRing-compatible object and writes the window state.
* Errors (ring.snapshot raising, etcd put_dict raising, dsautils
  missing) are logged once and swallowed; the search hot loop never
  raises through this publisher.
* Empty-ring case publishes ``n_committed=0`` with the *_specnum_*
  fields set to ``None`` so the consumer can detect "ring not yet
  primed" explicitly.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, List

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from dsart.services.search_ring_mon import (              # noqa: E402
    SearchRingMonPublisher,
    build_search_ring_mon_key,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCube:
    cube_id: int
    event_specnum_start: int
    t_det: int = 192
    n_fdm: int = 34
    n_grid: int = 256
    sample_period_specnum: int = 16
    mjd_start: float = 60948.5


class _FakeRing:
    """Minimal duck-typed ring honouring the surface
    :meth:`SearchRingMonPublisher.publish_from_ring` calls into.
    """

    def __init__(
        self,
        *,
        cubes: List[_FakeCube],
        depth: int = 16,
        t_det: int = 192,
        n_fdm: int = 34,
        n_grid: int = 256,
        sample_period_specnum: int = 16,
        snapshot_raises: Exception | None = None,
    ) -> None:
        self._cubes = list(cubes)
        self.depth = int(depth)
        self.t_det = int(t_det)
        self.n_fdm = int(n_fdm)
        self.n_grid = int(n_grid)
        self.sample_period_specnum = int(sample_period_specnum)
        self._snapshot_raises = snapshot_raises

    def snapshot(self) -> List[_FakeCube]:
        if self._snapshot_raises is not None:
            raise self._snapshot_raises
        # CubeRetentionRing.snapshot returns newest-first; mirror that.
        return list(self._cubes)


class _FakeStore:
    """In-memory put_dict capture, plus optional error injection."""

    def __init__(self, *, raise_on_put: Exception | None = None) -> None:
        self.puts: list[tuple[str, dict]] = []
        self._raise_on_put = raise_on_put

    def put_dict(self, key: str, payload: dict) -> None:
        if self._raise_on_put is not None:
            raise self._raise_on_put
        self.puts.append((str(key), dict(payload)))


# ---------------------------------------------------------------------------
# Key layout
# ---------------------------------------------------------------------------


def test_build_search_ring_mon_key_layout() -> None:
    assert build_search_ring_mon_key(1, 0) == "/mon/search/1/0/ring"
    assert build_search_ring_mon_key(13, 1) == "/mon/search/13/1/ring"


def test_build_search_ring_mon_key_int_coercion() -> None:
    # Bool / numpy ints / strings-that-look-like-ints should all coerce.
    assert build_search_ring_mon_key("9", "0") == "/mon/search/9/0/ring"
    assert build_search_ring_mon_key(True, False) == "/mon/search/1/0/ring"


def test_publisher_key_property_matches_layout() -> None:
    pub = SearchRingMonPublisher(
        search_node_id=2, gpu_half=1, store=_FakeStore(),
    )
    assert pub.key == "/mon/search/2/1/ring"


# ---------------------------------------------------------------------------
# Happy path — non-empty ring publishes the right window
# ---------------------------------------------------------------------------


def test_publish_from_nonempty_ring_writes_window() -> None:
    store = _FakeStore()
    pub = SearchRingMonPublisher(
        search_node_id=1, gpu_half=0, store=store,
    )
    # Newest is index 0 (CubeRetentionRing.snapshot iter convention).
    newest = _FakeCube(cube_id=1230, event_specnum_start=4_500_000)
    middle = _FakeCube(cube_id=1229, event_specnum_start=4_497_000)
    oldest = _FakeCube(cube_id=1215, event_specnum_start=4_470_000)
    ring = _FakeRing(cubes=[newest, middle, oldest])

    assert pub.publish_from_ring(ring) is True
    assert len(store.puts) == 1
    key, payload = store.puts[0]
    assert key == "/mon/search/1/0/ring"
    assert payload["n_committed"] == 3
    assert payload["depth"] == 16
    assert payload["t_det"] == 192
    assert payload["n_fdm"] == 34
    assert payload["n_grid"] == 256
    assert payload["newest_event_specnum_start"] == 4_500_000
    # newest_event_specnum_end = start + t_det * sample_period_specnum
    assert payload["newest_event_specnum_end"] == 4_500_000 + 192 * 16
    assert payload["oldest_event_specnum_start"] == 4_470_000
    assert payload["sample_period_specnum"] == 16
    assert payload["newest_cube_id"] == 1230
    assert payload["oldest_cube_id"] == 1215
    assert payload["newest_mjd_start"] == pytest.approx(60948.5)
    assert "ts_mono" in payload
    assert "ts_wall_unix" in payload
    assert payload["n_published"] == 1
    assert pub.n_published == 1
    assert pub.n_errors == 0


def test_publish_increments_n_published_counter() -> None:
    store = _FakeStore()
    pub = SearchRingMonPublisher(
        search_node_id=9, gpu_half=1, store=store,
    )
    ring = _FakeRing(cubes=[_FakeCube(cube_id=1, event_specnum_start=1_000)])
    for expected in (1, 2, 3, 4):
        assert pub.publish_from_ring(ring) is True
        assert store.puts[-1][1]["n_published"] == expected
    assert pub.n_published == 4
    assert pub.n_errors == 0


# ---------------------------------------------------------------------------
# Empty ring → publishes n_committed=0 with None specnums
# ---------------------------------------------------------------------------


def test_publish_empty_ring_writes_sentinel_specnums() -> None:
    store = _FakeStore()
    pub = SearchRingMonPublisher(
        search_node_id=2, gpu_half=0, store=store,
    )
    ring = _FakeRing(cubes=[], depth=16)
    assert pub.publish_from_ring(ring) is True
    payload = store.puts[0][1]
    assert payload["n_committed"] == 0
    assert payload["depth"] == 16
    assert payload["newest_event_specnum_start"] is None
    assert payload["newest_event_specnum_end"] is None
    assert payload["oldest_event_specnum_start"] is None
    assert payload["newest_cube_id"] is None
    assert payload["oldest_cube_id"] is None
    assert payload["newest_mjd_start"] is None


# ---------------------------------------------------------------------------
# Error paths — all swallowed
# ---------------------------------------------------------------------------


def test_publish_swallows_ring_snapshot_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore()
    pub = SearchRingMonPublisher(
        search_node_id=1, gpu_half=0, store=store,
    )
    ring = _FakeRing(
        cubes=[], snapshot_raises=RuntimeError("ring locked"),
    )
    with caplog.at_level(logging.WARNING, logger="dsart.services.search_ring_mon"):
        assert pub.publish_from_ring(ring) is False
    assert pub.n_errors == 1
    # The publisher must NOT have written anything when snapshot fails.
    assert store.puts == []
    # First failure is logged.
    assert any("ring.snapshot() failed" in r.getMessage() for r in caplog.records)


def test_publish_swallows_putdict_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore(raise_on_put=OSError("etcd disconnected"))
    pub = SearchRingMonPublisher(
        search_node_id=1, gpu_half=0, store=store,
    )
    ring = _FakeRing(
        cubes=[_FakeCube(cube_id=1, event_specnum_start=100_000)],
    )
    with caplog.at_level(logging.WARNING, logger="dsart.services.search_ring_mon"):
        assert pub.publish_from_ring(ring) is False
    assert pub.n_errors == 1
    # First failure is logged.
    assert any("put_dict" in r.getMessage() for r in caplog.records)


def test_subsequent_putdict_failures_are_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore(raise_on_put=OSError("etcd disconnected"))
    pub = SearchRingMonPublisher(
        search_node_id=1, gpu_half=0, store=store,
    )
    ring = _FakeRing(
        cubes=[_FakeCube(cube_id=1, event_specnum_start=100_000)],
    )
    with caplog.at_level(logging.WARNING, logger="dsart.services.search_ring_mon"):
        for _ in range(5):
            pub.publish_from_ring(ring)
    assert pub.n_errors == 5
    # Only the first failure logs at WARNING level.
    n_warn = sum(
        1 for r in caplog.records
        if r.levelno >= logging.WARNING
        and "put_dict" in r.getMessage()
    )
    assert n_warn == 1


def test_first_event_logs_info_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _FakeStore()
    pub = SearchRingMonPublisher(
        search_node_id=1, gpu_half=0, store=store,
    )
    ring = _FakeRing(
        cubes=[_FakeCube(cube_id=1, event_specnum_start=100_000)],
    )
    with caplog.at_level(logging.INFO, logger="dsart.services.search_ring_mon"):
        pub.publish_from_ring(ring)
        pub.publish_from_ring(ring)
        pub.publish_from_ring(ring)
    info_lines = [
        r for r in caplog.records
        if r.levelno == logging.INFO
        and "SearchRingMonPublisher up" in r.getMessage()
    ]
    assert len(info_lines) == 1


# ---------------------------------------------------------------------------
# Thread safety — concurrent publishes don't crash; n_published == calls
# ---------------------------------------------------------------------------


def test_concurrent_publishes_count_correctly() -> None:
    store = _FakeStore()
    pub = SearchRingMonPublisher(
        search_node_id=1, gpu_half=0, store=store,
    )
    ring = _FakeRing(
        cubes=[_FakeCube(cube_id=1, event_specnum_start=100_000)],
    )

    def worker() -> None:
        for _ in range(20):
            pub.publish_from_ring(ring)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert pub.n_published == 80
    assert pub.n_errors == 0
    assert len(store.puts) == 80
