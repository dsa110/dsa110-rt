"""Async clusterer harness — ThreadPoolExecutor wrapper for chunk 5.

Per M6 D5/D6 the clusterer runs in a single worker thread per
(search_node_id, gpu_half) process. The chunk-5 cube driver dispatches
the per-cube candidate list to this service and continues to the next
cube; the future resolves to a list of ``ClusterRecord``s when the
worker is done.

Per M6 D7/D8 the predicate / cube-dump dispatch happens AFTER the
clusterer future resolves (chunk 5 wires this).

Performance budget (M6 D5): HDBSCAN p99 ≤ 50 ms at production load
(~1000 candidates per cube). The chunk-6 ``bench/clusterer_throughput.py``
exercises this gate; if HDBSCAN exceeds the budget, the operator
flips ``ClustererConfig.backend = "dbscan"`` (sklearn DBSCAN, well-
characterised at sub-ms for the same workload).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np

from ..common.contracts import Candidate, ClusterRecord, CubeGeometry
from .forward import ClustererConfig, cluster_candidates

__all__ = [
    "ClustererService",
    "ClustererServiceResult",
]

_LOG = logging.getLogger(__name__)


class ClustererServiceResult:
    """Per-cube clusterer output (returned by the worker thread).

    Attributes:
        cube_id: matches ``CubeGeometry.cube_id``.
        labels: ``np.ndarray[len(cands)]`` int64 cluster labels (-1 = noise).
        records: ``List[ClusterRecord]``, one per cluster (incl. noise
            singletons). Order: cluster_id ≥ 0 ascending, noise singletons
            in input-list order.
        wall_ms: wall-clock duration of the clustering call (ms; for the
            chunk-6 throughput bench's p99 gate).
    """

    __slots__ = ("cube_id", "labels", "records", "wall_ms")

    def __init__(
        self,
        cube_id: int,
        labels: np.ndarray,
        records: List[ClusterRecord],
        wall_ms: float,
    ) -> None:
        self.cube_id = int(cube_id)
        self.labels = labels
        self.records = records
        self.wall_ms = float(wall_ms)


class ClustererService:
    """Async (single-thread-per-process) clusterer harness.

    Usage from chunk 5's per-cube driver:

        clusterer = ClustererService(config=ClustererConfig())
        clusterer.start()
        try:
            for slot in source:
                result = pipeline.process(slot)
                geom = build_geometry_from_slot(slot)
                future = clusterer.submit(result.candidates, geom)
                # ... do other per-cube work ...
                cluster_result = future.result(timeout=cube_period_s)
                # ... consume cluster_result.records, dispatch dumps, ...
        finally:
            clusterer.shutdown()

    Args:
        config: clusterer config; defaults to ``ClustererConfig()``.
        max_workers: ThreadPoolExecutor pool size. Default 1 (M6 D5: one
            clustering job at a time per process; the chunk-6 bench
            characterises p99 under this assumption).
    """

    def __init__(
        self,
        config: Optional[ClustererConfig] = None,
        *,
        max_workers: int = 1,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                f"max_workers={max_workers}, expected > 0"
            )
        self._config = config or ClustererConfig()
        self._max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stopping = threading.Event()
        self._n_submitted = 0
        self._n_completed = 0
        self._n_failed = 0
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Spin up the ThreadPoolExecutor."""
        if self._executor is not None:
            raise RuntimeError("ClustererService.start() called twice")
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="dsart-clusterer",
        )
        _LOG.info(
            "ClustererService up (backend=%s, mode=%s, workers=%d)",
            self._config.backend,
            self._config.feature_mode,
            self._max_workers,
        )

    def shutdown(self, *, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Drain in-flight clustering jobs and shut down the pool.

        Args:
            wait: if True, block until all in-flight jobs complete.
            timeout: max seconds to wait (None = no timeout).
        """
        self._stopping.set()
        if self._executor is None:
            return
        self._executor.shutdown(wait=wait)
        self._executor = None
        _LOG.info(
            "ClustererService down (submitted=%d completed=%d failed=%d)",
            self._n_submitted,
            self._n_completed,
            self._n_failed,
        )

    # -----------------------------------------------------------------
    # Submit
    # -----------------------------------------------------------------

    def submit(
        self,
        cands: Sequence[Candidate],
        geom: CubeGeometry,
    ) -> Future:  # type: ignore[type-arg]
        """Submit a per-cube candidate list for asynchronous clustering.

        Returns a Future that resolves to ``ClustererServiceResult``.

        Args:
            cands: per-cube candidate list (M5 detector output).
            geom: cube geometry sidecar.

        Returns:
            ``concurrent.futures.Future[ClustererServiceResult]``.

        Raises:
            RuntimeError: if start() hasn't been called or shutdown() has
                been called.
        """
        if self._executor is None:
            raise RuntimeError(
                "ClustererService.submit() requires start() first"
            )
        with self._lock:
            self._n_submitted += 1
        return self._executor.submit(self._run_one, cands, geom)

    def _run_one(
        self,
        cands: Sequence[Candidate],
        geom: CubeGeometry,
    ) -> ClustererServiceResult:
        """Worker-thread entry: time the clustering call + build the result."""
        import time

        t0 = time.perf_counter()
        try:
            labels, records = cluster_candidates(cands, geom, config=self._config)
            wall_ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self._n_completed += 1
            return ClustererServiceResult(
                cube_id=geom.cube_id,
                labels=labels,
                records=records,
                wall_ms=wall_ms,
            )
        except Exception:  # noqa: BLE001
            with self._lock:
                self._n_failed += 1
            _LOG.exception(
                "ClustererService: clustering failed for cube_id=%d (n_cands=%d)",
                geom.cube_id,
                len(cands),
            )
            raise

    # -----------------------------------------------------------------
    # Counters
    # -----------------------------------------------------------------

    @property
    def n_submitted(self) -> int:
        with self._lock:
            return self._n_submitted

    @property
    def n_completed(self) -> int:
        with self._lock:
            return self._n_completed

    @property
    def n_failed(self) -> int:
        with self._lock:
            return self._n_failed

    @property
    def config(self) -> ClustererConfig:
        return self._config
