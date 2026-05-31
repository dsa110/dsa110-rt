"""Cube-dump writer thread + bright-pulse predicate (M6 chunk 3).

Implements the two surfaces of the per-(search_node, gpu_half) cube
dumper described in ``M6_PLAN_FIXES.md`` D7/D8:

  1. ``BrightPulsePredicate`` — auto-trigger gate. Evaluates a
     ``ClusterRecord`` against a configurable predicate (SNR floor,
     fine-DM band, width ceiling, cluster-cardinality floor, plus a
     per-process holdoff) and returns ``True`` iff the cluster should
     fire a cube dump. Holdoff state is advanced as a side effect on
     a ``True`` return so consecutive bursts within ``holdoff_ms`` are
     squelched.

  2. ``CubeDumpWriter`` — single-worker writer thread fed by a bounded
     ``queue.Queue(maxsize=4)``. Each accepted submit pops in the
     drain loop and writes a ``[T_det, N_fine_DM, N_grid, N_grid]``
     float16 cube as an NPZ archive at
     ``${DUMP_ROOT}/cube_s${sid}_g${g}_${event_specnum_start}.npz``
     with a sidecar manifest. When the queue is full ``put_nowait``
     raises and the dispatch path logs a WARNING, increments the
     ``n_dropped`` counter, and returns ``False`` — the real-time hot
     path is non-blocking.

The auto-trigger path fires on the CURRENT cube (the GPU cube tensor
is still in scope right after clustering); the UDP path (chunk 4
listener, chunk 5 wiring) flips a one-shot flag that fires on the
NEXT cube. This module exposes both shapes via the same ``submit``
API: chunk 5 just builds an ``udp`` ``CubeDumpManifest`` (with
``cluster_record=None``) when consuming the flag.

NPZ schema (per D7):
    cube                  -> [T_det, N_fdm, N_grid, N_grid] float16
    mjd_start             -> 0-d float64
    event_specnum_start   -> 0-d int64
    t_det                 -> 0-d int32
    n_fdm_in_cube         -> 0-d int32
    n_grid                -> 0-d int32
    cluster_record        -> 0-d 'U' (json-encoded ``asdict``; ``'null'`` for udp)
    trigger_source        -> 0-d 'U'  ('auto' | 'udp')
    search_node_id        -> 0-d int32
    gpu_half              -> 0-d int32

Compatible with ``np.load(path, allow_pickle=False)`` because no field
is an object array (the JSON cluster record is stored as a unicode
0-d ndarray, mirroring ``DmPlan.to_npz``).
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import json
import logging
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Optional, Tuple

import numpy as np

from ..common.contracts import ClusterRecord, CubeDumpManifest
from ._clock import monotonic_ms

__all__ = [
    "BrightPulsePredicateConfig",
    "BrightPulsePredicate",
    "CubeDumpWriterConfig",
    "CubeDumpWriter",
]


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bright-pulse predicate (D8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BrightPulsePredicateConfig:
    """Configurable thresholds for the auto-trigger predicate (M6 D8).

    A cluster fires the cube dump iff ALL conditions hold:

      * ``snr >= min_snr``
      * ``dm_fine_min_pc_cc <= dm_fine_pc_cc <= dm_fine_max_pc_cc``
        (each ``None`` bound disables that side)
      * ``width_samples <= width_samples_max`` (``None`` disables)
      * ``cntc >= min_cntc`` (default 1 accepts singletons)
      * ``now_ms - last_dump_ms >= holdoff_ms`` (per-process)

    Args:
        min_snr: peak-candidate SNR floor in sigma. Default 10.0.
        dm_fine_min_pc_cc: lower fine-DM bound in pc cm^-3, inclusive.
            ``None`` disables the lower bound.
        dm_fine_max_pc_cc: upper fine-DM bound in pc cm^-3, inclusive.
            ``None`` disables the upper bound.
        width_samples_max: matched-filter width ceiling in samples,
            inclusive. ``None`` disables.
        min_cntc: cluster-cardinality floor (>= 1).
        holdoff_ms: per-process minimum gap between successful auto
            triggers in milliseconds. Bursts are rare; this prevents a
            sidelobe run from firing dozens of NPZs.
    """

    min_snr: float = 10.0
    dm_fine_min_pc_cc: Optional[float] = None
    dm_fine_max_pc_cc: Optional[float] = None
    width_samples_max: Optional[int] = None
    min_cntc: int = 1
    holdoff_ms: float = 5000.0


class BrightPulsePredicate:
    """Stateful evaluator for the bright-pulse auto-trigger gate.

    Holds the timestamp of the last successful dispatch; on each call
    re-evaluates the configurable thresholds and the holdoff window. A
    ``True`` return advances the holdoff clock; ``False`` does not, so
    consecutive failures don't move the deadline.

    The instance is intended to be owned by the per-(search_node,
    gpu_half) clusterer dispatch path (chunk 5 wiring). It is NOT
    thread-safe — use one instance per dispatch thread.

    Args:
        config: locked-in predicate thresholds.
        time_now_ms: callable returning the current monotonic time in
            milliseconds. Defaults to ``_clock.monotonic_ms``. Tests
            inject a fake clock to advance time without sleeping.
    """

    __slots__ = ("_config", "_time_now_ms", "_last_dispatch_ms")

    def __init__(
        self,
        config: BrightPulsePredicateConfig,
        *,
        time_now_ms: Callable[[], float] = monotonic_ms,
    ) -> None:
        if config.min_cntc < 1:
            raise ValueError(
                f"min_cntc={config.min_cntc}, expected >= 1"
            )
        if config.holdoff_ms < 0.0:
            raise ValueError(
                f"holdoff_ms={config.holdoff_ms}, expected >= 0"
            )
        if (
            config.dm_fine_min_pc_cc is not None
            and config.dm_fine_max_pc_cc is not None
            and config.dm_fine_min_pc_cc > config.dm_fine_max_pc_cc
        ):
            raise ValueError(
                f"dm_fine_min_pc_cc={config.dm_fine_min_pc_cc} > "
                f"dm_fine_max_pc_cc={config.dm_fine_max_pc_cc}"
            )
        self._config = config
        self._time_now_ms = time_now_ms
        self._last_dispatch_ms: Optional[float] = None

    def __call__(self, record: ClusterRecord) -> bool:
        """Return True iff ``record`` fires the auto-trigger predicate.

        Side effect: a ``True`` return advances the holdoff clock so
        the next predicate-passing record within ``holdoff_ms`` returns
        ``False``.

        Args:
            record: peak ``ClusterRecord`` from the per-cube clusterer.

        Returns:
            ``True`` iff every threshold passes AND the holdoff window
            has elapsed.
        """
        cfg = self._config
        if record.snr < cfg.min_snr:
            return False
        if (
            cfg.dm_fine_min_pc_cc is not None
            and record.dm_fine_pc_cc < cfg.dm_fine_min_pc_cc
        ):
            return False
        if (
            cfg.dm_fine_max_pc_cc is not None
            and record.dm_fine_pc_cc > cfg.dm_fine_max_pc_cc
        ):
            return False
        if (
            cfg.width_samples_max is not None
            and record.width_samples > cfg.width_samples_max
        ):
            return False
        if record.cntc < cfg.min_cntc:
            return False
        now_ms = self._time_now_ms()
        if (
            self._last_dispatch_ms is not None
            and (now_ms - self._last_dispatch_ms) < cfg.holdoff_ms
        ):
            return False
        self._last_dispatch_ms = now_ms
        return True

    @property
    def last_dispatch_ms(self) -> Optional[float]:
        """Timestamp (ms) of the last True return, or None if never fired."""
        return self._last_dispatch_ms


# ---------------------------------------------------------------------------
# Cube-dump writer thread (D7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubeDumpWriterConfig:
    """Configuration for ``CubeDumpWriter`` (M6 D7).

    Args:
        dump_root: directory under which NPZ dumps are written. Files
            land at ``${dump_root}/cube_s${sid}_g${g}_${event_specnum_start}.npz``.
            The directory is NOT auto-created — production wiring sets
            this from ``DSART_M6_CUBE_DUMP_DIR`` (or its default
            ``${REPO_ROOT}/bench/reports/M6/cube_dump/``) and the
            milestone bring-up provisions it.
        search_node_id: 0..N_SEARCH-1 of the dumping process.
        gpu_half: 0..N_SEARCH_GPU-1 of the dumping process.
        queue_maxsize: bounded queue depth between the dispatch path
            and the writer thread. Default 4 per D7. ``put_nowait`` on
            a full queue logs WARNING + increments ``n_dropped``.
    """

    dump_root: Path
    search_node_id: int
    gpu_half: int
    queue_maxsize: int = 4


# Sentinel pushed onto the queue by ``stop()`` to terminate the drain
# loop. Defined at module scope so it has a stable identity even if
# ``CubeDumpWriter`` is reloaded.
_DRAIN_SENTINEL = object()


# Default ring-buffer depth for ``CubeDumpWriter`` writer-time history.
# 256 samples is enough to smooth a few seconds of bursty production
# dumps for the cube-dump-e2e bench's writer_p99_ms metric while keeping
# the per-instance memory footprint bounded (256 × 8B = 2 KiB).
_DEFAULT_RECENT_WRITE_MS_CAPACITY: int = 256


class CubeDumpWriter:
    """Single-worker NPZ cube-dump writer thread (M6 D7).

    Uses one ``concurrent.futures.ThreadPoolExecutor(max_workers=1)``
    per ``CubeDumpWriter`` instance, fed by a bounded
    ``queue.Queue(maxsize=queue_maxsize)``. The dispatch path calls
    ``submit`` with the cube tensor and a fully-built
    ``CubeDumpManifest``; ``submit`` pushes onto the queue with
    ``put_nowait`` and returns ``True`` on success or ``False`` if the
    queue is full (with a WARNING log and ``n_dropped`` increment).

    The drain loop:

      1. Pops ``(cube, manifest)``.
      2. Converts the cube to ``np.ndarray`` of dtype float16 if it
         arrived as a torch tensor or in another dtype.
      3. ``np.savez`` to the canonical path.
      4. Increments ``n_dumped`` on success or ``n_failed`` on
         ``OSError`` (logged at ERROR; the worker thread does not
         re-raise into the dispatch path).

    The torch -> numpy conversion intentionally happens IN the writer
    thread, not in the dispatch path, so the real-time GPU pipeline
    doesn't pay the ``.cpu()`` synchronization cost.

    Counters (``n_dumped``, ``n_dropped``, ``n_failed``,
    ``queue_depth``) are read-only properties safe to inspect from
    the operator process.

    Lifecycle: instantiate, call ``start()`` once, call ``submit()``
    repeatedly from the dispatch thread, then call ``stop()`` once at
    shutdown to drain pending dumps and shut the executor down.
    """

    def __init__(
        self,
        config: CubeDumpWriterConfig,
        *,
        on_dump_complete: Optional[
            Callable[[Path, CubeDumpManifest], None]
        ] = None,
    ) -> None:
        if config.queue_maxsize <= 0:
            raise ValueError(
                f"queue_maxsize={config.queue_maxsize}, expected > 0"
            )
        self._config = config
        # M7.4 cube-uploader hook: invoked from the writer thread AFTER
        # a successful ``np.savez`` and the ``n_dumped`` increment, with
        # the on-disk path that was actually written plus the manifest.
        # The hook is wrapped in try/except inside ``_drain_loop`` so
        # it can never poison the writer thread. Production binds it
        # to :func:`dsart.coinc.cube_uploader.upload_event_cubes` via
        # ``services.search_compute.SearchComputeService.start``.
        self._on_dump_complete = on_dump_complete
        self._queue: queue.Queue[Any] = queue.Queue(
            maxsize=config.queue_maxsize
        )
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._drain_future: Optional[concurrent.futures.Future] = None
        self._started = False
        self._stopped = False
        self._n_dumped = 0
        self._n_dropped = 0
        self._n_failed = 0
        # M7.4 cube-uploader hook bookkeeping: count successful hook
        # invocations and exception swallows so operators can sanity-
        # check the rsync wiring without grovelling through logs.
        self._n_hook_ok = 0
        self._n_hook_failed = 0
        # Writer-time ring buffer. Each entry is the wall-clock duration
        # of a single ``np.savez`` call in milliseconds. Sized small so
        # the per-instance footprint is negligible (~2 KiB at the
        # default capacity); the cube-dump-e2e bench uses this to
        # compute ``writer_p99_ms``. Mutated only from the writer
        # thread; reads from the dispatch / operator thread observe a
        # consistent (deque-internal-locked) view.
        self._recent_write_ms: Deque[float] = collections.deque(
            maxlen=_DEFAULT_RECENT_WRITE_MS_CAPACITY
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the single-worker executor and the drain loop."""
        if self._started:
            return
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=(
                f"cube-dump-s{self._config.search_node_id}"
                f"-g{self._config.gpu_half}"
            ),
        )
        self._drain_future = self._executor.submit(self._drain_loop)
        self._started = True

    def stop(self) -> None:
        """Drain the queue and shut the executor down.

        Pushes a sentinel onto the queue (blocking ``put`` so it lands
        even if the queue is currently full — the drain loop will pop
        items in front of it first, eventually clearing space) and
        joins the worker. After ``stop()`` returns, all accepted dumps
        have been written and no truncated files remain.
        """
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._queue.put(_DRAIN_SENTINEL)
        assert self._executor is not None
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        cube: Any,
        manifest: CubeDumpManifest,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Enqueue a cube for asynchronous NPZ dump.

        Args:
            cube: cube tensor with shape
                ``[T_det, n_fdm_in_cube, n_grid, n_grid]``. Accepted
                as ``np.ndarray`` (any real dtype; cast to float16 in
                the writer thread) or as a torch tensor (same; the
                ``.detach().cpu().numpy()`` round-trip happens in the
                writer thread, not on the dispatch hot path).
            manifest: fully-validated ``CubeDumpManifest``. The
                writer thread builds the canonical NPZ path from the
                manifest's ``event_specnum_start`` and the writer's
                ``(search_node_id, gpu_half)`` config; the manifest's
                ``npz_path`` field is informational.

        Returns:
            ``True`` if the cube was accepted onto the queue;
            ``False`` if the queue was full. ``False`` returns log
            ``cube_dump_dropped: queue full (cube_id=%d)`` at WARNING
            and increment ``n_dropped``.

        Raises:
            RuntimeError: if called before ``start()`` or after
                ``stop()``.
        """
        if not self._started:
            raise RuntimeError("CubeDumpWriter.submit() before start()")
        if self._stopped:
            raise RuntimeError("CubeDumpWriter.submit() after stop()")
        try:
            self._queue.put_nowait((cube, manifest, on_complete))
            return True
        except queue.Full:
            _log.warning(
                "cube_dump_dropped: queue full (cube_id=%d)",
                manifest.cube_id,
            )
            self._n_dropped += 1
            return False

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    @property
    def n_dumped(self) -> int:
        """Number of NPZ dumps successfully written."""
        return self._n_dumped

    @property
    def n_dropped(self) -> int:
        """Number of submits dropped due to queue-full backpressure."""
        return self._n_dropped

    @property
    def n_failed(self) -> int:
        """Number of dumps that raised ``OSError`` in the writer thread."""
        return self._n_failed

    @property
    def n_hook_ok(self) -> int:
        """Number of times the ``on_dump_complete`` hook ran to completion."""
        return self._n_hook_ok

    @property
    def n_hook_failed(self) -> int:
        """Number of times the ``on_dump_complete`` hook raised (swallowed)."""
        return self._n_hook_failed

    @property
    def queue_depth(self) -> int:
        """Approximate current depth of the writer queue.

        ``Queue.qsize`` is documented as approximate (not reliable for
        synchronization decisions) but is fine for operator metrics.
        """
        return self._queue.qsize()

    @property
    def recent_write_ms_ms(self) -> Tuple[float, ...]:
        """Recent ``np.savez`` wall-clock durations (ms), oldest first.

        Snapshot of the writer-thread's ring buffer (capacity = 256 by
        default). Empty until the worker has finished at least one
        write. Used by ``bench/cube_dump_e2e.py`` to compute the
        ``writer_p50_ms`` / ``writer_p99_ms`` / ``writer_max_ms``
        operator-facing metrics without exposing the dispatch hot path
        to the timing instrumentation.

        The returned tuple is a snapshot; concurrent worker writes
        produce a new ring-buffer state but never mutate the returned
        sequence.
        """
        return tuple(self._recent_write_ms)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        """Pop-and-write loop, runs on the executor's single worker."""
        while True:
            item = self._queue.get()
            try:
                if item is _DRAIN_SENTINEL:
                    return
                cube, manifest, on_complete = item
                try:
                    path = self._write_one(cube, manifest)
                    self._n_dumped += 1
                    self._invoke_on_dump_complete(path, manifest)
                except OSError as exc:
                    _log.error(
                        "cube_dump write failed (cube_id=%d, "
                        "event_specnum_start=%d): %s",
                        manifest.cube_id,
                        manifest.event_specnum_start,
                        exc,
                    )
                    self._n_failed += 1
                except Exception as exc:  # pragma: no cover - defensive
                    _log.exception(
                        "cube_dump unexpected error (cube_id=%d): %s",
                        manifest.cube_id,
                        exc,
                    )
                    self._n_failed += 1
                finally:
                    # Release the ring slot-pin (if any) REGARDLESS of
                    # write success, so a failed/short dump can never
                    # permanently wedge a ring buffer out of reuse.
                    if on_complete is not None:
                        try:
                            on_complete()
                        except Exception:  # noqa: BLE001 - never poison worker
                            _log.exception(
                                "cube_dump on_complete (slot release) failed "
                                "(cube_id=%d)", manifest.cube_id,
                            )
            finally:
                self._queue.task_done()

    def _invoke_on_dump_complete(
        self, path: Path, manifest: CubeDumpManifest,
    ) -> None:
        """Fire the optional post-write hook, swallowing any error.

        Counts successes / failures so an operator can confirm the hook
        ran without grepping logs. The wrapper is private because the
        writer thread is the only legitimate caller (the FD-level
        write has already returned and ``n_dumped`` is incremented).
        """
        cb = self._on_dump_complete
        if cb is None:
            return
        try:
            cb(path, manifest)
            self._n_hook_ok += 1
        except Exception as exc:  # noqa: BLE001 — never poison the worker
            _log.warning(
                "cube_dump on_dump_complete hook failed "
                "(cube_id=%d, path=%s): %r",
                manifest.cube_id, path, exc,
            )
            self._n_hook_failed += 1

    def _write_one(
        self,
        cube: Any,
        manifest: CubeDumpManifest,
    ) -> Path:
        """Convert + persist a single (cube, manifest) pair to NPZ.

        Times the inner ``np.savez`` call with ``time.perf_counter`` and
        appends the duration (ms) to the ``recent_write_ms_ms`` ring
        buffer. The cube-coerce step is deliberately *not* timed: it
        runs in the writer thread for the same dispatch-hot-path
        reason as the np.savez call, but its cost is tied to torch
        sync rather than disk IO and would muddy the operator-facing
        writer-time metric.

        M7.4: when ``manifest.npz_path`` points at an absolute
        location (e.g. the C2 trigger listener routed the dump into a
        per-event subdir), honour it directly instead of using the
        writer's canonical ``${dump_root}/cube_s..npz`` template. The
        M6 legacy ``auto`` / ``udp`` paths still leave ``npz_path``
        informational (matches the writer's canonical layout) so this
        is a strict superset of the historical behaviour.
        """
        cube_np = _coerce_cube_to_float16_ndarray(cube)
        path = self._resolve_path(manifest)
        cluster_record_json = _cluster_record_to_json(manifest.cluster_record)
        # Precompute the (n_fdm, n_t) peak grid that the C2 plotter
        # otherwise rebuilds by streaming the full ~855 MB cube on h23.
        # Doing it here — once, on the writer thread, with the cube
        # already in L3 from the float16 coerce — turns the plotter's
        # dominant 8 × ~9 s reduction into 8 × <50 ms .npz lookups.
        # The reduction itself is fp16 numpy, ~1–2 s for a 4D cube of
        # shape (n_fdm, n_t, n_grid, n_grid); the np.savez stage stays
        # the writer-thread budget bottleneck.
        if cube_np.ndim == 4 and cube_np.size > 0:
            peak_grid = cube_np.max(axis=(2, 3))
        else:
            peak_grid = np.zeros((0, 0), dtype=cube_np.dtype)
        t_savez_start = time.perf_counter()
        np.savez(
            str(path),
            cube=cube_np,
            peak_grid=peak_grid,
            mjd_start=np.asarray(manifest.mjd_start, dtype="float64"),
            event_specnum_start=np.asarray(
                manifest.event_specnum_start, dtype="int64"
            ),
            t_det=np.asarray(manifest.t_det, dtype="int32"),
            n_fdm_in_cube=np.asarray(manifest.n_fdm_in_cube, dtype="int32"),
            n_grid=np.asarray(manifest.n_grid, dtype="int32"),
            cluster_record=np.asarray(cluster_record_json, dtype="U"),
            trigger_source=np.asarray(manifest.trigger_source, dtype="U"),
            search_node_id=np.asarray(
                self._config.search_node_id, dtype="int32"
            ),
            gpu_half=np.asarray(self._config.gpu_half, dtype="int32"),
        )
        write_ms = (time.perf_counter() - t_savez_start) * 1.0e3
        # ``collections.deque.append`` is documented thread-safe in
        # CPython (single-element atomic op); reads via ``tuple(deque)``
        # snapshot a consistent state. No explicit lock needed.
        self._recent_write_ms.append(write_ms)
        return path

    def _compose_path(self, manifest: CubeDumpManifest) -> Path:
        """Build the canonical NPZ path for ``manifest``."""
        sid = self._config.search_node_id
        g = self._config.gpu_half
        return self._config.dump_root / (
            f"cube_s{sid}_g{g}_{manifest.event_specnum_start}.npz"
        )

    def _resolve_path(self, manifest: CubeDumpManifest) -> Path:
        """Resolve the on-disk path for ``manifest``.

        Default behaviour (M6): ignore ``manifest.npz_path`` entirely
        — the writer composes ``${dump_root}/cube_s${sid}_g${g}_${ev}.npz``
        from its own config. The manifest's ``npz_path`` is a mirror of
        that canonical layout, intentionally informational so legacy
        dispatch callers don't have to know the writer's dump_root.

        M7.4 C2 trigger extension: when ``manifest.npz_path`` points at
        a *subdirectory* of the writer's ``dump_root`` (i.e. the C2
        trigger listener routed the dump into
        ``${dump_root}/<event_name>/cube_s..._g..._N.npz``), honour the
        override and auto-create the per-event subdir. Anything outside
        the writer's dump_root is treated as a stale informational
        placeholder and ignored — preserves the M6 contract for the
        existing dispatch callers + their tests.
        """
        canonical = self._compose_path(manifest)
        if not manifest.npz_path:
            return canonical
        candidate = Path(manifest.npz_path)
        dump_root = Path(self._config.dump_root).resolve()
        try:
            resolved_parent = candidate.resolve().parent
        except (OSError, RuntimeError):
            return canonical
        # Only honour the override when its parent is a *strict* subdir
        # of dump_root (depth ≥ 1 under dump_root). Same-parent equality
        # falls back to canonical so the M6 legacy callers see the
        # historical filename.
        try:
            rel = resolved_parent.relative_to(dump_root)
        except ValueError:
            return canonical
        if rel == Path("."):
            return canonical
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log.warning(
                "cube_dump: failed to create event dir %s: %r; "
                "falling back to canonical path %s",
                candidate.parent, exc, canonical,
            )
            return canonical
        return candidate


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _coerce_cube_to_float16_ndarray(cube: Any) -> np.ndarray:
    """Return ``cube`` as a ``np.ndarray`` of dtype float16.

    Accepts a ``np.ndarray`` directly or duck-types a torch tensor via
    ``detach().cpu().numpy()``. Conversion is best-effort: any object
    that surfaces ``.detach``, ``.cpu``, and ``.numpy`` works without
    importing torch (the test host has torch, the worktree host does
    not).
    """
    if isinstance(cube, np.ndarray):
        if cube.dtype == np.float16:
            return cube
        return cube.astype(np.float16, copy=False)
    if (
        hasattr(cube, "detach")
        and hasattr(cube, "cpu")
        and hasattr(cube, "numpy")
    ):
        # Torch tensor (or compatible duck-type). The .cpu() call is
        # the synchronization point we deliberately keep off the
        # dispatch hot path.
        arr = cube.detach().cpu().numpy()
        if arr.dtype == np.float16:
            return arr
        return arr.astype(np.float16, copy=False)
    raise TypeError(
        f"cube must be np.ndarray or torch tensor; got {type(cube).__name__}"
    )


def _cluster_record_to_json(record: Optional[ClusterRecord]) -> str:
    """Serialise a ``ClusterRecord`` (or ``None``) for the NPZ sidecar.

    UDP dumps carry no cluster metadata, so the field stores
    ``json.dumps(None)`` (``"null"``) per D7. Auto dumps carry the
    peak cluster's full dataclass, serialised via
    ``dataclasses.asdict`` -> ``json.dumps``.
    """
    if record is None:
        return json.dumps(None)
    return json.dumps(dataclasses.asdict(record))
