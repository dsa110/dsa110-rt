"""Proactive cube staging for bright candidates (search-side).

Background (measured live 2026-07-21)
-------------------------------------
C2-triggered cube dumps (:mod:`dsart.dump.c2_trigger_listener`) can only
dump a cube while it is still inside the search node's CPU retention ring
(:class:`dsart.services.cube_pipeline.CubeRetentionRing`). At the
production op-point that window is ``cube_ring_depth`` cubes
(``depth=12`` ⇒ ~2.4 s, or ~6.5 s at earlier depths). The end-to-end
burst→C2-request latency measured that night was 7.8-9 s (detect ~1.3 s
+ cube pipeline + C1 batch emit + C2 ingest + cluster eval + broadcast),
which is *structurally* longer than the ring window. Three consecutive
bright triggers were lost as ``too_late`` misses — including
``260721upyy`` at **101.8σ**, missed by ~1.3 s.

The search node ALREADY has the detection cube locally the moment the
detector fires — long before C2 can name an event. This module stages
that cube to disk *proactively* when a cube's peak candidate SNR clears
a (high) threshold, under a provisional ``pending_g<g>_<specnum>/`` key
(there is no event name yet). When the C2 trigger for that specnum later
arrives too late for the live ring, the listener *claims* the staged
copy — renaming it into the real ``<event_name>/`` archive and firing
the existing uploader — converting the miss into a successful dump.

Design (see ``bright-stage`` branch notes)
------------------------------------------
* **Trigger.** ``maybe_stage`` fires iff the enabled flag is set and the
  cube's peak candidate SNR ``>= snr_threshold`` (default 50.0). The
  actual NPZ write is handed to the shared :class:`CubeDumpWriter`
  (bounded queue, single writer thread) exactly like the udp/auto dump
  paths — the hot loop never blocks on IO.
* **Rate protection.** A per-half LRU budget of ``max_pending`` staged
  windows (oldest evicted, its ``pending`` dir removed) plus a
  ``min_interval_s`` floor keep an RFI storm from filling the disk.
  Unclaimed pending dirs are garbage-collected after ``ttl_s``.
* **Claim.** ``claim(event_specnum, event_name)`` finds the pending
  window covering ``event_specnum``, renames this half's NPZ into
  ``<dump_root>/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz``,
  fires the uploader, and forgets the entry (so repeated claims are
  idempotent no-ops).
* **Live-ring precedence.** On a C2 *hit* (cube still in the ring) the
  listener takes the live-ring dump path unchanged and calls
  ``drop_pending`` so the now-redundant staged copy is discarded.

The pending dir carries the ``gpu_half`` in its name
(``pending_g<g>_<specnum>``) so the two halves never share a provisional
dir — eviction/GC can remove a whole dir without racing the other half
(the shared *event* archive dir is per-half-file-scoped as before).

This module holds no torch/CUDA dependency; it operates on the cube
tensor object opaquely (the writer thread does the fp16 coerce).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..common.contracts import CubeDumpManifest
from .cube_dump import CubeDumpWriter

_LOG = logging.getLogger("dsart.dump.proactive_stager")

__all__ = [
    "ProactiveStagerConfig",
    "ProactiveCubeStager",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProactiveStagerConfig:
    """Thresholds + rate limits for :class:`ProactiveCubeStager`.

    Args:
        enabled: master gate. When ``False`` every ``maybe_stage`` is a
            cheap no-op (the C2 too_late path still runs unchanged).
        snr_threshold: per-cube peak-candidate SNR floor (σ) that arms a
            proactive stage. Default 50.0 — comfortably above the C1
            emit floor (``snr_min≈11``) and the RFI singles floor, so
            only genuinely bright cubes stage. Tonight's missed
            ``260721upyy`` peaked at 101.8σ, ~2× this.
        max_pending: LRU budget of concurrently staged (unclaimed)
            windows per gpu_half. Oldest is evicted (its pending dir
            removed) when a new stage would exceed it. Bounds worst-case
            staged footprint at ``max_pending`` cubes (~1.1 GiB each).
        min_interval_s: minimum wall gap between two proactive stages on
            one half; a burst/RFI run staging back-to-back is throttled.
        ttl_s: unclaimed pending dirs older than this are garbage
            collected (opportunistically, on each ``maybe_stage``).
    """

    enabled: bool = True
    snr_threshold: float = 50.0
    max_pending: int = 4
    min_interval_s: float = 2.0
    ttl_s: float = 600.0


# ---------------------------------------------------------------------------
# Pending bookkeeping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PendingEntry:
    """One staged, not-yet-claimed cube window."""

    specnum_start: int
    specnum_end_excl: int
    cube_id: int
    pending_dir: Path
    npz_path: Path
    staged_s: float


# ---------------------------------------------------------------------------
# Stager
# ---------------------------------------------------------------------------


class ProactiveCubeStager:
    """Search-side proactive cube stager + claim registry.

    Owned per ``(search_node_id, gpu_half)`` by
    :class:`dsart.services.search_compute.SearchComputeService`. Two call
    sites drive it, both on the service's asyncio event-loop thread:

      * the per-cube path (``maybe_stage``) after the detector ran;
      * the C2 trigger listener (``claim`` / ``drop_pending``) on a
        datagram.

    A lock guards the pending registry anyway so tests (and any future
    off-loop caller) are safe.

    Args:
        config: thresholds + rate limits.
        dump_root: the C1 dump root (same dir the C2 listener + uploader
            use). Pending dirs are created as
            ``<dump_root>/pending_g<g>_<specnum>/``; claimed cubes move
            to ``<dump_root>/<event_name>/``.
        search_node_id, gpu_half: identity of this dumping process.
        cube_dump_writer: shared :class:`CubeDumpWriter`. ``None``
            disables staging (``maybe_stage`` no-ops).
        upload_fn: optional ``(event_name, event_dir) -> None`` bound to
            the bounded uploader's ``submit`` at wiring time; fired after
            a successful claim. ``None`` = no upload (offline benches).
        time_now_s: monotonic clock injector (tests pass a fake).
    """

    __slots__ = (
        "_config",
        "_dump_root",
        "_sid",
        "_g",
        "_writer",
        "_upload_fn",
        "_time_now_s",
        "_lock",
        "_pending",
        "_last_stage_s",
        "_n_staged",
        "_n_below_thresh",
        "_n_rate_limited",
        "_n_submit_dropped",
        "_n_evicted",
        "_n_gc",
        "_n_claimed",
        "_n_claim_miss",
        "_n_dropped_live",
    )

    def __init__(
        self,
        config: ProactiveStagerConfig,
        *,
        dump_root: Path,
        search_node_id: int,
        gpu_half: int,
        cube_dump_writer: Optional[CubeDumpWriter] = None,
        upload_fn: Optional[Callable[[str, Path], None]] = None,
        time_now_s: Callable[[], float] = time.monotonic,
    ) -> None:
        if config.max_pending < 1:
            raise ValueError(
                f"max_pending={config.max_pending}, expected >= 1"
            )
        if config.min_interval_s < 0.0:
            raise ValueError(
                f"min_interval_s={config.min_interval_s}, expected >= 0"
            )
        if config.ttl_s <= 0.0:
            raise ValueError(f"ttl_s={config.ttl_s}, expected > 0")
        self._config = config
        self._dump_root = Path(dump_root)
        self._sid = int(search_node_id)
        self._g = int(gpu_half)
        self._writer = cube_dump_writer
        self._upload_fn = upload_fn
        self._time_now_s = time_now_s
        self._lock = threading.Lock()
        # Insertion-ordered: first item is the OLDEST (LRU eviction end).
        self._pending: "OrderedDict[int, _PendingEntry]" = OrderedDict()
        self._last_stage_s: Optional[float] = None
        # Counters (read via ``mon``).
        self._n_staged = 0
        self._n_below_thresh = 0
        self._n_rate_limited = 0
        self._n_submit_dropped = 0
        self._n_evicted = 0
        self._n_gc = 0
        self._n_claimed = 0
        self._n_claim_miss = 0
        self._n_dropped_live = 0

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def _pending_dir(self, specnum_start: int) -> Path:
        return self._dump_root / f"pending_g{self._g}_{int(specnum_start)}"

    def _npz_name(self, specnum: int) -> str:
        return f"cube_s{self._sid}_g{self._g}_{int(specnum)}.npz"

    # ------------------------------------------------------------------
    # Stage
    # ------------------------------------------------------------------

    def maybe_stage(
        self,
        *,
        cube_id: int,
        specnum_start: int,
        t_det: int,
        sample_period_specnum: int,
        n_fdm_in_cube: int,
        n_grid: int,
        mjd_start: float,
        max_snr: float,
        cube_tensor: Any,
    ) -> bool:
        """Stage this cube to a provisional dir iff it is bright enough.

        Non-blocking: the NPZ write is enqueued on the shared
        :class:`CubeDumpWriter` (``put_nowait``); a full queue drops the
        stage (counted) rather than stalling the hot loop.

        Args:
            cube_id, specnum_start, t_det, sample_period_specnum,
            n_fdm_in_cube, n_grid, mjd_start: cube geometry (from the
                ``CubeRingSlot`` + ``CubeGeometry``), used to build the
                manifest and the claimable specnum window.
            max_snr: peak candidate SNR in this cube (max over the
                detector's candidate list; 0.0 for an empty cube).
            cube_tensor: the post-detector cube (``result.cube``), passed
                straight to the writer — same object the udp/auto dump
                paths submit.

        Returns:
            ``True`` iff a stage was submitted to the writer queue.
        """
        cfg = self._config
        if not cfg.enabled or self._writer is None:
            return False
        # Opportunistic TTL sweep every call (cheap; O(pending)).
        self.gc()
        if float(max_snr) < cfg.snr_threshold:
            with self._lock:
                self._n_below_thresh += 1
            return False
        now_s = self._time_now_s()
        with self._lock:
            if (
                self._last_stage_s is not None
                and (now_s - self._last_stage_s) < cfg.min_interval_s
            ):
                self._n_rate_limited += 1
                return False

        specnum_start = int(specnum_start)
        end_excl = specnum_start + int(t_det) * int(sample_period_specnum)
        pending_dir = self._pending_dir(specnum_start)
        npz_path = pending_dir / self._npz_name(specnum_start)
        manifest = CubeDumpManifest(
            cube_id=int(cube_id),
            event_specnum_start=specnum_start,
            mjd_start=float(mjd_start),
            t_det=int(t_det),
            n_fdm_in_cube=int(n_fdm_in_cube),
            n_grid=int(n_grid),
            # "udp" == externally-triggered dump with no cluster record;
            # a claimed proactive cube is functionally a C2-triggered
            # dump, so this keeps the NPZ schema identical to the C2 path
            # (avoids widening the contract's trigger_source enum).
            trigger_source="udp",
            cluster_record=None,
            npz_path=str(npz_path),
            search_node_id=self._sid,
            gpu_half=self._g,
        )
        accepted = bool(
            self._writer.submit(cube=cube_tensor, manifest=manifest)
        )
        if not accepted:
            with self._lock:
                self._n_submit_dropped += 1
            _LOG.warning(
                "proactive_stage dropped (writer queue full): "
                "specnum_start=%d snr=%.1f", specnum_start, float(max_snr),
            )
            return False

        with self._lock:
            # Budget: evict oldest until there is room for this entry.
            # (Re-staging the same specnum just refreshes it.)
            self._pending.pop(specnum_start, None)
            while len(self._pending) >= cfg.max_pending:
                _old_specnum, old = self._pending.popitem(last=False)
                self._remove_dir(old.pending_dir)
                self._n_evicted += 1
                _LOG.info(
                    "proactive_stage evicted oldest pending (specnum_start=%d) "
                    "to stay under budget %d", _old_specnum, cfg.max_pending,
                )
            self._pending[specnum_start] = _PendingEntry(
                specnum_start=specnum_start,
                specnum_end_excl=end_excl,
                cube_id=int(cube_id),
                pending_dir=pending_dir,
                npz_path=npz_path,
                staged_s=now_s,
            )
            self._last_stage_s = now_s
            self._n_staged += 1
        _LOG.info(
            "proactive_stage armed: specnum window [%d, %d) snr=%.1f -> %s",
            specnum_start, end_excl, float(max_snr), npz_path,
        )
        return True

    # ------------------------------------------------------------------
    # Claim / drop (C2 listener side)
    # ------------------------------------------------------------------

    def _find_covering(self, event_specnum: int) -> Optional[int]:
        """Return the pending key whose window covers ``event_specnum``
        (newest match wins), or ``None``. Caller holds the lock."""
        ev = int(event_specnum)
        best: Optional[int] = None
        # OrderedDict is oldest->newest; iterate reversed for newest-first.
        for key in reversed(self._pending):
            entry = self._pending[key]
            if entry.specnum_start <= ev < entry.specnum_end_excl:
                best = key
                break
        return best

    def claim(
        self,
        *,
        event_specnum: int,
        event_name: str,
    ) -> Optional[Path]:
        """Claim a staged pending cube for a named event.

        Renames this half's staged NPZ into
        ``<dump_root>/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz``
        and fires the uploader. Idempotent: once claimed the entry is
        forgotten, so a repeated claim (or a claim for a specnum with no
        staged window) returns ``None``.

        Returns:
            The event archive dir on a successful claim, else ``None``.
        """
        event_name = str(event_name)
        with self._lock:
            key = self._find_covering(event_specnum)
            if key is None:
                self._n_claim_miss += 1
                return None
            entry = self._pending.pop(key)

        event_dir = self._dump_root / event_name
        final_path = event_dir / self._npz_name(event_specnum)
        try:
            event_dir.mkdir(parents=True, exist_ok=True)
            # The writer publishes the NPZ atomically (tmp + os.replace),
            # so if the file is present it is complete. If it is not yet
            # on disk (writer still draining, or the submit was dropped)
            # the claim fails gracefully -- the caller falls back to the
            # unchanged too_late accounting.
            os.replace(entry.npz_path, final_path)
        except FileNotFoundError:
            with self._lock:
                self._n_claim_miss += 1
            _LOG.warning(
                "proactive_stage claim: staged NPZ absent for event=%s "
                "specnum=%d (path=%s); leaving as too_late",
                event_name, int(event_specnum), entry.npz_path,
            )
            return None
        except OSError as exc:
            with self._lock:
                self._n_claim_miss += 1
            _LOG.warning(
                "proactive_stage claim failed for event=%s specnum=%d: %r",
                event_name, int(event_specnum), exc,
            )
            return None
        # Best-effort tidy of the now-empty provisional dir (other-half
        # files, if any, keep it alive -- rmdir only succeeds when empty).
        self._rmdir_if_empty(entry.pending_dir)
        with self._lock:
            self._n_claimed += 1
        _LOG.info(
            "proactive_stage CLAIMED: event=%s specnum=%d rescued %s -> %s",
            event_name, int(event_specnum), entry.npz_path, final_path,
        )
        if self._upload_fn is not None:
            try:
                self._upload_fn(event_name, event_dir)
            except Exception as exc:  # noqa: BLE001 - never poison caller
                _LOG.warning(
                    "proactive_stage claim: uploader submit failed for "
                    "event=%s: %r", event_name, exc,
                )
        return event_dir

    def drop_pending(self, event_specnum: int) -> bool:
        """Discard the staged pending cube covering ``event_specnum``.

        Called on a C2 *hit* (the cube was still in the live ring, so the
        listener dumps from the ring and the staged copy is redundant).
        Removes this half's pending dir. Returns ``True`` iff something
        was dropped.
        """
        with self._lock:
            key = self._find_covering(event_specnum)
            if key is None:
                return False
            entry = self._pending.pop(key)
            self._n_dropped_live += 1
        self._remove_dir(entry.pending_dir)
        _LOG.info(
            "proactive_stage dropped staged copy (live-ring dump won): "
            "specnum=%d dir=%s", int(event_specnum), entry.pending_dir,
        )
        return True

    # ------------------------------------------------------------------
    # GC
    # ------------------------------------------------------------------

    def gc(self) -> int:
        """Remove pending entries older than ``ttl_s``. Returns count."""
        cutoff = self._time_now_s() - self._config.ttl_s
        removed = 0
        with self._lock:
            stale = [
                k for k, e in self._pending.items() if e.staged_s < cutoff
            ]
            for k in stale:
                entry = self._pending.pop(k)
                self._remove_dir(entry.pending_dir)
                self._n_gc += 1
                removed += 1
        if removed:
            _LOG.info("proactive_stage gc removed %d stale pending dir(s)", removed)
        return removed

    # ------------------------------------------------------------------
    # Dir helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_dir(path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError as exc:  # pragma: no cover - defensive
            _LOG.warning("proactive_stage: failed to remove %s: %r", path, exc)

    @staticmethod
    def _rmdir_if_empty(path: Path) -> None:
        try:
            path.rmdir()
        except OSError:
            # Not empty (other half's file) or already gone -- leave it.
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_pending(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def mon(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self._config.enabled),
                "snr_threshold": float(self._config.snr_threshold),
                "n_pending": len(self._pending),
                "staged": self._n_staged,
                "below_thresh": self._n_below_thresh,
                "rate_limited": self._n_rate_limited,
                "submit_dropped": self._n_submit_dropped,
                "evicted": self._n_evicted,
                "gc": self._n_gc,
                "claimed": self._n_claimed,
                "claim_miss": self._n_claim_miss,
                "dropped_live": self._n_dropped_live,
            }
