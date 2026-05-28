"""M7.4 Phase 6c: search-compute cube-ring mon-key publisher.

Publishes the search-side :class:`CubeRetentionRing` window to
``/mon/search/<sid>/<gpu_half>/ring`` so external consumers (in
particular the dsa_monitor Control tab's "Dump Now" button) can pick
an ``event_specnum`` that's guaranteed to land inside the ring's
retention window of every search-half before issuing a synthetic C2
trigger.

Why a separate mon-key from ``/mon/corr_rt/<cn>/corr_fast``?
============================================================

``corr_fast`` publishes its *service-start* epoch (``block_n ×
NPACKETS_PER_BLOCK``) — that key drives the runtime-inject auto-arm
because the corr_fast inject path computes the apply specnum in
exactly the same domain.

The search-side cube_ring, however, keys retained cubes by
``slot.specnum_start`` — the input-data SNAP specnum (post search-rx
PSRDADA-buffer staging), which is offset from corr_fast's
service-start counter by at least the search-rx startup delta plus
any per-cube cadence drift. The 2026-05-28 Phase 6c soak (commit
8b3f2ad) exposed exactly this domain mismatch: ``cube_dump_now``
issued synthetic triggers using corr_fast's epoch and every search
half logged ``C2TriggerListener miss (too_early)`` even though the
trigger was correctly broadcast — the requested ``event_specnum``
was three orders of magnitude beyond the live ring.

The fix is to have search_compute publish *its own* ring window so
the dashboard speaks the search-side specnum domain natively.

Schema (etcd ``/mon/search/<sid>/<gpu_half>/ring``):

.. code-block:: text

    {
      "search_node_id":          int,
      "gpu_half":                int,
      "newest_event_specnum_start": int,   # ring's most-recent cube start
      "newest_event_specnum_end":   int,   # exclusive end of newest cube
      "oldest_event_specnum_start": int,   # ring's oldest committed cube start
      "n_committed":             int,      # number of cubes currently in ring
      "depth":                   int,      # ring slot count
      "t_det":                   int,      # samples per cube
      "n_fdm":                   int,      # fine-DM trials per cube
      "n_grid":                  int,      # grid edge (square)
      "sample_period_specnum":   int,      # specnums per search sample
      "newest_cube_id":          int,
      "oldest_cube_id":          int,
      "ts_mono":                 float,
      "ts_wall_unix":            float,
      "n_published":             int,      # cumulative publish count (debug)
    }

The publisher is wired into the existing
``_log_cube_progress`` cadence (one log line every
``_status_every_cubes`` = 10 cubes ≈ 1.3 s at 7.45 cubes/s), so the
etcd PUT rate is well under any sensible ceiling. The ``put_dict``
call is wrapped in ``try/except`` so an etcd hiccup never blocks the
hot loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional


LOG = logging.getLogger("dsart.services.search_ring_mon")


__all__ = [
    "SearchRingMonPublisher",
    "build_search_ring_mon_key",
]


def build_search_ring_mon_key(search_node_id: int, gpu_half: int) -> str:
    """Canonical etcd key for this publisher.

    Layout mirrors :func:`dsart.services.corr_fast_mon.build_corr_fast_mon_key`:
    ``/mon/search/<sid>/<gpu_half>/ring`` (s + g chosen so wildcards
    like ``/mon/search/*/0/ring`` give one half across the fleet).
    """
    return f"/mon/search/{int(search_node_id)}/{int(gpu_half)}/ring"


class SearchRingMonPublisher:
    """Best-effort publisher of one gpu_half's cube_ring state.

    Parameters
    ----------
    search_node_id, gpu_half
        Identify the publishing half. One publisher per
        ``(search_node_id, gpu_half)`` pair.
    store
        Optional pre-built ``DsaStore``. Production passes ``None``
        and the publisher lazily constructs one on first
        ``publish``; tests pass a mock so ``dsautils`` is not
        required at import time.
    """

    def __init__(
        self,
        *,
        search_node_id: int,
        gpu_half: int,
        store: Optional[Any] = None,
    ) -> None:
        self.search_node_id = int(search_node_id)
        self.gpu_half = int(gpu_half)
        self._store = store
        self._lock = threading.Lock()
        self._n_published = 0
        self._n_errors = 0
        self._last_publish_ts: float | None = None
        self._first_event_logged = False
        self._first_error_logged = False

    @property
    def key(self) -> str:
        return build_search_ring_mon_key(self.search_node_id, self.gpu_half)

    @property
    def n_published(self) -> int:
        return self._n_published

    @property
    def n_errors(self) -> int:
        return self._n_errors

    @property
    def last_publish_ts(self) -> float | None:
        return self._last_publish_ts

    def publish_from_ring(self, ring: Any) -> bool:
        """Snapshot ``ring`` and publish its window.

        ``ring`` is expected to be a
        :class:`dsart.services.cube_pipeline.CubeRetentionRing` (or
        any duck-typed equivalent — tests pass a stub).  An empty
        ring is published as ``n_committed=0`` with the *_specnum_*
        fields set to ``None`` so the consumer can detect that
        explicitly.
        """
        try:
            snapshot = ring.snapshot()
        except Exception as exc:                                # noqa: BLE001
            self._n_errors += 1
            if not self._first_error_logged:
                LOG.warning(
                    "SearchRingMonPublisher: ring.snapshot() failed: %s "
                    "(key=%s; subsequent failures will be silent)",
                    exc, self.key,
                )
                self._first_error_logged = True
            return False

        payload: dict[str, Any] = {
            "search_node_id": self.search_node_id,
            "gpu_half": self.gpu_half,
            "n_committed": int(len(snapshot)),
            "depth": int(getattr(ring, "depth", 0)),
            "t_det": int(getattr(ring, "t_det", 0)),
            "n_fdm": int(getattr(ring, "n_fdm", 0)),
            "n_grid": int(getattr(ring, "n_grid", 0)),
            "ts_mono": time.monotonic(),
            "ts_wall_unix": time.time(),
        }
        if snapshot:
            newest = snapshot[0]
            oldest = snapshot[-1]
            spp = int(
                getattr(newest, "sample_period_specnum", 1) or 1
            )
            t_det = int(getattr(newest, "t_det", 0))
            payload.update({
                "newest_event_specnum_start":
                    int(newest.event_specnum_start),
                "newest_event_specnum_end":
                    int(newest.event_specnum_start) + t_det * spp,
                "oldest_event_specnum_start":
                    int(oldest.event_specnum_start),
                "sample_period_specnum": spp,
                "newest_cube_id": int(newest.cube_id),
                "oldest_cube_id": int(oldest.cube_id),
                "newest_mjd_start": float(
                    getattr(newest, "mjd_start", 0.0)
                ),
            })
        else:
            payload.update({
                "newest_event_specnum_start": None,
                "newest_event_specnum_end": None,
                "oldest_event_specnum_start": None,
                "sample_period_specnum": int(
                    getattr(ring, "sample_period_specnum", 0)
                ) if hasattr(ring, "sample_period_specnum") else 0,
                "newest_cube_id": None,
                "oldest_cube_id": None,
                "newest_mjd_start": None,
            })

        return self._put(payload)

    def _put(self, payload: dict[str, Any]) -> bool:
        with self._lock:
            store = self._store
            if store is None:
                try:
                    from dsautils.dsa_store import DsaStore
                except Exception:                              # noqa: BLE001
                    self._n_errors += 1
                    if not self._first_error_logged:
                        LOG.warning(
                            "SearchRingMonPublisher: dsautils not "
                            "importable; mon-key %s will not be "
                            "published",
                            self.key,
                        )
                        self._first_error_logged = True
                    return False
                try:
                    store = DsaStore()
                except Exception as exc:                       # noqa: BLE001
                    self._n_errors += 1
                    LOG.warning(
                        "SearchRingMonPublisher: DsaStore() failed "
                        "(%s); mon-key %s will not be published this "
                        "cycle",
                        exc, self.key,
                    )
                    return False
                self._store = store

        payload["n_published"] = int(self._n_published) + 1
        try:
            self._store.put_dict(self.key, payload)
        except Exception as exc:                                # noqa: BLE001
            self._n_errors += 1
            if not self._first_error_logged:
                LOG.warning(
                    "SearchRingMonPublisher: first put_dict(%s) failed: "
                    "%s (subsequent failures will be silent)",
                    self.key, exc,
                )
                self._first_error_logged = True
            return False

        self._n_published += 1
        self._last_publish_ts = payload["ts_mono"]
        if not self._first_event_logged:
            LOG.info(
                "SearchRingMonPublisher up: key=%s n_committed=%d "
                "newest_event_specnum_start=%s",
                self.key, payload["n_committed"],
                payload.get("newest_event_specnum_start"),
            )
            self._first_event_logged = True
        return True
