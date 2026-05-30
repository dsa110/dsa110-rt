"""M7.6: search-compute C1→C2 metering mon-key publisher.

Publishes a per-``(search_node_id, gpu_half)`` rollup of the C1→C2
metering state to ``/mon/search_rt/<cn>/compute/<gpu_half>`` so the
``dsart_rt_to_influx`` pusher can surface it as the ``search_rt_compute``
measurement (one field set per search half) and Grafana can show, at a
glance, whether *any* search process is currently shedding candidates to
protect the C1→C2 path.

Why ``/mon/search_rt/<cn>/compute/<g>`` (and not ``/mon/search/...``)?
=====================================================================

The influx pusher only scans the ``/mon/search_rt/`` (etc.) prefixes and
already reserves ``/mon/search_rt/<cn>/compute/<half>`` as a planned key
(see ``KEY_SEARCH_COMPUTE`` in ``dsart_rt_to_influx/pusher.py``).  The
search node's ``--search-node-id`` *is* the etcd ``cn`` (1/2/9/13 — see
``configs/dsart_search_rt.yaml``), so publishing under that key lets the
metric ride an already-routed prefix and pick up the standard
``host`` / ``coarse_dm`` tags.

Update rate
===========

The publisher is fed once per ``window_blocks`` cubes (default 16 ≈ 2.1 s
at 7.45 cubes/s) with a *window average* so the etcd/influx PUT rate
stays low.  The ``put_dict`` call is wrapped in ``try/except`` so an etcd
hiccup never blocks the search hot loop.

Schema (etcd ``/mon/search_rt/<cn>/compute/<g>``):

.. code-block:: text

    {
      "search_node_id":            int,
      "gpu_half":                  int,
      "c1_metering_active":        int,    # 1 if any block in window metered
      "c1_metering_frac":          float,  # fraction of blocks that metered
      "c1_metered_dropped_mean":   float,  # mean cands dropped/block by cap
      "c1_metered_dropped_max":    int,    # worst single block in window
      "c1_cands_per_block_mean":   float,  # mean width-survivors/block (ctx)
      "c1_max_candidates_per_block": int,  # configured cap (0 = disabled)
      "n_blocks":                  int,    # window size actually averaged
      "ts_mono":                   float,
      "ts_wall_unix":              float,
      "time_mjd":                  float,
      "n_published":               int,
    }
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional


LOG = logging.getLogger("dsart.services.search_compute_mon")


__all__ = [
    "SearchComputeMonPublisher",
    "build_search_compute_mon_key",
]


def build_search_compute_mon_key(search_node_id: int, gpu_half: int) -> str:
    """Canonical etcd key: ``/mon/search_rt/<cn>/compute/<gpu_half>``."""
    return f"/mon/search_rt/{int(search_node_id)}/compute/{int(gpu_half)}"


class SearchComputeMonPublisher:
    """Best-effort publisher of one gpu_half's C1→C2 metering rollup.

    Parameters
    ----------
    search_node_id, gpu_half
        Identify the publishing half (``search_node_id`` is the etcd cn).
    store
        Optional pre-built ``DsaStore``. Production passes ``None`` and the
        publisher lazily constructs one on first ``publish``; tests pass a
        mock so ``dsautils`` is not required at import time.
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
        self._first_error_logged = False
        self._first_event_logged = False

    @property
    def key(self) -> str:
        return build_search_compute_mon_key(self.search_node_id, self.gpu_half)

    @property
    def n_published(self) -> int:
        return self._n_published

    def publish_metering(
        self,
        *,
        n_blocks: int,
        n_metered_blocks: int,
        dropped_sum: int,
        dropped_max: int,
        cands_sum: int,
        cap: int,
    ) -> bool:
        """Average the window counters and publish the rollup.

        ``n_blocks`` is the number of cubes accumulated since the last
        publish; the remaining args are the summed/extremal per-block
        counters over that window.
        """
        n = max(int(n_blocks), 1)
        frac = float(n_metered_blocks) / float(n)
        payload: dict[str, Any] = {
            "search_node_id": self.search_node_id,
            "gpu_half": self.gpu_half,
            "c1_metering_active": int(1 if n_metered_blocks > 0 else 0),
            "c1_metering_frac": round(frac, 4),
            "c1_metered_dropped_mean": round(float(dropped_sum) / float(n), 3),
            "c1_metered_dropped_max": int(dropped_max),
            "c1_cands_per_block_mean": round(float(cands_sum) / float(n), 3),
            "c1_max_candidates_per_block": int(cap),
            "n_blocks": int(n_blocks),
            "ts_mono": time.monotonic(),
            "ts_wall_unix": time.time(),
            "time_mjd": time.time() / 86400.0 + 40587.0,
        }
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
                            "SearchComputeMonPublisher: dsautils not "
                            "importable; mon-key %s will not be published",
                            self.key,
                        )
                        self._first_error_logged = True
                    return False
                try:
                    store = DsaStore()
                except Exception as exc:                       # noqa: BLE001
                    self._n_errors += 1
                    LOG.warning(
                        "SearchComputeMonPublisher: DsaStore() failed (%s); "
                        "mon-key %s will not be published this cycle",
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
                    "SearchComputeMonPublisher: first put_dict(%s) failed: "
                    "%s (subsequent failures will be silent)",
                    self.key, exc,
                )
                self._first_error_logged = True
            return False

        self._n_published += 1
        if not self._first_event_logged:
            LOG.info(
                "SearchComputeMonPublisher up: key=%s metering_active=%d "
                "frac=%.3f cap=%d",
                self.key, payload["c1_metering_active"],
                payload["c1_metering_frac"],
                payload["c1_max_candidates_per_block"],
            )
            self._first_event_logged = True
        return True
