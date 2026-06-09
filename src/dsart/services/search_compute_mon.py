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
    "build_search_noise_mon_key",
    "build_search_dump_health_mon_key",
]


def build_search_compute_mon_key(search_node_id: int, gpu_half: int) -> str:
    """Canonical etcd key: ``/mon/search_rt/<cn>/compute/<gpu_half>``."""
    return f"/mon/search_rt/{int(search_node_id)}/compute/{int(gpu_half)}"


def build_search_noise_mon_key(search_node_id: int, gpu_half: int) -> str:
    """Canonical etcd key for the Layer-2 σ_k noise rollup: ``/mon/search_rt/<cn>/noise/<gpu_half>``.

    Surfaces the EMA divisor health (median, p95, n_clamped_high) so the
    dashboard can flag a half whose noise estimator is inflated or
    repeatedly being clamped from above by ``layer2_sigma_max_ratio``.
    """
    return f"/mon/search_rt/{int(search_node_id)}/noise/{int(gpu_half)}"


def build_search_dump_health_mon_key(search_node_id: int, gpu_half: int) -> str:
    """Canonical etcd key for the cube-dump health rollup: ``/mon/search_rt/<cn>/dump/<gpu_half>``.

    Carries the cube-dump writer queue-full drop counter and the C2
    trigger listener miss counters (too_late / too_early / bad_*) so
    the dashboard can flag an undersized retention ring or a saturated
    writer queue without grovelling through logs.
    """
    return f"/mon/search_rt/{int(search_node_id)}/dump/{int(gpu_half)}"


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
        # T1/T8: per-key publish counters so each best-effort PUT can be
        # tracked independently of the C1-metering rollup. Reuses the
        # same lazy DsaStore handle.
        self._n_noise_published = 0
        self._n_dump_published = 0

    @property
    def key(self) -> str:
        return build_search_compute_mon_key(self.search_node_id, self.gpu_half)

    @property
    def noise_key(self) -> str:
        return build_search_noise_mon_key(self.search_node_id, self.gpu_half)

    @property
    def dump_health_key(self) -> str:
        return build_search_dump_health_mon_key(
            self.search_node_id, self.gpu_half,
        )

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

    def publish_noise(
        self,
        *,
        s_k_median: float,
        s_k_p95: float,
        s_k_max: float,
        s_k_min: float,
        n_clamped_high_total: int,
        n_clamped_high_max_per_kernel: int,
        sigma_max_ratio: float,
        cube_count: int,
        is_warming_up: bool,
        s_k_per_kernel: Optional[dict[str, float]] = None,
        n_clamp_escapes_total: int = 0,
        clamp_streak_max: int = 0,
    ) -> bool:
        """Publish the Layer-2 σ_k EMA health rollup.

        Best-effort: an etcd error is swallowed so the search hot loop
        never blocks. Returns True on a successful publish.

        ``s_k_per_kernel`` (2026-06-09) maps the canonical kernel_id
        (e.g. ``"unit:d1:b16"``) to that kernel's current σ_k. Each
        entry is flattened into the payload as ``s_k_<kernel_id with
        ':' → '_'>`` (e.g. ``s_k_unit_d1_b16``) so the influx pusher
        can forward one field per kernel without schema changes for
        every new bank shape. The med/p95/max rollup alone hid the
        2026-06-09 clamp-deadlock failure mode (two kernels stuck low
        while the median looked healthy).
        """
        payload: dict[str, Any] = {
            "search_node_id": self.search_node_id,
            "gpu_half": self.gpu_half,
            "s_k_median": float(s_k_median),
            "s_k_p95": float(s_k_p95),
            "s_k_max": float(s_k_max),
            "s_k_min": float(s_k_min),
            "n_clamped_high_total": int(n_clamped_high_total),
            "n_clamped_high_max_per_kernel": int(
                n_clamped_high_max_per_kernel
            ),
            "layer2_sigma_max_ratio": float(sigma_max_ratio),
            "layer2_cube_count": int(cube_count),
            "layer2_is_warming_up": bool(is_warming_up),
            "n_clamp_escapes_total": int(n_clamp_escapes_total),
            "clamp_streak_max": int(clamp_streak_max),
            "ts_mono": time.monotonic(),
            "ts_wall_unix": time.time(),
            "time_mjd": time.time() / 86400.0 + 40587.0,
        }
        if s_k_per_kernel:
            for kid, val in s_k_per_kernel.items():
                field = "s_k_" + str(kid).replace(":", "_").replace(
                    "-", "_"
                )
                payload[field] = float(val)
        return self._put(payload, key=self.noise_key, counter="noise")

    def publish_dump_health(
        self,
        *,
        cube_dump_n_dumped: int,
        cube_dump_n_dropped: int,
        cube_dump_n_failed: int,
        cube_dump_queue_depth: int,
        cube_dump_queue_maxsize: int,
        c2_trigger_received: int,
        c2_trigger_hits: int,
        c2_trigger_too_late: int,
        c2_trigger_too_early: int,
        c2_trigger_bad_magic: int,
        c2_trigger_bad_schema: int,
        c2_trigger_dispatch_dropped: int,
        cube_ring_depth: int,
        cube_ring_oldest_specnum: int,
        cube_ring_newest_end_specnum_excl: int,
    ) -> bool:
        """Publish the cube-dump + C2-trigger listener health rollup.

        Surfaces the dump path's silent-failure surfaces — writer queue
        full drops + listener too_late misses — so the dashboard can
        flag a saturated dump path before operators notice missing
        plots.
        """
        payload: dict[str, Any] = {
            "search_node_id": self.search_node_id,
            "gpu_half": self.gpu_half,
            "cube_dump_n_dumped": int(cube_dump_n_dumped),
            "cube_dump_n_dropped": int(cube_dump_n_dropped),
            "cube_dump_n_failed": int(cube_dump_n_failed),
            "cube_dump_queue_depth": int(cube_dump_queue_depth),
            "cube_dump_queue_maxsize": int(cube_dump_queue_maxsize),
            "c2_trigger_received": int(c2_trigger_received),
            "c2_trigger_hits": int(c2_trigger_hits),
            "c2_trigger_too_late": int(c2_trigger_too_late),
            "c2_trigger_too_early": int(c2_trigger_too_early),
            "c2_trigger_bad_magic": int(c2_trigger_bad_magic),
            "c2_trigger_bad_schema": int(c2_trigger_bad_schema),
            "c2_trigger_dispatch_dropped": int(c2_trigger_dispatch_dropped),
            "cube_ring_depth": int(cube_ring_depth),
            "cube_ring_oldest_specnum": int(cube_ring_oldest_specnum),
            "cube_ring_newest_end_specnum_excl": int(
                cube_ring_newest_end_specnum_excl
            ),
            "ts_mono": time.monotonic(),
            "ts_wall_unix": time.time(),
            "time_mjd": time.time() / 86400.0 + 40587.0,
        }
        return self._put(payload, key=self.dump_health_key, counter="dump")

    def _put(
        self,
        payload: dict[str, Any],
        *,
        key: Optional[str] = None,
        counter: str = "metering",
    ) -> bool:
        # ``key=None`` preserves the legacy metering publish path (writes
        # to ``self.key``). The T1/T8 paths pass ``key=`` to write to a
        # sibling ``noise/`` or ``dump/`` etcd key.
        target_key = self.key if key is None else key
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

        # Pick the per-counter cumulative published count for the
        # payload's ``n_published`` field so each etcd key carries an
        # independent monotonic counter (operators can tell whether the
        # noise / dump / metering streams are individually flowing).
        if counter == "noise":
            payload["n_published"] = int(self._n_noise_published) + 1
        elif counter == "dump":
            payload["n_published"] = int(self._n_dump_published) + 1
        else:
            payload["n_published"] = int(self._n_published) + 1
        try:
            self._store.put_dict(target_key, payload)
        except Exception as exc:                                # noqa: BLE001
            self._n_errors += 1
            if not self._first_error_logged:
                LOG.warning(
                    "SearchComputeMonPublisher: first put_dict(%s) failed: "
                    "%s (subsequent failures will be silent)",
                    target_key, exc,
                )
                self._first_error_logged = True
            return False

        if counter == "noise":
            self._n_noise_published += 1
        elif counter == "dump":
            self._n_dump_published += 1
        else:
            self._n_published += 1
        if not self._first_event_logged and counter == "metering":
            LOG.info(
                "SearchComputeMonPublisher up: key=%s metering_active=%d "
                "frac=%.3f cap=%d",
                target_key, payload["c1_metering_active"],
                payload["c1_metering_frac"],
                payload["c1_max_candidates_per_block"],
            )
            self._first_event_logged = True
        return True
