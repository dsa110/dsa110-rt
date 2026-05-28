"""M7.4 Phase 6c: corr_fast service-state mon-key publisher.

Publishes corr_fast's *service-start* epoch state to
``/mon/corr_rt/<chgroup>/corr_fast`` so external consumers (in particular
the dsa_monitor Control tab's "Send injection" form) can align
``apply_at_specnum`` with the corr_fast hot path's
``block_specnum_start = block_n * NPACKETS_PER_BLOCK`` reference.

Why a separate epoch from the SNAP-header (``last_seq_no``) one
already published under ``/mon/corr_rt/<cn>/capture/<port>``?
=====================================================================

Until the M7.2.8 corner-turn work lands, ``corr_fast_integration``
counts blocks from service start (``block_n = 0`` when the routine
launches) rather than from the SNAP packet header. That means the
SNAP-header epoch (~1.95 × 10¹⁰ after months of uptime) and the
corr_fast service-start epoch (~1.5 × 10⁷ after 17 min) differ by
three orders of magnitude. The dashboard's previous
``compute_arm_seq`` used the SNAP-header epoch — fine for
``utc_start`` (which is consumed by ``capture_control``) but wrong
for ``inject`` (which is consumed by ``OnlineInjector.apply_block``
inside corr_fast). This mon-key gives the dashboard the corr_fast
epoch.

Hot-path safety
===============

The publisher is wired into the existing every-16-blocks heartbeat
(``corr_fast_integration.run``'s ``if n_in % 16 == 0:`` log line), so
the etcd put rate is one PUT every ~2 s at 8 cubes/s — well below
any sensible ceiling. The ``put_dict`` call is wrapped in
``try/except`` so an etcd hiccup never blocks the hot loop. We log
the first failure and stay silent on subsequent ones, surfacing the
state via ``n_errors`` so the dashboard can flag a publisher that's
not reaching etcd.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional


NPACKETS_PER_BLOCK_DEFAULT = 2048


LOG = logging.getLogger("dsart.services.corr_fast_mon")


__all__ = [
    "CorrFastMonPublisher",
    "build_corr_fast_mon_key",
    "NPACKETS_PER_BLOCK_DEFAULT",
]


def build_corr_fast_mon_key(chgroup: int) -> str:
    """Canonical etcd key for this publisher.

    Mirrors the layout of ``capture_control`` / ``rfi_monitor_export`` /
    ``dsart_rt``: ``/mon/corr_rt/<chgroup>/corr_fast``.
    """
    return f"/mon/corr_rt/{int(chgroup)}/corr_fast"


class CorrFastMonPublisher:
    """Best-effort publisher of corr_fast service state to etcd.

    Parameters
    ----------
    chgroup:
        Sub-band index 0..15 (= corr-node ID).
    npackets_per_block:
        Samples-per-block constant. Defaults to 2048, matching the
        production ``NPACKETS_PER_BLOCK`` in ``corr_fast_integration``;
        the constant is repeated here so unit tests can import this
        module without dragging in torch / numpy.
    store:
        Optional pre-built ``DsaStore``. Production passes ``None`` and
        the publisher lazily constructs one on first ``publish``. Tests
        pass a mock so ``dsautils`` is not required at import time.
    """

    def __init__(
        self,
        *,
        chgroup: int,
        npackets_per_block: int = NPACKETS_PER_BLOCK_DEFAULT,
        store: Optional[Any] = None,
    ) -> None:
        self.chgroup = int(chgroup)
        self._npackets_per_block = int(npackets_per_block)
        self._store = store
        self._lock = threading.Lock()
        self._n_published = 0
        self._n_errors = 0
        self._last_publish_ts: float | None = None
        self._first_event_logged = False

    @property
    def key(self) -> str:
        return build_corr_fast_mon_key(self.chgroup)

    @property
    def n_published(self) -> int:
        return self._n_published

    @property
    def n_errors(self) -> int:
        return self._n_errors

    @property
    def last_publish_ts(self) -> float | None:
        return self._last_publish_ts

    def publish(
        self,
        *,
        block_n: int,
        n_processed: int = 0,
        n_drop: int = 0,
        n_tx: int = 0,
        last_block_ms: float | None = None,
        extra: dict | None = None,
    ) -> bool:
        """Publish one state snapshot.

        Best-effort: an etcd error is logged once and swallowed so the
        corr_fast hot loop never blocks on etcd. Returns ``True`` on
        successful publish, ``False`` otherwise.
        """
        with self._lock:
            store = self._store
            if store is None:
                try:
                    from dsautils.dsa_store import DsaStore
                except Exception:                            # noqa: BLE001
                    self._n_errors += 1
                    if not self._first_event_logged:
                        LOG.warning(
                            "CorrFastMonPublisher: dsautils not "
                            "importable; mon-key %s will not be "
                            "published",
                            self.key,
                        )
                        self._first_event_logged = True
                    return False
                try:
                    store = DsaStore()
                except Exception as exc:                     # noqa: BLE001
                    self._n_errors += 1
                    LOG.warning(
                        "CorrFastMonPublisher: DsaStore() failed "
                        "(%s); mon-key %s will not be published this "
                        "cycle",
                        exc, self.key,
                    )
                    return False
                self._store = store

        payload: dict[str, Any] = {
            "block_n": int(block_n),
            "block_specnum_start": (
                int(block_n) * self._npackets_per_block
            ),
            "npackets_per_block": int(self._npackets_per_block),
            "ts_mono": time.monotonic(),
            "ts_wall_unix": time.time(),
            "n_processed": int(n_processed),
            "n_drop": int(n_drop),
            "n_tx": int(n_tx),
        }
        if last_block_ms is not None:
            payload["last_block_ms"] = float(last_block_ms)
        if extra:
            payload.update(extra)

        try:
            self._store.put_dict(self.key, payload)
        except Exception as exc:                             # noqa: BLE001
            self._n_errors += 1
            if not self._first_event_logged:
                LOG.warning(
                    "CorrFastMonPublisher: first put_dict(%s) failed: "
                    "%s (subsequent failures will be silent)",
                    self.key, exc,
                )
                self._first_event_logged = True
            return False

        self._n_published += 1
        self._last_publish_ts = payload["ts_mono"]
        if not self._first_event_logged:
            LOG.info(
                "CorrFastMonPublisher up: key=%s block_n=%d "
                "block_specnum_start=%d",
                self.key, payload["block_n"],
                payload["block_specnum_start"],
            )
            self._first_event_logged = True
        return True
