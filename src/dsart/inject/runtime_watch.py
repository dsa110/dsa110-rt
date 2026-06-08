"""M7.4 Phase 6 runtime: DsaStore-based watcher that pushes
``InjectionConfig`` payloads into a live :class:`OnlineInjector`.

Why a *second* watcher when ``etcd_watcher.py`` already exists?
-----------------------------------------------------------------

The original :class:`dsart.inject.etcd_watcher.EtcdWatcher` was
written against the raw ``etcd3`` client during M3 chunk 3d so the
injector library could be exercised end-to-end before the broader
fleet-control surface settled. The rest of the production system
(``dsart_rt``, the dashboard, ops scripts) routes all etcd traffic
through :class:`dsautils.dsa_store.DsaStore` — that is the canonical
client + the place where retries, prefix-handling, and JSON
serialisation are unified.

For the M7.4 Phase 6 production wire-in we want the live injection
path to ride the *same* surface the operator already understands
(``/cmd/{ns}/{cn}`` PUT shapes the dashboard writes, with the
``DsaStore.add_watch`` retry semantics the orchestrator already
relies on). This module ports the parsing half of ``etcd_watcher``
to a DsaStore-friendly callback so the corr-fast service can opt in
without dragging in a second etcd client.

Wire-in convention
------------------

The dashboard / orchestrator writes ``{cmd: "inject", val: <cfg>}``
to ``/cmd/dsart/corr/<chgroup>/inject``. Each corr-fast service
watches *its own* per-chgroup key. When the watch fires:

1. The payload is parsed by :func:`handle_inject_payload` (shared
   logic in ``etcd_watcher.py`` — single source of truth for the
   inject wire schema).
2. Valid configs are queued via :meth:`OnlineInjector.add_pending`.
3. Malformed payloads are logged at WARNING and dropped (never
   raised — a bad operator submission must not crash the service).

Lifecycle
---------

:class:`RuntimeInjectWatch` owns the DsaStore handle + watch_id.
The owning ``run()`` function calls :meth:`start` once after
``build_context`` and :meth:`stop` in the ``finally`` block so
cleanup runs even on signal exit / exception.

Production hardening that lives elsewhere:
* DsaStore handles etcd reconnect transparently — no per-call retry
  needed here.
* The injector's :meth:`add_pending` raises on bad configs; we catch
  inside :func:`handle_inject_payload` so the watch thread keeps
  running.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from dsart.inject.etcd_watcher import handle_inject_payload
from dsart.inject.online import OnlineInjector


__all__ = [
    "RuntimeInjectWatch",
    "build_runtime_inject_key",
]


_LOG = logging.getLogger("dsart.inject.runtime_watch")


def build_runtime_inject_key(chgroup: int) -> str:
    """Canonical per-chgroup runtime-injection etcd key.

    Lives under ``/cmd/dsart/corr/<chgroup>/inject`` (one fixed key
    per chgroup, overwritten on each new injection). PUT semantics
    mean any new write triggers the watch; the injector's
    ``add_pending`` queue accumulates pending pulses on the
    receiving side so back-to-back writes are not racy.
    """
    return f"/cmd/dsart/corr/{int(chgroup)}/inject"


class RuntimeInjectWatch:
    """DsaStore watch on the per-chgroup inject key.

    Parameters
    ----------
    injector
        The :class:`OnlineInjector` the corr-fast service built in
        ``build_context``. Each watch event is dispatched into
        :meth:`OnlineInjector.add_pending` via the shared payload
        handler.
    chgroup
        Sub-band index 0..15. Used to derive the watch key.
    store
        Optional pre-built ``DsaStore``. When ``None`` (the production
        path) we lazily import + construct one. Tests pass a mock so
        the module does not require ``dsautils`` to import.
    """

    def __init__(
        self,
        *,
        injector: OnlineInjector,
        chgroup: int,
        store: Optional[Any] = None,
    ) -> None:
        self.injector = injector
        self.chgroup = int(chgroup)
        self._store = store
        self._watch_id: Optional[int] = None
        self._lock = threading.Lock()
        self._n_events = 0
        self._n_queued = 0
        # T2 (2026-06-07): keep the most recently queued inject's
        # identifying metadata so the corr_fast mon publisher can ship
        # it to etcd. Operators reading the dashboard then know not
        # only "this corr node received N injects" but also which
        # specific probe it last queued — the missing signal that made
        # 2026-06-07 partial-fan-out failures indistinguishable from
        # search-side detector pathology.
        self._last_inj_id: Optional[str] = None
        self._last_apply_at_specnum: Optional[int] = None
        self._last_event_unix: Optional[float] = None
        self._last_queued_unix: Optional[float] = None

    @property
    def key(self) -> str:
        return build_runtime_inject_key(self.chgroup)

    @property
    def n_events(self) -> int:
        """Total raw etcd PUT events received on the watch."""
        return self._n_events

    @property
    def n_queued(self) -> int:
        """Total injection configs successfully ``add_pending``'d."""
        return self._n_queued

    @property
    def last_inj_id(self) -> Optional[str]:
        """``inj_id`` of the most recently queued injection, or None."""
        return self._last_inj_id

    @property
    def last_apply_at_specnum(self) -> Optional[int]:
        """``apply_at_specnum`` of the most recently queued injection."""
        return self._last_apply_at_specnum

    @property
    def last_event_unix(self) -> Optional[float]:
        """Unix timestamp of the most recent watch event (raw or queued)."""
        return self._last_event_unix

    @property
    def last_queued_unix(self) -> Optional[float]:
        """Unix timestamp of the most recent successfully queued event."""
        return self._last_queued_unix

    def state(self) -> dict:
        """Snapshot of the watch counters + last-event metadata.

        Used by :class:`dsart.services.corr_fast_mon.CorrFastMonPublisher`
        to ship the inject path's status to ``/mon/corr_rt/<chgroup>/
        corr_fast`` so dashboard consumers can verify all 16 corr nodes
        received an injection without ssh+grep on each one.
        """
        return {
            "inject_n_events": int(self._n_events),
            "inject_n_queued": int(self._n_queued),
            "inject_last_inj_id": self._last_inj_id,
            "inject_last_apply_at_specnum": (
                int(self._last_apply_at_specnum)
                if self._last_apply_at_specnum is not None
                else None
            ),
            "inject_last_event_unix": (
                float(self._last_event_unix)
                if self._last_event_unix is not None
                else None
            ),
            "inject_last_queued_unix": (
                float(self._last_queued_unix)
                if self._last_queued_unix is not None
                else None
            ),
        }

    def start(self) -> None:
        """Open the DsaStore handle (if not provided) + register the
        watch. Idempotent: a second call is a no-op and logs at INFO.
        """
        with self._lock:
            if self._watch_id is not None:
                _LOG.info(
                    "RuntimeInjectWatch already started (key=%s watch_id=%s)",
                    self.key, self._watch_id,
                )
                return
            if self._store is None:
                # Local import so the module is importable on dev
                # nodes / unit-test hosts without dsautils.
                from dsautils.dsa_store import DsaStore
                self._store = DsaStore()
            self._watch_id = self._store.add_watch(self.key, self._on_event)
            _LOG.info(
                "M7.4 Phase 6 runtime-inject watch started: key=%s "
                "watch_id=%s",
                self.key, self._watch_id,
            )

    def stop(self) -> None:
        """Cancel the watch (idempotent). Errors are logged + swallowed
        so a noisy cancel never blocks service shutdown.
        """
        with self._lock:
            if self._watch_id is None:
                return
            try:
                self._store.cancel(self._watch_id)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "RuntimeInjectWatch cancel(%s) failed (continuing): %s",
                    self._watch_id, exc,
                )
            _LOG.info(
                "M7.4 Phase 6 runtime-inject watch stopped: key=%s "
                "events=%d queued=%d",
                self.key, self._n_events, self._n_queued,
            )
            self._watch_id = None

    def _on_event(self, event: Any) -> None:
        """DsaStore watch callback.

        ``DsaStore`` invokes the callback with the *parsed dict*
        already (it's already done ``json.loads`` on the etcd value).
        For symmetry with :func:`handle_inject_payload` (which takes
        bytes/str) we round-trip through ``json.dumps``; this is
        ~µs and keeps the parsing logic in one place.
        """
        with self._lock:
            self._n_events += 1
            self._last_event_unix = time.time()
        try:
            if isinstance(event, (bytes, bytearray, str)):
                payload = event
            else:
                payload = json.dumps(event)
            cfg = handle_inject_payload(payload, injector=self.injector)
        except Exception:  # noqa: BLE001 — defence in depth
            _LOG.exception(
                "RuntimeInjectWatch: error handling event=%r", event,
            )
            return
        if cfg is not None:
            with self._lock:
                self._n_queued += 1
                # T2: capture identifying metadata of the queued probe
                # so the corr_fast mon publisher can ship it to etcd.
                try:
                    self._last_inj_id = (
                        str(cfg.inj_id)
                        if getattr(cfg, "inj_id", None) is not None
                        else None
                    )
                    self._last_apply_at_specnum = (
                        int(cfg.apply_at_specnum)
                        if getattr(cfg, "apply_at_specnum", None) is not None
                        else None
                    )
                except Exception:  # noqa: BLE001 — never sink the loop
                    self._last_inj_id = None
                    self._last_apply_at_specnum = None
                self._last_queued_unix = time.time()
