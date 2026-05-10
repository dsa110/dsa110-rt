"""Minimal etcd watcher for online-injector commands (plan §4.7).

Watches the etcd key prefix ``<NAMESPACE>/cmd/corr/<n>/`` for
``inject``-flavoured commands and routes them into an
:class:`OnlineInjector` instance via ``injector.add_pending(cfg)``.
The namespace prefix follows :file:`PARALLEL_AGENTS.md` §4.6 — the
production path uses ``/cmd/dsart/corr/<n>/...`` while the M3 dev
prefix is ``/cmd/dsart-m3/corr/<n>/...`` (set via
``DSART_ETCD_NAMESPACE_PREFIX=m3``).

Two implementations:

  - :class:`EtcdWatcher` — production-path watcher using the
    ``etcd3`` client. Spawns a background watch thread; the etcd
    client owns the thread lifecycle. Cancel the watch via
    :meth:`EtcdWatcher.stop`.
  - :class:`MockEtcdWatcher` — pure-Python in-memory implementation
    that exposes the same ``start`` / ``stop`` / ``inject_one``
    surface but does not contact a real etcd server. Useful for
    unit tests and for nodes where the operator hasn't deployed
    an etcd cluster yet.

Both watchers share the same payload-handling logic (see
:func:`handle_inject_payload`); only the transport differs.

Production hardening (M7 territory): leases, retries, watch-
reconnect on disconnect, multi-node coordination via etcd's
linearisable read semantics. v1 here just gives the injector a
working CLI / etcd surface so chunk 3d can be exercised end-to-end
during M3 development.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable

from dsart.inject.online import InjectionConfig, OnlineInjector


__all__ = [
    "EtcdWatcher",
    "MockEtcdWatcher",
    "build_inject_key_prefix",
    "handle_inject_payload",
]


_LOG = logging.getLogger("dsart.inject.etcd_watcher")


# ---------------------------------------------------------------------------
# Key-prefix helpers
# ---------------------------------------------------------------------------


def build_inject_key_prefix(corr_index: int) -> str:
    """Return the etcd key prefix to watch for one corr-node injector.

    Prefix is ``<NAMESPACE>/cmd/corr/<n>/`` where ``<NAMESPACE>``
    follows :file:`PARALLEL_AGENTS.md` §4.6 — production unsetenv ⇒
    ``/cmd/dsart``, M3 dev ⇒ ``/cmd/dsart-m3``, M5 ⇒ ``/cmd/dsart-m5``.
    """
    ns = os.environ.get("DSART_ETCD_NAMESPACE_PREFIX", "").strip()
    if ns:
        prefix = f"/cmd/dsart-{ns}/corr/{corr_index}/"
    else:
        prefix = f"/cmd/dsart/corr/{corr_index}/"
    return prefix


# ---------------------------------------------------------------------------
# Payload handler (shared between EtcdWatcher and MockEtcdWatcher)
# ---------------------------------------------------------------------------


def handle_inject_payload(
    payload: bytes | str,
    *,
    injector: OnlineInjector,
) -> InjectionConfig | None:
    """Parse one etcd ``val`` payload and (if it's an inject) queue it.

    The wire schema is a JSON object of the form

    .. code-block:: json

        {
          "cmd": "inject",
          "val": {
            "inj_id": "...",
            "l_rad": 0.05,
            "m_rad": 0.0,
            "dm_pc_cm3": 200.0,
            "fluence_jy_ms": 10.0,
            "width_samples": 8,
            "profile": "gaussian",
            "apply_at_specnum": 17254656
          }
        }

    Non-``inject`` commands are silently dropped (other ``cmd:`` types
    — ``prepare``, ``start``, ``stop``, ``reload_cal`` — are handled
    by the corr-fast service's own command handler, not by this
    watcher).

    Returns the parsed :class:`InjectionConfig` for caller logging,
    or ``None`` if the payload was non-``inject`` or malformed (the
    function logs the malformed-payload reason at WARNING; never
    raises on parse errors so a bad operator submission cannot crash
    the watcher).
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            _LOG.warning("etcd payload not utf-8: %s", exc)
            return None
    else:
        text = payload

    try:
        wire = json.loads(text)
    except json.JSONDecodeError as exc:
        _LOG.warning("etcd payload not JSON: %s", exc)
        return None

    if not isinstance(wire, dict):
        _LOG.warning("etcd payload not a JSON object; type=%s", type(wire).__name__)
        return None

    cmd = wire.get("cmd")
    if cmd != "inject":
        return None

    val = wire.get("val")
    if val is None:
        _LOG.warning("inject payload missing 'val' key")
        return None

    try:
        # Accept val as either a nested JSON dict (preferred) or a
        # JSON-encoded string (some etcd CLIs do that).
        if isinstance(val, str):
            cfg = InjectionConfig.from_json(val)
        else:
            cfg = InjectionConfig.from_json(json.dumps(val))
    except (ValueError, TypeError) as exc:
        _LOG.warning("inject payload could not parse InjectionConfig: %s", exc)
        return None

    try:
        injector.add_pending(cfg)
    except ValueError as exc:
        _LOG.warning("inject payload rejected by injector: %s", exc)
        return None

    _LOG.info(
        "inject queued: id=%s (l, m)=(%.5f, %.5f) DM=%.2f fluence=%.2f Jy.ms",
        cfg.inj_id, cfg.l_rad, cfg.m_rad, cfg.dm_pc_cm3, cfg.fluence_jy_ms,
    )
    return cfg


# ---------------------------------------------------------------------------
# MockEtcdWatcher (in-memory; for tests + etcd-less dev nodes)
# ---------------------------------------------------------------------------


@dataclass
class MockEtcdWatcher:
    """In-memory stand-in for :class:`EtcdWatcher`.

    Same public surface (``start`` / ``stop`` / ``inject_one``) but
    doesn't contact etcd. Useful for unit tests and for nodes where
    the operator hasn't deployed an etcd cluster yet. Production etcd
    integration is M7 territory; this Mock keeps chunk 3d unblocked.

    Parameters
    ----------
    injector : OnlineInjector
        The injector to route commands into.
    corr_index : int
        Per-host corr-node index; used only by :func:`build_inject_key_prefix`
        for the ``key_prefix`` attribute.
    """

    injector: OnlineInjector
    corr_index: int = 0
    _running: bool = False

    @property
    def key_prefix(self) -> str:
        return build_inject_key_prefix(self.corr_index)

    def start(self) -> None:
        """No-op for the mock; kept for API parity with EtcdWatcher."""
        self._running = True

    def stop(self) -> None:
        """No-op for the mock; kept for API parity with EtcdWatcher."""
        self._running = False

    def inject_one(self, payload: bytes | str) -> InjectionConfig | None:
        """Synchronously route one ``val`` payload into the injector.

        Returns the parsed :class:`InjectionConfig` (or ``None`` on
        malformed / non-inject payload) for test-side assertion.
        """
        return handle_inject_payload(payload, injector=self.injector)


# ---------------------------------------------------------------------------
# EtcdWatcher (real etcd3 client)
# ---------------------------------------------------------------------------


class EtcdWatcher:
    """Production etcd watcher (etcd3 client).

    Starts a background watch on
    ``build_inject_key_prefix(corr_index)``; each PUT event triggers
    :func:`handle_inject_payload`.

    Lifecycle:
      1. Construct with ``injector`` + ``corr_index`` + an etcd
         endpoint. The constructor lazily imports ``etcd3`` so this
         module can be imported on nodes without the dep.
      2. Call :meth:`start` to issue the watch (returns immediately;
         the etcd3 client owns the background thread).
      3. Call :meth:`stop` at service shutdown to cancel the watch.

    The etcd client is a process-wide resource; if the caller wants
    to share an existing client, pass it explicitly via
    ``etcd_client``.

    Parameters
    ----------
    injector : OnlineInjector
        Where to route ``cmd: inject`` commands.
    corr_index : int
        Per-host corr-node index (the ``<n>`` in ``/cmd/corr/<n>``).
    etcd_host, etcd_port : str, int
        etcd endpoint. Defaults match the legacy DSA-110 etcd at
        ``etcdv3service.sas.pvt:2379`` but are overridable.
    etcd_client : object, optional
        Pre-built ``etcd3.client(...)``. When supplied, ``etcd_host``
        + ``etcd_port`` are ignored. Caller owns the client lifecycle.
    """

    def __init__(
        self,
        injector: OnlineInjector,
        corr_index: int,
        *,
        etcd_host: str = "etcdv3service.sas.pvt",
        etcd_port: int = 2379,
        etcd_client: object | None = None,
    ) -> None:
        self.injector = injector
        self.corr_index = int(corr_index)
        self.etcd_host = etcd_host
        self.etcd_port = etcd_port
        self._client = etcd_client
        self._owns_client = etcd_client is None
        self._watch_id: int | None = None
        self._lock = threading.Lock()

    @property
    def key_prefix(self) -> str:
        return build_inject_key_prefix(self.corr_index)

    def start(self) -> None:
        """Start the background watch on :attr:`key_prefix`."""
        with self._lock:
            if self._watch_id is not None:
                _LOG.info("EtcdWatcher already started (watch_id=%s)",
                          self._watch_id)
                return
            if self._client is None:
                try:
                    import etcd3
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "EtcdWatcher requires the `etcd3` python package; "
                        "install it or use MockEtcdWatcher instead"
                    ) from exc
                self._client = etcd3.client(
                    host=self.etcd_host, port=self.etcd_port,
                )
            cancel = getattr(self._client, "add_watch_prefix_callback", None)
            if cancel is None:
                raise RuntimeError(
                    "etcd client has no add_watch_prefix_callback method; "
                    "incompatible etcd3 version?"
                )
            self._watch_id = cancel(
                self.key_prefix, self._on_watch_event,
            )
            _LOG.info("EtcdWatcher started on prefix=%s (watch_id=%s)",
                      self.key_prefix, self._watch_id)

    def stop(self) -> None:
        """Cancel the watch + (if we own the client) close it."""
        with self._lock:
            if self._watch_id is not None and self._client is not None:
                try:
                    self._client.cancel_watch(self._watch_id)
                except Exception as exc:  # pragma: no cover
                    _LOG.warning("etcd cancel_watch failed: %s", exc)
                self._watch_id = None
            if self._owns_client and self._client is not None:
                try:
                    self._client.close()
                except Exception as exc:  # pragma: no cover
                    _LOG.warning("etcd client close failed: %s", exc)
                self._client = None

    # ---- internal callback (etcd3 thread context) ----

    def _on_watch_event(self, event: object) -> None:
        """etcd3 callback: dispatch each PUT event to handle_inject_payload."""
        events = getattr(event, "events", None)
        if events is None:
            # etcd3 sometimes delivers a single Event directly.
            events = [event]
        for ev in events:
            value = getattr(ev, "value", None)
            if value is None:
                continue
            try:
                handle_inject_payload(value, injector=self.injector)
            except Exception:  # pragma: no cover — defence-in-depth
                _LOG.exception("EtcdWatcher: error handling payload")
