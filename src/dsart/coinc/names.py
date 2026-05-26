"""Event-name allocator wrapping :func:`event.names.increment_name`.

In production on h23 the allocator reads the previous lastname from
etcd at ``/mon/corr/1/trigger``, calls :func:`event.names.increment_name`
to produce a fresh ``YYMMDD<suffix>`` name, then writes the new name
back as the lastname for the next allocation.

This module is built so it's importable on dev hosts that *don't*
have the legacy ``event`` package on ``sys.path`` — the legacy package
lives only under ``calibration23`` and on h23. The constructor lets
the caller inject a sys.path entry; falling back to a local in-memory
allocator (``YYMMDDaaaa``-then-increment) when:

  * the ``event.names`` import fails, or
  * ``dsautils.dsa_store.DsaStore`` fails to reach etcd, or
  * the caller explicitly opts into ``offline=True``.

The fallback only sees process-local state, so two C2 instances
running in tandem in fallback mode will collide. Production must run
exactly one C2.
"""

from __future__ import annotations

import logging
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "EventNameAllocator",
    "FallbackAllocator",
    "DEFAULT_EVENT_PKG_PATH",
    "ETCD_LASTNAME_KEY",
]


_LOG = logging.getLogger("dsart.coinc.names")

ETCD_LASTNAME_KEY: str = "/mon/corr/1/trigger"
DEFAULT_EVENT_PKG_PATH: Path = Path(
    "/home/ubuntu/proj/dsa110-shell/dsa110-event"
)


def _mjd_to_date_yymmdd(mjd: float) -> str:
    # MJD 40587 = 1970-01-01 (unix epoch). Avoid astropy round-trip on
    # the fallback path so the module stays stdlib-only when astropy
    # isn't available.
    unix = (mjd - 40587.0) * 86400.0
    dt = datetime(1970, 1, 1) + timedelta(seconds=unix)
    return f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}"


class FallbackAllocator:
    """In-memory allocator used when etcd / event.names are unavailable.

    Names look like ``YYMMDD<suffix>`` with a random 4-letter suffix;
    if the suffix collides with the last name issued, we resample
    (matches the legacy ``event.names.increment_name`` semantics).
    """

    def __init__(self, suffix_length: int = 4) -> None:
        self._suffix_length = suffix_length
        self._lastname: Optional[str] = None
        self._rng = random.Random()

    @property
    def lastname(self) -> Optional[str]:
        return self._lastname

    def reset(self) -> None:
        self._lastname = None

    def allocate(self, mjd: float) -> str:
        date = _mjd_to_date_yymmdd(mjd)
        for _ in range(64):  # paranoia: extremely unlikely to spin
            suffix = "".join(
                self._rng.choices(string.ascii_lowercase, k=self._suffix_length)
            )
            candidate = f"{date}{suffix}"
            if candidate != self._lastname:
                self._lastname = candidate
                return candidate
        raise RuntimeError(
            "fallback allocator could not produce a fresh suffix in 64 tries"
        )


class EventNameAllocator:
    """Production allocator: wraps ``event.names.increment_name`` + etcd.

    Init flags:

    * ``etcd_endpoints`` — passed through to ``dsautils.dsa_store.DsaStore``
      via ``DSA110_DSA_STORE_ENDPOINTS`` if provided. ``None`` lets DsaStore
      use its package default.
    * ``etcd_key`` — etcd key holding the lastname. Defaults to
      ``/mon/corr/1/trigger``.
    * ``event_pkg_path`` — directory containing the ``event`` package on
      this host. Defaults to the h23 location. Falsy → no path injection.
    * ``offline`` — if True, skip both the etcd and ``event.names`` paths
      and use :class:`FallbackAllocator` unconditionally. Useful for unit
      tests and dev hosts.
    """

    def __init__(
        self,
        *,
        etcd_endpoints: Optional[str] = None,
        etcd_key: str = ETCD_LASTNAME_KEY,
        event_pkg_path: Optional[Path] = DEFAULT_EVENT_PKG_PATH,
        offline: bool = False,
    ) -> None:
        self._etcd_endpoints = etcd_endpoints
        self._etcd_key = etcd_key
        self._event_pkg_path = (
            Path(event_pkg_path) if event_pkg_path else None
        )
        self._offline = offline
        self._fallback = FallbackAllocator()
        self._increment_name = None  # bound on first allocate
        self._store: Optional[Any] = None  # DsaStore instance, lazy

    # ----- lazy bootstrap ----------------------------------------------

    def _ensure_event_names(self) -> bool:
        """Try to import ``event.names.increment_name``; cache on success."""
        if self._increment_name is not None:
            return True
        if self._offline:
            return False
        if self._event_pkg_path is not None:
            p = str(self._event_pkg_path)
            if self._event_pkg_path.is_dir() and p not in sys.path:
                sys.path.insert(0, p)
        try:
            from event.names import increment_name  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "event.names not importable (%s); falling back to "
                "local allocator. Event names will not be visible to "
                "the legacy archive consumer.",
                exc,
            )
            return False
        self._increment_name = increment_name
        return True

    def _ensure_store(self) -> bool:
        if self._store is not None:
            return True
        if self._offline:
            return False
        try:
            from dsautils.dsa_store import DsaStore  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "dsautils.dsa_store not importable (%s); falling back to "
                "local allocator.",
                exc,
            )
            return False
        try:
            self._store = DsaStore()
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "DsaStore() init failed (%s); falling back to local allocator.",
                exc,
            )
            return False
        return True

    # ----- etcd helpers -------------------------------------------------

    def _read_lastname(self) -> Optional[str]:
        if not self._ensure_store():
            return None
        try:
            d = self._store.get_dict(self._etcd_key)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "etcd get_dict(%s) failed (%s); proceeding with no lastname",
                self._etcd_key, exc,
            )
            return None
        if not isinstance(d, dict) or not d:
            return None
        # Legacy convention: the dict has a single key whose name is the
        # lastname (matching event.names.get_lastname()).
        try:
            lastname, _ = next(iter(d.items()))
        except StopIteration:
            return None
        if not isinstance(lastname, str) or not lastname:
            return None
        return lastname

    def _write_lastname(self, newname: str) -> None:
        if not self._ensure_store():
            return
        try:
            # Mirror legacy semantics: write a single-key dict { newname: ... }
            # so the next ``get_dict().popitem()`` returns the new lastname.
            self._store.put_dict(
                self._etcd_key,
                {newname: {"mjd": None, "set_by": "dsart_c2"}},
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "etcd put_dict(%s, %s) failed (%s); next allocation may "
                "rediscover the same suffix",
                self._etcd_key, newname, exc,
            )

    # ----- public API ---------------------------------------------------

    @property
    def is_online(self) -> bool:
        """True iff *both* event.names and etcd are usable. Best-effort."""
        return self._ensure_event_names() and self._ensure_store()

    def allocate(self, mjd: float) -> str:
        if self._offline:
            return self._fallback.allocate(mjd)
        if not self._ensure_event_names():
            return self._fallback.allocate(mjd)
        lastname = self._read_lastname()
        try:
            assert self._increment_name is not None
            newname = self._increment_name(mjd, lastname=lastname)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "event.names.increment_name(mjd=%s, lastname=%r) raised %s; "
                "falling back",
                mjd, lastname, exc,
            )
            return self._fallback.allocate(mjd)
        if not isinstance(newname, str) or not newname:
            _LOG.warning(
                "event.names.increment_name returned bogus value %r; falling "
                "back",
                newname,
            )
            return self._fallback.allocate(mjd)
        self._write_lastname(newname)
        return newname
