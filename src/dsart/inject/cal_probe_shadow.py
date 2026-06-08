"""Search-side shadow of the dashboard-managed ``/cnf/inject/active/cal_probe_*``
registry, used to exempt calibration-probe candidates from the C1→C2
metering cap.

Background
----------

Per ``configs/dsart_search_rt.yaml::c1.max_candidates_per_block``, the
C1 emitter caps how many candidates each cube ships to C2 (default 8,
narrow-first then bright-first). This bounds the C2 ingest under RFI
floods but has a known failure mode: when an operator-fired calibration
probe lands during a noisy window, it can be silently shed if more than
``cap`` brighter / narrower candidates beat it on the per-cube ordering.
The dashboard's ``fire_calibration_probe`` then returns ``no_match``
without any indication that the probe actually reached the search-side
detector and was discarded by the meter.

This module mirrors the dashboard's ``/cnf/inject/active/cal_probe_*``
prefix on each search-compute process so the C1 emit path can identify
probe candidates BEFORE the metering cap is applied. Matches are
exempted from the cap; the rest of the candidate list is metered as
before.

Design
------

The shadow polls the prefix on a configurable cadence (default 2 s,
matching the existing ``cube_progress`` log interval). Polling — rather
than a watch — is deliberate:

* The dashboard already PUTs new probes a few seconds before the
  corresponding candidate reaches the search side (corr→search latency
  is ~1 s; matcher refresh interval on C2 is 1 s). 2 s polling on the
  search side keeps probes visible long before their candidates arrive.
* No new etcd watch wiring is required — the search-compute service
  already owns a :class:`dsautils.dsa_store.DsaStore` handle for the
  C1-metering and ring publishers.
* The hot-path lookup is a frozen-dict read; no locking on the per-cube
  C1 emit path.

The match predicate intentionally mirrors
:class:`dsart.coinc.inject_match.InjectionMatcher` so search-side and
C2-side bookkeeping agree on which candidate is "the probe":

  1. ``inj_id`` starts with ``cal_probe`` (operator-visible class
     marker; ``fire_calibration_probe`` uses this prefix by default).
  2. ``abs(cand.dm_pc_cc - inj.dm_pc_cm3) / inj.dm_pc_cm3 <
     dm_tol_frac`` (default 5 %).
  3. ``hypot(cand.l_rad - inj.l_rad, cand.m_rad - inj.m_rad) <
     lm_tol_rad`` (default 0.04 rad).
  4. ``cand.snr >= min_observed_snr`` (rejects sub-emit-floor false
     matches; same 6 σ default the C2 matcher uses).
  5. ``now_unix - inj.fired_at_unix < inj.ttl_s + grace_s``.

Width is deliberately NOT a match criterion (same rationale as the
C2-side matcher: injection ``width_samples`` is in native samples,
candidate ``width_samples`` is in search samples; DM-smearing widens
the observed core).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from dsart.coinc.inject_match import (
    ACTIVE_INJECT_PREFIX,
    DEFAULT_DM_TOL_FRAC,
    DEFAULT_LM_TOL_RAD,
    DEFAULT_MIN_OBSERVED_SNR,
    ActiveInjection,
)


__all__ = [
    "CAL_PROBE_INJ_ID_PREFIX",
    "DEFAULT_CAL_PROBE_REFRESH_S",
    "DEFAULT_CAL_PROBE_GRACE_S",
    "CalProbeShadow",
]


_LOG = logging.getLogger("dsart.inject.cal_probe_shadow")


#: Default ``inj_id`` prefix the dashboard's ``fire_calibration_probe``
#: emits ("cal_probe_dm{N}_w{W}_t{TS}"). Anything that starts with this
#: prefix is treated as a calibration probe and exempted from C1
#: metering. Operators who want to mark an injection as a probe outside
#: the dashboard helper can use any inj_id with this prefix.
CAL_PROBE_INJ_ID_PREFIX: str = "cal_probe"


#: How often the shadow re-reads the ``/cnf/inject/active/`` prefix.
#: 2 s is comfortably faster than the corr→search→C1 latency and well
#: above the etcd put rate the dashboard generates (probes are fired
#: at most once / few-seconds in calibration mode).
DEFAULT_CAL_PROBE_REFRESH_S: float = 2.0


#: Extra grace beyond ``ttl_s`` to accept a candidate that arrives just
#: after the dashboard's TTL. Mirrors :data:`inject_match.DEFAULT_EXPIRY_GRACE_S`
#: so the search and C2 sides agree on which probes are still live.
DEFAULT_CAL_PROBE_GRACE_S: float = 5.0


class CalProbeShadow:
    """Lazy snapshot of the active calibration-probe registry.

    Parameters
    ----------
    store:
        Optional pre-built ``DsaStore``. When ``None`` the shadow lazily
        constructs one on first :meth:`refresh`. Tests pass a mock
        DsaStore-like object exposing ``get_dict_prefix``.
    refresh_interval_s:
        Minimum seconds between refresh attempts. The hot path's
        :meth:`is_cal_probe_match` calls :meth:`maybe_refresh`, which
        no-ops when called more often than this; full refresh is paid
        at most once per interval.
    dm_tol_frac, lm_tol_rad, min_observed_snr, grace_s:
        Match-predicate tunables; see module docstring.
    inj_id_prefix:
        Prefix that marks a probe inj_id. The shadow only retains
        registry rows whose ``inj_id`` starts with this prefix; anything
        else (operator-driven target-SNR injections) is left alone.
    time_fn:
        Clock injector (tests pass deterministic clocks).
    """

    def __init__(
        self,
        *,
        store: Optional[Any] = None,
        refresh_interval_s: float = DEFAULT_CAL_PROBE_REFRESH_S,
        dm_tol_frac: float = DEFAULT_DM_TOL_FRAC,
        lm_tol_rad: float = DEFAULT_LM_TOL_RAD,
        min_observed_snr: float = DEFAULT_MIN_OBSERVED_SNR,
        grace_s: float = DEFAULT_CAL_PROBE_GRACE_S,
        inj_id_prefix: str = CAL_PROBE_INJ_ID_PREFIX,
        time_fn: Any = time.time,
    ) -> None:
        if refresh_interval_s <= 0.0:
            raise ValueError(
                f"refresh_interval_s={refresh_interval_s}, expected > 0"
            )
        if dm_tol_frac <= 0.0:
            raise ValueError(f"dm_tol_frac={dm_tol_frac}, expected > 0")
        if lm_tol_rad <= 0.0:
            raise ValueError(f"lm_tol_rad={lm_tol_rad}, expected > 0")
        if min_observed_snr < 0.0:
            raise ValueError(
                f"min_observed_snr={min_observed_snr}, expected >= 0"
            )
        if grace_s < 0.0:
            raise ValueError(f"grace_s={grace_s}, expected >= 0")
        self._store = store
        self._refresh_interval_s = float(refresh_interval_s)
        self._dm_tol_frac = float(dm_tol_frac)
        self._lm_tol_rad = float(lm_tol_rad)
        self._min_observed_snr = float(min_observed_snr)
        self._grace_s = float(grace_s)
        self._inj_id_prefix = str(inj_id_prefix)
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self._snapshot: Dict[str, ActiveInjection] = {}
        self._last_refresh_mono: float = 0.0
        self._n_refresh_ok = 0
        self._n_refresh_fail = 0
        self._first_failure_logged = False
        # Hot-path counters: how many candidates were exempted vs.
        # rejected as no-match. Surfaced by :meth:`stats` so the
        # search-compute service can include them in cube_progress.
        self._n_matched = 0
        self._n_no_match = 0

    @property
    def n_active(self) -> int:
        """Number of live cal probes in the current snapshot."""
        with self._lock:
            return len(self._snapshot)

    @property
    def n_refresh_ok(self) -> int:
        return self._n_refresh_ok

    @property
    def n_refresh_fail(self) -> int:
        return self._n_refresh_fail

    @property
    def n_matched(self) -> int:
        """Total candidates exempted from metering by this shadow."""
        return self._n_matched

    @property
    def n_no_match(self) -> int:
        """Total candidates checked that did NOT match a live probe."""
        return self._n_no_match

    def stats(self) -> Dict[str, int]:
        return {
            "cal_probe_n_active": self.n_active,
            "cal_probe_n_matched": int(self._n_matched),
            "cal_probe_n_no_match": int(self._n_no_match),
            "cal_probe_n_refresh_ok": int(self._n_refresh_ok),
            "cal_probe_n_refresh_fail": int(self._n_refresh_fail),
        }

    def snapshot(self) -> Dict[str, ActiveInjection]:
        """Cloned dict of the current shadow (intended for diagnostics)."""
        with self._lock:
            return dict(self._snapshot)

    def maybe_refresh(self) -> bool:
        """Refresh the shadow if the throttle interval has elapsed.

        Returns True if a refresh actually ran (whether it succeeded or
        not), False if the call was throttled. Hot-path callers
        (cube-loop logging) invoke this at every progress tick; the
        throttle keeps the etcd round-trip cost bounded.
        """
        now_mono = time.monotonic()
        if (now_mono - self._last_refresh_mono) < self._refresh_interval_s:
            return False
        self._last_refresh_mono = now_mono
        self._refresh_now()
        return True

    def _ensure_store(self) -> Optional[Any]:
        if self._store is not None:
            return self._store
        try:
            from dsautils.dsa_store import DsaStore
        except Exception:                                       # noqa: BLE001
            if not self._first_failure_logged:
                _LOG.warning(
                    "CalProbeShadow: dsautils not importable; cal-probe "
                    "metering bypass is DISABLED on this process"
                )
                self._first_failure_logged = True
            return None
        try:
            self._store = DsaStore()
        except Exception as exc:                                # noqa: BLE001
            if not self._first_failure_logged:
                _LOG.warning(
                    "CalProbeShadow: DsaStore() failed (%s); cal-probe "
                    "metering bypass is DISABLED on this process",
                    exc,
                )
                self._first_failure_logged = True
            return None
        return self._store

    def _refresh_now(self) -> None:
        """Read the prefix from etcd and rebuild the snapshot."""
        store = self._ensure_store()
        if store is None:
            self._n_refresh_fail += 1
            return
        try:
            raw = store.get_dict_prefix(ACTIVE_INJECT_PREFIX)
        except Exception as exc:                                # noqa: BLE001
            self._n_refresh_fail += 1
            if not self._first_failure_logged:
                _LOG.warning(
                    "CalProbeShadow: get_dict_prefix(%s) failed: %s "
                    "(subsequent failures silent)",
                    ACTIVE_INJECT_PREFIX, exc,
                )
                self._first_failure_logged = True
            return
        new_snapshot = self._parse_prefix_payload(raw)
        with self._lock:
            self._snapshot = new_snapshot
        self._n_refresh_ok += 1

    def _parse_prefix_payload(
        self, raw: Any,
    ) -> Dict[str, ActiveInjection]:
        """Convert a ``DsaStore.get_dict_prefix`` payload into a live
        ``inj_id -> ActiveInjection`` dict.

        Accepts either the etcd-style mapping ``{key: value_dict}`` or a
        flat iterable of value dicts (different DsaStore versions return
        different shapes). Drops malformed rows silently — a bad
        operator submission must not poison the search loop.
        """
        if raw is None:
            return {}
        rows: Iterable[Any]
        if isinstance(raw, Mapping):
            rows = raw.values()
        else:
            try:
                rows = list(raw)
            except TypeError:
                return {}
        out: Dict[str, ActiveInjection] = {}
        now_unix = float(self._time_fn())
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                inj_id_raw = row.get("inj_id")
            except AttributeError:
                continue
            if not isinstance(inj_id_raw, str):
                continue
            if not inj_id_raw.startswith(self._inj_id_prefix):
                continue
            try:
                inj = ActiveInjection.from_dict(row)
            except (ValueError, TypeError):
                continue
            # Keep only live (non-expired) probes — saves the per-cube
            # match path the TTL check.
            age = now_unix - inj.fired_at_unix
            if age > inj.ttl_s + self._grace_s:
                continue
            out[inj.inj_id] = inj
        return out

    def is_cal_probe_match(
        self,
        *,
        dm_pc_cc: float,
        l_rad: float,
        m_rad: float,
        snr: float,
        now_unix: Optional[float] = None,
    ) -> Optional[str]:
        """Return the matching ``inj_id`` (str) if this candidate is a
        live calibration probe, else ``None``.

        Hot-path safe: a single dict scan + arithmetic per call. Updates
        ``n_matched`` / ``n_no_match`` for surface-able diagnostics.
        """
        snr_f = float(snr)
        if snr_f < self._min_observed_snr:
            self._n_no_match += 1
            return None
        if not math.isfinite(dm_pc_cc) or dm_pc_cc <= 0.0:
            self._n_no_match += 1
            return None
        if now_unix is None:
            now_unix = float(self._time_fn())
        else:
            now_unix = float(now_unix)
        with self._lock:
            snap = self._snapshot
            for inj_id, inj in snap.items():
                if inj.dm_pc_cm3 == 0.0:
                    continue
                if abs(dm_pc_cc - inj.dm_pc_cm3) / abs(inj.dm_pc_cm3) >= (
                    self._dm_tol_frac
                ):
                    continue
                lm = math.hypot(
                    float(l_rad) - inj.l_rad,
                    float(m_rad) - inj.m_rad,
                )
                if lm >= self._lm_tol_rad:
                    continue
                age = now_unix - inj.fired_at_unix
                if age > inj.ttl_s + self._grace_s:
                    continue
                self._n_matched += 1
                return inj_id
        self._n_no_match += 1
        return None
