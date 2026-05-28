"""C1-candidate ↔ active-injection matcher (M7.4 Phase 6c.A).

The coincidencer aggregates C1 rows from all 8 search halves; this
module pairs them with the dashboard-published "active injection"
registry (``/cnf/inject/active/<inj_id>``) and emits per-match events
to ``/mon/dsart/inject/matches/<inj_id>``. Two consumers use it:

  * The dashboard's bootstrap-SNR calibration loop fires a known-
    fluence probe, polls the match key for the resulting observed_snr,
    and computes ``K = observed_snr × sqrt(width_samples / fluence)``.
    Future injections at a target SNR derive their fluence via
    ``fluence = width_samples × (target_snr / K)^2``.
  * The dashboard's Bursts panel reads the per-event C1 CSV; the
    coincidencer stamps each matched row's ``inj_id`` column with the
    matched id, so candidates can be visibly labelled as injections.

Why a separate module
---------------------

The coincidencer hot-path is already busy with windowing + graph
clustering + criteria evaluation. Match logic + registry refresh +
publish-to-etcd live here so the coincidencer change is a single
``on_c1_row`` hook + one background refresh in the existing mon-publish
loop.

Match semantics
---------------

A C1 row matches an :class:`ActiveInjection` when

  1. ``abs(row.dm_pc_cc - inj.dm_pc_cm3) / inj.dm_pc_cm3 < dm_tol_frac``
     (default 5 %; permissive enough to catch the coarse + fine DM
     quantisation residual the search-side de-disperser leaves), AND

  2. ``hypot(row.l_rad - inj.l_rad, row.m_rad - inj.m_rad) <
     lm_tol_rad`` (default 0.005 rad ≈ 17 arcmin — much wider than one
     image pixel so the C1 peak that doesn't quite land on the
     injection grid cell still matches), AND

  3. The injection has not yet expired
     (``now_unix - inj.fired_at_unix < inj.ttl_s``).

Multiple matching rows for the same injection are allowed; the matcher
keeps a per-``inj_id`` "best so far" (max ``snr``) and a rolling
history of the most recent N (default 16) matches. The published
payload at ``/mon/dsart/inject/matches/<inj_id>`` carries both, so the
dashboard can either read the brightest match (calibration) or scan
the history (sanity check that the injection produced a coherent
spike rather than just one outlier).

The matcher is **thread-safe** for the single-producer (C2's batch
handler) + single-refresher (mon-publish loop) pattern the
coincidencer uses today. The registry snapshot is taken once per
refresh and walked without locking by the producer; the per-``inj_id``
``_best`` dict is guarded by ``_lock``.

K-inference formula
-------------------

The voltage-domain injector materialises each contribution with a
peak amplitude ``amp = sqrt(|fluence_jy_ms| / width_samples)``; after
de-dispersion + matched-filter the detected SNR is linear in ``amp``
(plus an additive noise floor we ignore for calibration). We
parameterise

    observed_snr ≈ K(dm, width) × sqrt(fluence / width_samples)

so

    K = observed_snr / sqrt(fluence / width_samples)
      = observed_snr × sqrt(width_samples / fluence)

and the inverse is

    fluence = width_samples × (target_snr / K)^2

``K`` is published inside the match payload (along with the source
``fluence`` and ``observed_snr``) so the dashboard's
:class:`inject_calibration.CalibrationStore` can persist it after the
first probe. K is DM- and width-dependent (the DM-band sets the
de-disperser path; the width sets the matched-filter kernel), so the
calibration store keys on a coarse ``(dm_band, width)`` bucket.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

LOG = logging.getLogger("dsart.coinc.inject_match")


__all__ = [
    "ACTIVE_INJECT_PREFIX",
    "MATCH_EVENT_PREFIX",
    "DEFAULT_DM_TOL_FRAC",
    "DEFAULT_LM_TOL_RAD",
    "DEFAULT_REGISTRY_REFRESH_S",
    "DEFAULT_HISTORY_DEPTH",
    "ActiveInjection",
    "MatchResult",
    "compute_k_inferred",
    "InjectionMatcher",
    "build_active_inject_key",
    "build_match_event_key",
]


# ---------------------------------------------------------------------------
# Constants — single source of truth, mirrored on the dashboard side via
# ``inject_calibration`` so the two halves stay in lockstep.
# ---------------------------------------------------------------------------

#: etcd prefix for the dashboard-managed "active injection" registry.
ACTIVE_INJECT_PREFIX: str = "/cnf/inject/active/"

#: etcd prefix where this matcher publishes one event per
#: ``(inj_id, observed_snr)`` improvement.
MATCH_EVENT_PREFIX: str = "/mon/dsart/inject/matches/"

#: DM tolerance for a match (fractional). 5 % covers the coarse + fine
#: DM quantisation residuals at the injection DMs we test (150 / 500
#: / 1500 pc/cc); tighten to 2 % if cross-talk between adjacent DM
#: bands becomes a problem.
DEFAULT_DM_TOL_FRAC: float = 0.05

#: l/m tolerance for a match (rad). 0.04 rad ≈ 2.3° — generous enough
#: to absorb the geometric offset we observe on the live fleet (injections
#: at boresight l=m=0 surface in C1 candidates at l≈m≈0.02 rad due to
#: imager-pixelisation / phase-tracking residuals) while still rejecting
#: the bulk of off-axis RFI (the RFI cluster sits at l~0.025, m~0.035
#: and dm>1500, which is outside the dm_tol_frac window anyway).
DEFAULT_LM_TOL_RAD: float = 0.04

#: How often the matcher refreshes its registry snapshot from etcd.
#: 1 s keeps the matcher responsive to the dashboard's PUT without
#: flooding etcd; the matcher tolerates an injection landing at a row
#: that arrives < 1 s after the PUT (the next refresh catches up).
DEFAULT_REGISTRY_REFRESH_S: float = 1.0

#: Rolling history depth per ``inj_id`` retained in the published match
#: payload. 16 is plenty for the calibration use case (a single
#: ~32-sample-wide injection typically produces 1-4 candidates as it
#: sweeps through cubes) and keeps the etcd payload under 8 KB.
DEFAULT_HISTORY_DEPTH: int = 16


def build_active_inject_key(inj_id: str) -> str:
    """Canonical ``/cnf/inject/active/<inj_id>`` key."""
    if not inj_id:
        raise ValueError("inj_id must be non-empty")
    return f"{ACTIVE_INJECT_PREFIX}{inj_id}"


def build_match_event_key(inj_id: str) -> str:
    """Canonical ``/mon/dsart/inject/matches/<inj_id>`` key."""
    if not inj_id:
        raise ValueError("inj_id must be non-empty")
    return f"{MATCH_EVENT_PREFIX}{inj_id}"


# ---------------------------------------------------------------------------
# Data classes — the on-the-wire schema of the registry + match events.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveInjection:
    """One ``/cnf/inject/active/<inj_id>`` entry.

    The dashboard PUTs this dict at inject-fire time. The matcher
    reads it back on every refresh, deserialises into this dataclass,
    and matches incoming C1 rows against it.

    Attributes
    ----------
    inj_id:
        Operator-supplied id; primary key into the registry.
    dm_pc_cm3:
        DM the injection was synthesised at (pc / cm³).
    l_rad, m_rad:
        Direction cosines of the synthetic source (rad).
    width_samples:
        Injection FWHM in native samples; pinned to the
        :class:`dsart.inject.online.InjectionConfig` value the
        dashboard sent.
    fluence_jy_ms:
        Injection fluence (Jy·ms). The matcher publishes this back
        in the match payload so the dashboard's K-inference is
        self-contained.
    apply_at_specnum:
        corr_fast service-start epoch specnum at the injection peak;
        echoed for diagnostics.
    fired_at_unix:
        Wall-clock at the dashboard's PUT. Used to age out expired
        injections.
    ttl_s:
        How long after ``fired_at_unix`` the matcher will accept
        rows. Default 60 s (set on the dashboard side).
    fired_by:
        Operator label or ``"calibration_probe"``. Echoed in audit.
    target_snr:
        Optional. The SNR the dashboard was targeting (``None`` for a
        calibration probe; a number for a target-SNR-driven inject).
    """

    inj_id: str
    dm_pc_cm3: float
    l_rad: float
    m_rad: float
    width_samples: int
    fluence_jy_ms: float
    apply_at_specnum: int
    fired_at_unix: float
    ttl_s: float = 60.0
    fired_by: str = "unknown"
    target_snr: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ActiveInjection":
        """Construct from a permissive dict (etcd JSON payload)."""
        if not isinstance(d, Mapping):
            raise ValueError(
                f"ActiveInjection.from_dict: expected mapping, got "
                f"{type(d).__name__}"
            )
        try:
            inj_id = str(d["inj_id"])
            dm = float(d["dm_pc_cm3"])
            l_rad = float(d["l_rad"])
            m_rad = float(d["m_rad"])
            width = int(d["width_samples"])
            fluence = float(d["fluence_jy_ms"])
            apply_at = int(d["apply_at_specnum"])
            fired_at = float(d["fired_at_unix"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"ActiveInjection.from_dict: bad payload {d!r}: {exc}"
            ) from exc
        ttl = float(d.get("ttl_s", 60.0))
        fired_by = str(d.get("fired_by", "unknown"))
        target_snr_raw = d.get("target_snr")
        target_snr: Optional[float] = None
        if target_snr_raw is not None:
            try:
                target_snr = float(target_snr_raw)
            except (TypeError, ValueError):
                target_snr = None
        if not inj_id:
            raise ValueError("ActiveInjection.from_dict: empty inj_id")
        if width < 1:
            raise ValueError(
                f"ActiveInjection.from_dict: width_samples={width} < 1"
            )
        if not math.isfinite(dm) or dm == 0.0:
            raise ValueError(
                f"ActiveInjection.from_dict: dm_pc_cm3={dm} not usable"
            )
        return cls(
            inj_id=inj_id,
            dm_pc_cm3=dm,
            l_rad=l_rad,
            m_rad=m_rad,
            width_samples=width,
            fluence_jy_ms=fluence,
            apply_at_specnum=apply_at,
            fired_at_unix=fired_at,
            ttl_s=ttl,
            fired_by=fired_by,
            target_snr=target_snr,
        )

    def is_expired(self, now_unix: float) -> bool:
        """True iff the matcher should stop accepting rows for this id."""
        return now_unix > self.fired_at_unix + self.ttl_s


@dataclass(frozen=True)
class MatchResult:
    """One C1 row matched to one :class:`ActiveInjection`.

    Stored in the matcher's per-``inj_id`` best + history and emitted
    in the published payload at ``/mon/dsart/inject/matches/<inj_id>``.
    """

    inj_id: str
    observed_snr: float
    observed_dm_pc_cm3: float
    observed_l_rad: float
    observed_m_rad: float
    observed_width_samples: int
    observed_event_specnum: int
    observed_kernel_id: str
    observed_search_node_id: int
    observed_gpu_half: int
    observed_mjd: float
    matched_at_unix: float
    fluence_used_jy_ms: float
    K_inferred: float
    inject_dm_pc_cm3: float
    inject_l_rad: float
    inject_m_rad: float
    inject_width_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# K-inference (pure helper; pulled out for unit testing in isolation).
# ---------------------------------------------------------------------------


def compute_k_inferred(
    *, observed_snr: float, fluence_jy_ms: float, width_samples: int,
) -> float:
    """Return ``K = observed_snr × sqrt(width / fluence)``.

    Returns ``nan`` if any input is non-finite, non-positive, or the
    inferred K is non-finite. The dashboard treats a non-finite K as
    "calibration not usable" and refuses to translate target_snr →
    fluence in that case.
    """
    if not (math.isfinite(observed_snr) and math.isfinite(fluence_jy_ms)):
        return float("nan")
    if observed_snr <= 0.0 or fluence_jy_ms <= 0.0 or width_samples <= 0:
        return float("nan")
    try:
        return float(observed_snr) * math.sqrt(
            float(width_samples) / float(fluence_jy_ms)
        )
    except (ValueError, ZeroDivisionError):
        return float("nan")


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------


class _Store:
    """Duck-typed DsaStore surface used by the matcher.

    Tests pass a ``MagicMock`` or a plain object with ``get_dict`` /
    ``put_dict`` / ``get_etcd`` methods. Production uses
    :class:`dsautils.dsa_store.DsaStore`.
    """

    def get_dict(self, key: str) -> Any:  # pragma: no cover - protocol
        raise NotImplementedError

    def put_dict(self, key: str, payload: Mapping[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_etcd(self) -> Any:  # pragma: no cover - protocol
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


class InjectionMatcher:
    """C1-row ↔ active-injection matcher.

    Designed to be owned by the coincidencer. The coincidencer calls

      * :meth:`refresh_if_due` from its mon-publish loop (every 1 s),
      * :meth:`try_match` from its C1 batch handler on every row.

    The matcher publishes match events to etcd inside ``try_match``
    whenever the per-``inj_id`` best improves; the publish is best-
    effort (etcd errors are logged + swallowed) so a transient outage
    never blocks the C2 hot path.

    Construction
    ------------

    .. code-block:: python

        from dsart.coinc.inject_match import InjectionMatcher

        matcher = InjectionMatcher(
            store=dsa_store,
            refresh_s=1.0,
        )

    The matcher takes a duck-typed DsaStore (anything with
    ``get_dict`` / ``put_dict`` / ``get_etcd``). A test fixture
    can pass a :class:`unittest.mock.MagicMock` with the right
    interface.
    """

    def __init__(
        self,
        *,
        store: Any,
        refresh_s: float = DEFAULT_REGISTRY_REFRESH_S,
        dm_tol_frac: float = DEFAULT_DM_TOL_FRAC,
        lm_tol_rad: float = DEFAULT_LM_TOL_RAD,
        history_depth: int = DEFAULT_HISTORY_DEPTH,
        time_fn: Any = time.time,
        active_inject_prefix: str = ACTIVE_INJECT_PREFIX,
    ) -> None:
        if refresh_s <= 0.0:
            raise ValueError(f"refresh_s={refresh_s} must be positive")
        if dm_tol_frac <= 0.0:
            raise ValueError(f"dm_tol_frac={dm_tol_frac} must be positive")
        if lm_tol_rad <= 0.0:
            raise ValueError(f"lm_tol_rad={lm_tol_rad} must be positive")
        if history_depth < 1:
            raise ValueError(f"history_depth={history_depth} must be >= 1")
        self._store = store
        self._refresh_s = float(refresh_s)
        self._dm_tol_frac = float(dm_tol_frac)
        self._lm_tol_rad = float(lm_tol_rad)
        self._history_depth = int(history_depth)
        self._time = time_fn
        self._active_inject_prefix = str(active_inject_prefix)
        # Per-(inj_id) state — guarded by _lock.
        self._lock = threading.Lock()
        self._active: Dict[str, ActiveInjection] = {}
        self._best: Dict[str, MatchResult] = {}
        self._history: Dict[str, Deque[MatchResult]] = {}
        self._last_refresh_unix: float = 0.0
        # Counters — exposed via snapshot() for the coincidencer mon-points.
        self._n_rows_checked: int = 0
        self._n_matches: int = 0
        self._n_best_improved: int = 0
        self._n_publish_ok: int = 0
        self._n_publish_fail: int = 0
        self._n_refresh_ok: int = 0
        self._n_refresh_fail: int = 0

    # ---- public hot-path ----

    def refresh_if_due(self, *, force: bool = False) -> bool:
        """Refresh the active-injection registry from etcd if due.

        Returns True iff a refresh was performed this call.
        Refreshes on a wall-clock cadence (``refresh_s``); the caller
        can pass ``force=True`` after a known mutation (e.g. the
        coincidencer just heard an inject-fire log on its event bus).
        """
        now = float(self._time())
        if not force and (now - self._last_refresh_unix) < self._refresh_s:
            return False
        try:
            entries = self._read_active_entries()
        except Exception as exc:  # noqa: BLE001
            self._n_refresh_fail += 1
            LOG.warning(
                "InjectionMatcher refresh failed: %s", exc, exc_info=False,
            )
            self._last_refresh_unix = now
            return False
        self._n_refresh_ok += 1
        self._last_refresh_unix = now
        with self._lock:
            old_ids = set(self._active.keys())
            new_ids = {e.inj_id for e in entries}
            self._active = {e.inj_id: e for e in entries}
            # Drop best/history for ids no longer in the registry.
            for inj_id in old_ids - new_ids:
                self._best.pop(inj_id, None)
                self._history.pop(inj_id, None)
        return True

    def try_match(
        self,
        *,
        snr: float,
        dm_pc_cc: float,
        l_rad: float,
        m_rad: float,
        width_samples: int,
        event_specnum: int,
        kernel_id: str,
        search_node_id: int,
        gpu_half: int,
        mjd: float,
    ) -> Optional[MatchResult]:
        """Match one C1 row against the active registry; return best
        :class:`MatchResult` on a hit (else ``None``).

        Side effects: when this row improves the per-``inj_id`` best,
        the matcher PUTs an updated match-event payload to
        ``/mon/dsart/inject/matches/<inj_id>``. Publish failures are
        logged and swallowed.
        """
        self._n_rows_checked += 1
        now = float(self._time())
        # Snapshot the active list — refresh is single-threaded against
        # try_match in production (both run on the coincidencer asyncio
        # loop) but the lock is cheap insurance.
        with self._lock:
            actives = list(self._active.values())
        best_match: Optional[MatchResult] = None
        for inj in actives:
            if inj.is_expired(now):
                continue
            if not self._row_matches(
                inj=inj,
                dm_pc_cc=dm_pc_cc,
                l_rad=l_rad,
                m_rad=m_rad,
            ):
                continue
            K = compute_k_inferred(
                observed_snr=snr,
                fluence_jy_ms=inj.fluence_jy_ms,
                width_samples=inj.width_samples,
            )
            mr = MatchResult(
                inj_id=inj.inj_id,
                observed_snr=float(snr),
                observed_dm_pc_cm3=float(dm_pc_cc),
                observed_l_rad=float(l_rad),
                observed_m_rad=float(m_rad),
                observed_width_samples=int(width_samples),
                observed_event_specnum=int(event_specnum),
                observed_kernel_id=str(kernel_id),
                observed_search_node_id=int(search_node_id),
                observed_gpu_half=int(gpu_half),
                observed_mjd=float(mjd),
                matched_at_unix=now,
                fluence_used_jy_ms=float(inj.fluence_jy_ms),
                K_inferred=float(K),
                inject_dm_pc_cm3=float(inj.dm_pc_cm3),
                inject_l_rad=float(inj.l_rad),
                inject_m_rad=float(inj.m_rad),
                inject_width_samples=int(inj.width_samples),
            )
            self._n_matches += 1
            improved = self._update_best_and_history(mr)
            if improved:
                self._publish_match(inj.inj_id)
                if best_match is None or mr.observed_snr > best_match.observed_snr:
                    best_match = mr
            elif best_match is None:
                best_match = mr
        return best_match

    def matched_inj_id(
        self,
        *,
        dm_pc_cc: float,
        l_rad: float,
        m_rad: float,
    ) -> Optional[str]:
        """Return the first matching inj_id (no SNR / best update).

        Cheap, side-effect-free helper used by the coincidencer to
        stamp the C1 CSV's ``inj_id`` column without re-running the
        full :meth:`try_match` pipeline.
        """
        now = float(self._time())
        with self._lock:
            actives = list(self._active.values())
        for inj in actives:
            if inj.is_expired(now):
                continue
            if self._row_matches(
                inj=inj,
                dm_pc_cc=dm_pc_cc,
                l_rad=l_rad,
                m_rad=m_rad,
            ):
                return inj.inj_id
        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-ready snapshot of matcher state + counters.

        Folded into ``/mon/c2/h23`` by the coincidencer so an operator
        can see at a glance how many injections are active, how many
        rows have been checked, and how many matches have been
        published.
        """
        with self._lock:
            active = [
                {
                    "inj_id": inj.inj_id,
                    "dm_pc_cm3": inj.dm_pc_cm3,
                    "l_rad": inj.l_rad,
                    "m_rad": inj.m_rad,
                    "width_samples": inj.width_samples,
                    "fluence_jy_ms": inj.fluence_jy_ms,
                    "target_snr": inj.target_snr,
                    "fired_at_unix": inj.fired_at_unix,
                    "ttl_s": inj.ttl_s,
                }
                for inj in self._active.values()
            ]
            best = {
                inj_id: {
                    "observed_snr": mr.observed_snr,
                    "K_inferred": mr.K_inferred,
                    "matched_at_unix": mr.matched_at_unix,
                }
                for inj_id, mr in self._best.items()
            }
        return {
            "active_count": len(active),
            "active": active,
            "best": best,
            "rows_checked": int(self._n_rows_checked),
            "matches": int(self._n_matches),
            "best_improved": int(self._n_best_improved),
            "publish_ok": int(self._n_publish_ok),
            "publish_fail": int(self._n_publish_fail),
            "refresh_ok": int(self._n_refresh_ok),
            "refresh_fail": int(self._n_refresh_fail),
            "last_refresh_unix": float(self._last_refresh_unix),
        }

    # ---- internals ----

    def _row_matches(
        self,
        *,
        inj: ActiveInjection,
        dm_pc_cc: float,
        l_rad: float,
        m_rad: float,
    ) -> bool:
        """DM + l/m tolerance check (pure)."""
        try:
            dm_diff = abs(float(dm_pc_cc) - inj.dm_pc_cm3)
            dm_tol = self._dm_tol_frac * abs(inj.dm_pc_cm3)
        except (TypeError, ValueError):
            return False
        if dm_diff > dm_tol:
            return False
        try:
            lm_dist = math.hypot(
                float(l_rad) - inj.l_rad,
                float(m_rad) - inj.m_rad,
            )
        except (TypeError, ValueError):
            return False
        return lm_dist <= self._lm_tol_rad

    def _update_best_and_history(self, mr: MatchResult) -> bool:
        """Push ``mr`` into history; bump best if SNR improved.

        Returns True iff the per-``inj_id`` best changed.
        """
        with self._lock:
            hist = self._history.setdefault(
                mr.inj_id, deque(maxlen=self._history_depth),
            )
            hist.append(mr)
            prev = self._best.get(mr.inj_id)
            if prev is None or mr.observed_snr > prev.observed_snr:
                self._best[mr.inj_id] = mr
                self._n_best_improved += 1
                return True
            return False

    def _publish_match(self, inj_id: str) -> None:
        """PUT the current best + history for ``inj_id`` to etcd."""
        with self._lock:
            best = self._best.get(inj_id)
            hist = list(self._history.get(inj_id, ()))
            inj = self._active.get(inj_id)
        if best is None or inj is None:
            return
        payload: Dict[str, Any] = {
            "inj_id": inj_id,
            "best": best.to_dict(),
            "history": [m.to_dict() for m in hist],
            "n_matches": len(hist),
            "published_at_unix": float(self._time()),
            "active": {
                "dm_pc_cm3": inj.dm_pc_cm3,
                "l_rad": inj.l_rad,
                "m_rad": inj.m_rad,
                "width_samples": inj.width_samples,
                "fluence_jy_ms": inj.fluence_jy_ms,
                "target_snr": inj.target_snr,
                "fired_at_unix": inj.fired_at_unix,
                "ttl_s": inj.ttl_s,
                "fired_by": inj.fired_by,
                "apply_at_specnum": inj.apply_at_specnum,
            },
        }
        key = build_match_event_key(inj_id)
        try:
            self._store.put_dict(key, payload)
            self._n_publish_ok += 1
        except Exception as exc:  # noqa: BLE001
            self._n_publish_fail += 1
            LOG.warning(
                "InjectionMatcher publish %s failed: %s", key, exc,
                exc_info=False,
            )

    def _read_active_entries(self) -> List[ActiveInjection]:
        """Read every ``/cnf/inject/active/<inj_id>`` key from etcd.

        Uses the raw ``etcd3`` client surface (``store.get_etcd()``);
        each value is a JSON dict the dashboard PUT in :func:`send_inject`.
        Bad rows are skipped with a warning so a single malformed
        entry doesn't poison the whole registry.
        """
        import json

        # Duck-type the dashboard's ControlStore (lazy _ensure / _store).
        es = self._store
        if hasattr(self._store, "_ensure") and hasattr(self._store, "_store"):
            try:
                self._store._ensure()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"_ensure failed: {exc}") from exc
            es = self._store._store
        # The DsaStore wraps an etcd3 client; we use get_prefix for the
        # bulk read. Tests can hand us a mock that already exposes
        # ``get_etcd`` directly.
        client = es.get_etcd()
        pairs = list(client.get_prefix(self._active_inject_prefix))
        out: List[ActiveInjection] = []
        for value, _meta in pairs:
            try:
                if isinstance(value, (bytes, bytearray)):
                    payload = json.loads(value.decode("utf-8"))
                elif isinstance(value, str):
                    payload = json.loads(value)
                elif isinstance(value, Mapping):
                    payload = dict(value)
                else:
                    LOG.warning(
                        "InjectionMatcher: unknown value type %s; "
                        "skipping",
                        type(value).__name__,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "InjectionMatcher: bad active-inject JSON: %s",
                    exc,
                )
                continue
            try:
                out.append(ActiveInjection.from_dict(payload))
            except ValueError as exc:
                LOG.warning(
                    "InjectionMatcher: rejecting active-inject %r: %s",
                    payload, exc,
                )
                continue
        return out
