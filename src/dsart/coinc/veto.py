"""C2 trigger-suppression helpers: cluster-rate limiter + sidereal (l,m) veto.

Two independent dump-suppression mechanisms layered on top of the YAML
:class:`dsart.coinc.criteria.CriteriaEvaluator`. Both only ever *suppress*
a ``dump_all_gpus`` action — they never create a trigger.

1. :class:`ClusterRateLimiter` — a sliding-window count of EVERY cluster
   the coincidencer evaluates (fleet-wide; one C2 sees all 8 search
   nodes). When the count in the trailing ``window_s`` exceeds
   ``max_clusters`` the sky is in an RFI storm and dump-triggering
   actions are suppressed until the rate falls back under the cap. Uses
   a monotonic clock (rate is a now-property; restart resets it).

2. :class:`SiderealVetoRegistry` — a registry of sidereally-stationary
   sky positions that keep producing dump-worthy clusters. A real FRB
   does not repeat at a fixed (l, m); persistent continuum / RFI sources
   do. When ``>= min_hits`` dump-eligible clusters land within
   ``tol_rad`` of each other spanning ``>= min_span_s`` seconds, the
   position is promoted to an ACTIVE veto and subsequent dump-eligible
   clusters there are suppressed. Vetoes roll off ``expiry_s`` after
   their last hit. The registry is serialisable to / from etcd so it
   survives a C2 restart, is published for the sky-monitor display, and
   is clearable from the dashboard Control tab. Uses wall-clock unix
   time (24 h expiry must survive process restarts).

3. :func:`dm_comb_detected` — a broadband-RFI ("DM comb") veto. A real
   dispersed burst occupies ONE dispersion measure (a single cluster,
   possibly with a modest dm_iqr from adjacent trials). An impulsive
   broadband terrestrial signal has power at every dispersion delay, so
   it lights up many trial DMs *simultaneously at the same (l, m)* — and
   because those firing DM trials are often non-adjacent, the time-only
   connected-components clusterer splits them into several separate
   clusters. The per-cluster dm_iqr cap therefore never sees the full
   spread. This stateless predicate looks ACROSS the clusters currently
   in the C2 window: if ``>= min_clusters`` of them sit within
   ``lm_tol_rad`` (box metric) and ``dt_s`` of the candidate and their
   DM span (max-min) exceeds ``dm_span_min``, the candidate is a comb
   fragment and the dump is suppressed. Bright single-DM repeaters
   (e.g. a pulsar) are NOT caught — all their clusters share ~one DM, so
   the span condition never trips.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

_LOG = logging.getLogger("dsart.coinc.veto")

__all__ = [
    "ClusterRateLimiter",
    "VetoRegion",
    "SiderealVetoRegistry",
    "dm_comb_detected",
    "ARCSEC_TO_RAD",
]

ARCSEC_TO_RAD: float = math.pi / 180.0 / 3600.0


# ---------------------------------------------------------------------------
# Broadband-RFI ("DM comb") veto
# ---------------------------------------------------------------------------


def dm_comb_detected(
    l_rad: float,
    m_rad: float,
    t_s: float,
    sibling_clusters: "List[tuple]",
    *,
    lm_tol_rad: float,
    dt_s: float,
    min_clusters: int,
    dm_span_min: float,
) -> bool:
    """True if ``(l_rad, m_rad, t_s)`` is part of a broadband DM comb.

    ``sibling_clusters`` is an iterable of ``(l_rad, m_rad, dm_pc_cc,
    t_s)`` 4-tuples describing EVERY cluster currently in the C2 window
    (the candidate's own cluster is expected to be included — it matches
    its own box at distance 0, so the caller need not special-case it).

    The candidate is a comb fragment iff at least ``min_clusters``
    clusters fall inside the ``lm_tol_rad`` (Chebyshev / box) +/- ``dt_s``
    spatio-temporal box around it AND the DM span (max - min) across those
    matched clusters exceeds ``dm_span_min``.

    A real dispersed burst is a single (l, m, t, DM) peak, so it can
    never gather ``>= min_clusters`` co-located clusters spanning a wide
    DM range; a bright single-DM repeater (pulsar) gathers many clusters
    but at ~one DM, so the span gate keeps it safe.
    """
    if min_clusters <= 0 or dm_span_min < 0.0:
        return False
    matched_dms: List[float] = []
    for sib in sibling_clusters:
        try:
            sl, sm, sdm, st = sib
        except (ValueError, TypeError):
            continue
        if (
            abs(sl - l_rad) <= lm_tol_rad
            and abs(sm - m_rad) <= lm_tol_rad
            and abs(st - t_s) <= dt_s
        ):
            matched_dms.append(float(sdm))
    if len(matched_dms) < min_clusters:
        return False
    return (max(matched_dms) - min(matched_dms)) > dm_span_min


# ---------------------------------------------------------------------------
# Cluster-rate limiter (RFI-storm guard)
# ---------------------------------------------------------------------------


class ClusterRateLimiter:
    """Sliding-window count of all evaluated clusters.

    Records one timestamp per cluster via :meth:`record`; :meth:`exceeded`
    returns True once the trailing-``window_s`` count reaches
    ``max_clusters``. A non-positive ``max_clusters`` disables the limiter
    (``exceeded`` always False).
    """

    def __init__(
        self,
        *,
        window_s: float = 60.0,
        max_clusters: int = 100,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        self._window_s = float(window_s)
        self._max = int(max_clusters)
        self._now = now if now is not None else time.monotonic
        self._times: List[float] = []

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def max_clusters(self) -> int:
        return self._max

    def _trim(self, now: float) -> None:
        cutoff = now - self._window_s
        if self._times and self._times[0] < cutoff:
            self._times = [t for t in self._times if t >= cutoff]

    def record(self, now: Optional[float] = None) -> None:
        """Record one evaluated cluster."""
        t = self._now() if now is None else float(now)
        self._times.append(t)
        self._trim(t)

    def count(self, now: Optional[float] = None) -> int:
        """Number of clusters in the trailing window."""
        t = self._now() if now is None else float(now)
        self._trim(t)
        return len(self._times)

    def exceeded(self, now: Optional[float] = None) -> bool:
        """True if firing a dump now would be during an over-rate window."""
        if self._max <= 0:
            return False
        return self.count(now) >= self._max


# ---------------------------------------------------------------------------
# Sidereal (l, m) registry veto
# ---------------------------------------------------------------------------


@dataclass
class VetoRegion:
    """One accumulating sky position in the sidereal veto registry.

    A region becomes an ACTIVE veto (``is_active``) once it has gathered
    ``>= min_hits`` hits whose first→last span is ``>= min_span_s``.
    """

    l_rad: float
    m_rad: float
    n_hits: int
    first_hit_unix: float
    last_hit_unix: float
    added_unix: float = 0.0  # when promoted to active (0 = not yet active)

    def span_s(self) -> float:
        return self.last_hit_unix - self.first_hit_unix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l_rad": float(self.l_rad),
            "m_rad": float(self.m_rad),
            "n_hits": int(self.n_hits),
            "first_hit_unix": float(self.first_hit_unix),
            "last_hit_unix": float(self.last_hit_unix),
            "added_unix": float(self.added_unix),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VetoRegion":
        return cls(
            l_rad=float(d["l_rad"]),
            m_rad=float(d["m_rad"]),
            n_hits=int(d.get("n_hits", 0)),
            first_hit_unix=float(d.get("first_hit_unix", 0.0)),
            last_hit_unix=float(d.get("last_hit_unix", 0.0)),
            added_unix=float(d.get("added_unix", 0.0)),
        )


class SiderealVetoRegistry:
    """Registry of sidereally-stationary (l, m) dump-veto regions.

    Parameters
    ----------
    tol_rad:
        Match tolerance in radians. Two positions within ``tol_rad`` in
        BOTH l and m (Chebyshev / box metric) are the same region.
    min_hits:
        Hits required before a region becomes an active veto (default 3).
    min_span_s:
        First→last hit span (s) required before a region becomes active
        (default 60 s) — stops a single burst of co-incident clusters in
        one window from instantly vetoing a position.
    expiry_s:
        A region rolls off this many seconds after its LAST hit
        (default 24 h). Rolling: continued hits keep it alive.
    now:
        Wall-clock function (unix seconds); injectable for tests.
    """

    def __init__(
        self,
        *,
        tol_rad: float,
        min_hits: int = 3,
        min_span_s: float = 60.0,
        expiry_s: float = 86400.0,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        self._tol = float(tol_rad)
        self._min_hits = int(min_hits)
        self._min_span_s = float(min_span_s)
        self._expiry_s = float(expiry_s)
        self._now = now if now is not None else time.time
        self._regions: List[VetoRegion] = []
        # Monotonically-increasing generation counter, bumped on every
        # mutation so the service knows when to re-publish/persist.
        self._generation: int = 0

    # ----- introspection ------------------------------------------------

    @property
    def tol_rad(self) -> float:
        return self._tol

    @property
    def min_hits(self) -> int:
        return self._min_hits

    @property
    def min_span_s(self) -> float:
        return self._min_span_s

    @property
    def expiry_s(self) -> float:
        return self._expiry_s

    @property
    def generation(self) -> int:
        return self._generation

    def _region_is_active(self, r: VetoRegion) -> bool:
        return (
            r.n_hits >= self._min_hits
            and r.span_s() >= self._min_span_s
        )

    def active_regions(self, now: Optional[float] = None) -> List[VetoRegion]:
        """Currently-active (non-expired) veto regions."""
        self.expire(now)
        return [r for r in self._regions if self._region_is_active(r)]

    def all_regions(self) -> List[VetoRegion]:
        """Every tracked region (active + still-accumulating)."""
        return list(self._regions)

    # ----- core ---------------------------------------------------------

    def _match(self, l_rad: float, m_rad: float) -> Optional[VetoRegion]:
        best: Optional[VetoRegion] = None
        best_d = self._tol
        for r in self._regions:
            dl = abs(r.l_rad - l_rad)
            dm = abs(r.m_rad - m_rad)
            if dl <= self._tol and dm <= self._tol:
                d = max(dl, dm)
                if d <= best_d:
                    best_d = d
                    best = r
        return best

    def expire(self, now: Optional[float] = None) -> int:
        """Drop regions whose last hit is older than ``expiry_s``.

        Returns the number of regions removed.
        """
        t = self._now() if now is None else float(now)
        cutoff = t - self._expiry_s
        before = len(self._regions)
        kept = [r for r in self._regions if r.last_hit_unix >= cutoff]
        removed = before - len(kept)
        if removed:
            self._regions = kept
            self._generation += 1
        return removed

    def observe(
        self,
        l_rad: float,
        m_rad: float,
        now: Optional[float] = None,
    ) -> bool:
        """Record a dump-eligible cluster at ``(l_rad, m_rad)``.

        Updates / creates the matching region. Returns True iff this
        observation newly PROMOTED a region to active (so the caller can
        log the event).
        """
        t = self._now() if now is None else float(now)
        self.expire(t)
        r = self._match(l_rad, m_rad)
        if r is None:
            r = VetoRegion(
                l_rad=float(l_rad), m_rad=float(m_rad),
                n_hits=1, first_hit_unix=t, last_hit_unix=t,
            )
            self._regions.append(r)
            self._generation += 1
            return False
        was_active = self._region_is_active(r)
        # Running mean keeps the region centred on the source as the
        # centroid jitters by ~1 px between hits.
        n = r.n_hits
        r.l_rad = (r.l_rad * n + float(l_rad)) / (n + 1)
        r.m_rad = (r.m_rad * n + float(m_rad)) / (n + 1)
        r.n_hits = n + 1
        r.last_hit_unix = t
        self._generation += 1
        now_active = self._region_is_active(r)
        if now_active and not was_active:
            r.added_unix = t
            return True
        return False

    def is_vetoed(
        self,
        l_rad: float,
        m_rad: float,
        now: Optional[float] = None,
    ) -> bool:
        """True if ``(l_rad, m_rad)`` falls in an active veto region."""
        t = self._now() if now is None else float(now)
        self.expire(t)
        r = self._match(l_rad, m_rad)
        return r is not None and self._region_is_active(r)

    def clear(self) -> int:
        """Remove all regions; returns how many were dropped."""
        n = len(self._regions)
        if n:
            self._regions = []
            self._generation += 1
        return n

    # ----- (de)serialisation -------------------------------------------

    def to_payload(self, now: Optional[float] = None) -> Dict[str, Any]:
        """etcd / display payload (active regions only)."""
        t = self._now() if now is None else float(now)
        active = self.active_regions(t)
        return {
            "ts": t,
            "tol_rad": self._tol,
            "tol_arcsec": self._tol / ARCSEC_TO_RAD,
            "min_hits": self._min_hits,
            "min_span_s": self._min_span_s,
            "expiry_s": self._expiry_s,
            "n_active": len(active),
            "regions": [r.to_dict() for r in active],
        }

    def to_full_payload(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Persistence payload (ALL regions, so accumulating ones survive
        a restart too)."""
        t = self._now() if now is None else float(now)
        self.expire(t)
        return {
            "ts": t,
            "tol_rad": self._tol,
            "min_hits": self._min_hits,
            "min_span_s": self._min_span_s,
            "expiry_s": self._expiry_s,
            "regions": [r.to_dict() for r in self._regions],
        }

    def load_payload(self, doc: Optional[Mapping[str, Any]]) -> int:
        """Replace the in-memory regions from a persisted payload.

        Returns the number of regions loaded. Tolerant of missing /
        malformed input (loads nothing). Expired regions are dropped.
        """
        if not isinstance(doc, Mapping):
            return 0
        raw = doc.get("regions")
        if not isinstance(raw, list):
            return 0
        loaded: List[VetoRegion] = []
        for d in raw:
            if not isinstance(d, Mapping):
                continue
            try:
                loaded.append(VetoRegion.from_dict(d))
            except Exception:  # noqa: BLE001
                continue
        self._regions = loaded
        self._generation += 1
        self.expire()
        return len(self._regions)
