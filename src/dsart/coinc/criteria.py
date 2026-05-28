"""YAML-driven trigger-criteria evaluator with hot-reload + holdoff.

Each criterion is a :class:`TriggerClass` describing a ``require``
predicate over :class:`dsart.coinc.stats.ClusterStats` and an
``action``. The evaluator walks the classes top-to-bottom on each
cluster; the first one whose ``require`` block satisfies and whose
holdoff window has elapsed wins.

YAML schema (single source of truth: ``configs/c2_trigger_criteria.yaml``):

```yaml
trigger_classes:
  - name: bright_frb
    description: "..."            # optional, ignored at evaluation
    require:
      snr_max_min: 12.0
      n_events_min: 3
      n_search_nodes_min: 2
      dm_median_min_pc_cc: 50.0
      dm_median_max_pc_cc: 4000.0
      dm_iqr_max_pc_cc: 30.0
      width_median_max_samples: 32
      lm_diag_max_rad: 5.0e-3
    action: dump_all_gpus
    holdoff_s: 30.0
```

Recognised require keys (all optional, ANDed):

  * ``snr_max_min`` / ``snr_max_max``
  * ``snr_sum_min`` / ``snr_sum_max``
  * ``snr_mean_min`` / ``snr_mean_max``
  * ``n_events_min`` / ``n_events_max``
  * ``n_search_nodes_min`` / ``n_search_nodes_max``
  * ``n_gpu_halves_min`` / ``n_gpu_halves_max``
  * ``dm_median_min_pc_cc`` / ``dm_median_max_pc_cc``
  * ``dm_iqr_max_pc_cc`` — SNR-scaled cap. The yaml value is the cap
    applied at ``snr_max == DM_IQR_SNR_THRESHOLD`` (default 12); the
    effective cap grows hyperbolically and becomes infinite at
    ``snr_max >= DM_IQR_SNR_CEILING`` (default 50). This accommodates
    very-bright bursts that trip many DM trials simultaneously while
    keeping the tight cap for low-SNR candidates.
  * ``width_median_max_samples`` / ``width_median_min_samples``
  * ``lm_diag_max_rad`` / ``lm_diag_min_rad``
  * ``dm_galactic_fraction_max`` / ``dm_galactic_fraction_min`` —
    cluster DM median expressed as a fraction of the NE2001 max-LOS
    galactic DM at the current pointing (provided by
    ``/mon/array/gal_dm``, refreshed by the C2 service every
    ``coinc.gal_dm_poll_interval_s`` seconds, default 30). Used as
    the *sole* galactic / extragalactic discriminant:
    ``frac < 0.75 → Galactic disk; frac >= 0.75 → extragalactic``.
    Predicates return False when ``dm_galactic_fraction`` is NaN
    (i.e. /mon/array/gal_dm has never been read), so classes that
    don't gate on this key keep their existing behaviour.

Action values are opaque to this module; the service interprets
``dump_all_gpus`` and ``log_only`` (other values are returned through
as-is).

Hot-reload: the evaluator caches the file mtime on load and re-reads
on :meth:`reload_if_changed`. The C2 service calls it both on SIGHUP
and on each per-cluster eval as a defence-in-depth (the mtime check
is cheap — single stat).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import yaml

from .stats import ClusterStats

__all__ = [
    "TriggerClass",
    "CriteriaEvaluator",
    "BadCriteriaFile",
]


_LOG = logging.getLogger("dsart.coinc.criteria")


# SNR scaling for dm_iqr_max_pc_cc:
#
# A bright FRB sweeps a wide range of trial DMs simultaneously (since
# the matched-filter response stays above the C1 threshold across many
# adjacent fdm bins). Holding the cluster's dm_iqr to a tight cap (like
# 30 pc/cc) is therefore correct for low-SNR (genuine narrow-DM)
# candidates but rejects the brightest events outright. We scale the
# cap with ``cluster.snr_max`` along a hyperbolic curve:
#
#   - at ``snr_max == DM_IQR_SNR_THRESHOLD`` (12.0): cap = yaml_value
#     (so the operator's tunable matches the standard low-SNR regime)
#   - at ``snr_max >= DM_IQR_SNR_CEILING`` (50.0): cap = +inf
#     (saturated bursts always pass the IQR check)
#   - in between: cap = yaml_value * (CEILING - THRESHOLD) /
#                       (CEILING - snr_max)
#
# This matches the operator spec: "30 at the threshold, infinite at
# SNR>50". (Tunable thresholds are module-level constants rather than
# YAML keys to keep the require-block declarative.)
DM_IQR_SNR_THRESHOLD: float = 12.0
DM_IQR_SNR_CEILING: float = 50.0


def _dm_iqr_max_snr_scaled(stats: ClusterStats, t: float) -> bool:
    """``dm_iqr_max_pc_cc`` predicate, with hyperbolic SNR scaling.

    Returns True iff ``stats.dm_iqr`` satisfies the snr_max-scaled cap.
    See ``DM_IQR_SNR_THRESHOLD`` / ``DM_IQR_SNR_CEILING`` block above
    for the cap formula.
    """
    s = float(stats.snr_max)
    if s >= DM_IQR_SNR_CEILING:
        return True
    span = DM_IQR_SNR_CEILING - DM_IQR_SNR_THRESHOLD
    denom = DM_IQR_SNR_CEILING - max(s, DM_IQR_SNR_THRESHOLD)
    # max(...) clamps the denominator from below at ``span`` so that
    # for snr_max <= THRESHOLD the cap is exactly ``t`` (no scaling).
    cap = t * span / denom
    return float(stats.dm_iqr) <= cap


def _dm_galactic_fraction_max(stats: ClusterStats, t: float) -> bool:
    """``dm_galactic_fraction_max`` — Galactic-disk gate.

    Returns True iff ``stats.dm_galactic_fraction`` is finite AND
    ``<= t``. NaN → False so classes that gate on the galactic-DM
    discriminant simply don't match when /mon/array/gal_dm is
    unavailable; the operator gets log_only fallback.
    """
    frac = float(stats.dm_galactic_fraction)
    if not math.isfinite(frac):
        return False
    return frac <= t


def _dm_galactic_fraction_min(stats: ClusterStats, t: float) -> bool:
    """``dm_galactic_fraction_min`` — extragalactic gate."""
    frac = float(stats.dm_galactic_fraction)
    if not math.isfinite(frac):
        return False
    return frac >= t


# Lookup table mapping a require-key to:
#   (predicate)
# Predicate is a callable ``(stats, threshold) -> bool`` that returns
# True iff the cluster satisfies the require entry. Predicates receive
# the full ``ClusterStats`` so we can implement stat-coupled criteria
# (e.g. snr_max-scaled dm_iqr cap, see ``_dm_iqr_max_snr_scaled``).
_REQUIRE_PREDICATES: Tuple[
    Tuple[str, Callable[[ClusterStats, float], bool]], ...
] = (
    ("snr_max_min", lambda s, t: s.snr_max >= t),
    ("snr_max_max", lambda s, t: s.snr_max <= t),
    ("snr_sum_min", lambda s, t: s.snr_sum >= t),
    ("snr_sum_max", lambda s, t: s.snr_sum <= t),
    ("snr_mean_min", lambda s, t: s.snr_mean >= t),
    ("snr_mean_max", lambda s, t: s.snr_mean <= t),
    ("n_events_min", lambda s, t: s.n_events >= t),
    ("n_events_max", lambda s, t: s.n_events <= t),
    ("n_search_nodes_min", lambda s, t: s.n_search_nodes >= t),
    ("n_search_nodes_max", lambda s, t: s.n_search_nodes <= t),
    ("n_gpu_halves_min", lambda s, t: s.n_gpu_halves >= t),
    ("n_gpu_halves_max", lambda s, t: s.n_gpu_halves <= t),
    ("dm_median_min_pc_cc", lambda s, t: s.dm_median >= t),
    ("dm_median_max_pc_cc", lambda s, t: s.dm_median <= t),
    ("dm_iqr_max_pc_cc", _dm_iqr_max_snr_scaled),
    ("width_median_max_samples", lambda s, t: s.width_median <= t),
    ("width_median_min_samples", lambda s, t: s.width_median >= t),
    ("lm_diag_max_rad", lambda s, t: s.lm_diag_rad <= t),
    ("lm_diag_min_rad", lambda s, t: s.lm_diag_rad >= t),
    ("dm_galactic_fraction_max", _dm_galactic_fraction_max),
    ("dm_galactic_fraction_min", _dm_galactic_fraction_min),
)
_REQUIRE_KEYS = {key for key, _ in _REQUIRE_PREDICATES}


class BadCriteriaFile(ValueError):
    """Raised on malformed YAML / unknown require keys / unknown actions."""


@dataclass(frozen=True, slots=True)
class TriggerClass:
    name: str
    require: Dict[str, float]
    action: str
    holdoff_s: float = 0.0
    description: str = ""


def _validate_class(entry: Mapping[str, Any], idx: int) -> TriggerClass:
    if not isinstance(entry, Mapping):
        raise BadCriteriaFile(
            f"trigger_classes[{idx}] is not a mapping: {entry!r}"
        )
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise BadCriteriaFile(
            f"trigger_classes[{idx}].name must be a non-empty string"
        )
    require = entry.get("require", {}) or {}
    if not isinstance(require, Mapping):
        raise BadCriteriaFile(
            f"trigger_classes[{idx}].require must be a mapping"
        )
    unknown = set(require) - _REQUIRE_KEYS
    if unknown:
        raise BadCriteriaFile(
            f"trigger_classes[{idx}] '{name}' has unknown require keys: "
            f"{sorted(unknown)} (valid: {sorted(_REQUIRE_KEYS)})"
        )
    action = entry.get("action")
    if not isinstance(action, str) or not action:
        raise BadCriteriaFile(
            f"trigger_classes[{idx}].action must be a non-empty string"
        )
    holdoff = entry.get("holdoff_s", 0.0)
    try:
        holdoff = float(holdoff)
    except (TypeError, ValueError):
        raise BadCriteriaFile(
            f"trigger_classes[{idx}].holdoff_s must be a number"
        ) from None
    if holdoff < 0.0:
        raise BadCriteriaFile(
            f"trigger_classes[{idx}].holdoff_s={holdoff} must be >= 0"
        )
    description = str(entry.get("description", "") or "")
    return TriggerClass(
        name=name,
        require={k: float(v) for k, v in require.items()},
        action=action,
        holdoff_s=holdoff,
        description=description,
    )


def _load_yaml_classes(path: Path) -> List[TriggerClass]:
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, Mapping):
        raise BadCriteriaFile(
            f"{path}: top-level YAML must be a mapping; got {type(doc).__name__}"
        )
    classes = doc.get("trigger_classes")
    if classes is None:
        raise BadCriteriaFile(
            f"{path}: missing top-level 'trigger_classes' list"
        )
    if not isinstance(classes, list):
        raise BadCriteriaFile(
            f"{path}: 'trigger_classes' must be a list"
        )
    parsed: List[TriggerClass] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(classes):
        cls = _validate_class(entry, i)
        if cls.name in seen_names:
            raise BadCriteriaFile(
                f"{path}: duplicate trigger_class name {cls.name!r}"
            )
        seen_names.add(cls.name)
        parsed.append(cls)
    return parsed


def _matches(stats: ClusterStats, require: Mapping[str, float]) -> bool:
    for key, predicate in _REQUIRE_PREDICATES:
        if key not in require:
            continue
        if not predicate(stats, require[key]):
            return False
    return True


class CriteriaEvaluator:
    """Hot-reloadable evaluator over a list of :class:`TriggerClass`.

    Concurrency: single-threaded use only (called from the C2 service's
    asyncio loop). Holdoff state is in-process; restarting the service
    resets all holdoffs.
    """

    def __init__(self, path: Path, *, now: Optional[Any] = None) -> None:
        self._path = Path(path)
        self._now = now or time.monotonic
        self._classes: List[TriggerClass] = []
        self._mtime: float = -1.0
        self._last_fired_at: Dict[str, float] = {}
        self._reload()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def classes(self) -> Tuple[TriggerClass, ...]:
        return tuple(self._classes)

    # ----- loading ------------------------------------------------------

    def _reload(self) -> None:
        if not self._path.exists():
            raise BadCriteriaFile(f"criteria file not found: {self._path}")
        self._classes = _load_yaml_classes(self._path)
        self._mtime = self._path.stat().st_mtime
        _LOG.info(
            "criteria reload: %s now defines %d trigger classes (%s)",
            self._path, len(self._classes),
            ", ".join(c.name for c in self._classes),
        )

    def reload_if_changed(self) -> bool:
        """Re-read the YAML if the file mtime has changed; returns True
        on reload, False if nothing changed."""
        try:
            cur = self._path.stat().st_mtime
        except FileNotFoundError:
            _LOG.warning(
                "criteria file disappeared: %s (keeping last-loaded set)",
                self._path,
            )
            return False
        if cur != self._mtime:
            try:
                self._reload()
                return True
            except BadCriteriaFile:
                _LOG.exception(
                    "criteria reload of %s failed; keeping last-loaded set",
                    self._path,
                )
                return False
        return False

    def force_reload(self) -> None:
        """Reload unconditionally (used on SIGHUP)."""
        self._reload()

    # ----- evaluation ---------------------------------------------------

    def evaluate(self, stats: ClusterStats) -> Optional[TriggerClass]:
        """Walk classes top-to-bottom; return the first match whose
        holdoff has elapsed, or None if no class matches."""
        now = self._now()
        for cls in self._classes:
            if not _matches(stats, cls.require):
                continue
            if cls.holdoff_s > 0.0:
                last = self._last_fired_at.get(cls.name, -float("inf"))
                if (now - last) < cls.holdoff_s:
                    continue
            self._last_fired_at[cls.name] = now
            return cls
        return None

    def last_fired_at(self, name: str) -> Optional[float]:
        return self._last_fired_at.get(name)

    def last_fired_at_snapshot(self) -> Dict[str, float]:
        """Return a shallow copy of the per-class last-fired timestamps.

        Used by callers that need to roll back :meth:`evaluate`'s
        holdoff side-effect — notably the C2 coincidencer's dump-gate
        path: ``evaluate`` mutates ``_last_fired_at[name] = now`` on
        every match, but if the matched trigger ends up being
        suppressed by ``/cmd/c2/dumps_enabled`` we want to leave the
        holdoff state exactly as it was so the next genuine trigger
        fires immediately rather than being eaten by a stale holdoff.
        """
        return dict(self._last_fired_at)

    def restore_last_fired_at(
        self, name: str, prev_value: Optional[float],
    ) -> None:
        """Roll back ``_last_fired_at[name]`` to ``prev_value``.

        Pairs with :meth:`last_fired_at_snapshot`. ``prev_value=None``
        means "the class had never fired before" — we pop the entry
        entirely so the next match treats it as a cold start. This is
        the only public mutator on the holdoff map; the C2 service
        uses it from the dumps-suppression path. The evaluator stays
        single-threaded otherwise.
        """
        if prev_value is None:
            self._last_fired_at.pop(name, None)
        else:
            self._last_fired_at[name] = float(prev_value)
