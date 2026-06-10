"""dsa_monitor Phase 6c.A: bootstrap-SNR injection calibration.

The corr-side injector consumes ``fluence_jy_ms`` directly, but we
have no absolute flux calibration on a freshly-restarted fleet — so
"50 Jy·ms" really means "whatever SNR the matched filter pulls out
of that voltage scale". The operator wants to drive injections at a
*target SNR* (e.g. SNR=20) instead.

This module provides the two halves of a bootstrap calibration:

1. :class:`CalibrationStore` — persists a per-``dm_band`` calibration
   constant ``K`` in etcd at
   ``/cnf/inject/snr_calibration/<bucket>``. ``K`` parameterises

       observed_snr ≈ K × sqrt(fluence_jy_ms / width_samples)

   so the inverse is

       fluence_jy_ms = width_samples × (target_snr / K)^2.

   ``K`` depends on the search-side DM band (the coarse + fine
   de-disperser path the injection drives); we bucket by
   ``round(dm / 50) × 50`` to give the operator a small, predictable
   table. ``K`` is **width-independent** post-F-fix-injector-fluence-
   norm — the voltage-domain injector now obeys the radiometer law
   ``observed_snr ∝ fluence/√width`` exactly, so a single calibration
   probe at any width pins K for that DM band and predicts SNR for
   every other width. (Pre-fix, the injector violated the law by a
   factor of 1/W², which would have leaked into K and forced per-
   width calibration; that constraint is now removed.)

2. :func:`fire_calibration_probe` — fires one known-fluence injection
   with the existing :func:`control_store.control_inject_pulse`
   helper, records the corresponding ``/cnf/inject/active/<inj_id>``
   payload the C2 matcher consumes, then polls
   ``/mon/dsart/inject/matches/<inj_id>`` for the observed SNR. On a
   match it computes ``K = observed_snr × sqrt(width / fluence)``
   and stores it. On no-match it returns a structured error so the
   UI can surface "calibration failed: no C1 candidate within
   tolerance".

The Control tab uses both halves:

  * "Calibrate" button: caller picks (DM, l, m, width); the dashboard
    fires a probe with a generous default fluence (100 Jy·ms — bright
    enough to land at SNR ~ 30 at DM=500) and stores the resulting
    ``K``.
  * "Inject at target SNR": the dashboard reads K for the requested
    DM bucket, computes the implied fluence, and fires through the
    existing inject helper. If K is missing for the bucket the
    dashboard returns 412 Precondition Failed so the UI can prompt
    the operator to calibrate first.

All etcd surface is stdlib-only; the heavy
:class:`control_store.ControlStore` does the actual ``put_dict`` /
``get_dict`` so transport errors fail consistently with the rest of
the dashboard.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger("dsa_monitor.inject_calibration")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Active-injection registry prefix. Mirrors
#: :data:`dsart.coinc.inject_match.ACTIVE_INJECT_PREFIX`. Drift is
#: caught by the unit tests in
#: ``tests/test_dsa_monitor_inject_calibration.py``.
ACTIVE_INJECT_PREFIX: str = "/cnf/inject/active/"

#: Match-event prefix the C2 matcher publishes to. Mirrors
#: :data:`dsart.coinc.inject_match.MATCH_EVENT_PREFIX`.
MATCH_EVENT_PREFIX: str = "/mon/dsart/inject/matches/"

#: Per-bucket calibration store prefix.
CALIBRATION_PREFIX: str = "/cnf/inject/snr_calibration/"

#: DM bucket size (pc/cc). The operator-supplied DM is rounded to the
#: nearest multiple of this when computing the bucket key. 50 pc/cc
#: is wide enough that small DM mis-quantisations land in the same
#: bucket while still letting us distinguish the three commissioning
#: DMs (150, 500, 1500) cleanly.
DM_BUCKET_PC_CC: float = 50.0

#: Default fluence used by an explicit calibration probe (Jy·ms).
#:
#: 2026-06-10 rescale (supersedes the 2026-06-09 0.01 default): the
#: live fluence-response sweep mapped the usable probe window
#: precisely —
#:
#:   * < ~2e-4 Jy·ms  : in-cube peak below c1_snr_min (12σ) → no_match
#:   * 4e-4 .. 1e-3   : clean, repeatable matches, SNR ≈ 25-50,
#:                      K ≈ 2890-2980 across DM 150/500/1000 (w=4)
#:   * ≥ ~1.4e-3      : the cuFFT-cfp16 butterflies overflow → ±inf →
#:                      ±60000 plateau → boxcar scores go inf/NaN and
#:                      the candidate VANISHES (pre-hardening) or
#:                      saturates (post detector.input_clip_sigma).
#:
#: 7e-4 Jy·ms at width 4 lands at SNR ≈ 38: the middle of the clean
#: window, ≳3× above the c1_snr_min floor and ~2× below the fp16
#: overflow cliff. The dashboard caller can override this in the form
#: if the operator wants to titrate the calibration brightness.
DEFAULT_CALIBRATION_FLUENCE: float = 7e-4

#: Default probe width (NATIVE samples, 32.768 µs each). NOTE: any
#: width ≲ 32 native is sub-search-sample (the search integrates to
#: ~1.05 ms), so the burst energy lands in a single search sample and
#: the observed SNR scales ∝ fluence (NOT ∝ √(fluence/width)). The
#: stored K is therefore only self-consistent for probes fired at the
#: SAME width — keep this pinned to 4 (matches the 2026-06-10
#: calibration grid; a w=1 probe at the model-equivalent fluence
#: measured K ≈ 977 vs 2890 at w=4).
DEFAULT_CALIBRATION_WIDTH: int = 4

#: Default probe profile.
DEFAULT_CALIBRATION_PROFILE: str = "gaussian"

#: How long after fired_at_unix the matcher accepts rows for this
#: injection (seconds). Long enough to cover startup-jitter, NUMA
#: pinning hiccups, and the cube emission cadence at DM=1500 (which
#: takes ~3.5 cubes to fully de-disperse).
DEFAULT_INJECT_TTL_S: float = 60.0

#: Polling cadence for the match-event key (seconds).
DEFAULT_POLL_INTERVAL_S: float = 0.5

#: Default per-probe poll timeout (seconds). The probe waits this long
#: for the first match; if no match arrives the probe returns
#: ``ok=False`` with ``reason="no_match"``.
DEFAULT_POLL_TIMEOUT_S: float = 30.0


# ---------------------------------------------------------------------------
# Bucket key
# ---------------------------------------------------------------------------


#: Substring marker for legacy ``dmNNNN_wWWWW`` bucket keys produced
#: by the pre-F-fix-injector-fluence-norm dashboard. Such entries are
#: stale (their K absorbed the buggy injector's 1/W² scaling) and are
#: invisible to the new DM-only lookup; they remain visible to
#: :func:`delete_snr_calibrations` so the operator can wipe them.
LEGACY_BUCKET_INFIX: str = "_w"


def bucket_key(dm_pc_cm3: float) -> str:
    """Return the calibration bucket key for ``dm``.

    Format: ``dm{rounded_dm:04d}``. Rounded DM is
    ``round(dm / 50) × 50`` (clamped to ≥ 0).

    The bucket is intentionally width-independent: post-F-fix-injector-
    fluence-norm the injector obeys ``observed_snr ∝ fluence/√width``
    exactly, so K depends only on the DM band's coarse+fine
    de-disperser path. One probe at any width pins K for the band.

    Examples
    --------
    >>> bucket_key(500.0)
    'dm0500'
    >>> bucket_key(150.0)
    'dm0150'
    >>> bucket_key(1500.0)
    'dm1500'
    """
    if not math.isfinite(dm_pc_cm3):
        raise ValueError(f"dm_pc_cm3={dm_pc_cm3} is not finite")
    dm_round = max(
        0, int(round(float(dm_pc_cm3) / DM_BUCKET_PC_CC) * int(DM_BUCKET_PC_CC)),
    )
    return f"dm{dm_round:04d}"


def calibration_key(dm_pc_cm3: float) -> str:
    """Return the full etcd key for a DM-band bucket."""
    return f"{CALIBRATION_PREFIX}{bucket_key(dm_pc_cm3)}"


def is_legacy_bucket(bucket: str) -> bool:
    """Return True if ``bucket`` is the pre-fix ``dmNNNN_wWWWW``
    format. Such entries are kept around for the operator to delete
    via :func:`delete_snr_calibrations` but are ignored by lookup.
    """
    return LEGACY_BUCKET_INFIX in str(bucket)


def build_active_inject_key(inj_id: str) -> str:
    if not inj_id:
        raise ValueError("inj_id must be non-empty")
    return f"{ACTIVE_INJECT_PREFIX}{inj_id}"


def build_match_event_key(inj_id: str) -> str:
    if not inj_id:
        raise ValueError("inj_id must be non-empty")
    return f"{MATCH_EVENT_PREFIX}{inj_id}"


# ---------------------------------------------------------------------------
# CalibrationStore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationEntry:
    """One row of the per-bucket calibration table."""
    bucket: str
    dm_pc_cm3_rounded: int
    width_samples: int
    K: float
    last_fluence_jy_ms: float
    last_observed_snr: float
    last_inj_id: str
    last_calibrated_at_unix: float
    last_target_snr: Optional[float] = None
    actor: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationEntry":
        return cls(
            bucket=str(d["bucket"]),
            dm_pc_cm3_rounded=int(d["dm_pc_cm3_rounded"]),
            width_samples=int(d["width_samples"]),
            K=float(d["K"]),
            last_fluence_jy_ms=float(d["last_fluence_jy_ms"]),
            last_observed_snr=float(d["last_observed_snr"]),
            last_inj_id=str(d["last_inj_id"]),
            last_calibrated_at_unix=float(d["last_calibrated_at_unix"]),
            last_target_snr=(
                float(d["last_target_snr"])
                if d.get("last_target_snr") is not None else None
            ),
            actor=str(d.get("actor", "unknown")),
        )


class CalibrationStore:
    """Persists per-``dm_band`` ``K`` constants in etcd.

    The operator builds the table empirically by firing one
    :func:`fire_calibration_probe` per DM-band bucket they care
    about. The store is the single source of truth the SNR → fluence
    translation reads from at inject-time. K is width-independent
    (see module docstring), so a probe at any width pins K for every
    width within the band.
    """

    def __init__(self, control_store: Any) -> None:
        self._store = control_store

    def get(
        self, *, dm_pc_cm3: float,
    ) -> Optional[CalibrationEntry]:
        """Return the entry for the DM-band bucket, or None."""
        key = calibration_key(dm_pc_cm3)
        try:
            raw = self._store.get_dict(key)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("CalibrationStore.get(%s) failed: %s", key, exc)
            return None
        if not isinstance(raw, Mapping):
            return None
        try:
            return CalibrationEntry.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            LOG.warning(
                "CalibrationStore.get(%s) bad payload: %s", key, exc,
            )
            return None

    def put(self, entry: CalibrationEntry) -> str:
        """Persist an entry; returns the etcd key written."""
        key = f"{CALIBRATION_PREFIX}{entry.bucket}"
        self._store.put_dict(key, entry.to_dict())
        return key

    def list_all(self) -> List[CalibrationEntry]:
        """Return every present calibration entry, sorted by bucket."""
        # Use the raw etcd3 client via the same path
        # control_store.list_recent_audit walks.
        if hasattr(self._store, "_ensure"):
            try:
                self._store._ensure()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("CalibrationStore.list_all _ensure: %s", exc)
                return []
        client_owner = getattr(self._store, "_store", self._store)
        try:
            client = client_owner.get_etcd()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("CalibrationStore.list_all get_etcd: %s", exc)
            return []
        try:
            pairs = list(client.get_prefix(CALIBRATION_PREFIX))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("CalibrationStore.list_all get_prefix: %s", exc)
            return []
        out: List[CalibrationEntry] = []
        for value, _meta in pairs:
            try:
                if isinstance(value, (bytes, bytearray)):
                    payload = json.loads(value.decode("utf-8"))
                elif isinstance(value, str):
                    payload = json.loads(value)
                elif isinstance(value, Mapping):
                    payload = dict(value)
                else:
                    continue
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "CalibrationStore.list_all bad JSON: %s", exc,
                )
                continue
            try:
                out.append(CalibrationEntry.from_dict(payload))
            except (KeyError, TypeError, ValueError) as exc:
                LOG.warning(
                    "CalibrationStore.list_all bad entry: %s", exc,
                )
                continue
        out.sort(key=lambda e: e.bucket)
        return out

    def _client(self) -> Any:
        """Return the raw etcd3 client (same path :meth:`list_all`
        walks), or ``None`` if it cannot be reached."""
        if hasattr(self._store, "_ensure"):
            try:
                self._store._ensure()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("CalibrationStore._client _ensure: %s", exc)
                return None
        client_owner = getattr(self._store, "_store", self._store)
        try:
            return client_owner.get_etcd()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("CalibrationStore._client get_etcd: %s", exc)
            return None

    def delete(self, bucket: str) -> bool:
        """Delete one calibration bucket key.

        Returns ``True`` when etcd reports a key was removed,
        ``False`` when it was already absent. Raises on transport
        failure so the caller can record a per-bucket error.
        """
        client = self._client()
        if client is None:
            raise RuntimeError("etcd client unavailable")
        key = f"{CALIBRATION_PREFIX}{bucket}"
        return bool(client.delete(key))


# ---------------------------------------------------------------------------
# SNR ↔ fluence math
# ---------------------------------------------------------------------------


def snr_to_fluence(
    *, target_snr: float, K: float, width_samples: int,
) -> float:
    """Return the fluence (Jy·ms) needed to hit ``target_snr``.

    Inverts ``observed_snr = K × sqrt(fluence / width_samples)``:

        fluence = width_samples × (target_snr / K)^2.

    Raises :class:`ValueError` on non-positive K / width / target.
    """
    if not (math.isfinite(target_snr) and math.isfinite(K)):
        raise ValueError(
            f"target_snr={target_snr}, K={K}: must be finite"
        )
    if target_snr <= 0.0:
        raise ValueError(f"target_snr={target_snr} must be > 0")
    if K <= 0.0:
        raise ValueError(f"K={K} must be > 0")
    if width_samples < 1:
        raise ValueError(f"width_samples={width_samples} must be >= 1")
    ratio = float(target_snr) / float(K)
    return float(width_samples) * ratio * ratio


# ---------------------------------------------------------------------------
# Active-injection registry helpers
# ---------------------------------------------------------------------------


def publish_active_inject(
    store: Any,
    *,
    inj_id: str,
    dm_pc_cm3: float,
    l_rad: float,
    m_rad: float,
    width_samples: int,
    fluence_jy_ms: float,
    apply_at_specnum: int,
    fired_at_unix: Optional[float] = None,
    ttl_s: float = DEFAULT_INJECT_TTL_S,
    fired_by: str = "unknown",
    target_snr: Optional[float] = None,
) -> str:
    """PUT one ``/cnf/inject/active/<inj_id>`` row.

    The C2 matcher polls this prefix and uses the payload to match
    incoming C1 candidates. Returns the etcd key written.
    """
    fired = float(fired_at_unix) if fired_at_unix is not None else time.time()
    payload: Dict[str, Any] = {
        "inj_id": str(inj_id),
        "dm_pc_cm3": float(dm_pc_cm3),
        "l_rad": float(l_rad),
        "m_rad": float(m_rad),
        "width_samples": int(width_samples),
        "fluence_jy_ms": float(fluence_jy_ms),
        "apply_at_specnum": int(apply_at_specnum),
        "fired_at_unix": fired,
        "ttl_s": float(ttl_s),
        "fired_by": str(fired_by),
    }
    if target_snr is not None:
        payload["target_snr"] = float(target_snr)
    key = build_active_inject_key(inj_id)
    store.put_dict(key, payload)
    return key


def get_match_event(store: Any, inj_id: str) -> Optional[Dict[str, Any]]:
    """Return the current ``/mon/dsart/inject/matches/<inj_id>`` payload."""
    key = build_match_event_key(inj_id)
    try:
        raw = store.get_dict(key)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("get_match_event(%s) failed: %s", key, exc)
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    return None


# ---------------------------------------------------------------------------
# Fire-calibration-probe orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """Return value of :func:`fire_calibration_probe`."""
    ok: bool
    inj_id: str
    bucket: str
    K: Optional[float]
    observed_snr: Optional[float]
    observed_event_specnum: Optional[int]
    matched_at_unix: Optional[float]
    n_matches: int
    elapsed_s: float
    reason: str
    inject_response: Optional[Dict[str, Any]] = None
    active_key: Optional[str] = None
    calibration_key: Optional[str] = None
    observed_l_rad: Optional[float] = None
    observed_m_rad: Optional[float] = None
    observed_width_samples: Optional[int] = None
    match_lm_dist_rad: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fire_calibration_probe(
    store: Any,
    *,
    inject_fn: Any,
    dm_pc_cm3: float,
    l_rad: float = 0.0,
    m_rad: float = 0.0,
    width_samples: int = DEFAULT_CALIBRATION_WIDTH,
    fluence_jy_ms: float = DEFAULT_CALIBRATION_FLUENCE,
    profile: str = DEFAULT_CALIBRATION_PROFILE,
    margin_blocks: Optional[int] = None,
    chgroups: Optional[Iterable[int]] = None,
    inj_id_prefix: str = "cal_probe",
    user: Optional[str] = None,
    poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ttl_s: float = DEFAULT_INJECT_TTL_S,
    inj_id_override: Optional[str] = None,
    cal_store: Optional[CalibrationStore] = None,
    time_fn: Any = time.time,
    sleep_fn: Any = time.sleep,
) -> ProbeResult:
    """Fire a single calibration injection and read back K.

    Workflow

      1. Build a synthetic ``inj_id`` (or use ``inj_id_override``).
      2. PUT ``/cnf/inject/active/<inj_id>`` with the probe metadata
         so the C2 matcher matches incoming candidates against it.
      3. Call ``inject_fn(...)`` (typically
         :func:`control_store.control_inject_pulse`) which arms +
         broadcasts the injection.
      4. Poll ``/mon/dsart/inject/matches/<inj_id>`` every
         ``poll_interval_s`` until the first match arrives or
         ``poll_timeout_s`` elapses.
      5. On a match, compute K = observed_snr × sqrt(width / fluence)
         and persist it via :class:`CalibrationStore`.
      6. Return :class:`ProbeResult`.

    Parameters
    ----------
    store:
        DsaStore-like object (typically the dashboard's
        :class:`control_store.ControlStore`).
    inject_fn:
        Callable with the same signature as
        :func:`control_store.control_inject_pulse`. Tests inject a
        fake that just returns ``{"ok": True, ...}`` without
        actually contacting corr_fast.
    dm_pc_cm3, l_rad, m_rad, width_samples, fluence_jy_ms, profile:
        Standard injection parameters.
    margin_blocks, chgroups:
        Forwarded to ``inject_fn``.
    inj_id_prefix:
        Becomes part of ``inj_id`` if ``inj_id_override`` is unset.
    user:
        Operator label, forwarded to ``inject_fn`` + audit row.
    poll_timeout_s, poll_interval_s, ttl_s:
        Tunables for the wait + the registry TTL.
    inj_id_override:
        Force a specific inj_id (handy for tests).
    cal_store:
        Optional pre-built :class:`CalibrationStore`. Default
        constructs one wrapping ``store``.
    time_fn, sleep_fn:
        Tests inject deterministic clock + sleep helpers.

    Returns
    -------
    ProbeResult
        ``ok=True`` iff a match arrived and K is finite and positive.
        On failure ``reason`` is one of ``no_match``, ``inject_failed``,
        ``bad_K``, ``etcd_failure``.
    """
    t0 = float(time_fn())
    cs = cal_store if cal_store is not None else CalibrationStore(store)
    bucket = bucket_key(dm_pc_cm3)
    inj_id = inj_id_override or _build_probe_inj_id(
        prefix=inj_id_prefix, dm=dm_pc_cm3, width=width_samples,
        timestamp=t0,
    )
    # 1. Publish the active-injection registry row BEFORE arming the
    #    pulse so the matcher already has the entry by the time the
    #    coincidencer sees the corresponding C1 batch. The matcher's
    #    refresh cadence is 1 s; the corr_fast → search → C2 latency
    #    is ~1-3 s at steady state, comfortably wider.
    try:
        active_key = publish_active_inject(
            store,
            inj_id=inj_id,
            dm_pc_cm3=float(dm_pc_cm3),
            l_rad=float(l_rad),
            m_rad=float(m_rad),
            width_samples=int(width_samples),
            fluence_jy_ms=float(fluence_jy_ms),
            apply_at_specnum=0,           # placeholder; rewritten below
            fired_at_unix=t0,
            ttl_s=float(ttl_s),
            fired_by=str(user or "calibration_probe"),
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("publish_active_inject(%s) failed: %s", inj_id, exc)
        return ProbeResult(
            ok=False, inj_id=inj_id, bucket=bucket,
            K=None, observed_snr=None, observed_event_specnum=None,
            matched_at_unix=None, n_matches=0,
            elapsed_s=float(time_fn()) - t0,
            reason=f"etcd_failure_active: {exc}",
        )
    # 2. Fire the injection.
    inject_kwargs: Dict[str, Any] = {
        "inj_id": inj_id,
        "l_rad": float(l_rad),
        "m_rad": float(m_rad),
        "dm_pc_cm3": float(dm_pc_cm3),
        "fluence_jy_ms": float(fluence_jy_ms),
        "width_samples": int(width_samples),
        "profile": str(profile),
        "apply_at_specnum": None,         # auto-arm
        "user": user,
    }
    if margin_blocks is not None:
        inject_kwargs["margin_blocks"] = int(margin_blocks)
    if chgroups is not None:
        inject_kwargs["chgroups"] = tuple(int(c) for c in chgroups)
    try:
        inject_response = inject_fn(store, **inject_kwargs)
    except Exception as exc:  # noqa: BLE001
        LOG.error("inject_fn(%s) failed: %s", inj_id, exc)
        return ProbeResult(
            ok=False, inj_id=inj_id, bucket=bucket,
            K=None, observed_snr=None, observed_event_specnum=None,
            matched_at_unix=None, n_matches=0,
            elapsed_s=float(time_fn()) - t0,
            reason=f"inject_failed: {exc}",
            active_key=active_key,
        )
    if not isinstance(inject_response, Mapping):
        inject_response = {"raw": str(inject_response)}
    if not inject_response.get("ok"):
        return ProbeResult(
            ok=False, inj_id=inj_id, bucket=bucket,
            K=None, observed_snr=None, observed_event_specnum=None,
            matched_at_unix=None, n_matches=0,
            elapsed_s=float(time_fn()) - t0,
            reason=f"inject_failed: {inject_response.get('error', 'unknown')}",
            inject_response=dict(inject_response),
            active_key=active_key,
        )
    # 3. Update the active-inject row with the derived apply_at_specnum
    #    so the registry is self-describing for the dashboard's UI.
    derived_apply_at = _extract_apply_at(inject_response)
    if derived_apply_at is not None:
        try:
            publish_active_inject(
                store,
                inj_id=inj_id,
                dm_pc_cm3=float(dm_pc_cm3),
                l_rad=float(l_rad),
                m_rad=float(m_rad),
                width_samples=int(width_samples),
                fluence_jy_ms=float(fluence_jy_ms),
                apply_at_specnum=int(derived_apply_at),
                fired_at_unix=t0,
                ttl_s=float(ttl_s),
                fired_by=str(user or "calibration_probe"),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "publish_active_inject re-write for %s failed: %s",
                inj_id, exc,
            )
    # 4. Poll the match-event key.
    deadline = t0 + float(poll_timeout_s)
    while True:
        now = float(time_fn())
        match = get_match_event(store, inj_id)
        if isinstance(match, Mapping):
            best = match.get("best")
            n_matches = int(match.get("n_matches", 0) or 0)
            if isinstance(best, Mapping):
                observed_snr = best.get("observed_snr")
                observed_specnum = best.get("observed_event_specnum")
                matched_at = best.get("matched_at_unix")
                if isinstance(observed_snr, (int, float)) and observed_snr > 0:
                    # 2026-06-10 saturation guard: the search-side
                    # detector clips its σ-normalised input to ±250σ
                    # (cube_pipeline detector_input_clip_sigma), so an
                    # observed SNR at/above that rail carries NO
                    # amplitude information — computing K from it
                    # stores garbage (observed live: a DM-1000 probe
                    # at 1.2e-3 Jy·ms reported snr=250.25 → K=14448 vs
                    # the true ≈3300). Refuse to calibrate; the ladder
                    # retries at LOWER fluence.
                    if float(observed_snr) >= SATURATION_OBSERVED_SNR:
                        return ProbeResult(
                            ok=False, inj_id=inj_id, bucket=bucket,
                            K=None,
                            observed_snr=float(observed_snr),
                            observed_event_specnum=(
                                int(observed_specnum)
                                if isinstance(observed_specnum, int)
                                else None
                            ),
                            matched_at_unix=(
                                float(matched_at)
                                if isinstance(matched_at, (int, float))
                                else None
                            ),
                            n_matches=n_matches,
                            elapsed_s=now - t0,
                            reason=(
                                "saturated: observed_snr="
                                f"{float(observed_snr):.1f} >= "
                                f"{SATURATION_OBSERVED_SNR:.0f} (detector "
                                "input clip rail) — K not stored; retry "
                                "at lower fluence"
                            ),
                            inject_response=dict(inject_response),
                            active_key=active_key,
                        )
                    K = float(best.get("K_inferred", 0.0) or 0.0)
                    if not (math.isfinite(K) and K > 0.0):
                        # Best didn't compute a usable K (e.g. fluence
                        # was zero). Bail.
                        return ProbeResult(
                            ok=False, inj_id=inj_id, bucket=bucket,
                            K=K if math.isfinite(K) else None,
                            observed_snr=float(observed_snr),
                            observed_event_specnum=(
                                int(observed_specnum)
                                if isinstance(observed_specnum, int)
                                else None
                            ),
                            matched_at_unix=(
                                float(matched_at)
                                if isinstance(matched_at, (int, float))
                                else None
                            ),
                            n_matches=n_matches,
                            elapsed_s=now - t0,
                            reason="bad_K",
                            inject_response=dict(inject_response),
                            active_key=active_key,
                        )
                    obs_l = float(best.get("observed_l_rad", 0.0))
                    obs_m = float(best.get("observed_m_rad", 0.0))
                    obs_w = int(best.get("observed_width_samples", 0))
                    lm_dist = math.hypot(
                        obs_l - float(l_rad), obs_m - float(m_rad),
                    )
                    entry = CalibrationEntry(
                        bucket=bucket,
                        dm_pc_cm3_rounded=int(
                            round(float(dm_pc_cm3) / DM_BUCKET_PC_CC)
                            * int(DM_BUCKET_PC_CC),
                        ),
                        width_samples=int(width_samples),
                        K=K,
                        last_fluence_jy_ms=float(fluence_jy_ms),
                        last_observed_snr=float(observed_snr),
                        last_inj_id=inj_id,
                        last_calibrated_at_unix=(
                            float(matched_at)
                            if isinstance(matched_at, (int, float))
                            else now
                        ),
                        actor=str(user or "calibration_probe"),
                    )
                    try:
                        cal_key = cs.put(entry)
                    except Exception as exc:  # noqa: BLE001
                        LOG.error(
                            "CalibrationStore.put(%s) failed: %s",
                            entry.bucket, exc,
                        )
                        return ProbeResult(
                            ok=False, inj_id=inj_id, bucket=bucket,
                            K=K, observed_snr=float(observed_snr),
                            observed_event_specnum=(
                                int(observed_specnum)
                                if isinstance(observed_specnum, int)
                                else None
                            ),
                            matched_at_unix=(
                                float(matched_at)
                                if isinstance(matched_at, (int, float))
                                else None
                            ),
                            n_matches=n_matches,
                            elapsed_s=now - t0,
                            reason=f"etcd_failure_store: {exc}",
                            inject_response=dict(inject_response),
                            active_key=active_key,
                        )
                    return ProbeResult(
                        ok=True, inj_id=inj_id, bucket=bucket,
                        K=K,
                        observed_snr=float(observed_snr),
                        observed_event_specnum=(
                            int(observed_specnum)
                            if isinstance(observed_specnum, int)
                            else None
                        ),
                        matched_at_unix=(
                            float(matched_at)
                            if isinstance(matched_at, (int, float))
                            else None
                        ),
                        n_matches=n_matches,
                        elapsed_s=now - t0,
                        reason="ok",
                        inject_response=dict(inject_response),
                        active_key=active_key,
                        calibration_key=cal_key,
                        observed_l_rad=obs_l,
                        observed_m_rad=obs_m,
                        observed_width_samples=obs_w,
                        match_lm_dist_rad=lm_dist,
                    )
        if now >= deadline:
            return ProbeResult(
                ok=False, inj_id=inj_id, bucket=bucket,
                K=None, observed_snr=None, observed_event_specnum=None,
                matched_at_unix=None, n_matches=0,
                elapsed_s=now - t0,
                reason="no_match",
                inject_response=dict(inject_response),
                active_key=active_key,
            )
        sleep_fn(float(poll_interval_s))


def _build_probe_inj_id(
    *, prefix: str, dm: float, width: int, timestamp: float,
) -> str:
    """Build a unique-ish inj_id for a probe.

    Format: ``{prefix}_dm{rounded}_w{width}_t{unix_int}``.
    """
    dm_int = int(round(float(dm)))
    return (
        f"{prefix}_dm{dm_int}_w{int(width)}_t{int(round(float(timestamp)))}"
    )


# ---------------------------------------------------------------------------
# Health pre-flight + SNR ladder (T6 + T7 — 2026-06-07)
# ---------------------------------------------------------------------------


#: How recent the corr_fast service-state mon-key must be for the
#: precheck to consider that chgroup "alive". Default 30 s comfortably
#: outlasts the 16-block heartbeat at any sane block cadence.
DEFAULT_CORR_FAST_MAX_AGE_S: float = 30.0


#: How recent the search-compute mon-key must be for the precheck to
#: consider a search half "alive". The publisher fires every
#: ``cube_progress`` tick (≈ 2 s in production), so 30 s is generous.
DEFAULT_SEARCH_MAX_AGE_S: float = 30.0


#: Default SNR-ladder fluence multipliers for
#: :func:`fire_calibration_probe_with_ladder`. The first attempt is at
#: the requested ``fluence_jy_ms``; ``no_match`` falls back to ``×2``,
#: then ``×4``. Escalating is only safe because the 2026-06-09 default
#: fluence rescale (see :data:`DEFAULT_CALIBRATION_FLUENCE`) keeps
#: even the ×4 step ~3 orders of magnitude below fp16 saturation;
#: with the old 100 Jy·ms base the ladder drove probes deeper INTO
#: saturation, which is why it never rescued a no_match.
DEFAULT_FLUENCE_LADDER: Tuple[float, ...] = (1.0, 2.0, 4.0)

#: Hard ceiling on any single laddered probe fluence (Jy·ms).
#: 2026-06-10: the fluence-response sweep located the fp16-overflow
#: cliff at ~1.4e-3 Jy·ms (w=4): above it the cuFFT-cfp16 butterflies
#: overflow and the probe either vanishes (pre-hardening) or comes
#: back as a saturated ~250σ candidate — useless for fitting K either
#: way. Cap the ladder just below the cliff so escalation stays inside
#: the clean linear window (4e-4 .. 1.2e-3).
DEFAULT_MAX_PROBE_FLUENCE: float = 1.2e-3

#: Observed-SNR rail above which a match is treated as SATURATED and
#: K is NOT stored. The search detector clips its σ-normalised input
#: to ±250σ (cube_pipeline detector_input_clip_sigma = 250), so any
#: observed SNR at/near that value is amplitude-blind. 240 leaves a
#: little headroom for boxcar/normalisation wiggle around the rail.
SATURATION_OBSERVED_SNR: float = 240.0

#: Pause between ladder attempts (seconds). A probe absorbed into the
#: detector's sigma_k EMA locally inflates the noise estimate at its
#: own (DM, pixel) for ~minutes; an immediate brighter refire at the
#: SAME DM/position fights its predecessor's inflation. 60 s lets
#: sigma_k partially recover (live sweeps showed back-to-back probes
#: 45 s apart suppressing each other even at 10× the fluence).
DEFAULT_LADDER_STEP_DELAY_S: float = 60.0


@dataclass(frozen=True)
class HealthSnapshot:
    """Pre-flight snapshot returned by :func:`precheck_calibration_health`."""
    ok: bool
    reason: str
    inject_baselines: Dict[int, int]
    corr_fast_seen_chgroups: Tuple[int, ...]
    stale_chgroups: Tuple[int, ...]
    sick_search_halves: Tuple[Tuple[int, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_corr_fast_state(store: Any, chgroup: int) -> Optional[Mapping[str, Any]]:
    """Best-effort read of ``/mon/corr_rt/<chgroup>/corr_fast``.

    Returns ``None`` on any error / missing key (the caller treats that
    as "stale / unknown", which is itself a precheck failure).
    """
    key = f"/mon/corr_rt/{int(chgroup)}/corr_fast"
    try:
        raw = store.get_dict(key)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("precheck: get_dict(%s) failed: %s", key, exc)
        return None
    if not isinstance(raw, Mapping):
        return None
    return raw


def _read_search_compute_state(
    store: Any, search_node_id: int, gpu_half: int,
) -> Optional[Mapping[str, Any]]:
    """Best-effort read of ``/mon/search_rt/<sid>/compute/<g>``."""
    key = f"/mon/search_rt/{int(search_node_id)}/compute/{int(gpu_half)}"
    try:
        raw = store.get_dict(key)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("precheck: get_dict(%s) failed: %s", key, exc)
        return None
    if not isinstance(raw, Mapping):
        return None
    return raw


def precheck_calibration_health(
    store: Any,
    *,
    chgroups: Optional[Iterable[int]] = None,
    search_halves: Iterable[Tuple[int, int]] = (
        (1, 0), (1, 1), (2, 0), (2, 1),
        (9, 0), (9, 1), (13, 0), (13, 1),
    ),
    corr_fast_max_age_s: float = DEFAULT_CORR_FAST_MAX_AGE_S,
    search_max_age_s: float = DEFAULT_SEARCH_MAX_AGE_S,
    require_metering_inactive: bool = True,
    time_fn: Any = time.time,
) -> HealthSnapshot:
    """Pre-flight health check before firing a calibration probe.

    Verifies, before any injection is queued, that:

    * Every chgroup in ``chgroups`` has a recent
      ``/mon/corr_rt/<chgroup>/corr_fast`` heartbeat (default <
      30 s old). Stale chgroups are listed in ``stale_chgroups`` and
      surfaced via the ``reason`` string.
    * Every search half in ``search_halves`` has a recent
      ``/mon/search_rt/<sid>/compute/<g>`` heartbeat. When
      ``require_metering_inactive=True``, also asserts
      ``c1_metering_active=0`` — running calibration into a metered
      half is the failure mode the new T3 bypass guards against, but
      pre-flight is still cheaper than a 30 s ``no_match`` poll.

    Returns a :class:`HealthSnapshot` whose ``inject_baselines`` field
    is a per-chgroup snapshot of the current ``inject_n_queued``
    counter. The caller uses these baselines to detect partial
    fan-out *after* firing (a genuine miss leaves
    ``inject_n_queued`` unchanged on the affected chgroup).
    """
    chgroups_tuple = tuple(
        int(c) for c in (chgroups if chgroups is not None else range(16))
    )
    now = float(time_fn())
    seen: List[int] = []
    stale: List[int] = []
    inject_baselines: Dict[int, int] = {}
    for cg in chgroups_tuple:
        state = _read_corr_fast_state(store, cg)
        if state is None:
            stale.append(cg)
            continue
        ts_wall = state.get("ts_wall_unix")
        if isinstance(ts_wall, (int, float)) and (
            now - float(ts_wall) <= corr_fast_max_age_s
        ):
            seen.append(cg)
            try:
                baseline = int(state.get("inject_n_queued") or 0)
            except (TypeError, ValueError):
                baseline = 0
            inject_baselines[cg] = baseline
        else:
            stale.append(cg)
    sick: List[Tuple[int, int]] = []
    for sid, g in search_halves:
        state = _read_search_compute_state(store, sid, g)
        if state is None:
            sick.append((int(sid), int(g)))
            continue
        ts_wall = state.get("ts_wall_unix")
        if not (
            isinstance(ts_wall, (int, float))
            and (now - float(ts_wall) <= search_max_age_s)
        ):
            sick.append((int(sid), int(g)))
            continue
        if require_metering_inactive:
            metering_active = int(state.get("c1_metering_active") or 0)
            if metering_active:
                sick.append((int(sid), int(g)))

    if stale:
        return HealthSnapshot(
            ok=False,
            reason=(
                "corr_fast_stale: chgroups="
                + ",".join(str(c) for c in stale)
            ),
            inject_baselines=inject_baselines,
            corr_fast_seen_chgroups=tuple(seen),
            stale_chgroups=tuple(stale),
            sick_search_halves=tuple(sick),
        )
    if sick:
        return HealthSnapshot(
            ok=False,
            reason=(
                "search_unhealthy: halves="
                + ",".join(f"s{s}g{g}" for s, g in sick)
            ),
            inject_baselines=inject_baselines,
            corr_fast_seen_chgroups=tuple(seen),
            stale_chgroups=tuple(stale),
            sick_search_halves=tuple(sick),
        )
    return HealthSnapshot(
        ok=True,
        reason="ok",
        inject_baselines=inject_baselines,
        corr_fast_seen_chgroups=tuple(seen),
        stale_chgroups=tuple(stale),
        sick_search_halves=tuple(sick),
    )


def precheck_inject_fan_out(
    store: Any,
    *,
    baselines: Mapping[int, int],
    poll_timeout_s: float = 5.0,
    poll_interval_s: float = 0.5,
    time_fn: Any = time.time,
    sleep_fn: Any = time.sleep,
) -> Tuple[bool, Tuple[int, ...]]:
    """Verify each chgroup's ``inject_n_queued`` advanced past its
    baseline within ``poll_timeout_s``.

    Returns ``(ok, lagging_chgroups)``. ``ok=True`` iff every chgroup
    in ``baselines`` saw at least one new queued event. ``ok=False``
    surfaces the chgroups whose counter did NOT advance — the caller
    converts that into a ``partial_fan_out`` reason string instead of
    waiting the full match-poll timeout.
    """
    if not baselines:
        return True, tuple()
    deadline = float(time_fn()) + float(poll_timeout_s)
    pending = dict(baselines)
    while True:
        next_pending: Dict[int, int] = {}
        for cg, baseline in pending.items():
            state = _read_corr_fast_state(store, cg)
            if state is None:
                next_pending[cg] = baseline
                continue
            try:
                cur = int(state.get("inject_n_queued") or 0)
            except (TypeError, ValueError):
                cur = 0
            if cur > int(baseline):
                continue  # advanced — drop from pending
            next_pending[cg] = baseline
        if not next_pending:
            return True, tuple()
        if float(time_fn()) >= deadline:
            return False, tuple(sorted(next_pending.keys()))
        sleep_fn(float(poll_interval_s))
        pending = next_pending


def fire_calibration_probe_with_ladder(
    store: Any,
    *,
    inject_fn: Any,
    dm_pc_cm3: float,
    l_rad: float = 0.0,
    m_rad: float = 0.0,
    width_samples: int = DEFAULT_CALIBRATION_WIDTH,
    fluence_jy_ms: float = DEFAULT_CALIBRATION_FLUENCE,
    profile: str = DEFAULT_CALIBRATION_PROFILE,
    margin_blocks: Optional[int] = None,
    chgroups: Optional[Iterable[int]] = None,
    inj_id_prefix: str = "cal_probe",
    user: Optional[str] = None,
    poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ttl_s: float = DEFAULT_INJECT_TTL_S,
    cal_store: Optional[CalibrationStore] = None,
    health_check: bool = True,
    fluence_ladder: Iterable[float] = DEFAULT_FLUENCE_LADDER,
    max_probe_fluence: float = DEFAULT_MAX_PROBE_FLUENCE,
    ladder_step_delay_s: float = DEFAULT_LADDER_STEP_DELAY_S,
    fan_out_check_timeout_s: float = 5.0,
    time_fn: Any = time.time,
    sleep_fn: Any = time.sleep,
) -> "List[ProbeResult]":
    """Health-gated, SNR-laddered calibration probe (T6 + T7).

    Workflow:

      1. **T6 pre-flight**: when ``health_check`` is True (default),
         call :func:`precheck_calibration_health` to verify every corr
         node and search half is alive and not in active C1 metering.
         If the precheck fails, return a single :class:`ProbeResult`
         with ``ok=False`` and ``reason="health_check_failed:<details>"``
         BEFORE firing any injection.

      2. **First attempt**: fire :func:`fire_calibration_probe` at the
         operator's requested ``fluence_jy_ms``.

      3. **Post-fire fan-out check**: between the inject and the match
         poll, call :func:`precheck_inject_fan_out` against the
         baselines captured in step 1. If any chgroup did not see its
         ``inject_n_queued`` advance, return ``ok=False`` with
         ``reason="partial_fan_out:cn=[..]"`` instead of waiting the
         full ``poll_timeout_s`` for a ``no_match`` that would have
         been ambiguous.

      4. **T7 SNR ladder**: on a ``no_match`` (the only retriable
         class — partial_fan_out, inject_failed, etc. abort), retry
         the probe at successive ``fluence_ladder`` multipliers
         (default ×1, ×2, ×4). The first ``ok=True`` wins; the result
         list contains every attempt for forensic auditing.

         2026-06-09 saturation + sigma_k guards: every attempt's
         fluence is clamped to ``max_probe_fluence`` (a saturated
         probe clips its own SNR and leaves a railed sliding-mean
         ghost), the ladder stops escalating once the clamp engages,
         and ``ladder_step_delay_s`` is slept between attempts so the
         previous probe's sigma_k inflation at the same (DM, pixel)
         partially decays before the refire.
    
    Returns a list of :class:`ProbeResult` (one per attempt). The
    final entry is the operator-facing result. The list is intended
    for the audit trail / dashboard "calibration log" pane so the
    operator can see e.g. "first probe was metered, second succeeded
    at 2× fluence".
    """
    ladder = tuple(float(m) for m in fluence_ladder)
    if not ladder:
        ladder = (1.0,)

    snapshot: Optional[HealthSnapshot] = None
    if health_check:
        snapshot = precheck_calibration_health(
            store, chgroups=chgroups, time_fn=time_fn,
        )
        if not snapshot.ok:
            return [
                ProbeResult(
                    ok=False,
                    inj_id="",
                    bucket=bucket_key(dm_pc_cm3),
                    K=None,
                    observed_snr=None,
                    observed_event_specnum=None,
                    matched_at_unix=None,
                    n_matches=0,
                    elapsed_s=0.0,
                    reason=f"health_check_failed: {snapshot.reason}",
                )
            ]
    baselines: Dict[int, int] = (
        dict(snapshot.inject_baselines) if snapshot is not None else {}
    )

    results: List[ProbeResult] = []
    clamp = float(max_probe_fluence)
    # 2026-06-10: dynamic multiplier queue so a SATURATED attempt
    # (observed SNR pinned at the detector's ±250σ input-clip rail —
    # see fire_calibration_probe) can retry DOWNWARD. The classic
    # no_match path still walks the configured ascending ladder.
    pending: List[float] = list(ladder)
    descents = 0
    step_idx = -1
    while pending:
        mult = pending.pop(0)
        step_idx += 1
        if step_idx > 0 and ladder_step_delay_s > 0.0:
            LOG.info(
                "calibration ladder: sleeping %.0fs before attempt %d "
                "(sigma_k recovery)",
                float(ladder_step_delay_s), step_idx + 1,
            )
            sleep_fn(float(ladder_step_delay_s))
        attempt_fluence = float(fluence_jy_ms) * float(mult)
        clamped = clamp > 0.0 and attempt_fluence > clamp
        if clamped:
            LOG.warning(
                "calibration ladder: attempt %d fluence %.4g Jy*ms "
                "exceeds max_probe_fluence=%.4g; clamping (saturation "
                "guard)",
                step_idx + 1, attempt_fluence, clamp,
            )
            attempt_fluence = clamp
        result = fire_calibration_probe(
            store,
            inject_fn=inject_fn,
            dm_pc_cm3=dm_pc_cm3,
            l_rad=l_rad,
            m_rad=m_rad,
            width_samples=width_samples,
            fluence_jy_ms=attempt_fluence,
            profile=profile,
            margin_blocks=margin_blocks,
            chgroups=chgroups,
            inj_id_prefix=inj_id_prefix,
            user=user,
            poll_timeout_s=poll_timeout_s,
            poll_interval_s=poll_interval_s,
            ttl_s=ttl_s,
            cal_store=cal_store,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )

        # Post-fire fan-out check: do this once per attempt before we
        # accept ``no_match`` as a real failure. Skip when the inject
        # itself failed (no fan-out to verify).
        if (
            health_check
            and baselines
            and not result.ok
            and result.reason.startswith("inject_failed") is False
        ):
            ok_fan, lagging = precheck_inject_fan_out(
                store,
                baselines=baselines,
                poll_timeout_s=fan_out_check_timeout_s,
                poll_interval_s=poll_interval_s,
                time_fn=time_fn,
                sleep_fn=sleep_fn,
            )
            if not ok_fan:
                # Replace the no_match reason with the more informative
                # partial_fan_out reason; keep all other fields intact
                # so the dashboard sees the actual inj_id we tried.
                result = ProbeResult(
                    ok=False,
                    inj_id=result.inj_id,
                    bucket=result.bucket,
                    K=None,
                    observed_snr=None,
                    observed_event_specnum=None,
                    matched_at_unix=None,
                    n_matches=0,
                    elapsed_s=result.elapsed_s,
                    reason=(
                        "partial_fan_out: cn=["
                        + ",".join(str(c) for c in lagging)
                        + "]"
                    ),
                    inject_response=result.inject_response,
                    active_key=result.active_key,
                )
                results.append(result)
                # Partial fan-out is a hard fail — escalating fluence
                # won't help if some corr nodes never queued the probe.
                return results
            # Fan-out is fine; refresh baselines so the next ladder
            # attempt's fan-out check measures advances against THIS
            # attempt's queued state, not the very first.
            for cg, baseline in baselines.items():
                state = _read_corr_fast_state(store, cg)
                if isinstance(state, Mapping):
                    try:
                        baselines[cg] = int(
                            state.get("inject_n_queued") or baseline
                        )
                    except (TypeError, ValueError):
                        pass

        results.append(result)
        if result.ok:
            return results
        # SATURATED: the probe was detected but its SNR is pinned at
        # the detector's input-clip rail — escalating would only
        # saturate harder. Descend ×¼ (at most twice) instead.
        if result.reason.startswith("saturated"):
            if descents < 2:
                descents += 1
                down = float(mult) / 4.0
                LOG.warning(
                    "calibration ladder: attempt %d SATURATED "
                    "(observed_snr at clip rail); descending to "
                    "fluence multiplier %.3g",
                    step_idx + 1, down,
                )
                pending = [down]
                continue
            return results
        # Only retry on no_match; everything else is a hard fail.
        if not result.reason.startswith("no_match"):
            return results
        # Once the saturation clamp engages, escalating further is
        # pointless (every subsequent attempt would fire at the same
        # clamped fluence).
        if clamped:
            return results
    return results


def _extract_apply_at(inject_response: Mapping[str, Any]) -> Optional[int]:
    """Pull the corr-side apply_at_specnum out of a
    ``control_inject_pulse`` response (or ``None`` if absent).
    """
    val = inject_response.get("val")
    if isinstance(val, Mapping):
        apply_at = val.get("apply_at_specnum")
        if isinstance(apply_at, int):
            return apply_at
    return None


# ---------------------------------------------------------------------------
# Delete the fluence/SNR calibration table (Control-tab verb)
# ---------------------------------------------------------------------------


def delete_snr_calibrations(
    store: Any,
    *,
    dry_run: bool = True,
    user: Optional[str] = None,
    cal_store: Optional[CalibrationStore] = None,
) -> Dict[str, Any]:
    """Preview or delete the per-``(DM, width)`` injection fluence→SNR
    ``K`` calibration table at ``/cnf/inject/snr_calibration/*``.

    This is the bootstrap table the operator builds with
    :func:`fire_calibration_probe` (Control-tab "Calibrate"); wiping it
    lets them re-measure the fluence/SNR relation from scratch. It does
    **NOT** touch the on-sky beamformer-weights K-cal table (that lives
    on the corr nodes and is handled by :mod:`fleet_kcal`).

    Parameters
    ----------
    store:
        DsaStore-like object (the dashboard's ``ControlStore``).
    dry_run:
        When ``True`` (default), only enumerates the present buckets;
        nothing is removed.
    user:
        Operator label for the audit row.
    cal_store:
        Optional pre-built :class:`CalibrationStore` (tests inject a fake).

    Returns
    -------
    dict
        ``{"ok", "dry_run", "buckets": [...], "summary": {...}}`` where
        each bucket row carries ``status`` ∈ ``{exists, deleted,
        not_found, error}``.
    """
    cs = cal_store if cal_store is not None else CalibrationStore(store)
    try:
        entries = cs.list_all()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("delete_snr_calibrations: list_all failed")
        summary = {
            "n_buckets": 0, "n_deleted": 0, "n_present": 0, "n_failed": 0,
            "dry_run": bool(dry_run), "prefix": CALIBRATION_PREFIX,
        }
        return {
            "ok": False, "dry_run": bool(dry_run),
            "error": f"list_failed: {exc}", "buckets": [], "summary": summary,
        }

    buckets: List[Dict[str, Any]] = []
    for e in entries:
        row: Dict[str, Any] = {
            "bucket": e.bucket,
            "dm_pc_cm3_rounded": e.dm_pc_cm3_rounded,
            "width_samples": e.width_samples,
            "K": e.K,
            "last_fluence_jy_ms": e.last_fluence_jy_ms,
            "last_observed_snr": e.last_observed_snr,
            "status": "exists",
            "error": None,
        }
        if not dry_run:
            try:
                removed = cs.delete(e.bucket)
                row["status"] = "deleted" if removed else "not_found"
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "delete_snr_calibrations: delete(%s) failed: %s",
                    e.bucket, exc,
                )
                row["status"] = "error"
                row["error"] = f"{exc!r}"[:200]
        buckets.append(row)

    summary = {
        "n_buckets": len(buckets),
        "n_deleted": sum(1 for b in buckets if b["status"] == "deleted"),
        "n_present": sum(1 for b in buckets if b["status"] == "exists"),
        "n_not_found": sum(1 for b in buckets if b["status"] == "not_found"),
        "n_failed": sum(1 for b in buckets if b["status"] == "error"),
        "dry_run": bool(dry_run),
        "prefix": CALIBRATION_PREFIX,
    }
    overall_ok = summary["n_failed"] == 0
    _audit_snr_cal_delete(store, summary=summary, ok=overall_ok, user=user)
    return {
        "ok": overall_ok,
        "dry_run": bool(dry_run),
        "buckets": buckets,
        "summary": summary,
    }


def _audit_snr_cal_delete(
    store: Any, *, summary: Dict[str, Any], ok: bool, user: Optional[str],
) -> None:
    """Write one audit row for a fluence/SNR calibration wipe. Swallows
    failures so they can't break the operator-visible response."""
    if store is None:
        return
    try:
        from control_store import audit_log  # local import (test-friendly)
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "delete_snr_calibrations: control_store import failed: %s", exc,
        )
        return
    try:
        audit_log(
            store,
            namespace="dsa_monitor.inject_calibration",
            cn_target="-",
            cmd="delete_snr_cal",
            val=summary,
            ok=bool(ok),
            note=(
                f"dry_run={summary.get('dry_run')} "
                f"n_buckets={summary.get('n_buckets')} "
                f"n_deleted={summary.get('n_deleted')} "
                f"n_failed={summary.get('n_failed')}"
            ),
            user=user,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "delete_snr_calibrations: audit_log failed (continuing): %s", exc,
        )


__all__ = [
    "ACTIVE_INJECT_PREFIX",
    "MATCH_EVENT_PREFIX",
    "CALIBRATION_PREFIX",
    "DM_BUCKET_PC_CC",
    "DEFAULT_CALIBRATION_FLUENCE",
    "DEFAULT_CALIBRATION_WIDTH",
    "DEFAULT_CALIBRATION_PROFILE",
    "DEFAULT_INJECT_TTL_S",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_POLL_TIMEOUT_S",
    "DEFAULT_CORR_FAST_MAX_AGE_S",
    "DEFAULT_SEARCH_MAX_AGE_S",
    "DEFAULT_FLUENCE_LADDER",
    "DEFAULT_MAX_PROBE_FLUENCE",
    "DEFAULT_LADDER_STEP_DELAY_S",
    "LEGACY_BUCKET_INFIX",
    "CalibrationEntry",
    "CalibrationStore",
    "HealthSnapshot",
    "ProbeResult",
    "bucket_key",
    "calibration_key",
    "is_legacy_bucket",
    "build_active_inject_key",
    "build_match_event_key",
    "publish_active_inject",
    "get_match_event",
    "snr_to_fluence",
    "fire_calibration_probe",
    "fire_calibration_probe_with_ladder",
    "precheck_calibration_health",
    "precheck_inject_fan_out",
    "delete_snr_calibrations",
]
