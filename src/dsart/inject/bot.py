"""Injection bot: hourly end-to-end test injections + Slack reporting.

Standalone h23 service (``systemd/dsart_inject_bot.service``,
modeled on the annotation relay). Every ``interval_s`` (default 1 h,
jittered) it:

1. pre-checks fleet health (mirrors
   ``tools/dashboard/dsa_monitor/inject_calibration.precheck_calibration_health``
   — corr_fast heartbeats on all 16 chgroups, search-compute heartbeats
   on all 8 halves, ``c1_metering_active`` clear);
2. picks a random DM from ``dm_choices`` and a random target SNR in
   ``[target_snr_min, target_snr_max]`` — deliberately below the
   dashboard's ``IMAGER_SAFE_OBSERVED_SNR`` (30σ predicted) brightness
   guard, which this service never overrides (``allow_bright`` is
   intentionally not plumbed);
3. refreshes the DM bucket's K calibration first when it is older than
   ``k_max_age_s``, missing, or was measured at a pointing declination
   more than ``k_dec_tol_deg`` away from the current one (the K store
   itself carries no dec provenance — the bot records the dec it
   observed at calibration time in its own state file);
4. fires the injection through the dashboard's ``POST /control/inject``
   (so the K lookup, payload validation, and the fp16-imager brightness
   guard are shared with operator-fired injections), posts a plain-text
   "injection sent" Slack message with the full parameter set, and
   threads a "recovered / NOT recovered" reply once the C2 matcher's
   ``/mon/dsart/inject/matches/<inj_id>`` doc and the archived event
   settle;
5. appends one JSON line per attempt to ``results_path`` and, once a
   day at ``summary_hour_utc``, posts a summary text (counts, per-stage
   miss attribution, SNR recovery stats) with the publication-style
   summary figures threaded under it — the only images this service
   ever uploads.

Loss-stage attribution (see ``Outcome``):

* ``missed_search_or_c1`` — the C2 matcher never published a match doc:
  the search nodes never emitted a coincident C1 row (or C1->C2 transport
  failed).
* ``missed_c2`` — a match doc exists (C1 rows arrived and matched) but
  no archived event referenced the ``inj_id`` within the recovery
  window: the coincidencer matched members but never fired/archived an
  event.
* ``missed_c3`` — an archived, injection-tagged event exists but C3
  decided something other than KEEP. ``cube_veto.decide`` always KEEPs
  known injections, so this should never fire; it is logged loudly as
  an anomaly (it means the injection marker failed to propagate).

Everything is best-effort: a failure in any one cycle (Slack down,
dashboard restarting, etcd hiccup) is recorded and never aborts the
loop. The service never writes to etcd and never touches
``/dataz/dsa110/candidates`` — its only writes are its own JSONL/state/
PNG files and HTTP POSTs to the dashboard, which owns the etcd side.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from dsart.services.slack_notify import SlackNotifier, SlackNotifyConfig

LOG = logging.getLogger("dsart.inject.bot")

# ---------------------------------------------------------------------------
# Constants mirrored from the dashboard's inject_calibration module.
# tools/dashboard/dsa_monitor is not an importable package from the
# service env, so the handful of values the bot needs are pinned
# here; ``tests/test_inject_bot.py`` cross-checks them against the
# dashboard source so drift is caught in CI, same pattern the matcher
# uses for ACTIVE_INJECT_PREFIX.
# ---------------------------------------------------------------------------

#: C2 matcher's per-injection match doc (inject_match.MATCH_EVENT_PREFIX).
MATCH_EVENT_PREFIX = "/mon/dsart/inject/matches/"
#: etcd prefix of the per-bucket K entries (mirrors
#: tools/dashboard/dsa_monitor/inject_calibration.py). The bot writes
#: here only from the recal guard, to restore a known-good entry after
#: rejecting an anomalous probe.
SNR_CALIBRATION_PREFIX = "/cnf/inject/snr_calibration/"

#: DM bucket granularity of the K store (inject_calibration.DM_BUCKET_PC_CC).
DM_BUCKET_PC_CC = 50.0

#: Probe width every K in the store was measured at
#: (inject_calibration.DEFAULT_CALIBRATION_WIDTH, NATIVE 32.768 us
#: samples). Sub-search-sample widths make observed SNR scale linearly
#: with fluence, so target-SNR shots are only self-consistent at the
#: calibration width — the bot pins its width to this.
CALIBRATION_WIDTH_SAMPLES = 4

#: Search halves the health gate checks (sid, gpu_half) — mirrors
#: inject_calibration.precheck_calibration_health's default.
SEARCH_HALVES: Tuple[Tuple[int, int], ...] = (
    (1, 0), (1, 1), (2, 0), (2, 1), (9, 0), (9, 1), (13, 0), (13, 1),
)

#: Heartbeat staleness bounds (inject_calibration.DEFAULT_CORR_FAST_MAX_AGE_S
#: / DEFAULT_SEARCH_MAX_AGE_S).
CORR_FAST_MAX_AGE_S = 30.0
SEARCH_MAX_AGE_S = 30.0

#: Dashboard hard limit is INJECT_LM_MAX_RAD = 0.0279; beyond ~0.02 the
#: primary beam attenuates and FFT aliasing risk grows (see the
#: /control/inject docstring), so the bot default stays inside it.
DEFAULT_LM_MAX_RAD = 0.02


def bucket_key(dm_pc_cm3: float) -> str:
    """Mirror of ``inject_calibration.bucket_key`` (``dm{round/50*50:04d}``)."""
    if not math.isfinite(dm_pc_cm3):
        raise ValueError(f"dm_pc_cm3={dm_pc_cm3} is not finite")
    dm_round = max(
        0, int(round(float(dm_pc_cm3) / DM_BUCKET_PC_CC) * int(DM_BUCKET_PC_CC)),
    )
    return f"dm{dm_round:04d}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectBotConfig:
    enabled: bool = False
    #: Slack channel + token plumbing (SlackNotifyConfig subset).
    channel: str = ""
    token_file: str = ""
    token_env: str = "SLACK_TOKEN_DSA"
    slack_timeout_s: float = 10.0

    dashboard_base_url: str = "http://localhost:5778"
    #: HTTP timeout for dashboard POSTs. /control/inject_calibrate
    #: blocks through arm + a 30 s match poll, so this must comfortably
    #: exceed calibrate_poll_timeout_s.
    http_timeout_s: float = 180.0

    interval_s: float = 3600.0
    jitter_s: float = 300.0

    dm_choices: Tuple[float, ...] = (500.0, 1000.0, 1500.0, 2000.0)
    target_snr_min: float = 15.0
    target_snr_max: float = 25.0
    width_samples: int = CALIBRATION_WIDTH_SAMPLES
    profile: str = "gaussian"
    lm_max_rad: float = DEFAULT_LM_MAX_RAD

    k_max_age_s: float = 86400.0
    k_dec_tol_deg: float = 0.5
    #: Reject a recalibration whose K moved by more than this factor in
    #: either direction. A railed probe (fp16 detector saturation, e.g.
    #: 226 sigma from a ~27 sigma prediction) or an RFI-storm noise
    #: collapse bakes an absurd K that silently blinds the bucket for a
    #: day (2026-08-09 and 2026-08-11 dm2000 incidents). 0 disables.
    k_recal_max_change_factor: float = 3.0
    calibrate_poll_timeout_s: float = 30.0
    #: Wait after a K-calibration probe before firing the shot. The
    #: probe fires a real C2 trigger whose class holdoff (30 s,
    #: configs/c2_trigger_criteria.yaml) suppresses the next event —
    #: on first start the shot followed the probe by 6 s and was
    #: demoted to log_only (a false missed_c2). 90 s also clears the
    #: probe's sigma_k EMA self-inflation (see the dashboard ladder's
    #: 60 s step delay).
    post_calibration_settle_s: float = 90.0
    #: Beamformer-weights provenance: a K measured before the currently
    #: applied weights is stale regardless of age (new weights change
    #: the fluence -> SNR gain; 2026-08-09 update shifted buckets by
    #: 5-25%). Newest file mtime in this directory is compared against
    #: last_calibrated_at_unix. Empty string disables the check.
    weights_applied_dir: str = (
        "/dataz/dsa110/operations/beamformer_weights/applied"
    )

    recovery_timeout_s: float = 720.0
    recovery_poll_s: float = 10.0

    summary_hour_utc: int = 16

    results_path: str = (
        "/dataz/dsa110/operations/inject/inject_bot_results.jsonl"
    )
    state_path: str = "~/.dsa_monitor/inject_bot_state.json"
    summary_dir: str = "~/.dsa_monitor/inject_bot"
    candidates_root: str = "/dataz/dsa110/candidates"

    pointing_dec_etcd_key: str = "/mon/array/dec"
    #: Throttle for repeated "fleet unhealthy, skipping" Slack warnings.
    unhealthy_warn_interval_s: float = 21600.0
    #: Attention-message streak thresholds: N consecutive missed shots
    #: (a single miss can be threshold luck at the low end of the
    #: target range) / N consecutive not-searching cycles. One message
    #: per streak, re-armed when the condition clears.
    miss_alert_streak: int = 5
    unhealthy_alert_streak: int = 5

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "InjectBotConfig":
        d = d or {}
        dms = d.get("dm_choices", [500.0, 1000.0, 1500.0, 2000.0])
        return cls(
            enabled=bool(d.get("enabled", False)),
            channel=str(d.get("channel", "")),
            token_file=str(d.get("token_file", "")),
            token_env=str(d.get("token_env", "SLACK_TOKEN_DSA")),
            slack_timeout_s=float(d.get("slack_timeout_s", 10.0)),
            dashboard_base_url=str(
                d.get("dashboard_base_url", "http://localhost:5778")),
            http_timeout_s=float(d.get("http_timeout_s", 180.0)),
            interval_s=float(d.get("interval_s", 3600.0)),
            jitter_s=float(d.get("jitter_s", 300.0)),
            dm_choices=tuple(float(x) for x in dms),
            target_snr_min=float(d.get("target_snr_min", 15.0)),
            target_snr_max=float(d.get("target_snr_max", 25.0)),
            width_samples=int(
                d.get("width_samples", CALIBRATION_WIDTH_SAMPLES)),
            profile=str(d.get("profile", "gaussian")),
            lm_max_rad=float(d.get("lm_max_rad", DEFAULT_LM_MAX_RAD)),
            k_max_age_s=float(d.get("k_max_age_s", 86400.0)),
            k_dec_tol_deg=float(d.get("k_dec_tol_deg", 0.5)),
            k_recal_max_change_factor=float(
                d.get("k_recal_max_change_factor", 3.0)),
            calibrate_poll_timeout_s=float(
                d.get("calibrate_poll_timeout_s", 30.0)),
            post_calibration_settle_s=float(
                d.get("post_calibration_settle_s", 90.0)),
            weights_applied_dir=str(d.get(
                "weights_applied_dir",
                "/dataz/dsa110/operations/beamformer_weights/applied")),
            recovery_timeout_s=float(d.get("recovery_timeout_s", 720.0)),
            recovery_poll_s=float(d.get("recovery_poll_s", 10.0)),
            summary_hour_utc=int(d.get("summary_hour_utc", 16)),
            results_path=str(d.get(
                "results_path",
                "/dataz/dsa110/operations/inject/inject_bot_results.jsonl")),
            state_path=str(d.get(
                "state_path", "~/.dsa_monitor/inject_bot_state.json")),
            summary_dir=str(d.get(
                "summary_dir", "~/.dsa_monitor/inject_bot")),
            candidates_root=str(d.get(
                "candidates_root", "/dataz/dsa110/candidates")),
            pointing_dec_etcd_key=str(d.get(
                "pointing_dec_etcd_key", "/mon/array/dec")),
            unhealthy_warn_interval_s=float(
                d.get("unhealthy_warn_interval_s", 21600.0)),
            miss_alert_streak=int(d.get("miss_alert_streak", 5)),
            unhealthy_alert_streak=int(
                d.get("unhealthy_alert_streak", 5)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "InjectBotConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return cls.from_dict(doc.get("inject_bot"))


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


class Outcome:
    """String constants for the per-attempt ``outcome`` field."""
    RECOVERED = "recovered"
    MISSED_SEARCH_OR_C1 = "missed_search_or_c1"
    MISSED_C2 = "missed_c2"
    MISSED_C3 = "missed_c3"
    NOT_SEARCHING = "not_searching"
    GUARD_REJECTED = "guard_rejected"
    FIRE_FAILED = "fire_failed"

    MISSES = (MISSED_SEARCH_OR_C1, MISSED_C2, MISSED_C3)
    ALL = (
        RECOVERED, MISSED_SEARCH_OR_C1, MISSED_C2, MISSED_C3,
        NOT_SEARCHING, GUARD_REJECTED, FIRE_FAILED,
    )


_MISS_EXPLANATIONS = {
    Outcome.MISSED_SEARCH_OR_C1: (
        "lost at search/C1: the C2 matcher never saw a coincident "
        "candidate from any search half"
    ),
    Outcome.MISSED_C2: (
        "lost at C2: matcher saw candidate rows but the coincidencer "
        "never archived an event"
    ),
    Outcome.MISSED_C3: (
        "ANOMALY at C3: event archived and injection-tagged but C3 did "
        "not KEEP it (injections are always kept - marker propagation "
        "failure, investigate)"
    ),
}


# ---------------------------------------------------------------------------
# bot
# ---------------------------------------------------------------------------


class InjectBot:
    """One instance == one poll loop. All collaborators injectable for
    tests: ``store`` needs ``get_dict(key)``; ``http_post_form`` /
    ``http_get`` mimic the two ``requests`` calls; ``notifier`` needs
    ``post_text`` / ``post_file``."""

    def __init__(
        self,
        config: InjectBotConfig,
        *,
        store: Optional[Any] = None,
        notifier: Optional[Any] = None,
        http_post_form: Optional[Callable[..., Tuple[int, Dict[str, Any]]]] = None,
        http_get: Optional[Callable[..., Tuple[int, Dict[str, Any]]]] = None,
        rng: Optional[random.Random] = None,
        time_fn: Callable[[], float] = time.time,
        c2_trigger_reader: Optional[
            Callable[[float, float], List[Tuple[float, str]]]] = None,
        c2_discard_reader: Optional[
            Callable[[float, float], List[Tuple[float, str, str]]]] = None,
    ) -> None:
        self._cfg = config
        self._store = store
        self._c2_trigger_reader = c2_trigger_reader or self._read_c2_triggers
        self._c2_discard_reader = c2_discard_reader or self._read_c2_discards
        self._notifier = notifier or SlackNotifier(SlackNotifyConfig(
            enabled=bool(config.enabled and config.channel),
            channel=config.channel,
            token_file=config.token_file,
            token_env=config.token_env,
            timeout_s=config.slack_timeout_s,
        ))
        self._http_post_form = http_post_form or self._requests_post_form
        self._http_get = http_get or self._requests_get
        self._rng = rng or random.Random()
        self._time = time_fn
        self._state: Dict[str, Any] = {}
        self._state_loaded = False

    # ----- plumbing --------------------------------------------------------

    def _get_store(self) -> Any:
        if self._store is None:
            from dsautils.dsa_store import DsaStore  # heavy; deferred
            self._store = DsaStore()
        return self._store

    def _requests_post_form(
        self, url: str, fields: Mapping[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        import requests
        r = requests.post(
            url, data={k: str(v) for k, v in fields.items()},
            timeout=self._cfg.http_timeout_s,
        )
        try:
            doc = r.json()
        except ValueError:
            doc = {"ok": False, "error": f"non-json response ({r.status_code})"}
        return r.status_code, doc

    def _requests_get(self, url: str) -> Tuple[int, Dict[str, Any]]:
        import requests
        r = requests.get(url, timeout=self._cfg.http_timeout_s)
        try:
            doc = r.json()
        except ValueError:
            doc = {"ok": False, "error": f"non-json response ({r.status_code})"}
        return r.status_code, doc

    def _get_dict(self, key: str) -> Optional[Mapping[str, Any]]:
        try:
            raw = self._get_store().get_dict(key)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("etcd get_dict(%s) failed: %s", key, exc)
            return None
        return raw if isinstance(raw, Mapping) else None

    # ----- persistent state (last-calibration dec, throttles, summary) ----

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_loaded:
            p = Path(self._cfg.state_path).expanduser()
            try:
                self._state = json.loads(p.read_text())
            except (OSError, ValueError):
                self._state = {}
            self._state_loaded = True
        return self._state

    def _save_state(self) -> None:
        p = Path(self._cfg.state_path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(self._state, indent=1, sort_keys=True))
            tmp.replace(p)
        except OSError as exc:
            LOG.warning("state save to %s failed: %s", p, exc)

    # ----- health gate -----------------------------------------------------

    def check_health(self) -> Tuple[bool, str]:
        """Mirror of the dashboard's ``precheck_calibration_health``
        (read-only etcd; see the constants block for provenance)."""
        now = self._time()
        stale: List[int] = []
        for cg in range(16):
            state = self._get_dict(f"/mon/corr_rt/{cg}/corr_fast")
            ts = (state or {}).get("ts_wall_unix")
            if not (
                isinstance(ts, (int, float))
                and now - float(ts) <= CORR_FAST_MAX_AGE_S
            ):
                stale.append(cg)
        if stale:
            return False, "corr_fast_stale: chgroups=" + ",".join(
                str(c) for c in stale)
        sick: List[str] = []
        for sid, g in SEARCH_HALVES:
            state = self._get_dict(f"/mon/search_rt/{sid}/compute/{g}")
            ts = (state or {}).get("ts_wall_unix")
            if not (
                isinstance(ts, (int, float))
                and now - float(ts) <= SEARCH_MAX_AGE_S
            ):
                sick.append(f"s{sid}g{g}")
                continue
            if int((state or {}).get("c1_metering_active") or 0):
                sick.append(f"s{sid}g{g}")
        if sick:
            return False, "search_unhealthy: halves=" + ",".join(sick)
        return True, "ok"

    def read_pointing_dec(self) -> Optional[float]:
        """Live pointing dec (deg); mirrors coincidencer._read_pointing_dec."""
        doc = self._get_dict(self._cfg.pointing_dec_etcd_key)
        try:
            dec = float((doc or {})["dec_deg"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(dec) or not -90.0 <= dec <= 90.0:
            return None
        return dec

    # ----- K freshness -----------------------------------------------------

    def _get_calibration_entry(self, dm: float) -> Optional[Dict[str, Any]]:
        url = f"{self._cfg.dashboard_base_url}/control/inject_calibrations"
        status, doc = self._http_get(url)
        if status != 200 or not doc.get("ok"):
            LOG.warning("inject_calibrations GET failed: %s %s", status, doc)
            return None
        want = bucket_key(dm)
        for entry in doc.get("entries") or []:
            if entry.get("bucket") == want:
                return dict(entry)
        return None

    def _weights_mtime(self) -> Optional[float]:
        """Newest file mtime in the applied beamformer-weights dir, or
        None when the directory is unset, missing, or empty. Best-effort:
        a filesystem hiccup must never block an injection cycle."""
        path = self._cfg.weights_applied_dir
        if not path:
            return None
        try:
            mtimes = [
                e.stat().st_mtime for e in os.scandir(path) if e.is_file()
            ]
            return max(mtimes) if mtimes else None
        except OSError:
            return None

    def ensure_k_fresh(self, dm: float) -> Tuple[bool, Dict[str, Any]]:
        """Recalibrate the DM bucket when K is missing, stale, or was
        measured at a different pointing dec. Returns ``(usable,
        info)``; ``usable`` False means the bucket still has no valid K
        (the cycle is then recorded as fire_failed with the reason)."""
        cfg = self._cfg
        bucket = bucket_key(dm)
        info: Dict[str, Any] = {"bucket": bucket, "recalibrated": False}
        entry = self._get_calibration_entry(dm)
        now = self._time()
        dec_now = self.read_pointing_dec()
        info["pointing_dec_deg"] = dec_now

        cal_decs = self._load_state().setdefault("k_calibration_dec", {})
        reasons: List[str] = []
        if entry is None or not (float(entry.get("K") or 0) > 0):
            reasons.append("no_k")
        else:
            age = now - float(entry.get("last_calibrated_at_unix") or 0)
            info["k_age_s"] = age
            if age > cfg.k_max_age_s:
                reasons.append(f"stale({age / 3600.0:.1f}h)")
            cal_dec = cal_decs.get(bucket)
            if (
                dec_now is not None and cal_dec is not None
                and abs(float(cal_dec) - dec_now) > cfg.k_dec_tol_deg
            ):
                reasons.append(
                    f"dec_changed({float(cal_dec):.2f}->{dec_now:.2f})")
            weights_mtime = self._weights_mtime()
            if (
                weights_mtime is not None
                and float(entry.get("last_calibrated_at_unix") or 0)
                < weights_mtime
            ):
                reasons.append("weights_newer_than_k")

        if reasons:
            prev_entry = dict(entry) if entry is not None else None
            prev_k = float((entry or {}).get("K") or 0)
            info["recal_reasons"] = reasons
            LOG.info("K recal for %s: %s", bucket, ",".join(reasons))
            status, doc = self._http_post_form(
                f"{cfg.dashboard_base_url}/control/inject_calibrate",
                {
                    "dm_pc_cm3": dm,
                    "width_samples": cfg.width_samples,
                    "poll_timeout_s": cfg.calibrate_poll_timeout_s,
                    "user": "inject_bot",
                },
            )
            info["recalibrated"] = bool(doc.get("ok"))
            info["recal_response_status"] = status
            if not doc.get("ok"):
                info["recal_error"] = doc.get(
                    "error") or doc.get("reason") or f"http {status}"
                LOG.warning("K recal for %s failed: %s", bucket, info)
            entry = self._get_calibration_entry(dm)
            new_k = float((entry or {}).get("K") or 0)
            factor = cfg.k_recal_max_change_factor
            if (
                info["recalibrated"] and factor > 0
                and prev_entry is not None and prev_k > 0 and new_k > 0
                and (new_k > prev_k * factor or new_k < prev_k / factor)
            ):
                # A railed probe (detector saturation) or an RFI-storm
                # noise collapse bakes an absurd K; accepting it makes
                # every subsequent shot in the bucket invisibly faint
                # (or railed) until a human notices. Keep the previous
                # K, stamped now so the next cycle does not immediately
                # re-probe, and write it back over the poisoned entry.
                LOG.warning(
                    "K recal for %s REJECTED: K %.0f -> %.0f moved more "
                    "than x%.1f; restoring previous value",
                    bucket, prev_k, new_k, factor)
                restored = dict(prev_entry)
                restored["last_calibrated_at_unix"] = now
                restored["actor"] = "inject_bot_recal_guard"
                try:
                    self._get_store().put_dict(
                        SNR_CALIBRATION_PREFIX + bucket, restored)
                except Exception as exc:  # noqa: BLE001 — keep cycle alive
                    LOG.warning("recal-guard restore write failed: %s", exc)
                info["recal_anomalous"] = True
                info["rejected_K"] = new_k
                entry = restored
            if info["recalibrated"] and dec_now is not None:
                cal_decs[bucket] = dec_now
                self._save_state()
        elif dec_now is not None and bucket not in cal_decs:
            # First sighting of an already-fresh bucket: adopt the
            # current dec as its provenance baseline.
            cal_decs[bucket] = dec_now
            self._save_state()

        if entry is None or not (float(entry.get("K") or 0) > 0):
            info["error"] = "no usable K after recalibration attempt"
            return False, info
        info["K"] = float(entry["K"])
        return True, info

    # ----- one injection cycle ---------------------------------------------

    def _pick_params(self) -> Dict[str, Any]:
        cfg = self._cfg
        r = self._rng
        # Uniform over the inner disc-ish box; both axes independently
        # bounded well inside the dashboard's 0.0279 rad hard limit.
        return {
            "dm_pc_cm3": r.choice(cfg.dm_choices),
            "target_snr": round(
                r.uniform(cfg.target_snr_min, cfg.target_snr_max), 2),
            "l_rad": round(r.uniform(-cfg.lm_max_rad, cfg.lm_max_rad), 6),
            "m_rad": round(r.uniform(-cfg.lm_max_rad, cfg.lm_max_rad), 6),
            "width_samples": cfg.width_samples,
            "profile": cfg.profile,
        }

    def _make_inj_id(self, dm: float) -> str:
        """Short, readable, unique at one shot/hour:
        ``inj_070826_0315`` (UTC day-month-2-digit-year + hour-minute).
        DM and the rest of the parameters live in the Slack message /
        JSONL row, not the id."""
        del dm
        stamp = datetime.fromtimestamp(
            self._time(), timezone.utc).strftime("%d%m%y_%H%M")
        return f"inj_{stamp}"

    def run_cycle(self) -> Dict[str, Any]:
        """One full attempt. Returns the JSONL record (also appended to
        ``results_path``). Never raises."""
        try:
            record = self._run_cycle_inner()
        except Exception as exc:  # noqa: BLE001 — the loop must survive
            LOG.exception("bot cycle failed unexpectedly")
            record = {
                "ts_unix": self._time(),
                "outcome": Outcome.FIRE_FAILED,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record.setdefault("ts_unix", self._time())
        record["ts_utc"] = datetime.fromtimestamp(
            record["ts_unix"], timezone.utc).isoformat()
        self._append_result(record)
        try:
            self._update_streaks_and_escalate(record)
        except Exception:  # noqa: BLE001 — escalation must never kill a cycle
            LOG.exception("streak escalation failed")
        return record

    def _run_cycle_inner(self) -> Dict[str, Any]:
        cfg = self._cfg
        record: Dict[str, Any] = {"ts_unix": self._time()}

        ok, reason = self.check_health()
        if not ok:
            record["outcome"] = Outcome.NOT_SEARCHING
            record["reason"] = reason
            LOG.warning("cycle skipped: %s", reason)
            return record

        params = self._pick_params()
        record.update(params)
        dm = params["dm_pc_cm3"]

        usable, k_info = self.ensure_k_fresh(dm)
        record["k_info"] = k_info
        record["pointing_dec_deg"] = k_info.get("pointing_dec_deg")
        if not usable:
            record["outcome"] = Outcome.FIRE_FAILED
            record["reason"] = k_info.get("error", "no_k")
            return record
        if k_info.get("recalibrated") and cfg.post_calibration_settle_s > 0:
            # The probe fired a real trigger; wait out its class
            # holdoff (and sigma_k settling) or the shot lands as
            # log_only and reads as a false missed_c2.
            LOG.info(
                "post-calibration settle: %.0fs before firing the shot",
                cfg.post_calibration_settle_s)
            self._sleep(cfg.post_calibration_settle_s)

        inj_id = self._make_inj_id(dm)
        record["inj_id"] = inj_id
        status, doc = self._http_post_form(
            f"{cfg.dashboard_base_url}/control/inject",
            {
                "inj_id": inj_id,
                "l_rad": params["l_rad"],
                "m_rad": params["m_rad"],
                "dm_pc_cm3": dm,
                "target_snr": params["target_snr"],
                "width_samples": params["width_samples"],
                "profile": params["profile"],
            },
        )
        record["fire_response_status"] = status
        if not doc.get("ok"):
            record["reason"] = doc.get("error") or f"http {status}"
            record["outcome"] = (
                Outcome.GUARD_REJECTED
                if status == 400 and "imager-safe" in str(doc.get("error"))
                else Outcome.FIRE_FAILED
            )
            LOG.warning("inject POST failed (%s): %s", status, record["reason"])
            return record
        fired_at = self._time()
        record["fired_at_unix"] = fired_at
        record["fluence_jy_ms"] = (doc.get("val") or {}).get("fluence_jy_ms")

        sent_ts = self._post_sent_message(record)

        recovery = self.watch_recovery(inj_id, fired_at)
        record.update(recovery)

        self._post_recovery_message(record, sent_ts=sent_ts)
        return record

    # ----- Slack messages (plain text only) ---------------------------------

    #: Outcome colors for the Slack attachment bar (same mechanism the
    #: annotation relay uses; actual color without emoji).
    _COLOR_RECOVERED = "#2E7D32"
    _COLOR_MISSED = "#C62828"

    def _sent_text(self, record: Mapping[str, Any]) -> str:
        # NBSP joins every number to its unit so Slack's line wrapping
        # can never split them; (l, m) are reported in mrad at two
        # decimals (0.01 mrad ~ 2 arcsec, ample for a message).
        dec = record.get("pointing_dec_deg")
        return (
            "injection sent: `{inj_id}`\n"
            "DM {dm:.0f} pc/cc | target SNR {snr:.1f} | "
            "fluence {fl} | width {w} native | "
            "(l,m) = ({l:+.2f}, {m:+.2f}) mrad | pointing dec {dec}"
        ).format(
            inj_id=record.get("inj_id"),
            dm=float(record.get("dm_pc_cm3") or 0),
            snr=float(record.get("target_snr") or 0),
            fl=(
                f"{float(record['fluence_jy_ms']):.3g} Jy ms"
                if record.get("fluence_jy_ms") is not None else "n/a"
            ),
            w=record.get("width_samples"),
            l=float(record.get("l_rad") or 0) * 1e3,
            m=float(record.get("m_rad") or 0) * 1e3,
            dec=(f"{float(dec):.2f} deg" if dec is not None else "n/a"),
        )

    def _outcome_line(self, record: Mapping[str, Any]) -> str:
        """One line describing how the shot resolved (goes into the
        colored attachment of the edited sent message)."""
        outcome = record.get("outcome")
        if outcome == Outcome.RECOVERED:
            event = record.get("event")
            if event:
                # Same burst-page link the candidate cards carry.
                event_ref = "<{base}/bursts/{ev}|{ev}>".format(
                    base=self._cfg.dashboard_base_url.rstrip("/"), ev=event)
            else:
                event_ref = "(pending archive)"
            dm_obs = record.get("observed_dm_pc_cm3")
            line = (
                "recovered -> {event} | SNR {osnr:.1f} (ratio {ratio:.2f})"
                " | DM {odm} (delta {ddm}) | offset {off}"
            ).format(
                event=event_ref,
                osnr=float(record.get("observed_snr") or 0),
                ratio=float(record.get("snr_ratio") or 0),
                odm=(f"{float(dm_obs):.1f}"
                     if dm_obs is not None else "n/a"),
                ddm=(
                    f"{float(record['delta_dm_pc_cm3']):+.1f}"
                    if record.get("delta_dm_pc_cm3") is not None else "n/a"
                ),
                off=(
                    f"{float(record['offset_arcsec']):.0f} arcsec"
                    if record.get("offset_arcsec") is not None else "n/a"
                ),
            )
            # Cube count is printed ONLY when incomplete: a recovered
            # shot archived with < 8 cubes flags dump-path trouble;
            # a healthy 8/8 stays silent.
            cubes = record.get("cubes")
            if isinstance(cubes, int) and cubes < 8:
                line += f" | cubes {cubes}/8"
            return line
        if (
            outcome == Outcome.MISSED_C2
            and record.get("discarded_event")
        ):
            return (
                "NOT recovered: detected by the search (matched at "
                "{snr}) and C2 triggered `{ev}`, but the archive was "
                "discarded when only {cubes} cube dumps arrived, "
                "likely a cube-dump delivery failure"
            ).format(
                snr=(
                    f"{float(record['observed_snr']):.1f} sigma"
                    if record.get("observed_snr") is not None else "C1"
                ),
                ev=record["discarded_event"],
                cubes=record.get("discarded_cubes") or "partial",
            )
        if (
            outcome == Outcome.MISSED_C2
            and record.get("holdoff_suspect")
        ):
            return (
                "NOT recovered: detected by the search (matched at "
                "{snr}) but no event archived, likely the 30 s "
                "post-trigger holdoff after `{ev}`"
            ).format(
                snr=(
                    f"{float(record['observed_snr']):.1f} sigma"
                    if record.get("observed_snr") is not None else "C1"
                ),
                ev=record["holdoff_suspect"],
            )
        return "NOT recovered: " + _MISS_EXPLANATIONS.get(
            str(outcome), f"outcome={outcome}")

    def _post_sent_message(self, record: Mapping[str, Any]) -> Optional[str]:
        resp = self._notifier.post_text(
            self._sent_text(record) + "\n_awaiting recovery..._")
        if not resp.get("ok"):
            LOG.warning("slack sent-post failed: %s", resp)
            return None
        return resp.get("ts")

    def _post_recovery_message(
        self, record: Mapping[str, Any], *, sent_ts: Optional[str],
    ) -> None:
        """Resolve the shot in Slack: EDIT the sent message in place so
        the channel holds exactly one message per injection, outcome
        readable at a skim (colored attachment bar, no thread click).
        A miss additionally posts its own fresh top-level alert: quiet
        on success, loud on failure."""
        outcome = record.get("outcome")
        recovered = outcome == Outcome.RECOVERED
        line = self._outcome_line(record)
        attachment = {
            "color": (self._COLOR_RECOVERED if recovered
                      else self._COLOR_MISSED),
            "text": line,
            "fallback": line,
        }
        if sent_ts:
            resp = self._notifier.update_text(
                sent_ts, self._sent_text(record),
                attachments=[attachment])
            if not resp.get("ok"):
                LOG.warning("slack sent-edit failed: %s", resp)
                # Degrade to a fresh message so the outcome is never lost.
                self._notifier.post_text(
                    f"`{record.get('inj_id')}`: {line}",
                    attachments=[attachment])
        else:
            self._notifier.post_text(
                f"`{record.get('inj_id')}`: {line}",
                attachments=[attachment])

        if outcome == Outcome.MISSED_C3:
            LOG.error(
                "C3 rejected a tagged injection (%s): marker "
                "propagation failure, investigate", record.get("inj_id"),
            )

    def _update_streaks_and_escalate(self, record: Mapping[str, Any]) -> None:
        """Streak-based attention messages: a single miss (or one
        skipped cycle) never posts anything beyond the edited shot
        message — one miss at the low end of the target range can be
        threshold luck, and a per-miss alert would double-count misses
        for anyone skimming the channel. When ``miss_alert_streak``
        consecutive shots are missed, or the pipeline stays
        not-searching for ``unhealthy_alert_streak`` consecutive
        cycles, ONE attention message posts (per streak — it re-arms
        only after the condition clears). Streaks persist in the state
        file across restarts."""
        state = self._load_state()
        outcome = record.get("outcome")
        if outcome == Outcome.RECOVERED:
            state["miss_streak"] = 0
            state["miss_streak_ids"] = []
            state["unhealthy_streak"] = 0
        elif outcome in Outcome.MISSES:
            state["unhealthy_streak"] = 0
            streak = int(state.get("miss_streak") or 0) + 1
            state["miss_streak"] = streak
            ids = list(state.get("miss_streak_ids") or [])
            ids.append(str(record.get("inj_id")))
            state["miss_streak_ids"] = ids[-10:]
            # Re-alert at every multiple of the threshold (5, 10, 15,
            # ...): a persisting failure re-pings every ~threshold
            # cycles with the updated count, but never per-cycle.
            if (
                self._cfg.miss_alert_streak > 0
                and streak % self._cfg.miss_alert_streak == 0
            ):
                text = (
                    "*ATTENTION*: {n} consecutive test injections missed "
                    "({ids}). The search pipeline may be missing real "
                    "FRBs; latest loss stage: {why}"
                ).format(
                    n=streak,
                    ids=", ".join(f"`{i}`" for i in state["miss_streak_ids"]),
                    why=_MISS_EXPLANATIONS.get(
                        str(outcome), f"outcome={outcome}"),
                )
                resp = self._notifier.post_text(text, attachments=[{
                    "color": self._COLOR_MISSED,
                    "text": "", "fallback": text,
                }])
                if not resp.get("ok"):
                    LOG.warning("slack miss-streak alert failed: %s", resp)
        elif outcome == Outcome.NOT_SEARCHING:
            streak = int(state.get("unhealthy_streak") or 0) + 1
            state["unhealthy_streak"] = streak
            if (
                self._cfg.unhealthy_alert_streak > 0
                and streak % self._cfg.unhealthy_alert_streak == 0
            ):
                hours = streak * self._cfg.interval_s / 3600.0
                text = (
                    "*ATTENTION*: pipeline not searching for {n} "
                    "consecutive test-injection cycles (~{h:.0f} h); "
                    "no shots attempted. Latest reason: {reason}"
                ).format(n=streak, h=hours,
                         reason=record.get("reason", "unknown"))
                resp = self._notifier.post_text(text, attachments=[{
                    "color": self._COLOR_MISSED,
                    "text": "", "fallback": text,
                }])
                if not resp.get("ok"):
                    LOG.warning("slack unhealthy alert failed: %s", resp)
        # guard_rejected / fire_failed leave both streaks unchanged:
        # they are bot-side plumbing, not pipeline sensitivity signals.
        self._save_state()

    # ----- recovery watch ---------------------------------------------------

    def watch_recovery(
        self, inj_id: str, fired_at: float,
    ) -> Dict[str, Any]:
        """Poll the matcher doc + candidate archive until the injection
        settles or ``recovery_timeout_s`` elapses. Returns the fields to
        merge into the attempt record (``outcome`` + observed values)."""
        cfg = self._cfg
        deadline = fired_at + cfg.recovery_timeout_s
        match_doc: Optional[Mapping[str, Any]] = None
        event: Optional[str] = None
        out: Dict[str, Any] = {}
        while True:
            if match_doc is None:
                match_doc = self._get_dict(MATCH_EVENT_PREFIX + inj_id)
            if match_doc is not None and event is None:
                event = self._find_event_for_inj(inj_id, fired_at)
            if event is not None:
                break
            if self._time() >= deadline:
                break
            self._sleep(cfg.recovery_poll_s)

        if match_doc is None:
            out["outcome"] = Outcome.MISSED_SEARCH_OR_C1
            return out

        best = match_doc.get("best") or {}
        out["observed_snr"] = _as_float(best.get("observed_snr"))
        out["observed_dm_pc_cm3"] = _as_float(best.get("observed_dm_pc_cm3"))
        out["observed_l_rad"] = _as_float(best.get("observed_l_rad"))
        out["observed_m_rad"] = _as_float(best.get("observed_m_rad"))
        out["observed_search_node_id"] = best.get("observed_search_node_id")
        out["observed_gpu_half"] = best.get("observed_gpu_half")
        out["n_matches"] = match_doc.get("n_matches")
        inj = match_doc.get("active") or {}
        self._compute_deltas(out, inj)

        if event is None:
            out["outcome"] = Outcome.MISSED_C2
            # Matched at search but no event archived. Two known causes,
            # distinguished via the C2 journal: (a) C2 triggered on the
            # shot itself but DISCARDed the archive when the cube dumps
            # never arrived; (b) another event's 30 s post-trigger
            # holdoff (c2_trigger_criteria.yaml) swallowed the shot.
            try:
                triggers = self._c2_trigger_reader(
                    fired_at - 35.0, fired_at + 25.0)
            except Exception as exc:  # noqa: BLE001 — diagnosis only
                LOG.warning("holdoff-suspect lookup failed: %s", exc)
                triggers = []
            inj_dm = _as_float(inj.get("dm_pc_cm3"))
            own_name = None
            others: List[Tuple[float, str]] = []
            for t in triggers:
                ts, name, t_dm = (tuple(t) + (None,))[:3]
                ts = float(ts)
                # The shot's own trigger: fired after us, at our DM.
                # (Stubbed 2-tuple readers carry no DM and fall through
                # to the holdoff branch, the pre-existing behavior.)
                if (
                    own_name is None and inj_dm is not None
                    and t_dm is not None and ts >= fired_at - 5.0
                    and abs(float(t_dm) - inj_dm) <= max(10.0, 0.01 * inj_dm)
                ):
                    own_name = str(name)
                else:
                    others.append((ts, str(name)))
            if own_name is not None:
                try:
                    discards = self._c2_discard_reader(
                        fired_at, self._time() + 1.0)
                except Exception as exc:  # noqa: BLE001 — diagnosis only
                    LOG.warning("discard lookup failed: %s", exc)
                    discards = []
                for _dts, dname, cubes in discards:
                    if dname == own_name:
                        out["discarded_event"] = dname
                        out["discarded_cubes"] = cubes
                        break
            if "discarded_event" not in out and others:
                ts, name = max(others, key=lambda t: t[0])
                out["holdoff_suspect"] = name
                out["holdoff_suspect_unix"] = ts
            return out
        out["event"] = event
        out.update(self._read_event_details(event))
        if out.get("c3_keep") is False:
            out["outcome"] = Outcome.MISSED_C3
        else:
            out["outcome"] = Outcome.RECOVERED
        return out

    def _compute_deltas(
        self, out: Dict[str, Any], inj: Mapping[str, Any],
    ) -> None:
        il, im = _as_float(inj.get("l_rad")), _as_float(inj.get("m_rad"))
        idm = _as_float(inj.get("dm_pc_cm3"))
        tsnr = _as_float(inj.get("target_snr"))
        ol, om = out.get("observed_l_rad"), out.get("observed_m_rad")
        if None not in (il, im, ol, om):
            out["delta_l_rad"] = ol - il
            out["delta_m_rad"] = om - im
            # (l, m) are direction cosines; near beam centre (|l|,|m| <=
            # 0.028) the offset angle is hypot(dl, dm) radians to O(l^3),
            # converted at 206264.8 arcsec/rad. Both values live in the
            # corr gridder's INSTRUMENT frame (see sky_astrometry.py), so
            # this measures injection->recovery self-consistency; the
            # true-sky m-offset is larger by 1/cos(lat - dec) (~1.21x at
            # dec 71.6).
            out["offset_arcsec"] = math.hypot(
                ol - il, om - im) * (180.0 / math.pi * 3600.0)
        if idm is not None and out.get("observed_dm_pc_cm3") is not None:
            out["delta_dm_pc_cm3"] = out["observed_dm_pc_cm3"] - idm
        if tsnr and out.get("observed_snr") is not None:
            out["snr_ratio"] = out["observed_snr"] / tsnr

    def _find_event_for_inj(
        self, inj_id: str, fired_at: float,
    ) -> Optional[str]:
        """Scan recently-modified candidate dirs for a Level3 JSON whose
        ``injection.inj_ids`` names this injection. Read-only."""
        root = Path(self._cfg.candidates_root)
        try:
            entries = list(root.iterdir())
        except OSError as exc:
            LOG.warning("candidates root %s unreadable: %s", root, exc)
            return None
        for d in entries:
            try:
                if not d.is_dir() or d.stat().st_mtime < fired_at - 60.0:
                    continue
                l3 = d / "Level3" / f"{d.name}.json"
                if not l3.is_file():
                    continue
                doc = json.loads(l3.read_text())
            except (OSError, ValueError):
                continue
            inj_ids = ((doc.get("injection") or {}).get("inj_ids")) or []
            if inj_id in inj_ids:
                return d.name
        return None

    @staticmethod
    def _read_c2_triggers(t0: float, t1: float) -> List[Tuple[float, str]]:
        """C2 trigger broadcasts in ``[t0, t1]`` as ``(unix_ts, event)``,
        parsed from the dsart_c2 user-unit journal (same host, same
        user — the bot runs beside C2 on h23). Best-effort: any failure
        returns an empty list."""
        import re
        import subprocess
        try:
            out = subprocess.run(
                ["journalctl", "--user", "-u", "dsart_c2",
                 "--since", "@%d" % int(t0), "--until", "@%d" % (int(t1) + 1),
                 "--no-pager", "-o", "short-unix"],
                capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception as exc:  # noqa: BLE001
            LOG.warning("journalctl read failed: %s", exc)
            return []
        triggers: List[Tuple[float, str]] = []
        for line in out.splitlines():
            if "DUMP class=" not in line:
                continue
            m_ts = re.match(r"^(\d+\.\d+)", line)
            m_name = re.search(r"name=(\S+)", line)
            m_dm = re.search(r"dm_med=([0-9.]+)", line)
            if m_ts and m_name:
                triggers.append((
                    float(m_ts.group(1)), m_name.group(1),
                    float(m_dm.group(1)) if m_dm else None,
                ))
        return triggers

    @staticmethod
    def _read_c2_discards(t0: float, t1: float) -> List[Tuple[float, str, str]]:
        """C2 archive DISCARDs in ``[t0, t1]`` as ``(unix_ts, event,
        "n/total")`` — events whose cube dumps never arrived ("DISCARD
        <name>: 1/8 cubes after 360s"). Best-effort like
        :meth:`_read_c2_triggers`."""
        import re
        import subprocess
        try:
            out = subprocess.run(
                ["journalctl", "--user", "-u", "dsart_c2",
                 "--since", "@%d" % int(t0), "--until", "@%d" % (int(t1) + 1),
                 "--no-pager", "-o", "short-unix"],
                capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception as exc:  # noqa: BLE001
            LOG.warning("journalctl read failed: %s", exc)
            return []
        discards: List[Tuple[float, str, str]] = []
        for line in out.splitlines():
            m = re.match(
                r"^(\d+\.\d+).*DISCARD (\S+): (\d+/\d+) cubes", line)
            if m:
                discards.append((float(m.group(1)), m.group(2), m.group(3)))
        return discards

    def _read_event_details(self, event: str) -> Dict[str, Any]:
        """Cube count + C3 decision for an archived event (read-only)."""
        out: Dict[str, Any] = {}
        ev_dir = Path(self._cfg.candidates_root) / event
        try:
            out["cubes"] = len(list((ev_dir / "cubes").glob("*.npz")))
        except OSError:
            out["cubes"] = None
        c3_path = ev_dir / "C3_decision.json"
        if c3_path.is_file():
            try:
                c3 = json.loads(c3_path.read_text())
                out["c3_keep"] = bool(c3.get("keep"))
                out["c3_decision"] = "KEEP" if c3.get("keep") else "REJECT"
            except (OSError, ValueError):
                pass
        return out

    # ----- results log -------------------------------------------------------

    def _append_result(self, record: Mapping[str, Any]) -> None:
        p = Path(self._cfg.results_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            LOG.warning("results append to %s failed: %s", p, exc)

    def load_results_since(self, since_unix: float) -> List[Dict[str, Any]]:
        p = Path(self._cfg.results_path)
        rows: List[Dict[str, Any]] = []
        try:
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if float(row.get("ts_unix") or 0) >= since_unix:
                        rows.append(row)
        except OSError:
            pass
        return rows

    def _last_attempt_unix(self) -> Optional[float]:
        """Timestamp of the most recent attempt in the results JSONL,
        or None if there is none / the file is unreadable. Used by the
        startup crash-loop guard; reads only the file's tail."""
        p = Path(self._cfg.results_path)
        try:
            with p.open("rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 65536))
                lines = fh.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                ts = float(json.loads(line).get("ts_unix") or 0)
            except (ValueError, TypeError):
                continue
            if ts > 0:
                return ts
        return None

    # ----- daily summary -----------------------------------------------------

    def summary_due(self) -> bool:
        now = datetime.fromtimestamp(self._time(), timezone.utc)
        if now.hour < int(self._cfg.summary_hour_utc):
            return False
        last = str(self._load_state().get("last_summary_date") or "")
        return last != now.strftime("%Y-%m-%d")

    def compute_summary_stats(
        self, rows: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        by_outcome: Dict[str, int] = {}
        by_dm: Dict[str, Dict[str, int]] = {}
        missed_dms: Dict[str, List[str]] = {}
        ratios: List[float] = []
        for r in rows:
            outcome = str(r.get("outcome") or "unknown")
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
            if outcome in Outcome.MISSES:
                dm_val = r.get("dm_pc_cm3")
                missed_dms.setdefault(outcome, []).append(
                    f"{float(dm_val):.0f}" if dm_val is not None else "?")
            dm = r.get("dm_pc_cm3")
            if dm is not None:
                key = f"{float(dm):.0f}"
                slot = by_dm.setdefault(key, {"injected": 0, "recovered": 0})
                if outcome not in (
                    Outcome.NOT_SEARCHING, Outcome.GUARD_REJECTED,
                    Outcome.FIRE_FAILED,
                ):
                    slot["injected"] += 1
                if outcome == Outcome.RECOVERED:
                    slot["recovered"] += 1
            if r.get("snr_ratio") is not None:
                ratios.append(float(r["snr_ratio"]))
        injected = sum(
            v for k, v in by_outcome.items()
            if k in (Outcome.RECOVERED,) + Outcome.MISSES
        )
        recovered = by_outcome.get(Outcome.RECOVERED, 0)
        stats: Dict[str, Any] = {
            "injected": injected,
            "recovered": recovered,
            "by_outcome": by_outcome,
            "by_dm": by_dm,
            "missed_dms": missed_dms,
        }
        if ratios:
            ratios.sort()
            stats["snr_ratio_median"] = ratios[len(ratios) // 2]
            stats["snr_ratio_min"] = ratios[0]
            stats["snr_ratio_max"] = ratios[-1]
        return stats

    def _summary_text(self, stats: Mapping[str, Any]) -> str:
        day = datetime.fromtimestamp(
            self._time(), timezone.utc).strftime("%B %-d %Y")
        lines = [
            f"test injections: 24 h summary, {day}",
            "{inj} injected, {rec} recovered".format(
                inj=stats.get("injected", 0), rec=stats.get("recovered", 0)),
        ]
        by_dm = stats.get("by_dm") or {}
        if by_dm:
            lines.append("per DM: " + " | ".join(
                f"DM {dm}: {v['recovered']}/{v['injected']}"
                for dm, v in sorted(by_dm.items(), key=lambda kv: float(kv[0]))
            ))
        by_outcome = dict(stats.get("by_outcome") or {})
        missed_dms = dict(stats.get("missed_dms") or {})
        misses = {
            k: v for k, v in by_outcome.items() if k in Outcome.MISSES and v
        }
        if misses:
            parts = []
            for k, v in sorted(misses.items()):
                dms = missed_dms.get(k) or []
                at = f" (DM {', '.join(dms)})" if dms else ""
                parts.append(f"{v} {k}{at}")
            lines.append("missed: " + "; ".join(parts))
        skipped = by_outcome.get(Outcome.NOT_SEARCHING, 0)
        failed = (
            by_outcome.get(Outcome.FIRE_FAILED, 0)
            + by_outcome.get(Outcome.GUARD_REJECTED, 0)
        )
        if skipped or failed:
            lines.append(
                f"{skipped} not attempted (pipeline not searching), "
                f"{failed} fire failures")
        if stats.get("snr_ratio_median") is not None:
            lines.append(
                "observed/target SNR: median {med:.2f} "
                "(range {lo:.2f}-{hi:.2f})".format(
                    med=stats["snr_ratio_median"],
                    lo=stats["snr_ratio_min"], hi=stats["snr_ratio_max"]))
        return "\n".join(lines)

    # ----- daily summary figures ---------------------------------------------

    #: Fixed identity encoding per DM bucket: (fill, marker, edge).
    #: Open-Color vivid hues with a darker same-hue edge (reads richer
    #: than a white outline on a white ground); marker shape doubles
    #: the encoding so identity never rides on color alone. Misses use
    #: the SAME marker but OPEN (hollow) at recovered S/N = 0 — the
    #: standard filled-vs-open detection/non-detection convention — so
    #: a lost shot still says which DM it was.
    _DM_STYLE = {
        500.0: ("#4C6EF5", "o", "#364FC7"),
        1000.0: ("#F59F00", "s", "#E67700"),
        1500.0: ("#12B886", "^", "#087F5B"),
        2000.0: ("#BE4BDB", "D", "#9C36B5"),
    }
    _NEUTRAL = "#9E9E9E"
    _INK = "#262626"

    @staticmethod
    def _pub_rc() -> Dict[str, Any]:
        """Publication-style rcParams (applied via rc_context so
        nothing leaks into other users of matplotlib in-process)."""
        return {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11.0,
            "axes.labelsize": 11.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,
            "legend.fontsize": 10.0,
            "savefig.dpi": 200,
            "savefig.facecolor": "white",
            "figure.constrained_layout.use": True,
        }

    def _despine(self, ax: Any) -> None:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self._INK)
        ax.tick_params(colors=self._INK, labelsize=10)
        ax.yaxis.label.set_color(self._INK)
        ax.xaxis.label.set_color(self._INK)

    def _window_span(self, end_unix: float) -> str:
        """Fixed 24 h summary window ending at ``end_unix`` — the
        summary always covers exactly the past 24 h, independent of
        when the individual shots landed."""
        d1 = datetime.fromtimestamp(end_unix, timezone.utc)
        d0 = datetime.fromtimestamp(end_unix - 86400.0, timezone.utc)
        return f"{d0:%Y-%m-%d %H:%M} to {d1:%Y-%m-%d %H:%M} UTC"

    def _dm_legend_handles(self, line2d: Any, with_miss: bool) -> List[Any]:
        handles = [
            line2d([], [], marker=m, color=c, ls="none", ms=8,
                   mec=e, mew=0.9, label=f"DM {dm:.0f}")
            for dm, (c, m, e) in sorted(self._DM_STYLE.items())
        ]
        if with_miss:
            handles.append(line2d(
                [], [], marker="o", mfc="none", mec=self._INK, ls="none",
                ms=8, mew=1.1, label="open = missed"))
        return handles

    def render_summary_plots(
        self, rows: Sequence[Mapping[str, Any]], out_dir: Path,
        *, window_end_unix: Optional[float] = None,
    ) -> List[Tuple[Path, str]]:
        """Render the daily summary as SEPARATE publication-style
        figures (serif, inward ticks, despined, white-edged markers,
        legends outside the data area). Figure titles carry the fixed
        24 h window ending at ``window_end_unix`` (default: now).
        Returns ``[(path, title), ...]`` for the figures that rendered;
        empty list if matplotlib is unavailable or everything failed."""
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from matplotlib import rc_context
            from matplotlib.figure import Figure
            from matplotlib.lines import Line2D
        except Exception as exc:  # noqa: BLE001
            LOG.warning("matplotlib unavailable: %s", exc)
            return []

        fired = [
            r for r in rows
            if r.get("outcome") in (Outcome.RECOVERED,) + Outcome.MISSES
        ]
        span = self._window_span(
            self._time() if window_end_unix is None else window_end_unix)
        out_dir = Path(out_dir)
        out: List[Tuple[Path, str]] = []

        def _save(fig: Any, name: str, title: str) -> None:
            path = out_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path))
            out.append((path, title))

        with rc_context(self._pub_rc()):
            try:
                out_dir.mkdir(parents=True, exist_ok=True)

                # Figure 1: recovered vs injected S/N against the 1:1
                # line. Square axes, equal limits, legend fully outside
                # (a horizontal row above the axes) so it can never
                # touch the data or the reference line.
                # Square-ish axes without box_aspect: constrained
                # layout mis-measures a box-aspect axes and clips the
                # y-label, so the figure geometry itself carries the
                # near-square shape instead.
                fig = Figure(figsize=(6.3, 5.2))
                ax = fig.add_subplot(111)
                targets = [_as_float(r.get("target_snr")) for r in fired]
                targets = [t for t in targets if t is not None]
                observed = [_as_float(r.get("observed_snr")) for r in fired]
                observed = [o for o in observed if o is not None]
                lo = min(targets) - 2 if targets else 10.0
                hi = max(targets + observed) + 2 if targets else 30.0
                ax.plot([lo, hi], [lo, hi], color=self._NEUTRAL, lw=0.9,
                        ls=(0, (5, 3)), zorder=1)
                lbl = lo + 0.93 * (hi - lo)
                ax.annotate("1:1", (lbl, lbl),
                            xytext=(5, -6), textcoords="offset points",
                            fontsize=9.5, color=self._NEUTRAL,
                            ha="left", va="top")
                for r in fired:
                    dm = float(r.get("dm_pc_cm3") or 0)
                    color, marker, edge = self._DM_STYLE.get(
                        dm, (self._NEUTRAL, "o", self._INK))
                    t = _as_float(r.get("target_snr"))
                    o = _as_float(r.get("observed_snr"))
                    if t is None:
                        continue
                    if o is not None:
                        ax.scatter([t], [o], s=70, marker=marker,
                                   facecolor=color, edgecolor=edge,
                                   linewidth=0.9, alpha=0.9, zorder=3)
                    else:
                        # Miss: same DM marker, OPEN, at recovered
                        # S/N = 0 (filled = recovered, open = missed).
                        ax.scatter([t], [0.0], s=70, marker=marker,
                                   facecolor="none", edgecolor=edge,
                                   linewidth=1.4, zorder=3)
                ax.set_xlim(lo, hi)
                ax.set_ylim(-1.0, hi)
                ax.set_xlabel("injected S/N")
                ax.set_ylabel("recovered S/N")
                ax.set_title(f"Injection recovery, {span}",
                             fontsize=11.5, color=self._INK,
                             loc="left", pad=10)
                fig.legend(
                    handles=self._dm_legend_handles(Line2D, True),
                    loc="outside center right", ncol=1,
                    handletextpad=0.3, labelspacing=0.6, fontsize=9.5)
                self._despine(ax)
                _save(fig, "snr_recovery.png",
                      f"recovered vs injected S/N, {span}")

                # Figure 2: outcome counts (horizontal, direct-labeled;
                # the numbers are the axis).
                fig = Figure(figsize=(6.2, 3.2))
                ax = fig.add_subplot(111)
                outcome_colors = {
                    Outcome.RECOVERED: "#2E7D32",
                    Outcome.MISSED_SEARCH_OR_C1: "#C62828",
                    Outcome.MISSED_C2: "#E65100",
                    Outcome.MISSED_C3: "#6A1B9A",
                    Outcome.NOT_SEARCHING: self._NEUTRAL,
                    Outcome.GUARD_REJECTED: self._NEUTRAL,
                    Outcome.FIRE_FAILED: self._NEUTRAL,
                }
                counts = {o: 0 for o in Outcome.ALL}
                for r in rows:
                    o = str(r.get("outcome") or "")
                    if o in counts:
                        counts[o] += 1
                labels = [o for o in Outcome.ALL if counts[o] > 0] or [
                    Outcome.RECOVERED]
                labels = labels[::-1]  # recovered on top
                vals = [counts[o] for o in labels]
                bars = ax.barh(
                    range(len(labels)), vals,
                    color=[outcome_colors[o] for o in labels],
                    height=0.6, zorder=3)
                for rect, v in zip(bars, vals):
                    ax.annotate(
                        f" {v}",
                        (v, rect.get_y() + rect.get_height() / 2),
                        ha="left", va="center", fontsize=11,
                        color=self._INK)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(
                    [o.replace("_", " ") for o in labels])
                ax.set_xlim(0, max(vals) * 1.15 if vals else 1)
                ax.xaxis.set_visible(False)
                ax.minorticks_off()
                ax.tick_params(axis="y", length=0)
                for side in ("top", "right", "bottom"):
                    ax.spines[side].set_visible(False)
                ax.spines["left"].set_color(self._INK)
                ax.tick_params(colors=self._INK, labelsize=10.5)
                ax.set_title(f"Outcomes, {span}", fontsize=11.5,
                             color=self._INK, loc="left", pad=10)
                _save(fig, "outcomes.png", f"outcomes, {span}")

                # Figure 3: recovery accuracy — position offset between
                # injection and detection vs DM error.
                fig = Figure(figsize=(6.6, 4.9))
                ax = fig.add_subplot(111)
                for r in fired:
                    off = _as_float(r.get("offset_arcsec"))
                    ddm = _as_float(r.get("delta_dm_pc_cm3"))
                    if off is None or ddm is None:
                        continue
                    dm = float(r.get("dm_pc_cm3") or 0)
                    color, marker, edge = self._DM_STYLE.get(
                        dm, (self._NEUTRAL, "o", self._INK))
                    ax.scatter([ddm], [off], s=70, marker=marker,
                               facecolor=color, edgecolor=edge,
                               linewidth=0.9, alpha=0.9, zorder=3)
                ax.axvline(0.0, color=self._NEUTRAL, lw=0.8, alpha=0.7,
                           zorder=1)
                ax.set_xlabel(
                    "DM error (detection $-$ injection) [pc cm$^{-3}$]")
                ax.set_ylabel(
                    "position error (injection $-$ detection) [arcsec]")
                ax.set_ylim(bottom=0)
                ax.set_title(f"Recovery accuracy, {span}",
                             fontsize=11.5, color=self._INK,
                             loc="left", pad=10)
                fig.legend(
                    handles=self._dm_legend_handles(Line2D, False),
                    loc="outside center right", ncol=1,
                    handletextpad=0.3, labelspacing=0.6, fontsize=9.5)
                self._despine(ax)
                _save(fig, "accuracy.png", f"recovery accuracy, {span}")
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "summary figure render failed: %s", exc, exc_info=True)
        return out

    def post_daily_summary(self) -> Dict[str, Any]:
        """Compute stats, render the separate summary figures, post the
        stats text and thread the figures under it; stamps state so it
        runs once per UTC day. Never raises."""
        now = self._time()
        rows = self.load_results_since(now - 86400.0)
        stats = self.compute_summary_stats(rows)
        text = self._summary_text(stats)
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        out_dir = Path(self._cfg.summary_dir).expanduser() / day
        figures = self.render_summary_plots(
            rows, out_dir, window_end_unix=now)
        if not figures:
            text += "\n(summary figures failed to render - see journal)"
        resp = self._notifier.post_text(text)
        thread_ts = resp.get("ts") if resp.get("ok") else None
        n_uploaded = 0
        for path, title in figures:
            fresp = self._notifier.post_file(
                path, title=title, thread_ts=thread_ts)
            if fresp.get("ok"):
                n_uploaded += 1
            else:
                LOG.warning("summary figure upload failed: %s", fresp)
        if resp.get("ok"):
            self._load_state()["last_summary_date"] = day
            self._save_state()
        else:
            LOG.warning("daily summary post failed: %s", resp)
        return {
            "ok": bool(resp.get("ok")), "stats": stats,
            "figures": [str(p) for p, _ in figures],
            "n_uploaded": n_uploaded,
        }

    # ----- loop --------------------------------------------------------------

    _stop_event: Optional[threading.Event] = None

    def _sleep(self, seconds: float) -> None:
        ev = self._stop_event
        if ev is not None:
            ev.wait(seconds)
        else:
            time.sleep(seconds)

    def run_forever(self, stop_event: Optional[threading.Event] = None) -> None:
        cfg = self._cfg
        stop_event = stop_event or threading.Event()
        self._stop_event = stop_event
        LOG.info(
            "inject_bot: starting (interval=%.0fs jitter=%.0fs "
            "dms=%s snr=[%.1f, %.1f] channel=%s)",
            cfg.interval_s, cfg.jitter_s, list(cfg.dm_choices),
            cfg.target_snr_min, cfg.target_snr_max, cfg.channel or "(none)",
        )
        # Wall-clock anchored cadence: the next shot is scheduled one
        # interval after the PREVIOUS shot's scheduled time, not after
        # the cycle finished — a cycle spends up to ~13 min in the
        # recovery watch (plus K recals), and sleep-after-finish would
        # yield ~20-23 attempts/day instead of the expected 24. Jitter
        # perturbs each tick but does not accumulate. If a cycle ever
        # overruns its whole slot, the anchor is re-based (no backlog
        # of instant catch-up shots).
        # First shot fires immediately on startup (an end-to-end check
        # right after a (re)start is the point of this service) —
        # UNLESS the previous attempt was less than half an interval
        # ago, in which case wait out the remainder of its slot. This
        # keeps a crash-looping unit (Restart=on-failure, 10 s) from
        # firing an injection per restart.
        next_fire = self._time()
        last = self._last_attempt_unix()
        if last is not None:
            since = self._time() - last
            if 0.0 <= since < cfg.interval_s / 2.0:
                next_fire = last + cfg.interval_s
                LOG.info(
                    "startup: previous attempt %.0fs ago; first shot "
                    "delayed to its slot in %.0fs",
                    since, next_fire - self._time(),
                )
                stop_event.wait(max(0.0, next_fire - self._time()))
        while not stop_event.is_set():
            record = self.run_cycle()
            LOG.info(
                "cycle done: outcome=%s inj_id=%s",
                record.get("outcome"), record.get("inj_id"),
            )
            if self.summary_due():
                summary = self.post_daily_summary()
                LOG.info("daily summary posted: ok=%s", summary.get("ok"))
            next_fire += cfg.interval_s
            now = self._time()
            if next_fire < now + 60.0:
                next_fire = now + 60.0
            delay = max(
                60.0,
                next_fire - now + self._rng.uniform(
                    -cfg.jitter_s, cfg.jitter_s),
            )
            stop_event.wait(delay)
        LOG.info("inject_bot: stopped")


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hourly injection bot with Slack reporting")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--summary-now", action="store_true",
        help="render + post the 24 h summary immediately and exit")
    parser.add_argument(
        "--once", action="store_true",
        help="run a single injection cycle and exit (smoke test)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = InjectBotConfig.from_yaml(args.config)
    if not cfg.enabled and not (args.summary_now or args.once):
        LOG.error(
            "inject_bot: disabled in config (inject_bot.enabled); "
            "refusing to start the loop")
        return 2
    bot = InjectBot(cfg)

    if args.summary_now:
        result = bot.post_daily_summary()
        print(json.dumps(result.get("stats", {}), indent=1, sort_keys=True))
        return 0 if result.get("ok") else 1
    if args.once:
        record = bot.run_cycle()
        print(json.dumps(record, indent=1, sort_keys=True, default=str))
        return 0

    stop_event = threading.Event()

    def _handle_signal(signum: int, frame: Any) -> None:
        LOG.info("inject_bot: received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    bot.run_forever(stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
