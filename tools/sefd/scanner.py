#!/usr/bin/env python3
"""SEFD calibrator scanner — the casa38 background worker.

Watches ``/dataz/dsa110/operations/calibration/`` for new
``<YYYY-MM-DD>_<src>.ms`` measurement sets, runs
:func:`light_diagnostics.run_light_diagnostics` first (cheap, no
calibration), then :func:`full_pipeline.run_full_pipeline` (calibrate
+ image + SEFD), and writes outcomes into ``state.json`` plus per-day
PNG plots under ``results/<src>/<date>/``.

This module used to be embedded in ``app.py`` next to a Flask web
server on port 5777.  The web side has been retired: ``dsa_monitor``
renders the same ``state.json`` + ``results/`` tree natively (see
``tools/dashboard/dsa_monitor/sefd_view.py``), so the only thing
``sefd_dashboard.service`` still has to do is the casa38 scanning
work.  ``sefd_scanner.py`` is the entry-point script; ``app.py`` is
kept on disk for one release as a back-compat shim and re-imports the
helpers here.

State schema (per-entry under ``state.json[<date>_<src>]``):

    {
      "status":       "pending" | "light_processing" | "light_done" |
                      "full_processing" | "complete" |
                      "light_error" | "full_error" | "error",
      "updated":      ISO-8601 UTC timestamp of last status change,
      "date":         "YYYY-MM-DD",
      "source":       calibrator name,
      "path":         absolute path to the MS,
      "metrics":      {median_amplitude, median_noise, median_coherence,
                       median_autocorr_{xx,yy}, median_hi_peak_db},
      "full_metrics": {median_sefd, mean_sefd, std_sefd, n_baselines,
                       sefd_<bin>m, …},
      "error":        present iff status ends in "_error",
    }
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

# Add parent directory to path so the existing light_diagnostics +
# full_pipeline modules in the sefd repo resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_diagnostics import run_light_diagnostics
from full_pipeline import run_full_pipeline

# VLA calibrator manual (same file the calibration23 service selects
# from). Used to build a DEC-agnostic flux catalog so the scanner
# processes whatever calibrator the service calibrates on at the
# current pointing dec, with the flux taken from the manual.
try:
    from dsacalib.preprocess import read_vla_catalog as _read_vla_catalog
except Exception:                                          # noqa: BLE001
    _read_vla_catalog = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CALIBRATION_DIR = "/dataz/dsa110/operations/calibration"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
STATE_FILE = os.path.join(HERE, "state.json")
# Empty file we touch on every scan cycle (whether we processed
# anything or not).  ``dsa_monitor`` reads this file's mtime to tell
# operators "scanner is alive" vs "no MSes today" --- if we only
# touched state.json on transitions, the dashboard would falsely
# flag a quiet night as a dead scanner.
HEARTBEAT_FILE = os.path.join(HERE, "scanner_heartbeat")

LOG = logging.getLogger("sefd_scanner")

# Minimum 20cm flux for a calibrator to be tracked (operator request
# 2026-07-23: SEFD page shows only >2 Jy sources). Flux for the SEFD
# calc is taken from the VLA manual, not hard-coded.
MIN_FLUX_JY = 2.0

# Static fallback if the VLA manual can't be read (keeps the scanner
# alive on the previous dec's calibrators).
_STATIC_SOURCES = {
    "0318+164": {"flux_jy": 7.81},
    "0521+166": {"flux_jy": 8.47},
    "2253+161": {"flux_jy": 10.0},
}


def build_source_catalog(min_flux_jy: float = MIN_FLUX_JY) -> dict:
    """VLA-manual calibrators with 20cm flux > ``min_flux_jy``.

    Returns ``{source: {"flux_jy": <manual 20cm flux>}}``. DEC-agnostic:
    the calibration23 service only writes an MS for the calibrators it
    calibrates on at the current pointing dec, so intersecting the
    present MS files with this catalog (see ``find_calibrator_ms_files``)
    yields exactly the >2 Jy sources calibrated on for the current dec.
    """
    if _read_vla_catalog is None:
        LOG.warning("dsacalib unavailable; using static source catalog")
        return dict(_STATIC_SOURCES)
    try:
        df = _read_vla_catalog()
        cat = {
            str(name): {"flux_jy": round(float(row["flux_20_cm"]) / 1e3, 3)}
            for name, row in df.iterrows()
            if float(row["flux_20_cm"]) / 1e3 > min_flux_jy
        }
        LOG.info("VLA catalog: %d sources > %.1f Jy", len(cat), min_flux_jy)
        return cat or dict(_STATIC_SOURCES)
    except Exception:                                      # noqa: BLE001
        LOG.exception("VLA catalog load failed; using static catalog")
        return dict(_STATIC_SOURCES)


#: Flux catalog, built once at import (manual is static; a restart
#: picks up manual edits). Keyed by VLA source name.
SOURCES = build_source_catalog()

DEFAULT_LOOKBACK_DAYS = 7
SCAN_INTERVAL_SECONDS = 60

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

state_lock = threading.Lock()


def load_state() -> dict:
    """Read ``state.json``; return ``{}`` if missing."""
    with state_lock:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as fh:
                return json.load(fh)
        return {}


def save_state(state: dict) -> None:
    """Persist the in-memory state dict back to disk."""
    with state_lock:
        with open(STATE_FILE, "w") as fh:
            json.dump(state, fh, indent=2)


def update_entry_state(
    key: str, status: str, extra: Optional[dict] = None,
) -> None:
    """Atomic per-entry status update.

    Loads the full state, mutates one entry, and writes it back.
    Always touches ``updated`` so the dashboard's freshness pill sees
    a heartbeat each transition.
    """
    state = load_state()
    if key not in state:
        state[key] = {}
    state[key]["status"] = status
    state[key]["updated"] = datetime.utcnow().isoformat()
    if extra:
        state[key].update(extra)
    save_state(state)


# ---------------------------------------------------------------------------
# Scanner-side currently-processing tracking
# ---------------------------------------------------------------------------

processing_lock = threading.Lock()
_currently_processing: Optional[str] = None


def get_currently_processing() -> Optional[str]:
    """Best-effort accessor for telemetry/back-compat callers.

    The single source of truth is the in-progress entry's
    ``*_processing`` status inside ``state.json``; this in-memory
    variable is only useful inside the scanner process itself.
    """
    with processing_lock:
        return _currently_processing


def _set_currently_processing(value: Optional[str]) -> None:
    global _currently_processing
    with processing_lock:
        _currently_processing = value


# ---------------------------------------------------------------------------
# MS discovery
# ---------------------------------------------------------------------------


def is_ms_complete(ms_path: str) -> bool:
    """Heuristic "MS finished writing" gate.

    A casa MS is a directory; the writer touches subtables piecewise.
    We require ``table.dat`` at the top level AND a ``table.dat`` in
    each of ``ANTENNA/`` and ``SPECTRAL_WINDOW/`` before we hand the
    MS to casa (otherwise the calibration step blows up partway
    through with a half-readable table).
    """
    if not os.path.isdir(ms_path):
        return False
    if not os.path.isfile(os.path.join(ms_path, "table.dat")):
        return False
    for subtable in ("ANTENNA", "SPECTRAL_WINDOW"):
        if not os.path.isfile(
            os.path.join(ms_path, subtable, "table.dat")
        ):
            return False
    return True


def find_calibrator_ms_files(lookback_days: int = DEFAULT_LOOKBACK_DAYS):
    """All complete calibrator MSes whose date ≤ lookback_days old."""
    ms_files = []
    today = datetime.utcnow().date()
    # Discover every calibrator MS the service wrote, then keep those
    # whose source is a >2 Jy VLA calibrator (SOURCES). This tracks
    # whatever the service calibrates on at the current dec without a
    # hard-coded per-dec source list. MS names are ``<YYYY-MM-DD>_<src>.ms``
    # and VLA source names contain no underscore, so split once on "_".
    for ms_path in glob.glob(os.path.join(CALIBRATION_DIR, "*_*.ms")):
        basename = os.path.basename(ms_path)
        stem = basename[:-3] if basename.endswith(".ms") else basename
        try:
            date_str, source = stem.split("_", 1)
            obs_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except (ValueError, IndexError):
            continue
        if source not in SOURCES:
            continue
        if (today - obs_date).days <= lookback_days:
            if not is_ms_complete(ms_path):
                LOG.debug("Skipping incomplete MS: %s", ms_path)
                continue
            ms_files.append({
                "path": ms_path,
                "date": date_str,
                "source": source,
                "key": f"{date_str}_{source}",
            })
    return ms_files


# ---------------------------------------------------------------------------
# Main scanner loop
# ---------------------------------------------------------------------------


def _touch_heartbeat() -> None:
    """Bump :data:`HEARTBEAT_FILE`'s mtime to wall-clock now.

    Best-effort: if disk is full or perms are wrong we log and
    continue; the scanner loop itself must never die on a heartbeat
    failure.
    """
    try:
        with open(HEARTBEAT_FILE, "a"):
            os.utime(HEARTBEAT_FILE, None)
    except OSError as exc:
        LOG.warning("heartbeat touch failed: %s", exc)


def scanner_loop(interval_seconds: int = SCAN_INTERVAL_SECONDS) -> None:
    """Forever-loop: scan for new MSes, process them sequentially.

    Sequential rather than parallel because (a) the casa modules
    aren't thread-safe across runs, and (b) the scanner CPU/memory
    budget is shared with the dsart_h23 Flask dashboard on the same
    host.

    At the head and tail of every poll cycle we touch
    :data:`HEARTBEAT_FILE` so ``dsa_monitor``'s freshness pill keeps
    saying ``alive`` even on quiet nights when ``state.json`` itself
    doesn't change.
    """
    LOG.info("Scanner started (interval=%ds)", interval_seconds)
    _touch_heartbeat()  # set immediately so first dashboard hit is fresh

    while True:
        _touch_heartbeat()
        try:
            ms_files = find_calibrator_ms_files()
            state = load_state()

            retryable_statuses = {
                "error", "light_error", "full_error",
                "light_processing", "full_processing",
            }
            new_files = []
            for ms in ms_files:
                entry_status = state.get(ms["key"], {}).get("status")
                if entry_status is None or entry_status in retryable_statuses:
                    new_files.append(ms)

            for ms in sorted(new_files, key=lambda x: x["date"]):
                _set_currently_processing(ms["key"])
                existing = state.get(ms["key"], {})
                has_light = existing.get("status") in (
                    "light_done", "full_processing",
                )

                if not has_light:
                    update_entry_state(ms["key"], "light_processing", {
                        "date": ms["date"], "source": ms["source"],
                        "path": ms["path"],
                    })
                    try:
                        LOG.info("Light diagnostics: %s", ms["key"])
                        metrics = run_light_diagnostics(
                            ms["path"], ms["date"], ms["source"],
                            RESULTS_DIR,
                            every_nth_baseline=10, every_nth_antenna=10,
                        )
                        update_entry_state(
                            ms["key"], "light_done", {"metrics": metrics},
                        )
                    except Exception as exc:                       # noqa: BLE001
                        LOG.exception(
                            "Light diagnostics failed for %s", ms["key"],
                        )
                        update_entry_state(
                            ms["key"], "light_error", {"error": str(exc)},
                        )
                        _set_currently_processing(None)
                        continue

                update_entry_state(ms["key"], "full_processing")
                try:
                    LOG.info("Full pipeline: %s", ms["key"])
                    flux_jy = SOURCES[ms["source"]]["flux_jy"]
                    full_metrics = run_full_pipeline(
                        ms["path"], ms["date"], ms["source"],
                        RESULTS_DIR, cal_flux_jy=flux_jy,
                    )
                    update_entry_state(
                        ms["key"], "complete",
                        {"full_metrics": full_metrics},
                    )
                except Exception as exc:                          # noqa: BLE001
                    LOG.exception(
                        "Full pipeline failed for %s", ms["key"],
                    )
                    update_entry_state(
                        ms["key"], "full_error", {"error": str(exc)},
                    )

                _set_currently_processing(None)

        except Exception:                                        # noqa: BLE001
            LOG.exception("Scanner error (continuing)")
            _set_currently_processing(None)

        _touch_heartbeat()
        time.sleep(interval_seconds)
