#!/usr/bin/env python3
"""SEFD dashboard — DEPRECATED entry point.

The Flask UI that this script used to serve on port 5777 has been
merged into the ``dsa_monitor`` dashboard on h23 port 5778; see
``dsa110-rt/tools/dashboard/dsa_monitor/sefd_view.py`` for the new
SEFD rendering and ``dsa110-rt/tools/dashboard/dsa_monitor/app.py``
for the routes (``/sefds``, ``/sefds/source/<src>``,
``/sefds/day/<date>``, ``/sefds/results/<path>``,
``/api/sefd_status``).

The casa38-side scanner that produces ``state.json`` + the
``results/`` PNG tree now lives in :mod:`scanner` and is launched by
``sefd_scanner.py``; ``sefd_dashboard.service`` has been retargeted
at that entry point.

This file is kept for two reasons:

* back-compat: anyone importing ``app.load_state`` /
  ``app.find_calibrator_ms_files`` keeps working — those names are
  re-exported from :mod:`scanner` below.
* fallback: running ``python app.py`` directly still serves the old
  Flask UI on port 5777 in case the operator wants the standalone
  view back temporarily.  It prints a deprecation notice and warns
  that the merged UI in ``dsa_monitor`` is the canonical surface.
"""

from __future__ import annotations

import logging
import os
import threading

# Re-export scanner helpers for back-compat (old call sites that did
# ``from app import load_state`` keep working).
from scanner import (                                             # noqa: F401
    CALIBRATION_DIR,
    DEFAULT_LOOKBACK_DAYS,
    RESULTS_DIR,
    SCAN_INTERVAL_SECONDS,
    SOURCES,
    STATE_FILE,
    find_calibrator_ms_files,
    get_currently_processing,
    is_ms_complete,
    load_state,
    save_state,
    scanner_loop,
    update_entry_state,
)


# ---------------------------------------------------------------------------
# Legacy Flask UI (only built when this module is run as __main__).
# ---------------------------------------------------------------------------


def _build_flask_app():
    """Construct the legacy SEFD Flask app.

    Imported lazily so the rest of this module's surface (the
    re-exports above) does not pull Flask into casa38 processes that
    only need the scanner.
    """
    from datetime import datetime

    from flask import Flask, jsonify, render_template, request

    app = Flask(__name__)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    @app.route("/")
    def summary():
        lookback = int(request.args.get("days", DEFAULT_LOOKBACK_DAYS))
        state = load_state()
        ms_files = find_calibrator_ms_files(lookback_days=lookback)
        dates = sorted(set(ms["date"] for ms in ms_files), reverse=True)
        sources = sorted(SOURCES.keys())
        grid = {}
        for date in dates:
            grid[date] = {}
            for source in sources:
                key = f"{date}_{source}"
                if key in state:
                    grid[date][source] = state[key]
                else:
                    ms_exists = any(
                        ms["key"] == key for ms in ms_files
                    )
                    grid[date][source] = (
                        {"status": "pending"} if ms_exists else None
                    )
        return render_template(
            "summary.html",
            dates=dates, sources=sources, source_flux=SOURCES,
            grid=grid,
            currently_processing=get_currently_processing(),
            lookback=lookback,
        )

    @app.route("/source/<source_name>")
    def source_page(source_name):
        lookback = int(request.args.get("days", DEFAULT_LOOKBACK_DAYS))
        state = load_state()
        if source_name not in SOURCES:
            return "Source not found", 404
        entries = []
        for _, entry in sorted(state.items()):
            if entry.get("source") == source_name and entry.get("status") in (
                "light_done", "full_processing", "complete",
                "light_error", "full_error", "error",
            ):
                date = entry.get("date", "")
                today = datetime.utcnow().date()
                try:
                    obs = datetime.strptime(date, "%Y-%m-%d").date()
                    if (today - obs).days <= lookback:
                        entries.append(entry)
                except ValueError:
                    continue
        plot_dir = os.path.join(RESULTS_DIR, source_name)
        plots = {}
        if os.path.isdir(plot_dir):
            for fname in sorted(os.listdir(plot_dir)):
                if fname.endswith(".png"):
                    plots[fname] = f"/results/{source_name}/{fname}"
        return render_template(
            "source.html",
            source_name=source_name,
            source_flux=SOURCES[source_name],
            entries=entries, plots=plots, lookback=lookback,
        )

    @app.route("/day/<date>")
    def day_page(date):
        state = load_state()
        entries = {}
        for source in SOURCES:
            key = f"{date}_{source}"
            if key in state:
                entry = state[key]
                result_dir = os.path.join(RESULTS_DIR, source, date)
                entry_plots = {}
                if os.path.isdir(result_dir):
                    for fname in sorted(os.listdir(result_dir)):
                        if fname.endswith(".png"):
                            entry_plots[fname] = (
                                f"/results/{source}/{date}/{fname}"
                            )
                entry["plots"] = entry_plots
                entries[source] = entry
        return render_template(
            "day.html", date=date, sources=SOURCES, entries=entries,
        )

    @app.route("/results/<path:filepath>")
    def serve_result(filepath):
        from flask import send_from_directory
        return send_from_directory(RESULTS_DIR, filepath)

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "currently_processing": get_currently_processing(),
            "state": load_state(),
        })

    return app


# ---------------------------------------------------------------------------
# Main: deprecation notice + legacy Flask UI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("sefd_dashboard.app")
    log.warning(
        "sefd_dashboard/app.py is DEPRECATED. The SEFD UI now renders "
        "natively in dsa_monitor (h23:5778, /sefds tab); the scanner "
        "is split out into sefd_scanner.py and that is what "
        "sefd_dashboard.service should call. This script will serve "
        "the legacy port-5777 Flask UI for back-compat only.",
    )

    # Background scanner thread so the legacy UI keeps producing
    # updates if anyone is still running this directly.
    scanner_thread = threading.Thread(target=scanner_loop, daemon=True)
    scanner_thread.start()
    log.info("Legacy scanner thread started (mirrored from sefd_scanner)")

    flask_app = _build_flask_app()
    flask_app.run(host="0.0.0.0", port=5777, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
