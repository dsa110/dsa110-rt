#!/usr/bin/env python3
"""SEFD scanner entry-point — runs the calibrator MS scanner loop
with no Flask web server attached.

This is what ``sefd_dashboard.service`` invokes now that the SEFD UI
has been merged into the ``dsa_monitor`` dashboard (port 5778) on
h23.  The scanner stays here, in the casa38 conda env, because
``light_diagnostics`` + ``full_pipeline`` import casatools; the
dashboard reads our outputs (``state.json`` + ``results/``) over the
shared h23 filesystem via
``tools/dashboard/dsa_monitor/sefd_view.py``.

Run directly:

    /home/ubuntu/anaconda3/envs/casa38/bin/python -u sefd_scanner.py

Or under systemd:

    [Service]
    ExecStart=/home/ubuntu/anaconda3/envs/casa38/bin/python -u sefd_scanner.py
"""

from __future__ import annotations

import logging
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scanner import RESULTS_DIR, SCAN_INTERVAL_SECONDS, scanner_loop


def _setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )


def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT should exit cleanly so systemd's stop verb
    doesn't have to escalate to SIGKILL.

    The scanner's inner loop only blocks on :func:`time.sleep` or on
    casa subprocesses; the default handler is enough — we just need
    to make sure we exit with code 0 on SIGTERM so the unit doesn't
    flap into ``failed`` state.
    """
    def _bye(signum, _frame):
        logging.info("sefd_scanner: signal %s, exiting", signum)
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _bye)


def main() -> int:
    _setup_logging()
    _install_signal_handlers()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    interval = int(os.environ.get(
        "SEFD_SCAN_INTERVAL_S", str(SCAN_INTERVAL_SECONDS),
    ))
    logging.getLogger("sefd_scanner").info(
        "sefd_scanner starting (interval=%ds, results_dir=%s)",
        interval, RESULTS_DIR,
    )
    try:
        scanner_loop(interval_seconds=interval)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
