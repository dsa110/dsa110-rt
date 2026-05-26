#!/usr/bin/env python3
"""DSA-110 monitoring dashboard (h23).

Flask web app exposing three tabs:

  * **Antennas/RFI** — per-antenna bandpass + flag spectrum + 30-min
    waterfalls + monitor-points table.
  * **SEFDs** — existing SEFD dashboard running on the same host at
    port 5777; surfaced here as an external link / iframe.
  * **Burst candidates** — placeholder (architecture TBD).

Architecture (M7.6, see ``REPO/configs/dsart_pipeline_rt.yaml`` for
the producer side):

  16 corr nodes  ──HTTP──>  this app's RFIPoller thread  ──>  in-mem
       ^                       per-cn ring buffer (30 min)        ↓
       │                                                   page refresh
       └── /api/health, /api/latest, /api/recent          (matplotlib)

No shared filesystem. Plots are matplotlib PNGs rendered per-request
(no caching needed — the in-mem ring is the cache).

Lifecycle: Run as a systemd user service (see ``dsa_monitor.service``).
The poller thread starts at import-time; Flask serves HTTP requests.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_file

# We deploy as a directory; make sibling modules importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# dsart.common.constants (stdlib-only deps) imported via PYTHONPATH=src.
from corr_topology import CORR_NODES, CORR_NODES_BY_CHGROUP
from rfi_store import RFIPoller, RFIWindowStore
from rfi_client import RFIClient
from plot_render import (
    render_bandpass_spectrum,
    render_bandpass_waterfall,
    render_flag_spectrum,
    render_flag_waterfall,
    render_thumb_grid,
)
from ant_table import (
    build_ant_table,
    NANTS_DASH,
    all_ant_nums_in_cube_order,
    ant_idx_to_ant_num,
)
from cands_panel_funcs import ArchiveBrowser, DEFAULT_ARCHIVE_ROOT


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

LOG = logging.getLogger("dsa_monitor.app")

SEFD_DASHBOARD_URL = os.environ.get(
    "SEFD_DASHBOARD_URL", "http://lxd110h23:5777/",
)

store: RFIWindowStore
poller: Optional[RFIPoller] = None
_store_lock = threading.Lock()


# DsaStore wrapper for /mon/ant/<n>; instantiated lazily so import is
# fast and the dashboard still starts even if etcd is briefly down.
class _LazyEtcd:
    def __init__(self) -> None:
        self._store = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._store is not None:
                return
            from dsautils.dsa_store import DsaStore
            self._store = DsaStore()

    def get_dict(self, key: str):
        self._ensure()
        return self._store.get_dict(key)


etcd_store = _LazyEtcd()


# Burst-candidates archive browser (h23 reads /dataz/dsa110/candidates/).
# Module-level singleton — cheap to construct, all I/O is per-request.
cands_browser = ArchiveBrowser()


def _init_store_and_poller() -> None:
    """Module-level init. Spinning up here means the poller starts
    immediately on Flask import (so the first page render isn't empty
    after a fresh boot)."""
    global store, poller
    store = RFIWindowStore()
    poller = RFIPoller(store)
    poller.start()


_init_store_and_poller()


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.jinja_env.auto_reload = False


def _clamp_ant_idx(s: Optional[str]) -> int:
    if s is None or s == "":
        return 0
    try:
        idx = int(s)
    except ValueError:
        return 0
    return max(0, min(idx, NANTS_DASH - 1))


# ---------- Pages ----------------------------------------------------------


@app.route("/")
def index():
    # Default tab = Antennas/RFI.
    return antennas_rfi()


@app.route("/antennas", methods=["GET"])
def antennas_rfi():
    ant_idx = _clamp_ant_idx(request.args.get("ant"))
    snap = store.snapshot()
    table = build_ant_table(snap, etcd_store, ant_idx=ant_idx)
    # Per-cn health summary for the freshness panel
    nodes_status = []
    for cring in snap.per_chgroup:
        nodes_status.append({
            "cn_id": cring.cn.cn_id,
            "host": cring.cn.host,
            "chgroup": cring.cn.chgroup,
            "n_windows": len(cring.records),
            "last_publish_unix": cring.last_publish_unix,
            "last_fetch_unix": cring.last_fetch_unix,
            "last_fetch_ok": cring.last_fetch_ok,
            "last_seq": cring.last_seq,
        })
    # (cube_idx, ant_num) pairs, in cube-index order. Templates use the
    # cube_idx as the form value and the ant_num as the human label.
    ant_choices = list(zip(range(NANTS_DASH), all_ant_nums_in_cube_order()))
    return render_template(
        "antennas.html",
        active_tab="antennas",
        ant_idx=ant_idx,
        ant_num=ant_idx_to_ant_num(ant_idx),
        n_ants=NANTS_DASH,
        ant_choices=ant_choices,
        table=table,
        nodes_status=nodes_status,
        snapshot_unix=snap.snapshot_unix,
        sefd_url=SEFD_DASHBOARD_URL,
    )


@app.route("/sefds")
def sefds():
    return render_template(
        "sefds.html",
        active_tab="sefds",
        sefd_url=SEFD_DASHBOARD_URL,
    )


@app.route("/bursts")
def bursts():
    events = cands_browser.list_events()
    return render_template(
        "bursts.html",
        active_tab="bursts",
        sefd_url=SEFD_DASHBOARD_URL,
        events=events,
        archive_root=str(cands_browser.root),
        archive_available=cands_browser.is_available,
    )


@app.route("/bursts/<name>")
def burst_event(name: str):
    import json as _json
    detail = cands_browser.event_detail(name)
    if detail is None:
        abort(404)
    return render_template(
        "burst_event.html",
        active_tab="bursts",
        event=detail,
        metadata_pretty=_json.dumps(
            detail.metadata, indent=2, sort_keys=True, default=str,
        ),
        sefd_url=SEFD_DASHBOARD_URL,
    )


@app.route("/bursts/<name>/plot/<plot_name>")
def burst_event_plot(name: str, plot_name: str):
    p = cands_browser.plot_path(name, plot_name)
    if p is None:
        abort(404)
    return send_file(str(p), mimetype="image/png")


# ---------- Plot endpoints (PNG) ------------------------------------------


def _png_response(png_bytes: bytes):
    import io
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="plot.png",
    )


@app.route("/plot/bandpass.png")
def plot_bandpass():
    ant_idx = _clamp_ant_idx(request.args.get("ant"))
    snap = store.snapshot()
    png = render_bandpass_spectrum(
        snap, ant_idx=ant_idx,
        ant_label=str(ant_idx_to_ant_num(ant_idx)),
    )
    return _png_response(png)


@app.route("/plot/bandpass_wf.png")
def plot_bandpass_wf():
    ant_idx = _clamp_ant_idx(request.args.get("ant"))
    snap = store.snapshot()
    png = render_bandpass_waterfall(
        snap, ant_idx=ant_idx,
        ant_label=str(ant_idx_to_ant_num(ant_idx)),
    )
    return _png_response(png)


@app.route("/plot/flag_spectrum.png")
def plot_flag_spectrum():
    ant_idx = _clamp_ant_idx(request.args.get("ant"))
    snap = store.snapshot()
    png = render_flag_spectrum(
        snap, ant_idx=ant_idx,
        ant_label=str(ant_idx_to_ant_num(ant_idx)),
    )
    return _png_response(png)


@app.route("/plot/flag_wf.png")
def plot_flag_wf():
    ant_idx = _clamp_ant_idx(request.args.get("ant"))
    snap = store.snapshot()
    png = render_flag_waterfall(
        snap, ant_idx=ant_idx,
        ant_label=str(ant_idx_to_ant_num(ant_idx)),
    )
    return _png_response(png)


@app.route("/plot/thumb_grid.png")
def plot_thumb_grid():
    """12-col × 8-row fleet bandpass thumbnails (one PNG, all 96 antennas)."""
    snap = store.snapshot()
    png = render_thumb_grid(snap, ant_nums=all_ant_nums_in_cube_order())
    return _png_response(png)


# ---------- API ------------------------------------------------------------


@app.route("/api/status")
def api_status():
    snap = store.snapshot()
    return jsonify({
        "snapshot_unix": snap.snapshot_unix,
        "per_chgroup": [
            {
                "cn_id": s.cn.cn_id,
                "host": s.cn.host,
                "chgroup": s.cn.chgroup,
                "n_windows": len(s.records),
                "last_seq": s.last_seq,
                "last_publish_unix": s.last_publish_unix,
                "last_fetch_unix": s.last_fetch_unix,
                "last_fetch_ok": s.last_fetch_ok,
            }
            for s in snap.per_chgroup
        ],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    port = int(os.environ.get("DSA_MONITOR_PORT", "5778"))
    bind = os.environ.get("DSA_MONITOR_BIND", "0.0.0.0")
    LOG.info(
        "dsa_monitor up on %s:%d (sefd_url=%s)",
        bind, port, SEFD_DASHBOARD_URL,
    )
    app.run(host=bind, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
