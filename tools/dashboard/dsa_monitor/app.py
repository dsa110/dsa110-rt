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
from control_store import (
    CORR_CN_IDS,
    DEFAULT_ARM_SEQ_MARGIN,
    DEFAULT_INJECT_CHGROUPS,
    DEFAULT_INJECT_MARGIN_BLOCKS,
    SEARCH_CN_IDS,
    ControlStore,
    audit_log,
    compute_arm_seq,
    compute_inject_apply_at,
    control_inject_pulse,
    control_start_fleet,
    control_stop_fleet,
    control_utc_start_now,
    control_utc_stop_now,
    list_recent_audit,
)


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


# M7.4 Phase 8: write-side etcd surface for the Control tab. Built
# lazily so the dashboard still boots if etcd is briefly down. Kept
# separate from ``_LazyEtcd`` so the read-only path is provably
# write-free.
control_store = ControlStore()


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


# ---------- Control tab (M7.4 Phase 8) -------------------------------------
#
# The Control tab is the dashboard's first write surface. Every POST
# below goes through ``control_store.audit_log`` so the action is
# replayable from any etcd reader. The default network ACL on h23 is
# the only "auth" today — see docs/M7.4_PHASE8_CONTROL.md for the
# operational risk posture (network-bound + audited).


@app.route("/control", methods=["GET"])
def control_page():
    """Render the Control tab: state-summary panel + verb form."""
    # Pull a quick fleet snapshot so the operator can see what they're
    # about to act on. We don't block on etcd; failures render as
    # "unknown" in the template.
    try:
        arm_info = compute_arm_seq(control_store)
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("compute_arm_seq: %s", exc)
        arm_info = {
            "arm_seq": None, "polled": [], "answered": [], "missing": [],
            "max_last_seq_no": None, "max_source": None,
            "margin": DEFAULT_ARM_SEQ_MARGIN, "_error": str(exc),
        }
    try:
        recent = list_recent_audit(control_store, limit=20)
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("list_recent_audit: %s", exc)
        recent = []
    return render_template(
        "control.html",
        active_tab="control",
        corr_cn_ids=list(CORR_CN_IDS),
        search_cn_ids=list(SEARCH_CN_IDS),
        arm_info=arm_info,
        recent_audit=recent,
        default_arm_margin=DEFAULT_ARM_SEQ_MARGIN,
        default_inject_margin_blocks=DEFAULT_INJECT_MARGIN_BLOCKS,
    )


def _control_json_or_error(handler, **kwargs):
    """Wrap a control_store helper, converting exceptions into a
    JSON error payload + audit row so the UI doesn't blow up on a
    transient etcd hiccup.
    """
    user = request.form.get("user") or request.remote_addr or "anon"
    try:
        result = handler(control_store, user=user, **kwargs)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("control verb %s failed", handler.__name__)
        try:
            audit_log(
                control_store,
                namespace="control",
                cn_target="-",
                cmd=handler.__name__,
                val=kwargs,
                ok=False,
                note=f"exception: {exc!r}",
                user=user,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log also failed (continuing)")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(result)


@app.route("/control/start", methods=["POST"])
def control_start_post():
    dec_raw = request.form.get("obs_dec_deg", "").strip()
    obs_dec: Optional[float]
    if dec_raw == "":
        obs_dec = None
    else:
        try:
            obs_dec = float(dec_raw)
        except ValueError:
            return jsonify({
                "ok": False,
                "error": f"obs_dec_deg={dec_raw!r}: not a float",
            }), 400
    return _control_json_or_error(
        control_start_fleet, obs_dec_deg=obs_dec,
    )


@app.route("/control/stop", methods=["POST"])
def control_stop_post():
    confirm = request.form.get("confirm", "").strip().lower()
    if confirm != "stop":
        return jsonify({
            "ok": False,
            "error": (
                "stop requires confirm=stop in the POST body — "
                "this is a deliberate safety speed bump."
            ),
        }), 400
    fanout = request.form.get("fanout_corr_too", "true").lower() != "false"
    return _control_json_or_error(
        control_stop_fleet, fanout_corr_too=fanout,
    )


@app.route("/control/utc_start", methods=["POST"])
def control_utc_start_post():
    margin_raw = request.form.get("margin", "").strip()
    try:
        margin = int(margin_raw) if margin_raw else DEFAULT_ARM_SEQ_MARGIN
    except ValueError:
        return jsonify({
            "ok": False,
            "error": f"margin={margin_raw!r}: not an int",
        }), 400
    return _control_json_or_error(
        control_utc_start_now, margin=margin,
    )


@app.route("/control/utc_stop", methods=["POST"])
def control_utc_stop_post():
    return _control_json_or_error(control_utc_stop_now)


@app.route("/control/inject", methods=["POST"])
def control_inject_post():
    """M7.4 Phase 6: push a runtime injection to one or more chgroups.

    Form fields (all required except ``apply_at_specnum``,
    ``margin_blocks``, and ``chgroups``):

      inj_id          str (e.g. "phase6c_extragal_t1")
      l_rad           float, |l| < 1
      m_rad           float, |m| < 1 and l^2 + m^2 < 1
      dm_pc_cm3       float
      fluence_jy_ms   float
      width_samples   int, 1..MAX_WIDTH_SAMPLES
      profile         "gaussian" | "boxcar"
      apply_at_specnum   int (omit → auto-arm via
                            ``compute_inject_apply_at`` reading
                            ``/mon/corr_rt/<cn>/corr_fast``)
      margin_blocks   int blocks ahead of fleet ``block_n`` for
                      auto-arm (default
                      ``DEFAULT_INJECT_MARGIN_BLOCKS = 16``)
      chgroups        comma-separated ints, e.g. "0,1,2"
                      (omit → all 16)
    """
    f = request.form
    try:
        kwargs: dict[str, object] = {
            "inj_id": (f.get("inj_id") or "").strip() or None,
            "l_rad": float(f.get("l_rad", "0")),
            "m_rad": float(f.get("m_rad", "0")),
            "dm_pc_cm3": float(f.get("dm_pc_cm3", "0")),
            "fluence_jy_ms": float(f.get("fluence_jy_ms", "0")),
            "width_samples": int(f.get("width_samples", "1")),
            "profile": (f.get("profile") or "gaussian").strip(),
            "margin_blocks": int(
                f.get("margin_blocks") or DEFAULT_INJECT_MARGIN_BLOCKS
            ),
        }
    except ValueError as exc:
        return jsonify({"ok": False, "error": f"bad numeric field: {exc}"}), 400
    if not kwargs["inj_id"]:
        return jsonify({"ok": False, "error": "inj_id is required"}), 400

    raw_specnum = (f.get("apply_at_specnum") or "").strip()
    if raw_specnum:
        try:
            kwargs["apply_at_specnum"] = int(raw_specnum)
        except ValueError:
            return jsonify({
                "ok": False,
                "error": f"apply_at_specnum={raw_specnum!r}: not an int",
            }), 400
    else:
        kwargs["apply_at_specnum"] = None

    raw_chg = (f.get("chgroups") or "").strip()
    if raw_chg:
        try:
            cg = tuple(int(s) for s in raw_chg.split(",") if s.strip())
        except ValueError:
            return jsonify({
                "ok": False,
                "error": f"chgroups={raw_chg!r}: must be comma-separated ints",
            }), 400
        kwargs["chgroups"] = cg

    # Validate the InjectionConfig payload BEFORE doing any etcd
    # writes so a malformed payload (l^2 + m^2 >= 1, unknown
    # profile, etc.) surfaces as a clean 400 — _control_json_or_error
    # would otherwise wrap the ValueError into an opaque 500.
    from control_store import _validate_inject_payload          # local
    try:
        _validate_inject_payload({
            "inj_id": kwargs["inj_id"],
            "l_rad": kwargs["l_rad"],
            "m_rad": kwargs["m_rad"],
            "dm_pc_cm3": kwargs["dm_pc_cm3"],
            "fluence_jy_ms": kwargs["fluence_jy_ms"],
            "width_samples": kwargs["width_samples"],
            "profile": kwargs["profile"],
            "apply_at_specnum": kwargs["apply_at_specnum"] or 0,
        })
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return _control_json_or_error(control_inject_pulse, **kwargs)


@app.route("/control/recent_audit", methods=["GET"])
def control_recent_audit():
    """JSON GET for the audit-log panel (used by manual refresh)."""
    try:
        limit = int(request.args.get("limit", "20"))
    except ValueError:
        limit = 20
    rows = list_recent_audit(control_store, limit=limit)
    return jsonify({"rows": rows})


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
