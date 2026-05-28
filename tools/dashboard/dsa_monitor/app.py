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
from typing import Any, Optional

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
    fleet_restart_all,
    fleet_service_status,
    list_recent_audit,
)
from services_inventory import H20_HOSTNAMES, SERVICE_INVENTORY


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


@app.route("/control/dump_now", methods=["POST"])
def control_dump_now_post():
    """M7.4 Phase 6c: broadcast a one-shot synthetic C2 trigger to
    every search-half so the operator can capture the cubes currently
    in flight on every gpu_half (offline RFI-source debugging).

    Form fields:

      confirm    Must literally equal the string ``dump_now`` — a
                 deliberate speed bump on the fan-out verb. Anything
                 else returns 400.

    Returns:

      200 with the :class:`cube_dump_now.FleetDumpNowResult` dict
      (``event_name``, ``event_specnum``, per-half status, pass /
      fail counts) on every path where we actually broadcast.

      503 with ``{ok: false, error: "no corr_fast mon-keys; is the
      fleet up?"}`` when no ``/mon/corr_rt/<cn>/corr_fast`` publisher
      is answering — same shape ``compute_inject_apply_at`` uses.

      400 on missing / wrong confirm field.

      500 (via :func:`_control_json_or_error` fallback) on
      unexpected exceptions.
    """
    import cube_dump_now                                          # local

    confirm = request.form.get("confirm", "").strip()
    if confirm != "dump_now":
        return jsonify({
            "ok": False,
            "error": (
                "dump_now requires confirm=dump_now in the POST body — "
                "this is a deliberate safety speed bump."
            ),
        }), 400

    user = request.form.get("user") or request.remote_addr or "anon"
    try:
        result = cube_dump_now.fleet_dump_now(control_store, user=user)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("control_dump_now failed")
        try:
            audit_log(
                control_store, namespace="dsa_monitor.cube_dump_now",
                cn_target="search-halves",
                cmd="dump_now", val=None, ok=False,
                note=f"exception: {exc!r}", user=user,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log also failed (continuing)")
        return jsonify({"ok": False, "error": str(exc)}), 500

    # No corr_fast mon-keys / cold-boot → 503-equivalent payload.
    if result.error is not None:
        try:
            audit_log(
                control_store, namespace="dsa_monitor.cube_dump_now",
                cn_target="search-halves",
                cmd="dump_now", val=result.to_dict(), ok=False,
                note=f"refused: {result.error}", user=user,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log failed (continuing)")
        return jsonify(result.to_dict()), 503

    try:
        audit_log(
            control_store, namespace="dsa_monitor.cube_dump_now",
            cn_target="search-halves",
            cmd="dump_now", val=result.to_dict(),
            ok=bool(result.ok),
            note=(
                f"event={result.event_name} "
                f"specnum={result.event_specnum} "
                f"pass={result.pass_count} fail={result.fail_count}"
            ),
            user=user,
        )
    except Exception:                                              # noqa: BLE001
        LOG.exception("audit_log failed (continuing)")
    return jsonify(result.to_dict())


@app.route("/control/dumps_enabled", methods=["GET", "POST"])
def control_dumps_enabled():
    """M7.4 Phase 6c: read or flip the C2 dump-broadcast kill-switch.

    The C2 coincidencer polls ``/cmd/c2/dumps_enabled`` via a small
    in-process cache (``dsart.services.coincidencer.DumpsGate``) and
    skips the UDP fan-out into the C1 listeners when ``enabled=False``
    while still writing the per-event archive rows + logging
    ``WOULD-DUMP …`` lines. Missing key ⇒ enabled=True (fail-OPEN).

    GET
        Returns the current state ``{"enabled", "ts", "actor",
        "reason", "default"}`` (default=True iff the key is missing).

    POST form fields:

      ``enabled``  ``true`` / ``false`` (required).
      ``reason``   free text, 1..240 chars (required, audited).
      ``confirm``  must be the literal word ``suppress`` when
                   ``enabled=false``, ``enable`` when ``enabled=true``
                   — deliberate speed-bump on the flip.

    Returns:

      200 + new state on a successful flip (or on GET).
      400 on missing / wrong confirm, missing / empty / over-long
        reason, or unparseable ``enabled`` value.
      500 (with audit row) on unexpected etcd transport errors.
    """
    import dumps_gate                                            # local

    if request.method == "GET":
        try:
            state = dumps_gate.get_dumps_state(control_store)
        except Exception as exc:                                  # noqa: BLE001
            LOG.exception("get_dumps_state failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, **state})

    # ---- POST ----
    f = request.form
    enabled_raw = (f.get("enabled") or "").strip().lower()
    if enabled_raw in ("true", "1", "yes", "on"):
        new_enabled = True
    elif enabled_raw in ("false", "0", "no", "off"):
        new_enabled = False
    else:
        return jsonify({
            "ok": False,
            "error": (
                f"enabled={enabled_raw!r}: must be true / false "
                f"(true / 1 / yes / on or false / 0 / no / off)"
            ),
        }), 400

    reason_raw = (f.get("reason") or "").strip()
    if not reason_raw:
        return jsonify({
            "ok": False,
            "error": (
                "reason is required (non-empty string, max "
                f"{dumps_gate.MAX_REASON_LEN} chars)"
            ),
        }), 400
    if len(reason_raw) > dumps_gate.MAX_REASON_LEN:
        return jsonify({
            "ok": False,
            "error": (
                f"reason too long: {len(reason_raw)} chars (max "
                f"{dumps_gate.MAX_REASON_LEN})"
            ),
        }), 400

    confirm_raw = (f.get("confirm") or "").strip().lower()
    expected_confirm = "enable" if new_enabled else "suppress"
    if confirm_raw != expected_confirm:
        return jsonify({
            "ok": False,
            "error": (
                f"confirm={confirm_raw!r} does not match the requested "
                f"direction — type the literal word "
                f"{expected_confirm!r} to arm the flip"
            ),
        }), 400

    actor = (
        f.get("user") or request.remote_addr or "anon"
    )
    try:
        new_state = dumps_gate.set_dumps_state(
            control_store,
            enabled=new_enabled,
            reason=reason_raw,
            actor=actor,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("set_dumps_state failed")
        try:
            audit_log(
                control_store,
                namespace="c2.dumps_toggle",
                cn_target="h23",
                cmd="dumps_toggle",
                val={"enabled": new_enabled},
                ok=False,
                note=f"exception: {exc!r}",
                user=actor,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log fallback also failed (continuing)")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **new_state})


@app.route("/control/dumps_audit", methods=["GET"])
def control_dumps_audit():
    """Most recent dumps_toggle audit rows (JSON), newest first.

    Powers the Control tab's per-toggle history panel. Limit defaults
    to 5 (the panel size) and is capped at 100 to keep the response
    small.
    """
    import dumps_gate                                            # local
    try:
        limit = max(1, min(int(request.args.get("limit", "5")), 100))
    except ValueError:
        limit = 5
    try:
        rows = dumps_gate.list_recent_toggles(control_store, limit=limit)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("list_recent_toggles failed")
        return jsonify({"ok": False, "error": str(exc), "rows": []}), 500
    return jsonify({"ok": True, "rows": rows})


@app.route("/control/recent_audit", methods=["GET"])
def control_recent_audit():
    """JSON GET for the audit-log panel (used by manual refresh)."""
    try:
        limit = int(request.args.get("limit", "20"))
    except ValueError:
        limit = 20
    rows = list_recent_audit(control_store, limit=limit)
    return jsonify({"rows": rows})


# ---------- Control tab (M7.4 Phase 8 v2) ----------------------------------
#
# Two new operator actions:
#
#   * GET  /control/services_status — fleet-wide systemd / process
#     status table. Cheap (5 s timeout × 8-way fanout), called every
#     10 s by the dashboard JS for auto-refresh.
#   * POST /control/restart_all    — "cold recovery from stop" button.
#     Heavy (etcd + ssh fanout + lxc + local systemctl + deferred
#     self-restart). Runs in a background thread; the response is
#     202 + a job_id, the client polls
#     GET /control/restart_all/<job_id> for progress.
#
# The lxd110h20 host is excluded from EVERY restart fanout — see
# services_inventory.H20_HOSTNAMES + the unit-test pins in
# tests/test_fleet_services.py.


@app.route("/control/services_status", methods=["GET"])
def control_services_status():
    """Fleet services status JSON (one row per ServiceEntry)."""
    try:
        timeout_raw = request.args.get("timeout_s", "").strip()
        timeout_s = float(timeout_raw) if timeout_raw else None
    except ValueError:
        timeout_s = None
    user = request.remote_addr or "anon"
    try:
        result = fleet_service_status(
            control_store, user=user, timeout_s=timeout_s,
        )
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("fleet_service_status failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    result.setdefault("ok", True)
    result["h20_hostnames_excluded"] = sorted(H20_HOSTNAMES)
    result["inventory_size"] = len(SERVICE_INVENTORY)
    return jsonify(result)


# ---- Restart-all job registry (in-memory) ---------------------------------
#
# A POST /control/restart_all spins up a worker thread + returns
# immediately. The job's progress is published into this dict,
# keyed by a short job_id, and the client polls GET
# /control/restart_all/<job_id> until ``done`` flips True.
#
# In-memory is fine: the registry only needs to survive long enough
# for the operator to read the last status, and the dsa_monitor
# process restart in step 7 wipes it. (We log the result to etcd
# audit too, so a hard reload doesn't lose history.)

_restart_jobs: dict[str, dict[str, Any]] = {}
_restart_jobs_lock = threading.Lock()


def _new_job_id() -> str:
    import secrets
    return secrets.token_hex(6)


def _restart_worker(job_id: str, *, dry_run: bool, user: str) -> None:
    """Background thread body: run fleet_restart_all + stash the
    result + log it. Never raises (the worker is detached)."""
    try:
        result = fleet_restart_all(
            control_store, user=user, dry_run=dry_run,
        )
        with _restart_jobs_lock:
            _restart_jobs[job_id].update({
                "ok": bool(result.get("ok")),
                "done": True,
                "result": result,
                "finished_unix": int(__import__("time").time()),
            })
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("restart_worker job=%s failed", job_id)
        with _restart_jobs_lock:
            _restart_jobs[job_id].update({
                "ok": False,
                "done": True,
                "error": f"{type(exc).__name__}: {exc}",
                "finished_unix": int(__import__("time").time()),
            })


@app.route("/control/restart_all", methods=["POST"])
def control_restart_all_post():
    """Kick off a fleet restart in a background thread.

    Body fields:

    ``confirm``      Must literally be the string ``restart_all`` —
                     a deliberate speed-bump on the destructive verb.
    ``dry_run``      Optional; ``"true"`` / ``"1"`` short-circuits the
                     actual ssh / systemctl / Popen calls and returns
                     a JSON summary of what *would* have been done.

    Returns 202 + ``{job_id, poll_url}`` so the client can poll for
    completion. The dashboard process will be killed by step 7 (the
    deferred ``systemctl --user restart dsa_monitor.service``); the
    client should expect the next poll after ~2 s to fail and the
    page to need a manual reload.
    """
    confirm = request.form.get("confirm", "").strip()
    if confirm != "restart_all":
        return jsonify({
            "ok": False,
            "error": (
                "restart_all requires confirm=restart_all in the "
                "POST body — this is a deliberate safety speed bump."
            ),
        }), 400
    dry_raw = request.form.get("dry_run", "").strip().lower()
    dry_run = dry_raw in ("1", "true", "yes")
    user = request.form.get("user") or request.remote_addr or "anon"

    job_id = _new_job_id()
    with _restart_jobs_lock:
        _restart_jobs[job_id] = {
            "job_id": job_id,
            "started_unix": int(__import__("time").time()),
            "user": user,
            "dry_run": dry_run,
            "done": False,
            "ok": None,
        }
    th = threading.Thread(
        target=_restart_worker,
        kwargs={"job_id": job_id, "dry_run": dry_run, "user": user},
        name=f"restart_all_{job_id}",
        daemon=True,
    )
    th.start()
    return jsonify({
        "ok": True,
        "accepted": True,
        "job_id": job_id,
        "dry_run": dry_run,
        "poll_url": f"/control/restart_all/{job_id}",
        "note": (
            "Background fanout started. Poll the poll_url for the "
            "step-by-step result. The deferred dsa_monitor.service "
            "self-restart will kill this process ~2 s after step 6 "
            "completes; expect the next poll to fail and the page "
            "to need a manual reload."
        ),
    }), 202


@app.route("/control/restart_all/<job_id>", methods=["GET"])
def control_restart_all_poll(job_id: str):
    """Poll a previously-started restart_all job."""
    with _restart_jobs_lock:
        job = _restart_jobs.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "unknown job_id"}), 404
        # Copy so we can return outside the lock.
        snapshot = dict(job)
    return jsonify(snapshot)


# ---------- M7.4 Phase 6c+: pull-and-install dsa110-rt on every node -------
#
# Added in a separate file (``fleet_update.py``) so the ssh fan-out
# logic stays decoupled from ``control_store.py``. The route below
# is the only entry point into that module from Flask.


@app.route("/control/update_dsart", methods=["POST"])
def control_update_dsart_post():
    """Pull (or hard-reset) ``/home/ubuntu/proj/dsa110-rt`` on every
    corr + search node and report per-host pre/post SHAs.

    Form fields:

      dry_run     "true"/"false" (default "true"). When true,
                  runs only step 1 (pre-SHA + branch + porcelain) +
                  ``git fetch``; never calls ``git pull`` or
                  ``git reset``.
      force       "true"/"false" (default "false"). When true,
                  overrides dirty-worktree gate AND uses
                  ``git reset --hard origin/<branch>`` instead of
                  ``git pull --ff-only``.
      confirm     Must equal ``update_dsart`` when ``dry_run=false``.
      branch      Optional branch to fetch/pull. Blank → use whichever
                  branch each host is currently on (typical case).
      hosts       Optional comma-separated host list. Blank → all 20
                  corr + search hosts.
    """
    import fleet_update                                              # local

    f = request.form
    dry_run = (f.get("dry_run", "true").lower() != "false")
    force = (f.get("force", "false").lower() == "true")
    branch_raw = (f.get("branch") or "").strip()
    branch = branch_raw or None

    if not dry_run:
        confirm = (f.get("confirm") or "").strip()
        if confirm != "update_dsart":
            return jsonify({
                "ok": False,
                "error": (
                    "apply update_dsart requires confirm=update_dsart in "
                    "the POST body — this is a deliberate safety speed bump."
                ),
            }), 400

    raw_hosts = (f.get("hosts") or "").strip()
    hosts: Optional[list[str]]
    if raw_hosts:
        hosts = [h.strip() for h in raw_hosts.split(",") if h.strip()]
    else:
        hosts = None

    # Per-host timeout × 4 ssh calls / max_workers across 20 hosts is
    # well under the spec's 60 s synchronous-response ceiling at the
    # default max_workers=8.
    return _control_json_or_error(
        fleet_update.update_fleet,
        dry_run=dry_run,
        force=force,
        hosts=hosts,
        branch=branch,
    )


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
