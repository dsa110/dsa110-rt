#!/usr/bin/env python3
"""DSA-110 monitoring dashboard (h23).

Flask web app exposing four tabs:

  * **Antennas/RFI** — per-antenna bandpass + flag spectrum + 30-min
    waterfalls + monitor-points table.
  * **SEFDs** — native rendering of the SEFD scanner's outputs
    (``state.json`` + ``results/`` PNG tree); the scanner itself
    still runs in ``sefd_dashboard.service`` (casa38 conda env) but
    no longer serves a Flask app on port 5777 (see
    :mod:`sefd_view`).
  * **Burst candidates** — h23 candidate / cube archive browser.
  * **Control** — Phase 8 fleet verbs, injections, dumps gate, SNR
    calibration, fleet-recovery ops.

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
from cands_panel_funcs import (
    ArchiveBrowser,
    DEFAULT_ARCHIVE_ROOT,
    N_VOLTAGE_FRAGMENTS_TOTAL,
    _SAFE_EVENT_RE,
    partition_events_c3,
)
import annotations as ann
import event_astrometry as ea
from control_store import (
    CORR_CN_IDS,
    DEFAULT_ARM_SEQ_MARGIN,
    DEFAULT_BOUNCE_SLEEP_S,
    DEFAULT_INJECT_CHGROUPS,
    DEFAULT_INJECT_MARGIN_BLOCKS,
    SEARCH_CN_IDS,
    ControlStore,
    audit_log,
    bounce_search,
    c2_journal_tail_local,
    c2_mon_snapshot,
    compute_arm_seq,
    compute_inject_apply_at,
    compute_system_state,
    control_inject_pulse,
    control_start_fleet,
    control_stop_fleet,
    control_utc_start_now,
    control_utc_stop_now,
    fleet_restart_all,
    fleet_service_status,
    list_recent_audit,
    restart_c2_service_local,
    restart_h23_services_local,
)
from control_store import H23_DSART_UNITS
from services_inventory import H20_HOSTNAMES, SERVICE_INVENTORY
from sefd_view import (
    DEFAULT_RESULTS_DIR as SEFD_DEFAULT_RESULTS_DIR,
    DEFAULT_STATE_FILE as SEFD_DEFAULT_STATE_FILE,
    SefdView,
)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

LOG = logging.getLogger("dsa_monitor.app")

# Where the SEFD scanner (``sefd_dashboard.service``) puts its outputs.
# We read these two paths read-only — the scanner is the sole writer.
SEFD_STATE_FILE = os.environ.get(
    "SEFD_STATE_FILE", SEFD_DEFAULT_STATE_FILE,
)
SEFD_RESULTS_DIR = os.environ.get(
    "SEFD_RESULTS_DIR", SEFD_DEFAULT_RESULTS_DIR,
)
sefd_view = SefdView(
    state_file=SEFD_STATE_FILE, results_dir=SEFD_RESULTS_DIR,
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


# Sky monitor (E2E test 1): ingest + frame builder. Cheap to construct;
# all heavy work (combine + iFFT + PNG) happens per ~30 s frame inside
# ingest requests.
from sky_monitor import SkyMonitor


def _active_sidereal_vetos() -> list:
    """Active C2 sidereal (l,m) dump-veto regions for the sky overlay.

    Reads the registry C2 publishes to ``/mon/c2/sidereal_vetos``.
    Best-effort: any etcd problem yields no overlay (never raises into
    the frame builder).
    """
    try:
        doc = etcd_store.get_dict("/mon/c2/sidereal_vetos")
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(doc, dict):
        return []
    regions = doc.get("regions")
    out = []
    if isinstance(regions, list):
        tol = doc.get("tol_rad", 0.0)
        for r in regions:
            if not isinstance(r, dict):
                continue
            r = dict(r)
            r.setdefault("tol_rad", tol)
            out.append(r)
    return out


def _capture_armed_mjd():
    """Fleet capture-arm MJD from etcd (/mon/snap/1/armed_mjd) — the
    absolute time base that turns a sky snapshot's ``block_n`` into
    its exact capture time (2026-07-18 astrometry fix). Best-effort:
    any etcd problem returns None and the sky monitor falls back to
    corr wall clocks minus the measured export lag.
    """
    try:
        doc = etcd_store.get_dict("/mon/snap/1/armed_mjd")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(doc, dict):
        v = doc.get("armed_mjd")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


sky_mon = SkyMonitor(
    veto_provider=_active_sidereal_vetos,
    armed_mjd_provider=_capture_armed_mjd,
)


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
# Long-cache the static handler (annotations.js / typeahead.js): safe
# because templates reference them via static_v() cache-busting URLs
# (?v=<mtime>), so an edit changes the URL and busts the cache. In
# Flask >= 2 this config is consulted by the /static handler; plain
# send_file(...) calls with max_age unset stay uncached (verified in
# tests + live headers: the in-memory RFI live plots at /plot/*.png
# must never be browser-cached).
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 2592000          # 30 days


@app.template_global()
def static_v(filename: str) -> str:
    """Cache-busting /static URL: appends ?v=<file mtime>.

    Computed per-render (cheap stat); a deploy that touches the file
    yields a new URL, so the 30-day static cache never serves stale JS.
    """
    path = os.path.join(HERE, "static", filename)
    try:
        v = int(os.stat(path).st_mtime)
    except OSError:
        v = 0
    return f"/static/{filename}?v={v}"


def _html_revalidate(html: str):
    """ETag + no-cache wrapper for rendered HTML pages.

    Content is always revalidated (no-cache), but an unchanged reload
    costs ~0 bytes: md5 the body, and answer 304 with an empty body
    when the client's If-None-Match already has it.
    """
    import hashlib
    tag = hashlib.md5(html.encode("utf-8")).hexdigest()
    if request.if_none_match.contains(tag):
        resp = app.make_response(("", 304))
    else:
        resp = app.make_response(html)
    resp.set_etag(tag)
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def _no_store(resp):
    """Mark a (JSON) response explicitly uncacheable."""
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
    )


@app.route("/sefds")
def sefds():
    """SEFD landing page: per-day x per-source grid with the headline
    metrics inline.  Reads the scanner's ``state.json`` directly via
    the :class:`SefdView` singleton; no iframe."""
    import bfweights_update                                       # local
    import cal_visibility                                         # local

    try:
        lookback = int(request.args.get("days", "7"))
    except ValueError:
        lookback = 7
    summary = sefd_view.summary(lookback_days=lookback)
    try:
        pipeline_weights = cal_visibility.build_pipeline_weights_view(etcd_store)
    except Exception:  # noqa: BLE001 — panel is best-effort, page must
                        # still render if etcd is having a bad day.
        LOG.exception("sefds: pipeline-weights view failed")
        pipeline_weights = None
    return render_template(
        "sefds.html",
        active_tab="sefds",
        summary=summary,
        sefd_state_path=SEFD_STATE_FILE,
        sefd_state_error=sefd_view.state_error(),
        bf_latest=bfweights_update.latest_descriptors(summary.sources),
        pipeline_weights=pipeline_weights,
    )


@app.route("/sefds/source/<source_name>")
def sefds_source(source_name: str):
    """Per-source comparison: metrics table + diagnostic plots for
    every recent observation of one calibrator."""
    try:
        lookback = int(request.args.get("days", "7"))
    except ValueError:
        lookback = 7
    if source_name not in sefd_view.sources:
        abort(404)
    entries = sefd_view.source_entries(source_name, lookback_days=lookback)
    # Pre-resolve plot URLs once so the template only does dict
    # lookups; this also makes the absent-PNG case visible to the
    # template's ``onerror`` handler.
    entry_plots = {
        e.date: sefd_view.list_day_plots(source_name, e.date)
        for e in entries
    }
    return render_template(
        "sefd_source.html",
        active_tab="sefds",
        source_name=source_name,
        source_flux=sefd_view.sources[source_name],
        entries=entries,
        entry_plots=entry_plots,
        lookback=lookback,
    )


@app.route("/sefds/day/<date>")
def sefds_day(date: str):
    """Per-day detail: every source's status + metrics + plots for one
    observation date."""
    entries = sefd_view.day_entries(date)
    if not entries:
        # Render with an empty banner rather than 404 so the operator
        # can land on the page from a stale link without a hard error.
        pass
    plots_by_source = {
        src: sefd_view.list_day_plots(src, date) for src in entries
    }
    return render_template(
        "sefd_day.html",
        active_tab="sefds",
        date=date,
        sources=sefd_view.sources,
        entries=entries,
        plots_by_source=plots_by_source,
    )


@app.route("/sefds/results/<path:rel_path>")
def sefds_result(rel_path: str):
    """Serve one PNG out of the scanner's ``results/`` tree.

    Path-traversal is enforced by
    :meth:`SefdView.resolve_plot_path` — anything that resolves
    outside ``SEFD_RESULTS_DIR`` returns 404.
    """
    abs_path = sefd_view.resolve_plot_path(rel_path)
    if abs_path is None:
        abort(404)
    return send_file(abs_path, mimetype="image/png")


@app.route("/api/sefd_status")
def api_sefd_status():
    """JSON status for the SEFD scanner.  Used by the SEFD landing
    page's freshness pill (so the operator immediately sees a stale
    scanner without having to read mtimes off disk)."""
    summary = sefd_view.summary(lookback_days=1)
    return jsonify({
        "state_path": SEFD_STATE_FILE,
        "state_mtime_unix": summary.state_mtime_unix,
        "state_error": sefd_view.state_error(),
        "scanner_alive": summary.scanner_alive,
        "scanner_age_s": summary.scanner_age_s,
        "currently_processing": summary.currently_processing,
    })


# ---------- SEFD page: beamformer-weights (calibration) update -------------
#
# One button per calibrator on /sefds. The heavy lifting (descriptor
# discovery + the calibration23 container run + audit) lives in
# ``bfweights_update.py``; these routes are the only Flask entry points.


@app.route("/control/update_bfweights", methods=["POST"])
def control_update_bfweights_post():
    """Kick off ``update_bfweights.py <SRC>_<ISOT>`` in the
    calibration23 container, in a background thread.

    Form fields:

      ``source``   Calibrator name; must be one of the SEFD-tracked
                   sources. The newest generated solution for that
                   source is resolved server-side (the client also
                   sees it via the page context, purely for display).
      ``confirm``  Must literally equal ``update_bfweights``.
      ``dry_run``  Optional "true"/"1" — exercises the container
                   plumbing (env import + echo) without touching cals.

    Returns 202 + ``{job_id, poll_url}``; poll for completion.
    """
    import bfweights_update                                       # local

    confirm = (request.form.get("confirm") or "").strip()
    if confirm != "update_bfweights":
        return jsonify({
            "ok": False,
            "error": (
                "update_bfweights requires confirm=update_bfweights "
                "in the POST body — deliberate safety speed bump."
            ),
        }), 400
    source = (request.form.get("source") or "").strip()
    if source not in sefd_view.sources:
        return jsonify({
            "ok": False,
            "error": (
                f"source {source!r} is not an SEFD-tracked calibrator "
                f"(expected one of {sorted(sefd_view.sources)})"
            ),
        }), 400
    dry_raw = (request.form.get("dry_run") or "").strip().lower()
    dry_run = dry_raw in ("1", "true", "yes")
    user = request.form.get("user") or request.remote_addr or "anon"

    latest = bfweights_update.latest_descriptor(source)
    if latest is None:
        return jsonify({
            "ok": False,
            "error": (
                f"no beamformer_weights_{source}_*.yaml found under "
                f"{bfweights_update.GENERATED_DIR}"
            ),
        }), 412
    try:
        started = bfweights_update.start_update(
            latest["descriptor"], dry_run=dry_run, user=user,
            store=control_store,
        )
    except RuntimeError as exc:          # single-flight: already running
        return jsonify({"ok": False, "error": str(exc)}), 409
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({
        "ok": True,
        "accepted": True,
        **started,
        "age_hours": latest.get("age_hours"),
        "poll_url": f"/control/update_bfweights/{started['job_id']}",
    }), 202


@app.route("/control/update_bfweights/<job_id>", methods=["GET"])
def control_update_bfweights_poll(job_id: str):
    """Poll a previously-started update_bfweights job."""
    import bfweights_update                                       # local

    snap = bfweights_update.job_snapshot(job_id)
    if snap is None:
        return jsonify({"ok": False, "error": "unknown job_id"}), 404
    # snap's own "ok" is the JOB's success/failure (None while running,
    # True/False once snap["done"]) -- it must NOT clobber the
    # transport-level "ok" the client uses to detect a poll failure
    # (see sefds.html's poll()). Keep the job's ok visible under
    # "job_ok" and force the transport-level key to True since we did
    # find and return the job.
    payload = dict(snap)
    payload["job_ok"] = payload.get("ok")
    payload["ok"] = True
    return jsonify(payload)


def _refined_display(row: dict) -> dict:
    """Template-ready view of a current refined-position row: the stored
    fields + adaptive-precision sexagesimal strings + the table tooltip."""
    ra_hms, dec_dms = ea.sexagesimal_refined(
        row.get("ra_deg"), row.get("dec_deg"),
        row.get("ra_err_arcsec"), row.get("dec_err_arcsec"),
    )
    out = dict(row)
    out["ra_hms"] = ra_hms
    out["dec_dms"] = dec_dms
    out["tooltip"] = (
        f"refined: {row.get('method')} by {row.get('username')} · "
        f"{row.get('ra_deg'):.5f} {row.get('dec_deg'):+.5f} deg"
    )
    return out


def _refined_positions_bulk() -> dict:
    """{event: template-ready refined position} — ONE DB query.
    Best-effort: a DB hiccup must never 500 the list page."""
    try:
        bulk = ann.get_positions_bulk()
    except Exception:                                       # noqa: BLE001
        LOG.exception("get_positions_bulk failed (refined positions "
                      "skipped)")
        return {}
    return {name: _refined_display(row) for name, row in bulk.items()}


@app.route("/bursts")
def bursts():
    # Annotation-aware listing: the newest-N-by-mtime window UNION every
    # human-annotated event (so classified events / named sources stay
    # visible + searchable forever, never ageing out of the cap). The
    # counts drive the truncation notice below.
    listing = cands_browser.list_events_detailed()
    events = listing.events
    # Split by C3 cube-veto outcome. Fresh events C3 has not judged yet
    # (< 1 h old) ride the PASS tab with a "pending" badge — the
    # operator wants new triggers front and centre, and fail-open is
    # C3's own default. Events pending for > 1 h are zombies (the
    # cube/metadata transfer to h23 failed; C3 will never judge them)
    # and get their own tab so they stop cluttering PASS.
    events_pass, events_fail, events_zombie = partition_events_c3(events)
    # Human-annotation badges: one bulk read (no per-event query),
    # joined to the event rows in the template by event name.
    try:
        annot_current = ann.all_current()
        annot_source_names = ann.vocab().get("source_names", [])
    except Exception:                                       # noqa: BLE001
        LOG.exception("annotations.all_current failed (list badges skipped)")
        annot_current = {}
        annot_source_names = []
    tag_filter = (request.args.get("tag") or "").strip()
    # Current-source search (?source=): exact, case-insensitive match
    # against each event's CURRENT source name (from the same bulk
    # annot_current read — no per-event query). ANDs with ?tag= in the
    # row-filter logic in the template.
    source_filter = (request.args.get("source") or "").strip()
    source_filter_lc = source_filter.lower()
    # Filter dropdown options: built-ins always, plus any custom labels
    # actually in use, so the operator can slice by them too.
    present = set()
    for blk in annot_current.values():
        present.update((blk.get("labels") or {}).keys())
    custom_present = sorted(present - set(ann.BUILTIN_LABELS))
    annot_filter_labels = list(ann.BUILTIN_LABELS) + custom_present
    return _html_revalidate(render_template(
        "bursts.html",
        active_tab="bursts",
        events_pass=events_pass,
        events_fail=events_fail,
        n_voltage_total=N_VOLTAGE_FRAGMENTS_TOTAL,
        archive_root=str(cands_browser.root),
        archive_available=cands_browser.is_available,
        annot_current=annot_current,
        annot_builtins=list(ann.BUILTIN_LABELS),
        annot_filter_labels=annot_filter_labels,
        tag_filter=tag_filter,
        source_filter=source_filter,
        source_filter_lc=source_filter_lc,
        annot_source_names=annot_source_names,
        events_zombie=events_zombie,
        refined_positions=_refined_positions_bulk(),
        listing_truncated=listing.truncated,
        listing_n_total=listing.n_total,
        listing_n_newest=listing.n_newest,
        listing_n_annotated=listing.n_annotated,
    ))


@app.route("/bursts/<name>")
def burst_event(name: str):
    import json as _json
    detail = cands_browser.event_detail(name)
    if detail is None:
        abort(404)
    # Human-annotation context for the classification card. Best-effort:
    # a DB hiccup must never 500 the event page.
    try:
        annot_block = ann.event_annotations(name)
        annot_history = ann.event_history(name)
        annot_vocab = ann.vocab()
        next_unclassified = _next_unclassified_event(name)
    except Exception:                                       # noqa: BLE001
        LOG.exception("annotations context failed for %s", name)
        annot_block = {"event": name, "classifications": [],
                       "labels": {}, "source_name": None}
        annot_history = []
        annot_vocab = {"builtin_labels": list(ann.BUILTIN_LABELS),
                       "custom_tags": [], "labels": list(ann.BUILTIN_LABELS),
                       "users": [], "source_names": []}
        next_unclassified = None
    # Refined localization context. Best-effort, same policy as above.
    try:
        pos_row = ann.get_position(name)
        refined = _refined_display(pos_row) if pos_row else None
        pos_history = ann.get_position_history(name)
    except Exception:                                       # noqa: BLE001
        LOG.exception("position context failed for %s", name)
        refined = None
        pos_history = []
    return _html_revalidate(render_template(
        "burst_event.html",
        active_tab="bursts",
        event=detail,
        n_voltage_total=N_VOLTAGE_FRAGMENTS_TOTAL,
        c3_pretty=(
            _json.dumps(detail.c3_decision, indent=2, sort_keys=True,
                        default=str)
            if detail.c3_decision else None
        ),
        metadata_pretty=_json.dumps(
            detail.metadata, indent=2, sort_keys=True, default=str,
        ),
        annot_block=annot_block,
        annot_history=annot_history,
        annot_vocab=annot_vocab,
        annot_builtins=list(ann.BUILTIN_LABELS),
        next_unclassified=next_unclassified,
        refined=refined,
        pos_history=pos_history,
    ))


def _next_unclassified_event(current: str) -> Optional[str]:
    """Name of the next event (in the newest-first list order, wrapping)
    that has no current human classification. Returns None if every other
    event is already classified. Best-effort — swallows archive errors."""
    try:
        events = cands_browser.list_events()
    except Exception:                                       # noqa: BLE001
        return None
    names = [e.name for e in events]
    if current not in names:
        return None
    try:
        classified = ann.classified_events()
    except Exception:                                       # noqa: BLE001
        classified = set()
    n = len(names)
    start = names.index(current)
    for off in range(1, n + 1):
        cand = names[(start + off) % n]
        if cand == current:
            break
        if cand not in classified:
            return cand
    return None


# Event plot PNGs are immutable in practice but plotter.py's batch
# re-render tool can rewrite the same paths in place, so: 7-day max-age
# (NOT immutable) with the ETag/conditional defaults ON — a rewritten
# PNG is caught by revalidation after expiry / hard refresh.
@app.route("/bursts/<name>/plot/<plot_name>")
def burst_event_plot(name: str, plot_name: str):
    p = cands_browser.plot_path(name, plot_name)
    if p is None:
        abort(404)
    return send_file(str(p), mimetype="image/png", max_age=604800)


@app.route("/bursts/<name>/filplot/<plot_name>")
def burst_event_fil_plot(name: str, plot_name: str):
    """bbproc coherent-filterbank inspection PNGs (<cand>/filterbank/)."""
    p = cands_browser.fil_plot_path(name, plot_name)
    if p is None:
        abort(404)
    return send_file(str(p), mimetype="image/png", max_age=604800)


# ---------- Sky monitor (E2E test 1: "always seeing the sky") -------------
#
# Each corr node's corr_fast (--sky-export-url) POSTs its slot-0
# StaticSkyMean ~1 s window mean every 30 s; sky_monitor combines the 16
# chgroups in the UV plane, images, sigma-normalises, and writes
# greyscale frames to /dataz/dsa110/operations/sky_monitor/. The /sky
# tab serves a manual-refresh scrubbable movie (30 min default, 24 h
# reach-back).


@app.route("/sky")
def sky_tab():
    return render_template(
        "sky.html",
        active_tab="sky",
        sky_root=str(sky_mon.store.root),
    )


@app.route("/sky/ingest", methods=["POST"])
def sky_ingest():
    """Corr-node snapshot ingest. Body = npz bytes (see
    dsart.services.sky_export.build_snapshot_npz)."""
    body = request.get_data(cache=False)
    if not body:
        return jsonify({"ok": False, "error": "empty body"}), 400
    try:
        ack = sky_mon.ingest(body)
    except ValueError as exc:
        LOG.warning("sky ingest rejected from %s: %s",
                    request.remote_addr, exc)
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:                            # noqa: BLE001
        LOG.exception("sky ingest failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(ack)


@app.route("/sky/frames", methods=["GET"])
def sky_frames():
    """JSON frame index. Query param ``hours`` (float, default 0.5,
    max 24) — how far back to list."""
    try:
        hours = float(request.args.get("hours", "0.5"))
    except ValueError:
        return jsonify({"ok": False, "error": "bad hours"}), 400
    hours = max(0.0, min(hours, 24.0))
    import time as _time
    since = _time.time() - hours * 3600.0
    try:
        frames = sky_mon.store.list_frames(since_unix=since)
    except Exception as exc:                            # noqa: BLE001
        LOG.exception("sky frame listing failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "hours": hours, "frames": frames})


@app.route("/sky/frame/<day>/<name>")
def sky_frame_png(day: str, name: str):
    p = sky_mon.store.resolve_png(day, name)
    if p is None:
        abort(404)
    return send_file(str(p), mimetype="image/png")


@app.route("/sky/frame_meta/<day>/<name>")
def sky_frame_meta(day: str, name: str):
    """JSON sidecar of a frame (in-FOV NVSS source list + astrometry).
    ``{"ok": true, "nvss": []}`` for pre-overlay frames so the page
    table just renders empty."""
    p = sky_mon.store.resolve_sidecar(day, name)
    if p is None:
        return jsonify({"ok": True, "nvss": []})
    try:
        import json as _json
        return jsonify({"ok": True, **_json.loads(p.read_text())})
    except Exception as exc:                            # noqa: BLE001
        LOG.exception("sky frame_meta failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/sky/status", methods=["GET"])
def sky_status():
    """Per-chgroup snapshot freshness + frame counters (debugging
    which corr nodes are reporting)."""
    return jsonify(sky_mon.status())


@app.route("/sky/vetos", methods=["GET"])
def sky_vetos():
    """Live C2 sidereal (l,m) dump-veto registry (active regions).

    Reads ``/mon/c2/sidereal_vetos`` so the sky page lists the active
    vetos independent of frame cadence.
    """
    try:
        doc = etcd_store.get_dict("/mon/c2/sidereal_vetos")
    except Exception as exc:                                # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc),
                        "regions": [], "tol_arcsec": 0.0})
    if not isinstance(doc, dict):
        return jsonify({"ok": True, "regions": [], "tol_arcsec": 0.0})
    return jsonify({
        "ok": True,
        "regions": doc.get("regions", []),
        "tol_arcsec": doc.get("tol_arcsec", 0.0),
        "n_active": doc.get("n_active", 0),
        "ts": doc.get("ts"),
    })


# ---------- Plot endpoints (PNG) ------------------------------------------


def _png_response(png_bytes: bytes):
    """In-memory per-request render — must NEVER be browser-cached
    (these are the live RFI plots; a cached copy would freeze the
    dashboard's live view). max_age=0 + no-store is explicit armour
    against SEND_FILE_MAX_AGE_DEFAULT ever applying here."""
    import io
    resp = send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="plot.png",
        max_age=0,
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
    import inject_calibration as ic                                 # local
    import spectral_line_gate as slg                                # local
    try:
        spl_state = slg.get_spectral_line_state(control_store)
    except Exception as exc:                                        # noqa: BLE001
        LOG.warning("get_spectral_line_state: %s", exc)
        spl_state = {
            "subbands": {}, "n_enabled": 0, "default": True,
            "nfreq_int_choices": slg.divisors_of_nchan(),
            "default_integration_s": slg.DEFAULT_INTEGRATION_S,
            "_error": str(exc),
        }
    # Per-chgroup sub-band frequency coverage + native channel width, from
    # the authoritative band constants (band top f0=1.530 GHz, descending).
    try:
        from dsart.common.constants import (
            freq_GHz, NCHAN_PER_CHGROUP, DELTA_NU_CH_GHZ, N_CHGROUP,
        )
        spl_chgroup_freqs = {
            g: {
                "top_mhz": freq_GHz(g, 0) * 1.0e3,
                "bot_mhz": freq_GHz(g, NCHAN_PER_CHGROUP - 1) * 1.0e3,
            }
            for g in range(N_CHGROUP)
        }
        spl_raw_chan_khz = DELTA_NU_CH_GHZ * 1.0e6   # GHz -> kHz
    except Exception as exc:                                        # noqa: BLE001
        LOG.warning("spl freq map: %s", exc)
        spl_chgroup_freqs = {}
        spl_raw_chan_khz = 0.030517578125 * 1.0e3    # native Δν fallback
    return render_template(
        "control.html",
        active_tab="control",
        corr_cn_ids=list(CORR_CN_IDS),
        search_cn_ids=list(SEARCH_CN_IDS),
        arm_info=arm_info,
        recent_audit=recent,
        default_arm_margin=DEFAULT_ARM_SEQ_MARGIN,
        default_inject_margin_blocks=DEFAULT_INJECT_MARGIN_BLOCKS,
        default_bounce_sleep_s=DEFAULT_BOUNCE_SLEEP_S,
        snr_cal_prefix=ic.CALIBRATION_PREFIX,
        corr_nodes=list(CORR_NODES),
        spectral_line_state=spl_state,
        spectral_line_key=slg.SPECTRAL_LINE_KEY,
        spectral_line_max_integration_s=slg.MAX_INTEGRATION_S,
        spectral_line_min_integration_s=slg.MIN_INTEGRATION_S,
        spl_chgroup_freqs=spl_chgroup_freqs,
        spl_raw_chan_khz=spl_raw_chan_khz,
        h23_dsart_units=list(H23_DSART_UNITS),
    )


@app.route("/control/system_state", methods=["GET"])
def control_system_state():
    """One rolled-up fleet lifecycle state for the Control banner
    (Ready / Preparing / Prepared / Observing / Offline)."""
    try:
        st = compute_system_state(control_store)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("compute_system_state failed")
        return jsonify({
            "ok": False,
            "state": "offline",
            "label": "OFFLINE",
            "detail": f"state poll failed: {exc}",
            "safe_to_arm": False,
        }), 200
    st["ok"] = True
    return jsonify(st)


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

    user = request.form.get("user") or request.remote_addr or "anon"

    # Start-time housekeeping: wipe the accumulated "safe to delete on
    # restart" ephemera on every corr + search node BEFORE the start
    # verb fans out. h23 (the candidate/CSV archive) is never touched.
    # Best-effort — a cleanup failure must never block the start.
    import fleet_services                                            # local
    try:
        node_cleanup = fleet_services.cleanup_nodes_for_start()
    except Exception as exc:                                         # noqa: BLE001
        LOG.exception("start-time node cleanup failed (continuing)")
        node_cleanup = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        audit_log(
            control_store, namespace="corr_rt+search_rt", cn_target="all",
            cmd="start_cleanup", val=None, ok=bool(node_cleanup.get("ok")),
            note=(
                f"n_ok={node_cleanup.get('n_ok')} "
                f"n_failed={node_cleanup.get('n_failed')}"
            ),
            user=user,
        )
    except Exception:                                               # noqa: BLE001
        LOG.exception("start_cleanup audit_log failed (continuing)")

    try:
        result = control_start_fleet(
            control_store, obs_dec_deg=obs_dec, user=user,
        )
    except Exception as exc:                                         # noqa: BLE001
        LOG.exception("control verb control_start_fleet failed")
        try:
            audit_log(
                control_store, namespace="control", cn_target="-",
                cmd="control_start_fleet", val={"obs_dec_deg": obs_dec},
                ok=False, note=f"exception: {exc!r}", user=user,
            )
        except Exception:                                           # noqa: BLE001
            LOG.exception("audit_log also failed (continuing)")
        return jsonify({
            "ok": False, "error": str(exc), "node_cleanup": node_cleanup,
        }), 500
    result["node_cleanup"] = node_cleanup
    return jsonify(result)


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
    ``margin_blocks``, ``chgroups``, ``target_snr``, and
    ``fluence_jy_ms`` when ``target_snr`` is set):

      inj_id          str (e.g. "phase6c_extragal_t1")
      l_rad           float, |l| < 1, |l| <= INJECT_LM_MAX_RAD (0.0279)
      m_rad           float, |m| < 1 and l^2 + m^2 < 1,
                      |m| <= INJECT_LM_MAX_RAD (0.0279). Beyond the
                      imaged half-FoV the source FFT-aliases to the
                      conjugate sky position; recommended |l|,|m| <=
                      0.02 rad (primary-beam half-power ~0.025 rad).
      dm_pc_cm3       float
      fluence_jy_ms   float — required iff ``target_snr`` is unset
      target_snr      float (optional). When set, the dashboard
                      derives ``fluence_jy_ms`` from the stored K
                      calibration for the DM-band bucket (K is
                      width-independent post-F-fix-injector-fluence-
                      norm) and uses that instead of any operator-
                      supplied fluence. Returns 412 if K is missing
                      for the DM band.
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
    target_snr_raw = (f.get("target_snr") or "").strip()
    try:
        kwargs: dict[str, object] = {
            "inj_id": (f.get("inj_id") or "").strip() or None,
            "l_rad": float(f.get("l_rad", "0")),
            "m_rad": float(f.get("m_rad", "0")),
            "dm_pc_cm3": float(f.get("dm_pc_cm3", "0")),
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

    # Resolve fluence: either explicit OR derived from target_snr via K.
    target_snr: float | None = None
    if target_snr_raw:
        try:
            target_snr = float(target_snr_raw)
            if not (target_snr > 0):
                raise ValueError("must be > 0")
        except ValueError as exc:
            return jsonify({
                "ok": False,
                "error": f"target_snr={target_snr_raw!r}: {exc}",
            }), 400
        import inject_calibration as ic  # local
        cs = ic.CalibrationStore(control_store)
        entry = cs.get(dm_pc_cm3=float(kwargs["dm_pc_cm3"]))
        if entry is None or not (entry.K > 0):
            return jsonify({
                "ok": False,
                "error": (
                    f"No K calibration for DM bucket "
                    f"{ic.bucket_key(float(kwargs['dm_pc_cm3']))}; "
                    f"run /control/inject_calibrate first (any width)."
                ),
                "bucket": ic.bucket_key(float(kwargs["dm_pc_cm3"])),
            }), 412
        try:
            fluence = ic.snr_to_fluence(
                target_snr=target_snr, K=entry.K,
                width_samples=int(kwargs["width_samples"]),
            )
        except ValueError as exc:
            return jsonify({
                "ok": False, "error": f"snr_to_fluence: {exc}",
            }), 400
        kwargs["fluence_jy_ms"] = float(fluence)
        kwargs["target_snr"] = target_snr
    else:
        try:
            kwargs["fluence_jy_ms"] = float(f.get("fluence_jy_ms", "0"))
        except ValueError as exc:
            return jsonify({
                "ok": False, "error": f"bad fluence_jy_ms: {exc}",
            }), 400

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


@app.route("/control/inject_calibrate", methods=["POST"])
def control_inject_calibrate_post():
    """M7.4 Phase 6c.A: fire one calibration probe and store the
    resulting K for the DM-band bucket.

    K is width-independent post-F-fix-injector-fluence-norm — a probe
    at any width pins K for every width within the band. The probe's
    ``width_samples`` is recorded in the entry as informational
    metadata only.

    Form fields (all optional except ``dm_pc_cm3``):

      dm_pc_cm3       float — required
      l_rad, m_rad    floats (default 0, boresight)
      width_samples   int (default 32)
      fluence_jy_ms   float (default DEFAULT_CALIBRATION_FLUENCE,
                      0.01 Jy·ms after the 2026-06-09 rescale)
      profile         "gaussian" | "boxcar" (default gaussian)
      poll_timeout_s  float (default 30.0)
      chgroups        comma-separated ints (default all 16)
      margin_blocks   int (default DEFAULT_INJECT_MARGIN_BLOCKS)

    Returns a JSON :class:`inject_calibration.ProbeResult` dict.
    """
    import inject_calibration as ic   # local
    f = request.form
    try:
        kwargs: dict[str, Any] = {
            "dm_pc_cm3": float(f["dm_pc_cm3"]),
            "l_rad": float(f.get("l_rad", "0")),
            "m_rad": float(f.get("m_rad", "0")),
            "width_samples": int(
                f.get("width_samples", str(ic.DEFAULT_CALIBRATION_WIDTH)),
            ),
            "fluence_jy_ms": float(
                f.get("fluence_jy_ms", str(ic.DEFAULT_CALIBRATION_FLUENCE)),
            ),
            "profile": (f.get("profile") or ic.DEFAULT_CALIBRATION_PROFILE).strip(),
            "poll_timeout_s": float(
                f.get("poll_timeout_s", str(ic.DEFAULT_POLL_TIMEOUT_S)),
            ),
        }
    except (KeyError, ValueError) as exc:
        return jsonify({
            "ok": False, "error": f"bad form field: {exc}",
        }), 400

    raw_chg = (f.get("chgroups") or "").strip()
    if raw_chg:
        try:
            kwargs["chgroups"] = tuple(
                int(s) for s in raw_chg.split(",") if s.strip()
            )
        except ValueError:
            return jsonify({
                "ok": False,
                "error": f"chgroups={raw_chg!r}: must be comma-separated ints",
            }), 400

    raw_margin = (f.get("margin_blocks") or "").strip()
    if raw_margin:
        try:
            kwargs["margin_blocks"] = int(raw_margin)
        except ValueError:
            return jsonify({
                "ok": False,
                "error": f"margin_blocks={raw_margin!r}: not an int",
            }), 400

    user = request.form.get("user") or request.remote_addr or "anon"
    kwargs["user"] = user
    # Re-use the existing inject helper so the per-chgroup PUT path
    # (with its audit row + active-registry side effect) is shared
    # between the calibration probe and the operator's manual inject.
    kwargs["inject_fn"] = control_inject_pulse

    # T6/T7 (2026-06-07): per-form opt-in for the health-gated, SNR-
    # laddered calibration helper. Default ON so the operator-visible
    # behaviour is "calibrate just works" without a 30 s no_match
    # poll on a stale fleet. Pass ``health_check=0`` in the form to
    # fall back to the legacy single-shot probe for forensic / replay
    # use cases.
    health_check = (f.get("health_check") or "1").strip()
    use_ladder = (f.get("use_ladder") or "1").strip()
    raw_ladder = (f.get("fluence_ladder") or "").strip()
    if raw_ladder:
        try:
            ladder_steps = tuple(
                float(s) for s in raw_ladder.split(",") if s.strip()
            )
        except ValueError:
            return jsonify({
                "ok": False,
                "error": (
                    f"fluence_ladder={raw_ladder!r}: must be comma-"
                    "separated floats"
                ),
            }), 400
    else:
        ladder_steps = ic.DEFAULT_FLUENCE_LADDER

    try:
        if use_ladder not in ("0", "false", "False", "no", "off"):
            ladder_kwargs = dict(kwargs)
            ladder_kwargs["health_check"] = (
                health_check not in ("0", "false", "False", "no", "off")
            )
            ladder_kwargs["fluence_ladder"] = ladder_steps
            attempts = ic.fire_calibration_probe_with_ladder(
                control_store, **ladder_kwargs,
            )
            result = attempts[-1] if attempts else None
            if result is None:
                return jsonify({
                    "ok": False, "error": "no_attempts_made",
                }), 500
            payload = result.to_dict()
            payload["attempts"] = [a.to_dict() for a in attempts]
        else:
            result = ic.fire_calibration_probe(control_store, **kwargs)
            payload = result.to_dict()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("fire_calibration_probe failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    audit_log(
        control_store,
        namespace="dsart.inject",
        cn_target="fleet",
        cmd="inject_calibrate",
        val={
            "dm_pc_cm3": kwargs["dm_pc_cm3"],
            "width_samples": kwargs["width_samples"],
            "fluence_jy_ms": kwargs["fluence_jy_ms"],
            "observed_snr": result.observed_snr,
            "K": result.K,
            "bucket": result.bucket,
            "reason": result.reason,
            "n_attempts": len(payload.get("attempts", [])) or 1,
        },
        ok=bool(result.ok),
        note=f"elapsed={result.elapsed_s:.1f}s inj_id={result.inj_id}",
        user=user,
    )
    return jsonify(payload)


@app.route("/control/inject_calibrations", methods=["GET"])
def control_inject_calibrations_get():
    """List every persisted DM-band calibration bucket.

    Legacy ``dmNNNN_wWWWW`` rows (pre-F-fix-injector-fluence-norm)
    are returned alongside the new ``dmNNNN`` rows so the operator
    can spot and wipe them via the "Delete fluence/SNR calibration"
    Control verb.
    """
    import inject_calibration as ic   # local
    cs = ic.CalibrationStore(control_store)
    try:
        entries = cs.list_all()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("CalibrationStore.list_all failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({
        "ok": True,
        "entries": [e.to_dict() for e in entries],
    })


# ---------------------------------------------------------------------------
# Fringe-stopping table panel (M7.4 follow-up — surfaced 2026-06-03 after
# the cold-start meridian_fringestop crash on dec=+54.5 deadlocked the
# whole fast-corr → search path via bada → fada → corr_fast). The same
# class of cryptic "stuck after ~300 cubes" symptom is now caught by an
# explicit fstable-presence traffic light next to "Start fleet".
# ---------------------------------------------------------------------------


def _fstable_runtime_params() -> tuple[Optional[int], Optional[float], Optional[str]]:
    """Pull ``(nant, refmjd)`` from etcd's ``/cnf/corr`` + ``/cnf/fringe``.

    Returns ``(nant, refmjd, err)`` with ``err`` non-None when either
    key is missing/malformed — the routes surface that as ``ok=False``
    so the UI doesn't pretend to know the filename it would build.
    """
    try:
        corr = etcd_store.get_dict("/cnf/corr")
        fringe = etcd_store.get_dict("/cnf/fringe")
    except Exception as exc:                                  # noqa: BLE001
        LOG.warning("fstable: etcd read failed: %s", exc)
        return None, None, f"etcd read failed: {exc!r}"
    if not isinstance(corr, dict) or not isinstance(fringe, dict):
        return None, None, "/cnf/corr or /cnf/fringe missing"
    try:
        ant_od = corr["antenna_order"]
        nant = int(corr.get("nant", len(list(ant_od.values()))))
        refmjd = float(fringe["refmjd"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, None, f"cnf parse error: {exc!r}"
    return nant, refmjd, None


def _fstable_pointing_dec() -> tuple[Optional[float], Optional[str]]:
    """Best-effort pointing DEC for the traffic light.

    Tries ``/mon/array/dec`` first (the M7.4 CUSTOMDEC fallback in
    ``meridian_fringestop_rt`` reads the same key). Returns
    ``(dec_deg, err)``; ``err`` non-None when the key is missing.
    """
    try:
        d = etcd_store.get_dict("/mon/array/dec")
    except Exception as exc:                                  # noqa: BLE001
        return None, f"etcd read failed: {exc!r}"
    if not isinstance(d, dict) or "dec_deg" not in d:
        return None, "/mon/array/dec missing or malformed"
    try:
        return float(d["dec_deg"]), None
    except (TypeError, ValueError) as exc:
        return None, f"dec parse error: {exc!r}"


@app.route("/control/fstables/list", methods=["GET"])
def control_fstables_list():
    """Fleet inventory: h23 master copy + every corr node's local
    cache. Returns ``{ok, h23: [...], fleet: {host: {ok, entries, error}}}``.

    The frontend renders this as a per-DEC table where each row has
    one column per host (✓/✗ for presence).
    """
    import fstable_panel as fp                              # local
    from services_inventory import CORR_HOSTS
    try:
        h23 = [e.to_dict() for e in fp.list_h23_tables()]
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("list_h23_tables failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    try:
        fleet = fp.list_fleet_tables(list(CORR_HOSTS))
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("list_fleet_tables failed")
        return jsonify({
            "ok": False, "error": f"fleet inventory failed: {exc}",
            "h23": h23, "fleet": {},
        }), 500
    return jsonify({"ok": True, "h23": h23, "fleet": fleet})


@app.route("/control/fstables/current_status", methods=["GET"])
def control_fstables_current_status():
    """Traffic-light data for the operator's current pointing DEC.

    Query params:

      dec_deg   optional override (deg). When absent we read
                ``/mon/array/dec``.

    Response: see :func:`fstable_panel.current_dec_status`.
    """
    import fstable_panel as fp                              # local
    from services_inventory import CORR_HOSTS
    dec_raw_q = (request.args.get("dec_deg") or "").strip()
    dec_deg: Optional[float]
    if dec_raw_q:
        try:
            dec_deg = float(dec_raw_q)
        except ValueError:
            return jsonify({"ok": False, "error": f"bad dec_deg={dec_raw_q!r}"}), 400
    else:
        dec_deg, derr = _fstable_pointing_dec()
        if dec_deg is None:
            # Render as "unknown" rather than hard-failing — operator
            # may simply not have set the pointing yet.
            return jsonify(fp.current_dec_status(
                dec_deg=None, nant=96, refmjd=58849.0,
                corr_hosts=list(CORR_HOSTS),
            ) | {"_note": derr or "no dec"})

    nant, refmjd, perr = _fstable_runtime_params()
    if nant is None or refmjd is None:
        return jsonify({"ok": False, "error": perr or "cnf missing"}), 500

    try:
        status = fp.current_dec_status(
            dec_deg=dec_deg, nant=nant, refmjd=refmjd,
            corr_hosts=list(CORR_HOSTS),
        )
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("current_dec_status failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(status)


@app.route("/control/fstables/build", methods=["POST"])
def control_fstables_build_post():
    """Build one per-DEC fringe-stopping table on h23 in casa38.

    Form fields:

      dec_deg     float, required — operator's raw pointing dec;
                  snapped to the 0.25° grid before the build script
                  is invoked.
      force       optional "true"/"false" (default false). When true
                  we pass ``--force`` so an existing master file is
                  overwritten.

    Synchronous; ~30 s per DEC. Audited.
    """
    import fstable_panel as fp                              # local
    user = request.form.get("user") or request.remote_addr or "anon"
    dec_raw = (request.form.get("dec_deg") or "").strip()
    if not dec_raw:
        return jsonify({"ok": False, "error": "dec_deg required"}), 400
    try:
        dec_deg = float(dec_raw)
    except ValueError:
        return jsonify({"ok": False, "error": f"bad dec_deg={dec_raw!r}"}), 400
    force = (request.form.get("force") or "").strip().lower() in ("true", "1", "yes")

    nant, refmjd, perr = _fstable_runtime_params()
    if nant is None or refmjd is None:
        return jsonify({"ok": False, "error": perr or "cnf missing"}), 500

    try:
        res = fp.build_table_for_dec(
            dec_deg=dec_deg, nant=nant, refmjd=refmjd, force=force,
        )
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("build_table_for_dec failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    try:
        audit_log(
            control_store,
            namespace="control",
            cn_target="h23",
            cmd="fstable_build",
            val={
                "dec_deg_raw": res.dec_deg_raw,
                "dec_deg_grid": res.dec_deg_grid,
                "filename": res.expected_filename,
                "nant": nant, "refmjd": refmjd, "force": force,
            },
            ok=bool(res.ok),
            note=(
                f"elapsed={res.elapsed_s:.1f}s rc={res.rc} "
                f"err={res.error!r}"
            ),
            user=user,
        )
    except Exception:                                       # noqa: BLE001
        LOG.exception("audit_log fstable_build also failed (continuing)")
    return jsonify(res.to_dict())


@app.route("/control/fstables/deploy", methods=["POST"])
def control_fstables_deploy_post():
    """rsync one h23 master ``*.npz`` to every corr node.

    Form fields:

      filename   basename (e.g.
                 ``fringestopping_table_dec_+54.5000deg_96ant_refmjd58849.000000.npz``),
                 required. Must exist in
                 :data:`fstable_panel.H23_MASTER_DIR`.

    Returns per-host rsync results; ``all_ok=False`` if any host
    failed. Audited.
    """
    import fstable_panel as fp                              # local
    from services_inventory import CORR_HOSTS
    user = request.form.get("user") or request.remote_addr or "anon"
    filename = (request.form.get("filename") or "").strip()
    if not filename:
        return jsonify({"ok": False, "error": "filename required"}), 400
    # Belt-and-braces: refuse anything with a path separator or '..'
    # so a malformed/spoofed form can't ship arbitrary files out.
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        return jsonify({"ok": False, "error": "filename must be a bare basename"}), 400

    try:
        result = fp.deploy_table_to_fleet(filename, list(CORR_HOSTS))
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("deploy_table_to_fleet failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    try:
        audit_log(
            control_store,
            namespace="control",
            cn_target="fleet",
            cmd="fstable_deploy",
            val={"filename": filename},
            ok=bool(result.get("all_ok")),
            note=(
                "hosts ok="
                f"{sum(1 for h in result.get('hosts', []) if h.get('ok'))}/"
                f"{len(result.get('hosts', []))}"
            ),
            user=user,
        )
    except Exception:                                       # noqa: BLE001
        LOG.exception("audit_log fstable_deploy also failed (continuing)")
    return jsonify(result)


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


# --- M8 voltage dumps: /cmd/c2/voltages_enabled kill-switch -----------------
@app.route("/control/voltages_enabled", methods=["GET", "POST"])
def control_voltages_enabled():
    """Read or flip the C2 voltage-broadcast kill-switch.

    The C2 coincidencer polls ``/cmd/c2/voltages_enabled`` and only fans
    out the ``DUMP_VOLTAGE`` UDP triggers to the 16 corr nodes when
    ``enabled=True``. **Fail-CLOSED**: a missing key means voltages are
    DISABLED (an expensive ~103 GiB/event capability stays dark until an
    operator explicitly enables it). Injections never dump voltages
    regardless of this switch.

    GET  → ``{"enabled", "ts", "actor", "reason", "default"}``
           (default=True iff the key is missing → enabled=False).
    POST form fields:
      ``enabled``  ``true`` / ``false`` (required).
      ``reason``   free text, 1..240 chars (required, audited).
      ``confirm``  literal ``enable`` when enabling, ``disable`` when
                   disabling — deliberate speed-bump.
    """
    import voltage_controls                                       # local

    if request.method == "GET":
        try:
            state = voltage_controls.get_voltages_state(control_store)
        except Exception as exc:                                  # noqa: BLE001
            LOG.exception("get_voltages_state failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, **state})

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
                f"{voltage_controls.MAX_REASON_LEN} chars)"
            ),
        }), 400
    if len(reason_raw) > voltage_controls.MAX_REASON_LEN:
        return jsonify({
            "ok": False,
            "error": (
                f"reason too long: {len(reason_raw)} chars (max "
                f"{voltage_controls.MAX_REASON_LEN})"
            ),
        }), 400

    confirm_raw = (f.get("confirm") or "").strip().lower()
    expected_confirm = "enable" if new_enabled else "disable"
    if confirm_raw != expected_confirm:
        return jsonify({
            "ok": False,
            "error": (
                f"confirm={confirm_raw!r} does not match the requested "
                f"direction — type the literal word "
                f"{expected_confirm!r} to arm the flip"
            ),
        }), 400

    actor = f.get("user") or request.remote_addr or "anon"
    try:
        new_state = voltage_controls.set_voltages_state(
            control_store, enabled=new_enabled, reason=reason_raw, actor=actor,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("set_voltages_state failed")
        try:
            audit_log(
                control_store, namespace="c2.voltages_toggle",
                cn_target="h23", cmd="voltages_toggle",
                val={"enabled": new_enabled}, ok=False,
                note=f"exception: {exc!r}", user=actor,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log fallback also failed (continuing)")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **new_state})


@app.route("/control/voltages_audit", methods=["GET"])
def control_voltages_audit():
    """Most recent voltages_toggle audit rows (JSON), newest first."""
    import voltage_controls                                       # local
    try:
        limit = max(1, min(int(request.args.get("limit", "5")), 100))
    except ValueError:
        limit = 5
    try:
        rows = voltage_controls.list_recent_voltage_toggles(
            control_store, limit=limit)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("list_recent_voltage_toggles failed")
        return jsonify({"ok": False, "error": str(exc), "rows": []}), 500
    return jsonify({"ok": True, "rows": rows})


# --- M8 voltage dumps: /cmd/c3/flag_only keep-vs-delete mode ----------------
@app.route("/control/c3_mode", methods=["GET", "POST"])
def control_c3_mode():
    """Read or flip the C3 keep/delete mode (``/cmd/c3/flag_only``).

    C3 reads this per event. ``flag_only=True`` (the safe default) keeps
    everything: it collects voltages + logs every veto decision but does
    NO destructive REJECT cleanup. ``flag_only=False`` enables the
    conservative delete path (delete staged voltages + cubes, MOVE
    metadata/plots to ``candidates_rejected/``; never an ``rm -rf``).
    Missing key ⇒ C3 falls back to its configured default (also True).

    GET  → ``{"flag_only", "ts", "actor", "reason", "default"}``.
    POST form fields:
      ``flag_only``  ``true`` (keep) / ``false`` (delete) (required).
      ``reason``     free text, 1..240 chars (required, audited).
      ``confirm``    literal ``keep`` when flag_only=true, ``delete``
                     when flag_only=false.
    """
    import voltage_controls                                       # local

    if request.method == "GET":
        try:
            state = voltage_controls.get_c3_mode_state(control_store)
        except Exception as exc:                                  # noqa: BLE001
            LOG.exception("get_c3_mode_state failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, **state})

    f = request.form
    flag_raw = (f.get("flag_only") or "").strip().lower()
    if flag_raw in ("true", "1", "yes", "on", "keep"):
        new_flag_only = True
    elif flag_raw in ("false", "0", "no", "off", "delete"):
        new_flag_only = False
    else:
        return jsonify({
            "ok": False,
            "error": (
                f"flag_only={flag_raw!r}: must be true (keep) / false "
                f"(delete)"
            ),
        }), 400

    reason_raw = (f.get("reason") or "").strip()
    if not reason_raw:
        return jsonify({
            "ok": False,
            "error": (
                "reason is required (non-empty string, max "
                f"{voltage_controls.MAX_REASON_LEN} chars)"
            ),
        }), 400
    if len(reason_raw) > voltage_controls.MAX_REASON_LEN:
        return jsonify({
            "ok": False,
            "error": (
                f"reason too long: {len(reason_raw)} chars (max "
                f"{voltage_controls.MAX_REASON_LEN})"
            ),
        }), 400

    confirm_raw = (f.get("confirm") or "").strip().lower()
    # Enabling DELETE (flag_only=false) is the dangerous direction.
    expected_confirm = "keep" if new_flag_only else "delete"
    if confirm_raw != expected_confirm:
        return jsonify({
            "ok": False,
            "error": (
                f"confirm={confirm_raw!r} does not match the requested "
                f"direction — type the literal word "
                f"{expected_confirm!r} to arm the flip"
            ),
        }), 400

    actor = f.get("user") or request.remote_addr or "anon"
    try:
        new_state = voltage_controls.set_c3_mode_state(
            control_store, flag_only=new_flag_only, reason=reason_raw,
            actor=actor,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("set_c3_mode_state failed")
        try:
            audit_log(
                control_store, namespace="c3.mode_toggle",
                cn_target="h23", cmd="c3_mode_toggle",
                val={"flag_only": new_flag_only}, ok=False,
                note=f"exception: {exc!r}", user=actor,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log fallback also failed (continuing)")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **new_state})


@app.route("/control/c3_mode_audit", methods=["GET"])
def control_c3_mode_audit():
    """Most recent c3_mode_toggle audit rows (JSON), newest first."""
    import voltage_controls                                       # local
    try:
        limit = max(1, min(int(request.args.get("limit", "5")), 100))
    except ValueError:
        limit = 5
    try:
        rows = voltage_controls.list_recent_c3_toggles(
            control_store, limit=limit)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("list_recent_c3_toggles failed")
        return jsonify({"ok": False, "error": str(exc), "rows": []}), 500
    return jsonify({"ok": True, "rows": rows})


# --- operator-integration: human authority over the dsa110-operator agent ---
@app.route("/control/operator", methods=["GET", "POST"])
def control_operator():
    """Read or write the operator-agent authority key ``/cmd/operator/control``.

    GET  -> {ok, control: {...}, lease: {...}|null}
    POST -> form: agents_enabled (on/off), executor_email (optional),
            max_obs_seconds (int, 0=off), confirm (=operator), user.

    The dsa110-operator agent treats this key as a one-way human override:
    it can read it but never write it (its etcd write surface is confined to
    /operator/ and /cmd/ant/). ``max_obs_seconds`` is enforced strictly by a
    dsart_rt watchdog, not just the agent.
    """
    import operator_control as opc                              # local

    if request.method == "GET":
        try:
            return jsonify({"ok": True,
                            "control": opc.get_operator_control(control_store),
                            "lease": opc.get_operator_lease(control_store)})
        except Exception as exc:                                  # noqa: BLE001
            LOG.exception("get operator control failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    f = request.form
    confirm_raw = (f.get("confirm") or "").strip().lower()
    if confirm_raw != "operator":
        return jsonify({"ok": False,
                        "error": "type the literal word 'operator' to confirm"}), 400
    enabled_raw = (f.get("agents_enabled") or "").strip().lower()
    agents_enabled = enabled_raw in ("true", "1", "yes", "on")
    actor = f.get("user") or request.remote_addr or "anon"
    try:
        state = opc.set_operator_control(
            control_store,
            agents_enabled=agents_enabled,
            executor_email=(f.get("executor_email") or "").strip(),
            max_obs_seconds=(f.get("max_obs_seconds") or "0").strip() or "0",
            actor=actor)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:                                      # noqa: BLE001
        LOG.exception("set operator control failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "control": state})


@app.route("/control/spectral_line", methods=["GET", "POST"])
def control_spectral_line():
    """Read or write the per-sub-band spectral-line (SPL) config.

    The ``dsart_rt`` orchestrators read ``/cnf/spectral_line`` at
    ``start`` and, per node, spawn either ``meridian_fringestop_spl``
    (SPL on) or ``bada_null_drain`` (SPL off) as the constant second
    ``bada`` reader. Editing here does NOT hot-swap a running fleet —
    it takes effect on the next ``restart_all`` + ``start``.

    GET
        Returns the full 16-chgroup state (``get_spectral_line_state``).

    POST form fields:

      ``subbands``  JSON object mapping chgroup → ``{"enabled",
                    "integration_s", "nfreq_int"}`` (required).
      ``reason``    free text, 1..240 chars (required, audited).
      ``confirm``   must be the literal word ``spectral_line``.

    Returns 200 + new state on success / GET; 400 on validation /
    confirm errors; 500 (with audit row) on etcd transport errors.
    """
    import json                                                     # local
    import spectral_line_gate as slg                                # local

    if request.method == "GET":
        try:
            state = slg.get_spectral_line_state(control_store)
        except Exception as exc:                                    # noqa: BLE001
            LOG.exception("get_spectral_line_state failed")
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, **state})

    # ---- POST ----
    f = request.form
    confirm_raw = (f.get("confirm") or "").strip().lower()
    if confirm_raw != "spectral_line":
        return jsonify({
            "ok": False,
            "error": (
                "confirm must be the literal word 'spectral_line' "
                "to save the SPL config (deliberate speed-bump)."
            ),
        }), 400

    reason_raw = (f.get("reason") or "").strip()
    if not reason_raw:
        return jsonify({
            "ok": False,
            "error": (
                "reason is required (non-empty string, max "
                f"{slg.MAX_REASON_LEN} chars)"
            ),
        }), 400

    subbands_raw = f.get("subbands") or ""
    try:
        subbands = json.loads(subbands_raw)
    except Exception as exc:                                        # noqa: BLE001
        return jsonify({
            "ok": False,
            "error": f"subbands must be a JSON object: {exc}",
        }), 400
    if not isinstance(subbands, dict):
        return jsonify({
            "ok": False, "error": "subbands must be a JSON object",
        }), 400

    actor = f.get("user") or request.remote_addr or "anon"
    try:
        new_state = slg.set_spectral_line_state(
            control_store,
            subbands=subbands,
            reason=reason_raw,
            actor=actor,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:                                        # noqa: BLE001
        LOG.exception("set_spectral_line_state failed")
        try:
            audit_log(
                control_store,
                namespace="control.spectral_line",
                cn_target="fleet",
                cmd="spectral_line_set",
                val={"raw_len": len(subbands_raw)},
                ok=False,
                note=f"exception: {exc!r}",
                user=actor,
            )
        except Exception:                                          # noqa: BLE001
            LOG.exception("audit_log fallback also failed (continuing)")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **new_state})


@app.route("/control/spectral_line_audit", methods=["GET"])
def control_spectral_line_audit():
    """Most recent spectral_line save audit rows (JSON), newest first."""
    import spectral_line_gate as slg                                # local
    try:
        limit = max(1, min(int(request.args.get("limit", "5")), 100))
    except ValueError:
        limit = 5
    try:
        rows = slg.list_recent_changes(control_store, limit=limit)
    except Exception as exc:                                        # noqa: BLE001
        LOG.exception("list_recent_changes failed")
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


# ---------- Control tab (M7.4 Phase 8 v3): quick recovery + live C2 -------
#
# Three actions + three read-only views, all surfaced on the new
# "C2 / search quick recovery" + "Recent C2 activity" panels:
#
#   POST /control/bounce_search   — stop+sleep+start on selected sids,
#                                   repairs the c1_emit wedge.
#   POST /control/restart_c2      — local systemctl --user restart of
#                                   dsart_c2.service.
#   GET  /control/c2_snapshot     — /mon/c2/h23 counters + matcher
#                                   snapshot for the live mini-panel.
#   GET  /control/c2_journal      — recent FIRE/WOULD-DUMP/LOG class /
#                                   inject_match lines from journalctl.
#   GET  /control/recent_events   — last N candidate dirs (compact
#                                   JSON for inline display).
#
# Every POST is confirm-gated to mirror the existing stop / restart_all
# / dump_now speed-bumps.


@app.route("/control/bounce_search", methods=["POST"])
def control_bounce_search_post():
    """Bounce (stop+sleep+start) one or more search cn_ids.

    Repairs the recurring c1_emit-queue-wedge failure mode where the
    TCP socket to C2 looks connected but the send thread is blocked
    and ``total_dropped`` climbs without bound.

    Form fields:

      confirm    Must literally equal ``bounce_search``.
      cn_ids     Comma-separated sid list (e.g. ``1`` or ``1,2``).
                 Blank → bounce every search cn (SEARCH_CN_IDS).
      sleep_s    Optional float; default ``DEFAULT_BOUNCE_SLEEP_S``.

    Returns the dict :func:`bounce_search` returns, plus a ``user``
    field with the recorded actor.
    """
    confirm = request.form.get("confirm", "").strip()
    if confirm != "bounce_search":
        return jsonify({
            "ok": False,
            "error": (
                "bounce_search requires confirm=bounce_search in the "
                "POST body — this is a deliberate safety speed bump."
            ),
        }), 400

    cn_ids_raw = (request.form.get("cn_ids") or "").strip()
    cn_ids: Optional[list[int]]
    if cn_ids_raw == "":
        cn_ids = None
    else:
        try:
            cn_ids = [int(c.strip()) for c in cn_ids_raw.split(",") if c.strip()]
        except ValueError:
            return jsonify({
                "ok": False,
                "error": f"cn_ids={cn_ids_raw!r}: not a CSV of ints",
            }), 400
        if not cn_ids:
            return jsonify({
                "ok": False,
                "error": "cn_ids was empty after parsing",
            }), 400
        bad = [c for c in cn_ids if c not in SEARCH_CN_IDS]
        if bad:
            return jsonify({
                "ok": False,
                "error": (
                    f"cn_ids {bad} not in search SEARCH_CN_IDS="
                    f"{list(SEARCH_CN_IDS)}"
                ),
            }), 400

    sleep_raw = (request.form.get("sleep_s") or "").strip()
    try:
        sleep_s = float(sleep_raw) if sleep_raw else DEFAULT_BOUNCE_SLEEP_S
    except ValueError:
        return jsonify({
            "ok": False,
            "error": f"sleep_s={sleep_raw!r}: not a float",
        }), 400
    if sleep_s < 0.0 or sleep_s > 30.0:
        return jsonify({
            "ok": False,
            "error": f"sleep_s={sleep_s} outside [0, 30]",
        }), 400

    return _control_json_or_error(
        bounce_search, cn_ids=cn_ids, sleep_s=sleep_s,
    )


@app.route("/control/restart_c2", methods=["POST"])
def control_restart_c2_post():
    """Local ``systemctl --user restart dsart_c2.service`` on h23.

    Form fields:

      confirm    Must literally equal ``restart_c2``.

    Returns the dict :func:`restart_c2_service_local` returns.
    """
    confirm = request.form.get("confirm", "").strip()
    if confirm != "restart_c2":
        return jsonify({
            "ok": False,
            "error": (
                "restart_c2 requires confirm=restart_c2 in the POST "
                "body — this is a deliberate safety speed bump."
            ),
        }), 400
    user = request.form.get("user") or request.remote_addr or "anon"
    try:
        result = restart_c2_service_local(control_store, user=user)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("restart_c2_service_local failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(result)


@app.route("/control/restart_h23", methods=["POST"])
def control_restart_h23_post():
    """Restart all dsa110-rt h23 ``systemctl --user`` units.

    Cycles, in order, the units in
    :data:`control_store.H23_DSART_UNITS` (dsart_c2, dsart_c3,
    hiplot_c1, hiplot_c2) via local ``systemctl --user restart``. The
    dashboard's own unit is excluded. Best-effort: one unit failing does
    not abort the rest; ``ok`` is True only if every unit restarted.

    Form fields:

      confirm    Must literally equal ``restart_h23``.

    Returns the dict :func:`restart_h23_services_local` returns.
    """
    confirm = request.form.get("confirm", "").strip()
    if confirm != "restart_h23":
        return jsonify({
            "ok": False,
            "error": (
                "restart_h23 requires confirm=restart_h23 in the POST "
                "body — this is a deliberate safety speed bump."
            ),
        }), 400
    user = request.form.get("user") or request.remote_addr or "anon"
    try:
        result = restart_h23_services_local(control_store, user=user)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("restart_h23_services_local failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(result)


@app.route("/control/c2_snapshot", methods=["GET"])
def control_c2_snapshot():
    """Read /mon/c2/h23 + return a compact JSON snapshot."""
    try:
        snap = c2_mon_snapshot(control_store)
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("c2_mon_snapshot failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(snap)


@app.route("/control/c2_journal", methods=["GET"])
def control_c2_journal():
    """Return recent decision lines from dsart_c2.service's user
    journal.

    Query params:

      ``limit``   int, 1..500 (default 50).
      ``since``   journalctl --since string (default ``"10 min ago"``).
      ``all``     ``true`` returns every line; default keeps only the
                  FIRE / WOULD-DUMP / LOG class / inject_match lines.
    """
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 500))
    since = request.args.get("since", "10 min ago").strip() or "10 min ago"
    all_lines = (request.args.get("all", "false").lower()
                 in ("1", "true", "yes"))
    try:
        snap = c2_journal_tail_local(
            limit=limit, since=since,
            grep_re=None if all_lines else None,
        ) if all_lines else c2_journal_tail_local(
            limit=limit, since=since,
        )
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("c2_journal_tail_local failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(snap)


@app.route("/control/recent_events", methods=["GET"])
def control_recent_events():
    """Return up to N most recent candidate dirs (newest first) as a
    compact JSON list for the Control-tab "Recent C2 activity" panel.

    The full per-event view lives at ``/bursts/<name>``; this endpoint
    just gives the operator an at-a-glance "did my injection land?"
    view without leaving the Control tab.
    """
    try:
        limit = int(request.args.get("limit", "10"))
    except ValueError:
        limit = 10
    limit = max(1, min(limit, 50))
    try:
        events = cands_browser.list_events()
    except Exception as exc:                                       # noqa: BLE001
        LOG.exception("cands_browser.list_events failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    out = []
    for e in events[:limit]:
        out.append({
            "name": e.name,
            "mtime_unix": e.mtime_unix,
            "mjd_peak": e.mjd_peak,
            "trigger_class": e.trigger_class,
            "n_events": e.n_events,
            "snr_max": e.snr_max,
            "dm_median": e.dm_median,
            "l_median": e.l_median,
            "m_median": e.m_median,
            "n_cubes": e.n_cubes,
            "n_plots": e.n_plots,
        })
    return jsonify({
        "ok": True,
        "archive_root": str(cands_browser.root),
        "archive_available": cands_browser.is_available,
        "events": out,
    })


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


# ---------- Control tab: delete the injection fluence/SNR K calibration ----
#
# This wipes the per-(DM, width) bootstrap calibration the operator
# builds with the "Calibrate" button — the ``K`` constants in etcd at
# ``/cnf/inject/snr_calibration/*`` that map fluence -> observed SNR.
# Clearing it lets the operator re-measure the fluence/SNR relation.
#
# NOTE: this deliberately does NOT touch the on-sky beamformer-weights
# K-cal table on the corr nodes (that footgun used to live here; the
# ``fleet_kcal`` module is retained for CLI use but is no longer wired
# to the dashboard).


@app.route("/control/delete_snr_cal", methods=["POST"])
def control_delete_snr_cal_post():
    """Delete the injection fluence/SNR (K) calibration table in etcd
    and report per-bucket pass/fail.

    Form fields:

      dry_run     "true"/"false" (default "true"). When true, only
                  lists the present (DM, width) buckets; never removes
                  anything.
      confirm     Must equal ``delete_snr_cal`` when ``dry_run=false``.
    """
    import inject_calibration as ic                                 # local

    f = request.form
    dry_run = (f.get("dry_run", "true").lower() != "false")

    if not dry_run:
        confirm = (f.get("confirm") or "").strip()
        if confirm != "delete_snr_cal":
            return jsonify({
                "ok": False,
                "error": (
                    "delete_snr_cal requires confirm=delete_snr_cal in the "
                    "POST body — this is a deliberate safety speed bump."
                ),
            }), 400

    return _control_json_or_error(
        ic.delete_snr_calibrations,
        dry_run=dry_run,
    )


@app.route("/control/clear_sidereal_vetos", methods=["POST"])
def control_clear_sidereal_vetos_post():
    """Clear the C2 sidereal (l,m) dump-veto registry.

    Writes ``/cmd/c2/sidereal_vetos_clear = {"ts", "actor"}``; the C2
    service applies it within one mon-publish tick (~5 s), wiping the
    in-memory registry and rewriting ``/cnf/c2/sidereal_vetos`` empty.

    Form fields:
      confirm   Must equal ``clear_vetos``.
    """
    import time as _time

    f = request.form
    confirm = (f.get("confirm") or "").strip()
    if confirm != "clear_vetos":
        return jsonify({
            "ok": False,
            "error": (
                "clear_sidereal_vetos requires confirm=clear_vetos in the "
                "POST body — this is a deliberate safety speed bump."
            ),
        }), 400

    ts = _time.time()
    actor = (
        request.headers.get("X-Forwarded-User")
        or request.remote_addr
        or "dashboard"
    )
    try:
        control_store.put_dict(
            "/cmd/c2/sidereal_vetos_clear",
            {"ts": ts, "actor": str(actor)},
        )
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("clear_sidereal_vetos: etcd put failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "ts": ts, "actor": str(actor)})


# ---------- Human annotations (classification tagging) ---------------------
#
# No-login human classification of burst events. The storage layer lives
# in ``annotations.py`` (pure sqlite, unit-tested). The DB is central on
# h23 OUTSIDE the repo and OUTSIDE the candidate archive (default
# ~/.dsa_monitor/annotations.db, override $DSA_MONITOR_ANNOT_DB) so tags
# survive archive pruning. Per (event, user) last-click-wins; the source
# name is event-level. Every POST returns the updated current-annotations
# block for the event so the page can re-render without a reload.


def _annot_params() -> dict:
    """Pull params from a JSON body or form-encoded POST, uniformly."""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {k: request.form.get(k) for k in request.form}


@app.route("/annotations/user", methods=["POST"])
def annotations_user_post():
    p = _annot_params()
    try:
        name = ann.add_user(p.get("name"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(
        jsonify({"ok": True, "user": name, "vocab": ann.vocab()}))


@app.route("/annotations/classify", methods=["POST"])
def annotations_classify_post():
    p = _annot_params()
    label = p.get("label")
    if isinstance(label, str) and label.strip() == "":
        label = None
    try:
        block = ann.classify(p.get("event"), p.get("user"), label)
    except ann.UnknownUserError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(jsonify({"ok": True, **block}))


@app.route("/annotations/tag", methods=["POST"])
def annotations_tag_post():
    p = _annot_params()
    try:
        tag = ann.create_tag(p.get("tag"), p.get("user"))
    except ann.UnknownUserError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # If an event was supplied, hand back its refreshed block too so the
    # "+ tag then apply" flow can render in one round-trip.
    out = {"ok": True, "tag": tag, "vocab": ann.vocab()}
    ev = (p.get("event") or "").strip() if p.get("event") else ""
    if ev:
        out["event_block"] = ann.event_annotations(ev)
    return _no_store(jsonify(out))


@app.route("/annotations/source", methods=["POST"])
def annotations_source_post():
    p = _annot_params()
    try:
        block = ann.set_source(
            p.get("event"), p.get("user"), p.get("source_name")
        )
    except ann.UnknownUserError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(jsonify({"ok": True, **block}))


@app.route("/annotations/source/purge", methods=["POST"])
def annotations_source_purge_post():
    """Purge a source name from the vocabulary everywhere (typo
    cleanup). Moves the matching source_names rows into the
    source_names_purged graveyard; every affected event's current
    source falls back to its latest surviving row."""
    p = _annot_params()
    try:
        n_purged = ann.purge_source_name(p.get("name"), p.get("user"))
    except ann.UnknownUserError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(
        jsonify({"ok": True, "n_purged": n_purged, "vocab": ann.vocab()}))


@app.route("/api/annotations", methods=["GET"])
def api_annotations():
    """Read-only annotation query surface (the future agent-query API).

    Query params: ``event``, ``user``, ``label``, ``source``, ``since``,
    ``until``, ``history`` (0/1). Default returns CURRENT annotations
    (latest per event/user + current source per event); ``history=1``
    returns the full audit trail. Timestamps are UTC ISO-8601 and the
    since/until bounds compare lexicographically against them.
    """
    def _q(name):
        v = request.args.get(name)
        return v if (v is not None and v != "") else None

    history = (request.args.get("history", "0").strip().lower()
               in ("1", "true", "yes"))
    try:
        rows = ann.query_annotations(
            event=_q("event"), user=_q("user"), label=_q("label"),
            source=_q("source"), since=_q("since"), until=_q("until"),
            history=history,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(
        jsonify({"ok": True, "history": history, "annotations": rows}))


@app.route("/api/annotations/vocab", methods=["GET"])
def api_annotations_vocab():
    return _no_store(jsonify({"ok": True, **ann.vocab()}))


# ---------- refined localizations ------------------------------------------
# Operator-entered offline/outrigger positions. Stored in the same
# annotations DB (append-only audit, active-row current value); the
# pipeline position is snapshotted server-side at entry time so the DB
# backup is self-contained.


def _pipeline_position(name: str) -> Optional[dict]:
    """Current pipeline position block for an event, or None. Reads the
    archive via the cached ArchiveBrowser astrometry (cheap)."""
    try:
        det = cands_browser.event_detail(name)
    except Exception:                                       # noqa: BLE001
        LOG.exception("pipeline position read failed for %s", name)
        return None
    if det is None or det.ra_deg is None or det.dec_deg is None:
        return None
    return {"ra_deg": det.ra_deg, "dec_deg": det.dec_deg,
            "source": det.radec_source}


@app.route("/api/position/<name>", methods=["GET"])
def api_position_get(name: str):
    if not _SAFE_EVENT_RE.match(name):
        return jsonify({"ok": False, "error": "bad event name"}), 400
    try:
        pos_row = ann.get_position(name)
        history = ann.get_position_history(name)
    except Exception as exc:                                # noqa: BLE001
        LOG.exception("get_position failed for %s", name)
        return jsonify({"ok": False, "error": str(exc)}), 500
    return _no_store(jsonify({
        "ok": True,
        "refined": _refined_display(pos_row) if pos_row else None,
        "pipeline": _pipeline_position(name),
        "history": history,
    }))


@app.route("/api/position/<name>", methods=["POST"])
def api_position_post(name: str):
    if not _SAFE_EVENT_RE.match(name):
        return jsonify({"ok": False, "error": "bad event name"}), 400
    p = _annot_params()
    try:
        ra_deg = ea.parse_ra_str(p.get("ra"))
        dec_deg = ea.parse_dec_str(p.get("dec"))
        row = ann.set_position(
            name, ra_deg, dec_deg,
            p.get("ra_err_arcsec"), p.get("dec_err_arcsec"),
            p.get("method"), p.get("user"),
            pipe_snapshot=_pipeline_position(name),
        )
    except ann.UnknownUserError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(jsonify({
        "ok": True, "refined": _refined_display(row),
    }))


@app.route("/api/position/<name>", methods=["DELETE"])
@app.route("/api/position/<name>/clear", methods=["POST"])
def api_position_clear(name: str):
    if not _SAFE_EVENT_RE.match(name):
        return jsonify({"ok": False, "error": "bad event name"}), 400
    p = _annot_params()
    try:
        had_active = ann.clear_position(name, p.get("user"))
    except ann.UnknownUserError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return _no_store(jsonify({"ok": True, "cleared": had_active}))


# ---------- API ------------------------------------------------------------


@app.route("/api/status")
def api_status():
    snap = store.snapshot()
    return _no_store(jsonify({
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
    }))


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
        "dsa_monitor up on %s:%d "
        "(sefd_state=%s sefd_results=%s)",
        bind, port, SEFD_STATE_FILE, SEFD_RESULTS_DIR,
    )
    app.run(host=bind, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
