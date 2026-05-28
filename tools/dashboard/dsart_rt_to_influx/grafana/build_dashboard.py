#!/usr/bin/env python3
"""Generate (and optionally POST) the dsart-rt Grafana dashboard.

This dashboard mirrors the visual conventions of the long-running
``Correlator`` dashboard on ``lxd110h20:3000`` (Grafana 6.2.5,
schemaVersion 18) and renders the InfluxDB measurements written by
``tools/dashboard/dsart_rt_to_influx/pusher.py`` -- ``corr_rt_*`` and
``search_rt_*``.

Usage::

    # Just write the JSON next to this script
    ./build_dashboard.py

    # Write + POST to the live Grafana instance on h20
    ./build_dashboard.py --post \
        --grafana-url http://admin:adminLETmeIN@localhost:3000

The generator is intentionally a single self-contained module so we can
keep the dashboard JSON in version control and re-emit it deterministically.
The output ``dsart_rt_dashboard.json`` is what gets POSTed to Grafana via
``/api/dashboards/db``; it is committed alongside this script so manual
diffs are easy to read.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import urllib.request
from typing import Any, Dict, List, Optional

# Match the existing "Correlator" dashboard so the look-and-feel lines up.
DATASOURCE_NAME = "InfluxDB"
DASHBOARD_UID = "dsartRtMpV1"
DASHBOARD_TITLE = "dsart-rt (corr_rt + search_rt)"
SCHEMA_VERSION = 18
REFRESH = "5s"
GRID_W = 24


def _y() -> int:
    """Tiny generator so panel ``gridPos.y`` is implicit/auto-stacked."""
    return _y.cursor  # type: ignore[attr-defined]


_y.cursor = 0  # type: ignore[attr-defined]


def _bump_y(h: int) -> None:
    _y.cursor += h  # type: ignore[attr-defined]


_panel_id_counter = 0


def _next_panel_id() -> int:
    global _panel_id_counter
    _panel_id_counter += 1
    return _panel_id_counter


def graph_panel(
    title: str,
    raw_query: str,
    alias: str = "",
    *,
    w: int = 12,
    h: int = 7,
    x: int = 0,
    y_label: Optional[str] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    unit: str = "short",
    stack: bool = False,
    legend_right: bool = False,
    legend_values: bool = False,
    points: bool = False,
    line_width: int = 1,
    fill: int = 1,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Time-series graph panel with a raw InfluxQL query.

    Grafana 6.2.5 accepts ``rawQuery: true`` and an InfluxQL string in
    ``query``; the structured ``select`` block is not required.
    """
    return {
        "id": _next_panel_id(),
        "type": "graph",
        "title": title,
        "datasource": DATASOURCE_NAME,
        "description": description or "",
        "gridPos": {"x": x, "y": _y(), "w": w, "h": h},
        "targets": [
            {
                "refId": "A",
                "alias": alias,
                "policy": "default",
                "resultFormat": "time_series",
                "orderByTime": "ASC",
                "rawQuery": True,
                "query": raw_query,
                "tags": [],
                "groupBy": [],
                "select": [],
                "measurement": "",
            }
        ],
        "yaxes": [
            {
                "format": unit,
                "label": y_label,
                "logBase": 1,
                "min": (None if y_min is None else str(y_min)),
                "max": (None if y_max is None else str(y_max)),
                "show": True,
            },
            {
                "format": "short",
                "label": None,
                "logBase": 1,
                "min": None,
                "max": None,
                "show": True,
            },
        ],
        "xaxis": {"buckets": None, "mode": "time", "name": None, "show": True, "values": []},
        "legend": {
            "avg": False,
            "current": False,
            "max": False,
            "min": False,
            "rightSide": legend_right,
            "show": True,
            "total": False,
            "values": legend_values,
            "hideEmpty": True,
            "hideZero": False,
        },
        "lines": True,
        "linewidth": line_width,
        "points": points,
        "pointradius": 2,
        "fill": fill,
        "stack": stack,
        "nullPointMode": "null",
        "renderer": "flot",
        "tooltip": {"shared": True, "sort": 0, "value_type": "individual"},
        "thresholds": [],
        "timeFrom": None,
        "timeShift": None,
    }


# ---------------------------------------------------------------------------
# Panel definitions, top-to-bottom
# ---------------------------------------------------------------------------

def panels() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # === Row 1: fleet heartbeats ============================================
    out.append(graph_panel(
        title="corr_rt heartbeat — alive per cn",
        raw_query=(
            'SELECT last("alive") FROM "corr_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=7,
        y_min=0, y_max=1.05, unit="short",
        legend_right=True,
        description=(
            "1 = control plane saw a heartbeat for that cn in the last bucket. "
            "0/null = stale; per /mon/service/corr_rt cadence is 2 s, so any "
            "gap >5 s is meaningful."
        ),
    ))
    out.append(graph_panel(
        title="search_rt heartbeat — alive per cn",
        raw_query=(
            'SELECT last("alive") FROM "search_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=7,
        y_min=0, y_max=1.05, unit="short",
        legend_right=True,
        description="Same as corr but for the search_rt service heartbeat.",
    ))
    _bump_y(7)

    # === Row 2: corr_rt routine state =======================================
    out.append(graph_panel(
        title="corr_rt routine — mean(alive) per routine (fleet)",
        raw_query=(
            'SELECT mean("alive") FROM "corr_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=0, h=7,
        y_min=0, y_max=1.05, unit="short",
        legend_right=True,
        description=(
            "Fraction of the 16 cn that report each routine as 'running'. "
            "1.0 means the whole fleet is happy with that routine; anything "
            "<1 flags a node that is failed/dead."
        ),
    ))
    out.append(graph_panel(
        title="corr_rt routine — max(last_verb_age_s) per routine",
        raw_query=(
            'SELECT max("last_verb_age_s") FROM "corr_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=12, h=7,
        unit="s",
        legend_right=True,
        description=(
            "Worst-case staleness of last verbose state-change across the "
            "fleet for each routine. Spikes here usually mean a hung node."
        ),
    ))
    _bump_y(7)

    # === Row 3: capture pipeline ============================================
    out.append(graph_panel(
        title="Capture rate (Gb/s) per (cn,port)",
        raw_query=(
            'SELECT last("rate_gbps") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=7,
        unit="Gbits", y_min=0,
        legend_right=True,
        description=(
            "Wire-rate per UDP capture port. Healthy operating point is "
            "~9.6 Gb/s per port; sustained drops below 8 are suspicious."
        ),
    ))
    out.append(graph_panel(
        title="Recv packet rate (pps) per (cn,port)",
        raw_query=(
            'SELECT non_negative_derivative(last("n_recv_packets"), 1s) '
            'FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=12, h=7,
        unit="pps", y_min=0,
        legend_right=True,
        description=(
            "Per-port received packet rate computed from the cumulative "
            "n_recv_packets counter (pusher also publishes n_recv_packets_delta "
            "directly; this query rebuilds the rate from the raw counter so "
            "it survives publisher restarts cleanly)."
        ),
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="Kernel drop rate (pps) per (cn,port)",
        raw_query=(
            'SELECT non_negative_derivative(last("n_kernel_drop_packets"), 1s) '
            'FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=7,
        unit="pps", y_min=0,
        legend_right=True,
        description=(
            "Per-port kernel-level packet drops from /proc-style "
            "n_kernel_drop_packets cumulative counter. Any sustained "
            "non-zero value is bad."
        ),
    ))
    out.append(graph_panel(
        title="PSRDADA fill fraction — fleet mean per buffer",
        raw_query=(
            'SELECT mean("nfull") / mean("nbufs") FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "buffer" fill(null)'
        ),
        alias="$tag_buffer",
        w=12, x=12, h=7,
        unit="percentunit", y_min=0, y_max=1,
        legend_right=True,
        description=(
            "M7.4 Phase 7 buffer-health summary: fraction of the ring "
            "currently holding unread data, averaged across the 16 corr "
            "nodes per buffer (dada, eada, fada, bada). A buffer "
            "sustained > ~0.85 means the downstream consumer is "
            "falling behind (back-pressure). Source: "
            "/mon/corr_rt/<cn>.buffers.<k>.metric (filled by "
            "dada_dbmetric on the corr node)."
        ),
    ))
    _bump_y(7)

    # === Row 4: capture health flags ========================================
    out.append(graph_panel(
        title="Capture degraded count (per cn,port)",
        raw_query=(
            'SELECT sum("degraded") FROM "corr_rt_capture" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(0)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=6,
        unit="short", y_min=0,
        legend_right=True, stack=True,
        description=(
            "Number of /mon/corr_rt/<cn>/capture/<port> publishes that came "
            "in with state != UNAVAILABLE-style degraded=1 per bucket. Should "
            "be flat zero in normal ops."
        ),
    ))
    out.append(graph_panel(
        title="Capture arm_state_int (per cn,port)",
        raw_query=(
            'SELECT last("arm_state_int") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=12, h=6,
        unit="short", y_min=0, y_max=4,
        legend_right=True,
        description=(
            "Integer encoding of arm_state (CTRL_RUN=2 in normal ops; "
            "0=OFF, 1=ARMING, 3=DISARMING). Any drop off 2 means we lost "
            "the live correlator pipeline."
        ),
    ))
    _bump_y(6)

    # === Row 5: RFI =========================================================
    out.append(graph_panel(
        title="RFI total_flag_fraction — fleet mean per pol",
        raw_query=(
            'SELECT mean("total_flag_fraction") FROM "corr_rt_rfi" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "pol" fill(null)'
        ),
        alias="$tag_pol",
        w=12, x=0, h=7,
        unit="percentunit", y_min=0, y_max=1,
        legend_right=True,
        description=(
            "Mean fraction of (chan, time) cells flagged in the RFI rollup, "
            "averaged across the 16 cn. Tracks both polarisations plus the "
            "combined 'both' channel-axis OR."
        ),
    ))
    out.append(graph_panel(
        title="RFI total_flag_fraction per cn (pol=both)",
        raw_query=(
            'SELECT mean("total_flag_fraction") FROM "corr_rt_rfi" '
            "WHERE \"pol\" = 'both' AND $timeFilter "
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=7,
        unit="percentunit", y_min=0, y_max=1,
        legend_right=True,
        description=(
            "Same metric broken out per cn (combined-pol view), to spot "
            "single nodes that are seeing much more RFI than the fleet."
        ),
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="RFI frac_sk — fleet mean per pol",
        raw_query=(
            'SELECT mean("frac_sk") FROM "corr_rt_rfi" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "pol" fill(null)'
        ),
        alias="$tag_pol",
        w=12, x=0, h=7,
        unit="percentunit", y_min=0, y_max=1,
        legend_right=True,
        description=(
            "Spectral-kurtosis flag fraction per polarisation, fleet mean."
        ),
    ))
    out.append(graph_panel(
        title="RFI frac_kurt / frac_var — fleet mean per pol",
        raw_query=(
            'SELECT mean("frac_kurt") FROM "corr_rt_rfi" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "pol" fill(null)'
        ),
        alias="frac_kurt $tag_pol",
        w=12, x=12, h=7,
        unit="percentunit", y_min=0, y_max=1,
        legend_right=True,
        description=(
            "Note: this panel shows frac_kurt only; frac_var is on a sister "
            "panel below if you want to overlay them with a custom query."
        ),
    ))
    _bump_y(7)

    # === Row 6: search_rt routine state ====================================
    out.append(graph_panel(
        title="search_rt routine — mean(alive) per routine (fleet)",
        raw_query=(
            'SELECT mean("alive") FROM "search_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=0, h=7,
        y_min=0, y_max=1.05,
        legend_right=True,
        description=(
            "Fraction of the 4 search nodes that report each routine "
            "(search_rx, search_compute_0, search_compute_1) as running."
        ),
    ))
    out.append(graph_panel(
        title="search_rt routine — max(last_verb_age_s) per routine",
        raw_query=(
            'SELECT max("last_verb_age_s") FROM "search_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=12, h=7,
        unit="s",
        legend_right=True,
        description=(
            "Worst-case staleness of the last verbose state-change "
            "across the fleet for each search_rt routine."
        ),
    ))
    _bump_y(7)

    # === Row 7: service heartbeat cadences =================================
    out.append(graph_panel(
        title="corr_rt service heartbeat cadence (s)",
        raw_query=(
            'SELECT last("cadence_s") FROM "corr_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=6,
        unit="s", y_min=0,
        legend_right=True,
        description="Self-reported publish cadence; nominally 2 s.",
    ))
    out.append(graph_panel(
        title="search_rt service heartbeat cadence (s)",
        raw_query=(
            'SELECT last("cadence_s") FROM "search_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=6,
        unit="s", y_min=0,
        legend_right=True,
        description="Self-reported publish cadence; nominally 2 s.",
    ))
    _bump_y(6)

    # === Row 8: M7.4 Phase 7 — PSRDADA buffer health =======================
    # Source: /mon/corr_rt/<cn>.buffers.{dada,eada,fada,bada}.metric, filled
    # by dsart_rt._dada_dbmetric() every 2 s. The Influx pusher routes
    # numeric fields (nbufs, nfull, nclear, n_written, n_read, free_blocks)
    # onto the ``corr_rt_buffer`` measurement tagged (cn_id, host, buffer).
    out.append(graph_panel(
        title="PSRDADA nfull per (cn, buffer) — fada (corr→search merge)",
        raw_query=(
            'SELECT last("nfull") FROM "corr_rt_buffer" '
            "WHERE \"buffer\" = 'fada' AND $timeFilter "
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=7,
        unit="short", y_min=0,
        legend_right=True,
        description=(
            "Blocks currently full in the fada ring per corr node. "
            "fada is the merge output (dada + eada → fada) consumed by "
            "corr_slow + corr_fast. Sustained near nbufs (= 70) means "
            "the search-side pipeline is backed up and dropping cubes. "
            "Healthy: 0–5 blocks lingering."
        ),
    ))
    out.append(graph_panel(
        title="PSRDADA nfull per (cn, buffer) — dada / eada (SNAP capture)",
        raw_query=(
            'SELECT last("nfull") FROM "corr_rt_buffer" '
            "WHERE (\"buffer\" = 'dada' OR \"buffer\" = 'eada') AND $timeFilter "
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=12, h=7,
        unit="short", y_min=0,
        legend_right=True,
        description=(
            "Blocks currently full in dada / eada (the per-port SNAP "
            "capture rings, 20 blocks each). Sustained > ~10 means the "
            "merge stage is falling behind the kernel-side capture. A "
            "spike to 18-19 typically precedes kernel_drops_total > 0."
        ),
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="PSRDADA free_blocks per (cn, buffer) — fleet view",
        raw_query=(
            'SELECT last("free_blocks") FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=0, h=7,
        unit="short", y_min=0,
        legend_right=True,
        description=(
            "free_blocks = nbufs - nfull (the operationally useful "
            "headroom). A buffer dropping toward 0 is the back-pressure "
            "alarm — pair with the fill-fraction summary in row 3 to "
            "tell which buffer + node is choking first."
        ),
    ))
    out.append(graph_panel(
        title="PSRDADA n_written rate (blocks/s) per (cn, buffer)",
        raw_query=(
            'SELECT non_negative_derivative(last("n_written"), 1s) '
            'FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=12, h=7,
        unit="short", y_min=0,
        legend_right=True,
        description=(
            "Throughput rate of each ring (blocks/s). At the M7.4 "
            "production op-point dada/eada should run at ~9.6 blocks/s "
            "(one block per 4 specnums × 4096 native samples) and fada "
            "at ~7.45 blocks/s. A drop in fada below 7 cubes/s is the "
            "search-side gate alarm."
        ),
    ))
    _bump_y(7)

    return out


def build_dashboard() -> Dict[str, Any]:
    return {
        "annotations": {"list": []},
        "editable": True,
        "gnetId": None,
        "graphTooltip": 0,
        "id": None,
        "uid": DASHBOARD_UID,
        "title": DASHBOARD_TITLE,
        "tags": ["dsart-rt", "M7.6"],
        "schemaVersion": SCHEMA_VERSION,
        "style": "dark",
        "timezone": "",
        "version": 0,
        "refresh": REFRESH,
        "time": {"from": "now-30m", "to": "now"},
        "timepicker": {
            "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"],
            "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d"],
        },
        "templating": {"list": []},
        "links": [],
        "panels": panels(),
    }


def post_dashboard(
    dashboard: Dict[str, Any],
    grafana_url: str,
    auth: Optional[str] = None,
) -> None:
    payload = json.dumps({
        "dashboard": dashboard,
        "overwrite": True,
        "message": "auto-generated by tools/dashboard/dsart_rt_to_influx/grafana/build_dashboard.py",
    }).encode()
    headers = {"Content-Type": "application/json"}
    if auth:
        token = base64.b64encode(auth.encode()).decode()
        headers["Authorization"] = "Basic " + token
    req = urllib.request.Request(
        grafana_url.rstrip("/") + "/api/dashboards/db",
        data=payload,
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode()
    print("Grafana said:", body)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).with_name("dsart_rt_dashboard.json")),
        help="Where to write the dashboard JSON (default: alongside this script).",
    )
    ap.add_argument(
        "--post",
        action="store_true",
        help="Also POST the dashboard to Grafana via /api/dashboards/db.",
    )
    ap.add_argument(
        "--grafana-url",
        default="http://localhost:3000",
        help="Grafana base URL (default: http://localhost:3000).",
    )
    ap.add_argument(
        "--grafana-auth",
        default="admin:adminLETmeIN",
        help=(
            "Basic-auth user:pass for the Grafana admin API "
            "(default matches the existing lxd110h20 install)."
        ),
    )
    args = ap.parse_args(argv)

    dash = build_dashboard()
    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(dash, indent=2) + "\n")
    print("Wrote", out_path, "with", len(dash["panels"]), "panels")

    if args.post:
        post_dashboard(dash, args.grafana_url, args.grafana_auth)

    return 0


if __name__ == "__main__":
    sys.exit(main())
