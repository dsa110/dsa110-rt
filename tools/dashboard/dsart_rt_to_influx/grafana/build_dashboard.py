#!/usr/bin/env python3
"""Generate (and optionally POST) the dsart-rt Grafana dashboard.

This dashboard mirrors the visual conventions of the long-running
``Correlator`` dashboard on ``lxd110h20:3000`` (Grafana 6.2.5,
schemaVersion 18) and renders the InfluxDB measurements written by
``tools/dashboard/dsart_rt_to_influx/pusher.py`` -- ``corr_rt_*``,
``search_rt_*`` and ``c2_*``.

Layout (top to bottom):

  Row A. Fleet at-a-glance stats (corr/search alive count, fleet
         capture rate, time since last C2 trigger).
  Row B. Service heartbeats (corr + search alive matrix, cadence).
  Row C. Routine state (corr + search, fraction of fleet alive and
         worst verb age).
  Row D. Capture pipeline -- link rate (Gb/s) + recv pps per (cn,port).
  Row E. Capture integrity -- kernel drops, payload drops MB/s, arm
         state, degraded counts, packet-error counters (should be 0).
  Row F. Cube cadence -- n_block_writes rate per port; specnum drift.
  Row G. PSRDADA buffer health -- nfull per buffer, fill fraction,
         free_blocks, n_written rate.
  Row H. RFI summary -- total/bandpass/ant fractions and per-detector
         decomposition.
  Row I. C1 batch RX into C2 (connections_open, bytes_read,
         bad_schema/torn/bad_batch).
  Row J. C1 -> C2 candidate flow (rows_in rate, components evaluated
         rate, rows_late_drop).
  Row K. C2 triggers + plot pipeline (triggers_*, broadcast_send_*,
         pending_plots, plots_dispatched, csv_rotations).
  Row L. C2 last event panel -- name / class / age since.
  Row M. C2 housekeeping -- dumps gate, window_size, graph_size,
         gal_dm_max_los_pc_cc + poll fails.
  Row N. Inject-match -- active_count, rows_checked, matches,
         best_improved, publish_fail.

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
DASHBOARD_TITLE = "dsart-rt (corr_rt + search_rt + c2)"
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
    extra_targets: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    targets = [
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
    ]
    if extra_targets:
        for i, t in enumerate(extra_targets, start=1):
            t = dict(t)
            t.setdefault("refId", chr(ord("A") + i))
            t.setdefault("policy", "default")
            t.setdefault("resultFormat", "time_series")
            t.setdefault("orderByTime", "ASC")
            t.setdefault("rawQuery", True)
            t.setdefault("tags", [])
            t.setdefault("groupBy", [])
            t.setdefault("select", [])
            t.setdefault("measurement", "")
            targets.append(t)
    return {
        "id": _next_panel_id(),
        "type": "graph",
        "title": title,
        "datasource": DATASOURCE_NAME,
        "description": description or "",
        "gridPos": {"x": x, "y": _y(), "w": w, "h": h},
        "targets": targets,
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
        "thresholds": thresholds or [],
        "timeFrom": None,
        "timeShift": None,
    }


def singlestat_panel(
    title: str,
    raw_query: str,
    *,
    w: int = 6,
    h: int = 4,
    x: int = 0,
    unit: str = "short",
    value_name: str = "current",
    decimals: int = 0,
    sparkline_show: bool = True,
    sparkline_full: bool = False,
    color_value: bool = False,
    color_background: bool = False,
    thresholds: str = "",
    colors: Optional[List[str]] = None,
    description: Optional[str] = None,
    value_maps: Optional[List[Dict[str, Any]]] = None,
    mapping_type: int = 1,
) -> Dict[str, Any]:
    return {
        "id": _next_panel_id(),
        "type": "singlestat",
        "title": title,
        "datasource": DATASOURCE_NAME,
        "description": description or "",
        "gridPos": {"x": x, "y": _y(), "w": w, "h": h},
        "targets": [
            {
                "refId": "A",
                "alias": "",
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
        "format": unit,
        "valueName": value_name,
        "decimals": decimals,
        "sparkline": {
            "fillColor": "rgba(31, 118, 189, 0.18)",
            "full": sparkline_full,
            "lineColor": "rgb(31, 120, 193)",
            "show": sparkline_show,
            "ymin": None,
            "ymax": None,
        },
        "colorValue": color_value,
        "colorBackground": color_background,
        "thresholds": thresholds,
        "colors": colors or ["#299c46", "rgba(237, 129, 40, 0.89)", "#d44a3a"],
        "valueFontSize": "100%",
        "prefixFontSize": "50%",
        "postfixFontSize": "50%",
        "prefix": "",
        "postfix": "",
        "nullText": None,
        "nullPointMode": "connected",
        "valueMaps": value_maps or [{"op": "=", "text": "N/A", "value": "null"}],
        "mappingType": mapping_type,
        "mappingTypes": [
            {"name": "value to text", "value": 1},
            {"name": "range to text", "value": 2},
        ],
        "rangeMaps": [{"from": "null", "to": "null", "text": "N/A"}],
        "gauge": {
            "maxValue": 100,
            "minValue": 0,
            "show": False,
            "thresholdLabels": False,
            "thresholdMarkers": True,
        },
        "tableColumn": "",
        "timeFrom": None,
        "timeShift": None,
        "links": [],
    }


def row_panel(title: str, *, collapsed: bool = False) -> Dict[str, Any]:
    return {
        "id": _next_panel_id(),
        "type": "row",
        "title": title,
        "collapsed": collapsed,
        "gridPos": {"x": 0, "y": _y(), "w": 24, "h": 1},
        "panels": [],
    }


def text_panel(
    content_md: str, *, w: int = 24, h: int = 3, x: int = 0,
) -> Dict[str, Any]:
    return {
        "id": _next_panel_id(),
        "type": "text",
        "title": "",
        "datasource": None,
        "gridPos": {"x": x, "y": _y(), "w": w, "h": h},
        "mode": "markdown",
        "content": content_md,
        "links": [],
    }


def panels() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out.append(row_panel("A. Fleet at-a-glance"))
    _bump_y(1)
    out.append(singlestat_panel(
        title="Corr nodes alive (heartbeat last 30 s)",
        raw_query=(
            'SELECT count(distinct("cn_id")) FROM '
            '(SELECT last("alive") AS "alive" FROM "corr_rt_heartbeat" '
            'WHERE time > now() - 30s GROUP BY "cn_id")'
        ),
        w=6, h=4, x=0, unit="short",
        thresholds="14,15",
        colors=["#d44a3a", "rgba(237, 129, 40, 0.89)", "#299c46"],
        color_value=True, sparkline_show=False,
        description="Heartbeats in the last 30 s. Healthy target = 16; <16 means a node is dark.",
    ))
    out.append(singlestat_panel(
        title="Search nodes alive (heartbeat last 30 s)",
        raw_query=(
            'SELECT count(distinct("cn_id")) FROM '
            '(SELECT last("alive") AS "alive" FROM "search_rt_heartbeat" '
            'WHERE time > now() - 30s GROUP BY "cn_id")'
        ),
        w=6, h=4, x=6, unit="short",
        thresholds="3,4",
        colors=["#d44a3a", "rgba(237, 129, 40, 0.89)", "#299c46"],
        color_value=True, sparkline_show=False,
        description="Healthy target = 4 (cn 1, 2, 9, 13).",
    ))
    out.append(singlestat_panel(
        title="Fleet capture rate (Gb/s) -- 30s mean across 32 ports",
        raw_query=(
            'SELECT mean("rate_gbps") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND time > now() - 30s'
        ),
        w=6, h=4, x=12, unit="Gbits", decimals=2,
        description="Mean recv rate over all 32 (cn,port) capture instances. Healthy ~9.4-9.6 Gb/s.",
    ))
    out.append(singlestat_panel(
        title="Last C2 trigger MJD (24h window)",
        raw_query=(
            'SELECT last("last_event_mjd") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 24h'
        ),
        w=6, h=4, x=18, unit="none", decimals=4,
        sparkline_show=False,
        description="MJD of most recent C2 trigger over the last 24 h. Compare to current MJD to gauge staleness.",
    ))
    _bump_y(4)

    out.append(row_panel("B. Service heartbeats"))
    _bump_y(1)
    out.append(graph_panel(
        title="corr_rt heartbeat -- alive per cn",
        raw_query=(
            'SELECT last("alive") FROM "corr_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=7,
        y_min=0, y_max=1.05, unit="short",
        legend_right=True,
        description="1 = control plane saw a heartbeat in the last bucket. 0/null = stale; cadence is 2 s so gap >5 s is meaningful.",
    ))
    out.append(graph_panel(
        title="search_rt heartbeat -- alive per cn",
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
    out.append(graph_panel(
        title="corr_rt heartbeat cadence (s)",
        raw_query=(
            'SELECT last("cadence_s") FROM "corr_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=6, unit="s", y_min=0, legend_right=True,
        description="Self-reported publish cadence; nominally 2 s.",
    ))
    out.append(graph_panel(
        title="search_rt heartbeat cadence (s)",
        raw_query=(
            'SELECT last("cadence_s") FROM "search_rt_heartbeat" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=6, unit="s", y_min=0, legend_right=True,
        description="Self-reported publish cadence; nominally 2 s.",
    ))
    _bump_y(6)

    out.append(row_panel("C. Routine state"))
    _bump_y(1)
    out.append(graph_panel(
        title="corr_rt routine -- mean(alive) per routine (fleet)",
        raw_query=(
            'SELECT mean("alive") FROM "corr_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=0, h=7, y_min=0, y_max=1.05, unit="short", legend_right=True,
        description="Fraction of 16 cn reporting each routine as alive. 1.0 means whole fleet happy.",
    ))
    out.append(graph_panel(
        title="corr_rt routine -- max(last_verb_age_s) per routine",
        raw_query=(
            'SELECT max("last_verb_age_s") FROM "corr_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=12, h=7, unit="s", legend_right=True,
        description="Worst-case staleness of last verbose state-change across fleet.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="search_rt routine -- mean(alive) per routine",
        raw_query=(
            'SELECT mean("alive") FROM "search_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=0, h=7, y_min=0, y_max=1.05, legend_right=True,
        description="Fraction of 4 search nodes reporting each routine as running.",
    ))
    out.append(graph_panel(
        title="search_rt routine -- max(last_verb_age_s) per routine",
        raw_query=(
            'SELECT max("last_verb_age_s") FROM "search_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "routine" fill(null)'
        ),
        alias="$tag_routine",
        w=12, x=12, h=7, unit="s", legend_right=True,
        description="Worst-case staleness across the search_rt fleet.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="corr_rt routines alive -- count per cn (target 8)",
        raw_query=(
            'SELECT sum("alive") FROM "corr_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=6, unit="short", y_min=0, y_max=8.5, legend_right=True,
        description="Number of routines reporting alive per corr node. Healthy: 8.",
    ))
    out.append(graph_panel(
        title="search_rt routines alive -- count per cn (target 3)",
        raw_query=(
            'SELECT sum("alive") FROM "search_rt_routine" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=6, unit="short", y_min=0, y_max=3.5, legend_right=True,
        description="Number of routines reporting alive per search node. Healthy: 3.",
    ))
    _bump_y(6)
    out.append(graph_panel(
        title="C1->C2 metering active -- per search half (0=ok, 1=shedding)",
        raw_query=(
            'SELECT max("c1_metering_active") FROM "search_rt_compute" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "gpu_half" fill(null)'
        ),
        alias="cn $tag_cn_id g$tag_gpu_half",
        w=12, x=0, h=7, unit="short", y_min=0, y_max=1.1, legend_right=True,
        description=(
            "M7.6 C1->C2 metering. 1 = this search half hit the per-block "
            "candidate cap (c1.max_candidates_per_block) and shed the "
            "lowest-priority candidates (widest, then faintest) in the last "
            "16-block window to protect the C1->C2 path + C2 clustering. "
            "Sustained 1 on any half => RFI flood or cap set too low."
        ),
    ))
    out.append(graph_panel(
        title="C1->C2 metered drop -- mean cands/block shed per half",
        raw_query=(
            'SELECT mean("c1_metered_dropped_mean") FROM "search_rt_compute" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "gpu_half" fill(null)'
        ),
        alias="cn $tag_cn_id g$tag_gpu_half",
        w=12, x=12, h=7, unit="short", y_min=0, legend_right=True,
        description=(
            "Mean number of candidates dropped per block by the C1->C2 "
            "metering cap (16-block average). 0 = cap never bit. Rising "
            "values quantify how hard a half is shedding."
        ),
    ))
    _bump_y(7)

    out.append(row_panel("D. Capture pipeline (link rate + pps)"))
    _bump_y(1)
    out.append(graph_panel(
        title="Capture rate (Gb/s) per (cn,port)",
        raw_query=(
            'SELECT last("rate_gbps") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=7, unit="Gbits", y_min=0, legend_right=True,
        description="Wire-rate per UDP capture port. Healthy ~9.6 Gb/s per port.",
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
        w=12, x=12, h=7, unit="pps", y_min=0, legend_right=True,
        description="Per-port received packet rate from cumulative counter.",
    ))
    _bump_y(7)

    out.append(row_panel("E. Capture integrity (drops, arm state, errors)"))
    _bump_y(1)
    out.append(graph_panel(
        title="Kernel drop rate (pps) per (cn,port) -- target 0",
        raw_query=(
            'SELECT non_negative_derivative(last("n_dropped_kernel"), 1s) '
            'FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=7, unit="pps", y_min=0, legend_right=True,
        description="Per-port kernel-level packet drops from /proc/net/udp via n_dropped_kernel cumulative counter. Sustained non-zero = SO_RCVBUF (512 MiB) exhausted.",
    ))
    out.append(graph_panel(
        title="Dropped payload rate (MB/s) per (cn,port)",
        raw_query=(
            'SELECT last("rate_drop_mb_s") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=12, h=7, unit="MBs", y_min=0, legend_right=True,
        description="Pre-derived combined drop rate (kernel + payload + too-late). Healthy <1 MB/s; warn >10; page >50.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="Capture degraded count (per cn,port) -- stacked",
        raw_query=(
            'SELECT sum("degraded") FROM "corr_rt_capture" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(0)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=6, unit="short", y_min=0, legend_right=True, stack=True,
        description="Publishes that came in with degraded=1. Should be flat zero in normal ops.",
    ))
    out.append(graph_panel(
        title="Capture arm_state_int (per cn,port)",
        raw_query=(
            'SELECT last("arm_state_int") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=12, h=6, unit="short", y_min=0, y_max=4, legend_right=True,
        description="Integer encoding of arm_state (CTRL_RUN=2 in normal ops; 0=OFF, 1=ARMING, 3=DISARMING).",
    ))
    _bump_y(6)
    out.append(graph_panel(
        title="Cumulative n_wrong_size + n_recv_errors (per cn,port)",
        raw_query=(
            'SELECT last("n_wrong_size") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="wrong_size cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=6, unit="short", y_min=0, legend_right=True,
        extra_targets=[{
            "refId": "B",
            "alias": "recv_err cn $tag_cn_id port $tag_udp_port",
            "query": (
                'SELECT last("n_recv_errors") FROM "corr_rt_capture" '
                'WHERE "degraded" = 0 AND $timeFilter '
                'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
            ),
        }],
        description="Cumulative malformed UDP frames + recv errors since capture binary start. Both should be flat zero; growth = page.",
    ))
    out.append(graph_panel(
        title="Cumulative n_seq_skipped + n_too_late (per cn,port)",
        raw_query=(
            'SELECT last("n_seq_skipped") FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="seq_skip cn $tag_cn_id port $tag_udp_port",
        w=12, x=12, h=6, unit="short", y_min=0, legend_right=True,
        extra_targets=[{
            "refId": "B",
            "alias": "too_late cn $tag_cn_id port $tag_udp_port",
            "query": (
                'SELECT last("n_too_late") FROM "corr_rt_capture" '
                'WHERE "degraded" = 0 AND $timeFilter '
                'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
            ),
        }],
        description="Cumulative seq-no gaps and after-block-close arrivals. Small slow growth normal; sudden jumps = wire congestion.",
    ))
    _bump_y(6)

    out.append(row_panel("F. Cube cadence (block writes + specnum tracking)"))
    _bump_y(1)
    out.append(graph_panel(
        title="Block-write rate (blocks/s) per (cn,port) -- target ~7.45 Hz",
        raw_query=(
            'SELECT non_negative_derivative(last("n_block_writes"), 1s) '
            'FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval), "cn_id", "udp_port" fill(null)'
        ),
        alias="cn $tag_cn_id port $tag_udp_port",
        w=12, x=0, h=7, unit="ops", y_min=0, legend_right=True,
        thresholds=[
            {"value": 7.0, "colorMode": "critical", "fill": False, "line": True, "op": "lt"},
            {"value": 7.45, "colorMode": "ok", "fill": False, "line": True, "op": "gt"},
        ],
        description="Cumulative-derivative block-write rate per port. Production cadence 7.45 Hz; sustained <7.0 = back-pressure.",
    ))
    out.append(graph_panel(
        title="Fleet specnum spread -- max-min(last_seq_no)",
        raw_query=(
            'SELECT max("last_seq_no") - min("last_seq_no") '
            'FROM "corr_rt_capture" '
            'WHERE "degraded" = 0 AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="fleet (max-min) specnum spread",
        w=12, x=12, h=7, unit="short", y_min=0, legend_right=True,
        description="Per-bucket max-min of last_seq_no across the 32 capture instances. Healthy in steady state <~30; >300 means one node is far behind.",
    ))
    _bump_y(7)

    out.append(row_panel("G. PSRDADA buffer health"))
    _bump_y(1)
    out.append(text_panel(
        content_md=(
            "**Status: panels below depend on the `corr_rt_buffer` "
            "measurement, which is currently empty across the fleet.** "
            "Cause: corr nodes are running an old `dsart_rt.py` (pre-M7.4 "
            "Phase 7) whose `_dada_dbmetric` only tries `dada_dbmetric` on "
            "PATH (systemd-unit PATH does not include "
            "`/usr/local/bin`) and only reads stdout (the on-cluster "
            "`/usr/local/bin/dada_dbmetric` writes to **stderr**). The "
            "orchestrator therefore publishes `buffers.<k>.metric = {}` "
            "for every ring, and `pusher.make_buffer_points` correctly "
            "drops empty payloads.\n\n"
            "Fix (deploy + restart corr fleet, ~30 s downtime):\n"
            "1. dsa_monitor Control tab → **update_dsart**: dry-run "
            "preview, then apply (force=true if any corr node has dirty "
            "`/home/ubuntu/proj/dsa110-rt`).\n"
            "2. dsa_monitor Control tab → **stop fleet** (corr_too=true) "
            "then **start fleet**. This is the only way to load the new "
            "`_dada_dbmetric` into the running corr_rt orchestrators.\n"
            "3. Within ~2 s the pusher will start emitting `corr_rt_buffer` "
            "rows and these panels populate automatically (no dashboard "
            "edit needed).\n\n"
            "Reference: M7.6-MONITOR-POINTS.md §2.3; commit `c758672` "
            "(\"M7.4 Phase 7: fix dada_dbmetric stderr parsing\")."
        ),
        w=24, h=6,
    ))
    _bump_y(6)
    out.append(graph_panel(
        title="PSRDADA fill fraction -- fleet mean per buffer (nfull/nbufs)",
        raw_query=(
            'SELECT mean("nfull") / mean("nbufs") FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "buffer" fill(null)'
        ),
        alias="$tag_buffer",
        w=12, x=0, h=7, unit="percentunit", y_min=0, y_max=1, legend_right=True,
        description="Fraction of ring currently holding unread data, averaged across 16 corr nodes per buffer (dada, eada, fada, bada). Sustained >0.85 = consumer falling behind.",
    ))
    out.append(graph_panel(
        title="PSRDADA free_blocks per (cn, buffer)",
        raw_query=(
            'SELECT last("free_blocks") FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=12, h=7, unit="short", y_min=0, legend_right=True,
        description="free_blocks = nbufs - nfull. Dropping toward 0 = back-pressure alarm.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="PSRDADA nfull per cn -- fada (corr->search merge)",
        raw_query=(
            'SELECT last("nfull") FROM "corr_rt_buffer" '
            "WHERE \"buffer\" = 'fada' AND $timeFilter "
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=7, unit="short", y_min=0, legend_right=True,
        description="fada blocks currently full per corr node. Sustained near nbufs (=70) means search-side pipeline is backed up.",
    ))
    out.append(graph_panel(
        title="PSRDADA nfull per cn -- dada / eada (SNAP capture)",
        raw_query=(
            'SELECT last("nfull") FROM "corr_rt_buffer" '
            "WHERE (\"buffer\" = 'dada' OR \"buffer\" = 'eada') AND $timeFilter "
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=12, h=7, unit="short", y_min=0, legend_right=True,
        description="Per-port SNAP capture rings (20 blocks each). >10 = merge stage falling behind.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="PSRDADA n_written rate (blocks/s) per (cn, buffer)",
        raw_query=(
            'SELECT non_negative_derivative(last("n_written"), 1s) '
            'FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=0, h=7, unit="ops", y_min=0, legend_right=True,
        description="dada/eada nominal ~9.6 blocks/s, fada/bada ~7.45 blocks/s. fada below 7 = search-side gate alarm.",
    ))
    out.append(graph_panel(
        title="PSRDADA n_read rate (blocks/s) per (cn, buffer)",
        raw_query=(
            'SELECT non_negative_derivative(last("n_read"), 1s) '
            'FROM "corr_rt_buffer" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id", "buffer" fill(null)'
        ),
        alias="cn $tag_cn_id $tag_buffer",
        w=12, x=12, h=7, unit="ops", y_min=0, legend_right=True,
        description="Block-read rate (consumer side). Steady-state should match n_written; persistent gap means reader is slower than writer.",
    ))
    _bump_y(7)

    out.append(row_panel(
        "G2. Meridian fringestop (legacy UVH5 writer, casa38 -- bada reader)"
    ))
    _bump_y(1)
    out.append(graph_panel(
        title="Meridian ready nodes -- fleet sum (target 16)",
        raw_query=(
            'SELECT sum("ready") FROM '
            '(SELECT last("ready") AS "ready" FROM "corr_rt_meridian" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(0))'
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="ready cn count",
        w=12, x=0, h=7, unit="short", y_min=0, y_max=16, legend_right=True,
        thresholds=[
            {"value": 16, "colorMode": "ok", "fill": False, "line": True, "op": "ge"},
            {"value": 16, "colorMode": "warning", "fill": False, "line": True, "op": "lt"},
        ],
        description="Number of corr nodes whose meridian_fringestop wrapper reports ready=1 (sole bada reader). Should pin at 16 while observing; a drop = a node's legacy fringestopper died/wedged.",
    ))
    out.append(graph_panel(
        title="Meridian heartbeat freshness (age_s) per cn",
        raw_query=(
            'SELECT last("age_s") FROM "corr_rt_meridian" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=7, unit="s", y_min=0, legend_right=True,
        thresholds=[
            {"value": 60, "colorMode": "critical", "fill": False, "line": True, "op": "gt"},
        ],
        description="Seconds since each node's meridian_fringestop wrapper last published its heartbeat. Flat-low in normal ops; a climbing line = the routine stalled (no longer draining bada).",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="Meridian UVH5 output rate (files/s) per cn",
        raw_query=(
            'SELECT non_negative_derivative(last("n_hdf5"), 1s) '
            'FROM "corr_rt_meridian" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=0, h=7, unit="ops", y_min=0, legend_right=True,
        description="Derivative of the monotonic n_hdf5 counter -- rate at which each node writes fringestopped UVH5 files. Flat zero while observing = no output being produced.",
    ))
    out.append(graph_panel(
        title="Meridian snapped imaging declination (deg) per cn",
        raw_query=(
            'SELECT last("dec_deg") FROM "corr_rt_meridian" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=7, unit="degree", legend_right=True,
        description="Declination (snapped to the 0.25 deg fstable cache grid) each node is fringestopping to. All 16 should agree; a divergent node has a stale/mismatched fstable cache.",
    ))
    _bump_y(7)

    out.append(row_panel("H. RFI flagger summary"))
    _bump_y(1)
    out.append(graph_panel(
        title="RFI total_flag_fraction -- fleet mean per pol",
        raw_query=(
            'SELECT mean("total_flag_fraction") FROM "corr_rt_rfi" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "pol" fill(null)'
        ),
        alias="$tag_pol",
        w=12, x=0, h=7, unit="percentunit", y_min=0, y_max=1, legend_right=True,
        description="Mean fraction of (chan, time) cells flagged in the RFI rollup, averaged across the 16 cn.",
    ))
    out.append(graph_panel(
        title="RFI total_flag_fraction per cn (pol=both)",
        raw_query=(
            'SELECT mean("total_flag_fraction") FROM "corr_rt_rfi" '
            "WHERE \"pol\" = 'both' AND $timeFilter "
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=7, unit="percentunit", y_min=0, y_max=1, legend_right=True,
        description="Same metric broken out per cn (combined-pol view), to spot nodes seeing much more RFI than the fleet.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="RFI bandpass_channel_fraction -- fleet mean (per pol)",
        raw_query=(
            'SELECT mean("bandpass_channel_fraction") FROM "corr_rt_rfi" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "pol" fill(null)'
        ),
        alias="$tag_pol",
        w=12, x=0, h=6, unit="percentunit", y_min=0, y_max=1, legend_right=True,
        description="Fraction of channels flagged by the bandpass-outlier detector. Multiply by 96 for per-cn channel count after 4x freq-downsample.",
    ))
    out.append(graph_panel(
        title="RFI ant_fraction_flagged -- fleet mean (per pol)",
        raw_query=(
            'SELECT mean("ant_fraction_flagged") FROM "corr_rt_rfi" '
            'WHERE $timeFilter '
            'GROUP BY time($__interval), "pol" fill(null)'
        ),
        alias="$tag_pol",
        w=12, x=12, h=6, unit="percentunit", y_min=0, y_max=1, legend_right=True,
        description="Fraction of antennas classified as whole-antenna-bad. Multiply by 96 for per-cn count.",
    ))
    _bump_y(6)
    out.append(graph_panel(
        title="RFI per-detector breakdown -- fleet mean (pol=both)",
        raw_query=(
            'SELECT mean("frac_sk") FROM "corr_rt_rfi" '
            "WHERE \"pol\" = 'both' AND $timeFilter "
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="frac_sk",
        w=12, x=0, h=7, unit="percentunit", y_min=0, y_max=1, legend_right=True,
        extra_targets=[
            {"refId": "B", "alias": "frac_bp", "query": (
                'SELECT mean("frac_bp") FROM "corr_rt_rfi" '
                "WHERE \"pol\" = 'both' AND $timeFilter "
                'GROUP BY time($__interval) fill(null)'
            )},
            {"refId": "C", "alias": "frac_grp", "query": (
                'SELECT mean("frac_grp") FROM "corr_rt_rfi" '
                "WHERE \"pol\" = 'both' AND $timeFilter "
                'GROUP BY time($__interval) fill(null)'
            )},
            {"refId": "D", "alias": "frac_sumthr", "query": (
                'SELECT mean("frac_sumthr") FROM "corr_rt_rfi" '
                "WHERE \"pol\" = 'both' AND $timeFilter "
                'GROUP BY time($__interval) fill(null)'
            )},
            {"refId": "E", "alias": "frac_fa", "query": (
                'SELECT mean("frac_fa") FROM "corr_rt_rfi" '
                "WHERE \"pol\" = 'both' AND $timeFilter "
                'GROUP BY time($__interval) fill(null)'
            )},
        ],
        description="Per-detector flag fractions overlaid. Detectors are non-exclusive; total_flag_fraction is the OR (<= sum).",
    ))
    out.append(graph_panel(
        title="RFI publish age (s) -- per cn (pol=both)",
        raw_query=(
            'SELECT last("age_s") FROM "corr_rt_rfi" '
            "WHERE \"pol\" = 'both' AND $timeFilter "
            'GROUP BY time($__interval), "cn_id" fill(null)'
        ),
        alias="cn $tag_cn_id",
        w=12, x=12, h=7, unit="s", y_min=0, legend_right=True,
        description="age_s = publish_unix - time_unix. Healthy <=~2.2 s. Spikes mean rfi_monitor_export is behind the corr_fast shm producer.",
    ))
    _bump_y(7)

    out.append(row_panel("I. C1 batch receiver (C2 ingress on h23)"))
    _bump_y(1)
    out.append(singlestat_panel(
        title="C1 batch receiver -- connections_open",
        raw_query=(
            'SELECT last("connections_open") FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=0, unit="short",
        thresholds="7,8",
        colors=["#d44a3a", "rgba(237, 129, 40, 0.89)", "#299c46"],
        color_value=True, sparkline_show=True,
        description="TCP connections to C2 batch receiver. Expected = 1 per search-compute half = 8 in phase-B.",
    ))
    out.append(singlestat_panel(
        title="C1 batches OK (cumulative; includes heartbeats)",
        raw_query=(
            'SELECT last("batches_ok") FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=6, unit="short", decimals=0,
        description="All well-formed C1 batches decoded since C2 start, INCLUDING `n_candidates=0` heartbeats. Restart resets. In a quiet sky almost every batch is a heartbeat -- see Section J for the heartbeat vs candidate split.",
    ))
    out.append(singlestat_panel(
        title="C1 batches malformed (bad_schema + torn + bad_batch)",
        raw_query=(
            'SELECT last("bad_schema") + last("torn_batch") + last("bad_batch") '
            'FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=12, unit="short", decimals=0,
        thresholds="1,5",
        colors=["#299c46", "rgba(237, 129, 40, 0.89)", "#d44a3a"],
        color_value=True,
        description="Should be 0 in steady state. Growth = producer-side schema bug or socket corruption.",
    ))
    out.append(singlestat_panel(
        title="C1 bytes received (cumulative)",
        raw_query=(
            'SELECT last("bytes_read") FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=18, unit="bytes", decimals=2,
        description="Cumulative bytes pulled off the C1 sockets since C2 service start.",
    ))
    _bump_y(4)
    out.append(graph_panel(
        title="C1 batch rate (batches/s) into C2",
        raw_query=(
            'SELECT non_negative_derivative(last("batches_ok"), 1s) '
            'FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="batches/s",
        w=12, x=0, h=6, unit="ops", y_min=0,
        description="Rate of well-formed C1 batches into C2. Matches search-side detector cadence in normal ops.",
    ))
    out.append(graph_panel(
        title="C1 batch RX bandwidth (B/s)",
        raw_query=(
            'SELECT non_negative_derivative(last("bytes_read"), 1s) '
            'FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="bytes/s",
        w=12, x=12, h=6, unit="Bps", y_min=0,
        description="Aggregate TCP bandwidth into C2's batch receiver. Modest -- each C1 batch is a few hundred cand rows + header.",
    ))
    _bump_y(6)

    out.append(row_panel("J. C1 -> C2 candidate flow"))
    _bump_y(1)
    out.append(text_panel(
        content_md=(
            "**Read this if `rows_in/s = 0` while `batches_ok/s > 0`:** "
            "C2's `_on_batch` short-circuits on `n_candidates == 0` "
            "(empty heartbeat batches) and does NOT increment `rows_in`. "
            "Search nodes send a heartbeat every cube whether or not a "
            "cluster fires, so in a quiet sky almost every batch is a "
            "heartbeat. The overlay below puts `batches_ok/s` on top of "
            "`rows_in/s`: a flat gap between the two means \"pipeline up, "
            "no candidates this window\" (good). A `batches_ok/s` that "
            "ALSO collapses to 0 means \"no search-side production\" "
            "(check Row I connections_open, then n01 c1_emit queue)."
        ),
        w=24, h=3,
    ))
    _bump_y(3)
    out.append(graph_panel(
        title="C1 -> C2 flow -- batches_ok/s vs rows_in/s (heartbeat vs candidate)",
        raw_query=(
            'SELECT non_negative_derivative(last("batches_ok"), 1s) '
            'FROM "c2_receiver" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="batches_ok/s (heartbeats + candidates)",
        w=12, x=0, h=7, unit="ops", y_min=0, legend_right=True,
        extra_targets=[{
            "refId": "B", "alias": "rows_in/s (candidate rows only)",
            "query": (
                'SELECT non_negative_derivative(last("rows_in"), 1s) '
                'FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            ),
        }],
        description="batches_ok counts EVERY decoded C1 batch (incl. n_candidates=0 heartbeats). rows_in counts only candidate rows landed in the graph. A wide gap is normal in a quiet sky; convergence means a candidate-bearing window is in progress.",
    ))
    out.append(graph_panel(
        title="C2 components evaluated -- rate (components/s)",
        raw_query=(
            'SELECT non_negative_derivative(last("components_evaluated"), 1s) '
            'FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="components_evaluated/s",
        w=12, x=12, h=7, unit="ops", y_min=0,
        description="Rate at which C2 evaluates connected components in the cross-node coincidence graph. Each = candidate trigger tested against criteria. Flat 0 in a quiet sky is expected.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="C1 rows arriving at C2 -- rate (rows/s, candidate-only)",
        raw_query=(
            'SELECT non_negative_derivative(last("rows_in"), 1s) '
            'FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="rows_in/s",
        w=12, x=0, h=6, unit="ops", y_min=0,
        description="Rate of candidate rows (non-heartbeat) ingested by C2. Zero is the normal quiet-sky value -- compare with batches_ok/s in the overlay above.",
    ))
    out.append(graph_panel(
        title="C2 rows_late_drop -- cumulative",
        raw_query=(
            'SELECT last("rows_late_drop") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="rows_late_drop",
        w=12, x=12, h=6, unit="short", y_min=0, legend_right=False,
        description="Cumulative C1 rows dropped because they arrived after their batch window closed. Should stay flat at 0.",
    ))
    _bump_y(6)

    out.append(row_panel("K. C2 trigger pipeline"))
    _bump_y(1)
    out.append(graph_panel(
        title="C2 triggers -- cumulative by class",
        raw_query=(
            'SELECT last("triggers_dump") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="triggers_dump",
        w=12, x=0, h=7, unit="short", y_min=0, legend_right=True,
        extra_targets=[
            {"refId": "B", "alias": "triggers_log_only", "query": (
                'SELECT last("triggers_log_only") FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            )},
            {"refId": "C", "alias": "triggers_suppressed", "query": (
                'SELECT last("triggers_suppressed") FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            )},
        ],
        description="Cumulative C2 triggers by outcome: dump (issued cube-dump UDP), log_only (recorded but no dump), suppressed (dedupe / gate).",
    ))
    out.append(graph_panel(
        title="C2 trigger rate -- per hour, by class",
        raw_query=(
            'SELECT non_negative_derivative(last("triggers_dump"), 1h) '
            'FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="triggers_dump /hr",
        w=12, x=12, h=7, unit="short", y_min=0, legend_right=True,
        extra_targets=[
            {"refId": "B", "alias": "triggers_log_only /hr", "query": (
                'SELECT non_negative_derivative(last("triggers_log_only"), 1h) '
                'FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            )},
            {"refId": "C", "alias": "triggers_suppressed /hr", "query": (
                'SELECT non_negative_derivative(last("triggers_suppressed"), 1h) '
                'FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            )},
        ],
        description="Per-hour trigger rate. Useful for spotting RFI storms.",
    ))
    _bump_y(7)
    out.append(graph_panel(
        title="C2 broadcast results -- cumulative",
        raw_query=(
            'SELECT last("broadcast_send_ok") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="broadcast_send_ok",
        w=12, x=0, h=6, unit="short", y_min=0,
        extra_targets=[{
            "refId": "B", "alias": "broadcast_send_fail",
            "query": (
                'SELECT last("broadcast_send_fail") FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            ),
        }],
        description="Cumulative UDP broadcasts of C2 triggers to cube_dump subscribers. broadcast_send_fail growth = corr not receiving dump cmds.",
    ))
    out.append(graph_panel(
        title="C2 plot pipeline (pending + dispatched)",
        raw_query=(
            'SELECT last("pending_plots") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="pending_plots (gauge)",
        w=12, x=12, h=6, unit="short", y_min=0,
        extra_targets=[{
            "refId": "B", "alias": "plots_dispatched (cumulative)",
            "query": (
                'SELECT last("plots_dispatched") FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            ),
        }],
        description="pending_plots = post-trigger plot worker queue. Drops back to 0 within tens of seconds; climbing = plotter saturated.",
    ))
    _bump_y(6)

    out.append(row_panel("L. Last C2 trigger"))
    _bump_y(1)
    out.append(singlestat_panel(
        title="Last trigger -- event name",
        raw_query=(
            'SELECT last("last_event_name") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 24h'
        ),
        w=8, h=4, x=0, unit="none", decimals=0, sparkline_show=False,
        value_maps=[{"op": "=", "text": "-", "value": "null"}],
        description="Most recent C2 event name (canonical TNS-style identifier). Long stretches with no update = quiet sky.",
    ))
    out.append(singlestat_panel(
        title="Last trigger -- class",
        raw_query=(
            'SELECT last("last_trigger_class") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 24h'
        ),
        w=8, h=4, x=8, unit="none", decimals=0, sparkline_show=False,
        value_maps=[{"op": "=", "text": "-", "value": "null"}],
        description="Most recent C2 trigger class (dump / log_only / suppressed).",
    ))
    out.append(singlestat_panel(
        title="Last trigger -- MJD",
        raw_query=(
            'SELECT last("last_event_mjd") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 24h'
        ),
        w=8, h=4, x=16, unit="none", decimals=6, sparkline_show=False,
        description="MJD of the most recent C2 trigger. Match against wall-clock to gauge staleness.",
    ))
    _bump_y(4)

    out.append(row_panel("M. C2 housekeeping"))
    _bump_y(1)
    out.append(singlestat_panel(
        title="C2 dumps gate enabled",
        raw_query=(
            'SELECT last("dumps_enabled") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=4, h=4, x=0, unit="short",
        thresholds="0.5,1",
        colors=["#d44a3a", "rgba(237, 129, 40, 0.89)", "#299c46"],
        color_value=True, sparkline_show=False,
        value_maps=[
            {"op": "=", "text": "OFF", "value": "0"},
            {"op": "=", "text": "ON",  "value": "1"},
            {"op": "=", "text": "N/A", "value": "null"},
        ],
        description="Operator gate at /cnf/c2/dumps_enabled. When off, C2 still evaluates components but skips broadcasting cube_dump verbs.",
    ))
    out.append(singlestat_panel(
        title="C2 dumps gate read fails (cumulative)",
        raw_query=(
            'SELECT last("dumps_gate_fails") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=4, h=4, x=4, unit="short", decimals=0,
        thresholds="1,10",
        colors=["#299c46", "rgba(237, 129, 40, 0.89)", "#d44a3a"],
        color_value=True,
        description="Cumulative errors reading the gate from etcd. Should stay 0.",
    ))
    out.append(singlestat_panel(
        title="C2 uptime (s)",
        raw_query=(
            'SELECT last("uptime_s") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=4, h=4, x=8, unit="s", decimals=0, sparkline_show=False,
        description="Seconds since the coincidencer service entered run().",
    ))
    out.append(singlestat_panel(
        title="Galactic-DM (max LoS, pc cm^-3)",
        raw_query=(
            'SELECT last("gal_dm_max_los_pc_cc") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=4, h=4, x=12, unit="none", decimals=1,
        description="Latest line-of-sight Galactic DM cached by C2 from /mon/array/gal_dm (written by declination.service).",
    ))
    out.append(singlestat_panel(
        title="Gal-DM poll fails (cumulative)",
        raw_query=(
            'SELECT last("gal_dm_polls_fail") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=4, h=4, x=16, unit="short", decimals=0,
        thresholds="1,10",
        colors=["#299c46", "rgba(237, 129, 40, 0.89)", "#d44a3a"],
        color_value=True,
        description="Cumulative failures fetching /mon/array/gal_dm. Growth = dm_galactic_fraction discriminant falls back to nan.",
    ))
    out.append(singlestat_panel(
        title="CSV rotations (cumulative)",
        raw_query=(
            'SELECT last("csv_rotations") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=4, h=4, x=20, unit="short", decimals=0,
        description="Cumulative C1/C2 cand-CSV rotations (hourly roll).",
    ))
    _bump_y(4)
    out.append(graph_panel(
        title="C2 service queues -- window_size + graph_size + pending_plots",
        raw_query=(
            'SELECT last("window_size") FROM "c2_service" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="window_size",
        w=24, x=0, h=6, unit="short", y_min=0, legend_right=True,
        extra_targets=[
            {"refId": "B", "alias": "graph_size", "query": (
                'SELECT last("graph_size") FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            )},
            {"refId": "C", "alias": "pending_plots", "query": (
                'SELECT last("pending_plots") FROM "c2_service" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            )},
        ],
        description="Three internal queue depths overlaid. All should be modest; sustained climb = saturation.",
    ))
    _bump_y(6)

    out.append(row_panel("N. Voltage-injection match (M7.4 Phase 6c)"))
    _bump_y(1)
    out.append(singlestat_panel(
        title="Active injections",
        raw_query=(
            'SELECT last("active_count") FROM "c2_inject_match" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=0, unit="short", decimals=0, sparkline_show=True,
        description="Currently-armed voltage injections in /cnf/inject/active/. 0 in quiet ops.",
    ))
    out.append(singlestat_panel(
        title="Matches (cumulative)",
        raw_query=(
            'SELECT last("matches") FROM "c2_inject_match" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=6, unit="short", decimals=0,
        description="Cumulative C2 rows matched against active injections (DM and l/m within tolerance).",
    ))
    out.append(singlestat_panel(
        title="Best-improved (cumulative)",
        raw_query=(
            'SELECT last("best_improved") FROM "c2_inject_match" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=12, unit="short", decimals=0,
        description="Cumulative matches that bested the previous best-SNR record for their injection.",
    ))
    out.append(singlestat_panel(
        title="Publish failures",
        raw_query=(
            'SELECT last("publish_fail") FROM "c2_inject_match" '
            'WHERE "host" = \'h23\' AND time > now() - 30s'
        ),
        w=6, h=4, x=18, unit="short", decimals=0,
        thresholds="1,5",
        colors=["#299c46", "rgba(237, 129, 40, 0.89)", "#d44a3a"],
        color_value=True,
        description="Cumulative failures publishing /mon/dsart/inject/matches/* to etcd. Should stay 0.",
    ))
    _bump_y(4)
    out.append(graph_panel(
        title="Inject-match -- rows_checked rate (rows/s)",
        raw_query=(
            'SELECT non_negative_derivative(last("rows_checked"), 1s) '
            'FROM "c2_inject_match" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="rows_checked/s",
        w=12, x=0, h=6, unit="ops", y_min=0,
        description="Rate of C2 rows passed through the inject-matcher. Mirrors rows_in/s when an injection is active.",
    ))
    out.append(graph_panel(
        title="Inject-match -- evictions (cumulative)",
        raw_query=(
            'SELECT last("evicted_expired") FROM "c2_inject_match" '
            'WHERE "host" = \'h23\' AND $timeFilter '
            'GROUP BY time($__interval) fill(null)'
        ),
        alias="evicted_expired",
        w=12, x=12, h=6, unit="short", y_min=0,
        extra_targets=[{
            "refId": "B", "alias": "evict_fail",
            "query": (
                'SELECT last("evict_fail") FROM "c2_inject_match" '
                'WHERE "host" = \'h23\' AND $timeFilter '
                'GROUP BY time($__interval) fill(null)'
            ),
        }],
        description="Cumulative TTL evictions vs failed eviction attempts. Healthy: evict_fail flat at 0.",
    ))
    _bump_y(6)

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
