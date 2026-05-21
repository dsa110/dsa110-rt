#!/usr/bin/env python3
"""DSA-110 dsart-rt ``/mon/*`` → InfluxDB pusher (M7.6).

Polls the etcd prefixes the dsart-rt control plane writes to and
mirrors the records into InfluxDB 1.x line protocol.  Implements the
recipe in ``docs/m7/M7.6-MONITOR-POINTS.md`` §6 + ``M7.6-MONITOR-POINTS-SEARCH.md``
§5 so that the InfluxDB measurements line up 1:1 with what Grafana
panels are designed to query.

Key shapes covered (cardinalities per the M7.5 phase-B fleet):

  - ``/mon/corr_rt/<cn>``                  → ``corr_rt_routine`` (8/cn) + ``corr_rt_buffer``
  - ``/mon/corr_rt/<cn>/capture/<port>``   → ``corr_rt_capture`` (2/cn)
  - ``/mon/corr_rt/<cn>/rfi``              → ``corr_rt_rfi`` (3/cn, per-pol fan-out)
  - ``/mon/service/corr_rt/<cn>``          → ``corr_rt_heartbeat``
  - ``/mon/search_rt/<cn>``                → ``search_rt_routine`` (3/cn)
  - ``/mon/service/search_rt/<cn>``        → ``search_rt_heartbeat``

The four ``search_rt`` per-routine keys called out as "planned" in
``M7.6-MONITOR-POINTS-SEARCH.md`` §7 (``/mon/search_rt/<cn>/rx``,
``.../compute/<half>``, ``.../cands``) are matched but currently
**raise** at routing time so the pusher fails loudly the day a
publisher lands without a corresponding pusher update.

Side-channel behaviour mandated by the spec docs:

  - **Cumulative-counter delta + ``pid``-flip reset** on every
    ``corr_rt_capture`` row (§6.3 of the corr doc).  We emit both
    the raw counter and a pre-diffed ``*_delta`` so Grafana queries
    can use either.  A ``pid`` change or a non-monotonic counter
    snapshot resets the delta to ``0`` for that tick (rather than
    producing a negative spike).
  - **Suppress synthetic zero counters on UNAVAILABLE / degraded
    placeholders** (§6.4 of the corr doc).  We emit only the tag set
    plus ``degraded=1`` so the time series stays clean.
  - **Per-key dedup on ``mod_revision``** (§6.1 of the corr doc).
    Publishers re-PUT every 2 s; a 1 s poll cadence means roughly
    half of the polls are no-ops.

The script intentionally has **no dependency on the ``dsart`` package**.
The only external dependencies are ``dsautils`` (already on every
dsa110 box, including the ``casa38`` env on h20) and ``requests`` (a
transitive dep of ``dsautils`` and ``influxdb``).  This keeps the
pusher portable across the ``casa38`` (Py 3.8) and ``dsa110-rt``
(Py 3.11) envs.

CLI::

    python -m pusher \\
        --influx-url http://localhost:8086 \\
        --influx-db dsa110 \\
        --poll-cadence-s 1.0 \\
        --log-level INFO

Production deploy: ``/etc/systemd/system/dsart_rt_to_influx.service``
calling ``/home/ubuntu/bin/startDsartRtToInflux`` on ``lxd110h20``
(co-located with ``influxd``, loopback writes).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

LOG = logging.getLogger("dsart_rt_to_influx")


# ---------------------------------------------------------------------------
# Constants — etcd key shapes + schema-version pins
# ---------------------------------------------------------------------------

# Schema versions we know how to ingest.  See docs/m7/M7.6-MONITOR-POINTS.md
# §8 ("Schema version policy") + .../SEARCH.md §8.  When a publisher bumps
# its version we want the pusher to refuse the new payload (with a clear
# log line) rather than silently drop or mis-tag fields.
SUPPORTED_SCHEMA_VERSION: int = 1

# Etcd prefixes we scan on every tick.  Any key matched by ``KEY_*``
# regexes below must live under one of these prefixes.
PREFIXES: Tuple[str, ...] = (
    "/mon/corr_rt/",
    "/mon/search_rt/",
    "/mon/service/corr_rt/",
    "/mon/service/search_rt/",
)

# Per-key-shape regexes.  Order matters: ``_route_key`` tries them in
# the listed order so the more-specific shapes (capture, rfi) match
# before the bare ``/mon/corr_rt/<cn>`` rollup.
KEY_CORR_CAPTURE = re.compile(r"^/mon/corr_rt/(\d+)/capture/(\d+)$")
KEY_CORR_RFI = re.compile(r"^/mon/corr_rt/(\d+)/rfi$")
KEY_CORR_CN = re.compile(r"^/mon/corr_rt/(\d+)$")
KEY_CORR_HEARTBEAT = re.compile(r"^/mon/service/corr_rt/(\d+)$")

KEY_SEARCH_RX = re.compile(r"^/mon/search_rt/(\d+)/rx$")
KEY_SEARCH_COMPUTE = re.compile(r"^/mon/search_rt/(\d+)/compute/(\d+)$")
KEY_SEARCH_CANDS = re.compile(r"^/mon/search_rt/(\d+)/cands$")
KEY_SEARCH_CN = re.compile(r"^/mon/search_rt/(\d+)$")
KEY_SEARCH_HEARTBEAT = re.compile(r"^/mon/service/search_rt/(\d+)$")

# Capture-key cumulative-counter fields (corr doc §3.1) for delta math.
CAPTURE_CUMULATIVE_FIELDS: Tuple[str, ...] = (
    "n_recv_packets",
    "n_recv_bytes",
    "n_dropped_payload",
    "n_dropped_kernel",
    "n_seq_skipped",
    "n_too_late",
    "n_wrong_size",
    "n_recv_errors",
    "n_block_writes",
)

# RFI-key per-pol triplet metrics (corr doc §4.2).
RFI_TRIPLET_FIELDS: Tuple[str, ...] = (
    "total_flag_fraction",
    "bandpass_channel_fraction",
    "ant_fraction_flagged",
    "frac_sk",
    "frac_bp",
    "frac_grp",
    "frac_sumthr",
    "frac_fa",
)

# RFI-key per-window envelope (corr doc §4.1).  We duplicate these on
# every per-pol row so Grafana can `WHERE pol='both'` for a clean
# single-line view without joining.
RFI_ENVELOPE_FIELDS: Tuple[str, ...] = (
    "n_cubes",
    "n_cubes_warmup",
    "age_s",
    "seq",
    "block_n_start",
    "block_n_end",
)

# Static coarse-DM owner mapping per search doc §5.2.  Baked into the
# pusher because it only changes with a configs/dsart_search_rt.yaml
# push + service restart — no point watching for it.
COARSE_DM_OWNER: Dict[Tuple[int, int], int] = {
    (1, 0): 0, (1, 1): 1,
    (2, 0): 2, (2, 1): 3,
    (9, 0): 4, (9, 1): 5,
    (13, 0): 6, (13, 1): 7,
}

# Mon-publish cadence the orchestrator uses (corr/search doc §2.5).
# Used only as a sanity bound for the polling cadence.
PUBLISH_CADENCE_S: float = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# MJD epoch is JD 2400000.5 = 1858-11-17T00:00 UTC.  Modified Julian Day
# zero is exactly 40587 days before the Unix epoch (1970-01-01).
_MJD_UNIX_OFFSET_DAYS: float = 40587.0


def _mjd_to_unix_s(mjd: float) -> float:
    """Convert an MJD timestamp (UTC) to Unix seconds.

    Mirrors the inline formula in the corr/search docs:
    ``(mjd - 40587) * 86400``.
    """
    return (float(mjd) - _MJD_UNIX_OFFSET_DAYS) * 86400.0


def _host_for_cn(cn_id: int) -> str:
    """Authoritative host stem for a corr or search node.

    Both fleets follow the ``lxd110h<NN>`` convention (see
    ``tools/dashboard/dsa_monitor/corr_topology.py`` for the
    documented mapping).  We use this only as a fallback when the
    payload doesn't carry a ``host`` field of its own (e.g. the
    lock-free heartbeat key in §5 of the corr doc).
    """
    return "lxd110h{:02d}".format(int(cn_id))


# ---------------------------------------------------------------------------
# Line protocol encoder
# ---------------------------------------------------------------------------
#
# InfluxDB 1.x line protocol: see
# https://docs.influxdata.com/influxdb/v1.7/write_protocols/line_protocol_reference/
#
# Notes on escaping (per the spec):
#   - Measurement names: escape ',' and ' '
#   - Tag keys, tag values, field keys: escape ',', '=', ' '
#   - String field values: wrap in '"', escape '"' and '\'
#   - Integer fields are suffixed with 'i'
#   - Float fields: standard decimal repr; NaN / inf are NOT representable
#     and are dropped.
#   - Booleans: 'true' / 'false'

_MEASUREMENT_TRANS = str.maketrans({",": r"\,", " ": r"\ "})
_TAG_TRANS = str.maketrans({",": r"\,", "=": r"\=", " ": r"\ "})


def _escape_measurement(s: str) -> str:
    return s.translate(_MEASUREMENT_TRANS)


def _escape_tag(s: str) -> str:
    return s.translate(_TAG_TRANS)


def _format_field_value(v: Any) -> Optional[str]:
    """Encode one field value as line-protocol.  Returns ``None`` if the
    value is unrepresentable (e.g. NaN / inf / ``None``) so the caller
    can skip it cleanly."""
    if v is None:
        return None
    # bool must come before int — bool is a subclass of int.
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return "{:d}i".format(int(v))
    if isinstance(v, float):
        # NaN / inf cannot be expressed in line protocol; drop the field
        # rather than corrupting the series.
        if v != v or v == float("inf") or v == float("-inf"):
            return None
        return repr(float(v))
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return '"{}"'.format(escaped)
    return None


@dataclass
class Point:
    """One line-protocol point.  ``tags`` keys are sorted at encode time
    (per the Influx perf recommendation); ``fields`` keys are emitted in
    insertion order so the wire output is human-grepable."""

    measurement: str
    tags: Dict[str, str]
    fields: Dict[str, Any]
    timestamp_ns: int

    def to_line(self) -> str:
        m = _escape_measurement(self.measurement)
        # Influx drops empty tag values, so we filter them out before
        # we decide whether to emit a tag section at all (otherwise a
        # ``{"k": ""}`` tag set leaves a dangling comma in the line).
        kept_tags = [
            (k, v) for k, v in sorted(self.tags.items())
            if v != ""
        ]
        if kept_tags:
            tag_str = "," + ",".join(
                "{}={}".format(_escape_tag(str(k)), _escape_tag(str(v)))
                for k, v in kept_tags
            )
        else:
            tag_str = ""
        field_parts: List[str] = []
        for k, v in self.fields.items():
            fv = _format_field_value(v)
            if fv is None:
                continue
            field_parts.append("{}={}".format(_escape_tag(str(k)), fv))
        if not field_parts:
            # A point with no fields would be silently dropped by Influx
            # (and produces an opaque "unable to parse" 400 if the body
            # has nothing after the measurement+tags).  Surface this.
            raise ValueError(
                "Point({!r}) has no representable fields".format(self.measurement)
            )
        return "{}{} {} {:d}".format(m, tag_str, ",".join(field_parts),
                                     int(self.timestamp_ns))


# ---------------------------------------------------------------------------
# InfluxDB writer
# ---------------------------------------------------------------------------


class InfluxDBLineWriter:
    """Tiny HTTP line-protocol writer for InfluxDB 1.x.

    Stateless apart from the requests session and two counters used by
    ``_tick`` log lines.  No batching across ticks: one POST per call to
    ``write()``, body = newline-joined line-protocol.  At ~250 points
    per tick (16 cn × ~12 routine/buffer/capture/rfi rows + heartbeats
    + 4 search nodes × 3 routine + heartbeat) the body is ~30 KB —
    well under the default Influx 25 MB request limit.
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:8086",
        db: str = "dsa110",
        precision: str = "ns",
        timeout_s: float = 5.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.db = db
        self.precision = precision
        self.timeout_s = float(timeout_s)
        self._session = session if session is not None else requests.Session()
        self.n_writes_ok = 0
        self.n_writes_failed = 0

    def __repr__(self) -> str:
        return "<InfluxDBLineWriter url={} db={}>".format(self.url, self.db)

    @property
    def endpoint(self) -> str:
        return "{}/write?db={}&precision={}".format(
            self.url, self.db, self.precision
        )

    def write(self, points: Iterable[Point]) -> int:
        """Encode + POST all points in one request.  Returns the number
        of lines actually written (== ``len(points)`` minus any that
        had no representable fields)."""
        lines: List[str] = []
        for p in points:
            try:
                lines.append(p.to_line())
            except ValueError as exc:
                LOG.warning("dropping malformed point: %s", exc)
        if not lines:
            return 0
        body = "\n".join(lines).encode("utf-8")
        try:
            resp = self._session.post(
                self.endpoint,
                data=body,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            LOG.error("influx POST %s failed: %s", self.endpoint, exc)
            self.n_writes_failed += 1
            return 0
        if not (200 <= resp.status_code < 300):
            LOG.error(
                "influx POST %s -> %d: %s",
                self.endpoint, resp.status_code, resp.text[:300],
            )
            self.n_writes_failed += 1
            return 0
        self.n_writes_ok += 1
        return len(lines)


# ---------------------------------------------------------------------------
# Per-key point builders
# ---------------------------------------------------------------------------
#
# Each builder takes the decoded JSON payload + the parsed cn/port and
# returns a list of Point objects.  Pure functions modulo the explicit
# ``state_table`` argument for capture deltas; that makes unit testing
# straightforward (no service object needed).


def _routine_base_tags(payload: Dict[str, Any], cn_id: int,
                       default_instance: str) -> Dict[str, str]:
    """Common tag set for routine + buffer rows.  Reads ``host`` and
    ``instance`` from the payload when present (the orchestrator always
    populates them on the rollup; see corr doc §2.1) and falls back to
    derivable values otherwise."""
    host = payload.get("host") or _host_for_cn(cn_id)
    instance = payload.get("instance") or default_instance
    state = payload.get("state", "unknown")
    return {
        "cn_id": str(int(cn_id)),
        "host": str(host),
        "instance": str(instance),
        "state": str(state),
    }


def make_routine_points(
    payload: Dict[str, Any],
    *,
    cn_id: int,
    namespace: str,
    coarse_dm_owner: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[Point]:
    """Fan ``routines`` out into one Point per routine.

    namespace ∈ {"corr_rt", "search_rt"}.  Per search doc §5.2, search
    routines named ``search_compute_0`` / ``search_compute_1`` also
    carry a ``coarse_dm`` tag derived from ``coarse_dm_owner``.
    """
    if namespace not in ("corr_rt", "search_rt"):
        raise ValueError("namespace must be corr_rt or search_rt")
    measurement = "{}_routine".format(namespace)
    default_instance = "pipeline_rt" if namespace == "corr_rt" else "search_rt"
    base_tags = _routine_base_tags(payload, cn_id, default_instance)

    routines = payload.get("routines") or {}
    if not isinstance(routines, dict) or not routines:
        return []

    time_mjd = payload.get("time_mjd")
    if time_mjd is None:
        return []
    ts_ns = int(_mjd_to_unix_s(time_mjd) * 1e9)

    # Parent-level fields duplicated to every routine row so Grafana
    # doesn't need a join to colour an alive matrix by ``state`` /
    # ``last_verb_age``.  This matches the example block in
    # M7.6-MONITOR-POINTS-SEARCH.md §5.
    uptime_s = payload.get("uptime_s")
    last_verb = payload.get("last_verb") if isinstance(
        payload.get("last_verb"), dict
    ) else None

    out: List[Point] = []
    for routine_name, info in routines.items():
        info_dict = info if isinstance(info, dict) else {}
        tags = dict(base_tags)
        tags["routine"] = str(routine_name)

        # Search-only: tag the compute-half rows with their owned
        # coarse-DM trial.  Lets queries say
        # ``WHERE measurement='search_rt_routine' AND coarse_dm='4'``.
        if namespace == "search_rt" and coarse_dm_owner is not None:
            half: Optional[int] = None
            if routine_name == "search_compute_0":
                half = 0
            elif routine_name == "search_compute_1":
                half = 1
            if half is not None:
                cd = coarse_dm_owner.get((int(cn_id), half))
                if cd is not None:
                    tags["coarse_dm"] = str(int(cd))

        # pid may be None during early bring-up; coerce to -1 so the
        # field type stays consistent (Influx 1.x infers field type
        # from the first sample per series).
        pid_raw = info_dict.get("pid")
        try:
            pid_val = int(pid_raw) if pid_raw is not None else -1
        except (TypeError, ValueError):
            pid_val = -1
        fields: Dict[str, Any] = {
            "pid": pid_val,
            "alive": 1 if info_dict.get("alive") else 0,
        }
        if isinstance(uptime_s, (int, float)) and not isinstance(uptime_s, bool):
            fields["uptime_s"] = float(uptime_s)
        if last_verb is not None:
            lv_age = last_verb.get("age_s")
            if isinstance(lv_age, (int, float)) and not isinstance(lv_age, bool):
                fields["last_verb_age_s"] = float(lv_age)
            lv_name = last_verb.get("verb")
            if isinstance(lv_name, str):
                fields["last_verb"] = lv_name

        out.append(Point(
            measurement=measurement,
            tags=tags,
            fields=fields,
            timestamp_ns=ts_ns,
        ))
    return out


def make_buffer_points(
    payload: Dict[str, Any], *, cn_id: int,
) -> List[Point]:
    """One ``corr_rt_buffer`` Point per non-empty ``buffers.<k>.metric``.

    On the live fleet (May 2026) ``metric`` is always ``{}`` because
    ``dada_dbmetric`` isn't on the orchestrator PATH (corr doc §2.3).
    Per the doc rule ("do not emit InfluxDB points for empty metrics"),
    we silently skip those.  Search-side rollups always have
    ``buffers: {}`` so this returns an empty list there too — exactly
    matching the search doc §5.1 instruction.
    """
    buffers = payload.get("buffers") or {}
    if not isinstance(buffers, dict) or not buffers:
        return []
    time_mjd = payload.get("time_mjd")
    if time_mjd is None:
        return []
    ts_ns = int(_mjd_to_unix_s(time_mjd) * 1e9)

    host = payload.get("host") or _host_for_cn(cn_id)
    out: List[Point] = []
    for buf_name, buf_info in buffers.items():
        if not isinstance(buf_info, dict):
            continue
        metric = buf_info.get("metric")
        if not isinstance(metric, dict) or not metric:
            continue
        tags = {
            "cn_id": str(int(cn_id)),
            "host": str(host),
            "buffer": str(buf_name),
        }
        fields: Dict[str, Any] = {}
        for k, v in metric.items():
            if isinstance(v, bool):
                fields[k] = 1 if v else 0
            elif isinstance(v, (int, float)):
                fields[k] = v
            elif isinstance(v, str):
                fields[k] = v
            # Skip anything else (lists, nested dicts, etc.) — none of
            # the documented ``dada_dbmetric`` keys produce them.
        if not fields:
            continue
        out.append(Point(
            measurement="corr_rt_buffer",
            tags=tags, fields=fields, timestamp_ns=ts_ns,
        ))
    return out


def make_capture_points(
    payload: Dict[str, Any],
    *,
    cn_id: int,
    udp_port: int,
    state_table: Dict[Tuple[int, int], Dict[str, Any]],
) -> List[Point]:
    """One ``corr_rt_capture`` Point per capture key.

    ``state_table`` is mutated in place; the caller (the service) owns
    it across ticks so deltas survive between calls.

    Special-cases the "UNAVAILABLE / shm missing" placeholder
    (corr doc §3.1 last block + §6.4): emits a single row with the
    tag set + ``degraded=1`` only, *not* synthetic zero counters that
    would pollute the time series.  Also resets the delta state so the
    next live snapshot starts fresh.
    """
    schema_v = payload.get("schema_version")
    if schema_v is not None and int(schema_v) != SUPPORTED_SCHEMA_VERSION:
        LOG.warning(
            "capture cn=%d port=%d: unsupported schema_version=%s "
            "(pusher supports %d); skipping",
            cn_id, udp_port, schema_v, SUPPORTED_SCHEMA_VERSION,
        )
        return []

    arm_state = str(payload.get("arm_state", "UNKNOWN"))
    host = _host_for_cn(cn_id)

    tags: Dict[str, str] = {
        "cn_id": str(int(cn_id)),
        "host": host,
        "udp_port": str(int(udp_port)),
        "arm_state": arm_state,
    }
    control_port = payload.get("control_port")
    if isinstance(control_port, (int, float)) and not isinstance(control_port, bool):
        tags["control_port"] = str(int(control_port))

    # UNAVAILABLE placeholder: emit only the tag set + degraded=1.
    # arm_state_int == -1 is the doc-mandated canonical signal.
    arm_state_int = payload.get("arm_state_int")
    is_unavailable = (
        arm_state == "UNAVAILABLE"
        or arm_state_int == -1
        or (payload.get("shm_status") is not None and payload.get("degraded"))
    )
    if is_unavailable:
        # Reset delta tracking: next live snapshot will look like a
        # fresh binary start.  Avoids spurious huge deltas on
        # capture-restart.
        state_table.pop((int(cn_id), int(udp_port)), None)
        fields: Dict[str, Any] = {
            "degraded": 1,
            "arm_state_int": -1,
        }
        for k in ("shm_status", "reason"):
            v = payload.get(k)
            if isinstance(v, str):
                fields[k] = v
        ts_ns = int(time.time() * 1e9)
        return [Point(
            measurement="corr_rt_capture",
            tags=tags, fields=fields, timestamp_ns=ts_ns,
        )]

    # Healthy snapshot: prefer the shm-side ``last_update_utc_ns`` (the
    # actual time the counters were latched) over our local wall-clock.
    last_update_utc_ns = payload.get("last_update_utc_ns")
    if isinstance(last_update_utc_ns, (int, float)) and \
            not isinstance(last_update_utc_ns, bool) and last_update_utc_ns > 0:
        ts_ns = int(last_update_utc_ns)
    else:
        ts_ns = int(time.time() * 1e9)

    fields = {"degraded": 1 if payload.get("degraded") else 0}

    # Integer-typed scalars.
    for k in (
        "pid", "arm_state_int",
        "utc_start_specnum", "utc_stop_specnum", "last_seq_no",
        "socket_rcvbuf_bytes",
        "startup_utc_ns", "last_update_utc_ns",
        "rate_kernel_drop_pps",
    ):
        v = payload.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fields[k] = int(v)

    # Float-typed scalars.
    for k in ("rate_gbps", "rate_drop_mb_s", "age_ms"):
        v = payload.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fields[k] = float(v)

    # Cumulative counters: emit raw + pre-diffed delta.  Reset on pid
    # flip (capture binary restart) or on any non-monotonic dip
    # (counter rollover / shm corruption / etc.) so deltas never go
    # negative.
    state_key = (int(cn_id), int(udp_port))
    prev = state_table.get(state_key)
    new_pid_raw = payload.get("pid")
    try:
        new_pid: Optional[int] = (
            int(new_pid_raw) if new_pid_raw is not None else None
        )
    except (TypeError, ValueError):
        new_pid = None

    reset = prev is None
    if not reset and prev is not None and new_pid is not None \
            and prev.get("pid") is not None \
            and int(prev["pid"]) != new_pid:
        LOG.info(
            "capture cn=%d port=%d: pid flip %s -> %s; resetting deltas",
            cn_id, udp_port, prev.get("pid"), new_pid,
        )
        reset = True

    snap: Dict[str, Any] = {"pid": new_pid}
    for k in CAPTURE_CUMULATIVE_FIELDS:
        v = payload.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        v_int = int(v)
        fields[k] = v_int
        if reset:
            fields[k + "_delta"] = 0
        else:
            prev_v = prev.get(k) if prev is not None else None
            if prev_v is None or v_int < int(prev_v):
                fields[k + "_delta"] = 0
            else:
                fields[k + "_delta"] = v_int - int(prev_v)
        snap[k] = v_int
    state_table[state_key] = snap

    return [Point(
        measurement="corr_rt_capture",
        tags=tags, fields=fields, timestamp_ns=ts_ns,
    )]


def make_rfi_points(
    payload: Dict[str, Any], *, cn_id: int,
) -> List[Point]:
    """Fan one RFI window into 3 Points: pol0, pol1, both.

    Each row carries the same ``cn_id, host, pol`` tag set and the
    full 8-metric per-pol slice + the per-window envelope (n_cubes,
    seq, block_n_*, age_s).  Suggested-TSDB-unfolding block in
    corr doc §4.3.
    """
    schema_v = payload.get("schema_version")
    if schema_v is not None and int(schema_v) != SUPPORTED_SCHEMA_VERSION:
        LOG.warning(
            "rfi cn=%d: unsupported schema_version=%s "
            "(pusher supports %d); skipping",
            cn_id, schema_v, SUPPORTED_SCHEMA_VERSION,
        )
        return []

    host = _host_for_cn(cn_id)

    # Degraded placeholder: corr_fast disappeared or no records yet.
    # Emit one row with pol='both' + degraded=1; skip per-pol fanout.
    if payload.get("degraded") and payload.get("shm_status"):
        ts_unix = payload.get("time_unix")
        ts_ns = int(float(ts_unix) * 1e9) if isinstance(
            ts_unix, (int, float)
        ) else int(time.time() * 1e9)
        fields_d: Dict[str, Any] = {"degraded": 1}
        for k in ("shm_status", "reason"):
            v = payload.get(k)
            if isinstance(v, str):
                fields_d[k] = v
        return [Point(
            measurement="corr_rt_rfi",
            tags={"cn_id": str(int(cn_id)), "host": host, "pol": "both"},
            fields=fields_d,
            timestamp_ns=ts_ns,
        )]

    publish_unix = payload.get("publish_unix")
    if not isinstance(publish_unix, (int, float)) or isinstance(publish_unix, bool):
        return []
    ts_ns = int(float(publish_unix) * 1e9)

    out: List[Point] = []
    for pol_label in ("pol0", "pol1", "both"):
        tags = {
            "cn_id": str(int(cn_id)),
            "host": host,
            "pol": pol_label,
        }
        fields: Dict[str, Any] = {}
        for metric in RFI_TRIPLET_FIELDS:
            triplet = payload.get(metric)
            if isinstance(triplet, dict) and pol_label in triplet:
                v = triplet[pol_label]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    fields[metric] = float(v)
        # Envelope fields are independent of pol — duplicate them.
        for k in RFI_ENVELOPE_FIELDS:
            v = payload.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if isinstance(v, int):
                fields[k] = int(v)
            else:
                fields[k] = float(v)
        fields["degraded"] = 1 if payload.get("degraded") else 0
        if len(fields) <= 1:
            # No per-pol metrics for this row (e.g. malformed payload);
            # don't emit just the ``degraded=0`` ghost.
            continue
        out.append(Point(
            measurement="corr_rt_rfi",
            tags=tags, fields=fields, timestamp_ns=ts_ns,
        ))
    return out


def make_heartbeat_points(
    payload: Dict[str, Any], *, cn_id: int, namespace: str,
) -> List[Point]:
    """Flat 3-field heartbeat → one Point.

    namespace ∈ {"corr_rt", "search_rt"}.  Per the corr doc §5 and
    search doc §3 — these heartbeats are intentionally lock-free and
    never stale beyond the cadence; the synthetic ``alive=1`` field
    is the natural InfluxDB equivalent of "this orchestrator is up".
    """
    if namespace not in ("corr_rt", "search_rt"):
        raise ValueError("namespace must be corr_rt or search_rt")
    time_mjd = payload.get("time_mjd")
    if not isinstance(time_mjd, (int, float)) or isinstance(time_mjd, bool):
        return []
    ts_ns = int(_mjd_to_unix_s(time_mjd) * 1e9)
    tags = {
        "cn_id": str(int(cn_id)),
        "host": _host_for_cn(cn_id),
        "state": str(payload.get("state", "unknown")),
    }
    fields: Dict[str, Any] = {
        "alive": 1,
        "time_mjd": float(time_mjd),
    }
    cadence = payload.get("cadence")
    if isinstance(cadence, (int, float)) and not isinstance(cadence, bool):
        fields["cadence_s"] = float(cadence)
    return [Point(
        measurement="{}_heartbeat".format(namespace),
        tags=tags, fields=fields, timestamp_ns=ts_ns,
    )]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class InfluxPusherService:
    """Long-running poll loop.

    Construction is parametrised so unit tests can pass a stub etcd
    client + ``_FakeInfluxWriter``; the CLI ``main`` wires up the real
    ``DsaStore().get_etcd()`` and a :class:`InfluxDBLineWriter`.
    """

    def __init__(
        self,
        *,
        etcd_client: Any,
        influx_writer: Any,
        poll_cadence_s: float = 1.0,
        dedupe_on_mod_revision: bool = True,
        coarse_dm_owner: Optional[Dict[Tuple[int, int], int]] = None,
        prefixes: Tuple[str, ...] = PREFIXES,
    ) -> None:
        self._etcd = etcd_client
        self._writer = influx_writer
        self.poll_cadence_s = float(poll_cadence_s)
        self.dedupe = bool(dedupe_on_mod_revision)
        self.coarse_dm_owner = (
            coarse_dm_owner if coarse_dm_owner is not None else COARSE_DM_OWNER
        )
        self._prefixes = tuple(prefixes)
        self._stop = threading.Event()

        # Per-key dedup table: key_bytes -> last seen mod_revision.
        self._last_mod_rev: Dict[bytes, int] = {}
        # Per-(cn, port) capture cumulative-counter state.
        self._capture_state: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Warn-once set for unknown subkeys.
        self._warned_unknown_keys: set = set()

        # Counters (mostly for the periodic log line; the unit tests
        # also assert on these).
        self.n_ticks = 0
        self.n_points_built = 0
        self.n_points_written = 0
        self.n_keys_seen = 0
        self.n_keys_skipped_dedupe = 0
        self.n_parse_errors = 0
        self.n_route_errors = 0
        self.n_planned_key_hits = 0

    # ---- lifecycle ----------------------------------------------------

    def install_signals(self) -> None:
        # Only the main thread can install signal handlers; unit tests
        # may spin up a service from a worker thread and rely on
        # `_stop.set()` for shutdown.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        LOG.info("received signal %d, shutting down", signum)
        self._stop.set()

    def run(self, *, max_iters: Optional[int] = None) -> int:
        self.install_signals()
        LOG.info(
            "influx_pusher up: poll=%.2fs prefixes=%s writer=%s host=%s",
            self.poll_cadence_s, list(self._prefixes),
            self._writer, socket.gethostname(),
        )
        iters = 0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception:
                # Defensive: never crash the loop on a per-tick error;
                # the next tick will retry.
                LOG.exception("tick failed (continuing)")
            iters += 1
            if max_iters is not None and iters >= max_iters:
                LOG.info("max_iters %d reached", max_iters)
                break
            elapsed = time.time() - t0
            wait_s = max(0.0, self.poll_cadence_s - elapsed)
            if self._stop.wait(wait_s):
                break
        LOG.info(
            "influx_pusher exiting (ticks=%d built=%d written=%d "
            "ok=%d failed=%d)",
            self.n_ticks, self.n_points_built, self.n_points_written,
            getattr(self._writer, "n_writes_ok", -1),
            getattr(self._writer, "n_writes_failed", -1),
        )
        return 0

    # ---- per-tick logic ----------------------------------------------

    def _tick(self) -> int:
        """One poll + write cycle.  Returns the number of lines written
        to Influx (0 if everything was deduped or if the writer
        failed)."""
        points: List[Point] = []
        for prefix in self._prefixes:
            try:
                items = list(self._etcd.get_prefix(prefix))
            except Exception as exc:
                LOG.error("etcd get_prefix %r failed: %s", prefix, exc)
                continue
            for value, meta in items:
                key_bytes = getattr(meta, "key", None)
                if not isinstance(key_bytes, (bytes, bytearray)):
                    continue
                self.n_keys_seen += 1
                # Dedup on mod_revision so the same record isn't pushed
                # twice when the publisher writes every 2 s and we poll
                # every 1 s.
                if self.dedupe:
                    mod_rev = getattr(meta, "mod_revision", None)
                    if isinstance(mod_rev, int):
                        prev_rev = self._last_mod_rev.get(bytes(key_bytes))
                        if prev_rev is not None and prev_rev == mod_rev:
                            self.n_keys_skipped_dedupe += 1
                            continue
                        self._last_mod_rev[bytes(key_bytes)] = mod_rev
                key = key_bytes.decode("utf-8", errors="replace")
                payload = self._decode_payload(value, key)
                if payload is None:
                    continue
                points.extend(self._route(key, payload))

        self.n_points_built += len(points)
        n_written = self._writer.write(points) if points else 0
        self.n_points_written += n_written
        self.n_ticks += 1
        LOG.info(
            "tick %d: keys=%d skipped=%d built=%d written=%d "
            "capture_state_size=%d",
            self.n_ticks, self.n_keys_seen, self.n_keys_skipped_dedupe,
            len(points), n_written, len(self._capture_state),
        )
        return n_written

    def _decode_payload(
        self, value: bytes, key: str,
    ) -> Optional[Dict[str, Any]]:
        if not value:
            return None
        try:
            obj = json.loads(value.decode("utf-8"))
        except Exception as exc:
            LOG.warning("could not parse %s as JSON: %s", key, exc)
            self.n_parse_errors += 1
            return None
        if not isinstance(obj, dict):
            LOG.warning("payload for %s is not a dict (got %s)",
                        key, type(obj).__name__)
            self.n_parse_errors += 1
            return None
        return obj

    def _route(self, key: str, payload: Dict[str, Any]) -> List[Point]:
        """Dispatch one key to the right ``make_*`` builder.  Wraps each
        builder in a try/except so one bad payload can't poison the rest
        of the tick."""
        try:
            m = KEY_CORR_CAPTURE.match(key)
            if m:
                return make_capture_points(
                    payload, cn_id=int(m.group(1)),
                    udp_port=int(m.group(2)),
                    state_table=self._capture_state,
                )
            m = KEY_CORR_RFI.match(key)
            if m:
                return make_rfi_points(payload, cn_id=int(m.group(1)))
            m = KEY_CORR_CN.match(key)
            if m:
                cn = int(m.group(1))
                return (
                    make_routine_points(payload, cn_id=cn,
                                        namespace="corr_rt")
                    + make_buffer_points(payload, cn_id=cn)
                )
            m = KEY_CORR_HEARTBEAT.match(key)
            if m:
                return make_heartbeat_points(
                    payload, cn_id=int(m.group(1)), namespace="corr_rt",
                )
            m = KEY_SEARCH_CN.match(key)
            if m:
                cn = int(m.group(1))
                return make_routine_points(
                    payload, cn_id=cn, namespace="search_rt",
                    coarse_dm_owner=self.coarse_dm_owner,
                ) + make_buffer_points(payload, cn_id=cn)
            m = KEY_SEARCH_HEARTBEAT.match(key)
            if m:
                return make_heartbeat_points(
                    payload, cn_id=int(m.group(1)), namespace="search_rt",
                )

            # Planned but not-yet-defined: per search doc §7.4 we want
            # to fail loudly here so a new publisher landing on the
            # fleet doesn't silently drop on the floor.  Log-once per
            # key to avoid log spam.
            if (KEY_SEARCH_RX.match(key) or KEY_SEARCH_COMPUTE.match(key)
                    or KEY_SEARCH_CANDS.match(key)):
                self.n_planned_key_hits += 1
                if key not in self._warned_unknown_keys:
                    self._warned_unknown_keys.add(key)
                    LOG.error(
                        "key %r matches a planned schema "
                        "(M7.6-MONITOR-POINTS-SEARCH.md §7) but the "
                        "pusher has no builder for it yet — "
                        "update tools/dashboard/dsart_rt_to_influx/pusher.py",
                        key,
                    )
                return []

            # Truly unknown subkey — log once.
            if key not in self._warned_unknown_keys:
                self._warned_unknown_keys.add(key)
                LOG.warning("unknown key %r under tracked prefix; ignored", key)
            return []
        except Exception:
            LOG.exception("routing %r failed", key)
            self.n_route_errors += 1
            return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_default_etcd_client() -> Any:
    """Construct the production etcd client.  Imported lazily so the
    module can be loaded for unit tests without dsautils on the path."""
    from dsautils.dsa_store import DsaStore  # noqa: WPS433
    return DsaStore().get_etcd()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--influx-url", default="http://localhost:8086",
        help="InfluxDB 1.x base URL (default: %(default)s).  Co-locating "
             "this pusher on the influxd host gives a loopback POST.",
    )
    p.add_argument(
        "--influx-db", default="dsa110",
        help="InfluxDB database name (default: %(default)s).  Must already "
             "exist (the existing etcd2db service writes to the same DB).",
    )
    p.add_argument(
        "--influx-timeout-s", type=float, default=5.0,
        help="POST timeout (default: %(default)s).  At a ~30 KB body, "
             "5 s is comfortable even for the worst slow-loopback case.",
    )
    p.add_argument(
        "--poll-cadence-s", type=float, default=1.0,
        help="etcd poll cadence (default: %(default)s).  Per M7.6 corr "
             "doc §6.1: 1 s is twice the publish rate, so dedup-on-"
             "mod_revision is the norm.",
    )
    p.add_argument(
        "--no-dedupe", action="store_true",
        help="Disable mod_revision dedup; emit every key on every poll. "
             "Useful for debugging.",
    )
    p.add_argument(
        "--max-iters", type=int, default=None,
        help="Exit after N ticks (default: run forever).  Useful for "
             "smoke tests + CI.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    etcd_client = _make_default_etcd_client()
    writer = InfluxDBLineWriter(
        url=args.influx_url,
        db=args.influx_db,
        timeout_s=args.influx_timeout_s,
    )
    service = InfluxPusherService(
        etcd_client=etcd_client,
        influx_writer=writer,
        poll_cadence_s=args.poll_cadence_s,
        dedupe_on_mod_revision=not args.no_dedupe,
    )
    return service.run(max_iters=args.max_iters)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
