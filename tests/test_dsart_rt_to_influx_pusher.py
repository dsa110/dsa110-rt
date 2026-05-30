"""Unit tests for the dsart-rt → InfluxDB pusher.

The module under test lives at
``tools/dashboard/dsart_rt_to_influx/pusher.py`` (not under
``src/dsart/`` because we follow the dashboard convention of standalone
non-package dirs — see ``tools/dashboard/dsa_monitor/`` for the
precedent).  We poke ``sys.path`` here so the test file can import
``pusher`` directly.

Coverage:

  - Line-protocol encoding (escaping, type suffixes, NaN/inf drop).
  - ``make_routine_points``: corr_rt + search_rt fan-out, coarse_dm
    tagging of search compute halves, last_verb pass-through.
  - ``make_buffer_points``: empty-metric suppression, search-rt no-op.
  - ``make_capture_points``: schema-version refusal, UNAVAILABLE
    placeholder shape, healthy snapshot delta math, pid-flip reset,
    counter-rollback reset.
  - ``make_rfi_points``: 3-row per-pol fan-out, envelope duplication,
    degraded placeholder.
  - ``make_heartbeat_points``: corr + search.
  - ``InfluxPusherService._tick``: routing of all 6 live key shapes,
    mod_revision dedup, planned-key warn-once.
  - End-to-end smoke against a captured live payload (the same JSON
    blobs the M7.6 doc cites as worked examples).
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import pytest


# Make `tools/dashboard/dsart_rt_to_influx/pusher.py` importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PUSHER_DIR = os.path.join(_REPO_ROOT, "tools", "dashboard", "dsart_rt_to_influx")
if _PUSHER_DIR not in sys.path:
    sys.path.insert(0, _PUSHER_DIR)

import pusher  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures: live-shaped payloads (lifted from the M7.6 doc worked
# examples + a snapshot from etcd on 2026-05-21).
# ---------------------------------------------------------------------------


CORR_ROLLUP_N06 = {
    "cadence": 2.0,
    "time_mjd": 61181.88562527058,
    "instance": "pipeline_rt",
    "cn": 6,
    "host": "lxd110h06",
    "state": "running",
    "uptime_s": 8197.4,
    "routines": {
        "cap_a_real":         {"pid": 33294, "alive": True},
        "cap_b_real":         {"pid": 33295, "alive": True},
        "capture_control":    {"pid": 33301, "alive": True},
        "merge":              {"pid": 33306, "alive": True},
        "corr_slow":          {"pid": 33314, "alive": True},
        "bada_drain":         {"pid": 33315, "alive": True},
        "corr_fast":          {"pid": 33316, "alive": True},
        "rfi_monitor_export": {"pid": 33317, "alive": True},
    },
    "buffers": {
        "dada": {"key": "dada", "metric": {}},
        "eada": {"key": "eada", "metric": {}},
        "fada": {"key": "fada", "metric": {}},
        "bada": {"key": "bada", "metric": {}},
    },
    "last_verb": {"verb": "utc_start", "val": 2370369966, "age_s": 7971.779},
}


CORR_CAPTURE_N06_P4011 = {
    "schema_version": 1,
    "udp_port": 4011,
    "control_port": 11223,
    "pid": 33294,
    "arm_state": "WRITING",
    "arm_state_int": 2,
    "utc_start_specnum": 2370369966,
    "utc_stop_specnum": 0,
    "last_seq_no": 2613628958,
    "socket_rcvbuf_bytes": 536870912,
    "rate_gbps": 9.656,
    "rate_drop_mb_s": 0.898,
    "rate_kernel_drop_pps": 0,
    "n_recv_packets": 1999520865,
    "n_recv_bytes": 9213792145920,
    "n_dropped_payload": 1866861,
    "n_dropped_kernel": 0,
    "n_seq_skipped": 1,
    "n_too_late": 497,
    "n_wrong_size": 0,
    "n_recv_errors": 0,
    "n_block_writes": 59388,
    "startup_utc_ns": 1779389920614557889,
    "last_update_utc_ns": 1779398118298301616,
    "age_ms": 95.73,
    "degraded": False,
}


CORR_CAPTURE_UNAVAIL_P4012 = {
    "schema_version": 1,
    "udp_port": 4012,
    "arm_state": "UNAVAILABLE",
    "arm_state_int": -1,
    "degraded": True,
    "shm_status": "missing",
    "reason": "shm not present",
}


CORR_RFI_N06 = {
    "schema_version": 1,
    "cn_id": 6,
    "time_unix": 1779394750.893,
    "publish_unix": 1779394749.542,
    "age_s": 1.351,
    "degraded": False,
    "seq": 2143,
    "block_n_start": 34273,
    "block_n_end": 34288,
    "n_cubes": 16,
    "n_cubes_warmup": 0,
    "total_flag_fraction":      {"pol0": 0.237, "pol1": 0.127, "both": 0.182},
    "bandpass_channel_fraction":{"pol0": 0.003, "pol1": 0.001, "both": 0.001},
    "ant_fraction_flagged":     {"pol0": 0.073, "pol1": 0.052, "both": 0.063},
    "frac_sk":      {"pol0": 0.198, "pol1": 0.108, "both": 0.153},
    "frac_bp":      {"pol0": 0.003, "pol1": 0.001, "both": 0.002},
    "frac_grp":     {"pol0": 0.062, "pol1": 0.052, "both": 0.057},
    "frac_sumthr":  {"pol0": 0.052, "pol1": 0.040, "both": 0.046},
    "frac_fa":      {"pol0": 0.031, "pol1": 0.031, "both": 0.031},
}


CORR_HEARTBEAT_N06 = {
    "cadence": 2.0,
    "time_mjd": 61181.88644183713,
    "state": "running",
}


SEARCH_ROLLUP_N01 = {
    "cadence": 2.0,
    "time_mjd": 61181.92571291506,
    "instance": "search_rt",
    "cn": 1,
    "host": "lxd110h01",
    "state": "running",
    "uptime_s": 11753.431,
    "routines": {
        "search_rx":        {"pid": 29689, "alive": True},
        "search_compute_0": {"pid": 29690, "alive": True},
        "search_compute_1": {"pid": 29691, "alive": True},
    },
    "buffers": {},
    "last_verb": {"verb": "start", "val": None, "age_s": 11753.441},
}


SEARCH_HEARTBEAT_N01 = {
    "cadence": 2.0,
    "time_mjd": 61181.925712797325,
    "state": "running",
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMeta:
    """Mimic the etcd3 KVMetadata struct (.key, .mod_revision)."""

    def __init__(self, key: bytes, mod_revision: int = 1) -> None:
        self.key = key
        self.mod_revision = int(mod_revision)


class _FakeEtcd:
    """In-memory etcd client.  Stores ``{key_bytes -> (value_bytes,
    mod_revision)}`` and serves ``get_prefix(prefix_str)``."""

    def __init__(self) -> None:
        self._kv: Dict[bytes, Tuple[bytes, int]] = {}

    def set(self, key: str, value, *, mod_revision: int = 1) -> None:
        if isinstance(value, (dict, list)):
            body = json.dumps(value).encode("utf-8")
        elif isinstance(value, bytes):
            body = value
        else:
            body = str(value).encode("utf-8")
        self._kv[key.encode("utf-8")] = (body, int(mod_revision))

    def bump(self, key: str, value, *, mod_revision: Optional[int] = None) -> None:
        existing = self._kv.get(key.encode("utf-8"))
        if existing is None:
            self.set(key, value, mod_revision=mod_revision or 1)
            return
        old_rev = existing[1]
        new_rev = mod_revision if mod_revision is not None else old_rev + 1
        self.set(key, value, mod_revision=new_rev)

    def get_prefix(self, prefix: str):
        pb = prefix.encode("utf-8")
        for k, (v, rev) in sorted(self._kv.items()):
            if k.startswith(pb):
                yield v, _FakeMeta(k, rev)


class _FakeInfluxWriter:
    """Captures points instead of POSTing them.  Mirrors the
    ``InfluxDBLineWriter.write`` signature."""

    def __init__(self) -> None:
        self.batches: List[List[pusher.Point]] = []
        self.n_writes_ok = 0
        self.n_writes_failed = 0

    def write(self, points) -> int:
        pts = list(points)
        self.batches.append(pts)
        self.n_writes_ok += 1
        return len(pts)

    @property
    def all_points(self) -> List[pusher.Point]:
        return [p for batch in self.batches for p in batch]

    def points_for(self, measurement: str) -> List[pusher.Point]:
        return [p for p in self.all_points if p.measurement == measurement]


# ===========================================================================
# 1. Line-protocol encoder
# ===========================================================================


class TestLineProtocol:
    def test_int_field_has_i_suffix(self):
        p = pusher.Point(
            measurement="m",
            tags={"a": "1"},
            fields={"x": 42},
            timestamp_ns=1700000000000000000,
        )
        line = p.to_line()
        assert line == "m,a=1 x=42i 1700000000000000000"

    def test_float_field_no_suffix(self):
        p = pusher.Point(
            measurement="m", tags={}, fields={"r": 3.5},
            timestamp_ns=1,
        )
        assert "r=3.5 " in p.to_line() + " "

    def test_bool_field(self):
        p = pusher.Point(
            measurement="m", tags={}, fields={"ok": True, "no": False},
            timestamp_ns=1,
        )
        line = p.to_line()
        assert "ok=true" in line and "no=false" in line

    def test_string_field_escaping(self):
        p = pusher.Point(
            measurement="m", tags={},
            fields={"s": 'a "quoted" \\backslash'},
            timestamp_ns=1,
        )
        line = p.to_line()
        # Round-trip via the same escape rules.
        assert 's="a \\"quoted\\" \\\\backslash"' in line

    def test_nan_and_inf_dropped(self):
        p = pusher.Point(
            measurement="m", tags={},
            fields={"good": 1.0, "nan": float("nan"), "inf": float("inf")},
            timestamp_ns=1,
        )
        line = p.to_line()
        assert "good=1.0" in line
        assert "nan" not in line
        assert "inf" not in line

    def test_tag_value_escaping_for_spaces_commas_equals(self):
        p = pusher.Point(
            measurement="some,measure ment",
            tags={"k 1": "v=1", "k,2": "v 2"},
            fields={"x": 1},
            timestamp_ns=1,
        )
        line = p.to_line()
        assert line.startswith("some\\,measure\\ ment,")
        assert "k\\ 1=v\\=1" in line
        assert "k\\,2=v\\ 2" in line

    def test_tags_sorted_alphabetically(self):
        p = pusher.Point(
            measurement="m",
            tags={"z": "1", "a": "2", "m": "3"},
            fields={"x": 1},
            timestamp_ns=1,
        )
        line = p.to_line()
        # Tag section is between the first comma and the space before
        # the field section.
        assert line.startswith("m,a=2,m=3,z=1 x=1i ")

    def test_empty_tag_value_skipped(self):
        # Influx 1.x treats empty tag values as "tag absent"; emit
        # accordingly so the schema stays clean.
        p = pusher.Point(
            measurement="m", tags={"k": ""}, fields={"x": 1},
            timestamp_ns=1,
        )
        line = p.to_line()
        assert line == "m x=1i 1"

    def test_point_with_no_fields_raises(self):
        p = pusher.Point(
            measurement="m", tags={}, fields={}, timestamp_ns=1,
        )
        with pytest.raises(ValueError):
            p.to_line()


# ===========================================================================
# 2. MJD conversion
# ===========================================================================


def test_mjd_to_unix_zero_offset():
    # Documented in M7.6: (mjd - 40587) * 86400.
    assert pusher._mjd_to_unix_s(40587.0) == pytest.approx(0.0)


def test_mjd_to_unix_known_value():
    # Verified against the corr-doc §2.5 worked example.  Just sanity:
    # the formula and integer math don't go off the rails.
    secs = pusher._mjd_to_unix_s(CORR_ROLLUP_N06["time_mjd"])
    assert 1.7e9 < secs < 2.0e9


def test_host_for_cn():
    assert pusher._host_for_cn(6) == "lxd110h06"
    assert pusher._host_for_cn(22) == "lxd110h22"
    assert pusher._host_for_cn(1) == "lxd110h01"


# ===========================================================================
# 3. make_routine_points — corr_rt
# ===========================================================================


class TestMakeRoutinePointsCorr:
    def test_eight_routines_eight_points(self):
        pts = pusher.make_routine_points(
            CORR_ROLLUP_N06, cn_id=6, namespace="corr_rt",
        )
        assert len(pts) == 8
        for p in pts:
            assert p.measurement == "corr_rt_routine"
            assert p.tags["cn_id"] == "6"
            assert p.tags["host"] == "lxd110h06"
            assert p.tags["instance"] == "pipeline_rt"
            assert p.tags["state"] == "running"
            assert "routine" in p.tags

    def test_alive_encoded_as_int(self):
        pts = pusher.make_routine_points(
            CORR_ROLLUP_N06, cn_id=6, namespace="corr_rt",
        )
        for p in pts:
            assert p.fields["alive"] == 1
            assert isinstance(p.fields["alive"], int)
            assert not isinstance(p.fields["alive"], bool)
            assert isinstance(p.fields["pid"], int)

    def test_dead_routine_alive_zero(self):
        payload = json.loads(json.dumps(CORR_ROLLUP_N06))
        payload["routines"]["corr_fast"]["alive"] = False
        pts = pusher.make_routine_points(
            payload, cn_id=6, namespace="corr_rt",
        )
        by_routine = {p.tags["routine"]: p for p in pts}
        assert by_routine["corr_fast"].fields["alive"] == 0
        assert by_routine["cap_a_real"].fields["alive"] == 1

    def test_last_verb_age_propagated(self):
        pts = pusher.make_routine_points(
            CORR_ROLLUP_N06, cn_id=6, namespace="corr_rt",
        )
        for p in pts:
            assert p.fields["last_verb_age_s"] == pytest.approx(7971.779)
            assert p.fields["last_verb"] == "utc_start"
            assert p.fields["uptime_s"] == pytest.approx(8197.4)

    def test_no_routines_no_points(self):
        payload = {"routines": {}, "time_mjd": 61181.0}
        assert pusher.make_routine_points(
            payload, cn_id=6, namespace="corr_rt",
        ) == []

    def test_namespace_validation(self):
        with pytest.raises(ValueError):
            pusher.make_routine_points(
                {}, cn_id=6, namespace="bogus",
            )

    def test_missing_pid_becomes_negative_one(self):
        payload = json.loads(json.dumps(CORR_ROLLUP_N06))
        payload["routines"]["corr_fast"]["pid"] = None
        pts = pusher.make_routine_points(
            payload, cn_id=6, namespace="corr_rt",
        )
        by_routine = {p.tags["routine"]: p for p in pts}
        assert by_routine["corr_fast"].fields["pid"] == -1


# ===========================================================================
# 4. make_routine_points — search_rt
# ===========================================================================


class TestMakeRoutinePointsSearch:
    def test_three_routines_three_points(self):
        pts = pusher.make_routine_points(
            SEARCH_ROLLUP_N01, cn_id=1, namespace="search_rt",
            coarse_dm_owner=pusher.COARSE_DM_OWNER,
        )
        assert len(pts) == 3
        for p in pts:
            assert p.measurement == "search_rt_routine"
            assert p.tags["instance"] == "search_rt"
            assert p.tags["cn_id"] == "1"
            assert p.tags["host"] == "lxd110h01"

    def test_coarse_dm_tag_on_compute_halves(self):
        pts = pusher.make_routine_points(
            SEARCH_ROLLUP_N01, cn_id=1, namespace="search_rt",
            coarse_dm_owner=pusher.COARSE_DM_OWNER,
        )
        by_routine = {p.tags["routine"]: p for p in pts}
        assert by_routine["search_compute_0"].tags["coarse_dm"] == "0"
        assert by_routine["search_compute_1"].tags["coarse_dm"] == "1"
        # search_rx has no coarse_dm tag.
        assert "coarse_dm" not in by_routine["search_rx"].tags

    def test_coarse_dm_for_all_search_nodes(self):
        # Per search doc §5.2 + §4.
        expected = {1: (0, 1), 2: (2, 3), 9: (4, 5), 13: (6, 7)}
        for cn, (cd0, cd1) in expected.items():
            payload = json.loads(json.dumps(SEARCH_ROLLUP_N01))
            payload["cn"] = cn
            payload["host"] = "lxd110h{:02d}".format(cn)
            pts = pusher.make_routine_points(
                payload, cn_id=cn, namespace="search_rt",
                coarse_dm_owner=pusher.COARSE_DM_OWNER,
            )
            by_routine = {p.tags["routine"]: p for p in pts}
            assert by_routine["search_compute_0"].tags["coarse_dm"] == str(cd0)
            assert by_routine["search_compute_1"].tags["coarse_dm"] == str(cd1)


# ===========================================================================
# 5. make_buffer_points
# ===========================================================================


class TestMakeBufferPoints:
    def test_empty_metric_emits_nothing(self):
        # The live fleet has metric: {} for every buffer (doc §2.3).
        # The pusher must emit zero buffer rows in that case.
        assert pusher.make_buffer_points(CORR_ROLLUP_N06, cn_id=6) == []

    def test_search_rt_empty_buffers_emits_nothing(self):
        assert pusher.make_buffer_points(SEARCH_ROLLUP_N01, cn_id=1) == []

    def test_populated_metric_emits_point(self):
        payload = json.loads(json.dumps(CORR_ROLLUP_N06))
        payload["buffers"]["dada"]["metric"] = {
            "nbufs": 70, "nfull": 12, "nclear": 58,
        }
        pts = pusher.make_buffer_points(payload, cn_id=6)
        assert len(pts) == 1
        p = pts[0]
        assert p.measurement == "corr_rt_buffer"
        assert p.tags == {"cn_id": "6", "host": "lxd110h06", "buffer": "dada"}
        assert p.fields == {"nbufs": 70, "nfull": 12, "nclear": 58}

    def test_phase7_canonical_fields_with_free_blocks_emits_point(self):
        """M7.4 Phase 7: when _dada_dbmetric() succeeds it emits the full
        canonical set (nbufs / nfull / nclear / n_written / n_read /
        free_blocks / free / full). The pusher MUST forward all numeric
        fields onto the corr_rt_buffer measurement.
        """
        payload = json.loads(json.dumps(CORR_ROLLUP_N06))
        payload["buffers"]["fada"]["metric"] = {
            "nbufs": 70, "nfull": 3, "nclear": 67,
            "n_written": 12516, "n_read": 12513,
            "free_blocks": 67, "free": 67, "full": 3,
        }
        pts = pusher.make_buffer_points(payload, cn_id=6)
        assert len(pts) == 1
        p = pts[0]
        assert p.measurement == "corr_rt_buffer"
        assert p.tags == {"cn_id": "6", "host": "lxd110h06", "buffer": "fada"}
        # All eight numeric fields land on the point.
        for k in (
            "nbufs", "nfull", "nclear", "n_written", "n_read",
            "free_blocks", "free", "full",
        ):
            assert k in p.fields, f"missing {k}"
        assert p.fields["free_blocks"] == 67

    def test_phase7_error_only_metric_emits_nothing(self):
        """M7.4 Phase 7: when ``_dada_dbmetric`` fails it returns
        ``{"_error": "<reason>"}``. The pusher MUST skip that buffer
        (no numeric fields ⇒ no Grafana time-series point), even though
        the dict is non-empty.
        """
        payload = json.loads(json.dumps(CORR_ROLLUP_N06))
        payload["buffers"]["bada"]["metric"] = {
            "_error": "FileNotFoundError(dada_dbmetric): ...",
        }
        pts = pusher.make_buffer_points(payload, cn_id=6)
        assert pts == []

    def test_phase7_mixed_buffers_only_numeric_emit_points(self):
        """Mixed payload: one buffer has good metric, one has error-only.
        Exactly one point is emitted (the good one)."""
        payload = json.loads(json.dumps(CORR_ROLLUP_N06))
        payload["buffers"]["fada"]["metric"] = {
            "nbufs": 70, "nfull": 4, "free_blocks": 66,
        }
        payload["buffers"]["bada"]["metric"] = {
            "_error": "shmget: No such file or directory",
        }
        pts = pusher.make_buffer_points(payload, cn_id=6)
        assert len(pts) == 1
        assert pts[0].tags["buffer"] == "fada"
        assert pts[0].fields["free_blocks"] == 66


# ===========================================================================
# 6. make_capture_points
# ===========================================================================


class TestMakeCapturePoints:
    def test_healthy_snapshot_schema(self):
        state = {}
        pts = pusher.make_capture_points(
            CORR_CAPTURE_N06_P4011, cn_id=6, udp_port=4011,
            state_table=state,
        )
        assert len(pts) == 1
        p = pts[0]
        assert p.measurement == "corr_rt_capture"
        # Tag set per corr doc §6.2.
        assert p.tags == {
            "cn_id": "6", "host": "lxd110h06",
            "udp_port": "4011", "control_port": "11223",
            "arm_state": "WRITING",
        }
        # pid is a field, not a tag (doc §6.2 explicit recommendation).
        assert p.fields["pid"] == 33294
        assert p.fields["arm_state_int"] == 2
        assert p.fields["last_seq_no"] == 2613628958
        assert p.fields["rate_gbps"] == pytest.approx(9.656)
        assert p.fields["rate_kernel_drop_pps"] == 0
        assert p.fields["degraded"] == 0
        # Timestamp comes from the shm-side last_update_utc_ns.
        assert p.timestamp_ns == 1779398118298301616

    def test_first_sample_zero_deltas(self):
        state = {}
        pts = pusher.make_capture_points(
            CORR_CAPTURE_N06_P4011, cn_id=6, udp_port=4011,
            state_table=state,
        )
        for field in pusher.CAPTURE_CUMULATIVE_FIELDS:
            assert pts[0].fields[field + "_delta"] == 0, field
        # State table populated.
        assert (6, 4011) in state

    def test_subsequent_sample_correct_deltas(self):
        state = {}
        pusher.make_capture_points(
            CORR_CAPTURE_N06_P4011, cn_id=6, udp_port=4011,
            state_table=state,
        )
        next_payload = json.loads(json.dumps(CORR_CAPTURE_N06_P4011))
        next_payload["n_recv_packets"] += 14_900  # ~7.45 cubes × 2000 pkts
        next_payload["n_recv_bytes"] += 14_900 * 4608
        next_payload["n_block_writes"] += 15
        next_payload["last_update_utc_ns"] += 2_000_000_000  # +2 s
        pts2 = pusher.make_capture_points(
            next_payload, cn_id=6, udp_port=4011, state_table=state,
        )
        assert pts2[0].fields["n_recv_packets_delta"] == 14_900
        assert pts2[0].fields["n_recv_bytes_delta"] == 14_900 * 4608
        assert pts2[0].fields["n_block_writes_delta"] == 15
        # Counters that didn't move stayed at delta=0.
        assert pts2[0].fields["n_recv_errors_delta"] == 0

    def test_pid_flip_resets_deltas(self):
        state = {}
        pusher.make_capture_points(
            CORR_CAPTURE_N06_P4011, cn_id=6, udp_port=4011,
            state_table=state,
        )
        restarted = json.loads(json.dumps(CORR_CAPTURE_N06_P4011))
        restarted["pid"] = 99999  # binary restarted
        # Counters reset to small values (post-restart cold start).
        restarted["n_recv_packets"] = 50
        restarted["n_recv_bytes"] = 50 * 4608
        restarted["n_block_writes"] = 1
        pts = pusher.make_capture_points(
            restarted, cn_id=6, udp_port=4011, state_table=state,
        )
        # All deltas == 0 on the reset tick; we do NOT emit a negative
        # spike from the old huge counter minus the small new one.
        for field in pusher.CAPTURE_CUMULATIVE_FIELDS:
            assert pts[0].fields[field + "_delta"] == 0, field
        # State table updated to the new pid.
        assert state[(6, 4011)]["pid"] == 99999

    def test_counter_rollback_clamps_delta_to_zero(self):
        state = {}
        pusher.make_capture_points(
            CORR_CAPTURE_N06_P4011, cn_id=6, udp_port=4011,
            state_table=state,
        )
        rollback = json.loads(json.dumps(CORR_CAPTURE_N06_P4011))
        rollback["n_recv_packets"] = 1  # impossible drop
        pts = pusher.make_capture_points(
            rollback, cn_id=6, udp_port=4011, state_table=state,
        )
        assert pts[0].fields["n_recv_packets_delta"] == 0

    def test_unavailable_placeholder_minimal(self):
        state = {(6, 4012): {"pid": 99, "n_recv_packets": 100}}
        pts = pusher.make_capture_points(
            CORR_CAPTURE_UNAVAIL_P4012, cn_id=6, udp_port=4012,
            state_table=state,
        )
        assert len(pts) == 1
        p = pts[0]
        # Tags include the UNAVAILABLE arm_state per the doc.
        assert p.tags["arm_state"] == "UNAVAILABLE"
        assert p.tags["udp_port"] == "4012"
        # Fields: degraded=1, arm_state_int=-1, shm_status, reason.
        # Crucially NO synthetic zero counters polluting the series.
        assert p.fields["degraded"] == 1
        assert p.fields["arm_state_int"] == -1
        assert p.fields["shm_status"] == "missing"
        assert p.fields["reason"] == "shm not present"
        for k in pusher.CAPTURE_CUMULATIVE_FIELDS:
            assert k not in p.fields
            assert (k + "_delta") not in p.fields
        # State table cleared for this (cn, port) so the next live
        # snapshot starts fresh.
        assert (6, 4012) not in state

    def test_unsupported_schema_version_skipped(self):
        bad = json.loads(json.dumps(CORR_CAPTURE_N06_P4011))
        bad["schema_version"] = 999
        assert pusher.make_capture_points(
            bad, cn_id=6, udp_port=4011, state_table={},
        ) == []


# ===========================================================================
# 7. make_rfi_points
# ===========================================================================


class TestMakeRfiPoints:
    def test_fans_out_three_pols(self):
        pts = pusher.make_rfi_points(CORR_RFI_N06, cn_id=6)
        assert len(pts) == 3
        pols = {p.tags["pol"] for p in pts}
        assert pols == {"pol0", "pol1", "both"}
        for p in pts:
            assert p.measurement == "corr_rt_rfi"
            assert p.tags["cn_id"] == "6"
            assert p.tags["host"] == "lxd110h06"
            # Timestamp == publish_unix (ns).
            assert p.timestamp_ns == int(1779394749.542 * 1e9)

    def test_per_pol_metric_values_match_payload(self):
        pts = pusher.make_rfi_points(CORR_RFI_N06, cn_id=6)
        by_pol = {p.tags["pol"]: p for p in pts}
        assert by_pol["pol0"].fields["total_flag_fraction"] == pytest.approx(0.237)
        assert by_pol["pol1"].fields["total_flag_fraction"] == pytest.approx(0.127)
        assert by_pol["both"].fields["total_flag_fraction"] == pytest.approx(0.182)
        # And the detector-decomposition fields are present on every row.
        for pol_label in ("pol0", "pol1", "both"):
            for metric in pusher.RFI_TRIPLET_FIELDS:
                assert metric in by_pol[pol_label].fields

    def test_envelope_duplicated_per_pol(self):
        pts = pusher.make_rfi_points(CORR_RFI_N06, cn_id=6)
        for p in pts:
            assert p.fields["seq"] == 2143
            assert p.fields["n_cubes"] == 16
            assert p.fields["n_cubes_warmup"] == 0
            assert p.fields["block_n_start"] == 34273
            assert p.fields["block_n_end"] == 34288
            assert p.fields["age_s"] == pytest.approx(1.351)
            assert p.fields["degraded"] == 0

    def test_degraded_placeholder_single_row(self):
        payload = {
            "schema_version": 1, "cn_id": 6, "degraded": True,
            "shm_status": "missing_or_empty", "reason": "shm_not_present",
            "time_unix": 1779394750.0,
        }
        pts = pusher.make_rfi_points(payload, cn_id=6)
        assert len(pts) == 1
        assert pts[0].tags["pol"] == "both"
        assert pts[0].fields["degraded"] == 1
        assert pts[0].fields["shm_status"] == "missing_or_empty"

    def test_unsupported_schema_version_skipped(self):
        bad = json.loads(json.dumps(CORR_RFI_N06))
        bad["schema_version"] = 7
        assert pusher.make_rfi_points(bad, cn_id=6) == []


# ===========================================================================
# 8. make_heartbeat_points
# ===========================================================================


class TestMakeHeartbeatPoints:
    def test_corr_heartbeat(self):
        pts = pusher.make_heartbeat_points(
            CORR_HEARTBEAT_N06, cn_id=6, namespace="corr_rt",
        )
        assert len(pts) == 1
        p = pts[0]
        assert p.measurement == "corr_rt_heartbeat"
        assert p.tags == {"cn_id": "6", "host": "lxd110h06", "state": "running"}
        assert p.fields["alive"] == 1
        assert p.fields["cadence_s"] == pytest.approx(2.0)
        assert p.fields["time_mjd"] == pytest.approx(61181.88644183713)

    def test_search_heartbeat(self):
        pts = pusher.make_heartbeat_points(
            SEARCH_HEARTBEAT_N01, cn_id=1, namespace="search_rt",
        )
        assert len(pts) == 1
        assert pts[0].measurement == "search_rt_heartbeat"
        assert pts[0].tags == {
            "cn_id": "1", "host": "lxd110h01", "state": "running",
        }

    def test_missing_time_mjd_no_point(self):
        assert pusher.make_heartbeat_points(
            {"state": "running"}, cn_id=6, namespace="corr_rt",
        ) == []


# ===========================================================================
# 9. InfluxPusherService — routing + dedup + lifecycle
# ===========================================================================


def _fully_populated_etcd() -> _FakeEtcd:
    """Seed a fake etcd with one of every live key shape."""
    e = _FakeEtcd()
    e.set("/mon/corr_rt/6", CORR_ROLLUP_N06, mod_revision=10)
    e.set("/mon/corr_rt/6/capture/4011", CORR_CAPTURE_N06_P4011, mod_revision=20)
    e.set("/mon/corr_rt/6/capture/4012", CORR_CAPTURE_UNAVAIL_P4012, mod_revision=21)
    e.set("/mon/corr_rt/6/rfi", CORR_RFI_N06, mod_revision=30)
    e.set("/mon/service/corr_rt/6", CORR_HEARTBEAT_N06, mod_revision=40)
    e.set("/mon/search_rt/1", SEARCH_ROLLUP_N01, mod_revision=50)
    e.set("/mon/service/search_rt/1", SEARCH_HEARTBEAT_N01, mod_revision=60)
    return e


class TestInfluxPusherService:
    def test_tick_routes_every_live_shape(self):
        etcd = _fully_populated_etcd()
        writer = _FakeInfluxWriter()
        svc = pusher.InfluxPusherService(
            etcd_client=etcd, influx_writer=writer,
        )
        n = svc._tick()
        assert n == len(writer.all_points)
        measurements = {p.measurement for p in writer.all_points}
        assert measurements == {
            "corr_rt_routine",
            "corr_rt_capture",
            "corr_rt_rfi",
            "corr_rt_heartbeat",
            "search_rt_routine",
            "search_rt_heartbeat",
        }
        # 8 corr routines + 2 capture (one healthy, one UNAVAIL) +
        # 3 rfi pols + 1 corr heartbeat + 3 search routines +
        # 1 search heartbeat = 18 points.
        assert n == 18

    def test_mod_revision_dedupe_second_tick_empty(self):
        etcd = _fully_populated_etcd()
        writer = _FakeInfluxWriter()
        svc = pusher.InfluxPusherService(
            etcd_client=etcd, influx_writer=writer,
        )
        svc._tick()
        # No keys changed; second tick should write 0.
        assert svc._tick() == 0
        # The skipped count grew by 7 (one per key).
        assert svc.n_keys_skipped_dedupe == 7

    def test_revision_bump_re_emits(self):
        etcd = _fully_populated_etcd()
        writer = _FakeInfluxWriter()
        svc = pusher.InfluxPusherService(
            etcd_client=etcd, influx_writer=writer,
        )
        svc._tick()
        # Republish one key with a higher mod_revision; should re-emit
        # only that key's rows (8 routine + 0 buffer for the corr-rt
        # rollup).
        etcd.bump("/mon/corr_rt/6", CORR_ROLLUP_N06, mod_revision=11)
        before = len(writer.all_points)
        svc._tick()
        after = len(writer.all_points)
        assert after - before == 8

    def test_planned_search_key_warns_once_and_emits_nothing(self, caplog):
        etcd = _FakeEtcd()
        etcd.set("/mon/search_rt/1/rx", {"cn_id": 1, "schema_version": 1},
                 mod_revision=1)
        writer = _FakeInfluxWriter()
        svc = pusher.InfluxPusherService(
            etcd_client=etcd, influx_writer=writer,
        )
        with caplog.at_level("ERROR"):
            svc._tick()
        assert writer.all_points == []
        assert svc.n_planned_key_hits == 1
        # Idempotent: a second tick (with mod_rev bumped to bypass
        # dedup) shouldn't emit a second log line.
        etcd.bump("/mon/search_rt/1/rx", {"cn_id": 1}, mod_revision=2)
        n_before = len([r for r in caplog.records
                        if "planned schema" in r.message])
        svc._tick()
        n_after = len([r for r in caplog.records
                       if "planned schema" in r.message])
        assert n_after == n_before  # warn-once

    def test_etcd_error_does_not_crash_tick(self, caplog):
        class _BrokenEtcd:
            def get_prefix(self, prefix):
                raise RuntimeError("etcd unreachable")
        svc = pusher.InfluxPusherService(
            etcd_client=_BrokenEtcd(), influx_writer=_FakeInfluxWriter(),
        )
        with caplog.at_level("ERROR"):
            svc._tick()
        assert svc.n_ticks == 1
        assert any("etcd get_prefix" in r.message for r in caplog.records)

    def test_unknown_key_under_tracked_prefix_warn_once(self, caplog):
        etcd = _FakeEtcd()
        etcd.set("/mon/corr_rt/6/totally_made_up", {"x": 1}, mod_revision=1)
        svc = pusher.InfluxPusherService(
            etcd_client=etcd, influx_writer=_FakeInfluxWriter(),
        )
        with caplog.at_level("WARNING"):
            svc._tick()
            # Bump revision so it's processed again.
            etcd.bump("/mon/corr_rt/6/totally_made_up", {"x": 2},
                      mod_revision=2)
            svc._tick()
        unknown_warnings = [r for r in caplog.records
                            if "unknown key" in r.message]
        assert len(unknown_warnings) == 1

    def test_run_with_max_iters_returns(self):
        etcd = _fully_populated_etcd()
        writer = _FakeInfluxWriter()
        svc = pusher.InfluxPusherService(
            etcd_client=etcd, influx_writer=writer,
            poll_cadence_s=0.01,
        )
        # Run in a worker thread so SIGTERM handlers aren't installed.
        t = threading.Thread(
            target=svc.run, kwargs={"max_iters": 3}, daemon=True,
        )
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
        assert svc.n_ticks == 3


# ===========================================================================
# 10. End-to-end: every emitted point round-trips through to_line()
# ===========================================================================


def test_all_emitted_lines_are_well_formed():
    """Smoke test: with the full fixture set, every Point emitted by
    the service must encode cleanly to line protocol (no
    ``to_line()`` exceptions).  Catches schema regressions like
    "empty fields dict slipped through"."""
    etcd = _fully_populated_etcd()
    writer = _FakeInfluxWriter()
    svc = pusher.InfluxPusherService(
        etcd_client=etcd, influx_writer=writer,
    )
    svc._tick()
    for p in writer.all_points:
        line = p.to_line()
        # Sanity: each line has a measurement, fields, and timestamp,
        # space-separated.
        parts = line.split(" ")
        assert len(parts) >= 2
        assert parts[-1].isdigit()


def test_capture_delta_field_appears_in_line_protocol():
    state = {}
    pusher.make_capture_points(
        CORR_CAPTURE_N06_P4011, cn_id=6, udp_port=4011,
        state_table=state,
    )
    next_payload = json.loads(json.dumps(CORR_CAPTURE_N06_P4011))
    next_payload["n_recv_packets"] += 14_900
    next_payload["last_update_utc_ns"] += 2_000_000_000
    pts = pusher.make_capture_points(
        next_payload, cn_id=6, udp_port=4011, state_table=state,
    )
    line = pts[0].to_line()
    assert "n_recv_packets_delta=14900i" in line
    assert "n_recv_packets=" in line


# ===========================================================================
# 10b. make_search_compute_points (M7.6 C1→C2 metering rollup)
# ===========================================================================


SEARCH_COMPUTE_N02_G0 = {
    "search_node_id": 2,
    "gpu_half": 0,
    "c1_metering_active": 1,
    "c1_metering_frac": 0.25,
    "c1_metered_dropped_mean": 1.25,
    "c1_metered_dropped_max": 8,
    "c1_cands_per_block_mean": 10.0,
    "c1_max_candidates_per_block": 8,
    "n_blocks": 16,
    "ts_wall_unix": 1769000000.0,
    "host": "lxd110h02",
}


def test_make_search_compute_points_fields_and_tags():
    pts = pusher.make_search_compute_points(
        SEARCH_COMPUTE_N02_G0, cn_id=2, gpu_half=0,
        coarse_dm_owner={(2, 0): 4},
    )
    assert len(pts) == 1
    p = pts[0]
    assert p.measurement == "search_rt_compute"
    assert p.tags["cn_id"] == "2"
    assert p.tags["host"] == "lxd110h02"
    assert p.tags["gpu_half"] == "0"
    assert p.tags["coarse_dm"] == "4"
    assert p.fields["c1_metering_active"] == 1
    assert p.fields["c1_metering_frac"] == 0.25
    assert p.fields["c1_metered_dropped_mean"] == 1.25
    assert p.fields["c1_max_candidates_per_block"] == 8
    assert p.timestamp_ns == int(1769000000.0 * 1e9)
    # int fields encode with an 'i' suffix; floats without.
    line = p.to_line()
    assert "c1_metering_active=1i" in line
    assert "c1_metering_frac=0.25" in line


def test_make_search_compute_points_empty_payload_drops():
    assert pusher.make_search_compute_points(
        {"search_node_id": 2, "gpu_half": 0}, cn_id=2, gpu_half=0,
    ) == []


def test_route_search_compute_key():
    svc = pusher.InfluxPusherService.__new__(pusher.InfluxPusherService)
    svc._warned_unknown_keys = set()
    svc.n_planned_key_hits = 0
    svc.n_route_errors = 0
    svc.coarse_dm_owner = {(2, 0): 4}
    pts = svc._route("/mon/search_rt/2/compute/0", SEARCH_COMPUTE_N02_G0)
    assert len(pts) == 1
    assert pts[0].measurement == "search_rt_compute"
    # The compute key must NOT fall through to the planned-key warn path.
    assert svc.n_planned_key_hits == 0


# ===========================================================================
# 11. CLI argument parsing
# ===========================================================================


def test_cli_default_args():
    args = pusher._parse_args([])
    assert args.influx_url == "http://localhost:8086"
    assert args.influx_db == "dsa110"
    assert args.poll_cadence_s == 1.0
    assert not args.no_dedupe


def test_cli_overrides():
    args = pusher._parse_args([
        "--influx-url", "http://h20:8086",
        "--influx-db", "test",
        "--poll-cadence-s", "0.5",
        "--no-dedupe",
        "--max-iters", "10",
    ])
    assert args.influx_url == "http://h20:8086"
    assert args.influx_db == "test"
    assert args.poll_cadence_s == 0.5
    assert args.no_dedupe
    assert args.max_iters == 10
