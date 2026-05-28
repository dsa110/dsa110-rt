"""Unit tests for the dsa_monitor Control tab's etcd write helpers
(M7.4 Phase 8).

These tests exercise the module by mocking out the underlying
``DsaStore`` so we don't need a live etcd to validate:

* the etcd key paths follow ``/cmd/{namespace}/{cn}`` exactly,
* the payload shape is the canonical ``{"cmd", "val"}`` dict every
  ops script writes,
* broadcasts go to ``cn=0`` and search fanouts hit each search cn
  individually,
* ``compute_arm_seq`` walks the right 32 capture mon-keys and applies
  the margin only on success,
* the audit-log row carries the ISO-8601 timestamp + user + host
  fields the Control tab promises,
* ``control_start_fleet`` / ``control_stop_fleet`` /
  ``control_utc_start_now`` / ``control_utc_stop_now`` produce the
  JSON-ready dicts the Flask layer serialises back to the operator.

The module under test lives in
``tools/dashboard/dsa_monitor/control_store.py`` (one directory off
the canonical test PYTHONPATH); we make it importable by inserting
the directory at the top of sys.path before the import.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any
from unittest import mock

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
DSA_MONITOR_DIR = os.path.normpath(os.path.join(
    HERE, "..", "tools", "dashboard", "dsa_monitor",
))
if DSA_MONITOR_DIR not in sys.path:
    sys.path.insert(0, DSA_MONITOR_DIR)

import control_store  # noqa: E402


# ---------------------------------------------------------------------------
# Fake DsaStore that records every put_dict + answers a programmable
# get_dict.
# ---------------------------------------------------------------------------


class _FakeEtcdMeta:
    """Minimal stand-in for the etcd3 KVMetadata used by get_prefix."""

    def __init__(self, key: str) -> None:
        self.key = key.encode("utf-8") if isinstance(key, str) else key


class _FakeEtcdClient:
    """Minimal etcd3 client surface needed by list_recent_audit:
    exposes get_prefix(prefix) returning [(value_bytes, meta), ...].
    """

    def __init__(self, kv: dict[str, Any]) -> None:
        # Persist all writes in the same dict so put_dict can append.
        self._kv = kv

    def get_prefix(self, prefix: str):
        import json as _json
        for k, v in sorted(self._kv.items()):
            if not k.startswith(prefix):
                continue
            if isinstance(v, dict):
                value = _json.dumps(v).encode("utf-8")
            elif isinstance(v, (bytes, bytearray)):
                value = bytes(v)
            else:
                value = str(v).encode("utf-8")
            yield value, _FakeEtcdMeta(k)


class FakeDsaStore:
    def __init__(self, get_dict_responses: dict[str, Any] | None = None):
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self._responses = dict(get_dict_responses or {})
        self._kv_for_etcd: dict[str, Any] = {}
        self._lock = threading.Lock()

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.puts.append((key, dict(payload)))
            # Mirror the write into the fake-etcd KV map so the
            # get_etcd().get_prefix() path can find audit rows.
            self._kv_for_etcd[key] = dict(payload)

    def get_dict(self, key: str) -> Any:
        with self._lock:
            self.gets.append(key)
        # ``None`` for unknown keys mirrors real DsaStore behaviour.
        return self._responses.get(key)

    def get_etcd(self):
        return _FakeEtcdClient(self._kv_for_etcd)


@pytest.fixture()
def fake_store_pair():
    """Wire a fresh ControlStore that uses a FakeDsaStore for I/O."""
    fake = FakeDsaStore()
    cs = control_store.ControlStore()
    cs._store = fake                                               # bypass DsaStore import
    return cs, fake


# ---------------------------------------------------------------------------
# send_verb / broadcast_corr / fanout_search / fanout_corr
# ---------------------------------------------------------------------------


class TestSendVerb:
    def test_send_verb_writes_canonical_key_and_payload(self, fake_store_pair):
        cs, fake = fake_store_pair
        key = control_store.send_verb(
            cs, namespace="corr_rt", cn=6, cmd="start", val=53.85,
        )
        assert key == "/cmd/corr_rt/6"
        assert fake.puts == [("/cmd/corr_rt/6", {"cmd": "start", "val": 53.85})]

    def test_send_verb_rejects_unknown_namespace(self, fake_store_pair):
        cs, _ = fake_store_pair
        with pytest.raises(ValueError, match="namespace"):
            control_store.send_verb(
                cs, namespace="bogus", cn=6, cmd="start", val=None,
            )

    def test_send_verb_accepts_none_val(self, fake_store_pair):
        cs, fake = fake_store_pair
        control_store.send_verb(
            cs, namespace="search_rt", cn=1, cmd="stop", val=None,
        )
        assert fake.puts == [("/cmd/search_rt/1", {"cmd": "stop", "val": None})]


class TestBroadcastCorr:
    def test_broadcast_corr_goes_to_cn_zero(self, fake_store_pair):
        cs, fake = fake_store_pair
        key = control_store.broadcast_corr(cs, cmd="utc_start", val=12345)
        assert key == "/cmd/corr_rt/0"
        assert fake.puts == [
            ("/cmd/corr_rt/0", {"cmd": "utc_start", "val": 12345})
        ]


class TestFanoutSearch:
    def test_fanout_search_default_hits_all_four(self, fake_store_pair):
        cs, fake = fake_store_pair
        keys = control_store.fanout_search(cs, cmd="start", val=None)
        assert keys == [
            "/cmd/search_rt/1",
            "/cmd/search_rt/2",
            "/cmd/search_rt/9",
            "/cmd/search_rt/13",
        ]
        assert len(fake.puts) == 4
        for _, payload in fake.puts:
            assert payload == {"cmd": "start", "val": None}

    def test_fanout_search_custom_cn_ids(self, fake_store_pair):
        cs, fake = fake_store_pair
        keys = control_store.fanout_search(
            cs, cmd="stop", val=None, cn_ids=(1, 13),
        )
        assert keys == ["/cmd/search_rt/1", "/cmd/search_rt/13"]
        assert len(fake.puts) == 2


class TestFanoutCorr:
    def test_fanout_corr_default_hits_all_sixteen(self, fake_store_pair):
        cs, fake = fake_store_pair
        keys = control_store.fanout_corr(cs, cmd="stop", val=None)
        assert len(keys) == 16
        assert keys[0] == "/cmd/corr_rt/3"
        assert keys[-1] == "/cmd/corr_rt/22"
        assert all(p == {"cmd": "stop", "val": None} for _, p in fake.puts)


# ---------------------------------------------------------------------------
# compute_arm_seq
# ---------------------------------------------------------------------------


class TestComputeArmSeq:
    def test_walks_all_32_capture_keys(self, fake_store_pair):
        cs, fake = fake_store_pair
        control_store.compute_arm_seq(cs)
        # 16 corr cn × 2 ports = 32 polls
        assert len(fake.gets) == 32
        # First key should match cn=3 port=4011, last cn=22 port=4012
        assert fake.gets[0] == "/mon/corr_rt/3/capture/4011"
        assert fake.gets[-1] == "/mon/corr_rt/22/capture/4012"

    def test_returns_none_when_no_captures_answer(self, fake_store_pair):
        cs, _ = fake_store_pair
        out = control_store.compute_arm_seq(cs)
        assert out["arm_seq"] is None
        assert out["max_last_seq_no"] is None
        assert out["max_source"] is None
        assert len(out["missing"]) == 32
        assert out["answered"] == []

    def test_max_wins_across_all_captures(self):
        # Set up answers from a couple of captures with different
        # last_seq_no — ARM_SEQ = max + margin.
        responses = {
            "/mon/corr_rt/3/capture/4011":  {"last_seq_no": 100_000},
            "/mon/corr_rt/3/capture/4012":  {"last_seq_no":  90_000},
            "/mon/corr_rt/22/capture/4011": {"last_seq_no": 200_000},
            "/mon/corr_rt/22/capture/4012": {"last_seq_no": 150_000},
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_arm_seq(cs, margin=30_000)
        assert out["max_last_seq_no"] == 200_000
        assert out["max_source"] == "/mon/corr_rt/22/capture/4011"
        assert out["arm_seq"] == 230_000
        assert out["margin"] == 30_000
        assert len(out["answered"]) == 4
        assert len(out["missing"]) == 28

    def test_phase_b_margin_60k(self):
        responses = {
            "/mon/corr_rt/3/capture/4011": {"last_seq_no": 1_000_000},
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_arm_seq(cs, margin=60_000)
        assert out["arm_seq"] == 1_060_000

    def test_skips_non_dict_payload(self):
        # A capture entry of None or a string shouldn't crash; just
        # ends up in missing.
        responses = {
            "/mon/corr_rt/3/capture/4011": "garbage",
            "/mon/corr_rt/4/capture/4011": {"last_seq_no": 500},
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_arm_seq(cs, margin=100)
        assert out["arm_seq"] == 600


# ---------------------------------------------------------------------------
# audit_log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_writes_iso_timestamped_audit_row(self, fake_store_pair):
        cs, fake = fake_store_pair
        action = control_store.audit_log(
            cs,
            namespace="corr_rt",
            cn_target="0",
            cmd="utc_start",
            val=12345,
            ok=True,
            note="test",
            user="alice",
            host="lxd110h23",
        )
        # Exactly one put_dict.
        assert len(fake.puts) == 1
        key, payload = fake.puts[0]
        assert key.startswith("/mon/audit/control/")
        # ISO-8601 UTC with 'Z'.
        assert key.endswith("Z")
        assert payload["user"] == "alice"
        assert payload["host"] == "lxd110h23"
        assert payload["namespace"] == "corr_rt"
        assert payload["cn_target"] == "0"
        assert payload["cmd"] == "utc_start"
        assert payload["val"] == 12345
        assert payload["ok"] is True
        assert payload["note"] == "test"
        # The returned ControlAction matches the persisted dict.
        assert action.to_dict() == payload

    def test_audit_log_swallows_etcd_failure(self, fake_store_pair):
        cs, fake = fake_store_pair
        # Make put_dict raise.
        def _raise(*a, **kw): raise RuntimeError("etcd down")
        fake.put_dict = _raise        # type: ignore[assignment]
        # Should NOT raise — audit failures must not break the verb.
        action = control_store.audit_log(
            cs, namespace="corr_rt", cn_target="0",
            cmd="stop", val=None, ok=True, user="bob",
        )
        assert action.cmd == "stop"
        assert action.user == "bob"


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


class TestControlStartFleet:
    def test_writes_corr_broadcast_and_search_fanout(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.control_start_fleet(
            cs, obs_dec_deg=53.85, user="ops",
        )
        assert out["ok"] is True
        assert out["cmd"] == "start"
        assert out["val"] == 53.85
        assert out["corr_broadcast_key"] == "/cmd/corr_rt/0"
        assert out["search_fanout_keys"] == [
            "/cmd/search_rt/1",
            "/cmd/search_rt/2",
            "/cmd/search_rt/9",
            "/cmd/search_rt/13",
        ]
        # 1 corr broadcast + 4 search fanout + 1 audit row = 6 writes.
        assert len(fake.puts) == 6

    def test_null_dec_passes_through(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.control_start_fleet(cs, obs_dec_deg=None)
        assert out["val"] is None
        # Both writes carry val=None.
        verb_writes = [p for p in fake.puts if p[0].startswith("/cmd/")]
        for _, payload in verb_writes:
            assert payload == {"cmd": "start", "val": None}


class TestControlStopFleet:
    def test_default_fanout_corr_too(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.control_stop_fleet(cs, user="ops")
        assert out["ok"] is True
        assert out["cmd"] == "stop"
        # 1 corr broadcast + 16 corr fanout + 4 search fanout + 1 audit
        # = 22 writes.
        assert len(fake.puts) == 22
        # The corr_broadcast_key is the first corr write.
        assert out["corr_keys"][0] == "/cmd/corr_rt/0"

    def test_no_corr_fanout(self, fake_store_pair):
        cs, fake = fake_store_pair
        control_store.control_stop_fleet(cs, fanout_corr_too=False)
        # 1 corr broadcast + 4 search fanout + 1 audit = 6 writes.
        assert len(fake.puts) == 6


class TestControlUtcStartNow:
    def test_refuses_when_no_captures_answer(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.control_utc_start_now(cs, user="ops")
        assert out["ok"] is False
        assert "no captures answering" in out["error"]
        # Audit row should be written (the only put).
        assert len(fake.puts) == 1
        key, payload = fake.puts[0]
        assert key.startswith("/mon/audit/control/")
        assert payload["ok"] is False

    def test_broadcasts_arm_seq_on_success(self):
        responses = {
            "/mon/corr_rt/3/capture/4011": {"last_seq_no": 1_000_000},
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.control_utc_start_now(
            cs, margin=30_000, user="ops",
        )
        assert out["ok"] is True
        assert out["val"] == 1_030_000
        assert out["corr_broadcast_key"] == "/cmd/corr_rt/0"
        # Find the verb write.
        verb_writes = [
            (k, p) for k, p in fake.puts if k == "/cmd/corr_rt/0"
        ]
        assert len(verb_writes) == 1
        assert verb_writes[0][1] == {"cmd": "utc_start", "val": 1_030_000}


class TestControlUtcStopNow:
    def test_broadcasts_utc_stop_zero_margin(self):
        responses = {
            "/mon/corr_rt/3/capture/4011": {"last_seq_no": 5_000_000},
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.control_utc_stop_now(cs)
        assert out["ok"] is True
        assert out["val"] == 5_000_000          # margin=0
        verb_writes = [
            (k, p) for k, p in fake.puts if k == "/cmd/corr_rt/0"
        ]
        assert len(verb_writes) == 1
        assert verb_writes[0][1] == {"cmd": "utc_stop", "val": 5_000_000}

    def test_falls_back_to_zero_when_no_captures(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.control_utc_stop_now(cs)
        assert out["ok"] is True
        assert out["val"] == 0
        verb_writes = [
            (k, p) for k, p in fake.puts if k == "/cmd/corr_rt/0"
        ]
        assert verb_writes[0][1] == {"cmd": "utc_stop", "val": 0}


# ---------------------------------------------------------------------------
# Topology constants — the only place the dashboard pins fleet shape
# ---------------------------------------------------------------------------


class TestSendInject:
    """M7.4 Phase 6 runtime injection — etcd write surface."""

    _valid = dict(
        inj_id="phase6_unit_t1",
        l_rad=0.0,
        m_rad=0.0,
        dm_pc_cm3=500.0,
        fluence_jy_ms=50.0,
        width_samples=32,
        profile="gaussian",
        apply_at_specnum=1_000_000,
    )

    def test_writes_payload_to_every_requested_chgroup(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.send_inject(
            cs, chgroups=(0, 3, 15), **self._valid, user="ops",
        )
        assert out["ok"] is True
        assert out["chgroups"] == [0, 3, 15]
        assert out["keys"] == [
            "/cmd/dsart/corr/0/inject",
            "/cmd/dsart/corr/3/inject",
            "/cmd/dsart/corr/15/inject",
        ]
        # 3 inject PUTs + 1 audit row = 4 writes total.
        assert len(fake.puts) == 4
        # Every inject PUT carries the canonical wire shape.
        inject_puts = [
            p for p in fake.puts if p[0].startswith("/cmd/dsart/corr/")
        ]
        assert len(inject_puts) == 3
        for _, payload in inject_puts:
            assert payload["cmd"] == "inject"
            assert payload["val"]["inj_id"] == "phase6_unit_t1"
            assert payload["val"]["dm_pc_cm3"] == 500.0
            assert payload["val"]["profile"] == "gaussian"
            assert payload["val"]["apply_at_specnum"] == 1_000_000

    def test_default_chgroups_is_all_sixteen(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.send_inject(cs, **self._valid)
        assert out["chgroups"] == list(range(16))
        assert len(out["keys"]) == 16
        # 16 inject + 1 audit = 17 writes.
        assert len(fake.puts) == 17

    def test_validates_payload_before_any_etcd_write(self, fake_store_pair):
        cs, fake = fake_store_pair
        bad = dict(self._valid)
        # l^2 + m^2 = 0.65 + 0.81 = 1.46 >= 1.0 — should fail validation.
        bad["l_rad"] = 0.81
        bad["m_rad"] = 0.9
        with pytest.raises(ValueError, match="l_rad|m_rad|l²|>="):
            control_store.send_inject(cs, **bad)
        # No PUTs should have happened.
        assert fake.puts == []

    def test_validates_unknown_profile(self, fake_store_pair):
        cs, fake = fake_store_pair
        bad = dict(self._valid, profile="not_a_profile")
        with pytest.raises(ValueError, match="profile"):
            control_store.send_inject(cs, **bad)
        assert fake.puts == []

    def test_validates_width_samples_range(self, fake_store_pair):
        cs, fake = fake_store_pair
        bad = dict(self._valid, width_samples=0)
        with pytest.raises(ValueError, match="width_samples"):
            control_store.send_inject(cs, **bad)
        assert fake.puts == []

    def test_audit_row_summarises_fanout(self, fake_store_pair):
        cs, fake = fake_store_pair
        control_store.send_inject(
            cs, chgroups=(0, 1, 2), **self._valid, user="alice",
        )
        audit_puts = [
            p for p in fake.puts if p[0].startswith("/mon/audit/control/")
        ]
        assert len(audit_puts) == 1
        audit_payload = audit_puts[0][1]
        assert audit_payload["user"] == "alice"
        assert audit_payload["cmd"] == "inject"
        assert audit_payload["cn_target"] == "0,1,2"
        assert audit_payload["ok"] is True
        assert "chgroups=3" in audit_payload["note"]
        assert "apply_at_specnum=1000000" in audit_payload["note"]


class TestControlInjectPulse:
    """High-level helper that supports apply_at_specnum=None auto-arm."""

    _valid = dict(
        inj_id="phase6_auto_t1",
        l_rad=0.0,
        m_rad=0.0,
        dm_pc_cm3=500.0,
        fluence_jy_ms=50.0,
        width_samples=32,
        profile="gaussian",
    )

    def test_explicit_apply_at_specnum_passes_through(self, fake_store_pair):
        cs, fake = fake_store_pair
        out = control_store.control_inject_pulse(
            cs, apply_at_specnum=42_000, chgroups=(0,), **self._valid,
        )
        assert out["ok"] is True
        assert out["val"]["apply_at_specnum"] == 42_000
        assert out["auto_arm"] is False
        assert out["arm_info"] is None
        # Find the inject PUT and confirm specnum.
        for k, p in fake.puts:
            if k == "/cmd/dsart/corr/0/inject":
                assert p["val"]["apply_at_specnum"] == 42_000
                break
        else:
            pytest.fail("inject PUT not found")

    def test_auto_arm_uses_compute_inject_apply_at(self):
        """Phase 6c: auto-arm walks /mon/corr_rt/<cn>/corr_fast (NOT
        /capture/...) and arms at max(block_specnum_start) +
        margin_blocks × NPACKETS_PER_BLOCK."""
        now = time.time()
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 1000,
                "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now,
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.control_inject_pulse(
            cs, apply_at_specnum=None, margin_blocks=16,
            chgroups=(0,), **self._valid,
        )
        assert out["ok"] is True
        assert out["auto_arm"] is True
        # max=1000*2048=2_048_000, +16*2048=32_768 → 2_080_768
        assert out["val"]["apply_at_specnum"] == 1000 * 2048 + 16 * 2048
        assert out["arm_info"]["max_block_specnum_start"] == 1000 * 2048
        assert out["arm_info"]["max_block_n"] == 1000

    def test_auto_arm_refuses_when_no_corr_fast_publishers(
        self, fake_store_pair,
    ):
        cs, fake = fake_store_pair
        out = control_store.control_inject_pulse(
            cs, apply_at_specnum=None, chgroups=(0,), **self._valid,
        )
        assert out["ok"] is False
        assert "no /mon/corr_rt" in out["error"]
        # Only an audit row, no inject PUT.
        inject_puts = [
            p for p in fake.puts if p[0].startswith("/cmd/dsart/corr/")
        ]
        assert inject_puts == []
        audit_puts = [
            p for p in fake.puts if p[0].startswith("/mon/audit/control/")
        ]
        assert len(audit_puts) == 1
        assert audit_puts[0][1]["ok"] is False


class TestComputeInjectApplyAt:
    """M7.4 Phase 6c: compute_inject_apply_at walks
    /mon/corr_rt/<cn>/corr_fast and arms at
    max(block_specnum_start) + margin_blocks × NPACKETS_PER_BLOCK."""

    def test_single_chgroup_arms_at_max_plus_margin(self):
        now = time.time()
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 500,
                "block_specnum_start": 500 * 2048,
                "ts_wall_unix": now,
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_inject_apply_at(
            cs, margin_blocks=16, chgroups=(0,),
        )
        assert out["apply_at_specnum"] == 500 * 2048 + 16 * 2048
        assert out["max_block_specnum_start"] == 500 * 2048
        assert out["max_block_n"] == 500
        assert out["max_source"] == "/mon/corr_rt/0/corr_fast"
        assert out["answered"] == ["/mon/corr_rt/0/corr_fast"]
        assert out["stale"] == []

    def test_max_across_chgroups(self):
        now = time.time()
        # cg=0 at block 1000 (latest), cg=1 at block 980 (lagger).
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 1000, "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now,
            },
            "/mon/corr_rt/1/corr_fast": {
                "block_n": 980, "block_specnum_start": 980 * 2048,
                "ts_wall_unix": now,
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_inject_apply_at(
            cs, margin_blocks=16, chgroups=(0, 1),
        )
        # Pick MAX (1000), not min (980), so the lagger reaches it.
        assert out["max_block_specnum_start"] == 1000 * 2048
        assert out["apply_at_specnum"] == 1000 * 2048 + 16 * 2048
        assert out["max_source"] == "/mon/corr_rt/0/corr_fast"
        assert sorted(out["answered"]) == [
            "/mon/corr_rt/0/corr_fast",
            "/mon/corr_rt/1/corr_fast",
        ]

    def test_returns_none_when_no_publishers(self):
        fake = FakeDsaStore(get_dict_responses={})
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_inject_apply_at(
            cs, margin_blocks=16, chgroups=(0, 1, 2),
        )
        assert out["apply_at_specnum"] is None
        assert out["max_block_specnum_start"] is None
        assert out["max_source"] is None
        assert sorted(out["missing"]) == [
            "/mon/corr_rt/0/corr_fast",
            "/mon/corr_rt/1/corr_fast",
            "/mon/corr_rt/2/corr_fast",
        ]
        assert out["answered"] == []

    def test_stale_entries_rejected(self):
        """A publisher whose ts_wall_unix is older than max_age_s
        (default 10 s) does NOT contribute to the max."""
        now = time.time()
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 1000, "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now - 60,           # 60 s old → stale
            },
            "/mon/corr_rt/1/corr_fast": {
                "block_n": 50, "block_specnum_start": 50 * 2048,
                "ts_wall_unix": now,                # fresh
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_inject_apply_at(
            cs, margin_blocks=16, chgroups=(0, 1),
        )
        # Only chgroup=1 (fresh) contributes — block 50 not block 1000.
        assert out["max_block_specnum_start"] == 50 * 2048
        assert out["apply_at_specnum"] == 50 * 2048 + 16 * 2048
        assert out["stale"] == ["/mon/corr_rt/0/corr_fast"]
        assert out["answered"] == ["/mon/corr_rt/1/corr_fast"]

    def test_payload_without_block_specnum_start_is_missing(self):
        """If a corr_fast wrote ts_wall_unix but no
        block_specnum_start (old format), treat as missing."""
        now = time.time()
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "ts_wall_unix": now,
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_inject_apply_at(
            cs, margin_blocks=16, chgroups=(0,),
        )
        assert out["apply_at_specnum"] is None
        assert out["missing"] == ["/mon/corr_rt/0/corr_fast"]

    def test_npackets_per_block_constant_matches_corr_fast(self):
        """Pin the NPACKETS_PER_BLOCK constant in control_store
        against the canonical value in corr_fast_integration."""
        assert control_store.NPACKETS_PER_BLOCK == 2048
        assert control_store.DEFAULT_INJECT_MARGIN_BLOCKS == 16


class TestInjectKey:
    def test_canonical_key_layout(self):
        assert control_store._inject_key(0) == "/cmd/dsart/corr/0/inject"
        assert control_store._inject_key(15) == "/cmd/dsart/corr/15/inject"

    def test_default_chgroups_covers_all_sixteen(self):
        assert control_store.DEFAULT_INJECT_CHGROUPS == tuple(range(16))

    def test_corr_fast_mon_key_matches_dsart_publisher(self):
        """The dashboard's local key builder must match what
        dsart.services.corr_fast_mon.build_corr_fast_mon_key writes.
        """
        assert (
            control_store._corr_fast_mon_key(0)
            == "/mon/corr_rt/0/corr_fast"
        )
        assert (
            control_store._corr_fast_mon_key(15)
            == "/mon/corr_rt/15/corr_fast"
        )


class TestListRecentAudit:
    """Reads audit rows back via DsaStore.get_etcd().get_prefix() —
    the real DsaStore doesn't expose a recursive get of its own."""

    def test_returns_empty_when_no_rows(self, fake_store_pair):
        cs, _ = fake_store_pair
        assert control_store.list_recent_audit(cs) == []

    def test_returns_rows_sorted_newest_first(self, fake_store_pair):
        cs, fake = fake_store_pair
        # Write three audit rows directly (bypassing audit_log so we
        # can control the iso_ts ordering).
        for ts in ("2026-05-01T00:00:00.0Z",
                   "2026-05-03T00:00:00.0Z",
                   "2026-05-02T00:00:00.0Z"):
            fake.put_dict(
                f"/mon/audit/control/{ts}",
                {
                    "iso_ts": ts, "user": "ops", "host": "h23",
                    "namespace": "corr_rt", "cn_target": "0",
                    "cmd": "stop", "val": None,
                    "ok": True, "note": "",
                },
            )
        rows = control_store.list_recent_audit(cs)
        assert [r["iso_ts"] for r in rows] == [
            "2026-05-03T00:00:00.0Z",
            "2026-05-02T00:00:00.0Z",
            "2026-05-01T00:00:00.0Z",
        ]

    def test_respects_limit(self, fake_store_pair):
        cs, fake = fake_store_pair
        for i in range(8):
            ts = f"2026-05-{i+1:02d}T00:00:00.0Z"
            fake.put_dict(
                f"/mon/audit/control/{ts}",
                {"iso_ts": ts, "cmd": "stop", "ok": True, "user": "ops",
                 "host": "h23", "namespace": "corr_rt",
                 "cn_target": "0", "val": None, "note": ""},
            )
        rows = control_store.list_recent_audit(cs, limit=3)
        assert len(rows) == 3
        assert rows[0]["iso_ts"] == "2026-05-08T00:00:00.0Z"

    def test_round_trip_with_audit_log(self, fake_store_pair):
        cs, _ = fake_store_pair
        a = control_store.audit_log(
            cs, namespace="corr_rt", cn_target="0",
            cmd="utc_start", val=12345, ok=True,
            note="round-trip", user="alice",
        )
        rows = control_store.list_recent_audit(cs)
        assert len(rows) == 1
        assert rows[0]["cmd"] == "utc_start"
        assert rows[0]["user"] == "alice"
        assert rows[0]["iso_ts"] == a.iso_ts


class TestTopologyConstants:
    def test_corr_count_matches_phase_b(self):
        assert len(control_store.CORR_CN_IDS) == 16

    def test_search_count(self):
        assert control_store.SEARCH_CN_IDS == (1, 2, 9, 13)

    def test_capture_ports(self):
        assert control_store.CAPTURE_UDP_PORTS == (4011, 4012)

    def test_no_overlap_with_search(self):
        # cn 9, 13 host search_rt; must NOT appear in corr.
        assert 9 not in control_store.CORR_CN_IDS
        assert 13 not in control_store.CORR_CN_IDS
