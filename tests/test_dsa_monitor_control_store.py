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
        # 2026-06-10 staleness extrapolation: a fresh publisher
        # (age ≈ 0⁺) is rounded UP one whole block so we never target
        # a block boundary that has already started. So:
        # max = (1000 + 1) blocks, + 16-block margin.
        assert out["val"]["apply_at_specnum"] == (1000 + 1 + 16) * 2048
        assert out["arm_info"]["max_block_specnum_start"] == (1000 + 1) * 2048
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
        # 2026-06-10: fresh publisher (age ≈ 0⁺) extrapolates +1 block.
        assert out["apply_at_specnum"] == (500 + 1 + 16) * 2048
        assert out["max_block_specnum_start"] == (500 + 1) * 2048
        assert out["max_block_n"] == 500
        assert out["extrapolated_specnums"] == 2048
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
        # (+1 block: fresh-publisher extrapolation round-up.)
        assert out["max_block_specnum_start"] == (1000 + 1) * 2048
        assert out["apply_at_specnum"] == (1000 + 1 + 16) * 2048
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
        # (+1 block: fresh-publisher extrapolation round-up.)
        assert out["max_block_specnum_start"] == (50 + 1) * 2048
        assert out["apply_at_specnum"] == (50 + 1 + 16) * 2048
        assert out["stale"] == ["/mon/corr_rt/0/corr_fast"]
        assert out["answered"] == ["/mon/corr_rt/1/corr_fast"]

    def test_stale_within_window_is_extrapolated(self):
        """2026-06-10 root-cause regression (260610snoe/mamv): the
        corr_fast mon-point publishes every 16 blocks (~2.1 s), so a
        2.0-s-old (but accepted) entry must be advanced ~15 blocks to
        the live stream position before the margin is added —
        otherwise the entire margin is silently consumed by staleness
        and the corr nodes receive an apply_at already in the past
        (→ partial-band injection)."""
        now = time.time()
        age_s = 2.0
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 1000, "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now - age_s,
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.compute_inject_apply_at(
            cs, margin_blocks=32, chgroups=(0,),
        )
        block_s = control_store._SPECNUM_SECONDS * 2048   # ≈ 0.1342 s
        expect_blocks = int(age_s / block_s) + 1          # 14 + 1 = 15
        assert out["extrapolated_specnums"] == expect_blocks * 2048
        assert out["max_block_specnum_start"] == (1000 + expect_blocks) * 2048
        assert out["apply_at_specnum"] == (
            (1000 + expect_blocks + 32) * 2048
        )

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
        # 2026-06-10: 16 → 32 blocks (~4.3 s). The corr_fast mon-point
        # is published every 16 blocks, so a 16-block margin gave an
        # effective lead of 0–2.1 s depending on publish phase →
        # partial-band injections (260610snoe/mamv).
        assert control_store.DEFAULT_INJECT_MARGIN_BLOCKS == 32


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


# ---------------------------------------------------------------------------
# M7.4 Phase 8 v3: quick-recovery helpers
# ---------------------------------------------------------------------------


class TestBounceSearch:
    """``bounce_search`` cycles stop → sleep → start across the
    selected search cn_ids. The unit test pins the call order (sleep
    happens between the two fanouts) + the audit shape.
    """

    def test_default_hits_all_search_cn_ids(self, fake_store_pair):
        cs, fake = fake_store_pair
        sleeps: list[float] = []
        out = control_store.bounce_search(
            cs, user="ops", sleep_fn=sleeps.append,
        )
        assert out["ok"] is True
        assert out["cmd"] == "bounce_search"
        assert out["cn_ids"] == list(control_store.SEARCH_CN_IDS)
        assert out["stop_keys"] == [
            f"/cmd/search_rt/{c}" for c in control_store.SEARCH_CN_IDS
        ]
        assert out["start_keys"] == [
            f"/cmd/search_rt/{c}" for c in control_store.SEARCH_CN_IDS
        ]
        # 4 stop + 4 start + 1 audit = 9 writes.
        assert len(fake.puts) == 9
        assert sleeps == [control_store.DEFAULT_BOUNCE_SLEEP_S]

    def test_custom_cn_ids_only_bounces_those(self, fake_store_pair):
        cs, fake = fake_store_pair
        sleeps: list[float] = []
        out = control_store.bounce_search(
            cs, cn_ids=(1,), user="ops", sleep_fn=sleeps.append,
        )
        assert out["cn_ids"] == [1]
        assert out["stop_keys"] == ["/cmd/search_rt/1"]
        assert out["start_keys"] == ["/cmd/search_rt/1"]
        # 1 stop + 1 start + 1 audit = 3 writes.
        assert len(fake.puts) == 3

    def test_sleep_runs_between_stop_and_start(self, fake_store_pair):
        """Record the ordering of put_dict + sleep calls and assert
        the sleep falls strictly between the two fanouts."""
        cs, fake = fake_store_pair
        events: list[tuple[str, str]] = []

        original_put = fake.put_dict

        def record_put(key, payload):
            events.append(("put", key))
            return original_put(key, payload)

        fake.put_dict = record_put

        def record_sleep(s):
            events.append(("sleep", str(s)))

        control_store.bounce_search(
            cs, cn_ids=(1, 2), user="ops", sleep_fn=record_sleep,
        )
        # Expected order: 2x put (stop), 1x sleep, 2x put (start), 1x put (audit).
        kinds = [e[0] for e in events]
        assert kinds == ["put", "put", "sleep", "put", "put", "put"]
        # First two puts are stop fanout.
        assert events[0][1] == "/cmd/search_rt/1"
        assert events[1][1] == "/cmd/search_rt/2"
        # Next two are start fanout.
        assert events[3][1] == "/cmd/search_rt/1"
        assert events[4][1] == "/cmd/search_rt/2"
        # Last is the audit row.
        assert events[5][1].startswith("/mon/audit/control/")

    def test_audit_row_carries_cn_ids(self, fake_store_pair):
        cs, fake = fake_store_pair
        control_store.bounce_search(
            cs, cn_ids=(2,), user="alice", sleep_fn=lambda _s: None,
        )
        audit_writes = [p for p in fake.puts
                        if p[0].startswith("/mon/audit/control/")]
        assert len(audit_writes) == 1
        payload = audit_writes[0][1]
        assert payload["cmd"] == "bounce_search"
        assert payload["val"]["cn_ids"] == [2]
        assert payload["ok"] is True
        assert payload["user"] == "alice"

    def test_empty_cn_ids_raises(self, fake_store_pair):
        cs, _ = fake_store_pair
        with pytest.raises(ValueError, match="at least one cn_id"):
            control_store.bounce_search(
                cs, cn_ids=[], sleep_fn=lambda _s: None,
            )

    def test_negative_sleep_raises(self, fake_store_pair):
        cs, _ = fake_store_pair
        with pytest.raises(ValueError, match="sleep_s"):
            control_store.bounce_search(
                cs, sleep_s=-1.0, sleep_fn=lambda _s: None,
            )

    def test_payload_carries_val_none_on_start_and_stop(self, fake_store_pair):
        """Both fanouts MUST use val=None so the search orchestrator
        falls back to /mon/array/dec per the M7.4 CUSTOMDEC contract."""
        cs, fake = fake_store_pair
        control_store.bounce_search(
            cs, cn_ids=(1,), sleep_fn=lambda _s: None,
        )
        verb_writes = [p for p in fake.puts if p[0].startswith("/cmd/")]
        assert len(verb_writes) == 2
        for _, payload in verb_writes:
            assert payload["val"] is None


class TestRestartC2ServiceLocal:
    """``restart_c2_service_local`` is the only Control-tab verb that
    shells out locally (every other write goes through etcd). The
    tests pin the argv it calls + audit-row shape on success / failure
    / timeout / missing binary.
    """

    def test_success_returns_ok_true_and_audits(
        self, fake_store_pair, monkeypatch
    ):
        cs, fake = fake_store_pair
        captured: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["kwargs"] = dict(kwargs)
            return mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_c2_service_local(cs, user="ops")
        assert out["ok"] is True
        assert out["rc"] == 0
        assert captured["argv"] == [
            "systemctl", "--user", "restart", "dsart_c2.service",
        ]
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True
        # Audit row recorded.
        audit_writes = [p for p in fake.puts
                        if p[0].startswith("/mon/audit/control/")]
        assert len(audit_writes) == 1
        assert audit_writes[0][1]["cmd"] == "restart_c2"
        assert audit_writes[0][1]["ok"] is True

    def test_non_zero_rc_returns_ok_false(self, fake_store_pair, monkeypatch):
        cs, fake = fake_store_pair

        def fake_run(argv, **kwargs):
            return mock.Mock(
                returncode=5, stdout="oops\n", stderr="bad service",
            )

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_c2_service_local(cs, user="ops")
        assert out["ok"] is False
        assert out["rc"] == 5
        assert out["err"] == "rc=5"
        # Audit row still written, ok=False.
        audit_writes = [p for p in fake.puts
                        if p[0].startswith("/mon/audit/control/")]
        assert len(audit_writes) == 1
        assert audit_writes[0][1]["ok"] is False

    def test_timeout_returns_ok_false(self, fake_store_pair, monkeypatch):
        cs, _ = fake_store_pair
        import subprocess as _sp

        def fake_run(argv, **kwargs):
            raise _sp.TimeoutExpired(cmd=argv, timeout=10.0)

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_c2_service_local(cs, timeout_s=10.0)
        assert out["ok"] is False
        assert "timeout" in out["err"]

    def test_missing_binary_returns_ok_false(
        self, fake_store_pair, monkeypatch
    ):
        cs, _ = fake_store_pair

        def fake_run(argv, **kwargs):
            raise FileNotFoundError(2, "no such file", "systemctl")

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_c2_service_local(cs)
        assert out["ok"] is False
        assert "FileNotFoundError" in out["err"]


class TestRestartH23ServicesLocal:
    """``restart_h23_services_local`` cycles every h23 dsa110-rt unit via
    local ``systemctl --user restart`` and writes ONE summary audit row.
    """

    def test_all_units_restarted_in_order_ok(
        self, fake_store_pair, monkeypatch
    ):
        cs, fake = fake_store_pair
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_h23_services_local(cs, user="ops")
        assert out["ok"] is True
        assert out["cmd"] == "restart_h23_services"
        # One systemctl call per unit, in inventory order.
        restarted = [c[-1] for c in calls]
        assert restarted == list(control_store.H23_DSART_UNITS)
        assert all(c[:3] == ["systemctl", "--user", "restart"] for c in calls)
        assert [r["unit"] for r in out["results"]] == list(
            control_store.H23_DSART_UNITS)
        assert all(r["ok"] for r in out["results"])
        # Exactly one summary audit row.
        audit = [p for p in fake.puts
                 if p[0].startswith("/mon/audit/control/")]
        assert len(audit) == 1
        assert audit[0][1]["cmd"] == "restart_h23_services"
        assert audit[0][1]["ok"] is True
        assert audit[0][1]["val"] == {
            u: True for u in control_store.H23_DSART_UNITS}

    def test_one_unit_fails_others_still_run_and_overall_not_ok(
        self, fake_store_pair, monkeypatch
    ):
        cs, fake = fake_store_pair

        def fake_run(argv, **kwargs):
            unit = argv[-1]
            rc = 5 if unit == "dsart_c3.service" else 0
            return mock.Mock(returncode=rc, stdout="", stderr="boom")

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_h23_services_local(cs)
        assert out["ok"] is False
        # All units were still attempted (best-effort).
        assert len(out["results"]) == len(control_store.H23_DSART_UNITS)
        bad = [r for r in out["results"] if not r["ok"]]
        assert [r["unit"] for r in bad] == ["dsart_c3.service"]
        audit = [p for p in fake.puts
                 if p[0].startswith("/mon/audit/control/")]
        assert len(audit) == 1
        assert audit[0][1]["ok"] is False
        assert "dsart_c3.service" in audit[0][1]["note"]

    def test_timeout_on_one_unit_is_contained(
        self, fake_store_pair, monkeypatch
    ):
        cs, _ = fake_store_pair
        import subprocess as _sp

        def fake_run(argv, **kwargs):
            if argv[-1] == "hiplot_c1.service":
                raise _sp.TimeoutExpired(cmd=argv, timeout=30.0)
            return mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.restart_h23_services_local(cs)
        assert out["ok"] is False
        to = [r for r in out["results"] if r["unit"] == "hiplot_c1.service"][0]
        assert "timeout" in to["err"]
        # The other three still report a result.
        assert len(out["results"]) == len(control_store.H23_DSART_UNITS)


class TestC2MonSnapshot:
    """``c2_mon_snapshot`` derives a compact JSON-ready view of
    ``/mon/c2/h23``. Missing key → ok=False, present key → broken
    down into counters / dumps_enabled / inject_match buckets.
    """

    def test_missing_key_returns_ok_false(self, fake_store_pair):
        cs, _ = fake_store_pair
        out = control_store.c2_mon_snapshot(cs)
        assert out["ok"] is False
        assert out["raw"] is None
        assert out["counters"] == {}
        assert out["dumps_enabled"] is None

    def test_present_key_returns_compact_view(self):
        now = time.time()
        responses = {
            "/mon/c2/h23": {
                "dumps_enabled": True,
                "last_event_name": "260528ddsg",
                "last_trigger_class": "bright_frb_extragalactic",
                "counters": {
                    "rows_in": 12345,
                    "triggers_dump": 7,
                    "triggers_suppressed": 0,
                    "triggers_log_only": 42,
                },
                "inject_match": {
                    "active_count": 1,
                    "rows_checked": 12340,
                    "matches": 3,
                    "evicted_expired": 5,
                    "last_refresh_unix": now - 1.5,
                    "active": [{"inj_id": "test_bright"}],
                },
            },
        }
        fake = FakeDsaStore(get_dict_responses=responses)
        cs = control_store.ControlStore()
        cs._store = fake
        out = control_store.c2_mon_snapshot(cs)
        assert out["ok"] is True
        assert out["dumps_enabled"] is True
        assert out["last_event_name"] == "260528ddsg"
        assert out["counters"]["rows_in"] == 12345
        assert out["counters"]["triggers_dump"] == 7
        im = out["inject_match"]
        assert im["active_count"] == 1
        assert im["matches"] == 3
        assert im["evicted_expired"] == 5
        # last_refresh_age_s should be a small positive float.
        assert im["last_refresh_age_s"] is not None
        assert 0.0 < im["last_refresh_age_s"] < 60.0
        assert im["active"] == [{"inj_id": "test_bright"}]

    def test_etcd_failure_surfaces_as_ok_false(
        self, fake_store_pair, monkeypatch
    ):
        cs, fake = fake_store_pair

        def bad_get(_key):
            raise RuntimeError("etcd down")

        # Monkey-patch the FakeDsaStore.get_dict to raise.
        monkeypatch.setattr(fake, "get_dict", bad_get)
        out = control_store.c2_mon_snapshot(cs)
        assert out["ok"] is False
        assert "etcd down" in out["error"]


class TestC2JournalTailLocal:
    """``c2_journal_tail_local`` shells out to journalctl. We mock
    subprocess.run so the test stays fast + deterministic.
    """

    _SAMPLE_OUT = (
        "May 28 19:20:44 lxd110h23 dsart_c2[9292]: WOULD-DUMP class=bright n=5\n"
        "May 28 19:28:26 lxd110h23 dsart_c2[37721]: LOG class=log_only n=3 snr_max=14.12\n"
        "May 28 19:30:00 lxd110h23 dsart_c2[37721]: client connected /127.0.0.1:55432\n"
        "May 28 19:31:00 lxd110h23 dsart_c2[37721]: FIRE class=bright name=260528abcd snr_max=18.5\n"
        "May 28 19:32:00 lxd110h23 dsart_c2[37721]: inject_match: matched test_bright\n"
    )

    def test_default_filter_keeps_only_decision_lines(self, monkeypatch):
        def fake_run(argv, **kwargs):
            assert argv[0] == "journalctl"
            assert "--user" in argv
            assert "-u" in argv
            assert "dsart_c2.service" in argv
            return mock.Mock(
                returncode=0, stdout=self._SAMPLE_OUT, stderr="",
            )

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.c2_journal_tail_local(limit=10)
        assert out["ok"] is True
        # 5 raw lines, 4 match the decision regex (WOULD-DUMP, LOG class,
        # FIRE class, inject_match). The "client connected" line is dropped.
        assert out["raw_lines_scanned"] == 5
        assert len(out["lines"]) == 4
        joined = "\n".join(out["lines"])
        assert "client connected" not in joined
        assert "WOULD-DUMP" in joined
        assert "FIRE class" in joined
        assert "inject_match" in joined

    def test_unfiltered_keeps_every_line(self, monkeypatch):
        def fake_run(argv, **kwargs):
            return mock.Mock(
                returncode=0, stdout=self._SAMPLE_OUT, stderr="",
            )

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.c2_journal_tail_local(limit=100, grep_re=None)
        assert out["ok"] is True
        assert len(out["lines"]) == 5

    def test_limit_caps_returned_lines(self, monkeypatch):
        def fake_run(argv, **kwargs):
            return mock.Mock(
                returncode=0, stdout=self._SAMPLE_OUT, stderr="",
            )

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.c2_journal_tail_local(limit=2)
        # Default decision regex keeps 4 lines, limit=2 trims to 2
        # (keeping the newest, i.e. FIRE + inject_match).
        assert len(out["lines"]) == 2
        assert "inject_match" in out["lines"][-1]
        assert "FIRE class" in out["lines"][-2]

    def test_journalctl_failure_returns_ok_false(self, monkeypatch):
        def fake_run(argv, **kwargs):
            return mock.Mock(
                returncode=1, stdout="", stderr="permission denied",
            )

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.c2_journal_tail_local()
        assert out["ok"] is False
        assert "rc=1" in out["err"]
        assert out["lines"] == []

    def test_missing_journalctl_returns_ok_false(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise FileNotFoundError(2, "no journalctl", "journalctl")

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.c2_journal_tail_local()
        assert out["ok"] is False
        assert "FileNotFoundError" in out["err"]

    def test_timeout_returns_ok_false(self, monkeypatch):
        import subprocess as _sp

        def fake_run(argv, **kwargs):
            raise _sp.TimeoutExpired(cmd=argv, timeout=5.0)

        monkeypatch.setattr(control_store.subprocess, "run", fake_run)
        out = control_store.c2_journal_tail_local(timeout_s=5.0)
        assert out["ok"] is False
        assert "timeout" in out["err"]


# ---------------------------------------------------------------------------
# _capture_is_writing / the compute_system_state observing gate
# ---------------------------------------------------------------------------


class TestCaptureIsWriting:
    """A capture reports ``arm_state='WRITING'`` only while it crosses
    the armed specnum; minutes later it reads ``ARMED`` again while
    still writing every block. Regression for 2026-08-20, when a fully
    observing fleet showed up as "PREPARED - safe to arm".
    """

    def test_writing_label_is_writing(self):
        assert control_store._capture_is_writing({"arm_state": "WRITING"})

    def test_armed_past_start_specnum_is_writing(self):
        # the values actually observed on n04 at 05:32 UTC 2026-08-20
        assert control_store._capture_is_writing({
            "arm_state": "ARMED",
            "utc_start_specnum": 678660702,
            "utc_stop_specnum": 0,
            "last_seq_no": 681239010,
        })

    def test_armed_before_start_specnum_is_not_writing(self):
        assert not control_store._capture_is_writing({
            "arm_state": "ARMED",
            "utc_start_specnum": 678660702,
            "utc_stop_specnum": 0,
            "last_seq_no": 678600702,
        })

    def test_armed_inside_explicit_window_is_writing(self):
        assert control_store._capture_is_writing({
            "arm_state": "ARMED",
            "utc_start_specnum": 100,
            "utc_stop_specnum": 300,
            "last_seq_no": 250,
        })

    def test_armed_past_stop_specnum_is_not_writing(self):
        assert not control_store._capture_is_writing({
            "arm_state": "ARMED",
            "utc_start_specnum": 100,
            "utc_stop_specnum": 200,
            "last_seq_no": 250,
        })

    def test_waiting_for_arm_is_not_writing(self):
        assert not control_store._capture_is_writing({
            "arm_state": "WAITING_FOR_ARM",
            "utc_start_specnum": 0,
            "last_seq_no": 12345,
        })

    def test_never_armed_is_not_writing(self):
        assert not control_store._capture_is_writing({
            "arm_state": "ARMED",
            "utc_start_specnum": 0,
            "last_seq_no": 12345,
        })

    def test_missing_seq_fields_is_not_writing(self):
        assert not control_store._capture_is_writing({"arm_state": "ARMED"})

    @pytest.mark.parametrize("payload", [None, "ARMED", 17, [], {}])
    def test_non_dict_or_empty_is_not_writing(self, payload):
        assert not control_store._capture_is_writing(payload)


class TestSystemStateObservingGate:
    @staticmethod
    def _responses(capture: dict[str, Any]) -> dict[str, Any]:
        now_mjd = time.time() / 86400.0 + 40587.0
        return {
            "/mon/corr_rt/3": {
                "time_mjd": now_mjd,
                "state": "running",
                "routines": {"corr_fast": {"pid": 42}},
            },
            "/mon/corr_rt/3/capture/4011": capture,
            "/mon/search_rt/1": {"time_mjd": now_mjd, "state": "running"},
        }

    def _state_for(self, capture: dict[str, Any]) -> dict[str, Any]:
        cs = control_store.ControlStore()
        cs._store = FakeDsaStore(self._responses(capture))
        return control_store.compute_system_state(
            cs, corr_cn_ids=[3], search_cn_ids=[1], ports=[4011],
        )

    def test_armed_past_start_reads_as_observing(self):
        out = self._state_for({
            "arm_state": "ARMED",
            "utc_start_specnum": 678660702,
            "utc_stop_specnum": 0,
            "last_seq_no": 681239010,
        })
        assert out["state"] == "observing"
        assert out["counts"]["captures_writing"] == 1

    def test_armed_before_start_is_not_observing(self):
        out = self._state_for({
            "arm_state": "ARMED",
            "utc_start_specnum": 678660702,
            "utc_stop_specnum": 0,
            "last_seq_no": 678600702,
        })
        assert out["state"] != "observing"
        assert out["counts"]["captures_writing"] == 0
