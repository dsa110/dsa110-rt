"""Unit tests for the dsa_monitor Control tab's dumps-toggle module
(M7.4 Phase 6c).

Exercises the small dashboard front-end for ``/cmd/c2/dumps_enabled``:

* :func:`dumps_gate.get_dumps_state` — present-key happy path, missing
  key fail-OPEN with ``default=True``, malformed payload (no
  ``enabled`` field) treated as missing.
* :func:`dumps_gate.set_dumps_state` — happy path writes both the cmd
  key + the audit row, empty / over-long reasons raise ``ValueError``
  *before* any etcd write, audit-write failure is swallowed.
* :func:`dumps_gate.list_recent_toggles` — newest-first ordering,
  limit honoured, etcd errors return ``[]`` rather than raise.

The module under test lives in
``tools/dashboard/dsa_monitor/dumps_gate.py``; we make it importable
by inserting that directory at the top of ``sys.path`` exactly the
way :mod:`tests.test_dsa_monitor_control_store` does.

We also pin the key + audit prefix constants against the
authoritative copies in
:mod:`dsart.services.coincidencer` so the two halves cannot drift.
"""

from __future__ import annotations

import json
import os
import sys
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

# ``src`` so we can pin DUMPS_ENABLED_KEY against the canonical copy
# living in dsart.services.coincidencer. The dashboard module
# duplicates the constant; the assertion below catches drift.
SRC_DIR = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dumps_gate                                                 # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEtcdMeta:
    def __init__(self, key: str) -> None:
        self.key = key.encode("utf-8") if isinstance(key, str) else key


class _FakeEtcdClient:
    """Minimal etcd3 stand-in: get_prefix() yields (value_bytes, meta)."""

    def __init__(self, kv: dict[str, Any]) -> None:
        self._kv = kv

    def get_prefix(self, prefix: str):
        for k, v in sorted(self._kv.items()):
            if not k.startswith(prefix):
                continue
            if isinstance(v, dict):
                value = json.dumps(v).encode("utf-8")
            elif isinstance(v, (bytes, bytearray)):
                value = bytes(v)
            else:
                value = str(v).encode("utf-8")
            yield value, _FakeEtcdMeta(k)


class FakeDsaStore:
    """DsaStore stand-in with deterministic puts/gets."""

    def __init__(self, get_dict_responses: dict[str, Any] | None = None):
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self.put_raises: dict[str, Exception] = {}
        self._responses = dict(get_dict_responses or {})
        self._kv_for_etcd: dict[str, Any] = {}

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        if key in self.put_raises:
            raise self.put_raises[key]
        self.puts.append((key, dict(payload)))
        self._kv_for_etcd[key] = dict(payload)

    def get_dict(self, key: str) -> Any:
        self.gets.append(key)
        return self._responses.get(key)

    def get_etcd(self):
        return _FakeEtcdClient(self._kv_for_etcd)


@pytest.fixture()
def fake_store():
    """A fresh FakeDsaStore each test."""
    return FakeDsaStore()


# ---------------------------------------------------------------------------
# Constants — pinned against the authoritative copy in the service module
# ---------------------------------------------------------------------------


def test_dumps_enabled_key_pinned_to_service_copy() -> None:
    """The dashboard's DUMPS_ENABLED_KEY must equal the coincidencer's
    copy. The two duplicate the literal so the dashboard can build the
    URL without importing dsart (h23 dashboard env historically does
    not always have src/ on PYTHONPATH); but if they drift, the toggle
    silently writes to the wrong key. Pinned here.
    """
    from dsart.services.coincidencer import DUMPS_ENABLED_KEY as SVC_KEY
    assert dumps_gate.DUMPS_ENABLED_KEY == SVC_KEY


def test_dumps_audit_prefix_under_control_audit_namespace() -> None:
    """The audit rows must sit under /mon/audit/control/ so the
    canonical list_recent_audit() (in control_store.py) still picks
    them up alongside restart_all / inject rows.
    """
    assert dumps_gate.DUMPS_AUDIT_PREFIX.startswith("/mon/audit/control/")
    assert dumps_gate.DUMPS_AUDIT_PREFIX.endswith("/")
    assert dumps_gate.DUMPS_AUDIT_PREFIX == "/mon/audit/control/dumps_toggle/"


# ---------------------------------------------------------------------------
# get_dumps_state
# ---------------------------------------------------------------------------


class TestGetDumpsState:
    def test_missing_key_fail_open_default_true(self, fake_store) -> None:
        out = dumps_gate.get_dumps_state(fake_store)
        assert out == {
            "enabled": True,
            "ts": None,
            "actor": None,
            "reason": None,
            "default": True,
        }
        # Only one read happened; it hit the canonical key.
        assert fake_store.gets == [dumps_gate.DUMPS_ENABLED_KEY]

    def test_present_key_returns_wrapper(self) -> None:
        fake = FakeDsaStore(get_dict_responses={
            dumps_gate.DUMPS_ENABLED_KEY: {
                "enabled": False,
                "ts": 1700000000.0,
                "actor": "alice",
                "reason": "rfi-burst",
            },
        })
        out = dumps_gate.get_dumps_state(fake)
        assert out == {
            "enabled": False,
            "ts": 1700000000.0,
            "actor": "alice",
            "reason": "rfi-burst",
            "default": False,
        }

    def test_malformed_payload_treated_as_missing(self) -> None:
        # A dict missing the "enabled" key is malformed; fall back to
        # fail-OPEN default.
        for bogus in (
            {"foo": "bar"},
            [1, 2, 3],
            "garbage",
            42,
            None,
        ):
            fake = FakeDsaStore(get_dict_responses={
                dumps_gate.DUMPS_ENABLED_KEY: bogus,
            })
            out = dumps_gate.get_dumps_state(fake)
            assert out["enabled"] is True, repr(bogus)
            assert out["default"] is True, repr(bogus)

    def test_enabled_coerced_to_bool(self) -> None:
        fake = FakeDsaStore(get_dict_responses={
            dumps_gate.DUMPS_ENABLED_KEY: {
                "enabled": 0,                                   # int → False
                "ts": 1.0, "actor": "ops", "reason": "n",
            },
        })
        out = dumps_gate.get_dumps_state(fake)
        assert out["enabled"] is False
        assert out["default"] is False


# ---------------------------------------------------------------------------
# set_dumps_state
# ---------------------------------------------------------------------------


class TestSetDumpsState:
    def test_happy_path_writes_cmd_key_and_audit_row(self, fake_store) -> None:
        out = dumps_gate.set_dumps_state(
            fake_store,
            enabled=False,
            reason="RFI burst — known-bad-data window",
            actor="alice",
            host="lxd110h23",
            now_unix=1700000000.250,
        )
        # Returned state mirrors the persisted payload.
        assert out["enabled"] is False
        assert out["actor"] == "alice"
        assert out["reason"] == "RFI burst — known-bad-data window"
        assert out["ts"] == 1700000000.250
        assert out["default"] is False
        assert out["audit_key"].startswith(dumps_gate.DUMPS_AUDIT_PREFIX)

        # Two puts: the cmd key + one audit row.
        assert len(fake_store.puts) == 2
        cmd_writes = [
            p for p in fake_store.puts
            if p[0] == dumps_gate.DUMPS_ENABLED_KEY
        ]
        audit_writes = [
            p for p in fake_store.puts
            if p[0].startswith(dumps_gate.DUMPS_AUDIT_PREFIX)
        ]
        assert len(cmd_writes) == 1
        assert len(audit_writes) == 1

        cmd_key, cmd_payload = cmd_writes[0]
        assert cmd_payload["enabled"] is False
        assert cmd_payload["ts"] == 1700000000.250
        assert cmd_payload["actor"] == "alice"
        assert cmd_payload["reason"] == "RFI burst — known-bad-data window"

        audit_key, audit_payload = audit_writes[0]
        # Key must be unix_ms (= int(ts * 1000)).
        assert audit_key == (
            dumps_gate.DUMPS_AUDIT_PREFIX + str(int(1700000000.250 * 1000))
        )
        # Shape mirrors ControlAction.to_dict() so the unified audit
        # panel renders the row in the right slot.
        assert audit_payload["cmd"] == "dumps_toggle"
        assert audit_payload["namespace"] == "c2.dumps_toggle"
        assert audit_payload["val"] == {"enabled": False}
        assert audit_payload["ok"] is True
        assert audit_payload["user"] == "alice"
        assert audit_payload["host"] == "lxd110h23"
        assert audit_payload["note"] == out["reason"]
        # iso_ts is ISO-8601 UTC with the trailing Z.
        assert audit_payload["iso_ts"].endswith("Z")
        # And it's deterministic (because we injected now_unix).
        # 1700000000.250 → 2023-11-14T22:13:20.250000Z
        assert audit_payload["iso_ts"] == "2023-11-14T22:13:20.250000Z"

    def test_empty_reason_raises_before_any_etcd_write(self, fake_store) -> None:
        for bogus in ("", "   ", "\t\n", None):
            with pytest.raises(ValueError, match="reason"):
                dumps_gate.set_dumps_state(
                    fake_store, enabled=True, reason=bogus, actor="ops",
                )
        # NOT a single etcd write happened on any of those rejections.
        assert fake_store.puts == []

    def test_overlong_reason_raises_before_any_etcd_write(
        self, fake_store,
    ) -> None:
        too_long = "x" * (dumps_gate.MAX_REASON_LEN + 1)
        with pytest.raises(ValueError, match="too long"):
            dumps_gate.set_dumps_state(
                fake_store, enabled=True, reason=too_long, actor="ops",
            )
        assert fake_store.puts == []

    def test_max_length_reason_is_accepted(self, fake_store) -> None:
        # Right at the cap → must succeed.
        ok = "y" * dumps_gate.MAX_REASON_LEN
        out = dumps_gate.set_dumps_state(
            fake_store, enabled=True, reason=ok, actor="ops",
            now_unix=2.0,
        )
        assert out["reason"] == ok
        # Both writes happened.
        assert len(fake_store.puts) == 2

    def test_audit_write_failure_does_not_break_the_flip(self) -> None:
        # If the cmd key write succeeds but the audit row fails, we
        # must still return the new state (the operator's flip went
        # through; we just lost the audit row to a transient etcd
        # hiccup).
        fake = FakeDsaStore()
        original_put = fake.put_dict

        call_n = [0]

        def maybe_fail(key, payload):
            call_n[0] += 1
            if call_n[0] == 1:
                # First call: cmd key — let it through.
                return original_put(key, payload)
            # Second call: audit row — explode.
            raise RuntimeError("etcd timeout on audit row")

        fake.put_dict = maybe_fail                                 # type: ignore[assignment]
        out = dumps_gate.set_dumps_state(
            fake, enabled=False, reason="rfi", actor="ops",
            now_unix=10.0,
        )
        # Cmd write went through.
        assert out["enabled"] is False
        # Only the first (cmd) write made it into ``fake.puts``; the
        # audit write raised before append.
        assert len(fake.puts) == 1
        assert fake.puts[0][0] == dumps_gate.DUMPS_ENABLED_KEY

    def test_default_actor_falls_back_to_anon(self, fake_store) -> None:
        out = dumps_gate.set_dumps_state(
            fake_store, enabled=True, reason="manual flip",
            actor=None, now_unix=3.0,
        )
        assert out["actor"] == "anon"
        # Both writes carry that actor.
        for _, payload in fake_store.puts:
            assert payload.get("actor", payload.get("user")) == "anon"

    def test_reason_is_trimmed_before_write(self, fake_store) -> None:
        out = dumps_gate.set_dumps_state(
            fake_store, enabled=True, reason="   trimmed me   ",
            actor="ops", now_unix=4.0,
        )
        assert out["reason"] == "trimmed me"
        cmd_payload = fake_store.puts[0][1]
        assert cmd_payload["reason"] == "trimmed me"

    def test_enabled_coerced_to_bool_in_payload(self, fake_store) -> None:
        # Truthy non-bool input must still write a literal boolean.
        out = dumps_gate.set_dumps_state(
            fake_store, enabled=1, reason="bool", actor="ops",
            now_unix=5.0,
        )
        assert out["enabled"] is True
        cmd_payload = fake_store.puts[0][1]
        assert cmd_payload["enabled"] is True


# ---------------------------------------------------------------------------
# list_recent_toggles
# ---------------------------------------------------------------------------


class TestListRecentToggles:
    def _write_n(self, fake, n: int, *, base_ts: float = 1000.0) -> None:
        for i in range(n):
            dumps_gate.set_dumps_state(
                fake,
                enabled=(i % 2 == 0),
                reason=f"flip-{i}",
                actor=f"user-{i}",
                host="lxd110h23",
                now_unix=base_ts + i,
            )

    def test_returns_newest_first(self, fake_store) -> None:
        self._write_n(fake_store, 5)
        rows = dumps_gate.list_recent_toggles(fake_store, limit=5)
        assert len(rows) == 5
        # iso_ts descending.
        iso_seq = [r["iso_ts"] for r in rows]
        assert iso_seq == sorted(iso_seq, reverse=True)
        # Newest flip is the last one written (flip-4).
        assert rows[0]["note"] == "flip-4"
        assert rows[-1]["note"] == "flip-0"

    def test_limit_honoured(self, fake_store) -> None:
        self._write_n(fake_store, 10)
        rows = dumps_gate.list_recent_toggles(fake_store, limit=3)
        assert len(rows) == 3
        assert [r["note"] for r in rows] == ["flip-9", "flip-8", "flip-7"]

    def test_no_rows_returns_empty(self, fake_store) -> None:
        rows = dumps_gate.list_recent_toggles(fake_store, limit=5)
        assert rows == []

    def test_etcd_error_returns_empty_no_raise(self) -> None:
        class _BrokenClient:
            def get_prefix(self, prefix: str):
                raise RuntimeError("etcd unreachable")

        class _BrokenStore:
            def get_etcd(self):
                return _BrokenClient()

        rows = dumps_gate.list_recent_toggles(_BrokenStore(), limit=5)
        assert rows == []

    def test_works_through_control_store_lazy_wrapper(self, fake_store) -> None:
        """The dashboard wraps DsaStore in a tiny ControlStore that
        defers the import. dumps_gate must accept that wrapper too —
        same surface as list_recent_audit() in control_store.py uses.
        """
        # Build a ControlStore-shaped duck.
        import control_store                                     # noqa: WPS433
        cs = control_store.ControlStore()
        cs._store = fake_store                                    # type: ignore[attr-defined]
        # Seed with a row via the wrapper.
        dumps_gate.set_dumps_state(
            cs, enabled=False, reason="via ControlStore",
            actor="ops", now_unix=42.0,
        )
        rows = dumps_gate.list_recent_toggles(cs, limit=5)
        assert len(rows) == 1
        assert rows[0]["note"] == "via ControlStore"
        assert rows[0]["val"] == {"enabled": False}


# ---------------------------------------------------------------------------
# Flask route — POST / GET smoke tests
# ---------------------------------------------------------------------------
#
# We import the live app, swap its ``control_store`` for a FakeDsaStore,
# and drive the route with Flask's test client. This exercises the
# confirm-direction check + reason validation + JSON shape end-to-end.


@pytest.fixture()
def app_with_fake_store():
    """Yield (test_client, fake_store) backed by the live Flask app."""
    # Force the dashboard app's import to use a no-op poller; it
    # otherwise spins a background thread that pings real corr nodes.
    with mock.patch.dict(
        os.environ, {"DSA_MONITOR_RFI_POLLER_DISABLED": "1"},
    ):
        # The poller honours this env in some branches; in others it
        # just starts and the thread is a no-op when network is gone.
        import importlib
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        import app as _app                                       # noqa: WPS433

    fake = FakeDsaStore()
    _app.control_store._store = fake                              # type: ignore[attr-defined]
    client = _app.app.test_client()
    yield client, fake


class TestDumpsEnabledRoute:
    def test_get_returns_default_when_missing(self, app_with_fake_store) -> None:
        client, _ = app_with_fake_store
        r = client.get("/control/dumps_enabled")
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["enabled"] is True
        assert j["default"] is True

    def test_get_returns_present_state(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        fake._responses[dumps_gate.DUMPS_ENABLED_KEY] = {
            "enabled": False, "ts": 99.0, "actor": "bob", "reason": "rfi",
        }
        r = client.get("/control/dumps_enabled")
        assert r.status_code == 200
        j = r.get_json()
        assert j["enabled"] is False
        assert j["actor"] == "bob"
        assert j["reason"] == "rfi"
        assert j["default"] is False

    def test_post_happy_suppress_flip(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "reason": "RFI burst — fleet warmup",
            "confirm": "suppress",
            "user": "ops",
        })
        assert r.status_code == 200, r.get_data(as_text=True)
        j = r.get_json()
        assert j["ok"] is True
        assert j["enabled"] is False
        assert j["actor"] == "ops"
        # Two puts: cmd key + audit row.
        keys = [k for k, _ in fake.puts]
        assert dumps_gate.DUMPS_ENABLED_KEY in keys
        assert any(
            k.startswith(dumps_gate.DUMPS_AUDIT_PREFIX) for k in keys
        )

    def test_post_happy_enable_flip(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        # Seed a SUPPRESSED state so the "enable" verb is the right
        # direction.
        fake._responses[dumps_gate.DUMPS_ENABLED_KEY] = {
            "enabled": False, "ts": 1.0, "actor": "ops", "reason": "earlier",
        }
        r = client.post("/control/dumps_enabled", data={
            "enabled": "true",
            "reason": "RFI cleared; resume science",
            "confirm": "enable",
        })
        assert r.status_code == 200, r.get_data(as_text=True)
        j = r.get_json()
        assert j["ok"] is True
        assert j["enabled"] is True

    def test_post_wrong_confirm_direction(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        # Suppress was requested; typed "enable" by mistake — must 400.
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "reason": "test",
            "confirm": "enable",
        })
        assert r.status_code == 400
        j = r.get_json()
        assert j["ok"] is False
        assert "confirm" in j["error"].lower()
        # And no etcd writes happened.
        assert fake.puts == []

    def test_post_missing_confirm(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "reason": "test",
        })
        assert r.status_code == 400
        assert fake.puts == []

    def test_post_empty_reason(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "reason": "   ",
            "confirm": "suppress",
        })
        assert r.status_code == 400
        j = r.get_json()
        assert j["ok"] is False
        assert "reason" in j["error"].lower()
        assert fake.puts == []

    def test_post_missing_reason(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "confirm": "suppress",
        })
        assert r.status_code == 400
        assert fake.puts == []

    def test_post_overlong_reason(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "reason": "x" * (dumps_gate.MAX_REASON_LEN + 5),
            "confirm": "suppress",
        })
        assert r.status_code == 400
        assert fake.puts == []

    def test_post_bad_enabled_value(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "maybe",
            "reason": "x",
            "confirm": "suppress",
        })
        assert r.status_code == 400
        assert fake.puts == []

    def test_post_writes_audit_row_with_expected_payload(
        self, app_with_fake_store,
    ) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/dumps_enabled", data={
            "enabled": "false",
            "reason": "scheduled commissioning window",
            "confirm": "suppress",
            "user": "alice",
        })
        assert r.status_code == 200
        audits = [
            (k, p) for k, p in fake.puts
            if k.startswith(dumps_gate.DUMPS_AUDIT_PREFIX)
        ]
        assert len(audits) == 1
        _, payload = audits[0]
        assert payload["cmd"] == "dumps_toggle"
        assert payload["val"] == {"enabled": False}
        assert payload["note"] == "scheduled commissioning window"
        assert payload["user"] == "alice"

    def test_dumps_audit_endpoint_returns_latest(
        self, app_with_fake_store,
    ) -> None:
        client, _ = app_with_fake_store
        # Drive three flips through the live route.
        for i, (enabled, reason, direction) in enumerate([
            ("false", "first", "suppress"),
            ("true",  "second", "enable"),
            ("false", "third",  "suppress"),
        ]):
            r = client.post("/control/dumps_enabled", data={
                "enabled": enabled,
                "reason": reason,
                "confirm": direction,
            })
            assert r.status_code == 200, r.get_data(as_text=True)
            # Force monotonically-increasing timestamps so the audit
            # ordering test below is stable. The route uses
            # ``time.time()`` so back-to-back calls in the same
            # microsecond would collide — sleep enough to skip past.
            time.sleep(0.002)
        r = client.get("/control/dumps_audit?limit=5")
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        notes = [row["note"] for row in j["rows"]]
        assert notes[:3] == ["third", "second", "first"]
