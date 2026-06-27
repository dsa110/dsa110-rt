"""Unit tests for the dsa_monitor Control tab's M8 voltage-dump toggles
(``tools/dashboard/dsa_monitor/voltage_controls.py``).

Covers the two runtime switches the dashboard exposes:

* ``/cmd/c2/voltages_enabled`` — the C2 voltage-broadcast kill-switch,
  **fail-CLOSED** (missing key ⇒ disabled).
* ``/cmd/c3/flag_only`` — the C3 keep/delete mode, safe default
  ``flag_only=True`` (missing key ⇒ keep-only).

We exercise the get/set/list helpers directly + the four Flask routes
end-to-end (confirm-direction speed-bump, reason validation, audit-row
shape, fail-closed/safe defaults), and pin both keys against the
authoritative copies in the dsart service modules so the duplicated
constants cannot drift.
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

SRC_DIR = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import voltage_controls                                            # noqa: E402


# ---------------------------------------------------------------------------
# Fakes (same shape as test_dumps_gate_dashboard.py)
# ---------------------------------------------------------------------------


class _FakeEtcdMeta:
    def __init__(self, key: str) -> None:
        self.key = key.encode("utf-8") if isinstance(key, str) else key


class _FakeEtcdClient:
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
    def __init__(self, get_dict_responses: dict[str, Any] | None = None):
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self._responses = dict(get_dict_responses or {})
        self._kv_for_etcd: dict[str, Any] = {}

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        self.puts.append((key, dict(payload)))
        self._kv_for_etcd[key] = dict(payload)

    def get_dict(self, key: str) -> Any:
        self.gets.append(key)
        return self._responses.get(key)

    def get_etcd(self):
        return _FakeEtcdClient(self._kv_for_etcd)


@pytest.fixture()
def fake_store():
    return FakeDsaStore()


# ---------------------------------------------------------------------------
# Constants pinned against the dsart-side authoritative copies
# ---------------------------------------------------------------------------


def test_voltages_key_pinned_to_coincidencer_copy() -> None:
    from dsart.services.coincidencer import VOLTAGES_ENABLED_KEY as SVC_KEY
    assert voltage_controls.VOLTAGES_ENABLED_KEY == SVC_KEY


def test_c3_flag_only_key_pinned_to_c3_copy() -> None:
    from dsart.services.c3 import C3_FLAG_ONLY_KEY as SVC_KEY
    assert voltage_controls.C3_FLAG_ONLY_KEY == SVC_KEY


def test_audit_prefixes_under_control_namespace() -> None:
    for p in (
        voltage_controls.VOLTAGES_AUDIT_PREFIX,
        voltage_controls.C3_MODE_AUDIT_PREFIX,
    ):
        assert p.startswith("/mon/audit/control/")
        assert p.endswith("/")


# ---------------------------------------------------------------------------
# voltages_enabled (fail-CLOSED)
# ---------------------------------------------------------------------------


class TestVoltagesState:
    def test_missing_key_fail_closed_default(self, fake_store) -> None:
        out = voltage_controls.get_voltages_state(fake_store)
        assert out == {
            "enabled": False, "ts": None, "actor": None,
            "reason": None, "default": True,
        }
        assert fake_store.gets == [voltage_controls.VOLTAGES_ENABLED_KEY]

    def test_present_key(self) -> None:
        fake = FakeDsaStore(get_dict_responses={
            voltage_controls.VOLTAGES_ENABLED_KEY: {
                "enabled": True, "ts": 5.0, "actor": "alice", "reason": "frb",
            },
        })
        out = voltage_controls.get_voltages_state(fake)
        assert out == {
            "enabled": True, "ts": 5.0, "actor": "alice",
            "reason": "frb", "default": False,
        }

    def test_malformed_is_fail_closed(self) -> None:
        for bogus in ({"foo": 1}, [1], "x", 7, None):
            fake = FakeDsaStore(get_dict_responses={
                voltage_controls.VOLTAGES_ENABLED_KEY: bogus,
            })
            out = voltage_controls.get_voltages_state(fake)
            assert out["enabled"] is False and out["default"] is True

    def test_set_writes_cmd_and_audit(self, fake_store) -> None:
        out = voltage_controls.set_voltages_state(
            fake_store, enabled=True, reason="enable for FRB follow-up",
            actor="alice", host="lxd110h23", now_unix=1700000000.5,
        )
        assert out["enabled"] is True
        assert out["audit_key"].startswith(
            voltage_controls.VOLTAGES_AUDIT_PREFIX)
        cmd = [p for p in fake_store.puts
               if p[0] == voltage_controls.VOLTAGES_ENABLED_KEY]
        audit = [p for p in fake_store.puts
                 if p[0].startswith(voltage_controls.VOLTAGES_AUDIT_PREFIX)]
        assert len(cmd) == 1 and len(audit) == 1
        assert cmd[0][1]["enabled"] is True
        assert audit[0][1]["cmd"] == "voltages_toggle"
        assert audit[0][1]["namespace"] == "c2.voltages_toggle"
        assert audit[0][1]["val"] == {"enabled": True}

    def test_empty_reason_raises_no_write(self, fake_store) -> None:
        for bogus in ("", "  ", None):
            with pytest.raises(ValueError, match="reason"):
                voltage_controls.set_voltages_state(
                    fake_store, enabled=True, reason=bogus)
        assert fake_store.puts == []

    def test_list_newest_first(self, fake_store) -> None:
        for i in range(4):
            voltage_controls.set_voltages_state(
                fake_store, enabled=(i % 2 == 0), reason=f"v-{i}",
                actor="ops", now_unix=1000.0 + i)
        rows = voltage_controls.list_recent_voltage_toggles(
            fake_store, limit=3)
        assert [r["note"] for r in rows] == ["v-3", "v-2", "v-1"]


# ---------------------------------------------------------------------------
# c3 keep/delete mode (safe default: flag_only=True)
# ---------------------------------------------------------------------------


class TestC3ModeState:
    def test_missing_key_safe_default_keep_only(self, fake_store) -> None:
        out = voltage_controls.get_c3_mode_state(fake_store)
        assert out == {
            "flag_only": True, "ts": None, "actor": None,
            "reason": None, "default": True,
        }

    def test_present_delete_mode(self) -> None:
        fake = FakeDsaStore(get_dict_responses={
            voltage_controls.C3_FLAG_ONLY_KEY: {
                "flag_only": False, "ts": 1.0, "actor": "bob", "reason": "soak ok",
            },
        })
        out = voltage_controls.get_c3_mode_state(fake)
        assert out["flag_only"] is False and out["default"] is False

    def test_set_writes_cmd_and_audit(self, fake_store) -> None:
        out = voltage_controls.set_c3_mode_state(
            fake_store, flag_only=False, reason="flag-first soak clean",
            actor="bob", now_unix=2000.0)
        assert out["flag_only"] is False
        cmd = [p for p in fake_store.puts
               if p[0] == voltage_controls.C3_FLAG_ONLY_KEY]
        audit = [p for p in fake_store.puts
                 if p[0].startswith(voltage_controls.C3_MODE_AUDIT_PREFIX)]
        assert len(cmd) == 1 and len(audit) == 1
        assert cmd[0][1]["flag_only"] is False
        assert audit[0][1]["cmd"] == "c3_mode_toggle"
        assert audit[0][1]["val"] == {"flag_only": False}


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_with_fake_store():
    with mock.patch.dict(
        os.environ, {"DSA_MONITOR_RFI_POLLER_DISABLED": "1"},
    ):
        import importlib
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])
        import app as _app                                         # noqa: WPS433

    fake = FakeDsaStore()
    _app.control_store._store = fake                              # type: ignore[attr-defined]
    client = _app.app.test_client()
    yield client, fake


class TestVoltagesRoute:
    def test_get_default_disabled(self, app_with_fake_store) -> None:
        client, _ = app_with_fake_store
        j = client.get("/control/voltages_enabled").get_json()
        assert j["ok"] is True
        assert j["enabled"] is False
        assert j["default"] is True

    def test_post_enable_happy(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/voltages_enabled", data={
            "enabled": "true", "reason": "FRB follow-up", "confirm": "enable",
            "user": "ops",
        })
        assert r.status_code == 200, r.get_data(as_text=True)
        j = r.get_json()
        assert j["ok"] is True and j["enabled"] is True
        keys = [k for k, _ in fake.puts]
        assert voltage_controls.VOLTAGES_ENABLED_KEY in keys

    def test_post_wrong_confirm(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/voltages_enabled", data={
            "enabled": "true", "reason": "x", "confirm": "disable",
        })
        assert r.status_code == 400
        assert fake.puts == []

    def test_post_disable_requires_disable_word(
        self, app_with_fake_store,
    ) -> None:
        client, fake = app_with_fake_store
        fake._responses[voltage_controls.VOLTAGES_ENABLED_KEY] = {
            "enabled": True, "ts": 1.0, "actor": "o", "reason": "r",
        }
        ok = client.post("/control/voltages_enabled", data={
            "enabled": "false", "reason": "done", "confirm": "disable",
        })
        assert ok.status_code == 200, ok.get_data(as_text=True)
        assert ok.get_json()["enabled"] is False

    def test_audit_endpoint(self, app_with_fake_store) -> None:
        client, _ = app_with_fake_store
        for reason, direction, enabled in [
            ("first", "enable", "true"),
            ("second", "disable", "false"),
        ]:
            client.post("/control/voltages_enabled", data={
                "enabled": enabled, "reason": reason, "confirm": direction,
            })
            time.sleep(0.002)
        j = client.get("/control/voltages_audit?limit=5").get_json()
        assert j["ok"] is True
        assert [r["note"] for r in j["rows"]][:2] == ["second", "first"]


class TestC3ModeRoute:
    def test_get_default_keep_only(self, app_with_fake_store) -> None:
        client, _ = app_with_fake_store
        j = client.get("/control/c3_mode").get_json()
        assert j["ok"] is True
        assert j["flag_only"] is True
        assert j["default"] is True

    def test_post_enable_delete_happy(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/c3_mode", data={
            "flag_only": "false", "reason": "soak clean", "confirm": "delete",
            "user": "ops",
        })
        assert r.status_code == 200, r.get_data(as_text=True)
        j = r.get_json()
        assert j["ok"] is True and j["flag_only"] is False
        assert voltage_controls.C3_FLAG_ONLY_KEY in [k for k, _ in fake.puts]

    def test_post_delete_wrong_confirm(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        r = client.post("/control/c3_mode", data={
            "flag_only": "false", "reason": "x", "confirm": "keep",
        })
        assert r.status_code == 400
        assert fake.puts == []

    def test_post_back_to_keep(self, app_with_fake_store) -> None:
        client, fake = app_with_fake_store
        fake._responses[voltage_controls.C3_FLAG_ONLY_KEY] = {
            "flag_only": False, "ts": 1.0, "actor": "o", "reason": "r",
        }
        r = client.post("/control/c3_mode", data={
            "flag_only": "true", "reason": "pause deletes", "confirm": "keep",
        })
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json()["flag_only"] is True

    def test_audit_endpoint(self, app_with_fake_store) -> None:
        client, _ = app_with_fake_store
        client.post("/control/c3_mode", data={
            "flag_only": "false", "reason": "go-delete", "confirm": "delete",
        })
        j = client.get("/control/c3_mode_audit?limit=5").get_json()
        assert j["ok"] is True
        assert j["rows"][0]["note"] == "go-delete"
        assert j["rows"][0]["val"] == {"flag_only": False}
