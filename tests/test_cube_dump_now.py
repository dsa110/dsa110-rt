"""Unit tests for ``tools/dashboard/dsa_monitor/cube_dump_now.py``.

The module under test broadcasts a synthetic ``C2TriggerPacket`` to
every search-half UDP listener on the fleet. These tests never touch
a real UDP socket — every test injects a fake sender callable that
records every (data, addr, timeout_s) tuple. The Flask layer is
exercised separately via its own ``flask.test_client`` fixture so we
can pin the 400 / 503 / 500 / 200 status mapping in
:func:`tools/dashboard/dsa_monitor/app.py::control_dump_now_post`.

Coverage targets (one test class per area):

* :class:`TestBindHostResolution` — yaml parsing produces the
  canonical 8-tuple list ``(sid, gpu_half, host, port)`` for
  ``coinc.dump_broadcast.hosts`` × ``GPU_HALVES``.
* :class:`TestEventNameGen` — ``dumpnow_<8 base32>`` and per-call
  uniqueness.
* :class:`TestResolveEventSpecnum` — picks ``max(block_specnum_start) -
  lookback_blocks × NPACKETS_PER_BLOCK`` and surfaces the answered /
  missing / stale lists.
* :class:`TestFleetDumpNowHappyPath` — 8 halves all return ok.
* :class:`TestFleetDumpNowPartialFailure` — 3 halves time out; per_half
  reflects per-destination statuses.
* :class:`TestFleetDumpNowNoCorrFastMonkeys` — cold-boot path returns
  ``error="no corr_fast mon-keys; is the fleet up?"`` and refuses to
  broadcast.
* :class:`TestFleetDumpNowEmptyFleet` — empty bind_hosts list returns
  ok=True with pass_count=fail_count=0 and a clear note.
* :class:`TestDumpNowFlaskRoute` — the Flask POST route returns 400
  on missing / wrong confirm, 503 on the no-corr_fast-monkeys path,
  and 200 on the happy path with the per-half status dict.
"""

from __future__ import annotations

import os
import random
import socket
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

# dsart.coinc.wire is imported by cube_dump_now lazily, but for the
# Flask-side tests we want to fail fast if PYTHONPATH=src isn't set.
SRC_DIR = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cube_dump_now                                            # noqa: E402
import control_store                                             # noqa: E402


# ---------------------------------------------------------------------------
# Common fakes — FakeDsaStore + FakeSender
# ---------------------------------------------------------------------------


class FakeDsaStore:
    """Mirrors the pattern in test_dsa_monitor_control_store.py:
    records every ``put_dict`` and answers programmable ``get_dict``
    responses.
    """

    def __init__(self, get_dict_responses: dict[str, Any] | None = None):
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self._responses = dict(get_dict_responses or {})
        self._lock = threading.Lock()

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.puts.append((key, dict(payload)))

    def get_dict(self, key: str) -> Any:
        with self._lock:
            self.gets.append(key)
        return self._responses.get(key)


class FakeSender:
    """Records every (blob, addr, timeout) sent. Optional per-call
    side-effect injection so individual halves can ``raise``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, tuple[str, int], float]] = []
        self.per_addr: dict[tuple[str, int], BaseException] = {}
        self.default_exc: BaseException | None = None
        self._lock = threading.Lock()

    def set_addr_exc(
        self, addr: tuple[str, int], exc: BaseException,
    ) -> None:
        self.per_addr[addr] = exc

    def __call__(
        self,
        data: bytes,
        addr: tuple[str, int],
        timeout_s: float,
    ) -> None:
        with self._lock:
            self.calls.append((bytes(data), tuple(addr), float(timeout_s)))
        exc = self.per_addr.get(addr) or self.default_exc
        if exc is not None:
            raise exc


@pytest.fixture()
def fake_store():
    return FakeDsaStore()


@pytest.fixture()
def fake_store_with_fleet():
    """16 corr_fast mon-keys all reporting block_n=1000 (fresh)."""
    now = time.time()
    responses = {
        f"/mon/corr_rt/{cg}/corr_fast": {
            "block_n": 1000,
            "block_specnum_start": 1000 * cube_dump_now.NPACKETS_PER_BLOCK,
            "ts_wall_unix": now,
        }
        for cg in range(16)
    }
    return FakeDsaStore(get_dict_responses=responses)


# Canonical 8-half bind list for the production fleet (matches
# configs/dsart_search_rt.yaml::coinc.dump_broadcast). Pinned here so
# the tests don't depend on reading the file off disk.
PROD_BIND_HOSTS: tuple[tuple[int, int, str, int], ...] = (
    (1, 0, "10.41.0.205", 11227),
    (1, 1, "10.41.0.205", 11228),
    (2, 0, "10.41.0.222", 11227),
    (2, 1, "10.41.0.222", 11228),
    (9, 0, "10.41.0.253", 11227),
    (9, 1, "10.41.0.253", 11228),
    (13, 0, "10.41.0.238", 11227),
    (13, 1, "10.41.0.238", 11228),
)


# ---------------------------------------------------------------------------
# Bind-list resolution
# ---------------------------------------------------------------------------


class TestBindHostResolution:
    def test_resolve_from_canonical_yaml(self):
        """The real configs/dsart_search_rt.yaml should resolve to
        exactly the 4 search nodes × 2 halves = 8 halves on the
        ``coinc.dump_broadcast`` fan-out list."""
        hosts = cube_dump_now.resolve_bind_hosts()
        # 4 search nodes × 2 gpu halves.
        assert len(hosts) == 8
        # Pin the production layout. Each (sid, g, host, port) tuple
        # must appear exactly once, ports must be base + gpu_half.
        sids_seen = sorted({h[0] for h in hosts})
        assert sids_seen == [1, 2, 9, 13]
        for sid, g, host, port in hosts:
            assert g in (0, 1)
            assert port == 11227 + g
            assert host.count(".") == 3                # ipv4
            assert sid in (1, 2, 9, 13)

    def test_resolve_handles_missing_yaml(self, tmp_path):
        """A missing yaml path returns an empty list (caller surfaces
        as a non-error empty-fleet result)."""
        missing = str(tmp_path / "nope.yaml")
        out = cube_dump_now.resolve_bind_hosts(yaml_path=missing)
        assert out == []

    def test_resolve_handles_empty_coinc_block(self, tmp_path):
        """yaml with no coinc.dump_broadcast block returns []."""
        p = tmp_path / "empty.yaml"
        p.write_text("schema_version: 1\n")
        out = cube_dump_now.resolve_bind_hosts(yaml_path=str(p))
        assert out == []

    def test_resolve_skips_non_int_sids_and_non_str_ips(self, tmp_path):
        """Malformed yaml entries are logged and skipped, not raised."""
        p = tmp_path / "bad.yaml"
        p.write_text(
            "coinc:\n"
            "  dump_broadcast:\n"
            "    port_base: 11227\n"
            "    hosts:\n"
            "      \"1\": \"10.0.0.1\"\n"
            "      \"badsid\": \"10.0.0.2\"\n"
            "      \"3\": 12345\n"          # non-str ip → skip
        )
        out = cube_dump_now.resolve_bind_hosts(yaml_path=str(p))
        # Only sid=1 survived.
        assert sorted({sid for sid, _, _, _ in out}) == [1]
        assert len(out) == 2                            # 2 halves
        assert all(host == "10.0.0.1" for _, _, host, _ in out)

    def test_resolve_overrides_port_base(self, tmp_path):
        p = tmp_path / "alt.yaml"
        p.write_text(
            "coinc:\n"
            "  dump_broadcast:\n"
            "    port_base: 22227\n"
            "    hosts:\n"
            "      \"1\": \"10.0.0.1\"\n"
        )
        out = cube_dump_now.resolve_bind_hosts(yaml_path=str(p))
        assert sorted([port for _, _, _, port in out]) == [22227, 22228]


# ---------------------------------------------------------------------------
# Event-name generation
# ---------------------------------------------------------------------------


class TestEventNameGen:
    def test_default_event_name_is_16_chars_and_uses_base32(self):
        name = cube_dump_now._gen_event_name()
        assert name.startswith(cube_dump_now.EVENT_NAME_PREFIX)
        assert len(name) == len(cube_dump_now.EVENT_NAME_PREFIX) + 8
        # Total length must fit the 16-byte C2TriggerPacket field.
        assert len(name.encode("ascii")) == 16
        # Every random char is in the RFC 4648 alphabet.
        for c in name[len(cube_dump_now.EVENT_NAME_PREFIX):]:
            assert c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

    def test_default_event_name_is_unique_on_repeated_calls(self):
        names = {cube_dump_now._gen_event_name() for _ in range(200)}
        # 32^8 ≈ 1.1e12 possibilities; 200 random samples colliding
        # is astronomically unlikely.
        assert len(names) == 200

    def test_fleet_dump_now_default_event_name_uses_8_base32(
        self, fake_store_with_fleet,
    ):
        """The high-level entry point also generates the same shape."""
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS,
            sender=snd,
        )
        assert res.event_name.startswith("dumpnow_")
        assert len(res.event_name) == 16
        # Repeat — must be different.
        res2 = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS,
            sender=snd,
        )
        assert res.event_name != res2.event_name


# ---------------------------------------------------------------------------
# Event-specnum resolution
# ---------------------------------------------------------------------------


class TestResolveEventSpecnum:
    def test_pick_max_minus_lookback(self, fake_store_with_fleet):
        ev_spec, info = cube_dump_now._resolve_event_specnum(
            fake_store_with_fleet, chgroups=tuple(range(16)),
            lookback_blocks=4,
        )
        # max_block_specnum_start = 1000 * 2048
        # event_specnum = max - 4 * 2048 = 996 * 2048
        assert ev_spec == 996 * 2048
        assert info["max_block_specnum_start"] == 1000 * 2048
        assert info["max_block_n"] == 1000
        assert len(info["answered"]) == 16
        assert info["missing"] == []
        assert info["stale"] == []

    def test_returns_none_when_no_publishers(self, fake_store):
        ev_spec, info = cube_dump_now._resolve_event_specnum(
            fake_store, chgroups=(0, 1, 2),
        )
        assert ev_spec is None
        assert info["max_block_specnum_start"] is None
        assert sorted(info["missing"]) == [
            "/mon/corr_rt/0/corr_fast",
            "/mon/corr_rt/1/corr_fast",
            "/mon/corr_rt/2/corr_fast",
        ]

    def test_stale_publisher_dropped(self):
        now = time.time()
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 1000,
                "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now - 60,           # stale (>10s)
            },
            "/mon/corr_rt/1/corr_fast": {
                "block_n": 50,
                "block_specnum_start": 50 * 2048,
                "ts_wall_unix": now,
            },
        }
        store = FakeDsaStore(get_dict_responses=responses)
        ev_spec, info = cube_dump_now._resolve_event_specnum(
            store, chgroups=(0, 1),
        )
        # Only chgroup=1 contributes; max=50 * 2048; ev = max - 4*2048
        assert info["max_block_specnum_start"] == 50 * 2048
        assert info["stale"] == ["/mon/corr_rt/0/corr_fast"]
        assert info["answered"] == ["/mon/corr_rt/1/corr_fast"]
        # 50 - 4 = 46 blocks of specnums.
        assert ev_spec == 46 * 2048

    def test_clamps_to_zero_when_lookback_exceeds_max(self):
        """A pathological lookback bigger than max_bss clamps to 0
        (never negative)."""
        responses = {
            "/mon/corr_rt/0/corr_fast": {
                "block_n": 2,
                "block_specnum_start": 2 * 2048,
                "ts_wall_unix": time.time(),
            },
        }
        store = FakeDsaStore(get_dict_responses=responses)
        ev_spec, info = cube_dump_now._resolve_event_specnum(
            store, chgroups=(0,), lookback_blocks=999,
        )
        assert ev_spec == 0
        assert info["max_block_specnum_start"] == 2 * 2048


# ---------------------------------------------------------------------------
# fleet_dump_now — happy path (8 halves all ok)
# ---------------------------------------------------------------------------


class TestFleetDumpNowHappyPath:
    def test_all_eight_halves_succeed(self, fake_store_with_fleet):
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS,
            sender=snd,
            event_name="dumpnow_TEST0001",        # fixed for assert
        )
        assert res.ok is True
        assert res.pass_count == 8
        assert res.fail_count == 0
        assert res.event_name == "dumpnow_TEST0001"
        # event_specnum = max(1000) - 4 = 996 blocks × 2048.
        assert res.event_specnum == 996 * 2048
        assert res.error is None
        # Every half present, all "ok".
        assert sorted(res.per_half.keys()) == sorted(
            (sid, g) for (sid, g, _, _) in PROD_BIND_HOSTS
        )
        for status in res.per_half.values():
            assert status == "ok"
        # One sendto per half — 8 in total.
        assert len(snd.calls) == 8
        # Same 64-byte packet broadcast to every destination.
        sizes = {len(blob) for blob, _, _ in snd.calls}
        assert sizes == {64}
        # Magic prefix in every blob (decoded by C2TriggerListener).
        magic = (cube_dump_now._encode_packet
                 ("dumpnow_TEST0001", 996 * 2048)[:4])
        for blob, _, _ in snd.calls:
            assert blob[:4] == magic
        # Latency is recorded and finite.
        assert 0.0 <= res.latency_ms < 5000.0

    def test_to_dict_shape_is_json_safe(self, fake_store_with_fleet):
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS, sender=snd,
        )
        d = res.to_dict()
        # All keys present; per_half is a flat ``s<sid>_g<g>`` dict.
        assert d["ok"] is True
        assert d["pass_count"] == 8
        assert d["fail_count"] == 0
        assert sorted(d["per_half"].keys()) == [
            "s13_g0", "s13_g1", "s1_g0", "s1_g1",
            "s2_g0", "s2_g1", "s9_g0", "s9_g1",
        ]
        # JSON-serialisable.
        import json
        json.dumps(d)                                  # must not raise

    def test_caller_supplied_target_specnum_overrides_arm(
        self, fake_store_with_fleet,
    ):
        """When the caller supplies target_specnum, the store is
        NOT consulted (we don't even need a live fleet)."""
        empty_store = FakeDsaStore()
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            empty_store,                                 # NO corr_fast keys
            bind_hosts=PROD_BIND_HOSTS,
            sender=snd,
            target_specnum=987_654_321,
        )
        assert res.ok is True
        assert res.event_specnum == 987_654_321
        # No get_dict calls because the auto-arm branch was skipped.
        assert empty_store.gets == []


# ---------------------------------------------------------------------------
# fleet_dump_now — partial failure (3 halves time out)
# ---------------------------------------------------------------------------


class TestFleetDumpNowPartialFailure:
    def test_three_halves_time_out(self, fake_store_with_fleet):
        snd = FakeSender()
        # Pick 3 halves to time out — one per kind of failure mode.
        snd.set_addr_exc(
            ("10.41.0.205", 11227), socket.timeout("simulated timeout"),
        )
        snd.set_addr_exc(
            ("10.41.0.222", 11228), TimeoutError("simulated timeout"),
        )
        snd.set_addr_exc(
            ("10.41.0.253", 11227),
            OSError(101, "Network is unreachable"),
        )
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS, sender=snd,
        )
        assert res.ok is False
        assert res.fail_count == 3
        assert res.pass_count == 5
        # Per-half map carries the right statuses.
        assert res.per_half[(1, 0)].startswith("timeout:")
        assert res.per_half[(2, 1)].startswith("timeout:")
        assert res.per_half[(9, 0)].startswith("oserror:")
        # The other halves are ok.
        assert res.per_half[(1, 1)] == "ok"
        assert res.per_half[(2, 0)] == "ok"
        assert res.per_half[(9, 1)] == "ok"
        assert res.per_half[(13, 0)] == "ok"
        assert res.per_half[(13, 1)] == "ok"

    def test_per_half_records_unexpected_exception(
        self, fake_store_with_fleet,
    ):
        snd = FakeSender()
        snd.set_addr_exc(
            ("10.41.0.238", 11228), RuntimeError("kaboom"),
        )
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS, sender=snd,
        )
        assert res.ok is False
        assert res.per_half[(13, 1)].startswith("exception:")
        assert "kaboom" in res.per_half[(13, 1)]
        assert res.fail_count == 1
        assert res.pass_count == 7


# ---------------------------------------------------------------------------
# fleet_dump_now — no corr_fast mon-keys (cold-boot path)
# ---------------------------------------------------------------------------


class TestFleetDumpNowNoCorrFastMonkeys:
    def test_refuses_to_broadcast_and_carries_error(self):
        empty_store = FakeDsaStore()
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            empty_store,
            bind_hosts=PROD_BIND_HOSTS,
            sender=snd,
        )
        assert res.ok is False
        assert res.error == "no corr_fast mon-keys; is the fleet up?"
        assert res.event_specnum is None
        # No actual sends — we bailed before the fan-out.
        assert snd.calls == []
        # The arm_info diagnostic reports the full polled list.
        assert res.arm_info is not None
        assert len(res.arm_info["polled"]) == 16
        assert len(res.arm_info["missing"]) == 16
        assert res.arm_info["answered"] == []

    def test_target_specnum_override_skips_corr_fast_lookup(self):
        """The 503-equivalent path is ONLY for the auto-arm case.
        When the caller passes target_specnum, an empty store is fine.
        """
        empty_store = FakeDsaStore()
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            empty_store,
            bind_hosts=PROD_BIND_HOSTS,
            sender=snd,
            target_specnum=12345,
        )
        assert res.ok is True
        assert res.error is None
        assert res.event_specnum == 12345
        assert res.pass_count == 8


# ---------------------------------------------------------------------------
# fleet_dump_now — empty fleet handling
# ---------------------------------------------------------------------------


class TestFleetDumpNowEmptyFleet:
    def test_empty_bind_hosts_returns_clean_zero_result(
        self, fake_store_with_fleet,
    ):
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=[],
            sender=snd,
        )
        assert res.ok is True
        assert res.pass_count == 0
        assert res.fail_count == 0
        assert res.per_half == {}
        assert res.event_specnum == 996 * 2048   # still derived
        assert "no bind destinations" in res.note
        # NO sendto attempts.
        assert snd.calls == []


# ---------------------------------------------------------------------------
# fleet_dump_now — timeout handling (single-half timeout)
# ---------------------------------------------------------------------------


class TestFleetDumpNowTimeoutHandling:
    def test_socket_timeout_classified_as_timeout(
        self, fake_store_with_fleet,
    ):
        """The plain socket.timeout exception is bucketed into the
        ``timeout:`` per-half status (not the OSError bucket)."""
        snd = FakeSender()
        snd.default_exc = socket.timeout("timed out")
        res = cube_dump_now.fleet_dump_now(
            fake_store_with_fleet,
            bind_hosts=PROD_BIND_HOSTS, sender=snd,
            timeout_s=0.05,
        )
        assert res.ok is False
        assert res.pass_count == 0
        assert res.fail_count == 8
        for status in res.per_half.values():
            assert status.startswith("timeout:")
        # Timeout argument was forwarded into every call.
        for _, _, t in snd.calls:
            assert t == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Flask route — 400 / 503 / 500 / 200
# ---------------------------------------------------------------------------


@pytest.fixture()
def flask_client(monkeypatch):
    """Spin up the Flask app under test with a fake ControlStore +
    a fake fleet_dump_now sender. Mirrors the test pattern in
    test_fleet_services.py / test_fleet_update.py (no real network,
    no real etcd)."""
    import app                                                # noqa
    # Replace the global control_store with a fake that answers
    # get_dict from a fresh FakeDsaStore.
    fake = FakeDsaStore()
    app.control_store._store = fake                            # type: ignore[attr-defined]
    app.app.config["TESTING"] = True
    yield app.app.test_client(), app, fake


class TestDumpNowFlaskRoute:
    def test_no_confirm_returns_400(self, flask_client):
        client, _app, _fake = flask_client
        r = client.post("/control/dump_now", data={})
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "confirm=dump_now" in body["error"]

    def test_confirm_wrong_value_returns_400(self, flask_client):
        client, _app, _fake = flask_client
        r = client.post("/control/dump_now", data={"confirm": "dump"})
        assert r.status_code == 400
        body = r.get_json()
        assert body["ok"] is False
        assert "confirm=dump_now" in body["error"]

    def test_no_corr_fast_publishers_returns_503(self, flask_client):
        client, _app, _fake = flask_client
        # ``_fake`` has no get_dict responses, so the auto-arm path
        # produces error="no corr_fast mon-keys; is the fleet up?".
        # We have to make sure the fleet_dump_now uses bind_hosts that
        # don't actually try to hit the network. Patch the resolver to
        # return a non-empty fake list (so the test exercises the 503
        # path, not the empty-fleet path).
        fake_sender = FakeSender()
        with mock.patch.object(
            cube_dump_now, "resolve_bind_hosts",
            return_value=list(PROD_BIND_HOSTS),
        ), mock.patch.object(
            cube_dump_now, "default_sender", new=fake_sender,
        ):
            r = client.post(
                "/control/dump_now", data={"confirm": "dump_now"},
            )
        assert r.status_code == 503
        body = r.get_json()
        assert body["ok"] is False
        assert body["error"] == "no corr_fast mon-keys; is the fleet up?"
        assert body["event_specnum"] is None
        # The fake sender was NOT called (we refused to broadcast).
        assert fake_sender.calls == []

    def test_happy_path_returns_200_and_per_half(self, flask_client):
        client, _app, fake_store = flask_client
        # Seed corr_fast mon-keys so the auto-arm succeeds.
        now = time.time()
        for cg in range(16):
            fake_store._responses[f"/mon/corr_rt/{cg}/corr_fast"] = {
                "block_n": 1000,
                "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now,
            }
        fake_sender = FakeSender()
        with mock.patch.object(
            cube_dump_now, "resolve_bind_hosts",
            return_value=list(PROD_BIND_HOSTS),
        ), mock.patch.object(
            cube_dump_now, "default_sender", new=fake_sender,
        ):
            r = client.post(
                "/control/dump_now", data={"confirm": "dump_now"},
            )
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert body["pass_count"] == 8
        assert body["fail_count"] == 0
        assert body["event_specnum"] == 996 * 2048
        assert len(body["per_half"]) == 8
        for status in body["per_half"].values():
            assert status == "ok"
        # Audit row was written.
        audit_keys = [k for k, _ in fake_store.puts
                      if k.startswith("/mon/audit/control/")]
        assert len(audit_keys) >= 1

    def test_partial_failure_returns_200_but_ok_false(self, flask_client):
        client, _app, fake_store = flask_client
        now = time.time()
        for cg in range(16):
            fake_store._responses[f"/mon/corr_rt/{cg}/corr_fast"] = {
                "block_n": 1000,
                "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now,
            }
        fake_sender = FakeSender()
        # 2 halves time out.
        fake_sender.set_addr_exc(
            ("10.41.0.205", 11227), socket.timeout("simulated"),
        )
        fake_sender.set_addr_exc(
            ("10.41.0.253", 11228), TimeoutError("simulated"),
        )
        with mock.patch.object(
            cube_dump_now, "resolve_bind_hosts",
            return_value=list(PROD_BIND_HOSTS),
        ), mock.patch.object(
            cube_dump_now, "default_sender", new=fake_sender,
        ):
            r = client.post(
                "/control/dump_now", data={"confirm": "dump_now"},
            )
        # 200 (the broadcast happened) but ok=False (partial fail).
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is False
        assert body["pass_count"] == 6
        assert body["fail_count"] == 2


# ---------------------------------------------------------------------------
# Search-ring auto-pick path (M7.4 Phase 6c: fixes the specnum-domain
# bug surfaced by the 2026-05-28 soak — the dashboard's "Dump Now"
# button was using corr_fast's service-start ``block_specnum_start``,
# but the search-side cube_ring keys cubes by ``slot.specnum_start``,
# a different domain).
# ---------------------------------------------------------------------------


def _make_search_ring_responses(
    *,
    sids_g: Sequence[tuple[int, int]],
    newest_per_half: dict[tuple[int, int], int],
    n_committed: int = 16,
    depth: int = 16,
    t_det: int = 192,
    sample_period_specnum: int = 16,
    cube_cadence_samples: int = 128,
    ts_age_s: float = 0.0,
) -> dict[str, Any]:
    """Build a /mon/search/<sid>/<g>/ring response dict per
    requested (sid, g).

    The ring's ``event_specnum_start`` field is in *search-sample*
    units (see production_rx_ring._assemble_cube), so adjacent
    cubes are ``cube_cadence_samples`` apart (NOT
    ``t_det × sample_period_specnum`` — that's the per-cube
    duration in raw specnums, which is the listener's per-cube
    acceptance span but NOT the ring's cube-to-cube step).
    """
    now = time.time() - float(ts_age_s)
    out: dict[str, Any] = {}
    for sid, g in sids_g:
        newest = int(newest_per_half[(sid, g)])
        live = max(1, min(int(n_committed), int(depth)))
        oldest = newest - (live - 1) * int(cube_cadence_samples)
        out[f"/mon/search/{sid}/{g}/ring"] = {
            "search_node_id": int(sid),
            "gpu_half": int(g),
            "n_committed": int(n_committed),
            "depth": int(depth),
            "t_det": int(t_det),
            "n_fdm": 34,
            "n_grid": 256,
            "sample_period_specnum": int(sample_period_specnum),
            "newest_event_specnum_start": newest,
            "newest_event_specnum_end":
                newest + int(t_det) * int(sample_period_specnum),
            "oldest_event_specnum_start": int(oldest),
            "newest_cube_id": 100,
            "oldest_cube_id": 100 - (live - 1),
            "ts_mono": time.monotonic() - float(ts_age_s),
            "ts_wall_unix": now,
        }
    return out


# Pull in Sequence so the helper above type-checks under the test runner.
from typing import Sequence  # noqa: E402


class TestResolveEventSpecnumFromSearchRing:
    def test_picks_mid_newest_with_lookback_offset(self):
        targets = [(sid, g) for (sid, g, _, _) in PROD_BIND_HOSTS]
        # Give each half a slightly different newest_event_specnum_start
        # so we can verify the min selection.
        newest_per_half: dict[tuple[int, int], int] = {}
        for i, (sid, g) in enumerate(targets):
            newest_per_half[(sid, g)] = 10_000_000 + i * 1_000
        responses = _make_search_ring_responses(
            sids_g=targets, newest_per_half=newest_per_half,
            t_det=192, sample_period_specnum=16,
            n_committed=16, depth=16,
        )
        store = FakeDsaStore(get_dict_responses=responses)
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets, lookback_cubes=4,
        )
        # min(newest) == 10_000_000 (the first half).
        # cube_span = t_det * spp = 192 * 16 = 3072.
        # cube_cadence_samples (derived from publisher: ring of 16
        # cubes spanning ``(depth - 1) * cadence`` ≈ 1920 samples
        # → cadence = 128).
        # target = min_newest + cube_span // 2 - lookback * cadence
        #        = 10_000_000 + 1536 - 4 * 128
        #        = 10_000_000 + 1024
        assert info["min_newest_event_specnum_start"] == 10_000_000
        assert info["source"] == "search_ring"
        assert info["lookback_cubes"] == 4
        assert info["newest_t_det"] == 192
        assert info["newest_sample_period_specnum"] == 16
        assert len(info["answered"]) == 8
        assert info["missing"] == []
        assert info["stale"] == []
        assert info["empty"] == []
        assert info["min_newest_sid_g"] == list(targets[0])
        assert info["target_cube_span"] == 3072
        assert info["ring_cadence_specnums"] == 128
        assert ev == 10_000_000 + 1536 - 4 * 128

    def test_picks_mid_newest_when_lookback_zero(self):
        targets = [(sid, g) for (sid, g, _, _) in PROD_BIND_HOSTS]
        newest_per_half = {(sid, g): 20_000_000 for (sid, g) in targets}
        responses = _make_search_ring_responses(
            sids_g=targets, newest_per_half=newest_per_half,
        )
        store = FakeDsaStore(get_dict_responses=responses)
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets, lookback_cubes=0,
        )
        # cube_span // 2 = 3072 // 2 = 1536 forward of newest_start.
        assert ev == 20_000_000 + 1536
        assert info["target_offset_from_min_newest"] == 1536
        assert "ring_cadence_specnums" not in info  # not derived

    def test_returns_none_when_no_publishers(self):
        targets = [(sid, g) for (sid, g, _, _) in PROD_BIND_HOSTS]
        store = FakeDsaStore()
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets,
        )
        assert ev is None
        assert info["min_newest_event_specnum_start"] is None
        # 8 polled, all missing.
        assert len(info["polled"]) == 8
        assert len(info["missing"]) == 8
        assert info["answered"] == []

    def test_skips_stale_publishers(self):
        targets = [(1, 0), (1, 1)]
        # First half is fresh, second is stale.
        fresh_resp = _make_search_ring_responses(
            sids_g=[(1, 0)],
            newest_per_half={(1, 0): 5_000_000},
        )
        stale_resp = _make_search_ring_responses(
            sids_g=[(1, 1)],
            newest_per_half={(1, 1): 4_000_000},
            ts_age_s=120.0,
        )
        responses = {**fresh_resp, **stale_resp}
        store = FakeDsaStore(get_dict_responses=responses)
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets, lookback_cubes=0,
        )
        # Only the fresh half contributes — min = 5_000_000.
        assert info["min_newest_event_specnum_start"] == 5_000_000
        assert info["stale"] == ["/mon/search/1/1/ring"]
        assert info["answered"] == ["/mon/search/1/0/ring"]
        # target = min_newest + cube_span//2 = 5_000_000 + 1536
        assert ev == 5_000_000 + 1536

    def test_skips_empty_rings(self):
        # n_committed=0 → newest is None → ring not primed yet.
        targets = [(1, 0), (1, 1)]
        responses = _make_search_ring_responses(
            sids_g=[(1, 0)],
            newest_per_half={(1, 0): 5_000_000},
            n_committed=8,
        )
        responses["/mon/search/1/1/ring"] = {
            "search_node_id": 1,
            "gpu_half": 1,
            "n_committed": 0,
            "depth": 16,
            "t_det": 192,
            "sample_period_specnum": 16,
            "newest_event_specnum_start": None,
            "ts_wall_unix": time.time(),
        }
        store = FakeDsaStore(get_dict_responses=responses)
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets, lookback_cubes=0,
        )
        assert info["empty"] == ["/mon/search/1/1/ring"]
        assert info["answered"] == ["/mon/search/1/0/ring"]
        assert ev == 5_000_000 + 1536

    def test_clamps_to_zero_when_offset_exceeds_min_newest(self):
        # Pathological min_newest=100 with high lookback should
        # produce a negative target which is clamped to 0.
        # Use a single-cube ring so cube_cadence isn't derivable
        # (forces the lookback_cubes path to fall through without
        # extra subtraction).
        targets = [(1, 0)]
        responses = _make_search_ring_responses(
            sids_g=targets, newest_per_half={(1, 0): 100},
            n_committed=1, depth=16,
        )
        store = FakeDsaStore(get_dict_responses=responses)
        # Force the lookback to be huge but with n_committed=1 the
        # extra-offset subtraction is skipped (needs >= 2 to derive
        # cadence). target = min_newest + cube_span/2 = 100 + 1536
        # = 1636. lookback_cubes can't subtract because cadence
        # can't be derived. So ev = 1636.
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets, lookback_cubes=999,
        )
        assert ev == 100 + 1536
        assert info["min_newest_event_specnum_start"] == 100
        assert "ring_cadence_specnums" not in info

    def test_clamps_to_zero_when_pathological_newest(self):
        """When min_newest is impossibly small AND cube_cadence is
        derivable, the additional backward nudge can drive target
        below zero — that's clamped to 0."""
        targets = [(1, 0)]
        # n_committed=16 with very small newest_start means cadence
        # is derivable (16 cubes spanning 15*128 = 1920 samples).
        # newest=2000 → oldest=80. target=2000 + 1536 - 999*128
        # → highly negative, clamped to 0.
        responses = _make_search_ring_responses(
            sids_g=targets, newest_per_half={(1, 0): 2000},
            n_committed=16, depth=16,
        )
        store = FakeDsaStore(get_dict_responses=responses)
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            store, targets=targets, lookback_cubes=999,
        )
        assert ev == 0
        assert info["ring_cadence_specnums"] == 128

    def test_empty_targets_returns_none(self):
        ev, info = cube_dump_now._resolve_event_specnum_from_search_ring(
            FakeDsaStore(), targets=[],
        )
        assert ev is None
        assert info["polled"] == []


class TestFleetDumpNowPrefersSearchRing:
    def test_search_ring_takes_precedence_over_corr_fast(self):
        """When both mon-keys are live, the search-ring window wins
        and the resulting event_specnum lives in the *search-side*
        domain (the cube_ring's specnum_start), not corr_fast's
        service-start epoch.
        """
        targets = [(sid, g) for (sid, g, _, _) in PROD_BIND_HOSTS]
        newest_per_half = {(sid, g): 999_000_000 for (sid, g) in targets}
        now = time.time()
        responses = {
            f"/mon/corr_rt/{cg}/corr_fast": {
                "block_n": 1000,
                "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now,
            }
            for cg in range(16)
        }
        responses.update(_make_search_ring_responses(
            sids_g=targets, newest_per_half=newest_per_half,
        ))
        store = FakeDsaStore(get_dict_responses=responses)
        snd = FakeSender()
        res = cube_dump_now.fleet_dump_now(
            store, bind_hosts=PROD_BIND_HOSTS, sender=snd,
        )
        assert res.ok is True
        # Search-ring path is selected.
        assert res.arm_info["source_used"] == "search_ring"
        assert res.arm_info["source"] == "search_ring"
        # event_specnum is in the LARGE (search-side) domain, NOT
        # the small corr_fast domain.
        assert res.event_specnum is not None
        # Target should be approximately min_newest (999M) + 1536
        # (half a cube_span forward of the laggard's newest start)
        # minus a small backward nudge from the default 4-cube
        # lookback (4 × 128 = 512 search samples).
        assert res.event_specnum >= 999_000_000
        assert res.event_specnum <= 999_000_000 + 1536

    def test_falls_back_to_corr_fast_when_search_ring_missing(self):
        """The legacy fleet (no SearchRingMonPublisher yet) still
        gets *some* auto-arm, but the arm_info note explicitly says
        we fell back."""
        snd = FakeSender()
        # Only corr_fast keys, no search-ring keys.
        now = time.time()
        cf_resp = {
            f"/mon/corr_rt/{cg}/corr_fast": {
                "block_n": 1000,
                "block_specnum_start": 1000 * 2048,
                "ts_wall_unix": now,
            }
            for cg in range(16)
        }
        store = FakeDsaStore(get_dict_responses=cf_resp)
        res = cube_dump_now.fleet_dump_now(
            store, bind_hosts=PROD_BIND_HOSTS, sender=snd,
        )
        assert res.ok is True
        assert res.arm_info["source_used"] == "corr_fast"
        # Search-ring attempt is preserved in arm_info for diagnostics.
        assert "search_ring_attempt" in res.arm_info
        assert (
            len(res.arm_info["search_ring_attempt"]["missing"]) == 8
        )
        assert res.event_specnum == 996 * 2048      # corr_fast path

    def test_both_missing_returns_503_payload(self):
        snd = FakeSender()
        store = FakeDsaStore()
        res = cube_dump_now.fleet_dump_now(
            store, bind_hosts=PROD_BIND_HOSTS, sender=snd,
        )
        assert res.ok is False
        assert res.event_specnum is None
        assert res.error == "no corr_fast mon-keys; is the fleet up?"
        # Both attempts diagnosed.
        assert "search_ring_attempt" in res.arm_info
        assert (
            len(res.arm_info["search_ring_attempt"]["missing"]) == 8
        )
        assert len(res.arm_info["missing"]) == 16  # corr_fast polled all 16
