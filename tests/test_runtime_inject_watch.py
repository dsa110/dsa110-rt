"""M7.4 Phase 6 runtime: unit tests for
:mod:`dsart.inject.runtime_watch`.

These tests exercise the DsaStore-based watcher used by
``corr_fast_integration --inject-watch`` to consume runtime
``InjectionConfig`` payloads pushed to
``/cmd/dsart/corr/<chgroup>/inject``.

The DsaStore is mocked so the tests don't require a live etcd; the
``OnlineInjector`` is real but operates on a tiny synthetic antpos
array so construction completes in milliseconds.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Optional

import numpy as np
import pytest

from dsart.common.constants import NANTS, NCHAN_PER_CHGROUP
from dsart.inject.online import InjectionConfig, OnlineInjector
from dsart.inject.runtime_watch import (
    RuntimeInjectWatch,
    build_runtime_inject_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeStore:
    """Minimal DsaStore stand-in: records add_watch + cancel, lets the
    test drive the callback synchronously.
    """

    def __init__(self) -> None:
        self.watches: dict[int, tuple[str, Callable[[Any], None]]] = {}
        self.cancelled: list[int] = []
        self._next_id = 1
        self._lock = threading.Lock()

    def add_watch(self, key: str, cb: Callable[[Any], None]) -> int:
        with self._lock:
            wid = self._next_id
            self._next_id += 1
            self.watches[wid] = (key, cb)
            return wid

    def cancel(self, wid: int) -> None:
        with self._lock:
            if wid not in self.watches:
                raise RuntimeError(f"unknown watch_id={wid}")
            self.cancelled.append(wid)
            self.watches.pop(wid)

    def fire(self, wid: int, event: Any) -> None:
        """Invoke the registered callback (simulates etcd PUT)."""
        with self._lock:
            cb = self.watches[wid][1]
        cb(event)


@pytest.fixture()
def real_injector() -> OnlineInjector:
    rng = np.random.default_rng(20260527)
    # Tiny synthetic antpos so OnlineInjector construction is fast and
    # the phasor table fits comfortably; chgroup=0 selects the lowest
    # sub-band.
    antpos_e = rng.normal(0.0, 200.0, size=(NANTS,)).astype(np.float32)
    antpos_n = rng.normal(0.0, 200.0, size=(NANTS,)).astype(np.float32)
    import torch
    inj = OnlineInjector(
        antpos_e=antpos_e,
        antpos_n=antpos_n,
        chgroup=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return inj


_VALID_CFG_JSON = json.dumps({
    "inj_id": "phase6_runtime_t1",
    "l_rad": 0.0,
    "m_rad": 0.0,
    "dm_pc_cm3": 500.0,
    "fluence_jy_ms": 50.0,
    "width_samples": 32,
    "profile": "gaussian",
    "apply_at_specnum": 1_000_000,
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildRuntimeInjectKey:
    def test_canonical_format(self):
        assert build_runtime_inject_key(0) == "/cmd/dsart/corr/0/inject"
        assert build_runtime_inject_key(15) == "/cmd/dsart/corr/15/inject"

    def test_coerces_to_int(self):
        assert build_runtime_inject_key(3.0) == "/cmd/dsart/corr/3/inject"


class TestRuntimeInjectWatchLifecycle:
    def test_start_registers_watch_on_correct_key(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(
            injector=real_injector, chgroup=0, store=store,
        )
        assert w.key == "/cmd/dsart/corr/0/inject"
        w.start()
        assert len(store.watches) == 1
        wid = next(iter(store.watches))
        assert store.watches[wid][0] == "/cmd/dsart/corr/0/inject"
        assert w._watch_id == wid

    def test_start_is_idempotent(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=3, store=store)
        w.start()
        first_wid = w._watch_id
        w.start()
        assert w._watch_id == first_wid
        assert len(store.watches) == 1

    def test_stop_cancels_watch(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        wid = w._watch_id
        w.stop()
        assert wid in store.cancelled
        assert w._watch_id is None
        # Cancel a second time is a no-op.
        w.stop()

    def test_stop_swallows_cancel_errors(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        store.cancel = lambda _wid: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        w.stop()  # must not raise
        assert w._watch_id is None


class TestRuntimeInjectWatchDispatch:
    def test_valid_payload_queues_injection(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        wid = w._watch_id

        # Simulate an etcd PUT — DsaStore delivers the value as a dict.
        store.fire(wid, json.loads(_VALID_CFG_JSON_WIRE))
        assert w.n_events == 1
        assert w.n_queued == 1
        assert len(real_injector.pending) == 1
        assert "phase6_runtime_t1" in real_injector.pending

    def test_payload_can_be_bytes(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        store.fire(w._watch_id, _VALID_CFG_JSON_WIRE.encode("utf-8"))
        assert w.n_queued == 1

    def test_payload_can_be_str(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        store.fire(w._watch_id, _VALID_CFG_JSON_WIRE)
        assert w.n_queued == 1

    def test_non_inject_payload_is_dropped(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        store.fire(w._watch_id, {"cmd": "start", "val": 53.85})
        assert w.n_events == 1
        assert w.n_queued == 0
        assert len(real_injector.pending) == 0

    def test_malformed_payload_logged_and_dropped(self, real_injector, caplog):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        # Missing required fields → InjectionConfig.from_json raises,
        # handler swallows and returns None.
        store.fire(w._watch_id, {"cmd": "inject", "val": {"inj_id": "x"}})
        assert w.n_events == 1
        assert w.n_queued == 0
        assert len(real_injector.pending) == 0

    def test_injector_validation_failure_is_dropped(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        # l_rad too large → InjectionConfig.__post_init__ raises.
        bad = {
            "cmd": "inject",
            "val": {
                "inj_id": "bad", "l_rad": 1.5, "m_rad": 0.0,
                "dm_pc_cm3": 500.0, "fluence_jy_ms": 50.0,
                "width_samples": 32, "profile": "gaussian",
                "apply_at_specnum": 1000,
            },
        }
        store.fire(w._watch_id, bad)
        assert w.n_queued == 0
        assert len(real_injector.pending) == 0

    def test_multiple_payloads_accumulate(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        for i in range(5):
            payload = {
                "cmd": "inject",
                "val": json.loads(_VALID_CFG_JSON_WIRE)["val"] | {
                    "inj_id": f"phase6_burst_{i}",
                    "apply_at_specnum": 1_000_000 + i * 10_000,
                },
            }
            store.fire(w._watch_id, payload)
        assert w.n_events == 5
        assert w.n_queued == 5
        assert len(real_injector.pending) == 5
        assert set(real_injector.pending.keys()) == {
            f"phase6_burst_{i}" for i in range(5)
        }

    def test_watch_thread_survives_exception_in_handler(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        # Pathological payload that would explode json.dumps if we
        # tried to serialise it (function objects aren't JSON
        # serialisable).
        store.fire(w._watch_id, {"cmd": "inject", "val": lambda x: x})
        assert w.n_events == 1
        assert w.n_queued == 0
        # The watch is still alive; subsequent valid payload still
        # queues.
        store.fire(w._watch_id, json.loads(_VALID_CFG_JSON_WIRE))
        assert w.n_queued == 1


class TestRuntimeInjectWatchState:
    """T2 (2026-06-07): the ``state()`` snapshot reports both
    counters AND the most recently queued probe's identifying
    metadata. The corr_fast mon publisher ships this dict to etcd
    so the dashboard can verify all 16 corr nodes received an
    injection without ssh-ing into each one."""

    def test_state_initial(self, real_injector):
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=FakeStore())
        snap = w.state()
        assert snap["inject_n_events"] == 0
        assert snap["inject_n_queued"] == 0
        assert snap["inject_last_inj_id"] is None
        assert snap["inject_last_apply_at_specnum"] is None
        assert snap["inject_last_event_unix"] is None
        assert snap["inject_last_queued_unix"] is None

    def test_state_after_queue_records_inj_id_and_apply_at(self, real_injector):
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        store.fire(w._watch_id, json.loads(_VALID_CFG_JSON_WIRE))
        snap = w.state()
        assert snap["inject_n_events"] == 1
        assert snap["inject_n_queued"] == 1
        assert snap["inject_last_inj_id"] == "phase6_runtime_t1"
        assert snap["inject_last_apply_at_specnum"] == 1_000_000
        assert isinstance(snap["inject_last_event_unix"], float)
        assert isinstance(snap["inject_last_queued_unix"], float)

    def test_state_event_counter_advances_for_dropped_payloads(self, real_injector):
        """A non-inject payload still bumps ``inject_n_events`` (the
        operator can tell the watcher is alive) but leaves
        ``inject_n_queued`` and ``inject_last_inj_id`` unchanged."""
        store = FakeStore()
        w = RuntimeInjectWatch(injector=real_injector, chgroup=0, store=store)
        w.start()
        store.fire(w._watch_id, {"cmd": "shutdown"})
        snap = w.state()
        assert snap["inject_n_events"] == 1
        assert snap["inject_n_queued"] == 0
        assert snap["inject_last_inj_id"] is None
        assert snap["inject_last_apply_at_specnum"] is None


# The wire payload the watcher actually sees: same shape the dashboard
# writes (`{cmd: "inject", val: <InjectionConfig dict>}`).
_VALID_CFG_JSON_WIRE = json.dumps({
    "cmd": "inject",
    "val": json.loads(_VALID_CFG_JSON),
})
