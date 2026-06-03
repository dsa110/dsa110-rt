"""Unit tests for :mod:`tools.dashboard.dsa_monitor.inject_calibration`."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping
from unittest.mock import MagicMock

import pytest

_DASHBOARD_DIR = Path(__file__).parent.parent / "tools" / "dashboard" / "dsa_monitor"
sys.path.insert(0, str(_DASHBOARD_DIR))

import inject_calibration as ic  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEtcd:
    def __init__(self, payloads: Mapping[str, Mapping[str, Any]]):
        self._payloads = {
            k: json.dumps(v).encode("utf-8") for k, v in payloads.items()
        }

    def get_prefix(self, prefix: str):
        for k, v in self._payloads.items():
            if k.startswith(prefix):
                yield v, None


class FakeStore:
    """Mimics control_store.ControlStore for the calibration module.

    Surfaces:
      - get_dict(key) -> Mapping | None
      - put_dict(key, payload) -> None
      - _ensure() / _store / get_etcd() for the list_all path.
    """

    def __init__(self) -> None:
        self.kv: Dict[str, Dict[str, Any]] = {}
        self.put_calls: List[tuple[str, Dict[str, Any]]] = []
        self._ensured = False
        self._etcd: Any = None
        # control_store.ControlStore wraps a DsaStore; the matcher
        # walks store._store.get_etcd() but the dashboard side calls
        # store.get_etcd() directly so we expose it on the outer too.

    def get_dict(self, key: str) -> Any:
        return self.kv.get(key)

    def put_dict(self, key: str, payload: Mapping[str, Any]) -> None:
        self.put_calls.append((key, dict(payload)))
        self.kv[key] = dict(payload)

    def _ensure(self) -> None:
        self._ensured = True

    @property
    def _store(self):
        return self  # delegate to self for get_etcd

    def get_etcd(self) -> Any:
        if self._etcd is None:
            self._etcd = FakeEtcd(self.kv)
        else:
            # refresh on access
            self._etcd = FakeEtcd(self.kv)
        return self._etcd


# ---------------------------------------------------------------------------
# bucket_key + math
# ---------------------------------------------------------------------------


class TestBucketKey:
    @pytest.mark.parametrize(
        "dm,width,expected",
        [
            (500.0, 32, "dm0500_w0032"),
            (150.0, 64, "dm0150_w0064"),
            (1500.0, 32, "dm1500_w0032"),
            (149.9, 32, "dm0150_w0032"),  # rounds to 150
            (174.9, 32, "dm0150_w0032"),  # rounds to 150 (well below .5)
            (526.0, 32, "dm0550_w0032"),  # 526 > 525, unambiguously rounds up
            (524.0, 32, "dm0500_w0032"),  # 524 < 525, unambiguously rounds down
            (1530.0, 32, "dm1550_w0032"),
        ],
    )
    def test_known_buckets(self, dm, width, expected):
        # Python's round() uses banker's rounding (round-half-to-even);
        # boundary values (n + 0.5) are avoided here to keep the test
        # deterministic across platforms.
        assert ic.bucket_key(dm, width) == expected

    def test_bad_width_raises(self):
        with pytest.raises(ValueError):
            ic.bucket_key(500.0, 0)

    def test_nan_dm_raises(self):
        with pytest.raises(ValueError):
            ic.bucket_key(float("nan"), 32)

    def test_calibration_key_uses_prefix(self):
        assert ic.calibration_key(500.0, 32) == (
            ic.CALIBRATION_PREFIX + "dm0500_w0032"
        )


class TestSnrToFluence:
    def test_round_trip_with_K(self):
        # observed_snr = K × sqrt(fluence / width)
        # K=10, target_snr=20, width=32 ⇒ fluence = 32 × (20/10)^2 = 128
        out = ic.snr_to_fluence(target_snr=20.0, K=10.0, width_samples=32)
        assert out == pytest.approx(128.0)

    @pytest.mark.parametrize(
        "kw",
        [
            {"target_snr": 0.0, "K": 5.0, "width_samples": 32},
            {"target_snr": -1.0, "K": 5.0, "width_samples": 32},
            {"target_snr": 5.0, "K": 0.0, "width_samples": 32},
            {"target_snr": 5.0, "K": 5.0, "width_samples": 0},
            {"target_snr": float("nan"), "K": 5.0, "width_samples": 32},
        ],
    )
    def test_bad_inputs_raise(self, kw):
        with pytest.raises(ValueError):
            ic.snr_to_fluence(**kw)


# ---------------------------------------------------------------------------
# CalibrationStore
# ---------------------------------------------------------------------------


class TestCalibrationStore:
    def test_round_trip(self):
        store = FakeStore()
        cs = ic.CalibrationStore(store)
        entry = ic.CalibrationEntry(
            bucket="dm0500_w0032", dm_pc_cm3_rounded=500, width_samples=32,
            K=7.5, last_fluence_jy_ms=100.0, last_observed_snr=12.34,
            last_inj_id="cal_x", last_calibrated_at_unix=1_700_000_000.0,
            actor="tester",
        )
        key = cs.put(entry)
        assert key == ic.CALIBRATION_PREFIX + "dm0500_w0032"
        roundtrip = cs.get(dm_pc_cm3=500.0, width_samples=32)
        assert roundtrip is not None
        assert roundtrip == entry

    def test_get_missing_returns_none(self):
        cs = ic.CalibrationStore(FakeStore())
        assert cs.get(dm_pc_cm3=500.0, width_samples=32) is None

    def test_get_bad_payload_returns_none(self):
        store = FakeStore()
        store.kv[ic.CALIBRATION_PREFIX + "dm0500_w0032"] = {"junk": True}
        cs = ic.CalibrationStore(store)
        assert cs.get(dm_pc_cm3=500.0, width_samples=32) is None

    def test_list_all_returns_sorted(self):
        store = FakeStore()
        cs = ic.CalibrationStore(store)
        cs.put(ic.CalibrationEntry(
            bucket="dm1500_w0032", dm_pc_cm3_rounded=1500, width_samples=32,
            K=8.0, last_fluence_jy_ms=100.0, last_observed_snr=20.0,
            last_inj_id="x", last_calibrated_at_unix=1.0,
        ))
        cs.put(ic.CalibrationEntry(
            bucket="dm0150_w0032", dm_pc_cm3_rounded=150, width_samples=32,
            K=5.0, last_fluence_jy_ms=100.0, last_observed_snr=12.0,
            last_inj_id="y", last_calibrated_at_unix=2.0,
        ))
        cs.put(ic.CalibrationEntry(
            bucket="dm0500_w0032", dm_pc_cm3_rounded=500, width_samples=32,
            K=7.0, last_fluence_jy_ms=100.0, last_observed_snr=15.0,
            last_inj_id="z", last_calibrated_at_unix=3.0,
        ))
        out = cs.list_all()
        assert [e.bucket for e in out] == [
            "dm0150_w0032", "dm0500_w0032", "dm1500_w0032",
        ]


# ---------------------------------------------------------------------------
# publish_active_inject + get_match_event
# ---------------------------------------------------------------------------


class TestActiveInjectPublish:
    def test_publish_writes_expected_key_and_payload(self):
        store = FakeStore()
        key = ic.publish_active_inject(
            store,
            inj_id="cal_x",
            dm_pc_cm3=500.0,
            l_rad=0.0,
            m_rad=0.0,
            width_samples=32,
            fluence_jy_ms=100.0,
            apply_at_specnum=123456,
            fired_at_unix=1_700_000_000.0,
            ttl_s=60.0,
            fired_by="tester",
            target_snr=20.0,
        )
        assert key == ic.ACTIVE_INJECT_PREFIX + "cal_x"
        payload = store.kv[key]
        assert payload["dm_pc_cm3"] == 500.0
        assert payload["fluence_jy_ms"] == 100.0
        assert payload["target_snr"] == 20.0
        assert payload["fired_at_unix"] == 1_700_000_000.0

    def test_publish_skips_target_snr_when_none(self):
        store = FakeStore()
        ic.publish_active_inject(
            store,
            inj_id="cal_x",
            dm_pc_cm3=500.0, l_rad=0.0, m_rad=0.0,
            width_samples=32, fluence_jy_ms=100.0,
            apply_at_specnum=0, fired_at_unix=1.0,
        )
        payload = store.kv[ic.ACTIVE_INJECT_PREFIX + "cal_x"]
        assert "target_snr" not in payload


class TestGetMatchEvent:
    def test_present(self):
        store = FakeStore()
        store.kv[ic.MATCH_EVENT_PREFIX + "x"] = {
            "best": {"observed_snr": 12.3, "K_inferred": 5.0},
            "n_matches": 2,
        }
        out = ic.get_match_event(store, "x")
        assert out is not None
        assert out["best"]["observed_snr"] == 12.3

    def test_missing_returns_none(self):
        assert ic.get_match_event(FakeStore(), "missing") is None


# ---------------------------------------------------------------------------
# fire_calibration_probe
# ---------------------------------------------------------------------------


class TestFireCalibrationProbe:
    @staticmethod
    def _good_inject(store, **kwargs):
        return {
            "ok": True,
            "cmd": "inject",
            "val": dict(
                {
                    "inj_id": kwargs["inj_id"],
                    "l_rad": kwargs["l_rad"],
                    "m_rad": kwargs["m_rad"],
                    "dm_pc_cm3": kwargs["dm_pc_cm3"],
                    "fluence_jy_ms": kwargs["fluence_jy_ms"],
                    "width_samples": kwargs["width_samples"],
                    "profile": kwargs["profile"],
                    "apply_at_specnum": 1_000_000,
                },
            ),
        }

    def _seed_match(
        self, store: FakeStore, inj_id: str, *,
        observed_snr: float, K: float, observed_specnum: int = 12345,
        matched_at: float = 100.0,
        observed_l_rad: float = 0.0,
        observed_m_rad: float = 0.0,
        observed_width_samples: int = 32,
    ) -> None:
        store.kv[ic.MATCH_EVENT_PREFIX + inj_id] = {
            "best": {
                "observed_snr": observed_snr,
                "K_inferred": K,
                "observed_event_specnum": observed_specnum,
                "matched_at_unix": matched_at,
                "observed_l_rad": observed_l_rad,
                "observed_m_rad": observed_m_rad,
                "observed_width_samples": observed_width_samples,
            },
            "n_matches": 1,
        }

    def test_happy_path_stores_K(self):
        store = FakeStore()
        # Simulate the match arriving on the first poll iteration.
        now = [100.0]
        time_fn = lambda: now[0]   # noqa: E731
        sleep_calls: List[float] = []

        def sleep_fn(s):
            sleep_calls.append(s)
            now[0] += s
            # On first sleep, seed the match.
            if len(sleep_calls) == 1:
                # inj_id is deterministic given the (prefix, dm, width, ts)
                inj_id = ic._build_probe_inj_id(
                    prefix="cal_probe", dm=500.0, width=32, timestamp=100.0,
                )
                self._seed_match(
                    store, inj_id, observed_snr=20.0,
                    K=20.0 * math.sqrt(32.0 / 100.0),
                    observed_specnum=12345,
                    matched_at=100.0 + 0.5,
                )

        result = ic.fire_calibration_probe(
            store,
            inject_fn=self._good_inject,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=100.0,
            poll_timeout_s=10.0,
            poll_interval_s=0.5,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        assert result.ok is True
        assert result.reason == "ok"
        assert result.bucket == "dm0500_w0032"
        # K should match what we seeded.
        assert result.K == pytest.approx(
            20.0 * math.sqrt(32.0 / 100.0), rel=1e-9,
        )
        assert result.observed_snr == 20.0
        # Calibration entry persisted.
        cs = ic.CalibrationStore(store)
        got = cs.get(dm_pc_cm3=500.0, width_samples=32)
        assert got is not None
        assert got.K == pytest.approx(result.K)
        assert got.last_fluence_jy_ms == 100.0
        assert got.last_observed_snr == 20.0

    def test_fleet_lm_offset_match_still_stores_K(self):
        """Regression: a boresight-declared probe matched at the
        documented fleet l/m≈0.019 offset (with a low search-width)
        must still store K — the dashboard no longer re-gates on l/m
        or width (the C2 matcher owns those tolerances)."""
        store = FakeStore()
        now = [700.0]
        sleep_calls: List[float] = []

        def sleep_fn(s):
            sleep_calls.append(s)
            now[0] += s
            if len(sleep_calls) == 1:
                inj_id = ic._build_probe_inj_id(
                    prefix="cal_probe", dm=900.0, width=32, timestamp=700.0,
                )
                self._seed_match(
                    store, inj_id, observed_snr=20.34,
                    K=20.34 * math.sqrt(32.0 / 100.0),
                    observed_l_rad=0.0192,
                    observed_m_rad=0.0191,
                    observed_width_samples=4,
                )

        result = ic.fire_calibration_probe(
            store,
            inject_fn=self._good_inject,
            dm_pc_cm3=900.0,
            width_samples=32,
            fluence_jy_ms=100.0,
            l_rad=0.0,
            m_rad=0.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
        )
        assert result.ok is True
        assert result.reason == "ok"
        assert result.bucket == "dm0900_w0032"
        assert ic.CalibrationStore(store).get(
            dm_pc_cm3=900.0, width_samples=32,
        ) is not None

    def test_no_match_returns_failure(self):
        store = FakeStore()
        now = [200.0]
        time_fn = lambda: now[0]   # noqa: E731

        def sleep_fn(s):
            now[0] += s   # advance clock, but never seed a match

        result = ic.fire_calibration_probe(
            store,
            inject_fn=self._good_inject,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=100.0,
            poll_timeout_s=2.0,
            poll_interval_s=0.5,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        assert result.ok is False
        assert result.reason == "no_match"
        # CalibrationStore should NOT have been written.
        cs = ic.CalibrationStore(store)
        assert cs.get(dm_pc_cm3=500.0, width_samples=32) is None

    def test_inject_failure_returns_failure(self):
        store = FakeStore()
        now = [300.0]

        def bad_inject(store, **kw):
            return {"ok": False, "error": "no corr_fast publishers"}

        result = ic.fire_calibration_probe(
            store,
            inject_fn=bad_inject,
            dm_pc_cm3=500.0,
            width_samples=32,
            time_fn=lambda: now[0],
            sleep_fn=lambda s: None,
        )
        assert result.ok is False
        assert result.reason.startswith("inject_failed")

    def test_inject_callable_raising_returns_failure(self):
        store = FakeStore()
        now = [400.0]

        def raising(store, **kw):
            raise RuntimeError("etcd down")

        result = ic.fire_calibration_probe(
            store,
            inject_fn=raising,
            dm_pc_cm3=500.0,
            width_samples=32,
            time_fn=lambda: now[0],
            sleep_fn=lambda s: None,
        )
        assert result.ok is False
        assert result.reason.startswith("inject_failed")
        # Active-inject row was still written before inject raised.
        assert any(
            k.startswith(ic.ACTIVE_INJECT_PREFIX)
            for k, _ in store.put_calls
        )

    def test_active_inject_row_rewritten_with_apply_at(self):
        store = FakeStore()
        now = [500.0]
        sleep_calls = []

        def sleep_fn(s):
            sleep_calls.append(s)
            now[0] += s
            if len(sleep_calls) == 1:
                inj_id = ic._build_probe_inj_id(
                    prefix="cal_probe", dm=500.0, width=32, timestamp=500.0,
                )
                self._seed_match(
                    store, inj_id, observed_snr=15.0,
                    K=15.0 * math.sqrt(32.0 / 100.0),
                )

        result = ic.fire_calibration_probe(
            store,
            inject_fn=self._good_inject,
            dm_pc_cm3=500.0,
            width_samples=32,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
        )
        assert result.ok is True
        # Active-inject row should now carry the apply_at_specnum from
        # the inject response (1_000_000), not the placeholder 0.
        active = store.kv[ic.ACTIVE_INJECT_PREFIX + result.inj_id]
        assert active["apply_at_specnum"] == 1_000_000


# ---------------------------------------------------------------------------
# delete_snr_calibrations
# ---------------------------------------------------------------------------


class _FakeEtcdRW:
    """Read/write fake bound to a shared kv dict: supports the
    ``get_prefix`` walk :meth:`CalibrationStore.list_all` uses and the
    ``delete`` :meth:`CalibrationStore.delete` calls."""

    def __init__(self, kv: Dict[str, Dict[str, Any]]):
        self._kv = kv  # live reference, not a snapshot

    def get_prefix(self, prefix: str):
        for k, v in list(self._kv.items()):
            if k.startswith(prefix):
                yield json.dumps(v).encode("utf-8"), None

    def delete(self, key: str) -> bool:
        return self._kv.pop(key, None) is not None


class _FakeStoreRW(FakeStore):
    """FakeStore whose etcd client mutates the same kv (so deletes stick)."""

    def get_etcd(self) -> Any:
        return _FakeEtcdRW(self.kv)


class TestDeleteSnrCalibrations:
    def _seed(self, store: "_FakeStoreRW") -> None:
        cs = ic.CalibrationStore(store)
        for dm, width, K in [(150.0, 32, 9.5), (500.0, 64, 12.0)]:
            cs.put(ic.CalibrationEntry(
                bucket=ic.bucket_key(dm, width),
                dm_pc_cm3_rounded=int(round(dm / 50.0) * 50),
                width_samples=width,
                K=K,
                last_fluence_jy_ms=100.0,
                last_observed_snr=K * math.sqrt(100.0 / width),
                last_inj_id="seed",
                last_calibrated_at_unix=1.0,
            ))

    def test_dry_run_lists_without_deleting(self):
        store = _FakeStoreRW()
        self._seed(store)
        out = ic.delete_snr_calibrations(store, dry_run=True, user="t")
        assert out["ok"] is True
        assert out["dry_run"] is True
        assert out["summary"]["n_buckets"] == 2
        assert out["summary"]["n_deleted"] == 0
        assert out["summary"]["n_present"] == 2
        assert all(b["status"] == "exists" for b in out["buckets"])
        # Nothing actually removed.
        assert ic.CalibrationStore(store).get(
            dm_pc_cm3=150.0, width_samples=32) is not None

    def test_delete_removes_all_buckets(self):
        store = _FakeStoreRW()
        self._seed(store)
        out = ic.delete_snr_calibrations(store, dry_run=False, user="t")
        assert out["ok"] is True
        assert out["dry_run"] is False
        assert out["summary"]["n_buckets"] == 2
        assert out["summary"]["n_deleted"] == 2
        assert out["summary"]["n_failed"] == 0
        assert all(b["status"] == "deleted" for b in out["buckets"])
        # Both buckets gone.
        cs = ic.CalibrationStore(store)
        assert cs.get(dm_pc_cm3=150.0, width_samples=32) is None
        assert cs.get(dm_pc_cm3=500.0, width_samples=64) is None

    def test_empty_table_is_ok(self):
        store = _FakeStoreRW()
        out = ic.delete_snr_calibrations(store, dry_run=False, user="t")
        assert out["ok"] is True
        assert out["summary"]["n_buckets"] == 0
        assert out["buckets"] == []
