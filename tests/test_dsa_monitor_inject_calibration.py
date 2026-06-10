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
        "dm,expected",
        [
            (500.0, "dm0500"),
            (150.0, "dm0150"),
            (1500.0, "dm1500"),
            (149.9, "dm0150"),  # rounds to 150
            (174.9, "dm0150"),  # rounds to 150 (well below .5)
            (526.0, "dm0550"),  # 526 > 525, unambiguously rounds up
            (524.0, "dm0500"),  # 524 < 525, unambiguously rounds down
            (1530.0, "dm1550"),
        ],
    )
    def test_known_buckets(self, dm, expected):
        # Python's round() uses banker's rounding (round-half-to-even);
        # boundary values (n + 0.5) are avoided here to keep the test
        # deterministic across platforms.
        assert ic.bucket_key(dm) == expected

    def test_nan_dm_raises(self):
        with pytest.raises(ValueError):
            ic.bucket_key(float("nan"))

    def test_calibration_key_uses_prefix(self):
        assert ic.calibration_key(500.0) == ic.CALIBRATION_PREFIX + "dm0500"

    def test_is_legacy_bucket(self):
        # Pre-fix per-(DM, width) keys should be flagged as legacy so
        # the operator can wipe them via delete_snr_calibrations.
        assert ic.is_legacy_bucket("dm0500_w0032")
        assert ic.is_legacy_bucket("dm1500_w0064")
        # Post-fix DM-only keys are NOT legacy.
        assert not ic.is_legacy_bucket("dm0500")
        assert not ic.is_legacy_bucket("dm1500")


class TestSnrToFluence:
    def test_round_trip_with_K(self):
        # 2026-06-10 linear model: observed_snr = K × fluence / sqrt(width)
        # K=10, target_snr=20, width=32 ⇒ fluence = 20 × sqrt(32) / 10
        out = ic.snr_to_fluence(target_snr=20.0, K=10.0, width_samples=32)
        assert out == pytest.approx(20.0 * math.sqrt(32.0) / 10.0)

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
            bucket="dm0500", dm_pc_cm3_rounded=500, width_samples=32,
            K=7.5, last_fluence_jy_ms=100.0, last_observed_snr=12.34,
            last_inj_id="cal_x", last_calibrated_at_unix=1_700_000_000.0,
            actor="tester",
        )
        key = cs.put(entry)
        assert key == ic.CALIBRATION_PREFIX + "dm0500"
        roundtrip = cs.get(dm_pc_cm3=500.0)
        assert roundtrip is not None
        assert roundtrip == entry

    def test_get_missing_returns_none(self):
        cs = ic.CalibrationStore(FakeStore())
        assert cs.get(dm_pc_cm3=500.0) is None

    def test_get_bad_payload_returns_none(self):
        store = FakeStore()
        store.kv[ic.CALIBRATION_PREFIX + "dm0500"] = {"junk": True}
        cs = ic.CalibrationStore(store)
        assert cs.get(dm_pc_cm3=500.0) is None

    def test_get_ignores_legacy_per_width_keys(self):
        # Pre-fix entries lived under dmNNNN_wWWWW. The new DM-only
        # lookup must NOT see them (they remain visible to list_all
        # and delete_snr_calibrations so the operator can wipe them).
        store = FakeStore()
        store.kv[ic.CALIBRATION_PREFIX + "dm0500_w0032"] = ic.CalibrationEntry(
            bucket="dm0500_w0032", dm_pc_cm3_rounded=500, width_samples=32,
            K=99.0, last_fluence_jy_ms=100.0, last_observed_snr=99.0,
            last_inj_id="legacy", last_calibrated_at_unix=1.0,
        ).to_dict()
        cs = ic.CalibrationStore(store)
        assert cs.get(dm_pc_cm3=500.0) is None

    def test_get_is_width_independent(self):
        # K is stored under the DM bucket only; lookups for any width
        # at the same DM must return the same K.
        store = FakeStore()
        cs = ic.CalibrationStore(store)
        cs.put(ic.CalibrationEntry(
            bucket="dm0500", dm_pc_cm3_rounded=500, width_samples=32,
            K=7.5, last_fluence_jy_ms=100.0, last_observed_snr=13.27,
            last_inj_id="cal_w32", last_calibrated_at_unix=1.0,
        ))
        # Different injection widths all hit the same K.
        for _w in (1, 4, 16, 32, 64, 128, 256):
            got = cs.get(dm_pc_cm3=500.0)
            assert got is not None and got.K == 7.5

    def test_list_all_returns_sorted(self):
        store = FakeStore()
        cs = ic.CalibrationStore(store)
        cs.put(ic.CalibrationEntry(
            bucket="dm1500", dm_pc_cm3_rounded=1500, width_samples=32,
            K=8.0, last_fluence_jy_ms=100.0, last_observed_snr=20.0,
            last_inj_id="x", last_calibrated_at_unix=1.0,
        ))
        cs.put(ic.CalibrationEntry(
            bucket="dm0150", dm_pc_cm3_rounded=150, width_samples=32,
            K=5.0, last_fluence_jy_ms=100.0, last_observed_snr=12.0,
            last_inj_id="y", last_calibrated_at_unix=2.0,
        ))
        cs.put(ic.CalibrationEntry(
            bucket="dm0500", dm_pc_cm3_rounded=500, width_samples=32,
            K=7.0, last_fluence_jy_ms=100.0, last_observed_snr=15.0,
            last_inj_id="z", last_calibrated_at_unix=3.0,
        ))
        out = cs.list_all()
        assert [e.bucket for e in out] == ["dm0150", "dm0500", "dm1500"]


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
        assert result.bucket == "dm0500"
        # K should match what we seeded.
        assert result.K == pytest.approx(
            20.0 * math.sqrt(32.0 / 100.0), rel=1e-9,
        )
        assert result.observed_snr == 20.0
        # Calibration entry persisted.
        cs = ic.CalibrationStore(store)
        got = cs.get(dm_pc_cm3=500.0)
        assert got is not None
        assert got.K == pytest.approx(result.K)
        assert got.last_fluence_jy_ms == 100.0
        assert got.last_observed_snr == 20.0
        # Width is recorded as probe metadata only — the bucket is
        # DM-only, so a hypothetical injection at a different width
        # would read back the SAME K.
        assert got.width_samples == 32
        assert got.bucket == "dm0500"

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
        assert result.bucket == "dm0900"
        assert ic.CalibrationStore(store).get(dm_pc_cm3=900.0) is not None

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
        assert cs.get(dm_pc_cm3=500.0) is None

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
                bucket=ic.bucket_key(dm),
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
        assert ic.CalibrationStore(store).get(dm_pc_cm3=150.0) is not None

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
        assert cs.get(dm_pc_cm3=150.0) is None
        assert cs.get(dm_pc_cm3=500.0) is None

    def test_empty_table_is_ok(self):
        store = _FakeStoreRW()
        out = ic.delete_snr_calibrations(store, dry_run=False, user="t")
        assert out["ok"] is True
        assert out["summary"]["n_buckets"] == 0
        assert out["buckets"] == []

    def test_delete_cleans_up_legacy_per_width_buckets(self):
        # Legacy dmNNNN_wWWWW entries (pre-F-fix-injector-fluence-norm)
        # are stale but must remain visible to delete_snr_calibrations
        # so the operator can wipe them. Mix them with new dmNNNN
        # entries and confirm both are listed and removable.
        store = _FakeStoreRW()
        cs = ic.CalibrationStore(store)
        # Legacy: per-(DM, width)
        cs.put(ic.CalibrationEntry(
            bucket="dm0500_w0032", dm_pc_cm3_rounded=500, width_samples=32,
            K=99.0, last_fluence_jy_ms=100.0, last_observed_snr=99.0,
            last_inj_id="legacy_w32", last_calibrated_at_unix=1.0,
        ))
        cs.put(ic.CalibrationEntry(
            bucket="dm0500_w0064", dm_pc_cm3_rounded=500, width_samples=64,
            K=12.0, last_fluence_jy_ms=100.0, last_observed_snr=15.0,
            last_inj_id="legacy_w64", last_calibrated_at_unix=2.0,
        ))
        # New: DM-only
        cs.put(ic.CalibrationEntry(
            bucket="dm1500", dm_pc_cm3_rounded=1500, width_samples=32,
            K=8.0, last_fluence_jy_ms=100.0, last_observed_snr=14.0,
            last_inj_id="new", last_calibrated_at_unix=3.0,
        ))
        out = ic.delete_snr_calibrations(store, dry_run=False, user="t")
        assert out["ok"] is True
        assert out["summary"]["n_buckets"] == 3
        assert out["summary"]["n_deleted"] == 3
        # Inspecting the bucket rows directly confirms we hit both
        # the legacy keys and the new key.
        seen = sorted(b["bucket"] for b in out["buckets"])
        assert seen == ["dm0500_w0032", "dm0500_w0064", "dm1500"]


# ---------------------------------------------------------------------------
# T6/T7 — health-gated calibration probe + SNR ladder
# ---------------------------------------------------------------------------


def _seed_corr_fast_state(
    store: FakeStore,
    *,
    chgroups=range(16),
    inject_n_queued: int = 0,
    ts_wall_unix: float | None = None,
) -> None:
    """Seed all 16 corr_fast mon-keys with a healthy heartbeat."""
    if ts_wall_unix is None:
        ts_wall_unix = time.time()
    for cg in chgroups:
        store.put_dict(
            f"/mon/corr_rt/{cg}/corr_fast",
            {
                "chgroup": cg,
                "block_n": 1000,
                "n_processed": 1000,
                "n_drop": 0,
                "n_tx": 1000,
                "last_block_ms": 50.0,
                "ts_wall_unix": ts_wall_unix,
                "inject_n_events": inject_n_queued,
                "inject_n_queued": inject_n_queued,
            },
        )


def _seed_search_compute_state(
    store: FakeStore,
    *,
    halves=((1, 0), (1, 1), (2, 0), (2, 1), (9, 0), (9, 1), (13, 0), (13, 1)),
    ts_wall_unix: float | None = None,
    metering_active: int = 0,
) -> None:
    if ts_wall_unix is None:
        ts_wall_unix = time.time()
    for sid, g in halves:
        store.put_dict(
            f"/mon/search_rt/{sid}/compute/{g}",
            {
                "search_node_id": sid,
                "gpu_half": g,
                "c1_metering_active": metering_active,
                "c1_metering_frac": 0.0,
                "c1_metered_dropped_mean": 0.0,
                "c1_max_candidates_per_block": 8,
                "ts_wall_unix": ts_wall_unix,
            },
        )


class TestPrecheckCalibrationHealth:
    def test_healthy_fleet_returns_ok(self):
        store = FakeStore()
        _seed_corr_fast_state(store, inject_n_queued=42)
        _seed_search_compute_state(store)
        snap = ic.precheck_calibration_health(store)
        assert snap.ok is True
        assert snap.reason == "ok"
        # Baselines captured for fan-out check.
        assert snap.inject_baselines[0] == 42
        assert len(snap.inject_baselines) == 16

    def test_stale_corr_fast_blocks_probe(self):
        store = FakeStore()
        # Half the fleet is stale (heartbeat too old).
        _seed_corr_fast_state(
            store, chgroups=range(8), ts_wall_unix=time.time(),
        )
        _seed_corr_fast_state(
            store, chgroups=range(8, 16), ts_wall_unix=time.time() - 1000.0,
        )
        _seed_search_compute_state(store)
        snap = ic.precheck_calibration_health(store)
        assert snap.ok is False
        assert "corr_fast_stale" in snap.reason
        assert set(snap.stale_chgroups) == set(range(8, 16))

    def test_active_metering_blocks_probe_when_required(self):
        store = FakeStore()
        _seed_corr_fast_state(store)
        # All search halves healthy except (1, 0), which is metering.
        _seed_search_compute_state(
            store, halves=((1, 1), (2, 0), (2, 1)),
        )
        store.put_dict(
            "/mon/search_rt/1/compute/0",
            {
                "ts_wall_unix": time.time(),
                "c1_metering_active": 1,
                "c1_metering_frac": 0.5,
                "c1_metered_dropped_mean": 12.0,
                "c1_max_candidates_per_block": 8,
            },
        )
        snap = ic.precheck_calibration_health(
            store,
            search_halves=((1, 0), (1, 1), (2, 0), (2, 1)),
        )
        assert snap.ok is False
        assert "search_unhealthy" in snap.reason
        assert (1, 0) in snap.sick_search_halves

    def test_metering_check_can_be_disabled(self):
        store = FakeStore()
        _seed_corr_fast_state(store)
        store.put_dict(
            "/mon/search_rt/1/compute/0",
            {
                "ts_wall_unix": time.time(),
                "c1_metering_active": 1,  # would normally fail
                "c1_metering_frac": 0.5,
                "c1_metered_dropped_mean": 12.0,
                "c1_max_candidates_per_block": 8,
            },
        )
        snap = ic.precheck_calibration_health(
            store,
            search_halves=((1, 0),),
            require_metering_inactive=False,
        )
        assert snap.ok is True


class TestPrecheckInjectFanOut:
    def test_advanced_baselines_pass(self):
        store = FakeStore()
        _seed_corr_fast_state(store, inject_n_queued=10)
        # Bump every chgroup's queued counter to simulate a successful
        # fan-out.
        _seed_corr_fast_state(store, inject_n_queued=11)
        baselines = {cg: 10 for cg in range(16)}
        ok, lagging = ic.precheck_inject_fan_out(
            store, baselines=baselines, poll_timeout_s=0.0,
            sleep_fn=lambda s: None,
        )
        assert ok is True
        assert lagging == ()

    def test_partial_fan_out_lists_laggards(self):
        store = FakeStore()
        # Half the fleet bumped, half didn't.
        _seed_corr_fast_state(store, chgroups=range(8), inject_n_queued=11)
        _seed_corr_fast_state(store, chgroups=range(8, 16), inject_n_queued=10)
        baselines = {cg: 10 for cg in range(16)}
        ok, lagging = ic.precheck_inject_fan_out(
            store, baselines=baselines, poll_timeout_s=0.0,
            sleep_fn=lambda s: None,
        )
        assert ok is False
        assert sorted(lagging) == list(range(8, 16))


class TestFireCalibrationProbeWithLadder:
    @staticmethod
    def _good_inject(store, **kwargs):
        return TestFireCalibrationProbe._good_inject(store, **kwargs)

    def _seed_match(self, *args, **kwargs):
        return TestFireCalibrationProbe()._seed_match(*args, **kwargs)

    def test_health_check_failure_aborts_before_inject(self):
        # Empty store ⇒ no corr_fast heartbeats ⇒ all chgroups stale.
        store = FakeStore()
        attempts = ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=self._good_inject,
            dm_pc_cm3=500.0,
            time_fn=lambda: 100.0,
            sleep_fn=lambda s: None,
        )
        assert len(attempts) == 1
        assert attempts[0].ok is False
        assert attempts[0].reason.startswith("health_check_failed")
        # Most importantly, no inject_fn / publish_active_inject calls.
        assert all(
            not k.startswith(ic.ACTIVE_INJECT_PREFIX)
            for k, _ in store.put_calls
        )

    def test_first_attempt_succeeds(self):
        store = FakeStore()
        _seed_corr_fast_state(store)
        _seed_search_compute_state(store)
        now = [100.0]
        sleep_calls: List[float] = []

        def sleep_fn(s):
            sleep_calls.append(s)
            now[0] += s
            if len(sleep_calls) == 1:
                # Bump corr_fast counters so the post-fire fan-out
                # check passes.
                _seed_corr_fast_state(store, inject_n_queued=1)
                inj_id = ic._build_probe_inj_id(
                    prefix="cal_probe", dm=500.0, width=32, timestamp=100.0,
                )
                self._seed_match(
                    store, inj_id, observed_snr=20.0,
                    K=20.0 * math.sqrt(32.0 / 100.0),
                )

        attempts = ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=self._good_inject,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=100.0,
            poll_timeout_s=10.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
            fluence_ladder=(1.0, 2.0, 4.0),
        )
        assert len(attempts) == 1
        assert attempts[0].ok is True

    def test_no_match_climbs_ladder(self):
        store = FakeStore()
        _seed_corr_fast_state(store)
        _seed_search_compute_state(store)
        now = [200.0]
        ladder_attempts: List[float] = []

        # Inject_fn captures the fluence each attempt fires at.
        good_inject = self._good_inject

        def inject_fn(store, **kwargs):
            ladder_attempts.append(float(kwargs["fluence_jy_ms"]))
            # Bump counters so fan-out check passes.
            _seed_corr_fast_state(store, inject_n_queued=len(ladder_attempts))
            return good_inject(store, **kwargs)

        # Sleep advances the clock; after the THIRD attempt is fired
        # we seed the match so that attempt succeeds.
        attempt_idx = [0]

        def sleep_fn(s):
            now[0] += s
            # Once the third inject has been fired, seed its match.
            if len(ladder_attempts) >= 3 and attempt_idx[0] < 1:
                attempt_idx[0] += 1
                inj_id = ic._build_probe_inj_id(
                    prefix="cal_probe", dm=500.0, width=32,
                    timestamp=now[0] - s,
                )
                # Match seeded under the inj_id of the most recent
                # attempt — fire_calibration_probe rebuilds inj_id
                # inside each call, so re-derive the timestamp it used.

        attempts = ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=inject_fn,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=2e-4,
            poll_timeout_s=2.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
            fluence_ladder=(1.0, 2.0, 4.0),
        )
        # All 3 attempts should have fired (at 2e-4, 4e-4, 8e-4 Jy·ms
        # — all below the 1.2e-3 saturation clamp).
        assert ladder_attempts == [2e-4, 4e-4, 8e-4]
        # All attempts should ultimately fail ``no_match`` since we
        # never actually seeded a match key.
        assert len(attempts) == 3
        assert all(not a.ok for a in attempts)
        assert all(a.reason.startswith("no_match") for a in attempts)

    def test_ladder_clamps_to_max_probe_fluence_and_stops(self):
        """2026-06-09 saturation guard: a base fluence above
        ``max_probe_fluence`` is clamped, and the ladder does NOT
        escalate past the clamp (every further attempt would refire
        at the same fluence)."""
        store = FakeStore()
        _seed_corr_fast_state(store)
        _seed_search_compute_state(store)
        now = [200.0]
        ladder_attempts: List[float] = []
        good_inject = self._good_inject

        def inject_fn(store, **kwargs):
            ladder_attempts.append(float(kwargs["fluence_jy_ms"]))
            _seed_corr_fast_state(
                store, inject_n_queued=len(ladder_attempts),
            )
            return good_inject(store, **kwargs)

        def sleep_fn(s):
            now[0] += s

        attempts = ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=inject_fn,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=100.0,   # pre-2026-06-09 default: saturates
            poll_timeout_s=2.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
            fluence_ladder=(1.0, 2.0, 4.0),
        )
        assert ladder_attempts == [ic.DEFAULT_MAX_PROBE_FLUENCE]
        assert len(attempts) == 1
        assert not attempts[0].ok

    def test_saturated_match_refuses_K_and_ladder_descends(self):
        """2026-06-10 saturation guard: an observed SNR pinned at the
        detector's ±250σ input-clip rail carries no amplitude
        information — fire_calibration_probe must NOT store K
        (observed live: DM-1000 probe at 1.2e-3 Jy·ms reported
        snr=250.25 → bogus K=14448 stored). The ladder must then
        retry DOWNWARD (×¼), not escalate."""
        store = FakeStore()
        _seed_corr_fast_state(store)
        _seed_search_compute_state(store)
        now = [200.0]
        ladder_attempts: List[float] = []
        good_inject = self._good_inject

        def inject_fn(store, **kwargs):
            ladder_attempts.append(float(kwargs["fluence_jy_ms"]))
            _seed_corr_fast_state(
                store, inject_n_queued=len(ladder_attempts),
            )
            return good_inject(store, **kwargs)

        def sleep_fn(s):
            now[0] += s

        # Seed a SATURATED match for attempt 1's inj_id (built at the
        # initial clock value).
        inj_id = ic._build_probe_inj_id(
            prefix="cal_probe", dm=1000.0, width=4, timestamp=200.0,
        )
        self._seed_match(
            store, inj_id, observed_snr=250.25,
            K=250.25 * math.sqrt(4.0 / 7e-4),
        )

        attempts = ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=inject_fn,
            dm_pc_cm3=1000.0,
            width_samples=4,
            fluence_jy_ms=7e-4,
            poll_timeout_s=2.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
            fluence_ladder=(1.0, 2.0, 4.0),
        )
        # Attempt 1: saturated, no K stored. Attempt 2: descended ×¼.
        assert attempts[0].ok is False
        assert attempts[0].reason.startswith("saturated")
        assert attempts[0].K is None
        assert store.get_dict(
            f"{ic.CALIBRATION_PREFIX}dm1000"
        ) is None, "saturated probe must NOT write the calibration"
        assert len(ladder_attempts) == 2
        assert ladder_attempts[1] == pytest.approx(7e-4 / 4.0)
        # The descent went unmatched here (no seed) → no_match, and
        # the ladder does NOT climb back up.
        assert len(attempts) == 2
        assert attempts[1].reason.startswith("no_match")

    def test_ladder_sleeps_between_attempts(self):
        """sigma_k-recovery delay: ladder attempt N>1 is preceded by a
        ``ladder_step_delay_s`` sleep."""
        store = FakeStore()
        _seed_corr_fast_state(store)
        _seed_search_compute_state(store)
        now = [200.0]
        ladder_attempts: List[float] = []
        sleeps: List[float] = []
        good_inject = self._good_inject

        def inject_fn(store, **kwargs):
            ladder_attempts.append(float(kwargs["fluence_jy_ms"]))
            _seed_corr_fast_state(
                store, inject_n_queued=len(ladder_attempts),
            )
            return good_inject(store, **kwargs)

        def sleep_fn(s):
            sleeps.append(float(s))
            now[0] += s

        ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=inject_fn,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=2e-4,
            poll_timeout_s=2.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
            fluence_ladder=(1.0, 2.0),
            ladder_step_delay_s=60.0,
        )
        assert ladder_attempts == [2e-4, 4e-4]
        assert sleeps.count(60.0) == 1

    def test_partial_fan_out_aborts_ladder(self):
        store = FakeStore()
        _seed_corr_fast_state(store, chgroups=range(16), inject_n_queued=0)
        _seed_search_compute_state(store)

        # The inject helper writes its own active-inject row but does
        # NOT bump per-chgroup inject_n_queued (simulating partial
        # fan-out — some corr_fast nodes never receive the cmd).
        good_inject = self._good_inject

        def inject_fn(store, **kwargs):
            return good_inject(store, **kwargs)

        # Use an advancing clock so the no_match poll loop terminates
        # inside fire_calibration_probe (poll_timeout_s = 1.0).
        now = [300.0]

        def sleep_fn(s):
            now[0] += s

        attempts = ic.fire_calibration_probe_with_ladder(
            store,
            inject_fn=inject_fn,
            dm_pc_cm3=500.0,
            width_samples=32,
            fluence_jy_ms=100.0,
            poll_timeout_s=1.0,
            poll_interval_s=0.1,
            fan_out_check_timeout_s=0.0,
            time_fn=lambda: now[0],
            sleep_fn=sleep_fn,
            fluence_ladder=(1.0, 2.0),
        )
        # Should have ONE attempt followed by partial_fan_out abort —
        # the ladder doesn't escalate past a partial fan-out (a missing
        # corr node won't be cured by more fluence).
        assert len(attempts) == 1
        assert not attempts[0].ok
        assert attempts[0].reason.startswith("partial_fan_out")
