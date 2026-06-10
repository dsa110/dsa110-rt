"""Unit tests for :mod:`dsart.coinc.inject_match`."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Dict, List, Mapping
from unittest.mock import MagicMock

import pytest

from dsart.coinc.inject_match import (
    ACTIVE_INJECT_PREFIX,
    DEFAULT_DM_TOL_FRAC,
    DEFAULT_HISTORY_DEPTH,
    DEFAULT_LM_TOL_RAD,
    DEFAULT_MAX_OBSERVED_SNR,
    MATCH_EVENT_PREFIX,
    ActiveInjection,
    InjectionMatcher,
    MatchResult,
    build_active_inject_key,
    build_match_event_key,
    compute_k_inferred,
)


# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _make_inj_payload(
    *,
    inj_id: str = "inj_test_1",
    dm: float = 500.0,
    l: float = 0.0,
    m: float = 0.0,
    width: int = 32,
    fluence: float = 100.0,
    apply_at: int = 1_000_000,
    fired_at: float | None = None,
    ttl: float = 60.0,
    fired_by: str = "test",
    target_snr: Any = None,
) -> Dict[str, Any]:
    # Default to "now" so the eviction-grace sweep in
    # InjectionMatcher.refresh_if_due treats the entry as fresh. Tests
    # that need a specific age (TTL / expiry / grace) pass an explicit
    # ``fired_at`` and use a ``time_fn`` to anchor the matcher's clock.
    if fired_at is None:
        fired_at = time.time()
    payload: Dict[str, Any] = {
        "inj_id": inj_id,
        "dm_pc_cm3": dm,
        "l_rad": l,
        "m_rad": m,
        "width_samples": width,
        "fluence_jy_ms": fluence,
        "apply_at_specnum": apply_at,
        "fired_at_unix": fired_at,
        "ttl_s": ttl,
        "fired_by": fired_by,
    }
    if target_snr is not None:
        payload["target_snr"] = target_snr
    return payload


class FakeEtcd:
    """Minimal stand-in for the etcd3 client's ``get_prefix`` /
    ``delete`` surface used by :class:`InjectionMatcher`.

    ``delete`` removes the key from the in-memory store and records
    the call in :attr:`deletes` so eviction tests can assert on it.
    Inject a custom ``delete`` by overwriting the attribute to simulate
    transport errors.
    """

    def __init__(self, payloads: Mapping[str, Mapping[str, Any]]):
        self._payloads = {
            k: json.dumps(v).encode("utf-8") for k, v in payloads.items()
        }
        self.deletes: List[str] = []

    def get_prefix(self, prefix: str):
        for k, v in list(self._payloads.items()):
            if k.startswith(prefix):
                yield v, None

    def delete(self, key: str) -> None:
        self.deletes.append(key)
        self._payloads.pop(key, None)


class FakeStore:
    """Duck-typed DsaStore for the matcher (get_etcd + put_dict)."""

    def __init__(self, payloads: Mapping[str, Mapping[str, Any]] | None = None):
        self._etcd = FakeEtcd(payloads or {})
        self.puts: List[tuple[str, Dict[str, Any]]] = []
        self._put_should_fail = False
        self._lock = threading.Lock()

    def set_active(self, payloads: Mapping[str, Mapping[str, Any]]) -> None:
        self._etcd = FakeEtcd(payloads)

    def get_etcd(self) -> FakeEtcd:
        return self._etcd

    def put_dict(self, key: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if self._put_should_fail:
                raise RuntimeError("etcd put injected failure")
            self.puts.append((key, dict(payload)))

    def fail_puts(self, fail: bool) -> None:
        with self._lock:
            self._put_should_fail = bool(fail)


# ---------------------------------------------------------------------------
# compute_k_inferred
# ---------------------------------------------------------------------------


class TestComputeKInferred:
    def test_round_trip(self):
        # Pick a K, derive observed_snr, invert.
        K_true = 7.5
        fluence = 50.0
        width = 32
        # 2026-06-10 linear model: observed_snr = K * fluence / sqrt(width)
        observed = K_true * fluence / math.sqrt(width)
        K_back = compute_k_inferred(
            observed_snr=observed,
            fluence_jy_ms=fluence,
            width_samples=width,
        )
        assert K_back == pytest.approx(K_true, rel=1e-9)

    @pytest.mark.parametrize(
        "snr,fluence,width",
        [
            (-1.0, 50.0, 32),
            (10.0, -50.0, 32),
            (10.0, 50.0, 0),
            (0.0, 50.0, 32),
            (float("nan"), 50.0, 32),
            (10.0, float("nan"), 32),
        ],
    )
    def test_bad_inputs_return_nan(self, snr, fluence, width):
        K = compute_k_inferred(
            observed_snr=snr, fluence_jy_ms=fluence, width_samples=width,
        )
        assert math.isnan(K)


# ---------------------------------------------------------------------------
# ActiveInjection.from_dict
# ---------------------------------------------------------------------------


class TestActiveInjectionFromDict:
    def test_round_trip(self):
        p = _make_inj_payload(inj_id="aa", target_snr=20.0)
        inj = ActiveInjection.from_dict(p)
        assert inj.inj_id == "aa"
        assert inj.dm_pc_cm3 == 500.0
        assert inj.target_snr == 20.0
        assert inj.ttl_s == 60.0
        assert inj.is_expired(p["fired_at_unix"] + p["ttl_s"] + 0.001) is True
        assert inj.is_expired(p["fired_at_unix"]) is False

    def test_target_snr_none_when_missing(self):
        p = _make_inj_payload()
        inj = ActiveInjection.from_dict(p)
        assert inj.target_snr is None

    def test_bad_target_snr_falls_back_to_none(self):
        p = _make_inj_payload(target_snr="not-a-number")
        inj = ActiveInjection.from_dict(p)
        assert inj.target_snr is None

    @pytest.mark.parametrize(
        "missing_key",
        ["inj_id", "dm_pc_cm3", "l_rad", "m_rad", "width_samples",
         "fluence_jy_ms", "apply_at_specnum", "fired_at_unix"],
    )
    def test_missing_keys_raise(self, missing_key):
        p = _make_inj_payload()
        del p[missing_key]
        with pytest.raises(ValueError):
            ActiveInjection.from_dict(p)

    def test_zero_dm_raises(self):
        with pytest.raises(ValueError):
            ActiveInjection.from_dict(_make_inj_payload(dm=0.0))

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            ActiveInjection.from_dict(_make_inj_payload(width=0))


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


class TestKeyBuilders:
    def test_active_key(self):
        assert build_active_inject_key("aa") == ACTIVE_INJECT_PREFIX + "aa"

    def test_match_key(self):
        assert build_match_event_key("aa") == MATCH_EVENT_PREFIX + "aa"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            build_active_inject_key("")
        with pytest.raises(ValueError):
            build_match_event_key("")


# ---------------------------------------------------------------------------
# Matcher.refresh_if_due
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_refresh_populates_active(self):
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(inj_id="aa"),
            ACTIVE_INJECT_PREFIX + "bb": _make_inj_payload(
                inj_id="bb", dm=150.0,
            ),
        })
        m = InjectionMatcher(store=store, refresh_s=0.1)
        assert m.refresh_if_due(force=True) is True
        snap = m.snapshot()
        assert snap["active_count"] == 2
        ids = {a["inj_id"] for a in snap["active"]}
        assert ids == {"aa", "bb"}

    def test_refresh_obeys_cadence(self):
        store = FakeStore()
        now = [1000.0]
        m = InjectionMatcher(
            store=store, refresh_s=1.0, time_fn=lambda: now[0],
        )
        assert m.refresh_if_due() is True   # first call always refreshes
        assert m.refresh_if_due() is False  # cadence not elapsed
        now[0] += 1.5
        assert m.refresh_if_due() is True   # past cadence

    def test_refresh_removes_dropped_ids(self):
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(inj_id="aa"),
        })
        m = InjectionMatcher(store=store, refresh_s=0.1)
        m.refresh_if_due(force=True)
        assert m.snapshot()["active_count"] == 1
        # Drop aa, add bb.
        store.set_active({
            ACTIVE_INJECT_PREFIX + "bb": _make_inj_payload(
                inj_id="bb", dm=150.0,
            ),
        })
        m.refresh_if_due(force=True)
        snap = m.snapshot()
        assert {a["inj_id"] for a in snap["active"]} == {"bb"}

    def test_refresh_skips_bad_rows(self):
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(inj_id="aa"),
            ACTIVE_INJECT_PREFIX + "bad": {"junk": True},  # missing keys
        })
        m = InjectionMatcher(store=store, refresh_s=0.1)
        m.refresh_if_due(force=True)
        snap = m.snapshot()
        assert {a["inj_id"] for a in snap["active"]} == {"aa"}
        # Refresh still counted as ok (one bad row doesn't fail the
        # whole refresh).
        assert snap["refresh_ok"] == 1

    def test_refresh_failure_counted(self):
        # Build a duck-typed store whose get_etcd raises — without the
        # MagicMock lazy attribute pitfall (the matcher checks for
        # ControlStore-style _ensure/_store, and MagicMock auto-spawns
        # both, which would route around get_etcd entirely).
        class BrokenStore:
            def get_etcd(self):
                raise RuntimeError("etcd down")
            def put_dict(self, key, payload):  # pragma: no cover
                pass
        store = BrokenStore()
        m = InjectionMatcher(store=store, refresh_s=0.1)
        assert m.refresh_if_due(force=True) is False
        snap = m.snapshot()
        assert snap["refresh_fail"] == 1

    # ---- expiry-grace eviction sweep (M7.4 Phase 6c follow-up) ----

    def test_evicts_expired_after_grace(self):
        """An entry past ``fired_at + ttl + grace`` is deleted from etcd
        and dropped from the matcher's in-memory ``_active``.

        Mirrors the production sweep that keeps stale ``cal_probe_*``
        rows from accumulating in ``/cnf/inject/active/*`` after the
        dashboard fires + the C2 matcher consumes the result.
        """
        fired = 1_000.0
        ttl = 5.0
        grace = 2.0
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                inj_id="aa", fired_at=fired, ttl=ttl,
            ),
        })
        # Anchor matcher clock past ttl + grace.
        now = [fired + ttl + grace + 0.001]
        m = InjectionMatcher(
            store=store,
            refresh_s=0.1,
            time_fn=lambda: now[0],
            expiry_grace_s=grace,
        )
        m.refresh_if_due(force=True)

        snap = m.snapshot()
        assert snap["active_count"] == 0
        assert snap["evicted_expired"] == 1
        assert snap["evict_fail"] == 0
        assert store.get_etcd().deletes == [build_active_inject_key("aa")]

    def test_does_not_evict_within_grace(self):
        """An entry past ``ttl`` but still within the grace window is
        excluded from new matches by ``try_match`` (via ``is_expired``)
        but its etcd key is NOT deleted yet — gives a late C1 candidate
        a chance to surface before the row vanishes."""
        fired = 1_000.0
        ttl = 5.0
        grace = 2.0
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                inj_id="aa", fired_at=fired, ttl=ttl,
            ),
        })
        now = [fired + ttl + grace - 0.001]   # within grace, just under
        m = InjectionMatcher(
            store=store,
            refresh_s=0.1,
            time_fn=lambda: now[0],
            expiry_grace_s=grace,
        )
        m.refresh_if_due(force=True)

        snap = m.snapshot()
        assert snap["active_count"] == 1
        assert snap["evicted_expired"] == 0
        assert store.get_etcd().deletes == []

    def test_evict_fail_counted_when_delete_raises(self):
        """``delete`` failures are swallowed + counted: the row is still
        excluded from the new ``_active`` snapshot, but the etcd key
        survives so the next refresh can re-try."""
        fired = 1_000.0
        ttl = 5.0
        grace = 2.0
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                inj_id="aa", fired_at=fired, ttl=ttl,
            ),
        })

        def _boom(key):
            raise RuntimeError("etcd refused")

        store._etcd.delete = _boom

        now = [fired + ttl + grace + 1.0]
        m = InjectionMatcher(
            store=store,
            refresh_s=0.1,
            time_fn=lambda: now[0],
            expiry_grace_s=grace,
        )
        m.refresh_if_due(force=True)

        snap = m.snapshot()
        assert snap["evict_fail"] == 1
        assert snap["evicted_expired"] == 0
        assert snap["active_count"] == 0    # still filtered out of _active

    def test_eviction_disabled_when_grace_none(self):
        """Operators can pin ``expiry_grace_s=None`` to disable eviction
        (e.g. forensic replay where we want to keep every fired probe
        visible regardless of TTL)."""
        fired = 1_000.0
        ttl = 5.0
        store = FakeStore({
            ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                inj_id="aa", fired_at=fired, ttl=ttl,
            ),
        })
        now = [fired + ttl + 100.0]   # well past TTL
        m = InjectionMatcher(
            store=store,
            refresh_s=0.1,
            time_fn=lambda: now[0],
            expiry_grace_s=None,
        )
        m.refresh_if_due(force=True)

        snap = m.snapshot()
        assert snap["active_count"] == 1
        assert snap["evicted_expired"] == 0
        assert store.get_etcd().deletes == []

    def test_eviction_negative_grace_rejected(self):
        with pytest.raises(ValueError, match="expiry_grace_s"):
            InjectionMatcher(store=FakeStore(), expiry_grace_s=-1.0)


# ---------------------------------------------------------------------------
# Matcher.try_match
# ---------------------------------------------------------------------------


class TestTryMatch:
    def _matcher(self, **overrides) -> tuple[InjectionMatcher, FakeStore]:
        payloads = overrides.pop(
            "payloads",
            {
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=500.0, l=0.0, m=0.0,
                    width=32, fluence=100.0, fired_at=1_000.0, ttl=60.0,
                ),
            },
        )
        store = FakeStore(payloads)
        now = overrides.pop("now", 1_001.0)
        time_fn = overrides.pop("time_fn", lambda: now)
        m = InjectionMatcher(
            store=store, refresh_s=0.1, time_fn=time_fn, **overrides,
        )
        m.refresh_if_due(force=True)
        return m, store

    def _row_args(self, **kw) -> Dict[str, Any]:
        base = dict(
            snr=20.0,
            dm_pc_cc=500.0,
            l_rad=0.0,
            m_rad=0.0,
            width_samples=32,
            # Consistent with the default payload's apply_at=1_000_000
            # specnums: expected arrival = 1_000_000 / 16 = 62_500
            # search samples (the 2026-06-09 specnum-proximity gate).
            event_specnum=62_500,
            kernel_id="k0",
            search_node_id=1,
            gpu_half=0,
            mjd=60000.0,
        )
        base.update(kw)
        return base

    def test_exact_match(self):
        m, store = self._matcher()
        mr = m.try_match(**self._row_args())
        assert mr is not None
        assert mr.inj_id == "aa"
        assert mr.observed_snr == 20.0
        # 2026-06-10 linear fluence model:
        # K = observed × sqrt(width) / fluence = 20 × sqrt(32) / 100
        assert mr.K_inferred == pytest.approx(
            20.0 * math.sqrt(32.0) / 100.0, rel=1e-9,
        )
        # And a publish happened.
        assert len(store.puts) == 1
        key, payload = store.puts[0]
        assert key == build_match_event_key("aa")
        assert payload["best"]["observed_snr"] == 20.0
        assert payload["n_matches"] == 1
        assert payload["active"]["dm_pc_cm3"] == 500.0

    def test_dm_outside_tolerance(self):
        m, _ = self._matcher()
        # 10 % off — outside default 5 % tolerance.
        mr = m.try_match(**self._row_args(dm_pc_cc=550.0))
        assert mr is None

    def test_dm_within_tolerance(self):
        m, _ = self._matcher()
        # 2 % off — well within.
        mr = m.try_match(**self._row_args(dm_pc_cc=510.0))
        assert mr is not None

    def test_lm_outside_tolerance(self):
        m, _ = self._matcher()
        mr = m.try_match(
            **self._row_args(l_rad=DEFAULT_LM_TOL_RAD * 2.0),
        )
        assert mr is None

    def test_expired_injection_skipped(self):
        m, _ = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", fired_at=1_000.0, ttl=5.0,
                ),
            },
            now=1_010.0,  # past ttl
        )
        mr = m.try_match(**self._row_args())
        assert mr is None

    def test_best_improves_publishes_new(self):
        m, store = self._matcher()
        m.try_match(**self._row_args(snr=10.0))
        assert len(store.puts) == 1
        m.try_match(**self._row_args(snr=12.0))  # improves
        assert len(store.puts) == 2
        m.try_match(**self._row_args(snr=11.0))  # does NOT improve
        assert len(store.puts) == 2
        # Best stays at 12.0
        snap = m.snapshot()
        assert snap["best"]["aa"]["observed_snr"] == 12.0
        # History records all 3.
        assert snap["matches"] == 3
        assert snap["best_improved"] == 2

    def test_history_capped(self):
        m, store = self._matcher(history_depth=4)
        for snr in (10.0, 11.0, 12.0, 13.0, 14.0, 15.0):
            m.try_match(**self._row_args(snr=snr))
        # History should be capped at 4 entries; published payload
        # carries the last 4.
        last_key, last_payload = store.puts[-1]
        assert len(last_payload["history"]) == 4
        # Best stays at the running max.
        assert last_payload["best"]["observed_snr"] == 15.0

    def test_multiple_active_injections_match_separately(self):
        m, store = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=500.0, l=0.0, m=0.0,
                    fired_at=1_000.0, ttl=60.0,
                ),
                ACTIVE_INJECT_PREFIX + "bb": _make_inj_payload(
                    inj_id="bb", dm=150.0, l=0.0, m=0.0,
                    fired_at=1_000.0, ttl=60.0,
                ),
            },
            now=1_010.0,
        )
        # Row at DM=500 matches "aa" only.
        m.try_match(**self._row_args(dm_pc_cc=500.0, snr=20.0))
        # Row at DM=150 matches "bb" only.
        m.try_match(**self._row_args(dm_pc_cc=150.0, snr=10.0))
        snap = m.snapshot()
        assert set(snap["best"].keys()) == {"aa", "bb"}
        assert snap["best"]["aa"]["observed_snr"] == 20.0
        assert snap["best"]["bb"]["observed_snr"] == 10.0

    def test_publish_failure_counted(self):
        m, store = self._matcher()
        store.fail_puts(True)
        m.try_match(**self._row_args())
        snap = m.snapshot()
        assert snap["publish_fail"] == 1
        assert snap["publish_ok"] == 0
        # Best still recorded locally even though publish failed.
        assert snap["best"]["aa"]["observed_snr"] == 20.0

    def test_matched_inj_id_no_side_effects(self):
        m, store = self._matcher()
        out = m.matched_inj_id(dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0)
        assert out == "aa"
        # No puts.
        assert store.puts == []
        # No counter advance.
        snap = m.snapshot()
        assert snap["matches"] == 0

    def test_matched_inj_id_returns_none_when_no_match(self):
        m, _ = self._matcher()
        out = m.matched_inj_id(dm_pc_cc=999.0, l_rad=0.0, m_rad=0.0)
        assert out is None

    def test_rejects_sigma_scaling_glitch_by_snr_ceiling(self):
        """A width-1 60kσ hot-pixel glitch must NOT calibrate a probe,
        but the rejection comes from the SNR ceiling — not a width gate
        (injection width is native ~32µs, observed width is search ~1ms,
        so they are not comparable)."""
        m, store = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=150.0, width=32, fluence=10.0,
                    fired_at=1_000.0, ttl=60.0,
                ),
            },
            now=1_001.0,
        )
        mr = m.try_match(
            **self._row_args(
                dm_pc_cc=146.68,
                snr=60448.0,
                width_samples=1,
                l_rad=0.0192,
                m_rad=0.01905,
            ),
        )
        assert mr is None
        assert store.puts == []
        assert m.snapshot()["rows_rejected_quality"] == 1

    def test_accepts_real_injection_at_fleet_lm_offset(self):
        """A boresight-declared injection imaging at the documented
        fleet l/m≈0.019 offset with a low search-width must MATCH
        (regression: earlier width-ratio + l/m gates rejected these)."""
        m, store = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=900.0, l=0.0, m=0.0, width=32,
                    fluence=100.0, fired_at=1_000.0, ttl=60.0,
                ),
            },
            now=1_001.0,
        )
        mr = m.try_match(
            **self._row_args(
                dm_pc_cc=903.8,
                snr=20.34,
                width_samples=4,   # search samples; inj is 32 native
                l_rad=0.0192,
                m_rad=0.0191,
            ),
        )
        assert mr is not None
        assert mr.inj_id == "aa"
        assert mr.observed_snr == pytest.approx(20.34)
        assert len(store.puts) == 1

    def test_rejects_snr_above_ceiling(self):
        m, _ = self._matcher(max_observed_snr=500.0)
        mr = m.try_match(**self._row_args(snr=600.0))
        assert mr is None

    def test_prefers_closer_lm_over_higher_snr(self):
        """Quality ordering: boresight moderate SNR beats off-axis bright."""
        m, store = self._matcher()
        m.try_match(**self._row_args(snr=50.0, l_rad=0.03, m_rad=0.03))
        m.try_match(**self._row_args(snr=25.0, l_rad=0.0, m_rad=0.0))
        snap = m.snapshot()
        assert snap["best"]["aa"]["observed_snr"] == 25.0
        _, payload = store.puts[-1]
        assert payload["best"]["observed_l_rad"] == 0.0

    def test_width_not_gated(self):
        """Width is no longer a match gate (native-vs-search units)."""
        m, _ = self._matcher()
        assert m.try_match(**self._row_args(width_samples=16)) is not None
        assert m.try_match(**self._row_args(width_samples=1)) is not None

    def test_matched_inj_id_with_quality_gates(self):
        m, _ = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=150.0, width=32,
                    fired_at=1_000.0, ttl=60.0,
                ),
            },
            now=1_001.0,
        )
        # Real low-search-width injection match qualifies.
        assert m.matched_inj_id(
            dm_pc_cc=146.0, l_rad=0.0, m_rad=0.0,
            width_samples=4, snr=20.0,
        ) == "aa"
        # σ-scaling glitch rejected by SNR ceiling.
        assert m.matched_inj_id(
            dm_pc_cc=146.0, l_rad=0.0, m_rad=0.0,
            width_samples=1, snr=60448.0,
        ) is None


# ---------------------------------------------------------------------------
# Specnum-proximity gate (2026-06-09)
# ---------------------------------------------------------------------------


class TestSpecnumGate(TestTryMatch):
    """The 2026-06-09 cross-attribution fix: a row only matches an
    injection whose expected arrival (apply_at_specnum / 16) is within
    ``specnum_tol_samples`` SEARCH samples of the row's event_specnum.

    Reproduces the live-fleet failure: probes fired 30-50 s apart all
    stayed inside the 60 s TTL, so one candidate was attributed to
    every live inj_id (bogus K_inferred for all but the true one).
    """

    # Default payload: apply_at = 1_000_000 ⇒ expected sample 62_500.

    def test_row_inside_window_matches(self):
        m, _ = self._matcher()
        # +1300 samples ≈ DM-2500 dispersion sweep; inside ±4096.
        mr = m.try_match(**self._row_args(event_specnum=62_500 + 1_300))
        assert mr is not None

    def test_dm2500_late_arrival_matches(self):
        """2026-06-10 live regression: DM-2500 probes arrived ~2.5-3 k
        samples late (band-bottom sweep + stage-2 FIFO + batching) and
        were silently rejected by the old ±2048 gate. The widened
        ±4096 default must accept them."""
        m, _ = self._matcher()
        mr = m.try_match(**self._row_args(event_specnum=62_500 + 3_000))
        assert mr is not None
        assert m.snapshot()["rows_rejected_specnum"] == 0

    def test_row_outside_window_rejected(self):
        m, _ = self._matcher()
        # ~42 s late (the live cross-attribution case) ⇒ rejected.
        mr = m.try_match(**self._row_args(event_specnum=62_500 + 39_000))
        assert mr is None
        assert m.snapshot()["rows_rejected_specnum"] == 1

    def test_concurrent_probes_attributed_separately(self):
        """Two live probes 30 s apart: the candidate near probe bb's
        arrival matches ONLY bb, even though aa is also unexpired."""
        spp = 16
        aa_apply, bb_apply = 1_000_000, 1_000_000 + 30 * 938 * spp
        m, _ = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=500.0, apply_at=aa_apply,
                    fired_at=1_000.0, ttl=60.0,
                ),
                ACTIVE_INJECT_PREFIX + "bb": _make_inj_payload(
                    inj_id="bb", dm=500.0, apply_at=bb_apply,
                    fired_at=1_030.0, ttl=60.0,
                ),
            },
            now=1_031.0,
        )
        mr = m.try_match(
            **self._row_args(event_specnum=bb_apply // spp + 140),
        )
        assert mr is not None
        assert mr.inj_id == "bb"
        assert set(m.snapshot()["best"].keys()) == {"bb"}

    def test_origin_mismatch_falls_back_to_wall_clock_accept(self):
        """2026-06-10 live regression: a search-side bounce resets the
        C1 sample counters, so event_specnum no longer shares the corr
        specnum origin and the residual jumps by millions of samples.
        The gate must recognise this as an origin mismatch (residual >
        16 × tol) and fall back to MJD-vs-fired_at — here the row's
        event time is 10 s after the fire, well inside the +20 s
        window, so it matches."""
        m, _ = self._matcher()
        mjd_10s_after_fire = 40587.0 + (1_000.0 + 10.0) / 86400.0
        mr = m.try_match(**self._row_args(
            event_specnum=62_500 + 10_000_000,
            mjd=mjd_10s_after_fire,
        ))
        assert mr is not None
        assert m.snapshot()["rows_rejected_specnum"] == 0

    def test_origin_mismatch_wall_clock_reject_outside_window(self):
        """Origin mismatch AND the row's event time is 45 s after the
        fire (outside the +20 s wall window) → still rejected; the
        fallback must not reopen the cross-attribution hole."""
        m, _ = self._matcher()
        mjd_45s_after_fire = 40587.0 + (1_000.0 + 45.0) / 86400.0
        mr = m.try_match(**self._row_args(
            event_specnum=62_500 + 10_000_000,
            mjd=mjd_45s_after_fire,
        ))
        assert mr is None
        assert m.snapshot()["rows_rejected_specnum"] == 1

    def test_origin_mismatch_wall_clock_reject_before_fire(self):
        """Row event time 10 s BEFORE the fire cannot be the injection
        (window floor is -2 s)."""
        m, _ = self._matcher()
        mjd_10s_before_fire = 40587.0 + (1_000.0 - 10.0) / 86400.0
        mr = m.try_match(**self._row_args(
            event_specnum=62_500 + 10_000_000,
            mjd=mjd_10s_before_fire,
        ))
        assert mr is None
        assert m.snapshot()["rows_rejected_specnum"] == 1

    def test_in_origin_late_row_does_not_use_wall_fallback(self):
        """A row ~42 s late but within the SAME specnum origin
        (residual < 16 × tol) is a genuine timing miss — it must NOT be
        rescued by the wall-clock fallback even if its MJD looks
        plausible."""
        m, _ = self._matcher()
        mjd_5s_after_fire = 40587.0 + (1_000.0 + 5.0) / 86400.0
        mr = m.try_match(**self._row_args(
            event_specnum=62_500 + 39_000,
            mjd=mjd_5s_after_fire,
        ))
        assert mr is None
        assert m.snapshot()["rows_rejected_specnum"] == 1

    def test_gate_disabled_with_none(self):
        m, _ = self._matcher(specnum_tol_samples=None)
        mr = m.try_match(**self._row_args(event_specnum=62_500 + 39_000))
        assert mr is not None

    def test_legacy_payload_without_specnum_skips_gate(self):
        """apply_at_specnum <= 0 (legacy/manual rows) skips the gate."""
        m, _ = self._matcher(
            payloads={
                ACTIVE_INJECT_PREFIX + "aa": _make_inj_payload(
                    inj_id="aa", dm=500.0, apply_at=0,
                    fired_at=1_000.0, ttl=60.0,
                ),
            },
        )
        mr = m.try_match(**self._row_args(event_specnum=999_999))
        assert mr is not None

    def test_matched_inj_id_applies_gate_when_specnum_given(self):
        m, _ = self._matcher()
        assert m.matched_inj_id(
            dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0,
            width_samples=32, snr=20.0,
            event_specnum=62_500 + 100,
        ) == "aa"
        assert m.matched_inj_id(
            dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0,
            width_samples=32, snr=20.0,
            event_specnum=62_500 + 39_000,
        ) is None
        # No specnum supplied (legacy callers): gate skipped.
        assert m.matched_inj_id(
            dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0,
            width_samples=32, snr=20.0,
        ) == "aa"

    def test_bad_tol_raises(self):
        with pytest.raises(ValueError):
            InjectionMatcher(store=FakeStore(), specnum_tol_samples=0.0)
        with pytest.raises(ValueError):
            InjectionMatcher(
                store=FakeStore(), specnums_per_search_sample=0,
            )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestCtor:
    def test_bad_refresh_s_raises(self):
        with pytest.raises(ValueError):
            InjectionMatcher(store=FakeStore(), refresh_s=0.0)

    def test_bad_tolerances_raise(self):
        with pytest.raises(ValueError):
            InjectionMatcher(store=FakeStore(), dm_tol_frac=0.0)
        with pytest.raises(ValueError):
            InjectionMatcher(store=FakeStore(), lm_tol_rad=0.0)

    def test_bad_history_raises(self):
        with pytest.raises(ValueError):
            InjectionMatcher(store=FakeStore(), history_depth=0)

    def test_bad_snr_band_raises(self):
        with pytest.raises(ValueError):
            InjectionMatcher(
                store=FakeStore(),
                min_observed_snr=100.0,
                max_observed_snr=10.0,
            )

    def test_default_quality_constants(self):
        assert DEFAULT_MAX_OBSERVED_SNR == 2000.0
