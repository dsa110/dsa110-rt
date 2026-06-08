"""Unit tests for ``dsart.inject.cal_probe_shadow.CalProbeShadow`` (T3).

The shadow is the search-side mirror of the dashboard's
``/cnf/inject/active/cal_probe_*`` registry. It exempts calibration
probes from the C1→C2 metering cap so an operator-fired probe is never
silently shed during a candidate flood.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from dsart.inject.cal_probe_shadow import (
    CAL_PROBE_INJ_ID_PREFIX,
    DEFAULT_CAL_PROBE_REFRESH_S,
    CalProbeShadow,
)


def _probe_payload(
    *,
    inj_id: str,
    dm: float = 500.0,
    l_rad: float = 0.0,
    m_rad: float = 0.0,
    width: int = 32,
    fluence: float = 100.0,
    apply_at: int = 1234567,
    fired_at: float = 1_700_000_000.0,
    ttl: float = 60.0,
) -> Dict[str, Any]:
    return {
        "inj_id": inj_id,
        "dm_pc_cm3": dm,
        "l_rad": l_rad,
        "m_rad": m_rad,
        "width_samples": width,
        "fluence_jy_ms": fluence,
        "apply_at_specnum": apply_at,
        "fired_at_unix": fired_at,
        "ttl_s": ttl,
        "fired_by": "test",
    }


class _FakeStore:
    def __init__(self, payload: Dict[str, Dict[str, Any]]):
        self._payload = payload
        self.n_calls = 0

    def get_dict_prefix(self, prefix: str) -> Dict[str, Dict[str, Any]]:
        self.n_calls += 1
        return dict(self._payload)


def test_shadow_filters_to_cal_probe_prefix() -> None:
    """Only inj_ids starting with ``cal_probe`` end up in the snapshot."""
    payload = {
        "/cnf/inject/active/cal_probe_dm500_w32_t1": _probe_payload(
            inj_id="cal_probe_dm500_w32_t1", fired_at=time.time(),
        ),
        "/cnf/inject/active/manual_inject_1": _probe_payload(
            inj_id="manual_inject_1", fired_at=time.time(),
        ),
    }
    shadow = CalProbeShadow(store=_FakeStore(payload))
    shadow.maybe_refresh()
    assert shadow.n_active == 1
    snap = shadow.snapshot()
    assert list(snap.keys()) == ["cal_probe_dm500_w32_t1"]


def test_shadow_drops_expired_probes() -> None:
    """Probes whose ``ttl_s + grace`` has elapsed are NOT retained."""
    fired_long_ago = time.time() - 1000.0
    payload = {
        "/cnf/inject/active/cal_probe_old": _probe_payload(
            inj_id="cal_probe_old", fired_at=fired_long_ago, ttl=30.0,
        ),
        "/cnf/inject/active/cal_probe_new": _probe_payload(
            inj_id="cal_probe_new", fired_at=time.time(), ttl=30.0,
        ),
    }
    shadow = CalProbeShadow(store=_FakeStore(payload))
    shadow.maybe_refresh()
    snap = shadow.snapshot()
    assert "cal_probe_old" not in snap
    assert "cal_probe_new" in snap


def test_shadow_match_predicate_dm_and_lm() -> None:
    """A candidate matches a live probe iff DM is within
    ``dm_tol_frac`` AND (l, m) within ``lm_tol_rad``."""
    payload = {
        "/cnf/inject/active/cal_probe_dm500": _probe_payload(
            inj_id="cal_probe_dm500", dm=500.0, l_rad=0.01, m_rad=-0.01,
            fired_at=time.time(),
        ),
    }
    shadow = CalProbeShadow(store=_FakeStore(payload))
    shadow.maybe_refresh()
    # Within tolerances → matches.
    assert shadow.is_cal_probe_match(
        dm_pc_cc=510.0, l_rad=0.012, m_rad=-0.008, snr=15.0,
    ) == "cal_probe_dm500"
    # DM way off → no match.
    assert shadow.is_cal_probe_match(
        dm_pc_cc=1500.0, l_rad=0.01, m_rad=-0.01, snr=15.0,
    ) is None
    # l/m way off → no match.
    assert shadow.is_cal_probe_match(
        dm_pc_cc=505.0, l_rad=0.5, m_rad=0.5, snr=15.0,
    ) is None
    # Below SNR floor → no match.
    assert shadow.is_cal_probe_match(
        dm_pc_cc=505.0, l_rad=0.01, m_rad=-0.01, snr=2.0,
    ) is None
    assert shadow.n_matched >= 1
    assert shadow.n_no_match >= 3


def test_shadow_throttles_refresh() -> None:
    """``maybe_refresh`` no-ops within the throttle window."""
    payload = {
        "/cnf/inject/active/cal_probe_x": _probe_payload(
            inj_id="cal_probe_x", fired_at=time.time(),
        ),
    }
    store = _FakeStore(payload)
    shadow = CalProbeShadow(
        store=store, refresh_interval_s=10.0,
    )
    assert shadow.maybe_refresh() is True
    assert store.n_calls == 1
    # Second call within throttle window must be a no-op.
    assert shadow.maybe_refresh() is False
    assert store.n_calls == 1


def test_shadow_etcd_failure_does_not_raise() -> None:
    """Etcd errors are swallowed (probes simply stay un-exempted)."""

    class _BadStore:
        def get_dict_prefix(self, prefix):
            raise RuntimeError("etcd down")

    shadow = CalProbeShadow(store=_BadStore())
    shadow.maybe_refresh()  # Must not raise.
    assert shadow.n_refresh_fail == 1
    assert shadow.n_active == 0
    # Probing on an empty shadow is a hard "no match", not an error.
    assert shadow.is_cal_probe_match(
        dm_pc_cc=500.0, l_rad=0.0, m_rad=0.0, snr=20.0,
    ) is None


def test_shadow_validates_constructor_inputs() -> None:
    with pytest.raises(ValueError, match="refresh_interval_s"):
        CalProbeShadow(refresh_interval_s=0.0)
    with pytest.raises(ValueError, match="dm_tol_frac"):
        CalProbeShadow(dm_tol_frac=-1.0)
    with pytest.raises(ValueError, match="lm_tol_rad"):
        CalProbeShadow(lm_tol_rad=0.0)
    with pytest.raises(ValueError, match="grace_s"):
        CalProbeShadow(grace_s=-1.0)


def test_shadow_default_constants_are_sane() -> None:
    assert CAL_PROBE_INJ_ID_PREFIX == "cal_probe"
    assert DEFAULT_CAL_PROBE_REFRESH_S > 0.0
