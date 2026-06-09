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


# ---------------------------------------------------------------------------
# Real-DsaStore API path (2026-06-09 hotfix).
#
# Production ``DsaStore`` has NO ``get_dict_prefix`` method — only per-key
# ``get_dict``. Bulk prefix reads go through ``store.get_etcd().get_prefix(...)``
# which yields ``(value_bytes, meta)`` pairs. The 2026-06-09 search-compute
# log showed the shadow degrading silently (cal-probe bypass DISABLED) because
# the T3 ship missed this. These tests pin the production code path so a
# regression to the old name surfaces in CI.
# ---------------------------------------------------------------------------


class _FakeEtcdClient:
    """Mimics ``etcd3.Client.get_prefix`` return shape."""

    def __init__(self, items: Dict[str, Dict[str, Any]]):
        # items: key -> dict payload.
        import json as _json

        self._pairs = [
            (_json.dumps(v).encode("utf-8"), {"key": k.encode("utf-8")})
            for k, v in items.items()
        ]
        self.n_calls = 0

    def get_prefix(self, prefix: str):  # pragma: no cover - simple
        self.n_calls += 1
        return iter(self._pairs)


class _RealishStore:
    """Mimics the production ``DsaStore`` surface CalProbeShadow needs.

    Crucially does NOT expose ``get_dict_prefix`` (that's the test-fixture
    convenience the legacy ``_FakeStore`` exposes); only ``get_etcd``.
    """

    def __init__(self, items: Dict[str, Dict[str, Any]]):
        self._client = _FakeEtcdClient(items)

    def get_etcd(self):
        return self._client


def test_shadow_uses_get_etcd_get_prefix() -> None:
    """When the store has no ``get_dict_prefix`` we go via ``get_etcd``."""
    items = {
        "/cnf/inject/active/cal_probe_dm500_w32_t1": _probe_payload(
            inj_id="cal_probe_dm500_w32_t1", dm=500.0, fired_at=time.time(),
        ),
        "/cnf/inject/active/manual_inject_1": _probe_payload(
            inj_id="manual_inject_1", fired_at=time.time(),
        ),
    }
    store = _RealishStore(items)
    shadow = CalProbeShadow(store=store)
    assert shadow.maybe_refresh() is True
    assert shadow.n_refresh_ok == 1
    assert shadow.n_refresh_fail == 0
    assert store._client.n_calls == 1
    snap = shadow.snapshot()
    assert list(snap.keys()) == ["cal_probe_dm500_w32_t1"]


def test_shadow_etcd_prefix_value_decoding() -> None:
    """JSON bytes, str, and dict-mapping values are all accepted; bad
    rows are silently skipped."""

    class _MixedClient:
        def __init__(self) -> None:
            import json as _json

            good_bytes = _json.dumps(
                _probe_payload(
                    inj_id="cal_probe_bytes", dm=100.0,
                    fired_at=time.time(),
                ),
            ).encode("utf-8")
            good_str = _json.dumps(
                _probe_payload(
                    inj_id="cal_probe_str", dm=200.0, fired_at=time.time(),
                ),
            )
            good_dict = _probe_payload(
                inj_id="cal_probe_dict", dm=300.0, fired_at=time.time(),
            )
            self._pairs = [
                (good_bytes, {"k": b"k1"}),
                (good_str, {"k": b"k2"}),
                (good_dict, {"k": b"k3"}),
                (b"not json", {"k": b"k4"}),       # silently dropped
                (12345, {"k": b"k5"}),             # silently dropped
            ]

        def get_prefix(self, prefix: str):
            return iter(self._pairs)

    class _Wrap:
        def __init__(self, client):
            self._client = client

        def get_etcd(self):
            return self._client

    shadow = CalProbeShadow(store=_Wrap(_MixedClient()))
    shadow.maybe_refresh()
    assert shadow.n_refresh_ok == 1
    snap = shadow.snapshot()
    assert set(snap.keys()) == {
        "cal_probe_bytes", "cal_probe_str", "cal_probe_dict",
    }


def test_shadow_handles_get_etcd_failure() -> None:
    """A failing ``get_etcd().get_prefix`` must NOT raise, and must NOT
    keep retrying the noisy warning. The shadow stays empty."""

    class _ExplodingClient:
        def get_prefix(self, prefix):
            raise RuntimeError("etcd unavailable")

    class _Wrap:
        def __init__(self):
            self._n = 0

        def get_etcd(self):
            self._n += 1
            return _ExplodingClient()

    store = _Wrap()
    shadow = CalProbeShadow(store=store, refresh_interval_s=0.001)
    shadow.maybe_refresh()
    time.sleep(0.005)
    shadow.maybe_refresh()
    assert shadow.n_refresh_fail >= 1
    assert shadow.n_active == 0
