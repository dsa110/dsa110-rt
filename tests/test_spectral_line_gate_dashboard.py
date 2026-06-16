"""Unit tests for the dsa_monitor Control tab's spectral-line module.

Exercises ``spectral_line_gate``:

* :func:`get_spectral_line_state` — missing key (all 16 sub-bands
  defaulted, ``default=True``), present partial key (missing chgroups
  filled), ``nint`` hint derivation.
* :func:`set_spectral_line_state` — validation (range / divisor /
  reason), full-doc persistence + audit row, all-or-nothing on a bad
  row.
* :func:`validate_subband` / :func:`nint_from_integration_s` /
  :func:`divisors_of_nchan` helpers.

Constants are pinned against the orchestrator's authoritative copies in
:mod:`dsart.services.dsart_rt` so the two cannot drift.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

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

import spectral_line_gate as slg                                    # noqa: E402


class FakeDsaStore:
    def __init__(self, get_dict_responses: dict[str, Any] | None = None):
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.put_raises: dict[str, Exception] = {}
        self._responses = dict(get_dict_responses or {})

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        if key in self.put_raises:
            raise self.put_raises[key]
        self.puts.append((key, json.loads(json.dumps(payload))))

    def get_dict(self, key: str) -> Any:
        return self._responses.get(key)


# ---------------------------------------------------------------------------
# Constants pinned against the orchestrator copy
# ---------------------------------------------------------------------------


def test_key_pinned_to_orchestrator_copy() -> None:
    from dsart.services.dsart_rt import SPECTRAL_LINE_KEY as ORCH_KEY
    assert slg.SPECTRAL_LINE_KEY == ORCH_KEY


def test_tsamp_and_default_nint_pinned() -> None:
    from dsart.services import dsart_rt as orch
    assert slg.TSAMP_S == orch._SPL_TSAMP_S
    assert slg.DEFAULT_NINT == orch._SPL_DEFAULT_NINT
    assert slg.DEFAULT_NFREQ_INT == orch._SPL_DEFAULT_NFREQ_INT
    # The default integration time → exactly the production nint.
    assert slg.nint_from_integration_s(slg.DEFAULT_INTEGRATION_S) == slg.DEFAULT_NINT


def test_audit_prefix_under_control_namespace() -> None:
    assert slg.SPECTRAL_LINE_AUDIT_PREFIX.startswith("/mon/audit/control/")
    assert slg.SPECTRAL_LINE_AUDIT_PREFIX.endswith("/")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_divisors_of_nchan() -> None:
    ds = slg.divisors_of_nchan()
    assert ds[0] == 1
    assert slg.NCHAN_SPW in ds
    assert all(slg.NCHAN_SPW % d == 0 for d in ds)
    assert 8 in ds and 5 not in ds


def test_nint_from_integration_s_floor_one() -> None:
    assert slg.nint_from_integration_s(0.0) == 1
    assert slg.nint_from_integration_s(slg.TSAMP_S) == 1
    assert slg.nint_from_integration_s(slg.TSAMP_S * 10) == 10


class TestValidateSubband:
    def test_ok_enabled(self) -> None:
        out = slg.validate_subband(3, {
            "enabled": True, "integration_s": 5.0, "nfreq_int": 8,
        })
        assert out == {"enabled": True, "integration_s": 5.0, "nfreq_int": 8}

    def test_chgroup_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            slg.validate_subband(16, {"enabled": True})

    def test_bad_nfreq_when_enabled(self) -> None:
        with pytest.raises(ValueError):
            slg.validate_subband(0, {
                "enabled": True, "integration_s": 5.0, "nfreq_int": 5,
            })

    def test_integration_out_of_range_when_enabled(self) -> None:
        with pytest.raises(ValueError):
            slg.validate_subband(0, {
                "enabled": True, "integration_s": 999.0, "nfreq_int": 8,
            })

    def test_disabled_row_not_strictly_validated(self) -> None:
        # A disabled row with an otherwise-illegal nfreq_int still
        # normalises (so a later enable starts sane) rather than raising.
        out = slg.validate_subband(0, {
            "enabled": False, "integration_s": 999.0, "nfreq_int": 5,
        })
        assert out["enabled"] is False


# ---------------------------------------------------------------------------
# get_spectral_line_state
# ---------------------------------------------------------------------------


def test_get_missing_key_all_defaulted() -> None:
    store = FakeDsaStore()
    state = slg.get_spectral_line_state(store)
    assert state["default"] is True
    assert state["n_enabled"] == 0
    assert len(state["subbands"]) == slg.N_CHGROUPS
    for cg in range(slg.N_CHGROUPS):
        sb = state["subbands"][str(cg)]
        assert sb["enabled"] is False
        assert sb["nint"] == slg.DEFAULT_NINT


def test_get_partial_key_fills_missing() -> None:
    store = FakeDsaStore(get_dict_responses={
        slg.SPECTRAL_LINE_KEY: {
            "version": 1,
            "subbands": {
                "6": {"enabled": True, "integration_s": 12.884901888, "nfreq_int": 4},
            },
        },
    })
    state = slg.get_spectral_line_state(store)
    assert state["default"] is False
    assert state["n_enabled"] == 1
    assert state["subbands"]["6"]["enabled"] is True
    assert state["subbands"]["6"]["nfreq_int"] == 4
    # Every other chgroup defaulted to disabled.
    assert state["subbands"]["0"]["enabled"] is False


# ---------------------------------------------------------------------------
# set_spectral_line_state
# ---------------------------------------------------------------------------


def test_set_happy_path_writes_full_doc_and_audit() -> None:
    store = FakeDsaStore()
    out = slg.set_spectral_line_state(
        store,
        subbands={
            "6": {"enabled": True, "integration_s": 12.884901888, "nfreq_int": 4},
            "7": {"enabled": True, "integration_s": 12.884901888, "nfreq_int": 1},
        },
        reason="HI line test",
        actor="vikram",
        now_unix=1_700_000_000.0,
    )
    assert out["n_enabled"] == 2
    assert out["enabled_chgroups"] == [6, 7]
    # Two etcd writes: the config doc + the audit row.
    keys = [k for k, _ in store.puts]
    assert slg.SPECTRAL_LINE_KEY in keys
    assert any(k.startswith(slg.SPECTRAL_LINE_AUDIT_PREFIX) for k in keys)
    # Persisted doc is COMPLETE (all 16 chgroups), missing ones disabled.
    doc = dict(store.puts[0][1])
    assert doc["key" if False else "version"] == slg.SCHEMA_VERSION
    assert len(doc["subbands"]) == slg.N_CHGROUPS
    assert doc["subbands"]["0"]["enabled"] is False
    assert doc["subbands"]["6"]["nfreq_int"] == 4


def test_set_rejects_empty_reason_before_write() -> None:
    store = FakeDsaStore()
    with pytest.raises(ValueError):
        slg.set_spectral_line_state(
            store, subbands={"0": {"enabled": False}}, reason="  ",
        )
    assert store.puts == []


def test_set_bad_row_is_all_or_nothing() -> None:
    store = FakeDsaStore()
    with pytest.raises(ValueError):
        slg.set_spectral_line_state(
            store,
            subbands={"3": {"enabled": True, "integration_s": 5.0, "nfreq_int": 7}},
            reason="bad nfreq",
        )
    # Nothing written (validation happens before any put).
    assert store.puts == []
