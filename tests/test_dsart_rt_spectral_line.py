"""Unit tests for the spectral-line (SPL) plumbing in dsart_rt.

Pins three contracts:

1. ``RoutineSpec.spl_gate`` round-trips through ``PipelineConfig.from_dict``
   (None / "on" / "off", case-normalised).
2. ``RtOrchestrator._select_routines`` keeps exactly the right
   second-bada-reader routine for the node's SPL state: the SPL
   fringe-stopper (``spl_gate: on``) when enabled, the drain
   (``spl_gate: off``) when disabled, and always the ungated routines.
3. ``RtOrchestrator._substitute`` expands ``SPL_INTEGRATION_S`` /
   ``SPL_NFREQ_INT`` from the loaded per-node SPL state, and
   ``_load_spl_cfg`` resolves THIS node's chgroup entry (fail-safe to
   disabled on a miss).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dsart.services.dsart_rt import (  # noqa: E402
    PipelineConfig,
    RtOrchestrator,
    _SPL_DEFAULT_INTEGRATION_S,
    _SPL_DEFAULT_NFREQ_INT,
)


def _routine(name: str, *, cmd: str = "echo", args: str = "", spl_gate=None) -> dict:
    d = {"name": name, "cmd": cmd, "args": args or name}
    if spl_gate is not None:
        d["spl_gate"] = spl_gate
    return d


def _orch_with_spl(*, cn_id: int = 3, spl: dict | None = None) -> RtOrchestrator:
    """Bare orchestrator (no etcd/sockets) with just the bits the
    selection / substitution paths touch."""
    orch = RtOrchestrator.__new__(RtOrchestrator)
    orch.cn_id = cn_id
    orch._spl = spl or {
        "enabled": False,
        "integration_s": _SPL_DEFAULT_INTEGRATION_S,
        "nfreq_int": _SPL_DEFAULT_NFREQ_INT,
    }
    return orch


# ---------------------------------------------------------------------------
# 1. spl_gate round-trip
# ---------------------------------------------------------------------------


def test_spl_gate_round_trips_and_normalises() -> None:
    cfg = PipelineConfig.from_dict(
        {
            "schema_version": 1,
            "routines": [
                _routine("prod"),                       # no gate
                _routine("spl", spl_gate="ON"),         # upper → normalise
                _routine("drain", spl_gate="off"),
            ],
        }
    )
    by_name = {r.name: r for r in cfg.routines}
    assert by_name["prod"].spl_gate is None
    assert by_name["spl"].spl_gate == "on"
    assert by_name["drain"].spl_gate == "off"


# ---------------------------------------------------------------------------
# 2. routine selection
# ---------------------------------------------------------------------------


def _select_names(spl_enabled: bool) -> list[str]:
    cfg = PipelineConfig.from_dict(
        {
            "schema_version": 1,
            "routines": [
                _routine("corr_slow"),
                _routine("meridian_fringestop"),
                _routine("meridian_fringestop_spl", spl_gate="on"),
                _routine("bada_null_drain", spl_gate="off"),
            ],
        }
    )
    orch = _orch_with_spl(spl={
        "enabled": spl_enabled,
        "integration_s": _SPL_DEFAULT_INTEGRATION_S,
        "nfreq_int": _SPL_DEFAULT_NFREQ_INT,
    })
    orch._config = cfg
    return [r.name for r in orch._select_routines(cfg.routines)]


def test_select_routines_spl_off_keeps_drain_not_spl() -> None:
    names = _select_names(spl_enabled=False)
    assert "bada_null_drain" in names
    assert "meridian_fringestop_spl" not in names
    # Ungated routines always present.
    assert "corr_slow" in names
    assert "meridian_fringestop" in names


def test_select_routines_spl_on_keeps_spl_not_drain() -> None:
    names = _select_names(spl_enabled=True)
    assert "meridian_fringestop_spl" in names
    assert "bada_null_drain" not in names
    assert "corr_slow" in names
    assert "meridian_fringestop" in names


# ---------------------------------------------------------------------------
# 3. token substitution + _load_spl_cfg
# ---------------------------------------------------------------------------


def test_substitute_spl_tokens() -> None:
    orch = _orch_with_spl(spl={
        "enabled": True, "integration_s": 12.884901888, "nfreq_int": 4,
    })
    out = orch._substitute("--integration-s SPL_INTEGRATION_S", val=None)
    assert "SPL_INTEGRATION_S" not in out
    # repr(float(...)) so the wrapper parses it back exactly.
    assert "12.884901888" in out
    out2 = orch._substitute("--nfreq-int-spl SPL_NFREQ_INT", val=None)
    assert out2 == "--nfreq-int-spl 4"


def test_substitute_spl_does_not_disturb_cn_chgroup() -> None:
    # cn 5 → chgroup index 2 in the canonical corr-node order.
    orch = _orch_with_spl(cn_id=5, spl={
        "enabled": True, "integration_s": 1.0, "nfreq_int": 8,
    })
    assert orch._substitute("CHGROUP", val=None) == "2"
    assert orch._substitute("CN", val=None) == "5"


class _FakeStore:
    def __init__(self, doc):
        self._doc = doc

    def get_dict(self, key):
        assert key == "/cnf/spectral_line"
        return self._doc


def test_load_spl_cfg_resolves_this_nodes_chgroup() -> None:
    # cn 4 → chgroup 1.
    orch = _orch_with_spl(cn_id=4)
    orch._store = _FakeStore({
        "version": 1,
        "subbands": {
            "1": {"enabled": True, "integration_s": 5.0, "nfreq_int": 2},
            "2": {"enabled": True, "integration_s": 9.0, "nfreq_int": 8},
        },
    })
    orch._load_spl_cfg()
    assert orch._spl["enabled"] is True
    assert orch._spl["integration_s"] == 5.0
    assert orch._spl["nfreq_int"] == 2


def test_load_spl_cfg_missing_entry_is_disabled() -> None:
    orch = _orch_with_spl(cn_id=22)  # chgroup 15, not in the doc
    orch._store = _FakeStore({"version": 1, "subbands": {"0": {"enabled": True}}})
    orch._load_spl_cfg()
    assert orch._spl["enabled"] is False


def test_load_spl_cfg_missing_key_is_disabled() -> None:
    orch = _orch_with_spl(cn_id=3)
    orch._store = _FakeStore(None)
    orch._load_spl_cfg()
    assert orch._spl["enabled"] is False
