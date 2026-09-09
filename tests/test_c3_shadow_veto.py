"""C3 shadow veto: what the veto WOULD say about an injection.

``cube_veto.decide`` short-circuits injections to an unconditional KEEP
("injection - exempt from veto"), so in production the veto thresholds
are never exercised: every hourly injection yields a tautological KEEP
that says nothing about whether R1-R11 still behave. C3 therefore
re-runs the SAME metrics with ``is_injection=False`` and records the
counterfactual alongside the real decision.

These tests drive the real :func:`dsart.coinc.cube_veto.decide` (not a
stub) so they double as a check that a clean injection still clears the
live thresholds, and that a degraded one is caught.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.coinc.cube_veto import CubeMetrics                       # noqa: E402
from dsart.services import c3 as c3mod                              # noqa: E402


class FakeStore:
    def __init__(self, responses: Optional[dict[str, Any]] = None) -> None:
        self._responses = dict(responses or {})
        self.puts: list[tuple[str, dict]] = []

    def put_dict(self, key: str, value) -> None:
        self.puts.append((key, dict(value)))

    def get_dict(self, key: str):
        return self._responses.get(key)


def _make_service(tmp_path: Path):
    cfg = c3mod.C3Config(
        archive_root=tmp_path / "candidates",
        rejected_root=tmp_path / "candidates_rejected",
        state_path=tmp_path / "state.json",
        fired_injection_log=None,
        corr_nodes={},
        flag_only=True,
    )
    return c3mod.C3Service(cfg, mon_store=FakeStore()), cfg


def _seed_event(cfg, name: str) -> Path:
    ev = cfg.archive_root / name
    (ev / "Level3").mkdir(parents=True, exist_ok=True)
    (ev / "Level3" / f"{name}.json").write_text("{}")
    (ev / "cubes").mkdir(parents=True, exist_ok=True)
    (ev / "cubes" / "x.npz").write_bytes(b"\x00" * 8)
    (ev / "Level2" / "plots").mkdir(parents=True, exist_ok=True)
    (ev / "Level2" / "plots" / "panel.png").write_bytes(b"PNG")
    (ev / "Level2" / f"C2_{name}.csv").write_text("a\n1\n")
    return ev


def _clean_metrics() -> CubeMetrics:
    """A morphologically clean burst.

    Values taken from a real recovered injection (260824gxtm) so the
    test tracks what the pipeline actually produces.
    """
    return CubeMetrics(
        ok=True, tz_trig=42.13, tz_apex=42.13, imgz_apex=17.45,
        img_off_apex=0.0, t_shift=0, dm_shift_trials=0, dm_edge=0,
    )


def _uncorroborated_metrics() -> CubeMetrics:
    """Same event but the cube does not confirm the trigger (R10)."""
    return CubeMetrics(
        ok=True, tz_trig=2.0, tz_apex=42.13, imgz_apex=17.45,
        img_off_apex=0.0, t_shift=0, dm_shift_trials=0, dm_edge=0,
    )


def _patch(monkeypatch, *, is_inj: bool, metrics: CubeMetrics) -> None:
    monkeypatch.setattr(c3mod, "event_is_injection", lambda *a, **k: is_inj)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: metrics)
    # NOTE: cube_veto.decide is deliberately NOT stubbed.


def test_clean_injection_records_shadow_keep(tmp_path, monkeypatch) -> None:
    svc, cfg = _make_service(tmp_path)
    _patch(monkeypatch, is_inj=True, metrics=_clean_metrics())
    _seed_event(cfg, "260101clean")
    rec = svc.process_event("260101clean")
    # Real decision: the exemption, not a judgement.
    assert rec["is_injection"] is True
    assert rec["action"] == "KEEP"
    assert rec["rules_fired"] == []
    assert "exempt" in rec["notes"]
    # Shadow: judged as a real burst, it still passes.
    assert rec["shadow_action"] == "KEEP"
    assert rec["shadow_keep"] is True
    assert rec["shadow_rules_fired"] == []
    assert svc._counters["shadow_keep"] == 1
    assert svc._counters["shadow_reject"] == 0


def test_uncorroborated_injection_records_shadow_reject(
    tmp_path, monkeypatch,
) -> None:
    svc, cfg = _make_service(tmp_path)
    _patch(monkeypatch, is_inj=True, metrics=_uncorroborated_metrics())
    ev = _seed_event(cfg, "260101dud")
    rec = svc.process_event("260101dud")
    # The action taken is unchanged -- still the exempt KEEP.
    assert rec["action"] == "KEEP"
    assert rec["rules_fired"] == []
    # But the shadow flags that a real burst here would be rejected.
    assert rec["shadow_keep"] is False
    assert rec["shadow_action"] == "REJECT"
    assert any("R10" in r for r in rec["shadow_rules_fired"])
    assert svc._counters["shadow_reject"] == 1
    assert svc._counters["shadow_keep"] == 0
    # Observational only: nothing destructive happened.
    assert (ev / "cubes" / "x.npz").exists()


def test_real_event_has_no_shadow_fields(tmp_path, monkeypatch) -> None:
    svc, cfg = _make_service(tmp_path)
    _patch(monkeypatch, is_inj=False, metrics=_clean_metrics())
    _seed_event(cfg, "260101real")
    rec = svc.process_event("260101real")
    assert rec["is_injection"] is False
    assert "shadow_action" not in rec
    assert "shadow_keep" not in rec
    assert "shadow_rules_fired" not in rec
    assert svc._counters["shadow_keep"] == 0
    assert svc._counters["shadow_reject"] == 0


def test_injection_with_unusable_metrics_skips_shadow(
    tmp_path, monkeypatch,
) -> None:
    # metrics.ok False -> the real decision fail-opens to KEEP and there is
    # nothing meaningful to shadow-evaluate.
    svc, cfg = _make_service(tmp_path)
    _patch(monkeypatch, is_inj=True,
           metrics=CubeMetrics(ok=False, reason="no cube npz files"))
    _seed_event(cfg, "260101nometrics")
    rec = svc.process_event("260101nometrics")
    assert rec["action"] == "KEEP"
    assert "shadow_action" not in rec
    assert svc._counters["shadow_keep"] == 0
    assert svc._counters["shadow_reject"] == 0
