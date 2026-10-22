"""x2 D10: injection at the 2x op-point (bot, matcher, dashboard K tags),
plus the WS-O 2x config files parsed by the WS-D loaders (D6)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DASH = REPO / "tools" / "dashboard" / "dsa_monitor"

from dsart.coinc.criteria import CriteriaEvaluator  # noqa: E402
from dsart.coinc.inject_match import InjectionMatcher  # noqa: E402
from dsart.coinc.stats import compute_stats  # noqa: E402
from dsart.coinc.window import WindowEntry  # noqa: E402
from dsart.inject import bot as B  # noqa: E402
from dsart.services.coincidencer import CoincidencerConfig  # noqa: E402
from dsart.x2.opoint_runtime import ActiveOpPoint  # noqa: E402

SEL2 = "2x-gridded-ns2-cint4"
CRIT_2X = REPO / "configs" / "opoints" / "2x" / "c2_trigger_criteria.yaml"
SEARCH_2X = REPO / "configs" / "opoints" / "2x" / "dsart_search_rt.yaml"
SEARCH_1X = REPO / "configs" / "dsart_search_rt.yaml"


def _stats(width: int, period: float, snr: float = 20.0, dm: float = 900.0):
    e = WindowEntry(mjd=61335.0, snr=snr, l_rad=0.0, m_rad=0.0, l_pix=0,
                    m_pix=0, dm_pc_cc=dm, dm_idx_global=1, fine_dm_idx=1,
                    event_specnum=10, width_samples=width,
                    kernel_id="unit:d1:b4", flags=0, search_node_id=1,
                    gpu_half=0, cube_id=1, sample_period_us=period)
    return compute_stats([e], gal_dm_max_los=100.0)


# ----------------------------------------------------------- D6 files


def test_ws_o_2x_criteria_file_loads_and_gates_in_ms() -> None:
    import yaml
    doc = yaml.safe_load(CRIT_2X.read_text())
    assert "x2_todo" not in doc                     # D6 landed
    ev = CriteriaEvaluator(CRIT_2X, opoint=SEL2)
    names = [c.name for c in ev.classes]
    assert names[0] == "bright_frb_extragalactic"
    req = ev.classes[0].require
    assert req["width_median_max_ms"] == 16.8
    assert req["dm_median_max_pc_cc"] == 3100.0
    assert ev.evaluate(_stats(32, 524.288)).name == "bright_frb_extragalactic"
    ev2 = CriteriaEvaluator(CRIT_2X, opoint=SEL2)
    assert ev2.evaluate(_stats(33, 524.288)).name != "bright_frb_extragalactic"
    ev3 = CriteriaEvaluator(CRIT_2X, opoint=SEL2)
    assert ev3.evaluate(_stats(4, 524.288, dm=3050.0)).name == \
        "bright_frb_extragalactic"


def test_ws_o_2x_search_config_parses_with_2x_defaults() -> None:
    cfg = CoincidencerConfig.from_yaml(SEARCH_2X, override={"opoint": SEL2})
    assert cfg.cutout.enabled and cfg.cutout.wait_s == 120
    assert cfg.cutout.status_port == 11501
    assert cfg.archive_schema == 2
    assert cfg.partial_event_policy == "archive"
    assert cfg.fallback_sample_period_us == 524.288
    assert cfg.inject_specnum_tol_s == pytest.approx(4.294967296)
    one = CoincidencerConfig.from_yaml(SEARCH_1X, override={"opoint": "1x"})
    assert not one.cutout.enabled and one.archive_schema == 1
    assert one.partial_event_policy == "discard"
    assert one.fallback_sample_period_us == 1048.576
    assert one.inject_specnum_tol_s is None


# ------------------------------------------------------------- matcher


class _Store:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})

    def get_dict(self, key):
        return self.docs.get(key)

    def put_dict(self, key, value):
        self.docs[key] = dict(value)


def _matcher(**kw):
    return InjectionMatcher(store=_Store(), **kw)


def test_matcher_seconds_gate() -> None:
    from dsart.coinc.inject_match import ActiveInjection
    now = time.time()
    inj = ActiveInjection.from_dict({
        "inj_id": "inj_x", "dm_pc_cm3": 2500.0, "l_rad": 0.0, "m_rad": 0.0,
        "width_samples": 4, "fluence_jy_ms": 1e-3,
        "apply_at_specnum": 1_000_000, "fired_at_unix": now,
        "expires_at_unix": now + 60.0,
    })
    # 2x row 3.0 s late: 3.0 s / 524.288 us = 5722 T_ext samples
    exp2 = 1_000_000 // 8
    late = exp2 + 5722
    m1 = _matcher()
    assert not m1._row_matches_specnum(inj=inj, event_specnum=late,
                                       sample_period_us=524.288)
    m2 = _matcher(specnum_tol_s=4.294967296)
    assert m2._row_matches_specnum(inj=inj, event_specnum=late,
                                   sample_period_us=524.288)
    # 1x row in 1x units with the seconds gate: same window as today
    late1 = 1_000_000 // 16 + 4000
    assert m1._row_matches_specnum(inj=inj, event_specnum=late1,
                                   sample_period_us=1048.576)
    assert m2._row_matches_specnum(inj=inj, event_specnum=late1,
                                   sample_period_us=1048.576)
    assert not m2._row_matches_specnum(inj=inj, event_specnum=late1 + 200,
                                       sample_period_us=1048.576)


# ----------------------------------------------------------------- bot


def test_bot_from_yaml_per_opoint(tmp_path: Path) -> None:
    one = B.InjectBotConfig.from_yaml(SEARCH_1X,
                                      active=ActiveOpPoint("1x", "cli"))
    plain = B.InjectBotConfig.from_yaml(SEARCH_1X)
    assert one.width_choices == plain.width_choices
    assert one.native_per_search_sample == 32 == plain.native_per_search_sample
    assert one.max_c1c2_width_search_samples == 16
    assert one.k_family == "1x" and one.state_path == plain.state_path
    two = B.InjectBotConfig.from_yaml(SEARCH_2X, active=ActiveOpPoint(SEL2, "cli"))
    assert two.native_per_search_sample == 16
    assert two.max_c1c2_width_search_samples == 32
    assert two.k_family == "2x"
    assert two.state_path != one.state_path            # per-family state
    assert max(two.width_choices) <= 512                # 32 x 16 native


def test_bot_ignores_other_family_k(tmp_path: Path) -> None:
    b = B.bucket_key(500.0, 4)
    entries = {"ok": True, "entries": [
        {"bucket": b, "K": 2890.0, "k_family": "1x"}]}
    cfg = B.InjectBotConfig.from_dict(
        dict(enabled=True, state_path=str(tmp_path / "s.json"),
             results_path=str(tmp_path / "r.jsonl")), k_family="2x")
    bot = B.InjectBot(cfg, store=_Store(), notifier=object(),
                      http_get=lambda url, **kw: (200, entries),
                      http_post_form=lambda *a, **k: (500, {}))
    assert bot._get_calibration_entry(500.0, 4) is None
    cfg1 = B.InjectBotConfig.from_dict(
        dict(enabled=True, state_path=str(tmp_path / "s1.json"),
             results_path=str(tmp_path / "r.jsonl")), k_family="1x")
    bot1 = B.InjectBot(cfg1, store=_Store(), notifier=object(),
                       http_get=lambda url, **kw: (200, entries),
                       http_post_form=lambda *a, **k: (500, {}))
    assert bot1._get_calibration_entry(500.0, 4)["K"] == 2890.0


def test_owner_metering_gate(tmp_path: Path, monkeypatch) -> None:
    now = time.time()
    docs = {f"/mon/corr_rt/{cg}/corr_fast": {"ts_wall_unix": now}
            for cg in range(16)}
    for sid, g in B.SEARCH_HALVES:
        docs[f"/mon/search_rt/{sid}/compute/{g}"] = {
            "ts_wall_unix": now, "c1_metering_active": 1 if (sid, g) == (9, 1) else 0}
    for gate, ok in (("any", False), ("owner", True)):
        cfg = B.InjectBotConfig.from_dict(dict(
            enabled=True, state_path=str(tmp_path / f"{gate}.json"),
            results_path=str(tmp_path / "r.jsonl"), metering_gate=gate,
            plan_path="unused"))
        bot = B.InjectBot(cfg, store=_Store(docs), notifier=object(),
                          http_get=lambda url, **kw: (200, {}),
                          http_post_form=lambda *a, **k: (500, {}),
                          time_fn=lambda: now)
        # the shot's DM is owned by s1g0 (not the metering s9g1)
        monkeypatch.setattr(bot, "_owner_halves", lambda dm: {(1, 0)})
        res = bot.check_health(500.0)
        assert res[0] is ok, (gate, res)


# ------------------------------------------------------- dashboard K tag


def test_dashboard_k_entries_tagged_with_active_family(monkeypatch) -> None:
    sys.path.insert(0, str(DASH))
    try:
        import inject_calibration as IC
    finally:
        sys.path.remove(str(DASH))
    e = IC.CalibrationEntry(bucket="dm0500", dm_pc_cm3_rounded=500,
                            width_samples=4, K=100.0, last_fluence_jy_ms=1e-3,
                            last_observed_snr=20.0, last_inj_id="x",
                            last_calibrated_at_unix=1.0, actor="t")
    assert "k_family" not in e.to_dict()                # today's payload
    s0 = _Store()
    IC.CalibrationStore(s0).put(e)
    assert "k_family" not in s0.docs["/cnf/inject/snr_calibration/dm0500"]
    s2 = _Store({"/cnf/opoint": {"opoint": SEL2, "selection": SEL2}})
    IC.CalibrationStore(s2).put(e)
    assert s2.docs["/cnf/inject/snr_calibration/dm0500"]["k_family"] == "2x"
    back = IC.CalibrationEntry.from_dict(s2.docs["/cnf/inject/snr_calibration/dm0500"])
    assert back.k_family == "2x"
