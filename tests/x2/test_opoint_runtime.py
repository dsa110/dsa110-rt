"""dsart.x2.opoint_runtime + config_diff + the repo's op-point config sets (SPEC §10)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dsart.x2 import config_diff as cdiff
from dsart.x2 import opoint_runtime as opr
from dsart.x2.opoint import FORMATS_2X, OPPOINT_1X, opoint_from_id
from dsart.x2.plan import build_tiered_plan_for_opoint

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
NS2 = "2x-gridded-ns2-cint4"
NS4 = "2x-baselines-ns4-cint5"
NODE_PLAN = "/home/ubuntu/data/dm_plans/dm_plan_2x.npz"


@pytest.fixture(scope="module")
def plans():
    return {sel: build_tiered_plan_for_opoint(opoint_from_id(sel)) for sel in (NS2, NS4)}


class FakeStore:
    def __init__(self, kv=None, fail=0):
        self.kv = dict(kv or {})
        self.fail = fail
        self.gets = []

    def get_dict(self, key):
        self.gets.append(key)
        if self.fail:
            self.fail -= 1
            raise ConnectionError("etcd down")
        return self.kv.get(key)


# ---------------------------------------------------------------- selections


def test_selections():
    assert opr.UI_SELECTIONS[:2] == ("1x", "1x-legacy")
    assert set(opr.UI_SELECTIONS[2:]) == set(FORMATS_2X) and len(FORMATS_2X) == 6
    for sel in opr.UI_SELECTIONS:
        assert opr.validate_selection(sel) == sel
    assert opr.config_set_of("1x-legacy") == "1x-legacy"
    assert opr.config_set_of(NS2) == "2x"
    assert opr.op_of("1x-legacy") is OPPOINT_1X
    assert opr.op_of(NS4).n_sub == 4 and opr.op_of(NS4).value_kind == "baselines"
    assert opr.k_family_of("1x-legacy") == "1x" and opr.k_family_of(NS4) == "2x"
    for bad in ("2x", "4x", "2x-gridded-ns3-cint4", "2x-foo-ns2-cint4", ""):
        with pytest.raises(opr.OpPointError):
            opr.validate_selection(bad)


def test_keys():
    assert opr.set_key(NS2, "search_rt") == f"/cnf/opoints/{NS2}/search_rt"
    assert opr.meta_key("1x-legacy") == "/cnf/opoints/1x-legacy/meta"
    with pytest.raises(opr.OpPointError):
        opr.set_key("1x", "pipeline")
    assert opr.is_live_key("/cnf/pipeline_rt") and opr.is_live_key("/cnf/opoint")
    assert not opr.is_live_key(f"/cnf/opoints/{NS2}/pipeline_rt")
    assert opr.k_park_prefix("2x") == "/cnf/opoints/_kpark/2x/"
    with pytest.raises(ValueError):
        opr.k_park_prefix("1x-legacy")


def test_active_record_round_trip():
    rec = opr.make_active_record(NS2, plan_id="0xabc", plan_path="/p.npz", user="u")
    assert rec["opoint"] == NS2 and rec["selection"] == NS2 and rec["config_set"] == "2x"
    assert opr.parse_active_record(rec) == NS2
    rec = opr.make_active_record("1x-legacy")
    assert rec["opoint"] == "1x" and opr.parse_active_record(rec) == "1x-legacy"


@pytest.mark.parametrize("raw,want", [
    (None, None), ({"user": "x"}, None), ("1x-legacy", "1x-legacy"), (NS2, NS2),
    ({"opoint": NS4}, NS4), ({"opoint": "1x"}, "1x"),
    ({"opoint": "1x", "config_set": "1x-legacy"}, "1x-legacy"),
])
def test_parse_active_record(raw, want):
    assert opr.parse_active_record(raw) == want


@pytest.mark.parametrize("raw", [{"opoint": "3x"}, 7, ["2x"], {"opoint": "1x", "selection": NS2},
                                 {"selection": "2x"}])
def test_parse_active_record_invalid(raw):
    with pytest.raises(opr.OpPointError):
        opr.parse_active_record(raw)


def test_stamps():
    assert opr.stamp_matches({}, "1x") and opr.stamp_matches({}, "1x-legacy")
    assert not opr.stamp_matches({}, NS2)
    assert opr.stamp_matches({"opoint_selection": NS2}, NS2)
    assert not opr.stamp_matches({"opoint_selection": "1x"}, "1x-legacy")
    assert opr.rendered_selection({"opoint": "2x"}) is None


# ---------------------------------------------------------------- resolution


def test_resolve_cli_wins(tmp_path):
    s = FakeStore({"/cnf/opoint": {"opoint": NS2}})
    a = opr.resolve_active_opoint(s, cli="1x-legacy", cache_path=tmp_path / "c.json")
    assert (a.selection, a.source, a.opoint_id) == ("1x-legacy", "cli", "1x")
    assert s.gets == []
    with pytest.raises(opr.OpPointError):
        opr.resolve_active_opoint(s, cli="9x")


def test_resolve_from_etcd_and_unset(tmp_path):
    c = tmp_path / "c.json"
    rec = opr.make_active_record(NS2, plan_id="0x1", plan_path="/p")
    a = opr.resolve_active_opoint(FakeStore({"/cnf/opoint": rec}), cache_path=c)
    assert (a.selection, a.source, a.config_set, a.plan_id, a.plan_path) == \
        (NS2, "etcd", "2x", "0x1", "/p")
    assert a.to_dict()["opoint"] == NS2
    assert c.is_file()
    a = opr.resolve_active_opoint(FakeStore({}), cli="auto", cache_path=c)
    assert (a.selection, a.source) == ("1x", "etcd-unset")


def test_resolve_etcd_down_uses_cache_then_default(tmp_path):
    c = tmp_path / "c.json"
    opr.resolve_active_opoint(FakeStore({"/cnf/opoint": opr.make_active_record(NS4, plan_id="0x9")}),
                              cache_path=c)
    sleeps = []
    a = opr.resolve_active_opoint(FakeStore(fail=5), cache_path=c, retries=3,
                                  retry_sleep_s=0.25, sleep_fn=sleeps.append)
    assert (a.selection, a.source, a.plan_id) == (NS4, "cache", "0x9")
    assert "unreachable" in a.warning
    assert sleeps == [0.25, 0.25]
    a = opr.resolve_active_opoint(FakeStore(fail=5), cache_path=tmp_path / "none.json",
                                  retries=2, sleep_fn=lambda s: None)
    assert (a.selection, a.source) == ("1x", "default")
    assert a.warning


def test_resolve_retry_then_success(tmp_path):
    s = FakeStore({"/cnf/opoint": NS2}, fail=1)
    a = opr.resolve_active_opoint(s, cache_path=tmp_path / "c.json", sleep_fn=lambda s: None)
    assert (a.selection, a.source) == (NS2, "etcd")


def test_resolve_invalid_value_raises(tmp_path):
    with pytest.raises(opr.OpPointError):
        opr.resolve_active_opoint(FakeStore({"/cnf/opoint": {"opoint": "x"}}),
                                  cache_path=tmp_path / "c.json")


# ----------------------------------------------------------------- file paths


def _tree(tmp_path, with_2x=True):
    base = tmp_path / "configs"
    (base / "opoints" / "2x").mkdir(parents=True)
    for f in opr.CONFIG_FILES.values():
        (base / f).write_text("routines: []\n")
        if with_2x:
            (base / "opoints" / "2x" / f).write_text("opoint: 2x\nroutines: []\n")
    return base


def test_config_path_fallback_rules(tmp_path):
    base = _tree(tmp_path)
    assert opr.config_path("search_rt", "1x", base) == base / "dsart_search_rt.yaml"
    assert opr.config_path("c2_criteria", "1x-legacy", base) == base / "c2_trigger_criteria.yaml"
    for sel in (NS2, NS4, "2x"):
        assert opr.config_path("pipeline_rt", sel, base) == base / "opoints/2x/dsart_pipeline_rt.yaml"
    (base / "opoints" / "1x").mkdir()
    (base / "opoints" / "1x" / "dsart_search_rt.yaml").write_text("routines: []\n")
    assert opr.config_path("search_rt", "1x", base) == base / "opoints/1x/dsart_search_rt.yaml"
    with pytest.raises(opr.OpPointError):
        opr.config_path("bogus", "1x", base)


def test_config_path_2x_never_falls_back(tmp_path):
    base = _tree(tmp_path, with_2x=False)
    with pytest.raises(FileNotFoundError):
        opr.config_path("search_rt", NS2, base)


def test_map_config_path(tmp_path):
    base = _tree(tmp_path)
    unit_path = base / "dsart_search_rt.yaml"
    assert opr.map_config_path(unit_path, NS2) == base / "opoints/2x/dsart_search_rt.yaml"
    assert opr.map_config_path(unit_path, "1x") == unit_path
    assert opr.map_config_path(base / "opoints/2x/dsart_search_rt.yaml", "1x") == unit_path
    other = base / "chgroup_assignments.yaml"
    assert opr.map_config_path(other, NS2) == other


def test_select_service_config(tmp_path):
    base = _tree(tmp_path)
    s = FakeStore({"/cnf/opoint": opr.make_active_record(NS2, plan_id="0x5")})
    active, path = opr.select_service_config(base / "dsart_search_rt.yaml", store=s,
                                             cache_path=tmp_path / "c.json")
    assert active.selection == NS2 and active.plan_id == "0x5"
    assert path == base / "opoints/2x/dsart_search_rt.yaml"
    active, path = opr.select_service_config(base / "dsart_search_rt.yaml", cli_opoint="1x",
                                             store=s)
    assert active.source == "cli" and path == base / "dsart_search_rt.yaml"


def test_add_opoint_argument():
    p = argparse.ArgumentParser()
    opr.add_opoint_argument(p)
    assert p.parse_args([]).opoint == "auto"
    assert p.parse_args(["--opoint", NS2]).opoint == NS2
    assert p.parse_args(["--opoint", "1x-legacy"]).opoint == "1x-legacy"
    with pytest.raises(SystemExit):
        p.parse_args(["--opoint", "2x"])


# ------------------------------------------------------------------ rendering


def test_render_context_from_opoint_and_plan(plans):
    ctx = opr.render_context(NS2, plan=plans[NS2], plan_path=NODE_PLAN)
    assert (ctx["opoint_id"], ctx["n_sub"], ctx["n_streams"]) == (NS2, "2", "32")
    assert (ctx["t_ext_us"], ctx["chan_sum_factor"], ctx["t_int_fast_native"]) == ("524.288", "2", "8")
    assert ctx["plan_id"] == f"0x{plans[NS2].plan_id:016x}" and ctx["plan_path"] == NODE_PLAN
    # ownership d % 8: n_sub 2 -> 4 A + 2 B per half (half 7: 4 A + 1 B)
    assert ctx["owned_coarse_csv_h0"].split(",")[:3] == ["0", "8", "16"]
    assert (ctx["n_owned_A_h0"], ctx["n_owned_B_h0"]) == ("4", "2")
    assert (ctx["n_owned_A_h7"], ctx["n_owned_B_h7"]) == ("4", "1")
    n13 = sorted(int(x) for x in ctx["owned_coarse_csv_n13"].split(","))
    assert n13 == sorted(int(x) for h in (6, 7) for x in ctx[f"owned_coarse_csv_h{h}"].split(","))
    ctx4 = opr.render_context(NS4, plan=plans[NS4], plan_path=NODE_PLAN)
    assert (ctx4["n_sub"], ctx4["n_streams"]) == ("4", "64")
    assert all((ctx4[f"n_owned_A_h{h}"], ctx4[f"n_owned_B_h{h}"]) == ("2", "1") for h in range(8))
    assert opr.render_context("1x")["t_ext_us"] == "1048.576"
    with pytest.raises(opr.OpPointError):
        opr.render_context(NS2, plan=plans[NS2])


def test_check_plan_for(plans):
    assert opr.check_plan_for(NS2, plans[NS2]) == []
    assert opr.check_plan_for("2x-baselines-ns2-cint4", plans[NS2]) == []   # kind/bits: same plan
    assert opr.check_plan_for(NS2, plans[NS4])                             # n_sub differs
    assert opr.check_plan_for("1x", plans[NS2])


def test_render_rules():
    tmpl = {"opoint": "2x", "routines": [{"name": "a", "args": "--n ${n_sub} --x 1",
                                          "hostargs": {"n02": "--o ${opoint_id}"}}],
            "c1": {"v": 3}}
    ctx = opr.render_context(NS2)
    out = opr.render(tmpl, NS2, ctx)
    assert out["routines"][0]["args"] == "--n 2 --x 1"
    assert out["routines"][0]["hostargs"]["n02"] == f"--o {NS2}"
    assert (out["opoint"], out["opoint_selection"]) == (NS2, NS2)
    assert tmpl["routines"][0]["args"] == "--n ${n_sub} --x 1"      # input untouched
    with pytest.raises(opr.OpPointError, match="unknown template"):
        opr.render({"routines": [{"name": "a", "args": "${nope}"}]}, NS2, ctx)
    with pytest.raises(opr.OpPointError, match="outside routine"):
        opr.render({"routines": [], "c1": {"p": "${n_sub}"}}, NS2, ctx)
    with pytest.raises(opr.OpPointError, match="stamped"):
        opr.render({"opoint": "2x", "routines": []}, "1x", opr.render_context("1x"))
    one = opr.render({"routines": []}, "1x-legacy", opr.render_context("1x-legacy"))
    assert (one["opoint"], one["opoint_selection"]) == ("1x", "1x-legacy")
    assert "plan_id" not in one


# ----------------------------------------------------------------- validation


def test_find_placeholders_and_cross_refs():
    cfg = {
        "x2_todo": ["WS-D: review coinc"],
        "x2_cp2_recalibrate": ["X2TODO:WS-D:not-a-placeholder-here"],
        "routines": [{"name": "a", "args": "--x 1 --y X2TODO:WS-S:n-fdm --z X2TODO:WS-T:bits"}],
        "coinc": {"trigger_criteria_path": "/r/configs/c2_trigger_criteria.yaml"},
        "other": "/r/configs/opoints/2x/dsart_search_rt.yaml",
    }
    ph = opr.find_placeholders(cfg)
    assert ("x2_todo[0]", "WS-D: review coinc") in ph
    assert ("routines[a].args", "X2TODO:WS-S:n-fdm") in ph
    assert ("routines[a].args", "X2TODO:WS-T:bits") in ph
    assert len(ph) == 3
    assert opr.find_cross_refs(cfg, NS2) == [
        ("coinc.trigger_criteria_path", "configs/c2_trigger_criteria.yaml")]
    assert opr.find_cross_refs(cfg, "1x") == [
        ("other", "configs/opoints/2x/dsart_search_rt.yaml")]
    assert opr.recalibration_notes(cfg) == ["X2TODO:WS-D:not-a-placeholder-here"]


def test_validate_set_rendered_rules():
    good = {"routines": [], "opoint": NS2, "opoint_selection": NS2}
    rep = opr.validate_set({"pipeline_rt": good, "search_rt": good}, NS2, check_units=False)
    assert rep.ok and rep.summary() == "ok"
    rep = opr.validate_set({"pipeline_rt": good, "search_rt": good}, NS2)
    assert not rep.ok and any("not set explicitly" in p for p in rep.problems)
    rep = opr.validate_set({"pipeline_rt": {"routines": []}}, NS2, check_units=False)
    assert any("search_rt: missing" in p for p in rep.problems)
    assert any("rendered for None" in p for p in rep.problems)
    rep = opr.validate_set({"pipeline_rt": dict(good, opoint_selection=NS4),
                            "search_rt": "nope"}, NS2, check_units=False)
    assert any(f"expected '{NS2}'" in p for p in rep.problems)
    assert any("not a mapping" in p for p in rep.problems)
    left = dict(good, routines=[{"name": "a", "args": "--n ${n_sub}"}])
    rep = opr.validate_set({"pipeline_rt": left, "search_rt": good}, NS2, check_units=False)
    assert any("unrendered" in p for p in rep.problems)


def test_time_units_derive_from_the_opoint():
    t1, t2 = opr.time_units_for(OPPOINT_1X), opr.time_units_for(opoint_from_id(NS4))
    k = ("search_rt", "args", "search_compute_0", "--t-int-search-us")
    assert (t1[k], t2[k]) == (1048.576, 524.288)
    assert t2[("pipeline_rt", "args", "corr_fast", "--t-int-fast-native")] == 8
    assert t2[("pipeline_rt", "args", "corr_fast", "--chan-sum-factor")] == 2
    # the C1 width cap is 512 native = 16.8 ms at both op-points
    assert t1[("search_rt", "key", "c1.max_c1c2_width_samples")] == 16
    assert t2[("search_rt", "key", "c1.max_c1c2_width_samples")] == 32
    # the merger keeps 16 at 2x on purpose (wall-clock window; inventory finding 16)
    assert t1[("search_rt", "key", "c1.merger.sample_period_specnum")] == 16
    assert t2[("search_rt", "key", "c1.merger.sample_period_specnum")] == 16


def test_check_time_units_flags_missing_and_wrong():
    s = opr.render_repo_set("1x", CONFIGS)
    s["search_rt"]["routines"] = [
        dict(r, args=r["args"].replace("--t-int-search-us 1048.576", ""))
        if r["name"] == "search_compute_0" else r
        for r in s["search_rt"]["routines"]]
    s["search_rt"]["c1"] = dict(s["search_rt"]["c1"], max_c1c2_width_samples=32)
    probs = opr.check_time_units(s, "1x")
    assert any("search_compute_0] --t-int-search-us not set explicitly" in p for p in probs)
    assert any("max_c1c2_width_samples = 32" in p for p in probs)
    assert len(probs) == 2
    assert not opr.validate_set(s, "1x").ok
    assert opr.validate_set(s, "1x", check_units=False).ok


def test_routine_arg_last_occurrence_wins():
    cfg = {"routines": [{"name": "a", "args": "-u --x 1 --y 2 --x 3"}, {"name": "b", "args": ""}]}
    assert opr.routine_arg(cfg, "a", "--x") == "3"
    assert opr.routine_arg(cfg, "a", "--z") is None
    assert opr.routine_arg(cfg, "b", "--x") is None
    assert opr.routine_arg(cfg, "nope", "--x") is None


# ------------------------------------------------------------- config diff


def test_flatten_keys_routines_by_name():
    cfg = {"buffers": [{"k": "fada", "n": 70}], "routines": [{"name": "corr_fast", "args": "a b"}],
           "x": [1, 2], "e": {}}
    f = cdiff.flatten(cfg)
    assert f["buffers[fada].n"] == 70
    assert f["routines[corr_fast].args"] == "a b"
    assert f["x[1]"] == 2 and f["e"] == {}


def test_structural_diff_insert_does_not_shift():
    old = {"routines": [{"name": "a", "args": "-x 1"}, {"name": "b", "args": "-y 2"}]}
    new = {"routines": [{"name": "z", "args": "-q"}, {"name": "a", "args": "-x 1"},
                        {"name": "b", "args": "-y 3 --opoint 2x"}]}
    d = cdiff.structural_diff(old, new)
    kinds = {(e.path, e.kind) for e in d}
    assert kinds == {("routines[z].args", "added"), ("routines[z].name", "added"),
                     ("routines[b].args", "changed")}
    lines = "\n".join(cdiff.format_diff(d))
    assert "- 2" in lines and "+ 3 --opoint 2x" in lines
    assert cdiff.summarize(d) == {"added": 2, "removed": 0, "changed": 1}
    assert cdiff.structural_diff(new, new) == []
    assert {e.kind for e in cdiff.structural_diff(None, {"a": 1})} == {"added"}


def test_type_change_is_a_change():
    d = cdiff.structural_diff({"a": 1}, {"a": "1"})
    assert [(e.path, e.kind) for e in d] == [("a", "changed")]


def test_canonical_sha_ignores_key_order():
    assert cdiff.canonical_sha256({"a": 1, "b": [1, 2]}) == cdiff.canonical_sha256({"b": [1, 2], "a": 1})
    assert cdiff.canonical_sha256({"a": 1}) != cdiff.canonical_sha256({"a": 2})


# --------------------------------------------------- the repo's config sets


def test_repo_1x_sets_are_the_unstamped_base_files():
    for sel in ("1x", "1x-legacy"):
        for kind, fname in opr.CONFIG_FILES.items():
            assert opr.config_path(kind, sel, CONFIGS) == CONFIGS / fname
        raw = opr.load_repo_set(sel, CONFIGS)
        for cfg in raw.values():
            assert "opoint" not in cfg and "x2_todo" not in cfg
        s = opr.render_repo_set(sel, CONFIGS)
        rep = opr.validate_set(s, sel)
        assert rep.ok, rep.summary()
        assert s["search_rt"]["opoint"] == "1x" and s["search_rt"]["opoint_selection"] == sel
    for fname in opr.CONFIG_FILES.values():
        text = (CONFIGS / fname).read_text()
        assert "X2TODO" not in text and "${" not in text


@pytest.mark.parametrize("sel", [NS2, NS4])
def test_repo_2x_template_renders_per_format(sel, plans):
    s = opr.render_repo_set(sel, CONFIGS, plan=plans[sel], plan_path=NODE_PLAN)
    rep = opr.validate_set(s, sel)
    assert rep.ok, rep.summary()                    # stamps, units, no ${}, no placeholders
    op = opr.op_of(sel)
    pid = f"0x{plans[sel].plan_id:016x}"
    for inst in opr.INSTANCES:
        assert s[inst]["plan_id"] == pid
    sr = s["search_rt"]
    for half, name in ((0, "search_compute_0"), (1, "search_compute_1")):
        assert opr.routine_arg(sr, name, "--opoint") == sel
        assert opr.routine_arg(sr, name, "--plan-path") == NODE_PLAN
        assert opr.routine_arg(sr, name, "--plan-id") == pid
        assert opr.routine_arg(sr, name, "--gpu-half") == str(half)
        assert float(opr.routine_arg(sr, name, "--t-int-search-us")) == op.t_ext_us
    assert opr.routine_arg(sr, "search_rx", "--plan") == NODE_PLAN
    assert opr.routine_arg(sr, "search_rx", "--plan-id") == pid
    assert opr.routine_arg(sr, "search_rx", "--ring-version") == "4"
    cf = s["pipeline_rt"]
    assert opr.routine_arg(cf, "corr_fast", "--n-sub") == str(op.n_sub)
    assert opr.routine_arg(cf, "corr_fast", "--opoint") == sel


def test_repo_2x_template_is_self_contained():
    raw = opr.load_repo_set("2x", CONFIGS)
    for kind, cfg in raw.items():
        assert cfg["opoint"] == "2x", kind
        assert opr.find_cross_refs(cfg, NS2) == [], kind
    assert raw["search_rt"]["coinc"]["trigger_criteria_path"].endswith(
        "configs/opoints/2x/c2_trigger_criteria.yaml")
    for name in ("search_compute_0", "search_compute_1"):
        # the 2x halves read no yaml (their C1 values: tests/x2/test_g2_wiring.py)
        r = next(r for r in raw["search_rt"]["routines"] if r["name"] == name)
        assert "--config-yaml" not in r["args"] and "configs/dsart_" not in r["args"]


def test_repo_2x_set_keeps_the_1x_structure():
    """Same routines, buffers and top-level sections as 1x (values may differ)."""
    one = opr.load_repo_set("1x", CONFIGS)
    two = opr.load_repo_set("2x", CONFIGS)
    for inst in opr.INSTANCES:
        assert [r["name"] for r in one[inst]["routines"]] == [r["name"] for r in two[inst]["routines"]]
        assert one[inst]["buffers"] == two[inst]["buffers"]
        assert set(two[inst]) - set(one[inst]) <= {"opoint", "x2_todo", "x2_cp2_recalibrate"}
        assert set(one[inst]) - set(two[inst]) == set()


def test_repo_2x_k_isolation_and_c2_keys():
    one = opr.load_repo_set("1x", CONFIGS)
    two = opr.load_repo_set("2x", CONFIGS)
    assert two["search_rt"]["inject_bot"]["state_path"] != one["search_rt"]["inject_bot"]["state_path"]
    coinc = two["search_rt"]["coinc"]
    assert coinc["partial_event_policy"] == "archive"
    assert coinc["cutout"] == {"wait_s": 120, "status_port": 11501}
    assert "partial_event_policy" not in one["search_rt"]["coinc"]
    assert two["search_rt"]["c3"]["veto"]["r2_t_shift_ms"] == 31.5
    req = [c["require"] for c in two["c2_criteria"]["trigger_classes"]]
    assert all("width_median_max_samples" not in r for r in req)
    assert [r["width_median_max_ms"] for r in req if "width_median_max_ms" in r] == [16.8, 8.4, 33.6]
    assert all(r["dm_median_max_pc_cc"] >= 3000 for r in req if "dm_median_max_pc_cc" in r)
    assert opr.recalibration_notes(two["c2_criteria"])
    assert opr.recalibration_notes(one["c2_criteria"]) == []


def test_width_gates_sit_between_boxcar_widths():
    """No float ulp can move an allowed boxcar width across a 2x ms gate."""
    t_ext_ms = opoint_from_id(NS2).t_ext_us / 1e3
    widths_ms = [w * t_ext_ms for w in (1, 2, 4, 8, 16, 32, 64, 128)]
    for gate in (16.8, 8.4, 33.6):
        assert min(abs(gate - w) for w in widths_ms) > 1e-3


def test_repo_2x_placeholders_block_a_switch(plans):
    # every placeholder of the repo template was filled at G2 wiring (2026-10-29): switchable
    s = opr.render_repo_set(NS2, CONFIGS, plan=plans[NS2], plan_path=NODE_PLAN)
    rep = opr.validate_set(s, NS2)
    assert rep.ok and rep.placeholders == [], rep.summary()
    # one placeholder token or x2_todo entry anywhere blocks it again
    s["search_rt"]["routines"][0]["args"] += " --x X2TODO:WS-T:something"
    s["pipeline_rt"]["x2_todo"] = ["WS-C: open item"]
    rep = opr.validate_set(s, NS2)
    assert not rep.ok and "placeholder" in rep.summary()
    owners = {tok.split(":")[1] for _i, _p, tok in rep.placeholders if tok.startswith("X2TODO:")}
    assert owners == {"WS-T"} and len(rep.placeholders) == 2
