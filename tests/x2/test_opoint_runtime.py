"""dsart.x2.opoint_runtime + config_diff + the repo's op-point config sets (SPEC §10)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dsart.x2 import config_diff as cdiff
from dsart.x2 import opoint_runtime as opr

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


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


# --------------------------------------------------------------------- names


def test_names_and_keys():
    assert opr.OPOINTS == ("1x", "1x-legacy", "2x")
    assert opr.DEFAULT_OPOINT == "1x"
    assert opr.set_key("2x", "search_rt") == "/cnf/opoints/2x/search_rt"
    assert opr.meta_key("1x-legacy") == "/cnf/opoints/1x-legacy/meta"
    with pytest.raises(opr.OpPointError):
        opr.validate_name("4x")
    with pytest.raises(opr.OpPointError):
        opr.set_key("2x", "pipeline")
    assert opr.is_live_key("/cnf/pipeline_rt") and opr.is_live_key("/cnf/opoint")
    assert not opr.is_live_key("/cnf/opoints/2x/pipeline_rt")


@pytest.mark.parametrize("raw,want", [
    (None, None), ({"opoint": "2x", "user": "x"}, "2x"), ("1x-legacy", "1x-legacy"),
    ({"user": "x"}, None), (" 2x ", "2x"),
])
def test_parse_active_record(raw, want):
    assert opr.parse_active_record(raw) == want


@pytest.mark.parametrize("raw", [{"opoint": "3x"}, 7, ["2x"]])
def test_parse_active_record_invalid(raw):
    with pytest.raises(opr.OpPointError):
        opr.parse_active_record(raw)


def test_stamp():
    assert opr.stamp({"a": 1}, "2x") == {"a": 1, "opoint": "2x"}
    assert opr.stamp({"opoint": "2x"}, "2x") == {"opoint": "2x"}
    with pytest.raises(opr.OpPointError):
        opr.stamp({"opoint": "1x"}, "2x")
    assert opr.stamp_matches({}, "1x") and opr.stamp_matches({}, "1x-legacy")
    assert not opr.stamp_matches({}, "2x")
    assert opr.stamp_matches({"opoint": "2x"}, "2x")
    assert not opr.stamp_matches({"opoint": "1x"}, "1x-legacy")


# ---------------------------------------------------------------- resolution


def test_resolve_cli_wins(tmp_path):
    s = FakeStore({"/cnf/opoint": {"opoint": "2x"}})
    a = opr.resolve_active_opoint(s, cli="1x-legacy", cache_path=tmp_path / "c.json")
    assert (a.name, a.source) == ("1x-legacy", "cli")
    assert s.gets == []
    with pytest.raises(opr.OpPointError):
        opr.resolve_active_opoint(s, cli="9x")


def test_resolve_from_etcd_and_unset(tmp_path):
    c = tmp_path / "c.json"
    a = opr.resolve_active_opoint(FakeStore({"/cnf/opoint": {"opoint": "2x"}}), cache_path=c)
    assert (a.name, a.source) == ("2x", "etcd")
    assert c.is_file()
    a = opr.resolve_active_opoint(FakeStore({}), cli="auto", cache_path=c)
    assert (a.name, a.source) == ("1x", "etcd-unset")


def test_resolve_etcd_down_uses_cache_then_default(tmp_path):
    c = tmp_path / "c.json"
    opr.resolve_active_opoint(FakeStore({"/cnf/opoint": "2x"}), cache_path=c)
    sleeps = []
    a = opr.resolve_active_opoint(FakeStore(fail=5), cache_path=c, retries=3,
                                  retry_sleep_s=0.25, sleep_fn=sleeps.append)
    assert (a.name, a.source) == ("2x", "cache")
    assert "unreachable" in a.warning
    assert sleeps == [0.25, 0.25]
    a = opr.resolve_active_opoint(FakeStore(fail=5), cache_path=tmp_path / "none.json",
                                  retries=2, sleep_fn=lambda s: None)
    assert (a.name, a.source) == ("1x", "default")
    assert a.warning


def test_resolve_retry_then_success(tmp_path):
    s = FakeStore({"/cnf/opoint": {"opoint": "2x"}}, fail=1)
    a = opr.resolve_active_opoint(s, cache_path=tmp_path / "c.json", sleep_fn=lambda s: None)
    assert (a.name, a.source) == ("2x", "etcd")


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
    assert opr.config_path("pipeline_rt", "2x", base) == base / "opoints/2x/dsart_pipeline_rt.yaml"
    (base / "opoints" / "1x").mkdir()
    (base / "opoints" / "1x" / "dsart_search_rt.yaml").write_text("routines: []\n")
    assert opr.config_path("search_rt", "1x", base) == base / "opoints/1x/dsart_search_rt.yaml"
    with pytest.raises(opr.OpPointError):
        opr.config_path("bogus", "1x", base)


def test_config_path_2x_never_falls_back(tmp_path):
    base = _tree(tmp_path, with_2x=False)
    with pytest.raises(FileNotFoundError):
        opr.config_path("search_rt", "2x", base)


def test_map_config_path(tmp_path):
    base = _tree(tmp_path)
    unit_path = base / "dsart_search_rt.yaml"
    assert opr.map_config_path(unit_path, "2x") == base / "opoints/2x/dsart_search_rt.yaml"
    assert opr.map_config_path(unit_path, "1x") == unit_path
    assert opr.map_config_path(base / "opoints/2x/dsart_search_rt.yaml", "1x") == unit_path
    other = base / "chgroup_assignments.yaml"
    assert opr.map_config_path(other, "2x") == other


def test_select_service_config(tmp_path):
    base = _tree(tmp_path)
    s = FakeStore({"/cnf/opoint": {"opoint": "2x"}})
    active, path = opr.select_service_config(base / "dsart_search_rt.yaml", store=s,
                                             cache_path=tmp_path / "c.json")
    assert active.name == "2x" and path == base / "opoints/2x/dsart_search_rt.yaml"
    active, path = opr.select_service_config(base / "dsart_search_rt.yaml", cli_opoint="1x",
                                             store=s)
    assert active.source == "cli" and path == base / "dsart_search_rt.yaml"


def test_add_opoint_argument():
    p = argparse.ArgumentParser()
    opr.add_opoint_argument(p)
    assert p.parse_args([]).opoint == "auto"
    assert p.parse_args(["--opoint", "2x"]).opoint == "2x"
    with pytest.raises(SystemExit):
        p.parse_args(["--opoint", "3x"])


# ----------------------------------------------------------------- validation


def test_find_placeholders_and_cross_refs():
    cfg = {
        "x2_todo": ["WS-D: review coinc"],
        "routines": [{"name": "a", "args": "--x 1 --y X2TODO:WS-S:n-fdm --z X2TODO:WS-T:bits"}],
        "coinc": {"trigger_criteria_path": "/r/configs/c2_trigger_criteria.yaml"},
        "other": "/r/configs/opoints/2x/dsart_search_rt.yaml",
    }
    ph = opr.find_placeholders(cfg)
    assert ("x2_todo[0]", "WS-D: review coinc") in ph
    assert ("routines[a].args", "X2TODO:WS-S:n-fdm") in ph
    assert ("routines[a].args", "X2TODO:WS-T:bits") in ph
    assert len(ph) == 3
    assert opr.find_cross_refs(cfg, "2x") == [
        ("coinc.trigger_criteria_path", "configs/c2_trigger_criteria.yaml")]
    assert opr.find_cross_refs(cfg, "1x") == [
        ("other", "configs/opoints/2x/dsart_search_rt.yaml")]


def test_validate_set():
    good = {"routines": [], "opoint": "2x"}
    rep = opr.validate_set({"pipeline_rt": good, "search_rt": good}, "2x", require_stamp=True)
    assert rep.ok and rep.summary() == "ok"
    rep = opr.validate_set({"pipeline_rt": {"routines": []}}, "2x", require_stamp=True)
    assert not rep.ok
    assert any("search_rt: missing" in p for p in rep.problems)
    assert any("stamp" in p for p in rep.problems)
    rep = opr.validate_set({"pipeline_rt": {"routines": [], "opoint": "1x"},
                            "search_rt": "nope"}, "2x")
    assert any("expected '2x'" in p for p in rep.problems)
    assert any("not a mapping" in p for p in rep.problems)


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
    for op in ("1x", "1x-legacy"):
        for kind, fname in opr.CONFIG_FILES.items():
            assert opr.config_path(kind, op, CONFIGS) == CONFIGS / fname
        s = opr.load_repo_set(op, CONFIGS)
        rep = opr.validate_set(s, op)
        assert rep.ok, rep.summary()
        for cfg in s.values():
            assert "opoint" not in cfg and "x2_todo" not in cfg
    for fname in opr.CONFIG_FILES.values():
        assert "X2TODO" not in (CONFIGS / fname).read_text()


def test_repo_2x_set_is_complete_stamped_and_self_contained():
    s = opr.load_repo_set("2x", CONFIGS)
    for kind, cfg in s.items():
        assert opr.config_path(kind, "2x", CONFIGS).parent == CONFIGS / "opoints" / "2x"
        assert cfg["opoint"] == "2x", kind
    rep = opr.validate_set(s, "2x", require_stamp=True)
    assert rep.problems == []
    assert s["search_rt"]["coinc"]["trigger_criteria_path"].endswith(
        "configs/opoints/2x/c2_trigger_criteria.yaml")
    for name in ("search_compute_0", "search_compute_1"):
        r = next(r for r in s["search_rt"]["routines"] if r["name"] == name)
        assert "configs/opoints/2x/dsart_search_rt.yaml" in r["args"]


def test_repo_2x_set_keeps_the_1x_structure():
    """Same routines, buffers and top-level sections as 1x (values may differ)."""
    one = opr.load_repo_set("1x", CONFIGS)
    two = opr.load_repo_set("2x", CONFIGS)
    for inst in opr.INSTANCES:
        assert [r["name"] for r in one[inst]["routines"]] == [r["name"] for r in two[inst]["routines"]]
        assert one[inst]["buffers"] == two[inst]["buffers"]
        assert set(two[inst]) - set(one[inst]) == {"opoint", "x2_todo"}


def test_repo_2x_node_services_get_opoint_arg():
    two = opr.load_repo_set("2x", CONFIGS)
    by = {r["name"]: r for inst in opr.INSTANCES for r in two[inst]["routines"]}
    for name in ("corr_fast", "search_rx", "search_compute_0", "search_compute_1"):
        assert "--opoint 2x" in by[name]["args"], name


def test_repo_2x_placeholders_block_a_switch():
    s = opr.load_repo_set("2x", CONFIGS)
    rep = opr.validate_set(s, "2x")
    assert not rep.ok
    owners = {tok.split(":")[1] for _i, _p, tok in rep.placeholders if tok.startswith("X2TODO:")}
    assert {"WS-C", "WS-S", "WS-T", "WS-P"} <= owners
    assert "placeholder" in rep.summary()
