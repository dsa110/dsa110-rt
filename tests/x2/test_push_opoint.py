"""tools/ops/push_dsart_to_etcd.py --opoint (SPEC §10) against a fake etcd store."""

from __future__ import annotations

import copy
import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

from dsart.x2 import opoint_runtime as opr
from dsart.x2.opoint import opoint_from_id
from dsart.x2.plan import build_tiered_plan_for_opoint

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
NS2 = "2x-gridded-ns2-cint4"
NS4 = "2x-gridded-ns4-cint5"

_spec = importlib.util.spec_from_file_location(
    "push_dsart_to_etcd", REPO / "tools" / "ops" / "push_dsart_to_etcd.py")
push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(push)


class FakeStore:
    def __init__(self, kv=None):
        self.kv = copy.deepcopy(kv or {})
        self.puts: list[str] = []

    def get_dict(self, key):
        v = self.kv.get(key)
        return copy.deepcopy(v)

    def put_dict(self, key, value):
        self.puts.append(key)
        self.kv[key] = copy.deepcopy(value)


@pytest.fixture(scope="module")
def plan_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("plans")
    out = {}
    for sel in (NS2, NS4):
        p = d / f"{sel}.npz"
        build_tiered_plan_for_opoint(opoint_from_id(sel)).to_npz(p)
        out[sel] = p
    return out


def _clean_2x_configs(tmp_path) -> Path:
    """Copy of configs/ whose 2x template has every placeholder filled with a dummy."""
    dst = tmp_path / "configs"
    shutil.copytree(CONFIGS, dst, ignore=shutil.ignore_patterns("*.npz", "dm_plans"))
    for f in (dst / "opoints" / "2x").glob("*.yaml"):
        doc = yaml.safe_load(f.read_text())
        doc.pop("x2_todo", None)
        text = yaml.safe_dump(doc, sort_keys=False)
        text = opr.PLACEHOLDER_RE.sub("7", text)
        f.write_text(text)
    return dst


def _push(store, sel, cfg_dir, out, **kw):
    return push.push_opoint(store, sel, instances=["pipeline_rt", "search_rt"],
                            config_dir=Path(cfg_dir), out=out.append,
                            now_utc=lambda: "2026-10-22T00:00:00Z", **kw)


def test_dry_run_is_the_default_and_writes_nothing(plan_files):
    s = FakeStore()
    out: list[str] = []
    rc = _push(s, NS2, CONFIGS, out, plan_path=str(plan_files[NS2]))
    assert rc == push.RC_OK
    assert s.puts == []
    text = "\n".join(out)
    assert "DRY RUN" in text and "X2TODO" in text and "dry run: nothing written" in text
    assert f"/cnf/opoints/{NS2}/search_rt" in text
    assert "recalibrate at CP2" in text


def test_write_refuses_placeholders(plan_files):
    s = FakeStore()
    out: list[str] = []
    rc = _push(s, NS2, CONFIGS, out, plan_path=str(plan_files[NS2]), write=True)
    assert rc == push.RC_PLACEHOLDERS and s.puts == []


def test_write_with_allow_placeholders_writes_only_set_keys(plan_files):
    s = FakeStore({"/cnf/pipeline_rt": {"routines": []}, "/cnf/search_rt": {"routines": []}})
    out: list[str] = []
    rc = _push(s, NS2, CONFIGS, out, plan_path=str(plan_files[NS2]),
               plan_node_path="/home/ubuntu/data/dm_plans/p2.npz", write=True,
               allow_placeholders=True)
    assert rc == push.RC_OK, "\n".join(out)
    assert sorted(s.puts) == sorted([f"/cnf/opoints/{NS2}/pipeline_rt",
                                     f"/cnf/opoints/{NS2}/search_rt", f"/cnf/opoints/{NS2}/meta"])
    assert s.kv["/cnf/pipeline_rt"] == {"routines": []}            # live untouched
    sr = s.kv[f"/cnf/opoints/{NS2}/search_rt"]
    assert sr["opoint"] == NS2 and sr["opoint_selection"] == NS2
    assert sr["plan_path"] == "/home/ubuntu/data/dm_plans/p2.npz"
    assert opr.routine_arg(sr, "search_compute_0", "--dm-plan-path") == sr["plan_path"]
    meta = s.kv[f"/cnf/opoints/{NS2}/meta"]
    assert meta["opoint"] == NS2 and meta["config_set"] == "2x" and meta["plan_id"] == sr["plan_id"]
    assert meta["placeholders"] > 0 and meta["pushed_utc"] == "2026-10-22T00:00:00Z"
    # second run: no change
    out2: list[str] = []
    _push(s, NS2, CONFIGS, out2, plan_path=str(plan_files[NS2]),
          plan_node_path="/home/ubuntu/data/dm_plans/p2.npz")
    assert "(no change)" in "\n".join(out2)


def test_clean_set_writes_without_flag_and_diff_shows_tokens(tmp_path, plan_files):
    cfg = _clean_2x_configs(tmp_path)
    s = FakeStore()
    out: list[str] = []
    rc = _push(s, NS2, cfg, out, plan_path=str(plan_files[NS2]), write=True)
    assert rc == push.RC_OK, "\n".join(out)
    rep = opr.validate_set({i: s.kv[opr.set_key(NS2, i)] for i in opr.INSTANCES}, NS2)
    assert rep.ok, rep.summary()
    # change one arg in the template: the diff is reported token by token
    f = cfg / "opoints" / "2x" / "dsart_search_rt.yaml"
    f.write_text(f.read_text().replace("--layer1-max-samples 10000", "--layer1-max-samples 20000"))
    out = []
    _push(s, NS2, cfg, out, plan_path=str(plan_files[NS2]))
    text = "\n".join(out)
    assert "~ routines[search_compute_0].args (tokens)" in text
    assert "- 10000" in text and "+ 20000" in text


def test_plan_must_match_the_selection(plan_files):
    out: list[str] = []
    rc = _push(FakeStore(), NS2, CONFIGS, out, plan_path=str(plan_files[NS4]))
    assert rc == push.RC_INVALID and "not the" in "\n".join(out)
    out = []
    assert _push(FakeStore(), NS2, CONFIGS, out) == push.RC_USAGE        # no --plan
    out = []
    assert _push(FakeStore(), "1x", CONFIGS, out, plan_path=str(plan_files[NS2])) == push.RC_USAGE


def test_1x_push_renders_the_base_set():
    s = FakeStore()
    out: list[str] = []
    rc = _push(s, "1x-legacy", CONFIGS, out, write=True)
    assert rc == push.RC_OK, "\n".join(out)
    pr = s.kv["/cnf/opoints/1x-legacy/pipeline_rt"]
    base = yaml.safe_load((CONFIGS / "dsart_pipeline_rt.yaml").read_text())
    assert {k: v for k, v in pr.items() if k not in ("opoint", "opoint_selection")} == base
    assert (pr["opoint"], pr["opoint_selection"]) == ("1x", "1x-legacy")


def test_seed_from_live(tmp_path):
    live_p = {"routines": [{"name": "corr_fast", "args": "--t-int-fast-native 32 --chan-sum-factor 8"}],
              "drift": 1}
    live_s = yaml.safe_load((CONFIGS / "dsart_search_rt.yaml").read_text())
    s = FakeStore({"/cnf/pipeline_rt": live_p, "/cnf/search_rt": live_s})
    out: list[str] = []
    rc = _push(s, "1x-legacy", CONFIGS, out, seed_from_live=True, write=True)
    assert rc == push.RC_OK, "\n".join(out)
    assert s.kv["/cnf/opoints/1x-legacy/pipeline_rt"]["drift"] == 1
    assert s.kv["/cnf/opoints/1x-legacy/meta"]["source"] == "live"
    out = []
    assert _push(s, NS2, CONFIGS, out, seed_from_live=True) == push.RC_USAGE
    # live rendered for another selection: refused
    s.kv["/cnf/pipeline_rt"] = dict(live_p, opoint_selection="1x")
    out = []
    assert _push(s, "1x-legacy", CONFIGS, out, seed_from_live=True) == push.RC_INVALID


def test_time_units_are_enforced(tmp_path):
    cfg = tmp_path / "configs"
    shutil.copytree(CONFIGS, cfg, ignore=shutil.ignore_patterns("*.npz", "opoints"))
    f = cfg / "dsart_search_rt.yaml"
    f.write_text(f.read_text().replace("--t-int-search-us 1048.576", "--t-int-search-us 524.288"))
    s = FakeStore()
    out: list[str] = []
    rc = _push(s, "1x", cfg, out, write=True)
    assert rc == push.RC_INVALID and s.puts == []
    assert "= '524.288', op-point 1x needs 1048.576" in "\n".join(out)


def test_diff_live_and_bad_selection(plan_files):
    s = FakeStore({"/cnf/search_rt": {"routines": []}, "/cnf/pipeline_rt": {"routines": []}})
    out: list[str] = []
    _push(s, NS2, CONFIGS, out, plan_path=str(plan_files[NS2]), diff_live=True)
    assert "vs LIVE /cnf/search_rt" in "\n".join(out)
    out = []
    assert _push(s, "2x", CONFIGS, out) == push.RC_USAGE


def test_legacy_mode_unchanged(tmp_path):
    s = FakeStore()
    assert push.main(["--dry-run", "--config-dir", str(CONFIGS)], store=s) == 0
    assert s.puts == []
    assert push.main(["--config-dir", str(CONFIGS)], store=s) == 0
    assert sorted(s.puts) == ["/cnf/pipeline_rt", "/cnf/search_rt"]
    assert s.kv["/cnf/search_rt"] == yaml.safe_load((CONFIGS / "dsart_search_rt.yaml").read_text())
    assert push.main(["--write", "--config-dir", str(CONFIGS)], store=FakeStore()) == 2
    assert push.main(["--opoint", "1x", "--write", "--dry-run", "--config-dir", str(CONFIGS)],
                     store=FakeStore()) == 2


def test_main_opoint_mode_routes_to_push_opoint(plan_files):
    s = FakeStore()
    rc = push.main(["--opoint", NS2, "--plan", str(plan_files[NS2]), "--config-dir", str(CONFIGS)],
                   store=s)
    assert rc == push.RC_OK and s.puts == []
