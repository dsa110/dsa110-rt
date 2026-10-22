"""TieredDmPlan v3 structure, ownership, npz round trip, validators, v2 coexistence."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dsart.x2.opoint import opoint_2x
from dsart.x2.plan import (
    PLAN_VERSION,
    SEARCH_NODES,
    TieredDmPlan,
    build_tiered_plan_for_opoint,
    load_any_plan,
    plan_metadata_version,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=[2, 4])
def plan(request):
    return build_tiered_plan_for_opoint(opoint_2x(n_sub=request.param), git_sha="test", build_utc_ns=0)


def test_counts_follow_the_plan(plan):
    nA, nB = (32, 15) if plan.n_sub == 2 else (16, 8)
    assert plan.n_coarse == nA + nB
    assert list(np.diff(plan.tier_fine_offsets)) == [526, 232]
    lo, hi = (15, 18) if plan.n_sub == 2 else (28, 35)
    assert plan.n_fdm.min() >= lo and plan.n_fdm.max() <= hi
    assert len(set(plan.n_fdm.tolist())) > 1                      # variable fines per coarse
    assert plan.n_fdm.sum() == plan.n_fine
    assert list(plan.tier_t_native) == [16, 32] and list(plan.tier_q1_native) == [8, 8]
    assert np.all(plan.coarse_t_native[:nA] == 16) and np.all(plan.coarse_t_native[nA:] == 32)
    assert plan.fine_dm[0] == 50.0 and plan.fine_dm[526] == 1600.0 and plan.fine_dm[-1] < 3000.0
    # coarse = bucket centres, uniform width per tier
    wA = np.diff(plan.coarse_dm[:nA])
    assert np.allclose(wA, 1550.0 / nA) and np.isclose(plan.coarse_dm[0], 50 + 0.5 * 1550.0 / nA)


def test_ownership_interleaves_over_8_halves(plan):
    d = np.arange(plan.n_coarse)
    assert np.array_equal(plan.owner_half, d % 8)
    nA = int(plan.tier_coarse_offsets[1])
    for h in range(8):
        mine = plan.dms_of_half(h)
        nA_h, nB_h = int(np.sum(mine < nA)), int(np.sum(mine >= nA))
        if plan.n_sub == 2:
            assert (nA_h, nB_h) == ((4, 1) if h == 7 else (4, 2))
        else:
            assert (nA_h, nB_h) == (2, 1)
        assert plan.owner_node(int(mine[0])) == SEARCH_NODES[h // 2]
        assert plan.owner_gpu(int(mine[0])) == h % 2
    assert SEARCH_NODES == ("n01", "n02", "n09", "n13")


def test_global_fine_index(plan):
    """SPEC §2/§9: f_global = plan order (tier A then B = d ascending, fines ascending in d)."""
    f = np.arange(plan.n_fine)
    assert np.all(np.diff(plan.fine_to_coarse) >= 0) and np.all(np.diff(plan.fine_dm) > 0)
    assert plan.fine_offset.shape == (plan.n_coarse + 1,) and plan.fine_offset[-1] == plan.n_fine
    assert np.array_equal(plan.fine_local_idx, f - plan.fine_offset[plan.fine_to_coarse])
    for d in range(plan.n_coarse):
        sl = plan.fines_of(d)
        assert np.array_equal(plan.fine_to_coarse[sl], np.full(plan.n_fdm[d], d))
        assert np.array_equal(plan.fine_local_idx[sl], np.arange(plan.n_fdm[d]))
        assert np.all(plan.fine_tier[sl] == plan.coarse_tier[d])
    seen = []
    for h in range(8):
        fh = plan.fines_of_half(h)
        assert np.all(np.diff(fh) > 0)
        assert np.array_equal(fh, np.concatenate([np.arange(plan.fine_offset[d], plan.fine_offset[d + 1])
                                                  for d in plan.dms_of_half(h)]))
        assert np.all(plan.owner_half[plan.fine_to_coarse[fh]] == h)
        seen.append(fh)
    assert np.array_equal(np.sort(np.concatenate(seen)), f)


def test_plan_id_is_a_function_of_the_tables_only(plan, tmp_path):
    again = build_tiered_plan_for_opoint(opoint_2x(n_sub=plan.n_sub), git_sha="other", build_utc_ns=12345)
    assert again.plan_id == plan.plan_id and 0 < plan.plan_id < 2 ** 64
    other = build_tiered_plan_for_opoint(opoint_2x(n_sub=6 - plan.n_sub), git_sha="test", build_utc_ns=0)
    assert other.plan_id != plan.plan_id
    p = tmp_path / "plan.npz"
    plan.to_npz(p)
    assert TieredDmPlan.from_npz(p).plan_id == plan.plan_id
    bad = tmp_path / "bad.npz"

    def bump(a):
        a = a.copy()
        a.flat[123] += 1
        return a
    _tamper(p, bad, stage1_bins=bump)                       # structurally valid, different table
    assert TieredDmPlan.from_npz(bad).plan_id != plan.plan_id
    assert plan.summary()["plan_id"] == f"0x{plan.plan_id:016x}"


def test_shift_tables_shapes_signs_and_ring_extremes(plan):
    nsum = 192
    assert plan.stage1_bins.shape == (16, nsum, plan.n_coarse) and plan.stage1_bins.dtype == np.int32
    assert plan.stage2_bins.shape == (plan.n_fine, 16 * plan.n_sub) and plan.stage2_bins.dtype == np.int32
    per = nsum // plan.n_sub
    assert np.all(plan.stage1_bins[:, ::per, :] == 0) and np.all(plan.stage1_bins >= 0)
    # stage 1 grows within a sub-band (lower frequency -> larger advance)
    for s in range(plan.n_sub):
        assert np.all(np.diff(plan.stage1_bins[:, s * per:(s + 1) * per, :], axis=1) >= 0)
    for d in range(plan.n_coarse):
        s2 = plan.stage2_for_coarse(d)
        assert s2.shape == (plan.n_fdm[d], plan.n_streams)
        assert s2.max() == plan.stage2_max[d] and s2.min() == plan.stage2_min[d]
    # the band-bottom stream of chgroup 15 has the smallest delay; ring depth ~ DM-1600 sweep
    assert plan.stage2_max.max() < 2000
    assert int(plan.stage2_max[int(plan.tier_coarse_offsets[1]) - 1]) > 1700


def test_npz_round_trip(plan, tmp_path):
    p = tmp_path / "plan.npz"
    plan.to_npz(p)
    back = TieredDmPlan.from_npz(p, verify=True)
    assert plan_metadata_version(p) == PLAN_VERSION == 3
    for f in dataclasses.fields(TieredDmPlan):
        a, b = getattr(plan, f.name), getattr(back, f.name)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b), f.name
        else:
            assert a == b, f.name
    assert back.metadata["inputs"]["tiers"][1]["t_native"] == 32
    assert back.metadata["git_sha"] == "test" and back.metadata["build_utc"] == "1970-01-01T00:00:00Z"
    assert isinstance(load_any_plan(p), TieredDmPlan)
    with pytest.raises(ValueError):
        back.stage2_bins[0, 0] = 1                                  # read-only arrays


def _tamper(path, out, **changes):
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.array(z[k]) for k in z.files}
    for k, fn in changes.items():
        d[k] = fn(d[k])
    np.savez(out, **d)


def test_validators_and_rebuild_check_fail_loudly(plan, tmp_path):
    p = tmp_path / "plan.npz"
    plan.to_npz(p)
    bad = tmp_path / "bad.npz"

    def bump(a):
        a = a.copy()
        a.flat[123] += 1
        return a
    _tamper(p, bad, stage2_bins=bump)
    with pytest.raises(ValueError):                         # stage2_max/min no longer match, or rebuild
        TieredDmPlan.from_npz(bad, verify=True)
    _tamper(p, bad, stage1_bins=bump)
    TieredDmPlan.from_npz(bad)                              # structurally valid ...
    with pytest.raises(ValueError, match="stage1_bins"):    # ... but not what the inputs build
        TieredDmPlan.from_npz(bad, verify=True)
    _tamper(p, bad, owner_half=lambda a: a[::-1].copy())
    with pytest.raises(ValueError, match="owner_half"):
        TieredDmPlan.from_npz(bad)
    _tamper(p, bad, n_fdm=lambda a: a + 1)
    with pytest.raises(ValueError, match="n_fdm"):
        TieredDmPlan.from_npz(bad)

    def v2(m):
        meta = json.loads(str(m))
        meta["version"] = 2
        return np.asarray(json.dumps(meta), dtype="U")
    _tamper(p, bad, metadata=v2)
    with pytest.raises(ValueError, match="not a TieredDmPlan"):
        TieredDmPlan.from_npz(bad)


DEPLOYED_V2 = Path("/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_dmmax1500_tol1.265_csf8_explicit-v2.npz")


@pytest.fixture(scope="module")
def v2_plan(tmp_path_factory) -> Path:
    """A v2 plan built by today's builder (the deployed grid's recipe)."""
    import os
    out = tmp_path_factory.mktemp("v2") / "plan_v2.npz"
    env = dict(os.environ, DSART_CONFIG_DIR=str(REPO / "configs"), PYTHONPATH=str(REPO / "src"))
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_dm_plan_explicit.py"), "--out", str(out),
         "--dm-min", "100", "--dm-max", "1500", "--beta", "0.05", "--chan-sum-factor", "8",
         "--t-int-fast-native", "32", "--t-int-search-us", "1048.576"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return out


def _check_v2_loads(p: Path) -> None:
    from dsart.coarse_dm.dm_plan import DMPlan
    from dsart.common.contracts import DmPlan
    assert plan_metadata_version(p) == 2
    canon = DmPlan.from_npz(str(p))
    assert canon.metadata["version"] == 2 and canon.coarse_dm.shape == (8,)
    assert DMPlan.from_npz(str(p)).n_coarse == 8
    assert isinstance(load_any_plan(p), DmPlan)
    with pytest.raises(ValueError, match="not a TieredDmPlan"):
        TieredDmPlan.from_npz(p)


def test_v2_plans_still_load_via_dmplan(v2_plan):
    _check_v2_loads(v2_plan)


@pytest.mark.skipif(not DEPLOYED_V2.is_file(), reason="deployed v2 plan not on this host")
def test_deployed_v2_plan_still_loads():
    _check_v2_loads(DEPLOYED_V2)


def test_legacy_v1_config_plan_is_not_mistaken_for_v3():
    p = REPO / "configs" / "dm_plan.npz"
    assert plan_metadata_version(p) == 1
    with pytest.raises(ValueError, match="not a TieredDmPlan"):
        TieredDmPlan.from_npz(p)
    with pytest.raises(ValueError, match="unsupported"):
        load_any_plan(p)


def test_build_tool_writes_a_verified_plan(tmp_path):
    out = tmp_path / "p.npz"
    r = subprocess.run([sys.executable, str(REPO / "tools" / "build_dm_plan_tiers.py"),
                        "--opoint", "2x-baselines-ns4-cint5", "--out", str(out), "--git-sha", "tooltest"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)
    assert summary["opoint"] == "2x-baselines-ns4-cint5" and summary["n_coarse"] == 24
    back = TieredDmPlan.from_npz(out, verify=True)
    assert back.n_sub == 4 and back.metadata["opoint_id"] == "2x-baselines-ns4-cint5"
