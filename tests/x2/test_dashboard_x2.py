"""x2 D11: dashboard (counts, Dump Now at 2x, health/ownership) + Grafana 2x."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DASH = REPO / "tools" / "dashboard" / "dsa_monitor"
GRAF = REPO / "tools" / "dashboard" / "dsart_rt_to_influx" / "grafana"
SEL2 = "2x-gridded-ns2-cint4"


@pytest.fixture(scope="module")
def dash():
    sys.path.insert(0, str(DASH))
    sys.path.insert(0, str(GRAF))
    try:
        import build_dashboard  # noqa: F401
        import cands_panel_funcs
        import cube_dump_now
        import x2_health
        yield cands_panel_funcs, cube_dump_now, x2_health, build_dashboard
    finally:
        sys.path.remove(str(DASH))
        sys.path.remove(str(GRAF))


class _Store:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})

    def get_dict(self, key):
        return self.docs.get(key)


def test_event_counts_cutouts_vs_cubes(dash, tmp_path: Path) -> None:
    cp = dash[0]
    root = tmp_path / "cands"
    ev = root / "261022cntx"
    (ev / "cutouts").mkdir(parents=True)
    for sid, g in ((1, 0), (1, 1), (2, 0)):
        (ev / "cutouts" / f"cutout_s{sid}_g{g}.json").write_text("{}")
        (ev / "cutouts" / f"cutout_s{sid}_g{g}.npz").write_bytes(b"x")
    (ev / "Level3").mkdir()
    (ev / "Level3" / "261022cntx.json").write_text(json.dumps({
        "c2": {"snr_max": 20.0},
        "products": {"expected": ["s1g0", "s1g1", "s2g0", "s2g1"],
                     "missing": {"s2g1": "no_status"}, "status": "incomplete"}}))
    ev1 = root / "260801cube"
    (ev1 / "cubes").mkdir(parents=True)
    for i in range(8):
        (ev1 / "cubes" / f"cube_s{1 + i // 2}_g{i % 2}_5.npz").write_bytes(b"x")
    ab = cp.ArchiveBrowser(root)
    by = {e.name: e for e in ab.list_events()}
    assert (by["261022cntx"].n_cubes, by["261022cntx"].n_products_expected,
            by["261022cntx"].products_kind) == (3, 4, "cutouts")
    assert (by["260801cube"].n_cubes, by["260801cube"].n_products_expected,
            by["260801cube"].products_kind) == (8, 8, "cubes")
    d = ab.event_detail("261022cntx")
    assert len(d.cutouts) == 3 and d.products["status"] == "incomplete"
    assert ab.event_detail("260801cube").cutouts == ()


def test_dump_now_at_2x_sends_manual_cutouts(dash) -> None:
    from dsart.coinc import wire
    dn = dash[1]
    rec = {"opoint": SEL2, "selection": SEL2, "plan_id": "0x00000000000000ff"}
    store = _Store({"/cnf/opoint": rec,
                    **{f"/mon/search_rt/{s}/cutout/{g}": {"raw_ring_newest_ext": 10_000_000 + s}
                       for s in (1, 2, 9, 13) for g in (0, 1)}})
    assert dn.is_2x(dn.active_opoint_record(store))
    assert not dn.is_2x(dn.active_opoint_record(_Store()))
    sent = []
    hosts = [(s, g, f"10.0.0.{s}", 11227 + g) for s in (1, 2, 9, 13) for g in (0, 1)]
    res = dn.fleet_cutout_now(store, record=rec, halves=[(9, 1), (13, 0)],
                              raw=True, bind_hosts=hosts,
                              sender=lambda b, a, t: sent.append((b, a)))
    assert res.ok and res.pass_count == 2
    assert sorted(a for _, a in sent) == [("10.0.0.13", 11227), ("10.0.0.9", 11228)]
    req = wire.decode_cutout_request(sent[0][0])
    assert req.is_manual and req.wants_raw and not req.has_pixel
    assert req.plan_id == 0xFF
    margin = int(round(3.0e6 / 524.288))
    assert req.t_peak_ext == (10_000_009 - margin) - ((10_000_009 - margin) % 2)
    none = dn.fleet_cutout_now(_Store(), record=rec, bind_hosts=hosts,
                               sender=lambda *a: None)
    assert not none.ok and none.error


def test_x2_health_and_ownership(dash, tmp_path: Path) -> None:
    from dsart.x2.opoint import opoint_from_id
    from dsart.x2.plan import build_tiered_plan_for_opoint
    xh = dash[2]
    plan = build_tiered_plan_for_opoint(opoint_from_id(SEL2))
    pp = tmp_path / "plan.npz"
    plan.to_npz(pp)
    store = _Store({
        "/mon/search_rt/1/compute/0": {"gpu_duty": 0.5, "coarse_dms": "0,8"},
        "/mon/search_rt/1/compute/1": {"gpu_duty": 0.8},
        "/mon/search_rt/9/rx": {"loss_frac": 5e-5, "rx_gbps": 12.0},
        "/mon/corr_rt/0/corr_fast": {"block_ms_p50": 100.0, "block_ms_p99": 130.0},
        "/mon/corr_rt/1/corr_fast": {"block_ms_p50": 120.0, "block_ms_p99": 130.0},
    })
    d = xh.gather(store, {"opoint": SEL2, "plan_path": str(pp)})
    halves = {h["half"]: h for h in d["halves"]}
    assert halves["s1g0"]["duty_gate"] == "ok" and halves["s1g1"]["duty_gate"] == "FAIL"
    assert halves["s2g0"]["duty_gate"] == "unknown"
    corr = {c["chgroup"]: c for c in d["corr"]}
    assert corr[0]["block_gate"] == "ok" and corr[1]["block_gate"] == "FAIL"
    assert d["overall"] == "FAIL"
    own = d["ownership"]
    assert len(own) == 8 and sum(o["n_fine"] for o in own) == plan.n_fine
    assert all(c["coarse"] % 8 == o["half"] for o in own for c in o["coarse"])
    json.dumps(d)


def test_grafana_2x_variant(dash) -> None:
    bd = dash[3]
    bd._reset_layout()
    one = bd.build_dashboard()
    two = bd.build_dashboard(SEL2)
    assert one["uid"] == bd.DASHBOARD_UID and two["uid"] == bd.DASHBOARD_UID + "x2"
    q1 = json.dumps(one)
    q2 = json.dumps(two)
    assert "search_rt_dm" not in q1 and "corr_rt_corr_fast" not in q1
    assert "search_rt_dm" in q2 and "corr_rt_corr_fast" in q2
    assert "A2. Search GPU health" in q1 and "A2. Search GPU health" not in q2
    ids = [p["id"] for p in two["panels"]]
    assert len(ids) == len(set(ids)) and min(ids) == 1
