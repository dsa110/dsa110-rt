"""x2 D6: C2 cutout lifecycle, never-discard, op-point criteria, flags."""

from __future__ import annotations

import asyncio
import functools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import dsart.services.coincidencer as c2mod
from dsart.coinc import wire
from dsart.coinc.criteria import CriteriaEvaluator
from dsart.coinc.cutout_lifecycle import (
    CutoutConfig,
    CutoutTracker,
    build_request,
    expected_halves,
    products_block,
)
from dsart.coinc.stats import compute_stats
from dsart.coinc.window import WindowEntry
from dsart.services.coincidencer import (
    CoincidencerConfig,
    CoincidencerService,
    search_to_snap_specnum,
)

HOSTS = {1: "10.0.0.1", 2: "10.0.0.2", 9: "10.0.0.9", 13: "10.0.0.13"}
ALL8 = [(s, g) for s in (1, 2, 9, 13) for g in (0, 1)]


def asyncio_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


class _Store:
    def __init__(self) -> None:
        self.puts: Dict[str, Any] = {}

    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        if key.endswith("dumps_enabled"):
            return {"enabled": True, "ts": 0.0}
        return None

    def put_dict(self, key: str, value: Dict[str, Any]) -> None:
        self.puts[key] = dict(value)


class _Bc:
    """TriggerBroadcaster stand-in recording cube triggers and cutouts."""

    def __init__(self) -> None:
        self.cube_calls: List[Dict[str, Any]] = []
        self.sent: List[tuple] = []

    def broadcast(self, **kw) -> Dict[tuple, bool]:
        self.cube_calls.append(kw)
        return {sg: True for sg in ALL8}

    def send_to(self, blob: bytes, halves) -> Dict[tuple, bool]:
        halves = [tuple(h) for h in halves]
        self.sent.append((wire.decode_cutout_request(blob), halves))
        return {h: True for h in halves}

    def close(self) -> None:
        pass


def _criteria(tmp: Path, body: str) -> Path:
    p = tmp / "crit.yaml"
    p.write_text(body)
    return p


_DUMP_CRIT = """
trigger_classes:
  - name: bright
    require: {snr_max_min: 10.0, n_events_min: 1}
    action: dump_all_gpus
    holdoff_s: 30.0
  - name: galactic
    require: {snr_max_min: 10.0, n_events_min: 1}
    action: dump_all_gpus
    holdoff_s: 30.0
  - name: log_only
    require: {n_events_min: 1}
    action: log_only
"""


def _svc(tmp: Path, *, cutout: bool = True, policy: str = "discard",
         dedupe: bool = False, crit: str = _DUMP_CRIT, plan_path=None):
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1", bind_port=0, window_s=5.0,
        csv_dir_c1=tmp / "c1", csv_dir_c2=tmp / "c2",
        event_archive_root=tmp / "events",
        trigger_criteria_path=_criteria(tmp, crit),
        name_allocator_offline=True, gal_dm_poll_interval_s=60.0,
        startup_grace_s=0.0, dump_broadcast_hosts=dict(HOSTS),
        archive_schema=2 if cutout else 1, opoint="2x" if cutout else "",
        cutout=CutoutConfig(enabled=cutout, wait_s=120.0,
                            plan_path=plan_path),
        partial_event_policy=policy, dedupe_fired_clusters=dedupe,
        plotter_use_process_pool=False,
    )
    bc = _Bc()
    svc = CoincidencerService(config=cfg, mon_store=_Store(), broadcaster=bc)
    return svc, bc


def _batch(mjd: float, n: int, *, snr: float = 20.0, spec0: int = 1000,
           period: float = 524.288, sid: int = 1, g: int = 0,
           dm_idx: int = 5) -> wire.C1Batch:
    hdr = wire.build_header(
        cube_id=1, event_specnum_start=spec0, mjd_start=mjd,
        sample_period_specnum=8, sample_period_us=period, n_grid=256,
        n_fdm_in_cube=17, search_node_id=sid, gpu_half=g, n_candidates=n)
    rows = tuple(
        wire.C1CandidateRow(
            snr=snr + 0.1 * i, l_rad=0.001, m_rad=-0.002, l_pix=120,
            m_pix=131, dm_pc_cc=500.0 + i, dm_idx_global=dm_idx + i,
            fine_dm_idx=i, event_specnum=spec0 + 2 * i, width_samples=4,
            kernel_id="unit:d1:b4", flags=0)
        for i in range(n))
    return wire.C1Batch(header=hdr, candidates=rows)


def _manifest(ev: Path, sid: int, g: int) -> None:
    d = ev / "cutouts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"cutout_s{sid}_g{g}.npz").write_bytes(b"x")
    (d / f"cutout_s{sid}_g{g}.json").write_text("{}")


@pytest.fixture
def plots(monkeypatch):
    got: List[str] = []
    monkeypatch.setattr(c2mod, "enqueue_event",
                        lambda worker, name, root, **kw: got.append(name))
    return got


# ------------------------------------------------------------- criteria


def _stats(width: int, period: float, snr: float = 20.0):
    e = WindowEntry(mjd=61335.0, snr=snr, l_rad=0.0, m_rad=0.0, l_pix=0,
                    m_pix=0, dm_pc_cc=500.0, dm_idx_global=1, fine_dm_idx=1,
                    event_specnum=10, width_samples=width,
                    kernel_id="unit:d1:b4", flags=0, search_node_id=1,
                    gpu_half=0, cube_id=1, sample_period_us=period)
    return compute_stats([e])


def test_ms_width_predicates_and_opoint_sections(tmp_path: Path) -> None:
    p = _criteria(tmp_path, """
trigger_classes:
  - {name: top, require: {width_median_max_samples: 16}, action: log_only}
opoints:
  2x:
    trigger_classes:
      - {name: two_x, require: {width_median_max_ms: 16.8}, action: log_only}
""")
    top = CriteriaEvaluator(p)
    assert [c.name for c in top.classes] == ["top"]
    x2 = CriteriaEvaluator(p, opoint="2x-gridded-ns2-cint4")   # family match
    assert [c.name for c in x2.classes] == ["two_x"]
    assert CriteriaEvaluator(p, opoint="1x").classes[0].name == "top"
    # 32 T_ext = 16.777 ms passes; 33 does not
    assert x2.evaluate(_stats(32, 524.288)).name == "two_x"
    x2b = CriteriaEvaluator(p, opoint="2x")
    assert x2b.evaluate(_stats(33, 524.288)) is None
    # unknown period fails the ms gate closed
    assert CriteriaEvaluator(p, opoint="2x").evaluate(_stats(1, 0.0)) is None
    # hot reload keeps the section
    time.sleep(0.01)
    p.write_text(p.read_text().replace("16.8", "8.4"))
    import os
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert x2.reload_if_changed()
    assert x2.classes[0].require["width_median_max_ms"] == 8.4


def test_specnum_fallback_factor() -> None:
    assert search_to_snap_specnum(100, 1.0) == 1600
    assert search_to_snap_specnum(100, 1.0, 8) == 800
    assert search_to_snap_specnum(100, 524.288, 16) == 800


# --------------------------------------------------- tracker / request


def test_tracker_resends_until_ack(tmp_path: Path) -> None:
    t = [0.0]
    sent = []

    def send(blob, halves):
        sent.append((wire.decode_cutout_request(blob), list(halves)))
        return {h: True for h in halves}

    tr = CutoutTracker(send, CutoutConfig(ack_timeout_s=2.0, max_resends=2),
                       now=lambda: t[0])
    req = wire.CutoutRequest("261022abcd", 9, t_peak_ext=1000, t0_ext=744)
    tr.start(req, ALL8)
    assert len(sent) == 1 and not sent[0][0].flags & wire.CUTOUT_FLAG_RESEND
    ack = wire.CutoutStatus("261022abcd", 9, wire.CUTOUT_STATE_ACCEPTED, 1, 0)
    tr.on_datagram(wire.encode_cutout_status(ack))
    t[0] = 2.5
    tr.service_resends()
    assert len(sent[1][1]) == 7 and (1, 0) not in sent[1][1]
    assert sent[1][0].flags & wire.CUTOUT_FLAG_RESEND
    t[0] = 5.0
    tr.service_resends()
    t[0] = 8.0
    tr.service_resends()                      # max_resends=2 reached
    assert len(sent) == 3
    assert tr.counters["cutout_resends"] == 14
    # terminal status sticks; late ACCEPTED does not overwrite it
    done = wire.CutoutStatus("261022abcd", 9, wire.CUTOUT_STATE_PARTIAL, 2, 1,
                             reason=wire.CUTOUT_REASON_TOO_LATE)
    tr.on_status(done)
    tr.on_status(wire.CutoutStatus("261022abcd", 9,
                                   wire.CUTOUT_STATE_ACCEPTED, 2, 1))
    assert tr.half_states("261022abcd")[(2, 1)].state == "partial"
    assert not tr.on_status(wire.CutoutStatus("x", 1, 1, 1, 0))
    tr.on_datagram(b"junk")
    assert tr.counters["cutout_status_bad"] == 1


class _Plan:
    fine_to_coarse = [0, 0, 1, 1, 1, 2]
    owner_half = [0, 1, 2]


def test_build_request_from_peak_member() -> None:
    members = [
        WindowEntry(mjd=61335.0 + i * 1e-6, snr=s, l_rad=0.01, m_rad=0.02,
                    l_pix=100 + i, m_pix=50, dm_pc_cc=300.0 + i,
                    dm_idx_global=i, fine_dm_idx=i, event_specnum=5000 + i,
                    width_samples=2 + i, kernel_id="unit:d1:b2", flags=0,
                    search_node_id=1, gpu_half=0, cube_id=3,
                    sample_period_us=524.288)
        for i, s in enumerate((11.0, 30.0, 30.0, 12.0))]
    st = compute_stats(members)
    cfg = CutoutConfig(rows=512)
    r = build_request(event_name="261022abcd", request_id=4, stats=st,
                      members=members, cfg=cfg, trigger_class_name="bright",
                      is_injection=False, plan=_Plan(), plan_id=77)
    assert r.t_peak_ext == 5001 and r.l_pix == 101 and r.width_ext == 3
    assert r.fine_global_peak == 1 and r.coarse_global_peak == 0
    assert r.t0_ext % 2 == 0 and r.t0_ext <= 5001 < r.t0_ext + 512
    assert r.raw_pre_ext == 954 and r.plan_id == 77
    assert r.wants_raw and not r.flags & wire.CUTOUT_FLAG_INJECTION
    ri = build_request(event_name="261022abcd", request_id=5, stats=st,
                       members=members, cfg=cfg, trigger_class_name="bright",
                       is_injection=True, plan=_Plan())
    assert ri.flags & wire.CUTOUT_FLAG_INJECTION and not ri.wants_raw
    # a global index outside the plan (1x local index) is not forwarded
    far = [m for m in members]
    st2 = compute_stats(far)
    r2 = build_request(event_name="261022abcd", request_id=6, stats=st2,
                       members=[members[0]], cfg=cfg, trigger_class_name="b",
                       is_injection=False, plan=type("P", (), {
                           "fine_to_coarse": []})())
    assert r2.fine_global_peak == -1 and r2.coarse_global_peak == -1


def test_expected_halves_from_owner_map() -> None:
    assert expected_halves(_Plan()) == [(1, 0), (1, 1), (2, 0)]
    assert expected_halves(object()) == ALL8


def test_products_block_reasons() -> None:
    from dsart.coinc.cutout_lifecycle import HalfState
    hs = {(1, 1): HalfState(acked=True, state="failed", reason="too_late"),
          (2, 0): HalfState()}
    b = products_block(kind="cutouts", expected=[(1, 0), (1, 1), (2, 0), (9, 0)],
                       present=[(1, 0)], half_states=hs, plan_id=5)
    assert b["status"] == "incomplete" and b["present"] == ["s1g0"]
    assert b["missing"] == {"s1g1": "failed:too_late", "s2g0": "no_status",
                            "s9g0": "no_product"}


# ------------------------------------------------------- service level


@asyncio_test
async def test_cutout_event_complete(tmp_path: Path, plots) -> None:
    svc, bc = _svc(tmp_path)
    await svc._on_batch(_batch(61335.0, 3), peer_repr="x")
    name = svc._last_event_name
    assert bc.cube_calls == [] and len(bc.sent) == 1
    req, halves = bc.sent[0]
    assert sorted(halves) == ALL8 and req.event_name == name
    assert req.t_peak_ext == 1004 and req.l_pix == 120 and req.wants_raw
    ev = tmp_path / "events" / name
    # the record exists from fire time (C2 crash cannot lose it)
    assert (ev / "Level2" / f"C2_{name}.csv").is_file()
    assert (ev / "Level2" / "products_pending.json").is_file()
    assert not list((ev / "Level3").glob("*.json"))
    for sid, g in ALL8[:-1]:
        _manifest(ev, sid, g)
    svc._scan_pending_events(time.monotonic())
    assert name in svc._pending_plots                     # 7/8: keep waiting
    _manifest(ev, *ALL8[-1])
    svc._scan_pending_events(time.monotonic())
    l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
    assert l3["products"]["status"] == "complete"
    assert l3["schema_version"] == 2 and l3["c2"]["sample_period_us"] == 524.288
    assert plots == [name] and svc._counters["events_complete"] == 1
    assert not (ev / "Level2" / "products_pending.json").exists()
    mon = svc._mon_counters()
    assert mon["cutout_requests_sent"] == 8


@asyncio_test
async def test_cutout_event_incomplete_is_kept_and_late_recorded(
    tmp_path: Path, plots,
) -> None:
    svc, bc = _svc(tmp_path)
    await svc._on_batch(_batch(61335.0, 3), peer_repr="x")
    name = svc._last_event_name
    ev = tmp_path / "events" / name
    for sid, g in ALL8[:6]:
        _manifest(ev, sid, g)
    req = bc.sent[0][0]
    svc._cutout_tracker.on_status(wire.CutoutStatus(
        name, req.request_id, wire.CUTOUT_STATE_FAILED, 13, 0,
        reason=wire.CUTOUT_REASON_TOO_LATE))
    later = time.monotonic() + 121.0
    svc._scan_pending_events(later)
    assert ev.is_dir()                                     # never deleted
    l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
    p = l3["products"]
    assert p["status"] == "incomplete"
    assert p["missing"] == {"s13g0": "failed:too_late", "s13g1": "no_status"}
    assert svc._counters["events_incomplete"] == 1
    assert svc._counters["halves_missing"] == 2
    assert svc._counters["events_incomplete_discarded"] == 0
    _manifest(ev, 13, 1)
    svc._scan_pending_events(later + 5)
    l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
    assert [x["half"] for x in l3["products"]["late"]] == ["s13g1"]
    assert "s13g1" not in l3["products"]["missing"]
    assert svc._counters["events_finalised_late"] == 1


@asyncio_test
async def test_restart_recovers_pending_event(tmp_path: Path, plots) -> None:
    svc, bc = _svc(tmp_path)
    await svc._on_batch(_batch(61335.0, 3), peer_repr="x")
    name = svc._last_event_name
    ev = tmp_path / "events" / name
    svc2, bc2 = _svc(tmp_path)                    # "restarted" C2
    assert svc2._recover_pending_cutout_events() == 1
    assert name in svc2._pending_plots
    assert len(bc2.sent) == 1                     # still inside the deadline
    for sid, g in ALL8:
        _manifest(ev, sid, g)
    svc2._scan_pending_events(time.monotonic())
    l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
    assert l3["products"]["status"] == "complete"
    assert l3["c2"]["snr_max"] == pytest.approx(20.2)
    assert svc2._counters["events_recovered"] == 1


@asyncio_test
async def test_1x_archive_policy_keeps_partial_cube_events(
    tmp_path: Path, plots,
) -> None:
    svc, bc = _svc(tmp_path, cutout=False, policy="archive")
    await svc._on_batch(_batch(61335.0, 2, period=1048.576), peer_repr="x")
    name = svc._last_event_name
    assert len(bc.cube_calls) == 1 and bc.sent == []
    cubes = tmp_path / "events" / name / "cubes"
    cubes.mkdir(parents=True)
    for sid, g in ALL8[:5]:
        (cubes / f"cube_s{sid}_g{g}_1004.npz").write_bytes(b"x")
    svc._scan_pending_events(time.monotonic() + svc._config.plot_cube_wait_s + 1)
    ev = tmp_path / "events" / name
    assert ev.is_dir()
    l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
    assert l3["products"]["kind"] == "cubes"
    assert l3["products"]["status"] == "incomplete"
    assert len(l3["products"]["missing"]) == 3
    assert l3["schema_version"] == 1 and "sample_period_us" not in l3["c2"]
    assert svc._counters["events_incomplete_discarded"] == 0


@asyncio_test
async def test_1x_default_still_discards(tmp_path: Path, plots) -> None:
    svc, bc = _svc(tmp_path, cutout=False)
    await svc._on_batch(_batch(61335.0, 2, period=1048.576), peer_repr="x")
    name = svc._last_event_name
    (tmp_path / "events" / name / "cubes").mkdir(parents=True)
    svc._scan_pending_events(time.monotonic() + svc._config.plot_cube_wait_s + 1)
    assert not (tmp_path / "events" / name).exists()
    assert svc._counters["events_incomplete_discarded"] == 1


@asyncio_test
async def test_dedupe_flag_blocks_fall_through_duplicate(
    tmp_path: Path, plots,
) -> None:
    for dedupe, n_dumps in ((False, 2), (True, 1)):
        root = tmp_path / str(dedupe)
        root.mkdir()
        svc, bc = _svc(root, cutout=False, dedupe=dedupe)
        b1 = _batch(61335.0, 2, period=1048.576)
        await svc._on_batch(b1, peer_repr="x")
        # the same cluster grows; 'bright' is in holdoff, 'galactic' matches
        b2 = _batch(61335.0, 3, period=1048.576)
        await svc._on_batch(b2, peer_repr="x")
        assert len(bc.cube_calls) == n_dumps, dedupe
        if dedupe:
            assert svc._counters["dumps_dedup_suppressed"] == 1
