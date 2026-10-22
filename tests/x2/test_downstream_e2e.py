"""x2 D13: downstream end to end on CPU, plus the bright replay and 1x config checks.

C1 batch (2x units) -> C2 fires and sends CutoutRequests -> each half's
cutout is built from synthetic images with WS-D's reference
(``assemble_cutout``; stands in for WS-S's producer) and lands with its
manifest -> C2 finalises (complete) -> C3 metrics KEEP -> plots render.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import dsart.services.coincidencer as c2mod
from dsart.coinc import cube_veto as CV
from dsart.coinc import wire
from dsart.coinc.cutout_lifecycle import CutoutConfig
from dsart.services.coincidencer import CoincidencerConfig, CoincidencerService
from dsart.x2.cutout import schema as S
from dsart.x2.opoint import opoint_2x
from dsart.x2.plan import build_tiered_plan_for_opoint

REPO = Path(__file__).resolve().parents[2]
SIDS = (1, 2, 9, 13)
N = 32


@lru_cache(maxsize=None)
def _plan():
    return build_tiered_plan_for_opoint(opoint_2x("gridded", 4, 8))


class _Store:
    def get_dict(self, key):
        return {"enabled": True, "ts": 0.0} if key.endswith("dumps_enabled") else None

    def put_dict(self, key, value):
        pass


class _Bc:
    def __init__(self):
        self.sent = []

    def broadcast(self, **kw):
        raise AssertionError("2x must not send the 64-byte cube trigger")

    def send_to(self, blob, halves):
        halves = [tuple(h) for h in halves]
        self.sent.append((wire.decode_cutout_request(blob), halves))
        return {h: True for h in halves}

    def close(self):
        pass


def test_downstream_round_trip(tmp_path: Path, monkeypatch) -> None:
    plan = _plan()
    pp = tmp_path / "plan.npz"
    plan.to_npz(pp)
    crit = REPO / "configs" / "opoints" / "2x" / "c2_trigger_criteria.yaml"
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1", bind_port=0, window_s=5.0,
        csv_dir_c1=tmp_path / "c1", csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events", trigger_criteria_path=crit,
        name_allocator_offline=True, gal_dm_poll_interval_s=60.0,
        gal_dm_max_los_override=40.0, startup_grace_s=0.0,
        dump_broadcast_hosts={s: f"10.0.0.{s}" for s in SIDS},
        archive_schema=2, opoint="2x-gridded-ns4-cint4",
        cutout=CutoutConfig(enabled=True, plan_path=pp),
        partial_event_policy="archive", plotter_use_process_pool=False,
    )
    rendered = []
    monkeypatch.setattr(c2mod, "enqueue_event",
                        lambda w, name, root, **kw: rendered.append(name))
    bc = _Bc()
    svc = CoincidencerService(config=cfg, mon_store=_Store(), broadcaster=bc)
    # the burst: a tier-A fine DM owned by some half
    d = 5
    sl = plan.fines_of(d)
    fg = sl.start + 3
    dm = float(plan.fine_dm[fg])
    owner = int(plan.owner_half[d])
    hdr = wire.build_header(
        cube_id=1, event_specnum_start=20_000, mjd_start=61335.5,
        sample_period_specnum=8, sample_period_us=524.288, n_grid=N,
        n_fdm_in_cube=int(plan.n_fdm[d]), search_node_id=SIDS[owner // 2],
        gpu_half=owner % 2, n_candidates=1)
    row = wire.C1CandidateRow(
        snr=30.0, l_rad=0.001, m_rad=0.002, l_pix=12, m_pix=17, dm_pc_cc=dm,
        dm_idx_global=fg, fine_dm_idx=3, event_specnum=20_100, width_samples=2,
        kernel_id="unit:d1:b2", flags=0)
    asyncio.run(svc._on_batch(wire.C1Batch(header=hdr, candidates=(row,)),
                              peer_repr="x"))
    assert len(bc.sent) == 1
    req, halves = bc.sent[0]
    assert sorted(halves) == sorted((SIDS[h // 2], h % 2) for h in range(8))
    assert req.fine_global_peak == fg and req.coarse_global_peak == d
    assert req.plan_id == plan.plan_id and req.wants_raw
    name = req.event_name
    ev = cfg.event_archive_root / name

    # every half answers with its cutout (synthetic images, burst on the owner)
    for h in range(8):
        ax = S.fine_axis_for_half(plan, h, t_ext_native=16)
        n_f = len(ax["fine_dm_pc_cc"])
        rng = np.random.default_rng(h)
        imgs = rng.normal(size=(req.n_t, n_f, N, N)).astype(np.float32)
        tb = np.flatnonzero(ax["t_step_ext"] == 2)
        imgs[1::2, tb] = imgs[0::2, tb]
        hit = np.flatnonzero(ax["fine_global_idx"] == fg)
        if hit.size:
            r0 = req.t_peak_ext - req.t0_ext
            imgs[r0:r0 + 2, hit[0], 12, 17] += 25.0
        fields = S.assemble_cutout(
            axis=ax, images=imgs, n_t=req.n_t,
            sigma_applied=np.ones((req.n_t, n_f), np.float32),
            row_valid=np.ones((req.n_t, len(ax["coarse_owned"])), bool),
            input_valid_frac=np.ones(n_f, np.float32), request=req,
            identity=dict(search_node_id=SIDS[h // 2], gpu_half=h % 2,
                          half_index=h, plan_id=plan.plan_id),
            clock=dict(t_ext_us=524.288, native_per_ext=16),
            geometry=dict(n_grid=N))
        S.write_cutout(ev / "cutouts", fields)
    svc._scan_pending_events(time.monotonic())
    l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
    assert l3["products"]["status"] == "complete" and rendered == [name]
    assert l3["c2"]["sample_period_us"] == 524.288

    # C3 on the cutouts
    m = CV.compute_metrics(ev, name)
    assert m.ok and m.source == "cutouts" and m.t_shift == 0
    assert m.dm_shift_trials == 0 and m.tz_trig > 8.0
    assert CV.decide(m, is_injection=False,
                     thresholds=CV.CubeVetoThresholds(r2_t_shift_ms=31.5)).keep

    # the plots render from the cutouts
    from dsart.coinc import plotter as PL
    paths = PL.render_event_plots(PL.PlotJob(event_name=name,
                                             archive_root=cfg.event_archive_root))
    assert len(paths) == 4 and all(p.stat().st_size > 1000 for p in paths)


def test_1x_selection_changes_no_c2_setting() -> None:
    base = REPO / "configs" / "dsart_search_rt.yaml"
    a = CoincidencerConfig.from_yaml(base)
    b = CoincidencerConfig.from_yaml(base, override={"opoint": "1x"})
    import dataclasses
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    assert da.pop("opoint") == "" and db.pop("opoint") == "1x"
    assert da == db


def _tool():
    spec = importlib.util.spec_from_file_location(
        "bright_replay", REPO / "tools" / "c2" / "bright_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bright_replay_mechanics(tmp_path: Path) -> None:
    br = _tool()
    today, x2 = br.load_curves(REPO / "docs/x2/phase0/results/rowrule_bright.json")
    assert today.forward(100.0) == pytest.approx(79.9, abs=0.5)
    assert today.inverse(today.forward(100.0)) == pytest.approx(100.0, abs=0.5)
    assert br.decompress(15.0, 1, today, x2) == 15.0             # linear range
    assert br.decompress(80.0, 1, today, x2) > 95.0                # ~100 true
    assert br.decompress(80.0, 64, today, x2) == 80.0              # wide: 10/sample
    header = ("mjd_peak,snr_max,snr_sum,snr_mean,n_events,n_search_nodes,"
              "n_gpu_halves,dm_median,dm_iqr,dm_min,dm_max,l_median,m_median,"
              "lm_diag_rad,width_median,width_min,width_max,t_span_s,t_start_mjd,"
              "t_end_mjd,kernel_ids_distinct,gal_dm_max_los_pc_cc,"
              "dm_galactic_fraction,trigger_class,trigger\n")
    # a 34.5 sigma narrow galactic burst: today below bright_galactic's 35
    rows = [f"61335.1,34.5,34.5,34.5,1,1,1,150.0,0,150,150,0,0,0,1,1,1,0,"
            f"61335.1,61335.1,unit:d1:b1,400.0,0.375,log_only,\n"]
    p = tmp_path / "c2_20261022_05.csv"
    p.write_text(header + "".join(rows))
    rr = br.read_rows([tmp_path])
    one = br.replay(rr, REPO / "configs/c2_trigger_criteria.yaml", opoint=None,
                    mode="1x", today=today, x2=x2)
    lin = br.replay(rr, REPO / "configs/c2_trigger_criteria.yaml", opoint=None,
                    mode="1x+lin", today=today, x2=x2)
    assert one["classes"].get("bright_galactic", 0) == 0
    assert lin["classes"].get("bright_galactic", 0) == 1
    assert lin["changed_to_dump"][0]["snr_now"] > 35.0
