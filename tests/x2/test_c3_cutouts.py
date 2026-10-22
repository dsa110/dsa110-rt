"""x2 D7: C3 veto metrics on cutouts; REJECT drops raw windows."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dsart.coinc import cube_veto as CV
from dsart.coinc.archive import C1_WINDOW_CSV_FIELDS
from dsart.coinc.wire import CUTOUT_FLAG_CUTOUT, CutoutRequest
from dsart.services import c3 as c3mod
from dsart.x2.cutout import schema as S
from dsart.x2.opoint import opoint_2x
from dsart.x2.plan import build_tiered_plan_for_opoint

N = 32
NT = 128
T0 = 20_000
SIDS = [1, 2, 9, 13]


@lru_cache(maxsize=None)
def _plan():
    return build_tiered_plan_for_opoint(opoint_2x("gridded", 4, 8))


def _event(tmp: Path, name: str, fg: int, *, halves=range(8),
           burst_row: int = 64, burst_pix=(12, 17), amp: float = 40.0) -> Path:
    plan = _plan()
    ev = tmp / name
    d_trig = int(plan.fine_to_coarse[fg])
    for h in halves:
        ax = S.fine_axis_for_half(plan, h, t_ext_native=16)
        n_f = len(ax["fine_dm_pc_cc"])
        rng = np.random.default_rng(100 + h)
        imgs = rng.normal(size=(NT, n_f, N, N)).astype(np.float32)
        # tier-B fines carry each 1.05 ms value on two T_ext rows (SPEC §9)
        tb = np.flatnonzero(ax["t_step_ext"] == 2)
        imgs[1::2, tb] = imgs[0::2, tb]
        hit = np.flatnonzero(ax["fine_global_idx"] == fg)
        if hit.size:
            f = int(hit[0])
            step = int(ax["t_step_ext"][f])
            r0 = burst_row - burst_row % step
            imgs[r0:r0 + step, f, burst_pix[0], burst_pix[1]] += amp
        req = CutoutRequest(
            event_name=name, request_id=1, t_peak_ext=T0 + 64, t0_ext=T0,
            n_t=NT, width_ext=1, dm_peak_pc_cc=float(plan.fine_dm[fg]),
            l_pix=12, m_pix=17, fine_global_peak=fg, coarse_global_peak=d_trig,
            stamp_half_px=0, n_stamp_t=0, flags=CUTOUT_FLAG_CUTOUT)
        fields = S.assemble_cutout(
            axis=ax, images=imgs, n_t=NT,
            sigma_applied=np.ones((NT, n_f), np.float32),
            row_valid=np.ones((NT, len(ax["coarse_owned"])), bool),
            input_valid_frac=np.ones(n_f, np.float32), request=req,
            identity=dict(search_node_id=SIDS[h // 2], gpu_half=h % 2,
                          half_index=h, plan_id=plan.plan_id),
            clock=dict(t_ext_us=524.288, native_per_ext=16),
            geometry=dict(n_grid=N))
        S.write_cutout(ev / "cutouts", fields)
    (ev / "Level2").mkdir(parents=True, exist_ok=True)
    owner = int(plan.owner_half[d_trig])
    row = {k: "" for k in C1_WINDOW_CSV_FIELDS}
    row.update(dict(mjd=61335.0, event_specnum=T0 + 64, snr=30.0,
                    dm_pc_cc=float(plan.fine_dm[fg]), dm_idx_global=fg,
                    fine_dm_idx=int(plan.fine_local_idx[fg]), l_pix=12,
                    m_pix=17, width_samples=1, kernel_id="unit:d1:b1",
                    search_node_id=SIDS[owner // 2], gpu_half=owner % 2))
    with open(ev / "Level2" / f"C1_window_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(C1_WINDOW_CSV_FIELDS))
        w.writeheader()
        w.writerow(row)
    return ev


def _tier_fine(tier: int) -> int:
    plan = _plan()
    cand = np.flatnonzero(plan.fine_tier == tier)
    d = int(plan.fine_to_coarse[cand[0]]) + 1          # an interior bucket
    sl = plan.fines_of(d)
    return int(sl.start + (sl.stop - sl.start) // 2)


def test_real_burst_keeps_tier_a(tmp_path: Path) -> None:
    ev = _event(tmp_path, "261022aaaa", _tier_fine(0))
    m = CV.compute_metrics(ev, "261022aaaa")
    assert m.ok and m.source == "cutouts" and m.t_ext_us == 524.288
    assert m.t_shift == 0 and m.dm_shift_trials == 0 and m.apex_mismatch == 0
    assert m.tz_trig > 10 and m.img_off_apex < 1.0
    assert m.n_cubes == _plan().n_coarse
    d = CV.decide(m, is_injection=False)
    assert d.keep, d


def test_tier_b_rows_in_text_units(tmp_path: Path) -> None:
    fg = _tier_fine(1)
    ev = _event(tmp_path, "261022bbbb", fg, burst_row=64)
    m = CV.compute_metrics(ev, "261022bbbb")
    assert m.ok and m.t_apex % 2 == 0 and m.t_apex == 64
    assert m.apex_mismatch == 0 and m.img_off_apex < 1.0


def test_missing_owner_half_fails_open(tmp_path: Path) -> None:
    fg = _tier_fine(0)
    owner = int(_plan().owner_half[_plan().fine_to_coarse[fg]])
    ev = _event(tmp_path, "261022cccc", fg,
                halves=[h for h in range(8) if h != owner])
    m = CV.compute_metrics(ev, "261022cccc")
    assert not m.ok and m.reason == "trigger cutout absent"
    assert CV.decide(m, is_injection=False).keep


def test_r2_in_ms(tmp_path: Path) -> None:
    m = CV.CubeMetrics(ok=True, t_shift=50, t_ext_us=524.288, tz_trig=20.0,
                       tz_apex=20.0, imgz_apex=20.0, source="cutouts")
    # 50 T_ext = 26.2 ms: under 31.5 ms (rule in ms) but > 30 samples
    assert "R2_time_shift" in CV.decide(m, is_injection=False).rules_fired
    th = CV.CubeVetoThresholds(r2_t_shift_ms=31.5)
    assert "R2_time_shift" not in CV.decide(
        m, is_injection=False, thresholds=th).rules_fired
    m70 = CV.CubeMetrics(ok=True, t_shift=70, t_ext_us=524.288, tz_trig=20.0,
                         tz_apex=20.0, imgz_apex=20.0)
    assert "R2_time_shift" in CV.decide(
        m70, is_injection=False, thresholds=th).rules_fired
    # 1x metrics (t_ext_us 0) keep the samples rule even with ms configured
    m1 = CV.CubeMetrics(ok=True, t_shift=31, tz_trig=20.0, tz_apex=20.0,
                        imgz_apex=20.0)
    assert "R2_time_shift" in CV.decide(
        m1, is_injection=False, thresholds=th).rules_fired


class _Store:
    def get_dict(self, key: str) -> Optional[Any]:
        return None

    def put_dict(self, key, value) -> None:
        pass


def test_reject_deletes_raw_keeps_cutouts(tmp_path: Path) -> None:
    cfg = c3mod.C3Config(
        archive_root=tmp_path / "candidates",
        rejected_root=tmp_path / "rejected",
        state_path=tmp_path / "state.json", fired_injection_log=None,
        corr_nodes={}, flag_only=False)
    svc = c3mod.C3Service(cfg, mon_store=_Store())
    ev = cfg.archive_root / "261022dddd"
    (ev / "raw" / "raw_s1_g0").mkdir(parents=True)
    (ev / "raw" / "raw_s1_g0" / "raw_d000.npz").write_bytes(b"x" * 64)
    (ev / "cutouts").mkdir(parents=True)
    (ev / "cutouts" / "cutout_s1_g0.npz").write_bytes(b"c")
    (ev / "Level3").mkdir()
    out = svc._do_reject("261022dddd", ev,
                         CV.VetoDecision(keep=False, rules_fired=("R10",)),
                         flag_only=False)
    assert out["reject"]["raw_files_deleted"] == 1
    assert not (ev / "raw").exists()
    assert (ev / "cutouts" / "cutout_s1_g0.npz").is_file()
