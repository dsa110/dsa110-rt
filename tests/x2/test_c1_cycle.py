"""x2 D9: per-cycle C1 filtering, global fine index, time-filter FWHM."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dsart.coinc import archive as A
from dsart.coinc.units import fwhm_ms, time_filter_of
from dsart.coinc.window import WindowEntry
from dsart.common.contracts import Candidate, CandidateFlags, CubeGeometry
from dsart.services.c1_emit import (
    WideEscapeGovernor,
    candidate_to_c1_row,
    pool_cycle,
)


def _cand(snr, *, w=1, dm=300.0, kid="unit:d1:b1", t=1000):
    return Candidate(l=10.0, m=20.0, dm_fine=dm, dm_idx=2, event_specnum=t,
                     width_samples=w, kernel_id=kid, snr=snr,
                     detector_version="v1.M5", flags=int(CandidateFlags.NONE),
                     search_node_id=0, gpu_half=0)


def _geom(fine):
    return CubeGeometry(
        cube_id=1, specnum_start=1000, sample_period_specnum=8, t_det=384,
        n_grid=256, n_fdm_in_cube=len(fine), sample_period_us=524.288,
        cell_l_rad=1.5e-4, cell_m_rad=1.5e-4, l0_rad=0.0, m0_rad=0.0,
        fine_dm_pc_cc=np.asarray(fine, np.float64), mjd_start=61335.0)


def _cap8(cands):
    return sorted(cands, key=lambda c: -c.snr)[:8]


def test_pool_cycle_single_group_is_identity_wrapper() -> None:
    cs = [_cand(10 + i) for i in range(12)]
    out = pool_cycle([("d3", cs)], _cap8)
    assert out[0][0] == "d3" and out[0][1] == _cap8(cs)


def test_pool_cycle_caps_per_cycle_and_returns_per_cube() -> None:
    groups = [(f"d{d}", [_cand(10 + d + 0.1 * i, t=d * 100 + i) for i in range(6)])
              for d in range(6)]
    per_cube = sum(len(_cap8(list(c))) for _, c in groups)
    pooled = pool_cycle(groups, _cap8)
    assert per_cube == 36                                  # 6 cubes x 6
    assert sum(len(s) for _, s in pooled) == 8             # one cap per cycle
    assert [k for k, _ in pooled] == [k for k, _ in groups]
    for (k, survivors), (_, orig) in zip(pooled, groups):
        assert all(any(c is o for o in orig) for c in survivors)
    # the 8 brightest of the whole cycle survive (d5 and d4 hold them)
    assert [len(s) for _, s in pooled] == [0, 0, 0, 0, 2, 6]


def test_pool_cycle_rejects_invented_candidates() -> None:
    with pytest.raises(ValueError):
        pool_cycle([("a", [_cand(10)]), ("b", [_cand(11)])],
                   lambda cs: [_cand(99)])


def test_governor_sees_the_brightest_escape_of_the_cycle() -> None:
    gov = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    dim = _cand(21.0, w=64)
    bright = _cand(80.0, w=64)
    groups = [("first_dm", [dim]), ("peak_dm", [bright])]
    out = pool_cycle(groups, lambda cs: gov.admit(cs, 0.0)[0])
    assert out[0][1] == [] and out[1][1] == [bright]


def test_dm_idx_global_from_the_plan_map() -> None:
    fine = [300.0, 302.9, 305.8]
    c = _cand(12.0, dm=302.9)
    r1 = candidate_to_c1_row(c, geom=_geom(fine))
    assert r1.dm_idx_global == 2 and r1.fine_dm_idx == 1      # today's value
    r2 = candidate_to_c1_row(c, geom=_geom(fine), fine_global_idx=[117, 118, 119])
    assert r2.dm_idx_global == 118 and r2.fine_dm_idx == 1


def test_time_filter_tokens_and_fwhm() -> None:
    assert time_filter_of("unit:d1:b4") == ("b4", 4)
    assert time_filter_of("unit:d1:q16") == ("q16", 16)
    assert time_filter_of("unit:d1:t2") == ("t2", 2)
    assert time_filter_of("weird") == ("weird", 0)
    assert fwhm_ms("unit:d1:b4", 4, 1048.576) == pytest.approx(4.194304)
    assert fwhm_ms("unit:d1:b4", 4, 524.288) == pytest.approx(2.097152)
    # tier-B q8 reports 16 T_ext samples: FWHM 8.00 tier samples x 1.05 ms
    assert fwhm_ms("unit:d1:q8", 16, 524.288) == pytest.approx(8.0 * 1.048576)
    assert fwhm_ms("unit:d1:q4", 4, 524.288) == pytest.approx(3.67 * 0.524288)


def test_archive_width_ms_is_the_fwhm() -> None:
    e = WindowEntry(mjd=61335.0, snr=12.0, l_rad=0.0, m_rad=0.0, l_pix=0,
                    m_pix=0, dm_pc_cc=300.0, dm_idx_global=5, fine_dm_idx=1,
                    event_specnum=10, width_samples=4, kernel_id="unit:d1:q4",
                    flags=0, search_node_id=1, gpu_half=0, cube_id=1,
                    sample_period_us=524.288)
    row = A.entry_to_csv_row(e, schema=2)
    assert float(row["width_ms"]) == pytest.approx(3.67 * 0.524288)


def test_k_entry_filter_id_only_with_family() -> None:
    dash = Path(__file__).resolve().parents[2] / "tools/dashboard/dsa_monitor"
    sys.path.insert(0, str(dash))
    try:
        import inject_calibration as IC
    finally:
        sys.path.remove(str(dash))

    class _S:
        def __init__(self, docs=None):
            self.docs = dict(docs or {})

        def get_dict(self, k):
            return self.docs.get(k)

        def put_dict(self, k, v):
            self.docs[k] = dict(v)

    e = IC.CalibrationEntry(bucket="dm0500", dm_pc_cm3_rounded=500,
                            width_samples=4, K=1.0, last_fluence_jy_ms=1e-3,
                            last_observed_snr=20.0, last_inj_id="x",
                            last_calibrated_at_unix=1.0, actor="t",
                            filter_id="q4")
    s0 = _S()
    IC.CalibrationStore(s0).put(e)
    assert "filter_id" not in s0.docs["/cnf/inject/snr_calibration/dm0500"]
    s2 = _S({"/cnf/opoint": {"opoint": "2x-gridded-ns2-cint4",
                             "selection": "2x-gridded-ns2-cint4"}})
    IC.CalibrationStore(s2).put(e)
    assert s2.docs["/cnf/inject/snr_calibration/dm0500"]["filter_id"] == "q4"
