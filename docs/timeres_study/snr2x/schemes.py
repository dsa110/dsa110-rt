"""The schemes compared: what is deployed, the proposed 2x (report v4, roadmap step 1), and a
single-shot Levin search at 0.26 ms / 244 kHz."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from snr_model import OneStage, Scheme

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list  # noqa: E402

DEPLOYED = "/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_dmmax1500_tol1.265_csf8_explicit-v2.npz"
T1, T2, T4 = 1048.576, 524.288, 262.144
F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)


def current():
    """Deployed 2026-09-23: DM 100-1500, 8 coarse x 25 fine, T 1.05 ms, 244 kHz (csf 8),
    n_sub 4, stage-1 rounded native -> 1.05 ms, merged stage-2 rounding at 1.05 ms."""
    z = np.load(DEPLOYED, allow_pickle=True)
    meta = json.loads(str(z["metadata"]))
    T = float(meta["t_int_search_us"])
    assert abs(T - T1) < 1e-6 and int(meta["chan_sum_factor"]) == 8, meta
    tier = dict(fine=np.asarray(z["fine_dm"], float), coarse=np.asarray(z["coarse_dm"], float),
                f2c=np.asarray(z["fine_to_coarse"], int), q1=T, q2=T)
    return Scheme("current (deployed)", csf=8, n_sub=4, tiers=[tier])


def _tier(lo, hi, T, q1, n_sub, csf=2, tol=1.25):
    fine = gen_dm_list(lo, hi, T, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol)
    fine = fine[fine <= hi + 1e-9]
    n_c = max(1, math.ceil(0.0206 * (hi - lo) * (T1 / T) / n_sub))   # same density as plan6.py
    edges = np.linspace(lo, hi, n_c + 1)
    coarse = 0.5 * (edges[:-1] + edges[1:])
    f2c = np.clip(np.searchsorted(edges, fine, side="right") - 1, 0, n_c - 1)
    return dict(fine=fine, coarse=coarse, f2c=f2c, q1=q1, q2=T)


def proposed(q1=T4, n_sub=2, tol=1.25):
    """2x: 0.52 ms for DM 50-1600, 1.05 ms for 1600-3000; 61 kHz (csf 2); n_sub 2; stage-1 at
    q1 (0.26 ms = fast GEMM at t_int 8, or 0.52 ms = t_int 16); one stage-2 rounding at T."""
    tiers = [_tier(50.0, 1600.0, T2, q1, n_sub, tol=tol), _tier(1600.0, 3000.0, T1, min(q1, T1), n_sub, tol=tol)]
    lab = f"proposed 2x (stage-1 at {q1 / 1e3:.2f} ms)"
    return Scheme(lab, csf=2, n_sub=n_sub, tiers=tiers)


def gen_dmtrials(dm1, dm2, tol=1.25, nu_ghz=1.405, nchan=768, tsamp_us=262.144, b_mhz=0.24414):
    """Levin/Cordes step recursion, verbatim from the legacy single-stage generator
    (/media/ubuntu/ssd/vikram/scratch/gen_dmtrials.py, via _inspect/sensitivity/gen_legacy_grid.py)."""
    n2 = float(nchan) ** 2.0
    alp = 1.0 / (16.0 + n2)
    bet = tsamp_us ** 2.0
    trials = [float(dm1)]
    dm_prev, dm = float(dm1), 0.0
    while dm < dm2:
        dm = (n2 * alp * dm_prev
              + np.sqrt(16.0 * alp * (tol ** 2.0 - n2 * alp) * dm_prev ** 2.0
                        + 16.0 * alp * bet * (tol ** 2.0 - 1.0) * (nu_ghz ** 3.0 / 8.3 / b_mhz) ** 2.0))
        dm_prev = dm
        trials.append(float(dm))
    return np.asarray(trials)


def single_shot():
    """Single-shot incoherent dedispersion: Levin trials tol 1.25 over DM 80-1500, 244 kHz channels
    (csf 8), 0.262 ms sampling, one integer delay per channel, power-of-2 boxcars 1-256 samples
    (0.26-67 ms, the same longest boxcar as today's 1-64 at 1.05 ms)."""
    return OneStage("single-shot 0.26 ms / 244 kHz", csf=8, fine=gen_dmtrials(80.0, 1500.0), T_us=T4,
                    ladder=(1, 2, 4, 8, 16, 32, 64, 128, 256))


def describe(s):
    out = []
    for t in s.tiers:
        out.append(dict(dm=(float(t["fine"][0]), float(t["fine"][-1])), n_fine=int(len(t["fine"])),
                        n_coarse=int(len(t["coarse"])),
                        coarse_spacing=float(np.median(np.diff(t["coarse"]))) if len(t["coarse"]) > 1 else None,
                        fine_step=(float(np.diff(t["fine"]).min()), float(np.diff(t["fine"]).max())),
                        q1_us=float(t["q1"]), T_us=float(t["q2"])))
    return dict(name=s.name, csf=s.csf, chan_khz=30.517578125 * s.csf, n_sub=s.n_sub, tiers=out)


if __name__ == "__main__":
    for s in (current(), proposed(), proposed(q1=T2)):
        print(json.dumps(describe(s), indent=1))
