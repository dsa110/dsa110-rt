"""Baseline-consistent configurations (NICs stay in their x4 slots) -> results/baseline_grid.npz.

Network-limited, so streams cannot be sent finer than the search sample:
  bts2, bts4   two-stage, native stage-1, stage-2 shifts at T (streams at the tier cadence)
  btc2, btc1   two corner turns with the first hop cut to 0.52 ms / 122 kHz (the only CT1 that
               fits beside the SNAP traffic): one-shot shifts at 0.52 ms
Same DMs / widths / seeds as run_broad.py so the grids combine.
"""
import json
import time

import numpy as np

from configs import fine_list, two_stage
from emulate import Cfg, mean_recovered
from physics import T_NATIVE_US

# identical to run_broad.py (not imported: that module runs its grid at import)
WIDTHS_US = np.array([25, 50, 100, 200, 400, 700, 1000, 2000, 4000.0])
DMS = [300.0, 500.0, 800.0, 1200.0, 1500.0, 2500.0]
NDRAW, HI = 16, 3100.0
T1, T2, T4 = 1048.576, 524.288, 262.144


def oneshot(T, csf, q, name):
    return Cfg(name, T, csf=csf, stages=1, q1_us=q, fine_dm=fine_list(30, HI, T, csf))


def ts_q2T(T, csf, nc, ns, name):
    fine = fine_list(30, HI, T, csf)
    coarse, f2c = two_stage(fine, nc)
    return Cfg(name, T, csf=csf, stages=2, n_sub=ns, q1_us=T_NATIVE_US, q2_us=T,
               fine_dm=fine, coarse_dm=coarse, fine_to_coarse=f2c)


cfgs = {
    "bts2": ts_q2T(T2, 2, 32, 8, "2-stage @0.52 ms, stage-2 at T"),
    "bts4": ts_q2T(T4, 2, 32, 8, "2-stage @0.26 ms, stage-2 at T"),
    "btc2": oneshot(T2, 4, T2, "two CTs @0.52 ms (CT1 0.52 ms, 122 kHz)"),
    "btc1": oneshot(T1, 4, T2, "two CTs @1.05 ms (CT1 0.52 ms, 122 kHz)"),
}
out = {"widths_us": WIDTHS_US, "dms": np.array(DMS)}
meta = {}
t0 = time.time()
for key, cfg in cfgs.items():
    m = np.zeros((len(DMS), len(WIDTHS_US)))
    for i, dm in enumerate(DMS):
        for j, w in enumerate(WIDTHS_US):
            m[i, j] = mean_recovered(cfg, [dm], float(w), n_draw=NDRAW, seed=701 + i)[0]
    out[key] = m
    meta[key] = {"name": cfg.name, "n_fine": int(len(cfg.fine_dm))}
    print(f"{key:6s} W=100us DM300/800/1500/2500: "
          + " ".join(f"{m[DMS.index(d), 2]:.3f}" for d in (300.0, 800.0, 1500.0, 2500.0))
          + f"  [{time.time()-t0:.0f}s]", flush=True)
np.savez("results/baseline_grid.npz", **out)
json.dump(meta, open("results/baseline_grid_meta.json", "w"), indent=1)
