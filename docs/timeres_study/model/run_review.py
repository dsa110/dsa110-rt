"""Review run (2026-10-21, after user questions): DM ~500, intermediate channel
widths (csf 4), and one-shot vs two-stage on matched inputs -> results/review_grid.npz"""
import json
import time

import numpy as np

from configs import DEPLOYED_PLAN, legacy, one_stage, two_stage_cfg
from emulate import Cfg, mean_recovered
from physics import T_NATIVE_US

WIDTHS_US = np.array([50, 100, 150, 250, 400, 600, 1000, 2000.0])
DMS = [300.0, 500.0, 800.0, 1500.0]
T1, T4 = 1048.576, 262.144
z = np.load(DEPLOYED_PLAN, allow_pickle=True)


def deployed():
    return Cfg("today: 2-stage 8x4, 244 kHz, stage-1 at T", T1, csf=8, stages=2, n_sub=4,
               fine_dm=z["fine_dm"], coarse_dm=z["coarse_dm"], fine_to_coarse=z["fine_to_coarse"])


def oneshot_today_grid(T, csf, name):
    # one shot on corner-turned data: shifts can only be whole output samples
    return Cfg(name, T, csf=csf, stages=1, q1_us=T, fine_dm=z["fine_dm"])


cfgs = {
    "today": deployed(),
    "legacy": legacy(),
    "os_T1_c8": oneshot_today_grid(T1, 8, "1-shot @1.05 ms, 244 kHz (today's DM grid)"),
    "ts_T1_c8": two_stage_cfg(T1, 8, 16, 8, dm_hi=1600, name="2-stage 16x8 @1.05 ms, 244 kHz, native stage-1"),
    "ts_T1_c1": two_stage_cfg(T1, 1, 16, 8, dm_hi=1600, name="2-stage 16x8 @1.05 ms, 30.5 kHz, native stage-1"),
    "os_T4_c8": one_stage(T4, 8, dm_hi=1600, q_native=False, name="1-shot @0.26 ms, 244 kHz"),
    "os_T4_c4": one_stage(T4, 4, dm_hi=1600, q_native=False, name="1-shot @0.26 ms, 122 kHz"),
    "os_T4_c2": one_stage(T4, 2, dm_hi=1600, q_native=False, name="1-shot @0.26 ms, 61 kHz"),
    "ts_T4_c8": two_stage_cfg(T4, 8, 16, 8, dm_hi=1600, name="2-stage @0.26 ms, 244 kHz"),
    "ts_T4_c4": two_stage_cfg(T4, 4, 16, 8, dm_hi=1600, name="2-stage @0.26 ms, 122 kHz"),
    "ts_T4_c2": two_stage_cfg(T4, 2, 16, 8, dm_hi=1600, name="2-stage @0.26 ms, 61 kHz"),
    "ts_T4_c1": two_stage_cfg(T4, 1, 16, 8, dm_hi=1600, name="2-stage @0.26 ms, 30.5 kHz"),
}
for k in ("ts_T4_c8", "ts_T4_c4", "ts_T4_c2", "ts_T4_c1"):     # stage-2 at T/2 variants
    c = two_stage_cfg(T4, cfgs[k].csf, 16, 8, dm_hi=1600, name=cfgs[k].name + ", stage-2 T/2")
    c.q2_us = T4 / 2
    cfgs[k + "_h"] = c

out = {"widths_us": WIDTHS_US, "dms": np.array(DMS)}
meta = {}
t0 = time.time()
for key, cfg in cfgs.items():
    m = np.zeros((len(DMS), len(WIDTHS_US)))
    for i, dm in enumerate(DMS):
        for j, w in enumerate(WIDTHS_US):
            m[i, j] = mean_recovered(cfg, [dm], float(w), n_draw=20, seed=301 + i)[0]
    out[key] = m
    meta[key] = {"name": cfg.name, "n_fine": int(len(cfg.fine_dm))}
    i5 = DMS.index(500.0)
    print(f"{key:12s} nf={len(cfg.fine_dm):5d}  DM500 W=50/100/250/1000us: "
          + " ".join(f"{m[i5, j]:.3f}" for j in (0, 1, 3, 6)) + f"   [{time.time()-t0:.0f}s]", flush=True)
np.savez("results/review_grid.npz", **out)
json.dump(meta, open("results/review_grid_meta.json", "w"), indent=1)
