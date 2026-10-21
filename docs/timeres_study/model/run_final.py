"""Final recovered-S/N grid used by the report figures -> results/final_grid.npz."""
import json
import time

import numpy as np

from configs import deployed, legacy, one_stage, two_stage_cfg

WIDTHS_US = np.array([25, 50, 75, 100, 150, 250, 400, 600, 1000, 1500, 2500, 4000, 8000, 16000.0])
DMS = [300.0, 800.0, 1500.0, 2500.0]
T4 = 262.144


def ts(T, csf, nc, ns, hi, q2_half, name):
    c = two_stage_cfg(T, csf, nc, ns, dm_hi=hi, q1_native=True, name=name)
    if q2_half:
        c.q2_us = T / 2
    return c


cfgs = {
    "deployed": deployed(),
    "legacy": legacy(),
    "x1_c1": ts(1048.576, 1, 16, 8, 3100, False, "1.05 ms, 30.5 kHz, 16x8"),
    "x2_c1": ts(524.288, 1, 16, 8, 3100, True, "2x: 0.52 ms, 30.5 kHz, 16x8"),
    "x4_c8": ts(T4, 8, 32, 8, 3100, True, "4x: 0.26 ms, 244 kHz, 32x8"),
    "x4_c2": ts(T4, 2, 32, 8, 3100, True, "4x: 0.26 ms, 61 kHz, 32x8"),
    "x4_c1": ts(T4, 1, 32, 8, 3100, True, "4x: 0.26 ms, 30.5 kHz, 32x8"),
    "x4_c1_ideal": one_stage(T4, 1, dm_hi=3100, q_native=True, name="4x ideal 1-stage"),
    "x8_c1": ts(131.072, 1, 32, 8, 3100, True, "8x: 0.13 ms, 30.5 kHz, 32x8"),
    "x16_c1_ideal": one_stage(65.536, 1, dm_hi=3100, q_native=True, name="16x ideal 1-stage"),
}

out = {"widths_us": WIDTHS_US, "dms": np.array(DMS)}
meta = {}
t0 = time.time()
for key, cfg in cfgs.items():
    mean = np.zeros((len(DMS), len(WIDTHS_US)))
    for i, dm in enumerate(DMS):
        if key == "deployed" and dm > 1500:
            mean[i] = np.nan           # outside the deployed DM range
            continue
        for j, w in enumerate(WIDTHS_US):
            mean[i, j] = __import__("emulate").mean_recovered(cfg, [dm], float(w), n_draw=16,
                                                              seed=101 + i)[0]
    out[key] = mean
    meta[key] = {"name": cfg.name, "n_fine": int(len(cfg.fine_dm)), "T_us": cfg.T_us,
                 "csf": cfg.csf, "stages": cfg.stages}
    print(f"{key:14s} nf={len(cfg.fine_dm):5d} DM1500 W=.1/.25/1ms: "
          f"{mean[2, 3]:.3f} {mean[2, 5]:.3f} {mean[2, 8]:.3f} [{time.time()-t0:.0f}s]", flush=True)
np.savez("results/final_grid.npz", **out)
json.dump(meta, open("results/final_grid_meta.json", "w"), indent=1)
