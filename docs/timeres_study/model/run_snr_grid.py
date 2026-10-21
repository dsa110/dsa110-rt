"""Recovered-S/N grid for all configurations -> results/snr_grid.npz."""
import json
import sys
import time

import numpy as np

from configs import deployed, legacy, one_stage, two_stage_cfg
from emulate import mean_recovered

WIDTHS_US = np.array([50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 4000, 8000.0])
DMS = [300.0, 800.0, 1500.0]
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 16

cfgs = {
    "deployed": deployed(),
    "legacy": legacy(),
    "T262_c8_1st": one_stage(262.144, 8, name="4x, 244 kHz, ideal 1-stage"),
    "T262_c8_2st": two_stage_cfg(262.144, 8, 16, 8, name="4x, 244 kHz, 2-stage 16x8"),
    "T262_c2_2st": two_stage_cfg(262.144, 2, 16, 8, name="4x, 61 kHz, 2-stage 16x8"),
    "T262_c1_2st": two_stage_cfg(262.144, 1, 16, 8, name="4x, 30.5 kHz, 2-stage 16x8"),
    "T262_c1_2st_qT": two_stage_cfg(262.144, 1, 16, 8, q1_native=False,
                                    name="4x, 30.5 kHz, 2-stage 16x8, stage-1 at T"),
    "T262_c1_1st": one_stage(262.144, 1, name="4x, 30.5 kHz, ideal 1-stage"),
    "T131_c1_2st": two_stage_cfg(131.072, 1, 32, 8, name="8x, 30.5 kHz, 2-stage 32x8"),
    "T66_c1_1st": one_stage(65.536, 1, name="16x, 30.5 kHz, ideal 1-stage"),
}

out = {"widths_us": WIDTHS_US, "dms": np.array(DMS)}
meta = {}
t0 = time.time()
for key, cfg in cfgs.items():
    mean = np.zeros((len(DMS), len(WIDTHS_US)))
    p10 = np.zeros_like(mean)
    for i, dm in enumerate(DMS):
        for j, w in enumerate(WIDTHS_US):
            mean[i, j], p10[i, j] = mean_recovered(cfg, [dm], float(w), n_draw=NDRAW, seed=17 + i)
    out[key + "_mean"] = mean
    out[key + "_p10"] = p10
    meta[key] = {"name": cfg.name, "n_fine": int(len(cfg.fine_dm)), "T_us": cfg.T_us,
                 "csf": cfg.csf, "stages": cfg.stages, "n_sub": cfg.n_sub,
                 "n_coarse": int(len(cfg.coarse_dm)) if cfg.coarse_dm is not None else 0}
    print(f"{key:18s} n_fine={len(cfg.fine_dm):5d}  W=0.1/0.5/1 ms @DM1500: "
          f"{mean[2, 2]:.3f} {mean[2, 6]:.3f} {mean[2, 8]:.3f}   [{time.time()-t0:.0f}s]", flush=True)
np.savez("results/snr_grid.npz", **out)
json.dump(meta, open("results/snr_grid_meta.json", "w"), indent=1)
