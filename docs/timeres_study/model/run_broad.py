"""Broad comparison of dedispersion architectures across the DM range
-> results/broad_grid.npz.

Families (q = shift quantum, T = search sample):
  today      deployed 2-stage (8x4, 244 kHz summed before a T-quantised stage-1)
  os1_c8     one-shot, DM-partitioned (multicast) at today's T, 244 kHz
  ts*        two-stage: stage-1 at native precision on the corr node, stage-2 at T or T/2
  tc*        two corner turns: per-channel uv cells to cell owners, one-shot per cell
             (shift precision = CT1 sampling q), regather by DM group for imaging
  zoom       coherent beam from the voltage ring at native 32.8 us (no channel smear)
"""
import json
import sys
import time

import numpy as np

from configs import DEPLOYED_PLAN, fine_list, two_stage
from emulate import Cfg, mean_recovered
from physics import T_NATIVE_US

WIDTHS_US = np.array([25, 50, 100, 200, 400, 700, 1000, 2000, 4000.0])
DMS = [300.0, 500.0, 800.0, 1200.0, 1500.0, 2500.0]
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 16
HI = 3100.0
z = np.load(DEPLOYED_PLAN, allow_pickle=True)


def oneshot(T, csf, q, name):
    return Cfg(name, T, csf=csf, stages=1, q1_us=q, fine_dm=fine_list(30, HI, T, csf))


def twostage(T, csf, nc, ns, q2, name):
    fine = fine_list(30, HI, T, csf)
    coarse, f2c = two_stage(fine, nc)
    return Cfg(name, T, csf=csf, stages=2, n_sub=ns, q1_us=T_NATIVE_US, q2_us=q2,
               fine_dm=fine, coarse_dm=coarse, fine_to_coarse=f2c)


T1, T2, T4 = 1048.576, 524.288, 262.144
cfgs = {
    "today": Cfg("today", T1, csf=8, stages=2, n_sub=4, fine_dm=z["fine_dm"],
                 coarse_dm=z["coarse_dm"], fine_to_coarse=z["fine_to_coarse"]),
    "os1_c8": oneshot(T1, 8, T1, "one-shot multicast @1.05 ms, 244 kHz"),
    "ts1_c2": twostage(T1, 2, 32, 8, T1, "2-stage @1.05 ms, native stage-1"),
    "ts2_c2": twostage(T2, 2, 32, 8, T2 / 2, "2-stage @0.52 ms, stage-2 0.26 ms"),
    "tc2_c2": oneshot(T2, 2, T4, "two CTs @0.52 ms (CT1 0.26 ms, 61 kHz)"),
    "tc1_c2": oneshot(T1, 2, T4, "two CTs @1.05 ms (CT1 0.26 ms, 61 kHz)"),
    "ts4_c4": twostage(T4, 4, 32, 8, T4 / 2, "2-stage @0.26 ms, 122 kHz, stage-2 T/2"),
    "ts4_c2": twostage(T4, 2, 32, 8, T4 / 2, "2-stage @0.26 ms, 61 kHz, stage-2 T/2"),
    "tc4_c4": oneshot(T4, 4, T4, "two CTs @0.26 ms (CT1 0.26 ms, 122 kHz)"),
    "tc4_c2": oneshot(T4, 2, T4, "two CTs @0.26 ms (CT1 0.26 ms, 61 kHz)"),
    "tc4_c2h": oneshot(T4, 2, T4 / 2, "two CTs @0.26 ms (CT1 0.13 ms, 61 kHz)"),
    "zoom": Cfg("voltage zoom: coherent, 32.8 us", T_NATIVE_US, csf=1, stages=1,
                q1_us=T_NATIVE_US, coherent=True, fine_dm=np.arange(30, HI, 0.05)),
}

out = {"widths_us": WIDTHS_US, "dms": np.array(DMS)}
meta = {}
t0 = time.time()
for key, cfg in cfgs.items():
    m = np.zeros((len(DMS), len(WIDTHS_US)))
    for i, dm in enumerate(DMS):
        if dm > cfg.fine_dm.max() + 5:          # outside this plan's DM range: not searched
            m[i] = np.nan
            continue
        for j, w in enumerate(WIDTHS_US):
            m[i, j] = mean_recovered(cfg, [dm], float(w), n_draw=NDRAW, seed=701 + i)[0]
    out[key] = m
    meta[key] = {"name": cfg.name, "n_fine": int(len(cfg.fine_dm)), "T_us": cfg.T_us}
    print(f"{key:9s} nf={len(cfg.fine_dm):6d} W=100us DM300/800/1500/2500: "
          + " ".join(f"{m[DMS.index(d), 2]:.3f}" for d in (300.0, 800.0, 1500.0, 2500.0))
          + f"  [{time.time()-t0:.0f}s]", flush=True)
np.savez("results/broad_grid.npz", **out)
json.dump(meta, open("results/broad_grid_meta.json", "w"), indent=1)
