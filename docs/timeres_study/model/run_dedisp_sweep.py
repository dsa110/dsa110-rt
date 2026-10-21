"""How lossy is two-stage (sub-band) dedispersion at T=262 us, native channels?

-> results/dedisp_sweep.npz / .json
"""
import json
import time

import numpy as np

from configs import one_stage, two_stage_cfg
from emulate import mean_recovered
from physics import T_NATIVE_US

T = 262.144
W_US = [100.0, 250.0, 500.0, 1000.0]
rows = []


def run(tag, cfg, dms, dm_hi):
    res = {}
    for dm in dms:
        res[dm] = [mean_recovered(cfg, [dm], w, n_draw=20, seed=5)[0] for w in W_US]
    rows.append({"tag": tag, "dm_hi": dm_hi, "n_fine": int(len(cfg.fine_dm)),
                 "res": {str(k): v for k, v in res.items()}})
    print(f"{tag:34s} hi={dm_hi:5.0f} nf={len(cfg.fine_dm):5d} " +
          "  ".join(f"DM{int(k)}:" + "/".join(f"{x:.3f}" for x in v) for k, v in res.items()),
          flush=True)


t0 = time.time()
for dm_hi, dms in ((1600.0, (300.0, 800.0, 1500.0)), (3100.0, (1500.0, 2500.0))):
    run("1st q=native (ideal)", one_stage(T, 1, dm_hi=dm_hi, q_native=True), dms, dm_hi)
    run("1st q=T (brute force)", one_stage(T, 1, dm_hi=dm_hi, q_native=False), dms, dm_hi)
    for nc, ns in ((8, 4), (16, 4), (16, 8), (32, 8), (16, 16), (32, 16)):
        for q2 in (T, T / 2):
            c = two_stage_cfg(T, 1, nc, ns, dm_hi=dm_hi, q1_native=True)
            c.q2_us = q2
            run(f"2st {nc}x{ns} q1=nat q2={'T' if q2 == T else 'T/2'}", c, dms, dm_hi)
    print(f"[{time.time()-t0:.0f}s]", flush=True)
json.dump({"W_us": W_US, "T_us": T, "rows": rows}, open("results/dedisp_sweep.json", "w"), indent=1)
