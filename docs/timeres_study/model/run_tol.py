"""S/N cost of a coarser DM grid (Levin tol 1.4 vs 1.25) at the 4x low tier -> results/tol_grid.npz."""
import numpy as np

from configs import fine_list, two_stage
from emulate import Cfg, mean_recovered
from physics import T_NATIVE_US

WIDTHS_US = np.array([25, 50, 100, 200, 400, 700, 1000, 2000, 4000.0])
DMS = [300.0, 500.0, 800.0, 1200.0, 1500.0, 2500.0]
T1, T2, T4 = 1048.576, 524.288, 262.144


def ts(T, tol, name):
    fine = fine_list(30, 3100, T, 2, tol=tol)
    coarse, f2c = two_stage(fine, 32)
    return Cfg(name, T, csf=2, stages=2, n_sub=8, q1_us=T_NATIVE_US, q2_us=T,
               fine_dm=fine, coarse_dm=coarse, fine_to_coarse=f2c)


cfgs = {"bts4_t14": ts(T4, 1.4, "2-stage 0.26 ms, tol 1.4"), "bts2_t14": ts(T2, 1.4, "2-stage 0.52 ms, tol 1.4")}
out = {"widths_us": WIDTHS_US, "dms": np.array(DMS)}
for k, c in cfgs.items():
    m = np.zeros((len(DMS), len(WIDTHS_US)))
    for i, dm in enumerate(DMS):
        for j, w in enumerate(WIDTHS_US):
            m[i, j] = mean_recovered(c, [dm], float(w), n_draw=16, seed=701 + i)[0]
    out[k] = m
    print(k, len(c.fine_dm), "W=100us:", " ".join(f"{v:.3f}" for v in m[:, 2]), flush=True)
np.savez("results/tol_grid.npz", **out)
