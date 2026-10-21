"""Pipeline configurations compared in the study."""
from __future__ import annotations

import numpy as np

from emulate import Cfg
from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, T_NATIVE_US, gen_dm_list

DEPLOYED_PLAN = ("/home/ubuntu/data/dm_plans/"
                 "dm_plan_N8_dmmin100_dmmax1500_tol1.265_csf8_explicit-v2.npz")
F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)


def fine_list(dm_lo, dm_hi, T_us, csf, tol=1.25):
    dnu = DNU_NATIVE_MHZ * csf
    return gen_dm_list(dm_lo, dm_hi, T_us, dnu, int(round(6144 / csf)), F_CTR, tol=tol)


def two_stage(fine, n_coarse):
    """Equal-count contiguous buckets; coarse DM = bucket centre."""
    edges = np.linspace(0, len(fine), n_coarse + 1).round().astype(int)
    f2c = np.zeros(len(fine), dtype=int)
    coarse = np.zeros(n_coarse)
    for b in range(n_coarse):
        f2c[edges[b]:edges[b + 1]] = b
        seg = fine[edges[b]:edges[b + 1]]
        coarse[b] = 0.5 * (seg[0] + seg[-1])
    return coarse, f2c


def deployed():
    z = np.load(DEPLOYED_PLAN, allow_pickle=True)
    return Cfg("deployed: T=1.05 ms, 244 kHz, 8x4 two-stage", 1048.576, csf=8,
               stages=2, n_sub=4, fine_dm=z["fine_dm"], coarse_dm=z["coarse_dm"],
               fine_to_coarse=z["fine_to_coarse"])


def legacy():
    fine = gen_dm_list(30, 3000, 262.144, 0.244140625, 768,
                       F_TOP_MHZ - 0.244140625 * (768 / 2 - 0.5), ti_us=40, tol=1.5)
    return Cfg("legacy dsaX: T=0.26 ms, 244 kHz, 1-stage", 262.144, csf=8,
               stages=1, q1_us=262.144, fine_dm=fine)


def one_stage(T_us, csf, dm_hi=1600.0, tol=1.25, q_native=True, name=None):
    fine = fine_list(30, dm_hi, T_us, csf, tol)
    q = T_NATIVE_US if q_native else T_us
    nm = name or f"1-stage T={T_us/1e3:.3g} ms, {30.5*csf:.0f} kHz"
    return Cfg(nm, T_us, csf=csf, stages=1, q1_us=q, fine_dm=fine)


def two_stage_cfg(T_us, csf, n_coarse, n_sub, dm_hi=1600.0, tol=1.25,
                  q1_native=True, name=None):
    fine = fine_list(30, dm_hi, T_us, csf, tol)
    coarse, f2c = two_stage(fine, n_coarse)
    q1 = T_NATIVE_US if q1_native else T_us
    nm = name or (f"2-stage {n_coarse}x{n_sub} T={T_us/1e3:.3g} ms, "
                  f"{30.5*csf:.0f} kHz, q1={'native' if q1_native else 'T'}")
    return Cfg(nm, T_us, csf=csf, stages=2, n_sub=n_sub, q1_us=q1,
               fine_dm=fine, coarse_dm=coarse, fine_to_coarse=f2c)
