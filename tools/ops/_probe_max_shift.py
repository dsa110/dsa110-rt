"""Probe production DM plan: print max_shift, min/max shift per chgroup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.fine_dm.combiner import compute_time_shift_search
from dsart.services.search_compute import _dm_grids_from_npz  # type: ignore

# Production op-point: t_int_search_us=1048.576 (matches corr t_int_fast
# at 32x native integration, per configs/dsart_search_rt.yaml line 108).
T_INT_SEARCH_US_PROD = 1048.576


def main():
    plan_path = Path("/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz")
    cdm, fdm, ftc = _dm_grids_from_npz(plan_path, n_coarse=8)
    # Production n_fdm_in_cube = 34 (M7.2 op-point per dsart_search_rt.yaml).
    fdm = fdm[:34]
    ftc = ftc[:34]
    print(f"coarse_dm: n={len(cdm)} max={cdm.max():.1f} pc/cm3")
    print(f"fine_dm:   n={len(fdm)} max={fdm.max():.1f} pc/cm3")
    print(f"t_int_search_us = {T_INT_SEARCH_US_PROD} (production op-point)")
    print()

    tab = compute_time_shift_search(
        coarse_dm_pc_cm3=cdm,
        fine_dm_pc_cm3=fdm,
        fine_to_coarse=ftc,
        t_int_search_us=T_INT_SEARCH_US_PROD,
    )
    s = tab.shifts
    print(f"shifts shape: {s.shape}  (N_fdm x N_chgroup)")
    print(f"shifts dtype: {s.dtype}")
    print(f"shifts min={s.min()}, max={s.max()}, |max|={abs(s).max()}, mean={s.mean():.2f}")
    print()
    print("Per-chgroup max shift across all fdms:")
    for g in range(s.shape[1]):
        print(f"  chgroup {g:2d}: max={s[:, g].max():3d}  min={s[:, g].min():3d}")
    print()
    print("Per-fdm max shift across all chgroups:")
    for f in range(s.shape[0]):
        print(f"  fdm {f:2d}: max={s[f].max():3d}")
    print()
    max_pos = int(s.max(initial=0))
    max_neg = int((-s).max(initial=0))
    max_abs = max(max_pos, max_neg)
    print()
    print(f"max(+shift) = {max_pos} samples  (UV samples needed BEFORE t=0)")
    print(f"max(-shift) = {max_neg} samples  (UV samples needed AFTER t=T_det)")
    print(f"MAX |shift| = {max_abs} samples")
    pow2 = 1
    while pow2 < max_abs:
        pow2 <<= 1
    print(f"Rounded up to pow-2: {pow2} samples")
    print()
    print("== Buffer sizing implications ==")
    print(f"  UV span needed per cube: T_search + (max_pos + max_neg) "
          f"= T_search + {max_pos + max_neg}")
    print(f"  Symmetric pow-2 dedisp_overlap = {pow2} (each side)")
    print(f"  → UV buffer per chgroup: T_search + 2*{pow2} = "
          f"T_search + {2 * pow2}")


if __name__ == "__main__":
    main()
