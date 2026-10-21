"""Checks before trusting the comparison.

1. The generic shift model reproduces the PRODUCTION integer shifts for the deployed plan,
   bit for bit (DMPlan.delay_bins_per_subband, compute_stage2_shifts,
   compute_time_shift_search(merge_coarse_rounding=True, n_sub=4)).
2. Against the 2026-09-23 study (_inspect/sensitivity/dedisp_compare.py): current scheme,
   1 ms top-hat, mean over DM 102-1498 of recovered S/N relative to PERFECT dedispersion at
   the same 1.05 ms sampling = 0.798 there (with the measured sigma_k colour; white here).
3. Zero DM, zero smear: the matched-filter step returns 1.
"""
import numpy as np

import snr_model as M
from schemes import DEPLOYED, current

from dsart.coarse_dm.dm_plan import DMPlan
from dsart.coarse_dm.stage2_shifts import compute_stage2_shifts
from dsart.common.contracts import DmPlan
from dsart.fine_dm.combiner import compute_time_shift_search


def production_applied_us():
    T, n_sub, csf = 1048.576, 4, 8
    plan = DMPlan.from_summed_canonical(DmPlan.from_npz(DEPLOYED), chan_sum_factor=csf)
    z = np.load(DEPLOYED, allow_pickle=True)
    fine, f2c = np.asarray(z["fine_dm"], float), np.asarray(z["fine_to_coarse"], int)
    coarse = np.asarray(plan.dm_pc_cc, float)
    s1 = np.stack([np.asarray(plan.delay_bins_per_subband(g, n_sub)) for g in range(16)])  # (16,48,8)
    s2 = np.stack([np.asarray(compute_stage2_shifts(chgroup=g, coarse_dm_pc_cm3=coarse, t_int_corr_us=T)
                              .shifts_samples).ravel() for g in range(16)])               # (16,8)
    refs = np.stack([plan.subband_ref_freqs_GHz(g, n_sub) for g in range(16)])
    sf = np.asarray(compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine, fine_to_coarse=f2c, t_int_search_us=T,
        merge_coarse_rounding=True, t_int_corr_us=T, n_sub=n_sub, nu_subband_ref_GHz=refs).shifts)
    k = np.arange(384) // csf
    s = k // (48 // n_sub)
    out = np.zeros((len(fine), 16, 384))
    for f in range(len(fine)):
        c = f2c[f]
        out[f] = (-s1[:, k, c] + s2[:, c][:, None] + sf[f][np.arange(16)[:, None] * n_sub + s[None, :]]) * T
    return out


def main():
    cur = current()
    prod = production_applied_us()
    mine = np.stack([cur.applied_us(0, f) for f in range(len(cur.tiers[0]["fine"]))])
    d = np.abs(prod - mine).max()
    print(f"[1] max |production - model| applied shift over {prod.shape[0]} trials x 6144 ch: {d:.3g} us")
    assert d < 1e-6, d                     # identical integers; only float order differs

    W = 1000.0
    ker = M._tophat_ft(W)
    H0 = np.fft.rfft(np.eye(1, M.NGRID, M.NGRID // 2).ravel())       # perfect: one arrival, no smear
    perfect = M.snr_binned(H0, ker, cur.tiers[0]["q2"], W, "tophat")
    vals = []
    for dm in np.linspace(102, 1498, 120):
        vals.append(M.evaluate(cur, dm, [W], shape="tophat")["total"][0])
    print(f"[2] current, 1 ms top-hat, mean(102-1498) total/perfect-at-1.05ms = {np.mean(vals) / perfect:.3f} "
          f"(2026-09-23 study: 0.798 with sigma_k colour)")
    H = M.histogram(np.zeros((16, 384)), 0.0)
    print(f"[3] zero-DM matched filter, W=0.1 / 1 ms: "
          f"{M.snr_matched(H, M._gauss_ft(100.0), 100.0):.4f} / {M.snr_matched(H, M._gauss_ft(1000.0), 1000.0):.4f}")


if __name__ == "__main__":
    main()
