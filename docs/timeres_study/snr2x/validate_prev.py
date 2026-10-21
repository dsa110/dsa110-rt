"""Explain the 0.758 (this model) vs 0.798 (2026-09-23 study) gap for the current scheme at 1 ms.

The earlier `_profile_snrs` convolves the intrinsic pulse only when W > T (round(W/T) samples),
so a 1 ms burst at T = 1.05 ms was a zero-width arrival in every channel, while its `perfect`
reference kept the 0.95-sample width; and it had no native-channel smear.  Re-run this model
with those two simplifications to check they account for the difference.
"""
import numpy as np

import snr_model as M
from schemes import current

W = 1000.0
cur = current()
T = cur.tiers[0]["q2"]
box = M._tophat_ft(W)
H0 = np.fft.rfft(np.eye(1, M.NGRID, M.NGRID // 2).ravel())
perfect = M.snr_binned(H0, box, T, W, "tophat")
delta = np.ones(M.NGRID // 2 + 1)
dms = np.linspace(102, 1498, 120)


def mean_total(kern, no_smear):
    saved = M.SMEAR_COEF.copy()
    if no_smear:
        M.SMEAR_COEF[:] = 0.0
    vals = []
    for dm in dms:
        best = 0.0
        for j in np.argsort(np.abs(cur.trial_dm - dm))[:9]:
            ti, fi, _ = cur.trials[j]
            H = M.histogram(M.A_TRUE * dm + cur.applied_us(ti, fi), dm)
            best = max(best, M.snr_binned(H, kern, T, W, "tophat"))
        vals.append(best)
    M.SMEAR_COEF[:] = saved
    return np.mean(vals) / perfect


print(f"this model (1 ms top-hat, native smear):           {mean_total(box, False):.3f}")
print(f"no native smear:                                    {mean_total(box, True):.3f}")
print(f"no native smear + zero-width pulse (as 09-23 study): {mean_total(delta, True):.3f}   (study: 0.798)")
