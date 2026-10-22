"""Filter-limited efficiency of each bank: an UNSMEARED pulse (no dispersion residual, no channel
smear) sampled at T with random arrival phase, S/N / ideal vs W/T at 16 points per octave.  Isolates
what the bank itself costs (scalloping between widths, shape mismatch) from the dedispersion chain.
-> results/time_filters_scallop.json
"""
import json
import os

import numpy as np

from tf_common import RES, M

os.nice(10)
T = 524.288
BANKS = ("boxcar", "b2", "b3", "gauss", "b2h", "b3h", "b3x")
SHAPES = ("gauss", "tophat", "exp")
WT = 2.0 ** (np.arange(-2 * 16, 7 * 16 + 1) / 16.0)               # W/T = 0.25 .. 128
prof = np.zeros(M.NGRID)
prof[M.NGRID // 2] = 1.0
H = np.fft.rfft(prof)
res = np.zeros((len(SHAPES), len(BANKS), len(WT)))
for a, sh in enumerate(SHAPES):
    for c, x in enumerate(WT):
        r = M.snr_banks(H, M.kernel_x(x * T, sh), T, x * T, sh, BANKS)
        for b, bk in enumerate(BANKS):
            res[a, b, c] = r[bk]
out = {"T_us": T, "w_over_T": WT.tolist(), "banks": list(BANKS), "shapes": list(SHAPES), "stats": {}}
asym = (WT >= 4) & (WT <= 64)                                     # inside the bank, filter-limited
octv = (WT >= 8) & (WT <= 16)                                     # one octave, asymptotic regime
for a, sh in enumerate(SHAPES):
    for b, bk in enumerate(BANKS):
        y = res[a, b]
        out["stats"].setdefault(sh, {})[bk] = dict(
            mean_4_64=float(y[asym].mean()), min_4_64=float(y[asym].min()),
            at_bank_width_16=float(y[np.argmin(np.abs(WT - 16))]),
            mid_width_11p3=float(y[np.argmin(np.abs(WT - 16 / np.sqrt(2)))]),
            ripple_8_16=float(y[octv].max() - y[octv].min()),
            mean_0p5_2=float(y[(WT >= 0.5) & (WT <= 2)].mean()))
    print(f"\n{sh}: S/N / ideal (unsmeared), W/T 4-64 mean / min, at W=16 T, at W=11.3 T, ripple 8-16, mean 0.5-2")
    for bk in BANKS:
        s = out["stats"][sh][bk]
        print(f"  {bk:6s} {s['mean_4_64']:.4f} / {s['min_4_64']:.4f}   {s['at_bank_width_16']:.4f}  "
              f"{s['mid_width_11p3']:.4f}  {s['ripple_8_16']:.4f}  {s['mean_0p5_2']:.4f}")
out["snr"] = res.round(5).tolist()
json.dump(out, open(os.path.join(RES, "time_filters_scallop.json"), "w"), indent=1)
