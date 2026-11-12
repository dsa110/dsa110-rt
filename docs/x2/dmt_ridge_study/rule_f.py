"""f_fit threshold scan for the best-fit rule (B <= 6 MHz, sig >= 8): synthetic, real, narrow, unlabelled sky."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_eval as E  # noqa: E402


def flag(f, F, BW=6.0, S=8.0):
    return bool(f.get("r_ok")) and (f.get("r_bw_fit_mhz") or 1e9) <= BW and (f.get("r_f_fit") or 0) >= F \
        and (f.get("r_fit_sigma") or 0) >= S


bins = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 400)]
for F in (0.3, 0.4, 0.45, 0.5, 0.6):
    syn = "  ".join(f"{lo}-{hi}: {np.mean([flag(s, F) for s in E.syn if s.get('r_ok') and lo <= s['occ_true_mhz'] < hi]):.2f}"
                    for lo, hi in bins)
    real = [it for it in E.items if it[1].startswith("real")]
    nar = [it for it in E.items if it[1] == "narrow"]
    sky = [it for it in E.items if it[1] == "other" and it[2] == "cut-pix"]
    rmax = max((it[3].get("r_f_fit") or 0) for it in real if (it[3].get("r_bw_fit_mhz") or 1e9) <= 10 and it[3].get("r_ok"))
    print(f"F>={F:<4} | {syn} | real {sum(flag(it[3], F) for it in real)}/{len(real)} | "
          f"narrow {sum(flag(it[3], F) for it in nar)}/{len(nar)} | unlabelled sky (pix) "
          f"{sum(flag(it[3], F) for it in sky)}/{len(sky)} | max f of real with B<=10: {rmax:.2f}")
