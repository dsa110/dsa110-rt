"""Extra S/N loss from summing a sub-band's channels per baseline BEFORE gridding (the per-unique-
baseline streams) vs today's per-channel gridding, natural weighting, core array at DEC +71.63.

A source at direction cosines (l, m) gives each baseline a fringe that rotates across the sub-band;
the channel sum has amplitude sinc((u l + v m) B / nu) with (u, v) in wavelengths at the sub-band
centre.  Nearest-cell rounding is common to both schemes (one rounding per entry vs one per channel,
both uniform within a cell), so it is not counted here.  Output: mean over baselines (natural
weighting -> relative point-source S/N) at several field positions, n_sub 2 and 4, band top/centre.
"""
import json

import numpy as np

from nfilled import BE, BN, C, COSZ

CELL = 17.9                                            # lambda, full FoV
HALF = 1.0 / (2 * CELL)                                # field half-width in direction cosine


def factor(f_mhz, b_mhz, l, m):
    lam = C / (f_mhz * 1e6)
    u, v = BE / lam, BN * COSZ / lam
    x = (u * l + v * m) * b_mhz / f_mhz
    return float(np.mean(np.sinc(x)))                  # np.sinc = sin(pi x)/(pi x)


if __name__ == "__main__":
    out = {}
    pos = {"centre": (0, 0), "half-way EW": (HALF / 2, 0), "edge EW": (HALF, 0), "edge NS": (0, HALF),
           "corner": (HALF, HALF)}
    for n_sub, b in ((2, 11.71875 / 2), (4, 11.71875 / 4), (8, 11.71875 / 8)):
        for f in (1405.0, 1498.75):
            row = {k: factor(f, b, *lm) for k, lm in pos.items()}
            # mean over the image weighted by a Gaussian primary beam (FWHM 3.2 deg) squared
            g = np.linspace(-HALF, HALF, 33)
            L, M = np.meshgrid(g, g)
            sig = np.radians(3.2) / 2.3548
            w = np.exp(-(L ** 2 + M ** 2) / sig ** 2)
            fac = np.array([[factor(f, b, l, m) for l in g] for m in g])
            row["beam^2-weighted"] = float((fac * w).sum() / w.sum())
            out[f"n_sub{n_sub} {f:.0f} MHz"] = row
            print(f"n_sub {n_sub} ({b:.2f} MHz) at {f:.0f} MHz: " +
                  "  ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
    print(f"field half-width = {np.degrees(np.arcsin(HALF)):.2f} deg")
    json.dump(out, open("results/bw_smear.json", "w"), indent=1)
