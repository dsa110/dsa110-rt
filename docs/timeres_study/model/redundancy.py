"""How much does redundant-baseline averaging shrink the corner turn? -> results/redundancy.json.

Counts, for the core array at DEC +71.63 (same antenna file and projection as nfilled.py):
  * baselines, exactly redundant classes (positions agree to `tol` m), and uv cells per
    channel at the full-FoV cell (17.9 lambda) - gridding already merges what lands in one cell;
  * cells per corr-node stream when the FoV is halved along ONE axis (the 256x128 image) vs
    symmetrically (sqrt 2 per axis), for u-halving and v-halving;
  * the per-baseline alternative: ship redundancy-averaged VISIBILITIES (one value per unique
    baseline, frequency-independent identity) and grid at the owner - removes the union-over-
    channels growth but needs n_sub >= 4 to keep bandwidth smearing small.
"""
import json

import numpy as np

from nfilled import BE, BN, COSZ, C
from physics import FREQS_MHZ


def uniq(tol):
    q = np.stack([np.rint(BE / tol), np.rint(BN / tol)], 1).astype(np.int64)
    flip = (q[:, 1] < 0) | ((q[:, 1] == 0) & (q[:, 0] < 0))
    q[flip] *= -1
    return len(np.unique(q, axis=0))


def cells(freqs, cu=17.9, cv=17.9):
    s = set()
    for f in freqs:
        lam = C / (f * 1e6)
        u = BE / lam / cu
        v = BN * COSZ / lam / cv
        flip = (v < 0) | ((v == 0) & (u < 0))
        u = np.where(flip, -u, u)
        v = np.where(flip, -v, v)
        s.update(zip(np.rint(u).astype(int).tolist(), np.rint(v).astype(int).tolist()))
    return len(s)


def stream(g, n_sub, cu, cv, csf=2):
    ch = FREQS_MHZ[g * 384:(g + 1) * 384]
    per = 384 // n_sub
    return float(np.mean([cells(ch[k * per:(k + 1) * per:csf], cu, cv) for k in range(n_sub)]))


if __name__ == "__main__":
    out = {"n_baselines": int(len(BE))}
    for tol in (0.05, 0.25, 1.0):
        out[f"unique_tol{tol}m"] = uniq(tol)
    out["cells_per_channel_full"] = cells([FREQS_MHZ[3072]])
    print(out, flush=True)
    geo = {"full": (17.9, 17.9), "FoV/2 sym": (17.9 * 2 ** 0.5, 17.9 * 2 ** 0.5),
           "FoV/2 halve u": (35.8, 17.9), "FoV/2 halve v": (17.9, 35.8)}
    for name, (cu, cv) in geo.items():
        out[f"cells_per_channel {name}"] = cells([FREQS_MHZ[3072]], cu, cv)
        for n_sub in (1, 2, 4):
            v = np.mean([stream(g, n_sub, cu, cv) for g in (0, 5, 10, 15)])
            out[f"{name} nsub{n_sub}"] = float(v)
            print(f"{name:14s} n_sub {n_sub}: {v:6.0f} cells/stream  -> {n_sub * v:6.0f} per node per coarse DM",
                  flush=True)
    json.dump(out, open("results/redundancy.json", "w"), indent=1)
