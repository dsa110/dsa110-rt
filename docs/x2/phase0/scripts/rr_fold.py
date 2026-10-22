"""Side check: how many gridded values a Hermitian fold would save (cells k and -k merged by
conjugation; natural weighting is unchanged because conj V(-k) = V(k)).  Production 1x grids
single-sided per baseline orientation (no fold), which valuemaps.py reproduces.
-> printed + results/rowrule_fold.json"""
import json
import os

import numpy as np

from valuemaps import N_GRID, RES, cell_of, core_baselines, summed_freqs_ghz

out = {}
du, dv, _, _, _ = core_baselines()
for csf, n_sub in ((2, 2), (2, 4), (8, 4)):
    nus = summed_freqs_ghz(csf)
    per = nus.shape[1] // n_sub
    nofold, fold = [], []
    for g in range(16):
        for s in range(n_sub):
            f = nus[g, s * per:(s + 1) * per]
            row, col, ok = cell_of(du[:, None], dv[:, None], f[None, :])
            r, c = row[ok] - N_GRID // 2, col[ok] - N_GRID // 2
            nofold.append(np.unique(r * 1000 + c).size)
            flip = (r < 0) | ((r == 0) & (c < 0))
            r2, c2 = np.where(flip, -r, r), np.where(flip, -c, c)
            fold.append(np.unique(r2 * 1000 + c2).size)
    out[f"csf{csf}_nsub{n_sub}"] = {"cells_nofold": float(np.mean(nofold)), "cells_fold": float(np.mean(fold)),
                                    "saving": 1 - float(np.mean(fold)) / float(np.mean(nofold))}
    print(csf, n_sub, out[f"csf{csf}_nsub{n_sub}"])
json.dump(out, open(os.path.join(RES, "rowrule_fold.json"), "w"), indent=1)
