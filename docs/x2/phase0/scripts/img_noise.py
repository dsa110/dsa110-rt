"""Noise per unit point-source signal, A (gridded cells) vs B (per-baseline entries): Monte Carlo.

Analytically every pixel of the real dirty image has variance sum_k n_k / N^2 = 1 / N for ANY
placement of the contributions (each cell contributes Re(G_k e^{i theta}) with variance n_k), so the
per-pixel noise is identical in A and B and the point-source S/N ratio equals the peak ratio. The
spatial rms of one realisation scatters about that by ~1% (heavy cells dominate sum |G_k|^2 and the
noise is PSF-correlated), so average several realisations and report the per-realisation spread.
"""
import json
import os

import numpy as np

import img_response as R

os.nice(10)
G = R.Geometry()
rng = np.random.default_rng(11)
acc = {"A": [], "B2": [], "B4": []}
for _ in range(10):
    re = rng.standard_normal(int(G.N))
    im = rng.standard_normal(int(G.N))
    acc["A"].append(R.fft_image(G.grid_A(re, im), G.N).std() * np.sqrt(G.N))
    for n in (2, 4):
        acc[f"B{n}"].append(R.fft_image(G.grid_B(n, G.entries(n, re, im)), G.N).std() * np.sqrt(G.N))
out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in acc.items()}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(R.RES, "imaging_per_baseline_noise.json"), "w"), indent=1)
