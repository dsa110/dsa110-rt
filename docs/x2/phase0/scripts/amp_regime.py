"""Study 2: today's amplitude regime in numbers, and the equivalent constants for the 2x image units.

Today (1x, live since 2026-09-23):
  wire: per (stream, DM, sample) row, code = round(x * 127 / amax_row), codes summed WITHOUT the
        scale; image = irfft2 of the conj-folded half spectrum (0.5 baked in) = real(ifft2(G)),
        G = sum of codes per cell; uv prescale 1/256 before the cfp16 FFT; torch 'backward' 1/N^2.
  => raw image sigma (dimensionless codes) = sqrt(sum_streams sum_cells Var(code)) / N^2 x prescale.
2x: decoded value = (code x s_b, or x s_h for heavy) x sqrt(n) in sigma_hat units, fp32 butterflies,
  so raw image sigma = sqrt(sum_v n_v V_v) / N^2 (option A, the plan's '1/N^2') or exactly 1 if the
  output scale is c = 1 / sqrt(sum_v n_v V_v) (option B, static per pattern).
This script computes both sigmas, the Layer-1 floor ratio, the fp16 headroom of each path, and the
detector rail per filter, and writes results/amp_regime.json.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from rr_lib import Quant, heavy_step
from valuemaps import RES, load

NG = 256
PRESCALE = 1.0 / 256
L1_FLOOR_CFG = 5.0e-3
LIVE_SIGMA_RAW_NSUB4 = 0.064          # configs/dsart_search_rt.yaml:85-90 + fp16-fft-headroom note
OPT = {4: 0.3534, 5: 0.1932}


def today_sigma_raw(n_rows=400, seed=2):
    """Model raw (un-prescaled) image sigma of today's chain, Gaussian noise, real n per cell."""
    vm = load("today_g_ns4_csf8")
    rng = np.random.default_rng(seed)
    tot = 0.0
    amax_med = []
    for sid in np.unique(vm["stream"]):
        n = vm["n"][vm["stream"] == sid].astype(float)
        x = rng.standard_normal((n_rows, 2, n.size)) * np.sqrt(n)
        amax = np.abs(x).reshape(n_rows, -1).max(axis=1)
        c = np.clip(np.round(x * (127.0 / amax)[:, None, None]), -128, 127)
        tot += float(np.mean(np.sum(c[:, 0, :] ** 2, axis=1)))      # per-component variance sum
        amax_med.append(float(np.median(amax / np.sqrt(n.max()))))
    return math.sqrt(tot) / NG ** 2, float(np.median(amax_med))


def sigma_2x(fmt, bits, hfrac=3e-5):
    n = load(fmt)["n"].astype(float)
    N = n.sum()
    thr = hfrac * N
    s_h = heavy_step(n.max(), N)
    qm, qh = Quant("midtread", bits, OPT[bits]), Quant("midtread", 8, s_h)
    vm_ = qm.moments(0.0)[1][0]
    vh_ = qh.moments(0.0)[1][0]
    S = float((n[n < thr] * vm_).sum() + (n[n >= thr] * vh_).sum())
    return math.sqrt(S) / NG ** 2, S, N


def main():
    out = {}
    sr, amax_rel = today_sigma_raw()
    out["today"] = {
        "sigma_raw_model_unprescaled": sr,
        "sigma_raw_live_unprescaled": LIVE_SIGMA_RAW_NSUB4,
        "sigma_raw_live_prescaled": LIVE_SIGMA_RAW_NSUB4 * PRESCALE,
        "amax_over_sqrt_nmax_median": amax_rel,
        "layer1_floor_raw_prescaled": L1_FLOOR_CFG * PRESCALE,
        "layer1_floor_over_sigma_live": L1_FLOOR_CFG / LIVE_SIGMA_RAW_NSUB4,
        "layer1_floor_over_sigma_model": L1_FLOOR_CFG / sr,
        "fp16_fft_linear_to_sigma_prescaled": 65504.0 / (LIVE_SIGMA_RAW_NSUB4 * NG ** 2 * PRESCALE),
        "fp16_fft_linear_to_sigma_unprescaled": 65504.0 / (LIVE_SIGMA_RAW_NSUB4 * NG ** 2),
        "fp16_boxcar_worst_sum": 250.0 * 64 * 1,
    }
    f_ratio = out["today"]["layer1_floor_over_sigma_live"]
    out["2x"] = {}
    for fmt in ("g_ns2", "g_ns4", "b_ns2", "b_ns4"):
        for bits in (4, 5):
            s, S, N = sigma_2x(fmt, bits)
            out["2x"][f"{fmt}|cint{bits}"] = {
                "sum_n_V": S, "N": N,
                "A_sigma_img_1_over_N2": s,
                "A_layer1_floor": f_ratio * s,
                "A_fp16_ring_max_sigma": 65504.0 / s,
                "A_fp16_ulp_at_sigma_in_sigma": 2.0 ** (math.floor(math.log2(s)) - 10) / s,
                "A_fp16_cufft_fallback_overflow_sigma_no_prescale": 65504.0 / (s * NG ** 2),
                "A_fp16_cufft_fallback_overflow_sigma_prescale_1_256": 65504.0 / (s * NG ** 2 * PRESCALE),
                "B_output_scale": 1.0 / math.sqrt(S),
                "B_sigma_img": 1.0,
                "B_layer1_floor": f_ratio,
            }
    # detector rail per filter (per-sample clip 250 sigma, score / sigma_k with sigma_k ~ ||h||_2 c_k)
    rails = {}
    for w in (1, 2, 4, 8, 16, 32, 64):
        rails[f"boxcar{w}"] = 250.0 * math.sqrt(w)
    for m in (2, 4, 8, 16, 32):
        tri = np.convolve(np.ones(m), np.ones(m))
        b3 = np.convolve(tri, np.ones(m))
        rails[f"B2_m{m}"] = 250.0 * tri.sum() / math.sqrt((tri ** 2).sum())
        rails[f"B3_m{m}"] = 250.0 * b3.sum() / math.sqrt((b3 ** 2).sum())
    out["rail_ideal_c1"] = rails
    out["today_wire_single_sample_saturation_sigma"] = "see rowrule_bright.json mc today: ~150 (centre) / ~160 (random)"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "amp_regime.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
