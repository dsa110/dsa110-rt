"""Study 3: S/N and bias impact of storing the corr GEMM output as cfp16 before the stage-1 sums.

2x corr (plan, Architecture): the fused GEMM accumulates in fp32 over (2 pols x csf 2 channels x 8
native samples x 2 sub-times) = K 32 products of 4-bit SNAP voltages (calibration phase-only, folded
into the operand load) and writes ONE cfp16 value per (baseline, csf-2 channel, 0.262 ms sample);
stage-1 then sums those in fp32 (n sources per value x 2 or 4 samples).

Simulation (exact arithmetic otherwise):
  * per antenna complex voltages: Gaussian sky+receiver noise at SNAP level sigma_v (per component,
    counts), 4-bit two's-complement saturation at [-8, 7] (SNAP 4+4 bit), a common point-source
    signal of correlation coefficient rho, then a per-antenna phase-only calibration rotation
    (non-integer products, as on the GPU);
  * GEMM element = sum of K = 32 products, fp32 -> fp16 rounding (round-to-nearest-even, numpy);
  * value = sum of n elements (n = 1 .. 3936) and of the 2 samples of a tier-A stream sample.
Measured: relative rounding noise (rms (fp16 - fp32) / rms noise), S/N ratio of the matched
projection fp16 vs fp32, mean bias of the rounding, for sigma_v in {1.5, 2.5, 4} counts and for an
RFI-saturated case (all voltages at full scale: largest representable element, fp16 headroom).
-> results/cfp16.json
"""
from __future__ import annotations

import json
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from valuemaps import RES  # noqa: E402

K = 32


def quant4(x):
    return np.clip(np.rint(x), -8, 7)


def gemm_elements(rng, n_el, sigma_v, rho, n_ant=24):
    """n_el cross products (complex, fp64 exact) for random antenna pairs; signal correlation rho."""
    out = np.empty(n_el, np.complex128)
    done = 0
    while done < n_el:
        # K products per element: voltages (K, n_ant)
        s = rng.standard_normal((K, 1)) + 1j * rng.standard_normal((K, 1))
        e = rng.standard_normal((K, n_ant)) + 1j * rng.standard_normal((K, n_ant))
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, n_ant))        # instrumental phase per antenna
        v = sigma_v * (math.sqrt(rho) * s * np.conj(ph)[None, :] + math.sqrt(1 - rho) * e)
        vq = quant4(v.real) + 1j * quant4(v.imag)                  # SNAP 4+4 bit
        vc = vq * ph[None, :]                                      # phase-only cal on load
        a, b = np.triu_indices(n_ant, 1)
        prod = (np.conj(vc[:, a]) * vc[:, b]).sum(axis=0)          # (pairs,), signal now real
        m = min(prod.size, n_el - done)
        out[done:done + m] = prod[:m]
        done += m
    return out


def to_fp16(z):
    return z.real.astype(np.float32).astype(np.float16).astype(np.float64) + \
        1j * z.imag.astype(np.float32).astype(np.float16).astype(np.float64)


def main():
    rng = np.random.default_rng(17)
    out = {"cases": {}}
    for sigma_v in (1.5, 2.5, 4.0):
        for rho in (0.0, 0.02):
            z = gemm_elements(rng, 400_000, sigma_v, rho)
            z16 = to_fp16(z)
            err = z16 - z
            noise = z - z.mean()
            rel = math.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(noise) ** 2))
            # matched projection S/N over values of n elements (natural weighting = plain sums)
            res = {"rel_rounding_rms": rel, "max_abs_element": float(np.abs(z).max()),
                   "mean_bias_re_over_sigma": float(err.real.mean() / noise.real.std()),
                   "mean_bias_im_over_sigma": float(err.imag.mean() / noise.imag.std())}
            if rho > 0:
                for n in (1, 96, 3936):
                    m = (z.size // n) * n
                    v64 = z[:m].reshape(-1, n).sum(axis=1)
                    v16 = z16[:m].reshape(-1, n).sum(axis=1)
                    # S/N of the mean signal (real part after cal) per value, fp16 / fp32
                    snr64 = v64.real.mean() / v64.real.std()
                    snr16 = v16.real.mean() / v16.real.std()
                    res[f"snr_ratio_n{n}"] = float(snr16 / snr64)
            out["cases"][f"sigma_v{sigma_v}_rho{rho}"] = res
            print(sigma_v, rho, res, flush=True)
    # analytic: S/N factor from independent rounding noise of relative rms r is 1/sqrt(1 + r^2)
    r = max(c["rel_rounding_rms"] for c in out["cases"].values())
    out["snr_factor_worst"] = 1 / math.sqrt(1 + r * r)
    # headroom: the largest possible element (all 32 products at full scale |v| = 8 sqrt 2)
    out["max_possible_element"] = K * (8 * math.sqrt(2)) ** 2
    out["fp16_max"] = 65504.0
    print("S/N factor worst", out["snr_factor_worst"], "max element", out["max_possible_element"])
    json.dump(out, open(os.path.join(RES, "cfp16.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
