"""Trials-factor side of the filter banks, on white noise (unit-sigma, ||h||_2-normalised filters).

(1) per filter: lag-1 autocorrelation r1 of the output;
(2) correlation between ADJACENT-width filters of a bank (best relative lag) - how independent the
    7 widths are;
(3) false-alarm rate: exact per-filter upcrossing rate P(y_{t-1} <= u < y_t) (bivariate normal with
    correlation r1), summed over the bank (union bound = distinct-event rate before the width merge),
    at u = 9.55 (C1 floor) relative to the boxcar bank, and the threshold shift du that restores the
    boxcar false-alarm rate;
(4) Monte-Carlo check at u = 4.0 / 4.5: upcrossings of the bank MAX (what survives the cross-width
    merge) on 2^23 white-noise samples, bank vs boxcar.
-> results/time_filters_trials.json
"""
import json
import math
import os

import numpy as np

from tf_common import RES, M

BANK_KEYS = ("boxcar", "b2", "b3", "gauss", "b2h", "b3h", "b3x")

os.nice(10)


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def upcross_rate(r, u, n=4000):
    """P(X1 <= u < X2) for a standard bivariate normal with correlation r (per sample)."""
    if r <= 0:
        return (1 - Phi(u)) * Phi(u)
    s = math.sqrt(max(1 - r * r, 1e-300))
    xs = np.linspace(u, u + 12, n)
    f = np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi) * \
        0.5 * np.vectorize(math.erfc)(-((u - r * xs) / s) / math.sqrt(2))
    return float(np.trapezoid(f, xs))


def r1_of(h):
    h = h / np.sqrt(np.sum(h ** 2))
    return float(np.sum(h[1:] * h[:-1])) if len(h) > 1 else 0.0


def xcorr_best(h1, h2):
    a = h1 / np.sqrt(np.sum(h1 ** 2))
    b = h2 / np.sqrt(np.sum(h2 ** 2))
    return float(np.correlate(np.pad(b, (len(a), len(a))), a, mode="valid").max())


out = {"per_filter": {}, "adjacent_corr": {}, "rate_u9.55_rel_boxcar": {}, "du_sigma": {}, "mc": {}}
U = 9.55
rates = {}
for bk in BANK_KEYS:
    fl = M.BANKS[bk]
    r1 = [r1_of(f["h"]) for f in fl]
    out["per_filter"][bk] = [dict(w=f["w"], r1=round(r, 4)) for f, r in zip(fl, r1)]
    out["adjacent_corr"][bk] = [round(xcorr_best(fl[i]["h"], fl[i + 1]["h"]), 4) for i in range(len(fl) - 1)]
    rates[bk] = lambda u, r1=r1: sum(upcross_rate(r, u) for r in r1)
base = rates["boxcar"](U)
for bk in BANK_KEYS:
    rr = rates[bk](U) / base
    out["rate_u9.55_rel_boxcar"][bk] = rr
    lo, hi = -0.5, 0.5                     # du with rate_bk(U + du) = rate_box(U)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if rates[bk](U + mid) > base:
            lo = mid
        else:
            hi = mid
    out["du_sigma"][bk] = 0.5 * (lo + hi)
    print(f"{bk:6s} r1 {[round(x['r1'], 3) for x in out['per_filter'][bk]]}  adjacent corr "
          f"{out['adjacent_corr'][bk]}  rate@9.55 x{rr:.3f}  du {out['du_sigma'][bk]:+.4f} sigma")

# Monte-Carlo of the bank max (cross-width merge) at moderate thresholds
rng = np.random.default_rng(1)
NS = 1 << 23
x = rng.standard_normal(NS)
Xf = np.fft.rfft(x)
mc = {}
for bk in BANK_KEYS:
    mx = np.full(NS, -np.inf)
    for f in M.BANKS[bk]:
        h = np.zeros(NS)
        h[:f["support"]] = f["h"] / f["l2"]
        y = np.fft.irfft(Xf * np.fft.rfft(h), n=NS)[f["support"]:]   # drop the wrap-around head
        mx[f["support"]:] = np.maximum(mx[f["support"]:], y)
    mx = mx[200:]
    for u in (4.0, 4.5):
        n_up = int(np.sum((mx[1:] > u) & (mx[:-1] <= u)))
        mc.setdefault(str(u), {})[bk] = n_up
for u, d in mc.items():
    out["mc"][u] = {bk: dict(n_upcross=v, rel_boxcar=v / d["boxcar"]) for bk, v in d.items()}
    print(f"MC u={u}: " + "  ".join(f"{bk} {v} (x{v / d['boxcar']:.3f})" for bk, v in d.items()))
json.dump(out, open(os.path.join(RES, "time_filters_trials.json"), "w"), indent=1)
