"""Study 1b: bright response of the wire (per-sample image S/N in -> out), 1 .. 1000 sigma.

A point source of per-sample image S/N rho (ideal, natural weighting) gives value v the signal
a_v = rho * sqrt(n_v / N) in units of its own noise.  The owner's image signal is
sum_v sqrt(n_v) <projection of E[Q(a_v e^{i phi_v} + z)]> and the reference noise is the noise-only
image sigma (what Layer-1 measures on the bulk of the cube), so

    L(rho) = S_out(rho) / (rho * eta_eff)          (1 = perfectly linear wire)

Two source phases: 'centre' (phi_v = 0 for every value: all signal in the real part, the worst case
for per-component clipping -- a calibrated source at the phase centre) and 'random' (phi_v uniform,
a source well off-centre).

  * 2x rule (analytic, exact Gaussian moments): main values at b bits (mid-tread and mid-rise), heavy
    values at cint8 with s_h from rr_lib.heavy_step; also heavy threshold variants;
  * today's 1x per-row max-abs cint8 (Monte Carlo over rows; the row maximum moves with the burst);
  * 2x with the per-row anomaly guard (Monte Carlo): the row's own robust scale is used when it
    exceeds G x sigma_hat (G = 1.5) -- shows where the guard starts to compress a real burst.
-> results/rowrule_bright.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from rr_lib import Quant, heavy_step
from valuemaps import RES, load

RHO = np.array([1, 3, 10, 20, 35, 50, 75, 100, 150, 200, 250, 300, 400, 500, 700, 1000], float)
N_PHI = 24
OPT = {("midtread", 4): 0.3534, ("midtread", 5): 0.1932, ("midrise", 4): 0.3352, ("midrise", 5): 0.1881}


def analytic_curve(n, q_main, q_heavy, thr, phase):
    """L(rho) for the static-scale 2x rule."""
    un, cnt = np.unique(n, return_counts=True)
    N = float((un * cnt).sum())
    groups = [(un[un < thr], cnt[un < thr], q_main), (un[un >= thr], cnt[un >= thr], q_heavy)]
    groups = [g for g in groups if g[0].size]
    # noise-only variance and small-signal gain
    var0, num0 = 0.0, 0.0
    for u, c, q in groups:
        m1, m2, g = q.moments(0.0)
        var0 += float((c * u).sum()) * float(m2[0] - m1[0] ** 2)
        num0 += float((c * u).sum()) * float(g[0])
    eta = num0 / math.sqrt(N * var0)
    phis = [0.0] if phase == "centre" else list((np.arange(N_PHI) + 0.5) * 2 * np.pi / N_PHI)
    out = []
    for rho in RHO:
        s = 0.0
        for u, c, q in groups:
            a = rho * np.sqrt(u / N)
            acc = np.zeros(u.size)
            for ph in phis:
                cr, sr = math.cos(ph), math.sin(ph)
                acc += q.moments(a * cr)[0] * cr
                if abs(sr) > 1e-12:
                    acc += q.moments(a * sr)[0] * sr
            s += float((c * np.sqrt(u) * acc).sum()) / len(phis)
        out.append(s / math.sqrt(var0) / (rho * eta))
    return eta, out


RHO_MC = np.concatenate([[5.0], RHO, [2000.0, 5000.0, 20000.0]])


def mc_curve(fmt, encode, phase, n_rows=32, seed=3, stream_stride=1):
    """L(rho) by Monte Carlo for a per-row encoder; encode(x, n, phi) -> owner projection per row,
    linear in the decoded values, ODD in x (antithetic differences are then exact in expectation).
    The projection direction is the source phase per value.  Normalised at rho = 5 (linear)."""
    vm = load(fmt)
    n_all = vm["n"].astype(float)
    N = n_all.sum()
    rng = np.random.default_rng(seed)
    sids = np.unique(vm["stream"])[::stream_stride]
    res = np.zeros(len(RHO_MC))
    for st in sids:
        n = n_all[vm["stream"] == st]
        phi = np.zeros(n.size) if phase == "centre" else rng.uniform(0, 2 * np.pi, n.size)
        z = rng.standard_normal((n_rows, 2, n.size)) * np.sqrt(n)
        for i, rho in enumerate(RHO_MC):
            sig = np.stack([np.cos(phi), np.sin(phi)])[None] * (rho / math.sqrt(N) * n)[None, None, :]
            res[i] += 0.5 * float(np.mean(encode(z + sig, n, phi) - encode(z - sig, n, phi)))
    L = res / RHO_MC / (res[0] / RHO_MC[0])
    return {float(r): float(v) for r, v in zip(RHO_MC[1:], L[1:])}


def enc_today(x, n, phi):
    amax = np.abs(x).reshape(x.shape[0], -1).max(axis=1)
    c = np.clip(np.round(x * (127.0 / amax)[:, None, None]), -128, 127)
    return (c[:, 0, :] * np.cos(phi) + c[:, 1, :] * np.sin(phi)).sum(axis=1)


def enc_2x_guard(q_main, q_heavy, thr, G):
    """sigma_hat = 1 (tracked); per row: robust scale s_row = 1.4826 median |y| over all components;
    if s_row > G the row uses s_row as its scale (codes then decode in sigma_hat units WITHOUT the
    row factor, i.e. the row is down-weighted as today) -- G = inf disables the guard."""
    def f(x, n, phi):
        y = x / np.sqrt(n)
        s_row = 1.4826 * np.median(np.abs(y).reshape(y.shape[0], -1), axis=1)
        scale = np.where(s_row > G, s_row, 1.0)[:, None, None]
        heavy = n >= thr
        d = np.where(heavy, q_heavy(y / scale), q_main(y / scale))
        return ((d[:, 0, :] * np.cos(phi) + d[:, 1, :] * np.sin(phi)) * np.sqrt(n)).sum(axis=1)
    return f


def main():
    out = {"rho": RHO.tolist(), "analytic": {}, "mc": {}}
    for fmt in ("g_ns2", "g_ns4", "b_ns2", "b_ns4"):
        n = load(fmt)["n"].astype(float)
        N = n.sum()
        for hfrac in (1e-4, 3e-5):
            thr = hfrac * N
            s_h = heavy_step(n.max(), N)
            for kind, b in (("midtread", 4), ("midtread", 5), ("midrise", 4)):
                if hfrac != 1e-4 and kind == "midrise":
                    continue
                for phase in ("centre", "random"):
                    eta, L = analytic_curve(n, Quant(kind, b, OPT[(kind, b)]), Quant("midtread", 8, s_h), thr, phase)
                    key = f"{fmt}|{kind}{b}|heavy{hfrac:g}|{phase}"
                    out["analytic"][key] = {"eta": eta, "L": L}
                    print(key, f"eta {eta:.5f}", " ".join(f"{r:g}:{v:.4f}" for r, v in zip(RHO, L)), flush=True)
    # today (1x) and the guard variants, Monte Carlo
    for phase in ("centre", "random"):
        L = mc_curve("today_g_ns4_csf8", enc_today, phase, stream_stride=2)
        out["mc"][f"today|{phase}"] = L
        print("today", phase, " ".join(f"{r:g}:{v:.4f}" for r, v in L.items()), flush=True)
    for fmt in ("g_ns2", "b_ns4"):
        n = load(fmt)["n"].astype(float)
        N = n.sum()
        thr = 1e-4 * N
        s_h = heavy_step(n.max(), N)
        for G in (1.5, np.inf):
            for phase in ("centre", "random"):
                L = mc_curve(fmt, enc_2x_guard(Quant("midtread", 4, OPT[("midtread", 4)]),
                                               Quant("midtread", 8, s_h), thr, G), phase, stream_stride=4)
                key = f"{fmt}|midtread4|guard{G}|{phase}"
                out["mc"][key] = L
                print(key, " ".join(f"{r:g}:{v:.4f}" for r, v in L.items()), flush=True)
    json.dump(out, open(os.path.join(RES, "rowrule_bright.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
