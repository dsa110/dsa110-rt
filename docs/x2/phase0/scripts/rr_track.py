"""Study 1c: tracking the per-stream scale sigma_hat (EMA of a robust block estimate) under realistic
noise-level changes, choice of tau, start-up, and the per-row anomaly guard.

Per (stream, coarse DM): one robust estimate per corr block (134.218 ms) =
1.4826 * median |y| over the block's values x samples x (re, im), optionally after excluding
|y| > 6 sigma_hat_prev (SPEC v0.1 draft), then sigma_hat <- sigma_hat + alpha (e - sigma_hat),
alpha = 1 - exp(-dt / tau).  The block estimate's own statistical error is ~0.1 % (>= 1e6 components)
and is included as Gaussian jitter.

Scenarios (true per-stream noise sigma(t), 1 h unless stated; 64 streams with independent phases or
realisations so the cross-stream weighting error is visible):
  drift15   sigma = 1 + 0.15 sin(2 pi t / 600 s + phase_j)
  drift30   sigma = 1 + 0.30 sin(2 pi t / 300 s + phase_j)
  walk      log sigma = random walk, 10 % rms over 120 s, mean-reverting (OU, 10 min)
  flags     a fraction f of the stream's channels flagged on/off (f in {0, 0.3}, dwell exp 60 s)
            -> sigma = sqrt(1 - f) (channel flags thin every value of the stream alike)
  jumps     gain steps x1.3 or /1.3 every ~5 min (weight updates, SNAP re-arm)
  startup   8 static-sky warm-up blocks carry unsubtracted sky (sigma_eff = 5), then 1; EMA started
            at block 0 vs at the first post-warm-up block
Per block and stream: k = sigma_hat / sigma; quantiser efficiency eta_b(k) (mid-tread, b = 4, 5);
with the guard (G = 1.5) a row whose own robust scale exceeds G sigma_hat is coded with its own scale
(eta_b(1)) -- the guard's effect on these slow scenarios is nil unless k < 1/G.
Cross-stream weighting efficiency per block = mean(1/k) / sqrt(mean(1/k^2)) (decoded values are in
sigma_hat units; the ideal is today's 1/sigma semantics with the TRUE sigma).

Also: image-noise inflation of a single noisy row (noise x F in a fraction p of streams for one
sample) under today's per-row max-abs (exactly normalised), the 2x rule without and with the guard,
and the expected max-pixel S/N over one sample of the cube (256 x 256 x n_fdm ~ 17 fine DMs).
-> results/rowrule_track.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from rr_lib import Phi, Quant
from valuemaps import RES

DT = 0.134217728
OPT = {4: 0.3534, 5: 0.1932}
TAUS = (1.0, 3.0, 10.0, 30.0, 100.0)
N_STREAMS = 64
G = 1.5
EST_JITTER = 0.001


def inv_phi(p, lo=-10.0, hi=10.0):
    p = np.asarray(p, float)
    a = np.full(p.shape, lo)
    b = np.full(p.shape, hi)
    for _ in range(60):
        m = 0.5 * (a + b)
        below = Phi(m) < p
        a = np.where(below, m, a)
        b = np.where(below, b, m)
    return 0.5 * (a + b)


_C_GRID = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 2001))
_M_GRID = 1.4826 * inv_phi(0.5 + 0.25 * (2 * Phi(_C_GRID) - 1))   # median of |z| truncated at c


def block_estimate(sig, shat_prev, excl):
    """Expected robust estimate for true sigma `sig` given the exclusion threshold."""
    if not excl:
        return sig
    c = 6.0 * shat_prev / sig                    # threshold in true-sigma units
    return np.interp(np.log(c), np.log(_C_GRID), _M_GRID) * sig


def run_ema(sig, tau, excl, rng, init_block=0):
    """sig: (T, S) true sigma per block and stream -> sigma_hat (T, S)."""
    T, S = sig.shape
    alpha = 1.0 - math.exp(-DT / tau)
    sh = np.zeros((T, S))
    cur = None
    for t in range(T):
        if t < init_block:
            sh[t] = sig[t]                         # rows before the EMA starts: coded per row
            continue
        e = sig[t] if cur is None else block_estimate(sig[t], cur, excl)
        e = e * (1.0 + EST_JITTER * rng.standard_normal(S))
        cur = e if cur is None else cur + alpha * (e - cur)
        sh[t] = cur
    return sh


def eta_table(bits):
    q = Quant("midtread", bits, OPT[bits])
    ks = np.exp(np.linspace(np.log(0.05), np.log(20.0), 801))
    return ks, np.array([q.eta(1.0 / k) for k in ks])


def scenarios(rng, hours=1.0):
    T = int(hours * 3600 / DT)
    t = np.arange(T)[:, None] * DT
    ph = rng.uniform(0, 2 * np.pi, N_STREAMS)[None, :]
    out = {
        "drift15": 1 + 0.15 * np.sin(2 * np.pi * t / 600 + ph),
        "drift30": 1 + 0.30 * np.sin(2 * np.pi * t / 300 + ph),
    }
    # OU log random walk: 10 % rms increment over 120 s, reversion 600 s
    x = np.zeros((T, N_STREAMS))
    th = DT / 600.0
    sd = 0.10 * math.sqrt(DT / 120.0)
    for i in range(1, T):
        x[i] = x[i - 1] * (1 - th) + sd * rng.standard_normal(N_STREAMS)
    out["walk"] = np.exp(x)
    # flags on/off
    f = np.zeros((T, N_STREAMS))
    for j in range(N_STREAMS):
        i, on = 0, False
        while i < T:
            d = max(1, int(rng.exponential(60.0) / DT))
            f[i:i + d, j] = 0.3 if on else 0.0
            on = not on
            i += d
    out["flags"] = np.sqrt(1 - f)
    # gain jumps
    g = np.ones((T, N_STREAMS))
    for j in range(N_STREAMS):
        i, lvl = 0, 1.0
        while i < T:
            d = max(1, int(rng.exponential(300.0) / DT))
            g[i:i + d, j] = lvl
            lvl = lvl * (1.3 if rng.random() < 0.5 else 1 / 1.3)
            lvl = min(max(lvl, 1 / 1.7), 1.7)
            i += d
    out["jumps"] = g
    # start-up: 8 warm-up blocks with unsubtracted sky, then the residual noise, 120 s
    Ts = int(120 / DT)
    su = np.ones((Ts, N_STREAMS))
    su[:8] = 5.0
    out["startup"] = su
    return out


def summarise(sig, sh, ks, et, guard):
    k = sh / sig
    eta = np.interp(np.log(k), np.log(ks), et)
    if guard:
        eta = np.where(k < 1.0 / G, np.interp(0.0, np.log(ks), et), eta)
    winv = 1.0 / k
    weff = winv.mean(axis=1) / np.sqrt((winv ** 2).mean(axis=1))
    tot = eta.mean(axis=1) * weff
    return {"eta_mean": float(eta.mean()), "eta_min": float(eta.min()),
            "frac_blocks_eta_below_0.99": float((eta < 0.99).mean()),
            "weight_eff_mean": float(weff.mean()), "total_mean": float(tot.mean()),
            "total_min": float(tot.min()), "k_p01": float(np.percentile(k, 1)),
            "k_p99": float(np.percentile(k, 99))}


def surge_table():
    """Image-noise inflation for one row-sample with noise x F in a fraction p of the streams."""
    out = {}
    n_pix = 256 * 256 * 17
    # expected max of n_pix Gaussians (Gumbel location + Euler gamma * scale)
    a = math.sqrt(2 * math.log(n_pix))
    gmax = a - (math.log(math.log(n_pix)) + math.log(4 * math.pi)) / (2 * a) + 0.5772 / a
    for b in (4, 5):
        q = Quant("midtread", b, OPT[b])
        v1 = q.moments(0.0)[1][0]
        rows = {}
        for F in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0):
            vF = q.moments(0.0, F)[1][0] / v1                      # variance ratio, no guard
            vG = vF if F <= G else 1.0                              # guard: row coded at its own scale
            for p in (0.25, 1.0):
                infl = math.sqrt(1 - p + p * vF)
                inflG = math.sqrt(1 - p + p * vG)
                rows[f"F{F:g}_p{p:g}"] = {"inflation_noguard": infl, "inflation_guard": inflG,
                                          "max_pixel_snr_noguard": infl * gmax,
                                          "max_pixel_snr_guard": inflG * gmax,
                                          "today_inflation": 1.0}
        out[f"midtread{b}"] = rows
    out["expected_max_pixel_snr_unit_noise"] = gmax
    return out


def main():
    rng = np.random.default_rng(7)
    sc = scenarios(rng)
    res = {"tau_s": list(TAUS), "guard_G": G, "scenarios": {}}
    tabs = {b: eta_table(b) for b in (4, 5)}
    for name, sig in sc.items():
        for excl in (False, True):
            for tau in TAUS:
                inits = (0, 8) if name == "startup" else (0,)
                for init in inits:
                    sh = run_ema(sig, tau, excl, np.random.default_rng(11), init_block=init)
                    for b in (4, 5):
                        for guard in (False, True):
                            key = f"{name}|excl{int(excl)}|tau{tau:g}|init{init}|b{b}|guard{int(guard)}"
                            res["scenarios"][key] = summarise(sig, sh, *tabs[b], guard)
            print(name, excl, flush=True)
    res["surge"] = surge_table()
    # print a compact table
    for name in sc:
        for tau in TAUS:
            k = f"{name}|excl0|tau{tau:g}|init0|b4|guard1"
            r = res["scenarios"][k]
            print(f"{name:8s} tau {tau:5.0f}: eta4 {r['eta_mean']:.5f} (min {r['eta_min']:.4f}) "
                  f"weff {r['weight_eff_mean']:.5f} total {r['total_mean']:.5f} k[1,99]% "
                  f"{r['k_p01']:.3f} {r['k_p99']:.3f}")
    json.dump(res, open(os.path.join(RES, "rowrule_track.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
