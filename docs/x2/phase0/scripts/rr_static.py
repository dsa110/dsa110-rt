"""Study 1a: static quantisation efficiency per format with the REAL contribution counts.

  * optimum steps for mid-rise (SPEC v0.1 draft) and mid-tread quantisers at 4/5/6/8 bits;
  * eta vs a scale error (sigma_hat / true sigma), the input to the tracking study;
  * per format: main values at b bits + heavy values (n >= N/1e4) at cint8 with s_h set so the clip is
    >= 2x the heaviest value's S/N at a 250-sigma image rail -> effective image efficiency;
  * heavy-threshold scan (N/1e3 .. N/1e5) for the load / efficiency / clip trade;
  * today's 1x reference: per-row max-abs cint8 (round, mid-tread, amax/127) with today's n (Monte
    Carlo over rows, because the row maximum couples the values).
-> results/rowrule_static.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from rr_lib import Quant, eff_image, heavy_step, optimum_step
from valuemaps import FORMATS, RES, load

KINDS = ("midrise", "midtread")


def mc_rows(fmt, encode, n_rows=600, seed=1, streams=None):
    """Monte-Carlo point-source efficiency of a per-row encoder on noise rows of a format's streams.

    x_v = plain sum of n_v unit-noise contributions (re, im).  The owner sums decoded values with
    whatever static weight the encoder implies; the optimal statistic for a phase-centre source is
    P_opt = sum_v Re(x_v).  encode(x, n) -> P_q (the owner's projection, per row).  Rows are
    independent and the owner adds rows WITHOUT a row scale, so the image efficiency over many rows
    is E[P_q P_opt] / sqrt(E[P_q^2] E[P_opt^2]) -- this includes the row-to-row weight jitter."""
    vm = load(fmt)
    rng = np.random.default_rng(seed)
    sids = np.unique(vm["stream"]) if streams is None else streams
    num = dq = do = 0.0
    for st in sids:
        n = vm["n"][vm["stream"] == st].astype(float)
        x = rng.standard_normal((n_rows, 2, n.size)) * np.sqrt(n)
        p_opt = x[:, 0, :].sum(axis=1)
        p_q = encode(x, n)
        num += float(np.sum(p_q * p_opt))
        dq += float(np.sum(p_q ** 2))
        do += float(np.sum(p_opt ** 2))
    return num / math.sqrt(dq * do)


def enc_today(x, n):
    """Today: per-row amax/127 cint8, round half-even, clip, codes summed without the scale."""
    amax = np.abs(x).reshape(x.shape[0], -1).max(axis=1)
    c = np.clip(np.round(x * (127.0 / amax)[:, None, None]), -128, 127)
    return c[:, 0, :].sum(axis=1)


def enc_2x(q_main, q_heavy, thr):
    """2x: y = x / sqrt(n) (sigma_hat = 1 exactly), quantise, decode x sqrt(n)."""
    def f(x, n):
        y = x / np.sqrt(n)
        heavy = n >= thr
        d = np.where(heavy, q_heavy(y), q_main(y))
        return (d[:, 0, :] * np.sqrt(n)).sum(axis=1)
    return f


def main():
    out = {"optimum": {}, "eta_vs_scale": {}, "formats": {}, "heavy_scan": {}}
    opt = {}
    for kind in KINDS:
        for b in (4, 5, 6, 8):
            s, e = optimum_step(kind, b, lo=0.005 if b == 8 else 0.02)
            opt[(kind, b)] = s
            q = Quant(kind, b, s)
            out["optimum"][f"{kind}{b}"] = {"step": s, "eta": e, "clip_sigma": q.clip,
                                            "eta_scale_x0.7": q.eta(1 / 0.7), "eta_scale_x1.4": q.eta(1 / 1.4)}
            print(kind, b, out["optimum"][f"{kind}{b}"], flush=True)
    # eta vs scale error: sigma_hat = k * sigma_true  -> r = 1/k
    ks = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 3.0])
    for kind in KINDS:
        for b in (4, 5):
            q = Quant(kind, b, opt[(kind, b)])
            out["eta_vs_scale"][f"{kind}{b}"] = {float(k): q.eta(1.0 / k) for k in ks}
    # SPEC draft constants (mid-rise 0.336 / 0.191) for reference
    out["spec_draft"] = {"midrise4_0.336": Quant("midrise", 4, 0.336).eta(),
                         "midrise5_0.191": Quant("midrise", 5, 0.191).eta()}
    # formats
    for fmt in FORMATS:
        if fmt.startswith("today"):
            continue
        vm = load(fmt)
        n = vm["n"].astype(float)
        N = n.sum()
        thr = 1e-4 * N
        main_n, heavy_n = n[n < thr], n[n >= thr]
        s_h = heavy_step(heavy_n.max(), N)
        row = {"N": N, "n_max": float(n.max()), "heavy_count": int(heavy_n.size),
               "heavy_weight": float(heavy_n.sum() / N), "s_h": s_h,
               "heavy_clip_sigma": s_h * 127, "max_main_snr_at_250": 250 * math.sqrt(main_n.max() / N)}
        for kind in KINDS:
            for b in (4, 5):
                qm = Quant(kind, b, opt[(kind, b)])
                qh = Quant("midtread", 8, s_h)
                row[f"{kind}{b}"] = {"eta_eff": eff_image([(main_n, qm), (heavy_n, qh)]),
                                     "eta_all_main": eff_image([(n, qm)])}
        out["formats"][fmt] = row
        print(fmt, json.dumps(row), flush=True)
        # heavy threshold scan (mid-tread main values)
        scan = {}
        for frac in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            t = frac * N
            mn, hn = n[n < t], n[n >= t]
            sh = heavy_step(hn.max(), N) if hn.size else 0.0
            rr = {"heavy_count": int(hn.size), "heavy_weight": float(hn.sum() / N),
                  "max_main_snr_at_250": 250 * math.sqrt(mn.max() / N)}
            for b in (4, 5):
                groups = [(mn, Quant("midtread", b, opt[("midtread", b)]))]
                if hn.size:
                    groups.append((hn, Quant("midtread", 8, sh)))
                rr[f"eta_midtread{b}"] = eff_image(groups)
            scan[f"{frac:g}"] = rr
        out["heavy_scan"][fmt] = scan
    # today's 1x per-row max-abs cint8 (all 64 streams), and the per-row amax/127 row weights alone
    out["today_1x_maxabs_cint8"] = {
        "eta": mc_rows("today_g_ns4_csf8", enc_today),
        "eta_row_weight_only_no_quant": mc_rows(
            "today_g_ns4_csf8",
            lambda x, n: (x[:, 0, :] / np.abs(x).reshape(x.shape[0], -1).max(axis=1)[:, None]).sum(axis=1)),
        "eta_4bit_maxabs_same_rows": mc_rows(
            "today_g_ns4_csf8",
            lambda x, n: np.clip(np.round(x * (7.0 / np.abs(x).reshape(x.shape[0], -1).max(axis=1))[:, None, None]),
                                 -8, 7)[:, 0, :].sum(axis=1)),
    }
    print("today", out["today_1x_maxabs_cint8"], flush=True)
    # Monte-Carlo cross-check of the analytic eff_image for one format
    vm = load("g_ns2")
    thr = 1e-4 * vm["n"].sum()
    s_h = out["formats"]["g_ns2"]["s_h"]
    out["mc_check_g_ns2_midtread4"] = {
        "mc": mc_rows("g_ns2", enc_2x(Quant("midtread", 4, opt[("midtread", 4)]),
                                      Quant("midtread", 8, s_h), thr), n_rows=300),
        "analytic": out["formats"]["g_ns2"]["midtread4"]["eta_eff"]}
    print("mc check", out["mc_check_g_ns2_midtread4"], flush=True)
    json.dump(out, open(os.path.join(RES, "rowrule_static.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
