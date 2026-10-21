"""Package rate gains from results/broad_grid.npz -> results/rates.json.

Detection model (Euclidean source counts, rate ~ F_min^-1.5):
  full search      F_min = theta / f_search                         theta = 9.55 (C1 floor)
  search + zoom    F_min = max(theta1 / f_search, theta2 / (0.95 f_zoom))
                   theta1 = lowered first-stage threshold, theta2 = zoom confirmation
f = recovered fraction of an ideal matched filter from the emulator.
DM pdf: log-normal, median 500, sigma_ln 0.6 (P(DM>1500) = 3.4%); compared over DM <= 1500.
Width pdfs: log-normal, median 1 ms (sigma_ln 1.0) and median 0.3 ms (sigma_ln 0.8).
"""
import json

import numpy as np

G = np.load("results/broad_grid.npz")
W = G["widths_us"]
DMS = list(G["dms"])
THETA = 9.55
ZOOM_BEAM = 0.95


def tier_cfg(pkg, dm):
    fam = pkg["fam"]                      # 'ts' or 'tc'
    if pkg["kind"] == "today":
        return "today"
    if pkg["kind"] == "1x":
        return f"{fam}1_c2"
    if pkg["kind"] == "2x":
        return f"{fam}2_c2" if dm < 1600 else f"{fam}1_c2"
    if pkg["kind"] == "4xu":
        return f"{fam}4_c2"
    if pkg["kind"] == "2xu":
        return f"{fam}2_c2"
    if pkg["kind"] == "4x":
        lo = pkg.get("lo4", f"{fam}4_c2")
        return lo if dm < 800 else (f"{fam}2_c2" if dm < 1600 else f"{fam}1_c2")
    raise ValueError(pkg)


def dm_weights(dms, med=500.0, sig=0.6):
    ld = np.log(dms)
    edges = np.concatenate([[np.log(100)], 0.5 * (ld[1:] + ld[:-1]), [np.log(3000)]])
    from math import erf, sqrt
    cdf = lambda x: 0.5 * (1 + erf((x - np.log(med)) / (sig * sqrt(2))))   # noqa: E731
    return np.array([cdf(edges[i + 1]) - cdf(edges[i]) for i in range(len(dms))])


def width_pdf(med_us, sig):
    fine = np.linspace(np.log(W[0]), np.log(W[-1]), 400)
    p = np.exp(-0.5 * ((fine - np.log(med_us)) / sig) ** 2)
    return fine, p / p.sum()


def fmin(pkg, dm_i, fine):
    cfg = tier_cfg(pkg, DMS[dm_i])
    f = np.interp(fine, np.log(W), G[cfg][dm_i])
    if np.any(np.isnan(f)):
        return np.full_like(fine, np.inf)
    F = THETA / np.maximum(f, 1e-6)
    if pkg.get("zoom"):
        fz = np.interp(fine, np.log(W), G["zoom"][dm_i]) * ZOOM_BEAM
        F = np.maximum(pkg["theta1"] / np.maximum(f, 1e-6), pkg["theta2"] / fz)
    return F


def rate(pkg, pop, dm_max=1500):
    fine, pw = width_pdf(*pop)
    wdm = dm_weights(np.array(DMS))
    tot = 0.0
    for i, dm in enumerate(DMS):
        if dm > dm_max:
            continue
        tot += wdm[i] * np.sum(pw * fmin(pkg, i, fine) ** -1.5)
    return tot


POPS = {"median 1 ms": (1000.0, 1.0), "median 0.3 ms": (300.0, 0.8)}
PKGS = {
    "today": {"kind": "today", "fam": ""},
    "today + zoom": {"kind": "today", "fam": "", "zoom": True, "theta1": 8.0, "theta2": 8.0},
    "1x fixed (2-stage)": {"kind": "1x", "fam": "ts"},
    "1x fixed (two CTs)": {"kind": "1x", "fam": "tc"},
    "1x fixed + zoom": {"kind": "1x", "fam": "ts", "zoom": True, "theta1": 8.0, "theta2": 8.0},
    "2x tiers (2-stage)": {"kind": "2x", "fam": "ts"},
    "2x tiers (two CTs)": {"kind": "2x", "fam": "tc"},
    "2x tiers + zoom": {"kind": "2x", "fam": "ts", "zoom": True, "theta1": 8.0, "theta2": 8.0},
    "4x tiers (2-stage, 61 kHz)": {"kind": "4x", "fam": "ts"},
    "4x tiers (2-stage, 122 kHz)": {"kind": "4x", "fam": "ts", "lo4": "ts4_c4"},
    "4x tiers (two CTs, CT1 0.26 ms)": {"kind": "4x", "fam": "tc"},
    "4x tiers (two CTs, CT1 0.13 ms)": {"kind": "4x", "fam": "tc", "lo4": "tc4_c2h"},
    "4x tiers + zoom": {"kind": "4x", "fam": "ts", "zoom": True, "theta1": 8.0, "theta2": 8.0},
    "2x tiers + zoom (theta1 7.5)": {"kind": "2x", "fam": "ts", "zoom": True, "theta1": 7.5, "theta2": 8.0},
    "2x uniform (2-stage)": {"kind": "2xu", "fam": "ts"},
    "4x uniform (2-stage, 61 kHz)": {"kind": "4xu", "fam": "ts"},
    "4x uniform + zoom": {"kind": "4xu", "fam": "ts", "zoom": True, "theta1": 8.0, "theta2": 8.0},
}

if __name__ == "__main__":
    base = {p: rate(PKGS["today"], POPS[p]) for p in POPS}
    out = {}
    print(f"{'package':34s} " + "  ".join(f"{p:>14s}" for p in POPS))
    for name, pkg in PKGS.items():
        g = {p: rate(pkg, POPS[p]) / base[p] for p in POPS}
        out[name] = g
        print(f"{name:34s} " + "  ".join(f"{g[p]:14.2f}" for p in POPS))
    # S/N gain table at W = 0.1 ms per DM for the main single-stage options
    sn = {}
    j = list(W).index(100.0)
    for k in ("today", "ts1_c2", "tc1_c2", "ts2_c2", "tc2_c2", "ts4_c4", "ts4_c2", "tc4_c4", "tc4_c2",
              "tc4_c2h", "zoom"):
        sn[k] = [None if np.isnan(v) else round(float(v), 3) for v in G[k][:, j]]
    json.dump({"rate_gain": out, "pops": POPS, "f_at_100us": sn, "dms": DMS}, open("results/rates.json", "w"),
              indent=1)
