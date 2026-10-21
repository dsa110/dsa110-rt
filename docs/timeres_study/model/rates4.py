"""Rate gains for the v4 packages (no zoom, no threshold change) -> results/rates4.json."""
import json

import numpy as np

from rates import dm_weights, width_pdf

B = dict(np.load("results/broad_grid.npz"))
for f in ("results/baseline_grid.npz", "results/tol_grid.npz"):
    B.update({k: v for k, v in np.load(f).items() if k not in ("dms", "widths_us")})
W = B["widths_us"]
DMS = list(B["dms"])
POPS = {"m1": (1000.0, 1.0), "m03": (300.0, 0.8)}
PKG = {  # name: (config below edge, config edge-1600, config >= 1600, edge)
    "today": ("today", "today", "today", 800),
    "1x: two-stage done right": ("ts1_c2", "ts1_c2", "ts1_c2", 800),
    "2x: 0.52 ms below DM 1600": ("bts2", "bts2", "ts1_c2", 800),
    "4x below DM 500": ("bts4", "bts2", "ts1_c2", 500),
    "4x below DM 800": ("bts4", "bts2", "ts1_c2", 800),
    "4x below DM 800, tol 1.4": ("bts4_t14", "bts2", "ts1_c2", 800),
    "4x to DM 1600, tol 1.4": ("bts4_t14", "bts4_t14", "ts1_c2", 1600),
    "stretch: 4x below DM 800, finer shifts": ("tc4_c2h", "tc2_c2", "tc1_c2", 800),
    "stretch: 4x to DM 1600, finer shifts": ("tc4_c2h", "tc4_c2h", "tc1_c2", 1600),
}


def rate(cfgs, pop):
    a, b, c, edge = cfgs
    fine, pw = width_pdf(*pop)
    wdm = dm_weights(np.array(DMS))
    tot = 0.0
    for i, dm in enumerate(DMS):
        if dm > 1500:
            continue
        k = a if dm < edge else (b if dm < 1600 else c)
        f = np.interp(fine, np.log(W), B[k][i])
        tot += wdm[i] * np.sum(pw * (9.55 / np.maximum(f, 1e-6)) ** -1.5)
    return tot


if __name__ == "__main__":
    base = {p: rate(PKG["today"], POPS[p]) for p in POPS}
    out = {}
    j = list(W).index(100.0)
    for name, cfgs in PKG.items():
        g = {p: rate(cfgs, POPS[p]) / base[p] for p in POPS}
        sn = [float(B[cfgs[0] if dm < cfgs[3] else (cfgs[1] if dm < 1600 else cfgs[2])][i, j])
              for i, dm in enumerate(DMS)]
        out[name] = {"gain": g, "snr_100us": sn}
        print(f"{name:42s} x{g['m1']:.2f} / x{g['m03']:.2f}   S/N(0.1 ms) DM300..2500: "
              + " ".join(f"{v:.2f}" for v in sn))
    json.dump({"pkgs": out, "dms": DMS}, open("results/rates4.json", "w"), indent=1)
