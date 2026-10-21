"""Baseline (NICs stay in x4 slots) vs stretch (NIC in an x16 slot) package model.

Per-node ingress ceilings (Gb/s) for corner-turn data:
  baseline   corr: H_CORR beside ~18 Gb/s of SNAP (unknown, 0-4; b5 measures it)
             search: 17 (measured knee, single RX thread) / 22 (with RX tuning)
  +eno2      the 10G X550T port carries the first hop of two corner turns (needs cabling)
  stretch    ConnectX-4 in an x16 slot (x8): ~50 Gb/s; corr headroom ~30, search ~40
Owners: 16 corr second GPUs + 8 search GPUs (4 nodes); GPU load cap 0.8.
Per-image cost: 3.5 us (256x128, FoV/2) / 5.9 us (256x256) - rewritten owner kernels (b3).
Rate gains: Euclidean, DM log-normal median 500, DM <= 1500 (see rates.py).
"""
import json
import math

import numpy as np

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
CELLS_CH, CELLS_SB, CELLS_PLANE = 1748, 1900, 6000
BITS = 8                                      # cint4
GPU_CAP = 0.8
T1, T2, T4 = 1048.576, 524.288, 262.144
TIERS = {"1x": [(50, 3000, T1)],
         "2x": [(50, 1600, T2), (1600, 3000, T1)],
         "4x-lite": [(50, 500, T4), (500, 1600, T2), (1600, 3000, T1)],
         "4x": [(50, 800, T4), (800, 1600, T2), (1600, 3000, T1)],
         "4x-uniform": [(50, 3000, T4)]}
SCEN = {"baseline (corr +2, search 17)": dict(corr=2.0, search=17.0, eno2=False),
        "baseline (corr +4, search 22)": dict(corr=4.0, search=22.0, eno2=False),
        "baseline (corr +0, search 17)": dict(corr=0.0, search=17.0, eno2=False),
        "baseline + eno2 10G": dict(corr=2.0, search=17.0, eno2=True),
        "stretch: NIC in x16 slot": dict(corr=30.0, search=40.0, eno2=False)}


def n_dm(lo, hi, T, csf=2, tol=1.25):
    return len(gen_dm_list(lo, hi, T, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol))


def owners(sc):
    o = [{"node": f"c{i}", "net": sc["corr"], "gpu": 0.0, "in": 0.0} for i in range(16)]
    o += [{"node": f"s{i // 2}", "net": sc["search"] / 2, "gpu": 0.0, "in": 0.0} for i in range(8)]
    return o


def assign(items, own):
    """items: dicts with gbps, gpu. Largest data first; least-loaded owner with network room."""
    for it in sorted(items, key=lambda x: -x["gbps"]):
        room = [o for o in own if o["in"] + it["gbps"] <= o["net"] + 1e-9 and o["gpu"] + it["gpu"] <= GPU_CAP]
        if not room:
            return False
        o = min(room, key=lambda x: x["gpu"] + 0.05 * x["in"])
        o["in"] += it["gbps"]
        o["gpu"] += it["gpu"]
    return True


def two_stage(amb, sc, us_img, stream_half=False, n_sub=2):
    items = []
    for lo, hi, T in TIERS[amb]:
        need = 0.0206 * (hi - lo) * (T1 / T)
        n_c = max(1, math.ceil(need / n_sub))
        nf = n_dm(lo, hi, T)
        rate_stream = 1e6 / T * (2 if stream_half else 1)
        for _ in range(n_c):
            items.append({"gbps": 16 * n_sub * CELLS_SB * rate_stream * BITS / 1e9,
                          "gpu": nf / n_c * 1e6 / T * us_img * 1e-6})
    own = owners(sc)
    ok = assign(items, own)
    return finish(amb, "two-stage", sc, own, ok, items)


def two_ct(amb, sc, us_img, T_ct1, csf_ct1):
    ct1 = 16 * (384 // csf_ct1) * CELLS_CH * (1e6 / T_ct1) * BITS / 1e9
    own = owners(sc)
    if not sc["eno2"]:                       # every node is a cell owner and takes its CT1 share
        for o in own:
            o["in"] += ct1 / 20 / (2 if o["node"].startswith("s") else 1)
    items = []
    for lo, hi, T in TIERS[amb]:
        nf = n_dm(lo, hi, T)
        groups = max(1, math.ceil(nf / 8))   # 8 fine DMs per DM group
        for _ in range(groups):
            k = nf / groups
            items.append({"gbps": k * (1e6 / T) * CELLS_PLANE * BITS / 1e9,
                          "gpu": k * 1e6 / T * us_img * 1e-6})
    ok = all(o["in"] <= o["net"] + 1e-9 for o in own) and assign(items, own)
    r = finish(amb, f"two corner turns (CT1 {T_ct1/1e3:.2f} ms, {30.5*csf_ct1:.0f} kHz)", sc, own, ok, items)
    r["ct1_total"] = ct1
    return r


def finish(amb, scheme, sc, own, ok, items):
    node_in = {}
    for o in own:
        node_in[o["node"]] = node_in.get(o["node"], 0.0) + o["in"]
    return {"ambition": amb, "scheme": scheme, "fits": bool(ok),
            "max_gpu": max(o["gpu"] for o in own),
            "corr_extra_in": max(v for k, v in node_in.items() if k[0] == "c"),
            "search_in": max(v for k, v in node_in.items() if k[0] == "s"),
            "total_gpu": sum(it["gpu"] for it in items),
            "total_ct": sum(it["gbps"] for it in items)}


# ------------------------------------------------------------------ rate gains
B = dict(np.load("results/broad_grid.npz"))
B.update({k: v for k, v in np.load("results/baseline_grid.npz").items() if k not in ("dms", "widths_us")})
W = B["widths_us"]
DMS = list(B["dms"])
MAPS = {   # (scheme family, ambition) -> config per tier (DM < 800, 800-1600, >= 1600) ; lite: 500
    ("base-ts", "1x"): ("ts1_c2", "ts1_c2", "ts1_c2"),
    ("base-ts", "2x"): ("bts2", "bts2", "ts1_c2"),
    ("base-ts", "4x-lite"): ("bts4", "bts2", "ts1_c2"),
    ("base-ts", "4x"): ("bts4", "bts2", "ts1_c2"),
    ("base-tc", "1x"): ("btc1", "btc1", "btc1"),
    ("base-tc", "2x"): ("btc2", "btc2", "btc1"),
    ("str-ts", "2x"): ("ts2_c2", "ts2_c2", "ts1_c2"),
    ("str-ts", "4x"): ("ts4_c2", "ts2_c2", "ts1_c2"),
    ("str-tc", "1x"): ("tc1_c2", "tc1_c2", "tc1_c2"),
    ("str-tc", "2x"): ("tc2_c2", "tc2_c2", "tc1_c2"),
    ("str-tc", "4x"): ("tc4_c2h", "tc2_c2", "tc1_c2"),
    ("str-ts", "4x-uniform"): ("ts4_c2", "ts4_c2", "ts4_c2"),
}


def cfg_for(fam, amb, dm):
    a, b, c = MAPS[(fam, amb)]
    lo_edge = 500 if amb == "4x-lite" else 800
    return a if dm < lo_edge else (b if dm < 1600 else c)


def rate(fam, amb, pop, zoom_theta1=None):
    from rates import ZOOM_BEAM, dm_weights, width_pdf
    fine, pw = width_pdf(*pop)
    wdm = dm_weights(np.array(DMS))
    tot = 0.0
    for i, dm in enumerate(DMS):
        if dm > 1500:
            continue
        f = np.interp(fine, np.log(W), B["today" if fam == "today" else cfg_for(fam, amb, dm)][i])
        Fm = 9.55 / np.maximum(f, 1e-6)
        if zoom_theta1:
            fz = np.interp(fine, np.log(W), B["zoom"][i]) * ZOOM_BEAM
            Fm = np.maximum(zoom_theta1 / np.maximum(f, 1e-6), 8.0 / fz)
        tot += wdm[i] * np.sum(pw * Fm ** -1.5)
    return tot


if __name__ == "__main__":
    POPS = {"m1": (1000.0, 1.0), "m03": (300.0, 0.8)}
    base = {p: rate("today", None, POPS[p]) for p in POPS}
    gains = {}
    for key in MAPS:
        g = {p: rate(*key, POPS[p]) / base[p] for p in POPS}
        gz = {p: rate(*key, POPS[p], zoom_theta1=8.0) / base[p] for p in POPS}
        gains[f"{key[0]}|{key[1]}"] = {"gain": g, "gain_zoom8": gz}
        print(f"{key[0]:8s} {key[1]:8s}  x{g['m1']:.2f} / x{g['m03']:.2f}   +zoom x{gz['m1']:.2f} / x{gz['m03']:.2f}")
    feas = []
    for scn, sc in SCEN.items():
        for us, fov in ((3.5, "FoV/2"), (5.9, "full FoV")):
            for amb in ("1x", "2x", "4x-lite", "4x", "4x-uniform"):
                r = two_stage(amb, sc, us, stream_half=scn.startswith("stretch"))
                r.update(scenario=scn, fov=fov)
                feas.append(r)
                for T_ct1, csf in ((T2, 4), (T4, 4), (T4, 2)):
                    r = two_ct(amb, sc, us, T_ct1, csf)
                    r.update(scenario=scn, fov=fov)
                    feas.append(r)
    print()
    for r in feas:
        if r["fov"] == "FoV/2":
            print(f"{r['scenario']:31s} {r['ambition']:8s} {r['scheme']:38s} fits={str(r['fits']):5s} "
                  f"maxGPU {r['max_gpu']:.2f} corr+{r['corr_extra_in']:.1f} search {r['search_in']:.1f} "
                  f"CT {r['total_ct']:.0f}")
    json.dump({"gains": gains, "feas": feas}, open("results/plan3.json", "w"), indent=1)
