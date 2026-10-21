"""4x feasibility v6: measured fabric (b5b/b5c) + redundancy-aware streams -> results/plan6.json.

Fabric, per-node ingress available for corner-turn data (Gb/s):
  b5b: a corr node's x4-slot ConnectX-4 delivers <= ~17 Gb/s to the host whatever is offered
       (n04: 18.2/20.1/22.0/25.6 offered -> 17.2/17.1/16.9/16.7 delivered), and SNAP alone is
       16-18 -> corr headroom on the ConnectX is ZERO.
  search: H_s per node (b5c; argv[1]).
  eno2: the on-board X550T 10GBASE-T port (own PCIe link) adds ~9.4 Gb/s if cabled.
  stretch: card re-seated x8 -> ~2x the x4 ceiling (~34), corr = 34 - 18 SNAP.
Streams (cint4, 8 bit per cell), cells per (corr node, sub-band) stream:
  FoV/2 = 256x128 image = uv cell doubled along u: gridded 1773 (n_sub 2) / 1564 (n_sub 4)
  full FoV: per-unique-baseline (redundancy-averaged) visibilities, 1773 at any n_sub
            (gridded cells would be 2483 / 2112: the union over a chgroup's channels).
"""
import json
import math
import sys

from plan5 import AMB, B3, B3_READS, T1, n_dm

STREAM = {("FoV/2", 2): 1773.4, ("FoV/2", 4): 1564.5, ("full", 2): 1773.0, ("full", 4): 1773.0}
STREAM_V5 = {("FoV/2", 2): 2146.0, ("FoV/2", 4): 1954.0, ("full", 2): 2427.0, ("full", 4): 2084.0}
ENO2 = 9.4


def us_image(fov, n_sub, scale=1.0):
    b = B3[fov]
    comb = b["combine"] * 16 * n_sub * STREAM[(fov, n_sub)] / B3_READS
    return scale * 1e3 * (comb + b["image"] + b["detect"]) / 4096


def items(amb, fov, n_sub, tol=1.4, scale=1.0, cells=None):
    cells = cells or STREAM
    us = us_image(fov, n_sub, scale)
    out = []
    for lo, hi, T in AMB[amb]:
        nf = n_dm(lo, hi, T, tol=tol)
        n_c = max(1, math.ceil(0.0206 * (hi - lo) * (T1 / T) / n_sub))
        gbps = 16 * n_sub * cells[(fov, n_sub)] * (1e6 / T) * 8 / 1e9
        gpu = nf / n_c * (1e6 / T) * us * 1e-6
        k = max(1, math.ceil(gpu / 0.72))
        out += [{"gbps": gbps, "gpu": gpu / k} for _ in range(n_c * k)]
    return out


def fabric(H_s):
    return {"baseline": dict(corr=0.0, search=H_s),
            "eno2 on search nodes": dict(corr=0.0, search=H_s + ENO2),
            "eno2 on all nodes": dict(corr=ENO2, search=H_s + ENO2),
            "stretch: card in x16 slot": dict(corr=34 - 17, search=34.0)}


def fits(its, corr, search, n_search=4, cap_c=0.4, cap_s=0.8):
    own = [{"n": f"c{i}", "cap": cap_c, "gpu": 0.0} for i in range(16)]
    own += [{"n": f"s{i // 2}", "cap": cap_s, "gpu": 0.0} for i in range(2 * n_search)]
    net = {f"c{i}": corr for i in range(16)}
    net.update({f"s{i}": search for i in range(n_search)})
    used = {k: 0.0 for k in net}
    for it in sorted(its, key=lambda x: (-x["gbps"], -x["gpu"])):
        room = [o for o in own if used[o["n"]] + it["gbps"] <= net[o["n"]] + 1e-9
                and o["gpu"] + it["gpu"] <= o["cap"] + 1e-9]
        if not room:
            return False
        o = min(room, key=lambda x: (x["gpu"] / x["cap"], used[x["n"]]))
        o["gpu"] += it["gpu"]
        used[o["n"]] += it["gbps"]
    return True


if __name__ == "__main__":
    H_S = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    N_S = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    FAB = fabric(H_S)
    rows = []
    print(f"search ingress {H_S} Gb/s x {N_S} nodes;  us/image FoV/2 n_sub2 {us_image('FoV/2', 2):.2f}, "
          f"n_sub4 {us_image('FoV/2', 4):.2f}; full n_sub2 {us_image('full', 2):.2f}")
    for amb in AMB:
        for fov in ("FoV/2", "full"):
            for scale in (1.0, 0.7):
                best = None
                for n_sub in (2, 4):
                    its = items(amb, fov, n_sub, 1.4, scale)
                    ok = {f: fits(its, d["corr"], d["search"], N_S) for f, d in FAB.items()}
                    r = {"amb": amb, "fov": fov, "kernels": scale, "n_sub": n_sub,
                         "ct_gbps": sum(i["gbps"] for i in its), "gpu_eq": sum(i["gpu"] for i in its),
                         "ct_gbps_v5_cells": sum(i["gbps"] for i in items(amb, fov, n_sub, 1.4, scale, STREAM_V5)),
                         "fits": ok, "cheapest": next((f for f in FAB if ok[f]), "none")}
                    rows.append(r)
                    order = list(FAB) + ["none"]
                    if best is None or order.index(r["cheapest"]) < order.index(best["cheapest"]):
                        best = r
                print(f"{amb:18s} {fov:6s} kernels x{scale:.1f}  n_sub {best['n_sub']}  CT {best['ct_gbps']:6.1f} "
                      f"(v5 cells {best['ct_gbps_v5_cells']:6.1f})  GPU-eq {best['gpu_eq']:5.1f}  -> {best['cheapest']}")
    json.dump({"H_s": H_S, "n_search": N_S, "fabric": FAB, "rows": rows}, open("results/plan6.json", "w"), indent=1)
