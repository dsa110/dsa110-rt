"""Package model with explicit owner assignment and per-node NIC headroom.

Owners: 16 corr second GPUs (NIC headroom = NIC - 18 Gb/s SNAP) + 8 search GPUs
(2 per node, NIC shared).  Images balanced across owners; ingress checked per node.
Per-image GPU cost (rewritten owner kernels): 3.5 us (256x128, FoV/2) / 5.9 us (256^2).
"""
import json
import math

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
SNAP = 18.0
CELLS_CH, CELLS_SB, CELLS_PLANE = 1748, 1900, 6000
TIERS = {"1x": [(50, 3000, 1048.576)],
         "2x": [(50, 1600, 524.288), (1600, 3000, 1048.576)],
         "4x": [(50, 800, 262.144), (800, 1600, 524.288), (1600, 3000, 1048.576)]}


def n_dm(lo, hi, T, csf, tol=1.25):
    return len(gen_dm_list(lo, hi, T, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol))


def two_stage(kind, csf=2, bits=8, n_sub=4, nic=25.0):
    buckets = []
    for lo, hi, T in TIERS[kind]:
        need = 0.0206 * (hi - lo) * (1048.576 / T)          # n_c*n_s for <= 1 sample residual smear
        n_c = max(1, math.ceil(need / n_sub))
        nf = n_dm(lo, hi, T, csf)
        for _ in range(n_c):
            buckets.append({"T": T, "img": nf / n_c * 1e6 / T,
                            "gbps": 16 * n_sub * CELLS_SB * (1e6 / T) * bits / 1e9})
    img = sum(b["img"] for b in buckets)
    corr_egress = sum(b["gbps"] for b in buckets) / 16
    owners = [{"node": f"c{i}", "head": nic - SNAP - corr_egress * 0, "img": 0.0, "in": 0.0} for i in range(16)]
    owners += [{"node": f"s{i // 2}", "head": nic / 2, "img": 0.0, "in": 0.0} for i in range(8)]
    # largest-first greedy on images, respecting ingress headroom
    for b in sorted(buckets, key=lambda x: -x["img"]):
        cands = [o for o in owners if o["in"] + b["gbps"] <= o["head"] + 1e-9]
        o = min(cands or owners, key=lambda x: x["img"])
        o["img"] += b["img"]
        o["in"] += b["gbps"]
    return summarise(kind, "two-stage", csf, bits, img, owners, corr_egress, len(buckets), nic)


def two_ct(kind, csf=2, bits=8, nic=25.0, ct1_on_10g=False, T1=262.144):
    # CT1 runs at 262 us in every package (the emulated fidelity assumes it), so the
    # corr side and CT1 never change when the imaging cadence is upgraded later.
    img = sum(n_dm(lo, hi, T, csf) * 1e6 / T for lo, hi, T in TIERS[kind])
    ct1 = 16 * (384 // csf) * CELLS_CH * (1e6 / T1) * bits / 1e9       # Gb/s total
    ct2 = img * CELLS_PLANE * bits / 1e9
    n_cell_nodes = 20
    owners = [{"node": f"c{i}", "head": nic - SNAP, "img": img / 24, "in": ct2 / 24} for i in range(16)]
    owners += [{"node": f"s{i // 2}", "head": nic / 2, "img": img / 24, "in": ct2 / 24} for i in range(8)]
    if not ct1_on_10g:
        for o in owners:                                          # cell-owner ingress share per node
            o["in"] += ct1 / n_cell_nodes / (2 if o["node"].startswith("s") else 1)
    corr_egress = ct1 / 16 + ct2 / 24
    r = summarise(kind, "two-CT" + (" (CT1 on 10G)" if ct1_on_10g else ""), csf, bits, img, owners,
                  corr_egress, 0, nic)
    r["ct1_total"], r["ct2_total"] = ct1, ct2
    r["ct1_per_node"] = ct1 / n_cell_nodes
    r["ct2_per_owner"] = ct2 / 24
    return r


def summarise(kind, scheme, csf, bits, img, owners, corr_egress, n_buckets, nic):
    per_node = {}
    for o in owners:
        per_node.setdefault(o["node"], 0.0)
        per_node[o["node"]] += o["in"]
    corr_in = max(v for k, v in per_node.items() if k.startswith("c")) + SNAP
    search_in = max(v for k, v in per_node.items() if k.startswith("s"))
    maximg = max(o["img"] for o in owners)
    return {"kind": kind, "scheme": scheme, "csf": csf, "bits": bits, "nic": nic,
            "images_per_s": img, "x_today": img / (200 * 1e6 / 1048.576), "n_buckets": n_buckets,
            "gpu_tot_half": img * 3.5e-6, "gpu_tot_full": img * 5.9e-6,
            "max_owner_load_half": maximg * 3.5e-6, "max_owner_load_full": maximg * 5.9e-6,
            "corr_egress": corr_egress, "corr_ingress_incl_snap": corr_in, "search_node_ingress": search_in,
            "fits_x4": corr_in <= 25.0 and search_in <= 25.0}


ROWS = []
for kind in ("1x", "2x", "4x"):
    for csf in (4, 2):
        ROWS.append(two_stage(kind, csf))
        ROWS.append(two_ct(kind, csf))
        ROWS.append(two_ct(kind, csf, ct1_on_10g=True))
        ROWS.append(two_ct(kind, csf, nic=50.0))
# per-node ingress breakdown for the network figure (cint4, 61 kHz)
BREAK = []
for kind in ("1x", "2x", "4x"):
    ts = two_stage(kind, 2)
    BREAK.append({"label": f"{kind} two-stage", "snap": SNAP, "ct1": 0.0,
                  "ct2": ts["corr_ingress_incl_snap"] - SNAP, "search": ts["search_node_ingress"]})
    tc = two_ct(kind, 2)
    BREAK.append({"label": f"{kind} two corner turns", "snap": SNAP, "ct1": tc["ct1_per_node"],
                  "ct2": tc["ct2_per_owner"], "search": tc["search_node_ingress"]})

if __name__ == "__main__":
    print(f"{'pkg':3s} {'scheme':22s} {'csf':>3s} {'nic':>4s} {'img/s':>8s} {'xT':>5s} {'GPU1/2':>6s} "
          f"{'maxOwn1/2':>9s} {'maxOwnFull':>10s} {'corrOut':>7s} {'corrIn':>6s} {'srchIn':>6s} fits")
    for r in ROWS:
        print(f"{r['kind']:3s} {r['scheme']:22s} {r['csf']:3d} {r['nic']:4.0f} {r['images_per_s']:8.2g} "
              f"{r['x_today']:5.1f} {r['gpu_tot_half']:6.1f} {r['max_owner_load_half']:9.2f} "
              f"{r['max_owner_load_full']:10.2f} {r['corr_egress']:7.1f} {r['corr_ingress_incl_snap']:6.1f} "
              f"{r['search_node_ingress']:6.1f} {'yes' if r['corr_ingress_incl_snap'] <= r['nic'] and r['search_node_ingress'] <= r['nic'] else 'NO'}")
    json.dump({"rows": ROWS, "breakdown": BREAK}, open("results/packages2.json", "w"), indent=1)
    for b in BREAK:
        print(b)
