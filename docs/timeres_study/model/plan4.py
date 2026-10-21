"""4x feasibility on MEASURED costs (bench/timeres, 2026-10-21 gap) -> results/plan4.json.

Measured (n01/n04 GPU 0, prototype kernels, isolated):
  owner chain us/image  256x128 K7: 4.78   256x256 K7: 7.6-8.3   (combine 128 streams + irfft2 + detector)
  corr fast path at 0.26 ms: fast GEMM 23.4 ms + stage-1 (31 coarse x 2 sub-bands, 61 kHz) 35.7 ms per block
Imaging owners: 8 search GPUs (cap 0.8) + 16 corr GPUs with corr_slow resident (cap ~0.4) or
freed by an X-engine merge (cap 0.8); corr-node ingress headroom beside SNAP = H_c (b5).
Two-stage, streams at the tier cadence, n_sub = 2 buckets.
"""
import json
import math

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
CELLS_SB, BITS = 1900, 8
T1, T2, T4 = 1048.576, 524.288, 262.144
US = {"FoV/2": 4.78, "full": 8.0}


def n_dm(lo, hi, T, csf=2, tol=1.25):
    return len(gen_dm_list(lo, hi, T, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol))


def plan(edgeA=800, tol=1.25, csfA=2, fov="FoV/2", us_scale=1.0):
    tiers = [(50, edgeA, T4, csfA), (edgeA, 1600, T2, 2), (1600, 3000, T1, 2)]
    us = US[fov] * us_scale
    img = gbps = 0.0
    buckets = []
    for lo, hi, T, csf in tiers:
        nf = n_dm(lo, hi, T, csf, tol)
        need = 0.0206 * (hi - lo) * (T1 / T)
        n_c = max(1, math.ceil(need / 2))
        for _ in range(n_c):
            buckets.append({"gpu": nf / n_c * 1e6 / T * us * 1e-6,
                            "gbps": 16 * 2 * CELLS_SB * 1e6 / T * BITS / 1e9})
        img += nf * 1e6 / T
    return img, buckets


def fits(buckets, H_c, H_s=17.0, corr_cap=0.4, search_cap=0.8):
    own = [{"k": "c", "net": H_c, "cap": corr_cap, "gpu": 0.0, "in": 0.0} for _ in range(16)]
    own += [{"k": "s", "net": H_s / 2, "cap": search_cap, "gpu": 0.0, "in": 0.0} for _ in range(8)]
    for b in sorted(buckets, key=lambda x: -x["gpu"]):
        room = [o for o in own if o["in"] + b["gbps"] <= o["net"] + 1e-9 and o["gpu"] + b["gpu"] <= o["cap"] + 1e-9]
        if not room:
            return False, max(o["gpu"] for o in own)
        o = min(room, key=lambda x: x["gpu"] / x["cap"])
        o["gpu"] += b["gpu"]
        o["in"] += b["gbps"]
    return True, max(o["gpu"] for o in own)


if __name__ == "__main__":
    rows = []
    variants = [("4x, DM<800, tol 1.25, 61 kHz", dict(edgeA=800)),
                ("4x, DM<800, tol 1.4", dict(edgeA=800, tol=1.4)),
                ("4x, DM<800, 122 kHz low tier", dict(edgeA=800, csfA=4)),
                ("4x, DM<500", dict(edgeA=500)),
                ("4x, DM<500, tol 1.4", dict(edgeA=500, tol=1.4)),
                ("4x to DM 1600, tol 1.4", dict(edgeA=1600, tol=1.4)),
                ("4x to DM 1600, tol 1.25", dict(edgeA=1600)),
                ("2x (DM<1600 at 0.52 ms)", dict(edgeA=50))]
    for name, kw in variants:
        for fov in ("FoV/2", "full"):
            for scale, slab in ((1.0, "measured"), (0.7, "x0.7 optimised")):
                img, bk = plan(fov=fov, us_scale=scale, **kw)
                need = img * US[fov] * scale * 1e-6
                res = {}
                for H_c, ccap, lab in ((0.0, 0.4, "Hc0"), (2.0, 0.4, "Hc2"), (4.0, 0.4, "Hc4"),
                                       (2.0, 0.8, "Hc2_merged"), (30.0, 0.8, "stretch")):
                    ok, mx = fits(bk, H_c, H_s=40.0 if lab == "stretch" else 17.0, corr_cap=ccap)
                    res[lab] = {"fits": ok, "max_gpu": round(mx, 2)}
                rows.append({"variant": name, "fov": fov, "kernels": slab, "images_per_s": img,
                             "gpu_equiv": need, **res})
                print(f"{name:32s} {fov:6s} {slab:15s} img/s {img:8.3g} GPU-eq {need:5.1f}  " +
                      "  ".join(f"{k}:{'Y' if v['fits'] else 'n'}" for k, v in res.items()))
    json.dump(rows, open("results/plan4.json", "w"), indent=1)
