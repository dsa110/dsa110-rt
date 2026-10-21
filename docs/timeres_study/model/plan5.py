"""4x feasibility v5: measured kernels AND measured network inputs -> results/plan5.json.

Changes vs plan4:
  * cells per stream measured from antenna positions (results/cells_per_stream.json,
    61 kHz gridding): FoV/2 2371/2146/1954/1847 for n_sub 1/2/4/8 (plan4 used 1900 flat);
  * owner us/image is rebuilt from the b3 components, with the combine scaled to the
    stream x cell count actually read (b3 timed 128 streams x 1900 cells = the n_sub 8 shape);
  * the search-node knee and the corr headroom are inputs (b5c / b5b measure them),
    and n01 can be dropped (it lost 88% of the corner turn at every rate in b5).
Network per tier: n_coarse = ceil(0.0206 (hi - lo) (T1/T) / n_sub) coarse DMs (stage-1
residual <= T across a sub-band), each = 16 corr nodes x n_sub streams x C cells x
(1/T) x 8 bit (cint4).  A coarse DM whose fine DMs exceed one GPU is split, and every
part receives the full stream (the corner turn is duplicated).
"""
import json
import math

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
T1, T2, T4 = 1048.576, 524.288, 262.144
CELLS = json.load(open("results/cells_per_stream.json"))
# b3 owner e2e, chunk detector, K7, F8 (8 fine DMs x 512 rows = 4096 images), ms
B3 = {"FoV/2": dict(combine=7.84, image=7.95, detect=3.77),
      "full": dict(combine=8.5, image=16.5, detect=7.5)}
B3_READS = 128 * 1900
AMB = {"2x": [(50, 1600, T2), (1600, 3000, T1)],
       "4x below DM 500": [(50, 500, T4), (500, 1600, T2), (1600, 3000, T1)],
       "4x below DM 800": [(50, 800, T4), (800, 1600, T2), (1600, 3000, T1)],
       "4x below DM 1200": [(50, 1200, T4), (1200, 1600, T2), (1600, 3000, T1)],
       "4x to DM 1600": [(50, 1600, T4), (1600, 3000, T1)],
       "4x to DM 3000": [(50, 3000, T4)]}


def cells(fov, n_sub):
    return CELLS[f"{fov}_nsub{n_sub}"]


def us_image(fov, n_sub, scale=1.0):
    b = B3[fov]
    comb = b["combine"] * 16 * n_sub * cells(fov, n_sub) / B3_READS
    return scale * 1e3 * (comb + b["image"] + b["detect"]) / 4096


def n_dm(lo, hi, T, csf=2, tol=1.25):
    return len(gen_dm_list(lo, hi, T, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol))


def items(amb, fov="FoV/2", n_sub=2, tol=1.4, scale=1.0, cap=0.8):
    us = us_image(fov, n_sub, scale)
    out = []
    for lo, hi, T in AMB[amb]:
        nf = n_dm(lo, hi, T, tol=tol)
        n_c = max(1, math.ceil(0.0206 * (hi - lo) * (T1 / T) / n_sub))
        gbps = 16 * n_sub * cells(fov, n_sub) * (1e6 / T) * 8 / 1e9
        gpu = nf / n_c * (1e6 / T) * us * 1e-6
        k = max(1, math.ceil(gpu / (0.9 * cap)))           # split an over-full coarse DM
        out += [{"gbps": gbps, "gpu": gpu / k, "T": T} for _ in range(n_c * k)]
    return out


def fits(its, H_c, H_s, n_search=4, cap_c=0.4, cap_s=0.8):
    """Owners: 16 corr GPUs (cap_c, net H_c each); n_search nodes x 2 GPUs sharing H_s."""
    own = [{"n": f"c{i}", "cap": cap_c, "gpu": 0.0} for i in range(16)]
    own += [{"n": f"s{i // 2}", "cap": cap_s, "gpu": 0.0} for i in range(2 * n_search)]
    net = {f"c{i}": H_c for i in range(16)}
    net.update({f"s{i}": H_s for i in range(n_search)})
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


def min_hc(its, H_s, n_search, cap_c=0.4):
    for h in [x * 0.25 for x in range(0, 161)]:
        if fits(its, h, H_s, n_search, cap_c):
            return h
    return float("inf")


if __name__ == "__main__":
    import sys
    H_S = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    rows = []
    print(f"search knee {H_S} Gb/s per node")
    print("us/image FoV/2:", {n: round(us_image('FoV/2', n), 2) for n in (1, 2, 4, 8)},
          " full:", {n: round(us_image('full', n), 2) for n in (1, 2, 4, 8)})
    for amb in AMB:
        for fov in ("FoV/2", "full"):
            for n_sub in (2, 4, 8):
                for tol in (1.25, 1.4):
                    for scale in (1.0, 0.7):
                        its = items(amb, fov, n_sub, tol, scale)
                        r = {"amb": amb, "fov": fov, "n_sub": n_sub, "tol": tol, "kernels": scale,
                             "ct_gbps": sum(i["gbps"] for i in its), "gpu_eq": sum(i["gpu"] for i in its),
                             "us_image": us_image(fov, n_sub, scale)}
                        for ns in (4, 3):
                            r[f"hc_min_s{ns}"] = min_hc(its, H_S, ns)
                            r[f"hc_min_s{ns}_merged"] = min_hc(its, H_S, ns, cap_c=0.8)
                        rows.append(r)
                        if tol == 1.4 and scale == 1.0:
                            print(f"{amb:18s} {fov:6s} n_sub {n_sub}  CT {r['ct_gbps']:6.1f} Gb/s  "
                                  f"GPU-eq {r['gpu_eq']:5.1f}  min corr headroom: 4 search "
                                  f"{r['hc_min_s4']:5.2f} / 3 search {r['hc_min_s3']:5.2f}  "
                                  f"(slow+fast merged {r['hc_min_s4_merged']:5.2f})")
    json.dump({"H_s": H_S, "rows": rows}, open("results/plan5.json", "w"), indent=1)
