"""Package-level model: DM tiers, imaging load, network per node, rate gains.

Network units Gb/s; bits = bits per complex uv value (cint4 = 8).
Cells: 1748 per single channel, 1900 per sub-band stream (n_sub 8), 6000 in a
dedispersed full-band plane (nfilled.py).  SNAP ingress 18 Gb/s on corr nodes.
NIC: ConnectX-4 in an x4 slot (~25 Gb/s usable, both ports together) today;
~50 Gb/s if moved to one of the free x16 slots (x8 card).
"""
import json

import numpy as np

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
N_CORR, N_SEARCH = 16, 4
SNAP = 18.0
NIC_X4, NIC_X8 = 25.0, 50.0
CELLS_CH, CELLS_SB, CELLS_PLANE = 1748, 1900, 6000
# per-image GPU cost (us) for rewritten owner kernels, FoV/2 (256x128) | full FoV
COST_TS = {"half": 3.5, "full": 5.9}        # combine + FFT + detector + norm
COST_TC = {"half": 3.5, "full": 5.9}        # FDMT at cell owner ~= combine; FFT + det at DM owner


def n_dm(lo, hi, T, csf, tol=1.25):
    return len(gen_dm_list(lo, hi, T, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol))


def tiers(kind):
    """(dm_lo, dm_hi, T_us) tiers for each sampling ambition; DM 50-3000."""
    return {"1x": [(50, 3000, 1048.576)],
            "2x": [(50, 1600, 524.288), (1600, 3000, 1048.576)],
            "4x": [(50, 800, 262.144), (800, 1600, 524.288), (1600, 3000, 1048.576)]}[kind]


def package(kind, scheme, csf=2, bits=8, n_owner=24, ct1_T=None, nsub=8, ncoarse_per_tier=None):
    tl = tiers(kind)
    img = 0.0
    ndm_tot = 0
    for lo, hi, T in tl:
        n = n_dm(lo, hi, T, csf)
        ndm_tot += n
        img += n * 1e6 / T
    gb = lambda vals_per_s: vals_per_s * bits / 1e9            # noqa: E731
    res = {"kind": kind, "scheme": scheme, "csf": csf, "bits": bits, "n_dm": ndm_tot,
           "images_per_s": img, "x_today": img / (200 * 1e6 / 1048.576)}
    if scheme == "two-stage":
        # coarse buckets: product rule n_c*n_s >= ~0.0824*DM_range per tier at 262 us, scaled by T
        n_c_tot = 0
        ct = 0.0
        for lo, hi, T in tl:
            need = 0.0206 * (hi - lo) * (1048.576 / T)            # n_c*n_s for <=1 sample smear
            n_c = max(1, int(np.ceil(need / nsub)))
            n_c_tot += n_c
            ct += n_c * nsub * CELLS_SB * N_CORR * 1e6 / T        # all corr -> owners
        res.update(n_coarse=n_c_tot, ct_total=gb(ct), corr_egress=gb(ct) / N_CORR,
                   owner_ingress=gb(ct) / n_owner, ct1_total=0.0)
    elif scheme == "two-CT":
        T1 = ct1_T or min(T for _, _, T in tl)
        ct1 = N_CORR * (384 // csf) * CELLS_CH * 1e6 / T1           # per-channel cells, once
        ct2 = img * CELLS_PLANE                                    # dedispersed planes
        res.update(ct1_total=gb(ct1), ct_total=gb(ct1 + ct2), corr_egress=gb(ct1) / N_CORR + gb(ct2) / n_owner,
                   owner_ingress=gb(ct1) / n_owner + gb(ct2) / n_owner)
    elif scheme == "one-shot multicast":
        T = min(T for _, _, T in tl)
        one = (384 // csf) * CELLS_CH * 1e6 / T
        res.update(ct1_total=0.0, ct_total=gb(N_CORR * one), corr_egress=gb(one),
                   owner_ingress=gb(N_CORR * one))                 # every owner needs the whole band
    for fov, c in (("half", COST_TS), ("full", COST_TS)):
        res[f"gpu_{fov}"] = img * c[fov] / 1e6
    res["corr_ingress_with_snap"] = SNAP + res["owner_ingress"]
    return res


PKGS = []
for kind in ("1x", "2x", "4x"):
    for scheme in ("two-stage", "two-CT", "one-shot multicast"):
        for csf in (4, 2):
            PKGS.append(package(kind, scheme, csf=csf, bits=8))

if __name__ == "__main__":
    print(f"{'pkg':4s} {'scheme':19s} {'csf':>3s} {'nDM':>5s} {'img/s':>8s} {'xT':>5s} {'GPU1/2':>6s} "
          f"{'GPUfull':>7s} {'CT tot':>7s} {'corr out':>8s} {'owner in':>8s} {'corr in+SNAP':>12s}")
    for p in PKGS:
        print(f"{p['kind']:4s} {p['scheme']:19s} {p['csf']:3d} {p['n_dm']:5d} {p['images_per_s']:8.2g} "
              f"{p['x_today']:5.1f} {p['gpu_half']:6.1f} {p['gpu_full']:7.1f} {p['ct_total']:7.0f} "
              f"{p['corr_egress']:8.1f} {p['owner_ingress']:8.1f} {p['corr_ingress_with_snap']:12.1f}")
    json.dump(PKGS, open("results/packages.json", "w"), indent=1)
