"""Distinct single-side uv cells per sub-band stream from real antenna positions.

Each csf-8 channel of the sub-band is gridded (pillbox, nearest cell) onto
one lambda-uniform plane; the stream carries the union of hit cells.
"""
import numpy as np

from physics import FREQS_MHZ

SENS = "/home/ubuntu/vikram/dev/_inspect/sensitivity/"
a = np.fromfile(SENS + "antennas.out", dtype="<f4")
E, N = a[:96].astype(float), a[96:192].astype(float)
order = np.load(SENS + "antenna_order.npy", allow_pickle=True)
core = np.where(order <= 102)[0]
ia, ib = np.triu_indices(len(core), k=1)
BE = E[core][ib] - E[core][ia]
BN = N[core][ib] - N[core][ia]
C = 299792458.0
COSZ = np.cos(np.radians(71.63 - 37.234))


def cells(f_mhz_list, cell_lambda=17.9, cosz=COSZ, fov_scale=1.0):
    cl = cell_lambda * fov_scale
    s = set()
    for f in f_mhz_list:
        lam = C / (f * 1e6)
        u = BE / lam / cl
        v = BN * cosz / lam / cl
        flip = (v < 0) | ((v == 0) & (u < 0))
        u = np.where(flip, -u, u)
        v = np.where(flip, -v, v)
        s.update(zip(np.rint(u).astype(int).tolist(), np.rint(v).astype(int).tolist()))
    return len(s)


def per_stream(g, nsub, fov_scale=1.0, csf=8):
    ch = FREQS_MHZ[g * 384:(g + 1) * 384]
    per = 384 // nsub
    return float(np.mean([cells(ch[k * per:(k + 1) * per:csf], fov_scale=fov_scale)
                          for k in range(nsub)]))


if __name__ == "__main__":
    print("core ants", len(core), "baselines", len(BE), "max |b| m", np.hypot(BE, BN).max())
    for g in (0, 8, 15):
        row = []
        for nsub in (1, 4, 8, 16):
            row.append(f"nsub{nsub}: {per_stream(g, nsub):6.0f} "
                       f"(FoV/2 {per_stream(g, nsub, np.sqrt(2)):5.0f})")
        print(f"chgroup {g:2d}  ", "  ".join(row))
