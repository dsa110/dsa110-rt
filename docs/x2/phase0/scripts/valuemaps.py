"""Real contribution counts n per transmitted value, for every corner-turn format (Phase 0, shared).

Production conventions (grid/sparsity_pattern.build_pattern as deployed, cell_lambda_mode tee45):
  * antenna E/N from antennas.out (== _inspect/sensitivity/pipeline_antpos.npz), heights from the
    station survey (station_enu.npz; the core is flat to 1 cm, so 3-D and 2-D classes agree);
  * core = station number <= 102 (82 antennas, 3321 cross-baselines), xGPU upper-tri order b <= a,
    du = E[a] - E[b], dv = N[a] - N[b];
  * u = -du / lambda, v = -dv / lambda (F20 negation), NO declination projection, single-sided grid
    (no Hermitian fold on the wire), col = rint(u / cell) + 128, row = rint(v / cell) + 128,
    cell = 1 / (256 * 45 arcsec) = 17.90 lambda, cells outside [0, 256) dropped;
  * summed channel frequency = mean of its csf native channels (F33 band-centre convention);
    sub-band s of a chgroup = summed channels [s * per, (s + 1) * per).

Value kinds (docs/x2/SPEC.md section 3):
  gridded    one value per occupied cell of a (chgroup, sub-band) stream; n = number of
             (baseline, summed channel) sources landing in it;
  baselines  one value per exact 3-D redundancy class (5 cm) per stream, summed over the sub-band's
             channels; n = multiplicity x channels per sub-band; placed at the cell of the class uv
             at the sub-band CENTRE frequency (dropped if that cell is off the grid).
N_image = total contributions per image over all streams; heavy values have n >= N_image / 1e4.

    python valuemaps.py            -> results/valuemaps.json (+ results/valuemaps_<fmt>.npz)
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SENS = "/home/ubuntu/vikram/dev/_inspect/sensitivity/"

C_M_S = 299792458.0
NU_TOP_GHZ = 1.49875
DNU_GHZ = 250.0 / 8192.0 / 1e3
N_CHG, N_NAT = 16, 384
N_GRID = 256
CELL = 1.0 / (N_GRID * 45.0 * np.pi / 180.0 / 3600.0)          # 17.904 lambda
MAX_CORE = 102
HEAVY_FRAC = 1e-4

FORMATS = {                   # key: (kind, csf, n_sub)
    "today_g_ns4_csf8": ("gridded", 8, 4),
    "g_ns2": ("gridded", 2, 2),
    "g_ns4": ("gridded", 2, 4),
    "b_ns2": ("baselines", 2, 2),
    "b_ns4": ("baselines", 2, 4),
}


def antennas():
    """(E, N, U, station) of the 96 pipeline antennas, metres."""
    a = np.fromfile(SENS + "antennas.out", dtype="<f4")
    E, N = a[:96].astype(np.float64), a[96:192].astype(np.float64)
    order = np.load(SENS + "antenna_order.npy", allow_pickle=True).astype(int)
    st = np.load(SENS + "station_enu.npz")
    idx = {int(s): i for i, s in enumerate(st["station"])}
    U = np.array([st["U"][idx[int(s)]] for s in order])
    return E, N, U, order


def core_baselines():
    """du, dv, dw (m) of the core cross-baselines in xGPU upper-tri order (a > b)."""
    E, N, U, order = antennas()
    a_l, b_l = [], []
    for a in range(96):
        for b in range(a):
            if order[a] <= MAX_CORE and order[b] <= MAX_CORE:
                a_l.append(a)
                b_l.append(b)
    a_l, b_l = np.array(a_l), np.array(b_l)
    return E[a_l] - E[b_l], N[a_l] - N[b_l], U[a_l] - U[b_l], a_l, b_l


def summed_freqs_ghz(csf):
    """(16, 384 / csf) band-centre frequencies of the summed channels."""
    i = np.arange(N_CHG * N_NAT)
    f = NU_TOP_GHZ - i * DNU_GHZ
    return f.reshape(N_CHG, N_NAT // csf, csf).mean(axis=2)


def cell_of(du, dv, f_ghz):
    """(row, col, ok) of baseline(s) at frequency f (broadcasting), production convention."""
    lam = C_M_S / (np.asarray(f_ghz) * 1e9)
    col = np.rint(-du / lam / CELL).astype(np.int64) + N_GRID // 2
    row = np.rint(-dv / lam / CELL).astype(np.int64) + N_GRID // 2
    ok = (col >= 0) & (col < N_GRID) & (row >= 0) & (row < N_GRID)
    return row, col, ok


def redundancy_classes(tol=0.05):
    """class index per core baseline (3-D positions within tol, sign-canonical), conj flag, mult."""
    du, dv, dw, _, _ = core_baselines()
    q = np.stack([np.rint(du / tol), np.rint(dv / tol), np.rint(dw / tol)], 1).astype(np.int64)
    flip = (q[:, 1] < 0) | ((q[:, 1] == 0) & (q[:, 0] < 0))
    q[flip] *= -1
    uq, inv, mult = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    return inv.ravel(), flip, mult


def build(kind, csf, n_sub):
    """List of streams; each stream = dict(g, s, n (values,), row, col) plus totals."""
    du, dv, dw, _, _ = core_baselines()
    nus = summed_freqs_ghz(csf)
    per = nus.shape[1] // n_sub
    streams = []
    if kind == "gridded":
        for g in range(N_CHG):
            for s in range(n_sub):
                f = nus[g, s * per:(s + 1) * per]
                row, col, ok = cell_of(du[:, None], dv[:, None], f[None, :])
                key = (row * N_GRID + col)[ok]
                cells, n = np.unique(key, return_counts=True)
                streams.append(dict(g=g, s=s, n=n.astype(np.int64), row=cells // N_GRID,
                                    col=cells % N_GRID, n_dropped=int((~ok).sum())))
    else:
        cls, flip, mult = redundancy_classes()
        # representative uv of each class: the canonical (unflipped) member geometry
        rep_du = np.zeros(len(mult))
        rep_dv = np.zeros(len(mult))
        sgn = np.where(flip, -1.0, 1.0)
        np.add.at(rep_du, cls, sgn * du)
        np.add.at(rep_dv, cls, sgn * dv)
        rep_du /= mult
        rep_dv /= mult
        for g in range(N_CHG):
            for s in range(n_sub):
                fc = nus[g, s * per:(s + 1) * per].mean()
                row, col, ok = cell_of(rep_du, rep_dv, fc)
                n = (mult * per).astype(np.int64)
                streams.append(dict(g=g, s=s, n=n[ok], row=row[ok], col=col[ok], mult=mult[ok],
                                    n_dropped=int((mult[~ok] * per).sum())))
    return streams


def summarise(streams):
    n_all = np.concatenate([st["n"] for st in streams])
    N = int(n_all.sum())
    thr = HEAVY_FRAC * N
    heavy = n_all >= thr
    per_stream = np.array([len(st["n"]) for st in streams])
    heavy_ps = np.array([int((st["n"] >= thr).sum()) for st in streams])
    return dict(N_image=N, n_streams=len(streams), values_total=int(n_all.size),
                values_per_stream_mean=float(per_stream.mean()),
                values_per_stream_max=int(per_stream.max()),
                n_max=int(n_all.max()), n_median=float(np.median(n_all)),
                heavy_threshold=float(thr), heavy_values_total=int(heavy.sum()),
                heavy_frac_values=float(heavy.mean()),
                heavy_frac_weight=float(n_all[heavy].sum() / N),
                heavy_per_stream_mean=float(heavy_ps.mean()), heavy_per_stream_max=int(heavy_ps.max()),
                dropped_contrib=int(sum(st["n_dropped"] for st in streams)))


def load(fmt):
    """Cached per-format arrays: n_all, stream index per value, row, col (and the stream table)."""
    path = os.path.join(RES, f"valuemaps_{fmt}.npz")
    if not os.path.exists(path):
        save(fmt)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def save(fmt):
    kind, csf, n_sub = FORMATS[fmt]
    streams = build(kind, csf, n_sub)
    sid = np.concatenate([np.full(len(st["n"]), i) for i, st in enumerate(streams)])
    np.savez_compressed(os.path.join(RES, f"valuemaps_{fmt}.npz"),
                        n=np.concatenate([st["n"] for st in streams]), stream=sid,
                        row=np.concatenate([st["row"] for st in streams]),
                        col=np.concatenate([st["col"] for st in streams]),
                        g=np.array([st["g"] for st in streams]), s=np.array([st["s"] for st in streams]),
                        csf=csf, n_sub=n_sub, kind=kind)
    return streams


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    du, dv, dw, _, _ = core_baselines()
    cls, flip, mult = redundancy_classes()
    out = {"n_core_baselines": int(len(du)), "unique_3d_5cm": int(len(mult)),
           "mult_max": int(mult.max()), "singletons": int((mult == 1).sum()),
           "cell_lambda": CELL}
    print(out, flush=True)
    for fmt in FORMATS:
        s = summarise(save(fmt))
        out[fmt] = s
        print(fmt, json.dumps(s), flush=True)
    json.dump(out, open(os.path.join(RES, "valuemaps.json"), "w"), indent=1)
