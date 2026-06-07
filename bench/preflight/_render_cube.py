"""Render a recovered injection cube the way tools/viz/cube_burst_explorer.ipynb
does (3 panels), but reading our grid-runner C1 CSV directly so the real DM axis
and detected trial are shown. Standalone (no dsart dep)."""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mmap_cube(npz_path: str, key: str = "cube") -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as d:
        return np.asarray(d[key])


def scalar(npz_path: str, key: str, default=None):
    try:
        d = np.load(npz_path, allow_pickle=False)
        if key in d:
            return d[key].item() if d[key].shape == () else d[key]
    except Exception:
        pass
    return default


def read_c1(csv_path: str) -> list[dict]:
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def baseline_sigma(x: np.ndarray, t_burst: int, guard: int) -> np.ndarray:
    x = x.astype(np.float64)
    mask = np.ones(x.size, bool)
    a = max(0, t_burst - guard)
    b = min(x.size, t_burst + guard + 1)
    mask[a:b] = False
    base = x[mask]
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826
    if mad <= 0:
        mad = np.std(base) or 1.0
    return (x - med) / mad


def boxcar(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    cube = mmap_cube(args.cube)
    t_det, n_fdm, n_grid, _ = cube.shape
    tsamp_ms = float(scalar(args.cube, "sample_period_us", 1048.576)) / 1000.0

    rows = read_c1(args.c1)
    rows.sort(key=lambda r: -float(r["snr"]))
    top = rows[0]
    det_fdm = int(top["fine_dm_idx"])
    bright_l = int(top["l_pix"])
    bright_m = int(top["m_pix"])
    t_burst = int(top["t_in_cube"])
    det_dm = float(top["dm_pc_cc"])

    # real DM axis from C1 rows (fdm idx -> dm)
    fdm_dm = {int(r["fine_dm_idx"]): float(r["dm_pc_cc"]) for r in rows}
    known = sorted(fdm_dm)
    if len(known) >= 2:
        dm_axis = np.interp(np.arange(n_fdm), known, [fdm_dm[k] for k in known])
        dm_known = True
    else:
        dm_axis = np.arange(n_fdm, dtype=float)
        dm_known = False

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.2, 1.4], hspace=0.35)

    # ---- Panel 1: image time-series around the burst ----
    ext = 12
    rr0, rr1 = max(0, bright_l - ext), min(n_grid, bright_l + ext + 1)
    cc0, cc1 = max(0, bright_m - ext), min(n_grid, bright_m + ext + 1)
    n_img = 7
    stride = 4
    mid = n_img // 2
    gtop = gs[0].subgridspec(1, n_img, wspace=0.08)
    frames, labels = [], []
    for k in range(n_img):
        fs = t_burst + (k - mid) * stride - stride // 2
        a, b = max(0, fs), min(t_det, fs + stride)
        if b <= a:
            fr = np.full((rr1 - rr0, cc1 - cc0), np.nan, np.float32)
        else:
            fr = np.asarray(cube[a:b, det_fdm, rr0:rr1, cc0:cc1], np.float32).mean(0)
        frames.append(fr)
        labels.append(f"{(fs + stride/2 - t_burst) * tsamp_ms:+.1f} ms")
    stack = np.array(frames)
    fin = stack[np.isfinite(stack)]
    vmin, vmax = np.percentile(fin, [1, 99.7]) if fin.size else (0, 1)
    for k in range(n_img):
        ax = fig.add_subplot(gtop[0, k])
        ax.imshow(frames[k], origin="lower", vmin=vmin, vmax=vmax, cmap="magma",
                  extent=[cc0, cc1, rr0, rr1])
        ax.set_title(labels[k], fontsize=9)
        ax.plot(bright_m, bright_l, "c+", ms=10, mew=1.2)
        if k:
            ax.set_yticklabels([])
    fig.text(0.5, 0.905, f"Panel 1 — image frames at fdm={det_fdm} (DM={det_dm:.0f}), pixel ({bright_l},{bright_m})",
             ha="center", fontsize=11)

    # ---- Panel 2: multi-DM light curves at brightest pixel (sigma) ----
    axb = fig.add_subplot(gs[1])
    n_dms = min(9, n_fdm)
    start = int(np.clip(det_fdm - n_dms // 2, 0, max(0, n_fdm - n_dms)))
    fdm_idxs = np.arange(start, start + n_dms)
    t_ms = np.arange(t_det) * tsamp_ms
    lcs = np.asarray(cube[:, fdm_idxs, bright_l, bright_m], np.float32)
    sig = np.stack([boxcar(baseline_sigma(lcs[:, j], t_burst, 12), 1) for j in range(lcs.shape[1])], 1)
    off = max(1.15 * float(np.nanmax(np.abs(sig))), 4.0)
    for j, fi in enumerate(fdm_idxs):
        lbl = f"DM={dm_axis[fi]:.0f}" if dm_known else f"fdm={fi}"
        c = "crimson" if fi == det_fdm else "0.4"
        axb.plot(t_ms, sig[:, j] + j * off, color=c, lw=1.2 if fi == det_fdm else 0.8)
        axb.text(t_ms[-1] * 1.005, j * off, lbl, fontsize=8, color=c, va="center")
    axb.axvline(t_burst * tsamp_ms, color="c", ls=":", lw=1)
    axb.set_xlabel("time (ms)"); axb.set_ylabel("σ (offset per DM)")
    axb.set_title("Panel 2 — light curves vs DM trial at brightest pixel")

    # ---- Panel 3: detected-DM time series + DM-time waterfall ----
    gbot = gs[2].subgridspec(2, 1, height_ratios=[1, 2.4], hspace=0.05)
    ax_ts = fig.add_subplot(gbot[0])
    lc_det = boxcar(baseline_sigma(np.asarray(cube[:, det_fdm, bright_l, bright_m], np.float32), t_burst, 12), 1)
    ax_ts.plot(t_ms, lc_det, color="crimson", lw=1.2)
    ax_ts.axvline(t_burst * tsamp_ms, color="c", ls=":", lw=1)
    ax_ts.set_ylabel("σ"); ax_ts.set_xticklabels([])
    ax_ts.set_title(f"Panel 3 — detected DM={det_dm:.0f} time series (top) + DM–time waterfall (bottom)")

    ax_wf = fig.add_subplot(gbot[1])
    M = np.asarray(cube[:, :, bright_l, bright_m], np.float32).T  # (n_fdm, t_det)
    M = np.stack([baseline_sigma(M[i], t_burst, 12) for i in range(M.shape[0])], 0)
    extent = [0, t_det * tsamp_ms, dm_axis[0], dm_axis[-1]]
    im = ax_wf.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                      extent=extent, vmin=np.percentile(M, 2), vmax=np.percentile(M, 99.8))
    ax_wf.axhline(det_dm, color="w", ls="--", lw=0.8)
    ax_wf.axvline(t_burst * tsamp_ms, color="w", ls=":", lw=0.8)
    ax_wf.set_xlabel("time (ms)"); ax_wf.set_ylabel("DM (pc cm⁻³)" if dm_known else "fdm idx")
    fig.colorbar(im, ax=ax_wf, label="σ", pad=0.01)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=0.97)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
