"""Stitched all-owner DM-time waterfall at the burst pixel.

Concatenates the fine-DM grids of all 8 coarse-DM owners into one DM axis and
aligns their cube time axes via ``event_specnum_start`` (32 native spectra per
cube sample), so the full dispersion bowtie across every search GPU is visible
at a single sky pixel. Shows whether burst energy leaks into non-owning GPUs.

Usage:
  python _render_bowtie.py <injection_dir> <l_pix> <m_pix> <true_dm> <out_png> [title]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.preflight._inject_search_driver import (
    DEFAULT_DM_PLAN_PATH,
    owner_dm_grids,
)

SPECNUM_PER_SAMPLE = 32  # t_int_search 1048.576us / NATIVE 32.768us
DT_MS = 1048.576 / 1000.0


def robust_sigma(plane: np.ndarray) -> np.ndarray:
    """plane: (nfdm, t) -> per-row sigma using MAD over time."""
    out = np.empty_like(plane, dtype=np.float64)
    for i in range(plane.shape[0]):
        ts = plane[i].astype(np.float64)
        med = np.median(ts)
        mad = np.median(np.abs(ts - med)) * 1.4826
        if mad <= 0:
            mad = ts.std() or 1.0
        out[i] = (ts - med) / mad
    return out


def matched_sigma(plane: np.ndarray,
                  boxcars=(1, 2, 4, 8, 16, 32, 64, 128)) -> np.ndarray:
    """plane: (nfdm, t) -> per-row matched-boxcar detection sigma, i.e. the
    max over boxcar widths of the noise-normalised running boxcar sum (centred
    in time). This mirrors the detector's matched filter, so the bowtie tails
    of WIDE bursts (whose per-sample amplitude is far below threshold but whose
    boxcar-integrated SNR is large) become visible across non-owning owners."""
    nrow, T = plane.shape
    out = np.zeros((nrow, T), dtype=np.float64)
    for i in range(nrow):
        ts = plane[i].astype(np.float64)
        med = np.median(ts)
        mad = np.median(np.abs(ts - med)) * 1.4826
        if mad <= 0:
            mad = ts.std() or 1.0
        d = ts - med
        c = np.concatenate([[0.0], np.cumsum(d)])
        row = np.full(T, -np.inf)
        for b in boxcars:
            if b > T:
                break
            bsum = (c[b:] - c[:-b]) / (mad * np.sqrt(b))  # start-indexed, len T-b+1
            off = b // 2
            idx = np.arange(bsum.shape[0]) + off
            ok = idx < T
            np.maximum.at(row, idx[ok], bsum[ok])
        row[~np.isfinite(row)] = 0.0
        out[i] = row
    return out


def main() -> None:
    base = sys.argv[1]
    L, M = int(sys.argv[2]), int(sys.argv[3])
    true_dm = float(sys.argv[4])
    out_png = sys.argv[5]
    title = sys.argv[6] if len(sys.argv) > 6 else ""

    owners = []
    spec0 = None
    for o in range(8):
        fs = glob.glob("{}/owner{}/cube_*.npz".format(base, o))
        if not fs:
            continue
        d = np.load(fs[0])
        cube = d["cube"]
        spec = int(d["event_specnum_start"])
        spec0 = spec if spec0 is None else min(spec0, spec)
        g = owner_dm_grids(DEFAULT_DM_PLAN_PATH, o)
        fdm = np.asarray(g.fine_dm_local, float)
        order = np.argsort(fdm)
        plane = np.asarray(cube[:, :, L, M], np.float32).T  # (nfdm, t)
        plane = plane[order]
        fdm = fdm[order]
        S = matched_sigma(plane)
        owners.append(dict(o=o, spec=spec, fdm=fdm,
                           cdm=float(g.coarse_dm_local[0]), S=S))

    if not owners:
        print("no owner cubes found; skipping bowtie")
        return
    t_det = owners[0]["S"].shape[1]
    fig, (axw, axp) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw=dict(width_ratios=[3, 1], wspace=0.03))

    vmax = 8.0
    vmin = -3.0
    peak_dm, peak_sig = [], []
    for ow in owners:
        shift = (ow["spec"] - spec0) / SPECNUM_PER_SAMPLE
        t0 = shift * DT_MS
        t1 = (shift + t_det) * DT_MS
        dlo, dhi = ow["fdm"][0], ow["fdm"][-1]
        axw.imshow(ow["S"], origin="lower", aspect="auto", cmap="magma",
                   extent=[t0, t1, dlo, dhi], vmin=vmin, vmax=vmax)
        axw.axhline(dhi, color="cyan", lw=0.5, alpha=0.5)
        axw.text(t1 * 0.995, 0.5 * (dlo + dhi), "GPU{}".format(ow["o"]),
                 color="cyan", fontsize=8, va="center", ha="right")
        peak_dm.append(ow["fdm"])
        peak_sig.append(ow["S"].max(axis=1))

    axw.axhline(true_dm, color="lime", ls="--", lw=1.0)
    axw.text(axw.get_xlim()[0], true_dm, " true DM={:.0f}".format(true_dm),
             color="lime", fontsize=9, va="bottom")
    axw.set_xlabel("time (ms, common frame)")
    axw.set_ylabel("DM (pc cm$^{-3}$)")
    axw.set_title("all-owner DM-time at pixel ({},{})  "
                  "[matched-boxcar sigma, color clipped at {:.0f}]"
                  .format(L, M, vmax))

    pd = np.concatenate(peak_dm)
    ps = np.concatenate(peak_sig)
    order = np.argsort(pd)
    axp.plot(ps[order], pd[order], color="0.2", lw=1.0)
    axp.axhline(true_dm, color="lime", ls="--", lw=1.0)
    axp.axvline(5.0, color="red", ls=":", lw=0.8)
    axp.set_xlabel("peak sigma over time")
    axp.set_xscale("log")
    axp.set_ylim(axw.get_ylim())
    axp.set_yticklabels([])
    axp.set_title("bowtie envelope")
    axp.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13, y=0.95)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print("wrote", out_png)
    print("peak sigma per owner:",
          {ow["o"]: round(float(ow["S"].max()), 1) for ow in owners})


if __name__ == "__main__":
    main()
