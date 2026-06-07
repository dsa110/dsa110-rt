"""Manual fast-pipeline (corr-fast coarse-DM) vs real-time detector comparison.

Images the corr-fast owner-stream products (coarse-DM owner, summed over the 16
sub-band chgroups) on CPU and compares the recovered source against the
DeterministicDetector C1 output + cube for the same injection cell.

Cadence note: corr-fast dedispersed sample period == NATIVE_SAMPLE_US * 32 ==
search/detector cadence, so both light curves share the same dt.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
sys.path.insert(0, ".")

from dsart.services.corr_fast_integration import NATIVE_SAMPLE_US  # noqa: E402
from bench._corr_fast_replay import (  # noqa: E402
    dirty_image_from_dense_grid,
    sparse_to_dense_grid,
)

DT_MS = NATIVE_SAMPLE_US * 32.0 / 1000.0


def read_c1(csv_path: str) -> list[dict]:
    with open(csv_path) as fh:
        return sorted(csv.DictReader(fh), key=lambda r: -float(r["snr"]))


def image_corrfast(corr_dir: str, n_grid: int) -> np.ndarray:
    """Sum dense grids over all chgroups for the owner, return (T, N, N) f32 image."""
    files = sorted(glob.glob(str(Path(corr_dir) / "corr_out_g*.npz")))
    if not files:
        raise FileNotFoundError(f"no corr_out_g*.npz in {corr_dir}")
    dense_total = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        sp = torch.from_numpy(np.ascontiguousarray(d["owner_stream"]))  # (T, N_filled)
        ix_row = np.asarray(d["ix_row"])
        ix_col = np.asarray(d["ix_col"])
        dense = sparse_to_dense_grid(sp, ix_row, ix_col, n_grid)  # (T, N, N) complex
        dense_total = dense if dense_total is None else dense_total + dense
    img = dirty_image_from_dense_grid(dense_total)  # (T, N, N) f32
    img = img.cpu().numpy()
    # Zero-DM filter: subtract per-pixel temporal mean to remove the static DC /
    # phase-center term (matches the detector-side zero_dm_filter), exposing the
    # transient burst.
    img = img - img.mean(axis=0, keepdims=True)
    return img


def robust_sigma_lc(lc: np.ndarray, peak_t: int, guard: int = 12) -> np.ndarray:
    lc = lc.astype(np.float64)
    mask = np.ones(lc.size, bool)
    a, b = max(0, peak_t - guard), min(lc.size, peak_t + guard + 1)
    mask[a:b] = False
    base = lc[mask]
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826 or (np.std(base) or 1.0)
    return (lc - med) / mad


def fwhm_ms(sig: np.ndarray, peak_t: int) -> float:
    half = sig[peak_t] / 2.0
    if half <= 0:
        return float("nan")
    lo = peak_t
    while lo > 0 and sig[lo] > half:
        lo -= 1
    hi = peak_t
    while hi < sig.size - 1 and sig[hi] > half:
        hi += 1
    return (hi - lo) * DT_MS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True, help="grid cell dir (has corr_work/, cube, c1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-grid", type=int, default=256)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    cell = Path(args.cell_dir)
    c1_rows = read_c1(str(cell / "candidates_c1.csv"))
    top = c1_rows[0]
    det_fdm = int(top["fine_dm_idx"])
    det_l, det_m = int(top["l_pix"]), int(top["m_pix"])
    det_dm = float(top["dm_pc_cc"])
    det_snr = float(top["snr"])
    det_box = int(top["width_samples"])

    cube_path = sorted(glob.glob(str(cell / "cube_s*.npz")))[0]
    with np.load(cube_path, allow_pickle=False) as d:
        cube = np.asarray(d["cube"])  # (t_det, n_fdm, N, N) fp16
    t_det = cube.shape[0]

    # ---- corr-fast coarse-DM image ----
    img = image_corrfast(str(cell / "corr_work"), args.n_grid)  # (T, N, N)
    T = img.shape[0]
    # peak pixel over the brightest frame
    peak_frame = int(np.argmax(img.reshape(T, -1).max(axis=1)))
    cf_l, cf_m = np.unravel_index(int(np.argmax(img[peak_frame])), img[peak_frame].shape)
    cf_lc = img[:, cf_l, cf_m]
    cf_sig = robust_sigma_lc(cf_lc, peak_frame)
    cf_snr = float(cf_sig[peak_frame])
    cf_fwhm = fwhm_ms(cf_sig, peak_frame)

    # ---- detector light curve at detected DM + pixel ----
    det_lc = np.asarray(cube[:, det_fdm, det_l, det_m], np.float32)
    det_peak_t = int(np.argmax(det_lc))
    det_sig = robust_sigma_lc(det_lc, det_peak_t)
    det_fwhm = fwhm_ms(det_sig, det_peak_t)

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.subplots_adjust(wspace=0.28)

    # (a) corr-fast coarse image at peak
    ext = 16
    rr0, rr1 = max(0, cf_l - ext), min(args.n_grid, cf_l + ext + 1)
    cc0, cc1 = max(0, cf_m - ext), min(args.n_grid, cf_m + ext + 1)
    a0 = axes[0]
    im0 = a0.imshow(img[peak_frame, rr0:rr1, cc0:cc1], origin="lower", cmap="magma",
                    extent=[cc0, cc1, rr0, rr1])
    a0.plot(cf_m, cf_l, "c+", ms=12, mew=1.5, label=f"corr-fast peak ({cf_l},{cf_m})")
    a0.plot(det_m, det_l, "wx", ms=9, mew=1.5, label=f"detector C1 ({det_l},{det_m})")
    a0.legend(fontsize=8, loc="upper right")
    a0.set_title(f"(a) corr-fast coarse-DM image\n(owner→576 pc/cc), peak frame {peak_frame}")
    fig.colorbar(im0, ax=a0, fraction=0.046, pad=0.04)

    # (b) detector cube peak frame at detected DM
    a1 = axes[1]
    dframe = np.asarray(cube[det_peak_t, det_fdm, rr0:rr1, cc0:cc1], np.float32)
    im1 = a1.imshow(dframe, origin="lower", cmap="magma", extent=[cc0, cc1, rr0, rr1])
    a1.plot(det_m, det_l, "cx", ms=10, mew=1.5)
    a1.set_title(f"(b) detector cube frame\nfine DM={det_dm:.0f} (fdm {det_fdm}), t={det_peak_t}")
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)

    # (c) light curves aligned at peak (same cadence)
    a2 = axes[2]
    tax = (np.arange(T) - peak_frame) * DT_MS
    a2.plot(tax, cf_sig, color="0.35", lw=1.4,
            label=f"corr-fast coarse (FWHM {cf_fwhm:.1f} ms, SNR {cf_snr:.0f})")
    tax_d = (np.arange(t_det) - det_peak_t) * DT_MS
    a2.plot(tax_d, det_sig, color="crimson", lw=1.4,
            label=f"detector fine DM (FWHM {det_fwhm:.1f} ms, b{det_box})")
    a2.set_xlim(-40, 40)
    a2.axvline(0, color="c", ls=":", lw=1)
    a2.set_xlabel("time from peak (ms)"); a2.set_ylabel("σ")
    a2.set_title("(c) light curve: coarse vs fine dedispersion")
    a2.legend(fontsize=8)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.02)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")

    # ---- printed comparison table ----
    dpix = float(np.hypot(cf_l - det_l, cf_m - det_m))
    print("=" * 64)
    print("FAST-PIPELINE (corr-fast coarse-DM) vs REAL-TIME DETECTOR")
    print("=" * 64)
    print(f"{'quantity':<26}{'corr-fast':>16}{'detector':>16}")
    print("-" * 58)
    print(f"{'peak pixel (l,m)':<26}{f'({cf_l},{cf_m})':>16}{f'({det_l},{det_m})':>16}")
    print(f"{'pixel offset':<26}{'':>16}{f'{dpix:.1f}':>16}")
    print(f"{'DM (pc/cc)':<26}{'576 (coarse)':>16}{f'{det_dm:.0f} (fine)':>16}")
    print(f"{'peak SNR (σ)':<26}{f'{cf_snr:.0f}':>16}{f'{det_snr:.0f}':>16}")
    print(f"{'FWHM (ms)':<26}{f'{cf_fwhm:.1f}':>16}{f'{det_fwhm:.1f}':>16}")
    print("-" * 58)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
