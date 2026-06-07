"""Matched-boxcar probe of a single owner cube: find the peak detection
sigma over (fine-DM, boxcar-width, time) and report where it is, plus the
brightest pixel near a reference time. Diagnoses wide-burst detection.

Usage: python _wide_probe.py <cube.npz> [Lref] [Mref]
"""
from __future__ import annotations

import glob
import sys

import numpy as np


def boxcar_sigma(ts: np.ndarray, w: int) -> np.ndarray:
    ts = ts.astype(np.float64)
    if w > 1:
        k = np.ones(w)
        c = np.convolve(ts, k, mode="same") / np.sqrt(w)
    else:
        c = ts.copy()
    med = np.median(c)
    mad = np.median(np.abs(c - med)) * 1.4826
    if mad <= 0:
        mad = c.std() or 1.0
    return (c - med) / mad


def main() -> None:
    path = sys.argv[1]
    if "*" in path:
        path = glob.glob(path)[0]
    d = np.load(path)
    cube = np.asarray(d["cube"])  # (t, nfdm, N, N)
    t_det, n_fdm, N, _ = cube.shape
    print("cube", cube.shape, "specStart", int(d["event_specnum_start"]))

    boxes = [1, 2, 4, 8, 16, 24, 32]
    if len(sys.argv) >= 4:
        L, M = int(sys.argv[2]), int(sys.argv[3])
    else:
        # brightest pixel via DM-collapsed, time-collapsed power
        pwr = np.asarray(cube, np.float32).astype(np.float64)
        flat = (pwr ** 2).sum(axis=(0, 1))
        L, M = np.unravel_index(int(flat.argmax()), flat.shape)
        print("auto brightest pixel:", (L, M))

    best = (-1.0, None, None, None)
    for fi in range(n_fdm):
        ts = np.asarray(cube[:, fi, L, M], np.float32)
        for w in boxes:
            s = boxcar_sigma(ts, w)
            j = int(np.argmax(s))
            if s[j] > best[0]:
                best = (float(s[j]), fi, w, j)
    print("pixel ({},{}) peak matched sigma = {:.1f} at fdm={} box={} t={}".format(
        L, M, best[0], best[1], best[2], best[3]))
    # per-box best across all DM
    for w in boxes:
        bs = -1.0
        for fi in range(n_fdm):
            s = boxcar_sigma(np.asarray(cube[:, fi, L, M], np.float32), w)
            bs = max(bs, float(s.max()))
        print("  box={:>2}  best sigma over DM = {:.1f}".format(w, bs))


if __name__ == "__main__":
    main()
