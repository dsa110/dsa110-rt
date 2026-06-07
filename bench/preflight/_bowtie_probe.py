"""Diagnostic: measure burst energy at a fixed sky pixel across ALL coarse-DM
owners, to verify the dispersion-bowtie tails leak into non-owning GPUs.

Usage: python _bowtie_probe.py <injection_dir> <l_pix> <m_pix>
"""
from __future__ import annotations

import glob
import sys

import numpy as np

from bench.preflight._inject_search_driver import (
    DEFAULT_DM_PLAN_PATH,
    owner_dm_grids,
)


def robust_sigma(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(np.float64)
    med = np.median(ts)
    mad = np.median(np.abs(ts - med)) * 1.4826
    if mad <= 0:
        mad = ts.std() or 1.0
    return (ts - med) / mad


def main() -> None:
    base = sys.argv[1]
    L = int(sys.argv[2])
    M = int(sys.argv[3])
    hdr = ("own", "coarseDM", "DMlo", "DMhi", "peakSig@pix", "@DM", "@t", "specStart")
    print("{:>3} {:>8} {:>6} {:>6} {:>11} {:>6} {:>4} {:>9}".format(*hdr))
    for o in range(8):
        fs = glob.glob("{}/owner{}/cube_*.npz".format(base, o))
        if not fs:
            continue
        d = np.load(fs[0])
        cube = d["cube"]
        spec = int(d["event_specnum_start"])
        g = owner_dm_grids(DEFAULT_DM_PLAN_PATH, o)
        fdm = g.fine_dm_local
        cdm = float(g.coarse_dm_local[0])
        plane = np.asarray(cube[:, :, L, M], np.float32).T  # (nfdm, t)
        S = np.stack([robust_sigma(plane[i]) for i in range(plane.shape[0])], 0)
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        print("{:>3} {:>8.0f} {:>6.0f} {:>6.0f} {:>11.1f} {:>6.0f} {:>4d} {:>9d}".format(
            o, cdm, fdm.min(), fdm.max(), float(S[i, j]), float(fdm[i]), int(j), spec))


if __name__ == "__main__":
    main()
