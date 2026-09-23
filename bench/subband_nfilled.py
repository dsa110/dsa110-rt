#!/usr/bin/env python3
"""Per-stream n_filled for sub-band gridding, from the REAL antenna layout.

The search RX ring's slot stride is ``--n-filled`` (one value for every
stream), and its shared-memory size is

    n_corr * n_coarse_dm * t_buf_samples * n_filled * 2 B

so sizing it for n_sub > 1 needs the actual maximum over all
``16 * n_sub`` sub-band patterns -- which depends on the cal blob's
antenna positions and the core mask, not on a model.

Run on any node that has the cal blob::

    python bench/subband_nfilled.py --cal-blob-path .../antennas.out
"""
from __future__ import annotations

import argparse

import numpy as np

from dsart.common.constants import N_CHGROUP
from dsart.grid.sparsity_pattern import (
    IMAGE_PIXEL_ARCSEC_TEE,
    build_pattern,
    cell_lambda_for_pixel_arcsec,
)
from dsart.grid.subband import build_subband_patterns
from dsart.services.corr_fast_integration import load_antpos_from_cal_blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-blob-path", required=True)
    ap.add_argument("--n-sub", type=int, default=4)
    ap.add_argument("--n-grid", type=int, default=256)
    ap.add_argument("--chan-sum-factor", type=int, default=8)
    ap.add_argument("--dec-deg", type=float, default=71.63)
    ap.add_argument("--t-buf-samples", type=int, nargs="*",
                    default=[8192, 4096, 2048])
    a = ap.parse_args()

    e, n, core = load_antpos_from_cal_blob(a.cal_blob_path)
    cell = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, a.n_grid)
    whole, subs = [], []
    for g in range(N_CHGROUP):
        whole.append(int(build_pattern(
            e, n, chgroup=g, dec_deg=a.dec_deg, n_grid=a.n_grid,
            chan_sum_factor=a.chan_sum_factor, cell_lambda=cell,
            is_core_baseline_mask=core).n_filled))
        subs.append([int(p.n_filled) for p in build_subband_patterns(
            e, n, chgroup=g, n_sub=a.n_sub, dec_deg=a.dec_deg,
            n_grid=a.n_grid, kernel_support=1,
            chan_sum_factor=a.chan_sum_factor, cell_lambda=cell,
            is_core_baseline_mask=core)])
    subs = np.asarray(subs)
    print("chgroup  whole  sub-bands (n_sub=%d)   sum   sum/whole" % a.n_sub)
    for g in range(N_CHGROUP):
        print("  %2d    %5d  %s  %5d   %.2f"
              % (g, whole[g], " ".join("%5d" % v for v in subs[g]),
                 subs[g].sum(), subs[g].sum() / whole[g]))
    mx_w, mx_s = max(whole), int(subs.max())
    print()
    print("max whole-chgroup n_filled: %d (ring --n-filled today 5000)" % mx_w)
    print("max sub-band    n_filled: %d" % mx_s)
    print("total cells: whole %d, sub-band %d (x%.2f)"
          % (sum(whole), int(subs.sum()), subs.sum() / sum(whole)))
    print()
    print("RX ring shm, n_coarse_dm=8, bytes/cell=2:")
    for nf, n_corr, lbl in ((5000, 16, "today (16 x 5000)"),
                            (mx_s, 16 * a.n_sub, "sub-band, exact max")):
        for tb in a.t_buf_samples:
            gib = n_corr * 8 * tb * nf * 2 / 2 ** 30
            print("   %-22s t_buf=%-5d  %6.2f GiB" % (lbl, tb, gib))


if __name__ == "__main__":
    main()
