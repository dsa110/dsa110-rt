#!/usr/bin/env python3
"""Search-node GPU imager cost per cube: n_sub=1 dense vs n_sub=4 compact.

The 2026-09-23 live run at n_sub=4 measured the imager at 126 ms/cube
of GPU time (84 ms at n_sub=1) and the search halves fell to ~3.3
cubes/s against the 4.97 cubes/s cadence. This reproduces both feeds at
the production op-point with the REAL patterns (deployed cal blob, Dec
+71.63, tee45 cells, csf 8), so each substage can be attributed:

  feed     n_sub=1: scatter_compact_to_dense (compact wire -> dense int8)
           n_sub=4: transpose_compact_tmajor (compact wire -> time-major)
  combine  fused_dequant_combine_per_fdm_half x n_fdm  |  combine_tiled_v3
  fft      irfft2 over each FFT batch
  mask     edge mask + cast into the output cube

Run on a search node with the fleet stopped (idle GPU)::

    python bench/search_imager_n_sub.py --cal-blob-path .../antennas.out
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dsart.grid.sparsity_pattern import (
    IMAGE_PIXEL_ARCSEC_TEE,
    build_pattern,
    cell_lambda_for_pixel_arcsec,
)
from dsart.image.imager_gpu import build_default_gpu_imager
from dsart.image.tiled_combine_cuda import (
    build_inverse_lut,
    pad_t,
    transpose_compact_tmajor,
)
from dsart.services.corr_fast_integration import load_antpos_from_cal_blob
from dsart.transport.gpu_scatter import scatter_compact_to_dense, zero_dense_rows

N_GRID = 256
T_DET = 256
T_STREAM = 336          # production: t_det + pad 39 + 41
SHIFT_LO, SHIFT_HI = -41, 39
CUBE_MS = 201.326592


def patterns(cal_blob, n_sub, dec):
    e, n, core = load_antpos_from_cal_blob(cal_blob)
    cell = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, N_GRID)
    out = []
    for c in range(16 * n_sub):
        out.append(build_pattern(
            e, n, chgroup=c // n_sub, dec_deg=dec, n_grid=N_GRID,
            kernel_support=1, chan_sum_factor=8, cell_lambda=cell,
            is_core_baseline_mask=core,
            sub_band=None if n_sub == 1 else (c % n_sub, n_sub)))
    return out


def gpu_ms(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def run(pats, n_filled_max, n_fdm, reps, dev):
    n_chg = len(pats)
    nf = np.array([p.n_filled for p in pats], dtype=np.int32)
    lut = np.zeros((n_chg, n_filled_max), dtype=np.int32)
    for g, p in enumerate(pats):
        lut[g, :p.n_filled] = (p.ix_row.astype(np.int64) * N_GRID
                               + p.ix_col.astype(np.int64))
    rng = np.random.default_rng(0)
    cells = np.zeros((n_chg, T_STREAM, n_filled_max * 2), dtype=np.int8)
    for g in range(n_chg):
        cells[g, :, :nf[g] * 2] = rng.integers(
            -60, 61, (T_STREAM, nf[g] * 2), dtype=np.int8)
    shifts = rng.integers(SHIFT_LO, SHIFT_HI + 1, (n_fdm, n_chg)).astype(np.int32)
    t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    cells_g, lut_g, nf_g, sh_g = t(cells), t(lut), t(nf), t(shifts)
    im = build_default_gpu_imager(n_grid=N_GRID, t_det=T_DET, n_fdm=n_fdm,
                                  n_chgroup=n_chg, device=dev)
    res = {"n_chg": n_chg, "cells": int(nf.sum())}
    if n_chg == 16:
        dense = torch.zeros((n_chg, T_STREAM, 2, N_GRID, N_GRID),
                            dtype=torch.int8, device=dev)

        def feed():
            zero_dense_rows(dense=dense, t_det=T_STREAM)
            scatter_compact_to_dense(
                cells_packed=cells_g, lut=lut_g, n_filled_per_corr=nf_g,
                dense=dense, t_det=T_STREAM, n_grid=N_GRID,
                n_filled_max=n_filled_max)

        def image():
            im.process_cube(streams_cint8=dense, time_shifts_gpu=sh_g, t_lo=0)
    else:
        tmaj = torch.zeros((n_chg, n_filled_max, pad_t(T_STREAM), 2),
                           dtype=torch.int8, device=dev)
        inv = build_inverse_lut(lut_g, nf_g, n_grid=N_GRID)

        def feed():
            transpose_compact_tmajor(cells_g, tmaj)

        def image():
            im.process_cube(streams_cint8=tmaj, time_shifts_gpu=sh_g, t_lo=0,
                            compact_inv_lut=inv, compact_t_stream=T_STREAM,
                            compact_shift_bounds=(SHIFT_LO, SHIFT_HI))
    res["feed"] = gpu_ms(feed, reps)
    feed()
    im.pop_substage_timings()
    res["imager"] = gpu_ms(image, reps)
    sub = im.pop_substage_timings()
    res.update(combine=sub["combine_ms"], fft=sub["fft_ms"], mask=sub["mask_ms"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-blob-path", required=True)
    ap.add_argument("--dec-deg", type=float, default=71.63)
    ap.add_argument("--n-fdm", type=int, default=25)
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    print("device %s  t_det %d  t_stream %d  n_fdm %d"
          % (torch.cuda.get_device_name(0), T_DET, T_STREAM, a.n_fdm))
    print("%-22s %6s %7s %7s %8s %7s %7s %8s"
          % ("layout", "n_chg", "cells", "feed", "combine", "fft", "mask",
             "img+feed"))
    for n_sub, nfm in ((1, 5000), (4, 2600)):
        t0 = time.time()
        pats = patterns(a.cal_blob_path, n_sub, a.dec_deg)
        r = run(pats, nfm, a.n_fdm, a.reps, dev)
        print("%-22s %6d %7d %7.1f %8.1f %7.1f %7.1f %8.1f   (setup %.0fs)"
              % ("n_sub=%d %s" % (n_sub, "dense" if n_sub == 1 else "compact"),
                 r["n_chg"], r["cells"], r["feed"], r["combine"], r["fft"],
                 r["mask"], r["feed"] + r["imager"], time.time() - t0))
        torch.cuda.empty_cache()
    print("(ms per cube, GPU time on an idle GPU; cadence %.1f ms)" % CUBE_MS)


if __name__ == "__main__":
    main()
