#!/usr/bin/env python3
"""v2 combine: one block per uv cell. Bit-exact? Fast enough for n_sub=4?

Compares, at n_sb = 16 / 32 / 64 with n_fdm held at the production 34:

    dense      scatter_compact_to_dense + fused_dequant_..._half x n_fdm
    v1         time-minor tiled, block = (cell, small time tile)
    v2         re/im interleaved, block = (cell, tile sized to shared mem)

Run on a search node with the fleet stopped::

    conda activate dsa110-rt && python bench/combine_v2.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dsart.image.tiled_combine_cuda import (
    combine_tiled,
    combine_tiled_v2,
    max_tile_for,
    pad_t,
    scatter_compact_to_dense_tmin,
    scatter_compact_to_dense_v2,
)

from bench.coo_combine_equiv import (  # noqa: E402
    N_GRID, T_DET, T_LO, T_STREAM, build_inputs, run_dense,
)


def med(fn, dev, reps=5):
    fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-fdm", type=int, default=34)
    ap.add_argument("--threads", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dev = a.device
    n_fdm = a.n_fdm
    t_out_len = T_DET - T_LO
    t_pad = pad_t(T_STREAM)

    print("device %s  %s" % (dev, torch.cuda.get_device_name(
        int(dev.split(":")[1]))))
    print("n_fdm=%d (production), N=%d, t_stream=%d, t_pad=%d, t_lo=%d"
          % (n_fdm, N_GRID, T_STREAM, t_pad, T_LO))
    print()
    print("%-5s %-6s %10s %10s %10s %10s  %s"
          % ("n_sub", "n_sb", "dense ms", "v1 ms", "v2 ms", "v2 GiB",
             "v2 bit-exact"))

    for n_sub in (1, 2, 4):
        n_sb = 16 * n_sub
        torch.cuda.empty_cache()
        cells, lut, nf, shifts = build_inputs(n_sb, n_fdm, dev)
        spread = int(shifts.max().item()) - int(shifts.min().item())
        tile = max_tile_for(n_sb, spread, t_out_len)

        # v2
        torch.cuda.reset_peak_memory_stats(dev)
        d2 = torch.zeros((n_sb, N_GRID * N_GRID, t_pad, 2),
                         dtype=torch.int8, device=dev)

        def run_v2():
            scatter_compact_to_dense_v2(cells, lut, nf, d2, n_grid=N_GRID)
            return combine_tiled_v2(
                d2, shifts, n_grid=N_GRID, t_rows=T_STREAM,
                t_out_len=t_out_len, t_lo=T_LO, fftshift=True,
                threads=a.threads, tile=tile)

        out2 = run_v2()
        v2_ms = med(run_v2, dev, a.reps)
        v2_gib = torch.cuda.max_memory_allocated(dev) / 2**30

        # v1
        torch.cuda.empty_cache()
        d1 = torch.zeros((n_sb, 2, N_GRID * N_GRID, T_STREAM),
                         dtype=torch.int8, device=dev)
        t1 = 30 if n_fdm * 30 <= 1024 else 1024 // n_fdm

        def run_v1():
            scatter_compact_to_dense_tmin(cells, lut, nf, d1, n_grid=N_GRID)
            return combine_tiled(
                d1, shifts, n_grid=N_GRID, t_out_len=t_out_len,
                t_lo=T_LO, fftshift=True, tile=t1)

        run_v1()
        v1_ms = med(run_v1, dev, a.reps)
        del d1
        torch.cuda.empty_cache()

        # dense reference
        try:
            ref, _dn = run_dense(cells, lut, nf, shifts, n_sb, n_fdm, dev)
            dn_ms = med(
                lambda: run_dense(cells, lut, nf, shifts, n_sb, n_fdm, dev)[0],
                dev, a.reps)
            exact = bool(torch.equal(out2, ref))
            dn = "%10.2f" % dn_ms
            del ref, _dn
        except torch.cuda.OutOfMemoryError:
            dn, exact = "       OOM", None
            torch.cuda.empty_cache()

        print("%-5d %-6d %10s %10.2f %10.2f %10.2f  %s  (tile=%d)"
              % (n_sub, n_sb, dn, v1_ms, v2_ms, v2_gib,
                 "n/a" if exact is None else str(exact), tile))
        del cells, lut, nf, shifts, d2, out2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
