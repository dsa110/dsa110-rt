#!/usr/bin/env python3
"""Is the COO-direct combine bit-identical to the dense one, and faster?

The dense production path is

    zero_dense_rows + scatter_compact_to_dense   (expand COO -> dense)
    fused_dequant_combine_per_fdm_half           (once PER fine-DM)

and the COO path replaces all of it with a single input-stationary
kernel that never materialises the dense plane.

This script builds one random-but-realistic compact block, runs both,
and reports (a) exact equality of the resulting half2 planes and
(b) per-cube timing and peak memory for each.

Run on a search node with the fleet stopped::

    conda activate dsa110-rt && python bench/coo_combine_equiv.py
    python bench/coo_combine_equiv.py --n-sub 4     # 64 sub-bands
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dsart.image.coo_combine_cuda import accumulator_dtype_for, combine_coo
from dsart.image.fused_combine_cuda import (
    fused_dequant_combine_per_fdm_half,
)
from dsart.transport.gpu_scatter import (
    scatter_compact_to_dense,
    zero_dense_rows,
)

N_GRID = 256
T_STREAM = 322          # symmetric-padding production value
T_DET = 256
T_LO = 64               # M7.7.2 carry-over: 192 rows re-imaged
N_FILLED = 3247         # n02 live log, corr 0
N_FILLED_MAX = 5000     # provisioned wire-side dimension


def build_inputs(n_sb, n_fdm, device, seed=0):
    rng = np.random.default_rng(seed)
    # Static sparsity LUT: distinct filled cells per sub-band.
    lut = np.zeros((n_sb, N_FILLED_MAX), dtype=np.int32)
    for g in range(n_sb):
        lut[g, :N_FILLED] = rng.choice(
            N_GRID * N_GRID, size=N_FILLED, replace=False,
        ).astype(np.int32)
    n_filled = np.full((n_sb,), N_FILLED, dtype=np.int32)

    cells = np.zeros((n_sb, T_STREAM, N_FILLED_MAX * 2), dtype=np.int8)
    cells[:, :, : N_FILLED * 2] = rng.integers(
        -127, 128, size=(n_sb, T_STREAM, N_FILLED * 2), dtype=np.int8,
    )
    # Shifts in the measured production range (shifts.min=-34, max=32).
    shifts = rng.integers(-34, 33, size=(n_fdm, n_sb)).astype(np.int32)

    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return to(cells), to(lut), to(n_filled), to(shifts)


def run_dense(cells, lut, n_filled, shifts, n_sb, n_fdm, device):
    t_out_len = T_DET - T_LO
    dense = torch.zeros(
        (n_sb, T_STREAM, 2, N_GRID, N_GRID), dtype=torch.int8, device=device,
    )
    zero_dense_rows(dense=dense, t_det=T_STREAM)
    scatter_compact_to_dense(
        cells_packed=cells, lut=lut, n_filled_per_corr=n_filled,
        dense=dense, t_det=T_STREAM, n_grid=N_GRID,
        n_filled_max=N_FILLED_MAX,
    )
    n_half = N_GRID // 2 + 1
    plane = torch.empty(
        (t_out_len, N_GRID, n_half), dtype=torch.complex32, device=device,
    )
    out = torch.empty(
        (n_fdm, t_out_len, N_GRID, n_half, 2),
        dtype=torch.float16, device=device,
    )
    for f in range(n_fdm):
        fused_dequant_combine_per_fdm_half(
            dense, shifts[f].contiguous().int(), plane,
            t_lo=T_LO, fftshift=True,
        )
        out[f] = torch.view_as_real(plane)
    return out, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-sub", type=int, default=1,
                    help="sub-bands per chgroup (n_sb = 16 * n_sub)")
    ap.add_argument("--n-fdm", type=int, default=34)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tile", type=int, default=24)
    ap.add_argument("--skip-dense", action="store_true",
                    help="COO only (dense OOMs at large n_sb)")
    a = ap.parse_args()

    dev = a.device
    n_sb = 16 * a.n_sub
    n_fdm = a.n_fdm
    t_out_len = T_DET - T_LO
    print("device %s  %s" % (dev, torch.cuda.get_device_name(
        int(dev.split(":")[1]))))
    print("n_sub=%d -> n_sb=%d   n_fdm=%d   N=%d  t_stream=%d  t_lo=%d"
          % (a.n_sub, n_sb, n_fdm, N_GRID, T_STREAM, T_LO))
    print("accumulator dtype: %s  (fp16 exact while n_sb <= 16)"
          % accumulator_dtype_for(n_sb))
    print()

    cells, lut, n_filled, shifts = build_inputs(n_sb, n_fdm, dev)

    # ---------------- COO ----------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    out_coo = combine_coo(
        cells, lut, n_filled, shifts,
        n_grid=N_GRID, t_out_len=t_out_len, t_lo=T_LO, fftshift=True,
    )
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        combine_coo(cells, lut, n_filled, shifts, n_grid=N_GRID,
                    t_out_len=t_out_len, t_lo=T_LO, fftshift=True)
        torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    coo_ms = float(np.median(ts))
    coo_gib = torch.cuda.max_memory_allocated(dev) / 2**30
    print("COO-direct   %8.2f ms/cube   peak %.2f GiB" % (coo_ms, coo_gib))

    # ---------------- tiled (time-minor dense) ----------------
    from dsart.image.tiled_combine_cuda import (
        combine_tiled,
        scatter_compact_to_dense_tmin,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    tile = a.tile
    while n_fdm * tile > 1024:
        tile -= 1
    dense_t = torch.zeros(
        (n_sb, 2, N_GRID * N_GRID, T_STREAM), dtype=torch.int8, device=dev,
    )

    def tiled_once():
        scatter_compact_to_dense_tmin(
            cells, lut, n_filled, dense_t, n_grid=N_GRID)
        return combine_tiled(
            dense_t, shifts, n_grid=N_GRID, t_out_len=t_out_len,
            t_lo=T_LO, fftshift=True, tile=tile)

    out_tiled = tiled_once()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        tiled_once()
        torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    tiled_ms = float(np.median(ts))
    tiled_gib = torch.cuda.max_memory_allocated(dev) / 2**30
    print("tiled(B=%-2d)  %8.2f ms/cube   peak %.2f GiB"
          % (tile, tiled_ms, tiled_gib))

    if a.skip_dense:
        return

    # ---------------- dense ----------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        out_dense, dense = run_dense(
            cells, lut, n_filled, shifts, n_sb, n_fdm, dev)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            run_dense(cells, lut, n_filled, shifts, n_sb, n_fdm, dev)
            torch.cuda.synchronize(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        dense_ms = float(np.median(ts))
        dense_gib = torch.cuda.max_memory_allocated(dev) / 2**30
    except torch.cuda.OutOfMemoryError:
        print("dense        OOM (this is the n_sb>=48 wall)")
        return

    print("dense+scatter%8.2f ms/cube   peak %.2f GiB" % (dense_ms, dense_gib))
    print()
    print("vs dense:  COO   %.2fx speed, %.2fx memory"
          % (dense_ms / coo_ms, dense_gib / max(coo_gib, 1e-9)))
    print("           tiled %.2fx speed, %.2fx memory"
          % (dense_ms / tiled_ms, dense_gib / max(tiled_gib, 1e-9)))
    print()

    for name, got in (("COO", out_coo), ("tiled", out_tiled)):
        same = torch.equal(got, out_dense)
        print("%-6s BIT-IDENTICAL: %s" % (name, same))
        if not same:
            d = (got.float() - out_dense.float()).abs()
            nz = int((d > 0).sum().item())
            print("   differing %d / %d (%.3g%%)  max abs diff %.6g  "
                  "max |dense| %.6g"
                  % (nz, d.numel(), 100.0 * nz / d.numel(), float(d.max()),
                     float(out_dense.float().abs().max())))


if __name__ == "__main__":
    main()
