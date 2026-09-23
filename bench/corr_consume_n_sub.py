#!/usr/bin/env python3
"""Corr-node consume-phase cost vs n_sub, on the REAL code paths.

The corr block budget is 134.218 ms and the slowest nodes already run
~129 ms at n_sub=1, so every n_sub-dependent millisecond matters. The
consume phase is where n_sub enters:

  stage1   Stage1MultiDMCoarseDM.dedisperse_from_vis (fused Triton
           dedisp+grid over the concatenated sub-band cell axis)
  sky      StaticSkyMean.apply per coarse DM (per-cell boxcar)
  tx       AsyncTransportTx.transmit: pinned D2H of the whole cube plus
           the copy of each worker's DM slice into its shm ring, both
           ON the pipeline thread (the encode+sendto runs in workers)

Everything uses production geometry: the node's own cal blob (antpos +
Tee core mask), the deployed DM plan, tee45 cell scale, the 2-block
sliding window (256 fast-vis samples), 8 coarse DMs, 4 TX workers
sending real UDP to a local sink.

Run on a corr node with the fleet stopped::

    python bench/corr_consume_n_sub.py --cal-blob-path .../antennas.out \\
        --dm-plan-path .../dm_plan_N8_dmmin100_dmmax1500_tol1.265_csf8_explicit-v2.npz
"""
from __future__ import annotations

import argparse
import math
import socket
import time

import numpy as np
import torch

from dsart.coarse_dm.dm_plan import DMPlan
from dsart.common.contracts import DmPlan
from dsart.common.constants import NBASE
from dsart.grid.kernel import FastVisGridder
from dsart.grid.sparsity_pattern import (
    IMAGE_PIXEL_ARCSEC_TEE,
    build_pattern,
    cell_lambda_for_pixel_arcsec,
)
from dsart.grid.subband import build_subband_layout, build_subband_patterns
from dsart.services.corr_fast_integration import (
    StaticSkyMean,
    Stage1MultiDMCoarseDM,
    load_antpos_from_cal_blob,
)
from dsart.transport.async_tx import AsyncTransportTx, AsyncTransportTxConfig
from dsart.transport.tx_ring import CubeShmRingDims

CSF = 8
N_FV = 128
BLOCK_MS = 134.217728


def med(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-blob-path", required=True)
    ap.add_argument("--dm-plan-path", required=True)
    ap.add_argument("--chgroup", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-sub", type=int, nargs="*", default=[1, 4])
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.init()

    e, n, core = load_antpos_from_cal_blob(a.cal_blob_path)
    cell = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, 256)
    plan = DMPlan.from_summed_canonical(
        DmPlan.from_npz(a.dm_plan_path), chan_sum_factor=CSF)
    whole = build_pattern(e, n, chgroup=a.chgroup, dec_deg=71.63, n_grid=256,
                          chan_sum_factor=CSF, cell_lambda=cell,
                          is_core_baseline_mask=core)
    gridder = FastVisGridder.from_pattern(whole, e, n,
                                          is_core_baseline_mask=core,
                                          device=dev)
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", 0))
    port = sink.getsockname()[1]

    g = torch.Generator(device=dev).manual_seed(1)
    vis = (torch.randn(N_FV, NBASE, 384 // CSF, generator=g, device=dev)
           + 1j * torch.randn(N_FV, NBASE, 384 // CSF, generator=g, device=dev)
           ).to(torch.complex64)

    print("device %s  chgroup %d  whole n_filled %d"
          % (torch.cuda.get_device_name(0), a.chgroup, whole.n_filled))
    print("%-6s %8s %9s %9s %9s %9s %9s"
          % ("n_sub", "cells", "stage1", "sky", "tx(main)", "total", "vs n_sub=1"))
    base = None
    for n_sub in a.n_sub:
        layout = None
        streams = None
        n_cells = int(whole.n_filled)
        if n_sub > 1:
            pats = build_subband_patterns(
                e, n, chgroup=a.chgroup, n_sub=n_sub, dec_deg=71.63,
                n_grid=256, kernel_support=1, chan_sum_factor=CSF,
                cell_lambda=cell, is_core_baseline_mask=core)
            layout = build_subband_layout(pats, gridder, device=dev)
            n_cells = layout.n_filled_total
            streams = [(sid, pid, int(layout.offsets[s]), int(layout.offsets[s + 1]))
                       for s, (sid, pid) in enumerate(zip(layout.stream_ids,
                                                          layout.pattern_ids))]
        st1 = Stage1MultiDMCoarseDM(plan=plan, gridder=gridder,
                                    chgroup=a.chgroup, sliding_window=True,
                                    subband_layout=layout)
        st1.dedisperse_from_vis(vis, block_n=0)          # prime the window
        cube = st1.dedisperse_from_vis(vis, block_n=1)   # (8, 128, cells)
        t_s1, _ = med(lambda: st1.dedisperse_from_vis(vis, block_n=2), a.reps)

        sky = StaticSkyMean(window_blocks=8, warmup_cubes=0, n_dm=8)

        def do_sky():
            for c in range(cube.shape[0]):
                cube[c] = sky.apply(cube[c], dm_slot=c)
        t_sky, _ = med(do_sky, a.reps)

        cfg = AsyncTransportTxConfig(
            host="127.0.0.1", port=port, chgroup=a.chgroup, n_workers=4,
            n_dm_total=8,
            ring_dims=CubeShmRingDims(n_slots=4, shape=(2, N_FV, n_cells),
                                      dtype=np.dtype("complex64")),
            pattern_id=int(whole.pattern_id), n_grid=256,
            target_gbps_per_flow=1.0, corr_idx=a.chgroup, log_level="WARNING",
            shm_name_prefix=f"dsart-bench-consume-{n_sub}", streams=streams,
            reserve_timeout_s=10.0,
        )
        atx = AsyncTransportTx.spawn(cfg)
        # spawned workers import torch + open sockets before they drain;
        # production absorbs this in its Triton warm-up
        time.sleep(20.0)
        try:
            bn = [10]

            def do_tx():
                bn[0] += 1
                atx.transmit([cube], block_n=bn[0], rfi_warming_up=False,
                             specnum=bn[0])
                time.sleep(0.14)        # let workers drain like real blocks

            # time only the main-thread part: transmit() itself
            do_tx()
            ts = []
            for _ in range(a.reps):
                bn[0] += 1
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                atx.transmit([cube], block_n=bn[0], rfi_warming_up=False,
                             specnum=bn[0])
                ts.append((time.perf_counter() - t0) * 1e3)
                time.sleep(0.14)
            t_tx = float(np.median(ts))
        finally:
            atx.close()
        total = t_s1 + t_sky + t_tx
        if base is None:
            base = total
        print("%-6d %8d %9.2f %9.2f %9.2f %9.2f %9s"
              % (n_sub, n_cells, t_s1, t_sky, t_tx, total,
                 "%+.2f ms" % (total - base)))
    print("(block budget %.1f ms; the n_sub=1 row is today's consume cost"
          " for these three stages)" % BLOCK_MS)


if __name__ == "__main__":
    main()
