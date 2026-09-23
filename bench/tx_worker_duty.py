#!/usr/bin/env python3
"""TX worker duty cycle per 134.218 ms block, whole-chgroup vs sub-band.

Each async-TX worker owns ``n_dm_per_worker`` coarse DMs (2 in
production: 8 DMs over 4 workers) and must encode + sendto every
``(stream, dm, t)`` tile of one block before the next block arrives.
At n_sub=4 the tile count quadruples, which the 2026-09-22 study
measured at 116.9 ms/worker (87% duty) with the per-tile encode.

This times the REAL ``TransportTx._transmit_one_cube_prod`` (with real
``sendto`` into a local UDP sink that is never read) for both layouts.
Run it against both the old and new source trees to compare::

    PYTHONPATH=<tree>/src python bench/tx_worker_duty.py
"""
from __future__ import annotations

import argparse
import socket
import time

import numpy as np
import torch

from dsart.transport.prod_frame import BITS_CINT8_COMPLEX
from dsart.transport.tx import TransportTx, TransportTxProdConfig

BLOCK_MS = 134.217728


def make_tx(port, chgroup):
    cfg = TransportTxProdConfig(
        target_gbps_per_flow=1.0, pacer_headroom=1.05,
        bits_per_cell=BITS_CINT8_COMPLEX, t_int_factor=1, corr_idx=0,
        bucket_fifo_depth=4, dm_idx_offset=0, bypass_pacer=True,
        sndbuf_mib=64,
    )
    tx = TransportTx(host="127.0.0.1", port=port, chgroup=chgroup,
                     use_prod_frame=True, prod_config=cfg)
    tx.prepare_prod(pattern_id_by_chgroup={chgroup: 0x1234}, n_grid=256)
    return tx


def time_block(txs_and_cubes, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for tx, cube in txs_and_cubes:
            tx.transmit([cube], block_n=0, rfi_warming_up=False, specnum=0)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n-dm-per-worker", type=int, default=2)
    ap.add_argument("--n-fv", type=int, default=128)
    a = ap.parse_args()

    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", 0))
    port = sink.getsockname()[1]
    rng = np.random.default_rng(0)

    def cube(n_filled):
        x = (rng.standard_normal((a.n_dm_per_worker, a.n_fv, n_filled))
             + 1j * rng.standard_normal((a.n_dm_per_worker, a.n_fv, n_filled)))
        return torch.from_numpy(x.astype(np.complex64))

    print("tiles/block = n_stream * n_dm(%d) * n_fv(%d); budget %.1f ms"
          % (a.n_dm_per_worker, a.n_fv, BLOCK_MS))
    print("%-28s %8s %10s %10s %8s"
          % ("layout", "tiles", "p50 ms", "p90 ms", "duty"))
    for label, cells in (
        ("whole chgroup (1 x 3300)", [3300]),
        ("n_sub=4 (4 x ~2100)", [2090, 2120, 2150, 2200]),
    ):
        pairs = [(make_tx(port, 10 + i), cube(n)) for i, n in enumerate(cells)]
        time_block(pairs, 2)                       # warm-up
        p50, p90 = time_block(pairs, a.reps)
        tiles = len(cells) * a.n_dm_per_worker * a.n_fv
        print("%-28s %8d %10.2f %10.2f %7.0f%%"
              % (label, tiles, p50, p90, 100 * p50 / BLOCK_MS))
        for tx, _ in pairs:
            tx.close()


if __name__ == "__main__":
    main()
