#!/usr/bin/env python3
"""Search-node CPU compact assembly per cube: 16 vs 64 streams.

``RxRing.assemble_compact_block`` (C) walks every stream's window of
``n_rows`` (= t_stream, ~340 at the production op-point) slots and packs
them into the compact buffer the GPU consumes. It is the "scatter=...us"
term in the search_compute log (~24 ms/cube today) and scales with the
bytes it moves: n_corr * n_rows * n_filled_max * 2.

Uses the real C ring + assembler with every slot populated.
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from dsart.transport.recv_ring import VF_DATA_PRESENT, RxRing, RxRingDims

N_ROWS = 340
OWNED_DM = 2


def run(n_corr, n_filled_max, n_filled_real, t_buf, reps=12):
    name = f"/dsart-bench-asm-{uuid.uuid4().hex[:8]}"
    dims = RxRingDims(n_corr=n_corr, n_coarse_dm=8, t_buf_samples=t_buf,
                      n_filled_per_corr=n_filled_max, bytes_per_cell=2)
    ring = RxRing.open_or_create(name, dims)
    try:
        rng = np.random.default_rng(0)
        payload = rng.integers(-100, 100, n_filled_max * 2,
                               dtype=np.int8).tobytes()
        for c in range(n_corr):
            for t in range(N_ROWS + 8):
                ring.write_slot(corr=c, dm=OWNED_DM, t_seq=t, payload=payload,
                                validity_flags=VF_DATA_PRESENT,
                                scale=0.05, offset=0.0)
        nfp = np.full(n_corr, n_filled_real, dtype=np.int32)
        out = np.zeros((n_corr, N_ROWS, n_filled_max * 2), dtype=np.int8)
        kw = dict(specnum_start=0, t_det=N_ROWS, owned_dm=OWNED_DM,
                  n_filled_per_corr=nfp, n_filled_max=n_filled_max,
                  sidecar_t_stride=N_ROWS, compute_half=0,
                  out_cells_packed=out)
        ring.assemble_compact_block(**kw)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ring.assemble_compact_block(**kw)
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts)), out.nbytes / 1e6
    finally:
        ring.close()
        RxRing.unlink_name(name)


def main():
    print("%-30s %9s %9s %9s" % ("layout", "MB/cube", "ms/cube", "GB/s"))
    for lbl, n_corr, nfm, nfr, tb in (
        ("today: 16 x 5000 (3300 real)", 16, 5000, 3300, 8192),
        ("n_sub=4: 64 x 2600 (2150 real)", 64, 2600, 2150, 6144),
    ):
        ms, mb = run(n_corr, nfm, nfr, tb)
        print("%-30s %9.1f %9.2f %9.2f" % (lbl, mb, ms, mb / ms))
    print("(cube cadence 201.3 ms)")


if __name__ == "__main__":
    main()
