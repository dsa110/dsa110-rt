#!/usr/bin/env python3
"""What does ONE more coarse DM actually cost the corr node?

configs/dsart_pipeline_rt.yaml pins n_coarse at 8 citing ~3.1 ms per
coarse DM against a ~4 ms margin. That figure is the whole consume
phase divided by 8. But the 2026-09-22 gridder study found that 8.85 of
the 11.5 ms of ``_dedisperse_one_window`` is the (T,B,C)->(B,C,T)
permute, which is n_coarse-INDEPENDENT -- so the true marginal cost may
be far smaller, and the DM ceiling far cheaper to raise.

This sweeps the production fused Triton dedisp+grid kernel over n_dm at
the exact production geometry and separates the fixed permute from the
per-coarse-DM slope.

Run on a corr node with the fleet stopped::

    conda activate dsa110-rt && python bench/dedisp_ncoarse_scan.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import NBASE, NCHAN_PER_CHGROUP  # noqa: E402
from dsart.services.triton_dedisp import (  # noqa: E402
    build_cell_csr,
    fused_dedisp_triton,
)

CSF = 8
NCH = NCHAN_PER_CHGROUP // CSF          # 48
N_FV = 256                              # joined sliding window
MAX_SHIFT = 33                          # production max_bin_shift
N_FILLED = 3247                         # n04 live value, corr 0


def med(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--n-filled", type=int, default=N_FILLED)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.init()
    n_filled = a.n_filled
    t_dedisp = N_FV - MAX_SHIFT

    print("device: %s" % torch.cuda.get_device_name(0))
    print("geometry: n_fv=%d NBASE=%d nch=%d n_filled=%d t_dedisp=%d"
          % (N_FV, NBASE, NCH, n_filled, t_dedisp))
    print()

    rng = np.random.default_rng(0xBADD15)
    cim_bc_np = rng.integers(0, n_filled + 1, size=(NBASE * NCH,),
                             dtype=np.int64)
    cim_bc = torch.from_numpy(cim_bc_np).to(dev)
    csr_offs, csr_b, csr_c = build_cell_csr(
        cim_bc, n_filled=n_filled, nchan_eff=NCH, nbase=NBASE)

    g = torch.Generator(device=dev).manual_seed(0xC0DE)
    vis_TBC = (
        torch.rand(N_FV, NBASE, NCH, dtype=torch.float32, generator=g,
                   device=dev)
        + 1j * torch.rand(N_FV, NBASE, NCH, dtype=torch.float32,
                          generator=g, device=dev)
    ).to(torch.complex64)

    # the fixed, n_coarse-INDEPENDENT part of _dedisperse_one_window
    def permute():
        v = vis_TBC.permute(1, 2, 0).contiguous()
        return v.real.contiguous(), v.imag.contiguous()

    perm_ms = med(permute, a.reps)
    vis_re, vis_im = permute()
    print("permute + re/im split (fixed, n_coarse-independent): %.2f ms"
          % perm_ms)
    print()
    print("%-8s %10s %10s %12s" % ("n_dm", "kernel ms", "total ms",
                                   "marginal ms"))
    prev = None
    rows = []
    for n_dm in (1, 2, 4, 8, 10, 12, 16, 20, 24):
        bs = rng.integers(0, MAX_SHIFT + 1, size=(NCH, n_dm), dtype=np.int64)
        bin_shifts = torch.from_numpy(bs).to(dev).to(torch.int32).contiguous()
        try:
            ms = med(lambda: fused_dedisp_triton(
                vis_re, vis_im, bin_shifts=bin_shifts,
                csr_offs=csr_offs, csr_b=csr_b, csr_c=csr_c,
                n_filled=n_filled, t_dedisp=t_dedisp), a.reps)
        except torch.cuda.OutOfMemoryError:
            print("%-8d %10s" % (n_dm, "OOM"))
            torch.cuda.empty_cache()
            continue
        marg = "-" if prev is None else "%.3f" % ((ms - prev[1]) /
                                                  (n_dm - prev[0]))
        print("%-8d %10.2f %10.2f %12s" % (n_dm, ms, ms + perm_ms, marg))
        rows.append((n_dm, ms))
        prev = (n_dm, ms)
        torch.cuda.empty_cache()

    n = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows])
    slope, icept = np.polyfit(n, y, 1)
    print()
    print("fit over n_dm=1..%d: kernel = %.3f + %.3f * n_dm  ms"
          % (int(n[-1]), icept, slope))
    print("so one extra coarse DM costs %.3f ms of dedisp+grid" % slope)
    print("(the config's ~3.1 ms/coarse is the WHOLE consume phase / 8;")
    print(" cube D2H and TX encode also scale with n_dm -- see")
    print(" AsyncTransportTx and TransportTx._transmit_one_cube_prod.)")


if __name__ == "__main__":
    main()
