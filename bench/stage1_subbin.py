#!/usr/bin/env python3
"""Sub-bin stage 1: does it buy what the model says, and does it fit?

``apply_stage1_shifts`` rounds each channel's coarse-DM delay to a whole
``t_int_fast`` sample (+-524 us at the production op-point).
``apply_stage1_shifts_subbin`` rounds to ``t_int_fast / n_sub`` instead
and sums ``n_sub`` sub-bins per output sample. The offline model says
that is worth x1.092 combined with the merged per-chgroup rounding.

Two questions this answers on real hardware:

  ACCURACY     with a dispersed delta injected at a DM offset from the
               coarse trial, how much of the peak does each path
               recover? Channels are summed, so whole-sample rounding
               smears the peak across samples and sub-binning
               concentrates it.

  REAL TIME    ms per block for the gather itself, and the size of the
               finer visibility tensor the caller has to produce. The
               binding cost is NOT this function -- it is that the
               fast-corr GEMM must emit n_sub x more time samples.

``n_sub`` must divide ``ppfv = t_int_fast_native // NTIMES_PER_PACKET``
(= 16 at the production op-point), so the usable set is
{1, 2, 4, 8, 16}; a packet holds 2 native samples, so 16 (65.536 us) is
the floor, not 32.

Run on a corr node with the fleet stopped:
    conda activate dsa110-rt && python bench/stage1_subbin.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from dsart.coarse_dm.dm_plan import (
    DMPlan,
    build_summed_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
)
from dsart.coarse_dm.stage1 import (
    apply_stage1_shifts,
    apply_stage1_shifts_subbin,
)
from dsart.common.constants import NATIVE_SAMPLE_US, NBASE

CSF = 8
T_INT_FAST_NATIVE = 32
T_US = T_INT_FAST_NATIVE * NATIVE_SAMPLE_US        # 1048.576
NCHAN_EFF = 384 // CSF                             # 48
N_FV_PER_BLOCK = 128                               # 2048 packets / ppfv 16
N_SUBS = (1, 2, 4, 8, 16)


def make_plan(coarse_dm):
    freqs = build_summed_chgroup_freq_table_GHz(CSF)
    dm = np.asarray(coarse_dm, dtype=np.float64)
    return DMPlan(
        dm_pc_cc=dm,
        n_fine_per_coarse=34,
        t_int_fast_us=T_US,
        chgroup_freqs_GHz=freqs,
        _delay_native_samples_table=compute_delay_native_samples_table(
            dm, freqs),
        chan_sum_factor=CSF,
    )


# ---------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------

def accuracy(chgroup, coarse_dm, true_dm, n_base=4, device="cuda:0"):
    """Recovered peak of a dispersed delta, whole-bin vs each n_sub.

    The burst is laid down at its TRUE per-channel arrival time on a
    fine (native-resolution) time axis; each path then samples that
    axis at its own resolution and dedisperses at ``coarse_dm``.
    """
    plan = make_plan([coarse_dm])
    freqs = plan.chgroup_freqs_GHz[chgroup]            # (48,) GHz
    from dsart.common.constants import K_DM_MS_GHZ2_PC as K
    # arrival of the TRUE dm, relative to the chgroup top, in native samples
    tau_us = 1e3 * K * true_dm * (1.0 / freqs**2 - 1.0 / freqs[0] ** 2)
    out = {}
    for n_sub in N_SUBS:
        sub_us = T_US / n_sub
        n_t = N_FV_PER_BLOCK * n_sub
        # place the delta in the sub-bin it truly falls in
        vis = torch.zeros((n_t, NBASE, NCHAN_EFF), dtype=torch.complex64,
                          device=device)
        t0 = n_t // 4
        for c in range(NCHAN_EFF):
            idx = t0 + int(np.floor(tau_us[c] / sub_us))
            if 0 <= idx < n_t:
                vis[idx, :n_base, c] = 1.0
        t_ded = 48
        if n_sub == 1:
            got = apply_stage1_shifts(
                vis, plan, chgroup=chgroup, dm_idx=0, t_dedisp=t_ded)
        else:
            got = apply_stage1_shifts_subbin(
                vis, plan, chgroup=chgroup, dm_idx=0, n_sub=n_sub,
                t_dedisp=t_ded)
        prof = got[:, 0, :].real.sum(dim=1)            # sum over channels
        out[n_sub] = float(prof.max().item()) / NCHAN_EFF
        del vis, got
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------

def timing(chgroup, coarse_dm, device="cuda:0", reps=12):
    plan = make_plan([coarse_dm])
    res = {}
    for n_sub in N_SUBS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        n_t = N_FV_PER_BLOCK * n_sub
        try:
            vis = torch.zeros((n_t, NBASE, NCHAN_EFF),
                              dtype=torch.complex64, device=device)
        except torch.cuda.OutOfMemoryError:
            res[n_sub] = (None, None, "OOM allocating vis")
            torch.cuda.empty_cache()
            continue
        shifts = plan.delay_subbins_per_chgroup(chgroup, n_sub)[:, 0]
        t_ded = (n_t - int(shifts.max())) // n_sub
        t_ded = min(t_ded, 85)
        if t_ded <= 0:
            res[n_sub] = (None, None, "no usable range")
            del vis
            torch.cuda.empty_cache()
            continue

        def call():
            if n_sub == 1:
                return apply_stage1_shifts(
                    vis, plan, chgroup=chgroup, dm_idx=0, t_dedisp=t_ded)
            return apply_stage1_shifts_subbin(
                vis, plan, chgroup=chgroup, dm_idx=0, n_sub=n_sub,
                t_dedisp=t_ded)

        try:
            for _ in range(3):
                call()
            torch.cuda.synchronize(device)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                call()
                torch.cuda.synchronize(device)
                ts.append((time.perf_counter() - t0) * 1e3)
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            res[n_sub] = (float(np.median(ts)),
                          vis.numel() * 8 / 1e9, "t_ded=%d peak=%.2fGB"
                          % (t_ded, peak))
        except torch.cuda.OutOfMemoryError:
            res[n_sub] = (None, vis.numel() * 8 / 1e9, "OOM in gather")
        del vis
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chgroup", type=int, default=15)
    ap.add_argument("--coarse-dm", type=float, default=1200.0)
    ap.add_argument("--true-dm", type=float, default=1203.0)
    a = ap.parse_args()

    print("device: %s   %s" % (a.device, torch.cuda.get_device_name(
        int(a.device.split(":")[1]))))
    print("shapes: n_fv/block=%d  NBASE=%d  NCHAN=%d (csf %d)  T=%.3f us"
          % (N_FV_PER_BLOCK, NBASE, NCHAN_EFF, CSF, T_US))
    print("chgroup %d  coarse_dm %.1f  true_dm %.1f"
          % (a.chgroup, a.coarse_dm, a.true_dm))
    print()

    print("=== ACCURACY: recovered peak of a dispersed delta ===")
    acc = accuracy(a.chgroup, a.coarse_dm, a.true_dm, device=a.device)
    base = acc[1]
    for n_sub in N_SUBS:
        print("  n_sub=%-3d quantisation %8.3f us   peak %.4f   x%.3f"
              % (n_sub, T_US / n_sub, acc[n_sub], acc[n_sub] / base))

    print()
    print("=== REAL TIME: gather cost per block, and the vis the caller "
          "must produce ===")
    tm = timing(a.chgroup, a.coarse_dm, device=a.device)
    t1 = tm.get(1, (None,))[0]
    print("  %-8s %-12s %-12s %-10s %s"
          % ("n_sub", "gather ms", "vs n_sub=1", "vis GB", "note"))
    for n_sub in N_SUBS:
        ms, gb, note = tm[n_sub]
        print("  %-8d %-12s %-12s %-10s %s"
              % (n_sub,
                 "OOM" if ms is None else "%.2f" % ms,
                 "-" if (ms is None or not t1) else "x%.2f" % (ms / t1),
                 "-" if gb is None else "%.2f" % gb,
                 note))
    print()
    print("  NOTE: the gather is only part of it. Producing vis at "
          "n_sub x finer cadence means the fast-corr GEMM emits n_sub x "
          "more time samples, and every byte between it and this call "
          "grows with it. Measure that separately before believing any "
          "n_sub is affordable.")


if __name__ == "__main__":
    main()
