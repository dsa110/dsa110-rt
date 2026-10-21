"""B7 - corr-side stage-1 (per coarse DM, per sub-band) at the 4x op-point.

Stage-1 is a shift-and-add over the channels of a sub-band, per coarse DM:
    out[c, s, t, bl] = sum_{ch in s} vis[ch, t + shift[c, ch], bl]
which is exactly the structure of the tested prototype `combine_cells`
(streams -> channels, fine DMs -> coarse DMs, cells -> baselines).  Today's
production kernel re-reads its input per DM (0.966 + 0.753 * n_dm ms at x32 /
244 kHz); at 0.26 ms / 61 kHz that scales to ~12 ms per DM, so 4x needs a
kernel that reuses the input across DMs.  This times one chgroup per 134 ms
block: 192 channels (61 kHz) x 3321 core baselines x 512 samples, n_sub
sub-bands, n_c coarse DMs, shifts up to `spread` samples.  ~1 min.
"""
from __future__ import annotations

import torch

from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run
from kernels import combine_cells

BLOCK_MS = 134.217728


def run_case(n_c, n_sub, n_ch, n_bl, T, spread, dev, reps):
    per = n_ch // n_sub
    T_in = T + spread
    x = torch.randint(-3000, 3000, (per, T_in, n_bl), device=dev, dtype=torch.int16)
    frac = torch.linspace(0, 1, per, device=dev)
    shifts = torch.stack([torch.round(frac * spread * (c + 1) / n_c) for c in range(n_c)]).to(torch.int32)
    cellpos = torch.arange(n_bl, device=dev, dtype=torch.int32)
    out = torch.zeros(n_c * T, 2 * n_bl, device=dev, dtype=torch.float16)

    def stage1():
        for _ in range(n_sub):                 # one call per sub-band (same shapes)
            combine_cells(x, shifts, cellpos, out, T)

    ms = cuda_time_ms(stage1, reps=reps)
    return {"ms_per_block": ms, "frac_of_block": ms / BLOCK_MS,
            "per_coarse_dm_ms": ms / n_c}


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    if args.selftest:
        print("b7: kernel correctness is b0_selftest (combine_cells)")
        return
    gpu_busy_guard(args.force)
    rec = Recorder("b7_stage1", args)
    for n_c, n_sub, n_ch, spread, tag in ((31, 2, 192, 66, "tier A 4x, 61 kHz"),
                                          (16, 4, 192, 40, "tier A 4x, 61 kHz, 4 sub-bands"),
                                          (31, 2, 96, 66, "tier A 4x, 122 kHz"),
                                          (8, 4, 48, 64, "today's shape (x32 in time units)")):
        T = 512 if "today" not in tag else 128
        try_run(rec, f"stage1_nc{n_c}_nsub{n_sub}_nch{n_ch}",
                lambda: run_case(n_c, n_sub, n_ch, 3321, T, spread, args.device, args.reps),
                n_c=n_c, n_sub=n_sub, n_ch=n_ch, T=T, spread=spread, tag=tag)


if __name__ == "__main__":
    main()
