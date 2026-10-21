"""B3 - one imaging "owner" GPU, end to end, at the 4x op-point.

The key feasibility number: can ONE 2080 Ti image + search one coarse-DM
bucket in real time?  Per 512-sample (134.2 ms at 262 us) chunk and per pass of
F fine DMs it runs
   combine_cells (S sub-band streams x C cells, cint8) -> irfft2 (fp16)
   -> cumsum -> boxcar (prod per-voxel kernel, or prototype chunk-max)
and reports ms per chunk, us per image and the number of fine DMs one GPU can
sustain:  n_fdm_max = 134.2 ms / (ms per chunk per fine DM).
Model baseline: 1059 fine DMs over DM 30-1600 at 262 us / native channels,
i.e. ~44 per owner with 24 owners.  Defaults: S=128 (16 chgroups x 8 sub-bands),
C=1900 cells/stream (real-antpos count), spread=200 samples.  ~3 min.
"""
from __future__ import annotations

import math

import torch

from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run
from kernels import boxcar_chunkmax, combine_cells

try:
    from dsart.detector.triton_boxcar import multi_boxcar_argmax_triton
except Exception:
    multi_boxcar_argmax_triton = None

BLOCK_MS = 134.217728


def build(S, C, F, T, maxw, spread, H, W, dev):
    T_rows = T + maxw                         # rows needed incl. boxcar history
    T_in = T_rows + spread
    x = torch.randint(-20, 21, (S, T_in, C), device=dev, dtype=torch.int16)
    x = (x & 0xFF) | (torch.randint(-20, 21, (S, T_in, C), device=dev, dtype=torch.int16) << 8)
    # per-stream shift grows with fine-DM index (bucket spread) + stream frequency
    frac = torch.linspace(0, 1, S, device=dev)
    shifts = torch.stack([torch.round(frac * spread * (f + 1) / F) for f in range(F)]).to(torch.int32)
    Wh = W // 2 + 1
    cellpos = torch.randperm(H * Wh, device=dev)[:C].to(torch.int32)
    planes = torch.zeros(F * T_rows, 2 * H * Wh, device=dev, dtype=torch.float16)
    return x, shifts, cellpos, planes, T_rows


def run_case(det, S, C, F, T, K, H, W, spread, dev, reps):
    widths = torch.tensor([2 ** i for i in range(K)], dtype=torch.int32, device=dev)
    maxw = int(widths.max())
    offsets = (maxw // 2 - widths // 2).to(torch.int32)
    sig = (1.0 / torch.sqrt(widths.float())).contiguous()
    x, shifts, cellpos, planes, T_rows = build(S, C, F, T, maxw, spread, H, W, dev)
    Wh = W // 2 + 1
    cs = torch.empty(T_rows + 1, F, H, W, device=dev, dtype=torch.float16)
    stage = {}

    def combine():
        combine_cells(x, shifts, cellpos, planes, T_rows)

    def image():
        # fp16 (re, im) pairs -> complex32 view [F*T_rows, H, Wh]; fp16 cuFFT.
        uv = planes.view(F * T_rows, H, 2 * Wh).view(torch.complex32)
        img = torch.fft.irfft2(uv, s=(H, W))             # [F*T_rows, H, W] fp16
        cs[0].zero_()
        torch.cumsum(img.view(F, T_rows, H, W).transpose(0, 1), dim=0, out=cs[1:])

    def detect():
        if det == "prod":
            if multi_boxcar_argmax_triton is None:
                raise RuntimeError("dsart triton_boxcar not importable")
            return multi_boxcar_argmax_triton(cs, widths=widths, offsets=offsets,
                                              sigma_inv=sig, n_out=T, t_base=0)
        return boxcar_chunkmax(cs.view(cs.shape[0], -1), widths, offsets, sig, T, chunk=64)

    for name, fn in (("combine", combine), ("image", image), ("detect", detect)):
        stage[name] = cuda_time_ms(fn, reps=reps)
    tot = sum(stage.values())
    per_fdm = tot / F
    return {**{f"ms_{k}": v for k, v in stage.items()}, "ms_total": tot,
            "us_per_image": 1e3 * tot / (F * T),
            "n_fdm_max_realtime": math.floor(BLOCK_MS / per_fdm),
            "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--C", type=int, default=1900)
    ap.add_argument("--F", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--spread", type=int, default=200)
    args = ap.parse_args()
    if args.selftest:
        print("b3: run b0_selftest.py for kernel correctness")
        return
    gpu_busy_guard(args.force)
    rec = Recorder("b3_owner_e2e", args)
    for H, W in ((256, 256), (256, 128)):
        for K in (7, 9):
            for det in ("chunk", "prod"):
                for F in args.F:
                    torch.cuda.reset_peak_memory_stats(args.device)
                    try_run(rec, f"e2e_{det}_{H}x{W}_K{K}_F{F}",
                            lambda: run_case(det, args.S, args.C, F, args.T, K, H, W,
                                             args.spread, args.device, args.reps),
                            det=det, H=H, W=W, K=K, F=F, S=args.S, C=args.C, T=args.T)


if __name__ == "__main__":
    main()
