"""B2 - boxcar detector cost per image at 262 us sampling.

Brackets the detector:
  floor   : one streaming read of the image cube (amax over time chunks)
  prod    : cumsum + production Triton multi_boxcar_argmax (per-voxel max+argmax
            written), i.e. today's design at the new shape
  chunk   : cumsum + prototype boxcar_chunkmax (only per-(pixel, 64-sample
            chunk) maxima written)
Today: 5.8 us/image at b1..b64 (37.7 ms / 6528 images, n02 2026-09-22).
Shapes: T=512 (one 134 ms block at 262 us), F fine DMs per pass, grid 256^2 or
256x128 (FoV/2), widths b1..b64 (K=7) and b1..b256 (K=9).  ~2 min.
"""
from __future__ import annotations

import torch

from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run

try:
    from dsart.detector.triton_boxcar import multi_boxcar_argmax_triton
except Exception:                       # dsart not importable -> skip prod case
    multi_boxcar_argmax_triton = None


def make(T, F, H, W, maxw, dev):
    x = torch.randn(T + maxw, F, H, W, device=dev, dtype=torch.float16)
    return x


def run_case(kind, T, F, H, W, K, dev, reps):
    widths = torch.tensor([2 ** i for i in range(K)], dtype=torch.int32, device=dev)
    maxw = int(widths.max())
    offsets = (maxw // 2 - widths // 2).to(torch.int32)
    sig = (1.0 / torch.sqrt(widths.float())).contiguous()
    x = make(T, F, H, W, maxw, dev)
    cs = torch.empty(T + maxw + 1, F, H, W, device=dev, dtype=torch.float16)

    def do_cumsum():
        cs[0].zero_()
        torch.cumsum(x, dim=0, out=cs[1:])

    if kind == "floor":
        fn = lambda: x[:T].view(T // 64, 64, F, H, W).amax(dim=1)      # noqa: E731
    elif kind == "cumsum":
        fn = do_cumsum
    elif kind == "prod":
        if multi_boxcar_argmax_triton is None:
            raise RuntimeError("dsart.detector.triton_boxcar not importable")
        do_cumsum()
        fn = lambda: multi_boxcar_argmax_triton(cs, widths=widths, offsets=offsets,  # noqa: E731
                                                sigma_inv=sig, n_out=T, t_base=0)
    elif kind == "chunk":
        from kernels import boxcar_chunkmax
        do_cumsum()
        flat = cs.view(cs.shape[0], -1)
        fn = lambda: boxcar_chunkmax(flat, widths, offsets, sig, T, chunk=64)  # noqa: E731
    else:
        raise ValueError(kind)
    ms = cuda_time_ms(fn, reps=reps)
    del x, cs
    torch.cuda.empty_cache()
    return {"ms": ms, "us_per_image": 1e3 * ms / (T * F)}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--F", type=int, default=8)
    args = ap.parse_args()
    if args.selftest:
        print("b2: run b0_selftest.py for kernel correctness")
        return
    gpu_busy_guard(args.force)
    rec = Recorder("b2_detector", args)
    for H, W in ((256, 256), (256, 128)):
        for K in (7, 9):
            for kind in ("floor", "cumsum", "prod", "chunk"):
                try_run(rec, f"{kind}_{H}x{W}_K{K}",
                        lambda: run_case(kind, args.T, args.F, H, W, K, args.device, args.reps),
                        kind=kind, H=H, W=W, K=K, T=args.T, F=args.F)


if __name__ == "__main__":
    main()
