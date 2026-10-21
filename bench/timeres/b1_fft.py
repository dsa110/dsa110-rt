"""B1 - imaging FFT throughput (us per image) for candidate grids.

Answers: what does one dirty image cost on a 2080 Ti, and how much does the
FoV/2 trade save?  Today: 256x256 fp16 irfft2 = 2.1 us/image (n02, 09-22).

Cases: grid (256x256 | 256x128 | 192x192) x dtype (fp16 needs power-of-2)
x batch, plain irfft2 and irfft2 + fused-ish epilogue (scale * mask, fp16 out).
~1 min on one GPU.
"""
from __future__ import annotations

import torch

from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run

GRIDS = [(256, 256), (256, 128), (192, 192), (128, 128)]


def case(H, W, dtype, batch, epilogue, dev, reps):
    cdt = torch.complex32 if dtype == torch.float16 else torch.complex64
    x = torch.randn(batch, H, W // 2 + 1, 2, device=dev, dtype=dtype)
    x = torch.view_as_complex(x.float()).to(cdt) if cdt == torch.complex32 else torch.view_as_complex(x)
    mask = (torch.rand(H, W, device=dev) > 0.03).to(dtype)

    def run():
        img = torch.fft.irfft2(x, s=(H, W))
        if epilogue:
            img = (img * mask).to(torch.float16)
        return img

    ms = cuda_time_ms(run, reps=reps)
    return {"ms_per_batch": ms, "us_per_image": 1e3 * ms / batch,
            "images_per_s": batch / (ms * 1e-3)}


def selftest():
    x = torch.randn(4, 16, 9, dtype=torch.complex64)
    a = torch.fft.irfft2(x, s=(16, 16))
    assert a.shape == (4, 16, 16)
    print("b1 selftest ok")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--batches", type=int, nargs="*", default=[64, 256])
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    gpu_busy_guard(args.force)
    rec = Recorder("b1_fft", args)
    for H, W in GRIDS:
        pow2 = (H & (H - 1)) == 0 and (W & (W - 1)) == 0
        for dtype in ([torch.float16] if pow2 else []) + [torch.float32]:
            for b in args.batches:
                for epi in (False, True):
                    try_run(rec, f"irfft2_{H}x{W}_{str(dtype)[6:]}_b{b}{'_epi' if epi else ''}",
                            lambda: case(H, W, dtype, b, epi, args.device, args.reps),
                            H=H, W=W, dtype=str(dtype), batch=b, epilogue=epi)


if __name__ == "__main__":
    main()
