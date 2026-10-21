"""B4 - corr-node "X-engine" GEMM rates and PCIe copies (one corr GPU).

Answers: can ONE corr GPU correlate the slow (full 96-ant, XX/YY, K=4096) and
the fast products at 262 us, and what would a dedispersing correlator (stage-1
inside the GEMM: K spans channels x samples x pols per coarse DM) cost?
Complex GEMM is embedded as one real GEMM  [Ar Ai] @ [[Ar^T Ai^T],[Ai^T -Ar^T]]
(M=96, N=192, K=2k), fp16 inputs, fp32 accumulate.  Per 134.218 ms block:
  slow : batch 768 (384 ch x 2 pol), k=4096           (today 8.9 ms, n06)
  fast : batch 384 ch x 512 t, k=16 (8 samples x 2 pol, Stokes-I in K)
  ddc  : per coarse DM, batch n_sub x 512 t, k = 2 pol x (384/n_sub) ch x 8 samples
  int8 : torch._int_mm peak vs fp16 mm at 4096^3 (decides an IMMA correlator)
  pcie : pinned H2D of one 288 MiB fada block, D2H 0.5 GB
~2 min.  Run on a corr node's GPU during a gap (both GPUs idle).
"""
from __future__ import annotations

import torch

from common import Recorder, base_parser, cuda_time_ms, gpu_busy_guard, try_run

BLOCK_MS = 134.217728


def real_embed(batch, k, dev):
    a = torch.randn(batch, 96, 2 * k, device=dev, dtype=torch.float16)
    b = torch.randn(batch, 2 * k, 192, device=dev, dtype=torch.float16)
    return a, b


def gemm_case(batch_total, k, dev, reps, chunk=16384):
    """Time C = A@B for batch_total small GEMMs (chunked); return ms per block."""
    n = min(batch_total, chunk)
    a, b = real_embed(n, k, dev)
    out = torch.empty(n, 96, 192, device=dev, dtype=torch.float16)
    ms = cuda_time_ms(lambda: torch.bmm(a, b, out=out), reps=reps)
    ms_block = ms * batch_total / n
    flops = 2.0 * 96 * 192 * (2 * k) * batch_total
    return {"ms_per_block": ms_block, "tflops": flops / (ms_block * 1e-3) / 1e12,
            "frac_of_block": ms_block / BLOCK_MS}


def int8_case(n, dev, reps):
    a8 = torch.randint(-8, 8, (n, n), device=dev, dtype=torch.int8)
    b8 = torch.randint(-8, 8, (n, n), device=dev, dtype=torch.int8)
    ms8 = cuda_time_ms(lambda: torch._int_mm(a8, b8), reps=reps)
    a16, b16 = a8.half(), b8.half()
    ms16 = cuda_time_ms(lambda: torch.mm(a16, b16), reps=reps)
    ops = 2.0 * n ** 3
    return {"int8_tops": ops / ms8 / 1e9, "fp16_tflops": ops / ms16 / 1e9, "ratio": ms16 / ms8}


def pcie_case(dev, reps):
    h = torch.empty(288 * 2 ** 20, dtype=torch.uint8).pin_memory()
    d = torch.empty_like(h, device=dev)
    h2 = torch.empty(512 * 2 ** 20, dtype=torch.uint8).pin_memory()
    d2 = torch.empty_like(h2, device=dev)
    ms_h2d = cuda_time_ms(lambda: d.copy_(h, non_blocking=True), reps=reps)
    ms_d2h = cuda_time_ms(lambda: h2.copy_(d2, non_blocking=True), reps=reps)
    return {"h2d_ms_per_block": ms_h2d, "h2d_gbs": h.numel() / ms_h2d / 1e6,
            "d2h_gbs": h2.numel() / ms_d2h / 1e6}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-sub", type=int, nargs="*", default=[8, 16])
    args = ap.parse_args()
    if args.selftest:
        a, b = real_embed(2, 4, "cpu")
        assert torch.bmm(a.float(), b.float()).shape == (2, 96, 192)
        print("b4 selftest ok")
        return
    gpu_busy_guard(args.force)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    rec = Recorder("b4_xengine", args)
    dev, r = args.device, args.reps
    try_run(rec, "slow_k4096", lambda: gemm_case(768, 4096, dev, r, chunk=768), k=4096)
    for tint, k in ((32, 64), (16, 32), (8, 16), (4, 8)):
        try_run(rec, f"fast_tint{tint}", lambda: gemm_case(384 * (4096 // tint), k, dev, r),
                t_int_native=tint, k=k)
    for ns in args.n_sub:
        k = 2 * (384 // ns) * 8
        try_run(rec, f"ddc_per_coarse_dm_nsub{ns}", lambda: gemm_case(ns * 512, k, dev, r),
                n_sub=ns, k=k)
    try_run(rec, "int8_vs_fp16_4096", lambda: int8_case(4096, dev, r))
    try_run(rec, "pcie", lambda: pcie_case(dev, r))


if __name__ == "__main__":
    main()
