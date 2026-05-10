"""Microbench + correctness check for the on-wire fused fluff Triton kernel.

Compares the Phase 14 path

    raw bytes (on-wire) -> fused_int4_unpack_onwire_triton -> (R, I) fp16 GEMM

against the Phase 12 reference (Stage 1 fp32 transpose + Triton fused
Stages 2+3) at the production op-point.

Run on h01 with::

    PATH=/home/ubuntu/miniforge3/envs/dsa110-rt/bin:$PATH \\
    PYTHONPATH=/home/ubuntu/proj/dsa110-rt/src:/home/ubuntu/proj/dsa110-rt \\
    CUDA_VISIBLE_DEVICES=1 \\
    python bench/microbench_triton_unpack_onwire.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    unpack_int4_split,
)
from dsart.services.triton_unpack_onwire import (  # noqa: E402
    fused_int4_unpack_onwire_triton,
)


def _make_raw_block(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = NPACKETS_PER_BLOCK * NANTS * NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL
    return rng.integers(0, 256, size=n, dtype=np.uint8)


def _time_one(callable_, n_iters: int = 5):
    samples = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        callable_()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    a = np.asarray(samples)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 99))


def main():
    if not torch.cuda.is_available():
        print("CUDA not available - bailing")
        sys.exit(1)
    dev = torch.device("cuda:0")
    print("# device:", torch.cuda.get_device_name(dev))
    print("# torch: ", torch.__version__)
    import triton
    print("# triton:", triton.__version__)

    raw_np = _make_raw_block(seed=42)
    raw_gpu = torch.as_tensor(raw_np, device=dev).contiguous()
    print(f"# raw bytes: {raw_np.nbytes / 1024 / 1024:.1f} MB")

    R_ref, I_ref = unpack_int4_split(raw_np, device=dev, out_dtype=torch.float16)
    torch.cuda.synchronize()
    print(f"# ref shape:  R={tuple(R_ref.shape)}  I={tuple(I_ref.shape)}  dtype={R_ref.dtype}")

    R14, I14 = fused_int4_unpack_onwire_triton(
        raw_gpu,
        NPACKETS=NPACKETS_PER_BLOCK,
        NANTS=NANTS,
        NCHAN=NCHAN_PER_CHGROUP,
        NTIMES=NTIMES_PER_PACKET,
        NPOL=NPOL,
        scale=0.05,
        out_dtype=torch.float16,
    )
    torch.cuda.synchronize()
    print(f"# new shape:  R={tuple(R14.shape)}  I={tuple(I14.shape)}  dtype={R14.dtype}")

    print("\n## Correctness")
    r_eq = torch.equal(R_ref, R14)
    i_eq = torch.equal(I_ref, I14)
    print(f"  R bit-equal: {r_eq}")
    print(f"  I bit-equal: {i_eq}")
    if not (r_eq and i_eq):
        rd = (R_ref.float() - R14.float()).abs()
        idd = (I_ref.float() - I14.float()).abs()
        print(f"  R: max abs diff={rd.max().item():.4e}  mean={rd.mean().item():.4e}")
        print(f"  I: max abs diff={idd.max().item():.4e}  mean={idd.mean().item():.4e}")
        sys.exit(2)

    for _ in range(3):
        unpack_int4_split(raw_np, device=dev, out_dtype=torch.float16)
        fused_int4_unpack_onwire_triton(
            raw_gpu,
            NPACKETS=NPACKETS_PER_BLOCK, NANTS=NANTS, NCHAN=NCHAN_PER_CHGROUP,
            NTIMES=NTIMES_PER_PACKET, NPOL=NPOL,
            scale=0.05, out_dtype=torch.float16,
        )
    torch.cuda.synchronize()

    print("\n## Timing")
    n_iters = 10

    def _ref_call():
        unpack_int4_split(raw_np, device=dev, out_dtype=torch.float16)

    def _new_call():
        fused_int4_unpack_onwire_triton(
            raw_gpu,
            NPACKETS=NPACKETS_PER_BLOCK, NANTS=NANTS, NCHAN=NCHAN_PER_CHGROUP,
            NTIMES=NTIMES_PER_PACKET, NPOL=NPOL,
            scale=0.05, out_dtype=torch.float16,
        )

    def _ref_call_gpu_only():
        from dsart.services.triton_unpack import fused_int4_unpack_triton  # noqa: PLC0415
        fp32_2d = raw_gpu.view(torch.float32).view(
            NPACKETS_PER_BLOCK * NANTS, NCHAN_PER_CHGROUP,
        )
        fp32_T = fp32_2d.t().contiguous()
        bytes_T_layout = fp32_T.view(torch.uint8).view(
            NCHAN_PER_CHGROUP, NPACKETS_PER_BLOCK, NANTS,
            NTIMES_PER_PACKET, NPOL,
        )
        fused_int4_unpack_triton(bytes_T_layout, scale=0.05, out_dtype=torch.float16)

    print(f"  iters per measurement: {n_iters}")
    print(f"  {'path':<55s} {'mean':>9s} {'p50':>9s} {'p99':>9s}  ms")
    print("  " + "-" * 90)
    for name, fn in [
        ("Phase 12 ref incl H2D (unpack_int4_split full)", _ref_call),
        ("Phase 12 ref GPU-only (Stage1 + fused fluff)", _ref_call_gpu_only),
        ("Phase 14 NEW GPU-only (on-wire fused)", _new_call),
    ]:
        m, p50, p99 = _time_one(fn, n_iters=n_iters)
        print(f"  {name:<55s} {m:9.2f} {p50:9.2f} {p99:9.2f}")

    m_ref_gpu, _, _ = _time_one(_ref_call_gpu_only, n_iters=n_iters)
    m_new, _, _ = _time_one(_new_call, n_iters=n_iters)
    print(f"\n  GPU-only Phase 14 / Phase 12 = {m_new / m_ref_gpu:.2f}x of reference")
    print(f"  GPU-only saving: {m_ref_gpu - m_new:+.2f} ms / block")


if __name__ == "__main__":
    main()
