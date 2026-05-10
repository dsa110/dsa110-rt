"""Per-stage timing breakdown for the slow correlator + unpack microbench.

Mirrors ``bench/profile_fast_path_K1.py`` but for the M2 slow correlator
service. Reports per-block GPU ms for:

  A. unpack_int4_split    (= H2D copy + fp32 transpose + ASR fluff to fp16)
     A1. h2d_copy            (raw uint8 → device)
     A2. fp32_transpose      (the 384×196608 transpose)
     A3. permute_to_gemm     ((C, P, A, 2t, 2p) → (C, 2t, 2p, P, A))
     A4. asr_fluff           (int8 → real/imag fp16 with scale)
  B. SlowCorrKernel.compute_split  (4 fp16 GEMMs + reduce + cast)
  C. pack_bada_block       (cfp32 → uint8 view; should be ~0)

  Total wall vs RT block period (134.22 ms).

CLI::

    python bench/profile_slow_corr_block.py \\
        --device cuda --warmup 3 --n-blocks 10 --report-dir /tmp/slow-prof
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BADA_BYTES_PER_INTEGRATION,
    BLOCK_DURATION_S,
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    SlowCorrKernel,
    pack_bada_block,
    unpack_int4_split,
)
from dsart.services.slow_corr_kernel import _GEMM_LAYOUT_SHAPE  # noqa: E402

LEGACY_FLUFF_SCALE = 0.05

# ---------------------------------------------------------------------------
# Sub-stage instrumented unpack (mirrors src/dsart/services/slow_corr_kernel.py
# unpack_int4_split, but with CUDA event timers around each step).
# ---------------------------------------------------------------------------


def _ev():
    return torch.cuda.Event(enable_timing=True)


def unpack_instrumented(
    raw_arr: np.ndarray, *, device: torch.device, out_dtype=torch.float16,
    counters: dict | None = None,
):
    """Sub-stage instrumented mirror of ``unpack_int4_split``.

    Mirrors the auto-pin path (``maybe_register_host_buffer``) so the
    H2D timing here matches the production hot path. Without this the
    bench would always see the slow pageable ~66 ms H2D regardless of
    whether the production code is pinning.
    """
    if counters is None:
        counters = {}

    def _add(name, e0, e1):
        e1.synchronize()
        counters.setdefault(name, []).append(e0.elapsed_time(e1))

    # Auto-pin (idempotent; first call per buffer pays cudaHostRegister).
    if device.type == "cuda":
        from dsart.services.host_pin import maybe_register_host_buffer
        maybe_register_host_buffer(raw_arr)

    # A1. H2D copy
    e0, e1 = _ev(), _ev()
    e0.record()
    raw_t = torch.as_tensor(
        raw_arr.reshape(-1) if raw_arr.ndim != 1 else raw_arr,
        device=device,
    )
    e1.record()
    _add("A1_h2d_copy", e0, e1)

    n_fp32_cols = NCHAN_PER_CHGROUP
    fp32_2d = raw_t.view(torch.float32).view(
        NPACKETS_PER_BLOCK * NANTS, n_fp32_cols,
    )

    # A2. fp32 transpose
    e0, e1 = _ev(), _ev()
    e0.record()
    fp32_T = fp32_2d.t().contiguous()
    e1.record()
    _add("A2_fp32_transpose", e0, e1)

    bytes_T_layout = fp32_T.view(torch.uint8).view(
        NCHAN_PER_CHGROUP, NPACKETS_PER_BLOCK, NANTS,
        NTIMES_PER_PACKET, NPOL,
    )

    # A3. permute to GEMM layout
    e0, e1 = _ev(), _ev()
    e0.record()
    bytes_gemm = bytes_T_layout.permute(0, 3, 4, 1, 2).contiguous()
    e1.record()
    _add("A3_permute_to_gemm", e0, e1)

    # A4. ASR fluff (int8 → fp16)
    e0, e1 = _ev(), _ev()
    e0.record()
    raw_i8 = bytes_gemm.view(torch.int8)
    real_i8 = (raw_i8 << 4) >> 4
    imag_i8 = raw_i8 >> 4
    real = (real_i8.to(out_dtype) * LEGACY_FLUFF_SCALE).reshape(_GEMM_LAYOUT_SHAPE)
    imag = (imag_i8.to(out_dtype) * LEGACY_FLUFF_SCALE).reshape(_GEMM_LAYOUT_SHAPE)
    e1.record()
    _add("A4_asr_fluff", e0, e1)

    return real, imag


# ---------------------------------------------------------------------------
# Main bench driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-blocks", type=int, default=10)
    ap.add_argument("--report-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(0xC0DE)

    # Synthetic raw fada bytes (uint8). The actual values don't affect timing.
    raw = rng.integers(0, 256, size=FADA_BYTES_PER_BLOCK, dtype=np.uint8)

    kernel = SlowCorrKernel(device=device)

    counters: dict[str, list[float]] = {}

    def _add(name, e0, e1):
        e1.synchronize()
        counters.setdefault(name, []).append(e0.elapsed_time(e1))

    wall_per_block = []
    for i in range(args.warmup + args.n_blocks):
        record = i >= args.warmup
        sub_counters = counters if record else {}

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # A. Unpack (instrumented)
        e0, e1 = _ev(), _ev()
        e0.record()
        real_v, imag_v = unpack_instrumented(
            raw, device=device, counters=sub_counters,
        )
        e1.record()
        if record:
            _add("A_unpack_total", e0, e1)

        # B. compute_split
        e0, e1 = _ev(), _ev()
        e0.record()
        vis = kernel.compute_split(real_v, imag_v)
        e1.record()
        if record:
            _add("B_compute_split", e0, e1)
        del real_v, imag_v

        # C. pack_bada_block (CPU work; time on host)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pack0 = time.perf_counter()
        out_bytes = pack_bada_block(vis)
        if record:
            counters.setdefault("C_pack_bada_block", []).append(
                (time.perf_counter() - t_pack0) * 1000.0
            )
        del vis, out_bytes

        if device.type == "cuda":
            torch.cuda.synchronize()
        if record:
            wall_per_block.append((time.perf_counter() - t0) * 1000.0)

    block_period_ms = BLOCK_DURATION_S * 1000.0

    print()
    print(f"{'phase':<30s} {'mean':>9s} {'p50':>9s} {'p99':>9s}  {'% of p50 wall':>14s}")
    print("-" * 80)

    p50_wall = float(np.median(wall_per_block))
    order = [
        "A1_h2d_copy", "A2_fp32_transpose", "A3_permute_to_gemm",
        "A4_asr_fluff",
        "A_unpack_total",
        "B_compute_split", "C_pack_bada_block",
    ]
    summary = {"block_period_ms": block_period_ms,
               "wall_per_block_ms": wall_per_block}
    for name in order:
        if name not in counters:
            continue
        vals = np.array(counters[name])
        mean = float(vals.mean())
        p50 = float(np.median(vals))
        p99 = float(np.percentile(vals, 99))
        pct = 100.0 * p50 / p50_wall
        summary[name] = {"mean": mean, "p50": p50, "p99": p99}
        print(f"{name:<30s} {mean:9.2f} {p50:9.2f} {p99:9.2f}  {pct:13.1f}%")

    print("-" * 80)
    wall_mean = float(np.mean(wall_per_block))
    wall_p99 = float(np.percentile(wall_per_block, 99))
    rt_factor = p50_wall / block_period_ms
    print(f"{'WALL per block':<30s} {wall_mean:9.2f} {p50_wall:9.2f} {wall_p99:9.2f}")
    print(f"realtime block period = {block_period_ms:.2f} ms; "
          f"wall p50 / block_period = {rt_factor:.2f}x")
    if rt_factor < 1.0:
        print(f"  → REAL-TIME OK (margin {(1-rt_factor)*100:.1f}%)")
    else:
        print(f"  → {rt_factor:.2f}x SLOWER THAN REAL-TIME")

    if args.report_dir:
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.report_dir) / "slow_corr_breakdown.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
