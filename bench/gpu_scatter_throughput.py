#!/usr/bin/env python3
"""bench/gpu_scatter_throughput.py - M7.4.1 GPU-scatter A/B micro-bench.

Drives the production geometry (N_corr=16, T_det=192, N_grid=256,
n_filled_max=5000) through BOTH the M7.4 CPU-dense path and the
M7.4.1 compact + GPU-scatter path back-to-back on a synthetic RxRing
and reports per-stage wall time (CPU work, H2D, GPU scatter, total)
plus an end-to-end correctness check (the two dense planes MUST be
byte-identical).

The bench is the perf gate for M7.4.1: GPU-scatter MUST land the
per-cube build_cube budget so that the full pipeline (build + L1 +
detector) clears 134 ms = 7.45 cubes/s.

Usage:
    python -m bench.gpu_scatter_throughput \\
        [--n-iters 100] [--n-warmup 5]    \\
        [--n-corr 16] [--n-coarse-dm 8]    \\
        [--n-filled 5000] [--t-det 192]    \\
        [--n-grid 256] [--t-stream 270]    \\
        [--out bench/reports/<UTC>/gpu_scatter/]

Outputs (under --out):
    timings.ndjson  - one line per iter: {iter, path, stage, ns}
    summary.json    - config + percentile summary per stage per path
    bench.log       - human-readable log
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.transport.recv_ring import (  # noqa: E402
    RxRing,
    RxRingDims,
    VF_DATA_PRESENT,
    _get_lib,
)

_LOG = logging.getLogger("gpu_scatter_throughput")


# ---------------------------------------------------------------------------
# Synthetic ring fill - production-realistic geometry.
# ---------------------------------------------------------------------------


def _fill_ring(
    ring: RxRing,
    *,
    n_corr: int,
    n_coarse_dm: int,
    t_fill: int,
    n_filled: int,
    seed: int,
) -> None:
    """Fill ring rows [0, t_fill) for all (corr, dm) with deterministic data.

    Each slot is written with VF_DATA_PRESENT and a non-trivial scale to
    exercise the validity-mask code path. The payload is N(0, 50)
    int8s clipped to [-127, 127] (matches typical post-quant SK output).

    We only fill t_fill rows because every benchmark iteration reads
    specnum_start=0..(t_det-1) (the read is purely memory traffic, not
    data-novelty sensitive). For production geometry filling all
    32768 ring rows is ~10s of pure ctypes overhead so we skip it.
    """
    rng = np.random.default_rng(seed)
    block = rng.integers(
        -100, 100,
        size=(n_corr, n_coarse_dm, t_fill, n_filled * 2),
        dtype=np.int8,
    )
    for c in range(n_corr):
        for dm in range(n_coarse_dm):
            for tt in range(t_fill):
                ring.write_slot(
                    corr=c,
                    dm=dm,
                    t_seq=tt,
                    payload=block[c, dm, tt].tobytes(),
                    validity_flags=VF_DATA_PRESENT,
                    scale=0.05 + 0.001 * c,
                    offset=0.125,
                )


def _make_lut(n_corr: int, n_filled: int, n_grid: int) -> np.ndarray:
    """Per-corr LUT: cell k -> (ix=k%n_grid, iy=(k+corr)%n_grid).

    Spreads cells across the dense plane the same way the C parity test
    does, so we know the LUT is exercised against the full N_grid*N_grid
    address space.
    """
    lut = np.zeros((n_corr, n_filled), dtype=np.int32)
    for c in range(n_corr):
        for k in range(n_filled):
            ix = k % n_grid
            iy = (k + c) % n_grid
            lut[c, k] = ix * n_grid + iy
    return lut


# ---------------------------------------------------------------------------
# CUDA event timing helpers.
# ---------------------------------------------------------------------------


def _cuda_event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Wall-clock ms between two CUDA events. Both must be recorded.
    Caller is responsible for synchronization."""
    return float(start.elapsed_time(end))


# ---------------------------------------------------------------------------
# Path A: CPU-dense scatter + H2D (M7.4 baseline).
# ---------------------------------------------------------------------------


def _bench_cpu_dense(
    *,
    ring: RxRing,
    n_corr: int,
    n_coarse_dm: int,
    t_det: int,
    t_stream: int,
    n_grid: int,
    n_filled: int,
    owned_dm: int,
    device: torch.device,
    n_iters: int,
    n_warmup: int,
    lut: np.ndarray,
    nfp: np.ndarray,
) -> List[Dict[str, float]]:
    """Time the CPU-dense path: assemble_dense_block + H2D (pinned)."""

    bytes_per_cell = 2  # cint8 complex
    dense_shape = (n_corr, t_stream, 2, n_grid, n_grid)
    dense_bytes = int(np.prod(dense_shape)) * 1  # int8

    # Allocate persistent pinned host buffer (matches production path).
    pinned_host = torch.zeros(dense_shape, dtype=torch.int8, pin_memory=True)
    pinned_view_np = pinned_host.numpy()  # zero-copy view

    # Allocate persistent GPU dense buffer.
    dense_gpu = torch.zeros(dense_shape, dtype=torch.int8, device=device)

    # Per-(corr, t) scale/offset sidecars (T_stream-strided so they match
    # the GPU kernel's stride expectation).
    scale_host = np.zeros((n_corr, t_stream), dtype=np.float32)
    offre_host = np.zeros((n_corr, t_stream), dtype=np.float32)
    offim_host = np.zeros((n_corr, t_stream), dtype=np.float32)
    valid_host = np.zeros((t_det,), dtype=bool)

    results: List[Dict[str, float]] = []

    # Re-read the same cube each iter (we only filled [0, t_det) rows).
    specnum = 0
    for it in range(n_iters + n_warmup):
        t0 = time.perf_counter_ns()

        # C-side dense scatter (CPU). Returns numpy arrays that share
        # memory with the C-allocated tmp buffer; we copy into our
        # pinned host buffer below for fair pinned-H2D timing.
        (dense_c, scale_c, offre_c, offim_c, valid_c,
         n_o, n_p, n_d) = ring.assemble_dense_block(
            specnum_start=specnum,
            t_det=t_det,
            n_grid=n_grid,
            owned_dm=owned_dm,
            n_filled_per_corr=nfp,
            linear_lut_strided=lut,
            out_t_stride=t_stream,
            compute_half=0,
        )

        t_scatter_end = time.perf_counter_ns()

        # Copy C result into pinned host buffer (matches the production
        # path which writes directly into the pinned buffer via the C
        # API's out= argument). For this micro-bench we measure the
        # copy as part of the CPU work.
        pinned_view_np[...] = dense_c
        scale_host[...] = scale_c
        offre_host[...] = offre_c
        offim_host[...] = offim_c
        valid_host[...] = valid_c

        t_cpu_end = time.perf_counter_ns()

        # H2D timing via CUDA events.
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        h2d_start.record()
        dense_gpu.copy_(pinned_host, non_blocking=True)
        h2d_end.record()
        torch.cuda.synchronize()
        h2d_ms = _cuda_event_ms(h2d_start, h2d_end)

        t_total_end = time.perf_counter_ns()

        if it >= n_warmup:
            results.append(
                {
                    "iter": it - n_warmup,
                    "cpu_scatter_ns": float(t_scatter_end - t0),
                    "cpu_copy_ns": float(t_cpu_end - t_scatter_end),
                    "cpu_total_ns": float(t_cpu_end - t0),
                    "h2d_ms": h2d_ms,
                    "wall_total_ns": float(t_total_end - t0),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Path B: compact + GPU scatter (M7.4.1).
# ---------------------------------------------------------------------------


def _bench_gpu_scatter(
    *,
    ring: RxRing,
    n_corr: int,
    n_coarse_dm: int,
    t_det: int,
    t_stream: int,
    n_grid: int,
    n_filled: int,
    owned_dm: int,
    device: torch.device,
    n_iters: int,
    n_warmup: int,
    lut: np.ndarray,
    nfp: np.ndarray,
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    """Time the GPU-scatter path: assemble_compact_block + small H2D +
    GPU scatter kernel.

    Returns (timing records, last dense_gpu) so the caller can compare
    byte-equality against the CPU-dense reference.
    """
    from dsart.transport.gpu_scatter import (
        scatter_compact_to_dense,
        zero_dense_rows,
    )

    dense_shape = (n_corr, t_stream, 2, n_grid, n_grid)

    # Persistent GPU buffers (production pattern).
    dense_gpu = torch.zeros(dense_shape, dtype=torch.int8, device=device)
    cells_shape = (n_corr, t_det, n_filled * 2)
    cells_gpu = torch.zeros(cells_shape, dtype=torch.int8, device=device)

    # Static GPU tensors for LUT and n_filled_per_corr (transfer once).
    lut_gpu = torch.from_numpy(np.ascontiguousarray(lut)).to(
        device, non_blocking=False,
    )
    nfp_gpu = torch.from_numpy(np.ascontiguousarray(nfp)).to(
        device, non_blocking=False,
    )

    # Pinned host buffer for compact cells (production pattern).
    cells_host_pinned = torch.zeros(cells_shape, dtype=torch.int8,
                                    pin_memory=True)
    cells_host_np = cells_host_pinned.numpy()

    scale_host = np.zeros((n_corr, t_stream), dtype=np.float32)
    offre_host = np.zeros((n_corr, t_stream), dtype=np.float32)
    offim_host = np.zeros((n_corr, t_stream), dtype=np.float32)
    valid_host = np.zeros((t_det,), dtype=bool)

    # Warm up the kernels (NVRTC compile cost is ~0.6s first call).
    cells_dummy = torch.zeros((n_corr, t_det, n_filled * 2),
                              dtype=torch.int8, device=device)
    zero_dense_rows(dense=dense_gpu, t_det=t_det)
    scatter_compact_to_dense(
        cells_packed=cells_dummy,
        lut=lut_gpu,
        n_filled_per_corr=nfp_gpu,
        dense=dense_gpu,
        t_det=t_det,
        n_grid=n_grid,
        n_filled_max=n_filled,
    )
    torch.cuda.synchronize()

    results: List[Dict[str, float]] = []
    # Re-read the same cube each iter (we only filled [0, t_det) rows).
    specnum = 0
    for it in range(n_iters + n_warmup):
        t0 = time.perf_counter_ns()

        # C-side compact assembly. Returns a numpy array of shape
        # (n_corr, t_det, n_filled * 2) cint8.
        (cells_c, scale_c, offre_c, offim_c, valid_c,
         n_o, n_p, n_d) = ring.assemble_compact_block(
            specnum_start=specnum,
            t_det=t_det,
            owned_dm=owned_dm,
            n_filled_per_corr=nfp,
            n_filled_max=n_filled,
            sidecar_t_stride=t_stream,
            compute_half=0,
        )

        t_compact_end = time.perf_counter_ns()

        # Copy into pinned host buffer.
        cells_host_np[...] = cells_c
        scale_host[...] = scale_c
        offre_host[...] = offre_c
        offim_host[...] = offim_c
        valid_host[...] = valid_c

        t_cpu_end = time.perf_counter_ns()

        # H2D the compact bytes only.
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        h2d_start.record()
        cells_gpu.copy_(cells_host_pinned, non_blocking=True)
        h2d_end.record()

        # GPU scatter: zero rows [0, t_det) then scatter.
        scatter_start = torch.cuda.Event(enable_timing=True)
        scatter_end = torch.cuda.Event(enable_timing=True)
        scatter_start.record()
        zero_dense_rows(dense=dense_gpu, t_det=t_det)
        scatter_compact_to_dense(
            cells_packed=cells_gpu,
            lut=lut_gpu,
            n_filled_per_corr=nfp_gpu,
            dense=dense_gpu,
            t_det=t_det,
            n_grid=n_grid,
            n_filled_max=n_filled,
        )
        scatter_end.record()

        torch.cuda.synchronize()
        h2d_ms = _cuda_event_ms(h2d_start, h2d_end)
        scatter_ms = _cuda_event_ms(scatter_start, scatter_end)

        t_total_end = time.perf_counter_ns()

        if it >= n_warmup:
            results.append(
                {
                    "iter": it - n_warmup,
                    "cpu_compact_ns": float(t_compact_end - t0),
                    "cpu_copy_ns": float(t_cpu_end - t_compact_end),
                    "cpu_total_ns": float(t_cpu_end - t0),
                    "h2d_ms": h2d_ms,
                    "scatter_ms": scatter_ms,
                    "gpu_total_ms": h2d_ms + scatter_ms,
                    "wall_total_ns": float(t_total_end - t0),
                }
            )

    return results, dense_gpu


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


def _summarise(records: List[Dict[str, float]],
               keys: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        xs = [float(r[k]) for r in records if k in r]
        if not xs:
            out[k] = {"n": 0}
            continue
        out[k] = {
            "n": float(len(xs)),
            "mean": float(np.mean(xs)),
            "p50": _pct(xs, 50),
            "p90": _pct(xs, 90),
            "p99": _pct(xs, 99),
            "max": float(np.max(xs)),
        }
    return out


def _print_summary(name: str, recs: List[Dict[str, float]],
                   keys: List[str]) -> None:
    s = _summarise(recs, keys)
    print(f"=== {name} (n={len(recs)}) ===")
    print(f"{'stage':<25} {'mean':>10} {'p50':>10} {'p90':>10} "
          f"{'p99':>10} {'max':>10}")
    for k in keys:
        st = s[k]
        if st.get("n", 0) == 0:
            print(f"  {k:<23} (no data)")
            continue
        unit = "ms" if k.endswith("_ms") else "ms"
        scale = 1.0 if k.endswith("_ms") else 1e-6
        print(
            f"  {k:<23} {st['mean']*scale:>9.3f}{unit} "
            f"{st['p50']*scale:>9.3f}{unit} {st['p90']*scale:>9.3f}{unit} "
            f"{st['p99']*scale:>9.3f}{unit} {st['max']*scale:>9.3f}{unit}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iters", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-corr", type=int, default=16)
    ap.add_argument("--n-coarse-dm", type=int, default=8)
    ap.add_argument("--t-buf", type=int, default=512,
                    help="Ring depth. Production uses 32768 but we only "
                    "need enough to cover (n_iters * t_det). Keep small "
                    "so ring-fill stays fast.")
    ap.add_argument("--n-filled", type=int, default=5000)
    ap.add_argument("--t-det", type=int, default=192)
    ap.add_argument("--t-stream", type=int, default=270)
    ap.add_argument("--n-grid", type=int, default=256)
    ap.add_argument("--owned-dm", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234567)
    ap.add_argument("--out", default=None,
                    help="Output dir (default: bench/reports/<UTC>/gpu_scatter/)")
    ap.add_argument("--skip-correctness", action="store_true",
                    help="Skip the byte-by-byte CPU-vs-GPU parity check "
                    "(saves one H2D round-trip on shutdown)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir = args.out
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / "bench" / "reports" / ts / "gpu_scatter"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: extension must support compact assembler.
    lib = _get_lib()
    if not hasattr(lib, "rx_ring_assemble_compact_block"):
        _LOG.error(
            "C extension lacks rx_ring_assemble_compact_block; "
            "rebuild dsart/transport/recv_ring.c"
        )
        return 2

    # Need enough ring depth to step through n_iters * t_det without
    # wrapping. The current C scatter wraps t_seq automatically so this
    # is just for sanity.
    if args.t_buf < args.t_det * 4:
        _LOG.warning(
            "t_buf=%d is small relative to t_det=%d * 4; results "
            "may be affected by ring wraparound", args.t_buf, args.t_det,
        )

    dims = RxRingDims(
        n_corr=args.n_corr,
        n_coarse_dm=args.n_coarse_dm,
        t_buf_samples=args.t_buf,
        n_filled_per_corr=args.n_filled,
        bytes_per_cell=2,
    )

    name = f"/m741_bench_{uuid.uuid4().hex[:12]}"
    try:
        RxRing.unlink_name(name)
    except Exception:
        pass

    ring = RxRing.open_or_create(name, dims)

    try:
        # We only need to fill t_det worth of rows -- the benchmark reads
        # specnum_start=0 every iteration (data-novelty is irrelevant for
        # the perf measurement; cf. assemble_dense_block is bandwidth-bound).
        n_fill_rows = args.t_det
        _LOG.info(
            "Filling ring [0, %d) (%dC x %dDM x %dT x %d cells * 2 = %.1f MiB)...",
            n_fill_rows, dims.n_corr, dims.n_coarse_dm, n_fill_rows,
            dims.n_filled_per_corr,
            dims.n_corr * dims.n_coarse_dm * n_fill_rows
            * dims.n_filled_per_corr * 2 / 2**20,
        )
        t_fill = time.perf_counter_ns()
        _fill_ring(
            ring,
            n_corr=args.n_corr,
            n_coarse_dm=args.n_coarse_dm,
            t_fill=n_fill_rows,
            n_filled=args.n_filled,
            seed=args.seed,
        )
        _LOG.info("ring fill done in %.1fs",
                  (time.perf_counter_ns() - t_fill) / 1e9)

        lut = _make_lut(args.n_corr, args.n_filled, args.n_grid)
        nfp = np.full((args.n_corr,), args.n_filled, dtype=np.int32)

        device = torch.device(args.device)
        torch.cuda.set_device(device)

        _LOG.info("Running CPU-dense baseline (n=%d, warmup=%d)...",
                  args.n_iters, args.n_warmup)
        cpu_records = _bench_cpu_dense(
            ring=ring,
            n_corr=args.n_corr,
            n_coarse_dm=args.n_coarse_dm,
            t_det=args.t_det,
            t_stream=args.t_stream,
            n_grid=args.n_grid,
            n_filled=args.n_filled,
            owned_dm=args.owned_dm,
            device=device,
            n_iters=args.n_iters,
            n_warmup=args.n_warmup,
            lut=lut,
            nfp=nfp,
        )

        _LOG.info("Running GPU-scatter (n=%d, warmup=%d)...",
                  args.n_iters, args.n_warmup)
        gpu_records, last_gpu_dense = _bench_gpu_scatter(
            ring=ring,
            n_corr=args.n_corr,
            n_coarse_dm=args.n_coarse_dm,
            t_det=args.t_det,
            t_stream=args.t_stream,
            n_grid=args.n_grid,
            n_filled=args.n_filled,
            owned_dm=args.owned_dm,
            device=device,
            n_iters=args.n_iters,
            n_warmup=args.n_warmup,
            lut=lut,
            nfp=nfp,
        )

        # Correctness check on the last cube generated by both paths.
        # We can't compare last_gpu_dense to the CPU dense from the
        # last iter directly (we didn't keep it), so do one fresh comparison.
        if not args.skip_correctness:
            _LOG.info("Correctness check (final cube byte-identity)...")
            specnum_chk = 0  # Reset to a known specnum
            (dense_c_chk, *_rest) = ring.assemble_dense_block(
                specnum_start=specnum_chk,
                t_det=args.t_det,
                n_grid=args.n_grid,
                owned_dm=args.owned_dm,
                n_filled_per_corr=nfp,
                linear_lut_strided=lut,
                out_t_stride=args.t_stream,
                compute_half=0,
            )
            (cells_chk, *_rest2) = ring.assemble_compact_block(
                specnum_start=specnum_chk,
                t_det=args.t_det,
                owned_dm=args.owned_dm,
                n_filled_per_corr=nfp,
                n_filled_max=args.n_filled,
                sidecar_t_stride=args.t_stream,
                compute_half=0,
            )
            from dsart.transport.gpu_scatter import (
                scatter_compact_to_dense,
                zero_dense_rows,
            )
            dense_shape = (args.n_corr, args.t_stream, 2,
                           args.n_grid, args.n_grid)
            dense_gpu_chk = torch.zeros(dense_shape, dtype=torch.int8,
                                        device=device)
            cells_gpu_chk = torch.from_numpy(
                np.ascontiguousarray(cells_chk)
            ).to(device, non_blocking=False)
            lut_gpu = torch.from_numpy(
                np.ascontiguousarray(lut)
            ).to(device, non_blocking=False)
            nfp_gpu = torch.from_numpy(
                np.ascontiguousarray(nfp)
            ).to(device, non_blocking=False)
            zero_dense_rows(dense=dense_gpu_chk, t_det=args.t_det)
            scatter_compact_to_dense(
                cells_packed=cells_gpu_chk,
                lut=lut_gpu,
                n_filled_per_corr=nfp_gpu,
                dense=dense_gpu_chk,
                t_det=args.t_det,
                n_grid=args.n_grid,
                n_filled_max=args.n_filled,
            )
            torch.cuda.synchronize()
            dense_gpu_h = dense_gpu_chk.cpu().numpy()
            n_mismatch = int(
                np.sum(
                    dense_gpu_h[:, :args.t_det]
                    != dense_c_chk[:, :args.t_det]
                )
            )
            if n_mismatch != 0:
                _LOG.error(
                    "CORRECTNESS FAIL: %d byte mismatches between "
                    "GPU-scatter and CPU-dense planes", n_mismatch,
                )
                return 3
            _LOG.info("CORRECTNESS PASS: %d bytes match",
                      int(np.prod(dense_gpu_h[:, :args.t_det].shape)))

        # Summaries.
        print()
        _print_summary("CPU-dense (M7.4 baseline)", cpu_records,
                       keys=["cpu_scatter_ns", "cpu_copy_ns",
                             "cpu_total_ns", "h2d_ms", "wall_total_ns"])
        print()
        _print_summary("GPU-scatter (M7.4.1)", gpu_records,
                       keys=["cpu_compact_ns", "cpu_copy_ns",
                             "cpu_total_ns", "h2d_ms", "scatter_ms",
                             "gpu_total_ms", "wall_total_ns"])

        # Headline: end-to-end (assemble + copy + H2D + scatter) wall.
        cpu_wall = np.array([r["wall_total_ns"] for r in cpu_records]) / 1e6
        gpu_wall = np.array([r["wall_total_ns"] for r in gpu_records]) / 1e6
        cpu_cubes_per_s = 1000.0 / float(np.median(cpu_wall))
        gpu_cubes_per_s = 1000.0 / float(np.median(gpu_wall))
        speedup = float(np.median(cpu_wall)) / float(np.median(gpu_wall))
        print(
            f"\nHEADLINE: CPU-dense median = {np.median(cpu_wall):.2f} ms "
            f"({cpu_cubes_per_s:.2f} build/s); "
            f"GPU-scatter median = {np.median(gpu_wall):.2f} ms "
            f"({gpu_cubes_per_s:.2f} build/s); "
            f"speedup = {speedup:.2f}x"
        )

        rt_budget_ms = 134.218
        cpu_headroom = rt_budget_ms - float(np.percentile(cpu_wall, 99))
        gpu_headroom = rt_budget_ms - float(np.percentile(gpu_wall, 99))
        print(
            f"RT budget = {rt_budget_ms:.2f} ms (7.45 cubes/s). "
            f"CPU-dense p99 headroom = {cpu_headroom:+.2f} ms; "
            f"GPU-scatter p99 headroom = {gpu_headroom:+.2f} ms."
        )

        # Persist results.
        ndjson_path = out_dir / "timings.ndjson"
        with ndjson_path.open("w") as fh:
            for rec in cpu_records:
                fh.write(json.dumps({"path": "cpu_dense", **rec}) + "\n")
            for rec in gpu_records:
                fh.write(json.dumps({"path": "gpu_scatter", **rec}) + "\n")

        summary = {
            "config": {
                "n_iters": args.n_iters,
                "n_warmup": args.n_warmup,
                "n_corr": args.n_corr,
                "n_coarse_dm": args.n_coarse_dm,
                "t_buf": args.t_buf,
                "n_filled": args.n_filled,
                "t_det": args.t_det,
                "t_stream": args.t_stream,
                "n_grid": args.n_grid,
                "owned_dm": args.owned_dm,
                "device": str(device),
                "host": os.uname().nodename,
            },
            "cpu_dense": _summarise(cpu_records, [
                "cpu_scatter_ns", "cpu_copy_ns", "cpu_total_ns",
                "h2d_ms", "wall_total_ns",
            ]),
            "gpu_scatter": _summarise(gpu_records, [
                "cpu_compact_ns", "cpu_copy_ns", "cpu_total_ns",
                "h2d_ms", "scatter_ms", "gpu_total_ms", "wall_total_ns",
            ]),
            "headline": {
                "cpu_median_ms": float(np.median(cpu_wall)),
                "gpu_median_ms": float(np.median(gpu_wall)),
                "cpu_cubes_per_s": cpu_cubes_per_s,
                "gpu_cubes_per_s": gpu_cubes_per_s,
                "speedup_median": speedup,
                "rt_budget_ms": rt_budget_ms,
                "cpu_p99_headroom_ms": cpu_headroom,
                "gpu_p99_headroom_ms": gpu_headroom,
                "rt_gate_cpu": bool(cpu_headroom > 0),
                "rt_gate_gpu": bool(gpu_headroom > 0),
            },
        }
        with (out_dir / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
        _LOG.info("wrote %s + %s",
                  str(ndjson_path), str(out_dir / "summary.json"))
        return 0
    finally:
        ring.close()
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
