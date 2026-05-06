#!/usr/bin/env python3
"""bench/imager_only_gpu.py — M5 Chunk 6c follow-up: GPU imager-only
throughput probe (cuFFT-cfp16) at production geometry.

The chunk-6a numpy imager (``image/imager.py``) was a placeholder per
its own docstring (lines 18-24): pure-numpy CPU path with a
``complex64`` intermediate uv-grid. That's a 4× memory inflation over
the cint8 wire payload (per-block dequant scale/offset; D2/M1
``SparseCOOPayload``) and a 2× inflation over the production target
``complex32``. The chunk-6b production hardening pass was earmarked
to swap it for cuFFT-cfp16 on GPU; this bench is the prototype that
measures whether that path actually lands the imager throughput
budget.

Per-cube data flow (all on cuda):

  per-chgroup int8 [T_stream, 2, N_grid, N_grid] (cint8: re/im split)
       ↓ scale/offset dequant + cast → complex32 on GPU
  per-chgroup cfp16 [T_stream, N_grid, N_grid]
       ↓ for each f in [0, N_fdm):
       ↓     combine 16 chgroups via index-shifted sum (precomputed
       ↓        time_shift_per_chgroup)
       ↓     uv_slab cfp16 [T_det, N_grid, N_grid]
       ↓ ifft2 (cuFFT-cfp16, single-side identity §3.6.11)
       ↓ fftshift (DC → centre)
       ↓ Re(...) cast to fp16 + multiply by edge mask
  output_cube fp16 [T_det, N_fdm, N_grid, N_grid]

Memory budget (production T=512, N_fdm=32, N_grid=256, on 11 GiB 2080 Ti):
    16 per-chgroup int8 streams (T_stream=T_det+128, cint8 re/im split):
       16 × 640 × 2 × 256² × 1 B = 1.25 GiB
    1 active uv slab (cfp16):              512 × 256² × 4 B = 128 MiB
    1 active image slab (fp16, real):      512 × 256² × 2 B =  64 MiB
    output cube (fp16):           32 × 512 × 256² × 2 B = 2.0 GiB
    edge mask (fp32):                            256² × 4 B = 256 KiB
    -----------------------------------------------------------------
    peak ........................................... ~3.5 GiB

Fits comfortably. If even bigger geometries are needed, the per-fdm
loop can additionally be chunked along T_det (process T_chunk samples
at a time with a cuFFT plan reuse loop).

CLI:

  python -m bench.imager_only_gpu                                \\
      [--n-cubes 30] [--t-det 512] [--n-fdm 32] [--n-grid 256]   \\
      [--cube-dtype cfp16|cfp32]                                 \\
      [--seed 0]                                                 \\
      --out bench/reports/<UTC>/imager_gpu/M5/

Outputs:
  * stage_timings.ndjson — per-cube ``{cube_id, scatter_ns,
    combine_ns, ifft2_ns, mask_ns, total_ns}``.
  * summary.json        — config + percentile rollup + cubes/s.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

import warnings  # noqa: E402

# torch warns "ComplexHalf support is experimental..." every time we
# touch a complex32 tensor. The behaviour is stable enough for cuFFT
# (verified end-to-end above this commit) and we acknowledge the risk
# in the bench docstring + report.
warnings.filterwarnings("ignore", message=".*ComplexHalf.*experimental.*")

from dsart.image.imager import compute_edge_mask  # noqa: E402

_LOG = logging.getLogger("bench.imager_only_gpu")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Production geometry (plan §8 line 2317): T_det=512 cube samples,
# N_fdm=32 fine-DM trials per GPU half (D2/§3.6.13), N_grid=256 image
# pixels. The bench defaults to this.
DEFAULT_N_CUBES: int = 30
DEFAULT_T_DET: int = 512
DEFAULT_N_FDM: int = 32
DEFAULT_N_GRID: int = 256
DEFAULT_T_STREAM_PAD: int = 128  # max time-shift across chgroups (rough cap)
N_CHGROUP: int = 16

# Cube dtype is set on the OUTPUT cube; the FFT works at the matching
# complex dtype (cfp16 ↔ fp16, cfp32 ↔ fp32).
DEFAULT_CUBE_DTYPE: str = "cfp16"


@dataclass(frozen=True, slots=True)
class StageRecord:
    cube_id: int
    scatter_ns: int      # cint8→cfp32 dequant + scatter
    combine_ns: int      # 16-chgroup index-shifted sum (per cube total)
    ifft2_ns: int        # ifft2 (per cube total, all N_fdm slabs)
    mask_ns: int         # fftshift + edge-mask multiplication
    total_ns: int

    def to_json(self) -> Dict[str, int]:
        return {
            "cube_id": self.cube_id,
            "scatter_ns": self.scatter_ns,
            "combine_ns": self.combine_ns,
            "ifft2_ns": self.ifft2_ns,
            "mask_ns": self.mask_ns,
            "total_ns": self.total_ns,
        }


# ---------------------------------------------------------------------------
# Synthetic cint8 input generation (mock M3 wire payload)
# ---------------------------------------------------------------------------


def _build_synthetic_streams(
    *,
    n_chgroup: int,
    t_stream: int,
    n_grid: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Return ``[N_chgroup, T_stream, 2, N_grid, N_grid] int8`` simulated
    per-chgroup gridded uv-streams (re, im split along axis=2). This is
    a test-only generator; the real M3 sparse payload is denser at the
    centre and zero in the corners (per the gridder pattern), but for
    the imager-only throughput probe a uniform random fill is the
    pessimistic case — every cell does work.

    The cint8 representation is the per-cell operational form per
    M5_PLAN_FIXES.md D2 / common.contracts.SparseCOOPayload (16 bits
    per complex, 8 real + 8 imag); per-block scale/offset are folded
    into the dequant cast on GPU (single per-block scalar pair).
    """
    raw = rng.integers(
        low=-127, high=127, size=(n_chgroup, t_stream, 2, n_grid, n_grid),
        dtype=np.int8,
    )
    return torch.from_numpy(raw)


def _build_time_shifts(
    *, n_fdm: int, n_chgroup: int, t_stream_pad: int, rng: np.random.Generator,
) -> torch.Tensor:
    """Per-fdm × per-chgroup integer time shift in stream samples.

    Plan §3.6.3 fixes ``shift[chgroup-15] == 0`` (chgroup-15 is the
    reference); other chgroups have a positive shift up to a few tens
    of samples (the 16-chgroup band-shift across DSA-110's 100-MHz BW
    at high DMs). We use ``int32`` integers in
    ``[0, t_stream_pad)`` for the bench; chgroup-15 stays 0.
    """
    shifts = rng.integers(
        low=0, high=t_stream_pad, size=(n_fdm, n_chgroup), dtype=np.int32,
    )
    shifts[:, n_chgroup - 1] = 0
    return torch.from_numpy(shifts).contiguous()


# ---------------------------------------------------------------------------
# GPU imager core
# ---------------------------------------------------------------------------


@dataclass
class GpuImagerWorkspace:
    """Pre-allocated GPU buffers reused across cubes.

    Holds the persistent state: the edge mask + output cube buffer.
    Per-cube transient buffers (uv slab, image slab) are also held
    here and reused; one allocation per service-lifetime is cheaper
    than reallocating per cube, and `torch` reuses memory anyway via
    the caching allocator.
    """
    device: torch.device
    n_grid: int
    t_det: int
    n_fdm: int
    cube_dtype: torch.dtype  # output cube real dtype (fp16 or fp32)
    complex_dtype: torch.dtype  # uv-slab dtype (complex32 or complex64)
    edge_mask_real: torch.Tensor  # [N_grid, N_grid] real
    output_cube: torch.Tensor    # [T_det, N_fdm, N_grid, N_grid] real
    uv_slab: torch.Tensor        # [T_det, N_grid, N_grid] complex
    img_slab_real: torch.Tensor  # [T_det, N_grid, N_grid] real

    @classmethod
    def build(
        cls,
        *,
        device: torch.device,
        n_grid: int,
        t_det: int,
        n_fdm: int,
        cube_dtype: torch.dtype,
        complex_dtype: torch.dtype,
    ) -> "GpuImagerWorkspace":
        edge = compute_edge_mask(n_grid=n_grid, kernel_support=5)
        edge_t = torch.from_numpy(edge).to(device=device, dtype=cube_dtype)
        out = torch.empty(
            (t_det, n_fdm, n_grid, n_grid), dtype=cube_dtype, device=device,
        )
        uv = torch.empty(
            (t_det, n_grid, n_grid), dtype=complex_dtype, device=device,
        )
        img = torch.empty(
            (t_det, n_grid, n_grid), dtype=cube_dtype, device=device,
        )
        return cls(
            device=device, n_grid=n_grid, t_det=t_det, n_fdm=n_fdm,
            cube_dtype=cube_dtype, complex_dtype=complex_dtype,
            edge_mask_real=edge_t, output_cube=out,
            uv_slab=uv, img_slab_real=img,
        )


def _dequant_cint8_to_complex_gpu(
    streams_cint8: torch.Tensor,  # [N_chgroup, T_stream, 2, N_grid, N_grid] int8
    *,
    complex_dtype: torch.dtype,
    scale: float = 1.0,
    offset: float = 0.0,
) -> torch.Tensor:
    """Dequantise the per-chgroup cint8 streams → complex tensor on GPU.

    Output: ``[N_chgroup, T_stream, N_grid, N_grid]`` ``complex_dtype``.

    Done chgroup-by-chgroup to bound peak GPU memory: the intermediate
    fp32 → complex64 cast for one chgroup is ``T_stream × N_grid² × 8``
    bytes (≈ 320 MiB at production T_stream=640, N_grid=256), vs ≈ 5
    GiB if all 16 chgroups were converted at once.

    Per ``SparseCOOPayload``, the scale/offset are per-block scalars
    computed at the corr side from the filled cells. For the bench we
    pass the constants in (default scale=1, offset=0); production
    M3-emitted payloads carry them.

    ``torch.complex(re, im)`` only supports fp32/fp64, and
    ``torch.view_as_complex`` only supports fp32→cf64 / fp64→cf128.
    For ``complex_dtype = torch.complex32`` we therefore go via fp32 →
    cf64 → cf32.
    """
    n_chgroup, t_stream, _, n_grid, _ = streams_cint8.shape
    out = torch.empty(
        (n_chgroup, t_stream, n_grid, n_grid),
        dtype=complex_dtype, device=streams_cint8.device,
    )
    for g in range(n_chgroup):
        # [T_stream, 2, N_grid, N_grid] cint8 → fp32
        chg = streams_cint8[g].to(dtype=torch.float32)
        if scale != 1.0 or offset != 0.0:
            chg = chg * float(scale) + float(offset)
        # permute last to [T_stream, N_grid, N_grid, 2] for view_as_complex.
        chg = chg.permute(0, 2, 3, 1).contiguous()
        chg_complex64 = torch.view_as_complex(chg)  # [T_stream, N_grid, N_grid] cf64
        out[g] = chg_complex64.to(dtype=complex_dtype)
        del chg, chg_complex64
    return out


def _process_one_cube(
    *,
    streams_complex: torch.Tensor,  # [N_chgroup, T_stream, N_grid, N_grid] complex
    time_shifts: torch.Tensor,      # [N_fdm, N_chgroup] int32
    workspace: GpuImagerWorkspace,
) -> Dict[str, int]:
    """Run the full GPU imager for one cube. Writes into
    ``workspace.output_cube``. Returns per-stage ns dict."""
    n_chgroup = streams_complex.shape[0]
    t_stream = streams_complex.shape[1]
    t_det = workspace.t_det
    n_fdm = workspace.n_fdm
    edge = workspace.edge_mask_real
    output = workspace.output_cube
    uv = workspace.uv_slab
    img = workspace.img_slab_real

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_combine_total = 0
    t_ifft_total = 0
    t_mask_total = 0
    t_loop_start = time.perf_counter_ns()
    for f in range(n_fdm):
        # ---- combine: 16-chgroup index-shifted sum into uv slab. ----
        t0 = time.perf_counter_ns()
        uv.zero_()
        # Each chgroup g's contribution is stream[g, t - shift[f, g]] for
        # t in [0, t_det). We crop the source to the valid window
        # [shift, shift + t_det) and add into uv[:t_det].
        shifts_f = time_shifts[f]
        for g in range(n_chgroup):
            s = int(shifts_f[g].item())
            # t_stream is sized so that shift+t_det <= t_stream by
            # construction in the bench. Production has the same guarantee
            # via the receive-ring pre-roll buffer.
            if s + t_det > t_stream:
                continue  # zero-fill (production: cube-validity gate)
            uv += streams_complex[g, s : s + t_det]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        t_combine_total += (t1 - t0)

        # ---- ifft2 (cuFFT-cfp16 if complex32) + fftshift + Re(...) ----
        t2 = time.perf_counter_ns()
        img_complex = torch.fft.ifft2(uv)
        img_complex = torch.fft.fftshift(img_complex, dim=(-2, -1))
        img_real = img_complex.real.to(workspace.cube_dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter_ns()
        t_ifft_total += (t3 - t2)

        # ---- mask + write into output cube slot ----
        t4 = time.perf_counter_ns()
        torch.mul(img_real, edge, out=img)
        output[:, f, :, :] = img
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t5 = time.perf_counter_ns()
        t_mask_total += (t5 - t4)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total = time.perf_counter_ns() - t_loop_start
    return {
        "combine_ns": t_combine_total,
        "ifft2_ns": t_ifft_total,
        "mask_ns": t_mask_total,
        "total_ns": t_total,
    }


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


def percentiles(values_ns: Sequence[int]) -> Dict[str, float]:
    if not values_ns:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values_ns, dtype=np.int64)
    arr_ms = arr.astype(np.float64) / 1.0e6
    return {
        "p50": float(np.percentile(arr_ms, 50)),
        "p90": float(np.percentile(arr_ms, 90)),
        "p99": float(np.percentile(arr_ms, 99)),
        "mean": float(arr_ms.mean()),
        "max": float(arr_ms.max()),
    }


def _bench_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_log_path = out_dir / "bench.log"
    fh = logging.FileHandler(bench_log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _LOG.setLevel(logging.INFO)
    _LOG.addHandler(fh)
    _LOG.addHandler(logging.StreamHandler(sys.stdout))

    if not torch.cuda.is_available():
        _LOG.error("cuda is required for this bench; aborting")
        return 2

    n_cubes = int(args.n_cubes)
    t_det = int(args.t_det)
    n_fdm = int(args.n_fdm)
    n_grid = int(args.n_grid)
    cube_dtype_arg = str(args.cube_dtype)
    if cube_dtype_arg not in ("cfp16", "cfp32"):
        raise ValueError(
            f"--cube-dtype={cube_dtype_arg!r}; expected cfp16 or cfp32"
        )
    if cube_dtype_arg == "cfp16":
        cube_dtype = torch.float16
        complex_dtype = torch.complex32
    else:
        cube_dtype = torch.float32
        complex_dtype = torch.complex64

    t_stream_pad = int(args.t_stream_pad)
    t_stream = t_det + t_stream_pad

    device = torch.device("cuda")
    rng = np.random.default_rng(int(args.seed))

    _LOG.info(
        "config: n_cubes=%d T_det=%d N_fdm=%d N_grid=%d cube_dtype=%s "
        "complex_dtype=%s T_stream=%d (pad=%d)",
        n_cubes, t_det, n_fdm, n_grid, cube_dtype_arg, complex_dtype,
        t_stream, t_stream_pad,
    )

    # Pre-allocate the GPU workspace (output cube + transient buffers).
    workspace = GpuImagerWorkspace.build(
        device=device, n_grid=n_grid, t_det=t_det, n_fdm=n_fdm,
        cube_dtype=cube_dtype, complex_dtype=complex_dtype,
    )

    # Per-cube the "input" (per-chgroup int8 streams + time-shift table)
    # is regenerated. In production this comes from M3 RX-ring slots.
    records: List[StageRecord] = []
    bench_start_ns = time.perf_counter_ns()
    for cube_id in range(n_cubes):
        # ---- generate cint8 streams + dequant on GPU (scatter step) ----
        streams_cint8_cpu = _build_synthetic_streams(
            n_chgroup=N_CHGROUP, t_stream=t_stream, n_grid=n_grid, rng=rng,
        )
        time_shifts = _build_time_shifts(
            n_fdm=n_fdm, n_chgroup=N_CHGROUP,
            t_stream_pad=t_stream_pad, rng=rng,
        ).to(device=device)

        torch.cuda.synchronize()
        t_scatter_start = time.perf_counter_ns()
        streams_cint8_gpu = streams_cint8_cpu.to(device=device, non_blocking=False)
        streams_complex = _dequant_cint8_to_complex_gpu(
            streams_cint8_gpu, complex_dtype=complex_dtype,
        )
        torch.cuda.synchronize()
        scatter_ns = time.perf_counter_ns() - t_scatter_start

        # ---- run the per-fdm imager loop ----
        timings = _process_one_cube(
            streams_complex=streams_complex,
            time_shifts=time_shifts,
            workspace=workspace,
        )
        del streams_complex, streams_cint8_gpu

        rec = StageRecord(
            cube_id=cube_id,
            scatter_ns=int(scatter_ns),
            combine_ns=int(timings["combine_ns"]),
            ifft2_ns=int(timings["ifft2_ns"]),
            mask_ns=int(timings["mask_ns"]),
            total_ns=int(scatter_ns + timings["total_ns"]),
        )
        records.append(rec)
        if (cube_id + 1) % max(1, n_cubes // 10) == 0:
            _LOG.info(
                "cube=%d/%d total=%.2fms (scatter=%.2f combine=%.2f "
                "ifft2=%.2f mask=%.2f)",
                cube_id + 1, n_cubes,
                rec.total_ns / 1.0e6,
                rec.scatter_ns / 1.0e6,
                rec.combine_ns / 1.0e6,
                rec.ifft2_ns / 1.0e6,
                rec.mask_ns / 1.0e6,
            )

    bench_wall_s = (time.perf_counter_ns() - bench_start_ns) / 1.0e9
    achieved_cubes_per_s = len(records) / bench_wall_s if bench_wall_s > 0 else 0.0

    # ---- write outputs ----
    ndjson_path = out_dir / "stage_timings.ndjson"
    with ndjson_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json()) + "\n")

    summary = {
        "schema_version": 1,
        "bench": "imager_only_gpu",
        "milestone": "M5",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_cubes": n_cubes,
            "t_det": t_det,
            "n_fdm": n_fdm,
            "n_grid": n_grid,
            "n_chgroup": N_CHGROUP,
            "t_stream_pad": t_stream_pad,
            "cube_dtype": cube_dtype_arg,
            "complex_dtype": str(complex_dtype).rsplit(".", 1)[-1],
            "device": "cuda",
            "seed": int(args.seed),
        },
        "wall_clock_s": bench_wall_s,
        "achieved_cubes_per_s": achieved_cubes_per_s,
        "n_cubes_processed": len(records),
        "percentiles_ms": {
            "scatter": percentiles([r.scatter_ns for r in records]),
            "combine": percentiles([r.combine_ns for r in records]),
            "ifft2":    percentiles([r.ifft2_ns for r in records]),
            "mask":     percentiles([r.mask_ns for r in records]),
            "total":    percentiles([r.total_ns for r in records]),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _LOG.info("wrote %s", summary_path)
    pct = summary["percentiles_ms"]["total"]
    _LOG.info(
        "summary: %.2f cubes/s · total p50=%.2fms p99=%.2fms",
        achieved_cubes_per_s, pct["p50"], pct["p99"],
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-cubes", type=int, default=DEFAULT_N_CUBES)
    parser.add_argument("--t-det", type=int, default=DEFAULT_T_DET)
    parser.add_argument("--n-fdm", type=int, default=DEFAULT_N_FDM)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument(
        "--cube-dtype", type=str, default=DEFAULT_CUBE_DTYPE,
        choices=("cfp16", "cfp32"),
        help="Output cube real dtype (cfp16 = fp16 image, cfp32 = fp32 image). "
             "complex_dtype follows: cfp16 → torch.complex32 / cuFFT-cfp16; "
             "cfp32 → torch.complex64 / cuFFT-cfp32.",
    )
    parser.add_argument(
        "--t-stream-pad", type=int, default=DEFAULT_T_STREAM_PAD,
        help="Per-chgroup time-shift slack, in stream samples. "
             "Production T_stream = T_det + max_shift.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "imager_gpu" / "M5"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return _bench_main(args)


if __name__ == "__main__":
    sys.exit(main())
