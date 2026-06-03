"""bench/stage2_fifo_ab.py — A/B benchmark for the stage-2 FIFO swap.

Measures the corr-side per-cube cost (push + emit), warmup behaviour, and
memory footprint of the legacy uniform-depth :class:`Stage2FIFO` vs the
per-coarse-DM :class:`Stage2InterChgroupShiftFifo` ("Option A") that
finishes the stage-2 inter-chgroup time alignment on the corr side so
the search-side ``compute_time_shift_search`` can revert to
``include_coarse_offset=False``.

The bench plays N synthetic dedispersed cubes through each FIFO
implementation at the production op-point (M7.4 DM plan, t_int_corr_us
= 262.144 µs, T_dedisp configurable) and reports per-chgroup-and-coarse-
DM cost. Runs on CPU by default; pass ``--device cuda:0`` for GPU.

CLI::

    python -m bench.stage2_fifo_ab \\
        --dm-plan-path /home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz \\
        --t-dedisp 500 --n-filled 5000 --n-cubes 200 --warm-cubes 50 \\
        --device cuda:0 --report-json /tmp/stage2_ab.json

The output is a JSON line per (chgroup, mode) with median / p99 push-emit
ms, plus a printed table summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.coarse_dm.stage2_chgroup_alignment import (  # noqa: E402
    Stage2InterChgroupShiftFifo,
)
from dsart.coarse_dm.stage2_fifo import Stage2FIFO  # noqa: E402
from dsart.coarse_dm.stage2_shifts import (  # noqa: E402
    T_INT_CORR_US_DEFAULT,
    compute_stage2_shifts,
)
from dsart.common.constants import N_CHGROUP  # noqa: E402


def _make_synth_cube(
    n_dm: int, t_dedisp: int, n_filled: int, *, device: torch.device,
    dtype: torch.dtype, seed: int,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (n_dm, t_dedisp, n_filled), generator=g, dtype=torch.float32,
        device=device,
    ).to(dtype)


def _bench_mode(
    *,
    mode: str,
    chgroup: int,
    coarse_dm: np.ndarray,
    t_dedisp: int,
    n_filled: int,
    n_cubes: int,
    warm_cubes: int,
    device: torch.device,
    dtype: torch.dtype,
    t_int_corr_us: float,
) -> Dict[str, Any]:
    if mode == "uniform":
        # Match the M7.4 production: depth = COARSE_DM_FIFO_DEPTH_DEFAULT.
        # The legacy uniform FIFO carries one cube per "tick"; depth=8 is
        # the production default. Using a deeper FIFO would only affect
        # warmup, not per-push cost.
        from dsart.common.constants import COARSE_DM_FIFO_DEPTH_DEFAULT
        fifo = Stage2FIFO(depth=COARSE_DM_FIFO_DEPTH_DEFAULT)
        n_dm = coarse_dm.size
        push_fn = lambda c, bn: fifo.push_for_protocol(c, block_n=bn)
        emit_warmed = lambda: True  # uniform FIFO emits from push K onward
    elif mode == "per_coarse_dm":
        fifo = Stage2InterChgroupShiftFifo(
            chgroup=chgroup,
            coarse_dm_pc_cm3=coarse_dm,
            t_dedisp=t_dedisp,
            t_int_corr_us=t_int_corr_us,
        )
        n_dm = coarse_dm.size
        push_fn = lambda c, bn: fifo.push(c, block_n=bn)
        emit_warmed = lambda: fifo.warmed_up()
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Stream cubes rather than pre-allocating (avoids OOM at production
    # sizes where each cube is ~150 MiB).
    def _cube_for(k: int) -> torch.Tensor:
        return _make_synth_cube(
            n_dm, t_dedisp, n_filled, device=device, dtype=dtype, seed=k,
        )

    # Warmup pushes (not measured)
    n_warmup_emit = 0
    warmup_done_block = None
    for bn in range(warm_cubes):
        out = push_fn(_cube_for(bn), bn)
        if out:
            n_warmup_emit += 1
            if warmup_done_block is None and emit_warmed():
                warmup_done_block = bn

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    push_ms: List[float] = []
    emit_count = 0
    for bn in range(warm_cubes, warm_cubes + n_cubes):
        cube = _cube_for(bn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = push_fn(cube, bn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        push_ms.append((time.perf_counter() - t0) * 1000.0)
        emit_count += len(out)

    element_bytes = torch.tensor([], dtype=dtype).element_size()
    if mode == "per_coarse_dm":
        slice_bytes = t_dedisp * n_filled * element_bytes
        ring_lens = [len(r) for r in fifo._rings]  # type: ignore[attr-defined]
        mem_mib = sum(ring_lens) * slice_bytes / (1 << 20)
    else:
        cube_bytes = coarse_dm.size * t_dedisp * n_filled * element_bytes
        from dsart.common.constants import COARSE_DM_FIFO_DEPTH_DEFAULT
        mem_mib = COARSE_DM_FIFO_DEPTH_DEFAULT * cube_bytes / (1 << 20)

    return {
        "mode": mode,
        "chgroup": int(chgroup),
        "n_coarse_dm": int(coarse_dm.size),
        "t_dedisp": int(t_dedisp),
        "n_filled": int(n_filled),
        "n_cubes_measured": int(n_cubes),
        "warmup_cubes": int(warm_cubes),
        "push_ms_median": float(statistics.median(push_ms)),
        "push_ms_p99": float(np.percentile(push_ms, 99)),
        "push_ms_max": float(max(push_ms)),
        "push_ms_mean": float(statistics.mean(push_ms)),
        "emit_count": int(emit_count),
        "warmup_emits": int(n_warmup_emit),
        "warmup_done_at_block": (
            int(warmup_done_block) if warmup_done_block is not None else -1
        ),
        "mem_mib_steady": float(mem_mib),
        "device": str(device),
        "dtype": str(dtype),
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dm-plan-path", required=True, type=Path)
    p.add_argument("--chgroups", type=str, default="0,7,14,15",
                   help="Comma-sep chgroups to bench (default: 0,7,14,15 — "
                        "worst, mid, near-bottom, identity)")
    p.add_argument("--t-dedisp", type=int, default=500,
                   help="T_dedisp per cube push (production ~500)")
    p.add_argument("--n-filled", type=int, default=5000,
                   help="N_filled per cube (production ~5000)")
    p.add_argument("--n-cubes", type=int, default=200)
    p.add_argument("--warm-cubes", type=int, default=50,
                   help="Untimed warmup pushes before measurement")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="complex64",
                   choices=("complex64", "float32"))
    p.add_argument("--t-int-corr-us", type=float,
                   default=float(T_INT_CORR_US_DEFAULT))
    p.add_argument("--report-json", type=Path, default=None)
    args = p.parse_args(argv)

    dtype_map = {"complex64": torch.complex64, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    plan = np.load(args.dm_plan_path)
    coarse = np.asarray(plan["coarse_dm"], dtype=np.float64)

    chgroups = [int(x) for x in args.chgroups.split(",")]
    for g in chgroups:
        if not (0 <= g < N_CHGROUP):
            raise ValueError(f"chgroup {g} out of range")

    results: List[Dict[str, Any]] = []
    for g in chgroups:
        for mode in ("uniform", "per_coarse_dm"):
            tbl = compute_stage2_shifts(
                chgroup=g, coarse_dm_pc_cm3=coarse,
                t_int_corr_us=args.t_int_corr_us,
            )
            print(
                f"\n[chgroup={g} mode={mode}]  shifts={tbl.shifts_samples.tolist()}"
            )
            r = _bench_mode(
                mode=mode,
                chgroup=g,
                coarse_dm=coarse,
                t_dedisp=args.t_dedisp,
                n_filled=args.n_filled,
                n_cubes=args.n_cubes,
                warm_cubes=args.warm_cubes,
                device=device,
                dtype=dtype,
                t_int_corr_us=args.t_int_corr_us,
            )
            r["shifts_samples"] = tbl.shifts_samples.tolist()
            r["shift_max"] = int(tbl.shifts_samples.max(initial=0))
            results.append(r)

    print("\n" + "=" * 95)
    print(f"{'chgrp':>5} {'mode':>14} {'shift_max':>10} {'p50_ms':>8} "
          f"{'p99_ms':>8} {'max_ms':>8} {'emits':>6} {'mem_MiB':>8}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['chgroup']:>5} {r['mode']:>14} {r['shift_max']:>10} "
            f"{r['push_ms_median']:>8.3f} {r['push_ms_p99']:>8.3f} "
            f"{r['push_ms_max']:>8.3f} {r['emit_count']:>6d} "
            f"{r['mem_mib_steady']:>8.0f}"
        )
    print("=" * 95)

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
