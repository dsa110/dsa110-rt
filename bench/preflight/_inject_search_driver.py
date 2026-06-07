#!/usr/bin/env python3
"""bench/preflight/_inject_search_driver.py — corr→search bridge + real
search detector (Stage 2 of the correctness harness).

Takes the corr-fast coarse-DM products produced by
:mod:`bench.preflight._inject_corr_fixture` for ONE owner coarse-DM and
drives the **production search side** (combiner + GPU imager + Layer-1
σ-clip normalisation + ``DeterministicDetector``) to recover the
injected transient as detector ``Candidate``s, then emits:

  * ``candidates_c1.csv`` / ``.ndjson`` — C1 rows (via the production
    :func:`dsart.services.c1_emit.candidate_to_c1_row`).
  * a notebook-compatible NPZ cube
    (``tools/viz/cube_burst_explorer.ipynb`` schema).

Stage-2 (inter-chgroup) alignment approach
-------------------------------------------
The corr-fast single-block stage-1 output drops ``max_delay_bins``
tiles per block (``T_dedisp = n_fast_vis - max_delay_bins`` = 58 at
``t_int_fast_native=32``), so naively concatenating blocks leaves a
70-tile GAP between blocks on the time axis. The full inter-chgroup
dispersion delay at DM≈500 (~270 search samples) is FAR larger than 58,
so a burst dispersed across the 16 chgroups would land in those gaps
and be lost.

We avoid the gap entirely by running corr-fast with
``sliding_window=True`` (F34): each block then emits a FULL, contiguous
``n_fast_vis`` (=128) tile slice referenced to the chgroup TOP
(Convention A). Concatenating the resolved blocks gives a truly
gap-free per-chgroup true-time stream whose index equals the absolute
true-time tile (the cold-start block 0 occupies tiles ``[0, 128)`` and
is sliced off).

With a gap-free stream we do NOT need to hand-apply
``time_shift_corr_stage2``: instead we let the production search-side
combiner absorb the FULL ν_chgroup_TOP→ν_bot_proc delay by building the
``TimeShiftSearchTable`` with ``include_coarse_offset=True`` (the M7.4
"stage-2-absent escape hatch" — the documented, production-supported
code path for exactly this "corr did not apply stage-2" situation). The
shifts then telescope so the burst lands at a single cube-time across
all chgroups. We use M7.7 symmetric-shift padding
(``stream_origin_offset_samples = shifts.max()``) so the
``CubePipeline`` fused-L1 GPU fast path runs.

Execution / numerics notes (n01, 2x RTX 2080 Ti, 11 GB)
--------------------------------------------------------
* corr-fast is run **one chgroup per child process** (see
  :func:`_corr_streams_subprocess`): its GPU context (CUDA-graph /
  Triton int4-unpack state) cannot be torn down and rebuilt twice in a
  single process — the second ``build_context`` raises ``CUDA error:
  resource already mapped``. ``--reuse-corr`` caches each chgroup's
  owner-coarse-DM stream npz under ``<out>/corr_work``.
* Each chgroup has its OWN sparsity pattern (UV coverage differs);
  densification uses per-chgroup ``ix_row/ix_col`` (``cell_lambda`` is
  common so the dense grids stack pixel-for-pixel).
* The search runs in **two phases** so the fp32 imager workspace and
  the detector's fp32 boxcar-cumsum scratch never coexist: PHASE 1
  images + Layer-1-normalises every cube (imager resident) and stashes
  each L1-normalised fp16 cube on the host; PHASE 2 frees the imager and
  runs the detector. Imaging uses the complex64/fp32 numerical-audit
  path (``--no-audit-fp32`` to force production fp16/complex32) because
  a fluence~5e4 point source images to a flat-UV pattern whose coherent
  FFT sum overflows fp16; the L1-normalised cube is downcast to fp16 for
  the (memory-bounded) detector.
* All cubes share ONE noise-derived cf->cint8 scale (``fixed_scale``)
  so the frozen Layer-1 σ applies to every cube; the bright source
  clips at ±127 (still a huge-SNR detection) while the noise floor stays
  Layer-1-consistent (per-cube adaptive scale otherwise floods the
  noise cube with false positives).
* A zero-DM filter (subtract the per-pixel temporal mean) removes the
  static DC term that survives ``static_sky_disabled=True`` in corr-fast
  (its b128 time-boxcar otherwise overflows fp16 to inf at the phase
  centre). ``--no-zero-dm`` disables it.

Run on the GPU host (n01)::

    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
    conda activate dsa110-rt
    cd /home/ubuntu/proj/dsa110-rt
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:bench:. \\
        python bench/preflight/_inject_search_driver.py --owner-idx 1

Stage-3 entry point
-------------------
:func:`run_search_driver` is the importable callable a DM×width grid
runner loops over; see its docstring for the full signature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

# --- repo-root bootstrap so the file runs as a plain script too -----------
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os

os.environ.setdefault("DSART_TEST", "1")  # enable contract validation

from dsart.common.constants import (  # noqa: E402
    NU_BOT_PROC_GHZ,
    NU_CHGROUP_TOP_GHZ,
    NU_TOP_PROC_GHZ,
    N_CHGROUP,
    NATIVE_SAMPLE_US,
    SPECNUM_PERIOD_US,
)
from dsart.common.contracts import Candidate, CubeGeometry  # noqa: E402
from dsart.common.dispersion import delta_tau_us  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.fine_dm.combiner import (  # noqa: E402
    TimeShiftSearchTable,
    combine_chgroups,
    compute_time_shift_search,
)
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.c1_emit import candidate_to_c1_row  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import CubeRingSlot  # noqa: E402
from dsart.services.search_compute import _dm_grids_from_npz  # noqa: E402
from dsart.transport.quantize import (  # noqa: E402
    quantise_per_chgroup_into_cint8,
)

from bench._bank_mask import parse_bank_mask  # noqa: E402
from bench._corr_fast_replay import (  # noqa: E402
    dirty_image_from_dense_grid,
    sparse_to_dense_grid,
)
from bench.preflight._inject_corr_fixture import (  # noqa: E402
    DEFAULT_DM_PLAN_PATH,
    InjectionCell,
    run_corr_fast_injection,
)

__all__ = [
    "run_search_driver",
    "OwnerDmGrids",
    "owner_dm_grids",
]

#: Native samples per fada block (= 4096); 1 block = 128 fast-vis tiles
#: at t_int_fast_native=32.
NATIVE_SAMPLES_PER_BLOCK: int = 4096


# ---------------------------------------------------------------------------
# DM-plan owner slice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerDmGrids:
    """One owner coarse-DM's local fine-DM grid (search-side view)."""

    owner_idx: int
    coarse_dm_local: np.ndarray   # [1] float64 (owner coarse only)
    fine_dm_local: np.ndarray     # [K] float64
    fine_to_coarse_local: np.ndarray  # [K] int (all 0)
    coarse_dm_full: np.ndarray    # [N_coarse] float64

    @property
    def n_fdm(self) -> int:
        return int(self.fine_dm_local.shape[0])


def owner_dm_grids(dm_plan_path: str | Path, owner_idx: int,
                   *, n_coarse: int = 8) -> OwnerDmGrids:
    """Load the plan and restrict to the K fine-DM trials owned by
    ``owner_idx`` (mirrors ``search_node_throughput`` / production
    ``_select_dm_owner_half`` local convention)."""
    coarse_dm, fine_dm, fine_to_coarse = _dm_grids_from_npz(
        Path(dm_plan_path), n_coarse=n_coarse
    )
    mask = fine_to_coarse == owner_idx
    if not mask.any():
        raise ValueError(
            f"owner_idx={owner_idx} not in fine_to_coarse range "
            f"{int(fine_to_coarse.min())}..{int(fine_to_coarse.max())}"
        )
    fine_local = fine_dm[mask].astype(np.float64, copy=False)
    coarse_local = np.asarray(
        [coarse_dm[owner_idx]], dtype=np.float64
    )
    f2c_local = np.zeros(fine_local.shape[0], dtype=np.int32)
    return OwnerDmGrids(
        owner_idx=int(owner_idx),
        coarse_dm_local=coarse_local,
        fine_dm_local=fine_local,
        fine_to_coarse_local=f2c_local,
        coarse_dm_full=coarse_dm.astype(np.float64, copy=False),
    )


# ---------------------------------------------------------------------------
# Timing helpers (true-time tile bookkeeping; see module docstring)
# ---------------------------------------------------------------------------


def _chgroup_top_delay_tiles(dm_pc_cm3: float, t_int_search_us: float) -> np.ndarray:
    """Per-chgroup inter-band delay (true-time tiles) of the burst at
    each chgroup's TOP channel relative to NU_TOP_PROC (= chgroup-0 top).

    ``delay[g] = rint(Δτ(ν_chgroup_TOP[g], NU_TOP_PROC, dm) / t_int_search)``
    (non-negative; 0 for g=0).
    """
    top = np.asarray(NU_CHGROUP_TOP_GHZ, dtype=np.float64)
    out = np.zeros(N_CHGROUP, dtype=np.int64)
    for g in range(N_CHGROUP):
        d_us = delta_tau_us(float(top[g]), float(NU_TOP_PROC_GHZ), float(dm_pc_cm3))
        out[g] = int(np.rint(d_us / t_int_search_us))
    return out


# ---------------------------------------------------------------------------
# Bridge: corr-fast coarse-DM products -> per-chgroup true-time streams
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# corr-fast subprocess isolation
# ---------------------------------------------------------------------------
#
# The production corr-fast GPU context (CUDA-graph / Triton state in the
# int4-unpack + dedisperse kernels) cannot be torn down and rebuilt twice in
# one process: the SECOND ``build_context`` raises ``CUDA error: resource
# already mapped`` from the Triton unpack kernel. We therefore run corr-fast
# for ONE chgroup per child process and persist its owner-coarse-DM
# true-time stream to disk, then reassemble all 16 streams in the parent.
# This also keeps peak GPU memory low (one chgroup resident at a time).


def _corr_worker(spec_path: str, out_path: str) -> int:
    """Child entry point: run corr-fast for a single chgroup and dump its
    owner-coarse-DM sparse true-time stream + grid pattern to ``out_path``."""
    with open(spec_path) as f:
        spec = json.load(f)
    cell = InjectionCell(**spec["cell"])
    cfg = spec["cfg"]
    chg = int(spec["chgroup"])
    slice_owner = int(spec["slice_owner_idx"])
    save_all = bool(spec.get("save_all_owners", False))
    corr = run_corr_fast_injection(
        cell,
        n_grid=int(cfg["n_grid"]),
        n_blocks=int(cfg["n_blocks"]),
        t_int_fast_native=int(cfg["t_int_fast_native"]),
        chgroups=(chg,),
        device=str(cfg["device"]),
        seed=int(cfg["seed"]),
        dm_plan_path=cfg["dm_plan_path"],
        chan_sum_factor=int(cfg["chan_sum_factor"]),
        dm_chunk_size=2,
        sliding_window=True,
    )
    cg = corr["per_chgroup"][chg]
    save_kw = dict(
        ix_row=np.asarray(cg["ix_row"]),
        ix_col=np.asarray(cg["ix_col"]),
        cell_lambda=np.float64(cg["cell_lambda"]),
        plan_owner_idx=np.int64(corr["plan"]["owner_idx"]),
    )
    if save_all:
        # Persist ALL coarse-DM owner streams (N_DM, T, N_filled) so one
        # corr-fast pass can feed every search GPU (all-owners mode).
        np.savez(out_path, cube_cat=np.ascontiguousarray(cg["cube_cat"]), **save_kw)
    else:
        np.savez(
            out_path,
            owner_stream=np.ascontiguousarray(cg["cube_cat"][slice_owner]),
            **save_kw,
        )
    return 0


def _corr_streams_subprocess(
    *,
    cell_spec: dict,
    cfg_spec: dict,
    slice_owner_idx: int,
    work_dir: str | Path,
    verbose: bool,
    reuse: bool = False,
    save_all_owners: bool = False,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, float, int]:
    """Spawn one corr-fast child process per chgroup (process-isolated CUDA
    context) and reassemble the 16 owner-coarse-DM true-time streams.

    Returns ``(sparse_streams, ix_row_by_g, ix_col_by_g, cell_lambda,
    plan_owner_idx)``. Each chgroup has its OWN sparsity pattern (UV
    coverage differs per chgroup); ``cell_lambda`` is common
    (``cell_lambda_mode="common"``) so the dense grids stack pixel-for-pixel.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    streams: dict[int, np.ndarray] = {}
    ix_row_by_g: dict[int, np.ndarray] = {}
    ix_col_by_g: dict[int, np.ndarray] = {}
    cell_lambda = None
    plan_owner = None
    self_path = str(Path(__file__).resolve())
    for g in range(N_CHGROUP):
        spec = {
            "cell": cell_spec,
            "cfg": cfg_spec,
            "chgroup": int(g),
            "slice_owner_idx": int(slice_owner_idx),
            "save_all_owners": bool(save_all_owners),
        }
        spec_path = work / f"corr_spec_g{g:02d}.json"
        out_path = work / f"corr_out_g{g:02d}.npz"
        spec_path.write_text(json.dumps(spec))
        cached = reuse and out_path.exists()
        if not cached:
            if out_path.exists():
                out_path.unlink()
            cmd = [
                sys.executable, self_path,
                "--corr-worker-spec", str(spec_path),
                "--corr-worker-out", str(out_path),
            ]
            # The corr-fast CUDA-graph/Triton kernels can transiently fail
            # with "CUDA error: invalid argument" if launched in the instant
            # after a previous CUDA process released the device. Retry the
            # child a couple times with a settle delay before giving up.
            last = None
            for attempt in range(3):
                proc = subprocess.run(
                    cmd, env=dict(os.environ), capture_output=True, text=True
                )
                if proc.returncode == 0 and out_path.exists():
                    last = None
                    break
                last = proc
                if verbose:
                    print(f"  [corr g{g:02d}] attempt {attempt+1} failed "
                          f"(rc={proc.returncode}); settling 8s and retrying",
                          flush=True)
                time.sleep(8.0)
            if last is not None:
                raise RuntimeError(
                    f"corr worker chgroup {g} failed (rc={last.returncode}):\n"
                    f"--- stdout ---\n{last.stdout[-2000:]}\n"
                    f"--- stderr ---\n{last.stderr[-2000:]}"
                )
        with np.load(out_path) as d:
            if "cube_cat" in d:
                # all-owners file: slice the requested owner (enables one
                # shared corr pass to feed every search GPU).
                streams[g] = np.ascontiguousarray(
                    d["cube_cat"][int(slice_owner_idx)]
                )
            else:
                streams[g] = np.ascontiguousarray(d["owner_stream"])
            ix_row_by_g[g] = np.asarray(d["ix_row"])
            ix_col_by_g[g] = np.asarray(d["ix_col"])
            if cell_lambda is None:
                cell_lambda = float(d["cell_lambda"])
                plan_owner = int(d["plan_owner_idx"])
        if verbose:
            print(
                f"  [corr g{g:02d}] owner_stream {streams[g].shape} "
                f"{streams[g].dtype} n_filled={ix_row_by_g[g].shape[0]}",
                flush=True,
            )
    return streams, ix_row_by_g, ix_col_by_g, float(cell_lambda), int(plan_owner)


def _owner_true_time_stream(
    corr_result: dict, chgroup: int, owner_idx: int, *, sliding_window: bool,
) -> np.ndarray:
    """Return the chgroup's owner-coarse-DM sparse true-time stream
    ``[T_total, N_filled] complex64`` where index == absolute true-time
    tile. With sliding_window the cold-start block-0 tiles ``[0, 128)``
    are zeros (resolved blocks fill ``[128, n_blocks*128)`` gap-free)."""
    cg = corr_result["per_chgroup"][int(chgroup)]
    cube_cat = cg["cube_cat"]  # (N_DM, n_blocks*T_dedisp, N_filled)
    return np.ascontiguousarray(cube_cat[int(owner_idx)])


# ---------------------------------------------------------------------------
# Slot construction
# ---------------------------------------------------------------------------


def _build_slot_from_streams(
    *,
    per_chgroup_dense: dict[int, np.ndarray],
    time_shift_table: TimeShiftSearchTable,
    offset: int,
    t_det: int,
    n_grid: int,
    n_fdm: int,
    cube_id: int,
    specnum_start: int,
    target_max: int = 120,
    fixed_scale: Optional[float] = None,
) -> tuple[CubeRingSlot, float]:
    """Quantise per-chgroup dense complex streams to cint8 and wrap in a
    ``CubeRingSlot`` with M7.7 symmetric-shift padding.

    ``fixed_scale`` pins the cf->cint8 scale so EVERY cube (burn-in,
    injection, noise-only) shares one absolute scale. Without it the
    per-cube adaptive scale puts the bright injection (source-dominated
    max) and the noise cubes (noise-dominated max) on ~50x different
    scales, so the frozen Layer-1 σ no longer applies to the noise cube
    and pure noise floods the detector with false positives. With a
    shared noise-derived scale the source simply clips at ±127 (still a
    huge-SNR detection) while the noise floor stays Layer-1-consistent.
    """
    t_stream = int(next(iter(per_chgroup_dense.values())).shape[0])
    out_cint8 = np.zeros(
        (N_CHGROUP, t_stream, 2, n_grid, n_grid), dtype=np.int8
    )
    scale = quantise_per_chgroup_into_cint8(
        per_chgroup_dense,
        out_cint8=out_cint8,
        target_max=target_max,
        zero_fill_missing=True,
        fixed_scale=fixed_scale,
    )
    slot = CubeRingSlot(
        cube_id=int(cube_id),
        specnum_start=int(specnum_start),
        per_chgroup_streams=per_chgroup_dense,  # kept for CPU diag; GPU uses cint8
        time_shift_table=time_shift_table,
        validity_mask=np.ones((t_det, n_fdm), dtype=np.bool_),
        n_fdm_in_cube=int(n_fdm),
        t_det=int(t_det),
        n_grid=int(n_grid),
        per_chgroup_cint8_stack=out_cint8,
        per_chgroup_scale=None,  # unit-scale fast path; Layer-1 σ-clip normalises
        stream_origin_offset_samples=int(offset),
    )
    return slot, float(scale)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_search_driver(
    *,
    owner_idx: int = 1,
    dm_pc_cm3: Optional[float] = None,
    dm_target: float = 500.0,
    l_rad: float = 0.004,
    m_rad: float = -0.002,
    width_ms: float = 4.0,
    fluence_jy_ms: float = 5.0e4,
    profile: str = "gaussian",
    t_det: int = 192,
    n_fdm: Optional[int] = None,
    n_grid: int = 256,
    n_blocks: int = 8,
    t_int_fast_native: int = 32,
    chan_sum_factor: int = 8,
    dm_plan_path: str | Path = DEFAULT_DM_PLAN_PATH,
    device: str = "cuda:0",
    n_burnin: int = 8,
    threshold_sigma: float = 8.0,
    bank_mask: Optional[str] = None,
    seed: int = 42,
    search_node_id: int = 1,
    gpu_half: int = 1,
    out_dir: str | Path = "/tmp/inject_search",
    t_target_in_cube: Optional[int] = None,
    run_noise_only: bool = True,
    reuse_corr: bool = False,
    audit_fp32: bool = True,
    zero_dm_filter: bool = True,
    corr_work_dir: Optional[str | Path] = None,
    corr_save_all_owners: bool = False,
    quant_target_max: int = 120,
    verbose: bool = True,
) -> dict:
    """Drive the corr→search bridge + detector for ONE owner coarse-DM.

    Parameters
    ----------
    owner_idx : int
        Coarse-DM owner index ``c`` (e.g. 1 for DM≈500/576).
    dm_pc_cm3 : float | None
        Injected DM. ``None`` → snap to the owner's fine-DM grid value
        closest to ``dm_target``.
    dm_target : float
        Target DM used when ``dm_pc_cm3`` is None.
    l_rad, m_rad, width_ms, fluence_jy_ms, profile :
        Injection cell parameters (passed to the corr-fast injector).
    t_det, n_grid, n_blocks :
        Search cube geometry + number of corr-fast blocks. ``n_blocks``
        must be large enough that the dispersed track at the injected DM
        fits inside the resolved true-time range.
    n_fdm : int | None
        Number of fine-DM trials; ``None`` → owner's full K.
    n_burnin : int
        Number of noise-only burn-in cubes to drive before the injection
        cube (seeds Layer-1 σ + detector Layer-2 EMA).
    threshold_sigma : float
        Detector emit threshold.
    run_noise_only : bool
        Also drive one independent noise-only cube and count candidates
        > 12σ (false-positive sanity).

    Returns
    -------
    dict
        ``{"injected": {...}, "candidates": [...c1 rows...],
           "candidates_csv": path, "candidates_ndjson": path,
           "npz_path": path, "noise_only_fp_count": int,
           "diag": {...}, ...}``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dev = torch.device(device)
    t_int_search_us = float(t_int_fast_native) * float(NATIVE_SAMPLE_US)  # 1048.576
    sample_period_specnum = int(round(t_int_search_us / float(SPECNUM_PERIOD_US)))  # 16

    grids = owner_dm_grids(dm_plan_path, owner_idx)
    if n_fdm is None:
        n_fdm = grids.n_fdm
    if n_fdm != grids.n_fdm:
        raise ValueError(
            f"n_fdm={n_fdm} != owner {owner_idx} fine count {grids.n_fdm}"
        )

    # Snap injected DM to the owner's exact fine-DM grid value.
    if dm_pc_cm3 is None:
        finj = int(np.argmin(np.abs(grids.fine_dm_local - dm_target)))
        dm_pc_cm3 = float(grids.fine_dm_local[finj])
    else:
        finj = int(np.argmin(np.abs(grids.fine_dm_local - dm_pc_cm3)))
        dm_pc_cm3 = float(dm_pc_cm3)

    # ----- search-side time-shift table (escape hatch; full delay) -----
    tst = compute_time_shift_search(
        coarse_dm_pc_cm3=grids.coarse_dm_local,
        fine_dm_pc_cm3=grids.fine_dm_local,
        fine_to_coarse=grids.fine_to_coarse_local,
        t_int_search_us=t_int_search_us,
        include_coarse_offset=True,
    )
    shifts = tst.shifts  # [n_fdm, 16] int32, all >= 0
    offset = int(shifts.max())          # M7.7 pad_left = stream_origin_offset
    pad_right = int(max(0, -int(shifts.min())))  # 0 here (all positive)
    t_stream = int(t_det) + offset + pad_right

    # ----- choose injection peak_specnum to centre the dispersed track -----
    delay_tiles = _chgroup_top_delay_tiles(dm_pc_cm3, t_int_search_us)
    delay_max = int(delay_tiles.max())
    resolved_lo = NATIVE_SAMPLES_PER_BLOCK // (t_int_fast_native)  # = 128 tiles
    resolved_hi = int(n_blocks) * (NATIVE_SAMPLES_PER_BLOCK // t_int_fast_native)
    if t_target_in_cube is None:
        t_target_in_cube = int(t_det) // 2
    # Centre the [tau0, tau0+delay_max] track in the resolved range.
    tau0 = resolved_lo + ((resolved_hi - resolved_lo) - delay_max) // 2
    peak_specnum = int(round(tau0 * (t_int_search_us / float(SPECNUM_PERIOD_US))))
    # peak_native = 2*peak_specnum; tau0_actual = peak_native / t_int_fast_native.
    tau0_actual = (2 * peak_specnum) / float(t_int_fast_native)

    # Window origin so the burst lands at t_target_in_cube in the cube.
    # t* = (tau0 - w0) + shifts[finj, 0] - offset  (telescoped across g)
    w0 = int(round(tau0_actual + float(shifts[finj, 0]) - offset - t_target_in_cube))
    if w0 < resolved_lo:
        raise ValueError(
            f"window start w0={w0} < resolved_lo={resolved_lo}; "
            f"increase n_blocks or lower t_target_in_cube"
        )
    if w0 + t_stream > resolved_hi:
        raise ValueError(
            f"window end {w0 + t_stream} > resolved_hi={resolved_hi}; "
            f"increase n_blocks (need >= {math.ceil((w0 + t_stream) / 128)})"
        )

    if verbose:
        print("=" * 72, flush=True)
        print("corr->search injection driver", flush=True)
        print("=" * 72, flush=True)
        print(
            f"owner_idx={owner_idx} owner_dm={grids.coarse_dm_local[0]:.1f} "
            f"n_fdm={n_fdm}  injected dm={dm_pc_cm3:.3f} (fine idx {finj})",
            flush=True,
        )
        print(
            f"l={l_rad:+.4f} m={m_rad:+.4f} width={width_ms}ms "
            f"fluence={fluence_jy_ms:g}",
            flush=True,
        )
        print(
            f"t_int_search_us={t_int_search_us:.3f} "
            f"sample_period_specnum={sample_period_specnum}",
            flush=True,
        )
        print(
            f"shifts: min={int(shifts.min())} max={offset} "
            f"shifts[finj,0]={int(shifts[finj,0])}  "
            f"delay_max={delay_max} tiles",
            flush=True,
        )
        print(
            f"t_stream={t_stream} (t_det={t_det}+offset={offset}+pad_r={pad_right})  "
            f"tau0={tau0_actual:.0f}  w0={w0}  "
            f"peak_specnum={peak_specnum} (peak_native={2*peak_specnum})  "
            f"target t_in_cube={t_target_in_cube}",
            flush=True,
        )
        print(
            f"n_blocks={n_blocks} resolved true-time [{resolved_lo},{resolved_hi})",
            flush=True,
        )

    # ----- run corr-fast (sliding window -> gapless streams) -----
    # corr-fast is run ONE chgroup per child process: its GPU context
    # (CUDA-graph / Triton unpack kernel state) cannot be rebuilt twice in a
    # single process (second build_context -> "resource already mapped").
    cell_spec = {
        "l_rad": float(l_rad), "m_rad": float(m_rad),
        "dm_pc_cm3": float(dm_pc_cm3), "width_ms": float(width_ms),
        "fluence_jy_ms": float(fluence_jy_ms),
        "peak_specnum": int(peak_specnum), "profile": str(profile),
        "inj_id": f"search_owner{owner_idx}_dm{int(round(dm_pc_cm3))}",
    }
    cfg_spec = {
        "n_grid": int(n_grid), "n_blocks": int(n_blocks),
        "t_int_fast_native": int(t_int_fast_native),
        "device": str(device), "seed": int(seed),
        "dm_plan_path": str(dm_plan_path),
        "chan_sum_factor": int(chan_sum_factor),
    }
    if verbose:
        print("\n[corr-fast] running 16 chgroups x "
              f"{n_blocks} blocks (sliding_window=True, subprocess-isolated "
              f"per chgroup)...", flush=True)
    corr_work = Path(corr_work_dir) if corr_work_dir is not None else Path(out_dir) / "corr_work"
    sparse_streams, ix_row_by_g, ix_col_by_g, cell_lambda, corr_owner = (
        _corr_streams_subprocess(
            cell_spec=cell_spec,
            cfg_spec=cfg_spec,
            slice_owner_idx=owner_idx,
            work_dir=corr_work,
            verbose=verbose,
            reuse=reuse_corr,
            save_all_owners=corr_save_all_owners,
        )
    )
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if corr_owner != owner_idx and verbose:
        print(
            f"  NOTE: corr-fast owner_idx={corr_owner} (closest coarse to "
            f"dm={dm_pc_cm3}); using requested owner_idx={owner_idx} for "
            f"the search slice.",
            flush=True,
        )

    # ----- per-chgroup off-pulse noise sigma (for synthetic noise cubes) -----
    noise_sigma = np.zeros(N_CHGROUP, dtype=np.float64)
    for g in range(N_CHGROUP):
        s = sparse_streams[g]  # (T_total, N_filled) complex
        t0g = int(round(tau0_actual)) + int(delay_tiles[g])
        mask = np.ones(s.shape[0], dtype=bool)
        lo = max(0, t0g - 20)
        hi = min(s.shape[0], t0g + 20)
        mask[:resolved_lo] = False  # drop cold-start zeros
        mask[lo:hi] = False         # drop on-pulse
        off = s[mask]
        if off.size == 0:
            noise_sigma[g] = 1.0
        else:
            noise_sigma[g] = float(
                np.std(np.concatenate([off.real.ravel(), off.imag.ravel()]))
            )
    # Each chgroup has its own sparsity pattern (UV coverage differs).
    n_filled_by_g = {g: int(ix_row_by_g[g].shape[0]) for g in range(N_CHGROUP)}

    def _densify_window(g: int, sparse: np.ndarray, w_lo: int) -> np.ndarray:
        win = sparse[w_lo:w_lo + t_stream]  # (t_stream, N_filled_g)
        dense = sparse_to_dense_grid(
            torch.from_numpy(np.ascontiguousarray(win)),
            ix_row_by_g[g], ix_col_by_g[g], n_grid,
        )
        return dense.numpy().astype(np.complex64, copy=False)

    def _gen_noise_streams() -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for g in range(N_CHGROUP):
            sig = float(noise_sigma[g])
            nf = n_filled_by_g[g]
            vals = (
                rng.standard_normal((t_stream, nf)).astype(np.float32) * sig
                + 1j * rng.standard_normal((t_stream, nf)).astype(np.float32) * sig
            ).astype(np.complex64)
            dense = sparse_to_dense_grid(
                torch.from_numpy(vals), ix_row_by_g[g], ix_col_by_g[g], n_grid
            ).numpy().astype(np.complex64, copy=False)
            out[g] = dense
        return out

    # ----- build the injection cube's per-chgroup dense streams -----
    inj_dense: dict[int, np.ndarray] = {
        g: _densify_window(g, sparse_streams[g], w0) for g in range(N_CHGROUP)
    }

    # ----- CPU pre-combine diagnostic: confirm pixel + arrival t* -----
    eff_shifts = (shifts[finj] - offset).astype(np.int32)
    combined = combine_chgroups(
        per_chgroup_streams=inj_dense,
        time_shift_per_chgroup=eff_shifts,
        t_window=(0, t_det),
        n_grid=n_grid,
    )  # (t_det, N, N) complex
    img = dirty_image_from_dense_grid(
        torch.from_numpy(combined)
    ).numpy()  # (t_det, N, N) f32
    per_t_power = (img ** 2).reshape(t_det, -1).sum(axis=1)
    t_peak = int(np.argmax(per_t_power))
    peak_img = img[t_peak]
    flat = int(np.argmax(np.abs(peak_img)))
    pk_row, pk_col = np.unravel_index(flat, peak_img.shape)
    half = n_grid // 2
    diag = {
        "cpu_combine_t_peak": t_peak,
        "cpu_combine_peak_row": int(pk_row),
        "cpu_combine_peak_col": int(pk_col),
        "cpu_combine_dist_cells": float(math.hypot(pk_row - half, pk_col - half)),
        "expected_t_in_cube": int(t_target_in_cube),
        "noise_sigma_median": float(np.median(noise_sigma)),
    }
    if verbose:
        print(
            f"\n[CPU pre-combine diag] burst at t_in_cube={t_peak} "
            f"(expected ~{t_target_in_cube}); peak pixel "
            f"(row={pk_row}, col={pk_col}) center={half} "
            f"dist={diag['cpu_combine_dist_cells']:.2f} cells",
            flush=True,
        )

    # ----- build the search pipeline + detector -----
    # A fluence=5e4 point source images to a FLAT-UV pattern: nearly all
    # ~5100 gridded cells quantise near the cint8 max, so the coherent FFT
    # sum is ~5100*120 ≈ 6e5, far above the fp16 max (~6.5e4). The
    # complex32 GPU imager therefore overflows to inf in the FFT accumulate
    # (BEFORE any Layer-1 divide can help), which then poisons Layer-1.
    #
    # Fix: image + Layer-1-normalise in fp32 (complex64 imager — no
    # overflow), then DOWNCAST the L1-normalised cube to fp16 before the
    # detector. After normalisation the cube is in σ units (O(1) noise,
    # O(SNR) signal ≪ 6.5e4), so fp16 is exact-enough AND the detector's
    # full-cube fp32 temporaries (which OOM an 11 GB card at n_grid=256)
    # are avoided. Layer-1 normalises by the measured σ, so the absolute
    # visibility scale is irrelevant to the recovered SNR.
    img_dt = torch.float32 if audit_fp32 else torch.float16
    cplx_dt = torch.complex64 if audit_fp32 else torch.complex32
    det_dt = torch.float16  # detector always fp16 (memory); cube downcast post-L1
    image_tokens, dm_tokens, time_tokens = parse_bank_mask(bank_mask)
    bank = build_kernel_bank(
        image_tokens=image_tokens,
        dm_tokens=dm_tokens,
        time_tokens=time_tokens,
        dtype=det_dt,
    )
    detector = DeterministicDetector(
        kernel_bank=bank,
        threshold_sigma=float(threshold_sigma),
        detector_version="v1.M5",
        search_node_id=int(search_node_id),
        gpu_half=int(gpu_half),
        dtype=det_dt,
        device=dev,
        streaming=True,                # production-geometry-safe path
        streaming_tile_size=64,
    )
    pipeline_cfg = CubePipelineConfig(
        n_grid=n_grid,
        edge_mask_kernel_support=5,
        cube_dtype=img_dt,
        device=str(dev),
        image_backend="gpu",
        gpu_t_det=t_det,
        gpu_n_fdm=n_fdm,
        gpu_n_chgroup=N_CHGROUP,
        gpu_complex_dtype=cplx_dt,
        cube_pipeline_carry_over_re_imaging=False,
    )
    pipeline = CubePipeline(
        config=pipeline_cfg,
        detector=detector,
        layer1_state=Layer1State(
            n_fdm=n_fdm, n_burnin_cubes=max(1, n_burnin - 1),
            max_samples=1_000_000,
        ),
        fine_dm_pc_cm3=torch.as_tensor(
            grids.fine_dm_local, dtype=torch.float32, device=dev
        ),
    )

    # Imaging (fp32 GpuImager) and detection (fp32 boxcar-cumsum scratch)
    # have working sets that do NOT coexist in 11 GB at n_grid=256. We
    # therefore run the whole thing in two phases: PHASE 1 images +
    # Layer-1-normalises every cube (imager resident, detector idle),
    # stashing each L1-normalised fp16 cube on the HOST; then we free the
    # imager and PHASE 2 runs the detector on the stashed cubes.
    fine_dm_tensor = torch.as_tensor(
        grids.fine_dm_local, dtype=torch.float32, device=dev
    )

    def _image_normalise(slot: CubeRingSlot):
        """Image + Layer-1-normalise one cube; return (cube_fp16_cpu,
        sigma_cpu, vmask_cpu). Updates the stateful Layer-1 burn-in.

        A zero-DM filter (subtract the per-pixel temporal mean) removes
        the static DC term that survives ``static_sky_disabled=True`` in
        corr-fast. That DC sits at the phase centre and its b128
        time-boxcar otherwise overflows fp16 to inf; the transient burst
        (~4 of 192 samples) is essentially unaffected by the mean
        subtraction."""
        cube, vmask = pipeline._build_cube(slot)
        cube_norm, sigma = pipeline._layer1_normalise(cube)
        del cube
        if zero_dm_filter:
            cube_norm = cube_norm - cube_norm.mean(dim=0, keepdim=True)
        cube16 = cube_norm.to(det_dt).cpu()
        sig = sigma.detach().to("cpu")
        vm = vmask.detach().to("cpu")
        del cube_norm
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return cube16, sig, vm

    # Shared cf->cint8 scale derived from a reference noise cube so all
    # cubes are mutually scale-consistent (see _build_slot_from_streams).
    def _global_quant_scale(dense: dict[int, np.ndarray],
                            target_max: int = 120) -> float:
        gmax = 0.0
        for s in dense.values():
            gmax = max(gmax, float(np.abs(s.real).max(initial=0.0)),
                       float(np.abs(s.imag).max(initial=0.0)))
        return (float(target_max) / gmax) if gmax > 0.0 else 1.0

    ref_noise = _gen_noise_streams()
    quant_scale = _global_quant_scale(ref_noise, target_max=int(quant_target_max))
    if verbose:
        print(f"\n[search] shared fixed quant scale={quant_scale:.4g} "
              f"(noise-derived; injection clips at ±127)", flush=True)

    # ===== PHASE 1: image + Layer-1 (imager resident) =====
    if verbose:
        print(f"\n[search][phase1] {n_burnin} burn-in noise cubes "
              f"(image+Layer-1)...", flush=True)
    for k in range(n_burnin):
        noise_dense = ref_noise if k == 0 else _gen_noise_streams()
        slot, _ = _build_slot_from_streams(
            per_chgroup_dense=noise_dense,
            time_shift_table=tst, offset=offset,
            t_det=t_det, n_grid=n_grid, n_fdm=n_fdm,
            cube_id=k, specnum_start=k * t_det,
            fixed_scale=quant_scale,
        )
        _, sig_k, _ = _image_normalise(slot)
        if verbose:
            print(
                f"  burn-in cube {k} imaged "
                f"(sigma_l1 med={float(sig_k.float().median()):.4g})",
                flush=True,
            )

    inj_cube_id = n_burnin
    inj_specnum_start = int(w0 + offset)  # search-sample units (== true-time tile)
    slot_inj, scale_inj = _build_slot_from_streams(
        per_chgroup_dense=inj_dense,
        time_shift_table=tst, offset=offset,
        t_det=t_det, n_grid=n_grid, n_fdm=n_fdm,
        cube_id=inj_cube_id, specnum_start=inj_specnum_start,
        fixed_scale=quant_scale,
    )
    if verbose:
        print(f"\n[search][phase1] injection cube (id={inj_cube_id}, "
              f"specnum_start={inj_specnum_start}, quant scale={scale_inj:.4g})...",
              flush=True)
    inj_cube16, inj_sigma, inj_vmask = _image_normalise(slot_inj)
    inj_cube_max = float(inj_cube16.float().abs().max())
    if verbose:
        _ic = inj_cube16.float()
        print(f"  inj cube_norm: std={_ic.std():.3f} max={_ic.abs().max():.1f} "
              f">8σ cells={int((_ic.abs()>8).sum())}", flush=True)

    no_cube16 = no_sigma = no_vmask = None
    noise_specnum = int((inj_cube_id + 1) * t_det)
    if run_noise_only:
        noise_dense = _gen_noise_streams()
        slot_noise, _ = _build_slot_from_streams(
            per_chgroup_dense=noise_dense,
            time_shift_table=tst, offset=offset,
            t_det=t_det, n_grid=n_grid, n_fdm=n_fdm,
            cube_id=inj_cube_id + 1, specnum_start=noise_specnum,
            fixed_scale=quant_scale,
        )
        if verbose:
            print("\n[search][phase1] noise-only cube (image+Layer-1)...",
                  flush=True)
        no_cube16, no_sigma, no_vmask = _image_normalise(slot_noise)
        if verbose:
            _nc = no_cube16.float()
            print(f"  noise cube_norm: std={_nc.std():.3f} "
                  f"max={_nc.abs().max():.1f} >8σ cells={int((_nc.abs()>8).sum())}",
                  flush=True)

    # ===== free the imager, then PHASE 2: detection =====
    del pipeline
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    def _detect(cube16_cpu, sigma_cpu, vmask_cpu, specnum):
        c = cube16_cpu.to(dev)
        s = sigma_cpu.to(dev)
        v = vmask_cpu.to(dev)
        with torch.no_grad():
            out = detector.forward(
                c, v, s,
                event_specnum=int(specnum),
                fine_dm_pc_cm3=fine_dm_tensor,
            )
        del c, s, v
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out

    if verbose:
        print("\n[search][phase2] detecting injection cube...", flush=True)
    cands = list(_detect(inj_cube16, inj_sigma, inj_vmask, inj_specnum_start))
    res_inj = SimpleNamespace(cube=inj_cube16, candidates=cands)
    if verbose:
        print(f"  -> {len(cands)} candidates", flush=True)

    # ----- geometry for C1 projection -----
    pixel_size_lm = 1.0 / (n_grid * cell_lambda)
    mjd_start = 60000.0 + inj_specnum_start * t_int_search_us * 1e-6 / 86400.0
    geom = CubeGeometry(
        cube_id=inj_cube_id,
        specnum_start=inj_specnum_start,
        sample_period_specnum=sample_period_specnum,
        t_det=t_det,
        n_grid=n_grid,
        n_fdm_in_cube=n_fdm,
        sample_period_us=t_int_search_us,
        cell_l_rad=float(pixel_size_lm),
        cell_m_rad=float(pixel_size_lm),
        l0_rad=float(-half * pixel_size_lm),
        m0_rad=float(-half * pixel_size_lm),
        fine_dm_pc_cc=grids.fine_dm_local.astype(np.float64, copy=False),
        mjd_start=float(mjd_start),
    )

    rows = [candidate_to_c1_row(c, geom=geom) for c in cands]

    # ----- write C1 csv + ndjson -----
    csv_path = out_dir / "candidates_c1.csv"
    ndjson_path = out_dir / "candidates_c1.ndjson"
    fieldnames = [
        "snr", "l_pix", "m_pix", "l_rad", "m_rad", "dm_pc_cc",
        "fine_dm_idx", "dm_idx_global", "event_specnum", "width_samples",
        "kernel_id", "flags", "t_in_cube",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            t_in_cube = (
                (int(r.event_specnum) - inj_specnum_start)
            )
            w.writerow({
                "snr": f"{r.snr:.4f}", "l_pix": r.l_pix, "m_pix": r.m_pix,
                "l_rad": f"{r.l_rad:.6e}", "m_rad": f"{r.m_rad:.6e}",
                "dm_pc_cc": f"{r.dm_pc_cc:.4f}", "fine_dm_idx": r.fine_dm_idx,
                "dm_idx_global": r.dm_idx_global,
                "event_specnum": r.event_specnum,
                "width_samples": r.width_samples, "kernel_id": r.kernel_id,
                "flags": r.flags, "t_in_cube": t_in_cube,
            })
    with ndjson_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "snr": float(r.snr), "l_pix": int(r.l_pix), "m_pix": int(r.m_pix),
                "l_rad": float(r.l_rad), "m_rad": float(r.m_rad),
                "dm_pc_cc": float(r.dm_pc_cc), "fine_dm_idx": int(r.fine_dm_idx),
                "dm_idx_global": int(r.dm_idx_global),
                "event_specnum": int(r.event_specnum),
                "width_samples": int(r.width_samples), "kernel_id": r.kernel_id,
                "flags": int(r.flags),
            }) + "\n")

    # ----- write NPZ cube (notebook-compatible) -----
    cube_np = res_inj.cube.detach().to(torch.float16).cpu().numpy()
    npz_path = out_dir / (
        f"cube_s{search_node_id}_g{gpu_half}_{inj_specnum_start}.npz"
    )
    np.savez(
        npz_path,
        cube=cube_np.astype(np.float16),
        mjd_start=np.asarray(mjd_start, dtype=np.float64),
        event_specnum_start=np.asarray(inj_specnum_start, dtype=np.int64),
        t_det=np.asarray(t_det, dtype=np.int32),
        n_fdm_in_cube=np.asarray(n_fdm, dtype=np.int32),
        n_grid=np.asarray(n_grid, dtype=np.int32),
        cluster_record=np.asarray("null", dtype="U"),
        trigger_source=np.asarray("udp", dtype="U"),
        search_node_id=np.asarray(search_node_id, dtype=np.int32),
        gpu_half=np.asarray(gpu_half, dtype=np.int32),
    )

    # ----- noise-only false-positive check (phase-2 detection) -----
    noise_fp_count = None
    noise_cands = []
    if run_noise_only and no_cube16 is not None:
        if verbose:
            print("\n[search][phase2] detecting noise-only cube...", flush=True)
        noise_cands = list(
            _detect(no_cube16, no_sigma, no_vmask, noise_specnum)
        )
        noise_fp_count = int(sum(1 for c in noise_cands if c.snr > 12.0))
        if verbose:
            print(
                f"\n[noise-only] {len(noise_cands)} candidates total, "
                f"{noise_fp_count} with SNR>12σ",
                flush=True,
            )
            for c in sorted(noise_cands, key=lambda c: -c.snr)[:5]:
                print(f"    noise cand snr={c.snr:.1f} l={c.l} m={c.m} "
                      f"dm_idx={c.dm_idx} width={c.width_samples} "
                      f"kid={c.kernel_id}", flush=True)

    # ----- summarise -----
    result = {
        "injected": {
            "owner_idx": owner_idx,
            "owner_dm": float(grids.coarse_dm_local[0]),
            "dm_pc_cm3": dm_pc_cm3,
            "fine_dm_idx": finj,
            "l_rad": l_rad, "m_rad": m_rad,
            "l_pix_expected": int(round(half + l_rad / pixel_size_lm)),
            "m_pix_expected": int(round(half + m_rad / pixel_size_lm)),
            "width_ms": width_ms,
            "width_samples_expected": int(round(width_ms * 1e3 / t_int_search_us)),
            "peak_specnum": peak_specnum,
            "expected_event_specnum": inj_specnum_start + t_target_in_cube,
        },
        "candidates": [
            {
                "snr": float(r.snr), "l_pix": int(r.l_pix), "m_pix": int(r.m_pix),
                "dm_pc_cc": float(r.dm_pc_cc), "fine_dm_idx": int(r.fine_dm_idx),
                "event_specnum": int(r.event_specnum),
                "t_in_cube": int(r.event_specnum) - inj_specnum_start,
                "width_samples": int(r.width_samples), "kernel_id": r.kernel_id,
                "flags": int(r.flags),
            }
            for r in rows
        ],
        "candidates_csv": str(csv_path),
        "candidates_ndjson": str(ndjson_path),
        "npz_path": str(npz_path),
        "noise_only_fp_count": noise_fp_count,
        "noise_only_total": len(noise_cands),
        "inj_cube_max": inj_cube_max,
        "diag": diag,
        "specnum_start": inj_specnum_start,
        "n_fdm": n_fdm,
        "t_det": t_det,
        "n_grid": n_grid,
        "pixel_size_lm": float(pixel_size_lm),
    }
    return result


def run_all_owners(
    *,
    dm_pc_cm3: float,
    l_rad: float = 0.004,
    m_rad: float = -0.002,
    width_ms: float = 1.0,
    fluence_jy_ms: float = 200.0,
    profile: str = "gaussian",
    t_det: int = 192,
    n_grid: int = 256,
    n_blocks: Optional[int] = None,
    chan_sum_factor: int = 8,
    dm_plan_path: str | Path = DEFAULT_DM_PLAN_PATH,
    device: str = "cuda:0",
    n_burnin: int = 8,
    threshold_sigma: float = 8.0,
    seed: int = 42,
    out_dir: str | Path = "/tmp/inject_allgpu",
    run_noise_only_owning: bool = True,
    keep_corr_scratch: bool = False,
    quant_target_max: int = 120,
    verbose: bool = True,
) -> dict:
    """Inject ONE burst at the true DM, run corr-fast ONCE, then image+detect
    it through the **exact fp16 production search path** for EVERY coarse-DM
    owner (all search GPUs). Produces one notebook NPZ cube per owner.

    The shared corr-fast pass writes all-owners ``cube_cat`` files under
    ``out_dir/corr_shared``; owner ``o``'s cube + C1 land under
    ``out_dir/owner{o}``. The burst focuses sharply only in the owning GPU
    (coarse DM closest to ``dm_pc_cm3``); other GPUs show the smeared /
    mis-dedispersed response at the same centred time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_shared = out_dir / "corr_shared"

    coarse_dm, _, _ = _dm_grids_from_npz(Path(dm_plan_path), n_coarse=8)
    n_owners = int(coarse_dm.shape[0])
    owning = int(np.argmin(np.abs(coarse_dm - float(dm_pc_cm3))))
    if n_blocks is None:
        # window must fit the true-DM sweep: scale blocks with DM.
        n_blocks = 8 if dm_pc_cm3 <= 700 else (12 if dm_pc_cm3 <= 1400 else 20)

    if verbose:
        print("#" * 72, flush=True)
        print(f"ALL-OWNERS injection: dm={dm_pc_cm3} width={width_ms}ms "
              f"fluence={fluence_jy_ms:g} -> {n_owners} owners "
              f"(owning={owning}, n_blocks={n_blocks})", flush=True)
        print("#" * 72, flush=True)

    owners_out: list[dict] = []
    for o in range(n_owners):
        owner_dir = out_dir / f"owner{o}"
        if verbose:
            print(f"\n{'='*72}\n[all-owners] owner {o}/{n_owners-1} "
                  f"(coarse_dm={coarse_dm[o]:.1f})\n{'='*72}", flush=True)
        res = run_search_driver(
            owner_idx=o,
            dm_pc_cm3=float(dm_pc_cm3),     # TRUE DM (not snapped per owner)
            dm_target=float(dm_pc_cm3),
            l_rad=float(l_rad), m_rad=float(m_rad),
            width_ms=float(width_ms), fluence_jy_ms=float(fluence_jy_ms),
            profile=profile, t_det=t_det, n_grid=n_grid, n_fdm=None,
            n_blocks=int(n_blocks), chan_sum_factor=chan_sum_factor,
            dm_plan_path=dm_plan_path, device=device, n_burnin=n_burnin,
            threshold_sigma=threshold_sigma, seed=seed,
            search_node_id=o, gpu_half=o,
            out_dir=owner_dir,
            run_noise_only=bool(run_noise_only_owning and o == owning),
            reuse_corr=(o != 0),          # owner 0 runs corr; rest reuse
            audit_fp32=False,             # EXACT fp16/complex32 production imager
            zero_dm_filter=True,
            corr_work_dir=corr_shared,
            corr_save_all_owners=True,
            quant_target_max=int(quant_target_max),
            verbose=verbose,
        )
        cands = sorted(res["candidates"], key=lambda c: -c["snr"])
        top = cands[0] if cands else None
        owners_out.append({
            "owner_idx": o,
            "coarse_dm": float(coarse_dm[o]),
            "is_owning": (o == owning),
            "npz_path": res["npz_path"],
            "candidates_csv": res["candidates_csv"],
            "n_candidates": len(cands),
            "top_snr": (float(top["snr"]) if top else None),
            "top_dm": (float(top["dm_pc_cc"]) if top else None),
            "top_fdm": (int(top["fine_dm_idx"]) if top else None),
            "top_l": (int(top["l_pix"]) if top else None),
            "top_m": (int(top["m_pix"]) if top else None),
            "top_box": (int(top["width_samples"]) if top else None),
            "noise_only_fp_count": res.get("noise_only_fp_count"),
        })
        if verbose and top:
            print(f"  [owner {o}] top snr={top['snr']:.1f} dm={top['dm_pc_cc']:.0f} "
                  f"fdm={top['fine_dm_idx']} pix=({top['l_pix']},{top['m_pix']}) "
                  f"b{top['width_samples']}", flush=True)

    summary = {
        "dm_pc_cm3": float(dm_pc_cm3), "width_ms": float(width_ms),
        "fluence_jy_ms": float(fluence_jy_ms), "owning_owner": owning,
        "n_owners": n_owners, "owners": owners_out,
    }
    (out_dir / "allgpu_summary.json").write_text(json.dumps(summary, indent=2))
    if not keep_corr_scratch:
        import shutil
        shutil.rmtree(corr_shared, ignore_errors=True)
        if verbose:
            print(f"[all-owners] removed corr scratch {corr_shared}", flush=True)
    if verbose:
        print(f"\n[all-owners] owning={owning} "
              f"owning_snr={owners_out[owning]['top_snr']}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(result: dict) -> None:
    inj = result["injected"]
    print("\n" + "=" * 72, flush=True)
    print("VERIFICATION", flush=True)
    print("=" * 72, flush=True)
    print(
        f"injected: owner_idx={inj['owner_idx']} dm={inj['dm_pc_cm3']:.3f} "
        f"(fine idx {inj['fine_dm_idx']})  "
        f"truth pix (row,col)=({result['diag']['cpu_combine_peak_row']},"
        f"{result['diag']['cpu_combine_peak_col']})  "
        f"width~{inj['width_samples_expected']} samp  "
        f"expected event_specnum~{inj['expected_event_specnum']} "
        f"(t_in_cube~{result['diag']['expected_t_in_cube']})",
        flush=True,
    )
    cands = sorted(result["candidates"], key=lambda c: -c["snr"])
    print(f"\ntop candidates ({len(cands)} total):", flush=True)
    print(
        f"{'snr':>9} {'l_pix':>6} {'m_pix':>6} {'dm_pc_cc':>10} "
        f"{'fdm':>4} {'t_in_cube':>9} {'width':>6} {'kernel':>14}",
        flush=True,
    )
    for c in cands[:12]:
        print(
            f"{c['snr']:>9.2f} {c['l_pix']:>6} {c['m_pix']:>6} "
            f"{c['dm_pc_cc']:>10.2f} {c['fine_dm_idx']:>4} "
            f"{c['t_in_cube']:>9} {c['width_samples']:>6} {c['kernel_id']:>14}",
            flush=True,
        )

    # PASS criteria. The detector's (l_pix, m_pix) follow the cube's
    # (row, col) spatial axes; the faithful in-cube truth location is the
    # CPU pre-combine dirty-image peak (same imaging of the bridged
    # streams), so compare against (peak_row, peak_col).
    truth_row = int(result["diag"]["cpu_combine_peak_row"])
    truth_col = int(result["diag"]["cpu_combine_peak_col"])
    ok = []
    for c in cands:
        dist = math.hypot(c["l_pix"] - truth_row, c["m_pix"] - truth_col)
        dm_ok = abs(c["fine_dm_idx"] - inj["fine_dm_idx"]) <= 1
        if dist <= 4.0 and dm_ok and c["snr"] > 12.0:
            ok.append((c, dist))
    print("\n--- PASS check ---", flush=True)
    if ok:
        c, dist = ok[0]
        print(
            f"PASS: candidate snr={c['snr']:.1f} at (l_pix={c['l_pix']}, "
            f"m_pix={c['m_pix']}) dist={dist:.2f} cells, fine_dm_idx="
            f"{c['fine_dm_idx']} (inj {inj['fine_dm_idx']}), width="
            f"{c['width_samples']} (inj {inj['width_samples_expected']}), "
            f"t_in_cube={c['t_in_cube']}",
            flush=True,
        )
    else:
        print("FAIL: no candidate matched position+DM+SNR criteria", flush=True)
    print(
        f"\nnoise-only false positives (>12σ): "
        f"{result['noise_only_fp_count']} / {result['noise_only_total']} total",
        flush=True,
    )
    print(f"\nC1 csv:  {result['candidates_csv']}", flush=True)
    print(f"NPZ cube: {result['npz_path']}", flush=True)

    # NPZ load-back confirmation
    with np.load(result["npz_path"], allow_pickle=False) as d:
        print(
            f"NPZ load-back: cube.shape={d['cube'].shape} dtype={d['cube'].dtype}  "
            f"keys={sorted(d.keys())}",
            flush=True,
        )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner-idx", type=int, default=1)
    p.add_argument("--dm", type=float, default=None,
                   help="injected DM; default snaps to grid near --dm-target")
    p.add_argument("--dm-target", type=float, default=500.0)
    p.add_argument("--l", type=float, default=0.004, dest="l_rad")
    p.add_argument("--m", type=float, default=-0.002, dest="m_rad")
    p.add_argument("--width-ms", type=float, default=4.0)
    p.add_argument("--fluence", type=float, default=5.0e4)
    p.add_argument("--t-det", type=int, default=192)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--chan-sum-factor", type=int, default=8,
                   help="corr-fast channel-sum factor. 8 = production summed "
                        "plan (48 ch/chgroup, fits 11GB GPU with sliding "
                        "window); 1 = full 384 ch/chgroup (OOMs on 2080 Ti).")
    p.add_argument("--n-burnin", type=int, default=8)
    p.add_argument("--threshold-sigma", type=float, default=8.0)
    p.add_argument("--bank-mask", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dm-plan", type=str, default=DEFAULT_DM_PLAN_PATH,
                   dest="dm_plan_path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="/tmp/inject_search", dest="out_dir")
    p.add_argument("--no-noise-only", action="store_true")
    p.add_argument("--reuse-corr", action="store_true",
                   help="reuse cached per-chgroup corr-fast npz outputs in "
                        "<out>/corr_work if present (skips re-running corr).")
    p.add_argument("--no-audit-fp32", action="store_true",
                   help="use the production fp16/complex32 imager instead of "
                        "the complex64/fp32 numerical-audit path (fp16 "
                        "overflows on very bright injections).")
    p.add_argument("--no-zero-dm", action="store_true",
                   help="disable the zero-DM (per-pixel temporal-mean) filter "
                        "that removes the static DC / phase-centre artifact.")
    # Internal: single-chgroup corr-fast child process (see
    # _corr_streams_subprocess). Not for direct use.
    p.add_argument("--corr-worker-spec", type=str, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--corr-worker-out", type=str, default=None,
                   help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.corr_worker_spec is not None:
        return _corr_worker(args.corr_worker_spec, args.corr_worker_out)
    result = run_search_driver(
        owner_idx=args.owner_idx,
        dm_pc_cm3=args.dm,
        dm_target=args.dm_target,
        l_rad=args.l_rad, m_rad=args.m_rad,
        width_ms=args.width_ms, fluence_jy_ms=args.fluence,
        t_det=args.t_det, n_grid=args.n_grid, n_blocks=args.n_blocks,
        chan_sum_factor=args.chan_sum_factor,
        dm_plan_path=args.dm_plan_path, device=args.device,
        n_burnin=args.n_burnin, threshold_sigma=args.threshold_sigma,
        bank_mask=args.bank_mask, seed=args.seed, out_dir=args.out_dir,
        run_noise_only=not args.no_noise_only,
        reuse_corr=args.reuse_corr,
        audit_fp32=not args.no_audit_fp32,
        zero_dm_filter=not args.no_zero_dm,
    )
    _report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
