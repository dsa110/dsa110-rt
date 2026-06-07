#!/usr/bin/env python3
"""bench/preflight/_inject_corr_fixture.py — corr-fast injection fixture.

Reusable helper that drives the **production corr-fast pipeline** on
synthetic *pure-noise* voltages with a voltage-domain
:class:`dsart.inject.online.OnlineInjector`, producing the gridded
**coarse-DM** products that the SEARCH side consumes — plus a dirty
image for operator inspection.

It is the corr-side half of a correctness harness: the SEARCH stage
consumes the per-``(chgroup, coarse-DM)`` gridded uv streams produced
here, so this module is the canonical way to manufacture a known
ground-truth coarse-DM cube without PSRDADA / real voltages / cal.

What it exercises (and what it deliberately does NOT)
-----------------------------------------------------
Per the verification brief we run with:

* **NO calibration** (``cal_path=None`` ⇒ ``ctx.cal is None``). With
  pure-noise voltages and identity cal, the injected source images at
  its injected ``(l, m)`` in the *array-zenith* frame, so a source at
  ``(l, m) = (0, 0)`` lands at the image-centre pixel ``n_grid // 2``.
* **NO RFI flagging** (``rfi_enabled=False``) — there is nothing to
  flag in pure noise, and the flagger would otherwise mistake the
  injected narrowband-ish chirp for CW.
* **NO static-sky EMA** (``static_sky_disabled=True``) — a transient
  pulse is not the time-mean, but disabling the EMA keeps the
  verification image free of warmup transients.
* The **REAL coarse-DM stage** (``Stage1MultiDMCoarseDM``) wired by
  passing ``dm_plan_path`` to :class:`FastIntegrationConfig`. This is
  the key difference from :func:`bench._corr_fast_replay.replay_chgroup`
  (which uses ``NoOpCoarseDM``): we want the 8 coarse-DM grids that the
  search side consumes, not the single-DM gridded cube.

The injector is added **post-cal** (here, post-nothing) into the raw
voltage stream with a **bare geometric phasor** (``cal_gain=None``, set
by ``build_context``); the cold-plasma dispersion is referenced to
``NU_TOP_PROC_GHZ``.

Output shapes (verified against the code, see module-level notes)
-----------------------------------------------------------------
With the multi-DM path active, ``IntegrationOutput.gridded_minus_sky``
is the **dedispersed cube** of shape ``(N_DM, T_dedisp, N_filled)``
complex64 (NOT the single-DM ``(n_fv, N_filled)`` of the NoOp path):

* ``N_DM``      = number of coarse-DM trials (= ``plan.n_coarse`` = 8).
* ``T_dedisp``  = ``n_fast_vis_per_block - max_delay_bins_per_chgroup``
                  (uniform across DM trials; Convention-A reference =
                  chgroup TOP). At ``t_int_fast_native = 32`` →
                  ``n_fast_vis_per_block = 4096 / 32 = 128``.
* ``N_filled``  = number of filled uv-grid cells in the chgroup's
                  :class:`SparsityPattern` (``ctx.gridder.pattern.n_filled``).

The sparse cube is densified to a ``(T, N_grid, N_grid)`` uv-grid via
``sparse[:, k] -> dense[:, ix_row[k], ix_col[k]]`` and imaged with
``Re(fftshift(ifft2(ifftshift(grid))))`` — both helpers reused from
:mod:`bench._corr_fast_replay`.

CLI
---
Running this module as ``__main__`` executes the single verification
injection (l=m=0, DM=500, ~4 ms gaussian), builds a dirty image, finds
the peak pixel, converts it to ``(l, m)``, and prints the recovered vs
injected position, the peak SNR, and a per-coarse-DM peak-amplitude
table. The dirty image PNG is written to
``/tmp/inject_corrfast_verify.png``.

Run on the GPU host (n01) with::

    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
    conda activate dsa110-rt
    cd /home/ubuntu/proj/dsa110-rt
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:bench:. \\
        python bench/preflight/_inject_corr_fixture.py
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# --- repo-root bootstrap so the file runs as a plain script too -----------
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dsart.coarse_dm.dm_plan import DMPlan, load_dm_plan  # noqa: E402
from dsart.common.constants import (  # noqa: E402
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NATIVE_SAMPLE_US,
)
from dsart.inject.online import InjectionConfig  # noqa: E402
from dsart.services.corr_fast_integration import (  # noqa: E402
    FastIntegrationConfig,
    IntegrationOutput,
    _build_core_baseline_mask,
    build_context,
    process_block,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)

from bench._corr_fast_replay import (  # noqa: E402
    dirty_image_from_dense_grid,
    pixel_to_lm_radians,
    sparse_to_dense_grid,
)

__all__ = [
    "InjectionCell",
    "NATIVE_SAMPLES_PER_BLOCK",
    "run_corr_fast_injection",
    "synth_antpos",
    "default_dm_plan_path",
    "DEFAULT_DM_PLAN_PATH",
]

#: Native samples per fada block (= 2048 packets x 2 native samples).
NATIVE_SAMPLES_PER_BLOCK: int = NPACKETS_PER_BLOCK * NTIMES_PER_PACKET  # 4096

#: Production DM plan used by the verification CLI (N=8 coarse trials).
DEFAULT_DM_PLAN_PATH: str = (
    "/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz"
)


def default_dm_plan_path() -> str:
    """Return the default production DM-plan path (string)."""
    return DEFAULT_DM_PLAN_PATH


# ---------------------------------------------------------------------------
# One injection cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectionCell:
    """One synthetic injection (the harness's ground truth).

    Attributes
    ----------
    l_rad, m_rad : float
        Direction cosines of the synthetic source. ``(0, 0)`` = array
        zenith = image-centre pixel (no cal / no fringe-stop).
    dm_pc_cm3 : float
        Dispersion measure (pc / cm^3), cold-plasma law referenced to
        ``NU_TOP_PROC_GHZ``.
    width_ms : float
        FWHM pulse width in milliseconds. Converted to NATIVE samples
        via :attr:`width_samples` (1 native sample = 32.768 us).
    fluence_jy_ms : float
        Peak fluence (Jy.ms). Peak voltage amplitude scale is
        ``sqrt(fluence / width_samples)``.
    peak_specnum : int
        SNAP packet sequence number of the *top-channel* pulse peak
        (``OnlineInjector.apply_at_specnum``). Native peak position is
        ``2 * peak_specnum``.
    profile : str
        ``"gaussian"`` or ``"boxcar"``.
    inj_id : str
        Opaque id echoed in the injection log.
    """

    l_rad: float
    m_rad: float
    dm_pc_cm3: float
    width_ms: float
    fluence_jy_ms: float
    peak_specnum: int
    profile: str = "gaussian"
    inj_id: str = "verify"

    @property
    def width_samples(self) -> int:
        """FWHM width in NATIVE samples (round(width_ms / 32.768us))."""
        w = int(round(self.width_ms * 1.0e3 / NATIVE_SAMPLE_US))
        return max(1, w)

    def to_injection_config(self) -> InjectionConfig:
        """Materialise the :class:`InjectionConfig` for the injector."""
        return InjectionConfig(
            inj_id=self.inj_id,
            l_rad=float(self.l_rad),
            m_rad=float(self.m_rad),
            dm_pc_cm3=float(self.dm_pc_cm3),
            fluence_jy_ms=float(self.fluence_jy_ms),
            width_samples=int(self.width_samples),
            profile=str(self.profile),
            apply_at_specnum=int(self.peak_specnum),
        )


# ---------------------------------------------------------------------------
# Synthetic geometry + voltages
# ---------------------------------------------------------------------------


def synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic ``(NANTS,)`` E / N antenna positions (metres).

    Mirrors ``tests/test_voltage_fixture_replay.py::_synth_antpos``: a
    compact 82-antenna core (+-300 m) plus 14 outriggers, U = 0 (planar
    array — the DSA-110 core is approximately planar so the injector's
    U-component is omitted). Produces a clean point source at the
    injected ``(l, m)`` after gridding + iFFT.
    """
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _pure_noise_block(seed: int, chgroup: int, block_idx: int) -> np.ndarray:
    """One fada block of pure-noise voltages = random uint8 bytes.

    Matches ``tests/test_voltage_fixture_replay.py``: a fresh RNG per
    ``(chgroup, block)`` so each block is independent white noise. The
    bytes decode through the int4 ASR unpack into ~zero-mean voltages;
    the injector adds the deterministic point-source contribution on
    top.
    """
    rng = np.random.default_rng(
        (int(seed) << 20) ^ (int(chgroup) << 10) ^ int(block_idx)
    )
    return rng.integers(0, 256, size=FADA_BYTES_PER_BLOCK, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------


def run_corr_fast_injection(
    cell: InjectionCell,
    *,
    n_grid: int = 256,
    n_blocks: int = 6,
    t_int_fast_native: int = 32,
    chgroups: tuple[int, ...] = (0,),
    device: torch.device | str = "cuda:0",
    seed: int = 42,
    dm_plan_path: str | Path = DEFAULT_DM_PLAN_PATH,
    chan_sum_factor: int = 1,
    dm_chunk_size: int = 2,
    obs_dec_deg: float = 37.23,
    static_sky_disabled: bool = True,
    sliding_window: bool = False,
    antpos_seed: int = 42,
) -> dict:
    """Run the corr-fast pipeline + injector and return coarse-DM products.

    For each requested ``chgroup`` this builds synthetic pure-noise
    voltages, configures the :class:`OnlineInjector` via
    ``inject_configs``, constructs an :class:`IntegrationContext` with
    the **real** multi-DM coarse-DM stage (by passing ``dm_plan_path``),
    and runs :func:`process_block` over ``n_blocks`` blocks.

    Parameters
    ----------
    cell : InjectionCell
        Ground-truth injection.
    n_grid : int
        UV-grid / image side length (cells).
    n_blocks : int
        Number of fada blocks to process (block counter is 1-indexed).
    t_int_fast_native : int
        Fast-corr integration depth in NATIVE samples per fast-vis
        tile. MUST equal the DM plan's ``t_int_fast_native`` (the
        plan's ``t_int_fast_us / 32.768``) — ``build_context`` raises
        otherwise. For the production N=8 plan this is **32**.
    chgroups : tuple[int, ...]
        Chgroups to process (0..15). Default ``(0,)`` (top of band,
        minimal absolute dispersion offset).
    device : torch.device | str
        Compute device. Use ``"cuda:0"`` on n01.
    seed : int
        Seed for the pure-noise voltage RNG.
    dm_plan_path : str | Path
        Path to the canonical DM-plan ``.npz``.
    chan_sum_factor : int
        F33 channel-sum factor. ``1`` = full per-fine-channel plan
        (384 ch/chgroup); ``8`` = production summed plan (48 ch/chgroup).
    dm_chunk_size : int
        Coarse-DM trials per gridder.compute call (memory knob).
    obs_dec_deg : float
        Observation declination (deg). With no cal + planar array this
        does not move a ``(0, 0)`` source off the image centre; it only
        scales the baseline projection.
    static_sky_disabled : bool
        Disable the static-sky EMA (default True for a clean image).
    sliding_window : bool
        F34 2-block sliding-window stage-1. Default False (single-block
        dedispersion; no one-block latency / no zero first block).
    antpos_seed : int
        Seed for the synthetic antenna layout.

    Returns
    -------
    dict
        ``{"truth": {...}, "plan": {...}, "per_chgroup": {cg: {...}}}``
        where each per-chgroup dict carries:

        * ``"cubes"``      : list of ``(N_DM, T_dedisp, N_filled)``
                             complex64 numpy arrays (one per processed
                             block; empty cubes for sliding-window
                             cold-start are skipped).
        * ``"cube_cat"``   : ``(N_DM, n_blocks*T_dedisp, N_filled)``
                             complex64 — blocks concatenated on the
                             time axis.
        * ``"ix_row"``,
          ``"ix_col"``     : ``(N_filled,)`` uint16 sparsity pattern
                             (the gridder's filled-cell row/col map).
        * ``"cell_lambda"``: float — per-cell uv-grid extent (lambda).
        * ``"n_filled"``   : int.
        * ``"n_fast_vis"`` : int — fast-vis tiles per block.
        * ``"t_dedisp"``   : int — dedispersed time-axis length.
        * ``"inject_log"`` : list of per-block injector log dicts.
    """
    dev = torch.device(device)
    dm_plan_path = str(dm_plan_path)
    plan: DMPlan = load_dm_plan(dm_plan_path)

    # Pin: the plan's cadence must match the requested fast cadence
    # (build_context raises on mismatch, but failing here is clearer).
    plan_native = plan.t_int_fast_native
    if abs(plan_native - t_int_fast_native) > 1e-6:
        raise ValueError(
            f"t_int_fast_native={t_int_fast_native} does not match DM "
            f"plan's t_int_fast_native={plan_native} (t_int_fast_us="
            f"{plan.t_int_fast_us}). For this plan pass "
            f"t_int_fast_native={int(round(plan_native))}."
        )

    antpos_e, antpos_n = synth_antpos(seed=antpos_seed)
    core_mask = _build_core_baseline_mask(n_core=82)

    coarse_dm = np.asarray(plan.dm_pc_cc, dtype=np.float64)
    owner_idx = int(np.argmin(np.abs(coarse_dm - cell.dm_pc_cm3)))

    out: dict = {
        "truth": {
            "l_rad": float(cell.l_rad),
            "m_rad": float(cell.m_rad),
            "dm_pc_cm3": float(cell.dm_pc_cm3),
            "width_ms": float(cell.width_ms),
            "width_samples": int(cell.width_samples),
            "fluence_jy_ms": float(cell.fluence_jy_ms),
            "profile": cell.profile,
            "peak_specnum": int(cell.peak_specnum),
            "peak_native": 2 * int(cell.peak_specnum),
        },
        "plan": {
            "path": dm_plan_path,
            "coarse_dm": coarse_dm.tolist(),
            "n_coarse": int(plan.n_coarse),
            "owner_idx": owner_idx,
            "owner_dm": float(coarse_dm[owner_idx]),
            "t_int_fast_us": float(plan.t_int_fast_us),
            "t_int_fast_native": float(plan_native),
        },
        "config": {
            "n_grid": n_grid,
            "n_blocks": n_blocks,
            "t_int_fast_native": t_int_fast_native,
            "chan_sum_factor": chan_sum_factor,
            "dm_chunk_size": dm_chunk_size,
            "sliding_window": sliding_window,
            "static_sky_disabled": static_sky_disabled,
            "obs_dec_deg": obs_dec_deg,
            "device": str(dev),
        },
        "per_chgroup": {},
    }

    inj_cfg = cell.to_injection_config()

    for cg in chgroups:
        cfg = FastIntegrationConfig(
            chgroup=int(cg),
            obs_dec_rad=math.radians(obs_dec_deg),
            n_grid=int(n_grid),
            kernel_support=1,
            cell_lambda_mode="common",
            t_int_fast_native=int(t_int_fast_native),
            cal_path=None,
            rfi_enabled=False,
            static_sky_disabled=bool(static_sky_disabled),
            static_sky_warmup_cubes=0,
            dm_plan_path=Path(dm_plan_path),
            chan_sum_factor=int(chan_sum_factor),
            dm_chunk_size=int(dm_chunk_size),
            sliding_window=bool(sliding_window),
            inject_configs=(inj_cfg,),
        )

        ctx = build_context(
            cfg,
            device=dev,
            antpos_e=antpos_e,
            antpos_n=antpos_n,
            is_core_baseline_mask=core_mask,
        )
        if ctx.multi_dm_coarse_dm is None:
            raise RuntimeError(
                "multi-DM coarse-DM stage was not constructed; check "
                "dm_plan_path + t_int_fast_native pin."
            )

        pattern = ctx.gridder.pattern
        n_fast_vis = int(ctx.kernel.n_fast_vis_per_full_block)

        cubes: list[np.ndarray] = []
        inject_logs: list[dict] = []
        for block_n in range(1, n_blocks + 1):
            raw = _pure_noise_block(seed, int(cg), block_n)
            res: IntegrationOutput = process_block(raw, ctx=ctx, block_n=block_n)
            g = res.gridded_minus_sky
            if g is None:
                continue
            cube = g.detach().to("cpu").numpy()
            # Skip the sliding-window cold-start all-zero cube.
            if sliding_window and not np.any(cube):
                pass
            cubes.append(cube)
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        if not cubes:
            raise RuntimeError(f"chgroup {cg}: no cubes produced")

        cube_cat = np.concatenate(cubes, axis=1)  # (N_DM, sum T_dedisp, N_filled)
        t_dedisp = int(cubes[0].shape[1])

        out["per_chgroup"][int(cg)] = {
            "cubes": cubes,
            "cube_cat": cube_cat,
            "ix_row": np.asarray(pattern.ix_row),
            "ix_col": np.asarray(pattern.ix_col),
            "cell_lambda": float(pattern.cell_lambda),
            "n_filled": int(pattern.n_filled),
            "n_fast_vis": n_fast_vis,
            "t_dedisp": t_dedisp,
            "n_dm": int(cube_cat.shape[0]),
            "inject_log": inject_logs,
        }

    return out


# ---------------------------------------------------------------------------
# Image-plane analysis helpers
# ---------------------------------------------------------------------------


def _peak_tile_dirty_image(
    cube_cat: np.ndarray,
    dm_idx: int,
    ix_row: np.ndarray,
    ix_col: np.ndarray,
    n_grid: int,
    *,
    device: torch.device | str = "cpu",
    subtract_time_mean: bool = True,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Dirty image of the max-power time tile for one coarse-DM trial.

    Returns ``(image, peak_tile_idx, per_tile_power)`` where ``image``
    is ``(n_grid, n_grid)`` float32 and ``per_tile_power`` is the
    ``sum |grid|^2`` over filled cells per time tile.

    With ``subtract_time_mean`` (the default) the per-cell temporal
    mean is removed before imaging — this is the bench-side analogue of
    the production static-sky EMA. It is REQUIRED here because a zenith
    (l=m=0) source sits on top of the noise / autocorrelation DC bias,
    which also lands at the image-centre pixel; removing the temporal
    mean cancels that constant DC and leaves only the transient pulse,
    so the coarse-DM trial that best dedisperses the pulse shows the
    largest coherent residual.
    """
    dev = torch.device(device)
    sparse = torch.from_numpy(cube_cat[dm_idx]).to(dev)        # (T, N_filled) cplx
    if subtract_time_mean:
        sparse = sparse - sparse.mean(dim=0, keepdim=True)
    per_tile_power = (sparse.abs() ** 2).sum(dim=1)            # (T,)
    peak_tile = int(torch.argmax(per_tile_power).item())
    dense = sparse_to_dense_grid(
        sparse[peak_tile : peak_tile + 1], ix_row, ix_col, n_grid,
    )                                                          # (1, G, G)
    img = dirty_image_from_dense_grid(dense)[0]                # (G, G) f32
    return (
        img.detach().cpu().numpy(),
        peak_tile,
        per_tile_power.detach().cpu().numpy(),
    )


def _image_peak_and_snr(image: np.ndarray) -> tuple[int, int, float, float]:
    """Return ``(row, col, peak_value, snr)`` for a dirty image.

    SNR = ``(peak - median) / std`` where the off-source statistics
    exclude an 11x11 box around the peak so the source itself does not
    inflate the noise estimate.
    """
    flat = int(np.argmax(image))
    row, col = np.unravel_index(flat, image.shape)
    peak = float(image[row, col])

    mask = np.ones_like(image, dtype=bool)
    r0, r1 = max(0, row - 5), min(image.shape[0], row + 6)
    c0, c1 = max(0, col - 5), min(image.shape[1], col + 6)
    mask[r0:r1, c0:c1] = False
    off = image[mask]
    med = float(np.median(off))
    std = float(np.std(off))
    snr = (peak - med) / std if std > 0 else float("inf")
    return int(row), int(col), peak, float(snr)


def analyse_chgroup(
    result: dict,
    chgroup: int,
    *,
    device: torch.device | str = "cpu",
) -> dict:
    """Produce the verification numbers for one chgroup.

    Returns a dict with the recovered peak (row, col, l, m), SNR, the
    per-coarse-DM peak-amplitude table, and the owner-DM dirty image.
    """
    cg = result["per_chgroup"][chgroup]
    cube_cat = cg["cube_cat"]
    ix_row = cg["ix_row"]
    ix_col = cg["ix_col"]
    n_grid = result["config"]["n_grid"]
    cell_lambda = cg["cell_lambda"]
    owner_idx = result["plan"]["owner_idx"]
    coarse_dm = result["plan"]["coarse_dm"]
    n_dm = cube_cat.shape[0]

    # Per-coarse-DM peak amplitude (image-plane, peak time tile).
    dm_table = []
    for c in range(n_dm):
        img_c, tile_c, _ = _peak_tile_dirty_image(
            cube_cat, c, ix_row, ix_col, n_grid, device=device,
        )
        row_c, col_c, peak_c, snr_c = _image_peak_and_snr(img_c)
        dm_table.append(
            {
                "dm_idx": c,
                "dm_pc_cm3": float(coarse_dm[c]),
                "peak_tile": tile_c,
                "peak_row": row_c,
                "peak_col": col_c,
                "peak_value": peak_c,
                "snr": snr_c,
            }
        )

    # Owner-DM image for the position / SNR check.
    img, peak_tile, per_tile_power = _peak_tile_dirty_image(
        cube_cat, owner_idx, ix_row, ix_col, n_grid, device=device,
    )
    row, col, peak, snr = _image_peak_and_snr(img)
    l_rec, m_rec = pixel_to_lm_radians(
        row, col, n_grid=n_grid, cell_lambda=cell_lambda,
    )

    half = n_grid // 2
    d_cells = math.hypot(row - half, col - half)

    # Centre-pixel time series for the owner DM (mean-subtracted) — the
    # dedispersed point-source light curve at zenith. Used to read off
    # the recovered pulse width / arrival tile.
    centre_ts = _centre_pixel_timeseries(
        cube_cat, owner_idx, ix_row, ix_col, n_grid, device=device,
    )

    return {
        "chgroup": chgroup,
        "owner_idx": owner_idx,
        "owner_dm": float(coarse_dm[owner_idx]),
        "peak_tile": peak_tile,
        "peak_row": row,
        "peak_col": col,
        "center_pixel": half,
        "dist_from_center_cells": float(d_cells),
        "l_recovered": float(l_rec),
        "m_recovered": float(m_rec),
        "peak_value": peak,
        "snr": snr,
        "per_tile_power": per_tile_power,
        "dm_table": dm_table,
        "image": img,
        "cell_lambda": cell_lambda,
        "centre_ts": centre_ts,
    }


def _centre_pixel_timeseries(
    cube_cat: np.ndarray,
    dm_idx: int,
    ix_row: np.ndarray,
    ix_col: np.ndarray,
    n_grid: int,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Centre-pixel (l=m=0) dirty-image value per time tile (mean-sub)."""
    dev = torch.device(device)
    sparse = torch.from_numpy(cube_cat[dm_idx]).to(dev)        # (T, N_filled)
    sparse = sparse - sparse.mean(dim=0, keepdim=True)
    dense = sparse_to_dense_grid(sparse, ix_row, ix_col, n_grid)   # (T, G, G)
    img = dirty_image_from_dense_grid(dense)                       # (T, G, G)
    half = n_grid // 2
    return img[:, half, half].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------


def save_dirty_png(analysis: dict, truth: dict, png_path: str | Path) -> Optional[str]:
    """Save the owner-DM dirty image PNG with injected/recovered marks."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        np.save(str(png_path).replace(".png", ".npy"), analysis["image"])
        return None

    img = analysis["image"]
    n_grid = img.shape[0]
    half = n_grid // 2
    cell_lambda = analysis["cell_lambda"]
    pixel_size_lm = 1.0 / (n_grid * cell_lambda)
    extent_lm = half * pixel_size_lm
    extent = (-extent_lm, +extent_lm, -extent_lm, +extent_lm)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = float(np.percentile(np.abs(img), 99.8))
    im = ax.imshow(
        img, origin="lower", extent=extent, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="dirty-image flux (a.u.)")
    ax.plot(
        truth["l_rad"], truth["m_rad"], "o", mec="lime", mfc="none",
        markersize=16, mew=2,
        label=f"injected ({truth['l_rad']:+.4f}, {truth['m_rad']:+.4f})",
    )
    ax.plot(
        analysis["l_recovered"], analysis["m_recovered"], "+", color="k",
        markersize=16, mew=2,
        label=(
            f"recovered ({analysis['l_recovered']:+.4f}, "
            f"{analysis['m_recovered']:+.4f})"
        ),
    )
    ax.set_xlabel("l (rad, east+)")
    ax.set_ylabel("m (rad, north+)")
    ax.set_title(
        f"corr-fast injection — chgroup {analysis['chgroup']}, "
        f"coarse-DM idx {analysis['owner_idx']} "
        f"({analysis['owner_dm']:.0f} pc/cc), SNR={analysis['snr']:.1f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=120)
    plt.close(fig)
    return str(png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l", type=float, default=0.0, dest="l_rad")
    p.add_argument("--m", type=float, default=0.0, dest="m_rad")
    p.add_argument("--dm", type=float, default=500.0, dest="dm")
    p.add_argument("--width-ms", type=float, default=4.0, dest="width_ms")
    p.add_argument("--fluence", type=float, default=50000.0, dest="fluence",
                   help="Jy.ms. Pure-noise voltages need a large fluence for "
                        "the transient to clearly beat the noise after DC "
                        "(static-sky) removal; 5e4 gives SNR~90 at N_grid=256.")
    p.add_argument("--profile", type=str, default="gaussian",
                   choices=("gaussian", "boxcar"))
    p.add_argument("--n-grid", type=int, default=256, dest="n_grid")
    p.add_argument("--n-blocks", type=int, default=6, dest="n_blocks")
    p.add_argument("--t-int-fast-native", type=int, default=32,
                   dest="t_int_fast_native")
    p.add_argument("--chgroup", type=int, default=0, dest="chgroup")
    p.add_argument("--peak-block", type=int, default=3, dest="peak_block",
                   help="1-indexed block the top-channel pulse peak lands in.")
    p.add_argument("--peak-bin", type=int, default=40, dest="peak_bin",
                   help="fast-vis bin within the peak block for the peak.")
    p.add_argument("--chan-sum-factor", type=int, default=1,
                   dest="chan_sum_factor")
    p.add_argument("--dm-chunk-size", type=int, default=2, dest="dm_chunk_size")
    p.add_argument("--obs-dec-deg", type=float, default=37.23, dest="obs_dec_deg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dm-plan", type=str, default=DEFAULT_DM_PLAN_PATH,
                   dest="dm_plan_path")
    p.add_argument("--png", type=str, default="/tmp/inject_corrfast_verify.png")
    return p


def _default_peak_specnum(peak_block: int, peak_bin: int,
                          t_int_fast_native: int) -> int:
    """Compute apply_at_specnum so the top-channel peak lands at
    ``peak_bin`` of the (1-indexed) ``peak_block``.

    Block ``b`` covers native ``[b * NATIVE_SAMPLES_PER_BLOCK, ...)``;
    bin ``k`` is ``k * t_int_fast_native`` native samples into the
    block. ``peak_native = 2 * apply_at_specnum``.
    """
    peak_native = (
        peak_block * NATIVE_SAMPLES_PER_BLOCK
        + peak_bin * t_int_fast_native
    )
    return peak_native // 2


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    peak_specnum = _default_peak_specnum(
        args.peak_block, args.peak_bin, args.t_int_fast_native,
    )
    cell = InjectionCell(
        l_rad=args.l_rad,
        m_rad=args.m_rad,
        dm_pc_cm3=args.dm,
        width_ms=args.width_ms,
        fluence_jy_ms=args.fluence,
        peak_specnum=peak_specnum,
        profile=args.profile,
        inj_id="verify_zenith_dm500",
    )

    print("=" * 72, flush=True)
    print("corr-fast injection verification", flush=True)
    print("=" * 72, flush=True)
    print(
        f"injected: l={cell.l_rad:+.5f} m={cell.m_rad:+.5f} "
        f"dm={cell.dm_pc_cm3:.1f} width={cell.width_ms:.2f}ms "
        f"(={cell.width_samples} native) fluence={cell.fluence_jy_ms:.1f} "
        f"profile={cell.profile}",
        flush=True,
    )
    print(
        f"peak_specnum={cell.peak_specnum} (peak_native="
        f"{2 * cell.peak_specnum}); peak lands in block {args.peak_block} "
        f"bin {args.peak_bin}",
        flush=True,
    )

    result = run_corr_fast_injection(
        cell,
        n_grid=args.n_grid,
        n_blocks=args.n_blocks,
        t_int_fast_native=args.t_int_fast_native,
        chgroups=(args.chgroup,),
        device=args.device,
        seed=args.seed,
        dm_plan_path=args.dm_plan_path,
        chan_sum_factor=args.chan_sum_factor,
        dm_chunk_size=args.dm_chunk_size,
        obs_dec_deg=args.obs_dec_deg,
    )

    cg = result["per_chgroup"][args.chgroup]
    print(
        f"\ncoarse-DM cube shape per block: "
        f"(N_DM={cg['n_dm']}, T_dedisp={cg['t_dedisp']}, "
        f"N_filled={cg['n_filled']})  "
        f"n_fast_vis/block={cg['n_fast_vis']}  "
        f"cube_cat shape={cg['cube_cat'].shape}",
        flush=True,
    )

    analysis = analyse_chgroup(result, args.chgroup, device="cpu")

    print("\n--- recovered source (owner coarse-DM idx "
          f"{analysis['owner_idx']}, {analysis['owner_dm']:.1f} pc/cc) ---",
          flush=True)
    print(
        f"peak pixel (row, col) = ({analysis['peak_row']}, "
        f"{analysis['peak_col']})   center pixel = {analysis['center_pixel']}",
        flush=True,
    )
    print(
        f"distance from center  = {analysis['dist_from_center_cells']:.2f} cells",
        flush=True,
    )
    print(
        f"recovered (l, m)      = ({analysis['l_recovered']:+.6f}, "
        f"{analysis['m_recovered']:+.6f}) rad   "
        f"injected = ({cell.l_rad:+.6f}, {cell.m_rad:+.6f}) rad",
        flush=True,
    )
    print(
        f"peak value            = {analysis['peak_value']:.4g}   "
        f"image SNR = {analysis['snr']:.2f}   "
        f"peak time tile = {analysis['peak_tile']}",
        flush=True,
    )

    print("\n--- per-coarse-DM peak-amplitude table ---", flush=True)
    print(
        f"{'idx':>3} {'DM(pc/cc)':>10} {'tile':>5} {'row':>4} {'col':>4} "
        f"{'peak':>12} {'snr':>8}",
        flush=True,
    )
    owner = analysis["owner_idx"]
    for r in analysis["dm_table"]:
        marker = " <-- owner" if r["dm_idx"] == owner else ""
        print(
            f"{r['dm_idx']:>3} {r['dm_pc_cm3']:>10.1f} {r['peak_tile']:>5} "
            f"{r['peak_row']:>4} {r['peak_col']:>4} {r['peak_value']:>12.4g} "
            f"{r['snr']:>8.2f}{marker}",
            flush=True,
        )

    # Pulse light-curve (owner DM, centre pixel, mean-subtracted) around
    # the arrival tile — confirms the recovered width.
    ts = analysis["centre_ts"]
    pk = int(np.argmax(ts))
    lo, hi = max(0, pk - 5), min(len(ts), pk + 6)
    bin_us = args.t_int_fast_native * NATIVE_SAMPLE_US
    print(
        f"\n--- owner-DM centre-pixel light curve (tile bin = "
        f"{bin_us:.1f} us); peak tile {pk} ---",
        flush=True,
    )
    print("  tile :  " + " ".join(f"{t:>7d}" for t in range(lo, hi)), flush=True)
    print("  value:  " + " ".join(f"{ts[t]:>7.3f}" for t in range(lo, hi)),
          flush=True)
    above = np.where(ts > 0.5 * ts[pk])[0]
    if above.size:
        fwhm_tiles = int(above.max() - above.min()) + 1
        print(
            f"  approx FWHM = {fwhm_tiles} tiles (~{fwhm_tiles * bin_us / 1e3:.2f} "
            f"ms); injected width = {cell.width_ms:.2f} ms",
            flush=True,
        )

    png = save_dirty_png(analysis, result["truth"], args.png)
    if png:
        print(f"\ndirty image PNG -> {png}", flush=True)
    else:
        print("\nmatplotlib unavailable; saved .npy instead", flush=True)

    # Verification banner.
    pos_ok = analysis["dist_from_center_cells"] <= 4.0
    peaks = [r["peak_value"] for r in analysis["dm_table"]]
    dm_argmax = int(np.argmax(peaks))
    dm_ok = dm_argmax == owner
    print("\n--- verification ---", flush=True)
    print(f"  zenith within <=4 cells of center: {pos_ok} "
          f"({analysis['dist_from_center_cells']:.2f} cells)", flush=True)
    print(f"  strongest coarse-DM == owner idx {owner}: {dm_ok} "
          f"(argmax idx {dm_argmax})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
