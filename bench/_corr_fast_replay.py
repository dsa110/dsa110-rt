"""bench/_corr_fast_replay.py — in-memory voltage-fixture replay helper (M3 chunks 5+6).

This module is a **bench-internal helper** (underscore prefix) shared by:

* :mod:`bench.corr_fast_continuum_0319` — the chunk-5 0319+415 continuum
  imager (replays first N blocks per sb across all 16 chgroups).
* :mod:`bench.corr_fast_burst_250924mptq` — the chunk-6 250924mptq burst
  imager (replays the first ~8 blocks per sb at fast cadence).

It wraps the chunk-4 production orchestrator
:mod:`dsart.services.corr_fast_integration` and replays raw voltage
``.out`` files from the M2-validated dump tree:

    /home/ubuntu/data/voltages/<run_id>/voltages/<run_id>_sb<NN>_data.out
    /home/ubuntu/data/voltages/<run_id>/cals/beamformer_weights_sb<NN>*.dat

without touching PSRDADA. Each ``.out`` file is a contiguous sequence of
:data:`dsart.common.constants.FADA_BYTES_PER_BLOCK`-sized fada blocks
(15 blocks of 301,989,888 bytes each = 4,529,848,320 bytes = ~4.5 GB
per file).

The helper exposes three Class-A entry points (per
``PARALLEL_AGENTS.md`` §3 ownership tables):

* :func:`iterate_voltage_blocks` — mmap the file, yield FADA-sized chunks.
* :func:`replay_chgroup` — drive ``build_context`` + ``process_block``
  end-to-end, returning the per-block :class:`IntegrationOutput`s.
* :func:`accumulate_chgroup_grids` — sum gridded sparse-COO cubes
  across blocks per fast-vis tile (``(n_fv, N_filled)`` complex64).

Plus image-plane utilities used by both chunk-5 and chunk-6 benches:

* :func:`sparse_to_dense_grid` — scatter a sparse-COO ``(n_fv, N_filled)``
  cube into a dense ``(n_fv, N_grid, N_grid)`` complex grid.
* :func:`dirty_image_from_dense_grid` — ``Re(iFFT2(grid))`` per fast-vis
  tile, fft-shifted; returns ``(n_fv, N_grid, N_grid) float32``.
* :func:`pixel_to_lm_radians` / :func:`lm_to_pixel` — convert between
  cell coordinates and ``(l, m) ∈ rad`` for the predicted source overlay.

Sign convention notes (all inherited from upstream — this helper does
not introduce any new sign decisions):

* The gridder applies the F20 ``(u, v)`` negation internally (kernel.py
  line 276-277). After ``iFFT2`` + ``fftshift``, the (l, m) axes match
  the TMS Eq. 3.83 convention (peak at +l for east-of-zenith source,
  +m for north-of-zenith source).
* The cal-loader applies the F21 DEC-only fringe-stop phase fold so that
  a source at (HA=0, dec=obs_dec) lands at ``(l, m) ≈ (0, 0)`` regardless
  of declination (subject to within-chgroup smearing and the
  lambda-uniform gridder's per-chgroup cell-size variation).
* Each chgroup has its OWN ``cell_lambda`` (derived from
  ``max_baseline_lambda`` at that chgroup's frequencies), so the
  ``(N_grid, N_grid)`` images from different chgroups have slightly
  different ``(l, m)`` per-pixel scales. Per chunk-5 brief, dirty
  images are summed pixel-wise across chgroups (acceptable for the
  ≤4-cell tolerance criterion); chunk 7 will properly resample
  per-chgroup images onto a common ``(l, m)`` grid.

This helper is **CPU- and GPU-portable**: pass ``device=torch.device("cpu")``
for the synthetic tests in :mod:`tests.test_voltage_fixture_replay`, and
``device=torch.device("cuda:0")`` for the real benches on h01.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dsart.cal.cal_loader import CalMode
from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NCHAN_PER_CHGROUP,
    SPEED_OF_LIGHT_M_S,
    freq_GHz,
)
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    IntegrationContext,
    IntegrationOutput,
    NoOpCoarseDM,
    _build_core_baseline_mask,
    build_context,
    load_antpos_from_cal_blob,
    process_block,
)


__all__ = [
    "iterate_voltage_blocks",
    "replay_chgroup",
    "accumulate_chgroup_grids",
    "sparse_to_dense_grid",
    "dirty_image_from_dense_grid",
    "compute_chgroup_cell_lambda",
    "pixel_to_lm_radians",
    "lm_to_pixel",
]

LOG = logging.getLogger("bench._corr_fast_replay")


# ---------------------------------------------------------------------------
# Block iteration: mmap-friendly chunked reads
# ---------------------------------------------------------------------------


def iterate_voltage_blocks(
    voltage_path: Path,
    *,
    max_blocks: int | None = None,
    block_bytes: int = FADA_BYTES_PER_BLOCK,
    skip_blocks: int = 0,
) -> Iterator[bytes]:
    """Yield ``block_bytes``-sized chunks from a voltage ``.out`` file.

    Reads sequentially in 300 MB chunks; suitable for both the small
    synthetic-bytes tests (3-block files) and the full 15-block real
    fixture files (~4.5 GB).

    Args:
        voltage_path: path to a ``<run_id>_sb<NN>_data.out`` file.
        max_blocks: stop after yielding this many blocks. ``None`` =
            yield all blocks.
        block_bytes: bytes per block. Defaults to ``FADA_BYTES_PER_BLOCK``
            (= 301,989,888); override for tests with smaller synthetic
            blocks.
        skip_blocks: number of blocks to skip at the file head (used by
            the burst bench to seek to block N if N>0; chunk 6 reads
            from block 0 so default is 0).

    Yields:
        ``bytes`` of length exactly ``block_bytes``. The final partial
        block (if file length is not a multiple of ``block_bytes``) is
        DROPPED with a warning log line — this matches M2's
        ``voltage_fixture_slow_corr.py`` behaviour and the real
        DSA-110 dumper which always writes whole blocks.
    """
    voltage_path = Path(voltage_path)
    if not voltage_path.is_file():
        raise FileNotFoundError(f"voltage file not found: {voltage_path}")

    file_size = voltage_path.stat().st_size
    n_blocks_in_file = file_size // block_bytes
    n_partial = file_size - n_blocks_in_file * block_bytes
    if n_partial > 0:
        LOG.warning(
            "%s: %d trailing partial bytes (= %.3f blocks); will be dropped",
            voltage_path.name, n_partial, n_partial / block_bytes,
        )

    n_to_yield = n_blocks_in_file - skip_blocks
    if max_blocks is not None:
        n_to_yield = min(n_to_yield, max_blocks)
    if n_to_yield <= 0:
        LOG.warning(
            "%s: no blocks to yield (n_blocks_in_file=%d skip=%d max=%s)",
            voltage_path.name, n_blocks_in_file, skip_blocks, max_blocks,
        )
        return

    LOG.info(
        "iterate_voltage_blocks: %s file_size=%d (= %d blocks); "
        "skip=%d max=%s → yielding %d blocks",
        voltage_path.name, file_size, n_blocks_in_file,
        skip_blocks, max_blocks, n_to_yield,
    )

    with voltage_path.open("rb") as f:
        if skip_blocks > 0:
            f.seek(skip_blocks * block_bytes)
        for _ in range(n_to_yield):
            chunk = f.read(block_bytes)
            if len(chunk) != block_bytes:
                # Should never trip given file_size precheck, but
                # guards against concurrent truncation.
                LOG.warning(
                    "short read: got %d expected %d; stopping",
                    len(chunk), block_bytes,
                )
                return
            yield chunk


# ---------------------------------------------------------------------------
# replay_chgroup — drive build_context + process_block over N blocks
# ---------------------------------------------------------------------------


def replay_chgroup(
    voltage_path: Path,
    cal_path: Path | None,
    cfg: FastIntegrationConfig,
    *,
    max_blocks: int,
    device: torch.device,
    skip_blocks: int = 0,
    block_bytes: int = FADA_BYTES_PER_BLOCK,
    antpos_e: np.ndarray | None = None,
    antpos_n: np.ndarray | None = None,
    is_core_baseline_mask: np.ndarray | None = None,
) -> tuple[IntegrationContext, list[IntegrationOutput]]:
    """Replay one chgroup's voltages through the chunk-4 pipeline.

    Wraps ``build_context`` + a ``process_block`` loop. Returns the
    constructed context (so callers can inspect ``ctx.gridder.pattern``
    or ``ctx.kernel.n_fast_vis_per_full_block``) and the per-block
    :class:`IntegrationOutput`s.

    Args:
        voltage_path: path to ``<run_id>_sb<NN>_data.out``.
        cal_path: path to ``beamformer_weights_sb<NN>*.dat``. If
            ``None``, antpos must be supplied via ``antpos_e`` /
            ``antpos_n`` (synthetic-test path).
        cfg: :class:`FastIntegrationConfig`. ``cfg.chgroup`` and
            ``cfg.cal_path`` are honoured; this function does NOT
            mutate the cfg.
        max_blocks: replay this many fada blocks from the head of the
            file (after ``skip_blocks``). 0 → replay no blocks (returns
            empty list — useful for context-only construction in tests).
        device: where the compute runs.
        skip_blocks: skip this many blocks at file head before starting.
            Chunk-6 burst bench uses 0 (block 0 always); kept here for
            future flexibility.
        block_bytes: bytes per block. Defaults to FADA_BYTES_PER_BLOCK.
        antpos_e, antpos_n: ``(NANTS,)`` float32 antenna positions.
            If both ``cal_path`` AND these are supplied, the cal_path
            wins (provenance match for the gridder pattern). Only the
            no-cal synthetic-test path uses this override.
        is_core_baseline_mask: ``(NBASE,)`` bool mask. Defaults to the
            standard 82-core mask (same as
            :func:`dsart.services.corr_fast_integration.load_antpos_from_cal_blob`).

    Returns:
        ``(ctx, outputs)`` where ``len(outputs) <= max_blocks`` (may be
        fewer if the file is shorter than ``max_blocks * block_bytes``
        or if any blocks were dropped for size mismatch).
    """
    if cal_path is not None:
        ap_e, ap_n, core_mask = load_antpos_from_cal_blob(cal_path)
    else:
        if antpos_e is None or antpos_n is None:
            raise ValueError(
                "replay_chgroup: either cal_path OR (antpos_e, antpos_n) "
                "must be supplied; got neither"
            )
        ap_e = antpos_e
        ap_n = antpos_n
        core_mask = (
            is_core_baseline_mask
            if is_core_baseline_mask is not None
            else _build_core_baseline_mask(n_core=82)
        )

    ctx = build_context(
        cfg, device=device,
        antpos_e=ap_e, antpos_n=ap_n,
        is_core_baseline_mask=core_mask,
        coarse_dm=NoOpCoarseDM(),
    )

    outputs: list[IntegrationOutput] = []
    if max_blocks <= 0:
        return ctx, outputs

    for block_n, raw in enumerate(
        iterate_voltage_blocks(
            voltage_path,
            max_blocks=max_blocks,
            skip_blocks=skip_blocks,
            block_bytes=block_bytes,
        ),
        start=1,
    ):
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
        if raw_arr.nbytes != block_bytes:
            LOG.error(
                "block %d: got %d bytes, expected %d; skipping",
                block_n, raw_arr.nbytes, block_bytes,
            )
            continue
        out = process_block(raw_arr, ctx=ctx, block_n=block_n)
        outputs.append(out)
        LOG.debug(
            "replayed block %d/%d chgroup=%d gridded.shape=%s",
            block_n, max_blocks, cfg.chgroup,
            None if out.gridded_minus_sky is None
            else tuple(out.gridded_minus_sky.shape),
        )

    return ctx, outputs


# ---------------------------------------------------------------------------
# Per-block sparse-COO cube accumulation
# ---------------------------------------------------------------------------


def accumulate_chgroup_grids(
    outputs: list[IntegrationOutput],
    n_filled: int,
) -> torch.Tensor:
    """Concatenate per-block gridded cubes along the fast-vis-tile axis.

    Each :class:`IntegrationOutput.gridded_minus_sky` has shape
    ``(n_fv_per_block, n_filled)`` complex64. With ``B`` blocks the
    result has shape ``(B * n_fv_per_block, n_filled)``.

    Args:
        outputs: list of :class:`IntegrationOutput` returned by
            :func:`replay_chgroup`.
        n_filled: expected ``N_filled`` for shape sanity. Pulled from
            ``ctx.gridder.pattern.n_filled``.

    Returns:
        ``(n_fast_vis_total, n_filled) complex64`` torch tensor on the
        same device as the input outputs (typically GPU). Caller can
        ``.cpu()`` for downstream numpy work.

    Raises:
        ValueError: if any output has a None grid (block dropped) or
            shape mismatches ``n_filled``.
        ValueError: if ``outputs`` is empty.

    Notes:
        We CONCATENATE blocks, we do NOT sum them — the chunk-5
        continuum bench summing across BLOCKS happens in the bench
        downstream of this helper (after iFFT to image plane), and
        chunk-6 burst bench needs the full time series intact.
    """
    if not outputs:
        raise ValueError("accumulate_chgroup_grids: empty outputs list")
    grids: list[torch.Tensor] = []
    for i, out in enumerate(outputs):
        if out.gridded_minus_sky is None:
            raise ValueError(
                f"accumulate_chgroup_grids: outputs[{i}] has "
                f".gridded_minus_sky=None (block dropped?)"
            )
        g = out.gridded_minus_sky
        if g.ndim != 2:
            raise ValueError(
                f"accumulate_chgroup_grids: outputs[{i}].gridded_minus_sky "
                f"has shape {tuple(g.shape)}, expected 2D"
            )
        if g.shape[1] != n_filled:
            raise ValueError(
                f"accumulate_chgroup_grids: outputs[{i}].gridded_minus_sky "
                f"shape[1]={g.shape[1]} != n_filled={n_filled}"
            )
        grids.append(g)
    return torch.cat(grids, dim=0)                                   # (B*n_fv, n_filled)


# ---------------------------------------------------------------------------
# Sparse-COO cube → dense (l, m) image-plane utilities
# ---------------------------------------------------------------------------


def sparse_to_dense_grid(
    sparse: torch.Tensor,
    ix_row: np.ndarray,
    ix_col: np.ndarray,
    n_grid: int,
) -> torch.Tensor:
    """Scatter ``(n_fv, N_filled) complex`` sparse-COO into ``(n_fv, N_grid, N_grid) complex``.

    Mirrors the chunk-3a sparsity-pattern + gridder convention:
    ``sparse[fv, k]`` is the gridded visibility for grid cell
    ``(ix_row[k], ix_col[k])``. The dense grid has zeros everywhere
    except those filled cells.

    Args:
        sparse: ``(n_fv, N_filled)`` complex torch tensor.
        ix_row, ix_col: ``(N_filled,)`` uint16 (or int) numpy arrays —
            grid (row, col) of each filled cell. Pulled from
            :class:`SparsityPattern.ix_row` / ``.ix_col``.
        n_grid: side length of the dense grid (e.g. 256).

    Returns:
        ``(n_fv, N_grid, N_grid)`` complex tensor on the same device
        as ``sparse``. dtype = sparse.dtype.
    """
    if not sparse.is_complex():
        raise TypeError(
            f"sparse_to_dense_grid: sparse must be complex; got {sparse.dtype}"
        )
    if sparse.ndim != 2:
        raise ValueError(
            f"sparse_to_dense_grid: sparse must be 2D (n_fv, N_filled); "
            f"got {tuple(sparse.shape)}"
        )
    n_fv, n_filled = sparse.shape
    if ix_row.shape != (n_filled,) or ix_col.shape != (n_filled,):
        raise ValueError(
            f"sparse_to_dense_grid: ix_row.shape={ix_row.shape} or "
            f"ix_col.shape={ix_col.shape} != ({n_filled},)"
        )

    device = sparse.device
    dense = torch.zeros(
        (n_fv, n_grid, n_grid),
        dtype=sparse.dtype, device=device,
    )
    row_t = torch.from_numpy(np.asarray(ix_row, dtype=np.int64)).to(device)
    col_t = torch.from_numpy(np.asarray(ix_col, dtype=np.int64)).to(device)
    # Index assignment: dense[:, row_t, col_t] = sparse — broadcasts the
    # n_fv dim, gather along (row, col).
    dense[:, row_t, col_t] = sparse
    return dense


def dirty_image_from_dense_grid(dense: torch.Tensor) -> torch.Tensor:
    """Per-fast-vis-tile dirty image: ``Re(fftshift(iFFT2(ifftshift(grid))))``.

    Args:
        dense: ``(..., N_grid, N_grid)`` complex tensor (output of
            :func:`sparse_to_dense_grid`). Leading dims are looped over.

    Returns:
        Same leading shape, dtype float32, real-valued dirty image.

    Notes:
        Mirrors :func:`tools.viz.common.dirty_image_from_grid` (M2-validated)
        — uses ``ifftshift → ifft2 → fftshift``. The F20 ``(u, v)``
        negation is already applied INSIDE the gridder (kernel.py
        L276-277), so this function does NOT re-apply it.
    """
    grid_shifted = torch.fft.ifftshift(dense, dim=(-2, -1))
    img_complex = torch.fft.ifft2(grid_shifted, dim=(-2, -1))
    img = torch.fft.fftshift(img_complex, dim=(-2, -1))
    return img.real.to(torch.float32)


# ---------------------------------------------------------------------------
# Per-chgroup cell-size and (l, m) ↔ pixel
# ---------------------------------------------------------------------------


def compute_chgroup_cell_lambda(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    *,
    chgroup: int,
    n_grid: int,
    is_core_baseline_mask: np.ndarray | None = None,
) -> float:
    """Mirror :meth:`FastVisGridder.from_pattern` cell-size derivation.

    ``cell_lambda = max_baseline_lambda * 2 / n_grid``. Used by the
    benches to convert grid-pixel positions back to ``(l, m)`` in
    radians for the predicted-source overlay.

    Args:
        antpos_e, antpos_n: ``(NANTS,) float32`` antenna positions.
        chgroup: chgroup index 0..15.
        n_grid: gridder grid side length (default 256).
        is_core_baseline_mask: ``(NBASE,)`` bool mask. Default = standard
            82-core mask.

    Returns:
        ``cell_lambda`` in λ units (cycles per radian per cell — the
        pixel size of the dense (u, v) grid).
    """
    from dsart.grid.sparsity_pattern import _per_baseline_uv_meters

    if is_core_baseline_mask is None:
        is_core_baseline_mask = _build_core_baseline_mask(n_core=82)
    du_m, dv_m = _per_baseline_uv_meters(
        antpos_e, antpos_n,
        is_core_baseline_mask=is_core_baseline_mask,
    )
    nu_GHz = np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    wavelength_m = SPEED_OF_LIGHT_M_S / (nu_GHz * 1e9)
    u_lam = du_m[:, None] / wavelength_m[None, :]
    v_lam = dv_m[:, None] / wavelength_m[None, :]
    max_baseline_lambda = float(np.max(
        np.maximum(np.abs(u_lam), np.abs(v_lam))
    ))
    return max_baseline_lambda * 2.0 / n_grid


def pixel_to_lm_radians(
    ix_row: int | np.ndarray,
    ix_col: int | np.ndarray,
    *,
    n_grid: int,
    cell_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense-grid pixel coords to ``(l, m) ∈ rad``.

    Mirrors :func:`tools.viz.common.find_image_peaks` cell→lm math.

    The image cell size is ``1 / (n_grid * cell_lambda)`` rad. After
    fftshift the centre of the image (``ix_row = ix_col = n_grid // 2``)
    corresponds to ``(l, m) = (0, 0)``.

    Args:
        ix_row: pixel row(s) — int or array. ``ix_row`` indexes the
            v-axis (m). Convention matches :class:`SparsityPattern`.
        ix_col: pixel column(s). Indexes the u-axis (l).
        n_grid: image side length (cells).
        cell_lambda: per-cell λ-extent of the (u, v) grid (= output of
            :func:`compute_chgroup_cell_lambda`).

    Returns:
        ``(l_rad, m_rad)`` arrays / scalars matching ``ix_col`` / ``ix_row``.
    """
    half = n_grid // 2
    pixel_size_lm = 1.0 / (n_grid * cell_lambda)
    l_rad = (np.asarray(ix_col, dtype=np.float64) - half) * pixel_size_lm
    m_rad = (np.asarray(ix_row, dtype=np.float64) - half) * pixel_size_lm
    return l_rad, m_rad


def lm_to_pixel(
    l_rad: float | np.ndarray,
    m_rad: float | np.ndarray,
    *,
    n_grid: int,
    cell_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pixel_to_lm_radians`.

    Returns ``(ix_row, ix_col)`` of the pixel containing ``(l, m)`` after
    nearest-cell rounding. Mirrors the gridder's
    ``np.rint(u_lam / cell_lambda).astype(int) + n_grid//2`` pipeline.

    Out-of-grid (l, m) values clamp to ``[0, n_grid - 1]``; callers
    should sanity-check the returned indices against the grid extent.
    """
    half = n_grid // 2
    pixel_size_lm = 1.0 / (n_grid * cell_lambda)
    ix_col = np.rint(np.asarray(l_rad, dtype=np.float64) / pixel_size_lm).astype(np.int64) + half
    ix_row = np.rint(np.asarray(m_rad, dtype=np.float64) / pixel_size_lm).astype(np.int64) + half
    ix_col = np.clip(ix_col, 0, n_grid - 1)
    ix_row = np.clip(ix_row, 0, n_grid - 1)
    return ix_row, ix_col


# ---------------------------------------------------------------------------
# Default cfg builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayDefaults:
    """Sensible chunk-5/6 defaults; benches pick which to use."""

    chgroup: int
    obs_dec_deg: float
    n_grid: int = 256
    kernel_support: int = 1
    t_int_fast_native: int = 8                                       # production cadence
    cal_mode: str = CalMode.PHASE_ONLY
    cal_pol_swap: bool = False
    rfi_enabled: bool = False                                        # benches keep RFI off by default; brief covers
    static_sky_disabled: bool = False
    static_sky_alpha: float = 0.001
    static_sky_warmup_cubes: int = 8

    def to_cfg(
        self,
        *,
        cal_path: Optional[Path],
        flagants_path: Optional[Path] = None,
    ) -> FastIntegrationConfig:
        return FastIntegrationConfig(
            chgroup=self.chgroup,
            obs_dec_rad=math.radians(self.obs_dec_deg),
            n_grid=self.n_grid,
            kernel_support=self.kernel_support,
            t_int_fast_native=self.t_int_fast_native,
            cal_path=cal_path,
            cal_mode=self.cal_mode,
            cal_pol_swap=self.cal_pol_swap,
            flagants_path=flagants_path,
            rfi_enabled=self.rfi_enabled,
            static_sky_alpha=self.static_sky_alpha,
            static_sky_warmup_cubes=self.static_sky_warmup_cubes,
            static_sky_disabled=self.static_sky_disabled,
        )
