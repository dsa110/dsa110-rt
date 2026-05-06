"""Cube pipeline: ``CubeRingSlot`` → fine-DM combiner → 2D iFFT imager →
Layer-1 σ-clipped per-fdm normalisation → ``Detector.forward()`` →
``[Candidate]`` (M5 Chunk 6b-α).

This is the single-cube data-path used by ``services/search_compute.py``
and ``bench/search_node_throughput.py``. It is deliberately
side-effect-free: it does NOT touch the trigger emitter or holdoff
state — those live in the service's per-cube post-processing step
(``services/search_compute.py::_post_detect``).

Numerical contract:
    cube ∈ ℝ        (real fp16 image cube; single-side identity §3.6.11)
    sigma_layer1 ∈ ℝ⁺ per fine-DM trial; broadcast-divides the cube
                     before ``Detector.forward()``.
    validity_mask ∈ {True, False}^{T_det, N_fdm}; passes through to
                     the detector for §D14 Layer-2 EMA gating.

Stage timing (production, plan §8 lines 2316-2317):
    combiner (sparse-scatter) ............... 2-3 ms
    imager (2D iFFT + edge mask) ............ 1-2 ms
    Layer-1 σ-clip + normalise ............... 0.5 ms
    Detector.forward (128 kernels) ........... 18-22 ms
    Decoder + merger ........................ 1-2 ms
    Holdoff + emitter (per cube) ............. <1 ms
    -------------------------------------------------
    end-to-end per-cube budget (search_node_throughput) .... ~30 ms
    cube cadence at default ops ............................ 134 ms

The chunk-6b-α path runs the combiner + imager on CPU (numpy) and the
detector on whatever device the caller's ``DeterministicDetector`` is
configured for. The chunk-6b production hardening pass moves the
combiner + imager onto GPU + plumbs a cuFFT plan cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..common.contracts import Candidate  # noqa: F401  (re-exported via result)
from ..detector.forward import DeterministicDetector
from ..fine_dm.combiner import combine_chgroups
from ..image.imager import (
    apply_edge_mask,
    compute_edge_mask,
    dirty_image_from_uv_grid,
)
from ..noise_norm.layer1 import Layer1State
from .rx_ring import CubeRingSlot

__all__ = [
    "CubePipeline",
    "CubePipelineConfig",
    "CubePipelineResult",
]


# ---------------------------------------------------------------------------
# Config + per-cube result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubePipelineConfig:
    """Static (per-service-lifetime) configuration for the pipeline.

    Args:
        n_grid: spatial grid side; must match ``CubeRingSlot.n_grid``.
        edge_mask_kernel_support: gridding-kernel support (cells)
            used to size the edge mask. Default 5 per plan §3.6.5.
        edge_mask_sigma_l_pix: image-plane envelope σ in pixels;
            ``None`` defers to kernel-support floor only.
        edge_mask_envelope_threshold: -3 dB cutoff on the envelope.
        device: torch device for the detector + cube tensor. Default
            CPU; production uses ``cuda:1`` (D3 h01 isolation).
        cube_dtype: torch dtype for the assembled cube fed to the
            detector. Plan §3.6.11 pins ``fp16``; benches that need
            precise debug output may pass ``fp32``.
    """

    n_grid: int
    edge_mask_kernel_support: int = 5
    edge_mask_sigma_l_pix: Optional[float] = None
    edge_mask_envelope_threshold: float = 0.5
    device: str = "cpu"
    cube_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.n_grid <= 0 or self.n_grid & (self.n_grid - 1):
            raise ValueError(
                f"n_grid={self.n_grid}, expected positive power of two"
            )


@dataclass(frozen=True, slots=True)
class CubePipelineResult:
    """Per-cube pipeline output.

    Holds the assembled (post-imager, post-Layer-1) cube so benches
    (``bench/noise_norm_calibration.py``) can drive the FAR check off
    the same tensor the detector saw, without re-running the imager.

    Args:
        cube_id: ``CubeRingSlot.cube_id``.
        specnum_start: ``CubeRingSlot.specnum_start``.
        cube: ``[T_det, N_fdm, N_grid, N_grid]`` real cube post
            Layer-1 normalisation, in ``CubePipelineConfig.cube_dtype``.
        sigma_layer1: ``[N_fdm] float32`` per-fdm Layer-1 σ used to
            normalise the cube.
        validity_mask: ``[T_det, N_fdm] bool`` mask passed to the
            detector.
        candidates: list of ``Candidate``s emitted by the detector
            (post-decoder, post-merger).
        stage_timings_ns: per-stage wall-clock duration (ns) of the
            most recent ``process()`` call. Keys are
            ``{"build_cube", "layer1_norm", "detector_forward",
            "total"}``. Used by ``bench/search_node_throughput.py`` to
            build the per-stage histogram.
    """

    cube_id: int
    specnum_start: int
    cube: torch.Tensor
    sigma_layer1: torch.Tensor
    validity_mask: torch.Tensor
    candidates: List[Candidate]
    stage_timings_ns: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cube pipeline
# ---------------------------------------------------------------------------


class CubePipeline:
    """Stateful per-service cube pipeline.

    Holds the cached edge-mask + ``Layer1State`` (needed across cubes
    for the burn-in median) and the detector reference. Each call to
    ``process(slot)`` runs the full cube-time pipeline once.

    Construct once at service start; call ``process()`` per cube. Not
    thread-safe; one cube at a time per pipeline instance (the service
    can pipeline multiple cubes by holding multiple ``CubePipeline``
    instances bound to different GPU streams, but that's chunk-6b-prod).
    """

    def __init__(
        self,
        config: CubePipelineConfig,
        detector: DeterministicDetector,
        layer1_state: Optional[Layer1State] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.layer1_state = layer1_state
        self._device = torch.device(config.device)
        # Edge mask is constant per ops point; cache once on the device.
        mask_np = compute_edge_mask(
            n_grid=config.n_grid,
            kernel_support=config.edge_mask_kernel_support,
            sigma_l_pix=config.edge_mask_sigma_l_pix,
            envelope_threshold=config.edge_mask_envelope_threshold,
        )
        self._edge_mask = torch.from_numpy(mask_np).to(
            device=self._device, dtype=torch.float32
        )

    @property
    def edge_mask(self) -> torch.Tensor:
        return self._edge_mask

    def _build_cube(
        self, slot: CubeRingSlot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run combiner + imager + edge mask. Returns (cube, validity_mask)
        on ``self._device`` in ``self.config.cube_dtype`` for the cube
        and ``torch.bool`` for the validity mask.
        """
        if slot.n_grid != self.config.n_grid:
            raise ValueError(
                f"slot.n_grid={slot.n_grid} != pipeline n_grid="
                f"{self.config.n_grid}"
            )
        n_fdm = slot.n_fdm_in_cube
        n_grid = slot.n_grid
        t_det = slot.t_det
        # Image cube starts as fp32 and is downcast at the end so the
        # iFFT2 + masking are numerically stable.
        cube_np = np.empty(
            (t_det, n_fdm, n_grid, n_grid), dtype=np.float32
        )
        # Per-fine-DM combine + image. The shifts table covers all
        # fine-DM trials in the slot.
        for f in range(n_fdm):
            uv_slab = combine_chgroups(
                per_chgroup_streams=slot.per_chgroup_streams,
                time_shift_per_chgroup=slot.time_shift_table.shifts[f, :],
                t_window=(0, t_det),
                n_grid=n_grid,
            )
            img_slab = dirty_image_from_uv_grid(uv_slab)
            cube_np[:, f, :, :] = img_slab.astype(np.float32, copy=False)
        cube = torch.from_numpy(cube_np).to(self._device)
        cube = apply_edge_mask(cube, self._edge_mask)
        if cube.dtype != self.config.cube_dtype:
            cube = cube.to(self.config.cube_dtype)
        validity_mask = torch.from_numpy(
            np.ascontiguousarray(slot.validity_mask)
        ).to(device=self._device, dtype=torch.bool)
        return cube, validity_mask

    def _layer1_normalise(
        self,
        cube: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-fdm Layer-1 σ + return (cube_normalised, sigma).

        Uses the stateful ``Layer1State`` if present (production path:
        running burn-in across cubes); falls back to per-cube
        σ-clipped scalar if ``layer1_state is None`` (test/bench path
        that wants Layer-1 to be a no-op).
        """
        if self.layer1_state is None:
            # Compute σ but do NOT broadcast-divide — the caller is in
            # a unit-σ test setting (CubeInjectionConfig D8) and wants
            # the detector to see the cube as-is, post-edge-mask.
            n_fdm = cube.shape[1]
            sigma = torch.ones(
                (n_fdm,), dtype=torch.float32, device=cube.device
            )
            return cube, sigma
        sigma = self.layer1_state.update_and_query(cube=cube.to(torch.float32))
        if sigma.device != cube.device:
            sigma = sigma.to(cube.device)
        # Broadcast-divide. cube is [T_det, N_fdm, H, W]; sigma is [N_fdm].
        cube_normalised = cube / sigma[None, :, None, None].to(cube.dtype)
        return cube_normalised, sigma

    def process(self, slot: CubeRingSlot) -> CubePipelineResult:
        """Run the full per-cube pipeline. Synchronous (the I/O wait
        happens in the RxRing source's async iterator before the call).

        The detector's ``forward()`` accepts ``event_specnum`` as a
        kwarg and rebases ``Candidate.event_specnum`` to absolute
        specnums internally; we pass ``slot.specnum_start`` so
        downstream emitter sees absolute values.

        Stage-level wall-clock timings are captured into
        ``CubePipelineResult.stage_timings_ns`` for the throughput
        bench. ``time.perf_counter_ns`` is monotonic and high-resolution
        on linux; the stages are sequential so the timings sum to
        ``total`` modulo a few ns of bookkeeping.
        """
        t0 = time.perf_counter_ns()
        cube, validity_mask = self._build_cube(slot)
        t1 = time.perf_counter_ns()
        cube_norm, sigma_layer1 = self._layer1_normalise(cube)
        t2 = time.perf_counter_ns()
        with torch.no_grad():
            cands = self.detector.forward(
                cube_norm,
                validity_mask,
                sigma_layer1,
                event_specnum=int(slot.specnum_start),
            )
        t3 = time.perf_counter_ns()
        timings = {
            "build_cube": t1 - t0,
            "layer1_norm": t2 - t1,
            "detector_forward": t3 - t2,
            "total": t3 - t0,
        }
        return CubePipelineResult(
            cube_id=slot.cube_id,
            specnum_start=slot.specnum_start,
            cube=cube_norm,
            sigma_layer1=sigma_layer1,
            validity_mask=validity_mask,
            candidates=cands,
            stage_timings_ns=timings,
        )
