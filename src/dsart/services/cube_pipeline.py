"""Cube pipeline: ``CubeRingSlot`` → fine-DM combiner → 2D iFFT imager →
Layer-1 σ-clipped per-fdm normalisation → ``Detector.forward()`` →
``[Candidate]`` (M5 Chunk 6b-α + Chunk 8 GPU wiring).

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

Two image backends are supported (``CubePipelineConfig.image_backend``):

  * ``"cpu"`` — chunk-6a numpy/torch reference path (combine_chgroups
    + dirty_image_from_uv_grid + apply_edge_mask). Used by unit tests
    + the cube_injection bench. Operates on cf64 host streams.
  * ``"gpu"`` — chunk-8 production path: ``image.imager_gpu.GpuImager``
    + ``image.fused_combine_cuda.fused_dequant_combine_per_fdm`` (the
    fused cint8 → cfp16 dequant + per-fdm combine + cuFFT-cfp16 ifft2
    + edge-mask CUDA kernel chain, D21). At T_det=256/N_fdm=32/N_grid=
    256 the GPU backend hits ~9.8 cubes/s on h01 GPU 1 (the §3.6.3-
    correct version, post-D25 sign fix). The GPU path quantises the
    slot's cf64 streams to cint8 once on the host (D23/D25's
    ``transport.quantize.quantise_streams_global_cint8``) and pushes
    a 5-D cint8 tensor to the GPU; the M3 → M5 RX-ring (chunk-8b) will
    eliminate this host-side quantise once it lands.

§3.6.3 sign convention is consistent across both backends — see D25 in
``M5_PLAN_FIXES.md`` for the lock-in test that verifies
``combine_chgroups`` (CPU) and ``fused_dequant_combine_per_fdm`` (GPU)
agree.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from ..common.constants import N_CHGROUP
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


_LOG = logging.getLogger(__name__)


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
        image_backend: ``"cpu"`` (default; numpy reference path) or
            ``"gpu"`` (production fused-CUDA path via
            ``image.imager_gpu.GpuImager``). The GPU backend requires
            ``cuda`` to be available and that ``device`` is a CUDA
            device. The first call to ``process()`` lazily allocates
            the persistent GpuImager workspace; cube geometry is
            inferred from the first slot.
        gpu_t_det: optional pin for ``GpuImagerConfig.t_det``. ``None``
            ⇒ inferred from the first slot's ``t_det`` (for benches
            that vary T_det between runs).
        gpu_n_fdm: optional pin for ``GpuImagerConfig.n_fdm``. ``None``
            ⇒ inferred from the first slot's ``n_fdm_in_cube``.
        gpu_n_chgroup: pin for ``GpuImagerConfig.n_chgroup``; defaults
            to ``N_CHGROUP=16``.
        gpu_complex_dtype: pin for the GPU imager's intermediate
            complex dtype (``torch.complex32`` for production fp16,
            ``torch.complex64`` for the cf32 numerical-audit fallback).
            Must be compatible with ``cube_dtype``: complex32 →
            float16, complex64 → float32.
        quantise_target_max: clip target for the host-side
            cf → cint8 quantiser. The D25 default is 120 (leaves
            headroom for chgroup-summed roundoff).
        bake_quantise_scale: chunk-8(c) flag controlling whether the
            host-quantise global scale is fed forward to the GPU
            dequant kernel as ``1 / scale`` (broadcast over all
            chgroups) so the dirty-image output is in physical
            visibility units. Defaults to ``True``. Set ``False`` to
            keep the legacy "cint8-magnitude" output that earlier
            chunks produced (Layer-1 σ-clip downstream is cell-wise
            insensitive to a constant scale, so this flag is only
            observable to callers that consume the dirty image
            pre-Layer-1 — the synthetic-burst recovery bench, the
            per-fdm peak-amplitude logger).
    """

    n_grid: int
    edge_mask_kernel_support: int = 5
    edge_mask_sigma_l_pix: Optional[float] = None
    edge_mask_envelope_threshold: float = 0.5
    device: str = "cpu"
    cube_dtype: torch.dtype = torch.float16
    image_backend: Literal["cpu", "gpu"] = "cpu"
    gpu_t_det: Optional[int] = None
    gpu_n_fdm: Optional[int] = None
    gpu_n_chgroup: int = N_CHGROUP
    gpu_complex_dtype: torch.dtype = torch.complex32
    quantise_target_max: int = 120
    bake_quantise_scale: bool = True

    def __post_init__(self) -> None:
        if self.n_grid <= 0 or self.n_grid & (self.n_grid - 1):
            raise ValueError(
                f"n_grid={self.n_grid}, expected positive power of two"
            )
        if self.image_backend not in ("cpu", "gpu"):
            raise ValueError(
                f"image_backend={self.image_backend!r}; expected "
                "'cpu' or 'gpu'"
            )
        if self.image_backend == "gpu":
            # Validate the cube_dtype / complex_dtype pair so the
            # imager's edge-mask multiply is dtype-clean.
            if (self.gpu_complex_dtype == torch.complex32
                    and self.cube_dtype != torch.float16):
                raise ValueError(
                    "image_backend='gpu' with complex32 requires "
                    f"cube_dtype=float16; got {self.cube_dtype}"
                )
            if (self.gpu_complex_dtype == torch.complex64
                    and self.cube_dtype != torch.float32):
                raise ValueError(
                    "image_backend='gpu' with complex64 requires "
                    f"cube_dtype=float32; got {self.cube_dtype}"
                )
            if self.gpu_n_chgroup <= 0:
                raise ValueError(
                    f"gpu_n_chgroup={self.gpu_n_chgroup}; expected > 0"
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
        if config.image_backend == "gpu" and self._device.type != "cuda":
            raise ValueError(
                "image_backend='gpu' requires a cuda device; got "
                f"device={self._device}"
            )
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
        # GpuImager is built lazily on the first cube so the slot's
        # geometry can pin t_det / n_fdm when the config doesn't.
        self._gpu_imager: Optional[object] = None
        # Re-usable host pinned cint8 staging buffer for the GPU
        # backend; allocated on first cube.
        self._cint8_host_buf: Optional[np.ndarray] = None

    @property
    def edge_mask(self) -> torch.Tensor:
        return self._edge_mask

    @property
    def gpu_imager(self) -> Optional[object]:
        """The lazy-built GpuImager (None on CPU backend or before
        first cube). Exposed so benches + tests can inspect workspace
        sizes / config without re-building.
        """
        return self._gpu_imager

    def _build_cube(
        self, slot: CubeRingSlot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to the configured image backend."""
        if self.config.image_backend == "gpu":
            return self._build_cube_gpu(slot)
        return self._build_cube_cpu(slot)

    def _build_cube_cpu(
        self, slot: CubeRingSlot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunk-6a numpy reference path: combine_chgroups + dirty_image
        + edge_mask. Returns (cube, validity_mask) on ``self._device``
        in ``self.config.cube_dtype`` for the cube and ``torch.bool``
        for the validity mask.
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

    def _build_cube_gpu(
        self, slot: CubeRingSlot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunk-8 GPU path: stack + quantise the slot's cf64 chgroup
        streams to cint8, push to GPU, and run
        ``GpuImager.process_cube`` (fused dequant+combine+ifft2+mask).

        Returns the cube as a contiguous CUDA tensor in
        ``self.config.cube_dtype`` plus the validity mask on the same
        device. The GpuImager owns ``output_cube`` in-place; we clone
        only when the caller's pipeline holds onto multiple cubes
        simultaneously (the chunk-6b-α default is single-cube,
        synchronous, so the borrow is safe; if a future caller starts
        pipelining cubes via cuda streams it must clone before the next
        ``process_cube`` overwrites ``output_cube``).

        Production note (chunk-8b): the M3 → M5 RX-ring will deliver
        cint8 streams pre-staged on GPU; the host-side
        ``stack + quantise`` step here is bench-only / pre-RX-ring
        scaffolding.
        """
        from ..image.imager_gpu import GpuImager, GpuImagerConfig
        from ..transport.quantize import quantise_per_chgroup_into_cint8

        cfg = self.config
        if slot.n_grid != cfg.n_grid:
            raise ValueError(
                f"slot.n_grid={slot.n_grid} != pipeline n_grid={cfg.n_grid}"
            )
        n_chg = cfg.gpu_n_chgroup
        n_fdm = slot.n_fdm_in_cube
        t_det = slot.t_det

        if self._gpu_imager is None:
            t_det_cfg = cfg.gpu_t_det if cfg.gpu_t_det is not None else t_det
            n_fdm_cfg = cfg.gpu_n_fdm if cfg.gpu_n_fdm is not None else n_fdm
            self._gpu_imager = GpuImager.build(GpuImagerConfig(
                n_grid=cfg.n_grid,
                t_det=t_det_cfg,
                n_fdm=n_fdm_cfg,
                n_chgroup=n_chg,
                kernel_support=cfg.edge_mask_kernel_support,
                sigma_l_pix=cfg.edge_mask_sigma_l_pix,
                envelope_threshold=cfg.edge_mask_envelope_threshold,
                cube_dtype=cfg.cube_dtype,
                complex_dtype=cfg.gpu_complex_dtype,
                device=self._device,
            ))
            _LOG.info(
                "CubePipeline: built GpuImager (T_det=%d N_fdm=%d "
                "N_grid=%d N_chgroup=%d cube_dtype=%s complex_dtype=%s)",
                t_det_cfg, n_fdm_cfg, cfg.n_grid, n_chg,
                cfg.cube_dtype, cfg.gpu_complex_dtype,
            )

        imager_cfg = self._gpu_imager.config  # type: ignore[union-attr]
        if t_det != imager_cfg.t_det:
            raise ValueError(
                f"slot.t_det={t_det} != GpuImager.t_det="
                f"{imager_cfg.t_det}; pipeline geometry must be static"
            )
        if n_fdm != imager_cfg.n_fdm:
            raise ValueError(
                f"slot.n_fdm_in_cube={n_fdm} != GpuImager.n_fdm="
                f"{imager_cfg.n_fdm}; pipeline geometry must be static"
            )

        # Build a cf64 [N_chg, T_stream, N_grid, N_grid] stack from the
        # slot's per-chgroup dict; missing chgroups → zero-fill (the
        # imager kernel reads a zero contribution).
        # Infer T_stream from the first stream present.
        first_stream = next(iter(slot.per_chgroup_streams.values()))
        t_stream = int(first_stream.shape[0])
        if t_stream < t_det:
            raise ValueError(
                f"slot per-chgroup T_stream={t_stream} < T_det={t_det}; "
                "no fdm trial can fit"
            )
        if self._cint8_host_buf is None:
            self._cint8_host_buf = np.empty(
                (n_chg, t_stream, 2, cfg.n_grid, cfg.n_grid),
                dtype=np.int8,
            )
        elif self._cint8_host_buf.shape[1] != t_stream:
            # T_stream changed between cubes; reallocate (rare; warn).
            _LOG.warning(
                "CubePipeline: T_stream changed (%d → %d); reallocating "
                "cint8 host buffer",
                self._cint8_host_buf.shape[1], t_stream,
            )
            self._cint8_host_buf = np.empty(
                (n_chg, t_stream, 2, cfg.n_grid, cfg.n_grid),
                dtype=np.int8,
            )

        # Two delivery paths:
        #
        # 1) chunk-8b production (M3 RX-ring delivers cint8 already):
        #    ``slot.per_chgroup_cint8_stack`` is populated; we skip the
        #    host quantise and copy the cint8 stack straight to GPU.
        #    chunk-8(c): the slot may also carry per-chgroup
        #    (scale, offset) calibration metadata
        #    (``slot.per_chgroup_scale``, ``slot.per_chgroup_offset_*``)
        #    that the GPU imager applies inline so the dirty image is
        #    in physical visibility units.
        #
        # 2) bench / pre-chunk-8b: ``slot.per_chgroup_streams`` are cf
        #    streams. Quantise streaming per-chgroup into the re-used
        #    host buffer (avoids the dense cf32 stack + cf64 round-trip
        #    the original quantiser allocates, ~13 GiB transient host
        #    allocs at production geometry). The returned global scale
        #    is fed to the GPU imager as ``1 / scale`` broadcast over
        #    all chgroups so the dirty image is in the same physical
        #    units as path (1) — dropping a no-op (scale=1) leaves the
        #    output in cint8-magnitude units which Layer-1 normalises
        #    out cell-wise but loses the absolute-magnitude semantics
        #    production downstream may want.
        chgroup_scale_t: Optional[torch.Tensor] = None
        chgroup_offset_re_t: Optional[torch.Tensor] = None
        chgroup_offset_im_t: Optional[torch.Tensor] = None
        if slot.per_chgroup_cint8_stack is not None:
            cint8_src = slot.per_chgroup_cint8_stack
            if cint8_src.shape != (n_chg, t_stream, 2, cfg.n_grid, cfg.n_grid):
                raise ValueError(
                    f"slot.per_chgroup_cint8_stack.shape={cint8_src.shape}; "
                    f"expected ({n_chg}, {t_stream}, 2, {cfg.n_grid}, "
                    f"{cfg.n_grid})"
                )
            cint8_t = torch.from_numpy(
                np.ascontiguousarray(cint8_src)
            ).to(self._device)
            if slot.per_chgroup_scale is not None:
                chgroup_scale_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_scale, dtype=np.float32)
                ).to(self._device)
            if slot.per_chgroup_offset_re is not None:
                chgroup_offset_re_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_offset_re, dtype=np.float32)
                ).to(self._device)
            if slot.per_chgroup_offset_im is not None:
                chgroup_offset_im_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_offset_im, dtype=np.float32)
                ).to(self._device)
        else:
            quantise_scale = quantise_per_chgroup_into_cint8(
                slot.per_chgroup_streams,
                out_cint8=self._cint8_host_buf,
                target_max=cfg.quantise_target_max,
                zero_fill_missing=True,
            )
            # Push to GPU. ``self._cint8_host_buf`` is mutated in-place
            # next cube; the H2D copy must finalise before we re-use it.
            cint8_t = torch.from_numpy(self._cint8_host_buf).to(self._device)
            # Bake the inverse of the host-quantise scale into the
            # imager dequant so the output is in physical units. The
            # bench host-quantiser uses a single global scale across
            # all chgroups (per-chgroup would distort cross-chgroup
            # magnitude balance — see transport/quantize.py docstring),
            # so all 16 entries here share the same value. Production
            # (path 1) ships per-chgroup metadata directly.
            if cfg.bake_quantise_scale and quantise_scale > 0.0:
                inv_scale = 1.0 / float(quantise_scale)
                chgroup_scale_t = torch.full(
                    (n_chg,), inv_scale,
                    dtype=torch.float32, device=self._device,
                )
        shifts_t = torch.from_numpy(
            np.ascontiguousarray(slot.time_shift_table.shifts.astype(np.int32))
        ).to(self._device)

        cube = self._gpu_imager.process_cube(  # type: ignore[union-attr]
            streams_cint8=cint8_t,
            time_shifts_gpu=shifts_t,
            chgroup_scales=chgroup_scale_t,
            chgroup_offsets_re=chgroup_offset_re_t,
            chgroup_offsets_im=chgroup_offset_im_t,
        )
        # GpuImager owns output_cube in-place; clone so the caller can
        # hold it across the next cube. Cheap on cuda; ~3 ms at
        # production geometry vs the 100 ms imager.
        cube = cube.clone()
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
