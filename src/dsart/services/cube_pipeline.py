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

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    "PrefetchedCube",
    "PrefetchedH2dCube",
    "RetainedCube",
    "CubeRetentionRing",
    "find_cube_for_specnum",
]


_LOG = logging.getLogger(__name__)


# Maximum finite magnitude we keep in fp16 cube tensors. fp16 saturates
# above ~65504, so very bright bursts (or pathological dequant inputs)
# can produce inf/-inf at a handful of pixels in the imager output or
# after Layer-1 normalisation. Detection is robust to that (the detector
# already clips fp16 internally), but downstream consumers — diagnostic
# dumps, plots, clustering centroid math, c2 payloads — break on inf.
# We clip to ~92% of fp16 max so the few saturated samples become a
# finite "bright" value instead of inf. The bound is well above any
# realistic σ-normalised signal in normal operation.
_FP16_CLIP_MAX: float = 60000.0


def _clamp_inf_to_finite(cube: torch.Tensor) -> torch.Tensor:
    """Replace ±inf / NaN in a cube with finite, dtype-safe bounds.

    No-op when the cube has no non-finite values (fast path covers the
    common case at zero cost on quiet cubes). On bright bursts the
    saturated samples land at ±``_FP16_CLIP_MAX`` instead of ±inf so
    the rest of the pipeline can keep doing finite arithmetic on them.

    Runs in-place on the input tensor to avoid an extra full-cube
    allocation in the hot path.
    """
    if cube.dtype == torch.float16:
        clip = _FP16_CLIP_MAX
    else:
        # fp32 / fp64 / bf16: pick a dtype-relative bound. We still
        # clip so a hypothetical inf-producing op upstream can't poison
        # the rest of the pipeline.
        finfo = torch.finfo(cube.dtype)
        clip = finfo.max * 0.5
    return torch.nan_to_num(cube, nan=0.0, posinf=clip, neginf=-clip)


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
    # M7.4 (fp16 imager safety):
    # When False (default since the M7.4 SNR-explosion diagnosis), the
    # GPU imager runs the unit-scale fused-combine fast path — every
    # chgroup contributes its dequantised cint8 magnitude without the
    # per-chgroup mul-add. This matches the original chunk-6c design
    # (see ``image.fused_combine_cuda::fused_dequant_combine_per_fdm``
    # docstring: "Used by every existing caller ... where the global
    # scale lives downstream of the imager (Layer-1 σ-clip normalises
    # it out cell-wise)").
    #
    # Why default False:
    #   * Production fp16/cfp16 cuFFT IFFT2 butterflies have very
    #     little dynamic-range headroom (~6.5e4). When static-sky is
    #     OFF (or hasn't converged) and a bright source is present,
    #     per-chgroup quantiser scales become large (~1e2). After
    #     dequant+combine the UV slab values reach ~2e5, and the
    #     cuFFT-cfp16 intermediates saturate to NaN, producing
    #     all-NaN cubes and pathological Layer-1 σ estimates.
    #   * Layer-1 σ-clip is per-cell and absorbs any constant scaling,
    #     so dropping the calibration multiply does not affect the
    #     detector's SNR (verified on the 0319+415 continuum: bench
    #     numpy IFFT2 22.5σ vs fp16 unit-scale 22.6σ).
    #   * The fast path is also slightly cheaper (skips a fp32 mul-add
    #     per cell × N_chgroup); the imager remains FFT-bound and the
    #     plan §8 cube/s target is unaffected.
    # Set True only for benches/dumps that need physical-Jy amplitudes
    # in the raw cube (e.g. ``bench/imager_only_gpu.py``).
    gpu_imager_apply_per_chgroup_calibration: bool = False

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


@dataclass(slots=True)
class PrefetchedCube:
    """One cube pre-built on the optional prefetch stream."""

    slot: CubeRingSlot
    cube: torch.Tensor
    validity_mask: torch.Tensor
    build_start_ns: int
    build_ready_event: Optional[torch.cuda.Event] = None


@dataclass(slots=True)
class _H2dStaged:
    """Staged H2D tensors for the narrow-overlap path.

    Holds the GPU tensors and an optional CUDA event recorded on the
    H2D stream so the main stream can wait on the copy before running
    the imager. ``cint8_buf_idx`` records which ping-pong cint8 buffer
    the tensor lives in (so the prefetch for the next cube alternates
    targets and doesn't overwrite the in-flight imager input).
    """

    slot: CubeRingSlot
    cint8_t: torch.Tensor
    shifts_t: torch.Tensor
    chgroup_scale_t: Optional[torch.Tensor]
    chgroup_offset_re_t: Optional[torch.Tensor]
    chgroup_offset_im_t: Optional[torch.Tensor]
    cint8_buf_idx: int
    build_start_ns: int
    h2d_event: Optional[torch.cuda.Event] = None


@dataclass(slots=True)
class PrefetchedH2dCube:
    """Prefetched H2D-only handle (chunk-8d narrow overlap)."""

    staged: _H2dStaged


# ---------------------------------------------------------------------------
# C1 cube retention ring (M7.4) — depth-N pinned host buffer keeping the
# last N cube tensors live in CPU RAM so the C2 trigger listener can look
# up + dump any cube within the window.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetainedCube:
    """One pinned-host cube tensor retained on the CPU side for later
    C2-triggered NPZ dump. The full cube tensor is held by reference;
    do not mutate it after handing it to the ring.

    Args:
        cube_id: monotonic cube counter (= ``CubeRingSlot.cube_id``).
        event_specnum_start: spec num at sample 0 of this cube.
        mjd_start: double-precision MJD at sample 0.
        t_det: number of time samples in the cube.
        n_fdm: number of fine-DM trials in the cube
            (= ``CubeGeometry.n_fdm_in_cube``).
        n_grid: spatial grid side.
        sample_period_specnum: spec-nums per detector sample.
        sample_period_us: time between adjacent detector samples in µs.
        pinned_host_tensor: ``np.ndarray`` of shape
            ``[t_det, n_fdm, n_grid, n_grid]`` in fp16; intended to live
            in pinned host memory (the ring's slots own the
            allocations).
    """

    cube_id: int
    event_specnum_start: int
    mjd_start: float
    t_det: int
    n_fdm: int
    n_grid: int
    sample_period_specnum: int
    sample_period_us: float
    pinned_host_tensor: np.ndarray


def find_cube_for_specnum(
    ring: "CubeRetentionRing",
    event_specnum: int,
) -> Optional[RetainedCube]:
    """Return the retained cube whose ``[specnum_start, specnum_start +
    t_det * sample_period_specnum)`` window contains ``event_specnum``.

    Returns ``None`` if no retained cube covers the requested specnum.
    On a multi-hit (impossible at production geometry where cubes
    advance by ``cube_cadence_samples`` < ``t_det`` but possible during
    extreme overlap) the most recently written hit wins (the C2
    trigger always wants the freshest cube containing the event).
    """
    # Walk newest → oldest so the freshest hit wins.
    best: Optional[RetainedCube] = None
    for cube in ring.iter_newest_first():
        start = int(cube.event_specnum_start)
        end_exclusive = start + int(cube.t_det) * int(cube.sample_period_specnum)
        if start <= int(event_specnum) < end_exclusive:
            best = cube
            break
    return best


class CubeRetentionRing:
    """Circular buffer of the last N detector cubes (M7.4 C1 stage).

    Each ring slot owns a pinned-host ``np.ndarray`` sized for the
    fixed production geometry ``[t_det, n_fdm, n_grid, n_grid]`` fp16.
    The ring is written to with :meth:`stage_cube` which (a) ensures
    the destination slot is allocated, (b) does the GPU→host fp16
    copy, and (c) commits a new :class:`RetainedCube` record at the
    write head.

    At production geometry (``t_det=192``, ``n_fdm=34``, ``n_grid=256``,
    fp16) one cube ≈ 3.2 GiB; ``depth=8`` is ~26 GiB per gpu_half =
    ~52 GiB per search node (well inside the 256 GiB budget). The
    depth is a YAML knob (``c1.cube_ring_depth``) so the operator can
    throttle down to ``depth=2`` for tight memory during shake-down.

    Args:
        depth: ring size. ``>= 1``.
        t_det, n_fdm, n_grid: pin geometry. ``stage_cube`` raises if
            a new cube's shape doesn't match.
        pinned: when True, allocate the host buffers with
            ``torch.empty(..., pin_memory=True)`` and wrap as
            ``np.ndarray`` for the cross-thread interface. Tests pass
            ``pinned=False`` to skip the pin (no CUDA on CI hosts).
    """

    def __init__(
        self,
        *,
        depth: int,
        t_det: int,
        n_fdm: int,
        n_grid: int,
        pinned: bool = True,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth={depth}, expected ≥ 1")
        if t_det <= 0 or n_fdm <= 0 or n_grid <= 0:
            raise ValueError(
                f"t_det={t_det}, n_fdm={n_fdm}, n_grid={n_grid} all expected > 0"
            )
        self._depth = int(depth)
        self._t_det = int(t_det)
        self._n_fdm = int(n_fdm)
        self._n_grid = int(n_grid)
        self._pinned = bool(pinned)
        self._buffers: List[Optional[np.ndarray]] = [None] * self._depth
        self._records: List[Optional[RetainedCube]] = [None] * self._depth
        self._write_pos: int = 0
        self._n_committed: int = 0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def n_committed(self) -> int:
        """Number of cubes ever committed (monotonic; not the live
        count, which is ``min(n_committed, depth)``)."""
        return self._n_committed

    @property
    def write_pos(self) -> int:
        return self._write_pos

    @property
    def t_det(self) -> int:
        return self._t_det

    @property
    def n_fdm(self) -> int:
        return self._n_fdm

    @property
    def n_grid(self) -> int:
        return self._n_grid

    def _ensure_buffer(self, idx: int) -> np.ndarray:
        """Lazy-allocate the pinned host buffer at ring slot ``idx``."""
        buf = self._buffers[idx]
        shape = (self._t_det, self._n_fdm, self._n_grid, self._n_grid)
        if buf is None or buf.shape != shape or buf.dtype != np.float16:
            if self._pinned:
                # Pin via torch so the GPU-side ``cudaMemcpyAsync`` can
                # run on a side stream; expose the ndarray view to the
                # listener / writer threads. We don't import torch at
                # module-time here (already imported above for the
                # detector glue), but we keep the allocation behind a
                # try/except so CPU-only CI hosts still build the ring.
                try:
                    t = torch.empty(shape, dtype=torch.float16, pin_memory=True)
                    buf = t.numpy()
                except (RuntimeError, AssertionError):
                    buf = np.empty(shape, dtype=np.float16)
            else:
                buf = np.empty(shape, dtype=np.float16)
            self._buffers[idx] = buf
        return buf

    def stage_cube(
        self,
        *,
        cube_id: int,
        event_specnum_start: int,
        mjd_start: float,
        sample_period_specnum: int,
        sample_period_us: float,
        cube_tensor: Any,
    ) -> RetainedCube:
        """Copy ``cube_tensor`` into the next ring slot and commit a
        :class:`RetainedCube` record. ``cube_tensor`` may be a
        ``np.ndarray`` (CPU path / tests) or a torch tensor (GPU path);
        in either case the destination buffer is pinned-host fp16. The
        copy is synchronous from the caller's perspective — the
        production path issues it from the cube driver after the
        detector returns, so it does NOT block the GPU pipeline (the
        detector's fwd has already completed before we touch the ring).

        Returns:
            The freshly committed :class:`RetainedCube`.
        """
        if cube_id < 0:
            raise ValueError(f"cube_id={cube_id}, expected ≥ 0")
        idx = self._write_pos
        dest = self._ensure_buffer(idx)
        # Hot path: dest is a pinned-host fp16 ndarray (when
        # ``self._pinned`` is True) and ``cube_tensor`` is a CUDA fp16
        # torch tensor. The previous implementation did GPU→unpinned
        # CPU (.cpu().numpy()) then CPU→pinned CPU (np.copyto), which
        # required two full 815 MiB copies and stalled the main thread
        # for ~700 ms per cube (measured 1.0 cubes/s vs. the 7.45
        # cubes/s target on n01, 2026-05-26). We now copy in ONE step
        # via ``torch.Tensor.copy_`` with a zero-copy ``from_numpy``
        # wrap of the pinned dest — that issues a single
        # ``cudaMemcpyAsync`` DtoH into the DMA-able buffer.
        if isinstance(cube_tensor, np.ndarray):
            src = cube_tensor
            if src.dtype != np.float16:
                src = src.astype(np.float16, copy=False)
            if src.shape != dest.shape:
                raise ValueError(
                    f"cube_tensor.shape={src.shape} != ring geometry {dest.shape}"
                )
            np.copyto(dest, src, casting="same_kind")
        elif (
            hasattr(cube_tensor, "detach")
            and hasattr(cube_tensor, "cpu")
            and hasattr(cube_tensor, "numpy")
        ):
            src_t = cube_tensor.detach()
            if src_t.dtype != torch.float16:
                src_t = src_t.to(torch.float16)
            if tuple(src_t.shape) != tuple(dest.shape):
                raise ValueError(
                    f"cube_tensor.shape={tuple(src_t.shape)} != "
                    f"ring geometry {dest.shape}"
                )
            # Wrap the pinned-host destination as a torch tensor
            # (zero copy) and issue one direct DMA copy from GPU. The
            # ``non_blocking=True`` hint asks for an async DtoH copy
            # over a side stream; ``record_event`` captures completion
            # so consumers (the C2TriggerListener / cube dump writer)
            # can sync on this slot's event BEFORE reading the buffer.
            # We intentionally DO NOT synchronize here — that would
            # stall the main thread by ~80–150 ms per 815 MiB cube
            # (PCIe DMA bound) and serialise it against the prefetched
            # NEXT cube's compute, capping throughput at ~4 cubes/s.
            # The retention horizon (``depth`` cubes ≈ seconds) is
            # orders of magnitude longer than the DMA latency, so by
            # the time a C2 trigger reads back the slot the event has
            # long since fired.
            dest_t = torch.from_numpy(dest)
            dest_t.copy_(src_t, non_blocking=True)
            self._last_copy_event = None
            if src_t.is_cuda:
                ev = torch.cuda.Event(blocking=False)
                ev.record()
                self._last_copy_event = ev
        else:
            raise TypeError(
                f"cube_tensor must be np.ndarray or torch tensor; got "
                f"{type(cube_tensor).__name__}"
            )
        record = RetainedCube(
            cube_id=int(cube_id),
            event_specnum_start=int(event_specnum_start),
            mjd_start=float(mjd_start),
            t_det=int(self._t_det),
            n_fdm=int(self._n_fdm),
            n_grid=int(self._n_grid),
            sample_period_specnum=int(sample_period_specnum),
            sample_period_us=float(sample_period_us),
            pinned_host_tensor=dest,
        )
        self._records[idx] = record
        self._write_pos = (idx + 1) % self._depth
        self._n_committed += 1
        return record

    def iter_newest_first(self):
        """Iterator over live ``RetainedCube`` records from newest →
        oldest. Skips empty slots (first ``min(depth, n_committed)``
        bumps fill the ring)."""
        n_live = min(self._depth, self._n_committed)
        for k in range(n_live):
            idx = (self._write_pos - 1 - k) % self._depth
            rec = self._records[idx]
            if rec is not None:
                yield rec

    def iter_oldest_first(self):
        """Iterator over live ``RetainedCube`` records from oldest →
        newest. Useful for diagnostics."""
        n_live = min(self._depth, self._n_committed)
        start = (self._write_pos - n_live) % self._depth
        for k in range(n_live):
            idx = (start + k) % self._depth
            rec = self._records[idx]
            if rec is not None:
                yield rec

    def snapshot(self) -> List[RetainedCube]:
        """Live snapshot of retained cubes, newest first. Convenience
        wrapper around :meth:`iter_newest_first`."""
        return list(self.iter_newest_first())


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
        *,
        fine_dm_pc_cm3: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.layer1_state = layer1_state
        # M7.4 fix: detector.forward(..., fine_dm_pc_cm3=...) is what
        # makes the decoder write the actual physical DM (pc/cc) into
        # ``Candidate.dm_fine`` instead of falling back to the fdm INDEX
        # (decoder.py lines 206-209). Without this the C1 emitter ships
        # ``dm_pc_cc = fdm_idx`` and the C2 trigger criteria can't match
        # in pc/cc units (see C1 csv: ``dm_pc_cc`` and ``dm_idx_global``
        # were identical integers prior to this fix).
        self._fine_dm_pc_cm3: Optional[torch.Tensor] = fine_dm_pc_cm3
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
        # Re-usable CUDA staging buffers to avoid per-cube allocations
        # on the hot path.
        self._cint8_gpu_buf: Optional[torch.Tensor] = None
        # Ping-pong cint8 GPU buffers for the narrow-overlap H2D path
        # (cube N's H2D writes one; cube N+1's H2D writes the other so
        # the in-flight imager input is not clobbered). Built lazily
        # on first H2D prefetch.
        self._cint8_gpu_buf_pp: List[Optional[torch.Tensor]] = [None, None]
        self._cint8_pp_index: int = 0
        self._shifts_gpu_buf: Optional[torch.Tensor] = None
        self._validity_gpu_buf: Optional[torch.Tensor] = None
        self._validity_all_true_gpu: Optional[torch.Tensor] = None
        # M7.2-amend (2026-05-20): persistent pinned host buffer the
        # ``_build_validity_mask`` path copies the slot's freshly
        # alloc'd numpy validity mask into before the H2D. Pinning is
        # required for ``non_blocking=True`` to actually run async on
        # the prefetch stream; see ``_build_validity_mask`` for the
        # full rationale.
        self._validity_host_pinned: Optional[torch.Tensor] = None
        self._last_shifts_host_id: Optional[int] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        # Separate stream dedicated to the cint8 H2D copy in the narrow
        # overlap path. The H2D engine is independent from the GPU's
        # SMs so this can run fully concurrent with the main stream's
        # imager / detector kernels.
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        if self._device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self._device)
            self._h2d_stream = torch.cuda.Stream(device=self._device)
        # Single-pass Layer-1 fold-in (chunk-8d): the GPU imager can
        # multiply its output by ``edge_mask * (1/σ_layer1_prev[f])``
        # so the cube emerges already-Layer-1-normalised; the per-cube
        # ``cube / σ`` divide downstream is then a no-op. Activated by
        # default on the production GPU backend with a 1-cube lag
        # (same semantics as the single-pass Layer-2 σ_k EMA). Toggle
        # off via ``DSART_DISABLE_FUSED_LAYER1=1`` for A/B benches.
        self._fuse_layer1_into_imager = (
            self._device.type == "cuda"
            and config.image_backend == "gpu"
            and not bool(
                int(os.environ.get("DSART_DISABLE_FUSED_LAYER1", "0"))
            )
        )
        # Per-fdm σ_layer1 from the previous cube, used to seed the
        # imager's fused mask each cube. ``None`` means "no prev cube
        # yet" → imager runs with the constant ``edge_mask_real``
        # (effectively σ_prev = 1 for cube 0, which is also during
        # Layer-1 burn-in so the warmup flag is set anyway).
        self._sigma_layer1_prev: Optional[torch.Tensor] = None

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
        cube = _clamp_inf_to_finite(cube)
        validity_mask = torch.from_numpy(
            np.ascontiguousarray(slot.validity_mask)
        ).to(device=self._device, dtype=torch.bool)
        return cube, validity_mask

    def _build_cube_gpu(
        self, slot: CubeRingSlot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunk-8 GPU path: stage H2D for cint8/shifts/calibration,
        run ``GpuImager.process_cube`` (fused dequant+combine+ifft2+
        mask), and set up the validity mask. Returns the cube as a
        contiguous CUDA tensor in ``self.config.cube_dtype`` plus the
        validity mask on the same device.

        Production note (chunk-8b): the M3 → M5 RX-ring delivers
        cint8 streams pre-staged on host; the host-side
        ``stack + quantise`` fallback below is bench-only / pre-
        RX-ring scaffolding (used when ``per_chgroup_streams`` arrives
        without ``per_chgroup_cint8_stack`` populated).
        """
        staged = self._stage_h2d(slot, cint8_dest_idx=0, use_pp=False)
        cube = self._run_imager_from_staged(staged)
        validity_mask = self._build_validity_mask(slot)
        return cube, validity_mask

    def _stage_h2d(
        self,
        slot: CubeRingSlot,
        *,
        cint8_dest_idx: int = 0,
        use_pp: bool = False,
    ) -> _H2dStaged:
        """Pin host buffers and copy cint8 / shifts / calibration to
        the GPU. Returns the staged tensors but does NOT run the imager
        or any validity-mask setup.

        Args:
            slot: cube ring slot to stage.
            cint8_dest_idx: which ping-pong cint8 GPU buffer to target
                (0 or 1). Ignored when ``use_pp`` is False — the
                synchronous ``_build_cube_gpu`` path uses the single
                ``_cint8_gpu_buf`` slot to preserve historical
                behaviour.
            use_pp: when True, write into the ping-pong buffer at
                ``_cint8_gpu_buf_pp[cint8_dest_idx]``; the narrow-
                overlap path uses this so prefetch H2D for cube N+1
                doesn't clobber the imager's input for cube N.
        """
        from .host_pin import maybe_register_host_buffer
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
        enable_gpu_buf_reuse = bool(
            int(os.environ.get("DSART_ENABLE_GPU_BUF_REUSE", "0"))
        )

        def _resolve_cint8_buf(
            host_shape: Tuple[int, ...],
        ) -> torch.Tensor:
            """Return the GPU buffer to copy this cube's cint8 into.

            Picks the ping-pong slot when ``use_pp`` is True; otherwise
            falls back to the legacy single ``_cint8_gpu_buf`` field.
            """
            if use_pp:
                buf = self._cint8_gpu_buf_pp[cint8_dest_idx]
                if (
                    buf is None
                    or buf.shape != host_shape
                    or buf.device != self._device
                ):
                    buf = torch.empty(
                        host_shape, dtype=torch.int8, device=self._device,
                    )
                    self._cint8_gpu_buf_pp[cint8_dest_idx] = buf
                return buf
            if (
                self._cint8_gpu_buf is None
                or self._cint8_gpu_buf.shape != host_shape
                or self._cint8_gpu_buf.device != self._device
            ):
                self._cint8_gpu_buf = torch.empty(
                    host_shape, dtype=torch.int8, device=self._device,
                )
            return self._cint8_gpu_buf

        # Always non_blocking when use_pp (the caller runs on the H2D
        # stream and synchronises via a CUDA event); also when the
        # legacy gpu_buf_reuse path is on. Off otherwise to preserve
        # historical behaviour.
        non_blocking = bool(use_pp or enable_gpu_buf_reuse)

        if slot.per_chgroup_cint8_stack is not None:
            cint8_src = slot.per_chgroup_cint8_stack
            if cint8_src.shape != (n_chg, t_stream, 2, cfg.n_grid, cfg.n_grid):
                raise ValueError(
                    f"slot.per_chgroup_cint8_stack.shape={cint8_src.shape}; "
                    f"expected ({n_chg}, {t_stream}, 2, {cfg.n_grid}, "
                    f"{cfg.n_grid})"
                )
            cint8_host = np.ascontiguousarray(cint8_src)
            maybe_register_host_buffer(cint8_host)
            cint8_host_t = torch.from_numpy(cint8_host)
            if use_pp or enable_gpu_buf_reuse:
                gpu_buf = _resolve_cint8_buf(tuple(cint8_host_t.shape))
                gpu_buf.copy_(cint8_host_t, non_blocking=non_blocking)
                cint8_t = gpu_buf
            else:
                cint8_t = cint8_host_t.to(self._device)
            # M7.4: prefer per-(chgroup, t) sidecars when present. The
            # ProductionRxRingSource M7.4 scatter path populates these
            # via ``rx_ring_assemble_dense_block`` (per-slot scale +
            # offset from the wire). The H2D ships 2-D [N_chg, T_stream]
            # f32 tensors; the GPU dequant kernel detects rank 2 and
            # dispatches to ``fused_dequant_combine_per_fdm_per_t``.
            if slot.per_chgroup_scale_per_t is not None:
                chgroup_scale_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_scale_per_t, dtype=np.float32)
                ).to(self._device, non_blocking=non_blocking)
            elif slot.per_chgroup_scale is not None:
                chgroup_scale_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_scale, dtype=np.float32)
                ).to(self._device, non_blocking=non_blocking)
            if slot.per_chgroup_offset_re_per_t is not None:
                chgroup_offset_re_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_offset_re_per_t, dtype=np.float32)
                ).to(self._device, non_blocking=non_blocking)
            elif slot.per_chgroup_offset_re is not None:
                chgroup_offset_re_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_offset_re, dtype=np.float32)
                ).to(self._device, non_blocking=non_blocking)
            if slot.per_chgroup_offset_im_per_t is not None:
                chgroup_offset_im_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_offset_im_per_t, dtype=np.float32)
                ).to(self._device, non_blocking=non_blocking)
            elif slot.per_chgroup_offset_im is not None:
                chgroup_offset_im_t = torch.from_numpy(
                    np.ascontiguousarray(slot.per_chgroup_offset_im, dtype=np.float32)
                ).to(self._device, non_blocking=non_blocking)
        else:
            quantise_scale = quantise_per_chgroup_into_cint8(
                slot.per_chgroup_streams,
                out_cint8=self._cint8_host_buf,
                target_max=cfg.quantise_target_max,
                zero_fill_missing=True,
            )
            maybe_register_host_buffer(self._cint8_host_buf)
            cint8_host_t = torch.from_numpy(self._cint8_host_buf)
            if use_pp or enable_gpu_buf_reuse:
                gpu_buf = _resolve_cint8_buf(tuple(cint8_host_t.shape))
                gpu_buf.copy_(cint8_host_t, non_blocking=non_blocking)
                cint8_t = gpu_buf
            else:
                cint8_t = cint8_host_t.to(self._device)
            if cfg.bake_quantise_scale and quantise_scale > 0.0:
                inv_scale = 1.0 / float(quantise_scale)
                chgroup_scale_t = torch.full(
                    (n_chg,), inv_scale,
                    dtype=torch.float32, device=self._device,
                )
        shifts_host_np = np.ascontiguousarray(slot.time_shift_table.shifts.astype(np.int32))
        shifts_host_id = int(shifts_host_np.__array_interface__["data"][0])
        if enable_gpu_buf_reuse or use_pp:
            if (
                self._shifts_gpu_buf is None
                or self._shifts_gpu_buf.shape != shifts_host_np.shape
                or self._shifts_gpu_buf.device != self._device
            ):
                self._shifts_gpu_buf = torch.empty(
                    shifts_host_np.shape, dtype=torch.int32, device=self._device
                )
                self._last_shifts_host_id = None
            if self._last_shifts_host_id != shifts_host_id:
                shifts_host_t = torch.from_numpy(shifts_host_np)
                self._shifts_gpu_buf.copy_(
                    shifts_host_t, non_blocking=non_blocking,
                )
                self._last_shifts_host_id = shifts_host_id
            shifts_t = self._shifts_gpu_buf
        else:
            shifts_t = torch.from_numpy(shifts_host_np).to(self._device)

        return _H2dStaged(
            slot=slot,
            cint8_t=cint8_t,
            shifts_t=shifts_t,
            chgroup_scale_t=chgroup_scale_t,
            chgroup_offset_re_t=chgroup_offset_re_t,
            chgroup_offset_im_t=chgroup_offset_im_t,
            cint8_buf_idx=cint8_dest_idx,
            build_start_ns=0,  # caller fills in
        )

    def _run_imager_from_staged(self, staged: _H2dStaged) -> torch.Tensor:
        """Run the GPU imager on already-staged H2D tensors.

        Always runs on ``torch.cuda.current_stream()``. The caller is
        responsible for waiting on any H2D event before invoking this.
        """
        if self._gpu_imager is None:
            raise RuntimeError(
                "_run_imager_from_staged called before GpuImager was built"
            )
        # M7.4: gate per-chgroup calibration on cfg flag; default False
        # picks the unit-scale fused fast path (chunk-6c design) which is
        # fp16-safe and matches the bench's σ-normalised behaviour.
        # See ``CubePipelineConfig.gpu_imager_apply_per_chgroup_calibration``
        # for the full rationale.
        apply_cal = bool(
            self.config.gpu_imager_apply_per_chgroup_calibration
        )
        cube = self._gpu_imager.process_cube(  # type: ignore[union-attr]
            streams_cint8=staged.cint8_t,
            time_shifts_gpu=staged.shifts_t,
            chgroup_scales=staged.chgroup_scale_t if apply_cal else None,
            chgroup_offsets_re=staged.chgroup_offset_re_t if apply_cal else None,
            chgroup_offsets_im=staged.chgroup_offset_im_t if apply_cal else None,
        )
        return _clamp_inf_to_finite(cube)

    def _build_validity_mask(self, slot: CubeRingSlot) -> torch.Tensor:
        """Materialise the [T_det, N_fdm] bool validity mask on the
        GPU. Uses the cached all-true tensor when the slot reports no
        invalid (T, F) cells, otherwise performs a small H2D.
        """
        enable_gpu_buf_reuse = bool(
            int(os.environ.get("DSART_ENABLE_GPU_BUF_REUSE", "0"))
        )
        validity_np = np.ascontiguousarray(slot.validity_mask)
        if bool(validity_np.all()):
            if (
                self._validity_all_true_gpu is None
                or self._validity_all_true_gpu.shape != validity_np.shape
                or self._validity_all_true_gpu.device != self._device
            ):
                self._validity_all_true_gpu = torch.ones(
                    validity_np.shape, dtype=torch.bool, device=self._device
                )
            return self._validity_all_true_gpu
        if enable_gpu_buf_reuse:
            if (
                self._validity_gpu_buf is None
                or self._validity_gpu_buf.shape != validity_np.shape
                or self._validity_gpu_buf.device != self._device
            ):
                self._validity_gpu_buf = torch.empty(
                    validity_np.shape, dtype=torch.bool, device=self._device
                )
            # M7.2-amend (2026-05-20): persistent PINNED host buffer +
            # ``non_blocking=True`` H2D
            # ----------------------------------------------------------
            # Before this amend the validity copy ran with
            # ``non_blocking=False``, which under ``prefetch_build``
            # synchronously drained the ``_prefetch_stream`` queue
            # (cudaStreamSynchronize) at the END of every cube — i.e.
            # the prefetch was serialised with the previous cube's
            # detector + layer1 work on the main stream. py-spy
            # (2026-05-20) showed this single line as ~77% of the
            # per-cube CPU wall after the cint8 H2D became async.
            #
            # PyTorch only honours ``non_blocking=True`` for H2D when
            # the source memory is pinned (otherwise it silently falls
            # back to a synchronous copy). The slot validity mask is
            # freshly allocated by ``ProductionRxRingSource`` every
            # cube (np.broadcast_to(...).copy()), so we maintain a
            # persistent host-pinned PyTorch buffer here and copy the
            # numpy view into it before the H2D. Cost of the
            # ``numpy->pinned host`` copy is ~6.5 KiB per cube (T_det
            # x N_fdm bool); the resulting H2D is fully async and the
            # ``build_ready_event`` recorded immediately after in
            # ``prefetch_build`` is what downstream stages wait on
            # for correctness.
            if (
                self._validity_host_pinned is None
                or self._validity_host_pinned.shape != validity_np.shape
            ):
                self._validity_host_pinned = torch.empty(
                    validity_np.shape, dtype=torch.bool, pin_memory=True,
                )
            self._validity_host_pinned.copy_(torch.from_numpy(validity_np))
            self._validity_gpu_buf.copy_(
                self._validity_host_pinned, non_blocking=True,
            )
            return self._validity_gpu_buf
        return torch.from_numpy(validity_np).to(
            device=self._device, dtype=torch.bool
        )

    def _layer1_normalise(
        self,
        cube: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-fdm Layer-1 σ + return (cube_normalised, sigma).

        Uses the stateful ``Layer1State`` if present (production path:
        running burn-in across cubes); falls back to per-cube
        σ-clipped scalar if ``layer1_state is None`` (test/bench path
        that wants Layer-1 to be a no-op).

        Single-pass Layer-1 fused-imager mode (chunk-8d, active when
        ``self._fuse_layer1_into_imager`` is True): the imager has
        already multiplied the cube by ``1/σ_layer1_prev[f]``, so the
        incoming ``cube`` is in pre-divided units. Layer-1 σ is
        estimated on the pre-divided cube and multiplied by
        ``σ_layer1_prev`` to recover absolute units before feeding
        ``Layer1State``. The σ to use for the *next* cube is then
        pushed to the imager's per-fdm fused mask.
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
        if (
            self._fuse_layer1_into_imager
            and self.config.image_backend == "gpu"
        ):
            return self._layer1_normalise_fused(cube)
        sigma = self.layer1_state.update_and_query(cube=cube)
        if sigma.device != cube.device:
            sigma = sigma.to(cube.device)
        # Broadcast-divide. cube is [T_det, N_fdm, H, W]; sigma is [N_fdm].
        cube_normalised = cube / sigma[None, :, None, None].to(cube.dtype)
        return _clamp_inf_to_finite(cube_normalised), sigma

    def _layer1_normalise_fused(
        self,
        cube: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layer-1 estimation with the divide fused into the imager.

        On entry, ``cube`` has already been multiplied by
        ``1/σ_layer1_prev[f]`` inside the imager (or is in raw units
        on the very first cube, when ``self._sigma_layer1_prev`` is
        None). We:

          1. Estimate σ on the cube as-presented (``σ_observed``).
          2. Recover absolute-unit σ_this = σ_observed * σ_prev so
             the Layer1State burn-in/median tracks the same statistic
             it always has.
          3. Push the resulting per-fdm σ_for_use into the imager's
             fused mask cache so the NEXT cube emerges pre-divided
             by σ_for_use.

        The detector receives ``cube`` unchanged (already divided by
        σ_prev). With the single-pass Layer-2 EMA, the 1-cube lag in
        σ_layer1 composes cleanly with the existing 1-cube lag in
        σ_k — both apply during EMA burn-in and become bit-identical
        steady state after the burn-in completes.
        """
        sigma_observed = self.layer1_state._layer1_sigma_cached(cube)
        if self._sigma_layer1_prev is None:
            sigma_this = sigma_observed
        else:
            prev = self._sigma_layer1_prev.to(
                dtype=sigma_observed.dtype, device=sigma_observed.device
            )
            sigma_this = sigma_observed * prev
        sigma_for_use = self.layer1_state.update_and_query(
            per_fdm_sigma=sigma_this
        )
        if sigma_for_use.device != cube.device:
            sigma_for_use = sigma_for_use.to(cube.device)
        # Refresh imager's fused mask for the NEXT cube. ``set_edge_
        # mask_per_fdm`` consumes ``1/σ`` directly.
        sigma_inv = torch.reciprocal(
            sigma_for_use.clamp(min=torch.finfo(torch.float32).tiny)
        )
        if self._gpu_imager is not None:
            self._gpu_imager.set_edge_mask_per_fdm(sigma_inv)
        self._sigma_layer1_prev = sigma_for_use
        return _clamp_inf_to_finite(cube), sigma_for_use

    def prefetch_h2d(self, slot: CubeRingSlot) -> PrefetchedH2dCube:
        """Stage the cube's H2D copies on the dedicated H2D stream.

        Chunk-8d narrow overlap: only the cint8 / shifts / calibration
        host→device copy runs concurrently with the previous cube's
        imager + detector work. The imager runs on the main stream
        inside :meth:`process_h2d_prefetched` (after waiting on the
        H2D event). This avoids the SM contention that regressed the
        prior full-prefetch overlap path (the H2D engine is
        independent from the SMs).

        Uses ping-pong cint8 GPU buffers so cube N+1's H2D doesn't
        clobber cube N's in-flight imager input.
        """
        t0 = time.perf_counter_ns()
        # The first build (imager not yet built) cannot stage H2D on a
        # background stream because the imager build path inspects /
        # allocates the output cube on the main stream. Run inline.
        if self._h2d_stream is None or self._gpu_imager is None:
            staged = self._stage_h2d(slot, cint8_dest_idx=0, use_pp=False)
            staged.build_start_ns = t0
            return PrefetchedH2dCube(staged=staged)
        # Alternate ping-pong buffer index per call.
        idx = self._cint8_pp_index
        self._cint8_pp_index = 1 - self._cint8_pp_index
        with torch.cuda.stream(self._h2d_stream):
            staged = self._stage_h2d(
                slot, cint8_dest_idx=idx, use_pp=True,
            )
            event = torch.cuda.Event(enable_timing=False)
            event.record(self._h2d_stream)
        staged.build_start_ns = t0
        staged.h2d_event = event
        return PrefetchedH2dCube(staged=staged)

    def process_h2d_prefetched(
        self,
        prefetched: PrefetchedH2dCube,
    ) -> CubePipelineResult:
        """Run imager + Layer-1 + detector on a cube whose H2D was
        prefetched via :meth:`prefetch_h2d`.

        Main stream waits on the H2D event (if any), then runs the
        imager. From the caller's perspective ``build_cube`` is the
        sum of (a) the asynchronous H2D wait + (b) the imager
        compute, plus any contention from the next cube's prefetch.
        Layer-1 and detector behave as in :meth:`process`.
        """
        staged = prefetched.staged
        slot = staged.slot
        if staged.h2d_event is not None:
            torch.cuda.current_stream(self._device).wait_event(
                staged.h2d_event
            )
        # Imager + validity mask are computed on the main stream so
        # they overlap with the NEXT cube's H2D prefetch (issued by
        # the caller before process_h2d_prefetched is invoked).
        cube = self._run_imager_from_staged(staged)
        validity_mask = self._build_validity_mask(slot)
        t1 = time.perf_counter_ns()
        cube_norm, sigma_layer1 = self._layer1_normalise(cube)
        t2 = time.perf_counter_ns()
        with torch.no_grad():
            cands = self.detector.forward(
                cube_norm,
                validity_mask,
                sigma_layer1,
                event_specnum=int(slot.specnum_start),
                fine_dm_pc_cm3=self._fine_dm_pc_cm3,
            )
        t3 = time.perf_counter_ns()
        timings = {
            "build_cube": t1 - staged.build_start_ns,
            "layer1_norm": t2 - t1,
            "detector_forward": t3 - t2,
            "total": t3 - staged.build_start_ns,
        }
        self._debug_log_cube_stats(
            slot.cube_id, cube, cube_norm, sigma_layer1, validity_mask, cands,
        )
        return CubePipelineResult(
            cube_id=slot.cube_id,
            specnum_start=slot.specnum_start,
            cube=cube_norm,
            sigma_layer1=sigma_layer1,
            validity_mask=validity_mask,
            candidates=cands,
            stage_timings_ns=timings,
        )

    @staticmethod
    def _debug_log_cube_stats(
        cube_id: int,
        cube_raw: torch.Tensor,
        cube_norm: torch.Tensor,
        sigma_layer1: torch.Tensor,
        validity_mask: torch.Tensor,
        cands,
    ) -> None:
        """One-line stats per cube for M7.4 detector debug. Reductions
        happen in the cube's native dtype to avoid a fp32 copy that
        would blow the GPU memory budget for the [T_det,N_fdm,H,W]
        cube. We use ``.amin()/.amax()/.mean()/.std()`` and convert to
        python scalars via ``.item()``; ``mean`` and ``std`` upcast
        internally for fp16 but only on the scalar accumulator (no
        full-tensor allocation)."""
        try:
            # All reductions are scalar — no whole-cube copies.
            raw_min = float(cube_raw.amin().item())
            raw_max = float(cube_raw.amax().item())
            raw_mean = float(cube_raw.float().mean().item()) if cube_raw.numel() < (1 << 24) else float("nan")
            # For large cubes skip mean (mean() over fp16 needs fp32 accum, can OOM).
            # Use a small-slice mean as a cheap proxy.
            if cube_raw.numel() >= (1 << 24):
                # 64-pixel slice across first dim only — cheap.
                raw_mean = float(cube_raw[:1].float().mean().item())
            n_norm_min = float(cube_norm.amin().item())
            n_norm_max = float(cube_norm.amax().item())
            # Convert sigma_layer1 (small: [N_fdm]) without worries.
            sl = sigma_layer1.detach().to(torch.float32).flatten()
            sl_min = float(sl.amin().item())
            sl_max = float(sl.amax().item())
            sl_med = float(sl.median().item())
            vm = validity_mask.detach()
            n_valid = int(vm.sum().item())
            n_total = int(vm.numel())
            # peak SNR estimate: cube_norm.max(); if Layer1 normalised, max ≈ peak SNR
            _LOG.info(
                "cube_debug cid=%d shape=%s dtype=%s raw[min/max]=%.3g/%.3g raw_slice_mean=%.3g "
                "norm[min/max]=%.3g/%.3g sigma_l1[min/med/max]=%.3g/%.3g/%.3g "
                "valid=%d/%d (%.1f%%) n_cands=%d",
                cube_id,
                tuple(cube_raw.shape),
                str(cube_raw.dtype),
                raw_min, raw_max, raw_mean,
                n_norm_min, n_norm_max,
                sl_min, sl_med, sl_max,
                n_valid, n_total, 100.0 * n_valid / max(1, n_total),
                len(cands),
            )
            dump_dir = os.environ.get("DSART_M74_CUBE_DUMP_DIR")
            dump_every = int(os.environ.get("DSART_M74_CUBE_DUMP_EVERY", "0") or 0)
            if dump_dir and dump_every > 0 and (cube_id % dump_every) == 0:
                try:
                    os.makedirs(dump_dir, exist_ok=True)
                    # M7.4: dual-GPU search has two compute halves writing
                    # cubes for overlapping cube_ids; an optional
                    # ``DSART_M74_DUMP_TAG`` env (set per-half by the
                    # orchestrator) disambiguates so neither overwrites
                    # the other's dump. Default falls back to PID for
                    # unambiguous per-process files.
                    tag_env = os.environ.get("DSART_M74_DUMP_TAG", "")
                    tag = tag_env if tag_env else f"pid{os.getpid()}"
                    p = os.path.join(
                        dump_dir,
                        f"cube_{cube_id:06d}_{tag}.pt",
                    )
                    # Move to CPU in fp16 to keep the dump small.
                    cn_cpu = cube_norm.detach().to(
                        device="cpu",
                        dtype=torch.float16
                        if cube_norm.dtype.is_floating_point
                        else cube_norm.dtype,
                    )
                    cr_cpu = cube_raw.detach().to(
                        device="cpu",
                        dtype=torch.float16
                        if cube_raw.dtype.is_floating_point
                        else cube_raw.dtype,
                    )
                    torch.save(
                        {
                            "cube_raw": cr_cpu,
                            "cube_norm": cn_cpu,
                            "sigma_layer1": sigma_layer1.detach().cpu(),
                            "validity_mask": validity_mask.detach().cpu(),
                            "n_cands": len(cands),
                        },
                        p,
                    )
                    _LOG.info("cube_debug dumped to %s", p)
                except Exception as exc:
                    _LOG.warning("cube_debug dump failed cid=%d: %s", cube_id, exc)
        except Exception as exc:
            _LOG.warning("cube_debug log failed cid=%d: %s", cube_id, exc)

    def prefetch_build(self, slot: CubeRingSlot) -> PrefetchedCube:
        """Build one cube on the prefetch stream when available.

        The returned tensors remain valid until the next build reuses
        the internal GpuImager output buffer, so callers should consume
        the prefetched cube before launching another build on the same
        pipeline instance.
        """
        t0 = time.perf_counter_ns()
        if self._prefetch_stream is None:
            cube, validity_mask = self._build_cube(slot)
            return PrefetchedCube(
                slot=slot,
                cube=cube,
                validity_mask=validity_mask,
                build_start_ns=t0,
                build_ready_event=None,
            )
        with torch.cuda.stream(self._prefetch_stream):
            cube, validity_mask = self._build_cube(slot)
            ready = torch.cuda.Event(enable_timing=False)
            ready.record(self._prefetch_stream)
        return PrefetchedCube(
            slot=slot,
            cube=cube,
            validity_mask=validity_mask,
            build_start_ns=t0,
            build_ready_event=ready,
        )

    def process_prefetched(self, prefetched: PrefetchedCube) -> CubePipelineResult:
        """Run Layer-1 + detector on a prefetched cube."""
        slot = prefetched.slot
        if prefetched.build_ready_event is not None:
            torch.cuda.current_stream(self._device).wait_event(
                prefetched.build_ready_event
            )
        t1 = time.perf_counter_ns()
        cube_norm, sigma_layer1 = self._layer1_normalise(prefetched.cube)
        t2 = time.perf_counter_ns()
        with torch.no_grad():
            cands = self.detector.forward(
                cube_norm,
                prefetched.validity_mask,
                sigma_layer1,
                event_specnum=int(slot.specnum_start),
                fine_dm_pc_cm3=self._fine_dm_pc_cm3,
            )
        t3 = time.perf_counter_ns()
        self._debug_log_cube_stats(
            slot.cube_id, prefetched.cube, cube_norm, sigma_layer1,
            prefetched.validity_mask, cands,
        )
        timings = {
            "build_cube": t1 - prefetched.build_start_ns,
            "layer1_norm": t2 - t1,
            "detector_forward": t3 - t2,
            "total": t3 - prefetched.build_start_ns,
        }
        return CubePipelineResult(
            cube_id=slot.cube_id,
            specnum_start=slot.specnum_start,
            cube=cube_norm,
            sigma_layer1=sigma_layer1,
            validity_mask=prefetched.validity_mask,
            candidates=cands,
            stage_timings_ns=timings,
        )

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
                fine_dm_pc_cm3=self._fine_dm_pc_cm3,
            )
        t3 = time.perf_counter_ns()
        timings = {
            "build_cube": t1 - t0,
            "layer1_norm": t2 - t1,
            "detector_forward": t3 - t2,
            "total": t3 - t0,
        }
        self._debug_log_cube_stats(
            slot.cube_id, cube, cube_norm, sigma_layer1, validity_mask, cands,
        )
        return CubePipelineResult(
            cube_id=slot.cube_id,
            specnum_start=slot.specnum_start,
            cube=cube_norm,
            sigma_layer1=sigma_layer1,
            validity_mask=validity_mask,
            candidates=cands,
            stage_timings_ns=timings,
        )
