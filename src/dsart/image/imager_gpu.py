"""src/dsart/image/imager_gpu.py — production GPU dirty-imager
(plan §3.6.11 + §4.4 lines 1513-1515; chunk-8 hardening of D19/D20/D21).

The chunk-6a ``image/imager.py`` module shipped a CPU/numpy + cf32-
torch placeholder good enough for unit tests. This module is the
production-grade GPU path that the live ``services/search_compute.py``
service consumes. It composes:

  1. **Fused dequant + per-fdm combine** (D21): a single NVRTC kernel
     reads the M3 cint8 streams (``[N_chg, T_stream, 2, N_grid, N_grid]``
     split-plane re/im, native wire payload) at the per-fdm time
     shifts, accumulates across the 16 chgroups in int32 registers
     (exact for N_chg ≤ 16), and writes a cfp16 ``[T_det, N, N]`` uv
     slab in one kernel pass. cint8 input → cfp16 output is half the
     memory traffic of the chunk-6c-fused (cfp16 input) path.

  2. **cuFFT-cfp16 ``ifft2`` + fftshift** (D19): on h01 / RTX 2080 Ti
     this runs at ~5.2 µs / 256² FFT2 — within ~1.2× of the cuFFT
     theoretical roofline. With the chunk-8 fused combine in front,
     the imager is FFT-bound (combine ≈ 45% / ifft2 ≈ 44% / mask ≈ 11%
     at T_det=256 / N_fdm=32 / N_grid=256).

  3. **Edge mask** (§3.5 G11; reused from chunk-6a): multiplicative
     ``[N_grid, N_grid] cube_dtype`` mask zeroes the outer kernel-
     support / FFT-wraparound ring + the DC cell.

End-to-end production-geometry headline (h01 GPU 1, T_det=256,
N_fdm=32, N_grid=256, cfp16 output):
**9.79 cubes/s** (97.7 ms p50 total per cube; clears the plan §8 8
cubes/s target by 22%) — see ``D21`` in ``M5_PLAN_FIXES.md`` and
``bench/imager_only_gpu_results.md`` for the full A/B/C sweep.

Workspace ownership:
  - The live service builds **one** :class:`GpuImager` at startup
    (eats the cuFFT plan + workspace allocs at construction time, not
    in the hot loop) and re-uses ``output_cube`` / ``uv_slab`` /
    ``img_slab_real`` across every cube. The caching allocator
    handles the per-call transients.
  - The cint8 stream tensor is **caller-owned**: the M3 → M5 RX-ring
    pre-stages a single rolling buffer per service lifetime; this
    module only reads from it (the F26 sparse-COO scatter happens
    inside the fused kernel, not on a host-staging copy).

The pure-CPU/numpy chunk-6a ``image/imager.py`` module is retained
for unit tests + the chunk-7 captured-mode loader's cf64 scatter
path (pre-cint8-quantisation reference); they are NOT redundant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from dsart.image.imager import compute_edge_mask

_LOG = logging.getLogger(__name__)

__all__ = [
    "GpuImager",
    "GpuImagerConfig",
    "build_default_gpu_imager",
]


@dataclass
class GpuImagerConfig:
    """Production GPU imager configuration.

    All defaults match the operator-pinned v1 deployment geometry
    (D21: T_det=256, N_fdm=32, N_grid=256, cfp16 output).
    """
    n_grid: int = 256
    t_det: int = 256
    n_fdm: int = 32
    n_chgroup: int = 16
    kernel_support: int = 5
    sigma_l_pix: Optional[float] = None
    envelope_threshold: float = 0.5
    drop_dc: bool = True
    cube_dtype: torch.dtype = torch.float16
    complex_dtype: torch.dtype = torch.complex32
    device: Optional[torch.device] = None

    def resolved_device(self) -> torch.device:
        """Return ``self.device`` or fall back to ``cuda`` / ``cpu``."""
        if self.device is not None:
            return self.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@dataclass
class GpuImager:
    """Stateful GPU imager. Owns workspace + edge mask + the cuFFT
    plan (implicitly via the first ``ifft2`` call on a fixed-shape
    tensor — torch caches the plan).

    Build once per service lifetime via :meth:`build`; call
    :meth:`process_cube` in the hot loop. The output cube is written
    in-place to :attr:`output_cube` and also returned.
    """
    config: GpuImagerConfig
    device: torch.device
    edge_mask_real: torch.Tensor   # [N_grid, N_grid] cube_dtype
    output_cube: torch.Tensor      # [T_det, N_fdm, N_grid, N_grid] cube_dtype
    uv_slab: torch.Tensor          # [T_det, N_grid, N_grid] complex_dtype
    img_slab_real: torch.Tensor    # [T_det, N_grid, N_grid] cube_dtype

    @classmethod
    def build(cls, config: GpuImagerConfig) -> "GpuImager":
        """Allocate persistent GPU buffers + edge mask.

        Performs no compute; safe to call inside service init.
        """
        device = config.resolved_device()
        edge = compute_edge_mask(
            n_grid=config.n_grid,
            kernel_support=config.kernel_support,
            sigma_l_pix=config.sigma_l_pix,
            envelope_threshold=config.envelope_threshold,
            drop_dc=config.drop_dc,
        )
        edge_t = torch.from_numpy(edge).to(
            device=device, dtype=config.cube_dtype,
        )
        output = torch.empty(
            (config.t_det, config.n_fdm, config.n_grid, config.n_grid),
            dtype=config.cube_dtype, device=device,
        )
        uv = torch.empty(
            (config.t_det, config.n_grid, config.n_grid),
            dtype=config.complex_dtype, device=device,
        )
        img = torch.empty(
            (config.t_det, config.n_grid, config.n_grid),
            dtype=config.cube_dtype, device=device,
        )
        _LOG.info(
            "GpuImager.build: device=%s N_grid=%d T_det=%d N_fdm=%d "
            "cube_dtype=%s complex_dtype=%s",
            device, config.n_grid, config.t_det, config.n_fdm,
            config.cube_dtype, config.complex_dtype,
        )
        return cls(
            config=config, device=device,
            edge_mask_real=edge_t, output_cube=output,
            uv_slab=uv, img_slab_real=img,
        )

    # ------------------------------------------------------------------
    # Cube processing
    # ------------------------------------------------------------------

    def process_cube(
        self,
        *,
        streams_cint8: torch.Tensor,           # [N_chg, T_stream, 2, N, N] int8
        time_shifts_gpu: torch.Tensor,         # [N_fdm, N_chgroup] int32 cuda
        chgroup_scales: Optional[torch.Tensor] = None,
        chgroup_offsets_re: Optional[torch.Tensor] = None,
        chgroup_offsets_im: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full GPU imager for one cube.

        Per-fdm pipeline:
          1. Fused dequant + 16-chgroup index-shifted sum into ``uv_slab``
             (single CUDA kernel; reads cint8 streams, writes
             ``[T_det, N, N]`` complex_dtype). When the optional
             ``chgroup_scales`` / ``chgroup_offsets_*`` arrays are
             provided, the kernel applies the per-chgroup
             ``z[g] = scale[g] * cint8[g] + offset[g]`` calibration
             inline (chunk-8(c)); otherwise it runs the unit-scale
             int32-accumulate fast path (Layer-1 σ-clip downstream
             absorbs any constant scaling).
          2. ``Re(fftshift(ifft2(uv_slab)))`` → ``img_slab_real``.
          3. ``img_slab_real * edge_mask_real`` → ``output_cube[:, f]``.

        Args:
            streams_cint8: M3 wire payload, ``[N_chgroup, T_stream, 2,
                N_grid, N_grid] int8``. The 2-axis is split-plane
                (real / imaginary). Caller owns this tensor; this
                method only reads from it.
            time_shifts_gpu: ``[N_fdm, N_chgroup] int32`` per-(fdm,
                chgroup) sample shift (the fine-DM trial offsets), as
                produced by ``fine_dm/combiner.py::compute_time_shift_search``.
                Per §3.6.3 the fused kernel reads chgroup ``g``'s
                stream at ``streams_cint8[g, t - shifts[f, g]]`` for
                output cube-time ``t`` (zero-fill outside
                ``[0, T_stream)``). For a coherently dispersed pulse
                whose chgroup-15 sample lands at stream-time ``t_15``,
                the dedispersed cube peaks at cube-time ``t_15`` (the
                chgroup-15 row of the shift table is identically zero
                by §3.6.3).
            chgroup_scales: optional ``[N_chgroup] float32`` cuda
                tensor of per-chgroup multiplicative scales (chunk-
                8(c)). When provided, every fdm trial uses the same
                calibration vector (per-chgroup gain is constant
                within a cube). ``None`` → unit scale.
            chgroup_offsets_re: optional ``[N_chgroup] float32`` cuda
                tensor of per-chgroup real-part DC offsets. ``None``
                → zeros.
            chgroup_offsets_im: optional ``[N_chgroup] float32`` cuda
                tensor of per-chgroup imag-part DC offsets. ``None``
                → zeros.

        Returns:
            ``self.output_cube`` (``[T_det, N_fdm, N_grid, N_grid]
            cube_dtype``); written in-place.

        Raises:
            ValueError on shape / dtype mismatch.
        """
        from dsart.image.fused_combine_cuda import (
            fused_dequant_combine_per_fdm,
        )

        cfg = self.config
        # --- Validate shapes.
        if streams_cint8.dtype != torch.int8:
            raise ValueError(
                f"streams_cint8.dtype={streams_cint8.dtype}, expected int8 "
                "(M3 sparse-COO cint8 wire payload)"
            )
        if streams_cint8.ndim != 5:
            raise ValueError(
                f"streams_cint8 ndim={streams_cint8.ndim}, expected 5 "
                "([N_chg, T_stream, 2, N_grid, N_grid])"
            )
        n_chg, t_stream, two, n_g, n_g2 = streams_cint8.shape
        if two != 2:
            raise ValueError(
                f"streams_cint8 inner-2-axis={two}, expected 2 "
                "(real / imaginary split planes)"
            )
        if n_g != cfg.n_grid or n_g2 != cfg.n_grid:
            raise ValueError(
                f"streams_cint8 grid {n_g}x{n_g2} != config N_grid={cfg.n_grid}"
            )
        if n_chg != cfg.n_chgroup:
            raise ValueError(
                f"streams_cint8 N_chgroup={n_chg} != config "
                f"N_chgroup={cfg.n_chgroup}"
            )
        if time_shifts_gpu.dtype != torch.int32:
            raise ValueError(
                f"time_shifts_gpu.dtype={time_shifts_gpu.dtype}, "
                "expected int32"
            )
        if time_shifts_gpu.shape != (cfg.n_fdm, cfg.n_chgroup):
            raise ValueError(
                f"time_shifts_gpu.shape={tuple(time_shifts_gpu.shape)}, "
                f"expected ({cfg.n_fdm}, {cfg.n_chgroup})"
            )
        if t_stream < cfg.t_det:
            raise ValueError(
                f"streams_cint8 T_stream={t_stream} < T_det={cfg.t_det}; "
                "no fdm trial can fit"
            )

        # Calibration arrays (optional). Shape / dtype validation
        # happens inside fused_dequant_combine_per_fdm; we bind the
        # variables here so all per-fdm calls share the same tensors.
        for name, cal in (
            ("chgroup_scales", chgroup_scales),
            ("chgroup_offsets_re", chgroup_offsets_re),
            ("chgroup_offsets_im", chgroup_offsets_im),
        ):
            if cal is not None and cal.shape != (cfg.n_chgroup,):
                raise ValueError(
                    f"{name}.shape={tuple(cal.shape)}, "
                    f"expected ({cfg.n_chgroup},)"
                )

        for f in range(cfg.n_fdm):
            fused_dequant_combine_per_fdm(
                streams_cint8,
                time_shifts_gpu[f].contiguous(),
                self.uv_slab,
                scales=chgroup_scales,
                offsets_re=chgroup_offsets_re,
                offsets_im=chgroup_offsets_im,
            )
            img_complex = torch.fft.ifft2(self.uv_slab)
            img_complex = torch.fft.fftshift(img_complex, dim=(-2, -1))
            torch.mul(
                img_complex.real.to(cfg.cube_dtype),
                self.edge_mask_real,
                out=self.img_slab_real,
            )
            self.output_cube[:, f, :, :] = self.img_slab_real

        return self.output_cube


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_gpu_imager(
    *,
    n_grid: int = 256,
    t_det: int = 256,
    n_fdm: int = 32,
    n_chgroup: int = 16,
    cube_dtype: torch.dtype = torch.float16,
    complex_dtype: torch.dtype = torch.complex32,
    device: Optional[torch.device] = None,
) -> GpuImager:
    """Build a :class:`GpuImager` at the operator-pinned v1 geometry.

    Convenience entry point for the live service. All defaults track
    the deployment configuration (D21 in ``M5_PLAN_FIXES.md``).
    """
    return GpuImager.build(GpuImagerConfig(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chgroup,
        cube_dtype=cube_dtype, complex_dtype=complex_dtype, device=device,
    ))
