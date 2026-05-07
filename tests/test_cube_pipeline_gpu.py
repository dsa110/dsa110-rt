"""Tests for the CubePipeline GPU image backend (Chunk 8 wiring).

The GPU backend wraps ``image.imager_gpu.GpuImager`` (the production
fused dequant + combine + cuFFT-cfp16 ifft2 + edge-mask path,
post-D25). These tests verify:

  1. CPU and GPU backends agree on the cube up to fp16 ULP × scale
     differences for a synthetic dispersed pulse (locks the §3.6.3
     sign-convention parity at the **pipeline** level — the
     kernel-level lock-in lives in
     ``test_fused_combine_cuda::test_fused_combine_matches_combine_chgroups``).
  2. The GpuImager is built lazily (no allocation before first cube).
  3. Subsequent cubes reuse the same GpuImager workspace (no re-build).
  4. CPU-only ``image_backend='gpu'`` raises a clean ValueError.

Skipped on cuda-less CI; the test is gated by ``torch.cuda.is_available()``.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.constants import N_CHGROUP  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.fine_dm.combiner import compute_time_shift_search  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import CubeRingSlot  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dispersed_slot(
    *,
    t_det: int = 64,
    n_fdm: int = 4,
    n_grid: int = 16,
    true_dm: float = 50.0,
    t_int_search_us: float = 524.288,
    t_15: int = 50,
    pulse_amp: int = 100,
) -> Tuple[CubeRingSlot, np.ndarray]:
    """Build a CubeRingSlot with a dispersed unit pulse landing at the
    chgroup-15 burst-time ``t_15``. ``true_dm`` is the index-0 fdm
    trial; the surrounding trials are at ±0.5 pc/cc.
    """
    fine_dm = np.array(
        [true_dm + (f - n_fdm // 2) * 0.5 for f in range(n_fdm)],
        dtype=np.float64,
    )
    coarse_dm = np.array([0.0], dtype=np.float64)
    fine_to_coarse = np.zeros(n_fdm, dtype=np.int64)
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        t_int_search_us=t_int_search_us,
    )
    # T_stream must accommodate t_det + max_shift (chgroup 0 needs the
    # most lookback so its t-shift is non-negative across the cube).
    max_shift = int(table.shifts.max())
    t_stream = t_det + max_shift + 4

    target_fdm = np.argmin(np.abs(fine_dm - true_dm))
    shifts_at_true_dm = table.shifts[target_fdm]
    # Pulse stream-time per chgroup (so coherent dedispersion lands
    # at cube-time t_15 under §3.6.3's MINUS convention).
    t_burst_g = t_15 - shifts_at_true_dm  # one entry per chgroup

    streams = {}
    l_b, m_b = n_grid // 2 - 1, n_grid // 2 + 1   # off-center
    for g in range(N_CHGROUP):
        s = np.zeros((t_stream, n_grid, n_grid), dtype=np.complex64)
        t_g = int(t_burst_g[g])
        if 0 <= t_g < t_stream:
            s[t_g, l_b, m_b] = pulse_amp
        streams[g] = s

    validity = np.ones((t_det, n_fdm), dtype=np.bool_)
    slot = CubeRingSlot(
        cube_id=0,
        specnum_start=0,
        per_chgroup_streams=streams,
        time_shift_table=table,
        validity_mask=validity,
        n_fdm_in_cube=n_fdm,
        t_det=t_det,
        n_grid=n_grid,
    )
    return slot, np.asarray([target_fdm], dtype=np.int64)


def _make_detector(*, dtype: torch.dtype, device: str) -> DeterministicDetector:
    return DeterministicDetector(
        threshold_sigma=999.0,   # do not emit candidates from these tests
        detector_version="v1.M5",
        search_node_id=0,
        gpu_half=0,
        dtype=dtype,
        device=torch.device(device),
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_gpu_backend_requires_cuda_device() -> None:
    """image_backend='gpu' with device='cpu' raises a clean ValueError."""
    cfg = CubePipelineConfig(
        n_grid=16,
        device="cpu",
        cube_dtype=torch.float16,
        image_backend="gpu",
    )
    det = _make_detector(dtype=torch.float16, device="cpu")
    with pytest.raises(ValueError, match=r"requires a cuda device"):
        CubePipeline(config=cfg, detector=det)


def test_gpu_backend_complex_dtype_must_match_cube_dtype() -> None:
    """complex32 ⇒ float16; complex64 ⇒ float32."""
    with pytest.raises(ValueError, match=r"complex32 requires cube_dtype=float16"):
        CubePipelineConfig(
            n_grid=16,
            device="cuda",
            cube_dtype=torch.float32,        # mismatch
            gpu_complex_dtype=torch.complex32,
            image_backend="gpu",
        )
    with pytest.raises(ValueError, match=r"complex64 requires cube_dtype=float32"):
        CubePipelineConfig(
            n_grid=16,
            device="cuda",
            cube_dtype=torch.float16,        # mismatch
            gpu_complex_dtype=torch.complex64,
            image_backend="gpu",
        )


def test_image_backend_value_check() -> None:
    with pytest.raises(ValueError, match=r"image_backend="):
        CubePipelineConfig(n_grid=16, image_backend="gpu_v2")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CUDA-gated equivalence + lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CubePipeline GPU backend requires cuda",
)
def test_gpu_imager_built_lazily_and_reused() -> None:
    cfg = CubePipelineConfig(
        n_grid=16,
        edge_mask_kernel_support=3,
        device="cuda",
        cube_dtype=torch.float16,
        gpu_complex_dtype=torch.complex32,
        image_backend="gpu",
    )
    det = _make_detector(dtype=torch.float16, device="cuda")
    pipe = CubePipeline(config=cfg, detector=det)
    assert pipe.gpu_imager is None, "GpuImager must be lazy-built"

    slot, _ = _build_dispersed_slot(t_det=32, n_fdm=4, n_grid=16, t_15=20)
    res1 = pipe.process(slot)
    assert res1.cube.is_cuda
    assert res1.cube.dtype == torch.float16
    assert res1.cube.shape == (32, 4, 16, 16)
    imager_after_first = pipe.gpu_imager
    assert imager_after_first is not None

    res2 = pipe.process(slot)
    assert pipe.gpu_imager is imager_after_first, (
        "GpuImager must be reused across cubes"
    )
    # Cubes must be cloned (not the same backing tensor as imager.output_cube).
    assert res1.cube.data_ptr() != res2.cube.data_ptr(), (
        "process() must clone output_cube so consecutive cubes don't alias"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CubePipeline GPU backend requires cuda",
)
def test_cpu_and_gpu_backends_agree_on_dispersed_pulse() -> None:
    """CPU and GPU backends both peak at the same (t, fdm, l, m) for
    a synthetic dispersed pulse (locks the §3.6.3 sign-convention
    parity at the pipeline level).

    Amplitude note: the GPU backend round-trips through cint8 with a
    global max-abs scale (D25's ``transport.quantize.quantise_streams_global_cint8``),
    so the GPU cube is the CPU cube multiplied by that scale. The
    test therefore asserts:
      * Same (t, fdm, l, m) peak location across both backends.
      * GPU peak / CPU peak ratio matches the quantise scale within
        a small tolerance (cint8 rounding error is ≤ 1 ULP per cell).
    """
    n_grid = 16
    t_det = 64
    n_fdm = 4
    t_15 = 40
    pulse_amp = 100
    quantise_target_max = 120

    slot, target_fdm = _build_dispersed_slot(
        t_det=t_det, n_fdm=n_fdm, n_grid=n_grid, t_15=t_15,
        pulse_amp=pulse_amp,
    )
    target_fdm_idx = int(target_fdm[0])

    cpu_cfg = CubePipelineConfig(
        n_grid=n_grid,
        edge_mask_kernel_support=3,
        device="cpu",
        cube_dtype=torch.float32,
        image_backend="cpu",
    )
    cpu_det = _make_detector(dtype=torch.float32, device="cpu")
    cpu_pipe = CubePipeline(config=cpu_cfg, detector=cpu_det)
    cpu_res = cpu_pipe.process(slot)
    cpu_cube = cpu_res.cube.cpu().numpy()

    gpu_cfg = CubePipelineConfig(
        n_grid=n_grid,
        edge_mask_kernel_support=3,
        device="cuda",
        cube_dtype=torch.float32,
        gpu_complex_dtype=torch.complex64,    # cf32 audit path
        image_backend="gpu",
        quantise_target_max=quantise_target_max,
    )
    gpu_det = _make_detector(dtype=torch.float32, device="cuda")
    gpu_pipe = CubePipeline(config=gpu_cfg, detector=gpu_det)
    gpu_res = gpu_pipe.process(slot)
    gpu_cube = gpu_res.cube.cpu().numpy()

    # Same peak time-index across both backends (§3.6.3 sign convention
    # parity).
    def _peak_idx(cube: np.ndarray, fdm: int):
        plane = cube[:, fdm]
        flat_idx = int(np.argmax(plane.reshape(-1)))
        t_idx, lm_idx = divmod(flat_idx, plane.shape[1] * plane.shape[2])
        l_idx, m_idx = divmod(lm_idx, plane.shape[2])
        return t_idx, l_idx, m_idx

    cpu_peak = _peak_idx(cpu_cube, target_fdm_idx)
    gpu_peak = _peak_idx(gpu_cube, target_fdm_idx)
    assert cpu_peak[0] == t_15, f"CPU peak t={cpu_peak[0]} != t_15={t_15}"
    assert gpu_peak[0] == t_15, f"GPU peak t={gpu_peak[0]} != t_15={t_15}"
    assert cpu_peak == gpu_peak, (
        f"CPU peak {cpu_peak} != GPU peak {gpu_peak} "
        "(§3.6.3 sign-convention parity)"
    )

    # GPU / CPU amplitude ratio must equal the quantise scale within
    # cint8 rounding tolerance.
    cpu_pk_val = float(cpu_cube[cpu_peak[0], target_fdm_idx,
                                cpu_peak[1], cpu_peak[2]])
    gpu_pk_val = float(gpu_cube[gpu_peak[0], target_fdm_idx,
                                gpu_peak[1], gpu_peak[2]])
    expected_scale = quantise_target_max / pulse_amp     # 120 / 100 = 1.2
    ratio = gpu_pk_val / cpu_pk_val
    assert abs(ratio - expected_scale) / expected_scale < 0.05, (
        f"GPU/CPU peak ratio {ratio:.4f} != quantise scale "
        f"{expected_scale:.4f} (tol 5%)"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CubePipeline GPU backend requires cuda",
)
def test_gpu_pipeline_round_trip_with_detector() -> None:
    """Smoke: GPU pipeline + DeterministicDetector at a tiny geometry
    completes without raising and returns a valid CubePipelineResult.
    """
    cfg = CubePipelineConfig(
        n_grid=16,
        edge_mask_kernel_support=3,
        device="cuda",
        cube_dtype=torch.float16,
        gpu_complex_dtype=torch.complex32,
        image_backend="gpu",
    )
    det = _make_detector(dtype=torch.float16, device="cuda")
    pipe = CubePipeline(config=cfg, detector=det)

    slot, _ = _build_dispersed_slot(t_det=32, n_fdm=4, n_grid=16)
    res = pipe.process(slot)
    assert res.cube_id == 0
    assert res.specnum_start == 0
    assert res.cube.shape == (32, 4, 16, 16)
    assert res.cube.dtype == torch.float16
    assert res.validity_mask.shape == (32, 4)
    assert res.validity_mask.dtype == torch.bool
    # threshold_sigma=999 ⇒ no candidates emitted on this synthetic input.
    assert res.candidates == []
    # Stage timings populated.
    for k in ("build_cube", "layer1_norm", "detector_forward", "total"):
        assert k in res.stage_timings_ns and res.stage_timings_ns[k] > 0
