"""Tests for ``dsart.image.imager_gpu`` — the production GPU
dirty-imager (chunk-8 hardening landing point for D19 / D20 / D21).

Tests fall into two layers:

  (a) Schema / validation: shape + dtype + boundary checks on
      ``GpuImager.process_cube`` (run on any host; CUDA-free portion).

  (b) End-to-end numerical: build a tiny cube, run it through the
      GPU pipeline, verify the output cube's last-axis dtype is the
      configured ``cube_dtype`` and the active-region pixels are
      finite (skipped when no CUDA is available).

A bit-exact end-to-end correctness test against the chunk-6a
numpy reference is intentionally omitted — that comparison is
already covered for the inner combine kernel by
``tests/test_fused_combine_cuda.py`` (5+5 numerical-equivalence
tests). What this file adds is the **wiring** between combine /
ifft2 / mask in the production module, which is best exercised at
the level of "the cube has the right shape + dtype + finite values".
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

from dsart.image.imager_gpu import (
    GpuImager,
    GpuImagerConfig,
    build_default_gpu_imager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GpuImager tests")
    return torch.device("cuda")


def _make_streams(
    *, n_chg: int, t_stream: int, n_grid: int, device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Build a deterministic ``[n_chg, t_stream, 2, n_grid, n_grid] int8``
    cint8 stream (matches the M3 wire payload layout)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    streams = torch.randint(
        low=-127, high=128, size=(n_chg, t_stream, 2, n_grid, n_grid),
        generator=g, dtype=torch.int8,
    )
    return streams.to(device=device)


def _make_time_shifts(
    *, n_fdm: int, n_chgroup: int, t_stream: int, t_det: int,
    device: torch.device, seed: int = 0,
) -> torch.Tensor:
    """Build a deterministic ``[n_fdm, n_chgroup] int32 cuda`` shift
    table within the safe ``[0, t_stream - t_det]`` range.
    """
    rng = np.random.default_rng(seed)
    max_shift = max(0, t_stream - t_det)
    shifts = rng.integers(0, max_shift + 1, size=(n_fdm, n_chgroup))
    return torch.from_numpy(shifts.astype(np.int32)).to(device=device)


# ---------------------------------------------------------------------------
# Schema / validation (CUDA-optional — most checks before the kernel)
# ---------------------------------------------------------------------------


def test_config_resolved_device_falls_back() -> None:
    cfg = GpuImagerConfig()
    dev = cfg.resolved_device()
    assert isinstance(dev, torch.device)
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_config_explicit_device_wins() -> None:
    explicit = torch.device("cpu")
    cfg = GpuImagerConfig(device=explicit)
    assert cfg.resolved_device() == explicit


def test_build_allocates_workspace() -> None:
    """``GpuImager.build`` allocates the documented buffers at the
    configured shape + dtype, and produces no compute side effects.
    """
    device = _cuda_or_skip()
    cfg = GpuImagerConfig(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3,
        cube_dtype=torch.float16, complex_dtype=torch.complex32,
        device=device,
    )
    imager = GpuImager.build(cfg)
    assert imager.device == device
    assert imager.edge_mask_real.shape == (8, 8)
    assert imager.edge_mask_real.dtype == torch.float16
    assert imager.output_cube.shape == (4, 2, 8, 8)
    assert imager.output_cube.dtype == torch.float16
    assert imager.uv_slab.shape == (4, 8, 8)
    assert imager.uv_slab.dtype == torch.complex32
    assert imager.img_slab_real.shape == (4, 8, 8)


def test_build_default_gpu_imager_at_operator_geometry() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=64,  # smaller than production for test speed
        t_det=16, n_fdm=4, n_chgroup=4, device=device,
    )
    assert imager.config.n_grid == 64
    assert imager.config.t_det == 16
    assert imager.config.n_fdm == 4
    assert imager.config.n_chgroup == 4
    assert imager.config.cube_dtype == torch.float16
    assert imager.config.complex_dtype == torch.complex32


def test_process_cube_rejects_wrong_streams_dtype() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    bad = torch.zeros((3, 8, 2, 8, 8), dtype=torch.float32, device=device)
    shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="dtype"):
        imager.process_cube(streams_cint8=bad, time_shifts_gpu=shifts)


def test_process_cube_rejects_wrong_streams_layout() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    # Missing the inner-2 split-plane axis.
    bad = torch.zeros((3, 8, 8, 8), dtype=torch.int8, device=device)
    shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="ndim"):
        imager.process_cube(streams_cint8=bad, time_shifts_gpu=shifts)


def test_process_cube_rejects_wrong_chgroup_count() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    bad = torch.zeros((4, 8, 2, 8, 8), dtype=torch.int8, device=device)
    shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="N_chgroup"):
        imager.process_cube(streams_cint8=bad, time_shifts_gpu=shifts)


def test_process_cube_rejects_wrong_grid() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    bad = torch.zeros((3, 8, 2, 16, 16), dtype=torch.int8, device=device)
    shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="grid"):
        imager.process_cube(streams_cint8=bad, time_shifts_gpu=shifts)


def test_process_cube_rejects_inner_axis_not_2() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    bad = torch.zeros((3, 8, 3, 8, 8), dtype=torch.int8, device=device)
    shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="inner-2-axis"):
        imager.process_cube(streams_cint8=bad, time_shifts_gpu=shifts)


def test_process_cube_rejects_short_t_stream() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=8, n_fdm=2, n_chgroup=3, device=device,
    )
    bad = torch.zeros((3, 4, 2, 8, 8), dtype=torch.int8, device=device)
    shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="T_stream"):
        imager.process_cube(streams_cint8=bad, time_shifts_gpu=shifts)


def test_process_cube_rejects_wrong_shifts_dtype() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    streams = torch.zeros((3, 8, 2, 8, 8), dtype=torch.int8, device=device)
    bad = torch.zeros((2, 3), dtype=torch.int64, device=device)
    with pytest.raises(ValueError, match="time_shifts_gpu.dtype"):
        imager.process_cube(streams_cint8=streams, time_shifts_gpu=bad)


def test_process_cube_rejects_wrong_shifts_shape() -> None:
    device = _cuda_or_skip()
    imager = build_default_gpu_imager(
        n_grid=8, t_det=4, n_fdm=2, n_chgroup=3, device=device,
    )
    streams = torch.zeros((3, 8, 2, 8, 8), dtype=torch.int8, device=device)
    bad = torch.zeros((4, 3), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="shape"):
        imager.process_cube(streams_cint8=streams, time_shifts_gpu=bad)


# ---------------------------------------------------------------------------
# End-to-end numerical (CUDA-only)
# ---------------------------------------------------------------------------


def test_process_cube_writes_finite_output_cube() -> None:
    """Run a small cube end-to-end. Output dtype + shape are correct
    and the active-region pixels are finite (no NaN / Inf from FFT
    underflow on cfp16).
    """
    device = _cuda_or_skip()
    n_chg, n_fdm, t_det, t_stream, n_grid = 4, 4, 16, 32, 32
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams = _make_streams(
        n_chg=n_chg, t_stream=t_stream, n_grid=n_grid, device=device, seed=1,
    )
    shifts = _make_time_shifts(
        n_fdm=n_fdm, n_chgroup=n_chg, t_stream=t_stream, t_det=t_det,
        device=device, seed=1,
    )
    out = imager.process_cube(streams_cint8=streams, time_shifts_gpu=shifts)
    assert out is imager.output_cube
    assert out.shape == (t_det, n_fdm, n_grid, n_grid)
    assert out.dtype == torch.float16
    out_f32 = out.to(torch.float32)
    assert torch.isfinite(out_f32).all().item()
    # Edge mask must zero the outer ring (npad ≥ 4 at kernel_support=5,
    # n_grid=32 → at minimum the corner pixel is masked).
    assert (out_f32[:, :, 0, 0] == 0.0).all().item()
    assert (out_f32[:, :, -1, -1] == 0.0).all().item()
    # Cube can't be all zeros if streams are non-zero.
    assert out_f32.abs().max().item() > 0.0


def test_process_cube_zero_streams_produce_zero_cube() -> None:
    """All-zero cint8 input → all-zero output (sanity check on the
    fused-combine accumulation + FFT-of-zero behaviour).
    """
    device = _cuda_or_skip()
    n_chg, n_fdm, t_det, t_stream, n_grid = 4, 4, 16, 32, 32
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams = torch.zeros(
        (n_chg, t_stream, 2, n_grid, n_grid), dtype=torch.int8, device=device,
    )
    shifts = torch.zeros((n_fdm, n_chg), dtype=torch.int32, device=device)
    out = imager.process_cube(streams_cint8=streams, time_shifts_gpu=shifts)
    assert (out == 0).all().item()


def test_process_cube_reuses_workspace_across_calls() -> None:
    """Two consecutive calls produce the same output for the same input
    (i.e. workspace state is not contaminated across cubes).
    """
    device = _cuda_or_skip()
    n_chg, n_fdm, t_det, t_stream, n_grid = 4, 2, 8, 16, 16
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams = _make_streams(
        n_chg=n_chg, t_stream=t_stream, n_grid=n_grid, device=device, seed=2,
    )
    shifts = _make_time_shifts(
        n_fdm=n_fdm, n_chgroup=n_chg, t_stream=t_stream, t_det=t_det,
        device=device, seed=2,
    )
    out1 = imager.process_cube(
        streams_cint8=streams, time_shifts_gpu=shifts,
    ).clone()
    out2 = imager.process_cube(
        streams_cint8=streams, time_shifts_gpu=shifts,
    ).clone()
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Chunk-8(c) — per-chgroup (scale, offset) plumbing through GpuImager
# ---------------------------------------------------------------------------


def test_process_cube_unit_calibration_matches_default() -> None:
    """``process_cube`` with explicit unit-scale + zero-offset arrays
    matches the default (None) call cell-for-cell. Verifies the calib
    dispatch is a no-op when the calibration is trivial.
    """
    device = _cuda_or_skip()
    n_chg, n_fdm, t_det, t_stream, n_grid = 4, 4, 16, 32, 32
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams = _make_streams(
        n_chg=n_chg, t_stream=t_stream, n_grid=n_grid, device=device, seed=5,
    )
    shifts = _make_time_shifts(
        n_fdm=n_fdm, n_chgroup=n_chg, t_stream=t_stream, t_det=t_det,
        device=device, seed=5,
    )
    out_default = imager.process_cube(
        streams_cint8=streams, time_shifts_gpu=shifts,
    ).clone()

    scales = torch.ones((n_chg,), dtype=torch.float32, device=device)
    offsets = torch.zeros((n_chg,), dtype=torch.float32, device=device)
    out_calib = imager.process_cube(
        streams_cint8=streams, time_shifts_gpu=shifts,
        chgroup_scales=scales,
        chgroup_offsets_re=offsets,
        chgroup_offsets_im=offsets,
    ).clone()
    # fp32 (1*x = int(x) cast to fp32) reduction matches int32 reduction
    # bit-exact for ≤16 chgroups; the iFFT2 + edge mask are deterministic
    # so the cubes must be byte-identical.
    assert torch.equal(out_default, out_calib)


def test_process_cube_constant_scale_scales_output_linearly() -> None:
    """``process_cube`` with a uniform scale=k applied to every chgroup
    produces an output cube that is a factor of ~k larger than the
    default unit-scale output (linear in the dequant gain).
    """
    device = _cuda_or_skip()
    n_chg, n_fdm, t_det, t_stream, n_grid = 4, 2, 8, 16, 16
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams = _make_streams(
        n_chg=n_chg, t_stream=t_stream, n_grid=n_grid, device=device, seed=7,
    )
    shifts = _make_time_shifts(
        n_fdm=n_fdm, n_chgroup=n_chg, t_stream=t_stream, t_det=t_det,
        device=device, seed=7,
    )
    out_default = imager.process_cube(
        streams_cint8=streams, time_shifts_gpu=shifts,
    ).clone().to(torch.float32)

    k = 2.5
    scales = torch.full((n_chg,), k, dtype=torch.float32, device=device)
    out_scaled = imager.process_cube(
        streams_cint8=streams, time_shifts_gpu=shifts, chgroup_scales=scales,
    ).clone().to(torch.float32)

    # Where the default output is meaningfully non-zero (away from the
    # edge mask), the scaled output should be k * default. Use a
    # masked relative-error check; cells near zero get absolute atol.
    abs_default = out_default.abs()
    threshold = 1e-2 * abs_default.max()
    mask = abs_default > threshold
    if mask.any():
        ratio = (out_scaled[mask] / out_default[mask]).abs()
        # cf16 → fp32 path has ~3% relative error from the cf16 round
        # trip; tolerate 5%.
        assert torch.all((ratio - k).abs() < k * 0.05).item(), (
            f"scale-by-{k} produced unexpected ratio: "
            f"min={ratio.min().item():.3f} max={ratio.max().item():.3f}"
        )


def test_process_cube_rejects_calibration_wrong_shape() -> None:
    """Per-chgroup arrays must be ``(N_chgroup,)``."""
    device = _cuda_or_skip()
    n_chg, n_fdm, t_det, t_stream, n_grid = 4, 2, 8, 16, 16
    imager = build_default_gpu_imager(
        n_grid=n_grid, t_det=t_det, n_fdm=n_fdm, n_chgroup=n_chg,
        device=device,
    )
    streams = _make_streams(
        n_chg=n_chg, t_stream=t_stream, n_grid=n_grid, device=device,
    )
    shifts = _make_time_shifts(
        n_fdm=n_fdm, n_chgroup=n_chg, t_stream=t_stream, t_det=t_det,
        device=device,
    )
    bad_scales = torch.ones((n_chg + 1,), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match=r"chgroup_scales\.shape"):
        imager.process_cube(
            streams_cint8=streams, time_shifts_gpu=shifts,
            chgroup_scales=bad_scales,
        )
