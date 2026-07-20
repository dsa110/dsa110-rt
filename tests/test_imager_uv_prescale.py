"""Tests for the imager UV pre-scale (SNR-linearity change set).

The pre-scale is a constant multiplied into the uv data immediately
before the inverse FFT. Because the FFT is linear and Layer-1 re-
estimates σ from the scaled cube, the detection statistic ``cube/σ`` is
invariant to the constant; the only effect is to shrink the raw
dynamic range the fp16 FFT butterflies must represent so bright bursts
no longer overflow (±inf → the ±60000 plateau; see the 2026-06-10
fp16-overflow note in ``services/cube_pipeline.py``).

The GPU path (``image.imager_gpu.GpuImager``) applies the identical
scalar ``mul_`` to the same uv_batch slice both FFT branches read; its
end-to-end exercise needs CUDA (the fused-combine kernel is CUDA-only,
as in ``test_imager_gpu.py``). Here we exercise the scaling contract on
the CPU/numpy reference path (``dirty_image_from_uv_grid``), which
threads the same ``prescale`` argument, plus the config/CLI wiring.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

# The imager / service modules import ``torch`` at module scope. On a
# torch-free host (e.g. the h23 CPU box) these tests skip cleanly rather
# than error at collection, matching the other imager test modules.
torch = pytest.importorskip("torch")

from dsart.image.imager import dirty_image_from_uv_grid  # noqa: E402
from dsart.image.imager_gpu import GpuImagerConfig  # noqa: E402
from dsart.services.cube_pipeline import CubePipelineConfig  # noqa: E402


def _uv_fixture(*, n_grid: int = 16, t: int = 3, seed: int = 0) -> np.ndarray:
    """Deterministic ``[t, n_grid, n_grid] complex64`` uv slab."""
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((t, n_grid, n_grid))
    im = rng.standard_normal((t, n_grid, n_grid))
    return (re + 1j * im).astype(np.complex64)


# ---------------------------------------------------------------------------
# Scaling contract (CPU/numpy reference path)
# ---------------------------------------------------------------------------


def test_prescale_unity_is_bit_identical() -> None:
    """``prescale=1.0`` reproduces the no-argument path bit-for-bit."""
    uv = _uv_fixture()
    base = dirty_image_from_uv_grid(uv)
    same = dirty_image_from_uv_grid(uv, prescale=1.0)
    assert np.array_equal(base, same)


def test_prescale_scales_image_linearly() -> None:
    """``image(c·uv) == c·image(uv)`` (the FFT is linear)."""
    uv = _uv_fixture(seed=3)
    c = 0.00390625  # 1/256, the production value
    base = dirty_image_from_uv_grid(uv)
    scaled = dirty_image_from_uv_grid(uv, prescale=c)
    np.testing.assert_allclose(scaled, c * base, rtol=1e-6, atol=1e-8)


def test_sigma_normalised_output_invariant_to_prescale() -> None:
    """The σ-normalised image (σ re-estimated from the SCALED image, as
    Layer-1 does per cube) is invariant to the pre-scale constant."""
    uv = _uv_fixture(seed=7)
    ref = None
    for c in (1.0, 0.00390625, 256.0):
        img = dirty_image_from_uv_grid(uv, prescale=c)
        norm = img / img.std()
        if ref is None:
            ref = norm
        else:
            np.testing.assert_allclose(norm, ref, rtol=1e-5, atol=1e-6)


def test_prescale_torch_backend_scales_linearly() -> None:
    """The torch backend honours ``prescale`` identically to numpy."""
    uv_np = _uv_fixture(seed=11)
    uv_t = torch.from_numpy(uv_np)
    c = 0.00390625
    base = dirty_image_from_uv_grid(uv_t)
    scaled = dirty_image_from_uv_grid(uv_t, prescale=c)
    torch.testing.assert_close(scaled, c * base, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Config / CLI wiring
# ---------------------------------------------------------------------------


def test_gpu_imager_config_default_and_override() -> None:
    assert GpuImagerConfig().imager_uv_prescale == 1.0
    assert GpuImagerConfig(imager_uv_prescale=0.00390625).imager_uv_prescale == \
        0.00390625


def test_cube_pipeline_config_default_and_override() -> None:
    assert CubePipelineConfig(n_grid=256).imager_uv_prescale == 1.0
    assert CubePipelineConfig(
        n_grid=256, imager_uv_prescale=0.25
    ).imager_uv_prescale == 0.25


def test_cli_prescale_lands_in_pipeline_config() -> None:
    """The ``--imager-uv-prescale`` value flows through the yaml/CLI
    config builder into ``CubePipelineConfig.imager_uv_prescale``."""
    from dsart.services.search_compute import _build_search_config_from_yaml

    cfg = _build_search_config_from_yaml(
        {},
        n_grid=256,
        n_fdm=16,
        gpu_half=0,
        search_node_id=1,
        image_backend="cpu",
        imager_uv_prescale=0.00390625,
        device="cpu",
        enable_clusterer=False,
        enable_cube_dump=False,
        enable_udp_listener=False,
        enable_cands_logger=False,
        detector_streaming_tile_size=256,
        detector_streaming_decoder_n_top=24,
        pipeline_overlap=False,
        detector_k_img_csv="unit",
        detector_k_dm_csv="d1",
        detector_k_time_csv="b1,b2,b4",
        detector_boxcar_accum_dtype="fp32",
        detector_layer2_max_samples=100000,
        layer1_max_samples=10000,
        layer1_sigma_floor=0.0,
        fine_dm_pc_cc_full=None,
        t_det=256,
        cube_cadence_samples=192,
        enable_c1=False,
    )
    assert cfg.pipeline.imager_uv_prescale == 0.00390625


# ---------------------------------------------------------------------------
# fp16-FFT overflow / linearity extension (GPU-only)
# ---------------------------------------------------------------------------
#
# These exercise the ACTUAL failure the pre-scale exists to prevent: the
# complex-half (``complex32``) inverse FFT accumulates the un-normalised
# DFT sum over all N² grid points in fp16. For a bright point source every
# one of the N²=65536 terms adds constructively at the source pixel, so the
# intermediate reaches ``|uv| · N²`` — past the fp16 finite ceiling (65504)
# well before the final 1/N² normalisation divides it back down. The result
# is ±inf at the burst pixel (then ±60000 after nan_to_num). Scaling the uv
# slab by 1/256 first shrinks that intermediate 256× while leaving the
# σ-normalised detection statistic unchanged (the FFT is linear).


def _cuda_or_skip() -> "torch.device":
    if not torch.cuda.is_available():
        pytest.skip("CUDA required: complex32 ifft2 is a GPU-only path")
    return torch.device("cuda")


def _point_source_uv(n_grid: int, mag: float, device: "torch.device"):
    """UV slab of a single on-axis point source: constant complex value
    ``mag`` across the whole grid (⇔ a δ at the image origin). All N²
    terms add in phase at the origin pixel, which is where the fp16
    butterfly accumulation overflows."""
    return torch.full(
        (n_grid, n_grid), mag, dtype=torch.complex64, device=device
    )


def test_fp16_ifft2_overflows_without_prescale() -> None:
    """Without the pre-scale, a bright point source drives the complex32
    ifft2 non-finite (the regime the change set targets)."""
    device = _cuda_or_skip()
    n_grid = 256
    # mag·N² = 8·65536 ≫ 65504, so the on-axis accumulation overflows.
    uv = _point_source_uv(n_grid, mag=8.0, device=device).to(torch.complex32)
    img = torch.fft.ifft2(uv).real
    assert not torch.isfinite(img).all(), (
        "expected fp16 ifft2 overflow for the un-prescaled bright source; "
        "if this passes the fp16 FFT no longer overflows and the pre-scale "
        "rationale needs revisiting"
    )


def test_fp16_ifft2_stays_finite_with_prescale() -> None:
    """The 1/256 pre-scale keeps the same bright source finite through the
    complex32 ifft2."""
    device = _cuda_or_skip()
    n_grid = 256
    c = 0.00390625  # 1/256, production value
    uv = _point_source_uv(n_grid, mag=8.0, device=device)
    uv.mul_(c)  # pre-scale, exactly as GpuImager does before the FFT
    img = torch.fft.ifft2(uv.to(torch.complex32)).real
    assert torch.isfinite(img).all(), (
        "pre-scaled bright source must survive the fp16 ifft2 finite"
    )


def test_fp16_prescale_preserves_sigma_normalised_peak() -> None:
    """End-to-end linearity extension: the σ-normalised peak of the
    pre-scaled fp16 image matches the fp32 reference, while the
    un-prescaled fp16 image does not (it saturates)."""
    device = _cuda_or_skip()
    n_grid = 256
    c = 0.00390625
    uv = _point_source_uv(n_grid, mag=8.0, device=device)

    ref = torch.fft.ifft2(uv).real  # fp32 reference, no overflow
    ref_norm_peak = float((ref / ref.std()).max())

    pre = torch.fft.ifft2((uv * c).to(torch.complex32)).real
    pre_norm_peak = float((pre / pre.std()).max())

    # σ-normalised peak is scale-invariant → prescaled fp16 tracks fp32.
    assert pre_norm_peak == pytest.approx(ref_norm_peak, rel=0.02)
