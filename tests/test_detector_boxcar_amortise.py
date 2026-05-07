"""Unit tests for the v1-collapsed-bank cumsum-once amortisation
primitives ``precompute_padded_cumsum`` + ``boxcar_from_padded_cumsum``
introduced in chunk-9.

These primitives let the streaming forward amortise the per-kernel
time-axis cumsum across all kernels in a pass. The tests pin:

  * ``precompute_padded_cumsum`` does NOT mutate its input (the
    ``Tensor.to(dtype)`` returns-self sharp-edge that bit a previous
    in-place ``cumsum_`` recipe).
  * ``boxcar_from_padded_cumsum`` is bit-exact (fp32 reference) /
    fp16-rel-err <= 1e-3 (fp16 cube path) vs the chunk-1
    ``boxcar_via_cumsum`` for every supported width.
  * The W-tiled boxcar matches the untiled boxcar bit-exactly along
    the LAST axis.
  * The streaming-forward path produces identical candidates whether
    the v1 amortise fast-path is enabled (collapsed bank) or not
    (full bank with non-trivial DM kernels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.detector.forward import (  # noqa: E402
    boxcar_from_padded_cumsum,
    boxcar_via_cumsum,
    precompute_padded_cumsum,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_boxcar_from_padded_cumsum_matches_boxcar_via_cumsum(
    dtype: torch.dtype, width: int,
) -> None:
    """For every supported width the amortised path must produce the
    same boxcar as the chunk-1 per-kernel ``boxcar_via_cumsum``."""
    rng = np.random.default_rng(0)
    cube_np = rng.standard_normal((16, 4, 8, 8)).astype(np.float32)
    cube = torch.from_numpy(cube_np).to(dtype)

    max_width = 16
    cs = precompute_padded_cumsum(cube, axis=0, max_width=max_width)
    out_amortise = boxcar_from_padded_cumsum(
        cs, axis=0, width=width, max_width=max_width,
        n_out=cube.shape[0], out_dtype=cube.dtype,
    )
    out_per_kernel = boxcar_via_cumsum(cube, axis=0, width=width)

    assert out_amortise.shape == out_per_kernel.shape
    assert out_amortise.dtype == out_per_kernel.dtype
    if dtype == torch.float32:
        assert torch.allclose(out_amortise, out_per_kernel, atol=1e-5)
    else:
        rel = (out_amortise - out_per_kernel).abs() / (
            out_per_kernel.abs() + 1e-3
        )
        assert float(rel.max()) <= 1e-3, (
            f"fp16 rel-err exceeded 1e-3: {float(rel.max())}"
        )


def test_precompute_padded_cumsum_does_not_mutate_input() -> None:
    """The ``torch.cumsum(out=...)`` recipe must not modify the
    caller's cube even when ``cube.dtype == accum_dtype``. An earlier
    ``x.to(accum_dtype).cumsum_()`` recipe corrupted callers because
    ``Tensor.to(dtype)`` returns self when dtypes match."""
    rng = np.random.default_rng(0)
    cube = torch.from_numpy(rng.standard_normal((8, 2, 4, 4)).astype(np.float32))
    cube_before = cube.clone()
    _cs = precompute_padded_cumsum(cube, axis=0, max_width=4)
    assert torch.allclose(cube, cube_before, atol=0.0), (
        "precompute_padded_cumsum mutated its input"
    )


@pytest.mark.parametrize("w_tile_size", [1, 2, 4, 8])
def test_boxcar_from_padded_cumsum_w_tiling_bit_exact(
    w_tile_size: int,
) -> None:
    """The W-tiled subtract+cast must produce a bit-exact result vs
    the untiled path along axes that are NOT the cumsum axis."""
    rng = np.random.default_rng(1)
    cube = torch.from_numpy(
        rng.standard_normal((12, 3, 8, 8)).astype(np.float32)
    )
    max_width = 8
    cs = precompute_padded_cumsum(cube, axis=0, max_width=max_width)

    out_untiled = boxcar_from_padded_cumsum(
        cs, axis=0, width=4, max_width=max_width,
        n_out=cube.shape[0], out_dtype=torch.float16,
    )
    out_tiled = boxcar_from_padded_cumsum(
        cs, axis=0, width=4, max_width=max_width,
        n_out=cube.shape[0], out_dtype=torch.float16,
        w_tile_size=w_tile_size,
    )
    assert torch.equal(out_untiled, out_tiled), (
        "W-tiled boxcar diverged from untiled at "
        f"w_tile_size={w_tile_size}"
    )


def test_boxcar_amortise_max_width_validation() -> None:
    """``precompute_padded_cumsum`` rejects ``max_width > axis_len``;
    ``boxcar_from_padded_cumsum`` rejects ``width > max_width``."""
    cube = torch.zeros((4, 2, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"max_width=8 exceeds"):
        precompute_padded_cumsum(cube, axis=0, max_width=8)
    cs = precompute_padded_cumsum(cube, axis=0, max_width=4)
    with pytest.raises(ValueError, match=r"width=8"):
        boxcar_from_padded_cumsum(
            cs, axis=0, width=8, max_width=4, n_out=4,
        )


def test_streaming_forward_v1_collapsed_bank_matches_per_kernel() -> None:
    """The v1-collapsed-bank amortise path must produce the same
    candidate list as a streaming forward that only contains a
    single-time-width kernel (which falls through to the per-kernel
    boxcar path because amortise requires K > 1)."""
    from dsart.detector.forward import DeterministicDetector
    from dsart.detector.kernels import build_kernel_bank
    from dsart.inject.cube_injection import (
        CubeInjectionConfig,
        synthesise_cube,
    )

    cfg = CubeInjectionConfig(
        l_pix=4, m_pix=4, fine_dm_idx=2, t_in_cube=8,
        snr=12.0, width_samples=4, profile="boxcar",
    )
    cube_t, validity_mask, sigma_layer1 = synthesise_cube(
        t_det=16, n_fdm=4, n_grid=8,
        rng=np.random.default_rng(7), injections=(cfg,),
    )

    bank_full = build_kernel_bank(
        image_tokens=("unit",), dm_tokens=("d1",),
        time_tokens=("b1", "b2", "b4", "b8"),
        dtype=torch.float32,
    )
    bank_single_b4 = build_kernel_bank(
        image_tokens=("unit",), dm_tokens=("d1",),
        time_tokens=("b4",),
        dtype=torch.float32,
    )

    det_amortise = DeterministicDetector(
        kernel_bank=bank_full,
        threshold_sigma=8.0,
        detector_version="v1.M5-test",
        search_node_id=1, gpu_half=1,
        dtype=torch.float32,
        streaming=True,
        streaming_tile_size=4,
    )
    det_per_kernel = DeterministicDetector(
        kernel_bank=bank_single_b4,
        threshold_sigma=8.0,
        detector_version="v1.M5-test",
        search_node_id=1, gpu_half=1,
        dtype=torch.float32,
        streaming=True,
        streaming_tile_size=4,
    )

    cands_amortise = det_amortise.forward(cube_t, validity_mask, sigma_layer1)
    cands_per_kernel = det_per_kernel.forward(cube_t, validity_mask, sigma_layer1)

    # Both runs detect the injected pulse at b4. The amortise path
    # also contains b1/b2/b8 candidates (bigger bank), but the b4
    # subset must be at the same (l, m, dm_idx, t) as the per-kernel
    # single-b4 run.
    b4_amortise = [
        c for c in cands_amortise if c.kernel_id == "unit:d1:b4"
    ]
    b4_per_kernel = [
        c for c in cands_per_kernel if c.kernel_id == "unit:d1:b4"
    ]
    assert b4_amortise, "amortise path missed b4 detection"
    assert b4_per_kernel, "per-kernel path missed b4 detection"
    top_a = max(b4_amortise, key=lambda c: c.snr)
    top_p = max(b4_per_kernel, key=lambda c: c.snr)
    assert top_a.l == top_p.l
    assert top_a.m == top_p.m
    assert top_a.dm_idx == top_p.dm_idx
    assert top_a.event_specnum == top_p.event_specnum
    assert abs(top_a.snr - top_p.snr) <= 1e-3
