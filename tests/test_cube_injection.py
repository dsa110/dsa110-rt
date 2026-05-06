"""Tests for ``dsart.inject.cube_injection`` (M5 chunk 2).

Verifies the cube-level injection harness — the post-imager detector
unit-test injector that's the primary driver of M5 detector development
(plan §8 line 2329; D1 / D8 / D12 / F10 / F11 in M5_PLAN_FIXES.md).

Coverage:

  * ``CubeInjectionConfig`` validation: bad indices / non-finite SNR /
    unsupported profile reject; ``from_lm_radians`` round-trips through
    pixel space.
  * ``synthesise_cube`` shape / dtype contracts (D1: float32 cube;
    [T_det, N_fdm] bool validity_mask; [T_det, N_fdm] fp32 sigma_layer1
    all-ones per F11).
  * ``synthesise_cube`` noise statistics (D8: σ=1 per cell; sigma-clipped
    std within tolerance over a large enough sample).
  * ``add_injection`` writes the boxcar profile in the right cells and
    nothing else.
  * **D12 invariant** — width-matched boxcar recovers the injected SNR
    exactly (no detector loss): for an injected pulse of (snr, width),
    summing the cube along the time axis over [t_lo, t_hi) equals
    ``snr × √width`` at the injection cell.
  * ``iter_snr_width_grid`` produces the canonical sweep used by the
    Chunk-5 bench.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.inject.cube_injection import (  # noqa: E402
    PROFILE_FAMILIES,
    CubeInjectionConfig,
    add_injection,
    iter_snr_width_grid,
    synthesise_cube,
)


# ---------------------------------------------------------------------------
# CubeInjectionConfig validation
# ---------------------------------------------------------------------------


def test_config_basic_roundtrip() -> None:
    cfg = CubeInjectionConfig(
        l_pix=10, m_pix=12, fine_dm_idx=3, t_in_cube=200, snr=8.0, width_samples=4
    )
    d = cfg.asdict()
    assert d["l_pix"] == 10 and d["m_pix"] == 12
    assert d["snr"] == 8.0 and d["width_samples"] == 4
    assert d["profile"] == "boxcar"


def test_config_negative_indices_rejected() -> None:
    with pytest.raises(ValueError, match="l_pix"):
        CubeInjectionConfig(
            l_pix=-1, m_pix=0, fine_dm_idx=0, t_in_cube=0, snr=8.0, width_samples=2
        )
    with pytest.raises(ValueError, match="m_pix"):
        CubeInjectionConfig(
            l_pix=0, m_pix=-2, fine_dm_idx=0, t_in_cube=0, snr=8.0, width_samples=2
        )
    with pytest.raises(ValueError, match="fine_dm_idx"):
        CubeInjectionConfig(
            l_pix=0, m_pix=0, fine_dm_idx=-1, t_in_cube=0, snr=8.0, width_samples=2
        )


def test_config_non_finite_snr_rejected() -> None:
    with pytest.raises(ValueError, match="snr"):
        CubeInjectionConfig(
            l_pix=0, m_pix=0, fine_dm_idx=0, t_in_cube=0,
            snr=float("nan"), width_samples=2,
        )
    with pytest.raises(ValueError, match="snr"):
        CubeInjectionConfig(
            l_pix=0, m_pix=0, fine_dm_idx=0, t_in_cube=0,
            snr=-3.0, width_samples=2,
        )


def test_config_unsupported_profile_rejected() -> None:
    """v1 supports only 'boxcar' per D12; gaussian / scattered raise
    NotImplementedError so the bench can't accidentally request a v2
    feature."""
    with pytest.raises(NotImplementedError, match="profile"):
        CubeInjectionConfig(
            l_pix=0, m_pix=0, fine_dm_idx=0, t_in_cube=0,
            snr=8.0, width_samples=2, profile="gaussian",
        )


def test_config_zero_width_rejected() -> None:
    with pytest.raises(ValueError, match="width_samples"):
        CubeInjectionConfig(
            l_pix=0, m_pix=0, fine_dm_idx=0, t_in_cube=0,
            snr=8.0, width_samples=0,
        )


def test_from_lm_radians_round_trip() -> None:
    """Pixel-space construction from sky cosines centres the phase
    centre at (N_grid//2, N_grid//2) per F10."""
    n_grid = 64
    px_scale = 0.001  # rad/cell
    cfg = CubeInjectionConfig.from_lm_radians(
        l_rad=0.0, m_rad=0.0, n_grid=n_grid, pixel_scale_rad=px_scale,
        fine_dm_idx=0, t_in_cube=10, snr=8.0, width_samples=4,
    )
    assert cfg.l_pix == n_grid // 2
    assert cfg.m_pix == n_grid // 2

    cfg = CubeInjectionConfig.from_lm_radians(
        l_rad=+5 * px_scale, m_rad=-3 * px_scale,
        n_grid=n_grid, pixel_scale_rad=px_scale,
        fine_dm_idx=0, t_in_cube=10, snr=8.0, width_samples=4,
    )
    assert cfg.l_pix == n_grid // 2 + 5
    assert cfg.m_pix == n_grid // 2 - 3


def test_from_lm_radians_out_of_grid_rejected() -> None:
    n_grid = 32
    px_scale = 0.001
    with pytest.raises(ValueError, match="l_pix"):
        CubeInjectionConfig.from_lm_radians(
            l_rad=+1.0, m_rad=0.0, n_grid=n_grid, pixel_scale_rad=px_scale,
            fine_dm_idx=0, t_in_cube=10, snr=8.0, width_samples=4,
        )


# ---------------------------------------------------------------------------
# synthesise_cube shape / dtype invariants (D1 / D8 / F11)
# ---------------------------------------------------------------------------


def test_synthesise_cube_shapes_dtypes_default() -> None:
    """Cube/validity_mask/sigma_layer1 dtype + shape contract."""
    cube, mask, sigma1 = synthesise_cube(
        t_det=32, n_fdm=8, n_grid=16,
        rng=np.random.default_rng(42),
    )
    assert cube.shape == (32, 8, 16, 16)
    assert cube.dtype == torch.float32  # D1
    assert mask.shape == (32, 8) and mask.dtype == torch.bool
    assert torch.all(mask)  # all-True per F11
    assert sigma1.shape == (32, 8) and sigma1.dtype == torch.float32
    assert torch.allclose(sigma1, torch.ones_like(sigma1))  # all 1.0 per F11


def test_synthesise_cube_noise_statistics_d8() -> None:
    """D8: thermal-noise background is iid N(0, 1) per cell."""
    cube, _, _ = synthesise_cube(
        t_det=128, n_fdm=4, n_grid=32,
        rng=np.random.default_rng(20260505),
    )
    flat = cube.numpy().ravel()
    # ~half a million cells; the empirical mean / std should be tight to
    # the analytic N(0, 1) expectation.
    assert abs(float(flat.mean())) < 0.01
    assert abs(float(flat.std()) - 1.0) < 0.01


def test_synthesise_cube_noise_std_scales() -> None:
    """noise_std parameter scales linearly."""
    cube, _, _ = synthesise_cube(
        t_det=128, n_fdm=4, n_grid=32, noise_std=2.5,
        rng=np.random.default_rng(20260505),
    )
    assert abs(float(cube.numpy().std()) - 2.5) < 0.05


def test_synthesise_cube_noise_std_zero_yields_zero_cube() -> None:
    """noise_std=0 short-circuits the rng and returns an all-zero cube
    (used by the decoder unit tests where noise would mask the
    injection's exact-recovery test)."""
    cube, _, _ = synthesise_cube(
        t_det=16, n_fdm=4, n_grid=8, noise_std=0.0,
    )
    assert torch.all(cube == 0)


def test_synthesise_cube_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        synthesise_cube(t_det=0, n_fdm=4, n_grid=8)
    with pytest.raises(ValueError):
        synthesise_cube(t_det=8, n_fdm=4, n_grid=8, noise_std=-1)


# ---------------------------------------------------------------------------
# add_injection cell placement
# ---------------------------------------------------------------------------


def test_add_injection_writes_only_target_cell_column() -> None:
    """The boxcar profile writes only into the (l_pix, m_pix, fine_dm_idx)
    column over [t_lo, t_hi); every other cell is unchanged."""
    cube = torch.zeros(32, 4, 8, 8, dtype=torch.float32)
    cfg = CubeInjectionConfig(
        l_pix=3, m_pix=5, fine_dm_idx=2, t_in_cube=16, snr=8.0, width_samples=4,
    )
    add_injection(cube, cfg)

    # Every other column should be exactly zero.
    cube_np = cube.numpy()
    target_zeroed = cube_np.copy()
    target_zeroed[:, 2, 3, 5] = 0.0
    assert np.all(target_zeroed == 0)

    # The target column should be non-zero in the [t_lo, t_hi) window only.
    expected_amp = 8.0 / math.sqrt(4)
    column = cube_np[:, 2, 3, 5]
    nonzero_idxs = np.where(column != 0)[0]
    # Width=4, t_in_cube=16 → window is [14, 18) per centring convention.
    assert list(nonzero_idxs) == [14, 15, 16, 17]
    assert np.allclose(column[14:18], expected_amp)


def test_add_injection_amplitude_matches_d12_formula() -> None:
    """D12: per-cell amplitude is snr / sqrt(width_samples). Verify
    across (snr, width) sweep that a width-matched boxcar over the
    target column equals snr exactly."""
    for snr in (6.0, 8.0, 10.0, 12.0, 15.0):
        for width in (2, 4, 8, 16, 32, 64, 128):
            t_det = 256
            cube = torch.zeros(t_det, 1, 4, 4, dtype=torch.float32)
            cfg = CubeInjectionConfig(
                l_pix=2, m_pix=2, fine_dm_idx=0,
                t_in_cube=t_det // 2, snr=snr, width_samples=width,
            )
            add_injection(cube, cfg)
            # Sum across the cube's time axis at the target cell — for a
            # boxcar of width W, the sum equals amp · W = snr · √W.
            target_sum = float(cube[:, 0, 2, 2].sum())
            expected_sum = snr * math.sqrt(width)
            assert abs(target_sum - expected_sum) < 1e-3, (
                f"snr={snr}, width={width}: sum={target_sum} != "
                f"expected {expected_sum}"
            )


def test_add_injection_rejects_out_of_cube_indices() -> None:
    """Indices outside the cube's actual dims raise ValueError so the
    bench's manifest-loader catches typos at config-load time, not at
    candidate-emit time."""
    cube = torch.zeros(16, 4, 8, 8, dtype=torch.float32)
    cfg = CubeInjectionConfig(
        l_pix=99, m_pix=0, fine_dm_idx=0, t_in_cube=4, snr=8.0, width_samples=2,
    )
    with pytest.raises(ValueError, match="l_pix"):
        add_injection(cube, cfg)


def test_add_injection_rejects_wrong_dtype() -> None:
    """Cube must be float32 per D1; passing fp16 catches the bench's
    accidental down-cast at config-load time."""
    cube = torch.zeros(16, 4, 8, 8, dtype=torch.float16)
    cfg = CubeInjectionConfig(
        l_pix=0, m_pix=0, fine_dm_idx=0, t_in_cube=4, snr=8.0, width_samples=2,
    )
    with pytest.raises(TypeError, match="dtype"):
        add_injection(cube, cfg)


def test_synthesise_cube_with_injections_matches_addition() -> None:
    """Composition: synthesise_cube(injections=(cfg,)) is equivalent to
    synthesise_cube() + add_injection(cfg) on the same seed."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    cfg = CubeInjectionConfig(
        l_pix=4, m_pix=4, fine_dm_idx=1, t_in_cube=8, snr=10.0, width_samples=2,
    )
    cube_a, _, _ = synthesise_cube(
        t_det=16, n_fdm=2, n_grid=8, injections=(cfg,), rng=rng_a,
    )
    cube_b, _, _ = synthesise_cube(t_det=16, n_fdm=2, n_grid=8, rng=rng_b)
    add_injection(cube_b, cfg)
    assert torch.allclose(cube_a, cube_b)


# ---------------------------------------------------------------------------
# Bench helper
# ---------------------------------------------------------------------------


def test_iter_snr_width_grid_default_shape() -> None:
    """Default sweep covers the plan-pinned 5 SNRs × 7 widths grid."""
    grid = iter_snr_width_grid(l_pix=4, m_pix=4, fine_dm_idx=0, t_in_cube=10)
    assert len(grid) == 5 * 7
    assert all(isinstance(c, CubeInjectionConfig) for c in grid)
    snrs = sorted({c.snr for c in grid})
    widths = sorted({c.width_samples for c in grid})
    assert snrs == [6.0, 8.0, 10.0, 12.0, 15.0]
    assert widths == [2, 4, 8, 16, 32, 64, 128]


def test_iter_snr_width_grid_custom() -> None:
    grid = iter_snr_width_grid(
        snrs=(8.0,), widths=(2, 4),
        l_pix=4, m_pix=4, fine_dm_idx=0, t_in_cube=10,
    )
    assert len(grid) == 1 * 2
    assert {c.width_samples for c in grid} == {2, 4}


def test_profile_families_immutable() -> None:
    """The supported-profile tuple is exposed for the bench's manifest
    validator; it must be a tuple (immutable) so the bench can rely on
    the v1 contract."""
    assert isinstance(PROFILE_FAMILIES, tuple)
    assert PROFILE_FAMILIES == ("boxcar",)
