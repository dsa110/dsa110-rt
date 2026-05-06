"""Tests for ``dsart.noise_norm.layer1`` (M5 chunk 3).

Plan §3.6.9 lines 984-1013. Coverage:

  * ``sigma_clipped_std`` returns ~1.0 on iid N(0, 1) input;
    NaN-aware (excludes NaN cells); robust to a tiny outlier fraction.
  * ``layer1_global_scalar`` returns ``[N_fdm] float32`` and reduces
    over (T_det, H, W) per fdm independently.
  * ``Layer1State`` 5-cube burn-in: cubes 0-4 return the median of the
    history seen so far; from cube 5+, returns the current cube's σ
    directly.
  * Burn-in is robust to a single contaminated cube (the median of 5
    samples isn't shifted by one outlier).
  * ``Layer1State.reset()`` clears state.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.noise_norm.layer1 import (  # noqa: E402
    Layer1State,
    layer1_global_scalar,
    sigma_clipped_std,
)


# 3-iteration 3σ clipping on Gaussian data systematically under-estimates
# σ by ~1.5% — the asymptotic value of `sqrt(mean((s - μ)²))` over a
# 3σ-truncated normal is `sqrt(1 - 2·3·φ(3) / (1-2·Φ(-3))) ≈ 0.9854` (a
# well-known clipped-std bias; see e.g. astropy.stats.sigma_clipped_stats
# which carries an optional `cenfunc='median', stdfunc='mad_std'`
# correction we deliberately do NOT apply per plan §3.6.9 lines 985-993
# which pins the un-corrected σ-clip form). Tests therefore allow a 3%
# absolute tolerance to accommodate this bias.
_SIGMA_CLIP_BIAS_TOLERANCE = 0.03


def test_sigma_clipped_std_unit_gaussian() -> None:
    """N(0, 1) input → σ ≈ 0.985 (3σ-clip bias) within tolerance."""
    rng = np.random.default_rng(20260505)
    x = torch.from_numpy(rng.standard_normal(200_000).astype(np.float32))
    s = sigma_clipped_std(x)
    assert abs(s - 1.0) < _SIGMA_CLIP_BIAS_TOLERANCE


def test_sigma_clipped_std_robust_to_outliers() -> None:
    """A tiny outlier fraction (1%) at 10σ doesn't shift the result much
    (would inflate the unclipped std by ~ sqrt(0.99 + 100 · 0.01) ≈ 1.41;
    the σ-clipped result stays near 0.985)."""
    rng = np.random.default_rng(20260505)
    x = rng.standard_normal(100_000).astype(np.float32)
    n_out = 1_000  # 1%
    x[:n_out] = 10.0  # very far from 0
    s = sigma_clipped_std(torch.from_numpy(x))
    assert abs(s - 1.0) < _SIGMA_CLIP_BIAS_TOLERANCE
    # Compare to the unclipped std for sanity:
    unclipped = float(np.sqrt(np.mean((x - np.median(x)) ** 2)))
    assert unclipped > 1.3  # outliers really do inflate the unclipped form


def test_sigma_clipped_std_handles_nans() -> None:
    """NaN cells are excluded from the σ-clip estimator (the upstream
    edge-mask sets cells outside the §3.6.5 G11 envelope to NaN)."""
    rng = np.random.default_rng(20260505)
    x = rng.standard_normal(100_000).astype(np.float32)
    x[::5] = np.nan  # 20% NaN
    s = sigma_clipped_std(torch.from_numpy(x))
    assert abs(s - 1.0) < _SIGMA_CLIP_BIAS_TOLERANCE


def test_sigma_clipped_std_all_nan_returns_zero() -> None:
    x = torch.full((100,), float("nan"))
    assert sigma_clipped_std(x) == 0.0


def test_sigma_clipped_std_empty_returns_zero() -> None:
    assert sigma_clipped_std(torch.empty(0)) == 0.0


def test_sigma_clipped_std_zero_variance_returns_zero() -> None:
    """Constant input → σ = 0; the clip loop short-circuits on σ=0."""
    x = torch.full((1000,), 3.14)
    assert sigma_clipped_std(x) == 0.0


def test_layer1_global_scalar_per_fdm_independence() -> None:
    """Each fine_DM trial gets its own σ; the reduction is over
    (T_det, H, W) only, not pooled across fdms."""
    rng = np.random.default_rng(20260505)
    cube = torch.from_numpy(
        rng.standard_normal((32, 4, 8, 8)).astype(np.float32)
    )
    # Scale fdm trial 2 by a factor of 5 → its σ should be ~5× (modulo
    # the universal 3σ-clip bias factor of ~0.985).
    cube[:, 2, :, :] *= 5.0
    sigmas = layer1_global_scalar(cube)
    assert sigmas.shape == (4,)
    assert sigmas.dtype == torch.float32
    for fdm in (0, 1, 3):
        assert abs(float(sigmas[fdm]) - 1.0) < _SIGMA_CLIP_BIAS_TOLERANCE
    assert 4.5 < float(sigmas[2]) < 5.5


def test_layer1_global_scalar_rejects_non_4d() -> None:
    with pytest.raises(ValueError, match="dim"):
        layer1_global_scalar(torch.zeros(5))


# ---------------------------------------------------------------------------
# Layer1State burn-in
# ---------------------------------------------------------------------------


def _noise_cube(t: int = 16, n_fdm: int = 4, h: int = 8, scale: float = 1.0,
                seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.standard_normal((t, n_fdm, h, h)).astype(np.float32)
    ) * scale


def test_layer1_state_warms_up_then_passes_through() -> None:
    state = Layer1State(n_fdm=3, n_burnin_cubes=5)
    assert state.is_warming_up
    for cube_idx in range(5):
        cube = _noise_cube(n_fdm=3, seed=cube_idx)
        s = state.update_and_query(cube)
        assert s.shape == (3,)
        assert s.dtype == torch.float32
    assert not state.is_warming_up
    assert state.cube_count == 5


def test_layer1_state_burnin_is_median_of_history() -> None:
    """Cube 0 returns its own σ; cube 1 returns median of [σ0, σ1]; ...
    cube N-1 returns median of all N seen so far."""
    state = Layer1State(n_fdm=1, n_burnin_cubes=5)
    # Pass per-fdm σs directly (simpler than tweaking cubes).
    sigmas = [1.0, 1.5, 2.0, 0.5, 3.0]  # explicitly chosen
    medians_so_far = [
        np.median([1.0]),                       # 1.0
        np.median([1.0, 1.5]),                  # 1.25
        np.median([1.0, 1.5, 2.0]),             # 1.5
        np.median([1.0, 1.5, 2.0, 0.5]),        # 1.25
        np.median([1.0, 1.5, 2.0, 0.5, 3.0]),   # 1.5
    ]
    for i, sigma in enumerate(sigmas):
        out = state.update_and_query(
            per_fdm_sigma=torch.tensor([sigma], dtype=torch.float32)
        )
        assert abs(float(out[0]) - medians_so_far[i]) < 1e-5, (
            f"cube {i}: got {float(out[0])}, expected {medians_so_far[i]}"
        )
    # Cube 5 (post-burnin): returns current cube's σ directly.
    out = state.update_and_query(
        per_fdm_sigma=torch.tensor([99.0], dtype=torch.float32)
    )
    assert abs(float(out[0]) - 99.0) < 1e-5


def test_layer1_state_burnin_robust_to_one_outlier() -> None:
    """The burn-in uses the median (not mean) so a single contaminated
    cube doesn't shift σ much. Plan §1011 calls this out as the
    motivating use case."""
    state = Layer1State(n_fdm=1, n_burnin_cubes=5)
    # Four normal cubes (σ ≈ 1) and one wildly-contaminated cube
    # (σ = 100). Final burn-in median should be ≈ 1, NOT (1·4+100)/5 = 20.8.
    sigmas = [1.0, 1.0, 1.0, 1.0, 100.0]
    last = None
    for sigma in sigmas:
        last = state.update_and_query(
            per_fdm_sigma=torch.tensor([sigma], dtype=torch.float32)
        )
    assert last is not None
    assert abs(float(last[0]) - 1.0) < 1e-5  # median([1,1,1,1,100]) = 1


def test_layer1_state_reset_clears_state() -> None:
    state = Layer1State(n_fdm=2, n_burnin_cubes=5)
    for _ in range(3):
        state.update_and_query(_noise_cube(n_fdm=2))
    assert state.cube_count == 3
    state.reset()
    assert state.cube_count == 0
    assert state.is_warming_up


def test_layer1_state_rejects_both_or_neither_input() -> None:
    state = Layer1State(n_fdm=2, n_burnin_cubes=5)
    with pytest.raises(ValueError, match="exactly one"):
        state.update_and_query()
    with pytest.raises(ValueError, match="exactly one"):
        state.update_and_query(
            cube=_noise_cube(n_fdm=2),
            per_fdm_sigma=torch.tensor([1.0, 1.0]),
        )


def test_layer1_state_rejects_wrong_sigma_shape() -> None:
    state = Layer1State(n_fdm=2, n_burnin_cubes=5)
    with pytest.raises(ValueError, match="shape"):
        state.update_and_query(per_fdm_sigma=torch.tensor([1.0]))
