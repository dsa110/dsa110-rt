"""Tests for ``dsart.noise_norm.layer2`` (M5 chunk 3).

Plan §3.6.10 lines 1015-1035 + plan §3.6.13 ``test_layer2_interior_ema``.
Coverage:

  * ``layer2_interior_sigma`` shape / dtype contract; reduces over
    interior slice [n_kernel_max_t//2, T_det − n_kernel_max_t//2] and
    returns ``[K] float32``.
  * Interior σ is *higher* than full-cube σ on a noise-only cube whose
    boundary 25% has partial-width zero-padding bias (plan §1132
    invariant). We synthesise that bias by injecting a deliberately
    suppressed boundary on the score.
  * ``Layer2State`` Welford burn-in over first ``n_burnin`` cubes —
    s_k converges to the running mean of per-cube σs.
  * Post-burn-in: EMA with γ = 1 - exp(-cube_cadence_s/τ_s); decay
    rate matches analytic.
  * ``valid=False`` skips the update (state is unchanged after the call,
    cube_count still increments — wait, plan §319 says invalid cubes
    skip BOTH forward and noise updates; we test that the EMA value is
    preserved).

Also tests the integration with ``DeterministicDetector`` end-to-end:

  * After ``layer2_n_burnin`` noise-only cubes, the detector's emitted
    candidate count drops to ~0 (the EMA has learned the per-kernel σ_k
    and threshold rejects the noise tail).
  * NOISE_WARMUP flag is set on every Candidate while the EMA is in
    burn-in; cleared from cube ``layer2_n_burnin`` onward.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.common.contracts import CandidateFlags  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.inject.cube_injection import (  # noqa: E402
    CubeInjectionConfig,
    synthesise_cube,
)
from dsart.noise_norm.layer2 import (  # noqa: E402
    Layer2State,
    layer2_interior_sigma,
)


def _scores_random(K: int = 4, T: int = 32, N_fdm: int = 4, H: int = 8,
                   seed: int = 42, scale: float = 1.0) -> torch.Tensor:
    """Random per-kernel score tensor."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.standard_normal((K, T, N_fdm, H, H)).astype(np.float32) * scale
    )


# ---------------------------------------------------------------------------
# layer2_interior_sigma
# ---------------------------------------------------------------------------


def test_layer2_interior_sigma_shape_dtype() -> None:
    scores = _scores_random(K=8, T=512, N_fdm=4, H=16)
    sigmas = layer2_interior_sigma(scores, n_kernel_max_t=128)
    assert sigmas.shape == (8,)
    assert sigmas.dtype == torch.float32


def test_layer2_interior_sigma_unit_input_yields_unit_sigma() -> None:
    """N(0, 1) score → σ ≈ 0.985 per kernel (3σ-clip bias; same as
    Layer-1's clipped-std test_sigma_clipped_std_unit_gaussian)."""
    scores = _scores_random(K=2, T=512, N_fdm=4, H=16, scale=1.0)
    sigmas = layer2_interior_sigma(scores, n_kernel_max_t=128)
    for k in range(2):
        # Tolerance covers the 3σ-clip bias (asymptotic ~0.985).
        assert abs(float(sigmas[k]) - 1.0) < 0.03


def test_layer2_interior_sigma_excludes_biased_boundary() -> None:
    """If we deliberately suppress the boundary 25% of the cube to
    simulate the §3.6.12 partial-width bias, the interior σ should
    remain near 1 while the full-cube σ drops noticeably below 1."""
    rng = np.random.default_rng(20260505)
    K, T, N_fdm, H = 1, 512, 2, 16  # noqa: N806
    scores = torch.from_numpy(
        rng.standard_normal((K, T, N_fdm, H, H)).astype(np.float32)
    )
    # Suppress the boundary 25% (first 64 + last 64 samples) by
    # multiplying by 0.5 — this mimics the partial-width zero-pad bias.
    scores[:, :64, :, :, :] *= 0.5
    scores[:, -64:, :, :, :] *= 0.5

    sigma_interior = float(
        layer2_interior_sigma(scores, n_kernel_max_t=128)[0]
    )
    # Full-cube σ for the SAME tensor.
    from dsart.noise_norm.layer1 import sigma_clipped_std
    sigma_full = sigma_clipped_std(scores[0])

    assert sigma_interior > sigma_full, (
        f"interior σ ({sigma_interior}) should be > full-cube σ "
        f"({sigma_full}) when boundary is suppressed"
    )
    # And the ratio should be in the plan-pinned range [1.05, 1.20]
    # (we deliberately chose a 0.5x suppression which puts the ratio
    # near the upper end of the plan-cited interval).
    ratio = sigma_interior / sigma_full
    assert 1.05 <= ratio <= 1.30, (
        f"interior/full ratio = {ratio:.3f} outside plan-pinned [1.05, 1.20] "
        f"(allowing 1.30 for the 0.5x boundary scaling chosen here)"
    )


def test_layer2_interior_sigma_small_cube_falls_back_to_full() -> None:
    """If T_det ≤ n_kernel_max_t, the interior slice is empty, so we
    fall back to using the full cube. Used by small unit-test cubes.
    Tolerance is wider here because the sample size is much smaller
    (16 × 2 × 4 × 4 = 512 cells per kernel; sample-σ uncertainty ≈
    sqrt(2/N) ≈ 6%)."""
    scores = _scores_random(K=2, T=16, N_fdm=2, H=4)
    sigmas = layer2_interior_sigma(scores, n_kernel_max_t=128)
    assert sigmas.shape == (2,)
    for k in range(2):
        # Combine 3σ-clip bias (~0.015) + small-N uncertainty (~6%):
        # tolerance widened to 0.10.
        assert abs(float(sigmas[k]) - 1.0) < 0.10


# ---------------------------------------------------------------------------
# Layer2State burn-in + EMA
# ---------------------------------------------------------------------------


def test_layer2_state_init_defaults() -> None:
    s = Layer2State(n_kernels=4)
    assert s.n_kernels == 4
    assert s.s_k.shape == (4,)
    assert s.cube_count == 0
    assert s.is_warming_up
    # CUBE_CADENCE_S_DEFAULT = CUBE_CADENCE_SAMPLES_DEFAULT (256) ×
    # T_INT_SEARCH_US_DEFAULT (= 16 × NATIVE_SAMPLE_US (32.768)) × 1e-6
    # = 256 × 524.288e-6 = 0.134217728 s. With τ_s = 30 → γ ≈ 0.004464.
    expected_gamma = 1.0 - math.exp(-0.134217728 / 30.0)
    assert s.gamma == pytest.approx(expected_gamma, rel=1e-4)


def test_layer2_state_burnin_is_running_mean() -> None:
    """During burn-in, s_k is the Welford running mean of per-cube σs."""
    s = Layer2State(n_kernels=1, n_burnin=5)
    sigmas = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_means = [1.0, 1.5, 2.0, 2.5, 3.0]
    for i, sigma in enumerate(sigmas):
        s_k, _ = s.update_and_query(
            per_kernel_sigma=torch.tensor([sigma], dtype=torch.float32)
        )
        assert abs(float(s_k[0]) - expected_means[i]) < 1e-5


def test_layer2_state_post_burnin_is_ema() -> None:
    """After n_burnin cubes the update switches to EMA: s_k ← γ·σ_cube +
    (1-γ)·s_k."""
    s = Layer2State(n_kernels=1, n_burnin=3)
    # Burn-in: 3 cubes at σ=1 → s_k = 1.
    for _ in range(3):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0]))
    assert not s.is_warming_up
    # Now feed σ=2; s_k should jump by γ × (2 - 1) = γ.
    pre = float(s.s_k[0])
    s_k, _ = s.update_and_query(per_kernel_sigma=torch.tensor([2.0]))
    expected_post = pre + s.gamma * (2.0 - pre)
    assert abs(float(s_k[0]) - expected_post) < 1e-5


def test_layer2_state_invalid_cube_skips_update() -> None:
    """When valid=False (RFI'd / warmup-flagged cube), the EMA is NOT
    updated — but the cube_count does NOT increment either (per plan
    §319 invalid cubes skip BOTH forward and noise updates)."""
    s = Layer2State(n_kernels=1, n_burnin=3)
    # Burn-in to a known value.
    for _ in range(3):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0]))
    pre_count = s.cube_count
    pre_value = float(s.s_k[0])
    # Now pass valid=False with a wildly different σ.
    s_k, _ = s.update_and_query(
        per_kernel_sigma=torch.tensor([100.0]), valid=False,
    )
    assert s.cube_count == pre_count
    assert abs(float(s_k[0]) - pre_value) < 1e-5


def test_layer2_state_zero_sigma_falls_back_to_previous() -> None:
    """A degenerate cube that produces σ_cube=0 for some kernel does
    NOT poison the EMA — the previous value is reused for that kernel."""
    s = Layer2State(n_kernels=2, n_burnin=2)
    s.update_and_query(per_kernel_sigma=torch.tensor([1.0, 1.0]))
    s.update_and_query(per_kernel_sigma=torch.tensor([2.0, 2.0]))
    # Now feed σ=0 for kernel 1; it should retain its previous value.
    pre = s.s_k[1].clone()
    s_k, _ = s.update_and_query(per_kernel_sigma=torch.tensor([3.0, 0.0]))
    assert s_k[1] == pre


def test_layer2_state_reset_clears_state() -> None:
    s = Layer2State(n_kernels=2, n_burnin=5)
    for _ in range(3):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0, 1.0]))
    s.reset()
    assert s.cube_count == 0
    assert torch.allclose(s.s_k, torch.ones(2))


def test_layer2_state_sigma_max_ratio_clamps_high_outliers() -> None:
    """T1 (2026-06-07): an anomalous high-σ cube must NOT lift σ_k by
    more than ``sigma_max_ratio`` — the offending kernel's update is
    rejected and ``n_clamped_high`` is bumped, so the post-anomaly
    relaxation window collapses from ~τ_s to a single sample."""
    s = Layer2State(n_kernels=3, n_burnin=2, sigma_max_ratio=4.0)
    # Burn in at σ=1 so s_k = 1.
    for _ in range(2):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0, 1.0, 1.0]))
    assert not s.is_warming_up
    assert s.n_clamped_high == 0
    # Anomaly: kernel 1 gets σ=10 (10× s_k_prev — exceeds 4× ceiling).
    # Kernel 0 gets σ=2 (within 4× — should be blended). Kernel 2 gets
    # σ=4 (exactly at the ceiling — not strictly greater, blended).
    pre = s.s_k.clone()
    s_k, _ = s.update_and_query(
        per_kernel_sigma=torch.tensor([2.0, 10.0, 4.0]),
    )
    # Kernel 0: blended.
    expected_0 = pre[0] + s.gamma * (2.0 - pre[0])
    assert abs(float(s_k[0]) - float(expected_0)) < 1e-5
    # Kernel 1: clamped — s_k stays at the prior value.
    assert float(s_k[1]) == float(pre[1])
    # Kernel 2: at ceiling, NOT strictly greater → blended through.
    expected_2 = pre[2] + s.gamma * (4.0 - pre[2])
    assert abs(float(s_k[2]) - float(expected_2)) < 1e-5
    assert s.n_clamped_high == 1
    per_k = s.per_kernel_clamped_high
    assert int(per_k[0]) == 0
    assert int(per_k[1]) == 1
    assert int(per_k[2]) == 0


def test_layer2_state_sigma_max_ratio_disabled_by_default() -> None:
    """Default ``sigma_max_ratio=0.0`` preserves bit-for-bit legacy
    behaviour (no clamping, no counter increments)."""
    s = Layer2State(n_kernels=1, n_burnin=2)
    for _ in range(2):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0]))
    pre = s.s_k.clone()
    s_k, _ = s.update_and_query(per_kernel_sigma=torch.tensor([100.0]))
    expected = pre + s.gamma * (100.0 - pre)
    assert abs(float(s_k[0]) - float(expected)) < 1e-4
    assert s.n_clamped_high == 0


def test_layer2_state_reset_clears_clamp_counters() -> None:
    """``reset()`` must clear the new T1 counters too."""
    s = Layer2State(n_kernels=2, n_burnin=2, sigma_max_ratio=2.0)
    for _ in range(2):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0, 1.0]))
    s.update_and_query(per_kernel_sigma=torch.tensor([10.0, 10.0]))
    assert s.n_clamped_high == 2
    s.reset()
    assert s.n_clamped_high == 0
    assert int(s.per_kernel_clamped_high.sum()) == 0


def test_layer2_state_sigma_max_ratio_validates_input() -> None:
    with pytest.raises(ValueError, match="sigma_max_ratio"):
        Layer2State(n_kernels=1, sigma_max_ratio=-1.0)


def test_layer2_state_rejects_bad_inputs() -> None:
    s = Layer2State(n_kernels=2, n_burnin=5)
    with pytest.raises(ValueError, match="exactly one"):
        s.update_and_query()
    with pytest.raises(ValueError, match="shape"):
        s.update_and_query(per_kernel_sigma=torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# Detector end-to-end with Layer-2 wired
# ---------------------------------------------------------------------------


def test_detector_warmup_flag_set_during_burnin() -> None:
    """Plan §1610: the detector sets flags.bit3 = noise_warmup on
    every emitted Candidate while the Layer-2 EMA is in burn-in."""
    cfg = CubeInjectionConfig(
        l_pix=8, m_pix=8, fine_dm_idx=2, t_in_cube=32,
        snr=15.0, width_samples=4,
    )
    cube, validity, sigma1 = synthesise_cube(
        t_det=64, n_fdm=4, n_grid=16,
        injections=(cfg,),
        rng=np.random.default_rng(20260505),
    )
    # Use a small bank so the test is fast; n_burnin=5 (small).
    bank = build_kernel_bank(
        image_tokens=("unit",),
        dm_tokens=("d1",),
        time_tokens=("b1", "b4"),
    )
    det = DeterministicDetector(
        kernel_bank=bank, threshold_sigma=8.0, dtype=torch.float16,
        layer2_n_burnin=5, layer2_seed_unit=True,
    )
    out = det.forward(cube.to(torch.float16), validity, sigma1)
    # Cube 0 → still warming up.
    assert det.layer2_state.cube_count == 1
    assert det.layer2_state.is_warming_up
    if out:
        for cand in out:
            assert cand.flags & int(CandidateFlags.NOISE_WARMUP), (
                f"expected NOISE_WARMUP flag during burn-in; got flags={cand.flags}"
            )


def test_detector_warmup_flag_clears_after_burnin() -> None:
    """After ``n_burnin`` cubes the warmup flag clears."""
    bank = build_kernel_bank(
        image_tokens=("unit",), dm_tokens=("d1",),
        time_tokens=("b1", "b4"),
    )
    det = DeterministicDetector(
        kernel_bank=bank, threshold_sigma=8.0, dtype=torch.float16,
        layer2_n_burnin=3, layer2_seed_unit=True,
    )
    rng = np.random.default_rng(20260505)
    # Push 3 noise-only cubes through to complete burn-in.
    for _ in range(3):
        cube, validity, sigma1 = synthesise_cube(
            t_det=64, n_fdm=4, n_grid=16, rng=rng,
        )
        det.forward(cube.to(torch.float16), validity, sigma1)
    assert det.layer2_state.cube_count == 3
    assert not det.layer2_state.is_warming_up

    # Now inject + run; emitted candidates should NOT carry NOISE_WARMUP.
    cfg = CubeInjectionConfig(
        l_pix=8, m_pix=8, fine_dm_idx=2, t_in_cube=32,
        snr=15.0, width_samples=4,
    )
    cube, validity, sigma1 = synthesise_cube(
        t_det=64, n_fdm=4, n_grid=16,
        injections=(cfg,), rng=rng,
    )
    out = det.forward(cube.to(torch.float16), validity, sigma1)
    assert len(out) >= 1
    for cand in out:
        assert not (cand.flags & int(CandidateFlags.NOISE_WARMUP)), (
            f"expected NOISE_WARMUP cleared post-burn-in; got flags={cand.flags}"
        )


def test_detector_layer2_ema_converges_on_noise_cubes() -> None:
    """Feed N noise-only cubes; the EMA's per-kernel s_k should
    converge near the analytic noise std for each kernel triple
    (since cube_injection cubes have σ=1 per cell, the post-conv σ
    for kernel (k_dm, k_time) = (1, K_t) is sqrt(K_t)).

    Restricted to k_dm=1 kernels to avoid the small-bench fine-DM-axis
    boundary bias: a width-K_dm=3 kernel applied to a small cube with
    N_fdm=4 has 2/4 = 50% of fdm trials reading off the cube edge into
    zero-padding, biasing σ_k well below the analytic sqrt(3) value
    (this is the §3.6.10 line 1018-1035 boundary-bias issue, but on the
    fine-DM axis instead of the time axis, and at a much higher boundary
    fraction than production where N_fdm=100). At production sizing
    the fdm-edge fraction is ~2% and the bias is negligible. Chunk-6
    bench/noise_norm_calibration.py validates the full kernel bank
    with production-sized cubes.
    """
    bank = build_kernel_bank(
        image_tokens=("unit",), dm_tokens=("d1",),
        time_tokens=("b1", "b4", "b16"),
    )
    det = DeterministicDetector(
        kernel_bank=bank, threshold_sigma=8.0, dtype=torch.float16,
        layer2_n_burnin=10, layer2_seed_unit=False,  # start at 1.0
    )
    rng = np.random.default_rng(20260505)
    for _ in range(10):
        cube, validity, sigma1 = synthesise_cube(
            t_det=128, n_fdm=4, n_grid=16, rng=rng,
        )
        det.forward(cube.to(torch.float16), validity, sigma1)
    # After 10 cubes the Welford running mean should be near analytic.
    # Tolerance: small bench has H=W=16, T=128 (interior 0..127 since
    # T < n_kernel_max_t falls back to full cube), N_fdm=4 → per-kernel
    # cube has ~131k samples → sample-σ uncertainty ~sqrt(2/N) ~0.4%/cube;
    # 10 cubes mean tightens by sqrt(10) ~ 0.13%. Plus 3σ-clip bias of
    # ~1.5%. Use a 5% relative tolerance.
    s_k = det.layer2_state.s_k
    for k_idx, kernel in enumerate(det.kernel_bank):
        analytic = math.sqrt(kernel.k_dm_width * kernel.k_time_width)
        empirical = float(s_k[k_idx])
        assert abs(empirical - analytic) / analytic < 0.05, (
            f"kernel {kernel.kernel_id}: empirical s_k={empirical:.3f} "
            f"vs analytic={analytic:.3f}"
        )
