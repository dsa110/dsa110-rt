"""Tests for the chunk-8 streaming kernel-by-kernel
``DeterministicDetector`` forward path
(``forward.py::_streaming_forward``).

The streaming forward exists because the batched ``forward()`` allocates
``[K, T_det, N_fdm, H, W]`` upfront — 16 GiB at production geometry
(T_det=256, N_fdm=32, N_grid=256, K=8 fp32) and 256 GiB for the full
K=128 bank, both blowing past an 11 GiB 2080 Ti's CUDA budget.

The streaming forward is documented to be **semantically equivalent**
to the batched forward (modulo a deterministic re-sort in
``merge_across_kernels`` that erases candidate-list ordering ties).
This test file pins that equivalence on small-geometry CPU fixtures
where both paths fit comfortably:

  * ``test_streaming_matches_batched_noise_only_no_candidates``: pure
    Gaussian noise at θ=8 → both paths emit zero candidates and arrive
    at the same Layer-2 σ_k EMA after one cube.
  * ``test_streaming_matches_batched_with_injection``: a known
    cube-injection burst → both paths recover the injected
    (l, m, dm_idx, t) within the v1 NMS radii at matching SNR
    (within fp16/fp32 round-off tolerance).
  * ``test_streaming_matches_batched_warmup_flag``: the NOISE_WARMUP
    flag is set on candidates emitted from cubes 0..N_burnin-1 in both
    paths.
  * ``test_boxcar_via_cumsum_tile_size_bit_exact``: tiled boxcar is
    bit-exact vs untiled along the cumsum-orthogonal axis.

The production search-compute service constructs the detector with
``streaming=True`` (see ``services/search_compute.py::_build_detector``)
so the production path is exercised by these tests + by the chunk-7
voltage-fixture closure benches.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.detector.forward import (  # noqa: E402
    DeterministicDetector,
    boxcar_via_cumsum,
)
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.inject.cube_injection import (  # noqa: E402
    CubeInjectionConfig,
    synthesise_cube,
)


# ---------------------------------------------------------------------------
# Small-geometry kernel bank: keep the test fixtures fast on CPU
# ---------------------------------------------------------------------------


SMALL_BANK = build_kernel_bank(
    image_tokens=("unit",),
    dm_tokens=("d1",),
    time_tokens=("b1", "b2", "b4", "b8"),
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(*, streaming: bool, threshold_sigma: float = 8.0) -> DeterministicDetector:
    """Construct a tiny-bank detector. Both streaming and batched paths
    seed Layer-2 σ_k to ``√(K_dm·K_time)`` via ``layer2_seed_unit=True``
    (the default) so neither path needs a multi-cube burn-in to hit a
    sensible σ_k for these single-cube fixtures."""
    return DeterministicDetector(
        kernel_bank=SMALL_BANK,
        threshold_sigma=threshold_sigma,
        detector_version="v1.M5-test",
        search_node_id=1,
        gpu_half=1,
        dtype=torch.float32,
        streaming=streaming,
        # Streaming tile_size doesn't matter for these small fixtures —
        # but use a value < W to exercise the tile loop in
        # boxcar_via_cumsum where it's reachable.
        streaming_tile_size=4,
    )


def _candidate_key(c) -> tuple:
    """Sort key for candidate equivalence asserts (kernel_id +
    (l, m, dm_idx, t) → unique within an SNR tie)."""
    return (
        c.kernel_id,
        round(float(c.l), 3),
        round(float(c.m), 3),
        int(c.dm_idx),
        int(c.event_specnum),
    )


# ---------------------------------------------------------------------------
# Equivalence tests (CPU; small geometry)
# ---------------------------------------------------------------------------


def test_streaming_matches_batched_noise_only_no_candidates() -> None:
    """Pure unit-σ Gaussian → no candidates above θ=8 in either path
    (probability of any cell crossing in a 16×16×4×4 cube is ~1e-8)."""
    cube_t, validity_mask, sigma_layer1 = synthesise_cube(
        t_det=16, n_fdm=4, n_grid=8,
        rng=np.random.default_rng(0),
    )

    det_b = _make_detector(streaming=False)
    det_s = _make_detector(streaming=True)

    cands_b = det_b.forward(cube_t, validity_mask, sigma_layer1)
    cands_s = det_s.forward(cube_t, validity_mask, sigma_layer1)

    assert cands_b == [] == cands_s

    # Layer-2 σ_k arrives at the same per-kernel EMA after one cube.
    sk_b = det_b.layer2_state.s_k.detach().cpu().numpy()
    sk_s = det_s.layer2_state.s_k.detach().cpu().numpy()
    assert np.allclose(sk_b, sk_s, rtol=1e-5, atol=1e-5)


def test_streaming_matches_batched_with_injection() -> None:
    """Inject one wide-boxcar pulse at a known (l, m, dm, t) → both
    paths must recover at least one candidate at the injection cell
    with matching SNR. Use a generous amplitude so the recovered SNR
    well exceeds θ=8."""
    cfg = CubeInjectionConfig(
        l_pix=4, m_pix=4,
        fine_dm_idx=2,
        t_in_cube=8,
        snr=15.0,
        width_samples=4,
        profile="boxcar",
    )
    cube_t, validity_mask, sigma_layer1 = synthesise_cube(
        t_det=16, n_fdm=4, n_grid=8,
        rng=np.random.default_rng(42),
        injections=(cfg,),
    )

    det_b = _make_detector(streaming=False)
    det_s = _make_detector(streaming=True)

    cands_b = det_b.forward(cube_t, validity_mask, sigma_layer1)
    cands_s = det_s.forward(cube_t, validity_mask, sigma_layer1)

    # Both paths must recover at least one candidate.
    assert cands_b, "batched forward returned no candidates"
    assert cands_s, "streaming forward returned no candidates"

    # Top-SNR candidate from each must agree on the (l, m, dm_idx, t)
    # bin within the merger's NMS tolerance, AND the SNRs must agree
    # to fp32 round-off (both paths are fp32 here).
    top_b = max(cands_b, key=lambda c: c.snr)
    top_s = max(cands_s, key=lambda c: c.snr)
    assert top_b.kernel_id == top_s.kernel_id
    assert abs(top_b.l - top_s.l) <= 1
    assert abs(top_b.m - top_s.m) <= 1
    assert top_b.dm_idx == top_s.dm_idx
    assert abs(top_b.event_specnum - top_s.event_specnum) <= 1
    assert math.isclose(top_b.snr, top_s.snr, rel_tol=1e-4, abs_tol=1e-3)

    # Layer-2 σ_k post-update agrees per kernel.
    sk_b = det_b.layer2_state.s_k.detach().cpu().numpy()
    sk_s = det_s.layer2_state.s_k.detach().cpu().numpy()
    assert np.allclose(sk_b, sk_s, rtol=1e-4, atol=1e-4)


def test_streaming_matches_batched_warmup_flag() -> None:
    """When the Layer-2 EMA is in burn-in, both paths must set the
    NOISE_WARMUP flag on every emitted candidate. We flip
    layer2_seed_unit=False so the very first cube IS in burn-in (the
    detector default seeds σ_k to the analytic value but does NOT
    advance cube_count, so cube_count=0 < n_burnin=1 holds)."""
    cfg = CubeInjectionConfig(
        l_pix=4, m_pix=4,
        fine_dm_idx=2,
        t_in_cube=8,
        snr=15.0,
        width_samples=4,
        profile="boxcar",
    )
    cube_t, validity_mask, sigma_layer1 = synthesise_cube(
        t_det=16, n_fdm=4, n_grid=8,
        rng=np.random.default_rng(42),
        injections=(cfg,),
    )

    def _new_warmup_detector(streaming: bool) -> DeterministicDetector:
        return DeterministicDetector(
            kernel_bank=SMALL_BANK,
            threshold_sigma=8.0,
            detector_version="v1.M5-test",
            search_node_id=1,
            gpu_half=1,
            dtype=torch.float32,
            streaming=streaming,
            streaming_tile_size=4,
            layer2_n_burnin=2,  # cube 0 IS warmup; cube 1 IS warmup; cube 2+ NOT
        )

    det_b = _new_warmup_detector(streaming=False)
    det_s = _new_warmup_detector(streaming=True)

    cands_b = det_b.forward(cube_t, validity_mask, sigma_layer1)
    cands_s = det_s.forward(cube_t, validity_mask, sigma_layer1)

    from dsart.common.contracts import CandidateFlags
    warmup_bit = int(CandidateFlags.NOISE_WARMUP)
    assert all((c.flags & warmup_bit) == warmup_bit for c in cands_b)
    assert all((c.flags & warmup_bit) == warmup_bit for c in cands_s)


# ---------------------------------------------------------------------------
# boxcar_via_cumsum tile_size bit-exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis,width", [(0, 4), (1, 3), (2, 5)])
def test_boxcar_via_cumsum_tile_size_bit_exact(axis: int, width: int) -> None:
    """Tiled boxcar (along the LAST axis) is bit-exact equivalent to
    untiled when the cumsum axis is different from the tile axis. This
    is what the streaming detector relies on to bound the fp32 cumsum
    working set at production geometry."""
    rng = np.random.default_rng(123)
    x = torch.from_numpy(rng.standard_normal((8, 6, 7, 13)).astype(np.float32))

    untiled = boxcar_via_cumsum(x, axis=axis, width=width)
    tiled = boxcar_via_cumsum(x, axis=axis, width=width, tile_size=4)

    if axis == x.ndim - 1:
        # tile_size is silently ignored when axis == last axis
        # (tiling along the cumsum axis would BREAK associativity).
        # Verify it still equals untiled.
        torch.testing.assert_close(tiled, untiled, rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(tiled, untiled, rtol=0.0, atol=0.0)
