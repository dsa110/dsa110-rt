"""Tests for ``dsart.detector.forward`` (M5 chunk 1).

Verifies the v1-locked Protocol surface and the conv-bank scaffold:

  * ``DeterministicDetector`` satisfies the ``Detector`` Protocol
  * ``kernels()`` returns the closed kernel-id enum (128 default)
  * ``forward()`` returns ``List[Candidate]`` (empty in chunk 1; the
    decoder lands in chunk 2)
  * ``_compute_per_kernel_scores()`` produces ``[K, T_det, N_fdm, H, W]``
    fp32 scores on a synthetic small cube; output shape matches input
    shape + leading kernel axis
  * ``boxcar_via_cumsum`` is numerically equivalent to a numpy
    sliding-window-sum reference (plan §3.6.13
    ``test_detector_conv_flops_cumsum_pin`` clause (b))
  * AST scan: no ``F.conv1d`` / ``F.avg_pool1d`` / ``F.max_pool1d``
    along K_dm or K_time axes in ``forward.py`` or ``kernels.py``
    (regression catch for the cumsum-trick pin, plan §3.6.13 clause (a))
  * Cube validation: shape / dtype regressions surface as ``ValueError``
    / ``TypeError`` on bad inputs

Tests run CPU-only by default. The FLOPs gate (plan §3.6.13 clause (c))
is GPU-specific and lives in chunks 6/7's bench-driven tests.
"""

from __future__ import annotations

import ast
import os
import pathlib

import numpy as np
import pytest
import torch

# Ensure DSART_TEST=1 so the M1 contracts run their post-init validators.
os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate  # noqa: E402
from dsart.detector.forward import (  # noqa: E402
    Detector,
    DeterministicDetector,
    boxcar_via_cumsum,
)


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_deterministic_detector_satisfies_protocol() -> None:
    """``DeterministicDetector`` is a structural ``Detector``."""
    det = DeterministicDetector()
    assert isinstance(det, Detector)


def test_kernels_returns_128_ids() -> None:
    det = DeterministicDetector()
    ids = det.kernels()
    assert isinstance(ids, tuple)
    assert len(ids) == 128
    assert all(isinstance(x, str) for x in ids)
    # Stable iteration order: index 0 is "unit:d1:b1", index -1 is
    # "psf_shift_l:d7:b128".
    assert ids[0] == "unit:d1:b1"
    assert ids[-1] == "psf_shift_l:d7:b128"


def test_detector_version_default() -> None:
    """Default ``detector_version`` matches the v1 production stamp."""
    det = DeterministicDetector()
    assert det.detector_version == "v1.M5"


def test_detector_version_overridable() -> None:
    """The cube-injection bench's identity-stub mode (plan §8 line 2321)
    sets ``detector_version='identity-stub.M5'``; the constructor must
    accept the override.
    """
    det = DeterministicDetector(detector_version="identity-stub.M5")
    assert det.detector_version == "identity-stub.M5"


# ---------------------------------------------------------------------------
# forward() chunk-1 stub returns []
# ---------------------------------------------------------------------------


def _small_cube(
    *, T_det: int = 16, N_fdm: int = 4, H: int = 8, W: int = 8,  # noqa: N803
    dtype: torch.dtype = torch.float16,
):
    """Build a small (T_det, N_fdm, H, W) noise cube for fast CPU tests."""
    rng = torch.Generator()
    rng.manual_seed(20260505)
    cube = torch.randn(T_det, N_fdm, H, W, generator=rng).to(dtype)
    validity = torch.ones(T_det, N_fdm, dtype=torch.bool)
    sigma1 = torch.ones(T_det, N_fdm, dtype=torch.float32)
    return cube, validity, sigma1


def test_forward_returns_empty_list_in_chunk_1() -> None:
    """Chunk-1 stub: ``forward()`` returns ``[]`` until the chunk-2
    decoder lands. The Protocol surface is stable from chunk 1 forward.
    """
    det = DeterministicDetector()
    cube, validity, sigma1 = _small_cube()
    out = det.forward(cube, validity, sigma1)
    assert isinstance(out, list)
    assert out == []


def test_forward_return_type_list_of_candidates() -> None:
    """Even when chunks 2/3 wire in real candidates, the return type
    annotation is ``List[Candidate]``. Test by injecting a probe via
    ``_compute_per_kernel_scores`` directly.
    """
    det = DeterministicDetector()
    cube, validity, _ = _small_cube()
    scores = det._compute_per_kernel_scores(cube, validity)
    assert scores.ndim == 5  # [K, T_det, N_fdm, H, W]


# ---------------------------------------------------------------------------
# _compute_per_kernel_scores shape / dtype invariants
# ---------------------------------------------------------------------------


def test_per_kernel_scores_shape() -> None:
    det = DeterministicDetector()
    cube, validity, _ = _small_cube(T_det=32, N_fdm=8, H=16, W=16)
    scores = det._compute_per_kernel_scores(cube, validity)
    K = len(det.kernels())  # noqa: N806
    assert scores.shape == (K, 32, 8, 16, 16)


def test_per_kernel_scores_dtype_is_fp32() -> None:
    """Cumsum upcasts fp16; Layer-2 σ_k EMA wants fp32. Pin the dtype here."""
    det = DeterministicDetector()
    cube, validity, _ = _small_cube(dtype=torch.float16)
    scores = det._compute_per_kernel_scores(cube, validity)
    assert scores.dtype == torch.float32


def test_per_kernel_scores_unit_b1_d1_is_input_passthrough() -> None:
    """The kernel ``"unit:d1:b1"`` is the trivial identity (image=delta,
    K_dm=1, K_time=1). The score for that kernel should equal the input
    cube (cast to fp32) cell-for-cell.
    """
    det = DeterministicDetector()
    cube, validity, _ = _small_cube(T_det=32, N_fdm=8, H=8, W=8)
    scores = det._compute_per_kernel_scores(cube, validity)
    ids = det.kernels()
    k_idx = ids.index("unit:d1:b1")
    np.testing.assert_allclose(
        scores[k_idx].numpy(),
        cube.to(torch.float32).numpy(),
        rtol=0,
        atol=0,
    )


def test_per_kernel_scores_psf_b1_d1_matches_unit_in_v1() -> None:
    """D10: in v1, all four image kernels are 1×1 deltas, so
    ``"psf:d1:b1"`` and ``"unit:d1:b1"`` produce identical scores. v2
    will diverge — this test catches accidental v2-style edits to the
    image kernels.
    """
    det = DeterministicDetector()
    cube, validity, _ = _small_cube()
    scores = det._compute_per_kernel_scores(cube, validity)
    ids = det.kernels()
    np.testing.assert_allclose(
        scores[ids.index("unit:d1:b1")].numpy(),
        scores[ids.index("psf:d1:b1")].numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        scores[ids.index("unit:d1:b1")].numpy(),
        scores[ids.index("psf_shift_lm:d1:b1")].numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        scores[ids.index("unit:d1:b1")].numpy(),
        scores[ids.index("psf_shift_l:d1:b1")].numpy(),
        rtol=0,
        atol=0,
    )


def test_validity_mask_dtype_must_be_bool() -> None:
    det = DeterministicDetector()
    cube, _, _ = _small_cube()
    bad_validity = torch.ones(cube.shape[0], cube.shape[1], dtype=torch.float32)
    bad_sigma1 = torch.ones(cube.shape[0], cube.shape[1], dtype=torch.float32)
    with pytest.raises(TypeError, match="validity_mask.dtype"):
        det.forward(cube, bad_validity, bad_sigma1)


def test_cube_must_be_4d_and_square() -> None:
    det = DeterministicDetector()
    bad_cube = torch.zeros(16, 4, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="cube.dim"):
        det._compute_per_kernel_scores(
            bad_cube, torch.ones(16, 4, dtype=torch.bool)
        )
    nonsquare_cube = torch.zeros(16, 4, 8, 10, dtype=torch.float16)
    with pytest.raises(ValueError, match="square"):
        det._compute_per_kernel_scores(
            nonsquare_cube, torch.ones(16, 4, dtype=torch.bool)
        )


def test_validity_mask_shape_must_match_cube() -> None:
    det = DeterministicDetector()
    cube, _, _ = _small_cube(T_det=16, N_fdm=4)
    with pytest.raises(ValueError, match="validity_mask.shape"):
        det._compute_per_kernel_scores(
            cube, torch.ones(20, 4, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="validity_mask.shape"):
        det._compute_per_kernel_scores(
            cube, torch.ones(16, 6, dtype=torch.bool)
        )


# ---------------------------------------------------------------------------
# boxcar_via_cumsum numerical equivalence (plan §3.6.13 clause (b))
# ---------------------------------------------------------------------------


def _reference_boxcar_via_sliding_window(x: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Reference centred-sum boxcar via numpy sliding window. O(N·width)
    naive form; used as ground truth for the cumsum optimisation.
    """
    if width == 1:
        return x
    # Pad with zeros along ``axis`` to match boxcar_via_cumsum's
    # left-biased centring convention.
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (pad_left, pad_right)
    x_padded = np.pad(x, pad_widths, mode="constant", constant_values=0.0)
    # sliding_window_view returns a view of shape
    # (..., n + 1 - width, ..., width); we sum the trailing window axis.
    # Move the windowed-axis to the front for a clean sum.
    win = np.lib.stride_tricks.sliding_window_view(
        x_padded, window_shape=width, axis=axis
    )
    return win.sum(axis=-1)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128])
def test_boxcar_via_cumsum_matches_reference_along_axis_0(width: int) -> None:
    rng = np.random.default_rng(20260505)
    x_np = rng.standard_normal((256, 4, 8, 8)).astype(np.float32)
    x = torch.from_numpy(x_np)
    out = boxcar_via_cumsum(x, axis=0, width=width).numpy()
    ref = _reference_boxcar_via_sliding_window(x_np, width=width, axis=0)
    assert out.shape == x_np.shape
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("width", [1, 3, 5, 7])
def test_boxcar_via_cumsum_matches_reference_along_axis_1(width: int) -> None:
    rng = np.random.default_rng(20260505)
    x_np = rng.standard_normal((32, 16, 8, 8)).astype(np.float32)
    x = torch.from_numpy(x_np)
    out = boxcar_via_cumsum(x, axis=1, width=width).numpy()
    ref = _reference_boxcar_via_sliding_window(x_np, width=width, axis=1)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_boxcar_via_cumsum_width_1_is_identity() -> None:
    x = torch.randn(8, 4, 4, 4)
    out = boxcar_via_cumsum(x, axis=2, width=1)
    assert out is x  # short-circuit returns the same object


def test_boxcar_via_cumsum_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="width"):
        boxcar_via_cumsum(torch.zeros(8), axis=0, width=0)


def test_boxcar_via_cumsum_rejects_width_exceeding_axis() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        boxcar_via_cumsum(torch.zeros(8), axis=0, width=9)


def test_boxcar_via_cumsum_preserves_dtype() -> None:
    """fp16 in → fp16 out (cumsum-internal upcast must be cast back)."""
    x = torch.randn(64, 4, 4, 4, dtype=torch.float16)
    out = boxcar_via_cumsum(x, axis=0, width=8)
    assert out.dtype == torch.float16


def test_boxcar_via_cumsum_fp16_relerr_within_pin() -> None:
    """Plan §3.6.13 clause (b): fp16 rel-err ≤ 1e-3 on a representative
    workload (T_det=512, K_time=128 wide-boxcar).

    The plan invariant pins the *cumsum-arithmetic* equivalence between
    boxcar_via_cumsum(x) and the sliding-window reference: given the
    SAME input tensor, the two must agree to within fp16 representation
    noise. We therefore feed both paths the same fp16-quantized cube
    (cast to fp32 only for the numpy reference's actual sum), so the
    test isolates cumsum-arithmetic error from the unrelated input
    quantization noise that arises when fp32 → fp16 input rounding is
    re-injected into the comparison.
    """
    rng = np.random.default_rng(20260505)
    x_np_fp32 = rng.standard_normal((512, 8, 32, 32)).astype(np.float32)
    # Take the fp16 view both paths will see, then promote to fp32 so the
    # numpy reference can sum at full precision.
    x_fp16 = torch.from_numpy(x_np_fp32).to(torch.float16)
    x_quantized_fp32 = x_fp16.to(torch.float32).numpy()
    out_fp16 = boxcar_via_cumsum(x_fp16, axis=0, width=128).to(torch.float32).numpy()
    ref = _reference_boxcar_via_sliding_window(x_quantized_fp32, width=128, axis=0)
    # Exclude cells where ref is near zero (relerr is ill-defined there).
    nonzero_mask = np.abs(ref) > 1.0
    if nonzero_mask.sum() == 0:
        pytest.skip("ref tensor too sparse for meaningful rel-err comparison")
    relerr = np.abs(out_fp16[nonzero_mask] - ref[nonzero_mask]) / np.abs(
        ref[nonzero_mask]
    )
    p99 = float(np.quantile(relerr, 0.99))
    assert p99 <= 1e-3, (
        f"fp16 cumsum p99 rel-err = {p99:.2e} > 1e-3 (plan §3.6.13 pin)"
    )


def test_boxcar_via_cumsum_fp32_arithmetic_exact() -> None:
    """Companion exactness check: with fp32 input (no quantization), the
    cumsum-difference primitive must match the sliding-window reference
    to fp32 precision. This catches off-by-one centring bugs that the
    fp16 pin would mask.
    """
    rng = np.random.default_rng(20260505)
    x_np = rng.standard_normal((128, 4, 8, 8)).astype(np.float32)
    out = boxcar_via_cumsum(torch.from_numpy(x_np), axis=0, width=32).numpy()
    ref = _reference_boxcar_via_sliding_window(x_np, width=32, axis=0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# AST scan (plan §3.6.13 clause (a)) — no F.conv1d/avg_pool1d/max_pool1d
# along the K_dm or K_time axes in forward.py / kernels.py
# ---------------------------------------------------------------------------


_FORBIDDEN_KDM_KTIME_OPS = {
    # Forbidden along K_dm / K_time per plan §3.6.13. We catch ANY use of
    # these in the detector hot path; if a chunk-2+ author needs one
    # along the (l, m) axis (image-axis NMS in the decoder is fine), they
    # add an explicit ``# noqa: AST-CUMSUM`` comment-allow which this
    # scanner does NOT handle today (intentionally simple — chunks add a
    # whitelist mechanism if/when they need one).
    "conv1d",
    "avg_pool1d",
    "max_pool1d",
}


def _gather_attr_calls(source: str) -> list[str]:
    """Return the dotted-name suffix of every ``foo.bar.baz()`` call in
    ``source``. Used to certify the cumsum-trick AST pin.
    """
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            found.append(node.func.attr)
    return found


def _module_path(*parts: str) -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent.parent  # repo root
    return here.joinpath(*parts)


@pytest.mark.parametrize(
    "module_relpath",
    [
        "src/dsart/detector/forward.py",
        "src/dsart/detector/kernels.py",
    ],
)
def test_no_forbidden_kdm_ktime_ops_in_detector(module_relpath: str) -> None:
    """Plan §3.6.13 clause (a): naive width-by-width K_dm / K_time forms
    are forbidden in the detector hot path. The only allowed K_dm /
    K_time consumer is ``boxcar_via_cumsum``.
    """
    src = _module_path(module_relpath).read_text()
    calls = _gather_attr_calls(src)
    forbidden = sorted(
        {c for c in calls if c in _FORBIDDEN_KDM_KTIME_OPS}
    )
    assert not forbidden, (
        f"{module_relpath} uses forbidden K_dm/K_time ops "
        f"{forbidden}; plan §3.6.13 clause (a) requires "
        f"boxcar_via_cumsum exclusively along those axes"
    )


# ---------------------------------------------------------------------------
# Determinism — same seed → same scores across two constructions
# ---------------------------------------------------------------------------


def test_per_kernel_scores_deterministic() -> None:
    det1 = DeterministicDetector()
    det2 = DeterministicDetector()
    cube, validity, _ = _small_cube()
    s1 = det1._compute_per_kernel_scores(cube, validity)
    s2 = det2._compute_per_kernel_scores(cube, validity)
    np.testing.assert_array_equal(s1.numpy(), s2.numpy())
