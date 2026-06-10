"""Tests for the 2026-06-10 bright-burst / fp16-overflow hardening
(``CubePipelineConfig.detector_input_clip_sigma``).

Failure mode being guarded: a very bright burst overflows the
cuFFT-cfp16 butterflies → ±inf at the burst pixels → the post-imager
``_clamp_inf_to_finite`` rewrites them to ±60000 (finite). The
detector's boxcar stages then SUM those cells: a w=4 time boxcar over
60000-valued cells is 240000 → fp16 inf in the score tensor, and the
subsequent DM-axis boxcar computes inf − inf = NaN — the burst (and
its whole tile) silently vanishes. Observed live 2026-06-10: probes at
≥1.4e-3 Jy·ms produced NO candidates while 0.5-1.0e-3 Jy·ms probes
matched cleanly at SNR 30-50.

With the clip enabled, the σ-normalised cube is clamped to ±C before
``Detector.forward()`` so every downstream boxcar sum stays finite in
fp16 and the burst degrades to a saturated-but-reported candidate.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

from dsart.detector.forward import (  # noqa: E402
    DeterministicDetector,
    boxcar_from_padded_cumsum,
    precompute_padded_cumsum,
)
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)


def _make_pipe(*, clip: float) -> CubePipeline:
    cfg = CubePipelineConfig(
        n_grid=16,
        device="cpu",
        cube_dtype=torch.float16,
        image_backend="cpu",
        detector_input_clip_sigma=clip,
    )
    det = DeterministicDetector(
        threshold_sigma=999.0,
        detector_version="v1.M5",
        search_node_id=0,
        gpu_half=0,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    return CubePipeline(config=cfg, detector=det)


def _cube_with_plateau(value: float = 60000.0) -> torch.Tensor:
    """[T=8, F=2, 16, 16] fp16 cube: unit noise floor + a 4-sample
    ``value`` plateau at one pixel (the post-nan_to_num artefact
    signature of an fp16-overflowed burst)."""
    torch.manual_seed(0)
    cube = torch.randn(8, 2, 16, 16).to(torch.float16)
    cube[2:6, 1, 8, 8] = value
    cube[3, 0, 4, 4] = -value
    return cube


def test_clip_clamps_plateau_in_place() -> None:
    pipe = _make_pipe(clip=250.0)
    cube = _cube_with_plateau()
    out = pipe._apply_detector_input_clip(cube)
    assert out is cube, "clip must be in-place (no copy)"
    assert float(cube.max()) <= 250.0
    assert float(cube.min()) >= -250.0
    assert float(cube[2, 1, 8, 8]) == 250.0
    assert float(cube[3, 0, 4, 4]) == -250.0
    # Noise-floor cells untouched.
    assert abs(float(cube[0, 0, 0, 0])) < 250.0


def test_clip_disabled_is_noop() -> None:
    pipe = _make_pipe(clip=0.0)
    cube = _cube_with_plateau()
    before = cube.clone()
    pipe._apply_detector_input_clip(cube)
    assert torch.equal(cube, before), "clip<=0 must be a strict no-op"


def test_layer1_normalise_applies_clip_on_unit_sigma_path() -> None:
    """The layer1_state=None (unit-σ) path must still clip — that is
    the path the synthetic-injection benches exercise."""
    pipe = _make_pipe(clip=250.0)
    cube = _cube_with_plateau()
    cube_norm, sigma = pipe._layer1_normalise(cube)
    assert float(cube_norm.max()) <= 250.0
    assert torch.all(sigma == 1.0)


def test_boxcar_overflow_regression() -> None:
    """Demonstrate the exact failure: fp16 boxcar over a ±60000
    plateau goes inf/NaN; over the clipped cube it stays finite."""
    def boxcar_w4(x: torch.Tensor) -> torch.Tensor:
        cs = precompute_padded_cumsum(x, axis=0, max_width=4)
        return boxcar_from_padded_cumsum(
            cs, axis=0, width=4, max_width=4,
            n_out=x.shape[0], out_dtype=torch.float16,
        )

    raw = _cube_with_plateau(60000.0)
    scores_raw = boxcar_w4(raw)
    assert bool(torch.isinf(scores_raw).any()), (
        "regression setup: unclipped 60000-plateau must overflow the "
        "fp16 boxcar output (4 × 60000 > 65504)"
    )

    pipe = _make_pipe(clip=250.0)
    clipped = pipe._apply_detector_input_clip(_cube_with_plateau(60000.0))
    scores_clipped = boxcar_w4(clipped)
    assert bool(torch.isfinite(scores_clipped).all()), (
        "clipped cube must produce fully finite boxcar scores"
    )
    # The burst is still prominently visible (saturated, not vanished).
    # The centred w=4 window at t=4 covers samples 2..5 == the full
    # plateau (4 × 250 = 1000).
    assert float(scores_clipped[4, 1, 8, 8]) >= 4 * 250.0 * 0.99


def test_config_default_is_off() -> None:
    """Library default is 0 (off) so benches/tests are bit-identical
    unless they opt in; production turns it on via the yaml shim
    (``detector.input_clip_sigma``, default 250)."""
    cfg = CubePipelineConfig(n_grid=16)
    assert cfg.detector_input_clip_sigma == 0.0
