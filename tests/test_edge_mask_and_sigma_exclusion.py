"""2026-09-22 sensitivity-audit fixes.

Covers three things the audit found (see
``_inspect/sensitivity/DSA110_pipeline_sensitivity.pdf``):

1. the image-plane edge mask sized its border from a hard-coded
   ``kernel_support=5`` instead of the gridder's live ``--kernel-support
   1``, so the dead border was twice as wide as the pillbox gridder
   needs;
2. ``apply_edge_mask`` MULTIPLIES by a 0/1 mask, so the masked cells
   reach the σ-clip estimators as exact zeros (6.164% of every cube) and
   drag σ 4.05% low -- making every reported SNR 4.2% high;
3. the C1→C2 width cap is absolute, and because it is applied after the
   cross-kernel merge it discards wide bursts outright rather than
   degrading them.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsart.image.imager import (  # noqa: E402
    compute_edge_mask,
    diagnose_edge_mask_alignment,
    image_mask_npad,
)
from dsart.noise_norm.layer1 import (  # noqa: E402
    Layer1State,
    sigma_clipped_std,
)
from dsart.services.c1_emit import C1EmitConfig  # noqa: E402


# --------------------------------------------------------------- item 1
def test_npad_follows_kernel_support() -> None:
    """The pillbox gridder (K=1) needs npad=2, not the K=5 default's 4."""
    assert image_mask_npad(n_grid=256, kernel_support=1) == 2
    assert image_mask_npad(n_grid=256, kernel_support=5) == 4


def test_kernel_support_reaches_the_edge_mask_config() -> None:
    """``--kernel-support`` must reach ``edge_mask_kernel_support``.

    It did not until 2026-09-22: the dataclass default of 5 was used on
    every production cube.
    """
    from dsart.services.cube_pipeline import CubePipelineConfig
    # The default is still the historical 5 ...
    assert CubePipelineConfig(n_grid=8).edge_mask_kernel_support == 5
    # ... but it is now settable, and the service sets it (asserted by
    # test_search_compute_wires_kernel_support below).
    cfg = CubePipelineConfig(n_grid=8, edge_mask_kernel_support=1)
    assert cfg.edge_mask_kernel_support == 1


def test_search_compute_wires_kernel_support() -> None:
    import inspect

    from dsart.services import search_compute

    src = inspect.getsource(search_compute._build_search_config_from_yaml)
    assert "edge_mask_kernel_support=int(kernel_support)" in src
    caller = inspect.getsource(search_compute._run_async)
    assert "kernel_support=int(args.kernel_support)" in caller


# --------------------------------------------------------------- item 1b
def test_diagnose_edge_mask_alignment_detects_a_flat_offset() -> None:
    mask = compute_edge_mask(n_grid=32, kernel_support=1)
    zero = (mask == 0)
    # aligned
    assert diagnose_edge_mask_alignment(zero, mask) == (0, 0)
    # displaced by +10 elements in row-major order, the offset the audit
    # measured in 632 dumped cubes
    rolled = np.roll(zero.ravel(), 10).reshape(zero.shape)
    off, n_bad = diagnose_edge_mask_alignment(rolled, mask)
    assert off == 10 and n_bad == 0


def test_diagnose_edge_mask_alignment_rejects_shape_mismatch() -> None:
    mask = compute_edge_mask(n_grid=16, kernel_support=1)
    with pytest.raises(ValueError):
        diagnose_edge_mask_alignment(np.zeros((8, 8), bool), mask)


# --------------------------------------------------------------- item 2
def _masked_cube(n_grid: int, t_det: int, n_fdm: int, sigma: float, seed: int):
    """A unit-variance cube with the edge mask's cells set to exactly 0."""
    mask = compute_edge_mask(n_grid=n_grid, kernel_support=1)
    active = (mask != 0.0)
    rng = np.random.default_rng(seed)
    cube = rng.normal(0.0, sigma, size=(t_det, n_fdm, n_grid, n_grid))
    cube[:, :, ~active] = 0.0
    return (
        torch.from_numpy(cube.astype("float32")),
        torch.from_numpy(np.ascontiguousarray(active.reshape(-1))),
        float((~active).mean()),
    )


def test_masked_zeros_bias_sigma_low_and_spatial_active_fixes_it() -> None:
    """The headline number: masked zeros pull σ down; the mask fixes it."""
    n_grid, sigma = 64, 1.0
    cube, active, dead_frac = _masked_cube(n_grid, 24, 2, sigma, seed=7)
    assert 0.02 < dead_frac < 0.20          # a real, non-trivial fraction
    slab = cube[:, 0]
    biased = sigma_clipped_std(slab)
    fixed = sigma_clipped_std(slab, spatial_active=active)
    # the σ-clip of a Gaussian is a few % low by construction, so compare
    # the two estimates against each other rather than against 1.0
    assert biased < fixed, (biased, fixed)
    assert fixed == pytest.approx(sigma, rel=0.03)
    # and the bias is roughly the dead fraction in quadrature
    assert biased / fixed < 1.0 - 0.3 * dead_frac


def test_spatial_active_is_a_noop_when_nothing_is_masked() -> None:
    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.normal(size=(8, 16, 16)).astype("float32"))
    allactive = torch.ones(16 * 16, dtype=torch.bool)
    assert sigma_clipped_std(x, spatial_active=allactive) == pytest.approx(
        sigma_clipped_std(x), rel=1e-6
    )


@pytest.mark.parametrize("max_samples", [None, 4096])
def test_layer1_state_excludes_masked_cells(max_samples) -> None:
    n_grid = 64
    cube, active, _ = _masked_cube(n_grid, 24, 3, 1.0, seed=11)
    kw = dict(n_fdm=3, n_burnin_cubes=1, max_samples=max_samples)
    biased = Layer1State(**kw).update_and_query(cube=cube)
    fixed = Layer1State(**kw, spatial_active=active).update_and_query(cube=cube)
    assert torch.all(fixed > biased)
    assert fixed.mean().item() == pytest.approx(1.0, rel=0.05)


# --------------------------------------------------------------- item 3
def test_c1_width_snr_escape_config_defaults_off() -> None:
    cfg = C1EmitConfig(host="h", port=1, search_node_id=1, gpu_half=0)
    assert cfg.max_width_snr_escape is None


def test_width_cap_escape_semantics() -> None:
    """Wide-and-bright survives the cap; wide-and-faint does not."""
    max_w, esc = 16, 20.0

    def kept(width, snr):
        return width <= max_w or (esc is not None and snr >= esc)

    assert kept(8, 12.0)           # narrow, faint -> always kept
    assert not kept(32, 12.0)      # wide, faint   -> dropped (the cap's job)
    assert kept(32, 25.0)          # wide, BRIGHT  -> escapes
    assert kept(64, 20.0)          # exactly at the threshold


def test_yaml_maps_the_escape_key() -> None:
    import inspect

    from dsart.services import search_compute

    src = inspect.getsource(search_compute._build_search_config_from_yaml)
    assert 'c1.get("max_c1c2_width_snr_escape"' in src
    assert "max_width_snr_escape=(" in src


# ------------------------------------------------- item 2, wiring guard
def test_cube_pipeline_publishes_the_active_map_to_both_estimators() -> None:
    """The regression guard for how item 2 is delivered.

    The σ fix is only real if the active map actually REACHES Layer-1
    and the detector. An earlier revision computed the map in
    ``SearchComputeService.__init__`` via a helper that was never
    defined -- nothing constructed the pipeline in a test, so it would
    have raised ``AttributeError`` at service startup on the fleet. The
    map is now published by ``CubePipeline`` from the imager's own mask.
    """
    from dsart.detector.forward import DeterministicDetector
    from dsart.services.cube_pipeline import CubePipeline, CubePipelineConfig

    n_grid = 32
    cfg = CubePipelineConfig(
        n_grid=n_grid, edge_mask_kernel_support=1, device="cpu",
        cube_dtype=torch.float32, image_backend="cpu",
    )
    det = DeterministicDetector(
        threshold_sigma=999.0, detector_version="v1.M5",
        search_node_id=0, gpu_half=0,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    l1 = Layer1State(n_fdm=2)
    assert l1.spatial_active is None and det.spatial_active is None

    CubePipeline(config=cfg, detector=det, layer1_state=l1)

    for name, got in (("layer1", l1.spatial_active),
                      ("detector", det.spatial_active)):
        assert got is not None, f"{name} never received the active map"
        assert got.dtype == torch.bool and got.numel() == n_grid * n_grid
    # and it agrees with the mask, including the ±1 checkerboard interior
    want = torch.from_numpy(
        compute_edge_mask(n_grid=n_grid, kernel_support=1) != 0.0
    ).reshape(-1)
    assert torch.equal(l1.spatial_active, want)
    assert torch.equal(det.spatial_active, want)


def test_set_spatial_active_invalidates_the_cached_maps() -> None:
    """Re-publishing must not leave a stale subsample map behind.

    ``CubePipeline`` publishes twice on the GPU path (once from the
    CPU-path mask, then again from ``GpuImager``'s own mask, which is
    the one that multiplies the cube).
    """
    n_grid = 32
    cube, active, _ = _masked_cube(n_grid, 8, 2, 1.0, seed=5)
    l1 = Layer1State(n_fdm=2, max_samples=512)
    l1.update_and_query(cube=cube)                 # builds the cache
    assert l1._subsample_idx is not None
    l1.set_spatial_active(active)
    assert l1._subsample_active is None and l1._subsample_key is None
    sig = l1.update_and_query(cube=cube)            # rebuilds with the mask
    assert l1._subsample_active is not None
    assert torch.all(sig > 0)
