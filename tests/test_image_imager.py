"""Tests for ``dsart.image.imager`` (M5 chunk 6a).

Covers:
  * ``image_mask_npad`` — kernel-support floor and envelope cutoff;
    rejects bad inputs.
  * ``compute_edge_mask`` — interior is 1, ring is 0, DC cell is 0
    when ``drop_dc=True``, broadcasts cleanly against any leading
    batch dim of the image.
  * ``dirty_image_from_uv_grid`` — single-side identity
    ``Re(iFFT2(V_pos)) = 0.5 · iFFT2(V_full)`` (plan §3.6.11), DC-only
    grid produces a uniform image, parseval-ish energy preservation,
    consistency between numpy and torch backends, leading-batch-dim
    pass-through.
  * ``apply_edge_mask`` — multiplicative, dtype-preserving, zeros
    pixels in the ring while leaving interior untouched.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

from dsart.image.imager import (  # noqa: E402
    apply_edge_mask,
    compute_edge_mask,
    dirty_image_from_uv_grid,
    image_mask_npad,
)


# ---------------------------------------------------------------------------
# image_mask_npad
# ---------------------------------------------------------------------------


def test_image_mask_npad_kernel_floor() -> None:
    """When sigma_l_pix is None the envelope term is 0; npad = ceil(K/2)+2."""
    assert image_mask_npad(n_grid=64, kernel_support=5) == 4
    assert image_mask_npad(n_grid=64, kernel_support=7) == 5
    assert image_mask_npad(n_grid=64, kernel_support=1) == 2


def test_image_mask_npad_envelope_term_dominates() -> None:
    """If σ_l is small (envelope shrinks fast), pad grows toward N/2."""
    # σ_l_pix=8, N=64, threshold=0.5 (-3 dB)
    # rad = 8 · sqrt(2 ln 2) ≈ 9.42
    # pad_envelope = ceil(32 - 9.42) = 23
    npad = image_mask_npad(
        n_grid=64,
        kernel_support=5,
        sigma_l_pix=8.0,
        envelope_threshold=0.5,
    )
    assert npad == 23


def test_image_mask_npad_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="power of two"):
        image_mask_npad(n_grid=63, kernel_support=5)
    with pytest.raises(ValueError, match="power of two"):
        image_mask_npad(n_grid=0, kernel_support=5)
    with pytest.raises(ValueError, match="kernel_support"):
        image_mask_npad(n_grid=64, kernel_support=0)
    with pytest.raises(ValueError, match="envelope_threshold"):
        image_mask_npad(
            n_grid=64,
            kernel_support=5,
            sigma_l_pix=8.0,
            envelope_threshold=0.0,
        )


# ---------------------------------------------------------------------------
# compute_edge_mask
# ---------------------------------------------------------------------------


def test_compute_edge_mask_basic_shape_and_dtype() -> None:
    mask = compute_edge_mask(n_grid=32, kernel_support=5)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.float32
    assert ((mask == 0) | (mask == 1)).all()


def test_compute_edge_mask_interior_is_one_ring_is_zero() -> None:
    """For ``npad=4`` (K=5, no envelope), the inner [4, 28) × [4, 28) is 1
    except the DC cell at (16, 16); the outer ring is 0.
    """
    mask = compute_edge_mask(
        n_grid=32, kernel_support=5, drop_dc=True
    )
    assert mask[16, 16] == 0.0  # DC dropped
    interior = mask[4:28, 4:28].copy()
    interior[16 - 4, 16 - 4] = 1.0  # restore for "all-ones" check
    assert np.all(interior == 1.0)
    # Outer ring all zero:
    assert np.all(mask[:4, :] == 0.0)
    assert np.all(mask[-4:, :] == 0.0)
    assert np.all(mask[:, :4] == 0.0)
    assert np.all(mask[:, -4:] == 0.0)


def test_compute_edge_mask_drop_dc_false_keeps_dc() -> None:
    mask = compute_edge_mask(
        n_grid=16, kernel_support=3, drop_dc=False
    )
    dc = 16 // 2
    assert mask[dc, dc] == 1.0


def test_compute_edge_mask_collapses_when_pad_too_large() -> None:
    """If 2·npad ≥ N_grid the mask is all zeros (defensive)."""
    mask = compute_edge_mask(
        n_grid=8,
        kernel_support=5,
        sigma_l_pix=0.5,
        envelope_threshold=0.5,
    )
    assert np.all(mask == 0.0)


# ---------------------------------------------------------------------------
# dirty_image_from_uv_grid (numpy)
# ---------------------------------------------------------------------------


def test_dirty_image_dc_only_uniform_image_np() -> None:
    """A grid with only the DC cell at the fftshift-centre nonzero gives a
    uniform image with value (DC value) / N². ``+uv``-only ⇒ ``Re``-only;
    here the full grid is purely real so the imaginary part is zero
    everywhere by construction.
    """
    n = 8
    uv = np.zeros((n, n), dtype=np.complex64)
    uv[0, 0] = 1.0  # DC cell in UN-shifted layout (fftshift centre is wrap-of-DC)
    img = dirty_image_from_uv_grid(uv)
    np.testing.assert_allclose(img, np.full((n, n), 1.0 / n**2), rtol=1e-6, atol=1e-6)


def test_dirty_image_single_side_identity_np() -> None:
    """Single-side identity (plan §3.6.11): for a real ground-truth image
    with FFT ``V_full = FFT2(I)``, the +uv half ``V_pos`` defined as

        V_pos = V_full where u > 0, V_pos = 0.5 · V_full where u = 0,
        V_pos = 0 elsewhere

    satisfies ``2 · Re(iFFT2(V_pos)) == I``. This is the lossless
    +uv-only gridder identity that the §3.6.5 G5 gridder relies on.
    """
    rng = np.random.default_rng(0)
    n = 8
    img_truth = rng.standard_normal((n, n)).astype(np.float32)
    full_uv = np.fft.fft2(img_truth)
    # Build V_pos: keep +u half, halve u=0 column.
    v_pos = np.zeros_like(full_uv)
    v_pos[:, 1:n // 2] = full_uv[:, 1:n // 2]
    v_pos[:, 0] = 0.5 * full_uv[:, 0]
    # The Nyquist column at u = n/2 also needs to be halved (it's its
    # own conjugate-symmetry partner for even n):
    v_pos[:, n // 2] = 0.5 * full_uv[:, n // 2]
    # Run through the imager (which adds an fftshift on top of iFFT2):
    img_recovered = dirty_image_from_uv_grid(v_pos)
    # Undo the imager's fftshift to recover the un-shifted image:
    img_recovered_unshifted = np.fft.ifftshift(img_recovered)
    np.testing.assert_allclose(
        2.0 * img_recovered_unshifted, img_truth,
        rtol=1e-4, atol=1e-5,
    )


def test_dirty_image_batch_dim_pass_through_np() -> None:
    rng = np.random.default_rng(7)
    n = 8
    batch_shape = (3, 5)
    re = rng.standard_normal((*batch_shape, n, n)).astype(np.float32)
    im = rng.standard_normal((*batch_shape, n, n)).astype(np.float32)
    uv = (re + 1j * im).astype(np.complex64)
    img = dirty_image_from_uv_grid(uv)
    assert img.shape == (*batch_shape, n, n)
    # The combiner-side preserves leading batch axes; spot-check a couple
    # of slices match the per-slice fft path:
    expected_slice = np.fft.fftshift(np.fft.ifft2(uv[1, 2])).real
    np.testing.assert_allclose(img[1, 2], expected_slice, rtol=1e-5, atol=1e-5)


def test_dirty_image_out_dtype_np() -> None:
    n = 8
    uv = np.zeros((n, n), dtype=np.complex64)
    uv[0, 0] = 1.0
    img16 = dirty_image_from_uv_grid(uv, out_dtype=np.float16)
    assert img16.dtype == np.float16


# ---------------------------------------------------------------------------
# dirty_image_from_uv_grid (torch backend)
# ---------------------------------------------------------------------------


def test_dirty_image_torch_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    n = 16
    uv_np = (
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    ).astype(np.complex64)
    img_np = dirty_image_from_uv_grid(uv_np)
    img_t = dirty_image_from_uv_grid(torch.from_numpy(uv_np))
    assert isinstance(img_t, torch.Tensor)
    np.testing.assert_allclose(
        img_t.cpu().numpy(), img_np, rtol=1e-5, atol=1e-5
    )


def test_dirty_image_torch_out_dtype_cast() -> None:
    n = 8
    uv = torch.zeros((n, n), dtype=torch.complex64)
    uv[0, 0] = 1.0
    img16 = dirty_image_from_uv_grid(uv, out_dtype=torch.float16)
    assert img16.dtype == torch.float16


def test_dirty_image_torch_batch_pass_through() -> None:
    rng = np.random.default_rng(11)
    n = 8
    uv = (
        rng.standard_normal((4, n, n)) + 1j * rng.standard_normal((4, n, n))
    ).astype(np.complex64)
    img_np = dirty_image_from_uv_grid(uv)
    img_t = dirty_image_from_uv_grid(torch.from_numpy(uv)).cpu().numpy()
    np.testing.assert_allclose(img_t, img_np, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# apply_edge_mask
# ---------------------------------------------------------------------------


def test_apply_edge_mask_zeros_ring_only_np() -> None:
    n = 16
    img = np.ones((n, n), dtype=np.float32)
    mask = compute_edge_mask(n_grid=n, kernel_support=5, drop_dc=False)
    out = apply_edge_mask(img, mask)
    npad = image_mask_npad(n_grid=n, kernel_support=5)
    assert np.all(out[:npad, :] == 0.0)
    assert np.all(out[-npad:, :] == 0.0)
    assert np.all(out[npad:n - npad, npad:n - npad] == 1.0)


def test_apply_edge_mask_torch_dtype_preserved() -> None:
    n = 16
    img = torch.ones((n, n), dtype=torch.float16)
    mask = compute_edge_mask(n_grid=n, kernel_support=5, drop_dc=False)
    out = apply_edge_mask(img, mask)
    assert out.dtype == torch.float16
    assert out.shape == (n, n)


def test_apply_edge_mask_batch_broadcast_np() -> None:
    """Mask is [N, N]; broadcast over leading batch dims of the image."""
    n = 8
    batch = (2, 3)
    img = np.ones((*batch, n, n), dtype=np.float32)
    mask = compute_edge_mask(n_grid=n, kernel_support=3, drop_dc=False)
    out = apply_edge_mask(img, mask)
    assert out.shape == (*batch, n, n)
    npad = image_mask_npad(n_grid=n, kernel_support=3)
    assert np.all(out[..., :npad, :] == 0.0)
    assert np.all(
        out[..., npad:n - npad, npad:n - npad] == 1.0
    )
