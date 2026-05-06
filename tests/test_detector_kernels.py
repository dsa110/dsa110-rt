"""Tests for ``dsart.detector.kernels`` (M5 chunk 1).

Verifies the v1 kernel-bank construction:

  * 128 triples produced at default tokens (4 image × 4 dm × 8 time, D2)
  * Every kernel_id parses cleanly via ``Candidate._check_kernel_id`` from
    M1 contracts (so trigger packets carry valid kernel_ids end-to-end)
  * All four image-kernel slots are L1-normalised 1×1 deltas in v1 (D10)
  * Iteration order is deterministic and stable (image outer, dm middle,
    time inner) — chunk-3 σ_k EMA + chunk-2 decoder rely on this index
    layout
  * Token-validation rejects out-of-namespace inputs (defensive)
"""

from __future__ import annotations

import os

import pytest
import torch

# Ensure DSART_TEST=1 so M1 contracts run their post-init validators.
os.environ.setdefault("DSART_TEST", "1")

from dsart.common.constants import (  # noqa: E402
    DETECTOR_DM_KERNELS,
    DETECTOR_IMAGE_KERNELS,
    DETECTOR_K_DM_WIDTHS,
    DETECTOR_K_TIME_WIDTHS,
    DETECTOR_TIME_KERNELS,
)
from dsart.common.contracts import _check_kernel_id  # noqa: E402
from dsart.detector.kernels import (  # noqa: E402
    DEFAULT_DETECTOR_DTYPE,
    Kernel,
    build_kernel_bank,
    make_image_kernel,
)


# ---------------------------------------------------------------------------
# Bank construction
# ---------------------------------------------------------------------------


def test_default_bank_size_is_128() -> None:
    """D2: K_img × K_dm × K_time = 4 × 4 × 8 = 128 default triples."""
    bank = build_kernel_bank()
    assert len(bank) == 128
    assert len(bank) == (
        len(DETECTOR_IMAGE_KERNELS)
        * len(DETECTOR_DM_KERNELS)
        * len(DETECTOR_TIME_KERNELS)
    )


def test_bank_kernel_ids_are_unique_and_well_formed() -> None:
    bank = build_kernel_bank()
    ids = [k.kernel_id for k in bank]
    assert len(ids) == len(set(ids)), "duplicate kernel_id"
    for kid in ids:
        # M1 contracts._check_kernel_id raises on bad form / token.
        _check_kernel_id(kid)


def test_bank_iteration_order_is_image_dm_time() -> None:
    """Index ordering invariant: ``k = i_img * 4 * 8 + i_dm * 8 + i_time``.

    Chunk-2 decoder + chunk-3 σ_k EMA + cube_injection FAR analytics all
    rely on this layout (D2 + plan §4.4 noise_norm interior-EMA struct).
    """
    bank = build_kernel_bank()
    n_dm = len(DETECTOR_DM_KERNELS)
    n_time = len(DETECTOR_TIME_KERNELS)
    for i_img, image_token in enumerate(DETECTOR_IMAGE_KERNELS):
        for i_dm, dm_token in enumerate(DETECTOR_DM_KERNELS):
            for i_time, time_token in enumerate(DETECTOR_TIME_KERNELS):
                k_idx = i_img * n_dm * n_time + i_dm * n_time + i_time
                k = bank[k_idx]
                assert k.image_token == image_token
                assert k.dm_token == dm_token
                assert k.time_token == time_token
                assert k.kernel_id == f"{image_token}:{dm_token}:{time_token}"


def test_bank_widths_match_token_decoding() -> None:
    bank = build_kernel_bank()
    for k in bank:
        assert k.k_dm_width == int(k.dm_token[1:])
        assert k.k_time_width == int(k.time_token[1:])
        assert k.k_dm_width in DETECTOR_K_DM_WIDTHS
        assert k.k_time_width in DETECTOR_K_TIME_WIDTHS


def test_bank_widths_obey_parity_invariants() -> None:
    """K_dm widths are odd (centred boxcar); K_time widths are
    powers of two (per ``common.constants`` docstrings).
    """
    bank = build_kernel_bank()
    for k in bank:
        assert k.k_dm_width % 2 == 1, (
            f"K_dm width must be odd; got {k.k_dm_width} for {k.kernel_id}"
        )
        assert (k.k_time_width & (k.k_time_width - 1)) == 0, (
            f"K_time width must be power of 2; got {k.k_time_width} "
            f"for {k.kernel_id}"
        )


def test_bank_record_is_frozen_and_slots() -> None:
    """``Kernel`` is a frozen dataclass with slots — ABI-stable + immutable.

    Mutating fields raises ``dataclasses.FrozenInstanceError``.
    """
    import dataclasses

    bank = build_kernel_bank()
    k = bank[0]
    assert isinstance(k, Kernel)
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.kernel_id = "tampered"  # type: ignore[misc]


def test_subset_bank_size() -> None:
    bank = build_kernel_bank(
        image_tokens=("unit",),
        dm_tokens=("d1", "d3"),
        time_tokens=("b1", "b16", "b128"),
    )
    assert len(bank) == 6
    assert {k.kernel_id for k in bank} == {
        "unit:d1:b1", "unit:d1:b16", "unit:d1:b128",
        "unit:d3:b1", "unit:d3:b16", "unit:d3:b128",
    }


def test_invalid_token_raises() -> None:
    with pytest.raises(ValueError, match="image_token"):
        build_kernel_bank(image_tokens=("not_an_image_token",))
    with pytest.raises(ValueError, match="dm_token"):
        build_kernel_bank(dm_tokens=("d99",))
    with pytest.raises(ValueError, match="time_token"):
        build_kernel_bank(time_tokens=("b3",))


# ---------------------------------------------------------------------------
# Image-kernel construction (D10: all four are 1×1 delta in v1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image_token", list(DETECTOR_IMAGE_KERNELS))
def test_v1_image_kernel_is_1x1_delta(image_token: str) -> None:
    """D10: all four v1 image-kernel slots ship as 1×1 delta tensors."""
    k = make_image_kernel(image_token)
    assert k.shape == (1, 1), (
        f"image_token={image_token!r}: v1 must be 1×1 delta per D10; "
        f"got shape {tuple(k.shape)}"
    )
    assert k.dtype == DEFAULT_DETECTOR_DTYPE
    assert float(k.item()) == 1.0


@pytest.mark.parametrize("image_token", list(DETECTOR_IMAGE_KERNELS))
def test_v1_image_kernel_l1_normalised(image_token: str) -> None:
    """L1 normalisation (Σ_cells = 1) per plan §3.6.5 G6 / line 486."""
    k = make_image_kernel(image_token)
    assert pytest.approx(float(k.abs().sum()), abs=1e-6) == 1.0


def test_make_image_kernel_rejects_unknown_token() -> None:
    with pytest.raises(ValueError):
        make_image_kernel("not_a_real_token")


# ---------------------------------------------------------------------------
# Kernel buffer sharing in the bank (v1 invariant: all 4 image kernels are
# the same delta tensor; chunk-2 may grow this when v2 PSF kernels land)
# ---------------------------------------------------------------------------


def test_v1_image_kernel_buffers_are_shared_across_image_tokens() -> None:
    """In v1, all four image kernels are identical 1×1 deltas — no point
    in keeping four separate buffers. ``DeterministicDetector`` shares
    buffers; this test certifies the underlying construction supports it.
    """
    bank = build_kernel_bank()
    image_kernels_by_token: dict[str, torch.Tensor] = {}
    for k in bank:
        if k.image_token not in image_kernels_by_token:
            image_kernels_by_token[k.image_token] = k.image_kernel
        else:
            # Same per-token tensor identity (build_kernel_bank reuses one
            # tensor per image_token across all dm × time inner products).
            assert k.image_kernel is image_kernels_by_token[k.image_token]
    # All four v1 image kernels are numerically identical (per D10).
    refs = list(image_kernels_by_token.values())
    for r in refs[1:]:
        assert torch.equal(refs[0], r)
