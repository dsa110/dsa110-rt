"""kernel_id validation accepts the x2 b3x time tokens without changing the 1x default bank."""

import pytest

from dsart.common.constants import (
    DETECTOR_K_TIME_WIDTHS,
    DETECTOR_TIME_KERNELS,
    DETECTOR_TIME_KERNELS_VALID,
    DETECTOR_TIME_KERNELS_X2_B3X,
)
from dsart.common.contracts import _check_kernel_id


def test_1x_bank_unchanged():
    assert DETECTOR_TIME_KERNELS == tuple(f"b{w}" for w in DETECTOR_K_TIME_WIDTHS)
    assert DETECTOR_TIME_KERNELS_VALID[:len(DETECTOR_TIME_KERNELS)] == DETECTOR_TIME_KERNELS


@pytest.mark.parametrize("t", DETECTOR_TIME_KERNELS_X2_B3X + DETECTOR_TIME_KERNELS)
def test_tokens_accepted(t):
    _check_kernel_id(f"{_img()}:{_dm()}:{t}")


@pytest.mark.parametrize("t", ["q2", "t4", "b3", "g8", ""])
def test_unknown_tokens_rejected(t):
    with pytest.raises(ValueError):
        _check_kernel_id(f"{_img()}:{_dm()}:{t}")


def _img():
    from dsart.common.constants import DETECTOR_IMAGE_KERNELS
    return DETECTOR_IMAGE_KERNELS[0]


def _dm():
    from dsart.common.constants import DETECTOR_DM_KERNELS
    return DETECTOR_DM_KERNELS[0]
