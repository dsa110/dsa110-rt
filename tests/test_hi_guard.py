"""Unit tests for :mod:`dsart.rfi.hi_guard`.

The guard exists because the offline study measured the online flagger
excising Galactic HI at up to 99.5% occupancy — correct on the fast
path, wrong on the slow path where the visibilities go to UVH5 for
imaging and calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsart.common.constants import NCHAN_PER_CHGROUP, N_CHGROUP, freq_GHz
from dsart.rfi.hi_guard import (
    HI_REST_GHZ,
    describe_hi_guard,
    hi_guard_channels,
    hi_guard_freq_range_GHz,
)


def test_default_band_matches_the_requested_velocities():
    """-350 to +200 km/s radio velocity -> 1419.458-1422.064 MHz."""
    f_lo, f_hi = hi_guard_freq_range_GHz()
    assert f_lo * 1e3 == pytest.approx(1419.458, abs=0.002)
    assert f_hi * 1e3 == pytest.approx(1422.064, abs=0.002)
    # A negative (blueshifted) velocity must give the HIGHER frequency.
    assert f_hi > HI_REST_GHZ > f_lo


def test_guard_lands_only_in_chgroup_6():
    """86 channels, all in chgroup 6 — so the guard is a no-op on
    fifteen of the sixteen corr nodes. Worth pinning: someone testing
    on any other node will see it do nothing, and that is correct."""
    per_group = {
        g: int(hi_guard_channels(g).sum()) for g in range(N_CHGROUP)
    }
    assert per_group[6] == 86
    assert all(v == 0 for g, v in per_group.items() if g != 6)
    idx = np.flatnonzero(hi_guard_channels(6))
    assert (int(idx[0]), int(idx[-1])) == (209, 294)
    # Contiguous — a gap would mean the frequency map is not monotonic.
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))


def test_every_guarded_channel_is_inside_the_band():
    f_lo, f_hi = hi_guard_freq_range_GHz()
    guard = hi_guard_channels(6)
    for c in range(NCHAN_PER_CHGROUP):
        f = freq_GHz(6, c)
        assert bool(guard[c]) == (f_lo <= f <= f_hi), (c, f)


def test_hi_rest_frequency_is_protected():
    """The line itself must be inside the guard, whatever the limits."""
    guard = hi_guard_channels(6)
    nearest = min(
        range(NCHAN_PER_CHGROUP),
        key=lambda c: abs(freq_GHz(6, c) - HI_REST_GHZ),
    )
    assert bool(guard[nearest])


def test_wider_velocities_protect_more_channels():
    narrow = int(hi_guard_channels(6, v_lo_kms=-100, v_hi_kms=100).sum())
    wide = int(hi_guard_channels(6, v_lo_kms=-500, v_hi_kms=500).sum())
    assert wide > narrow > 0


def test_inverted_velocity_window_raises():
    """Swapping the limits is an easy mistake — the LOW velocity gives
    the HIGH frequency — so it must fail loudly, not silently protect
    nothing."""
    with pytest.raises(ValueError, match="must be <"):
        hi_guard_freq_range_GHz(200.0, -350.0)


def test_describe_is_honest_about_inactive_nodes():
    assert "PROTECTED" in describe_hi_guard(6)
    assert "inactive on this node" in describe_hi_guard(0)
