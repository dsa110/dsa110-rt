"""Galactic HI protection band for the slow-correlator flagger.

Why this exists
===============

The offline study of five voltage dumps (2026-09-08) found that **the
online flagger excises the Galactic HI line at up to 99.5% occupancy**.
That is the right call on the fast path — the FRB search does not want
a bright 21 cm line in its dedispersion trials, and HI is fully
resolved out on every baseline anyway (it carries 28% of its channel's
power but reaches only ``|V| = 0.003``, exactly the clean-channel
floor, because at 21 cm even a 6 m spacing resolves anything broader
than ~2 degrees).

On the **slow** path it is the wrong call. Slow visibilities go to
UVH5 for imaging and calibration, and there HI is science. So when
flagging is enabled on the slow correlator, this module carves out a
protected band that no detector may flag.

Velocity convention
===================

Limits are given as radio velocities relative to the HI rest
frequency:

    v = c (nu_0 - nu) / nu_0        =>       nu = nu_0 (1 - v/c)

so a *negative* velocity (blueshift) is a *higher* frequency. The
default −350 to +200 km/s spans 1419.458–1422.064 GHz, comfortably
covering Galactic emission at OVRO's latitude including high-velocity
clouds.

Scope
=====

At the DSA-110 channel plan the guard lands entirely inside
**chgroup 6, local channels 209-294** (86 of 6144 channels). It is a
no-op on the other fifteen corr nodes, which is worth knowing before
wondering why the guard "does nothing" on the node you are testing on.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

from dsart.common.constants import NCHAN_PER_CHGROUP, freq_GHz

LOG = logging.getLogger("dsart.rfi.hi_guard")

#: HI rest frequency, IAU 2009 best value.
HI_REST_GHZ: Final[float] = 1.420405751768

#: Speed of light, km/s.
_C_KMS: Final[float] = 299792.458

#: Default radio-velocity limits of the protected band, km/s.
#: Negative = blueshifted = higher frequency.
HI_GUARD_V_LO_KMS: Final[float] = -350.0
HI_GUARD_V_HI_KMS: Final[float] = 200.0


def hi_guard_freq_range_GHz(
    v_lo_kms: float = HI_GUARD_V_LO_KMS,
    v_hi_kms: float = HI_GUARD_V_HI_KMS,
) -> tuple[float, float]:
    """``(f_min, f_max)`` in GHz for a radio-velocity window.

    Raises:
        ValueError: ``v_lo_kms >= v_hi_kms`` (the window is empty or
            inverted, which almost certainly means the arguments were
            swapped).
    """
    if v_lo_kms >= v_hi_kms:
        raise ValueError(
            f"v_lo_kms={v_lo_kms} must be < v_hi_kms={v_hi_kms}; "
            "these are velocities, and the LOW velocity gives the HIGH "
            "frequency"
        )
    f_hi = HI_REST_GHZ * (1.0 - v_lo_kms / _C_KMS)
    f_lo = HI_REST_GHZ * (1.0 - v_hi_kms / _C_KMS)
    return (f_lo, f_hi)


def hi_guard_channels(
    chgroup: int,
    *,
    v_lo_kms: float = HI_GUARD_V_LO_KMS,
    v_hi_kms: float = HI_GUARD_V_HI_KMS,
    n_chan: int = NCHAN_PER_CHGROUP,
) -> np.ndarray:
    """Bool mask ``[n_chan]``, True for channels inside the guard.

    Args:
        chgroup: this node's chgroup index (0-15). Needed because the
            channel-to-frequency map depends on it.
        v_lo_kms, v_hi_kms: radio-velocity limits of the guard.
        n_chan: channels per chgroup.

    Returns:
        A bool array; all-False on every chgroup that does not contain
        the HI line, which is fifteen of the sixteen.
    """
    f_lo, f_hi = hi_guard_freq_range_GHz(v_lo_kms, v_hi_kms)
    freqs = np.array(
        [freq_GHz(int(chgroup), c) for c in range(int(n_chan))],
        dtype=np.float64,
    )
    return (freqs >= f_lo) & (freqs <= f_hi)


def describe_hi_guard(
    chgroup: int,
    *,
    v_lo_kms: float = HI_GUARD_V_LO_KMS,
    v_hi_kms: float = HI_GUARD_V_HI_KMS,
) -> str:
    """One-line human summary, for the service's startup log."""
    f_lo, f_hi = hi_guard_freq_range_GHz(v_lo_kms, v_hi_kms)
    guard = hi_guard_channels(
        chgroup, v_lo_kms=v_lo_kms, v_hi_kms=v_hi_kms,
    )
    n = int(guard.sum())
    if n == 0:
        return (
            f"HI guard {v_lo_kms:+.0f}..{v_hi_kms:+.0f} km/s "
            f"({f_lo * 1e3:.3f}-{f_hi * 1e3:.3f} MHz): no channels in "
            f"chgroup {chgroup} — inactive on this node"
        )
    idx = np.flatnonzero(guard)
    return (
        f"HI guard {v_lo_kms:+.0f}..{v_hi_kms:+.0f} km/s "
        f"({f_lo * 1e3:.3f}-{f_hi * 1e3:.3f} MHz): {n} channels "
        f"{int(idx[0])}-{int(idx[-1])} of chgroup {chgroup} are "
        f"PROTECTED from flagging"
    )
