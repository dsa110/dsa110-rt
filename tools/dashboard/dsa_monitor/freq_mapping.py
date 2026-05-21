"""Frequency-axis helpers for the dashboard's per-antenna plots.

Builds the concatenated freq axis spanning all 16 chgroups for both
the full-resolution (NCHAN_PER_CHGROUP = 384) and the downsampled
(NCHAN_DS = 96 at 4× downsample) cases. Channel ordering is
descending-in-frequency within each chgroup (the DSA-110
convention; see :func:`dsart.common.constants.freq_GHz`).

Concatenation runs in increasing chgroup index (chgroup 0 first,
chgroup 15 last). The chgroup-to-chgroup boundary frequencies are
contiguous (NU_CHGROUP_BOT_GHZ[g] == NU_CHGROUP_TOP_GHZ[g+1] - Δν),
so the concatenated axis is a strictly monotonic descending freq
axis across the full processed band (1.480 → 1.311 GHz).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from dsart.common.constants import (
    DELTA_NU_CH_GHZ,
    N_CHGROUP,
    NCHAN_PER_CHGROUP,
    freq_GHz,
)


def chgroup_freq_table_GHz() -> np.ndarray:
    """Full-resolution ``(N_CHGROUP, NCHAN_PER_CHGROUP)`` table."""
    return np.asarray(
        [
            [freq_GHz(g, ch) for ch in range(NCHAN_PER_CHGROUP)]
            for g in range(N_CHGROUP)
        ],
        dtype=np.float64,
    )


def downsampled_chgroup_freq_table_GHz(ds_factor: int) -> np.ndarray:
    """Build the freq table for a frequency-downsampled aggregator.

    Each downsampled bin gets the **mean** frequency of its
    constituent fine channels (consistent with the
    :class:`dsart.services.rfi_window.RFIWindowAggregator`'s
    sum-then-divide accumulation semantics).
    """
    if NCHAN_PER_CHGROUP % ds_factor != 0:
        raise ValueError(
            f"ds_factor={ds_factor} must divide "
            f"NCHAN_PER_CHGROUP={NCHAN_PER_CHGROUP}"
        )
    full = chgroup_freq_table_GHz()                  # (G, NCHAN)
    n_chan_ds = NCHAN_PER_CHGROUP // ds_factor
    # Reshape -> (G, n_chan_ds, ds_factor); mean over the inner axis.
    return full.reshape(N_CHGROUP, n_chan_ds, ds_factor).mean(axis=2)


def concat_axis_GHz(table_GHz: np.ndarray) -> np.ndarray:
    """Flatten an (N_CHGROUP, NCHAN_DS) table to a single descending
    monotone freq axis of length ``N_CHGROUP * NCHAN_DS``."""
    if table_GHz.shape[0] != N_CHGROUP:
        raise ValueError(
            f"freq table chgroup axis = {table_GHz.shape[0]}, "
            f"expected {N_CHGROUP}"
        )
    return table_GHz.reshape(-1).astype(np.float64, copy=False)


def chgroup_boundary_freqs_GHz(table_GHz: np.ndarray) -> np.ndarray:
    """Return the (N_CHGROUP-1,) array of inter-chgroup boundary freqs,
    for placing vertical separator lines on the concatenated plot."""
    G = table_GHz.shape[0]
    boundaries = np.empty(G - 1, dtype=np.float64)
    for g in range(G - 1):
        # Boundary = midpoint of last bin of g and first bin of g+1.
        boundaries[g] = 0.5 * (table_GHz[g, -1] + table_GHz[g + 1, 0])
    return boundaries


# Cache the production-default 4× downsampled table at import time.
_FREQ_TABLE_DS4_GHZ: Final[np.ndarray] = downsampled_chgroup_freq_table_GHz(4)
_FREQ_AXIS_DS4_GHZ: Final[np.ndarray] = concat_axis_GHz(_FREQ_TABLE_DS4_GHZ)


def production_freq_axis_GHz() -> np.ndarray:
    """The 1536-element descending freq axis used by the production
    M7.6 aggregator (freq_downsample=4)."""
    return _FREQ_AXIS_DS4_GHZ.copy()


def production_chgroup_boundaries_GHz() -> np.ndarray:
    """The 15 inter-chgroup boundary freqs for the production axis."""
    return chgroup_boundary_freqs_GHz(_FREQ_TABLE_DS4_GHZ)
