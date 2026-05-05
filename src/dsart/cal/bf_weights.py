"""Loader for legacy `beamformer_weights_*.dat` calibration blobs (D17).

These are the binary cal solutions used by `dsaX_bfCorr` in beamformer
mode (`bfCorr.cu:1387-1392`). M2 uses them for the test-only
`corr_slow_compute --apply-cal` flag (D17 / F17 in
``M2_PLAN_FIXES.md``); production slow visibilities are uncalibrated.

File format (74,496 bytes total = 18,624 × float32 LE):

    h_winp[0       : 96]                    antpos_e[ant]    (m, ITRF east)
    h_winp[96      : 192]                   antpos_n[ant]    (m, ITRF north)
    h_winp[192     : 192 + 96*48*2*2]       cal[ant, ch_coarse, pol, ri]
                                            ri=0 → real, ri=1 → imag

The cal axis ordering matches the yaml metadata
(``cal_solutions.weights_axis0: antenna``,
 ``weights_axis1: frequency``,
 ``weights_axis2: pol``,
 ``cal_solutions.pol_order: [B, A]``  ⇒ pol index 0 = B, 1 = A).

48 cal coarse channels = ``NCHAN_PER_CHGROUP // 8`` because the yaml
solutions were derived from time/frequency-integrated visibilities at
8× downsampling. Each coarse channel applies to 8 adjacent fine
voltage channels (user note 2026-05-05).

This module is **read-only**: it parses the blob, NEVER writes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from dsart.common.constants import NANTS, NCHAN_PER_CHGROUP, NPOL


# --- format constants (legacy bfCorr binary layout) ----------------------

#: Coarse cal channels per chgroup. Pinned by bfCorr (`NCHAN_PER_PACKET/8 = 48`)
#: and by the cal yaml `weight_files` axis1 length.
NCAL_COARSE_PER_CHGROUP: Final[int] = NCHAN_PER_CHGROUP // 8  # = 48

#: Fine voltage channels per coarse cal channel. Pinned by user note
#: 2026-05-05 ("each value applies to 8 adjacent channels").
N_FINE_PER_CAL_COARSE: Final[int] = NCHAN_PER_CHGROUP // NCAL_COARSE_PER_CHGROUP  # = 8

#: Number of float32 antpos values: 2 axes (E, N) × NANTS antennas.
_ANTPOS_FLOATS: Final[int] = 2 * NANTS  # = 192

#: Number of float32 cal-weight values: NANTS × NCAL_COARSE × NPOL × 2 (R/I).
_CAL_FLOATS: Final[int] = NANTS * NCAL_COARSE_PER_CHGROUP * NPOL * 2  # = 18432

#: Total file size in bytes (= 74,496 for DSA-110: 96 ants, 48 coarse, 2 pol).
BF_WEIGHTS_FILE_SIZE: Final[int] = 4 * (_ANTPOS_FLOATS + _CAL_FLOATS)  # = 74496


# --- dataclass result ----------------------------------------------------


@dataclass(frozen=True)
class BfWeights:
    """Parsed contents of one `beamformer_weights_*.dat` blob.

    Attributes
    ----------
    antpos_e, antpos_n : np.ndarray (NANTS,) float32
        Antenna positions in metres (ITRF east, north). Used by bfCorr's
        `populate_weights_matrix` for beam-steering geometry; not
        consumed by the slow correlator (we use them only for sanity
        logging).
    gains : np.ndarray (NANTS, NCAL_COARSE_PER_CHGROUP, NPOL) complex64
        Per-(ant, coarse_ch, pol) complex gain. Pol axis order matches
        the cal yaml's `pol_order` (default DSA-110 convention is
        ``[B, A]`` ⇒ pol_index 0 = B, 1 = A).
    source_path : pathlib.Path
        Provenance: where the blob was loaded from.
    """

    antpos_e: np.ndarray
    antpos_n: np.ndarray
    gains: np.ndarray
    source_path: Path

    @property
    def n_flagged(self) -> int:
        """Count of (ant, coarse_ch, pol) cells that are exactly zero
        — bfCorr writes zeros for solutions flagged by CASA upstream."""
        return int(np.sum(self.gains == 0.0))

    @property
    def magnitude_summary(self) -> dict[str, float]:
        """Summary of |gain| over non-zero cells."""
        mag = np.abs(self.gains)
        nz = mag[mag > 0]
        if nz.size == 0:
            return {"n_nonzero": 0, "mag_p50": float("nan"),
                    "mag_p99": float("nan"), "mag_max": float("nan")}
        return {
            "n_nonzero": int(nz.size),
            "mag_p50": float(np.median(nz)),
            "mag_p99": float(np.percentile(nz, 99)),
            "mag_max": float(np.max(nz)),
        }


# --- loader --------------------------------------------------------------


def load_bf_weights(path: str | Path) -> BfWeights:
    """Load a `beamformer_weights_*.dat` blob (74,496 bytes, fp32 LE).

    Parameters
    ----------
    path : str or Path
        Path to the .dat file.

    Returns
    -------
    BfWeights
        Parsed contents.

    Raises
    ------
    FileNotFoundError
        If `path` doesn't exist.
    ValueError
        If file size != BF_WEIGHTS_FILE_SIZE.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"cal blob {p} not found")
    sz = p.stat().st_size
    if sz != BF_WEIGHTS_FILE_SIZE:
        raise ValueError(
            f"cal blob {p} size {sz} != expected {BF_WEIGHTS_FILE_SIZE} "
            f"(= 4*(2*{NANTS} + {NANTS}*{NCAL_COARSE_PER_CHGROUP}*{NPOL}*2))"
        )

    raw = np.fromfile(str(p), dtype=np.float32, count=-1)
    if raw.size != _ANTPOS_FLOATS + _CAL_FLOATS:
        raise ValueError(
            f"cal blob {p}: read {raw.size} floats, expected "
            f"{_ANTPOS_FLOATS + _CAL_FLOATS}"
        )

    antpos_e = raw[0:NANTS].copy()                          # (96,)
    antpos_n = raw[NANTS:_ANTPOS_FLOATS].copy()             # (96,)

    cal_flat = raw[_ANTPOS_FLOATS:]                          # (18432,)
    cal_5d = cal_flat.reshape(
        NANTS, NCAL_COARSE_PER_CHGROUP, NPOL, 2,             # (96, 48, 2, 2)
    )
    gains = (cal_5d[..., 0] + 1j * cal_5d[..., 1]).astype(np.complex64)

    return BfWeights(
        antpos_e=antpos_e,
        antpos_n=antpos_n,
        gains=gains,
        source_path=p,
    )


# --- helpers for the apply-cal pipeline ---------------------------------


def upsample_coarse_to_fine(
    gains_coarse: np.ndarray,
    *,
    n_fine: int = NCHAN_PER_CHGROUP,
) -> np.ndarray:
    """Replicate each coarse cal channel across `n_fine // n_coarse` fine
    voltage channels (D17 user note: "each value applies to 8 adjacent
    channels").

    Parameters
    ----------
    gains_coarse : np.ndarray
        Shape `(NANTS, NCAL_COARSE_PER_CHGROUP, NPOL)` complex (any precision).
    n_fine : int
        Number of fine channels to expand to. Must be an exact multiple
        of `gains_coarse.shape[1]`.

    Returns
    -------
    np.ndarray
        Shape `(NANTS, n_fine, NPOL)`, same dtype as input.
    """
    n_coarse = gains_coarse.shape[1]
    if n_fine % n_coarse != 0:
        raise ValueError(f"n_fine={n_fine} not a multiple of n_coarse={n_coarse}")
    rep = n_fine // n_coarse
    return np.repeat(gains_coarse, rep, axis=1)


def normalize_phase_only(gains: np.ndarray) -> np.ndarray:
    """Divide each non-zero (ant, ch, pol) by its magnitude (matches
    bfCorr's `wnorm` step at `bfCorr.cu:1138-1142`). Zero entries pass
    through unchanged (so flagged solutions stay flagged).

    Returns a fresh array; input is not modified.
    """
    mag = np.abs(gains)
    out = gains.copy()
    nz = mag > 0
    out[nz] = out[nz] / mag[nz]
    return out


def maybe_swap_pol(gains: np.ndarray, swap: bool) -> np.ndarray:
    """If `swap`, flip the pol axis (axis -1) of `gains`. No-op otherwise.

    The cal yaml's default `pol_order` is `[B, A]`. If voltage data
    happens to be `[A, B]` (operator-overridable via
    ``corr_slow_compute --cal-pol-swap``), swapping aligns them.
    """
    if not swap:
        return gains
    return np.ascontiguousarray(gains[..., ::-1])
