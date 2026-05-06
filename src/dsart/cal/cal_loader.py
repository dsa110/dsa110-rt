"""Fast-corr cal-tensor loader with F21 DEC-only phase fold.

Loads a legacy ``beamformer_weights_*.dat`` cal blob (via M2's
:mod:`dsart.cal.bf_weights`), upsamples coarse → fine, optionally
normalises to phase-only, optionally swaps the pol axis, **folds in
the F21 per-(ant, ch) DEC-only fringe-stop phase**, and returns the
broadcast-ready ``(cal_real, cal_imag)`` torch tensors that
:func:`dsart.services.slow_corr_kernel.apply_cal_split` consumes.

This module is **fast-corr-only** (M3). The slow-corr (M2) keeps its
own simpler cal pipeline in
:mod:`dsart.services.corr_slow_compute._build_cal_tensors` because
``meridian_fringestop`` does the fringe-stop downstream in casa38.

Why F21 (DEC-only phase fold) is required for the fast-corr path
================================================================

The DSA-110 SNAPs digitise voltages without any analog phase rotation,
so the recorded ``E_a(f, t)`` voltages are referenced to each
antenna's actual location at HA=0, regardless of the source's
declination relative to the array's mechanical pointing.

For an interferometer at HA=0 with antennas at ENU positions
``(E_a, N_a, U_a)`` observing a source at declination ``δ_src``, the
geometric voltage phase is

    φ_a(f) = −2π f · sin(δ_src − φ_lat) · N_a / c

(only the N component matters at HA=0; E and U cancel out at the
meridian for DSA-110's planar core). The visibility ``V_ab = E_a* · E_b``
inherits the differential phase ``φ_b − φ_a``.

To put the source at the iFFT image origin ``(l, m) ≈ (0, 0)``, we
need to multiply each voltage by ``exp(+i φ_a)``, equivalently
multiply the cal weight by ``exp(−i φ_a)``:

    cal_with_dec[a, ch, pol] = cal[a, ch, pol]
                  · exp(−2π i · f[ch] · sin(δ_obs − φ_lat) · N_a / c)

where ``δ_obs`` is the operator-supplied observing dec (set via etcd
``cmd: prepare`` in production, or CLI in benches).

Without this fold, a source at e.g. δ_src = 53.85° (the 250924mptq
burst dec) with the array at φ_lat = 37.234° lives at
``m ≈ sin(16.6°) ≈ 0.286`` rad in the iFFT image — outside the FoV
``1 / duv ≈ 0.17`` rad at ``N_grid = 256`` and the DSA-110 core's
~150 m max baseline. The slow-corr doesn't need this because
``meridian_fringestop`` does it downstream (per-baseline, in casa38);
the fast-corr has no such downstream step.

Sign-convention pin to the legacy beamformer (bfCorr)
=====================================================

The formula above matches the central-beam (``bm = 127``) version of
``populate_weights_matrix`` in
``dsa110-xengine/src/dsaX_bfCorr.cu`` lines 1082-1085 verbatim:

.. code-block:: cuda

    // iArm == 1 (N-S antennas; the iArm == 0 / E-W branch carries the
    // beam-cross-track offset which is not relevant for our central beam)
    theta = sep_ns*(127.-bm*1.)*PI/10800. - (PI/180.)*dec;  // for bm=127, theta = -(π/180)·dec
    afac = -2.*PI*fqs[fq]*sinf(theta)/CVAC;
    twr = cosf(afac*antpos_n[a+48*iArm]);
    twi = sinf(afac*antpos_n[a+48*iArm]);

with ``dec = (PHI_LAT_OVRO_DEG - obsdec)`` (see ``dsaX_bfCorr.cu`` line
1159: ``populate_weights_matrix<<<...>>>(..., 37.23-(d->obsdec))``).
Substituting:

* ``theta = -(π/180) · (PHI_LAT - obs_dec_deg)`` rad
* ``= (π/180) · (obs_dec_deg - PHI_LAT)`` rad
* ``= obs_dec_rad − PHI_LAT_OVRO_RAD``
* so ``afac · antpos_n[a] = -2π · f · sin(obs_dec_rad − PHI_LAT_OVRO_RAD) · antpos_n[a] / c``.

Per-antenna phase is exactly what F21 specifies. The combined
weight ``twr + i·twi`` is then multiplied with the cal as
``war = twr·cal_R − twi·cal_I``, ``wai = twi·cal_R + twr·cal_I``
(bfCorr.cu lines 1086-1093), which is the standard complex multiply
``(twr + i·twi) · (cal_R + i·cal_I)``. Our implementation does the
same multiply in numpy/torch.

Test :func:`tests.test_cal_loader_dec_phase.test_F21_4_bfcorr_round_trip`
pins this sign convention bit-for-bit against bfCorr's eq.

References
==========

* Plan §4.2 — fast-corr cal subsection.
* :mod:`dsart.cal.bf_weights` — M2 cal-blob loader (Class C, M3-owned).
* :func:`dsart.services.slow_corr_kernel.apply_cal_split` — consumer
  of the broadcast tensors returned here (D17 / F17).
* :func:`dsart.services.slow_corr_kernel.make_cal_broadcast_tensors`
  — converts ``(NANTS, NCHAN_PER_CHGROUP, NPOL)`` complex64 to the
  broadcast layout this module returns.
* ``M3_PLAN_FIXES.md`` F21 — full design + acceptance-test list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from dsart.cal.bf_weights import (
    BfWeights,
    load_bf_weights,
    maybe_swap_pol,
    normalize_phase_only,
    upsample_coarse_to_fine,
)
from dsart.common.constants import (
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    PHI_LAT_OVRO_RAD,
    freq_GHz,
)
from dsart.services.slow_corr_kernel import make_cal_broadcast_tensors


__all__ = [
    "SPEED_OF_LIGHT_M_S",
    "CalMode",
    "FastCorrCalTensors",
    "compute_dec_phase",
    "load_cal_with_dec_phase",
]


#: Vacuum speed of light in metres per second (CODATA 2018; matches
#: the value bfCorr's ``CVAC`` macro uses).
SPEED_OF_LIGHT_M_S: Final[float] = 299_792_458.0


# Cal-mode tokens. Mirror M2's ``--cal-mode {phase, full}`` (D17 / F17),
# kept as string constants so the etcd ``cmd: prepare`` payload + the
# bench CLI can use the same vocabulary. The slow-corr (M2) accepts
# these literally in its CLI; M3's fast-corr cal-loader does the same.
class CalMode:
    """String constants for ``cal_mode`` parameter."""

    PHASE_ONLY: Final[str] = "phase_only"
    """Divide each non-zero (ant, ch, pol) cal cell by its magnitude.

    Matches bfCorr's ``wnorm`` step at ``dsaX_bfCorr.cu:1138-1142``.
    Used in production fast-corr because amplitude calibration is
    folded into the bandpass elsewhere; phase-only avoids accidental
    amplitude drift between cal updates.
    """
    FULL: Final[str] = "full"
    """Apply the full complex cal (gain + phase + bandpass).

    Matches the slow-corr's ``--cal-mode full`` (M2 D17). Forces
    fp32 in the slow path due to potential overflow; for the fast-corr
    keep fp16 unless tests show overflow (which would be a new D-item).
    """

    _VALID: Final[tuple[str, ...]] = (PHASE_ONLY, FULL)


@dataclass(frozen=True)
class FastCorrCalTensors:
    """Result of :func:`load_cal_with_dec_phase`.

    Attributes
    ----------
    cal_real, cal_imag : torch.Tensor
        Broadcast-ready (NCHAN_PER_CHGROUP, 1, NPOL, 1, NANTS) float
        tensors. Drop-in for
        :func:`dsart.services.slow_corr_kernel.apply_cal_split`'s
        ``cal_real_fine`` / ``cal_imag_fine`` arguments.
    raw_bf_weights : :class:`dsart.cal.bf_weights.BfWeights`
        The raw cal blob loaded from disk (antpos + gains +
        provenance). Caller may use it for sanity logging or to
        cross-check antpos.
    obs_dec_rad : float
        The observing declination this cal tensor was phased to (rad).
    chgroup : int
        The corr-node chgroup index (0..15) — pinned because the
        per-channel frequencies depend on it.
    cal_mode : str
        ``"phase_only"`` or ``"full"``.
    pol_swap : bool
        Whether the pol axis was swapped at load time.
    info : dict
        Logging metadata (n_flagged, mag_p50, mag_p99, etc.).
    """

    cal_real: torch.Tensor
    cal_imag: torch.Tensor
    raw_bf_weights: BfWeights
    obs_dec_rad: float
    chgroup: int
    cal_mode: str
    pol_swap: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Pure F21 math
# ---------------------------------------------------------------------------


def compute_dec_phase(
    *,
    chgroup: int,
    obs_dec_rad: float,
    antpos_n: np.ndarray,
) -> np.ndarray:
    """F21 DEC-only phase factor per (antenna, channel).

    Returns the bare DEC-fringe-stop phase, suitable for multiplying
    into a complex cal array of shape ``(NANTS, NCHAN_PER_CHGROUP, NPOL)``
    via broadcasting on the pol axis. Useful for tests that want to
    validate the F21 fold in isolation, without confounding it with
    cal-blob effects.

    Parameters
    ----------
    chgroup : int
        Corr-node chgroup index (0..15). Determines the per-channel
        frequencies via :func:`dsart.common.constants.freq_GHz`.
    obs_dec_rad : float
        Observing source declination (rad). This is the dec we want to
        bring to image-plane origin; cf. ``δ_obs`` in the module
        docstring's formula.
    antpos_n : np.ndarray
        Per-antenna N-S position in metres. Shape ``(NANTS,)`` float
        (any precision; will be cast to float64 for the math). Same
        sign convention as :class:`dsart.cal.bf_weights.BfWeights.antpos_n`
        (bfCorr's ``antpos_n``, ITRF north).

    Returns
    -------
    np.ndarray
        Shape ``(NANTS, NCHAN_PER_CHGROUP)`` complex64. Each element

        .. math::
            \\exp\\!\\left(-2\\pi i \\cdot f[\\mathrm{ch}] \\cdot
                \\sin(\\delta_{\\mathrm{obs}} - \\varphi_{\\mathrm{lat}})
                \\cdot N_a / c\\right)

        Same sign convention as bfCorr's ``twr + i·twi`` for the
        central beam (see module docstring).

    Notes
    -----
    Always evaluated at float64 precision internally; the returned
    array is downcast to complex64 (the same precision as
    :class:`dsart.cal.bf_weights.BfWeights.gains`) so it can be
    multiplied straight into the cal array without dtype-promotion
    surprises.
    """
    if not 0 <= chgroup < 16:
        raise ValueError(f"chgroup={chgroup}, expected 0..15")
    if antpos_n.shape != (NANTS,):
        raise ValueError(
            f"antpos_n shape {antpos_n.shape}, expected ({NANTS},)"
        )

    sin_delta = math.sin(obs_dec_rad - PHI_LAT_OVRO_RAD)

    f_hz = np.asarray(
        [freq_GHz(chgroup, ch) * 1.0e9 for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )                                                       # (NCHAN_PER_CHGROUP,)

    n_a = antpos_n.astype(np.float64, copy=False)            # (NANTS,)

    # arg[a, ch] = -2π · f[ch] · sin(δ_obs − φ_lat) · N_a / c
    arg = (
        -2.0 * math.pi * sin_delta / SPEED_OF_LIGHT_M_S
        * f_hz[None, :] * n_a[:, None]
    )                                                       # (NANTS, NCHAN_PER_CHGROUP)

    return np.exp(1j * arg).astype(np.complex64)


# ---------------------------------------------------------------------------
# Full pipeline: load cal + fold F21 → broadcast tensors
# ---------------------------------------------------------------------------


def load_cal_with_dec_phase(
    cal_path: str | Path,
    *,
    chgroup: int,
    obs_dec_rad: float,
    cal_mode: str = CalMode.PHASE_ONLY,
    pol_swap: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> FastCorrCalTensors:
    """Load a cal blob, fold in the F21 DEC-only phase, return broadcast tensors.

    This is the production fast-corr cal-load entry point. Mirrors M2's
    :func:`dsart.services.corr_slow_compute._build_cal_tensors` but
    with the F21 DEC-phase fold inserted between the upsample-to-fine
    step and the broadcast-tensor packaging.

    Parameters
    ----------
    cal_path : str or Path
        Path to a legacy ``beamformer_weights_*.dat`` blob. Read-only.
    chgroup : int
        Corr-node chgroup index (0..15). Pinned because the per-channel
        frequencies the F21 fold uses depend on it.
    obs_dec_rad : float
        Observing source declination (rad). Folded into the per-(ant, ch)
        cal phase per F21.
    cal_mode : str, optional
        :attr:`CalMode.PHASE_ONLY` (default) or :attr:`CalMode.FULL`.
        Phase-only divides each non-zero cal cell by its magnitude
        (matches bfCorr's ``wnorm``); full retains gain magnitudes.
    pol_swap : bool, optional
        If True, flip the cal pol axis (use when voltage data is
        ``[A, B]`` but cal yaml's ``pol_order`` is ``[B, A]``). Mirrors
        ``corr_slow_compute --cal-pol-swap`` (D17 / F17).
    device : torch.device or str, optional
        Torch device for the output tensors. Default ``"cpu"`` (test
        / CI usage); production passes a CUDA device.
    dtype : torch.dtype, optional
        Torch dtype for the output tensors. Default ``torch.float16``
        (matches the M2 fp16 GEMM path / D8). Use ``torch.float32`` for
        CPU debug or precision-sensitive tests.

    Returns
    -------
    FastCorrCalTensors
        Frozen dataclass with ``cal_real`` / ``cal_imag`` broadcast
        tensors + the raw :class:`BfWeights` + ``obs_dec_rad`` /
        ``chgroup`` / ``cal_mode`` / ``pol_swap`` for provenance, plus
        an ``info`` dict for logging.

    Raises
    ------
    FileNotFoundError, ValueError
        Propagated from :func:`dsart.cal.bf_weights.load_bf_weights`.
    ValueError
        If ``cal_mode`` is not in :attr:`CalMode._VALID`, or
        ``chgroup`` is out of range.
    """
    if cal_mode not in CalMode._VALID:
        raise ValueError(
            f"cal_mode={cal_mode!r}, expected one of {CalMode._VALID}"
        )
    if not 0 <= chgroup < 16:
        raise ValueError(f"chgroup={chgroup}, expected 0..15")

    bfw = load_bf_weights(cal_path)
    gains = bfw.gains                                       # (96, 48, 2) complex64
    if cal_mode == CalMode.PHASE_ONLY:
        gains = normalize_phase_only(gains)
    gains = maybe_swap_pol(gains, swap=pol_swap)
    gains_fine = upsample_coarse_to_fine(gains)             # (96, 384, 2)

    dec_phase = compute_dec_phase(                          # (96, 384) complex64
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=bfw.antpos_n,
    )
    # Broadcast over pol axis. Gain × DEC-phase is associative + commutative
    # (both are scalar complex multiplications per cell).
    gains_fine_with_dec = (
        gains_fine * dec_phase[:, :, None]
    ).astype(np.complex64)                                  # (96, 384, 2)

    cal_real, cal_imag = make_cal_broadcast_tensors(
        gains_fine_with_dec, device=device, dtype=dtype,
    )
    info = {
        "cal_path": str(bfw.source_path),
        "chgroup": chgroup,
        "obs_dec_rad": float(obs_dec_rad),
        "obs_dec_deg": math.degrees(obs_dec_rad),
        "cal_mode": cal_mode,
        "pol_swap": pol_swap,
        "n_flagged": bfw.n_flagged,
        **{f"cal_{k}": v for k, v in bfw.magnitude_summary.items()},
    }
    return FastCorrCalTensors(
        cal_real=cal_real,
        cal_imag=cal_imag,
        raw_bf_weights=bfw,
        obs_dec_rad=float(obs_dec_rad),
        chgroup=chgroup,
        cal_mode=cal_mode,
        pol_swap=pol_swap,
        info=info,
    )
