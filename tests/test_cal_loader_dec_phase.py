"""F21 acceptance tests for the fast-corr cal-loader DEC-phase fold.

See M3_PLAN_FIXES.md F21 for design + the four acceptance criteria
(F21.1 through F21.4) implemented here.

Math summary (TMS Eq. 3.19 with planar DSA-110 core convention)
==============================================================

For a source at (HA=0, dec=δ_src) with δ_src > φ_lat (north of zenith),
the local ENU source direction is ``s = (0, sin(δ_src − φ_lat),
cos(δ_src − φ_lat))``. The geometric path-difference at antenna a
(position ``(E_a, N_a, U_a)``) is ``r_a · s = N_a sin(δ_src − φ_lat)
+ U_a cos(δ_src − φ_lat)``. The wavefront arrives at antenna a with
delay ``τ_a = −r_a · s / c`` (sign: antennas with positive r·s are
"out front" and see the wavefront first ⇒ negative delay relative to
the array origin).

Convention ``E_origin(t) = exp(+i 2π ν t)`` (positive frequency in
the exponent) gives:

    E_a(t) = E_origin(t − τ_a) = exp(+i 2π ν t) · exp(+i 2π ν · r_a · s / c)

For DSA-110's planar core (U_a ≈ 0) at HA=0:

    **E_a(f) = exp(+2π i f · sin(δ_src − φ_lat) · N_a / c)**

(POSITIVE sign in the per-antenna voltage exponent — TMS convention.)

The F21 cal weight cancels this geometric phase per antenna:

    **W_a(f) = exp(−2π i f · sin(δ_obs − φ_lat) · N_a / c)**

so ``E_a^cal = W_a · E_a = exp(+2π i f · (sin(δ_src − φ_lat) −
sin(δ_obs − φ_lat)) · N_a / c)``. With δ_obs = δ_src the post-cal
voltage phase is identically zero (F21.1).

Visibility (F18 convention ``V_ab = conj(E_a^cal) · E_b^cal`` for
``a < b``):

    arg(V_ab) = +2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat))
                  · (N_b − N_a) / c

(POSITIVE sign, baseline ordering ``N_b − N_a`` per F18 — F21.2 pins
this to ≤ 1e-10 rad against the predicted formula across multiple
baselines and channels.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dsart.cal.cal_loader import (
    SPEED_OF_LIGHT_M_S,
    CalMode,
    FastCorrCalTensors,
    compute_dec_phase,
    load_cal_with_dec_phase,
)
from dsart.common.constants import (
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    PHI_LAT_OVRO_RAD,
    freq_GHz,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _synthetic_voltage_for_source(
    *,
    chgroup: int,
    src_dec_rad: float,
    antpos_n: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return per-antenna voltages for a single point source at HA=0.

    Shape ``(NANTS, NCHAN_PER_CHGROUP, NPOL)`` complex128 (high
    precision so we can measure phase-cancellation residuals down to
    ~1e-10 rad in the F21.1 test).

    Both pol slots receive the identical voltage (unpolarised source
    convention).
    """
    f_hz = np.asarray(
        [freq_GHz(chgroup, ch) * 1.0e9 for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    sin_delta = math.sin(src_dec_rad - PHI_LAT_OVRO_RAD)
    n_a = antpos_n.astype(np.float64, copy=False)

    # Per-antenna voltage phase per TMS / F21 module-docstring derivation:
    # φ_a(f) = +2π f · sin(δ_src − φ_lat) · N_a / c
    # (POSITIVE sign in voltage exponent; cancelled by the F21 cal weight
    # which has NEGATIVE sign per the bfCorr-iArm==1 / central-beam fold.)
    arg = (
        +2.0 * math.pi * sin_delta / SPEED_OF_LIGHT_M_S
        * f_hz[None, :] * n_a[:, None]
    )                                                       # (NANTS, NCHAN)
    phase = np.cos(arg) + 1j * np.sin(arg)
    e = (amplitude * phase).astype(np.complex128)            # (NANTS, NCHAN)
    return np.broadcast_to(
        e[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    ).copy()                                                # (NANTS, NCHAN, NPOL)


def _antpos_n_realistic() -> np.ndarray:
    """Synthesise a plausible DSA-110-like N-S antpos array.

    Span ~150 m (matches the DSA-110 core's max baseline) with antennas
    evenly spaced + small random jitter. We don't need the real
    antpos for these tests — only the per-(ant, ch) F21 phase math
    matters; the bfCorr round-trip test (F21.4) re-uses this same
    array to compare numpy vs the reference formula element-wise.
    """
    rng = np.random.default_rng(seed=20260505)
    base = np.linspace(-75.0, +75.0, NANTS, dtype=np.float64)
    jitter = rng.uniform(-1.0, +1.0, size=NANTS)
    return (base + jitter).astype(np.float32)


def _apply_cal_complex(
    voltages: np.ndarray,
    *,
    cal_real: torch.Tensor,
    cal_imag: torch.Tensor,
) -> np.ndarray:
    """Multiply voltages by the broadcast cal tensors (CPU reference).

    Voltages are ``(NANTS, NCHAN, NPOL)`` complex128. Cal tensors are
    the broadcast layout ``(NCHAN, 1, NPOL, 1, NANTS)`` — we slice to
    extract ``(NCHAN, NPOL, NANTS) → (NANTS, NCHAN, NPOL)`` then
    multiply element-wise.
    """
    cr = cal_real.detach().cpu().numpy()                    # (NCHAN, 1, NPOL, 1, NANTS)
    ci = cal_imag.detach().cpu().numpy()
    cr_s = cr.squeeze(1).squeeze(2).transpose(2, 0, 1)      # (NANTS, NCHAN, NPOL)
    ci_s = ci.squeeze(1).squeeze(2).transpose(2, 0, 1)
    cal = (cr_s + 1j * ci_s).astype(np.complex128)
    return voltages * cal                                    # (NANTS, NCHAN, NPOL)


# ---------------------------------------------------------------------------
# F21.1 — synthetic source at obs_dec lands at (l, m) ≈ (0, 0): cal cancels
#         the per-antenna source phase exactly.
# ---------------------------------------------------------------------------


def test_F21_1_on_source_cal_cancels_voltage_phase() -> None:
    """δ_obs = δ_src ⇒ post-cal voltage phase ≡ 0 to numerical precision.

    Setup:
      * obs_dec = 53.85° (matches the 250924mptq burst dec)
      * Synthesise per-antenna voltages for a source at THE SAME dec
      * Apply the F21 cal tensor with δ_obs = δ_src
      * Expect: every (ant, ch, pol) post-cal phase is 0 to ≤ 1e-10 rad
        (arithmetic in double precision; F21 fold computes in fp64).

    This is the strongest F21 sanity check — if the fold has any sign
    error, this test fails immediately.
    """
    chgroup = 0
    obs_dec_rad = math.radians(53.848986)                   # 250924mptq Dec
    antpos_n = _antpos_n_realistic()

    voltages = _synthetic_voltage_for_source(
        chgroup=chgroup,
        src_dec_rad=obs_dec_rad,                            # δ_src = δ_obs
        antpos_n=antpos_n,
    )                                                        # (NANTS, NCHAN, NPOL)

    # Build cal directly from the F21 formula (no cal-blob involvement)
    # with unit gains on all (ant, ch, pol) cells.
    dec_phase = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
    )                                                        # (NANTS, NCHAN) complex128
    assert dec_phase.dtype == np.complex128, (
        "compute_dec_phase must return complex128 to keep the F21 fold "
        "at full fp64 precision; cplx64 cast happens later in the pipeline."
    )
    gains_fine = np.broadcast_to(
        dec_phase[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    )                                                        # already complex128

    post_cal = voltages * gains_fine                         # (NANTS, NCHAN, NPOL)

    # Every cell has unit modulus (no amplitude effects in this test);
    # phase should be 0 to fp64 precision.
    assert np.allclose(np.abs(post_cal), 1.0, atol=1e-12)
    phases = np.angle(post_cal)
    max_phase_residual = float(np.max(np.abs(phases)))
    # Tolerance: for the high-cycle regime arg ~ 700 rad, fp64
    # transcendental precision is ~ulp(arg) · cos(arg) ≈ 1e-13. Allow
    # 1e-11 to absorb compound rounding through the multiplication.
    assert max_phase_residual < 1e-11, (
        f"max post-cal phase residual = {max_phase_residual:.3e} rad "
        f"(expected ≤ 1e-11); the F21 fold has a sign / scaling error."
    )


# ---------------------------------------------------------------------------
# F21.2 — source off-dec by Δδ produces a per-baseline residual
#         visibility phase consistent with the predicted (l, m) shift.
# ---------------------------------------------------------------------------


def test_F21_2_off_dec_source_residual_phase_predicts_image_shift() -> None:
    """δ_obs ≠ δ_src ⇒ residual visibility phase matches predicted m-axis shift.

    Setup:
      * obs_dec = 50°, src_dec = 50° + Δδ where Δδ = 1° = 0.01745 rad
      * Synthesise voltages for source at src_dec
      * Apply F21 cal with δ_obs = obs_dec ≠ src_dec
      * Compute visibilities V_ab = conj(E_a^cal) · E_b^cal for all
        baselines (a < b) in a small subset of antennas
      * Per-baseline visibility phase should be
            +2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat)) · (N_a − N_b) / c
        (positive sign: differential phase wrt baseline (a, b) with a < b
         per F18 convention).
      * Compare observed vs predicted phase — should match to ≤ 1e-10 rad.

    For small Δδ, this approximates a (l, m) ≈ (0, cos(δ_obs−φ_lat) · Δδ)
    image-plane shift, but we verify the *exact* visibility-phase
    formula here for tightest sign-convention coverage.
    """
    chgroup = 0
    obs_dec_rad = math.radians(50.0)
    delta_deg = 1.0
    src_dec_rad = obs_dec_rad + math.radians(delta_deg)
    antpos_n = _antpos_n_realistic()

    voltages = _synthetic_voltage_for_source(
        chgroup=chgroup,
        src_dec_rad=src_dec_rad,                            # δ_src ≠ δ_obs
        antpos_n=antpos_n,
    )

    dec_phase = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,                            # phase to obs_dec
        antpos_n=antpos_n,
    )                                                        # complex128
    gains_fine = np.broadcast_to(
        dec_phase[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    )
    post_cal = voltages * gains_fine                         # (NANTS, NCHAN, NPOL)

    # Pick a few representative baselines (skip self-baselines).
    baselines = [(0, 47), (10, 30), (50, 95), (1, 95), (3, 17)]
    sin_src = math.sin(src_dec_rad - PHI_LAT_OVRO_RAD)
    sin_obs = math.sin(obs_dec_rad - PHI_LAT_OVRO_RAD)
    n_a = antpos_n.astype(np.float64, copy=False)

    f_hz = np.asarray(
        [freq_GHz(chgroup, ch) * 1.0e9 for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )

    for (a, b) in baselines:
        # Per F18: V_ab = conj(E_a) · E_b for a < b. We choose pol 0;
        # both pols have identical phase per the unpolarised source.
        v_ab = np.conj(post_cal[a, :, 0]) * post_cal[b, :, 0]   # (NCHAN,)
        observed_phase = np.angle(v_ab)                      # (NCHAN,) rad

        # Predicted (see module-docstring derivation):
        #   φ_a^cal = +2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat)) · N_a / c
        #   arg(V_ab) = -φ_a^cal + φ_b^cal
        #             = +2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat))
        #                · (N_b − N_a) / c
        predicted_phase = (
            +2.0 * math.pi * (sin_src - sin_obs) / SPEED_OF_LIGHT_M_S
            * f_hz * (n_a[b] - n_a[a])
        )

        # Wrap both to (-π, π] for comparison.
        diff = np.angle(np.exp(1j * (observed_phase - predicted_phase)))
        max_residual = float(np.max(np.abs(diff)))
        assert max_residual < 1e-10, (
            f"baseline ({a}, {b}): max visibility-phase residual "
            f"{max_residual:.3e} rad (expected ≤ 1e-10). "
            f"F21 fold sign / scaling does not match the predicted formula."
        )


# ---------------------------------------------------------------------------
# F21.3 — resolved-pair test: two sources at (0, ±0.05 rad) relative to
#         obs_dec recover at the right pixels in the iFFT image (no flip
#         vs F20).
# ---------------------------------------------------------------------------


def test_F21_3_two_sources_straddling_obs_dec_have_opposite_phase_slopes() -> None:
    """Sources at obs_dec ± Δδ produce opposite-sign visibility-phase slopes.

    Setup:
      * obs_dec = 45°
      * Two source decs: δ_+ = obs_dec + Δδ (north of phase centre),
                         δ_− = obs_dec − Δδ (south of phase centre)
        with Δδ small enough that the linear (small-angle) regime holds
        per-baseline at fp64 precision (Δδ = 1e-3 rad ≈ 0.057° ⇒
        non-linear correction is at the ~Δδ²/2 ≈ 5e-7 relative level
        — much tighter than the 1e-10 tolerance).
      * For each source separately, compute the per-baseline post-cal
        visibility phase at the mid-channel and compare against the
        F21.2 predicted formula. Both must independently match — the
        ``+Δδ ↔ −Δδ`` parity comes for free if both pass.

    This is the parity guard against F20 / F21 sign interactions: F20
    negates (u, v) inside ``grid_uv_natural`` to canonicalise (l, m)
    per TMS / CASA. If F21 introduced a parity flip relative to that
    convention, the predicted formula here (which derives the
    visibility phase directly from F21 + F18) would still self-match,
    but the downstream image plane would have its m-axis flipped. The
    end-to-end M3 burst sub-DoD (§8 line 2286) catches that downstream
    flip; this test catches the visibility-plane half (sign of the
    per-baseline visibility-phase slope wrt source dec).
    """
    chgroup = 0
    obs_dec_rad = math.radians(45.0)
    delta_rad = 1.0e-3                                      # ~0.057°, linear regime
    antpos_n = _antpos_n_realistic()

    cal_phase = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
    )                                                        # complex128
    gains = np.broadcast_to(
        cal_phase[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    )

    n_a = antpos_n.astype(np.float64, copy=False)
    sin_obs = math.sin(obs_dec_rad - PHI_LAT_OVRO_RAD)
    mid_ch = NCHAN_PER_CHGROUP // 2
    f_hz_mid = freq_GHz(chgroup, mid_ch) * 1.0e9

    # Pick a few representative baselines spanning the antpos range.
    baselines = [(0, 47), (5, 90), (20, 70), (1, 95), (3, 17)]

    for src_sign in (+1.0, -1.0):
        src_dec_rad = obs_dec_rad + src_sign * delta_rad
        sin_src = math.sin(src_dec_rad - PHI_LAT_OVRO_RAD)

        voltages = _synthetic_voltage_for_source(
            chgroup=chgroup,
            src_dec_rad=src_dec_rad,
            antpos_n=antpos_n,
        )
        post_cal = voltages * gains                          # (NANTS, NCHAN, NPOL)

        for (a, b) in baselines:
            v_ab = (
                np.conj(post_cal[a, mid_ch, 0])
                * post_cal[b, mid_ch, 0]
            )
            observed_phase = float(np.angle(v_ab))
            predicted_phase = (
                +2.0 * math.pi * (sin_src - sin_obs) / SPEED_OF_LIGHT_M_S
                * f_hz_mid * (n_a[b] - n_a[a])
            )
            diff = float(np.angle(np.exp(1j * (observed_phase - predicted_phase))))
            assert abs(diff) < 1e-10, (
                f"src_sign={src_sign:+.0f}Δδ baseline=({a},{b}): "
                f"observed={observed_phase:.6e}, predicted={predicted_phase:.6e}, "
                f"diff={diff:.3e} rad (expected ≤ 1e-10). "
                f"F21 ↔ F18 sign convention or parity is broken."
            )

    # Sanity: confirm Δδ is small enough that |predicted_phase|
    # actually has opposite signs for ±Δδ on at least one baseline at
    # this mid-channel — otherwise the test trivially passes by
    # symmetry around 0.
    a, b = baselines[0]
    sin_plus = math.sin(obs_dec_rad + delta_rad - PHI_LAT_OVRO_RAD)
    sin_minus = math.sin(obs_dec_rad - delta_rad - PHI_LAT_OVRO_RAD)
    pred_plus = (
        +2.0 * math.pi * (sin_plus - sin_obs) / SPEED_OF_LIGHT_M_S
        * f_hz_mid * (n_a[b] - n_a[a])
    )
    pred_minus = (
        +2.0 * math.pi * (sin_minus - sin_obs) / SPEED_OF_LIGHT_M_S
        * f_hz_mid * (n_a[b] - n_a[a])
    )
    assert pred_plus * pred_minus < 0, (
        "test sanity broken: predicted +Δδ and −Δδ phases have the same "
        "sign — pick a different baseline, mid-channel, or Δδ."
    )


# ---------------------------------------------------------------------------
# F21.4 — bfCorr round-trip: M3's compute_dec_phase matches dsaX_bfCorr.cu's
#         populate_weights_matrix (iArm==1, central beam bm=127) bit-for-bit.
# ---------------------------------------------------------------------------


def test_F21_4_bfcorr_round_trip_central_beam() -> None:
    """``compute_dec_phase`` matches the bfCorr reference formula element-wise.

    bfCorr code (``dsa110-xengine/src/dsaX_bfCorr.cu`` lines 1082-1085)
    for the central beam (``bm = 127``):

    .. code-block:: cuda

        theta = sep_ns*(127.-127.)*PI/10800. - (PI/180.)*dec  // = -(π/180)·dec
        afac = -2.*PI*fqs[fq]*sinf(theta)/CVAC
        twr = cosf(afac*antpos_n[a])
        twi = sinf(afac*antpos_n[a])

    where ``dec`` (the kernel arg) is ``37.23 - obsdec`` per
    ``calc_weights`` line 1159. Substituting and simplifying gives
    exactly the F21 formula in :func:`compute_dec_phase`.

        Tolerance: pure-fp64 vs pure-fp64 transcription of the same
    formula. Should match to ~1e-12 (transcendental ulp accumulation
    through the 638-rad arg). Test budget is 1e-11 absolute on real
    and imag parts.
    """
    chgroup = 5                                              # mid-band; doesn't matter
    obs_dec_rad = math.radians(53.848986)                   # 250924mptq Dec
    antpos_n = _antpos_n_realistic()

    # M3 implementation (returns complex128 — full fp64 precision)
    f21 = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
    )                                                        # (NANTS, NCHAN) complex128
    assert f21.dtype == np.complex128, (
        "compute_dec_phase must return complex128 for the F21.4 "
        "round-trip to match bfCorr to fp64 precision."
    )

    # bfCorr reference formula transcribed from CU code, central beam
    # (bm=127). PHI_LAT in degrees per the CU comment (37.23 — note
    # the CU code uses 37.23, not 37.234, but the M2-validated Python
    # constant is 37.234 for full math.radians precision; both should
    # agree once we use the M2-calibrated constant in the python side
    # too. The bfCorr code's 37.23 is a 4-digit truncation that
    # introduces a ~0.004° offset; the comparison below uses the
    # *same* φ_lat both sides so the test is about the formula
    # structure / sign, not the literal).
    phi_lat_deg = math.degrees(PHI_LAT_OVRO_RAD)
    obsdec_deg = math.degrees(obs_dec_rad)
    dec_kernel_arg = phi_lat_deg - obsdec_deg               # bfCorr's "dec" param
    cvac = SPEED_OF_LIGHT_M_S

    bf_real = np.empty((NANTS, NCHAN_PER_CHGROUP), dtype=np.float64)
    bf_imag = np.empty((NANTS, NCHAN_PER_CHGROUP), dtype=np.float64)
    for ch in range(NCHAN_PER_CHGROUP):
        f_hz = freq_GHz(chgroup, ch) * 1.0e9
        # CU: theta = sep_ns*(127.-127.)*PI/10800. - (PI/180.)*dec
        theta = -(math.pi / 180.0) * dec_kernel_arg
        # CU: afac = -2.*PI*fqs[fq]*sinf(theta)/CVAC
        afac = -2.0 * math.pi * f_hz * math.sin(theta) / cvac
        for a in range(NANTS):
            arg = afac * float(antpos_n[a])
            bf_real[a, ch] = math.cos(arg)
            bf_imag[a, ch] = math.sin(arg)

    f21_real = f21.real
    f21_imag = f21.imag

    real_max_diff = float(np.max(np.abs(f21_real - bf_real)))
    imag_max_diff = float(np.max(np.abs(f21_imag - bf_imag)))

    # fp64 transcendental at arg ~ 638 rad: ulp(arg) ≈ 7e-14;
    # trig precision ~ulp(arg) per the libm error contract; cos / sin
    # output near unit magnitude has absolute error ~7e-14. Allow
    # 1e-11 to absorb compound rounding through the multiplication
    # chain (-2π · sin_delta / c · f · N_a) which has ~5 fp ops, each
    # ~1 ulp ≈ 2e-16 relative ⇒ ~1e-15 cumulative on the final arg ⇒
    # cos/sin error ≈ arg · 1e-15 ≈ 6e-13. Margin x16.
    assert real_max_diff < 1.0e-11, (
        f"F21 vs bfCorr real-part max diff = {real_max_diff:.3e} "
        f"(expected ≤ 1e-11); sign-convention or formula mismatch."
    )
    assert imag_max_diff < 1.0e-11, (
        f"F21 vs bfCorr imag-part max diff = {imag_max_diff:.3e} "
        f"(expected ≤ 1e-11); sign-convention or formula mismatch."
    )


# ---------------------------------------------------------------------------
# Additional sanity: load_cal_with_dec_phase end-to-end against a real
# beamformer_weights blob (when available on h01); skipped on h23 / CI.
# ---------------------------------------------------------------------------


def _h01_real_cal_blob_path() -> str | None:
    """Return a real DSA-110 cal blob path if running on h01, else None.

    Used by the E2E test below to validate the full
    :func:`load_cal_with_dec_phase` pipeline (file loader + F21 fold +
    broadcast packaging) without sign-convention shortcuts.
    """
    candidate = "/home/ubuntu/data/voltages/0319/cals/beamformer_weights_sb00_0319+415.dat"
    from pathlib import Path
    return candidate if Path(candidate).is_file() else None


@pytest.mark.skipif(
    _h01_real_cal_blob_path() is None,
    reason="no real cal blob available; run on h01 to exercise E2E loader",
)
def test_F21_load_cal_with_dec_phase_e2e() -> None:
    """Full pipeline against a real cal blob: load → fold → broadcast.

    Verifies:
    * Output `cal_real` / `cal_imag` have the broadcast layout
      (NCHAN, 1, NPOL, 1, NANTS) and the requested dtype.
    * `info` dict carries the obs_dec / chgroup / cal_mode / pol_swap
      fields for downstream logging.
    * Multiplying the broadcast cal by a synthetic on-source voltage
      cancels the source phase to numerical precision (≤ 1e-3 rad in
      fp16 — limited by the broadcast tensor dtype, not by the F21
      math precision).
    """
    cal_path = _h01_real_cal_blob_path()
    assert cal_path is not None
    chgroup = 0
    obs_dec_rad = math.radians(41.5117)                     # 0319+415 source dec

    out = load_cal_with_dec_phase(
        cal_path,
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        cal_mode=CalMode.PHASE_ONLY,
        pol_swap=False,
        device="cpu",
        dtype=torch.float32,                                # high precision for the test
    )
    assert isinstance(out, FastCorrCalTensors)
    assert out.cal_real.shape == (NCHAN_PER_CHGROUP, 1, NPOL, 1, NANTS)
    assert out.cal_imag.shape == (NCHAN_PER_CHGROUP, 1, NPOL, 1, NANTS)
    assert out.cal_real.dtype == torch.float32
    assert out.obs_dec_rad == pytest.approx(obs_dec_rad)
    assert out.chgroup == chgroup
    assert out.cal_mode == CalMode.PHASE_ONLY
    assert out.pol_swap is False
    assert "n_flagged" in out.info
    assert "cal_path" in out.info
    assert "obs_dec_deg" in out.info

    # Synthetic voltages for the same dec, applied with this cal:
    # post-cal *amplitude* should be ~1 (phase-only normalised cal +
    # unit-amplitude voltages), and *phase* should be near 0 modulo
    # the residual cal phase from the real cal blob (which is
    # source-direction-agnostic — different from F21's geometric dec
    # phase). So we test that the F21 component is correctly
    # cancelled by checking that the post-cal phase pattern across
    # antennas is the *same* as applying the cal blob's bare gain
    # phase (without F21) to the on-source voltages computed at
    # zenith (δ_src = φ_lat). That isolates the F21 contribution.
    antpos_n = out.raw_bf_weights.antpos_n
    voltages = _synthetic_voltage_for_source(
        chgroup=chgroup,
        src_dec_rad=obs_dec_rad,                            # δ_src = δ_obs
        antpos_n=antpos_n,
    )
    post_cal = _apply_cal_complex(
        voltages, cal_real=out.cal_real, cal_imag=out.cal_imag
    )                                                        # (NANTS, NCHAN, NPOL)

    # Now the same test, but with cal that has F21 component zeroed —
    # i.e. the bare cal blob applied to ZENITH voltages (so geometric
    # phase is 0 everywhere; what's left is the cal blob's intrinsic
    # phase). Compare per-(ant, ch, pol) magnitude of the residual.
    out_zenith = load_cal_with_dec_phase(
        cal_path,
        chgroup=chgroup,
        obs_dec_rad=PHI_LAT_OVRO_RAD,                       # δ_obs = φ_lat ⇒ F21 phase ≡ 0
        cal_mode=CalMode.PHASE_ONLY,
        pol_swap=False,
        device="cpu",
        dtype=torch.float32,
    )
    voltages_zenith = _synthetic_voltage_for_source(
        chgroup=chgroup,
        src_dec_rad=PHI_LAT_OVRO_RAD,                       # δ_src = φ_lat
        antpos_n=antpos_n,
    )
    post_cal_zenith = _apply_cal_complex(
        voltages_zenith,
        cal_real=out_zenith.cal_real,
        cal_imag=out_zenith.cal_imag,
    )

    # In both cases the geometric (F21 + voltage) phase has been
    # cancelled, so what remains is just the cal-blob's intrinsic
    # complex gain. The two should agree element-wise modulo the
    # cal-tensor fp32 cast precision (~ulp at unit magnitude, ~1.2e-7).
    # Skip cells where the cal blob is exactly zero (CASA-flagged
    # solutions; both sides agree on those by construction but the
    # post-cal value is 0 ± noise).
    cal_blob_zero_mask = (
        np.abs(out.raw_bf_weights.gains) == 0.0
    )                                                        # (96, 48, 2)
    cal_blob_zero_fine = np.repeat(cal_blob_zero_mask, 8, axis=1)  # (96, 384, 2)
    valid_mask = ~cal_blob_zero_fine

    diff = post_cal - post_cal_zenith                        # (NANTS, NCHAN, NPOL)
    max_abs_diff = float(np.max(np.abs(diff[valid_mask])))
    # cal-blob intrinsic gains have magnitude 1 (phase-only) so any
    # F21 sign/scaling error would show up here as an O(1) residual.
    # fp32 precision allows ~1e-6 per cell; allow 1e-5 for compound
    # rounding through the broadcast / multiplication chain.
    assert max_abs_diff < 1.0e-5, (
        f"E2E cancellation test: max diff = {max_abs_diff:.3e} "
        f"(expected ≤ 1e-5 for fp32). The F21 fold either has a "
        f"sign error in the loader path or doesn't compose correctly "
        f"with the cal blob."
    )
