"""F21 acceptance tests for the fast-corr cal-loader DEC-phase fold.

See M3_PLAN_FIXES.md F21 for design + the four acceptance criteria
(F21.1 through F21.4) implemented here.

Math summary:
* Source at (HA=0, dec=δ_src) produces per-antenna voltage phase
  ``φ_a(f) = −2π f · sin(δ_src − φ_lat) · N_a / c``  (N-S baseline only;
  HA=0 so E-W and U components cancel for the planar DSA-110 core).
* Visibility per F18 is ``V_ab = E_a^* · E_b`` so the visibility phase
  is ``+2π f · sin(δ_src − φ_lat) · (N_a − N_b) / c``  (i.e. the
  *opposite sign* of the per-antenna voltage phase, *and* the
  per-baseline N is N_a−N_b not N_b−N_a; the latter follows from
  ``conj(E_a) · E_b = exp(+i φ_a) · exp(+i φ_b · 0)`` etc — see
  derivation in F21.1's docstring).
* The F21 cal weight ``W_a(f) = exp(−2π i f · sin(δ_obs − φ_lat) · N_a / c)``
  pre-multiplies E_a → ``E_a^cal = W_a · E_a``. Per-antenna phase becomes
  ``φ_a^cal = φ_a + arg(W_a) = −2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat))
  · N_a / c``. With δ_obs = δ_src this is identically zero (F21.1).
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

    # Per-antenna voltage phase: φ_a(f) = −2π f · sin(δ_src − φ_lat) · N_a / c
    arg = (
        -2.0 * math.pi * sin_delta / SPEED_OF_LIGHT_M_S
        * f_hz[None, :] * n_a[:, None]
    )                                                       # (NANTS, NCHAN)
    phase = np.exp(1j * arg)
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
    )                                                        # (NANTS, NCHAN) complex64
    gains_fine = np.broadcast_to(
        dec_phase[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    ).astype(np.complex128)

    post_cal = voltages * gains_fine                         # (NANTS, NCHAN, NPOL)

    # Every cell has unit modulus (no amplitude effects in this test);
    # phase should be 0 exactly to numerical precision in fp64.
    assert np.allclose(np.abs(post_cal), 1.0, atol=1e-10)
    phases = np.angle(post_cal)
    max_phase_residual = float(np.max(np.abs(phases)))
    assert max_phase_residual < 1e-10, (
        f"max post-cal phase residual = {max_phase_residual:.3e} rad "
        f"(expected ≤ 1e-10); the F21 fold has a sign / scaling error."
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
    )
    gains_fine = np.broadcast_to(
        dec_phase[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    ).astype(np.complex128)
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
        # Per F18: V_ab = conj(E_a) · E_b for a < b. We choose pol 0; both
        # pols have identical phase per the unpolarised source.
        v_ab = np.conj(post_cal[a, :, 0]) * post_cal[b, :, 0]   # (NCHAN,)
        observed_phase = np.angle(v_ab)                      # (NCHAN,) rad

        # Predicted: combining per-antenna phases φ_a^cal − φ_b^cal where
        #   φ_a^cal = -2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat)) · N_a / c
        # gives V_ab phase = +2π f · (sin(δ_src − φ_lat) − sin(δ_obs − φ_lat))
        #                       · (N_a − N_b) / c
        predicted_phase = (
            +2.0 * math.pi * (sin_src - sin_obs) / SPEED_OF_LIGHT_M_S
            * f_hz * (n_a[a] - n_a[b])
        )

        # Wrap both to (-π, π] for comparison.
        diff = np.angle(np.exp(1j * (observed_phase - predicted_phase)))
        max_residual = float(np.max(np.abs(diff)))
        assert max_residual < 1e-10, (
            f"baseline ({a}, {b}): max visibility-phase residual "
            f"{max_residual:.3e} rad (expected ≤ 1e-10). "
            f"F21 fold sign / scaling does not match the bfCorr eq."
        )


# ---------------------------------------------------------------------------
# F21.3 — resolved-pair test: two sources at (0, ±0.05 rad) relative to
#         obs_dec recover at the right pixels in the iFFT image (no flip
#         vs F20).
# ---------------------------------------------------------------------------


def test_F21_3_resolved_pair_no_flip_relative_to_F20() -> None:
    """Two sources straddling obs_dec recover at the predicted m-axis pixels
    (and same sign as the prediction — the F20 (u, v) negation in
    grid_uv_natural already canonicalises (l, m) per TMS / CASA, so any
    sign-flip introduced by F21 would manifest as the two sources
    swapping sides on the image m-axis).

    This is a *visibility-plane* test (no actual gridder/iFFT used —
    that's covered by tests/test_voltage_fixture_slow_corr_smoke.py
    end-to-end). Here we verify that for a source at +Δδ from obs_dec,
    the visibility phase increases linearly with N_b − N_a in the same
    sense as bfCorr predicts; and for a source at −Δδ it decreases with
    the same magnitude. If F21 introduced a parity flip, the two
    sources' phase-vs-N curves would NOT be mirror-symmetric.
    """
    chgroup = 0
    obs_dec_rad = math.radians(45.0)
    delta_rad = 0.05                                        # ~2.86°
    antpos_n = _antpos_n_realistic()

    cal_phase = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
    )
    gains = np.broadcast_to(
        cal_phase[:, :, None], (NANTS, NCHAN_PER_CHGROUP, NPOL)
    ).astype(np.complex128)

    n_a = antpos_n.astype(np.float64, copy=False)
    sin_obs = math.sin(obs_dec_rad - PHI_LAT_OVRO_RAD)
    f_hz_mid = freq_GHz(chgroup, NCHAN_PER_CHGROUP // 2) * 1.0e9   # use mid-channel for clarity

    sin_plus = math.sin(obs_dec_rad + delta_rad - PHI_LAT_OVRO_RAD)
    sin_minus = math.sin(obs_dec_rad - delta_rad - PHI_LAT_OVRO_RAD)

    # Voltages and post-cal phases for each source separately.
    plus_vals = []
    minus_vals = []
    for src_sign, sin_src, store in [
        (+1, sin_plus, plus_vals),
        (-1, sin_minus, minus_vals),
    ]:
        src_dec_rad = obs_dec_rad + src_sign * delta_rad
        v = _synthetic_voltage_for_source(
            chgroup=chgroup,
            src_dec_rad=src_dec_rad,
            antpos_n=antpos_n,
        )
        post_cal = v * gains
        # All baselines (a, b) with a < b — collect visibility phase at one freq.
        for a in range(0, NANTS, 19):
            for b in range(a + 1, NANTS, 19):
                v_ab = (
                    np.conj(post_cal[a, NCHAN_PER_CHGROUP // 2, 0])
                    * post_cal[b, NCHAN_PER_CHGROUP // 2, 0]
                )
                store.append((a, b, np.angle(v_ab)))

    # For each baseline that appears in both source's lists, check the
    # phases are equal in magnitude and OPPOSITE in sign (mirror about
    # zero). If F21 had a sign flip, we'd see them with the SAME sign
    # for the same Δδ, breaking this test.
    plus_dict = {(a, b): p for (a, b, p) in plus_vals}
    minus_dict = {(a, b): p for (a, b, p) in minus_vals}
    common = sorted(set(plus_dict) & set(minus_dict))
    assert len(common) >= 5, "test setup error: not enough baselines"
    max_asym = 0.0
    for (a, b) in common:
        # Sum of phases should be ~0 (mirror-symmetric).
        s = np.angle(np.exp(1j * (plus_dict[(a, b)] + minus_dict[(a, b)])))
        max_asym = max(max_asym, abs(s))
    assert max_asym < 1e-9, (
        f"max +Δδ vs −Δδ phase-sum residual = {max_asym:.3e} rad "
        f"(expected ≤ 1e-9). The F21 fold is NOT mirror-symmetric "
        f"about obs_dec; likely a parity bug relative to F20."
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

    Tolerance: this is a pure-numpy (fp64 → complex64) vs C-eq
    (transcribed in numpy) comparison; should match to fp32 precision.
    The test budget is 1e-6 absolute on real and imag parts.
    """
    chgroup = 5                                              # mid-band; doesn't matter
    obs_dec_rad = math.radians(53.848986)                   # 250924mptq Dec
    antpos_n = _antpos_n_realistic()

    # M3 implementation
    f21 = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
    )                                                        # (NANTS, NCHAN) complex64

    # bfCorr reference formula transcribed from CU code, central beam
    # (bm=127). PHI_LAT in degrees per the CU comment (37.23 — note the
    # CU code uses 37.23, not 37.234, but the M2-validated Python
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
            arg = afac * antpos_n[a]
            bf_real[a, ch] = math.cos(arg)
            bf_imag[a, ch] = math.sin(arg)

    f21_real = f21.real.astype(np.float64)
    f21_imag = f21.imag.astype(np.float64)

    real_max_diff = float(np.max(np.abs(f21_real - bf_real)))
    imag_max_diff = float(np.max(np.abs(f21_imag - bf_imag)))

    # complex64 ≈ 7-digit precision; allow 2e-7 to be safe for the
    # outermost N_a · f product magnitudes (max ~ 3e10 m·Hz ≈ 30 GHz·m
    # → ~100 cycles; precision of cos is ~ulp(1) = 1e-7).
    assert real_max_diff < 2.0e-7, (
        f"F21 vs bfCorr real-part max diff = {real_max_diff:.3e} "
        f"(expected ≤ 2e-7); sign-convention or formula mismatch."
    )
    assert imag_max_diff < 2.0e-7, (
        f"F21 vs bfCorr imag-part max diff = {imag_max_diff:.3e} "
        f"(expected ≤ 2e-7); sign-convention or formula mismatch."
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
    # complex gain. The two should agree element-wise (within fp32
    # precision).
    max_abs_diff = float(np.max(np.abs(post_cal - post_cal_zenith)))
    # cal-blob intrinsic gains have magnitude 1 (phase-only) so any
    # F21 sign/scaling error would show up here as an O(1) residual.
    assert max_abs_diff < 1.0e-4, (
        f"E2E cancellation test: max diff = {max_abs_diff:.3e} "
        f"(expected ≤ 1e-4 for fp32). The F21 fold either has a "
        f"sign error in the loader path or doesn't compose correctly "
        f"with the cal blob."
    )
