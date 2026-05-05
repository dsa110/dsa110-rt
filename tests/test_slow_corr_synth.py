"""tests/test_slow_corr_synth.py — M2 DoD acceptance tests for the slow correlator.

Plan §8 M2 DoD (lines 2167-2170): validates `slow_corr_kernel.compute_split`
on three deterministic synthetic inputs:

  1. **Thermal noise**: per-(ant, ch, pol) complex Gaussian at known RMS.
     Auto-correlations recover the per-(ant, ch, pol) sample variance to
     within fractional Wishart prediction error; cross-correlations
     integrate down toward zero (reference comparison in fp32).
  2. **Point source** at known `(l, m)` with known fluence: visibility
     tensor matches the analytical planar-wave model
       V_ij(ν) = |S|² · exp(-2πi (b_i − b_j) · ŝ · ν / c)
     under D5's `V_ij = conj(E_i) · E_j` convention (matches bfCorr).
  3. **Per-(ant, pol) bandpass**: shape applied to (1) is recovered in
     the auto-correlations (per-channel `|bp|²` profile preserved).

Per F1 revised in M2_PLAN_FIXES.md: NO calibration is applied in the
slow correlator — the bandpass test verifies GEMM-faithfulness to the
input voltage bandpass, not that the correlator applies a bandpass cal.

Layout: GPU is required for the full-size end-to-end tests because the
kernel is hardcoded to a full fada page (302 MB). Unit tests for
indexing, fluff math, and round-trip helpers run on CPU.
"""

from __future__ import annotations

import os

# Match other dsart tests: enable DSART_TEST=1 before any dsart import.
os.environ.setdefault("DSART_TEST", "1")

import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BADA_NPOL,
    BLOCK_SAMPLES_SPECNUM,
    LEGACY_FLUFF_SCALE,
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    HALF_FAC_DEFAULT,
    SlowCorrKernel,
    _FADA_VOLT_SHAPE,
    _GEMM_LAYOUT_SHAPE,
    pack_bada_block,
    unpack_int4_split,
    upper_tri_indices,
)

# ---------------------------------------------------------------------------
# Constants + deterministic seeds
# ---------------------------------------------------------------------------

# Per plan §10 line 1809-1810 these would normally live in `tests/seeds.py`
# (M0 deliverable, not yet authored). Inline here until that file exists.
THERMAL_NOISE_BASE = 0xD5A1107E
POINT_SOURCE_BASE = 0xD5A1107F
BANDPASS_BASE = 0xD5A1107A

NPACKETS = BLOCK_SAMPLES_SPECNUM                             # 2048
NTIMES_PER_PACKET = 2
N_TIME = NPACKETS * NTIMES_PER_PACKET                        # 4096

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

GPU_AVAILABLE = torch.cuda.is_available()
gpu_required = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="needs CUDA (kernel is full-size only)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack_bytes_int4(real_q: np.ndarray, imag_q: np.ndarray) -> np.ndarray:
    """Pack arrays of int4 nibbles (in [-8, 7]) into uint8 bytes.

    Layout: low nibble = real, high nibble = imag (matches fada / bfCorr).
    """
    assert real_q.shape == imag_q.shape
    real_u4 = (real_q.astype(np.int8) & 0x0F).astype(np.uint8)
    imag_u4 = (imag_q.astype(np.int8) & 0x0F).astype(np.uint8)
    return ((imag_u4 & 0x0F) << 4) | real_u4


def _quantize_to_int4(x: np.ndarray) -> np.ndarray:
    """Round-and-saturate to int4 range [-8, 7], returning int8 dtype."""
    return np.clip(np.round(x), -8, 7).astype(np.int8)


def _voltages_to_int4_bytes(volts_post_fluff: np.ndarray) -> np.ndarray:
    """Inverse of fluff: complex voltages (post-fluff units) → packed int4 bytes.

    Quantization saturates outside [-8, 7] (after dividing by `LEGACY_FLUFF_SCALE`).
    """
    if volts_post_fluff.shape != _FADA_VOLT_SHAPE:
        raise ValueError(
            f"volts shape {volts_post_fluff.shape} != fada {_FADA_VOLT_SHAPE}"
        )
    pre = volts_post_fluff / LEGACY_FLUFF_SCALE
    real_q = _quantize_to_int4(pre.real)
    imag_q = _quantize_to_int4(pre.imag)
    return _pack_bytes_int4(real_q, imag_q)


def _kernel_correlate(
    raw_bytes: np.ndarray, *,
    device: torch.device,
    out_dtype: torch.dtype = torch.float16,
    half_fac: int = HALF_FAC_DEFAULT,
) -> np.ndarray:
    """Run the production kernel end-to-end. Returns complex64 numpy of shape
    `(NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)`."""
    real_v, imag_v = unpack_int4_split(
        raw_bytes, device=device, out_dtype=out_dtype,
    )
    kernel = SlowCorrKernel(device=device, half_fac=half_fac)
    vis = kernel.compute_split(real_v, imag_v)
    out = vis.detach().cpu().numpy()
    # Free GPU buffers eagerly so consecutive tests don't pile up VRAM.
    del real_v, imag_v, vis, kernel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _autocorr_indices(nants: int = NANTS) -> np.ndarray:
    """Baseline indices of auto-correlations: bls_idx = a*(a+1)/2 + a."""
    return np.array([a * (a + 1) // 2 + a for a in range(nants)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Unit tests (run on CPU, no GPU needed)
# ---------------------------------------------------------------------------


def test_baseline_indexing_xgpu_upper_tri() -> None:
    """`upper_tri_indices()` matches xGPU upper-tri convention.

    Per D4 in M2_PLAN_FIXES.md and `dsaX_bfCorr.cu` lines 467-481:
        bls_idx = a*(a+1)/2 + b,  for b ≤ a (auto-corrs included on b == a).
    """
    a, b = upper_tri_indices(NANTS)
    assert a.shape == (NBASE,)
    assert b.shape == (NBASE,)
    assert (b <= a).all(), "b must be ≤ a (upper-triangle)"
    # Verify against the formula.
    for k in range(NBASE):
        bls_idx = int(a[k]) * (int(a[k]) + 1) // 2 + int(b[k])
        assert bls_idx == k, f"bls_idx mismatch at k={k}"
    # Auto-correlations on diagonal: bls_idx = a*(a+1)/2 + a.
    auto_idx = _autocorr_indices(NANTS)
    np.testing.assert_array_equal(a[auto_idx], np.arange(NANTS))
    np.testing.assert_array_equal(b[auto_idx], np.arange(NANTS))


def test_int8_asr_fluff_matches_legacy_c() -> None:
    """PyTorch int8 << / >> match C signed-char shift semantics on every byte 0..255.

    The fluff path in `unpack_int4_split` (D16 in M2_PLAN_FIXES.md) relies on
    PyTorch's `>>` on `torch.int8` being arithmetic shift right. Verify
    against a direct Python reference for every possible byte.
    """
    raw = torch.arange(256, dtype=torch.uint8)
    i8 = raw.view(torch.int8)
    real = ((i8 << 4) >> 4).tolist()
    imag = (i8 >> 4).tolist()
    expected_real, expected_imag = [], []
    for byte in range(256):
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        if lo >= 8:
            lo -= 16
        if hi >= 8:
            hi -= 16
        expected_real.append(lo)
        expected_imag.append(hi)
    assert real == expected_real, "int8 ((<<4)>>4) mismatch (real nibble)"
    assert imag == expected_imag, "int8 (>>4) mismatch (imag nibble)"


def test_pack_helper_roundtrip() -> None:
    """`_pack_bytes_int4` → fluff round-trip is exact for in-range integers."""
    rng = np.random.default_rng(0xCAFE)
    n = 1024
    real_int = rng.integers(-8, 8, size=n).astype(np.int8)
    imag_int = rng.integers(-8, 8, size=n).astype(np.int8)
    packed = _pack_bytes_int4(real_int, imag_int)

    # Fluff via the production code path (CPU, fp32 to avoid fp16 oddities).
    raw_t = torch.as_tensor(packed, device="cpu")
    raw_i8 = raw_t.view(torch.int8)
    real_recovered = ((raw_i8 << 4) >> 4).numpy().astype(np.int8)
    imag_recovered = (raw_i8 >> 4).numpy().astype(np.int8)
    np.testing.assert_array_equal(real_recovered, real_int)
    np.testing.assert_array_equal(imag_recovered, imag_int)


# ---------------------------------------------------------------------------
# GPU end-to-end tests (the M2 DoD)
# ---------------------------------------------------------------------------


@gpu_required
def test_kernel_sign_convention_v_ij_conj_ei_ej() -> None:
    """V_ij = conj(E_i) · E_j (D5; matches bfCorr `dsaX_bfCorr.cu:559-581`).

    Construct E_a = 1 + 0j and E_b = 0 + 1j everywhere (post-fluff units).
    Expected (for any baseline pair (a, b) where b < a):
        V[a, b] = sum_t conj(E_a) · E_b = K · (1 - 0j) · (0 + 1j) = K · j
        V_real == 0,  V_imag == K · scale²
    A wrong sign convention (V_ij = E_i · conj(E_j)) would give -K·j instead.
    """
    device = torch.device("cuda")
    real_int = np.zeros(_FADA_VOLT_SHAPE, dtype=np.int8)
    imag_int = np.zeros(_FADA_VOLT_SHAPE, dtype=np.int8)
    # ant = a (e.g. 5) has E = (1, 0); ant = b (e.g. 3) has E = (0, 1).
    # Use real units: int4 value 1 → fluffs to 0.05; same for imag.
    a_test, b_test = 5, 3
    real_int[:, a_test] = 1
    imag_int[:, b_test] = 1
    raw_bytes = _pack_bytes_int4(real_int, imag_int)

    # Use fp32 inputs to remove fp16 rounding from the comparison.
    vis = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float32)

    # Locate (a, b) baseline.
    a_idx, b_idx = upper_tri_indices(NANTS)
    # bls_idx = a*(a+1)/2 + b for b < a, with a=5, b=3 → 5*6/2 + 3 = 18.
    bls = a_test * (a_test + 1) // 2 + b_test
    assert int(a_idx[bls]) == a_test and int(b_idx[bls]) == b_test, (
        "baseline ordering changed; update test"
    )

    # Expected: V[bls] = sum_t conj((1, 0)) · (0, 1) · scale² = K · scale² · j
    expected_imag = N_TIME * (LEGACY_FLUFF_SCALE ** 2)
    # All channels and pols see the same E (no chan/pol structure in this test).
    np.testing.assert_allclose(
        vis[bls].real, 0.0, atol=1e-4,
        err_msg="V_real should be 0 for orthogonal-real-imag antennas",
    )
    np.testing.assert_allclose(
        vis[bls].imag, expected_imag, rtol=1e-4,
        err_msg=(
            "V_imag wrong sign or magnitude — expected V_ij = conj(E_i) · E_j "
            "convention (D5 / bfCorr line 559-581). A wrong sign would give "
            "negative imag."
        ),
    )


@gpu_required
def test_kernel_fp16_matches_fp32_within_hmma_precision() -> None:
    """Production fp16 path reproduces fp32-input kernel within HMMA precision.

    The production path runs fp16 inputs through HMMA (fp32 accumulators, fp16
    output). The fp32 path runs fp32 inputs through FMA (fp32 throughout).
    Both should agree to ~1% on per-element level given the 0.05 fluff that
    keeps |E|² ≤ 0.16.
    """
    device = torch.device("cuda")
    rng = np.random.default_rng(THERMAL_NOISE_BASE + 99)
    sigma_pre = 2.0  # post-fluff σ = 0.1
    real_q = _quantize_to_int4(rng.normal(0, sigma_pre, size=_FADA_VOLT_SHAPE))
    imag_q = _quantize_to_int4(rng.normal(0, sigma_pre, size=_FADA_VOLT_SHAPE))
    raw_bytes = _pack_bytes_int4(real_q, imag_q)

    vis_fp16 = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float16)
    vis_fp32 = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float32)

    # Per-element fractional difference (autocorrs dominate magnitude).
    typical = float(np.abs(vis_fp32).mean())
    abs_diff = np.abs(vis_fp16.astype(np.complex128) - vis_fp32.astype(np.complex128))
    max_frac = abs_diff.max() / typical
    mean_frac = abs_diff.mean() / typical
    assert mean_frac < 0.005, (
        f"fp16 vs fp32 mean fractional diff = {mean_frac:.3e} > 0.5%"
    )
    assert max_frac < 0.05, (
        f"fp16 vs fp32 max fractional diff = {max_frac:.3e} > 5%"
    )


@gpu_required
def test_thermal_noise_autocorr_recovers_variance() -> None:
    """Thermal noise: per-(ant, ch, pol) variance recovered to ≤ 10× Wishart bound.

    Plan §8 M2 DoD line 2168 (a). Per autocorr fractional uncertainty is
    ~1/√K = 1.6%; mean over `NANTS` autocorrs is ~1/√(K·NANTS) ≈ 1.6e-3.
    We allow a 10× safety margin for quantization + pseudorandom-seed
    fluctuations.
    """
    device = torch.device("cuda")
    rng = np.random.default_rng(THERMAL_NOISE_BASE)
    sigma_pre = 2.0  # post-fluff σ = 0.1; max int4 |x| ≈ 4σ ≈ 8 (no saturation)
    real_q = _quantize_to_int4(rng.normal(0, sigma_pre, size=_FADA_VOLT_SHAPE))
    imag_q = _quantize_to_int4(rng.normal(0, sigma_pre, size=_FADA_VOLT_SHAPE))
    raw_bytes = _pack_bytes_int4(real_q, imag_q)

    vis = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float32)

    # Compute the EXPECTED autocorr from the actually-quantized voltages
    # (avoids dequantization-vs-truth mismatch).
    real_post = real_q.astype(np.float64) * LEGACY_FLUFF_SCALE
    imag_post = imag_q.astype(np.float64) * LEGACY_FLUFF_SCALE
    pwr = (real_post ** 2 + imag_post ** 2)        # (NPACKETS, NANTS, NCHAN, 2t, NPOL)
    expected_autocorr = pwr.sum(axis=(0, 3))[..., :BADA_NPOL]   # (NANTS, NCHAN, BADA_NPOL)

    auto_idx = _autocorr_indices()
    autocorr = vis[auto_idx]                                    # (NANTS, NCHAN, BADA_NPOL)

    # Autocorrs should be (close to) real.
    max_im_re_ratio = float(
        (np.abs(autocorr.imag) / np.maximum(np.abs(autocorr.real), 1e-30)).max()
    )
    assert max_im_re_ratio < 1e-4, (
        f"autocorrs not real, max |Im|/|Re| = {max_im_re_ratio:.3e}"
    )

    # GEMM-faithfulness: per-(ant, ch, pol) autocorr should equal the analytic
    # sum of |E|² over time. Tolerance: fp32 GEMM precision ~ 1e-5 relative.
    rel_err = np.abs(autocorr.real / expected_autocorr - 1)
    p99 = float(np.percentile(rel_err.flatten(), 99))
    assert p99 < 1e-3, (
        f"autocorr GEMM-faithfulness fail: 99th-percentile relative error "
        f"{p99:.3e} > 1e-3"
    )

    # Statistical sanity: mean of all autocorrs should match the analytic
    # population mean K·2σ² to within the Wishart envelope.
    sigma_post = sigma_pre * LEGACY_FLUFF_SCALE
    pop_mean = N_TIME * 2.0 * (sigma_post ** 2)
    obs_mean = float(autocorr.real.mean())
    sample_unc = pop_mean / math.sqrt(N_TIME * NANTS * NCHAN_PER_CHGROUP * BADA_NPOL)
    rel_err_mean = abs(obs_mean / pop_mean - 1)
    assert rel_err_mean < 10 * sample_unc / pop_mean, (
        f"observed mean autocorr {obs_mean:.4f} vs expected {pop_mean:.4f} "
        f"(rel err {rel_err_mean:.3e} > 10× Wishart {sample_unc/pop_mean:.3e})"
    )


@gpu_required
def test_thermal_noise_xcorr_integrates_down() -> None:
    """Thermal noise: cross-correlations integrate down toward zero with 1/√N rate.

    Plan §8 M2 DoD line 2168 (b).
    """
    device = torch.device("cuda")
    rng = np.random.default_rng(THERMAL_NOISE_BASE + 1)
    sigma_pre = 2.0
    real_q = _quantize_to_int4(rng.normal(0, sigma_pre, size=_FADA_VOLT_SHAPE))
    imag_q = _quantize_to_int4(rng.normal(0, sigma_pre, size=_FADA_VOLT_SHAPE))
    raw_bytes = _pack_bytes_int4(real_q, imag_q)

    vis = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float32)

    # Mask autocorrs out.
    auto_idx = _autocorr_indices()
    is_auto = np.zeros(NBASE, dtype=bool)
    is_auto[auto_idx] = True
    xcorr = vis[~is_auto]                                       # (NBASE-NANTS, NCHAN, BADA_NPOL)

    # Per-real-component xcorr expected std = √(K · σ⁴ · 2) under the
    # complex Gaussian model (V = sum_t E_i* E_j; var(V_R) = K · σ_i² σ_j² ·2).
    sigma_post = sigma_pre * LEGACY_FLUFF_SCALE
    expected_std_per_component = math.sqrt(N_TIME) * (sigma_post ** 2) * math.sqrt(2.0)

    # Empirical: per-(ch, pol) std across baselines should be within ~10% of expected.
    obs_std_real = float(xcorr.real.std())
    obs_std_imag = float(xcorr.imag.std())
    for label, obs in (("real", obs_std_real), ("imag", obs_std_imag)):
        rel = abs(obs / expected_std_per_component - 1)
        assert rel < 0.1, (
            f"xcorr {label}-component std {obs:.4f} vs expected "
            f"{expected_std_per_component:.4f} (rel err {rel:.3e} > 10%)"
        )

    # Mean of all xcorr should be near zero (large-N average).
    # Std of the mean = expected_std / √(N_xcorr × NCHAN × BADA_NPOL).
    n_samples = (NBASE - NANTS) * NCHAN_PER_CHGROUP * BADA_NPOL
    mean_unc = expected_std_per_component / math.sqrt(n_samples)
    obs_mean_real = float(xcorr.real.mean())
    obs_mean_imag = float(xcorr.imag.mean())
    assert abs(obs_mean_real) < 6 * mean_unc, (
        f"xcorr real mean {obs_mean_real:.4e} > 6σ = {6*mean_unc:.4e}"
    )
    assert abs(obs_mean_imag) < 6 * mean_unc, (
        f"xcorr imag mean {obs_mean_imag:.4e} > 6σ = {6*mean_unc:.4e}"
    )


@gpu_required
def test_point_source_planar_wave_phase_recovery() -> None:
    """Point source at known (l, m): V_ij phase matches analytical planar-wave model.

    Plan §8 M2 DoD line 2169. Uses a synthetic linear east-west antenna
    array (no real antpos dependency).
    """
    device = torch.device("cuda")

    # Synthetic E-W linear array: 0.1-m spacing per antenna.
    ant_pos_m = np.zeros((NANTS, 3), dtype=np.float64)
    ant_pos_m[:, 0] = 0.1 * np.arange(NANTS)

    # Source direction (small l so we don't fold past Nyquist).
    l, m = 0.05, 0.0
    n = math.sqrt(1.0 - l * l - m * m)
    s_hat = np.array([l, m, n], dtype=np.float64)

    # Frequency: 1.45-1.5 GHz (DSA-like band, 50 MHz wide).
    nu_GHz = np.linspace(1.5, 1.45, NCHAN_PER_CHGROUP)        # decreasing per dsa convention
    nu_Hz = nu_GHz * 1e9

    # Per-(ant, ch) phase φ = 2π · (b · ŝ) · ν / c (no time variation).
    bdotS = ant_pos_m @ s_hat                                  # (NANTS,)
    phase = 2 * math.pi * bdotS[:, None] * nu_Hz[None, :] / SPEED_OF_LIGHT_M_PER_S
    # Pre-fluff amplitude — keep within ±7 to avoid saturation.
    S_pre = 4.0
    E_ant_ch = S_pre * np.exp(1j * phase)                       # (NANTS, NCHAN)

    # Broadcast to fada layout (constant across pkt, t_sub, pol).
    E_full = np.broadcast_to(
        E_ant_ch[None, :, :, None, None], _FADA_VOLT_SHAPE,
    ).copy()

    real_q = _quantize_to_int4(E_full.real)
    imag_q = _quantize_to_int4(E_full.imag)
    raw_bytes = _pack_bytes_int4(real_q, imag_q)

    vis = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float32)

    # Analytical model on the actually-quantized voltages.
    # E_real_post[ant, ch] · scale = post-fluff voltage for this synthetic
    # (constant across pkt, t_sub, pol so we can read directly from real_q[0]).
    E_post = (real_q[0, :, :, 0, 0].astype(np.float64) +
              1j * imag_q[0, :, :, 0, 0].astype(np.float64)) * LEGACY_FLUFF_SCALE
    # V_analytic[ij, ch] = sum_t conj(E_i) · E_j = K · conj(E_i) · E_j
    a_idx, b_idx = upper_tri_indices(NANTS)
    V_analytic = N_TIME * np.conj(E_post[a_idx]) * E_post[b_idx]
    # (NBASE, NCHAN); broadcast across pol for comparison
    V_analytic = V_analytic[..., None].repeat(BADA_NPOL, axis=-1)

    # Compare every baseline's complex visibility to the analytic prediction.
    typical = float(np.abs(V_analytic).mean())
    abs_diff = np.abs(vis.astype(np.complex128) - V_analytic)
    p99 = float(np.percentile(abs_diff.flatten() / typical, 99))
    assert p99 < 0.02, (
        f"point-source visibility 99th-pct relative error {p99:.3e} > 2% "
        f"(quantization-noise floor expected ~1%)"
    )

    # Spot-check the phase rotation on a long baseline (a=NANTS-1, b=0).
    bls = (NANTS - 1) * NANTS // 2 + 0
    phase_obs = np.angle(vis[bls, :, 0])
    phase_ana = np.angle(V_analytic[bls, :, 0])
    phase_diff = np.angle(np.exp(1j * (phase_obs - phase_ana)))   # wrap
    assert float(np.abs(phase_diff).max()) < 0.05, (
        f"point-source phase max diff on (a={NANTS-1}, b=0) = "
        f"{float(np.abs(phase_diff).max()):.3f} rad > 0.05 rad"
    )


@gpu_required
def test_bandpass_shape_recovered_in_autocorr() -> None:
    """Per-(ant, pol) bandpass shape on input voltages is preserved in autocorr.

    Plan §8 M2 DoD line 2170. Per F1 revised: GEMM faithfulness, NOT cal
    application. Slow correlator does not apply or remove bandpass; the
    autocorr just reflects the input voltage |bp|² shape.
    """
    device = torch.device("cuda")
    rng = np.random.default_rng(BANDPASS_BASE)

    # Smooth bandpass shape: linear tilt + slight curvature.
    nu_norm = np.linspace(-1, 1, NCHAN_PER_CHGROUP)
    bandpass_shape = 1.0 + 0.3 * nu_norm + 0.2 * (2 * nu_norm ** 2 - 1)  # (NCHAN,)
    # Per-(ant, pol) random multiplier in [0.8, 1.2].
    ant_pol_gain = 1.0 + 0.4 * (rng.random((NANTS, NPOL)) - 0.5)
    bp = ant_pol_gain[:, None, :] * bandpass_shape[None, :, None]  # (NANTS, NCHAN, NPOL) real
    bp = bp.astype(np.float64)

    # Thermal noise base — choose σ small so |bp · noise| stays in ±7 int4 range.
    sigma_pre_base = 1.0
    thermal_R = rng.normal(0, sigma_pre_base, size=_FADA_VOLT_SHAPE)
    thermal_I = rng.normal(0, sigma_pre_base, size=_FADA_VOLT_SHAPE)

    # Multiply (R, I) by real bandpass.
    bp_broad = bp[None, :, :, None, :]  # (1, NANTS, NCHAN, 1, NPOL)
    real_pre = thermal_R * bp_broad
    imag_pre = thermal_I * bp_broad

    real_q = _quantize_to_int4(real_pre)
    imag_q = _quantize_to_int4(imag_pre)
    raw_bytes = _pack_bytes_int4(real_q, imag_q)

    vis = _kernel_correlate(raw_bytes, device=device, out_dtype=torch.float32)

    auto_idx = _autocorr_indices()
    auto = vis[auto_idx].real                                   # (NANTS, NCHAN, BADA_NPOL)

    # Normalised per-(ant, pol) shape across the band.
    auto_norm = auto / auto.mean(axis=1, keepdims=True)
    bp2 = (bp ** 2)[..., :BADA_NPOL]                            # (NANTS, NCHAN, BADA_NPOL)
    bp2_norm = bp2 / bp2.mean(axis=1, keepdims=True)

    rel_err = np.abs(auto_norm / bp2_norm - 1)
    # Per-bin Wishart noise on autocorr is ~1/√K = 1.6%; quantization adds a
    # bit more. 99th percentile bound at 5% comfortably covers both.
    p99 = float(np.percentile(rel_err.flatten(), 99))
    assert p99 < 0.05, (
        f"bandpass shape recovery: 99th-pct relative error = {p99:.3e} > 5% "
        f"(expected Wishart 1/√K + quantization ≪ 5%)"
    )


@gpu_required
def test_pack_bada_block_byte_layout_matches_meridian_fringestop() -> None:
    """`pack_bada_block` produces bytes that view as `(nbls, nchan, npol)` complex64.

    Mirrors `dsamfs/utils.py::read_buffer` lines 141-155 (legacy assumption
    embedded in `meridian_fringestop`).
    """
    rng = np.random.default_rng(0x600D)
    vis = torch.from_numpy(
        rng.standard_normal((NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)).astype(np.float32)
        + 1j * rng.standard_normal((NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)).astype(np.float32)
    ).to(torch.complex64)
    bytes_out = pack_bada_block(vis)
    assert bytes_out.dtype == np.uint8
    assert bytes_out.size == NBASE * NCHAN_PER_CHGROUP * BADA_NPOL * 8

    # Round-trip view: reinterpret as complex64 and reshape — should equal vis.
    vis_recovered = (
        bytes_out.view(np.complex64)
        .reshape(NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)
    )
    np.testing.assert_array_equal(vis_recovered, vis.numpy())


# ---------------------------------------------------------------------------
# Optional: half_fac=4 (legacy bfCorr K-chunking) sanity check.
# ---------------------------------------------------------------------------


@gpu_required
def test_half_fac_4_matches_half_fac_1_within_fp16_precision() -> None:
    """`half_fac=4` (legacy bfCorr chunking) gives the same answer as default.

    With the 0.05 fluff the K=4096 fp16 sum is ≤ 655 — well within fp16
    range — so halfFac=4's per-chunk fp32 cast doesn't change the answer
    materially. This test guards against regressions in the chunked code
    path (used as a safety knob for production-RFI conditions).
    """
    device = torch.device("cuda")
    rng = np.random.default_rng(THERMAL_NOISE_BASE + 200)
    real_q = _quantize_to_int4(rng.normal(0, 2.0, size=_FADA_VOLT_SHAPE))
    imag_q = _quantize_to_int4(rng.normal(0, 2.0, size=_FADA_VOLT_SHAPE))
    raw_bytes = _pack_bytes_int4(real_q, imag_q)

    vis_hf1 = _kernel_correlate(raw_bytes, device=device,
                                 out_dtype=torch.float16, half_fac=1)
    vis_hf4 = _kernel_correlate(raw_bytes, device=device,
                                 out_dtype=torch.float16, half_fac=4)

    typical = float(np.abs(vis_hf1).mean())
    max_diff = float(np.abs(vis_hf1 - vis_hf4).max() / typical)
    assert max_diff < 0.01, (
        f"halfFac=4 vs halfFac=1 max diff {max_diff:.3e} > 1% — "
        f"fp16 K-chunked accumulation regressed"
    )
