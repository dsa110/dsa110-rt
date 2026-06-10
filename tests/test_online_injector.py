"""tests/test_online_injector.py — M3 chunk 3d acceptance tests.

Pins :mod:`dsart.inject.online` against the per-cached-table math
(phasor / dispersion / profile), the per-block contribution scaling,
and the F22 visibility-phase guard (the strongest sign-convention
test for the injector — pins to F18 / §8.M2-carryover-line-2227's
DSA-110 voltage convention).

All tests use synthetic inputs (no h01 voltage fixtures); CPU paths
where they exercise the same numerical kernels as the GPU path so the
suite runs in seconds even without a GPU. The few GPU-required tests
guard with ``pytest.mark.skipif`` and use small synthetic geometries.

See M3_PLAN_FIXES.md for the proposed F-item flagging the chunk-3d
briefing's negative-sign-per-ant-phasor sentence (the implementation
uses the POSITIVE-sign convention so test_F22_* below passes).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    K_DM_MS_GHZ2_PC,
    NANTS,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NPOL,
    NU_CHGROUP_BOT_GHZ,
    NU_TOP_PROC_GHZ,
    SPEED_OF_LIGHT_M_S,
    freq_GHz,
)
from dsart.inject.etcd_watcher import (  # noqa: E402
    MockEtcdWatcher,
    build_inject_key_prefix,
    handle_inject_payload,
)
from dsart.inject.online import (  # noqa: E402
    MAX_WIDTH_SAMPLES,
    InjectionConfig,
    OnlineInjector,
    build_dispersion_delay_table_ms,
    build_phasor_table,
    build_profile_vector,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    upper_tri_indices,
)


GPU_AVAILABLE = torch.cuda.is_available()
gpu_required = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="needs CUDA",
)


# ---------------------------------------------------------------------------
# Synthetic geometry helper
# ---------------------------------------------------------------------------


def _antpos_synth_2d(rng: np.random.Generator | None = None) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Synthetic DSA-110-like 96-ant 2D layout.

    ~150 m E-W spread × ~50 m N-S spread, all U=0 (planar core).
    Returns ``(antpos_e, antpos_n, antpos_u)`` each ``(NANTS,)`` float64.
    """
    rng = np.random.default_rng(0xCAFE) if rng is None else rng
    e = np.linspace(-75.0, +75.0, NANTS) + rng.normal(0, 0.5, NANTS)
    n = np.linspace(-25.0, +25.0, NANTS) + rng.normal(0, 0.3, NANTS)
    u = np.zeros(NANTS, dtype=np.float64)
    return e, n, u


def _make_injector(
    *,
    chgroup: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    antpos: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> OnlineInjector:
    if antpos is None:
        e, n, u = _antpos_synth_2d()
    else:
        e, n, u = antpos
    return OnlineInjector(
        antpos_e=e, antpos_n=n, chgroup=chgroup,
        device=torch.device(device), dtype=dtype, antpos_u=u,
    )


def _zero_voltages(
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate empty (zero) GEMM-layout voltage tensors."""
    shape = (
        NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
        NPACKETS_PER_BLOCK, NANTS,
    )
    return (
        torch.zeros(shape, device=device, dtype=dtype),
        torch.zeros(shape, device=device, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# 1. Phasor table — unit modulus + correct phase
# ---------------------------------------------------------------------------


def test_phasor_table_unit_modulus() -> None:
    """Every cached phasor cell has |phasor| ≈ 1.0 to numerical precision.

    Built in complex128 + cast to complex64 inside :func:`build_phasor_table`,
    so the cast residual is bounded by the complex64 epsilon (~1e-7).
    """
    e, n, u = _antpos_synth_2d()
    p = build_phasor_table(
        l_rad=0.05, m_rad=-0.03,
        antpos_e=e, antpos_n=n, antpos_u=u,
        chgroup=3,
    )
    assert p.shape == (NANTS, NCHAN_PER_CHGROUP)
    assert p.dtype == np.complex64
    mag = np.abs(p)
    # complex64's float32 representation gives ~1e-7 relative error
    # on the unit-circle round-trip.
    assert np.max(np.abs(mag - 1.0)) < 1e-5, (
        f"phasor unit-modulus violated: max |·| - 1 = "
        f"{float(np.max(np.abs(mag - 1.0))):.3e}"
    )


def test_phasor_table_phase_matches_formula() -> None:
    """Phase per (ant, ch) matches +2π·ν·(E·l + N·m + U·n)/c exactly."""
    e, n, u = _antpos_synth_2d()
    l, m = 0.04, 0.02
    n_dir = math.sqrt(1.0 - l * l - m * m)
    p = build_phasor_table(
        l_rad=l, m_rad=m,
        antpos_e=e, antpos_n=n, antpos_u=u,
        chgroup=2,
    )
    # Hand-compute expected phase for a few cells (full grid would be
    # ~37k checks; sample 50 (ant, ch) pairs).
    rng = np.random.default_rng(0xBEEF)
    a_pick = rng.integers(0, NANTS, size=50)
    c_pick = rng.integers(0, NCHAN_PER_CHGROUP, size=50)
    for a, c in zip(a_pick, c_pick):
        f_hz = freq_GHz(2, int(c)) * 1.0e9
        rdotS = e[a] * l + n[a] * m + u[a] * n_dir
        expected_arg = 2.0 * math.pi * f_hz * rdotS / SPEED_OF_LIGHT_M_S
        observed_arg = math.atan2(p[a, c].imag, p[a, c].real)
        # Wrap into [-π, π] for comparison.
        diff = (observed_arg - expected_arg + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-5, (
            f"(a={a}, c={c}): phase diff {diff:.3e} > 1e-5"
        )


# ---------------------------------------------------------------------------
# 2. Dispersion delay table — cold-plasma law to 1e-6 ms
# ---------------------------------------------------------------------------


def test_dispersion_delay_table_matches_cold_plasma() -> None:
    """Top-vs-bottom delta at DM=400 in chgroup 0 matches cold-plasma analytical.

    Reference ν_top = NU_TOP_PROC_GHZ = 1.49875 GHz.
    Reference ν_bot = NU_CHGROUP_BOT_GHZ[0] (bottom of chgroup 0).
    Expected: K · DM · (1/ν_bot² − 1/ν_top²) ms.
    """
    DM = 400.0
    delay_ms = build_dispersion_delay_table_ms(
        dm_pc_cm3=DM, chgroup=0,
    )
    assert delay_ms.shape == (NCHAN_PER_CHGROUP,)
    assert delay_ms.dtype == np.float32

    # Top channel (local_ch = 0) is at the top of the chgroup, which for
    # chgroup 0 is also NU_TOP_PROC_GHZ; expected delay = 0.
    assert abs(float(delay_ms[0])) < 1e-6, (
        f"top-channel delay = {float(delay_ms[0]):.3e} ms; expected 0"
    )

    # Bottom channel (local_ch = NCHAN-1) gives the chgroup-bottom delay.
    nu_bot = NU_CHGROUP_BOT_GHZ[0]
    expected = K_DM_MS_GHZ2_PC * DM * (
        1.0 / (nu_bot * nu_bot) - 1.0 / (NU_TOP_PROC_GHZ * NU_TOP_PROC_GHZ)
    )
    observed = float(delay_ms[-1])
    assert abs(observed - expected) < 1e-6, (
        f"bottom-channel delay = {observed:.6f} ms vs expected "
        f"{expected:.6f} ms (diff {observed - expected:.3e})"
    )

    # Monotone-positive across descending-frequency channel axis.
    diffs = np.diff(delay_ms)
    assert (diffs >= -1e-7).all(), (
        f"dispersion delay not monotone-non-decreasing; "
        f"min Δ = {float(diffs.min()):.3e}"
    )


# ---------------------------------------------------------------------------
# 3. Profile vector — √(φ(t)/g) "voltage-as-√flux" normalisation
# ---------------------------------------------------------------------------


def test_profile_vector_normalization_gaussian() -> None:
    """Gaussian profile vector is √(φ/g) (voltage-as-√flux convention).

    The cached profile is ``√(φ(t)/g)`` where φ is the unit-PEAK
    Gaussian and g = √(π/(4 ln 2)) is the family area-per-FWHM
    constant. So the centre value is ``1/√g ≈ 0.969`` and the
    discrete L²-sum (= ∫ profile² ≈ ∫ φ/g) equals
    ``∫φ dt / g = (W·g)/g = W`` — i.e. the squared-profile area
    equals the FWHM, regardless of W. (After ``√(fluence/W)``
    apply-time scaling the per-cell signal is ``√F(t)`` and the
    matched-filter SNR scales as ``fluence/√W``.)
    """
    g_norm = math.sqrt(math.pi / (4.0 * math.log(2.0)))
    for width in (4, 8, 16, 64, 256):
        prof = build_profile_vector(
            width_samples=width, profile="gaussian",
        )
        assert prof.shape == (2 * MAX_WIDTH_SAMPLES + 1,)
        assert prof.dtype == np.float32
        # Centre value = √(1/g) = (4 ln 2 / π)^¼.
        peak_expected = 1.0 / math.sqrt(g_norm)
        peak_observed = float(prof[MAX_WIDTH_SAMPLES])
        rel_err_peak = abs(peak_observed - peak_expected) / peak_expected
        assert rel_err_peak < 1e-5, (
            f"width={width}: peak {peak_observed:.6e} vs expected "
            f"{peak_expected:.6e} (rel err {rel_err_peak:.3e})"
        )
        # ∑ profile² ≈ ∫ φ dt / g = W (FWHM in native samples).
        sumsq_observed = float(np.sum(prof.astype(np.float64) ** 2))
        rel_err_sumsq = abs(sumsq_observed - float(width)) / float(width)
        assert rel_err_sumsq < 1e-3, (
            f"width={width}: ∑profile² = {sumsq_observed:.6e} vs "
            f"expected width = {width} (rel err {rel_err_sumsq:.3e})"
        )


def test_profile_vector_boxcar_normalization() -> None:
    """Boxcar profile vector is unit-PEAK √(φ/1) = φ inside the window."""
    for width in (1, 2, 7, 32, 128):
        prof = build_profile_vector(
            width_samples=width, profile="boxcar",
        )
        # Boxcar peak = 1 (g=1 ⇒ √(φ/g) = φ; φ = 1 in the window).
        # Centre cell is in the half-open interval [-w/2, +w/2).
        assert abs(float(prof[MAX_WIDTH_SAMPLES]) - 1.0) < 1e-6
        # ∑ profile² = ∑ profile = width (every in-window sample is 1).
        np.testing.assert_allclose(
            float(np.sum(prof.astype(np.float64) ** 2)),
            float(width), atol=1e-6,
            err_msg=f"width={width}: ∑profile² != width",
        )
        np.testing.assert_allclose(
            float(prof.sum()), float(width), atol=1e-6,
            err_msg=f"width={width}: boxcar sum != width",
        )


# ---------------------------------------------------------------------------
# 4. apply_block — no pending injection ⇒ noop
# ---------------------------------------------------------------------------


def test_apply_block_no_injection_pending_is_noop() -> None:
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    real_v, imag_v = _zero_voltages()
    real_orig = real_v.clone()
    imag_orig = imag_v.clone()
    log = inj.apply_block(real_v, imag_v, block_specnum_start=10_000)
    assert log["active_inj_ids"] == []
    assert log["contributions"] == []
    assert log["n_purged"] == 0
    assert torch.equal(real_v, real_orig)
    assert torch.equal(imag_v, imag_orig)


# ---------------------------------------------------------------------------
# 5. apply_block — active injection adds to voltages with correct
#    fluence + phase scaling
# ---------------------------------------------------------------------------


def test_apply_block_active_injection_adds_to_voltages() -> None:
    """Active injection deposits a contribution whose RMS scales with √(fluence/width)."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    apply_native = NPACKETS_PER_BLOCK * NTIMES_PER_PACKET // 2
    cfg = InjectionConfig(
        inj_id="test_active_1",
        l_rad=0.0, m_rad=0.0,                  # zenith → no per-(ant, ch) phase
        dm_pc_cm3=0.0,                         # no dispersion
        fluence_jy_ms=4.0,
        width_samples=16,
        profile="boxcar",
        apply_at_specnum=apply_native // 2,
    )
    inj.add_pending(cfg)
    real_v, imag_v = _zero_voltages()
    log = inj.apply_block(real_v, imag_v, block_specnum_start=0)
    assert len(log["active_inj_ids"]) == 1
    assert log["active_inj_ids"][0] == "test_active_1"

    # At zenith, phasor = (1 + 0j) for every ant and channel, so the
    # voltage contribution lives entirely in the real part.
    nz_real = real_v[real_v != 0]
    nz_imag = imag_v[imag_v != 0]
    assert nz_real.numel() > 0, "real voltages should have non-zero contribution"
    assert nz_imag.numel() == 0, (
        "imag voltages should be zero (zenith ⇒ pure-real phasor)"
    )

    # Boxcar (g = 1): every in-window cell holds peak-1 · √(fluence/W) =
    # √(fluence/W). With the post-fix √(φ/g) profile convention this is
    # the per-cell voltage = √F(t) for an in-window sample.
    expected_cell = math.sqrt(
        cfg.fluence_jy_ms / cfg.width_samples,
    )
    np.testing.assert_allclose(
        nz_real.numpy(), expected_cell, rtol=1e-5,
        err_msg=(
            f"per-cell voltage contribution (sample {float(nz_real[0]):.6e}) "
            f"vs expected {expected_cell:.6e}"
        ),
    )

    # Per-block log RMS should match the per-cell value too (since all
    # in-window cells are identical and zenith-pure-real).
    contrib = log["contributions"][0]
    # The contribution log RMS is computed across the FULL (C, T, P, A)
    # grid post-broadcast to NPOL — 0 outside the window. RMS = (in_window
    # cells * value²) / (total cells) ⇒ √(p · v²) where p is fill
    # fraction. Here in-window count per (ant, ch) = width_samples (= 16);
    # full-grid native-time count = NPACKETS * NTIMES = 4096; ⇒ p =
    # 16 / 4096 = 1/256.
    fill_frac = cfg.width_samples / (NPACKETS_PER_BLOCK * NTIMES_PER_PACKET)
    expected_rms = math.sqrt(fill_frac) * expected_cell
    assert abs(contrib["rms_real_added"] - expected_rms) / expected_rms < 0.05, (
        f"log rms_real_added {contrib['rms_real_added']:.4e} vs expected "
        f"{expected_rms:.4e}"
    )


# ---------------------------------------------------------------------------
# 6. F22 — visibility phase recovery matches +2πν(b·ŝ)/c
# ---------------------------------------------------------------------------


def test_F22_visibility_phase_matches_lm_target() -> None:
    """Sign-convention pin: visibility phase from injector matches F18.

    Synthesise an injection at (l, m) = (0.05, -0.03), DM=0, boxcar
    profile of width 8 samples. Pick ONE native time-sample inside
    the boxcar window, manually GEMM two arbitrary baselines via
    M2's ``upper_tri_indices`` extracting at ``V[b_idx, a_idx]`` (=
    ``conj(E_lower) · E_higher``, F18). Confirm the recovered
    visibility phase across all 384 channels matches

        +2π · ν · ((E_b - E_a) · l + (N_b - N_a) · m) / c

    within < 0.01 rad.

    The OPPOSITE sign would indicate that the per-antenna phasor
    convention used by the injector is the briefing's stated negative
    sign — which we explicitly *do not* implement (see the chunk-3d
    final report for the F-item proposal).
    """
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    e, n, _u = _antpos_synth_2d()

    L, M = 0.05, -0.03
    cfg = InjectionConfig(
        inj_id="F22_pin",
        l_rad=L, m_rad=M,
        dm_pc_cm3=0.0,
        fluence_jy_ms=1.0,
        width_samples=8,
        profile="boxcar",
        apply_at_specnum=128,
    )
    inj.add_pending(cfg)
    real_v, imag_v = _zero_voltages(device="cpu", dtype=torch.float32)
    inj.apply_block(real_v, imag_v, block_specnum_start=0)

    # Pick a (pkt, t_sub) inside the boxcar window centred at
    # apply_at_specnum=128 ⇒ peak_native = 256, half-width 4 ⇒ window
    # native ∈ [252, 260). Any (pkt, t_sub) with pkt*2+t_sub ∈ that
    # window works; choose pkt=128, t_sub=0 → native_pos = 256.
    pkt, tsub = 128, 0
    pol = 0
    # Voltage @ this (pkt, t_sub, pol) for every ant: shape (NCHAN, NANTS).
    e_ant_ch_real = real_v[:, tsub, pol, pkt, :]                # (NCHAN, NANTS)
    e_ant_ch_imag = imag_v[:, tsub, pol, pkt, :]
    e_ant_ch = (e_ant_ch_real + 1j * e_ant_ch_imag).numpy()    # complex128

    # All cells should be non-zero (zenith would put 0 in imag, but
    # off-zenith puts both); pick TWO baselines and form the F18
    # visibility manually.
    a_idx, b_idx = upper_tri_indices(NANTS)
    # Pick a few baselines with different (a, b) pairs. (a > b in our
    # convention; F18 visibility is conj(E_b) · E_a = conj(E_lower) ·
    # E_higher.)
    bls_picks = [(20, 5), (60, 1), (95, 0)]
    for a, b in bls_picks:
        # Manual GEMM at ONE time sample → V[b, a] = conj(E_b) · E_a.
        e_b = e_ant_ch[:, b]                                    # (NCHAN,)
        e_a = e_ant_ch[:, a]
        v = np.conj(e_b) * e_a                                  # (NCHAN,)
        observed_phase = np.angle(v)

        # Expected phase across channels:
        #   +2π · ν · ((E_a - E_b) · l + (N_a - N_b) · m) / c
        # ('a' is higher index here, matching the F22 docstring's b > a).
        f_hz = np.asarray(
            [freq_GHz(0, ch) * 1.0e9 for ch in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        )
        bdotS = (e[a] - e[b]) * L + (n[a] - n[b]) * M
        expected_phase = (
            2.0 * math.pi * f_hz * bdotS / SPEED_OF_LIGHT_M_S
        )
        # Wrap into [-π, π] before comparison.
        diff = (
            (observed_phase - expected_phase + math.pi) % (2.0 * math.pi)
            - math.pi
        )
        max_abs_diff = float(np.max(np.abs(diff)))
        assert max_abs_diff < 0.01, (
            f"baseline (a={a}, b={b}): max |phase diff| = "
            f"{max_abs_diff:.4e} rad > 0.01 rad. The sign-convention "
            "pin (F22) failed — the per-antenna phasor sign or the "
            "F18 visibility convention has drifted."
        )


# ---------------------------------------------------------------------------
# 7. apply_block — injection outside the block's specnum window ⇒ noop
# ---------------------------------------------------------------------------


def test_apply_block_injection_outside_window_is_noop() -> None:
    """Far-future injection: contribution is empty (but injection stays pending)."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    cfg = InjectionConfig(
        inj_id="future_inj",
        l_rad=0.0, m_rad=0.0,
        dm_pc_cm3=0.0,
        fluence_jy_ms=10.0, width_samples=4,
        profile="gaussian",
        apply_at_specnum=10_000_000,           # way past block start
    )
    inj.add_pending(cfg)
    real_v, imag_v = _zero_voltages()
    log = inj.apply_block(real_v, imag_v, block_specnum_start=0)
    assert log["active_inj_ids"] == [], (
        "future injection should not contribute to current block"
    )
    assert log["n_purged"] == 0, (
        "future injection should remain pending (purge only when past)"
    )
    assert "future_inj" in inj.pending
    # Voltages still zero.
    assert torch.equal(real_v, torch.zeros_like(real_v))
    assert torch.equal(imag_v, torch.zeros_like(imag_v))


def test_apply_block_injection_far_past_is_purged() -> None:
    """Past-tense injection: purged from pending (and contributes nothing)."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    cfg = InjectionConfig(
        inj_id="past_inj",
        l_rad=0.0, m_rad=0.0,
        dm_pc_cm3=0.0,
        fluence_jy_ms=10.0, width_samples=4,
        profile="gaussian",
        apply_at_specnum=0,
    )
    inj.add_pending(cfg)
    real_v, imag_v = _zero_voltages()
    # Block starts at specnum 5_000_000 → all inj footprint is in the
    # past, should be purged (no contribution).
    log = inj.apply_block(
        real_v, imag_v, block_specnum_start=5_000_000,
    )
    assert log["active_inj_ids"] == []
    assert log["n_purged"] == 1
    assert "past_inj" not in inj.pending
    # 2026-06-10 late-command instrumentation: purged with zero
    # contribution → counted as never_applied (the "injection never
    # showed up" smoking gun).
    assert inj.n_never_applied == 1
    assert inj.n_applied_clean == 0
    assert inj.n_applied_partial == 0


# ---------------------------------------------------------------------------
# 7b. 2026-06-10 late-command instrumentation (260610snoe/mamv root
#     cause): a command arriving after the pipeline passed apply_at
#     loses its leading (high-frequency, small-dispersion-delay)
#     channels → partial-band injection. The injector must count and
#     classify clean / partial / never-applied first applications.
# ---------------------------------------------------------------------------


def test_apply_block_on_time_injection_counts_clean() -> None:
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    cfg = InjectionConfig(
        inj_id="clean_inj",
        l_rad=0.0, m_rad=0.0,
        dm_pc_cm3=0.0,
        fluence_jy_ms=10.0, width_samples=4,
        profile="gaussian",
        apply_at_specnum=128,
    )
    inj.add_pending(cfg)
    real_v, imag_v = _zero_voltages(device="cpu", dtype=torch.float32)
    log = inj.apply_block(real_v, imag_v, block_specnum_start=0)
    assert "clean_inj" in log["active_inj_ids"]
    assert inj.n_applied_clean == 1
    assert inj.n_applied_partial == 0
    assert inj.n_never_applied == 0


def test_apply_block_late_command_counts_partial_band() -> None:
    """A dispersed (high-DM) injection whose apply_at is already past
    when the first block arrives: the band-top channels' windows have
    passed (lost) but the band-bottom channels' dispersion delay keeps
    their windows in the future — the injection still contributes,
    but PARTIALLY. The instrumentation must classify it as partial
    and report the number of fully-lost channels."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    # DM 2500 at chgroup 0: per-channel delays span ~0 (top channel)
    # to a few thousand native samples within the chgroup — enough to
    # straddle a block boundary.
    # Place the peak so the dispersed window straddles the boundary
    # between block 0 and block 1 (native position 4096): at DM 2500
    # the chgroup-0 in-band dispersion spans ~2200 native samples, so
    # peak_native = 3000 puts roughly the top half of the band's
    # windows before 4096 (lost when the command arrives at block 1)
    # and the bottom half after (still injectable).
    block_native = NPACKETS_PER_BLOCK * NTIMES_PER_PACKET   # 4096
    cfg = InjectionConfig(
        inj_id="late_inj",
        l_rad=0.0, m_rad=0.0,
        dm_pc_cm3=2500.0,
        fluence_jy_ms=10.0, width_samples=4,
        profile="gaussian",
        apply_at_specnum=(block_native - 1096) // 2,   # peak_native = 3000
    )
    inj.add_pending(cfg)
    active = inj.pending["late_inj"]
    disp = active.dispersion_offset_samples
    # Sanity: the dispersed signal really does straddle native 4096.
    assert active.peak_native + int(disp.min().item()) < block_native
    assert active.peak_native + int(disp.max().item()) > block_native
    real_v, imag_v = _zero_voltages(device="cpu", dtype=torch.float32)
    # Command "arrives" only at block 1 — block 0 was never offered.
    log = inj.apply_block(
        real_v, imag_v, block_specnum_start=NPACKETS_PER_BLOCK,
    )
    # Still contributes (band-bottom channels in window)…
    assert "late_inj" in log["active_inj_ids"]
    # …but classified PARTIAL with a non-trivial channel loss.
    assert inj.n_applied_partial == 1
    assert inj.n_applied_clean == 0
    assert inj.last_partial_lost_channels > 0


# ---------------------------------------------------------------------------
# 8. apply_block — boundary-straddling injection across two blocks
# ---------------------------------------------------------------------------


def test_apply_block_injection_at_block_boundary() -> None:
    """Boxcar straddling block boundary: leading half + trailing half = whole.

    Place a boxcar of width ``W`` whose centre lands AT the boundary
    between two blocks (the boundary is `block_specnum_start * 2` for
    the second block). Run ``apply_block`` on both blocks; assert
    each block sees exactly ``W/2`` non-zero native samples and the
    concatenation reconstructs the full ``W``-sample boxcar.
    """
    width = 32                                   # even ⇒ symmetric split
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    # apply_at_specnum * 2 = peak_native; peak == native sample
    # NPACKETS_PER_BLOCK*NTIMES = 4096 (= start of block 1 in native units)
    peak_native = NPACKETS_PER_BLOCK * NTIMES_PER_PACKET   # = 4096
    cfg = InjectionConfig(
        inj_id="boundary_inj",
        l_rad=0.0, m_rad=0.0,
        dm_pc_cm3=0.0,
        fluence_jy_ms=4.0,
        width_samples=width,
        profile="boxcar",
        apply_at_specnum=peak_native // 2,
    )
    inj.add_pending(cfg)

    # Block 0: specnum_start = 0; native window [0, 4096).
    real_v0, imag_v0 = _zero_voltages(device="cpu", dtype=torch.float32)
    log0 = inj.apply_block(real_v0, imag_v0, block_specnum_start=0)
    assert "boundary_inj" in log0["active_inj_ids"]

    # Block 1: specnum_start = NPACKETS_PER_BLOCK; native window
    # [4096, 8192).
    real_v1, imag_v1 = _zero_voltages(device="cpu", dtype=torch.float32)
    log1 = inj.apply_block(
        real_v1, imag_v1, block_specnum_start=NPACKETS_PER_BLOCK,
    )
    assert "boundary_inj" in log1["active_inj_ids"]

    # Boxcar value per (ch, t, pol, pkt, ant) cell INSIDE the window =
    #   phasor[ant, ch] * 1.0 * sqrt(fluence/width)
    # (post-fix √(φ/g) profile convention; boxcar peak = 1, g = 1).
    # At zenith with single-pol same-value, the per-cell magnitude is
    # 1.0 * 1.0 * sqrt(F/W) = const. Count non-zero native samples
    # per pol per ant per channel: should be width/2 in each block.
    # Pick ant=0, ch=0, pol=0, count non-zero entries over (pkt, t_sub).
    block0_nz_per_cell = (real_v0[0, :, 0, :, 0] != 0).sum().item()
    block1_nz_per_cell = (real_v1[0, :, 0, :, 0] != 0).sum().item()
    assert block0_nz_per_cell == width // 2, (
        f"block 0: expected {width//2} non-zero native samples, got "
        f"{block0_nz_per_cell}"
    )
    assert block1_nz_per_cell == width // 2, (
        f"block 1: expected {width//2} non-zero native samples, got "
        f"{block1_nz_per_cell}"
    )

    # Concatenated profile (block 0 native [0, 4096) + block 1 native
    # [4096, 8192)) along the time axis should hold exactly ``width``
    # non-zero entries — the full boxcar.
    # Reshape (TIMES_PER_PACKET, NPACKETS) → flat native time per cell.
    # GEMM time order: native_t = pkt * 2 + t_sub, so reshape via
    # permute(1, 0).reshape(-1).
    cell0 = real_v0[0, :, 0, :, 0].permute(1, 0).reshape(-1)   # (NPACKETS*NTIMES,)
    cell1 = real_v1[0, :, 0, :, 0].permute(1, 0).reshape(-1)
    full = torch.cat([cell0, cell1])
    assert (full != 0).sum().item() == width, (
        f"concatenated nz count {(full!=0).sum().item()} != {width}"
    )


# ---------------------------------------------------------------------------
# 9. etcd payload round-trip
# ---------------------------------------------------------------------------


def test_etcd_payload_round_trip() -> None:
    """JSON serialise + parse round-trip is bit-exact for InjectionConfig."""
    cfg = InjectionConfig(
        inj_id="frb_240505_test",
        l_rad=0.0314159, m_rad=-0.027182,
        dm_pc_cm3=404.688,
        fluence_jy_ms=12.5,
        width_samples=8,
        profile="gaussian",
        apply_at_specnum=17254656,
    )
    s = cfg.to_json()
    cfg2 = InjectionConfig.from_json(s)
    assert cfg2 == cfg, (
        f"round-trip mismatch: original {cfg} vs parsed {cfg2}"
    )
    # Bytes input also accepted.
    cfg3 = InjectionConfig.from_json(s.encode("utf-8"))
    assert cfg3 == cfg


def test_etcd_payload_round_trip_bad_payloads() -> None:
    """Malformed payloads fail with ValueError; missing/extra keys flagged."""
    with pytest.raises(ValueError):
        InjectionConfig.from_json('{"inj_id": "x"}')        # missing keys
    with pytest.raises(ValueError):
        InjectionConfig.from_json('not-json')               # not JSON
    with pytest.raises(ValueError):
        InjectionConfig.from_json('[1, 2, 3]')              # not a dict
    base_dict = {
        "inj_id": "x", "l_rad": 0.0, "m_rad": 0.0, "dm_pc_cm3": 0.0,
        "fluence_jy_ms": 1.0, "width_samples": 4, "profile": "gaussian",
        "apply_at_specnum": 0, "extra_key": "boom",
    }
    import json as _json
    with pytest.raises(ValueError):
        InjectionConfig.from_json(_json.dumps(base_dict))   # extra key


# ---------------------------------------------------------------------------
# Bonus: etcd watcher (Mock) plumbing
# ---------------------------------------------------------------------------


def test_mock_etcd_watcher_routes_inject_payload() -> None:
    """``MockEtcdWatcher.inject_one`` parses + queues an inject payload."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    watcher = MockEtcdWatcher(injector=inj, corr_index=0)
    watcher.start()
    cfg_dict = {
        "inj_id": "watch_test",
        "l_rad": 0.04, "m_rad": 0.0,
        "dm_pc_cm3": 100.0, "fluence_jy_ms": 5.0,
        "width_samples": 4, "profile": "gaussian",
        "apply_at_specnum": 1000,
    }
    payload = {"cmd": "inject", "val": cfg_dict}
    import json as _json
    cfg = watcher.inject_one(_json.dumps(payload))
    assert cfg is not None
    assert cfg.inj_id == "watch_test"
    assert "watch_test" in inj.pending
    watcher.stop()


def test_mock_etcd_watcher_drops_non_inject_commands() -> None:
    """Non-inject commands (e.g. ``cmd: prepare``) are silently ignored."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    watcher = MockEtcdWatcher(injector=inj, corr_index=0)
    cfg = watcher.inject_one('{"cmd": "prepare", "val": {}}')
    assert cfg is None
    assert inj.pending == {}


def test_mock_etcd_watcher_handles_malformed_payload() -> None:
    """Garbage JSON / wrong-shape payloads return None and don't raise."""
    inj = _make_injector(chgroup=0, device="cpu", dtype=torch.float32)
    watcher = MockEtcdWatcher(injector=inj, corr_index=0)
    assert watcher.inject_one("not-json") is None
    assert watcher.inject_one('{"cmd": "inject"}') is None  # missing val
    assert watcher.inject_one('{"cmd": "inject", "val": "not-a-dict"}') is None
    assert inj.pending == {}


def test_build_inject_key_prefix_uses_namespace_env() -> None:
    """``DSART_ETCD_NAMESPACE_PREFIX`` (PARALLEL_AGENTS.md §4.6) honoured."""
    # Production unset.
    saved = os.environ.pop("DSART_ETCD_NAMESPACE_PREFIX", None)
    try:
        assert build_inject_key_prefix(0) == "/cmd/dsart/corr/0/"
        # M3 dev.
        os.environ["DSART_ETCD_NAMESPACE_PREFIX"] = "m3"
        assert build_inject_key_prefix(2) == "/cmd/dsart-m3/corr/2/"
        # M5 dev.
        os.environ["DSART_ETCD_NAMESPACE_PREFIX"] = "m5"
        assert build_inject_key_prefix(7) == "/cmd/dsart-m5/corr/7/"
    finally:
        if saved is None:
            os.environ.pop("DSART_ETCD_NAMESPACE_PREFIX", None)
        else:
            os.environ["DSART_ETCD_NAMESPACE_PREFIX"] = saved


# ---------------------------------------------------------------------------
# Optional GPU smoke test (only runs when CUDA is available)
# ---------------------------------------------------------------------------


@gpu_required
def test_gpu_smoke_apply_block_runs_at_full_size() -> None:
    """Full-size GEMM-layout apply_block runs without error on GPU at fp16."""
    device = torch.device("cuda")
    e, n, _u = _antpos_synth_2d()
    inj = OnlineInjector(
        antpos_e=e, antpos_n=n, chgroup=0,
        device=device, dtype=torch.float16,
    )
    cfg = InjectionConfig(
        inj_id="gpu_smoke",
        l_rad=0.05, m_rad=0.02,
        dm_pc_cm3=200.0, fluence_jy_ms=10.0,
        width_samples=8, profile="gaussian",
        apply_at_specnum=NPACKETS_PER_BLOCK // 2,
    )
    inj.add_pending(cfg)
    shape = (
        NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
        NPACKETS_PER_BLOCK, NANTS,
    )
    real_v = torch.zeros(shape, device=device, dtype=torch.float16)
    imag_v = torch.zeros(shape, device=device, dtype=torch.float16)
    log = inj.apply_block(real_v, imag_v, block_specnum_start=0)
    assert "gpu_smoke" in log["active_inj_ids"]
    # Some non-zero contribution made it in.
    assert float(real_v.abs().sum().item()) > 0
    assert float(imag_v.abs().sum().item()) > 0
    del real_v, imag_v
    torch.cuda.empty_cache()
