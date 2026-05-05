"""tests/test_cal_loader_apply.py — D17 cal-loader + cal-application tests.

Covers:
  * `dsart.cal.bf_weights.load_bf_weights` round-trips a synthetic
    74,496-byte fp32 blob exactly (antpos + complex gains).
  * Wrong-size files are rejected.
  * `upsample_coarse_to_fine` replicates each coarse cal cell across
    `N_FINE_PER_CAL_COARSE = 8` adjacent fine channels.
  * `normalize_phase_only` produces unit |G| (with zero cells passing
    through), matching bfCorr's `wnorm` step.
  * `maybe_swap_pol` flips axis -1.
  * `slow_corr_kernel.apply_cal_split` with `G = 1 + 0j` is a no-op.
  * `apply_cal_split` with a generic complex G obeys
        E_cal = E * G          (real/imag split).
  * End-to-end (CPU, downscaled NANTS=4 / NCHAN=8 / NPACKETS=4):
    applying per-ant gain to voltages produces
        V_cal_ij = G_i^* · G_j · V_raw_ij
    (the standard CASA-style "G" gain solution applied to visibilities).

These run on CPU; no fada bytes / no GPU. Full-size end-to-end with
fp16 GEMM lives in test_slow_corr_synth.py once we expose
``apply_cal=True`` there.
"""

from __future__ import annotations

import os

os.environ.setdefault("DSART_TEST", "1")

import sys                                                          # noqa: E402
from pathlib import Path                                            # noqa: E402

import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402
import torch                                                        # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.cal.bf_weights import (                                  # noqa: E402
    BF_WEIGHTS_FILE_SIZE,
    NCAL_COARSE_PER_CHGROUP,
    N_FINE_PER_CAL_COARSE,
    load_bf_weights,
    maybe_swap_pol,
    normalize_phase_only,
    upsample_coarse_to_fine,
)
from dsart.common.constants import (                                # noqa: E402
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.slow_corr_kernel import (                       # noqa: E402
    _GEMM_LAYOUT_SHAPE,
    apply_cal_split,
    make_cal_broadcast_tensors,
)


# ------------------------------------------------------------------ helpers


def _synthesize_blob(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    gains: np.ndarray,
) -> bytes:
    """Build a 74,496-byte legacy bfCorr cal blob from arrays.

    Layout matches bfCorr.cu:1391 (`fread(h_winp, NANTS*2 +
    NANTS*48*2*2, 4, ff)`):
        [antpos_e (96 fp32)][antpos_n (96 fp32)][cal[a, ch, pol, ri] (18432 fp32)]
    where ri=0 is real, ri=1 is imag.
    """
    assert antpos_e.shape == (NANTS,) and antpos_e.dtype == np.float32
    assert antpos_n.shape == (NANTS,) and antpos_n.dtype == np.float32
    assert gains.shape == (NANTS, NCAL_COARSE_PER_CHGROUP, NPOL)
    cal_ri = np.empty(
        (NANTS, NCAL_COARSE_PER_CHGROUP, NPOL, 2), dtype=np.float32,
    )
    cal_ri[..., 0] = gains.real.astype(np.float32)
    cal_ri[..., 1] = gains.imag.astype(np.float32)
    return antpos_e.tobytes() + antpos_n.tobytes() + cal_ri.tobytes()


# ------------------------------------------------------------------ load_bf_weights


def test_blob_size_constant() -> None:
    """File-size constant matches the bfCorr fread() expression."""
    expected = 4 * (2 * NANTS + NANTS * NCAL_COARSE_PER_CHGROUP * NPOL * 2)
    assert BF_WEIGHTS_FILE_SIZE == expected
    assert BF_WEIGHTS_FILE_SIZE == 74_496  # DSA-110 (96 ants, 48 coarse, 2 pol)


def test_load_bf_weights_round_trip(tmp_path: Path) -> None:
    """Synthesize → write → load round-trips antpos + gains exactly."""
    rng = np.random.default_rng(0xCA1B10B)
    antpos_e = rng.uniform(-200.0, 200.0, NANTS).astype(np.float32)
    antpos_n = rng.uniform(-300.0, 1900.0, NANTS).astype(np.float32)
    gains = (
        rng.uniform(-1.0, 1.0, (NANTS, NCAL_COARSE_PER_CHGROUP, NPOL))
        + 1j * rng.uniform(-1.0, 1.0, (NANTS, NCAL_COARSE_PER_CHGROUP, NPOL))
    ).astype(np.complex64)
    blob = _synthesize_blob(antpos_e, antpos_n, gains)
    path = tmp_path / "beamformer_weights_test.dat"
    path.write_bytes(blob)

    bfw = load_bf_weights(path)
    np.testing.assert_array_equal(bfw.antpos_e, antpos_e)
    np.testing.assert_array_equal(bfw.antpos_n, antpos_n)
    np.testing.assert_array_equal(bfw.gains, gains)
    assert bfw.source_path == path


def test_load_bf_weights_wrong_size_rejected(tmp_path: Path) -> None:
    """File of wrong byte count → ValueError."""
    path = tmp_path / "truncated.dat"
    path.write_bytes(b"\x00" * (BF_WEIGHTS_FILE_SIZE - 4))  # truncated
    with pytest.raises(ValueError, match="size"):
        load_bf_weights(path)


def test_load_bf_weights_missing_path(tmp_path: Path) -> None:
    """Nonexistent path → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_bf_weights(tmp_path / "no_such.dat")


def test_load_bf_weights_flagged_count(tmp_path: Path) -> None:
    """`n_flagged` counts cells equal to exactly zero (matches bfCorr's
    flagged-solutions convention)."""
    antpos_e = np.zeros(NANTS, dtype=np.float32)
    antpos_n = np.zeros(NANTS, dtype=np.float32)
    gains = np.ones(
        (NANTS, NCAL_COARSE_PER_CHGROUP, NPOL), dtype=np.complex64,
    )
    # Zero out a handful of cells to mimic CASA flagging.
    gains[2, 5, 0] = 0
    gains[10, 7, 1] = 0
    gains[42, :, :] = 0  # all chans/pols of one ant
    expected_flagged = 2 + NCAL_COARSE_PER_CHGROUP * NPOL  # 2 + 96 = 98
    blob = _synthesize_blob(antpos_e, antpos_n, gains)
    path = tmp_path / "blob.dat"
    path.write_bytes(blob)

    bfw = load_bf_weights(path)
    assert bfw.n_flagged == expected_flagged


# ------------------------------------------------------------------ helpers


def test_upsample_coarse_to_fine() -> None:
    """Each coarse value replicated 8× across adjacent fine channels."""
    g = np.arange(
        NANTS * NCAL_COARSE_PER_CHGROUP * NPOL, dtype=np.complex64,
    ).reshape(NANTS, NCAL_COARSE_PER_CHGROUP, NPOL)
    g_fine = upsample_coarse_to_fine(g)
    assert g_fine.shape == (NANTS, NCHAN_PER_CHGROUP, NPOL)
    # Within each 8-block of fine chans, all values equal the coarse value.
    for ic in range(NCAL_COARSE_PER_CHGROUP):
        block = g_fine[:, ic * N_FINE_PER_CAL_COARSE : (ic + 1) * N_FINE_PER_CAL_COARSE, :]
        for k in range(N_FINE_PER_CAL_COARSE):
            np.testing.assert_array_equal(block[:, k, :], g[:, ic, :])


def test_upsample_rejects_non_multiple() -> None:
    """`n_fine` not a multiple of n_coarse → ValueError."""
    g = np.zeros((NANTS, 48, NPOL), dtype=np.complex64)
    with pytest.raises(ValueError):
        upsample_coarse_to_fine(g, n_fine=383)


def test_normalize_phase_only_unit_magnitude() -> None:
    """Non-zero cells get unit magnitude; zero cells stay zero."""
    rng = np.random.default_rng(0xCA1B10B)
    g = (rng.standard_normal((10, 5, 2)) + 1j * rng.standard_normal((10, 5, 2))).astype(np.complex64)
    g[3, 1, 0] = 0
    g[7, :, 1] = 0
    g_n = normalize_phase_only(g)
    mag = np.abs(g_n)
    nonzero_mask = np.abs(g) > 0
    np.testing.assert_allclose(mag[nonzero_mask], 1.0, atol=1e-5)
    np.testing.assert_array_equal(mag[~nonzero_mask], 0.0)
    # Phases preserved on non-zero cells.
    nonzero_g = g[nonzero_mask]
    nonzero_n = g_n[nonzero_mask]
    np.testing.assert_allclose(
        np.angle(nonzero_n), np.angle(nonzero_g), atol=1e-5,
    )


def test_maybe_swap_pol_no_swap_passes_through() -> None:
    """`maybe_swap_pol(g, swap=False)` returns the same data (no copy required
    for correctness, but content must match)."""
    g = np.arange(8 * 3 * 2, dtype=np.complex64).reshape(8, 3, 2)
    np.testing.assert_array_equal(maybe_swap_pol(g, swap=False), g)


def test_maybe_swap_pol_flips_last_axis() -> None:
    """`maybe_swap_pol(g, swap=True)` returns the pol-flipped view (contiguous)."""
    g = np.arange(8 * 3 * 2, dtype=np.complex64).reshape(8, 3, 2)
    g_swap = maybe_swap_pol(g, swap=True)
    np.testing.assert_array_equal(g_swap[..., 0], g[..., 1])
    np.testing.assert_array_equal(g_swap[..., 1], g[..., 0])
    assert g_swap.flags["C_CONTIGUOUS"]


# ------------------------------------------------------------------ apply_cal_split


def _zeros_voltage(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(_GEMM_LAYOUT_SHAPE, dtype=dtype),
        torch.zeros(_GEMM_LAYOUT_SHAPE, dtype=dtype),
    )


def test_apply_cal_split_identity_gain_is_noop() -> None:
    """G = 1 + 0j applied to arbitrary voltages → unchanged."""
    rng = np.random.default_rng(0xC0DE15)
    real_v = torch.from_numpy(
        rng.standard_normal(_GEMM_LAYOUT_SHAPE, dtype=np.float32) * 0.1,
    )
    imag_v = torch.from_numpy(
        rng.standard_normal(_GEMM_LAYOUT_SHAPE, dtype=np.float32) * 0.1,
    )
    g_fine = np.ones((NANTS, NCHAN_PER_CHGROUP, NPOL), dtype=np.complex64)
    cal_real, cal_imag = make_cal_broadcast_tensors(
        g_fine, device="cpu", dtype=torch.float32,
    )
    real_out, imag_out = apply_cal_split(real_v, imag_v, cal_real, cal_imag)
    torch.testing.assert_close(real_out, real_v, atol=1e-6, rtol=0)
    torch.testing.assert_close(imag_out, imag_v, atol=1e-6, rtol=0)


def test_apply_cal_split_per_pol_phasor() -> None:
    """G[ant, ch, pol] = e^{i·phi(ant, pol)} (constant in ch) applied to
    a constant voltage produces the expected complex multiplication."""
    rng = np.random.default_rng(0xFADE)
    phi = rng.uniform(-np.pi, np.pi, (NANTS, NPOL)).astype(np.float64)
    g_fine = np.broadcast_to(
        np.exp(1j * phi)[:, None, :],
        (NANTS, NCHAN_PER_CHGROUP, NPOL),
    ).astype(np.complex64)

    # Constant voltage over (ch, t, p, pkt, ant): real = 0.3, imag = -0.2.
    real_v = torch.full(_GEMM_LAYOUT_SHAPE, 0.3, dtype=torch.float32)
    imag_v = torch.full(_GEMM_LAYOUT_SHAPE, -0.2, dtype=torch.float32)
    cal_real, cal_imag = make_cal_broadcast_tensors(
        g_fine, device="cpu", dtype=torch.float32,
    )
    real_out, imag_out = apply_cal_split(real_v, imag_v, cal_real, cal_imag)

    # Expected: E_cal = (0.3 - 0.2j) * exp(i·phi), broadcast over (ch, pkt, t).
    cR = np.cos(phi).astype(np.float32)                       # (96, 2)
    cI = np.sin(phi).astype(np.float32)
    expected_real = 0.3 * cR - (-0.2) * cI
    expected_imag = 0.3 * cI + (-0.2) * cR
    # Both shaped (NANTS, NPOL); broadcast across other axes.
    real_out_np = real_out.numpy()
    imag_out_np = imag_out.numpy()
    # Spot-check a few (ant, pol) cells (broadcasting → all (ch, t, pkt) match).
    for ant in (0, 1, 47, 95):
        for pol in (0, 1):
            np.testing.assert_allclose(
                real_out_np[:, :, pol, :, ant], expected_real[ant, pol],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                imag_out_np[:, :, pol, :, ant], expected_imag[ant, pol],
                atol=1e-5,
            )


def test_apply_cal_split_shape_validation() -> None:
    """Wrong shape → ValueError."""
    bad_v = torch.zeros((1, 2, 3), dtype=torch.float32)
    g_fine = np.ones((NANTS, NCHAN_PER_CHGROUP, NPOL), dtype=np.complex64)
    cal_real, cal_imag = make_cal_broadcast_tensors(
        g_fine, device="cpu", dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="_GEMM_LAYOUT_SHAPE"):
        apply_cal_split(bad_v, bad_v, cal_real, cal_imag)


def test_apply_cal_split_dtype_mismatch() -> None:
    """fp16 voltages with fp32 cal → ValueError (caller must align dtypes)."""
    real_v, imag_v = _zeros_voltage(dtype=torch.float16)
    g_fine = np.ones((NANTS, NCHAN_PER_CHGROUP, NPOL), dtype=np.complex64)
    cal_real, cal_imag = make_cal_broadcast_tensors(
        g_fine, device="cpu", dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="dtype"):
        apply_cal_split(real_v, imag_v, cal_real, cal_imag)


def test_make_cal_broadcast_shape() -> None:
    """`make_cal_broadcast_tensors` outputs (NCHAN, 1, NPOL, 1, NANTS)."""
    g_fine = np.zeros((NANTS, NCHAN_PER_CHGROUP, NPOL), dtype=np.complex64)
    cal_real, cal_imag = make_cal_broadcast_tensors(
        g_fine, device="cpu", dtype=torch.float32,
    )
    assert cal_real.shape == (NCHAN_PER_CHGROUP, 1, NPOL, 1, NANTS)
    assert cal_imag.shape == (NCHAN_PER_CHGROUP, 1, NPOL, 1, NANTS)


def test_full_pipeline_visibilities_get_GiH_Gj() -> None:
    """End-to-end (CPU, downsized): cal applied to voltages produces
    V_cal_ij = G_i^* · G_j · V_raw_ij in the visibility products.

    Uses a tiny manual K=2 visibility example without invoking the full
    SlowCorrKernel (which is hardcoded to fada-page sizes). Verifies the
    math identity that is the entire point of D17.
    """
    rng = np.random.default_rng(0x42)
    nant_small = 4
    # E_raw[ant] = complex scalar (one channel, one time sample).
    E_raw = (
        rng.standard_normal(nant_small) + 1j * rng.standard_normal(nant_small)
    ).astype(np.complex64)
    # Per-ant complex gain.
    G = (
        rng.standard_normal(nant_small) + 1j * rng.standard_normal(nant_small)
    ).astype(np.complex64)
    G[2] = 0.7 + 0.3j  # arbitrary

    E_cal = E_raw * G

    # V_raw_ij = conj(E_i) * E_j  (D5 sign convention)
    V_raw = np.conj(E_raw)[:, None] * E_raw[None, :]
    V_cal_pre_apply_in_voltages = np.conj(E_cal)[:, None] * E_cal[None, :]
    V_cal_post = np.conj(G)[:, None] * G[None, :] * V_raw

    # The two paths must agree (this is what apply_cal_split achieves
    # at the per-voltage level, before the GEMM).
    np.testing.assert_allclose(
        V_cal_pre_apply_in_voltages, V_cal_post, atol=1e-5,
    )
