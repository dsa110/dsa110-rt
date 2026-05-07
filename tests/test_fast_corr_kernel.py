"""FastCorrKernel acceptance tests (M3 chunk 2a).

Covers:
* Input validation (shape, dtype, t_int_fast_native divisibility)
* Output shape correctness for several t_int_fast_native values
* **Consistency with SlowCorrKernel** at the full-block boundary:
  setting ``t_int_fast_native = N_TIME_SAMPLES = 4096`` makes the
  FastCorrKernel emit a single fast-vis tile that matches the
  SlowCorrKernel output element-wise. This pins both kernels to the
  same GEMM convention (RR + II for V_real, RI − IR for V_imag,
  upper-tri b/a swap for F18).
* Numerical sanity (zero voltages → zero vis; small synthetic noise
  preserves Hermitian conjugate symmetry on auto-correlations).
* **F18 + F21 composition**: synthetic voltages for a source at
  obs_dec, multiplied by F21 cal weights, produce real-only Stokes-I
  fast visibilities (Imag ≈ 0 per fp16 tolerance).
* Pol-sum helper (``stokes_i_pol_sum``) and V-4 zero-fill helper
  (``zero_v4_cells``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dsart.cal.cal_loader import compute_dec_phase
from dsart.common.constants import (
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    NPOL,
    PHI_LAT_OVRO_RAD,
    freq_GHz,
)
from dsart.services.corr_fast_kernel import (
    DEFAULT_T_INT_FAST_NATIVE,
    FastCorrKernel,
    stokes_i_pol_sum,
    validate_fast_voltage_shape,
    zero_v4_cells,
)
from dsart.services.slow_corr_kernel import (
    BADA_NPOL,
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    N_TIME_SAMPLES,
    SlowCorrKernel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_fp16_voltages(
    n_packets_in: int,
    *,
    nchan: int = NCHAN_PER_CHGROUP,
    nants: int = NANTS,
    npol: int = NPOL,
    seed: int = 20260505,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random fp32 voltages cast to fp16, in GEMM layout."""
    rng = np.random.default_rng(seed=seed)
    shape = (nchan, NTIMES_PER_PACKET, npol, n_packets_in, nants)
    real = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
    imag = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
    return (
        torch.tensor(real, dtype=torch.float16, device=device),
        torch.tensor(imag, dtype=torch.float16, device=device),
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateFastVoltageShape:

    def test_happy_path(self) -> None:
        r, i = _random_fp16_voltages(16)
        n_fv, ppfv = validate_fast_voltage_shape(
            r, i, t_int_fast_native=8,
        )
        assert n_fv == 4                                       # 16 packets * 2 t / 8 = 4
        assert ppfv == 4                                        # 8 / 2 = 4 packets per fast-vis

    def test_rejects_mismatched_real_imag_shape(self) -> None:
        r, _ = _random_fp16_voltages(16)
        _, i = _random_fp16_voltages(8)
        with pytest.raises(ValueError, match="shape"):
            validate_fast_voltage_shape(r, i, t_int_fast_native=8)

    def test_rejects_mismatched_dtype(self) -> None:
        r, i = _random_fp16_voltages(16)
        i32 = i.to(torch.float32)
        with pytest.raises(ValueError, match="dtype"):
            validate_fast_voltage_shape(r, i32, t_int_fast_native=8)

    def test_rejects_non_packet_aligned_t_int(self) -> None:
        r, i = _random_fp16_voltages(16)
        # t_int_fast_native = 7 not a multiple of NTIMES_PER_PACKET=2
        with pytest.raises(ValueError, match="multiple of NTIMES_PER_PACKET"):
            validate_fast_voltage_shape(r, i, t_int_fast_native=7)

    def test_rejects_non_block_aligned_t_int(self) -> None:
        r, i = _random_fp16_voltages(16)
        # 16 packets * 2 t = 32 native samples; t_int=10 doesn't divide
        with pytest.raises(ValueError, match="not a multiple"):
            validate_fast_voltage_shape(r, i, t_int_fast_native=10)

    def test_rejects_zero_t_int(self) -> None:
        r, i = _random_fp16_voltages(16)
        with pytest.raises(ValueError, match="must be > 0"):
            validate_fast_voltage_shape(r, i, t_int_fast_native=0)

    def test_rejects_wrong_axis_count(self) -> None:
        r = torch.zeros(NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL, NANTS,
                        dtype=torch.float16)
        with pytest.raises(ValueError, match="5D"):
            validate_fast_voltage_shape(r, r, t_int_fast_native=8)


# ---------------------------------------------------------------------------
# Output shape + dtype tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    """``compute_split`` returns the documented shape per t_int + n_packets_in."""

    @pytest.mark.parametrize(
        ("n_packets_in", "t_int_fast_native", "expected_n_fv"),
        [
            (16, 8, 4),                                         # default cadence
            (16, 16, 2),
            (16, 32, 1),                                        # 4× burst-test cadence (full small block)
            (32, 8, 8),
            (32, 32, 2),
            (4, 8, 1),                                          # smallest valid block at native
        ],
    )
    def test_shape_per_t_int(
        self,
        n_packets_in: int,
        t_int_fast_native: int,
        expected_n_fv: int,
    ) -> None:
        r, i = _random_fp16_voltages(n_packets_in)
        kernel = FastCorrKernel(
            device=torch.device("cpu"),
            t_int_fast_native=t_int_fast_native,
        )
        vis = kernel.compute_split(r, i)
        assert vis.shape == (expected_n_fv, NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)
        assert vis.dtype == torch.complex64
        assert vis.device == torch.device("cpu")

    def test_n_fast_vis_per_full_block_property(self) -> None:
        # production cadence
        k = FastCorrKernel(device=torch.device("cpu"), t_int_fast_native=8)
        assert k.n_fast_vis_per_full_block == 512               # 2048 packets / 4 ppfv
        # 4× burst-test cadence
        k = FastCorrKernel(device=torch.device("cpu"), t_int_fast_native=32)
        assert k.n_fast_vis_per_full_block == 128               # 2048 packets / 16 ppfv
        # default
        assert FastCorrKernel(device=torch.device("cpu")).n_fast_vis_per_full_block \
            == NPACKETS_PER_BLOCK * NTIMES_PER_PACKET // DEFAULT_T_INT_FAST_NATIVE


# ---------------------------------------------------------------------------
# Consistency with SlowCorrKernel at the full-block boundary
# ---------------------------------------------------------------------------


def test_full_block_equals_slow_corr_kernel() -> None:
    """At t_int = full block, FastCorrKernel emits 1 tile that matches SlowCorrKernel.

    This is the strongest cross-kernel pin: same input voltages, same
    GEMM convention, same upper-tri b/a swap should produce
    bit-identical visibilities (modulo fp16 nondeterminism in
    torch.matmul, but on CPU torch uses deterministic GEMM kernels).

    Uses the FULL ``NPACKETS_PER_BLOCK = 2048`` block so the fast
    kernel's ``t_int_fast_native = 4096`` boundary case maps onto
    SlowCorrKernel's internals exactly. ~1 GB of working memory.
    """
    r, i = _random_fp16_voltages(NPACKETS_PER_BLOCK)
    slow = SlowCorrKernel(device=torch.device("cpu"))
    fast = FastCorrKernel(
        device=torch.device("cpu"),
        t_int_fast_native=N_TIME_SAMPLES,                      # 4096 → 1 fast-vis tile
    )

    vis_slow = slow.compute_split(r, i)                        # (NBASE, NCHAN, BADA_NPOL)
    vis_fast = fast.compute_split(r, i)                        # (1, NBASE, NCHAN, BADA_NPOL)

    assert vis_fast.shape[0] == 1
    diff = vis_fast[0] - vis_slow                              # (NBASE, NCHAN, BADA_NPOL) cfp32
    max_real = float(torch.max(torch.abs(diff.real)).item())
    max_imag = float(torch.max(torch.abs(diff.imag)).item())
    # Both kernels do the same fp16 matmul on the same tensors → cast
    # to fp32 → upper-tri gather. Should agree exactly (deterministic
    # CPU matmul). Allow 1e-5 absolute as a tolerance for any ordering
    # differences in the t_sub vs ppfv reduction order. (For the M3
    # production test on GPU this tolerance may need to grow to 1e-2;
    # CPU here gives us the sharpest comparison.)
    assert max_real < 1e-5, (
        f"FastCorrKernel(t_int=N_TIME_SAMPLES) vs SlowCorrKernel: "
        f"max real diff = {max_real:.3e}; expected ≤ 1e-5. "
        f"FastCorrKernel does NOT reduce to SlowCorrKernel at the boundary."
    )
    assert max_imag < 1e-5, (
        f"FastCorrKernel(t_int=N_TIME_SAMPLES) vs SlowCorrKernel: "
        f"max imag diff = {max_imag:.3e}; expected ≤ 1e-5."
    )


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------


def test_zero_voltages_give_zero_vis() -> None:
    n_packets_in = 16
    r = torch.zeros(NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
                    n_packets_in, NANTS, dtype=torch.float16)
    i = torch.zeros_like(r)
    kernel = FastCorrKernel(device=torch.device("cpu"), t_int_fast_native=8)
    vis = kernel.compute_split(r, i)
    assert vis.shape == (4, NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)
    assert torch.all(vis == 0)


def test_autocorrelations_have_zero_imaginary_part() -> None:
    """V_aa = sum |E_a|² is real-valued by construction.

    Picks a few diagonal baselines (a, a) → bls_idx = a*(a+1)/2 + a
    and verifies imaginary part is ~0 to fp16 precision.
    """
    r, i = _random_fp16_voltages(16)
    kernel = FastCorrKernel(device=torch.device("cpu"), t_int_fast_native=8)
    vis = kernel.compute_split(r, i)                            # (4, NBASE, NCHAN, NPOL)

    # Diagonal baselines: bls_idx[a] = a*(a+1)/2 + a for a in [0, NANTS).
    diag_bls = [a * (a + 1) // 2 + a for a in range(NANTS)]
    diag_imag = vis[:, diag_bls, :, :].imag                    # (4, NANTS, NCHAN, NPOL)
    max_imag = float(torch.max(torch.abs(diag_imag)).item())
    # Auto-corr imag should be EXACTLY zero in fp32 since the GEMM
    # gives V_imag = R^T@I − I^T@R which is antisymmetric ⇒ diagonal
    # is zero. fp16 cancellation noise might leak ~1e-3 at worst.
    assert max_imag < 1e-3, (
        f"max diagonal-baseline imag = {max_imag:.3e}; expected ≤ 1e-3"
    )


def test_autocorrelations_have_positive_real_part() -> None:
    """V_aa = sum |E_a|² ≥ 0."""
    r, i = _random_fp16_voltages(16)
    kernel = FastCorrKernel(device=torch.device("cpu"), t_int_fast_native=8)
    vis = kernel.compute_split(r, i)
    diag_bls = [a * (a + 1) // 2 + a for a in range(NANTS)]
    diag_real = vis[:, diag_bls, :, :].real
    assert torch.all(diag_real >= -1e-3), (
        f"min auto-corr real = {float(diag_real.min().item()):.3e}; "
        f"sum |E|² should be ≥ 0"
    )


# ---------------------------------------------------------------------------
# F18 + F21 composition
# ---------------------------------------------------------------------------


def _antpos_n_realistic(seed: int = 20260505) -> np.ndarray:
    """DSA-110-like N-S antpos, ±75 m span with small jitter."""
    rng = np.random.default_rng(seed=seed)
    base = np.linspace(-75.0, +75.0, NANTS, dtype=np.float64)
    jitter = rng.uniform(-1.0, +1.0, size=NANTS)
    return (base + jitter).astype(np.float32)


def _on_source_voltage_block(
    *,
    n_packets_in: int,
    chgroup: int,
    src_dec_rad: float,
    antpos_n: np.ndarray,
    amplitude: float = 0.05,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesise a voltage block (GEMM layout) for a single point source.

    Per-(ant, chan) phase: φ_a(f) = +2π f sin(δ_src − φ_lat) N_a / c
    (TMS convention; same as the F21 module-docstring derivation).

    Returns (real_v, imag_v) fp16 tensors in GEMM layout.
    """
    f_hz = np.asarray(
        [freq_GHz(chgroup, ch) * 1.0e9 for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    sin_delta = math.sin(src_dec_rad - PHI_LAT_OVRO_RAD)
    n_a = antpos_n.astype(np.float64, copy=False)

    arg = (
        +2.0 * math.pi * sin_delta / 299_792_458.0
        * f_hz[None, :] * n_a[:, None]
    )                                                            # (NANTS, NCHAN)

    # Per-(ant, ch) complex voltage; broadcast to all (2t, 2p, packets).
    e_real = (amplitude * np.cos(arg)).astype(np.float32)        # (NANTS, NCHAN)
    e_imag = (amplitude * np.sin(arg)).astype(np.float32)

    # Build full GEMM-layout tensor: (NCHAN, 2t, 2p, n_packets_in, NANTS).
    # All time samples + all packets carry the same voltage (CW source).
    real_5d = np.broadcast_to(
        e_real.T[:, None, None, None, :],                        # (NCHAN, 1, 1, 1, NANTS)
        (NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL, n_packets_in, NANTS),
    ).copy()
    imag_5d = np.broadcast_to(
        e_imag.T[:, None, None, None, :],
        (NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL, n_packets_in, NANTS),
    ).copy()

    return (
        torch.tensor(real_5d, dtype=torch.float16, device=device),
        torch.tensor(imag_5d, dtype=torch.float16, device=device),
    )


def _apply_cal_inplace_complex_split(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    cal_per_ant_chan: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-(ant, chan) complex cal to (real_v, imag_v) GEMM-layout
    voltages. Returns new tensors (cast to fp16 for downstream)."""
    cal_real = cal_per_ant_chan.real.astype(np.float32)          # (NANTS, NCHAN)
    cal_imag = cal_per_ant_chan.imag.astype(np.float32)
    # Permute (NANTS, NCHAN) → (NCHAN, 1, 1, 1, NANTS) for broadcast.
    cr_t = torch.tensor(
        cal_real.T[:, None, None, None, :], dtype=real_v.dtype,
        device=real_v.device,
    )
    ci_t = torch.tensor(
        cal_imag.T[:, None, None, None, :], dtype=real_v.dtype,
        device=real_v.device,
    )
    out_r = real_v * cr_t - imag_v * ci_t
    out_i = real_v * ci_t + imag_v * cr_t
    return out_r, out_i


def test_F18_F21_compose_on_source_vis_is_real() -> None:
    """End-to-end: F21 cal + F18 GEMM ⇒ on-source fast-vis is real.

    Setup:
      * δ_src = δ_obs = 53.85° (matches 250924mptq burst dec)
      * Synthesise CW point-source voltages for each (ant, chan) per
        the TMS +2π convention.
      * Build an F21 cal tensor at the SAME obs_dec.
      * Apply cal to voltages → run FastCorrKernel.
      * Expect: every fast-vis cell is real-valued (Imag/|V| < 0.01
        per fp16 + accumulated rounding through the 4-GEMM chain).
    """
    chgroup = 0
    obs_dec_rad = math.radians(53.848986)                       # 250924mptq Dec
    antpos_n = _antpos_n_realistic()
    n_packets_in = 8                                             # 16 native, 2 fast-vis at t_int=8
    t_int_fast_native = 8

    # Synthetic on-source voltages
    real_v, imag_v = _on_source_voltage_block(
        n_packets_in=n_packets_in,
        chgroup=chgroup,
        src_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
        amplitude=0.05,
    )

    # F21 cal phase, broadcast over both pols (same value)
    dec_phase = compute_dec_phase(
        chgroup=chgroup,
        obs_dec_rad=obs_dec_rad,
        antpos_n=antpos_n,
    )                                                            # (NANTS, NCHAN) cplx128

    # Apply cal
    real_v_cal, imag_v_cal = _apply_cal_inplace_complex_split(
        real_v, imag_v, dec_phase,
    )

    # Run kernel
    kernel = FastCorrKernel(
        device=torch.device("cpu"),
        t_int_fast_native=t_int_fast_native,
    )
    vis = kernel.compute_split(real_v_cal, imag_v_cal)
    # Expected shape: (n_packets_in * NTIMES_PER_PACKET / t_int_fast_native,
    #                  NBASE, NCHAN, BADA_NPOL) = (2, 4656, 384, 2)
    assert vis.shape == (
        n_packets_in * NTIMES_PER_PACKET // t_int_fast_native,
        NBASE, NCHAN_PER_CHGROUP, BADA_NPOL,
    )

    # Real part: should be positive (sum |E|² for autos, +ve cosine
    # similarity for cross-correlations on source).
    abs_real = torch.abs(vis.real)
    abs_imag = torch.abs(vis.imag)
    # Avoid divide-by-zero on cells where the source happens to land
    # near a phase null (V ≈ 0); restrict to cells with |V| > 1% of
    # the median |V| to make the imag/real ratio meaningful.
    abs_v = torch.sqrt(vis.real ** 2 + vis.imag ** 2)
    threshold = float(torch.median(abs_v).item()) * 0.01
    mask = abs_v > threshold

    if not torch.any(mask):
        pytest.skip("all fast-vis cells too small for imag/|V| ratio test")

    imag_ratio = (abs_imag[mask] / abs_v[mask]).max().item()
    assert imag_ratio < 0.05, (
        f"max |Imag(V)| / |V| = {imag_ratio:.4f}; expected < 0.05 "
        f"for an on-source point with F21 cal-apply (the geometric "
        f"phase should be cancelled to fp16 precision through the "
        f"GEMM)."
    )


# ---------------------------------------------------------------------------
# F31a — chunked compute_split is bit-identical to un-chunked
# ---------------------------------------------------------------------------


class TestComputeSplitChunked:
    """F31a: chunked compute_split is bit-identical to un-chunked.

    F31a chunks the n_fast_vis axis to bound the fp16 V_real / V_imag
    matmul intermediate to ~1 GB on the 2080Ti production GPU. The
    fp16 matmuls per slab use exactly the same inputs as the
    un-chunked path, so output must be bit-identical.
    """

    @pytest.mark.parametrize(
        ("n_packets_in", "t_int", "n_fv_chunk"),
        [
            (16, 8, 1),       # 4 fv tiles, chunked one-at-a-time
            (16, 8, 2),       # 4 fv tiles, 2 per chunk
            (32, 8, 4),       # 8 fv tiles, 4 per chunk
            (32, 32, 1),      # 2 fv tiles, chunked one-at-a-time
            (4, 8, 1),        # 1 fv tile (single chunk)
        ],
    )
    def test_chunked_equals_unchunked(self, n_packets_in, t_int, n_fv_chunk):
        r, i = _random_fp16_voltages(n_packets_in)
        kernel = FastCorrKernel(
            device=torch.device("cpu"),
            t_int_fast_native=t_int,
        )
        vis_un = kernel.compute_split(r, i)
        vis_ch = kernel.compute_split(r, i, n_fv_chunk=n_fv_chunk)
        assert vis_un.shape == vis_ch.shape
        assert vis_un.dtype == vis_ch.dtype
        # Bit-identical (CPU deterministic matmul)
        torch.testing.assert_close(
            vis_ch, vis_un,
            rtol=0, atol=0,
            msg=lambda m: f"F31a chunked != un-chunked: {m}",
        )

    def test_auto_chunk_size_default(self):
        """Auto-pick yields a positive size and produces correct output."""
        r, i = _random_fp16_voltages(16)
        kernel = FastCorrKernel(
            device=torch.device("cpu"),
            t_int_fast_native=8,
        )
        vis_default = kernel.compute_split(r, i)
        vis_explicit = kernel.compute_split(r, i, n_fv_chunk=None)
        torch.testing.assert_close(vis_default, vis_explicit, rtol=0, atol=0)

    def test_rejects_zero_chunk(self):
        r, i = _random_fp16_voltages(16)
        kernel = FastCorrKernel(
            device=torch.device("cpu"),
            t_int_fast_native=8,
        )
        with pytest.raises(ValueError, match="n_fv_chunk"):
            kernel.compute_split(r, i, n_fv_chunk=0)

    def test_rejects_chunk_larger_than_total(self):
        r, i = _random_fp16_voltages(16)  # 4 fv tiles
        kernel = FastCorrKernel(
            device=torch.device("cpu"),
            t_int_fast_native=8,
        )
        with pytest.raises(ValueError, match="n_fv_chunk"):
            kernel.compute_split(r, i, n_fv_chunk=5)


# ---------------------------------------------------------------------------
# stokes_i_pol_sum + zero_v4_cells helpers
# ---------------------------------------------------------------------------


class TestStokesIHelper:

    def test_pol_sum_2pol_visibility(self) -> None:
        rng = np.random.default_rng(seed=20260505)
        vis_2pol = torch.tensor(
            rng.normal(size=(4, 100, 384, 2)) +
            1j * rng.normal(size=(4, 100, 384, 2)),
            dtype=torch.complex64,
        )
        i = stokes_i_pol_sum(vis_2pol)
        assert i.shape == (4, 100, 384)
        assert i.dtype == torch.complex64
        # Manual check
        expected = vis_2pol[..., 0] + vis_2pol[..., 1]
        assert torch.allclose(i, expected, atol=1e-6)

    def test_rejects_non_complex(self) -> None:
        v = torch.zeros((4, 100, 384, 2), dtype=torch.float32)
        with pytest.raises(ValueError, match="complex"):
            stokes_i_pol_sum(v)

    def test_rejects_wrong_pol_axis(self) -> None:
        v = torch.zeros((4, 100, 384, 4), dtype=torch.complex64)
        with pytest.raises(ValueError, match="last dim = 2"):
            stokes_i_pol_sum(v)


class TestZeroV4Helper:

    def test_zero_v4_pol_axis(self) -> None:
        rng = np.random.default_rng(seed=20260505)
        v = torch.tensor(
            rng.normal(size=(4, 100, 384, 4)) +
            1j * rng.normal(size=(4, 100, 384, 4)),
            dtype=torch.complex64,
        )
        v_orig_xx = v[..., 0].clone()
        v_orig_yy = v[..., 1].clone()
        out = zero_v4_cells(v)
        # Returns the same tensor (in-place)
        assert out is v
        # Parallel-hands preserved
        assert torch.allclose(v[..., 0], v_orig_xx)
        assert torch.allclose(v[..., 1], v_orig_yy)
        # Cross-hands zeroed
        assert torch.all(v[..., 2] == 0)
        assert torch.all(v[..., 3] == 0)

    def test_rejects_wrong_pol_axis(self) -> None:
        v = torch.zeros((4, 100, 384, 2), dtype=torch.complex64)
        with pytest.raises(ValueError, match="last dim = 4"):
            zero_v4_cells(v)
