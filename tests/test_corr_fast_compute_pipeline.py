"""corr_fast_compute spine-pipeline smoke tests (M3 chunk 2b).

Exercises :func:`compute_block` (the ``unpack_int4_split → cal-apply
with F21 → FastCorrKernel → stokes_i_pol_sum`` chain) without PSRDADA,
using synthetic raw-byte fada pages. Validates wiring + numerical
sanity. The full PSRDADA loop in :func:`run` is exercised on h01 in
chunk 5 (voltage fixture replay) where real fada pages are present.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from dsart.cal.cal_loader import (
    CalMode,
    load_cal_with_dec_phase,
)
from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    PHI_LAT_OVRO_RAD,
)
from dsart.services.corr_fast_compute import (
    _build_cal_tensors_with_f21,
    compute_block,
)
from dsart.services.corr_fast_kernel import FastCorrKernel
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _synthetic_fada_block(seed: int = 20260505) -> np.ndarray:
    """Generate a deterministic synthetic fada block (raw int4 bytes).

    Each byte carries two 4-bit signed components (real + imag) per
    M2's int4-fluff convention. Random uniform fill — no specific
    source structure, just non-zero data to exercise the GEMM.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, size=FADA_BYTES_PER_BLOCK,
                        dtype=np.uint8)


def _h01_real_cal_blob_path() -> str | None:
    candidate = "/home/ubuntu/data/voltages/0319/cals/beamformer_weights_sb00_0319+415.dat"
    return candidate if Path(candidate).is_file() else None


# ---------------------------------------------------------------------------
# compute_block — happy path
# ---------------------------------------------------------------------------


def test_compute_block_no_cal_t_int_full_block_returns_one_tile() -> None:
    """t_int = full block → 1 fast-vis tile, shape matches SlowCorrKernel pol-sum."""
    raw = _synthetic_fada_block()
    device = torch.device("cpu")
    kernel = FastCorrKernel(
        device=device,
        t_int_fast_native=NPACKETS_PER_BLOCK * NTIMES_PER_PACKET,    # = 4096 → 1 tile
    )
    vis = compute_block(raw, kernel=kernel, cal=None, voltage_dtype=torch.float16)
    # Output shape: (n_fast_vis=1, NBASE, NCHAN); pol-summed (no pol axis).
    assert vis.shape == (1, NBASE, NCHAN_PER_CHGROUP)
    assert vis.dtype == torch.complex64
    assert torch.all(torch.isfinite(vis.real))
    assert torch.all(torch.isfinite(vis.imag))
    # Auto-correlations on the diagonal (bls_idx[a] = a*(a+1)/2 + a)
    # carry sum |E_xx|² + sum |E_yy|² ≥ 0 — both pols add, so even
    # safer that real ≥ 0.
    diag_bls = [a * (a + 1) // 2 + a for a in range(NANTS)]
    assert torch.all(vis[0, diag_bls, :].real >= -1e-3), (
        "Stokes I autocorrelations should be ≥ 0"
    )


@pytest.mark.parametrize(
    "t_int_fast_native, expected_n_fast_vis_at_full_block",
    [
        (8, 512),                                                    # production cadence
        (16, 256),
        (32, 128),                                                   # 4× burst-test cadence
        (4096, 1),                                                   # full-block boundary
    ],
)
def test_compute_block_returns_expected_n_fast_vis(
    t_int_fast_native: int,
    expected_n_fast_vis_at_full_block: int,
) -> None:
    raw = _synthetic_fada_block()
    kernel = FastCorrKernel(
        device=torch.device("cpu"),
        t_int_fast_native=t_int_fast_native,
    )
    vis = compute_block(raw, kernel=kernel, cal=None, voltage_dtype=torch.float16)
    assert vis.shape == (expected_n_fast_vis_at_full_block, NBASE, NCHAN_PER_CHGROUP)


# ---------------------------------------------------------------------------
# compute_block — with cal (F21 DEC-phase folded in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _h01_real_cal_blob_path() is None,
    reason="no real cal blob available; run on h01 to exercise cal path",
)
def test_compute_block_with_cal_reduces_outrigger_baselines_to_zero() -> None:
    """With phase-only cal, the outrigger antennas (zero-cal cells) produce
    zero-valued baselines via F18 + apply_cal_split semantics.

    Specifically: any (a, b) baseline where cal[a, ch, pol] = 0 OR
    cal[b, ch, pol] = 0 has the corresponding voltage zeroed by the
    cal multiply, so the post-GEMM visibility cell is zero. This
    validates that the cal-apply layer correctly implements the
    bandpass-zero ⇒ baseline-zero contract.
    """
    cal_path = _h01_real_cal_blob_path()
    assert cal_path is not None
    raw = _synthetic_fada_block()
    device = torch.device("cpu")

    cal = _build_cal_tensors_with_f21(
        Path(cal_path),
        chgroup=0,
        obs_dec_rad=math.radians(41.5117),                           # 0319+415
        cal_mode=CalMode.PHASE_ONLY,
        cal_pol_swap=False,
        device=device,
        dtype=torch.float32,
    )
    kernel = FastCorrKernel(
        device=device,
        t_int_fast_native=NPACKETS_PER_BLOCK * NTIMES_PER_PACKET,
    )
    vis = compute_block(raw, kernel=kernel, cal=cal,
                        voltage_dtype=torch.float32)

    # Identify (a, b) baselines where cal is zero on at least one antenna.
    # cal.cal_real shape: (NCHAN, 1, NPOL, 1, NANTS). Take ch=0, pol=0.
    cal_zero_per_ant = (
        torch.abs(cal.cal_real[0, 0, 0, 0, :]) +
        torch.abs(cal.cal_imag[0, 0, 0, 0, :])
    ) == 0
    cal_zero_per_ant_np = cal_zero_per_ant.numpy()
    n_zero_ants = int(cal_zero_per_ant_np.sum())
    if n_zero_ants == 0:
        pytest.skip("real cal blob has no zero-cal antennas at ch=0; "
                    "test setup not exercised")

    # For baselines (a, b) with a OR b cal-zero, |vis| should be tiny
    # (limited by fp16 noise, ~< 1e-2 of the median |vis|).
    n_zero_baselines_checked = 0
    for a in range(NANTS):
        for b in range(a + 1):                                       # b ≤ a (xGPU lower-tri)
            if cal_zero_per_ant_np[a] or cal_zero_per_ant_np[b]:
                bls_idx = a * (a + 1) // 2 + b
                v = abs(complex(vis[0, bls_idx, 0]))
                assert v < 1e-3, (
                    f"baseline ({a},{b}) cal-zero on at least one ant; "
                    f"|vis|={v:.3e} (expected ≤ 1e-3 for cal-zero ⇒ "
                    f"baseline-zero contract)"
                )
                n_zero_baselines_checked += 1
    assert n_zero_baselines_checked > 0


# ---------------------------------------------------------------------------
# Wiring sanity: kernel + cal device pinning + dtype
# ---------------------------------------------------------------------------


def test_compute_block_device_pinning_propagates() -> None:
    """All intermediate tensors land on the kernel's device."""
    raw = _synthetic_fada_block()
    device = torch.device("cpu")
    kernel = FastCorrKernel(
        device=device,
        t_int_fast_native=NPACKETS_PER_BLOCK * NTIMES_PER_PACKET,
    )
    vis = compute_block(raw, kernel=kernel, cal=None, voltage_dtype=torch.float16)
    assert vis.device == device


def test_compute_block_handles_zero_voltages() -> None:
    """All-zero raw input → all-zero fast vis."""
    raw = np.zeros(FADA_BYTES_PER_BLOCK, dtype=np.uint8)
    kernel = FastCorrKernel(
        device=torch.device("cpu"),
        t_int_fast_native=NPACKETS_PER_BLOCK * NTIMES_PER_PACKET,
    )
    vis = compute_block(raw, kernel=kernel, cal=None, voltage_dtype=torch.float16)
    assert vis.shape == (1, NBASE, NCHAN_PER_CHGROUP)
    assert torch.all(vis == 0)
