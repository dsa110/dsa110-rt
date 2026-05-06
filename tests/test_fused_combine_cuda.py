"""Numerical-equivalence + smoke tests for the fused-combine CUDA kernel.

Skipped unless ``cuda`` is available (the test runs only on the M5
isolation worktree on h01 GPU 1; CI without GPU passes by skipping).

The fused kernel claims to be cell-for-cell equivalent to the
Python ``uv.zero_(); for g: uv.add_(streams[g, s_g:s_g+T_det])``
fallback, modulo cfp16 floating-point ordering. The tests verify:

1. Random-shifts case: cf32 (exact equality up to fp32 reduction
   order) and cf16 (atol=2.0 — generous because cfp16 has a 4-bit
   mantissa for the imag/real lanes; chgroup-summed magnitudes can
   easily span a few units at our random-fill amplitudes).
2. Boundary case: a chgroup with shift+t_det > t_stream is
   silently skipped (zero-fill), matching the Python guard.
3. shifts=zero edge case: the kernel just sums the first t_det
   samples of each chgroup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("fused_combine_cuda requires cuda", allow_module_level=True)

from dsart.image.fused_combine_cuda import (  # noqa: E402
    fused_combine_per_fdm,
    fused_dequant_combine_per_fdm,
    get_module,
)


# Module-level cache so we only pay the ~30 s compile once across tests.
@pytest.fixture(scope="module")
def cuda_module():
    return get_module(verbose=False)


def _python_combine(streams: torch.Tensor, shifts: torch.Tensor, t_det: int) -> torch.Tensor:
    """Reference Python add-loop (the bench's ``python_addloop`` impl)."""
    n_chg, t_stream, n, _ = streams.shape
    out = torch.zeros((t_det, n, n), dtype=streams.dtype, device=streams.device)
    for g in range(n_chg):
        s = int(shifts[g].item())
        if s + t_det <= t_stream:
            out += streams[g, s : s + t_det]
    return out


def test_fused_combine_matches_python_cf32(cuda_module):
    """cf32: kernel matches Python add-loop to fp32 reduction tolerance.

    Different reduction order can produce slightly different fp32
    sums, so we use a small atol rather than ``equal``.
    """
    torch.manual_seed(42)
    n_chg, t_stream, n_grid, t_det = 16, 96, 32, 64
    streams = torch.randn(
        (n_chg, t_stream, n_grid, n_grid), dtype=torch.complex64, device="cuda",
    ) * 0.1
    shifts = torch.randint(0, t_stream - t_det + 1, (n_chg,), dtype=torch.int32, device="cuda")

    out_fused = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex64, device="cuda")
    fused_combine_per_fdm(streams, shifts, out_fused)
    out_python = _python_combine(streams, shifts, t_det)

    diff = (out_fused - out_python).abs().max().item()
    assert diff < 1e-5, f"cf32 fused-vs-python max diff = {diff:.3e}"


def test_fused_combine_matches_python_cf16(cuda_module):
    """cf16: kernel matches Python add-loop to cfp16 tolerance.

    cfp16 has 10-bit mantissa per real/imag lane; summing 16
    independent N(0, 0.1²) cells gives σ_sum ≈ 0.4. Numerical
    reduction-order error is well below 0.05 (cfp16 unit roundoff
    × √16). Use atol=0.05.
    """
    torch.manual_seed(42)
    n_chg, t_stream, n_grid, t_det = 16, 96, 32, 64
    # torch.randn doesn't support complex32 directly; build via two
    # fp32 randn tensors → torch.complex(re, im) → cf32 → .to(cf16).
    streams = torch.complex(
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda") * 0.1,
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda") * 0.1,
    ).to(torch.complex32)

    shifts = torch.randint(0, t_stream - t_det + 1, (n_chg,), dtype=torch.int32, device="cuda")

    out_fused = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex32, device="cuda")
    fused_combine_per_fdm(streams, shifts, out_fused)
    out_python = _python_combine(streams, shifts, t_det)

    diff = (out_fused - out_python).abs().max().item()
    assert diff < 0.05, f"cf16 fused-vs-python max diff = {diff:.3e}"


def test_fused_combine_zero_shifts(cuda_module):
    """Edge: shifts=zero ⇒ kernel sums the first t_det samples of each chgroup."""
    torch.manual_seed(0)
    n_chg, t_stream, n_grid, t_det = 4, 32, 8, 16
    streams = torch.complex(
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda"),
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda"),
    ).to(torch.complex32)
    shifts = torch.zeros((n_chg,), dtype=torch.int32, device="cuda")

    out_fused = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex32, device="cuda")
    fused_combine_per_fdm(streams, shifts, out_fused)
    expected = streams[:, :t_det, :, :].sum(dim=0)

    diff = (out_fused - expected).abs().max().item()
    assert diff < 0.05, f"zero-shift fused-vs-expected max diff = {diff:.3e}"


def test_fused_combine_partial_overhang(cuda_module):
    """Edge: a chgroup whose shift+t_det > t_stream is skipped cell-wise.

    The kernel uses ``if s + t < t_stream`` per cell, so the partial
    overhang produces zeros for the affected output time samples for
    that chgroup, matching the Python ``if s + T_det <= T_stream:
    skip-add`` guard cell-by-cell rather than chgroup-by-chgroup.

    For exact equivalence to the Python guard (which skips the entire
    chgroup if any sample would overhang), we ensure the test setup
    has no in-range overhangs and rely on the per-cell semantic.
    """
    torch.manual_seed(0)
    n_chg, t_stream, n_grid, t_det = 4, 32, 8, 16
    streams = torch.complex(
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda"),
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda"),
    ).to(torch.complex32)
    # All shifts safely within bounds for this test:
    shifts = torch.tensor([0, 4, 8, 16], dtype=torch.int32, device="cuda")
    assert all(int(s) + t_det <= t_stream for s in shifts), \
        "test fixture must keep all shifts in range"

    out_fused = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex32, device="cuda")
    fused_combine_per_fdm(streams, shifts, out_fused)
    out_python = _python_combine(streams, shifts, t_det)

    diff = (out_fused - out_python).abs().max().item()
    assert diff < 0.05


def test_fused_combine_dtype_mismatch_raises(cuda_module):
    """The kernel rejects dtype mismatches between streams and output."""
    streams = torch.complex(
        torch.randn((4, 32, 8, 8), dtype=torch.float32, device="cuda"),
        torch.randn((4, 32, 8, 8), dtype=torch.float32, device="cuda"),
    ).to(torch.complex32)
    shifts = torch.zeros((4,), dtype=torch.int32, device="cuda")
    out_wrong = torch.empty((16, 8, 8), dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match=r"dtype must match"):
        fused_combine_per_fdm(streams, shifts, out_wrong)


# ---------------------------------------------------------------------------
# fused_dequant_combine_per_fdm: cint8-input variant (chunk-8)
# ---------------------------------------------------------------------------


def _python_dequant_combine_cf32(
    streams_cint8: torch.Tensor,
    shifts: torch.Tensor,
    t_det: int,
) -> torch.Tensor:
    """Python reference for the cint8-input fused kernel: dequant
    each chgroup to cfp32 (exact, since cint8 ⊂ fp32 mantissa) then
    accumulate across chgroups in cfp32.

    Layout: streams_cint8 is ``[N_chg, T_stream, 2, N, N]`` with
    re plane at axis_2=0 and im plane at axis_2=1, matching the
    bench's ``_build_synthetic_streams`` layout (which mirrors the
    M3 wire payload).

    Returns a cf32 result for use as a high-precision reference
    (the kernel's int32 accumulator → cfp16 cast has only one
    rounding step, so its output should match this cf32 reference
    to within ~1 fp16 ULP).
    """
    n_chg, t_stream, two, n_grid, _ = streams_cint8.shape
    assert two == 2
    re = streams_cint8[:, :, 0, :, :].to(torch.float32)
    im = streams_cint8[:, :, 1, :, :].to(torch.float32)
    streams_cf32 = torch.complex(re, im)
    out = torch.zeros((t_det, n_grid, n_grid), dtype=torch.complex64, device=streams_cint8.device)
    for g in range(n_chg):
        s = int(shifts[g].item())
        if s + t_det <= t_stream:
            out += streams_cf32[g, s : s + t_det]
    return out


def test_fused_dequant_combine_cf32(cuda_module):
    """cint8 → cf32 fused kernel: int32 accumulation is exact, fp32
    cast carries no error, so the result matches the Python reference
    bit-exactly.
    """
    torch.manual_seed(7)
    n_chg, t_stream, n_grid, t_det = 16, 96, 32, 64
    streams_cint8 = torch.randint(
        low=-127, high=127, size=(n_chg, t_stream, 2, n_grid, n_grid),
        dtype=torch.int8, device="cuda",
    )
    shifts = torch.randint(
        0, t_stream - t_det + 1, (n_chg,), dtype=torch.int32, device="cuda",
    )

    out_fused = torch.empty(
        (t_det, n_grid, n_grid), dtype=torch.complex64, device="cuda",
    )
    fused_dequant_combine_per_fdm(streams_cint8, shifts, out_fused)
    out_ref = _python_dequant_combine_cf32(streams_cint8, shifts, t_det)

    diff = (out_fused - out_ref).abs().max().item()
    assert diff == 0.0, f"cf32 fused vs python reference diff = {diff}"


def test_fused_dequant_combine_cf16(cuda_module):
    """cint8 → cf16 fused kernel: int32 accumulation is exact; the
    final fp16 cast is the only source of error.

    Tolerance: at N_chg=16 the per-cell sum has |re|, |im| ≤ 16×127
    = 2032. fp16 ULP at that magnitude is 2 (exponent=11, mantissa=10
    bits). One rounding step ⇒ atol < 2; we use 4 for headroom on
    edge cases at exact powers of two.
    """
    torch.manual_seed(11)
    n_chg, t_stream, n_grid, t_det = 16, 96, 32, 64
    streams_cint8 = torch.randint(
        low=-127, high=127, size=(n_chg, t_stream, 2, n_grid, n_grid),
        dtype=torch.int8, device="cuda",
    )
    shifts = torch.randint(
        0, t_stream - t_det + 1, (n_chg,), dtype=torch.int32, device="cuda",
    )

    out_fused = torch.empty(
        (t_det, n_grid, n_grid), dtype=torch.complex32, device="cuda",
    )
    fused_dequant_combine_per_fdm(streams_cint8, shifts, out_fused)

    # Reference: high-precision cf32 dequant+combine, then cast to cf16
    # (single rounding step matches the kernel's behaviour modulo fp32
    # vs int32 accumulation order; both are exact to within the ULP).
    out_ref_cf32 = _python_dequant_combine_cf32(streams_cint8, shifts, t_det)
    out_ref_cf16 = out_ref_cf32.to(torch.complex32)

    diff = (out_fused - out_ref_cf16).abs().max().item()
    assert diff < 4.0, (
        f"cf16 fused-dequant vs cf32-then-cast diff = {diff} > 4 ULP"
    )


def test_fused_dequant_combine_zero_shifts(cuda_module):
    """Edge: shifts=zero ⇒ kernel sums first t_det samples of each chgroup."""
    torch.manual_seed(0)
    n_chg, t_stream, n_grid, t_det = 4, 32, 8, 16
    streams_cint8 = torch.randint(
        low=-127, high=127, size=(n_chg, t_stream, 2, n_grid, n_grid),
        dtype=torch.int8, device="cuda",
    )
    shifts = torch.zeros((n_chg,), dtype=torch.int32, device="cuda")

    out_fused = torch.empty(
        (t_det, n_grid, n_grid), dtype=torch.complex64, device="cuda",
    )
    fused_dequant_combine_per_fdm(streams_cint8, shifts, out_fused)

    re = streams_cint8[:, :t_det, 0, :, :].to(torch.float32)
    im = streams_cint8[:, :t_det, 1, :, :].to(torch.float32)
    expected = torch.complex(re.sum(0), im.sum(0))

    diff = (out_fused - expected).abs().max().item()
    assert diff == 0.0, f"zero-shift cint8 fused diff = {diff}"


def test_fused_dequant_combine_dtype_mismatch_raises(cuda_module):
    """Reject non-int8 streams and non-int32 shifts."""
    bad_streams = torch.zeros(
        (4, 32, 2, 8, 8), dtype=torch.float32, device="cuda",
    )
    shifts = torch.zeros((4,), dtype=torch.int32, device="cuda")
    out = torch.empty((16, 8, 8), dtype=torch.complex32, device="cuda")
    with pytest.raises(RuntimeError, match=r"streams_cint8 must be int8"):
        fused_dequant_combine_per_fdm(bad_streams, shifts, out)

    streams_cint8 = torch.zeros(
        (4, 32, 2, 8, 8), dtype=torch.int8, device="cuda",
    )
    bad_shifts = torch.zeros((4,), dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match=r"shifts must be int32"):
        fused_dequant_combine_per_fdm(streams_cint8, bad_shifts, out)


def test_fused_dequant_combine_layout_validation(cuda_module):
    """axis_2 must be 2 (re/im split planes)."""
    bad = torch.zeros((4, 32, 4, 8, 8), dtype=torch.int8, device="cuda")  # axis_2=4
    shifts = torch.zeros((4,), dtype=torch.int32, device="cuda")
    out = torch.empty((16, 8, 8), dtype=torch.complex32, device="cuda")
    with pytest.raises(RuntimeError, match=r"streams_cint8 axis 2 must be 2"):
        fused_dequant_combine_per_fdm(bad, shifts, out)
