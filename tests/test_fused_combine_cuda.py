"""Numerical-equivalence + smoke tests for the fused-combine CUDA kernel.

Skipped unless ``cuda`` is available (the test runs only on the M5
isolation worktree on h01 GPU 1; CI without GPU passes by skipping).

The fused kernel implements the §3.6.3 sign convention:

    ``output[t] = sum_g streams[g, t - shifts[g]]``

with cell-wise zero-fill for out-of-range ``t - shifts[g]``. This is
identical to ``fine_dm/combiner.py::combine_chgroups`` (the CPU
reference). The tests verify:

1. Random-shifts case: cf32 (exact equality up to fp32 reduction
   order) and cf16 (atol < 1 ULP × √16).
2. Boundary case: cube-time samples ``t < shifts[g]`` get a zero
   contribution from chgroup ``g`` (matches combine_chgroups'
   zero-fill on samples not yet present in the stream).
3. shifts=zero edge case: the kernel just sums the first t_det
   samples of each chgroup.
4. ``test_fused_combine_matches_combine_chgroups`` (D25): a synthetic
   dispersed pulse + ``compute_time_shift_search`` table gives a
   coherent dedispersion peak at the chgroup-15 burst time —
   verifying GPU and CPU agree on §3.6.3's sign.
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
    """Reference Python add-loop matching ``combine_chgroups`` and §3.6.3:

        ``out[t] = sum_g streams[g, t - shifts[g]]``  with zero-fill
        outside ``[0, T_stream)``.
    """
    n_chg, t_stream, n, _ = streams.shape
    out = torch.zeros((t_det, n, n), dtype=streams.dtype, device=streams.device)
    for g in range(n_chg):
        s = int(shifts[g].item())
        t_in_lo = max(0, s)
        t_in_hi = min(t_det, t_stream + s)
        if t_in_hi > t_in_lo:
            out[t_in_lo:t_in_hi] += streams[g, t_in_lo - s : t_in_hi - s]
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


def test_fused_combine_zero_fill_below_shift(cuda_module):
    """§3.6.3 zero-fill: cube-time t < shifts[g] gets zero contribution
    from chgroup g (the corresponding stream sample is t - s < 0).

    With shifts = [0, 4, 8, 12] and t_det=16, output at t < 12 gets
    contributions only from a subset of chgroups; output at t = 12..15
    gets contributions from all four. The Python reference (now matching
    combine_chgroups) is the source of truth.
    """
    torch.manual_seed(0)
    n_chg, t_stream, n_grid, t_det = 4, 32, 8, 16
    streams = torch.complex(
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda"),
        torch.randn((n_chg, t_stream, n_grid, n_grid), dtype=torch.float32, device="cuda"),
    ).to(torch.complex32)
    shifts = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")

    out_fused = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex32, device="cuda")
    fused_combine_per_fdm(streams, shifts, out_fused)
    out_python = _python_combine(streams, shifts, t_det)

    diff = (out_fused - out_python).abs().max().item()
    assert diff < 0.05

    # Sanity: at t=0, only chgroup 0 contributes (others have shift > 0).
    expected_t0 = streams[0, 0]
    diff_t0 = (out_fused[0] - expected_t0).abs().max().item()
    assert diff_t0 < 0.05, f"t=0 should equal chgroup-0 only; diff={diff_t0:.3e}"


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
    apply the §3.6.3 ``out[t] = sum_g stream[g, t - shift[g]]`` rule.

    Layout: streams_cint8 is ``[N_chg, T_stream, 2, N, N]`` with
    re plane at axis_2=0 and im plane at axis_2=1, matching the
    bench's ``_build_synthetic_streams`` layout (which mirrors the
    M3 wire payload).
    """
    n_chg, t_stream, two, n_grid, _ = streams_cint8.shape
    assert two == 2
    re = streams_cint8[:, :, 0, :, :].to(torch.float32)
    im = streams_cint8[:, :, 1, :, :].to(torch.float32)
    streams_cf32 = torch.complex(re, im)
    out = torch.zeros((t_det, n_grid, n_grid), dtype=torch.complex64, device=streams_cint8.device)
    for g in range(n_chg):
        s = int(shifts[g].item())
        t_in_lo = max(0, s)
        t_in_hi = min(t_det, t_stream + s)
        if t_in_hi > t_in_lo:
            out[t_in_lo:t_in_hi] += streams_cf32[g, t_in_lo - s : t_in_hi - s]
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


# ---------------------------------------------------------------------------
# D25 — §3.6.3 sign-convention coherence: GPU agrees with combine_chgroups
# ---------------------------------------------------------------------------


def test_fused_combine_matches_combine_chgroups(cuda_module):
    """GPU kernel + ``compute_time_shift_search`` table coherently sums
    a synthetic dispersed pulse to the **same cube-time** as the CPU
    reference ``combine_chgroups``. This locks §3.6.3's ``out[t] =
    sum_g stream[g, t - shift[g]]`` convention across GPU and CPU.

    Setup: place a unit pulse in chgroup g at stream-time ``t_g = t_15
    - shifts[15-row, g]`` for a true DM. The dedispersed cube must
    peak at cube-time ``t_15`` with magnitude == N_chgroup.
    """
    import numpy as np
    from dsart.fine_dm.combiner import combine_chgroups, compute_time_shift_search

    n_chg, t_stream, n_grid, t_det = 16, 384, 8, 256
    true_dm = 100.0
    t_15 = 250

    table = compute_time_shift_search(
        coarse_dm_pc_cm3=np.array([0.0]),
        fine_dm_pc_cm3=np.array([true_dm]),
        fine_to_coarse=np.zeros(1, dtype=np.int64),
        t_int_search_us=524.288,
    )
    shifts_np = table.shifts[0]   # [N_chg]
    t_burst = t_15 - shifts_np    # per-chgroup stream-time of the pulse

    # Build per-chgroup cf32 streams with a unit pulse at the dispersed
    # arrival time per chgroup. Same fixture for CPU and GPU so any
    # disagreement reflects a sign-convention divergence.
    streams_cf = np.zeros((n_chg, t_stream, n_grid, n_grid), dtype=np.complex64)
    l_b, m_b = n_grid // 2, n_grid // 2
    for g in range(n_chg):
        streams_cf[g, int(t_burst[g]), l_b, m_b] = 1.0

    cpu_out = combine_chgroups(
        per_chgroup_streams={g: streams_cf[g] for g in range(n_chg)},
        time_shift_per_chgroup=shifts_np,
        t_window=(0, t_det),
        n_grid=n_grid,
    )
    cpu_peak_t = int(np.argmax(np.abs(cpu_out[:, l_b, m_b])))
    cpu_peak_val = float(np.abs(cpu_out[cpu_peak_t, l_b, m_b]))
    assert cpu_peak_t == t_15, (
        f"CPU combine_chgroups must peak at t_15={t_15}; got {cpu_peak_t}"
    )
    assert abs(cpu_peak_val - n_chg) < 1e-5, (
        f"CPU coherent peak should equal N_chg={n_chg}; got {cpu_peak_val}"
    )

    streams_t = torch.from_numpy(streams_cf).to(torch.complex64).cuda()
    shifts_t = torch.from_numpy(shifts_np.astype(np.int32)).cuda()
    out_t = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex64, device="cuda")
    fused_combine_per_fdm(streams_t, shifts_t, out_t)
    gpu_re = out_t.real.to(torch.float32).cpu().numpy()
    gpu_peak_t = int(np.argmax(np.abs(gpu_re[:, l_b, m_b])))
    gpu_peak_val = float(np.abs(gpu_re[gpu_peak_t, l_b, m_b]))
    assert gpu_peak_t == t_15, (
        f"GPU fused_combine must peak at t_15={t_15}; got {gpu_peak_t} "
        "(sign-convention regression vs CPU combine_chgroups)"
    )
    assert abs(gpu_peak_val - n_chg) < 1e-3, (
        f"GPU coherent peak should equal N_chg={n_chg}; got {gpu_peak_val}"
    )

    # The two should agree numerically.
    cpu_re = cpu_out.real.astype(np.float32)
    diff = float(np.abs(gpu_re - cpu_re).max())
    assert diff < 1e-4, (
        f"GPU vs CPU max-abs diff = {diff:.3e}; should be near machine-eps "
        "(both implement out[t] = sum_g stream[g, t - shift[g]])"
    )


def test_fused_dequant_combine_matches_cint8_combine_chgroups(cuda_module):
    """Same §3.6.3 lock-in as test_fused_combine_matches_combine_chgroups,
    but for the cint8-input dequant+combine variant.
    """
    import numpy as np
    from dsart.fine_dm.combiner import combine_chgroups, compute_time_shift_search

    n_chg, t_stream, n_grid, t_det = 16, 384, 8, 256
    true_dm = 100.0
    t_15 = 250

    table = compute_time_shift_search(
        coarse_dm_pc_cm3=np.array([0.0]),
        fine_dm_pc_cm3=np.array([true_dm]),
        fine_to_coarse=np.zeros(1, dtype=np.int64),
        t_int_search_us=524.288,
    )
    shifts_np = table.shifts[0]
    t_burst = t_15 - shifts_np

    streams_cint8 = np.zeros((n_chg, t_stream, 2, n_grid, n_grid), dtype=np.int8)
    l_b, m_b = n_grid // 2, n_grid // 2
    for g in range(n_chg):
        streams_cint8[g, int(t_burst[g]), 0, l_b, m_b] = 100  # real plane

    streams_t = torch.from_numpy(streams_cint8).cuda()
    shifts_t = torch.from_numpy(shifts_np.astype(np.int32)).cuda()
    out_t = torch.empty((t_det, n_grid, n_grid), dtype=torch.complex64, device="cuda")
    fused_dequant_combine_per_fdm(streams_t, shifts_t, out_t)
    gpu_re = out_t.real.to(torch.float32).cpu().numpy()
    peak_t = int(np.argmax(np.abs(gpu_re[:, l_b, m_b])))
    peak_val = float(np.abs(gpu_re[peak_t, l_b, m_b]))
    assert peak_t == t_15, f"dequant fused must peak at t_15={t_15}; got {peak_t}"
    assert abs(peak_val - 100 * n_chg) < 1e-3, (
        f"dequant coherent peak should equal 100×N_chg={100 * n_chg}; got {peak_val}"
    )
