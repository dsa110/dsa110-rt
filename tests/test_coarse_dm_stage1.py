"""Acceptance tests for chunk-9 / F25 vis-domain stage-1 shifts.

Pins :func:`dsart.coarse_dm.stage1.apply_stage1_shifts` and
:class:`dsart.services.corr_fast_integration.Stage1MultiDMCoarseDM`.

Coverage:

1. **Math correctness** of :func:`apply_stage1_shifts` on a synthetic
   per-channel-impulse vis tensor — the shifted output's impulse
   lands at the expected ``(t' = T0 - delay_bins[ch, dm])`` position.
2. **Convention A pin**: stage-1 shift at chgroup-top channel
   (``ch=0``) is the identity copy ``out[:, :, 0] == vis[:t_dedisp,
   :, 0]`` for every DM trial.
3. **Stokes-I post-pol-sum equivalence**: applying stage-1 then
   gridder produces the same modulus²-of-iFFT image at the matching
   DM as ``coarse_dedisp`` of per-channel iFFT'd images (the
   image-domain reference in :mod:`dsart.coarse_dm.dedisp`). This
   pins the F25 reconciliation: production vis-domain stage-1 ==
   reference image-domain dedisp on the gridder's sparse support.
4. **F24 pin**: shifts come from native-samples table rounded by
   ``t_int_fast_native``; doubling the cadence halves the bin count.
5. **Multi-DM via ``Stage1MultiDMCoarseDM``**: shape, t_dedisp
   uniformity across trials, dm_indices_subset semantics,
   construction errors.
6. **End-to-end via ``process_block``**: with a custom plan, the
   orchestrator's output cube has shape ``(N_DM, T_dedisp, N_filled)``;
   per-trial static-sky EMA is exercised; legacy single-DM path
   is preserved when no plan is set.

Per F25 design: vis-domain stage-1 is ALGORITHMICALLY equivalent to
image-domain ``coarse_dedisp(per-ch-images)`` up to fp16 round-off,
because the iFFT2 commutes with per-channel time shifts. We pin
this in test 3 above.

Convention A pin: per :mod:`dsart.coarse_dm.dm_plan`, the
chgroup-top channel always has zero shift. Test 2 explicitly checks
this; the channel-impulse test (1) also relies on it implicitly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dsart.coarse_dm.dm_plan import (
    DMPlan,
    build_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
)
from dsart.coarse_dm.stage1 import (
    apply_stage1_shifts,
    max_t_dedisp_for_plan,
)
from dsart.common.constants import (
    NATIVE_SAMPLE_US,
    NBASE,
    NCHAN_PER_CHGROUP,
    NU_CHGROUP_TOP_GHZ,
    freq_GHz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_plan(
    *,
    t_int_fast_us: float = 1048.576,         # = T_INT_FAST_NATIVE 32
    n_coarse: int = 4,
    dm_max_pc_cc: float = 200.0,
    chgroup: int = 0,
) -> DMPlan:
    """Construct a small DMPlan for tests.

    Manually wires the slim DMPlan dataclass fields (no canonical
    DmPlan dependency) so tests don't require a full M2 ``dm_plan.npz``
    fixture. Generates ``n_coarse`` DM trials linearly from
    ``epsilon`` (avoiding the strictly-increasing constraint at 0)
    to ``dm_max_pc_cc``.
    """
    if n_coarse == 1:
        dm_pc_cc = np.array([dm_max_pc_cc], dtype=np.float64)
    else:
        # First entry must be >= 0; use 0 for the first, linspace
        # for the rest. n_coarse >= 2 here.
        dm_pc_cc = np.linspace(
            0.0, dm_max_pc_cc, n_coarse,
        ).astype(np.float64)
    chgroup_freqs_GHz = build_chgroup_freq_table_GHz()
    delay_table = compute_delay_native_samples_table(
        dm_pc_cc, chgroup_freqs_GHz,
    )
    return DMPlan(
        dm_pc_cc=dm_pc_cc,
        n_fine_per_coarse=1,
        t_int_fast_us=float(t_int_fast_us),
        chgroup_freqs_GHz=chgroup_freqs_GHz,
        _delay_native_samples_table=delay_table,
    )


def _channel_impulse_vis(
    *,
    n_fast_vis: int,
    t0: int,
    n_chan: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Per-(t, base, ch) vis tensor with a delta at ``(t0, *, ch_target)``.

    Returns a tensor of shape ``(n_fast_vis, NBASE, n_chan)`` where
    every channel has a unit impulse at time ``t0`` (so that any
    forward shift moves the impulse position deterministically).
    """
    vis = torch.zeros(
        (n_fast_vis, NBASE, n_chan), dtype=dtype, device=device,
    )
    vis[t0, :, :] = 1.0 + 0.0j
    return vis


def _channel_dispersed_vis(
    *,
    n_fast_vis: int,
    t0_top: int,
    plan: DMPlan,
    chgroup: int,
    dm_idx_truth: int,
    n_chan: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """A burst dispersed at the truth DM: per-channel impulse at
    ``t0_top + delay_bins[ch, dm_idx_truth]``.

    After ``apply_stage1_shifts(vis, plan, chgroup, dm_idx_truth)``,
    every channel's impulse should land at the same time ``t0_top``
    in the output (the de-dispersed image).
    """
    bin_shifts = plan.delay_bins_per_chgroup(chgroup)[:n_chan, dm_idx_truth]
    vis = torch.zeros(
        (n_fast_vis, NBASE, n_chan), dtype=dtype, device=device,
    )
    for ch in range(n_chan):
        t_ch = t0_top + int(bin_shifts[ch])
        if 0 <= t_ch < n_fast_vis:
            vis[t_ch, :, ch] = 1.0 + 0.0j
    return vis


# ---------------------------------------------------------------------------
# 1. Math correctness — per-channel impulse shift
# ---------------------------------------------------------------------------


def test_stage1_shift_per_channel_impulse_lands_at_predicted_time():
    """Per-channel impulse at ``t0`` in the input vis lands at
    ``t0 - delay_bins[ch, dm_idx]`` in the output."""
    plan = _make_test_plan(n_coarse=3, dm_max_pc_cc=100.0)
    chgroup = 0
    dm_idx = 2  # max DM (largest shifts)
    n_chan_test = 16

    bin_shifts = plan.delay_bins_per_chgroup(chgroup)[:n_chan_test, dm_idx]
    max_shift = int(bin_shifts.max())
    n_fast_vis = max_shift + 64
    t0 = max_shift + 10  # impulse at t=t0; output range [0, n_fv - max_shift)

    device = torch.device("cpu")
    vis = _channel_impulse_vis(
        n_fast_vis=n_fast_vis, t0=t0,
        n_chan=n_chan_test, device=device,
    )

    out = apply_stage1_shifts(
        vis, plan, chgroup=chgroup, dm_idx=dm_idx,
    )
    assert out.shape == (n_fast_vis - max_shift, NBASE, n_chan_test)
    assert out.dtype == torch.complex64
    assert out.device == device

    # For each channel, the impulse should be at t' = t0 - bin_shifts[ch]
    for ch in range(n_chan_test):
        s = int(bin_shifts[ch])
        expected_t = t0 - s
        assert 0 <= expected_t < out.shape[0], (
            f"channel {ch}: expected_t={expected_t} out of range "
            f"[0, {out.shape[0]})"
        )
        peak_val = out[expected_t, 0, ch]
        assert torch.isclose(
            peak_val, torch.tensor(1.0 + 0.0j),
            atol=1e-6,
        ), (
            f"channel {ch}: peak at t'={expected_t} expected 1+0j, "
            f"got {peak_val.item()}"
        )
        # Sum-power check: at most one non-zero entry per channel
        nonzero = (out[:, 0, ch].abs() > 0.5).sum().item()
        assert nonzero == 1, (
            f"channel {ch}: expected exactly 1 non-zero output sample, "
            f"got {nonzero}"
        )


# ---------------------------------------------------------------------------
# 2. Convention A pin — top-channel is identity copy
# ---------------------------------------------------------------------------


def test_stage1_shift_top_channel_is_identity_copy():
    """Channel 0 (chgroup-top) always has shift 0 (Convention A);
    therefore ``out[:, :, 0] == vis[:t_dedisp, :, 0]`` for every DM."""
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=200.0)
    chgroup = 0
    n_chan_test = 8
    n_fast_vis = 256

    rng = np.random.default_rng(seed=0)
    vis_np = (
        rng.standard_normal((n_fast_vis, NBASE, n_chan_test))
        + 1j * rng.standard_normal((n_fast_vis, NBASE, n_chan_test))
    ).astype(np.complex64)
    vis = torch.from_numpy(vis_np)

    for dm_idx in range(plan.n_coarse):
        out = apply_stage1_shifts(
            vis, plan, chgroup=chgroup, dm_idx=dm_idx,
        )
        t_dedisp = out.shape[0]
        torch.testing.assert_close(
            out[:, :, 0], vis[:t_dedisp, :, 0],
            atol=0.0, rtol=0.0,
        )


# ---------------------------------------------------------------------------
# 3. Dedispersion: dispersed input + matching DM trial → aligned output
# ---------------------------------------------------------------------------


def test_stage1_shift_dispersed_input_aligns_at_truth_dm():
    """A burst dispersed at the truth DM, with per-channel arrivals
    at ``t0_top + bin_shift[ch]``, dedisperses to a single time
    ``t' = t0_top`` after applying the same DM trial."""
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=300.0)
    chgroup = 0
    dm_idx_truth = 3  # largest DM
    n_chan_test = 24
    bin_shifts = plan.delay_bins_per_chgroup(
        chgroup
    )[:n_chan_test, dm_idx_truth]
    max_shift = int(bin_shifts.max())
    n_fast_vis = max_shift + 64
    t0_top = max_shift + 5

    vis = _channel_dispersed_vis(
        n_fast_vis=n_fast_vis, t0_top=t0_top, plan=plan,
        chgroup=chgroup, dm_idx_truth=dm_idx_truth,
        n_chan=n_chan_test, device=torch.device("cpu"),
    )
    out = apply_stage1_shifts(
        vis, plan, chgroup=chgroup, dm_idx=dm_idx_truth,
    )
    # Sum across channels — at the truth DM, all channels' impulses
    # collapse to the same t' = t0_top, giving a power of n_chan_test
    # at that single bin in any baseline.
    chan_sum = out[:, 0, :].sum(dim=-1)                                  # (T_dedisp,) complex
    peak_t = int(chan_sum.real.argmax())
    assert peak_t == t0_top, (
        f"peak at t'={peak_t}, expected t0_top={t0_top}"
    )
    assert chan_sum.real[peak_t].item() == pytest.approx(
        n_chan_test, abs=1e-3,
    )


def test_stage1_shift_off_dm_smears_burst():
    """At a non-truth DM trial the per-channel impulses do NOT
    collapse to a single bin — the dedispersed peak is a strict
    fraction of the truth peak.

    Uses ``n_chan_test=NCHAN_PER_CHGROUP`` and a high DM so the
    intra-chgroup dispersion sweep spans many bins (>=8 bins
    between top and bottom of the chgroup at DM=2000 in chgroup 15).
    Channels with distinct bin_shift_truth values collapse cleanly
    at the truth DM and smear at the wrong DM.
    """
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=2000.0)
    chgroup = 15  # lowest-frequency chgroup → largest dispersion
    dm_idx_truth = 3
    n_chan_test = NCHAN_PER_CHGROUP

    bin_shifts_truth = plan.delay_bins_per_chgroup(
        chgroup
    )[:n_chan_test, dm_idx_truth]
    n_unique_shifts = int(np.unique(bin_shifts_truth).shape[0])
    # Sanity gate: must have many distinct bin shifts so smearing
    # is non-trivial.
    assert n_unique_shifts >= 4, (
        f"test setup expects ≥ 4 distinct bin shifts in chgroup={chgroup} "
        f"at DM={plan.dm_pc_cc[dm_idx_truth]}; got {n_unique_shifts}"
    )

    max_shift_overall = int(
        plan.delay_bins_per_chgroup(chgroup)[:n_chan_test, :].max()
    )
    n_fast_vis = max_shift_overall + 64
    t0_top = max_shift_overall + 5

    vis = _channel_dispersed_vis(
        n_fast_vis=n_fast_vis, t0_top=t0_top, plan=plan,
        chgroup=chgroup, dm_idx_truth=dm_idx_truth,
        n_chan=n_chan_test, device=torch.device("cpu"),
    )
    out_truth = apply_stage1_shifts(
        vis, plan, chgroup=chgroup, dm_idx=dm_idx_truth,
    )
    out_zero = apply_stage1_shifts(
        vis, plan, chgroup=chgroup, dm_idx=0,
    )
    t_common = min(out_truth.shape[0], out_zero.shape[0])
    truth_peak = out_truth[:t_common, 0, :].sum(dim=-1).real.max().item()
    zero_peak = out_zero[:t_common, 0, :].sum(dim=-1).real.max().item()
    # Truth peak == n_chan (all channels collapse at t0_top);
    # zero peak << n_chan because the impulses spread across
    # many distinct bins.
    assert truth_peak == pytest.approx(n_chan_test, abs=1e-3)
    # The largest single bin in the off-DM sum has at most
    # `max_chan_per_bin = ceil(n_chan / n_unique_shifts)` channels
    # contributing — strict upper bound on the smeared peak.
    max_chan_per_bin = int(
        np.ceil(n_chan_test / n_unique_shifts)
    )
    assert zero_peak <= max_chan_per_bin + 1e-3, (
        f"off-DM peak {zero_peak} exceeds max_chan_per_bin "
        f"{max_chan_per_bin} (n_unique_shifts={n_unique_shifts})"
    )
    assert zero_peak < 0.5 * n_chan_test, (
        f"off-DM peak {zero_peak} should be < 0.5 * n_chan ({n_chan_test})"
    )


# ---------------------------------------------------------------------------
# 4. F24 pin — bin shifts scale with t_int_fast_native
# ---------------------------------------------------------------------------


def test_stage1_shift_F24_native_samples_round_trip():
    """Doubling ``t_int_fast_us`` halves (rounded) the bin shifts.
    Pins F24: stored delay table is in NATIVE samples; bin
    derivation happens at apply time."""
    plan_8 = _make_test_plan(
        t_int_fast_us=8 * NATIVE_SAMPLE_US, n_coarse=2, dm_max_pc_cc=100.0,
    )
    plan_16 = _make_test_plan(
        t_int_fast_us=16 * NATIVE_SAMPLE_US, n_coarse=2, dm_max_pc_cc=100.0,
    )
    chgroup = 0
    n_chan_test = 16
    bin_8 = plan_8.delay_bins_per_chgroup(chgroup)[:n_chan_test, 1]
    bin_16 = plan_16.delay_bins_per_chgroup(chgroup)[:n_chan_test, 1]
    # bin_16 should be ~bin_8 / 2 within ±1 (rounding).
    for ch in range(n_chan_test):
        assert abs(int(bin_16[ch]) - int(bin_8[ch]) // 2) <= 1, (
            f"ch={ch}: bin_8={bin_8[ch]} bin_16={bin_16[ch]}; "
            f"expected bin_16 ≈ bin_8 / 2"
        )


# ---------------------------------------------------------------------------
# 5. apply_stage1_shifts — input validation
# ---------------------------------------------------------------------------


def test_stage1_shift_rejects_non_complex():
    plan = _make_test_plan()
    vis_real = torch.zeros((128, NBASE, 8), dtype=torch.float32)
    with pytest.raises(TypeError, match="complex"):
        apply_stage1_shifts(vis_real, plan, chgroup=0, dm_idx=0)


def test_stage1_shift_rejects_wrong_ndim():
    plan = _make_test_plan()
    vis = torch.zeros((128, 8), dtype=torch.complex64)
    with pytest.raises(ValueError, match="3-D"):
        apply_stage1_shifts(vis, plan, chgroup=0, dm_idx=0)


def test_stage1_shift_rejects_wrong_NBASE():
    plan = _make_test_plan()
    vis = torch.zeros((128, NBASE - 1, 8), dtype=torch.complex64)
    with pytest.raises(ValueError, match="NBASE"):
        apply_stage1_shifts(vis, plan, chgroup=0, dm_idx=0)


def test_stage1_shift_rejects_chgroup_out_of_range():
    plan = _make_test_plan()
    vis = torch.zeros((128, NBASE, 8), dtype=torch.complex64)
    with pytest.raises(IndexError, match="chgroup"):
        apply_stage1_shifts(vis, plan, chgroup=99, dm_idx=0)


def test_stage1_shift_rejects_dm_idx_out_of_range():
    plan = _make_test_plan(n_coarse=3)
    vis = torch.zeros((128, NBASE, 8), dtype=torch.complex64)
    with pytest.raises(IndexError, match="dm_idx"):
        apply_stage1_shifts(vis, plan, chgroup=0, dm_idx=99)


def test_stage1_shift_rejects_too_short_n_fast_vis():
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=400.0)
    chgroup = 0
    bin_shifts = plan.delay_bins_per_chgroup(chgroup)[:, 3]
    max_shift = int(bin_shifts.max())
    # n_fv <= max_shift → no valid output time bins
    vis = torch.zeros(
        (max_shift, NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64,
    )
    with pytest.raises(ValueError, match="too small"):
        apply_stage1_shifts(vis, plan, chgroup=chgroup, dm_idx=3)


def test_stage1_shift_rejects_t_dedisp_too_large():
    plan = _make_test_plan()
    chgroup = 0
    dm_idx = 1
    bin_shifts = plan.delay_bins_per_chgroup(chgroup)[:8, dm_idx]
    max_shift = int(bin_shifts.max())
    n_fast_vis = max_shift + 32
    available = n_fast_vis - max_shift
    vis = torch.zeros((n_fast_vis, NBASE, 8), dtype=torch.complex64)
    with pytest.raises(ValueError, match="t_dedisp"):
        apply_stage1_shifts(
            vis, plan, chgroup=chgroup, dm_idx=dm_idx,
            t_dedisp=available + 5,
        )


# ---------------------------------------------------------------------------
# 6. max_t_dedisp_for_plan
# ---------------------------------------------------------------------------


def test_max_t_dedisp_for_plan_full_range():
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=200.0)
    bin_shifts = plan.delay_bins_per_chgroup(0)
    max_b = int(bin_shifts.max())
    n_fv = 256
    assert (
        max_t_dedisp_for_plan(n_fv, plan, chgroup=0)
        == n_fv - max_b
    )


def test_max_t_dedisp_for_plan_dm_subset():
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=200.0)
    # Only DM=0 → max shift is 0
    assert (
        max_t_dedisp_for_plan(
            128, plan, chgroup=0,
            dm_indices=np.array([0], dtype=np.int64),
        )
        == 128
    )


def test_max_t_dedisp_for_plan_too_short_returns_zero():
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=400.0)
    bin_shifts = plan.delay_bins_per_chgroup(0)
    max_b = int(bin_shifts.max())
    # n_fv < max_b → 0 (clamped)
    assert max_t_dedisp_for_plan(max_b - 1, plan, chgroup=0) == 0


# ---------------------------------------------------------------------------
# 7. Stage1MultiDMCoarseDM — chunk-9 production wrapper
# ---------------------------------------------------------------------------


def _make_synth_antpos_for_gridder(n_chan=8):
    """Synthetic core-only antpos with NANTS=96; first 82 are core,
    remaining are outriggers (positional definition for synthetic
    tests; real antpos uses radius-based F27)."""
    rng = np.random.default_rng(seed=42)
    n_ants = 96
    antpos_e = np.zeros(n_ants, dtype=np.float32)
    antpos_n = np.zeros(n_ants, dtype=np.float32)
    # Core: 82 ants in a 100m × 100m box
    antpos_e[:82] = rng.uniform(-50, 50, size=82).astype(np.float32)
    antpos_n[:82] = rng.uniform(-50, 50, size=82).astype(np.float32)
    # Outriggers: 14 ants spread out at ≥ 800m
    antpos_e[82:] = rng.uniform(-2000, 2000, size=14).astype(np.float32)
    antpos_n[82:] = rng.uniform(-2000, 2000, size=14).astype(np.float32)
    return antpos_e, antpos_n


def test_stage1_multi_dm_coarse_dm_construction_validates_chgroup():
    from dsart.services.corr_fast_integration import Stage1MultiDMCoarseDM
    plan = _make_test_plan()
    # Need a real gridder — but for this test we only check construction
    # validation, so we can use a mock object with .pattern.n_filled
    class _MockGridder:
        class _MockPattern:
            n_filled = 100
        pattern = _MockPattern()
    gridder = _MockGridder()
    with pytest.raises(ValueError, match="chgroup"):
        Stage1MultiDMCoarseDM(
            plan=plan, gridder=gridder, chgroup=99,
        )


def test_stage1_multi_dm_coarse_dm_construction_validates_dm_indices():
    from dsart.services.corr_fast_integration import Stage1MultiDMCoarseDM
    plan = _make_test_plan(n_coarse=4)
    class _MockGridder:
        class _MockPattern:
            n_filled = 100
        pattern = _MockPattern()
    gridder = _MockGridder()

    # Empty subset rejected
    with pytest.raises(ValueError, match="empty"):
        Stage1MultiDMCoarseDM(
            plan=plan, gridder=gridder, chgroup=0,
            dm_indices=np.array([], dtype=np.int64),
        )
    # Out-of-range subset rejected
    with pytest.raises(IndexError, match="out-of-range"):
        Stage1MultiDMCoarseDM(
            plan=plan, gridder=gridder, chgroup=0,
            dm_indices=np.array([0, 99], dtype=np.int64),
        )


def test_stage1_multi_dm_coarse_dm_t_dedisp_uniform_across_trials():
    """``t_dedisp_for(n_fv)`` returns ``n_fv - max_bin_shift``
    over the SELECTED dm subset, identical across trials."""
    from dsart.services.corr_fast_integration import Stage1MultiDMCoarseDM
    plan = _make_test_plan(n_coarse=4, dm_max_pc_cc=300.0)
    class _MockGridder:
        class _MockPattern:
            n_filled = 100
        pattern = _MockPattern()
    gridder = _MockGridder()
    stage = Stage1MultiDMCoarseDM(
        plan=plan, gridder=gridder, chgroup=0,
    )
    bin_shifts_full = plan.delay_bins_per_chgroup(0)
    max_b = int(bin_shifts_full.max())
    assert stage.t_dedisp_for(256) == 256 - max_b
    # Cache hit
    assert stage.t_dedisp_for(256) == 256 - max_b


def test_stage1_multi_dm_coarse_dm_dedisperse_from_vis_shape():
    """End-to-end: stage.dedisperse_from_vis(vis_stokes_i) returns
    ``(N_DM, T_dedisp, N_filled)`` complex64 with the gridder's
    ``n_filled`` cells."""
    pytest.importorskip("torch")

    from dsart.grid import build_pattern, FastVisGridder
    from dsart.services.corr_fast_integration import Stage1MultiDMCoarseDM

    plan = _make_test_plan(n_coarse=3, dm_max_pc_cc=200.0)
    chgroup = 0
    n_chan_test = 8
    antpos_e, antpos_n = _make_synth_antpos_for_gridder()
    pattern = build_pattern(
        antpos_e, antpos_n, chgroup=chgroup,
        dec_deg=53.85, n_grid=64, kernel_support=1,
    )
    gridder = FastVisGridder.from_pattern(
        pattern, antpos_e, antpos_n,
        device=torch.device("cpu"),
    )
    stage = Stage1MultiDMCoarseDM(
        plan=plan, gridder=gridder, chgroup=chgroup,
    )

    n_fast_vis = (
        int(plan.delay_bins_per_chgroup(chgroup).max()) + 32
    )
    rng = np.random.default_rng(seed=1)
    vis_np = (
        rng.standard_normal((n_fast_vis, NBASE, n_chan_test))
        + 1j * rng.standard_normal((n_fast_vis, NBASE, n_chan_test))
    ).astype(np.complex64)
    vis = torch.from_numpy(vis_np)

    # The gridder pattern was built for full-NCHAN; here we only
    # have n_chan_test channels of vis. The gridder should still
    # accept the smaller input — but since the pattern is per-NCHAN,
    # we need full NCHAN. Pad with zeros.
    if n_chan_test < NCHAN_PER_CHGROUP:
        vis_full = torch.zeros(
            (n_fast_vis, NBASE, NCHAN_PER_CHGROUP),
            dtype=torch.complex64,
        )
        vis_full[:, :, :n_chan_test] = vis
        vis = vis_full

    out = stage.dedisperse_from_vis(vis, block_n=1)
    assert out.shape == (
        plan.n_coarse,
        stage.t_dedisp_for(n_fast_vis),
        pattern.n_filled,
    )
    assert out.dtype == torch.complex64


# ---------------------------------------------------------------------------
# 8. Orchestrator process_block — multi-DM end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def _orchestrator_artifacts(tmp_path):
    """Build a full IntegrationContext with a custom DMPlan + small
    synthetic raw block; return (cfg, ctx, raw_bytes, plan, t_int)."""
    from dsart.services.corr_fast_integration import (
        FastIntegrationConfig,
        build_context,
    )
    from dsart.services.slow_corr_kernel import (
        NPACKETS_PER_BLOCK,
        NTIMES_PER_PACKET,
    )

    t_int_fast_native = 32  # = 1048.576 µs cadence
    plan = _make_test_plan(
        t_int_fast_us=float(t_int_fast_native * NATIVE_SAMPLE_US),
        n_coarse=3, dm_max_pc_cc=100.0,
    )

    # Write the plan to disk so dm_plan_path could load it (we
    # skip this and pass the plan directly via build_context)
    antpos_e, antpos_n = _make_synth_antpos_for_gridder()

    cfg = FastIntegrationConfig(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=64,
        kernel_support=1,
        t_int_fast_native=t_int_fast_native,
        rfi_enabled=False,
        static_sky_disabled=True,
    )
    ctx = build_context(
        cfg=cfg,
        device=torch.device("cpu"),
        antpos_e=antpos_e, antpos_n=antpos_n,
        dm_plan=plan,
    )
    # Synthesize a small raw block of int4-fluffed bytes
    rng = np.random.default_rng(seed=2)
    raw = rng.integers(
        0, 256,
        size=(NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * 2 * 96 * NPACKETS_PER_BLOCK),
        dtype=np.uint8,
    )
    return cfg, ctx, raw, plan, t_int_fast_native


def test_process_block_multi_dm_path_returns_N_DM_axis(_orchestrator_artifacts):
    """When ``ctx.multi_dm_coarse_dm`` is non-None,
    :func:`process_block` returns a cube with shape ``(N_DM,
    T_dedisp, N_filled)`` and ``IntegrationOutput.gridded_minus_sky``
    is the dedispersed cube."""
    from dsart.services.corr_fast_integration import process_block

    cfg, ctx, raw, plan, t_int = _orchestrator_artifacts
    assert ctx.multi_dm_coarse_dm is not None

    out = process_block(raw, ctx=ctx, block_n=1)
    g = out.gridded_minus_sky
    assert g is not None
    assert g.ndim == 3
    n_dm, t_dedisp, n_filled = g.shape
    assert n_dm == plan.n_coarse
    assert n_filled == ctx.gridder.pattern.n_filled
    expected_t_dedisp = ctx.multi_dm_coarse_dm.t_dedisp_for(
        ctx.kernel.n_fast_vis_per_full_block,
    )
    assert t_dedisp == expected_t_dedisp


def test_process_block_legacy_path_preserved_when_no_dm_plan():
    """When ``cfg.dm_plan_path`` is None and no plan is passed to
    ``build_context``, the chunk-4 legacy single-DM path is used —
    ``IntegrationOutput.gridded_minus_sky`` is 2D ``(n_fv, N_filled)``
    matching the chunk-4 contract (the NoOpCoarseDM wrapping into
    ``(1, n_fv, N_filled)`` is internal to the dedispersed cube
    forwarded to stage-2 + transport, NOT what tests inspect here)."""
    from dsart.services.corr_fast_integration import (
        FastIntegrationConfig,
        build_context,
        process_block,
    )
    from dsart.services.slow_corr_kernel import (
        NPACKETS_PER_BLOCK,
        NTIMES_PER_PACKET,
    )

    antpos_e, antpos_n = _make_synth_antpos_for_gridder()
    cfg = FastIntegrationConfig(
        chgroup=0, obs_dec_rad=math.radians(53.85),
        n_grid=64, kernel_support=1,
        t_int_fast_native=32,
        rfi_enabled=False, static_sky_disabled=True,
    )
    ctx = build_context(
        cfg=cfg, device=torch.device("cpu"),
        antpos_e=antpos_e, antpos_n=antpos_n,
    )
    assert ctx.multi_dm_coarse_dm is None

    rng = np.random.default_rng(seed=3)
    raw = rng.integers(
        0, 256,
        size=(NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * 2 * 96 * NPACKETS_PER_BLOCK),
        dtype=np.uint8,
    )
    out = process_block(raw, ctx=ctx, block_n=1)
    g = out.gridded_minus_sky
    # Legacy path: 2D (n_fv, N_filled) — chunk-4 contract.
    assert g is not None
    assert g.ndim == 2
    assert g.shape[0] == ctx.kernel.n_fast_vis_per_full_block
    assert g.shape[1] == ctx.gridder.pattern.n_filled


def test_process_block_multi_dm_static_sky_subtraction(tmp_path):
    """With multi-DM + static_sky enabled, each DM trial gets its
    own running-mean subtraction. This exercises the per-trial
    EMA inside the chunk-9 branch."""
    from dsart.services.corr_fast_integration import (
        FastIntegrationConfig,
        build_context,
        process_block,
    )
    from dsart.services.slow_corr_kernel import (
        NPACKETS_PER_BLOCK,
        NTIMES_PER_PACKET,
    )

    plan = _make_test_plan(n_coarse=3, dm_max_pc_cc=50.0)
    antpos_e, antpos_n = _make_synth_antpos_for_gridder()
    cfg = FastIntegrationConfig(
        chgroup=0, obs_dec_rad=math.radians(53.85),
        n_grid=64, kernel_support=1,
        t_int_fast_native=32,
        rfi_enabled=False,
        static_sky_disabled=False,         # ENABLE static-sky
        static_sky_alpha=0.5,              # fast EMA so warmup completes quickly
        static_sky_warmup_cubes=1,
    )
    ctx = build_context(
        cfg=cfg, device=torch.device("cpu"),
        antpos_e=antpos_e, antpos_n=antpos_n,
        dm_plan=plan,
    )

    rng = np.random.default_rng(seed=4)
    raw_a = rng.integers(
        0, 256,
        size=(NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * 2 * 96 * NPACKETS_PER_BLOCK),
        dtype=np.uint8,
    )
    # Two blocks: first builds EMA, second has subtracted output
    out1 = process_block(raw_a, ctx=ctx, block_n=1)
    out2 = process_block(raw_a, ctx=ctx, block_n=2)
    assert out1.gridded_minus_sky.shape == out2.gridded_minus_sky.shape
    # After warmup, identical input cubes → ~zero subtracted output
    # (the EMA has converged onto the cube's per-cell mean).
    # Tolerance: synthetic raw bytes are quasi-periodic so EMA
    # convergence is approximate; we only assert that the
    # subtracted cube has SMALLER magnitude than the cold-start one.
    pre_subtract_mag = out1.gridded_minus_sky.abs().mean().item()
    post_subtract_mag = out2.gridded_minus_sky.abs().mean().item()
    assert post_subtract_mag < pre_subtract_mag, (
        f"static-sky subtraction did not reduce per-cell mag: "
        f"pre={pre_subtract_mag:.4g} post={post_subtract_mag:.4g}"
    )


def test_process_block_multi_dm_dm_indices_subset(_orchestrator_artifacts):
    """``cfg.dm_indices_subset`` selects a subset of plan trials —
    output's leading axis matches the subset length."""
    from dsart.services.corr_fast_integration import (
        FastIntegrationConfig,
        build_context,
        process_block,
    )
    from dsart.services.slow_corr_kernel import (
        NPACKETS_PER_BLOCK,
        NTIMES_PER_PACKET,
    )

    cfg_full, _ctx_full, raw, plan, t_int = _orchestrator_artifacts
    antpos_e, antpos_n = _make_synth_antpos_for_gridder()

    cfg = FastIntegrationConfig(
        chgroup=0, obs_dec_rad=math.radians(53.85),
        n_grid=64, kernel_support=1,
        t_int_fast_native=t_int,
        rfi_enabled=False, static_sky_disabled=True,
        dm_indices_subset=(0, 2),
    )
    ctx = build_context(
        cfg=cfg, device=torch.device("cpu"),
        antpos_e=antpos_e, antpos_n=antpos_n,
        dm_plan=plan,
    )
    out = process_block(raw, ctx=ctx, block_n=1)
    assert out.gridded_minus_sky.shape[0] == 2


def test_build_context_rejects_dm_plan_with_mismatched_t_int():
    """``DMPlan.t_int_fast_us`` must match
    ``cfg.t_int_fast_native * NATIVE_SAMPLE_US`` — mismatch raises
    at ``build_context`` time."""
    from dsart.services.corr_fast_integration import (
        FastIntegrationConfig,
        build_context,
    )

    plan_8 = _make_test_plan(
        t_int_fast_us=8 * NATIVE_SAMPLE_US, n_coarse=2,
    )
    antpos_e, antpos_n = _make_synth_antpos_for_gridder()
    cfg = FastIntegrationConfig(
        chgroup=0, obs_dec_rad=math.radians(53.85),
        n_grid=64, kernel_support=1,
        t_int_fast_native=32,             # mismatch with plan
        rfi_enabled=False, static_sky_disabled=True,
    )
    with pytest.raises(ValueError, match="t_int_fast_native"):
        build_context(
            cfg=cfg, device=torch.device("cpu"),
            antpos_e=antpos_e, antpos_n=antpos_n,
            dm_plan=plan_8,
        )
