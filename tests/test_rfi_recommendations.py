"""Tests for the RFI-flagger review items R1, R2, R3, R5, R6, R9.

Each test pins a claim that was wrong, unstated, or unverifiable
before. Where a number is quoted it was measured on 2026-09-13, not
assumed — several of the review's own figures did not survive contact
with the deployed geometry and the tests record the measured values.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dsart.rfi import far_calibration as FC          # noqa: E402
from dsart.rfi import null_watchdog as NW            # noqa: E402
from dsart.rfi import sk as S                        # noqa: E402
from dsart.rfi.bandpass_outlier import bandpass_outlier_mask   # noqa: E402
from dsart.rfi.combine import FlagSourceBit          # noqa: E402
from dsart.rfi.group_outlier import group_outlier_mask         # noqa: E402
from dsart.rfi.hi_guard import (                     # noqa: E402
    DEFAULT_PROTECTED_LINES,
    ProtectedLine,
    hi_guard_channels,
    protected_line_channels,
)
from dsart.rfi.persistence import (                  # noqa: E402
    DEFAULT_PER_DETECTOR_SPECS,
    PerDetectorPersistence,
)

M = 64


def _sk_of(power: np.ndarray, m: int = M) -> float:
    s1 = torch.tensor([[[[power.sum()]]]], dtype=torch.float64)
    s2 = torch.tensor([[[[(power ** 2).sum()]]]], dtype=torch.float64)
    return float(S.compute_sk(s1, s2, m))


def _moments(power: np.ndarray, m: int = M):
    """(s1, s2) shaped (N_acc=1, 1, 1, 1) for the mask helpers."""
    s1 = torch.tensor([[[[power.sum()]]]], dtype=torch.float32)
    s2 = torch.tensor([[[[(power ** 2).sum()]]]], dtype=torch.float32)
    return {m: s1}, {m: s2}


# ---------------------------------------------------------------------------
# R1 — the SK sign convention
# ---------------------------------------------------------------------------


def test_continuous_rfi_depresses_sk_below_one():
    """The docstring used to say CW LIFTS SK. It depresses it.

    A continuous carrier makes the channel power deterministic, so
    Var(p)/E[p]^2 -> 0 and SK -> 0.
    """
    lo, _hi = S.sk_thresholds(M, 1e-4)
    sk = _sk_of(np.full(M, 66.0))
    assert sk < lo
    assert sk == pytest.approx(0.0, abs=1e-9)


def test_intermittent_rfi_lifts_sk_above_one():
    rng = np.random.default_rng(3)
    _lo, hi = S.sk_thresholds(M, 1e-4)
    bursty = np.zeros(M)
    bursty[:3] = rng.exponential(size=3) * 200.0     # 3/64 duty cycle
    assert _sk_of(bursty) > hi


def test_thermal_noise_sits_inside_the_band():
    rng = np.random.default_rng(11)
    lo, hi = S.sk_thresholds(M, 1e-4)
    assert lo < _sk_of(rng.exponential(size=M)) < hi


def test_sk_high_bit_separates_carrier_from_bursty():
    """Bit 7 is the duty-cycle diagnostic R1 asked for."""
    rng = np.random.default_rng(3)
    carrier = np.full(M, 66.0)
    bursty = np.zeros(M)
    bursty[:3] = rng.exponential(size=3) * 200.0

    any_c, high_c = S.sk_combined_masks(*_moments(carrier), far=1e-4)
    assert bool(any_c.any()) and not bool(high_c.any())   # carrier

    any_b, high_b = S.sk_combined_masks(*_moments(bursty), far=1e-4)
    assert bool(any_b.any()) and bool(high_b.any())       # bursty

    # sk_mask stays the OR of the two sides.
    s1, s2 = _moments(bursty)
    low_m, high_m = S.sk_masks(s1[M], s2[M], M, far=1e-4)
    assert torch.equal(S.sk_mask(s1[M], s2[M], M, far=1e-4), low_m | high_m)


def test_sk_high_is_the_last_free_bit():
    assert int(FlagSourceBit.SK_HIGH) == 128
    used = [b for b in FlagSourceBit if b != FlagSourceBit.NONE]
    assert len({int(b) for b in used}) == len(used)        # distinct
    assert max(int(b) for b in used) == 128                # uint8 full


# ---------------------------------------------------------------------------
# R2 — analytic Pearson IV thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [64, 256, 1024, 4096])
def test_sk_moments_are_exact(m):
    """E[SK] is exactly 1, and the variance matches Nita & Gary."""
    mean, var, skew, kurt = S.sk_moments(m)
    assert mean == pytest.approx(1.0, abs=1e-12)
    ng = 4.0 * m * (m - 1) / ((m - 2) * (m + 2) * (m + 3))
    assert var == pytest.approx(ng, rel=1e-3)
    assert skew > 0.0 and kurt > 3.0                       # right-skewed
    if m == 64:
        # The value the sk.py docstring quotes empirically.
        assert skew == pytest.approx(1.1, abs=0.05)


@pytest.mark.parametrize("m", [256, 1024, 4096])
def test_pearson4_agrees_with_mc_where_it_is_trusted(m):
    p4 = S.pearson4_sk_thresholds(m, 1e-4)
    mc = S._mc_sk_thresholds(m, 1e-4, n_trials=200_000)
    assert p4[0] == pytest.approx(mc[0], rel=0.05)
    assert p4[1] == pytest.approx(mc[1], rel=0.05)


def test_pearson4_is_not_used_below_the_measured_crossover():
    """R2 assumed Pearson IV was exact. It is a four-moment FIT.

    At M=64 it lands 5.9 sigma low on realised FAR, so production
    keeps the Monte Carlo there.
    """
    assert S.PEARSON4_MIN_M == 256
    p4_low, _ = S.pearson4_sk_thresholds(64, 1e-4)
    prod_low, _ = S.sk_thresholds(64, 1e-4)
    assert prod_low != pytest.approx(p4_low, rel=1e-6)
    assert prod_low > p4_low            # MC's lower bound is higher


def test_mc_thresholds_are_chunked_not_one_shot():
    """The one-shot allocation was ~61 GB at M=4096."""
    assert S._MC_CHUNK_BYTES <= 512 * 1024 * 1024
    per_chunk = S._MC_CHUNK_BYTES // (4096 * 8)
    assert per_chunk * 4096 * 8 <= S._MC_CHUNK_BYTES


# ---------------------------------------------------------------------------
# R3 — FAR-calibrated MAD-outlier thresholds
# ---------------------------------------------------------------------------


def test_bandpass_far_calibration_hits_its_target():
    rng = np.random.default_rng(5)
    g = torch.tensor(
        rng.gamma(4096, 1 / 4096, size=(96, 384, 2)), dtype=torch.float32)
    for far in (1e-3, 1e-4):
        frac = float(
            bandpass_outlier_mask(g, far=far, m_acc=4096).float().mean())
        assert frac == pytest.approx(far, rel=0.6)


def test_k_five_was_never_five_sigma():
    """The knob did not mean what it said — but not by the factor the
    review claimed (it simulated M=1; the detector runs at M=4096)."""
    far_at_k5 = FC.mad_outlier_far(384, 5.0, 4096)
    gaussian_5sigma = 5.733e-7
    assert far_at_k5 > gaussian_5sigma          # hotter than Gaussian
    assert far_at_k5 < 1e-4                     # but nowhere near 1.2e-4
    assert 2.0 < far_at_k5 / gaussian_5sigma < 10.0


def test_explicit_k_still_overrides_far():
    rng = np.random.default_rng(5)
    g = torch.tensor(
        rng.gamma(4096, 1 / 4096, size=(32, 384, 2)), dtype=torch.float32)
    a = bandpass_outlier_mask(g, k=5.0)
    b = bandpass_outlier_mask(g, k=5.0, m_acc=4096)
    assert torch.equal(a, b)                    # m_acc ignored when k given
    with pytest.raises(ValueError):
        bandpass_outlier_mask(g, k=5.0, far=1e-4)


def test_group_far_calibration_runs_and_is_monotonic():
    k_loose = FC.group_threshold_k(96, 1e-3, n_chan=384, m_acc=4096)
    k_tight = FC.group_threshold_k(96, 1e-5, n_chan=384, m_acc=4096)
    assert k_tight > k_loose
    rng = np.random.default_rng(9)
    g = torch.tensor(
        rng.gamma(4096, 1 / 4096, size=(96, 384, 2)), dtype=torch.float32)
    assert int(group_outlier_mask(g, far=1e-4, m_acc=4096).sum()) == 0


def test_far_thresholds_are_cached_and_deterministic():
    a = FC.bandpass_threshold_k(384, 1e-4, m_acc=4096)
    b = FC.bandpass_threshold_k(384, 1e-4, m_acc=4096)
    assert a == b


# ---------------------------------------------------------------------------
# R5 — per-detector latch windows
# ---------------------------------------------------------------------------


def _mask(val: bool, shape=(2, 3, 2)) -> torch.Tensor:
    return torch.full(shape, val, dtype=torch.bool)


def test_sum_threshold_is_not_latched():
    """Latching a dilation dilates in time too."""
    assert DEFAULT_PER_DETECTOR_SPECS["sumthr"] is None
    p = PerDetectorPersistence(cadence_s=0.134)
    assert "sumthr" not in p.detector_names


def test_group_outlier_latches_longer_than_sk():
    sk_w, sk_h = DEFAULT_PER_DETECTOR_SPECS["sk"]
    gr_w, gr_h = DEFAULT_PER_DETECTOR_SPECS["group"]
    assert gr_w > sk_w and gr_h > sk_h


def test_per_detector_latch_holds_only_the_firing_detector():
    p = PerDetectorPersistence(
        cadence_s=1.0,
        specs={"sk": (2.0, 5.0), "group": (2.0, 5.0), "sumthr": None},
        latch_frac=1.0,
    )
    on, off = _mask(True), _mask(False)
    # Two cubes of SK -> SK latches; group never fires.
    for _ in range(2):
        held, stats = p.update({"sk": on, "group": off, "sumthr": on})
    assert bool(held.all())
    assert set(stats) == {"sk", "group"}
    # SK stops: the hold keeps it up.
    held, _ = p.update({"sk": off, "group": off, "sumthr": off})
    assert bool(held.all())


def test_per_detector_rejects_unknown_only_masks():
    p = PerDetectorPersistence(cadence_s=1.0, specs={"sk": (2.0, 5.0)})
    with pytest.raises(ValueError):
        p.update({"bandpass": _mask(True)})


# ---------------------------------------------------------------------------
# R6 — configurable protected lines (SLOW path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chgroup", list(range(16)))
def test_hi_default_is_reproduced_exactly(chgroup):
    hi_only = protected_line_channels(
        chgroup, lines=(DEFAULT_PROTECTED_LINES[0],))
    assert np.array_equal(hi_guard_channels(chgroup), hi_only)


def test_out_of_band_lines_cost_nothing():
    """OH is at 1.61-1.72 GHz, outside the 1.311-1.499 GHz band."""
    for cg in range(16):
        assert np.array_equal(
            protected_line_channels(cg),
            protected_line_channels(cg, lines=(DEFAULT_PROTECTED_LINES[0],)),
        )


def test_a_new_line_can_be_protected_without_a_code_change():
    # A line placed inside chgroup 0 (top of band, ~1.4988 GHz).
    line = ProtectedLine("TEST", 1.4980, v_lo_kms=-200.0, v_hi_kms=200.0)
    assert protected_line_channels(0, lines=(line,)).any()
    assert not protected_line_channels(15, lines=(line,)).any()


def test_protected_line_rejects_an_inverted_window():
    with pytest.raises(ValueError):
        ProtectedLine("BAD", 1.42, v_lo_kms=200.0,
                      v_hi_kms=-200.0).freq_range_GHz()


# ---------------------------------------------------------------------------
# R9 part 2 — null-rate watchdog
# ---------------------------------------------------------------------------


def test_watchdog_flags_a_detector_off_its_null():
    v = NW.check_null_rates(
        {"sk": 1e-4, "bandpass": 3e-2},
        {"sk": 1e-4, "bandpass": 1e-4},
        n_cells=10_000_000,
    )
    by = {x.detector: x for x in v}
    assert by["sk"].ok
    assert not by["bandpass"].ok and by["bandpass"].ratio == pytest.approx(300.0)


def test_watchdog_calls_a_thin_sample_inconclusive_not_cold():
    v = NW.check_null_rates({"sk": 0.0}, {"sk": 1e-4}, n_cells=1000)
    assert v[0].ok and "inconclusive" in v[0].note


def test_watchdog_skips_detectors_with_no_far():
    v = NW.check_null_rates({"flagants": 0.02}, {"sk": 1e-4})
    assert [x.detector for x in v] == ["sk"]
    assert "not reported" in v[0].note


def test_quietest_channels_picks_the_low_occupancy_half():
    counts = np.zeros((4, 100, 2), dtype=np.int64)
    counts[:, 50:, :] = 99                       # top half is filthy
    clean = NW.quietest_channels(counts, quantile=0.5)
    assert clean[:50].all() and not clean[50:].any()
