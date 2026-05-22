"""Tests for the C1 cross-kernel merger (M7.4).

``src/dsart/detector/merger.py::merge_across_kernels_c1`` ships the
new C1 geometry locked 2026-05-21 in
``docs/c1c2/C1C2_DESIGN.md`` §2.3:

  * OR-mode (l, m) suppression to merge across the EW / NS arms of
    the DSA cross PSF.
  * Width-aware time window
    ``dt_specnum_max = t_frac * 0.5 * (w_i + w_j) * sample_period_specnum``.
  * Fine-DM axis half-window via ``dm_max_trials``.

These tests stress every axis independently + the cross-axis
interactions. They do NOT exercise the legacy merger; that's still
covered by ``tests/test_detector_merger.py``.
"""

from __future__ import annotations

import os
import random

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.detector.merger import (  # noqa: E402
    MergerConfig,
    merge_across_kernels_c1,
)


def _cand(
    *,
    snr: float,
    l: float = 32.0,
    m: float = 32.0,
    dm_idx: int = 10,
    event_specnum: int = 256,
    kernel_id: str = "unit:d1:b1",
    width_samples: int = 1,
) -> Candidate:
    return Candidate(
        l=l,
        m=m,
        dm_fine=float(dm_idx),
        dm_idx=dm_idx,
        event_specnum=event_specnum,
        width_samples=width_samples,
        kernel_id=kernel_id,
        snr=snr,
        detector_version="v1.M5",
        flags=int(CandidateFlags.NONE),
        search_node_id=0,
        gpu_half=0,
    )


def _cfg(
    *,
    lm_max_cells: int = 3,
    dm_max_trials: int = 2,
    t_frac: float = 1.0,
    sample_period_specnum: int = 16,
) -> MergerConfig:
    return MergerConfig(
        lm_max_cells=lm_max_cells,
        dm_max_trials=dm_max_trials,
        t_frac=t_frac,
        sample_period_specnum=sample_period_specnum,
    )


# ---------------------------------------------------------------------------
# MergerConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults_match_design_doc() -> None:
    cfg = MergerConfig()
    assert cfg.lm_max_cells == 3
    assert cfg.dm_max_trials == 2
    assert cfg.t_frac == 1.0
    assert cfg.sample_period_specnum == 16


def test_config_rejects_negative_lm() -> None:
    with pytest.raises(ValueError, match="lm_max_cells"):
        MergerConfig(lm_max_cells=-1)


def test_config_rejects_negative_dm() -> None:
    with pytest.raises(ValueError, match="dm_max_trials"):
        MergerConfig(dm_max_trials=-1)


def test_config_rejects_negative_t_frac() -> None:
    with pytest.raises(ValueError, match="t_frac"):
        MergerConfig(t_frac=-0.1)


def test_config_rejects_zero_sample_period_specnum() -> None:
    with pytest.raises(ValueError, match="sample_period_specnum"):
        MergerConfig(sample_period_specnum=0)


# ---------------------------------------------------------------------------
# Trivial / empty inputs
# ---------------------------------------------------------------------------


def test_empty_input_yields_empty_output() -> None:
    assert merge_across_kernels_c1([], _cfg()) == []


def test_single_candidate_survives_trivially() -> None:
    c = _cand(snr=9.0)
    out = merge_across_kernels_c1([c], _cfg())
    assert out == [c]


# ---------------------------------------------------------------------------
# OR semantics over (l, m): EITHER axis close → merge
# ---------------------------------------------------------------------------


def test_or_lm_small_delta_l_large_delta_m_merges() -> None:
    """Δl small, Δm large → in_l True, in_m False, in_lm_or_cross True
    → lower-SNR sibling suppressed (merged into stronger arm)."""
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0)
    # Within Δl=0 ≤ 3 → in_l. Δm = 100 ≫ 3 → in_m False. OR → True.
    b = _cand(snr=10.0, l=32.0, m=132.0)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1
    assert out[0].snr == 12.0


def test_or_lm_small_delta_m_large_delta_l_merges() -> None:
    """Symmetric to above on the other arm."""
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0)
    # Δl large, Δm small → in_m True → OR True → merge.
    b = _cand(snr=10.0, l=132.0, m=32.0)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1
    assert out[0].snr == 12.0


def test_or_lm_both_axes_far_no_merge() -> None:
    """Δl and Δm BOTH out of range → in_lm_or_cross False → both
    survive."""
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0)
    b = _cand(snr=10.0, l=132.0, m=132.0)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_or_lm_inclusive_at_exact_boundary_l() -> None:
    """At exactly |Δl| = lm_max_cells the predicate is True (≤
    comparison) → merge."""
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0)
    b = _cand(snr=10.0, l=32.0 + 3, m=132.0)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1
    assert out[0].snr == 12.0


def test_or_lm_inclusive_at_exact_boundary_m() -> None:
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0)
    b = _cand(snr=10.0, l=132.0, m=32.0 + 3)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1
    assert out[0].snr == 12.0


def test_or_lm_just_outside_l_and_far_m_no_merge() -> None:
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0)
    b = _cand(snr=10.0, l=32.0 + 4, m=132.0)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# fine-DM axis half-window
# ---------------------------------------------------------------------------


def test_fdm_within_window_merges() -> None:
    cfg = _cfg(dm_max_trials=2, lm_max_cells=3)
    a = _cand(snr=12.0, dm_idx=10)
    b = _cand(snr=10.0, dm_idx=10 + 2)  # within
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1
    assert out[0].snr == 12.0


def test_fdm_outside_window_no_merge() -> None:
    cfg = _cfg(dm_max_trials=2, lm_max_cells=3)
    a = _cand(snr=12.0, dm_idx=10)
    b = _cand(snr=10.0, dm_idx=10 + 3)  # |Δdm| = 3 > 2 → fail
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_fdm_dominates_when_lm_satisfied() -> None:
    """Even with (l, m) within OR-cross, fdm-fail → no merge."""
    cfg = _cfg(dm_max_trials=2, lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0, dm_idx=10)
    b = _cand(snr=10.0, l=33.0, m=33.0, dm_idx=10 + 5)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Width-aware time window
# ---------------------------------------------------------------------------


def test_time_window_width_one_separation_4_no_merge() -> None:
    """w=1 + w=1, sample_period_specnum=16, t_frac=1.0:
    dt_specnum_max = 1.0 * 0.5 * (1+1) * 16 = 16. So |Δspecnum| > 16
    → no merge. Two pulsar pulses at 4 *samples* apart = 64 specnums
    → far outside window → both survive."""
    cfg = _cfg()
    a = _cand(snr=12.0, event_specnum=256, width_samples=1)
    b = _cand(snr=10.0, event_specnum=256 + 4 * 16, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_time_window_width_one_separation_within_merges() -> None:
    """w=1 + w=1: dt_specnum_max = 16. Separation of 1 sample = 16
    specnums → exactly at the boundary (≤) → merge."""
    cfg = _cfg()
    a = _cand(snr=12.0, event_specnum=256, width_samples=1)
    b = _cand(snr=10.0, event_specnum=256 + 16, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1


def test_time_window_width_one_separation_just_over_no_merge() -> None:
    """Same as above but 17 specnums → just outside (> 16) → no merge."""
    cfg = _cfg()
    a = _cand(snr=12.0, event_specnum=256, width_samples=1)
    b = _cand(snr=10.0, event_specnum=256 + 17, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_time_window_width_32_separation_16_samples_merges() -> None:
    """w=32 + w=32: dt_specnum_max = 1.0 * 0.5 * (32+32) * 16 =
    32 * 16 = 512 specnums. Separation of 16 samples = 256 specnums
    → within → merge."""
    cfg = _cfg()
    a = _cand(snr=12.0, event_specnum=10_000, width_samples=32)
    b = _cand(snr=10.0, event_specnum=10_000 + 16 * 16, width_samples=32)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1


def test_time_window_asymmetric_widths_uses_mean_overlap() -> None:
    """Width 32 + width 1: dt_specnum_max = 0.5 * (32 + 1) * 16 = 264
    specnums. Separation of 17 specnums → inside → merge."""
    cfg = _cfg()
    a = _cand(snr=12.0, event_specnum=10_000, width_samples=32)
    b = _cand(snr=10.0, event_specnum=10_017, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1


def test_t_frac_half_tightens_window() -> None:
    """t_frac = 0.5 halves the merge window. With w=w=32:
    dt_specnum_max = 0.5 * 0.5 * 64 * 16 = 256 specnums. Separation
    of 257 → just outside → no merge."""
    cfg = _cfg(t_frac=0.5)
    a = _cand(snr=12.0, event_specnum=10_000, width_samples=32)
    b = _cand(snr=10.0, event_specnum=10_000 + 257, width_samples=32)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_sample_period_specnum_scales_window() -> None:
    """sample_period_specnum = 1 → dt_specnum_max for w=w=1 is
    0.5 * 2 * 1 = 1 specnum → very tight."""
    cfg = _cfg(sample_period_specnum=1)
    a = _cand(snr=12.0, event_specnum=256, width_samples=1)
    b = _cand(snr=10.0, event_specnum=256 + 2, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Cross-axis interactions — only suppress when ALL three predicates pass.
# ---------------------------------------------------------------------------


def test_in_lm_or_pass_but_t_fail_no_merge() -> None:
    """lm OR True, fdm True, but t-axis fails → no merge."""
    cfg = _cfg()
    a = _cand(snr=12.0, l=32.0, m=32.0, dm_idx=10, event_specnum=256, width_samples=1)
    # event_specnum diff = 17 > 16 → t fails.
    b = _cand(snr=10.0, l=33.0, m=33.0, dm_idx=10, event_specnum=273, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_in_t_and_fdm_pass_but_lm_or_fail_no_merge() -> None:
    cfg = _cfg(lm_max_cells=3)
    a = _cand(snr=12.0, l=32.0, m=32.0, dm_idx=10, event_specnum=256)
    b = _cand(snr=10.0, l=132.0, m=132.0, dm_idx=10, event_specnum=256)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_all_three_pass_merges() -> None:
    """Canonical FRB-like neighbour pulse: (l, m) within cross arm, dm
    within window, time within width window → merge."""
    cfg = _cfg()
    a = _cand(snr=12.0, l=32.0, m=32.0, dm_idx=10, event_specnum=256, width_samples=4)
    b = _cand(snr=10.0, l=34.0, m=70.0, dm_idx=11, event_specnum=300, width_samples=4)
    # dt_specnum_max = 1.0 * 0.5 * (4+4) * 16 = 64. |256-300| = 44 ≤ 64 ✓
    # |Δl|=2 ≤ 3 ✓ (in_l), so in_lm_or_cross ✓
    # |Δdm|=1 ≤ 2 ✓
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1
    assert out[0].snr == 12.0


# ---------------------------------------------------------------------------
# Determinism / ordering invariants
# ---------------------------------------------------------------------------


def test_input_order_invariant() -> None:
    cfg = _cfg()
    cands = [
        _cand(snr=12.0, kernel_id="psf:d1:b1"),
        _cand(snr=10.0, kernel_id="unit:d1:b1"),
        _cand(snr=11.0, kernel_id="psf_shift_l:d1:b1"),
    ]
    a = merge_across_kernels_c1(cands, cfg)
    b = merge_across_kernels_c1(list(reversed(cands)), cfg)
    rng = random.Random(42)
    shuf = cands.copy()
    rng.shuffle(shuf)
    c = merge_across_kernels_c1(shuf, cfg)
    assert {x.snr for x in a} == {x.snr for x in b} == {x.snr for x in c}


def test_tie_break_deterministic() -> None:
    """SNR ties resolve on (event_specnum, dm_idx, l, m, kernel_id);
    'psf' < 'unit' alphabetically wins."""
    cfg = _cfg()
    a = _cand(snr=10.0, kernel_id="unit:d1:b1")
    b = _cand(snr=10.0, kernel_id="psf:d1:b1")
    out_ab = merge_across_kernels_c1([a, b], cfg)
    out_ba = merge_across_kernels_c1([b, a], cfg)
    assert len(out_ab) == 1 and len(out_ba) == 1
    assert out_ab[0].kernel_id == "psf:d1:b1"
    assert out_ba[0].kernel_id == "psf:d1:b1"


# ---------------------------------------------------------------------------
# FRB-burst / distant-pulses sanity
# ---------------------------------------------------------------------------


def test_frb_burst_collapses_to_one() -> None:
    """Single δ-pulse fires from many (img, dm, time) triples with
    small jitter within the width-aware time window → exactly one C1
    survivor. Uses width tokens b4..b32 (the widths that actually
    overlap a typical bright pulse) so the t-window covers the bank's
    natural per-kernel boxcar-centre jitter.
    """
    cfg = _cfg()
    pulse_l, pulse_m, pulse_dm, pulse_t = 32.0, 32.0, 20, 30_000
    cands = []
    rng = random.Random(0)
    for img in ("unit", "psf", "psf_shift_lm", "psf_shift_l"):
        for dm in ("d1", "d3"):
            for tw in ("b4", "b8", "b16", "b32"):
                kid = f"{img}:{dm}:{tw}"
                snr = 14.0 + rng.uniform(-1.0, 1.0)
                w = int(tw[1:])
                # Time-axis jitter scaled to the per-kernel boxcar
                # centre uncertainty (≈ ± w/2 detector samples =
                # ± w/2 * sample_period_specnum raw specnums).
                t_jitter = rng.choice([-w // 2, 0, 0, w // 2]) * cfg.sample_period_specnum
                cands.append(_cand(
                    snr=snr,
                    l=pulse_l + rng.choice([-1, 0, 0, 1]),
                    m=pulse_m + rng.choice([-1, 0, 0, 1]),
                    dm_idx=pulse_dm + rng.choice([-1, 0, 0, 1]),
                    event_specnum=pulse_t + t_jitter,
                    kernel_id=kid,
                    width_samples=w,
                ))
    out = merge_across_kernels_c1(cands, cfg)
    assert len(out) == 1, (
        f"expected 1 survivor for a single FRB-like pulse, got {len(out)}; "
        f"survivors={[(c.kernel_id, c.snr, c.l, c.m, c.dm_idx, c.event_specnum) for c in out]}"
    )
    expected_max_snr = max(c.snr for c in cands)
    assert out[0].snr == pytest.approx(expected_max_snr)


def test_distant_pulses_both_survive() -> None:
    cfg = _cfg()
    pulse_a = _cand(snr=12.0, l=10.0, m=10.0, dm_idx=10, event_specnum=100, width_samples=1)
    pulse_b = _cand(snr=11.0, l=200.0, m=200.0, dm_idx=200, event_specnum=1_000_000, width_samples=1)
    out = merge_across_kernels_c1([pulse_a, pulse_b], cfg)
    assert len(out) == 2


def test_multi_pulsar_pulses_width_one_separated_4_samples() -> None:
    """The spec calls this out explicitly: w=1 candidates separated
    by 4 *samples* (64 specnums at sps=16) should NOT merge under
    the default t_frac=1.0 window (4 specnums)."""
    cfg = _cfg()
    a = _cand(snr=11.0, event_specnum=10_000, width_samples=1)
    b = _cand(snr=10.0, event_specnum=10_000 + 4 * 16, width_samples=1)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 2


def test_width_32_separation_16_samples_merges() -> None:
    """Spec acceptance: w=32, separation 16 samples = 256 specnums
    is ≤ 0.5 * (32+32) * 16 = 512 → merge."""
    cfg = _cfg()
    a = _cand(snr=11.0, event_specnum=10_000, width_samples=32)
    b = _cand(snr=10.0, event_specnum=10_000 + 16 * 16, width_samples=32)
    out = merge_across_kernels_c1([a, b], cfg)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# SNR-sort respects ordering — highest SNR survives in chains
# ---------------------------------------------------------------------------


def test_snr_sort_chain_keeps_highest() -> None:
    cfg = _cfg()
    a = _cand(snr=20.0, l=32.0, m=32.0)
    b = _cand(snr=15.0, l=32.0, m=33.0)
    c = _cand(snr=10.0, l=32.0, m=34.0)
    out = merge_across_kernels_c1([a, b, c], cfg)
    assert len(out) == 1
    assert out[0].snr == 20.0
