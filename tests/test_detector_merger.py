"""Tests for ``dsart.detector.merger`` (M5 chunk 2).

Plan §4.4 line 1589 cross-kernel SNR-sort + 4D merge-radius suppression.

Coverage:

  * Empty input → empty output.
  * Single candidate → trivially survives.
  * Two candidates within the 4D merge radius → only the higher-SNR
    survives, with its kernel_id recorded.
  * Two candidates outside the merge radius (along any one axis) → both
    survive.
  * Per-axis radius semantics: l, m use ``merge_radius_lm``; fdm uses
    ``merge_radius_fdm``; t uses ``merge_radius_t``. Verified
    independently per axis.
  * Tie-break determinism: SNR ties resolve on (event_specnum, dm_idx,
    l, m, kernel_id) for stable output regardless of input order.
  * Many-candidate "FRB-like burst" pattern: a single δ-pulse fires from
    ~16 nearby (img, dm, time) triples; merger collapses to exactly 1
    survivor (the plan §8 line 2329 acceptance criterion).
"""

from __future__ import annotations

import os
import random

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.detector.merger import (  # noqa: E402
    DEFAULT_MERGE_RADIUS_FDM,
    DEFAULT_MERGE_RADIUS_LM,
    DEFAULT_MERGE_RADIUS_T,
    merge_across_kernels,
)


def _cand(
    *, snr: float, l: float = 4.0, m: float = 4.0, dm_idx: int = 10,
    event_specnum: int = 256, kernel_id: str = "unit:d1:b1",
    width_samples: int = 1,
) -> Candidate:
    return Candidate(
        l=l, m=m, dm_fine=float(dm_idx), dm_idx=dm_idx,
        event_specnum=event_specnum, width_samples=width_samples,
        kernel_id=kernel_id, snr=snr, detector_version="v1.M5",
        flags=int(CandidateFlags.NONE), search_node_id=0, gpu_half=0,
    )


def test_merge_empty_input_yields_empty_output() -> None:
    assert merge_across_kernels([]) == []


def test_merge_single_candidate_survives_trivially() -> None:
    c = _cand(snr=9.0)
    out = merge_across_kernels([c])
    assert out == [c]


def test_merge_two_within_radius_keeps_higher_snr_and_records_kernel_id() -> None:
    """Two candidates within the 4D merge radius collapse to the one
    with higher SNR; the surviving Candidate's kernel_id is the one that
    won (per plan §1589 "winning kernel triple's kernel_id is recorded")."""
    higher = _cand(snr=12.0, kernel_id="psf:d3:b4")
    lower = _cand(snr=10.0, kernel_id="unit:d1:b1")
    # Same (l, m, dm, event_specnum) — definitely within all radii.
    out = merge_across_kernels([lower, higher])
    assert len(out) == 1
    assert out[0].snr == 12.0
    assert out[0].kernel_id == "psf:d3:b4"


def test_merge_two_outside_lm_radius_both_survive() -> None:
    """Two candidates separated by > merge_radius_lm in either l or m
    axis both survive (per-axis half-window, NOT 2-norm)."""
    a = _cand(snr=12.0, l=4.0, m=4.0)
    b = _cand(snr=10.0, l=4.0, m=4.0 + DEFAULT_MERGE_RADIUS_LM + 1)
    out = merge_across_kernels([a, b])
    assert len(out) == 2


def test_merge_two_outside_fdm_radius_both_survive() -> None:
    a = _cand(snr=12.0, dm_idx=10)
    b = _cand(snr=10.0, dm_idx=10 + DEFAULT_MERGE_RADIUS_FDM + 1)
    out = merge_across_kernels([a, b])
    assert len(out) == 2


def test_merge_two_outside_t_radius_both_survive() -> None:
    a = _cand(snr=12.0, event_specnum=100)
    b = _cand(snr=10.0, event_specnum=100 + DEFAULT_MERGE_RADIUS_T + 1)
    out = merge_across_kernels([a, b])
    assert len(out) == 2


def test_merge_radius_inclusive_at_exact_boundary() -> None:
    """At exactly the radius distance the candidate IS within the
    radius (≤ comparison) → suppressed."""
    a = _cand(snr=12.0, l=4.0, m=4.0)
    b = _cand(snr=10.0, l=4.0, m=4.0 + DEFAULT_MERGE_RADIUS_LM)
    out = merge_across_kernels([a, b])
    assert len(out) == 1
    assert out[0].snr == 12.0


def test_merge_per_axis_independence() -> None:
    """A candidate within the (l, m, t) radii but outside the fdm radius
    survives — the four axes are independent half-windows."""
    a = _cand(snr=12.0, l=4.0, m=4.0, dm_idx=10, event_specnum=256)
    b = _cand(
        snr=10.0,
        l=4.0, m=4.0, event_specnum=256,
        dm_idx=10 + DEFAULT_MERGE_RADIUS_FDM + 1,
    )
    out = merge_across_kernels([a, b])
    assert len(out) == 2


def test_merge_input_order_invariant() -> None:
    """Result is invariant to input permutation (sorted internally)."""
    cands = [
        _cand(snr=12.0, kernel_id="psf:d1:b1", event_specnum=100),
        _cand(snr=10.0, kernel_id="unit:d1:b1", event_specnum=100),
        _cand(snr=11.0, kernel_id="psf_shift_l:d1:b1", event_specnum=100),
    ]
    a = merge_across_kernels(cands)
    b = merge_across_kernels(list(reversed(cands)))
    rng = random.Random(42)
    shuf = cands.copy()
    rng.shuffle(shuf)
    c = merge_across_kernels(shuf)
    # All three runs produce the same survivor SNRs.
    assert {x.snr for x in a} == {x.snr for x in b} == {x.snr for x in c}


def test_merge_tie_break_deterministic() -> None:
    """SNR ties resolve on (event_specnum, dm_idx, l, m, kernel_id) so
    re-runs produce bit-identical survivor lists."""
    a = _cand(snr=10.0, kernel_id="unit:d1:b1")
    b = _cand(snr=10.0, kernel_id="psf:d1:b1")  # tie on SNR; ki sorts later
    # Both within all radii (same l, m, dm, t). The kernel_id tie-break
    # is alphabetic so 'psf' < 'unit' → 'psf' wins.
    out_ab = merge_across_kernels([a, b])
    out_ba = merge_across_kernels([b, a])
    assert len(out_ab) == 1 and len(out_ba) == 1
    assert out_ab[0].kernel_id == "psf:d1:b1"
    assert out_ba[0].kernel_id == "psf:d1:b1"


def test_merge_frb_burst_collapses_to_one() -> None:
    """Plan §8 line 2329 acceptance: a single δ-pulse fires from ~8-16
    nearby (img, dm, time) kernel triples and the merger collapses them
    to exactly one Candidate. We simulate the per-kernel decoder output
    for one pulse at (l=32, m=32, dm_idx=20, event_specnum=300) — each
    kernel triple's NMS picks a slightly-shifted local-max because of the
    boxcar widths, but all are within the 4D merge radius.
    """
    pulse_l, pulse_m, pulse_dm, pulse_t = 32.0, 32.0, 20, 300
    cands = []
    # 4 image × 4 dm × 8 time = 128 triples; we model that ~16 of them
    # actually exceed threshold for one strong pulse, with small jitter
    # in (l, m, fdm, t) per triple.
    rng = random.Random(0)
    for img in ("unit", "psf", "psf_shift_lm", "psf_shift_l"):
        for dm in ("d1", "d3"):
            for tw in ("b1", "b2"):
                kid = f"{img}:{dm}:{tw}"
                snr = 14.0 + rng.uniform(-1.0, 1.0)
                # jitter within ±2 cells / ±1 fdm / ±2 samples
                cands.append(_cand(
                    snr=snr,
                    l=pulse_l + rng.choice([-1, 0, 0, 1]),
                    m=pulse_m + rng.choice([-1, 0, 0, 1]),
                    dm_idx=pulse_dm + rng.choice([-1, 0, 0, 1]),
                    event_specnum=pulse_t + rng.choice([-2, -1, 0, 1, 2]),
                    kernel_id=kid,
                    width_samples=int(tw[1:]),
                ))
    out = merge_across_kernels(cands)
    assert len(out) == 1, (
        f"expected 1 survivor for a single FRB-like pulse, got {len(out)}; "
        f"survivors={[(c.kernel_id, c.snr, c.l, c.m, c.dm_idx, c.event_specnum) for c in out]}"
    )
    # The survivor should be the highest-SNR member of the set.
    expected_max_snr = max(c.snr for c in cands)
    assert out[0].snr == pytest.approx(expected_max_snr)


def test_merge_distant_pulses_both_survive() -> None:
    """Two truly distant pulses (well outside the merge radii) BOTH
    survive — verifies the merger doesn't over-collapse."""
    pulse_a = _cand(snr=12.0, l=10.0, m=10.0, dm_idx=10, event_specnum=100)
    pulse_b = _cand(
        snr=11.0, l=200.0, m=200.0, dm_idx=200, event_specnum=10000,
    )
    out = merge_across_kernels([pulse_a, pulse_b])
    assert len(out) == 2


def test_merge_negative_radius_rejected() -> None:
    with pytest.raises(ValueError, match="merge_radius"):
        merge_across_kernels([], merge_radius_lm=-1)
