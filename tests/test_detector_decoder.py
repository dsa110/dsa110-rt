"""Tests for ``dsart.detector.decoder`` (M5 chunk 2).

Coverage:

  * ``decode_local_max`` empty-list short-circuit on noise-only score.
  * ``decode_local_max`` finds a single δ-pulse and returns the right
    ``Candidate`` fields.
  * Per-kernel NMS suppresses multiple peaks within the radius:
    Δfdm = k_dm_width // 2 + 1, Δt = k_time_width // 2 + 1, Δl = Δm = 2.
  * Returned ``Candidate`` carries the right kernel_id / width_samples /
    detector_version / search_node_id / gpu_half stamps.
  * ``filter_to_canonical`` halo gate flags candidates outside
    ``[dm_idx_canonical_lo, dm_idx_canonical_hi]`` with HALO_DROPPED.
  * Time-edge gate flags candidates within ``n_kernel_max_t // 2`` of
    either cube edge with TIME_EDGE_DROPPED.
  * Both gates can fire simultaneously (flag-or composition).
  * Dropped candidates carry the original SNR + kernel_id (only flags differ).
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.detector.decoder import decode_local_max, filter_to_canonical  # noqa: E402


# ---------------------------------------------------------------------------
# decode_local_max
# ---------------------------------------------------------------------------


def _make_zero_score(t: int = 32, n_fdm: int = 8, h: int = 16, w: int = 16):
    return torch.zeros(t, n_fdm, h, w, dtype=torch.float32)


def test_decode_local_max_empty_on_zero_score() -> None:
    """No cells above threshold → empty list short-circuit."""
    score = _make_zero_score()
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
    )
    assert out == []


def test_decode_local_max_finds_single_pulse() -> None:
    """A single super-threshold cell becomes exactly one Candidate with
    correct (l, m, fdm, snr) values stamped."""
    score = _make_zero_score()
    score[10, 3, 4, 5] = 12.5
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
    )
    assert len(out) == 1
    cand = out[0]
    assert isinstance(cand, Candidate)
    assert (int(cand.l), int(cand.m)) == (4, 5)
    assert cand.dm_idx == 3
    assert cand.snr == pytest.approx(12.5)
    assert cand.event_specnum == 10
    assert cand.kernel_id == "unit:d1:b1"
    assert cand.width_samples == 1
    assert cand.flags == int(CandidateFlags.NONE)


def test_decode_local_max_threshold_filters_below() -> None:
    """Cells exactly at threshold are pruned (strict >, not >=)."""
    score = _make_zero_score()
    score[5, 0, 4, 4] = 8.0  # exactly at threshold
    score[6, 0, 8, 8] = 9.0  # above threshold
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
    )
    assert len(out) == 1
    assert out[0].snr == pytest.approx(9.0)


def test_decode_local_max_nms_suppresses_within_spatial_radius() -> None:
    """Two peaks within the spatial NMS radius (Δl = Δm = 2): the
    smaller one is suppressed."""
    score = _make_zero_score()
    score[10, 0, 4, 4] = 12.0
    score[10, 0, 5, 5] = 10.0  # within Δl = Δm = 2
    score[10, 0, 9, 9] = 11.0  # outside the 2-cell radius from (4, 4)
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
    )
    snrs = sorted([c.snr for c in out], reverse=True)
    assert snrs == [12.0, 11.0]  # the 10.0 peak was suppressed


def test_decode_local_max_nms_suppresses_within_time_radius() -> None:
    """Two peaks within Δt = k_time_width // 2 + 1: smaller suppressed."""
    score = _make_zero_score()
    # k_time_width = 4 → Δt = 4 // 2 + 1 = 3
    score[10, 0, 4, 4] = 12.0
    score[12, 0, 4, 4] = 10.0  # within 3 samples of t=10
    score[20, 0, 4, 4] = 11.0  # outside the 3-sample radius from t=10
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d3:b4",
        k_dm_width=3, k_time_width=4,
    )
    snrs = sorted([c.snr for c in out], reverse=True)
    assert snrs == [12.0, 11.0]


def test_decode_local_max_nms_suppresses_within_fdm_radius() -> None:
    """Two peaks within Δfdm = k_dm_width // 2 + 1: smaller suppressed."""
    score = _make_zero_score()
    # k_dm_width = 5 → Δfdm = 5 // 2 + 1 = 3
    score[10, 4, 5, 5] = 12.0
    score[10, 6, 5, 5] = 10.0  # within 3 fdm trials of fdm=4
    score[10, 7, 5, 5] = 11.0  # also within (Δ = 3 from fdm=4)
    score[10, 0, 5, 5] = 9.0   # outside (Δ = 4 from fdm=4)
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d5:b1",
        k_dm_width=5, k_time_width=1,
    )
    snrs = sorted([c.snr for c in out], reverse=True)
    assert snrs == [12.0, 9.0]


def test_decode_local_max_stamps_search_node_and_gpu_half() -> None:
    """The (search_node_id, gpu_half) pair stamps onto every Candidate."""
    score = _make_zero_score()
    score[5, 0, 4, 4] = 9.0
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
        search_node_id=2, gpu_half=1,
    )
    assert out[0].search_node_id == 2
    assert out[0].gpu_half == 1


def test_decode_local_max_dm_lookup_via_fine_to_coarse() -> None:
    """If fine_to_coarse / fine_dm_pc_cm3 are passed, dm_idx + dm_fine
    are populated from those tables."""
    score = _make_zero_score(n_fdm=4)
    score[0, 2, 0, 0] = 9.0
    fine_to_coarse = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    fine_dm_pc = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
        fine_to_coarse=fine_to_coarse, fine_dm_pc_cm3=fine_dm_pc,
    )
    assert out[0].dm_idx == 1  # fine_to_coarse[2] = 1
    assert out[0].dm_fine == pytest.approx(30.0)


def test_decode_local_max_event_specnum_offset() -> None:
    """Candidate.event_specnum = caller's event_specnum + cube-relative t."""
    score = _make_zero_score()
    score[7, 0, 4, 4] = 9.0
    out = decode_local_max(
        score, threshold=8.0, kernel_id="unit:d1:b1",
        k_dm_width=1, k_time_width=1,
        event_specnum=1_000_000,
    )
    assert out[0].event_specnum == 1_000_007


# ---------------------------------------------------------------------------
# filter_to_canonical
# ---------------------------------------------------------------------------


def _candidate(
    *, dm_idx: int = 5, event_specnum: int = 256,
    flags: int = int(CandidateFlags.NONE),
) -> Candidate:
    return Candidate(
        l=4.0, m=4.0, dm_fine=float(dm_idx), dm_idx=dm_idx,
        event_specnum=event_specnum, width_samples=4,
        kernel_id="unit:d3:b4", snr=9.0, detector_version="v1.M5",
        flags=flags, search_node_id=0, gpu_half=0,
    )


def test_filter_to_canonical_passes_interior_candidate() -> None:
    """A candidate well inside the canonical zone passes both gates."""
    cand = _candidate(dm_idx=10, event_specnum=256)
    emit, dropped = filter_to_canonical(
        [cand],
        dm_idx_canonical_lo=5, dm_idx_canonical_hi=15,
        t_det=512, n_kernel_max_t=128,
    )
    assert len(emit) == 1 and not dropped
    assert emit[0].flags == int(CandidateFlags.NONE)


def test_filter_to_canonical_halo_drops() -> None:
    """dm_idx outside [lo, hi] → HALO_DROPPED."""
    below = _candidate(dm_idx=4, event_specnum=256)
    above = _candidate(dm_idx=16, event_specnum=256)
    emit, dropped = filter_to_canonical(
        [below, above],
        dm_idx_canonical_lo=5, dm_idx_canonical_hi=15,
        t_det=512, n_kernel_max_t=128,
    )
    assert emit == []
    assert len(dropped) == 2
    for c in dropped:
        assert c.flags & int(CandidateFlags.HALO_DROPPED)
        assert not (c.flags & int(CandidateFlags.TIME_EDGE_DROPPED))


def test_filter_to_canonical_time_edge_drops() -> None:
    """t_in_cube within n_kernel_max_t // 2 of either edge → TIME_EDGE_DROPPED.
    With t_det=512, n_kernel_max_t=128, the edge regions are [0, 64) and
    [448, 512)."""
    early = _candidate(dm_idx=10, event_specnum=30)   # in [0, 64)
    late = _candidate(dm_idx=10, event_specnum=500)   # in [448, 512)
    interior = _candidate(dm_idx=10, event_specnum=256)
    emit, dropped = filter_to_canonical(
        [early, late, interior],
        dm_idx_canonical_lo=5, dm_idx_canonical_hi=15,
        t_det=512, n_kernel_max_t=128,
    )
    assert len(emit) == 1
    assert emit[0].event_specnum == 256
    assert {c.event_specnum for c in dropped} == {30, 500}
    for c in dropped:
        assert c.flags & int(CandidateFlags.TIME_EDGE_DROPPED)
        assert not (c.flags & int(CandidateFlags.HALO_DROPPED))


def test_filter_to_canonical_both_gates_compose() -> None:
    """A candidate that fails both gates carries both flag bits."""
    cand = _candidate(dm_idx=20, event_specnum=10)
    emit, dropped = filter_to_canonical(
        [cand],
        dm_idx_canonical_lo=5, dm_idx_canonical_hi=15,
        t_det=512, n_kernel_max_t=128,
    )
    assert emit == []
    assert len(dropped) == 1
    assert dropped[0].flags & int(CandidateFlags.HALO_DROPPED)
    assert dropped[0].flags & int(CandidateFlags.TIME_EDGE_DROPPED)


def test_filter_to_canonical_preserves_pre_existing_flags() -> None:
    """Pre-existing flag bits (NOISE_WARMUP, RFI_WARMING_UP) survive
    the gate."""
    pre = _candidate(
        dm_idx=10, event_specnum=256,
        flags=int(CandidateFlags.NOISE_WARMUP),
    )
    emit, dropped = filter_to_canonical(
        [pre], dm_idx_canonical_lo=5, dm_idx_canonical_hi=15,
        t_det=512, n_kernel_max_t=128,
    )
    assert len(emit) == 1
    assert emit[0].flags == int(CandidateFlags.NOISE_WARMUP)


def test_filter_to_canonical_dropped_preserve_snr_and_kernel_id() -> None:
    """Dropped candidates carry the original SNR + kernel_id (only flags
    differ); this is what the candidate-log reader relies on for offline
    reconstruction (plan §3 line 611)."""
    cand = _candidate(dm_idx=999, event_specnum=256)
    _, dropped = filter_to_canonical(
        [cand], dm_idx_canonical_lo=5, dm_idx_canonical_hi=15,
        t_det=512, n_kernel_max_t=128,
    )
    assert dropped[0].snr == 9.0
    assert dropped[0].kernel_id == "unit:d3:b4"
    assert dropped[0].dm_idx == 999  # unchanged
