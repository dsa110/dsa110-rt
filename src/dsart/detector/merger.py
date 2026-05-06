"""Cross-kernel SNR-sort + 4D merge-radius suppression
(plan §4.4 line 1589; M5 PARALLEL_AGENTS.md §3 Class A).

Consumes the union of per-kernel ``Candidate`` lists from
``decoder.decode_local_max`` (one list per kernel triple in the K=128
bank), sorts by SNR descending, and for each candidate in order
suppresses any later candidate within a 4D merge radius:

    Δl_merge   = 3 cells
    Δm_merge   = 3 cells
    Δfdm_merge = 5 fine-DM trials
    Δt_merge   = max time-kernel boxcar width (= 128 samples at default ops)

The winning kernel triple's ``kernel_id`` is recorded on the surviving
``Candidate`` (this is the FIRST item — highest SNR — that swallowed the
neighborhood). This collapses the ~8-16 redundant detections that a
single FRB pulse fires across neighbouring (img, dm, time) triples into
**one** record per pulse (the operator-facing acceptance per plan §8
line 2329 "cross-kernel merge collapses to exactly 1 Candidate per
injection").

The algorithm is O(N log N) sort + O(N²) suppression scan; N ≪ 1000 per
cube at θ=8 in practice (a noisy cube fires < 16 raw candidates, plus
~8-16 per pulse), so the Python loop is fine. If N grows large enough
to matter (v2 / large kernel banks), the suppression can be replaced
with a KD-tree or a per-cube spatial hash.
"""

from __future__ import annotations

from typing import List

from ..common.contracts import Candidate

__all__ = [
    "merge_across_kernels",
    "DEFAULT_MERGE_RADIUS_LM",
    "DEFAULT_MERGE_RADIUS_FDM",
    "DEFAULT_MERGE_RADIUS_T",
]


DEFAULT_MERGE_RADIUS_LM: int = 3
"""Plan §4.4 line 1589 default merge radius for the (l, m) axes (cells)."""

DEFAULT_MERGE_RADIUS_FDM: int = 5
"""Plan §4.4 line 1589 default merge radius for the fine-DM axis (trials)."""

DEFAULT_MERGE_RADIUS_T: int = 128
"""Plan §4.4 line 1589 default merge radius for the time axis (samples).
Equals max(K_time_widths) at default ops. The bench can override per
``configs/config_compute_search.yaml::decoder.merge_radius_t`` (which
the plan defaults to the literal string ``"max_time_kernel_width"`` —
the search_compute service resolves it at startup)."""


def merge_across_kernels(
    candidates: List[Candidate],
    *,
    merge_radius_lm: int = DEFAULT_MERGE_RADIUS_LM,
    merge_radius_fdm: int = DEFAULT_MERGE_RADIUS_FDM,
    merge_radius_t: int = DEFAULT_MERGE_RADIUS_T,
) -> List[Candidate]:
    """Cross-kernel SNR-sort + 4D merge-radius suppression.

    Args:
        candidates: flat union of per-kernel ``Candidate`` lists. Order
            is irrelevant — the function sorts internally.
        merge_radius_lm: half-window radius along the (l, m) axes
            (cells). Defaults to plan §1589 = 3.
        merge_radius_fdm: half-window radius along the fine-DM axis
            (trials). Defaults to plan §1589 = 5.
        merge_radius_t: half-window radius along the time axis (in
            t_int_search samples). Defaults to plan §1589 =
            ``max(K_time_widths)`` = 128.

    Returns:
        ``List[Candidate]`` — the pruned set of survivors. Every
        survivor's neighborhood (within the four merge radii along
        l, m, fdm, t respectively) contains no other candidate of equal
        or higher SNR.
    """
    if merge_radius_lm < 0 or merge_radius_fdm < 0 or merge_radius_t < 0:
        raise ValueError(
            f"merge_radius_lm/fdm/t must be ≥ 0; got "
            f"({merge_radius_lm}, {merge_radius_fdm}, {merge_radius_t})"
        )
    if not candidates:
        return []

    # Sort by SNR descending; tie-break on (event_specnum, dm_idx, l, m,
    # kernel_id) for determinism (Python's sort is stable, so this also
    # preserves insertion order within an SNR tie when those tie-break
    # keys coincide).
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            -c.snr,
            c.event_specnum,
            c.dm_idx,
            c.l,
            c.m,
            c.kernel_id,
        ),
    )

    survivors: List[Candidate] = []
    for cand in sorted_cands:
        suppressed = False
        for s in survivors:
            if (
                abs(cand.event_specnum - s.event_specnum) <= merge_radius_t
                and abs(cand.dm_idx - s.dm_idx) <= merge_radius_fdm
                and abs(cand.l - s.l) <= merge_radius_lm
                and abs(cand.m - s.m) <= merge_radius_lm
            ):
                # Plan §1589 reads "within a 4D merge-radius (Δl=3, Δm=3,
                # Δfdm=5, Δt=128)" — per-axis half-windows, not a 4-norm
                # radius. We test each axis independently above.
                suppressed = True
                break
        if not suppressed:
            survivors.append(cand)
    return survivors
