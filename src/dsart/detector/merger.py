"""Cross-kernel SNR-sort + 4D merge-radius suppression.

This module ships two SNR-sorted suppression scans:

  * :func:`merge_across_kernels_c1` — the **new C1 geometry** (M7.4,
    locked 2026-05-21 in ``docs/c1c2/C1C2_DESIGN.md`` §2.3). Suppression
    rule per ``MergerConfig`` (defaults ``lm_max_cells=3``,
    ``dm_max_trials=2``, ``t_frac=1.0``,
    ``sample_period_specnum=16``)::

        dt_specnum_max = cfg.t_frac * 0.5 * (c.w + s.w)
                                            * cfg.sample_period_specnum
        in_t           = |c.event_specnum - s.event_specnum| ≤ dt_specnum_max
        in_fdm         = |c.dm_idx - s.dm_idx|              ≤ cfg.dm_max_trials
        in_lm_or_cross = (|c.l - s.l| ≤ cfg.lm_max_cells)  OR
                         (|c.m - s.m| ≤ cfg.lm_max_cells)

        suppress c iff in_t AND in_fdm AND in_lm_or_cross

    The OR over (l, m) is intentional — a real burst spread along
    either the EW or NS arm of the cross-shaped DSA-110 core PSF leaves
    survivors at small Δl or small Δm respectively, and we want to
    collapse those into a single peak. The time predicate is
    width-aware: the half-overlap edge between the two boxcars'
    matched-filter widths, scaled by ``sample_period_specnum`` so
    ``event_specnum`` (raw SNAP units) directly compares to the
    sample-scale window.

  * :func:`merge_across_kernels` — the **legacy axis-AND box merger**
    (plan §4.4 line 1589). Per-axis half-window radii on (l, m, fdm,
    t) with AND semantics across all four axes. Kept as a deprecation
    alias for the chunk-2 unit tests + offline tools that still expect
    the old radii. The C1 production path (``DeterministicDetector``
    with a ``MergerConfig``) routes through ``merge_across_kernels_c1``.

Both scans are O(N log N) sort + O(N²) suppression. N is tiny (a
single cube emits ≤ a few hundred raw candidates pre-merge in
practice), so the Python loop is fine. If N ever grows large enough
to matter, the suppression can be replaced with a KD-tree or a per-
cube spatial hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..common.contracts import Candidate

__all__ = [
    "MergerConfig",
    "merge_across_kernels_c1",
    "merge_across_kernels",
    "DEFAULT_MERGE_RADIUS_LM",
    "DEFAULT_MERGE_RADIUS_FDM",
    "DEFAULT_MERGE_RADIUS_T",
]


# ---------------------------------------------------------------------------
# C1 merger (new, locked 2026-05-21)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MergerConfig:
    """C1 cross-kernel merger geometry (``docs/c1c2/C1C2_DESIGN.md`` §2.3).

    Defaults match the locked production knobs for M7.4 bring-up.

    Args:
        lm_max_cells: OR-mode half-window on the (l, m) axes in cells.
            Two candidates merge if EITHER ``|Δl| ≤ lm_max_cells`` OR
            ``|Δm| ≤ lm_max_cells`` (cross-shaped DSA core PSF leaves
            sidelobes along one arm at a time).
        dm_max_trials: half-window on the fine-DM axis, ``|Δdm_idx| ≤
            dm_max_trials``.
        t_frac: scale factor on the half-overlap edge of the two
            candidates' matched-filter widths. ``t_frac = 1.0`` keeps
            the natural half-window edge predicate; <1.0 is useful only
            for stress tests.
        sample_period_specnum: number of raw SNAP spec-num units per
            detector sample (= 16 at default ops; same value the cube
            geometry carries on the wire).
    """

    lm_max_cells: int = 3
    dm_max_trials: int = 2
    t_frac: float = 1.0
    sample_period_specnum: int = 16

    def __post_init__(self) -> None:
        if self.lm_max_cells < 0:
            raise ValueError(
                f"lm_max_cells={self.lm_max_cells}, expected ≥ 0"
            )
        if self.dm_max_trials < 0:
            raise ValueError(
                f"dm_max_trials={self.dm_max_trials}, expected ≥ 0"
            )
        if self.t_frac < 0.0:
            raise ValueError(
                f"t_frac={self.t_frac}, expected ≥ 0"
            )
        if self.sample_period_specnum <= 0:
            raise ValueError(
                f"sample_period_specnum={self.sample_period_specnum}, expected > 0"
            )


def merge_across_kernels_c1(
    candidates: List[Candidate],
    cfg: MergerConfig,
) -> List[Candidate]:
    """C1 cross-kernel SNR-sort merger with OR-cross spatial geometry.

    See module docstring + ``MergerConfig`` for the suppression rule.

    Args:
        candidates: flat union of per-kernel ``Candidate`` lists. Order
            is irrelevant — the function sorts internally by descending
            SNR, with deterministic tie-breakers.
        cfg: merger geometry knobs (see :class:`MergerConfig`).

    Returns:
        ``List[Candidate]`` — the pruned survivors. Every survivor is
        the highest-SNR member of its neighbourhood under the C1
        suppression rule.
    """
    if not candidates:
        return []

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

    lm_max = int(cfg.lm_max_cells)
    dm_max = int(cfg.dm_max_trials)
    t_frac = float(cfg.t_frac)
    sps = float(cfg.sample_period_specnum)

    survivors: List[Candidate] = []
    for cand in sorted_cands:
        suppressed = False
        for s in survivors:
            dt_specnum_max = (
                t_frac
                * 0.5
                * float(int(cand.width_samples) + int(s.width_samples))
                * sps
            )
            in_t = (
                abs(int(cand.event_specnum) - int(s.event_specnum))
                <= dt_specnum_max
            )
            if not in_t:
                continue
            in_fdm = (
                abs(int(cand.dm_idx) - int(s.dm_idx)) <= dm_max
            )
            if not in_fdm:
                continue
            in_l = abs(float(cand.l) - float(s.l)) <= lm_max
            in_m = abs(float(cand.m) - float(s.m)) <= lm_max
            in_lm_or_cross = in_l or in_m
            if in_lm_or_cross:
                suppressed = True
                break
        if not suppressed:
            survivors.append(cand)
    return survivors


# ---------------------------------------------------------------------------
# Legacy axis-AND box merger (plan §4.4) — kept for chunk-2 unit tests
# + offline tools while the C1 path migrates to ``merge_across_kernels_c1``.
# ---------------------------------------------------------------------------


DEFAULT_MERGE_RADIUS_LM: int = 3
"""Plan §4.4 line 1589 legacy default merge radius for the (l, m) axes (cells)."""

DEFAULT_MERGE_RADIUS_FDM: int = 5
"""Plan §4.4 line 1589 legacy default merge radius for the fine-DM axis (trials)."""

DEFAULT_MERGE_RADIUS_T: int = 128
"""Plan §4.4 line 1589 legacy default merge radius for the time axis (samples)."""


def merge_across_kernels(
    candidates: List[Candidate],
    *,
    merge_radius_lm: int = DEFAULT_MERGE_RADIUS_LM,
    merge_radius_fdm: int = DEFAULT_MERGE_RADIUS_FDM,
    merge_radius_t: int = DEFAULT_MERGE_RADIUS_T,
) -> List[Candidate]:
    """Legacy cross-kernel SNR-sort + 4D merge-radius suppression
    (plan §4.4 line 1589). All four axes use independent half-window
    radii with AND semantics — a candidate is suppressed only if it
    is within the radius on ALL four (l, m, fdm, t) axes.

    Kept as a deprecation alias while the C1 path migrates to
    :func:`merge_across_kernels_c1`; the legacy default radii
    (lm=3, fdm=5, t=128) are wrong for the C1 geometry. New code
    should construct a :class:`MergerConfig` and call
    ``merge_across_kernels_c1`` instead.
    """
    if merge_radius_lm < 0 or merge_radius_fdm < 0 or merge_radius_t < 0:
        raise ValueError(
            f"merge_radius_lm/fdm/t must be ≥ 0; got "
            f"({merge_radius_lm}, {merge_radius_fdm}, {merge_radius_t})"
        )
    if not candidates:
        return []

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
                suppressed = True
                break
        if not suppressed:
            survivors.append(cand)
    return survivors
