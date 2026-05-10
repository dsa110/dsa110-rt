"""Per-(l, m, kernel) holdoff state machine (plan §4.4 line 1718;
§2 line 265 "shared per-node holdoff" pin).

After firing on a Candidate, suppress further firings whose
(rounded l, rounded m, kernel_id) cell key matches for
``trigger_holdoff_ms = 50 ms``. This collapses the typical 3-5
sample-period peak structure that one FRB pulse fires across the
matched-filter response into one trigger per pulse.

**v1 single-process implementation**: this module ships an in-memory
dict keyed on the cell hash. Memory: ~64 B per active cell × ~few
hundred cells / cube × cube_cadence ≈ negligible. The plan §1718
mmap-shared posix-shm form (cross-GPU-half on the same search node)
is deferred to Chunk-6 ``services/search_compute.py`` where the
parent ``dsart-search-rx@<s>`` service owns the shm segment;
HoldoffStateMachine here is the in-process "v1 unit test" form that
``cube_injection_detector.py`` (Chunk 5) uses against
``MockTriggerListener``.

The ``HoldoffStateMachine`` API is the same in both forms — the
production single-shm wrapper just substitutes a CAS-on-shm cell
update for the dict assignment.

Cell-key construction: ``round(l, k_lm), round(m, k_lm), kernel_id``
per plan §1718. The default ``k_lm = 1`` (= round to nearest cell);
operators can tune via the constructor for sub-pixel-stable / coarser
holdoff cells.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from ..common.contracts import Candidate

__all__ = [
    "HoldoffCellKey",
    "HoldoffStateMachine",
    "DEFAULT_HOLDOFF_MS",
    "DEFAULT_LM_ROUND",
]


DEFAULT_HOLDOFF_MS: float = 50.0
"""Plan §4.4 line 1718 default holdoff window in milliseconds."""

DEFAULT_LM_ROUND: float = 1.0
"""Plan §4.4 line 1718 default round_l_m_factor (cells)."""


HoldoffCellKey = Tuple[int, int, str]
"""``(round(l / k_lm), round(m / k_lm), kernel_id)`` — int rounding so
the cell key is hashable. l, m carry integer pixel indices in v1; the
default round factor of 1.0 is therefore an identity."""


def make_cell_key(
    cand: Candidate,
    *,
    k_lm: float = DEFAULT_LM_ROUND,
) -> HoldoffCellKey:
    """Build the holdoff cell key for a Candidate per plan §4.4
    line 1718."""
    if k_lm <= 0:
        raise ValueError(f"k_lm={k_lm}, expected > 0")
    return (
        int(round(cand.l / k_lm)),
        int(round(cand.m / k_lm)),
        cand.kernel_id,
    )


class HoldoffStateMachine:
    """In-memory per-(l, m, kernel) holdoff state machine.

    Stateful: holds a dict ``{cell_key: last_emit_utc_ns}`` of every
    cell that has fired within the last ``holdoff_ms`` ms. The dict is
    pruned lazily on access — cells whose ``last_emit_utc_ns`` is
    older than ``now - holdoff_ms`` are dropped on the next
    ``check_and_register`` for that cell or via an explicit
    ``prune(now_utc_ns)``.

    Plan §4.4 line 1718 calls out a fixed-size hash-table of ~64k
    entries on production (shared via posix-shm); the v1 single-process
    in-memory form has unbounded-dict footprint but with lazy prune
    the working set is the active cell count (a few hundred at most
    per cube), so the dict stays small.

    Args:
        holdoff_ms: holdoff window in milliseconds. Default 50.
        k_lm: cell-key rounding factor for (l, m). Default 1.0
            (identity on integer-pixel-l/m).
    """

    def __init__(
        self,
        *,
        holdoff_ms: float = DEFAULT_HOLDOFF_MS,
        k_lm: float = DEFAULT_LM_ROUND,
    ) -> None:
        if holdoff_ms < 0:
            raise ValueError(f"holdoff_ms={holdoff_ms}, expected ≥ 0")
        if k_lm <= 0:
            raise ValueError(f"k_lm={k_lm}, expected > 0")
        self.holdoff_ms = float(holdoff_ms)
        self.k_lm = float(k_lm)
        self._cells: Dict[HoldoffCellKey, int] = {}

    @property
    def active_cells(self) -> int:
        """Number of cells currently tracked (telemetry / test
        introspection)."""
        return len(self._cells)

    def reset(self) -> None:
        """Clear all holdoff state."""
        self._cells.clear()

    def prune(self, now_utc_ns: int) -> int:
        """Drop cells whose last_emit_utc_ns is older than
        ``now - holdoff_ms``. Returns the count dropped. Run this
        opportunistically (e.g. once per cube) to bound the dict size
        if your candidate stream wanders across many cells."""
        cutoff = int(now_utc_ns) - int(self.holdoff_ms * 1e6)
        dropped = [k for k, t in self._cells.items() if t < cutoff]
        for k in dropped:
            del self._cells[k]
        return len(dropped)

    def is_suppressed(
        self,
        cand: Candidate,
        now_utc_ns: int,
    ) -> bool:
        """Check whether this Candidate's cell is currently in holdoff
        (read-only; does NOT register the candidate). Used by tests /
        telemetry; the production hot path uses
        ``check_and_register`` so the suppression check and the
        registration are atomic."""
        key = make_cell_key(cand, k_lm=self.k_lm)
        last = self._cells.get(key)
        if last is None:
            return False
        return (now_utc_ns - last) < int(self.holdoff_ms * 1e6)

    def check_and_register(
        self,
        cand: Candidate,
        now_utc_ns: int,
    ) -> bool:
        """Atomic check-and-register: if the Candidate's cell is in
        holdoff, return True (suppress); otherwise register the
        emission timestamp on the cell and return False (allow).

        Returns:
            True if the Candidate is suppressed by holdoff (caller
            should NOT emit), False if the candidate cleared holdoff
            (caller should proceed; the cell is now registered as
            emitted at ``now_utc_ns``).
        """
        key = make_cell_key(cand, k_lm=self.k_lm)
        last = self._cells.get(key)
        if last is not None and (now_utc_ns - last) < int(self.holdoff_ms * 1e6):
            return True
        self._cells[key] = int(now_utc_ns)
        return False
