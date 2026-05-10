"""Trigger-emit predicate chain (plan §4.4 lines 1671-1714, CONC-3).

This is the single place where emit/drop is decided. The chain is an
ordered sequence of ``TriggerCondition`` instances; the first one to
return ``emit=False`` short-circuits the rest. Adding a new emit
condition is one new file under ``conditions/`` + one yaml line in
``config_compute_search.yaml::trigger_predicate_chain``; no other code
change.

This module owns the Protocol surface (``TriggerCondition``,
``TriggerContext``, ``TriggerDecision``) and the chain evaluator
(``evaluate_chain``). The conditions themselves live one-per-file
under ``dsart.trigger.conditions``.

State management contract: the chain evaluator is **stateless** — any
counter or running state lives inside the individual ``TriggerCondition``
instances (e.g. the rate-limit token bucket's current bucket level).
This means a single chain instance must not be shared across multiple
search-compute processes (each process owns its own emitter and chain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Protocol, runtime_checkable

from ..common.contracts import Candidate

__all__ = [
    "TriggerCondition",
    "TriggerContext",
    "TriggerDecision",
    "evaluate_chain",
]


@dataclass(frozen=True, slots=True)
class TriggerContext:
    """Per-evaluation context the chain may inspect (read-only, slotted
    so a ``TriggerCondition.evaluate`` cannot accidentally mutate it).

    Built once by the emitter for each Candidate, just before the chain
    runs. Immutable for the duration of one evaluation.
    """

    cube_id: int
    cube_emitted_in_kernel: Mapping[str, int]
    cube_emitted_total: int
    now_utc_ns: int


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Result of one TriggerCondition's ``evaluate``. ``reason`` is
    short-tag (the ``TriggerCondition.name`` is what's logged on a
    drop, but the per-condition reason is what's surfaced for the
    operator-facing dashboard suffix per plan line 1696)."""

    emit: bool
    reason: Optional[str] = None


@runtime_checkable
class TriggerCondition(Protocol):
    """Decides whether a Candidate emits. Stateless or self-contained
    stateful (no side effects beyond the returned reason + per-condition
    counters owned by the condition itself)."""

    name: str

    def evaluate(
        self,
        cand: Candidate,
        ctx: TriggerContext,
    ) -> TriggerDecision: ...


def evaluate_chain(
    conditions: List[TriggerCondition],
    cand: Candidate,
    ctx: TriggerContext,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Evaluate a chain of TriggerConditions against ``cand`` + ``ctx``.

    Short-circuits on the first ``emit=False`` returned. Returns
    ``(emit, condition_name, condition_reason)`` triple:

      - emit=True: chain passed → emit; condition_name and reason are
        both ``None`` (no condition rejected).
      - emit=False: chain dropped → condition_name is the
        ``TriggerCondition.name`` of the first condition that returned
        emit=False; reason is its ``TriggerDecision.reason`` (may be
        None if the condition didn't supply one — in which case the
        condition name doubles as the reason).
    """
    for cond in conditions:
        decision = cond.evaluate(cand, ctx)
        if not decision.emit:
            return False, cond.name, decision.reason
    return True, None, None
