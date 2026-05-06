"""Per-cube emit caps (plan §4.4 lines 1703-1704).

Two conditions:

  - ``PerCubePerKernelCap`` — at most ``max_per_kernel`` Candidates
    emitted per kernel triple per cube. Plan default 4.
  - ``PerCubeTotalCap`` — at most ``max_total`` Candidates emitted per
    cube across all kernels. Plan default 16.

Both read from the immutable ``TriggerContext`` (``cube_id``,
``cube_emitted_in_kernel``, ``cube_emitted_total``); the emitter
maintains the per-cube counters and rebuilds the context for every
candidate. Conditions themselves are stateless across cubes (the
per-cube state lives in the emitter / context).
"""

from __future__ import annotations

from ...common.contracts import Candidate
from ..predicate import TriggerContext, TriggerDecision

__all__ = ["PerCubePerKernelCap", "PerCubeTotalCap"]


class PerCubePerKernelCap:
    """Drops the Candidate if this cube has already emitted
    ``max_per_kernel`` candidates from the same kernel triple.
    """

    def __init__(
        self,
        *,
        max_per_kernel: int = 4,
        name: str = "per_cube_per_kernel_cap",
    ) -> None:
        if max_per_kernel < 1:
            raise ValueError(f"max_per_kernel={max_per_kernel}, expected ≥ 1")
        self.max_per_kernel = int(max_per_kernel)
        self.name = str(name)

    def evaluate(
        self,
        cand: Candidate,
        ctx: TriggerContext,
    ) -> TriggerDecision:
        already = ctx.cube_emitted_in_kernel.get(cand.kernel_id, 0)
        if already >= self.max_per_kernel:
            return TriggerDecision(
                emit=False,
                reason=f"kernel={cand.kernel_id}>={self.max_per_kernel}",
            )
        return TriggerDecision(emit=True)


class PerCubeTotalCap:
    """Drops the Candidate if this cube has already emitted
    ``max_total`` candidates total."""

    def __init__(
        self,
        *,
        max_total: int = 16,
        name: str = "per_cube_total_cap",
    ) -> None:
        if max_total < 1:
            raise ValueError(f"max_total={max_total}, expected ≥ 1")
        self.max_total = int(max_total)
        self.name = str(name)

    def evaluate(
        self,
        cand: Candidate,
        ctx: TriggerContext,
    ) -> TriggerDecision:
        if ctx.cube_emitted_total >= self.max_total:
            return TriggerDecision(
                emit=False,
                reason=f"cube_total>={self.max_total}",
            )
        return TriggerDecision(emit=True)
