"""SNR threshold trigger condition (plan §4.4 line 1702).

Drops candidates with ``snr < min_snr``. Stateless. Default
``min_snr = 8.0`` per ``config_compute_search.yaml`` and the §4.4
empirical FAR analysis.
"""

from __future__ import annotations

from ...common.contracts import Candidate
from ..predicate import TriggerContext, TriggerDecision

__all__ = ["SnrThreshold"]


class SnrThreshold:
    """Drops candidates whose ``snr`` is below ``min_snr``.

    Args:
        min_snr: minimum SNR to emit. Default 8.0 (the production
            ``detector_threshold_sigma``). Plan §4.4 line 1700-1702.
        name: short identifier surfaced in /mon counters and the
            ndjson ``predicate_reason`` field (default
            ``"snr_threshold"``).
    """

    def __init__(self, *, min_snr: float = 8.0, name: str = "snr_threshold") -> None:
        if not (min_snr > 0):
            raise ValueError(f"min_snr={min_snr}, expected > 0")
        self.min_snr = float(min_snr)
        self.name = str(name)

    def evaluate(
        self,
        cand: Candidate,
        ctx: TriggerContext,
    ) -> TriggerDecision:
        if cand.snr < self.min_snr:
            return TriggerDecision(
                emit=False, reason=f"snr<{self.min_snr}",
            )
        return TriggerDecision(emit=True)
