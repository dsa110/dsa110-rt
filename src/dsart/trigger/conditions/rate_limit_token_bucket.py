"""Emit-rate token-bucket condition (plan §4.4 line 1705-1718).

A standard token bucket: ``rate_per_s`` tokens added per second,
capped at ``burst`` tokens; each emit costs 1 token. Drops when the
bucket is empty. Plan defaults: ``rate_per_s=10, burst=50``.

The bucket is **stateful per condition instance** — the chain owner
must hold one instance for the lifetime of the search-compute process.
The emitter constructs the chain once at startup; conditions persist
across cubes.

Time source: ``ctx.now_utc_ns``. The emitter is responsible for
supplying a monotonic ``now`` (drift across DST or NTP-step is OK
because the rate window is bounded; if ``now`` ever goes backwards by
more than ``2 / rate_per_s``, the implementation clamps Δt to 0 to
prevent a negative bucket fill).
"""

from __future__ import annotations

from ...common.contracts import Candidate
from ..predicate import TriggerContext, TriggerDecision

__all__ = ["RateLimitTokenBucket"]


class RateLimitTokenBucket:
    """Drops candidates above the configured emit rate.

    Args:
        rate_per_s: token refill rate (tokens added per second).
            Default 10.
        burst: bucket capacity (max tokens). Default 50.
        name: short identifier. Default ``"ratelimit"`` (the
            condition-name plan §1718 / §3 line 383 reserves the wire
            ack-reason ``"ratelimit"`` for; keeping the condition's
            name aligned makes the offline log self-explaining).
    """

    def __init__(
        self,
        *,
        rate_per_s: float = 10.0,
        burst: int = 50,
        name: str = "ratelimit",
    ) -> None:
        if not (rate_per_s > 0):
            raise ValueError(f"rate_per_s={rate_per_s}, expected > 0")
        if burst < 1:
            raise ValueError(f"burst={burst}, expected ≥ 1")
        self.rate_per_s = float(rate_per_s)
        self.burst = float(burst)
        self.name = str(name)
        # Bucket starts full so the very first ``burst`` triggers always
        # emit (matches plan §4.4 line 1718's "fire 100 triggers/s for
        # 30 s, only burst + rate_per_s × 30 are sent" expected
        # behaviour).
        self._tokens = float(burst)
        self._last_refill_utc_ns: int | None = None

    @property
    def tokens(self) -> float:
        """Current bucket level (cloned from internal state for
        telemetry / test introspection)."""
        return float(self._tokens)

    def reset(self) -> None:
        """Refill bucket; clear the last-refill timestamp. Used when the
        emitter restarts and the operator wants a clean slate."""
        self._tokens = float(self.burst)
        self._last_refill_utc_ns = None

    def evaluate(
        self,
        cand: Candidate,
        ctx: TriggerContext,
    ) -> TriggerDecision:
        now = int(ctx.now_utc_ns)
        if self._last_refill_utc_ns is None:
            dt_s = 0.0
        else:
            dt_ns = now - self._last_refill_utc_ns
            # Clamp negative dt (clock went backwards) to 0; the bucket
            # is conservative — never refill from a backwards step.
            dt_s = max(0.0, dt_ns * 1e-9)
        self._last_refill_utc_ns = now
        self._tokens = min(self.burst, self._tokens + self.rate_per_s * dt_s)

        if self._tokens < 1.0:
            return TriggerDecision(emit=False, reason="ratelimit")
        self._tokens -= 1.0
        return TriggerDecision(emit=True)
