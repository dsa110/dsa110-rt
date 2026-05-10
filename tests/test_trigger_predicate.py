"""Tests for ``dsart.trigger.predicate`` + the four v1 conditions
(M5 chunk 4; plan §4.4 lines 1671-1718).

Coverage:

  * SnrThreshold drops below threshold; emits at/above.
  * PerCubePerKernelCap drops after max_per_kernel emits per kernel
    triple per cube; resets on a new cube.
  * PerCubeTotalCap drops after max_total emits per cube.
  * RateLimitTokenBucket fires burst freely, then rate-limits.
  * evaluate_chain short-circuits on first emit=False.
  * Default chain wiring (the four conditions in plan-default order)
    behaves end-to-end as expected.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.trigger.conditions import (  # noqa: E402
    PerCubePerKernelCap,
    PerCubeTotalCap,
    RateLimitTokenBucket,
    SnrThreshold,
)
from dsart.trigger.predicate import (  # noqa: E402
    TriggerCondition,
    TriggerContext,
    TriggerDecision,
    evaluate_chain,
)


def _cand(*, snr: float = 9.0, kernel_id: str = "psf:d3:b16",
          dm_idx: int = 10, l: float = 4.0, m: float = 4.0,
          flags: int = int(CandidateFlags.NONE)) -> Candidate:
    return Candidate(
        l=l, m=m, dm_fine=float(dm_idx), dm_idx=dm_idx,
        event_specnum=256, width_samples=4,
        kernel_id=kernel_id, snr=snr, detector_version="v1.M5",
        flags=flags, search_node_id=0, gpu_half=0,
    )


def _ctx(*, cube_id: int = 0, total: int = 0,
         per_kernel: dict | None = None,
         now_utc_ns: int = 1_000_000_000) -> TriggerContext:
    return TriggerContext(
        cube_id=cube_id,
        cube_emitted_in_kernel=per_kernel or {},
        cube_emitted_total=total,
        now_utc_ns=now_utc_ns,
    )


# ---------------------------------------------------------------------------
# SnrThreshold
# ---------------------------------------------------------------------------


def test_snr_threshold_emits_at_or_above() -> None:
    cond = SnrThreshold(min_snr=8.0)
    assert cond.evaluate(_cand(snr=10.0), _ctx()).emit
    assert cond.evaluate(_cand(snr=8.0), _ctx()).emit


def test_snr_threshold_drops_below() -> None:
    cond = SnrThreshold(min_snr=8.0)
    decision = cond.evaluate(_cand(snr=7.5), _ctx())
    assert not decision.emit
    assert decision.reason is not None and "snr" in decision.reason


def test_snr_threshold_satisfies_protocol() -> None:
    assert isinstance(SnrThreshold(), TriggerCondition)


def test_snr_threshold_negative_min_rejected() -> None:
    with pytest.raises(ValueError, match="min_snr"):
        SnrThreshold(min_snr=-1.0)


# ---------------------------------------------------------------------------
# PerCubePerKernelCap
# ---------------------------------------------------------------------------


def test_per_kernel_cap_emits_under_limit() -> None:
    cond = PerCubePerKernelCap(max_per_kernel=2)
    ctx = _ctx(per_kernel={"psf:d3:b16": 1})
    assert cond.evaluate(_cand(kernel_id="psf:d3:b16"), ctx).emit


def test_per_kernel_cap_drops_at_limit() -> None:
    cond = PerCubePerKernelCap(max_per_kernel=2)
    ctx = _ctx(per_kernel={"psf:d3:b16": 2})
    assert not cond.evaluate(_cand(kernel_id="psf:d3:b16"), ctx).emit


def test_per_kernel_cap_independent_kernels() -> None:
    """Each kernel triple has its own counter."""
    cond = PerCubePerKernelCap(max_per_kernel=2)
    ctx = _ctx(per_kernel={"psf:d3:b16": 99, "unit:d1:b1": 0})
    # Kernel "psf:d3:b16" is over the cap…
    assert not cond.evaluate(_cand(kernel_id="psf:d3:b16"), ctx).emit
    # …but "unit:d1:b1" is fresh.
    assert cond.evaluate(_cand(kernel_id="unit:d1:b1"), ctx).emit


# ---------------------------------------------------------------------------
# PerCubeTotalCap
# ---------------------------------------------------------------------------


def test_per_cube_total_cap_emits_under_limit() -> None:
    cond = PerCubeTotalCap(max_total=4)
    assert cond.evaluate(_cand(), _ctx(total=3)).emit


def test_per_cube_total_cap_drops_at_limit() -> None:
    cond = PerCubeTotalCap(max_total=4)
    assert not cond.evaluate(_cand(), _ctx(total=4)).emit


# ---------------------------------------------------------------------------
# RateLimitTokenBucket
# ---------------------------------------------------------------------------


def test_rate_limit_burst_fires_freely() -> None:
    """The first ``burst`` triggers always emit (bucket starts full)."""
    cond = RateLimitTokenBucket(rate_per_s=10.0, burst=5)
    now = 1_000_000_000_000
    for _ in range(5):
        assert cond.evaluate(_cand(), _ctx(now_utc_ns=now)).emit
    # 6th drops (bucket empty, dt=0).
    assert not cond.evaluate(_cand(), _ctx(now_utc_ns=now)).emit


def test_rate_limit_refills_at_rate() -> None:
    """After waiting 1/rate_per_s seconds, exactly one new token is
    available."""
    cond = RateLimitTokenBucket(rate_per_s=10.0, burst=3)
    t0 = 1_000_000_000_000
    # Drain the bucket.
    for _ in range(3):
        assert cond.evaluate(_cand(), _ctx(now_utc_ns=t0)).emit
    # No tokens left; this drops.
    assert not cond.evaluate(_cand(), _ctx(now_utc_ns=t0)).emit
    # Wait 100 ms (= 1 / 10 = exactly 1 token at rate=10/s).
    t1 = t0 + 100_000_000
    assert cond.evaluate(_cand(), _ctx(now_utc_ns=t1)).emit
    # The next one drops again.
    assert not cond.evaluate(_cand(), _ctx(now_utc_ns=t1)).emit


def test_rate_limit_caps_at_burst() -> None:
    """Bucket is capped at ``burst``; long idle doesn't accumulate
    extra tokens."""
    cond = RateLimitTokenBucket(rate_per_s=10.0, burst=3)
    t0 = 1_000_000_000_000
    cond.evaluate(_cand(), _ctx(now_utc_ns=t0))  # warm up timestamp
    # Wait 30 s (way more than burst/rate = 0.3 s).
    t1 = t0 + 30_000_000_000
    fires = 0
    while True:
        d = cond.evaluate(_cand(), _ctx(now_utc_ns=t1))
        if not d.emit:
            break
        fires += 1
        if fires > 100:
            pytest.fail("bucket never drained — cap broken?")
    # We fired the bucket size minus the one we used at t0 plus one
    # extra at t1 (the remaining 2 from the original burst):
    # actually: start full at burst=3, use 1 at t0 → 2 left, +30s gives
    # 30·10 = 300 tokens uncapped, but capped at 3, so we should have
    # 3 tokens at t1. Fires = 3.
    assert fires == 3


def test_rate_limit_handles_clock_backwards() -> None:
    """If now_utc_ns goes backwards (NTP step), the bucket clamps Δt
    to 0 and doesn't refill from a backwards step."""
    cond = RateLimitTokenBucket(rate_per_s=10.0, burst=2)
    t0 = 1_000_000_000_000
    cond.evaluate(_cand(), _ctx(now_utc_ns=t0))
    cond.evaluate(_cand(), _ctx(now_utc_ns=t0))
    # Now go backwards by 1 s.
    t1 = t0 - 1_000_000_000
    # Bucket should still be empty.
    assert not cond.evaluate(_cand(), _ctx(now_utc_ns=t1)).emit


def test_rate_limit_reset_refills_bucket() -> None:
    cond = RateLimitTokenBucket(rate_per_s=10.0, burst=3)
    for _ in range(3):
        cond.evaluate(_cand(), _ctx())
    assert cond.tokens < 1.0
    cond.reset()
    assert cond.tokens == 3.0


# ---------------------------------------------------------------------------
# evaluate_chain
# ---------------------------------------------------------------------------


def test_evaluate_chain_passes_through() -> None:
    chain = [
        SnrThreshold(min_snr=8.0),
        PerCubePerKernelCap(max_per_kernel=4),
        PerCubeTotalCap(max_total=16),
    ]
    emit, name, reason = evaluate_chain(chain, _cand(snr=10.0), _ctx())
    assert emit
    assert name is None
    assert reason is None


def test_evaluate_chain_short_circuits_on_first_drop() -> None:
    """The PerKernelCap drops; later conditions are not evaluated."""
    last_call_count = {"n": 0}

    class _CountingCond:
        name = "counter"

        def evaluate(self, cand, ctx) -> TriggerDecision:
            last_call_count["n"] += 1
            return TriggerDecision(emit=True)

    chain = [
        SnrThreshold(min_snr=8.0),
        PerCubePerKernelCap(max_per_kernel=2),
        _CountingCond(),
    ]
    ctx = _ctx(per_kernel={"psf:d3:b16": 99})
    emit, name, reason = evaluate_chain(chain, _cand(snr=10.0), ctx)
    assert not emit
    assert name == "per_cube_per_kernel_cap"
    assert reason is not None
    assert last_call_count["n"] == 0  # short-circuited


def test_default_chain_v1_basic_path() -> None:
    """The v1 default chain emits a 10σ candidate and rejects a 6σ one."""
    chain = [
        SnrThreshold(min_snr=8.0),
        PerCubePerKernelCap(max_per_kernel=4),
        PerCubeTotalCap(max_total=16),
        RateLimitTokenBucket(rate_per_s=10.0, burst=50),
    ]
    emit, name, _ = evaluate_chain(chain, _cand(snr=10.0), _ctx())
    assert emit
    emit, name, _ = evaluate_chain(chain, _cand(snr=6.0), _ctx())
    assert not emit
    assert name == "snr_threshold"
