"""Unit tests for the C2 cluster-rate limiter + sidereal (l,m) veto."""

from __future__ import annotations

from dsart.coinc.veto import (
    ARCSEC_TO_RAD,
    ClusterRateLimiter,
    SiderealVetoRegistry,
)


class _Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


# ---------------------------------------------------------------------------
# ClusterRateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_counts_within_window():
    clk = _Clock()
    rl = ClusterRateLimiter(window_s=60.0, max_clusters=100, now=clk)
    for _ in range(99):
        rl.record()
    assert rl.count() == 99
    assert not rl.exceeded()
    rl.record()
    assert rl.count() == 100
    assert rl.exceeded()


def test_rate_limiter_trims_old():
    clk = _Clock()
    rl = ClusterRateLimiter(window_s=60.0, max_clusters=5, now=clk)
    for _ in range(5):
        rl.record()
    assert rl.exceeded()
    clk.advance(61.0)
    # All old timestamps fall out of the window.
    assert rl.count() == 0
    assert not rl.exceeded()


def test_rate_limiter_disabled_when_max_nonpositive():
    clk = _Clock()
    rl = ClusterRateLimiter(window_s=60.0, max_clusters=0, now=clk)
    for _ in range(1000):
        rl.record()
    assert not rl.exceeded()


# ---------------------------------------------------------------------------
# SiderealVetoRegistry
# ---------------------------------------------------------------------------


def _reg(clk, **kw):
    tol_rad = kw.pop("tol_arcsec", 90.0) * ARCSEC_TO_RAD
    return SiderealVetoRegistry(
        tol_rad=tol_rad,
        min_hits=kw.pop("min_hits", 3),
        min_span_s=kw.pop("min_span_s", 60.0),
        expiry_s=kw.pop("expiry_s", 86400.0),
        now=clk,
    )


def test_veto_requires_min_hits_and_span():
    clk = _Clock(1000.0)
    reg = _reg(clk)
    l, m = 0.01, 0.02
    # hit 1
    assert reg.observe(l, m) is False
    assert not reg.is_vetoed(l, m)
    # hit 2 (still < min_hits)
    clk.advance(40.0)
    assert reg.observe(l, m) is False
    assert not reg.is_vetoed(l, m)
    # hit 3 but span only 40 s < 60 s -> NOT promoted yet
    clk.advance(10.0)  # span now 50 s
    assert reg.observe(l, m) is False
    assert not reg.is_vetoed(l, m)
    # hit 4 crossing the 60 s span -> promoted
    clk.advance(20.0)  # span now 70 s, 4 hits
    promoted = reg.observe(l, m)
    assert promoted is True
    assert reg.is_vetoed(l, m)
    assert len(reg.active_regions()) == 1


def test_veto_tolerance_box_match():
    clk = _Clock(1000.0)
    reg = _reg(clk, tol_arcsec=90.0)
    l, m = 0.0, 0.0
    half = 80.0 * ARCSEC_TO_RAD  # within 90"
    far = 200.0 * ARCSEC_TO_RAD  # outside 90"
    reg.observe(l, m); clk.advance(40)
    reg.observe(l + half, m); clk.advance(40)
    reg.observe(l, m + half)
    assert reg.is_vetoed(l, m)
    # A nearby position within tol is vetoed; a far one is not.
    assert reg.is_vetoed(l + half, m + half)
    assert not reg.is_vetoed(l + far, m + far)


def test_veto_rolling_expiry():
    clk = _Clock(1000.0)
    reg = _reg(clk, expiry_s=100.0)
    l, m = 0.05, -0.03
    reg.observe(l, m); clk.advance(40)
    reg.observe(l, m); clk.advance(40)
    reg.observe(l, m)  # 3 hits, span 80 s
    assert reg.is_vetoed(l, m)
    clk.advance(101.0)  # last hit now older than expiry
    assert not reg.is_vetoed(l, m)
    assert reg.active_regions() == []


def test_veto_persistence_roundtrip():
    clk = _Clock(1000.0)
    reg = _reg(clk)
    l, m = 0.012, 0.034
    reg.observe(l, m); clk.advance(40)
    reg.observe(l, m); clk.advance(40)
    reg.observe(l, m)
    assert reg.is_vetoed(l, m)
    payload = reg.to_full_payload()

    clk2 = _Clock(clk.t)
    reg2 = _reg(clk2)
    n = reg2.load_payload(payload)
    assert n == 1
    assert reg2.is_vetoed(l, m)


def test_veto_clear():
    clk = _Clock(1000.0)
    reg = _reg(clk)
    l, m = 0.0, 0.0
    reg.observe(l, m); clk.advance(40)
    reg.observe(l, m); clk.advance(40)
    reg.observe(l, m)
    assert reg.is_vetoed(l, m)
    dropped = reg.clear()
    assert dropped == 1
    assert not reg.is_vetoed(l, m)
    assert reg.active_regions() == []


def test_veto_one_off_does_not_veto():
    """A single bright cluster at a fresh (l,m) (real FRB) is never
    vetoed and leaves only an inactive accumulating region."""
    clk = _Clock(1000.0)
    reg = _reg(clk)
    assert reg.observe(0.1, 0.1) is False
    assert not reg.is_vetoed(0.1, 0.1)
    assert reg.active_regions() == []
    assert len(reg.all_regions()) == 1
