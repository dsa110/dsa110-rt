"""Unit tests for the C2 cluster-rate limiter + sidereal (l,m) veto."""

from __future__ import annotations

from dsart.coinc.veto import (
    ARCSEC_TO_RAD,
    ClusterRateLimiter,
    SiderealVetoRegistry,
    dm_comb_detected,
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


# ---------------------------------------------------------------------------
# dm_comb_detected
# ---------------------------------------------------------------------------

_LM_TOL = 90.0 * ARCSEC_TO_RAD


def _comb_kw(**over):
    kw = dict(
        lm_tol_rad=_LM_TOL, dt_s=2.0, min_clusters=3, dm_span_min=300.0,
    )
    kw.update(over)
    return kw


def test_dm_comb_detected_broadband():
    """7 co-located clusters spanning DM 165..1162 within 0.55 s -> comb
    (the 2026-06-14 08:36:04 event)."""
    l, m, t0 = 0.02, -0.01, 1000.0
    dms = [165.0, 296.0, 430.0, 561.0, 800.0, 980.0, 1162.0]
    sibs = [(l, m, dm, t0 + 0.08 * i) for i, dm in enumerate(dms)]
    assert dm_comb_detected(l, m, t0, sibs, **_comb_kw())


def test_dm_comb_single_dm_repeater_safe():
    """Many co-located clusters at ~one DM (a bright pulsar) is NOT a
    comb -> the DM-span gate keeps it safe."""
    l, m, t0 = 0.02, -0.01, 1000.0
    sibs = [(l, m, 158.0 + 5.0 * i, t0 + 0.05 * i) for i in range(8)]
    assert not dm_comb_detected(l, m, t0, sibs, **_comb_kw())


def test_dm_comb_isolated_burst_safe():
    """A lone dispersed burst (its own cluster only) is never a comb."""
    l, m, t0 = 0.02, -0.01, 1000.0
    sibs = [(l, m, 700.0, t0)]
    assert not dm_comb_detected(l, m, t0, sibs, **_comb_kw())


def test_dm_comb_requires_min_clusters():
    """Wide DM span but only 2 co-located clusters -> below min_clusters."""
    l, m, t0 = 0.02, -0.01, 1000.0
    sibs = [(l, m, 200.0, t0), (l, m, 900.0, t0 + 0.1)]
    assert not dm_comb_detected(l, m, t0, sibs, **_comb_kw())


def test_dm_comb_spatial_separation_excluded():
    """Clusters far in (l,m) are not counted even with wide DM spread."""
    l, m, t0 = 0.02, -0.01, 1000.0
    far = m + 10.0 * _LM_TOL
    sibs = [
        (l, m, 200.0, t0),
        (l, far, 600.0, t0 + 0.1),
        (l, far, 1100.0, t0 + 0.2),
    ]
    assert not dm_comb_detected(l, m, t0, sibs, **_comb_kw())


def test_dm_comb_temporal_separation_excluded():
    """Clusters outside dt_s are not part of the same comb."""
    l, m, t0 = 0.02, -0.01, 1000.0
    sibs = [
        (l, m, 200.0, t0),
        (l, m, 600.0, t0 + 5.0),
        (l, m, 1100.0, t0 + 9.0),
    ]
    assert not dm_comb_detected(l, m, t0, sibs, **_comb_kw())


def test_dm_comb_disabled_by_zero_min_clusters():
    l, m, t0 = 0.02, -0.01, 1000.0
    sibs = [(l, m, 165.0 + 200.0 * i, t0 + 0.1 * i) for i in range(5)]
    assert not dm_comb_detected(l, m, t0, sibs, **_comb_kw(min_clusters=0))
