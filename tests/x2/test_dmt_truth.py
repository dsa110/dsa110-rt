"""tools/x2/dmt_truth.py: the plan's own stage-1/stage-2 tables applied to a
dynamic spectrum reproduce the 2x DM-time geometry (R12 study, 2026-11-12)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PLAN = Path("/home/ubuntu/data/dm_plans/dm_plan_tiered_ns4.npz")
sys.path.insert(0, str(REPO / "tools" / "x2"))

pytestmark = pytest.mark.skipif(not PLAN.exists(), reason="needs the deployed 2x plan")


@pytest.fixture(scope="module")
def plan():
    from dsart.x2.plan import TieredDmPlan
    return TieredDmPlan.from_npz(PLAN)


def _burst(spec: np.ndarray, *, dm: float, n_q1: int, t_bot_q1: int, amp: float, seed: int = 0):
    import dmt_truth as T
    from dsart.coinc.dmt_ridge import K_DM_MS_GHZ2
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_q1, T.NG, T.NK)).astype(np.float32)
    nu = T._nu_summed()
    d_bot = K_DM_MS_GHZ2 * dm * (nu[-1, -1] ** -2 - nu[0, 0] ** -2) / T.Q1_MS
    starts = np.rint(t_bot_q1 - d_bot + K_DM_MS_GHZ2 * dm * (nu ** -2 - nu[0, 0] ** -2) / T.Q1_MS).astype(int)
    for dt in range(4):
        x[starts + dt, np.arange(T.NG)[:, None], np.arange(T.NK)[None, :]] += amp * spec
    return x


def test_windowed_emulation_equals_full(plan) -> None:
    import dmt_truth as T
    x = np.random.default_rng(1).normal(size=(2600, T.NG, T.NK)).astype(np.float32)
    full = T.emulate_2x(x, plan)
    win = T.emulate_2x(x, plan, dm_range=(100.0, 900.0), t_window=(700, 1100))
    sel = (plan.fine_dm >= 100.0) & (plan.fine_dm <= 900.0)
    a, b = full[sel, 700:1100], win[sel]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.mean() > 0.8 and np.allclose(a[both], b[both], atol=1e-3)
    assert not (np.isfinite(b) & ~np.isfinite(a)).any()
    assert np.isnan(win[~sel]).all()


def test_broadband_burst_lands_at_its_dm_and_bottom_referenced_time(plan) -> None:
    import dmt_truth as T
    dm, t_bot_ext = 600.0, 1200
    x = _burst(np.ones((T.NG, T.NK), np.float32), dm=dm, n_q1=2 * (t_bot_ext + 300), t_bot_q1=2 * t_bot_ext,
               amp=0.6)
    plane = T.emulate_2x(x, plan, dm_range=(dm - 300, dm + 300), t_window=(t_bot_ext - 100, t_bot_ext + 100))
    i, t = np.unravel_index(np.nanargmax(plane), plane.shape)
    assert abs(plan.fine_dm[i] - dm) < 3.0
    assert abs((t + t_bot_ext - 100) - (t_bot_ext + 1)) <= 2


def test_narrowband_burst_draws_its_frequency(plan) -> None:
    import dmt_truth as T
    from dsart.coinc.dmt_ridge import boxcar_rows, ridge_features, robust_rows
    nu = T._nu_summed() * 1e3
    spec = np.exp(-0.5 * ((nu - 1340.0) / 0.4) ** 2).astype(np.float32)
    dm, t_bot_ext = 500.0, 1100
    # ~15 of 3072 channels carry it: amp 30 -> band-sum S/N ~18 in a 1 ms boxcar
    x = _burst(spec, dm=dm, n_q1=2 * (t_bot_ext + 300), t_bot_q1=2 * t_bot_ext, amp=30.0, seed=2)
    lo = t_bot_ext - 256
    plane = T.emulate_2x(x, plan, dm_range=(dm - 700, dm + 700), t_window=(lo, lo + 512))
    z = robust_rows(boxcar_rows(plane, 2))
    i = int(np.argmin(np.abs(plan.fine_dm - dm)))
    t = 250 + int(np.nanargmax(z[i, 250:262]))
    f, _ = ridge_features(z, plan.fine_dm, dm_trig=float(plan.fine_dm[i]), t_trig=t, width=2)
    assert f.ok and f.freq_mhz == pytest.approx(1340.0, abs=2.0) and f.ridge_frac > 0.3
