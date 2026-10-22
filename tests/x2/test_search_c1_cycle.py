"""Per-cycle C1 pooling (x2/search/c1_cycle.py).

* With one DM per cycle at today's units the pool IS today's per-cube chain
  (search_compute._submit_c1_batch: de-rate, DM floor, width cap + escape +
  governor, metering) -- compared step for step with the production helpers.
* A flood over 6 DMs ships <= cap rows per cycle and the brightest DM's
  candidate always survives; governor tokens go to the brightest escape
  across DMs.
"""
from __future__ import annotations

import numpy as np

from dsart.x2.search.c1_cycle import C1CycleConfig, C1CyclePool
from dsart.x2.search.candidates import Candidate2x


def _cand(rng, d, tier, snr, width_ext, t, f=0, fine_global=None, l=None):
    return Candidate2x(half=0, d=d, tier=tier, fdm_local=f,
                       fine_global=fine_global if fine_global is not None else 30 * d + f,
                       dm=float(rng.uniform(100, 1500)), l=int(rng.integers(0, 256)) if l is None
                       else l, m=int(rng.integers(0, 256)), t_tier=t, event_specnum=t,
                       width_ext=width_ext, support_ext=width_ext, filt=f"b{width_ext}",
                       snr=float(snr))


def test_single_dm_1x_equals_todays_per_cube_chain():
    from dsart.services.c1_emit import WideEscapeGovernor
    from dsart.services.search_compute import (derate_noise_color, filter_unphysical_narrow,
                                               meter_candidates)

    rng = np.random.default_rng(0)
    for trial in range(5):
        widths = [1, 2, 4, 8, 16, 32, 64]
        cands = [_cand(rng, 0, "1x", rng.uniform(9.6, 45), int(rng.choice(widths)),
                     int(rng.integers(0, 192)), f=int(rng.integers(0, 25))) for _ in range(40)]
        pool = C1CyclePool(C1CycleConfig.today_1x())
        kept2x, rows = pool.filter(cands, now_s=100.0 + trial)
        # today, literally
        leg = [c.to_legacy(search_node_id=0, gpu_half=0, dm_index_mode="local") for c in cands]
        leg, _ = derate_noise_color(leg, 4.0, 8.0, t_search_us=1048.576)
        leg, _ = filter_unphysical_narrow(leg, 0.6, t_search_us=1048.576)
        kept = [c for c in leg if c.width_samples <= 16 or c.snr >= 20.0]
        narrow = [c for c in kept if c.width_samples <= 16]
        esc = [c for c in kept if c.width_samples > 16]
        if esc:
            adm, _ = WideEscapeGovernor(1, 3, 20.0).admit(esc, 100.0 + trial)
            kept = narrow + adm
        today, _ = meter_candidates(kept, 8, snr_protected_slots=2)
        key = lambda c: (c.l, c.m, c.dm_idx, c.event_specnum, c.width_samples,  # noqa: E731
                         round(c.snr, 9), c.kernel_id)
        assert [key(c) for c in rows] == [key(c) for c in today]
        assert len(kept2x) == len(rows)


def test_flood_over_six_dms_is_capped_per_cycle_and_keeps_the_brightest():
    rng = np.random.default_rng(1)
    pool = C1CyclePool(C1CycleConfig())
    for cycle in range(20):
        cands = []
        for d in range(6):
            for _ in range(30):
                cands.append(_cand(rng, d, "A" if d < 4 else "B", rng.uniform(9.6, 16),
                                   int(rng.choice([1, 2, 4, 8])) * (2 if d >= 4 else 1),
                                   int(rng.integers(0, 384))))
        bright_d = int(rng.integers(0, 6))
        bright = _cand(rng, bright_d, "A" if bright_d < 4 else "B", 75.0,
                       16 if bright_d < 4 else 32, 200, l=77)
        cands.insert(int(rng.integers(0, len(cands))), bright)
        kept, rows = pool.filter(cands, now_s=float(cycle) * 0.2013)
        assert len(rows) <= 8
        assert any(c is bright or (c.l == 77 and c.snr > 50) for c in kept)


def test_governor_tokens_go_to_the_brightest_escape_across_dms():
    rng = np.random.default_rng(2)
    pool = C1CyclePool(C1CycleConfig(escape_per_block=1))
    esc = [_cand(rng, d, "A", s, 64, 100, l=10 * d) for d, s in ((0, 25.0), (3, 60.0), (5, 30.0))]
    kept, _rows = pool.filter(esc, now_s=0.0)
    assert [c.snr for c in kept] == [60.0]
    assert pool.counters["escape_throttled"] == 2


def test_tier_b_units_in_derate_and_floor():
    """Tier-B candidates are judged in their own 1.048576 ms samples:
    the same physical width/DM keeps the same factor as at 1x."""
    from dsart.services.search_compute import boxcar_noise_color_factor

    rng = np.random.default_rng(3)
    c = _cand(rng, 40, "B", 30.0, 8, 500)                    # 8 T_ext = 4 tier-B samples
    pool = C1CyclePool(C1CycleConfig())
    assert pool._width_tier(c) == 4 and pool._tier_us(c) == 1048.576
    f = boxcar_noise_color_factor(c.dm, 4, chan_sum_factor=2, t_search_us=1048.576)
    kept, _ = pool.filter([c], now_s=0.0)
    applied = 1.0 + 4.0 * (f - 1.0)
    assert kept and abs(kept[0].snr - (30.0 / applied if applied > 1 else 30.0)) < 1e-9
