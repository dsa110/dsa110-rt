"""C1 stage per CYCLE per half (2x): pool every owned DM's candidates first.

Today ``search_compute._submit_c1_batch`` runs the C1 filter chain once per
cube: noise-colour de-rate -> DM-smearing floor -> width cap with the
brightness escape -> WideEscapeGovernor -> metering (cap 8, SNR-protected
slots). In 2x a half produces one cube per owned DM per cycle (3-6), so a
per-cube chain multiplies the fleet row ceiling and hands governor tokens to
whichever DM is processed first (WS-D ``tools/c2/rate_bench.py``: 1903 rows/s,
C2 at 9.4x real time, vs 320 rows/s pooled). :class:`C1CyclePool` runs the
SAME chain, with the SAME production helpers, once over the cycle's pooled
candidates, so tokens and slots go to the brightest across DMs. With one DM
per cycle and today's units it is exactly today's per-cube chain
(tests/x2/test_search_c1_cycle.py).

Units: the de-rate and the DM floor use each candidate's OWN tier sample
(tier B internal 1.048576 ms) and width in tier samples, with the op-point's
``chan_sum_factor`` (2 in 2x; today's hard-coded 8 in 1x); the width cap is
in external samples (32 T_ext = 16.8 ms in 2x, 16 in 1x). Metering and the
governor see the legacy ``Candidate`` rows that will be shipped.

The per-candidate steps (de-rate, DM floor) run before pooling (they are
element-wise); the pooled steps go through ``pool_cycle``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

from .candidates import Candidate2x

__all__ = ["C1CycleConfig", "C1CyclePool", "TIER_PERIOD_US"]

TIER_PERIOD_US: Dict[str, float] = {"A": 524.288, "B": 1048.576, "1x": 1048.576}


@dataclass(frozen=True)
class C1CycleConfig:
    chan_sum_factor: int = 2
    ext_period_us: float = 524.288
    noise_color_strength: Optional[float] = 4.0
    noise_color_snr_floor: Optional[float] = 8.0
    dm_width_floor_frac: Optional[float] = 0.6
    max_width_ext: Optional[int] = 32
    max_width_snr_escape: Optional[float] = 20.0
    escape_per_block: int = 1
    escape_burst: int = 3
    escape_refill_s: float = 20.0
    max_candidates_per_cycle: Optional[int] = 8
    snr_protected_slots: int = 2
    dm_index_mode: str = "global"

    @classmethod
    def today_1x(cls) -> "C1CycleConfig":
        """``configs/dsart_search_rt.yaml`` c1 block at today's op-point."""
        return cls(chan_sum_factor=8, ext_period_us=1048.576, max_width_ext=16,
                   dm_index_mode="local")


class C1CyclePool:
    def __init__(self, cfg: C1CycleConfig) -> None:
        from ...services.c1_emit import WideEscapeGovernor

        self.cfg = cfg
        self.gov = WideEscapeGovernor(cfg.escape_per_block, cfg.escape_burst,
                                      cfg.escape_refill_s)
        self.counters = dict(dropped_color=0, dropped_dmfloor=0, dropped_width=0,
                             width_escaped=0, escape_throttled=0, dropped_meter=0, cycles=0)

    def _tier_us(self, c: Candidate2x) -> float:
        return TIER_PERIOD_US.get(c.tier, self.cfg.ext_period_us)

    def _width_tier(self, c: Candidate2x) -> int:
        return max(1, int(round(c.width_ext * self.cfg.ext_period_us / self._tier_us(c))))

    def filter(self, cands: List[Candidate2x], *, now_s: float, search_node_id: int = 0,
               gpu_half: int = 0,
               cal_probe_predicate: Optional[Callable] = None) -> Tuple[List[Candidate2x], list]:
        """One cycle's pooled candidates -> ``(kept Candidate2x, legacy rows)``."""
        from ...services.search_compute import (
            boxcar_noise_color_factor,
            dm_smear_samples,
            meter_candidates,
        )

        cfg = self.cfg
        cnt = self.counters
        cnt["cycles"] += 1
        # 1. DM-aware noise-colour de-rate (per candidate, its own tier units)
        if cfg.noise_color_strength is not None and cfg.noise_color_strength > 0 and cands:
            floor = cfg.noise_color_snr_floor
            kept = []
            for c in cands:
                f = boxcar_noise_color_factor(float(c.dm), self._width_tier(c),
                                              chan_sum_factor=cfg.chan_sum_factor,
                                              t_search_us=self._tier_us(c))
                applied = 1.0 + float(cfg.noise_color_strength) * (f - 1.0)
                if applied <= 1.0:
                    kept.append(c)
                    continue
                snr = float(c.snr) / applied
                if floor is not None and snr < float(floor):
                    continue
                kept.append(replace(c, snr=snr))
            cnt["dropped_color"] += len(cands) - len(kept)
            cands = kept
        # 2. DM-smearing floor (impulsive RFI at high trial DM)
        if cfg.dm_width_floor_frac is not None and cfg.dm_width_floor_frac > 0 and cands:
            kept = [c for c in cands if float(self._width_tier(c)) >= cfg.dm_width_floor_frac
                    * dm_smear_samples(float(c.dm), chan_sum_factor=cfg.chan_sum_factor,
                                       t_search_us=self._tier_us(c))]
            cnt["dropped_dmfloor"] += len(cands) - len(kept)
            cands = kept
        # 3-4. width cap + brightness escape + governor + metering, ONCE per
        # cycle over the half's per-DM cubes: WS-D's D9 ``pool_cycle`` with
        # today's chain on the legacy rows that will be shipped
        from ...services.c1_emit import pool_cycle

        legacy = [c.to_legacy(search_node_id=search_node_id, gpu_half=gpu_half,
                              dm_index_mode=cfg.dm_index_mode) for c in cands]
        back = {id(lg): c for lg, c in zip(legacy, cands)}
        by_d: Dict[int, list] = {}
        for lg, c in zip(legacy, cands):
            by_d.setdefault(int(c.d), []).append(lg)
        groups = [(d, by_d[d]) for d in sorted(by_d)] or [(-1, [])]
        now = float(now_s)

        def chain(pooled: list) -> list:
            mw = cfg.max_width_ext
            if mw is not None and pooled:
                esc = (float(cfg.max_width_snr_escape) if cfg.max_width_snr_escape is not None
                       and float(cfg.max_width_snr_escape) > 0 else None)
                narrow = [c for c in pooled if int(c.width_samples) <= int(mw)]
                escapes = [c for c in pooled if int(c.width_samples) > int(mw)
                           and esc is not None and float(c.snr) >= esc]
                cnt["dropped_width"] += len(pooled) - len(narrow) - len(escapes)
                cnt["width_escaped"] += len(escapes)
                if escapes:
                    admitted, n_thr = self.gov.admit(escapes, now)
                    cnt["escape_throttled"] += n_thr
                    escapes = admitted
                pooled = narrow + escapes
            kept, n_met = meter_candidates(pooled, cfg.max_candidates_per_cycle,
                                           always_keep_predicate=cal_probe_predicate,
                                           snr_protected_slots=int(cfg.snr_protected_slots))
            cnt["dropped_meter"] += n_met
            return list(kept)

        kept_l = [c for _d, surv in pool_cycle(groups, chain) for c in surv]
        return [back[id(lg)] for lg in kept_l], list(kept_l)
