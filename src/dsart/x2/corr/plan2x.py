"""Corr-side view of WS-P's tiered 2x DM plan (SPEC §2, frozen v1).

The corr node needs only part of :class:`dsart.x2.plan.TieredDmPlan`: per
tier the coarse DMs, the stream sample ``T_native`` (16 tier A, 32 tier
B), the stage-1 quantum ``q1_native`` (8 = the t_int-8 fast GEMM) and the
stage-1 table ``stage1_bins[g, k, d]`` (q1 units; the corr ADVANCES
summed channel ``k``: ``out[t] = vis_k[t + stage1_bins]``). Stage 2 is
the search's in 2x (corr stage-2 FIFO off); fine DMs never reach the corr.

:class:`CorrPlan2x` slices those arrays per tier. :func:`stage1_a1_us` is
an independent restatement of the oracle's stage-1 coefficient
(``docs/timeres_study/snr2x/schemes.py::Scheme.a1``) that the tests use
to cross-check the plan and the oracle against each other.

Global coarse index ``d``: tier A coarse 0..nA-1, then tier B; the owner
search half of ``d`` is ``d % 8``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import K_DM_MS_GHZ2_PC, NATIVE_SAMPLE_US, NCHAN_PER_CHGROUP

__all__ = ["CorrTier", "CorrPlan2x", "proposed_corr_plan", "stage1_a1_us"]


def stage1_a1_us(csf: int, n_sub: int) -> np.ndarray:
    """``(16, nsum)`` stage-1 delay per pc/cc (us) of each summed channel
    relative to its sub-band reference (the sub-band's first summed
    channel); the oracle's ``Scheme.a1`` restated independently."""
    nus = build_summed_chgroup_freq_table_GHz(int(csf))
    per = nus.shape[1] // n_sub
    refs = nus[:, ::per][:, :n_sub]
    return 1e3 * K_DM_MS_GHZ2_PC * (1.0 / nus ** 2 - 1.0 / np.repeat(refs, per, axis=1) ** 2)


@dataclass(frozen=True)
class CorrTier:
    name: str
    dm_lo: float
    dm_hi: float
    T_native: int
    q1_native: int
    coarse_dm: np.ndarray
    d0: int

    @property
    def n_coarse(self) -> int:
        return int(self.coarse_dm.shape[0])

    @property
    def decim(self) -> int:
        """q1 samples summed into one stream sample (2 tier A, 4 tier B)."""
        return self.T_native // self.q1_native

    @property
    def T_us(self) -> float:
        return self.T_native * NATIVE_SAMPLE_US


@dataclass(frozen=True)
class CorrPlan2x:
    csf: int
    n_sub: int
    tiers: tuple[CorrTier, ...]
    stage1_global: np.ndarray     # (16, nsum, n_coarse) int64, q1 units
    plan_id: int = 0

    @property
    def nsum(self) -> int:
        return NCHAN_PER_CHGROUP // self.csf

    @property
    def per(self) -> int:
        return self.nsum // self.n_sub

    @property
    def n_coarse_total(self) -> int:
        return sum(t.n_coarse for t in self.tiers)

    def tier_of(self, d: int) -> tuple[int, int]:
        """``(tier index, local coarse index)`` of global coarse ``d``."""
        for ti, t in enumerate(self.tiers):
            if t.d0 <= d < t.d0 + t.n_coarse:
                return ti, d - t.d0
        raise IndexError(f"coarse index {d} outside the plan")

    @staticmethod
    def owner_half(d: int) -> int:
        return int(d) % 8

    def stage1_bins(self, ti: int) -> np.ndarray:
        """``(16, nsum, n_coarse_tier)`` int64 stage-1 advance in q1 units."""
        t = self.tiers[ti]
        return self.stage1_global[:, :, t.d0:t.d0 + t.n_coarse]

    def stage1_bins_global(self) -> np.ndarray:
        return self.stage1_global

    @classmethod
    def from_tiered_plan(cls, plan) -> "CorrPlan2x":
        """From :class:`dsart.x2.plan.TieredDmPlan` (SPEC §2 field names)."""
        tiers = []
        off = np.asarray(plan.tier_coarse_offsets)
        for i in range(int(plan.n_tiers)):
            lo, hi = int(off[i]), int(off[i + 1])
            tiers.append(CorrTier(
                name=str(plan.tier_name[i]), dm_lo=float(plan.tier_dm_lo[i]),
                dm_hi=float(plan.tier_dm_hi[i]), T_native=int(plan.tier_t_native[i]),
                q1_native=int(plan.tier_q1_native[i]),
                coarse_dm=np.asarray(plan.coarse_dm[lo:hi], np.float64), d0=lo,
            ))
        return cls(csf=int(plan.chan_sum_factor), n_sub=int(plan.n_sub), tiers=tuple(tiers),
                   stage1_global=np.asarray(plan.stage1_bins, np.int64),
                   plan_id=int(plan.plan_id))


def proposed_corr_plan(n_sub: int = 2) -> CorrPlan2x:
    """The approved 2x op-point's corr view (WS-P ``build_tiered_plan``)."""
    from dsart.x2.plan import build_tiered_plan  # noqa: PLC0415
    return CorrPlan2x.from_tiered_plan(build_tiered_plan(n_sub=int(n_sub)))
