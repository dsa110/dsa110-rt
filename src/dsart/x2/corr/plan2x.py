"""Corr-side view of the tiered 2x DM plan (SPEC §1-§2).

The corr node needs only part of WS-P's ``TieredDmPlan``: per tier the
coarse DMs, the stream sample ``T_native`` (16 tier A, 32 tier B), the
stage-1 quantum ``q1_native`` (8 = the t_int-8 fast GEMM), and the
stage-1 bin table ``stage1_bins[g, k, d]``. Stage 2 is applied by the
search in 2x (corr stage-2 FIFO off), and fine DMs never reach the corr.

:func:`proposed_corr_plan` builds that view directly from the approved
op-point (``docs/timeres_study/snr2x/schemes.py::proposed``), using the
oracle's own arithmetic order so every rounding tie lands the same way:

    a1[g, k] = 1e3 * K * (1/nu[g,k]^2 - 1/nu_ref[g, s(k)]^2)   (us per pc/cc)
    nat      = rint(a1 * DM_c / 32.768)                        (native samples)
    bins     = rint(nat / q1_native)                           (q1 samples)

with ``nu`` the band-centre summed-channel table
(``build_summed_chgroup_freq_table_GHz(csf)``) and ``nu_ref`` the first
(highest) summed channel of each sub-band. :meth:`CorrPlan2x.
from_tiered_plan` adapts WS-P's ``TieredDmPlan`` once it lands
(SPEC §2 attribute names) so the corr code is indifferent to the source;
``tests/x2/test_corr2x_ref.py`` pins both against the oracle.

Global coarse index ``d``: tier A coarse 0..nA-1, then tier B. The owner
search half of ``d`` is ``d % 8`` (SPEC §2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import K_DM_MS_GHZ2_PC, N_CHGROUP, NATIVE_SAMPLE_US, NCHAN_PER_CHGROUP

__all__ = ["CorrTier", "CorrPlan2x", "proposed_corr_plan", "stage1_a1_us"]

#: Today's search sample (us) -- the reference cadence of the coarse-count rule.
T1_US = 1048.576


def stage1_a1_us(csf: int, n_sub: int) -> np.ndarray:
    """``(16, nsum)`` stage-1 delay per pc/cc (us) of each summed channel
    relative to its sub-band reference (the sub-band's first summed
    channel). >= 0; exactly the oracle's ``Scheme.a1``."""
    nus = build_summed_chgroup_freq_table_GHz(int(csf))              # (16, nsum)
    nsum = nus.shape[1]
    if nsum % n_sub:
        raise ValueError(f"n_sub={n_sub} must divide nsum={nsum}")
    per = nsum // n_sub
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
    a1_us: np.ndarray = field(repr=False)

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
        nat = np.rint(self.a1_us[:, :, None] * t.coarse_dm[None, None, :] / NATIVE_SAMPLE_US)
        return np.rint(nat / t.q1_native).astype(np.int64)

    def stage1_bins_global(self) -> np.ndarray:
        """``(16, nsum, n_coarse_total)`` over the global coarse index."""
        return np.concatenate([self.stage1_bins(i) for i in range(len(self.tiers))], axis=2)

    def max_stage1_bins(self, ti: int) -> int:
        return int(self.stage1_bins(ti).max())

    @classmethod
    def from_tiered_plan(cls, plan) -> "CorrPlan2x":
        """Adapter for WS-P's ``TieredDmPlan`` (SPEC §2 field names:
        ``csf``/``chan_sum_factor``, ``n_sub``, ``tiers[i].T_native``,
        ``.coarse_dm``, ``.q1_native``). The stage-1 table is recomputed
        here and must equal ``plan.stage1_bins`` (checked)."""
        csf = int(getattr(plan, "csf", getattr(plan, "chan_sum_factor", 0)))
        n_sub = int(plan.n_sub)
        tiers, d0 = [], 0
        names = "ABCDEFGH"
        for i, t in enumerate(plan.tiers):
            cd = np.asarray(t.coarse_dm, dtype=np.float64)
            lo = float(getattr(t, "dm_lo", cd[0]))
            hi = float(getattr(t, "dm_hi", cd[-1]))
            tiers.append(CorrTier(names[i], lo, hi, int(t.T_native), int(t.q1_native), cd, d0))
            d0 += cd.shape[0]
        out = cls(csf=csf, n_sub=n_sub, tiers=tuple(tiers), a1_us=stage1_a1_us(csf, n_sub))
        theirs = getattr(plan, "stage1_bins", None)
        if theirs is not None:
            theirs = np.asarray(theirs() if callable(theirs) else theirs)
            if not np.array_equal(theirs.astype(np.int64), out.stage1_bins_global()):
                raise ValueError("TieredDmPlan.stage1_bins disagrees with the corr view")
        return out


def _tier(name, lo, hi, T_native, q1_native, n_sub, d0) -> CorrTier:
    T_us = T_native * NATIVE_SAMPLE_US
    n_c = max(1, math.ceil(0.0206 * (hi - lo) * (T1_US / T_us) / n_sub))
    edges = np.linspace(lo, hi, n_c + 1)
    coarse = 0.5 * (edges[:-1] + edges[1:])
    return CorrTier(name, float(lo), float(hi), int(T_native), int(q1_native), coarse, int(d0))


def proposed_corr_plan(n_sub: int = 2, *, q1_native: int = 8, csf: int = 2) -> CorrPlan2x:
    """The approved 2x op-point (SPEC §1): tier A DM 50-1600 at 16 native,
    tier B 1600-3000 at 32 native, csf 2, stage-1 quantum ``q1_native``."""
    a = _tier("A", 50.0, 1600.0, 16, q1_native, n_sub, 0)
    b = _tier("B", 1600.0, 3000.0, 32, min(q1_native, 32), n_sub, a.n_coarse)
    return CorrPlan2x(csf=int(csf), n_sub=int(n_sub), tiers=(a, b),
                      a1_us=stage1_a1_us(csf, n_sub))


def _check_chgroups() -> None:  # pragma: no cover - import-time sanity
    assert N_CHGROUP == 16


_check_chgroups()
