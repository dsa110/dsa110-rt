"""Owner geometry for the 2x search engine (SPEC §2, §7, §8).

Everything the owner engine needs to know about the DMs one search half
owns, in one place and independent of where the numbers came from:

* :class:`DmSpec` -- one owned coarse DM: tier, tier sample in native
  samples, its fine DMs (global indices + values) and the per-(fine DM,
  stream) integer stage-2 shift the search applies (out row ``t`` of fine
  DM ``f`` reads stream ``s`` at row ``t - shifts[f, s]``).
* :class:`OwnerPlan` -- the DMs of ONE half + the clock conventions
  (cycle 201.326592 ms aligned to absolute native sample, external sample).

Builders:

* :func:`owner_plan_from_tiered` -- from the frozen WS-P ``TieredDmPlan``
  (``src/dsart/x2/plan.py``, SPEC §2); **the per-DM n_fdm always comes
  from the plan** (the hand-edited ``--n-fdm`` trap of 2026-09-23 cannot
  recur).
* :func:`owner_plan_from_tiers` -- small synthetic plans for tests: per-tier
  (T_native, coarse, fine, f2c) lists; stage-2 bins from the sub-band
  reference table exactly as SPEC §2 defines them.
* :func:`owner_plan_1x` -- today's op-point (one coarse DM per half,
  merged-rounding signed shifts from the production
  ``compute_time_shift_search``), for 1x-compat.

Clock (SPEC §8): absolute native sample index ``n``; a tier sample is
``t_native`` native samples (16 tier A, 32 tier B / 1x); the external
sample downstream is ``ext_native`` native samples (16 in 2x, 32 in 1x).
A cycle is ``CYCLE_NATIVE = 6144`` native samples = 384 tier-A / 192
tier-B / 192 1x samples, and cycle ``k`` starts at native ``k * 6144``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ...common.constants import NATIVE_SAMPLE_US, N_CHGROUP, NU_BOT_PROC_GHZ

__all__ = [
    "CYCLE_NATIVE",
    "N_HALVES",
    "SEARCH_NODES",
    "DmSpec",
    "OwnerPlan",
    "owner_half",
    "half_node_gpu",
    "stage2_bins_subband",
    "owner_plan_from_tiers",
    "owner_plan_from_tiered",
    "owner_plan_1x",
]

#: One search cycle in native samples: 201.326592 ms = 384 x 0.524288 ms
#: = 192 x 1.048576 ms. Cycles are aligned to absolute native sample 0.
CYCLE_NATIVE: int = 6144

#: Search halves (4 nodes x 2 GPUs). Coarse DM d is owned by half d % 8.
N_HALVES: int = 8

#: N_SEARCH order of the search nodes (SPEC §2): half h -> node h // 2.
SEARCH_NODES: tuple = ("n01", "n02", "n09", "n13")


def owner_half(d: int, n_halves: int = N_HALVES) -> int:
    """Owner half of global coarse DM index ``d`` (SPEC §2: ``d % 8``)."""
    return int(d) % int(n_halves)


def half_node_gpu(half: int) -> tuple:
    """``(node name, gpu index)`` of search half ``half``."""
    return SEARCH_NODES[int(half) // 2], int(half) % 2


@dataclass(frozen=True)
class DmSpec:
    """One owned coarse DM, as the engine consumes it.

    Attributes:
        d: global coarse DM index (tier A first, then tier B; 1x: the
            plan's coarse index).
        tier: ``"A"``, ``"B"`` or ``"1x"``.
        t_native: native samples per tier sample (16, 32).
        coarse_dm: coarse DM value (pc/cc).
        fine_dm: ``[n_fdm]`` fine DM values (pc/cc), the plan's order.
        fine_global: ``[n_fdm]`` global fine-DM index (C1 ``dm_idx`` in 2x).
        shifts: ``[n_fdm, n_stream]`` int32. Output row ``t`` of fine DM
            ``f`` reads stream ``s`` at row ``t - shifts[f, s]`` (the
            production ``combine_sparse_v4`` convention). Non-negative in
            2x (stage-2 delays to nu_bot); signed in 1x (merged rounding).
    """

    d: int
    tier: str
    t_native: int
    coarse_dm: float
    fine_dm: np.ndarray
    fine_global: np.ndarray
    shifts: np.ndarray

    def __post_init__(self) -> None:
        sh = np.ascontiguousarray(self.shifts, dtype=np.int32)
        object.__setattr__(self, "shifts", sh)
        object.__setattr__(self, "fine_dm",
                           np.ascontiguousarray(self.fine_dm, dtype=np.float64))
        object.__setattr__(self, "fine_global",
                           np.ascontiguousarray(self.fine_global, dtype=np.int64))
        if sh.ndim != 2 or sh.shape[0] != self.fine_dm.shape[0]:
            raise ValueError(
                f"DM {self.d}: shifts {sh.shape} vs {self.fine_dm.shape[0]} fine DMs")
        if self.fine_global.shape != self.fine_dm.shape:
            raise ValueError(f"DM {self.d}: fine_global/fine_dm length mismatch")
        if CYCLE_NATIVE % int(self.t_native):
            raise ValueError(f"t_native={self.t_native} does not divide the cycle")

    @property
    def n_fdm(self) -> int:
        return int(self.shifts.shape[0])

    @property
    def n_stream(self) -> int:
        return int(self.shifts.shape[1])

    @property
    def hop(self) -> int:
        """Tier samples per cycle (384 tier A, 192 tier B / 1x)."""
        return CYCLE_NATIVE // int(self.t_native)

    @property
    def max_pos(self) -> int:
        """History rows the combine reads before an output row."""
        return int(max(0, int(self.shifts.max(initial=0))))

    @property
    def max_neg(self) -> int:
        """Future rows the combine reads after an output row (1x only)."""
        return int(max(0, -int(self.shifts.min(initial=0))))

    def cycle_rows(self, k: int) -> tuple:
        """Absolute tier-sample rows ``[t0, t0 + hop)`` of cycle ``k``."""
        t0 = int(k) * self.hop
        return t0, t0 + self.hop


@dataclass(frozen=True)
class OwnerPlan:
    """The DMs owned by ONE search half plus the clock conventions."""

    half: int
    dms: tuple
    n_stream: int
    ext_native: int = 16
    opoint: str = "2x"
    plan_id: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for spec in self.dms:
            if spec.n_stream != self.n_stream:
                raise ValueError(
                    f"DM {spec.d}: {spec.n_stream} streams, plan says {self.n_stream}")
            if spec.t_native % self.ext_native:
                raise ValueError(
                    f"DM {spec.d}: tier sample {spec.t_native} is not a whole number "
                    f"of external samples ({self.ext_native})")

    def by_d(self, d: int) -> DmSpec:
        for spec in self.dms:
            if spec.d == int(d):
                return spec
        raise KeyError(d)

    @property
    def ext_period_us(self) -> float:
        return self.ext_native * NATIVE_SAMPLE_US

    def ext_factor(self, spec: DmSpec) -> int:
        """External samples per tier sample (1 tier A / 1x, 2 tier B)."""
        return int(spec.t_native) // int(self.ext_native)

    def summary(self) -> dict:
        return dict(
            half=self.half, opoint=self.opoint, plan_id=self.plan_id,
            n_stream=self.n_stream, ext_native=self.ext_native,
            dms=[dict(d=s.d, tier=s.tier, t_native=s.t_native, n_fdm=s.n_fdm,
                      hop=s.hop, max_pos=s.max_pos, max_neg=s.max_neg,
                      coarse_dm=s.coarse_dm) for s in self.dms],
        )


# ---------------------------------------------------------------------------
# stage-2 shifts (SPEC §2) -- the search applies ONE rounding at the fine DM
# ---------------------------------------------------------------------------


def stage2_bins_subband(fine_dm: np.ndarray, *, t_tier_us: float, csf: int,
                        n_sub: int) -> np.ndarray:
    """``[n_fine, 16 * n_sub]`` int32 = rint(a2[g, s] * DM_f / T_tier).

    ``a2`` is the dispersion delay from the sub-band reference (the
    sub-band's first summed channel, ``subband_ref_freqs_table_GHz``) down
    to nu_bot_proc, i.e. what ``docs/timeres_study/snr2x/snr_model.py::
    Scheme.stage2_bins`` computes. Stand-in for WS-P's ``stage2_bins``
    until ``x2/plan.py`` lands; the engine itself never recomputes shifts
    when a plan supplies them.
    """
    from ...coarse_dm.dm_plan import subband_ref_freqs_table_GHz
    from ...common.constants import K_DM_MS_GHZ2_PC

    refs = subband_ref_freqs_table_GHz(int(csf), int(n_sub))          # (16, n_sub)
    a2 = 1e3 * K_DM_MS_GHZ2_PC * (1.0 / NU_BOT_PROC_GHZ ** 2 - 1.0 / refs ** 2)
    a2 = a2.reshape(-1)                                               # stream = g*n_sub+s
    fine = np.asarray(fine_dm, dtype=np.float64)
    return np.rint(a2[None, :] * fine[:, None] / float(t_tier_us)).astype(np.int32)


def _plan_id(parts: Iterable) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(np.asarray(p).tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()


def owner_plan_from_tiers(
    tiers: Sequence[dict],
    *,
    half: int,
    n_sub: int,
    csf: int = 2,
    n_halves: int = N_HALVES,
    ext_native: int = 16,
    shifts_override: Optional[Sequence[np.ndarray]] = None,
) -> OwnerPlan:
    """Owner plan for ``half`` from per-tier DM lists.

    Args:
        tiers: one dict per tier, in global order (A then B), with keys
            ``t_native`` (16 / 32), ``coarse`` ``[n_c]``, ``fine``
            ``[n_f]``, ``f2c`` ``[n_f]`` (local coarse index), optional
            ``name``.
        half: which half's DMs to keep (global coarse ``d % n_halves``).
        shifts_override: optional per-tier ``[n_f, n_stream]`` stage-2
            tables (a plan's own ``stage2_bins``); computed here otherwise.
    """
    dms: List[DmSpec] = []
    d0 = 0
    f0 = 0
    n_stream = N_CHGROUP * int(n_sub)
    ids = []
    for ti, tier in enumerate(tiers):
        tn = int(tier["t_native"])
        coarse = np.asarray(tier["coarse"], dtype=np.float64)
        fine = np.asarray(tier["fine"], dtype=np.float64)
        f2c = np.asarray(tier["f2c"], dtype=np.int64)
        if shifts_override is not None:
            s2 = np.asarray(shifts_override[ti], dtype=np.int32)
        else:
            s2 = stage2_bins_subband(fine, t_tier_us=tn * NATIVE_SAMPLE_US,
                                     csf=csf, n_sub=n_sub)
        if s2.shape != (fine.shape[0], n_stream):
            raise ValueError(f"tier {ti}: stage-2 table {s2.shape}")
        ids += [tn, coarse, fine, f2c, s2]
        for c in range(coarse.shape[0]):
            d = d0 + c
            if owner_half(d, n_halves) != int(half):
                continue
            sel = np.nonzero(f2c == c)[0]
            if sel.size == 0:
                raise ValueError(f"tier {ti} coarse {c}: no fine DMs (plan build failed)")
            dms.append(DmSpec(
                d=d, tier=str(tier.get("name", "AB"[ti] if ti < 2 else str(ti))),
                t_native=tn, coarse_dm=float(coarse[c]), fine_dm=fine[sel],
                fine_global=f0 + sel, shifts=s2[sel],
            ))
        d0 += coarse.shape[0]
        f0 += fine.shape[0]
    return OwnerPlan(half=int(half), dms=tuple(dms), n_stream=n_stream,
                     ext_native=int(ext_native), opoint="2x",
                     plan_id=_plan_id(ids + [n_sub, csf]))


def owner_plan_from_tiered(plan, *, half: int, ext_native: int = 16) -> OwnerPlan:
    """Owner plan of ``half`` from a frozen WS-P ``TieredDmPlan`` (SPEC §2).

    Uses only the frozen API: ``dms_of_half``, ``stage2_for_coarse`` (read
    convention ``stream[t - stage2_bins]``), ``fine_offset`` (global fine
    index = plan order = C1 ``dm_idx_global``), ``coarse_t_native``,
    ``coarse_tier``, ``tier_name``, ``plan_id``. ``n_fdm`` per DM comes from
    the plan (never a hand-edited ``--n-fdm``). Shifts are never recomputed.
    """
    dms: List[DmSpec] = []
    for d in plan.dms_of_half(int(half)):
        d = int(d)
        sl = plan.fines_of(d)
        fg = np.arange(sl.start, sl.stop, dtype=np.int64)
        dms.append(DmSpec(
            d=d, tier=str(plan.tier_name[int(plan.coarse_tier[d])]),
            t_native=int(plan.coarse_t_native[d]), coarse_dm=float(plan.coarse_dm[d]),
            fine_dm=np.asarray(plan.fine_dm[sl]), fine_global=fg,
            shifts=np.asarray(plan.stage2_for_coarse(d)),
        ))
        if dms[-1].n_fdm != int(plan.n_fdm[d]):
            raise ValueError(f"DM {d}: plan n_fdm {int(plan.n_fdm[d])} != fines {dms[-1].n_fdm}")
    if not dms:
        raise ValueError(f"half {half} owns no DMs in this plan")
    return OwnerPlan(half=int(half), dms=tuple(dms), n_stream=int(plan.n_streams),
                     ext_native=int(ext_native), opoint="2x", plan_id=f"{int(plan.plan_id):016x}",
                     extra=dict(n_sub=int(plan.n_sub), chan_sum_factor=int(plan.chan_sum_factor)))


def owner_plan_1x(
    *,
    coarse_dm: np.ndarray,
    fine_dm: np.ndarray,
    fine_to_coarse: np.ndarray,
    owned: Sequence[int],
    half: int = 0,
    n_sub: int = 4,
    csf: int = 8,
    t_native: int = 32,
    symmetric: bool = True,
) -> OwnerPlan:
    """Today's op-point: one plan, coarse DMs ``owned`` by this half.

    Shifts are exactly the production table
    (``compute_time_shift_search(merge_coarse_rounding=True, n_sub,
    nu_subband_ref)`` restricted to the owned coarse DM, local
    ``fine_to_coarse`` zero, as ``_select_dm_owner_half`` does), so the
    1x-compat engine reads the same stream rows as production.
    ``dm_idx`` in 1x is the LOCAL fine index (today's C1 semantics).
    """
    from ...coarse_dm.dm_plan import subband_ref_freqs_table_GHz
    from ...fine_dm.combiner import compute_time_shift_search

    del symmetric  # production pads symmetrically; the engine reads signed shifts directly
    T = t_native * NATIVE_SAMPLE_US
    refs = subband_ref_freqs_table_GHz(int(csf), int(n_sub)) if n_sub > 1 else None
    dms = []
    fine_dm = np.asarray(fine_dm, dtype=np.float64)
    f2c = np.asarray(fine_to_coarse, dtype=np.int64)
    for c in owned:
        sel = np.nonzero(f2c == int(c))[0]
        if sel.size == 0:
            raise ValueError(f"no fine DMs map to coarse {c}")
        table = compute_time_shift_search(
            coarse_dm_pc_cm3=np.asarray([coarse_dm[int(c)]], dtype=np.float64),
            fine_dm_pc_cm3=fine_dm[sel],
            fine_to_coarse=np.zeros(sel.size, dtype=np.int64),
            t_int_search_us=T, merge_coarse_rounding=True, t_int_corr_us=T,
            n_sub=int(n_sub), nu_subband_ref_GHz=refs,
        )
        dms.append(DmSpec(d=int(c), tier="1x", t_native=int(t_native),
                          coarse_dm=float(coarse_dm[int(c)]), fine_dm=fine_dm[sel],
                          fine_global=np.arange(sel.size), shifts=table.shifts))
    n_stream = int(dms[0].n_stream)
    return OwnerPlan(half=int(half), dms=tuple(dms), n_stream=n_stream,
                     ext_native=int(t_native), opoint="1x",
                     plan_id=_plan_id([coarse_dm, fine_dm, f2c, n_sub, csf, list(owned)]))
