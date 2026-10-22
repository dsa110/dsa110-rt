"""Owner candidates in external-clock units (SPEC §7, §8) + chunk-max decoder.

:class:`Candidate2x` is what one owner half reports per detection:
global coarse ``d``, tier, global fine-DM index and value, ``(l, m)``
pixel, centre time and width in EXTERNAL samples (T_ext = 524.288 us in
2x; tier-B samples count 2 each; 1x keeps today's 1.048576 ms sample),
the filter (family + width), S/N and flags.
:meth:`Candidate2x.to_legacy` maps it onto today's 12-field
``common.contracts.Candidate`` for the existing C1 path (WS-D owns any
schema change): ``dm_idx`` = global fine index in 2x (today's local fine
index in 1x), ``event_specnum`` / ``width_samples`` in external samples,
``kernel_id`` ``unit:d1:b<n>`` with ``n`` the filter's support rounded up
to a power of two (B-spline family and true width kept on Candidate2x).

:func:`decode_chunkmax` is today's ``decode_topk_argmax_lowmem`` applied
to the per-(pixel, 64-centre chunk) maxima: global top-``n_top`` by S/N
(stable on ties), threshold, merge-radius suppression with AND semantics
(``|dt| <= 128, |dfdm| <= 5, |dl|, |dm| <= 3`` in tier samples / local
fine index / pixels), winner filter from the packed ``(t, k)``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

import numpy as np

from .filters import FilterBank, unpack_tk

__all__ = ["Candidate2x", "decode_chunkmax", "merge_c1", "NOISE_WARMUP_FLAG",
           "PARTIAL_FANIN_FLAG"]

#: ``CandidateFlags.NOISE_WARMUP`` (Layer-2 burn-in), same bit as today.
NOISE_WARMUP_FLAG = 1 << 3
#: Owner-specific: the cycle had rows flagged invalid (fan-in deadline).
PARTIAL_FANIN_FLAG = 1 << 7


@dataclass(frozen=True)
class Candidate2x:
    half: int
    d: int
    tier: str
    fdm_local: int
    fine_global: int
    dm: float
    l: int
    m: int
    t_tier: int              # absolute centre, tier samples
    event_specnum: int       # absolute centre, external samples
    width_ext: int           # filter base width in external samples
    support_ext: int         # filter support (L) in external samples
    filt: str                # e.g. "b8", "t4", "g2"
    snr: float
    flags: int = 0

    def to_legacy(self, *, search_node_id: int, gpu_half: int, dm_index_mode: str = "global",
                  detector_version: str = "x2.owner"):
        from ...common.contracts import Candidate

        n = 1
        while n < max(1, self.support_ext):
            n *= 2
        n = min(n, 128)
        dm_idx = self.fine_global if dm_index_mode == "global" else self.fdm_local
        return Candidate(
            l=float(self.l), m=float(self.m), dm_fine=float(self.dm), dm_idx=int(dm_idx),
            event_specnum=int(self.event_specnum), width_samples=int(self.width_ext),
            kernel_id=f"unit:d1:b{n}", snr=float(self.snr), detector_version=detector_version,
            flags=int(self.flags), search_node_id=int(search_node_id), gpu_half=int(gpu_half),
        )

    def with_flags(self, flags: int) -> "Candidate2x":
        return replace(self, flags=int(flags))


def decode_chunkmax(
    best: np.ndarray,
    tk: np.ndarray,
    *,
    c_first: int,
    chunk: int,
    n_grid: int,
    bank: FilterBank,
    threshold: float,
    n_top: int,
    spec,
    half: int,
    ext_factor: int,
    radius_t: int = 128,
    radius_f: int = 5,
    radius_lm: int = 3,
    flags: int = 0,
) -> List[Candidate2x]:
    """``best/tk [n_ch, F, P]`` (from K3b) -> candidates of one DM-cycle."""
    n_ch, F, P = best.shape
    flat = np.asarray(best, dtype=np.float32).reshape(-1)
    k = int(min(max(1, n_top), flat.size))
    if k <= 0 or not np.isfinite(flat).any() or float(np.nanmax(flat)) <= threshold:
        return []
    part = np.argpartition(-flat, k - 1)[:k] if k < flat.size else np.arange(flat.size)
    order = part[np.lexsort((part, -flat[part]))]            # value desc, index asc
    tkf = np.asarray(tk).reshape(-1)
    out: List[Candidate2x] = []
    acc: list = []
    for ix in order:
        v = float(flat[ix])
        if not v > threshold:
            break
        ci, rem = divmod(int(ix), F * P)
        f, p = divmod(rem, P)
        l_, m_ = divmod(p, n_grid)
        tin, kk = unpack_tk(tkf[ix])
        t = int(c_first + ci * chunk + int(tin))
        if any(abs(t - a) <= radius_t and abs(f - b) <= radius_f
               and abs(l_ - c) <= radius_lm and abs(m_ - dd) <= radius_lm
               for a, b, c, dd in acc):
            continue
        acc.append((t, f, l_, m_))
        flt = bank.filters[int(kk)]
        out.append(Candidate2x(
            half=int(half), d=int(spec.d), tier=str(spec.tier), fdm_local=int(f),
            fine_global=int(spec.fine_global[f]), dm=float(spec.fine_dm[f]),
            l=int(l_), m=int(m_), t_tier=t, event_specnum=t * int(ext_factor),
            width_ext=int(flt.width) * int(ext_factor),
            support_ext=int(flt.length) * int(ext_factor),
            filt=flt.name, snr=v, flags=int(flags),
        ))
    return out


def merge_c1(cands: List[Candidate2x], *, lm_max_cells: int = 5, dm_max_trials: int = 2,
             t_frac: float = 1.0, sample_period_specnum: int = 16,
             dm_index_mode: str = "global") -> List[Candidate2x]:
    """Today's ``merge_across_kernels_c1`` verbatim (OR-cross (l, m),
    ``|d dm_idx| <= dm_max_trials``, ``|d event_specnum| <= t_frac * (w_i +
    w_j) / 2 * sample_period_specnum``) on external samples. NB: today's
    window multiplies sample counts by ``sample_period_specnum`` (16) while
    comparing sample-unit ``event_specnum`` -- reproduced as is; the
    external-clock audit (WS-D) owns any change."""
    key = (lambda c: c.fine_global) if dm_index_mode == "global" else (lambda c: c.fdm_local)
    srt = sorted(cands, key=lambda c: (-c.snr, c.event_specnum, key(c), c.l, c.m, c.filt))
    keep: List[Candidate2x] = []
    for c in srt:
        sup = False
        for s in keep:
            win = t_frac * 0.5 * float(c.width_ext + s.width_ext) * float(sample_period_specnum)
            if abs(c.event_specnum - s.event_specnum) > win:
                continue
            if abs(key(c) - key(s)) > dm_max_trials:
                continue
            if abs(c.l - s.l) <= lm_max_cells or abs(c.m - s.m) <= lm_max_cells:
                sup = True
                break
        if not sup:
            keep.append(c)
    return keep

