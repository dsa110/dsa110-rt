"""Time filters + the K3b streaming detector, numpy twin (plan "Time filters",
Phase 0 study 5 ``docs/x2/phase0/time_filters.md``).

A filter is a cascade of integer box (running-sum) stages of lengths
``stages = (L1, ..., Lm)`` (length-1 stages are identities and dropped):
support ``L = sum(Li - 1) + 1``, taps = the convolution of the boxes
(integer sums, not averages), centred ``a = L // 2`` -- today's boxcar
convention (window ``[c - w//2, c - w//2 + w - 1]``). Banks:

* ``"box"`` -- today's ladder ``b1 .. b64`` (stages ``(w,)``);
* ``"b3x"`` -- the Phase 0 recommendation (8 filters): ``b1``, ``b2``,
  ``t2`` = [1,2,1], then FWHM-matched quadratic B-splines ``g4 .. g64``
  with near-equal stage lengths (3,3,3), (6,6,7), (12,13,13),
  (25,25,26), (50,50,51); 148 history rows;
* ``"t"`` / ``"g"`` families at arbitrary widths (studies, tests).

Per filter: ``seed = ||h||_2`` (Layer-2 sigma_k seed, the white-noise
sigma of the integer-sum output) and the Layer-2 floor
``0.5 * seed / sqrt(w)`` with ``w`` the nominal width (0.5 for every
boxcar = today's floor, never rescaled with image units -- Phase 0 §2).

K3b, exactly (this twin and ``kernels_triton.k3b_detect`` perform the
same fp32 operations in the same order, so they agree bit for bit):

* input ``xs = clip(fl32(x16) * sigma_inv[f], -clip, +clip)`` (today's
  250-sigma clip on the Layer-1-normalised input);
* stage ``j`` of a filter is kept at the lag set ``O_j`` (``O_m = {0}``,
  ``O_{j-1} = O_j U (O_j + L_j)``); per centre, stage 1 at each lag ``o``:
  ``S <- (S + xs[e - o]) - xs[e - o - L1]``, then stage ``j``:
  ``S_j(o) <- (S_j(o) + S_{j-1}(o)) - S_{j-1}(o + L_j)``; a filter without
  stages outputs ``xs[e]``. Rows before the filter's first needed row read
  as zero, so every cycle restarts from a zero state (bounded error, no
  cross-cycle drift);
* the loop runs over CENTRES: filter ``k`` consumes window end
  ``e = c + D_k`` (``D_k = L_k - 1 - a_k``) so every filter reports the
  same centre in the same iteration; each family group runs a warm-up of
  ``max(L_k) - 1`` iterations;
* ``snr = S * s_inv[k]``; per (pixel, 64-centre chunk) keep the max over
  (centre, filter) -- strict ``>``; groups run in bank order and accumulate
  into the same chunk maxima -- plus the packed ``(t_in_chunk << 8) | k``;
* taps: the raw outputs of every centre at fixed tap pixels feed the
  per-DM Layer-2 sigma_k estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TimeFilter",
    "FilterBank",
    "B3_STAGES",
    "k3b_twin",
    "pack_tk",
    "unpack_tk",
]

#: FWHM-matched quadratic B-spline stage lengths (Phase 0 study 5).
B3_STAGES = {2: (1, 2, 2), 4: (3, 3, 3), 8: (6, 6, 7), 16: (12, 13, 13), 32: (25, 25, 26),
             64: (50, 50, 51)}


def _stages_for(family: str, w: int) -> tuple:
    w = int(w)
    if family == "b":
        return (w,)
    if family == "t":
        return (w, w)
    if family == "g":
        if w in B3_STAGES:
            return B3_STAGES[w]
        a = max(1, int(round(w / 1.268)))
        return (a, a, a)
    raise ValueError(f"filter family {family!r}")


@dataclass(frozen=True)
class TimeFilter:
    name: str
    width: int                  # nominal width (tier samples)
    stages: Tuple[int, ...]     # box lengths, identities dropped

    @classmethod
    def make(cls, family: str, w: int, name: Optional[str] = None) -> "TimeFilter":
        st = tuple(int(x) for x in _stages_for(family, w) if int(x) > 1)
        return cls(name or f"{family}{int(w)}", int(w), st)

    @property
    def family(self) -> str:
        return self.name.rstrip("0123456789")

    @property
    def order(self) -> int:
        return len(self.stages)

    @property
    def length(self) -> int:
        return sum(L - 1 for L in self.stages) + 1

    @property
    def a(self) -> int:
        return self.length // 2

    @property
    def d(self) -> int:
        """Window end minus centre."""
        return self.length - 1 - self.a

    def taps(self) -> np.ndarray:
        h = np.ones(1)
        for L in self.stages:
            h = np.convolve(h, np.ones(L))
        return h

    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.taps() ** 2)))

    @property
    def layer2_floor(self) -> float:
        return 0.5 * self.norm / np.sqrt(float(self.width))

    def lag_sets(self) -> list:
        """``O_j`` for j = 1..m (stage 1 first)."""
        if not self.stages:
            return []
        sets = [None] * self.order
        sets[-1] = (0,)
        for j in range(self.order - 1, 0, -1):
            Lj = self.stages[j]
            sets[j - 1] = tuple(sorted(set(sets[j]) | {o + Lj for o in sets[j]}))
        return sets


@dataclass(frozen=True)
class FilterBank:
    """Ordered filters, grouped into families (each group = one K3b pass)."""

    filters: Tuple[TimeFilter, ...]

    @classmethod
    def build(cls, widths: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
              families: Sequence[str] = ("b",)) -> "FilterBank":
        if tuple(families) == ("b3x",):
            return cls.b3x()
        out = [TimeFilter.make(fam, w) for fam in families for w in widths]
        if len(out) > 255:
            raise ValueError("at most 255 filters (packed k is 8 bits)")
        return cls(tuple(out))

    @classmethod
    def b3x(cls) -> "FilterBank":
        """Phase 0 bank: b1, b2, t2 = [1,2,1], g4 .. g64."""
        f = [TimeFilter.make("b", 1), TimeFilter.make("b", 2),
             TimeFilter("t2", 2, (2, 2))]
        f += [TimeFilter("g%d" % w, w, B3_STAGES[w]) for w in (4, 8, 16, 32, 64)]
        return cls(tuple(f))

    def __len__(self) -> int:
        return len(self.filters)

    @property
    def max_length(self) -> int:
        return max(f.length for f in self.filters)

    @property
    def max_a(self) -> int:
        return max(f.a for f in self.filters)

    @property
    def max_d(self) -> int:
        return max(f.d for f in self.filters)

    @property
    def seeds(self) -> np.ndarray:
        return np.array([f.norm for f in self.filters], dtype=np.float64)

    @property
    def layer2_floors(self) -> np.ndarray:
        return np.array([f.layer2_floor for f in self.filters], dtype=np.float64)

    def family_groups(self) -> list:
        """``[(order, k0, k1)]``: maximal runs of filters with the same order
        (each run is one K3b launch with a uniform state shape)."""
        groups, k0 = [], 0
        while k0 < len(self.filters):
            o = self.filters[k0].order
            k1 = k0
            while k1 < len(self.filters) and self.filters[k1].order == o:
                k1 += 1
            groups.append((o, k0, k1))
            k0 = k1
        return groups

    def history(self, delay: Optional[int] = None) -> int:
        """Image rows needed before the cycle's first new row ``t0`` when
        the centres are ``[t0 - delay, t0 + hop - delay)``."""
        dly = self.max_d if delay is None else int(delay)
        if dly < self.max_d:
            raise ValueError(f"delay {dly} < longest filter's end offset {self.max_d}")
        return dly + self.max_a

    def kernel_ids(self) -> list:
        return [f.name for f in self.filters]


def pack_tk(t_in_chunk: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (np.asarray(t_in_chunk, dtype=np.int32) << 8) | np.asarray(k, dtype=np.int32)


def unpack_tk(tk: np.ndarray) -> tuple:
    tk = np.asarray(tk, dtype=np.int32)
    return tk >> 8, tk & 0xFF


def k3b_twin(
    ring: np.ndarray,
    *,
    ring_row0: int,
    c_first: int,
    n_c: int,
    sigma_inv: np.ndarray,
    s_inv: np.ndarray,
    bank: FilterBank,
    clip: float = 250.0,
    chunk: int = 64,
    tap_pixels: Optional[np.ndarray] = None,
) -> tuple:
    """Numpy twin of K3b for one DM-cycle.

    Args:
        ring: image ring ``[R, F, P]`` float16 (``P = N*N``); absolute row
            ``r`` lives in slot ``(r - ring_row0) % R``.
        c_first, n_c: centres ``[c_first, c_first + n_c)``; ``n_c`` a
            multiple of ``chunk``.
        sigma_inv: ``[F]`` Layer-1 ``1/sigma``; ``s_inv``: ``[K]`` Layer-2.
        tap_pixels: optional ``[n_tap]`` pixel indices for the Layer-2 taps.

    Returns:
        ``(best [n_ch, F, P] float32, tk [n_ch, F, P] int32,
        taps [K, F, n_tap, n_c] float32 or None)``.
    """
    R, F, P = ring.shape
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    n_ch = n_c // chunk
    si = np.asarray(sigma_inv, dtype=np.float32)[:, None]
    sk = np.asarray(s_inv, dtype=np.float32)
    clip32 = np.float32(clip)
    K = len(bank)
    taps_out = None
    if tap_pixels is not None:
        tap_pixels = np.asarray(tap_pixels, dtype=np.int64)
        taps_out = np.zeros((K, F, tap_pixels.size, n_c), dtype=np.float32)
    best = np.full((n_ch, F, P), -np.inf, dtype=np.float32)
    tk = np.zeros((n_ch, F, P), dtype=np.int32)
    zero = np.zeros((F, P), dtype=np.float32)

    def xs_at(row: int, r_start: int) -> np.ndarray:
        if row < r_start:
            return zero
        x = ring[(row - ring_row0) % R].astype(np.float32)
        return np.minimum(np.maximum(x * si, -clip32), clip32)

    for _order, k0, k1 in bank.family_groups():
        grp = bank.filters[k0:k1]
        lmax = max(f.length for f in grp)
        lags = [f.lag_sets() for f in grp]
        state = [{(j, o): np.zeros((F, P), dtype=np.float32)
                  for j in range(len(ls)) for o in ls[j]} for ls in lags]
        cur_best = best[0].copy()
        cur_tk = tk[0].copy()
        for c in range(c_first - (lmax - 1), c_first + n_c):
            out_c = c >= c_first
            ci = c - c_first
            for kk, flt in enumerate(grp):
                k = k0 + kk
                e = c + flt.d
                r_start = c_first - flt.a
                if not flt.stages:
                    score = xs_at(e, r_start)
                else:
                    st, ls = state[kk], lags[kk]
                    L1 = flt.stages[0]
                    for o in ls[0]:
                        st[(0, o)] = (st[(0, o)] + xs_at(e - o, r_start)) \
                            - xs_at(e - o - L1, r_start)
                    for j in range(1, flt.order):
                        Lj = flt.stages[j]
                        for o in ls[j]:
                            st[(j, o)] = (st[(j, o)] + st[(j - 1, o)]) - st[(j - 1, o + Lj)]
                    score = st[(flt.order - 1, 0)]
                if not out_c:
                    continue
                if taps_out is not None:
                    taps_out[k, :, :, ci] = score[:, tap_pixels]
                v = score * sk[k]
                better = v > cur_best
                cur_best = np.where(better, v, cur_best)
                cur_tk = np.where(better, np.int32(((ci % chunk) << 8) | k), cur_tk)
            if out_c and (ci % chunk) == chunk - 1:
                best[ci // chunk] = cur_best
                tk[ci // chunk] = cur_tk
                nxt = ci // chunk + 1
                if nxt < n_ch:
                    cur_best = best[nxt].copy()
                    cur_tk = tk[nxt].copy()
    return best, tk, taps_out
