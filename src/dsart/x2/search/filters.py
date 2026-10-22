"""Time filters + the K3b streaming detector, numpy twin (plan "Time filters").

Filters are B-splines of a width-``w`` boxcar: order 1 (``b`` boxcar),
order 2 (``t`` triangle), order 3 (``g``, ~Gaussian), length
``L = m (w - 1) + 1``, centred ``a = L // 2`` (today's boxcar convention:
window ``[c - w//2, c - w//2 + w - 1]``). Their Layer-2 seed is
``||h||_2`` (``sqrt(w)`` for a boxcar = today's ``sqrt(k_dm k_time)``).

K3b, exactly (this twin and ``kernels_triton.k3b_detect`` are the same
sequence of fp32 operations, so they agree bit for bit):

* input ``xs = clip(fl32(x16) * sigma_inv[f], -clip, +clip)`` -- today's
  250-sigma input clip on the Layer-1-normalised cube;
* order-``m`` filter = ``m`` cascaded fp32 RUNNING SUMS of width ``w``:
  stage ``j`` is kept at lags ``l = 0 .. m - j`` (window ends ``e - l w``)
  and updated ``S_j(e') <- (S_j(e') + in(e')) - in(e' - w)`` with
  ``in = xs`` for stage 1 and ``in = S_{j-1}`` after; rows before the
  filter's first needed row read as zero, so every cycle restarts from a
  zero state (bounded error, no cross-cycle drift);
* the loop runs over CENTRES ``c``: filter ``k`` consumes window end
  ``e = c + D_k`` (``D_k = L_k - 1 - a_k``), so every filter reports the
  same centre in the same iteration; a warm-up of ``max(L_k) - 1``
  iterations precedes the first centre;
* ``snr_k = S_m * s_inv[k]`` (fp32); per (pixel, 64-centre chunk) keep the
  max over (centre, filter) -- strictly greater wins, so ties go to the
  earliest centre then the lowest filter index (families processed in
  bank order, accumulating into the same chunk maxima) -- plus the packed
  ``(t_in_chunk << 8) | k``;
* taps: the raw ``S_m`` of every centre at a fixed list of tap pixels
  feeds the per-DM Layer-2 sigma_k estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TimeFilter",
    "FilterBank",
    "k3b_twin",
    "pack_tk",
    "unpack_tk",
]

_ORDER = {"b": 1, "t": 2, "g": 3}


@dataclass(frozen=True)
class TimeFilter:
    family: str
    width: int

    @property
    def order(self) -> int:
        return _ORDER[self.family]

    @property
    def name(self) -> str:
        return f"{self.family}{self.width}"

    @property
    def length(self) -> int:
        return self.order * (self.width - 1) + 1

    @property
    def a(self) -> int:
        return self.length // 2

    @property
    def d(self) -> int:
        """Window end minus centre."""
        return self.length - 1 - self.a

    def taps(self) -> np.ndarray:
        box = np.ones(self.width)
        h = box
        for _ in range(self.order - 1):
            h = np.convolve(h, box)
        return h

    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self.taps() ** 2)))


@dataclass(frozen=True)
class FilterBank:
    """Ordered filters; families contiguous, widths ascending within one."""

    filters: Tuple[TimeFilter, ...]

    @classmethod
    def build(cls, widths: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
              families: Sequence[str] = ("b",)) -> "FilterBank":
        out = []
        for fam in families:
            if fam not in _ORDER:
                raise ValueError(f"filter family {fam!r}")
            for w in widths:
                out.append(TimeFilter(fam, int(w)))
        if len(out) > 255:
            raise ValueError("at most 255 filters (packed k is 8 bits)")
        return cls(tuple(out))

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

    def family_groups(self) -> list:
        """``[(order, k0, k1)]`` index ranges of each family."""
        groups, k0 = [], 0
        while k0 < len(self.filters):
            fam = self.filters[k0].family
            k1 = k0
            while k1 < len(self.filters) and self.filters[k1].family == fam:
                k1 += 1
            groups.append((self.filters[k0].order, k0, k1))
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
            ``r`` lives in slot ``(r - ring_row0) % R`` -- ``ring_row0`` is
            any absolute row congruent to slot 0 (the engine uses 0).
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

    # one pass per family (the kernel launches once per family and
    # accumulates into the same chunk maxima)
    for _order, k0, k1 in bank.family_groups():
        fams = bank.filters[k0:k1]
        lmax = max(f.length for f in fams)
        state = []
        for flt in fams:
            m = flt.order
            state.append({(j, l): np.zeros((F, P), dtype=np.float32)
                          for j in range(1, m + 1) for l in range(0, m - j + 1)})
        cur_best = best[0].copy()
        cur_tk = tk[0].copy()
        for c in range(c_first - (lmax - 1), c_first + n_c):
            out_c = c >= c_first
            ci = c - c_first
            for kk, flt in enumerate(fams):
                k = k0 + kk
                m, w, st = flt.order, flt.width, state[kk]
                e = c + flt.d
                r_start = c_first - flt.a
                for l in range(0, m):                              # stage 1, lags 0..m-1
                    ee = e - l * w
                    st[(1, l)] = (st[(1, l)] + xs_at(ee, r_start)) - xs_at(ee - w, r_start)
                for j in range(2, m + 1):
                    for l in range(0, m - j + 1):
                        st[(j, l)] = (st[(j, l)] + st[(j - 1, l)]) - st[(j - 1, l + 1)]
                if not out_c:
                    continue
                score = st[(m, 0)]
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
