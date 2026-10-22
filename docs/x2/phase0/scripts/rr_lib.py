"""Row-rule library (Phase 0 study 1): quantisers and their EXACT Gaussian moments.

All amplitudes are per real component in units of the tracked per-stream scale sigma_hat, after the
static sqrt(n) normalisation (docs/x2/SPEC.md section 4).  A value whose true noise is r * sigma_hat
and whose signal is a (in units of its own noise) enters the quantiser as y = r * (a + z), z ~ N(0,1).

Quantisers (step s, b bits, decoded value in sigma_hat units):
  midrise   levels (j + 0.5) s, j = -2^(b-1) .. 2^(b-1)-1  (SPEC v0.1 draft: floor + 0.5)
  midtread  levels j s,         j = -(2^(b-1)-1) .. 2^(b-1)-1 (round; code -2^(b-1) left free as a
            "no data" sentinel; today's cint8 is this with b = 8)
Moments are computed exactly from the normal CDF over the quantiser bins (no Monte Carlo).
"""
from __future__ import annotations

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_erf = np.frompyfunc(math.erf, 1, 1)
SQ2 = math.sqrt(2.0)


def Phi(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf(x / SQ2).astype(float))


def phi(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class Quant:
    """A scalar quantiser: edges e_1 < ... < e_{L-1} (bins are (-inf, e_1], ..., (e_{L-1}, inf)) and
    reconstruction levels (L,), both in sigma_hat units."""

    def __init__(self, kind, bits, step):
        self.kind, self.bits, self.step = kind, int(bits), float(step)
        h = 2 ** (bits - 1)
        if kind == "midrise":
            j = np.arange(-h, h)
            self.levels = (j + 0.5) * step
            self.edges = np.arange(-h + 1, h) * step
        elif kind == "midtread":
            j = np.arange(-(h - 1), h)
            self.levels = j * step
            self.edges = (np.arange(-(h - 1), h - 1) + 0.5) * step
        else:
            raise ValueError(kind)
        self.clip = float(self.levels[-1])              # largest reconstruction level

    def __call__(self, y):
        """Decoded value(s) for input y (sigma_hat units)."""
        idx = np.searchsorted(self.edges, y, side="left")
        return self.levels[idx]

    def moments(self, a, r=1.0):
        """E[Q], E[Q^2], dE[Q]/da for y = r (a + z), z ~ N(0,1); a may be an array."""
        a = np.atleast_1d(np.asarray(a, dtype=float))
        e = self.edges[None, :] / r - a[:, None]        # edges in z units
        cdf = Phi(e)
        p = np.diff(np.concatenate([np.zeros((len(a), 1)), cdf, np.ones((len(a), 1))], axis=1), axis=1)
        m1 = p @ self.levels
        m2 = p @ (self.levels ** 2)
        # d/da E[Q] = sum over edges of (level jump) * pdf(edge)
        jumps = np.diff(self.levels)
        g = phi(e) @ jumps
        return m1, m2, g

    def eta(self, r=1.0):
        """Single-value S/N efficiency (noise-only input, small signal) = E[Q'] / sd(Q)."""
        m1, m2, g = self.moments(0.0, r)
        return float(g[0] / math.sqrt(m2[0] - m1[0] ** 2))


def optimum_step(kind, bits, lo=0.02, hi=1.5, n=400):
    """Step maximising eta for unit Gaussian input (golden refine on a grid)."""
    grid = np.linspace(lo, hi, n)
    e = np.array([Quant(kind, bits, s).eta() for s in grid])
    i = int(np.argmax(e))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n - 1)]
    for _ in range(40):
        c1, c2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
        if Quant(kind, bits, c1).eta() > Quant(kind, bits, c2).eta():
            b = c2
        else:
            a = c1
    s = 0.5 * (a + b)
    return s, Quant(kind, bits, s).eta()


def eff_image(groups, r_of=None):
    """Image point-source S/N efficiency (vs unquantised, natural weighting) for a set of value
    groups: list of (n array, Quant).  r_of(n) -> per-value noise ratio (default 1).
    eta = sum n g / sqrt(N * sum n V)."""
    num, den, N = 0.0, 0.0, 0.0
    for n, q in groups:
        n = np.asarray(n, float)
        un, cnt = np.unique(n, return_counts=True)
        w = un * cnt
        r = np.ones_like(un) if r_of is None else np.asarray(r_of(un), float)
        for rv in np.unique(r):
            sel = r == rv
            m1, m2, g = q.moments(0.0, rv)
            # signal per value: dE[Q(r(a+z))]/da = r * g_r(a) ... moments() already returns d/da
            num += float(w[sel].sum()) * float(g[0])
            den += float(w[sel].sum()) * float(m2[0] - m1[0] ** 2)
        N += float(w.sum())
    return num / math.sqrt(N * den)


def split_heavy(n, frac=1e-4):
    n = np.asarray(n)
    thr = frac * n.sum()
    return n[n < thr], n[n >= thr], thr


def heavy_step(n_max, N, rail=250.0, margin=2.0, bits=8, kind="midtread"):
    """s_h so the heavy clip (top level) is >= margin x the value's S/N at an image rail."""
    top = (2 ** (bits - 1) - 1) if kind == "midtread" else (2 ** (bits - 1) - 0.5)
    return margin * rail * math.sqrt(n_max / N) / top
