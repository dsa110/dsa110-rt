"""Row rule — the numpy REFERENCE for SPEC §4 (corr packer + owner decode).

WS-C's GPU packer and WS-S's decoder must reproduce this module bit for bit.
Every arithmetic step is pinned to float32 with round-to-nearest-even and NO
fused multiply-add (CUDA: ``__fmul_rn`` / ``__fadd_rn`` / ``__fdiv_rn`` /
``rintf`` / ``floorf``, or ``--fmad=false``):

Per (stream, coarse DM) flow, per corr block b of S samples (S = 256 tier A,
128 tier B; one block = 4096 native samples = 134.217728 ms) with values
``x[j, v]`` (complex64), heavy values first (``v < n_heavy``):

1. Normalise: ``r_v = f32(1 / sqrt(n_v))`` (float64 sqrt and divide, rounded
   once); ``y = (f32(re x) * r_v, f32(im x) * r_v)``.
2. Robust estimate on the subsample ``j % est_stride == 0`` (default 16):
   a value is used iff both components are finite, it is not exactly zero
   (re == 0 and im == 0 — flagged data) and, when ``sigma_prev > 0``,
   ``|re| <= T and |im| <= T`` with ``T = f32(6) * sigma_prev``. With m used
   values, if ``m < max(16, ceil(M / 8))`` (M = values in the subsample) the
   block does not update sigma. Otherwise ``med`` = the LOWER median (rank
   ``(2m - 1) // 2``, ascending) of the 2m magnitudes ``|re|, |im|`` and
   ``est = f32(1.4826) * med``.
3. EMA: if ``sigma_prev`` is 0 (uninitialised) ``sigma = est``; else
   ``sigma = sigma_prev + alpha * (est - sigma_prev)`` evaluated as
   ``d = est - sigma_prev; m = alpha * d; sigma = sigma_prev + m`` (three f32
   ops), ``alpha = f32(-expm1(-block_s / tau_s))`` (tau 10 s). Block b is
   quantised with the sigma AFTER its own update.
4. Codes: ``step = f32(s) * sigma``; ``inv = f32(1) / step`` (f32 division);
   ``q = y * inv`` per component. MID-TREAD (default, zero-preserving):
   ``code = clip(rint(q), -(2^(c-1) - 1), 2^(c-1) - 1)`` (half-to-even; the
   code -2^(c-1) is never emitted). MID-RISE (selectable, flag bit6):
   ``code = clip(floor(q), -2^(c-1), 2^(c-1) - 1)``. c = 4/5/6 bits per main
   component (s_main), c = 8 for heavy values (s_heavy). Non-finite q, or
   ``step <= 0`` / non-finite (sigma uninitialised), gives code 0.
5. Owner decode in sigma units: ``v = f32(code) * s`` (mid-tread) or
   ``(f32(code) + 0.5) * s`` (mid-rise), s = s_main or s_heavy; gridding
   multiplies by ``f32(sqrt(n_v))`` (natural weighting of what was summed;
   1/sigma weighting across flows, as today's per-row max-abs semantics).

Why mid-tread by default (SPEC v0.1 said mid-rise): a mid-rise code cannot
represent 0, so every exactly-zero (flagged / excised) value decodes to
``+0.5 step (1 + i)``. Array-burst excision zeroes whole blocks, which would
put a coherent ``0.5 s sqrt(n)`` offset on every cell for the excision's
duration — a phase-centre point source switching on and off. Mid-tread costs
0.07% of S/N at 4 bits (efficiency 0.9935 vs 0.9942) and 0.01% at 5 bits
(Gaussian-optimal uniform steps: mid-tread 15 levels 0.3534, 31 levels 0.1932;
mid-rise 16 levels 0.3352, 32 levels 0.1881). Phase 0 confirms the steps.

``v1_rowmax_cint8`` reproduces today's 1x row rule (tx.py) for 1x-compat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NATIVE_SAMPLE_S: float = 32.768e-6
BLOCK_NATIVE: int = 4096
BLOCK_S: float = BLOCK_NATIVE * NATIVE_SAMPLE_S        # 0.134217728 s

MODE_MIDTREAD = "midtread"
MODE_MIDRISE = "midrise"

DEFAULT_S_MAIN: dict[tuple[str, int], float] = {
    (MODE_MIDTREAD, 4): 0.3534,
    (MODE_MIDTREAD, 5): 0.1932,
    (MODE_MIDTREAD, 6): 0.1055,
    (MODE_MIDRISE, 4): 0.336,
    (MODE_MIDRISE, 5): 0.191,
    (MODE_MIDRISE, 6): 0.1041,
}
DEFAULT_S_HEAVY: float = 0.1


@dataclass(frozen=True)
class RowRule:
    """Parameters of the row rule. ``comp_bits`` = main bits per component."""

    comp_bits: int = 4
    mode: str = MODE_MIDTREAD
    s_main: float | None = None
    s_heavy: float = DEFAULT_S_HEAVY
    tau_s: float = 10.0
    block_s: float = BLOCK_S
    est_stride: int = 16
    excl_sigma: float = 6.0
    robust_k: float = 1.4826
    min_frac: float = 0.125
    min_count: int = 16

    def __post_init__(self) -> None:
        if self.comp_bits not in (4, 5, 6):
            raise ValueError(f"comp_bits={self.comp_bits} not in (4, 5, 6)")
        if self.mode not in (MODE_MIDTREAD, MODE_MIDRISE):
            raise ValueError(f"mode={self.mode!r}")
        if self.est_stride < 1:
            raise ValueError("est_stride must be >= 1")

    @property
    def bits_main(self) -> int:
        """Frame-v2 ``bits_main`` (complex bits per main value)."""
        return 2 * self.comp_bits

    @property
    def midrise(self) -> bool:
        return self.mode == MODE_MIDRISE

    @property
    def s_main_f32(self) -> np.float32:
        s = self.s_main if self.s_main is not None else DEFAULT_S_MAIN[(self.mode, self.comp_bits)]
        return F32(s)

    @property
    def s_heavy_f32(self) -> np.float32:
        return F32(self.s_heavy)

    @property
    def alpha_f32(self) -> np.float32:
        return F32(-math.expm1(-self.block_s / self.tau_s))

    def code_range(self, comp_bits: int) -> tuple[int, int]:
        lo = -(1 << (comp_bits - 1))
        hi = (1 << (comp_bits - 1)) - 1
        return (lo, hi) if self.midrise else (lo + 1, hi)

    @property
    def main_range(self) -> tuple[int, int]:
        return self.code_range(self.comp_bits)

    @property
    def heavy_range(self) -> tuple[int, int]:
        return self.code_range(8)


# ---------------------------------------------------------------------------
# Static per-value factors
# ---------------------------------------------------------------------------


def rsqrt_n(n: np.ndarray) -> np.ndarray:
    """``f32(1 / sqrt(n))`` per value (float64 math, one rounding)."""
    n = np.asarray(n, dtype=np.int64)
    if n.size and n.min() < 1:
        raise ValueError("contribution counts must be >= 1")
    return (1.0 / np.sqrt(n.astype(np.float64))).astype(np.float32)


def sqrt_n(n: np.ndarray) -> np.ndarray:
    """``f32(sqrt(n))`` per value — the owner's natural-weighting factor."""
    n = np.asarray(n, dtype=np.int64)
    if n.size and n.min() < 1:
        raise ValueError("contribution counts must be >= 1")
    return np.sqrt(n.astype(np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# Steps 1-4
# ---------------------------------------------------------------------------


def normalise(x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``y = x * r`` per component in float32. ``x [..., V]`` complex."""
    x = np.asarray(x)
    r = np.asarray(r, dtype=np.float32)
    re = np.ascontiguousarray(x.real, dtype=np.float32) * r
    im = np.ascontiguousarray(x.imag, dtype=np.float32) * r
    return re.astype(np.float32, copy=False), im.astype(np.float32, copy=False)


def robust_estimate(re: np.ndarray, im: np.ndarray, sigma_prev: np.float32,
                    rule: RowRule) -> tuple[np.float32 | None, int]:
    """Step 2 on the estimation subsample. Returns ``(est or None, m)``."""
    re = np.asarray(re, dtype=np.float32)
    im = np.asarray(im, dtype=np.float32)
    a_re = np.abs(re)
    a_im = np.abs(im)
    keep = np.isfinite(re) & np.isfinite(im) & ~((re == 0) & (im == 0))
    sp = F32(sigma_prev)
    if np.isfinite(sp) and sp > 0:
        thr = F32(rule.excl_sigma) * sp
        keep &= (a_re <= thr) & (a_im <= thr)
    m = int(keep.sum())
    m_min = max(int(rule.min_count), int(math.ceil(rule.min_frac * re.size)))
    if m < m_min or m == 0:
        return None, m
    comps = np.concatenate([a_re[keep], a_im[keep]])
    kth = (comps.size - 1) // 2
    med = np.partition(comps, kth)[kth]
    return F32(F32(rule.robust_k) * F32(med)), m


def ema_update(sigma_prev: np.float32, est: np.float32 | None,
               rule: RowRule) -> np.float32:
    """Step 3."""
    sp = F32(sigma_prev)
    if est is None:
        return sp
    e = F32(est)
    if not (np.isfinite(sp) and sp > 0):
        return e
    d = F32(e - sp)
    m = F32(rule.alpha_f32 * d)
    return F32(sp + m)


def quantise(re: np.ndarray, im: np.ndarray, step: np.float32,
             code_range: tuple[int, int], midrise: bool) -> np.ndarray:
    """Step 4 -> ``int8 [..., 2]`` (re, im)."""
    re = np.asarray(re, dtype=np.float32)
    im = np.asarray(im, dtype=np.float32)
    out_shape = re.shape + (2,)
    st = F32(step)
    if not (np.isfinite(st) and st > 0):
        return np.zeros(out_shape, dtype=np.int8)
    inv = F32(F32(1.0) / st)
    q = np.stack([re * inv, im * inv], axis=-1).astype(np.float32, copy=False)
    q = np.where(np.isfinite(q), q, F32(0.0))
    c = np.floor(q) if midrise else np.rint(q)
    lo, hi = code_range
    return np.clip(c, lo, hi).astype(np.int8)


# ---------------------------------------------------------------------------
# Step 5 (owner)
# ---------------------------------------------------------------------------


def decode(codes: np.ndarray, s: np.float32, midrise: bool) -> np.ndarray:
    """Codes ``[..., 2]`` -> complex64 values in sigma_hat units."""
    c = np.asarray(codes).astype(np.float32)
    if midrise:
        c = c + F32(0.5)
    v = c * F32(s)
    return (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)


def decode_row(codes_heavy: np.ndarray, codes_main: np.ndarray,
               rule: RowRule) -> np.ndarray:
    """Heavy-first value vector ``[..., n_values]`` in sigma_hat units."""
    vh = decode(codes_heavy, rule.s_heavy_f32, rule.midrise)
    vm = decode(codes_main, rule.s_main_f32, rule.midrise)
    return np.concatenate([vh, vm], axis=-1)


def natural_weight(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """``v * f32(sqrt(n))`` per component (the owner's gridding weight)."""
    w = sqrt_n(n)
    re = np.asarray(v.real, dtype=np.float32) * w
    im = np.asarray(v.imag, dtype=np.float32) * w
    return (re + 1j * im).astype(np.complex64)


# ---------------------------------------------------------------------------
# Per-flow stateful quantiser (what the corr packer runs per (stream, DM))
# ---------------------------------------------------------------------------


@dataclass
class BlockCodes:
    codes_heavy: np.ndarray       # int8 [S, n_heavy, 2]
    codes_main: np.ndarray        # int8 [S, n_main, 2]
    sigma: np.float32             # sigma_hat used for this block
    step_main: np.float32
    step_heavy: np.float32
    est: np.float32 | None        # this block's robust estimate (None = no update)
    n_used: int                   # values used by the estimate


@dataclass
class RowQuantiser:
    """Row-rule state of ONE (stream, coarse DM) flow."""

    rule: RowRule
    n_contrib: np.ndarray         # int [n_values], heavy first
    n_heavy: int
    sigma: np.float32 = field(default_factory=lambda: F32(0.0))

    def __post_init__(self) -> None:
        self.n_contrib = np.asarray(self.n_contrib, dtype=np.int64)
        if not 0 <= self.n_heavy <= self.n_contrib.size:
            raise ValueError("n_heavy out of range")
        self._r = rsqrt_n(self.n_contrib)

    @property
    def n_values(self) -> int:
        return int(self.n_contrib.size)

    def process_block(self, x: np.ndarray) -> BlockCodes:
        """Quantise one block ``x [S, n_values]`` (complex)."""
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.n_values:
            raise ValueError(f"x must be [S, {self.n_values}], got {x.shape}")
        rule = self.rule
        re, im = normalise(x, self._r)
        sub = slice(0, x.shape[0], rule.est_stride)
        est, m = robust_estimate(re[sub], im[sub], self.sigma, rule)
        self.sigma = ema_update(self.sigma, est, rule)
        step_m = F32(rule.s_main_f32 * self.sigma)
        step_h = F32(rule.s_heavy_f32 * self.sigma)
        nh = self.n_heavy
        ch = quantise(re[:, :nh], im[:, :nh], step_h, rule.heavy_range, rule.midrise)
        cm = quantise(re[:, nh:], im[:, nh:], step_m, rule.main_range, rule.midrise)
        return BlockCodes(ch, cm, self.sigma, step_m, step_h, est, m)


# ---------------------------------------------------------------------------
# 1x-compat: today's per-row max-abs cint8 (tx.py TransportTx._encode_rows)
# ---------------------------------------------------------------------------


def v1_rowmax_cint8(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Today's row rule: per row max-abs -> cint8, codes ``int8 [R, N, 2]``
    and float32 scales ``[R]``; byte-identical to ``TransportTx._encode_rows``
    at ``bits_per_cell = 16``."""
    arr = np.asarray(rows).astype(np.complex64, copy=False)
    n_rows, n_filled = arr.shape
    re_im = np.empty((n_rows, n_filled, 2), dtype=np.float32)
    re_im[:, :, 0] = arr.real
    re_im[:, :, 1] = arr.imag
    if n_filled == 0:
        amax = np.zeros(n_rows, dtype=np.float32)
    else:
        amax = np.abs(re_im).reshape(n_rows, -1).max(axis=1)
    scales = (amax.astype(np.float64) / 127.0).astype(np.float32)
    scales[amax == 0.0] = np.float32(1.0)
    inv = (1.0 / scales.astype(np.float64)).astype(np.float32)
    q = np.clip(np.round(re_im * inv[:, None, None]), -128.0, 127.0).astype(np.int8)
    return q, scales


__all__ = [
    "BLOCK_NATIVE", "BLOCK_S", "NATIVE_SAMPLE_S", "MODE_MIDTREAD",
    "MODE_MIDRISE", "DEFAULT_S_MAIN", "DEFAULT_S_HEAVY", "RowRule",
    "rsqrt_n", "sqrt_n", "normalise", "robust_estimate", "ema_update",
    "quantise", "decode", "decode_row", "natural_weight", "BlockCodes",
    "RowQuantiser", "v1_rowmax_cint8",
]
