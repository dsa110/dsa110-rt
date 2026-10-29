"""Row rule — the numpy REFERENCE for SPEC §4 (corr packer + owner decode).

Frozen with Phase 0 (``docs/x2/phase0/rowrule.md``). WS-C's GPU packer and
WS-S's decoder must reproduce this module bit for bit. Every arithmetic step
is float32 with round-to-nearest-even and NO fused multiply-add (CUDA:
``__fmul_rn`` / ``__fadd_rn`` / ``__fdiv_rn`` / ``rintf``, or
``--fmad=false``). "Lower median" of m numbers = the element of rank
``(m - 1) // 2`` in ascending order (a pure selection, no averaging).

Per (stream, coarse DM) flow, per corr block b of S samples (S = 256 tier A,
128 tier B; one block = 4096 native samples = 134.217728 ms) with values
``x[j, v]`` (complex64), heavy values first (``v < n_heavy``; heavy means
``n_v >= 3e-5 N_image``, from WS-P's value map):

1. Normalise: ``r_v = f32(1 / sqrt(n_v))`` (float64, one rounding);
   ``y = (f32(re x) * r_v, f32(im x) * r_v)``.
2. sigma_hat (per flow):
   - block estimate on the subsample ``j % est_stride == 0`` (16): a value is
     *used* iff both components are finite and at least one exceeds
     ``thr = f32(f32(1e-6) * sigma_hat_prev)`` in magnitude (v1.1; with
     sigma_hat_prev = 0 that is "not exactly 0 + 0i", v1.0's rule: flagged /
     excised). With m used values out of M, the block is SKIPPED
     (no update) when ``m < max(16, ceil(M / 2))`` (< 50% used). Else
     ``est = f32(1.4826) * lowmed(|re|, |im| of the used values)`` (2m numbers).
     No outlier exclusion (the median's 50% breakdown point suffices).
   - warm-up: blocks ``b < warmup_blocks`` (8 = the corr static-sky warm-up)
     never touch sigma_hat. The first non-skipped block after it sets
     ``sigma_hat = est``; later ones ``d = est - s; m = alpha * d; s = s + m``
     (three f32 ops), ``alpha = f32(-expm1(-block_s / 3 s))`` -- except that
     (v1.1) ``est > f32(f32(1.5) * s)`` resets ``s = est`` (a level step up
     re-converges in one block instead of guarding every row for seconds).
     ``est`` is floored at ``f32(1e-30)``.
   - v1.1 (2026-10-29, ledger S2g; ``RowRule.used_floor`` / ``reset_factor``,
     0 = v1.0): after a long whole-cube excision the static-sky residue
     (~1e-9 of sigma) counted as used data, sigma_hat decayed to it, and the
     release guarded every row for ~16 s. Both rules are inert on stationary
     noise (est / sigma_hat within 0.993..1.028 over 545 replayed blocks).
   - block b is coded with sigma_hat AFTER its own update. While sigma_hat is
     0 (warm-up, or no valid block yet) every code is NO_DATA.
3. Row guard (per sample j): ``s_row = f32(1.4826) * lowmed(|re|, |im|)`` over
   the row's used values (same "used" rule and ``thr``; a row with < 50% used values is
   never guarded). The row is GUARDED iff ``s_row > f32(1.5) * sigma_hat``; it
   is then coded with ``sigma_row = s_row`` instead of sigma_hat (a broadband
   noise surge keeps today's per-row 1/scale semantics; on normal noise the
   median of ~3.5k-5.5k numbers is within 2% of sigma_hat, ~30 sd from the
   trip, and a point source trips it only above ~500 sigma per sample).
4. Codes, per component, with ``sig`` = sigma_hat or sigma_row:
   ``step = f32(s) * sig``; ``inv = f32(1) / step``; ``q = y * inv``;
   ``code = clip(rint(q), -(2^(c-1) - 1), 2^(c-1) - 1)`` — MID-TREAD
   (half-to-even), c = 4/5/6 for main values (s = s_main), 8 for heavy
   values (s = s_heavy). ``-2^(c-1)`` is the NO_DATA sentinel: emitted for a
   non-finite component and for every component while sigma is 0; it is
   never a quantisation level.
5. Owner decode in sigma_hat units: ``v = f32(code) * s`` (s_main or
   s_heavy), NO_DATA -> 0.0; gridding multiplies by ``f32(sqrt(n_v))``. No
   row factor (a guarded row is thereby weighted 1/sigma_row, today's
   semantics); fan-in zero-fill is 0.0 in DECODED space, never a code.

Steps: ``s_main`` 0.3534 (4-bit, 15 levels) / 0.1932 (5-bit) / 0.1055 (6-bit)
— Gaussian-optimal mid-tread; ``s_heavy = f32(2 * 250 * sqrt(n_max / N) / 127)``
= 0.0773 (gridded n_sub 2) / 0.0547 (n_sub 4) — the heaviest value clips at
2x its S/N for a 250-sigma image.

Why mid-tread (SPEC v0.1 said mid-rise): a mid-rise code cannot represent 0,
so exactly-zero (flagged / excised) values decode to a coherent
``+0.5 step (1 + i)``: a constant fake source of 5-8 sigma in b64 per
flagged antenna (Phase 0 §5). Mid-tread costs 0.07% (4-bit) / 0.01% (5-bit).

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

DEFAULT_S_MAIN: dict[int, float] = {4: 0.3534, 5: 0.1932, 6: 0.1055}
DEFAULT_S_HEAVY: dict[int, float] = {2: 0.0773, 4: 0.0547}   # by n_sub (gridded and baselines)
HEAVY_FRAC: float = 3e-5
RAIL_SIGMA: float = 250.0
SIGMA_FLOOR = F32(1e-30)


def s_heavy_for(n_max: int, n_image: int, rail: float = RAIL_SIGMA) -> np.float32:
    """Heavy step that clips the heaviest value (n_max) at 2x its S/N for a
    ``rail``-sigma image: ``f32(2 * rail * sqrt(n_max / N) / 127)``."""
    return F32(2.0 * rail * math.sqrt(n_max / n_image) / 127.0)


def no_data_code(comp_bits: int) -> int:
    """The NO_DATA sentinel, ``-2**(comp_bits - 1)`` (decodes to 0)."""
    return -(1 << (comp_bits - 1))


@dataclass(frozen=True)
class RowRule:
    """Parameters of the row rule. ``comp_bits`` = main bits per component."""

    comp_bits: int = 4
    s_main: float | None = None
    s_heavy: float = DEFAULT_S_HEAVY[2]
    tau_s: float = 3.0
    block_s: float = BLOCK_S
    est_stride: int = 16
    robust_k: float = 1.4826
    min_used_frac: float = 0.5
    min_count: int = 16
    warmup_blocks: int = 8
    guard_factor: float = 1.5
    #: v1.1: a value whose |re| and |im| are both <= used_floor * sigma_hat_prev is unused
    used_floor: float = 1e-6
    #: v1.1: est > reset_factor * sigma_hat resets sigma_hat = est (0 = never: v1.0)
    reset_factor: float = 1.5

    def __post_init__(self) -> None:
        if self.comp_bits not in (4, 5, 6):
            raise ValueError(f"comp_bits={self.comp_bits} not in (4, 5, 6)")
        if self.est_stride < 1 or self.warmup_blocks < 0:
            raise ValueError("est_stride >= 1 and warmup_blocks >= 0 required")

    @classmethod
    def for_format(cls, bits_main: int, n_sub: int, **kw) -> "RowRule":
        """Rule for a 2x format: ``bits_main`` 8/10/12, ``n_sub`` 2/4."""
        return cls(comp_bits=int(bits_main) // 2, s_heavy=DEFAULT_S_HEAVY[int(n_sub)], **kw)

    @property
    def bits_main(self) -> int:
        """Frame-v2 ``bits_main`` (complex bits per main value)."""
        return 2 * self.comp_bits

    @property
    def s_main_f32(self) -> np.float32:
        return F32(self.s_main if self.s_main is not None else DEFAULT_S_MAIN[self.comp_bits])

    @property
    def s_heavy_f32(self) -> np.float32:
        return F32(self.s_heavy)

    @property
    def alpha_f32(self) -> np.float32:
        return F32(-math.expm1(-self.block_s / self.tau_s))

    @property
    def used_floor_f32(self) -> np.float32:
        return F32(self.used_floor)

    @property
    def reset_factor_f32(self) -> np.float32:
        return F32(self.reset_factor)

    def used_threshold(self, sigma_prev) -> np.float32:
        """``thr = f32(f32(used_floor) * sigma_hat_prev)`` (0 while sigma_hat is 0)."""
        return F32(self.used_floor_f32 * F32(sigma_prev))

    @property
    def main_range(self) -> tuple[int, int]:
        c = (1 << (self.comp_bits - 1)) - 1
        return (-c, c)

    @property
    def heavy_range(self) -> tuple[int, int]:
        return (-127, 127)


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


def _used(re: np.ndarray, im: np.ndarray, thr=F32(0.0)) -> np.ndarray:
    """Finite, and |re| or |im| above ``thr`` (``thr = 0``: not exactly 0 + 0i)."""
    t = F32(thr)
    fin = np.isfinite(re) & np.isfinite(im)
    with np.errstate(invalid="ignore"):
        big = (np.abs(re) > t) | (np.abs(im) > t)
    return fin & big


def lower_median(a: np.ndarray) -> np.float32:
    """Element of rank ``(len - 1) // 2`` (ascending)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    k = (a.size - 1) // 2
    return F32(np.partition(a, k)[k])


def robust_estimate(re: np.ndarray, im: np.ndarray,
                    rule: RowRule, thr=F32(0.0)) -> tuple[np.float32 | None, int]:
    """Block estimate on the subsample (``thr``: the used threshold, v1.1).
    Returns ``(est or None, m_used)``."""
    re = np.asarray(re, dtype=np.float32)
    im = np.asarray(im, dtype=np.float32)
    used = _used(re, im, thr)
    m = int(used.sum())
    m_min = max(int(rule.min_count), int(math.ceil(rule.min_used_frac * re.size)))
    if m < m_min or m == 0:
        return None, m
    med = lower_median(np.concatenate([np.abs(re[used]), np.abs(im[used])]))
    est = F32(F32(rule.robust_k) * med)
    return (est if est > SIGMA_FLOOR else SIGMA_FLOOR), m


def ema_update(sigma_prev: np.float32, est: np.float32 | None,
               rule: RowRule) -> np.float32:
    sp = F32(sigma_prev)
    if est is None:
        return sp
    e = F32(est)
    if not (np.isfinite(sp) and sp > 0):
        return e
    if rule.reset_factor > 0 and e > F32(rule.reset_factor_f32 * sp):
        return e                                     # v1.1: a step up re-converges at once
    d = F32(e - sp)
    m = F32(rule.alpha_f32 * d)
    return F32(sp + m)


def row_scales(re: np.ndarray, im: np.ndarray, rule: RowRule, thr=F32(0.0)) -> np.ndarray:
    """Per-sample robust scale ``f32(1.4826) * lowmed`` over each row's used
    values (``re/im [S, V]``, used threshold ``thr``); 0 where the row has < 50% used values."""
    S, V = re.shape
    out = np.zeros(S, dtype=np.float32)
    used = _used(re, im, thr)
    need = max(int(rule.min_count), int(math.ceil(rule.min_used_frac * V)))
    for j in range(S):
        u = used[j]
        if int(u.sum()) < need:
            continue
        med = lower_median(np.concatenate([np.abs(re[j, u]), np.abs(im[j, u])]))
        out[j] = F32(F32(rule.robust_k) * med)
    return out


def quantise(re: np.ndarray, im: np.ndarray, step, code_range: tuple[int, int],
             no_data: int) -> np.ndarray:
    """Mid-tread codes ``int8 [..., 2]``. ``step`` is a scalar or an array
    broadcasting against ``re`` (e.g. per sample ``[S, 1]``); ``step <= 0`` or
    non-finite components give NO_DATA."""
    re = np.asarray(re, dtype=np.float32)
    im = np.asarray(im, dtype=np.float32)
    st = np.asarray(step, dtype=np.float32)
    ok_step = np.isfinite(st) & (st > 0)
    safe = np.where(ok_step, st, F32(1.0)).astype(np.float32)
    inv = (F32(1.0) / safe).astype(np.float32)
    q = np.stack([re * inv, im * inv], axis=-1).astype(np.float32, copy=False)
    lo, hi = code_range
    c = np.clip(np.rint(np.where(np.isfinite(q), q, F32(0.0))), lo, hi).astype(np.int8)
    bad = ~np.isfinite(q) | ~np.broadcast_to(ok_step, re.shape)[..., None]
    c[bad] = no_data
    return c


# ---------------------------------------------------------------------------
# Step 5 (owner)
# ---------------------------------------------------------------------------


def decode(codes: np.ndarray, s, comp_bits: int) -> np.ndarray:
    """Codes ``[..., 2]`` -> complex64 values in sigma_hat units (NO_DATA -> 0)."""
    c = np.asarray(codes)
    v = c.astype(np.float32) * F32(s)
    v[c == no_data_code(comp_bits)] = F32(0.0)
    return (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)


def decode_row(codes_heavy: np.ndarray, codes_main: np.ndarray,
               rule: RowRule) -> np.ndarray:
    """Heavy-first value vector ``[..., n_values]`` in sigma_hat units."""
    vh = decode(codes_heavy, rule.s_heavy_f32, 8)
    vm = decode(codes_main, rule.s_main_f32, rule.comp_bits)
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
    sigma: np.float32             # sigma_hat used for this block (0 = no data)
    sigma_row: np.ndarray         # f32 [S]: the sigma each sample was coded with
    guarded: np.ndarray           # bool [S]: row guard fired
    step_main: np.ndarray         # f32 [S] = s_main * sigma_row (frame step_main)
    step_heavy: np.ndarray        # f32 [S] = s_heavy * sigma_row
    est: np.float32 | None        # this block's estimate (None = skipped / warm-up)
    n_used: int                   # values used by the estimate


@dataclass
class RowQuantiser:
    """Row-rule state of ONE (stream, coarse DM) flow."""

    rule: RowRule
    n_contrib: np.ndarray         # int [n_values], heavy first
    n_heavy: int
    sigma: np.float32 = field(default_factory=lambda: F32(0.0))
    blocks_seen: int = 0

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
        S = x.shape[0]
        re, im = normalise(x, self._r)
        est, m = None, 0
        thr = rule.used_threshold(self.sigma)           # from sigma_hat BEFORE this block
        if self.blocks_seen >= rule.warmup_blocks:
            sub = slice(0, S, rule.est_stride)
            est, m = robust_estimate(re[sub], im[sub], rule, thr)
            self.sigma = ema_update(self.sigma, est, rule)
        self.blocks_seen += 1
        sig = F32(self.sigma)
        if sig > 0:
            srow = row_scales(re, im, rule, thr)
            guarded = srow > F32(F32(rule.guard_factor) * sig)
            sigma_row = np.where(guarded, srow, sig).astype(np.float32)
        else:
            guarded = np.zeros(S, bool)
            sigma_row = np.zeros(S, np.float32)
        step_m = (rule.s_main_f32 * sigma_row).astype(np.float32)
        step_h = (rule.s_heavy_f32 * sigma_row).astype(np.float32)
        nh = self.n_heavy
        ch = quantise(re[:, :nh], im[:, :nh], step_h[:, None], rule.heavy_range, no_data_code(8))
        cm = quantise(re[:, nh:], im[:, nh:], step_m[:, None], rule.main_range,
                      no_data_code(rule.comp_bits))
        return BlockCodes(ch, cm, sig, sigma_row, guarded, step_m, step_h, est, m)


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
    "BLOCK_NATIVE", "BLOCK_S", "NATIVE_SAMPLE_S", "DEFAULT_S_MAIN",
    "DEFAULT_S_HEAVY", "HEAVY_FRAC", "RAIL_SIGMA", "SIGMA_FLOOR", "s_heavy_for",
    "no_data_code", "RowRule", "rsqrt_n", "sqrt_n", "normalise", "lower_median",
    "robust_estimate", "ema_update", "row_scales", "quantise", "decode",
    "decode_row", "natural_weight", "BlockCodes", "RowQuantiser",
    "v1_rowmax_cint8",
]
