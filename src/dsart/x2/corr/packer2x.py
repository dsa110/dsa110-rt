"""GPU packer for the 2x corner turn: the frozen row rule (SPEC §4).

Input: the stage-1 + static-sky output ``x`` ``(n_d, S, V)`` complex64 of one
tier (S = 256 tier A / 128 tier B per corr block; the values of the
chgroup's ``n_sub`` streams concatenated in wire order, heavy first within
each stream). Every (stream, coarse DM) flow gets its own sigma_hat.

The packer is bit-equal to WS-T's reference
``dsart.x2.quant.RowQuantiser.process_block`` on the same inputs and takes
its parameters from ``quant.RowRule`` itself (Phase 0's rules are on, none
is optional here):

1. ``r = f32(1/sqrt(n))`` (``quant.rsqrt_n``); ``y = f32(x) * r``.
2. Blocks ``b < warmup_blocks`` (8, the static-sky warm-up) never touch
   sigma_hat. Later blocks estimate on the rows ``j % est_stride == 0``: a
   value is *used* iff finite and not 0 + 0i; fewer than ``max(min_count,
   ceil(min_used_frac * M))`` used (blocks < 50% non-zero) -> no update; else
   ``est = max(f32(1.4826) * lowmed, 1e-30)`` with lowmed = the element of
   rank ``(2m - 1) // 2`` of the 2m magnitudes (EXACT: a segmented sort of
   the 1/16 subsample per flow). EMA ``d = est - s; m = alpha*d; s = s + m``
   (tau 3 s), the first update sets ``s = est``; the block is coded with the
   sigma AFTER its own update.
3. Row guard: ``s_row = f32(1.4826) * lowmed`` of the row's used magnitudes
   (rows < 50% used are never guarded); a row is guarded iff ``s_row >
   f32(1.5) * sigma_hat`` and is then coded with ``s_row``.
4. ``step = f32(s) * sigma``; ``inv = f32(1) / step``; ``q = y * inv``;
   mid-tread ``rint`` clipped to ``+-(2^(c-1) - 1)``; ``-2^(c-1)`` (NO_DATA)
   for a non-finite q and for every code while sigma is 0.

**Exact row medians without sorting every row.** ``f32(k * u)`` is monotone
in ``u``, so ``s_row > G`` (``G = f32(1.5) * sigma_hat``) iff ``lowmed >= t``
with ``t`` the smallest float32 for which ``f32(k * t) > G``
(:func:`guard_threshold`, exact by construction and checked); and
``lowmed >= t`` iff at most ``(2m - 1) // 2`` of the row's 2m magnitudes are
``< t``. The guard DECISION therefore costs one counting pass over the data
(:func:`row_stats_triton`); the exact ``s_row`` (a sort) is computed only
for guarded rows: up to ``guard_capacity`` (8) per flow on the device with
fixed shapes (no host sync), and -- if a flow has more guarded rows in one
block (a broadband surge) -- by :meth:`RowRulePacker.resolve_overflow`, which
the 2x pipeline calls at its D2H sync point (the CPU path resolves inline).
The CUDA pipeline runs with ``guard_capacity=0``: on normal noise no row is
guarded, so the steady state does no row sorting at all, and every flow with
a guarded row is recoded exactly at the sync point.

Kernels (Triton on CUDA, launched with ``enable_fp_fusion=False``; a torch
twin is the CPU path and the test oracle):

* :func:`row_stats_triton` -- per row: used count, count of magnitudes < t;
* :func:`codes_triton` -- per value: ``y = x * r``, ``q = y * inv_row``,
  rint by the 1.5 * 2^23 magic constant after clamping |q| <= 256 (exact
  half-to-even; no FMA, no libdevice: the interpreter runs it), clip,
  NO_DATA; int8 (re, im) codes, heavy (c = 8) and main (c = comp_bits);
* :func:`pack_bits_triton` -- main codes into the frame-v2 little-endian bit
  stream, written straight into the payload rows.

Every float operation is a single IEEE op (mul, sub, add, div, compare) or a
selection, identical on CPU, GPU and in numpy, so the codes match quant.py
bit for bit. Payload per stream: uint8 ``(n_d, S, bytes_per_sample)`` laid out
exactly as ``prod_frame2.pack_samples``.

``pack_v1`` (today's per-row max-abs cint8, 1x-compat) is unchanged.
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

from dsart.x2 import quant

# NB: no ``from __future__ import annotations`` (Triton constexpr annotations).

__all__ = [
    "RowRulePacker",
    "codes_torch",
    "codes_triton",
    "guard_threshold",
    "pack_bits_torch",
    "pack_bits_triton",
    "pack_v1",
    "row_stats_torch",
    "row_stats_triton",
]

_F32_MAX = tl.constexpr(3.4028234663852886e38)
_RINT_MAGIC = tl.constexpr(12582912.0)          # 1.5 * 2**23: (q + M) - M = rint(q), |q| <= 2**22
_RINT_CLAMP = tl.constexpr(256.0)               # |q| clamp before the magic rint (> every code range)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _row_stats_kernel(x_ptr, r_ptr, t_ptr, m_ptr, c_ptr, S, V, s_d, s_t,
                      BLOCK_V: tl.constexpr):
    """One program per (flow d, row j): ``m`` = used values, ``c`` = used
    magnitudes (re and im) ``< t[d]``."""
    row = tl.program_id(0)
    d = row // S
    j = row - d * S
    t = tl.load(t_ptr + d)
    base = x_ptr + d * s_d + j * s_t
    m = tl.zeros((BLOCK_V,), tl.int32)
    c = tl.zeros((BLOCK_V,), tl.int32)
    for v0 in range(0, V, BLOCK_V):
        v = v0 + tl.arange(0, BLOCK_V)
        mv = v < V
        xre = tl.load(base + 2 * v, mask=mv, other=0.0)
        xim = tl.load(base + 2 * v + 1, mask=mv, other=0.0)
        r = tl.load(r_ptr + v, mask=mv, other=0.0)
        yre = xre * r
        yim = xim * r
        are = tl.abs(yre)
        aim = tl.abs(yim)
        used = (are <= _F32_MAX) & (aim <= _F32_MAX) & ((yre != 0.0) | (yim != 0.0))
        m += used.to(tl.int32)
        c += (used & (are < t)).to(tl.int32) + (used & (aim < t)).to(tl.int32)
    tl.store(m_ptr + row, tl.sum(m, axis=0))
    tl.store(c_ptr + row, tl.sum(c, axis=0))


@triton.jit
def _codes_kernel(x_ptr, r_ptr, invm_ptr, invh_ptr, okm_ptr, okh_ptr, out_ptr, S, V, NH,
                  s_d, s_t, CMAX: tl.constexpr, BLOCK_V: tl.constexpr):
    """Program (row = d * S + j, value block): int8 (re, im) codes of one row
    segment; values ``v < NH`` are heavy (8-bit, step inv_h), the rest main."""
    row = tl.program_id(0)
    vb = tl.program_id(1)
    d = row // S
    j = row - d * S
    v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    mv = v < V
    base = x_ptr + d * s_d + j * s_t
    xre = tl.load(base + 2 * v, mask=mv, other=0.0)
    xim = tl.load(base + 2 * v + 1, mask=mv, other=0.0)
    r = tl.load(r_ptr + v, mask=mv, other=0.0)
    heavy = v < NH
    inv = tl.where(heavy, tl.load(invh_ptr + row), tl.load(invm_ptr + row))
    ok = tl.where(heavy, tl.load(okh_ptr + row), tl.load(okm_ptr + row)) != 0
    hi = tl.where(heavy, 127.0, CMAX * 1.0)
    nod = tl.where(heavy, -128, -CMAX - 1)
    yre = xre * r
    yim = xim * r
    qre = yre * inv
    qim = yim * inv
    fre = tl.abs(qre) <= _F32_MAX
    fim = tl.abs(qim) <= _F32_MAX
    sre = tl.minimum(tl.maximum(tl.where(fre, qre, 0.0), -_RINT_CLAMP), _RINT_CLAMP)
    sim = tl.minimum(tl.maximum(tl.where(fim, qim, 0.0), -_RINT_CLAMP), _RINT_CLAMP)
    rre = (sre + _RINT_MAGIC) - _RINT_MAGIC
    rim = (sim + _RINT_MAGIC) - _RINT_MAGIC
    cre = tl.minimum(tl.maximum(rre, -hi), hi).to(tl.int32)
    cim = tl.minimum(tl.maximum(rim, -hi), hi).to(tl.int32)
    cre = tl.where(fre & ok, cre, nod)
    cim = tl.where(fim & ok, cim, nod)
    o = out_ptr + row * (2 * V) + 2 * v
    tl.store(o, cre.to(tl.int8), mask=mv)
    tl.store(o + 1, cim.to(tl.int8), mask=mv)


@triton.jit
def _pack_bits_kernel(codes_ptr, out_ptr, n_comp, n_bytes, s_in, s_out,
                      B: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pb = tl.program_id(1)
    j = pb * BLOCK + tl.arange(0, BLOCK)
    mj = j < n_bytes
    bit0 = j * 8
    c0 = bit0 // B
    sh = bit0 - c0 * B
    MASK: tl.constexpr = (1 << B) - 1
    base = codes_ptr + row * s_in
    v0 = tl.load(base + c0, mask=mj & (c0 < n_comp), other=0).to(tl.int32) & MASK
    v1 = tl.load(base + c0 + 1, mask=mj & (c0 + 1 < n_comp), other=0).to(tl.int32) & MASK
    v2 = tl.load(base + c0 + 2, mask=mj & (c0 + 2 < n_comp), other=0).to(tl.int32) & MASK
    acc = (v0 >> sh) | (v1 << (B - sh)) | (v2 << (2 * B - sh))
    tl.store(out_ptr + row * s_out + j, (acc & 255).to(tl.uint8), mask=mj)


# ---------------------------------------------------------------------------
# launchers + torch twins
# ---------------------------------------------------------------------------


def _check_xs(xs: torch.Tensor) -> None:
    """``xs`` = float32 ``(n_d, S, V, 2)`` view with (V, 2) contiguous rows."""
    if xs.dtype != torch.float32 or xs.dim() != 4 or xs.shape[3] != 2:
        raise ValueError(f"xs must be float32 (n_d, S, V, 2), got {xs.dtype} {tuple(xs.shape)}")
    if xs.stride(3) != 1 or xs.stride(2) != 2:
        raise ValueError("xs rows must be contiguous (V, 2)")


def row_stats_triton(xs: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *, block_v: int = 256):
    """``(m, cnt)`` int32 ``(n_d, S)`` (see :func:`row_stats_torch`)."""
    _check_xs(xs)
    n_d, S, V, _ = xs.shape
    m = torch.empty((n_d, S), dtype=torch.int32, device=xs.device)
    c = torch.empty_like(m)
    if n_d * S == 0:
        return m, c
    _row_stats_kernel[(n_d * S,)](xs, r.contiguous(), t.contiguous(), m, c, S, V,
                                  xs.stride(0), xs.stride(1), BLOCK_V=block_v,
                                  enable_fp_fusion=False)
    return m, c


def row_stats_torch(xs: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
    """Twin of :func:`row_stats_triton`: per row the used count ``m`` and the
    number of used magnitudes (re and im) below the flow's ``t``."""
    y = xs * r[None, None, :, None]
    a = y.abs()
    used = torch.isfinite(y).all(-1) & (y != 0).any(-1)
    m = used.sum(-1, dtype=torch.int32)
    lt = (a < t[:, None, None, None]) & used[..., None]
    return m, lt.sum((-1, -2), dtype=torch.int32)


def codes_triton(xs, r, inv_m, inv_h, ok_m, ok_h, *, n_heavy: int, comp_bits: int,
                 block_v: int = 256) -> torch.Tensor:
    """int8 codes ``(n_d, S, V, 2)`` (see :func:`codes_torch`)."""
    _check_xs(xs)
    n_d, S, V, _ = xs.shape
    out = torch.empty((n_d, S, V, 2), dtype=torch.int8, device=xs.device)
    if n_d * S * V == 0:
        return out
    grid = (n_d * S, triton.cdiv(V, block_v))
    _codes_kernel[grid](xs, r.contiguous(), inv_m.contiguous(), inv_h.contiguous(),
                        ok_m.to(torch.int8).contiguous(), ok_h.to(torch.int8).contiguous(), out,
                        S, V, int(n_heavy), xs.stride(0), xs.stride(1),
                        CMAX=(1 << (comp_bits - 1)) - 1, BLOCK_V=block_v, enable_fp_fusion=False)
    return out


def codes_torch(xs, r, inv_m, inv_h, ok_m, ok_h, *, n_heavy: int, comp_bits: int) -> torch.Tensor:
    """Twin of :func:`codes_triton` == ``quant.quantise``: ``q = (x * r) *
    inv_row``; mid-tread ``rint`` clipped to ``+-(2^(c-1) - 1)``; NO_DATA
    ``-2^(c-1)`` for non-finite q or a row whose step is not finite > 0.
    ``inv_*`` / ``ok_*`` are per row ``(n_d, S)``; the first ``n_heavy``
    values use the heavy step and c = 8."""
    y = xs * r[None, None, :, None]
    V = xs.shape[2]
    nh = int(n_heavy)
    heavy = (torch.arange(V, device=xs.device) < nh)[None, None, :, None]
    inv = torch.where(heavy, inv_h[:, :, None, None], inv_m[:, :, None, None])
    q = y * inv
    fin = torch.isfinite(q)
    cm = (1 << (comp_bits - 1)) - 1
    hi = torch.where(heavy, torch.tensor(127.0, device=xs.device),
                     torch.tensor(float(cm), device=xs.device))
    c = torch.round(torch.where(fin, q, torch.zeros_like(q)))
    c = torch.minimum(torch.maximum(c, -hi), hi).to(torch.int8)
    ok = torch.where(heavy, ok_h[:, :, None, None], ok_m[:, :, None, None])
    nod = torch.where(heavy, torch.tensor(-128, dtype=torch.int8, device=xs.device),
                      torch.tensor(-cm - 1, dtype=torch.int8, device=xs.device))
    return torch.where(fin & ok, c, nod)


def pack_bits_triton(codes: torch.Tensor, bits: int, block: int = 256,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """``codes`` int8 ``(rows, n_comp)`` (inner dim contiguous, any row
    stride) -> uint8 ``(rows, ceil(n_comp*bits/8))``, into ``out`` if given
    (inner dim contiguous, any row stride: e.g. the main part of payload rows)."""
    rows, n_comp = codes.shape
    n_bytes = (n_comp * bits + 7) // 8
    if codes.stride(1) != 1:
        codes = codes.contiguous()
    if out is None:
        out = torch.empty((rows, n_bytes), dtype=torch.uint8, device=codes.device)
    elif out.shape != (rows, n_bytes) or out.stride(1) != 1 or out.dtype != torch.uint8:
        raise ValueError(f"out must be uint8 ({rows}, {n_bytes}) with a contiguous inner dim")
    if rows * n_bytes == 0:
        return out
    grid = (rows, triton.cdiv(n_bytes, block))
    _pack_bits_kernel[grid](codes, out, n_comp, n_bytes, codes.stride(0), out.stride(0),
                            B=bits, BLOCK=block)
    return out


def pack_bits_torch(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-expansion twin of :func:`pack_bits_triton`."""
    rows, n_comp = codes.shape
    c = codes.to(torch.int32) & ((1 << bits) - 1)
    sh = torch.arange(bits, device=codes.device, dtype=torch.int32)
    b = ((c.unsqueeze(-1) >> sh) & 1).reshape(rows, n_comp * bits)
    pad = (-(n_comp * bits)) % 8
    if pad:
        b = torch.cat([b, b.new_zeros((rows, pad))], dim=1)
    w = (1 << torch.arange(8, device=codes.device, dtype=torch.int32))
    return (b.reshape(rows, -1, 8) * w).sum(-1).to(torch.uint8)


def pack_v1(x: torch.Tensor):
    """Today's v1 cint8 over the last axis: ``x`` complex ``(..., V)`` ->
    ``(q int8 (..., V, 2), scales float32 (...))``, == ``_encode_rows``."""
    ri = torch.view_as_real(x.to(torch.complex64).contiguous())
    amax = ri.abs().flatten(-2).amax(-1) if ri.shape[-2] else ri.new_zeros(ri.shape[:-2])
    scales = (amax.double() / 127.0).float()
    scales = torch.where(amax == 0.0, torch.ones_like(scales), scales)
    inv = (1.0 / scales.double()).float()
    q = torch.round(ri * inv[..., None, None]).clamp(-128.0, 127.0).to(torch.int8)
    return q, scales


# ---------------------------------------------------------------------------
# the guard threshold
# ---------------------------------------------------------------------------


def guard_threshold(G: torch.Tensor, k: float, *, iters: int = 4):
    """Smallest float32 ``t`` with ``f32(k * t) > G`` per element, and a
    validity mask (False where no float32 qualifies, e.g. ``G = inf``).

    Starts at ``f32(G / k)`` (float64 division, one rounding: within ~2 ulp
    of the answer) and walks with ``nextafter``; ``valid`` additionally
    proves minimality (``f32(k * prev(t)) <= G``). All float32 IEEE, so CPU
    and GPU agree."""
    k32 = torch.tensor(float(np.float32(k)), dtype=torch.float32, device=G.device)
    t = (G.double() / float(np.float32(k))).float()
    zero = torch.zeros_like(t)
    inf = torch.full_like(t, float("inf"))

    def pred(u):
        return (k32 * u) > G

    for _ in range(iters):
        dn = torch.nextafter(t, zero)
        t = torch.where(pred(dn), dn, t)
    for _ in range(iters):
        t = torch.where(pred(t), t, torch.nextafter(t, inf))
    valid = pred(t) & ~pred(torch.nextafter(t, zero)) & torch.isfinite(G)
    return t, valid


# ---------------------------------------------------------------------------
# the packer
# ---------------------------------------------------------------------------


class RowRulePacker:
    """Row-rule packer for one tier of one chgroup: every (stream, coarse DM)
    flow of the tier at once, each with its own sigma_hat (see module doc).

    ``rule`` is ``dsart.x2.quant.RowRule`` (e.g. ``RowRule.for_format(bits_main,
    n_sub)``); ``value_offsets`` / ``n_contrib`` / ``n_heavy`` describe the
    concatenated streams. ``use_triton`` defaults to CUDA."""

    def __init__(self, *, n_d: int, value_offsets, n_contrib, n_heavy, rule, device="cpu",
                 guard_capacity: int = 8, use_triton: bool | None = None) -> None:
        if not isinstance(rule, quant.RowRule):
            raise TypeError(f"rule must be dsart.x2.quant.RowRule, got {type(rule).__name__}")
        self.rule = rule
        self.voff = [int(v) for v in np.asarray(value_offsets)]
        self.n_heavy = [int(h) for h in np.asarray(n_heavy)]
        self.n_sub = len(self.voff) - 1
        if len(self.n_heavy) != self.n_sub:
            raise ValueError("n_heavy needs one entry per stream")
        self.device = torch.device(device)
        self.r = torch.from_numpy(quant.rsqrt_n(np.asarray(n_contrib))).to(self.device)
        if self.r.numel() != self.voff[-1]:
            raise ValueError("n_contrib must cover every value")
        self.sigma = torch.zeros((int(n_d), self.n_sub), dtype=torch.float32, device=self.device)
        self.blocks_seen = 0
        self.guard_capacity = max(0, int(guard_capacity))
        self.use_triton = (self.device.type == "cuda") if use_triton is None else bool(use_triton)
        self.n_overflow_resolved = 0

        def f32(v):
            return torch.tensor(float(np.float32(v)), dtype=torch.float32, device=self.device)

        self._k = f32(rule.robust_k)
        self._alpha_t = f32(rule.alpha_f32)
        self._s_main = f32(rule.s_main_f32)
        self._s_heavy = f32(rule.s_heavy_f32)
        self._guard = f32(rule.guard_factor)
        self._floor = f32(quant.SIGMA_FLOOR)
        self._one = f32(1.0)
        self._need = [max(int(rule.min_count),
                          int(math.ceil(rule.min_used_frac * (self.voff[s + 1] - self.voff[s]))))
                      for s in range(self.n_sub)]

    # ------------------------------------------------------------------

    @property
    def bits_main(self) -> int:
        return self.rule.bits_main

    def reset(self) -> None:
        self.sigma.zero_()
        self.blocks_seen = 0

    def bytes_per_sample(self, s: int) -> int:
        nv = self.voff[s + 1] - self.voff[s]
        return 2 * self.n_heavy[s] + ((nv - self.n_heavy[s]) * self.bits_main + 7) // 8

    # ------------------------------------------------------------------

    def _lowmed(self, y: torch.Tensor):
        """``y`` float32 ``(..., V, 2)`` -> (lower median of the used values'
        2m magnitudes, m) over the last two axes, exact (sort + select)."""
        used = torch.isfinite(y).all(-1) & (y != 0).any(-1)
        a = torch.where(used[..., None], y.abs(), torch.full_like(y, float("inf")))
        a = a.flatten(-2)
        srt = torch.sort(a, dim=-1).values
        m = used.sum(-1)
        idx = torch.div(2 * m - 1, 2, rounding_mode="floor").clamp(min=0)
        return srt.gather(-1, idx.unsqueeze(-1)).squeeze(-1), m

    def _estimate(self, xs: torch.Tensor, r: torch.Tensor):
        rule = self.rule
        n_d = xs.shape[0]
        sub = xs[:, :: rule.est_stride]
        y = sub * r[None, None, :, None]
        med, m = self._lowmed(y.reshape(n_d, -1, 2))
        M = sub.shape[1] * sub.shape[2]
        m_min = max(int(rule.min_count), int(math.ceil(rule.min_used_frac * M)))
        ok = (m >= m_min) & (m > 0)
        est = self._k * med
        est = torch.where(est > self._floor, est, self._floor)
        return est, ok, m

    def _ema(self, sp, est, ok):
        is_set = torch.isfinite(sp) & (sp > 0)
        d = est - sp
        mm = self._alpha_t * d
        upd = sp + mm
        return torch.where(ok, torch.where(is_set, upd, est), sp)

    def _code_stream(self, xs: torch.Tensor, r: torch.Tensor, sig: torch.Tensor, s: int,
                     capacity: int) -> dict:
        """Guard + codes + payload of stream ``s`` for the flows in ``xs``
        (float32 ``(n_f, S, V, 2)`` view) at sigma_hat ``sig`` ``(n_f,)``."""
        n_f, S, V, _ = xs.shape
        nh = self.n_heavy[s]
        dev = xs.device
        pos = sig > 0
        G = self._guard * sig
        t, valid = guard_threshold(G, float(self.rule.robust_k))
        t = torch.where(valid, t, torch.zeros_like(t))
        if self.use_triton:
            m_row, cnt = row_stats_triton(xs, r, t)
        else:
            m_row, cnt = row_stats_torch(xs, r, t)
        half = torch.div(2 * m_row - 1, 2, rounding_mode="floor")
        guarded = (pos & valid)[:, None] & (m_row >= self._need[s]) & (cnt <= half)
        sigma_row = torch.where(pos, sig, torch.zeros_like(sig))[:, None].expand(n_f, S).contiguous()
        K = min(int(capacity), S)
        if n_f and S and K:
            order = torch.argsort(guarded.to(torch.int8), dim=1, descending=True, stable=True)[:, :K]
            cand = torch.gather(guarded, 1, order)
            rows = xs[torch.arange(n_f, device=dev)[:, None], order]          # (n_f, K, V, 2)
            med, _ = self._lowmed(rows * r[None, None, :, None])
            s_c = self._k * med
            sigma_row.scatter_(1, order, torch.where(cand, s_c, torch.gather(sigma_row, 1, order)))
        overflow = guarded.sum(1) > K
        step_m = self._s_main * sigma_row
        step_h = self._s_heavy * sigma_row
        ok_m = torch.isfinite(step_m) & (step_m > 0)
        ok_h = torch.isfinite(step_h) & (step_h > 0)
        inv_m = self._one / torch.where(ok_m, step_m, self._one)
        inv_h = self._one / torch.where(ok_h, step_h, self._one)
        cb = self.rule.comp_bits
        if self.use_triton:
            codes = codes_triton(xs, r, inv_m, inv_h, ok_m, ok_h, n_heavy=nh, comp_bits=cb)
        else:
            codes = codes_torch(xs, r, inv_m, inv_h, ok_m, ok_h, n_heavy=nh, comp_bits=cb)
        bps = 2 * nh + ((V - nh) * 2 * cb + 7) // 8
        payload = torch.empty((n_f * S, bps), dtype=torch.uint8, device=dev)
        flat = codes.reshape(n_f * S, V * 2)
        if nh:
            payload[:, : 2 * nh] = flat[:, : 2 * nh].view(torch.uint8)
        main = flat[:, 2 * nh:]
        if self.use_triton:
            pack_bits_triton(main, cb, out=payload[:, 2 * nh:])
        else:
            payload[:, 2 * nh:] = pack_bits_torch(main, cb)
        return {
            "payload": payload.reshape(n_f, S, bps),
            "codes_heavy": codes[:, :, :nh], "codes_main": codes[:, :, nh:],
            "sigma": sig, "sigma_row": sigma_row, "step_main": step_m, "step_heavy": step_h,
            "guarded": guarded, "guard_overflow": overflow,
        }

    # ------------------------------------------------------------------

    def pack(self, x: torch.Tensor, *, resolve_overflow: bool = True) -> list:
        """One corr block of the tier: ``x`` complex ``(n_d, S, V)`` -> per
        stream ``{"payload" uint8 (n_d, S, bytes_per_sample), "codes_heavy",
        "codes_main" int8 (n_d, S, n, 2), "sigma" (n_d,) (sigma_hat after this
        block's update; 0 = no data yet), "sigma_row", "step_main",
        "step_heavy" (n_d, S) (the frame steps), "guarded" bool (n_d, S),
        "guard_overflow" bool (n_d,), "est" (n_d,) (NaN = no update),
        "n_used" (n_d,)}``.

        With ``resolve_overflow=False`` (the CUDA pipeline) nothing here waits
        on the device; the caller must then pass the result through
        :meth:`resolve_overflow` before using it."""
        xr = torch.view_as_real(x.to(torch.complex64).contiguous())
        n_d = xr.shape[0]
        do_est = self.blocks_seen >= int(self.rule.warmup_blocks)
        self.blocks_seen += 1
        out = []
        for s in range(self.n_sub):
            lo, hi = self.voff[s], self.voff[s + 1]
            xs = xr[:, :, lo:hi]
            r = self.r[lo:hi]
            sp = self.sigma[:, s]
            if do_est:
                est, ok, m_used = self._estimate(xs, r)
                sig = self._ema(sp, est, ok)
                self.sigma[:, s] = sig
                est = torch.where(ok, est, torch.full_like(est, float("nan")))
            else:
                sig = sp.clone()
                est = torch.full_like(sp, float("nan"))
                m_used = torch.zeros(n_d, dtype=torch.int64, device=self.device)
            res = self._code_stream(xs, r, sig, s, self.guard_capacity)
            res["est"] = est
            res["n_used"] = m_used
            out.append(res)
        if resolve_overflow:
            self.resolve_overflow(out, x)
        return out

    def resolve_overflow(self, results: list, x: torch.Tensor) -> list:
        """Recode the flows that had more than ``guard_capacity`` guarded rows
        in this block (exact s_row for all of them). Syncs with the device
        once per stream; a no-op unless a broadband surge tripped the guard
        on many rows. ``x`` is the tensor :meth:`pack` was given."""
        xr = None
        for s, res in enumerate(results):
            ov = res["guard_overflow"]
            if not bool(ov.any()):
                continue
            if xr is None:
                xr = torch.view_as_real(x.to(torch.complex64).contiguous())
            flows = torch.nonzero(ov).flatten()
            lo, hi = self.voff[s], self.voff[s + 1]
            xs = xr[flows][:, :, lo:hi]
            S = xs.shape[1]
            sub = self._code_stream(xs, self.r[lo:hi], res["sigma"][flows], s, capacity=S)
            for key in ("payload", "codes_heavy", "codes_main", "sigma_row", "step_main",
                        "step_heavy", "guarded", "guard_overflow"):
                res[key][flows] = sub[key]
            self.n_overflow_resolved += int(flows.numel())
        return results
