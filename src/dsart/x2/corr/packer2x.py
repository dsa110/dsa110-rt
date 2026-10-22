"""GPU packer for the 2x corner turn (x2 WS-C deliverable 3).

Two modes over the stage-1 + static-sky output ``x`` ``(n_d, n_t, V)``
complex64 of one tier (the values of all ``n_sub`` streams of the chgroup
concatenated in wire order, heavy first within each stream):

**Row rule** -- WS-T's reference :mod:`dsart.x2.quant` (SPEC §4) applied to
every (stream, coarse DM) flow at once. The packer must be bit-equal to
``quant.RowQuantiser.process_block`` on the same inputs, so it follows the
reference op for op in float32:

1. ``r = f32(1/sqrt(n))`` (float64, rounded once); ``y = f32(x) * r`` per
   component;
2. robust estimate on samples ``j % est_stride == 0``: values that are
   finite, not exactly zero (flagged) and -- if ``excl_sigma`` is finite
   and sigma is set -- within ``f32(excl) * sigma`` in both components;
   fewer than ``max(min_count, ceil(min_frac * M))`` of them -> no update;
   else ``est = f32(robust_k) * LOWER median`` (rank ``(2m-1)//2``) of the
   2m magnitudes;
3. EMA ``d = est - s; m = alpha*d; s = s + m`` (three f32 ops), ``alpha =
   f32(-expm1(-block_s/tau_s))``, initialised by the first estimate; the
   block is quantised with the sigma AFTER its own update;
4. ``step = f32(s_x) * sigma``, ``inv = f32(1)/step``, ``q = y*inv``;
   mid-tread ``rint`` clipped to ``+-(2^(c-1) - 1)`` (code ``-2^(c-1)`` is
   the "no data" sentinel and never emitted) or mid-rise ``floor`` clipped
   to ``[-2^(c-1), 2^(c-1)-1]``; non-finite q or an unset sigma -> 0.

Two Phase-0 rules that WS-T has not frozen in ``quant.py`` yet are
implemented here behind parameters that are OFF by default, so the
default packer stays bit-equal to the reference: ``hold_blocks`` (sigma
may not initialise before the static-sky warm-up ends) and ``guard_g``
(the per-row guard: a row whose own robust scale -- ``f32(robust_k)`` x
lower median of its non-zero ``|re|, |im|`` -- exceeds ``guard_g`` x sigma
is coded with that scale; the applied per-row scale is returned for the
header). Phase 0's tau 3 s / no exclusion / ``min_frac`` 0.5 are plain
parameter values.

Output per stream: ``payload`` uint8 ``(n_d, n_t, bytes_per_sample)`` laid
out exactly as ``prod_frame2.pack_samples`` (heavy (re, im) int8 pairs,
then the main codes as one little-endian bit stream, re then im per value,
zero-padded to a byte), plus the codes, sigma and steps for the headers.

**v1 cint8** (today's frames, 1x-compat): per (d, t) row of a stream,
``amax``, ``scale = f32(f64(amax)/127)``, ``q = clip(rint(x * f32(1/scale)))``
-- bit-identical to ``TransportTx._encode_rows``.

Everything is torch: the CPU execution is the twin, and every op is exact
or IEEE-rounded identically on CPU and GPU (mul, sub, add, div, rint,
floor, clamp, sort-select; one op per kernel, so no FMA contraction), so
codes and bytes match bit for bit. The bit packer is a Triton kernel on
CUDA (:func:`pack_bits_triton`, interpreter-tested) and a bit-expansion
fallback elsewhere.
"""
import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

# NB: no ``from __future__ import annotations`` (Triton constexpr annotations).

__all__ = [
    "DEFAULT_S_MAIN",
    "MODE_MIDRISE",
    "MODE_MIDTREAD",
    "PackerRule",
    "RowRulePacker",
    "heavy_step_phase0",
    "pack_bits_torch",
    "pack_bits_triton",
    "pack_v1",
]

MODE_MIDTREAD = "midtread"
MODE_MIDRISE = "midrise"
DEFAULT_S_MAIN = {
    (MODE_MIDTREAD, 4): 0.3534, (MODE_MIDTREAD, 5): 0.1932, (MODE_MIDTREAD, 6): 0.1055,
    (MODE_MIDRISE, 4): 0.336, (MODE_MIDRISE, 5): 0.191, (MODE_MIDRISE, 6): 0.1041,
}


def heavy_step_phase0(n_max: int, n_image: int, *, rail_sigma: float = 250.0) -> float:
    """Phase 0 §2 heavy step: ``s_h = 2 x rail x sqrt(n_max / N) / 127`` (clip at
    >= 2x the image rail for the heaviest value); 0.0773 at n_sub 2, 0.0547 at 4."""
    return 2.0 * rail_sigma * math.sqrt(n_max / n_image) / 127.0


@dataclass(frozen=True)
class PackerRule:
    """Field-for-field mirror of ``dsart.x2.quant.RowRule`` plus two Phase-0
    extras (off by default). Build it from the reference with
    :meth:`from_quant` so the two cannot drift."""

    comp_bits: int = 4
    mode: str = MODE_MIDTREAD
    s_main: float | None = None
    s_heavy: float = 0.1
    tau_s: float = 10.0
    block_s: float = 4096 * 32.768e-6
    est_stride: int = 16
    excl_sigma: float = 6.0
    robust_k: float = 1.4826
    min_frac: float = 0.125
    min_count: int = 16
    hold_blocks: int = 0
    guard_g: float | None = None

    SHARED = ("comp_bits", "mode", "s_main", "s_heavy", "tau_s", "block_s", "est_stride",
              "excl_sigma", "robust_k", "min_frac", "min_count")

    def __post_init__(self) -> None:
        if self.comp_bits not in (4, 5, 6):
            raise ValueError(f"comp_bits={self.comp_bits}")
        if self.mode not in (MODE_MIDTREAD, MODE_MIDRISE):
            raise ValueError(f"mode={self.mode!r}")
        if self.est_stride < 1:
            raise ValueError("est_stride must be >= 1")

    @property
    def midrise(self) -> bool:
        return self.mode == MODE_MIDRISE

    @property
    def bits_main(self) -> int:
        return 2 * self.comp_bits

    @property
    def s_main_value(self) -> float:
        return float(self.s_main if self.s_main is not None
                     else DEFAULT_S_MAIN[(self.mode, self.comp_bits)])

    @property
    def alpha(self) -> float:
        return float(np.float32(-math.expm1(-self.block_s / self.tau_s)))

    def code_range(self, comp_bits: int) -> tuple:
        lo, hi = -(1 << (comp_bits - 1)), (1 << (comp_bits - 1)) - 1
        return (lo, hi) if self.midrise else (lo + 1, hi)

    @classmethod
    def from_quant(cls, rule, **extra) -> "PackerRule":
        """From ``dsart.x2.quant.RowRule``; refuses reference fields this
        packer does not implement (a new rule never passes silently)."""
        names = {f.name for f in dataclasses.fields(rule)}
        unknown = names - set(cls.SHARED) - {"hold_blocks", "guard_g"}
        if unknown:
            raise ValueError(f"quant.RowRule fields the packer does not implement: {sorted(unknown)}")
        kw = {n: getattr(rule, n) for n in names}
        kw.update(extra)
        return cls(**kw)


@triton.jit
def _pack_bits_kernel(codes_ptr, out_ptr, n_comp, n_bytes,
                      B: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pb = tl.program_id(1)
    j = pb * BLOCK + tl.arange(0, BLOCK)
    mj = j < n_bytes
    bit0 = j * 8
    c0 = bit0 // B
    sh = bit0 - c0 * B
    MASK: tl.constexpr = (1 << B) - 1
    base = codes_ptr + row * n_comp
    v0 = tl.load(base + c0, mask=mj & (c0 < n_comp), other=0).to(tl.int32) & MASK
    v1 = tl.load(base + c0 + 1, mask=mj & (c0 + 1 < n_comp), other=0).to(tl.int32) & MASK
    v2 = tl.load(base + c0 + 2, mask=mj & (c0 + 2 < n_comp), other=0).to(tl.int32) & MASK
    acc = (v0 >> sh) | (v1 << (B - sh)) | (v2 << (2 * B - sh))
    tl.store(out_ptr + row * n_bytes + j, (acc & 255).to(tl.uint8), mask=mj)


def pack_bits_triton(codes: torch.Tensor, bits: int, block: int = 256) -> torch.Tensor:
    """``codes`` int8 ``(rows, n_comp)`` -> uint8 ``(rows, ceil(n_comp*bits/8))``."""
    rows, n_comp = codes.shape
    n_bytes = (n_comp * bits + 7) // 8
    out = torch.empty((rows, n_bytes), dtype=torch.uint8, device=codes.device)
    grid = (rows, triton.cdiv(n_bytes, block))
    _pack_bits_kernel[grid](codes.contiguous(), out, n_comp, n_bytes, B=bits, BLOCK=block)
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


class RowRulePacker:
    """Row-rule packer for one tier of one chgroup: every (stream, coarse DM)
    flow of the tier at once, each with its own sigma (see module doc)."""

    def __init__(self, *, n_d: int, value_offsets, n_contrib, n_heavy, rule: PackerRule,
                 device="cpu") -> None:
        self.rule = rule
        self.voff = [int(v) for v in np.asarray(value_offsets)]
        self.n_heavy = [int(h) for h in np.asarray(n_heavy)]
        n = np.asarray(n_contrib, np.int64)
        if n.size and n.min() < 1:
            raise ValueError("contribution counts must be >= 1")
        self.r = torch.from_numpy((1.0 / np.sqrt(n.astype(np.float64))).astype(np.float32)).to(device)
        self.n_sub = len(self.voff) - 1
        self.sigma = torch.zeros((n_d, self.n_sub), dtype=torch.float32, device=device)
        self.blocks_seen = 0
        self.device = torch.device(device)

        def f32(v):
            return torch.tensor(float(np.float32(v)), dtype=torch.float32, device=device)

        self._k = f32(rule.robust_k)
        self._alpha = f32(rule.alpha)
        self._s_main = f32(rule.s_main_value)
        self._s_heavy = f32(rule.s_heavy)
        self._excl = f32(rule.excl_sigma) if math.isfinite(rule.excl_sigma) else None
        self._guard = f32(rule.guard_g) if rule.guard_g is not None else None
        self._one = f32(1.0)

    @property
    def bits_main(self) -> int:
        return self.rule.bits_main

    def reset(self) -> None:
        self.sigma.zero_()
        self.blocks_seen = 0

    def bytes_per_sample(self, s: int) -> int:
        nv = self.voff[s + 1] - self.voff[s]
        return 2 * self.n_heavy[s] + ((nv - self.n_heavy[s]) * self.bits_main + 7) // 8

    @staticmethod
    def _lower_median(a_re, a_im, keep):
        """Rows ``(R, M)`` -> (lower median of the kept 2m magnitudes, m)."""
        inf = torch.full_like(a_re, float("inf"))
        comps = torch.cat([torch.where(keep, a_re, inf), torch.where(keep, a_im, inf)], dim=1)
        m = keep.sum(dim=1)
        srt, _ = torch.sort(comps, dim=1)
        idx = ((2 * m - 1).clamp(min=0) // 2).unsqueeze(1)
        return srt.gather(1, idx).squeeze(1), m

    def _estimate(self, re, im, sp):
        rule = self.rule
        sub_re = re[:, :: rule.est_stride].reshape(re.shape[0], -1)
        sub_im = im[:, :: rule.est_stride].reshape(im.shape[0], -1)
        keep = (torch.isfinite(sub_re) & torch.isfinite(sub_im)
                & ~((sub_re == 0) & (sub_im == 0)))
        a_re, a_im = sub_re.abs(), sub_im.abs()
        if self._excl is not None:
            thr = (self._excl * sp)[:, None]
            on = (torch.isfinite(sp) & (sp > 0))[:, None]
            keep = keep & (~on | ((a_re <= thr) & (a_im <= thr)))
        med, m = self._lower_median(a_re, a_im, keep)
        m_min = max(int(rule.min_count), int(math.ceil(rule.min_frac * sub_re.shape[1])))
        return self._k * med, (m >= m_min) & (m > 0)

    def _ema(self, sp, est, ok):
        is_set = torch.isfinite(sp) & (sp > 0)
        d = est - sp
        mm = self._alpha * d
        return torch.where(ok, torch.where(is_set, sp + mm, est), sp)

    def _codes(self, re, im, step, comp_bits):
        """``re, im`` (n_d, n_t, n); ``step`` (n_d, n_t) -> int8 (n_d, n_t, n, 2)."""
        lo, hi = self.rule.code_range(comp_bits)
        valid = torch.isfinite(step) & (step > 0)
        inv = (self._one / torch.where(valid, step, self._one))[:, :, None]
        q = torch.stack([re * inv, im * inv], dim=-1)
        q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
        c = torch.floor(q) if self.rule.midrise else torch.round(q)
        c = c.clamp(lo, hi)
        return torch.where(valid[:, :, None, None], c, torch.zeros_like(c)).to(torch.int8)

    def _row_scale(self, re, im):
        n_d, n_t, V = re.shape
        rr, ii = re.reshape(n_d * n_t, V), im.reshape(n_d * n_t, V)
        keep = torch.isfinite(rr) & torch.isfinite(ii) & ~((rr == 0) & (ii == 0))
        med, m = self._lower_median(rr.abs(), ii.abs(), keep)
        return torch.where(m > 0, self._k * med, torch.zeros_like(med)).reshape(n_d, n_t)

    def pack(self, x: torch.Tensor) -> list:
        """``x`` complex64 ``(n_d, n_t, V)`` -> per stream ``{"payload": uint8
        (n_d, n_t, bytes_per_sample), "codes_heavy", "codes_main", "sigma"
        (n_d,), "step_main", "step_heavy" (n_d,), "row_sigma" (n_d, n_t)}``."""
        rule = self.rule
        xr = torch.view_as_real(x.to(torch.complex64).contiguous())
        hold = self.blocks_seen < int(rule.hold_blocks)
        out = []
        for s in range(self.n_sub):
            lo, hi = self.voff[s], self.voff[s + 1]
            nh = self.n_heavy[s]
            r = self.r[lo:hi]
            re = xr[:, :, lo:hi, 0] * r
            im = xr[:, :, lo:hi, 1] * r
            sp = self.sigma[:, s]
            if hold:
                sig = sp.clone()
            else:
                est, ok = self._estimate(re, im, sp)
                sig = self._ema(sp, est, ok)
                self.sigma[:, s] = sig
            row_sig = sig[:, None].expand(-1, re.shape[1])
            if self._guard is not None:
                rs = self._row_scale(re, im)
                row_sig = torch.where(rs > (self._guard * sig)[:, None], rs, row_sig)
            h = self._codes(re[:, :, :nh], im[:, :, :nh], self._s_heavy * row_sig, 8)
            m = self._codes(re[:, :, nh:], im[:, :, nh:], self._s_main * row_sig, rule.comp_bits)
            n_d, n_t = re.shape[0], re.shape[1]
            rows = m.reshape(n_d * n_t, -1)
            packed = (pack_bits_triton(rows, rule.comp_bits) if rows.is_cuda
                      else pack_bits_torch(rows, rule.comp_bits))
            payload = torch.cat([h.reshape(n_d * n_t, -1).view(torch.uint8), packed], dim=1)
            out.append({
                "payload": payload.reshape(n_d, n_t, -1),
                "codes_heavy": h, "codes_main": m, "sigma": sig,
                "step_main": self._s_main * sig, "step_heavy": self._s_heavy * sig,
                "row_sigma": row_sig,
            })
        self.blocks_seen += 1
        return out
