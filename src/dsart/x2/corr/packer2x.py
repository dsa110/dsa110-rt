"""GPU packer for the 2x corner turn (x2 WS-C deliverable 3).

Two modes over the stage-1 + static-sky output ``x`` ``(n_d, n_t, V)``
complex64 of one tier (values of all ``n_sub`` streams of the chgroup
concatenated, wire order, heavy first in each stream):

**Row rule** (SPEC §4, the WS-T/Phase-0 freeze; parameters are arguments):

1. ``y = x * (1/sqrt(n))`` -- equal noise per value;
2. per (stream, coarse DM): robust ``sigma`` = 1.4826 x the LOWER median of
   ``|re|, |im|`` over the block's samples (every ``robust_stride``-th
   sample), excluding components above ``clip_sigma`` x the previous
   sigma; slow EMA (``tau_s``); a block is quantised with the PREVIOUS
   sigma (its own estimate only for the first block), so quantisation never
   waits on its own statistics;
3. main values: ``code = clip(floor(y / (s_main sigma)), -2^(b-1), 2^(b-1)-1)``
   per component (mid-rise), ``b = bits_main / 2``; heavy values: cint8 with
   step ``s_heavy sigma``;
4. payload per sample: the heavy (re, im) int8 pairs, then the main codes
   as a little-endian bit stream (re then im per value), padded to a byte
   (SPEC §5). One row of bytes per (d, t): the C sender (WS-T) cuts
   datagrams of ``k = floor(8900 / bytes_per_sample)`` rows and writes the
   72-byte v2 headers (step values travel there for monitoring).

**v1 cint8** (today's frames, 1x-compat): per (d, t) row of a stream,
``amax``, ``scale = f32(f64(amax)/127)``, ``q = clip(rint(x * f32(1/scale)))``
-- bit-identical to ``TransportTx._encode_rows``.

Everything is torch: the CPU execution is the twin, and every op is exact
or IEEE-rounded identically on CPU and GPU (multiply, divide, floor,
clamp, sort-select), so codes and bytes match bit for bit. The bit
packer is a Triton kernel on CUDA (:func:`pack_bits_triton`, interpreter-
tested) and a bit-expansion fallback elsewhere.
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

# NB: no ``from __future__ import annotations`` (Triton constexpr annotations).

__all__ = [
    "RowRulePacker",
    "S_MAIN",
    "pack_bits_torch",
    "pack_bits_triton",
    "pack_v1",
]

S_MAIN = {4: 0.336, 5: 0.191, 6: 0.106}


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


def pack_v1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    """Row-rule quantiser + payload builder for one tier of one chgroup."""

    def __init__(
        self,
        *,
        n_d: int,
        value_offsets,
        n_contrib,
        n_heavy,
        bits_main: int,
        s_heavy: float = 0.1,
        s_main: float | None = None,
        tau_s: float = 10.0,
        block_s: float = 4096 * 32.768e-6,
        clip_sigma: float = 6.0,
        robust_stride: int = 4,
        device="cpu",
    ) -> None:
        if bits_main not in (8, 10, 12):
            raise ValueError("bits_main is per complex value: 8, 10 or 12")
        self.b = bits_main // 2
        self.bits_main = int(bits_main)
        self.s_main = float(S_MAIN[self.b] if s_main is None else s_main)
        self.s_heavy = float(s_heavy)
        self.alpha = float(1.0 - math.exp(-block_s / tau_s))
        self.clip = float(clip_sigma)
        self.stride = int(robust_stride)
        self.voff = [int(v) for v in np.asarray(value_offsets)]
        self.n_heavy = [int(h) for h in np.asarray(n_heavy)]
        n = np.asarray(n_contrib, np.float64)
        self.inv_sqrt_n = torch.from_numpy((1.0 / np.sqrt(n)).astype(np.float32)).to(device)
        self.n_sub = len(self.voff) - 1
        self.sigma = torch.full((n_d, self.n_sub), float("nan"), dtype=torch.float32,
                                device=device)
        self.device = torch.device(device)

    def reset(self) -> None:
        self.sigma.fill_(float("nan"))

    def bytes_per_sample(self, s: int) -> int:
        nv = self.voff[s + 1] - self.voff[s]
        return 2 * self.n_heavy[s] + ((nv - self.n_heavy[s]) * self.bits_main + 7) // 8

    def _robust(self, yr_s: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """``yr_s`` (n_d, n_t, V_s, 2); lower median of |comp| per d."""
        comp = yr_s[:, :: self.stride].abs().reshape(yr_s.shape[0], -1)
        lim = torch.where(torch.isnan(prev), torch.full_like(prev, float("inf")), self.clip * prev)
        comp = torch.where(comp <= lim[:, None], comp, torch.full_like(comp, float("inf")))
        n_ok = (comp < float("inf")).sum(dim=1)
        srt, _ = torch.sort(comp, dim=1)
        idx = ((n_ok - 1).clamp(min=0) // 2).unsqueeze(1)
        med = srt.gather(1, idx).squeeze(1)
        est = torch.tensor(1.4826, dtype=torch.float32, device=comp.device) * med
        return torch.where(n_ok > 0, est, prev)

    def pack(self, x: torch.Tensor) -> list[dict]:
        """``x`` complex64 ``(n_d, n_t, V)`` -> per stream ``{"payload":
        uint8 (n_d, n_t, bytes_per_sample), "step_main": (n_d,), "step_heavy":
        (n_d,), "sigma": (n_d,)}``. Advances the sigma EMA."""
        n_d, n_t, V = x.shape
        yr = torch.view_as_real(x.contiguous()) * self.inv_sqrt_n[None, None, :, None]
        out = []
        for s in range(self.n_sub):
            lo, hi = self.voff[s], self.voff[s + 1]
            nh = self.n_heavy[s]
            ys = yr[:, :, lo:hi]
            prev = self.sigma[:, s]
            est = self._robust(ys, prev)
            unset = torch.isnan(prev)
            use = torch.where(unset, est, prev)
            self.sigma[:, s] = torch.where(unset, est, prev + self.alpha * (est - prev))
            use = use.clamp(min=1e-30)
            step_m = self.s_main * use
            step_h = self.s_heavy * use
            lo_c, hi_c = -(1 << (self.b - 1)), (1 << (self.b - 1)) - 1
            h = torch.floor(ys[:, :, :nh] / step_h[:, None, None, None]).clamp(-128, 127).to(torch.int8)
            m = torch.floor(ys[:, :, nh:] / step_m[:, None, None, None]).clamp(lo_c, hi_c).to(torch.int8)
            rows = m.reshape(n_d * n_t, -1)
            if rows.is_cuda:
                packed = pack_bits_triton(rows, self.b)
            else:
                packed = pack_bits_torch(rows, self.b)
            payload = torch.cat([h.reshape(n_d * n_t, -1).view(torch.uint8), packed], dim=1)
            out.append({
                "payload": payload.reshape(n_d, n_t, -1),
                "step_main": step_m, "step_heavy": step_h, "sigma": use,
                "codes_heavy": h, "codes_main": m,
            })
        return out
