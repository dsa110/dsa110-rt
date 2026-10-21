"""Small Triton kernels for the time-resolution benches (prototypes, not production).

* combine_cells: fine-DM shift-and-add of S sub-band uv streams (packed cint8)
  straight into the occupied cells of a dense half-plane FFT input buffer.
* boxcar_chunkmax: multi-width boxcar on a padded cumsum, but writing only the
  per-(pixel, time-chunk) maximum instead of a per-voxel max/argmax cube.

Both are validated against numpy/torch references by `--selftest`
(TRITON_INTERPRET=1 runs them on CPU).

NB: no `from __future__ import annotations` here - Triton must see real
`tl.constexpr` annotation objects.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _combine_kernel(x_ptr, shift_ptr, cellpos_ptr, out_ptr,
                    S, T_in, C, T_out, PLANE,
                    BT: tl.constexpr, BC: tl.constexpr):
    f = tl.program_id(0)
    pt = tl.program_id(1)
    pc = tl.program_id(2)
    t = tl.arange(0, BT)[:, None] + pt * BT
    c = tl.arange(0, BC)[None, :] + pc * BC
    m = (t < T_out) & (c < C)
    acc_r = tl.zeros((BT, BC), dtype=tl.float32)
    acc_i = tl.zeros((BT, BC), dtype=tl.float32)
    zt = tl.zeros((BT, 1), dtype=tl.int32)
    for s in range(0, S):
        # shift loaded as a broadcast column (same value) - keeps the Triton
        # CPU interpreter happy; costs nothing on the GPU (one cached line).
        sh = tl.load(shift_ptr + zt + (f * S + s))
        v = tl.load(x_ptr + (t + sh) * C + c + s * T_in * C, mask=m, other=0).to(tl.int32)
        re = (v << 24) >> 24          # low byte, sign-extended
        im = v >> 8                   # high byte (arithmetic shift keeps sign)
        acc_r += re.to(tl.float32)
        acc_i += im.to(tl.float32)
    pos = tl.load(cellpos_ptr + c, mask=c < C, other=0)
    o = out_ptr + (f * T_out + t) * PLANE + 2 * pos
    tl.store(o, acc_r.to(tl.float16), mask=m)
    tl.store(o + 1, acc_i.to(tl.float16), mask=m)


def combine_cells(x_packed, shifts, cellpos, out_planes, T_out, BT=32, BC=64):
    """x_packed int16 [S, T_in, C] (lo byte = re, hi byte = im);
    shifts int32 [F, S]; cellpos int32 [C] (complex index inside one plane);
    out_planes fp16 [F*T_out, PLANE] (PLANE = 2 * H * (W//2+1)), pre-zeroed."""
    S, T_in, C = x_packed.shape
    F = shifts.shape[0]
    grid = (F, triton.cdiv(T_out, BT), triton.cdiv(C, BC))
    _combine_kernel[grid](x_packed, shifts, cellpos, out_planes, S, T_in, C, T_out,
                          out_planes.shape[1], BT=BT, BC=BC)


def combine_reference(x_packed, shifts, cellpos, T_out, plane):
    x = x_packed.to(torch.int32)
    re = ((x << 24) >> 24).float()
    im = (x >> 8).float()
    F, S = shifts.shape
    out = torch.zeros(F * T_out, plane, dtype=torch.float32)
    for f in range(F):
        ar = torch.zeros(T_out, x.shape[2])
        ai = torch.zeros(T_out, x.shape[2])
        for s in range(S):
            sh = int(shifts[f, s])
            ar += re[s, sh:sh + T_out]
            ai += im[s, sh:sh + T_out]
        out[f * T_out:(f + 1) * T_out, 2 * cellpos.long()] = ar
        out[f * T_out:(f + 1) * T_out, 2 * cellpos.long() + 1] = ai
    return out


@triton.jit
def _chunkmax_kernel(cs_ptr, w_ptr, off_ptr, sig_ptr, out_snr_ptr, out_tk_ptr, P,
                     CHUNK: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs = tl.arange(0, BLOCK) + pid_p * BLOCK
    m = offs < P
    best = tl.full((BLOCK,), -1e30, dtype=tl.float32)
    btk = tl.zeros((BLOCK,), dtype=tl.int32)
    zb = tl.zeros((BLOCK,), dtype=tl.int32)
    t0 = pid_c * CHUNK
    for tt in range(0, CHUNK):
        for k in tl.static_range(K):
            o = tl.load(off_ptr + zb + k)
            w = tl.load(w_ptr + zb + k)
            s = tl.load(sig_ptr + zb + k)
            tv = o + (t0 + tt)
            lo = tl.load(cs_ptr + offs + tv * P, mask=m, other=0.0).to(tl.float32)
            hi = tl.load(cs_ptr + offs + (tv + w) * P, mask=m, other=0.0).to(tl.float32)
            v = (hi - lo) * s
            better = v > best
            best = tl.where(better, v, best)
            btk = tl.where(better, tt * 64 + k, btk)
    tl.store(out_snr_ptr + pid_c * P + offs, best, mask=m)
    tl.store(out_tk_ptr + pid_c * P + offs, btk, mask=m)


def boxcar_chunkmax(cs, widths, offsets, sigma_inv, n_out, chunk=64, block=256):
    """cs [Tpad, P] (P = F*H*W flattened); returns (snr [n_out//chunk, P] fp32,
    packed t_in_chunk*64 + k [.., P] int32)."""
    P = cs.shape[1]
    nch = n_out // chunk
    snr = torch.empty(nch, P, dtype=torch.float32, device=cs.device)
    tk = torch.empty(nch, P, dtype=torch.int32, device=cs.device)
    grid = (triton.cdiv(P, block), nch)
    _chunkmax_kernel[grid](cs, widths, offsets, sigma_inv, snr, tk, P,
                           CHUNK=chunk, K=int(widths.numel()), BLOCK=block)
    return snr, tk


def chunkmax_reference(cs, widths, offsets, sigma_inv, n_out, chunk):
    cs = cs.float()
    K = widths.numel()
    vals = torch.stack([(cs[offsets[k] + widths[k]: offsets[k] + widths[k] + n_out]
                         - cs[offsets[k]: offsets[k] + n_out]) * sigma_inv[k] for k in range(K)])
    nch = n_out // chunk
    v = vals[:, :nch * chunk].reshape(K, nch, chunk, -1).amax(dim=(0, 2))
    return v


def pack_cint8(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    """int8 re/im -> int16 with re in the low byte and im in the high byte."""
    return ((im.to(torch.int32) << 8) | (re.to(torch.int32) & 0xFF)).to(torch.int16)
