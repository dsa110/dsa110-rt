"""Triton kernels of the owner engine: K1 (CSR combine) and K3b (detector).

Both perform exactly the fp32 operation sequence of their numpy twins
(``combine.k1_twin``, ``filters.k3b_twin``), so they agree bit for bit,
including under ``TRITON_INTERPRET=1`` on CPU tensors (CPU CI). FMA
contraction cannot change the result: every product is separated from the
following add by a ``tl.where`` select (masking / clipping), which LLVM does
not contract through.

K1 (``k1_combine``): one program per (cell block, row block, fine DM);
dynamic loop over the cell's entries (no 64-entry cap -- a 2x half-cell
can carry more), decode ``(code + off) * c_v`` from the time-minor code
ring, conjugate, accumulate, x ``0.5 (-1)^(u+w)``.

K3b (``k3b_detect``): one launch per filter (bank order), one program per
(pixel block, fine DM), a sequential loop over centres keeping the
filter's running-sum cascade in registers (``ORDER`` 0..3 constexpr; the
stage-1 lags {0, L2, L3, L2+L3} kept as four states -- duplicates when
L2 == L3 are bit-identical to the twin's deduplicated set), per-chunk max
and packed ``(t, k)`` read-modify-written in global memory across the
filter launches, raw outputs stored at tap pixels.

This is the correctness-first v1 of K3b (8 launches for b3x, the image
re-read per filter); the fused smem-stage-ring variant is a CP1
optimisation with the same twin (see ``bench/timeres/b8_owner2x.py``).

Loop bounds are ``tl.constexpr`` (one specialisation per (n_c, filter
length) / max entries): the Triton 3.1 interpreter cannot index a runtime
scalar with numpy >= 2.

No ``from __future__ import annotations``: Triton must see real
``tl.constexpr`` annotations.
"""
import numpy as np
import torch
import triton
import triton.language as tl

__all__ = ["k1_combine", "k3b_detect"]


@triton.jit
def _k1_kernel(codes_ptr, cv_ptr, ptr_ptr, es_ptr, ev_ptr, ec_ptr, hs_ptr, sh_ptr, out_ptr,
               n_act, n_t, t_lo, R, V, S, off,
               MAX_NE: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    f = tl.program_id(2)
    cells = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    tt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    cm = cells < n_act
    tm = tt < n_t
    p0 = tl.load(ptr_ptr + cells, mask=cm, other=0)
    p1 = tl.load(ptr_ptr + cells + 1, mask=cm, other=0)
    ne = p1 - p0
    acc_re = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for j in range(0, MAX_NE):
        has = (ne > j) & cm
        e = p0 + j
        s = tl.load(es_ptr + e, mask=has, other=0)
        v = tl.load(ev_ptr + e, mask=has, other=0)
        cj = tl.load(ec_ptr + e, mask=has, other=0)
        c = tl.load(cv_ptr + s * V + v, mask=has, other=0.0)
        sh = tl.load(sh_ptr + f * S + s, mask=has, other=0)
        row = (t_lo + tt)[:, None] - sh[None, :]
        slot = ((row % R) + R) % R
        base = (s * V + v)[None, :] * (R * 2) + slot * 2
        m2 = has[None, :] & tm[:, None]
        cre = tl.load(codes_ptr + base, mask=m2, other=0).to(tl.float32)
        cim = tl.load(codes_ptr + base + 1, mask=m2, other=0).to(tl.float32)
        vre = (cre + off) * c[None, :]
        vim = (cim + off) * c[None, :]
        vim = tl.where(cj[None, :] != 0, -vim, vim)
        vre = tl.where(m2, vre, 0.0)
        vim = tl.where(m2, vim, 0.0)
        acc_re = acc_re + vre
        acc_im = acc_im + vim
    hs = tl.load(hs_ptr + cells, mask=cm, other=0.0)
    o_re = acc_re * hs[None, :]
    o_im = acc_im * hs[None, :]
    optr = out_ptr + ((f * n_t + tt)[:, None] * n_act + cells[None, :]) * 2
    om = tm[:, None] & cm[None, :]
    tl.store(optr, o_re, mask=om)
    tl.store(optr + 1, o_im, mask=om)


def k1_combine(codes: torch.Tensor, c_v: torch.Tensor, csr_t: dict, shifts: torch.Tensor, *,
               t_lo: int, n_t: int, offset: float = 0.0, out: torch.Tensor = None,
               block_c: int = 64, block_t: int = 16) -> torch.Tensor:
    """``uv [F, n_t, n_act]`` complex64 (as float32 ``[..., 2]`` storage).

    ``codes`` int8 ``[S, V, R, 2]`` contiguous; ``c_v`` float32 ``[S, V]``;
    ``csr_t`` = :func:`csr_tensors`; ``shifts`` int32 ``[F, S]``.
    """
    S, V, R, two = codes.shape
    F = int(shifts.shape[0])
    n_act = int(csr_t["n_act"])
    if out is None:
        out = torch.empty((F, n_t, n_act, 2), dtype=torch.float32, device=codes.device)
    grid = (triton.cdiv(n_act, block_c), triton.cdiv(n_t, block_t), F)
    _k1_kernel[grid](codes, c_v, csr_t["cell_ptr"], csr_t["ent_stream"], csr_t["ent_value"],
                     csr_t["ent_conj"], csr_t["half_sign"], shifts, out,
                     n_act, int(n_t), int(t_lo), int(R), int(V), int(S), float(offset),
                     MAX_NE=int(csr_t["max_ne"]), BLOCK_C=block_c, BLOCK_T=block_t)
    return out


def csr_tensors(csr, device) -> dict:
    """Device copies of an :class:`~dsart.x2.search.csr.OwnerCsr`."""
    def t(a, dt):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    return dict(
        n_act=csr.n_active, max_ne=csr.max_entries,
        cell_ptr=t(csr.cell_ptr, torch.int32), ent_stream=t(csr.ent_stream, torch.int32),
        ent_value=t(csr.ent_value, torch.int32), ent_conj=t(csr.ent_conj, torch.int32),
        half_sign=t(np.float32(0.5) * csr.sign, torch.float32),
    )


# ---------------------------------------------------------------------------
# K3b
# ---------------------------------------------------------------------------


@triton.jit
def _xs(ring_ptr, row, r_start, row0, R, FP, base, pm, si, clip):
    ok = pm & (row >= r_start)
    slot = (((row - row0) % R) + R) % R
    x = tl.load(ring_ptr + slot * FP + base, mask=ok, other=0.0).to(tl.float32)
    y = x * si
    y = tl.minimum(tl.maximum(y, -clip), clip)
    return tl.where(ok, y, 0.0)


@triton.jit
def _k3b_kernel(ring_ptr, si_ptr, best_ptr, tk_ptr, tapidx_ptr, taps_ptr,
                R, F, P, row0, c_first, n_c, s_inv_k, clip, k_id, first,
                D, A, L1, L2, L3, n_tap,
                LEN: tl.constexpr, N_C: tl.constexpr, ORDER: tl.constexpr,
                CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid_p = tl.program_id(0)
    f = tl.program_id(1)
    pix = pid_p * BLOCK + tl.arange(0, BLOCK)
    pm = pix < P
    si = tl.load(si_ptr + f)
    FP = F * P
    base = f * P + pix
    r_start = c_first - A
    tap = tl.load(tapidx_ptr + pix, mask=pm, other=-1)
    s10 = tl.zeros((BLOCK,), dtype=tl.float32)
    s11 = tl.zeros((BLOCK,), dtype=tl.float32)
    s12 = tl.zeros((BLOCK,), dtype=tl.float32)
    s13 = tl.zeros((BLOCK,), dtype=tl.float32)
    s20 = tl.zeros((BLOCK,), dtype=tl.float32)
    s21 = tl.zeros((BLOCK,), dtype=tl.float32)
    s30 = tl.zeros((BLOCK,), dtype=tl.float32)
    cur = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    ctk = tl.zeros((BLOCK,), dtype=tl.int32)
    for it in range(0, N_C + LEN - 1):
        c = c_first - (LEN - 1) + it
        e = c + D
        if ORDER == 0:
            score = _xs(ring_ptr, e, r_start, row0, R, FP, base, pm, si, clip)
        elif ORDER == 1:
            s10 = (s10 + _xs(ring_ptr, e, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L1, r_start, row0, R, FP, base, pm, si, clip)
            score = s10
        elif ORDER == 2:
            s10 = (s10 + _xs(ring_ptr, e, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L1, r_start, row0, R, FP, base, pm, si, clip)
            s11 = (s11 + _xs(ring_ptr, e - L2, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L2 - L1, r_start, row0, R, FP, base, pm, si, clip)
            s20 = (s20 + s10) - s11
            score = s20
        else:
            s10 = (s10 + _xs(ring_ptr, e, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L1, r_start, row0, R, FP, base, pm, si, clip)
            s11 = (s11 + _xs(ring_ptr, e - L2, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L2 - L1, r_start, row0, R, FP, base, pm, si, clip)
            s12 = (s12 + _xs(ring_ptr, e - L3, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L3 - L1, r_start, row0, R, FP, base, pm, si, clip)
            s13 = (s13 + _xs(ring_ptr, e - L3 - L2, r_start, row0, R, FP, base, pm, si, clip)) \
                - _xs(ring_ptr, e - L3 - L2 - L1, r_start, row0, R, FP, base, pm, si, clip)
            s20 = (s20 + s10) - s11
            s21 = (s21 + s12) - s13
            s30 = (s30 + s20) - s21
            score = s30
        ci = c - c_first
        out_c = ci >= 0
        cic = tl.maximum(ci, 0)
        chunk = cic // CHUNK
        tin = cic % CHUNK
        is_start = out_c & (tin == 0)
        prev_b = tl.load(best_ptr + chunk * FP + base, mask=pm & is_start & (first == 0),
                         other=float("-inf"))
        prev_k = tl.load(tk_ptr + chunk * FP + base, mask=pm & is_start & (first == 0),
                         other=0)
        cur = tl.where(is_start, prev_b, cur)
        ctk = tl.where(is_start, prev_k, ctk)
        tl.store(taps_ptr + ((k_id * F + f) * n_tap + tap) * n_c + cic, score,
                 mask=pm & (tap >= 0) & out_c)
        v = score * s_inv_k
        better = (v > cur) & out_c
        cur = tl.where(better, v, cur)
        ctk = tl.where(better, tin * 256 + k_id, ctk)
        is_end = out_c & (tin == CHUNK - 1)
        tl.store(best_ptr + chunk * FP + base, cur, mask=pm & is_end)
        tl.store(tk_ptr + chunk * FP + base, ctk, mask=pm & is_end)


def k3b_detect(ring: torch.Tensor, *, ring_row0: int, c_first: int, n_c: int,
               sigma_inv: torch.Tensor, s_inv, bank, clip: float = 250.0, chunk: int = 64,
               tap_pixels=None, block: int = 128):
    """K3b for one DM-cycle: ``(best [n_ch, F, P] f32, tk int32, taps [K, F, n_tap, n_c])``.

    ``ring`` float16 ``[R, F, P]`` contiguous; ``sigma_inv`` float32 ``[F]``;
    ``s_inv`` ``[K]`` (host floats). One launch per filter, bank order.
    """
    R, F, P = ring.shape
    dev = ring.device
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    n_ch = n_c // chunk
    best = torch.empty((n_ch, F, P), dtype=torch.float32, device=dev)
    tk = torch.zeros((n_ch, F, P), dtype=torch.int32, device=dev)
    tapidx = torch.full((P,), -1, dtype=torch.int32, device=dev)
    n_tap = 0
    if tap_pixels is not None:
        tp = torch.as_tensor(np.asarray(tap_pixels), dtype=torch.long, device=dev)
        n_tap = int(tp.numel())
        tapidx[tp] = torch.arange(n_tap, dtype=torch.int32, device=dev)
    K = len(bank)
    taps = torch.zeros((K, F, max(n_tap, 1), n_c), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(P, block), F)
    s_inv = np.asarray(s_inv, dtype=np.float32)
    for k, flt in enumerate(bank.filters):
        st = list(flt.stages) + [1, 1, 1]
        _k3b_kernel[grid](ring, sigma_inv, best, tk, tapidx, taps,
                          int(R), int(F), int(P), int(ring_row0), int(c_first), int(n_c),
                          float(s_inv[k]), float(clip), int(k), int(k == 0),
                          int(flt.d), int(flt.a),
                          int(st[0]), int(st[1]), int(st[2]), int(max(n_tap, 1)),
                          LEN=int(flt.length), N_C=int(n_c), ORDER=int(flt.order),
                          CHUNK=int(chunk), BLOCK=int(block))
    return best, tk, (taps if n_tap else None)
