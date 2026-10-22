"""K2/K3a: half-plane uv cells -> masked, scaled fp16 images.

The image is today's ``irfft2(G_h, s=(N, N))`` (backward norm 1/N^2) of the
folded, checkerboarded half spectrum, times today's edge mask and a scalar
image scale -- written into the image ring as float16.

Three implementations of the same map:

* :func:`image_from_uv_torch` ``mode="irfft2"`` -- plain dense ``irfft2``
  (complex64; any device). Plain cuFFT is NOT the 2x target (+2.3 us/image).
* :func:`image_from_uv_torch` ``mode="pruned"`` -- the cuFFT column-pruned
  fallback: a batched length-N column IFFT over ONLY the columns that hold
  a cell off ``u = 0`` (<= 57 of 129 for the DSA-110 core), the
  ``u = 0``-only columns as exact constants, then a row ``irfft``.
* :func:`image_from_uv_twin` -- numpy twin of the CUDA warp-FFT pair in
  ``kernels_cuda`` (K2: radix-``N/32`` in-lane x 32-lane shuffle DIF per
  column; K3a: real-to-complex packing + a radix-``N/64`` x 32 lane
  128-point IFFT per row), fp32 with the SAME twiddle tables, in the same
  index order, so it validates the kernels' index/layout logic exactly and
  their numerics to fp32 rounding (FMA contraction aside).

C2R semantics: like ``numpy``/``torch`` ``irfft``, the imaginary parts of
the DC and Nyquist columns are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .csr import OwnerCsr

__all__ = [
    "FftPlan",
    "image_from_uv_torch",
    "image_from_uv_twin",
    "twiddles",
    "bitrev5",
]


def twiddles(n: int) -> np.ndarray:
    """fp32 ``exp(+2 pi i k / n)`` table, ``k = 0 .. n-1`` (computed in
    float64, rounded once -- the CUDA kernels get this exact table)."""
    k = np.arange(n, dtype=np.float64)
    return np.exp(2j * np.pi * k / n).astype(np.complex64)


def bitrev5(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    r = np.zeros_like(x)
    for b in range(5):
        r |= ((x >> b) & 1) << (4 - b)
    return r


@dataclass(frozen=True)
class FftPlan:
    """Static column/row bookkeeping for one owner CSR."""

    n_grid: int
    fft_cols: np.ndarray        # [n_fc] columns needing a column FFT
    const_cols: np.ndarray      # [n_cc] columns whose only cell is u = 0
    cell_col_slot: np.ndarray   # [n_act] index into fft_cols, or -1
    cell_const_slot: np.ndarray  # [n_act] index into const_cols, or -1
    cell_u: np.ndarray
    cell_w: np.ndarray

    @classmethod
    def build(cls, csr: OwnerCsr) -> "FftPlan":
        fc = csr.fft_cols
        cc = csr.const_cols
        col_slot = np.full(csr.n_active, -1, dtype=np.int32)
        const_slot = np.full(csr.n_active, -1, dtype=np.int32)
        pos_f = {int(w): i for i, w in enumerate(fc)}
        pos_c = {int(w): i for i, w in enumerate(cc)}
        for a in range(csr.n_active):
            w = int(csr.cell_w[a])
            if w in pos_f:
                col_slot[a] = pos_f[w]
            elif w in pos_c:
                const_slot[a] = pos_c[w]
        return cls(n_grid=csr.n_grid, fft_cols=fc, const_cols=cc,
                   cell_col_slot=col_slot, cell_const_slot=const_slot,
                   cell_u=csr.cell_u.copy(), cell_w=csr.cell_w.copy())

    @property
    def n_half(self) -> int:
        return self.n_grid // 2 + 1


def image_from_uv_torch(uv, plan: FftPlan, mask_scaled, *, mode: str = "pruned"):
    """``uv [B, n_act]`` complex64 tensor -> ``[B, N, N]`` float32 image
    (times ``mask_scaled [N, N]`` = edge mask x image scale x ... ).

    ``mode="pruned"``: column-pruned (cuFFT fallback), ``"irfft2"``: dense.
    """
    import torch

    B = uv.shape[0]
    N = plan.n_grid
    nh = plan.n_half
    dev = uv.device
    cu = torch.as_tensor(plan.cell_u, dtype=torch.long, device=dev)
    cw = torch.as_tensor(plan.cell_w, dtype=torch.long, device=dev)
    if mode == "irfft2":
        G = torch.zeros((B, N, nh), dtype=torch.complex64, device=dev)
        G[:, cu, cw] = uv
        img = torch.fft.irfft2(G, s=(N, N))
    elif mode == "pruned":
        fc = plan.fft_cols
        rows = torch.zeros((B, N, nh), dtype=torch.complex64, device=dev)
        if fc.size:
            sel = np.nonzero(plan.cell_col_slot >= 0)[0]
            cols = torch.zeros((B, int(fc.size), N), dtype=torch.complex64, device=dev)
            idx = torch.as_tensor(sel, dtype=torch.long, device=dev)
            cslot = torch.as_tensor(plan.cell_col_slot[sel], dtype=torch.long, device=dev)
            cols[:, cslot, cu[idx]] = uv[:, idx]
            # unnormalised column IFFT (norm="forward" puts 1/N on the forward side)
            colt = torch.fft.ifft(cols, dim=-1, norm="forward")            # [B, n_fc, N(y)]
            rows[:, :, torch.as_tensor(fc, dtype=torch.long, device=dev)] = colt.transpose(1, 2)
        if plan.const_cols.size:
            sel = np.nonzero(plan.cell_const_slot >= 0)[0]
            idx = torch.as_tensor(sel, dtype=torch.long, device=dev)
            w = torch.as_tensor(plan.cell_w[sel], dtype=torch.long, device=dev)
            rows[:, :, w] = uv[:, idx][:, None, :]
        img = torch.fft.irfft(rows, n=N, dim=-1) / float(N)                 # irfft has 1/N
    else:
        raise ValueError(f"mode={mode!r}")
    return img * mask_scaled


# ---------------------------------------------------------------------------
# numpy twin of the CUDA warp FFT pair
# ---------------------------------------------------------------------------


def _lane_dif32(x: np.ndarray, tw: np.ndarray, n_tw: int) -> np.ndarray:
    """32-point inverse DIF across the LAST axis (= the 32 lanes), radix 2,
    span 16 -> 1, exactly the shuffle network of the kernels. Output in
    bit-reversed lane order. ``tw`` is the ``n_tw``-point table."""
    x = x.copy()
    lane = np.arange(32)
    h = 16
    while h >= 1:
        partner = x[..., lane ^ h]
        top = (lane & h) == 0
        j = lane & (h - 1)
        wv = tw[(j * (n_tw // (2 * h))) % n_tw]                 # W_{2h}^j
        x = np.where(top, x + partner, (partner - x) * wv).astype(np.complex64)
        h //= 2
    return x


def k2_twin(col_vals: np.ndarray, n_grid: int) -> np.ndarray:
    """Column IFFT twin: ``col_vals [..., N]`` (u) -> ``[..., N]`` (y),
    unnormalised. Lane ``l`` holds ``u = l + 32 j`` (``j < R = N/32``)."""
    N = int(n_grid)
    R = N // 32
    if R * 32 != N:
        raise ValueError("K2 twin needs N = 32 R")
    tw = twiddles(N)
    x = col_vals.astype(np.complex64).reshape(col_vals.shape[:-1] + (R, 32))  # [..., j, l]
    # in-lane R-point inverse DFT over j: A[l, y2] = sum_j x[l + 32 j] W_R^{j y2}
    j = np.arange(R)
    WR = tw[((np.outer(j, j) * (N // R)) % N)]                                  # [j, y2]
    A = np.einsum("...jl,jy->...yl", x, WR).astype(np.complex64)                # [..., y2, l]
    # twiddle W_N^{l y2}
    l = np.arange(32)
    A = (A * tw[(np.outer(j, l)) % N]).astype(np.complex64)
    # 32-point across lanes for each y2 (output bit-reversed in lanes)
    B = _lane_dif32(A, tw, N)                                                   # [..., y2, lane]
    y1 = bitrev5(l)
    out = np.zeros(col_vals.shape[:-1] + (N,), dtype=np.complex64)
    # y = y2 + R * y1
    for y2 in range(R):
        out[..., y2 + R * y1] = B[..., y2, :]
    return out


def k3a_twin(rows: np.ndarray, n_grid: int) -> np.ndarray:
    """Row C2R twin: ``rows [..., N/2 + 1]`` half spectrum -> ``[..., N]``
    real, unnormalised (= N * irfft). Via the real-to-complex packing and
    an N/2-point complex IFFT, lanes holding ``k = k1 + 32 k2``."""
    N = int(n_grid)
    M = N // 2
    R2 = M // 32
    if R2 * 32 != M:
        raise ValueError("K3a twin needs N = 64 R2")
    tw = twiddles(N)
    Y = rows.astype(np.complex64)
    k = np.arange(M)
    Yk = Y[..., :M].copy()
    Yk[..., 0] = Y[..., 0].real
    Yr = np.conj(Y[..., M - k])                       # conj(Y[M - k]), k = 0 -> Y[M]
    Yr[..., 0] = Y[..., M].real
    Fe = (Yk + Yr).astype(np.complex64)
    Fo = ((Yk - Yr) * tw[k]).astype(np.complex64)     # W_N^k
    Z = (Fe + 1j * Fo).astype(np.complex64)           # [..., M]
    z = Z.reshape(Z.shape[:-1] + (R2, 32))            # [..., k2, k1]
    j = np.arange(R2)
    WR = tw[((np.outer(j, j) * (N // R2)) % N)]       # W_{R2}^{k2 n2}
    A = np.einsum("...jl,jy->...yl", z, WR).astype(np.complex64)   # [..., n2, k1]
    l = np.arange(32)
    A = (A * tw[(2 * np.outer(j, l)) % N]).astype(np.complex64)    # W_M^{k1 n2} = W_N^{2 k1 n2}
    B = _lane_dif32(A, tw, N)
    n1 = bitrev5(l)
    zz = np.zeros(Z.shape[:-1] + (M,), dtype=np.complex64)
    for n2 in range(R2):
        zz[..., n2 + R2 * n1] = B[..., n2, :]
    out = np.zeros(Y.shape[:-1] + (N,), dtype=np.float32)
    out[..., 0::2] = zz.real
    out[..., 1::2] = zz.imag
    return out


def image_from_uv_twin(uv: np.ndarray, plan: FftPlan, mask_scaled: np.ndarray,
                       out_dtype: Optional[type] = np.float16) -> np.ndarray:
    """Twin of K2 + K3a: ``uv [B, n_act]`` -> ``[B, N, N]`` image
    ``irfft2(G_h) * mask_scaled`` (same convention as
    :func:`image_from_uv_torch`). The kernels apply the 1/N^2 inside the
    single per-pixel constant ``fl32(mask_scaled / N^2)``; so does this."""
    N = plan.n_grid
    nh = plan.n_half
    B = uv.shape[0]
    rows = np.zeros((B, N, nh), dtype=np.complex64)
    if plan.fft_cols.size:
        sel = np.nonzero(plan.cell_col_slot >= 0)[0]
        cols = np.zeros((B, plan.fft_cols.size, N), dtype=np.complex64)
        cols[:, plan.cell_col_slot[sel], plan.cell_u[sel]] = uv[:, sel]
        colt = k2_twin(cols, N)                                  # [B, n_fc, N(y)]
        rows[:, :, plan.fft_cols] = np.swapaxes(colt, 1, 2)
    if plan.const_cols.size:
        sel = np.nonzero(plan.cell_const_slot >= 0)[0]
        rows[:, :, plan.cell_w[sel]] = uv[:, sel][:, None, :]
    img = k3a_twin(rows, N)                                      # [B, N, N] (y, x)
    pix = (np.asarray(mask_scaled, dtype=np.float64) / float(N * N)).astype(np.float32)
    out = (img * pix).astype(np.float32)
    return out.astype(out_dtype) if out_dtype is not None else out
