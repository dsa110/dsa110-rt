"""Microbench + correctness check for the Triton fused dedisp kernel."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import NATIVE_SAMPLE_US, NBASE, NCHAN_PER_CHGROUP  # noqa: E402
from dsart.services.triton_dedisp import (  # noqa: E402
    build_cell_csr,
    fused_dedisp_triton,
)


def reference_path(
    vis_T: torch.Tensor,                  # cfp32 (T, C, B)  (post-permute layout)
    bin_shifts: torch.Tensor,             # int64 (C, n_dm)
    cim_cb: torch.Tensor,                 # int64 (C*B,) — gridder cell map keyed (c, b)
    *, n_filled: int, t_dedisp: int, dm_chunk: int,
):
    """Mirror of corr_fast_integration._dedisperse_one_window post Phase 9."""
    n_fv, nch, nb = vis_T.shape
    n_dm = bin_shifts.shape[1]
    device = vis_T.device
    out = torch.empty((n_dm, t_dedisp, n_filled),
                      dtype=torch.complex64, device=device)
    t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)
    for c0 in range(0, n_dm, dm_chunk):
        c1 = min(c0 + dm_chunk, n_dm)
        chunk = c1 - c0
        t_chunk = chunk * t_dedisp
        bs_chunk = bin_shifts[:, c0:c1]
        t_idx_2d = (
            bs_chunk.t()[:, None, :] + t_arange[None, :, None]
        ).reshape(t_chunk, nch)
        t_idx_3d = t_idx_2d[:, :, None].expand(t_chunk, nch, nb)
        gathered = torch.gather(vis_T, 0, t_idx_3d)
        src = gathered.reshape(t_chunk, nch * nb)
        out_c = torch.zeros((t_chunk, n_filled + 1),
                            dtype=torch.complex64, device=device)
        out_c.index_add_(1, cim_cb, src)
        out[c0:c1] = out_c[:, :n_filled].reshape(chunk, t_dedisp, n_filled)
    return out


def main():
    device = torch.device("cuda")
    n_fv = 512
    chan_sum_factor = 8
    nch = NCHAN_PER_CHGROUP // chan_sum_factor
    nb = NBASE
    n_dm = 24
    max_shift = 100
    t_dedisp = n_fv - max_shift
    dm_chunk = 1

    # --- build a realistic n_filled ≈ 460, cim layout same as production ---
    rng = np.random.default_rng(0xBADD15)
    n_filled = 460
    cim_bc_np = rng.integers(0, n_filled + 1, size=(nb * nch,), dtype=np.int64)
    # Ensure the sentinel value (n_filled) appears (off-grid sources)
    n_off = max(1, int(0.02 * cim_bc_np.size))
    cim_bc_np[:n_off] = n_filled
    rng.shuffle(cim_bc_np)
    cim_bc = torch.from_numpy(cim_bc_np).to(device)
    # Reshape -> (B, C) layout
    cim_bc_2d = cim_bc.view(nb, nch)
    cim_cb = cim_bc_2d.t().contiguous().view(-1)               # (C*B,) int64

    # --- vis ---
    g = torch.Generator(device=device).manual_seed(0xC0DE)
    vis_TBC = (torch.rand(n_fv, nb, nch, dtype=torch.float32, generator=g, device=device) +
               1j * torch.rand(n_fv, nb, nch, dtype=torch.float32, generator=g, device=device)).to(torch.complex64)
    vis_T = vis_TBC.permute(0, 2, 1).contiguous()              # (T, C, B) cfp32 — phase-5 layout

    # --- bin shifts ---
    bs = rng.integers(0, max_shift + 1, size=(nch, n_dm), dtype=np.int64)
    bin_shifts = torch.from_numpy(bs).to(device)

    # ---------------- reference ----------------
    print("running reference (Phase 9 path)...")
    out_ref = reference_path(vis_T, bin_shifts, cim_cb,
                             n_filled=n_filled, t_dedisp=t_dedisp,
                             dm_chunk=dm_chunk)
    print(f"  ref shape: {tuple(out_ref.shape)}")

    # ---------------- triton ----------------
    print("building CSR...")
    csr_offs, csr_b, csr_c = build_cell_csr(
        cim_bc, n_filled=n_filled, nchan_eff=nch, nbase=nb,
    )
    print(f"  CSR: {csr_offs.shape[0]-1} cells, {csr_b.numel()} sources kept")

    # vis layout for triton: (B, C, T) split into real/imag fp32
    vis_BCT = vis_TBC.permute(1, 2, 0).contiguous()            # (B, C, T) cfp32
    vis_BCT_re = vis_BCT.real.contiguous()
    vis_BCT_im = vis_BCT.imag.contiguous()
    bin_shifts_i32 = bin_shifts.to(torch.int32).contiguous()

    out_re_t, out_im_t = fused_dedisp_triton(
        vis_BCT_re, vis_BCT_im,
        bin_shifts=bin_shifts_i32,
        csr_offs=csr_offs, csr_b=csr_b, csr_c=csr_c,
        n_filled=n_filled, t_dedisp=t_dedisp,
    )
    out_tri = torch.complex(out_re_t, out_im_t)

    # ---- correctness ----
    diff = (out_ref - out_tri).abs().max().item()
    rel = diff / out_ref.abs().max().item()
    print(f"max |Δ| vs ref: {diff:.3e}   (rel = {rel:.3e})")

    # ---------------- bench ----------------
    n_warm, n_iter = 5, 30

    def _ev(): return torch.cuda.Event(enable_timing=True)

    print("\nbench:")

    # reference
    for _ in range(n_warm):
        reference_path(vis_T, bin_shifts, cim_cb,
                       n_filled=n_filled, t_dedisp=t_dedisp, dm_chunk=dm_chunk)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        e0, e1 = _ev(), _ev()
        e0.record()
        reference_path(vis_T, bin_shifts, cim_cb,
                       n_filled=n_filled, t_dedisp=t_dedisp, dm_chunk=dm_chunk)
        e1.record(); e1.synchronize()
        times.append(e0.elapsed_time(e1))
    print(f"  REFERENCE  (Phase 9 gather + complex scatter):  p50={np.median(times):.2f} ms")

    # triton, BLOCK_T sweep
    for blk in [16, 32, 64, 128]:
        for _ in range(n_warm):
            fused_dedisp_triton(vis_BCT_re, vis_BCT_im,
                                bin_shifts=bin_shifts_i32,
                                csr_offs=csr_offs, csr_b=csr_b, csr_c=csr_c,
                                n_filled=n_filled, t_dedisp=t_dedisp,
                                BLOCK_T=blk)
        torch.cuda.synchronize()
        times_t = []
        for _ in range(n_iter):
            e0, e1 = _ev(), _ev()
            e0.record()
            fused_dedisp_triton(vis_BCT_re, vis_BCT_im,
                                bin_shifts=bin_shifts_i32,
                                csr_offs=csr_offs, csr_b=csr_b, csr_c=csr_c,
                                n_filled=n_filled, t_dedisp=t_dedisp,
                                BLOCK_T=blk)
            e1.record(); e1.synchronize()
            times_t.append(e0.elapsed_time(e1))
        p50 = float(np.median(times_t))
        print(f"  TRITON BLOCK_T={blk:3d}: p50={p50:.2f} ms")


if __name__ == "__main__":
    main()
