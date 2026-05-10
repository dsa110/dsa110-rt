"""Fused gather-by-destination dedispersion + grid kernel (Triton).

Replaces the gather + complex64 ``index_add_`` scatter pair in
``corr_fast_integration._dedisperse_one_window`` with a single kernel
that walks each grid cell's source list directly:

  * **Skip** the (T_chunk, C, B) cfp32 intermediate (~1.78 GB / chunk
    of allocator pressure that the current path materialises).
  * **Skip atomicAdds** — each (dm, t, grid_cell) output is computed
    by ONE program that reads its sources sequentially, sums in
    fp32 registers, and writes once. With the cell-source CSR
    layout we know which sources contribute to each cell, so no
    atomic contention.

Inputs:
  vis_BCT       : cfp32 (NBASE, NCHAN_eff, T_full) — the permuted vis
                  buffer. Caller is responsible for the one-time
                  permute. Note this is a DIFFERENT layout from the
                  Phase-5 (T, C, B) layout — we want T innermost so
                  the BLOCK_T threads inside a program coalesce.
  bin_shifts    : int32 (NCHAN_eff, n_dm) — per-(c, dm) time-bin shift
  csr_offs      : int32 (n_filled+1,) — CSR row pointers into csr_b/csr_c
  csr_b         : int32 (NSRC',) — antenna-baseline index per source
  csr_c         : int32 (NSRC',) — channel index per source

NOTE: NSRC' = NCHAN_eff * NBASE - n_unfilled is the number of sources
that actually map to a real grid cell (the ``n_filled`` sentinel
discards the rest).

Outputs:
  out           : cfp32 (n_dm, t_dedisp, n_filled)

The CSR is built once at construction time from the gridder's
cell_index_map, then re-used every block. Pre-compute cost is one
sort + cumsum on (NSRC',) data — under 1 ms on the GPU.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# CSR builder — runs once at context construction.
# ---------------------------------------------------------------------------


def build_cell_csr(
    cell_index_map: torch.Tensor,                 # (NSRC,) int64, layout = b*C + c
    *,
    n_filled: int,
    nchan_eff: int,
    nbase: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a CSR mapping ``grid_cell → list of source (b, c) pairs``.

    The legacy ``cell_index_map`` is (NSRC,) int64 with the convention
    ``cim[b*C + c] = grid_cell`` and ``grid_cell == n_filled`` for
    cells that don't map to a filled grid (e.g. K=1 sources off-grid).
    The CSR strips those, giving us a compact representation for the
    Triton kernel.

    Returns (csr_offs, csr_b, csr_c) as int32 tensors on the same
    device as ``cell_index_map``.
    """
    device = cell_index_map.device
    if cell_index_map.numel() != nchan_eff * nbase:
        raise ValueError(
            f"cell_index_map.numel()={cell_index_map.numel()} != "
            f"nchan_eff*nbase = {nchan_eff*nbase}"
        )
    cim = cell_index_map.view(-1).long()                   # (NSRC,) int64
    keep = cim < n_filled                                  # (NSRC,) bool
    src_idx = torch.nonzero(keep, as_tuple=False).flatten()        # (NSRC',) int64
    cell_for_src = cim[src_idx]                                    # (NSRC',) int64

    # Sort sources by destination cell so all sources of the same cell
    # land in a contiguous CSR row.
    sort_idx = torch.argsort(cell_for_src)
    src_sorted = src_idx[sort_idx]                                  # (NSRC',) int64
    cell_sorted = cell_for_src[sort_idx]

    # b, c for each source. Layout convention for cell_index_map is
    # cim[b*C + c] = cell, so b = src // C, c = src % C.
    b = (src_sorted // nchan_eff).to(torch.int32).contiguous()
    c = (src_sorted %  nchan_eff).to(torch.int32).contiguous()

    # CSR offsets: count sources per cell, then cumsum.
    counts = torch.bincount(cell_sorted, minlength=n_filled).to(torch.int32)
    csr_offs = torch.empty(n_filled + 1, dtype=torch.int32, device=device)
    csr_offs[0] = 0
    csr_offs[1:] = torch.cumsum(counts, 0)
    return csr_offs, b, c


# ---------------------------------------------------------------------------
# Triton kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_dedisp_kernel(
    vis_re_ptr, vis_im_ptr,                  # fp32 (B, C, T) — split real/imag
    shifts_ptr,                              # int32 (C, n_dm)
    csr_offs_ptr,                            # int32 (n_filled+1,)
    csr_b_ptr, csr_c_ptr,                    # int32 (NSRC',)
    out_re_ptr, out_im_ptr,                  # fp32 (n_dm, t_dedisp, n_filled)
    n_dm, t_dedisp, n_filled,
    NCHAN_EFF: tl.constexpr, NBASE: tl.constexpr,
    vis_stride_b, vis_stride_c, vis_stride_t,
    sh_stride_c, sh_stride_dm,
    out_stride_dm, out_stride_t, out_stride_g,
    BLOCK_T: tl.constexpr,
):
    """One program → one (dm, t-tile, grid_cell) output column.

    Grid: (n_dm, ceil(t_dedisp / BLOCK_T), n_filled).

    Per program:
      * Load ``[csr_offs[g], csr_offs[g+1])`` source indices.
      * For each source, load BLOCK_T contiguous time samples
        (coalesced because vis is (B, C, T) layout) at the
        DM-shifted offset and accumulate into BLOCK_T fp32 registers.
      * Single write of BLOCK_T output cells.
    """
    pid_dm = tl.program_id(0)
    pid_tt = tl.program_id(1)
    pid_g  = tl.program_id(2)

    t0 = pid_tt * BLOCK_T
    offs_t = t0 + tl.arange(0, BLOCK_T)                 # (BLOCK_T,)
    mask_t = offs_t < t_dedisp

    # Source range for this grid cell
    src0 = tl.load(csr_offs_ptr + pid_g)
    src1 = tl.load(csr_offs_ptr + pid_g + 1)

    acc_re = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for s in range(src0, src1):
        b = tl.load(csr_b_ptr + s)                      # int32
        c = tl.load(csr_c_ptr + s)                      # int32
        sh = tl.load(shifts_ptr + c * sh_stride_c + pid_dm * sh_stride_dm)
        # vis_re[b, c, sh + offs_t]
        ptrs_re = (vis_re_ptr
                   + b  * vis_stride_b
                   + c  * vis_stride_c
                   + (sh + offs_t) * vis_stride_t)
        ptrs_im = (vis_im_ptr
                   + b  * vis_stride_b
                   + c  * vis_stride_c
                   + (sh + offs_t) * vis_stride_t)
        vr = tl.load(ptrs_re, mask=mask_t, other=0.0)
        vi = tl.load(ptrs_im, mask=mask_t, other=0.0)
        acc_re += vr
        acc_im += vi

    out_offs = (pid_dm * out_stride_dm
                + offs_t * out_stride_t
                + pid_g  * out_stride_g)
    tl.store(out_re_ptr + out_offs, acc_re, mask=mask_t)
    tl.store(out_im_ptr + out_offs, acc_im, mask=mask_t)


def fused_dedisp_triton(
    vis_BCT_re: torch.Tensor,                # fp32 (NBASE, NCHAN_eff, T_full)
    vis_BCT_im: torch.Tensor,
    *,
    bin_shifts: torch.Tensor,                # int32 (NCHAN_eff, n_dm)
    csr_offs: torch.Tensor,                  # int32 (n_filled+1,)
    csr_b: torch.Tensor,                     # int32 (NSRC',)
    csr_c: torch.Tensor,                     # int32 (NSRC',)
    n_filled: int,
    t_dedisp: int,
    BLOCK_T: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused dedisp+grid kernel. Returns (out_re, out_im) fp32
    of shape ``(n_dm, t_dedisp, n_filled)``.
    """
    assert vis_BCT_re.is_contiguous() and vis_BCT_im.is_contiguous()
    assert vis_BCT_re.dtype == torch.float32 and vis_BCT_im.dtype == torch.float32
    nb, nc, t_full = vis_BCT_re.shape
    n_dm = bin_shifts.shape[1]
    out_re = torch.empty((n_dm, t_dedisp, n_filled),
                         dtype=torch.float32, device=vis_BCT_re.device)
    out_im = torch.empty_like(out_re)

    grid = (n_dm, triton.cdiv(t_dedisp, BLOCK_T), n_filled)
    _fused_dedisp_kernel[grid](
        vis_BCT_re, vis_BCT_im,
        bin_shifts, csr_offs, csr_b, csr_c,
        out_re, out_im,
        n_dm, t_dedisp, n_filled,
        nc, nb,
        vis_BCT_re.stride(0), vis_BCT_re.stride(1), vis_BCT_re.stride(2),
        bin_shifts.stride(0), bin_shifts.stride(1),
        out_re.stride(0), out_re.stride(1), out_re.stride(2),
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return out_re, out_im
