"""The two kernel-level claims behind the pipelined orchestrator's GPU
bit-identity, checked under ``TRITON_INTERPRET=1`` on the CPU:

1. today's fused fast-GEMM kernel, launched with the time-minor ring as
   its output (new store strides), stores exactly the bits it stores into
   its own ``(n_fv, NBASE, NCHAN_eff)`` tensor, and nothing outside the
   slab;
2. today's stage-1 dedisp kernel computing only ``n_fv`` rows returns
   exactly the first ``n_fv`` rows of the ``t_dedisp_joined`` call (each
   output row is an independent fixed-order sum).
"""
from __future__ import annotations

from ._interp import run_interpreted

_GEMM = r"""
import torch
from dsart.services.triton_corr_fused import fused_corr_post_no_cat_triton
from dsart.x2.corr.pipeline1x import fused_corr_no_cat_into
torch.manual_seed(0)
n_fv, nchan, csf, nvp, nants, K = 3, 4, 2, 2, 32, 16
nchan_eff = nchan // csf
nbase = nants * (nants + 1) // 2
B = n_fv * nchan * nvp
R = torch.randn(B, K, nants).half()
I = torch.randn(B, K, nants).half()
ref_re, ref_im = fused_corr_post_no_cat_triton(
    R, I, n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff, nvp=nvp, csf=csf, nbase=nbase)
pad = 4
ring_re = torch.full((nbase, nchan_eff, 2 * n_fv + 2 * pad), float("nan"))
ring_im = torch.full_like(ring_re, float("nan"))
fused_corr_no_cat_into(R, I, ring_re[:, :, pad:pad + n_fv], ring_im[:, :, pad:pad + n_fv],
                       n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff, nvp=nvp, csf=csf, nbase=nbase)
# the reference kernel leaves lower-triangle-only slots unwritten (torch.empty);
# compare on the baselines it does write (all of them for nants % 16 == 0)
same = (torch.equal(ring_re[:, :, pad:pad + n_fv], ref_re.permute(1, 2, 0))
        and torch.equal(ring_im[:, :, pad:pad + n_fv], ref_im.permute(1, 2, 0)))
outside = torch.isnan(torch.cat([ring_re[:, :, :pad], ring_re[:, :, pad + n_fv:],
                                 ring_im[:, :, :pad], ring_im[:, :, pad + n_fv:]], -1)).all().item()
nonzero = bool(ref_re.abs().sum() > 0)
print("RESULT", same, outside, nonzero)
"""

_DEDISP = r"""
import torch
from dsart.services.triton_dedisp import build_cell_csr, fused_dedisp_triton
torch.manual_seed(1)
B, C, n_fv, n_dm, n_filled = 10, 4, 16, 3, 7
T = 2 * n_fv
cim = torch.randint(0, n_filled + 1, (B * C,))          # n_filled = unfilled sentinel
offs, cb, cc = build_cell_csr(cim, n_filled=n_filled, nchan_eff=C, nbase=B)
sh = torch.randint(0, 9, (C, n_dm), dtype=torch.int32)
sh[0] = 0
re = torch.randn(B, C, T)
im = torch.randn(B, C, T)
t_joined = T - int(sh.max())
res = []
for bt in (128, 8):
    fr, fi = fused_dedisp_triton(re, im, bin_shifts=sh, csr_offs=offs, csr_b=cb, csr_c=cc,
                                 n_filled=n_filled, t_dedisp=t_joined, BLOCK_T=bt)
    pr, pi = fused_dedisp_triton(re, im, bin_shifts=sh, csr_offs=offs, csr_b=cb, csr_c=cc,
                                 n_filled=n_filled, t_dedisp=n_fv, BLOCK_T=bt)
    res.append(torch.equal(fr[:, :n_fv], pr) and torch.equal(fi[:, :n_fv], pi))
# and the kernel really is the stage-1 shift-and-grid
ref = torch.zeros(n_dm, n_fv, n_filled, dtype=torch.float64)
for g in range(n_filled):
    for s in range(int(offs[g]), int(offs[g + 1])):
        b, c = int(cb[s]), int(cc[s])
        for d in range(n_dm):
            ref[d, :, g] += re[b, c, int(sh[c, d]):int(sh[c, d]) + n_fv].double()
res.append(torch.allclose(pr.double(), ref, atol=1e-4))
print("RESULT", *res)
"""


def test_gemm_into_ring_strides_bit_identical() -> None:
    out = run_interpreted(_GEMM)
    assert "RESULT True True True" in out, out


def test_stage1_n_fv_rows_bit_identical() -> None:
    out = run_interpreted(_DEDISP)
    assert "RESULT True True True" in out, out
