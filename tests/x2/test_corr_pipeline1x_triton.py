"""The two kernel-level claims behind the pipelined orchestrator's GPU
bit-identity, checked under ``TRITON_INTERPRET=1`` on the CPU:

1. today's fused fast-GEMM kernel, launched with the time-minor ring as
   its output (new store strides), stores exactly the bits it stores into
   its own ``(n_fv, NBASE, NCHAN_eff)`` tensor, and nothing outside the
   slab;
2. today's stage-1 dedisp kernel computing only ``n_fv`` rows returns
   exactly the first ``n_fv`` rows of the ``t_dedisp_joined`` call (each
   output row is an independent fixed-order sum).

Both also match an independent float64 torch reference, and a MUTANT
kernel (one source edit) must fail that reference check -- proof that the
interpreter really executed the kernel (``tests/x2/_interp.py``).
"""
from __future__ import annotations

from ._interp import run_interpreted

_GEMM = r"""
import torch
import dsart.services.triton_corr_fused as tcf
from dsart.services.triton_corr_fused import fused_corr_post_no_cat_triton
from dsart.x2.corr.pipeline1x import fused_corr_no_cat_into
torch.manual_seed(0)
n_fv, nchan, csf, nvp, nants, K = 3, 4, 2, 2, 32, 16
nchan_eff = nchan // csf
nbase = nants * (nants + 1) // 2
B = n_fv * nchan * nvp
R = torch.randn(B, K, nants).half()
I = torch.randn(B, K, nants).half()

# independent float64 reference: V(m <= n) = sum conj(E_m) E_n, summed over
# pols and the csf channels, stored at bls = n (n + 1) / 2 + m
Rd = R.double().reshape(n_fv, nchan, nvp, K, nants)
Id = I.double().reshape(n_fv, nchan, nvp, K, nants)
ein = lambda a, b: torch.einsum("fcpkm,fcpkn->fcmn", a, b)
Vr = (ein(Rd, Rd) + ein(Id, Id)).reshape(n_fv, nchan_eff, csf, nants, nants).sum(2)
Vi = (ein(Rd, Id) - ein(Id, Rd)).reshape(n_fv, nchan_eff, csf, nants, nants).sum(2)
m_idx, n_idx = torch.triu_indices(nants, nants)
bls = n_idx * (n_idx + 1) // 2 + m_idx
want_re = torch.zeros(n_fv, nbase, nchan_eff, dtype=torch.float64)
want_im = torch.zeros_like(want_re)
want_re[:, bls, :] = Vr[:, :, m_idx, n_idx].permute(0, 2, 1)
want_im[:, bls, :] = Vi[:, :, m_idx, n_idx].permute(0, 2, 1)

def close(re, im):
    return (torch.allclose(re.double(), want_re, rtol=1e-4, atol=1e-3)
            and torch.allclose(im.double(), want_im, rtol=1e-4, atol=1e-3))

ref_re, ref_im = fused_corr_post_no_cat_triton(
    R, I, n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff, nvp=nvp, csf=csf, nbase=nbase)
pad = 4
ring_re = torch.full((nbase, nchan_eff, 2 * n_fv + 2 * pad), float("nan"))
ring_im = torch.full_like(ring_re, float("nan"))
fused_corr_no_cat_into(R, I, ring_re[:, :, pad:pad + n_fv], ring_im[:, :, pad:pad + n_fv],
                       n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff, nvp=nvp, csf=csf, nbase=nbase)
same = (torch.equal(ring_re[:, :, pad:pad + n_fv], ref_re.permute(1, 2, 0))
        and torch.equal(ring_im[:, :, pad:pad + n_fv], ref_im.permute(1, 2, 0)))
outside = torch.isnan(torch.cat([ring_re[:, :, :pad], ring_re[:, :, pad + n_fv:],
                                 ring_im[:, :, :pad], ring_im[:, :, pad + n_fv:]], -1)).all().item()
correct = close(ref_re, ref_im)
# mutant: flip one term of the imaginary accumulation -> must miss the reference
tcf._fused_corr_post_no_cat_kernel = _mutant(
    "dsart.services.triton_corr_fused", "_fused_corr_post_no_cat_kernel",
    "acc_im -= tl.dot(I_M, R_N, out_dtype=tl.float32)",
    "acc_im += tl.dot(I_M, R_N, out_dtype=tl.float32)")
mut_re = torch.zeros(nbase, nchan_eff, n_fv)
mut_im = torch.zeros_like(mut_re)
fused_corr_no_cat_into(R, I, mut_re, mut_im, n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
                       nvp=nvp, csf=csf, nbase=nbase)
caught = not close(mut_re.permute(2, 0, 1), mut_im.permute(2, 0, 1))
print("RESULT", same, outside, correct, caught)
"""

_DEDISP = r"""
import torch
import dsart.services.triton_dedisp as tdd
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
ref = torch.zeros(n_dm, n_fv, n_filled, dtype=torch.float64)
for g in range(n_filled):
    for s in range(int(offs[g]), int(offs[g + 1])):
        b, c = int(cb[s]), int(cc[s])
        for d in range(n_dm):
            ref[d, :, g] += re[b, c, int(sh[c, d]):int(sh[c, d]) + n_fv].double()
res = []
for bt in (128, 8):
    fr, fi = fused_dedisp_triton(re, im, bin_shifts=sh, csr_offs=offs, csr_b=cb, csr_c=cc,
                                 n_filled=n_filled, t_dedisp=t_joined, BLOCK_T=bt)
    pr, pi = fused_dedisp_triton(re, im, bin_shifts=sh, csr_offs=offs, csr_b=cb, csr_c=cc,
                                 n_filled=n_filled, t_dedisp=n_fv, BLOCK_T=bt)
    res.append(torch.equal(fr[:, :n_fv], pr) and torch.equal(fi[:, :n_fv], pi))
res.append(torch.allclose(pr.double(), ref, atol=1e-4))
tdd._fused_dedisp_kernel = _mutant("dsart.services.triton_dedisp", "_fused_dedisp_kernel",
                                   "        acc_re += vr\n", "        acc_re += 2.0 * vr\n")
mr, mi = fused_dedisp_triton(re, im, bin_shifts=sh, csr_offs=offs, csr_b=cb, csr_c=cc,
                             n_filled=n_filled, t_dedisp=n_fv, BLOCK_T=8)
res.append(not torch.allclose(mr.double(), ref, atol=1e-4))
print("RESULT", *res)
"""


def test_gemm_into_ring_strides_bit_identical() -> None:
    out = run_interpreted(_GEMM)
    assert "RESULT True True True True" in out, out


def test_stage1_n_fv_rows_bit_identical() -> None:
    out = run_interpreted(_DEDISP)
    assert "RESULT True True True True" in out, out
