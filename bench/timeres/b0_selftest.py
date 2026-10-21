"""B0 - correctness of the prototype kernels at toy sizes.

On a node:  python b0_selftest.py --device cuda:0
Off-GPU:    TRITON_INTERPRET=1 python b0_selftest.py --device cpu
"""
from __future__ import annotations

import argparse

import torch

from kernels import (boxcar_chunkmax, chunkmax_reference, combine_cells,
                     combine_reference, pack_cint8)


def test_combine(dev):
    g = torch.Generator().manual_seed(1)
    S, T_out, C, F, spread, plane = 6, 40, 37, 3, 9, 2 * 16 * 9
    T_in = T_out + spread
    re = torch.randint(-128, 128, (S, T_in, C), generator=g, dtype=torch.int8)
    im = torch.randint(-128, 128, (S, T_in, C), generator=g, dtype=torch.int8)
    xp = pack_cint8(re, im)
    shifts = torch.randint(0, spread + 1, (F, S), generator=g, dtype=torch.int32)
    cellpos = torch.randperm(plane // 2, generator=g)[:C].to(torch.int32)
    out = torch.zeros(F * T_out, plane, dtype=torch.float16, device=dev)
    combine_cells(xp.to(dev), shifts.to(dev), cellpos.to(dev), out, T_out, BT=16, BC=16)
    ref = combine_reference(xp, shifts, cellpos, T_out, plane)
    err = (out.float().cpu() - ref).abs().max().item()
    assert err == 0.0, f"combine mismatch {err}"
    print("combine_cells: exact match")


def test_chunkmax(dev):
    g = torch.Generator().manual_seed(2)
    n_out, chunk, P = 32, 8, 50
    widths = torch.tensor([1, 2, 4, 8], dtype=torch.int32)
    maxw = int(widths.max())
    offsets = (maxw // 2 - widths // 2).to(torch.int32)
    sig = torch.tensor([1.0, 0.71, 0.5, 0.35], dtype=torch.float32)
    x = torch.randn(n_out + maxw, P, generator=g)
    cs = torch.cat([torch.zeros(1, P), x.cumsum(0)]).to(torch.float32)
    snr, tk = boxcar_chunkmax(cs.to(dev), widths.to(dev), offsets.to(dev), sig.to(dev),
                              n_out, chunk=chunk, block=16)
    ref = chunkmax_reference(cs, widths, offsets, sig, n_out, chunk)
    err = (snr.cpu() - ref).abs().max().item()
    assert err < 1e-4, f"chunkmax mismatch {err}"
    print(f"boxcar_chunkmax: max |err| {err:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    test_combine(a.device)
    test_chunkmax(a.device)
    print("b0 selftest PASSED")
