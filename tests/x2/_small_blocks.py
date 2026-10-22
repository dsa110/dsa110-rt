"""Small fada blocks for CPU tests (x2 rule: keep node CPU tests short).

Production fada pages are 2048 packets (288 MiB); ``unpack_int4_split``
and ``apply_cal_split`` hard-check that size. The orchestration tests do
not need it: :func:`patch_small_blocks` swaps both functions -- in every
module that imported them -- for shape-agnostic versions with the same
arithmetic, so the legacy loop and the restructured orchestrators run the
SAME functions on ``n_packets``-packet pages and can be compared bit for
bit. Full-size behaviour is CP1 GPU work.
"""
from __future__ import annotations

import numpy as np
import torch

from dsart.common.constants import LEGACY_FLUFF_SCALE, NANTS, NCHAN_PER_CHGROUP


def small_page_bytes(n_packets: int) -> int:
    return n_packets * NANTS * NCHAN_PER_CHGROUP * 4


def make_small_unpack(n_packets: int):
    def unpack(raw, *, device="cpu", scale=LEGACY_FLUFF_SCALE, out_dtype=torch.float16):
        b = np.asarray(raw, np.uint8).reshape(n_packets, NANTS, NCHAN_PER_CHGROUP, 2, 2).view(np.int8)
        re = ((b << 4).astype(np.int8) >> 4).transpose(2, 3, 4, 0, 1)      # (C, 2t, 2p, pkt, A)
        im = (b >> 4).transpose(2, 3, 4, 0, 1)
        r = torch.from_numpy(np.ascontiguousarray(re)).to(out_dtype) * scale
        i = torch.from_numpy(np.ascontiguousarray(im)).to(out_dtype) * scale
        return r.to(device), i.to(device)
    return unpack


def cal_any_shape(real_v, imag_v, cal_real_fine, cal_imag_fine):
    """``slow_corr_kernel._apply_cal_split_eager`` without the size check."""
    return (real_v * cal_real_fine - imag_v * cal_imag_fine,
            real_v * cal_imag_fine + imag_v * cal_real_fine)


def patch_small_blocks(monkeypatch, n_packets: int) -> int:
    """Install the small-page unpack + cal everywhere; returns page bytes."""
    import dsart.services.corr_fast_integration as cfi
    import dsart.x2.corr.pipeline1x as p1
    import dsart.x2.corr.pipeline2x as p2
    up = make_small_unpack(n_packets)
    for mod in (cfi, p1):
        monkeypatch.setattr(mod, "unpack_int4_split", up)
    for mod in (cfi, p1, p2):
        monkeypatch.setattr(mod, "apply_cal_split", cal_any_shape)
    return small_page_bytes(n_packets)
