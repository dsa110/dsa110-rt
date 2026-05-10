"""Fused int4 fluff + permute Triton kernel for unpack_int4_split.

Replaces Stages 2 (`bytes_T_layout.permute(...).contiguous()`) and 3
(int4 ASR sign-extend + scale + cast to fp16, in 4 separate PyTorch
elementwise kernels) of ``unpack_int4_split`` with one Triton kernel
that goes ``post-transpose bytes → (real, imag) fp16 in GEMM layout``
in a single pass.

Mirrors bfCorr's fused ``corr_input_copy`` (`dsaX_bfCorr.cu` lines
281-284), which is the reference unpack pipeline this code targets.

Memory traffic (per block at production op-point, 288 MB byte input):

  Current (Stages 2+3):
    * Stage 2 permute:       288 MB read + 288 MB write       (576 MB)
    * Stage 3 ASR low:       288 MB read + 288 MB write       (576 MB)
    * Stage 3 ASR high:      288 MB read + 288 MB write       (576 MB)
    * Stage 3 cast real:     288 MB read + 588 MB write       (876 MB)
    * Stage 3 cast imag:     288 MB read + 588 MB write       (876 MB)
    --                       ----------------------           -------
                                                              3.48 GB

  Fused Triton kernel:
    * 288 MB read + 1.17 GB write                             1.46 GB

At 600 GB/s peak DRAM bandwidth this should drop ~38 ms of Stages 2+3
work to ~5-8 ms — a ~30 ms saving on the 64 ms unpack stage.

Note: Stage 1 (the fp32-reinterpret 2D transpose) still runs upstream
and is itself ~25 ms; fusing it into the kernel would require reading
from the on-wire stride-4 bytes layout (uncoalesced) and is a bigger
lift, deferred.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_int4_fused_kernel(
    in_ptr,                                  # uint8 (NCHAN, NPACKETS, NANTS, 2t, 2p)
    real_ptr, imag_ptr,                      # fp16 (NCHAN, 2t, 2p, NPACKETS, NANTS)
    NPACKETS: tl.constexpr,
    NANTS: tl.constexpr,
    NTIMES: tl.constexpr,                    # = 2
    NPOL: tl.constexpr,                      # = 2
    scale,                                   # runtime fp32
    in_stride_ch, in_stride_pkt, in_stride_ant, in_stride_t, in_stride_p,
    out_stride_ch, out_stride_t, out_stride_p, out_stride_pkt, out_stride_ant,
    BLOCK_PKT: tl.constexpr,
    BLOCK_ANT: tl.constexpr,
):
    """One program → BLOCK_PKT × BLOCK_ANT bytes for one (ch, t, p).

    Grid: (NCHAN, NTIMES * NPOL, ceil(NPACKETS / BLOCK_PKT)).

    Each thread reads one byte, does both int4 sign-extends, scales,
    casts to fp16, and writes to the GEMM layout. Coalesced reads
    (along ant dim, stride NTIMES*NPOL=4 bytes) and coalesced writes
    (along ant dim, stride 2 bytes).
    """
    pid_ch = tl.program_id(0)
    pid_tp = tl.program_id(1)
    pid_pk = tl.program_id(2)

    ntime = pid_tp // NPOL
    npol  = pid_tp %  NPOL

    pk_offs  = pid_pk * BLOCK_PKT + tl.arange(0, BLOCK_PKT)
    ant_offs = tl.arange(0, BLOCK_ANT)
    mask_pk  = pk_offs  < NPACKETS
    mask_ant = ant_offs < NANTS

    in_addr = (pid_ch * in_stride_ch
               + pk_offs[:, None]  * in_stride_pkt
               + ant_offs[None, :] * in_stride_ant
               + ntime             * in_stride_t
               + npol              * in_stride_p)
    bytes_t = tl.load(
        in_ptr + in_addr,
        mask=mask_pk[:, None] & mask_ant[None, :],
        other=0,
    )                                                          # uint8 (BLOCK_PKT, BLOCK_ANT)

    # ---- int4 sign-extend ----
    # Lower nibble (real):  ASR-equivalent on 4-bit signed = (b & 0xF) - 16*(b & 0x8 != 0)
    # Upper nibble (imag):  same on (b >> 4)
    low  = (bytes_t  & 0xF).to(tl.int32)
    high = ((bytes_t >> 4) & 0xF).to(tl.int32)
    real_nib = tl.where(low  >= 8, low  - 16, low ).to(tl.float32)
    imag_nib = tl.where(high >= 8, high - 16, high).to(tl.float32)

    real_val = (real_nib * scale).to(tl.float16)
    imag_val = (imag_nib * scale).to(tl.float16)

    out_addr = (pid_ch  * out_stride_ch
                + ntime * out_stride_t
                + npol  * out_stride_p
                + pk_offs[:, None]  * out_stride_pkt
                + ant_offs[None, :] * out_stride_ant)
    tl.store(real_ptr + out_addr, real_val, mask=mask_pk[:, None] & mask_ant[None, :])
    tl.store(imag_ptr + out_addr, imag_val, mask=mask_pk[:, None] & mask_ant[None, :])


def fused_int4_unpack_triton(
    bytes_T_layout: torch.Tensor,            # uint8 (NCHAN, NPACKETS, NANTS, 2t, 2p)
    *,
    scale: float,
    out_dtype: torch.dtype = torch.float16,
    BLOCK_PKT: int = 16,
    BLOCK_ANT: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused int4 fluff kernel; returns (real, imag) fp16 in
    GEMM layout ``(NCHAN, 2t, 2p, NPACKETS, NANTS)``."""
    assert bytes_T_layout.dtype == torch.uint8
    assert bytes_T_layout.is_contiguous()
    assert bytes_T_layout.device.type == "cuda"
    assert out_dtype == torch.float16, "Triton fast path only supports fp16 output"
    NCHAN, NPACKETS, NANTS, NTIMES, NPOL = bytes_T_layout.shape

    real = torch.empty(
        (NCHAN, NTIMES, NPOL, NPACKETS, NANTS),
        dtype=out_dtype, device=bytes_T_layout.device,
    )
    imag = torch.empty_like(real)

    in_strides  = bytes_T_layout.stride()
    out_strides = real.stride()

    grid = (NCHAN, NTIMES * NPOL, triton.cdiv(NPACKETS, BLOCK_PKT))
    _unpack_int4_fused_kernel[grid](
        bytes_T_layout, real, imag,
        NPACKETS, NANTS, NTIMES, NPOL,
        float(scale),
        in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[4],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
        BLOCK_PKT=BLOCK_PKT, BLOCK_ANT=BLOCK_ANT,
        num_warps=4,
    )
    return real, imag
