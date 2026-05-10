"""Fused on-wire-bytes → fp16 GEMM-layout Triton kernel (RT Phase 14).

Replaces all of ``unpack_int4_split``'s Stage 1 (fp32-reinterpret 2D
transpose, ~24 ms) + Stage 2 (byte permute) + Stage 3 (int4 ASR fluff +
scale + fp16 cast) — i.e. everything between the H2D copy and the
returned ``(real, imag)`` fp16 GEMM tensors — with a single Triton
kernel that reads from the *on-wire* layout
``(NPACKETS, NANTS, NCHAN, 2t, 2p)`` directly and writes the GEMM-layout
``(NCHAN, 2t, 2p, NPACKETS, NANTS)`` outputs in one pass.

Key insight: in the on-wire layout, the 4 bytes for a fixed
``(pkt, ant, ch)`` are exactly the (2t, 2p) cube and live at consecutive
offsets. Viewing the raw bytes as fp32 gives one 4-byte cell per
``(pkt, ant, ch)``; loading a tile of fp32s and bit-decomposing each
yields all 8 fp16 outputs (4 real + 4 imag) per byte read — *no Stage 1
transpose needed*.

Coalescing strategy:

* **Load** as ``(BLOCK_PKT, BLOCK_ANT, BLOCK_CH)`` fp32 with
  ``BLOCK_CH`` innermost. Triton lays threads out so the inner tile dim
  varies fastest within a warp; with ch-stride = 1 fp32 (= 4 bytes) per
  channel, 32 threads load 128 contiguous bytes = 1 cache line. ✓
* **Store** as ``(BLOCK_CH, BLOCK_PKT, BLOCK_ANT)`` fp16 (after a
  ``tl.permute((2, 0, 1))`` in registers / shared memory). Innermost
  tile dim is now BLOCK_ANT with stride 1 fp16 (= 2 bytes) per ant;
  32 threads write 64 contiguous bytes = ½ cache line. ✓

Memory traffic per block (production op-point: 288 MB on-wire bytes):

* Pre-Phase-14 (Stage 1 + Stage 2/3 fused):
  * Stage 1 fp32 transpose:        288 MB read + 288 MB write   (576 MB)
  * Stage 2/3 fused (Phase 12):    288 MB read + 1.17 GB write  (1.46 GB)
  * Total:                                                       2.04 GB
* Phase 14 (single fused kernel):
  * 288 MB read + 1.17 GB write                                  1.46 GB

The bandwidth saving is ~580 MB / block → ~0.97 ms at 600 GB/s peak.
The bigger win is eliminating the ~24 ms Stage 1 cuBLAS transpose
launch + its allocator pressure.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_int4_onwire_kernel(
    in_fp32_ptr,                                # fp32 view of on-wire bytes:
                                                # (NPACKETS, NANTS, NCHAN) where each
                                                # fp32 cell holds (2t, 2p) bytes.
    real_ptr, imag_ptr,                         # fp16 (NCHAN, 2t, 2p, NPACKETS, NANTS)
    NPACKETS: tl.constexpr,
    NANTS: tl.constexpr,
    NCHAN: tl.constexpr,
    NTIMES: tl.constexpr,                       # = 2
    NPOL: tl.constexpr,                         # = 2
    scale,                                      # runtime fp32
    in_stride_pkt_fp32,                         # = NANTS * NCHAN
    in_stride_ant_fp32,                         # = NCHAN
    in_stride_ch_fp32,                          # = 1
    out_stride_ch, out_stride_t, out_stride_p,
    out_stride_pkt, out_stride_ant,
    BLOCK_PKT: tl.constexpr,
    BLOCK_ANT: tl.constexpr,
    BLOCK_CH:  tl.constexpr,
):
    """One program → BLOCK_PKT × BLOCK_ANT × BLOCK_CH fp32 cells loaded.

    Grid: ``(NPACKETS // BLOCK_PKT, NANTS // BLOCK_ANT, NCHAN // BLOCK_CH)``.

    All blocks must evenly divide their respective dims at the
    production op-point (NPACKETS=2048, NANTS=96, NCHAN=384), avoiding
    masked stores — pick block sizes accordingly.
    """
    pid_pk = tl.program_id(0)
    pid_an = tl.program_id(1)
    pid_ch = tl.program_id(2)

    pk_idx = pid_pk * BLOCK_PKT + tl.arange(0, BLOCK_PKT)        # (BLOCK_PKT,)
    an_idx = pid_an * BLOCK_ANT + tl.arange(0, BLOCK_ANT)        # (BLOCK_ANT,)
    ch_idx = pid_ch * BLOCK_CH  + tl.arange(0, BLOCK_CH)         # (BLOCK_CH,)

    pk_mask = pk_idx < NPACKETS
    an_mask = an_idx < NANTS
    ch_mask = ch_idx < NCHAN

    in_addr = (
        pk_idx[:, None, None] * in_stride_pkt_fp32
        + an_idx[None, :, None] * in_stride_ant_fp32
        + ch_idx[None, None, :] * in_stride_ch_fp32
    )
    mask_in = (
        pk_mask[:, None, None]
        & an_mask[None, :, None]
        & ch_mask[None, None, :]
    )

    fp32_tile = tl.load(in_fp32_ptr + in_addr, mask=mask_in, other=0.0)
    i32_tile  = fp32_tile.to(tl.int32, bitcast=True)              # (BLOCK_PKT, BLOCK_ANT, BLOCK_CH)

    # Bytes are little-endian on x86/CUDA — byte 0 is LSB of the int32.
    # Verified against the post-Stage-1-transpose reference: byte 0 =
    # (t=0, p=0), byte 1 = (t=0, p=1), byte 2 = (t=1, p=0), byte 3 =
    # (t=1, p=1).
    b00 = (i32_tile         & 0xFF).to(tl.int32)
    b01 = ((i32_tile >>  8) & 0xFF).to(tl.int32)
    b10 = ((i32_tile >> 16) & 0xFF).to(tl.int32)
    b11 = ((i32_tile >> 24) & 0xFF).to(tl.int32)

    # int4 fluff (inlined; triton.jit can't see nested defs):
    #   real_nibble = ASR-sign-extend(b & 0xF)
    #   imag_nibble = ASR-sign-extend((b >> 4) & 0xF)
    # Mirrors slow_corr_kernel.unpack_int4_split's Stage 3.
    low00 = b00 & 0xF; high00 = (b00 >> 4) & 0xF
    low01 = b01 & 0xF; high01 = (b01 >> 4) & 0xF
    low10 = b10 & 0xF; high10 = (b10 >> 4) & 0xF
    low11 = b11 & 0xF; high11 = (b11 >> 4) & 0xF

    r00 = ((tl.where(low00  >= 8, low00  - 16, low00 ).to(tl.float32) * scale).to(tl.float16))
    r01 = ((tl.where(low01  >= 8, low01  - 16, low01 ).to(tl.float32) * scale).to(tl.float16))
    r10 = ((tl.where(low10  >= 8, low10  - 16, low10 ).to(tl.float32) * scale).to(tl.float16))
    r11 = ((tl.where(low11  >= 8, low11  - 16, low11 ).to(tl.float32) * scale).to(tl.float16))
    i00 = ((tl.where(high00 >= 8, high00 - 16, high00).to(tl.float32) * scale).to(tl.float16))
    i01 = ((tl.where(high01 >= 8, high01 - 16, high01).to(tl.float32) * scale).to(tl.float16))
    i10 = ((tl.where(high10 >= 8, high10 - 16, high10).to(tl.float32) * scale).to(tl.float16))
    i11 = ((tl.where(high11 >= 8, high11 - 16, high11).to(tl.float32) * scale).to(tl.float16))

    # In-register / shared-memory permute so output stores have ant
    # innermost (= fp16 stride 2 bytes for coalesced writes).
    r00_T = tl.permute(r00, (2, 0, 1))                            # (BLOCK_CH, BLOCK_PKT, BLOCK_ANT)
    r01_T = tl.permute(r01, (2, 0, 1))
    r10_T = tl.permute(r10, (2, 0, 1))
    r11_T = tl.permute(r11, (2, 0, 1))
    i00_T = tl.permute(i00, (2, 0, 1))
    i01_T = tl.permute(i01, (2, 0, 1))
    i10_T = tl.permute(i10, (2, 0, 1))
    i11_T = tl.permute(i11, (2, 0, 1))

    out_addr_base = (
        ch_idx[:, None, None] * out_stride_ch
        + pk_idx[None, :, None] * out_stride_pkt
        + an_idx[None, None, :] * out_stride_ant
    )
    mask_out = (
        ch_mask[:, None, None]
        & pk_mask[None, :, None]
        & an_mask[None, None, :]
    )

    tl.store(real_ptr + out_addr_base + 0 * out_stride_t + 0 * out_stride_p, r00_T, mask=mask_out)
    tl.store(real_ptr + out_addr_base + 0 * out_stride_t + 1 * out_stride_p, r01_T, mask=mask_out)
    tl.store(real_ptr + out_addr_base + 1 * out_stride_t + 0 * out_stride_p, r10_T, mask=mask_out)
    tl.store(real_ptr + out_addr_base + 1 * out_stride_t + 1 * out_stride_p, r11_T, mask=mask_out)
    tl.store(imag_ptr + out_addr_base + 0 * out_stride_t + 0 * out_stride_p, i00_T, mask=mask_out)
    tl.store(imag_ptr + out_addr_base + 0 * out_stride_t + 1 * out_stride_p, i01_T, mask=mask_out)
    tl.store(imag_ptr + out_addr_base + 1 * out_stride_t + 0 * out_stride_p, i10_T, mask=mask_out)
    tl.store(imag_ptr + out_addr_base + 1 * out_stride_t + 1 * out_stride_p, i11_T, mask=mask_out)


def fused_int4_unpack_onwire_triton(
    raw_bytes_gpu: torch.Tensor,                # uint8 (NPACKETS * NANTS * NCHAN * 2 * 2,)
                                                # OR (NPACKETS, NANTS, NCHAN, 2, 2)
                                                # — must be C-contiguous on-wire layout.
    *,
    NPACKETS: int,
    NANTS: int,
    NCHAN: int,
    NTIMES: int = 2,
    NPOL: int = 2,
    scale: float = 0.05,
    out_dtype: torch.dtype = torch.float16,
    BLOCK_PKT: int = 4,
    BLOCK_ANT: int = 32,
    BLOCK_CH:  int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the on-wire fused fluff kernel; returns (real, imag) fp16
    in GEMM layout ``(NCHAN, 2t, 2p, NPACKETS, NANTS)``."""
    assert raw_bytes_gpu.dtype == torch.uint8
    assert raw_bytes_gpu.is_contiguous()
    assert raw_bytes_gpu.device.type == "cuda"
    assert out_dtype == torch.float16, "Triton fast path only supports fp16 output"
    assert NTIMES == 2 and NPOL == 2, (
        "Hard-coded for the fada (2t, 2p) cube layout; if these ever "
        "change, the byte-extract logic in the kernel needs updating."
    )

    expected = NPACKETS * NANTS * NCHAN * NTIMES * NPOL
    assert raw_bytes_gpu.numel() == expected, (
        f"raw_bytes_gpu size {raw_bytes_gpu.numel()} != expected {expected}"
    )

    # All production tile sizes evenly divide the production grid:
    # NPACKETS=2048 / 4 = 512, NANTS=96 / 32 = 3, NCHAN=384 / 32 = 12.
    assert NPACKETS % BLOCK_PKT == 0
    assert NANTS    % BLOCK_ANT == 0
    assert NCHAN    % BLOCK_CH  == 0

    # View raw uint8 as fp32 (NPACKETS, NANTS, NCHAN); each fp32 cell
    # holds the 4 bytes of one (pkt, ant, ch)'s (2t, 2p) cube.
    flat = raw_bytes_gpu.view(-1) if raw_bytes_gpu.ndim != 1 else raw_bytes_gpu
    in_fp32 = flat.view(torch.float32).view(NPACKETS, NANTS, NCHAN)

    real = torch.empty(
        (NCHAN, NTIMES, NPOL, NPACKETS, NANTS),
        dtype=out_dtype, device=raw_bytes_gpu.device,
    )
    imag = torch.empty_like(real)

    in_strides  = in_fp32.stride()
    out_strides = real.stride()

    grid = (NPACKETS // BLOCK_PKT, NANTS // BLOCK_ANT, NCHAN // BLOCK_CH)
    _unpack_int4_onwire_kernel[grid](
        in_fp32, real, imag,
        NPACKETS, NANTS, NCHAN, NTIMES, NPOL,
        float(scale),
        in_strides[0], in_strides[1], in_strides[2],
        out_strides[0], out_strides[1], out_strides[2],
        out_strides[3], out_strides[4],
        BLOCK_PKT=BLOCK_PKT, BLOCK_ANT=BLOCK_ANT, BLOCK_CH=BLOCK_CH,
        num_warps=4,
    )
    return real, imag
