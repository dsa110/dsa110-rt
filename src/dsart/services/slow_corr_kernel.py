"""Slow correlator kernel (M2; plan §8 lines 2161-2177).

Pure-tensor pipeline that fluffs `fada` int4 voltages and computes
upper-triangle visibilities for `bada`. Device-agnostic (CUDA / CPU)
so tests in `test_slow_corr_synth.py` exercise this module directly
without PSRDADA.

Pipeline (per fada block):

    raw bytes [301,989,888]              # one fada page
        │
        │  unpack_int4_split (fluff to two real fp16 tensors)
        │
        ▼
    R, I  fp16  [2048, 96, 384, 2t, 2p]  each
        │
        │  permute → (ch, pol, ant, t_total)
        │
        ▼
    R, I  fp16  [384, 2, 96, 4096]
        │
        │  K-CHUNKED REAL-IMAG SPLIT GEMM (mirrors legacy halfFac=4):
        │    V_real += (R@R.T + I@I.T) per K-chunk → fp32 accum
        │    V_imag += (R@I.T - I@R.T) per K-chunk → fp32 accum
        │  (tensor cores engage on fp16 inputs; output cast to fp32 between chunks)
        │
        ▼
    V_full complex64 [384, 2, 96, 96]
        │
        │  upper-triangle scatter to bls_idx = a*(a+1)/2 + b, b ≤ a
        │
        ▼
    vis complex64 [4656, 384, 2]
        │
        │  pack_bada_block (bytes view)
        │
        ▼
    raw bytes [28,606,464]               # one bada page

Numeric strategy (D3 / D4 revised 2026-05-04 — tensor cores are essential):

  * fp16 inputs to torch.matmul → engages HMMA tensor cores on Turing
    (2080 Ti: 130 TFLOPs vs 13 TFLOPs at fp32). TF32 input fast-path
    only exists on Ampere+ which we don't have.
  * Real-imag split: V_ij = sum_t conj(E_i(t)) * E_j(t)
                          = sum_t (R_i - jI_i)(R_j + jI_j)
                          = (R_i R_j + I_i I_j) + j(R_i I_j - I_i R_j)
    → 4 real GEMMs per K-chunk (vs 1 complex-fp32 GEMM that bypasses TCs).
  * K-chunking (legacy halfFac=4): split K=4096 into 4 chunks of K=1024
    each. Per-chunk output cast .to(fp32) before accumulation. Keeps
    intermediate fp16 outputs well within fp16's ±65504 range
    (per-chunk peak ≈ 1024 × 0.4² ≈ 164) AND gives fp32 final precision
    via the cross-chunk sum.

NO calibration is applied (D2 in M2_PLAN_FIXES.md): slow visibilities
are uncalibrated by design — the user derives cal solutions from them
downstream. The legacy ×0.05 fluff scale (LEGACY_FLUFF_SCALE) IS
applied so output amplitudes carry the same physical units as
`dsaX_bfCorr` corr-mode (which preserves any user-side cal code that
calibrates against the historical scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from dsart.common.constants import (
    BADA_NPOL,
    BLOCK_SAMPLES_SPECNUM,
    LEGACY_FLUFF_SCALE,
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    NPOL,
)


# --- module-level constants (echoed for explicit binding in this file) ----

NPACKETS_PER_BLOCK: Final[int] = BLOCK_SAMPLES_SPECNUM     # 2048
NTIMES_PER_PACKET: Final[int] = 2
N_TIME_SAMPLES: Final[int] = NPACKETS_PER_BLOCK * NTIMES_PER_PACKET  # 4096

#: Default K-chunking for the GEMM. Legacy `dsaX_bfCorr.cu` (lines 41-42)
#: uses `halfFac=4` "to prevent overflow", but legacy fluffs ×0.05 BEFORE
#: the GEMM (matching us). With the 0.05 pre-fluff, |E|² ≤ 0.16 and the
#: K=4096 sum ≤ 655 — well within fp16 max (65504) by ~100×. Setting the
#: default to 1 collapses 16 matmuls + 16 cast/add launches to 4 matmuls
#: (batched over the leading (ch, pol) dims by torch.matmul). Override
#: with `half_fac=4` if production-RFI conditions ever push fp16 close
#: to overflow.
HALF_FAC_DEFAULT: Final[int] = 1

# fada raw layout (C-order strides): byte index = pkt * 96*384*2*2 + ant * 384*2*2 + ...
_FADA_VOLT_SHAPE: Final[tuple[int, ...]] = (
    NPACKETS_PER_BLOCK,  # 2048
    NANTS,               # 96
    NCHAN_PER_CHGROUP,   # 384
    NTIMES_PER_PACKET,   # 2
    NPOL,                # 2
)

# bada output: (NBASE, NCHAN, BADA_NPOL) complex64
_BADA_VIS_SHAPE: Final[tuple[int, ...]] = (NBASE, NCHAN_PER_CHGROUP, BADA_NPOL)


# --- baseline-index precomputation (D4: legacy xGPU upper-triangle) -------


def upper_tri_indices(nants: int = NANTS) -> tuple[np.ndarray, np.ndarray]:
    """Indices `(a_idx, b_idx)` over `nbase = nants*(nants+1)//2` with `b ≤ a`.

    Order matches `dsaX_bfCorr.cu` lines 467-481 (`for i in range(NANTS):
    for j in range(i+1): h_idxs[ii] = i*NANTS + j; ii++`); equivalent to
    `bls_idx = a*(a+1)//2 + b`. Auto-correlations included on the diagonal
    (a == b). Output dtype int64 for use as torch indices.
    """
    nbase = nants * (nants + 1) // 2
    a_idx = np.empty(nbase, dtype=np.int64)
    b_idx = np.empty(nbase, dtype=np.int64)
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            a_idx[k] = a
            b_idx[k] = b
            k += 1
    return a_idx, b_idx


# --- fluff: int4 complex bytes → two real fp16 tensors --------------------


def _build_fluff_lut(
    *, device: torch.device | str, scale: float, out_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-byte real / imag lookup tables for `unpack_int4_split`.

    Each LUT has 256 entries indexed by the raw byte value; LUT_real[b]
    = sign-extended low nibble of b, scaled. LUT_imag[b] = sign-extended
    high nibble of b, scaled. After this, fluff is a single gather per
    output (one read of the raw bytes, one write of the fluffed value).
    """
    arr = np.arange(256, dtype=np.int16)
    lo = (arr & 0x0F).astype(np.int16)
    hi = ((arr >> 4) & 0x0F).astype(np.int16)
    lo[lo >= 8] -= 16
    hi[hi >= 8] -= 16
    np_dtype = {torch.float16: np.float16, torch.float32: np.float32}.get(out_dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported out_dtype {out_dtype}")
    lut_real = torch.from_numpy((lo.astype(np_dtype)) * np.array(scale, dtype=np_dtype))
    lut_imag = torch.from_numpy((hi.astype(np_dtype)) * np.array(scale, dtype=np_dtype))
    return lut_real.to(device), lut_imag.to(device)


# Module-level LUT cache keyed by (device_str, scale, dtype) so a long-running
# service builds the LUTs exactly once per (device, scale, dtype) combo. The
# tables are 256 entries × 2 bytes each = 512 B per LUT; trivially small.
_FLUFF_LUT_CACHE: dict[tuple[str, float, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def unpack_int4_split(
    raw: np.ndarray | bytes | memoryview,
    *,
    device: torch.device | str = "cpu",
    scale: float = LEGACY_FLUFF_SCALE,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fluff one fada page into separate real / imag fp16 tensors.

    Each input byte holds two signed 4-bit nibbles (low = real, high = imag),
    each in [-8, 7] after sign-extension; both are multiplied by `scale`
    (`LEGACY_FLUFF_SCALE = 0.05` matches `dsaX_bfCorr.cu` lines 273-286).

    Returns the real and imag parts as SEPARATE tensors (rather than a single
    `torch.complex` tensor) so the downstream GEMM can use fp16 tensor cores
    via the real-imag split form (D3/D4 in M2_PLAN_FIXES.md).

    Implementation: per-byte LUT gather. Each output element is `LUT[b]`
    where `b` is one of 256 possible byte values. Gather is a single
    fused kernel pass: ~300 MB read + 600 MB write at >500 GB/s ≈ 2 ms
    on a 2080 Ti, vs ~100 ms for the dtype-promotion + `where` chain.

    Parameters
    ----------
    raw : np.ndarray | bytes | memoryview
        Raw fada bytes; must be exactly `prod(_FADA_VOLT_SHAPE)` long
        (= 301,989,888 = `FADA_BYTES_PER_BLOCK`).
    device : torch.device | str
        Where to materialise the output tensors.
    scale : float
        Per-component scale factor (default = legacy 0.05).
    out_dtype : torch.dtype
        Output element dtype. Default `torch.float16` enables tensor cores
        in the downstream GEMM. Pass `torch.float32` for CPU / debugging
        (no tensor cores anyway).

    Returns
    -------
    (real, imag) : tuple[torch.Tensor, torch.Tensor]
        Both tensors of shape `_FADA_VOLT_SHAPE = (2048, 96, 384, 2, 2)`
        and dtype `out_dtype`.
    """
    if isinstance(raw, (bytes, memoryview)):
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
    elif isinstance(raw, np.ndarray):
        raw_arr = raw if raw.dtype == np.uint8 else raw.view(np.uint8)
    else:
        raise TypeError(f"unsupported raw type {type(raw)!r}")

    expected_n = int(np.prod(_FADA_VOLT_SHAPE))
    if raw_arr.size != expected_n:
        raise ValueError(
            f"fada raw size {raw_arr.size} != expected {expected_n} "
            f"(= prod{_FADA_VOLT_SHAPE})"
        )

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_key = (str(dev), float(scale), out_dtype)
    luts = _FLUFF_LUT_CACHE.get(cache_key)
    if luts is None:
        luts = _build_fluff_lut(device=dev, scale=scale, out_dtype=out_dtype)
        _FLUFF_LUT_CACHE[cache_key] = luts
    lut_real, lut_imag = luts

    # uint8 → int64 indices (PyTorch only allows long indices for gather).
    # Allocates a temp 4× the raw size. To avoid that, use take which can
    # accept any int-dtype index in newer pytorch, but long-indexed gather
    # is more portable.
    raw_t = torch.as_tensor(raw_arr, device=dev)               # uint8 (288 MB)
    idx = raw_t.to(torch.long)                                 # int64 (2.3 GB temp)
    real = lut_real[idx].reshape(_FADA_VOLT_SHAPE)             # fp16 (288 MB)
    imag = lut_imag[idx].reshape(_FADA_VOLT_SHAPE)             # fp16 (288 MB)
    del idx, raw_t
    return real, imag


# --- per-block correlator (4-real-GEMM real-imag split + halfFac=4) -------


@dataclass
class SlowCorrKernel:
    """Stateful per-device correlator kernel.

    Wraps the per-(device, dtype) precomputed indices so multiple blocks
    can be processed without re-allocating. Construct once at service
    startup; call `compute_split(R, I)` per block on the (real, imag)
    tensors returned by `unpack_int4_split`.
    """

    device: torch.device
    nants: int = NANTS
    nchan: int = NCHAN_PER_CHGROUP
    nvolt_pol: int = NPOL
    nbada_pol: int = BADA_NPOL
    n_time_samples: int = N_TIME_SAMPLES
    half_fac: int = HALF_FAC_DEFAULT
    accum_dtype: torch.dtype = torch.float32

    _a_idx: torch.Tensor = field(init=False)
    _b_idx: torch.Tensor = field(init=False)
    _nbase: int = field(init=False)

    def __post_init__(self) -> None:
        dev = torch.device(self.device)
        # Normalise "cuda" → "cuda:N" so equality tests against the user's
        # (..).to(device) tensor (which always carries an explicit index)
        # don't spuriously fail. CPU devices are always "cpu" without index.
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.device = dev
        if self.nbada_pol > self.nvolt_pol:
            raise ValueError(
                f"nbada_pol ({self.nbada_pol}) > nvolt_pol ({self.nvolt_pol})"
            )
        if self.n_time_samples % self.half_fac != 0:
            raise ValueError(
                f"n_time_samples ({self.n_time_samples}) not divisible by "
                f"half_fac ({self.half_fac})"
            )
        a_idx, b_idx = upper_tri_indices(self.nants)
        self._a_idx = torch.from_numpy(a_idx).to(self.device)
        self._b_idx = torch.from_numpy(b_idx).to(self.device)
        self._nbase = self._a_idx.numel()

    def compute_split(
        self,
        real_v: torch.Tensor,
        imag_v: torch.Tensor,
    ) -> torch.Tensor:
        """Correlate one block from split-real-imag fp16 voltages.

        Parameters
        ----------
        real_v, imag_v : torch.Tensor
            Real and imag parts of the voltage tensor, each of shape
            `(2048, 96, 384, 2t, 2p)` and dtype fp16/fp32 (must match each
            other and be on `self.device`). Output of `unpack_int4_split`.

        Returns
        -------
        torch.Tensor
            complex64 visibilities of shape `(NBASE=4656, NCHAN=384,
            BADA_NPOL=2)`. Auto-correlations on the diagonal
            (bls_idx 0, 2, 5, 9, ... = `a*(a+1)/2 + a`).
        """
        for name, t in (("real_v", real_v), ("imag_v", imag_v)):
            if t.device != self.device:
                raise ValueError(f"{name} on {t.device}, kernel on {self.device}")
            if t.shape != _FADA_VOLT_SHAPE:
                raise ValueError(
                    f"{name} shape {tuple(t.shape)}, expected {_FADA_VOLT_SHAPE}"
                )
        if real_v.dtype != imag_v.dtype:
            raise ValueError(
                f"real_v dtype {real_v.dtype} != imag_v dtype {imag_v.dtype}"
            )

        # Reshape (pkt, ant, ch, t_sub, pol) → (ch, pol, ant, t_total).
        # `permute(...)` returns a view; `.contiguous()` materialises for the
        # GEMM (cuBLAS does not handle non-contiguous fp16 tensors well).
        # `.reshape` after `.contiguous()` is a free view.
        R = real_v.permute(2, 4, 1, 0, 3).contiguous().reshape(
            self.nchan, self.nvolt_pol, self.nants, self.n_time_samples,
        )
        I = imag_v.permute(2, 4, 1, 0, 3).contiguous().reshape(
            self.nchan, self.nvolt_pol, self.nants, self.n_time_samples,
        )
        # Free the original (pkt-major) buffers immediately. The permuted
        # `.contiguous()` copies above are independent allocations.
        del real_v, imag_v

        if self.half_fac == 1:
            # Fast path: single batched fp16 matmul per real/imag GEMM.
            # torch.matmul automatically batches over the (ch, pol) leading
            # dims → a single cuBLAS HgemmStridedBatched call per matmul.
            # Output is fp16; cast to fp32 immediately for the final sum.
            R_T = R.transpose(-1, -2)
            I_T = I.transpose(-1, -2)
            V_real = torch.matmul(R, R_T)                       # fp16 (ch, pol, 96, 96)
            V_real = V_real.add_(torch.matmul(I, I_T)).to(self.accum_dtype)
            V_imag = torch.matmul(R, I_T)
            V_imag = V_imag.sub_(torch.matmul(I, R_T)).to(self.accum_dtype)
        else:
            # Legacy halfFac path: split K into `half_fac` sub-batches with
            # fp32 accumulation across chunks (mirrors `dsaX_bfCorr.cu` lines
            # 41-42). Useful when production data ranges push fp16 outputs
            # close to overflow.
            chunk_K = self.n_time_samples // self.half_fac
            V_real = torch.zeros(
                (self.nchan, self.nvolt_pol, self.nants, self.nants),
                dtype=self.accum_dtype, device=self.device,
            )
            V_imag = torch.zeros_like(V_real)
            for c in range(self.half_fac):
                sl = slice(c * chunk_K, (c + 1) * chunk_K)
                R_c = R[..., sl]
                I_c = I[..., sl]
                R_cT = R_c.transpose(-1, -2)
                I_cT = I_c.transpose(-1, -2)
                # V_real[i,j] = sum_t  R_i(t) R_j(t) + I_i(t) I_j(t)
                # V_imag[i,j] = sum_t  R_i(t) I_j(t) - I_i(t) R_j(t)
                # (D5: V_ij = E_i^* · E_j  ⇒  imag = R_i I_j - I_i R_j.)
                V_real.add_(torch.matmul(R_c, R_cT).to(self.accum_dtype))
                V_real.add_(torch.matmul(I_c, I_cT).to(self.accum_dtype))
                V_imag.add_(torch.matmul(R_c, I_cT).to(self.accum_dtype))
                V_imag.sub_(torch.matmul(I_c, R_cT).to(self.accum_dtype))

        # Restrict to the bada-output pol count.
        V_real_b = V_real[:, : self.nbada_pol, :, :]                # (ch, npol, 96, 96)
        V_imag_b = V_imag[:, : self.nbada_pol, :, :]

        # Upper-triangle gather (D4): vis[bls, ch, pol] = V[ch, pol, a, b]
        # with bls_idx = a*(a+1)/2 + b, b ≤ a.
        vis_real = V_real_b[..., self._a_idx, self._b_idx]          # (ch, npol, NBASE)
        vis_imag = V_imag_b[..., self._a_idx, self._b_idx]
        vis_real = vis_real.permute(2, 0, 1).contiguous()           # (NBASE, ch, npol)
        vis_imag = vis_imag.permute(2, 0, 1).contiguous()
        vis = torch.complex(vis_real, vis_imag)                     # complex64
        return vis                                                  # (4656, 384, 2)


# --- pack: complex64 tensor → bada page bytes -----------------------------


def pack_bada_block(vis: torch.Tensor) -> np.ndarray:
    """Convert visibility tensor to the byte layout meridian_fringestop expects.

    Per Subagent B's reading of `dsamfs/utils.py::read_buffer` (lines
    141-155): bada bytes are interpreted as `float32` pairs
    (`view(np.float32).reshape(-1, 2).view(np.complex64).reshape(-1, nbls,
    nchan, npol)`). On-wire layout is row-major `(nbls, nchan, npol)`
    complex64, which is what we already have.
    """
    if vis.shape != _BADA_VIS_SHAPE:
        raise ValueError(
            f"vis shape {tuple(vis.shape)}, expected {_BADA_VIS_SHAPE}"
        )
    if vis.dtype != torch.complex64:
        raise ValueError(f"vis dtype {vis.dtype}, expected complex64")

    arr = vis.detach().cpu().contiguous().numpy()                   # complex64
    return arr.view(np.uint8).reshape(-1)                           # bytes view
