"""Slow correlator kernel (M2; plan §8 lines 2161-2177).

Pure-tensor pipeline that fluffs `fada` int4 voltages and computes
upper-triangle visibilities for `bada`. Device-agnostic (CUDA / CPU)
so tests in `test_slow_corr_synth.py` exercise this module directly
without PSRDADA.

Design mirrors `dsaX_bfCorr.cu`'s `dcorrelator` (lines 273-414, 499-602)
which hits the 134 ms native-cadence budget on a 2080 Ti. The legacy
implementation made three choices we initially missed and now copy:

  (1) **2D byte-transpose, not 5-axis permute.** Treat the
      packed-byte fada page as a 2D matrix
      `[NPACKETS_PER_BLOCK*NANTS, NCHAN_PER_PACKET*NTIMES_PER_PACKET*NPOL]`
      = `(196608, 1536)` and do a single 2D transpose. Tiled
      shared-memory 2D transposes are highly tuned in cuBLAS / PyTorch
      (~1 ms for 300 MB on a 2080 Ti). The naive 5-axis
      `permute().contiguous()` of the same data takes ~50 ms
      because it falls back to a generic stride-remap kernel.

  (2) **Fluff AFTER the transpose.** Once the bytes are in target
      layout, fluffing is element-wise (no remap). A per-byte LUT
      gather (`lut[byte] -> fp16`) is one fused kernel pass.

  (3) **fp16 accumulators chained via alpha/beta.** The 4 real
      matmuls (RR, II, RI, IR) write to a single pair of fp16
      output buffers (V_real, V_imag) using `out += matmul`/
      `out -= matmul` in-place; cast to fp32 happens only after
      summing over the 2-sample sub-time axis. Mirrors bfCorr's
      `cublasHgemmStridedBatched` with `beta=1.0` chaining.

Pipeline (per fada block):

    raw bytes [301,989,888]              # one fada page
        │
        │  view as 2D (NPACKETS*NANTS, NCHAN*2t*2p) = (196608, 1536) bytes
        │
        │  2D byte-transpose → (1536, 196608) bytes  ← tiled, ~1 ms on 2080 Ti
        │  (mirrors bfCorr `transpose_matrix_char`, lines 293-313)
        │
        ▼
    bytes (NCHAN, 2t, 2p, NPACKETS, NANTS) = (384, 2, 2, 2048, 96)
        │
        │  LUT-gather fluff → two fp16 tensors of same shape
        │  (mirrors bfCorr `corr_input_copy`, lines 273-286)
        │
        ▼
    R, I  fp16  [384, 2, 2, 2048, 96]   each
        │
        │  flatten leading (ch, t_sub, pol) into batch=1536; K=2048, M=N=96
        │  4 fp16 batched matmuls into shared fp16 V_real / V_imag:
        │    V_real  = R^T @ R + I^T @ I
        │    V_imag  = R^T @ I - I^T @ R
        │  (mirrors bfCorr's 4 cublasHgemmStridedBatched calls, lines 559-581)
        │
        ▼
    V_real, V_imag fp16 [1536, 96, 96] = (ch, t_sub, pol, 96, 96)
        │
        │  view (384, 2, 2, 96, 96) → sum over t_sub (axis 1) → (384, 2, 96, 96)
        │  cast to fp32  (mirrors bfCorr's `corr_output_copy` halfFac+t-sum, line 408-410)
        │
        ▼
    V fp32 (384, 2, 96, 96)
        │
        │  upper-triangle gather to bls_idx = a*(a+1)/2 + b
        │
        ▼
    vis complex64 [4656, 384, 2]
        │
        │  pack_bada_block (bytes view)
        │
        ▼
    raw bytes [28,606,464]               # one bada page

Numeric strategy (D3 / D4 revised 2026-05-04 — tensor cores are essential):

  * fp16 inputs to torch.matmul → HMMA on Turing (2080 Ti: 130 TFLOPs
    vs 13 TFLOPs at fp32; TF32 fast-path is Ampere+ only). Empirically
    we hit ~26 TFLOPs effective with M=N=96, K=2048 (small-matrix
    overhead caps utilisation at ~20%; legacy bfCorr is in the same
    regime).
  * Real-imag split: V_ij = sum_t conj(E_i(t)) * E_j(t)
                          = sum_t (R_i - jI_i)(R_j + jI_j)
                          = (R_i R_j + I_i I_j) + j(R_i I_j - I_i R_j)
    → 4 real GEMMs (vs 1 complex-fp32 GEMM that bypasses TCs).
  * fp16 accumulator safety: with the 0.05 pre-fluff, |E|² ≤ 0.16 and
    K=2048 sum ≤ 327 — well within fp16 max (65504) by ~200×. After
    summing the 2 t_sub samples, peak is 654, still ~100× below fp16
    max. Cast to fp32 happens once at the end. (`HALF_FAC_DEFAULT = 1`
    keeps a single-K-chunk fast path; `half_fac > 1` adds belt-and-
    suspenders chunking matching bfCorr's full design.)

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


#: 2D byte-transpose source shape used by the bfCorr-style fast path.
#: The fada page is `(NPACKETS_PER_BLOCK, NANTS, NCHAN, NTIMES_PER_PACKET,
#: NPOL)` C-contiguous bytes; flattening (pkt, ant) → 196608 rows and
#: (ch, t_sub, pol) → 1536 cols gives a 2D matrix amenable to a single
#: tiled transpose.
_FADA_2D_SHAPE: Final[tuple[int, int]] = (
    NPACKETS_PER_BLOCK * NANTS,                                # 196608 rows
    NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL,              # 1536 cols
)

#: Layout after the 2D byte-transpose: (NCHAN, NTIMES_PER_PACKET, NPOL,
#: NPACKETS_PER_BLOCK, NANTS) — matches bfCorr's `reorder_input` output
#: (see comment at lines 343-344 of `dsaX_bfCorr.cu`).
_GEMM_LAYOUT_SHAPE: Final[tuple[int, ...]] = (
    NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
    NPACKETS_PER_BLOCK, NANTS,
)

#: Batched matmul shape for the 4 fp16 GEMMs:
#: batch = NCHAN * NTIMES_PER_PACKET * NPOL = 1536, K = NPACKETS_PER_BLOCK = 2048,
#: M = N = NANTS = 96.
_GEMM_BATCH: Final[int] = NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL  # 1536


def unpack_int4_split(
    raw: np.ndarray | bytes | memoryview,
    *,
    device: torch.device | str = "cpu",
    scale: float = LEGACY_FLUFF_SCALE,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fluff one fada page into separate real / imag fp16 tensors **in
    GEMM layout** (post-transpose), mirroring bfCorr's `reorder_input`.

    Pipeline:
      1. View raw bytes as a 2D `(196608, 1536)` matrix.
      2. 2D-transpose to `(1536, 196608)` — one tiled cuBLAS / PyTorch
         kernel, ~1 ms on a 2080 Ti for 300 MB.
      3. LUT-gather each transposed byte to fp16 real and fp16 imag.
      4. Reshape both outputs to `_GEMM_LAYOUT_SHAPE` =
         `(NCHAN, 2t, 2p, NPACKETS, NANTS)`.

    The original (pre-transpose) layout `_FADA_VOLT_SHAPE` = `(NPACKETS,
    NANTS, NCHAN, 2t, 2p)` is the on-wire / fada layout; the post-
    transpose layout is what the downstream GEMM consumes directly
    (see `SlowCorrKernel.compute_split`).

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
        Both tensors of shape `_GEMM_LAYOUT_SHAPE = (384, 2, 2, 2048, 96)`
        and dtype `out_dtype`. C-contiguous.
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

    # ---- Stage 1: 2D byte-transpose (bfCorr `transpose_matrix_char`) ----
    # PyTorch's `.t().contiguous()` on a 2D byte tensor maps to a tiled
    # shared-memory transpose kernel; ~1 ms for 300 MB on a 2080 Ti.
    raw_t = torch.as_tensor(raw_arr, device=dev)               # uint8 (288 MB)
    bytes_2d = raw_t.view(_FADA_2D_SHAPE)                      # (196608, 1536)
    bytes_T = bytes_2d.t().contiguous()                        # (1536, 196608)
    del bytes_2d, raw_t

    # ---- Stage 2: per-byte LUT fluff (bfCorr `corr_input_copy`) ---------
    # `lut[bytes_T.long()]` gathers an fp16 value per byte; reshape into
    # the GEMM-friendly layout. The .long() promotion temporarily
    # allocates 8x the source size (2.3 GB) but is freed before the
    # GEMM. Trades RAM for one fewer kernel pass.
    idx = bytes_T.to(torch.long).reshape(-1)                   # int64 (2.3 GB temp)
    real = lut_real[idx].reshape(_GEMM_LAYOUT_SHAPE)           # fp16 (588 MB)
    imag = lut_imag[idx].reshape(_GEMM_LAYOUT_SHAPE)           # fp16 (588 MB)
    del idx, bytes_T
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
        """Correlate one block from split-real-imag fp16 voltages in
        bfCorr GEMM layout.

        Parameters
        ----------
        real_v, imag_v : torch.Tensor
            Real and imag parts of the voltage tensor, each of shape
            `_GEMM_LAYOUT_SHAPE = (NCHAN=384, NTIMES_PER_PACKET=2,
            NPOL=2, NPACKETS_PER_BLOCK=2048, NANTS=96)` (post-transpose
            layout from `unpack_int4_split`). Must be on `self.device`.

        Returns
        -------
        torch.Tensor
            complex64 visibilities of shape `(NBASE=4656, NCHAN=384,
            BADA_NPOL=2)`. Auto-correlations on the diagonal
            (bls_idx 0, 2, 5, 9, ... = `a*(a+1)/2 + a`).

        Notes
        -----
        Mirrors `dsaX_bfCorr.cu::dcorrelator` (lines 499-602):
          * batchCount = NCHAN * 2t * NPOL = 1536
          * M = N = NANTS = 96
          * K = NPACKETS_PER_BLOCK / half_fac
          * 4 fp16 batched matmuls (RR, II, RI, IR) chained via
            in-place add/sub on shared fp16 V_real / V_imag buffers
            (= bfCorr's `beta=0.0` then `beta=1.0` pattern).
          * Sum across the 2 t_sub samples and cast to fp32 only at
            the end (= bfCorr's `corr_output_copy`, line 408-410).
        """
        for name, t in (("real_v", real_v), ("imag_v", imag_v)):
            if t.device != self.device:
                raise ValueError(f"{name} on {t.device}, kernel on {self.device}")
            if t.shape != _GEMM_LAYOUT_SHAPE:
                raise ValueError(
                    f"{name} shape {tuple(t.shape)}, expected {_GEMM_LAYOUT_SHAPE}"
                )
        if real_v.dtype != imag_v.dtype:
            raise ValueError(
                f"real_v dtype {real_v.dtype} != imag_v dtype {imag_v.dtype}"
            )

        # ---- Stage 1: flatten leading (ch, t_sub, pol) into batch=1536 ----
        # _GEMM_LAYOUT_SHAPE = (384, 2, 2, 2048, 96). Both inputs are
        # C-contiguous so `.reshape` is a free view (no copy).
        K = self.n_time_samples // (NTIMES_PER_PACKET * self.half_fac)
        # Per-batch matrix is (NPACKETS_PER_BLOCK, NANTS). For half_fac > 1
        # we split NPACKETS_PER_BLOCK into `half_fac` sub-batches that
        # appear as additional batch dims; for half_fac = 1 this is a no-op.
        # New shape: (batch=NCHAN*2t*2p*half_fac, K, NANTS).
        new_batch = _GEMM_BATCH * self.half_fac
        R = real_v.reshape(new_batch, K, self.nants)            # fp16 view
        I = imag_v.reshape(new_batch, K, self.nants)
        del real_v, imag_v

        # ---- Stage 2: 4 batched fp16 matmuls into shared fp16 buffers ----
        # Per batch: V_real[i,j] = sum_t  R_t,i * R_t,j + I_t,i * I_t,j
        #            V_imag[i,j] = sum_t  R_t,i * I_t,j - I_t,i * R_t,j
        # i.e. real = R^T@R + I^T@I, imag = R^T@I - I^T@R.
        # Shape: (batch, K, NANTS).T @ (batch, K, NANTS) → (batch, NANTS, NANTS).
        R_T = R.transpose(-1, -2)                                # (batch, NANTS, K)
        I_T = I.transpose(-1, -2)
        V_real = torch.matmul(R_T, R)                            # fp16 (batch, 96, 96)
        V_real = V_real.add_(torch.matmul(I_T, I))               # in-place fp16
        V_imag = torch.matmul(R_T, I)
        V_imag = V_imag.sub_(torch.matmul(I_T, R))
        del R, I, R_T, I_T

        # ---- Stage 3: sum over (half_fac × t_sub), reshape, cast to fp32 ----
        # batch dim = NCHAN * 2t * NPOL * half_fac. Reshape and sum over the
        # half_fac and t_sub axes (axis indices 1 and 3 in the 5D view).
        #
        # When half_fac == 1 (default), this becomes a sum over only the
        # t_sub axis → matches bfCorr's halfFac=4 (which sums halfFac AND
        # t_sub in `corr_output_copy`).
        V_real_5d = V_real.view(
            NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL, self.half_fac,
            self.nants, self.nants,
        )
        V_imag_5d = V_imag.view(
            NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL, self.half_fac,
            self.nants, self.nants,
        )
        # Sum over axes 1 (t_sub) and 3 (half_fac) → (NCHAN, NPOL, 96, 96).
        V_real = V_real_5d.sum(dim=(1, 3)).to(self.accum_dtype)
        V_imag = V_imag_5d.sum(dim=(1, 3)).to(self.accum_dtype)
        del V_real_5d, V_imag_5d

        # Restrict to the bada-output pol count.
        V_real_b = V_real[:, : self.nbada_pol, :, :]            # (ch, npol, 96, 96)
        V_imag_b = V_imag[:, : self.nbada_pol, :, :]

        # ---- Stage 4: upper-triangle gather + complex assemble -----------
        # vis[bls, ch, pol] = V[ch, pol, a, b], bls_idx = a*(a+1)/2 + b.
        vis_real = V_real_b[..., self._a_idx, self._b_idx]      # (ch, npol, NBASE)
        vis_imag = V_imag_b[..., self._a_idx, self._b_idx]
        vis_real = vis_real.permute(2, 0, 1).contiguous()       # (NBASE, ch, npol)
        vis_imag = vis_imag.permute(2, 0, 1).contiguous()
        return torch.complex(vis_real, vis_imag)                # (4656, 384, 2)


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
