"""Slow correlator kernel (M2; plan §8 lines 2161-2177).

Pure-tensor pipeline that fluffs `fada` int4 voltages and computes
upper-triangle visibilities for `bada`. Device-agnostic (CUDA / CPU)
so tests in `test_slow_corr_synth.py` exercise this module directly
without PSRDADA.

Pipeline (per fada block):

    raw bytes [301,989,888]              # one fada page
        │
        │  unpack_int4_complex (fluff)
        │
        ▼
    voltages complex64 [2048, 96, 384, 2t, 2p]
        │
        │  (transpose to (ch, pol, ant, t_total))
        │
        ▼
    E complex64 [384, 2, 96, 4096]
        │
        │  V[ch, pol] = E[ch, pol].conj() @ E[ch, pol].transpose(-1, -2)
        │  (D5: V_ij = E_i^* · E_j; sum over t_total)
        │
        ▼
    V_full complex64 [384, 2, 96, 96]
        │
        │  upper-triangle scatter to bls_idx = a*(a+1)/2 + b, b ≤ a
        │  (D4: matches legacy xGPU upper-triangle order; auto-corrs included)
        │
        ▼
    vis complex64 [4656, 384, 2]
        │
        │  pack_bada_block (bytes view)
        │
        ▼
    raw bytes [28,606,464]               # one bada page

NO calibration is applied (D2 in M2_PLAN_FIXES.md): slow visibilities
are uncalibrated by design — the user derives cal solutions from them
downstream. The legacy ×0.05 fluff scale (LEGACY_FLUFF_SCALE) IS
applied so output amplitudes carry the same physical units as
`dsaX_bfCorr` corr-mode (preserves any legacy cal code that already
calibrates against the historical scale).

Numeric choices (D3, D4):
  * fp32 throughout (`torch.float32` real-part / `torch.complex64`
    complex). Legacy uses fp16 with halfFac=4 to dodge fp16 overflow;
    M2 fresh-impl picks fp32 for simplicity. Legacy's `2048 × 0.05 ×
    0.05 ≈ 5` per-cell sum is well within fp32 precision; per-block
    GEMM ≈ 6 GFLOPs which is < 1 ms on a modern GPU and ~50 ms on CPU
    (fits comfortably under the 134-ms native cadence even on CPU).
  * `torch.matmul` (cuBLAS Cgemm under the hood on CUDA; MKL on CPU).
"""

from __future__ import annotations

from dataclasses import dataclass
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


# --- fluff: int4 complex bytes → complex tensor ---------------------------


def unpack_int4_complex(
    raw: np.ndarray | bytes | memoryview,
    *,
    device: torch.device | str = "cpu",
    scale: float = LEGACY_FLUFF_SCALE,
) -> torch.Tensor:
    """Fluff one fada page: signed 4-bit packed-cplx bytes → complex64 tensor.

    Each input byte holds two signed 4-bit nibbles (low = real, high = imag),
    each in [-8, 7] after sign-extension; both are multiplied by `scale`
    (`LEGACY_FLUFF_SCALE = 0.05` matches legacy `dsaX_bfCorr.cu` lines
    273-286 `corr_input_copy`).

    Parameters
    ----------
    raw : np.ndarray | bytes | memoryview
        Raw fada bytes; must be exactly `FADA_BYTES_PER_BLOCK = 301,989,888`
        bytes long (= prod(_FADA_VOLT_SHAPE)).
    device : torch.device | str
        Where to materialise the output tensor.
    scale : float
        Per-component scale factor (default = legacy 0.05).

    Returns
    -------
    torch.Tensor
        complex64 tensor of shape `_FADA_VOLT_SHAPE = (2048, 96, 384, 2, 2)`.

    Notes
    -----
    Sign-extension uses an arithmetic trick: `int8(byte_nibble) - 16` if
    nibble ≥ 8 else `int8(byte_nibble)`. Equivalent to legacy C
    `(char)(((unsigned char)x << 4)) >> 4` but without UB on signed shifts.
    """
    if isinstance(raw, (bytes, memoryview)):
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
    elif isinstance(raw, np.ndarray):
        if raw.dtype != np.uint8:
            raw_arr = raw.view(np.uint8)
        else:
            raw_arr = raw
    else:
        raise TypeError(f"unsupported raw type {type(raw)!r}")

    expected_n = int(np.prod(_FADA_VOLT_SHAPE))
    if raw_arr.size != expected_n:
        raise ValueError(
            f"fada raw size {raw_arr.size} != expected {expected_n} "
            f"(= prod{_FADA_VOLT_SHAPE})"
        )

    raw_t = torch.as_tensor(raw_arr, device=device)            # uint8

    real_nib = raw_t & 0x0F                                    # uint8 in [0, 15]
    imag_nib = raw_t >> 4                                       # uint8 in [0, 15]

    real_i16 = real_nib.to(torch.int16)
    imag_i16 = imag_nib.to(torch.int16)
    real_i16 = torch.where(real_i16 >= 8, real_i16 - 16, real_i16)
    imag_i16 = torch.where(imag_i16 >= 8, imag_i16 - 16, imag_i16)

    real = real_i16.to(torch.float32).mul_(scale)
    imag = imag_i16.to(torch.float32).mul_(scale)
    voltages = torch.complex(real, imag)                       # complex64

    return voltages.reshape(_FADA_VOLT_SHAPE)


# --- per-block correlator (GEMM + upper-triangle scatter) -----------------


@dataclass
class SlowCorrKernel:
    """Stateful per-device correlator kernel.

    Wraps the per-(device, dtype) precomputed indices so multiple blocks
    can be processed without re-allocating. Construct once at service
    startup; call `compute(voltages)` per block.
    """

    device: torch.device
    nants: int = NANTS
    nchan: int = NCHAN_PER_CHGROUP
    nvolt_pol: int = NPOL
    nbada_pol: int = BADA_NPOL
    n_time_samples: int = N_TIME_SAMPLES

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
        a_idx, b_idx = upper_tri_indices(self.nants)
        self._a_idx = torch.from_numpy(a_idx).to(self.device)
        self._b_idx = torch.from_numpy(b_idx).to(self.device)
        self._nbase = self._a_idx.numel()

    def compute(self, voltages: torch.Tensor) -> torch.Tensor:
        """Correlate one block of voltages.

        Parameters
        ----------
        voltages : torch.Tensor
            complex64 tensor of shape `(2048, 96, 384, 2t, 2p)` (= the
            output of `unpack_int4_complex`). Must be on `self.device`.

        Returns
        -------
        torch.Tensor
            complex64 visibilities of shape `(NBASE=4656, NCHAN=384,
            BADA_NPOL=2)`. Auto-correlations are included on the diagonal
            (bls_idx 0, 2, 5, 9, ... = `a*(a+1)/2 + a`).
        """
        if voltages.device != self.device:
            raise ValueError(
                f"voltages on {voltages.device}, kernel on {self.device}"
            )
        if voltages.dtype != torch.complex64:
            raise ValueError(
                f"voltages dtype {voltages.dtype}, expected complex64"
            )
        if voltages.shape != _FADA_VOLT_SHAPE:
            raise ValueError(
                f"voltages shape {tuple(voltages.shape)}, expected {_FADA_VOLT_SHAPE}"
            )

        # Reshape to (ch, pol, ant, t_total).
        # Source layout: (pkt, ant, ch, t_sub, pol).
        # Permute to (ch, pol, ant, pkt, t_sub) then merge (pkt, t_sub) → t_total.
        # `permute(...)` returns a view; `.contiguous()` materialises for the
        # GEMM (cuBLAS does not handle non-contiguous tensors well).
        E = voltages.permute(2, 4, 1, 0, 3).contiguous()            # (ch, pol, ant, pkt, t_sub)
        E = E.view(self.nchan, self.nvolt_pol, self.nants, self.n_time_samples)

        # Per-pol slow-corr GEMM:
        #   V_full[ch, pol, i, j] = sum_t conj(E[ch, pol, i, t]) * E[ch, pol, j, t]
        # = (E.conj()) @ E.transpose(-1, -2)         shape (ch, pol, ants, ants)
        # D5 in M2_PLAN_FIXES.md: V_ij = E_i^* · E_j (NOT textbook E_i · E_j^*).
        E_conj = E.conj()
        V_full = torch.matmul(E_conj, E.transpose(-1, -2))           # (ch, pol, 96, 96)

        # Upper-triangle scatter: vis[bls, ch, pol] = V_full[ch, pol, a, b]
        # with bls_idx = a*(a+1)/2 + b, b ≤ a (D4: legacy xGPU order).
        # Slice only the bada pols if BADA_NPOL < NPOL.
        V_full_b = V_full[:, : self.nbada_pol, :, :]                 # (ch, npol_bada, 96, 96)

        # Gather over the (a, b) axes via fancy indexing on the last 2 dims.
        # V_full_b shape after index: (ch, npol_bada, NBASE)
        vis_chpolbls = V_full_b[..., self._a_idx, self._b_idx]
        # Permute to (NBASE, ch, npol_bada).
        vis = vis_chpolbls.permute(2, 0, 1).contiguous()
        return vis                                                   # (4656, 384, 2)


# --- pack: complex64 tensor → bada page bytes -----------------------------


def pack_bada_block(vis: torch.Tensor) -> np.ndarray:
    """Convert visibility tensor to the byte layout meridian_fringestop expects.

    Per Subagent B's reading of `dsamfs/utils.py::read_buffer` (lines
    141-155): bada bytes are interpreted as `float32` pairs
    (`view(np.float32).reshape(-1, 2).view(np.complex64).reshape(-1, nbls,
    nchan, npol)`). So the on-wire layout is row-major
    `(nbls, nchan, npol)` complex64, which is what we already have.

    Parameters
    ----------
    vis : torch.Tensor
        complex64 visibilities, shape `(NBASE, NCHAN, BADA_NPOL)`.

    Returns
    -------
    np.ndarray
        uint8 view of length `BADA_BYTES_PER_INTEGRATION = 28,606,464`.
        Caller writes this into `writer.getNextPage()` then calls
        `markFilled()`.
    """
    if vis.shape != _BADA_VIS_SHAPE:
        raise ValueError(
            f"vis shape {tuple(vis.shape)}, expected {_BADA_VIS_SHAPE}"
        )
    if vis.dtype != torch.complex64:
        raise ValueError(f"vis dtype {vis.dtype}, expected complex64")

    arr = vis.detach().cpu().contiguous().numpy()                   # complex64
    return arr.view(np.uint8).reshape(-1)                           # bytes view
