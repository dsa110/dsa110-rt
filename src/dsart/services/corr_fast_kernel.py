"""Fast correlator kernel (M3; plan §4.2 line ~1080 / §8 line 2262).

Peer to :mod:`dsart.services.slow_corr_kernel`. Same 4-real-GEMM
structure as M2's :class:`SlowCorrKernel` (RR + II for V_real, RI − IR
for V_imag, fp16 inputs → fp32 accumulator with HALF_FAC chaining)
but emits **N independent fast-vis tiles per voltage block** instead
of one block-integrated visibility.

Why a peer kernel (not a flag on SlowCorrKernel)
================================================

The slow correlator's GEMM hard-codes summation across the full
``NPACKETS_PER_BLOCK · NTIMES_PER_PACKET = 4096`` time-sample axis
into one ``(NBASE, NCHAN, BADA_NPOL)`` output block (matches
``dsaX_bfCorr.cu::dcorrelator``). The fast correlator instead
needs ``n_fast_vis = 4096 / t_int_fast_native`` independent
output tiles per block, each summing only ``t_int_fast_native``
consecutive native time samples (262.144 µs at native, 4× that
during the M3 burst sub-DoD per the agreed test-mode override).

A boolean flag on the slow kernel would conflate two different
output layouts; a peer class is clearer + easier to test.

Pipeline (per voltage block)
============================

Input voltages must be in M2's GEMM layout (post-``unpack_int4_split``):

    (NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
     n_packets_in, NANTS)                fp16 (or fp32 for CPU debug)

with ``n_packets_in`` a multiple of ``packets_per_fast_vis =
t_int_fast_native // NTIMES_PER_PACKET``. The block doesn't have
to be the full ``NPACKETS_PER_BLOCK = 2048`` — tests pass smaller
blocks (e.g. 16 packets → 4 fast-vis tiles at t_int_fast_native=8).

::

    R, I  fp16  [NCHAN, 2t, 2p, n_packets_in, NANTS]
        │
        │  reshape n_packets_in → (n_fast_vis, packets_per_fast_vis)
        │  ⇒ [NCHAN, 2t, 2p, n_fast_vis, packets_per_fast_vis, NANTS]
        │
        │  permute n_fast_vis to leading position
        │  ⇒ [n_fast_vis, NCHAN, 2t, 2p, packets_per_fast_vis, NANTS]
        │
        │  flatten leading dims into batch
        │  ⇒ [batch=n_fast_vis*NCHAN*2t*2p, packets_per_fast_vis, NANTS]
        │
        │  4 batched fp16 matmuls into shared fp16 V_real / V_imag
        │  V_real = R^T @ R + I^T @ I       (fp16 buffers, HMMA)
        │  V_imag = R^T @ I − I^T @ R
        │
        ▼
    V_real, V_imag fp16 [batch, NANTS, NANTS]
        │
        │  reshape batch → (n_fast_vis, NCHAN, 2t, 2p, NANTS, NANTS)
        │  sum over t_sub axis (the 2-element 2t dim) → (n_fast_vis,
        │  NCHAN, 2p, NANTS, NANTS)
        │  cast to fp32 (single point of accumulator-precision drop;
        │  matches M2's pattern)
        │
        ▼
    V fp32 [n_fast_vis, NCHAN, 2p, NANTS, NANTS]
        │
        │  restrict to nbada_pol parallel-hand pols
        │
        ▼
    V fp32 [n_fast_vis, NCHAN, BADA_NPOL=2, NANTS, NANTS]
        │
        │  upper-triangle gather with F18 b/a index swap (see
        │  SlowCorrKernel docstring for the long explanation —
        │  same convention applies here)
        │
        ▼
    vis cfp32 [n_fast_vis, NBASE=4656, NCHAN, BADA_NPOL=2]

Output convention pinned by F18 + F21
=====================================

Per :mod:`dsart.cal.cal_loader` and the M3 chunk_1 acceptance tests,
the cal-apply step folds the F21 DEC-only fringe-stop phase into the
voltage tensor BEFORE this kernel sees it. Combined with F18's
``V_ij = conj(E_i) · E_j`` for ``i ≤ j`` baseline ordering (matching
dsamfs / pyuvdata), an on-source point at obs_dec produces a
real-only fast visibility on every (baseline, channel, fast-vis tile)
to fp16 numerical precision (≤ ~3e-3 rad equivalent). The
cross-validation test ``test_fast_corr_kernel_F18_F21_compose``
exercises this end-to-end against synthetic on-source voltages.

Memory budget
=============

The output tensor for a full block at the production t_int_fast_native=8
is ``512 × 4656 × 384 × 2 × 8 bytes ≈ 7.3 GB`` cfp32 — far too large
to materialise. Production wires this kernel directly to the gridder
(M3 chunk 3a), which reduces ``(n_fast_vis, NBASE, NCHAN)`` to
``(n_fast_vis, NCHAN_grouped, N_GRID_FAST²)`` sparse uv-cells per
tile, dropping the per-tile output to ~ tens of MB.

For chunk 2 (kernel-only, no gridder yet), tests use **small
synthetic blocks** (e.g. 16 packets → 4 fast-vis tiles at t_int=8)
that fit comfortably in 100s of MB. The full-block production budget
is enforced by the chunk 3a / chunk 4 integration, not by this kernel.

F31a — chunking the n_fast_vis axis
-----------------------------------

Independent of the cfp32 output-tensor budget above, the kernel's
fp16 ``V_real`` / ``V_imag`` GEMM intermediate has shape
``(n_fast_vis * NCHAN * NTIMES_PER_PACKET * NPOL, NANTS, NANTS)``.
At the production cadence (``t_int_fast_native=8`` ⇒ ``n_fast_vis=512``,
``NCHAN=384``, ``NTIMES_PER_PACKET=2``, ``NPOL=2``, ``NANTS=96``)
that's ``512 * 384 * 2 * 2 * 96² * 2 bytes ≈ 14.5 GB`` per buffer —
which doesn't fit on the **11 GB 2080Ti** production GPU (the
deployment target is the 2080Ti, NOT an A6000; confirmed 2026-05).

F31a tames this by chunking the leading ``n_fast_vis`` axis inside
:meth:`FastCorrKernel.compute_split` so each per-slab ``V_real`` /
``V_imag`` intermediate stays under :data:`_F31A_CHUNK_TARGET_BYTES`
(= 1 GB target peak by default). With that target the auto-pick
yields ``n_fv_chunk = 32`` at production → 16 slabs of ~865 MB each,
leaving headroom for input voltages, the upper-tri output, and any
gridder workspace that runs alongside.

Chunking is **bit-identical** to the un-chunked path: each slab feeds
the same fp16 matmul kernels with the exact same per-batch-element
inputs as a monolithic call would, and the slab outputs are
``torch.cat``'d on the leading ``n_fast_vis`` axis without any extra
arithmetic. Callers can pass ``n_fv_chunk=`` explicitly to override
the auto-pick (e.g. for benchmarking).

References
==========

* Plan §4.2 — fast-corr description.
* :mod:`dsart.services.slow_corr_kernel` — the M2 peer kernel; this
  module deliberately mirrors its GEMM structure for code-review
  readability + perf-pattern consistency.
* :mod:`dsart.cal.cal_loader` — F21 cal-apply that runs upstream of
  this kernel.
* M3_PLAN_FIXES.md — chunks 2, 3a, 3b for downstream integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from dsart.common.constants import (
    BADA_NPOL,
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    T_INT_FAST_NATIVE,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    upper_tri_indices,
)


__all__ = [
    "DEFAULT_T_INT_FAST_NATIVE",
    "FastCorrKernel",
    "stokes_i_pol_sum",
    "validate_fast_voltage_shape",
    "zero_v4_cells",
]


#: Default fast-corr integration depth in NATIVE samples per fast-vis tile.
#: Production target = T_INT_FAST_NATIVE = 8 (262.144 µs cadence). Tests
#: + the M3 burst sub-DoD use the 4× override (32 native samples,
#: 1048.576 µs) per user clarification 2026-05-05.
DEFAULT_T_INT_FAST_NATIVE: Final[int] = T_INT_FAST_NATIVE  # = 8


#: F31a auto-chunk target peak (in bytes) for the per-slab fp16 V_real /
#: V_imag matmul intermediate inside :meth:`FastCorrKernel.compute_split`.
#: When ``n_fv_chunk`` is left as ``None`` (default), compute_split picks
#: the largest power of two ``n_fv_chunk`` such that
#:
#:     ``n_fv_chunk * nchan * NTIMES_PER_PACKET * nvolt_pol * nants² * 2``
#:
#: bytes ≤ this target. At production
#: (``nchan=384, NTIMES_PER_PACKET=2, nvolt_pol=2, nants=96``) the
#: per-fast-vis fp16 footprint is ~27 MB, so the auto-pick lands at
#: ``n_fv_chunk = 32`` → 16 slabs of ~865 MB each. Sized to leave
#: comfortable headroom on the 11 GB 2080Ti production GPU.
#: This is a *target peak*, not a hard ceiling; callers can override
#: by passing ``n_fv_chunk=`` explicitly. See the module-level
#: "Memory budget — F31a" docstring section for the full rationale.
_F31A_CHUNK_TARGET_BYTES: Final[int] = 1 << 30                # 1 GiB


def validate_fast_voltage_shape(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    *,
    t_int_fast_native: int,
    nants: int = NANTS,
    nchan: int = NCHAN_PER_CHGROUP,
    nvolt_pol: int = NPOL,
) -> tuple[int, int]:
    """Validate fast-corr voltage shapes + return ``(n_fast_vis, packets_per_fast_vis)``.

    Parameters
    ----------
    real_v, imag_v : torch.Tensor
        Voltage real / imag in M2's GEMM layout
        ``(nchan, NTIMES_PER_PACKET, nvolt_pol, n_packets_in, nants)``.
        n_packets_in must be > 0 and must satisfy
        ``(NTIMES_PER_PACKET * n_packets_in) % t_int_fast_native == 0``.
    t_int_fast_native : int
        Number of NATIVE samples per fast-vis output tile. Must be a
        positive multiple of ``NTIMES_PER_PACKET = 2`` (so the integration
        boundary is packet-aligned — the unpack_int4_split tensor has
        no sub-packet time axis to split on).

    Returns
    -------
    (n_fast_vis_per_block, packets_per_fast_vis) : tuple of int
        n_fast_vis_per_block = NTIMES_PER_PACKET * n_packets_in /
        t_int_fast_native (number of fast-vis tiles emitted per call).
        packets_per_fast_vis = t_int_fast_native // NTIMES_PER_PACKET
        (number of input packets summed per output tile).

    Raises
    ------
    ValueError
        On any shape / dtype / divisibility mismatch.
    """
    if real_v.shape != imag_v.shape:
        raise ValueError(
            f"real_v shape {tuple(real_v.shape)} != imag_v shape {tuple(imag_v.shape)}"
        )
    if real_v.dtype != imag_v.dtype:
        raise ValueError(
            f"real_v dtype {real_v.dtype} != imag_v dtype {imag_v.dtype}"
        )
    if real_v.ndim != 5:
        raise ValueError(
            f"voltage tensor must be 5D (nchan, NTIMES_PER_PACKET, nvolt_pol, "
            f"n_packets_in, nants), got {real_v.ndim}D shape "
            f"{tuple(real_v.shape)}"
        )
    nch, nt, npol, n_packets_in, na = tuple(real_v.shape)
    if nch != nchan:
        raise ValueError(f"voltage nchan={nch}, expected {nchan}")
    if nt != NTIMES_PER_PACKET:
        raise ValueError(
            f"voltage NTIMES_PER_PACKET={nt}, expected {NTIMES_PER_PACKET}"
        )
    if npol != nvolt_pol:
        raise ValueError(f"voltage nvolt_pol={npol}, expected {nvolt_pol}")
    if na != nants:
        raise ValueError(f"voltage nants={na}, expected {nants}")
    if n_packets_in <= 0:
        raise ValueError(f"voltage n_packets_in must be > 0, got {n_packets_in}")

    if t_int_fast_native <= 0:
        raise ValueError(
            f"t_int_fast_native must be > 0, got {t_int_fast_native}"
        )
    if t_int_fast_native % NTIMES_PER_PACKET != 0:
        raise ValueError(
            f"t_int_fast_native ({t_int_fast_native}) must be a multiple of "
            f"NTIMES_PER_PACKET ({NTIMES_PER_PACKET}); the GEMM layout has no "
            f"sub-packet time axis to split on."
        )
    packets_per_fast_vis = t_int_fast_native // NTIMES_PER_PACKET
    if n_packets_in % packets_per_fast_vis != 0:
        raise ValueError(
            f"n_packets_in ({n_packets_in}) not a multiple of "
            f"packets_per_fast_vis ({packets_per_fast_vis} = "
            f"t_int_fast_native / NTIMES_PER_PACKET); cannot tile the "
            f"voltage block evenly into fast-vis integrations."
        )
    n_fast_vis_per_block = n_packets_in // packets_per_fast_vis
    return n_fast_vis_per_block, packets_per_fast_vis


@dataclass
class FastCorrKernel:
    """Stateful per-device fast-correlator kernel.

    Construct once at service startup; call :meth:`compute_split` per
    block on the (real, imag) tensors returned by
    :func:`dsart.services.slow_corr_kernel.unpack_int4_split` (after
    cal-apply with F21).

    Attributes
    ----------
    device : torch.device
        Where to run the GEMMs.
    t_int_fast_native : int
        Number of NATIVE samples per fast-vis output tile. Must be a
        positive multiple of NTIMES_PER_PACKET (= 2). Default
        :data:`DEFAULT_T_INT_FAST_NATIVE` = 8 (= 262.144 µs production
        cadence).
    nants, nchan, nvolt_pol, nbada_pol : int
        Same semantics as :class:`SlowCorrKernel`.
    accum_dtype : torch.dtype
        Dtype the t_sub sum is cast to before the upper-tri gather.
        Default ``torch.float32`` (matches M2). The 4 GEMMs themselves
        always run in the input dtype (fp16 for tensor cores).
    """

    device: torch.device
    t_int_fast_native: int = DEFAULT_T_INT_FAST_NATIVE
    nants: int = NANTS
    nchan: int = NCHAN_PER_CHGROUP
    nvolt_pol: int = NPOL
    nbada_pol: int = BADA_NPOL
    accum_dtype: torch.dtype = torch.float32

    _a_idx: torch.Tensor = field(init=False)
    _b_idx: torch.Tensor = field(init=False)
    _nbase: int = field(init=False)
    _packets_per_fast_vis: int = field(init=False)

    def __post_init__(self) -> None:
        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.device = dev
        if self.nbada_pol > self.nvolt_pol:
            raise ValueError(
                f"nbada_pol ({self.nbada_pol}) > nvolt_pol ({self.nvolt_pol})"
            )
        if self.t_int_fast_native % NTIMES_PER_PACKET != 0:
            raise ValueError(
                f"t_int_fast_native ({self.t_int_fast_native}) must be a multiple of "
                f"NTIMES_PER_PACKET ({NTIMES_PER_PACKET})"
            )
        if self.t_int_fast_native <= 0:
            raise ValueError(
                f"t_int_fast_native must be > 0, got {self.t_int_fast_native}"
            )
        self._packets_per_fast_vis = self.t_int_fast_native // NTIMES_PER_PACKET
        a_idx, b_idx = upper_tri_indices(self.nants)
        self._a_idx = torch.from_numpy(a_idx).to(self.device)
        self._b_idx = torch.from_numpy(b_idx).to(self.device)
        self._nbase = self._a_idx.numel()

    @property
    def n_fast_vis_per_full_block(self) -> int:
        """Number of fast-vis tiles emitted per full ``NPACKETS_PER_BLOCK`` block.

        = ``(NPACKETS_PER_BLOCK * NTIMES_PER_PACKET) / t_int_fast_native``
        = ``NPACKETS_PER_BLOCK / packets_per_fast_vis``.

        For the production cadence (``t_int_fast_native=8``): 512.
        For the M3 burst-test 4× cadence (``t_int_fast_native=32``): 128.
        """
        return NPACKETS_PER_BLOCK // self._packets_per_fast_vis

    def compute_split(
        self,
        real_v: torch.Tensor,
        imag_v: torch.Tensor,
        *,
        n_fv_chunk: int | None = None,
    ) -> torch.Tensor:
        """Correlate one block, emitting ``n_fast_vis`` independent tiles.

        Parameters
        ----------
        real_v, imag_v : torch.Tensor
            Voltage real / imag in GEMM layout
            ``(nchan, NTIMES_PER_PACKET, nvolt_pol, n_packets_in, nants)``.
            ``n_packets_in`` must be a positive multiple of
            ``packets_per_fast_vis = t_int_fast_native / NTIMES_PER_PACKET``.
            For full-block production, ``n_packets_in = NPACKETS_PER_BLOCK
            = 2048``; tests may pass smaller values.
        n_fv_chunk : int | None, keyword-only, default ``None``
            **F31a — chunking the n_fast_vis axis.** Per-slab fast-vis
            count for the inner fp16 matmul. ``None`` (default)
            auto-picks the largest power of two such that the per-slab
            fp16 ``V_real`` / ``V_imag`` intermediate stays under
            :data:`_F31A_CHUNK_TARGET_BYTES` (= 1 GB target peak),
            capped at the total ``n_fast_vis`` for this call. At
            production cadence (``n_fast_vis=512``) the auto-pick is
            32 → 16 slabs of ~865 MB each, keeping the kernel inside
            the 11 GB 2080Ti budget. Pass an explicit positive integer
            in ``[1, n_fast_vis_total]`` to override (e.g. for
            benchmarking). The chunked path is **bit-identical** to
            ``n_fv_chunk = n_fast_vis_total`` (same per-batch fp16
            matmul inputs, same upper-tri gather, ``torch.cat`` on
            the leading axis introduces no arithmetic). See the
            module-level "Memory budget — F31a" section for context.

        Returns
        -------
        torch.Tensor
            complex64 fast visibilities of shape
            ``(n_fast_vis, NBASE, NCHAN, BADA_NPOL)`` where
            ``n_fast_vis = (NTIMES_PER_PACKET * n_packets_in) /
            t_int_fast_native``.

            Auto-correlations on the diagonal (``bls_idx 0, 2, 5, 9, ...
            = a*(a+1)/2 + a``). Upper-tri gather uses the F18 b/a swap
            so vis[bls, ch, pol] matches ``conj(E_lower) · E_higher``
            (dsamfs / pyuvdata convention).
        """
        n_fast_vis, packets_per_fast_vis = validate_fast_voltage_shape(
            real_v, imag_v,
            t_int_fast_native=self.t_int_fast_native,
            nants=self.nants,
            nchan=self.nchan,
            nvolt_pol=self.nvolt_pol,
        )
        if real_v.device != self.device:
            raise ValueError(
                f"real_v on {real_v.device}, kernel on {self.device}"
            )
        if imag_v.device != self.device:
            raise ValueError(
                f"imag_v on {imag_v.device}, kernel on {self.device}"
            )
        n_packets_in = real_v.shape[3]
        assert n_packets_in == n_fast_vis * packets_per_fast_vis  # validate guarantee

        # ---- F31a: pick / validate the per-slab n_fv_chunk ----
        if n_fv_chunk is None:
            n_fv_chunk = self._auto_n_fv_chunk(n_fast_vis)
        elif not (1 <= n_fv_chunk <= n_fast_vis):
            raise ValueError(
                f"n_fv_chunk={n_fv_chunk} must be in "
                f"[1, n_fv_total={n_fast_vis}]"
            )

        # ---- F31a: slice n_packets_in into n_fv_chunk slabs ----
        # Each slab covers exactly `n_fv_slab` consecutive fast-vis
        # tiles, i.e. `n_fv_slab * packets_per_fast_vis` packets.
        # `.contiguous()` is required because the slab is sliced on
        # the n_packets_in axis (not the trailing axis), and
        # _compute_one_slab calls `.view()` to reshape it; PyTorch
        # rejects `.view()` on a non-contiguous tensor.
        out_chunks: list[torch.Tensor] = []
        for fv0 in range(0, n_fast_vis, n_fv_chunk):
            fv1 = min(fv0 + n_fv_chunk, n_fast_vis)
            n_fv_slab = fv1 - fv0
            p0 = fv0 * packets_per_fast_vis
            p1 = fv1 * packets_per_fast_vis
            real_slab = real_v[:, :, :, p0:p1, :].contiguous()
            imag_slab = imag_v[:, :, :, p0:p1, :].contiguous()
            vis_slab = self._compute_one_slab(
                real_slab, imag_slab,
                n_fv_slab=n_fv_slab,
                packets_per_fast_vis=packets_per_fast_vis,
            )
            out_chunks.append(vis_slab)
            # Drop slab refs eagerly so PyTorch's allocator can reuse
            # the workspace for the next slab. Critical on the 2080Ti;
            # harmless on CPU.
            del real_slab, imag_slab, vis_slab

        return torch.cat(out_chunks, dim=0)

    def _auto_n_fv_chunk(self, n_fv_total: int) -> int:
        """Pick the largest power-of-two ``n_fv_chunk`` keeping the per-slab
        fp16 ``V_real`` / ``V_imag`` intermediate under
        :data:`_F31A_CHUNK_TARGET_BYTES`.

        Per-slab fp16 footprint per buffer (one of V_real, V_imag):
        ``n_fv_chunk * nchan * NTIMES_PER_PACKET * nvolt_pol *
        nants² * 2 bytes``.
        """
        bytes_per_fv = (
            self.nchan
            * NTIMES_PER_PACKET
            * self.nvolt_pol
            * self.nants * self.nants
            * 2                                                  # fp16 = 2 B
        )
        # Largest fast-vis count whose fp16 V_real fits in the target.
        # `max(..., 1)` guards the (highly hypothetical) case of a
        # per-fast-vis footprint already over the target, which would
        # otherwise yield a chunk size of 0 and an infinite loop above.
        max_fv = max(_F31A_CHUNK_TARGET_BYTES // bytes_per_fv, 1)
        # Round down to a power of two. Power-of-two slabs guarantee
        # n_fast_vis % n_fv_chunk == 0 for all n_fast_vis values that
        # come out of the realistic t_int_fast_native ladder (which is
        # itself a power of two), so the trailing slab is never a
        # ragged remainder at production scales.
        chunk = 1 << (max_fv.bit_length() - 1)
        return min(chunk, n_fv_total)

    def _compute_one_slab(
        self,
        real_v_slab: torch.Tensor,
        imag_v_slab: torch.Tensor,
        *,
        n_fv_slab: int,
        packets_per_fast_vis: int,
    ) -> torch.Tensor:
        """Compute fast-vis output for one F31a slab of ``n_fv_slab`` tiles.

        Inputs are sliced + contiguous voltage tensors of shape
        ``(nchan, NTIMES_PER_PACKET, nvolt_pol,
        n_fv_slab * packets_per_fast_vis, nants)`` (i.e. a
        ``[:, :, :, p0:p1, :].contiguous()`` view of the full block
        voltages). Returns a slab of fast visibilities of shape
        ``(n_fv_slab, NBASE, NCHAN, BADA_NPOL)`` cfp32.

        This is the per-slab body of the un-chunked kernel; chunking
        on the leading n_fast_vis axis is purely a memory-budget
        transform (each slab feeds the same per-batch fp16 matmul
        inputs as a monolithic call would), so concatenating slab
        outputs reproduces the un-chunked result bit-identically on
        deterministic backends.
        """
        # ---- Stage 1: reshape n_packets_in into (n_fv_slab, ppfv) ----
        # Layout: (NCHAN, 2t, 2p, n_fv_slab, ppfv, NANTS) — pure view.
        R5 = real_v_slab.view(
            self.nchan, NTIMES_PER_PACKET, self.nvolt_pol,
            n_fv_slab, packets_per_fast_vis, self.nants,
        )
        I5 = imag_v_slab.view(
            self.nchan, NTIMES_PER_PACKET, self.nvolt_pol,
            n_fv_slab, packets_per_fast_vis, self.nants,
        )

        # ---- Stage 2: move n_fv_slab to leading dim ----
        # New layout: (n_fv_slab, NCHAN, 2t, 2p, ppfv, NANTS).
        # `permute` returns a non-contiguous view; `.contiguous()`
        # materialises it once so downstream `.reshape` calls don't
        # incur per-call layout copies.
        R6 = R5.permute(3, 0, 1, 2, 4, 5).contiguous()
        I6 = I5.permute(3, 0, 1, 2, 4, 5).contiguous()
        del R5, I5

        # ---- Stage 3: flatten leading dims into batched matmul ----
        # (n_fv_slab * NCHAN * 2t * 2p, ppfv, NANTS).
        new_batch = n_fv_slab * self.nchan * NTIMES_PER_PACKET * self.nvolt_pol
        R = R6.reshape(new_batch, packets_per_fast_vis, self.nants)  # fp16 view
        I = I6.reshape(new_batch, packets_per_fast_vis, self.nants)
        del R6, I6

        # ---- Stage 4: the 4 batched fp16 matmuls (same as SlowCorrKernel) ----
        # Per batch: V_real = R^T @ R + I^T @ I,
        #            V_imag = R^T @ I - I^T @ R.
        # (batch, ppfv, NANTS).T @ (batch, ppfv, NANTS) → (batch, NANTS, NANTS).
        R_T = R.transpose(-1, -2)                                # (batch, NANTS, ppfv)
        I_T = I.transpose(-1, -2)
        V_real = torch.matmul(R_T, R)                            # fp16 (batch, 96, 96)
        V_real = V_real.add_(torch.matmul(I_T, I))               # in-place fp16
        V_imag = torch.matmul(R_T, I)
        V_imag = V_imag.sub_(torch.matmul(I_T, R))
        del R, I, R_T, I_T

        # ---- Stage 5: sum over t_sub (axis 2 in the 6D view), cast to fp32 ----
        # batch dim = n_fv_slab * NCHAN * 2t * 2p. Reshape and sum the 2t axis.
        V_real_6d = V_real.view(
            n_fv_slab, self.nchan, NTIMES_PER_PACKET, self.nvolt_pol,
            self.nants, self.nants,
        )
        V_imag_6d = V_imag.view(
            n_fv_slab, self.nchan, NTIMES_PER_PACKET, self.nvolt_pol,
            self.nants, self.nants,
        )
        # Sum over t_sub → (n_fv_slab, NCHAN, nvolt_pol, NANTS, NANTS).
        V_real = V_real_6d.sum(dim=2).to(self.accum_dtype)
        V_imag = V_imag_6d.sum(dim=2).to(self.accum_dtype)
        del V_real_6d, V_imag_6d

        # Restrict to bada-output pol count (parallel-hands by default).
        V_real_b = V_real[:, :, : self.nbada_pol, :, :]          # (fv, ch, npol, 96, 96)
        V_imag_b = V_imag[:, :, : self.nbada_pol, :, :]

        # ---- Stage 6: upper-triangle gather + complex assemble ----
        # Same b/a index swap as SlowCorrKernel (see its docstring for
        # the long F18 explanation). vis[fv, bls, ch, pol] = V[fv, ch,
        # pol, b, a], bls_idx = a*(a+1)/2 + b.
        vis_real = V_real_b[..., self._b_idx, self._a_idx]       # (fv, ch, npol, NBASE)
        vis_imag = V_imag_b[..., self._b_idx, self._a_idx]
        # Move (NBASE) before (ch, pol).
        vis_real = vis_real.permute(0, 3, 1, 2).contiguous()     # (fv, NBASE, ch, npol)
        vis_imag = vis_imag.permute(0, 3, 1, 2).contiguous()
        return torch.complex(vis_real, vis_imag)                 # (fv, 4656, 384, 2)


# ---------------------------------------------------------------------------
# Downstream helpers (small enough to live with the kernel; kept here
# so chunks 3a / 3b / 4 don't have to add a new module just for these).
# ---------------------------------------------------------------------------


def stokes_i_pol_sum(vis: torch.Tensor) -> torch.Tensor:
    """Pol-sum two parallel-hand visibilities into Stokes I (per-tile, per-baseline, per-channel).

    ``I = (V_xx + V_yy)`` (no 1/2 normalisation — the search pipeline
    accumulates absolute power, not specific intensity, and downstream
    detectors apply their own normalisation per noise-power estimate).

    Parameters
    ----------
    vis : torch.Tensor
        Fast vis tensor, complex dtype, shape
        ``(..., NBADA_POL=2)`` with the last axis ordered ``(V_xx, V_yy)``
        or ``(V_yy, V_xx)`` (the sum is symmetric so the order doesn't
        matter for Stokes I).

    Returns
    -------
    torch.Tensor
        Same shape as input minus the trailing pol axis. Same dtype.
    """
    if vis.shape[-1] != 2:
        raise ValueError(
            f"expected last dim = 2 (V_xx, V_yy), got shape {tuple(vis.shape)}"
        )
    if not vis.is_complex():
        raise ValueError(
            f"expected complex visibility, got dtype {vis.dtype}"
        )
    return vis[..., 0] + vis[..., 1]


def zero_v4_cells(vis_v4: torch.Tensor) -> torch.Tensor:
    """Zero the cross-hand pol cells of a 4-pol visibility tensor in-place.

    Provided for downstream stages that consume a 4-pol layout
    ``(V_xx, V_yy, V_xy, V_yx)`` but where M3 only computes the
    parallel-hands. Callers can pass a 4-pol tensor with V_xy / V_yx
    set to whatever (uninitialised, garbage from a 4-pol kernel they
    later switched off, etc.) and this helper will null those cells
    deterministically before the downstream stage runs.

    Parameters
    ----------
    vis_v4 : torch.Tensor
        Shape ``(..., 4)`` with last axis ordered
        ``(V_xx, V_yy, V_xy, V_yx)``.

    Returns
    -------
    torch.Tensor
        The same tensor (view), with positions [..., 2] and [..., 3]
        set to zero. **In-place.**
    """
    if vis_v4.shape[-1] != 4:
        raise ValueError(
            f"expected last dim = 4 (V_xx, V_yy, V_xy, V_yx), "
            f"got shape {tuple(vis_v4.shape)}"
        )
    vis_v4[..., 2] = 0
    vis_v4[..., 3] = 0
    return vis_v4
