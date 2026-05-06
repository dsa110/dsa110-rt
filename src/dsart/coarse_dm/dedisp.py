"""Coarse-DM image-cube dedisperser (M3 chunk 3b; plan §3.6.2 + §4.2).

Acts on a per-chgroup image cube ``[T_fast, NCHAN_PER_CHGROUP, N_grid,
N_grid]`` (complex Stokes-I per channel, output of the gridder + iFFT)
and produces a coarse-DM-resolved image cube ``[T_dedisp, N_DM_coarse,
N_grid, N_grid]`` (real Stokes-I power, summed across channels at each
trial DM).

Pipeline placement (per plan §4.2 streaming pipeline)::

    fast-corr GEMM  →  cal/RFI weight  →  pol-sum (Stokes I)  →
    [stage-1 per-channel intra-chgroup coarse-DM]  →  gridder  →
    sparse-COO gather  →  static-sky subtract  →  quantize  →
    stage-2 cross-chgroup FIFO  →  transport-TX

Where this module fits:

* In **production** (chunk 4 corr_fast_compute integration), the per-
  channel intra-chgroup integer-sample shifts are applied **on the
  visibility tensor**, BEFORE the gridder, exactly as plan §4.2 step 4
  describes (`stage1_per_channel_shift`). The math (which channel
  shifts by how much, integer-rounded, anchored to a chgroup
  reference frequency) is what this module provides — the precomputed
  delay table from :class:`dsart.coarse_dm.dm_plan.DMPlan`.

* The **image-cube form** :func:`coarse_dedisp` here (post-gridder,
  post-iFFT, *image*-domain dedispersion) is the algorithmically
  equivalent reference primitive used by:
    - the chunk-3b acceptance tests (synthetic image-cube bursts);
    - the chunk-3b operator-review bench (`bench/coarse_dm_recovery.py`);
    - the M5 search-side detector's coarse-DM image-cube cross-check
      (read-only consumer);
    - any future debug scope where one wants to inspect what an image-
      domain dedispersion of the captured per-chgroup transport-TX
      cubes (replayed back through the iFFT) looks like.
  The math (Σ_ch shift(pwr[t, ch], delay) ) is identical between this
  image-power-domain primitive and the production visibility-domain
  stage-1 (a linear combination of pre-gridder shifted visibilities,
  followed by an iFFT and modulus-squared, gives the same per-pixel
  coarse-DM image up to fp16 round-off — established by chunk-3b's
  ``test_coarse_dedisp_recovers_synthetic_burst``).

Sign convention (Convention A)
==============================

Per :mod:`dsart.coarse_dm.dm_plan`: the per-(chgroup, ch, dm) shift
is non-negative, identically zero at the chgroup's TOP channel, and
maximal at the BOT channel. The dedispersion is then:

.. math::

   \\text{out}[t', g, c, l, m] = \\sum_\\text{ch}
       \\text{pwr}[t' + \\Delta_\\text{bins}(g, ch, c), ch, l, m]

with ``t' ∈ [0, T_fast - max_Δ_bins(g, c))`` for output to be valid.
Since ``Δ_bins(g, ch=0, c) = 0``, the channel-0 contribution is
``pwr[t', 0, l, m]`` directly (no shift); the channel-(N-1)
contribution at the longest delay is ``pwr[t' + max_Δ, N-1, l, m]``,
i.e. we read FORWARD into the cube to gather the dispersed signal
back into the dedispersed time bin.

A burst that arrives at the chgroup-top channel at fast-vis bin ``T0``
will appear in this dedispersed output at ``t' = T0`` (when run at
the matching DM trial). This is what
``test_coarse_dedisp_recovers_synthetic_burst`` pins.

Native-sample alignment (F24)
=============================

The shift is computed as
``round(delay_native_samples[g, ch, c] / t_int_fast_native)`` — i.e.
the canonical delay is in NATIVE samples (32.768 µs), and the bin
shift is derived once at apply time. See
:meth:`dsart.coarse_dm.dm_plan.DMPlan.delay_bins` for the full
derivation. ``test_F24_coarse_dm_uses_native_t_axis`` pins this.

Performance
===========

The reference implementation here is a **straightforward Python loop
over (dm_idx, ch_idx)** with PyTorch slice-add for each ``(ch, dm)``
pair. For the chunk-3b smoke tests at synthetic ``N_grid ∈ {16, 32,
64}`` and ``NCHAN ∈ {16, 24}``, this is fast enough (≤ 100 ms on CPU
for the worst test). The production GPU path lives in chunk 4 (the
fused per-coarse stage-1 cupy/triton kernel referenced in plan §4.2
line 1303) and shares only the delay table + sign convention with
this primitive. **No effort is spent here on hot-path GPU speedup**;
this is a reference primitive, not a service.

A small per-chgroup optimisation IS done: channels with identical
``(shift, dm)`` are grouped so the slice-add happens once per unique
shift (not once per channel), which buys a ~10× speedup at native
NCHAN=384 (most contiguous channel ranges share a shift at low DMs).

References
==========

* Plan §3.2 (DM plan).
* Plan §3.6.1 (dispersion law) + §3.6.2 (DEDISP architecture).
* Plan §4.2 lines 1296-1303 + 1322-1346 (streaming pipeline).
* :class:`dsart.coarse_dm.dm_plan.DMPlan` — delay-table provider.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch

from dsart.common.constants import N_CHGROUP, NCHAN_PER_CHGROUP
from dsart.coarse_dm.dm_plan import DMPlan


__all__ = [
    "coarse_dedisp",
    "max_output_t_dedisp",
]


_CFLOAT_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.complex32, torch.complex64, torch.complex128,
)


def max_output_t_dedisp(
    t_fast: int, plan: DMPlan, *, chgroup: int,
    dm_indices: np.ndarray | None = None,
) -> int:
    """``T_dedisp`` for a per-chgroup cube of length ``T_fast`` at this DMPlan.

    ``T_dedisp = T_fast - max_bins`` where ``max_bins`` is the largest
    bin shift across (channel, DM trial) for this chgroup. If
    ``dm_indices`` is given, the max is over only those trials.

    Returns ``0`` (clamped non-negative) when the cube is too short to
    accommodate any usable dedispersed bin; callers may treat 0 as
    "skip emit" (matches plan §3.6.2 stage-2 warm-up no-emit policy).
    """
    bin_shifts = plan.delay_bins_per_chgroup(chgroup)               # (NCHAN, N_coarse) int64
    if dm_indices is not None:
        bin_shifts = bin_shifts[:, np.asarray(dm_indices, dtype=np.int64)]
    max_b = int(bin_shifts.max()) if bin_shifts.size else 0
    return max(0, int(t_fast) - max_b)


def coarse_dedisp(
    cubes_in: torch.Tensor,
    plan: DMPlan,
    *,
    chgroup: int,
    dm_indices: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dedisperse a per-chgroup image cube across channels at each coarse-DM trial.

    Parameters
    ----------
    cubes_in : torch.Tensor
        Shape ``(T_fast, NCHAN_PER_CHGROUP, N_grid, N_grid)`` complex
        (cfp16 / cfp32 / cfp64). Per-channel image cube at fast-vis
        cadence; output of (gridder → iFFT → image plane).
    plan : DMPlan
        Coarse-DM plan whose ``t_int_fast_us`` matches the cube's time
        cadence. ``plan.chgroup_freqs_GHz[chgroup]`` is used for the
        per-channel delay computation.
    chgroup : int
        Chgroup index ``0..N_CHGROUP-1`` for the input cube. Selects
        which row of ``plan._delay_native_samples_table`` is used.
    dm_indices : torch.Tensor | None
        Optional ``(N_dm_subset,) int64`` subset of coarse-DM trial
        indices to compute. If ``None``, computes all
        ``plan.n_coarse`` trials. Useful when:
            - a chunk-3b smoke test wants to skip expensive trials, or
            - the chunk-4 production loop unrolls one trial at a time
              for memory-peak control (plan §4.2 line 1302).
    output_dtype : torch.dtype
        Real dtype for the output cube. Default ``float16`` matches
        the corr-side stage-2 FIFO storage convention; tests that need
        tighter rounding pass ``float32``.

    Returns
    -------
    torch.Tensor
        Shape ``(T_dedisp, N_dm_subset, N_grid, N_grid)``
        ``output_dtype`` real, where ``T_dedisp = T_fast -
        max_delay_bins`` (over the selected DM subset). Each pixel is
        the sum of ``|cube_in|²`` across all channels after applying
        the per-channel forward bin shift for the trial DM.

    Raises
    ------
    TypeError
        If ``cubes_in`` is not complex.
    ValueError
        If shape mismatches the expected
        ``(T, NCHAN_PER_CHGROUP, N_grid, N_grid)`` form, or if
        ``T_fast`` is too small for any selected DM trial (max bin
        shift ≥ T_fast).

    Notes
    -----
    - **Stokes-I power domain**: the input is the complex image
      (output of iFFT2 of the +uv-grid; F20 sign convention applies
      upstream — this module DOES NOT re-negate). The output is the
      modulus-squared (real, ≥ 0); the dedispersed value is the
      across-channel sum of these per-channel powers. This matches
      the standard incoherent-dedispersion definition and is what the
      M5 search-side detector pass consumes.
    - **Accumulator dtype**: fp32. Even at ``NCHAN_PER_CHGROUP = 384``
      and ``output_dtype = float16``, the in-kernel accumulator runs
      in fp32 to avoid the fp16-ish-of-N-summands accuracy loss that
      :doc:`gridder.kernel <dsart.grid.kernel>` G10 documents.
      Final cast to ``output_dtype`` happens at the return.
    - **Channel grouping for slice-add**: channels that share the
      same ``bin_shift`` for a given DM trial are accumulated
      together via a single ``out[:, c_i] += pwr[shift:shift+T_dedisp].sum(dim=1)``
      slice-add. At small ``N_chan_test`` this is a no-op; at native
      ``NCHAN_PER_CHGROUP = 384`` with low DMs this collapses to a
      ~10× fewer slice-adds.
    """
    if not 0 <= chgroup < N_CHGROUP:
        raise ValueError(
            f"chgroup={chgroup}, expected 0..{N_CHGROUP - 1}"
        )
    if cubes_in.dtype not in _CFLOAT_DTYPES:
        raise TypeError(
            f"cubes_in.dtype={cubes_in.dtype}, expected complex "
            f"(complex32/64/128)"
        )
    if cubes_in.ndim != 4:
        raise ValueError(
            f"cubes_in must be 4-D (T_fast, NCHAN, N_grid, N_grid); "
            f"got {cubes_in.ndim}-D shape {tuple(cubes_in.shape)}"
        )
    t_fast, nchan, ng_r, ng_c = cubes_in.shape
    # Allow a reduced channel count for synthetic tests that build a
    # smaller-than-NCHAN cube: but require a slice of plan's delay
    # table that matches it. The freq table is full-NCHAN, so we slice.
    if nchan > NCHAN_PER_CHGROUP:
        raise ValueError(
            f"cubes_in.shape[1]={nchan} > NCHAN_PER_CHGROUP="
            f"{NCHAN_PER_CHGROUP}"
        )
    if ng_r != ng_c:
        # Non-square grids allowed but flagged; the production gridder
        # always produces N_grid × N_grid square.
        pass

    # ------------------------------------------------------------------
    # Resolve dm_indices and bin shifts
    # ------------------------------------------------------------------
    if dm_indices is None:
        dm_idx_np = np.arange(plan.n_coarse, dtype=np.int64)
    else:
        dm_idx_np = dm_indices.cpu().numpy().astype(np.int64, copy=False)
        if dm_idx_np.ndim != 1:
            raise ValueError("dm_indices must be 1-D")
        if dm_idx_np.size and (
            dm_idx_np.min() < 0 or dm_idx_np.max() >= plan.n_coarse
        ):
            raise IndexError(
                f"dm_indices contain out-of-range trials (max={int(dm_idx_np.max())},"
                f" plan.n_coarse={plan.n_coarse})"
            )
    n_dm = dm_idx_np.shape[0]
    if n_dm == 0:
        raise ValueError("dm_indices is empty; nothing to dedisperse")

    # Full-chgroup bin-shift table, then slice to the requested
    # channel count (synthetic tests use NCHAN_test < NCHAN_PER_CHGROUP).
    bin_shifts_full = plan.delay_bins_per_chgroup(chgroup)          # (NCHAN_full, N_coarse)
    bin_shifts_subset = bin_shifts_full[:nchan][:, dm_idx_np]        # (NCHAN_test, N_dm)
    max_shift = int(bin_shifts_subset.max())
    if t_fast - max_shift <= 0:
        raise ValueError(
            f"cubes_in T_fast={t_fast} too short for max bin shift "
            f"{max_shift} (requires T_fast > max_shift); "
            f"selected DMs span {plan.dm_pc_cc[dm_idx_np[0]]:.1f}.."
            f"{plan.dm_pc_cc[dm_idx_np[-1]]:.1f} pc/cc on chgroup={chgroup}"
        )
    t_dedisp = t_fast - max_shift

    # ------------------------------------------------------------------
    # Compute the per-channel power cube once (fp32 real)
    # ------------------------------------------------------------------
    # cubes_in is complex; pwr = re² + im² (cheaper than .abs().pow(2)
    # which goes through sqrt then square; same answer in fp32).
    re = cubes_in.real.to(torch.float32)
    im = cubes_in.imag.to(torch.float32)
    pwr = re * re + im * im                                        # (T_fast, NCHAN, ng, ng) fp32
    del re, im

    # ------------------------------------------------------------------
    # Allocate output (fp32 accumulator; cast at return)
    # ------------------------------------------------------------------
    out = torch.zeros(
        (t_dedisp, n_dm, ng_r, ng_c),
        dtype=torch.float32,
        device=cubes_in.device,
    )

    # ------------------------------------------------------------------
    # Inner loop: per-DM, group channels by shift and slice-add
    # ------------------------------------------------------------------
    for c_i in range(n_dm):
        shifts_c = bin_shifts_subset[:, c_i]                        # (NCHAN_test,) int64
        # Group channels by shift value to coalesce slice-adds.
        unique_shifts, inverse = np.unique(shifts_c, return_inverse=True)
        for s_i, s in enumerate(unique_shifts.tolist()):
            ch_in_group = np.where(inverse == s_i)[0]
            ch_t = torch.from_numpy(ch_in_group).to(
                device=cubes_in.device, dtype=torch.int64,
            )
            # pwr[shift:shift+T_dedisp, channels in group, l, m].sum(dim=1)
            pwr_slice = pwr.index_select(1, ch_t)                   # (T_fast, |grp|, ng, ng)
            window = pwr_slice[s:s + t_dedisp]                      # (T_dedisp, |grp|, ng, ng)
            out[:, c_i] += window.sum(dim=1)
            del pwr_slice, window

    return out.to(output_dtype)
