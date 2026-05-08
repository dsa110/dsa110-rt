"""Coarse-DM stage-1 vis-domain integer-sample shifts (chunk 9; F25).

Per plan §4.2 step 4 (line 1296-1303), the production multi-DM-trial
integration applies per-channel **integer-sample time shifts** on the
visibility tensor ``(n_fast_vis, NBASE, NCHAN)`` BEFORE the gridder,
once per coarse-DM trial. The gridder then sums across channels to
produce a per-trial gridded cube that is *coherently dedispersed at
that trial DM* — i.e. ``gridder.compute(stage1_shift(vis, plan,
dm_idx))`` is the per-trial coarse-DM image cube in sparse-COO form.

This is the **production** companion to the image-domain reference
primitive in :mod:`dsart.coarse_dm.dedisp` (which acts on
``[T_fast, NCHAN, N_grid, N_grid]`` per-channel image cubes after the
iFFT). Both implementations share the F24 native-samples → bin-shifts
convention (round once at apply time) + the Convention A delay
reference (chgroup-top channel = zero shift). They are equivalent up
to fp16 round-off (``coarse_dedisp(grid_per_ch(vis, ...)) ==
modulus²(iFFT(gridder(stage1_shift(vis, ...))))`` over the sparse
support of the gridder pattern).

Pipeline placement (per plan §4.2 streaming pipeline, lines 1283-1346
+ F25 reconciliation in M3_PLAN_FIXES.md)::

    fast-corr GEMM  →  cal/RFI weight  →  pol-sum (Stokes I)  →
    ┌─────────────────────────────────────┐
    │ stage-1 per-channel shift           │  ← this module (apply_stage1_shifts)
    │   for each coarse-DM trial          │
    └─────────────────────────────────────┘
                     │ (T_dedisp, NBASE, NCHAN) cfp32, per trial
                     ▼
              gridder.compute  →  (T_dedisp, N_filled) cfp32, per trial
                     │
                     ▼ stack across trials
              (N_DM, T_dedisp, N_filled) cfp32  →  static-sky
              subtract → quantize → stage-2 FIFO → transport-TX

Sign convention (Convention A — F24)
====================================

Per :mod:`dsart.coarse_dm.dm_plan`, the per-(chgroup, ch, dm) bin
shift is non-negative, identically zero at the chgroup's TOP channel,
and maximal at the BOT channel. The shift is applied as a FORWARD
read on the time axis::

    out[t', :, ch] = vis[t' + Δ_bins(g, ch, c), :, ch]

with ``t' ∈ [0, n_fast_vis - max_Δ_bins(g, c))``. The chgroup-top
channel (Δ_bins == 0) contributes ``vis[t', :, 0]`` unchanged; the
channel at the longest delay reads forward by max_Δ_bins.

A burst that arrives at the chgroup-top channel at fast-vis tile
``T0`` will appear in the stage-1 output at ``t' = T0`` for the
matching DM trial (and at incorrect / smeared times for other
trials).

Performance + memory
====================

The reference implementation here is a **per-DM, per-unique-shift
slice-copy** loop. For NCHAN_PER_CHGROUP=384 with low-DM trials,
many channels share the same bin shift (especially at coarse t_int);
grouping channels by shift coalesces the slice-copies to ~10×
fewer kernel launches than a naive per-channel loop.

Memory: per call, we allocate one ``(T_dedisp, NBASE, NCHAN)``
output tensor — ~9 MB at default ops (T_dedisp=512, NBASE=4656,
NCHAN=384, complex64). The Python loop over (channel-shift groups)
re-uses this same buffer; no transient ``(NCHAN,)`` indexing tensors.

This is a Python reference primitive — chunk-9 :class:`Stage1MultiDMCoarseDM`
calls it once per DM trial, allowing the orchestrator to free
intermediates between trials. A fused per-trial GPU kernel (cupy /
triton) is deferred to plan §4.2 line 1303 / chunk 10 hardening.

References
==========

* Plan §3.6.2 (DEDISP architecture).
* Plan §4.2 line 1296-1303 (stage-1 per-channel shifts).
* M3_PLAN_FIXES.md F24 (native-samples convention) + F25 (chunk-4
  reconciliation deferred to chunk 9 — this module is the chunk-9
  half of that reconciliation).
* :mod:`dsart.coarse_dm.dm_plan` — delay-table provider.
* :mod:`dsart.coarse_dm.dedisp` — image-domain reference primitive
  (algorithmically equivalent post-iFFT path).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch

from dsart.common.constants import N_CHGROUP, NBASE, NCHAN_PER_CHGROUP
from dsart.coarse_dm.dm_plan import DMPlan


__all__ = [
    "apply_stage1_shifts",
    "max_t_dedisp_for_plan",
]


_CFLOAT_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.complex32, torch.complex64, torch.complex128,
)


def max_t_dedisp_for_plan(
    n_fast_vis: int,
    plan: DMPlan,
    *,
    chgroup: int,
    dm_indices: np.ndarray | None = None,
) -> int:
    """``T_dedisp`` for a per-chgroup tile cube of length ``n_fast_vis``.

    ``T_dedisp = n_fast_vis - max_bins`` where ``max_bins`` is the
    largest bin shift across (channel, DM trial) for this chgroup at
    the plan's ``t_int_fast_us``. If ``dm_indices`` is given, the max
    is over only those trials.

    Returns ``0`` (clamped non-negative) when the cube is too short to
    accommodate any usable dedispersed bin; callers may treat 0 as
    "skip emit" (matches plan §3.6.2 stage-2 warm-up no-emit policy).
    """
    bin_shifts = plan.delay_bins_per_chgroup(chgroup)
    if dm_indices is not None:
        bin_shifts = bin_shifts[:, np.asarray(dm_indices, dtype=np.int64)]
    max_b = int(bin_shifts.max()) if bin_shifts.size else 0
    return max(0, int(n_fast_vis) - max_b)


def apply_stage1_shifts(
    vis: torch.Tensor,
    plan: DMPlan,
    *,
    chgroup: int,
    dm_idx: int,
    t_dedisp: int | None = None,
) -> torch.Tensor:
    """Apply per-channel forward integer-bin time shifts to a vis tensor.

    Parameters
    ----------
    vis : torch.Tensor
        Shape ``(n_fast_vis, NBASE, NCHAN)`` complex (cfp32 / cfp16).
        Stokes-I visibility tensor at fast-vis cadence (output of
        :func:`dsart.services.corr_fast_kernel.stokes_i_pol_sum`).
        Synthetic tests may use ``NCHAN < NCHAN_PER_CHGROUP`` and the
        delay table is sliced accordingly (channel ordering must
        match the chgroup's top-down ordering: ``ch=0`` is the top
        channel of the chgroup).
    plan : DMPlan
        Coarse-DM plan whose ``t_int_fast_us`` matches the vis tile
        cadence. ``plan.delay_bins_per_chgroup(chgroup)`` provides
        the per-(ch, dm) integer bin shifts.
    chgroup : int
        Chgroup index ``0..N_CHGROUP-1``.
    dm_idx : int
        Coarse-DM trial index ``0..plan.n_coarse-1``.
    t_dedisp : int, optional
        If given, output truncates to this many time bins (must be
        ``<= n_fast_vis - max_bin_shift_at(chgroup, dm_idx)``). When
        ``None`` (default), output uses the full available range
        ``n_fast_vis - max_bin_shift_at(chgroup, dm_idx)``. Callers
        that want a uniform output time axis across all DM trials
        pass the global ``min(T_dedisp_g_c)`` over their (g, c)
        sweep.

    Returns
    -------
    torch.Tensor
        Shape ``(t_dedisp, NBASE, NCHAN)`` same complex dtype + device
        as ``vis``. Channel ordering preserved.

    Raises
    ------
    TypeError
        If ``vis`` is not complex.
    ValueError
        If shape mismatches the expected ``(n_fast_vis, NBASE,
        NCHAN)`` form, if NCHAN exceeds NCHAN_PER_CHGROUP, or if
        ``t_dedisp`` exceeds the available range after applying the
        max bin shift.
    IndexError
        If ``chgroup`` or ``dm_idx`` are out of range.

    Notes
    -----
    - **Single-launch gather (real-time perf Phase 1)**: builds a
      ``(t_dedisp, 1, NCHAN_v)`` int64 time-index tensor whose entry
      ``[t, 0, c]`` is ``t + bin_shifts[c]``, expands the NBASE dim
      with stride 0 (no copy), and runs a single ``vis.gather(0, ...)``
      to materialise the shifted output. One CUDA-kernel launch per
      call, no Python loop over per-shift-group ``index_select`` /
      ``index_copy_`` ops. The previous per-shift-group form did
      O(unique_shifts) small ops and was launch-overhead-bound on a
      2080 Ti at the production op-point (96 ms/call observed vs
      ~10 ms/call expected at peak DRAM bandwidth — see
      ``bench/profile_fast_path_K1.py`` baseline).
    - **No re-allocation**: ``torch.gather`` allocates the output
      once. Total alloc traffic per call is
      ``t_dedisp * NBASE * NCHAN * sizeof(complex)`` — for the
      production op-point ~900 MB at NCHAN=48 (chan-summed).
    - **F24 pin**: bin shifts come from
      ``plan.delay_bins_per_chgroup(chgroup)`` which rounds the
      stored NATIVE-sample delay table to bins via
      ``round(delay_native / t_int_fast_native)``. This keeps the
      production stage-1 lossless w.r.t. the canonical DM plan even
      under non-default ``t_int_fast_native``.
    """
    if not 0 <= chgroup < N_CHGROUP:
        raise IndexError(
            f"chgroup={chgroup}, expected 0..{N_CHGROUP - 1}"
        )
    if not 0 <= dm_idx < plan.n_coarse:
        raise IndexError(
            f"dm_idx={dm_idx}, expected 0..{plan.n_coarse - 1}"
        )
    if vis.dtype not in _CFLOAT_DTYPES:
        raise TypeError(
            f"vis.dtype={vis.dtype}, expected complex (complex32/64/128)"
        )
    if vis.ndim != 3:
        raise ValueError(
            f"vis must be 3-D (n_fast_vis, NBASE, NCHAN); "
            f"got {vis.ndim}-D shape {tuple(vis.shape)}"
        )
    n_fv, n_base_v, n_chan_v = vis.shape
    if n_base_v != NBASE:
        raise ValueError(
            f"vis NBASE axis size {n_base_v} != NBASE={NBASE}"
        )
    if n_chan_v > NCHAN_PER_CHGROUP:
        raise ValueError(
            f"vis NCHAN axis size {n_chan_v} > NCHAN_PER_CHGROUP="
            f"{NCHAN_PER_CHGROUP}"
        )

    # ------------------------------------------------------------------
    # Resolve bin shifts for this (chgroup, dm) — sliced to vis NCHAN
    # so synthetic tests using a reduced channel count still work.
    # ------------------------------------------------------------------
    bin_shifts_full = plan.delay_bins_per_chgroup(chgroup)            # (NCHAN_full, N_coarse)
    bin_shifts = bin_shifts_full[:n_chan_v, dm_idx]                   # (NCHAN_v,) int64
    max_shift = int(bin_shifts.max())
    available = n_fv - max_shift
    if available <= 0:
        raise ValueError(
            f"n_fast_vis={n_fv} too small for max bin shift {max_shift} "
            f"(chgroup={chgroup}, dm_idx={dm_idx}); requires "
            f"n_fast_vis > max_shift"
        )
    if t_dedisp is None:
        t_dedisp = available
    else:
        t_dedisp = int(t_dedisp)
        if t_dedisp <= 0:
            raise ValueError(
                f"t_dedisp={t_dedisp}, must be > 0"
            )
        if t_dedisp > available:
            raise ValueError(
                f"t_dedisp={t_dedisp} > available={available} (n_fast_vis="
                f"{n_fv}, max_bin_shift={max_shift})"
            )

    # ------------------------------------------------------------------
    # Single-launch gather along the time axis. The index has shape
    # (t_dedisp, 1, NCHAN_v) and is broadcast (stride=0) along the NBASE
    # dim — torch.gather(0, ...) accepts arbitrary input/index strides
    # so no materialisation of a (t_dedisp, NBASE, NCHAN_v) int64 tensor
    # (~13 GB at the production op-point) is required.
    # ------------------------------------------------------------------
    bin_shifts_t = torch.as_tensor(
        bin_shifts, dtype=torch.int64, device=vis.device,
    )                                                                 # (NCHAN_v,)
    t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=vis.device)
    # (t_dedisp, 1, NCHAN_v) — broadcast across NBASE on the gather call
    t_idx = (t_arange[:, None, None] + bin_shifts_t[None, None, :])
    t_idx_b = t_idx.expand(t_dedisp, n_base_v, n_chan_v)
    out = vis.gather(0, t_idx_b)
    return out
