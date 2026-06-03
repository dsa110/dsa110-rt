"""Fine-DM combiner (plan §3.6.3 + §4.4 lines 1471-1512).

Search-side stage that combines the 16 corr-node chgroup streams
(post-stage-1 + stage-2 dedispersion to ν_bot_proc at ``coarse_dm[c]``)
into per-fine-DM uv-grid head blocks. Per fine-DM trial ``f``, with
``c = fine_to_coarse[f]``, the combiner reads each chgroup ``g``'s
ring slot at:

    ring[g, c, t - Δt_samples_search[f, g]]   for t ∈ [t_head_start, t_head_start + head_block_samples)

and sums the per-chgroup uv-grids cell-wise into one ``[head_block_samples,
N_grid, N_grid] complex`` slab. The imager (``image/imager.py``) then
applies ``Re(iFFT2(...))`` + fftshift + edge mask to produce the cube
slab ``image_cube[t, f, l, m] real fp16``.

Two public entry points (Chunk 6a scope; the production fused
sparse-scatter-and-sum cupy kernel is Chunk 6b):

  * ``compute_time_shift_search(coarse_dm, fine_dm, fine_to_coarse,
        t_int_search_us=T_INT_SEARCH_US_DEFAULT,
        nu_chgroup_bot_GHz=...)`` —
    builds the ``[N_fine, N_chgroup] int32`` table per the §3.6.3
    formula. Pure-numpy; lives off the hot path.

  * ``combine_chgroups(per_chgroup_streams, time_shift_per_chgroup,
        t_window)`` —
    accepts pre-decoded dense per-chgroup uv-streams and gathers
    them with the integer-sample shifts. Useful for unit tests + the
    cube-injection bench's voltage-fixture cross-check (chunk 7) where
    the synthetic streams come from a numpy generator, not the M3
    sparse-COO transport.

The sparse-scatter-and-sum fused kernel (production hot path, plan
§4.4 line 1486) lands in Chunk 6b once `services/search_compute.py`
wires the receive-ring → combiner edge. M3 owns the receive-ring's
sparse-COO format; M5 consumes a pre-decoded dense stream here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from ..common.constants import (
    NU_CHGROUP_BOT_GHZ,
    NU_CHGROUP_TOP_GHZ,
    NU_BOT_PROC_GHZ,
    N_CHGROUP,
    T_INT_SEARCH_US_DEFAULT,
)
from ..common.dispersion import delta_tau_us

__all__ = [
    "TimeShiftSearchTable",
    "compute_time_shift_search",
    "combine_chgroups",
    "sparse_to_dense_grid",
]


# ---------------------------------------------------------------------------
# Per-fine-DM × per-chgroup time-shift table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeShiftSearchTable:
    """Per-fine-DM × per-chgroup integer-sample residual shift (plan §3.6.3).

    Args:
        shifts: ``[N_fine, N_chgroup] int32``. Element ``[f, g]`` is the
            number of ``t_int_search_us`` samples by which chgroup ``g``'s
            stream must be ADVANCED (read-later) to align to ν_bot_proc
            at ``fine_dm[f]`` after stage-1+stage-2 alignment to
            ``coarse_dm[c = fine_to_coarse[f]]``.
        fine_to_coarse: ``[N_fine] int64``; identifies the coarse-DM
            cell each fine-DM trial belongs to.
        t_int_search_us: search-side sample period (µs). Pinned to
            ``T_INT_SEARCH_US_DEFAULT = 524.288 µs`` at default ops.
    """

    shifts: np.ndarray
    fine_to_coarse: np.ndarray
    t_int_search_us: float
    #: True when the table was built with ``include_coarse_offset=True``
    #: AND the per-chgroup reference is chgroup-TOP (the M7.4 stage-2-
    #: absent escape hatch, corrected 2026-05-29 to match the Convention-A
    #: corr-side stage-1 output). In that mode chgroup-15's row is NOT
    #: identically zero: chgroup-15's TOP freq (1.32297 GHz) sits above
    #: ν_bot_proc (1.31128 GHz), so chgroup-15 carries a real (small,
    #: DM-dependent ~242-sample-at-DM3000) alignment shift. The §3.6.3
    #: "chgroup-15 row is zero" invariant only holds for the
    #: bottom-referenced (stage-2-present / include_coarse_offset=False)
    #: table, so it is skipped here.
    coarse_offset_baked: bool = False

    def __post_init__(self) -> None:
        if self.shifts.ndim != 2:
            raise ValueError(
                f"shifts must be 2D [N_fine, N_chgroup]; got shape {self.shifts.shape}"
            )
        if self.shifts.shape[1] != N_CHGROUP:
            raise ValueError(
                f"shifts.shape[1]={self.shifts.shape[1]} != N_CHGROUP={N_CHGROUP}"
            )
        if self.shifts.dtype != np.int32:
            raise TypeError(
                f"shifts.dtype={self.shifts.dtype}, expected int32"
            )
        if self.fine_to_coarse.ndim != 1:
            raise ValueError(
                f"fine_to_coarse must be 1D; got shape {self.fine_to_coarse.shape}"
            )
        if self.fine_to_coarse.shape[0] != self.shifts.shape[0]:
            raise ValueError(
                f"fine_to_coarse.shape[0]={self.fine_to_coarse.shape[0]} != "
                f"shifts.shape[0]={self.shifts.shape[0]} (N_fine)"
            )
        # As of 2026-06-03 BOTH the include_coarse_offset=True (M7.4
        # escape hatch) and include_coarse_offset=False (Option A
        # post-stage-2) paths reference each chgroup's TOP channel
        # (Convention A — matches the corr-side stage-1 output). With
        # chgroup-15's TOP at 1.32297 GHz vs ν_bot_proc 1.31128 GHz,
        # chgroup-15 carries a real (small, DM-dependent) shift. The
        # pre-2026-06-03 "[:,15] == 0" invariant is therefore retired;
        # we no longer enforce it.
        # Shifts are SIGNED int32 under the v2 DM-plan convention (per
        # user clarification 2026-05-18). With the even-K-around-coarse
        # GPU partitioning, each GPU owns K = N_fine/8 fine DMs that sit
        # SYMMETRICALLY around its assigned coarse_dm[i]. Fines BELOW the
        # coarse have δdm < 0 → negative shifts (read PAST data from the
        # rolling RX ring; naturally available, no rewind needed). Fines
        # ABOVE the coarse have δdm > 0 → positive shifts (read FUTURE
        # data; supplied by the corr-side one-sided rewind / dsaX_hella
        # convention). combine_chgroups() applies stream[t - shift] which
        # works for both signs as long as the rolling buffer has the
        # required past+future coverage; the validity gate at the cube
        # boundary is enforced upstream.
        if self.t_int_search_us <= 0:
            raise ValueError(
                f"t_int_search_us must be > 0; got {self.t_int_search_us!r}"
            )


def compute_time_shift_search(
    *,
    coarse_dm_pc_cm3: np.ndarray,
    fine_dm_pc_cm3: np.ndarray,
    fine_to_coarse: np.ndarray,
    t_int_search_us: float = T_INT_SEARCH_US_DEFAULT,
    nu_chgroup_bot_GHz: Optional[Tuple[float, ...]] = None,
    nu_chgroup_top_GHz: Optional[Tuple[float, ...]] = None,
    nu_bot_proc_GHz: float = NU_BOT_PROC_GHZ,
    include_coarse_offset: bool = False,
) -> TimeShiftSearchTable:
    """Build the per-fine-DM × per-chgroup integer-sample shift table.

    Per plan §3.6.3:

        Δt_samples_search[f, g] = rint(
            Δτ_us(ν_chgroup_bot_g, ν_bot_proc, δdm[f]) / t_int_search_us
        )

    where ``δdm[f] = fine_dm[f] - coarse_dm[fine_to_coarse[f]]``. The
    rounding rule is ``numpy.rint`` (half-to-even / banker's rounding)
    per the §3.6 rounding-direction lock.

    M7.4 stage-2-absent escape hatch (``include_coarse_offset=True``):
    when the upstream corr-side pipeline does NOT apply the per-coarse-DM
    stage-2 inter-chgroup alignment to ν_bot_proc (which is the current
    state of the production transport TX path — see ``coarse_dm/stage2_fifo.py``
    docstring re: "per-(g, c) depth-sizing happening at the integration
    site" + the lack of any apply-stage-2 surface in
    ``transport/``), the search-side shifts must absorb the FULL
    per-chgroup inter-band delay for ``fine_dm[f]`` (not just the
    δdm differential):

        Δt_samples_search[f, g] = rint(
            Δτ_us(ν_bot_proc, ν_chgroup_REF_g, fine_dm[f]) / t_int_search_us
        )

    Per-chgroup reference frequency convention (2026-06-03 unified):
    BOTH the ``include_coarse_offset=True`` escape hatch and the
    ``include_coarse_offset=False`` Option-A residual path reference
    each chgroup's TOP channel (``NU_CHGROUP_TOP_GHZ``). The corr-side
    coarse-DM stage-1 (``coarse_dm/dm_plan.py``, "Convention A") aligns
    each chgroup's channels to that chgroup's TOP, so stage-1's output
    represents the signal as it arrived at ``ν_chgroup_TOP[g]``. Stage-2
    (``coarse_dm/stage2_shifts.py``) likewise uses TOP, so when its
    per-(g, c) FIFO is enabled the resulting stream is referenced to
    ``ν_bot_proc``; the search-side fine-DM residual then only adds
    ``Δτ(ν_bot_proc, ν_chgroup_TOP[g], δdm)``. The cancellation between
    stage-2 (corr) and stage-3 (search) is bit-for-bit modulo ±0.5
    sample rint() noise — pinned by
    ``test_stage2_shifts.test_cross_stage_residual_against_baked_search_shifts``.

    Before the unification, the include_coarse_offset=False path used a
    BOT reference. Composed with the TOP-referenced stage-2 (or with
    the TOP-referenced include_coarse_offset=True path) that produced a
    constant -2.45% detected-vs-injected DM bias (the within-chgroup
    dispersion span). The 2026-05-29 fix corrected the
    include_coarse_offset=True path; the 2026-06-03 unification extends
    the TOP reference to the include_coarse_offset=False path so the
    corr-side stage-2 Option-A wire-in composes correctly.

    Because chgroup-15's TOP (1.32297 GHz) is above ν_bot_proc (1.31128
    GHz, the band bottom = chgroup-15's BOT), chgroup-15 carries a
    real (small, DM-dependent) shift in both paths and the §3.6.3
    "[:,15] == 0" invariant is retired. ``ProductionRxRing._t_stream =
    t_det + max(|shifts|)`` auto-sizes the history window.

    Sign convention (v2, 2026-05-18): shifts are SIGNED int32. δdm can
    be positive (fine above coarse → POSITIVE shift, advance/read-FUTURE)
    or negative (fine below coarse → NEGATIVE shift, retreat/read-PAST).
    Both signs are required under the v2 even-K-around-coarse partition.

    Args:
        coarse_dm_pc_cm3: ``[N_coarse] float64`` coarse-DM trial table.
        fine_dm_pc_cm3: ``[N_fine] float64`` fine-DM trial table.
        fine_to_coarse: ``[N_fine] int64`` mapping each fine trial to its
            coarse cell.
        t_int_search_us: sample period in µs (default 524.288 — D9
            hold-over from slow_corr; same value used here).
        nu_chgroup_bot_GHz: per-chgroup band-bottom frequencies; default
            uses ``common.constants.NU_CHGROUP_BOT_GHZ``. The 16-tuple
            ordering matches `chgroup_idx ∈ 0..15` with chgroup-0 the
            top of the band and chgroup-15 the bottom.
        nu_bot_proc_GHz: processed-band bottom frequency (= chgroup-15's
            band-bottom by construction).

    Returns:
        ``TimeShiftSearchTable`` with the validated shift array.
    """
    if coarse_dm_pc_cm3.ndim != 1 or fine_dm_pc_cm3.ndim != 1:
        raise ValueError("coarse_dm and fine_dm must both be 1D")
    if fine_to_coarse.ndim != 1:
        raise ValueError("fine_to_coarse must be 1D")
    if fine_to_coarse.shape[0] != fine_dm_pc_cm3.shape[0]:
        raise ValueError(
            f"fine_to_coarse.shape[0]={fine_to_coarse.shape[0]} != "
            f"len(fine_dm)={fine_dm_pc_cm3.shape[0]}"
        )
    if int(fine_to_coarse.max()) >= len(coarse_dm_pc_cm3):
        raise ValueError(
            f"fine_to_coarse.max()={int(fine_to_coarse.max())} >= "
            f"len(coarse_dm)={len(coarse_dm_pc_cm3)}"
        )

    chgroup_bot = (
        np.asarray(nu_chgroup_bot_GHz, dtype=np.float64)
        if nu_chgroup_bot_GHz is not None
        else np.asarray(NU_CHGROUP_BOT_GHZ, dtype=np.float64)
    )
    if chgroup_bot.shape != (N_CHGROUP,):
        raise ValueError(
            f"nu_chgroup_bot_GHz must be a {N_CHGROUP}-tuple; got "
            f"shape {chgroup_bot.shape}"
        )
    # DM-OFFSET FIX (2026-05-29): the escape-hatch (include_coarse_offset)
    # path references each chgroup's TOP channel (Convention-A corr-side
    # stage-1 output) instead of its bottom. See compute_time_shift_search
    # docstring. ``nu_chgroup_top_GHz`` lets callers pass the exact
    # chan-summed top-channel band-center (= corr DMPlan chgroup_freqs[:,0])
    # for a zero residual; default NU_CHGROUP_TOP_GHZ (band-top) leaves a
    # negligible +0.023% (≪ one fine-DM step).
    chgroup_top = (
        np.asarray(nu_chgroup_top_GHz, dtype=np.float64)
        if nu_chgroup_top_GHz is not None
        else np.asarray(NU_CHGROUP_TOP_GHZ, dtype=np.float64)
    )
    if chgroup_top.shape != (N_CHGROUP,):
        raise ValueError(
            f"nu_chgroup_top_GHz must be a {N_CHGROUP}-tuple; got "
            f"shape {chgroup_top.shape}"
        )

    # Per-chgroup reference: TOP for BOTH paths as of 2026-06-03. Stage-1
    # output is TOP-referenced (Convention A); ``include_coarse_offset=True``
    # (M7.4 escape hatch) absorbs the full ν_TOP_g→ν_bot_proc delay on
    # the search side; ``include_coarse_offset=False`` (Option A) lets
    # the corr-side stage-2 FIFO apply the same delay (referenced to
    # TOP) and the search side then only applies the δdm-residual.
    # Either way, using a TOP reference on the search side is required
    # for the cross-stage cancellation to be bit-for-bit (mod ±0.5
    # rint() noise).
    del chgroup_bot  # retained for back-compat argument; no longer used
    nu_chgroup_ref = chgroup_top

    n_fine = fine_dm_pc_cm3.shape[0]
    shifts = np.zeros((n_fine, N_CHGROUP), dtype=np.int32)
    for f in range(n_fine):
        c = int(fine_to_coarse[f])
        if include_coarse_offset:
            ddm = float(fine_dm_pc_cm3[f])
        else:
            ddm = float(fine_dm_pc_cm3[f] - coarse_dm_pc_cm3[c])
        for g in range(N_CHGROUP):
            d_us = delta_tau_us(
                float(nu_bot_proc_GHz), float(nu_chgroup_ref[g]), ddm
            )
            shifts[f, g] = int(np.rint(d_us / t_int_search_us))
    # v2 (2026-05-18): shifts are SIGNED — δdm < 0 (fine below coarse)
    # yields negative shifts (read PAST data from the rolling RX ring).
    # See TimeShiftSearchTable docstring for the full convention.
    return TimeShiftSearchTable(
        shifts=shifts,
        fine_to_coarse=fine_to_coarse.astype(np.int64, copy=False),
        t_int_search_us=float(t_int_search_us),
        coarse_offset_baked=bool(include_coarse_offset),
    )


# ---------------------------------------------------------------------------
# Sparse-COO → dense uv-grid scatter (test/utility-grade; the production
# hot path's fused cupy kernel lives in Chunk 6b)
# ---------------------------------------------------------------------------


def sparse_to_dense_grid(
    *,
    linear_indices: np.ndarray,
    values: np.ndarray,
    n_grid: int,
) -> np.ndarray:
    """Scatter a flat-COO sparse vector into a dense ``[N_grid, N_grid]``
    complex grid.

    Args:
        linear_indices: ``[N_filled] int64`` linear indices into the
            ``N_grid × N_grid`` grid (row-major; ``v_pix * N_grid + u_pix``).
        values: ``[N_filled] complex64/complex128`` cell values.
        n_grid: grid side length.

    Returns:
        ``[N_grid, N_grid] complex64`` grid; cells absent from the
        sparse vector are zero. Duplicate ``linear_indices`` are summed
        (numpy ``add.at`` semantics).
    """
    if linear_indices.ndim != 1 or values.ndim != 1:
        raise ValueError(
            "linear_indices and values must both be 1D"
        )
    if linear_indices.shape != values.shape:
        raise ValueError(
            f"linear_indices.shape={linear_indices.shape} != "
            f"values.shape={values.shape}"
        )
    if not np.issubdtype(linear_indices.dtype, np.integer):
        raise TypeError(
            f"linear_indices.dtype={linear_indices.dtype}, expected integer"
        )
    if int(linear_indices.max(initial=0)) >= n_grid * n_grid:
        raise ValueError(
            f"linear_indices.max()={int(linear_indices.max())} >= "
            f"n_grid²={n_grid * n_grid}"
        )
    grid_flat = np.zeros(n_grid * n_grid, dtype=np.complex64)
    np.add.at(grid_flat, linear_indices, values.astype(np.complex64, copy=False))
    return grid_flat.reshape(n_grid, n_grid)


# ---------------------------------------------------------------------------
# Per-fine-DM combiner (across the 16 chgroups, time-shifted)
# ---------------------------------------------------------------------------


def combine_chgroups(
    *,
    per_chgroup_streams: Mapping[int, np.ndarray],
    time_shift_per_chgroup: np.ndarray,
    t_window: Tuple[int, int],
    n_grid: Optional[int] = None,
) -> np.ndarray:
    """Combine pre-decoded per-chgroup uv-streams into a single fine-DM
    head-block uv-slab.

    Per plan §4.4 lines 1486-1493: for each cube-time ``t`` in
    ``[t_head_start, t_head_start + head_block_samples)``, this reads
    chgroup ``g``'s dense stream at ``stream[g][t - shift[g]]`` (the
    "advance" shift; see §3.6.3 sign convention) and sums across the 16
    chgroups into a single complex grid.

    The per-chgroup stream is expected to be a dense
    ``[T_stream, N_grid, N_grid] complex`` array (the chunk-6b fused
    cupy kernel will replace this with a sparse-scatter pass over the
    pinned-host receive ring; the dense-stream form is the test path
    + the chunk-7 voltage-fixture cross-check path).

    Args:
        per_chgroup_streams: dict mapping ``chgroup_idx ∈ 0..15`` to a
            ``[T_stream, N_grid, N_grid] complex`` ndarray.
        time_shift_per_chgroup: ``[N_chgroup] int32`` row of the
            ``TimeShiftSearchTable``; usually ``table.shifts[f, :]``
            for some fine-DM trial ``f``.
        t_window: ``(t_lo, t_hi)`` (right-exclusive) of cube-time
            samples to combine; the per-chgroup stream is read at
            ``[t - shift[g] for t in range(t_lo, t_hi)]``. Out-of-range
            reads are zero-filled (the combiner DOES NOT raise; the
            chunk-2 cube-validity gate is responsible for rejecting
            cubes whose receive-ring slots haven't filled yet).
        n_grid: grid side length; defaults to inferring from the first
            stream's shape.

    Returns:
        ``[head_block_samples, N_grid, N_grid] complex64`` slab
        (``head_block_samples == t_hi - t_lo``).

    Raises:
        ValueError if ``time_shift_per_chgroup`` is not the right
        shape, or if the streams' shapes are inconsistent.
    """
    if time_shift_per_chgroup.shape != (N_CHGROUP,):
        raise ValueError(
            f"time_shift_per_chgroup must have shape ({N_CHGROUP},); "
            f"got {time_shift_per_chgroup.shape}"
        )
    if not np.issubdtype(time_shift_per_chgroup.dtype, np.integer):
        raise TypeError(
            f"time_shift_per_chgroup.dtype={time_shift_per_chgroup.dtype}, "
            f"expected integer"
        )
    t_lo, t_hi = int(t_window[0]), int(t_window[1])
    if t_hi <= t_lo:
        raise ValueError(f"t_window={t_window!r} has non-positive width")

    # Infer n_grid from the first available stream if not supplied.
    if n_grid is None:
        for stream in per_chgroup_streams.values():
            if stream.ndim != 3 or stream.shape[1] != stream.shape[2]:
                continue
            n_grid = int(stream.shape[1])
            break
    if n_grid is None:
        raise ValueError(
            "could not infer n_grid; pass `n_grid=` explicitly when "
            "per_chgroup_streams is empty or non-square"
        )

    head_block_samples = t_hi - t_lo
    out = np.zeros(
        (head_block_samples, n_grid, n_grid), dtype=np.complex64
    )

    for g in range(N_CHGROUP):
        stream = per_chgroup_streams.get(g)
        if stream is None:
            # Missing chgroup → zero contribution (cube-validity gate
            # at the caller's level enforces presence; combiner is
            # tolerant so unit tests can probe edge cases).
            continue
        if stream.ndim != 3:
            raise ValueError(
                f"per_chgroup_streams[{g}].ndim={stream.ndim}, expected 3 "
                f"[T_stream, N_grid, N_grid]"
            )
        if stream.shape[1] != n_grid or stream.shape[2] != n_grid:
            raise ValueError(
                f"per_chgroup_streams[{g}].shape={stream.shape} != "
                f"(_, {n_grid}, {n_grid})"
            )
        shift = int(time_shift_per_chgroup[g])
        T_stream = stream.shape[0]
        # Compute valid (cube-time) window: stream is read at
        # ``[t - shift for t in range(t_lo, t_hi)]``; require
        # 0 ≤ t - shift < T_stream.
        t_in_lo = max(t_lo, shift)
        t_in_hi = min(t_hi, T_stream + shift)
        if t_in_hi <= t_in_lo:
            continue
        out_slice = slice(t_in_lo - t_lo, t_in_hi - t_lo)
        stream_slice = slice(t_in_lo - shift, t_in_hi - shift)
        out[out_slice] += stream[stream_slice].astype(np.complex64, copy=False)

    return out
