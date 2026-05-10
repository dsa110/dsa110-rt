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
        if not np.all(self.shifts[:, N_CHGROUP - 1] == 0):
            raise ValueError(
                "time_shift_search[:, 15] must be 0 for every fine-DM row by "
                "§3.6.3 sign convention (chgroup 15 is the reference; "
                "ν_chgroup_bot[15] == ν_bot_proc)"
            )
        if int(np.min(self.shifts)) < 0:
            raise ValueError(
                "time_shift_search must be non-negative for δdm ≥ 0 "
                "(plan §3.6.3 sign convention)"
            )
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
    nu_bot_proc_GHz: float = NU_BOT_PROC_GHZ,
) -> TimeShiftSearchTable:
    """Build the per-fine-DM × per-chgroup integer-sample shift table.

    Per plan §3.6.3:

        Δt_samples_search[f, g] = rint(
            Δτ_us(ν_chgroup_bot_g, ν_bot_proc, δdm[f]) / t_int_search_us
        )

    where ``δdm[f] = fine_dm[f] - coarse_dm[fine_to_coarse[f]]``. The
    rounding rule is ``numpy.rint`` (half-to-even / banker's rounding)
    per the §3.6 rounding-direction lock.

    Sign convention: shifts are non-negative for ``δdm ≥ 0``. The
    chgroup-15 row is identically zero (chgroup-15's band-bottom IS
    ``ν_bot_proc`` by construction; see ``common/constants.py::NU_CHGROUP_BOT_GHZ``).

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

    n_fine = fine_dm_pc_cm3.shape[0]
    shifts = np.zeros((n_fine, N_CHGROUP), dtype=np.int32)
    for f in range(n_fine):
        c = int(fine_to_coarse[f])
        ddm = float(fine_dm_pc_cm3[f] - coarse_dm_pc_cm3[c])
        for g in range(N_CHGROUP):
            # Argument order: (nu_low, nu_high). For g < 15 we have
            # chgroup_bot[g] > nu_bot_proc (chgroup-0 is band-top, -15
            # is band-bottom = nu_bot_proc), so nu_bot_proc is the LOW
            # frequency. δdm > 0 then yields Δτ > 0; chgroup g leads
            # ν_bot_proc by Δτ and must be advanced by ``+rint(Δτ /
            # t_int)`` samples to align.
            d_us = delta_tau_us(
                float(nu_bot_proc_GHz), float(chgroup_bot[g]), ddm
            )
            shifts[f, g] = int(np.rint(d_us / t_int_search_us))
    # Force chgroup-15 row to zero (paranoia: floating-point noise in
    # delta_tau at the construction frequency could yield ±1-sample
    # noise away from zero on some platforms).
    shifts[:, N_CHGROUP - 1] = 0
    if int(shifts.min()) < 0:
        raise ValueError(
            "computed shifts include negative values for δdm ≥ 0; "
            "check coarse_dm / fine_dm / fine_to_coarse alignment"
        )
    return TimeShiftSearchTable(
        shifts=shifts,
        fine_to_coarse=fine_to_coarse.astype(np.int64, copy=False),
        t_int_search_us=float(t_int_search_us),
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
