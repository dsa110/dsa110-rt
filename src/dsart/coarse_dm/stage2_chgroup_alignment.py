"""Corr-side stage-2 inter-chgroup time-alignment (Option A wire-in).

Production-grade replacement for the uniform-depth :class:`Stage2FIFO`
on the corr_fast TX path. Implements the :class:`Stage2FifoStage`
Protocol so it drops in via ``corr_fast_integration._Stage2FIFOAdapter``
'-style wrapping with no churn at the call site.

What it does
============

On every block, ``corr_fast`` pushes a dedispersed cube of shape::

    (N_DM, T_dedisp, N_filled)  cfp16 / complex64

This module's :class:`Stage2InterChgroupShiftFifo` takes that cube and
applies a PER-coarse-DM integer-sample roll along the time axis, so the
cube emerging at the TX boundary is already aligned to ``ν_bot_proc``.

The per-coarse-DM shifts are computed once at construction time from
:func:`dsart.coarse_dm.stage2_shifts.compute_stage2_shifts` keyed on the
local chgroup. The rolls run from 0 samples (chgroup-15) to ~9100
samples (chgroup-0 at the M7.4 250924mptq plan's largest coarse-DM).

The roll is implemented by a per-coarse-DM cube-history deque (one
slot per cube already pushed), with a sample-precise read at offset
``Δ_c``. Reads cross cube boundaries cleanly; the residual is exactly
0 samples relative to the math helper, NOT ±T_dedisp/2 as a coarse
cube-granular FIFO would give.

Search-side flag flip
=====================

When this FIFO is installed, the search side MUST set
``compute_time_shift_search(include_coarse_offset=False)`` (the default).
Otherwise both sides apply the per-chgroup delay and the search-side
ring buffer over-budgets while the burst lands at the wrong fdm.

Memory budget
=============

Per-coarse-DM ring depth = ``ceil(Δ_c / T_dedisp) + 1`` cubes' worth
of one coarse-DM slice (shape ``(T_dedisp, N_filled)``). At the M7.4
250924mptq op point with chgroup-0 (the worst case), the sum across
all 8 coarse DMs is about 900 MB cfp16. That sits just inside the
plan §3.6.2 855 MB stage-2 budget; chgroups 1..15 are smaller.

References
==========

* :mod:`dsart.coarse_dm.stage2_shifts` — math helper that computes
  the per-coarse-DM shifts this FIFO applies.
* :mod:`dsart.coarse_dm.stage2_per_dm_fifo` — slice-unit-agnostic
  per-coarse-DM FIFO container; this module uses a specialised
  cube-history layout instead because the wire-in needs to slice
  individual cubes (not whole slices) to get sample-exact rolling.
* :mod:`dsart.coarse_dm.stage2_fifo` — legacy uniform-depth FIFO.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import torch

from dsart.coarse_dm.stage2_shifts import (
    Stage2ShiftTable,
    compute_stage2_shifts,
    T_INT_CORR_US_DEFAULT,
)
from dsart.common.constants import N_CHGROUP


__all__ = [
    "Stage2InterChgroupShiftFifo",
]


class Stage2InterChgroupShiftFifo:
    """Per-coarse-DM time-roll FIFO satisfying the ``Stage2FifoStage``
    Protocol.

    Parameters
    ----------
    chgroup : int
        Local chgroup index (0..15). chgroup-15 is a degenerate
        identity (all shifts zero); the FIFO still works there and
        just emits the input cube unchanged.
    coarse_dm_pc_cm3 : np.ndarray
        ``(N_coarse,)`` coarse-DM trial values; identifies the
        per-coarse-DM shifts via
        :func:`compute_stage2_shifts`.
    t_dedisp : int
        Number of fast-vis samples per cube along the time axis
        (the cube's ``shape[1]``). Must match every incoming cube
        forever.
    t_int_corr_us : float, optional
        Corr-fast sample period in µs. Default
        :data:`T_INT_CORR_US_DEFAULT` = 262.144 µs.
    shift_table : Stage2ShiftTable, optional
        Pre-computed shift table. When provided the chgroup /
        coarse-DM args must agree with it (validated at construction).
        When omitted, the FIFO calls :func:`compute_stage2_shifts`
        for the caller.
    """

    __slots__ = (
        "_chgroup",
        "_t_dedisp",
        "_shifts",
        "_k_blocks",
        "_r_samples",
        "_max_back",
        "_rings",
        "_pushed",
        "_ref_shape",
        "_ref_dtype",
        "_ref_device",
        "_n_dm",
    )

    def __init__(
        self,
        *,
        chgroup: int,
        coarse_dm_pc_cm3: np.ndarray,
        t_dedisp: int,
        t_int_corr_us: float = T_INT_CORR_US_DEFAULT,
        shift_table: Optional[Stage2ShiftTable] = None,
    ) -> None:
        if t_dedisp < 1:
            raise ValueError(f"t_dedisp={t_dedisp} must be >= 1")
        if chgroup < 0 or chgroup >= N_CHGROUP:
            raise ValueError(
                f"chgroup={chgroup} out of range [0, {N_CHGROUP})"
            )
        if coarse_dm_pc_cm3.ndim != 1:
            raise ValueError(
                f"coarse_dm_pc_cm3 must be 1D, got "
                f"ndim={coarse_dm_pc_cm3.ndim}"
            )

        if shift_table is None:
            shift_table = compute_stage2_shifts(
                chgroup=chgroup,
                coarse_dm_pc_cm3=coarse_dm_pc_cm3,
                t_int_corr_us=t_int_corr_us,
            )
        else:
            if shift_table.chgroup != chgroup:
                raise ValueError(
                    f"shift_table.chgroup={shift_table.chgroup} != "
                    f"chgroup arg={chgroup}"
                )
            if not np.array_equal(
                shift_table.coarse_dm_pc_cm3, coarse_dm_pc_cm3
            ):
                raise ValueError(
                    "shift_table.coarse_dm_pc_cm3 != coarse_dm_pc_cm3 arg"
                )

        self._chgroup = int(chgroup)
        self._t_dedisp = int(t_dedisp)
        # Pre-compute the (k, r) decomposition once. shape (N_DM,).
        shifts = shift_table.shifts_samples.astype(np.int64, copy=False)
        self._shifts: np.ndarray = shifts
        self._k_blocks: np.ndarray = shifts // self._t_dedisp
        self._r_samples: np.ndarray = shifts - self._k_blocks * self._t_dedisp
        # Per-coarse-DM ring depth: max history needed is (k+1) cubes
        # because when r > 0 we read from ring[k+1]. For r == 0 we
        # only need ring[k]. We allocate (k+1)-deep for every DM to
        # cover both cases uniformly.
        self._max_back: np.ndarray = self._k_blocks + 1
        # Total ring slot count = max(k+1) across DMs, but each DM
        # only reads up to its own depth. Use a single deque per DM.
        self._rings: list[deque[torch.Tensor]] = [
            deque(maxlen=int(self._max_back[c]) + 1)
            for c in range(shifts.size)
        ]
        self._pushed: int = 0
        # Reference shape/dtype/device pinned on first push (matches
        # Stage2FIFO contract).
        self._ref_shape: Optional[tuple[int, ...]] = None
        self._ref_dtype: Optional[torch.dtype] = None
        self._ref_device: Optional[torch.device] = None
        self._n_dm: int = int(shifts.size)

    # ---- read-only state queries --------------------------------------

    @property
    def chgroup(self) -> int:
        return self._chgroup

    @property
    def t_dedisp(self) -> int:
        return self._t_dedisp

    @property
    def n_dm(self) -> int:
        return self._n_dm

    @property
    def shifts_samples(self) -> np.ndarray:
        """Per-coarse-DM integer-sample shifts (read-only view)."""
        return self._shifts

    @property
    def max_ring_depth_in_cubes(self) -> int:
        """Worst-case per-coarse-DM ring depth in whole cubes."""
        return int(self._max_back.max())

    def _required_depth(self, c: int) -> int:
        """Per-coarse-DM ring depth required to produce its rolled slice.

        Derivation (k = δ // T, r = δ - k*T):

            r == 0 : output[c] = ring[k]                → need ``len >= k + 1``
            r >  0 : output[c] = mix(ring[k+1], ring[k]) → need ``len >= k + 2``

        ``max_back[c] = k + 1`` so ``required = max_back[c]`` for r==0
        and ``max_back[c] + 1`` for r>0.
        """
        k_plus_1 = int(self._max_back[c])
        return k_plus_1 if int(self._r_samples[c]) == 0 else k_plus_1 + 1

    def warmed_up(self) -> bool:
        """``True`` once EVERY coarse-DM is warm (full-cube emit).

        Useful for monitoring + tests; the actual emit gate is
        :meth:`any_warmed_up` so the search side starts seeing data
        as soon as the smallest-shift coarse-DM is ready.
        """
        return all(
            len(self._rings[c]) >= self._required_depth(c)
            for c in range(self._n_dm)
        )

    def any_warmed_up(self) -> bool:
        """``True`` once at least ONE coarse-DM is warm.

        :meth:`push` emits a cube on every tick after this becomes
        ``True``. Un-warmed coarse-DMs get zero-valued slices in the
        emitted cube (their detector outputs will look like noise
        until they warm up). This matches the production semantic
        where corr_fast runs continuously: warmup is a startup cost,
        not a hot-loop concern.
        """
        return any(
            len(self._rings[c]) >= self._required_depth(c)
            for c in range(self._n_dm)
        )

    def per_dm_warmed(self) -> list[bool]:
        """Per-coarse-DM warm-up state. Useful for mon-points + tests."""
        return [
            len(self._rings[c]) >= self._required_depth(c)
            for c in range(self._n_dm)
        ]

    # ---- push hook ----------------------------------------------------

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        """Implement :class:`Stage2FifoStage`: ingest one cube, return
        the aligned cube for TX (or ``[]`` during warm-up).
        """
        if not isinstance(dedispersed, torch.Tensor):
            raise TypeError(
                "Stage2InterChgroupShiftFifo.push expected torch.Tensor; "
                f"got {type(dedispersed).__name__}"
            )
        if dedispersed.ndim != 3:
            raise ValueError(
                f"dedispersed.ndim={dedispersed.ndim}, expected 3 "
                f"(N_DM, T_dedisp, N_filled)"
            )
        n_dm_in, t_in, _ = dedispersed.shape
        if n_dm_in != self._n_dm:
            raise ValueError(
                f"dedispersed.shape[0]={n_dm_in} != n_dm={self._n_dm}"
            )
        if t_in != self._t_dedisp:
            raise ValueError(
                f"dedispersed.shape[1]={t_in} != t_dedisp={self._t_dedisp}"
            )

        # Pin shape/dtype/device on first push.
        if self._ref_shape is None:
            self._ref_shape = tuple(dedispersed.shape)
            self._ref_dtype = dedispersed.dtype
            self._ref_device = dedispersed.device
        else:
            if dedispersed.dtype != self._ref_dtype:
                raise ValueError(
                    f"dedispersed.dtype={dedispersed.dtype} != "
                    f"first-push dtype={self._ref_dtype}"
                )
            if dedispersed.device != self._ref_device:
                raise ValueError(
                    f"dedispersed.device={dedispersed.device} != "
                    f"first-push device={self._ref_device}"
                )

        # Push each coarse-DM slice into its own ring (newest first =
        # appendleft so ring[0] is the cube just pushed).
        for c in range(self._n_dm):
            self._rings[c].appendleft(dedispersed[c].contiguous())
        self._pushed += 1

        if not self.any_warmed_up():
            return []

        # Construct the aligned cube. Un-warmed coarse-DMs get zero
        # slices so the emitted tensor has a uniform shape. The
        # detector's local σ_k pass handles the zero slices as noise.
        out = torch.zeros_like(dedispersed)
        for c in range(self._n_dm):
            if len(self._rings[c]) < self._required_depth(c):
                continue
            k = int(self._k_blocks[c])
            r = int(self._r_samples[c])
            ring = self._rings[c]
            if r == 0:
                # Pure block-shift: output is the cube k blocks back.
                out[c] = ring[k]
            else:
                # Crossed-boundary slice: out[0:r] = ring[k+1][T-r:T],
                # out[r:T] = ring[k][0:T-r].
                t = self._t_dedisp
                out[c, 0:r] = ring[k + 1][t - r:t]
                out[c, r:t] = ring[k][0:t - r]
        return [out]
