"""Per-coarse-DM stage-2 timing FIFO (Option A foundation).

A bank of independent FIFOs — one per coarse-DM trial — each sized to
its own target delay. The depth is expressed in *units of the slice
that is being pushed* (the caller decides whether that is one
fast-vis sample, one cube, or any other unit).

Quantisation note (read before wiring this into a TX path)
==========================================================

The per-(chgroup, coarse-DM) stage-2 delays computed by
:func:`dsart.coarse_dm.stage2_shifts.compute_stage2_shifts` are in
units of **single fast-vis samples** (262.144 µs at the M7.4 op
point). At chgroup-0 / coarse_dm[7]=4452 pc/cm³ that is ~9115 samples
≈ 2.4 s of buffered history.

If the TX wire-in pushes whole *cubes* (e.g., T_dedisp=512 samples per
cube), the cube-granular FIFO depth is ``ceil(delay_samples /
T_dedisp)`` and the residual per cube can be ±T_dedisp/2 samples,
i.e. ±256 samples in the search cadence at the default op point. That
residual is too coarse for the search-side ring buffer to absorb in
``compute_time_shift_search`` with ``include_coarse_offset=False`` —
it would re-inflate ``_t_stream`` by exactly the amount we are trying
to reclaim by moving stage-2 to corr-side.

For the full production-grade Option A the TX path needs a
**sample-granular** per-(g, c) ring buffer (this class supports that
if the caller pushes one fast-vis sample at a time, but a dedicated
sample-ring implementation is preferable for cache + bandwidth).
The cube-granular path is OK for benches or for chgroup-15 (which is
identically zero-delay anyway).

This class is intentionally slice-unit-agnostic so both the
sample-granular and the cube-granular wire-ins can share the same
container. The next change to ship Option A end-to-end should
either:
  (a) push individual fast-vis samples (sample-granular, accurate),
      or
  (b) push T-sample blocks and absorb the residual in stage-3 with a
      sub-T offset table (cube-granular + soft correction).
Both paths are valid. The choice is a downstream design decision
for the transport-TX integrator; this container does not assume one.

Plumbing
========

How the delay is implemented
============================

Each coarse-DM slice of an incoming cube is enqueued into its own
FIFO of depth ``D_c`` (where ``D_c >= 1``). Every push of a fresh
slice into FIFO[c] returns the slice that is now ``D_c`` time-steps
behind, which is the slice the transport must emit on this tick.

Concretely::

    tick 0 push slice for c, D_c=3        FIFO[c] = [s0]           → no emit
    tick 1 push slice for c, D_c=3        FIFO[c] = [s0, s1]       → no emit
    tick 2 push slice for c, D_c=3        FIFO[c] = [s0, s1, s2]   → emit s0
    tick 3 push slice for c, D_c=3        FIFO[c] = [s1, s2, s3]   → emit s1
    ...

In steady state every push emits exactly one slice. During warm-up
(first ``D_c - 1`` pushes) no slice is emitted — the search side
accounts for this with its standard burn-in.

For ``D_c == 0`` (e.g., chgroup-15 against ν_bot_proc) the FIFO is a
pass-through: push returns the input unchanged.

For non-uniform ``D_c`` across coarse-DMs, this bank holds different
amounts of state per coarse-DM. With the M7.4 250924mptq DM plan
(``coarse_dm = [258, 387, 581, 873, 1312, 1971, 2962, 4452]`` pc/cm³)
and chgroup-0, the depths run roughly ``[2, 3, 5, 7, 11, 17, 25, 38]``
fast-vis samples — small enough that GPU memory is dominated by the
cube tensors themselves, not the FIFO chain.

Status
======

* Math helper (:mod:`dsart.coarse_dm.stage2_shifts`): SHIPPED + tested.
* This FIFO container: SHIPPED + tested below.
* Wiring into ``corr_fast_integration`` TX path: NOT YET. Default code
  paths still use :class:`Stage2FIFO` (uniform depth) and the M7.4
  search-side baked-in shifts (``include_coarse_offset=True``). The
  next change will:

  1. swap the uniform :class:`Stage2FIFO` for this per-coarse-DM
     :class:`PerCoarseDmStage2FIFO`,
  2. flip ``include_coarse_offset`` back to ``False`` in the
     ``compute_time_shift_search`` call site, and
  3. shrink the search-side ``ProductionRxRing._t_stream`` window to
     the stage-3-only minimum.
"""
from __future__ import annotations

from collections import deque
from typing import Iterator, Optional, Sequence

import numpy as np
import torch


__all__ = [
    "PerCoarseDmStage2FIFO",
]


class PerCoarseDmStage2FIFO:
    """Bank of per-coarse-DM FIFOs with individual target depths.

    Parameters
    ----------
    depths_per_coarse_dm : sequence of int
        ``(N_coarse,)`` non-negative depths. ``depths[c] == 0`` means
        the coarse-DM-``c`` channel is a pass-through (no delay).
        Typically computed from
        :func:`dsart.coarse_dm.stage2_shifts.compute_stage2_shifts`.

    Notes
    -----
    The container is dtype/shape-agnostic per coarse-DM slice — each
    slice can be any :class:`torch.Tensor`. The first push for a given
    coarse-DM pins ``(shape, dtype, device)``; subsequent pushes for
    that coarse-DM are checked against the pin to catch producer bugs
    (matches :class:`Stage2FIFO`'s contract).

    No CUDA stream interaction: the FIFO only stores tensor references,
    it does not allocate or copy. The caller is responsible for stream
    sync if the tensors carry device-side write dependencies.
    """

    __slots__ = (
        "_depths",
        "_bufs",
        "_ref_shape",
        "_ref_dtype",
        "_ref_device",
    )

    def __init__(self, depths_per_coarse_dm: Sequence[int]) -> None:
        depths = [int(d) for d in depths_per_coarse_dm]
        if not depths:
            raise ValueError("depths_per_coarse_dm must be non-empty")
        for c, d in enumerate(depths):
            if d < 0:
                raise ValueError(
                    f"depths_per_coarse_dm[{c}]={d} < 0; must be >= 0"
                )
        self._depths: tuple[int, ...] = tuple(depths)
        # One deque per coarse-DM. Pass-through coarse-DMs (depth 0)
        # get a None placeholder so the bookkeeping is trivial.
        self._bufs: list[Optional[deque[torch.Tensor]]] = [
            deque(maxlen=max(d, 1)) if d > 0 else None for d in depths
        ]
        self._ref_shape: list[Optional[tuple[int, ...]]] = [None] * len(depths)
        self._ref_dtype: list[Optional[torch.dtype]] = [None] * len(depths)
        self._ref_device: list[Optional[torch.device]] = [None] * len(depths)

    @property
    def n_coarse(self) -> int:
        return len(self._depths)

    @property
    def depths(self) -> tuple[int, ...]:
        return self._depths

    def depth(self, coarse_dm_index: int) -> int:
        return int(self._depths[coarse_dm_index])

    def __len__(self) -> int:
        """Total number of slices currently buffered across all coarse-DMs."""
        return sum(
            (0 if buf is None else len(buf))
            for buf in self._bufs
        )

    def occupancy(self, coarse_dm_index: int) -> int:
        buf = self._bufs[coarse_dm_index]
        return 0 if buf is None else len(buf)

    def push(
        self,
        coarse_dm_index: int,
        slice_: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Push one coarse-DM slice; return the slice the TX should now emit.

        Parameters
        ----------
        coarse_dm_index : int
            Which coarse-DM the slice belongs to. Must be in
            ``[0, n_coarse)``.
        slice_ : torch.Tensor
            The cube slice for this coarse-DM. The first push for this
            coarse-DM pins the ``(shape, dtype, device)`` triplet; every
            subsequent push is checked against it.

        Returns
        -------
        torch.Tensor or None
            * If ``depth == 0`` (pass-through): the input tensor is
              returned as-is.
            * If ``depth > 0`` and the FIFO is NOT yet full: the slice
              is enqueued and ``None`` is returned (warm-up).
            * If ``depth > 0`` and the FIFO is full: the slice is
              enqueued at the tail and the slice currently at the head
              is returned (the one that is now ``depth`` ticks old).

        Raises
        ------
        TypeError
            If ``slice_`` is not a :class:`torch.Tensor`.
        ValueError
            On out-of-range ``coarse_dm_index`` or
            ``(shape, dtype, device)`` mismatch.
        """
        if not isinstance(slice_, torch.Tensor):
            raise TypeError(
                "PerCoarseDmStage2FIFO.push expected torch.Tensor; got "
                f"{type(slice_).__name__}"
            )
        if coarse_dm_index < 0 or coarse_dm_index >= self.n_coarse:
            raise ValueError(
                f"coarse_dm_index={coarse_dm_index} out of range "
                f"[0, {self.n_coarse})"
            )
        # Per-coarse-DM shape/dtype/device pin (matches Stage2FIFO).
        ref_shape = self._ref_shape[coarse_dm_index]
        if ref_shape is None:
            self._ref_shape[coarse_dm_index] = tuple(slice_.shape)
            self._ref_dtype[coarse_dm_index] = slice_.dtype
            self._ref_device[coarse_dm_index] = slice_.device
        else:
            if tuple(slice_.shape) != ref_shape:
                raise ValueError(
                    f"coarse_dm={coarse_dm_index}: slice shape "
                    f"{tuple(slice_.shape)} != first-push shape "
                    f"{ref_shape}"
                )
            if slice_.dtype != self._ref_dtype[coarse_dm_index]:
                raise ValueError(
                    f"coarse_dm={coarse_dm_index}: slice dtype "
                    f"{slice_.dtype} != first-push dtype "
                    f"{self._ref_dtype[coarse_dm_index]}"
                )
            if slice_.device != self._ref_device[coarse_dm_index]:
                raise ValueError(
                    f"coarse_dm={coarse_dm_index}: slice device "
                    f"{slice_.device} != first-push device "
                    f"{self._ref_device[coarse_dm_index]}"
                )

        depth = self._depths[coarse_dm_index]
        buf = self._bufs[coarse_dm_index]
        if depth == 0 or buf is None:
            return slice_

        if len(buf) < depth:
            buf.append(slice_)
            return None
        emitted = buf.popleft()
        buf.append(slice_)
        return emitted

    def flush(self, coarse_dm_index: int) -> list[torch.Tensor]:
        """Drain all remaining slices for one coarse-DM, oldest-first.

        Useful on shutdown when the operator wants the trailing warm-up
        slices off-chip rather than dropped on the floor. After a flush
        the coarse-DM is empty; subsequent pushes warm up again.
        """
        buf = self._bufs[coarse_dm_index]
        if buf is None:
            return []
        out = list(buf)
        buf.clear()
        return out

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        """Iterate ``(coarse_dm_index, occupancy, depth)`` triples.

        Convenience for monitoring + tests; does NOT expose the
        underlying tensors.
        """
        for c in range(self.n_coarse):
            yield (c, self.occupancy(c), self._depths[c])
