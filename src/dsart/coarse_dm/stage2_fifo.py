"""Stage-2 coarse-DM cube FIFO (M3 chunk 3b; plan §3.6.2 + §4.2 step 8b).

A bounded ring buffer holding the last ``K`` coarse-DM image cubes
(or sparse-COO value vectors, depending on the integration site)
between the corr-side dedisperser and the corr-side transport-TX.

Two consumer surfaces share this contract:

1. **Corr-side transport-TX** (chunk 4 wires this): on every push,
   the latest cube is forwarded to the transport layer. The FIFO's
   role on the corr side is mainly the *timing* buffer for
   cross-chgroup alignment to ν_bot_proc — see plan §3.6.2 stage-2
   description and the §4.2 streaming pipeline lines 1322-1346 for
   the full per-(chgroup, coarse_dm) FIFO depth derivation
   (``Δt_samples_corr_stage2[g, c] / t_int_factor``). The
   chunk-3b implementation here is the *container* — a uniform-depth
   ring of cubes, with the per-(g, c) depth-sizing happening at the
   integration site.

2. **Search-side cross-coarse-DM detector context** (M5 read-only
   consumer): on every push, the last K cubes form the rolling
   detector context window. The detector pass reads `as_list()` to
   obtain the K cubes in chronological (oldest-first) order,
   independent of how many push-and-evict cycles have happened.

Both consumers see the same simple API:

* :meth:`Stage2FIFO.push(cube) -> evicted` — push one cube; returns
  the evicted (oldest) cube if the ring was full, else ``None``.
* :meth:`Stage2FIFO.peek_latest() -> Optional[Tensor]` — the most
  recently pushed cube (None if empty).
* :meth:`Stage2FIFO.as_list() -> list[Tensor]` — current contents in
  oldest-first order; length 0..K.
* :meth:`Stage2FIFO.full() -> bool`, :meth:`empty() -> bool`,
  :meth:`__len__() -> int`.

Type contract
=============

Cubes are :class:`torch.Tensor` of any shape and dtype, but every
push must match the **first** push's ``shape``, ``dtype``, and
``device`` — mismatches raise :class:`ValueError`. (This catches the
common bug where chunk 4 forgets to re-quantise after a config swap;
the FIFO refusing the push surfaces the bug at the producer rather
than letting cross-cube heterogeneity propagate to TX.) Empty FIFOs
accept any first push.

Memory
======

A :class:`Stage2FIFO` of depth ``K`` holding ``(T_dedisp, N_DM,
N_grid, N_grid) cfp16`` cubes uses ``K × T × N_DM × N_grid² × 4 B``
GPU/CPU memory. At default ops (K=4, T=512, N_DM=24, N_grid=256,
cfp16) this is ≈ 100 MB — well below the per-corr-node 855 MB
stage-2 FIFO budget pinned by plan §3.6.2 / §4.2 streaming pipeline.

References
==========

* Plan §3.6.2 lines ~726-770 — stage-2 FIFO depth + cross-chgroup
  alignment role.
* Plan §4.2 lines ~1322-1346 — streaming pipeline placement +
  per-corr-node memory peak.
* :data:`dsart.common.constants.COARSE_DM_FIFO_DEPTH_DEFAULT` —
  default ``K`` for the chunk-3b smoke FIFO.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import torch

from dsart.common.constants import COARSE_DM_FIFO_DEPTH_DEFAULT


__all__ = [
    "Stage2FIFO",
]


class Stage2FIFO:
    """Bounded FIFO of coarse-DM cubes; oldest-evicting.

    Parameters
    ----------
    depth : int
        Maximum number of cubes the ring can hold. Default
        :data:`dsart.common.constants.COARSE_DM_FIFO_DEPTH_DEFAULT`.
        Must be ≥ 1 (a depth-0 FIFO would refuse every push and be
        useless; we error early rather than silently no-op).

    Attributes
    ----------
    depth : int
        Capacity (read-only after construction).
    """

    __slots__ = ("_depth", "_buf", "_ref_shape", "_ref_dtype", "_ref_device")

    def __init__(self, depth: int = COARSE_DM_FIFO_DEPTH_DEFAULT) -> None:
        if depth < 1:
            raise ValueError(f"depth={depth}, expected ≥ 1")
        self._depth = int(depth)
        self._buf: deque[torch.Tensor] = deque(maxlen=self._depth)
        # First-push reference shape/dtype/device — set on first push,
        # asserted on every subsequent push.
        self._ref_shape: tuple[int, ...] | None = None
        self._ref_dtype: torch.dtype | None = None
        self._ref_device: torch.device | None = None

    # ------------------------------------------------------------------
    # Capacity / state queries
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Maximum number of cubes the ring can hold."""
        return self._depth

    def __len__(self) -> int:
        return len(self._buf)

    def empty(self) -> bool:
        """True iff no cubes have been pushed (or all have been popped)."""
        return len(self._buf) == 0

    def full(self) -> bool:
        """True iff the next push will evict the oldest cube."""
        return len(self._buf) == self._depth

    # ------------------------------------------------------------------
    # Push / peek / iterate
    # ------------------------------------------------------------------

    def push(self, cube: torch.Tensor) -> torch.Tensor | None:
        """Push one cube; return the evicted oldest cube if the ring was full.

        Parameters
        ----------
        cube : torch.Tensor
            Any shape / dtype / device, but every push after the first
            must match the **first** push's ``(shape, dtype, device)``
            triplet. The FIFO does not copy the input tensor — callers
            that intend to mutate the cube after pushing must clone
            first.

        Returns
        -------
        torch.Tensor | None
            ``None`` if the ring had spare capacity (the cube was
            inserted without eviction). The evicted oldest tensor if
            the ring was full pre-push (the new cube is now at the
            tail; the returned tensor is what was at the head).
            **The caller owns the returned tensor**; the FIFO no
            longer holds a reference to it.

        Raises
        ------
        TypeError
            If ``cube`` is not a :class:`torch.Tensor`.
        ValueError
            If the cube's ``(shape, dtype, device)`` triplet differs
            from the first-pushed cube's. The error message names
            which axis differs to make debugging the producer easy.
        """
        if not isinstance(cube, torch.Tensor):
            raise TypeError(
                f"Stage2FIFO.push expected torch.Tensor; got "
                f"{type(cube).__name__}"
            )
        if self._ref_shape is None:
            self._ref_shape = tuple(cube.shape)
            self._ref_dtype = cube.dtype
            self._ref_device = cube.device
        else:
            if tuple(cube.shape) != self._ref_shape:
                raise ValueError(
                    f"Stage2FIFO.push cube shape={tuple(cube.shape)} "
                    f"!= first-push shape={self._ref_shape}"
                )
            if cube.dtype != self._ref_dtype:
                raise ValueError(
                    f"Stage2FIFO.push cube dtype={cube.dtype} != "
                    f"first-push dtype={self._ref_dtype}"
                )
            if cube.device != self._ref_device:
                raise ValueError(
                    f"Stage2FIFO.push cube device={cube.device} != "
                    f"first-push device={self._ref_device}"
                )

        evicted: torch.Tensor | None = None
        if len(self._buf) == self._depth:
            # deque.append with maxlen would silently drop; we want to
            # surface the evicted cube so the caller can forward it.
            evicted = self._buf.popleft()
        self._buf.append(cube)
        return evicted

    def peek_latest(self) -> torch.Tensor | None:
        """Most recently pushed cube (the tail), or ``None`` if empty.

        For the corr-side transport-TX integration, this is what gets
        sent on every push tick: the latest cube. Does not modify the
        FIFO state.
        """
        if not self._buf:
            return None
        return self._buf[-1]

    def peek_oldest(self) -> torch.Tensor | None:
        """Oldest cube (the head), or ``None`` if empty.

        For the search-side detector context window, the oldest cube
        is the start of the rolling detector look-back. Does not
        modify the FIFO state.
        """
        if not self._buf:
            return None
        return self._buf[0]

    def as_list(self) -> list[torch.Tensor]:
        """Current contents in oldest-first order (length 0..depth).

        The list is a fresh copy of the FIFO's references; mutating
        the returned list (`append`/`pop`/`reverse`) is safe and does
        NOT change the FIFO state. The contained tensors, however,
        are the same Python objects the FIFO holds; mutating their
        data WILL be visible to subsequent reads.
        """
        return list(self._buf)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate cubes in oldest-first order."""
        return iter(self._buf)

    # ------------------------------------------------------------------
    # Chunk-4 integration adapter
    # ------------------------------------------------------------------

    def push_for_protocol(
        self, cube: torch.Tensor, *, block_n: int = 0,
    ) -> list[torch.Tensor]:
        """Adapter matching the chunk-4 ``Stage2FifoStage`` Protocol.

        The chunk-4 ``corr_fast_integration`` orchestrator (in
        ``dsart.services.corr_fast_integration``) declares::

            class Stage2FifoStage(Protocol):
                def push(self, dedispersed: torch.Tensor, *, block_n: int)
                    -> list[torch.Tensor]: ...

        — it treats the FIFO as returning a *list* of evictees (any
        length 0..N) so the transport-TX layer downstream can treat
        emit semantics uniformly. :meth:`push` here returns
        ``Optional[Tensor]`` (cleaner for the per-push case where at
        most one cube is evicted), so this method wraps the call and
        produces the protocol-shaped list.

        Use this adapter at the chunk-4 integration site::

            ctx.stage2_fifo = Stage2FIFO(depth=...)
            # ... in run_block():
            cubes_for_tx = ctx.stage2_fifo.push_for_protocol(
                dedispersed, block_n=block_n,
            )

        ``block_n`` is accepted but ignored — the FIFO does not store
        per-block metadata (the cube itself carries any timestamp
        information via its caller-attached attrs, or via the
        transport header at TX time).
        """
        evicted = self.push(cube)
        return [] if evicted is None else [evicted]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Drop all cubes; reset the type-contract reference triplet.

        After :meth:`clear`, the next push re-establishes
        ``(shape, dtype, device)``. Use this on a service ``cmd: stop``
        / ``cmd: start`` cycle (plan §3.6.2 stage-2 FIFO is *not*
        checkpointed v1, so any restart re-warms the FIFO from cold).
        """
        self._buf.clear()
        self._ref_shape = None
        self._ref_dtype = None
        self._ref_device = None

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._buf)
        ref = "uninit" if self._ref_shape is None else (
            f"shape={self._ref_shape} dtype={self._ref_dtype} "
            f"device={self._ref_device}"
        )
        return f"Stage2FIFO(depth={self._depth}, n={n}, ref={ref})"
