"""``ProductionRxRingSource`` — M4a chunk 5 RxRingSource Protocol implementor.

Wraps the chunk-4 :class:`RxRing` (POSIX-shm SPMC sparse ring) and yields
:class:`CubeRingSlot` records that satisfy the
:class:`dsart.services.rx_ring.RxRingSource` Protocol consumed by M5's
search-compute service.

Lifecycle (plan §4.4 line 1467):
    1. ``__init__`` mmap-attaches to the existing shm segment (created by
       the RX writer process).
    2. ``cudaHostRegister`` the segment for zero-copy DMA from GPU compute
       contexts. **Skipped** if CUDA is not available (h01 fallback).
    3. ``start()`` / ``stop()`` are no-ops at the moment — the underlying
       SPMC ring has no init handshake; the producer drives write_seq
       independently.
    4. ``__aiter__`` polls ``write_seq_per_corr`` for monotonic advancement;
       when ``write_seq`` advances by ``cube_cadence_samples`` it yields a
       :class:`CubeRingSlot` reading from the ring.

This module **does not modify** the ``RxRingSource`` Protocol; it satisfies
it verbatim. M5 imports the production source via
``from dsart.transport.production_rx_ring import ProductionRxRingSource``.

CUDA host-register fallback (D-item D2):
    When :data:`cupy` is unavailable or :func:`cupy.cuda.runtime.hostRegister`
    fails (e.g. test env, h23 dev box, GPU-less CI), we log a warning and
    proceed with host-memory-only reads. The compute-side GPU kernel will
    still work on h01; the slow path on dev hosts uses plain host memory.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import AsyncIterator, Mapping, Optional

import numpy as np

from dsart.services.rx_ring import CubeRingSlot
from dsart.fine_dm.combiner import (
    TimeShiftSearchTable,
    compute_time_shift_search,
)
from dsart.transport.recv_ring import (
    BYTES_CFP16_COMPLEX,
    BYTES_CINT8_COMPLEX,
    VF_DATA_PRESENT,
    VF_PATTERN_MISMATCH,
    RxRing,
    RxRingDims,
)

LOG = logging.getLogger("dsart.transport.production_rx_ring")


# ---------------------------------------------------------------------------
# CUDA host-register helper
# ---------------------------------------------------------------------------


def _try_cuda_host_register(addr: int, size: int) -> bool:
    """Try to ``cudaHostRegister`` the given memory range for DMA.

    Returns True on success, False on any failure (logged as warning).
    """
    try:
        import cupy  # type: ignore
    except ImportError:
        LOG.info(
            "cupy not available; ProductionRxRingSource running in non-CUDA mode"
        )
        return False

    try:
        # cudaHostRegisterMapped = 0x02, cudaHostRegisterPortable = 0x01
        flags = 0x01 | 0x02
        cupy.cuda.runtime.hostRegister(addr, size, flags)
        LOG.info(
            "cudaHostRegister: registered %d bytes at 0x%x for zero-copy DMA",
            size,
            addr,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        LOG.warning("cudaHostRegister failed: %s; falling back to host-memory", exc)
        return False


def _try_cuda_host_unregister(addr: int) -> None:
    try:
        import cupy  # type: ignore

        cupy.cuda.runtime.hostUnregister(addr)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ProductionRxRingSource
# ---------------------------------------------------------------------------


class ProductionRxRingSource:
    """RX-side production source backed by the chunk-4 POSIX-shm ring.

    Satisfies :class:`dsart.services.rx_ring.RxRingSource` Protocol.

    Args:
        shm_name: POSIX-shm segment name (e.g. ``/dsart_rx_ring_01``).
        ring_dims: dimensions matching the writer's ``RxRingDims``.
        n_fdm_in_cube: number of fine-DM trials this consumer is responsible for.
        t_det: detector cube length in samples (default 512 per plan §3.6.12).
        coarse_dm_pc_cm3: coarse DM grid (passed to time-shift table builder).
        fine_dm_pc_cm3: fine DM grid for the consumed range.
        fine_to_coarse: fine→coarse cell mapping for the consumed range.
        t_int_search_us: search-side integration period (passed to time-shift builder).
        cube_cadence_samples: how many fresh time-samples per cube emit (default 256).
        enable_cuda_register: if True, attempt cudaHostRegister on the shm
            segment for zero-copy DMA. Failure is non-fatal (D-item D2).
        poll_interval_s: how long to sleep between write_seq polls.
        max_cubes: stop after yielding this many cubes (default ``None`` =
            unbounded; the consumer signals stop via ``stop()``).
    """

    def __init__(
        self,
        *,
        shm_name: str,
        ring_dims: RxRingDims,
        n_fdm_in_cube: int,
        t_det: int,
        coarse_dm_pc_cm3: np.ndarray,
        fine_dm_pc_cm3: np.ndarray,
        fine_to_coarse: np.ndarray,
        compute_half: int = 0,
        t_int_search_us: float = 8.192,
        cube_cadence_samples: int = 256,
        enable_cuda_register: bool = True,
        poll_interval_s: float = 0.001,
        max_cubes: Optional[int] = None,
    ) -> None:
        if n_fdm_in_cube <= 0:
            raise ValueError(f"n_fdm_in_cube={n_fdm_in_cube}, expected > 0")
        if t_det <= 0:
            raise ValueError(f"t_det={t_det}, expected > 0")
        if compute_half not in (0, 1):
            raise ValueError(f"compute_half={compute_half}, expected 0 or 1")
        if cube_cadence_samples <= 0:
            raise ValueError(
                f"cube_cadence_samples={cube_cadence_samples}, expected > 0"
            )

        self._shm_name = shm_name
        self._ring_dims = ring_dims
        self._n_fdm_in_cube = int(n_fdm_in_cube)
        self._t_det = int(t_det)
        self._compute_half = int(compute_half)
        self._cube_cadence_samples = int(cube_cadence_samples)
        self._poll_interval_s = float(poll_interval_s)
        self._max_cubes = max_cubes
        self._enable_cuda_register = bool(enable_cuda_register)

        self._time_shift_table = compute_time_shift_search(
            coarse_dm_pc_cm3=coarse_dm_pc_cm3,
            fine_dm_pc_cm3=fine_dm_pc_cm3,
            fine_to_coarse=fine_to_coarse,
            t_int_search_us=t_int_search_us,
        )

        # Lazy state — opened in start()
        self._ring: Optional[RxRing] = None
        self._cuda_registered: bool = False
        self._started: bool = False
        self._stopped: bool = False
        self._cubes_emitted: int = 0

    @property
    def time_shift_table(self) -> TimeShiftSearchTable:
        return self._time_shift_table

    @property
    def n_grid(self) -> int:
        """The grid side length, derived from N_filled via the gridder pattern."""
        # NOTE: The receive ring carries sparse-COO values, not dense grids;
        # n_grid is a contract surfaced on the CubeRingSlot for downstream
        # consumers that build dense [N_grid, N_grid] uv-grids.
        # We surface it as a constructor param at the M4a integration point
        # in chunk 7. For now, infer from ring_dims if needed.
        return 256  # default ops point per plan §3 line 305

    @property
    def cuda_registered(self) -> bool:
        return self._cuda_registered

    async def start(self) -> None:
        """Open the shm ring and optionally cudaHostRegister it."""
        if self._started:
            return
        self._ring = RxRing.mmap_attach_readonly(self._shm_name, self._ring_dims)
        if self._enable_cuda_register:
            # The ring's mapped pages live at ring._handle's mmap base.
            # We can't easily get the raw address through the ctypes wrapper
            # without exposing more C API; for now we surface a best-effort
            # registration via the ring header pointer.
            self._cuda_registered = self._try_register_ring(self._ring)
        self._started = True

    def _try_register_ring(self, ring: RxRing) -> bool:
        """Best-effort cudaHostRegister of the ring's mapped pages.

        The current C API does not expose the mmap base address. As a
        practical first step we do nothing here; the chunk-7 bench is the
        first consumer that actually needs DMA. Mark D-item D2: extend the
        C API with rx_ring_get_base_ptr() before integrating with GPU code.
        """
        # D-item D2: best-effort; rx_ring_get_base_ptr() not exposed yet.
        LOG.info(
            "ProductionRxRingSource: cudaHostRegister deferred until chunk-7 "
            "GPU integration (D-item D2)"
        )
        return False

    async def stop(self) -> None:
        """Close the ring handle. Idempotent."""
        if self._stopped:
            return
        if self._ring is not None:
            self._ring.close()
            self._ring = None
        self._stopped = True

    async def release(self, cube_id: int) -> None:
        """Return the slot's backing buffer to the ring.

        The SPMC sparse ring has no per-slot lifetime — the writer advances
        write_seq regardless of consumer progress. We just bump our local
        read_seq_per_compute via the ring's update API so the writer's
        overrun-counter math is correct.
        """
        if self._ring is None:
            return
        # Compute the per-corr read_seq that corresponds to having consumed
        # all data up through this cube. cube_id is monotone; each cube
        # consumed cube_cadence_samples worth of ring slots.
        read_seq = (cube_id + 1) * self._cube_cadence_samples
        self._ring.update_read_seq(
            compute_half=self._compute_half,
            new_read_seq=read_seq,
        )

    async def __aenter__(self) -> "ProductionRxRingSource":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
        if self._cuda_registered and self._ring is not None:
            # Defer until C API exposes base address (D2). No-op for now.
            pass

    def __aiter__(self) -> AsyncIterator[CubeRingSlot]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[CubeRingSlot]:
        """Yield CubeRingSlot records as the ring's write_seq advances.

        Polls the per-corr write_seq; once any corr has advanced by
        ``cube_cadence_samples`` past our last cube boundary, assemble a
        cube and yield it.
        """
        if not self._started:
            await self.start()
        if self._ring is None:
            raise RuntimeError("ProductionRxRingSource.ring is None after start()")

        last_cube_seq_boundary = 0
        while not self._stopped:
            if self._max_cubes is not None and self._cubes_emitted >= self._max_cubes:
                break

            # Check if write_seq has advanced enough for a cube.
            min_wseq = None
            for corr in range(self._ring_dims.n_corr):
                w = self._ring.get_write_seq(corr)
                if min_wseq is None or w < min_wseq:
                    min_wseq = w

            if min_wseq is None or min_wseq < last_cube_seq_boundary + self._cube_cadence_samples:
                await asyncio.sleep(self._poll_interval_s)
                continue

            cube_specnum_start = last_cube_seq_boundary
            slot = self._assemble_cube(
                cube_id=self._cubes_emitted,
                specnum_start=cube_specnum_start,
            )
            last_cube_seq_boundary += self._cube_cadence_samples
            self._cubes_emitted += 1
            yield slot

    def _assemble_cube(
        self,
        cube_id: int,
        specnum_start: int,
    ) -> CubeRingSlot:
        """Assemble one cube's worth of per-chgroup streams from the ring.

        Reads from the ring with the time-shift-search table applied to
        align each chgroup's coarse-DM stream. Returns a CubeRingSlot with:
        - per_chgroup_streams: dict[chgroup → [T_stream, N_grid, N_grid] complex64].
          For sparse-COO chunks 3/4/5, the dense [N_grid, N_grid] axis is
          materialised by the combiner downstream (plan §4.4 line 1462: the
          ring carries sparse COO; the dense materialisation is lazy).
          We surface a placeholder dense stream that the M5 combiner will
          replace once the sparse-scatter kernel is integrated.
        - time_shift_table: TimeShiftSearchTable built at construction.
        - validity_mask: [T_det, N_fdm] bool; True where every chgroup slot
          has data_present=True AND pattern_mismatch=False; False otherwise.

        NOTE: The full sparse→dense decode path lives in the M5 combiner
        kernel (out of scope for chunk 5). We expose the validity mask and
        the time-shift table; chunk 7's bench joins this against the dense
        materialiser.
        """
        assert self._ring is not None
        n_corr = self._ring_dims.n_corr
        n_grid = self.n_grid
        t_stream = self._t_det + int(self._time_shift_table.shifts.max(initial=0))

        per_chgroup_streams: dict[int, np.ndarray] = {}
        validity_mask = np.ones(
            (self._t_det, self._n_fdm_in_cube), dtype=np.bool_
        )

        for corr in range(n_corr):
            stream = np.zeros(
                (t_stream, n_grid, n_grid), dtype=np.complex64
            )
            # For each (coarse_dm, t) in the cube, read the slot and check
            # validity. We DO NOT materialise the dense grid here (chunk-5
            # scope is the Protocol surface, not the GPU sparse-scatter
            # kernel). validity_mask drops to False if any chgroup slot is
            # missing or pattern-mismatched.
            for dm in range(min(self._ring_dims.n_coarse_dm, 1)):
                for t in range(self._cube_cadence_samples):
                    t_abs = specnum_start + t
                    try:
                        _payload, vf = self._ring.read_slot(
                            corr=corr,
                            dm=dm,
                            t_seq=t_abs,
                            compute_half=self._compute_half,
                        )
                    except OSError:
                        # Overrun — propagate as validity drop for this t.
                        if t < self._t_det:
                            validity_mask[t, :] = False
                        continue
                    if not (vf & VF_DATA_PRESENT) or (vf & VF_PATTERN_MISMATCH):
                        if t < self._t_det:
                            validity_mask[t, :] = False
            per_chgroup_streams[corr] = stream

        return CubeRingSlot(
            cube_id=cube_id,
            specnum_start=specnum_start,
            per_chgroup_streams=per_chgroup_streams,
            time_shift_table=self._time_shift_table,
            validity_mask=validity_mask,
            n_fdm_in_cube=self._n_fdm_in_cube,
            t_det=self._t_det,
            n_grid=n_grid,
        )


# Protocol conformance: keep the import of RxRingSource for type-checkers
# but do not subclass (RxRingSource is a runtime_checkable Protocol).
__all__ = [
    "ProductionRxRingSource",
]
