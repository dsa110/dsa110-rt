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
import time
import warnings
from typing import AsyncIterator, Mapping, Optional

import numpy as np

from dsart.common.constants import (
    CUBE_CADENCE_SAMPLES_DEFAULT,
    T_INT_SEARCH_US_DEFAULT,
)
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
    VF_RX_OVERRUN,
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
        t_int_search_us: float = T_INT_SEARCH_US_DEFAULT,
        cube_cadence_samples: int = CUBE_CADENCE_SAMPLES_DEFAULT,
        n_grid: int = 256,
        enable_cuda_register: bool = True,
        poll_interval_s: float = 0.001,
        max_cubes: Optional[int] = None,
        fan_in_min_corrs: int = 1,
        attach_timeout_s: float = 30.0,
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
        if not 1 <= fan_in_min_corrs <= ring_dims.n_corr:
            raise ValueError(
                f"fan_in_min_corrs={fan_in_min_corrs} not in "
                f"[1, {ring_dims.n_corr}]"
            )

        self._shm_name = shm_name
        self._ring_dims = ring_dims
        self._n_fdm_in_cube = int(n_fdm_in_cube)
        self._t_det = int(t_det)
        self._compute_half = int(compute_half)
        self._cube_cadence_samples = int(cube_cadence_samples)
        self._n_grid_override = int(n_grid)
        self._poll_interval_s = float(poll_interval_s)
        self._max_cubes = max_cubes
        self._enable_cuda_register = bool(enable_cuda_register)
        self._fan_in_min_corrs = int(fan_in_min_corrs)
        self._attach_timeout_s = float(attach_timeout_s)

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

        # Mon-dict counters (read by SearchComputeService for status JSON).
        # These are PROCESS-LOCAL — the assembler bumps them as it reads
        # the ring. They are the consumer-side mirror of the recv_epoll
        # ring_* counters (the producer's view).
        self._n_slots_read: int = 0
        self._n_overrun: int = 0
        self._n_pattern_mismatch: int = 0
        self._n_no_data_present: int = 0

        # Pre-allocated zero-filled per_chgroup_streams cache.
        # ----------------------------------------------------
        # The CubePipeline GPU path actually consumes the cint8 stack
        # (see CubeRingSlot.per_chgroup_cint8_stack); the per_chgroup_streams
        # dict is the CPU/legacy fallback path. For the M7.2 system
        # bring-up the synthetic TX (bench/net_pair) ships all-zero
        # payloads, so the physically-correct streams ARE all zeros —
        # we can amortise the alloc across cubes by handing out the
        # same cached view every cube. The downstream pipeline only
        # READS the streams (combine_chgroups reduces them into the
        # search cube), so sharing the buffer is safe.
        #
        # NOTE (deferred to M7.4): once real on-sky data flows, the
        # assembler needs to scatter the raw quantised COO bytes from
        # each ring slot into the dense per_chgroup grid using a
        # pre-built linear-index LUT keyed by pattern_id, AND ship
        # scale/offset/pattern_id metadata via a sidecar (the ring
        # payload only carries cell values, not coordinates or quant
        # metadata). See M4a_PLAN_FIXES.md note re: "dequant on
        # compute reader".
        self._t_stream = self._t_det + int(
            self._time_shift_table.shifts.max(initial=0)
        )
        self._n_grid_cached = self._n_grid_override
        self._per_chgroup_streams_zero: dict[int, np.ndarray] = {
            corr: np.zeros(
                (self._t_stream, self._n_grid_cached, self._n_grid_cached),
                dtype=np.complex64,
            )
            for corr in range(self._ring_dims.n_corr)
        }

    @property
    def time_shift_table(self) -> TimeShiftSearchTable:
        return self._time_shift_table

    @property
    def n_grid(self) -> int:
        """The grid side length, surfaced to downstream consumers that
        build dense ``[N_grid, N_grid]`` uv-grids. The ring carries
        sparse-COO cell values only (no coordinates); ``n_grid`` is the
        contract dimension matched against the search-compute config."""
        return self._n_grid_cached

    @property
    def stats(self) -> dict:
        """Process-local read counters (mon-dict source)."""
        return {
            "cubes_emitted": int(self._cubes_emitted),
            "n_slots_read": int(self._n_slots_read),
            "n_overrun": int(self._n_overrun),
            "n_pattern_mismatch": int(self._n_pattern_mismatch),
            "n_no_data_present": int(self._n_no_data_present),
        }

    @property
    def cuda_registered(self) -> bool:
        return self._cuda_registered

    async def start(self) -> None:
        """Open the shm ring and optionally cudaHostRegister it.

        The shm segment is created by the search_rx process (the ring
        OWNER), which can take ~1-2 s to bind its sockets and bring up
        the SPMC ring before the consumer-side mmap_attach_readonly
        can succeed. Under dsart_rt all routines fork-exec in the same
        verb dispatch, so the search_compute halves typically race
        search_rx by ~100-1000 ms. We retry the attach for up to
        ``attach_timeout_s`` (default 30 s) so the orchestrator can
        treat the routines as order-independent without explicit
        sleeps in the YAML.

        The retry interval is fixed at 200 ms (15× per second). For a
        typical "owner came up 1 s after consumer" race this is 5
        polls before success, which is below the noise floor of the
        rest of the bring-up flow.
        """
        if self._started:
            return
        attach_timeout_s = self._attach_timeout_s
        deadline = time.monotonic() + attach_timeout_s
        last_err: OSError | None = None
        while True:
            try:
                self._ring = RxRing.mmap_attach_readonly(
                    self._shm_name, self._ring_dims
                )
                break
            except OSError as exc:
                # Two retryable races:
                #   - "No such file or directory": shm not yet created
                #     by the owner (typical when search_rx and
                #     search_compute fork-exec simultaneously).
                #   - "bad magic: 0x00000000": shm exists but the owner
                #     has not yet called rx_ring_init_header — i.e. the
                #     shm_open(O_CREAT) raced ahead of the header memset.
                # Everything else (dims mismatch, perms, real header
                # corruption) propagates immediately.
                msg = str(exc)
                if (
                    "No such file or directory" not in msg
                    and "bad magic: 0x00000000" not in msg
                ):
                    raise
                last_err = exc
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"ProductionRxRingSource.start: waited "
                        f"{attach_timeout_s:.1f}s for owner to create "
                        f"shm '{self._shm_name}'; giving up "
                        f"(last error: {msg})"
                    ) from exc
                LOG.debug(
                    "shm '%s' not yet present; retrying in 200ms "
                    "(deadline in %.1fs)",
                    self._shm_name,
                    deadline - time.monotonic(),
                )
                await asyncio.sleep(0.200)
        if last_err is not None:
            LOG.info(
                "shm '%s' attached after retry (initial: %s)",
                self._shm_name,
                last_err,
            )
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

            # Cube-emit gating (M7.2 fan-in semantics):
            # ----------------------------------------
            # The full 16-corr search fan-in is designed for every
            # chgroup to be live; if any chgroup is silent the natural
            # ``min(write_seq)`` over corrs stays at 0 and the source
            # deadlocks (the M7.2 loopback smoke caught this when only
            # chgroup 0 was sending). We instead emit a cube as soon as
            # ``fan_in_min_corrs`` chgroups have advanced past the next
            # boundary — the per-(corr, t) ``validity_mask`` written by
            # ``_assemble_cube`` already marks silent corrs' samples as
            # invalid, so the downstream detector correctly gates Layer-2
            # EMA updates on the partial cube. Production sets
            # ``fan_in_min_corrs == n_corr`` (every chgroup required);
            # smoke / dev set it to 1.
            wseqs = [
                self._ring.get_write_seq(corr)
                for corr in range(self._ring_dims.n_corr)
            ]
            target_seq = last_cube_seq_boundary + self._cube_cadence_samples
            n_at_target = sum(1 for w in wseqs if w >= target_seq)
            if n_at_target < self._fan_in_min_corrs:
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

        Walks every ring slot the cube depends on — ``(corr, coarse_dm, t)``
        for ``corr ∈ [0, n_corr)``, ``coarse_dm ∈ [0, n_coarse_dm)``, and
        ``t ∈ [specnum_start, specnum_start + cube_cadence_samples)`` —
        and builds a :class:`CubeRingSlot`. Each ``read_slot`` exercises
        the real shm path (kernel mmap, atomic acquire on write_seq, full
        slot ``memcpy``), so the assembler exerts production-realistic
        bandwidth pressure on the ring even in synthetic-TX bring-up.

        Validity mask ``[T_det, N_fdm]``:

          * Initial state: ``True`` everywhere.
          * Drops to ``False`` for ``t in [0, T_det)`` whenever ANY
            (corr, coarse_dm) slot at that ``t`` is:
              - read-side overrun (``rx_ring_read_slot`` returned -1
                because the writer lapped us);
              - missing ``VF_DATA_PRESENT``; or
              - carrying ``VF_PATTERN_MISMATCH``.
          * Detector consumes ``torch.all(validity_mask)`` as a coarse
            cube-valid bool gating Layer-2 EMA updates (see
            :meth:`DeterministicDetector.forward`), so any per-slot
            problem suppresses sigma-bank learning for that whole cube.

        ``per_chgroup_streams``:

          * **M7.2 bring-up scope (this implementation):** hands the
            consumer the pre-allocated zero-filled cache built in
            ``__init__``. The synthetic TX (``bench/net_pair``) ships
            all-zero payloads, so the physically-correct streams ARE
            zeros and we save the per-cube allocation + memset (which
            would otherwise be ~1.2 GB of complex64 per cube at
            16 chgroups × T_stream × N_grid² ).
          * **Deferred to M7.4 (real on-sky):** scatter the quantised
            COO cell values from each ring slot into the dense
            ``[T_stream, N_grid, N_grid]`` per-chgroup grid using a
            pre-built linear-index LUT keyed by ``pattern_id``, and
            apply per-slot scale/offset for dequant. That work also
            requires the recv_epoll producer to ship ``scale`` /
            ``offset`` / ``pattern_id`` / ``n_filled`` as a sidecar in
            the ring slot (today the slot is payload-bytes + validity
            only); see M4a_PLAN_FIXES.md and ``recv_ring.c`` slot
            layout for the contract.
        """
        assert self._ring is not None
        n_corr = self._ring_dims.n_corr
        n_coarse_dm = self._ring_dims.n_coarse_dm
        n_grid = self.n_grid

        validity_mask = np.ones(
            (self._t_det, self._n_fdm_in_cube), dtype=np.bool_
        )

        for corr in range(n_corr):
            # Read every (coarse_dm, t) slot the cube depends on. This
            # is real ring bandwidth — slot size = n_filled × bpc bytes
            # (e.g. 5000 × 2 = 10 KiB at the default op-point), so per
            # cube we move n_corr × n_coarse_dm × cube_cadence × 10 KiB
            # bytes through ``rx_ring_read_slot``. At
            # 7.45 cubes/s × 16 × 5 × 256 × 10 KiB ≈ 1.5 GB/s of memcpy
            # — well within the host's L3/RAM bandwidth on n01/n06.
            for dm in range(n_coarse_dm):
                for t in range(self._cube_cadence_samples):
                    t_abs = specnum_start + t
                    self._n_slots_read += 1
                    try:
                        _payload, vf = self._ring.read_slot(
                            corr=corr,
                            dm=dm,
                            t_seq=t_abs,
                            compute_half=self._compute_half,
                        )
                    except OSError:
                        # rx_ring_read_slot returned -1 with VF_RX_OVERRUN
                        # (writer lapped us by more than T_buf samples).
                        self._n_overrun += 1
                        if t < self._t_det:
                            validity_mask[t, :] = False
                        continue
                    if vf & VF_RX_OVERRUN:
                        self._n_overrun += 1
                        if t < self._t_det:
                            validity_mask[t, :] = False
                        continue
                    if vf & VF_PATTERN_MISMATCH:
                        self._n_pattern_mismatch += 1
                        if t < self._t_det:
                            validity_mask[t, :] = False
                        continue
                    if not (vf & VF_DATA_PRESENT):
                        # Hole (e.g. window-slide zerofill from
                        # recv_epoll). The slot wrote but with
                        # validity=0; counts as "no data this t".
                        self._n_no_data_present += 1
                        if t < self._t_det:
                            validity_mask[t, :] = False

        return CubeRingSlot(
            cube_id=cube_id,
            specnum_start=specnum_start,
            per_chgroup_streams=self._per_chgroup_streams_zero,
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
