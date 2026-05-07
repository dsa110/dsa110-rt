"""Receive-ring source abstraction for the search-compute service (M5 Chunk 6b-α).

The production receive-ring is M4a-owned: a POSIX-shm ring fed by the
search-RX layer (§3.5 lines 1396-1421 + plan §4.2 transport spec). The
on-the-wire payload is the M3-emitted ``SparseCOOPayload`` (one per
``(chgroup, coarse_dm, time_bin)`` tuple, see
``common.contracts.SparseCOOPayload``); the search-RX defragments and
deposits these into the ring; M5's combiner reads them out.

To let M5 develop and bench the detector independently of M4a, this
module defines:

  * ``RxRingSource`` — an async ``Protocol`` providing
    ``acquire_next_cube()`` / ``release(cube_id)``. The production M4a
    implementation will satisfy this Protocol; benches can plug in
    synthetic sources.
  * ``CubeRingSlot`` — the per-cube data tuple yielded by the source
    (per-chgroup dense uv-streams + cube metadata + validity mask).
  * ``SyntheticRxRingSource`` — a deterministic numpy-backed source
    used by ``bench/search_node_throughput.py`` and unit tests. It
    generates noise-only or noise+injection per-chgroup streams without
    any sparse-COO transport.

The combiner (``fine_dm/combiner.py``) reads the per-chgroup streams
out of the slot, applies the §3.6.3 integer-sample shifts, and feeds
the imager. The chunk-6b-α scope is the **dense-stream** path: the
fused sparse-scatter-and-sum cupy kernel that consumes raw
``SparseCOOPayload`` records is deferred to the later chunk-6b
production-perf hardening pass once M4a's receive-ring API is locked.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import (
    AsyncIterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from ..common.constants import (
    NU_BOT_PROC_GHZ,
    NU_CHGROUP_BOT_GHZ,
    N_CHGROUP,
    T_INT_SEARCH_US_DEFAULT,
)
from ..fine_dm.combiner import (
    TimeShiftSearchTable,
    compute_time_shift_search,
)

__all__ = [
    "CubeRingSlot",
    "RxRingSource",
    "SyntheticRxRingSource",
    "SyntheticInjection",
]


# ---------------------------------------------------------------------------
# Cube ring-slot data tuple
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubeRingSlot:
    """One cube's worth of pre-imaging input.

    The production M4a implementation hands back a slot that maps each
    ``chgroup ∈ 0..15`` to a dense
    ``[T_stream, N_grid, N_grid] complex64`` uv-grid stream (already
    coarse-dedispersed at the corr-side per §3.6.2 + §3.6.3 stage-2;
    M4a is responsible for the receive-ring → combiner edge dense
    decode). The combiner consumes these streams + the
    ``time_shift_table`` to assemble the per-fine-DM head block.

    Fields:
        cube_id: monotonically-increasing sequence number
            (``specnum_block_start``-equivalent for cube cadence).
        specnum_start: first spectral-sample number covered by this
            cube. Used downstream to stamp ``Candidate.event_specnum``.
        per_chgroup_streams: dict ``chgroup ∈ 0..15 → [T_stream, N_grid,
            N_grid] complex64``. ``T_stream`` MUST be ≥ ``T_det +
            max(time_shift_table.shifts)`` so the combiner's
            head-block reads land in-range.
        time_shift_table: ``TimeShiftSearchTable`` for the fine-DM
            trials covered by this cube. The combiner reads
            ``per_chgroup_streams[g]`` at ``[t - shifts[f, g]]`` for
            each ``(t, f)`` cube-time / fine-DM pair.
        validity_mask: ``[T_det, N_fdm] bool``. False on RFI'd or
            warmup samples; the detector skips Layer-2 EMA updates on
            invalid cubes (§D14).
        n_fdm_in_cube: number of fine-DM trials this cube covers; the
            combiner builds ``[T_det, n_fdm_in_cube, N_grid, N_grid]``.
        t_det: number of cube-time samples per cube (production: 512).
        n_grid: spatial grid side (production: 256).
        per_chgroup_cint8_stack: optional pre-quantised cint8 stream
            stack ``[N_chg, T_stream, 2, N_grid, N_grid] int8`` (split-
            plane re/im, M3 wire layout). When present, the GPU
            ``_build_cube_gpu`` path bypasses host-side cf -> cint8
            quantisation and copies straight to the GPU. This is the
            chunk-8b RX-ring contract: M3 emits cint8 streams already
            quantised with per-block (scale, offset); production never
            re-quantises on the search node. Bench paths (synthetic
            source, ``--prequantise``) populate this field to emulate
            chunk-8b delivery and isolate the GPU pipeline cost from
            host-side bench scaffolding. Default ``None`` keeps the
            chunk-6a contract (cf streams only).
    """

    cube_id: int
    specnum_start: int
    per_chgroup_streams: Mapping[int, np.ndarray]
    time_shift_table: TimeShiftSearchTable
    validity_mask: np.ndarray
    n_fdm_in_cube: int
    t_det: int
    n_grid: int
    per_chgroup_cint8_stack: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.cube_id < 0:
            raise ValueError(f"cube_id={self.cube_id}, expected ≥ 0")
        if self.specnum_start < 0:
            raise ValueError(f"specnum_start={self.specnum_start}, expected ≥ 0")
        if self.t_det <= 0:
            raise ValueError(f"t_det={self.t_det}, expected > 0")
        if self.n_grid <= 0 or self.n_grid & (self.n_grid - 1):
            raise ValueError(
                f"n_grid={self.n_grid}, expected positive power of two"
            )
        if self.n_fdm_in_cube <= 0:
            raise ValueError(
                f"n_fdm_in_cube={self.n_fdm_in_cube}, expected > 0"
            )
        if self.validity_mask.shape != (self.t_det, self.n_fdm_in_cube):
            raise ValueError(
                f"validity_mask.shape={self.validity_mask.shape} != "
                f"({self.t_det}, {self.n_fdm_in_cube})"
            )
        if self.validity_mask.dtype != np.bool_:
            raise TypeError(
                f"validity_mask.dtype={self.validity_mask.dtype}, expected bool"
            )
        if self.time_shift_table.shifts.shape[0] != self.n_fdm_in_cube:
            raise ValueError(
                f"time_shift_table covers {self.time_shift_table.shifts.shape[0]} "
                f"fine-DM trials != n_fdm_in_cube={self.n_fdm_in_cube}"
            )
        if self.per_chgroup_cint8_stack is not None:
            cint8 = self.per_chgroup_cint8_stack
            if cint8.dtype != np.int8:
                raise TypeError(
                    f"per_chgroup_cint8_stack.dtype={cint8.dtype}, expected int8"
                )
            if cint8.ndim != 5 or cint8.shape[2] != 2:
                raise ValueError(
                    f"per_chgroup_cint8_stack.shape={cint8.shape}, expected "
                    f"(N_chg, T_stream, 2, N_grid, N_grid)"
                )
            if cint8.shape[3] != self.n_grid or cint8.shape[4] != self.n_grid:
                raise ValueError(
                    f"per_chgroup_cint8_stack n_grid axes "
                    f"({cint8.shape[3]}, {cint8.shape[4]}) != n_grid="
                    f"{self.n_grid}"
                )


# ---------------------------------------------------------------------------
# RxRingSource Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RxRingSource(Protocol):
    """Async source of ``CubeRingSlot``s for the search-compute service.

    The production implementation (M4a) blocks on a POSIX-shm
    ``head_block_ready`` semaphore and yields slots referencing the
    shared-memory data without copy. Bench / test implementations
    generate slots synchronously from numpy.

    Lifecycle:
      * ``await source.start()`` — initialise (open shm, etc.).
      * ``async for slot in source:`` — iterate cubes; yields
        ``CubeRingSlot``.
      * ``await source.release(slot.cube_id)`` — return the slot's
        backing buffer to the ring (the production path is bound by
        this; benches may no-op).
      * ``await source.stop()`` — tear down.

    Implementations should be `async-iterable`; the canonical pattern is::

        async with source:
            async for slot in source:
                cands = await pipeline.process(slot)
                await source.release(slot.cube_id)
    """

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def release(self, cube_id: int) -> None: ...

    def __aiter__(self) -> AsyncIterator[CubeRingSlot]: ...


# ---------------------------------------------------------------------------
# SyntheticRxRingSource — bench / test source
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticInjection:
    """One injection in the synthetic source (bench-side counterpart to
    ``CubeInjectionConfig``, but at the pre-imaging uv-grid level).

    The synthetic source places a delta-amplitude bump in the
    pre-imaging uv-grid stream of a single chgroup at a known
    ``(cube_idx, t_in_cube, l_pix, m_pix)``. This is a coarse stand-in
    for a sky source — the bench/throughput path doesn't need
    Hermitian-symmetric injection because the throughput bench gates
    on rate and end-to-end latency, not detection quality. For
    detection-quality benches, use ``bench/cube_injection_detector.py``
    (which bypasses the imager entirely).
    """

    cube_idx: int
    t_in_cube: int
    l_pix: int
    m_pix: int
    amplitude: float = 1.0


class SyntheticRxRingSource:
    """Deterministic numpy-backed RX-ring source for benches + tests.

    Generates per-chgroup dense uv-streams of complex Gaussian noise
    (σ = 1 per cell, real and imaginary parts iid), with optional
    delta-amplitude injections at known cells. The
    ``time_shift_table`` is computed once at construction from the
    coarse/fine DM grids the caller supplies.

    Args:
        n_cubes: total number of cubes to yield before stopping.
        t_det: cube-time samples per cube.
        n_fdm: number of fine-DM trials per cube.
        n_grid: spatial grid side.
        coarse_dm_pc_cm3: ``[N_coarse]`` coarse-DM grid.
        fine_dm_pc_cm3: ``[N_fdm]`` fine-DM grid.
        fine_to_coarse: ``[N_fdm]`` fine→coarse cell mapping.
        rng: optional ``np.random.Generator`` for reproducibility.
        injections: optional list of ``SyntheticInjection`` to splat
            into the streams.
        cube_cadence_s: simulated wall-clock cube cadence; the source
            ``await asyncio.sleep`` between cubes. Default 0 (yield as
            fast as the consumer can drain).
    """

    def __init__(
        self,
        *,
        n_cubes: int,
        t_det: int,
        n_fdm: int,
        n_grid: int,
        coarse_dm_pc_cm3: np.ndarray,
        fine_dm_pc_cm3: np.ndarray,
        fine_to_coarse: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        injections: Sequence[SyntheticInjection] = (),
        cube_cadence_s: float = 0.0,
        t_int_search_us: float = T_INT_SEARCH_US_DEFAULT,
        pre_quantise: bool = False,
        prequantise_target_max: int = 120,
    ) -> None:
        if n_cubes <= 0:
            raise ValueError(f"n_cubes={n_cubes}, expected > 0")
        self._n_cubes = int(n_cubes)
        self._t_det = int(t_det)
        self._n_fdm = int(n_fdm)
        self._n_grid = int(n_grid)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._injections = tuple(injections)
        self._cube_cadence_s = float(cube_cadence_s)
        self._time_shift_table = compute_time_shift_search(
            coarse_dm_pc_cm3=coarse_dm_pc_cm3,
            fine_dm_pc_cm3=fine_dm_pc_cm3,
            fine_to_coarse=fine_to_coarse,
            t_int_search_us=t_int_search_us,
        )
        # T_stream covers the cube + the worst-case shift, so the
        # combiner can safely read [t - shift] for every (t, fdm) pair.
        self._max_shift = int(self._time_shift_table.shifts.max(initial=0))
        self._t_stream = self._t_det + self._max_shift
        self._cubes_emitted = 0
        self._started = False
        self._stopped = False
        # Bench-only chunk-8b RX-ring emulation: pre-quantise one cube
        # of cf32 streams to cint8 once and yield the same cached cint8
        # stack on every iteration. Lets the throughput bench measure
        # the GPU pipeline cost in isolation from host-side bench
        # scaffolding (synthetic source generation + cf -> cint8
        # quantise). Production M3 RX-ring delivers cint8 already.
        self._pre_quantise = bool(pre_quantise)
        self._prequantise_target_max = int(prequantise_target_max)
        self._cached_cint8: Optional[np.ndarray] = None
        self._cached_streams: Optional[Mapping[int, np.ndarray]] = None

    @property
    def time_shift_table(self) -> TimeShiftSearchTable:
        return self._time_shift_table

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._stopped = True

    async def release(self, cube_id: int) -> None:
        # Synthetic source allocates per-cube; nothing to release.
        return None

    def _gen_cube_streams(self, cube_idx: int) -> Mapping[int, np.ndarray]:
        """Per-chgroup ``[T_stream, N_grid, N_grid] complex64`` streams.

        Each cell is iid complex Gaussian (σ_re = σ_im = 1/√2 so the
        magnitude has unit variance per component pair). Injections
        for ``cube_idx == self._cubes_emitted`` are splatted on top of
        chgroup 15 (the reference, no shift) at the requested cell —
        this is a minimal injection for latency/throughput benching.
        """
        streams: dict[int, np.ndarray] = {}
        shape = (self._t_stream, self._n_grid, self._n_grid)
        for g in range(N_CHGROUP):
            re = self._rng.standard_normal(shape).astype(np.float32) * (1.0 / np.sqrt(2.0))
            im = self._rng.standard_normal(shape).astype(np.float32) * (1.0 / np.sqrt(2.0))
            streams[g] = (re + 1j * im).astype(np.complex64)
        for inj in self._injections:
            if inj.cube_idx != cube_idx:
                continue
            if not (0 <= inj.t_in_cube < self._t_det):
                continue
            if not (0 <= inj.l_pix < self._n_grid):
                continue
            if not (0 <= inj.m_pix < self._n_grid):
                continue
            streams[N_CHGROUP - 1][inj.t_in_cube, inj.l_pix, inj.m_pix] += np.complex64(
                inj.amplitude + 0.0j
            )
        return streams

    async def __aenter__(self) -> "SyntheticRxRingSource":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def __aiter__(self) -> AsyncIterator[CubeRingSlot]:
        if not self._started:
            await self.start()
        while self._cubes_emitted < self._n_cubes and not self._stopped:
            cube_idx = self._cubes_emitted
            if self._pre_quantise:
                # Generate one canonical cube + quantise once; reuse for
                # every cube. Emulates the chunk-8b RX-ring contract
                # (M3 emits cint8 already; search-node never re-quantises).
                if self._cached_cint8 is None:
                    from ..transport.quantize import (
                        quantise_per_chgroup_into_cint8,
                    )
                    self._cached_streams = self._gen_cube_streams(0)
                    self._cached_cint8 = np.empty(
                        (
                            N_CHGROUP, self._t_stream, 2,
                            self._n_grid, self._n_grid,
                        ),
                        dtype=np.int8,
                    )
                    quantise_per_chgroup_into_cint8(
                        self._cached_streams,
                        out_cint8=self._cached_cint8,
                        target_max=self._prequantise_target_max,
                        zero_fill_missing=True,
                    )
                streams = self._cached_streams
                cint8_stack = self._cached_cint8
            else:
                streams = self._gen_cube_streams(cube_idx)
                cint8_stack = None
            slot = CubeRingSlot(
                cube_id=cube_idx,
                specnum_start=cube_idx * self._t_det,
                per_chgroup_streams=streams,
                time_shift_table=self._time_shift_table,
                validity_mask=np.ones(
                    (self._t_det, self._n_fdm), dtype=np.bool_
                ),
                n_fdm_in_cube=self._n_fdm,
                t_det=self._t_det,
                n_grid=self._n_grid,
                per_chgroup_cint8_stack=cint8_stack,
            )
            self._cubes_emitted += 1
            yield slot
            if self._cube_cadence_s > 0:
                await asyncio.sleep(self._cube_cadence_s)
