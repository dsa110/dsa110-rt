"""Async TX: off-load encode + ``sendto`` to per-DM worker subprocesses.

Production async path for the M7.2 corner-turn: the corr-side
GPU-pipeline process splits each multi-DM cube along its DM axis,
hands each DM-slice to a worker subprocess via :class:`CubeShmRing`,
and immediately returns control to the next-block GPU compute. The
worker subprocess (one per DM-split) owns its own :class:`TransportTx`
(prod-frame mode, cint8, MTU fragmentation, token-bucket pacing) and
performs the encode + ``sendto`` loop without blocking the GPU
pipeline.

Why subprocesses and not threads:
    The Python GIL serialises numpy encode work and the ``sendto``
    syscall path. A single Python thread sustains ≈ 0.41 Gb/s at the
    prod op-point (M4a doc :file:`docs/m4a/prod_rate_findings.md`).
    The per-corr egress at the M7.2 production op-point (N=8 DM trials,
    32× integration) is ≈ 1.05 Gb/s. Adding worker threads in-process
    does not scale past one core's worth of work; subprocesses do.

Architecture::

    ┌──────────────────────────────────────────────────────────────┐
    │ corr_fast (1 process)                                        │
    │                                                              │
    │  process_block:                                              │
    │    GPU: unpack → corr → grid → multi_dm → static_sky         │
    │    Stage2FIFO.push  →  cube (torch.Tensor on GPU)            │
    │    AsyncTransportTx.transmit:                                │
    │      D2H cube (~10 ms PCIe at N=8)                           │
    │      for w in workers:                                       │
    │        ring[w].reserve_slot()                                │
    │        ring[w].copy_to_slot(cube[w.dm_slice])                │
    │        ring[w].publish_slot()                                │
    │      return immediately   ←── TX overlaps with next block    │
    └──────────────────────────────────────────────────────────────┘
              │              │              │              │
              ▼              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ worker 0     │ │ worker 1     │ │ worker 2     │ │ worker 3     │
    │ DM[0:2]      │ │ DM[2:4]      │ │ DM[4:6]      │ │ DM[6:8]      │
    │              │ │              │ │              │ │              │
    │ TransportTx  │ │ TransportTx  │ │ TransportTx  │ │ TransportTx  │
    │  prod-frame  │ │  prod-frame  │ │  prod-frame  │ │  prod-frame  │
    │  cint8 +     │ │  cint8 +     │ │  cint8 +     │ │  cint8 +     │
    │  fragmenter  │ │  fragmenter  │ │  fragmenter  │ │  fragmenter  │
    │  + bucket    │ │  + bucket    │ │  + bucket    │ │  + bucket    │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           └─────────────── 40 GbE NIC ──────────────────────┘
                                  │
                                  ▼  (one search node receives all flows)

DM-split convention:
    With ``n_workers=W`` and ``n_dm_total=N``, worker ``w`` owns the
    contiguous DM-axis slice ``range(w*N//W, (w+1)*N//W)``. This is
    forward-compatible to M7.3 4-search-node fan-out — there the
    split is across SEARCH NODES not workers, and the production
    pattern is exactly ``n_workers = 4`` per corr node, one worker per
    search-node destination.

Lifecycle:
    Construct via :func:`AsyncTransportTx.spawn` (factory) which
    allocates shm rings + spawns workers. Use :meth:`transmit` on the
    hot path; call :meth:`close` on shutdown (sends poison pills, joins
    workers, unlinks shm). Idempotent close + signal-safe.

Failure modes:
    - Worker crash: the shm ring's reserve_slot eventually times out;
      :meth:`transmit` re-raises :class:`TxRingBackpressureError`,
      which the corr_fast supervisor logs + treats as a fatal stop
      (the corr-side cannot proceed if egress is stuck).
    - Producer crash: workers' ``wait_slot`` times out periodically;
      they exit on poison pill OR on a sustained idle window > 30 s
      (configurable via ``worker_idle_exit_s``).
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from dsart.transport.tx_ring import (
    CubeShmRing,
    CubeShmRingDims,
    SlotMeta,
    TxRingBackpressureError,
)

LOG = logging.getLogger("dsart.transport.async_tx")


# ---------------------------------------------------------------------------
# Worker config + entrypoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AsyncTxWorkerCfg:
    """Pickleable config for a TX worker subprocess.

    Carries everything the worker needs to construct its TransportTx
    + attach to its shm ring without touching the main process again.
    """

    worker_idx: int
    n_workers: int
    dm_lo: int
    dm_hi: int  # half-open
    shm_name: str
    ring_dims: CubeShmRingDims
    host: str
    port: int
    chgroup: int
    target_gbps_per_flow: float
    pattern_id: int
    n_grid: int
    bucket_fifo_depth: int = 4
    pacer_headroom: float = 1.05
    t_int_factor: int = 1
    corr_idx: int = 0
    worker_idle_exit_s: float = 300.0
    """If wait_slot returns None continuously for this many seconds
    AFTER at least one cube has been processed, the worker assumes
    the producer is gone and exits. The "after at least one cube"
    gate handles slow producer startups (Triton JIT can take 30 s
    before the first cube emerges). The 5 min default leaves plenty
    of margin past the producer's wall-time bring-up while still
    cleaning up if the producer crashes silently."""
    log_level: str = "INFO"


def _async_tx_worker_main(
    cfg: _AsyncTxWorkerCfg,
    ready_q: mp.Queue,
    done_q: mp.Queue,
    stats_q: mp.Queue,
) -> None:
    """Subprocess entrypoint. Owns one :class:`TransportTx` + one
    :class:`CubeShmRing` consumer side. Loops::

        while True:
            meta = ring.wait_slot(timeout_s=0.5)
            if meta is None:                       # poison pill or idle
                if poisoned or idle_exceeded:
                    break
                continue
            cube = ring.view_slot(meta.slot_idx)
            tx.transmit([torch.from_numpy(cube[:meta.n_dm, :meta.n_fv])],
                        block_n=meta.block_n,
                        rfi_warming_up=meta.rfi_warming_up,
                        specnum=meta.specnum)
            ring.release_slot(meta.slot_idx)
    """
    # Configure logging in the child (spawn-start has no inherited handlers).
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format=(
            f"%(asctime)s %(levelname)s "
            f"async_tx.w{cfg.worker_idx} %(message)s"
        ),
    )
    log = logging.getLogger(f"dsart.transport.async_tx.w{cfg.worker_idx}")

    # Block SIGINT in the worker: the parent's signal handler issues a
    # graceful shutdown via poison pill on the queue. Without this the
    # child sees the same SIGINT and exits before draining the queue,
    # leaking shm + queue state.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

    # Reimport here — spawn start re-executes module init in the child.
    from dsart.transport.prod_frame import BITS_CINT8_COMPLEX
    from dsart.transport.tx import TransportTx, TransportTxProdConfig

    prod_cfg = TransportTxProdConfig(
        target_gbps_per_flow=cfg.target_gbps_per_flow,
        pacer_headroom=cfg.pacer_headroom,
        bits_per_cell=BITS_CINT8_COMPLEX,
        t_int_factor=cfg.t_int_factor,
        corr_idx=cfg.corr_idx,
        bucket_fifo_depth=cfg.bucket_fifo_depth,
    )
    tx = TransportTx(
        host=cfg.host,
        port=cfg.port,
        chgroup=int(cfg.chgroup),
        use_prod_frame=True,
        prod_config=prod_cfg,
    )
    tx.prepare_prod(
        pattern_id_by_chgroup={int(cfg.chgroup): int(cfg.pattern_id)},
        n_grid=int(cfg.n_grid),
    )
    log.info(
        "TransportTx ready: %s:%d chgroup=%d dm[%d:%d) gbps/flow=%.3f "
        "n_grid=%d pattern_id=%d",
        cfg.host, cfg.port, cfg.chgroup, cfg.dm_lo, cfg.dm_hi,
        cfg.target_gbps_per_flow, cfg.n_grid, cfg.pattern_id,
    )

    ring = CubeShmRing(
        name=cfg.shm_name,
        dims=cfg.ring_dims,
        ready_q=ready_q,
        done_q=done_q,
        owner=False,
    )

    n_cubes = 0
    n_frames = 0
    n_drops_at_start = int(tx.tx_dropped_payloads)
    t_start = time.monotonic()
    idle_since: float | None = None
    encode_ms_samples: list[float] = []

    try:
        while True:
            meta: SlotMeta | None = ring.wait_slot(timeout_s=0.5)
            if meta is None:
                # wait_slot returns None for two cases:
                #   (a) ready_q.get timed out (no cubes pending)
                #   (b) poison pill posted by the producer
                # CubeShmRing.poisoned latches on (b) so we can
                # distinguish here.
                if ring.poisoned:
                    log.info("poison pill received; exiting")
                    break
                # Idle-timeout exit. Only fires AFTER at least one
                # cube has been processed; this prevents the worker
                # from exiting while the producer is still in its
                # Triton-JIT warm-up window (can be 30 s on first
                # spawn). Once the producer has demonstrated liveness
                # by sending at least one cube, a sustained idle
                # window is interpreted as "producer crashed" and the
                # worker exits cleanly.
                if n_cubes == 0:
                    continue
                if idle_since is None:
                    idle_since = time.monotonic()
                elif (time.monotonic() - idle_since) > cfg.worker_idle_exit_s:
                    log.info(
                        "idle > %.1fs after %d cubes; exiting "
                        "(producer likely crashed)",
                        cfg.worker_idle_exit_s, n_cubes,
                    )
                    break
                continue
            idle_since = None

            t_encode_start = time.monotonic()
            cube_view = ring.view_slot(meta.slot_idx)
            # Build a TX cube whose shape matches the logical cube the
            # producer copied in. ``ring_dims.shape`` is the *upper-
            # bound* shape; ``meta.n_dm/n_fv/n_filled`` are the actual
            # populated extents for this block (only meta.n_filled may
            # ever change in steady state; the producer pre-pads to
            # ring_dims for the others).
            logical = cube_view[:meta.n_dm, :meta.n_fv, :meta.n_filled]
            # TransportTx.transmit expects torch.Tensor (complex). The
            # from_numpy view shares memory with the shm slot; no copy.
            tx_input = torch.from_numpy(logical)
            try:
                n_sent = tx.transmit(
                    [tx_input],
                    block_n=int(meta.block_n),
                    rfi_warming_up=bool(meta.rfi_warming_up),
                    specnum=int(meta.specnum),
                )
            except Exception:
                log.exception(
                    "tx.transmit raised on block_n=%d", meta.block_n,
                )
                n_sent = 0
            t_encode_ms = (time.monotonic() - t_encode_start) * 1e3
            encode_ms_samples.append(t_encode_ms)

            # Release BEFORE the next dequeue: the producer is free to
            # reuse the slot now (the TX data is already on the wire).
            ring.release_slot(meta.slot_idx)

            n_cubes += 1
            n_frames += int(n_sent)
            if n_cubes % 512 == 0:
                if encode_ms_samples:
                    enc_p50 = float(np.percentile(encode_ms_samples, 50))
                    enc_p99 = float(np.percentile(encode_ms_samples, 99))
                    enc_max = float(np.max(encode_ms_samples))
                else:
                    enc_p50 = enc_p99 = enc_max = 0.0
                log.info(
                    "n_cubes=%d n_frames=%d drops=%d "
                    "encode_ms p50=%.2f p99=%.2f max=%.2f",
                    n_cubes, n_frames,
                    int(tx.tx_dropped_payloads) - n_drops_at_start,
                    enc_p50, enc_p99, enc_max,
                )
                encode_ms_samples.clear()
    finally:
        elapsed = time.monotonic() - t_start
        try:
            stats_q.put_nowait({
                "worker_idx": cfg.worker_idx,
                "n_cubes": int(n_cubes),
                "n_frames": int(n_frames),
                "elapsed_s": float(elapsed),
                "tx_dropped_payloads": int(
                    tx.tx_dropped_payloads - n_drops_at_start
                ),
            })
        except Exception:  # pragma: no cover — shutdown path
            log.warning("failed to post final stats")
        try:
            ring.close()
        except Exception:  # pragma: no cover
            log.exception("ring.close() raised")
        try:
            tx.close()
        except Exception:  # pragma: no cover
            log.exception("tx.close() raised")
        log.info(
            "exiting: n_cubes=%d n_frames=%d elapsed=%.1fs",
            n_cubes, n_frames, elapsed,
        )


# ---------------------------------------------------------------------------
# AsyncTransportTx — main-process façade
# ---------------------------------------------------------------------------


@dataclass
class AsyncTransportTxConfig:
    """Construction-time configuration for :class:`AsyncTransportTx`.

    Mirrors the fields of :class:`TransportTxProdConfig` that the
    workers need, plus async-specific fields (worker count, ring
    dims, destination tuple).
    """

    host: str
    port: int
    chgroup: int
    n_workers: int
    n_dm_total: int
    ring_dims: CubeShmRingDims
    pattern_id: int
    n_grid: int
    target_gbps_per_flow: float = 0.073
    pacer_headroom: float = 1.05
    bucket_fifo_depth: int = 4
    t_int_factor: int = 1
    corr_idx: int = 0
    reserve_timeout_s: float = 1.0
    """How long :meth:`transmit` will wait for a worker to free a slot
    before raising TxRingBackpressureError. 1 s ≈ 7 cube periods."""
    queue_max_size: int = 32
    """mp.Queue maxsize. Should be ≥ n_slots; 32 is plenty for the
    default 4-slot ring."""
    log_level: str = "INFO"
    worker_idle_exit_s: float = 30.0
    shm_name_prefix: str = "dsart-corr-tx"


@dataclass
class _WorkerHandle:
    """Bookkeeping per worker subprocess (main-process side)."""

    proc: mp.Process
    ready_q: mp.Queue
    done_q: mp.Queue
    stats_q: mp.Queue
    ring: CubeShmRing
    dm_lo: int
    dm_hi: int  # half-open
    cfg: _AsyncTxWorkerCfg

    @property
    def alive(self) -> bool:
        return self.proc.is_alive()


class AsyncTransportTx:
    """Main-process façade for the per-DM TX worker subprocess pool.

    Constructed via :meth:`spawn`. Hot-path API is :meth:`transmit`,
    matching the :class:`TransportTxStage` Protocol signature so it
    drops in behind a thin adapter (see
    :class:`dsart.services.corr_fast_integration._AsyncTransportTxAdapter`).

    Counters (mon-key surface):
        n_cubes_in           — cubes accepted by :meth:`transmit`
        n_cubes_per_worker   — list[int], cubes routed to each worker
        n_backpressure       — total backpressure events across rings
        n_workers_alive      — current worker subprocess count
    """

    __slots__ = (
        "_cfg",
        "_workers",
        "_dm_splits",
        "_closed",
        "_corr_idx",
        "_block_specnum_inflight",
        "n_cubes_in",
    )

    def __init__(
        self,
        cfg: AsyncTransportTxConfig,
        workers: list[_WorkerHandle],
        dm_splits: list[tuple[int, int]],
    ) -> None:
        self._cfg = cfg
        self._workers = workers
        self._dm_splits = dm_splits
        self._closed: bool = False
        self._corr_idx: int = int(cfg.corr_idx)
        self.n_cubes_in: int = 0
        # _block_specnum_inflight: per-worker count of cubes still being
        # consumed. Not strictly needed (the rings have their own count)
        # but useful for the mon-key surface.
        self._block_specnum_inflight: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def spawn(
        cls, cfg: AsyncTransportTxConfig, *, start_method: str = "spawn",
    ) -> "AsyncTransportTx":
        """Allocate the shm rings + queues, spawn the worker subprocesses,
        and return a ready-to-use AsyncTransportTx.

        Args:
            cfg: :class:`AsyncTransportTxConfig`.
            start_method: ``multiprocessing.get_context()`` argument.
                Default ``"spawn"`` to avoid inheriting torch / CUDA
                state.
        """
        if cfg.n_workers < 1:
            raise ValueError(f"n_workers={cfg.n_workers} must be >= 1")
        if cfg.n_dm_total < cfg.n_workers:
            raise ValueError(
                f"n_dm_total={cfg.n_dm_total} < n_workers={cfg.n_workers}; "
                f"each worker needs at least one DM trial"
            )
        if cfg.n_dm_total % cfg.n_workers != 0:
            LOG.warning(
                "AsyncTransportTx: n_dm_total=%d not divisible by "
                "n_workers=%d; trailing workers will get extra trials",
                cfg.n_dm_total, cfg.n_workers,
            )

        dm_splits: list[tuple[int, int]] = []
        for w in range(cfg.n_workers):
            lo = (w * cfg.n_dm_total) // cfg.n_workers
            hi = ((w + 1) * cfg.n_dm_total) // cfg.n_workers
            if w == cfg.n_workers - 1:
                hi = cfg.n_dm_total
            dm_splits.append((lo, hi))

        # Sanity-check ring_dims against the per-worker dm count: the
        # ring's leading axis must be >= max(per-worker dm count). If
        # the caller passed a uniform ring_dims (typical), we accept it
        # iff it covers the longest slice.
        max_dm_per_worker = max(hi - lo for lo, hi in dm_splits)
        if cfg.ring_dims.shape[0] < max_dm_per_worker:
            raise ValueError(
                f"ring_dims.shape[0]={cfg.ring_dims.shape[0]} < "
                f"max dm-per-worker = {max_dm_per_worker}"
            )

        ctx = mp.get_context(start_method)
        workers: list[_WorkerHandle] = []
        pid = os.getpid()
        try:
            for w, (lo, hi) in enumerate(dm_splits):
                shm_name = (
                    f"{cfg.shm_name_prefix}-{cfg.corr_idx}"
                    f"-pid{pid}-w{w}"
                )
                ready_q: mp.Queue = ctx.Queue(maxsize=cfg.queue_max_size)
                done_q: mp.Queue = ctx.Queue(maxsize=cfg.queue_max_size)
                stats_q: mp.Queue = ctx.Queue(maxsize=4)

                ring = CubeShmRing(
                    name=shm_name,
                    dims=cfg.ring_dims,
                    ready_q=ready_q,
                    done_q=done_q,
                    owner=True,
                )
                w_cfg = _AsyncTxWorkerCfg(
                    worker_idx=w,
                    n_workers=cfg.n_workers,
                    dm_lo=lo,
                    dm_hi=hi,
                    shm_name=shm_name,
                    ring_dims=cfg.ring_dims,
                    host=cfg.host,
                    port=cfg.port,
                    chgroup=cfg.chgroup,
                    target_gbps_per_flow=cfg.target_gbps_per_flow,
                    pattern_id=cfg.pattern_id,
                    n_grid=cfg.n_grid,
                    bucket_fifo_depth=cfg.bucket_fifo_depth,
                    pacer_headroom=cfg.pacer_headroom,
                    t_int_factor=cfg.t_int_factor,
                    corr_idx=cfg.corr_idx,
                    worker_idle_exit_s=cfg.worker_idle_exit_s,
                    log_level=cfg.log_level,
                )
                proc = ctx.Process(
                    target=_async_tx_worker_main,
                    args=(w_cfg, ready_q, done_q, stats_q),
                    name=f"async-tx-corr{cfg.corr_idx}-w{w}",
                    daemon=False,
                )
                proc.start()
                workers.append(
                    _WorkerHandle(
                        proc=proc,
                        ready_q=ready_q,
                        done_q=done_q,
                        stats_q=stats_q,
                        ring=ring,
                        dm_lo=lo,
                        dm_hi=hi,
                        cfg=w_cfg,
                    )
                )
            LOG.info(
                "AsyncTransportTx spawned: corr_idx=%d n_workers=%d "
                "dm_split=%s host=%s port=%d gbps/flow=%.3f",
                cfg.corr_idx, cfg.n_workers, dm_splits,
                cfg.host, cfg.port, cfg.target_gbps_per_flow,
            )
        except Exception:
            # Tear down any workers we did spawn.
            for wh in workers:
                try:
                    wh.ring.signal_worker_exit()
                except Exception:
                    pass
                try:
                    wh.proc.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    wh.ring.close()
                except Exception:
                    pass
            raise

        return cls(cfg, workers, dm_splits)

    # ------------------------------------------------------------------
    # Hot-path API
    # ------------------------------------------------------------------

    def transmit(
        self,
        cubes_for_tx: Sequence[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
        specnum: int | None = None,
    ) -> int:
        """Off-load encode + ``sendto`` of each cube to the worker pool.

        ``cubes_for_tx`` is the list returned by ``Stage2FIFO.push``.
        Each cube is a sparse-COO complex torch.Tensor of shape
        ``(n_dm_total, n_fv, n_filled)``. The DM axis is split across
        ``n_workers`` workers; each worker gets the slice
        ``cube[dm_lo:dm_hi]`` (contiguous along DM).

        The cube is D2H-copied **once** here (under the assumption
        ``cube`` is on GPU) into a host-side contiguous numpy buffer;
        then each per-worker slice is :func:`numpy.copyto`-d into the
        worker's shm slot. With a complex64 cube at the N=8 op-point
        (≈ 32 MiB total), D2H runs ~10 ms and the slot copies are
        ~1 ms each — well under the 134 ms block period.

        Returns 0 (the actual frame count is in the workers' stats; the
        chunk-4 IntegrationOutput.n_tx field becomes a lower-bound
        counter at this point — it accounts for cubes HANDED OFF, not
        cubes already on the wire).
        """
        if self._closed:
            raise RuntimeError("AsyncTransportTx.transmit on closed instance")
        if not cubes_for_tx:
            return 0
        if specnum is None:
            raise ValueError(
                "AsyncTransportTx.transmit requires specnum (prod-frame path)"
            )

        n_cubes_handed_off = 0
        for cube in cubes_for_tx:
            if not isinstance(cube, torch.Tensor):
                raise TypeError(
                    f"AsyncTransportTx.transmit: cube must be torch.Tensor, "
                    f"got {type(cube).__name__}"
                )
            if cube.ndim != 3:
                raise ValueError(
                    f"AsyncTransportTx.transmit: cube.ndim={cube.ndim} "
                    f"(expected 3 = (N_DM, n_fv, N_filled)); image-cube "
                    f"path not supported here"
                )
            n_dm, n_fv, n_filled = cube.shape
            if n_dm != self._cfg.n_dm_total:
                raise ValueError(
                    f"AsyncTransportTx.transmit: cube N_DM={n_dm} != "
                    f"cfg.n_dm_total={self._cfg.n_dm_total}"
                )
            if (
                n_fv > self._cfg.ring_dims.shape[1]
                or n_filled > self._cfg.ring_dims.shape[2]
            ):
                raise ValueError(
                    f"AsyncTransportTx.transmit: cube shape ({n_dm}, {n_fv}, "
                    f"{n_filled}) exceeds ring_dims.shape="
                    f"{self._cfg.ring_dims.shape}"
                )

            # Single D2H copy for the whole cube. Use .numpy() if
            # already on CPU; otherwise .cpu().numpy().
            cube_cpu_t = cube.detach().to("cpu", copy=False).contiguous()
            cube_np = cube_cpu_t.numpy()  # zero-copy view onto host

            ring_dm = self._cfg.ring_dims.shape[0]
            for w, wh in enumerate(self._workers):
                lo, hi = wh.dm_lo, wh.dm_hi
                slice_np = cube_np[lo:hi]
                # Pad to ring_dims.shape if this slice is smaller along
                # the DM axis (i.e. the last worker absorbs the remainder
                # of an uneven split). We always copy into a fixed-size
                # ring slot to keep the layout uniform; the worker reads
                # only the first ``meta.n_dm`` rows.
                slot_idx = wh.ring.reserve_slot(
                    timeout_s=self._cfg.reserve_timeout_s,
                )
                if slice_np.shape == self._cfg.ring_dims.shape:
                    wh.ring.copy_to_slot(slot_idx, slice_np)
                else:
                    # Build a padded view (zeros in unused rows / cols).
                    # Cheap allocation per block (~32 MiB / W); for
                    # production W=4 this is 8 MiB. Avoid by sizing
                    # ring_dims exactly to the largest per-worker slice.
                    pad = np.zeros(
                        self._cfg.ring_dims.shape,
                        dtype=self._cfg.ring_dims.dtype,
                    )
                    pad[: slice_np.shape[0], : slice_np.shape[1],
                        : slice_np.shape[2]] = slice_np
                    wh.ring.copy_to_slot(slot_idx, pad)
                wh.ring.publish_slot(
                    slot_idx,
                    block_n=int(block_n),
                    specnum=int(specnum),
                    n_dm=int(hi - lo),
                    n_fv=int(n_fv),
                    n_filled=int(n_filled),
                    rfi_warming_up=bool(rfi_warming_up),
                )
            n_cubes_handed_off += 1
            self.n_cubes_in += 1

        return n_cubes_handed_off

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self, *, join_timeout_s: float = 5.0) -> None:
        """Signal workers to exit, join them, unlink shm + queues.

        Idempotent. Logs final per-worker stats from ``stats_q``.
        """
        if self._closed:
            return
        self._closed = True
        LOG.info("AsyncTransportTx.close: signalling %d workers", len(self._workers))
        for wh in self._workers:
            try:
                wh.ring.signal_worker_exit()
            except Exception:  # pragma: no cover
                LOG.exception("signal_worker_exit raised")

        deadline = time.monotonic() + float(join_timeout_s)
        for wh in self._workers:
            remaining = max(0.05, deadline - time.monotonic())
            wh.proc.join(timeout=remaining)
            if wh.proc.is_alive():
                LOG.warning(
                    "worker %d did not exit within %.1fs; terminating",
                    wh.cfg.worker_idx, join_timeout_s,
                )
                wh.proc.terminate()
                wh.proc.join(timeout=2.0)
            # Drain stats_q.
            try:
                while True:
                    stats = wh.stats_q.get_nowait()
                    LOG.info("worker stats: %s", stats)
            except Exception:
                pass
        for wh in self._workers:
            try:
                wh.ring.close()
            except Exception:  # pragma: no cover
                LOG.exception("ring.close() raised on worker %d", wh.cfg.worker_idx)
        LOG.info("AsyncTransportTx.close: done")

    def __enter__(self) -> "AsyncTransportTx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_workers(self) -> int:
        return len(self._workers)

    @property
    def n_workers_alive(self) -> int:
        return sum(1 for wh in self._workers if wh.alive)

    def dm_split(self, worker_idx: int) -> tuple[int, int]:
        return self._dm_splits[worker_idx]

    def stats(self) -> dict[str, Any]:
        ring_stats = [wh.ring.stats() for wh in self._workers]
        return {
            "n_cubes_in": int(self.n_cubes_in),
            "n_workers": self.n_workers,
            "n_workers_alive": self.n_workers_alive,
            "dm_splits": list(self._dm_splits),
            "rings": ring_stats,
            "n_backpressure_total": sum(
                int(r["n_backpressure"]) for r in ring_stats
            ),
        }
