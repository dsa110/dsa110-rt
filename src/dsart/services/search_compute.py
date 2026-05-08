"""Search-node compute service entry (RX ring → detector → cluster → dump).

M5 chunk 6b-α + M6 chunk 5: long-running asyncio orchestrator that
drives the ``CubePipeline`` from a pluggable ``RxRingSource``, then
hands per-cube candidates to:

  * the M6 :class:`ClustererService` (HDBSCAN/DBSCAN; ThreadPoolExecutor);
  * the M6 :class:`BrightPulsePredicate` + :class:`CubeDumpWriter`
    (bright-pulse cluster predicate → NPZ dump on a writer thread);
  * the M6 :class:`UdpTriggerListener` (one-shot "dump next cube" flag);
  * the M6 :class:`CandsLogger` (T1 per-candidate + T2 per-cluster
    hourly-rotated ASCII rows).

All four sub-systems are *optional*. If their config is ``None`` the
service still processes cubes (for unit-test wiring + the M5-only
detector path). Production wires all four.

Per-cube driver order (M6 D7 / D9 / chunk 5):

  1. ``slot = await source.acquire()``
  2. ``geom = self._geom_from_slot(slot)``
  3. ``udp_armed = listener.consume_dump_next_cube_flag()``  ← BEFORE
     pipeline.process so we can decide whether to dump cube N before
     committing CPU/GPU work that may release the cube tensor.
  4. ``result = pipeline.process(slot)``
  5. If ``udp_armed``: ``cube_dump.submit(result.cube, manifest_udp)``
  6. ``cluster_result = clusterer.submit(...).result()``  ← blocks
     up to a configurable timeout (default = cube cadence). The chunk-6
     bench (`bench/clusterer_throughput.py`) gates on the p99 budget so
     this should never time out at production load.
  7. For each cluster record: if predicate fires, submit auto dump.
  8. ``cands_logger.write_cube(...)`` writes T1 + T2 rows.
  9. ``await source.release(slot.cube_id)``.

Production path runs ``main()`` from a CLI; benches subclass / wire
the service directly to inject a synthetic source. Per M-defer (plan
§M-defer) the M5 trigger emitter is retired; voltage triggering is
delegated to the corr-side ``dsa110-xengine`` framework.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..cluster import (
    CandsLogger,
    CandsLoggerConfig,
    ClustererConfig,
    ClustererService,
)
from ..common.contracts import Candidate, CubeGeometry
from ..detector.forward import DeterministicDetector
from ..dump import (
    BrightPulsePredicate,
    BrightPulsePredicateConfig,
    CubeDumpWriter,
    CubeDumpWriterConfig,
    UdpTriggerListener,
    UdpTriggerListenerConfig,
)
from ..common.contracts import CubeDumpManifest
from ..noise_norm.layer1 import Layer1State
from .cube_pipeline import CubePipeline, CubePipelineConfig
from .rx_ring import CubeRingSlot, RxRingSource

__all__ = [
    "SearchComputeConfig",
    "SearchComputeService",
    "main",
]


_LOG = logging.getLogger("dsart.services.search_compute")


# ---------------------------------------------------------------------------
# Service config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchComputeConfig:
    """Static (per-process-lifetime) service config (plan §9 O-4 +
    ``configs/config_compute_search.yaml`` + M6 chunks 1-5 sub-system
    configs).

    The M5 chunk-6b-α scope ships fields needed by the pipeline +
    Layer-1 state. The M6 chunk-5 wiring adds optional sub-system
    configs (clusterer, cube_dump, udp_listener, cands_logger) and the
    cube-geometry hyperparameters needed to construct a
    :class:`CubeGeometry` per cube.

    Sub-system gating: each ``*_config`` field defaults to ``None``.
    If ``None`` the corresponding sub-system is not instantiated and
    its per-cube hook is a no-op. Tests that only exercise the
    detector path leave them all None; production wires all four.

    Args:
        pipeline: ``CubePipelineConfig`` (M5 chunk 6a).
        n_fdm: full-cube fine-DM count (M5 chunk 6b).
        detector_*: detector knobs (M5 chunk 4-6).
        layer1_*: Layer-1 σ-clip knobs (M5 chunk 3).
        search_node_id, gpu_half: process identity.
        cube_cell_l_rad, cube_cell_m_rad: gridder cell pitch in radians
            (production: derived from λ / (n_grid · cell_λ); for v1
            we let the operator pin a constant per-deployment).
        cube_l0_rad, cube_m0_rad: (l, m) at pixel index 0. Default 0.0.
        cube_sample_period_us: time between adjacent samples in µs
            (defaults to ``T_INT_SEARCH_US_DEFAULT`` from constants).
        cube_sample_period_specnum: spec-nums per detector sample
            (default 16 = ``t_int_search_us / t_int_fast_us`` at
            production ops point).
        mjd_at_specnum_0: MJD at specnum 0 of this run; ``mjd_start``
            for cube N is computed as ``mjd_at_specnum_0 +
            (specnum_start * t_int_fast_us / 1e6 / 86400)``. v1
            placeholder is 0.0; production must override.
        fine_dm_pc_cc_full: full fine-DM grid (``DmPlan.fine_dm``);
            required if ``clusterer_config`` is set so the per-cube
            slice can be built. ``None`` defers to a synthetic linear
            grid (test convenience).
        clusterer_config / cube_dump_writer_config /
        bright_pulse_predicate_config / udp_trigger_listener_config /
        cands_logger_config: M6 chunk 1-4 sub-system configs.
        cluster_result_timeout_s: max seconds to block on the cluster
            future per cube. Default 1.0 s (well above the chunk-6
            bench's p99 = 50 ms gate).
    """

    pipeline: CubePipelineConfig
    n_fdm: int
    detector_threshold_sigma: float = 8.0
    detector_dtype: torch.dtype = torch.float16
    detector_device: str = "cpu"
    detector_version: str = "v1.M5"
    detector_streaming: bool = True
    detector_streaming_tile_size: int = 64
    search_node_id: int = 1
    gpu_half: int = 1
    layer1_n_burnin_cubes: int = 5
    layer1_n_sigma: float = 3.0
    layer1_n_iterations: int = 3

    # --- M6 chunk 5: cube-geometry hyperparameters ---------------------
    cube_cell_l_rad: float = 1.5e-4
    cube_cell_m_rad: float = 1.5e-4
    cube_l0_rad: float = 0.0
    cube_m0_rad: float = 0.0
    cube_sample_period_us: float = 131.072
    cube_sample_period_specnum: int = 16
    mjd_at_specnum_0: float = 0.0
    fine_dm_pc_cc_full: Optional[np.ndarray] = None

    # --- M6 chunk 5: sub-system configs (all optional) ----------------
    clusterer_config: Optional[ClustererConfig] = None
    cube_dump_writer_config: Optional[CubeDumpWriterConfig] = None
    bright_pulse_predicate_config: Optional[BrightPulsePredicateConfig] = None
    udp_trigger_listener_config: Optional[UdpTriggerListenerConfig] = None
    cands_logger_config: Optional[CandsLoggerConfig] = None

    # --- M6 chunk 5: orchestration knobs ------------------------------
    cluster_result_timeout_s: float = 1.0


# ---------------------------------------------------------------------------
# SearchComputeService
# ---------------------------------------------------------------------------


class SearchComputeService:
    """Asyncio orchestrator. One service instance per (search-node, GPU-half).

    Args:
        config: ``SearchComputeConfig`` resolved from yaml + CLI.
        source: any ``RxRingSource``; M4a in production, synthetic in
            benches.
        detector: optional pre-constructed ``DeterministicDetector``
            (lets benches inject a seeded detector). Default: build
            one from ``config``.
        layer1_state: optional pre-constructed ``Layer1State`` (same).
    """

    def __init__(
        self,
        config: SearchComputeConfig,
        source: RxRingSource,
        *,
        detector: Optional[DeterministicDetector] = None,
        layer1_state: Optional[Layer1State] = None,
    ) -> None:
        self._config = config
        self._source = source
        self._detector = detector or self._build_detector(config)
        self._layer1_state = layer1_state or Layer1State(
            n_fdm=config.n_fdm,
            n_burnin_cubes=config.layer1_n_burnin_cubes,
            n_sigma=config.layer1_n_sigma,
            n_iterations=config.layer1_n_iterations,
        )
        self._pipeline = CubePipeline(
            config=config.pipeline,
            detector=self._detector,
            layer1_state=self._layer1_state,
        )
        self._stopping = asyncio.Event()
        self._cubes_processed = 0
        self._candidates_emitted = 0
        self._clusters_emitted = 0
        self._auto_dumps_dispatched = 0
        self._udp_dumps_dispatched = 0
        self._cluster_timeouts = 0

        # Sub-systems — built in start() if their config is non-None.
        self._clusterer: Optional[ClustererService] = None
        self._predicate: Optional[BrightPulsePredicate] = None
        self._cube_dump: Optional[CubeDumpWriter] = None
        self._udp_listener: Optional[UdpTriggerListener] = None
        self._cands_logger: Optional[CandsLogger] = None

    @staticmethod
    def _build_detector(config: SearchComputeConfig) -> DeterministicDetector:
        # Pass ``device=`` to the constructor (NOT ``.to(device)`` after-
        # the-fact) so the internal ``Layer2State._s_k`` is allocated
        # directly on the target device (see M5 chunk 6 lock).
        return DeterministicDetector(
            threshold_sigma=config.detector_threshold_sigma,
            detector_version=config.detector_version,
            search_node_id=config.search_node_id,
            gpu_half=config.gpu_half,
            dtype=config.detector_dtype,
            device=torch.device(config.detector_device),
            streaming=config.detector_streaming,
            streaming_tile_size=config.detector_streaming_tile_size,
        )

    @property
    def detector(self) -> DeterministicDetector:
        return self._detector

    @property
    def pipeline(self) -> CubePipeline:
        return self._pipeline

    @property
    def cubes_processed(self) -> int:
        return self._cubes_processed

    @property
    def candidates_emitted(self) -> int:
        return self._candidates_emitted

    @property
    def clusters_emitted(self) -> int:
        return self._clusters_emitted

    @property
    def auto_dumps_dispatched(self) -> int:
        return self._auto_dumps_dispatched

    @property
    def udp_dumps_dispatched(self) -> int:
        return self._udp_dumps_dispatched

    @property
    def cluster_timeouts(self) -> int:
        return self._cluster_timeouts

    @property
    def clusterer(self) -> Optional[ClustererService]:
        return self._clusterer

    @property
    def cube_dump(self) -> Optional[CubeDumpWriter]:
        return self._cube_dump

    @property
    def udp_listener(self) -> Optional[UdpTriggerListener]:
        return self._udp_listener

    @property
    def cands_logger(self) -> Optional[CandsLogger]:
        return self._cands_logger

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Bring up the RX-ring source + all configured sub-systems."""
        await self._source.start()
        cfg = self._config
        if cfg.clusterer_config is not None:
            self._clusterer = ClustererService(config=cfg.clusterer_config)
            self._clusterer.start()
        if cfg.bright_pulse_predicate_config is not None:
            self._predicate = BrightPulsePredicate(
                config=cfg.bright_pulse_predicate_config
            )
        if cfg.cube_dump_writer_config is not None:
            # M6 D7 dump_root may not exist yet on first run (production:
            # /home/ubuntu/data/m6/cube_dump; tests: tmp_path/dumps).
            # The writer thread's np.savez does not auto-mkdir, so make
            # sure the dir is in place before any dispatch happens.
            Path(cfg.cube_dump_writer_config.dump_root).mkdir(
                parents=True, exist_ok=True
            )
            self._cube_dump = CubeDumpWriter(config=cfg.cube_dump_writer_config)
            self._cube_dump.start()
        if cfg.udp_trigger_listener_config is not None:
            self._udp_listener = UdpTriggerListener(
                config=cfg.udp_trigger_listener_config
            )
            await self._udp_listener.start()
        if cfg.cands_logger_config is not None:
            self._cands_logger = CandsLogger(config=cfg.cands_logger_config)
        _LOG.info(
            "SearchComputeService up "
            "(sid=%d, gpu_half=%d, cluster=%s, dump=%s, udp=%s, log=%s)",
            cfg.search_node_id,
            cfg.gpu_half,
            self._clusterer is not None,
            self._cube_dump is not None,
            self._udp_listener is not None,
            self._cands_logger is not None,
        )

    async def stop(self) -> None:
        self._stopping.set()
        if self._udp_listener is not None:
            await self._udp_listener.stop()
        if self._cube_dump is not None:
            self._cube_dump.stop()
        if self._clusterer is not None:
            self._clusterer.shutdown(wait=True)
        if self._cands_logger is not None:
            self._cands_logger.close()
        await self._source.stop()
        _LOG.info(
            "SearchComputeService stopped: cubes=%d cands=%d clusters=%d "
            "auto_dumps=%d udp_dumps=%d cluster_timeouts=%d",
            self._cubes_processed,
            self._candidates_emitted,
            self._clusters_emitted,
            self._auto_dumps_dispatched,
            self._udp_dumps_dispatched,
            self._cluster_timeouts,
        )

    # -----------------------------------------------------------------
    # Per-cube driver
    # -----------------------------------------------------------------

    def _geom_from_slot(self, slot: CubeRingSlot) -> CubeGeometry:
        """Build a ``CubeGeometry`` from the slot + the static config.

        Production is expected to provide ``fine_dm_pc_cc_full`` covering
        the full fine-DM grid; the per-cube slice is taken from the
        first ``slot.n_fdm_in_cube`` entries (v1: M5 emits cubes covering
        the same DM trial range each call). Bench fallback: synthesise a
        linear grid ``np.linspace(50, 800, n_fdm_in_cube)``.
        """
        cfg = self._config
        if cfg.fine_dm_pc_cc_full is not None:
            full = np.asarray(cfg.fine_dm_pc_cc_full, dtype=np.float64)
            if full.shape[0] < slot.n_fdm_in_cube:
                raise ValueError(
                    f"fine_dm_pc_cc_full has {full.shape[0]} entries but "
                    f"slot covers {slot.n_fdm_in_cube} fine-DM trials"
                )
            fine_dm = full[: slot.n_fdm_in_cube].astype(np.float64, copy=True)
        else:
            fine_dm = np.linspace(
                50.0, 800.0, slot.n_fdm_in_cube, dtype=np.float64
            )
        # mjd_start = mjd_at_specnum_0 + specnum_start * t_int_fast_us / 1e6 / 86400.
        # t_int_fast_us = sample_period_us / sample_period_specnum.
        t_int_fast_us = cfg.cube_sample_period_us / cfg.cube_sample_period_specnum
        mjd_start = cfg.mjd_at_specnum_0 + (
            slot.specnum_start * t_int_fast_us * 1e-6 / 86400.0
        )
        return CubeGeometry(
            cube_id=slot.cube_id,
            specnum_start=slot.specnum_start,
            sample_period_specnum=cfg.cube_sample_period_specnum,
            t_det=slot.t_det,
            n_grid=slot.n_grid,
            n_fdm_in_cube=slot.n_fdm_in_cube,
            sample_period_us=cfg.cube_sample_period_us,
            cell_l_rad=cfg.cube_cell_l_rad,
            cell_m_rad=cfg.cube_cell_m_rad,
            l0_rad=cfg.cube_l0_rad,
            m0_rad=cfg.cube_m0_rad,
            fine_dm_pc_cc=fine_dm,
            mjd_start=mjd_start,
        )

    async def _process_one_cube(
        self, slot: CubeRingSlot
    ) -> List[Candidate]:
        cfg = self._config

        # Step 1 — UDP arm check (M6 D9): a UDP datagram arriving any
        # time before this cube was dequeued arms a single dump-next
        # for THIS cube. Subsequent UDPs during this cube don't stack.
        udp_armed = False
        if self._udp_listener is not None:
            udp_armed = self._udp_listener.consume_dump_next_cube_flag()

        # Step 2 — pipeline (M5).
        result = self._pipeline.process(slot)
        self._cubes_processed += 1
        self._candidates_emitted += len(result.candidates)

        # Build geometry once for both clusterer + dumps.
        geom = self._geom_from_slot(slot)

        # Step 3 — UDP dump (this cube). Submit BEFORE clustering so the
        # writer-thread hand-off doesn't compete with clustering for CPU.
        if udp_armed and self._cube_dump is not None:
            udp_manifest = CubeDumpManifest(
                cube_id=slot.cube_id,
                event_specnum_start=slot.specnum_start,
                mjd_start=geom.mjd_start,
                t_det=slot.t_det,
                n_fdm_in_cube=slot.n_fdm_in_cube,
                n_grid=slot.n_grid,
                trigger_source="udp",
                cluster_record=None,
                npz_path=self._cube_dump_path("udp", slot.specnum_start),
                search_node_id=cfg.search_node_id,
                gpu_half=cfg.gpu_half,
            )
            accepted = self._cube_dump.submit(
                cube=result.cube, manifest=udp_manifest
            )
            if accepted:
                self._udp_dumps_dispatched += 1

        # Step 4 — clusterer (M6 chunk 1; ThreadPoolExecutor).
        if self._clusterer is not None:
            future = self._clusterer.submit(result.candidates, geom)
            try:
                cluster_result = future.result(
                    timeout=cfg.cluster_result_timeout_s
                )
            except Exception:  # noqa: BLE001
                _LOG.warning(
                    "ClustererService timed out / failed on cube_id=%d "
                    "(n_cands=%d); skipping cube's records",
                    slot.cube_id,
                    len(result.candidates),
                )
                self._cluster_timeouts += 1
                return result.candidates
            self._clusters_emitted += len(cluster_result.records)

            # Step 5 — auto dumps (M6 D8 predicate).
            triggered_ids: set[int] = set()
            for record in cluster_result.records:
                if self._predicate is not None and self._predicate(record):
                    triggered_ids.add(record.cluster_id)
                    if self._cube_dump is not None:
                        auto_manifest = CubeDumpManifest(
                            cube_id=slot.cube_id,
                            event_specnum_start=slot.specnum_start,
                            mjd_start=geom.mjd_start,
                            t_det=slot.t_det,
                            n_fdm_in_cube=slot.n_fdm_in_cube,
                            n_grid=slot.n_grid,
                            trigger_source="auto",
                            cluster_record=record,
                            npz_path=self._cube_dump_path(
                                "auto", record.event_specnum
                            ),
                            search_node_id=cfg.search_node_id,
                            gpu_half=cfg.gpu_half,
                        )
                        accepted = self._cube_dump.submit(
                            cube=result.cube, manifest=auto_manifest
                        )
                        if accepted:
                            self._auto_dumps_dispatched += 1

            # Step 6 — T1/T2 ASCII rows (M6 D1).
            if self._cands_logger is not None:
                self._cands_logger.write_cube(
                    cands=result.candidates,
                    cluster_labels=cluster_result.labels,
                    cluster_records=cluster_result.records,
                    geom=geom,
                    triggered_cluster_ids=triggered_ids,
                )

        return result.candidates

    def _cube_dump_path(self, source: str, key_specnum: int) -> str:
        """Build the canonical per-cube NPZ path.

        Uses the configured ``cube_dump_writer_config.dump_root``. The
        file name template per M6 D7 is
        ``cube_s${sid}_g${g}_${event_specnum}.npz`` for auto dumps; UDP
        dumps follow the same template using the cube's
        ``specnum_start`` as the event_specnum.

        Note: the writer thread builds the *actual* on-disk path from
        the manifest's ``event_specnum_start`` (per ``CubeDumpWriter``
        contract — the manifest's ``npz_path`` field is informational
        and intentionally not used by the writer). This helper produces
        a value that *matches* the writer's canonical layout so
        consumers reading the manifest can find the file.
        """
        cfg = self._config
        assert cfg.cube_dump_writer_config is not None
        root = cfg.cube_dump_writer_config.dump_root
        return str(
            Path(root)
            / f"cube_s{cfg.search_node_id}_g{cfg.gpu_half}_{int(key_specnum)}.npz"
        )

    async def run(self) -> None:
        """Main loop. Iterates over RX-ring slots until source exhausts
        or ``stop()`` is called.
        """
        async for slot in self._source:
            if self._stopping.is_set():
                break
            await self._process_one_cube(slot)
            await self._source.release(slot.cube_id)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Production CLI entry. Chunk 6b-α + M6 chunk 5 ship the orchestrator
    + all sub-system wiring; real RX-ring binding (M4a POSIX-shm),
    config-loader, etcd watch, and SIGTERM handling land in the chunk-6b
    hardening pass once M4a's receive-ring API is locked.
    """
    raise NotImplementedError(
        "search_compute production CLI lands in M5 Chunk 6b hardening "
        "(needs M4a receive-ring API). M5 Chunk 6b-α + M6 chunk 5 "
        "deliver the SearchComputeService class with full M6 sub-system "
        "wiring for use by benches + unit tests."
    )


if __name__ == "__main__":
    main()
