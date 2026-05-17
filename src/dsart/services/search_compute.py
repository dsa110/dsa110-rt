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

import argparse
import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from ..cluster import (
    CandsLogger,
    CandsLoggerConfig,
    ClustererConfig,
    ClustererService,
)
from ..common.constants import (
    CUBE_CADENCE_SAMPLES_DEFAULT,
    T_INT_SEARCH_US_DEFAULT,
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

        Emits one INFO line per ``_status_every_cubes`` cubes (default
        10) so the M7.2 / production soaks have a visible progress
        signal in the orchestrator logs without having to wait for a
        SIGTERM-triggered final stats line.
        """
        status_every = 10
        next_status = status_every
        loop_start = time.monotonic()
        last_log = loop_start
        last_cubes = 0
        async for slot in self._source:
            if self._stopping.is_set():
                break
            await self._process_one_cube(slot)
            await self._source.release(slot.cube_id)
            if self._cubes_processed >= next_status:
                now = time.monotonic()
                d_cubes = self._cubes_processed - last_cubes
                dt = max(now - last_log, 1e-9)
                _LOG.info(
                    "cube_progress: cubes=%d cands=%d clusters=%d "
                    "(%.2f cubes/s last %.1fs; %.2f cubes/s overall)",
                    self._cubes_processed,
                    self._candidates_emitted,
                    self._clusters_emitted,
                    d_cubes / dt,
                    dt,
                    self._cubes_processed / max(now - loop_start, 1e-9),
                )
                next_status += status_every
                last_log = now
                last_cubes = self._cubes_processed


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI helpers (M7.2)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader (lazy import; sub-system configs only)."""
    import yaml  # local import; pyyaml is a runtime dep already
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dm_grids_from_npz(
    dm_plan_path: Path, n_coarse: int, n_fdm: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DM grids from a DmPlan NPZ. Falls back to a synthetic linear
    grid if the file is unreadable. The NPZ layout (set by the corr-side
    ``dm_plan`` tooling): ``coarse_dm`` (or ``coarse_dm_pc_cm3``),
    ``fine_dm`` (or ``fine_dm_pc_cm3``), ``fine_to_coarse``."""
    with np.load(dm_plan_path) as npz:
        coarse_key = next(
            (k for k in ("coarse_dm_pc_cm3", "coarse_dm") if k in npz),
            None,
        )
        fine_key = next(
            (k for k in ("fine_dm_pc_cm3", "fine_dm") if k in npz),
            None,
        )
        if coarse_key is None or fine_key is None:
            raise KeyError(
                f"DM plan {dm_plan_path} missing 'coarse_dm'/'fine_dm' keys; "
                f"found {list(npz.keys())}"
            )
        coarse_dm = np.asarray(npz[coarse_key], dtype=np.float64)
        fine_dm = np.asarray(npz[fine_key], dtype=np.float64)
        if "fine_to_coarse" in npz:
            f2c = np.asarray(npz["fine_to_coarse"], dtype=np.int32)
        else:
            # Default: every fine maps to coarse cell 0 (safe; matches
            # the M4a unit-test convention).
            f2c = np.zeros(len(fine_dm), dtype=np.int32)
    # Trim to the requested sizes if the on-disk plan is bigger.
    coarse_dm = coarse_dm[:n_coarse]
    fine_dm = fine_dm[:n_fdm]
    f2c = f2c[:n_fdm]
    if len(f2c) != len(fine_dm):
        raise ValueError(
            f"DM plan {dm_plan_path}: fine_to_coarse length "
            f"{len(f2c)} != fine_dm length {len(fine_dm)}"
        )
    return coarse_dm, fine_dm, f2c


def _synthetic_dm_grids(
    n_coarse: int, n_fdm: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bench / smoke DM grids: all fine DMs map to coarse cell 0 so
    ``compute_time_shift_search`` produces non-negative shifts."""
    coarse_dm = np.linspace(0.0, 300.0, n_coarse, dtype=np.float64)
    fine_dm = np.linspace(0.0, 100.0, n_fdm, dtype=np.float64)
    f2c = np.zeros(n_fdm, dtype=np.int32)
    return coarse_dm, fine_dm, f2c


def _build_search_config_from_yaml(
    yaml_doc: dict,
    *,
    n_grid: int,
    n_fdm: int,
    gpu_half: int,
    search_node_id: int,
    image_backend: str,
    device: str,
    enable_clusterer: bool,
    enable_cube_dump: bool,
    enable_udp_listener: bool,
    enable_cands_logger: bool,
    fine_dm_pc_cc_full: Optional[np.ndarray],
) -> SearchComputeConfig:
    """Translate ``configs/config_compute_search.yaml`` into
    ``SearchComputeConfig`` with optional M6 sub-systems gated on the
    per-subsystem ``enable_*`` flags. This is the M7.2 inline shim; the
    full ``config_loader`` integration is the M7.2.5 follow-up."""
    det = yaml_doc.get("detector", {}) or {}
    noise = yaml_doc.get("noise", {}) or {}
    pipe_cfg = CubePipelineConfig(
        n_grid=int(n_grid),
        image_backend=image_backend,   # "cpu" or "gpu"
        device=device,
    )

    clusterer_cfg: Optional[ClustererConfig] = None
    if enable_clusterer:
        cl = yaml_doc.get("clusterer", {}) or {}
        kwargs: dict[str, Any] = {}
        if "backend" in cl:
            kwargs["backend"] = str(cl["backend"])
        if "feature_mode" in cl:
            kwargs["feature_mode"] = str(cl["feature_mode"])
        if "weights" in cl:
            kwargs["weights"] = tuple(float(w) for w in cl["weights"])
        if "min_cluster_size" in cl:
            kwargs["min_cluster_size"] = int(cl["min_cluster_size"])
        if "min_samples" in cl:
            kwargs["min_samples"] = int(cl["min_samples"])
        if "cluster_selection_epsilon" in cl:
            kwargs["cluster_selection_epsilon"] = float(
                cl["cluster_selection_epsilon"]
            )
        if "dbscan_eps" in cl:
            kwargs["dbscan_eps"] = float(cl["dbscan_eps"])
        if "dbscan_min_samples" in cl:
            kwargs["dbscan_min_samples"] = int(cl["dbscan_min_samples"])
        if "metric" in cl:
            kwargs["metric"] = str(cl["metric"])
        clusterer_cfg = ClustererConfig(**kwargs)

    cube_dump_cfg: Optional[CubeDumpWriterConfig] = None
    if enable_cube_dump:
        cd = yaml_doc.get("cube_dump", {}) or {}
        cube_dump_cfg = CubeDumpWriterConfig(
            dump_root=Path(cd.get("dump_root", "/tmp/dsart-cube-dump")),
            search_node_id=int(cd.get("search_node_id", search_node_id)),
            gpu_half=int(cd.get("gpu_half", gpu_half)),
            queue_maxsize=int(cd.get("queue_maxsize", 4)),
        )

    udp_listener_cfg: Optional[UdpTriggerListenerConfig] = None
    if enable_udp_listener:
        ul = yaml_doc.get("udp_trigger_listener", {}) or {}
        udp_listener_cfg = UdpTriggerListenerConfig(
            host=str(ul.get("host", "127.0.0.1")),
            port=int(ul.get("port", 11227)),
        )

    cands_logger_cfg: Optional[CandsLoggerConfig] = None
    if enable_cands_logger:
        cl = yaml_doc.get("cands_logger", {}) or {}
        cands_logger_cfg = CandsLoggerConfig(
            log_root=Path(cl.get("log_root", "/tmp/dsart-cands-log")),
            search_node_id=int(cl.get("search_node_id", search_node_id)),
            gpu_half=int(cl.get("gpu_half", gpu_half)),
        )

    predicate_cfg: Optional[BrightPulsePredicateConfig] = None
    if enable_cube_dump:
        bp = yaml_doc.get("bright_pulse_predicate", {}) or {}
        kwargs2: dict[str, Any] = {}
        for k in ("min_snr", "dm_fine_min_pc_cc", "dm_fine_max_pc_cc",
                  "width_samples_max", "min_cntc", "holdoff_ms"):
            if k in bp:
                kwargs2[k] = bp[k]
        if kwargs2:
            predicate_cfg = BrightPulsePredicateConfig(**kwargs2)
        else:
            predicate_cfg = BrightPulsePredicateConfig()

    return SearchComputeConfig(
        pipeline=pipe_cfg,
        n_fdm=int(n_fdm),
        detector_threshold_sigma=float(det.get("threshold_sigma", 8.0)),
        detector_device=device,
        search_node_id=int(search_node_id),
        gpu_half=int(gpu_half),
        layer1_n_burnin_cubes=int(noise.get("layer1_n_burnin_cubes", 5)),
        fine_dm_pc_cc_full=fine_dm_pc_cc_full,
        clusterer_config=clusterer_cfg,
        cube_dump_writer_config=cube_dump_cfg,
        bright_pulse_predicate_config=predicate_cfg,
        udp_trigger_listener_config=udp_listener_cfg,
        cands_logger_config=cands_logger_cfg,
    )


def _install_signal_handlers(loop: asyncio.AbstractEventLoop,
                             stop_event: asyncio.Event) -> None:
    """Wire SIGTERM/SIGINT to set ``stop_event`` so the main loop
    exits cleanly. Mirrors the dsart_rt orchestrator's contract."""
    def _handle(signum: int) -> None:
        _LOG.info("received signal %d; stopping…", signum)
        stop_event.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle, sig)
        except (NotImplementedError, RuntimeError):
            # Not all platforms / event loops support add_signal_handler;
            # fall back to signal.signal which is good enough for the
            # orchestrator-launched case.
            signal.signal(sig, lambda s, _f: _handle(s))


async def _run_async(args: argparse.Namespace) -> int:
    # Lazy import so the CLI parser still works without the C extension
    # when --help is invoked on a dev box.
    from ..transport.production_rx_ring import ProductionRxRingSource
    from ..transport.recv_ring import (
        BYTES_CFP16_COMPLEX,
        BYTES_CINT8_COMPLEX,
        RxRingDims,
    )

    bpc = (BYTES_CINT8_COMPLEX if args.bytes_per_cell == 2
           else BYTES_CFP16_COMPLEX)
    dims = RxRingDims(
        n_corr=args.n_corr,
        n_coarse_dm=args.n_coarse_dm,
        t_buf_samples=args.t_buf_samples,
        n_filled_per_corr=args.n_filled,
        bytes_per_cell=bpc,
    )

    if args.dm_plan_path is not None and args.dm_plan_path.exists():
        coarse_dm, fine_dm, f2c = _dm_grids_from_npz(
            args.dm_plan_path, args.n_coarse_dm, args.n_fdm
        )
        _LOG.info(
            "DM grids loaded from %s: n_coarse=%d n_fdm=%d",
            args.dm_plan_path, len(coarse_dm), len(fine_dm),
        )
    else:
        coarse_dm, fine_dm, f2c = _synthetic_dm_grids(
            args.n_coarse_dm, args.n_fdm
        )
        _LOG.warning(
            "DM plan path %s unavailable; using synthetic grids "
            "(coarse=%d fine=%d) — this is OK for M7.2 fake-data soak "
            "but PRODUCTION must point --dm-plan-path at the corr-side "
            "DmPlan NPZ.",
            args.dm_plan_path, args.n_coarse_dm, args.n_fdm,
        )

    source = ProductionRxRingSource(
        shm_name=args.shm_name,
        ring_dims=dims,
        n_fdm_in_cube=args.n_fdm,
        t_det=args.t_det,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=f2c,
        compute_half=args.gpu_half,
        t_int_search_us=args.t_int_search_us,
        cube_cadence_samples=args.cube_cadence_samples,
        n_grid=args.n_grid,
        enable_cuda_register=args.cuda_register,
        poll_interval_s=args.poll_interval_s,
        max_cubes=args.max_cubes if args.max_cubes > 0 else None,
        fan_in_min_corrs=args.fan_in_min_corrs,
    )

    if args.config_yaml is not None and args.config_yaml.exists():
        yaml_doc = _load_yaml(args.config_yaml)
        _LOG.info("loaded search-compute yaml from %s", args.config_yaml)
    else:
        yaml_doc = {}
        if args.config_yaml is not None:
            _LOG.warning(
                "config_yaml=%s not found; using empty doc (all "
                "M6 sub-systems disabled by default).",
                args.config_yaml,
            )

    cfg = _build_search_config_from_yaml(
        yaml_doc,
        n_grid=args.n_grid,
        n_fdm=args.n_fdm,
        gpu_half=args.gpu_half,
        search_node_id=args.search_node_id,
        image_backend=args.image_backend,
        device=args.device,
        enable_clusterer=args.enable_clusterer,
        enable_cube_dump=args.enable_cube_dump,
        enable_udp_listener=args.enable_udp_listener,
        enable_cands_logger=args.enable_cands_logger,
        fine_dm_pc_cc_full=fine_dm,
    )

    service = SearchComputeService(config=cfg, source=source)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)

    async def _signal_watcher() -> None:
        await stop_event.wait()
        _LOG.info("stop_event fired; signalling service.stop()…")
        await service.stop()

    await service.start()
    watcher_task = asyncio.create_task(_signal_watcher())
    rc = 0
    try:
        await service.run()
    except Exception:  # noqa: BLE001
        _LOG.exception("search_compute service crashed")
        rc = 1
    finally:
        stop_event.set()
        await watcher_task
        # service.stop() may have been invoked by the watcher already;
        # call it again so the source/sub-systems are definitely torn
        # down on the normal exit path.
        await service.stop()
    _LOG.info(
        "search_compute exit: cubes=%d cands=%d clusters=%d rc=%d",
        service.cubes_processed, service.candidates_emitted,
        service.clusters_emitted, rc,
    )
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Ring binding (M7.2.3 / Phase B contract) ---------------------
    p.add_argument("--shm-name", required=True,
                   help="POSIX-shm name to attach (created by the "
                        "search-rx process, e.g. /dsart-rxring-n01).")
    p.add_argument("--n-corr", type=int, default=16,
                   help="ring dim: number of correlator groups (default: 16)")
    p.add_argument("--n-coarse-dm", type=int, default=5,
                   help="ring dim: number of coarse DM slabs (default: 5 = "
                        "M7.1 op-point)")
    p.add_argument("--t-buf-samples", type=int, default=4096,
                   help="ring dim: time-axis depth in search-cadence samples "
                        "(default 4096 ≈ 16 cube cadences of headroom).")
    p.add_argument("--n-filled", type=int, default=5000,
                   help="ring dim: cells per (corr, dm) slot")
    p.add_argument("--bytes-per-cell", type=int, default=2, choices=(2, 4),
                   help="ring dim: 2=cint8 complex (prod default), "
                        "4=cfp16 complex (debug)")

    # --- Cube assembler tuning ----------------------------------------
    p.add_argument("--n-fdm", type=int, default=16,
                   help="fine-DM trials per cube for THIS half "
                        "(default: 16). Production: 32 per half = 64 total.")
    p.add_argument("--t-det", type=int, default=512,
                   help="detector window length in samples (default: 512 = "
                        "2 × block_samples_search). Must be ≥ cube_cadence "
                        "for the validity_mask to cover every t.")
    p.add_argument("--cube-cadence-samples", type=int,
                   default=CUBE_CADENCE_SAMPLES_DEFAULT,
                   help=f"slots per cube (default: {CUBE_CADENCE_SAMPLES_DEFAULT})")
    p.add_argument("--t-int-search-us", type=float,
                   default=T_INT_SEARCH_US_DEFAULT,
                   help=f"search-cadence sample period in µs "
                        f"(default: {T_INT_SEARCH_US_DEFAULT})")
    p.add_argument("--n-grid", type=int, default=256,
                   help="uv-grid side length (default: 256)")
    p.add_argument("--cuda-register", action="store_true",
                   help="attempt cudaHostRegister on the ring (deferred "
                        "D-item D2; safe to leave off until the C API "
                        "exposes rx_ring_get_base_ptr).")
    p.add_argument("--poll-interval-s", type=float, default=0.001,
                   help="how long the assembler sleeps between write_seq "
                        "polls (default 1ms; production-rate-safe).")
    p.add_argument("--fan-in-min-corrs", type=int, default=16,
                   help="minimum number of chgroups that must have "
                        "advanced past the next cube boundary before "
                        "we emit a cube (default 16 = production "
                        "strict all-chgroups-required; M7.2 smoke "
                        "should pass 1 to allow partial fan-in).")

    # --- SearchComputeService identity --------------------------------
    p.add_argument("--gpu-half", type=int, default=0, choices=(0, 1),
                   help="which compute half this process serves (0 or 1)")
    p.add_argument("--search-node-id", type=int, default=1,
                   help="search node id (1..4 in production)")
    p.add_argument("--device", default="cpu",
                   help="torch device for detector + cube tensor "
                        "(cpu / cuda / cuda:N). Default: cpu")
    p.add_argument("--image-backend", default="cpu",
                   choices=("cpu", "gpu"),
                   help="CubePipeline image backend (cpu / gpu). Default: cpu "
                        "for M7.2 bring-up; flip to gpu once a real GPU is "
                        "available on the search nodes.")

    # --- DM plan source -----------------------------------------------
    p.add_argument("--dm-plan-path", type=Path, default=None,
                   help="path to a DmPlan NPZ "
                        "(keys: coarse_dm_pc_cm3, fine_dm_pc_cm3, "
                        "fine_to_coarse). Falls back to a synthetic linear "
                        "grid if omitted/unreadable.")

    # --- M6 sub-system gating + config --------------------------------
    p.add_argument("--config-yaml", type=Path, default=None,
                   help="optional configs/config_compute_search.yaml. "
                        "Only consulted for the sub-system blocks gated "
                        "by --enable-* flags below.")
    p.add_argument("--enable-clusterer", action="store_true")
    p.add_argument("--enable-cube-dump", action="store_true")
    p.add_argument("--enable-udp-listener", action="store_true")
    p.add_argument("--enable-cands-logger", action="store_true")
    p.add_argument("--enable-all", action="store_true",
                   help="shortcut: enable all M6 sub-systems above")

    # --- Lifecycle ----------------------------------------------------
    p.add_argument("--max-cubes", type=int, default=0,
                   help="stop after this many cubes (default 0 = unlimited; "
                        "production: 0, run until SIGTERM).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = p.parse_args(argv)
    if args.enable_all:
        args.enable_clusterer = True
        args.enable_cube_dump = True
        args.enable_udp_listener = True
        args.enable_cands_logger = True

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
