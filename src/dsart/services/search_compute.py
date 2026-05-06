"""Search-node compute service entry (RX ring → detector → triggers).

M5 Chunk 6b-α: long-running asyncio orchestrator that drives the
``CubePipeline`` from a pluggable ``RxRingSource`` and fans out each
cube's emitted ``Candidate`` list through a ``TriggerEmitter`` to a
configured set of correlation-listener endpoints.

Responsibilities (plan §3.6 + §4.4):
  1. Bring up RX-ring source (M4a in production; ``SyntheticRxRingSource``
     for benches / unit tests).
  2. Construct ``CubePipeline`` + ``DeterministicDetector`` +
     ``Layer1State`` from the resolved ``SearchComputeConfig``.
  3. Bring up ``TriggerEmitter`` with predicate chain + holdoff state
     machine + endpoint pool.
  4. Per-cube loop:
       - acquire next slot from the RX ring
       - pipeline.process(slot)              → CubePipelineResult
       - emitter.process_candidates(...)     → fan-out
       - source.release(slot.cube_id)
  5. Graceful shutdown via ``await service.stop()``: stops the RX-ring,
     drains the emitter, closes endpoint connections.

Production path runs ``main()`` from a CLI; benches subclass / wire
the service directly to inject a synthetic source.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from ..common.contracts import Candidate
from ..detector.forward import DeterministicDetector
from ..noise_norm.layer1 import Layer1State
from ..trigger.conditions import (
    PerCubePerKernelCap,
    PerCubeTotalCap,
    RateLimitTokenBucket,
    SnrThreshold,
)
from ..trigger.predicate import TriggerCondition
from ..trigger.emitter import (
    ConnectionEndpoint,
    EmitRecord,
    TriggerEmitter,
    TriggerEmitterConfig,
)
from ..trigger.holdoff import HoldoffStateMachine
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
    ``configs/config_compute_search.yaml``).

    The chunk-6b-α scope ships fields needed by the pipeline + emitter
    + Layer-1 state. Endpoint discovery (etcd watch) and dynamic
    reconfiguration are deferred to chunk-6b production hardening.
    """

    pipeline: CubePipelineConfig
    n_fdm: int
    detector_threshold_sigma: float = 8.0
    detector_dtype: torch.dtype = torch.float16
    detector_device: str = "cpu"
    detector_version: str = "v1.M5"
    search_node_id: int = 1
    gpu_half: int = 1
    layer1_n_burnin_cubes: int = 5
    layer1_n_sigma: float = 3.0
    layer1_n_iterations: int = 3
    holdoff_ms: float = 50.0
    snr_threshold: float = 8.0
    per_cube_per_kernel_cap: int = 4
    per_cube_total_cap: int = 16
    rate_limit_per_s: float = 10.0
    rate_limit_burst: int = 10
    correlation_endpoints: Sequence[ConnectionEndpoint] = field(default_factory=tuple)


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
        self._emitter: Optional[TriggerEmitter] = None
        self._stopping = asyncio.Event()
        self._cubes_processed = 0
        self._candidates_emitted = 0
        self._records_dispatched = 0

    @staticmethod
    def _build_detector(config: SearchComputeConfig) -> DeterministicDetector:
        return DeterministicDetector(
            threshold_sigma=config.detector_threshold_sigma,
            detector_version=config.detector_version,
            search_node_id=config.search_node_id,
            gpu_half=config.gpu_half,
            dtype=config.detector_dtype,
        ).to(torch.device(config.detector_device))

    @property
    def detector(self) -> DeterministicDetector:
        return self._detector

    @property
    def pipeline(self) -> CubePipeline:
        return self._pipeline

    @property
    def emitter(self) -> Optional[TriggerEmitter]:
        return self._emitter

    @property
    def cubes_processed(self) -> int:
        return self._cubes_processed

    @property
    def candidates_emitted(self) -> int:
        return self._candidates_emitted

    @property
    def records_dispatched(self) -> int:
        return self._records_dispatched

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Bring up the RX-ring source + emitter."""
        await self._source.start()
        conditions: List[TriggerCondition] = [
            SnrThreshold(min_snr=self._config.snr_threshold),
            PerCubePerKernelCap(max_per_kernel=self._config.per_cube_per_kernel_cap),
            PerCubeTotalCap(max_total=self._config.per_cube_total_cap),
            RateLimitTokenBucket(
                rate_per_s=self._config.rate_limit_per_s,
                burst=self._config.rate_limit_burst,
            ),
        ]
        holdoff = HoldoffStateMachine(holdoff_ms=self._config.holdoff_ms)
        emitter_cfg = TriggerEmitterConfig(
            search_node_id=self._config.search_node_id,
            gpu_half=self._config.gpu_half,
            endpoints=list(self._config.correlation_endpoints),
            conditions=conditions,
            holdoff=holdoff,
        )
        self._emitter = TriggerEmitter(emitter_cfg)
        await self._emitter.start()
        _LOG.info(
            "SearchComputeService up (sid=%d, gpu_half=%d, %d endpoints)",
            self._config.search_node_id,
            self._config.gpu_half,
            len(self._config.correlation_endpoints),
        )

    async def stop(self) -> None:
        self._stopping.set()
        if self._emitter is not None:
            await self._emitter.stop()
        await self._source.stop()
        _LOG.info(
            "SearchComputeService stopped: cubes=%d candidates=%d records=%d",
            self._cubes_processed,
            self._candidates_emitted,
            self._records_dispatched,
        )

    # -----------------------------------------------------------------
    # Per-cube driver
    # -----------------------------------------------------------------

    async def _process_one_cube(
        self, slot: CubeRingSlot
    ) -> List[EmitRecord]:
        result = self._pipeline.process(slot)
        self._cubes_processed += 1
        self._candidates_emitted += len(result.candidates)
        if self._emitter is None:
            raise RuntimeError(
                "SearchComputeService.start() must be called before processing cubes"
            )
        records = await self._emitter.process_candidates(
            slot.cube_id, result.candidates
        )
        self._records_dispatched += len(records)
        return records

    async def run(self) -> None:
        """Main loop. Iterates over RX-ring slots until source exhausts
        or ``stop()`` is called.
        """
        if self._emitter is None:
            raise RuntimeError(
                "SearchComputeService.run() requires start() first"
            )
        async for slot in self._source:
            if self._stopping.is_set():
                break
            await self._process_one_cube(slot)
            await self._source.release(slot.cube_id)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Production CLI entry. Chunk 6b-α ships the orchestrator skeleton;
    real RX-ring binding (M4a POSIX-shm), config-loader, etcd watch, and
    SIGTERM handling land in the chunk-6b hardening pass once M4a's
    receive-ring API is locked.
    """
    raise NotImplementedError(
        "search_compute production CLI lands in M5 Chunk 6b hardening "
        "(needs M4a receive-ring API). Chunk 6b-α delivers the "
        "SearchComputeService class for use by benches + unit tests."
    )


if __name__ == "__main__":
    main()
