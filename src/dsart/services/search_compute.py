"""Search-node compute service entry (RX ring → detector).

M5 Chunk 6b-α (M6 chunk 0: trigger emitter retired): long-running asyncio
orchestrator that drives the ``CubePipeline`` from a pluggable
``RxRingSource``. Per-cube ``Candidate`` lists are surfaced from
``CubePipelineResult``; downstream cluster + cube-dump + UDP-listener
integration lands in M6 chunks 1-5.

Responsibilities (plan §3.6 + §4.4):
  1. Bring up RX-ring source.
  2. Construct CubePipeline + DeterministicDetector + Layer1State.
  3. Per-cube loop: acquire slot → pipeline.process(slot) → release.
     (M6 chunks 1-5 will add: clusterer + cube-dump + UDP listener.)

Production path runs ``main()`` from a CLI; benches subclass / wire
the service directly to inject a synthetic source.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from ..common.contracts import Candidate
from ..detector.forward import DeterministicDetector
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
    ``configs/config_compute_search.yaml``).

    The chunk-6b-α scope ships fields needed by the pipeline + Layer-1
    state. Endpoint discovery (etcd watch) and dynamic reconfiguration
    are deferred to chunk-6b production hardening.

    The chunk-8 production wiring uses the GPU image backend by
    default — the pipeline's ``image_backend="gpu"`` config selects
    the fused dequant + combine + cuFFT-cfp16 ifft2 + edge-mask
    pipeline (D21 / D25). Benches that need the chunk-6a numpy CPU
    reference path (e.g. cube_injection_detector) build their own
    ``CubePipelineConfig(image_backend="cpu")`` and pass it directly
    to ``CubePipeline``.
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
        self._stopping = asyncio.Event()
        self._cubes_processed = 0
        self._candidates_emitted = 0

    @staticmethod
    def _build_detector(config: SearchComputeConfig) -> DeterministicDetector:
        # Pass ``device=`` to the constructor (NOT ``.to(device)`` after-
        # the-fact) so the internal ``Layer2State._s_k`` is allocated
        # directly on the target device. ``.to(device)`` reassigns the
        # registered ``_sigma_k`` buffer but Layer2State retains a
        # pointer to the original CPU tensor, so the EMA divisor would
        # straddle devices on cuda. Passing ``device=`` at construction
        # time keeps Layer2State + ``_sigma_k`` on the same device.
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

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Bring up the RX-ring source."""
        await self._source.start()
        _LOG.info(
            "SearchComputeService up (sid=%d, gpu_half=%d)",
            self._config.search_node_id,
            self._config.gpu_half,
        )

    async def stop(self) -> None:
        self._stopping.set()
        await self._source.stop()
        _LOG.info(
            "SearchComputeService stopped: cubes=%d candidates=%d",
            self._cubes_processed,
            self._candidates_emitted,
        )

    # -----------------------------------------------------------------
    # Per-cube driver
    # -----------------------------------------------------------------

    async def _process_one_cube(
        self, slot: CubeRingSlot
    ) -> List[Candidate]:
        result = self._pipeline.process(slot)
        self._cubes_processed += 1
        self._candidates_emitted += len(result.candidates)
        return result.candidates

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
