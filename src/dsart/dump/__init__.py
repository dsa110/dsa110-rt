"""dsart.dump — cube-dump writer + UDP trigger listener (M6 chunks 3+4).

Public API:

  * ``BrightPulsePredicateConfig`` / ``BrightPulsePredicate`` — auto-trigger
    gate (D8). Owned by the per-(search_node, gpu_half) clusterer dispatch
    path; on a True return the cluster's cube is enqueued for dumping.
  * ``CubeDumpWriterConfig`` / ``CubeDumpWriter`` — single-worker writer
    thread fed by a bounded queue. Persists ``[T_det, N_fdm, N_grid,
    N_grid]`` float16 cubes as NPZ archives with the
    ``CubeDumpManifest`` sidecar (D7).
  * ``UdpTriggerListenerConfig`` / ``UdpTriggerListener`` — asyncio UDP
    listener for external "dump next cube" trigger requests (D9, chunk 4).

See ``M6_PLAN_FIXES.md`` D7-D9 + D12 for the locked design decisions and
``CubeDumpManifest`` / ``ClusterRecord`` in ``dsart.common.contracts``
for the data contracts.
"""

from __future__ import annotations

from .cube_dump import (
    BrightPulsePredicate,
    BrightPulsePredicateConfig,
    CubeDumpWriter,
    CubeDumpWriterConfig,
)
from .udp_listener import UdpTriggerListener, UdpTriggerListenerConfig

__all__ = [
    "BrightPulsePredicate",
    "BrightPulsePredicateConfig",
    "CubeDumpWriter",
    "CubeDumpWriterConfig",
    "UdpTriggerListener",
    "UdpTriggerListenerConfig",
]
