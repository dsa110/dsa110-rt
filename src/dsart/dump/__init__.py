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

from typing import TYPE_CHECKING

# Lazy re-exports (PEP 562). The cube-dump path (``cube_dump``,
# ``c2_trigger_listener``) pulls in the heavy GPU/imaging stack (torch et
# al.) that only exists on search nodes. The corr-node voltage-retention
# modules (``voltage_ring``, ``voltage_trigger_listener``) live in this same
# package but are pure-CPU/numpy. Importing them must NOT drag in torch, so
# we defer the cube imports to first attribute access instead of doing them
# eagerly at package import time.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .c2_trigger_listener import C2TriggerListener, C2TriggerListenerConfig
    from .cube_dump import (
        BrightPulsePredicate,
        BrightPulsePredicateConfig,
        CubeDumpWriter,
        CubeDumpWriterConfig,
    )
    from .udp_listener import UdpTriggerListener, UdpTriggerListenerConfig

_LAZY_EXPORTS = {
    "C2TriggerListener": ".c2_trigger_listener",
    "C2TriggerListenerConfig": ".c2_trigger_listener",
    "BrightPulsePredicate": ".cube_dump",
    "BrightPulsePredicateConfig": ".cube_dump",
    "CubeDumpWriter": ".cube_dump",
    "CubeDumpWriterConfig": ".cube_dump",
    "UdpTriggerListener": ".udp_listener",
    "UdpTriggerListenerConfig": ".udp_listener",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib

    mod = importlib.import_module(module, __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY_EXPORTS))
