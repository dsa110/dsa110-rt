"""DSA-110 search-node clusterer subpackage (M6).

This package owns the per-cube clusterer (HDBSCAN with DBSCAN
fallback — chunk 1) and the T1/T2 ASCII candidate logger (chunk 2).
The cube-dump writer + UDP listener live in the sibling
``dsart.dump`` package.

Chunk-2 deliverables (always available):
    * :class:`CandsLogger` — hourly-rotated, flock-serialised T1/T2
      writer (D1/D2).
    * :class:`CandsLoggerConfig` — per-process config dataclass.

Chunk-1 deliverables are re-exported lazily so this ``__init__`` does
not break if a future split lands one chunk before the other.
"""

from __future__ import annotations

from .cands_logger import CandsLogger, CandsLoggerConfig

__all__ = [
    "CandsLogger",
    "CandsLoggerConfig",
]

# Best-effort re-export of chunk-1 symbols. We don't import these eagerly
# at module-level because the chunk-2 deliverable must stand on its own
# (per spec: "rely on dsart.common.contracts only"), and partial
# chunk-1 landings (e.g. forward.py without state.py) must not break
# the chunk-2 import path.
try:  # pragma: no cover — exercised only when chunk 1 has fully landed.
    from .forward import (
        ClustererBackend,
        ClustererConfig,
        cluster_candidates,
    )

    __all__.extend(["ClustererBackend", "ClustererConfig", "cluster_candidates"])
except ImportError:
    pass

try:  # pragma: no cover
    from .features import (
        DEFAULT_WEIGHTS,
        FeatureMode,
        candidates_to_features,
        candidates_to_real_coords,
    )

    __all__.extend(
        ["DEFAULT_WEIGHTS", "FeatureMode", "candidates_to_features",
         "candidates_to_real_coords"]
    )
except ImportError:
    pass

try:  # pragma: no cover
    from .state import ClusterState  # type: ignore[attr-defined]

    __all__.append("ClusterState")
except ImportError:
    pass
