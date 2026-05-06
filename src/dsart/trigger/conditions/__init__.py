"""Trigger-emit condition implementations (plan §4.4 line 1671-1714).

One file per condition. The default v1 chain (configured in
``config_compute_search.yaml::trigger_predicate_chain``) is:

    SnrThreshold(min_snr=8.0)
    PerCubePerKernelCap(max_per_kernel=4)
    PerCubeTotalCap(max_total=16)
    RateLimitTokenBucket(rate_per_s=10, burst=50)

Future conditions (e.g. LearnedClassifier, PointingExclusion) plug in
here as new files + one yaml line; no other code change needed.
"""

from .per_cube_caps import PerCubePerKernelCap, PerCubeTotalCap
from .rate_limit_token_bucket import RateLimitTokenBucket
from .snr_threshold import SnrThreshold

__all__ = [
    "PerCubePerKernelCap",
    "PerCubeTotalCap",
    "RateLimitTokenBucket",
    "SnrThreshold",
]
