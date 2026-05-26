"""DSA-110 real-time pipeline subpackage: detector.

The v1 deterministic conv-bank detector (plan §3.6 / §4.4) is split
across four files per ``PARALLEL_AGENTS.md`` §3 Class A ownership:

  - ``forward.py``  — ``Detector`` Protocol + ``DeterministicDetector``
                      Module wiring conv-bank → decoder → merger →
                      canonical-zone gate; the ``boxcar_via_cumsum``
                      primitive (the only allowed K_dm/K_time consumer
                      per plan §3.6.13).
  - ``kernels.py``  — Kernel-bank construction (K=128 default per D2).
  - ``decoder.py``  — Per-kernel local-max NMS + canonical-zone emit gate.
  - ``merger.py``   — Cross-kernel SNR-sort + 4D merge-radius suppression.

Chunk 3 added the sibling ``noise_norm/`` subpackage (Layer-1
σ-clipped global scalar + Layer-2 per-kernel σ_k EMA); the
``DeterministicDetector._sigma_k`` buffer mirrors the EMA-tracked
tensor (D11 placeholder retired) and the warmup flag-bit-set logic
fires while the EMA is in burn-in.

This decomposition diverges from plan §3 line 94 which lists
``interface.py / v1_deterministic.py / decoder.py / noise_norm.py``;
see F8 in ``M5_PLAN_FIXES.md`` for the rename rationale (plan updated
during M5 hardening).
"""

from .decoder import decode_local_max, filter_to_canonical
from .forward import Detector, DeterministicDetector, boxcar_via_cumsum
from .kernels import DEFAULT_DETECTOR_DTYPE, Kernel, build_kernel_bank, make_image_kernel
from .merger import (
    DEFAULT_MERGE_RADIUS_FDM,
    DEFAULT_MERGE_RADIUS_LM,
    DEFAULT_MERGE_RADIUS_T,
    MergerConfig,
    merge_across_kernels,
    merge_across_kernels_c1,
)

__all__ = [
    "Detector",
    "DeterministicDetector",
    "boxcar_via_cumsum",
    "Kernel",
    "build_kernel_bank",
    "make_image_kernel",
    "DEFAULT_DETECTOR_DTYPE",
    "decode_local_max",
    "filter_to_canonical",
    "MergerConfig",
    "merge_across_kernels",
    "merge_across_kernels_c1",
    "DEFAULT_MERGE_RADIUS_LM",
    "DEFAULT_MERGE_RADIUS_FDM",
    "DEFAULT_MERGE_RADIUS_T",
]
