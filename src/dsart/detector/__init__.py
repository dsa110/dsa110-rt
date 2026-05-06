"""DSA-110 real-time pipeline subpackage: detector.

The v1 deterministic conv-bank detector (plan §3.6 / §4.4) lives in
``forward.py`` (``Detector`` Protocol + ``DeterministicDetector``
Module + ``boxcar_via_cumsum`` primitive) and ``kernels.py`` (kernel-bank
construction, K=128 default per D2 in M5_PLAN_FIXES.md).

Chunk 2 will add ``decoder.py`` (per-kernel local-max NMS +
canonical-zone emit gate) and ``merger.py`` (cross-kernel SNR-sort + 4D
merge-radius suppression). Chunk 3 will add the sibling ``noise_norm/``
subpackage (Layer-1 σ-clipped global scalar + Layer-2 per-kernel σ_k
EMA).

File-name decomposition (``forward.py / decoder.py / merger.py /
kernels.py``) follows ``PARALLEL_AGENTS.md`` §3 Class A ownership; this
diverges from plan §3 line 94 which lists ``interface.py /
v1_deterministic.py / decoder.py / noise_norm.py``. See F8 in
``M5_PLAN_FIXES.md`` for the rename rationale; plan will be updated
during M5 hardening.
"""

from .forward import Detector, DeterministicDetector, boxcar_via_cumsum
from .kernels import DEFAULT_DETECTOR_DTYPE, Kernel, build_kernel_bank, make_image_kernel

__all__ = [
    "Detector",
    "DeterministicDetector",
    "boxcar_via_cumsum",
    "Kernel",
    "build_kernel_bank",
    "make_image_kernel",
    "DEFAULT_DETECTOR_DTYPE",
]
