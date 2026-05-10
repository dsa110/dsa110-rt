"""DSA-110 real-time pipeline subpackage: inject.

Two injectors live here:

  - :mod:`dsart.inject.online` (M3 chunk 3d) — voltage-domain online
    injector consumed by ``corr_fast_compute`` (plan §4.7).
  - ``dsart.inject.cube_injection`` (M5; not yet imported here) —
    post-imaging detector unit-test injector. The two share the
    ``(l, m, dm, fluence, width_samples, profile)`` schema where it
    overlaps; the voltage-level injector is the end-to-end gate at
    M6, the cube-level injector is the M5 detector unit-test gate.
"""

from dsart.inject.online import (
    InjectionConfig,
    OnlineInjector,
)

__all__ = [
    "InjectionConfig",
    "OnlineInjector",
]
