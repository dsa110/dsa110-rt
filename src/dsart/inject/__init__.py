"""DSA-110 real-time pipeline subpackage: inject.

Two injectors live here, plus the automated prober built on them:

  - :mod:`dsart.inject.online` (M3 chunk 3d) — voltage-domain online
    injector consumed by ``corr_fast_compute`` (plan §4.7).
  - ``dsart.inject.cube_injection`` (M5; not yet imported here) —
    post-imaging detector unit-test injector. The two share the
    ``(l, m, dm, fluence, width_samples, profile)`` schema where it
    overlaps; the voltage-level injector is the end-to-end gate at
    M6, the cube-level injector is the M5 detector unit-test gate.
  - ``dsart.inject.bot`` (not imported here; run as its own h23
    user unit ``dsart_inject_bot``) — hourly end-to-end test
    injections through the dashboard's /control/inject, with Slack
    per-shot and daily-summary reporting. Offline campaign analysis
    lives in ``tools/inject/``.
"""

# Lazy (PEP 562) re-exports: dsart.inject.online imports torch, which
# is present on the corr/search nodes but NOT in the h23 service env
# where dsart.inject.bot runs. An eager import here would make the
# whole package unimportable on h23; lazy resolution keeps the public
# API identical on nodes that have torch and defers the ImportError to
# first use elsewhere.
__all__ = [
    "InjectionConfig",
    "OnlineInjector",
]


def __getattr__(name):
    if name in __all__:
        from dsart.inject import online
        return getattr(online, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
