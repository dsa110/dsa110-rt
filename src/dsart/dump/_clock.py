"""Injectable monotonic clock helper for the cube-dump module (M6 chunk 3).

The bright-pulse predicate enforces a per-process holdoff between dumps.
Tests need to advance the clock without sleeping, so the predicate
accepts a ``time_now_ms`` callable. The default implementation here
wraps ``time.monotonic`` so production has no extra overhead.
"""

from __future__ import annotations

import time

__all__ = ["monotonic_ms"]


def monotonic_ms() -> float:
    """Return the current monotonic time in milliseconds.

    Wraps ``time.monotonic`` (which is in seconds) so the predicate's
    holdoff arithmetic can be done in milliseconds without per-call
    multiplication scattered around the code.

    Returns:
        Monotonic time in milliseconds (float). The absolute value is
        meaningless; only differences are valid.
    """
    return time.monotonic() * 1000.0
