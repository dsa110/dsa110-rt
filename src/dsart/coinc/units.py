"""Downstream sample-unit helpers (x2 SPEC §8, ``docs/x2/downstream_inventory.md`` D1).

Downstream of the search engine every width, specnum and time offset is
counted in the op-point's EXTERNAL sample: 1048.576 us at 1x (32 native
samples, 16 SNAP specnums) and 524.288 us at 2x (16 native, 8 specnums).
The period travels in every C1 batch header (``sample_period_us``) and,
from archive schema 2 on, in the archived CSVs and Level3 JSON. Consumers
must take it from there and fall back to the 1x value ONLY when an
artefact predates the field -- which keeps every 1x output bit-identical
(1x artefacts never carry the field, or carry exactly 1048.576).

Two different "16"s exist (inventory §0): SNAP specnums per 1x sample
(becomes 8 at 2x) and native samples per 2x sample (was 32 at 1x). The
helpers below derive both from the period so neither is ever hard-coded.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from ..common.constants import NATIVE_SAMPLE_US, SPECNUM_PERIOD_US

__all__ = [
    "T_SEARCH_1X_US",
    "sample_period_us_or_default",
    "period_from_mapping",
    "native_per_sample",
    "specnums_per_sample",
    "width_ms",
]

#: The 1x external (search) sample period in microseconds. Only ever a
#: FALLBACK for artefacts that predate the explicit ``sample_period_us``.
T_SEARCH_1X_US: float = 1048.576

#: Periods below this are placeholders (e.g. a heartbeat's 1.0), never a
#: real search cadence (the shortest op-point is 524.288 us).
_MIN_PLAUSIBLE_PERIOD_US: float = 100.0


def sample_period_us_or_default(
    value: Any, default: float = T_SEARCH_1X_US,
) -> float:
    """Return ``value`` as a float period in us, or ``default``.

    ``None``, non-numeric, non-finite and implausibly small values
    (heartbeat placeholders) all map to ``default``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v) or v < _MIN_PLAUSIBLE_PERIOD_US:
        return float(default)
    return v


def period_from_mapping(
    d: Optional[Mapping[str, Any]],
    *keys: str,
    default: float = T_SEARCH_1X_US,
) -> float:
    """First plausible period among ``keys`` of ``d`` (Level3 ``c2`` dict,
    CSV row, ...), else ``default``. Keys default to the archive names."""
    if not keys:
        keys = ("sample_period_us", "peak_sample_period_us")
    if d:
        for k in keys:
            if k in d:
                v = sample_period_us_or_default(d.get(k), default=float("nan"))
                if math.isfinite(v):
                    return v
    return float(default)


def _integer_ratio(period_us: float, unit_us: float) -> int:
    ratio = float(period_us) / float(unit_us)
    n = int(round(ratio))
    if n < 1 or abs(n * unit_us - float(period_us)) > 1e-6:
        raise ValueError(
            f"sample period {period_us!r} us is not an integer multiple of "
            f"{unit_us} us"
        )
    return n


def native_per_sample(sample_period_us: float) -> int:
    """Native (32.768 us) samples per external sample: 32 at 1x, 16 at 2x."""
    return _integer_ratio(sample_period_us, NATIVE_SAMPLE_US)


def specnums_per_sample(sample_period_us: float) -> int:
    """SNAP specnums (65.536 us) per external sample: 16 at 1x, 8 at 2x."""
    return _integer_ratio(sample_period_us, SPECNUM_PERIOD_US)


def width_ms(width_samples: Any, sample_period_us: Any) -> float:
    """Width in ms of ``width_samples`` external samples (NaN if unusable)."""
    try:
        w = float(width_samples)
    except (TypeError, ValueError):
        return float("nan")
    return w * sample_period_us_or_default(sample_period_us) / 1000.0
