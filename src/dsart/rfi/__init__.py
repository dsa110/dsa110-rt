"""DSA-110 voltage-domain RFI flagger (M3 chunk 3c).

Public surface:

* :func:`flag_block` — top-level one-shot flagger. Takes voltage
  tensors in the M2 GEMM layout, returns ``(mask, source_tags)``.
* :class:`RFIFlagger` — stateful flagger holding the cold-start
  warmup counter and configured thresholds. Production callers use
  this; one-shot tests can use :func:`flag_block`.
* :class:`FlagBlockResult` — per-cube output dataclass.
* :class:`FlagSourceBit` — IntFlag enum for the per-cell source tag
  bitfield.
* :class:`MockTransportHeader` — chunk-3c stand-in for the live
  transport header used by :mod:`bench.rfi_warmup`.
* :class:`FlagPersistence` — optional latch that keeps a cell flagged
  for ``hold_s`` once the detectors have flagged it for a whole
  ``latch_window_s`` (see :mod:`dsart.rfi.persistence`).
* :func:`load_flagants` / :func:`load_flagants_torch` — legacy
  ``flagants.dat`` loader.

The lower-level per-detector helpers (:mod:`dsart.rfi.autos`,
:mod:`dsart.rfi.sk`, :mod:`dsart.rfi.bandpass_outlier`,
:mod:`dsart.rfi.group_outlier`, :mod:`dsart.rfi.sum_threshold`) are
exposed verbatim so unit tests can pin individual detector behaviour.
"""

from __future__ import annotations

from dsart.rfi.autos import (
    DEFAULT_M_VALUES,
    TOTAL_NATIVE_T,
    AutoSpectra,
    compute_autos,
    compute_autos_from_complex,
)
from dsart.rfi.bandpass_outlier import (
    DEFAULT_BANDPASS_K,
    bandpass_outlier_mask,
)
from dsart.rfi.combine import (
    FlagBlockResult,
    FlagSourceBit,
    MockTransportHeader,
    RFIFlagger,
    flag_block,
)
from dsart.rfi.flagants_loader import (
    load_flagants,
    load_flagants_torch,
    parse_flagants_text,
)
from dsart.rfi.group_outlier import DEFAULT_GROUP_K, group_outlier_mask
from dsart.rfi.persistence import (
    HOLD_S_DEFAULT,
    LATCH_FRAC_DEFAULT,
    LATCH_WINDOW_S_DEFAULT,
    FlagPersistence,
    PersistenceStats,
    seconds_to_cubes,
)
from dsart.rfi.sk import (
    DEFAULT_SK_FAR,
    compute_sk,
    gaussian_sk_thresholds,
    sk_combined_mask,
    sk_mask,
    sk_thresholds,
)
from dsart.rfi.sum_threshold import (
    DEFAULT_ETA,
    DEFAULT_MAX_M,
    sum_threshold_1d,
    sum_threshold_2d,
)

__all__ = [
    # autos
    "AutoSpectra",
    "DEFAULT_M_VALUES",
    "TOTAL_NATIVE_T",
    "compute_autos",
    "compute_autos_from_complex",
    # SK
    "DEFAULT_SK_FAR",
    "compute_sk",
    "gaussian_sk_thresholds",
    "sk_combined_mask",
    "sk_mask",
    "sk_thresholds",
    # bandpass-outlier
    "DEFAULT_BANDPASS_K",
    "bandpass_outlier_mask",
    # group-outlier
    "DEFAULT_GROUP_K",
    "group_outlier_mask",
    # sum-threshold
    "DEFAULT_ETA",
    "DEFAULT_MAX_M",
    "sum_threshold_1d",
    "sum_threshold_2d",
    # flagants.dat
    "load_flagants",
    "load_flagants_torch",
    "parse_flagants_text",
    # combine
    "FlagBlockResult",
    "FlagSourceBit",
    "MockTransportHeader",
    "RFIFlagger",
    "flag_block",
    # time persistence
    "FlagPersistence",
    "PersistenceStats",
    "HOLD_S_DEFAULT",
    "LATCH_FRAC_DEFAULT",
    "LATCH_WINDOW_S_DEFAULT",
    "seconds_to_cubes",
]
