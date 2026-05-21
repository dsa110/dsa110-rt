"""Cube-index ↔ real-antenna-number mapping.

The 96 antennas the corr pipeline streams in its (NANTS, NCHAN, NPOL)
cubes are NOT numbered 1..96 — DSA-110 has gaps and out-of-band IDs
(e.g. cube_idx 47 → antenna 102, cube_idx 48 → antenna 116). The
canonical map lives in ``configs/corr_setup_96.yaml::antenna_order``,
pushed to etcd at ``/cnf/corr`` and consumed by the entire pipeline.

We re-export the same map here as a python constant so the dashboard
doesn't need pyyaml at runtime nor a file-resolution coupling to the
configs directory. If the YAML ever changes, this MUST be re-synced
(``tests/test_antenna_map.py`` should catch the drift; if not, this
docstring is your reminder).

The mapping is **bijective** — every cube index 0..95 corresponds to
exactly one real antenna number, and the operator-facing
``/mon/ant/<N>`` etcd keys use the **real antenna number**, not the
cube index. Always translate at the dashboard boundary.
"""

from __future__ import annotations

from typing import Final


# Verbatim copy of configs/corr_setup_96.yaml::antenna_order (May 2026).
# DO NOT hand-edit — this should track the corr_setup_96.yaml.
_ANT_NUM_BY_CUBE_IDX: Final[tuple[int, ...]] = (
    # cube_idx 0..8 → DSA antennas 1..9
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    # cube_idx 9..46 → DSA antennas 11..51 (skips 10, monotonic)
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51,
    # cube_idx 47, 48 → out-of-band IDs 102, 116
    102, 116,
    # cube_idx 49..82 → DSA antennas 68..101 (monotonic)
    68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101,
    # cube_idx 83..95 → DSA antennas 103..115 (skips 102)
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
)
assert len(_ANT_NUM_BY_CUBE_IDX) == 96, (
    f"expected 96 cube indices, got {len(_ANT_NUM_BY_CUBE_IDX)}"
)
assert len(set(_ANT_NUM_BY_CUBE_IDX)) == 96, (
    "antenna numbers must be unique"
)


NANTS_CUBE: Final[int] = 96


def cube_idx_to_ant_num(cube_idx: int) -> int:
    """0-based cube index → 1-based operator-facing antenna number.

    Raises ValueError for out-of-range indices so callers don't
    silently scribble to the wrong antenna table cell.
    """
    if not (0 <= cube_idx < NANTS_CUBE):
        raise ValueError(
            f"cube_idx {cube_idx} out of range [0, {NANTS_CUBE})"
        )
    return _ANT_NUM_BY_CUBE_IDX[cube_idx]


_CUBE_IDX_BY_ANT_NUM: Final[dict[int, int]] = {
    a: i for i, a in enumerate(_ANT_NUM_BY_CUBE_IDX)
}


def ant_num_to_cube_idx(ant_num: int) -> int:
    """Operator-facing antenna number → 0-based cube index.

    Raises KeyError if the antenna number is not part of the 96-ant
    correlator config.
    """
    try:
        return _CUBE_IDX_BY_ANT_NUM[int(ant_num)]
    except KeyError as e:
        raise KeyError(
            f"antenna {ant_num} not in correlator config (96-ant set)"
        ) from e


def all_ant_nums() -> tuple[int, ...]:
    """All 96 real antenna numbers, in cube-index order."""
    return _ANT_NUM_BY_CUBE_IDX
