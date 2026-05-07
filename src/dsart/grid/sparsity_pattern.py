"""Deterministic sparsity-pattern builder for the fast-vis uv-grid (M3 chunk 3a).

Pinned by plan §3 lines 305-307 + §4.3 line 1378 (Option C) +
``PARALLEL_AGENTS.md`` §3 (Class C ownership):

* The fast-vis pipeline grids per-baseline, per-channel Stokes-I
  visibilities into a ``[N_grid, N_grid]`` single-side ``+uv`` complex
  grid (no Hermitian mirror cell). Only a small fraction (~7-12%) of
  cells are touched by any baseline; the rest are skipped via a
  pre-computed sparsity pattern.
* The pattern is **never transported on the wire or via etcd**. Both
  ``corr_fast_compute`` (this side) and ``dsart-search-rx`` (the search
  side, M5) call :func:`build_pattern` at ``cmd: prepare`` and arrive at
  bit-identical ``(ix_row, ix_col)`` arrays from the inputs they both
  share via etcd ``/cnf/corr_setup_96`` plus the ``dec_deg`` payload of
  the verb itself plus the static ``n_grid`` / ``kernel_support`` from
  ``configs/config_compute_*.yaml``.
* :data:`SparsityPattern.pattern_id` is a 64-bit blake2b hash of the
  *inputs* (not of the pattern bytes), so it certifies "both ends saw
  the same antpos + chgroup_table + dec_deg + n_grid + K_support" —
  which together with the deterministic build below implies byte-
  identical ``(ix_row, ix_col)`` arrays. Carried in every value-channel
  datagram header (§4.3) for per-packet verification.

Sign convention (F20)
=====================

The DSA-110 voltage time-frequency convention produces visibilities
``V(u, v) ∝ exp(+2πi · b · ŝ · ν / c)`` — opposite of textbook CASA.
Internal imagers based on ``np.fft.ifft2`` (which applies the positive
exponent ``exp(+2πi(ul + vm))``) recover the source at ``(-l, -m)``
unless ``(u, v)`` are negated once at the gridder. The reference slow-
corr imager ``tools/viz/common.py::grid_uv_natural`` applies this
negation at lines ~352-368 and is the "ground truth" for the parity
gate; the fast-corr gridder mirrors that convention here so both
imagers land sources at the same ``(+l, +m)``.

Since the only consumer of the pattern is the gridder + the search-side
scatter kernel, the F20 negation is captured *here*: cell indices are
computed from ``-(u, v) / cell_lambda``, equivalently to "compute
``round(u/cell)`` then mirror about the grid centre". A unit-test in
``tests/test_sparsity_pattern.py::test_F20_uv_negation_applied`` pins
the convention against ``grid_uv_natural`` for a single synthetic
baseline.

Class ownership
===============

Class C per ``PARALLEL_AGENTS.md`` §3 — M3 owns; M5 reads via
``from dsart.grid.sparsity_pattern import SparsityPattern, build_pattern,
predict_pattern_id``. M5 does **not** edit this file directly; if M5
needs a function added, opens a small PR + M3 acks (per the ownership
table).

References
==========

* Plan §3 lines 305-309 — sparsity pattern + ``pattern_id`` semantics.
* Plan §4.2 lines 1350-1358 — gridder + sparse-pattern build at
  ``cmd: prepare``.
* Plan §4.3 lines 1378-1381 — Option C local-rebuild contract.
* ``tools/viz/common.py::grid_uv_natural`` — slow-corr reference imager
  (F20 ``(u, v)`` negation, lines ~352-368). Class C; not modified by
  this chunk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np

from dsart.common.constants import (
    KERNEL_SUPPORT_DEFAULT,
    N_GRID_DEFAULT,
    NANTS,
    NCHAN_PER_CHGROUP,
    NU_CHGROUP_TOP_GHZ,
    PATTERN_DEC_QUANT_DEG,
    PATTERN_HASH_BYTES,
    PATTERN_ID_PERSON,
    freq_GHz,
)


__all__ = [
    "SPEED_OF_LIGHT_M_PER_S",
    "SparsityPattern",
    "build_pattern",
    "compute_antpos_hash",
    "compute_chgroup_table_hash",
    "predict_pattern_id",
    "quantise_dec_deg",
]


#: Speed of light (m / s). Local copy avoids importing astropy from a
#: hot-path-relevant module; the value is fixed by definition (CODATA).
SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0


# ---------------------------------------------------------------------------
# Hash helpers (deterministic across the corr ↔ search ends)
# ---------------------------------------------------------------------------


def _blake2b_u64(payload: bytes) -> int:
    """Hash ``payload`` to a 64-bit unsigned int (little-endian).

    Uses the project-wide :data:`PATTERN_ID_PERSON` personalisation tag
    so unrelated 64-bit blake2b digests in the codebase do not collide
    with pattern hashes in any future debug log analysis.
    """
    h = hashlib.blake2b(
        payload,
        digest_size=PATTERN_HASH_BYTES,
        person=PATTERN_ID_PERSON,
    )
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def compute_antpos_hash(antpos_e: np.ndarray, antpos_n: np.ndarray) -> int:
    """Hash the antpos arrays (E, N) for use as a ``pattern_id`` input.

    Per plan §3 line 307: ``antpos_hash = blake2b_64(antennas.out
    bytes-on-disk)`` — but operationally the corr/search agents read
    antpos from etcd ``/cnf/corr_setup_96``, not from the on-disk
    ``antennas.out``. The two must yield the *same* hash when the
    operator-side loader populates etcd from ``antennas.out`` verbatim.

    To keep this deterministic and free of file-path / endianness
    pitfalls, both ends call this function on the float32 arrays they
    each load (from etcd or from disk); the operator-side loader is
    responsible for ensuring the arrays match in dtype / order. Tests
    pin both ends.

    Args:
        antpos_e, antpos_n: ``(NANTS,)`` float32 antenna offsets in
            metres. Cast to float32 internally so callers can pass
            float64 without breaking the hash.

    Returns:
        64-bit unsigned int.
    """
    if antpos_e.shape != (NANTS,) or antpos_n.shape != (NANTS,):
        raise ValueError(
            f"antpos shapes must be ({NANTS},), got "
            f"E={antpos_e.shape}, N={antpos_n.shape}"
        )
    e32 = np.ascontiguousarray(antpos_e, dtype=np.float32)
    n32 = np.ascontiguousarray(antpos_n, dtype=np.float32)
    return _blake2b_u64(e32.tobytes() + n32.tobytes())


def compute_chgroup_table_hash(
    chgroup_ch0: tuple[int, ...] | np.ndarray | None = None,
    nu_chgroup_top_GHz: tuple[float, ...] | np.ndarray | None = None,
) -> int:
    """Hash the ``chgroup → ch0 / ν_chgroup_top`` table for ``pattern_id``.

    Per plan §3 line 307: ``chgroup_table_hash =
    blake2b_64(/cnf/corr_setup_96.chgroup_table bytes)``. We hash the
    *derived* canonical representation (``CHGROUP_CH0`` + the
    ``NU_CHGROUP_TOP_GHZ`` tuple from :mod:`dsart.common.constants`) so
    both ends produce the same hash without depending on any operator-
    side YAML serialisation order.

    Args:
        chgroup_ch0: optional override for the ``ch0[g]`` table. Defaults
            to :data:`dsart.common.constants.CHGROUP_CH0`.
        nu_chgroup_top_GHz: optional override for the ``ν_chgroup_top``
            table. Defaults to :data:`dsart.common.constants.NU_CHGROUP_TOP_GHZ`.

    Returns:
        64-bit unsigned int.
    """
    from dsart.common.constants import CHGROUP_CH0  # avoid top-level cycle

    ch0_arr = np.asarray(
        CHGROUP_CH0 if chgroup_ch0 is None else chgroup_ch0,
        dtype=np.int32,
    )
    nu_arr = np.asarray(
        NU_CHGROUP_TOP_GHZ if nu_chgroup_top_GHz is None else nu_chgroup_top_GHz,
        dtype=np.float64,
    )
    return _blake2b_u64(
        ch0_arr.tobytes(order="C") + nu_arr.tobytes(order="C")
    )


def quantise_dec_deg(dec_deg: float) -> float:
    """Quantise ``dec_deg`` onto the ``PATTERN_DEC_QUANT_DEG`` grid.

    Plan §3 line 307. Returns a float (NOT an int) so the quantised
    value can be rendered into the pattern_id payload as bytes of a
    fixed-precision float, sidestepping any "did the operator pass
    41.234 vs 41.249?" bin-edge ambiguity.

    Examples (with ``PATTERN_DEC_QUANT_DEG = 0.25``)::

        quantise_dec_deg(41.001) == 41.0
        quantise_dec_deg(41.234) == 41.25
        quantise_dec_deg(41.249) == 41.25
        quantise_dec_deg(41.499) == 41.5
    """
    return float(np.round(dec_deg / PATTERN_DEC_QUANT_DEG)) * PATTERN_DEC_QUANT_DEG


def _pattern_id_payload(
    *,
    chgroup: int,
    dec_deg_quant: float,
    n_grid: int,
    kernel_support: int,
    antpos_hash: int,
    chgroup_table_hash: int,
) -> bytes:
    """Pack the input tuple into a fixed-layout byte string for hashing.

    Single source of truth for the ``pattern_id`` payload — both
    :func:`build_pattern` and :func:`predict_pattern_id` call this so
    they cannot diverge.

    Layout (little-endian, fixed widths to avoid Python int variable-
    length encoding biting us)::

        offset  size   field
        0       2      uint16  chgroup
        2       2      uint16  n_grid
        4       2      uint16  kernel_support
        6       8      float64 dec_deg_quant
        14      8      uint64  antpos_hash
        22      8      uint64  chgroup_table_hash
        total: 30 bytes
    """
    if not 0 <= chgroup < (1 << 16):
        raise ValueError(f"chgroup={chgroup} out of uint16 range")
    if not 0 < n_grid < (1 << 16):
        raise ValueError(f"n_grid={n_grid} out of uint16 range")
    if not 0 < kernel_support < (1 << 16):
        raise ValueError(
            f"kernel_support={kernel_support} out of uint16 range"
        )
    if not 0 <= antpos_hash < (1 << 64):
        raise ValueError(f"antpos_hash={antpos_hash} not a uint64")
    if not 0 <= chgroup_table_hash < (1 << 64):
        raise ValueError(
            f"chgroup_table_hash={chgroup_table_hash} not a uint64"
        )
    return (
        np.uint16(chgroup).tobytes()
        + np.uint16(n_grid).tobytes()
        + np.uint16(kernel_support).tobytes()
        + np.float64(dec_deg_quant).tobytes()
        + np.uint64(antpos_hash).tobytes()
        + np.uint64(chgroup_table_hash).tobytes()
    )


# ---------------------------------------------------------------------------
# SparsityPattern dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparsityPattern:
    """Result of :func:`build_pattern` (plan §3 lines 305-307).

    Both ``corr_fast_compute`` (M3) and ``dsart-search-rx`` (M5) call
    :func:`build_pattern` at ``cmd: prepare``; the byte-identical
    ``(ix_row, ix_col)`` arrays + ``pattern_id`` are the integration
    contract between the two ends. The pattern itself is *not* sent
    over the wire; only the small ``(antpos, chgroup_table)`` inputs
    flow (already in etcd).

    Attributes
    ----------
    ix_row : np.ndarray (N_filled,) uint16
        Grid **row** of each filled cell. Row indexes the v-axis (cf.
        ``tools/viz/common.py::grid_uv_natural`` which uses
        ``np.add.at(grid, (iy, ix), ...)`` with ``iy`` = row).
    ix_col : np.ndarray (N_filled,) uint16
        Grid **column** of each filled cell. Column indexes the u-axis.
    pattern_id : int
        ``blake2b_64(chgroup, dec_deg_quantised_to_0p25_deg, n_grid,
        kernel_support, antpos_hash, chgroup_table_hash)``. **Hash of
        inputs**, not of the output bytes (plan §3 line 307).
    n_grid : int
        Grid side length (cells). Stored for downstream sanity asserts.
    n_filled : int
        ``len(ix_row)``. Cached because ``len()`` on a frozen ndarray
        attribute is fine but having it as a field makes log lines and
        report tables easier to format.
    chgroup : int
        The chgroup index this pattern was built for.
    dec_deg_quant : float
        Quantised declination (degrees) used in the ``pattern_id`` hash.
    kernel_support : int
        Cells of gridding kernel support (1 = pillbox / nearest cell).
    antpos_hash : int
        ``blake2b_64`` of the antpos arrays used (provenance).
    chgroup_table_hash : int
        ``blake2b_64`` of the chgroup → ch0 / ν_chgroup_top table
        (provenance).
    """

    ix_row: np.ndarray
    ix_col: np.ndarray
    pattern_id: int
    n_grid: int
    n_filled: int
    chgroup: int
    dec_deg_quant: float
    kernel_support: int
    antpos_hash: int
    chgroup_table_hash: int


# ---------------------------------------------------------------------------
# Core/outrigger discrimination (plan §3 line 446 / F27)
# ---------------------------------------------------------------------------


#: Default core-radius cut (m). NOTE (F32): radius is INSUFFICIENT for
#: discriminating DSA-110 core vs outrigger antennas, because outrigger
#: stations 103/104/.../115 lie at radii (~420-1100 m) that overlap with
#: the legitimate core stations 99-102 (which have N ≈ 423-441 m). The
#: canonical discriminator is the STATION NUMBER: stations ≤ 102 are
#: core, 103-116 are outriggers (12 outriggers + 1 outrigger-set
#: station = 14 outriggers). Production reads ``is_core`` from etcd
#: ``/cnf/corr_setup_96`` (plan §3 line 446); for cal-blob fixtures use
#: :func:`core_baseline_mask_from_station_numbers` with the
#: ``antenna_order`` from the cal yaml. The radius helper below is
#: retained ONLY for synthetic tests where station numbers are absent.
CORE_RADIUS_M_DEFAULT: float = 500.0

#: Canonical core-antenna count (plan §3 line 446) = 82, when the array
#: is the standard 96-ant DSA-110 layout (96 = 82 core + 14 outriggers).
N_CORE_DEFAULT: int = 82

#: Largest station number that is a CORE antenna in the canonical
#: DSA-110 layout. Station ≤ this is core; station > this is an
#: outrigger. Used by :func:`core_baseline_mask_from_station_numbers`.
MAX_CORE_STATION_DEFAULT: int = 102


def core_baseline_mask_from_antpos(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    *,
    n_core: int | None = None,
    r_core_m: float | None = None,
) -> np.ndarray:
    """``(NBASE,) bool`` baseline mask: True iff both antennas are core.

    The "core" antennas are the geometrically central ones (radius
    ``√(e² + n²)`` from the array origin small relative to the
    outriggers). Production reads ``is_core`` from etcd
    ``/cnf/corr_setup_96`` (plan §3 line 446); this helper is the
    bench/test fallback when antpos comes from a cal-blob and etcd is
    not available.

    Specify exactly one of:

    * ``n_core``: keep the ``n_core`` smallest-radius antennas as core.
      Robust to antpos ordering.
    * ``r_core_m``: keep antennas with radius < ``r_core_m`` (m).

    The mask iterates baselines in the xGPU upper-tri order
    (``b ≤ a``); index ``k`` corresponds to baseline ``(a, b)``
    matching :func:`dsart.services.slow_corr_kernel.upper_tri_indices`.

    Parameters
    ----------
    antpos_e, antpos_n : np.ndarray (NANTS,) float
        Antenna E/N positions in metres.
    n_core : int, optional
        Keep this many smallest-radius antennas as core. Default
        ``N_CORE_DEFAULT = 82``.
    r_core_m : float, optional
        Alternative spec: keep antennas with radius below this cut (m).

    Returns
    -------
    np.ndarray (NBASE,) bool
        True for baselines with BOTH endpoints in the core set.

    Notes
    -----
    Why this exists (F27): the legacy positional helpers in
    ``test_sparsity_pattern.py`` and friends defined "core" as
    "antennas 0..n_core-1", but the DSA-110 cal-blob antpos is **not**
    sorted by radius — ant index 48 is an OUTRIGGER (r ≈ 1008 m) and
    ant index 83 is a CORE antenna (r ≈ 423 m). The positional helper
    leaked outrigger baselines into the core image and dropped real
    core baselines, leaving stray fills in the outer uv-plane of the
    chunk-3a footprint plot. This helper fixes that by working off the
    actual antpos, matching the production etcd-driven path.
    """
    if (n_core is None) == (r_core_m is None):
        raise ValueError(
            "core_baseline_mask_from_antpos: pass exactly one of "
            "(n_core, r_core_m); got both or neither"
        )
    antpos_e = np.asarray(antpos_e, dtype=np.float64)
    antpos_n = np.asarray(antpos_n, dtype=np.float64)
    if antpos_e.shape != antpos_n.shape or antpos_e.ndim != 1:
        raise ValueError(
            f"antpos_e shape {antpos_e.shape} / antpos_n shape "
            f"{antpos_n.shape} must match and be 1-D"
        )
    nants = int(antpos_e.shape[0])
    radii = np.hypot(antpos_e, antpos_n)                        # (NANTS,)

    if n_core is not None:
        if not 0 < n_core <= nants:
            raise ValueError(
                f"n_core={n_core}, expected 1..{nants}"
            )
        # Smallest-radius antennas are core. argsort is stable but
        # ties don't matter here (radii are physical distances).
        sorted_idx = np.argsort(radii, kind="stable")
        is_core_ant = np.zeros(nants, dtype=bool)
        is_core_ant[sorted_idx[:n_core]] = True
    else:
        is_core_ant = (radii < float(r_core_m))

    nbase = nants * (nants + 1) // 2
    mask = np.zeros(nbase, dtype=bool)
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            mask[k] = is_core_ant[a] and is_core_ant[b]
            k += 1
    return mask


def core_baseline_mask_from_station_numbers(
    antenna_order: list[int] | np.ndarray,
    *,
    max_core_station: int = MAX_CORE_STATION_DEFAULT,
) -> np.ndarray:
    """``(NBASE,) bool`` baseline mask: True iff both antennas are core.

    Canonical DSA-110 core/outrigger discrimination by STATION NUMBER:
    station ≤ ``max_core_station`` (default 102) is core; station >
    is an outrigger. This is robust to cal-blob antpos ordering and
    matches production etcd ``/cnf/corr_setup_96::is_core`` (plan §3
    line 446).

    F32 (M3 carryover): the radius-based heuristic
    :func:`core_baseline_mask_from_antpos` cannot separate core from
    outrigger reliably for DSA-110 because outrigger stations 103-115
    have radii (~420-1100 m) that OVERLAP with legit core stations
    99-102 (N ≈ 423-441 m). For example, station 103 (E=+198, N=−374,
    r=423 m) and station 100 (E=+9, N=+424, r=424 m) have nearly
    identical radii but station 100 is core, 103 is outrigger.
    Station-number is the only unambiguous discriminator.

    Parameters
    ----------
    antenna_order : list[int] or array-like (NANTS,)
        Per-fada-slot DSA-110 station numbers, e.g. from the cal yaml's
        ``cal_solutions.antenna_order``.
    max_core_station : int
        Largest station number that is core. Default
        :data:`MAX_CORE_STATION_DEFAULT` = 102.

    Returns
    -------
    np.ndarray (NBASE,) bool
        True for baselines with BOTH endpoints in the core set.
    """
    antenna_order = np.asarray(antenna_order, dtype=np.int64)
    if antenna_order.ndim != 1:
        raise ValueError(
            f"antenna_order must be 1-D; got shape {antenna_order.shape}"
        )
    nants = int(antenna_order.shape[0])
    is_core_ant = (antenna_order <= int(max_core_station))
    nbase = nants * (nants + 1) // 2
    mask = np.zeros(nbase, dtype=bool)
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            mask[k] = is_core_ant[a] and is_core_ant[b]
            k += 1
    return mask


# ---------------------------------------------------------------------------
# build_pattern + predict_pattern_id
# ---------------------------------------------------------------------------


def _per_baseline_uv_meters(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    *,
    is_core_baseline_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-baseline ``(du_m, dv_m)`` in metres for cross-baselines only.

    Iterates baselines in xGPU upper-tri order with ``b ≤ a`` (matching
    :func:`dsart.services.slow_corr_kernel.upper_tri_indices` and the
    F18 visibility convention ``V_ij = conj(E_lower) · E_higher``). For
    each kept baseline ``(a, b)``::

        du_m = antpos_e[a] - antpos_e[b]
        dv_m = antpos_n[a] - antpos_n[b]

    Auto-correlations (``a == b``) are always excluded. If
    ``is_core_baseline_mask`` is provided, ``bls_idx`` for which the
    mask is False is also excluded (drops outrigger-touching baselines
    per plan §3 line 452).

    The F20 ``(u, v)`` negation is **not** applied here; it lives in
    :func:`build_pattern` so this internal helper stays trivially
    correct as a "raw geometric" computation that any reviewer can
    audit against an antpos diagram.

    Returns
    -------
    (du_m, dv_m) : tuple of np.ndarray
        Each ``(N_kept,)`` float64. Order matches the kept-baseline
        order of the xGPU upper-tri iteration.
    """
    nants = antpos_e.shape[0]
    nbase = nants * (nants + 1) // 2
    if is_core_baseline_mask is not None and is_core_baseline_mask.shape != (nbase,):
        raise ValueError(
            f"is_core_baseline_mask shape {is_core_baseline_mask.shape} != "
            f"({nbase},)"
        )

    # Vectorised: build (a, b) lists then index.
    a_list = np.empty(nbase, dtype=np.int64)
    b_list = np.empty(nbase, dtype=np.int64)
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            a_list[k] = a
            b_list[k] = b
            k += 1
    is_cross = a_list != b_list                                     # drops autos
    keep = is_cross
    if is_core_baseline_mask is not None:
        keep = keep & np.asarray(is_core_baseline_mask, dtype=bool)
    a_kept = a_list[keep]
    b_kept = b_list[keep]
    du_m = antpos_e[a_kept].astype(np.float64) - antpos_e[b_kept].astype(np.float64)
    dv_m = antpos_n[a_kept].astype(np.float64) - antpos_n[b_kept].astype(np.float64)
    return du_m, dv_m


def build_pattern(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    *,
    chgroup: int,
    dec_deg: float,
    n_grid: int = N_GRID_DEFAULT,
    kernel_support: int = KERNEL_SUPPORT_DEFAULT,
    is_core_baseline_mask: np.ndarray | None = None,
    antpos_hash: int | None = None,
    chgroup_table_hash: int | None = None,
) -> SparsityPattern:
    """Build the deterministic sparsity pattern for one chgroup at one DEC.

    Both ``corr_fast_compute`` (M3) and ``dsart-search-rx`` (M5) call
    this at ``cmd: prepare``; bit-identical inputs yield bit-identical
    ``(ix_row, ix_col)`` arrays and ``pattern_id``. See module docstring
    + plan §3 lines 305-307 + §4.3 line 1378 (Option C).

    Args:
        antpos_e: ``(NANTS,)`` antenna east offset (m, ITRF). Float32 or
            float64 accepted; cast internally.
        antpos_n: ``(NANTS,)`` antenna north offset (m, ITRF).
        chgroup: corr-node frequency-band index (``0..N_CHGROUP-1``).
            Picks ``ν_chgroup_top`` via
            :data:`dsart.common.constants.NU_CHGROUP_TOP_GHZ`.
        dec_deg: pointing declination in degrees. Quantised to
            :data:`dsart.common.constants.PATTERN_DEC_QUANT_DEG` (= 0.25
            deg) before hashing into ``pattern_id`` so declinations
            within ± 0.125 deg of each other share a pattern.
            **NOTE**: the cell rounding itself uses the *un-quantised*
            ``dec_deg`` only insofar as the (u, v) computation is
            DEC-independent in the lambda-uniform convention adopted by
            this gridder (plan §4.2 line 1350 G3 + G12 are the only DEC-
            dependent inputs and live in M3 chunk 4's ``uv_table``
            builder, not here). Today ``dec_deg`` enters the pattern
            *only* through the hash; the geometric build is identical
            for all declinations.
        n_grid: gridder grid side length. Defaults to
            :data:`dsart.common.constants.N_GRID_DEFAULT` (= 256).
            Pattern fill fraction is monotone non-increasing in
            ``n_grid`` (plan §3 line 305 m1 pin).
        kernel_support: cells of gridding kernel support. Defaults to
            :data:`dsart.common.constants.KERNEL_SUPPORT_DEFAULT` (= 1,
            pillbox / nearest cell). Values > 1 are reserved for the M3
            hardening pass (Gaussian taper, plan §4.2 line 1351 G7);
            this chunk validates the input and folds it into
            ``pattern_id`` so a future taper extension cannot silently
            reuse an old-K pattern, but does NOT widen the per-baseline
            cell write-set.
        is_core_baseline_mask: optional ``(NBASE,)`` bool, True where
            both antennas of the baseline are in the 82-ant core
            (per plan §3 line 452 / V-4 outrigger zero-fill convention).
            None ⇒ keep all cross-baselines. Autos are always excluded.
        antpos_hash: optional pre-computed ``compute_antpos_hash``
            return value, to avoid re-hashing the same arrays in tight
            loops (the bench builds patterns for many ``(chgroup, n_grid,
            dec)`` combos against the same antpos). None ⇒ compute here.
        chgroup_table_hash: optional pre-computed
            ``compute_chgroup_table_hash`` value. None ⇒ compute here
            from the package constants.

    Returns:
        :class:`SparsityPattern` with ``ix_row``, ``ix_col`` lex-sorted
        for byte-reproducibility (plan §3 line 306 promises bit-
        identical arrays from the same inputs).

    Raises:
        ValueError: on any out-of-range / shape-mismatched input.
    """
    if antpos_e.shape != (NANTS,) or antpos_n.shape != (NANTS,):
        raise ValueError(
            f"antpos shapes must be ({NANTS},); got E={antpos_e.shape}, "
            f"N={antpos_n.shape}"
        )
    if not 0 <= chgroup < len(NU_CHGROUP_TOP_GHZ):
        raise ValueError(
            f"chgroup={chgroup} out of range 0..{len(NU_CHGROUP_TOP_GHZ) - 1}"
        )
    if n_grid <= 0:
        raise ValueError(f"n_grid={n_grid} must be > 0")
    # Pow-of-2 is enforced downstream by SparseCOOPayload (M1 contract;
    # required by the search-side iFFT). build_pattern itself only needs
    # n_grid ≤ 65535 (uint16 cell-index packing — see below). The bench
    # exercises n_grid=384 (non-pow-of-2) for fill-fraction reporting,
    # so we accept it here even though the wire-format payload rejects
    # it. Sub-agents who feed a non-pow-of-2 pattern into the production
    # transport will fail at the SparseCOOPayload construction step,
    # not silently here.
    if kernel_support <= 0:
        raise ValueError(f"kernel_support={kernel_support} must be > 0")
    if kernel_support != 1:
        # Documented carve-out: not silently dropped, just not implemented.
        # Tests assert this raises so future-K consumers don't get a
        # stale pillbox pattern.
        raise NotImplementedError(
            f"kernel_support={kernel_support} > 1 (Gaussian taper) is "
            f"reserved for the M3 hardening pass per plan §4.2 line 1351 "
            f"(G7); chunk 3a only ships pillbox (kernel_support=1)."
        )

    if antpos_hash is None:
        antpos_hash = compute_antpos_hash(antpos_e, antpos_n)
    if chgroup_table_hash is None:
        chgroup_table_hash = compute_chgroup_table_hash()
    dec_deg_quant = quantise_dec_deg(dec_deg)

    pattern_id = _blake2b_u64(_pattern_id_payload(
        chgroup=chgroup,
        dec_deg_quant=dec_deg_quant,
        n_grid=n_grid,
        kernel_support=kernel_support,
        antpos_hash=antpos_hash,
        chgroup_table_hash=chgroup_table_hash,
    ))

    # ---- Geometric build (DEC-independent in the lambda-uniform
    #      convention adopted here; see docstring) ----------------------
    du_m, dv_m = _per_baseline_uv_meters(
        antpos_e, antpos_n, is_core_baseline_mask=is_core_baseline_mask
    )

    # Per-channel wavelengths over this chgroup's 384 fine channels.
    nu_GHz = np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / (nu_GHz * 1e9)            # (NCHAN,)

    # Outer product → (N_kept, NCHAN) in λ. Float64 throughout to keep
    # cell rounding deterministic at the 10⁻⁶-cell level (the rint() at
    # the end discards the tail).
    u_lam = du_m[:, None] / wavelength_m[None, :]                     # (Nkept, NCHAN)
    v_lam = dv_m[:, None] / wavelength_m[None, :]

    # F20 negation (see module docstring). Apply once here so the
    # downstream gridder + sparse-scatter inherit the convention without
    # having to think about it.
    u_lam = -u_lam
    v_lam = -v_lam

    # Cell scale per plan §3 line 305: ``duv = max_baseline_lambda /
    # (N_grid/2)``  ⇔  ``cell_lambda = max_baseline_lambda * 2 / N_grid``.
    # ``max_baseline_lambda`` is the largest ``|u|`` or ``|v|`` over all
    # kept baselines AT THE TOP OF THIS CHGROUP (smallest wavelength,
    # largest λ-units value). Computing it from the outer-product
    # tensor makes the cell scale automatically track outrigger
    # exclusion + the chgroup-local frequency.
    max_baseline_lambda = float(np.max(np.maximum(
        np.abs(u_lam), np.abs(v_lam),
    )))
    if max_baseline_lambda == 0.0:
        raise ValueError(
            "max_baseline_lambda == 0 (degenerate antpos / mask); "
            "cannot build a sparsity pattern."
        )
    cell_lambda = max_baseline_lambda * 2.0 / n_grid

    half = n_grid // 2
    ix_col = np.rint(u_lam / cell_lambda).astype(np.int64) + half     # u-axis
    ix_row = np.rint(v_lam / cell_lambda).astype(np.int64) + half     # v-axis

    # Drop cells outside the grid (no toroidal wrap; matches
    # ``grid_uv_natural`` semantics).
    in_grid = (
        (ix_row >= 0) & (ix_row < n_grid)
        & (ix_col >= 0) & (ix_col < n_grid)
    )
    ix_row_flat = ix_row[in_grid].astype(np.uint32)
    ix_col_flat = ix_col[in_grid].astype(np.uint32)

    # Deduplicate + sort for bit-reproducibility (plan §3 line 306).
    # Pack (row, col) into a single uint32 key for unique() — N_grid
    # ≤ 65535 by SparseCOOPayload.__post_init__ (uint16 wire field), so
    # both halves fit in 16 bits.
    if n_grid > 65535:
        raise ValueError(
            f"n_grid={n_grid} > 65535 cannot be packed into uint16 cell"
            f" indices (wire-format limit, contracts.py)."
        )
    keys = (ix_row_flat.astype(np.uint32) << 16) | ix_col_flat.astype(np.uint32)
    unique_keys = np.unique(keys)                                     # sorted ascending
    ix_row_u = ((unique_keys >> 16) & 0xFFFF).astype(np.uint16)
    ix_col_u = (unique_keys & 0xFFFF).astype(np.uint16)
    n_filled = int(unique_keys.size)

    return SparsityPattern(
        ix_row=ix_row_u,
        ix_col=ix_col_u,
        pattern_id=pattern_id,
        n_grid=n_grid,
        n_filled=n_filled,
        chgroup=chgroup,
        dec_deg_quant=dec_deg_quant,
        kernel_support=kernel_support,
        antpos_hash=antpos_hash,
        chgroup_table_hash=chgroup_table_hash,
    )


def predict_pattern_id(
    *,
    chgroup: int,
    dec_deg: float,
    n_grid: int = N_GRID_DEFAULT,
    kernel_support: int = KERNEL_SUPPORT_DEFAULT,
    antpos_e: np.ndarray | None = None,
    antpos_n: np.ndarray | None = None,
    antpos_hash: int | None = None,
    chgroup_table_hash: int | None = None,
) -> int:
    """Compute ``pattern_id`` without building the full pattern.

    Used in the ``cmd: prepare`` discovery handshake — both ends compute
    ``predict_pattern_id`` from inputs they already share, then exchange
    a tiny ACK to verify they match before committing to the (much
    larger) :func:`build_pattern` call. Same hash semantics as
    :func:`build_pattern.pattern_id`.

    Args:
        chgroup, dec_deg, n_grid, kernel_support: as in
            :func:`build_pattern`.
        antpos_e, antpos_n: optional antpos arrays; if provided, used
            to compute ``antpos_hash`` via :func:`compute_antpos_hash`.
        antpos_hash: optional pre-computed antpos hash. Must be provided
            if ``antpos_e``/``antpos_n`` are not.
        chgroup_table_hash: optional pre-computed chgroup-table hash.
            Defaults to :func:`compute_chgroup_table_hash` over the
            package constants.

    Returns:
        64-bit unsigned int matching
        :attr:`SparsityPattern.pattern_id` for the same inputs.
    """
    if antpos_hash is None:
        if antpos_e is None or antpos_n is None:
            raise ValueError(
                "predict_pattern_id requires either antpos_hash, or both "
                "antpos_e and antpos_n."
            )
        antpos_hash = compute_antpos_hash(antpos_e, antpos_n)
    if chgroup_table_hash is None:
        chgroup_table_hash = compute_chgroup_table_hash()
    dec_deg_quant = quantise_dec_deg(dec_deg)
    return _blake2b_u64(_pattern_id_payload(
        chgroup=chgroup,
        dec_deg_quant=dec_deg_quant,
        n_grid=n_grid,
        kernel_support=kernel_support,
        antpos_hash=antpos_hash,
        chgroup_table_hash=chgroup_table_hash,
    ))

