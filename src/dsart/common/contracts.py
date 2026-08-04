"""Data-plane contract dataclasses (plan §3 + §3.2 + M6 chunk 1).

These frozen, slotted dataclasses are the integration contract between
workstreams. Sub-agents wire functions against these types; the
``__post_init__`` shape/dtype/value asserts run only when ``DSART_TEST=1``
(see ``config_loader.DSART_TEST``) so the production hot path has zero
assert overhead.

Contracts:

    Voltages              — merged voltage tensor (in-process, GPU)
    SparseCOOPayload      — corr → search value vector (in-process; wire
                            byte layout in §4.3)
    Candidate             — detector output (search → clusterer)
    DmPlan                — DM plan struct (build_dm_plan.py output;
                            commits to configs/dm_plan.npz)
    CubeGeometry          — per-cube geometric metadata (M6 chunk 1)
    ClusterRecord         — clusterer output, one row per cluster (M6 ch1)
    CubeDumpManifest      — sidecar metadata for each NPZ cube dump (M6 ch3)

(The pre-M6-pivot ``TriggerPacket`` / ``TriggerAck`` contracts were
dropped in the M6 chunk-9 hardening sweep — voltage-trigger handoff is
operator-mediated via the legacy ``dsa110-xengine`` framework, see
plan §M6 / §M-defer. A future revival of the original-M6 voltage-
trigger workstream will re-lock these under whatever new transport is
chosen.)

All fields use ``Final``-style typing — re-binding fails at runtime via
``frozen=True``. Mutating ndarray *contents* is still possible (frozen
only protects field rebinding), but downstream consumers must not.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config_loader import DSART_TEST
from .constants import (
    DETECTOR_DM_KERNELS,
    DETECTOR_IMAGE_KERNELS,
    DETECTOR_TIME_KERNELS,
    DM_PLAN_METADATA_VERSION,
    NCHAN_PER_CHGROUP,
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    SPARSE_COO_BITS_VALID,
    VOLTAGES_DTYPES_VALID,
    VOLTAGES_SHAPE,
)

__all__ = [
    "CandidateFlags",
    "Voltages",
    "SparseCOOPayload",
    "Candidate",
    "DmPlan",
    "CubeGeometry",
    "ClusterRecord",
    "CubeDumpManifest",
]


# ---------------------------------------------------------------------------
# Candidate.flags bit table (M1 plan fix F5; D5 lock)
# ---------------------------------------------------------------------------


class CandidateFlags(enum.IntFlag):
    """Bit mask for ``Candidate.flags`` (plan §3.1 flags table; F5/D5).

    Bits 0-2 reserved for future use (Layer-3 / off-zenith-rejection).
    """

    NONE = 0
    NOISE_WARMUP = 1 << 3        # Layer-2 σ EMA still in burn-in
    RFI_WARMING_UP = 1 << 4      # corr-side RFI Stats A/B/C burn-in
    HALO_DROPPED = 1 << 5        # candidate at halo edge of canonical zone
    TIME_EDGE_DROPPED = 1 << 6   # candidate at T_det edge (warmup or wrap)


def _check_kernel_id(kernel_id: str) -> None:
    """Validate ``kernel_id`` shape ``"k_img:k_dm:k_time"`` (plan §3.1 line 475)."""
    parts = kernel_id.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"kernel_id must be 'k_img:k_dm:k_time'; got {kernel_id!r}"
        )
    img, dm, t = parts
    if img not in DETECTOR_IMAGE_KERNELS:
        raise ValueError(
            f"kernel_id image token {img!r} not in {DETECTOR_IMAGE_KERNELS}"
        )
    if dm not in DETECTOR_DM_KERNELS:
        raise ValueError(
            f"kernel_id dm token {dm!r} not in {DETECTOR_DM_KERNELS}"
        )
    if t not in DETECTOR_TIME_KERNELS:
        raise ValueError(
            f"kernel_id time token {t!r} not in {DETECTOR_TIME_KERNELS}"
        )


# ---------------------------------------------------------------------------
# Voltages — plan §3 line 299 + D1/D2 locks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Voltages:
    """Merged voltage tensor on GPU (D1 lock: 5-axis legacy layout).

    Tensor shape [2048 pkt, 96 ant, 384 ch, 2 t, 2 pol]. ``dtype`` is
    int8 (default; quasi-no-op fluffing of int4-on-wire) or float16
    (debug). int4-on-wire is a transport detail and not represented here.

    Args:
        tensor: ndarray with shape ``VOLTAGES_SHAPE``.
        specnum0: SNAP packet sequence number at the FIRST sample of
            this block (uint64). Block-period is BLOCK_SAMPLES_SPECNUM
            (= 2048) specnums.
        utc_block_start_ns: UTC timestamp of ``specnum0`` in nanoseconds
            since Unix epoch. Derived from ``specnum0`` + the on-host
            specnum-to-UTC table; carried explicitly to avoid recomputing
            in every consumer.
    """

    tensor: np.ndarray
    specnum0: int
    utc_block_start_ns: int

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        t = self.tensor
        if not isinstance(t, np.ndarray):
            raise TypeError(f"Voltages.tensor must be np.ndarray; got {type(t).__name__}")
        if t.shape != VOLTAGES_SHAPE:
            raise ValueError(
                f"Voltages.tensor.shape {t.shape} != expected {VOLTAGES_SHAPE}"
            )
        if t.dtype.name not in VOLTAGES_DTYPES_VALID:
            raise TypeError(
                f"Voltages.tensor.dtype {t.dtype.name!r} not in {VOLTAGES_DTYPES_VALID}"
            )
        if not isinstance(self.specnum0, int) or self.specnum0 < 0:
            raise ValueError(f"specnum0 must be non-negative int; got {self.specnum0!r}")
        if not isinstance(self.utc_block_start_ns, int) or self.utc_block_start_ns < 0:
            raise ValueError(
                f"utc_block_start_ns must be non-negative int; got {self.utc_block_start_ns!r}"
            )


# ---------------------------------------------------------------------------
# SparseCOOPayload — plan §3 line 310 + §4.3 lines 1360-1394 + F1/F2/F3 fixes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SparseCOOPayload:
    """Corr → search single-band value vector (in-process form).

    Wire byte layout is in plan §4.3 (72-byte header + payload). This
    dataclass is the *assembled* per-(coarse_dm, time_bin, chgroup)
    value vector after defragmentation. The transport layer (M4) owns
    the on-the-wire (header, frag_idx, n_frags, seq, etc.) decomposition.

    Args:
        values: ndarray ``[N_filled, 2]`` (real, imag pair). dtype ``int8``
            for cint8 (operational; bits_per_cell=16) or ``float16`` for
            cfp16 (debug; bits_per_cell=32). Per-cell complex; the trailing
            axis carries the (real, imag) pair.
        bits_per_cell: bits-per-COMPLEX-cell — 16 (cint8 complex) or 32
            (cfp16 complex). Matches §9 ops-table convention (line 2327)
            and §4.3 wire format (line 1384). See revised F2 in
            M1_PLAN_FIXES.md.
        chgroup: 0..15.
        dm_idx: coarse DM trial index into ``DmPlan.coarse_dm``.
        specnum: block start specnum. UTC is recovered from this via the
            §3.5 conversion at the receiver (F1: ``utc_block_start_ns`` is
            NOT carried — neither here nor on the wire — to avoid drift
            between the wire form and the dataclass).
        n_grid: gridder grid size (``N_grid``) at the time this payload
            was assembled. Allows the receiver to verify ``n_filled``
            against the cached pattern table.
        n_filled: ``len(values)``. Stored to avoid recomputing.
        pattern_id: blake2b_64 hash of ``(chgroup, dec_quant_0p25, n_grid,
            kernel_support, antpos_hash, chgroup_table_hash)``. Used by
            the receiver to certify input alignment (§3 line 307).
        t_int: post-integration factor in t_int_fast units (typically 16
            at default ops point per O-4).
        scale: cint8 dequant scale (per-block, computed over filled cells).
        offset: cint8 dequant offset (per-block).
    """

    values: np.ndarray
    bits_per_cell: int
    chgroup: int
    dm_idx: int
    specnum: int
    n_grid: int
    n_filled: int
    pattern_id: int
    t_int: int
    scale: float
    offset: float

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        v = self.values
        if not isinstance(v, np.ndarray):
            raise TypeError(f"values must be np.ndarray; got {type(v).__name__}")
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(
                f"values shape must be [N_filled, 2] (re, im); got {v.shape}"
            )
        if v.shape[0] != self.n_filled:
            raise ValueError(
                f"n_filled={self.n_filled} != values.shape[0]={v.shape[0]}"
            )
        if self.bits_per_cell not in SPARSE_COO_BITS_VALID:
            raise ValueError(
                f"bits_per_cell {self.bits_per_cell} not in {SPARSE_COO_BITS_VALID}"
            )
        expected_dtype = "int8" if self.bits_per_cell == 16 else "float16"
        if v.dtype.name != expected_dtype:
            raise TypeError(
                f"values.dtype {v.dtype.name!r} != {expected_dtype!r} "
                f"for bits_per_cell={self.bits_per_cell}"
            )
        if not 0 <= self.chgroup < N_CHGROUP:
            raise ValueError(f"chgroup={self.chgroup}, expected 0..{N_CHGROUP - 1}")
        if self.dm_idx < 0:
            raise ValueError(f"dm_idx={self.dm_idx}, expected ≥ 0")
        if self.n_grid <= 0 or self.n_grid & (self.n_grid - 1):
            raise ValueError(f"n_grid={self.n_grid}, expected positive power of two")
        if self.n_filled <= 0:
            raise ValueError(f"n_filled={self.n_filled}, expected > 0")
        if self.t_int <= 0:
            raise ValueError(f"t_int={self.t_int}, expected > 0")
        if not (0 <= self.pattern_id < (1 << 64)):
            raise ValueError(f"pattern_id={self.pattern_id} not a uint64")


# ---------------------------------------------------------------------------
# Candidate — plan §3 lines 320-352 + F5/D5 (flags table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """Detector output (search → trigger emitter), 11 fields per O-7 trim.

    Schema is the integration contract between detector and trigger
    emitter. Adding fields with defaults is backwards-compatible; removing
    or reinterpreting fields requires a v2 schema bump.

    See plan §3 line 320-352 for field semantics. ``flags`` uses
    ``CandidateFlags`` (D5 lock) — store as ``int`` for stable ABI.
    """

    l: float
    m: float
    dm_fine: float
    dm_idx: int
    event_specnum: int
    width_samples: int
    kernel_id: str
    snr: float
    detector_version: str
    flags: int
    search_node_id: int
    gpu_half: int

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        if self.dm_fine < 0:
            raise ValueError(f"dm_fine={self.dm_fine}, expected ≥ 0")
        if self.dm_idx < 0:
            raise ValueError(f"dm_idx={self.dm_idx}, expected ≥ 0")
        if self.event_specnum < 0:
            raise ValueError(f"event_specnum={self.event_specnum}, expected ≥ 0")
        if self.width_samples <= 0:
            raise ValueError(f"width_samples={self.width_samples}, expected > 0")
        _check_kernel_id(self.kernel_id)
        if not 0 <= self.search_node_id < N_SEARCH:
            raise ValueError(
                f"search_node_id={self.search_node_id}, expected 0..{N_SEARCH - 1}"
            )
        if not 0 <= self.gpu_half < N_SEARCH_GPU:
            raise ValueError(
                f"gpu_half={self.gpu_half}, expected 0..{N_SEARCH_GPU - 1}"
            )
        if self.flags < 0 or self.flags >= (1 << 32):
            raise ValueError(f"flags={self.flags}, expected uint32")


# ---------------------------------------------------------------------------
# DmPlan — plan §3.2 lines 542-571
# ---------------------------------------------------------------------------


_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "band_top_GHz",
    "band_bot_GHz",
    "BW_MHz",
    "N_chan_proc_native",
    "t_int_fast_us",
    "t_int_search_us",
    "tol",
    "build_utc_ns",
    "git_sha",
    "version",
)


@dataclass(frozen=True, slots=True)
class DmPlan:
    """DM plan struct produced by ``tools/build_dm_plan.py`` (§3.2).

    Round-trips through ``configs/dm_plan.npz`` via ``to_npz()`` /
    ``from_npz()``. All array fields use the dtypes pinned in §3.2.

    See plan §3.2 lines 542-571 for the full schema and
    ``constants.py::DM_PLAN_METADATA_VERSION`` for the v1 → v2 schema
    history (the v2 even-K-around-coarse partition landed for M7.2 per
    user direction 2026-05-18). The CSR pair (``fine_offsets_idx``,
    ``fine_offsets_flat``) replaces the legacy list-of-ragged-arrays
    representation that npz cannot round-trip.

    Sign conventions (v2):
      • ``time_shift_corr_stage1`` and ``time_shift_corr_stage2`` remain
        non-negative (corr-side alignment always shifts to ν_bot_proc).
      • ``time_shift_search`` is SIGNED int32 — fines BELOW their coarse
        carry negative shifts (read PAST data from the rolling RX ring;
        naturally available), fines ABOVE carry positive shifts (read
        FUTURE data delivered by the corr-side one-sided rewind).
        ``time_shift_search[:, 15]`` is identically 0 (ν_chgroup_bot[15]
        == ν_bot_proc by construction).
      • ``fine_offsets_flat`` (= δdm in pc/cm³) is SIGNED under v2.

    Per-(search, GPU) ranges (v2):
      • ``dm_idx_range_canonical_per_gpu[s, g] = (i, i)`` where
        ``i = N_SEARCH_GPU * s + g``. Each search GPU owns EXACTLY ONE
        coarse cube (no halo, no inter-GPU coarse overlap).
      • Per-GPU fine-DM ownership is the K = N_fine/N_coarse contiguous
        fines ``fine_dm[(2s+g)*K : (2s+g+1)*K]`` (derivable from
        ``fine_to_coarse[f] = f // K``; not stored separately).
      • ``dm_overlap_coarse = 0``.
    """

    dm_min: float
    dm_max: float
    tol: float
    fine_dm: np.ndarray
    coarse_dm: np.ndarray
    fine_to_coarse: np.ndarray
    fine_offsets_idx: np.ndarray
    fine_offsets_flat: np.ndarray
    time_shift_corr_stage1: np.ndarray
    time_shift_corr_stage2: np.ndarray
    time_shift_search: np.ndarray
    dm_idx_range_canonical: np.ndarray
    dm_idx_range_consumed: np.ndarray
    dm_idx_range_canonical_per_gpu: np.ndarray
    dm_idx_range_consumed_per_gpu: np.ndarray
    dm_overlap_coarse: int
    metadata: dict

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        n_fine = self.fine_dm.shape[0]
        n_coarse = self.coarse_dm.shape[0]
        self._check_array("fine_dm", "float64", (n_fine,))
        self._check_array("coarse_dm", "float64", (n_coarse,))
        self._check_array("fine_to_coarse", "int32", (n_fine,))
        self._check_array("fine_offsets_idx", "int32", (n_coarse + 1,))
        self._check_array(
            "fine_offsets_flat",
            "float64",
            (int(self.fine_offsets_idx[-1]),),
        )
        self._check_array(
            "time_shift_corr_stage1",
            "int32",
            (N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse),
        )
        self._check_array(
            "time_shift_corr_stage2", "int32", (N_CHGROUP, n_coarse)
        )
        self._check_array(
            "time_shift_search", "int32", (n_fine, N_CHGROUP)
        )
        self._check_array(
            "dm_idx_range_canonical", "int32", (N_SEARCH, 2)
        )
        self._check_array(
            "dm_idx_range_consumed", "int32", (N_SEARCH, 2)
        )
        self._check_array(
            "dm_idx_range_canonical_per_gpu",
            "int32",
            (N_SEARCH, N_SEARCH_GPU, 2),
        )
        self._check_array(
            "dm_idx_range_consumed_per_gpu",
            "int32",
            (N_SEARCH, N_SEARCH_GPU, 2),
        )

        if self.dm_min < 0 or self.dm_max <= self.dm_min:
            raise ValueError(
                f"dm_min={self.dm_min}, dm_max={self.dm_max}: require 0 ≤ dm_min < dm_max"
            )
        if self.tol <= 0:
            raise ValueError(f"tol={self.tol}, expected > 0")
        if self.dm_overlap_coarse < 0:
            raise ValueError(
                f"dm_overlap_coarse={self.dm_overlap_coarse}, expected ≥ 0"
            )
        # Strictly increasing fine_dm
        if n_fine >= 2 and not np.all(np.diff(self.fine_dm) > 0):
            raise ValueError("fine_dm must be strictly increasing")
        # Strictly increasing coarse_dm
        if n_coarse >= 2 and not np.all(np.diff(self.coarse_dm) > 0):
            raise ValueError("coarse_dm must be strictly increasing")
        # Metadata keys
        missing = set(_REQUIRED_METADATA_KEYS) - set(self.metadata.keys())
        if missing:
            raise ValueError(
                f"DmPlan.metadata missing required keys: {sorted(missing)}"
            )
        if self.metadata["version"] != DM_PLAN_METADATA_VERSION:
            raise ValueError(
                f"metadata.version={self.metadata['version']}, "
                f"expected {DM_PLAN_METADATA_VERSION}"
            )

    def _check_array(self, name: str, dtype: str, shape: tuple) -> None:
        arr = getattr(self, name)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"DmPlan.{name} must be np.ndarray; got {type(arr).__name__}")
        if arr.dtype.name != dtype:
            raise TypeError(f"DmPlan.{name}.dtype = {arr.dtype.name!r}, expected {dtype!r}")
        if arr.shape != shape:
            raise ValueError(f"DmPlan.{name}.shape = {arr.shape}, expected {shape}")

    # ------------------------------------------------------------------
    # NPZ round-trip
    # ------------------------------------------------------------------

    def to_npz(self, path: str) -> None:
        """Write to a NumPy .npz archive. Metadata is JSON-serialised.

        Per §3.2 line 542-571: scalars are stored as 0-d ndarrays, ranges
        and shifts as named arrays, metadata as a JSON-encoded ndarray
        of dtype 'U' (avoids object pickle).
        """
        import json

        np.savez(
            path,
            dm_min=np.asarray(self.dm_min, dtype="float64"),
            dm_max=np.asarray(self.dm_max, dtype="float64"),
            tol=np.asarray(self.tol, dtype="float64"),
            fine_dm=self.fine_dm,
            coarse_dm=self.coarse_dm,
            fine_to_coarse=self.fine_to_coarse,
            fine_offsets_idx=self.fine_offsets_idx,
            fine_offsets_flat=self.fine_offsets_flat,
            time_shift_corr_stage1=self.time_shift_corr_stage1,
            time_shift_corr_stage2=self.time_shift_corr_stage2,
            time_shift_search=self.time_shift_search,
            dm_idx_range_canonical=self.dm_idx_range_canonical,
            dm_idx_range_consumed=self.dm_idx_range_consumed,
            dm_idx_range_canonical_per_gpu=self.dm_idx_range_canonical_per_gpu,
            dm_idx_range_consumed_per_gpu=self.dm_idx_range_consumed_per_gpu,
            dm_overlap_coarse=np.asarray(self.dm_overlap_coarse, dtype="int32"),
            metadata=np.asarray(json.dumps(self.metadata), dtype="U"),
        )

    @classmethod
    def from_npz(cls, path: str) -> "DmPlan":
        """Load from a NumPy .npz archive."""
        import json

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls(
                dm_min=float(data["dm_min"]),
                dm_max=float(data["dm_max"]),
                tol=float(data["tol"]),
                fine_dm=data["fine_dm"].astype("float64", copy=False),
                coarse_dm=data["coarse_dm"].astype("float64", copy=False),
                fine_to_coarse=data["fine_to_coarse"].astype("int32", copy=False),
                fine_offsets_idx=data["fine_offsets_idx"].astype("int32", copy=False),
                fine_offsets_flat=data["fine_offsets_flat"].astype("float64", copy=False),
                time_shift_corr_stage1=data["time_shift_corr_stage1"].astype("int32", copy=False),
                time_shift_corr_stage2=data["time_shift_corr_stage2"].astype("int32", copy=False),
                time_shift_search=data["time_shift_search"].astype("int32", copy=False),
                dm_idx_range_canonical=data["dm_idx_range_canonical"].astype("int32", copy=False),
                dm_idx_range_consumed=data["dm_idx_range_consumed"].astype("int32", copy=False),
                dm_idx_range_canonical_per_gpu=data["dm_idx_range_canonical_per_gpu"].astype("int32", copy=False),
                dm_idx_range_consumed_per_gpu=data["dm_idx_range_consumed_per_gpu"].astype("int32", copy=False),
                dm_overlap_coarse=int(data["dm_overlap_coarse"]),
                metadata=metadata,
            )


# ---------------------------------------------------------------------------
# CubeGeometry — M6 chunk 1 (per-cube geometric metadata for clusterer +
# T1/T2 logger; passed alongside the per-cube candidate list)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubeGeometry:
    """Per-cube geometric metadata used by the clusterer to convert
    detector pixel/index coordinates to real-unit physical coordinates
    for ``ClusterRecord`` and T1/T2 ASCII logging (M6 D1/D3).

    The detector emits ``Candidate`` records with ``l`` and ``m`` as
    float-cast pixel indices (see ``detector.decoder.decode_local_max``)
    and ``dm_fine`` as the resolved fine-DM in pc cm⁻³. The clusterer
    needs (a) the pixel→radian conversion to write real-unit T1/T2
    rows, and (b) the cube-specnum origin to convert ``event_specnum``
    to a sample index ``t_in_cube``. Both come from this struct.

    Args:
        cube_id: monotonic cube counter (matches CubeRingSlot.cube_id).
        specnum_start: spec num at sample 0 of this cube. Used to
            compute t_in_cube = (event_specnum - specnum_start) //
            sample_period_specnum.
        sample_period_specnum: spec-nums per detector sample (= 16 in
            production: t_int_search_us / t_int_fast_us).
        t_det: number of time samples in the cube (= 256 in production).
        n_grid: grid size on each spatial axis (= 256 in production).
        n_fdm_in_cube: number of fine-DM trials in this cube
            (= len(fine_dm_pc_cc)).
        sample_period_us: time between adjacent detector samples in
            microseconds (= t_int_search_us in DmPlan.metadata).
        cell_l_rad: l-axis pixel pitch in radians (positive scalar).
            Computed by the gridder from λ / (n_grid · cell_λ).
        cell_m_rad: m-axis pixel pitch in radians (positive scalar).
        l0_rad: l value at pixel index 0 (offset to true sky origin).
            Default 0.0 = pixel-centred grid.
        m0_rad: m value at pixel index 0. Default 0.0.
        fine_dm_pc_cc: fine-DM grid in pc cm⁻³, monotonic increasing,
            shape ``[n_fdm_in_cube]`` float64.
        mjd_start: double-precision MJD at sample 0 of this cube.
            Computed by the caller (production: from the host
            specnum→UTC table; bench: from utc_block_start_ns).

    Notes:
        ``fine_dm_pc_cc`` is a *view* of the cube's fine-DM column
        (typically a slice of ``DmPlan.fine_dm`` for the consumed range
        of this (search_node, gpu_half)). The clusterer treats it as
        read-only; mutations leak across cubes.
    """

    cube_id: int
    specnum_start: int
    sample_period_specnum: int
    t_det: int
    n_grid: int
    n_fdm_in_cube: int
    sample_period_us: float
    cell_l_rad: float
    cell_m_rad: float
    l0_rad: float
    m0_rad: float
    fine_dm_pc_cc: np.ndarray
    mjd_start: float

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        if self.cube_id < 0:
            raise ValueError(f"cube_id={self.cube_id}, expected ≥ 0")
        if self.specnum_start < 0:
            raise ValueError(f"specnum_start={self.specnum_start}, expected ≥ 0")
        if self.sample_period_specnum <= 0:
            raise ValueError(
                f"sample_period_specnum={self.sample_period_specnum}, expected > 0"
            )
        if self.t_det <= 0:
            raise ValueError(f"t_det={self.t_det}, expected > 0")
        if self.n_grid <= 0 or self.n_grid & (self.n_grid - 1):
            raise ValueError(f"n_grid={self.n_grid}, expected positive power of two")
        if self.n_fdm_in_cube <= 0:
            raise ValueError(f"n_fdm_in_cube={self.n_fdm_in_cube}, expected > 0")
        if self.sample_period_us <= 0.0:
            raise ValueError(f"sample_period_us={self.sample_period_us}, expected > 0")
        if self.cell_l_rad <= 0.0:
            raise ValueError(f"cell_l_rad={self.cell_l_rad}, expected > 0")
        if self.cell_m_rad <= 0.0:
            raise ValueError(f"cell_m_rad={self.cell_m_rad}, expected > 0")
        if not isinstance(self.fine_dm_pc_cc, np.ndarray):
            raise TypeError(
                f"fine_dm_pc_cc must be np.ndarray; got {type(self.fine_dm_pc_cc).__name__}"
            )
        if self.fine_dm_pc_cc.dtype.name != "float64":
            raise TypeError(
                f"fine_dm_pc_cc.dtype = {self.fine_dm_pc_cc.dtype.name!r}, expected 'float64'"
            )
        if self.fine_dm_pc_cc.shape != (self.n_fdm_in_cube,):
            raise ValueError(
                f"fine_dm_pc_cc.shape = {self.fine_dm_pc_cc.shape}, "
                f"expected ({self.n_fdm_in_cube},)"
            )
        if not np.isfinite(self.mjd_start):
            raise ValueError(f"mjd_start={self.mjd_start} not finite")


# ---------------------------------------------------------------------------
# ClusterRecord — M6 chunk 1 (one record per cluster from
# cluster.forward.cluster_candidates; the integration contract between the
# clusterer and the ASCII logger / cube-dump predicate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClusterRecord:
    """One cluster record emitted by ``cluster.forward.cluster_candidates``
    (M6 D1/D3/D6).

    Per M6 D6 the clusterer is per-cube; ``cluster_id`` is unique within
    a (cube_id, search_node_id, gpu_half) triple, NOT globally. Noise
    points (HDBSCAN/DBSCAN label = -1) are emitted as singleton records
    with ``cluster_id = -1`` and ``cntc = cntb_lm = cntb_dm = 1``.

    The peak (highest-SNR) candidate's full feature set is flattened into
    this record. The integer-index fields (l_pix, m_pix, fine_dm_idx,
    t_in_cube) carry the detector-frame coordinates; the real-unit
    fields (l_rad, m_rad, dm_fine_pc_cc, t_seconds) are derived from
    those through the ``CubeGeometry`` sidecar at clustering time. T1/T2
    log rows always write the real units (D1/D3).

    Args:
        cluster_id: per-cube cluster id (-1 for HDBSCAN/DBSCAN noise
            points; ≥0 for actual clusters).
        cube_id: monotonic cube counter (= CubeGeometry.cube_id).
        cntc: cluster cardinality (number of candidates in cluster;
            = 1 for noise points).
        cntb_lm: number of unique (l_pix, m_pix) cells in cluster.
        cntb_dm: number of unique fine_dm_idx trials in cluster.
        peak_candidate_idx: index of the peak (highest-SNR) candidate
            in the per-cube input candidate list.
        l_rad: peak's l in radians (= l_pix * cell_l_rad + l0_rad).
        m_rad: peak's m in radians (= m_pix * cell_m_rad + m0_rad).
        l_pix: peak's cube pixel index ∈ [0, n_grid).
        m_pix: peak's cube pixel index ∈ [0, n_grid).
        dm_fine_pc_cc: peak's fine DM in pc cm⁻³.
        fine_dm_idx: peak's index into CubeGeometry.fine_dm_pc_cc
            ∈ [0, n_fdm_in_cube).
        t_in_cube: peak's sample index ∈ [0, t_det).
        t_seconds: peak's time in seconds since cube start
            (= t_in_cube * sample_period_us / 1e6).
        width_samples: peak's matched-filter width in samples.
        snr: peak's score in σ.
        kernel_id: peak's detector kernel triple id ("k_img:k_dm:k_time").
        event_specnum: peak's absolute spec num.
        search_node_id: 0..N_SEARCH-1.
        gpu_half: 0..N_SEARCH_GPU-1.
    """

    cluster_id: int
    cube_id: int
    cntc: int
    cntb_lm: int
    cntb_dm: int
    peak_candidate_idx: int
    l_rad: float
    m_rad: float
    l_pix: int
    m_pix: int
    dm_fine_pc_cc: float
    fine_dm_idx: int
    t_in_cube: int
    t_seconds: float
    width_samples: int
    snr: float
    kernel_id: str
    event_specnum: int
    search_node_id: int
    gpu_half: int

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        if self.cluster_id < -1:
            raise ValueError(
                f"cluster_id={self.cluster_id}, expected ≥ -1 (-1 = noise)"
            )
        if self.cube_id < 0:
            raise ValueError(f"cube_id={self.cube_id}, expected ≥ 0")
        if self.cntc <= 0:
            raise ValueError(f"cntc={self.cntc}, expected > 0")
        if self.cntb_lm <= 0 or self.cntb_lm > self.cntc:
            raise ValueError(
                f"cntb_lm={self.cntb_lm}, expected 0 < cntb_lm ≤ cntc={self.cntc}"
            )
        if self.cntb_dm <= 0 or self.cntb_dm > self.cntc:
            raise ValueError(
                f"cntb_dm={self.cntb_dm}, expected 0 < cntb_dm ≤ cntc={self.cntc}"
            )
        if self.peak_candidate_idx < 0:
            raise ValueError(
                f"peak_candidate_idx={self.peak_candidate_idx}, expected ≥ 0"
            )
        if self.l_pix < 0:
            raise ValueError(f"l_pix={self.l_pix}, expected ≥ 0")
        if self.m_pix < 0:
            raise ValueError(f"m_pix={self.m_pix}, expected ≥ 0")
        if self.dm_fine_pc_cc < 0:
            raise ValueError(f"dm_fine_pc_cc={self.dm_fine_pc_cc}, expected ≥ 0")
        if self.fine_dm_idx < 0:
            raise ValueError(f"fine_dm_idx={self.fine_dm_idx}, expected ≥ 0")
        if self.t_in_cube < 0:
            raise ValueError(f"t_in_cube={self.t_in_cube}, expected ≥ 0")
        if self.width_samples <= 0:
            raise ValueError(f"width_samples={self.width_samples}, expected > 0")
        if self.event_specnum < 0:
            raise ValueError(f"event_specnum={self.event_specnum}, expected ≥ 0")
        _check_kernel_id(self.kernel_id)
        if not 0 <= self.search_node_id < N_SEARCH:
            raise ValueError(
                f"search_node_id={self.search_node_id}, expected 0..{N_SEARCH - 1}"
            )
        if not 0 <= self.gpu_half < N_SEARCH_GPU:
            raise ValueError(
                f"gpu_half={self.gpu_half}, expected 0..{N_SEARCH_GPU - 1}"
            )


# ---------------------------------------------------------------------------
# CubeDumpManifest — M6 chunk 3 (sidecar metadata for each NPZ cube dump)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubeDumpManifest:
    """Sidecar metadata for each NPZ cube dump (M6 D7).

    Holds the bookkeeping needed to: (a) re-load the dump, (b) attribute
    the dump back to its source trigger (auto-clustered cluster vs UDP),
    and (c) audit the writer-thread queue from the operator side.

    The NPZ file itself contains the cube tensor + a compact summary
    of these fields; this dataclass is the in-process form used by the
    cube-dump writer thread (M6 chunk 3) and the cube-dump bench (M6
    chunk 7).

    Args:
        cube_id: monotonic cube counter (matches CubeRingSlot.cube_id).
        event_specnum_start: spec num at sample 0 of the dumped cube.
        mjd_start: double-precision MJD at sample 0 of dumped cube.
        t_det: number of time samples in the dumped cube.
        n_fdm_in_cube: number of fine-DM trials in the dumped cube.
        n_grid: grid size on each spatial axis.
        trigger_source: "auto" (cluster crossed bright-pulse predicate)
            or "udp" (external UDP listener fired).
        cluster_record: peak ClusterRecord for "auto" dumps; None for
            "udp" dumps (which don't carry per-candidate metadata —
            see M6 D9/D12).
        npz_path: absolute path to the NPZ file on disk.
        search_node_id: 0..N_SEARCH-1 (the dumping process).
        gpu_half: 0..N_SEARCH_GPU-1 (the dumping process).
    """

    cube_id: int
    event_specnum_start: int
    mjd_start: float
    t_det: int
    n_fdm_in_cube: int
    n_grid: int
    trigger_source: str
    cluster_record: Optional[ClusterRecord]
    npz_path: str
    search_node_id: int
    gpu_half: int
    #: TRUE spec num / MJD at cube sample 0 (2026-08-04).
    #:
    #: ``event_specnum_start`` above is documented as sample 0, and is
    #: that on the auto/udp paths -- but
    #: ``dump/c2_trigger_listener._build_manifest`` overwrites it with
    #: ``packet.event_specnum`` (the TRIGGER specnum) because the NPZ
    #: filename is composed from it and C3 + the dashboards glob on
    #: ``cube_s*_g*_<trigger_specnum>.npz``. The consequence is that in
    #: every archived C2-triggered dump ``(event_specnum -
    #: event_specnum_start) / sample_period_specnum == 0``, so the
    #: detector's own in-cube time index is unrecoverable offline and the
    #: cube's phase relative to the 128-search-sample corr blocks cannot
    #: be computed. That has now blocked two separate investigations
    #: (placing the burst in 260801rmep/bdga, and testing whether
    #: 260804jbpj's t=203 break lands on a corr-block seam).
    #:
    #: Rather than repurpose ``event_specnum_start`` and break the
    #: filename contract, these carry the real anchor. ``None`` on
    #: manifests built before this existed; readers must fall back.
    cube_specnum_start: Optional[int] = None
    cube_mjd_start: Optional[float] = None
    #: spec-nums per detector sample, needed to turn the anchor into a
    #: time index: ``t = (event_specnum - cube_specnum_start) //
    #: sample_period_specnum``. Recorded rather than assumed because it
    #: is derived from the live op-point (t_int_search_us /
    #: t_int_fast_us) -- CubeGeometry's docstring still says "= 16 in
    #: production", which was true when t_int_fast_native was 2; at the
    #: current 32 it is 1. Guessing it silently scales every offline
    #: time index by 16x.
    sample_period_specnum: Optional[int] = None

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        if self.cube_id < 0:
            raise ValueError(f"cube_id={self.cube_id}, expected ≥ 0")
        if self.event_specnum_start < 0:
            raise ValueError(
                f"event_specnum_start={self.event_specnum_start}, expected ≥ 0"
            )
        if not np.isfinite(self.mjd_start):
            raise ValueError(f"mjd_start={self.mjd_start} not finite")
        if self.t_det <= 0:
            raise ValueError(f"t_det={self.t_det}, expected > 0")
        if self.n_fdm_in_cube <= 0:
            raise ValueError(f"n_fdm_in_cube={self.n_fdm_in_cube}, expected > 0")
        if self.n_grid <= 0 or self.n_grid & (self.n_grid - 1):
            raise ValueError(
                f"n_grid={self.n_grid}, expected positive power of two"
            )
        if self.trigger_source not in ("auto", "udp"):
            raise ValueError(
                f"trigger_source={self.trigger_source!r}, expected 'auto' or 'udp'"
            )
        if self.trigger_source == "auto" and self.cluster_record is None:
            raise ValueError(
                "trigger_source='auto' requires cluster_record to be non-None"
            )
        if self.trigger_source == "udp" and self.cluster_record is not None:
            raise ValueError(
                "trigger_source='udp' requires cluster_record to be None"
            )
        if not self.npz_path:
            raise ValueError("npz_path must be non-empty")
        if not 0 <= self.search_node_id < N_SEARCH:
            raise ValueError(
                f"search_node_id={self.search_node_id}, expected 0..{N_SEARCH - 1}"
            )
        if not 0 <= self.gpu_half < N_SEARCH_GPU:
            raise ValueError(
                f"gpu_half={self.gpu_half}, expected 0..{N_SEARCH_GPU - 1}"
            )
