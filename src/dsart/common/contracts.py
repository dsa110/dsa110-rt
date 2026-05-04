"""Data-plane contract dataclasses (plan §3 + §3.2).

These six frozen, slotted dataclasses are the integration contract between
workstreams. Sub-agents wire functions against these types; the
``__post_init__`` shape/dtype/value asserts run only when ``DSART_TEST=1``
(see ``config_loader.DSART_TEST``) so the production hot path has zero
assert overhead.

Contracts:

    Voltages              — merged voltage tensor (in-process, GPU)
    SparseCOOPayload      — corr → search value vector (in-process; wire
                            byte layout in §4.3)
    Candidate             — detector output (search → trigger emitter)
    TriggerPacket         — emitted trigger (search → corr; JSON over TCP)
    TriggerAck            — corr → search ACK (two-stage; JSON over TCP)
    DmPlan                — DM plan struct (build_dm_plan.py output;
                            commits to configs/dm_plan.npz)

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
    TRIGGER_ACK_REASONS,
    TRIGGER_ACK_STAGES,
    TRIGGER_OPERATOR_SEARCH_NODE_ID,
    TRIGGER_PRIORITIES,
    TRIGGER_SCHEMA_VERSION,
    VOLTAGES_DTYPES_VALID,
    VOLTAGES_SHAPE,
)

__all__ = [
    "CandidateFlags",
    "Voltages",
    "SparseCOOPayload",
    "Candidate",
    "TriggerPacket",
    "TriggerAck",
    "DmPlan",
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
            for cint8 (operational; bits_per_cell=8) or ``float16`` for
            cfp16 (debug; bits_per_cell=16). Per-cell complex; the trailing
            axis carries the (real, imag) pair.
        bits_per_cell: 8 (cint8) or 16 (cfp16). See F2 — plan comment
            "16 or 32" was a typo.
        chgroup: 0..15.
        dm_idx: coarse DM trial index into ``DmPlan.coarse_dm``.
        specnum: block start specnum.
        utc_block_start_ns: UTC ns of ``specnum``. Carried in the wire
            header per §3 line 310. F1 in M1_PLAN_FIXES.md notes that
            ``Voltages`` does NOT carry this redundantly — it lives on
            the wire-payload side only.
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
    utc_block_start_ns: int
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
        expected_dtype = "int8" if self.bits_per_cell == 8 else "float16"
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
# TriggerPacket — plan §3 lines 355-372 + F4 fix
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriggerPacket:
    """Search → corr trigger (newline-delimited JSON over persistent TCP).

    Wire form: ``json.dumps(asdict(packet))``. Schema versioned via ``v``.

    ``trigger_id`` format is ``s<sid>-g<g>-<counter>`` per F4 (§4.4 long
    form, plan §3 example was abbreviated). Operator-issued triggers
    use ``search_node_id == TRIGGER_OPERATOR_SEARCH_NODE_ID (= 255)``
    and ``trigger_id == "op-<utc_ns>"``.

    Fields that the emitter derives (not present on Candidate):
      - ``trigger_id`` (per-(search_node, gpu) monotonic counter)
      - ``emit_utc_ns`` (now)
      - ``event_utc_ns`` (from ``event_specnum`` via specnum→UTC table)
      - ``actions``, ``priority``, ``src_name`` (from policy + trigger
        predicate; defaults are pinned in §4.4)
      - ``n_pre_blocks``, ``n_post_blocks`` (None → corr listener uses
        config defaults of 10/5).
      - ``fine_dm_trial`` (= ``dm_idx`` in v1 per O-7 trim).
    """

    trigger_id: str
    search_node_id: int
    emit_utc_ns: int
    event_specnum: int
    event_utc_ns: int
    l: float
    m: float
    dm_fine: float
    dm_idx: int
    fine_dm_trial: int
    width_samples: int
    kernel_id: str
    snr: float
    actions: dict
    priority: str
    src_name: str
    n_pre_blocks: Optional[int] = None
    n_post_blocks: Optional[int] = None
    v: int = TRIGGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        if self.v != TRIGGER_SCHEMA_VERSION:
            raise ValueError(
                f"v={self.v}, expected {TRIGGER_SCHEMA_VERSION}"
            )
        sid = self.search_node_id
        if not (0 <= sid < N_SEARCH or sid == TRIGGER_OPERATOR_SEARCH_NODE_ID):
            raise ValueError(
                f"search_node_id={sid}, expected 0..{N_SEARCH - 1} "
                f"or {TRIGGER_OPERATOR_SEARCH_NODE_ID} (operator)"
            )
        if not self.trigger_id:
            raise ValueError("trigger_id must be non-empty")
        if self.priority not in TRIGGER_PRIORITIES:
            raise ValueError(
                f"priority={self.priority!r} not in {TRIGGER_PRIORITIES}"
            )
        _check_kernel_id(self.kernel_id)
        if self.event_specnum < 0:
            raise ValueError(f"event_specnum={self.event_specnum}, expected ≥ 0")
        if self.dm_fine < 0:
            raise ValueError(f"dm_fine={self.dm_fine}, expected ≥ 0")
        if self.dm_idx < 0:
            raise ValueError(f"dm_idx={self.dm_idx}, expected ≥ 0")
        if self.width_samples <= 0:
            raise ValueError(f"width_samples={self.width_samples}, expected > 0")
        if self.n_pre_blocks is not None and self.n_pre_blocks < 0:
            raise ValueError(f"n_pre_blocks={self.n_pre_blocks}, expected ≥ 0")
        if self.n_post_blocks is not None and self.n_post_blocks < 0:
            raise ValueError(f"n_post_blocks={self.n_post_blocks}, expected ≥ 0")
        if not isinstance(self.actions, dict):
            raise TypeError(
                f"actions must be dict; got {type(self.actions).__name__}"
            )


# ---------------------------------------------------------------------------
# TriggerAck — plan §3 lines 374-382
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriggerAck:
    """Corr → search ACK record (newline-delimited JSON; two stages).

    The corr listener sends two ACK records per trigger over the same
    TCP connection: one with ``stage="accepted"`` (within ~ms) and one
    with ``stage="completed"`` (after dump completes; ~100-700 ms).

    Stage-specific fields are optional on the dataclass; ``__post_init__``
    enforces presence per ``stage`` value. JSON serialisation drops
    ``None``-valued fields per §4.5 line 1718.

    ``reason`` is non-None only when ``accepted=False``. Allowed values
    in ``TRIGGER_ACK_REASONS`` per §3 line 383 + §4.5 line 1718
    ``dump_queue_full`` (which §3 missed; F2/F-related).
    """

    trigger_id: str
    stage: str
    ack_utc_ns: int
    accepted: Optional[bool] = None
    reason: Optional[str] = None
    queue_depth: Optional[int] = None
    dup_of: Optional[str] = None
    voltage_dump_path: Optional[str] = None
    filterbank_paths: Optional[tuple] = None  # tuple of str (frozen-friendly)
    dump_completion_utc_ns: Optional[int] = None
    dump_duration_ms: Optional[int] = None
    v: int = TRIGGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not DSART_TEST:
            return
        if self.v != TRIGGER_SCHEMA_VERSION:
            raise ValueError(f"v={self.v}, expected {TRIGGER_SCHEMA_VERSION}")
        if self.stage not in TRIGGER_ACK_STAGES:
            raise ValueError(
                f"stage={self.stage!r} not in {TRIGGER_ACK_STAGES}"
            )
        if not self.trigger_id:
            raise ValueError("trigger_id must be non-empty")
        if self.stage == "accepted":
            if self.accepted is None:
                raise ValueError("stage='accepted' requires `accepted` bool")
            if not self.accepted:
                if self.reason is None or self.reason not in TRIGGER_ACK_REASONS:
                    raise ValueError(
                        f"reason={self.reason!r} required and must be in "
                        f"{TRIGGER_ACK_REASONS} when accepted=False"
                    )
                if self.reason == "dup" and not self.dup_of:
                    raise ValueError("reason='dup' requires `dup_of`")
        elif self.stage == "completed":
            if self.dump_completion_utc_ns is None:
                raise ValueError(
                    "stage='completed' requires `dump_completion_utc_ns`"
                )
            if self.dump_duration_ms is None or self.dump_duration_ms < 0:
                raise ValueError(
                    f"dump_duration_ms={self.dump_duration_ms}, expected ≥ 0"
                )
            if self.filterbank_paths is not None and not isinstance(
                self.filterbank_paths, tuple
            ):
                raise TypeError(
                    f"filterbank_paths must be tuple; got "
                    f"{type(self.filterbank_paths).__name__}"
                )


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

    See plan §3.2 lines 542-571 for the full schema. The CSR pair
    (``fine_offsets_idx``, ``fine_offsets_flat``) replaces the legacy
    list-of-ragged-arrays representation that npz cannot round-trip.
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
