"""Numerical and structural constants for the dsa110-rt pipeline.

Single source of truth for every constant referenced anywhere in
``src/dsart/`` or under ``tools/`` / ``tests/`` / ``bench/``. Values are
*derived* from primary constants where possible (e.g. ``NU_TOP_PROC_GHZ``
is computed from ``F0_CONF_GHZ`` + ``CH0_CHGROUP_0`` + ``DELTA_NU_CH_GHZ``
rather than re-stated as a literal), so a typo in one primary value is
caught by a downstream computed-value mismatch instead of silently
propagating.

Citation convention: each block notes the plan section that pins the
constant. M1 plan fixes (F8-F10 in ``M1_PLAN_FIXES.md``) are applied
here; the plan literals will be updated in the M1 hardening pass.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Time constants (plan §3.1 lines ~391-400)
# ---------------------------------------------------------------------------

NATIVE_SAMPLE_US: float = 32.768
"""SNAP voltage sample period (µs). Set by SNAP firmware; never changes."""

SPECNUM_PERIOD_US: float = 65.536
"""SNAP packet sequence number period (µs). 1 specnum = 2 native samples."""

BLOCK_SAMPLES_NATIVE: int = 4096
"""Native samples per PSRDADA block."""

BLOCK_SAMPLES_SPECNUM: int = 2048
"""Specnums per PSRDADA block (= BLOCK_SAMPLES_NATIVE // 2)."""

BLOCK_DURATION_S: float = BLOCK_SAMPLES_NATIVE * NATIVE_SAMPLE_US * 1e-6
"""Block duration in seconds (= 0.134217728 s exactly)."""

BLOCK_DURATION_MS: float = BLOCK_DURATION_S * 1000.0
"""Block duration in ms (≈ 134.218 ms; plan uses the rounded form)."""

T_INT_FAST_NATIVE: int = 8
"""Default fast-corr integration factor (native samples per fast-corr bin)."""

T_INT_FAST_US_DEFAULT: float = T_INT_FAST_NATIVE * NATIVE_SAMPLE_US
"""Default fast-corr post-integration cadence (µs) = 262.144."""


# ---------------------------------------------------------------------------
# Configured + processed band (plan §3.1 lines ~417-426)
# ---------------------------------------------------------------------------

F0_CONF_GHZ: float = 1.530
"""Configured band top (corr_setup_96.yaml::f0_GHz). Highest frequency in
the SNAP firmware's full digitised band."""

BW_CONF_GHZ: float = 0.250
"""Configured band width in GHz (corr_setup_96.yaml::bw_GHz)."""

N_CHAN_CONF: int = 8192
"""Configured channel count (corr_setup_96.yaml::nchan)."""

DELTA_NU_CH_GHZ: float = BW_CONF_GHZ / N_CHAN_CONF
"""Per-native-channel bandwidth (GHz). = 3.0517578125e-5 GHz exactly."""

# Per-corr-node slice
N_CHGROUP: int = 16
"""Number of corr nodes / chgroups."""

NCHAN_PER_CHGROUP: int = 384
"""Native channels per chgroup (corr_setup_96.yaml::nchan_spw)."""

N_CHAN_PROC_NATIVE: int = N_CHGROUP * NCHAN_PER_CHGROUP
"""Total native channels in the processed band (= 6144)."""

CH0_CHGROUP_0: int = 1024
"""First processed system channel index (chgroup 0's local ch=0)."""

# Processed band edges (plan §3.1 line ~418; M1 plan-fix F8 — edge convention).
#
# Both band ends use the EDGE convention (= upper edge of the channel that
# anchors that end of the band), so that:
#
#     NU_TOP_PROC_GHZ        ≡ freq_GHz(0, 0)       = f0 - 1024·Δν
#     NU_BOT_PROC_GHZ        ≡ freq_GHz(15, 383)    = f0 - 7167·Δν
#     NU_CHGROUP_TOP_GHZ[g]  = freq_GHz(g, 0)
#     NU_CHGROUP_BOT_GHZ[g]  = freq_GHz(g, 383)
#
# Critically, this gives  NU_CHGROUP_BOT_GHZ[15] == NU_BOT_PROC_GHZ  exactly,
# which is the precondition for `time_shift_corr_stage2[15, c] == 0` (plan §3.2
# line 577 / §3.6.2 stage-2 invariant) holding for ALL DMs, not just for low
# DMs where rint() happens to round to 0.
#
# Plan literal updates captured in M1_PLAN_FIXES.md F8/F9:
#   - NU_BOT_PROC_GHZ:  1.311265 GHz  →  1.311281 GHz  (was centre-of-ch-7167;
#                       now upper-edge-of-ch-7167 per the §3.1 closed-form).
#   - BW_PROC_MHZ:      187.485 MHz   →  187.469 MHz   (= 6143·Δν · 1000;
#                       was the asymmetric edge-top minus centre-bot value).
#   - Δτ_proc(DM=3000): 1699.5 ms     →  1697.5 ms     (F11; computed from
#                       the corrected literals + standard formula).
#
# Plan §3.1 line 408 chgroup-top table values (1498.75 MHz, 1487.03125, ...)
# remain correct: those are the EDGE values that NU_CHGROUP_TOP_GHZ already
# returned under the previous mixed convention.
NU_TOP_PROC_GHZ: float = F0_CONF_GHZ - CH0_CHGROUP_0 * DELTA_NU_CH_GHZ
"""Top of processed band (GHz) = upper edge of ch_sys=1024. 1.49875 GHz exact."""

NU_BOT_PROC_GHZ: float = F0_CONF_GHZ - (CH0_CHGROUP_0 + N_CHAN_PROC_NATIVE - 1) * DELTA_NU_CH_GHZ
"""Bottom of processed band (GHz) = upper edge of ch_sys=7167.
≈ 1.311281 GHz (= freq_GHz(15, 383); see edge-convention note above)."""

BW_PROC_GHZ: float = NU_TOP_PROC_GHZ - NU_BOT_PROC_GHZ
BW_PROC_MHZ: float = BW_PROC_GHZ * 1000.0
"""Processed bandwidth (MHz). = 6143 · DELTA_NU_CH_GHZ * 1000 ≈ 187.469 MHz
under the edge-edge convention. Plan literal 187.485 MHz was derived from the
old asymmetric convention and is updated by M1 plan-fix F9."""


# ---------------------------------------------------------------------------
# Chgroup → ch0 table (plan §3.1 lines 405-414)
# ---------------------------------------------------------------------------
# Hostnames in legacy config_dsa96_corr.yaml skip 'corr01' (hosts are
# corr00, corr02, ..., corr16 → chgroup indices 0..15). The CHGROUP
# INDEX maps deterministically to ch0; the HOSTNAME mapping is
# operator-pickable and lives in configs/chgroup_assignments.yaml.

CHGROUP_CH0: tuple[int, ...] = tuple(
    CH0_CHGROUP_0 + g * NCHAN_PER_CHGROUP for g in range(N_CHGROUP)
)
"""ch0[g] for g ∈ 0..15. (1024, 1408, 1792, ..., 6784)."""


def freq_GHz(chgroup: int, local_ch: int) -> float:
    """Frequency of (chgroup, local_ch) per plan §3.1 lines 421-425.

    Channel ordering is descending in frequency (chan_ascending=False
    in corr_setup_96.yaml): higher local_ch = lower frequency.
    """
    if not 0 <= chgroup < N_CHGROUP:
        raise ValueError(f"chgroup={chgroup}, expected 0..{N_CHGROUP - 1}")
    if not 0 <= local_ch < NCHAN_PER_CHGROUP:
        raise ValueError(f"local_ch={local_ch}, expected 0..{NCHAN_PER_CHGROUP - 1}")
    ch_sys = CHGROUP_CH0[chgroup] + local_ch
    return F0_CONF_GHZ - ch_sys * DELTA_NU_CH_GHZ


# Per-chgroup local band edges
NU_CHGROUP_TOP_GHZ: tuple[float, ...] = tuple(
    freq_GHz(g, 0) for g in range(N_CHGROUP)
)
"""Top of each chgroup's local band (highest ν, local_ch=0)."""

NU_CHGROUP_BOT_GHZ: tuple[float, ...] = tuple(
    freq_GHz(g, NCHAN_PER_CHGROUP - 1) for g in range(N_CHGROUP)
)
"""Bottom of each chgroup's local band (lowest ν, local_ch=383)."""


# ---------------------------------------------------------------------------
# Dispersion constant (plan §3.6.1 line 702)
# ---------------------------------------------------------------------------

K_DM_MS_GHZ2_PC: float = 4.148808
"""Dispersion constant: τ_ms = K · DM / ν_GHz². Units: ms · GHz² · cm³ / pc.
Standard pulsar-astronomy value; matches scratch gen_dmtrials.py."""


# ---------------------------------------------------------------------------
# Geometry (plan §3.1 line ~446; M1 plan fix F10)
# ---------------------------------------------------------------------------

PHI_LAT_OVRO_DEG: float = 37.234
"""OVRO array latitude (deg). Source value referenced in plan prose."""

PHI_LAT_OVRO_RAD: float = math.radians(PHI_LAT_OVRO_DEG)
"""OVRO array latitude (rad). Full math.radians(37.234) precision
(≈ 0.6498508034). Plan §3.1 literal '0.64980' is a 5-digit truncation
that mismatches by ~5e-4 rad (~3 arcmin); F10 captures the plan update."""


# ---------------------------------------------------------------------------
# Antenna + baseline counts (plan §3 line 300; §3.1 line ~437)
# ---------------------------------------------------------------------------

NANTS: int = 96
"""Total antennas in the merged voltage tensor."""

NPOL: int = 2
"""Polarization count on voltages and per-pol stats."""

NBASE: int = NANTS * (NANTS + 1) // 2
"""xGPU upper-triangle baseline count INCLUDING auto-correlations
(diagonal). = 4656. Plan §3 line 300 pins this."""


# ---------------------------------------------------------------------------
# Pipeline structural constants (plan §3 line ~313; §9)
# ---------------------------------------------------------------------------

N_CORR: int = 16
"""Corr nodes (= N_CHGROUP). Kept distinct in case of future refactor
where a chgroup is served by multiple corr nodes."""

N_SEARCH: int = 4
"""Search nodes."""

N_SEARCH_GPU: int = 2
"""GPUs per search node."""

T_DET_SAMPLES_DEFAULT: int = 512
"""Default detector cube depth in t_int_search_us samples (plan §3.6.12;
= ceil_to_block(256 + 128 + 128) = 2 blocks @ default ops point)."""


# ---------------------------------------------------------------------------
# DM plan defaults (plan §3.2 line 505)
# ---------------------------------------------------------------------------

DM_MIN_DEFAULT: float = 0.0
DM_MAX_DEFAULT: float = 3000.0
DM_TOL_DEFAULT: float = 1.5

DM_PLAN_METADATA_VERSION: int = 1
"""Schema version for the dm_plan.npz metadata dict (plan §3.2 line 570)."""


# ---------------------------------------------------------------------------
# Detector kernel banks (plan §3.1 lines 475-477)
# ---------------------------------------------------------------------------

DETECTOR_K_TIME_WIDTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
"""Detector boxcar widths in t_int_search_us samples. Power-of-two only."""

DETECTOR_K_DM_WIDTHS: tuple[int, ...] = (1, 3, 5, 7)
"""Detector boxcar widths in fine-DM bins. Odd only (centred boxcar)."""

DETECTOR_K_DM_WIDEST: int = max(DETECTOR_K_DM_WIDTHS)
"""Widest detector DM boxcar width (= 7). Drives dm_overlap_coarse halo
size in plan §3.2 step 5."""

DETECTOR_IMAGE_KERNELS: tuple[str, ...] = (
    "unit",
    "psf",
    "psf_shift_lm",
    "psf_shift_l",
)
"""Image-kernel namespace tokens (first token of kernel_id)."""

DETECTOR_DM_KERNELS: tuple[str, ...] = tuple(
    f"d{w}" for w in DETECTOR_K_DM_WIDTHS
)
"""DM-kernel namespace tokens (second token of kernel_id; e.g. 'd3')."""

DETECTOR_TIME_KERNELS: tuple[str, ...] = tuple(
    f"b{w}" for w in DETECTOR_K_TIME_WIDTHS
)
"""Time-kernel namespace tokens (third token of kernel_id; e.g. 'b16')."""


# ---------------------------------------------------------------------------
# Transport (plan §4.3 lines 1360-1394)
# ---------------------------------------------------------------------------

TRANSPORT_MAGIC: int = 0xD5A1107E
TRANSPORT_HEADER_VERSION: int = 1
TRANSPORT_HEADER_BYTES: int = 72
"""72-byte fixed header. Layout: see SparseCOOPayloadHeader in contracts.py."""

SPARSE_COO_BITS_VALID: tuple[int, ...] = (8, 16)
"""Valid bits_per_cell values: 8 = cint8 (operational), 16 = cfp16 (debug).
Plan §4.3 comment 'bits=16 or 32' is a typo; M1 plan fix F2."""


# ---------------------------------------------------------------------------
# Trigger packet (plan §3 lines 354-383)
# ---------------------------------------------------------------------------

TRIGGER_SCHEMA_VERSION: int = 1

TRIGGER_PRIORITIES: tuple[str, ...] = ("high", "normal", "low")

TRIGGER_ACK_STAGES: tuple[str, ...] = ("accepted", "completed")

TRIGGER_ACK_REASONS: tuple[str, ...] = (
    "dup",
    "ratelimit",
    "bad_schema",
    "listener_overloaded",
    "action_unsupported",
    "dump_queue_full",
)
"""Allowed values of TriggerAck.reason when accepted=False (plan §3 line 383
+ §4.5 line 1718 dump_queue_full)."""

TRIGGER_OPERATOR_SEARCH_NODE_ID: int = 255
"""search_node_id reserved for ctrltrigger operator-issued triggers
(plan §3.4 line 627; §4.6 line 1728). Detector-emitted candidates use 0..3."""


# ---------------------------------------------------------------------------
# Voltage tensor convention (plan §3 line 299; D1 lock in M1_PLAN_FIXES.md)
# ---------------------------------------------------------------------------

VOLTAGES_SHAPE: tuple[int, int, int, int, int] = (
    BLOCK_SAMPLES_SPECNUM,
    NANTS,
    NCHAN_PER_CHGROUP * N_CHGROUP // N_CHGROUP,  # 384, kept symbolic
    2,  # native samples per packet
    NPOL,
)
"""Merged voltage tensor shape: [2048 pkt, 96 ant, 384 ch, 2 t, 2 pol].
5-axis legacy layout per D1 — collapsing pkt×t into native_t=4096
would require a real transpose (~5-10 ms / block on GPU HBM); the
GEMM and voltage ring both expect this 5-axis form. Downstream code
that needs native_t derives it as `2*p + t` semantically."""

VOLTAGES_DTYPES_VALID: tuple[str, ...] = ("int8", "float16")
"""Valid in-process Voltages dtypes per D2: int8 fluffed (default,
quasi-no-op from int4 wire) or fp16 fluffed (debug). int4-on-wire is a
transport detail (plan §3 line 297) and lives in capture/, not contracts."""
