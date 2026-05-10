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
# Vacuum speed of light (CODATA 2018; matches bfCorr's `CVAC` macro).
# ---------------------------------------------------------------------------
# Note: ``src/dsart/cal/cal_loader.py`` defines a local ``SPEED_OF_LIGHT_M_S``
# with the same value (M3 chunk 1 / F21). The duplicate is harmless; both
# paths are pinned to the same CODATA literal and the cal_loader copy is
# scheduled to be migrated to import from here during M3 hardening. Added
# in M3 chunk 3d (online injector) which needs it for the per-(ant, ch)
# phasor table; importing from ``cal_loader`` would create a hard dep
# from ``inject/`` on the cal layer.

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
"""Vacuum speed of light in metres per second."""


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

SPARSE_COO_BITS_VALID: tuple[int, ...] = (16, 32)
"""Valid bits_per_cell values, in bits-per-COMPLEX-cell convention (matches
both §9 ops-table (line 2327) and §4.3 wire format (line 1384)):
  - 16 = cint8 complex (operational; 8 bits real + 8 bits imag)
  - 32 = cfp16 complex (debug; 16 bits real + 16 bits imag)
M1 plan fix F2 originally proposed bits-per-component {8, 16}; reverted at
hardening because §9's link-budget table uses bits-per-complex-cell and is
the dominant convention. The §4.2 line 1296 `quantize_per_block(..., bits=8,
complex=True)` literal is the per-component value (different parameter, same
quantize)."""


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


# ---------------------------------------------------------------------------
# Slow correlator (M2; plan §8 lines 2161-2177)
# ---------------------------------------------------------------------------

LEGACY_FLUFF_SCALE: float = 0.05
"""4-bit voltage fluff scale factor. Pinned by legacy
`dsaX_bfCorr.cu::corr_input_copy` (lines 273-286): each signed-4-bit
nibble is sign-extended to a fp16 in [-8, 7] and then multiplied by 0.05.
M2's `slow_corr_kernel.unpack_int4_complex` mirrors this convention so
auto-correlation amplitudes carry the same physical units as the legacy
`dsaX_bfCorr` output. Per D2 in M2_PLAN_FIXES.md, NO other gain
calibration is applied in `corr_slow_compute`."""

BADA_NPOL: int = 2
"""Slow-correlator output `npol` per D1 in M2_PLAN_FIXES.md: 2 parallel-pol
products (XX = pol_0 × pol_0, YY = pol_1 × pol_1) — NOT the 4-hand
{XX, XY, YX, YY} form. Matches legacy `bada` block layout consumed by
`meridian_fringestop` (`dsamfs/utils.py::read_buffer(reader, nbls,
nchan, npol=2)`)."""

BADA_BYTES_PER_INTEGRATION: int = NBASE * NCHAN_PER_CHGROUP * BADA_NPOL * 8
"""Bada block size = NBASE × NCHAN × NPOL × sizeof(complex64) = 28,606,464.
Pinned by Subagent A's reading of `dsaX_bfCorr.cu` lines 1455-1460
(`output_size = NBASE*NCHAN_PER_PACKET*2*2*4`) and `dsamfs/utils.py`
lines 141-155 (`data.view(np.float32).reshape(-1, 2).view(np.complex64)
.reshape(-1, nbls, nchan, npol)`). Per D8 in M2_PLAN_FIXES.md, dtype is
`complex64` (NOT `cfp16`)."""

FADA_BYTES_PER_BLOCK: int = (
    BLOCK_SAMPLES_SPECNUM * NANTS * NCHAN_PER_CHGROUP * 2 * NPOL * 1
)
"""Fada block size = NPACKETS × NANTS × NCHAN × 2t × 2p × 1 byte
(4-bit cplx packed = 1 byte / complex sample) = 301,989,888.
Matches `configs/config_corr.yaml::buffers.fada.bytes_per_block`."""


# ---------------------------------------------------------------------------
# Fast-vis sparse uv-grid (M3 chunk 3a; plan §3 lines 305-309 + §4.2 line 1350)
# ---------------------------------------------------------------------------

N_GRID_DEFAULT: int = 256
"""Default fast-vis uv-grid side length (cells per axis). Plan §3 line 305
+ §3 line 309 (size table). Power-of-two for the search-side iFFT."""

KERNEL_SUPPORT_DEFAULT: int = 1
"""Default gridding kernel support in cells (1 = nearest-cell pillbox).
Plan §3 line 306 + §4.2 line 1351 (G7) — wider Gaussian supports
(K ∈ {3, 5, 7}) reserved for the M3 hardening pass; the chunk 3a
gridder ships pillbox + leaves the LUT plumbing as a future follow-up."""

PATTERN_ID_PERSON: bytes = b"dsart-pattern"
"""``hashlib.blake2b(person=...)`` personalisation tag for ``pattern_id``
hashing (plan §3 line 307). Pinned here as a single source of truth so
both ``corr_fast_compute`` (M3) and ``dsart-search-rx`` (M5) compute the
same hash. **Length must be ≤ 16 bytes** (blake2b person-string limit);
``len(b"dsart-pattern") == 13`` ✓."""

PATTERN_DEC_QUANT_DEG: float = 0.25
"""Quantisation bin (degrees) for ``dec_deg`` going into ``pattern_id``.
Plan §3 line 307: "dec_deg quantised to 0.25 deg". Bins are
``round(dec_deg / 0.25) * 0.25``; declinations within ± 0.125 deg of
each other share a pattern.

This bin width was chosen so that pattern reuse covers a full transit
(the array's primary-beam HWHM is ~ 1.75° at 1.4 GHz, so a 0.25°
quantisation costs ≤ 0.06 cells of (u, v) drift even at the longest
core baseline) without producing observably different cell rounding."""

PATTERN_HASH_BYTES: int = 8
"""``pattern_id`` is the leading ``PATTERN_HASH_BYTES`` of a blake2b
digest, interpreted as little-endian uint64 (plan §3 line 307: 64-bit
hash). This must equal 8 to match the wire-format header field
``SparseCOOPayloadHeader.pattern_id`` (uint64)."""


# ---------------------------------------------------------------------------
# RFI flagger (M3 chunk 3c; plan §4.2 step 2 / step 3 cold-start)
# ---------------------------------------------------------------------------

RFI_BANDPASS_TAU_S: float = 30.0
"""Plan §4.2 step 3 ``τ_B`` — running-bandpass IIR time constant (s).
The bandpass-outlier ``B_running`` 1-pole IIR uses this as its time
constant; the cold-start window before SK and group-outlier suffice
on their own is ``5·τ_B`` ≈ 150 s.

The chunk-3c bandpass-outlier (per-cube static median+MAD) does not
itself need an IIR warmup, but the warmup state machine inside
:class:`dsart.rfi.combine.RFIFlagger` honours the ``5·τ_B`` cube
window so the live ``corr_fast_compute`` service (parent M3 agent)
can swap the IIR form into the same architecture without touching
the warmup contract. Pinned in plan §4.2 step 3 derivation."""

RFI_BANDPASS_WARMUP_CUBES_DEFAULT: int = int(
    round(5.0 * RFI_BANDPASS_TAU_S / BLOCK_DURATION_S)
)
"""Default cold-start warmup window in cubes — = ``5·τ_B / 134.218 ms`` ≈
1118 cubes at the canonical ``τ_B = 30 s``. During this window
``flags.bit4 = rfi_warming_up`` is set in the transport header and
the bandpass-outlier detector is bypassed; SK and group-outlier
remain active. Tests override this to a small integer (1-5)."""

RFI_SK_FAR_DEFAULT: float = 1.0e-4
"""Default per-(ant, ch, pol, M) two-sided false-alarm rate for the SK
detector. Pinned in plan §4.2 step 2."""

RFI_BANDPASS_K_DEFAULT: float = 5.0
"""Default outlier-σ threshold for the bandpass-outlier detector
(briefing §4.2). MAD-σ units."""

RFI_GROUP_K_DEFAULT: float = 5.0
"""Default outlier-σ threshold for the group-outlier detector
(briefing §4.2). MAD-σ units."""

RFI_SK_M_VALUES_DEFAULT: tuple[int, ...] = (64, 256, 1024, 4096)
"""SK accumulation depths per cube. All four divide the canonical
4096-sample cube exactly (yielding ``N_acc ∈ {64, 16, 4, 1}``)."""

RFI_SUM_THRESHOLD_MAX_M_DEFAULT: int = 8
"""SumThreshold post-pass max sliding-window length (Offringa 2010
default)."""

RFI_SUM_THRESHOLD_ETA_DEFAULT: float = 1.5
"""SumThreshold post-pass threshold-shape parameter (Offringa 2010
default; per-window threshold is ``M / η^log2(M)``)."""


# ---------------------------------------------------------------------------
# Coarse-DM stage-2 (M3 chunk 3b; plan §4.2 step 8b + §3.6.2 stage-2 FIFO)
# ---------------------------------------------------------------------------

COARSE_DM_FIFO_DEPTH_DEFAULT: int = 4
"""Default depth (in cubes) of :class:`dsart.coarse_dm.Stage2FIFO`.

The corr-side production stage-2 FIFO depth is `Δt_samples_corr_stage2[g, c] /
t_int_factor` and is per-(chgroup, coarse_dm) (plan §3.6.2 / §4.2 streaming
pipeline lines 1322-1346) — but that contract is enforced at the
``corr_fast_compute`` integration site (chunk 4) which sizes per-(g, c)
against the canonical :class:`dsart.common.contracts.DmPlan.time_shift_corr_stage2`
table. The constant pinned here is the chunk-3b *FIFO container* default
depth (uniform across all (g, c) slots) — it sets the cube-count
capacity of the cross-coarse-DM detector-context window on the SEARCH
side (plan §3.6.12 ``T_det``) and the smoke-test transport-TX FIFO on
the corr side. M5's search-side detector (parent's coordination) reads
this default but is free to override per ``configs/config_compute_search.yaml``.
4 = ``ceil(T_det_default / cube_dt)`` at the default operating point
(``T_det = 512`` search samples × 524.288 µs / cube ≈ 134.218 ms ≈
``BLOCK_DURATION_S``)."""


# ---------------------------------------------------------------------------
# Detector noise-norm pins (M5 chunk 3; plan §3.6.9 / §3.6.10 / §3.6.12)
# ---------------------------------------------------------------------------

T_INT_FACTOR_DEFAULT: int = 16
"""Default native→search post-integration factor.
``t_int_search_us = T_INT_FACTOR_DEFAULT × NATIVE_SAMPLE_US = 16 ×
32.768 = 524.288 µs``. Matches ``configs/config_compute_corr.yaml::
t_int_factor`` and ``configs/operating_points.yaml::default``. (NB:
``t_int_fast_us = 8 × NATIVE_SAMPLE_US = 262.144`` is configured
independently in `configs/config_compute_corr.yaml`; the
fast→search ratio at the default operating point happens to be
``T_INT_FACTOR_DEFAULT / 8 = 2``, but the canonical ratio
sub-agents read against is the **native→search** one.)"""

T_INT_SEARCH_US_DEFAULT: float = T_INT_FACTOR_DEFAULT * NATIVE_SAMPLE_US
"""Search-stage sample period in µs (= 524.288 at default ops).
Pinned by plan §3.5 ``test_constants_pinned``: ``t_int_search_us ==
524.288``."""

CUBE_CADENCE_SAMPLES_DEFAULT: int = 256
"""Cube cadence in t_int_search_us samples (plan §3.6.12 line 1514: one
new cube every block-cadence = 134.218 ms = 256 search-cadence samples).
Search-node throughput is ``1 / (CUBE_CADENCE_SAMPLES_DEFAULT ×
T_INT_SEARCH_US_DEFAULT × 1e-6) ≈ 7.45 cubes/s`` at default ops."""

CUBE_CADENCE_S_DEFAULT: float = (
    CUBE_CADENCE_SAMPLES_DEFAULT * T_INT_SEARCH_US_DEFAULT * 1e-6
)
"""Cube cadence in seconds (= 0.134218 s at default ops). Used by the
Layer-2 EMA's ``γ = 1 - exp(-cube_cadence_s / τ_s)`` smoothing factor."""

N_KERNEL_MAX_T_DEFAULT: int = 128
"""Widest detector time-kernel boxcar width (plan §3.6.10 + §3.6.12).
Equals ``max(DETECTOR_K_TIME_WIDTHS)`` at default config and drives the
Layer-2 interior-only σ_k EMA's edge-trim length and the canonical-zone
emit gate's time-edge gate."""

NOISE_LAYER1_N_BURNIN_CUBES_DEFAULT: int = 5
"""Layer-1 σ-clipped global-scalar burn-in length (plan §3.6.9 line 997
+ ``configs/config_compute_search.yaml::noise.layer1_n_burnin_cubes``).
For the first 5 cubes after ``cmd: start``, σ_layer1[fdm] is the median
of the 5 most recent per-cube σs (robust against single-cube RFI burst
contamination); from cube 6 onward, σ uses the current cube only.
Tunable in §9: ``{1, 5 (default), 10}``."""

NOISE_LAYER2_TAU_S_DEFAULT: float = 30.0
"""Layer-2 EMA time constant in seconds (plan §3.6.10 line 1025;
``configs/config_compute_search.yaml::noise.layer2_tau_s``). At
``CUBE_CADENCE_S_DEFAULT`` this gives ``γ ≈ 0.00447`` (smoothing on a
~30 s window). Tunable in §9 + the noise-norm calibration bench (M5
DoD §8 line 2326): ``{30, 60}`` both must satisfy the FAR gate."""

NOISE_LAYER2_N_BURNIN_DEFAULT: int = 30
"""Layer-2 σ_k EMA burn-in length in cubes (plan §3.6.10 line 1026
+ ``configs/config_compute_search.yaml::noise.layer2_n_burnin``). Sets
``flags.bit3 = noise_warmup`` on every emitted Candidate during the
first 30 cubes after ``cmd: start``; from cube 30 onward the warmup
flag clears and the EMA replaces the Welford running mean."""

NOISE_SIGMA_CLIP_NSIGMA_DEFAULT: float = 3.0
"""σ-clip threshold for ``sigma_clipped_std`` (plan §3.6.9 line 985).
3-iteration loop with 3σ clip per iteration is the v1 default; deeper
clipping or a different kernel is out of scope for v1."""

NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT: int = 3
"""Number of σ-clip iterations (plan §3.6.9 line 988)."""
