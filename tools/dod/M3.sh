#!/usr/bin/env bash
# M3 Definition-of-Done (§8 line 2262-2294) — runs on h01.
#
# M3 is the corr-node fast-vis path: corr_fast_compute reads fada → autos →
# RFI flagger → cal+bandpass+DEC-phase → fast-corr GEMM → grid → static-sky →
# coarse-DM → quantize → enqueue for transport. Far more moving parts than
# M2; built incrementally as chunks. This file starts as a SKELETON: it
# gates on the substrate (M3_preflight + PARALLEL_AGENTS.md conventions +
# M2 hardened) and emits stage="in progress (substrate only)" until the
# first real M3 chunk lands and adds the next gate.
#
# As each M3 chunk lands, the chunk's authoring sub-agent (or this file's
# editor) appends a new STEP block here and updates the stage-gating
# logic. The DoD never claims more than what's actually been built.
#
# Stage labels (mirrors M0/M1/M2 JSON shape consumed by Mn+1):
#   - failed                              -> some STEP failed; exit 1
#   - in progress (substrate only)         -> preflight + M2-gate pass; no
#                                             M3 chunks landed yet
#   - in progress (chunks: <list>)        -> some chunks complete, more
#                                             remain
#   - complete (needs operator approval)  -> all chunks pass auto, no marker
#   - complete (approved)                 -> marker present, M3_PLAN_FIXES.md
#                                             still in repo
#   - complete (hardened)                 -> marker present + M3_PLAN_FIXES.md
#                                             retired (M3 hardening)
#
# §6 conda-activate shell pattern: 'set -u' DROPPED (conda MKL hook
# references MKL_INTERFACE_LAYER without default); 'pipefail' kept.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# PARALLEL_AGENTS.md §4 conventions. Caller may override but the canonical
# h01 M3 run uses these.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m3}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m3}"

M3_STATUS_JSON="${M3_STATUS_JSON:-${HOME}/dsart-m3-status.json}"
M2_STATUS_JSON="${M2_STATUS_JSON:-${HOME}/dsart-m2-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"

# Per-milestone flock guard (PARALLEL_AGENTS.md §4.4). Falls back to /tmp
# if /var/lock isn't writable (some distros restrict it).
M3_LOCKFILE="${M3_LOCKFILE:-/var/lock/dsart-m3.lock}"
if ! touch "${M3_LOCKFILE}" 2>/dev/null; then
  M3_LOCKFILE="/tmp/dsart-m3.lock"
fi

# Operator-approval marker (mirrors D11 from M2).
M3_OPERATOR_APPROVAL_FILE="${M3_OPERATOR_APPROVAL_FILE:-${REPO_ROOT}/bench/reports/M3/m_operator_approved.yaml}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M3:${STEP}] FAIL $*"
  cat > "${M3_STATUS_JSON}" <<JSON
{"milestone": "M3", "stage": "failed", "step": "${STEP}", "host": "$(hostname -s)", "phase": "a", "utc_iso": "$(date -u +%FT%TZ)"}
JSON
  exit 1
}
pass() {
  echo "[M3:${STEP}] PASS"
}
warn() {
  echo "[M3:${STEP}] WARN $*"
}

# Per-milestone flock guard. Prevents two parallel M3.sh runs from stomping
# each other. M5.sh has its own lockfile so M3 and M5 can run side-by-side.
exec {LOCKFD}>"${M3_LOCKFILE}" || { echo "FATAL: cannot open ${M3_LOCKFILE}"; exit 1; }
if ! flock -n "$LOCKFD"; then
  echo "FATAL: another M3 run already in progress (lock=${M3_LOCKFILE})"
  exit 1
fi
echo "== M3 DoD: lock acquired (${M3_LOCKFILE}) =="

echo "== M3 DoD: gate on M3_preflight =="
bash "${SCRIPT_DIR}/M3_preflight.sh"

STEP="m1_status"
[[ -f "${M1_STATUS_JSON}" ]] || fail "missing ${M1_STATUS_JSON}"
python3 - <<PY || fail "M1 status JSON did not validate"
import json
s = json.load(open("${M1_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M1", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
print(f"M1 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m2_status"
# M2 must be hardened (preferred) or approved (tolerated).
[[ -f "${M2_STATUS_JSON}" ]] || fail "missing ${M2_STATUS_JSON}"
python3 - <<PY || fail "M2 status JSON did not validate"
import json
s = json.load(open("${M2_STATUS_JSON}"))
stage = s.get("stage", "")
assert stage.startswith("complete"), f"M2 stage not complete: {s!r}"
assert s.get("milestone", "") == "M2", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
print(f"M2 stage: {stage!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m3_substrate_files"
# The minimum substrate that must exist BEFORE any M3 chunk fires. This
# list grows as chunks land.
declare -a M3_SUBSTRATE_FILES=(
  "PARALLEL_AGENTS.md"
  "M3_PLAN_FIXES.md"
  "tools/dod/M3.sh"
  "tools/dod/M3_preflight.sh"
)
for f in "${M3_SUBSTRATE_FILES[@]}"; do
  [[ -f "${REPO_ROOT}/${f}" ]] || fail "missing substrate ${f}"
done
echo "  ${#M3_SUBSTRATE_FILES[@]} substrate files present"
pass

# ---------------------------------------------------------------------------
# M3 chunk gates — append below as chunks land.
# ---------------------------------------------------------------------------
#
# Each chunk adds:
#   STEP="<chunk_name>"
#   <invocations / pytest / bench>
#   pass
#
# When the last chunk gates pass, the operator-approval block + final
# stamp logic at the bottom of this file kicks in (mirrors M2.sh lines
# 213-280).

# --- chunk 1: cal-apply with F21 DEC-only phase fold (M3_PLAN_FIXES.md F21) -
# Validates the fast-corr cal-tensor loader: complex cal blob → fine-channel
# upsample → optional phase-only normalisation / pol-swap → F21 fringe-stop
# phase fold → broadcast-ready torch tensors. Four acceptance tests pin the
# sign convention to bfCorr's central-beam formula (iArm==1, bm=127) and
# verify on-source phase cancellation, off-source residual phase, and a
# parity guard against F20.
STEP="chunk_1_cal_apply_with_F21_dec_phase"
python -m pytest tests/test_cal_loader_dec_phase.py -q --tb=short \
  || fail "F21 cal-loader acceptance pytests failed"
pass

# --- chunk 2a: FastCorrKernel (peer to SlowCorrKernel) ---------------------
# 24 acceptance tests covering input validation, output shape per t_int,
# numerical sanity (zero / autos / Hermitian), the F18 + F21 composition
# (on-source vis is real to fp16 precision), and the helper functions.
# Note the test_full_block_equals_slow_corr_kernel boundary test runs the
# full 2048-packet correlation on CPU with both kernels; ~3 min wall time.
# That's the strongest cross-kernel correctness pin and only needs to pass
# once per release; chunks 4 / 5 will add GPU tests with smaller blocks.
STEP="chunk_2a_fast_corr_kernel"
python -m pytest tests/test_fast_corr_kernel.py -q --tb=short \
  || fail "FastCorrKernel acceptance pytests failed"
pass

# --- chunk 2b: corr_fast_compute service shell -----------------------------
# 8 smoke tests for the production compute_block() spine that wires together
# unpack_int4_split (M2) → load_cal_with_dec_phase (chunk 1) → FastCorrKernel
# (chunk 2a) → Stokes-I sum. Tests cover (a) shape/dtype contract for the
# default no-cal full-block tile, (b) parameterised t_int_fast_native sweep
# across 8/16/32/4096 packets per tile, (c) the cal-zero ⇒ outrigger-baseline
# zero contract using the real h01 cal blob, (d) device pinning propagation,
# and (e) zero-voltage handling. Wall time ~12 min on CPU, dominated by the
# four full-block correlations in (b). The bulk of cal-loader / kernel
# correctness is already pinned by chunks 1 / 2a above; this chunk just
# verifies the wiring.
STEP="chunk_2b_corr_fast_compute_service_shell"
python -m pytest tests/test_corr_fast_compute_pipeline.py -q --tb=short \
  || fail "corr_fast_compute service-shell smoke tests failed"
pass

# --- chunk 3a: sparsity pattern + fast-vis gridder -------------------------
# 31 acceptance tests across two suites:
# - tests/test_sparsity_pattern.py (20): SparsityPattern dataclass shape/
#   dtype + DEC quantisation + pattern_id sensitivity to all five inputs +
#   F20 (u,v) negation pinned against tools/viz/common.grid_uv_natural +
#   build_pattern input validation + frozen-dataclass smoke + h01-only
#   n_filled fill-fraction band check.
# - tests/test_fast_vis_gridder.py (11): output shape/dtype + cell-weights
#   constancy + parity vs grid_uv_natural single + multi-channel + zero-vis
#   trivial + autos / outriggers excluded + linearity in vis + construction
#   validation. Pillbox K=1; K>1 deferred to chunk 10 hardening.
STEP="chunk_3a_gridder_sparsity_pattern"
python -m pytest tests/test_sparsity_pattern.py tests/test_fast_vis_gridder.py \
  -q --tb=short \
  || fail "gridder + sparsity-pattern acceptance pytests failed"
pass

# --- chunk 3c: RFI flagger -------------------------------------------------
# 18 acceptance tests covering: autos accumulator (S1 / S2 at M ∈ {64, 256,
# 1024, 4096}) + GEMM-layout vs complex-layout parity + SK threshold
# monotonicity + SK FAR on thermal noise (with per-M Gaussian-approx
# tolerance, see M3_PLAN_FIXES.md F23) + SK CW detection + bandpass-outlier
# narrowband CW + group-outlier dead-antenna + SumThreshold dilation /
# isolated / all-zero / + flagants.dat round-trip + invalid input rejects
# + RFIFlagger combine OR-logic on orthogonal scenarios + warmup state
# machine + one-shot flag_block + source-tag-bit disjointness.
STEP="chunk_3c_rfi_flagger"
python -m pytest tests/test_rfi_flagger.py -q --tb=short \
  || fail "RFI flagger acceptance pytests failed"
pass

# --- chunk 3d: voltage-domain online injector ------------------------------
# 19 acceptance tests covering: phasor-table unit-modulus + phase-formula
# (F22 sign convention) + cold-plasma dispersion delay table vs analytic +
# Gaussian / boxcar profile-vector normalisation + apply_block no-op /
# active / boundary / outside-window / purge-far-past + F22 visibility-phase
# pin + etcd JSON round-trip (round-trip / bad-payload reject) +
# MockEtcdWatcher routes inject + drops non-inject + handles malformed +
# DSART_ETCD_NAMESPACE_PREFIX honoured + GPU smoke (skipped on CPU-only
# runners).
STEP="chunk_3d_online_injector"
python -m pytest tests/test_online_injector.py -q --tb=short \
  || fail "online injector acceptance pytests failed"
pass

# ---------------------------------------------------------------------------
# Stage stamping
# ---------------------------------------------------------------------------

CHUNKS_DONE=(
  "chunk_1_cal_apply_with_F21_dec_phase"
  "chunk_2a_fast_corr_kernel"
  "chunk_2b_corr_fast_compute_service_shell"
  "chunk_3a_gridder_sparsity_pattern"
  "chunk_3c_rfi_flagger"
  "chunk_3d_online_injector"
)
CHUNKS_REMAINING=(   # update as chunks land; empty when M3 is complete
  "chunk_3b_coarse_dm_static_sky"
  "chunk_4_corr_fast_integration"
  "chunk_5_voltage_fixture_continuum"
  "chunk_6_voltage_fixture_burst_250924mptq"
  "chunk_7_16chgroup_alignment_preview"
  "chunk_8_transport_loopback_capture"
  "chunk_9_dod_orchestrator_completion"
  "chunk_10_hardening"
)

if [[ ${#CHUNKS_REMAINING[@]} -gt 0 ]]; then
  STAGE='in progress (substrate only)'
  if [[ ${#CHUNKS_DONE[@]} -gt 0 ]]; then
    DONE_LIST=$(IFS=,; echo "${CHUNKS_DONE[*]}")
    STAGE="in progress (chunks: ${DONE_LIST})"
  fi
else
  # All chunks landed → check operator-approval marker (mirrors M2.sh).
  # Filled in by chunk_10.
  STAGE='complete (needs operator approval)'
  if [[ -f "${M3_OPERATOR_APPROVAL_FILE}" ]]; then
    STAGE='complete (approved)'
    if [[ ! -f "${REPO_ROOT}/M3_PLAN_FIXES.md" ]]; then
      STAGE='complete (hardened)'
    fi
  fi
fi

GIT_SHA=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UTC_NOW=$(date -u +%FT%TZ)

cat > "${M3_STATUS_JSON}" <<JSON
{
  "milestone": "M3",
  "stage": "${STAGE}",
  "host": "$(hostname -s)",
  "phase": "a",
  "utc_iso": "${UTC_NOW}",
  "git_sha": "${GIT_SHA}",
  "git_branch": "${GIT_BRANCH}",
  "agent_isolation": {
    "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
    "buffer_key_prefix": "${DSART_BUFFER_KEY_PREFIX}",
    "etcd_namespace_prefix": "${DSART_ETCD_NAMESPACE_PREFIX}",
    "lockfile": "${M3_LOCKFILE}"
  },
  "chunks_done": [
    "chunk_1_cal_apply_with_F21_dec_phase",
    "chunk_2a_fast_corr_kernel",
    "chunk_2b_corr_fast_compute_service_shell",
    "chunk_3a_gridder_sparsity_pattern",
    "chunk_3c_rfi_flagger",
    "chunk_3d_online_injector"
  ],
  "chunks_remaining": [
    "chunk_3b_coarse_dm_static_sky",
    "chunk_4_corr_fast_integration",
    "chunk_5_voltage_fixture_continuum",
    "chunk_6_voltage_fixture_burst_250924mptq",
    "chunk_7_16chgroup_alignment_preview",
    "chunk_8_transport_loopback_capture",
    "chunk_9_dod_orchestrator_completion",
    "chunk_10_hardening"
  ]
}
JSON

echo ""
echo "== M3 DoD: stage=${STAGE} =="
echo "   wrote ${M3_STATUS_JSON}"
