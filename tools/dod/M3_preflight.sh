#!/usr/bin/env bash
# M3 pre-flight readiness check (§8 line 2262 / §4.2) — runs on h01.
#
# Purpose: fail fast if anything M3 depends on is broken before authoring.
# M3 builds services/corr_fast_compute.py + RFI flagger + cal-apply +
# fast-corr GEMM + gridder + coarse-DM + static-sky + transport-TX, so it
# layers on top of M2 + M1 + M0 deliverables:
#
#   * M2 must be hardened (slow-corr GEMM helpers + voltage_layout_transform
#     + unpack_int4_split + apply_cal_split are reused as-is per §8 M3 line
#     2264; do NOT re-port).
#   * M1 must be hardened (DmPlan loader for the custom single-cell
#     dm_plan_burst_*.npz the burst sub-DoD needs).
#   * Per-agent isolation env vars MUST be set to the M3 defaults
#     (PARALLEL_AGENTS.md §4):
#       - CUDA_VISIBLE_DEVICES = 0     (matches numa_topology.yaml h01
#                                       dsart-corr-fast@01 cuda_device)
#       - DSART_BUFFER_KEY_PREFIX = m3 (so fada→fa3a, bada→ba3a, etc.)
#       - DSART_ETCD_NAMESPACE_PREFIX = m3 (so /cnf/dsart-m3/...)
#   * /var/lock/ writable for the per-milestone flock guard.
#   * /home/ubuntu/data/voltages/0319/        (continuum fixture; reused
#                                              for M3 continuum sub-DoD).
#   * /home/ubuntu/data/voltages/250924mptq/  (burst fixture; M3 burst
#                                              sub-DoD + 16-chgroup
#                                              alignment preview).
#
# §6 conda-activate shell pattern: 'set -u' DROPPED (conda MKL hook
# references MKL_INTERFACE_LAYER without default); 'pipefail' kept.
#
# Read-only: this script never writes to the repo or to /home/ubuntu/data/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# Per-milestone env defaults (PARALLEL_AGENTS.md §4). Caller may override
# for advanced testing, but the canonical h01 M3 run uses these.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m3}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m3}"

M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M2_STATUS_JSON="${M2_STATUS_JSON:-${HOME}/dsart-m2-status.json}"
VF_ROOT="${VF_ROOT:-/home/ubuntu/data/voltages}"
M3_LOCKFILE="${M3_LOCKFILE:-/var/lock/dsart-m3.lock}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M3_pre:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M3_pre:${STEP}] PASS"
}
warn() {
  echo "[M3_pre:${STEP}] WARN $*"
}

STEP="host_identity"
[[ "$(hostname -s)" == "lxd110h01" ]] || fail "expected lxd110h01, got $(hostname -s)"
pass

STEP="m1_status"
# M1 must be complete (hardened or approved tolerated).
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
# M2 must be hardened (preferred) or approved (tolerated). M3 depends on
# every M2 deliverable per §8 M3 line 2264 ("M3 inherits §8.M2-carryover
# above as binding context").
[[ -f "${M2_STATUS_JSON}" ]] || fail "missing ${M2_STATUS_JSON}"
python3 - <<PY || fail "M2 status JSON did not validate"
import json
s = json.load(open("${M2_STATUS_JSON}"))
stage = s.get("stage", "")
assert stage.startswith("complete"), f"M2 stage not complete: {s!r}"
assert s.get("milestone", "") == "M2", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
# 'complete (hardened)' is preferred; 'complete (approved)' tolerated.
if "hardened" not in stage and "approved" not in stage:
    print(f"WARN: M2 stage is {stage!r}; M3 prefers hardened/approved")
print(f"M2 stage: {stage!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m1_artifacts"
# M1 produced configs/dm_plan.npz; verify it round-trips.
[[ -f "${REPO_ROOT}/configs/dm_plan.npz" ]] || fail "missing configs/dm_plan.npz"
DSART_TEST=1 python3 - <<PY || fail "dm_plan.npz did not validate"
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from dsart.common.contracts import DmPlan
plan = DmPlan.from_npz("${REPO_ROOT}/configs/dm_plan.npz")
assert plan.fine_dm.shape[0] > 100
assert plan.coarse_dm.shape[0] > 8
PY
pass

STEP="m2_artifacts"
# M2-validated GPU helpers that M3 reuses verbatim per §8 M3 line 2264.
declare -a M2_REUSE_FILES=(
  "src/dsart/services/slow_corr_kernel.py"     # voltage_layout_transform, unpack_int4_split, apply_cal_split (D15/D16/F17)
  "src/dsart/services/corr_slow_compute.py"    # cal_loader integration pattern (F17/D17)
  "src/dsart/cal/bf_weights.py"                # legacy cal-blob loader (Class C, M3-owned)
  "tools/viz/common.py"                        # gridder + iFFT helpers with F20 (u,v) negation
  "bench/replay_voltage_dump.py"               # PSRDADA writer (F8/D7)
  "bench/casa38_meridian_wrapper.py"           # casa38 monkey-patch shim (D14)
)
for f in "${M2_REUSE_FILES[@]}"; do
  [[ -f "${REPO_ROOT}/${f}" ]] || fail "missing M2-reuse deliverable ${f}"
done
echo "  ${#M2_REUSE_FILES[@]} M2-reuse files present"
pass

STEP="git_clean"
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes in ${REPO_ROOT}; run 'git status'"
pass

STEP="git_branch"
# M3 work happens on m3/main (PARALLEL_AGENTS.md §2). Reject 'main' and any
# m5/* branch to prevent accidental cross-agent commits.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
case "${BRANCH}" in
  m3/main|m3/*) ;;  # OK
  main)
    fail "current branch=main; M3 work belongs on m3/main per PARALLEL_AGENTS.md §2"
    ;;
  m5/*)
    fail "current branch=${BRANCH}; that's M5 territory — switch to m3/main"
    ;;
  *)
    warn "current branch=${BRANCH} (expected m3/main or m3/*); continuing but verify"
    ;;
esac
git remote -v | grep -q '^origin' || fail "missing origin remote"
echo "  branch=${BRANCH}"
pass

STEP="agent_isolation_env"
# PARALLEL_AGENTS.md §4 conventions. Reject any unexpected setting that
# would collide with the M5 agent on the same host.
[[ "${CUDA_VISIBLE_DEVICES}" == "0" ]] \
  || fail "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, expected 0 (M3 owns GPU 0; M5 owns GPU 1)"
[[ "${DSART_BUFFER_KEY_PREFIX}" == "m3" ]] \
  || fail "DSART_BUFFER_KEY_PREFIX=${DSART_BUFFER_KEY_PREFIX}, expected m3"
[[ "${DSART_ETCD_NAMESPACE_PREFIX}" == "m3" ]] \
  || fail "DSART_ETCD_NAMESPACE_PREFIX=${DSART_ETCD_NAMESPACE_PREFIX}, expected m3"
echo "  CUDA=${CUDA_VISIBLE_DEVICES} bufprefix=${DSART_BUFFER_KEY_PREFIX} etcdprefix=${DSART_ETCD_NAMESPACE_PREFIX}"
pass

STEP="gpu_visibility"
# With CUDA_VISIBLE_DEVICES=0, torch should see exactly one GPU and it
# should be the corr-fast-pinned device (PCI 0000:3b:00.0 per
# numa_topology.yaml; CUDA :0 in BUS_ID order).
python - <<'PY' || fail "GPU 0 not visible to torch"
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import torch
assert torch.cuda.is_available(), "torch.cuda not available"
n = torch.cuda.device_count()
assert n == 1, f"with CUDA_VISIBLE_DEVICES=0, expected 1 visible GPU; got {n}"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  GPU 0: {name} (sm_{cap[0]}{cap[1]})")
PY
pass

STEP="lockfile_writable"
# /var/lock should be writable for our flock guard. If it's not (some
# distros restrict it), fall back to /tmp.
if [[ ! -d "$(dirname "${M3_LOCKFILE}")" ]]; then
  fail "lockfile parent $(dirname "${M3_LOCKFILE}") does not exist"
fi
if ! touch "${M3_LOCKFILE}.test" 2>/dev/null; then
  warn "${M3_LOCKFILE} parent not writable; M3.sh will fall back to /tmp/dsart-m3.lock"
else
  rm -f "${M3_LOCKFILE}.test"
fi
echo "  lockfile target: ${M3_LOCKFILE}"
pass

STEP="conda_env"
[[ "${CONDA_DEFAULT_ENV:-}" == "dsa110-rt" ]] || fail "CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', expected dsa110-rt"
pass

STEP="dsart_import"
python -c 'import dsart, dsart.common.host, dsart.common.config_loader, dsart.common.contracts, dsart.common.constants, dsart.common.dispersion' \
  || fail "dsart package import"
pass

STEP="m2_helpers_import"
# The fast-corr GEMM reuses M2's slow_corr_kernel helpers per §8 M3 line
# 2264. Import-check them so a partial M2 install fails fast. Note: the
# D15 voltage-layout transform (fp32-reinterpret 2D byte-transpose) is
# *inlined* inside unpack_int4_split (not a standalone export); see
# slow_corr_kernel.py lines 279-313 ("Stage 1: fp32-reinterpret 2D
# transpose") + the module docstring §(1).
python - <<'PY' || fail "M2 GPU helpers not importable"
from dsart.services.slow_corr_kernel import (
    unpack_int4_split,        # D15 + D16: fp32-reinterpret 2D byte-transpose + int8-ASR fluff
    apply_cal_split,          # F17/D17: per-(ant, ch, pol) cal-apply on the split fp16 tensors
    make_cal_broadcast_tensors,  # cal-tensor packing for apply_cal_split
    SlowCorrKernel,           # the fp16 chained-matmul GEMM driver (D8/F12)
    pack_bada_block,          # F8/D7: cfp32 bada packer for replay_voltage_dump
    upper_tri_indices,        # F18: PyTorch row-major upper-tri-gather index swap
)
print("  unpack_int4_split / apply_cal_split / make_cal_broadcast_tensors /")
print("  SlowCorrKernel / pack_bada_block / upper_tri_indices OK")
PY
pass

STEP="numerical_deps"
python - <<'PY' || fail "numerical deps"
import importlib
mods = ["numpy", "yaml", "pytest", "jsonschema", "torch", "astropy"]
for m in mods:
    importlib.import_module(m)
import numpy, torch
print(f"  numpy={numpy.__version__} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
PY
pass

STEP="psrdada_python"
python - <<'PY' || fail "psrdada-python not importable"
import psrdada
from psrdada import Reader, Writer
print(f"  psrdada.__file__={psrdada.__file__}")
PY
pass

STEP="voltage_fixture_root"
# Both fixtures must be present for the M3 user-facing sub-DoDs. Note:
# the on-disk file PREFIX may differ from the directory name (e.g. the
# 0319 dump dir contains 0319bbb_sb<NN>_data.out files; the run-id
# embedded in the filename is the legacy DSA-110 trigger ID, not always
# the parent dir name). Glob *_sb<NN>_data.out + T2_*.json without
# assuming the prefix matches the dir.
[[ -d "${VF_ROOT}" ]] || fail "${VF_ROOT} not present"
echo "  ${VF_ROOT}: $(find "${VF_ROOT}" -maxdepth 1 -mindepth 1 -type d | wc -l) run-id subdir(s)"
declare -A M3_FIXTURES_MIN_SB=(
  ["0319"]="15"           # continuum, M2 acceptance, reused for M3 continuum sub-DoD
                          # (§8 line 2282). KNOWN DATA GAP: missing sb12; M2 dispatched
                          # 15 of 16 SBs (sb00..sb11 + sb13..sb15).
  ["250924mptq"]="16"     # burst, DM≈404.7, Dec=53.85°; M3 burst sub-DoD
                          # (§8 line 2286) + 16-chgroup alignment preview (§8 line 2291)
                          # + M5 voltage-fixture gate (§8 line 2330).
)
for run_id in "${!M3_FIXTURES_MIN_SB[@]}"; do
  min_sb="${M3_FIXTURES_MIN_SB[${run_id}]}"
  d="${VF_ROOT}/${run_id}"
  [[ -d "${d}" ]] || fail "expected fixture ${d} not present"
  [[ -d "${d}/voltages" ]] || fail "${d}/voltages/ missing"
  [[ -d "${d}/cals" ]] || fail "${d}/cals/ missing"
  n_sb=$(find "${d}/voltages" -maxdepth 1 -name '*_sb*_data.out' -type f | wc -l)
  [[ "${n_sb}" -ge "${min_sb}" ]] || fail "${d}: expected >= ${min_sb} _sb<NN>_data.out files, got ${n_sb}"
  n_t2=$(find "${d}/voltages" -maxdepth 1 -name 'T2_*.json' -type f -not -name '*~' | wc -l)
  [[ "${n_t2}" -ge 1 ]] || fail "${d}: missing T2_*.json"
  echo "  ${run_id}: ${n_sb} SB voltage dumps + ${n_t2} T2_*.json (min_sb=${min_sb})"
done
pass

STEP="m3_target_files"
# Inform-only: M3 deliverables that may already exist from re-runs.
for relpath in \
  src/dsart/services/corr_fast_compute.py \
  src/dsart/rfi/__init__.py \
  src/dsart/cal/antennas_out.py \
  src/dsart/coarse_dm/dedisp.py \
  src/dsart/grid/kernel.py \
  src/dsart/grid/sparsity_pattern.py \
  src/dsart/inject/online.py \
  bench/voltage_fixture_fast_corr_continuum.py \
  bench/voltage_fixture_fast_corr_burst.py \
  bench/fast_path_throughput.py \
  bench/fast_isolation.py \
  bench/static_sky_subtract.py \
  bench/rfi_calibration.py \
  bench/rfi_warmup.py \
  bench/cal_reload.py \
  tests/test_corr_fast_synth.py \
  M3_PLAN_FIXES.md \
  tools/dod/M3.sh
do
  if [[ -e "${REPO_ROOT}/${relpath}" ]]; then
    echo "  info: ${relpath} already present (re-run / partial M3 — OK)"
  fi
done
pass

STEP="parallel_agents_doc"
# PARALLEL_AGENTS.md must be on disk; M3 + M5 both bind to its conventions.
[[ -f "${REPO_ROOT}/PARALLEL_AGENTS.md" ]] || fail "missing PARALLEL_AGENTS.md"
pass

STEP="writable_paths"
[[ -w "${REPO_ROOT}/configs" ]]     || fail "${REPO_ROOT}/configs not writable"
[[ -w "${REPO_ROOT}/tools" ]]       || fail "${REPO_ROOT}/tools not writable"
[[ -w "${REPO_ROOT}/src" ]]         || fail "${REPO_ROOT}/src not writable"
[[ -w "${REPO_ROOT}/tests" ]]       || fail "${REPO_ROOT}/tests not writable"
[[ -w "${REPO_ROOT}/bench" ]]       || fail "${REPO_ROOT}/bench not writable"
mkdir -p "${REPO_ROOT}/src/dsart/rfi"     2>/dev/null || true
mkdir -p "${REPO_ROOT}/src/dsart/cal"     2>/dev/null || true
mkdir -p "${REPO_ROOT}/src/dsart/coarse_dm" 2>/dev/null || true
mkdir -p "${REPO_ROOT}/src/dsart/grid"    2>/dev/null || true
mkdir -p "${REPO_ROOT}/src/dsart/inject"  2>/dev/null || true
pass

echo "M3_preflight PASS"
