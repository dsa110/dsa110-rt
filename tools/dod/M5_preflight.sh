#!/usr/bin/env bash
# M5 pre-flight readiness check (§8 line 2312-2341) — runs on h01.
#
# Purpose: fail fast if anything M5 depends on is broken before authoring.
# M5 builds the search-node detector pipeline (services/search_compute.py +
# fine_dm/combiner + image/imager + detector module + cube-injection
# detector unit-test injector + Layer-1/Layer-2 noise normalization +
# trigger emitter), so it depends on:
#   * M1 must be hardened (DmPlan + dataclass contracts + DM-plan npz)
#   * M4a is the production input source (search-RX → POSIX-shm receive
#     ring), but per PARALLEL_AGENTS.md §1 + the cube_injection bench
#     (plan §8 line 2329), M5 develops the detector independently of
#     M4a. Inform-only check on M4a here; not a hard gate.
#   * conda env dsa110-rt with torch + numpy + astropy
#   * GPU 1 visible (PARALLEL_AGENTS.md §4.2: M5 → CUDA_VISIBLE_DEVICES=1)
#   * /home/ubuntu/data/voltages/ root present (M5 voltage-fixture gate
#     consumes 250924mptq for the burst end-to-end test, but only after
#     M3 has emitted the captured transport-TX .npz set; informational)
#   * tools/viz/common.py present + read-only-importable (M3 owns; M5
#     consumes via tools/viz/search_helpers.py)
#   * configs/dm_plan.npz round-trips through DmPlan.from_npz()
#   * configs/config_compute_search.yaml schema-valid
#   * /var/lock/dsart-m5.lock writable (PARALLEL_AGENTS.md §4.4)
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
# pyproject.toml uses [tool.setuptools.packages.find] where=["src"], so the
# `dsart` package lives at ${REPO_ROOT}/src/dsart. PYTHONPATH must point at
# the `src` parent — pointing at ${REPO_ROOT} alone falls through to the
# conda env's editable install (which on h01 may resolve to the M3 working
# tree, parallel to ours, that doesn't carry M5 detector modules).
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# PARALLEL_AGENTS.md §4 isolation envelope (D3 in M5_PLAN_FIXES.md).
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m5}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M4A_STATUS_JSON="${M4A_STATUS_JSON:-${HOME}/dsart-m4a-status.json}"
VOLTAGES_ROOT="${VOLTAGES_ROOT:-/home/ubuntu/data/voltages}"
M5_LOCKFILE="${M5_LOCKFILE:-/var/lock/dsart-m5.lock}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M5_pre:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M5_pre:${STEP}] PASS"
}
warn() {
  echo "[M5_pre:${STEP}] WARN $*"
}

STEP="host_identity"
[[ "$(hostname -s)" == "lxd110h01" ]] || fail "expected lxd110h01, got $(hostname -s)"
pass

STEP="m1_status"
# M1 must be complete (hardened preferred, plain "complete" tolerated).
[[ -f "${M1_STATUS_JSON}" ]] || fail "missing ${M1_STATUS_JSON}"
python3 - <<PY || fail "M1 status JSON did not validate"
import json
s = json.load(open("${M1_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M1", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
assert s.get("phase", "") == "a", f"wrong phase: {s!r}"
print(f"M1 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m1_artifacts"
# M1 produced configs/dm_plan.npz; verify it's there and round-trips
# through the DmPlan dataclass (DSART_TEST=1 enables shape asserts).
[[ -f "${REPO_ROOT}/configs/dm_plan.npz" ]] || fail "missing configs/dm_plan.npz"
DSART_TEST=1 python3 - <<PY || fail "dm_plan.npz did not validate"
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from dsart.common.contracts import DmPlan
plan = DmPlan.from_npz("${REPO_ROOT}/configs/dm_plan.npz")
assert plan.fine_dm.shape[0] > 100
assert plan.coarse_dm.shape[0] > 8
print(f"DmPlan: {plan.fine_dm.shape[0]} fine, {plan.coarse_dm.shape[0]} coarse")
PY
pass

STEP="m4a_status_advisory"
# Per PARALLEL_AGENTS.md §1: M5 deps are {M1, M4a}, but the cube_injection
# detector bench (plan §8 line 2329) explicitly bypasses every upstream
# stage so M5 develops independently of M4a. Treat M4a as informational —
# the production search_compute service will fail without M4a's transport
# RX, but the cube-injection critical path does not. The voltage-fixture
# gate (plan §8 line 2330) needs M3 (not M4a) — covered separately at
# voltage_fixture_check below.
if [[ -f "${M4A_STATUS_JSON}" ]]; then
  python3 - <<PY || warn "M4a status JSON malformed; continuing (M5 cube-injection path is independent)"
import json
s = json.load(open("${M4A_STATUS_JSON}"))
print(f"M4a stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
else
  warn "no ${M4A_STATUS_JSON}; M5 search_compute production path will not run end-to-end yet, but cube-injection critical path is unaffected"
fi
pass

STEP="git_clean"
# m5/main worktree should be clean. We're on m5/main (not main); the
# branch check below verifies that.
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes in ${REPO_ROOT}; run 'git status'"
pass

STEP="git_branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "${BRANCH}" == "m5/main" ]] || fail "expected branch=m5/main, got ${BRANCH}"
git remote -v | grep -q '^origin' || fail "missing origin remote"
pass

STEP="conda_env"
[[ "${CONDA_DEFAULT_ENV:-}" == "dsa110-rt" ]] || fail "CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', expected dsa110-rt"
pass

STEP="dsart_import"
# Core M1 modules must import cleanly. M5 modules (detector, fine_dm,
# image, inject, noise_norm, trigger) currently only contain __init__.py
# stubs; importing them is cheap and catches any package-discovery bugs
# from the kickoff scaffolding.
python -c '
import dsart, dsart.common.host, dsart.common.config_loader
import dsart.common.contracts, dsart.common.constants, dsart.common.dispersion
import dsart.detector, dsart.fine_dm, dsart.image
import dsart.inject, dsart.trigger
' || fail "dsart package import"
pass

STEP="numerical_deps"
python - <<'PY' || fail "numerical deps"
import importlib
mods = ["numpy", "yaml", "pytest", "jsonschema", "torch", "astropy"]
for m in mods:
    importlib.import_module(m)
import numpy, torch
print(f"numpy={numpy.__version__} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
PY
pass

STEP="gpu_visible"
# PARALLEL_AGENTS.md §4.2: M5 → CUDA_VISIBLE_DEVICES=1. Verify the
# visible device is exactly 1 GPU and that it's the 2080 Ti (Turing,
# SM_75) M5 develops against.
python - <<'PY' || fail "GPU 1 not visible to torch"
import torch
n = torch.cuda.device_count()
assert n == 1, f"expected 1 visible GPU (CUDA_VISIBLE_DEVICES=1), got {n}"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name!r} sm_{cap[0]}{cap[1]}")
# 2080 Ti is sm_75; tolerate any sm_70+ for dev hosts that aren't h01.
assert cap[0] >= 7, f"GPU compute capability {cap}, expected sm_70+ (Turing or newer)"
PY
pass

STEP="config_compute_search_yaml"
# Schema-validate configs/config_compute_search.yaml (M0 stub) — verify
# the keys M5 chunks consume are present and well-typed.
python - <<PY || fail "configs/config_compute_search.yaml schema invalid"
import yaml
with open("${REPO_ROOT}/configs/config_compute_search.yaml") as f:
    cfg = yaml.safe_load(f)
required_top = {"schema_version", "dm_plan_path", "detector_class", "detector",
                "cube", "noise", "trigger", "decoder"}
missing = required_top - set(cfg.keys())
if missing:
    raise SystemExit(f"missing top-level keys: {sorted(missing)}")
det = cfg["detector"]
det_required = {"threshold_sigma", "k_img", "k_dm", "k_time", "kernel_dtype",
                "k_time_widths"}
det_missing = det_required - set(det.keys())
if det_missing:
    raise SystemExit(f"detector missing keys: {sorted(det_missing)}")
n_triples = det["k_img"] * det["k_dm"] * det["k_time"]
assert n_triples == 128, f"K_img*K_dm*K_time = {n_triples}, expected 128 (D2 in M5_PLAN_FIXES.md)"
assert det["kernel_dtype"] in ("fp16", "fp32"), f"kernel_dtype={det['kernel_dtype']}"
print(f"detector: K_img={det['k_img']} K_dm={det['k_dm']} K_time={det['k_time']} -> {n_triples} triples; threshold={det['threshold_sigma']}")
trig = cfg["trigger"]
trig_required = {"max_emit_per_s", "max_per_cube_per_image_kernel", "max_per_cube_total",
                 "max_dispatch_per_s", "dedup_specnum_step", "dedup_lm_step",
                 "dedup_ttl_ms", "holdoff_ms", "completion_timeout_s"}
trig_missing = trig_required - set(trig.keys())
if trig_missing:
    raise SystemExit(f"trigger missing keys: {sorted(trig_missing)}")
PY
pass

STEP="viz_common_present"
# tools/viz/common.py is M3-owned (M2 hardened). M5's tools/viz/search_helpers.py
# may import from it read-only. Verify it's importable.
[[ -f "${REPO_ROOT}/tools/viz/common.py" ]] || fail "missing tools/viz/common.py (M3-owned, M2-hardened)"
PYTHONPATH="${REPO_ROOT}/tools:${PYTHONPATH}" python -c 'import viz.common' \
  || fail "tools/viz/common.py not importable"
pass

STEP="voltages_root"
# /home/ubuntu/data/voltages/ must exist; the M5 voltage-fixture gate
# (plan §8 line 2330) consumes 250924mptq once M3 has emitted the
# captured transport-TX .npz set. Inform-only here — M5 cube-injection
# critical path does not depend on this. The fixture check is a soft
# advisory at the M5.sh level (D4 in M5_PLAN_FIXES.md).
if [[ ! -d "${VOLTAGES_ROOT}" ]]; then
  warn "${VOLTAGES_ROOT} not present; M5 voltage-fixture gate (plan §8 line 2330) will be skipped"
else
  burst_dir="${VOLTAGES_ROOT}/250924mptq"
  if [[ -d "${burst_dir}" ]]; then
    n_sb=$(find "${burst_dir}/voltages" -maxdepth 1 -name '*_sb*_data.out' 2>/dev/null | wc -l)
    echo "  ${burst_dir}: ${n_sb} sb*_data.out files"
    [[ -f "${burst_dir}/voltages/T2_250924mptq.json" ]] \
      && echo "  T2_250924mptq.json present (D4: burst manifest source)" \
      || warn "T2_250924mptq.json missing — manifest synthesis will fail"
  else
    warn "${burst_dir} missing — M5 voltage-fixture burst gate (D4) will be skipped"
  fi
fi
pass

STEP="lockfile_writable"
# PARALLEL_AGENTS.md §4.4: per-milestone lockfile. Verify the path is
# writable (or creatable). Don't actually take the lock here — M5.sh
# does that via flock.
LOCKDIR="$(dirname "${M5_LOCKFILE}")"
if [[ -d "${LOCKDIR}" ]] && [[ -w "${LOCKDIR}" ]]; then
  echo "  ${M5_LOCKFILE} dir writable"
elif [[ -f "${M5_LOCKFILE}" ]] && [[ -w "${M5_LOCKFILE}" ]]; then
  echo "  ${M5_LOCKFILE} writable (already exists)"
else
  warn "${M5_LOCKFILE} dir not writable; M5.sh flock will fail. Try: sudo install -d -m 1777 ${LOCKDIR}, or override with M5_LOCKFILE=\$HOME/.dsart-m5.lock"
fi
pass

STEP="m5_target_files"
# Inform-only: M5 deliverables that may already exist from re-runs / chunk
# landings. Class A files M5 owns per PARALLEL_AGENTS.md §3.
for relpath in \
  src/dsart/services/search_compute.py \
  src/dsart/fine_dm/combiner.py \
  src/dsart/image/imager.py \
  src/dsart/detector/forward.py \
  src/dsart/detector/decoder.py \
  src/dsart/detector/merger.py \
  src/dsart/detector/kernels.py \
  src/dsart/inject/cube_injection.py \
  src/dsart/noise_norm/__init__.py \
  src/dsart/trigger/emitter.py \
  bench/search_node_throughput.py \
  bench/noise_norm_calibration.py \
  bench/trigger_emitter_wiring.py \
  bench/cube_injection_detector.py \
  bench/voltage_fixture_search.py \
  tools/viz/search_detector_check.py \
  tools/viz/search_helpers.py \
  tools/dod/M5.sh \
  tools/dod/M5_preflight.sh
do
  if [[ -e "${REPO_ROOT}/${relpath}" ]]; then
    echo "  info: ${relpath} present"
  fi
done
pass

STEP="writable_paths"
[[ -w "${REPO_ROOT}/configs" ]] || fail "${REPO_ROOT}/configs not writable"
[[ -w "${REPO_ROOT}/tools" ]]   || fail "${REPO_ROOT}/tools not writable"
[[ -w "${REPO_ROOT}/src" ]]     || fail "${REPO_ROOT}/src not writable"
[[ -w "${REPO_ROOT}/tests" ]]   || fail "${REPO_ROOT}/tests not writable"
[[ -w "${REPO_ROOT}/bench" ]]   || fail "${REPO_ROOT}/bench not writable"
mkdir -p "${REPO_ROOT}/tools/viz" 2>/dev/null || true
[[ -d "${REPO_ROOT}/tools/viz" ]] || fail "${REPO_ROOT}/tools/viz not creatable"
mkdir -p "${REPO_ROOT}/bench/reports/M5" 2>/dev/null || true
[[ -d "${REPO_ROOT}/bench/reports/M5" ]] || fail "${REPO_ROOT}/bench/reports/M5 not creatable"
pass

echo "M5_preflight PASS"
