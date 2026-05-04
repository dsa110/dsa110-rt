#!/usr/bin/env bash
# M1 pre-flight readiness check (§8 / §3.2) — runs on h01.
#
# Purpose: fail fast if anything M1 depends on is broken before authoring +
# DoD scripts run. Modeled on M0 hardening lessons:
#   * 'set -u' DROPPED — conda's MKL activate.d hook references
#     MKL_INTERFACE_LAYER without a default; under -u that aborts conda
#     activate before any STEP runs (§6 conda-activate shell pattern).
#   * MKL_INTERFACE_LAYER exported defensively.
#   * Each STEP is a one-line PASS/FAIL gate, identical lexical pattern
#     to M0.sh (so logs are diff-friendly across milestones).
#
# Read-only: this script never writes to the repo or to /home/ubuntu/data/.
# It is safe to re-run any time as a "is the dev env healthy?" smoke.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M1_pre:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M1_pre:${STEP}] PASS"
}

STEP="host_identity"
[[ "$(hostname -s)" == "lxd110h01" ]] || fail "expected lxd110h01, got $(hostname -s)"
pass

STEP="m0_status"
# M0 must be complete before M1 starts. Tolerate descriptive suffixes on
# 'stage' (e.g. 'complete (hardened)') the same way M0_prereq.sh does.
[[ -f "${M0_STATUS_JSON}" ]] || fail "missing ${M0_STATUS_JSON}"
python3 - <<PY || fail "M0 status JSON did not validate"
import json, sys
s = json.load(open("${M0_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M0",            f"wrong milestone: {s!r}"
assert s.get("host", "")      == "lxd110h01",     f"wrong host: {s!r}"
assert s.get("phase", "")     == "a",             f"wrong phase: {s!r}"
PY
pass

STEP="git_clean"
# An unclean tree at preflight means the operator forgot to commit/push
# something on the Mac/h23 side, or h01 has local edits — either way M1
# authoring should not start until resolved (mirrors M0 hardening rule).
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes in ${REPO_ROOT}; run 'git status'"
pass

STEP="git_branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "${BRANCH}" == "main" ]] || fail "expected branch=main, got ${BRANCH}"
git remote -v | grep -q '^origin' || fail "missing origin remote"
pass

STEP="conda_env"
# Confirms 'conda activate dsa110-rt' actually landed in the right env
# (paranoia: a stray CONDA_DEFAULT_ENV in ~/.bashrc could redirect us).
[[ "${CONDA_DEFAULT_ENV:-}" == "dsa110-rt" ]] || fail "CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', expected dsa110-rt"
pass

STEP="dsart_import"
python -c 'import dsart, dsart.common.host, dsart.common.config_loader' \
  || fail "dsart package import"
pass

STEP="numerical_deps"
python - <<'PY' || fail "numerical deps"
import numpy, yaml, pytest, jsonschema
print(f"numpy={numpy.__version__} yaml={yaml.__version__} "
      f"pytest={pytest.__version__} jsonschema={jsonschema.__version__}")
PY
pass

STEP="operating_points"
# M1 build_dm_plan.py reads operating_points.yaml::default to pick
# t_int_search_us. Verify the pointer + the keys it'll need are present.
python - <<PY || fail "operating_points.yaml not M1-ready"
from pathlib import Path
from dsart.common import config_loader
op = config_loader.load(Path("${REPO_ROOT}/configs/operating_points.yaml"))
default_name = op.get("default")
assert default_name, "operating_points.yaml: missing 'default' key"
rows = op.get("rows", {})
assert default_name in rows, f"default {default_name!r} not in rows"
row = rows[default_name]
for k in ("t_int_search_us", "t_int_factor", "N_grid", "kernel_support"):
    assert k in row, f"default row {default_name!r} missing key {k!r}"
print(f"default ops: {default_name} t_int_search_us={row['t_int_search_us']} "
      f"t_int_factor={row['t_int_factor']}")
PY
pass

STEP="corr_setup_96"
# constants.py at M1 reads corr_setup_96.yaml::antenna_order to size
# antenna-related constants; build_dm_plan.py does NOT need it (DM plan is
# host-agnostic), but having it loadable is a generic dev-env health gate.
python - <<PY || fail "corr_setup_96.yaml not M1-ready"
from pathlib import Path
from dsart.common import config_loader
cs = config_loader.load(Path("${REPO_ROOT}/configs/corr_setup_96.yaml"))
ao = cs.get("antenna_order")
assert ao, "corr_setup_96.yaml: missing antenna_order"
assert len(ao) == 96, f"antenna_order has {len(ao)} entries, expected 96"
PY
pass

STEP="m1_target_files"
# Inform-only sweep: M1 deliverables that are expected to NOT exist yet
# (or to be a no-op stub from M0 scaffolding). Re-running preflight after
# M1 chunks land will surface these as already-present — that's fine.
for relpath in \
  src/dsart/common/contracts.py \
  src/dsart/common/constants.py \
  tools/build_dm_plan.py \
  tests/test_contracts.py \
  tests/test_numerical_conventions.py \
  configs/dm_plan.npz \
  tools/dod/M1.sh
do
  if [[ -e "${REPO_ROOT}/${relpath}" ]]; then
    echo "  info: ${relpath} already present (re-run / partial M1 — OK)"
  fi
done
pass

STEP="writable_paths"
# Anything M1 writes must be writable by the running user. configs/ is the
# only target (committed dm_plan.npz). /home/ubuntu/data/ is M0-owned and
# not used by M1.
[[ -w "${REPO_ROOT}/configs" ]] || fail "${REPO_ROOT}/configs not writable"
[[ -w "${REPO_ROOT}/tools" ]]   || fail "${REPO_ROOT}/tools not writable"
pass

echo "M1_preflight PASS"
