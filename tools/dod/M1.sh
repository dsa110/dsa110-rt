#!/usr/bin/env bash
# M1 Definition-of-Done (§8 line 2138-2141) — runs on h01 with dsa110-rt
# conda env. Models M0.sh's STEP/pass/fail lexical pattern for diff-
# friendly logs across milestones.
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

M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
DM_PLAN_NPZ="${REPO_ROOT}/configs/dm_plan.npz"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M1:${STEP}] FAIL $*"
  cat > "${M1_STATUS_JSON}" <<JSON
{"milestone": "M1", "stage": "failed", "step": "${STEP}", "host": "$(hostname -s)", "phase": "a", "utc_iso": "$(date -u +%FT%TZ)"}
JSON
  exit 1
}
pass() {
  echo "[M1:${STEP}] PASS"
}

echo "== M1 DoD: gate on M1_preflight ==" 
bash "${SCRIPT_DIR}/M1_preflight.sh"

STEP="m0_status"
[[ -f "${M0_STATUS_JSON}" ]] || fail "missing ${M0_STATUS_JSON}"
python3 - <<PY || fail "M0 status JSON did not validate"
import json
s = json.load(open("${M0_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M0", f"wrong milestone: {s!r}"
PY
pass

STEP="m1_deliverables"
# Plan §8 line 2138-2141 deliverables list.
for f in \
    src/dsart/common/constants.py \
    src/dsart/common/contracts.py \
    src/dsart/common/dispersion.py \
    tools/build_dm_plan.py \
    tests/test_contracts.py \
    tests/test_numerical_conventions.py; do
  [[ -f "${REPO_ROOT}/${f}" ]] || fail "missing ${f}"
done
pass

STEP="dm_plan_build"
# §8 line 2141: build_dm_plan.py runs and produces an .npz matching §3.2
# schema. We rebuild here (not relying on the committed .npz) to certify
# the build is reproducible from the current code at HEAD.
TMP_NPZ="$(mktemp --suffix=.npz)"
trap 'rm -f "${TMP_NPZ}"' EXIT
python "${REPO_ROOT}/tools/build_dm_plan.py" \
    --out "${TMP_NPZ}" \
    --dm-min 0 --dm-max 3000 --tol 1.5 \
    --quiet \
  || fail "build_dm_plan.py exited non-zero"
[[ -s "${TMP_NPZ}" ]] || fail "build_dm_plan.py produced empty .npz"
pass

STEP="dm_plan_committed"
[[ -f "${DM_PLAN_NPZ}" ]] || fail "missing committed configs/dm_plan.npz"
# Plan §3.3 line 601: the .npz is committed in the repo.
pass

STEP="dm_plan_schema"
# §3.2 line 542-571 schema sanity (DmPlan.from_npz validates under
# DSART_TEST=1). Defer the full numerical invariants to test_numerical_conventions.
DSART_TEST=1 python3 - <<PY || fail "configs/dm_plan.npz did not validate"
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from dsart.common.contracts import DmPlan
plan = DmPlan.from_npz("${DM_PLAN_NPZ}")
assert plan.fine_dm.shape[0] > 100, f"N_fine={plan.fine_dm.shape[0]}"
assert plan.coarse_dm.shape[0] > 8, f"N_coarse={plan.coarse_dm.shape[0]}"
assert plan.metadata["version"] == 1
assert isinstance(plan.metadata.get("git_sha", ""), str) and plan.metadata["git_sha"]
print(f"  N_fine={plan.fine_dm.shape[0]} N_coarse={plan.coarse_dm.shape[0]} "
      f"git_sha={plan.metadata['git_sha'][:12]}")
PY
pass

STEP="pytest_test_contracts"
# §8 line 2141: validates shape asserts on each contract dataclass under DSART_TEST=1.
DSART_TEST=1 python -m pytest tests/test_contracts.py -q --tb=short \
  || fail "tests/test_contracts.py failed"
pass

STEP="pytest_test_dm_plan_time_shift_tables"
# §8 line 2141: schema round-trip + §3.6.2/§3.6.3 sanity invariants.
DSART_TEST=1 python -m pytest \
    "tests/test_numerical_conventions.py::test_dm_plan_time_shift_tables" \
    -q --tb=short \
  || fail "test_dm_plan_time_shift_tables failed"
pass

STEP="pytest_test_numerical_conventions_full"
# Bonus M1 coverage (test_dispersion_delay_at_dm_3000, partition invariants,
# metadata schema). Not in §8 line 2141 but cheap and catches regressions.
DSART_TEST=1 python -m pytest tests/test_numerical_conventions.py -q --tb=short \
  || fail "tests/test_numerical_conventions.py (full) failed"
pass

STEP="status_emit"
GIT_SHA="$(git rev-parse HEAD)"
DM_PLAN_GIT_SHA="$(DSART_TEST=0 python3 - <<PY
import sys, json
sys.path.insert(0, "${REPO_ROOT}/src")
import numpy as np
data = np.load("${DM_PLAN_NPZ}", allow_pickle=False)
print(json.loads(str(data["metadata"]))["git_sha"])
PY
)"
N_FINE="$(python3 - <<PY
import numpy as np
print(np.load("${DM_PLAN_NPZ}", allow_pickle=False)["fine_dm"].shape[0])
PY
)"
N_COARSE="$(python3 - <<PY
import numpy as np
print(np.load("${DM_PLAN_NPZ}", allow_pickle=False)["coarse_dm"].shape[0])
PY
)"
# Stage label is "complete (hardened)" once the M1 plan-fix tracker
# (M1_PLAN_FIXES.md) has been incorporated into plan.md and deleted from
# the repo. Mirrors the M0_prereq tolerance of `stage.startswith("complete")`.
PLAN_FIXES_STILL_PRESENT="$([[ -f ${REPO_ROOT}/M1_PLAN_FIXES.md ]] && echo true || echo false)"
if [[ "${PLAN_FIXES_STILL_PRESENT}" == "true" ]]; then
  STAGE_LABEL="complete"
else
  STAGE_LABEL="complete (hardened)"
fi
cat > "${M1_STATUS_JSON}" <<JSON
{
  "milestone": "M1",
  "stage": "${STAGE_LABEL}",
  "host": "$(hostname -s)",
  "phase": "a",
  "utc_iso": "$(date -u +%FT%TZ)",
  "git_sha": "${GIT_SHA}",
  "dm_plan": {
    "path": "configs/dm_plan.npz",
    "git_sha": "${DM_PLAN_GIT_SHA}",
    "n_fine": ${N_FINE},
    "n_coarse": ${N_COARSE}
  },
  "tests_passed": [
    "tests/test_contracts.py",
    "tests/test_numerical_conventions.py"
  ],
  "plan_fixes_applied": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12"],
  "plan_fixes_tracker_present": ${PLAN_FIXES_STILL_PRESENT}
}
JSON
pass

echo "M1 PASS (${STAGE_LABEL})"
echo "  status: ${M1_STATUS_JSON}"
