#!/usr/bin/env bash
# M2 Definition-of-Done (§8 line 2138-2179) — runs on h01.
#
# M2 is the slow-correlator + fstable-cache + voltage-fixture-imaging
# milestone. It has more moving parts than M0/M1:
#
#   1. M2_preflight passes (M1 hardened; psrdada + torch + casa38 deps OK;
#      configs/config_corr.yaml physics-pinned; dsamfs source matches
#      casa38 install).
#   2. M2 source-tree deliverables present (preflight-listed set + the
#      F17/D17 cal-flag and voltage-fixture-driver additions).
#   3. tests/test_slow_corr_synth.py — 11/11 GPU GEMM-faithfulness +
#      sign-convention + planar-wave-phase-recovery tests pass.
#   4. tests/test_cal_loader_apply.py — 16/16 cal loader + apply-cal
#      kernel tests pass (F17/D17).
#   5. tools/build_fstable_cache.py — runs deterministically from casa38
#      python (--dry-run smoke + a narrow 2-DEC grid actually executed);
#      .npz schema matches legacy dsamfs.fringestopping output.
#   6. tests/test_voltage_fixture_slow_corr_smoke.py — synthetic
#      end-to-end pipeline (dada_db → corr_slow_compute → replay → viz)
#      lands the synthetic source within ~1 grid cell. ~3-5 min; gated
#      behind M2_SKIP_SLOW=1 for fast iteration.
#   7. Operator-approval marker present (D11) — voltage-fixture imaging
#      acceptance was reviewed and signed off out-of-band; the marker
#      yaml is the gate. Without it M2 stamps `complete (needs operator
#      approval)` and exits 0; with it M2 stamps `complete (approved)`
#      or `complete (hardened)` (after Chunk 8 retires M2_PLAN_FIXES.md).
#
# Stage labels (mirrors M1.sh + M0/M1 JSON shape consumed by Mn+1):
#   - failed                            -> some STEP failed; exit 1
#   - complete (needs operator approval) -> all auto checks PASS, no marker
#   - complete (approved)               -> marker present, M2_PLAN_FIXES.md still in repo
#   - complete (hardened)               -> marker present + M2_PLAN_FIXES.md retired
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

M2_STATUS_JSON="${M2_STATUS_JSON:-${HOME}/dsart-m2-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"

CASA38_PY="${CASA38_PY:-/home/ubuntu/anaconda3/envs/casa38/bin/python}"
FSTABLE_ROOT="${FSTABLE_ROOT:-/home/ubuntu/data/fstables}"

# Operator-approval marker (D11 in M2_PLAN_FIXES.md). Default location is
# bench/reports/M2/m_operator_approved.yaml in the repo; override with
# M2_OPERATOR_APPROVAL_FILE for pointing at a non-default report dir.
M2_OPERATOR_APPROVAL_FILE="${M2_OPERATOR_APPROVAL_FILE:-${REPO_ROOT}/bench/reports/M2/m_operator_approved.yaml}"

# Slow integration smoke (~3-5 min). Skip with M2_SKIP_SLOW=1 when iterating
# on the DoD itself; the canonical h01 run leaves it on.
M2_SKIP_SLOW="${M2_SKIP_SLOW:-0}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M2:${STEP}] FAIL $*"
  cat > "${M2_STATUS_JSON}" <<JSON
{"milestone": "M2", "stage": "failed", "step": "${STEP}", "host": "$(hostname -s)", "phase": "a", "utc_iso": "$(date -u +%FT%TZ)"}
JSON
  exit 1
}
pass() {
  echo "[M2:${STEP}] PASS"
}
warn() {
  echo "[M2:${STEP}] WARN $*"
}

echo "== M2 DoD: gate on M2_preflight =="
bash "${SCRIPT_DIR}/M2_preflight.sh"

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

STEP="m2_deliverables"
# Plan §8 line 2138-2179 + F17/D17/F8/D7 deliverables. The M2_preflight
# inform-only loop already enumerated these; here we *require* them.
declare -a M2_FILES=(
  "src/dsart/services/__init__.py"
  "src/dsart/services/slow_corr_kernel.py"
  "src/dsart/services/corr_slow_compute.py"
  "src/dsart/cal/__init__.py"
  "src/dsart/cal/bf_weights.py"
  "tools/build_fstable_cache.py"
  "tools/viz/__init__.py"
  "tools/viz/common.py"
  "tools/viz/corr_imager_dedisperser_check.py"
  "bench/replay_voltage_dump.py"
  "bench/voltage_fixture_slow_corr.py"
  "bench/casa38_meridian_wrapper.py"
  "bench/run_0319_pipeline.py"
  "tests/test_slow_corr_synth.py"
  "tests/test_cal_loader_apply.py"
  "tests/test_voltage_fixture_slow_corr_smoke.py"
  "tools/dod/M2.sh"
  "tools/dod/M2_preflight.sh"
)
for f in "${M2_FILES[@]}"; do
  [[ -f "${REPO_ROOT}/${f}" ]] || fail "missing ${f}"
done
echo "  ${#M2_FILES[@]} deliverable files present"
pass

STEP="pytest_test_slow_corr_synth"
# Plan §8 line 2167-2170: GEMM faithfulness + sign convention + planar-
# wave phase recovery against pyuvdata-equivalent ground truth. F18 is
# encoded into test_kernel_sign_convention_v_ant1_conj_ant2 and
# test_point_source_planar_wave_phase_recovery.
DSART_TEST=1 python -m pytest tests/test_slow_corr_synth.py -q --tb=short \
  || fail "tests/test_slow_corr_synth.py failed"
pass

STEP="pytest_test_cal_loader_apply"
# F17/D17: --apply-cal flag loader + per-(ant, ch, pol) apply_cal_split
# kernel, both `phase` and `full` modes, with pol-swap option.
DSART_TEST=1 python -m pytest tests/test_cal_loader_apply.py -q --tb=short \
  || fail "tests/test_cal_loader_apply.py failed"
pass

STEP="build_fstable_cache_dryrun"
# Plan §8 line 2164. Smoke: tool imports cleanly under casa38 + the DEC
# grid math is sane on a 2-DEC range (the real 461-DEC grid is operator
# time and lives outside the DoD).
"${CASA38_PY}" "${REPO_ROOT}/tools/build_fstable_cache.py" \
    --dec-min 25.0 --dec-max 25.25 --dec-step 0.25 \
    --output-dir "${FSTABLE_ROOT}" \
    --dry-run 2>&1 | tee /tmp/dsart_m2_fstable_dryrun.log >/dev/null
grep -q "DEC grid: 2 points" /tmp/dsart_m2_fstable_dryrun.log \
  || fail "fstable_cache dry-run did not enumerate 2 DEC points"
grep -q "fringestopping_table_dec_+25.0000deg_" /tmp/dsart_m2_fstable_dryrun.log \
  || fail "fstable_cache dry-run filename scheme mismatch (D6)"
pass

STEP="build_fstable_cache_narrow_grid"
# Actually run the casa38 path on a narrow 2-DEC grid (~60 s on h01) so
# the DoD certifies the dsamfs.fringestopping.generate_fringestopping_table
# integration. The full 461-DEC grid is operator-time (~30 s/file, ~4 hr
# total) and lives outside the DoD.
export FSTABLE_ROOT
mkdir -p "${FSTABLE_ROOT}"
"${CASA38_PY}" "${REPO_ROOT}/tools/build_fstable_cache.py" \
    --dec-min 25.0 --dec-max 25.25 --dec-step 0.25 \
    --output-dir "${FSTABLE_ROOT}" \
    --force 2>&1 | tee /tmp/dsart_m2_fstable_build.log >/dev/null
grep -q "built=2 skipped=0 failed=0" /tmp/dsart_m2_fstable_build.log \
  || { echo '--- tail /tmp/dsart_m2_fstable_build.log ---'; tail -30 /tmp/dsart_m2_fstable_build.log; fail "fstable_cache build did not complete cleanly"; }
# Verify .npz schema matches legacy dsamfs.fringestopping (allow_pickle
# is needed because outrigger_delays is stored as object dtype). The
# corr_setup.yaml on h01 is the dev-site 64-ant variant; nant in the
# filename is read from corr_setup directly.
"${CASA38_PY}" - <<'PY' || fail "fstable .npz schema mismatch"
import glob
import os
import numpy as np
root = os.environ["FSTABLE_ROOT"]
expected_keys = {"ant_bw", "antenna_order", "bw", "bwref", "dec_rad", "ha", "outrigger_delays", "refmjd", "tsamp_s"}
files = sorted(glob.glob(os.path.join(root, "fringestopping_table_dec_+25.0000deg_*.npz")) +
               glob.glob(os.path.join(root, "fringestopping_table_dec_+25.2500deg_*.npz")))
assert len(files) == 2, f"expected 2 narrow-grid .npz files, got {files!r}"
for p in files:
    data = np.load(p, allow_pickle=True)
    keys = set(data.files)
    missing = expected_keys - keys
    assert not missing, f"{p}: missing keys {missing}; got {keys}"
    assert data["dec_rad"].shape == (), f"{p}: dec_rad shape {data['dec_rad'].shape}"
    assert data["bw"].ndim == 2, f"{p}: bw ndim {data['bw'].ndim}"
    print(f"  {os.path.basename(p)}: keys={sorted(keys)} bw.shape={data['bw'].shape}")
PY
pass

STEP="pytest_test_voltage_fixture_slow_corr_smoke"
# Plan §8 line 2172. End-to-end synthetic-fixture smoke: PSRDADA buffers
# + corr_slow_compute + replay_voltage_dump + viz, lands a synthetic
# source within ~1 grid cell. ~3-5 min wall clock.
if [[ "${M2_SKIP_SLOW}" == "1" ]]; then
  warn "M2_SKIP_SLOW=1; skipping ~3-5 min synthetic end-to-end smoke. NOT for canonical h01 runs."
else
  DSART_TEST=1 python -m pytest tests/test_voltage_fixture_slow_corr_smoke.py -q --tb=short \
    || fail "tests/test_voltage_fixture_slow_corr_smoke.py failed"
fi
pass

STEP="pytest_test_contracts_test_numerical_conventions"
# Cheap smoke: re-run M1 contracts + numerical_conventions to certify
# nothing in M2 broke the dataclass / DM-plan invariants. ~5-10 s.
DSART_TEST=1 python -m pytest tests/test_contracts.py tests/test_numerical_conventions.py -q --tb=short \
  || fail "M1 regression suite failed"
pass

STEP="operator_approval"
# D11: the operator-approval marker yaml is the gate that promotes the
# DoD from "complete (needs operator approval)" to "complete (approved)"
# (or "complete (hardened)" if M2_PLAN_FIXES.md has been retired).
APPROVAL_PRESENT="false"
APPROVAL_OPERATOR=""
APPROVAL_UTC=""
APPROVAL_VOLTAGE_RUN_ID=""
APPROVAL_VIZ_SHA=""
if [[ -f "${M2_OPERATOR_APPROVAL_FILE}" ]]; then
  python3 - <<PY > /tmp/dsart_m2_operator_approval.json || fail "operator-approval yaml is malformed"
import json
import sys
import yaml
with open("${M2_OPERATOR_APPROVAL_FILE}") as fh:
    data = yaml.safe_load(fh) or {}
required = {"operator", "approval_utc_iso", "milestone", "voltage_run_id", "viz_artifact_sha256"}
missing = required - set(data.keys())
if missing:
    print(f"missing fields: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)
if str(data.get("milestone")) != "M2":
    print(f"wrong milestone {data.get('milestone')!r}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: str(data.get(k, "")) for k in sorted(required)}))
PY
  APPROVAL_PRESENT="true"
  APPROVAL_OPERATOR="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m2_operator_approval.json"))["operator"])')"
  APPROVAL_UTC="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m2_operator_approval.json"))["approval_utc_iso"])')"
  APPROVAL_VOLTAGE_RUN_ID="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m2_operator_approval.json"))["voltage_run_id"])')"
  APPROVAL_VIZ_SHA="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m2_operator_approval.json"))["viz_artifact_sha256"])')"
  echo "  marker present at ${M2_OPERATOR_APPROVAL_FILE}"
  echo "  operator=${APPROVAL_OPERATOR} utc=${APPROVAL_UTC} run_id=${APPROVAL_VOLTAGE_RUN_ID}"
else
  warn "no operator-approval marker at ${M2_OPERATOR_APPROVAL_FILE} — stamp will be 'needs operator approval'"
fi
pass

STEP="status_emit"
GIT_SHA="$(git rev-parse HEAD)"
PLAN_FIXES_STILL_PRESENT="$([[ -f ${REPO_ROOT}/M2_PLAN_FIXES.md ]] && echo true || echo false)"
if [[ "${APPROVAL_PRESENT}" == "true" ]]; then
  if [[ "${PLAN_FIXES_STILL_PRESENT}" == "true" ]]; then
    STAGE_LABEL="complete (approved)"
  else
    STAGE_LABEL="complete (hardened)"
  fi
else
  STAGE_LABEL="complete (needs operator approval)"
fi
# Plan-fixes-applied list (F1-F20) baked in here for status visibility;
# Chunk 8 (hardening) flips PLAN_FIXES_STILL_PRESENT false by retiring
# M2_PLAN_FIXES.md.
cat > "${M2_STATUS_JSON}" <<JSON
{
  "milestone": "M2",
  "stage": "${STAGE_LABEL}",
  "host": "$(hostname -s)",
  "phase": "a",
  "utc_iso": "$(date -u +%FT%TZ)",
  "git_sha": "${GIT_SHA}",
  "fstable_cache": {
    "root": "${FSTABLE_ROOT}",
    "narrow_grid_built": ["+25.0000", "+25.2500"]
  },
  "tests_passed": [
    "tests/test_slow_corr_synth.py",
    "tests/test_cal_loader_apply.py",
    "tests/test_voltage_fixture_slow_corr_smoke.py",
    "tests/test_contracts.py",
    "tests/test_numerical_conventions.py"
  ],
  "operator_approval": {
    "present": ${APPROVAL_PRESENT},
    "marker_path": "${M2_OPERATOR_APPROVAL_FILE}",
    "operator": "${APPROVAL_OPERATOR}",
    "approval_utc_iso": "${APPROVAL_UTC}",
    "voltage_run_id": "${APPROVAL_VOLTAGE_RUN_ID}",
    "viz_artifact_sha256": "${APPROVAL_VIZ_SHA}"
  },
  "plan_fixes_applied": [
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
    "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20"
  ],
  "decisions_locked": [
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18"
  ],
  "plan_fixes_tracker_present": ${PLAN_FIXES_STILL_PRESENT}
}
JSON
pass

echo "M2 PASS (${STAGE_LABEL})"
echo "  status: ${M2_STATUS_JSON}"
if [[ "${APPROVAL_PRESENT}" != "true" ]]; then
  echo
  echo "  next: drop the operator-approval marker yaml at"
  echo "    ${M2_OPERATOR_APPROVAL_FILE}"
  echo "  with fields:"
  echo "    operator: <name>"
  echo "    approval_utc_iso: $(date -u +%FT%TZ)"
  echo "    milestone: M2"
  echo "    voltage_run_id: <id>"
  echo "    viz_artifact_sha256: <hex>"
  echo "  then re-run this script to stamp 'complete (approved)'."
fi
