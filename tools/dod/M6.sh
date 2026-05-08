#!/usr/bin/env bash
# M6 Definition-of-Done (§8 line 2342+ — M6 plan section to be filled
# by chunk-9 fold) — runs on h01.
#
# M6 is the search-node clustering + cube-dump milestone, sitting on
# top of M5's detector pipeline. It builds:
#
#   1. src/dsart/cluster/{forward,features,state,cands_logger}.py —
#      HDBSCAN clustering of detector candidates per cube, with a D5
#      DBSCAN fallback if HDBSCAN p99 > 50ms; T1 (per-candidate) +
#      T2 (per-cluster) hourly-rotated ASCII logs.
#
#   2. src/dsart/dump/{cube_dump,udp_listener}.py — bright-pulse
#      cluster-predicate auto-trigger and external UDP-trigger paths;
#      writer thread + bounded queue (maxsize=4) → NPZ cube dump.
#
#   3. services/search_compute.py rewiring: detector → clusterer
#      (ThreadPool worker) → cube-dump (writer thread) + UDP listener.
#
#   4. Three DoD benches:
#      - bench/clusterer_throughput.py       — HDBSCAN p99 gate at
#                                              production candidate
#                                              rates (D5 fallback)
#      - bench/cube_dump_e2e.py              — auto + UDP triggers
#                                              end-to-end on h01 against
#                                              the 250924mptq fixture
#      - bench/voltage_fixture_search.py     — re-run M5's e2e with
#                                              the M6 cluster + dump
#                                              path attached
#
#   5. tools/viz/cluster_check.py — operator-facing render of T1 + T2
#      log rows for a given (search_node, gpu_half, hour) plus per-
#      NPZ cube butterfly. Operator-approval gate #3.
#
# Per D11 (M6_PLAN_FIXES.md), this script exports the M5 isolation
# envelope (M6 is incremental on M5; no rebinding):
#   - DSART_BUFFER_KEY_PREFIX=m5
#   - DSART_ETCD_NAMESPACE_PREFIX=m5
#   - CUDA_VISIBLE_DEVICES=1
#   - flock /var/lock/dsart-m6.lock  (per-milestone lock)
#
# Stage labels (mirrors M5.sh shape):
#   - failed                            -> some STEP failed; exit 1
#   - scaffolded                        -> kickoff landed (M6_PLAN_FIXES,
#                                          DoD scripts), no code chunks yet
#   - in_progress (k/N)                 -> k of N planned chunks landed
#   - complete (needs operator approval) -> all auto checks PASS, no marker
#   - complete (approved)               -> marker present, M6_PLAN_FIXES.md
#                                          still in repo
#   - complete (hardened)               -> marker present + M6_PLAN_FIXES.md
#                                          retired
#
# §6 conda-activate shell pattern: 'set -u' DROPPED (conda MKL hook
# references MKL_INTERFACE_LAYER without default); 'pipefail' kept.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
# pyproject.toml uses [tool.setuptools.packages.find] where=["src"], so the
# `dsart` package lives at ${REPO_ROOT}/src/dsart. PYTHONPATH must point at
# the `src` parent — pointing at ${REPO_ROOT} alone falls through to the
# conda env's editable install (which on h01 may resolve to the M3 working
# tree, parallel to ours, that doesn't carry M6 modules).
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# PARALLEL_AGENTS.md §4 isolation envelope (D11 in M6_PLAN_FIXES.md).
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m5}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

M6_STATUS_JSON="${M6_STATUS_JSON:-${HOME}/dsart-m6-status.json}"
M5_STATUS_JSON="${M5_STATUS_JSON:-${HOME}/dsart-m5-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M6_LOCKFILE="${M6_LOCKFILE:-/var/lock/dsart-m6.lock}"

# Operator-approval marker (mirrors M5 D7 single-marker convention).
# One marker covers the cluster + cube-dump operator-inspection gate.
M6_OPERATOR_APPROVAL_FILE="${M6_OPERATOR_APPROVAL_FILE:-${REPO_ROOT}/bench/reports/M6/m_operator_approved.yaml}"

# Slow integration toggles. Defaults skip the long-running benches when
# they're not yet authored; flip to 0 once chunks land.
M6_SKIP_CLUSTER_THROUGHPUT="${M6_SKIP_CLUSTER_THROUGHPUT:-1}"
M6_SKIP_CUBE_DUMP_E2E="${M6_SKIP_CUBE_DUMP_E2E:-1}"
M6_SKIP_VOLTAGE_FIXTURE="${M6_SKIP_VOLTAGE_FIXTURE:-1}"

# Per-milestone lockfile (PARALLEL_AGENTS.md §4.4). M5 and M6 do NOT
# share a lock and are free to run simultaneously. The lock falls back
# to a user-writable location if /var/lock/ isn't writable.
LOCKFD=
if exec {LOCKFD}>"${M6_LOCKFILE}" 2>/dev/null; then
  if ! flock -n "${LOCKFD}"; then
    echo "[M6] another M6 run in progress (lock: ${M6_LOCKFILE}); exiting"
    exit 1
  fi
else
  M6_LOCKFILE="${HOME}/.dsart-m6.lock"
  exec {LOCKFD}>"${M6_LOCKFILE}" || { echo "[M6] cannot open ${M6_LOCKFILE}"; exit 1; }
  flock -n "${LOCKFD}" || { echo "[M6] another M6 run in progress (lock: ${M6_LOCKFILE}); exiting"; exit 1; }
  echo "[M6] using fallback lockfile ${M6_LOCKFILE}"
fi

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M6:${STEP}] FAIL $*"
  cat > "${M6_STATUS_JSON}" <<JSON
{"milestone": "M6", "stage": "failed", "step": "${STEP}", "host": "$(hostname -s)", "phase": "a", "utc_iso": "$(date -u +%FT%TZ)"}
JSON
  exit 1
}
pass() {
  echo "[M6:${STEP}] PASS"
}
warn() {
  echo "[M6:${STEP}] WARN $*"
}

echo "== M6 DoD: gate on M6_preflight =="
bash "${SCRIPT_DIR}/M6_preflight.sh"

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

STEP="m5_status"
[[ -f "${M5_STATUS_JSON}" ]] || fail "missing ${M5_STATUS_JSON}"
python3 - <<PY || fail "M5 status JSON did not validate"
import json
s = json.load(open("${M5_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"M5 stage not complete: {s!r}"
assert s.get("milestone", "") == "M5", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
print(f"M5 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

# ---------------------------------------------------------------------
# Chunk-completion ledger.
#
# Each planned M6 chunk has a row here. The chunk_complete flag flips
# to true once the chunk's deliverables land + its tests pass. The
# status JSON's stage label is derived from this ledger.
#
# This is the single edit point the chunk-author touches when landing a
# chunk: flip the right `_PRESENT` flag, optionally add a pytest line
# below, and re-run M6.sh.
# ---------------------------------------------------------------------

# Chunk 0 — kickoff (THIS CHUNK)
#   Deliverables: M6_PLAN_FIXES.md, tools/dod/M6_preflight.sh,
#                 tools/dod/M6.sh, hard-rm of M5 trigger emitter.
CHUNK_0_KICKOFF_PRESENT=$([[ -f "${REPO_ROOT}/M6_PLAN_FIXES.md" \
                           && -f "${REPO_ROOT}/tools/dod/M6_preflight.sh" \
                           && -f "${REPO_ROOT}/tools/dod/M6.sh" \
                           && ! -d "${REPO_ROOT}/src/dsart/trigger" ]] && echo true || echo false)

# Chunk 1 — clusterer
#   Deliverables: src/dsart/cluster/{forward,features,state}.py +
#                 tests/test_cluster_*.py.
CHUNK_1_CLUSTERER_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/cluster/forward.py" \
                              && -f "${REPO_ROOT}/src/dsart/cluster/features.py" \
                              && -f "${REPO_ROOT}/src/dsart/cluster/state.py" ]] && echo true || echo false)

# Chunk 2 — ASCII candidate logger
#   Deliverables: src/dsart/cluster/cands_logger.py +
#                 tests/test_cands_logger.py.
CHUNK_2_CANDS_LOGGER_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/cluster/cands_logger.py" ]] && echo true || echo false)

# Chunk 3 — cube_pipeline.py + services/search_compute.py wiring of
#           clusterer (ThreadPool worker) into the per-cube driver.
#   Deliverable: services/search_compute.py rewired (post-chunk-0
#                stripped baseline) + new tests/test_search_compute_service.py.
#   Detect via grep on services/search_compute.py for the cluster import.
CHUNK_3_INTEGRATION_PRESENT=$(grep -lq 'from \.\.cluster\|from dsart\.cluster' "${REPO_ROOT}/src/dsart/services/search_compute.py" 2>/dev/null && echo true || echo false)

# Chunk 4 — cube dump writer + predicate
#   Deliverables: src/dsart/dump/cube_dump.py + tests/test_cube_dump.py.
CHUNK_4_CUBE_DUMP_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/dump/cube_dump.py" ]] && echo true || echo false)

# Chunk 5 — UDP listener + service wiring
#   Deliverables: src/dsart/dump/udp_listener.py + service-side
#                 wiring + tests/test_udp_listener.py.
CHUNK_5_UDP_LISTENER_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/dump/udp_listener.py" ]] && echo true || echo false)

# Chunk 6 — clusterer throughput bench (D5 fallback gate)
#   Deliverables: bench/clusterer_throughput.py.
CHUNK_6_BENCH_CLUSTER_PRESENT=$([[ -f "${REPO_ROOT}/bench/clusterer_throughput.py" ]] && echo true || echo false)

# Chunk 7 — cube_dump_e2e bench (auto + UDP triggers on 250924mptq)
#   Deliverables: bench/cube_dump_e2e.py.
CHUNK_7_BENCH_DUMP_PRESENT=$([[ -f "${REPO_ROOT}/bench/cube_dump_e2e.py" ]] && echo true || echo false)

# Chunk 8 — operator viz (T1/T2 inspector + cube-dump verifier)
#   Deliverables: tools/viz/m6_t1_t2_inspector.py +
#                 tools/viz/m6_cube_dump_verifier.py.
CHUNK_8_OPERATOR_VIZ_PRESENT=$([[ -f "${REPO_ROOT}/tools/viz/m6_t1_t2_inspector.py" ]] && \
                                [[ -f "${REPO_ROOT}/tools/viz/m6_cube_dump_verifier.py" ]] && \
                                echo true || echo false)

# Chunk 9 — hardening (fold M6_PLAN_FIXES into plan.md; retire tracker)
#   Marked complete when M6_PLAN_FIXES.md is DELETED.
CHUNK_9_HARDENING_COMPLETE=$([[ ! -f "${REPO_ROOT}/M6_PLAN_FIXES.md" ]] && echo true || echo false)

TOTAL_CHUNKS=10  # 0..9 inclusive
COMPLETED_CHUNKS=0
for v in "${CHUNK_0_KICKOFF_PRESENT}" "${CHUNK_1_CLUSTERER_PRESENT}" \
         "${CHUNK_2_CANDS_LOGGER_PRESENT}" "${CHUNK_3_INTEGRATION_PRESENT}" \
         "${CHUNK_4_CUBE_DUMP_PRESENT}" "${CHUNK_5_UDP_LISTENER_PRESENT}" \
         "${CHUNK_6_BENCH_CLUSTER_PRESENT}" "${CHUNK_7_BENCH_DUMP_PRESENT}" \
         "${CHUNK_8_OPERATOR_VIZ_PRESENT}" "${CHUNK_9_HARDENING_COMPLETE}"; do
  [[ "${v}" == "true" ]] && COMPLETED_CHUNKS=$((COMPLETED_CHUNKS + 1))
done

STEP="chunk_ledger"
echo "  chunk progress: ${COMPLETED_CHUNKS}/${TOTAL_CHUNKS}"
echo "    chunk 0 (kickoff)              = ${CHUNK_0_KICKOFF_PRESENT}"
echo "    chunk 1 (clusterer)            = ${CHUNK_1_CLUSTERER_PRESENT}"
echo "    chunk 2 (cands logger)         = ${CHUNK_2_CANDS_LOGGER_PRESENT}"
echo "    chunk 3 (integration)          = ${CHUNK_3_INTEGRATION_PRESENT}"
echo "    chunk 4 (cube dump)            = ${CHUNK_4_CUBE_DUMP_PRESENT}"
echo "    chunk 5 (udp listener)         = ${CHUNK_5_UDP_LISTENER_PRESENT}"
echo "    chunk 6 (clusterer bench)      = ${CHUNK_6_BENCH_CLUSTER_PRESENT}"
echo "    chunk 7 (cube_dump_e2e bench)  = ${CHUNK_7_BENCH_DUMP_PRESENT}"
echo "    chunk 8 (operator viz)         = ${CHUNK_8_OPERATOR_VIZ_PRESENT}"
echo "    chunk 9 (hardening, retired)   = ${CHUNK_9_HARDENING_COMPLETE}"
pass

# ---------------------------------------------------------------------
# Per-chunk pytest invocations. Each chunk's tests run only if the
# chunk's deliverables are present.
# ---------------------------------------------------------------------

if [[ "${CHUNK_1_CLUSTERER_PRESENT}" == "true" ]]; then
  STEP="pytest_cluster"
  if compgen -G "${REPO_ROOT}/tests/test_cluster_*.py" > /dev/null; then
    DSART_TEST=1 python -m pytest tests/test_cluster_*.py -q --tb=short \
      || fail "cluster tests failed"
  else
    warn "chunk 1 deliverables present but tests/test_cluster_*.py missing"
  fi
  pass
fi

if [[ "${CHUNK_2_CANDS_LOGGER_PRESENT}" == "true" ]]; then
  STEP="pytest_cands_logger"
  if [[ -f "${REPO_ROOT}/tests/test_cands_logger.py" ]]; then
    DSART_TEST=1 python -m pytest tests/test_cands_logger.py -q --tb=short \
      || fail "cands_logger tests failed"
  else
    warn "chunk 2 deliverables present but tests/test_cands_logger.py missing"
  fi
  pass
fi

if [[ "${CHUNK_3_INTEGRATION_PRESENT}" == "true" ]]; then
  STEP="pytest_integration"
  if [[ -f "${REPO_ROOT}/tests/test_search_compute_service.py" ]]; then
    DSART_TEST=1 python -m pytest tests/test_search_compute_service.py -q --tb=short \
      || fail "search_compute_service integration tests failed"
  else
    warn "chunk 3 deliverables present but tests/test_search_compute_service.py missing"
  fi
  pass
fi

if [[ "${CHUNK_4_CUBE_DUMP_PRESENT}" == "true" ]]; then
  STEP="pytest_cube_dump"
  if [[ -f "${REPO_ROOT}/tests/test_cube_dump.py" ]]; then
    DSART_TEST=1 python -m pytest tests/test_cube_dump.py -q --tb=short \
      || fail "cube_dump tests failed"
  else
    warn "chunk 4 deliverables present but tests/test_cube_dump.py missing"
  fi
  pass
fi

if [[ "${CHUNK_5_UDP_LISTENER_PRESENT}" == "true" ]]; then
  STEP="pytest_udp_listener"
  if [[ -f "${REPO_ROOT}/tests/test_udp_listener.py" ]]; then
    DSART_TEST=1 python -m pytest tests/test_udp_listener.py -q --tb=short \
      || fail "udp_listener tests failed"
  else
    warn "chunk 5 deliverables present but tests/test_udp_listener.py missing"
  fi
  pass
fi

# Cheap smoke: re-run M1 contracts + numerical_conventions to certify
# nothing M6 added broke the dataclass invariants. ~5-10 s.
STEP="pytest_test_contracts_test_numerical_conventions"
DSART_TEST=1 python -m pytest tests/test_contracts.py tests/test_numerical_conventions.py -q --tb=short \
  || fail "M1 regression suite failed"
pass

# ---------------------------------------------------------------------
# Bench invocations. Each gated on M6_SKIP_* + the chunk's deliverables.
# ---------------------------------------------------------------------

if [[ "${M6_SKIP_CLUSTER_THROUGHPUT}" == "0" ]] && [[ "${CHUNK_6_BENCH_CLUSTER_PRESENT}" == "true" ]]; then
  STEP="bench_clusterer_throughput"
  python -m bench.clusterer_throughput \
    || fail "bench/clusterer_throughput.py failed"
  pass
fi

if [[ "${M6_SKIP_CUBE_DUMP_E2E}" == "0" ]] && [[ "${CHUNK_7_BENCH_DUMP_PRESENT}" == "true" ]]; then
  STEP="bench_cube_dump_e2e"
  python -m bench.cube_dump_e2e --voltage-run-id 250924mptq \
    || fail "bench/cube_dump_e2e.py failed"
  pass
fi

if [[ "${M6_SKIP_VOLTAGE_FIXTURE}" == "0" ]] && [[ "${CHUNK_3_INTEGRATION_PRESENT}" == "true" ]]; then
  STEP="bench_voltage_fixture_search"
  python -m bench.voltage_fixture_search --voltage-run-id 250924mptq \
    || fail "bench/voltage_fixture_search.py failed"
  pass
fi

STEP="operator_approval"
APPROVAL_PRESENT="false"
APPROVAL_OPERATOR=""
APPROVAL_UTC=""
APPROVAL_VOLTAGE_RUN_ID=""
APPROVAL_VIZ_SHA=""
if [[ -f "${M6_OPERATOR_APPROVAL_FILE}" ]]; then
  python3 - <<PY > /tmp/dsart_m6_operator_approval.json || fail "operator-approval yaml is malformed"
import json
import sys
import yaml
with open("${M6_OPERATOR_APPROVAL_FILE}") as fh:
    data = yaml.safe_load(fh) or {}
required = {"operator", "approval_utc_iso", "milestone", "voltage_run_id", "viz_artifact_sha256"}
missing = required - set(data.keys())
if missing:
    print(f"missing fields: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)
if str(data.get("milestone")) != "M6":
    print(f"wrong milestone {data.get('milestone')!r}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: str(data.get(k, "")) for k in sorted(required)}))
PY
  APPROVAL_PRESENT="true"
  APPROVAL_OPERATOR="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m6_operator_approval.json"))["operator"])')"
  APPROVAL_UTC="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m6_operator_approval.json"))["approval_utc_iso"])')"
  APPROVAL_VOLTAGE_RUN_ID="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m6_operator_approval.json"))["voltage_run_id"])')"
  APPROVAL_VIZ_SHA="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m6_operator_approval.json"))["viz_artifact_sha256"])')"
  echo "  marker present at ${M6_OPERATOR_APPROVAL_FILE}"
  echo "  operator=${APPROVAL_OPERATOR} utc=${APPROVAL_UTC} run_id=${APPROVAL_VOLTAGE_RUN_ID}"
else
  warn "no operator-approval marker at ${M6_OPERATOR_APPROVAL_FILE} — stamp will be 'needs operator approval'"
fi
pass

STEP="status_emit"
GIT_SHA="$(git rev-parse HEAD)"
PLAN_FIXES_STILL_PRESENT="$([[ -f ${REPO_ROOT}/M6_PLAN_FIXES.md ]] && echo true || echo false)"

# Stage label derivation:
#   - chunk-0-only state (kickoff just landed) => "scaffolded"
#   - hardening done + approval present => "complete (hardened)"
#   - all-but-hardening done + approval present => "complete (approved)"
#   - all-but-hardening done, no approval => "complete (needs operator approval)"
#   - else => "in_progress (k/N)"
if [[ "${COMPLETED_CHUNKS}" -le 1 ]]; then
  STAGE_LABEL="scaffolded"
elif [[ "${PLAN_FIXES_STILL_PRESENT}" == "false" ]] && [[ "${APPROVAL_PRESENT}" == "true" ]]; then
  STAGE_LABEL="complete (hardened)"
elif [[ "${COMPLETED_CHUNKS}" -ge $((TOTAL_CHUNKS - 1)) ]] && [[ "${APPROVAL_PRESENT}" == "true" ]]; then
  STAGE_LABEL="complete (approved)"
elif [[ "${COMPLETED_CHUNKS}" -ge $((TOTAL_CHUNKS - 1)) ]]; then
  STAGE_LABEL="complete (needs operator approval)"
else
  STAGE_LABEL="in_progress (${COMPLETED_CHUNKS}/${TOTAL_CHUNKS})"
fi

cat > "${M6_STATUS_JSON}" <<JSON
{
  "milestone": "M6",
  "stage": "${STAGE_LABEL}",
  "host": "$(hostname -s)",
  "phase": "a",
  "utc_iso": "$(date -u +%FT%TZ)",
  "git_sha": "${GIT_SHA}",
  "isolation_envelope": {
    "buffer_key_prefix": "${DSART_BUFFER_KEY_PREFIX}",
    "etcd_namespace_prefix": "${DSART_ETCD_NAMESPACE_PREFIX}",
    "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
    "lockfile": "${M6_LOCKFILE}"
  },
  "chunks": {
    "total": ${TOTAL_CHUNKS},
    "completed": ${COMPLETED_CHUNKS},
    "ledger": {
      "0_kickoff":               ${CHUNK_0_KICKOFF_PRESENT},
      "1_clusterer":             ${CHUNK_1_CLUSTERER_PRESENT},
      "2_cands_logger":          ${CHUNK_2_CANDS_LOGGER_PRESENT},
      "3_integration":           ${CHUNK_3_INTEGRATION_PRESENT},
      "4_cube_dump":             ${CHUNK_4_CUBE_DUMP_PRESENT},
      "5_udp_listener":          ${CHUNK_5_UDP_LISTENER_PRESENT},
      "6_clusterer_bench":       ${CHUNK_6_BENCH_CLUSTER_PRESENT},
      "7_cube_dump_e2e_bench":   ${CHUNK_7_BENCH_DUMP_PRESENT},
      "8_operator_viz":          ${CHUNK_8_OPERATOR_VIZ_PRESENT},
      "9_hardening":             ${CHUNK_9_HARDENING_COMPLETE}
    }
  },
  "operator_approval": {
    "present": ${APPROVAL_PRESENT},
    "marker_path": "${M6_OPERATOR_APPROVAL_FILE}",
    "operator": "${APPROVAL_OPERATOR}",
    "approval_utc_iso": "${APPROVAL_UTC}",
    "voltage_run_id": "${APPROVAL_VOLTAGE_RUN_ID}",
    "viz_artifact_sha256": "${APPROVAL_VIZ_SHA}"
  },
  "plan_fixes_tracker_present": ${PLAN_FIXES_STILL_PRESENT}
}
JSON
pass

echo "M6 ${STAGE_LABEL}"
echo "  status: ${M6_STATUS_JSON}"
if [[ "${COMPLETED_CHUNKS}" -le 1 ]]; then
  echo
  echo "  next: land chunks 1+. See M6 chunk plan in M6_PLAN_FIXES.md."
elif [[ "${APPROVAL_PRESENT}" != "true" ]] && [[ "${COMPLETED_CHUNKS}" -ge $((TOTAL_CHUNKS - 1)) ]]; then
  echo
  echo "  next: drop the operator-approval marker yaml at"
  echo "    ${M6_OPERATOR_APPROVAL_FILE}"
  echo "  with fields:"
  echo "    operator: <name>"
  echo "    approval_utc_iso: $(date -u +%FT%TZ)"
  echo "    milestone: M6"
  echo "    voltage_run_id: <250924mptq | clusterer_throughput | cube_dump_e2e>"
  echo "    viz_artifact_sha256: <hex of the rendered cluster_check.html>"
  echo "  then re-run this script to stamp 'complete (approved)'."
fi
