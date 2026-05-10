#!/usr/bin/env bash
# M5 Definition-of-Done (§8 line 2312-2341) — runs on h01.
#
# M5 is the search-node detector pipeline milestone. It builds:
#
#   1. src/dsart/services/search_compute.py — production service:
#      receive-ring → fine-DM combiner → 2D iFFT imager + edge mask →
#      Layer-1 σ-clipped per-cube global normalization → Detector.forward()
#      (v1 deterministic conv bank, K_img×K_dm×K_time = 128 default kernel
#      triples; Layer-2 σ_k EMA inside; per-kernel local-max decoder;
#      cross-kernel merge) → holdoff state machine → emit-rate token bucket
#      → trigger emitter.
#
#   2. src/dsart/inject/cube_injection.py — post-imaging detector unit-test
#      injector. Bypasses every upstream stage; feeds straight into
#      Detector.forward(). Lets M5 develop the detector + decoder + emitter
#      independently of any upstream correctness (M3 / M4a).
#
#   3. Five DoD benches:
#      - bench/cube_injection_detector.py    — primary detector correctness
#                                              gate (h01 alone, no M3 dep)
#      - bench/search_node_throughput.py     — sustained throughput +
#                                              fp16/fp32 parity + detector
#                                              swap (h01 alone)
#      - bench/noise_norm_calibration.py     — Layer-2 calibration gate
#      - bench/trigger_emitter_wiring.py     — TCP fan-out + holdoff +
#                                              rate-limit
#      - bench/voltage_fixture_search.py     — M5 end-to-end on real burst
#                                              fixture; depends on M3 having
#                                              emitted the captured
#                                              transport-TX .npz set
#
#   4. tools/viz/search_detector_check.py + tools/viz/search_helpers.py —
#      operator-facing viz tool with --mode {cube_injection, burst}.
#
# Two operator-approval gates per plan §8 lines 2329 + 2339:
#   * Cube-injection detector inspection (synthetic; no fixture)
#   * Voltage-fixture search (real burst; depends on M3)
#
# Per D7 in M5_PLAN_FIXES.md, both are tracked by a single marker yaml;
# different voltage_run_id values distinguish them ("cube_injection" vs
# the burst run id).
#
# Stage labels (mirrors M2.sh shape, plus a "scaffolded" pre-chunk state
# that the M2 history elided because M2 was authored in a single sitting):
#   - failed                            -> some STEP failed; exit 1
#   - scaffolded                        -> kickoff landed (M5_PLAN_FIXES,
#                                          DoD scripts), no code chunks yet
#   - in_progress (k/N)                 -> k of N planned chunks landed
#   - complete (needs operator approval) -> all auto checks PASS, no marker
#   - complete (approved)               -> marker present, M5_PLAN_FIXES.md
#                                          still in repo
#   - complete (hardened)               -> marker present + M5_PLAN_FIXES.md
#                                          retired
#
# Per PARALLEL_AGENTS.md §4 (D3 in M5_PLAN_FIXES.md), this script exports
# the M5 isolation envelope:
#   - DSART_BUFFER_KEY_PREFIX=m5  (fada→fa5a, bada→ba5a, dada→da5a)
#   - DSART_ETCD_NAMESPACE_PREFIX=m5  (/cnf/dsart/... → /cnf/dsart-m5/...)
#   - CUDA_VISIBLE_DEVICES=1  (M5 → GPU 1; M3 → GPU 0; no contention)
#   - flock /var/lock/dsart-m5.lock  (per-milestone lock; M3 has its own)
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
# tree, parallel to ours, that doesn't carry M5 detector modules).
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# PARALLEL_AGENTS.md §4 isolation envelope (D3 in M5_PLAN_FIXES.md).
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m5}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

M5_STATUS_JSON="${M5_STATUS_JSON:-${HOME}/dsart-m5-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M5_LOCKFILE="${M5_LOCKFILE:-/var/lock/dsart-m5.lock}"

# Operator-approval marker (D7 in M5_PLAN_FIXES.md). One marker covers
# both the cube-injection gate (voltage_run_id="cube_injection") and the
# voltage-fixture gate (voltage_run_id=<burst_run_id>). Override with
# M5_OPERATOR_APPROVAL_FILE for non-default report dirs.
M5_OPERATOR_APPROVAL_FILE="${M5_OPERATOR_APPROVAL_FILE:-${REPO_ROOT}/bench/reports/M5/m_operator_approved.yaml}"

# Slow integration toggles. Defaults skip the long-running benches when
# they're not yet authored; flip to 0 once chunks land.
M5_SKIP_THROUGHPUT="${M5_SKIP_THROUGHPUT:-1}"
M5_SKIP_NOISE_NORM="${M5_SKIP_NOISE_NORM:-1}"
M5_SKIP_TRIGGER_WIRING="${M5_SKIP_TRIGGER_WIRING:-1}"
M5_SKIP_CUBE_INJECTION="${M5_SKIP_CUBE_INJECTION:-1}"
M5_SKIP_VOLTAGE_FIXTURE="${M5_SKIP_VOLTAGE_FIXTURE:-1}"

# Per-milestone lockfile (PARALLEL_AGENTS.md §4.4). M3 and M5 do NOT share
# a lock and are free to run simultaneously. The lock falls back to a
# user-writable location if /var/lock/ isn't writable on the dev host.
LOCKFD=
if exec {LOCKFD}>"${M5_LOCKFILE}" 2>/dev/null; then
  if ! flock -n "${LOCKFD}"; then
    echo "[M5] another M5 run in progress (lock: ${M5_LOCKFILE}); exiting"
    exit 1
  fi
else
  M5_LOCKFILE="${HOME}/.dsart-m5.lock"
  exec {LOCKFD}>"${M5_LOCKFILE}" || { echo "[M5] cannot open ${M5_LOCKFILE}"; exit 1; }
  flock -n "${LOCKFD}" || { echo "[M5] another M5 run in progress (lock: ${M5_LOCKFILE}); exiting"; exit 1; }
  echo "[M5] using fallback lockfile ${M5_LOCKFILE}"
fi

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M5:${STEP}] FAIL $*"
  cat > "${M5_STATUS_JSON}" <<JSON
{"milestone": "M5", "stage": "failed", "step": "${STEP}", "host": "$(hostname -s)", "phase": "a", "utc_iso": "$(date -u +%FT%TZ)"}
JSON
  exit 1
}
pass() {
  echo "[M5:${STEP}] PASS"
}
warn() {
  echo "[M5:${STEP}] WARN $*"
}

echo "== M5 DoD: gate on M5_preflight =="
bash "${SCRIPT_DIR}/M5_preflight.sh"

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

# ---------------------------------------------------------------------
# Chunk-completion ledger.
#
# Each planned M5 chunk has a row here. The "chunk_complete" flag flips
# to true once the chunk's deliverables land + its tests pass. The status
# JSON's stage label is derived from this ledger.
#
# This is the single edit point the chunk-author touches when landing a
# chunk: flip the right `_PRESENT` flag, optionally add a pytest line
# below, and re-run M5.sh.
# ---------------------------------------------------------------------

# Chunk 0 — kickoff (THIS CHUNK)
#   Deliverables: M5_PLAN_FIXES.md, tools/dod/M5_preflight.sh,
#                 tools/dod/M5.sh skeleton.
CHUNK_0_KICKOFF_PRESENT=$([[ -f "${REPO_ROOT}/M5_PLAN_FIXES.md" \
                           && -f "${REPO_ROOT}/tools/dod/M5_preflight.sh" \
                           && -f "${REPO_ROOT}/tools/dod/M5.sh" ]] && echo true || echo false)

# Chunk 1 — substrate + Detector Protocol
#   Deliverables: src/dsart/detector/{forward,kernels}.py (Protocol +
#                 v1 deterministic conv bank stub), tests/test_detector_protocol.py.
CHUNK_1_DETECTOR_PROTO_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/detector/forward.py" \
                                   && -f "${REPO_ROOT}/src/dsart/detector/kernels.py" ]] && echo true || echo false)

# Chunk 2 — cube_injection + decoder + merger
#   Deliverables: src/dsart/inject/cube_injection.py, detector/decoder.py,
#                 detector/merger.py, tests/test_detector_decoder.py,
#                 tests/test_detector_merger.py, tests/test_cube_injection.py.
CHUNK_2_DETECTOR_CORE_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/inject/cube_injection.py" \
                                  && -f "${REPO_ROOT}/src/dsart/detector/decoder.py" \
                                  && -f "${REPO_ROOT}/src/dsart/detector/merger.py" ]] && echo true || echo false)

# Chunk 3 — Layer-1 + Layer-2 noise normalization
#   Deliverables: src/dsart/noise_norm/{layer1,layer2}.py,
#                 tests/test_noise_norm*.py.
CHUNK_3_NOISE_NORM_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/noise_norm/layer1.py" \
                              && -f "${REPO_ROOT}/src/dsart/noise_norm/layer2.py" ]] && echo true || echo false)

# Chunk 4 — trigger emitter + connection pool + holdoff + rate limit
#   Deliverables: src/dsart/trigger/emitter.py + helpers,
#                 tests/test_trigger_emitter.py.
CHUNK_4_TRIGGER_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/trigger/emitter.py" ]] && echo true || echo false)

# Chunk 5 — bench/cube_injection_detector.py + tools/viz/search_detector_check.py
#            + tools/viz/search_helpers.py (operator-facing detector gate)
#   Deliverables: bench/cube_injection_detector.py, tools/viz/search_detector_check.py,
#                 tools/viz/search_helpers.py.
CHUNK_5_CUBE_INJECTION_BENCH_PRESENT=$([[ -f "${REPO_ROOT}/bench/cube_injection_detector.py" \
                                        && -f "${REPO_ROOT}/tools/viz/search_detector_check.py" \
                                        && -f "${REPO_ROOT}/tools/viz/search_helpers.py" ]] && echo true || echo false)

# Chunk 6 — fine_dm/combiner + image/imager + services/search_compute
#            + bench/search_node_throughput + bench/noise_norm_calibration
#            + bench/trigger_emitter_wiring
#   Deliverables: src/dsart/fine_dm/combiner.py, image/imager.py,
#                 services/search_compute.py, the three benches,
#                 corresponding tests/test_*.py.
CHUNK_6_SEARCH_COMPUTE_PRESENT=$([[ -f "${REPO_ROOT}/src/dsart/fine_dm/combiner.py" \
                                  && -f "${REPO_ROOT}/src/dsart/image/imager.py" \
                                  && -f "${REPO_ROOT}/src/dsart/services/search_compute.py" \
                                  && -f "${REPO_ROOT}/bench/search_node_throughput.py" \
                                  && -f "${REPO_ROOT}/bench/noise_norm_calibration.py" \
                                  && -f "${REPO_ROOT}/bench/trigger_emitter_wiring.py" ]] && echo true || echo false)

# Chunk 7 — bench/voltage_fixture_search.py (depends on M3 emitted .npz)
#   Deliverables: bench/voltage_fixture_search.py + viz --mode burst
#                 (already in chunk 5's tools/viz/search_detector_check.py).
CHUNK_7_VOLTAGE_FIXTURE_PRESENT=$([[ -f "${REPO_ROOT}/bench/voltage_fixture_search.py" ]] && echo true || echo false)

# Chunk 8 — hardening (retire M5_PLAN_FIXES.md after folding F+D into plan.md)
#   Marked complete when M5_PLAN_FIXES.md is DELETED (after plan.md edits land).
CHUNK_8_HARDENING_COMPLETE=$([[ ! -f "${REPO_ROOT}/M5_PLAN_FIXES.md" ]] && echo true || echo false)

TOTAL_CHUNKS=9  # 0..8 inclusive
COMPLETED_CHUNKS=0
for v in "${CHUNK_0_KICKOFF_PRESENT}" "${CHUNK_1_DETECTOR_PROTO_PRESENT}" \
         "${CHUNK_2_DETECTOR_CORE_PRESENT}" "${CHUNK_3_NOISE_NORM_PRESENT}" \
         "${CHUNK_4_TRIGGER_PRESENT}" "${CHUNK_5_CUBE_INJECTION_BENCH_PRESENT}" \
         "${CHUNK_6_SEARCH_COMPUTE_PRESENT}" "${CHUNK_7_VOLTAGE_FIXTURE_PRESENT}" \
         "${CHUNK_8_HARDENING_COMPLETE}"; do
  [[ "${v}" == "true" ]] && COMPLETED_CHUNKS=$((COMPLETED_CHUNKS + 1))
done

STEP="chunk_ledger"
echo "  chunk progress: ${COMPLETED_CHUNKS}/${TOTAL_CHUNKS}"
echo "    chunk 0 (kickoff)              = ${CHUNK_0_KICKOFF_PRESENT}"
echo "    chunk 1 (detector protocol)    = ${CHUNK_1_DETECTOR_PROTO_PRESENT}"
echo "    chunk 2 (detector core)        = ${CHUNK_2_DETECTOR_CORE_PRESENT}"
echo "    chunk 3 (noise normalization)  = ${CHUNK_3_NOISE_NORM_PRESENT}"
echo "    chunk 4 (trigger emitter)      = ${CHUNK_4_TRIGGER_PRESENT}"
echo "    chunk 5 (cube_injection bench) = ${CHUNK_5_CUBE_INJECTION_BENCH_PRESENT}"
echo "    chunk 6 (search_compute svc)   = ${CHUNK_6_SEARCH_COMPUTE_PRESENT}"
echo "    chunk 7 (voltage_fixture gate) = ${CHUNK_7_VOLTAGE_FIXTURE_PRESENT}"
echo "    chunk 8 (hardening, retired)   = ${CHUNK_8_HARDENING_COMPLETE}"
pass

# ---------------------------------------------------------------------
# Per-chunk pytest invocations. Each chunk's tests run only if the
# chunk's deliverables are present. As chunks land, fill in the bodies
# below; the skeleton runs the M1 regression suite as a baseline.
# ---------------------------------------------------------------------

if [[ "${CHUNK_1_DETECTOR_PROTO_PRESENT}" == "true" ]]; then
  STEP="pytest_detector_protocol"
  if compgen -G "${REPO_ROOT}/tests/test_detector_protocol.py" > /dev/null; then
    DSART_TEST=1 python -m pytest tests/test_detector_protocol.py -q --tb=short \
      || fail "tests/test_detector_protocol.py failed"
  else
    warn "no tests/test_detector_protocol.py yet; chunk 1 deliverables present but tests missing"
  fi
  pass
fi

if [[ "${CHUNK_2_DETECTOR_CORE_PRESENT}" == "true" ]]; then
  STEP="pytest_detector_core"
  if compgen -G "${REPO_ROOT}/tests/test_detector_decoder.py" "${REPO_ROOT}/tests/test_detector_merger.py" "${REPO_ROOT}/tests/test_cube_injection.py" > /dev/null; then
    DSART_TEST=1 python -m pytest tests/test_detector_decoder.py tests/test_detector_merger.py tests/test_cube_injection.py -q --tb=short \
      || fail "detector core tests failed"
  else
    warn "chunk 2 deliverables present but one of test_detector_decoder.py / test_detector_merger.py / test_cube_injection.py missing"
  fi
  pass
fi

if [[ "${CHUNK_3_NOISE_NORM_PRESENT}" == "true" ]]; then
  STEP="pytest_noise_norm"
  if compgen -G "${REPO_ROOT}/tests/test_noise_norm*.py" > /dev/null; then
    DSART_TEST=1 python -m pytest tests/test_noise_norm*.py -q --tb=short \
      || fail "noise_norm tests failed"
  else
    warn "chunk 3 deliverables present but tests/test_noise_norm*.py missing"
  fi
  pass
fi

if [[ "${CHUNK_4_TRIGGER_PRESENT}" == "true" ]]; then
  STEP="pytest_trigger_emitter"
  if compgen -G "${REPO_ROOT}/tests/test_trigger_emitter.py" > /dev/null; then
    DSART_TEST=1 python -m pytest tests/test_trigger_emitter.py -q --tb=short \
      || fail "tests/test_trigger_emitter.py failed"
  else
    warn "chunk 4 deliverables present but tests/test_trigger_emitter.py missing"
  fi
  pass
fi

# Chunks 5/6/7 are bench-driven; pytest-level smoke for them lives in
# tests/test_cube_injection_detector_bench_smoke.py etc. and is added by
# the chunk author. The benches themselves run as separate STEPs below
# (gated on M5_SKIP_*).

STEP="pytest_test_contracts_test_numerical_conventions"
# Cheap smoke: re-run M1 contracts + numerical_conventions to certify
# nothing M5 added (e.g. CandidateFlags bit 7+ allocations) broke the
# dataclass invariants. ~5-10 s.
DSART_TEST=1 python -m pytest tests/test_contracts.py tests/test_numerical_conventions.py -q --tb=short \
  || fail "M1 regression suite failed"
pass

# ---------------------------------------------------------------------
# Bench invocations. Each gated on M5_SKIP_* and on the corresponding
# chunk's deliverables. Bench skeletons exit 0 if their dependencies
# aren't yet wired; the M5_SKIP_* default of 1 keeps the DoD fast in
# scaffolded / partial-progress mode.
# ---------------------------------------------------------------------

if [[ "${M5_SKIP_CUBE_INJECTION}" == "0" ]] && [[ "${CHUNK_5_CUBE_INJECTION_BENCH_PRESENT}" == "true" ]]; then
  STEP="bench_cube_injection_detector"
  # Plan §8 line 2329: parametric (snr, width) sweep + viz heatmap.
  # The bench writes to bench/reports/<UTC>/cube_injection/M5/.
  python -m bench.cube_injection_detector --quick-sweep \
    || fail "bench/cube_injection_detector.py failed"
  pass
fi

if [[ "${M5_SKIP_NOISE_NORM}" == "0" ]] && [[ "${CHUNK_6_SEARCH_COMPUTE_PRESENT}" == "true" ]]; then
  STEP="bench_noise_norm_calibration"
  # Plan §8 lines 2322-2327: 4-sub-condition Layer-2 calibration gate.
  python -m bench.noise_norm_calibration \
    || fail "bench/noise_norm_calibration.py failed"
  pass
fi

if [[ "${M5_SKIP_TRIGGER_WIRING}" == "0" ]] && [[ "${CHUNK_4_TRIGGER_PRESENT}" == "true" ]]; then
  STEP="bench_trigger_emitter_wiring"
  # Plan §8 line 2328: 16 persistent connections, ack latency p99 ≤ 20 ms,
  # rate-limit token bucket; mock listener on 127.0.0.1:11227.
  python -m bench.trigger_emitter_wiring \
    || fail "bench/trigger_emitter_wiring.py failed"
  pass
fi

if [[ "${M5_SKIP_THROUGHPUT}" == "0" ]] && [[ "${CHUNK_6_SEARCH_COMPUTE_PRESENT}" == "true" ]]; then
  STEP="bench_search_node_throughput"
  # Plan §8 lines 2317-2321: 30-min sustained throughput + injection
  # correctness + cross-kernel merge + fp16/fp32 parity + detector swap.
  python -m bench.search_node_throughput --duration 1800 \
    || fail "bench/search_node_throughput.py failed"
  pass
fi

if [[ "${M5_SKIP_VOLTAGE_FIXTURE}" == "0" ]] && [[ "${CHUNK_7_VOLTAGE_FIXTURE_PRESENT}" == "true" ]]; then
  STEP="bench_voltage_fixture_search"
  # Plan §8 line 2330: end-to-end on 250924mptq burst fixture, depends on
  # M3 having emitted the captured transport-TX .npz set.
  python -m bench.voltage_fixture_search --voltage-run-id 250924mptq \
    || fail "bench/voltage_fixture_search.py failed"
  pass
fi

STEP="operator_approval"
# D7 (M5_PLAN_FIXES.md): single marker yaml covers both the cube-injection
# gate (voltage_run_id="cube_injection") and the voltage-fixture gate
# (voltage_run_id=<burst_run_id>). The operator drops the marker after
# inspecting the rendered viz reports.
APPROVAL_PRESENT="false"
APPROVAL_OPERATOR=""
APPROVAL_UTC=""
APPROVAL_VOLTAGE_RUN_ID=""
APPROVAL_VIZ_SHA=""
if [[ -f "${M5_OPERATOR_APPROVAL_FILE}" ]]; then
  python3 - <<PY > /tmp/dsart_m5_operator_approval.json || fail "operator-approval yaml is malformed"
import json
import sys
import yaml
with open("${M5_OPERATOR_APPROVAL_FILE}") as fh:
    data = yaml.safe_load(fh) or {}
required = {"operator", "approval_utc_iso", "milestone", "voltage_run_id", "viz_artifact_sha256"}
missing = required - set(data.keys())
if missing:
    print(f"missing fields: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)
if str(data.get("milestone")) != "M5":
    print(f"wrong milestone {data.get('milestone')!r}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: str(data.get(k, "")) for k in sorted(required)}))
PY
  APPROVAL_PRESENT="true"
  APPROVAL_OPERATOR="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m5_operator_approval.json"))["operator"])')"
  APPROVAL_UTC="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m5_operator_approval.json"))["approval_utc_iso"])')"
  APPROVAL_VOLTAGE_RUN_ID="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m5_operator_approval.json"))["voltage_run_id"])')"
  APPROVAL_VIZ_SHA="$(python3 -c 'import json; print(json.load(open("/tmp/dsart_m5_operator_approval.json"))["viz_artifact_sha256"])')"
  echo "  marker present at ${M5_OPERATOR_APPROVAL_FILE}"
  echo "  operator=${APPROVAL_OPERATOR} utc=${APPROVAL_UTC} run_id=${APPROVAL_VOLTAGE_RUN_ID}"
else
  warn "no operator-approval marker at ${M5_OPERATOR_APPROVAL_FILE} — stamp will be 'needs operator approval'"
fi
pass

STEP="status_emit"
GIT_SHA="$(git rev-parse HEAD)"
PLAN_FIXES_STILL_PRESENT="$([[ -f ${REPO_ROOT}/M5_PLAN_FIXES.md ]] && echo true || echo false)"

# Stage label derivation:
#   - If COMPLETED_CHUNKS == TOTAL_CHUNKS - 1 (only hardening unfinished)
#     AND all-bench skips are 0 AND approval marker present => approved
#   - If hardening also done => hardened
#   - If all M5-internal benches passing but voltage-fixture skipped =>
#     "complete (needs operator approval, voltage-fixture gate pending M3)"
#   - Else => "in_progress (k/N)"
#   - On chunk-0-only state (kickoff just landed, no other chunks) =>
#     "scaffolded"
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

cat > "${M5_STATUS_JSON}" <<JSON
{
  "milestone": "M5",
  "stage": "${STAGE_LABEL}",
  "host": "$(hostname -s)",
  "phase": "a",
  "utc_iso": "$(date -u +%FT%TZ)",
  "git_sha": "${GIT_SHA}",
  "isolation_envelope": {
    "buffer_key_prefix": "${DSART_BUFFER_KEY_PREFIX}",
    "etcd_namespace_prefix": "${DSART_ETCD_NAMESPACE_PREFIX}",
    "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
    "lockfile": "${M5_LOCKFILE}"
  },
  "chunks": {
    "total": ${TOTAL_CHUNKS},
    "completed": ${COMPLETED_CHUNKS},
    "ledger": {
      "0_kickoff":               ${CHUNK_0_KICKOFF_PRESENT},
      "1_detector_protocol":     ${CHUNK_1_DETECTOR_PROTO_PRESENT},
      "2_detector_core":         ${CHUNK_2_DETECTOR_CORE_PRESENT},
      "3_noise_norm":            ${CHUNK_3_NOISE_NORM_PRESENT},
      "4_trigger_emitter":       ${CHUNK_4_TRIGGER_PRESENT},
      "5_cube_injection_bench":  ${CHUNK_5_CUBE_INJECTION_BENCH_PRESENT},
      "6_search_compute":        ${CHUNK_6_SEARCH_COMPUTE_PRESENT},
      "7_voltage_fixture":       ${CHUNK_7_VOLTAGE_FIXTURE_PRESENT},
      "8_hardening":             ${CHUNK_8_HARDENING_COMPLETE}
    }
  },
  "operator_approval": {
    "present": ${APPROVAL_PRESENT},
    "marker_path": "${M5_OPERATOR_APPROVAL_FILE}",
    "operator": "${APPROVAL_OPERATOR}",
    "approval_utc_iso": "${APPROVAL_UTC}",
    "voltage_run_id": "${APPROVAL_VOLTAGE_RUN_ID}",
    "viz_artifact_sha256": "${APPROVAL_VIZ_SHA}"
  },
  "plan_fixes_tracker_present": ${PLAN_FIXES_STILL_PRESENT}
}
JSON
pass

echo "M5 ${STAGE_LABEL}"
echo "  status: ${M5_STATUS_JSON}"
if [[ "${COMPLETED_CHUNKS}" -le 1 ]]; then
  echo
  echo "  next: land chunks 1+. See M5 chunk plan in the kickoff transcript."
elif [[ "${APPROVAL_PRESENT}" != "true" ]] && [[ "${COMPLETED_CHUNKS}" -ge $((TOTAL_CHUNKS - 1)) ]]; then
  echo
  echo "  next: drop the operator-approval marker yaml at"
  echo "    ${M5_OPERATOR_APPROVAL_FILE}"
  echo "  with fields:"
  echo "    operator: <name>"
  echo "    approval_utc_iso: $(date -u +%FT%TZ)"
  echo "    milestone: M5"
  echo "    voltage_run_id: <cube_injection | burst run id>"
  echo "    viz_artifact_sha256: <hex of the rendered report.html>"
  echo "  then re-run this script to stamp 'complete (approved)'."
fi
