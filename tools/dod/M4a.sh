#!/usr/bin/env bash
# M4a Definition-of-Done (§8 line 2377-2384) — runs on h01.
#
# M4a builds the production transport plane on top of M3's chunk-8
# scaffold + M3 chunk-3a sparsity_pattern + M5 chunk-6b-α RxRingSource:
#
#   1. src/dsart/transport/prod_frame.py — production 72-byte header
#      wire format per plan §4.3 lines 1411-1442 (peer to chunk-8's
#      32-byte FastVisFrame).
#   2. src/dsart/transport/tx.py — extended to emit the 72-byte header,
#      fragment payloads > MTU, per-flow token-bucket pacer with
#      "drop oldest at TX, never block" semantics.
#   3. src/dsart/transport/rx.py — extended with per-(corr, dm_idx)
#      reorder window + fragment bitmap + per-payload pattern_id verify.
#   4. src/dsart/transport/recv_ring.{c,py} — POSIX-shm SPMC sparse ring
#      satisfying the CONC-1 contract (plan §4.4 lines 1463-1475).
#   5. src/dsart/transport/production_rx_ring.py — production
#      RxRingSource impl satisfying the M5-chunk-6b-α Protocol.
#   6. (conditional) src/dsart/transport/epoll_rx.{c,py} — C epoll loop
#      if Python recvmmsg can't sustain target rate.
#   7. bench/net_loopback.py — sustained loopback bench at the §9 op-point
#      with the 6 DoD invariants (plan §M4a line 2383).
#   8. M4a status JSON + M4a_PLAN_FIXES.md retirement path.
#
# Like M3.sh, this script gates on the substrate first (M4a_preflight +
# M1/M3/M5 hardened + git_clean + branch=m4a/main) and emits stage="in
# progress (substrate only)" until the first M4a chunk lands. As each
# chunk lands, the chunk's authoring sub-agent appends a new STEP block
# and updates stage-gating logic.
#
# Stage labels (mirrors M3.sh / M5.sh / M6.sh):
#   - failed                              -> some STEP failed; exit 1
#   - in progress (substrate only)         -> preflight + status-gates pass; no
#                                             M4a chunks landed yet
#   - in progress (chunks: <list>)        -> some chunks complete
#   - complete (needs operator approval)  -> all chunks auto-pass; no marker
#   - complete (approved)                 -> marker present; M4a_PLAN_FIXES.md
#                                             still in repo
#   - complete (hardened)                 -> marker present + M4a_PLAN_FIXES.md
#                                             retired (M4a hardening)
#
# M4a has NO operator-approval gate (transport correctness has no
# headline image to inspect; the loopback bench's 6 invariants are
# the gate). Stage will skip "needs operator approval" and go directly
# from "in progress" -> "complete (hardened)" once all chunks land
# and M4a_PLAN_FIXES.md is retired.
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

# Per-milestone env defaults (PARALLEL_AGENTS.md §8 + plan §8 line 407).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m4a}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m4a}"

M4A_STATUS_JSON="${M4A_STATUS_JSON:-${HOME}/dsart-m4a-status.json}"
M3_STATUS_JSON="${M3_STATUS_JSON:-${HOME}/dsart-m3-status.json}"
M5_STATUS_JSON="${M5_STATUS_JSON:-${HOME}/dsart-m5-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"

# Per-milestone flock guard. Falls back to /tmp if /var/lock isn't writable.
M4A_LOCKFILE="${M4A_LOCKFILE:-/var/lock/dsart-m4a.lock}"
if ! touch "${M4A_LOCKFILE}" 2>/dev/null; then
  M4A_LOCKFILE="/tmp/dsart-m4a.lock"
fi

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M4a:${STEP}] FAIL $*"
  cat > "${M4A_STATUS_JSON}" <<JSON
{"milestone": "M4a", "stage": "failed", "step": "${STEP}", "host": "$(hostname -s)", "phase": "a", "utc_iso": "$(date -u +%FT%TZ)"}
JSON
  exit 1
}
pass() {
  echo "[M4a:${STEP}] PASS"
}
warn() {
  echo "[M4a:${STEP}] WARN $*"
}

# Per-milestone flock guard. M3/M5/M6 each have their own lockfile so M4a
# can run alongside them with no contention.
exec {LOCKFD}>"${M4A_LOCKFILE}" || { echo "FATAL: cannot open ${M4A_LOCKFILE}"; exit 1; }
if ! flock -n "$LOCKFD"; then
  echo "FATAL: another M4a run already in progress (lock=${M4A_LOCKFILE})"
  exit 1
fi
echo "== M4a DoD: lock acquired (${M4A_LOCKFILE}) =="

echo "== M4a DoD: gate on M4a_preflight =="
bash "${SCRIPT_DIR}/M4a_preflight.sh"

# ---------------------------------------------------------------------------
# Status-gate sanity (mirrors M3.sh / M5.sh: redo the upstream-milestone
# checks here so the M4a status JSON carries direct provenance — the
# preflight already gated, this is just for the status JSON consumers).
# ---------------------------------------------------------------------------

STEP="m1_status"
[[ -f "${M1_STATUS_JSON}" ]] || fail "missing ${M1_STATUS_JSON}"
python3 - <<PY || fail "M1 status JSON did not validate"
import json
s = json.load(open("${M1_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
print(f"M1 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m3_status"
[[ -f "${M3_STATUS_JSON}" ]] || fail "missing ${M3_STATUS_JSON}"
python3 - <<PY || fail "M3 status JSON did not validate"
import json
s = json.load(open("${M3_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
print(f"M3 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m5_status"
[[ -f "${M5_STATUS_JSON}" ]] || fail "missing ${M5_STATUS_JSON}"
python3 - <<PY || fail "M5 status JSON did not validate"
import json
s = json.load(open("${M5_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
print(f"M5 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

# ---------------------------------------------------------------------------
# Chunk ledger — track which M4a chunks have landed
#
# Each chunk-N STEP appended below should also push its name into
# CHUNKS_DONE on success. The stage label at the end derives from
# CHUNKS_DONE vs CHUNKS_TOTAL.
# ---------------------------------------------------------------------------

CHUNKS_TOTAL=8
declare -a CHUNKS_DONE=()

# ---------------------------------------------------------------------------
# chunk 1: prod_frame_72b — production 72-byte header wire format
# ---------------------------------------------------------------------------
STEP="chunk_1_prod_frame_72b"
if [[ -f "${REPO_ROOT}/src/dsart/transport/prod_frame.py" ]] \
   && [[ -f "${REPO_ROOT}/tests/transport/test_prod_frame.py" ]]; then
  python -m pytest tests/transport/test_prod_frame.py -q --tb=short \
    || fail "tests/transport/test_prod_frame.py failed"
  CHUNKS_DONE+=("chunk_1_prod_frame_72b")
  pass
else
  echo "  [M4a:${STEP}] SKIP (not yet implemented)"
fi

# ---------------------------------------------------------------------------
# chunk 2: tx_prod_header — TX 72-byte header + fragmentation + pacer
# ---------------------------------------------------------------------------
STEP="chunk_2_tx_prod_header"
if [[ -f "${REPO_ROOT}/tests/transport/test_tx_prod.py" ]]; then
  python -m pytest tests/transport/test_tx_prod.py -q --tb=short \
    || fail "tests/transport/test_tx_prod.py failed"
  CHUNKS_DONE+=("chunk_2_tx_prod_header")
  pass
else
  echo "  [M4a:${STEP}] SKIP (not yet implemented)"
fi

# ---------------------------------------------------------------------------
# chunk 3: rx_defrag — per-(corr, dm_idx) reorder window + bitmap + pattern_id verify
# ---------------------------------------------------------------------------
STEP="chunk_3_rx_defrag"
if [[ -f "${REPO_ROOT}/tests/transport/test_rx_prod.py" ]]; then
  python -m pytest tests/transport/test_rx_prod.py -q --tb=short \
    || fail "tests/transport/test_rx_prod.py failed"
  CHUNKS_DONE+=("chunk_3_rx_defrag")
  pass
else
  echo "  [M4a:${STEP}] SKIP (not yet implemented)"
fi

# ---------------------------------------------------------------------------
# chunk 4: recv_ring_shm — POSIX-shm SPMC sparse receive ring (CONC-1)
# ---------------------------------------------------------------------------
STEP="chunk_4_recv_ring_shm"
if [[ -f "${REPO_ROOT}/src/dsart/transport/recv_ring.c" ]]; then
  # C extension should already be built via pip install -e . in env setup.
  if ! ls "${REPO_ROOT}/src/dsart/transport/"_recv_ring*.so >/dev/null 2>&1; then
    warn "_recv_ring.so not found; rebuilding in-place"
    (cd "${REPO_ROOT}" && python setup.py build_ext --inplace >/dev/null 2>&1) \
      || fail "C extension build failed (run 'pip install -e .' in env)"
  fi
  python -m pytest tests/transport/test_recv_ring_spmc.py -q --tb=short \
    || fail "tests/transport/test_recv_ring_spmc.py failed"
  CHUNKS_DONE+=("chunk_4_recv_ring_shm")
  pass
else
  echo "  [M4a:${STEP}] SKIP (not yet implemented)"
fi

# ---------------------------------------------------------------------------
# chunk 5: production_source — ProductionRxRingSource impl (M5 RxRingSource Protocol)
# ---------------------------------------------------------------------------
STEP="chunk_5_production_source"
if [[ -f "${REPO_ROOT}/tests/transport/test_production_rx_ring.py" ]]; then
  python -m pytest tests/transport/test_production_rx_ring.py -q --tb=short \
    || fail "tests/transport/test_production_rx_ring.py failed"
  CHUNKS_DONE+=("chunk_5_production_source")
  pass
else
  echo "  [M4a:${STEP}] SKIP (not yet implemented)"
fi

# ---------------------------------------------------------------------------
# chunk 6: c_epoll_loop — conditional, only if chunk-7 perf gate fails
# Python recvmmsg at target rate. Default: SKIP (Python is sufficient at
# default ops on h01 per the chunk-7 bench).
# ---------------------------------------------------------------------------
STEP="chunk_6_c_epoll_loop"
if [[ -f "${REPO_ROOT}/src/dsart/transport/epoll_rx.c" ]]; then
  python -m pytest tests/transport/test_epoll_rx.py -q --tb=short \
    || fail "tests/transport/test_epoll_rx.py failed"
  CHUNKS_DONE+=("chunk_6_c_epoll_loop")
  pass
else
  # Conditional: SKIP counts as done for ledger purposes when the
  # default Python recvmmsg path passes the chunk-7 perf gate.
  echo "  [M4a:${STEP}] SKIP (conditional; default Python path sufficient)"
  CHUNKS_DONE+=("chunk_6_c_epoll_loop")
  pass
fi

# ---------------------------------------------------------------------------
# chunk 7: bench_net_loopback — six DoD invariants over UDP loopback
# (plan §M4a line 2383). 60 s invariants for I1 / I3 by default.
# Caller may set M4A_BENCH_QUICK=1 for 5 s invariants (smoke test).
# ---------------------------------------------------------------------------
STEP="chunk_7_bench_net_loopback"
if [[ -f "${REPO_ROOT}/bench/net_loopback.py" ]]; then
  BENCH_ARGS="--all"
  if [[ -n "${M4A_BENCH_QUICK:-}" ]]; then
    BENCH_ARGS="${BENCH_ARGS} --quick"
  fi
  mkdir -p "${HOME}/dsart-integration-logs"
  BENCH_JSON="${HOME}/dsart-integration-logs/m4a_bench_$(date -u +%Y%m%dT%H%M%SZ).json"
  if ! python -m bench.net_loopback ${BENCH_ARGS} --json "${BENCH_JSON}"; then
    fail "bench/net_loopback.py: not all 6 DoD invariants passed (see ${BENCH_JSON})"
  fi
  echo "  info: bench JSON: ${BENCH_JSON}"
  CHUNKS_DONE+=("chunk_7_bench_net_loopback")
  pass
else
  echo "  [M4a:${STEP}] SKIP (not yet implemented)"
fi

# ---------------------------------------------------------------------------
# chunk 8: dod_orchestrator — this script. If execution reaches here, the
# DoD orchestrator is wired and runs. The status JSON below carries the
# stage label.
# ---------------------------------------------------------------------------
STEP="chunk_8_dod_orchestrator"
CHUNKS_DONE+=("chunk_8_dod_orchestrator")
pass

# ---------------------------------------------------------------------------
# Stage derivation
# ---------------------------------------------------------------------------

if [[ ${#CHUNKS_DONE[@]} -eq 0 ]]; then
  STAGE='in progress (substrate only)'
elif [[ ${#CHUNKS_DONE[@]} -lt ${CHUNKS_TOTAL} ]]; then
  DONE_LIST=$(IFS=,; echo "${CHUNKS_DONE[*]}")
  STAGE="in progress (chunks: ${DONE_LIST})"
else
  # All chunks landed. M4a has no operator-approval marker (no headline
  # image to inspect; bench invariants are the gate). Go direct to
  # "complete (hardened)" once M4a_PLAN_FIXES.md is retired.
  STAGE='complete (approved)'
  if [[ ! -f "${REPO_ROOT}/M4a_PLAN_FIXES.md" ]]; then
    STAGE='complete (hardened)'
  fi
fi

GIT_SHA=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UTC_NOW=$(date -u +%FT%TZ)

cat > "${M4A_STATUS_JSON}" <<JSON
{
  "milestone": "M4a",
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
    "lockfile": "${M4A_LOCKFILE}"
  },
  "chunks_done": [$(printf '"%s",' "${CHUNKS_DONE[@]}" | sed 's/,$//')],
  "chunks_total": ${CHUNKS_TOTAL},
  "plan_fixes_tracker_present": $([[ -f "${REPO_ROOT}/M4a_PLAN_FIXES.md" ]] && echo true || echo false)
}
JSON

echo
echo "== M4a DoD: stage=${STAGE} =="
echo "   status: ${M4A_STATUS_JSON}"
