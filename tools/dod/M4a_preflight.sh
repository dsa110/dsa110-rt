#!/usr/bin/env bash
# M4a pre-flight readiness check (§8 line 2377 / §4.3) — runs on h01.
#
# Purpose: fail fast if anything M4a depends on is broken before authoring.
# M4a builds the production transport plane (72-byte header, fragment
# reassembly, POSIX-shm SPMC receive ring, per-payload pattern_id verify)
# on top of M3's chunk-8 transport scaffold + M3 chunk-3a sparsity_pattern
# helpers + M5 chunk-6b-α RxRingSource Protocol. So it depends on M3 and
# M5 being merged into main:
#
#   * M3 must be complete (hardened or approved tolerated). M4a reuses
#     M3's `transport/{tx,rx,frame}.py`, `transport/captured_npz.py`,
#     `transport/loopback_capture.py`, and `grid/sparsity_pattern.py`
#     (build_pattern + _pattern_id_payload + predict_pattern_id).
#   * M5 must be complete (hardened or approved tolerated). M4a's
#     production RxRingSource impl satisfies the Protocol defined in
#     `src/dsart/services/rx_ring.py` (M5 chunk-6b-α).
#   * Per-agent isolation env vars MUST be set to the M4a defaults
#     (PARALLEL_AGENTS.md §8 + plan §8 line 407):
#       - CUDA_VISIBLE_DEVICES = 0 OR 1   (M4a is transport-only; can
#                                          share either GPU with M3/M5
#                                          with negligible contention)
#       - DSART_BUFFER_KEY_PREFIX = m4a   (so any shm buffers carry m4a
#                                          prefix, distinct from m3/m5)
#       - DSART_ETCD_NAMESPACE_PREFIX = m4a
#   * Port range 127.0.0.1:9000-9015 (per-chgroup) reserved for M4a's
#     loopback transport; the bench will attempt SO_REUSEPORT but
#     concurrent M3/M5 runs MUST NOT bind these.
#   * /var/lock/ writable for the per-milestone flock guard.
#
# §6 conda-activate shell pattern: 'set -u' DROPPED (conda MKL hook
# references MKL_INTERFACE_LAYER without default); 'pipefail' kept.
#
# Read-only: this script never writes to the repo.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# Per-milestone env defaults. Caller may override for advanced testing,
# but the canonical h01 M4a run uses these.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m4a}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m4a}"

M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M3_STATUS_JSON="${M3_STATUS_JSON:-${HOME}/dsart-m3-status.json}"
M5_STATUS_JSON="${M5_STATUS_JSON:-${HOME}/dsart-m5-status.json}"
M4A_LOCKFILE="${M4A_LOCKFILE:-/var/lock/dsart-m4a.lock}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M4a_pre:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M4a_pre:${STEP}] PASS"
}
warn() {
  echo "[M4a_pre:${STEP}] WARN $*"
}

STEP="host_identity"
[[ "$(hostname -s)" == "lxd110h01" ]] || fail "expected lxd110h01, got $(hostname -s)"
pass

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

STEP="m3_status"
# M3 must be complete (hardened or approved tolerated). M4a uses M3's
# transport scaffold + sparsity_pattern + captured_npz.
[[ -f "${M3_STATUS_JSON}" ]] || fail "missing ${M3_STATUS_JSON}"
python3 - <<PY || fail "M3 status JSON did not validate"
import json
s = json.load(open("${M3_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M3", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
print(f"M3 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m5_status"
# M5 must be complete (hardened or approved tolerated). M4a's production
# RxRingSource impl satisfies the Protocol defined in M5 chunk-6b-α.
[[ -f "${M5_STATUS_JSON}" ]] || fail "missing ${M5_STATUS_JSON}"
python3 - <<PY || fail "M5 status JSON did not validate"
import json
s = json.load(open("${M5_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M5", f"wrong milestone: {s!r}"
print(f"M5 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="git_clean"
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes in ${REPO_ROOT}; run 'git status'"
pass

STEP="git_branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "${BRANCH}" == "m4a/main" || "${BRANCH}" == "integration/m4a"* ]] \
  || fail "expected branch=m4a/main or integration/m4a*, got ${BRANCH}"
git remote -v | grep -q '^origin' || fail "missing origin remote"
pass

STEP="conda_env"
[[ "${CONDA_DEFAULT_ENV:-}" == "dsa110-rt" ]] || fail "CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', expected dsa110-rt"
pass

STEP="dsart_import"
python -c "import dsart; from dsart.transport import frame, tx, rx, loopback_capture, captured_npz; from dsart.grid.sparsity_pattern import build_pattern, predict_pattern_id, SparsityPattern; from dsart.services.rx_ring import RxRingSource, CubeRingSlot" \
  || fail "dsart import failed (chunk-8 transport scaffold or M5 rx_ring missing)"
pass

STEP="m4a_lockfile"
if ! touch "${M4A_LOCKFILE}" 2>/dev/null; then
  warn "${M4A_LOCKFILE} not writable; falling back to /tmp/dsart-m4a.lock"
  M4A_LOCKFILE="/tmp/dsart-m4a.lock"
  touch "${M4A_LOCKFILE}" || fail "cannot create fallback lockfile ${M4A_LOCKFILE}"
fi
pass

STEP="m4a_ports_free"
# Verify the 16-chgroup port range 9000..9015 has nothing bound.
PORTS_IN_USE=$(ss -lun 2>/dev/null | awk '$5 ~ /:(900[0-9]|901[0-5])$/ {print $5}' || true)
if [[ -n "${PORTS_IN_USE}" ]]; then
  fail "M4a ports 9000-9015 already bound: ${PORTS_IN_USE}"
fi
pass

STEP="parallel_agents_doc"
[[ -f "${REPO_ROOT}/PARALLEL_AGENTS.md" ]] || fail "missing PARALLEL_AGENTS.md"
pass

STEP="writable_paths"
mkdir -p "${REPO_ROOT}/bench/reports" "${HOME}/dsart-integration-logs" 2>/dev/null
[[ -w "${REPO_ROOT}/bench/reports" ]] || fail "${REPO_ROOT}/bench/reports not writable"
[[ -w "${HOME}" ]] || fail "${HOME} not writable"
pass

STEP="m4a_target_files"
# Sanity: chunk-8 scaffold exists; M4a will extend in-place + add new
# modules. List only the files chunks 1..8 need either to exist (read)
# or to be writeable (write).
for f in src/dsart/transport/frame.py src/dsart/transport/tx.py src/dsart/transport/rx.py src/dsart/grid/sparsity_pattern.py src/dsart/services/rx_ring.py; do
  [[ -f "${REPO_ROOT}/${f}" ]] || fail "missing M4a-required source: ${f}"
done
echo "  info: chunk-8 scaffold + M3 sparsity + M5 rx_ring Protocol all present"
pass

echo "M4a_preflight PASS"
