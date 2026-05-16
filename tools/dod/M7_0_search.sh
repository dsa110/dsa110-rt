#!/usr/bin/env bash
# M7.0 DoD (search side) — control-plane skeleton smoke on one search node.
#
# Verifies the search-side instance of dsart_rt is wired correctly:
#
#   1. dsart_rt -in search_rt -cn <n> boots.
#   2. /mon/service/search_rt/<n> heartbeat publishes state=stopped.
#   3. `start` verb -> state=running. dsart_search_rt.yaml is empty by
#      design (M7.2 wires search_rx + search_compute), so routines={}
#      and buffers={} are the EXPECTED healthy state for M7.0.
#   4. `stop` verb -> state=stopped.
#   5. SIGTERM exits cleanly.
#
# Canonical host: n01 (matches Q7 of the M7 scoping doc). Run from the
# m7/control-plane branch on a search node with dsa110-rt conda env +
# the production etcd endpoint reachable.
#
# Exit: 0 = PASS; 1 = FAIL.
set -eo pipefail
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU,LP64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CN=${DSART_M7_0_SEARCH_CN:-1}
NS=search_rt
LOG=${DSART_M7_0_SEARCH_LOG:-/tmp/dsart-rt-m7_0-search.log}
PIDFILE=${DSART_M7_0_SEARCH_PIDFILE:-/tmp/dsart-rt-m7_0-search.pid}

cleanup() {
  rc=$?
  if [ -f "$PIDFILE" ]; then
    kill -KILL "$(cat $PIDFILE)" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  exit "$rc"
}
trap cleanup EXIT

source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate dsa110-rt

echo "=== M7.0 search-side smoke on $(hostname) (cn=${CN}) ==="

echo
echo "=== STEP 1: launch dsart_rt -in search_rt -cn ${CN} ==="
nohup python3 -u -m dsart.services.dsart_rt -in search_rt -cn "${CN}" \
                                            --log-level INFO \
                                            >"${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
sleep 3
pid=$(cat "${PIDFILE}")
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "FAIL: orchestrator died at startup" >&2
  tail -30 "${LOG}" >&2
  exit 1
fi
echo "orchestrator pid=${pid}"

echo
echo "=== STEP 2: /mon/service/${NS}/${CN} heartbeat reports state=stopped ==="
python3 -c "
import sys, json
from dsautils.dsa_store import DsaStore
d = DsaStore().get_dict('/mon/service/${NS}/${CN}') or {}
print(json.dumps(d, indent=2))
if d.get('state') != 'stopped':
    print('FAIL: expected state=stopped', file=sys.stderr); sys.exit(1)
print('OK heartbeat')"

echo
echo "=== STEP 3: send start verb ==="
python3 -c "from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/${NS}/${CN}', {'cmd': 'start', 'val': None})"
sleep 4

echo
echo "=== STEP 4: state=running; routines={} and buffers={} (expected for M7.0 search-side) ==="
python3 -c "
import sys, json
from dsautils.dsa_store import DsaStore
d = DsaStore().get_dict('/mon/${NS}/${CN}') or {}
print(json.dumps(d, indent=2))
ok = True
if d.get('state') != 'running':
    print(f\"FAIL state={d.get('state')!r}\", file=sys.stderr); ok=False
if d.get('routines'):
    print(f\"FAIL routines should be empty for M7.0: {d.get('routines')}\", file=sys.stderr); ok=False
if d.get('buffers'):
    print(f\"FAIL buffers should be empty for M7.0: {d.get('buffers')}\", file=sys.stderr); ok=False
sys.exit(0 if ok else 1)"

echo
echo "=== STEP 5: send stop verb ==="
python3 -c "from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/${NS}/${CN}', {'cmd': 'stop', 'val': None})"
sleep 3

echo
echo "=== STEP 6: state=stopped ==="
python3 -c "
import sys, json
from dsautils.dsa_store import DsaStore
d = DsaStore().get_dict('/mon/${NS}/${CN}') or {}
print(json.dumps(d, indent=2))
sys.exit(0 if d.get('state') == 'stopped' else 1)"

echo
echo "=== STEP 7: SIGTERM orchestrator; verify clean exit ==="
kill -TERM "$(cat "${PIDFILE}")"
sleep 3
if kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
  echo "FAIL: orchestrator still running after SIGTERM" >&2
  exit 1
fi
rm -f "${PIDFILE}"
echo "OK orchestrator exited cleanly"

echo
echo "=== M7.0 search-side PASS ==="
echo "log: ${LOG}"
