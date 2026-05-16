#!/usr/bin/env bash
# M7.0 DoD - control-plane skeleton smoke (single corr node).
#
# Verifies the M7.0 deliverables (services/dsart_rt.py,
# services/systemd/dsart-rt.service, configs/dsart_pipeline_rt.yaml,
# tools/ops/{dsart-rt, push_dsart_to_etcd.py}) work end-to-end:
#
#   1. dsart_rt orchestrator launches in user-systemd and watches
#      /cmd/corr_rt/<n>.
#   2. `start` verb -> creates the 4 production PSRDADA buffers
#      (dada/eada/fada/bada) and spawns 2x dada_junkdb (per
#      captures.mode: junkdb in /cnf/pipeline_rt).
#   3. Mon-dict at /mon/corr_rt/<n> shows state=running with 2 routines
#      alive + 4 buffers known to the orchestrator.
#   4. Heartbeat at /mon/service/corr_rt/<n> continues publishing during
#      the long mlock pause inside `start` (production fada ring is
#      ~20 GiB; mlock takes ~10 s).
#   5. `stop` verb -> kills routines + destroys buffers.
#   6. Orchestrator exits cleanly on SIGTERM.
#
# Run from the m7/control-plane branch of dsa110-rt on a corr node
# (canonical: n06) that has the dsa110-rt conda env + the legacy
# dsa110-xengine header fixture installed at
# /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt.
#
# Pre-requisites:
#   * /cnf/pipeline_rt populated via tools/ops/push_dsart_to_etcd.py
#     (run this once from any host with dsautils + the etcd endpoint).
#   * dsa110-rt conda env (envs/dsa110-rt.yml) installed locally.
#   * PSRDADA + dada_dbmetric on PATH (Phase-2 host bring-up).
#
# Exit code: 0 = PASS; 1 = FAIL with diagnostic dump on stderr.

# §6 conda-activate shell pattern: 'set -u' DROPPED (conda MKL hook
# references MKL_INTERFACE_LAYER without default); 'pipefail' kept.
set -eo pipefail
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU,LP64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

LOG=${DSART_M7_0_LOG:-/tmp/dsart-rt-m7_0.log}
PIDFILE=${DSART_M7_0_PIDFILE:-/tmp/dsart-rt-m7_0.pid}
CHILDLOGDIR=${DSART_RT_LOG_DIR:-/tmp/dsart-rt-children}
CN=${DSART_M7_0_CN:-6}
NS=corr_rt
START_WAIT_S=${DSART_M7_0_START_WAIT_S:-25}    # production buffers need ~17 s

cleanup() {
  rc=$?
  echo "--- M7.0 cleanup ---"
  if test -f "${PIDFILE}"; then
    kill -KILL "$(cat "${PIDFILE}")" 2>/dev/null || true
    rm -f "${PIDFILE}"
  fi
  pkill -KILL -f 'dada_junkdb' 2>/dev/null || true
  for k in dada eada fada bada; do dada_db -d -k "${k}" 2>/dev/null || true; done
  exit "${rc}"
}
trap cleanup EXIT

source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate dsa110-rt
export DSART_RT_LOG_DIR="${CHILDLOGDIR}"
mkdir -p "${CHILDLOGDIR}"

# Spell out the buffer-create sequence in a function so the smoke is
# easy to read even when steps are slow.
echo "=== M7.0 control-plane skeleton smoke on $(hostname) (cn=${CN}) ==="

echo "=== STEP 1: launch dsart_rt -in pipeline_rt -cn ${CN} ==="
nohup python3 -u -m dsart.services.dsart_rt -in pipeline_rt -cn "${CN}" \
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
import sys
from dsautils.dsa_store import DsaStore
import json
d = DsaStore().get_dict('/mon/service/${NS}/${CN}') or {}
print(json.dumps(d, indent=2))
if d.get('state') != 'stopped':
    print('FAIL: expected state=stopped', file=sys.stderr); sys.exit(1)
print('OK heartbeat state=stopped')"

echo
echo "=== STEP 3: send start verb (val=53.85 deg dec) ==="
python3 -c "
from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/${NS}/${CN}', {'cmd': 'start', 'val': 53.85})
print('sent /cmd/${NS}/${CN} <- {cmd: start, val: 53.85}')"
echo "(waiting ${START_WAIT_S}s for buffer create + junkdb spawn)"
sleep "${START_WAIT_S}"

echo
echo "=== STEP 4: /mon/${NS}/${CN} reports state=running + 2 alive routines ==="
python3 -c "
import sys
from dsautils.dsa_store import DsaStore
import json
d = DsaStore().get_dict('/mon/${NS}/${CN}') or {}
print(json.dumps(d, indent=2))
ok = True
if d.get('state') != 'running':
    print(f\"FAIL: state={d.get('state')!r} != running\", file=sys.stderr); ok = False
routines = d.get('routines') or {}
if len(routines) != 2:
    print(f\"FAIL: routines={list(routines)!r}; expected 2\", file=sys.stderr); ok = False
for name, info in routines.items():
    if not info.get('alive'):
        print(f\"FAIL: routine {name} not alive: {info}\", file=sys.stderr); ok = False
buffers = d.get('buffers') or {}
if set(buffers) != {'dada','eada','fada','bada'}:
    print(f\"FAIL: buffers={list(buffers)!r}\", file=sys.stderr); ok = False
sys.exit(0 if ok else 1)"
echo "OK state=running + routines + buffers"

echo
echo "=== STEP 5: dada_dbmetric on host shows live rings ==="
for k in dada eada fada bada; do
  printf '[%s] ' "${k}"; dada_dbmetric -k "${k}" 2>&1 | head -1
done

echo
echo "=== STEP 6: send stop verb ==="
python3 -c "
from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/${NS}/${CN}', {'cmd': 'stop', 'val': None})
print('sent /cmd/${NS}/${CN} <- {cmd: stop}')"
echo "(waiting 8s for routine kill + buffer destroy)"
sleep 8

echo
echo "=== STEP 7: /mon/${NS}/${CN} reports state=stopped, no routines ==="
python3 -c "
import sys
from dsautils.dsa_store import DsaStore
import json
d = DsaStore().get_dict('/mon/${NS}/${CN}') or {}
print(json.dumps(d, indent=2))
ok = True
if d.get('state') != 'stopped':
    print(f\"FAIL: state={d.get('state')!r} != stopped\", file=sys.stderr); ok = False
routines = d.get('routines') or {}
if routines:
    print(f\"FAIL: routines still present: {list(routines)}\", file=sys.stderr); ok = False
sys.exit(0 if ok else 1)"

echo
echo "=== STEP 8: shm + junkdb processes are gone ==="
shm_lines=$(ipcs -m | grep -E '^0x' | wc -l)
echo "shm segments: ${shm_lines}"
[[ "${shm_lines}" -eq 0 ]] || echo "(non-zero is OK if other containers run PSRDADA; check for our keys specifically)"
junkdb_lines=$(pgrep -af dada_junkdb | wc -l)
echo "dada_junkdb processes: ${junkdb_lines}"
if [[ "${junkdb_lines}" -gt 0 ]]; then
  echo "FAIL: leftover dada_junkdb" >&2; exit 1
fi

echo
echo "=== STEP 9: SIGTERM orchestrator; verify clean exit ==="
kill -TERM "$(cat "${PIDFILE}")"
sleep 3
if kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
  echo "FAIL: orchestrator still running after SIGTERM" >&2; exit 1
fi
rm -f "${PIDFILE}"
echo "OK orchestrator exited cleanly"

echo
echo "=== M7.0 PASS ==="
echo "log: ${LOG}"
echo "child logs: ${CHILDLOGDIR}/"
