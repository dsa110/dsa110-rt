#!/bin/bash
# M7.2 16x1 fleet launcher.
# Stages:
#   1. Launch search_rt orchestrator on n01 (cn=1); send `start`.
#   2. Wait 30 s for search-rx + search_compute warmup.
#   3. Launch corr_rt orchestrators on all 16 corr nodes; send `start`.
#   4. Wait 180 s for corr_fast/slow warmup (sentinel) + junkdb start +
#      steady state.
# After this script returns, snapshot+soak is driven by the sibling
# _m72_16x1_snapshot.sh / _m72_16x1_stop.sh scripts.
set -u

CORR_NODES_CN=(3 4 5 6 7 8 10 11 12 14 15 16 18 19 21 22)
SEARCH_NODE="n01"
SEARCH_CN=1
OBS_DEC=53.85

REPO=/home/ubuntu/proj/dsa110-rt

launch_orchestrator() {
  local host=$1
  local instance=$2
  local cn=$3
  ssh -o ConnectTimeout=5 -n "${host}.pro.pvt" "
    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
    conda activate dsa110-rt 2>/dev/null
    cd ${REPO}
    export PYTHONPATH=${REPO}/src
    export DSART_RT_LOG_DIR=/tmp/dsart-rt-children
    export DSART_RT_GATE_TIMEOUT_S=300
    setsid nohup python3 -u -m dsart.services.dsart_rt -in ${instance} -cn ${cn} --log-level INFO \
        > /tmp/dsart-rt-${instance}-${cn}.log 2>&1 < /dev/null &
    echo \$! > /tmp/dsart-rt-${instance}-${cn}.pid
    disown
    sleep 2
    if kill -0 \$(cat /tmp/dsart-rt-${instance}-${cn}.pid) 2>/dev/null; then
      echo alive
    else
      echo DEAD
      tail -10 /tmp/dsart-rt-${instance}-${cn}.log
    fi
  " 2>&1 | tail -3
}

send_start() {
  local host=$1
  local instance=$2
  local cn=$3
  local val=$4
  ssh -o ConnectTimeout=5 -n "${host}.pro.pvt" "
    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
    conda activate dsa110-rt 2>/dev/null
    python3 -c \"from dsautils.dsa_store import DsaStore; DsaStore().put_dict('/cmd/${instance}/${cn}', {'cmd':'start','val':${val}})\" 2>&1 | tail -3
    echo SENT_${cn}
  " 2>&1 | tail -2
}

CN_TO_HOST() {
  case "$1" in
    3) echo n03;; 4) echo n04;; 5) echo n05;; 6) echo n06;;
    7) echo n07;; 8) echo n08;; 10) echo n10;; 11) echo n11;;
    12) echo n12;; 14) echo n14;; 15) echo n15;; 16) echo n16;;
    18) echo n18;; 19) echo n19;; 21) echo n21;; 22) echo n22;;
  esac
}

echo "=== STAGE 1: launch search_rt on ${SEARCH_NODE} (cn=${SEARCH_CN}) ==="
echo -n "${SEARCH_NODE}: "
launch_orchestrator "${SEARCH_NODE}" search_rt "${SEARCH_CN}"

echo
echo "=== STAGE 2: send `start` to search_rt; warmup 30 s ==="
echo -n "${SEARCH_NODE}: "
send_start "${SEARCH_NODE}" search_rt "${SEARCH_CN}" "None"
sleep 30

echo
echo "=== STAGE 3: launch corr_rt orchestrators on 16 corr nodes (parallel) ==="
for cn in "${CORR_NODES_CN[@]}"; do
  h=$(CN_TO_HOST "$cn")
  ( echo -n "$h(cn=$cn): "; launch_orchestrator "$h" pipeline_rt "$cn" ) &
done
wait

echo
echo "=== STAGE 4: send `start` to 16 corrs (parallel) ==="
for cn in "${CORR_NODES_CN[@]}"; do
  h=$(CN_TO_HOST "$cn")
  ( echo -n "$h(cn=$cn): "; send_start "$h" corr_rt "$cn" "${OBS_DEC}" ) &
done
wait

echo
echo "=== STAGE 5: warmup wait — 180 s for corr_fast JIT + sentinel + junkdb spawn ==="
sleep 180
echo "wait done; fleet should now be in steady state."
