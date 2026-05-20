#!/bin/bash
# M7.3 16x4 fleet launcher.
# 16 corr nodes feed 4 search nodes; each search node owns 2 coarse DMs.
#
# Stages:
#   1. Launch search_rt orchestrators on all 4 search nodes (n01/n02/n09/n13).
#      Each picks up its per-host --coarse-dm-owners-half-{0,1} overrides
#      from dsart_search_rt.yaml hostargs (M7.3 amend, 2026-05-20).
#   2. Send `start` to all 4 search_rt; wait 30 s for ring init + GPU
#      JIT (search_compute halves do their own ~3-4 min cold start in
#      the background, but the rx side is up immediately).
#   3. Launch corr_rt orchestrators on all 16 corr nodes (parallel).
#   4. Send `start` to 16 corrs. Each corr_fast spawns 4 async-tx
#      workers; each worker targets a DIFFERENT search node per the
#      --transport-tx-worker-hosts list in dsart_pipeline_rt.yaml.
#      Coarse-DM mask is 0xFF so all 4 workers transmit (DMs 0-1 to
#      n01, 2-3 to n02, 4-5 to n09, 6-7 to n13).
#   5. Wait 300 s for full warm-up (the corr_fast cold start is
#      ~5 min including DM-plan + Triton + cal load).
set -u

CORR_NODES_CN=(3 4 5 6 7 8 10 11 12 14 15 16 18 19 21 22)
# Each (cn -> n<NN>) on the search side; cn matches the YAML routine
# substitutions but the operator picks them to be unique per search
# node.
SEARCH_NODES=("n01:1" "n02:2" "n09:9" "n13:13")
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

echo "=== STAGE 1: launch search_rt on ALL 4 search nodes (parallel) ==="
for sn in "${SEARCH_NODES[@]}"; do
  host="${sn%%:*}"
  cn="${sn##*:}"
  ( echo -n "$host(cn=$cn): "; launch_orchestrator "$host" search_rt "$cn" ) &
done
wait

echo
echo "=== STAGE 2: send start to 4 search_rt; warmup 30 s ==="
for sn in "${SEARCH_NODES[@]}"; do
  host="${sn%%:*}"
  cn="${sn##*:}"
  ( echo -n "$host(cn=$cn): "; send_start "$host" search_rt "$cn" "None" ) &
done
wait
sleep 30

echo
echo "=== STAGE 3: launch corr_rt orchestrators on 16 corr nodes (parallel) ==="
for cn in "${CORR_NODES_CN[@]}"; do
  h=$(CN_TO_HOST "$cn")
  ( echo -n "$h(cn=$cn): "; launch_orchestrator "$h" pipeline_rt "$cn" ) &
done
wait

echo
echo "=== STAGE 4: send start to 16 corrs (parallel) ==="
for cn in "${CORR_NODES_CN[@]}"; do
  h=$(CN_TO_HOST "$cn")
  ( echo -n "$h(cn=$cn): "; send_start "$h" corr_rt "$cn" "${OBS_DEC}" ) &
done
wait

echo
echo "=== STAGE 5: warmup wait — 300 s for corr_fast JIT + sentinel + junkdb spawn ==="
sleep 300
echo "wait done; fleet should now be in steady state."
