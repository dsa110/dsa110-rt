#!/usr/bin/env bash
# _m74_synth_fada_fleet_launch.sh --- spurious-candidate noise run.
#
# 16x4 fleet launch in captures.mode=synth_fada (pure-Gaussian
# voltages written straight to fada by bench/replay_voltage_dump
# --synthesize). corr_fast then runs the SAME real gridder + staged
# dedispersion + transport-to-search as on-sky; search runs the real
# fine-DM dedisp + detector -> C1. The only difference vs on-sky is
# the INPUT is thermal noise (no sky, no RFI), so any C1 candidate is
# by construction a false alarm. Used to characterise the per-fine-DM
# false-alarm rate per GPU half (startup vs persistent) and confirm /
# refute the n01-half0 low-DM edge artifact on pure noise.
#
# Derived from _m75_phaseB_16x4_launch.sh, dropping the real-mode
# capture-shm freshness check + utc_start arming (STAGES 4-5): in
# synth_fada there is no SNAP capture binary and cap_synth_fada needs
# no arming -- it is the sole fada writer and provides the PSRDADA
# header corr_fast.getHeader() blocks on.
#
# Sequence:
#   0  Cleanup all 20 nodes (corr + search)
#   1  Push pipeline_rt + search_rt YAML to etcd (mode MUST be synth_fada)
#   2  Launch search_rt on 4 search nodes; start; warmup
#   3  Launch corr_rt on 16 corr nodes; start; warmup (corr_fast cold start)
#   4  Brief health soak; leave fleet UP for C1 collection
set -uo pipefail

CORR_NODES_CN=(3 4 5 6 7 8 10 11 12 14 15 16 18 19 21 22)
declare -A CN_TO_HOST=(
  [3]=n03 [4]=n04 [5]=n05 [6]=n06 [7]=n07 [8]=n08
  [10]=n10 [11]=n11 [12]=n12 [14]=n14 [15]=n15 [16]=n16
  [18]=n18 [19]=n19 [21]=n21 [22]=n22
)
SEARCH_NODES_SPECS=("n01:1" "n02:2" "n09:9" "n13:13")
OBS_DEC=${OBS_DEC:-53.85}
REPO=/home/ubuntu/proj/dsa110-rt
WARMUP_CORR_S=${WARMUP_CORR_S:-240}
WARMUP_SEARCH_S=${WARMUP_SEARCH_S:-60}
SOAK_S=${SOAK_S:-180}

echo "==============================================================="
echo " synth_fada 16x4 fleet launch -- pure-Gaussian noise run"
echo "   corr nodes (cn): ${CORR_NODES_CN[*]}"
echo "   search nodes:     ${SEARCH_NODES_SPECS[*]}"
echo "   warmups: search=${WARMUP_SEARCH_S}s corr=${WARMUP_CORR_S}s  soak=${SOAK_S}s"
echo "==============================================================="

# ---- STAGE 0 ---- cleanup --------------------------------------------------
echo
echo "=== STAGE 0: full fleet cleanup (parallel) ==="
bash /home/ubuntu/vikram/dev/dsa110-rt/tools/ops/_m72_16x4_cleanup.sh 2>&1 | tail -6

# ---- STAGE 1 ---- push YAML configs ---------------------------------------
echo
echo "=== STAGE 1: push pipeline_rt + search_rt YAML to etcd ==="
cd /home/ubuntu/vikram/dev/dsa110-rt
python3 tools/ops/push_dsart_to_etcd.py --instance all 2>&1 | tail -6
echo "--- verify pushed captures.mode ---"
/home/ubuntu/anaconda3/envs/casa38/bin/python3 - <<'PY' 2>&1 | tail -3 || true
from dsautils.dsa_store import DsaStore
m=((DsaStore().get_dict("/cnf/pipeline_rt") or {}).get("captures") or {}).get("mode")
print("pushed captures.mode =", m)
assert m == "synth_fada", f"REFUSING: captures.mode={m!r} != synth_fada"
print("OK synth_fada confirmed")
PY

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
  " 2>&1 | tail -1
}

send_verb_local() {
    local namespace=$1
    local cn=$2
    local verb=$3
    local val=$4
    /home/ubuntu/anaconda3/envs/casa38/bin/python3 -c "
from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/${namespace}/${cn}', {'cmd':'${verb}','val':${val}})
print('${namespace}/${cn} -> ${verb} val=${val}')
"
}

# ---- STAGE 2 ---- search nodes --------------------------------------------
echo
echo "=== STAGE 2a: launch search_rt orchestrators on 4 search nodes ==="
for sn in "${SEARCH_NODES_SPECS[@]}"; do
  h="${sn%%:*}"; cn="${sn##*:}"
  ( echo -n "$h(cn=$cn): "; launch_orchestrator "$h" search_rt "$cn" ) &
done
wait

echo
echo "=== STAGE 2b: send start to 4 search_rt; warmup ${WARMUP_SEARCH_S}s ==="
for sn in "${SEARCH_NODES_SPECS[@]}"; do
  cn="${sn##*:}"
  send_verb_local search_rt "$cn" start "None"
done
sleep "${WARMUP_SEARCH_S}"

# ---- STAGE 3 ---- corr nodes ----------------------------------------------
echo
echo "=== STAGE 3a: launch corr_rt orchestrators on 16 corr nodes ==="
for cn in "${CORR_NODES_CN[@]}"; do
  h="${CN_TO_HOST[$cn]}"
  ( echo -n "$h(cn=$cn): "; launch_orchestrator "$h" pipeline_rt "$cn" ) &
done
wait

echo
echo "=== STAGE 3b: send start to 16 corr_rt (broadcast key) ==="
send_verb_local corr_rt 0 start "${OBS_DEC}"

echo
echo "=== STAGE 3c: warmup wait ${WARMUP_CORR_S}s (corr_fast cold start) ==="
for i in $(seq 1 $((WARMUP_CORR_S / 30))); do
    sleep 30
    elapsed=$((i*30))
    # synth_fada has NO capture shm; gauge readiness via corr_fast ready
    # sentinels written right before corr_fast enters its main loop.
    ssh -o ConnectTimeout=5 -n n06.pro.pvt 'ls /tmp/dsart-corr-fast-*.ready 2>/dev/null | wc -l' \
        > /tmp/_synth_n06_ready.txt 2>/dev/null
    n06ready=$(cat /tmp/_synth_n06_ready.txt 2>/dev/null || echo 0)
    echo "    warmup t=${elapsed}s  n06_corr_fast_ready=${n06ready}"
done

# ---- STAGE 4 ---- brief health soak; leave fleet UP -----------------------
echo
echo "=== STAGE 4: ${SOAK_S}s health soak (fleet stays UP afterwards) ==="
SOAK_S="${SOAK_S}" /home/ubuntu/anaconda3/envs/casa38/bin/python3 - <<'PY'
import os, time
from dsautils.dsa_store import DsaStore
s = DsaStore()
T = int(os.environ["SOAK_S"]); INTERVAL = 30
CORRS = [3,4,5,6,7,8,10,11,12,14,15,16,18,19,21,22]
SEARCH = [1,2,9,13]
def gd(k):
    try: return s.get_dict(k) or {}
    except: return {}
t0 = time.monotonic()
while True:
    t = time.monotonic() - t0
    if t >= T: break
    time.sleep(INTERVAL)
    ca=ct=0
    for cn in CORRS:
        r=(gd(f"/mon/corr_rt/{cn}") or {}).get("routines") or {}
        ct+=len(r); ca+=sum(1 for st in r.values() if st.get("alive"))
    sa=stt=0
    for cn in SEARCH:
        r=(gd(f"/mon/search_rt/{cn}") or {}).get("routines") or {}
        stt+=len(r); sa+=sum(1 for st in r.values() if st.get("alive"))
    print(f"  t={t:5.0f}s  corr {ca}/{ct} alive  search {sa}/{stt} alive")
PY

echo
echo "=== synth_fada fleet launch complete; fleet left UP ==="
echo "Collect C1 from search nodes' rolling CSVs; run _m72_16x4_cleanup.sh to tear down."
