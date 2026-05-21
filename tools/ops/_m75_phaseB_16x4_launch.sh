#!/usr/bin/env bash
# _m75_phaseB_16x4_launch.sh --- M7.5 Phase B.
#
# 16x4 fleet launch in captures.mode=real (live SNAPs). Builds on
# the M7.3 16x4 launcher but adds (a) explicit YAML push to etcd
# before any orchestrators come up so they consume the real-mode
# routine set rather than the previously-pushed junkdb set, and
# (b) a utc_start broadcast against /cmd/corr_rt/0 once every
# capture's shm has a fresh last_seq_no (the orchestrator's UTC
# poke replaces my Phase-A per-node fan-out).
#
# Sequence:
#   0  Cleanup all 20 nodes (corr + search)
#   1  Push pipeline_rt + search_rt YAML to etcd
#   2  Launch search_rt on 4 search nodes; start verb; 60 s warmup
#      (search_rx + search_compute_{0,1} cold start)
#   3  Launch corr_rt on 16 corr nodes; start verb; 240 s warmup
#      (corr_fast JIT + DM-plan + cal + Triton; 90-180 s typ.)
#   4  Verify every capture shm has fresh last_seq_no (no stale 0)
#   5  Pick ARM_SEQ = max(last_seq_no across all 32 capture shms) +
#      60 k specnums (~4 s ahead); broadcast utc_start to corr_rt/0
#   6  Soak; collect mon-dict snapshots from corrs + search nodes
#   7  Final summary + stop everywhere
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
SOAK_S=${SOAK_S:-600}
SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-30}

echo "==============================================================="
echo " M7.5 Phase B: 16x4 fleet launch -- captures.mode=real"
echo "   corr nodes (cn): ${CORR_NODES_CN[*]}"
echo "   search nodes:     ${SEARCH_NODES_SPECS[*]}"
echo "   warmups: search=${WARMUP_SEARCH_S}s corr=${WARMUP_CORR_S}s   soak=${SOAK_S}s"
echo "==============================================================="

# ---- STAGE 0 ---- cleanup --------------------------------------------------
echo
echo "=== STAGE 0: full fleet cleanup (parallel) ==="
bash /home/ubuntu/vikram/dev/dsa110-rt/tools/ops/_m72_16x4_cleanup.sh 2>&1 | tail -6

# ---- STAGE 1 ---- push YAML configs ---------------------------------------
echo
echo "=== STAGE 1: push pipeline_rt + search_rt YAML to etcd ==="
cd /home/ubuntu/vikram/dev/dsa110-rt
python3 tools/ops/push_dsart_to_etcd.py --instance all 2>&1 | tail -4

# helper functions reused from _m72_16x4_launch.sh
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
    # send a single command to a single (namespace,cn) pair from this host
    local namespace=$1
    local cn=$2
    local verb=$3
    local val=$4    # bash literal that becomes a Python expression (int, str, None)
    python3 -c "
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
send_verb_local corr_rt 0 start "${OBS_DEC}"   # /cmd/corr_rt/0 broadcasts to all

echo
echo "=== STAGE 3c: warmup wait ${WARMUP_CORR_S}s (corr_fast cold start) ==="
for i in $(seq 1 $((WARMUP_CORR_S / 30))); do
    sleep 30
    elapsed=$((i*30))
    ssh -o ConnectTimeout=5 -n n06.pro.pvt 'ls /dev/shm/dsart-capture-* 2>/dev/null | wc -l' \
        > /tmp/_m75_n06_shm.txt 2>/dev/null
    n06shm=$(cat /tmp/_m75_n06_shm.txt 2>/dev/null || echo 0)
    echo "    warmup t=${elapsed}s  n06_shm=${n06shm}/2"
done

# ---- STAGE 4 ---- verify capture shms have fresh last_seq_no --------------
echo
echo "=== STAGE 4: verify capture shms across 16 corr nodes ==="
PARALLEL_REPORT=/tmp/_m75_fresh_check.txt
> "$PARALLEL_REPORT"
for cn in "${CORR_NODES_CN[@]}"; do
  h="${CN_TO_HOST[$cn]}"
  ( ssh -o ConnectTimeout=10 -n "${h}.pro.pvt" "
    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh 2>/dev/null
    conda activate dsa110-rt 2>/dev/null
    export PYTHONPATH=${REPO}/src
    python3 - <<PY 2>/dev/null
from dsart.capture import mon_shm
out = []
for p in (4011, 4012):
    try:
        with mon_shm.MonShm.open(p) as m:
            s = m.snapshot()
            out.append(f'{p}:{s.last_seq_no}:{s.n_recv_packets}')
    except Exception as e:
        out.append(f'{p}:ERR:{type(e).__name__}')
print('${h}|cn=${cn}|' + ' '.join(out))
PY
" 2>/dev/null
  ) >> "$PARALLEL_REPORT" &
done
wait
cat "$PARALLEL_REPORT" | sort

# ---- STAGE 5 ---- compute + broadcast ARM_SEQ -----------------------------
echo
echo "=== STAGE 5: compute ARM_SEQ + broadcast utc_start ==="
# pick max last_seq_no across all 32 capture shms + safety margin
ARM_SEQ=$(awk -F'|' '{
    n = split($3, kv, " ")
    for (i = 1; i <= n; i++) {
        split(kv[i], parts, ":")
        v = parts[2]
        if (v ~ /^[0-9]+$/ && v + 0 > maxv) maxv = v + 0
    }
}
END { print maxv + 60000 }' "$PARALLEL_REPORT")
echo "    ARM_SEQ=${ARM_SEQ}"
if [ -z "$ARM_SEQ" ] || [ "$ARM_SEQ" = "60000" ]; then
    echo "FAIL: no fresh last_seq across the fleet; aborting"
    exit 1
fi
send_verb_local corr_rt 0 utc_start "${ARM_SEQ}"

# ---- STAGE 6 ---- soak ----------------------------------------------------
echo
echo "=== STAGE 6: ${SOAK_S}s soak ==="
python3 - <<PY
import json, time
from dsautils.dsa_store import DsaStore

s = DsaStore()
T = ${SOAK_S}
INTERVAL = ${SAMPLE_INTERVAL_S}
CORRS = [3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22]
SEARCH = [1, 2, 9, 13]   # n01, n02, n09, n13 -> cn ids

def gd(k):
    try: return s.get_dict(k) or {}
    except: return {}

t0 = time.monotonic()
while True:
    t = time.monotonic() - t0
    if t >= T: break
    time.sleep(INTERVAL)
    print(f'--- t={t:5.0f}s ---')

    # corr-side roll-up
    cap_blocks_min, cap_blocks_max = None, None
    cap_drops = 0
    cap_armed = 0
    cap_total = 0
    for cn in CORRS:
        for port in (4011, 4012):
            cap = gd(f'/mon/corr_rt/{cn}/capture/{port}')
            if not cap: continue
            cap_total += 1
            if cap.get('arm_state') == 'WRITING': cap_armed += 1
            b = cap.get('n_block_writes', -1)
            if isinstance(b, int) and b >= 0:
                cap_blocks_min = b if cap_blocks_min is None else min(cap_blocks_min, b)
                cap_blocks_max = b if cap_blocks_max is None else max(cap_blocks_max, b)
            cap_drops += cap.get('n_dropped_kernel', 0) or 0
    print(f'  captures: {cap_armed}/{cap_total} armed  blocks=[{cap_blocks_min}, {cap_blocks_max}]  kernel_drops_total={cap_drops}')

    # corr_fast roll-up via /mon/corr_rt/<cn>/corr_fast/state if exposed; otherwise look at the
    # main /mon/corr_rt/<cn> dict for routine health.
    corr_fast_lat = []
    n_alive_routines = 0
    n_routines_total = 0
    for cn in CORRS:
        d = gd(f'/mon/corr_rt/{cn}')
        for nm, st in (d.get('routines') or {}).items():
            n_routines_total += 1
            if st.get('alive'): n_alive_routines += 1
    print(f'  corr routines: {n_alive_routines}/{n_routines_total} alive')

    # search-side roll-up
    rx_pkts, rx_drops = 0, 0
    sc_blocks = []
    for cn in SEARCH:
        d = gd(f'/mon/search_rt/{cn}')
        for nm, st in (d.get('routines') or {}).items():
            if not st.get('alive') and nm.startswith('search_'):
                print(f"    WARN: {cn} routine {nm} DEAD")
PY

# ---- STAGE 7 ---- final snapshot via _m72_snapshot.py + summary -----------
echo
echo "=== STAGE 7: final fleet snapshot ==="
cd /home/ubuntu/vikram/dev/dsa110-rt
python3 tools/ops/_m72_snapshot.py 2>&1 | head -200

echo
echo "=== M7.5 Phase B launch complete ==="
echo "Soak is over. Run tools/ops/_m72_16x4_cleanup.sh to tear down."
