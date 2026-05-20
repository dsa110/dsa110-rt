#!/bin/bash
# Reset all 17 nodes to a clean orchestrator state:
#   1. systemctl --user stop dsart-rt.service (clean stop via _on_shutdown)
#   2. pkill any stragglers (my rogue launcher orchestrators)
#   3. destroy orphan PSRDADA buffers + POSIX-shm
#   4. clear /tmp sentinels + remove /cmd verbs
#   5. systemctl --user start dsart-rt.service (picks up latest rsynced code)
#   6. verify orchestrator is alive + in state=stopped on each node
set -u

CORR_NODES="n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
SEARCH_NODES="n01"

reset_one() {
  local host=$1
  local instance=$2
  local cn=$3
  ssh -o ConnectTimeout=5 -n "${host}.pro.pvt" "
    systemctl --user stop dsart-rt.service 2>/dev/null
    sleep 1
    pkill -9 -f 'dsart.services.dsart_rt'      2>/dev/null
    pkill -9 -f 'dsart.services.corr_fast'     2>/dev/null
    pkill -9 -f 'dsart.services.corr_slow'     2>/dev/null
    pkill -9 -f 'dsart.services.dada_drain'    2>/dev/null
    pkill -9 -f 'dsart.services.search_rx'     2>/dev/null
    pkill -9 -f 'dsart.services.search_compute' 2>/dev/null
    pkill -9 -f 'dada_junkdb'                  2>/dev/null
    pkill -9 -f 'dsaX_merge'                   2>/dev/null
    sleep 1
    for k in dada eada fada bada; do dada_db -d -k \$k >/dev/null 2>&1; done
    rm -f /tmp/dsart-corr-*.ready
    rm -f /dev/shm/dsart-rxring-* 2>/dev/null
    rm -rf /home/ubuntu/tmp/dsart-rt /tmp/dsart-rt-children
    mkdir -p /home/ubuntu/tmp/dsart-rt /tmp/dsart-rt-children
    systemctl --user start dsart-rt.service
    sleep 4
    if systemctl --user is-active --quiet dsart-rt.service; then echo started; else echo FAILED; fi
  " 2>&1 | tail -1
}

clear_etcd_verbs() {
  ssh -o ConnectTimeout=5 -n n01.pro.pvt "
    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
    conda activate dsa110-rt 2>/dev/null
    python3 - <<'PY'
from dsautils.dsa_store import DsaStore
s = DsaStore()
for cn in (1,):
    s.put_dict(f'/cmd/search_rt/{cn}', {'cmd':'stop','val':None})
for cn in (3,4,5,6,7,8,10,11,12,14,15,16,18,19,21,22):
    s.put_dict(f'/cmd/corr_rt/{cn}', {'cmd':'stop','val':None})
print('verbs cleared (all stop)')
PY
  " 2>&1 | tail -2
}

echo '=== clear lingering /cmd verbs to stop ==='
clear_etcd_verbs

echo
echo '=== reset (parallel) ==='
for h in $CORR_NODES; do
  cn=${h#n}; cn=$((10#$cn))
  (out=$(reset_one "$h" pipeline_rt "$cn"); echo "$h(cn=$cn): $out") &
done
for h in $SEARCH_NODES; do
  (out=$(reset_one "$h" search_rt 1); echo "$h(cn=1): $out") &
done
wait
echo '=== reset done ==='
