#!/bin/bash
# Probe each corr node for: ssh access, proj dir, cal bundle, dm plan.
# Used by the M7.2 16x1 fleet-readiness check.
set -u

NODES="n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"

probe_one() {
  local host=$1
  local sb=$2
  local cal="/home/ubuntu/data/voltages/250924mptq/cals/beamformer_weights_${sb}.dat"
  local flag="/home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat"
  local dmp="/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -n "${host}.pro.pvt" "
    out=ok
    [ -d /home/ubuntu/proj/dsa110-rt ] || out=\"\$out|no-proj\"
    [ -f $cal ] || out=\"\$out|no-cal-$sb\"
    [ -f $flag ] || out=\"\$out|no-flagants\"
    [ -f $dmp ] || out=\"\$out|no-dmplan\"
    [ -f /usr/local/bin/dada_junkdb ] || out=\"\$out|no-dada_junkdb\"
    [ -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_merge ] || out=\"\$out|no-dsaX_merge\"
    [ -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt ] || out=\"\$out|no-header\"
    [ -d /home/ubuntu/miniforge3/envs/dsa110-rt ] || out=\"\$out|no-conda-env\"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ',' | sed 's/,$//' | xargs -I{} echo \"\$out|gpus={}\"
  " 2>/dev/null || echo "ssh-FAIL"
}

declare -A CN_TO_SB=(
  [n03]=sb00 [n04]=sb01 [n05]=sb02 [n06]=sb03 [n07]=sb04 [n08]=sb05
  [n10]=sb06 [n11]=sb07 [n12]=sb08 [n14]=sb09 [n15]=sb10 [n16]=sb11
  [n18]=sb12 [n19]=sb13 [n21]=sb14 [n22]=sb15
)

echo "host    | sb    | status"
echo "--------+-------+--------------------------------------------------"
for h in $NODES; do
  sb=${CN_TO_SB[$h]}
  s=$(probe_one "$h" "$sb")
  printf "%-7s | %-5s | %s\n" "$h" "$sb" "$s"
done
