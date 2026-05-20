#!/bin/bash
# Aggregate per-corr-node steady-state stats (n_in, last_block, n_drop, n_tx,
# wire_drops, pacer_qd, corr_slow ms) into a single table.
set -u

CORR_NODES="n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"

per_node() {
  local h=$1
  ssh -o ConnectTimeout=5 -n "${h}.pro.pvt" "
    cf_last=\$(tail -200 /home/ubuntu/tmp/dsart-rt/corr_rt-*-corr_fast.log 2>/dev/null | grep -E '^[0-9].*processed n_in' | tail -1)
    cs_last=\$(tail -200 /home/ubuntu/tmp/dsart-rt/corr_rt-*-corr_slow.log 2>/dev/null | grep -E '^[0-9].*processed n_in' | tail -1)
    tx_last=\$(tail -200 /home/ubuntu/tmp/dsart-rt/corr_rt-*-corr_fast.log 2>/dev/null | grep -E 'n_cubes=.*n_frames' | tail -1)
    util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '/' | sed 's|/$||')
    echo \"$h | cf_last_block=\$(echo \$cf_last | grep -oE 'last_block=[0-9.]+ms' | head -1) | cf_n_in=\$(echo \$cf_last | grep -oE 'n_in=[0-9]+' | head -1) | cf_n_drop=\$(echo \$cf_last | grep -oE 'n_drop=[0-9]+' | head -1) | cf_n_tx=\$(echo \$cf_last | grep -oE 'n_tx=[0-9]+' | head -1) | cs_last_block=\$(echo \$cs_last | grep -oE 'last_block=[0-9.]+ms' | head -1) | tx=\$(echo \$tx_last | grep -oE '(wire_drops|pacer_qd|n_frames|n_cubes|max)=[0-9.]+' | tr '\n' ',') | gpu=\$util\"
  " 2>&1 | tail -1
}

echo '=== per-corr-node steady-state stats ==='
for h in $CORR_NODES; do
  per_node "$h"
done
