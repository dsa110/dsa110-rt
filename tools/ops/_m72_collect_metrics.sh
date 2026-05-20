#!/bin/bash
# Collect per-node performance metrics from the 16x1 fleet.
# Writes to /home/ubuntu/vikram/tmp/m72-16x1/<host>.txt (one file per node).
set -u

CORR_NODES="n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
SEARCH_NODES="n01"
OUTDIR="/home/ubuntu/vikram/tmp/m72-16x1"
mkdir -p "$OUTDIR"

collect_corr() {
  local host=$1
  ssh -o ConnectTimeout=5 -n "${host}.pro.pvt" "
    echo '==== host: $host ===='
    echo '==== dada/eada/fada/bada (full/nbufs) ===='
    for k in dada eada fada bada; do
      raw=\$(dada_dbmetric -k \$k 2>/dev/null | head -1)
      echo \"  \$k: \$raw\"
    done
    echo
    echo '==== corr_fast tail (last 30) ===='
    tail -30 /home/ubuntu/tmp/dsart-rt/corr_rt-*-corr_fast.log 2>/dev/null | grep -E '(processed|TransportTx|async_tx|summary|ERROR|WARNING)' | tail -20
    echo
    echo '==== corr_slow tail (last 30) ===='
    tail -30 /home/ubuntu/tmp/dsart-rt/corr_rt-*-corr_slow.log 2>/dev/null | grep -E '(processed|ERROR|WARNING|summary)' | tail -10
    echo
    echo '==== merge tail (last 20) ===='
    tail -20 /home/ubuntu/tmp/dsart-rt/corr_rt-*-merge.log 2>/dev/null | tail -10
    echo
    echo '==== nvidia-smi util/mem ===='
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
  "
}

collect_search() {
  local host=$1
  ssh -o ConnectTimeout=5 -n "${host}.pro.pvt" "
    echo '==== host: $host (SEARCH) ===='
    echo '==== search_rx tail (last 30) ===='
    tail -50 /home/ubuntu/tmp/dsart-rt/search_rt-*-search_rx.log 2>/dev/null | grep -E '(status|stats|n_received|n_committed|ERROR|WARNING)' | tail -20
    echo
    echo '==== search_compute_0 tail (last 30) ===='
    tail -50 /home/ubuntu/tmp/dsart-rt/search_rt-*-search_compute_0.log 2>/dev/null | grep -E '(cube|cubes/s|p50|p99|forward|build_cube|ERROR|WARNING)' | tail -20
    echo
    echo '==== search_compute_1 tail (last 30) ===='
    tail -50 /home/ubuntu/tmp/dsart-rt/search_rt-*-search_compute_1.log 2>/dev/null | grep -E '(cube|cubes/s|p50|p99|forward|build_cube|ERROR|WARNING)' | tail -20
    echo
    echo '==== nvidia-smi util/mem ===='
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
  "
}

echo '=== collecting in parallel ==='
for h in $CORR_NODES; do
  ( collect_corr "$h" > "$OUTDIR/$h.txt" 2>&1; echo "  $h done" ) &
done
for h in $SEARCH_NODES; do
  ( collect_search "$h" > "$OUTDIR/$h.txt" 2>&1; echo "  $h done" ) &
done
wait
echo '=== collection done ==='
echo "outputs in $OUTDIR"
ls -la "$OUTDIR/" | tail -20
