#!/bin/bash
# Pre-test cleanup on all 17 nodes: kill stale dsart-rt processes, drop
# orphan PSRDADA buffers + POSIX-shm rings, clear /tmp sentinels.
set -u

CORR_NODES="n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
SEARCH_NODES="n01"

cleanup_corr() {
  local host=$1
  ssh -o ConnectTimeout=10 -n "${host}.pro.pvt" "
    # M7.2-amend (2026-05-20): PID-based kill instead of pkill -f.
    # In practice pkill -f sometimes fails to match python -u -m
    # dsart.services.* processes on heavily-loaded nodes; iterating
    # ps -eo pid,cmd and kill-by-PID is rock-solid.
    for round in 1 2; do
      for p in \$(ps -eo pid,cmd | grep -E 'dsart|corr_fast|corr_slow|dada_drain|dada_junkdb|dsaX_merge' | grep -v grep | awk '{print \$1}'); do
        kill -9 \$p 2>/dev/null
      done
      sleep 2
    done
    # PSRDADA buffers + System V SHM cleanup (the dada_db -d call
    # plus an ipcrm sweep for any orphan ubuntu-owned SHM segments
    # that lingered after a hard kill).
    for k in dada eada fada bada gada hada; do dada_db -d -k \$k >/dev/null 2>&1; done
    ipcs -m | awk '/ubuntu/ {print \$2}' | xargs -r -I{} ipcrm -m {} 2>/dev/null
    rm -f /tmp/dsart-corr-*.ready
    rm -rf /tmp/dsart-rt-children
    mkdir -p /tmp/dsart-rt-children
    echo OK
  " 2>&1 | tail -1
}

cleanup_search() {
  local host=$1
  ssh -o ConnectTimeout=10 -n "${host}.pro.pvt" "
    for round in 1 2; do
      for p in \$(ps -eo pid,cmd | grep -E 'dsart|search_compute|search_rx' | grep -v grep | awk '{print \$1}'); do
        kill -9 \$p 2>/dev/null
      done
      sleep 2
    done
    rm -f /dev/shm/dsart-rxring-* /dev/shm/dsart-* 2>/dev/null
    rm -rf /tmp/dsart-rt-children
    mkdir -p /tmp/dsart-rt-children
    echo OK
  " 2>&1 | tail -1
}

echo '=== cleanup (parallel) ==='
for h in $CORR_NODES; do
  echo -n "$h: "; cleanup_corr "$h" &
done
for h in $SEARCH_NODES; do
  echo -n "$h: "; cleanup_search "$h" &
done
wait
echo '=== cleanup done ==='
