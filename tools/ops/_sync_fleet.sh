#!/bin/bash
# Sync the local workspace to all 16 corr nodes + n01 (search node).
# Runs rsync in parallel; logs each node's last 5 lines.
set -u

SRC_DIR="/home/ubuntu/vikram/dev/dsa110-rt/"
DEST_PATH="/home/ubuntu/proj/dsa110-rt/"

NODES="n01 n02 n03 n04 n05 n06 n07 n08 n09 n10 n11 n12 n13 n14 n15 n16 n18 n19 n21 n22"

LOG_DIR="/tmp/m72-sync"
mkdir -p "$LOG_DIR"

sync_one() {
  local host=$1
  # IMPORTANT: We exclude *.so so we don't wipe the C extensions
  # built locally on the target node (recv_ring + recv_epoll cpython-*.so).
  # The .o intermediates are excluded for the same reason. Cleaning up
  # stale Python-2/3.8 artifacts is the responsibility of the target's
  # build step.
  rsync -av --delete \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.so' \
    --exclude '*.o' \
    --exclude '.pytest_cache/' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '*.egg-info/' \
    --exclude '/tmp_artifacts/' \
    "$SRC_DIR" "${host}.pro.pvt:${DEST_PATH}" \
    > "$LOG_DIR/${host}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$host: OK ($(grep -c '^[^ ]' $LOG_DIR/${host}.log) lines)"
  else
    echo "$host: FAIL (rc=$rc)"
    tail -5 "$LOG_DIR/${host}.log" | sed "s|^|  $host| |"
  fi
}

echo "=== syncing to ${NODES} ==="
for h in $NODES; do
  sync_one "$h" &
done
wait

# After syncing pure-Python sources, rebuild the C extensions on every
# search node so the .so files match the recv_ring.h / recv_epoll.c
# layouts we just shipped. The corr nodes don't use _recv_epoll on the
# read side (they don't run search_rx), but the build is cheap so we
# do every node for symmetry. Build runs in parallel.
# 2026-05-20: the n02/n09/n13 search-node bootstrap blew up because the
# .so files were >5 days stale (no recv_epoll_add_port symbol) -- this
# step prevents recurrence.
echo "=== rebuilding C extensions on every node (parallel) ==="
build_one() {
  local host=$1
  ssh -o ConnectTimeout=5 "${host}.pro.pvt" "
    source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
    conda activate dsa110-rt 2>/dev/null
    cd ${DEST_PATH}
    python setup.py build_ext --inplace > /tmp/m72-build.log 2>&1
    if [ \$? -eq 0 ]; then
      echo '${host}: build OK'
    else
      echo '${host}: build FAIL'
      tail -5 /tmp/m72-build.log
    fi
  " 2>&1 | tail -3
}
for h in $NODES; do
  build_one "$h" &
done
wait
echo "=== done ==="
