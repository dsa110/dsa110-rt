#!/bin/bash
# Re-run every Phase-0 study script that depends on the value maps (CPU, nice 10, one process at a
# time, single-threaded BLAS).  Usage: bash run_all.sh [script ...]   (default: the list below)
set -eo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/ubuntu/anaconda3/envs/dsart_h23/bin/python
SCRIPTS=("$@")
if [ ${#SCRIPTS[@]} -eq 0 ]; then
  SCRIPTS=(valuemaps rr_static rr_hot rr_load rr_fold amp_regime rr_track rr_bright cfp16 img_response img_noise)
fi
for s in "${SCRIPTS[@]}"; do
  echo "== $s $(date -u +%H:%M:%S)"
  nice -n 10 "$PY" "$s.py" > "/tmp/phase0_$s.log" 2>&1 || { echo "FAILED $s"; tail -20 "/tmp/phase0_$s.log"; exit 1; }
  tail -2 "/tmp/phase0_$s.log"
done
