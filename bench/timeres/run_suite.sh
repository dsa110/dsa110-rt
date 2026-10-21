#!/bin/bash
# Run the GPU part of the time-resolution bench suite on ONE node, in an
# observing gap.  Total ~10 min per GPU.  Usage (on the node):
#   rsync -a <worktree>/bench/timeres/ ~/timeres_bench_src/      # from h23
#   ssh nNN; cd ~/timeres_bench_src && bash run_suite.sh [gpu ...]
# Default GPUs: 0.  Results: ~/timeres_bench/<UTC>/results.jsonl + summary.txt
set -eo pipefail
export MKL_INTERFACE_LAYER=GNU,LP64
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dsa110-rt
export PYTHONPATH="$HOME/proj/dsa110-rt/src:$PYTHONPATH"   # production triton_boxcar
cd "$(dirname "$0")"
GPUS=${*:-0}
OUTDIR=~/timeres_bench/$(date -u +%Y%m%dT%H%MZ)_$(hostname -s)
mkdir -p "$OUTDIR"
OUT=$OUTDIR/results.jsonl
nvidia-smi --query-gpu=index,name,clocks.max.sm,temperature.gpu --format=csv > "$OUTDIR/gpus.csv"
for g in $GPUS; do
  D=cuda:$g
  echo "=== GPU $g: b0 selftest"
  if python b0_selftest.py --device "$D" > "$OUTDIR/b0_gpu$g.log" 2>&1; then
    echo "  kernels OK"; KOK=1
  else
    echo "  prototype kernels FAILED selftest (see b0_gpu$g.log) - chunk/e2e cases will error"; KOK=0
  fi
  python b1_fft.py      --device "$D" --out "$OUT"
  python b2_detector.py --device "$D" --out "$OUT"
  python b3_owner_e2e.py --device "$D" --out "$OUT"
  python b4_xengine.py  --device "$D" --out "$OUT"
done
python summarize.py "$OUT" | tee "$OUTDIR/summary.txt"
echo "results in $OUTDIR"
