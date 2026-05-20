#!/bin/bash
# Baseline search-node bench at production geometry, full M7.2 op-point.
# Run on n01 only (needs torch + CUDA).
# NOTE: NOT using `set -u` because conda activate's libblas_mkl_activate.sh
# touches unbound MKL_INTERFACE_LAYER under `set -u`.
set -e
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate dsa110-rt
cd /home/ubuntu/proj/dsa110-rt

OUT=${OUT:-/tmp/bench_baseline}
T_DET=${T_DET:-256}
EXTRA=${EXTRA:-"--pipeline-overlap"}
mkdir -p "$OUT"

ls -la src/dsart/transport/_recv_*.so | head -3
echo
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | head -3
echo

N_CUBES=${N_CUBES:-40}
PYTHONPATH=src DSART_TEST=1 CUDA_VISIBLE_DEVICES=0 python -u -m bench.search_node_throughput \
    --n-cubes $N_CUBES \
    --t-det $T_DET --n-fdm 34 --n-grid 256 \
    --device cuda --cube-dtype fp16 \
    --image-backend gpu \
    --bank-mask 'k_img=unit;k_dm=d1;k_time=b1,b2,b4,b8,b16,b32,b64' \
    --detector-streaming \
    --detector-streaming-tile-size 256 \
    --detector-streaming-decoder-n-top 24 \
    --detector-boxcar-accum-dtype fp16 \
    --detector-layer2-max-samples ${L2_MAX:-100000} \
    --layer1-max-samples ${L1_MAX:-1000000} \
    --prequantise $EXTRA \
    --out "$OUT" 2>&1 | tail -50
