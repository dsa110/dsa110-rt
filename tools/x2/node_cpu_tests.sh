#!/bin/bash
# x2 CPU test harness: run dsart tests on a SEARCH NODE (CLAUDE.md: never on h23), CPU only, at
# the lowest priority, while the node keeps observing.
#
#   tools/x2/node_cpu_tests.sh [-n NODE] [-t TAG] [-b] [-g GPUS] [-e KEY=VALUE]... -- tests/x2/test_a.py [tests/...]
#
# - scratch copy of the node's own checkout (inherits its cpython-311 C extensions), then this
#   worktree's src/ tests/ tools/ configs/ bench/ docs/ overlaid on it (never the production tree);
# - -b rebuilds C extensions in the scratch copy (setup.py build_ext --inplace);
# - every file runs separately: nice 19, ionice idle, one core, single-threaded BLAS, no GPU
#   (CUDA_VISIBLE_DEVICES=""), `-m "not x2_gpu"`, shell timeout (no pytest-timeout on nodes);
# - aborts if `import dsart` does not resolve to the scratch copy;
# - every build/test runs under tools/x2/prod_guard.sh, which kills it if any search half on the node
#   starts to lag (2026-10-22: an mmap reader of the RX ring cost 41.5 s of production data on n13);
#   a trip stops the run (exit 4).
# - -g GPUS (e.g. -g 0,1): GPU mode for a node with production STOPPED (rev 5, observing off): the guard
#   refuses unless no production process runs and kills the run if one starts; CUDA_VISIBLE_DEVICES=GPUS,
#   x2_gpu tests included, no single-core pin, nice 5, 3600 s timeout per file.
# - -e KEY=VALUE (repeatable): extra environment for the tests (e.g. DSART_X2_I1_VOLTAGES=...).
set -eo pipefail
NODE=n13.pro.pvt
TAG=$(git rev-parse --abbrev-ref HEAD | tr '/' '_')
BUILD=0
GPUS=
XENV=
while getopts "n:t:bg:e:" o; do
  case $o in n) NODE=$OPTARG ;; t) TAG=$OPTARG ;; b) BUILD=1 ;; g) GPUS=$OPTARG ;; e) XENV="$XENV $OPTARG" ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
[ "$1" = "--" ] && shift
[ $# -ge 1 ] || { echo "usage: $0 [-n node] [-t tag] [-b] -- test files..."; exit 2; }
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python
SCR=/home/ubuntu/x2_scratch/$TAG
LOW="nice -n 19 ionice -c3"
ENVS="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_INTERFACE_LAYER=GNU,LP64 PYTHONPATH=$SCR/src DSART_CONFIG_DIR=$SCR/configs PYTHONDONTWRITEBYTECODE=1"
CORE=$(ssh -o BatchMode=yes "$NODE" nproc)
CORE=$((CORE - 1))
PIN="taskset -c $CORE"
MARK="-m 'not x2_gpu'"
TMO=900
GUARD="bash tools/x2/prod_guard.sh"
if [ -n "$GPUS" ]; then
  LOW="nice -n 5"
  ENVS="CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_INTERFACE_LAYER=GNU,LP64 PYTHONPATH=$SCR/src DSART_CONFIG_DIR=$SCR/configs PYTHONDONTWRITEBYTECODE=1"
  PIN=
  MARK=
  TMO=3600
  GUARD="env GUARD_REQUIRE_STOPPED=1 bash tools/x2/prod_guard.sh"
fi
ENVS="$ENVS$XENV"

ssh -o BatchMode=yes "$NODE" "test -d $SCR || { mkdir -p $(dirname $SCR) && $LOW cp -a /home/ubuntu/proj/dsa110-rt $SCR; }"
rsync -a --exclude='*.so' --exclude='__pycache__' --exclude='*.pdf' --exclude='*.png' \
  "$ROOT/src" "$ROOT/tests" "$ROOT/tools" "$ROOT/configs" "$ROOT/bench" "$ROOT/docs" "$ROOT/setup.py" \
  "$NODE:$SCR/"
if [ $BUILD = 1 ]; then
  ssh -o BatchMode=yes "$NODE" "cd $SCR && $GUARD $LOW $PIN env $ENVS $PY setup.py -q build_ext --inplace > /dev/null"
fi
WHERE=$(ssh -o BatchMode=yes "$NODE" "cd $SCR && env $ENVS $PY -c 'import dsart; print(dsart.__file__)'")
case "$WHERE" in
  "$SCR"/*) ;;
  *) echo "ABORT: dsart imports from $WHERE, not the scratch copy $SCR"; exit 3 ;;
esac

fail=0
for f in "$@"; do
  out=$(ssh -o BatchMode=yes "$NODE" "cd $SCR && $GUARD $LOW $PIN env $ENVS \
        timeout $TMO $PY -m pytest -q -p no:cacheprovider $MARK $f 2>&1 | tail -15" || true)
  echo "=== $f"
  echo "$out"
  if echo "$out" | grep -q "PRODUCTION_GUARD"; then
    echo "ABORT: production guard tripped on $NODE; remaining files not run"
    exit 4
  fi
  # a file whose tests all skip on CPU (CUDA-only) or are deselected (x2_gpu) is not a failure
  # case-insensitive: pytest reports a missing file as "ERROR: file or directory not found" + "no tests ran"
  echo "$out" | grep -q -E "passed|skipped|deselected|no tests ran" && ! echo "$out" | grep -q -i -E "failed|error" || fail=1
done
exit $fail
