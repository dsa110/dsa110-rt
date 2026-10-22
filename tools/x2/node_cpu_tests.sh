#!/bin/bash
# x2 CPU test harness: run dsart tests on a SEARCH NODE (CLAUDE.md: never on h23), CPU only, at
# the lowest priority, while the node keeps observing.
#
#   tools/x2/node_cpu_tests.sh [-n NODE] [-t TAG] [-b] -- tests/x2/test_a.py [tests/...]
#
# - scratch copy of the node's own checkout (inherits its cpython-311 C extensions), then this
#   worktree's src/ tests/ tools/ configs/ bench/ docs/ overlaid on it (never the production tree);
# - -b rebuilds C extensions in the scratch copy (setup.py build_ext --inplace);
# - every file runs separately: nice 19, ionice idle, one core, single-threaded BLAS, no GPU
#   (CUDA_VISIBLE_DEVICES=""), `-m "not x2_gpu"`, shell timeout (no pytest-timeout on nodes);
# - aborts if `import dsart` does not resolve to the scratch copy.
set -eo pipefail
NODE=n13.pro.pvt
TAG=$(git rev-parse --abbrev-ref HEAD | tr '/' '_')
BUILD=0
while getopts "n:t:b" o; do
  case $o in n) NODE=$OPTARG ;; t) TAG=$OPTARG ;; b) BUILD=1 ;; *) exit 2 ;; esac
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

ssh -o BatchMode=yes "$NODE" "test -d $SCR || { mkdir -p $(dirname $SCR) && $LOW cp -a /home/ubuntu/proj/dsa110-rt $SCR; }"
rsync -a --exclude='*.so' --exclude='__pycache__' --exclude='*.pdf' --exclude='*.png' \
  "$ROOT/src" "$ROOT/tests" "$ROOT/tools" "$ROOT/configs" "$ROOT/bench" "$ROOT/docs" "$ROOT/setup.py" \
  "$NODE:$SCR/"
if [ $BUILD = 1 ]; then
  ssh -o BatchMode=yes "$NODE" "cd $SCR && $LOW taskset -c $CORE env $ENVS $PY setup.py -q build_ext --inplace > /dev/null"
fi
WHERE=$(ssh -o BatchMode=yes "$NODE" "cd $SCR && env $ENVS $PY -c 'import dsart; print(dsart.__file__)'")
case "$WHERE" in
  "$SCR"/*) ;;
  *) echo "ABORT: dsart imports from $WHERE, not the scratch copy $SCR"; exit 3 ;;
esac

fail=0
for f in "$@"; do
  out=$(ssh -o BatchMode=yes "$NODE" "cd $SCR && $LOW taskset -c $CORE env $ENVS timeout 900 \
        $PY -m pytest -q -p no:cacheprovider -m 'not x2_gpu' $f 2>&1 | tail -15" || true)
  echo "=== $f"
  echo "$out"
  echo "$out" | grep -q -E "passed|no tests ran" && ! echo "$out" | grep -q -E "failed|error" || fail=1
done
exit $fail
