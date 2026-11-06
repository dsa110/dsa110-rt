#!/bin/bash
# Deploy this worktree as a scratch copy on many nodes in parallel (fleet runs; never the production tree):
#
#   tools/x2/fleet_scratch.sh -t TAG [-b] [-n "n01 n02 ..."]
#
# Per node, as tools/x2/node_cpu_tests.sh: copy the node's own checkout to /home/ubuntu/x2_scratch/TAG once
# (inherits its cpython-311 C extensions), overlay src/ tests/ tools/ configs/ bench/ docs/ setup.py, and
# with -b rebuild the C extensions (setup.py build_ext --inplace) under tools/x2/prod_guard.sh STOPPED mode.
# Prints one line per node: the commit deployed, whether `import dsart` resolves to the scratch copy, and the
# capture binary's mtime.
set -eo pipefail
TAG=
BUILD=0
NODES="n01 n02 n09 n13 n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
while getopts "t:bn:" o; do
  case $o in t) TAG=$OPTARG ;; b) BUILD=1 ;; n) NODES=$OPTARG ;; *) exit 2 ;; esac
done
[ -n "$TAG" ] || { echo "usage: $0 -t TAG [-b] [-n \"nodes\"]"; exit 2; }
ROOT=$(git rev-parse --show-toplevel)
REV=$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet HEAD -- src tools configs bench setup.py || echo +dirty)
PY=/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python
SCR=/home/ubuntu/x2_scratch/$TAG
LOGDIR=$(mktemp -d)

one() {
  local n=$1 h=$1.pro.pvt
  ssh -o BatchMode=yes "$h" "test -d $SCR || { mkdir -p $(dirname $SCR) && nice -n 19 cp -a /home/ubuntu/proj/dsa110-rt $SCR; }"
  rsync -a --exclude='*.so' --exclude='__pycache__' --exclude='*.pdf' --exclude='*.png' \
    "$ROOT/src" "$ROOT/tests" "$ROOT/tools" "$ROOT/configs" "$ROOT/bench" "$ROOT/docs" "$ROOT/setup.py" "$h:$SCR/"
  ssh -o BatchMode=yes "$h" "echo $REV > $SCR/X2_REV"
  if [ $BUILD = 1 ]; then
    ssh -o BatchMode=yes "$h" "cd $SCR && env GUARD_REQUIRE_STOPPED=1 bash tools/x2/prod_guard.sh nice -n 5 \
      env MKL_INTERFACE_LAYER=GNU,LP64 $PY setup.py -q build_ext --inplace > $SCR/build_ext.log 2>&1 || \
      { echo BUILD_FAILED; tail -5 $SCR/build_ext.log; }"
  fi
  ssh -o BatchMode=yes "$h" "cd $SCR && echo rev=\$(cat X2_REV) dsart=\$(env PYTHONPATH=$SCR/src $PY -c \
    'import dsart; print(\"scratch\" if dsart.__file__.startswith(\"$SCR/\") else dsart.__file__)') \
    capture=\$(stat -c %y src/dsart/capture/dsart_capture_manythread | cut -c1-16)"
}

for n in $NODES; do
  one "$n" > "$LOGDIR/$n.log" 2>&1 &
done
wait
for n in $NODES; do
  printf "%-4s %s\n" "$n" "$(tail -3 "$LOGDIR/$n.log" | tr '\n' ' ')"
done
rm -rf "$LOGDIR"
