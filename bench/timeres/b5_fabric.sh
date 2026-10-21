#!/bin/bash
# B5 - corner-turn ingress at the 4x op-point, reusing tools/dod/corner_turn.sh
# (bench/net_pair.py on every node).  Run FROM h23 at the repo root of a
# checkout whose tools/dod/corner_turn.sh matches the nodes, in an observing
# gap (corr + search pipelines stopped).  ~6 min total (4 x 60 s + setup).
#
# What it answers: can an imaging owner take one coarse-DM bucket of
# 16 chgroups x 8 sub-bands x 1900 cells at 262 us
#   cint4: 7.4 Gb/s per bucket      cint8: 14.8 Gb/s per bucket
# while (on corr nodes) the SNAP ingress (~18 Gb/s) shares the NIC?  Every NIC
# is a ConnectX-4 trained at PCIe Gen3 x4 (~25 Gb/s usable) - see report.
#
#   S1  7 owners (4 search + 3 corr), 7.4 Gb/s each  (cint4, 1 bucket)
#   S2  7 owners, 12 Gb/s each
#   S3  4 search owners, 14.8 Gb/s each (cint4, 2 buckets per search node)
#   S4  4 search owners, 20 Gb/s each (find the knee)
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
DUR=${DUR:-60}
OWNERS_ALL="n01 n02 n09 n13 n04 n11 n22"   # corner_turn.sh knows these RX IPs (not n03: bad DIMM)
SEARCH="n01 n02 n09 n13"
TX="n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"   # n03 left out (DIMM hangs)
NTX=$(echo $TX | wc -w)

bg_ingress() {   # background NIC ingress (SNAPs may still be streaming in a gap)
  for n in n04 n11 n22; do
    r=$(ssh -o BatchMode=yes "$n.pro.pvt" 'f=/sys/class/net/enp129s0f0/statistics/rx_bytes; a=$(cat $f); sleep 5; b=$(cat $f); echo $(( (b-a)*8/5/1000000 ))' 2>/dev/null)
    echo "  $n background ingress: ${r} Mb/s"
  done
}

echo "== background ingress on corr owners (before)"; bg_ingress
run() {  # name rx_list per_rx_gbps
  local rate; rate=$(awk "BEGIN{printf \"%.4f\", $3/$NTX/6}")
  echo "== $1: RX=[$2] per-RX ${3} Gb/s (6 flows x ${rate} Gb/s per TX->RX pair)"
  bash tools/dod/corner_turn.sh --duration "$DUR" --tx-list "$TX" --rx-list "$2" \
       --n-flows 6 --rate "$rate" 2>&1 | tail -25
}
run S1 "$OWNERS_ALL" 7.4
run S2 "$OWNERS_ALL" 12
run S3 "$SEARCH" 14.8
run S4 "$SEARCH" 20
echo "logs: ~/dsart-corner-turn-logs/<UTC>/ (one dir per scenario)"
