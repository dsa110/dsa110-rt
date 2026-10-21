#!/bin/bash
# B5 - corner-turn ingress under the BASELINE (ConnectX-4s stay in their x4 slots,
# ~25 Gb/s per node for both ports).  Reuses tools/dod/corner_turn.sh
# (bench/net_pair.py on every node).  Run FROM h23 at a checkout whose
# tools/dod/corner_turn.sh matches the nodes, in an observing gap with the
# corr + search pipelines stopped - but ideally with the SNAPs still streaming,
# because the key unknown is how much a corr node can take BESIDE its ~18 Gb/s
# of SNAP data.  ~6 min total (4 x 60 s + setup).
#
#   S1  3 corr nodes as imaging owners at 2 Gb/s each   (baseline 4x needs ~2)
#   S2  3 corr nodes at 4 Gb/s each                      (optimistic baseline)
#   S3  4 search nodes at 13 Gb/s each                   (baseline 4x search share)
#   S4  4 search nodes at 20 Gb/s each                   (above today's ~17 Gb/s knee)
# Pass: loss < 1e-4 and no rise in the corr capture drop counters.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
DUR=${DUR:-60}
CORR_RX="n04 n11 n22"                    # corner_turn.sh knows these RX IPs (not n03: bad DIMM)
SEARCH_RX="n01 n02 n09 n13"
TX="n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
NTX=$(echo $TX | wc -w)

bg_ingress() {   # background NIC ingress on the corr receivers (SNAPs streaming?)
  for n in $CORR_RX; do
    r=$(ssh -o BatchMode=yes "$n.pro.pvt" 'f=/sys/class/net/enp129s0f0/statistics/rx_bytes; a=$(cat $f); sleep 5; b=$(cat $f); echo $(( (b-a)*8/5/1000000 ))' 2>/dev/null)
    echo "  $n background ingress: ${r} Mb/s"
  done
}

echo "== background ingress on corr receivers (SNAP traffic) before the test"; bg_ingress
run() {  # name rx_list per_rx_gbps rx_impl
  local rate; rate=$(awk "BEGIN{printf \"%.4f\", $3/$NTX/6}")
  echo "== $1: RX=[$2] per-RX ${3} Gb/s (6 flows x ${rate} Gb/s per TX->RX pair, RX impl $4)"
  RX_IMPL=$4 bash bench/timeres/corner_turn_rx.sh --duration "$DUR" --tx-list "$TX" --rx-list "$2" \
       --n-flows 6 --rate "$rate" 2>&1 | grep -v -E "launching|launched|^\s*$" | tail -30 || true
}
# corr nodes: python receiver (their _recv_epoll extension is a stale py3.8 build)
run S1 "$CORR_RX" 2 python
run S2 "$CORR_RX" 4 python
run S3 "$SEARCH_RX" 13 epoll
run S4 "$SEARCH_RX" 20 epoll
echo "== background ingress after"; bg_ingress
echo "logs: ~/dsart-corner-turn-logs/<UTC>/ (one dir per scenario)"
