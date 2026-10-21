#!/bin/bash
# B5c - search-node corner-turn ingress knee with NIC counters.  b5 S3 (13 Gb/s
# per search node, 15 TX x 6 flows, epoll RX) already lost ~3.7% of frames on
# n02/n09/n13, below the ~17 Gb/s knee measured 2026-09-22 with single-fragment
# datagrams, so sweep lower and log rx_discards_phy / rx_out_of_buffer /
# UDP RcvbufErrors (nic_sampler.sh) to see WHERE the loss happens.
# Run FROM h23 in an observing gap, pipelines stopped.  ~9 min.
# Output: $OUT/b5c/<node>.txt + marks.txt + ct_<g>.log  (b5b_analyse.py works on it)
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
RXS=${RXS:-"n02 n09 n13"}                 # n01 lost 88% at every rate in b5 (separate fault)
TX="n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
NTX=$(echo $TX | wc -w)
SAMP=/tmp/timeres_b5c_nic.txt
OUTD="${OUT:-/tmp}/b5c"
mkdir -p "$OUTD"
: > "$OUTD/marks.txt"

for n in $RXS; do
  scp -q bench/timeres/nic_sampler.sh "$n.pro.pvt:/tmp/timeres_nic_sampler.sh"
  ssh -o BatchMode=yes "$n.pro.pvt" \
    "nohup bash /tmp/timeres_nic_sampler.sh enp129s0f0 ${NSAMP:-130} 5 > $SAMP 2>&1 < /dev/null &"
done
echo "baseline $(date +%s) $(( $(date +%s) + 25 ))" >> "$OUTD/marks.txt"
sleep 30
for g in ${RATES:-6 9 11}; do
  rate=$(awk "BEGIN{printf \"%.4f\", $g/$NTX/6}")
  echo "== ${g} Gb/s per search RX"
  bash bench/timeres/corner_turn_rx.sh --duration 45 --tx-list "$TX" --rx-list "$RXS" \
       --n-flows 6 --rate "$rate" > "$OUTD/ct_${g}.log" 2>&1 || true
  t0=$(grep -o "(unix [0-9]*" "$OUTD/ct_${g}.log" | head -1 | awk '{print $2}')
  echo "g${g} ${t0} $(( t0 + 45 ))" >> "$OUTD/marks.txt"
  grep -E "Per-RX agg" "$OUTD/ct_${g}.log" || true
done
for n in $RXS; do scp -q "$n.pro.pvt:$SAMP" "$OUTD/$n.txt"; done
echo "samples in $OUTD/"
