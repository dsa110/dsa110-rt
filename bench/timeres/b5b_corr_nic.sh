#!/bin/bash
# B5b - corr-node ingress headroom at the NIC (beside the SNAP stream), measured
# with the ConnectX-4 hardware counters rather than the receiver: the corr
# nodes' _recv_epoll is a stale py3.8 build, and the python receiver cannot
# keep up, so b5's S1/S2 only measured python.  nic_sampler.sh logs
# rx_bytes_phy / rx_discards_phy / rx_out_of_buffer / UDP RcvbufErrors every
# 5 s on each receiver while corner_turn_rx.sh offers 2, 4 and 8 Gb/s.
# rx_discards_phy is the x4-slot signature (NIC buffer full because PCIe
# cannot drain it); rx_out_of_buffer is the host not posting buffers.
# Run FROM h23 in an observing gap with the pipelines stopped, SNAPs streaming.
# Output: $OUT/b5b/<node>.txt + $OUT/b5b/marks.txt  (analyse with b5b_analyse.py)
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
CORR_RX="n04 n11 n22"
TX="n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
NTX=$(echo $TX | wc -w)
SAMP=/tmp/timeres_b5b_nic.txt
OUTD="${OUT:-/tmp}/b5b"
mkdir -p "$OUTD"
: > "$OUTD/marks.txt"

for n in $CORR_RX; do
  scp -q bench/timeres/nic_sampler.sh "$n.pro.pvt:/tmp/timeres_nic_sampler.sh"
  ssh -o BatchMode=yes "$n.pro.pvt" \
    "nohup bash /tmp/timeres_nic_sampler.sh enp129s0f0 ${NSAMP:-150} 5 > $SAMP 2>&1 < /dev/null &"
done
echo "== sampler started on $CORR_RX; 60 s of SNAP-only baseline"
echo "baseline $(date +%s) $(( $(date +%s) + 55 ))" >> "$OUTD/marks.txt"
sleep 60
for g in 2 4 8; do
  rate=$(awk "BEGIN{printf \"%.4f\", $g/$NTX/6}")
  echo "== ${g} Gb/s per corr RX"
  RX_IMPL=python bash bench/timeres/corner_turn_rx.sh --duration 45 --tx-list "$TX" --rx-list "$CORR_RX" \
       --n-flows 6 --rate "$rate" > "$OUTD/ct_${g}.log" 2>&1 || true
  # [T0, T0 + 45 s] is the traffic window
  t0=$(grep -o "(unix [0-9]*" "$OUTD/ct_${g}.log" | head -1 | awk '{print $2}')
  echo "g${g} ${t0} $(( t0 + 45 ))" >> "$OUTD/marks.txt"
  grep -E "Per-RX agg" "$OUTD/ct_${g}.log" || true
done
for n in $CORR_RX; do scp -q "$n.pro.pvt:$SAMP" "$OUTD/$n.txt"; done
echo "samples in $OUTD/"
