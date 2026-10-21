#!/bin/bash
# nic_sampler.sh <iface> <n_samples> <period_s> - one line per sample:
#   <unix time> rx_bytes_phy rx_packets_phy rx_discards_phy rx_out_of_buffer udp_rcvbuf_errors
for ((i = 0; i < $2; i++)); do
  s=$(ethtool -S "$1" | awk '$1 == "rx_bytes_phy:" {b = $2} $1 == "rx_packets_phy:" {p = $2}
        $1 == "rx_discards_phy:" {d = $2} $1 == "rx_out_of_buffer:" {o = $2} END {print b, p, d, o}')
  u=$(awk '/^Udp:/ && $2 ~ /^[0-9]/ {print $6}' /proc/net/snmp)
  echo "$(date +%s.%N) $s $u"
  sleep "$3"
done
