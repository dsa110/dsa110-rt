#!/bin/bash
# READ-ONLY NIC counter sampler for the x2 checkpoint runner (cp_runner.py). Safe while observing:
# it only reads `ethtool -S`, /proc/net/snmp and sysfs.
#
#   ssh <node> 'bash -s <n_samples> <period_s>' < tools/x2/cp_runner_nic_sample.sh > <node>.txt
#
# The data interface is READ FROM THE NODE, never hard-coded: the interface holding the 10.41.0.x
# address, resolved through a bridge (n02: br2) to its physical member, else the bond's active slave.
# (Normally enp129s0f0; n14 runs on f1 after an in-memory failover that reverts on reboot.)
# Output: "# iface <name> mtu <mtu> host <host>" then one line per sample:
#   <unix> rx_bytes_phy rx_packets_phy rx_discards_phy rx_out_of_buffer rx_crc_errors_phy
#          tx_pause_ctrl_phy udp_RcvbufErrors udp_InErrors ip_ReasmFails
N=${1:-60}
P=${2:-5}
IF=$(ip -4 -br addr show 2>/dev/null | awk '$3 ~ /^10\.41\.0\./ {print $1; exit}')
if [ -n "$IF" ] && [ -d "/sys/class/net/$IF/brif" ]; then
  M=$(ls "/sys/class/net/$IF/brif" | grep -E '^(enp|ens|eth)' | head -1)
  [ -n "$M" ] && IF=$M
fi
if [ -n "$IF" ] && [ -r "/sys/class/net/$IF/bonding/active_slave" ]; then
  IF=$(cat "/sys/class/net/$IF/bonding/active_slave")
fi
[ -n "$IF" ] || IF=enp129s0f0
echo "# iface $IF mtu $(cat /sys/class/net/$IF/mtu 2>/dev/null) host $(hostname)"
for ((i = 0; i < N; i++)); do
  s=$(ethtool -S "$IF" 2>/dev/null | awk '
      $1 == "rx_bytes_phy:" {b = $2} $1 == "rx_packets_phy:" {p = $2}
      $1 == "rx_discards_phy:" {d = $2} $1 == "rx_out_of_buffer:" {o = $2}
      $1 == "rx_crc_errors_phy:" {c = $2} $1 == "tx_pause_ctrl_phy:" {x = $2}
      END {print b+0, p+0, d+0, o+0, c+0, x+0}')
  u=$(awk '/^Udp:/ && $2 ~ /^[0-9]/ {print $6, $4}' /proc/net/snmp)
  r=$(awk '/^Ip:/ && $2 ~ /^[0-9]/ {print $17}' /proc/net/snmp)
  echo "$(date +%s.%N) $s $u $r"
  sleep "$P"
done
