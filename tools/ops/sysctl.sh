#!/usr/bin/env bash
# Apply §6.1 NIC sysctl headroom (NOPASSWD allowlist per §6.2).
# Ubuntu 18.04 path: /sbin/sysctl (pre-/usr-merge).
# Ubuntu 20.04+ would use /usr/sbin/sysctl. Sudoers allowlist must match.
set -euo pipefail

SYSCTL=/sbin/sysctl # Ubuntu 18.04; Ubuntu 20.04+ would use /usr/sbin/sysctl.

apply_min() {
  local key=$1 target=$2
  local current
  current=$("$SYSCTL" -n "$key")
  if (( current >= target )); then
    printf 'OK   %s = %s  (>= target %s; preserving)\n' "$key" "$current" "$target"
  else
    printf 'BUMP %s  %s -> %s\n' "$key" "$current" "$target"
    sudo -n "$SYSCTL" -w "${key}=${target}"
  fi
}

apply_min net.core.rmem_max            268435456
apply_min net.core.wmem_max            268435456
apply_min net.core.netdev_max_backlog  100000

# §11.4 corner-turn fix (2026-05-15): With 16→4 fan-in at 7 Gb/s,
# the kernel reassembles ~5M fragments/s. Default 256 KiB ipfrag pool
# overflows in <1 ms, causing 90%+ ReasmFails and 1 Gb/s throughput
# ceiling on n01/n02. 256 MiB pool comfortably absorbs >100 ms of
# fragments at 7 Gb/s.  ipfrag_time dropped 30s -> 10s so stale
# half-fragments are aged out faster, keeping the live working set
# small.
# See m4b-deploy/CORNER-TURN-REPORT.md and dsart-corner-turn-logs/.
apply_min net.ipv4.ipfrag_high_thresh  268435456
apply_min net.ipv4.ipfrag_low_thresh   201326592
# ipfrag_time uses "set if greater" semantics inverted -- we want a
# SMALLER value than the default 30. apply_min won't do that. Force it.
sudo -n "$SYSCTL" -w net.ipv4.ipfrag_time=10 >/dev/null
printf 'SET  %-30s -> 10\n' net.ipv4.ipfrag_time
