#!/usr/bin/env bash
# Apply §6.1 NIC sysctl headroom (NOPASSWD allowlist per §6.2).
# Ubuntu 18.04 path: /sbin/sysctl (pre-/usr-merge).
# Ubuntu 20.04+ would use /usr/sbin/sysctl. Sudoers allowlist must match.
set -euo pipefail

SYSCTL=/sbin/sysctl

sudo -n "$SYSCTL" -w net.core.rmem_max=268435456
sudo -n "$SYSCTL" -w net.core.wmem_max=268435456
sudo -n "$SYSCTL" -w net.core.netdev_max_backlog=100000
sudo -n "$SYSCTL" -p

echo "sysctl.sh: net.core.rmem_max=$(sysctl -n net.core.rmem_max)"
echo "sysctl.sh: net.core.wmem_max=$(sysctl -n net.core.wmem_max)"
echo "sysctl.sh: net.core.netdev_max_backlog=$(sysctl -n net.core.netdev_max_backlog)"
