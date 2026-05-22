#!/usr/bin/env bash
# Legacy-stack shutdown for the C2 cutover (h23).
#
# Automates §6 of docs/c1c2/C1C2_DESIGN.md:
#
#   1. Stop legacy run_T2* user services in calibration23.
#      (run on h23: relies on `lxc exec calibration23 -- systemctl
#      --user stop` working via the operator's lxc CLI.)
#   2. Stop legacy hiplot.service (port 5007) on h23.
#   3. Stop legacy tasktrigger.service on h23.
#   4. Snapshot the legacy cluster_output/ dir.
#
# Idempotent — re-running after a successful shutdown should be a
# no-op (it'll just re-confirm the services are inactive and skip the
# snapshot if it already exists).
#
# DO NOT run this on the dev workstation. Intended for h23 only.

set -euo pipefail

LOG_PREFIX="[c2/legacy_shutdown.sh]"
log() { echo "${LOG_PREFIX} $*"; }
warn() { echo "${LOG_PREFIX} WARN: $*" >&2; }

# --- 1. legacy T2 (in calibration23 LXD container) ----------------------

if command -v lxc >/dev/null 2>&1; then
  log "Stopping legacy T2 (calibration23, systemd-user)…"
  T2_UNITS=$(
    lxc exec calibration23 -- \
      bash -lc "systemctl --user list-units --no-legend --no-pager 'run_T2*' 2>/dev/null" \
      | awk '{print $1}' || true
  )
  if [[ -z "${T2_UNITS}" ]]; then
    log "  no run_T2* units found in calibration23 (already stopped or never existed)"
  else
    for u in ${T2_UNITS}; do
      log "  stopping ${u} in calibration23"
      lxc exec calibration23 -- bash -lc \
        "systemctl --user stop ${u}" || warn "    stop ${u} failed"
    done
  fi
else
  warn "lxc CLI not available; skipping legacy T2 stop. Stop manually in calibration23: 'systemctl --user stop run_T2*'"
fi

# --- 2. legacy hiplot.service ------------------------------------------

stop_user_unit() {
  local unit="$1"
  if systemctl --user is-active --quiet "${unit}" 2>/dev/null; then
    log "stopping ${unit}"
    systemctl --user stop "${unit}" || warn "  systemctl stop ${unit} failed"
  else
    log "${unit}: already inactive"
  fi
  if systemctl --user is-enabled --quiet "${unit}" 2>/dev/null; then
    log "disabling ${unit} so it doesn't auto-restart"
    systemctl --user disable "${unit}" || warn "  systemctl disable ${unit} failed"
  fi
}

stop_user_unit hiplot.service

# --- 3. legacy tasktrigger.service -------------------------------------

stop_user_unit tasktrigger.service

# --- 4. snapshot legacy cluster_output ---------------------------------

LEGACY_SRC=/dataz/dsa110/operations/T2/cluster_output
LEGACY_DST=/dataz/dsa110/operations/T2/cluster_output.legacy

if [[ -d "${LEGACY_SRC}" ]]; then
  if [[ -d "${LEGACY_DST}" ]]; then
    log "snapshot ${LEGACY_DST} already exists — skipping rename (idempotent)"
  else
    log "snapshotting ${LEGACY_SRC} -> ${LEGACY_DST}"
    # rsync preserves perms / ownership; we don't delete the original
    # (the operator can verify the snapshot before reclaiming space).
    rsync -aHv "${LEGACY_SRC}/" "${LEGACY_DST}/" \
      || warn "rsync legacy cluster_output failed"
  fi
else
  log "${LEGACY_SRC} not found — nothing to snapshot"
fi

log "DONE. Verify with: systemctl --user status hiplot.service tasktrigger.service"
log "Then bring up new stack: tools/c2/install.sh"
