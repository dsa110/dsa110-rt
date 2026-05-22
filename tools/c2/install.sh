#!/usr/bin/env bash
# C2 systemd-user unit installer. Idempotent:
#
#   * Symlinks the three unit files from the repo into
#     ~/.config/systemd/user/.
#   * systemctl --user daemon-reload.
#   * systemctl --user enable --now {dsart_c2,hiplot_c1,hiplot_c2}.service
#
# Safe to re-run; symlinks are -sf'd, enable --now is a no-op if the
# unit is already enabled + active.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/systemd"
DST_DIR="${HOME}/.config/systemd/user"

UNITS=(
  dsart_c2.service
  hiplot_c1.service
  hiplot_c2.service
)

echo "[c2/install.sh] repo=${REPO_ROOT}"
echo "[c2/install.sh] target=${DST_DIR}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "FATAL: ${SRC_DIR} not found" >&2
  exit 2
fi

mkdir -p "${DST_DIR}"
for unit in "${UNITS[@]}"; do
  src="${SRC_DIR}/${unit}"
  dst="${DST_DIR}/${unit}"
  if [[ ! -f "${src}" ]]; then
    echo "FATAL: ${src} not found" >&2
    exit 3
  fi
  ln -sf "${src}" "${dst}"
  echo "[c2/install.sh] symlinked ${unit} -> ${src}"
done

# Ensure the cluster-output dirs exist so hiplot doesn't fail at start.
sudo install -d -o "${USER}" -g "${USER}" \
  /dataz/dsa110/operations/C1/cluster_output \
  /dataz/dsa110/operations/C2/cluster_output \
  /dataz/dsa110/candidates \
  2>/dev/null || {
  # If we can't sudo, the user is on a host without /dataz/ (dev box).
  # The install on h23 should be run by the operator.
  echo "[c2/install.sh] WARN: could not create /dataz/dsa110/operations/{C1,C2}/cluster_output"
  echo "[c2/install.sh] WARN: the hiplot units will fail to start until those dirs exist."
}

systemctl --user daemon-reload

for unit in "${UNITS[@]}"; do
  if systemctl --user is-enabled --quiet "${unit}"; then
    echo "[c2/install.sh] ${unit}: already enabled"
  else
    systemctl --user enable "${unit}"
    echo "[c2/install.sh] ${unit}: enabled"
  fi
  if systemctl --user is-active --quiet "${unit}"; then
    echo "[c2/install.sh] ${unit}: already active"
  else
    systemctl --user start "${unit}"
    echo "[c2/install.sh] ${unit}: started"
  fi
done

systemctl --user --no-pager status "${UNITS[@]}" || true
