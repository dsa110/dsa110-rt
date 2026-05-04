#!/usr/bin/env bash
# M0 Definition-of-Done (§8) — intended for h01 with dsa110-rt conda env + PSRDADA rings.
# set -u DROPPED intentionally:
#   Conda's MKL activate.d hook references MKL_INTERFACE_LAYER
#   without a default; under 'set -u' that aborts conda activate
#   before any STEP runs. Same issue + fix as
#   tools/ops/install_psrdada.sh. See plan §6.
#   pipefail kept so step failures aren't masked by tee.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# M0 smoke: small ephemeral buffers, just enough to exercise plumbing.
# Production buffer sizing per plan §6.1 lives in configs/config_corr.yaml
# and is exercised in M2+ DoD, not M0.
M0_DADA_BSIZE=$((4 * 1024 * 1024))   # 4 MiB
M0_DADA_NBUFS=4                       # 16 MiB total — fits in any node

# PSRDADA buffer lifecycle for M0 smokes.
# PSRDADA's -k flag is HEX, not ASCII (so 'dada' -> 0xdada). Per buffer
# key, dada_db creates SysV shm + semaphore objects whose ipcs keys share a
# common hex suffix (<key>). Prefix nibbles are assigned by PSRDADA and are
# not fixed per role (observed 0x0000–0x0011 across runs in M0 chunk 2C).
# Existence/cleanup match any prefix: ^0x[0-9a-f]+<key>$.
_dada_buffer_exists() {
  local key="$1"
  # Match any shm whose key ends in <key> (PSRDADA assigns sequential
  # prefixes — observed 0x0000, 0x000a, 0x000b, 0x000e through 0x0011
  # across runs; not a fixed role nibble).
  ipcs -m 2>/dev/null | awk '{print $1}' | grep -qiE "^0x[0-9a-f]+${key}$"
}
_dada_force_cleanup() {
  local key="$1"
  local hex
  while read -r hex; do
    [ -n "$hex" ] || continue
    ipcrm -M "$hex" >/dev/null 2>&1 || true
  done < <(ipcs -m 2>/dev/null | awk -v k="$key" '$1 ~ "^0x[0-9a-f]+" k "$" { print $1 }')
  while read -r hex; do
    [ -n "$hex" ] || continue
    ipcrm -S "$hex" >/dev/null 2>&1 || true
  done < <(ipcs -s 2>/dev/null | awk -v k="$key" '$1 ~ "^0x[0-9a-f]+" k "$" { print $1 }')
}
_dada_teardown() {
  local key="$1"
  if _dada_buffer_exists "$key"; then
    if ! dada_db -k "$key" -d >/dev/null 2>&1; then
      _dada_force_cleanup "$key"
    fi
  fi
}
_dada_setup() {
  local key="$1" bsize="$2" nbufs="$3"
  _dada_teardown "$key"
  dada_db -k "$key" -b "$bsize" -n "$nbufs" >/dev/null
}
trap 'for k in dada dadc; do _dada_teardown "$k"; done' EXIT

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

echo "== M0: defensive dsart editable install (tools/ops/install_dsart.sh) =="
bash "${REPO_ROOT}/tools/ops/install_dsart.sh"

STEP=""
fail() {
  echo "[M0:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M0:${STEP}] PASS"
}

STEP="gpu_runtime"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 2 and torch.cuda.get_device_capability(0)[0] >= 7' \
  || fail "torch CUDA check"
OUT=$(python -c 'import torch; x = torch.zeros(1, device="cuda:0"); y = torch.zeros(1, device="cuda:1"); print((x + 1).item(), (y + 1).item())') \
  || fail "dual-GPU smoke"
[[ "${OUT}" == "1.0 1.0" ]] || fail "unexpected gpu smoke output: ${OUT}"
pass

STEP="plumbing_junkdb"
echo "== [M0:plumbing_junkdb] dada_junkdb against ephemeral buffer =="
JUNKDB_KEY=dada
_dada_setup "$JUNKDB_KEY" "$M0_DADA_BSIZE" "$M0_DADA_NBUFS"
DSART_JUNKDB_HEADER="${DSART_JUNKDB_HEADER:-tests/fixtures/headers/correlator_header_dsaX.txt}"
JUNKDB_RC=0
timeout 5 dada_junkdb -k "$JUNKDB_KEY" -t 1 -r 1 "$DSART_JUNKDB_HEADER" \
  > /tmp/dsart_m0_junkdb.log 2>&1 || JUNKDB_RC=$?
_dada_teardown "$JUNKDB_KEY"
if [ "$JUNKDB_RC" -ne 0 ]; then
  echo "[M0:plumbing_junkdb] FAIL dada_junkdb run (rc=$JUNKDB_RC)"
  echo '--- tail /tmp/dsart_m0_junkdb.log ---'
  tail -20 /tmp/dsart_m0_junkdb.log 2>/dev/null || true
  exit 1
fi
echo "[M0:plumbing_junkdb] PASS"

STEP="plumbing_fake_capture_dada"
echo "== [M0:plumbing_fake_capture_dada] fake_capture_dada with dbnull consumer =="
FCD_KEY=dadc
_dada_setup "$FCD_KEY" "$M0_DADA_BSIZE" "$M0_DADA_NBUFS"
# Drainer in background; reads endlessly until killed.
dada_dbnull -k "$FCD_KEY" -q >/dev/null 2>&1 &
FCD_DBNULL_PID=$!
# Bounded writer; --secs 1 must finish in ~1 sec; hard timeout 10 sec.
FCD_RC=0
timeout 10 python -m bench.fake_capture_dada --dada-key "$FCD_KEY" --secs 1 \
  > /tmp/dsart_m0_fcd.log 2>&1 || FCD_RC=$?
# Stop drainer; ignore exit code (we kill it).
kill "$FCD_DBNULL_PID" 2>/dev/null || true
wait "$FCD_DBNULL_PID" 2>/dev/null || true
_dada_teardown "$FCD_KEY"
if [ "$FCD_RC" -ne 0 ]; then
  echo "[M0:plumbing_fake_capture_dada] FAIL fake_capture_dada exit=$FCD_RC"
  echo '--- tail /tmp/dsart_m0_fcd.log ---'
  tail -20 /tmp/dsart_m0_fcd.log 2>/dev/null || true
  exit 1
fi
echo "[M0:plumbing_fake_capture_dada] PASS"

STEP="plumbing_fake_corr"
python -m bench.fake_corr_to_search --self-test --rate native || fail "fake_corr_to_search"
pass

STEP="configs"
shopt -s nullglob
for y in "${REPO_ROOT}/configs/"*.yaml; do
  python -m dsart.common.config_loader "${y}" >/dev/null || fail "${y}"
done
shopt -u nullglob
pass

STEP="voltage_fixtures"
VF_ROOT=/home/ubuntu/data/voltage_fixtures
mkdir -p "${VF_ROOT}"
[[ -d "${VF_ROOT}" ]] || fail "missing ${VF_ROOT}"
python - <<'PY' || fail "template/schema validation"
import json
import os
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

repo = Path(os.environ["REPO_ROOT"])
schema_path = repo / "tests" / "fixtures" / "voltage_fixture_manifest.schema.json"
schema = json.loads(schema_path.read_text())
v = Draft202012Validator(schema)
for name in ("manifest.template.continuum.yaml", "manifest.template.burst.yaml"):
    p = repo / "voltage_fixtures" / name
    inst = yaml.safe_load(p.read_text())
    v.validate(inst)

vf_root = Path("/home/ubuntu/data/voltage_fixtures")
runs = sorted(p for p in vf_root.iterdir() if p.is_dir())
if runs:
    import subprocess
    import sys

    rid = runs[0].name
    manifest = vf_root / rid / "manifest.yaml"
    if manifest.is_file():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "bench.replay_voltage_dump",
                "--run-id",
                rid,
                "--chgroups",
                "0",
                "--rate",
                "fast",
                "--dry-run",
            ],
            cwd=os.environ["REPO_ROOT"],
        )
PY
pass

STEP="host_identity"
python -c 'import dsart.common.host as h; assert h.PHASE == "a"' || fail "host phase"
pass

echo "== [M0:cpu_affinity] checking taskset + systemd CPUAffinity availability =="
if ! command -v taskset >/dev/null 2>&1; then
  echo "[M0:cpu_affinity] FAIL  taskset not found (util-linux missing? unexpected on Ubuntu 18.04)"
  exit 1
fi
taskset --version >/dev/null 2>&1 || {
  echo "[M0:cpu_affinity] FAIL  taskset --version returned non-zero"; exit 1; }
systemctl --user show -p DefaultCPUAccounting >/dev/null 2>&1 || {
  echo "[M0:cpu_affinity] FAIL  systemctl --user show failed (linger not enabled? §6.2)"; exit 1; }
echo "[M0:cpu_affinity] PASS"

STEP="mem_available"
# M0 smoke threshold — small, easily met. M2+ DoD will use a
# per-host threshold derived from configs/numa_topology.yaml::mem_kb
# (sum across NUMA nodes); 96 GiB original hardcode was infeasible
# on h01 (95 GB marketing = 93 GiB usable). See plan §8.
M0_MIN_MEM_GIB=16
awk -v min=$((M0_MIN_MEM_GIB * 1024 * 1024)) '/^MemAvailable:/ { exit !($2 >= min) }' /proc/meminfo \
  || fail "MemAvailable < ${M0_MIN_MEM_GIB} GiB"
pass

for k in dada dadc; do _dada_teardown "$k"; done

echo "M0 PASS"
