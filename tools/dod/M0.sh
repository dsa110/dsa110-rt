#!/usr/bin/env bash
# M0 Definition-of-Done (§8) — intended for h01 with dsa110-rt conda env + PSRDADA rings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

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
command -v dada_junkdb >/dev/null || fail "dada_junkdb not in PATH"
command -v dada_dbmetric >/dev/null || fail "dada_dbmetric not in PATH"
dada_junkdb -k dada -r 1124 -t 6 || fail "dada_junkdb run"
dada_dbmetric -k dada >/tmp/m0_dada_metric.txt || fail "dada_dbmetric"
pass

STEP="plumbing_fake_capture"
python -m bench.fake_capture_dada --rate native --secs 60 --seed 0 || fail "fake_capture_dada"
pass

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
awk '/^MemAvailable:/ { exit !($2 >= 96 * 1024 * 1024) }' /proc/meminfo || fail "MemAvailable < 96 GiB"
pass

echo "M0 PASS"
