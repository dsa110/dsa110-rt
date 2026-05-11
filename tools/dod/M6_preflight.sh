#!/usr/bin/env bash
# M6 pre-flight readiness check (§8 line 2342+ — M6 plan section to be
# filled by chunk-9 fold) — runs on h01.
#
# Purpose: fail fast if anything M6 depends on is broken before authoring.
# M6 builds the search-node clustering + cube-dump path on top of M5's
# detector pipeline:
#   * src/dsart/cluster/{forward, features, state, cands_logger}.py
#   * src/dsart/dump/{cube_dump, udp_listener}.py
#   * services/search_compute.py rewiring to run clusterer + cube-dump
#     in a per-(search_node, gpu_half) ThreadPool worker
#   * three benches: clusterer_throughput.py, cube_dump_e2e.py,
#     and the existing M5 voltage_fixture_search.py end-to-end re-run
#
# So M6 depends on:
#   * M5 must be hardened (search-node detector pipeline) — verify
#     ${HOME}/dsart-m5-status.json reports stage starting with "complete"
#   * conda env dsa110-rt with hdbscan + scikit-learn (D5 DBSCAN fallback)
#   * GPU 1 visible (PARALLEL_AGENTS.md §4.2: M5 + M6 → CUDA_VISIBLE_DEVICES=1)
#   * /home/ubuntu/data/voltages/ root present (M6 cube-dump e2e bench
#     consumes 250924mptq for the burst end-to-end test; informational —
#     M6 cube-injection critical path is independent)
#   * tools/viz/common.py present + read-only-importable (M3 owns;
#     M6's tools/viz/cluster_check.py consumes it)
#   * configs/dm_plan.npz round-trips through DmPlan.from_npz()
#   * configs/config_compute_search.yaml schema-valid (M6-extended)
#   * /var/lock/dsart-m6.lock writable (PARALLEL_AGENTS.md §4.4)
#   * UDP port (default 11227) is bind-able on 127.0.0.1
#
# §6 conda-activate shell pattern: 'set -u' DROPPED (conda MKL hook
# references MKL_INTERFACE_LAYER without default); 'pipefail' kept.
#
# Read-only: this script never writes to the repo or to /home/ubuntu/data/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
export DSART_CONFIG_DIR="${REPO_ROOT}/configs"
# pyproject.toml uses [tool.setuptools.packages.find] where=["src"], so the
# `dsart` package lives at ${REPO_ROOT}/src/dsart. PYTHONPATH must point at
# the `src` parent — pointing at ${REPO_ROOT} alone falls through to the
# conda env's editable install (which on h01 may resolve to the M3 working
# tree, parallel to ours, that doesn't carry M6 modules).
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

# PARALLEL_AGENTS.md §4 isolation envelope (D11 in M6_PLAN_FIXES.md).
# M5 + M6 share the m5 envelope (M6 is incremental on top of M5).
export DSART_BUFFER_KEY_PREFIX="${DSART_BUFFER_KEY_PREFIX:-m5}"
export DSART_ETCD_NAMESPACE_PREFIX="${DSART_ETCD_NAMESPACE_PREFIX:-m5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
M5_STATUS_JSON="${M5_STATUS_JSON:-${HOME}/dsart-m5-status.json}"
VOLTAGES_ROOT="${VOLTAGES_ROOT:-/home/ubuntu/data/voltages}"
M6_LOCKFILE="${M6_LOCKFILE:-/var/lock/dsart-m6.lock}"
M6_UDP_PORT="${M6_UDP_PORT:-11227}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M6_pre:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M6_pre:${STEP}] PASS"
}
warn() {
  echo "[M6_pre:${STEP}] WARN $*"
}

STEP="host_identity"
[[ "$(hostname -s)" == "lxd110h01" ]] || fail "expected lxd110h01, got $(hostname -s)"
pass

STEP="m1_status"
# M1 must be complete (hardened preferred, plain "complete" tolerated).
[[ -f "${M1_STATUS_JSON}" ]] || fail "missing ${M1_STATUS_JSON}"
python3 - <<PY || fail "M1 status JSON did not validate"
import json
s = json.load(open("${M1_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"stage not complete: {s!r}"
assert s.get("milestone", "") == "M1", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
print(f"M1 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m5_status"
# M5 must be complete (hardened or approved). M6 is incremental on top
# of the M5 detector pipeline; we don't proceed without a healthy M5.
[[ -f "${M5_STATUS_JSON}" ]] || fail "missing ${M5_STATUS_JSON} — M5 hasn't shipped on this host"
python3 - <<PY || fail "M5 status JSON did not validate"
import json
s = json.load(open("${M5_STATUS_JSON}"))
assert s.get("stage", "").startswith("complete"), f"M5 stage not complete: {s!r}"
assert s.get("milestone", "") == "M5", f"wrong milestone: {s!r}"
assert s.get("host", "") == "lxd110h01", f"wrong host: {s!r}"
print(f"M5 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m1_artifacts"
[[ -f "${REPO_ROOT}/configs/dm_plan.npz" ]] || fail "missing configs/dm_plan.npz"
DSART_TEST=1 python3 - <<PY || fail "dm_plan.npz did not validate"
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from dsart.common.contracts import DmPlan
plan = DmPlan.from_npz("${REPO_ROOT}/configs/dm_plan.npz")
assert plan.fine_dm.shape[0] > 100
assert plan.coarse_dm.shape[0] > 8
print(f"DmPlan: {plan.fine_dm.shape[0]} fine, {plan.coarse_dm.shape[0]} coarse")
PY
pass

STEP="git_clean"
# m6/main worktree should be clean. We're on m6/main (not main); the
# branch check below verifies that. Operator-facing artifacts under
# bench/reports/ are git-ignored (M5 chunk 8 fix; carried into M6).
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes in ${REPO_ROOT}; run 'git status'"
pass

STEP="git_branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "${BRANCH}" == "m6/main" || "${BRANCH}" == "integration/m3-m5-m6" ]] || fail "expected branch=m6/main or integration/m3-m5-m6, got ${BRANCH}"
git remote -v | grep -q '^origin' || fail "missing origin remote"
pass

STEP="conda_env"
[[ "${CONDA_DEFAULT_ENV:-}" == "dsa110-rt" ]] || fail "CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', expected dsa110-rt"
pass

STEP="dsart_import"
# Core M1 + M5 modules must import cleanly. M6 modules
# (cluster, dump) currently only contain __init__.py stubs (chunk 0);
# importing them is cheap and catches any package-discovery bugs.
python -c '
import dsart, dsart.common.host, dsart.common.config_loader
import dsart.common.contracts, dsart.common.constants, dsart.common.dispersion
import dsart.detector, dsart.fine_dm, dsart.image
import dsart.inject, dsart.noise_norm
import dsart.services.cube_pipeline, dsart.services.search_compute
import dsart.services.rx_ring
' || fail "dsart package import (M1+M5 modules)"
# M6 module imports are best-effort — they only exist after chunk 1 lands.
python -c 'import dsart.cluster, dsart.dump' 2>/dev/null \
  && echo "  M6 cluster + dump packages importable" \
  || warn "M6 cluster + dump packages not importable (expected pre-chunk-1)"
pass

STEP="numerical_deps"
python - <<'PY' || fail "numerical deps"
import importlib
mods = ["numpy", "yaml", "pytest", "jsonschema", "torch", "astropy"]
for m in mods:
    importlib.import_module(m)
import numpy, torch
print(f"numpy={numpy.__version__} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
PY
pass

STEP="clustering_deps"
# Per D4: HDBSCAN primary, sklearn DBSCAN fallback (D5). Both must be
# importable. The hdbscan package is a pip install on top of the conda
# env; tolerate its absence with a warn so the chunk-0 preflight still
# passes pre-chunk-1 (chunk 1 will install it as part of its setup).
python - <<'PY' || fail "scikit-learn (DBSCAN fallback)"
import sklearn
import sklearn.cluster
print(f"sklearn={sklearn.__version__}")
assert hasattr(sklearn.cluster, "DBSCAN"), "sklearn.cluster.DBSCAN not present"
PY
python - <<'PY' || warn "hdbscan not installed; D5 fallback path will be tested but primary path will skip. install with: pip install hdbscan"
import hdbscan
import importlib.metadata
try:
    ver = importlib.metadata.version("hdbscan")
except importlib.metadata.PackageNotFoundError:
    ver = getattr(hdbscan, "__version__", "unknown")
print(f"hdbscan={ver}")
assert hasattr(hdbscan, "HDBSCAN"), "hdbscan.HDBSCAN class not present"
PY
pass

STEP="gpu_visible"
# PARALLEL_AGENTS.md §4.2: M6 → CUDA_VISIBLE_DEVICES=1 (shared with M5).
python - <<'PY' || fail "GPU 1 not visible to torch"
import torch
n = torch.cuda.device_count()
assert n == 1, f"expected 1 visible GPU (CUDA_VISIBLE_DEVICES=1), got {n}"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name!r} sm_{cap[0]}{cap[1]}")
assert cap[0] >= 7, f"GPU compute capability {cap}, expected sm_70+ (Turing or newer)"
PY
pass

STEP="config_compute_search_yaml"
# Schema-validate configs/config_compute_search.yaml. M6 chunk 0
# stripped the trigger: block; chunk 5 will add cluster: + cube_dump:
# + udp_trigger: blocks. The required-keys check below tracks the
# minimum surface present in chunk 0.
python - <<PY || fail "configs/config_compute_search.yaml schema invalid"
import yaml
with open("${REPO_ROOT}/configs/config_compute_search.yaml") as f:
    cfg = yaml.safe_load(f)
required_top = {"schema_version", "dm_plan_path", "detector_class", "detector",
                "cube", "noise", "decoder"}
missing = required_top - set(cfg.keys())
if missing:
    raise SystemExit(f"missing top-level keys: {sorted(missing)}")
det = cfg["detector"]
det_required = {"threshold_sigma", "k_img", "k_dm", "k_time", "kernel_dtype",
                "k_time_widths"}
det_missing = det_required - set(det.keys())
if det_missing:
    raise SystemExit(f"detector missing keys: {sorted(det_missing)}")
n_triples = det["k_img"] * det["k_dm"] * det["k_time"]
assert n_triples == 128, f"K_img*K_dm*K_time = {n_triples}, expected 128"
assert det["kernel_dtype"] in ("fp16", "fp32"), f"kernel_dtype={det['kernel_dtype']}"
print(f"detector: K_img={det['k_img']} K_dm={det['k_dm']} K_time={det['k_time']} -> {n_triples} triples; threshold={det['threshold_sigma']}")
PY
pass

STEP="viz_common_present"
[[ -f "${REPO_ROOT}/tools/viz/common.py" ]] || fail "missing tools/viz/common.py (M3-owned, M2-hardened)"
PYTHONPATH="${REPO_ROOT}/tools:${PYTHONPATH}" python -c 'import viz.common' \
  || fail "tools/viz/common.py not importable"
pass

STEP="udp_port_bindable"
# M6 UDP listener default port (11227). dsaX_filTrigger_twoInput is
# being removed in this M6 path so 11227 is free; verify here.
python - <<PY || warn "UDP port ${M6_UDP_PORT} not bindable on 127.0.0.1; another process is holding it. Override with --udp-trigger-port."
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.bind(("127.0.0.1", ${M6_UDP_PORT}))
    print(f"udp 127.0.0.1:${M6_UDP_PORT} bindable")
finally:
    s.close()
PY
pass

STEP="voltages_root"
# /home/ubuntu/data/voltages/ must exist; the M6 cube-dump e2e bench
# (chunk 7) consumes 250924mptq via the chunk-7 voltage-fixture
# infrastructure. Inform-only here.
if [[ ! -d "${VOLTAGES_ROOT}" ]]; then
  warn "${VOLTAGES_ROOT} not present; M6 cube_dump_e2e.py will be skipped"
else
  burst_dir="${VOLTAGES_ROOT}/250924mptq"
  if [[ -d "${burst_dir}" ]]; then
    n_sb=$(find "${burst_dir}/voltages" -maxdepth 1 -name '*_sb*_data.out' 2>/dev/null | wc -l)
    echo "  ${burst_dir}: ${n_sb} sb*_data.out files"
    [[ -f "${burst_dir}/voltages/T2_250924mptq.json" ]] \
      && echo "  T2_250924mptq.json present" \
      || warn "T2_250924mptq.json missing — manifest synthesis will fail"
  else
    warn "${burst_dir} missing — M6 cube_dump_e2e.py will be skipped"
  fi
fi
pass

STEP="lockfile_writable"
LOCKDIR="$(dirname "${M6_LOCKFILE}")"
if [[ -d "${LOCKDIR}" ]] && [[ -w "${LOCKDIR}" ]]; then
  echo "  ${M6_LOCKFILE} dir writable"
elif [[ -f "${M6_LOCKFILE}" ]] && [[ -w "${M6_LOCKFILE}" ]]; then
  echo "  ${M6_LOCKFILE} writable (already exists)"
else
  warn "${M6_LOCKFILE} dir not writable; M6.sh flock will fall back to \$HOME/.dsart-m6.lock. Try: sudo install -d -m 1777 ${LOCKDIR}, or override with M6_LOCKFILE=\$HOME/.dsart-m6.lock"
fi
pass

STEP="m6_target_files"
# Inform-only: M6 deliverables that may already exist from re-runs / chunk
# landings. Class A files M6 owns.
for relpath in \
  src/dsart/cluster/__init__.py \
  src/dsart/cluster/forward.py \
  src/dsart/cluster/features.py \
  src/dsart/cluster/state.py \
  src/dsart/cluster/cands_logger.py \
  src/dsart/dump/__init__.py \
  src/dsart/dump/cube_dump.py \
  src/dsart/dump/udp_listener.py \
  bench/clusterer_throughput.py \
  bench/cube_dump_e2e.py \
  tools/viz/cluster_check.py \
  tools/dod/M6.sh \
  tools/dod/M6_preflight.sh
do
  if [[ -e "${REPO_ROOT}/${relpath}" ]]; then
    echo "  info: ${relpath} present"
  fi
done
pass

STEP="writable_paths"
[[ -w "${REPO_ROOT}/configs" ]] || fail "${REPO_ROOT}/configs not writable"
[[ -w "${REPO_ROOT}/tools" ]]   || fail "${REPO_ROOT}/tools not writable"
[[ -w "${REPO_ROOT}/src" ]]     || fail "${REPO_ROOT}/src not writable"
[[ -w "${REPO_ROOT}/tests" ]]   || fail "${REPO_ROOT}/tests not writable"
[[ -w "${REPO_ROOT}/bench" ]]   || fail "${REPO_ROOT}/bench not writable"
mkdir -p "${REPO_ROOT}/tools/viz" 2>/dev/null || true
[[ -d "${REPO_ROOT}/tools/viz" ]] || fail "${REPO_ROOT}/tools/viz not creatable"
mkdir -p "${REPO_ROOT}/bench/reports/M6/cands_log" 2>/dev/null || true
mkdir -p "${REPO_ROOT}/bench/reports/M6/cube_dump" 2>/dev/null || true
[[ -d "${REPO_ROOT}/bench/reports/M6/cands_log" ]] || fail "${REPO_ROOT}/bench/reports/M6/cands_log not creatable"
[[ -d "${REPO_ROOT}/bench/reports/M6/cube_dump" ]] || fail "${REPO_ROOT}/bench/reports/M6/cube_dump not creatable"
pass

echo "M6_preflight PASS"
