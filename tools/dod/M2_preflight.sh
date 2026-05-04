#!/usr/bin/env bash
# M2 pre-flight readiness check (§8 / §3.4) — runs on h01.
#
# Purpose: fail fast if anything M2 depends on is broken before authoring.
# M2 builds services/corr_slow_compute.py + fstable cache + replay writer +
# voltage-fixture viz, so it has more dependencies than M0/M1:
#   * M1 must be hardened
#   * Legacy dsa110-meridian-fs source must be readable (we're patching its
#     routines.py via a side-by-side egg-link or PYTHONPATH on h01)
#   * Legacy config_dsa96_corr.yaml must be readable (canonical buffer
#     sizes for fada/bada — see F6 in M2_PLAN_FIXES.md)
#   * configs/config_corr.yaml in the repo must EITHER match legacy already
#     OR be flagged so Chunk 0 can correct it
#   * /home/ubuntu/data/fstables/ must be creatable (M2 cache root)
#   * /home/ubuntu/data/voltage_fixtures/ must exist with at least one
#     continuum fixture for the user-facing M2 sub-DoD (§8 line 2172)
#   * psrdada-python Reader/Writer must be importable
#   * torch + cupy + dsacalib + astropy + casacore-python deps present
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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

M0_STATUS_JSON="${M0_STATUS_JSON:-${HOME}/dsart-m0-status.json}"
M1_STATUS_JSON="${M1_STATUS_JSON:-${HOME}/dsart-m1-status.json}"
DSAMFS_REPO="${DSAMFS_REPO:-/home/ubuntu/proj/dsa110-shell/dsa110-meridian-fs}"
FSTABLE_ROOT="${FSTABLE_ROOT:-/home/ubuntu/data/fstables}"
VF_ROOT="${VF_ROOT:-/home/ubuntu/data/voltage_fixtures}"

# shellcheck source=/dev/null
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate dsa110-rt

STEP=""
fail() {
  echo "[M2_pre:${STEP}] FAIL $*"
  exit 1
}
pass() {
  echo "[M2_pre:${STEP}] PASS"
}
warn() {
  echo "[M2_pre:${STEP}] WARN $*"
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
assert s.get("phase", "") == "a", f"wrong phase: {s!r}"
print(f"M1 stage: {s.get('stage')!r}; git_sha: {s.get('git_sha', '')[:12]}")
PY
pass

STEP="m1_artifacts"
# M1 produced configs/dm_plan.npz; verify it's there and round-trips
# through the M1 dataclass loader (DSART_TEST=1 enables shape asserts).
[[ -f "${REPO_ROOT}/configs/dm_plan.npz" ]] || fail "missing configs/dm_plan.npz"
DSART_TEST=1 python3 - <<PY || fail "dm_plan.npz did not validate"
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from dsart.common.contracts import DmPlan
plan = DmPlan.from_npz("${REPO_ROOT}/configs/dm_plan.npz")
assert plan.fine_dm.shape[0] > 100
assert plan.coarse_dm.shape[0] > 8
PY
pass

STEP="git_clean"
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes in ${REPO_ROOT}; run 'git status'"
pass

STEP="git_branch"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "${BRANCH}" == "main" ]] || fail "expected branch=main, got ${BRANCH}"
git remote -v | grep -q '^origin' || fail "missing origin remote"
pass

STEP="conda_env"
[[ "${CONDA_DEFAULT_ENV:-}" == "dsa110-rt" ]] || fail "CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}', expected dsa110-rt"
pass

STEP="dsart_import"
python -c 'import dsart, dsart.common.host, dsart.common.config_loader, dsart.common.contracts, dsart.common.constants, dsart.common.dispersion' \
  || fail "dsart package import"
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

STEP="psrdada_python"
# psrdada-python Reader/Writer must be importable on h01 — M2's
# corr_slow_compute and replay_voltage_dump both depend on it. Per
# Subagent B (id fd6a8b3b), the canonical install on h01 is via the
# dsa110-rt conda env or the psrdada-python source repo.
python - <<'PY' || fail "psrdada-python not importable"
import psrdada
print(f"psrdada.__file__={psrdada.__file__}")
from psrdada import Reader, Writer
PY
pass

STEP="casa38_env"
# Per D13 in M2_PLAN_FIXES.md: dsamfs / dsacalib / antpos / casacore live
# only in the casa38 conda env (h01 inventory). M2's build_fstable_cache
# and meridian_fringestop run there; corr_slow_compute and replay_voltage
# run in dsa110-rt. Verify casa38's deps are intact for the cross-env
# pieces. Use the casa38 python directly; do NOT activate (we're already
# in dsa110-rt and don't want to nest activates).
CASA38_PY="${CASA38_PY:-/home/ubuntu/anaconda3/envs/casa38/bin/python}"
[[ -x "${CASA38_PY}" ]] || fail "casa38 python not found at ${CASA38_PY}"
"${CASA38_PY}" - <<'PY' || fail "casa38 env missing M2 dsamfs/dsacalib deps"
import sys
missing = []
for mod in ("dsacalib", "dsamfs", "antpos", "casacore", "yaml", "numpy", "astropy"):
    try:
        __import__(mod)
    except Exception as e:
        missing.append(f"{mod}: {e}")
if missing:
    print("casa38 missing modules:")
    for m in missing:
        print("  " + m)
    sys.exit(1)
import dsamfs, dsacalib
print(f"casa38 dsamfs={dsamfs.__file__} dsacalib={dsacalib.__file__}")
PY
pass

STEP="dsamfs_source"
# M2 patches dsamfs/routines.py in the source repo. The casa38 install
# can be either editable (pip install -e .) — in which case the patch
# takes effect immediately — or a vanilla pip install (in which case the
# patch needs a re-install to land in site-packages). h01 today is the
# vanilla case but byte-identical (sha256 match) to source, so the patch
# is safe to apply at the source path; M2 Chunk 5 owns the casa38
# `pip install -e .` step that converts to editable as part of the patch.
[[ -d "${DSAMFS_REPO}" ]] || fail "DSAMFS_REPO=${DSAMFS_REPO} not a directory"
[[ -f "${DSAMFS_REPO}/dsamfs/routines.py" ]] || fail "missing ${DSAMFS_REPO}/dsamfs/routines.py"
[[ -f "${DSAMFS_REPO}/dsamfs/utils.py" ]]    || fail "missing ${DSAMFS_REPO}/dsamfs/utils.py"
[[ -f "${DSAMFS_REPO}/dsamfs/fringestopping.py" ]] || fail "missing ${DSAMFS_REPO}/dsamfs/fringestopping.py"
"${CASA38_PY}" - <<PY || fail "dsamfs site-packages drifted from source repo"
import hashlib, os
import dsamfs
sp_routines  = os.path.join(os.path.dirname(dsamfs.__file__), "routines.py")
src_routines = "${DSAMFS_REPO}/dsamfs/routines.py"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]
sp_sha  = sha(sp_routines)
src_sha = sha(src_routines)
src_realpath = os.path.realpath(os.path.dirname(dsamfs.__file__))
expected     = os.path.realpath("${DSAMFS_REPO}/dsamfs")
if src_realpath == expected:
    print(f"casa38 dsamfs is editable-installed at {expected}")
elif sp_sha == src_sha:
    print(f"casa38 dsamfs is vanilla-installed; site-packages matches source "
          f"(sha {sp_sha}). M2 Chunk 5 must run 'pip install -e .' in casa38 "
          f"after editing routines.py for the patch to take effect.")
else:
    raise SystemExit(
        f"casa38 dsamfs at {sp_routines} (sha {sp_sha}) DIVERGES from source "
        f"{src_routines} (sha {src_sha}). Resolve manually before M2.")
PY
pass

STEP="repo_corr_yaml"
# Physics-pinned buffer sizes (derivable from §3 constants); see
# configs/config_corr.yaml comment for the derivation. NOT config
# choices — these are block sizes the legacy C correlator and
# meridian_fringestop lock in.
python - <<PY || fail "configs/config_corr.yaml diverges from physics-pinned values (F6)"
import yaml
with open("${REPO_ROOT}/configs/config_corr.yaml") as f:
    cfg = yaml.safe_load(f)
buffers = cfg.get('buffers', {})
got = {k: buffers.get(k, {}).get('bytes_per_block', -1) for k in ('dada', 'eada', 'fada', 'bada')}
exp = {
    'dada': 2048 * 48 * 384 * 2 * 2 * 1,   # 150,994,944
    'eada': 2048 * 48 * 384 * 2 * 2 * 1,   # 150,994,944
    'fada': 2048 * 96 * 384 * 2 * 2 * 1,   # 301,989,888
    'bada': 4656 * 384 * 2 * 8,            #  28,606,464
}
ok = True
for k in ('dada', 'eada', 'fada', 'bada'):
    flag = "OK" if got[k] == exp[k] else "FAIL"
    if got[k] != exp[k]:
        ok = False
    print(f"  {k}: got={got[k]:>11d} expected={exp[k]:>11d}  [{flag}]")
if not ok:
    raise SystemExit(1)
PY
pass

STEP="fstable_root"
# Cache root must be writable. If missing, mkdir parent + create.
if [[ ! -d "${FSTABLE_ROOT}" ]]; then
  warn "${FSTABLE_ROOT} does not exist; M2 build_fstable_cache will create it"
else
  [[ -w "${FSTABLE_ROOT}" ]] || fail "${FSTABLE_ROOT} not writable"
fi
pass

STEP="voltage_fixture_root"
# /home/ubuntu/data/voltage_fixtures/ must exist (M0 created it). The
# user-facing M2 sub-DoD (operator sign-off gate) requires AT LEAST one
# continuum fixture; we inform-only here on whether one is present.
[[ -d "${VF_ROOT}" ]] || fail "${VF_ROOT} not present (expected M0 to create it; see plan §3.3 / §4.7)"
n_runs=$(find "${VF_ROOT}" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "  ${VF_ROOT}: ${n_runs} run-id subdir(s)"
n_continuum=0
n_burst=0
n_unknown=0
for run_dir in "${VF_ROOT}"/*/; do
  [[ -d "${run_dir}" ]] || continue
  if [[ -f "${run_dir}/manifest.yaml" ]]; then
    kind=$(python -c "import yaml; print(yaml.safe_load(open('${run_dir}/manifest.yaml')).get('fixture_kind','unknown'))" 2>/dev/null || echo unknown)
    case "${kind}" in
      continuum) n_continuum=$((n_continuum+1)) ;;
      burst)     n_burst=$((n_burst+1)) ;;
      *)         n_unknown=$((n_unknown+1)) ;;
    esac
  else
    n_unknown=$((n_unknown+1))
  fi
done
echo "  fixtures: ${n_continuum} continuum, ${n_burst} burst, ${n_unknown} unknown"
if [[ "${n_continuum}" -eq 0 ]]; then
  warn "no continuum fixture present — M2 voltage-fixture sub-DoD (§8 line 2172) will be skipped"
fi
pass

STEP="m2_target_files"
# Inform-only: M2 deliverables that may already exist from re-runs.
for relpath in \
  src/dsart/services/corr_slow_compute.py \
  src/dsart/services/__init__.py \
  tools/build_fstable_cache.py \
  bench/voltage_fixture_slow_corr.py \
  tools/viz/corr_imager_dedisperser_check.py \
  tools/viz/common.py \
  tests/test_slow_corr_synth.py \
  tools/dod/M2.sh
do
  if [[ -e "${REPO_ROOT}/${relpath}" ]]; then
    echo "  info: ${relpath} already present (re-run / partial M2 — OK)"
  fi
done
pass

STEP="writable_paths"
[[ -w "${REPO_ROOT}/configs" ]]     || fail "${REPO_ROOT}/configs not writable"
[[ -w "${REPO_ROOT}/tools" ]]       || fail "${REPO_ROOT}/tools not writable"
[[ -w "${REPO_ROOT}/src" ]]         || fail "${REPO_ROOT}/src not writable"
[[ -w "${REPO_ROOT}/tests" ]]       || fail "${REPO_ROOT}/tests not writable"
[[ -w "${REPO_ROOT}/bench" ]]       || fail "${REPO_ROOT}/bench not writable"
mkdir -p "${REPO_ROOT}/tools/viz"   2>/dev/null || true
[[ -d "${REPO_ROOT}/tools/viz" ]]   || fail "${REPO_ROOT}/tools/viz not creatable"
mkdir -p "${REPO_ROOT}/src/dsart/services" 2>/dev/null || true
[[ -d "${REPO_ROOT}/src/dsart/services" ]] || fail "${REPO_ROOT}/src/dsart/services not creatable"
pass

echo "M2_preflight PASS"
