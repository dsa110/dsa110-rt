#!/usr/bin/env bash
#
# Build the per-DEC fringe-stopping table bank and deploy it to the corr nodes.
#
# This is a thin orchestrator around two existing pieces:
#   1. tools/build_fstable_cache.py  — builds one .npz per DEC (run under the
#      casa38 conda env, sourcing params from live etcd so the cached tables
#      match meridian_fringestop's runtime asserts → guaranteed cache hit).
#   2. rsync over ssh               — fans the bank out to every corr node's
#      production read path (/home/ubuntu/data/fstables), where
#      meridian_fringestop loads them at startup.
#
# Run it ON h23 (it has etcd, casa38, and ssh to the corr nodes). The build is
# resumable: existing tables are skipped unless --force, so a re-run only fills
# gaps. The default DEC range is the one requested: -10 .. 89.0 deg, step 0.25.
#
# Usage:
#   tools/build_and_deploy_fstables.sh [options]
#
# Options:
#   --dec-min DEG     low DEC, inclusive            (default -10)
#   --dec-max DEG     high DEC, inclusive           (default 89.0)
#   --dec-step DEG    grid step                     (default 0.25, the runtime grid)
#   --output-dir DIR  h23 master dir for the bank   (default <repo>/var/fstables)
#   --casa38-py PATH  python that has dsamfs/dsacalib/dsautils
#                     (default /home/ubuntu/anaconda3/envs/casa38/bin/python)
#   --hosts "a b c"   override the corr-node list (space-separated FQDNs)
#   --jobs N          parallel rsync fan-out width  (default: one per host)
#   --build-only      build the bank, do not deploy
#   --deploy-only     skip the build, just deploy what's already in --output-dir
#   --force           rebuild tables that already exist (passes --force to builder)
#   --dry-run         show what would happen; build a 5-DEC preview, no deploy
#   -h, --help        this help
#
# The full -10..89 / 0.25 grid is 397 tables; the build is the slow part
# (minutes per table on a 96-antenna array → hours total). Deploy is seconds.
set -uo pipefail

# --- defaults --------------------------------------------------------------
DEC_MIN="-10"
DEC_MAX="89.0"
DEC_STEP="0.25"
CASA38_PY="/home/ubuntu/anaconda3/envs/casa38/bin/python"
CORR_FSTABLE_DIR="/home/ubuntu/data/fstables"     # meridian_fringestop read path
JOBS=""
BUILD=1
DEPLOY=1
FORCE=0
DRY_RUN=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SCRIPT="${REPO_ROOT}/tools/build_fstable_cache.py"
OUTPUT_DIR="${REPO_ROOT}/var/fstables"            # h23 master copy (gitignored)

# The 16 corr-node FQDNs, mirroring
# tools/dashboard/dsa_monitor/services_inventory.py::_CORR_CN_IDS
# (cn ids 3-8,10-12,14-16,18,19,21,22 → nNN.pro.pvt). Override with --hosts.
CORR_HOSTS=(
  n03.pro.pvt n04.pro.pvt n05.pro.pvt n06.pro.pvt n07.pro.pvt n08.pro.pvt
  n10.pro.pvt n11.pro.pvt n12.pro.pvt
  n14.pro.pvt n15.pro.pvt n16.pro.pvt
  n18.pro.pvt n19.pro.pvt
  n21.pro.pvt n22.pro.pvt
)

SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes)

usage() { sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dec-min)    DEC_MIN="${2:?}"; shift 2;;
    --dec-max)    DEC_MAX="${2:?}"; shift 2;;
    --dec-step)   DEC_STEP="${2:?}"; shift 2;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2;;
    --casa38-py)  CASA38_PY="${2:?}"; shift 2;;
    --hosts)      read -r -a CORR_HOSTS <<< "${2:?}"; shift 2;;
    --jobs)       JOBS="${2:?}"; shift 2;;
    --build-only) DEPLOY=0; shift;;
    --deploy-only) BUILD=0; shift;;
    --force)      FORCE=1; shift;;
    --dry-run)    DRY_RUN=1; shift;;
    -h|--help)    usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

: "${JOBS:=${#CORR_HOSTS[@]}}"

echo "==> repo:        $REPO_ROOT"
echo "==> output dir:  $OUTPUT_DIR  (h23 master)"
echo "==> corr path:   $CORR_FSTABLE_DIR  (on each corr node)"
echo "==> DEC grid:    ${DEC_MIN} .. ${DEC_MAX} step ${DEC_STEP}"
echo "==> corr hosts:  ${#CORR_HOSTS[@]}  (${CORR_HOSTS[*]})"
echo

mkdir -p "$OUTPUT_DIR"

# --- build -----------------------------------------------------------------
if [[ "$BUILD" == "1" ]]; then
  if [[ ! -x "$CASA38_PY" && ! -f "$CASA38_PY" ]]; then
    echo "ERROR: casa38 python not found: $CASA38_PY (pass --casa38-py)" >&2
    exit 1
  fi
  if [[ ! -f "$BUILD_SCRIPT" ]]; then
    echo "ERROR: build script not found: $BUILD_SCRIPT" >&2
    exit 1
  fi
  build_args=(
    "$CASA38_PY" -u "$BUILD_SCRIPT"
    --from-etcd
    --dec-min "$DEC_MIN" --dec-max "$DEC_MAX" --dec-step "$DEC_STEP"
    --output-dir "$OUTPUT_DIR"
  )
  [[ "$FORCE" == "1" ]] && build_args+=(--force)
  [[ "$DRY_RUN" == "1" ]] && build_args+=(--dry-run)

  echo "==> building bank:"
  echo "    ${build_args[*]}"
  if "${build_args[@]}"; then
    echo "==> build step finished OK"
  else
    rc=$?
    echo "WARNING: build step returned rc=$rc (some DECs may have failed)." >&2
    echo "         Deploying whatever was produced; re-run to fill gaps." >&2
  fi
  echo
fi

# --- deploy ----------------------------------------------------------------
if [[ "$DEPLOY" != "1" ]]; then
  echo "==> --build-only: skipping deploy."
  exit 0
fi

n_master=$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
echo "==> master bank has ${n_master} .npz file(s) in $OUTPUT_DIR"
if [[ "$n_master" == "0" ]]; then
  echo "ERROR: nothing to deploy — no .npz in $OUTPUT_DIR." >&2
  echo "       Build first (drop --deploy-only), or check --output-dir." >&2
  exit 1
fi

deploy_one() {  # <host> <status_file>
  local host="$1" sf="$2" rc
  ssh "${SSH_OPTS[@]}" -n "$host" "mkdir -p '$CORR_FSTABLE_DIR'" 2>>"$sf.err"
  rsync -a --partial \
    -e "ssh ${SSH_OPTS[*]}" \
    --include='*.npz' --exclude='*' \
    "$OUTPUT_DIR/" "$host:$CORR_FSTABLE_DIR/" >>"$sf.err" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    local cnt
    cnt=$(ssh "${SSH_OPTS[@]}" -n "$host" \
            "find '$CORR_FSTABLE_DIR' -maxdepth 1 -name '*.npz' -type f 2>/dev/null | wc -l" \
          2>>"$sf.err" | tr -d ' ')
    echo "ok ${cnt:-?}" > "$sf"
  else
    echo "fail rc=$rc" > "$sf"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "==> [dry-run] would rsync ${OUTPUT_DIR}/*.npz to ${#CORR_HOSTS[@]} hosts:"
  for h in "${CORR_HOSTS[@]}"; do
    echo "    rsync -a --include='*.npz' --exclude='*' $OUTPUT_DIR/ $h:$CORR_FSTABLE_DIR/"
  done
  exit 0
fi

echo "==> deploying to ${#CORR_HOSTS[@]} corr node(s) (parallel x${JOBS}) ..."
tmpd="$(mktemp -d)"
trap 'rm -rf "$tmpd"' EXIT

i=0
for host in "${CORR_HOSTS[@]}"; do
  deploy_one "$host" "$tmpd/$host" &
  i=$((i + 1))
  if (( i % JOBS == 0 )); then wait; fi
done
wait

# --- summary ---------------------------------------------------------------
echo
echo "==> deploy summary (master has ${n_master}):"
n_ok=0; n_fail=0
for host in "${CORR_HOSTS[@]}"; do
  sf="$tmpd/$host"
  if [[ -f "$sf" ]]; then
    read -r status detail < "$sf"
  else
    status="fail"; detail="no-result"
  fi
  if [[ "$status" == "ok" ]]; then
    n_ok=$((n_ok + 1))
    printf "    %-16s OK   (%s .npz on node)\n" "$host" "$detail"
  else
    n_fail=$((n_fail + 1))
    printf "    %-16s FAIL (%s)\n" "$host" "$detail"
    [[ -s "$sf.err" ]] && sed 's/^/        /' "$sf.err" | tail -3
  fi
done

echo
echo "==> done: ${n_ok} ok, ${n_fail} failed (of ${#CORR_HOSTS[@]})"
[[ "$n_fail" -eq 0 ]] || exit 1
