#!/usr/bin/env bash
# Standalone A/B repro for the ~368-cube corr_fast stall (Option A wire-in,
# 2026-06-03 production regression).
#
# Sets up a private fada-shaped PSRDADA ring (key c0d0), feeds it with
# bench.replay_voltage_dump --synthesize at native cadence, runs corr_fast
# with --stage2-mode {uniform | per_coarse_dm}, and watches for the loop to
# stall. On stall it dumps:
#   * py-spy stack of corr_fast
#   * dada_dbmetric on the source ring
#   * GPU memory + driver state
# then SIGTERMs everything.
#
# UDP TX is pointed at 127.0.0.1 on four ports; no listener is required
# because UDP sendto with no receiver just gets discarded by the kernel
# (no back-pressure, no ICMP unreachable in the loopback path).
#
# Usage:
#   bench/option_a_stall_repro.sh {uniform | per_coarse_dm} [stall_timeout_s]
#
# Examples:
#   bench/option_a_stall_repro.sh uniform 120
#   bench/option_a_stall_repro.sh per_coarse_dm 120
#
# The script must be run from the dsa110-rt repo root on a corr node with
# the dsa110-rt conda env. It uses the same cal blob path as production.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODE [stall_timeout_s]" >&2
    echo "  MODE = uniform | per_coarse_dm" >&2
    exit 2
fi
MODE="$1"
STALL_TIMEOUT_S="${2:-90}"

case "$MODE" in
    uniform|per_coarse_dm) ;;
    *) echo "bad mode: $MODE; want uniform or per_coarse_dm" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python
PY_SPY=/home/ubuntu/miniforge3/envs/dsa110-rt/bin/py-spy

OUT_DIR="/tmp/option_a_stall_repro/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

# private fada ring; mirror prod fada (288 MB blocks, 70 deep, 2 readers).
KEY=c0d0
N_BLOCKS_SRC=2000          # source writes plenty; bench stops on corr_fast side
N_BUFS=70
BUFSZ=301989888

CHGROUP=0                  # n03 → chgroup 0 (the worst-case for stage-2 shifts)
DEC_DEG=54.5734
CAL=/home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out
FLAGANTS=/home/ubuntu/data/voltages/250924mptq/cals/flagants.dat
DM_PLAN=/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz

SRC_LOG="$OUT_DIR/source.log"
CORR_LOG="$OUT_DIR/corr_fast.log"
METRIC_LOG="$OUT_DIR/dada_dbmetric.log"
PYSPY_LOG="$OUT_DIR/py-spy.dump"
GPU_LOG="$OUT_DIR/nvidia-smi.log"
SUMMARY="$OUT_DIR/summary.txt"

cleanup_done=0
cleanup() {
    [[ "$cleanup_done" -eq 1 ]] && return
    cleanup_done=1
    echo "[$(date -u +%FT%TZ)] cleanup: tearing down" | tee -a "$SUMMARY"
    if [[ -n "${CORR_PID:-}" ]]; then
        kill -TERM "$CORR_PID" 2>/dev/null || true
    fi
    if [[ -n "${SRC_PID:-}" ]]; then
        kill -TERM "$SRC_PID" 2>/dev/null || true
    fi
    sleep 2
    if [[ -n "${CORR_PID:-}" ]]; then kill -KILL "$CORR_PID" 2>/dev/null || true; fi
    if [[ -n "${SRC_PID:-}" ]]; then kill -KILL "$SRC_PID" 2>/dev/null || true; fi
    dada_db -k "$KEY" -d 2>>"$SUMMARY" || true
    echo "[$(date -u +%FT%TZ)] cleanup: done" | tee -a "$SUMMARY"
}
trap cleanup EXIT INT TERM

echo "[$(date -u +%FT%TZ)] starting repro: mode=$MODE timeout=${STALL_TIMEOUT_S}s out=$OUT_DIR" | tee "$SUMMARY"

# 1. create private fada ring
echo "[$(date -u +%FT%TZ)] alloc dada_db -k $KEY -b $BUFSZ -n $N_BUFS -r 2" | tee -a "$SUMMARY"
dada_db -k "$KEY" -d 2>/dev/null || true     # tear down stale ring if any
dada_db -k "$KEY" -b "$BUFSZ" -n "$N_BUFS" -r 2 2>&1 | tee -a "$SUMMARY"

# 2. spawn synth source (fada-format synthetic voltages at native cadence).
echo "[$(date -u +%FT%TZ)] spawn synth source -> key=$KEY" | tee -a "$SUMMARY"
"$PY" -u -m bench.replay_voltage_dump \
    --synthesize \
    --n-blocks "$N_BLOCKS_SRC" \
    --synth-thermal-sigma 1.5 \
    --fada-key "$KEY" \
    --rate native \
    --seed 12345 \
    >"$SRC_LOG" 2>&1 &
SRC_PID=$!
echo "  source pid=$SRC_PID" | tee -a "$SUMMARY"

# 3. spawn dada_dbmetric poller (1 Hz, monotone tag for grep'ing later).
(
    while true; do
        ts=$(date -u +%FT%TZ)
        echo "=== $ts ==="
        dada_dbmetric -k "$KEY" 2>&1 || true
        sleep 1
    done
) >"$METRIC_LOG" 2>&1 &
METRIC_PID=$!
trap "kill $METRIC_PID 2>/dev/null; cleanup" EXIT INT TERM

# Give the source ~3 s of head-start so getNextPage doesn't block before
# the writer has stamped the header (which is needed for Reader.connect).
sleep 3

# 4. spawn corr_fast with the same args as prod, except:
#    - --fada-key c0d0 (private ring)
#    - --transport-tx-host 127.0.0.1 + worker-hosts all loopback (UDP drain)
#    - no ready-sentinel-path (we don't have an orchestrator gating us)
#    - --max-blocks 0 (unlimited; stall detector decides when to stop)
echo "[$(date -u +%FT%TZ)] spawn corr_fast --stage2-mode $MODE" | tee -a "$SUMMARY"
"$PY" -u -m dsart.services.corr_fast_integration \
    --fada-key "$KEY" \
    --chgroup "$CHGROUP" \
    --obs-dec-deg "$DEC_DEG" \
    --device cuda:0 \
    --apply-cal "$CAL" \
    --cal-mode phase_only \
    --flagants "$FLAGANTS" \
    --sk-far 1e-4 \
    --bandpass-k 5.0 \
    --group-k 5.0 \
    --sumthr-max-m 8 \
    --sumthr-eta 1.5 \
    --rfi-m-values 64,256,1024,4096 \
    --t-int-fast-native 32 \
    --dm-plan-path "$DM_PLAN" \
    --chan-sum-factor 8 \
    --sliding-window \
    --n-grid 256 \
    --transport-tx-mode prod \
    --transport-tx-workers 4 \
    --transport-tx-host 127.0.0.1 \
    --transport-tx-base-port 16625 \
    --transport-tx-coarse-dm-mask 0xFF \
    --transport-tx-worker-hosts 127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
    --stage2-mode "$MODE" \
    --max-blocks 0 \
    --output-dir "$OUT_DIR/grid" \
    --blocks-output-mode none \
    --log-level INFO \
    >"$CORR_LOG" 2>&1 &
CORR_PID=$!
echo "  corr_fast pid=$CORR_PID" | tee -a "$SUMMARY"

# 5. wait + watch for stall. Re-check the "processed n_in=" timestamp every
# 5 s; if no new line has appeared in $STALL_TIMEOUT_S seconds AND we've
# already seen at least one "processed n_in=" line, declare stall.
last_processed_mtime=0
last_n_in=0
stall_at_n_in=""
last_seen_t=$(date +%s)

while kill -0 "$CORR_PID" 2>/dev/null; do
    sleep 5
    # latest "processed n_in=N" line, if any
    latest=$(grep -E 'processed n_in=' "$CORR_LOG" | tail -1 || true)
    if [[ -n "$latest" ]]; then
        cur_n_in=$(echo "$latest" | sed -E 's/.*n_in=([0-9]+).*/\1/')
        if [[ "$cur_n_in" -gt "$last_n_in" ]]; then
            last_n_in="$cur_n_in"
            last_seen_t=$(date +%s)
        fi
    fi
    now_t=$(date +%s)
    idle=$((now_t - last_seen_t))
    if [[ "$last_n_in" -gt 0 && "$idle" -gt "$STALL_TIMEOUT_S" ]]; then
        stall_at_n_in="$last_n_in"
        echo "[$(date -u +%FT%TZ)] STALL DETECTED: no progress for ${idle}s after n_in=$last_n_in" | tee -a "$SUMMARY"
        break
    fi
    # also stop if we hit a healthy plateau >> the prod stall point
    if [[ "$last_n_in" -ge 600 ]]; then
        echo "[$(date -u +%FT%TZ)] reached n_in=$last_n_in > 600 with no stall; declaring HEALTHY" | tee -a "$SUMMARY"
        break
    fi
done

# 6. diagnostics on stall (or healthy stop).
echo "[$(date -u +%FT%TZ)] capturing diagnostics" | tee -a "$SUMMARY"
nvidia-smi >"$GPU_LOG" 2>&1 || true
if [[ -n "$stall_at_n_in" ]]; then
    echo "[$(date -u +%FT%TZ)] py-spy dump corr_fast pid=$CORR_PID" | tee -a "$SUMMARY"
    sudo "$PY_SPY" dump --pid "$CORR_PID" >"$PYSPY_LOG" 2>&1 || \
        "$PY_SPY" dump --pid "$CORR_PID" >"$PYSPY_LOG" 2>&1 || true
    # extra: try to read /proc/PID/status, /proc/PID/stack
    cat "/proc/$CORR_PID/status" >>"$PYSPY_LOG" 2>&1 || true
    cat "/proc/$CORR_PID/stack"  >>"$PYSPY_LOG" 2>&1 || true
    cat "/proc/$CORR_PID/wchan"  >>"$PYSPY_LOG" 2>&1 || true
    echo                          >>"$PYSPY_LOG"
    ls -d /proc/"$CORR_PID"/task/* | while read tdir; do
        tid=$(basename "$tdir")
        echo "--- thread $tid ---" >>"$PYSPY_LOG"
        cat "$tdir/comm" 2>/dev/null >>"$PYSPY_LOG" || true
        cat "$tdir/wchan" 2>/dev/null >>"$PYSPY_LOG" || true
        echo                          >>"$PYSPY_LOG"
        cat "$tdir/stack" 2>/dev/null >>"$PYSPY_LOG" || true
        echo                          >>"$PYSPY_LOG"
    done
fi

# 7. summary table.
{
    echo
    echo "============================================================"
    echo "REPRO SUMMARY"
    echo "  mode=$MODE"
    echo "  out=$OUT_DIR"
    echo "  stall_detected=${stall_at_n_in:-no}"
    echo "  last_n_in=$last_n_in"
    echo "  last_processed_line=$(grep -E 'processed n_in=' "$CORR_LOG" | tail -1 || true)"
    echo "  corr_fast pid alive=$(kill -0 "$CORR_PID" 2>/dev/null && echo yes || echo no)"
    echo
    echo "  last 20 lines of corr_fast log:"
    tail -20 "$CORR_LOG" | sed 's/^/    /'
    echo
    echo "  last dbmetric:"
    tail -10 "$METRIC_LOG" | sed 's/^/    /'
    echo "============================================================"
} | tee -a "$SUMMARY"

# 8. teardown.
kill $METRIC_PID 2>/dev/null || true
cleanup
