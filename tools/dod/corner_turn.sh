#!/usr/bin/env bash
# corner_turn.sh — full-fleet 16→4 corner-turn bench at production per-pair rate.
#
# This is the fleet-wide multi-corr fan-in test that the M4b core DoD report
# called out as the natural follow-up once Phase-2 fan-out reached ≥ 4 corr
# nodes (now Phase-2 complete on all 18 of n01..n22 minus n17/n20). It
# extends bench/net_pair.py's single-pair test to the production topology
# documented in REALTIME_FRB_SEARCH.md §1:
#
#     16 corr (TX) nodes  ──UDP──►  4 search (RX) nodes
#
#     TX (corr):  n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22
#     RX (search): n01 n02 n09 n13
#
#     Each TX → all 4 RX  (4 outgoing streams per TX)
#     Each RX → all 16 TX (16 incoming streams per RX)
#     Total fabric: 64 unidirectional UDP streams
#
#     Per-pair rate: 6 dm_idx flows × 0.073 Gb/s/flow = 0.44 Gb/s
#                    (matches plan §11 line 2654 per-pair production rate)
#     Per-RX aggregate ingress: 16 × 0.44 ≈ 7.0 Gb/s
#                    (matches M4a chunk-6 C-epoll RX ceiling on h01 loopback)
#     Total fabric load: 64 × 0.44 ≈ 28.16 Gb/s
#
# Port plan (per REALTIME_FRB_SEARCH.md §1, dbnic convention):
#   port = 6625 + chgroup   where chgroup ∈ 0..15 is the corr-node id.
#   Each RX node listens on 16 ports (one per corr); each TX node uses one
#   port (its chgroup port) when targeting any of the 4 search-node IPs.
#
#     Chgroup → TX node map (per the existing corr-farm ordering):
#       0 → n03   4 → n07   8 → n12  12 → n18
#       1 → n04   5 → n08   9 → n14  13 → n19
#       2 → n05   6 → n10  10 → n15  14 → n21
#       3 → n06   7 → n11  11 → n16  15 → n22
#
# DoD invariants this bench asserts (5-min sustained run):
#   I1  every pair sustains 0.44 ± 0.022 Gb/s (±5%) over 300 s
#   I2  per-pair fragment_loss_estimate < 1e-4
#   I3  per-pair pattern_mismatch_count == 0
#   I4  per-pair tx_dropped_payloads == 0
#         (no upstream backpressure into the TX queue at production rate;
#          c.f. plan §4.3 line 1447 "drop oldest, don't block")
#   I5  each RX aggregate ingress 7.0 ± 0.35 Gb/s (±5%)
#
# All five invariants PASS → "corner-turn PASS at production rate".
#
# Synchronization is via --start-at (a unix UTC float that both sides sleep
# until). T0 = now + 60 s gives the orchestrator a quiet window to ssh-launch
# all 64 processes before the wire starts.
#
# Usage:
#   bash tools/dod/corner_turn.sh                       # default: 300 s, all 16×4
#   bash tools/dod/corner_turn.sh --duration 60         # quick run
#   bash tools/dod/corner_turn.sh --smoke               # 1 TX → 1 RX, 10 s
#   bash tools/dod/corner_turn.sh --tx-list "n03 n04" --rx-list "n01"  # subset
#
# Outputs:
#   ~/dsart-corner-turn-logs/<utc>/                     (on h23)
#     ├── tx_<TX>_to_<RX>.json                          (64 TX counters)
#     ├── rx_<RX>_from_<TX>.json                        (64 RX counters)
#     ├── summary.json                                  (per-pair + aggregates)
#     ├── verdict.txt                                   (PASS/FAIL per invariant)
#     └── *.launch.log                                  (ssh launch transcripts)
#
# Per-node logs (on each node):
#   ~/dsart-ct-tx-<RX>.log                              (TX stdout/err)
#   ~/dsart-ct-rx-<TX>.log                              (RX stdout/err)
#
# Implementation: pure bash on h23 + ssh+nohup launchers. No new code in
# bench/. The bench/net_pair.py CLI already supports everything we need.

set -eo pipefail

# ─── 1. Defaults + arg parsing ─────────────────────────────────────────────

DURATION_S=300
SMOKE=NO
START_LEAD_S=60
RATE_GBPS_PER_FLOW=0.073
N_FLOWS=6
SYNC_TOLERANCE_S=30   # margin we wait past end-of-run before reaping
# Per-RX listen address = each search node's 10.41.0.x data-plane IP (br2).
declare -A RX_IP=(
    [n01]=10.41.0.205
    [n02]=10.41.0.222
    [n09]=10.41.0.253
    [n13]=10.41.0.238
    # ----------------------------------------------------------------------
    # OPTIONAL DEBUG RX TARGETS — corr-hosts on raw enp129s0f0 (no br2).
    # Not part of the production 4-search topology. Used 2026-05-15 to
    # confirm the n01/n02 ipfrag bottleneck was br2-adjacent (15 → n11
    # hit full rate; 16 → n01 then capped at 1 Gb/s before the ipfrag
    # fix). Leaving here as a known-good A/B path; if you change the
    # default --rx-list these are otherwise unused.
    [n03]=10.41.0.224
    [n04]=10.41.0.138
    [n11]=10.41.0.216
    [n22]=10.41.0.233
)
# Per-TX chgroup id (= 6625-offset port).
declare -A TX_CHGROUP=(
    [n03]=0   [n04]=1   [n05]=2   [n06]=3
    [n07]=4   [n08]=5   [n10]=6   [n11]=7
    [n12]=8   [n14]=9   [n15]=10  [n16]=11
    [n18]=12  [n19]=13  [n21]=14  [n22]=15
)
TX_LIST="n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22"
RX_LIST="n01 n02 n09 n13"
PORT_BASE=6625

usage() {
    cat <<USAGE
corner_turn.sh — 16→4 corner-turn bench at production per-pair rate.

Usage:
  bash $0 [options]
    --duration N         seconds (default $DURATION_S)
    --start-lead N       sync barrier from now (default $START_LEAD_S s)
    --tx-list "n03 n04"  override TX (corr) list
    --rx-list "n01"      override RX (search) list
    --smoke              shortcut: --duration 10, 1×1 first-of-list pair
    --rate N             rate-gbps-per-flow (default $RATE_GBPS_PER_FLOW)
    --n-flows N          flows per pair (default $N_FLOWS → 0.44 Gb/s/pair)
    -h | --help          this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION_S=$2; shift 2 ;;
        --start-lead) START_LEAD_S=$2; shift 2 ;;
        --tx-list) TX_LIST=$2; shift 2 ;;
        --rx-list) RX_LIST=$2; shift 2 ;;
        --smoke) SMOKE=YES; DURATION_S=10; START_LEAD_S=20; shift ;;
        --rate) RATE_GBPS_PER_FLOW=$2; shift 2 ;;
        --n-flows) N_FLOWS=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1"; usage; exit 1 ;;
    esac
done

if [ "$SMOKE" = YES ]; then
    TX_LIST=$(echo "$TX_LIST" | awk '{print $1}')   # first TX only
    RX_LIST=$(echo "$RX_LIST" | awk '{print $1}')   # first RX only
fi

TXS=($TX_LIST)
RXS=($RX_LIST)
N_TX=${#TXS[@]}
N_RX=${#RXS[@]}
N_PAIRS=$((N_TX * N_RX))

T0=$(($(date +%s) + START_LEAD_S))
T0_ISO=$(date -u -d @"$T0" +%FT%TZ)
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR=$HOME/dsart-corner-turn-logs/$RUN_TAG
REMOTE_DIR=$HOME/dsart-corner-turn/$RUN_TAG
mkdir -p "$LOG_DIR"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no)

# Each net_pair process launched via this preamble — sources conda + cds repo.
PY_PRE='source ~/miniforge3/etc/profile.d/conda.sh && conda activate dsa110-rt && export PYTHONUTF8=1 && cd ~/proj/dsa110-rt'

PY=python   # under the activated env

# ─── 2. Banner ────────────────────────────────────────────────────────────

PER_PAIR_GBPS=$(awk "BEGIN { printf \"%.3f\", $N_FLOWS * $RATE_GBPS_PER_FLOW }")
PER_RX_GBPS=$(awk "BEGIN { printf \"%.3f\", $N_TX * $N_FLOWS * $RATE_GBPS_PER_FLOW }")
TOTAL_GBPS=$(awk "BEGIN { printf \"%.3f\", $N_PAIRS * $N_FLOWS * $RATE_GBPS_PER_FLOW }")

{
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  corner-turn bench (16→4 fleet, production per-pair rate)"
    echo "═══════════════════════════════════════════════════════════════════════"
    printf "  TX list (%2d): %s\n" "$N_TX" "$TX_LIST"
    printf "  RX list (%2d): %s\n" "$N_RX" "$RX_LIST"
    printf "  Pairs       : %d (%d TX × %d RX)\n" "$N_PAIRS" "$N_TX" "$N_RX"
    printf "  Duration    : %d s\n" "$DURATION_S"
    printf "  T0 (sync)   : %s  (unix %d, lead %ds)\n" "$T0_ISO" "$T0" "$START_LEAD_S"
    printf "  Per-pair    : %d flows × %s Gb/s = %s Gb/s\n" "$N_FLOWS" "$RATE_GBPS_PER_FLOW" "$PER_PAIR_GBPS"
    printf "  Per-RX agg  : %s Gb/s  (ingress)\n" "$PER_RX_GBPS"
    printf "  Total fabric: %s Gb/s\n" "$TOTAL_GBPS"
    printf "  Log dir     : %s\n" "$LOG_DIR"
    [ "$SMOKE" = YES ] && echo "  (smoke mode)"
    echo "═══════════════════════════════════════════════════════════════════════"
} | tee "$LOG_DIR/banner.txt"

# ─── 3. Preflight ─────────────────────────────────────────────────────────

ALL_NODES=("${TXS[@]}" "${RXS[@]}")
echo
echo "[$(date -u +%FT%TZ)] preflight: checking ${#ALL_NODES[@]} nodes…"
PREFLIGHT_FAIL=0

# Wrap remote preflight commands in a here-doc to avoid quote nesting hell.
# Each node prints exactly one of:
#   OK <iface> <ip> mtu=<mtu>
#   FAIL <reason>
for n in "${ALL_NODES[@]}"; do
    out=$(ssh "${SSH_OPTS[@]}" "$n.pro.pvt" REMOTE_DIR="$REMOTE_DIR" bash -s <<'REMOTE' 2>&1
set -eo pipefail
export PYTHONUTF8=1
source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null || { echo "FAIL no miniforge3"; exit 1; }
conda activate dsa110-rt 2>/dev/null || { echo "FAIL no dsa110-rt env"; exit 1; }
cd ~/proj/dsa110-rt 2>/dev/null || { echo "FAIL no ~/proj/dsa110-rt"; exit 1; }
python -c 'import dsart, dsart.transport.recv_epoll, bench.net_pair' 2>&1 \
    || { echo "FAIL dsart import"; exit 1; }
mkdir -p "$REMOTE_DIR"
read IFACE CIDR <<< "$(ip -4 -br addr show 2>/dev/null | awk '$3 ~ /^10\.41\.0\./ {print $1, $3; exit}')"
[ -n "$IFACE" ] || { echo "FAIL no 10.41.0.x iface"; exit 1; }
MTU=$(cat /sys/class/net/"$IFACE"/mtu 2>/dev/null)
echo "OK $IFACE $CIDR mtu=$MTU"
REMOTE
)
    if echo "$out" | grep -q '^OK '; then
        echo "  ✓ $n: $(echo "$out" | grep '^OK ' | head -1)"
    else
        echo "  ✗ $n: $out"
        PREFLIGHT_FAIL=1
    fi
done
[ "$PREFLIGHT_FAIL" -ne 0 ] && { echo "PREFLIGHT FAILED"; exit 2; }
echo "[$(date -u +%FT%TZ)] preflight OK"

# ─── 4. Launch all 64 RX side first, then 64 TX side ──────────────────────
#
# RX must be listening before TX hits the wire. We launch all RX processes
# first (parallel ssh + nohup), wait for them to settle (~3s), then launch
# all TX processes.

launch_rx() {
    local rx=$1 tx=$2
    local rx_ip=${RX_IP[$rx]}
    local chgroup=${TX_CHGROUP[$tx]}
    local port=$((PORT_BASE + chgroup))
    local counters="$REMOTE_DIR/rx_${rx}_from_${tx}.json"
    local log="$LOG_DIR/launch_rx_${rx}_from_${tx}.log"
    {
        echo "[$(date -u +%FT%TZ)] launching RX on $rx for stream from $tx (chgroup $chgroup, port $port)"
        ssh "${SSH_OPTS[@]}" "$rx.pro.pvt" "
            $PY_PRE
            nohup $PY -m bench.net_pair --mode rx \
                --listen-addr $rx_ip --listen-port $port \
                --n-flows $N_FLOWS --rate-gbps-per-flow $RATE_GBPS_PER_FLOW \
                --chgroup $chgroup --duration-s $DURATION_S \
                --start-at $T0 --counters-out '$counters' \
                --rx-impl epoll \
                > ~/dsart-ct-rx-${tx}.log 2>&1 < /dev/null &
            echo launched pid=\$! at \$(date -u +%FT%TZ) chgroup=$chgroup port=$port
        "
    } > "$log" 2>&1
}

launch_tx() {
    local tx=$1 rx=$2
    local rx_ip=${RX_IP[$rx]}
    local chgroup=${TX_CHGROUP[$tx]}
    local port=$((PORT_BASE + chgroup))
    local counters="$REMOTE_DIR/tx_${tx}_to_${rx}.json"
    local log="$LOG_DIR/launch_tx_${tx}_to_${rx}.log"
    {
        echo "[$(date -u +%FT%TZ)] launching TX on $tx → $rx ($rx_ip:$port, chgroup $chgroup)"
        ssh "${SSH_OPTS[@]}" "$tx.pro.pvt" "
            $PY_PRE
            nohup $PY -m bench.net_pair --mode tx \
                --target-addr $rx_ip --target-port $port \
                --n-flows $N_FLOWS --rate-gbps-per-flow $RATE_GBPS_PER_FLOW \
                --chgroup $chgroup --duration-s $DURATION_S \
                --start-at $T0 --counters-out '$counters' \
                > ~/dsart-ct-tx-${rx}.log 2>&1 < /dev/null &
            echo launched pid=\$! at \$(date -u +%FT%TZ) chgroup=$chgroup port=$port → $rx_ip
        "
    } > "$log" 2>&1
}

echo
echo "[$(date -u +%FT%TZ)] launching $N_PAIRS RX processes (one per [RX, TX] pair)…"
for rx in "${RXS[@]}"; do
    for tx in "${TXS[@]}"; do
        launch_rx "$rx" "$tx" &
    done
done
wait
echo "[$(date -u +%FT%TZ)] RX side launched. sleeping 3 s to let listeners settle…"
sleep 3

echo "[$(date -u +%FT%TZ)] launching $N_PAIRS TX processes (one per [TX, RX] pair)…"
for tx in "${TXS[@]}"; do
    for rx in "${RXS[@]}"; do
        launch_tx "$tx" "$rx" &
    done
done
wait
echo "[$(date -u +%FT%TZ)] TX side launched."

# ─── 5. Wait for run to complete ───────────────────────────────────────────

# Total wall time = lead + duration + tolerance
WAIT_S=$((START_LEAD_S + DURATION_S + SYNC_TOLERANCE_S - (T0 - $(($(date +%s) - START_LEAD_S)))))
END_AT=$((T0 + DURATION_S + SYNC_TOLERANCE_S))
echo
echo "[$(date -u +%FT%TZ)] T0 in $((T0 - $(date +%s))) s; sleeping until $(date -u -d @$END_AT +%FT%TZ) (T0 + duration + ${SYNC_TOLERANCE_S} s margin)"
while [ "$(date +%s)" -lt "$END_AT" ]; do
    remaining=$((END_AT - $(date +%s)))
    if [ $((remaining % 60)) -eq 0 ] && [ "$remaining" -gt 0 ]; then
        echo "[$(date -u +%FT%TZ)] $remaining s remaining…"
    fi
    sleep 5
done

# ─── 6. Verify all processes exited ───────────────────────────────────────

echo
echo "[$(date -u +%FT%TZ)] verifying all 128 processes exited (64 TX + 64 RX)…"
STILL_RUNNING=0
for n in "${ALL_NODES[@]}"; do
    cnt=$(ssh "${SSH_OPTS[@]}" "$n.pro.pvt" "pgrep -fc 'bench.net_pair' 2>/dev/null || echo 0" 2>/dev/null)
    if [ "$cnt" -gt 0 ]; then
        echo "  ! $n: $cnt net_pair processes still running"
        STILL_RUNNING=$((STILL_RUNNING + cnt))
    fi
done
if [ "$STILL_RUNNING" -gt 0 ]; then
    echo "WARN: $STILL_RUNNING processes still running; waiting another 30 s…"
    sleep 30
fi

# ─── 7. Collect counters ──────────────────────────────────────────────────

echo
echo "[$(date -u +%FT%TZ)] collecting counter JSONs from all nodes…"
COLLECT_FAIL=0
for rx in "${RXS[@]}"; do
    for tx in "${TXS[@]}"; do
        scp -q "${SSH_OPTS[@]}" "$rx.pro.pvt:$REMOTE_DIR/rx_${rx}_from_${tx}.json" "$LOG_DIR/" 2>/dev/null || {
            echo "  ! missing rx_${rx}_from_${tx}.json"
            COLLECT_FAIL=$((COLLECT_FAIL + 1))
        }
    done
done
for tx in "${TXS[@]}"; do
    for rx in "${RXS[@]}"; do
        scp -q "${SSH_OPTS[@]}" "$tx.pro.pvt:$REMOTE_DIR/tx_${tx}_to_${rx}.json" "$LOG_DIR/" 2>/dev/null || {
            echo "  ! missing tx_${tx}_to_${rx}.json"
            COLLECT_FAIL=$((COLLECT_FAIL + 1))
        }
    done
done
N_COLLECTED=$(ls "$LOG_DIR"/*.json 2>/dev/null | wc -l)
echo "  collected $N_COLLECTED / $((2 * N_PAIRS)) counter JSONs ($COLLECT_FAIL missing)"

# ─── 8. Aggregate + verdict ───────────────────────────────────────────────

python3 - <<PY > "$LOG_DIR/verdict.txt"
import glob, json, os, sys
from collections import defaultdict

log_dir = "$LOG_DIR"
target_per_pair_gbps = $PER_PAIR_GBPS
target_per_rx_gbps   = $PER_RX_GBPS
duration_s          = $DURATION_S
rate_tol = 0.05  # ±5%
# Fragment loss budget: 1e-4 per plan §M4b DoD I1. For very short smoke
# runs (< 30 s) the startup transient (RX socket warm-up, first few cubes
# arriving before reassembly hash is hot) often pushes a single pair over
# 1e-4; relax to 5e-4 in those cases. The 5-min production run uses 1e-4.
frag_loss_budget = 1e-4 if duration_s >= 30 else 5e-4

pairs = defaultdict(dict)  # (tx, rx) -> {"tx": {...}, "rx": {...}}

def strip_json(s):
    return s[:-5] if s.endswith(".json") else s

def load(prefix):
    for path in sorted(glob.glob(os.path.join(log_dir, prefix + "_*.json"))):
        try:
            with open(path) as f: d = json.load(f)
        except Exception as e:
            print("  ! parse " + path + ": " + str(e), file=sys.stderr)
            continue
        # filenames: tx_<TX>_to_<RX>.json  or  rx_<RX>_from_<TX>.json
        fn = strip_json(os.path.basename(path))
        parts = fn.split("_")
        if parts[0] == "tx":
            tx, rx = parts[1], parts[3]
        else:
            rx, tx = parts[1], parts[3]
        pairs[(tx, rx)][parts[0]] = d

load("tx"); load("rx")

n_pairs = len(pairs)
n_pass_i1 = n_pass_i2 = n_pass_i3 = n_pass_i4 = 0
per_pair_rows = []
per_rx_ingress = defaultdict(float)
fail_rows = []

for (tx, rx), sides in sorted(pairs.items()):
    rxd = sides.get("rx", {})
    txd = sides.get("tx", {})
    rx_gbps = rxd.get("achieved_gbps_aggregate", 0.0)
    tx_gbps = txd.get("achieved_gbps_aggregate", 0.0)
    frag_loss = rxd.get("fragment_loss_estimate_fraction", 1.0)
    pattern_mismatch = rxd.get("pattern_mismatch_count", 9999)
    tx_drops = txd.get("tx_dropped_payloads_total", 0)
    if tx_drops is None: tx_drops = 0
    window_slide = rxd.get("window_slide_zerofill_count", 0)
    sendto_errors = txd.get("sendto_errors_total", 0) or 0
    i1 = (1 - rate_tol) * target_per_pair_gbps <= rx_gbps <= (1 + rate_tol) * target_per_pair_gbps
    i2 = frag_loss < frag_loss_budget
    i3 = pattern_mismatch == 0
    i4 = tx_drops == 0 and sendto_errors == 0
    n_pass_i1 += int(i1); n_pass_i2 += int(i2); n_pass_i3 += int(i3); n_pass_i4 += int(i4)
    per_pair_rows.append({
        "tx": tx, "rx": rx,
        "rx_gbps": rx_gbps, "tx_gbps": tx_gbps,
        "frag_loss": frag_loss, "pattern_mismatch": pattern_mismatch,
        "tx_drops": tx_drops, "sendto_errors": sendto_errors,
        "window_slide_zerofill": window_slide,
        "I1": i1, "I2": i2, "I3": i3, "I4": i4,
    })
    per_rx_ingress[rx] += rx_gbps
    if not (i1 and i2 and i3 and i4):
        fail_rows.append((tx, rx, rx_gbps, frag_loss, pattern_mismatch, tx_drops, window_slide))

n_pass_i5 = 0
for rx, agg in per_rx_ingress.items():
    if (1 - rate_tol) * target_per_rx_gbps <= agg <= (1 + rate_tol) * target_per_rx_gbps:
        n_pass_i5 += 1

all_pass = (n_pass_i1 == n_pairs and n_pass_i2 == n_pairs and
            n_pass_i3 == n_pairs and n_pass_i4 == n_pairs and
            n_pass_i5 == len(per_rx_ingress))

print("==================== CORNER-TURN VERDICT ====================")
print("  pairs       : {}".format(n_pairs))
print("  duration    : {} s".format(duration_s))
print("  I1 rate     : {}/{} pass (per-pair {:.3f} ± {:.0f}%)".format(n_pass_i1, n_pairs, target_per_pair_gbps, rate_tol*100))
print("  I2 frag-loss: {}/{} pass (< {:.0e})".format(n_pass_i2, n_pairs, frag_loss_budget))
print("  I3 pattern  : {}/{} pass (mismatch == 0)".format(n_pass_i3, n_pairs))
print("  I4 tx-drops : {}/{} pass (drops + sendto_errors == 0)".format(n_pass_i4, n_pairs))
print("  I5 RX agg   : {}/{} pass (per-RX {:.3f} ± {:.0f}%)".format(n_pass_i5, len(per_rx_ingress), target_per_rx_gbps, rate_tol*100))
print()
print("  Per-RX ingress aggregates:")
for rx in sorted(per_rx_ingress):
    print("    {}: {:.3f} Gb/s".format(rx, per_rx_ingress[rx]))
print()
print("  VERDICT: {}".format("PASS" if all_pass else "FAIL"))
print()
if fail_rows:
    print("  Failing pairs ({}):".format(len(fail_rows)))
    for tx, rx, gbps, fl, pm, td, ws in fail_rows[:20]:
        print("    {}->{}: rx_gbps={:.3f} frag_loss={:.2e} pattern_mismatch={} tx_drops={} window_slide={}".format(tx, rx, gbps, fl, pm, td, ws))

summary = {
    "verdict": "PASS" if all_pass else "FAIL",
    "duration_s": duration_s,
    "T0_unix": $T0,
    "n_pairs": n_pairs,
    "target_per_pair_gbps": target_per_pair_gbps,
    "target_per_rx_gbps": target_per_rx_gbps,
    "frag_loss_budget": frag_loss_budget,
    "invariants": {
        "I1_rate": "{}/{}".format(n_pass_i1, n_pairs),
        "I2_frag_loss": "{}/{}".format(n_pass_i2, n_pairs),
        "I3_pattern_mismatch": "{}/{}".format(n_pass_i3, n_pairs),
        "I4_tx_drops": "{}/{}".format(n_pass_i4, n_pairs),
        "I5_rx_aggregate": "{}/{}".format(n_pass_i5, len(per_rx_ingress)),
    },
    "per_rx_ingress_gbps": dict(per_rx_ingress),
    "per_pair": per_pair_rows,
}
with open(os.path.join(log_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY

echo
cat "$LOG_DIR/verdict.txt"
echo
echo "Full summary: $LOG_DIR/summary.json"
echo "All artifacts: $LOG_DIR/"

VERDICT=$(grep -oE 'VERDICT: (PASS|FAIL)' "$LOG_DIR/verdict.txt" | awk '{print $2}')
[ "$VERDICT" = "PASS" ] && exit 0 || exit 1
