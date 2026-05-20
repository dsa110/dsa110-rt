#!/usr/bin/env bash
# _m75_test1_single_capture.sh --- M7.5 Test 1.
#
# A single capture instance on n06: SNAP UDP -> dada_capture_manythread
# -> PSRDADA dada buffer -> dada_drain. Designed to validate the
# vendored binary + recvmmsg path against live SNAP packets at full
# wire rate (~244 k pps / port).
#
# Success criteria:
#   1. Capture binary stays armed and writes blocks for the full
#      DURATION_S without crashing.
#   2. n_block_writes increments at the expected cadence (~7.45/s);
#      a 60 s soak should produce ~447 blocks.
#   3. n_dropped_kernel stays at 0 (otherwise SO_RCVBUF is too
#      small or the recv hot path is too slow to drain the kernel
#      queue).
#   4. n_recv_errors stays at 0.
#   5. dada_drain keeps up: 'full' count on the dada buffer never
#      exceeds ~2.
#
# Run from n06:  bash tools/ops/_m75_test1_single_capture.sh
set -uo pipefail

PORT="${PORT:-4011}"            # SNAP UDP data port we listen on
CPORT="${CPORT:-11223}"         # control UDP port (legacy default)
KEY_HEX="${KEY_HEX:-dada}"      # PSRDADA key
DURATION_S="${DURATION_S:-60}"
DATA_IP="${DATA_IP:-10.41.0.203}"   # n06 enp129s0f0
CN_ID="${CN_ID:-6}"             # n06 -> corr-node 6
HEADER="${HEADER:-/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt}"

DSART_ROOT="/home/ubuntu/proj/dsa110-rt"
CAP_BIN="${DSART_ROOT}/src/dsart/capture/dsart_capture_manythread"
LOG=/tmp/_m75_test1_capture.log
DRAIN_LOG=/tmp/_m75_test1_drain.log
SHM="/dev/shm/dsart-capture-${PORT}"

echo "=== M7.5 Test 1: single-instance capture on n06 ==="
echo "    CAP_BIN=${CAP_BIN}"
echo "    PORT=${PORT} CPORT=${CPORT} KEY=0x${KEY_HEX} duration=${DURATION_S}s"
echo "    DATA_IP=${DATA_IP}"

# Pre-emptive cleanup
pkill -f "dsart_capture_manythread.*-p ${PORT}" 2>/dev/null || true
pkill -f "dsart.services.dada_drain.*--key ${KEY_HEX}" 2>/dev/null || true
sleep 0.5
dada_db -d -k "${KEY_HEX}" >/dev/null 2>&1 || true
rm -f "${SHM}" 2>/dev/null || true

cleanup_on_exit() {
    set +e
    echo
    echo "=== cleanup ==="
    [[ -n "${CAP_PID:-}" ]] && kill -TERM "${CAP_PID}" 2>/dev/null
    [[ -n "${DRAIN_PID:-}" ]] && kill -TERM "${DRAIN_PID}" 2>/dev/null
    sleep 0.5
    [[ -n "${CAP_PID:-}" ]] && kill -KILL "${CAP_PID}" 2>/dev/null
    [[ -n "${DRAIN_PID:-}" ]] && kill -KILL "${DRAIN_PID}" 2>/dev/null
    dada_db -d -k "${KEY_HEX}" >/dev/null 2>&1
}
trap cleanup_on_exit EXIT

if [[ ! -x "${CAP_BIN}" ]]; then
    echo "FAIL: ${CAP_BIN} missing -- run setup.py build_ext --inplace first"
    exit 1
fi

# Production-grade buffer: 144 MiB x 20 bufs (matches dada in configs/dsart_pipeline_rt.yaml)
echo "+ dada_db -k ${KEY_HEX} -b 150994944 -n 20 -c 1 -r 1 -l"
dada_db -k "${KEY_HEX}" -b 150994944 -n 20 -c 1 -r 1 -l 1>/dev/null
echo "OK: created dada buffer 0x${KEY_HEX}  (144 MiB x 20)"

# Activate dsa110-rt conda env for dada_drain + monitor script.
set +eu
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate dsa110-rt 2>/dev/null
set -eu

export PYTHONPATH="${DSART_ROOT}/src"

# Spawn dada_drain BEFORE the capture binary so the buffer has a
# reader as soon as it has bytes (corr_slow / corr_fast aren't here
# to read, so without a drain the ring fills + blocks the writer).
python3 -u -m dsart.services.dada_drain --key "${KEY_HEX}" --log-every 32 \
    > "${DRAIN_LOG}" 2>&1 &
DRAIN_PID=$!
echo "OK: dada_drain pid=${DRAIN_PID}"
sleep 0.5

# Spawn capture.
"${CAP_BIN}" \
    -j "${DATA_IP}" -i 127.0.0.1 \
    -p "${PORT}" -q "${CPORT}" \
    -o "${KEY_HEX}" \
    -f "${HEADER}" \
    > "${LOG}" 2>&1 &
CAP_PID=$!
echo "OK: dsart_capture_manythread pid=${CAP_PID}"

# Wait for shm.
for i in $(seq 1 30); do
    [[ -f "${SHM}" ]] && break
    sleep 0.1
done
if [[ ! -f "${SHM}" ]]; then
    echo "FAIL: ${SHM} did not appear in 3 s"
    tail -30 "${LOG}"
    exit 1
fi
echo "OK: ${SHM} online"

# Inspect the initial snapshot to confirm the binary saw the SNAP
# packets (last_seq_no should already be advancing).
python3 - <<PY
import time
from dsart.capture import mon_shm
with mon_shm.MonShm.open(${PORT}) as m:
    t0 = time.monotonic()
    s1 = m.snapshot()
    time.sleep(1.0)
    s2 = m.snapshot()
    elapsed = time.monotonic() - t0
    print(f"OK: pre-arm stats (1 s window)")
    print(f"    arm_state={s1.arm_state.name} -> {s2.arm_state.name}")
    print(f"    last_seq: {s1.last_seq_no} -> {s2.last_seq_no} (+{s2.last_seq_no - s1.last_seq_no})")
    print(f"    n_recv_packets: {s1.n_recv_packets} -> {s2.n_recv_packets} (+{s2.n_recv_packets - s1.n_recv_packets})")
    print(f"    n_recv_bytes:   {s1.n_recv_bytes} -> {s2.n_recv_bytes} (rate ~{(s2.n_recv_bytes - s1.n_recv_bytes)*8/elapsed/1e9:.3f} Gb/s)")
    print(f"    n_recv_errors: {s2.n_recv_errors}")
    print(f"    n_dropped_kernel: {s2.n_dropped_kernel}")
    print(f"    socket_rcvbuf_bytes: {s2.socket_rcvbuf_bytes}")
    assert s2.n_recv_packets > s1.n_recv_packets, "no packets received from SNAPs!"
    assert s2.arm_state.name == "WAITING_FOR_ARM", \
        f"unexpected arm_state {s2.arm_state.name} (deterministic arm must be enforced)"
PY

# Arm UTC_START = last_seq + ~30000 specnums (~2 s from now).
#
# Production-grade arming requires that we read a NON-STALE last_seq
# from the shm: at startup the C binary's stats_thread sleeps 2 s
# before its first tick, and (pre-fix) the shm could read 0 even
# while recv threads were already draining packets. The fix landed
# in dsart_capture_manythread.c (recv_thread now atomic_stores
# last_seq_no on every packet, ~1 ns / packet overhead). This poll
# is the operator-side belt-and-braces companion to that fix --
# it guarantees we never arm against a stale 0 even if the binary
# is started in a way that delays first-packet receipt (e.g. NIC
# coming up, SNAPs not yet streaming).
echo
echo "=== waiting for first fresh last_seq from shm ==="
python3 - <<PY
import time
from dsart.capture import mon_shm
deadline = time.monotonic() + 5.0
with mon_shm.MonShm.open(${PORT}) as m:
    while time.monotonic() < deadline:
        snap = m.snapshot()
        if snap.last_seq_no > 0 and snap.n_recv_packets > 100:
            print(f"OK: shm fresh: last_seq_no={snap.last_seq_no} n_recv={snap.n_recv_packets}")
            break
        time.sleep(0.1)
    else:
        raise SystemExit("FAIL: shm last_seq_no stayed 0 / no packets after 5 s")
PY

echo
echo "=== arming UTC_START ==="
ARM_SEQ=$(python3 -c "
from dsart.capture import mon_shm
with mon_shm.MonShm.open(${PORT}) as m:
    snap = m.snapshot()
assert snap.last_seq_no > 0, f'last_seq_no still stale: {snap.last_seq_no}'
print(snap.last_seq_no + 30000)
")
echo "OK: arming UTC_START=${ARM_SEQ}"
python3 -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b'UTC_START-${ARM_SEQ}', ('127.0.0.1', ${CPORT}))
sock.close()
print('OK: sent UTC_START-${ARM_SEQ} to 127.0.0.1:${CPORT}')
"

# Watch counters for DURATION_S, printing every 5 s.
echo
echo "=== ${DURATION_S} s soak ==="
python3 - <<PY
import time
from dsart.capture import mon_shm

T = ${DURATION_S}
INTERVAL = 5.0
with mon_shm.MonShm.open(${PORT}) as m:
    prev = m.snapshot()
    t0 = time.monotonic()
    print(f"{'t':>4} {'arm':<9} {'block_wr':>9} {'recv_pkt':>13} {'+pkt/s':>10} "
          f"{'+kern_drop/s':>14} {'rate_Gbps':>10} {'last_seq':>14}")
    print("-" * 100)
    while True:
        t = time.monotonic() - t0
        if t >= T:
            break
        time.sleep(INTERVAL)
        cur = m.snapshot()
        dt = INTERVAL
        dpkts = cur.n_recv_packets - prev.n_recv_packets
        ddrops = cur.n_dropped_kernel - prev.n_dropped_kernel
        dblocks = cur.n_block_writes - prev.n_block_writes
        rate = (cur.n_recv_bytes - prev.n_recv_bytes) * 8 / dt / 1e9
        print(f"{t:>4.0f} {cur.arm_state.name:<9} "
              f"{cur.n_block_writes:>9d} {cur.n_recv_packets:>13d} "
              f"{dpkts/dt:>10.0f} {ddrops/dt:>14.0f} "
              f"{rate:>10.3f} {cur.last_seq_no:>14d}")
        prev = cur
    final = m.snapshot()

print()
print("=== final summary ===")
print(f"    duration_s            = {time.monotonic() - t0:.1f}")
print(f"    final arm_state       = {final.arm_state.name}")
print(f"    n_block_writes        = {final.n_block_writes}")
print(f"    n_recv_packets        = {final.n_recv_packets}")
print(f"    n_recv_bytes          = {final.n_recv_bytes}")
print(f"    n_dropped_payload     = {final.n_dropped_payload}")
print(f"    n_dropped_kernel      = {final.n_dropped_kernel}")
print(f"    n_seq_skipped         = {final.n_seq_skipped}")
print(f"    n_too_late            = {final.n_too_late}")
print(f"    n_wrong_size          = {final.n_wrong_size}")
print(f"    n_recv_errors         = {final.n_recv_errors}")
print(f"    last_seq_no           = {final.last_seq_no}")
print()
# Quality gate
expected_blocks = int(${DURATION_S} * 7.45)
if final.n_block_writes < expected_blocks * 0.95:
    print(f"WARN: block_writes={final.n_block_writes} below 95% of expected {expected_blocks}")
if final.n_dropped_kernel > 0:
    print(f"WARN: kernel drops detected: {final.n_dropped_kernel}")
if final.n_recv_errors > 0:
    print(f"WARN: recv_errors > 0: {final.n_recv_errors}")
if final.arm_state.name != "WRITING":
    print(f"WARN: final arm_state {final.arm_state.name} != WRITING")
PY

echo
echo "=== drain log tail ==="
tail -5 "${DRAIN_LOG}"

echo
echo "=== capture log tail ==="
tail -10 "${LOG}"

echo
echo "=== M7.5 Test 1 done ==="
