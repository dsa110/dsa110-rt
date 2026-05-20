#!/usr/bin/env bash
# _capture_smoke.sh --- runs a 10 s soak of dsart_capture_manythread
# against a synthetic dada_junkdb feeder + a PSRDADA buffer on the
# loopback interface. Verifies:
#
#   1. Binary starts cleanly
#   2. shm /dev/shm/dsart-capture-<port> appears and has the right
#      magic + version
#   3. Python sidecar (dsart.services.capture_control) reads the shm
#      and publishes to mock-etcd
#   4. Binary shuts down cleanly on SIGTERM
#
# Intended to be run on a node that has libpsrdada (n06+). Used both
# as a CI gate via the dsa110-rt fleet sync, and as an ad-hoc dev
# tool for chasing capture-binary regressions.
#
# Usage:  bash tools/ops/_capture_smoke.sh
set -euo pipefail

CAP_DIR="${CAP_DIR:-/home/ubuntu/proj/dsa110-rt/src/dsart/capture}"
PORT="${PORT:-4099}"           # any unused port
KEY="${KEY:-1234}"             # PSRDADA key (anything unused)
DURATION_S="${DURATION_S:-8}"

CAP_BIN="${CAP_DIR}/dsart_capture_manythread"
LOG=/tmp/_capture_smoke.log
SHM="/dev/shm/dsart-capture-${PORT}"

cleanup() {
    [[ -n "${CAP_PID:-}" ]] && kill -TERM "${CAP_PID}" 2>/dev/null || true
    sleep 0.5
    [[ -n "${CAP_PID:-}" ]] && kill -KILL "${CAP_PID}" 2>/dev/null || true
    # dada_db segfaults when the buffer doesn't exist; swallow stderr.
    dada_db -d -k "${KEY}" >/dev/null 2>&1 || true
    rm -f "${SHM}"
}
trap cleanup EXIT

# Pre-emptive cleanup of any leftover state from a prior crash.
dada_db -d -k "${KEY}" >/dev/null 2>&1 || true
rm -f "${SHM}" 2>/dev/null || true

echo "=== _capture_smoke.sh ==="
echo "    CAP_BIN=${CAP_BIN}"
echo "    PORT=${PORT}  KEY=0x${KEY}  duration=${DURATION_S}s"

if [[ ! -x "${CAP_BIN}" ]]; then
    echo "FAIL: ${CAP_BIN} not found / not executable"
    exit 1
fi

# Create the PSRDADA buffer. 8 MiB block x 8 bufs = small but valid.
dada_db -k "${KEY}" -b 9437184 -n 8 -c 1 -r 1 -l 1>/dev/null
echo "OK: created dada buffer 0x${KEY}"

# Spawn the binary on the loopback interface.
"${CAP_BIN}" \
    -j 127.0.0.1 -i 127.0.0.1 \
    -p "${PORT}" -q $((PORT + 1000)) \
    -k "${KEY}" \
    > "${LOG}" 2>&1 &
CAP_PID=$!
echo "OK: dsart_capture_manythread pid=${CAP_PID}"

# Wait for the shm to appear (the binary opens it in dsart_capture_mon_open
# called from main() near startup; should take <500 ms).
for i in $(seq 1 50); do
    if [[ -f "${SHM}" ]]; then
        echo "OK: shm appeared after ${i}*100ms: ${SHM}"
        break
    fi
    sleep 0.1
done
if [[ ! -f "${SHM}" ]]; then
    echo "FAIL: shm ${SHM} did not appear in 5 s"
    cat "${LOG}"
    exit 1
fi

# Validate the shm magic via Python.
export PYTHONPATH=/home/ubuntu/proj/dsa110-rt/src
set +eu
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate dsa110-rt 2>/dev/null
set -eu

python3 - <<PY
import sys, time
from dsart.capture import mon_shm

with mon_shm.MonShm.open(${PORT}) as m:
    snap = m.snapshot()
    print(f"OK: shm magic / version ABI checks pass")
    print(f"    udp_port={snap.udp_port}  control_port={snap.control_port}")
    print(f"    arm_state={snap.arm_state.name}  pid={snap.pid}")
    print(f"    socket_rcvbuf_bytes={snap.socket_rcvbuf_bytes}")
    print(f"    age_ms={snap.age_ms:.1f}")
    assert snap.udp_port == ${PORT}, "udp_port mismatch"
    assert snap.arm_state.name == "WAITING_FOR_ARM", (
        f"unexpected arm_state {snap.arm_state.name} -- "
        f"should be WAITING_FOR_ARM until utc_start is sent"
    )
    assert snap.age_ms < 1000.0, "stats_thread not ticking"
    assert snap.socket_rcvbuf_bytes > 0, "SO_RCVBUF not reported"
    print("OK: snapshot fields look sane")
PY

# Sleep a few seconds so we see the age_ms advancing.
sleep 3

python3 - <<PY
from dsart.capture import mon_shm
with mon_shm.MonShm.open(${PORT}) as m:
    snap = m.snapshot()
    print(f"OK: age_ms after 3 s = {snap.age_ms:.1f} (should still be <1000)")
    assert snap.age_ms < 1000.0, f"shm stale ({snap.age_ms:.1f} ms)"
PY

# Sidecar smoke: spin up the Python publisher against a mock store.
python3 - <<PY
from dsart.services.capture_control import CaptureControlService

class _MockStore:
    def __init__(self): self.puts = []
    def put_dict(self, k, v): self.puts.append((k, v))

store = _MockStore()
svc = CaptureControlService(udp_ports=(${PORT},), cn_id=99, store=store)
svc._tick()
assert store.puts, "sidecar did not publish anything"
key, payload = store.puts[-1]
print(f"OK: sidecar published to {key}")
print(f"    arm_state={payload['arm_state']} degraded={payload['degraded']}")
print(f"    rate_gbps={payload['rate_gbps']} age_ms={payload['age_ms']}")
assert payload['arm_state'] == 'WAITING_FOR_ARM'
assert payload['degraded'] is False
PY

# Sanity: send UTC_START and verify arm_state transitions to ARMED.
python3 - <<PY
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"UTC_START-99999999", ("127.0.0.1", $((PORT + 1000))))
sock.close()
print("OK: sent UTC_START-99999999")
PY

sleep 1

python3 - <<PY
from dsart.capture import mon_shm
with mon_shm.MonShm.open(${PORT}) as m:
    snap = m.snapshot()
    print(f"OK: after UTC_START, arm_state={snap.arm_state.name} utc_start_specnum={snap.utc_start_specnum}")
    assert snap.arm_state == mon_shm.ArmState.ARMED
    assert snap.utc_start_specnum == 99999999
PY

# Exercise the recvmmsg path: spray a batch of 200 well-formed UDP
# packets (1 specnum each) at the binary, then check the shm
# counters move. We don't have block-write going (the seq numbers
# we send put us well before the armed seq), so n_recv_packets is
# the cumulative count from the recv loop's start-condition gate
# and the actual block-write counter stays at 0; what we're really
# testing is that recvmmsg dispatches without errors.
python3 - <<PY
import socket, struct, time

# Each packet is 4616 B: 8 B header + 4608 B payload. Build a single
# template; vary seq_no across packets.
def make_pkt(seq_no, ant_id):
    # Packed exactly as the SNAP firmware emits (see legacy
    # bit-shuffle in dsart_capture_manythread.c::recv_thread).
    b0 = (seq_no >> 27) & 0xFF
    b1 = (seq_no >> 19) & 0xFF
    b2 = (seq_no >> 11) & 0xFF
    b3 = (seq_no >> 3)  & 0xFF
    b4 = (seq_no & 0x7) << 5
    b5 = 0
    b6 = (ant_id >> 8) & 0xFF
    b7 = ant_id & 0xFF
    hdr = bytes([b0, b1, b2, b3, b4, b5, b6, b7])
    return hdr + b"\x00" * 4608

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
N = 200
for i in range(N):
    pkt = make_pkt(seq_no=i+1, ant_id=(i % 32) * 3)
    sock.sendto(pkt, ("127.0.0.1", ${PORT}))
    if i % 50 == 49:
        time.sleep(0.005)  # let the kernel drain
sock.close()
print(f"OK: sprayed {N} UDP packets at port ${PORT}")

# Let the recvmmsg loop drain.
time.sleep(0.5)

from dsart.capture import mon_shm
with mon_shm.MonShm.open(${PORT}) as m:
    snap = m.snapshot()
    print(f"OK: post-spray n_recv_packets={snap.n_recv_packets} "
          f"n_recv_bytes={snap.n_recv_bytes} "
          f"last_seq_no={snap.last_seq_no} "
          f"n_recv_errors={snap.n_recv_errors}")
    # The recv path counts via the block-complete sums (legacy
    # accounting). Block-write doesn't fire because we sent fewer
    # packets than packets_per_buffer, but ``last_seq_no`` should
    # have moved off zero.
    assert snap.last_seq_no > 0, (
        f"recvmmsg path drained no packets: last_seq_no={snap.last_seq_no}"
    )
    assert snap.n_recv_packets >= N, (
        f"per-packet counter underreports: got {snap.n_recv_packets}, "
        f"expected >= {N}"
    )
    assert snap.n_recv_bytes >= N * 4608, (
        f"per-byte counter underreports: got {snap.n_recv_bytes}, "
        f"expected >= {N * 4608}"
    )
    assert snap.n_recv_errors == 0, (
        f"recvmmsg returned errors: {snap.n_recv_errors}"
    )
    print("OK: recvmmsg path is alive on the loopback feeder")
PY

# Clean shutdown via SIGTERM. Give the binary up to 5 s to exit
# (recv threads loop on recvmmsg with non-blocking; control thread
# wakes every 500 ms via SO_RCVTIMEO).
kill -TERM "${CAP_PID}"
for i in $(seq 1 50); do
    if ! kill -0 "${CAP_PID}" 2>/dev/null; then
        echo "OK: binary exited cleanly after ${i}*100ms on SIGTERM"
        break
    fi
    sleep 0.1
done
if kill -0 "${CAP_PID}" 2>/dev/null; then
    echo "FAIL: binary still running 5s after SIGTERM (will SIGKILL)"
    kill -KILL "${CAP_PID}"
    exit 1
fi

# Mon shm should be unlinked by the atexit hook on a clean SIGTERM
# shutdown. If it's still there, that's a regression in the signal
# handler ↔ atexit chain.
if [[ -f "${SHM}" ]]; then
    echo "FAIL: ${SHM} not unlinked at exit"
    exit 1
fi
echo "OK: shm cleanly unlinked on shutdown"

echo "=== _capture_smoke.sh: PASS ==="
