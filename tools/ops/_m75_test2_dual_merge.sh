#!/usr/bin/env bash
# _m75_test2_dual_merge.sh --- M7.5 Test 2.
#
# Two capture instances + dsaX_merge on n06: SNAP UDP -> {dada,eada}
# -> dsaX_merge -> fada -> dada_drain. This is the full corr-node
# write path minus the GPU consumers (corr_slow / corr_fast).
#
# Success criteria:
#   1. Both capture binaries stay armed (arm_state=WRITING) and
#      write blocks at ~7.45/s for the full DURATION_S.
#   2. dsaX_merge advances fada at ~7.45/s (merging dada+eada into
#      288 MiB merged blocks).
#   3. n_dropped_kernel = 0 on BOTH capture instances (the second
#      port doubles the recv pressure; sysctls are sized for it).
#   4. n_recv_errors = 0 on BOTH.
#   5. dada_drain keeps up on fada (which is the bottleneck since
#      merge writes 2x the byte rate as either capture).
#
# Run from n06:  bash tools/ops/_m75_test2_dual_merge.sh
set -uo pipefail

PORT_A="${PORT_A:-4011}"
PORT_B="${PORT_B:-4012}"
CPORT_A="${CPORT_A:-11223}"
CPORT_B="${CPORT_B:-11224}"
KEY_A="${KEY_A:-dada}"
KEY_B="${KEY_B:-eada}"
KEY_F="${KEY_F:-fada}"
DURATION_S="${DURATION_S:-60}"
DATA_IP="${DATA_IP:-10.41.0.203}"
CN_ID="${CN_ID:-6}"
HEADER="${HEADER:-/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/correlator_header_dsaX.txt}"

DSART_ROOT="/home/ubuntu/proj/dsa110-rt"
CAP_BIN="${DSART_ROOT}/src/dsart/capture/dsart_capture_manythread"
MERGE_BIN="/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_merge"

LOG_A=/tmp/_m75_test2_cap_a.log
LOG_B=/tmp/_m75_test2_cap_b.log
LOG_M=/tmp/_m75_test2_merge.log
LOG_D=/tmp/_m75_test2_drain.log

SHM_A="/dev/shm/dsart-capture-${PORT_A}"
SHM_B="/dev/shm/dsart-capture-${PORT_B}"

echo "=== M7.5 Test 2: dual-capture + merge on n06 ==="
echo "    CAP_BIN=${CAP_BIN}"
echo "    MERGE_BIN=${MERGE_BIN}"
echo "    PORT_A=${PORT_A}->0x${KEY_A}  PORT_B=${PORT_B}->0x${KEY_B}  MERGE->0x${KEY_F}"
echo "    duration=${DURATION_S}s"

# Pre-emptive cleanup.
pkill -f "dsart_capture_manythread.*-p ${PORT_A}" 2>/dev/null || true
pkill -f "dsart_capture_manythread.*-p ${PORT_B}" 2>/dev/null || true
pkill -f "dsaX_merge.*-o ${KEY_F}" 2>/dev/null || true
pkill -f "dsart.services.dada_drain.*--key ${KEY_F}" 2>/dev/null || true
sleep 0.5
for k in "${KEY_F}" "${KEY_B}" "${KEY_A}"; do
    dada_db -d -k "${k}" >/dev/null 2>&1 || true
done
rm -f "${SHM_A}" "${SHM_B}" 2>/dev/null || true

cleanup_on_exit() {
    set +e
    echo
    echo "=== cleanup ==="
    [[ -n "${CAP_A_PID:-}" ]] && kill -TERM "${CAP_A_PID}" 2>/dev/null
    [[ -n "${CAP_B_PID:-}" ]] && kill -TERM "${CAP_B_PID}" 2>/dev/null
    [[ -n "${MERGE_PID:-}" ]] && kill -TERM "${MERGE_PID}" 2>/dev/null
    [[ -n "${DRAIN_PID:-}" ]] && kill -TERM "${DRAIN_PID}" 2>/dev/null
    sleep 0.7
    for p in "${CAP_A_PID:-}" "${CAP_B_PID:-}" "${MERGE_PID:-}" "${DRAIN_PID:-}"; do
        [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null
    done
    for k in "${KEY_F}" "${KEY_B}" "${KEY_A}"; do
        dada_db -d -k "${k}" >/dev/null 2>&1
    done
}
trap cleanup_on_exit EXIT

if [[ ! -x "${CAP_BIN}" ]]; then
    echo "FAIL: ${CAP_BIN} missing"; exit 1
fi
if [[ ! -x "${MERGE_BIN}" ]]; then
    echo "FAIL: ${MERGE_BIN} missing"; exit 1
fi

# Create the production-sized PSRDADA rings:
#   dada: 144 MiB x 20  (1 reader, used by merge)
#   eada: 144 MiB x 20  (1 reader)
#   fada: 288 MiB x 70  (1 reader for this test; production is r=2)
echo
echo "+ dada_db -k ${KEY_A} -b 150994944 -n 20 -c 1 -r 1 -l"
dada_db -k "${KEY_A}" -b 150994944 -n 20 -c 1 -r 1 -l >/dev/null
echo "+ dada_db -k ${KEY_B} -b 150994944 -n 20 -c 1 -r 1 -l"
dada_db -k "${KEY_B}" -b 150994944 -n 20 -c 1 -r 1 -l >/dev/null
echo "+ dada_db -k ${KEY_F} -b 301989888 -n 70 -c 1 -r 1 -l"
dada_db -k "${KEY_F}" -b 301989888 -n 70 -c 1 -r 1 -l >/dev/null
echo "OK: created dada / eada / fada"

# Activate conda env.
set +eu
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh 2>/dev/null
conda activate dsa110-rt 2>/dev/null
set -eu
export PYTHONPATH="${DSART_ROOT}/src"

# Order of spawn: drain → merge → captures.
# (drain attaches to fada; merge attaches to fada as writer +
# dada/eada as readers; captures attach to dada/eada as writers.)
python3 -u -m dsart.services.dada_drain --key "${KEY_F}" --log-every 32 \
    > "${LOG_D}" 2>&1 &
DRAIN_PID=$!
echo "OK: dada_drain[fada] pid=${DRAIN_PID}"
sleep 0.3

"${MERGE_BIN}" -i "${KEY_A}" -j "${KEY_B}" -o "${KEY_F}" -m -c 12 \
    > "${LOG_M}" 2>&1 &
MERGE_PID=$!
echo "OK: dsaX_merge pid=${MERGE_PID}"
sleep 0.3

"${CAP_BIN}" -j "${DATA_IP}" -i 127.0.0.1 \
    -p "${PORT_A}" -q "${CPORT_A}" -o "${KEY_A}" -f "${HEADER}" \
    > "${LOG_A}" 2>&1 &
CAP_A_PID=$!
"${CAP_BIN}" -j "${DATA_IP}" -i 127.0.0.1 \
    -p "${PORT_B}" -q "${CPORT_B}" -o "${KEY_B}" -f "${HEADER}" \
    > "${LOG_B}" 2>&1 &
CAP_B_PID=$!
echo "OK: cap_a pid=${CAP_A_PID}  cap_b pid=${CAP_B_PID}"

# Wait for shm files.
for i in $(seq 1 30); do
    [[ -f "${SHM_A}" && -f "${SHM_B}" ]] && break
    sleep 0.1
done
if [[ ! -f "${SHM_A}" || ! -f "${SHM_B}" ]]; then
    echo "FAIL: shms did not appear in 3 s (A=$([[ -f ${SHM_A} ]] && echo OK || echo MISS) "
    echo "B=$([[ -f ${SHM_B} ]] && echo OK || echo MISS))"
    tail -10 "${LOG_A}" "${LOG_B}"; exit 1
fi
echo "OK: both shms online"

# Wait for fresh last_seq on both ports.
echo
echo "=== waiting for first fresh last_seq from both shms ==="
python3 - <<PY
import time
from dsart.capture import mon_shm

ports = (${PORT_A}, ${PORT_B})
deadline = time.monotonic() + 5.0
mons = {p: mon_shm.MonShm.open(p) for p in ports}
try:
    while time.monotonic() < deadline:
        ok = True
        for p, m in mons.items():
            s = m.snapshot()
            if s.last_seq_no == 0 or s.n_recv_packets < 100:
                ok = False
                break
        if ok:
            for p, m in mons.items():
                s = m.snapshot()
                print(f"OK: port={p} last_seq_no={s.last_seq_no} n_recv={s.n_recv_packets}")
            break
        time.sleep(0.1)
    else:
        raise SystemExit("FAIL: shm last_seq_no stayed 0 / no packets after 5 s")
finally:
    for m in mons.values():
        m.close()
PY

# Arm both UTC_STARTs to the SAME specnum (= max(last_seq) + 30k) so
# the captures start the same block frame. This mirrors the
# orchestrator's UTC_START verb behaviour (one verb -> both ports
# armed identically via _send_utc_udp).
ARM_SEQ=$(python3 -c "
from dsart.capture import mon_shm
vals = []
for p in (${PORT_A}, ${PORT_B}):
    with mon_shm.MonShm.open(p) as m:
        vals.append(m.snapshot().last_seq_no)
print(max(vals) + 30000)
")
echo
echo "=== arming UTC_START=${ARM_SEQ} on both ports ==="
python3 -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for port in (${CPORT_A}, ${CPORT_B}):
    sock.sendto(b'UTC_START-${ARM_SEQ}', ('127.0.0.1', port))
sock.close()
print('OK: sent UTC_START-${ARM_SEQ} to both control ports')
"

# Soak.
echo
echo "=== ${DURATION_S} s soak ==="
python3 - <<PY
import subprocess, time
from dsart.capture import mon_shm

T = ${DURATION_S}
INTERVAL = 5.0

def dada_dbmetric(key):
    """Return {full, free} for ring ``key``."""
    try:
        out = subprocess.check_output(
            ["dada_dbmetric", "-k", key], text=True, timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return {}
    d = {}
    for tok in out.replace(",", " ").split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            try: d[k] = int(v)
            except: pass
    if not d and "," in out:
        # positional CSV: nbufs,nfull,nclear,...
        fields = out.splitlines()[0].split(",")
        if len(fields) >= 2:
            try:
                d["nbufs"] = int(fields[0])
                d["full"] = int(fields[1])
            except: pass
    return d

mons = {p: mon_shm.MonShm.open(p) for p in (${PORT_A}, ${PORT_B})}
try:
    prev = {p: m.snapshot() for p, m in mons.items()}
    t0 = time.monotonic()
    print(f"{'t':>4} | "
          f"{'A.block':>7} {'A.pkt/s':>8} {'A.kdrop/s':>10} "
          f"{'B.block':>7} {'B.pkt/s':>8} {'B.kdrop/s':>10} "
          f"| dada/eada/fada (full/nbufs)")
    print("-" * 120)
    while True:
        t = time.monotonic() - t0
        if t >= T: break
        time.sleep(INTERVAL)
        cur = {p: m.snapshot() for p, m in mons.items()}
        a, b = cur[${PORT_A}], cur[${PORT_B}]
        ap, bp = prev[${PORT_A}], prev[${PORT_B}]
        da = (a.n_recv_packets - ap.n_recv_packets) / INTERVAL
        db = (b.n_recv_packets - bp.n_recv_packets) / INTERVAL
        kda = (a.n_dropped_kernel - ap.n_dropped_kernel) / INTERVAL
        kdb = (b.n_dropped_kernel - bp.n_dropped_kernel) / INTERVAL
        m_dada = dada_dbmetric("${KEY_A}")
        m_eada = dada_dbmetric("${KEY_B}")
        m_fada = dada_dbmetric("${KEY_F}")
        def fmt(m):
            f = m.get("full", "?")
            n = m.get("nbufs", "?")
            return f"{f}/{n}"
        print(f"{t:>4.0f} | "
              f"{a.n_block_writes:>7d} {da:>8.0f} {kda:>10.0f} "
              f"{b.n_block_writes:>7d} {db:>8.0f} {kdb:>10.0f} "
              f"| {fmt(m_dada)} {fmt(m_eada)} {fmt(m_fada)}")
        prev = cur
    final = {p: m.snapshot() for p, m in mons.items()}
finally:
    for m in mons.values():
        m.close()

print()
print("=== final summary ===")
for p, s in final.items():
    print(f"  port {p}:")
    print(f"    arm_state          = {s.arm_state.name}")
    print(f"    n_block_writes     = {s.n_block_writes}")
    print(f"    n_recv_packets     = {s.n_recv_packets}")
    print(f"    n_recv_bytes       = {s.n_recv_bytes}")
    print(f"    n_dropped_payload  = {s.n_dropped_payload}")
    print(f"    n_dropped_kernel   = {s.n_dropped_kernel}")
    print(f"    n_seq_skipped      = {s.n_seq_skipped}")
    print(f"    n_too_late         = {s.n_too_late}")
    print(f"    n_recv_errors      = {s.n_recv_errors}")
PY

echo
echo "=== drain log tail ==="
tail -5 "${LOG_D}"
echo "=== merge log tail ==="
tail -5 "${LOG_M}"

echo
echo "=== M7.5 Test 2 done ==="
