#!/usr/bin/env bash
# _m75_phaseA_n06_full_corr.sh --- M7.5 Phase A.
#
# Run the FULL corr-side dsart_rt pipeline on n06 only with
# captures.mode=real (live SNAP capture). This exercises the
# real-mode wiring end-to-end on a single corr node:
#
#   cap_a_real + cap_b_real  (SNAP UDP 4011 -> dada,  4012 -> eada)
#   capture_control          (POSIX-shm mon publisher sidecar)
#   merge                    (dada + eada -> fada)
#   corr_slow                (fada -> bada)
#   bada_drain               (bada -> /dev/null)
#   corr_fast                (fada -> TX -> 4 search nodes' UDP)
#
# The 4 search nodes are NOT running in this phase -- corr_fast's
# UDP datagrams will hit closed ports and be silently dropped at
# the search-node end. That's intentional: this phase isolates the
# corr-side wiring under real captures so any cross-node search-
# side failure in Phase B can be unambiguously attributed to the
# distributed half (cube assembly, ring slots, etc.) and not to
# the corr-side cap/merge/corr_slow/corr_fast plumbing.
#
# Success criteria:
#   1. Orchestrator on n06 reaches WRITING on both capture shms
#      (port 4011 + 4012) within 5 s of utc_start verb.
#   2. capture mon-keys in etcd (/mon/corr_rt/6/capture/{4011,4012})
#      show n_dropped_kernel=0, n_recv_errors=0 throughout soak.
#   3. mon-dict /mon/corr_rt/6/dsart_rt shows all routines alive
#      (cap_a_real, cap_b_real, capture_control, merge, corr_slow,
#      bada_drain, corr_fast) with healthy ring fullness.
#   4. fada full/nbufs stays low (corr_fast keeping pace; <= 5/70).
#   5. corr_fast in-flight mon-dict shows block-level latency
#      <= 134.218 ms (the real-time budget).
set -uo pipefail

CN=${CN:-6}
HOST=${HOST:-n06}
DURATION_S=${DURATION_S:-60}
SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-10}
OBS_DEC=${OBS_DEC:-53.85}
WARMUP_S=${WARMUP_S:-180}        # corr_fast cold start is ~90 s; 180 = 2x headroom
REPO=/home/ubuntu/proj/dsa110-rt

echo "==============================================================="
echo " M7.5 Phase A: full corr-side pipeline on ${HOST} (cn=${CN})"
echo "   duration=${DURATION_S}s   warmup=${WARMUP_S}s"
echo "==============================================================="

# -- (0) cleanup state on n06 -----------------------------------------------
echo
echo "=== STAGE 0: cleanup n06 ==="
ssh -o ConnectTimeout=10 -n "${HOST}.pro.pvt" "
for round in 1 2; do
  for p in \$(ps -eo pid,cmd | grep -E 'dsart|corr_fast|corr_slow|dada_drain|dada_junkdb|dsaX_merge|dsart_capture' | grep -v grep | awk '{print \$1}'); do
    kill -9 \$p 2>/dev/null
  done
  sleep 2
done
for k in dada eada fada bada gada hada; do dada_db -d -k \$k >/dev/null 2>&1; done
ipcs -m | awk '/ubuntu/ {print \$2}' | xargs -r -I{} ipcrm -m {} 2>/dev/null
rm -f /dev/shm/dsart-capture-* /tmp/dsart-corr-*.ready 2>/dev/null
rm -rf /tmp/dsart-rt-children
mkdir -p /tmp/dsart-rt-children
echo cleanup OK
" 2>&1 | tail -2

# -- (1) push the (already real-mode) YAML config to etcd -------------------
echo
echo "=== STAGE 1: push pipeline_rt YAML (captures.mode=real) to etcd ==="
cd /home/ubuntu/vikram/dev/dsa110-rt
python3 tools/ops/push_dsart_to_etcd.py --instance pipeline_rt 2>&1 | tail -3

# -- (2) launch dsart_rt orchestrator on n06 --------------------------------
echo
echo "=== STAGE 2: launch dsart_rt orchestrator on ${HOST} ==="
ssh -o ConnectTimeout=10 -n "${HOST}.pro.pvt" "
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate dsa110-rt 2>/dev/null
cd ${REPO}
export PYTHONPATH=${REPO}/src
export DSART_RT_LOG_DIR=/tmp/dsart-rt-children
export DSART_RT_GATE_TIMEOUT_S=300
setsid nohup python3 -u -m dsart.services.dsart_rt -in pipeline_rt -cn ${CN} --log-level INFO \
    > /tmp/dsart-rt-pipeline_rt-${CN}.log 2>&1 < /dev/null &
echo \$! > /tmp/dsart-rt-pipeline_rt-${CN}.pid
disown
sleep 3
if kill -0 \$(cat /tmp/dsart-rt-pipeline_rt-${CN}.pid) 2>/dev/null; then
  echo orchestrator alive pid=\$(cat /tmp/dsart-rt-pipeline_rt-${CN}.pid)
else
  echo DEAD
  tail -15 /tmp/dsart-rt-pipeline_rt-${CN}.log
fi
" 2>&1 | tail -2

# -- (3) send start verb (spawns ALL routines via wave1/wave2 gating) -------
echo
echo "=== STAGE 3: send start verb to dsart_rt (val=${OBS_DEC}) ==="
# IMPORTANT: instance=pipeline_rt but namespace=corr_rt -- the
# orchestrator listens on /cmd/${namespace}/${cn}. See
# dsart_rt._build_orchestrator() for the mapping.
python3 -c "
from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/corr_rt/${CN}', {'cmd':'start','val':${OBS_DEC}})
print('sent start to /cmd/corr_rt/${CN}')
"

# -- (4) wait for warmup --------------------------------------------------
echo
echo "=== STAGE 4: warmup wait ${WARMUP_S}s (corr_fast cold start + sentinels) ==="
for i in $(seq 1 $((WARMUP_S / 15))); do
    sleep 15
    elapsed=$((i*15))
    ssh -o ConnectTimeout=5 -n "${HOST}.pro.pvt" "
        cap_shm=0
        [ -f /dev/shm/dsart-capture-4011 ] && cap_shm=\$((cap_shm+1))
        [ -f /dev/shm/dsart-capture-4012 ] && cap_shm=\$((cap_shm+1))
        sent_fast=\$(ls /tmp/dsart-corr-fast-${CN}.ready 2>/dev/null | wc -l)
        sent_slow=\$(ls /tmp/dsart-corr-slow-${CN}.ready 2>/dev/null | wc -l)
        nproc=\$(ps -ef | grep -E 'dsart_capture|dsaX_merge|corr_fast|corr_slow|capture_control|bada_drain' | grep -v grep | wc -l)
        printf '    warmup t=%3ds  shm=%d/2  fast_ready=%d  slow_ready=%d  procs=%d\n' \
            ${elapsed} \$cap_shm \$sent_fast \$sent_slow \$nproc
    " 2>&1 | tail -1
done

# -- (5) read live capture shms to determine ARM_SEQ ------------------------
echo
echo "=== STAGE 5: poll capture shms; compute arm specnum ==="
ARM_SEQ=$(ssh -o ConnectTimeout=10 -n "${HOST}.pro.pvt" "
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate dsa110-rt 2>/dev/null
export PYTHONPATH=${REPO}/src
python3 - <<PY
from dsart.capture import mon_shm
import time
deadline = time.monotonic() + 5.0
vals = []
for p in (4011, 4012):
    with mon_shm.MonShm.open(p) as m:
        while time.monotonic() < deadline:
            s = m.snapshot()
            if s.last_seq_no > 0 and s.n_recv_packets > 1000:
                vals.append(s.last_seq_no)
                break
            time.sleep(0.1)
        else:
            raise SystemExit(f'FAIL: port {p} shm stale')
print(max(vals) + 30000)
PY
" 2>&1 | tail -1)
echo "    ARM_SEQ=${ARM_SEQ}"

# -- (6) send utc_start verb ------------------------------------------------
echo
echo "=== STAGE 6: send utc_start verb (val=${ARM_SEQ}) ==="
python3 -c "
from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/corr_rt/${CN}', {'cmd':'utc_start','val':${ARM_SEQ}})
print('sent utc_start to /cmd/corr_rt/${CN}')
"

# -- (7) soak; poll mon-dict + capture mon-keys -----------------------------
echo
echo "=== STAGE 7: ${DURATION_S}s soak (poll every ${SAMPLE_INTERVAL_S}s) ==="
python3 - <<PY
import time, json
from dsautils.dsa_store import DsaStore

s = DsaStore()
CN = ${CN}
T = ${DURATION_S}
INTERVAL = ${SAMPLE_INTERVAL_S}

def safe_get(k):
    try:
        return s.get_dict(k) or {}
    except Exception as exc:
        return {"_err": repr(exc)}

t0 = time.monotonic()
print(f"{'t':>4} | {'A.arm':<9} {'A.blocks':>8} {'A.kdrop':>8} | "
      f"{'B.arm':<9} {'B.blocks':>8} {'B.kdrop':>8} | "
      f"{'fada f/n':>9} {'bada f/n':>9} | {'state':<10}")
print("-" * 110)
while True:
    t = time.monotonic() - t0
    if t >= T: break
    time.sleep(INTERVAL)
    cap_a = safe_get(f"/mon/corr_rt/{CN}/capture/4011")
    cap_b = safe_get(f"/mon/corr_rt/{CN}/capture/4012")
    dsart = safe_get(f"/mon/corr_rt/{CN}")
    bufs = (dsart.get("buffers") or {})
    def fbn(k):
        m = (bufs.get(k) or {}).get("metric") or {}
        return f"{m.get('nfull','?')}/{m.get('nbufs','?')}"
    state = dsart.get("state", "?")
    aa = cap_a.get("arm_state", "?")
    ab = cap_b.get("arm_state", "?")
    print(f"{t:>4.0f} | {aa:<9} {cap_a.get('n_block_writes','?'):>8} {cap_a.get('n_dropped_kernel','?'):>8} | "
          f"{ab:<9} {cap_b.get('n_block_writes','?'):>8} {cap_b.get('n_dropped_kernel','?'):>8} | "
          f"{fbn('fada'):>9} {fbn('bada'):>9} | {state:<10}")

print()
print("=== final snapshot ===")
for port in (4011, 4012):
    cap = safe_get(f"/mon/corr_rt/{CN}/capture/{port}")
    print(f"  capture port={port}:")
    for k in ("arm_state","n_block_writes","n_recv_packets","n_recv_bytes",
             "n_dropped_payload","n_dropped_kernel","n_seq_skipped",
             "n_too_late","n_recv_errors","last_seq_no","socket_rcvbuf_bytes",
             "rate_gbps_milli","rate_drop_milli"):
        if k in cap: print(f"    {k:<22} = {cap[k]}")

dsart = safe_get(f"/mon/corr_rt/{CN}")
print(f"  dsart_rt state         = {dsart.get('state')}")
print(f"  uptime_s               = {dsart.get('uptime_s')}")
ch = dsart.get("routines", {})
print(f"  routines:")
for nm, st in ch.items():
    print(f"    {nm:<22} pid={st.get('pid')} alive={st.get('alive')}")
print(f"  buffers (nfull / nbufs):")
for k, st in (dsart.get("buffers") or {}).items():
    m = st.get("metric") or {}
    nfull = m.get('nfull', '?')
    nbufs = m.get('nbufs', '?')
    nclear = m.get('nclear', '?')
    print(f"    {k:<6} {nfull}/{nbufs}  nclear={nclear}")
PY

# -- (8) graceful stop ------------------------------------------------------
echo
echo "=== STAGE 8: stop verb + cleanup ==="
python3 -c "
from dsautils.dsa_store import DsaStore
DsaStore().put_dict('/cmd/corr_rt/${CN}', {'cmd':'stop','val':None})
print('sent stop')
"
sleep 5

ssh -o ConnectTimeout=10 -n "${HOST}.pro.pvt" "
for round in 1 2; do
  for p in \$(ps -eo pid,cmd | grep -E 'dsart|corr_fast|corr_slow|dada_drain|dada_junkdb|dsaX_merge|dsart_capture' | grep -v grep | awk '{print \$1}'); do
    kill -9 \$p 2>/dev/null
  done
  sleep 2
done
for k in dada eada fada bada; do dada_db -d -k \$k >/dev/null 2>&1; done
ipcs -m | awk '/ubuntu/ {print \$2}' | xargs -r -I{} ipcrm -m {} 2>/dev/null
rm -f /dev/shm/dsart-capture-* /tmp/dsart-corr-*.ready 2>/dev/null
echo final cleanup OK
" 2>&1 | tail -1

echo
echo "=== M7.5 Phase A done ==="
