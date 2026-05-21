"""Long-soak monitor for M7.5 Phase B (16x4 real-mode fleet).

Polls /mon/corr_rt/<cn> and /mon/search_rt/<cn> every INTERVAL_S
seconds, prints a compact one-line summary + flags any
regressions, and appends per-snapshot JSON to a file.

Designed to be called via ``python -u`` from a `nohup` so it can
sit in the background for an hour-plus soak without holding the
parent shell. CTRL-C / SIGTERM stops cleanly and prints the
final summary.

Usage:
    python -u tools/ops/_m75_phaseB_monitor.py [INTERVAL_S] [JSON_LOG]

Defaults: INTERVAL_S=300, JSON_LOG=/tmp/_m75_phaseB_snapshots.jsonl
"""
from __future__ import annotations

import json
import signal
import sys
import time
from typing import Any

from dsautils.dsa_store import DsaStore

CORRS = [3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22]
SEARCH = [(1, "n01"), (2, "n02"), (9, "n09"), (13, "n13")]

INTERVAL_S = int(sys.argv[1]) if len(sys.argv) > 1 else 300
JSON_LOG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/_m75_phaseB_snapshots.jsonl"

_stop = False


def _on_signal(_sig, _frame) -> None:
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


def gd(s: DsaStore, k: str) -> dict[str, Any]:
    try:
        return s.get_dict(k) or {}
    except Exception:
        return {}


def snapshot(s: DsaStore) -> dict[str, Any]:
    snap: dict[str, Any] = {"t": time.time(), "corr": {}, "search": {}}
    # capture rollup
    cap_blocks: list[int] = []
    cap_armed = 0
    cap_total = 0
    kdrop_total = 0
    pdrop_total = 0
    err_total = 0
    seq_skipped_total = 0
    per_port_drops: dict[str, int] = {}
    for cn in CORRS:
        for port in (4011, 4012):
            cap = gd(s, f"/mon/corr_rt/{cn}/capture/{port}")
            if not cap:
                continue
            cap_total += 1
            if cap.get("arm_state") == "WRITING":
                cap_armed += 1
            nb = cap.get("n_block_writes")
            if isinstance(nb, int):
                cap_blocks.append(nb)
            kdrop = cap.get("n_dropped_kernel", 0) or 0
            pdrop = cap.get("n_dropped_payload", 0) or 0
            err = cap.get("n_recv_errors", 0) or 0
            ss = cap.get("n_seq_skipped", 0) or 0
            kdrop_total += kdrop
            pdrop_total += pdrop
            err_total += err
            seq_skipped_total += ss
            per_port_drops[f"cn{cn}.p{port}"] = pdrop
    snap["captures"] = {
        "armed": cap_armed,
        "total": cap_total,
        "blocks_min": min(cap_blocks) if cap_blocks else None,
        "blocks_max": max(cap_blocks) if cap_blocks else None,
        "blocks_spread": (max(cap_blocks) - min(cap_blocks)) if cap_blocks else None,
        "kernel_drops": kdrop_total,
        "payload_drops": pdrop_total,
        "recv_errors": err_total,
        "seq_skipped": seq_skipped_total,
        "per_port_drops": per_port_drops,
    }
    # routine health
    routines_alive = 0
    routines_total = 0
    uptimes: list[float] = []
    for cn in CORRS:
        d = gd(s, f"/mon/corr_rt/{cn}")
        if not d:
            continue
        upt = d.get("uptime_s")
        if isinstance(upt, (int, float)):
            uptimes.append(float(upt))
        for st in (d.get("routines") or {}).values():
            routines_total += 1
            if st.get("alive"):
                routines_alive += 1
    snap["corr_routines"] = {
        "alive": routines_alive,
        "total": routines_total,
        "uptime_min_s": min(uptimes) if uptimes else None,
        "uptime_max_s": max(uptimes) if uptimes else None,
    }
    # search side: just routine alive counts (rx + compute0 + compute1)
    search_alive = 0
    search_total = 0
    search_uptimes = []
    for cn, host in SEARCH:
        d = gd(s, f"/mon/search_rt/{cn}")
        if not d:
            continue
        if isinstance(d.get("uptime_s"), (int, float)):
            search_uptimes.append(float(d["uptime_s"]))
        for st in (d.get("routines") or {}).values():
            search_total += 1
            if st.get("alive"):
                search_alive += 1
    snap["search_routines"] = {
        "alive": search_alive,
        "total": search_total,
        "uptime_min_s": min(search_uptimes) if search_uptimes else None,
        "uptime_max_s": max(search_uptimes) if search_uptimes else None,
    }
    return snap


def fmt_line(snap: dict[str, Any], t0: float) -> str:
    t_rel = (snap["t"] - t0) / 60.0
    c = snap["captures"]
    r = snap["corr_routines"]
    sr = snap["search_routines"]
    flag = ""
    if c["kernel_drops"] > 0:
        flag += " KDROPS!"
    if c["recv_errors"] > 0:
        flag += " RECV_ERR!"
    if c["armed"] < c["total"]:
        flag += f" ARMED_{c['armed']}/{c['total']}!"
    if r["alive"] < r["total"]:
        flag += f" CORR_RT_DEAD_{r['total'] - r['alive']}!"
    if sr["alive"] < sr["total"]:
        flag += f" SEARCH_DEAD_{sr['total'] - sr['alive']}!"
    return (
        f"[t={t_rel:6.1f}min] caps={c['armed']}/{c['total']} "
        f"blocks={c['blocks_max']} (spread={c['blocks_spread']}) "
        f"kdrop={c['kernel_drops']} pdrop={c['payload_drops']} "
        f"err={c['recv_errors']} seq_skip={c['seq_skipped']} "
        f"corr_rou={r['alive']}/{r['total']} "
        f"search_rou={sr['alive']}/{sr['total']}{flag}"
    )


def main() -> int:
    store = DsaStore()
    t0 = time.time()
    print(f"M7.5 Phase B monitor: interval={INTERVAL_S}s log={JSON_LOG}", flush=True)
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(t0))}", flush=True)
    first = snapshot(store)
    print(fmt_line(first, t0), flush=True)
    with open(JSON_LOG, "a") as fh:
        fh.write(json.dumps(first) + "\n")
    prev_kdrop = first["captures"]["kernel_drops"]
    prev_err = first["captures"]["recv_errors"]
    while not _stop:
        # Sleep in 5-s slices so we react to SIGTERM quickly.
        slept = 0.0
        while slept < INTERVAL_S and not _stop:
            time.sleep(min(5.0, INTERVAL_S - slept))
            slept += 5.0
        if _stop:
            break
        snap = snapshot(store)
        # Detect step in kdrop / err vs previous tick.
        cur_kdrop = snap["captures"]["kernel_drops"]
        cur_err = snap["captures"]["recv_errors"]
        delta_flag = ""
        if cur_kdrop > prev_kdrop:
            delta_flag += f" +KDROPS={cur_kdrop - prev_kdrop}"
        if cur_err > prev_err:
            delta_flag += f" +RECV_ERR={cur_err - prev_err}"
        line = fmt_line(snap, t0) + delta_flag
        print(line, flush=True)
        with open(JSON_LOG, "a") as fh:
            fh.write(json.dumps(snap) + "\n")
        prev_kdrop = cur_kdrop
        prev_err = cur_err
    print("monitor stopping; final summary:", flush=True)
    snap = snapshot(store)
    print(fmt_line(snap, t0), flush=True)
    with open(JSON_LOG, "a") as fh:
        fh.write(json.dumps(snap) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
