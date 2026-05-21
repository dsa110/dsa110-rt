"""Snapshot /mon for the 16x1 fleet: orchestrator state, routine pids,
ring depths, search-side cubes/sec.

Used by tools/ops/_m72_16x1_snapshot.sh — kept Python-side so the JSON
parsing is robust to any new mon-dict fields without rewriting bash.
"""
import json
import sys

from dsautils.dsa_store import DsaStore

CORR_NODES_CN = (3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22)
# M7.3 16x4 expanded the search side from 1 -> 4 nodes.
# n01 -> cn 1, n02 -> cn 2, n09 -> cn 9, n13 -> cn 13.
SEARCH_CN_LIST = (1, 2, 9, 13)
SEARCH_CN = 1  # legacy single-node alias for 16x1 callers

store = DsaStore()


def fmt_dada(d):
    if not d:
        return "-"
    if isinstance(d, dict):
        # Newer dada_dbmetric returns key=value parsed dict.
        free = d.get("free")
        full = d.get("full")
        nbufs = d.get("nbufs")
        if free is not None and full is not None and nbufs is not None:
            return f"{full}/{nbufs}"
        return str(d)
    return str(d)


def summarise_corr(cn):
    d = store.get_dict(f"/mon/corr_rt/{cn}") or {}
    state = d.get("state", "?")
    routines = d.get("routines") or {}
    rou_states = {}
    for name, info in routines.items():
        alive = "Y" if info.get("alive") else "n"
        rou_states[name] = alive
    buffers = d.get("buffers") or {}
    bufs = {k: fmt_dada(v) for k, v in buffers.items()}
    uptime = d.get("uptime_s")
    # Capture mon: pull /mon/corr_rt/<cn>/capture/<port> for each
    # SNAP UDP port (4011/4012). The sidecar publishes either a
    # full snapshot or a 'shm_status: missing' placeholder; we
    # collapse to one summary string per port for the snapshot.
    capture_summary = {}
    for port in (4011, 4012):
        cap = store.get_dict(f"/mon/corr_rt/{cn}/capture/{port}") or {}
        if not cap:
            capture_summary[port] = "-"
            continue
        if cap.get("shm_status") == "missing":
            capture_summary[port] = "MISS"
            continue
        arm = cap.get("arm_state", "?")
        gbps = cap.get("rate_gbps", 0.0)
        kdrop = cap.get("rate_kernel_drop_pps", 0)
        degraded = cap.get("degraded", False)
        flag = "!" if degraded or kdrop > 0 else ""
        capture_summary[port] = f"{arm[:4]}/{gbps:.2f}{flag}"
    return {
        "state": state,
        "uptime_s": uptime,
        "routines": rou_states,
        "buffers": bufs,
        "capture": capture_summary,
    }


def summarise_search(cn):
    d = store.get_dict(f"/mon/search_rt/{cn}") or {}
    state = d.get("state", "?")
    routines = d.get("routines") or {}
    rou_states = {name: "Y" if info.get("alive") else "n"
                  for name, info in routines.items()}
    return {
        "state": state,
        "uptime_s": d.get("uptime_s"),
        "routines": rou_states,
    }


def main():
    print("=== SEARCH NODES ===")
    for cn in SEARCH_CN_LIST:
        s = summarise_search(cn)
        print(f"  cn={cn:>2} state={s['state']:<9} uptime={s['uptime_s']!s:>8}s "
              f"routines={s['routines']}")
    print()
    print("=== CORR NODES ===")
    header = (f"{'cn':>3} {'state':<9} {'up':>6} {'cap_a':>5} {'cap_b':>5} "
              f"{'merge':>5} {'cs':>5} {'cf':>5} {'drain':>5} "
              f"{'mon':>4} | dada/eada/fada/bada | cap4011 / cap4012")
    print(header)
    print("-" * len(header))
    for cn in CORR_NODES_CN:
        c = summarise_corr(cn)
        r = c["routines"]
        b = c["buffers"]
        up = (f"{c['uptime_s']:.0f}s"
              if isinstance(c['uptime_s'], (int, float)) else "-")
        cap_a = r.get("cap_a_junkdb") or r.get("cap_a_real") or "-"
        cap_b = r.get("cap_b_junkdb") or r.get("cap_b_real") or "-"
        mon_alive = r.get("capture_control", "-")
        cap_state = c.get("capture", {})
        print(f"{cn:>3} {c['state']:<9} {up:>6} {cap_a:>5} {cap_b:>5} "
              f"{r.get('merge','-'):>5} {r.get('corr_slow','-'):>5} "
              f"{r.get('corr_fast','-'):>5} {r.get('bada_drain','-'):>5} "
              f"{mon_alive:>4} | "
              f"{b.get('dada','-')}/{b.get('eada','-')}/"
              f"{b.get('fada','-')}/{b.get('bada','-')} | "
              f"{cap_state.get(4011, '-'):>10} / {cap_state.get(4012, '-'):>10}")


if __name__ == "__main__":
    sys.exit(main() or 0)
