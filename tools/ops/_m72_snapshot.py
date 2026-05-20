"""Snapshot /mon for the 16x1 fleet: orchestrator state, routine pids,
ring depths, search-side cubes/sec.

Used by tools/ops/_m72_16x1_snapshot.sh — kept Python-side so the JSON
parsing is robust to any new mon-dict fields without rewriting bash.
"""
import json
import sys

from dsautils.dsa_store import DsaStore

CORR_NODES_CN = (3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22)
SEARCH_CN = 1

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
    return {
        "state": state,
        "uptime_s": uptime,
        "routines": rou_states,
        "buffers": bufs,
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
    print("=== SEARCH NODE ===")
    s = summarise_search(SEARCH_CN)
    print(f"  cn={SEARCH_CN} state={s['state']} uptime={s['uptime_s']}s "
          f"routines={s['routines']}")
    print()
    print("=== CORR NODES ===")
    header = f"{'cn':>3} {'state':<9} {'up':>6} {'cap_a':>5} {'cap_b':>5} " \
             f"{'merge':>5} {'cs':>5} {'cf':>5} {'drain':>5} | dada/eada/fada/bada"
    print(header)
    print("-" * len(header))
    for cn in CORR_NODES_CN:
        c = summarise_corr(cn)
        r = c["routines"]
        b = c["buffers"]
        up = (f"{c['uptime_s']:.0f}s"
              if isinstance(c['uptime_s'], (int, float)) else "-")
        # Pick the right cap routine name based on what showed up
        cap_a = r.get("cap_a_junkdb") or r.get("cap_a_real") or "-"
        cap_b = r.get("cap_b_junkdb") or r.get("cap_b_real") or "-"
        print(f"{cn:>3} {c['state']:<9} {up:>6} {cap_a:>5} {cap_b:>5} "
              f"{r.get('merge','-'):>5} {r.get('corr_slow','-'):>5} "
              f"{r.get('corr_fast','-'):>5} {r.get('bada_drain','-'):>5} "
              f"| {b.get('dada','-')}/{b.get('eada','-')}/"
              f"{b.get('fada','-')}/{b.get('bada','-')}")


if __name__ == "__main__":
    sys.exit(main() or 0)
