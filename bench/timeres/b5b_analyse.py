"""Analyse b5b_corr_nic.sh output: per node and window, NIC ingress rate and drop fractions.
Usage: python b5b_analyse.py <dir with marks.txt and <node>.txt> ["n02 n09 n13"]"""
import json
import sys

import numpy as np

d = sys.argv[1].rstrip("/") + "/"
marks = [ln.split() for ln in open(d + "marks.txt") if ln.strip()]
out = {}
for node in sys.argv[2].split() if len(sys.argv) > 2 else ("n04", "n11", "n22"):
    try:
        a = np.loadtxt(d + node + ".txt")
    except OSError:
        continue
    t = a[:, 0]
    out[node] = {}
    for name, t0, t1 in marks:
        t0, t1 = float(t0) + 3, float(t1) - 3          # stay inside the window
        i = np.flatnonzero((t >= t0) & (t <= t1))
        if len(i) < 2:
            continue
        j0, j1 = i[0], i[-1]
        dt = t[j1] - t[j0]
        db, dp, dd, do, du = (a[j1, 1:] - a[j0, 1:])
        out[node][name] = {"gbps_phy": 8 * db / dt / 1e9, "pkts": dp, "discard_frac": dd / max(dp, 1),
                           "out_of_buffer_frac": do / max(dp, 1), "udp_rcvbuf_err_per_s": du / dt}
        r = out[node][name]
        print(f"{node} {name:9s} phy {r['gbps_phy']:6.2f} Gb/s  discards {r['discard_frac']:.2e}  "
              f"out_of_buffer {r['out_of_buffer_frac']:.2e}  udp rcvbuf err/s {r['udp_rcvbuf_err_per_s']:.0f}")
json.dump(out, open(d + "b5b_summary.json", "w"), indent=1)
