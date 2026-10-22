"""Review checks, part 2: (a) quantisation efficiency with today's per-row max-abs scaling (linear for
bright bursts, no clipping) at 4/5/6 bits; (b) corner-turn load with framing for every candidate
format, and the per-node clean ingress each needs to stay at <= 80%."""
import json
import math

import numpy as np

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

rng = np.random.default_rng(5)


def eta_amax(bits, n_entries, n_rows=3000):
    """Each row (stream, sample) of n_entries complex values; re and im quantised with one row
    scale = amax/(2^(bits-1) - 0.5), mid-rise levels, no clipping (as today's cint8 amax/127)."""
    x = rng.standard_normal((n_rows, 2 * n_entries))
    top = 2 ** (bits - 1) - 0.5
    step = np.abs(x).max(axis=1, keepdims=True) / top
    q = (np.floor(x / step) + 0.5) * step
    return float(np.corrcoef(x.ravel(), q.ravel())[0, 1])


out = {"eta_amax": {}}
for b in (4, 5, 6, 8):
    out["eta_amax"][b] = {c: eta_amax(b, c) for c in (1773, 2112, 2483)}
    print(f"{b}-bit, per-row amax scale: eta " + "  ".join(f"C={c}: {v:.4f}" for c, v in out['eta_amax'][b].items()))

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
T1, T2 = 1048.576, 524.288


def load(cells, n_sub, bits, k=4):
    """Total corner-turn Gb/s on the wire (UDP payload + 72 B header + 28 B IP/UDP + 38 B Ethernet
    framing per datagram of k samples), for the two-tier plan of snr2x proposed()."""
    tot = 0.0
    for lo, hi, T in ((50, 1600, T2), (1600, 3000, T1)):
        n_c = math.ceil(0.0206 * (hi - lo) * (T1 / T) / n_sub)
        streams = n_c * n_sub * 16
        per_sample = cells * 2 * bits / 8 + 4                     # entries + fp32 row scale
        datagram = k * per_sample + 72 + 28 + 38
        tot += streams * (1e6 / T) / k * datagram * 8
    return tot / 1e9


rows = []
for name, cells, n_sub in (("gridded n_sub 2", 2483, 2), ("gridded n_sub 4", 2112, 4),
                           ("per-baseline n_sub 2", 1773, 2), ("per-baseline n_sub 4", 1773, 4)):
    for bits in (4, 5, 6):
        k = 4 if bits == 4 else (3 if bits == 5 else 2)
        while k * (cells * 2 * bits / 8 + 4) > 8900:
            k -= 1
        g = load(cells, n_sub, bits, k)
        rows.append({"format": name, "bits": bits, "k": k, "total_gbps": g, "per_node": g / 4,
                     "clean_needed_80pct": g / 4 / 0.8})
        print(f"{name:22s} cint{bits} k={k}: {g:5.1f} Gb/s total, {g/4:5.2f}/node -> needs >= {g/4/0.8:5.2f} Gb/s clean")
out["load"] = rows
json.dump(out, open("results/review_checks2.json", "w"), indent=1)
