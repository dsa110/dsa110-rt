"""Study 1d: corner-turn load per format with the REAL per-stream value counts, heavy values and
framing, per search node (interleaved DM ownership), and the clean ingress each node needs at <= 80%.

Wire (docs/x2/SPEC.md section 5): per sample n_heavy x 2 B (cint8 re, im) + the main values packed at
b bits per component, padded to a byte; k = floor(8900 / bytes per sample) samples per datagram;
per datagram 72 B frame-v2 header + 28 B IP/UDP + 38 B Ethernet (preamble 8, header 14, FCS 4,
inter-frame gap 12).  Tiers (schemes.proposed): A DM 50-1600 at 524.288 us, B 1600-3000 at
1048.576 us; coarse count ceil(0.0206 dDM (T1/T) / n_sub) -> n_sub 2: 32 + 15, n_sub 4: 16 + 8.
Global coarse index d = tier-A 0..nA-1 then tier B; owner half = d % 8, node = half // 2.
Also the S/N table: dedispersion vs the model (snr2x), imaging (study 4), quantisation (this study).
-> results/rowrule_load.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from valuemaps import RES, load

T1, T2 = 1048.576, 524.288
HDR = 72 + 28 + 38
MAXPAY = 8900
SNR2X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../timeres_study/snr2x/results/snr2x.npz")


def tiers(n_sub):
    out = []
    for lo, hi, T in ((50.0, 1600.0, T2), (1600.0, 3000.0, T1)):
        out.append((math.ceil(0.0206 * (hi - lo) * (T1 / T) / n_sub), T))
    return out


def stream_bytes(fmt, bits, heavy_frac):
    """Per (g, s) stream: bytes per sample on the wire incl. per-datagram overheads."""
    vm = load(fmt)
    n_all = vm["n"]
    thr = heavy_frac * n_all.sum() if heavy_frac else np.inf
    per = []
    for sid in np.unique(vm["stream"]):
        n = n_all[vm["stream"] == sid]
        nh = int((n >= thr).sum())
        nm = n.size - nh
        bps = nh * 2 + math.ceil(nm * 2 * bits / 8)
        k = MAXPAY // bps
        per.append(dict(n_values=int(n.size), n_heavy=nh, bytes_per_sample=bps, k=k,
                        wire_bytes_per_sample=(k * bps + HDR) / k))
    return per


def loads(fmt, bits, heavy_frac):
    n_sub = int(load(fmt)["n_sub"])
    per = stream_bytes(fmt, bits, heavy_frac)
    per_dm_bps = sum(p["wire_bytes_per_sample"] for p in per)       # one coarse DM, all streams
    dms = []
    for nc, T in tiers(n_sub):
        dms += [1e6 / T] * nc                                          # samples/s per stream
    gbps_dm = np.array(dms) * per_dm_bps * 8 / 1e9
    node = np.zeros(4)
    for d, g in enumerate(gbps_dm):
        node[(d % 8) // 2] += g
    dgram_s = sum(np.array(dms).sum() * 1.0 / p["k"] for p in per)
    return dict(total_gbps=float(gbps_dm.sum()), node_gbps=node.tolist(), node_max=float(node.max()),
                clean_needed_80=float(node.max() / 0.8), n_coarse=[t[0] for t in tiers(n_sub)],
                values_per_stream_mean=float(np.mean([p["n_values"] for p in per])),
                heavy_per_stream_mean=float(np.mean([p["n_heavy"] for p in per])),
                k_range=[int(min(p["k"] for p in per)), int(max(p["k"] for p in per))],
                datagrams_per_s_total=float(dgram_s),
                datagrams_per_s_node_max=float(dgram_s * node.max() / gbps_dm.sum()))


def dedisp_ratio():
    """snr2x: mean total S/N of proposed n_sub 4 / n_sub 2 (same DM sampling), per width."""
    z = np.load(SNR2X)
    a, b = z["prop_q026_chain"][:, 3, :], z["prop_ns4_chain"][:, 3, :]
    dm_a, dm_b = z["prop_q026_dm"], z["prop_ns4_dm"]
    w = z["widths_ms"]
    out = {}
    for i, wm in enumerate(w):
        out[f"{wm:g}ms"] = float(np.mean(b[:, i]) / np.mean(a[:, i]))
    out["all_widths_0.1-16ms"] = float(np.mean(b[:, 1:]) / np.mean(a[:, 1:]))
    out["same_dm_grid"] = bool(np.allclose(dm_a, dm_b))
    return out


def main():
    out = {"loads": {}, "dedisp_ns4_over_ns2": dedisp_ratio()}
    print(out["dedisp_ns4_over_ns2"], flush=True)
    for fmt in ("g_ns2", "g_ns4", "b_ns2", "b_ns4"):
        for bits in (4, 5):
            for hf in (1e-4, 3e-5):
                key = f"{fmt}|cint{bits}|heavy{hf:g}"
                out["loads"][key] = loads(fmt, bits, hf)
                r = out["loads"][key]
                print(f"{key:26s} total {r['total_gbps']:5.1f} Gb/s, node max {r['node_max']:5.2f} "
                      f"(nodes {' '.join(f'{x:.2f}' for x in r['node_gbps'])}) -> clean {r['clean_needed_80']:5.2f}; "
                      f"values/stream {r['values_per_stream_mean']:.0f}, heavy {r['heavy_per_stream_mean']:.0f}, "
                      f"k {r['k_range']}, dgram/s node max {r['datagrams_per_s_node_max']:.0f}", flush=True)
    json.dump(out, open(os.path.join(RES, "rowrule_load.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
