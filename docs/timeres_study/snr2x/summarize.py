"""Band summaries of results/snr2x.npz -> results/snr2x_summary.json (+ printed tables)."""
import json

import numpy as np

Z = np.load("results/snr2x.npz")
W = Z["widths_ms"]
BANDS = [(50, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
KEYS = {"current": "current", "prop_q026": "2x (stage-1 0.26 ms)", "prop_q052": "2x (stage-1 0.52 ms)",
        "single": "single-shot 0.26 ms/244k"}
SEARCHED = {"current": (100, 1500), "single": (80, 1500)}       # outside = not searched


def band_stats(key, lo, hi):
    dm = Z[f"{key}_dm"]
    ch = Z[f"{key}_chain"]                       # (n, 4, nw): smear, sample, dmgrid, total
    m = (dm >= lo) & (dm < hi)
    if key in SEARCHED:                          # outside the searched range = not searched
        m &= (dm >= SEARCHED[key][0]) & (dm <= SEARCHED[key][1])
    if m.sum() < 5:
        return None
    c = ch[m]
    smear, sample, grid, total = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return {"n": int(m.sum()),
            "total_mean": total.mean(0).tolist(), "total_p10": np.percentile(total, 10, axis=0).tolist(),
            "total_min": total.min(0).tolist(),
            "f_smear": smear.mean(0).tolist(), "f_sample": (sample / smear).mean(0).tolist(),
            "f_dmgrid": (grid / sample).mean(0).tolist(), "f_integer": (total / grid).mean(0).tolist()}


out = {"widths_ms": W.tolist(), "bands": BANDS, "stats": {}}
for k in KEYS:
    out["stats"][k] = {f"{lo}-{hi}": band_stats(k, lo, hi) for lo, hi in BANDS}
json.dump(out, open("results/snr2x_summary.json", "w"), indent=1)

print("mean recovered S/N (fraction of an ideal matched filter), widths (ms):", " ".join(f"{w:5g}" for w in W))
for lo, hi in BANDS:
    print(f"DM {lo}-{hi}")
    for k, lab in KEYS.items():
        s = out["stats"][k][f"{lo}-{hi}"]
        if s is None:
            print(f"  {lab:22s} not searched")
            continue
        print(f"  {lab:22s} " + " ".join(f"{v:5.3f}" for v in s["total_mean"])
              + "   p10 " + " ".join(f"{v:4.2f}" for v in s["total_p10"]))
print("\nloss factors at DM 500-1000 (smear / sampling / DM grid / integer delays):")
for k, lab in KEYS.items():
    s = out["stats"][k]["500-1000"]
    for i, w in enumerate(W):
        if w in (0.1, 0.5, 1.0, 4.0):
            print(f"  {lab:22s} W {w:4g} ms: {s['f_smear'][i]:.3f}  {s['f_sample'][i]:.3f}  "
                  f"{s['f_dmgrid'][i]:.3f}  {s['f_integer'][i]:.3f}  -> {s['total_mean'][i]:.3f}")
