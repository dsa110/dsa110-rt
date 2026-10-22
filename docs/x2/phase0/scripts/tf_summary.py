"""Summarise the time-filter grids -> results/time_filters.json (numbers quoted in time_filters.md)."""
import json
import os

import numpy as np

from tf_common import RES, M


def merged(kind):
    """Concatenate the bank axis of <kind>.npz, <kind>_hybrid.npz and <kind>_x.npz (the boxcar reference
    in every file must agree exactly)."""
    a = np.load(os.path.join(RES, f"time_filters_{kind}.npz"))
    banks, S = [str(b) for b in a["banks"]], a["snr"]
    for suffix in ("_hybrid", "_x"):
        p = os.path.join(RES, f"time_filters_{kind}{suffix}.npz")
        if not os.path.exists(p):
            continue
        h = np.load(p)
        hb = [str(b) for b in h["banks"]]
        assert np.allclose(h["dm"], a["dm"])
        dev = np.abs(h["snr"][:, :, hb.index("boxcar")] - S[:, :, banks.index("boxcar")]).max()
        assert dev < 1e-12, dev
        for i, b in enumerate(hb):
            if b not in banks:
                banks.append(b)
                S = np.concatenate([S, h["snr"][:, :, i:i + 1]], axis=2)
    return a["dm"], a["widths_ms"], S, [str(s) for s in a["shapes"]], banks


dm, W, S, shapes, banks = merged("main")              # S: (n_dm, shape, bank, width)
dd, DW, DS, _, dbanks = merged("dense")
assert dbanks == banks
ib = banks.index("boxcar")
wsel = (W >= 0.1 - 1e-9) & (W <= 16 + 1e-9)
dsel = (DW >= 0.1) & (DW <= 16)
i1 = int(np.argmin(np.abs(W - 1.0)))
tierA, tierB = dm < 1600, dm >= 1600
out = {"op_point": "proposed(q1=262.144, n_sub=2): tier A DM 50-1600 at 0.524 ms, tier B 1600-3000 at 1.049 ms",
       "n_dm_main": int(len(dm)), "widths_ms": W.tolist(), "dense_widths_ms": DW.tolist(),
       "n_dm_dense": int(len(dd)), "banks": {}, "per_width_mean": {}, "dense_per_width_mean": {}}

for b, bk in enumerate(banks):
    e = {}
    for s, sh in enumerate(shapes):
        x, x0 = S[:, s, b][:, wsel], S[:, s, ib][:, wsel]
        r = x / x0
        dx, dx0 = DS[:, s, b][:, dsel], DS[:, s, ib][:, dsel]
        dr = dx.mean(axis=0) / dx0.mean(axis=0)               # per dense width, DM-mean ratio
        e[sh] = dict(
            mean_0p1_16=float(x.mean()), ratio_of_means=float(x.mean() / x0.mean()),
            mean_ratio=float(r.mean()), min_ratio=float(r.min()), max_ratio=float(r.max()),
            frac_points_below_boxcar=float((r < 1 - 1e-6).mean()),
            frac_points_below_0p99=float((r < 0.99).mean()),
            tierA_ratio=float(x[tierA].mean() / x0[tierA].mean()),
            tierB_ratio=float(x[tierB].mean() / x0[tierB].mean()),
            at_1ms=float(S[:, s, b, i1].mean()), at_1ms_ratio=float(S[:, s, b, i1].mean() / S[:, s, ib, i1].mean()),
            dense_ratio=float(dx.mean() / dx0.mean()),
            dense_worst_ratio=float(dr.min()), dense_worst_width_ms=float(DW[dsel][np.argmin(dr)]),
            dense_best_ratio=float(dr.max()), dense_best_width_ms=float(DW[dsel][np.argmax(dr)]))
        out["per_width_mean"].setdefault(sh, {})[bk] = S[:, s, b].mean(axis=0).round(4).tolist()
        out["dense_per_width_mean"].setdefault(sh, {})[bk] = DS[:, s, b].mean(axis=0).round(4).tolist()
    e["all_shapes_ratio_of_means"] = float(np.mean([e[sh]["ratio_of_means"] for sh in shapes]))
    e["all_shapes_at_1ms_ratio"] = float(np.mean([e[sh]["at_1ms_ratio"] for sh in shapes]))
    fl = M.BANKS[bk]
    e["filters"] = [dict(w=f["w"], lengths=f["lengths"], support=f["support"], fwhm=round(f["fwhm"], 3),
                         l2_unnormalised=round(f["l2"], 4), flops_per_output=f["flops_per_output"])
                    for f in fl]
    e["history_rows"] = max(f["support"] for f in fl) - 1
    e["flops_per_output_bank"] = sum(f["flops_per_output"] for f in fl)
    out["banks"][bk] = e

for sh in shapes:
    print(f"\n== {sh} pulses: S/N / ideal, mean over DM 50-3000 x widths 0.1-16 ms ==")
    for bk in banks:
        e = out["banks"][bk][sh]
        print(f"{bk:6s} mean {e['mean_0p1_16']:.4f} x{e['ratio_of_means']:.4f} (A x{e['tierA_ratio']:.4f}, "
              f"B x{e['tierB_ratio']:.4f})  1 ms {e['at_1ms']:.4f} x{e['at_1ms_ratio']:.4f}  "
              f"min x{e['min_ratio']:.3f} max x{e['max_ratio']:.3f} <1: {e['frac_points_below_boxcar']:.2f} "
              f"<0.99: {e['frac_points_below_0p99']:.2f}  dense x{e['dense_ratio']:.4f} worst "
              f"x{e['dense_worst_ratio']:.4f}@{e['dense_worst_width_ms']:.2f} best x{e['dense_best_ratio']:.4f}"
              f"@{e['dense_best_width_ms']:.2f}")
    print("per width:", W.tolist())
    for bk in banks:
        print(f"  {bk:6s}", out["per_width_mean"][sh][bk])
for bk in banks:
    e = out["banks"][bk]
    print(bk, "all shapes x", round(e["all_shapes_ratio_of_means"], 4), "1 ms x",
          round(e["all_shapes_at_1ms_ratio"], 4), "history", e["history_rows"], "flops/out",
          e["flops_per_output_bank"])
for extra in ("trials", "scallop", "banks", "rednoise"):
    p = os.path.join(RES, f"time_filters_{extra}.json")
    if os.path.exists(p):
        out[f"{extra}_json"] = json.load(open(p))
if "b3x" in out["banks"]:
    bx = out["banks"]["b3x"]
    rn = out.get("rednoise_json", {}).get("cases", {}).get("residual share kept", {})
    out["key"] = {
        "recommended_bank": "b3x: boxcar 1, boxcar 2, triangle (2,2), B3 (3,3,3) (6,6,7) (12,13,13) (25,25,26) (50,50,51)",
        "history_rows": bx["history_rows"], "adds_per_output": bx["flops_per_output_bank"],
        "boxcar_history_rows": out["banks"]["boxcar"]["history_rows"],
        "gain_vs_boxcar_white": {sh: round(bx[sh]["ratio_of_means"], 4) for sh in shapes},
        "gain_vs_boxcar_1ms": {sh: round(bx[sh]["at_1ms_ratio"], 4) for sh in shapes},
        "gain_vs_boxcar_dense": {sh: round(bx[sh]["dense_ratio"], 4) for sh in shapes},
        "gain_vs_boxcar_red_noise_conservative": {sh: round(rn.get(f"b3x_gain_{sh}", {}).get("with_red_noise", float("nan")), 4)
                                                  for sh in shapes},
        "false_alarm_rate_vs_boxcar_u9p55": round(out["trials_json"]["rate_u9.55_rel_boxcar"]["b3x"], 3)
        if "trials_json" in out else None,
        "sigma_k_seeds_l2": [f["l2_unnormalised"] for f in bx["filters"]],
    }
    print(json.dumps(out["key"], indent=1))
json.dump(out, open(os.path.join(RES, "time_filters.json"), "w"), indent=1)
