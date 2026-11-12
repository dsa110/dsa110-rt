"""Evaluate narrow-band rules on re-scored synthetic / truth / 2x-cutout features.

Rule family: flag if  bw_fit_mhz <= BW  and  f_fit >= F  and  fit_sigma >= S   (and optionally ridge_frac >= R)
Truth classes as in ridge_analysis.py. Prints a grid of (BW, F, S) with synthetic TPR (true occ <= 6 MHz),
synthetic FPR (true occ >= 20 MHz incl. scintillated / flat), real FPR (every astrophysical / injected
event) and narrow TPR (voltage-confirmed narrow-band).
"""
import itertools
import json
import os
import sys

import numpy as np

TMP = os.environ.get("DMT_STUDY_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
FRB_LABELLED = {"260715twmx", "260924quga", "261110zzeg", "261110zzeg_recal"}


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


syn = load(f"{TMP}/rs_synth.jsonl")
tru = load(f"{TMP}/rs_truth.jsonl")
cut = load(f"{TMP}/rs_cut.jsonl")


def c3(ev):
    try:
        return json.load(open(f"/dataz/dsa110/candidates/{ev}/C3_decision.json"))
    except (OSError, ValueError):
        return {}


def l3(ev):
    for root in ("/dataz/dsa110/candidates", "/dataz/dsa110/candidates_rejected"):
        try:
            return json.load(open(f"{root}/{ev}/Level3/{ev}.json"))
        except (OSError, ValueError):
            continue
    return {}


volt = {r["name"]: r for r in tru}


def tclass(name, r):
    f1 = r.get("frac_best_1mhz")
    if f1 is not None and f1 == f1 and f1 >= 0.5:
        return "narrow"
    if name[:2] in ("23", "24", "25"):
        return "real:frb_legacy"
    if name in FRB_LABELLED:
        return "real:frb"
    if 150 <= float(r.get("dm_trig", 0)) <= 170:
        return "real:psr"
    if (r.get("snr_fil") or 99) < 6:
        return "other"
    return "other"


items = []          # (name, class, source, feats)
for r in tru:
    items.append((r["name"], tclass(r["name"], r), "volt", r))
for r in cut:
    ev = r["name"]
    meta = l3(ev)
    trig = (meta.get("trigger") or {}).get("class", "")
    if c3(ev).get("is_injection") or (meta.get("injection") or {}).get("is_injection"):
        cls = "real:inj"
    elif ev in volt and tclass(ev, volt[ev]) == "narrow":
        cls = "narrow"
    elif ev in FRB_LABELLED:
        cls = "real:frb"
    elif trig in ("bright_galactic", "bright_pulsar") and 150 <= (r.get("dm") or 0) <= 170:
        cls = "real:psr"
    else:
        cls = "other"
    for kind in ("pix", "max"):
        items.append((ev, cls, f"cut-{kind}", {f"r_{k}": v for k, v in r[kind].items()}))


def flag(f, BW, F, S, R=0.0):
    return bool(f.get("r_ok")) and (f.get("r_bw_fit_mhz") or 1e9) <= BW and (f.get("r_f_fit") or 0) >= F \
        and (f.get("r_fit_sigma") or 0) >= S and (f.get("r_ridge_frac") or 0) >= R


def rates(BW, F, S, R=0.0, kind="cut-pix"):
    s_n = [s for s in syn if s.get("r_ok") and s["occ_true_mhz"] <= 6.0]
    s_b = [s for s in syn if s.get("r_ok") and s["occ_true_mhz"] >= 20.0]
    tpr = np.mean([flag(s, BW, F, S, R) for s in s_n]) if s_n else np.nan
    fpr = np.mean([flag(s, BW, F, S, R) for s in s_b]) if s_b else np.nan
    real = [it for it in items if it[1].startswith("real") and it[2] in ("volt", kind)]
    narrow = [it for it in items if it[1] == "narrow" and it[2] in ("volt", kind)]
    rf = [it[0] + "/" + it[2] for it in real if flag(it[3], BW, F, S, R)]
    nf = np.mean([flag(it[3], BW, F, S, R) for it in narrow]) if narrow else np.nan
    return tpr, fpr, len(s_n), len(s_b), rf, len(real), nf, len(narrow)


if __name__ == "__main__":
    print(f"synthetic {len(syn)}, truth {len(tru)}, cutout events {len(cut)}")
    from collections import Counter
    print("classes:", Counter((it[1], it[2]) for it in items))
    print("\nBW  F    S   | synth TPR(<=6MHz)  FPR(>=20MHz) | real flagged / n | narrow TPR")
    for BW, F, S in itertools.product([4, 6, 8], [0.2, 0.3, 0.4], [8, 15]):
        tpr, fpr, nn, nb, rf, nr, nf, nnar = rates(BW, F, S)
        print(f"{BW:<3} {F:<4} {S:<3} | {tpr:6.3f} (n={nn})  {fpr:6.3f} (n={nb}) | {len(rf)}/{nr} {rf[:3]} | "
              f"{nf:.2f} (n={nnar})")
    # synthetic detail at the default
    BW, F, S = (float(x) for x in (sys.argv[1:4] if len(sys.argv) >= 4 else (6, 0.3, 8)))
    print(f"\n== synthetic at BW {BW} F {F} S {S}: flag rate by true occupied bandwidth x width x S/N")
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 200)]:
        line = []
        for ws, wl in (({1, 2}, "<=1ms"), ({4, 8}, "2-4ms"), ({16, 32}, "8-17ms")):
            for snr_lo, snr_hi in ((0, 15), (15, 1e9)):
                xs = [s for s in syn if s.get("r_ok") and lo <= s["occ_true_mhz"] < hi and s["width"] in ws
                      and snr_lo <= s["snr"] < snr_hi]
                line.append(f"{wl} S/N{'<15' if snr_hi == 15 else '>=15'} "
                            f"{(np.mean([flag(s, BW, F, S) for s in xs]) if xs else float('nan')):.2f}({len(xs)})")
        print(f"  occ {lo}-{hi} MHz: " + "  ".join(line))
    print("\n== scintillated / flat synthetic flagged at the default:")
    for s in syn:
        if s.get("r_ok") and s["occ_true_mhz"] >= 20 and flag(s, BW, F, S):
            print("  ", {k: s.get(k) for k in ("name", "family", "bw_mhz", "nud_mhz", "width", "snr", "dm",
                                                 "occ_true_mhz", "r_bw_fit_mhz", "r_f_fit", "r_fit_sigma",
                                                 "r_ridge_frac")})
