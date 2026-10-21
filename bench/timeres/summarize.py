"""Summarise bench/timeres results.jsonl into the numbers the report needs."""
import json
import sys
from collections import defaultdict

rows = [json.loads(ln) for ln in open(sys.argv[1]) if ln.strip()]
by = defaultdict(list)
for r in rows:
    by[r["bench"]].append(r)


def show(bench, cols):
    print(f"\n## {bench}")
    for r in by.get(bench, []):
        if not r.get("ok"):
            print(f"  {r['case']:40s} FAILED {r.get('error')}")
            continue
        print(f"  {r['case']:40s} " + "  ".join(f"{c}={r[c]:.3g}" for c in cols if c in r))


show("b1_fft", ["us_per_image", "images_per_s"])
show("b2_detector", ["us_per_image", "ms"])
show("b3_owner_e2e", ["us_per_image", "ms_combine", "ms_image", "ms_detect",
                      "n_fdm_max_realtime", "peak_gib"])
show("b4_xengine", ["ms_per_block", "tflops", "frac_of_block", "int8_tops", "fp16_tflops",
                    "ratio", "h2d_gbs", "d2h_gbs"])
show("b6_zoom", ["ms_beamform_per_block", "ms_dedisp", "ms_per_candidate"])

e2e = [r for r in by.get("b3_owner_e2e", []) if r.get("ok")]
if e2e:
    best = max(e2e, key=lambda r: r["n_fdm_max_realtime"])
    print(f"\nBest owner config: {best['case']}  -> {best['n_fdm_max_realtime']} fine DMs "
          f"per GPU in real time ({best['us_per_image']:.2f} us/image).")
    print("Model needs ~44 fine DMs/owner (24 owners, DM<=1500 native) or ~78 (DM<=3000).")
