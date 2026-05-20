"""Pretty-print per-stage percentile comparison from multiple bench runs."""
import json
import sys
from pathlib import Path


def show(path):
    with open(path) as f:
        s = json.load(f)
    cfg = s["config"]
    p = s["percentiles_ms"]
    print(
        "T_det={t_det:3d} t={total_p50:5.1f}ms (build={bc_p50:5.1f} "
        "layer1={l1_p50:5.1f} det={det_p50:5.1f})  "
        "ach={ach:.2f} cubes/s".format(
            t_det=cfg["t_det"],
            total_p50=p["total_pipeline"]["p50"],
            bc_p50=p["build_cube"]["p50"],
            l1_p50=p["layer1_norm"]["p50"],
            det_p50=p["detector_forward"]["p50"],
            ach=s["achieved_cubes_per_s"],
        )
    )


for p in sys.argv[1:]:
    try:
        show(p)
    except Exception as e:
        print(f"{p}: {e}")
