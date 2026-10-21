"""Throughput / GPU / network cost model for the time-resolution design points.

Measured anchors (see report appendix for sources):
  search, per 256^2 image, today (n_sub=4, n_fdm=25, b1..b64):
    combine v4 4.08 us (19.6 ms / 4800, n13 0ebac08)
    irfft2 fp16 2.11 us (13.78 ms / 6528, n02 2026-09-22)
    edge mask 1.11 us, Layer-1 0.60 us, detector 5.78 us (37.7 ms / 6528)
    => 13.7 us of GPU kernels; live wall 201.327 ms / 4800 = 41.9 us
  uv cells per (sub-band stream): ~1900 (nfilled.py, real antpos; 2134 live at n_sub=4)
Projected per-image costs for a rewritten search kernel chain are explicit
assumptions (PROJ_*), to be firmed up by bench/timeres/.
"""
from __future__ import annotations

import json

import numpy as np

from physics import DNU_NATIVE_MHZ, F_BOT_MHZ, F_TOP_MHZ, gen_dm_list

F_CTR = 0.5 * (F_TOP_MHZ + F_BOT_MHZ)
TODAY_GPU_US = 4.08 + 2.11 + 1.11 + 0.60 + 5.78       # 13.7
TODAY_WALL_US = 201.327e3 / 4800                      # 41.9
PROJ_FULL_US = 2.1 + 2.0 + 1.5 + 0.3                  # FFT + combine + detector + norm  = 5.9
PROJ_HALF_US = 1.05 + 1.5 + 0.75 + 0.2                # 256x128 grid (FoV/2)          = 3.5
CELLS = 1900.0
N_CORR = 16


def n_fine(lo, hi, T_us, csf, tol=1.25):
    return len(gen_dm_list(lo, hi, T_us, DNU_NATIVE_MHZ * csf, int(round(6144 / csf)), F_CTR, tol=tol))


def point(name, T_us, csf, dm_hi, n_c, n_s, q2_half=False, bits=8, dm_lo=50.0):
    nf = 200 if name.startswith("today") else n_fine(dm_lo, dm_hi, T_us, csf)
    rate = 1e6 / T_us
    img = nf * rate
    stream_rate = rate * (2 if q2_half else 1)
    bytes_per_cell = 2 * bits / 8
    per_stream_gbps = CELLS * stream_rate * bytes_per_cell * 8 / 1e9
    corr_out = n_c * n_s * per_stream_gbps
    owner_in = N_CORR * n_s * per_stream_gbps        # one coarse DM bucket
    return {
        "name": name, "T_us": T_us, "csf": csf, "dm_hi": dm_hi, "n_fine": nf,
        "n_coarse": n_c, "n_sub": n_s, "images_per_s": img,
        "x_today": img / (200 * 1e6 / 1048.576),
        "gpu_today_wall": img * TODAY_WALL_US / 1e6,
        "gpu_today_kernels": img * TODAY_GPU_US / 1e6,
        "gpu_proj_full": img * PROJ_FULL_US / 1e6,
        "gpu_proj_half": img * PROJ_HALF_US / 1e6,
        "corr_out_gbps": corr_out, "owner_in_gbps": owner_in,
        "fleet_gbps": N_CORR * corr_out, "bits": bits, "q2_half": q2_half,
    }


POINTS = [
    point("today (1.05 ms, 244 kHz, 8x4)", 1048.576, 8, 1500, 8, 4),
    point("1.05 ms, 30.5 kHz, 16x8", 1048.576, 1, 1500, 16, 8),
    point("2x: 0.52 ms, 30.5 kHz, 16x8", 524.288, 1, 1500, 16, 8),
    point("4x: 0.26 ms, 244 kHz, 16x8", 262.144, 8, 1500, 16, 8),
    point("4x: 0.26 ms, 61 kHz, 16x8", 262.144, 2, 1500, 16, 8),
    point("4x: 0.26 ms, 30.5 kHz, 16x8", 262.144, 1, 1500, 16, 8),
    point("4x: 0.26 ms, 30.5 kHz, 16x8, cint4", 262.144, 1, 1500, 16, 8, bits=4),
    point("4x to DM 3000, 32x8, cint4", 262.144, 1, 3000, 32, 8, bits=4),
    point("8x: 0.13 ms, 30.5 kHz, 32x8, cint4", 131.072, 1, 1500, 32, 8, bits=4),
]

if __name__ == "__main__":
    hdr = ("design point", "N_fine", "img/s", "x today", "GPU@today", "GPU@proj", "GPU@proj FoV/2",
           "corr out Gb/s", "bucket in Gb/s", "fleet Gb/s")
    print("%-38s %6s %9s %7s %9s %8s %10s %9s %9s %9s" % hdr)
    for p in POINTS:
        print("%-38s %6d %9.3g %7.1f %9.1f %8.1f %10.1f %9.1f %9.1f %9.0f" % (
            p["name"], p["n_fine"], p["images_per_s"], p["x_today"], p["gpu_today_wall"],
            p["gpu_proj_full"], p["gpu_proj_half"], p["corr_out_gbps"], p["owner_in_gbps"],
            p["fleet_gbps"]))
    # DDplan-style: 262 us below DM 1500, 524 us above
    a = n_fine(50, 1500, 262.144, 1) * 1e6 / 262.144
    b = n_fine(1500, 3000, 524.288, 1) * 1e6 / 524.288
    print(f"\nDDplan 50-1500 @262 + 1500-3000 @524 (native): {a + b:.3g} img/s "
          f"(vs {n_fine(50, 3000, 262.144, 1) * 1e6 / 262.144:.3g} all at 262 us)")
    for tol in (1.25, 1.4, 1.5):
        print(f"tol {tol}: N_fine 50-1500 @262 native = {n_fine(50, 1500, 262.144, 1, tol)}")
    json.dump(POINTS, open("results/cost_points.json", "w"), indent=1)
