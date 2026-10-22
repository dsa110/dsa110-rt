"""Checks before the grid run: (1) bank table; (2) the generic filter engine with the boxcar bank
reproduces the original snr_binned / evaluate()['total'] of the unchanged model at a few DMs."""
import json
import os
import time

import numpy as np

from tf_common import M, scheme

os.nice(10)
rows = []
for bk, fl in M.BANKS.items():
    for f in fl:
        rows.append(dict(bank=bk, w=f["w"], lengths=f["lengths"], support=f["support"],
                         fwhm=round(f["fwhm"], 3), l2=round(f["l2"], 4), stages=f["stages"],
                         flops_per_output=f["flops_per_output"]))
        print(f"{bk:6s} w={f['w']:2d} lengths={str(f['lengths']):14s} support={f['support']:3d} "
              f"fwhm={f['fwhm']:7.3f} l2={f['l2']:9.3f} flops/out={f['flops_per_output']}")

S = scheme()
t0 = time.time()
dev = []
for dm in (75.3, 412.0, 1203.7, 1650.2, 2811.0):
    r = M.evaluate(S, dm, [w * 1e3 for w in (0.1, 1.0, 8.0)])["total"]
    b = M.evaluate_banks(S, dm, [w * 1e3 for w in (0.1, 1.0, 8.0)], shapes=("gauss",),
                         bank_keys=("boxcar",))[0, 0]
    dev.append(float(np.max(np.abs(r - b))))
    print(f"DM {dm:7.1f}: original {np.round(r, 5)}  bank engine {np.round(b, 5)}")
print(f"max |diff| = {max(dev):.2e}   [{time.time() - t0:.1f}s]")
json.dump({"bank_table": rows, "boxcar_engine_vs_original_maxdiff": max(dev)},
          open(os.path.join(os.path.dirname(__file__), "..", "results", "time_filters_banks.json"), "w"),
          indent=1)
