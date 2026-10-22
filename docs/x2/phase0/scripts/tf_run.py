"""Grid run: recovered S/N / ideal for every filter bank x pulse shape x width x DM at the proposed 2x
op-point (proposed(q1=262.144, n_sub=2)).  2 worker processes (Phase 0 CPU budget), nice 10.

  python tf_run.py main    -> results/time_filters_main.npz   (run_snr2x WIDTHS_MS, DM 50-3000 step 4
                                                                with the same jitter as snr2x prop_q026)
  python tf_run.py dense   -> results/time_filters_dense.npz  (31 widths, 4 per octave 0.088-16 ms, on
                                                                60 DMs) for the between-width scalloping
  python tf_run.py main_hybrid | dense_hybrid  -> the same grids for the hybrid banks (boxcar at w <= 2,
                                                  B2 / B3 above) with a boxcar reference in the same run
  python tf_run.py main_x | dense_x            -> the 8-filter bank b3x (boxcar 1, 2, [1,2,1], B3 4..64)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tf_common import BANK_KEYS, RES, SHAPES, WIDTHS_MS, M, scheme

STEP, JIT = 4.0, 1.5
_S = {}


def dms_main():
    """Identical to run_snr2x._dms(50, 3000, seed 12) (the prop_q026 key)."""
    base = np.arange(50.0, 3000.0 + 1e-9, STEP)
    rng = np.random.default_rng(12)
    return np.clip(base + rng.uniform(-JIT, JIT, base.size), 50.0, 3000.0)


def dms_dense():
    rng = np.random.default_rng(7)
    return np.sort(np.concatenate([np.linspace(55, 1590, 40) + rng.uniform(-3, 3, 40),
                                   np.linspace(1610, 2990, 20) + rng.uniform(-3, 3, 20)]))


DENSE_WIDTHS_MS = 2.0 ** (np.arange(-14, 17) / 4.0)                   # 0.088 .. 16 ms


HYBRID_KEYS = ("boxcar", "b2h", "b3h")     # boxcar at w <= 2 (B-splines degenerate), B-spline above
X_KEYS = ("boxcar", "b3x")                 # 8 filters: boxcar 1, 2, triangle [1,2,1], B3 4..64


def _work(args):
    dm, widths, banks = args
    if "s" not in _S:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        _S["s"] = scheme()
    return dm, M.evaluate_banks(_S["s"], float(dm), [w * 1e3 for w in widths], SHAPES, banks)


def main(mode):
    """mode: main | dense | main_hybrid | dense_hybrid."""
    os.nice(10)
    banks = HYBRID_KEYS if mode.endswith("_hybrid") else (X_KEYS if mode.endswith("_x") else BANK_KEYS)
    if mode.startswith("main"):
        dms, widths = dms_main(), np.array(WIDTHS_MS)
    else:
        dms, widths = dms_dense(), DENSE_WIDTHS_MS
    res = np.zeros((len(dms), len(SHAPES), len(banks), len(widths)))
    idx = {float(d): i for i, d in enumerate(dms)}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for n, (dm, arr) in enumerate(ex.map(_work, [(d, widths, banks) for d in dms], chunksize=4)):
            res[idx[float(dm)]] = arr
            if n % 50 == 0:
                print(f"{n}/{len(dms)} [{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(RES, f"time_filters_{mode}.npz"), dm=dms, widths_ms=widths, snr=res,
             shapes=np.array(SHAPES), banks=np.array(banks))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
