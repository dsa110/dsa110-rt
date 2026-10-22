"""Grid: loss chain vs DM and width for each scheme -> results/snr2x.npz.  ~1 min per scheme, 6 CPU.
Usage: python run_snr2x.py [scheme keys ...]   (default all; named keys are (re)computed and merged)."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import snr_model as M
from schemes import T2, T4, current, proposed, single_shot

WIDTHS_MS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
SCHEMES = {"current": (current, (60.0, 1700.0)),
           "prop_q026": (lambda: proposed(q1=T4), (50.0, 3000.0)),
           "prop_q052": (lambda: proposed(q1=T2), (50.0, 3000.0)),
           "single": (single_shot, (50.0, 1700.0)),
           "prop_ns4": (lambda: proposed(q1=T4, n_sub=4), (50.0, 3000.0))}
STEP, JIT = 4.0, 1.5
_S = {}


def _dms(lo, hi, seed):
    base = np.arange(lo, hi + 1e-9, STEP)
    rng = np.random.default_rng(seed)
    return np.clip(base + rng.uniform(-JIT, JIT, base.size), lo, hi)


def _work(args):
    key, dm = args
    if key not in _S:
        _S[key] = SCHEMES[key][0]()
    r = M.evaluate(_S[key], float(dm), list(WIDTHS_MS * 1e3))
    return key, dm, np.stack([r["smear"], r["sample"], r["dmgrid"], r["total"]])


def main():
    os.nice(10)
    keys = sys.argv[1:] or list(SCHEMES)
    jobs = []
    dms = {}
    for i, (key, (_, (lo, hi))) in enumerate(SCHEMES.items()):
        if key in keys:
            dms[key] = _dms(lo, hi, 11 + i)
            jobs += [(key, d) for d in dms[key]]
    res = {k: np.zeros((len(dms[k]), 4, len(WIDTHS_MS))) for k in keys}
    idx = {k: {float(d): i for i, d in enumerate(dms[k])} for k in keys}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        for n, (key, dm, arr) in enumerate(ex.map(_work, jobs, chunksize=8)):
            res[key][idx[key][float(dm)]] = arr
            if n % 300 == 0:
                print(f"{n}/{len(jobs)} [{time.time() - t0:.0f}s]", flush=True)
    os.makedirs("results", exist_ok=True)
    out = dict(np.load("results/snr2x.npz")) if os.path.exists("results/snr2x.npz") else {}
    out["widths_ms"] = WIDTHS_MS
    for k in keys:
        out[f"{k}_dm"] = dms[k]
        out[f"{k}_chain"] = res[k]             # (n_dm, [smear, sample, dmgrid, total], n_w)
    np.savez("results/snr2x.npz", **out)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
