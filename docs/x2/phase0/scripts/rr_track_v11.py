"""Study 1c re-run for the row rule v1.1 (ledger S2g, 2026-10-29): rr_track.py's tracking model
with sigma_hat reset to the block estimate when ``est > 1.5 sigma_hat`` (the used-threshold half
of v1.1, ``|re|, |im| <= 1e-6 sigma_hat`` unused, never touches these scenarios: no value is that
small). Every rowrule.md section-3 scenario at tau 3 s, 4- and 5-bit, guard on, v1.0 vs v1.1, plus
the case v1.1 was made for: a 200 s whole-cube excision whose static-sky residue is 1e-9 of the
noise, then the release.

Pass criterion (rowrule.md section 3): every scenario's total within 0.04% of static (eta of a
static sigma_hat); v1.1 must not differ from v1.0 there.
-> results/rowrule_track_v11.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import rr_track as rt
from valuemaps import RES

TAU = 3.0
RESET = 1.5


def run_ema_v11(sig, tau, excl, rng, init_block=0, reset=RESET, excised=None, residue=1e-9,
                floor=1e-6):
    """rr_track.run_ema plus v1.1. ``excised`` (T,) bool: blocks whose data are the static-sky
    residue only (``residue`` x sigma): v1.0 (floor 0) estimates them, v1.1 skips them."""
    T, S = sig.shape
    alpha = 1.0 - math.exp(-rt.DT / tau)
    sh = np.zeros((T, S))
    cur = None
    for t in range(T):
        if t < init_block:
            sh[t] = sig[t]
            continue
        if excised is not None and excised[t]:
            if floor > 0 and cur is not None:                 # v1.1: residue <= floor x sigma_hat
                sh[t] = cur
                continue
            e = residue * sig[t]
        else:
            e = sig[t] if cur is None else rt.block_estimate(sig[t], cur, excl)
        e = e * (1.0 + rt.EST_JITTER * rng.standard_normal(S))
        if cur is None:
            cur = e
        else:
            upd = cur + alpha * (e - cur)
            cur = np.where(e > reset * cur, e, upd) if reset > 0 else upd
        sh[t] = cur
    return sh


def main():
    rng = np.random.default_rng(7)
    sc = rt.scenarios(rng)
    tabs = {b: rt.eta_table(b) for b in (4, 5)}
    static = {b: float(np.interp(0.0, np.log(tabs[b][0]), tabs[b][1])) for b in (4, 5)}
    res = {"tau_s": TAU, "reset_factor": RESET, "static_eta": static, "scenarios": {}}
    worst = {}
    for name, sig in sc.items():
        inits = (0, 8) if name == "startup" else (0,)
        for init in inits:
            for ver, reset in (("v1.0", 0.0), ("v1.1", RESET)):
                sh = run_ema_v11(sig, TAU, False, np.random.default_rng(11), init_block=init, reset=reset)
                for b in (4, 5):
                    r = rt.summarise(sig, sh, *tabs[b], True)
                    r["loss_vs_static"] = 1.0 - r["total_mean"] / static[b]
                    key = f"{name}|init{init}|b{b}|{ver}"
                    res["scenarios"][key] = r
                    print(f"{key:28s} total {r['total_mean']:.5f} static {static[b]:.5f} "
                          f"loss {100 * r['loss_vs_static']:.4f}%", flush=True)
                    if not (name == "startup" and init == 0):     # EMA-from-block-0 is not the rule
                        worst[(b, ver)] = max(worst.get((b, ver), 0.0), r["loss_vs_static"])
    # excision -> release: 60 s of data, 200 s excised (residue 1e-9), 60 s of data
    T1, T2, T3 = (int(s / rt.DT) for s in (60, 200, 60))
    T = T1 + T2 + T3
    sig = np.ones((T, rt.N_STREAMS))
    exc = np.zeros(T, bool)
    exc[T1:T1 + T2] = True
    for ver, reset, floor in (("v1.0", 0.0, 0.0), ("v1.1", RESET, 1e-6)):
        sh = run_ema_v11(sig, TAU, False, np.random.default_rng(12), reset=reset, excised=exc, floor=floor)
        post = sh[T1 + T2:] / sig[T1 + T2:]
        guarded = (post < 1.0 / rt.G).any(axis=1)                # rows of these blocks guarded
        blocks = int(guarded.sum())
        res["scenarios"][f"excision200s|{ver}"] = {"release_blocks_guarded": blocks,
                                                   "release_s_guarded": blocks * rt.DT,
                                                   "k_first_release_block": float(post[0].mean())}
        print(f"excision 200 s {ver}: {blocks} release blocks guarded ({blocks * rt.DT:.1f} s), "
              f"k at release {post[0].mean():.3g}", flush=True)
    res["worst_loss_vs_static"] = {f"b{b}|{v}": w for (b, v), w in worst.items()}
    res["pass_0.04pct"] = all(w <= 4e-4 for w in worst.values())
    print("worst loss vs static (excl. EMA-from-block-0 start-up):",
          {f"b{b} {v}": f"{100 * w:.4f}%" for (b, v), w in worst.items()}, "PASS" if res["pass_0.04pct"] else "FAIL")
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, "rowrule_track_v11.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
