#!/usr/bin/env python3
"""Synthetic calibration of the DM-time ridge features against TRUE bandwidth (R12 study, P0).

Plants a dispersed burst with a known spectrum in white noise at 0.262 ms x
61 kHz (the 2x stage-1 input), recreates the 2x DM-time plane with the plan's
own tables (``dmt_truth.emulate_2x``) and records the ridge features next to
the truth: spectrum family, FWHM bandwidth, scintle bandwidth, width, S/N, DM.

Families: ``flat``; ``gauss`` (band-limited, FWHM ``bw``); ``scint`` (flat or
band-limited envelope x exponential scintles of decorrelation ``nud``);
``train`` (narrow-band pulse train, 3-15 ms spacing). Each case draws its
centre frequency and noise from ``--seed`` + case index (reproducible).

    python tools/x2/dmt_synth.py --plan PLAN.npz --out synth.jsonl --n 400 --worker 0 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dmt_truth import NG, NK, Q1_MS, _nu_summed, emulate_2x, save_plane  # noqa: E402

from dsart.coinc.dmt_ridge import (K_DM_MS_GHZ2, RidgeConfig, boxcar_rows,  # noqa: E402
                                   ridge_features, robust_rows)
from dsart.x2.plan import TieredDmPlan  # noqa: E402

CH_MHZ = 0.06103515625
NU = _nu_summed() * 1e3                                                # (16, 192) MHz


def spectrum(rng, fam: str, bw: float, nud: float, nu_c: float) -> np.ndarray:
    nu = NU.reshape(-1)
    if fam == "flat":
        a = np.ones_like(nu)
    else:
        a = np.exp(-0.5 * ((nu - nu_c) / (bw / 2.3548)) ** 2) if bw > 0 else np.ones_like(nu)
    if nud > 0:
        # exponential scintles: |complex Gaussian field smoothed over nud|^2
        n = nu.size
        f = rng.normal(size=n) + 1j * rng.normal(size=n)
        k = np.exp(-0.5 * (np.arange(-4 * n // 2, 4 * n // 2) * CH_MHZ / (nud / 2.0)) ** 2)
        k = k[np.abs(np.arange(-4 * n // 2, 4 * n // 2)) * CH_MHZ < 4 * nud]
        sm = np.convolve(f, k, "same")
        s = np.abs(sm) ** 2
        a = a * s / s.mean()
    return a.reshape(NG, NK)


def occupied_true(a: np.ndarray) -> float:
    p = a.reshape(-1)
    return float(p.sum() ** 2 / (p ** 2).sum() * CH_MHZ)


def make_case(i: int, seed: int, rng_master) -> dict:
    rng = np.random.default_rng(seed * 100003 + i)
    fam = rng.choice(["flat", "gauss", "gauss", "scint", "scint", "train"], p=[.1, .25, .2, .2, .1, .15])
    if fam == "train":
        bw = float(rng.choice([0.3, 0.6, 1.0, 2.0, 4.0]))
    elif fam == "flat":
        bw = 0.0
    else:
        bw = float(np.exp(rng.uniform(np.log(1.0), np.log(120.0))))
    nud = float(rng.choice([0.5, 1.0, 2.0, 5.0])) if fam == "scint" else 0.0
    if fam == "scint" and rng.random() < 0.4:
        bw = 0.0                                                         # flat envelope, scintillated
    return dict(case=i, family=str(fam), bw_mhz=round(bw, 3), nud_mhz=nud,
                width=int(rng.choice([1, 2, 2, 4, 8, 16, 32])), snr=float(rng.choice([10, 13, 20, 40, 150])),
                dm=float(rng.choice([150, 400, 800, 1500, 2500])),
                nu_c=float(rng.uniform(1320, 1490)), n_pulses=int(rng.integers(2, 6)) if fam == "train" else 1,
                spacing_ext=int(rng.integers(6, 30)), seed=int(seed * 100003 + i))


def run_case(c: dict, plan: TieredDmPlan, cfg: RidgeConfig, planes_dir=None) -> dict:
    rng = np.random.default_rng(c["seed"])
    a = spectrum(rng, "flat" if c["family"] == "flat" else "gauss", c["bw_mhz"], c["nud_mhz"], c["nu_c"])
    if c["family"] == "flat" and c["nud_mhz"] > 0:
        a = spectrum(rng, "flat", 0.0, c["nud_mhz"], c["nu_c"])
    dm = c["dm"]
    w_ext = c["width"]
    nu = NU / 1e3
    sweep = K_DM_MS_GHZ2 * (nu ** -2 - nu[0, 0] ** -2)                    # ms per pc/cc, (16,192)
    s_max_q1 = K_DM_MS_GHZ2 * (dm + 700) * (nu[-1, -1] ** -2 - nu[0, 0] ** -2) / Q1_MS
    t_bot_ext = 256 + int(s_max_q1 / 2) + 64
    n_q1 = 2 * (t_bot_ext + 256) + 200
    x = rng.normal(size=(n_q1, NG, NK)).astype(np.float32)
    # matched S/N of the band sum (unit noise per channel-sample): A * n_on * sum(a) / sqrt(C n_on)
    n_on = 2 * w_ext
    amp = c["snr"] * np.sqrt(a.size * n_on) / (n_on * a.sum())
    t_bot_q1 = 2 * t_bot_ext
    d_bot = K_DM_MS_GHZ2 * dm * (nu[-1, -1] ** -2 - nu[0, 0] ** -2) / Q1_MS
    starts = np.rint(t_bot_q1 - d_bot + sweep * dm / Q1_MS).astype(np.int64)  # per channel, top-referenced
    for p in range(c["n_pulses"]):
        off = 2 * p * c["spacing_ext"]
        for dt in range(n_on):
            x[starts + off + dt, np.arange(NG)[:, None], np.arange(NK)[None, :]] += amp * a
    lo, hi = t_bot_ext - 256, t_bot_ext + 256
    t0 = time.time()
    plane = emulate_2x(x, plan, dm_range=(dm - 700, dm + 700), t_window=(lo, hi))
    z = robust_rows(boxcar_rows(plane, w_ext))
    i_trig = int(np.argmin(np.abs(plan.fine_dm - dm)))
    row = z[i_trig, 256 - 6:256 + 6]
    t_trig = 256 - 6 + int(np.nanargmax(row)) if np.isfinite(row).any() else 256
    f, _ = ridge_features(z, plan.fine_dm, dm_trig=float(plan.fine_dm[i_trig]), t_trig=t_trig, width=w_ext,
                          cfg=cfg)
    save_plane(planes_dir, f"synth{c['case']:04d}", z, plan.fine_dm, t_trig=t_trig, width=w_ext,
               dm_trig=float(plan.fine_dm[i_trig]), occ_true_mhz=occupied_true(a),
               **{k: v for k, v in c.items() if k != "width"})
    return {**c, "occ_true_mhz": round(occupied_true(a), 2), "emu_sec": round(time.time() - t0, 1),
            **{f"r_{k}": v for k, v in f.to_dict().items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--planes", default="", help="directory for the normalised planes (re-scoring)")
    a = ap.parse_args(argv)
    plan = TieredDmPlan.from_npz(a.plan)
    cfg = RidgeConfig()
    with open(a.out, "a") as fh:
        for i in range(a.worker, a.n, a.workers):
            c = make_case(i, a.seed, None)
            try:
                rec = run_case(c, plan, cfg, planes_dir=a.planes)
            except Exception as exc:  # noqa: BLE001
                rec = {**c, "error": repr(exc)[:200]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
