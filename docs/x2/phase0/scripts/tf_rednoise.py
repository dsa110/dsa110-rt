"""Red-noise check: the live cube noise is not white in time (sigma_k grows faster than sqrt(w): 1.163 x
at b64, a slowly varying image residual, pipeline-sensitivity audit 2026-09-20).  A smoother, longer
filter integrates that residual differently from a boxcar, so the white-noise gain could shrink.

Input (read-only): _inspect/sensitivity/out_cubenoise/deep/deep_*.json `sigma_k_pooled` = the Layer-2
sigma_k of the b1..b64 boxcars on 32 live cubes at 1.048576 ms (today).  Fit rho(k) = (1 - c) delta_k +
c exp(-k / tau) (white + one slow component, lag in 1x samples) to sigma_k(w) / sqrt(w); map to the 2x
tier-A sample (lag x 2); the residual's share of the per-sample variance is (a) kept (conservative) or
(b) halved (the white part per sample doubles at half the integration, the slow residual does not).
Report per bank filter: noise inflation sigma_k(real) / ||h||_2 and the S/N gain vs the boxcar of the
same nominal width after dividing by the inflation ratio.
-> results/time_filters_rednoise.json
"""
import glob
import json
import os

import numpy as np

from tf_common import RES, M

os.nice(10)
files = sorted(glob.glob("/home/ubuntu/vikram/dev/_inspect/sensitivity/out_cubenoise/deep/deep_*.json"))
sk = np.array([rec["sigma_k_pooled"] for f in files for rec in json.load(open(f))
               if "sigma_k_pooled" in rec], float)                                # (n_cube, 7)
W1 = np.array([1, 2, 4, 8, 16, 32, 64])
ratio = (sk / sk[:, :1]).mean(axis=0) / np.sqrt(W1)                               # sigma_k(w)/(sqrt(w) sigma_1)


def var_ratio(h, rho):
    """Var(sum h x) / Var_white for unit-variance x with ACF rho (rho[0] = 1)."""
    L = len(h)
    lag = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    return float(h @ (rho[lag]) @ h / np.sum(h ** 2))


def rho_model(c, tau, n, lag_scale=1.0):
    k = np.arange(n) / lag_scale
    r = c * np.exp(-k / tau)
    r[0] = 1.0
    return r


best = None
for c in np.linspace(0.0, 0.2, 201):
    for tau in np.geomspace(1, 2000, 120):
        rho = rho_model(c, tau, 70)
        pred = np.array([np.sqrt(var_ratio(np.ones(w), rho)) for w in W1])
        err = np.sum((pred - ratio) ** 2)
        if best is None or err < best[0]:
            best = (err, c, tau, pred)
_, c1, tau1, pred = best
print(f"live sigma_k/sqrt(w) (1x): {np.round(ratio, 4)}  fit c={c1:.4f} tau={tau1:.1f} samples -> "
      f"{np.round(pred, 4)}  ({len(files)} files)")
out = {"live_ratio_1x": ratio.tolist(), "fit": {"c": c1, "tau_1x_samples": tau1, "pred": pred.tolist()},
       "cases": {}}
for case, c2 in (("residual share kept", c1), ("residual share halved", c1 / 2)):
    rho = rho_model(c2, tau1 * 2, 200)            # 2x samples: lag in 0.524 ms = half a 1x sample
    infl = {bk: [np.sqrt(var_ratio(f["h"], rho)) for f in M.BANKS[bk]] for bk in ("boxcar", "b3", "b3x", "gauss")}
    out["cases"][case] = {bk: [round(x, 4) for x in v] for bk, v in infl.items()}
    print(f"{case}: inflation sigma_k/||h|| per filter")
    for bk, v in infl.items():
        print(f"  {bk:6s} {np.round(v, 4)}")
    # same-nominal-width comparison B3 vs boxcar: extra noise factor the B3 pays relative to the boxcar
    rel = np.array(infl["b3"]) / np.array(infl["boxcar"])
    out["cases"][case]["b3_over_boxcar_inflation"] = rel.round(4).tolist()
    print(f"  B3 / boxcar inflation at w=1..64: {np.round(rel, 4)}")
    # realised b3x gain: divide each (DM, width) ratio by the B3/boxcar inflation at the bank width
    # nearest the pulse width in that DM's tier sample (boxcars at w <= 2, so no penalty there)
    z = np.load(os.path.join(RES, "time_filters_main.npz"))
    zx = np.load(os.path.join(RES, "time_filters_main_x.npz"))
    dm, Wms = z["dm"], z["widths_ms"]
    ib = list(z["banks"]).index("boxcar")
    ix = list(zx["banks"]).index("b3x")
    T = np.where(dm < 1600, 0.524288, 1.048576)
    wstar = np.clip(2.0 ** np.rint(np.log2(np.maximum(1.0, Wms[None, :] / T[:, None]))), 1, 64)
    pen = np.interp(np.log2(wstar), np.log2(W1), rel)
    pen[wstar <= 2] = 1.0
    sel = (Wms >= 0.1) & (Wms <= 16)
    i1 = int(np.argmin(np.abs(Wms - 1.0)))
    for s, sh in enumerate([str(x) for x in z["shapes"]]):
        b, x = z["snr"][:, s, ib], zx["snr"][:, s, ix]
        white = x[:, sel].mean() / b[:, sel].mean()
        real = (x / pen)[:, sel].mean() / b[:, sel].mean()
        r1 = (x / pen)[:, i1].mean() / b[:, i1].mean()
        per_w = ((x / pen).mean(axis=0) / b.mean(axis=0)).round(4).tolist()
        out["cases"][case][f"b3x_gain_{sh}"] = dict(white=float(white), with_red_noise=float(real),
                                                    at_1ms=float(r1), per_width=per_w)
        print(f"  b3x vs boxcar, {sh}: white x{white:.4f} -> with red noise x{real:.4f}; 1 ms x{r1:.4f}; "
              f"per width {per_w}")
json.dump(out, open(os.path.join(RES, "time_filters_rednoise.json"), "w"), indent=1)
