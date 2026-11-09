"""Figures for the 2x development report (docs/x2/report/): sensitivity vs DM for the legacy-style single-stage
search, today's 1x dsart and the deployed 2x, from the snr2x loss-chain model; the 2x/1x gain vs width with the
measured injection K ratios (campaign) overlaid when available.

    python make_figs.py [--campaign x2_campaign.jsonl] [--k1x k_snapshot_1x.json]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SNR2X = os.path.join(HERE, "..", "..", "timeres_study", "snr2x", "results", "snr2x.npz")
OUT = os.path.join(HERE, "figs")
#: reference palette, fixed order (validated: dataviz validate_palette.py, light surface)
C2X, C1X, CLEG = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
#: the deployed format's quantisation factor on top of the dedispersion model (ledger S1: gridded ns4 cint4)
Q_2X = 0.9945
SERIES = (("prop_ns4", "2x dsart (deployed)", C2X, "-", Q_2X),
          ("current", "1x dsart (until 11-08)", C1X, "--", 1.0),
          ("single", "legacy-style single-stage", CLEG, ":", 1.0))


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _smooth(x, y, win=31):
    o = np.argsort(x)
    x, y = x[o], y[o]
    k = win // 2
    ys = np.array([np.median(y[max(0, i - k):i + k + 1]) for i in range(len(y))])
    return x, ys


def fig_sensitivity(z):
    W = list(z["widths_ms"])
    widths = (0.2, 0.5, 1.0, 4.0)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True, facecolor=SURF)
    for ax, w in zip(axes.flat, widths):
        _style(ax)
        j = W.index(w)
        for key, label, col, ls, q in SERIES:
            dm, ch = z[f"{key}_dm"], z[f"{key}_chain"][:, 3, j] * q
            x, y = _smooth(dm, ch)
            ax.plot(x, y, color=col, ls=ls, lw=2.0, label=label)
        ax.axvline(1600, color=MUTED, lw=0.8)
        ax.text(1640, 0.98, "2x tier B: 1.05 ms", fontsize=7, color=INK2, va="top")
        ax.set_title(f"burst width {w:g} ms", fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 3050)
    for ax in axes[1]:
        ax.set_xlabel("DM (pc cm$^{-3}$)", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("recovered fraction of ideal S/N", fontsize=9, color=INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, "sensitivity_vs_dm.png")
    fig.savefig(p, dpi=200, facecolor=SURF)
    plt.close(fig)
    return p


def band_mean(z, key, lo, hi):
    dm, ch = z[f"{key}_dm"], z[f"{key}_chain"][:, 3, :]
    m = (dm >= lo) & (dm < hi)
    return ch[m].mean(axis=0)


def bucket(dm: float, width: int) -> str:
    """The injector's K key: ``dm0250`` (width 4) or ``dm0250@w0032``."""
    b = f"dm{int(round(dm / 50.0)) * 50:04d}"
    return b if int(width) == 4 else f"{b}@w{int(width):04d}"


def load_k(campaign: str | None, k1x: str | None):
    """(dm, width_native, K_2x, K_1x) for every bucket measured in both."""
    out = []
    if not campaign or not os.path.exists(campaign) or not k1x or not os.path.exists(k1x):
        return out
    old = json.load(open(k1x))
    old = {k.rsplit("/", 1)[-1]: v.get("K") for k, v in old.items() if isinstance(v, dict) and v.get("K")}
    new = {}
    for line in open(campaign):
        d = json.loads(line)
        if d.get("phase") == "calibrate" and d.get("ok") and d.get("K"):
            new[bucket(float(d["dm"]), int(d["width"]))] = (float(d["dm"]), int(d["width"]),
                                                            float(d["K"]))
    for b, (dm, w, k2) in sorted(new.items()):
        if b in old:
            out.append((dm, w, k2, float(old[b])))
    return out


def fig_gain(z, kpairs):
    W = z["widths_ms"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURF)
    _style(ax)
    bands = ((100, 500, "-"), (500, 1000, "--"), (1000, 1500, ":"))
    for lo, hi, ls in bands:
        g = band_mean(z, "prop_ns4", lo, hi) * Q_2X / band_mean(z, "current", lo, hi)
        ax.plot(W, g, color=C2X, ls=ls, lw=2.0, label=f"model, DM {lo}-{hi}")
    if kpairs:
        dm = np.array([p[0] for p in kpairs])
        w_ms = np.array([p[1] for p in kpairs]) * 32.768e-3
        r = np.array([p[2] / p[3] for p in kpairs])
        ax.scatter(w_ms, r, s=26, color=C1X, edgecolor=SURF, linewidth=1.0, zorder=3,
                   label="measured: injection K(2x) / K(1x), per (DM, width) bucket")
        for x, y, d in zip(w_ms, r, dm):
            ax.annotate(f"{d:.0f}", (x, y), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.axhline(1.0, color=AXIS, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("burst width (ms)", fontsize=9, color=INK2)
    ax.set_ylabel("S/N gain, 2x over 1x", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    p = os.path.join(OUT, "gain_vs_width.png")
    fig.savefig(p, dpi=200, facecolor=SURF)
    plt.close(fig)
    return p


_SMOOTHED: dict = {}


def model_eta(z, key, dm, w_ms, q=1.0):
    """Model recovered fraction at (dm, width) (running median over DM, log-width interpolation)."""
    W = np.log(np.asarray(z["widths_ms"], float))
    if key not in _SMOOTHED:
        _SMOOTHED[key] = [_smooth(z[f"{key}_dm"], z[f"{key}_chain"][:, 3, j]) for j in range(len(W))]
    x = np.log(np.clip(w_ms, np.exp(W[0]), np.exp(W[-1])))
    col = [np.interp(dm, *xy) for xy in _SMOOTHED[key]]
    return float(np.interp(x, W, col)) * q


def load_k2x(campaign: str | None):
    """[(dm, width_native, K, observed_snr)] of every ok calibrate probe (last one per bucket)."""
    if not campaign or not os.path.exists(campaign):
        return []
    got = {}
    for line in open(campaign):
        d = json.loads(line)
        if d.get("phase") == "calibrate" and d.get("ok") and d.get("K"):
            got[(float(d["dm"]), int(d["width"]))] = float(d["K"]), float(d.get("observed_snr", 0))
    return [(dm, w, k, s) for (dm, w), (k, s) in sorted(got.items())]


def fig_k_shape(z, k2x):
    """K = S/N sqrt(w) / F is proportional to the recovered fraction of ideal S/N, so the measured K per
    (DM, width), scaled by ONE constant K0, should follow the model curves. K0 = median(K / model)."""
    if not k2x:
        return None, None
    widths = sorted({w for _dm, w, _k, _s in k2x})
    eta = {(dm, w): model_eta(z, "prop_ns4", dm, w * 32.768e-3, Q_2X) for dm, w, _k, _s in k2x}
    k0 = float(np.median([k / eta[(dm, w)] for dm, w, k, _s in k2x]))
    ncol = 4
    nrow = -(-len(widths) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.2, 2.7 * nrow + 0.6), sharex=True, sharey=True,
                             facecolor=SURF, squeeze=False)
    dmg = np.linspace(50, 3000, 300)
    resid = []
    for ax, w in zip(axes.flat, widths):
        _style(ax)
        w_ms = w * 32.768e-3
        ax.plot(dmg, [model_eta(z, "prop_ns4", d, w_ms, Q_2X) for d in dmg], color=C2X, lw=2.0,
                label="model (2x, deployed)")
        pts = [(dm, k / k0) for dm, ww, k, _s in k2x if ww == w]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=30, color=C1X, edgecolor=SURF,
                   linewidth=1.0, zorder=3, label="measured K / K0")
        resid += [p[1] / eta[(p[0], w)] for p in pts]
        ax.axvline(1600, color=MUTED, lw=0.8)
        ax.set_title(f"width {w_ms:.2g} ms ({w} native)", fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 1.3)
    for ax in axes.flat[len(widths):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("DM (pc cm$^{-3}$)", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("recovered fraction", fontsize=9, color=INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "k_vs_model.png")
    fig.savefig(p, dpi=200, facecolor=SURF)
    plt.close(fig)
    r = np.asarray(resid)
    stats = dict(k0=k0, n=int(r.size), median=float(np.median(r)),
                 p16=float(np.percentile(r, 16)), p84=float(np.percentile(r, 84)),
                 within_20pct=float(np.mean(np.abs(r - 1) <= 0.2)),
                 outliers=[(dm, w, round(k / k0 / eta[(dm, w)], 2)) for dm, w, k, _s in k2x
                           if abs(k / k0 / eta[(dm, w)] - 1) > 0.3])
    return p, stats


def k1x_fit(z, k1x: str | None, k0_2x: float | None):
    """The 1x K buckets against the 1x ('current') model with their own K0: the scatter, and
    K0(2x) / K0(1x) -- the S/N per unit fluence the model does NOT explain (epoch-mixed: the 1x K
    were measured on several days)."""
    if not k1x or not os.path.exists(k1x) or not k0_2x:
        return None
    rows = []
    for k, v in json.load(open(k1x)).items():
        if not (isinstance(v, dict) and v.get("K") and k.startswith("/cnf/inject/snr_calibration/")):
            continue
        b = k.rsplit("/", 1)[-1]
        dm = float(b[2:6])
        w = int(b.split("@w")[1]) if "@w" in b else 4
        rows.append((dm, w, float(v["K"]), model_eta(z, "current", dm, w * 32.768e-3)))
    if not rows:
        return None
    r = np.array([k / e for _dm, _w, k, e in rows])
    k0 = float(np.median(r))
    rel = r / k0
    return dict(n=len(rows), k0_1x=k0, k0_ratio_2x_over_1x=float(k0_2x / k0),
                p16=float(np.percentile(rel, 16)), p84=float(np.percentile(rel, 84)),
                within_20pct=float(np.mean(np.abs(rel - 1) <= 0.2)))


def summary_table(z):
    """Mean recovered fraction per DM band at 0.5 / 1 / 4 ms for the three searches."""
    W = list(z["widths_ms"])
    rows = []
    for lo, hi in ((50, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)):
        row = {"band": f"{lo}-{hi}"}
        for key, label, _c, _l, q in SERIES:
            dm = z[f"{key}_dm"]
            if not ((dm >= lo) & (dm < hi)).any():
                row[key] = None
                continue
            m = band_mean(z, key, lo, hi) * q
            row[key] = {f"{w:g}": float(m[W.index(w)]) for w in (0.2, 0.5, 1.0, 4.0)}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--k1x", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(SNR2X, allow_pickle=True)
    kp = load_k(a.campaign, a.k1x)
    k_shape, k_stats = fig_k_shape(z, load_k2x(a.campaign))
    out = {"sensitivity": fig_sensitivity(z), "gain": fig_gain(z, kp), "table": summary_table(z),
           "k_pairs": kp, "k_shape": k_shape, "k_shape_stats": k_stats,
           "k1x_fit": k1x_fit(z, a.k1x, (k_stats or {}).get("k0"))}
    json.dump(out, open(os.path.join(OUT, "figs.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
