"""Figures for the current-vs-2x S/N note -> ../figs_snr2x/."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "../figs_snr2x/"
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8c86"
CUR, PRO, ONE = "#eb6834", "#2a78d6", "#1baf7a"
SCH = (("current", CUR, "current (deployed)"), ("prop_q026", PRO, "proposed 2x"),
       ("single", ONE, "single-shot 0.26 ms / 244 kHz"))
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 1.6, "legend.fontsize": 8.0})
Z = np.load("results/snr2x.npz")
W = Z["widths_ms"]


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def series(key, stage=3):
    dm = Z[f"{key}_dm"]
    o = np.argsort(dm)
    return dm[o], Z[f"{key}_chain"][o][:, stage]


def band(key, lo, hi, stage=3):
    dm, v = series(key, stage)
    m = (dm >= lo) & (dm <= hi)
    return v[m]


# ---------------------------------------------------------------- 1: S/N vs DM at three widths
def f_dm():
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.4), sharex=True)
    for ax, w in zip(axes, (0.2, 1.0, 4.0)):
        j = int(np.argmin(np.abs(W - w)))
        for key, col, lab in SCH:
            dm, v = series(key)
            ax.plot(dm, v[:, j], color=col, lw=0.9, label=lab)
        ax.axvspan(103, 1496, color=CUR, alpha=0.06, lw=0)
        ax.axvline(1600, color=GREY, lw=0.8, ls=(0, (3, 3)))
        ax.set_ylim(0, 1)
        ax.set_ylabel("S/N / ideal")
        ax.text(0.995, 0.06, f"W = {w:g} ms (Gaussian FWHM)", transform=ax.transAxes, ha="right",
                fontsize=8.4, color=INK2)
    axes[0].legend(loc="upper left", ncol=3, fontsize=7.4, bbox_to_anchor=(0.0, 1.04))
    axes[0].text(1625, 0.62, "2x sample: 0.52 ms | 1.05 ms", fontsize=7.4, color=INK2)
    axes[-1].set_xlim(0, 3000)
    axes[-1].set_xlabel("true DM (pc cm$^{-3}$)")
    axes[0].set_title("Recovered S/N vs DM (shaded: current search range 103-1496)")
    save(fig, "s1_vs_dm.png")


# ---------------------------------------------------------------- 2: S/N vs width + gain
def f_width():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for key, col, lab in (("current", CUR, "current"), ("prop_q026", PRO, "proposed 2x"),
                          ("single", ONE, "single-shot")):
        v = band(key, 100, 1500)
        a.fill_between(W, np.percentile(v, 10, axis=0), v.max(0), color=col, alpha=0.15, lw=0)
        a.plot(W, v.mean(0), "-o", color=col, ms=3.5, label=lab)
    v = band("prop_q026", 1500, 3000)
    a.plot(W, v.mean(0), "--o", color=PRO, ms=3.0, lw=1.1, label="proposed 2x, DM 1500-3000")
    a.set_xscale("log")
    a.set_ylim(0, 1)
    a.set_xlabel("pulse width W (ms)")
    a.set_ylabel("S/N / ideal, DM 100-1500")
    a.legend(loc="lower right", fontsize=7.2)
    a.set_title("Recovered S/N (mean; band 10th pct-max)", fontsize=9.6)
    shades = {"prop_q026": ("#b7d3f6", "#2a78d6", "#0d366b"), "single": ("#a8e3c8", "#1baf7a", "#0b5c3e")}
    for key, nm in (("prop_q026", "2x"), ("single", "single-shot")):
        for (lo, hi), c in zip(((100, 500), (500, 1000), (1000, 1500)), shades[key]):
            r = band(key, lo, hi).mean(0) / band("current", lo, hi).mean(0)
            b.plot(W, r, "-o", color=c, ms=3.0, lw=1.3, label=f"{nm}, DM {lo}-{hi}")
    r = band("prop_q052", 100, 1500).mean(0) / band("current", 100, 1500).mean(0)
    b.plot(W, r, ":", color=INK2, lw=1.2, label="2x, stage-1 0.52 ms, DM 100-1500")
    b.axhline(1, color=GREY, lw=0.8)
    b.set_xscale("log")
    b.set_ylim(0.9, 2.15)
    b.set_xlabel("pulse width W (ms)")
    b.set_ylabel("proposed / current")
    b.legend(loc="upper right", fontsize=6.4)
    b.set_title("S/N gain over the current scheme", fontsize=9.6)
    fig.tight_layout()
    save(fig, "s2_vs_width.png")


# ---------------------------------------------------------------- 3: loss cascade
def f_cascade():
    stages = ["ideal", "+ channel\nsmear", "+ sampling\n& boxcars", "+ DM trial\nspacing", "+ integer\ndelays"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(len(stages))
    ends = []
    for w, ls in ((0.2, "-"), (1.0, "--")):
        j = int(np.argmin(np.abs(W - w)))
        for key, col, lab in (("current", CUR, "current"), ("prop_q026", PRO, "2x"), ("single", ONE, "single-shot")):
            c = Z[f"{key}_chain"]
            dm = Z[f"{key}_dm"]
            m = (dm >= 500) & (dm <= 1000)
            y = np.r_[1.0, c[m][:, :, j].mean(0)]
            ax.plot(x, y, ls, marker="o", color=col, ms=4, label=f"{lab}, W = {w:g} ms")
            ends.append([y[-1], col])
    last = -1.0
    for v, col in sorted(ends):                         # keep end labels >= 0.045 apart
        last = max(v, last + 0.045)
        ax.text(x[-1] + 0.08, last, f"{v:.2f}", va="center", fontsize=7.6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7.8)
    ax.set_xlim(-0.2, len(stages) - 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("S/N / ideal (DM 500-1000)")
    ax.legend(fontsize=7.0, ncol=2, loc="lower left")
    ax.set_title("Where the S/N goes, step by step")
    save(fig, "s3_cascade.png")


if __name__ == "__main__":
    f_dm(); f_width(); f_cascade()
    print("ok")
