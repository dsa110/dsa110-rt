"""Figures for report v3 (baseline = NICs stay in x4 slots) -> ../figs_v3/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = "../figs_v3/"
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8c86"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0, "legend.fontsize": 8.0})


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight")
    plt.close(fig)


P = json.load(open("results/plan3.json"))
FEAS = [r for r in P["feas"] if r["scenario"].startswith("baseline (corr +2") and r["fov"] == "FoV/2"]
G = P["gains"]


# ------------------------------------------------------------------ N1 per-node network budget
def n1():
    ts = {r["ambition"]: r for r in FEAS if r["scheme"] == "two-stage"}
    rows = [("today", 1.07, 4.3),
            ("1x two-stage", ts["1x"]["corr_extra_in"], ts["1x"]["search_in"]),
            ("2x two-stage", ts["2x"]["corr_extra_in"], ts["2x"]["search_in"]),
            ("4x two-stage", ts["4x"]["corr_extra_in"], ts["4x"]["search_in"]),
            ("two corner turns: CT1 alone, 0.52 ms / 122 kHz", 2.0, 4.1),
            ("two corner turns: CT1 alone, 0.26 ms / 61 kHz", 8.2, 16.4)]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
    y = np.arange(len(rows))[::-1]
    # corr node
    ax = axs[0]
    for yy, (lab, c, s) in zip(y, rows):
        ax.barh(yy, 18.0, color=GREY, height=0.6)
        ax.barh(yy, c, left=18.2, color=SLOT[0] if "corner" not in lab else SLOT[3], height=0.6)
        ax.text(18.4 + c, yy, f"+{c:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvspan(20, 25, color="#fbe3e1", zorder=0)
    ax.axvline(50, color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.text(20.3, y[0] + 0.55, "x4 limit:\nuncertain 20-25", fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(0, 32)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.3)
    ax.set_title("corr node (incl. ~18 Gb/s SNAP)")
    ax.set_xlabel("ingress (Gb/s)")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    ax.grid(axis="y", visible=False)
    # search node
    ax = axs[1]
    for yy, (lab, c, s) in zip(y, rows):
        ax.barh(yy, s, color=SLOT[0] if "corner" not in lab else SLOT[3], height=0.6)
        ax.text(s + 0.3, yy, f"{s:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(17, color=SLOT[7], lw=1.3)
    ax.axvline(22, color=SLOT[7], lw=0.9, ls=(0, (4, 3)))
    ax.text(17.3, y[0] + 0.55, "measured knee 17\n(22 with RX tuning?)", fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(0, 32)
    ax.set_title("search node")
    ax.set_xlabel("ingress (Gb/s)")
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=GREY, label="SNAP voltages"),
                        Patch(color=SLOT[0], label="two-stage corner turn (owner share)"),
                        Patch(color=SLOT[3], label="first hop of two corner turns (every node)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.4)
    fig.suptitle("Baseline: corner-turn data must flow corr -> search; corr nodes have ~2-4 Gb/s to spare",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, "n1_network_budget.png")


# ------------------------------------------------------------------ N2 rate gains baseline vs stretch
def n2():
    order = [("base-ts|1x", "1x"), ("base-ts|2x", "2x"), ("base-ts|4x-lite", "4x\n<DM 500"),
             ("base-ts|4x", "4x\n<DM 800"), ("str-tc|4x", "stretch\n4x, 2 CTs")]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, (pop, title) in zip(axs, (("m1", "width median 1 ms"), ("m03", "width median 0.3 ms"))):
        x = np.arange(len(order))
        v0 = [G[k]["gain"][pop] for k, _ in order]
        v1 = [G[k]["gain_zoom8"][pop] for k, _ in order]
        cols = [SEQ[2]] * 4 + [SLOT[6]]
        ax.bar(x - 0.19, v0, 0.36, color=cols)
        ax.bar(x + 0.19, v1, 0.36, color=cols, alpha=0.55, hatch="////", edgecolor=SURF, lw=0)
        for xi, a, b in zip(x, v0, v1):
            ax.text(xi - 0.19, a + 0.03, f"{a:.2f}", ha="center", fontsize=6.6, color=INK2)
            ax.text(xi + 0.19, b + 0.03, f"{b:.2f}", ha="center", fontsize=6.6, color=INK2)
        ax.axhline(1, color=GREY, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([o for _, o in order], fontsize=7.4)
        ax.set_title(title)
        ax.set_ylim(0, 3.0)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("detection-rate gain vs today")
    fig.legend(handles=[Patch(color=SEQ[2], label="baseline, search alone"),
                        Patch(facecolor=SEQ[2], alpha=0.55, hatch="////", edgecolor=SURF,
                              label="+ voltage zoom, threshold 9.55 -> 8"),
                        Patch(color=SLOT[6], label="stretch (NIC in x16 slot)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=7.4)
    fig.suptitle("Rate gain (Euclidean, DM median 500, DM <= 1500)", fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, "n2_rates.png")


# ------------------------------------------------------------------ N3 S/N vs width (baseline configs)
def n3():
    B = dict(np.load("results/broad_grid.npz"))
    B.update({k: v for k, v in np.load("results/baseline_grid.npz").items() if k not in ("dms", "widths_us")})
    W = B["widths_us"] / 1e3
    DMS = list(B["dms"])
    sel = [("today (1.05 ms, 244 kHz, 8x4)", "today", SLOT[1]),
           ("1x two-stage done right", "ts1_c2", SLOT[7]),
           ("2x (0.52 ms), streams at T", "bts2", SLOT[6]),
           ("4x (0.26 ms), streams at T", "bts4", SLOT[0]),
           ("voltage zoom (coherent, 32.8 us)", "zoom", SLOT[5])]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, dm in zip(axs, (500.0, 1200.0)):
        i = DMS.index(dm)
        for lab, k, c in sel:
            ax.plot(W, B[k][i], color=c, marker="o", ms=3, lw=2.4 if k == "bts4" else 1.8, label=lab)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(f"DM {int(dm)}")
        ax.set_xlabel("intrinsic burst FWHM (ms)")
    axs[0].set_ylabel("S/N recovered / ideal matched filter")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
    fig.suptitle("Baseline configurations: recovered S/N", fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, "n3_snr.png")


if __name__ == "__main__":
    n1(); n2(); n3()
    print("ok")
