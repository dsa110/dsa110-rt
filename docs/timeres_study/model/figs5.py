"""Report-v4 figures that need the measured fabric (b5b/b5c) and plan6 -> ../figs_v4/.
Usage: python figs5.py <search ingress Gb/s> [n_search]
(figs4.py m2/m4/m5 are re-used; run `python figs4.py` first or let this call them.)"""
import json
import sys

import numpy as np
from matplotlib.patches import Patch

import figs4
from figs4 import GREY, INK2, SEQ, SLOT, SURF, plt, save
from plan6 import AMB, fabric, fits, items, us_image

H_S = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
N_S = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FAB = "../bench_results/fabric/"


# ---------------------------------------------------------------- F1 fabric: delivered vs offered
def f_fabric():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    groups = [("b5b_corr_nic", ("n04", "n11", "n22"), SLOT[1], "corr node (SNAP + corner turn)"),
              ("b5c_search_knee", ("n02", "n09", "n13"), SLOT[0], "search node (corner turn only)")]
    for sub, nodes, c, lab in groups:
        try:
            S = json.load(open(FAB + sub + "/b5b_summary.json"))
        except OSError:
            continue
        first = True
        for n in nodes:
            if n not in S:
                continue
            pts = sorted(((v["gbps_phy"], v["gbps_phy"] * (1 - v["discard_frac"])) for v in S[n].values()))
            x, y = np.array(pts).T
            ax.plot(x, y, "-o", color=c, ms=4.5, lw=1.4, label=lab if first else None)
            first = False
    ax.plot([0, 27], [0, 27], color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.text(21.5, 23.2, "no loss", fontsize=7.4, color=INK2, rotation=27)
    ax.axhspan(16.6, 17.3, color=SLOT[3], alpha=0.25, lw=0)
    ax.text(0.6, 17.6, "x4-slot ceiling ~17 Gb/s delivered", fontsize=7.4, color=INK2)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)
    ax.set_xlabel("offered at the NIC (rx_bytes_phy, Gb/s)")
    ax.set_ylabel("delivered to host (Gb/s)")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.set_title("Measured: each ConnectX-4 delivers ~17 Gb/s; SNAP alone fills a corr node's")
    save(fig, "f1_fabric.png")


# ---------------------------------------------------------------- F2 redundancy: entries per stream
def f_redundancy():
    R = json.load(open("results/redundancy.json"))
    rows = [("core baselines", R["n_baselines"], GREY),
            ("unique (redundant averaged)", R["unique_tol0.05m"], SEQ[1]),
            ("uv cells, one channel", R["cells_per_channel_full"], SEQ[1]),
            ("gridded stream, full FoV, n_sub 2", R["full nsub2"], SLOT[1]),
            ("gridded stream, full FoV, n_sub 4", R["full nsub4"], SLOT[1]),
            ("gridded stream, FoV/2 symmetric, n_sub 2 (v3/plan5)", R["FoV/2 sym nsub2"], SLOT[1]),
            ("gridded stream, 256x128 (u halved), n_sub 2", R["FoV/2 halve u nsub2"], SLOT[0]),
            ("gridded stream, 256x128 (u halved), n_sub 4", R["FoV/2 halve u nsub4"], SLOT[0]),
            ("per-unique-baseline stream, any n_sub", R["unique_tol0.05m"], SLOT[0])]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, c) in zip(y, rows):
        ax.barh(yy, v, 0.62, color=c)
        ax.text(v + 30, yy, f"{v:.0f}", va="center", fontsize=7.4, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.set_xlim(0, 3700)
    ax.set_xlabel("complex entries per (corr node, sub-band) stream per sample")
    ax.set_title("Redundancy is already averaged by gridding; the union over channels is not")
    ax.grid(axis="y", visible=False)
    save(fig, "f2_redundancy.png")


# ---------------------------------------------------------------- F3 corner-turn volume vs fabric scenarios
AMBS = ["2x", "4x below DM 500", "4x below DM 800", "4x below DM 1200", "4x to DM 1600"]
LABS = ["2x", "4x\n< DM 500", "4x\n< DM 800", "4x\n< DM 1200", "4x\nto DM 1600"]


def f_network():
    F = fabric(H_S)
    vol = [sum(i["gbps"] for i in items(a, "FoV/2", 2)) for a in AMBS]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(AMBS))
    ax.bar(x, vol, 0.55, color=SEQ[2])
    for xi, v in zip(x, vol):
        ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=7.6, color=INK2)
    caps = [(k, N_S * d["search"] + 16 * d["corr"]) for k, d in F.items()]
    top = 280
    for (k, c), col in zip(caps, (INK2, SLOT[5], SLOT[6], SLOT[7])):
        if c > top:
            ax.text(-0.42, top - 12, f"{k}: {c:.0f} Gb/s (off scale)", ha="left", fontsize=7.2, color=INK2)
            continue
        ax.axhline(c, color=col, lw=1.3, ls=(0, (4, 3)) if k == "baseline" else "-")
        ax.text(-0.42, c + 3, f"{k}: {c:.0f}", ha="left", fontsize=7.2, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABS, fontsize=7.8)
    ax.set_ylim(0, top)
    ax.set_xlim(-0.45, len(AMBS) - 0.5)
    ax.set_ylabel("corner-turn data, Gb/s (FoV/2, cint4)")
    ax.set_title("4x is network-bound: the two-stage corner turn grows as 1/T$^2$")
    ax.grid(axis="x", visible=False)
    save(fig, "f3_network.png")


# ---------------------------------------------------------------- F4 feasibility matrix
def f_matrix():
    F = fabric(H_S)
    opts = [("full FoV", "full", 1.0, "#1baf7a"), ("full FoV, optimised kernels", "full", 0.7, "#8fd9b6"),
            ("FoV/2", "FoV/2", 1.0, SLOT[0]), ("FoV/2, optimised kernels", "FoV/2", 0.7, "#9cc4f0")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for j, (fk, d) in enumerate(F.items()):
        for i, a in enumerate(AMBS):
            got = None
            for lab, fov, sc, col in opts:
                if any(fits(items(a, fov, ns, 1.4, sc), d["corr"], d["search"], N_S) for ns in (2, 4)):
                    got = (lab, col)
                    break
            lab, col = got or ("does not fit", "#e4e3df")
            ax.add_patch(plt.Rectangle((j + 0.04, len(AMBS) - 1 - i + 0.06), 0.92, 0.88, color=col, lw=0))
            ax.text(j + 0.5, len(AMBS) - 1 - i + 0.5, lab.replace(", ", ",\n"), ha="center", va="center",
                    fontsize=6.6, color="#0b0b0b")
    ax.set_xlim(0, len(F))
    ax.set_ylim(0, len(AMBS))
    ax.set_xticks(np.arange(len(F)) + 0.5)
    ax.set_xticklabels([k.replace(": ", ":\n") for k in F], fontsize=7.4)
    ax.set_yticks(np.arange(len(AMBS)) + 0.5)
    ax.set_yticklabels([lb.replace("\n", " ") for lb in LABS[::-1]], fontsize=7.6)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Best configuration that fits (search ingress {H_S:.0f} Gb/s x {N_S} nodes, measured)")
    save(fig, "f4_matrix.png")


# ---------------------------------------------------------------- F5 owner cost (measured, rescaled to n_sub 2 streams)
def f_owner():
    n_img = 4096
    rows = [("measured: 256x128, 128 streams x 1900", 7.84, 7.95, 3.77),
            ("256x128, 32 streams x 1773 (n_sub 2)", 7.84 * 32 * 1773 / (128 * 1900), 7.95, 3.77),
            ("measured: 256x256, 128 streams x 1900", 8.5, 16.5, 7.5),
            ("256x256, 32 streams x 1773 (n_sub 2)", 8.5 * 32 * 1773 / (128 * 1900), 16.5, 7.5)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, c, im, d) in zip(y, rows):
        x0 = 0
        for v, col in ((c, SLOT[1]), (im, SLOT[0]), (d, SLOT[5])):
            us = 1e3 * v / n_img
            ax.barh(yy, us - 0.03, left=x0, color=col, height=0.6)
            x0 += us
        ax.text(x0 + 0.1, yy, f"{x0:.1f} us", va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.set_xlim(0, 9.5)
    ax.set_xlabel("us per dirty image (2080 Ti, prototype Triton kernels + cuFFT)")
    ax.legend(handles=[Patch(color=SLOT[1], label="combine streams"), Patch(color=SLOT[0], label="irfft2 + cumsum"),
                       Patch(color=SLOT[5], label="boxcar detector")], ncol=3, fontsize=7.2,
              loc="upper center", bbox_to_anchor=(0.45, -0.32))
    ax.set_title("Imaging owner: 3.3 us/image at 256x128 -> 79 fine DMs per GPU at 0.26 ms")
    ax.grid(axis="y", visible=False)
    save(fig, "f5_owner.png")


if __name__ == "__main__":
    figs4.m2(); figs4.m4(); figs4.m5()
    f_fabric(); f_redundancy(); f_network(); f_matrix(); f_owner()
    print("us/image", us_image("FoV/2", 2), "ok")
