"""Figures for report v2 -> ../figs_v2/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from physics import DNU_NATIVE_MHZ, FREQS_MHZ, chan_smear_us  # noqa: E402

OUT = "../figs_v2/"
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
GREY = "#8d8c86"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0, "legend.fontsize": 8.2})
# fixed entity -> colour (same in every figure)
ENT = {"today": ("today (1.05 ms, 244 kHz, 8x4)", SLOT[1]),
       "x1": ("1x fixed (1.05 ms, 61 kHz, shifts at 0.26 ms)", SLOT[7]),
       "x2": ("2x (0.52 ms, 61 kHz)", SLOT[6]),
       "x4": ("4x (0.26 ms, 61 kHz)", SLOT[0]),
       "x4c4": ("4x (0.26 ms, 122 kHz)", SLOT[4]),
       "x4c8": ("4x (0.26 ms, 244 kHz)", SLOT[3]),
       "zoom": ("voltage zoom (coherent, 32.8 us)", SLOT[5])}


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- R1 effective resolution + DM pdf
def r1():
    dm = np.linspace(0, 3000, 301)

    def rms(dnu):
        return np.array([np.sqrt(np.mean(chan_smear_us(d, dnu, FREQS_MHZ) ** 2)) for d in dm])

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.axvspan(186, 1340, color="#f0efec", zorder=0)
    ax.axvline(500, color="#b9b8b2", lw=0.8, zorder=0)
    ax.text(505, 2.6, "FRB DM median ~500\n(shaded: 90% range)", fontsize=7.5, color=INK2, va="top")
    cases = [("today", 1048.576, 8, 540.0), ("x2", 524.288, 2, 0.0), ("x4c8", 262.144, 8, 0.0),
             ("x4c4", 262.144, 4, 0.0), ("x4", 262.144, 2, 0.0)]
    for key, T, csf, extra in cases:
        w = np.sqrt(T ** 2 + rms(DNU_NATIVE_MHZ * csf) ** 2 + extra ** 2) / 1e3
        lab, c = ENT[key]
        ax.plot(dm, w, color=c, lw=2.6 if key == "x4" else 1.9, label=lab)
    ax.plot(dm, np.full_like(dm, 0.0328), color=ENT["zoom"][1], lw=1.9, label=ENT["zoom"][0])
    nat = rms(DNU_NATIVE_MHZ) / 1e3
    ax.plot(dm, nat, color=GREY, lw=1.2, ls=(0, (4, 3)), label="smear of one native 30.5 kHz channel")
    ax.set_yscale("log")
    ax.set_xlim(0, 3000)
    ax.set_ylim(0.02, 3)
    ax.set_xlabel("DM (pc cm$^{-3}$)")
    ax.set_ylabel("effective time resolution (ms)")
    ax.set_title("With 61-122 kHz channels the sample sets the resolution below DM ~1000")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17), fontsize=7.8)
    save(fig, "r1_resolution.png")


# ---------------------------------------------------------------- R2 S/N vs width
B = np.load("results/broad_grid.npz")
W = B["widths_us"] / 1e3
DMS = list(B["dms"])


def r2():
    sel = [("today", "today"), ("x1", "tc1_c2"), ("x2", "ts2_c2"), ("x4", "ts4_c2"), ("zoom", "zoom")]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, dm in zip(axs, (500.0, 1500.0)):
        i = DMS.index(dm)
        for ent, k in sel:
            lab, c = ENT[ent]
            ax.plot(W, B[k][i], color=c, marker="o", ms=3, lw=2.5 if ent == "x4" else 1.8, label=lab)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(f"DM {int(dm)}")
        ax.set_xlabel("intrinsic burst FWHM (ms)")
    axs[0].set_ylabel("S/N recovered / ideal matched filter")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7.6)
    fig.suptitle("Recovered S/N: the sampling step dominates; the zoom recovers almost everything",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, "r2_snr_vs_width.png")


# ---------------------------------------------------------------- R3 one-shot vs two-stage, matched
def r3():
    j = list(B["widths_us"]).index(100.0)
    groups = [
        ("1.05 ms", [("today: 2-stage, shifts at 1.05 ms, 244 kHz", "today", SLOT[1]),
                     ("one-shot multicast, shifts 1.05 ms, 244 kHz", "os1_c8", "#c9c8c2"),
                     ("2-stage, native stage-1, stage-2 at 1.05 ms", "ts1_c2", SEQ[1]),
                     ("two corner turns, shifts at 0.26 ms", "tc1_c2", SEQ[3])]),
        ("0.52 ms", [("2-stage, stage-2 at 0.26 ms", "ts2_c2", SEQ[1]),
                     ("two corner turns, shifts at 0.26 ms", "tc2_c2", SEQ[3])]),
        ("0.26 ms", [("2-stage, stage-2 at 0.13 ms", "ts4_c2", SEQ[1]),
                     ("two corner turns, shifts at 0.26 ms", "tc4_c2", SEQ[2]),
                     ("two corner turns, shifts at 0.13 ms", "tc4_c2h", SEQ[3])]),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, dm in zip(axs, (500.0, 1500.0)):
        i = DMS.index(dm)
        y = 0
        ticks, labs = [], []
        for gname, rows in groups:
            for lab, k, c in rows:
                v = B[k][i, j]
                ax.barh(y, v, color=c, height=0.7)
                ax.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=7, color=INK2)
                ticks.append(y)
                labs.append(f"{gname}: {lab}")
                y -= 1
            y -= 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlim(0, 0.65)
        ax.set_title(f"DM {int(dm)}, 0.1 ms burst")
        ax.set_xlabel("S/N recovered / ideal")
        ax.grid(axis="y", visible=False)
    fig.suptitle("At matched shift precision and channel width, one-shot = two-stage",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, "r3_oneshot_vs_twostage.png")


# ---------------------------------------------------------------- R4 architecture sketches
def box(ax, x, y, w, h, t, fc, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec="#8d8c86", lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(ax, a, b, t=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2))
    if t:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.035, t, ha="center", fontsize=6.6, color=INK2)


def r4():
    CA, CB, CS, CN = "#cde2fb", "#d7f0e5", "#fde6d6", "#f0efec"
    fig, axs = plt.subplots(3, 1, figsize=(7.6, 7.6))
    for ax in axs:
        ax.set_axis_off()
        ax.set_xlim(-0.01, 1.02)
        ax.set_ylim(0, 1)
    ax = axs[0]
    ax.text(0, 0.97, "A. Two-stage, one corner turn (today's shape, done right)", fontsize=9.5,
            fontweight="bold", va="top")
    box(ax, 0.0, 0.25, 0.27, 0.55, "corr GPU A\nX-engine: slow + fast\nstage-1 at native 32.8 us\n"
        "per coarse DM x 2-4 sub-bands\n(DM-tiered cadence)\ngrid, 4-bit, send", CA)
    box(ax, 0.36, 0.33, 0.2, 0.4, "corner turn\nsub-band streams\nper coarse bucket\n2-13 Gb/s per\nreceiving node", CN)
    box(ax, 0.65, 0.25, 0.35, 0.55, "24 imaging owners\n(16 corr GPU B + 8 search)\nstage-2 combine (one\n"
        "rounding) -> FFT -> boxcar\n-> C1/C2", CB)
    arr(ax, (0.27, 0.53), (0.36, 0.53))
    arr(ax, (0.56, 0.53), (0.65, 0.53))
    ax.text(0, 0.12, "+ fits today's x4 NICs (corr <= 22 Gb/s incl. SNAP)   - corr nodes carry the DM plan "
            "(buckets, tiers)   - owner loss = DM gap", fontsize=7, color=INK2)
    ax = axs[1]
    ax.text(0, 0.97, "B. Two corner turns, one-shot dedispersion (your proposal)", fontsize=9.5,
            fontweight="bold", va="top")
    box(ax, 0.0, 0.25, 0.2, 0.55, "corr GPU A\nX-engine\ngrid every\n61 kHz channel\nat 0.26 ms\n4-bit, send", CA)
    box(ax, 0.26, 0.25, 0.18, 0.55, "CT1 by uv cell\n~8 Gb/s/node\n(eno2 10G or\nNIC in x16 slot)", CN)
    box(ax, 0.5, 0.25, 0.18, 0.55, "cell owners\n(any node)\none-shot FDMT\nall DMs, DM-\ntiered output", CB)
    box(ax, 0.73, 0.25, 0.27, 0.55, "CT2 by DM group\n-> 24 imaging owners\nFFT -> boxcar\n-> C1/C2", CS)
    arr(ax, (0.2, 0.53), (0.26, 0.53))
    arr(ax, (0.44, 0.53), (0.5, 0.53))
    arr(ax, (0.68, 0.53), (0.73, 0.53))
    ax.text(0, 0.12, "+ corr side fixed at final resolution; DM plan / cadence only downstream; owner loss "
            "= small loss everywhere   - CT1 needs +8 Gb/s/node", fontsize=7, color=INK2)
    ax = axs[2]
    ax.text(0, 0.97, "C. Voltage zoom on candidates (add to A or B)", fontsize=9.5, fontweight="bold",
            va="top")
    box(ax, 0.0, 0.25, 0.22, 0.55, "search candidate\n>= theta1 (~8 sigma)\n+ governor\n(top-N / s)", CS)
    box(ax, 0.29, 0.25, 0.3, 0.55, "16 corr nodes: 25 s voltage ring\n3-9 coherent beams at (l, m)\n"
                                   "coherent dedispersion per channel\nlocal DM search, native 32.8 us", CA)
    box(ax, 0.66, 0.25, 0.34, 0.55, "aggregator: sum 16 chgroups\nS/N, width, DM at native res.\n"
                                    "-> C2 / dump with refined params", CB)
    arr(ax, (0.22, 0.53), (0.29, 0.53))
    arr(ax, (0.59, 0.53), (0.66, 0.53))
    ax.text(0.255, 0.86, "request ~3-5 s\nafter the burst", ha="center", fontsize=6.6, color=INK2)
    ax.text(0.625, 0.86, "~50 MB per\ncandidate", ha="center", fontsize=6.6, color=INK2)
    ax.text(0, 0.12, "+ 0.92 of ideal at any DM, any width; ~ms of GPU per candidate (bbproc kernels)   "
            "- detection still set by theta1 of the search", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "r4_architectures.png")


# ---------------------------------------------------------------- R5 network per node
def r5():
    P = json.load(open("results/packages2.json"))
    br = P["breakdown"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    y = np.arange(len(br))[::-1]
    for yy, b in zip(y, br):
        x0 = 0
        for val, c in ((b["snap"], GREY), (b["ct1"], SLOT[3]), (b["ct2"], SLOT[0])):
            if val > 0:
                ax.barh(yy, val - 0.2, left=x0, color=c, height=0.62)
                x0 += val
        ax.text(x0 + 0.5, yy, f"{x0:.1f}", va="center", fontsize=8, color=INK2)
    ax.axvline(25, color=SLOT[7], lw=1.4)
    ax.axvline(50, color=GREY, lw=1.0, ls=(0, (4, 3)))
    ax.text(25.4, y[0] + 0.5, "NIC today: x4 slot, ~25 Gb/s", fontsize=7.4, color=INK2)
    ax.text(50.4, y[0] + 0.5, "NIC in x16 slot\n(x8 card): ~50 Gb/s", fontsize=7.4, color=INK2, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"corr node, {b['label']}" for b in br], fontsize=7.8)
    ax.set_xlim(0, 62)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.0)
    ax.set_xlabel("corr-node NIC ingress (Gb/s), 4-bit uv values, 61 kHz")
    ax.legend(handles=[Patch(color=GREY, label="SNAP voltages"), Patch(color=SLOT[3], label="CT1 (per-channel cells)"),
                       Patch(color=SLOT[0], label="corner turn to imaging owners")],
              loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, fontsize=7.5)
    ax.set_title("Two corner turns needs the NIC moved to an x16 slot (or eno2 10G for CT1)")
    ax.grid(axis="y", visible=False)
    save(fig, "r5_network.png")


# ---------------------------------------------------------------- R6 GPU budget
def r6():
    P = json.load(open("results/packages2.json"))["rows"]
    pick = {(r["kind"], r["scheme"], r["csf"]) for r in P}
    rows = [r for r in P if r["scheme"] == "two-stage" and r["csf"] == 2]
    from packages2 import n_dm
    unif = {"1x": n_dm(50, 3000, 1048.576, 2) / 1048.576e-6, "2x": n_dm(50, 3000, 524.288, 2) / 524.288e-6,
            "4x": n_dm(50, 3000, 262.144, 2) / 262.144e-6}     # images/s, uniform cadence to DM 3000
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(3)
    tiers = [r["images_per_s"] * 3.5e-6 for r in rows]
    tiers_full = [r["images_per_s"] * 5.9e-6 for r in rows]
    uni_full = [unif[k] * 5.9e-6 for k in ("1x", "2x", "4x")]
    ax.bar(x - 0.27, uni_full, 0.26, color=SEQ[0], label="uniform cadence to DM 3000, full FoV")
    ax.bar(x, tiers_full, 0.26, color=SEQ[2], label="DM-tiered, full FoV (256x256)")
    ax.bar(x + 0.27, tiers, 0.26, color=SEQ[4], label="DM-tiered, FoV/2 (256x128)")
    for yv, t, ls in ((8, "8 search GPUs", (0, (1, 2))), (24, "24 owners", (0, (5, 3)))):
        ax.axhline(yv, color=INK2, lw=1.0, ls=ls, label=f"capacity: {t}")
    ax.set_xticks(x)
    ax.set_xticklabels(["1x (1.05 ms)", "2x (0.52 ms)", "4x (0.26 ms)"])
    ax.set_ylabel("GPU-equivalents (rewritten kernels)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 100)
    ax.legend(fontsize=7.3, ncol=2, loc="upper left")
    ax.set_title("DM tiers halve the 4x imaging load; everything fits in 24 owners")
    ax.grid(axis="x", visible=False)
    save(fig, "r6_gpu.png")


# ---------------------------------------------------------------- R7 rate gains
def r7():
    R = json.load(open("results/rates.json"))["rate_gain"]
    order = [("1x fixed (2-stage)", "1x, 2-stage"), ("1x fixed (two CTs)", "1x, two CTs"),
             ("2x tiers (2-stage)", "2x tiers"), ("4x tiers (2-stage, 61 kHz)", "4x tiers"),
             ("4x uniform (2-stage, 61 kHz)", "4x uniform"),
             ("today + zoom", "today + zoom"), ("2x tiers + zoom", "2x tiers + zoom"),
             ("4x tiers + zoom", "4x tiers + zoom")]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(order))
    for k, (pop, c) in enumerate((("median 1 ms", SEQ[2]), ("median 0.3 ms", SEQ[4]))):
        v = [R[n][pop] for n, _ in order]
        ax.bar(x + (k - 0.5) * 0.36, v, 0.34, color=c, label=f"width population: {pop}")
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.36, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=6.8, color=INK2)
    ax.axhline(1, color=GREY, lw=0.8)
    ax.axvline(4.5, color="#b9b8b2", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([o for _, o in order], fontsize=7.6, rotation=20, ha="right")
    ax.set_ylabel("detection-rate gain vs today")
    ax.set_ylim(0, 3.0)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Rate gain (Euclidean, DM median 500, DM <= 1500)")
    ax.grid(axis="x", visible=False)
    save(fig, "r7_rates.png")


if __name__ == "__main__":
    r1(); r2(); r3(); r4(); r5(); r6(); r7()
    print("ok")
