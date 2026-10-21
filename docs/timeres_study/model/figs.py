"""Figures + derived numbers for the time-resolution report -> ../figs/, results/derived.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from physics import DNU_NATIVE_MHZ, FREQS_MHZ, chan_smear_us  # noqa: E402

OUT = "../figs/"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# fixed entity -> categorical slot (reference palette, light mode, fixed order)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ENT = {  # key: (label, slot)
    "x4_c1": ("4x: 0.26 ms, 30.5 kHz (proposed)", 0),
    "deployed": ("today: 1.05 ms, 244 kHz", 1),
    "legacy": ("legacy dsaX: 0.26 ms, 244 kHz", 2),
    "x4_c8": ("4x: 0.26 ms, 244 kHz (same channels)", 3),
    "x4_c2": ("4x: 0.26 ms, 61 kHz", 4),
    "x8_c1": ("8x: 0.13 ms, 30.5 kHz", 5),
    "x2_c1": ("2x: 0.52 ms, 30.5 kHz", 6),
    "x1_c1": ("1.05 ms, 30.5 kHz (dedisp. fix only)", 7),
}
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # blue ramp, ordinal-safe steps

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0, "legend.fontsize": 8.5,
})
derived = {}


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- F1 effective resolution
def f1():
    dm = np.linspace(0, 3000, 301)
    nu = FREQS_MHZ

    def rms_smear(dnu_mhz):
        return np.array([np.sqrt(np.mean(chan_smear_us(d, dnu_mhz, nu) ** 2)) for d in dm])

    cases = [
        ("deployed", 1048.576, 8, 540.0),     # + rms intra-sub-band residual of the 8x4 chain
        ("legacy", 262.144, 8, 0.0),
        ("x4_c8", 262.144, 8, 0.0),
        ("x4_c2", 262.144, 2, 0.0),
        ("x4_c1", 262.144, 1, 0.0),
        ("x8_c1", 131.072, 1, 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for key, T, csf, extra in cases:
        if key == "legacy":
            continue                                   # identical curve to x4_c8
        w = np.sqrt(T ** 2 + rms_smear(DNU_NATIVE_MHZ * csf) ** 2 + extra ** 2) / 1e3
        lab, s = ENT[key]
        if key == "x4_c8":
            lab = "0.26 ms, 244 kHz (= legacy dsaX)"
        ax.plot(dm, w, color=SLOT[s], lw=2.6 if key == "x4_c1" else 2.0, label=lab)
    # best any incoherent channelisation of the 30.5 kHz x 32.8 us native grid can do
    best = []
    for d in dm:
        opts = [np.sqrt((32.768 * k) ** 2 + np.mean(chan_smear_us(d, DNU_NATIVE_MHZ / k, nu) ** 2))
                for k in (1, 2, 4, 8, 16)]
        best.append(min(opts) / 1e3)
    ax.plot(dm, best, color=INK2, lw=1.2, ls=(0, (4, 3)),
            label="floor: best up-channelisation (k=1..16)")
    for y, t, xx, va, f in ((1.048576, "today's sample 1.05 ms", 2050, "bottom", 1.04),
                            (0.262144, "4x target 0.26 ms", 1650, "top", 0.96)):
        ax.axhline(y, color="#b9b8b2", lw=0.8, zorder=0)
        ax.text(xx, y * f, t, ha="left", va=va, color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(0, 3000)
    ax.set_ylim(0.05, 3)
    ax.set_xlabel("DM (pc cm$^{-3}$)")
    ax.set_ylabel("effective time resolution (ms)")
    ax.set_title("Sampling and channel width set the floor; today's chain is ~4-6x coarser than 4x-native")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    save(fig, "f1_effective_resolution.png")


# ---------------------------------------------------------------- F2 recovered S/N vs width
G = np.load("results/final_grid.npz")
W = G["widths_us"] / 1e3
DMS = list(G["dms"])


def f2():
    keys = ["deployed", "legacy", "x4_c8", "x4_c2", "x4_c1", "x8_c1"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for ax, dm in zip(axs, (300.0, 1500.0)):
        i = DMS.index(dm)
        for k in keys:
            lab, s = ENT[k]
            ax.plot(W, G[k][i], color=SLOT[s], lw=2.6 if k == "x4_c1" else 1.8,
                    marker="o", ms=3.2, label=lab)
        ax.set_xscale("log")
        ax.set_title(f"DM {int(dm)}")
        ax.set_xlabel("intrinsic burst FWHM (ms)")
        ax.axvspan(0.025, 1.0, color="#f0efec", zorder=0)
        ax.text(0.03, 0.97, "50% of DSA bursts", va="top", fontsize=7.5, color=INK2)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("S/N recovered / ideal matched filter")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.6, bbox_to_anchor=(0.5, -0.0))
    fig.suptitle("What reaches the detector (emulated chain incl. sampling, channel smear, DM grid)",
                 fontsize=10.5, fontweight="bold", y=1.02)
    save(fig, "f2_recovered_snr.png")


def f3():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, dm in enumerate((300.0, 800.0, 1500.0)):
        i = DMS.index(dm)
        g = G["x4_c1"][i] / G["deployed"][i]
        ax.plot(W, g, color=SEQ[j + 1], marker="o", ms=3.2, label=f"4x native vs today, DM {int(dm)}")
        g2 = G["x1_c1"][i] / G["deployed"][i]
        ax.plot(W, g2, color=SEQ[j + 1], ls=(0, (3, 2)), lw=1.4,
                label=f"dedispersion fix only (1.05 ms), DM {int(dm)}")
    ax.axhline(1, color="#b9b8b2", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("intrinsic burst FWHM (ms)")
    ax.set_ylabel("S/N gain at fixed fluence (x)")
    ax.set_title("Gain over the deployed chain: x2.5-3 for sub-0.3 ms bursts, x1.6 even at 1 ms")
    h, l = ax.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    ax.legend([h[i] for i in order], [l[i] for i in order], ncol=2, fontsize=7.4, loc="upper right")
    ax.set_ylim(0.9, 3.9)
    save(fig, "f3_gain_vs_today.png")


def rate_gains():
    """Euclidean rate gain: <f_new^1.5>/<f_old^1.5> over a log-normal width population."""
    lw = np.log(W)
    fine = np.linspace(lw[0], lw[-1], 600)          # uniform in log W (grid points are not)
    out = {}
    for med, sig, tag in ((1.0, 1.0, "median 1 ms (DSA-like)"), (0.3, 0.8, "median 0.3 ms (narrow)")):
        p = np.exp(-0.5 * ((fine - np.log(med)) / sig) ** 2)
        p /= p.sum()
        row = {}
        for k in ("legacy", "x1_c1", "x2_c1", "x4_c8", "x4_c2", "x4_c1", "x8_c1"):
            gains = []
            for dm in (300.0, 800.0, 1500.0):
                i = DMS.index(dm)
                fn = np.interp(fine, lw, G[k][i])
                fo = np.interp(fine, lw, G["deployed"][i])
                gains.append((p * fn ** 1.5).sum() / (p * fo ** 1.5).sum())
            row[k] = float(np.mean(gains))
        out[tag] = row
    derived["rate_gain"] = out
    return out


# ---------------------------------------------------------------- F4 dedispersion fidelity
def f4():
    sw = json.load(open("results/dedisp_sweep.json"))
    wi = sw["W_us"].index(250.0)
    want = [("1st q=native (ideal)", "ideal 1-stage (native shifts)"),
            ("1st q=T (brute force)", "brute-force 1-stage (shifts at T)"),
            ("2st 8x4 q1=nat q2=T", "2-stage 8x4 (today's product)"),
            ("2st 16x8 q1=nat q2=T", "2-stage 16x8"),
            ("2st 16x8 q1=nat q2=T/2", "2-stage 16x8, stage-2 at T/2"),
            ("2st 32x8 q1=nat q2=T/2", "2-stage 32x8, stage-2 at T/2")]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, (hi, dm) in zip(axs, ((1600.0, "1500"), (3100.0, "2500"))):
        rows = {r["tag"].strip(): r for r in sw["rows"] if r["dm_hi"] == hi}
        vals = [rows[t]["res"][dm + ".0"][wi] for t, _ in want]
        ref = vals[1]
        y = np.arange(len(want))[::-1]
        cols = ["#c9c8c2", "#8d8c86", "#eb6834", "#86b6ef", "#3987e5", "#1c5cab"]
        ax.barh(y, vals, color=cols, height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
        ax.axvline(ref, color="#8d8c86", lw=0.8, ls=(0, (3, 2)))
        ax.set_xlim(0.3, 0.86)
        ax.set_title(f"DM {dm} (DM plan to {int(hi) - 100})")
        ax.set_xlabel("S/N recovered / ideal")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(np.arange(len(want))[::-1])
    axs[0].set_yticklabels([lab for _, lab in want], fontsize=8)
    fig.suptitle("Sized right, 2-stage = brute force (T 0.26 ms, 30.5 kHz, 0.25 ms burst)",
                 fontsize=10.5, fontweight="bold", y=1.03)
    save(fig, "f4_dedisp_fidelity.png")


# ---------------------------------------------------------------- F5 compute scaling
def f5():
    pts = json.load(open("results/cost_points.json"))
    sel = [0, 2, 3, 4, 5, 7, 8]
    names = ["today", "2x\n30.5 kHz", "4x\n244 kHz", "4x\n61 kHz", "4x\n30.5 kHz",
             "4x 30.5 kHz\nDM<=3000", "8x\n30.5 kHz"]
    tiers = [("gpu_today_wall", "at today's in-situ cost (41.9 us/image)"),
             ("gpu_today_kernels", "today's kernels, no CPU stalls (13.7 us)"),
             ("gpu_proj_full", "rewritten owner kernels (5.9 us, assumed)"),
             ("gpu_proj_half", "rewritten + FoV/2 grid 256x128 (3.5 us, assumed)")]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(sel))
    bw = 0.2
    for j, (k, lab) in enumerate(tiers):
        v = [pts[i][k] for i in sel]
        ax.bar(x + (j - 1.5) * bw, v, width=bw - 0.02, color=SEQ[j], label=lab)
    for y, t, ls in ((8, "capacity: 8 search GPUs", (0, (1, 2))),
                     (24, "capacity: 24 = 16 corr 2nd GPUs + 8 search", (0, (5, 3))),
                     (40, "capacity: all 40 GPUs", (0, (8, 2, 2, 2)))):
        ax.axhline(y, color="#52514e", lw=1.0, ls=ls, label=t)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 3000)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("2080 Ti GPUs for imaging + detection")
    ax.set_title("Imaging work grows ~16-20x for 4x; it fits only with new kernels AND the corr GPUs")
    ax.legend(fontsize=7.2, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "f5_compute.png")


# ---------------------------------------------------------------- F6 network
def f6():
    rows = [("corr node today", 18.0, 1.07, 0.0),
            ("search node today", 0.0, 4.27, 0.0),
            ("corr owner, 4x, cint4 (1 bucket)", 18.0, 7.4, 0.0),
            ("corr owner, 4x, cint8 (1 bucket)", 18.0, 14.8, 0.0),
            ("search owner, 4x, cint4 (2 buckets)", 0.0, 14.8, 0.0),
            ("search owner, 4x, cint8 (2 buckets)", 0.0, 29.6, 0.0)]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, snap, ct, _) in zip(y, rows):
        ax.barh(yy, snap, color="#8d8c86", height=0.6)
        ax.barh(yy, ct, left=snap + (0.25 if snap else 0), color=SLOT[0], height=0.6)
        ax.text(snap + ct + 0.6, yy, f"{snap + ct:.1f}", va="center", fontsize=8, color=INK2)
    from matplotlib.patches import Patch
    ax.axvline(25, color=SLOT[7], lw=1.4)
    ax.text(25.4, y[0] + 0.55, "PCIe Gen3 x4 NIC link, ~25 Gb/s usable (all 20 nodes)", fontsize=7.5,
            color=INK2, va="bottom")
    ax.axvline(40, color="#8d8c86", lw=0.9, ls=(0, (4, 3)))
    ax.text(40.3, y[2], "40G\nport", fontsize=7.5, color=INK2, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("NIC ingress (Gb/s)")
    ax.set_xlim(0, 46)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.1)
    ax.legend(handles=[Patch(color="#8d8c86", label="SNAP voltages (capture)"),
                       Patch(color=SLOT[0], label="corner turn (uv sub-band streams)")],
              loc="upper center", fontsize=7.5, bbox_to_anchor=(0.45, -0.2), ncol=2)
    ax.set_title("The NIC's PCIe x4 link, not the 40G port, is the network limit")
    ax.grid(axis="y", visible=False)
    save(fig, "f6_network.png")


# ---------------------------------------------------------------- F7 architecture diagram
def box(ax, x, y, w, h, text, fc, ec="#8d8c86", fs=7.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, wrap=True)


def arrow(ax, x0, y0, x1, y1, text=None, fs=6.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=INK2))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, text, ha="center", fontsize=fs, color=INK2)


def f7():
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 7.4))
    for _a in axs:
        _a.set_xlim(-0.01, 1.02)
    C_A, C_B, C_S, C_N = "#cde2fb", "#d7f0e5", "#fde6d6", "#f0efec"
    # ---- today
    ax = axs[0]
    ax.set_axis_off()
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(0, 1)
    ax.text(0, 0.98, "TODAY (16 corr + 4 search nodes)", fontsize=10, fontweight="bold", va="top")
    box(ax, 0.0, 0.55, 0.13, 0.25, "SNAP UDP\n18 Gb/s\n-> capture,\nmerge -> fada", C_N)
    box(ax, 0.18, 0.66, 0.30, 0.22, "corr GPU0: corr_slow\nH2D+unpack 32 ms, GEMM 9 ms\n-> bada 134 ms vis -> UVH5\n(~47% busy)", C_A)
    box(ax, 0.18, 0.30, 0.30, 0.30, "corr GPU1: corr_fast (124/134 ms)\nH2D+unpack AGAIN, RFI, cal\nGEMM sums 8 ch (244 kHz)\nstage-1 x 8 coarse DMs @1.05 ms\ngrid, static sky, Python TX", C_A)
    box(ax, 0.58, 0.30, 0.17, 0.30, "corner turn\n16 -> 4 nodes\n~1 Gb/s per corr\n(cint8, 64 streams\nx 2 DM per node)", C_N)
    box(ax, 0.80, 0.30, 0.20, 0.30, "search GPUs x8\n1 coarse DM each\ncombine 25 fine DM\nirfft2 256^2\nboxcar b1-b64\n(4.96 of 4.967 cubes/s)", C_S)
    box(ax, 0.80, 0.05, 0.20, 0.16, "C1 -> C2 (h23)\n-> C3 voltage dumps", C_N)
    arrow(ax, 0.13, 0.70, 0.18, 0.76)
    arrow(ax, 0.13, 0.62, 0.18, 0.47)
    arrow(ax, 0.48, 0.45, 0.58, 0.45)
    arrow(ax, 0.75, 0.45, 0.80, 0.45)
    arrow(ax, 0.90, 0.30, 0.90, 0.21)
    ax.text(0.18, 0.18, "Limits: 244 kHz channels summed in the GEMM; 8 coarse DMs pinned to the 8 search GPUs;\n"
            "all corr work after the GEMM scales 1/t_int (x8 measured 169 ms in May); two GPUs\n"
            "both copy + unpack + correlate the same voltages; search GPUs at their limit.",
            fontsize=7.4, color=INK2, va="top")
    # ---- proposed
    ax = axs[1]
    ax.set_axis_off()
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(0, 1)
    ax.text(0, 0.98, "PROPOSED (same 40 GPUs, 24 imaging owners)", fontsize=10, fontweight="bold", va="top")
    box(ax, 0.0, 0.58, 0.13, 0.22, "SNAP UDP\n18 Gb/s\n-> capture,\nmerge -> fada", C_N)
    box(ax, 0.17, 0.48, 0.31, 0.44, "corr GPU A: X-engine + front end\none H2D + unpack\nRFI moments (unchanged excision)\nslow: 96 ant XX/YY K=4096 -> bada\nfast: core Stokes-I @0.26 ms\n384 native ch (no channel sum)\nstage-1 at native precision\n16-32 coarse DM x 8 sub-bands\ngrid -> 1900 cells/stream\nstatic sky, cint4 on GPU, C TX", C_A)
    box(ax, 0.55, 0.62, 0.18, 0.26, "corner turn\n16 -> 24 owners\n7.4 Gb/s per bucket\n(cint4)\nNIC x4 -> x8 advised", C_N)
    box(ax, 0.78, 0.66, 0.22, 0.26, "corr GPU B x16\nimaging owner:\none coarse bucket\ncombine (1 rounding)\nirfft2 256x128\nboxcar b1-b256", C_B)
    box(ax, 0.78, 0.30, 0.22, 0.30, "search GPUs x8\nsame owner code\n(2 buckets / node)\n+ C1 aggregation", C_S)
    box(ax, 0.55, 0.10, 0.20, 0.16, "C1 -> C2 (h23)\n-> C3 voltage dumps\n(widths in samples x4)", C_N)
    arrow(ax, 0.13, 0.69, 0.17, 0.69)
    arrow(ax, 0.48, 0.74, 0.55, 0.74)
    arrow(ax, 0.73, 0.78, 0.78, 0.78)
    arrow(ax, 0.73, 0.70, 0.80, 0.55)
    arrow(ax, 0.86, 0.66, 0.86, 0.60)
    arrow(ax, 0.80, 0.33, 0.75, 0.20, "candidates")
    ax.text(0.0, 0.40, "Images never cross the network\n(a bucket's cube is ~6-12x the\nuv streams that make it), so\n"
            "imaging and detection stay\ntogether on each owner.\n\nSlow vis from the same voltage\nload - no duplicated H2D,\n"
            "unpack or correlation;\nbada contract unchanged.", fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "f7_architecture.png")


# ---------------------------------------------------------------- F8 corr GPU budget
def f8():
    from matplotlib.patches import Patch
    kinds = {"PCIe copy (DMA engine)": "#c9c8c2", "unpack": "#86b6ef", "RFI + cal": "#1baf7a",
             "correlation GEMM": "#2a78d6", "reorder / pack": "#eda100",
             "dedispersion + gridding / combine": "#eb6834", "static sky / quantise / TX / norm": "#6250d6",
             "FFT imaging": "#e34948", "boxcar detector": "#008300"}
    rows = [
        ("today GPU0 corr_slow", [("PCIe copy (DMA engine)", 24, 1), ("unpack", 8, 1),
                                  ("correlation GEMM", 9, 1), ("reorder / pack", 7, 1)]),
        ("today GPU1 corr_fast", [("PCIe copy (DMA engine)", 24, 1), ("unpack", 9, 1), ("RFI + cal", 22, 1),
                                  ("correlation GEMM", 44, 1), ("reorder / pack", 9, 1),
                                  ("dedispersion + gridding / combine", 6, 1),
                                  ("static sky / quantise / TX / norm", 10, 1)]),
        ("proposed GPU A (4x)", [("PCIe copy (DMA engine)", 24, 1), ("unpack", 8, 1), ("RFI + cal", 15, 1),
                                 ("correlation GEMM", 9, 1), ("correlation GEMM", 20, 0),
                                 ("dedispersion + gridding / combine", 18, 0),
                                 ("static sky / quantise / TX / norm", 4, 0)]),
        ("proposed GPU B (owner)", [("PCIe copy (DMA engine)", 14, 0),
                                    ("dedispersion + gridding / combine", 34, 0), ("FFT imaging", 24, 0),
                                    ("boxcar detector", 17, 0), ("static sky / quantise / TX / norm", 5, 0)]),
    ]
    notes = {("proposed GPU A (4x)", 3): "slow", ("proposed GPU A (4x)", 4): "fast t_int 8"}
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, segs) in zip(y, rows):
        x0 = 0
        for j, (kind, ms, measured) in enumerate(segs):
            ax.barh(yy, ms - 0.6, left=x0, color=kinds[kind], height=0.62,
                    hatch=None if measured else "////", edgecolor=SURF, lw=0)
            note = notes.get((lab, j))
            if note and ms >= 9:
                ax.text(x0 + ms / 2, yy + 0.43, note, ha="center", va="bottom", fontsize=6.5, color=INK2)
            x0 += ms
        ax.text(x0 + 1, yy, f"{x0} ms", va="center", fontsize=8, color=INK2)
    ax.axvline(134.2, color="#52514e", lw=1.2)
    ax.text(133.5, y[0] + 0.45, "134.2 ms block", fontsize=7.5, color=INK2, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("ms per 134.2 ms block (hatched = estimate, to be benchmarked; DMA copies overlap compute)")
    ax.set_xlim(0, 150)
    ax.set_ylim(y[-1] - 0.5, y[0] + 0.9)
    ax.set_title("One corr GPU can carry slow + fast at 4x; the other becomes an imaging owner")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in kinds.items()], ncol=3, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.45, -0.22))
    save(fig, "f8_corr_budget.png")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5(); f6(); f7(); f8()
    rg = rate_gains()
    json.dump(derived, open("results/derived.json", "w"), indent=1)
    print(json.dumps(rg, indent=1))
