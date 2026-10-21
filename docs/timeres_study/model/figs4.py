"""Figures for report v4 (measured benchmarks; 4x focus) -> ../figs_v4/.
Usage: python figs4.py <measured corr headroom Gb/s or 'nan'>"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plan4 import n_dm  # noqa: E402

OUT = "../figs_v4/"
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
HC_MEAS = float(sys.argv[1]) if len(sys.argv) > 1 else float("nan")
T1, T2, T4 = 1048.576, 524.288, 262.144


def save(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- M1 owner cost breakdown (measured)
def m1():
    # b3 e2e, chunk detector, K7, F8 (n01/n04 GPU0): ms per 8 fine DMs x 576 rows -> us per output image
    n_img = 8 * 512
    rows = [("256x128 (FoV/2)", 7.84, 7.95, 3.77), ("256x256 (full FoV)", 8.5, 16.5, 7.5)]
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, c, im, d) in zip(y, rows):
        x0 = 0
        for v, col, nm in ((c, SLOT[1], "combine (128 sub-band streams)"), (im, SLOT[0], "irfft2 + cumsum"),
                           (d, SLOT[5], "boxcar detector (chunk-max)")):
            us = 1e3 * v / n_img
            ax.barh(yy, us - 0.03, left=x0, color=col, height=0.6)
            x0 += us
        ax.text(x0 + 0.1, yy, f"{x0:.1f} us/image", va="center", fontsize=8, color=INK2)
    ax.axvline(3.5, color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.text(3.55, y[0] + 0.42, "v3 assumption (FoV/2)", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("measured us per dirty image (prototype kernels, 2080 Ti, isolated)")
    ax.legend(handles=[Patch(color=SLOT[1], label="combine (128 sub-band streams)"),
                       Patch(color=SLOT[0], label="irfft2 + cumsum"),
                       Patch(color=SLOT[5], label="boxcar detector")],
              loc="upper center", bbox_to_anchor=(0.45, -0.38), ncol=3, fontsize=7.4)
    ax.set_title("Imaging owner, end to end: 4.8 us/image at FoV/2 -> 54 fine DMs per GPU at 0.26 ms")
    ax.grid(axis="y", visible=False)
    save(fig, "m1_owner_cost.png")


# ---------------------------------------------------------------- M2 corr GPU budgets at 4x (measured parts)
def m2():
    kinds = {"PCIe H2D (DMA, overlaps)": "#c9c8c2", "unpack": "#86b6ef", "RFI + cal": "#1baf7a",
             "slow GEMM (K=4096)": "#1c5cab", "fast GEMM (t_int 8)": "#2a78d6",
             "stage-1, 31 coarse x 2 sub-bands": "#eb6834", "grid / sky / pack": "#6250d6"}
    rows = [("GPU0 corr_slow (as today)", [("PCIe H2D (DMA, overlaps)", 24.6, 1), ("unpack", 8, 0),
                                           ("slow GEMM (K=4096)", 8.9, 1), ("grid / sky / pack", 7, 0)]),
            ("GPU1 corr_fast at 4x, 61 kHz", [("PCIe H2D (DMA, overlaps)", 24.6, 1), ("unpack", 8, 0),
                                              ("RFI + cal", 15, 0), ("fast GEMM (t_int 8)", 23.4, 1),
                                              ("stage-1, 31 coarse x 2 sub-bands", 35.7, 1),
                                              ("grid / sky / pack", 6, 0)]),
            ("GPU1 corr_fast at 4x, 122 kHz low tier", [("PCIe H2D (DMA, overlaps)", 24.6, 1), ("unpack", 8, 0),
                                                        ("RFI + cal", 15, 0), ("fast GEMM (t_int 8)", 23.4, 1),
                                                        ("stage-1, 31 coarse x 2 sub-bands", 17.6, 1),
                                                        ("grid / sky / pack", 6, 0)])]
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, segs) in zip(y, rows):
        x0 = 0
        for k, ms, meas in segs:
            ax.barh(yy, ms - 0.6, left=x0, color=kinds[k], height=0.6, hatch=None if meas else "////",
                    edgecolor=SURF, lw=0)
            x0 += ms
        ax.text(x0 + 1, yy, f"{x0:.0f} ms ({x0 - 24.6:.0f} of compute)", va="center", fontsize=7.6, color=INK2)
    ax.axvline(134.2, color=INK2, lw=1.1)
    ax.text(133.5, y[-1] - 0.45, "134.2 ms block", fontsize=7.4, color=INK2, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
    ax.set_xlim(0, 165)
    ax.set_xlabel("ms per block (solid = measured in this gap; hatched = earlier measurements / estimates)")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in kinds.items()], ncol=3, fontsize=6.8,
              loc="upper center", bbox_to_anchor=(0.45, -0.3))
    ax.set_title("Corr side: a 4x fast path fits on one corr GPU; today's slow/fast split can stay")
    ax.grid(axis="y", visible=False)
    save(fig, "m2_corr_budget.png")


# ---------------------------------------------------------------- M3 capacity vs corr headroom
def m3():
    from plan4 import plan
    hc = np.linspace(0, 6, 61)
    gpu_per_gbps = 0.16             # imaging GPU-equiv per Gb/s of corner-turn data at FoV/2 (measured cost)
    cap = 8 * 0.8 + 16 * np.minimum(0.4, gpu_per_gbps * hc)
    cap_merged = 8 * 0.8 + 16 * np.minimum(0.8, gpu_per_gbps * hc)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(hc, cap, color=INK, lw=2.2, label="baseline capacity (corr_slow stays on GPU0)")
    ax.plot(hc, cap_merged, color=INK2, lw=1.4, ls=(0, (4, 3)), label="capacity if slow+fast merge onto one GPU")
    variants = [("2x", dict(edgeA=50), SLOT[6]), ("4x below DM 800, tol 1.4", dict(edgeA=800, tol=1.4), SLOT[0]),
                ("4x below DM 800, tol 1.25", dict(edgeA=800), SLOT[3]),
                ("4x to DM 1600, tol 1.4", dict(edgeA=1600, tol=1.4), SLOT[7])]
    for lab, kw, c in variants:
        img, _ = plan(fov="FoV/2", **kw)
        need = img * 4.78e-6
        ax.axhline(need, color=c, lw=1.6)
        ax.text(6.02, need, f"{lab}: {need:.1f}", va="center", fontsize=7.4, color=INK2)
    if np.isfinite(HC_MEAS):
        ax.axvline(HC_MEAS, color=SLOT[5], lw=1.4)
        ax.text(HC_MEAS + 0.05, 1.0, f"measured\n{HC_MEAS:.0f} Gb/s", fontsize=7.4, color=INK2)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 16)
    ax.set_xlabel("corr-node ingress available beside SNAP (Gb/s)")
    ax.set_ylabel("imaging GPU-equivalents (FoV/2)")
    ax.set_title("Baseline 4x below DM 800 needs ~2 Gb/s of corr headroom")
    ax.legend(loc="upper left", fontsize=7.4)
    save(fig, "m3_capacity.png")


# ---------------------------------------------------------------- M4 rate gains (no zoom)
def m4():
    R = json.load(open("results/rates4.json"))["pkgs"]
    order = [("1x: two-stage done right", "1x"), ("2x: 0.52 ms below DM 1600", "2x"),
             ("4x below DM 500", "4x\n<DM 500"), ("4x below DM 800, tol 1.4", "4x\n<DM 800"),
             ("4x to DM 1600, tol 1.4", "4x to\nDM 1600")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(order))
    for k, (pop, lab, c) in enumerate((("m1", "width median 1 ms", SEQ[2]), ("m03", "width median 0.3 ms", SEQ[4]))):
        v = [R[n]["gain"][pop] for n, _ in order]
        ax.bar(x + (k - 0.5) * 0.36, v, 0.34, color=c, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.36, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=7, color=INK2)
    ax.axhline(1, color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([o for _, o in order], fontsize=7.8)
    ax.set_ylabel("detection-rate gain vs today")
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=7.6, loc="upper left")
    ax.set_title("Rate gain from time resolution alone (no threshold change)")
    ax.grid(axis="x", visible=False)
    save(fig, "m4_rates.png")


# ---------------------------------------------------------------- M5 owner-side spectral-occupancy test (model)
def m5():
    """Per-candidate spectrum from the 32 sub-band streams an owner already holds
    (16 chgroups x n_sub 2 = 5.86 MHz each).  The pixel value is the sum of the
    per-stream contributions, so the spectrum costs one 2146-cell DFT per stream.
    Statistic: fraction of S/N^2 in the best two ADJACENT sub-bands (robust to an
    interferer on a chgroup edge, like 1335.02 MHz)."""
    rng = np.random.default_rng(5)
    K, n_mc = 32, 4000
    snr = np.linspace(8, 40, 17)
    f = 1311.25 + 5.859 * (np.arange(K) + 0.5)

    def spec(kind, S):
        if kind == "broadband":
            g = np.ones((n_mc, K))
        elif kind.startswith("band-limited"):
            bw = float(kind.split()[1])
            c = rng.uniform(f[0], f[-1], (n_mc, 1))
            g = np.exp(-0.5 * ((f - c) / (bw / 2.355)) ** 2)
        elif kind == "scintillated (5 MHz)":
            g = rng.exponential(1.0, (n_mc, K))
        elif kind == "interferer 17 MHz":
            c = rng.integers(0, K - 3, (n_mc, 1))
            g = ((np.arange(K) >= c) & (np.arange(K) < c + 3)).astype(float)
        else:                                   # ~1 MHz interferer: one sub-band
            c = rng.integers(0, K, (n_mc, 1))
            g = (np.arange(K) == c).astype(float)
        s = g * S * np.sqrt(K) / g.sum(1, keepdims=True)   # total S/N = sum(x)/sqrt(K) = S
        x = s + rng.standard_normal((n_mc, K))
        p = np.clip(x, 0, None) ** 2
        pair = p[:, :-1] + p[:, 1:]
        return pair.max(1) / np.maximum(p.sum(1), 1e-9)

    kinds = [("broadband", SLOT[0]), ("scintillated (5 MHz)", SLOT[6]), ("band-limited 60 MHz", SLOT[2]),
             ("band-limited 20 MHz", SLOT[4]), ("interferer 17 MHz", SLOT[3]), ("interferer ~1 MHz", SLOT[7])]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    for k, c in kinds:
        q = np.array([np.percentile(spec(k, S), [1, 50, 99]) for S in snr])
        ax.fill_between(snr, q[:, 0], q[:, 2], color=c, alpha=0.18, lw=0)
        ax.plot(snr, q[:, 1], color=c, label=k)
    ax.axhline(0.9, color=INK2, lw=1.1)
    ax.text(39.5, 0.915, "reject above 0.9", ha="right", fontsize=7.4, color=INK2)
    ax.axhline(0.5, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.text(39.5, 0.515, "tag 0.5-0.9", ha="right", fontsize=7.4, color=INK2)
    ax.set_xlim(8, 40)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("candidate S/N (all sub-bands)")
    ax.set_ylabel("S/N$^2$ in best 2 adjacent\nsub-bands (fraction)")
    ax.legend(fontsize=7.2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Owner-side spectral test (model)")
    save(fig, "m5_spectral_test.png")


# ---------------------------------------------------------------- M6 corner-turn volume vs fabric
def m6(H_c, H_s, n_search_ok):
    R = json.load(open("results/plan5.json"))["rows"]
    amb = ["2x", "4x below DM 500", "4x below DM 800", "4x below DM 1200", "4x to DM 1600"]
    lab = ["2x", "4x\n< DM 500", "4x\n< DM 800", "4x\n< DM 1200", "4x\nto DM 1600"]
    pick = {a: min((r for r in R if r["amb"] == a and r["fov"] == "FoV/2" and r["tol"] == 1.4
                    and r["kernels"] == 1.0 and r["n_sub"] in (2, 4)), key=lambda r: r["ct_gbps"]) for a in amb}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(amb))
    ax.bar(x, [pick[a]["ct_gbps"] for a in amb], 0.55, color=SEQ[2])
    for xi, a in zip(x, amb):
        ax.text(xi, pick[a]["ct_gbps"] + 3, f"{pick[a]['ct_gbps']:.0f}", ha="center", fontsize=7.6, color=INK2)
    s_cap = n_search_ok * H_s
    ax.axhline(s_cap, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(4.45, s_cap + 2, f"{n_search_ok} search nodes x {H_s:.0f} Gb/s", ha="right", fontsize=7.4, color=INK2)
    if np.isfinite(H_c):
        tot = s_cap + 16 * H_c
        ax.axhline(tot, color=SLOT[5], lw=1.4)
        ax.text(4.45, tot + 2, f"+ 16 corr nodes x {H_c:.0f} Gb/s (measured)", ha="right", fontsize=7.4,
                color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7.8)
    ax.set_ylabel("corner-turn data (Gb/s, cint4)")
    ax.set_title("4x is network-bound: two-stage volume grows as 1/T$^2$")
    ax.grid(axis="x", visible=False)
    save(fig, "m6_network.png")


if __name__ == "__main__":
    H_S = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    NS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    m1(); m2(); m3(); m4(); m5(); m6(HC_MEAS, H_S, NS)
    print("ok")
