#!/usr/bin/env python3
"""Two-figure summary of the 2026-08-10 on-site RFI scenario tests.

Figure 1 — spectra: the narrow-band residual of antennas 50, 51 and a clean
control, for the baseline, for S4 (minex ON) and for the mean of the
minex-OFF test windows.

Figure 2 — scenario contrast: the comb amplitude through the 12 h with the
test windows marked, and the same as a per-window bar chart with an
off-comb control.

Both use the WITHIN-WINDOW narrow-band residual

    resid(f) = log10 S1(f) - medfilt(log10 S1(f), 21 chan, per node block)

rather than a baseline-subtracted difference. That matters for readability:
in a difference, a spike means the line CHANGED, so S4 -- the one window
that still HAS the comb -- comes out flattest and looks like the clean one.
Here a positive spike means the RFI is present in that window, full stop.

The running median is taken WITHIN each corr node's 96-channel block, never
across a boundary: the 1536-channel axis is 16 independently-calibrated node
bandpasses concatenated, and a kernel straddling a seam would drag the step
between two nodes into the residual and manufacture a line at each of the 16
joins. The 21-channel (2.6 MHz) kernel is deliberately far narrower than the
10 MHz comb spacing so it cannot absorb the comb itself.

1.400-1.440 GHz is masked throughout (Galactic HI).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
from scipy.ndimage import median_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, "/home/ubuntu/vikram/dev/dsa110-rt/src")
sys.path.insert(0, "/home/ubuntu/vikram/dev/dsa110-rt/tools/dashboard/dsa_monitor")
os.environ.setdefault("DSART_TEST", "1")
from freq_mapping import production_freq_axis_GHz     # noqa: E402

HI_LO, HI_HI = 1.400, 1.440
NCH_NODE, KERNEL = 96, 21
PDT_OFFSET_H = 7.0

#: (tag, label, start PDT, end PDT, minex state)
SCEN = [
    ("S1", "shack main breaker OFF",                    "11:11", "11:31", "off"),
    ("S2", "starlink OFF, shack PSUs on, ants off",     "11:31", "11:51", "off"),
    ("S3", "starlink ON, shack PSUs on, ants off",      "11:52", "12:12", "off"),
    ("S4", "mtex ALL OFF, minex 6,3,7,8,9 ON",          "12:16", "12:36", "ON"),
    ("S5", "mtex 4,5 holding, minex ALL OFF",           "12:45", "13:05", "off"),
]
SCOL = {"S1": "#e67e22", "S2": "#2471a3", "S3": "#c0392b",
        "S4": "#117a3d", "S5": "#7d3c98"}
#: strong (20 MHz) comb lines
COMB = (1.330, 1.350, 1.370, 1.450, 1.470, 1.490)
ACOL = {50: "#c0392b", 51: "#e74c3c", 1: "#1f618d", 102: "#117a3d"}


def pdt_to_unix(day_unix, hhmm):
    d = dt.datetime.utcfromtimestamp(day_unix).date()
    h, m = (int(x) for x in hhmm.split(":"))
    return dt.datetime(d.year, d.month, d.day, h, m,
                       tzinfo=dt.timezone.utc).timestamp() + PDT_OFFSET_H * 3600.0


def within_residual(y):
    """Narrow-band residual of one spectrum, per corr-node block."""
    out = np.full_like(y, np.nan)
    for g in range(y.size // NCH_NODE):
        sl = slice(g * NCH_NODE, (g + 1) * NCH_NODE)
        blk = y[sl]
        ok = np.isfinite(blk)
        if ok.sum() < NCH_NODE // 2:
            continue
        filled = np.where(ok, blk, np.nanmedian(blk[ok]))
        out[sl] = np.where(ok, blk - median_filter(filled, size=KERNEL,
                                                   mode="nearest"), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.reduced)
    spec, t = z["spec"], z["t_bin"]
    ants = [int(v) for v in z["ants"]]
    ai = {n: i for i, n in enumerate(ants)}
    f = production_freq_axis_GHz()
    hi = (f >= HI_LO) & (f <= HI_HI)
    keep = ~hi

    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.nanmean(np.log10(np.where(spec > 0, spec, np.nan)), axis=3)
    L[:, :, hi] = np.nan

    wins = [(tag, lab, (t >= pdt_to_unix(t[0], s)) & (t < pdt_to_unix(t[0], e)),
             pdt_to_unix(t[0], s), pdt_to_unix(t[0], e), mx)
            for tag, lab, s, e, mx in SCEN]
    first, last = min(w[3] for w in wins), max(w[4] for w in wins)
    base_m, post_m = t < first, t >= last

    # window-mean spectra -> residual (mean first, then detrend: less noisy)
    def resid_of(mask):
        with np.errstate(invalid="ignore"):
            m = np.nanmean(L[mask], axis=0)
        return np.stack([within_residual(m[k]) for k in range(m.shape[0])])

    R = {"baseline": resid_of(base_m), "post": resid_of(post_m)}
    for tag, lab, m, _, _, _ in wins:
        R[tag] = resid_of(m)
    off_tags = [w[0] for w in wins if w[5] == "off"]
    R["minexoff"] = np.nanmean(np.stack([R[tg] for tg in off_tags]), axis=0)

    # per-time-bin residual, for the time series and the error bars
    nb, na, nch = L.shape
    per_bin = np.full((nb, na, nch), np.nan, dtype=np.float32)
    for b in range(nb):
        for k in range(na):
            per_bin[b, k] = within_residual(L[b, k])
    ci = [int(np.argmin(np.abs(f - c))) for c in COMB]
    off = keep.copy()
    for c in [int(np.argmin(np.abs(f - g)))
              for g in np.arange(1.31, 1.4951, 0.01)]:
        off[max(0, c - 4):c + 5] = False
    with np.errstate(invalid="ignore"):
        amp = np.nanmean(per_bin[:, :, ci], axis=2)              # (nb, na)
        ctl = np.nanmean(np.abs(per_bin[:, :, np.where(off)[0]]), axis=2)

    with PdfPages(a.out) as pdf:
        _fig_spectra(pdf, f, keep, R, ai)
        _fig_contrast(pdf, t, amp, ctl, ai, wins, base_m, post_m, last)
    print("wrote %s (%.2f MiB)" % (a.out, os.path.getsize(a.out) / 2**20))
    return 0


def _fig_spectra(pdf, f, keep, R, ai):
    gap = lambda y: np.where(keep, y, np.nan)                    # noqa: E731
    rows = [(50, "ant 50  (target)"), (51, "ant 51  (target)"),
            (102, "ant 102  (control, north of the Tee) — note the y scale")]
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True)
    for ax, (an, ttl) in zip(axes, rows):
        k = ai[an]
        for c in COMB:
            ax.axvline(c, color="0.85", lw=.9, zorder=0)
        ax.axvspan(HI_LO, HI_HI, color="0.9", zorder=0)
        # minex-OFF drawn LAST/on top: it is the flat one, and underneath
        # the other two it is invisible, which is the whole point of the panel.
        ax.plot(f, gap(R["baseline"][k]), lw=2.6, color="k", zorder=2,
                label="baseline  (pre-test, minex on)")
        ax.plot(f, gap(R["S4"][k]), lw=1.8, ls="--", color="#117a3d", zorder=3,
                label="S4  minex 6,3,7,8,9 ON, all mtex off")
        ax.plot(f, gap(R["minexoff"][k]), lw=1.3, color="#7d3c98", zorder=5,
                label="minex OFF  (mean of S1, S2, S3, S5)")
        ax.axhline(0, color="k", lw=.6)
        ax.set_ylabel("narrow-band\nresidual [dex]")
        ax.grid(alpha=.25)
        ax.set_title(ttl, fontsize=10, loc="left", fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=.95)
    axes[-1].set_xlabel("frequency [GHz]     grey verticals = the 20 MHz comb "
                        "lines · grey band = HI mask, excluded")
    fig.suptitle("The comb is present only when the minex drives are powered\n"
                 "Narrow-band residual within each window — a POSITIVE SPIKE "
                 "MEANS THE RFI IS PRESENT THEN.\n"
                 "Baseline (black) and S4 (green) lie on top of each other at "
                 "every comb line; minex-off (purple) runs flat through them.",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90))
    pdf.savefig(fig); plt.close(fig)


def _fig_contrast(pdf, t, amp, ctl, ai, wins, base_m, post_m, last):
    hrs = (t - t[0]) / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.5),
                             gridspec_kw=dict(height_ratios=[1.15, 1]))

    ax = axes[0]
    for i, (tag, lab, m, t0, t1, mx) in enumerate(wins):
        ax.axvspan((t0 - t[0]) / 3600.0, (t1 - t[0]) / 3600.0,
                   color=SCOL[tag], alpha=.22, zorder=0)
        # Tag only, staggered in height. The windows are 20 min on a 12 h
        # axis, so anything longer than the tag overlaps its neighbours; the
        # minex state per window is spelled out on the bar panel below.
        ax.annotate(tag, xy=(((t0 + t1) / 2 - t[0]) / 3600.0,
                             .97 if i % 2 == 0 else .885),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=9, color=SCOL[tag], fontweight="bold")
    ax.annotate("S4 is the only minex-ON window", xy=(.5, .78),
                xycoords="axes fraction", ha="center", fontsize=8.5,
                style="italic", color="#117a3d")
    for an in (50, 51, 102):
        y = amp[:, ai[an]]
        ax.plot(hrs, np.where(y > 0, y, np.nan),
                lw=(1.7 if an != 102 else 1.0), color=ACOL[an],
                label="ant %d" % an)
    ax.plot(hrs, np.where(ctl[:, ai[50]] > 0, ctl[:, ai[50]], np.nan),
            lw=.9, color="0.5", ls=":", label="off-comb control (ant 50)")
    ax.axvline((last - t[0]) / 3600.0, color="k", ls="--", lw=1.1)
    ax.annotate("array left in the S5 config (minex off) →",
                xy=((last - t[0]) / 3600.0 + .15, .06),
                xycoords=("data", "axes fraction"), fontsize=8.5)
    ax.set_yscale("log"); ax.set_ylim(2e-4, .2)
    ax.set_ylabel("comb amplitude [dex]")
    ax.set_xlabel("hours since capture start  (2026-08-10 08:32 PDT / 15:32 UTC)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8.5, ncol=4, loc="lower left", framealpha=.95)
    ax.set_title("comb amplitude through the 12 h — it switches with the minex "
                 "drives and nothing else", fontsize=10, loc="left",
                 fontweight="bold")

    ax = axes[1]
    cols = ([("baseline", base_m, "k")]
            + [(w[0], w[2], SCOL[w[0]]) for w in wins]
            + [("post-test", post_m, "0.55")])
    show = [50, 51, 1, 102]
    x = np.arange(len(cols))
    w = .8 / len(show)
    for i, an in enumerate(show):
        mu = [np.nanmean(amp[m, ai[an]]) for _, m, _ in cols]
        er = [np.nanstd(amp[m, ai[an]]) / max(np.sqrt(m.sum()), 1)
              for _, m, _ in cols]
        ax.bar(x + i * w - .4 + w / 2, mu, w, yerr=er, color=ACOL[an],
               edgecolor="k", linewidth=.5, error_kw=dict(lw=.9),
               label="ant %d" % an)
    ax.plot(x, [np.nanmean(ctl[m, ai[50]]) for _, m, _ in cols], "k_",
            ms=26, mew=2, label="off-comb control")
    ax.set_yscale("log"); ax.set_ylim(3e-4, .2)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (n, "minex ON" if n in ("baseline", "S4")
                                    else "minex off") for n, _, _ in cols],
                       fontsize=8.5)
    ax.set_ylabel("comb amplitude [dex]")
    ax.grid(alpha=.3, axis="y", which="both")
    ax.legend(fontsize=8.5, ncol=5, loc="upper right", framealpha=.95)
    ax.set_title("same, per configuration  (error bars = SEM over 60 s bins)",
                 fontsize=10, loc="left", fontweight="bold")

    fig.suptitle("Scenario contrast — only S4 reproduces the baseline, and S4 "
                 "is the sole window with minex powered\n"
                 "Excluded by the tests: Starlink (on in S3), the shack PSUs "
                 "(S1 breaker-off ≈ S2 PSUs-on), the mtex drives (S5).\n"
                 "Ant 1 is high in every window — two always-on lines at "
                 "1350/1450 MHz, a different source that no test affected.",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90))
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
