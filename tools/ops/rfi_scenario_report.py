#!/usr/bin/env python3
"""Narrow-band RFI report for the 2026-08-10 on-site scenario tests.

Plots only, in four sections:

  1. raw spectra, per antenna, one curve per configuration
  2. baseline-subtracted spectra, per configuration, RFI spikes marked
  3. autocorrelation of the baseline-subtracted spectra
  4. baseline-subtracted waterfalls over the full 12 h

Method
------
Everything after section 1 is a BASELINE-SUBTRACTED spectrum:

    delta(f) = log10 S1_window(f) - log10 S1_baseline(f)

Working in log10 and subtracting means the bandpass, the per-antenna gain
and the node-to-node calibration steps cancel exactly -- they are common to
the baseline and to every test window. So no detrending is needed, and
delta is a pure log ratio: +0.30 dex is 2x the baseline power at that
channel, -0.30 dex is half. That also makes it directly comparable between
antennas whose raw levels differ by orders of magnitude.

A negative delta is the interesting case here: it means a line that was
present in the baseline and that the configuration REMOVED.

The baseline is PRE-test only (08:32-11:11 PDT). After the last window the
array was deliberately left in the S5 configuration for optical pointing,
so the post-test hours are shown as a separate persistence check and never
folded into the baseline.

1.400-1.440 GHz is masked everywhere: Galactic HI corrupts the baseline
there and would otherwise dominate every difference.

Spike marking (section 2) is done against a LOCAL median of delta, not
against zero, so a slow broadband drift between two windows cannot be
mistaken for a forest of narrow lines. The threshold is a robust
1.4826*MAD, so a handful of real spikes do not inflate the scatter that
defines them.

Section 3 plots the ACF as a COVARIANCE rather than normalised to 1 at zero
lag: a normalised ACF makes pure noise in a clean antenna look identical to
a strong comb in a dirty one. It is divided by the number of overlapping
channels at each lag, so the triangular taper does not suppress the peaks
it exists to find, and it is evaluated separately either side of the HI
mask so the masked gap never enters the correlation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, List, Tuple

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
PDT_OFFSET_H = 7.0                    # PDT = UTC-7
SPIKE_KERNEL = 41                     # local-median window for spike finding
NCH_NODE = 96                         # channels per corr node block
WITHIN_KERNEL = 21                    # 2.6 MHz: narrower than the 10 MHz comb
SPIKE_NSIGMA = 5.0
ACF_LAG_MIN, ACF_LAG_MAX = 2.0, 30.0

#: (tag, label, start PDT, end PDT), as run on 2026-08-10.
SCENARIOS: List[Tuple[str, str, str, str]] = [
    ("S1", "shack main breaker OFF",                     "11:11", "11:31"),
    ("S2", "starlink OFF, shack PSUs on, all ants off",  "11:31", "11:51"),
    ("S3", "starlink ON, shack PSUs on, all ants off",   "11:52", "12:12"),
    ("S4", "mtex ALL OFF, minex 6,3,7,8,9 ON",           "12:16", "12:36"),
    ("S5", "mtex 4,5 holding, minex ALL OFF",            "12:45", "13:05"),
]
SCOL = {"S1": "#e67e22", "S2": "#2471a3", "S3": "#c0392b",
        "S4": "#117a3d", "S5": "#7d3c98"}

GROUPS: Dict[str, List[int]] = {
    "TARGET   ant 50, 51": [50, 51],
    "east of Tee   ant 1, 2": [1, 2],
    "north of Tee   ant 100, 102": [100, 102],
    "outriggers   ant 108, 114, 115, 116": [108, 114, 115, 116],
}
ACOL = {50: "#c0392b", 51: "#e74c3c", 1: "#1f618d", 2: "#5dade2",
        100: "#117a3d", 102: "#52be80", 108: "#7d3c98", 114: "#000000",
        115: "#af7ac5", 116: "#d2b4de"}


def pdt_to_unix(day_unix: float, hhmm: str) -> float:
    d = dt.datetime.utcfromtimestamp(day_unix).date()
    h, m = (int(x) for x in hhmm.split(":"))
    return dt.datetime(d.year, d.month, d.day, h, m,
                       tzinfo=dt.timezone.utc).timestamp() + PDT_OFFSET_H * 3600.0


def gapped(y: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """NaN out the masked channels so a line BREAKS across the HI gap.

    Plotting f[keep] against y[keep] instead would silently join the last
    channel below 1.400 to the first above 1.440 with a straight segment
    that looks like real, very smooth data.
    """
    return np.where(keep, y, np.nan)


def peak_rows(img: np.ndarray, nrows: int = 384) -> np.ndarray:
    """Shrink the frequency axis keeping the most EXTREME value per block.

    A comb line is 1-2 channels out of 1536. Rendering that array into a few
    hundred pixels of figure height averages each line together with its
    neighbours and washes it out -- the lines are real but invisible. Taking
    the largest-|value| channel in each block instead guarantees a narrow
    line survives to the display, at the cost of making the noise floor look
    slightly rougher.
    """
    nch = img.shape[0]
    g = max(1, nch // nrows)
    n = (nch // g) * g
    r = img[:n].reshape(-1, g, img.shape[1])
    allnan = np.all(~np.isfinite(r), axis=1)
    idx = np.nanargmax(np.where(np.isfinite(r), np.abs(r), -1.0), axis=1)
    out = np.take_along_axis(r, idx[:, None, :], axis=1)[:, 0, :]
    return np.where(allnan, np.nan, out)


def within_residual(y: np.ndarray) -> np.ndarray:
    """Narrow-band residual of ONE window, referenced to nothing else.

    log10 S1 minus a running median taken WITHIN each corr node's
    96-channel block (never across a boundary -- that would drag the step
    between two independently-calibrated node bandpasses into the residual
    and manufacture a false line at each of the 16 seams).

    This exists because the baseline-subtracted panels inverted the natural
    reading: there, a spike means the line CHANGED, so the one window that
    still has the comb (S4) is the flattest, which looks like "no RFI in
    S4" when it means the opposite. Here a positive spike means the RFI IS
    PRESENT in that window, full stop.
    """
    out = np.full_like(y, np.nan)
    for g in range(y.size // NCH_NODE):
        sl = slice(g * NCH_NODE, (g + 1) * NCH_NODE)
        blk = y[sl]
        ok = np.isfinite(blk)
        if ok.sum() < NCH_NODE // 2:
            continue
        filled = np.where(ok, blk, np.nanmedian(blk[ok]))
        sm = median_filter(filled, size=WITHIN_KERNEL, mode="nearest")
        out[sl] = np.where(ok, blk - sm, np.nan)
    return out


def highpass(y: np.ndarray) -> np.ndarray:
    """Remove the slowly-varying part, keeping only narrow-band structure.

    Needed before the ACF: the baseline-subtracted spectra carry a
    common-mode broadband pedestal (~-0.02 to -0.03 dex, present even in
    antennas with no RFI at all) from sky/gain drift between the 2.6 h
    baseline and a 20-minute window. Left in, that drift produces a large
    smooth undulation that dominates the covariance and buries the narrow
    comb peaks the panel exists to show.
    """
    ok = np.isfinite(y)
    if ok.sum() < SPIKE_KERNEL:
        return np.full_like(y, np.nan)
    filled = np.where(ok, y, np.nanmedian(y[ok]))
    return np.where(ok, y - median_filter(filled, size=SPIKE_KERNEL,
                                          mode="nearest"), np.nan)


def find_spikes(delta: np.ndarray, keep: np.ndarray):
    """Channels where delta departs from its LOCAL median, and the threshold.

    Against a local median rather than zero: a slow broadband offset between
    two windows would otherwise light up the whole band as 'spikes'.
    """
    d = np.where(np.isfinite(delta) & keep, delta, np.nan)
    filled = np.where(np.isfinite(d), d, np.nanmedian(d))
    loc = d - median_filter(filled, size=SPIKE_KERNEL, mode="nearest")
    mad = np.nanmedian(np.abs(loc - np.nanmedian(loc)))
    sig = 1.4826 * mad if mad > 0 else np.nan
    if not np.isfinite(sig) or sig == 0:
        return np.array([], dtype=int), np.nan
    return np.where(np.abs(loc) > SPIKE_NSIGMA * sig)[0], sig


def cov_acf(y, f, keep):
    """Covariance vs frequency lag; unbiased for overlap, HI gap excluded."""
    dfMHz = abs(np.diff(f)).mean() * 1e3
    n = int(ACF_LAG_MAX / dfMHz)
    idx = np.where(keep & np.isfinite(y))[0]
    if idx.size < 2:
        return np.arange(n) * dfMHz, np.zeros(n)
    runs, start = [], idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            runs.append((start, idx[i - 1])); start = idx[i]
    runs.append((start, idx[-1]))
    acc, w = np.zeros(n), 0.0
    for lo, hicut in runs:
        seg = np.nan_to_num(y[lo:hicut + 1])
        if seg.size < n + 50:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[seg.size - 1:][:n]
        acc += ac / (seg.size - np.arange(n)) * seg.size
        w += seg.size
    return np.arange(n) * dfMHz, (acc / w if w else acc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.reduced)
    spec, t = z["spec"], z["t_bin"]
    ants = [int(v) for v in z["ants"]]
    f = production_freq_axis_GHz()
    ai = {n: i for i, n in enumerate(ants)}

    hi = (f >= HI_LO) & (f <= HI_HI)
    keep = ~hi

    wins = []
    for tag, lab, s, e in SCENARIOS:
        t0, t1 = pdt_to_unix(t[0], s), pdt_to_unix(t[0], e)
        wins.append((tag, lab, (t >= t0) & (t < t1), t0, t1))
    first_test = min(w[3] for w in wins)
    last_test = max(w[4] for w in wins)
    base_m, post_m = t < first_test, t >= last_test

    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log10(np.where(spec > 0, spec, np.nan))
    logs[:, :, hi, :] = np.nan
    # pol-average once, up front: every panel below is pol-mean
    with np.errstate(invalid="ignore"):
        L = np.nanmean(logs, axis=3)                       # (nb, nant, nch)

        def wmean(m):
            return np.nanmean(L[m], axis=0)                # (nant, nch)

        base = wmean(base_m)
        raw = {"baseline": base}
        for tag, lab, m, _, _ in wins:
            raw[tag] = wmean(m)
        raw["post-test"] = wmean(post_m)
        delta = {k: v - base for k, v in raw.items() if k != "baseline"}
        within = {k: np.stack([within_residual(v[i]) for i in range(v.shape[0])])
                  for k, v in raw.items()}
        wf = L - base[None, :, :]                          # (nb, nant, nch)

    print("bins: baseline=%d %s post=%d"
          % (base_m.sum(), " ".join("%s=%d" % (w[0], w[2].sum()) for w in wins),
             post_m.sum()))

    with PdfPages(a.out) as pdf:
        _sec1_raw(pdf, f, keep, raw, wins, ai)
        _sec1b_within(pdf, f, keep, within, wins, ai)
        _sec2_delta(pdf, f, keep, delta, wins, ants, ai)
        _sec3_both_acf(pdf, f, keep, within, delta, wins, ai)
        _sec3_acf(pdf, f, keep, delta, wins, ai)
        _sec4_waterfall(pdf, t, f, wf, ants, wins, ai, last_test, keep)
    print("wrote %s (%.1f MiB)" % (a.out, os.path.getsize(a.out) / 2**20))
    return 0


def _order(names, wins):
    return ["baseline"] + [w[0] for w in wins] + ["post-test"]


def _style(nm):
    if nm == "baseline":
        return dict(color="k", lw=2.4, zorder=2)
    if nm == "post-test":
        return dict(color="0.55", lw=1.0, ls=":", zorder=3)
    # S4 is the one configuration that reproduces the baseline, so it is
    # drawn dashed, wider and on top: a solid line of equal width is simply
    # invisible under five others that agree with each other.
    if nm == "S4":
        return dict(color=SCOL[nm], lw=1.9, ls="--", zorder=6)
    return dict(color=SCOL[nm], lw=0.9, zorder=4)


# ---------------------------------------------------------------- section 1
def _sec1_raw(pdf, f, keep, raw, wins, ai):
    labs = dict((w[0], w[1]) for w in wins)
    for gi, (gname, glist) in enumerate(GROUPS.items()):
        fig, axes = plt.subplots(len(glist), 1, sharex=True, squeeze=False,
                                 figsize=(14, 2.9 * len(glist) + 1.5))
        for ax, an in zip(axes[:, 0], glist):
            k = ai[an]
            for nm in _order(raw, wins):
                ax.plot(f, gapped(raw[nm][k], keep), label=(
                    nm if nm in ("baseline", "post-test")
                    else "%s  %s" % (nm, labs[nm])), **_style(nm))
            ax.axvspan(HI_LO, HI_HI, color="0.88", zorder=0)
            ax.set_ylabel("ant %d\nlog10 S1" % an)
            ax.grid(alpha=.25)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7.5, ncol=4, loc="upper center",
                   bbox_to_anchor=(.5, .935), frameon=False)
        axes[-1, 0].set_xlabel("frequency [GHz]   —   grey = HI mask, excluded")
        fig.suptitle("1 · RAW SPECTRA — %s\n"
                     "all configurations overlaid (pol-averaged); curves lying "
                     "on top of each other means that configuration changed "
                     "nothing" % gname, fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .885)); pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- section 2
def _sec2_delta(pdf, f, keep, delta, wins, ants, ai):
    # (a) one page per scenario, all antennas
    for tag, lab, m, _, _ in wins:
        fig, axes = plt.subplots(len(GROUPS), 1, figsize=(14, 10.5), sharex=True)
        for ax, (gname, glist) in zip(axes, GROUPS.items()):
            for an in glist:
                k = ai[an]
                d = delta[tag][k]
                ax.plot(f, gapped(d, keep), lw=.7, color=ACOL[an],
                        label="ant %d" % an)
                sp, sig = find_spikes(d, keep)
                if sp.size:
                    ax.plot(f[sp], d[sp], "v", ms=4, mfc="none",
                            mec=ACOL[an], mew=.9)
                ped = np.nanmedian(np.where(keep, d, np.nan))
                ax.axhline(ped, color=ACOL[an], lw=.6, ls=":", alpha=.8)
            ax.axhline(0, color="k", lw=.7)
            ax.axvspan(HI_LO, HI_HI, color="0.88", zorder=0)
            ax.set_ylabel("Δ log10 S1\n[dex]"); ax.grid(alpha=.25)
            ax.legend(fontsize=7, ncol=4, loc="lower right")
            ax.set_title(gname, fontsize=9, loc="left", fontweight="bold")
        axes[-1].set_xlabel("frequency [GHz]   —   grey = HI mask, excluded")
        fig.suptitle(
            "2 · BASELINE-SUBTRACTED — %s  %s\n"
            "▽ = spike >%.0fσ from the LOCAL median.   negative spike = a line "
            "present in the baseline that this configuration REMOVED.\n"
            "dotted line per antenna = its broadband pedestal (common-mode "
            "sky/gain drift vs the 2.6 h baseline, NOT RFI) — judge spikes "
            "against that, not against zero"
            % (tag, lab, SPIKE_NSIGMA),
            fontweight="bold", fontsize=11, color=SCOL[tag])
        fig.tight_layout(rect=(0, 0, 1, .90)); pdf.savefig(fig); plt.close(fig)

    # (b) the two target antennas, every configuration on one page
    labs = dict((w[0], w[1]) for w in wins)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, an in zip(axes, (50, 51)):
        k = ai[an]
        for nm in [w[0] for w in wins] + ["post-test"]:
            d = delta[nm][k]
            ax.plot(f, gapped(d, keep), label=(
                nm if nm == "post-test" else "%s  %s" % (nm, labs[nm])),
                **_style(nm))
        ax.axhline(0, color="k", lw=.8)
        for c in (1.330, 1.350, 1.370, 1.450, 1.470, 1.490):
            ax.axvline(c, color="0.8", lw=.7, zorder=0)
        ax.axvspan(HI_LO, HI_HI, color="0.88", zorder=0)
        ax.set_ylabel("ant %d\nΔ log10 S1 [dex]" % an); ax.grid(alpha=.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, ncol=3, loc="upper center",
               bbox_to_anchor=(.5, .925), frameon=False)
    axes[-1].set_xlabel("frequency [GHz]   —   grey verticals = 20 MHz comb "
                        "lines, grey band = HI mask")
    fig.suptitle("2 · BASELINE-SUBTRACTED — target antennas, all "
                 "configurations\nCAREFUL: S4 (green dashed) lying flat on the pedestal means the "
                 "comb is UNCHANGED — i.e. STILL PRESENT, as in the baseline — NOT that "
                 "the RFI is gone.\nThe other five dig negative spikes 0.02–0.07 dex deep: "
                 "those are the windows where the comb was REMOVED.  See §1b.\nDips shared by ALL "
                 "six curves, S4 included (e.g. 1.3555, 1.380 GHz), are "
                 "transient RFI in the baseline average, not "
                 "configuration-dependent.", fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .875)); pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------- section 1b
COMB_LINES = (1.330, 1.350, 1.370, 1.450, 1.470, 1.490)


def _sec1b_within(pdf, f, keep, within, wins, ai):
    """Within-window narrow-band residual: a spike means RFI IS PRESENT."""
    labs = dict((w[0], w[1]) for w in wins)
    for gname, glist in GROUPS.items():
        fig, axes = plt.subplots(len(glist), 1, sharex=True, squeeze=False,
                                 figsize=(14, 2.9 * len(glist) + 1.7))
        for ax, an in zip(axes[:, 0], glist):
            k = ai[an]
            for nm in _order(within, wins):
                ax.plot(f, gapped(within[nm][k], keep), label=(
                    nm if nm in ("baseline", "post-test")
                    else "%s  %s" % (nm, labs[nm])), **_style(nm))
            for c in COMB_LINES:
                ax.axvline(c, color="0.82", lw=.8, zorder=0)
            ax.axvspan(HI_LO, HI_HI, color="0.88", zorder=0)
            ax.set_ylabel("ant %d\nresidual [dex]" % an)
            ax.grid(alpha=.25); ax.axhline(0, color="k", lw=.6)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7.5, ncol=4, loc="upper center",
                   bbox_to_anchor=(.5, .925), frameon=False)
        axes[-1, 0].set_xlabel("frequency [GHz]   —   grey verticals = 20 MHz "
                               "comb lines, grey band = HI mask")
        fig.suptitle("1b · NARROW-BAND RESIDUAL, WITHIN EACH WINDOW — %s\n"
                     "bandpass removed per corr-node block; NO reference to "
                     "any other window.  A POSITIVE SPIKE = THE RFI IS "
                     "PRESENT IN THAT WINDOW.\n"
                     "Read this before the baseline-subtracted pages, where a "
                     "spike means the line CHANGED, not that it is present."
                     % gname, fontweight="bold", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, .875)); pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- section 3a
def _sec3_both_acf(pdf, f, keep, within, delta, wins, ai):
    """The two ACF framings side by side, so the inversion is explicit."""
    labs = dict((w[0], w[1]) for w in wins)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for row, an in enumerate((50, 51)):
        k = ai[an]
        for col, (src, ttl) in enumerate((
                (within, "WITHIN-WINDOW  —  peak = comb IS PRESENT"),
                (delta, "BASELINE-SUBTRACTED  —  peak = comb CHANGED"))):
            ax = axes[row, col]
            seen = []
            names = _order(src, wins) if col == 0 else (
                [w[0] for w in wins] + ["post-test"])
            for nm in names:
                y = src[nm][k]
                lag, ac = cov_acf(highpass(y) if col == 1 else y, f, keep)
                ax.plot(lag, ac * 1e6, label=(
                    nm if nm in ("baseline", "post-test")
                    else "%s" % nm), **_style(nm))
                seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
            _acf_axes(ax, seen)
            ax.set_ylabel(r"covariance [$10^{-6}$ dex$^2$]")
            ax.set_xlabel("frequency lag [MHz]")
            ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2, loc="upper right")
            ax.set_title("ant %d — %s" % (an, ttl), fontsize=9.5, loc="left",
                         fontweight="bold")
    fig.suptitle("3 · THE TWO ACF FRAMINGS, SIDE BY SIDE\n"
                 "LEFT: baseline and S4 peak at 10/20 MHz — the comb is "
                 "present in those two windows and absent in the other five.\n"
                 "RIGHT: the same fact seen as a difference — S4 is flat "
                 "because nothing changed, which is NOT the same as no RFI.",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .90)); pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- section 3
def _acf_axes(ax, curves):
    for mm in np.arange(10, ACF_LAG_MAX + 1, 10):
        ax.axvline(mm, color="k", ls="--", lw=.9, alpha=.5)
    for mm in np.arange(5, ACF_LAG_MAX + 1, 10):
        ax.axvline(mm, color="k", ls=":", lw=.7, alpha=.3)
    ax.set_xlim(ACF_LAG_MIN, ACF_LAG_MAX)
    if curves:
        v = np.concatenate(curves); v = v[np.isfinite(v)]
        if v.size:
            pad = .12 * max(v.max() - v.min(), 1e-12)
            ax.set_ylim(v.min() - pad, v.max() + pad)


def _sec3_acf(pdf, f, keep, delta, wins, ai):
    labs = dict((w[0], w[1]) for w in wins)
    # (a) target antennas, per configuration
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, an in zip(axes, (50, 51)):
        k, seen = ai[an], []
        for nm in [w[0] for w in wins] + ["post-test"]:
            lag, ac = cov_acf(highpass(delta[nm][k]), f, keep)
            ax.plot(lag, ac * 1e6, label=(
                nm if nm == "post-test" else "%s  %s" % (nm, labs[nm])),
                **_style(nm))
            seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
        _acf_axes(ax, seen)
        ax.set_ylabel(r"covariance [$10^{-6}$ dex$^2$]")
        ax.set_xlabel("frequency lag [MHz]"); ax.grid(alpha=.3)
        ax.legend(fontsize=6.5, ncol=2, loc="upper left")
        ax.set_title("ant %d" % an, fontsize=11, loc="left", fontweight="bold")
    fig.suptitle("3 · ACF OF THE BASELINE-SUBTRACTED SPECTRA — target "
                 "antennas\nNARROW peaks at exactly 10 and 20 MHz = the comb "
                 "(present in all five minex-off windows, absent in S4).\n"
                 "The BROAD 22–26 MHz bump is common-mode drift — it appears "
                 "in every antenna and every window, including clean ones — "
                 "disregard it.\ndashed 10/20/30, dotted 5/15/25 MHz; "
                 "broadband pedestal removed first",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .90)); pdf.savefig(fig); plt.close(fig)

    # (b) every antenna, one representative comb-off configuration
    for tag in ("S1", "S4"):
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        for ax, sub in zip(axes, ([50, 51, 1, 2],
                                  [100, 102, 108, 114, 115, 116])):
            seen = []
            for an in sub:
                lag, ac = cov_acf(highpass(delta[tag][ai[an]]), f, keep)
                ax.plot(lag, ac * 1e6, lw=1.2, color=ACOL[an],
                        label="ant %d" % an)
                seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
            _acf_axes(ax, seen)
            ax.set_ylabel(r"covariance [$10^{-6}$ dex$^2$]")
            ax.set_xlabel("frequency lag [MHz]"); ax.grid(alpha=.3)
            ax.legend(fontsize=8, ncol=6, loc="upper right")
        axes[0].set_title("target + east of Tee", fontsize=10, loc="left",
                          fontweight="bold")
        axes[1].set_title("north of Tee + outriggers  (note the y scale)",
                          fontsize=10, loc="left", fontweight="bold")
        fig.suptitle("3 · ACF OF THE BASELINE-SUBTRACTED SPECTRA — %s  %s\n"
                     "all antennas compared (broadband pedestal removed first)"
                     % (tag, labs[tag]),
                     fontweight="bold", fontsize=12, color=SCOL[tag])
        fig.tight_layout(rect=(0, 0, 1, .92)); pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- section 4
def _sec4_waterfall(pdf, t, f, wf, ants, wins, ai, last_test, keep):
    """Full-12 h and test-period-zoom waterfalls of the baseline-subtracted
    spectra.

    Each time bin's own broadband offset is removed before imaging. That
    offset is the common-mode pedestal (present even in RFI-free antennas);
    left in, it sets the colour scale and shows up as vertical banding
    across the whole 12 h, which swamps the narrow horizontal lines these
    panels exist to show.
    """
    hrs = (t - t[0]) / 3600.0
    x_last = (last_test - t[0]) / 3600.0
    zoom_lo = min((w[3] - t[0]) / 3600.0 for w in wins) - 0.35
    zoom_hi = x_last + 0.35
    for an in ants:
        k = ai[an]
        with np.errstate(invalid="ignore"):
            ped = np.nanmedian(np.where(keep, wf[:, k, :], np.nan),
                               axis=1)[:, None]
        img = (wf[:, k, :] - ped).T                        # (nch, nb)
        # Scale to the per-channel noise, not to a high percentile. The
        # comb lines are 8-40 sigma deep but occupy only 6 of 1208
        # channels, so a 99.5th-percentile scale is set by the strongest
        # transient instead and renders the comb almost white. 12 sigma
        # puts the comb firmly in colour; the few brightest transients
        # saturate, which is the right trade here.
        mad = np.nanmedian(np.abs(img - np.nanmedian(img)))
        sig = 1.4826 * mad
        v = 12.0 * sig if np.isfinite(sig) and sig > 0 else 1.0
        img = peak_rows(img)
        fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                                 gridspec_kw=dict(height_ratios=[1.35, 1]))
        for ax, (lo, hicut, zoomed) in zip(
                axes, [(hrs[0], hrs[-1], False), (zoom_lo, zoom_hi, True)]):
            im = ax.imshow(img, aspect="auto", origin="upper", cmap="RdBu_r",
                           vmin=-v, vmax=v, interpolation="nearest",
                           extent=[hrs[0], hrs[-1], f[-1], f[0]])
            ax.set_xlim(lo, hicut)
            for tag, lab, m, t0, t1 in wins:
                x0, x1 = (t0 - t[0]) / 3600.0, (t1 - t[0]) / 3600.0
                for x in (x0, x1):
                    ax.axvline(x, color=SCOL[tag], lw=1.4)
                if zoomed:
                    ax.annotate("%s  %s" % (tag, lab), xy=((x0 + x1) / 2, 1.005),
                                xycoords=("data", "axes fraction"),
                                color=SCOL[tag], fontsize=7, ha="left",
                                va="bottom", rotation=32, fontweight="bold")
            ax.axvline(x_last, color="k", ls="--", lw=1.1)
            ax.axhspan(HI_LO, HI_HI, color="0.6", alpha=.8)
            ax.set_ylabel("frequency [GHz]")
            fig.colorbar(im, ax=ax, label="Δ log10 S1 [dex]"
                         if zoomed else "Δ log10 S1 vs baseline [dex]")
        axes[0].annotate("left in S5 config →", xy=(x_last + .12, .03),
                         xycoords=("data", "axes fraction"), fontsize=8)
        axes[0].set_xlabel("hours since capture start "
                           "(2026-08-10 08:32 PDT / 15:32 UTC)")
        axes[1].set_xlabel("hours since capture start — ZOOM on the test period")
        fig.suptitle("4 · BASELINE-SUBTRACTED WATERFALL — antenna %d\n"
                     "blue = below baseline (line REMOVED), red = above.  each "
                     "bin's broadband offset removed, so only narrow-band "
                     "structure is shown.\nfrequency axis shrunk by "
                     "PEAK magnitude per block, so single-channel lines are "
                     "not averaged away.  grey band = HI mask" % an,
                     fontweight="bold", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .93)); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
