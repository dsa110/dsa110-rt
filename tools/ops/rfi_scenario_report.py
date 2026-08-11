#!/usr/bin/env python3
"""Narrow-band RFI report for the 2026-08-10 on-site scenario tests.

Plots only, by request.

Method
------
The quantity plotted throughout is a NARROW-BAND RESIDUAL:

    resid(f) = log10 S1(f) - medfilt(log10 S1(f), MEDFILT_CH)

Subtracting a running median removes the smooth bandpass, the
antenna-to-antenna gain differences, and any broad structure, leaving only
features narrower than a couple of MHz. That is what makes an antenna
comparison meaningful at all: raw S1 differs by orders of magnitude between
antennas, so a raw overlay shows gain, not interference.

MEDFILT_CH = 21 channels (2.56 MHz), deliberately much narrower than the
10 MHz comb spacing under investigation -- a ~10 MHz median window would
partially absorb a 10 MHz-spaced comb and hide the very thing being looked
for.

The median is taken WITHIN each corr node's 96-channel block, never across
a block boundary. The 1536-channel axis is 16 independently-calibrated node
bandpasses concatenated; a kernel straddling a boundary would drag the step
between two nodes into the residual and manufacture a false line at each of
the 16 seams.

1.400-1.440 GHz is masked everywhere (Galactic HI corrupts the baseline
there), plus a 1 MHz guard at each mask and band edge, since the running
median is one-sided at an edge and produces a spurious line there.

Two statistics are used, and it matters which:

* The ACF of resid(f) in frequency lag measures PERIODICITY -- a comb of
  equally-spaced lines peaks at multiples of its spacing. It is shown as a
  COVARIANCE (dex^2), not normalised to 1 at zero lag: a normalised ACF
  makes pure noise in a clean antenna look identical to a strong comb in a
  dirty one. It is evaluated separately on the two contiguous sub-bands
  either side of the HI mask and combined, and divided by the number of
  overlapping channels at each lag so the triangular taper does not
  suppress the very peaks being looked for.

* The comb amplitude is the mean residual on the comb grid channels, split
  into the strong (20 MHz) and weak (interleaved 10 MHz) sets, with an
  OFF-COMB CONTROL: every channel more than 4 channels from any grid point.
  The control is what makes the comparison trustworthy -- it absorbs any
  common-mode drift between the baseline and a test window, so a change in
  the comb is only believable if the control does not move with it.

Errors are the standard error over the 60 s time bins in each window, so a
20-minute scenario is compared against the pre-test baseline on equal
footing.

The baseline is PRE-test only. After the last window the array was
deliberately left in the S5 configuration for optical pointing, so the
post-test hours are shown separately as a persistence check, never folded
into the baseline.
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

MEDFILT_CH = 21
NCH_NODE = 96
HI_LO, HI_HI = 1.400, 1.440
#: A grid point this close to a mask or band edge is dropped. The running
#: median is one-sided there, which drives the residual to identically zero
#: and would silently dilute the comb average with a dead channel.
EDGE_GUARD_GHZ = 0.003
PDT_OFFSET_H = 7.0                        # PDT = UTC-7
COMB_STEP = 0.010                         # candidate comb spacing [GHz]
COMB_ANCHOR = 1.330                       # strong lines sit on 20 MHz from here

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

#: 50/51 are the antennas under investigation; the rest are references.
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
    naive = dt.datetime(d.year, d.month, d.day, h, m, tzinfo=dt.timezone.utc)
    return naive.timestamp() + PDT_OFFSET_H * 3600.0


def narrowband(x: np.ndarray) -> np.ndarray:
    """log10 S1 minus a running median taken within each node block."""
    out = np.full_like(x, np.nan)
    for g in range(x.size // NCH_NODE):
        sl = slice(g * NCH_NODE, (g + 1) * NCH_NODE)
        blk = x[sl]
        ok = np.isfinite(blk)
        if ok.sum() < NCH_NODE // 2:
            continue
        filled = np.where(ok, blk, np.nanmedian(blk[ok]))
        sm = median_filter(filled, size=MEDFILT_CH, mode="nearest")
        out[sl] = np.where(ok, blk - sm, np.nan)
    return out


def comb_channels(f, keep):
    """Strong (20 MHz), weak (interleaved), and off-comb control channels."""
    grid = np.arange(1.310, 1.4951, COMB_STEP)
    strong, weak, allc = [], [], []
    for g in grid:
        c = int(np.argmin(np.abs(f - g)))
        allc.append(c)
        if not keep[c]:
            continue
        # a grid point sitting on a mask or band edge picks up the
        # one-sided running median, not a line -- drop it
        if min(abs(g - HI_LO), abs(g - HI_HI),
               abs(g - f[0]), abs(g - f[-1])) < EDGE_GUARD_GHZ:
            continue
        (strong if round((g - COMB_ANCHOR) * 1e3) % 20 == 0 else weak).append(c)
    off = keep.copy()
    for c in allc:
        off[max(0, c - 4):c + 5] = False
    return np.array(strong), np.array(weak), np.where(off)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.reduced)
    spec = z["spec"]                       # (nb, nant, 1536, 2)
    t = z["t_bin"]
    ants = [int(v) for v in z["ants"]]
    f = production_freq_axis_GHz()
    ai = {n: i for i, n in enumerate(ants)}
    nbins, nant, nch, npol = spec.shape

    hi = (f >= HI_LO) & (f <= HI_HI)
    keep = ~hi
    cs, cw, coff = comb_channels(f, keep)

    wins = []
    for tag, lab, s, e in SCENARIOS:
        t0, t1 = pdt_to_unix(t[0], s), pdt_to_unix(t[0], e)
        wins.append((tag, lab, (t >= t0) & (t < t1), t0, t1))
    first_test = min(w[3] for w in wins)
    last_test = max(w[4] for w in wins)
    base_m = t < first_test
    post_m = t >= last_test

    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log10(np.where(spec > 0, spec, np.nan))
    resid = np.full_like(logs, np.nan)
    for b in range(nbins):
        for k in range(nant):
            for p in range(npol):
                resid[b, k, :, p] = narrowband(logs[b, k, :, p])
    resid[:, :, hi, :] = np.nan

    with np.errstate(invalid="ignore"):
        tot = np.nanmean(logs[:, :, keep, :], axis=2)          # (nb,nant,npol)
        # comb amplitude per time bin, per antenna
        amp = {"strong": np.nanmean(resid[:, :, cs, :], axis=(2, 3)),
               "weak": np.nanmean(resid[:, :, cw, :], axis=(2, 3)),
               "off": np.nanmean(resid[:, :, coff, :], axis=(2, 3))}

    def wmean(m):
        with np.errstate(invalid="ignore"):
            return np.nanmean(resid[m], axis=0)

    base = wmean(base_m)
    scen = {tag: wmean(m) for tag, lab, m, _, _ in wins}
    post = wmean(post_m)

    allw = ([("baseline", base_m)] + [(w[0], w[2]) for w in wins]
            + [("post-test", post_m)])
    print("bins: " + " ".join("%s=%d" % (n, m.sum()) for n, m in allw))
    print("comb channels: strong=%d weak=%d off-control=%d"
          % (cs.size, cw.size, coff.size))

    with PdfPages(a.out) as pdf:
        _p_comb_time(pdf, t, amp, ants, wins, ai, base_m, last_test)
        _p_comb_bars(pdf, amp, ants, allw, ai)
        _p_acf_all(pdf, f, keep, base, ants, ai)
        _p_acf_scen(pdf, f, keep, base, scen, post, wins, ai)
        _p_grid_stem(pdf, f, keep, base, scen, post, wins, ants, ai, cs, cw)
        _p_zoom(pdf, f, base, scen, wins, ai)
        _p_spectra(pdf, f, keep, base, scen, wins, ai)
        _p_difference(pdf, f, keep, base, scen, wins, ai)
        _p_total_power(pdf, t, tot, ants, wins, ai, base_m)
        _p_waterfall(pdf, t, f, resid, wins, ai)
    print("wrote %s (%.1f MiB)" % (a.out, os.path.getsize(a.out) / 2**20))
    return 0


def _shade(ax, t, wins, label=True):
    for tag, lab, m, t0, t1 in wins:
        ax.axvspan((t0 - t[0]) / 3600.0, (t1 - t[0]) / 3600.0,
                   color=SCOL[tag], alpha=.22, zorder=0)
        if label:
            ax.annotate(tag, xy=(((t0 + t1) / 2 - t[0]) / 3600.0, 1.0),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=8, color=SCOL[tag],
                        fontweight="bold")


def _p_comb_time(pdf, t, amp, ants, wins, ai, base_m, last_test):
    hrs = (t - t[0]) / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, key, ttl in zip(
            axes, ("strong", "weak", "off"),
            ("comb, STRONG lines (20 MHz spacing)",
             "comb, WEAK interleaved lines (the intermediate 10 MHz points)",
             "OFF-COMB CONTROL — all channels >4 ch from any grid point")):
        for an in ants:
            y = amp[key][:, ai[an]]
            # mask non-positive rather than clipping: on a log axis a clip
            # turns ant 114's noise excursions into spikes down to the floor
            # that cover every other trace.
            ax.plot(hrs, np.where(y > 0, y, np.nan),
                    lw=(1.6 if an in (50, 51) else .8),
                    color=("0.6" if an == 114 else ACOL[an]),
                    zorder=(3 if an in (50, 51) else 2),
                    label="ant %d" % an)
        ax.set_yscale("log"); ax.set_ylim(2e-4, 1.0)
        ax.set_ylabel("mean residual [dex]")
        ax.grid(alpha=.3, which="both")
        ax.set_title(ttl, fontsize=10, loc="left", fontweight="bold")
        _shade(ax, t, wins, label=(ax is axes[0]))
        ax.axvline((last_test - t[0]) / 3600.0, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=7, ncol=10, loc="lower left")
    axes[-1].annotate("← baseline (pre-test)", xy=(0.2, 0.06),
                      xycoords=("data", "axes fraction"), fontsize=8)
    axes[-1].annotate("array left in S5 config →  (persistence check, "
                      "not a baseline)",
                      xy=((last_test - t[0]) / 3600.0 + .1, 0.06),
                      xycoords=("data", "axes fraction"), fontsize=8)
    axes[-1].set_xlabel("hours since capture start "
                        "(2026-08-10 08:32 PDT / 15:32 UTC)")
    fig.suptitle("Comb amplitude through the 12 h capture\n"
                 "shaded = on-site test configurations   —   log scale",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95)); pdf.savefig(fig); plt.close(fig)


def _p_comb_bars(pdf, amp, ants, allw, ai):
    names = [n for n, _ in allw]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (gname, glist) in zip(axes.ravel(), GROUPS.items()):
        w = .8 / (len(glist) * 2)
        for gi, an in enumerate(glist):
            for si, (key, hatch) in enumerate((("strong", ""), ("weak", "///"))):
                mu = [np.nanmean(amp[key][m, ai[an]]) for _, m in allw]
                er = [np.nanstd(amp[key][m, ai[an]])
                      / max(np.sqrt(m.sum()), 1) for _, m in allw]
                ax.bar(x + (2 * gi + si) * w - .4 + w / 2, mu, w, yerr=er,
                       color=ACOL[an], alpha=(1.0 if si == 0 else .45),
                       hatch=hatch, edgecolor="k", linewidth=.4,
                       error_kw=dict(lw=.8),
                       label="ant %d %s" % (an, key))
        ctl = [np.nanmean(amp["off"][m, ai[glist[0]]]) for _, m in allw]
        ax.plot(x, ctl, "k_", ms=22, mew=2, label="off-comb control")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1.0)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("mean residual [dex]"); ax.grid(alpha=.3, axis="y",
                                                      which="both")
        ax.legend(fontsize=6.5, ncol=2, loc="upper right")
        ax.set_title(gname, fontsize=10, loc="left", fontweight="bold")
    fig.suptitle("Comb amplitude per configuration, with off-comb control\n"
                 "solid = strong 20 MHz lines, hatched = weak interleaved "
                 "lines, black dash = control  (error bars = SEM over 60 s bins)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93)); pdf.savefig(fig); plt.close(fig)


def _cov_acf(y, f, keep, max_lag_MHz=30.0):
    """Covariance in frequency lag, unbiased for overlap, HI mask excluded."""
    dfMHz = abs(np.diff(f)).mean() * 1e3
    n = int(max_lag_MHz / dfMHz)
    idx = np.where(keep)[0]
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
        ac = ac / (seg.size - np.arange(n))       # unbiased for overlap
        acc += ac * seg.size; w += seg.size
    return np.arange(n) * dfMHz, (acc / w if w else acc)


#: The ACF is plotted from here on, never from zero: the lag-0 spike is
#: just the total residual variance and is 10-20x the comb peaks, so
#: including it flattens the structure the panel exists to show.
ACF_LAG_MIN = 2.0


def _acf_axes(ax, curves):
    for mm in np.arange(10, 31, 10):
        ax.axvline(mm, color="k", ls="--", lw=1.0, alpha=.55)
    for mm in np.arange(5, 31, 10):
        ax.axvline(mm, color="k", ls=":", lw=.7, alpha=.35)
    ax.set_xlim(ACF_LAG_MIN, 30)
    if curves:
        v = np.concatenate(curves)
        v = v[np.isfinite(v)]
        if v.size:
            lo, hicut = v.min(), v.max()
            pad = .12 * max(hicut - lo, 1e-9)
            ax.set_ylim(lo - pad, hicut + pad)


def _p_acf_all(pdf, f, keep, base, ants, ai):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, sub in zip(axes, ([50, 51, 1, 2], [100, 102, 108, 114, 115, 116])):
        seen = []
        for an in sub:
            lag, ac = _cov_acf(np.nanmean(base[ai[an]], axis=-1), f, keep)
            ax.plot(lag, ac * 1e6, lw=1.2, color=ACOL[an], label="ant %d" % an)
            seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
        _acf_axes(ax, seen)
        ax.set_ylabel(r"covariance [$10^{-6}$ dex$^2$]")
        ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=6, loc="upper right")
        ax.set_xlabel("frequency lag [MHz]")
    axes[0].set_title("target + east of Tee", fontsize=10, loc="left",
                      fontweight="bold")
    axes[1].set_title("north of Tee + outriggers  (note the y scale)",
                      fontsize=10, loc="left", fontweight="bold")
    fig.suptitle("COMB TEST, baseline — autocorrelation of the narrow-band "
                 "residual\nplotted as covariance, so height reflects real "
                 "line power (dashed 10/20/30 MHz, dotted 5/15/25 MHz)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93)); pdf.savefig(fig); plt.close(fig)


def _p_acf_scen(pdf, f, keep, base, scen, post, wins, ai):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, an in zip(axes.ravel(), (50, 51, 1, 2)):
        k = ai[an]
        seen = []
        lag, ac = _cov_acf(np.nanmean(base[k], axis=-1), f, keep)
        ax.plot(lag, ac * 1e6, lw=3.0, color="k", label="baseline", zorder=2)
        seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
        for tag, lab, m, _, _ in wins:
            lag, ac = _cov_acf(np.nanmean(scen[tag][k], axis=-1), f, keep)
            # S4 is the one window that reproduces the baseline, so it is
            # drawn dashed and on top -- a solid line of equal width would
            # simply be hidden under the baseline where they agree.
            ax.plot(lag, ac * 1e6, lw=(1.6 if tag == "S4" else 1.0),
                    ls=("--" if tag == "S4" else "-"),
                    color=SCOL[tag], label="%s %s" % (tag, lab),
                    zorder=(5 if tag == "S4" else 3))
            seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
        lag, ac = _cov_acf(np.nanmean(post[k], axis=-1), f, keep)
        ax.plot(lag, ac * 1e6, lw=1.0, color="0.5", ls="--", label="post-test")
        seen.append((ac * 1e6)[lag >= ACF_LAG_MIN])
        _acf_axes(ax, seen)
        ax.set_ylabel(r"covariance [$10^{-6}$ dex$^2$]")
        ax.set_xlabel("frequency lag [MHz]"); ax.grid(alpha=.3)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.set_title("ant %d" % an, fontsize=11, loc="left", fontweight="bold")
    fig.suptitle("COMB TEST per configuration — does the periodicity survive?",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)


def _p_grid_stem(pdf, f, keep, base, scen, post, wins, ants, ai, cs, cw):
    cols = [("baseline", base, "k")] + \
           [(w[0], scen[w[0]], SCOL[w[0]]) for w in wins] + \
           [("post-test", post, "0.5")]
    for gname, glist in GROUPS.items():
        fig, axes = plt.subplots(len(glist), 1, squeeze=False, sharex=True,
                                 figsize=(14, 2.5 * len(glist) + 1.3))
        for ax, an in zip(axes[:, 0], glist):
            k = ai[an]
            w = .85 / len(cols)
            for i, (nm, arr, col) in enumerate(cols):
                y = np.nanmean(arr[k], axis=-1)
                for j, c in enumerate(np.concatenate([cs, cw])):
                    ax.bar(f[c] * 1e3 + (i - len(cols) / 2) * 0.9, y[c],
                           0.85, color=col, edgecolor="none",
                           label=(nm if j == 0 else None))
            for c in cs:
                ax.axvline(f[c] * 1e3, color="0.8", lw=.6, zorder=0)
            ax.axvspan(HI_LO * 1e3, HI_HI * 1e3, color="0.88", zorder=0)
            ax.set_ylabel("ant %d\n[dex]" % an); ax.grid(alpha=.25, axis="y")
            ax.axhline(0, color="k", lw=.6)
        axes[0, 0].legend(fontsize=7, ncol=8, loc="upper left")
        axes[-1, 0].set_xlabel("frequency [MHz]  —  bars at the 10 MHz comb "
                               "grid; grey band = HI mask")
        fig.suptitle("Residual on the 10 MHz comb grid — %s\n"
                     "tall bars every 20 MHz, short bars on the interleaved "
                     "10 MHz points" % gname, fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .93)); pdf.savefig(fig); plt.close(fig)


def _p_zoom(pdf, f, base, scen, wins, ai):
    bands = [(1.325, 1.375), (1.445, 1.495)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for row, an in enumerate((50, 51)):
        for col, (lo, hicut) in enumerate(bands):
            ax = axes[row, col]
            k = ai[an]
            sel = (f >= lo) & (f <= hicut)
            ax.plot(f[sel] * 1e3, np.nanmean(base[k], axis=-1)[sel], lw=2.4,
                    color="k", label="baseline", zorder=2)
            labs = dict((w[0], w[1]) for w in wins)
            for tag, ls, lw in (("S4", "--", 1.4), ("S5", "-", 1.0)):
                ax.plot(f[sel] * 1e3, np.nanmean(scen[tag][k], axis=-1)[sel],
                        lw=lw, ls=ls, color=SCOL[tag], zorder=5,
                        label="%s %s" % (tag, labs[tag]))
            for g in np.arange(np.ceil(lo * 100) / 100, hicut, 0.010):
                ax.axvline(g * 1e3, color="0.85", lw=.7, zorder=0)
            ax.set_xlabel("frequency [MHz]"); ax.set_ylabel("residual [dex]")
            ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper right")
            ax.set_title("ant %d   %.3f-%.3f GHz" % (an, lo, hicut),
                         fontsize=10, loc="left", fontweight="bold")
    fig.suptitle("Zoom on the comb — minex ON (S4) reproduces the baseline "
                 "lines, minex OFF (S5) removes them\n"
                 "grey verticals = 10 MHz grid", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93)); pdf.savefig(fig); plt.close(fig)


def _p_spectra(pdf, f, keep, base, scen, wins, ai):
    for gname, glist in GROUPS.items():
        fig, axes = plt.subplots(len(glist), 1, sharex=True, squeeze=False,
                                 figsize=(14, 2.9 * len(glist) + 1.4))
        for ax, an in zip(axes[:, 0], glist):
            k = ai[an]
            y = np.nanmean(base[k], axis=-1)
            step = 4.0 * np.nanstd(y[keep])
            ax.plot(f[keep], y[keep], lw=.55, color="k", label="baseline")
            for i, (tag, lab, m, _, _) in enumerate(wins):
                ys = np.nanmean(scen[tag][k], axis=-1)
                ax.plot(f[keep], ys[keep] + step * (i + 1), lw=.55,
                        color=SCOL[tag], label="%s  %s" % (tag, lab))
            ax.axvspan(HI_LO, HI_HI, color="0.88", zorder=0)
            ax.set_ylabel("ant %d\nresidual [dex]" % an); ax.grid(alpha=.25)
        axes[0, 0].legend(fontsize=6.5, ncol=3, loc="upper left")
        axes[-1, 0].set_xlabel("frequency [GHz]   —   grey = HI mask, excluded")
        fig.suptitle("Narrow-band residual spectra — %s\n"
                     "(offset vertically per scenario)" % gname,
                     fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)


def _p_difference(pdf, f, keep, base, scen, wins, ai):
    for tag, lab, m, _, _ in wins:
        fig, axes = plt.subplots(len(GROUPS), 1, figsize=(14, 10.5), sharex=True)
        for ax, (gname, glist) in zip(axes, GROUPS.items()):
            for an in glist:
                k = ai[an]
                d = np.nanmean(scen[tag][k] - base[k], axis=-1)
                ax.plot(f[keep], d[keep], lw=.6, color=ACOL[an],
                        label="ant %d" % an)
            ax.axhline(0, color="k", lw=.7)
            ax.axvspan(HI_LO, HI_HI, color="0.88", zorder=0)
            ax.set_ylabel("Δ residual\n[dex]"); ax.grid(alpha=.25)
            ax.legend(fontsize=7, ncol=4, loc="upper right")
            ax.set_title(gname, fontsize=9, loc="left", fontweight="bold")
        axes[-1].set_xlabel("frequency [GHz]   —   grey = HI mask, excluded")
        fig.suptitle("%s  %s   minus baseline\n"
                     "negative = a line present in the baseline that this "
                     "configuration removed" % (tag, lab),
                     fontweight="bold", fontsize=12, color=SCOL[tag])
        fig.tight_layout(rect=(0, 0, 1, .93)); pdf.savefig(fig); plt.close(fig)


def _p_total_power(pdf, t, tot, ants, wins, ai, base_m):
    hrs = (t - t[0]) / 3600.0
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for an in ants:
        y = np.nanmean(tot[:, ai[an], :], axis=1)
        ax.plot(hrs, y - np.nanmedian(y[base_m]), lw=.9, color=ACOL[an],
                label="ant %d" % an)
    _shade(ax, t, wins)
    ax.set_ylabel("total in-band power [dex, rel. baseline]")
    ax.set_xlabel("hours since capture start")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=10, loc="lower right")
    fig.suptitle("Total in-band power — sanity check that no antenna lost "
                 "signal during a test\n(a comb that vanishes because the "
                 "antenna went dead would show up here)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .90)); pdf.savefig(fig); plt.close(fig)


def _p_waterfall(pdf, t, f, resid, wins, ai):
    hrs = (t - t[0]) / 3600.0
    for an in (50, 51, 1, 100, 114):
        if an not in ai:
            continue
        k = ai[an]
        with np.errstate(invalid="ignore"):
            img = np.nanmean(resid[:, k, :, :], axis=-1).T
        fig, ax = plt.subplots(figsize=(14, 5.5))
        v = np.nanpercentile(np.abs(img), 99.5)
        im = ax.imshow(img, aspect="auto", origin="upper", cmap="magma",
                       vmin=0, vmax=(v if np.isfinite(v) and v > 0 else 1),
                       extent=[hrs[0], hrs[-1], f[-1], f[0]])
        for tag, lab, m, t0, t1 in wins:
            for x in ((t0 - t[0]) / 3600.0, (t1 - t[0]) / 3600.0):
                ax.axvline(x, color="c", lw=.8)
            ax.annotate(tag, xy=(((t0 + t1) / 2 - t[0]) / 3600.0, 1.0),
                        xycoords=("data", "axes fraction"), color="k",
                        fontsize=8, ha="center", va="bottom", fontweight="bold")
        ax.axhspan(HI_LO, HI_HI, color="c", alpha=.35)
        fig.colorbar(im, ax=ax, label="narrow-band residual [dex]")
        ax.set_xlabel("hours since capture start")
        ax.set_ylabel("frequency [GHz]")
        ax.set_title("Antenna %d — narrow-band residual, full 12 h "
                     "(cyan band = HI mask)" % an, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
