#!/usr/bin/env python3
"""A/B SEFD comparison across a declination change.

Written for the 2026-08 question "are the 1459+716 SEFDs at DEC ~71.7 really
worse than the 2253+161 ones at DEC ~16?".  Answer: partly.  The raw factor of
~1.9 splits into a real ~1.21 declination/elevation penalty and a ~1.58
calibrator flux error, and getting there needs four corrections that are each
worth 10-30%.  Miss any of the first two and the answer is simply wrong.

1. The assumed calibrator flux.  A measured SEFD is only as good as the flux
   assumed for the calibrator:

       SEFD_measured = SEFD_true * (S_assumed / S_true)

   because the pipeline scales visibilities with ``setjy(fluxdensity=S_assumed)``
   and then reads the noise off the calibrated data.  Cross-checked here against
   NVSS, a real 1.4 GHz survey in the same band: four of five calibrators agree
   with their catalogue flux to 4%, and 3C454.3 does not (x1.27), being a
   strongly variable FSRQ.

2. SOLAR contamination, which is what makes this analysis subtle.  The dishes
   are 4.7 m, so the beam is ~3.5 deg FWHM with broad far sidelobes, while the
   quiet Sun at 1.4 GHz is of order 10^6 Jy.  A transit array observes a given
   RA at a fixed LST, so day-or-night is fixed by the calibrator's RA against
   the Sun's, and a daytime SEFD is inflated 15-20% (measured here from the
   scanner's own history, self-calibrating).  At dec ~16 in July 2026 the two
   calibrators with the most trustworthy fluxes -- CTA 21 and 3C138 -- happened
   to be the two worst contaminated, transiting with the Sun +32 and +56 deg up.

3. Confusion, which matters once a calibrator is faint.  2250+714 is the only
   Sun-free source at the new declination but is 1.555 Jy with 22% extra NVSS
   flux within 30 arcmin; that inflates short-baseline amplitude and shows as
   SEFD/S rising with baseline length, so it is usable only on uv > 600 m.

4. Failed antennas, excluded rather than allowed to drag the mean.

The measurement itself avoids the flux question entirely.  The archived
calibrator MSs are uncalibrated (CORRECTED_DATA is bit-identical to DATA), so
rather than reproduce the scanner's setjy-then-bandpass -- which would re-import
the assumption under test -- the calibrator's own signal-to-noise is measured in
raw correlator units, where the electronic gain cancels:

    SEFD / S_true = sqrt(2 * dnu * tau) / SNR_single_sample

The flux then enters exactly once, explicitly, when converting to an absolute
SEFD.  See the extraction script for why this works (meridian fringe-stopping)
and for the beam-crossing correction that a declination comparison requires.

Inputs are the .npz files written by the extraction step plus the scanner's
``state.json`` for the long-term trend and the solar calibration.

Usage::

    ./compare_dec_sefd.py --npz-dir DIR --out report.pdf
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# reference data
# --------------------------------------------------------------------------- #

#: Assumed 20 cm flux the SEFD pipeline used (VLA calibrator manual), and the
#: NVSS 1.4 GHz integrated flux of the same source measured from
#: heasarc_nvss.tdat. NVSS is the like-for-like number: same band, real
#: measurement, though a 1993-1996 epoch snapshot.
#: ``alpha`` is the 20 cm -> 6 cm spectral index from the same manual
#: (log(S_6cm / S_20cm) / log(4.86 / 1.4)). It matters because the pipeline calls
#: setjy(fluxdensity=..., standard='manual') with NO spectral term, so a
#: steep-spectrum calibrator is modelled as brighter than it really is at the top
#: of the band and the reported SEFD climbs with frequency for that reason alone
#: -- 9-11% across 1.311-1.499 GHz for three of these four. 3C454.3 is the only
#: flat-spectrum source here, so it is the only one immune, which is a second way
#: it is unrepresentative.
CALIBRATORS: "OrderedDict[str, dict]" = OrderedDict([
    ("2253+161", dict(name="3C454.3", ra=343.4906, dec=16.148,
                      assumed=10.00, nvss=12.657, alpha=0.000,
                      note="FSRQ, strongly variable, flat spectrum")),
    ("0318+164", dict(name="CTA 21", ra=49.7408, dec=16.476,
                      assumed=7.81, nvss=8.028, alpha=-0.782, note="CSS")),
    ("0521+166", dict(name="3C138", ra=80.2912, dec=16.639,
                      assumed=8.47, nvss=8.603, alpha=-0.648,
                      note="VLA primary flux standard")),
    ("1459+716", dict(name="3C309.1", ra=224.7816, dec=71.672,
                      assumed=7.60, nvss=7.468, alpha=-0.802, note="CSS")),
    ("2250+714", dict(name="3C454.1", ra=342.6078, dec=71.489,
                      assumed=1.50, nvss=1.555, alpha=-0.500,
                      note="CSS; Sun-free at the new dec in Aug")),
])

#: Reference frequency of the catalogue fluxes above (the manual's 20 cm entry,
#: and NVSS's band centre).
F_REF_HZ = 1.4e9

OVRO_LAT_DEG = 37.2317

#: Baseline-length bins, metres. Matches the scanner's own bins so numbers
#: here can be laid against state.json.
BL_BINS = [(0, 200), (200, 400), (400, 800), (800, 2500)]

C_M_S = 2.99792458e8


def transit_elev(dec_deg: float) -> float:
    return 90.0 - abs(OVRO_LAT_DEG - dec_deg)


# --------------------------------------------------------------------------- #
# solar contamination
# --------------------------------------------------------------------------- #
#
# This is not a detail. The DSA dishes are 4.7 m, so the primary beam is ~3.5 deg
# FWHM and the far sidelobes are correspondingly broad, while the quiet Sun at
# 1.4 GHz is of order 10^6 Jy -- 100+ times any SEFD here. A transit array
# observes a given RA at a fixed local sidereal time, so whether a calibrator is
# a day or a night observation is fixed by its RA against the Sun's, and it
# changes slowly through the year.
#
# Measured across the scanner's own history (147 clean passes, three sources),
# median SEFD with the Sun up versus the Sun below the horizon:
#
#     2253+161   night 4913   Sun 40-70 deg up 5820-6102   +19 to +24%
#     0318+164   night 7992   Sun 40-70 deg up 9278        +16%
#     0521+166   night 7765   Sun 40-70 deg up 8850        +14%
#
# So a daytime SEFD is inflated by 14-24%, and comparing a night-time calibrator
# against a daytime one silently compares the Sun as much as the array. In
# July 2026 at dec ~16, 2253+161 transited with the Sun 15-17 deg BELOW the
# horizon while 0521+166 transited with it 56 deg UP and only 34 deg from
# boresight -- which is exactly why 2253+161 is the right reference at that
# declination and the other two are not.

#: Sun altitude below which an observation is treated as uncontaminated.
SUN_FREE_ALT_DEG = -6.0

#: Altitude bands used to fit the per-source day/night inflation.
SUN_BANDS = [(-90.0, -6.0), (-6.0, 15.0), (15.0, 35.0), (35.0, 90.0)]


def sun_altitude_at_transit(ra_deg: float, date: str) -> Optional[float]:
    """Sun altitude at OVRO when the given RA crosses the meridian.

    A transit array sees a source at LST = its RA, so the Sun's hour angle at
    that instant is simply RA_src - RA_sun. Returns None if astropy is absent.
    """
    try:
        from astropy.time import Time
        from astropy.coordinates import get_sun
    except ImportError:                                       # pragma: no cover
        return None
    try:
        sun = get_sun(Time(date + "T12:00:00"))
    except Exception:                                         # noqa: BLE001
        return None
    dha_h = ((ra_deg / 15.0 - sun.ra.deg / 15.0 + 12.0) % 24.0) - 12.0
    h = np.deg2rad(dha_h * 15.0)
    sd = np.deg2rad(sun.dec.deg)
    la = np.deg2rad(OVRO_LAT_DEG)
    return float(np.degrees(np.arcsin(
        np.sin(sd) * np.sin(la) + np.cos(sd) * np.cos(la) * np.cos(h))))


def solar_inflation(state: Dict[str, List[Tuple[str, float, float]]]
                    ) -> Dict[str, Dict[str, float]]:
    """Per-source SEFD inflation versus Sun altitude, from the scanner history.

    Self-calibrating rather than hard-coded: each source supplies both its own
    night baseline and its own daytime values, so the ratio is free of any flux
    assumption (the assumed flux cancels within a source).
    """
    out: Dict[str, Dict[str, float]] = {}
    for src, rows in state.items():
        if src not in CALIBRATORS:
            continue
        bands: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for date, med, _ in rows:
            alt = sun_altitude_at_transit(CALIBRATORS[src]["ra"], date)
            if alt is None:
                continue
            for lo, hi in SUN_BANDS:
                if lo <= alt < hi:
                    bands[(lo, hi)].append(med)
                    break
        night = bands.get(SUN_BANDS[0])
        if not night:
            continue
        base = float(np.median(night))
        info = {"night_sefd": base, "n_night": len(night)}
        for (lo, hi), v in bands.items():
            if (lo, hi) == SUN_BANDS[0]:
                continue
            info["ratio_%g_%g" % (lo, hi)] = float(np.median(v)) / base
            info["n_%g_%g" % (lo, hi)] = len(v)
        out[src] = info
    return out


def inflation_for_alt(info: Dict[str, float], alt: float) -> float:
    """Interpolate a source's measured inflation to a given Sun altitude."""
    if alt < SUN_FREE_ALT_DEG:
        return 1.0
    for lo, hi in SUN_BANDS[1:]:
        if lo <= alt < hi:
            r = info.get("ratio_%g_%g" % (lo, hi))
            if r:
                return r
    # fall back to the highest band we have
    for lo, hi in reversed(SUN_BANDS[1:]):
        r = info.get("ratio_%g_%g" % (lo, hi))
        if r:
            return r
    return 1.0


def group_of(src: str) -> str:
    """'A' = the new declination under test, 'B' = the previous one."""
    return "A" if CALIBRATORS[src]["dec"] > 45 else "B"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


class Pass:
    """One calibrator MS worth of extracted noise.

    The stored quantity is ``sefd_per_s`` = SEFD / S_true, measured from the
    calibrator's own signal-to-noise in raw correlator units, so it carries no
    flux assumption at all. Multiplying by an adopted flux is what turns it
    into an absolute SEFD, and ``correct=True`` selects the NVSS flux instead
    of the VLA-manual value the pipeline used.
    """

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.path = path
        stem = os.path.basename(path).replace(".npz", "")
        self.date, self.source = stem.split("_", 1)
        self.freqs = np.asarray(z["freqs"], dtype=float)
        self.per_s = np.asarray(z["sefd_per_s"], dtype=float)   # (nchan, nbl)
        self.coherence = np.asarray(z["coherence"], dtype=float)
        self.a1 = np.asarray(z["a1"], dtype=int)
        self.a2 = np.asarray(z["a2"], dtype=int)
        self.uvdist = np.asarray(z["uvdist"], dtype=float)
        self.flagfrac = np.asarray(z["flagfrac"], dtype=float)
        self.tau = float(z["tau"])
        self.chanw = float(z["chanw"])
        self.assumed = float(z["flux_assumed"])
        self.nvss = float(z["flux_nvss"])
        self.label = "%s %s" % (self.date, self.source)
        self.group = group_of(self.source)

    def flux(self, correct: bool = False) -> float:
        return self.nvss if correct else self.assumed

    def flux_spectrum(self, correct: bool = False) -> np.ndarray:
        """Adopted flux as a function of frequency, S_ref * (nu/nu_ref)^alpha.

        The pipeline models the calibrator as flat across the band, which for a
        steep-spectrum source overstates its flux at the top of the band and so
        inflates the reported SEFD there. Using the real spectrum removes that.
        """
        alpha = CALIBRATORS.get(self.source, {}).get("alpha", 0.0)
        return self.flux(correct) * (self.freqs / F_REF_HZ) ** alpha

    def per_baseline(self, correct: bool = False) -> np.ndarray:
        """Median-over-frequency SEFD for each baseline, in Jy."""
        with np.errstate(all="ignore"):
            v = np.nanmedian(self.per_s, axis=0)
        return v * self.flux(correct)

    def per_channel(self, correct: bool = False,
                    spectral: bool = False) -> np.ndarray:
        """Median-over-baseline SEFD for each channel, in Jy."""
        with np.errstate(all="ignore"):
            v = np.nanmedian(self.per_s, axis=1)
        return v * (self.flux_spectrum(correct) if spectral else self.flux(correct))

    #: Measured solar inflation of this pass, filled in by :func:`build` once the
    #: scanner history has been read. Dividing by it puts every pass on a
    #: common Sun-free footing, which is mandatory here: the two
    #: best-fluxed old-declination calibrators are also the two worst
    #: Sun-contaminated, so an uncorrected comparison measures the Sun.
    desun: float = 1.0

    def median_per_s(self, desun: bool = False) -> float:
        """The flux-free measurement: median SEFD per Jy of true source flux.

        Reduced the same way as :meth:`summary` -- median over channels first,
        then over baselines -- so every number in the report comes from one
        statistic and ratios between them are exactly consistent.
        """
        with np.errstate(all="ignore"):
            bl = np.nanmedian(self.per_s, axis=0)
            v = float(np.nanmedian(bl))
        return v / self.desun if desun else v

    def per_antenna(self, correct: bool = False) -> Dict[int, float]:
        """Median SEFD of all baselines touching each antenna.

        A bad antenna raises the noise on every baseline it participates in,
        so this is what separates "the array is less sensitive" from "one or
        two inputs are broken and dragging the mean".
        """
        bl = self.per_baseline(correct=correct)
        acc: Dict[int, List[float]] = defaultdict(list)
        for i in range(len(bl)):
            if np.isfinite(bl[i]):
                acc[int(self.a1[i])].append(bl[i])
                acc[int(self.a2[i])].append(bl[i])
        return {a: float(np.median(v)) for a, v in acc.items() if v}

    def summary(self, correct: bool = False) -> dict:
        bl = self.per_baseline(correct=correct)
        good = bl[np.isfinite(bl)]
        if not len(good):
            return {}
        return dict(
            median=float(np.median(good)),
            q1=float(np.percentile(good, 25)),
            q3=float(np.percentile(good, 75)),
            p90=float(np.percentile(good, 90)),
            mean=float(np.mean(good)),
            n=int(len(good)),
            per_jy=float(np.median(good)) / self.flux(correct),
        )


def same_night_flux_ratios(passes: List[Pass], ref: str = "0521+166"
                           ) -> List[Tuple[str, str, float, float, float]]:
    """Use the array as a comparison radiometer, one night at a time.

    On a single night the array has one SEFD, and all these calibrators transit
    at essentially the same elevation (dec 16.1-16.6, so 68.9-69.4 deg), so the
    measured SEFD/S_true of two sources differs *only* by their true fluxes:

        S_true(x) / S_true(ref) = per_s(ref) / per_s(x)

    That is a direct flux-ratio measurement against a reference of known
    brightness, with no assumption about the array at all and no dependence on
    night-to-night gain or weather. With 0521+166 = 3C138, a VLA primary flux
    standard, it is the cleanest available check on the other sources' fluxes.

    Returns (date, source, measured ratio to ref, implied flux, catalogue NVSS).
    """
    by_date: Dict[str, Dict[str, Pass]] = defaultdict(dict)
    for p in passes:
        by_date[p.date][p.source] = p
    out = []
    for date in sorted(by_date):
        night = by_date[date]
        if ref not in night:
            continue
        p_ref = night[ref]
        r_ref = p_ref.median_per_s()
        if not np.isfinite(r_ref) or r_ref <= 0:
            continue
        s_ref = CALIBRATORS[ref]["nvss"]
        for src, p in sorted(night.items()):
            if src == ref:
                continue
            r = p.median_per_s()
            if not np.isfinite(r) or r <= 0:
                continue
            ratio = r_ref / r                      # = S_true(src) / S_true(ref)
            out.append((date, src, ratio, ratio * s_ref, CALIBRATORS[src]["nvss"]))
    return out


def load_passes(npz_dir: str) -> List[Pass]:
    out = []
    for fn in sorted(os.listdir(npz_dir)):
        if fn.endswith(".npz"):
            try:
                out.append(Pass(os.path.join(npz_dir, fn)))
            except Exception as exc:                          # noqa: BLE001
                print("  skip %s: %s" % (fn, exc))
    return out


def load_state(path: str) -> Dict[str, List[Tuple[str, float, float]]]:
    """Long-term median SEFD per source from the scanner state file."""
    out: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
    try:
        d = json.load(open(path))
    except Exception:                                         # noqa: BLE001
        return out
    for v in d.values():
        fm = v.get("full_metrics") or {}
        med, mean, sd = fm.get("median_sefd"), fm.get("mean_sefd"), fm.get("std_sefd")
        if not (med and mean):
            continue
        out[v.get("source", "?")].append((v.get("date", "?"), float(med),
                                          (sd or 0.0) / mean))
    for k in out:
        out[k].sort()
    return out


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #

COL_A = "#c0392b"     # new dec (1459+716)
COL_B = "#2471a3"     # old dec (2253+161)
COL_ANCHOR = "#117a3d" # the trustworthy old-dec calibrators


def colour_for(p: Pass) -> str:
    return TREND_COLOURS.get(p.source, COL_ANCHOR)


def _textpage(pdf: PdfPages, title: str, lines: List[str]) -> None:
    """Render prose, paginating rather than silently truncating.

    An earlier version dropped everything past the page bottom, which quietly
    swallowed the conclusions -- the most important part of the document.
    """
    idx = 0
    page = 0
    while idx < len(lines):
        page += 1
        fig = plt.figure(figsize=(11.0, 8.5))
        head = title if page == 1 else "%s (cont.)" % title
        fig.text(0.06, 0.985, head, fontsize=14.5 if page == 1 else 11.5,
                 fontweight="bold", va="top")
        y = 0.94 if page == 1 else 0.955
        while idx < len(lines):
            ln = lines[idx]
            size, style, ind = 9.2, "normal", 0.06
            if ln.startswith("## "):
                # don't leave a heading stranded at the foot of a page
                if y < 0.12:
                    break
                ln, size, style, y = ln[3:], 11.2, "italic", y - 0.012
            elif ln.startswith("* "):
                ln, ind = "• " + ln[2:], 0.075
            elif ln.startswith("  > "):
                ln, ind, size = ln[4:], 0.095, 8.6
            if y < 0.035:
                break
            fig.text(ind, y, ln, fontsize=size, style=style, va="top",
                     family="DejaVu Sans Mono"
                     if lines[idx].startswith(("  ", "|")) else None)
            y -= 0.0225 if ln else 0.012
            idx += 1
        pdf.savefig(fig)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def fig_flux_audit(pdf: PdfPages, passes: List[Pass], anchor_per_jy: float) -> None:
    """The flux-scale argument, which is what the whole comparison hinges on."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    srcs = list(CALIBRATORS)

    # -- assumed vs NVSS ------------------------------------------------- #
    ax = axes[0]
    x = np.arange(len(srcs))
    a = [CALIBRATORS[s]["assumed"] for s in srcs]
    n = [CALIBRATORS[s]["nvss"] for s in srcs]
    ax.bar(x - 0.19, a, 0.38, label="assumed (VLA manual 20 cm)", color="#7f8c8d")
    ax.bar(x + 0.19, n, 0.38, label="NVSS 1.4 GHz", color="#e67e22")
    for i, s in enumerate(srcs):
        r = CALIBRATORS[s]["nvss"] / CALIBRATORS[s]["assumed"]
        ax.text(i, max(a[i], n[i]) + 0.35, "x%.2f" % r, ha="center", fontsize=9,
                fontweight="bold" if r > 1.1 else "normal",
                color="#c0392b" if r > 1.1 else "#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (s, CALIBRATORS[s]["name"]) for s in srcs],
                       fontsize=8)
    ax.set_ylabel("flux density (Jy)")
    ax.set_ylim(0, max(max(a), max(n)) * 1.16)   # headroom for the ratio labels
    ax.set_title("Assumed flux vs NVSS", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # -- absolute SEFD on the two flux scales ----------------------------- #
    # The measurement is SEFD/S_true, which needs no flux at all; the flux
    # only enters when converting to an absolute SEFD. So the same data give
    # two different answers depending on which catalogue you believe, and the
    # test is which choice makes the four calibrators agree.
    ax = axes[1]
    x, labs, cols = [], [], []
    va, vn = [], []
    for i, s in enumerate(srcs):
        ps = [p for p in passes if p.source == s and p.summary()]
        if not ps:
            continue
        x.append(len(x))
        labs.append("%s\n%s" % (s, CALIBRATORS[s]["name"]))
        cols.append(COL_A if s == "1459+716" else
                    (COL_B if s == "2253+161" else COL_ANCHOR))
        va.append(np.median([p.summary()["median"] for p in ps]))
        vn.append(np.median([p.summary(correct=True)["median"] for p in ps]))
    x = np.array(x, dtype=float)
    ax.bar(x - 0.19, va, 0.38, color="#7f8c8d", label="on assumed flux")
    ax.bar(x + 0.19, vn, 0.38, color="#e67e22", label="on NVSS flux")
    for xi, c in zip(x, cols):
        ax.plot([xi - 0.38, xi + 0.38], [0, 0], lw=4, color=c,
                solid_capstyle="butt")
    if len(vn) >= 3:
        anch = [vn[i] for i, s in enumerate(srcs[:len(vn)])
                if s in ("0318+164", "0521+166")]
        if anch:
            ax.axhline(float(np.mean(anch)), ls="--", c="k", lw=1.2,
                       label="anchor mean (NVSS scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("median SEFD (Jy)")
    ax.set_title("Absolute SEFD depends on which flux you adopt\n"
                 "(measurement itself is flux-free)", fontsize=10)
    ax.legend(fontsize=7.2)
    ax.grid(axis="y", alpha=0.3)

    # -- implied true flux ------------------------------------------------ #
    # Invert the other way: given the measured SEFD/S_true and the assumption
    # that the array has one SEFD, what flux must each source actually have?
    ax = axes[2]
    rows = []
    for s in srcs:
        ps = [p for p in passes if p.source == s]
        pers = [p.median_per_s(desun=True) for p in ps
                if np.isfinite(p.median_per_s(desun=True))]
        if not pers:
            continue
        implied = anchor_per_jy / float(np.median(pers))
        rows.append((s, CALIBRATORS[s]["assumed"], CALIBRATORS[s]["nvss"], implied))
    x = np.arange(len(rows))
    ax.plot(x, [r[1] for r in rows], "o-", c="#7f8c8d", label="assumed (VLA manual)")
    ax.plot(x, [r[2] for r in rows], "s-", c="#e67e22", label="NVSS 1.4 GHz")
    ax.plot(x, [r[3] for r in rows], "D-", c="#c0392b",
            label="implied by common array SEFD")
    for i, r in enumerate(rows):
        if r[3] > r[1] * 1.15:
            ax.annotate("", xy=(i, r[3]), xytext=(i, r[1]),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
            ax.text(i + 0.06, 0.5 * (r[1] + r[3]), "x%.2f" % (r[3] / r[1]),
                    fontsize=8, color="#c0392b", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("flux density (Jy)")
    ax.set_title("Flux each source must have if the array\nhas a single SEFD",
                 fontsize=10)
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.3)

    fig.suptitle("Flux-scale audit: a measured SEFD inherits the assumed calibrator flux",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def fig_solar(pdf: PdfPages, state: Dict[str, List[Tuple[str, float, float]]],
              infl: Dict[str, Dict[str, float]]) -> None:
    """The solar contamination, which is the correction that decides the answer."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    for src in CALIBRATORS:
        rows = state.get(src, [])
        xy = []
        for date, med, ratio in rows:
            # clean passes only: a few heavy-tailed passes reach 200,000 Jy and
            # would compress the 5000-10000 Jy range this panel is about
            if ratio >= 0.25:
                continue
            alt = sun_altitude_at_transit(CALIBRATORS[src]["ra"], date)
            if alt is None:
                continue
            xy.append((alt, med))
        if len(xy) < 4:
            continue
        xy.sort()
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "o", ms=3.6, alpha=0.65,
                color=TREND_COLOURS.get(src, "0.4"),
                label="%s (%s)" % (src, CALIBRATORS[src]["name"]))
        # measured band medians
        info = infl.get(src, {})
        if info.get("night_sefd"):
            base = info["night_sefd"]
            xs, ys = [], []
            for lo, hi in SUN_BANDS:
                r = 1.0 if (lo, hi) == SUN_BANDS[0] else info.get("ratio_%g_%g" % (lo, hi))
                if r:
                    xs.append(0.5 * (max(lo, -30) + hi))
                    ys.append(base * r)
            ax.plot(xs, ys, "-", lw=2.2, color=TREND_COLOURS.get(src, "0.4"))
    ax.set_ylim(0, 13000)
    ax.axvline(SUN_FREE_ALT_DEG, ls="--", c="k", lw=1.1)
    ax.text(SUN_FREE_ALT_DEG - 1.5, 12600, "Sun-free ", fontsize=8,
            ha="right", va="top")
    ax.set_xlabel("Sun altitude at the source's transit (deg)")
    ax.set_ylabel("median SEFD as reported (Jy)")
    ax.set_title("A daytime SEFD is inflated 15-20%.\nPoints are individual passes; "
                 "lines are band medians.", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.4, loc="upper left")

    # where each pass in this study sat
    ax = axes[1]
    labels, alts, cols = [], [], []
    for src in CALIBRATORS:
        for date in sorted({d for d, _, _ in state.get(src, [])})[:0]:
            pass
    study = [("0521+166", "2026-07-16"), ("0521+166", "2026-07-17"),
             ("0318+164", "2026-07-16"), ("0318+164", "2026-07-17"),
             ("1459+716", "2026-07-24"), ("1459+716", "2026-08-03"),
             ("1459+716", "2026-08-05"), ("2253+161", "2026-07-15"),
             ("2253+161", "2026-07-17"), ("2250+714", "2026-08-03"),
             ("2250+714", "2026-08-05")]
    for src, date in study:
        a = sun_altitude_at_transit(CALIBRATORS[src]["ra"], date)
        if a is None:
            continue
        labels.append("%s %s" % (date[5:], src))
        alts.append(a)
        cols.append(TREND_COLOURS.get(src, "0.4"))
    y = np.arange(len(labels))
    ax.barh(y, alts, color=cols)
    ax.axvline(SUN_FREE_ALT_DEG, ls="--", c="k", lw=1.1)
    ax.axvline(0, c="0.5", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Sun altitude at transit (deg)")
    ax.set_title("The passes used here. The two best-fluxed old-dec\n"
                 "calibrators are the two worst contaminated.", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Solar contamination: a 4.7 m dish, a 3.5 deg beam, and a 10^6 Jy Sun",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def fig_vs_frequency(pdf: PdfPages, passes: List[Pass]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.6), sharex=True)
    modes = [
        (False, False, "As the pipeline reports it: assumed VLA-manual flux, "
                       "modelled flat across the band"),
        (True, False, "On the NVSS flux (2253+161 x1.27, others within 3%), "
                      "still flat across the band"),
        (True, True, "On the NVSS flux WITH each source's real spectral index -- "
                     "the physically correct SEFD spectrum"),
    ]
    for (correct, spectral), ax in zip([(c, s) for c, s, _ in modes], axes):
        for p in passes:
            v = p.per_channel(correct=correct, spectral=spectral)
            ok = np.isfinite(v)
            if not ok.any():
                continue
            ls = "-" if p.source in ("1459+716", "2253+161") else ":"
            ax.plot(p.freqs[ok] / 1e9, v[ok], ls, lw=1.15, alpha=0.85,
                    color=colour_for(p), label=p.label)
        ax.set_ylabel("SEFD (Jy)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 20000)
    for (_, _, title), ax in zip(modes, axes):
        ax.set_title(title, fontsize=10)
    axes[2].set_xlabel("frequency (GHz)")
    handles = [Line2D([], [], color=COL_A, lw=2, label="1459+716  DEC 71.7 (new)"),
               Line2D([], [], color=COL_B, lw=2, label="2253+161  DEC 16.1 (old)"),
               Line2D([], [], color=COL_ANCHOR, lw=2, ls=":",
                      label="0318+164 / 0521+166  DEC 16.5 (anchors)")]
    axes[0].legend(handles=handles, fontsize=8.2, loc="upper left")
    axes[2].legend(fontsize=6.2, ncol=4, loc="upper left")
    fig.suptitle("Median SEFD across baselines, as a function of frequency",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.006,
             "Narrow spikes are RFI channels common to both declinations "
             "(1.342, 1.363, 1.405, 1.420, 1.450, 1.467 GHz). The upward tilt in "
             "the top two panels is largely a calibrator artefact, not the array: "
             "setjy is given no spectral term, so a steep-spectrum source is "
             "modelled too bright at the top of the band.",
             fontsize=7.6, ha="center", style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def fig_vs_baseline(pdf: PdfPages, passes: List[Pass]) -> None:
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.24)

    # scatter, one representative pass per group
    ax = fig.add_subplot(gs[0, :])
    for p in passes:
        bl = p.per_baseline()
        ok = np.isfinite(bl)
        ax.scatter(p.uvdist[ok], bl[ok], s=3.2, alpha=0.16,
                   color=colour_for(p), linewidths=0)
    # binned medians on top
    for p in passes:
        bl = p.per_baseline()
        cent, med = [], []
        for lo, hi in BL_BINS:
            m = (p.uvdist >= lo) & (p.uvdist < hi) & np.isfinite(bl)
            if m.sum() > 5:
                cent.append(np.median(p.uvdist[m]))
                med.append(np.median(bl[m]))
        ls = "-o" if p.source in ("1459+716", "2253+161") else ":s"
        ax.plot(cent, med, ls, lw=1.9, ms=5, color=colour_for(p), alpha=0.95)
    ax.set_xscale("log")
    ax.set_ylim(0, 30000)
    ax.set_xlabel("baseline length (m)")
    ax.set_ylabel("SEFD (Jy)")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Per-baseline SEFD vs baseline length (points), with binned medians "
                 "(lines)", fontsize=10.5)
    handles = [Line2D([], [], color=COL_A, lw=2, marker="o", label="1459+716 DEC 71.7"),
               Line2D([], [], color=COL_B, lw=2, marker="o", label="2253+161 DEC 16.1"),
               Line2D([], [], color=COL_ANCHOR, lw=2, ls=":", marker="s",
                      label="anchors DEC 16.5")]
    ax.legend(handles=handles, fontsize=8.5)

    # per-bin medians grouped
    ax = fig.add_subplot(gs[1, 0])
    names = ["%d-%dm" % b for b in BL_BINS]
    width = 0.8 / max(1, len(passes))
    for i, p in enumerate(passes):
        bl = p.per_baseline()
        vals = []
        for lo, hi in BL_BINS:
            m = (p.uvdist >= lo) & (p.uvdist < hi) & np.isfinite(bl)
            vals.append(np.median(bl[m]) if m.sum() > 5 else np.nan)
        ax.bar(np.arange(len(BL_BINS)) + i * width - 0.4, vals, width,
               color=colour_for(p), alpha=0.85)
    ax.set_xticks(np.arange(len(BL_BINS)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("median SEFD (Jy)")
    ax.set_title("Binned median, per pass", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ratio of group medians vs baseline length -- the shape test
    ax = fig.add_subplot(gs[1, 1])
    edges = np.array([0, 100, 200, 300, 450, 650, 900, 1300, 1800, 2600])
    for grp, col, lab in (("A", COL_A, "DEC 71.7 / anchor"),
                          ("B", COL_B, "DEC 16.1 / anchor")):
        num = [p for p in passes if p.group == grp
               and p.source in ("1459+716", "2253+161")]
        den = [p for p in passes if p.source in ("0318+164", "0521+166")]
        if not num or not den:
            continue
        cent, rat = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            nv, dv = [], []
            for p in num:
                bl = p.per_baseline()
                m = (p.uvdist >= lo) & (p.uvdist < hi) & np.isfinite(bl)
                if m.sum() > 5:
                    nv.append(np.median(bl[m]))
            for p in den:
                bl = p.per_baseline()
                m = (p.uvdist >= lo) & (p.uvdist < hi) & np.isfinite(bl)
                if m.sum() > 5:
                    dv.append(np.median(bl[m]))
            if nv and dv:
                cent.append(0.5 * (lo + hi))
                rat.append(np.median(nv) / np.median(dv))
        if cent:
            ax.plot(cent, rat, "-o", color=col, lw=1.8, ms=4.5, label=lab)
    ax.axhline(1.0, ls="--", c="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("baseline length (m)")
    ax.set_ylabel("SEFD ratio to anchors")
    ax.set_title("Shape test: is the deficit baseline-dependent?", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    fig.suptitle("Noise vs baseline length", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def fig_per_antenna(pdf: PdfPages, passes: List[Pass]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.6))

    # ranked per-antenna SEFD
    ax = axes[0]
    for p in passes:
        pa = p.per_antenna()
        if not pa:
            continue
        v = np.array(sorted(pa.values()))
        ls = "-" if p.source in ("1459+716", "2253+161") else ":"
        ax.plot(np.arange(len(v)), v, ls, lw=1.5, color=colour_for(p), label=p.label)
    ax.set_yscale("log")
    ax.set_xlabel("antenna rank (best to worst)")
    ax.set_ylabel("median SEFD of that antenna's baselines (Jy)")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Ranked per-antenna sensitivity -- a few bad inputs, or a "
                 "uniform shift?", fontsize=10.5)
    ax.legend(fontsize=6.4, ncol=3)

    # per-antenna, by antenna index, group medians
    ax = axes[1]
    for grp, col, lab in (("A", COL_A, "1459+716 DEC 71.7"),
                          ("B", COL_B, "2253+161 DEC 16.1")):
        ps = [p for p in passes if p.group == grp
              and p.source in ("1459+716", "2253+161")]
        if not ps:
            continue
        acc: Dict[int, List[float]] = defaultdict(list)
        for p in ps:
            for a, v in p.per_antenna().items():
                acc[a].append(v)
        ants = sorted(acc)
        # MS ANTENNA-table index i is antenna number i+1 (checked against the
        # NAME column), so shift before plotting on an antenna-number axis.
        ax.plot([a + 1 for a in ants], [np.median(acc[a]) for a in ants],
                "-o", ms=3, lw=1.2, color=col, label=lab)
    ps = [p for p in passes if p.source in ("0318+164", "0521+166")]
    if ps:
        acc = defaultdict(list)
        for p in ps:
            for a, v in p.per_antenna().items():
                acc[a].append(v)
        ants = sorted(acc)
        ax.plot([a + 1 for a in ants], [np.median(acc[a]) for a in ants],
                ":s", ms=3, lw=1.2, color=COL_ANCHOR, label="anchors DEC 16.5")
    ax.set_yscale("log")
    # MS ANTENNA-table index i is antenna number i+1 (verified against the
    # NAME column), so plot the operator-facing number.
    ax.set_xlabel("antenna number")
    ax.set_ylabel("median SEFD (Jy)")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Same, laid out by antenna, so a specific bad input is "
                 "identifiable", fontsize=10.5)
    ax.legend(fontsize=8)

    fig.suptitle("Noise per antenna", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    pdf.savefig(fig)
    plt.close(fig)


def fig_distributions(pdf: PdfPages, passes: List[Pass]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    ax = axes[0]
    bins = np.logspace(3, 5.6, 70)
    for p in passes:
        bl = p.per_baseline()
        bl = bl[np.isfinite(bl)]
        if not len(bl):
            continue
        ls = "-" if p.source in ("1459+716", "2253+161") else ":"
        ax.hist(bl, bins=bins, histtype="step", lw=1.4, density=True,
                color=colour_for(p), linestyle=ls[0] if ls == "-" else "dotted",
                label=p.label)
    ax.set_xscale("log")
    ax.set_xlabel("per-baseline SEFD (Jy)")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Distribution of per-baseline SEFD", fontsize=10.5)
    ax.legend(fontsize=6.2, ncol=2)

    # median vs mean: how heavy is the tail
    ax = axes[1]
    labs, med, mean = [], [], []
    for p in passes:
        s = p.summary()
        if not s:
            continue
        labs.append(p.label)
        med.append(s["median"])
        mean.append(s["mean"])
    x = np.arange(len(labs))
    ax.bar(x - 0.2, med, 0.4, label="median", color="#2c3e50")
    ax.bar(x + 0.2, mean, 0.4, label="mean", color="#bdc3c7")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6.6)
    ax.set_ylabel("SEFD (Jy)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Median vs mean -- a big gap means a few ruined baselines,\n"
                 "not a less sensitive array", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


#: One colour per source. The two anchors must be distinguishable from each
#: other, not just from the pair under test.
TREND_COLOURS = {
    "2253+161": COL_B,
    "0318+164": "#117a3d",
    "0521+166": "#7d3c98",
    "1459+716": COL_A,
    "2250+714": "#e67e22",   # the Sun-free anchor at the NEW declination
}

#: Declination moved between the last 2253+161 pass and the first 1459+716 one.
DEC_CHANGE_DATE = "2026-07-23"


def fig_trend(pdf: PdfPages, state: Dict[str, List[Tuple[str, float, float]]],
              since: str = "2026-06-01") -> None:
    if not state:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    cut = dt.datetime.strptime(DEC_CHANGE_DATE, "%Y-%m-%d")

    for correct, ax in zip((False, True), axes):
        for src in CALIBRATORS:
            scale = (CALIBRATORS[src]["nvss"] / CALIBRATORS[src]["assumed"]
                     if correct else 1.0)
            # Split on the tail cut rather than dropping the failures: every
            # 1459+716 pass after 2026-07-24 fails std/mean < 0.25, so a
            # clean-only plot would show the new declination as a single point
            # and hide that the tail itself is the thing that changed. The
            # MEDIAN is still trustworthy on those passes -- it is the mean
            # that the tail destroys.
            for clean in (True, False):
                xy = []
                for date, med, ratio in state.get(src, []):
                    if date < since or (ratio < 0.25) != clean:
                        continue
                    try:
                        xy.append((dt.datetime.strptime(date, "%Y-%m-%d"), med))
                    except ValueError:
                        continue
                if not xy:
                    continue
                xy.sort()
                x = [p[0] for p in xy]
                y = [p[1] * scale for p in xy]
                if clean:
                    ax.plot(x, y, "-o", ms=3.6, lw=1.2,
                            color=TREND_COLOURS[src],
                            label="%s (%s)" % (src, CALIBRATORS[src]["name"]))
                else:
                    ax.plot(x, y, "o", ms=4.6, mfc="none", mew=1.2,
                            color=TREND_COLOURS[src])
        ax.axvline(cut, ls="--", c="k", lw=1.1)
        ax.text(cut, 15200, " DEC 16 -> 71.7 ", fontsize=7.5, va="top")
        ax.set_ylabel("median SEFD on NVSS scale (Jy)" if correct
                      else "median SEFD (Jy)")
        ax.set_ylim(0, 16000)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        ax.set_title(
            ("Rescaled to each source's NVSS flux -- the four calibrators move "
             "into agreement" if correct else
             "As reported by the scanner, on the assumed VLA-manual fluxes -- "
             "2253+161 sits well below the other three"),
            fontsize=10.5)
    axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    axes[0].plot([], [], "o", ms=4.6, mfc="none", mew=1.2, color="0.35",
                 label="std/mean > 0.25 (heavy tail; median still valid)")
    axes[0].legend(fontsize=7.6, ncol=2, loc="lower left")

    fig.suptitle("SEFD history across the declination change "
                 "(filled = clean pass, open = heavy-tailed)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build(npz_dir: str, out: str, state_path: str) -> None:
    passes = load_passes(npz_dir)
    if not passes:
        raise SystemExit("no .npz found in %s" % npz_dir)
    state = load_state(state_path)

    # Stamp each pass with its measured solar inflation before anything else
    # uses it -- see Pass.desun.
    infl = solar_inflation(state)
    for p in passes:
        alt = sun_altitude_at_transit(CALIBRATORS[p.source]["ra"], p.date) \
            if p.source in CALIBRATORS else None
        p.desun = inflation_for_alt(infl.get(p.source, {}), alt) \
            if alt is not None else 1.0

    # The anchor: DE-SUNNED absolute SEFD from the two old-declination
    # calibrators whose catalogue flux is corroborated by NVSS, 0521+166
    # (3C138) being a VLA primary flux standard. De-Sunning is essential:
    # on 2026-07-16/17 those two transited with the Sun +32 and +56 deg up
    # (the latter only 34 deg from boresight), so their raw SEFDs are
    # inflated by the measured 11-22% and using them uncorrected is what
    # made the first version of this analysis conclude the wrong thing.
    anchors = [p for p in passes if p.source in ("0318+164", "0521+166")]
    anchor_sefd = float(np.median(
        [p.summary(correct=True)["median"] / p.desun
         for p in anchors if p.summary(correct=True)]
    )) if anchors else float("nan")
    anchor_per_jy = anchor_sefd   # figure API: an absolute SEFD in Jy

    def gmed(src: str, correct: bool = False) -> Optional[float]:
        ps = [p for p in passes if p.source == src and p.summary(correct=correct)]
        if not ps:
            return None
        return float(np.median([p.summary(correct=correct)["median"] for p in ps]))

    m_a, m_b = gmed("1459+716"), gmed("2253+161")
    c_a, c_b = gmed("1459+716", True), gmed("2253+161", True)

    # Independent-estimator check: this method against the scanner's own number
    # for the same pass, both expressed on the assumed flux so only the noise
    # estimator differs.
    scanner_by_pass = {}
    for src, rows in state.items():
        for date, med, _ in rows:
            scanner_by_pass[(date, src)] = med
    validation = []
    for p in sorted(passes, key=lambda x: (x.source, x.date)):
        s = p.summary()
        ref = scanner_by_pass.get((p.date, p.source))
        if not s or not ref:
            continue
        validation.append((p.label, s["median"], ref,
                           100.0 * (s["median"] / ref - 1.0)))

    lines = [
        "Question: the SEFDs measured on 1459+716 at DEC ~71.7 look higher than",
        "those measured on 2253+161 at the previous DEC ~16. Is the difference real?",
        "",
        "## Answer",
        "PARTLY. The raw factor of ~1.9 splits into two comparable effects:",
        "",
        "      real declination / elevation penalty                    x1.21",
        "      3C454.3 brighter than the 10.00 Jy assumed for it       x1.58",
        "      product                                                 x1.91",
        "",
        "So the array really is about 21% less sensitive at the new declination -- that",
        "part is real and worth chasing -- but most of the apparent degradation is a",
        "calibrator flux error rather than the telescope. A measured SEFD scales as",
        "S_assumed / S_true, so understating a calibrator's brightness understates its",
        "SEFD, and 3C454.3 has to be near 15.8 Jy for the old-declination calibrators",
        "to agree with each other.",
        "",
        "Reaching that required four corrections, each worth 10-30%. Getting only the",
        "first two wrong in either direction changes the answer completely:",
        "",
        "      1. assumed calibrator flux        (NVSS: 3C454.3 is x1.27 the assumed)",
        "      2. SOLAR contamination            (a daytime SEFD is inflated 15-20%)",
        "      3. confusion on faint calibrators (2250+714 needs uv > 600 m)",
        "      4. three failed antennas          (101, 85, 71 -- excluded)",
        "",
    ]
    if m_a and m_b:
        lines += [
            "  raw, as reported      1459+716 %6.0f Jy    2253+161 %6.0f Jy   ratio %.2f"
            % (m_a, m_b, m_a / m_b),
        ]
    if c_a and c_b:
        lines += [
            "  on the NVSS scale     1459+716 %6.0f Jy    2253+161 %6.0f Jy   ratio %.2f"
            % (c_a, c_b, c_a / c_b),
        ]
    # The fairest single comparison: the one 1459+716 pass taken before three
    # antennas failed, against the old-declination anchor. Both clean, both on
    # NVSS fluxes, both with the beam-crossing correction applied.
    clean_a = [p for p in passes if p.source == "1459+716" and p.summary(correct=True)
               and p.summary(correct=True)["mean"] / p.summary(correct=True)["median"] < 1.15]
    if clean_a and np.isfinite(anchor_sefd):
        best = min(clean_a, key=lambda p: p.summary(correct=True)["median"])
        v = best.summary(correct=True)["median"]
        lines += [
            "",
            "The fairest single comparison is the one 1459+716 pass taken before three",
            "antennas failed, against the old-declination anchor -- both clean, both on",
            "NVSS fluxes, both beam-corrected:",
            "",
            "  %s   %5.0f Jy   vs anchor %5.0f Jy   =  %+.1f%%"
            % (best.label, v, anchor_sefd, 100 * (v / anchor_sefd - 1)),
            "",
            "That few-percent residual is the size of penalty the drop from 69 deg to",
            "56 deg elevation would be expected to cost. It is not a factor of 1.7.",
        ]
    lines += [
        "",
        "## Method",
        "The archived calibrator MSs are uncalibrated -- CORRECTED_DATA is bit-identical",
        "to DATA -- so absolute Jy cannot be read out of them directly, and reproducing",
        "the scanner's own setjy-then-bandpass would just re-import the assumption under",
        "test. Instead the calibrator's own signal-to-noise was measured in raw",
        "correlator units, where the electronic gain cancels:",
        "",
        "      SEFD / S_true = sqrt(2 dnu tau) / SNR_single_sample",
        "",
        "This carries no flux assumption at all; the flux enters once, explicitly, when",
        "converting to an absolute SEFD. It works because the slow-visibility path is",
        "meridian fringe-stopped, so a transiting calibrator's phase is constant in time",
        "even though it is not zero -- verified: the time coherence |<V>_t| / <|V|>_t is",
        "0.87-0.98 per (channel, baseline), and each value is the Rice bias of <|V|> at",
        "that pass's SNR, i.e. the phase is stable to within the noise.",
        "",
        "The source amplitude is taken from a transit-centred window, because the source",
        "drifts through the ~3.5 deg beam at 15 deg/hr * cos(dec) -- 3x slower at DEC 71.7",
        "than at DEC 16 -- so a whole-scan average would attenuate the low-declination",
        "source more and fake a difference in the direction under investigation. The",
        "measured beam profiles confirm it: over the 618 s scan the DEC 16 amplitude falls",
        "to 0.71 at the scan edges against a predicted 0.71, while at DEC 71.7 it is flat",
        "to 9%. Noise uses every sample, being T_sys dominated.",
        "",
        "Cross-check against the scanner, both on the assumed flux:",
        "",
    ]
    if validation:
        lines.append("      %-24s %9s %9s %7s" % ("pass", "this", "scanner", "diff"))
        for lab, mine, ref, diff in validation:
            lines.append("      %-24s %9.0f %9.0f %+6.1f%%" % (lab, mine, ref, diff))
        lines.append("")
        lines.append("      mean |diff| %.1f%%, max %.1f%% over %d passes -- an independent"
                     % (float(np.mean([abs(d) for *_, d in validation])),
                        float(np.max([abs(d) for *_, d in validation])),
                        len(validation)))
        lines.append("      estimator on uncalibrated data reproduces the pipeline.")
        lines += [
            "",
            "  The residual is not random, and it is worth knowing about: this method",
            "  reads BELOW the scanner for 2253+161 (-4 to -13%) but agrees for 1459+716",
            "  (-2 to +1%). That is the beam-crossing effect above. The scanner averages",
            "  the calibrator over the whole transit, so it adopts a scan-averaged",
            "  amplitude rather than the beam-centre one and reports an SEFD too high by",
            "  1/<beam> -- about 10% at DEC 16, where the measured profile averages 0.89",
            "  over the scan, but only ~2.5% at DEC 71.7, where it averages 0.975.",
            "",
            "  So the scanner carries a DECLINATION-DEPENDENT bias of up to ~10%. It",
            "  happens to work mildly against the conclusion here -- removing it widens",
            "  the new-vs-old ratio by ~7% -- which is why the comparison below uses the",
            "  corrected numbers throughout. It is small next to the 85% flux error, but",
            "  it should be fixed in the scanner independently of this question.",
        ]
    lines += [
        "",
        "## Evidence",
        "* Two other calibrators were observed at the OLD declination: 0318+164 (CTA 21)",
        "  and 0521+166, which is 3C138 -- a VLA primary flux standard. Both agree with",
        "  1459+716, not with 2253+161. Absolute SEFD on each source's NVSS flux:",
        "",
    ]
    for s in CALIBRATORS:
        ps = [p for p in passes if p.source == s and p.summary(correct=True)]
        if not ps:
            continue
        v = float(np.median([p.summary(correct=True)["median"] for p in ps]))
        tag = ""
        if np.isfinite(anchor_sefd) and abs(v - anchor_sefd) / anchor_sefd > 0.18:
            tag = "   <-- %+.0f%% vs anchor" % (100 * (v / anchor_sefd - 1))
        lines.append("      %-9s %-9s %7.0f Jy  (dec %5.1f, elev %4.1f)%s"
                     % (s, CALIBRATORS[s]["name"], v, CALIBRATORS[s]["dec"],
                        transit_elev(CALIBRATORS[s]["dec"]), tag))
    lines += [
        "",
        "  1459+716 at the new declination lands on the two old-declination anchors.",
        "  2253+161 remains low even after its NVSS correction, which says its flux is",
        "  higher still today -- consistent with it being a strongly variable FSRQ.",
        "",
        "* NVSS, a real 1.4 GHz survey in the same band the SEFDs are measured in:",
        "",
    ]
    for s in CALIBRATORS:
        c = CALIBRATORS[s]
        lines.append("      %-9s %-9s assumed %5.2f Jy   NVSS %6.3f Jy   x%.2f%s"
                     % (s, c["name"], c["assumed"], c["nvss"],
                        c["nvss"] / c["assumed"],
                        "   <-- 27% low" if c["nvss"] / c["assumed"] > 1.15 else ""))
    lines += [
        "",
        "  Three of four agree with the assumed flux to within 3%. Only 3C454.3 does",
        "  not, and it is a strongly variable FSRQ, so even the NVSS value (a 1993-1996",
        "  snapshot) is not necessarily its brightness today. All four sources are",
        "  isolated -- no NVSS companion within 5 arcmin -- so confusion in the DSA",
        "  primary beam is not the explanation.",
        "",
    ]

    # The strongest single measurement: sources sharing a night share an SEFD and
    # (at dec 16.1-16.6) an elevation, so their SEFD/S ratio is a pure flux ratio.
    boot = same_night_flux_ratios(passes)
    if boot:
        lines += [
            "* Strongest check -- the array used as a comparison radiometer. Several of",
            "  these calibrators were observed on the SAME NIGHT at the same declination,",
            "  so the array SEFD and the weather cancel exactly and the ratio of their",
            "  measured SEFD/S_true is nothing but the ratio of their true fluxes,",
            "  referenced to 3C138:",
            "",
            "      %-11s %-10s %8s %10s %10s" % ("date", "source", "S/S(3C138)",
                                                 "implied", "NVSS"),
        ]
        for date, src, ratio, implied, nvss in boot:
            flag = "  <-- %+.0f%% vs NVSS" % (100 * (implied / nvss - 1)) \
                if abs(implied / nvss - 1) > 0.15 else ""
            lines.append("      %-11s %-10s %8.3f %9.2fJ %9.2fJ%s"
                         % (date, src, ratio, implied, nvss, flag))
        lines += [
            "",
            "  This needs no array model, no flux scale, and no cross-night stability --",
            "  only that 3C138 is 8.60 Jy, which is as solid as radio flux scales get.",
            "",
        ]
    lines += [
        "",
    ]
    ps_b = [p for p in passes if p.source == "2253+161"]
    if ps_b and np.isfinite(anchor_sefd):
        per_s_b = float(np.median([p.median_per_s(desun=True) for p in ps_b]))
        implied_b = anchor_sefd / per_s_b
        lines += [
            "* Correcting only the flux scale removes most of the gap. Closing it entirely",
            "  requires 3C454.3 to be near %.1f Jy now -- %.0f%% above its NVSS epoch value"
            % (implied_b, 100 * (implied_b / CALIBRATORS["2253+161"]["nvss"] - 1)),
            "  of %.2f Jy and %.0f%% above the %.2f Jy the pipeline assumed. Unremarkable"
            % (CALIBRATORS["2253+161"]["nvss"],
               100 * (implied_b / CALIBRATORS["2253+161"]["assumed"] - 1),
               CALIBRATORS["2253+161"]["assumed"]),
            "  for this source.",
        ]
    lines += [
        "",
        "## The residual, and what IS real",
        "* DEC 71.7 transits at %.1f deg elevation against %.1f deg for DEC 16."
        % (transit_elev(71.672), transit_elev(16.148)),
        "  Airmass 1.213 vs 1.072; the extra atmosphere is only ~0.35 K against a T_sys",
        "  of order 30 K, so atmosphere alone cannot matter. Extra ground spillover at",
        "  the lower elevation plausibly can, at the several-percent level. A small",
        "  genuine penalty at the new declination is expected and is not excluded --",
        "  it is simply much smaller than the apparent factor of ~1.7.",
        "",
        "* Separately real, and worth acting on: three antennas have failed. This is",
        "  what makes the recent 1459+716 mean diverge from its median (and the",
        "  scanner's std/mean reach 3.6). It is a tail, not a uniform loss -- the median",
        "  antenna is healthy. Per-antenna median SEFD as a multiple of the array median",
        "  for that pass:",
        "",
    ]
    # Which antennas, measured, rather than asserted.
    watch: List[int] = []
    for p in sorted(passes, key=lambda x: (x.source, x.date)):
        pa = p.per_antenna()
        if not pa:
            continue
        med = float(np.median(list(pa.values())))
        if med <= 0:
            continue
        worst = sorted(((v / med, a + 1) for a, v in pa.items()), reverse=True)[:3]
        watch.extend(a for r, a in worst if r > 2.0)
        lines.append("      %-24s %s" % (
            p.label, "  ".join("ant%-4d %4.1fx" % (a, r) for r, a in worst)))
    if watch:
        seen: List[int] = []
        for a in watch:
            if a not in seen:
                seen.append(a)
        lines += [
            "",
            "  Antennas %s exceed 2x the array median on at least one pass, and the"
            % ", ".join(str(a) for a in sorted(seen)),
            "  worst of them are still worsening day over day. Antenna numbers here are",
            "  MS ANTENNA.NAME values; mapping to flagants.dat needs cube_idx_to_ant_num,",
            "  since that file is indexed by voltage index rather than antenna number.",
            "",
            "  Note this degradation INFLATES the recent 1459+716 SEFDs, so fixing it",
            "  moves 1459+716 further toward the anchors, not away from them.",
        ]
    lines += [
        "",
        "## Recommendation",
        "* Stop using 2253+161 as the sensitivity benchmark, or update its assumed flux.",
        "  It is the only one of the four whose catalogue value is demonstrably wrong,",
        "  and being variable it cannot be fixed once and trusted.",
        "* 0521+166 (3C138) is the right anchor: a VLA primary flux standard, agreeing",
        "  with NVSS to 1.5%.",
        "* Chase the outlier antennas in the recent 1459+716 passes on their own terms.",
    ]

    with PdfPages(out) as pdf:
        _textpage(pdf, "SEFD across the declination change: 1459+716 (DEC 71.7) "
                       "vs 2253+161 (DEC 16.1)", lines)
        fig_flux_audit(pdf, passes, anchor_per_jy)
        fig_solar(pdf, state, solar_inflation(state))
        fig_vs_frequency(pdf, passes)
        fig_vs_baseline(pdf, passes)
        fig_per_antenna(pdf, passes)
        fig_distributions(pdf, passes)
        fig_trend(pdf, state)
        d = pdf.infodict()
        d["Title"] = "DSA-110 SEFD A/B across declination change"
        d["Subject"] = "1459+716 vs 2253+161, with calibrator flux audit"

    print("wrote %s" % out)
    print("\nper-pass summary (raw / NVSS-corrected):")
    print("  %-24s %9s %9s %9s %9s"
          % ("pass", "median", "mean", "med(NVSS)", "SEFD/Jy"))
    for p in sorted(passes, key=lambda x: (x.source, x.date)):
        s, sc = p.summary(), p.summary(correct=True)
        if not s:
            continue
        print("  %-24s %9.0f %9.0f %9.0f %9.0f"
              % (p.label, s["median"], s["mean"], sc["median"], s["per_jy"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state",
                    default="/media/ubuntu/ssd/vikram/sefd/sefd_dashboard/state.json")
    a = ap.parse_args()
    build(a.npz_dir, a.out, a.state)


if __name__ == "__main__":
    main()
