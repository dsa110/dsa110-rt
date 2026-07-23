"""Candidate-card compositor for Slack notifications (best-effort, pure).

Consolidates the handful of per-event plots that already exist on disk after
a C2/C3 KEEP plus the key numbers from ``Level3/<name>.json`` ("c2" dict)
and the voltage-retention report into a single readable PNG, so a human
scanning Slack can judge FRB-vs-RFI without opening several separate files.

Style: plain white background, serif font, a dense compact metadata block
up top and an evenly-sized 2x2 plot grid — deliberately closer to the
legacy T3 ``filplot()`` figure
(``dsa110-T3/dsaT3/filplot_funcs.py``, ``plotfour()`` ~lines 90-365: serif
rcParams, ``fig.suptitle`` metadata line, tight ``constrained_layout``
grid) than to a "designed dashboard" look.

Two data domains are kept visually distinct — they must never be
interleaved:

* **cubes** — search-node real-time imaging products: ``image_peak``,
  ``lightcurve``, ``dm_time``, ``kernel_snrs`` (all under
  ``Level2/plots/``). Available immediately at C3 KEEP time. Panel
  ORDER AND GRID mirror the dashboard's own burst page
  (``tools/dashboard/dsa_monitor/templates/burst_event.html``, the
  "Plots" card): plots are listed alphabetically by filename in a 2-column
  CSS grid, i.e. row-major order dm_time, image_peak, kernel_snrs,
  lightcurve — reproduced here as an explicit 2x2 grid, top row
  (dm_time | image_peak), bottom row (kernel_snrs | lightcurve), all four
  panels drawn at the same size (no panel is enlarged/dominant).
* **voltages** — the bbproc coherent-beam re-processing of the dumped
  voltages (``filterbank/<name>.png``), which typically lands later (or
  not at all, if the dump/bbproc step failed or was skipped).

``mode="cubes"`` (default) renders only the 2x2 cubes grid — this is the
card posted seconds after the C3 decision, so it cannot depend on
voltages/filterbank existing. ``mode="full"`` appends the bbproc
voltages panel below the cubes grid, under its own plain-text section
label and a thin rule (not a colored bar — kept in the same plain/dense
style as the rest of the card).

Pure function, no network, no writes outside ``out_path``. Every failure
mode (missing plot, corrupt PNG, missing/garbled Level3 fields) is caught
and turned into either a labelled "not available" placeholder panel or an
``ok=False`` return — this must never raise into the caller (``c3.py`` /
``slack_notify.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOG = logging.getLogger("dsart.services.candidate_card")

#: unix epoch expressed as MJD (same constant used in slack_notify.py /
#: c3.py / cal_hdf5_archive.py for mjd -> unix conversions).
_MJD_UNIX_EPOCH = 40587.0

#: Search-cube sample period, in ms, for converting the c2 dict's
#: ``width_median``/``width_*`` (in cube samples) to physical time.
#: Authoritative value: the production search-node cadence
#: ``t_int_search_us = 1048.576`` us (32x the corr native 32.768us tick),
#: pinned in ``configs/dsart_search_rt.yaml`` (e.g. lines 221/309,
#: ``--t-int-search-us 1048.576``) and ``configs/operating_points.yaml``
#: line 76. This is the same constant the C2 plotter falls back to for a
#: cube's ``sample_period_us`` when the NPZ doesn't carry one explicitly
#: (``src/dsart/coinc/plotter.py:649``), so it is the correct per-sample
#: duration for widths measured in cube samples.
_SEARCH_SAMPLE_PERIOD_MS = 1048.576e-3

#: (panel-name, filename-template, title) for the four cube panels, in the
#: exact 2x2 row-major order used by the dashboard's burst page (see
#: module docstring): top-left, top-right, bottom-left, bottom-right.
_CUBE_PANELS = (
    ("dm_time", "dm_time_{name}.png", "DM–time"),
    ("image_peak", "image_peak_{name}.png", "Sky image at peak (l, m)"),
    ("kernel_snrs", "kernel_snrs_{name}.png", "Per-kernel significance"),
    ("lightcurve", "lightcurve_{name}.png", "Light curve"),
)

#: uniform panel-title font size (aesthetics: consistent across the card,
#: regardless of the panel's on-page size).
_PANEL_TITLE_FONTSIZE = 12.5

_BG = "#ffffff"
_FG = "#000000"
_MUTED = "#555555"
_RULE = "#000000"

#: legacy filplot_funcs.py rcParams this card's rc_context mirrors (serif,
#: dense, publication-style — see module docstring).
_RC = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": _PANEL_TITLE_FONTSIZE,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#000000",
    "text.color": _FG,
    "axes.labelcolor": _FG,
    "xtick.color": _FG,
    "ytick.color": _FG,
}

# ---------------------------------------------------------------------------
# RA/Dec + galactic (l, b) derivation
#
# Mirrors tools/dashboard/dsa_monitor/event_astrometry.py's "modern schema"
# recipe: apparent (l, m, LST, pointing-dec) -> TETE -> ICRS, then ICRS ->
# galactic via astropy's SkyCoord.galactic. Deliberately NOT imported from
# there — dashboard code must not be a dependency of src/dsart (dependency
# direction: dsart -> nothing in tools/dashboard). The astrometry math is
# small and well-cited, so it is duplicated here rather than shared.
#
# Site position: latitude reuses dsart's own
# ``dsart.common.constants.PHI_LAT_OVRO_DEG`` (37.234 deg); longitude and
# height match tools/dashboard/dsa_monitor/sky_astrometry.py
# (OVRO_LON_DEG=-118.283, height=1222.0 m) since dsart has no longitude
# constant of its own yet.
#
# Position uncertainty: checked (read-only) how the live dashboard shows
# this. ``event_astrometry.py`` defines a fixed
# ``SIGMA_POS_ARCSEC = 60.0`` ("half-pixel image quantization ~22.5 arcsec
# + end-to-end astrometric tie validated at ~1.2 arcmin on pulsar
# crossings") for any position it computed itself (sources "level3",
# "manual", "filterbank" — i.e. anything using this same l/m/pointing-dec
# recipe); legacy stored (pre-2026-07) positions get no uncertainty at all
# ("unknown"). ``templates/burst_event.html`` renders this literally as
# "± 60 arcsec (1 sigma, conservative)" next to the sexagesimal RA/Dec.
# Since this card's positions are always freshly computed via the
# identical recipe (never a legacy passthrough), we use that same fixed
# ±60″ figure rather than inventing a pixel-scale-derived number.
# ---------------------------------------------------------------------------

_OVRO_LON_DEG: float = -118.283
_OVRO_HEIGHT_M: float = 1222.0

#: see the derivation note above — matches
#: tools/dashboard/dsa_monitor/event_astrometry.py's SIGMA_POS_ARCSEC.
_SIGMA_POS_ARCSEC: float = 60.0


def _ovro_lat_deg() -> float:
    try:
        from dsart.common.constants import PHI_LAT_OVRO_DEG
        return float(PHI_LAT_OVRO_DEG)
    except Exception:  # noqa: BLE001 — fall back to the literal
        return 37.234


@dataclass(frozen=True)
class _Position:
    ra_deg: float
    dec_deg: float
    l_gal_deg: float
    b_gal_deg: float


def _compute_position(c2row: Mapping[str, Any]) -> Optional[_Position]:
    """ICRS (ra, dec) + galactic (l, b), all in degrees, from an event's c2
    dict — or None if inputs are missing/non-finite or astropy/the
    transform is unavailable. Never raises."""
    try:
        mjd = float(c2row.get("t_peak_mjd"))
        l_rad = float(c2row.get("l_median"))
        m_rad = float(c2row.get("m_median"))
        pointing_dec_deg = float(c2row.get("pointing_dec_deg"))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v)
               for v in (mjd, l_rad, m_rad, pointing_dec_deg)):
        return None

    try:
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        from astropy.utils import iers
        import astropy.units as u

        # Pin IERS to offline/stale mode before any Time math — astropy
        # raises on network-unavailable auto-download otherwise. A stale
        # predictive UT1-UTC is well below the ~1 arcmin accuracy this
        # recipe targets (mirrors event_astrometry.py._ensure_iers).
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None

        site = EarthLocation(
            lat=_ovro_lat_deg() * u.deg, lon=_OVRO_LON_DEG * u.deg,
            height=_OVRO_HEIGHT_M * u.m,
        )
        t = Time(mjd, format="mjd", scale="utc", location=site)
        lst_deg = float(t.sidereal_time("apparent").to_value(u.deg))

        dec_app = pointing_dec_deg + math.degrees(m_rad)
        cos_dec = math.cos(math.radians(dec_app))
        if abs(cos_dec) < 1e-9:
            return None
        ra_app = lst_deg + math.degrees(l_rad) / cos_dec

        coord = SkyCoord(
            ra=ra_app * u.deg, dec=dec_app * u.deg,
            frame="tete", obstime=t,
        )
        icrs = coord.icrs
        ra_deg = float(icrs.ra.to_value(u.deg)) % 360.0
        dec_deg = float(icrs.dec.to_value(u.deg))
        if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
            return None

        gal = icrs.galactic
        l_gal_deg = float(gal.l.to_value(u.deg)) % 360.0
        b_gal_deg = float(gal.b.to_value(u.deg))
        if not (math.isfinite(l_gal_deg) and math.isfinite(b_gal_deg)):
            return None

        return _Position(ra_deg, dec_deg, l_gal_deg, b_gal_deg)
    except Exception as exc:  # noqa: BLE001 — astropy missing/transform fail
        LOG.warning("candidate_card: position compute failed: %s", exc)
        return None


def _format_ra_hms_colon(ra_deg: float) -> str:
    """"HH:MM:SS.s" — sexagesimal hours, one decimal on seconds."""
    hours = (ra_deg % 360.0) / 15.0
    total_tenths = int(round(hours * 3600.0 * 10.0))
    total_tenths %= 24 * 3600 * 10
    hh, rem = divmod(total_tenths, 3600 * 10)
    mm, rem = divmod(rem, 600)
    ss = rem / 10.0
    return f"{hh:02d}:{mm:02d}:{ss:04.1f}"


def _format_dec_dms_colon(dec_deg: float) -> str:
    """"+DD:MM:SS.s" — sexagesimal degrees, one decimal on arcseconds."""
    sign = "+" if dec_deg >= 0 else "-"
    total_tenths = int(round(abs(dec_deg) * 3600.0 * 10.0))
    dd, rem = divmod(total_tenths, 3600 * 10)
    mm, rem = divmod(rem, 600)
    ss = rem / 10.0
    return f"{sign}{dd:02d}:{mm:02d}:{ss:04.1f}"


def _position_header_str(c2row: Mapping[str, Any]) -> str:
    """"RA hh:mm:ss.s(±60″)  Dec +dd:mm:ss.s(±60″)   l=NNN.N°  b=±NN.N°",
    or an "unavailable" line if the position couldn't be computed."""
    pos = _compute_position(c2row)
    if pos is None:
        return "RA/Dec/Galactic: unavailable"
    sigma = int(round(_SIGMA_POS_ARCSEC))
    return (
        f"RA {_format_ra_hms_colon(pos.ra_deg)}(±{sigma}″)   "
        f"Dec {_format_dec_dms_colon(pos.dec_deg)}(±{sigma}″)   "
        f"l={pos.l_gal_deg:.1f}°   b={pos.b_gal_deg:+.1f}°"
    )


# ---------------------------------------------------------------------------
# misc formatting
# ---------------------------------------------------------------------------


def _mjd_to_utc_str(mjd: Any) -> Optional[str]:
    try:
        from datetime import datetime, timezone
        unix_s = (float(mjd) - _MJD_UNIX_EPOCH) * 86400.0
        return datetime.fromtimestamp(
            unix_s, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _fmt(value: Any, fmt: str = "{}") -> str:
    if value is None:
        return "n/a"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _width_str(width_samp: Any) -> str:
    """"X ms (N samp)" — width_samp converted via
    ``_SEARCH_SAMPLE_PERIOD_MS`` (see its derivation comment), samples kept
    in the bracket for cross-checking against the raw c2 field. "n/a" if
    width_samp is missing/non-numeric."""
    try:
        samp = float(width_samp)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(samp):
        return "n/a"
    ms = samp * _SEARCH_SAMPLE_PERIOD_MS
    return f"{ms:.1f} ms ({samp:.0f} samp)"


def _resolve_detection_dm(ev_dir: Path, name: str) -> Optional[float]:
    """DM (pc cm⁻³) of the peak C1 detection (max snr) for this event.

    Reads ``Level2/C1_window_<name>.csv`` via
    ``dsart.coinc.plotter._read_window_csv_rows`` /
    ``_peak_from_csv_rows`` — the SAME resolution the plotter uses to
    place the crosshair/reticle on the dm_time, image_peak, and
    lightcurve panels (see ``plotter._resolve_burst``'s CSV path). The
    header should therefore quote that same DM, not the c2 dict's
    ``dm_median`` (a cluster-wide median across every member, which can
    read substantially different from the one detection the σ in this
    header and the panel reticles are keyed to — e.g. 169.6 vs 127.8 for
    260723zmtr). Returns ``None`` if the CSV is missing/unparsable; the
    caller then falls back to ``c2row["dm_median"]`` under an explicit
    "DM(cluster med)" label so the header never silently mislabels a
    cluster statistic as a detection value.
    """
    try:
        from dsart.coinc.plotter import (
            _peak_from_csv_rows, _read_window_csv_rows,
        )
        # _read_window_csv_rows takes an *archive root* and appends
        # ``<event_name>/Level2/C1_window_<event_name>.csv`` itself (see
        # plotter.py's ``render_event_plots``/``regenerate_recent_events``
        # callers, which pass ``job.archive_root``) — but this module's
        # ``ev_dir`` IS the event directory already (``ev_dir ==
        # archive_root / name``, see ``_render_card``'s panel-path
        # construction just above). Pass the parent so the two path
        # conventions line up.
        rows = _read_window_csv_rows(Path(ev_dir).parent, name)
        peak = _peak_from_csv_rows(rows)
    except Exception as exc:  # noqa: BLE001 — must never raise into the card
        LOG.warning(
            "candidate_card: detection-DM resolution failed for %s: %s",
            name, exc,
        )
        return None
    return float(peak.dm_pc_cc) if peak is not None else None


def _line1_str(
    c2row: Mapping[str, Any],
    dm_value: Optional[float] = None,
    dm_label: str = "DM",
) -> str:
    """"UTC ...    MJD NNNNN.NNNNN    σ=...    DM=... pc cm⁻³    width=...".

    ``dm_value``/``dm_label`` let the caller override which DM is quoted
    and how it's labelled (see :func:`_resolve_detection_dm`); when
    omitted this falls back to ``c2row["dm_median"]`` labelled "DM" (the
    pre-v3.6 behaviour), which is also what every existing caller/test
    that doesn't pass these gets.
    """
    utc = _mjd_to_utc_str(c2row.get("t_peak_mjd"))
    mjd_str = _fmt(c2row.get("t_peak_mjd"), "{:.5f}")
    snr = c2row.get("snr_max")
    dm = dm_value if dm_value is not None else c2row.get("dm_median")
    width = c2row.get("width_median")
    return (
        f"UTC {utc or 'unknown'}    "
        f"MJD {mjd_str}    "
        f"σ={_fmt(snr, '{:.1f}')}    "
        f"{dm_label}={_fmt(dm, '{:.1f}')} pc cm⁻³    "
        f"width={_width_str(width)}"
    )


def _line2_str(
    c2row: Mapping[str, Any], keep_report: Mapping[str, Any],
) -> str:
    """"(l, m)=...    pointing dec=...    gal DM(max LOS)=...    voltage
    fragments=..." — ``gal_dm_max_los_pc_cc`` is the same Level3 "c2"
    field the dashboard's Level3-metadata JSON dump shows verbatim (see
    ``tools/dashboard/dsa_monitor/templates/burst_event.html`` — it has
    no dedicated widget/label for this field, just the raw pretty-printed
    JSON, so there is no distinct dashboard label to mirror here beyond
    the field's own name)."""
    l_val = c2row.get("l_median")
    m_val = c2row.get("m_median")
    pointing_dec = c2row.get("pointing_dec_deg")
    gal_dm = c2row.get("gal_dm_max_los_pc_cc")

    n_present = keep_report.get("n_fragments_present")
    n_total = keep_report.get("n_fragments_total")
    frag_str = (
        f"{n_present}/{n_total}"
        if n_present is not None and n_total is not None else "n/a")

    lm_str = (
        f"({float(l_val):.5f}, {float(m_val):.5f}) rad"
        if l_val is not None and m_val is not None else "n/a")

    return (
        f"(l, m)={lm_str}    "
        f"pointing dec={_fmt(pointing_dec, '{:.2f}')}°    "
        f"gal DM(max LOS)={_fmt(gal_dm, '{:.1f}')} pc cm⁻³    "
        f"voltage fragments={frag_str}"
    )


# ---------------------------------------------------------------------------
# panel loading / drawing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Panel:
    """One image panel destined for the grid: either a loaded array, or a
    placeholder to draw instead."""
    name: str
    title: str
    image: Any = None          # numpy array from matplotlib.image.imread
    placeholder: Optional[str] = None


def _load_panel(name: str, title: str, path: Path) -> _Panel:
    if not path.is_file():
        return _Panel(name=name, title=title,
                       placeholder=f"{title}\n(not available)")
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(str(path))
        return _Panel(name=name, title=title, image=img)
    except Exception as exc:  # noqa: BLE001 — corrupt/unreadable PNG
        LOG.warning("candidate_card: failed to read %s: %s", path, exc)
        return _Panel(name=name, title=title,
                      placeholder=f"{title}\n(unreadable)")


def _draw_panel(ax: Any, panel: _Panel) -> None:
    """Draw one image panel. No section title of our own is drawn here —
    the archived source PNGs already carry their own embedded
    ``ax.set_title`` (see ``coinc/plotter.py``), so a second title here
    would just duplicate it and burn vertical space (v3.4: removed the
    former ``_draw_title_strip`` layer entirely — see git history for
    the prior title-strip approach and why it existed)."""
    ax.set_facecolor(_BG)
    if panel.image is not None:
        # v3.7: aspect="auto" stretches the source PNG to fill its axes
        # box exactly, so both panels of a row share identical width AND
        # height with edges aligned (user requirement: per-row uniform
        # extents). Row heights are the mean of the row-mates' native
        # aspect heights, so the residual stretch per panel is mild.
        ax.set_anchor("N")
        ax.imshow(panel.image, aspect="auto")
    else:
        # Missing/unreadable source PNG: this placeholder text is the
        # only thing identifying the panel in that case (no embedded
        # title exists to fall back on), so it keeps the panel's name.
        ax.text(
            0.5, 0.5, panel.placeholder or "not available",
            ha="center", va="center", fontsize=11, color=_MUTED,
            transform=ax.transAxes, wrap=True,
        )
    ax.axis("off")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def render_card(
    ev_dir: Path,
    name: str,
    c2row: Mapping[str, Any],
    out_path: Path,
    keep_report: Optional[Mapping[str, Any]] = None,
    mode: str = "cubes",
) -> Dict[str, Any]:
    """Compose the candidate card PNG. Always returns a status dict, never
    raises.

    Parameters
    ----------
    mode:
        ``"cubes"`` (default) — only the 2x2 grid of search-node cube
        panels (dm_time, image_peak, kernel_snrs, lightcurve — dashboard
        panel order, see module docstring); the filterbank waterfall is
        not loaded/attempted at all (no placeholder for it). This is the
        fast card posted seconds after the C3 decision.
        ``"full"`` — the same 2x2 cubes grid, with the bbproc
        voltages/filterbank waterfall appended below under its own
        plain-text section label, for later/manual comparison.

    Returns
    -------
    dict with keys ``ok`` (bool), ``path`` (str or None), ``error``
    (str or None), ``panels`` (list of panel names actually included, i.e.
    an image was found and loaded — placeholders are not counted).
    """
    try:
        return _render_card(ev_dir, name, c2row or {}, out_path,
                             keep_report or {}, mode)
    except Exception as exc:  # noqa: BLE001 — must never raise
        LOG.exception("candidate_card %s: unexpected failure", name)
        return {"ok": False, "path": None,
                "error": f"{type(exc).__name__}: {exc}", "panels": []}


def _render_card(
    ev_dir: Path,
    name: str,
    c2row: Mapping[str, Any],
    out_path: Path,
    keep_report: Mapping[str, Any],
    mode: str,
) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ev_dir = Path(ev_dir)
    out_path = Path(out_path)
    mode = "full" if mode == "full" else "cubes"

    # ----- gather cube panels (always), in dashboard row-major order ----
    cube_panels: List[_Panel] = []
    for panel_name, fname_tmpl, title in _CUBE_PANELS:
        p = ev_dir / "Level2" / "plots" / fname_tmpl.format(name=name)
        cube_panels.append(_load_panel(panel_name, title, p))

    # ----- voltages panel (only loaded/attempted in "full" mode; cubes
    # mode ignores the filterbank waterfall entirely — no placeholder) --
    fb_panel: Optional[_Panel] = None
    if mode == "full":
        fb_path = ev_dir / "filterbank" / f"{name}.png"
        fb_panel = _load_panel(
            "filterbank", "Coherent-beam dynamic spectrum (bbproc)",
            fb_path)

    panels_present = [p.name for p in cube_panels if p.image is not None]
    if fb_panel is not None and fb_panel.image is not None:
        panels_present.append(fb_panel.name)

    # ----- header values -----------------------------------------------
    position_str = _position_header_str(c2row)
    # v3.6: quote the peak C1 detection's DM (same one the panel
    # reticles are keyed to), not the c2 cluster median — see
    # _resolve_detection_dm's docstring. Fall back to the cluster median
    # under an explicit "DM(cluster med)" label if the C1-window CSV
    # isn't there/parsable (e.g. very old archived events, or the cubes-
    # only synthetic fixtures some tests use).
    det_dm = _resolve_detection_dm(ev_dir, name)
    if det_dm is not None:
        line1 = _line1_str(c2row, dm_value=det_dm, dm_label="DM")
    else:
        line1 = _line1_str(c2row, dm_label="DM(cluster med)")
    line2 = _line2_str(c2row, keep_report)

    # ----- figure layout (serif, plain, dense — see module docstring) --
    #
    # Sized in real inches from the ACTUAL source-image aspect ratios (not
    # a fixed fig_h with generic height_ratios) so each panel fills its
    # column edge-to-edge with only small consistent gutters — the figure
    # grows taller as needed rather than stretching/padding panels to an
    # arbitrary box. Target width ~16in @ 100dpi = 1600px, per orchestrator
    # review.
    # 150 dpi (2400px wide): Slack's inline preview downsamples — at
    # 100 dpi the panel tick fonts went soft when Slack scaled the card.
    fig_w, dpi = 16.0, 150
    # top_m bumped 0.15->0.28in (v3.3): the header's 22pt bold name text
    # was clipping against the figure's top edge with only 0.15in above
    # it (see _draw_header_block's docstring — clip_on is now also False
    # there as a second, structural fix).
    left_m, right_m, top_m, bottom_m = 0.15, 0.15, 0.28, 0.15  # inches
    gutter = 0.18  # inches — small consistent gutter between panels
    default_aspect = 1.3  # fallback (w/h) for placeholder panels

    def _aspect(panel: _Panel) -> float:
        if panel.image is not None:
            h, w = panel.image.shape[0], panel.image.shape[1]
            if h > 0 and w > 0:
                return w / h
        return default_aspect

    usable_w = fig_w - left_m - right_m
    col_w = (usable_w - gutter) / 2.0  # two equal columns, bottom row

    # Top row (dm_time | image_peak) equal 50/50, same as the bottom row
    # (a v3.6 45/55 skew toward the sky image made it oversized relative
    # to the DM-time waterfall and was reverted on review).
    _TOP_ROW_SPLIT = 0.50  # dm_time's share; image_peak gets the rest
    row1_total_w = usable_w - gutter
    col_w_dm = row1_total_w * _TOP_ROW_SPLIT
    col_w_img = row1_total_w * (1.0 - _TOP_ROW_SPLIT)
    top_col_w_avg = row1_total_w / 2.0  # for the wspace fraction below

    row2_aspect = (_aspect(cube_panels[2]) + _aspect(cube_panels[3])) / 2.0
    # v3.4: no more title-strip row reserved above each panel — the
    # archived source PNGs already carry their own embedded titles (see
    # _draw_panel). v3.7: each row's height is the MEAN of its two
    # panels' native aspect heights and both draw with aspect="auto"
    # (see _draw_panel) — identical box extents per row, mild stretch.
    row1_h = (
        col_w / _aspect(cube_panels[0]) + col_w / _aspect(cube_panels[1])
    ) / 2.0
    row2_h = col_w / row2_aspect

    # v3.2 tightened this from a legacy 1.55/0.26 fixed box down to
    # 1.05/0.14 (dense, no dead space below the 4 lines). v3.3 opened it
    # back up slightly (1.05->1.20, 0.14->0.18) to give the now-centered,
    # lower-positioned name (see _draw_header_block) proper headroom
    # instead of packing it against the figure's top edge.
    header_h = 1.20  # inches — dense-but-not-cramped metadata block
    header_gap = 0.18  # clearance before the row-1 panel titles below

    fb_h = 0.0
    fb_gap = 0.0
    if mode == "full":
        # v3.4: no more plain-text section label above the bbproc panel
        # — the archived filterbank PNG already carries its own embedded
        # title ("dsa110-bbproc candidate inspection"), so no extra
        # label height is reserved here either.
        fb_aspect = _aspect(fb_panel) if fb_panel is not None else 1.4
        fb_h = usable_w / fb_aspect
        fb_gap = gutter

    fig_h = (
        top_m + header_h + header_gap + row1_h + gutter + row2_h
        + fb_gap + fb_h + bottom_m
    )

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)

        row_heights = [header_h, header_gap, row1_h, gutter, row2_h]
        if mode == "full":
            row_heights += [fb_gap, fb_h]
        gs = gridspec.GridSpec(
            len(row_heights), 1, height_ratios=row_heights, hspace=0.0,
            top=1.0 - top_m / fig_h, bottom=bottom_m / fig_h,
            left=left_m / fig_w, right=1.0 - right_m / fig_w,
        )

        _draw_header_block(fig, gs[0], name, line1, line2, position_str)

        # v3.4: each cube panel now gets the whole row height (no title
        # strip reserved above it) — the archived source PNG's own
        # embedded title is all the label the panel gets.
        gs_row1 = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[2],
            width_ratios=[_TOP_ROW_SPLIT, 1.0 - _TOP_ROW_SPLIT],
            wspace=gutter / top_col_w_avg,
        )
        for i in (0, 1):
            _draw_panel(fig.add_subplot(gs_row1[0, i]), cube_panels[i])

        gs_row2 = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[4], wspace=gutter / col_w,
        )
        for i in (0, 1):
            _draw_panel(fig.add_subplot(gs_row2[0, i]), cube_panels[2 + i])

        if mode == "full":
            ax_fb = fig.add_subplot(gs[6])
            _draw_fb_panel(ax_fb, fb_panel)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # max zlib compression — this is a plain, mostly-white composite
        # of already-rasterized source PNGs, so the extra compression
        # effort is cheap and keeps the (now larger, ~1600px-wide) card
        # under the ~1.5MB target.
        fig.savefig(str(out_path), facecolor=_BG, dpi=dpi,
                   pil_kwargs={"compress_level": 9})
        plt.close(fig)

    return {"ok": True, "path": str(out_path), "error": None,
            "panels": panels_present}


def _draw_fb_panel(ax: Any, fb_panel: Optional[_Panel]) -> None:
    if fb_panel is None:
        fb_panel = _Panel(
            name="filterbank",
            title="Coherent-beam dynamic spectrum (bbproc)",
            placeholder="Coherent-beam dynamic spectrum (bbproc)\n"
                        "(not available)",
        )
    _draw_panel(ax, fb_panel)


def _draw_header_block(
    fig: Any, subplot_spec: Any, name: str, line1: str, line2: str,
    position_str: str,
) -> None:
    """All header text in ONE axes, with explicit tight y-spacing between
    lines — a dense compact metadata block (legacy filplot-style
    suptitle), rather than four separately gridspec'd rows whose
    proportional spacing left visible dead space between lines.

    v3.3: centered (previously left-aligned at x=0.0) for a more
    deliberate look, and re-y-positioned with more headroom above the
    name — at the old y=0.90/va="center"/clip_on=True the 22pt bold
    name's ascenders had only ~0.10 * header_h of headroom before the
    axes' own top edge, which clipped visibly. ``clip_on=False`` removes
    the clipping hazard outright (nothing else shares this axes to be
    clipped against); the y-positions are also pulled down slightly
    (0.90->0.83) and given more gap before line1 (0.62->0.52) so the
    block reads as deliberately spaced rather than packed against the
    figure's top edge.
    """
    ax = fig.add_subplot(subplot_spec)
    ax.set_facecolor(_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    common = dict(ha="center", va="center", transform=ax.transAxes,
                 clip_on=False, color=_FG)
    # v3.5: bumped a notch (22->24, 12.5->14) for clarity — the header
    # block was reading small relative to the (now similarly-sized, 16pt)
    # embedded panel titles below it.
    ax.text(0.5, 0.83, name, fontsize=24, fontweight="bold", **common)
    ax.text(0.5, 0.52, line1, fontsize=14, **common)
    ax.text(0.5, 0.28, line2, fontsize=14, **common)
    ax.text(0.5, 0.06, position_str, fontsize=14, fontweight="bold",
           **common)
