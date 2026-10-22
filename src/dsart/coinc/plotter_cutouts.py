"""The four C2 event panels from x2 cutouts (docs/x2/downstream_inventory.md D8).

Entered from :func:`dsart.coinc.plotter.render_event_plots` only when the
event has ``cutouts/``; 1x events (``cubes/``) never reach this module, so
their PNGs are byte-for-byte unchanged. Same file names, figure geometry,
fonts and colour conventions as the cube panels, with three differences
the 2x data force:

* time axes are in ms relative to the detection (rows are external
  samples, 0.524 ms; tier-B DMs repeat each 1.05 ms value on two rows);
* the DM-time waterfall stacks EVERY half's fines sorted by DM (2x
  ownership is interleaved, d % 8), labelled from the per-fine DMs the
  cutouts carry -- never inferred from a plan or the (sid, g) order; a gap
  in DM coverage (a missing half) is drawn as an empty band;
* the image panel shows the owner half's ``peak_image_w`` (full plane,
  mean over the detected width) with the detected pixel.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..x2.cutout.products import EventProducts, PeakPlane

_LOG = logging.getLogger("dsart.coinc.plotter_cutouts")

__all__ = ["render_cutout_plots", "PeakInfo", "assemble_dm_time"]

#: DM gaps larger than this many median spacings get an empty band.
_GAP_FACTOR = 4.0


class PeakInfo:
    """The detection the panels are centred on (max-SNR C1 member)."""

    def __init__(self, *, fine_global: int, event_specnum: int, width: int,
                 l_pix: int, m_pix: int, dm: float, snr: float,
                 sid: int, g: int, kernel_id: str = "") -> None:
        self.fine_global = int(fine_global)
        self.event_specnum = int(event_specnum)
        self.width = max(1, int(width))
        self.l_pix = int(l_pix)
        self.m_pix = int(m_pix)
        self.dm = float(dm)
        self.snr = float(snr)
        self.sid = int(sid)
        self.g = int(g)
        self.kernel_id = str(kernel_id)


def _peak_info(job) -> Optional[PeakInfo]:
    if job.members:
        m = max(job.members, key=lambda e: float(e.snr))
        return PeakInfo(
            fine_global=m.dm_idx_global, event_specnum=m.event_specnum,
            width=m.width_samples, l_pix=m.l_pix, m_pix=m.m_pix,
            dm=m.dm_pc_cc, snr=m.snr, sid=m.search_node_id, g=m.gpu_half,
            kernel_id=m.kernel_id)
    p = job.archive_root / job.event_name / "Level2" / f"C1_window_{job.event_name}.csv"
    best: Optional[dict] = None
    try:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    if best is None or float(r["snr"]) > float(best["snr"]):
                        best = r
                except (KeyError, ValueError):
                    continue
    except OSError:
        return None
    if best is None:
        return None
    try:
        return PeakInfo(
            fine_global=int(float(best["dm_idx_global"])),
            event_specnum=int(float(best["event_specnum"])),
            width=int(float(best["width_samples"])),
            l_pix=int(float(best["l_pix"])), m_pix=int(float(best["m_pix"])),
            dm=float(best["dm_pc_cc"]), snr=float(best["snr"]),
            sid=int(float(best["search_node_id"])), g=int(float(best["gpu_half"])),
            kernel_id=str(best.get("kernel_id", "")))
    except (KeyError, ValueError):
        return None


def _missing_note(job) -> str:
    p = job.archive_root / job.event_name / "Level3" / f"{job.event_name}.json"
    try:
        with open(p) as fh:
            prod = (json.load(fh) or {}).get("products") or {}
    except (OSError, ValueError):
        return ""
    miss = prod.get("missing") or {}
    return f" · missing {', '.join(sorted(miss))}" if miss else ""


def assemble_dm_time(
    planes: Sequence[PeakPlane], t_peak_ext: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Stack every plane's fines sorted by DM on one physical time grid.

    Returns ``(img (rows, T), t_ms (T,), row_dm (rows,), row_step (rows,),
    row_src)``; ``row_src[i] = (plane index, plane-local fine, fine_global)``
    (``(-1, -1, -1)`` for an inserted gap row, whose image row is NaN).
    """
    planes = [p for p in planes if p.fine_dm is not None and p.peak_grid.size]
    if not planes:
        return (np.zeros((0, 0)), np.zeros(0), np.zeros(0), np.zeros(0, int), [])
    t_us = float(planes[0].t_ext_us or 524.288)
    t_lo = min(int(p.t0_ext) for p in planes)
    t_hi = max(int(p.t0_ext) + p.peak_grid.shape[0] for p in planes)
    n_t = t_hi - t_lo
    entries = []
    for i, p in enumerate(planes):
        for j in range(p.peak_grid.shape[1]):
            entries.append((float(p.fine_dm[j]), i, j,
                            int(p.fine_global_idx[j]) if p.fine_global_idx is not None else -1))
    entries.sort()
    dms = np.array([e[0] for e in entries])
    steps = np.diff(dms)
    med = float(np.median(steps)) if steps.size else 0.0
    rows_img: List[np.ndarray] = []
    row_dm: List[float] = []
    row_step: List[int] = []
    row_src: List[Tuple[int, int, int]] = []
    prev = None
    for dm, i, j, fg in entries:
        if prev is not None and med > 0 and dm - prev > _GAP_FACTOR * med:
            rows_img.append(np.full(n_t, np.nan, np.float32))
            row_dm.append(float("nan"))
            row_step.append(0)
            row_src.append((-1, -1, -1))
        p = planes[i]
        row = np.full(n_t, np.nan, np.float32)
        off = int(p.t0_ext) - t_lo
        series = np.asarray(p.peak_grid[:, j], np.float32).copy()
        if p.row_valid is not None:
            series[~np.asarray(p.row_valid, bool)] = np.nan
        row[off:off + series.shape[0]] = series
        rows_img.append(row)
        row_dm.append(dm)
        row_step.append(int(p.t_step))
        row_src.append((i, j, fg))
        prev = dm
    t_ms = (np.arange(n_t) + t_lo - int(t_peak_ext)) * t_us / 1000.0
    return (np.stack(rows_img), t_ms, np.asarray(row_dm), np.asarray(row_step),
            row_src)


def _nan_boxcar(row: np.ndarray, w: int, boxcar) -> np.ndarray:
    if w <= 1:
        return row
    ok = np.isfinite(row)
    out = boxcar(np.where(ok, row, 0.0), w)
    out[~ok] = np.nan
    return out


def render_cutout_plots(job, products: EventProducts) -> List[Path]:
    from . import plotter as P   # styling + shared panels (lazy: no cycle)

    plots_dir = job.archive_root / job.event_name / "Level2" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    name = job.event_name
    peak = _peak_info(job)
    _, kernels = P._resolve_burst(job)
    written: List[Path] = []
    note = _missing_note(job)
    planes = products.planes
    owner: Optional[Tuple[PeakPlane, int]] = (
        products.plane_of_fine(peak.fine_global) if peak is not None else None)
    w = peak.width if (peak is not None and P._smoothing_enabled()) else 1
    t_peak = peak.event_specnum if peak is not None else (
        int(planes[0].t0_ext + planes[0].peak_grid.shape[0] // 2) if planes else 0)

    # ---- DM x time ------------------------------------------------------
    path = plots_dir / f"dm_time_{name}.png"
    img, t_ms, row_dm, row_step, row_src = assemble_dm_time(planes, t_peak)
    if img.size == 0:
        written.append(P._placeholder(path, f"DM × time — {name}", "no cutout data"))
    else:
        import matplotlib.pyplot as plt
        sm = np.stack([_nan_boxcar(r, w, P._boxcar) for r in img])
        z = P._robust_row_normalise(sm)
        finite = z[np.isfinite(z)]
        vmax = max(float(np.percentile(finite, 99.9)), 0.5 * float(finite.max()), 1.0) \
            if finite.size else 1.0
        dt = float(t_ms[1] - t_ms[0]) if t_ms.size > 1 else 0.5
        fig = plt.figure(figsize=(10.0, 9.0))
        ax = fig.add_axes(P._PANEL_RECT)
        im = ax.imshow(z, aspect="auto", origin="lower", cmap="viridis",
                       vmin=-2.0, vmax=vmax, interpolation="nearest",
                       extent=(t_ms[0] - dt / 2, t_ms[-1] + dt / 2,
                               -0.5, z.shape[0] - 0.5))
        ax.tick_params(axis="x", labelsize=P._TICK_FONTSIZE)
        cax = fig.add_axes(P._CBAR_RECT)
        cb = fig.colorbar(im, cax=cax, label=(
            "image-max amplitude (robust σ per DM row"
            + (f", boxcar w={w}" if w > 1 else "") + ")"))
        cb.ax.tick_params(labelsize=P._TICK_FONTSIZE)
        cb.set_label(cb.ax.get_ylabel(), fontsize=P._AXIS_LABEL_FONTSIZE)
        good = np.flatnonzero(np.isfinite(row_dm))
        ticks = np.unique(np.linspace(good[0], good[-1], min(12, good.size)).round().astype(int))
        ticks = np.array([t for t in ticks if np.isfinite(row_dm[t])])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{row_dm[t]:.0f}" for t in ticks], fontsize=P._TICK_FONTSIZE)
        ax.set_ylabel("dispersion measure (pc cm⁻³)", fontsize=P._AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("time relative to the detection (ms)", fontsize=P._AXIS_LABEL_FONTSIZE)
        tier_b = np.flatnonzero(row_step >= 2)
        if tier_b.size and tier_b[0] > 0:
            ax.axhline(tier_b[0] - 0.5, color="w", lw=0.8, ls=":", alpha=0.8)
            ax.text(1.002, tier_b[0], "tier B\n(1.05 ms)", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", fontsize=7, color="0.35", clip_on=False)
        for r, src in enumerate(row_src):
            if src[0] < 0:
                ax.axhspan(r - 0.5, r + 0.5, color="0.85", alpha=0.6, lw=0)
        burst_row = None
        if peak is not None:
            hits = [r for r, s in enumerate(row_src) if s[2] == peak.fine_global]
            burst_row = hits[0] if hits else None
            ax.axvline(0.0, color=P._RETICLE, lw=1.0, ls="--", alpha=0.9)
            if burst_row is not None:
                ax.axhline(burst_row, color=P._RETICLE, lw=1.0, ls="--", alpha=0.9)
                ax.plot(0.0, burst_row, marker="+", color=P._RETICLE, ms=16, mew=2.0)
        if finite.size:
            ar, at = np.unravel_index(int(np.nanargmax(z)), z.shape)
            ax.plot(t_ms[at], ar, marker="x", color="w", ms=12, mew=2.0, ls="none")
        ax.set_title(f"DM × time (waterfall) — {name}", fontsize=P._TITLE_FONTSIZE,
                     pad=P._TITLE_PAD)
        if peak is not None:
            wms = peak.width * float(planes[0].t_ext_us or 524.288) / 1000.0
            P._fit_title_line(fig, ax, (
                f"DM {peak.dm:.1f} pc cm⁻³ · SNR {peak.snr:.1f} (detector) · "
                f"w {wms:.2f} ms" + note))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # ---- image at peak -----------------------------------------------------
    path = plots_dir / f"image_peak_{name}.png"
    cut = None
    if owner is not None:
        cut = products.cutouts.get(owner[0].half)
    if cut is None:
        for c in products.cutouts.values():
            if int(c.scalar("has_peak_image")):
                cut = c
                break
    if cut is None or not int(cut.scalar("has_peak_image")):
        written.append(P._placeholder(path, f"image at peak — {name}",
                                      "no peak image (owner half missing)" + note))
    else:
        import matplotlib.pyplot as plt
        img2 = np.asarray(cut["peak_image_w"], np.float32)
        lp, mp = int(cut.scalar("l_pix_req")), int(cut.scalar("m_pix_req"))
        wreq = int(cut.scalar("width_ext_req"))
        fig = plt.figure(figsize=(10.0, 9.0))
        ax = fig.add_axes(P._PANEL_RECT)
        im = ax.imshow(img2, origin="lower", cmap="magma")
        ax.tick_params(axis="both", labelsize=P._TICK_FONTSIZE)
        cax = fig.add_axes(P._CBAR_RECT)
        cb = fig.colorbar(im, cax=cax, label="amplitude")
        cb.ax.tick_params(labelsize=P._TICK_FONTSIZE)
        cb.set_label(cb.ax.get_ylabel(), fontsize=P._AXIS_LABEL_FONTSIZE)
        if lp >= 0 and mp >= 0:
            ax.axhline(lp, color=P._RETICLE, lw=0.8, ls="--", alpha=0.8)
            ax.axvline(mp, color=P._RETICLE, lw=0.8, ls="--", alpha=0.8)
            ax.add_patch(plt.Circle((mp, lp), radius=8.0, fill=False,
                                    edgecolor=P._RETICLE, lw=1.8))
        ax.set_xlabel("m (pix)", fontsize=P._AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("l (pix)", fontsize=P._AXIS_LABEL_FONTSIZE)
        t_us = float(cut.scalar("t_ext_us"))
        dm_req = float(cut.scalar("dm_req"))
        ax.set_title(
            f"image at (DM={dm_req:.1f}, mean of {wreq * t_us / 1000:.2f} ms) — {name}",
            fontsize=P._TITLE_FONTSIZE, pad=P._TITLE_PAD)
        P._fit_title_line(fig, ax, f"detected pixel (l,m)=({lp},{mp}) · "
                                   f"s{cut.sid}g{cut.gpu_half}" + note)
        fig.text(P._CAPTION_X, P._CAPTION_Y,
                 "red circle = burst position reported by the detector",
                 ha="center", va="baseline", fontsize=P._LEGEND_FONTSIZE, color="#3b4252")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # ---- lightcurve ---------------------------------------------------------
    path = plots_dir / f"lightcurve_{name}.png"
    if owner is None:
        written.append(P._placeholder(path, f"lightcurve — {name}",
                                      "detection's DM not in any cutout" + note))
    else:
        import matplotlib.pyplot as plt
        plane, j = owner
        t_us = float(plane.t_ext_us or 524.288)
        x = (np.arange(plane.peak_grid.shape[0]) + int(plane.t0_ext) - t_peak) * t_us / 1000.0
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        ax.plot(x, P._robust_z(plane.peak_grid[:, j]), color="#b2bec3", lw=1.0,
                alpha=0.9, label="image max (context)")
        if plane.pix_lc is not None:
            pix = np.asarray(plane.pix_lc[:, j], np.float32)
            ax.plot(x, P._robust_z(pix), color="#0984e3", lw=1.2,
                    label=f"pixel ({peak.l_pix},{peak.m_pix}) raw")
            if w > 1:
                ax.plot(x, P._robust_z(P._boxcar(pix, w)), color="#d63031", lw=1.8,
                        label=f"pixel, boxcar {w * t_us / 1000:.2f} ms (detector-matched)")
        ax.axvline(0.0, color=P._RETICLE, lw=1.4, ls="--", label="detection")
        ax.tick_params(axis="both", labelsize=P._TICK_FONTSIZE)
        ax.legend(loc="upper right", fontsize=P._LEGEND_FONTSIZE)
        ax.set_xlabel("time relative to the detection (ms)",
                      fontsize=P._AXIS_LABEL_FONTSIZE_SMALL)
        ax.set_ylabel("σ above the row's own median", fontsize=P._AXIS_LABEL_FONTSIZE_SMALL)
        ax.set_title(f"lightcurve — {name} · DM {plane.fine_dm[j]:.1f}",
                     fontsize=P._TITLE_FONTSIZE)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # ---- kernel SNRs (detection metadata only; unchanged) -----------------
    written.append(P._render_kernel_snrs(plots_dir, name, kernels))
    _LOG.info("plotter: event=%s rendered from %d cutout planes (%d halves)",
              name, len(planes), len(products.halves))
    return written


def describe(products: EventProducts) -> Dict[str, Any]:
    """Small JSON-able summary (dashboard / logs)."""
    return {"kind": products.kind, "halves": sorted(products.halves),
            "planes": len(products.planes), "errors": dict(products.errors)}
