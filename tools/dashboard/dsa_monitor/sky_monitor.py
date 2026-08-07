"""Sky-monitor backend: ingest corr-node static-sky snapshots, combine
the 16 chgroups in the UV plane, image, and serve a scrubbable movie.

E2E correctness test 1 ("always seeing the sky"). Each corr node's
``corr_fast`` (with ``--sky-export-url``) POSTs its slot-0
:class:`StaticSkyMean` window mean (a ~1 s sliding boxcar, kept short
relative to the ~6.9 s/2π max core fringe rate) — ``(N_filled,)``
complex64 gridded
visibilities plus the sparsity-pattern indices — to ``/sky/ingest``
every 30 s. This module:

1. keeps the latest snapshot per chgroup in memory,
2. every ``frame_interval_s`` builds a fleet-combined dirty image
   (scatter each chgroup's sparse cells into the dense 256×256 UV
   grid, weight by ``1/amp_scale²`` to flatten the bandpass, sum,
   then ``Re(fftshift(ifft2(ifftshift(grid))))`` — the exact
   convention of ``bench/_corr_fast_replay.dirty_image_from_dense_grid``
   and the search-side imager), with two h23-only anti-aliasing steps
   (2026-06-09): 2× UV zero-padding (exact band-limited oversampling
   of the same FOV → 512×512 frames, the ~1-px PSF main lobe renders
   smoothly instead of as single hard pixels) and pillbox grid
   correction (divides out the nearest-cell-gridding sinc envelope so
   edge-of-FOV sources show at true relative strength). True gridding
   anti-aliasing (suppressing out-of-FOV aliases) is NOT possible
   post-gridding — that would need the corr-side G7 Gaussian kernel
   (``kernel_support ∈ {3,5}``), which changes ``pattern_id``
   fleet-wide and the search-side scatter cost,
3. sigma-normalises (robust MAD noise estimate over image pixels) and
   writes a greyscale PNG + a float32 NPZ to
   ``/dataz/dsa110/operations/sky_monitor/frames/YYYYMMDD/``,
4. prunes frames older than the retention window (default 48 h, so
   the UI's 24 h scrub never hits a hole).

Amplitude-cal caveat (by design, agreed 2026-06-09): production
``corr_fast`` runs ``--cal-mode phase_only``, so the gridded vis still
carry the instrumental gain *magnitudes*. The per-baseline amplitude
solutions cannot be applied after gridding (cells mix baselines), so
each corr node ships a per-chgroup scalar ``amp_scale`` (median |G| of
its cal solutions) and we divide that chgroup's vis by ``amp_scale²``
(baseline gain = product of two antenna gains). That flattens the
dominant amplitude structure — the bandpass shape across the 16
chgroups — and the per-image sigma normalisation absorbs the overall
scale. Within-chgroup per-antenna amplitude errors remain (small PSF
perturbation only).

No torch dependency: pure numpy + matplotlib(Agg), so the dashboard
env stays light.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Force Agg before pyplot-adjacent imports (matches plot_render.py).
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg                       # noqa: E402
from matplotlib.figure import Figure                   # noqa: E402

import sky_astrometry                                  # noqa: E402  (local)

LOG = logging.getLogger("dsa_monitor.sky_monitor")

#: Wire-format version — must match dsart.services.sky_export.
SKY_SNAPSHOT_VERSION: int = 1

#: Frame store root. /dataz is the operations NFS share (user-chosen
#: 2026-06-09); override via env for tests / dev hosts.
SKY_MONITOR_ROOT: Path = Path(os.environ.get(
    "DSA_SKY_MONITOR_ROOT", "/dataz/dsa110/operations/sky_monitor",
))

#: Greyscale display stretch in sigma units. -3σ..+10σ shows the noise
#: floor as mid-grey texture and saturates only genuinely bright
#: continuum sources.
PNG_VMIN_SIGMA: float = -3.0
PNG_VMAX_SIGMA: float = 10.0

#: Frame filename: sky_<unix_ts:int>_n<chgroups>.png / .npz. The ts and
#: chgroup count are encoded in the name so the index endpoint never
#: has to open the NPZ.
_FRAME_RE = re.compile(r"^sky_(?P<ts>\d+)_n(?P<ncg>\d+)\.(?:png|npz|json)$")

#: Seconds per fada block (mirrors dsart.services.sky_export.BLOCK_S;
#: duplicated so the dashboard env does not import the dsart package).
BLOCK_S: float = 2048 * 65.536e-6

#: corr_fast capture→export latency, used only as the ABSOLUTE time
#: fallback when the etcd capture-arm anchor is unavailable. Measured
#: 2026-07-18: unix_ts − (armed_mjd + block_n·BLOCK_S) = 1.96 ± 0.01 s
#: across all 16 corr nodes.
EXPORT_LAG_FALLBACK_S: float = 2.0

#: MJD → unix epoch offset (unix = (mjd − 40587) · 86400).
_MJD_UNIX_EPOCH: float = 40587.0

#: Reference per-antenna SEFD (Jy) the noise report compares against
#: (operator-expected value, 2026-07-18).
SEFD_REFERENCE_JY: float = 7000.0

#: Effective parameters for the SEFD-referenced noise prediction:
#: N core antennas (outriggers are cal-zeroed in fast-vis), Stokes-I
#: pol count, processed bandwidth (REALTIME_FRB_SEARCH.md §15.18)
#: derated by the typical RFI-flagged fraction (~3%).
SEFD_N_ANT: int = 82
SEFD_N_POL: int = 2
SEFD_BW_HZ: float = 187.485e6 * 0.97

#: NVSS S/N threshold to call a source "detected" (image markers +
#: flux-scale regression membership).
NVSS_DETECT_SNR: float = 7.0

#: Minimum primary-beam attenuation for a source to enter the
#: flux-scale regression (heavily attenuated edge sources contribute
#: mostly PB-model error, not scale information).
NVSS_FLUXSCALE_MIN_PB: float = 0.15

#: Robust per-chgroup UV-cell amplitude clip (2026-07-18). The
#: static-sky mean of UN-fringestopped drift vis integrates
#: terrestrial RFI and crosstalk COHERENTLY (they don't fringe), so a
#: small set of UV cells — concentrated near the u≈0 / v≈0 arm axes —
#: carries amplitudes orders of magnitude above the ~few-Jy per-cell
#: thermal level and paints the whole frame with stripes. Sky signal
#: cannot do this: even a 1 Jy source adds ≲ a Jy to every cell. Cells
#: with |V| > CLIP × median|V| (per chgroup) are zeroed before the
#: combine; the count is recorded per chgroup. Caveat: the Sun (~1e5
#: Jy) would trip the clip everywhere — daytime frames will clip
#: aggressively and are flagged by a high clipped fraction.
UV_CLIP_K: float = 8.0

#: Instrument-static baseline subtraction (2026-07-19). Beyond the
#: hot cells the clip removes, a pervasive static component (low-level
#: crosstalk / correlator artifacts / RFI, measured at ~10–20% complex
#: correlation between snapshots 6 min apart even after excluding the
#: top-1% amplitude cells) sits in MOST cells and dominates the frame
#: over the mJy-level sky. It is static in the INSTRUMENT (u, v)
#: frame, while the drifting sky winds its per-cell phase by ≫2π
#: between 30 s snapshots at nearly all baselines — so the per-cell
#: temporal median over the last ``STATIC_SUB_MAXLEN`` snapshots
#: estimates the instrumental baseline with ≲10% sky contamination
#: (only cells at u ≲ 20 λ retain appreciable sky coherence across
#: the history). Subtracting it before the combine leaves
#: thermal-noise-limited sky images. Requires
#: ``STATIC_SUB_MIN_HIST`` snapshots of history per chgroup
#: (~5 min after a cold start; seeded from stored frames on restart).
STATIC_SUB_MIN_HIST: int = 10
STATIC_SUB_MAXLEN: int = 40


def snapshot_data_mid_unix(
    meta: dict[str, Any], *, armed_unix: Optional[float],
) -> float:
    """Unix time of the CENTER of a snapshot's ~1 s averaging window.

    Preferred path: the capture-arm anchor —
    ``armed_unix + block_n·BLOCK_S`` is the capture time of the
    window's newest block (verified against the C3 voltage-dump
    manifests to ~0.2 ms), minus half the ``window_blocks`` boxcar.

    Fallback (no/stale arm anchor): the corr's wall clock at export
    (``unix_ts``) minus the measured capture→export latency.
    """
    wb = float(meta.get("window_blocks") or 8)
    half_window_s = (wb - 1.0) / 2.0 * BLOCK_S
    unix_ts = float(meta.get("unix_ts") or 0.0)
    block_n = meta.get("block_n")
    if armed_unix is not None and block_n is not None:
        t_newest = armed_unix + float(block_n) * BLOCK_S
        # A re-arm (new armed_mjd) or stale etcd value makes the
        # block-derived time disagree wildly with the corr wall clock;
        # trust the wall clock (minus measured lag) in that case.
        if unix_ts <= 0.0 or abs(unix_ts - t_newest) < 30.0:
            return t_newest - half_window_s
    return unix_ts - EXPORT_LAG_FALLBACK_S - half_window_s


def parse_snapshot_npz(body: bytes) -> dict[str, Any]:
    """Decode one corr-node snapshot payload (mirror of
    ``dsart.services.sky_export.parse_snapshot_npz`` — duplicated here
    so the dashboard env does not import the dsart package).
    """
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:
            version = int(z["version"])
            if version != SKY_SNAPSHOT_VERSION:
                raise ValueError(
                    f"sky snapshot version {version} != "
                    f"{SKY_SNAPSHOT_VERSION}"
                )
            vis = np.asarray(z["vis"], dtype=np.complex64)
            ix_row = np.asarray(z["ix_row"], dtype=np.uint16)
            ix_col = np.asarray(z["ix_col"], dtype=np.uint16)
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    except (KeyError, OSError, json.JSONDecodeError,
            zipfile.BadZipFile) as exc:
        raise ValueError(f"malformed sky snapshot: {exc!r}") from exc
    n_filled = vis.shape[0]
    if vis.ndim != 1 or ix_row.shape != (n_filled,) or ix_col.shape != (n_filled,):
        raise ValueError(
            f"sky snapshot shape mismatch: vis={vis.shape} "
            f"ix_row={ix_row.shape} ix_col={ix_col.shape}"
        )
    if not isinstance(meta, dict) or "chgroup" not in meta:
        raise ValueError("sky snapshot meta missing 'chgroup'")
    return {"vis": vis, "ix_row": ix_row, "ix_col": ix_col, "meta": meta}


# ---------------------------------------------------------------------------
# Imaging primitives (numpy mirrors of the search-side torch versions)
# ---------------------------------------------------------------------------


def combine_chgroups_to_uv(
    snapshots: list[dict[str, Any]],
    *,
    n_grid: int = 256,
    align_dl_rad: Optional[dict[int, float]] = None,
    uv_clip_k: float = UV_CLIP_K,
) -> tuple[np.ndarray, list[int], dict[int, dict[str, int]]]:
    """Sum the per-chgroup sparse snapshots into one dense UV grid.

    Each chgroup's vis is divided by ``amp_scale²`` (bandpass
    flattening — see module docstring) before scattering. Chgroups
    whose ``n_grid`` doesn't match are skipped with a warning (a corr
    node running a stale config must not corrupt the whole frame).

    ``align_dl_rad`` (2026-07-18 astrometry fix): per-chgroup image-
    plane l-shift (rad) applied as a UV phase ramp before scattering.
    The 16 snapshots in a frame are taken up to ~30 s apart in DATA
    time; the drift-scan phase center moves ~7.3 arcmin of RA in 30 s,
    so combining them unaligned smears every source into a trail of
    faint copies. The ramp ``vis · exp(−2πi · u_λ · Δl)`` translates a
    chgroup's image by ``+Δl`` (the M2-validated FFT convention:
    ``image(l,m) = Σ V[r,c] · exp(+2πi·cell_λ·((c−N/2)·l + (r−N/2)·m))``,
    so ``u_λ = (ix_col − N/2)·cell_lambda``). The Hermitian half-plane
    convention is preserved (the implicit conjugate cells pick up the
    conjugate ramp).

    ``uv_clip_k`` (2026-07-18): per-chgroup robust amplitude clip —
    zero cells with ``|V| > k · median|V|`` before scattering (see
    ``UV_CLIP_K``). 0 disables. Clip counts are returned per chgroup.

    Returns ``(uv_grid complex64 (n_grid, n_grid), used_chgroups,
    clip_stats)`` where ``clip_stats`` maps chgroup →
    ``{n_clipped, n_cells}``.
    """
    uv = np.zeros((n_grid, n_grid), dtype=np.complex64)
    used: list[int] = []
    clip_stats: dict[int, dict[str, int]] = {}
    for snap in snapshots:
        meta = snap["meta"]
        cg = int(meta["chgroup"])
        if int(meta.get("n_grid", n_grid)) != n_grid:
            LOG.warning(
                "sky combine: chgroup %d n_grid=%s != %d; skipping",
                cg, meta.get("n_grid"), n_grid,
            )
            continue
        amp_scale = float(meta.get("amp_scale", 1.0) or 1.0)
        w = 1.0 / (amp_scale * amp_scale) if amp_scale > 0 else 1.0
        rows = snap["ix_row"].astype(np.int64)
        cols = snap["ix_col"].astype(np.int64)
        if rows.size and (rows.max() >= n_grid or cols.max() >= n_grid):
            LOG.warning(
                "sky combine: chgroup %d pattern indices out of range; "
                "skipping", cg,
            )
            continue
        vis = snap["vis"] * np.float32(w)
        n_clipped = 0
        if uv_clip_k and uv_clip_k > 0 and vis.size:
            amp = np.abs(vis)
            med = float(np.median(amp))
            if med > 0:
                bad = amp > uv_clip_k * med
                n_clipped = int(bad.sum())
                if n_clipped:
                    vis = np.where(bad, np.complex64(0), vis)
        clip_stats[cg] = {"n_clipped": n_clipped, "n_cells": int(vis.size)}
        dl = float((align_dl_rad or {}).get(cg, 0.0))
        cell_lambda = float(meta.get("cell_lambda", 0.0) or 0.0)
        if dl != 0.0 and cell_lambda > 0.0:
            u_lam = (cols - n_grid // 2).astype(np.float64) * cell_lambda
            vis = (vis * np.exp(-2j * np.pi * u_lam * dl)).astype(
                np.complex64
            )
        # np.add.at: pattern cells are unique per chgroup, but += via
        # ufunc.at is safe even if they ever are not.
        np.add.at(uv, (rows, cols), vis)
        used.append(cg)
    return uv, sorted(used), clip_stats


def pillbox_grid_correction(n_pix: int, *, cap: float = 2.5) -> np.ndarray:
    """``(n_pix, n_pix)`` image-plane correction for K=1 gridding.

    The production corr_fast gridder snaps each (baseline, channel)
    visibility to its NEAREST grid cell (``kernel_support=1`` pillbox;
    the G7 Gaussian kernel exists but is not enabled — flipping it
    would change ``pattern_id`` fleet-wide). Nearest-cell snapping is
    convolution with a one-cell-wide pillbox followed by sampling, so
    the dirty image is multiplied by the pillbox's transform: a
    separable ``sinc(f)`` envelope with ``f ∈ [-1/2, 1/2)`` across the
    field of view. Sources at the FOV edge are attenuated to
    ``sinc(1/2) ≈ 0.64`` (corners ``≈ 0.41``).

    This returns ``1 / (sinc(fx) · sinc(fy))`` — the standard imaging
    "grid correction" — so edge sources display at their true relative
    strength. Noise is amplified by the same factor, so SNR is
    unchanged; the per-image sigma stretch just shows slightly more
    texture toward the corners. ``cap`` bounds the correction (the
    geometric max is ``(π/2)² ≈ 2.47`` in the corners).
    """
    f = (np.arange(n_pix, dtype=np.float64) - n_pix // 2) / float(n_pix)
    c1 = 1.0 / np.sinc(f)                  # np.sinc is sin(πf)/(πf)
    corr = np.outer(c1, c1)
    return np.minimum(corr, cap).astype(np.float32)


def dirty_image_from_uv(
    uv: np.ndarray,
    *,
    oversample: int = 1,
    grid_correct: bool = False,
) -> np.ndarray:
    """``Re(fftshift(ifft2(ifftshift(uv))))`` — byte-matches
    ``bench/_corr_fast_replay.dirty_image_from_dense_grid`` (M2-
    validated convention; the F20 (u, v) negation is already applied
    inside the corr-side gridder) at the defaults.

    Anti-aliasing options (2026-06-09, h23-side only):

    * ``oversample > 1``: zero-pad the centred UV grid by the given
      factor before the iFFT. Because the UV data have finite support
      (256² cells), this is EXACT band-limited (Dirichlet)
      interpolation of the same dirty image over the same FOV — the
      ~1-px PSF main lobe of the critically-sampled 256² image is
      rendered smoothly instead of aliasing into single hard pixels.
      No information is added or lost.
    * ``grid_correct``: divide out the pillbox (nearest-cell) gridding
      envelope — see :func:`pillbox_grid_correction`.
    """
    n = int(uv.shape[0])
    oversample = max(1, int(oversample))
    if oversample > 1:
        n_os = n * oversample
        big = np.zeros((n_os, n_os), dtype=np.complex64)
        lo = (n_os - n) // 2                  # centred DC stays at n_os/2
        big[lo:lo + n, lo:lo + n] = uv
        uv = big
    g = np.fft.ifftshift(uv)
    img = np.fft.fftshift(np.fft.ifft2(g)).real.astype(np.float32)
    if oversample > 1:
        # ifft2 normalises by n_os² not n²; restore the 256²-grid scale
        # so recorded medians/sigmas stay comparable across oversample.
        img *= np.float32(oversample * oversample)
    if grid_correct:
        img *= pillbox_grid_correction(img.shape[0])
    return np.ascontiguousarray(img)


def robust_sigma(img: np.ndarray) -> tuple[float, float]:
    """``(median, sigma)`` via the MAD estimator (1.4826 × MAD).

    Robust to the bright continuum sources that are the whole point
    of this monitor — a handful of strong pixels barely moves the
    median absolute deviation of 65k pixels.
    """
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    sigma = 1.4826 * mad
    if sigma <= 0.0 or not np.isfinite(sigma):
        # Degenerate (all-zero) image — avoid div-by-zero downstream.
        sigma = float(np.std(img)) or 1.0
    return med, sigma


#: NVSS overlay: flux cut and S/N aperture (peak pixel within this
#: radius of the predicted position). 2026-07-19: flux cut lowered
#: 100 → 40 mJy (frame σ ≈ 8 mJy after the astrometry fixes, so
#: ~40 mJy sources are detectable and were showing as unlabeled
#: blobs) and the per-frame source cap raised 40 → 80 (a cap the
#: FOV's bright-source count exceeded caused sources to drop in and
#: out of the list — and their labels to flicker — as the field
#: drifted).
NVSS_MIN_MJY: float = 40.0
NVSS_APERTURE_ARCSEC: float = 90.0
NVSS_MAX_SOURCES: int = 80

#: Astrometric self-cal loop (2026-07-19): the per-frame xcorr
#: self-check residual is fed back as an EMA'd (Δl, Δm) correction to
#: the predicted positions (and the graticule), closing out the
#: quasi-static ~(−4, −3) px residual (pointing / aberration / F21
#: reference scale). Only updates on confident locks; hard-capped so
#: a bad frame can't run the correction away.
ASTROM_SELFCAL_MIN_Z: float = 15.0
ASTROM_SELFCAL_GAIN: float = 0.5
ASTROM_SELFCAL_MAX_PX: float = 15.0


def measure_source_peak(
    image: np.ndarray,
    *,
    row: float,
    col: float,
    median: float,
    sigma: float,
    radius_pix: float,
) -> tuple[float, float, float, float]:
    """Peak within ``radius_pix`` of (row, col).

    Returns ``(snr, peak_minus_median, row_peak, col_peak)`` — the
    peak-pixel S/N, its median-subtracted image value (the quantity
    the flux-scale regression consumes), and the peak's pixel
    coordinates (for measured-vs-predicted astrometric offsets).
    All NaN when the aperture is entirely off-frame.
    """
    nanret = (float("nan"),) * 4
    n_pix = image.shape[0]
    r = int(np.ceil(radius_pix))
    r0, r1 = int(np.floor(row)) - r, int(np.floor(row)) + r + 1
    c0, c1 = int(np.floor(col)) - r, int(np.floor(col)) + r + 1
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, n_pix), min(c1, n_pix)
    if r0 >= r1 or c0 >= c1:
        return nanret
    win = image[r0:r1, c0:c1]
    rr, cc = np.mgrid[r0:r1, c0:c1]
    mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius_pix ** 2
    if not mask.any():
        return nanret
    vals = np.where(mask, win, -np.inf)
    flat = int(np.argmax(vals))
    pr, pc = np.unravel_index(flat, win.shape)
    gr, gc = r0 + int(pr), c0 + int(pc)
    peak = float(image[gr, gc] - median)
    row_pk, col_pk = float(gr), float(gc)
    # Parabolic sub-pixel refinement (2026-07-19): at the frames' 2×
    # oversampling the PSF main lobe is ~2-3 px wide, and the peak
    # PIXEL underestimates a source landing between samples by up to
    # ~10-20% depending on sub-pixel phase — a systematic S/N
    # underestimate. A separable 3-point parabola through the peak
    # recovers the continuous maximum to ~1-2%.
    if 1 <= gr < n_pix - 1 and 1 <= gc < n_pix - 1:
        v0 = float(image[gr, gc])
        boost = 0.0
        vr_m, vr_p = float(image[gr - 1, gc]), float(image[gr + 1, gc])
        den = 2.0 * v0 - vr_m - vr_p
        if den > 0:
            dr_sub = 0.5 * (vr_p - vr_m) / den
            if abs(dr_sub) <= 1.0:
                boost += 0.25 * (vr_p - vr_m) * dr_sub
                row_pk = gr + dr_sub
        vc_m, vc_p = float(image[gr, gc - 1]), float(image[gr, gc + 1])
        den = 2.0 * v0 - vc_m - vc_p
        if den > 0:
            dc_sub = 0.5 * (vc_p - vc_m) / den
            if abs(dc_sub) <= 1.0:
                boost += 0.25 * (vc_p - vc_m) * dc_sub
                col_pk = gc + dc_sub
        peak = v0 + boost - median
    return peak / sigma, peak, row_pk, col_pk


def measure_source_snr(
    image: np.ndarray,
    *,
    row: float,
    col: float,
    median: float,
    sigma: float,
    radius_pix: float,
) -> float:
    """(peak pixel within ``radius_pix`` of (row, col) − median) / sigma.

    Returns NaN when the aperture is entirely off-frame.
    """
    snr, _, _, _ = measure_source_peak(
        image, row=row, col=col, median=median, sigma=sigma,
        radius_pix=radius_pix,
    )
    return snr


def measure_astrometric_offset(
    image: np.ndarray,
    *,
    median: float,
    sigma: float,
    nvss_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Global (Δrow, Δcol) between the image and the NVSS predictions.

    FFT cross-correlation of the σ-clipped image against a
    flux-weighted delta map at the predicted source pixels. Returns
    ``{z, drow_px, dcol_px, n_sources}``; ``z`` is the correlation
    peak in std units of the correlation plane — ≳15 is a confident
    lock, and then (drow, dcol) ≈ (0, 0) is the health condition.
    This catches wholesale mapping errors that per-source apertures
    at the (wrong) predicted positions are blind to (2026-07-19
    lesson: a 26-yr precession offset read as "no detections").
    """
    n_pix = int(image.shape[0])
    s = np.clip((np.asarray(image, dtype=np.float64) - median) / sigma,
                -2.0, 30.0)
    dmap = np.zeros((n_pix, n_pix))
    n_src = 0
    for r in nvss_rows:
        row, col = r.get("row"), r.get("col")
        if row is None or col is None:
            continue
        ri, ci = int(round(float(row))), int(round(float(col)))
        if 0 <= ri < n_pix and 0 <= ci < n_pix:
            dmap[ri, ci] += float(np.sqrt(max(r.get("flux_mjy", 0.0), 0.0)))
            n_src += 1
    if n_src < 3:
        return {"z": 0.0, "drow_px": None, "dcol_px": None,
                "n_sources": n_src}
    F = np.fft.fft2(s - s.mean())
    G = np.fft.fft2(dmap - dmap.mean())
    xc = np.fft.fftshift(np.fft.ifft2(F * np.conj(G)).real)
    k = int(np.argmax(xc))
    r0, c0 = np.unravel_index(k, xc.shape)
    z = float((xc[r0, c0] - xc.mean()) / xc.std())
    # Sub-pixel refinement of the correlation peak (3-point parabola
    # per axis) so the self-cal feedback isn't quantised to whole px.
    drow = float(r0 - n_pix // 2)
    dcol = float(c0 - n_pix // 2)
    if 1 <= r0 < n_pix - 1 and 1 <= c0 < n_pix - 1:
        den = 2.0 * xc[r0, c0] - xc[r0 - 1, c0] - xc[r0 + 1, c0]
        if den > 0:
            drow += float(0.5 * (xc[r0 + 1, c0] - xc[r0 - 1, c0]) / den)
        den = 2.0 * xc[r0, c0] - xc[r0, c0 - 1] - xc[r0, c0 + 1]
        if den > 0:
            dcol += float(0.5 * (xc[r0, c0 + 1] - xc[r0, c0 - 1]) / den)
    return {
        "z": z,
        "drow_px": round(drow, 2),
        "dcol_px": round(dcol, 2),
        "n_sources": n_src,
    }


def complex_median(stack: np.ndarray) -> np.ndarray:
    """Component-wise (re, im) median along axis 0 — the robust
    per-cell instrumental-baseline estimator."""
    return (np.median(stack.real, axis=0)
            + 1j * np.median(stack.imag, axis=0)).astype(np.complex64)


def sefd_predicted_sigma_mjy(
    sefd_jy: float, *, window_s: float,
) -> float:
    """Point-source image noise (mJy) predicted for a per-antenna SEFD.

    Standard naturally-weighted synthesis radiometer equation for the
    Stokes-I dirty image::

        σ_I = SEFD / sqrt(N_ant · (N_ant − 1) · n_pol · Δν · τ)

    with the module's effective constants (82 core antennas, dual-pol
    Stokes-I sum, 187.485 MHz processed band × 0.97 RFI derate) and
    the snapshot's ~1.07 s window. Known unmodelled inefficiencies
    (4-bit quantization, pillbox gridding peak smearing, ≤4% fringe
    decorrelation on the longest core baselines, per-antenna amplitude
    mis-weighting under phase-only cal) all act to RAISE the implied
    SEFD a few percent — so the implied value is an effective
    system SEFD, a slightly conservative upper bound on the
    radiometric per-antenna SEFD.
    """
    n = float(SEFD_N_ANT)
    denom = np.sqrt(
        n * (n - 1.0) * SEFD_N_POL * SEFD_BW_HZ * max(window_s, 1e-6)
    )
    return float(sefd_jy) / denom * 1e3


def fit_flux_scale(
    nvss_rows: list[dict[str, Any]],
    *,
    min_snr: float = NVSS_DETECT_SNR,
    min_pb: float = NVSS_FLUXSCALE_MIN_PB,
) -> tuple[Optional[float], int]:
    """Image-units-per-mJy from detected NVSS sources.

    Through-origin least squares of ``peak`` (median-subtracted image
    units) against ``flux_mjy · pb`` (apparent flux after primary-beam
    attenuation) over rows with ``snr ≥ min_snr`` and ``pb ≥ min_pb``.
    Confusion / resolved-source scatter is real but unbiased to first
    order; the regression is dominated by the brightest detections.

    Returns ``(k_units_per_mjy | None, n_sources_used)``. A single
    detection is accepted (the scale is then that source's peak/flux
    ratio — noisy, but a usable per-frame anchor; the n_sources count
    is surfaced so the reader can weigh it).
    """
    xs: list[float] = []
    ys: list[float] = []
    for r in nvss_rows:
        snr = r.get("snr")
        peak = r.get("peak")
        pb = r.get("pb")
        if snr is None or peak is None or pb is None:
            continue
        if not (np.isfinite(snr) and np.isfinite(peak) and np.isfinite(pb)):
            continue
        if snr < min_snr or pb < min_pb:
            continue
        xs.append(float(r["flux_mjy"]) * float(pb))
        ys.append(float(peak))
    if len(xs) < 1:
        return None, len(xs)
    x = np.asarray(xs)
    y = np.asarray(ys)
    sxx = float(np.dot(x, x))
    if sxx <= 0.0:
        return None, len(xs)
    k = float(np.dot(x, y) / sxx)
    if not np.isfinite(k) or k <= 0.0:
        return None, len(xs)
    return k, len(xs)


def _nice_step(span_deg: float, *, n_target: int = 4) -> float:
    """Pick a 1/2/2.5/5×10ⁿ step giving ~n_target divisions of span."""
    raw = span_deg / max(n_target, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if mult * mag >= raw:
            return float(mult * mag)
    return float(10.0 * mag)


def render_annotated_png(
    png_path: Path,
    image: np.ndarray,
    *,
    median: float,
    sigma: float,
    ra0_deg: float,
    dec0_deg: float,
    fov_rad: float,
    ts: float,
    used_chgroups: list[int],
    veto_rows: Optional[list[dict[str, Any]]] = None,
    nvss_rows: Optional[list[dict[str, Any]]] = None,
    noise: Optional[dict[str, Any]] = None,
    astrom_dlm_rad: Optional[tuple[float, float]] = None,
) -> None:
    """Write the sigma-stretched greyscale frame with an RA/Dec grid
    and a colorbar (2026-06-09 request; NVSS detection markers
    re-added 2026-07-18 now that the data-time alignment makes the
    associations trustworthy).

    Axes are (l, m) offsets in degrees about the meridian phase center
    (origin='lower': north up, east RIGHT — instrument frame). The
    RA/Dec graticule is SIN-deprojected through
    :mod:`sky_astrometry`; labels are RA hh:mm and Dec degrees.

    ``veto_rows`` (2026-06-14): active sidereal (l,m) dump-veto regions
    from the C2 registry (``/mon/c2/sidereal_vetos``). Each is drawn as
    a red tolerance-radius circle + index label so the operator can see
    which sky positions are currently suppressing dumps.
    """
    fov_deg = float(np.rad2deg(fov_rad))
    half = fov_deg / 2.0
    img_sigma = (image - median) / sigma

    fig = Figure(figsize=(8.0, 7.8), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        img_sigma, cmap="gray",
        vmin=PNG_VMIN_SIGMA, vmax=PNG_VMAX_SIGMA,
        origin="lower", extent=(-half, half, -half, half),
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("σ (robust per-frame MAD)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel("l offset (deg) — east →")
    ax.set_ylabel("m offset (deg) — north ↑")

    # ---- RA/Dec graticule ------------------------------------------------
    grid_kw = dict(color="#46d4ff", lw=0.6, alpha=0.55, zorder=3)
    lbl_kw = dict(color="#46d4ff", fontsize=7, alpha=0.9, zorder=4)
    cosd = max(np.cos(np.deg2rad(dec0_deg)), 1e-3)
    # Astrometric self-cal offset, applied to the sky→image mapping
    # (same correction the source predictions carry).
    dl_off_deg = float(np.rad2deg((astrom_dlm_rad or (0.0, 0.0))[0]))
    dm_off_deg = float(np.rad2deg((astrom_dlm_rad or (0.0, 0.0))[1]))

    dec_step = _nice_step(fov_deg)
    dec_vals = np.arange(
        np.floor((dec0_deg - half) / dec_step) * dec_step,
        dec0_deg + half + dec_step, dec_step,
    )
    ra_span = fov_deg / cosd
    ra_step = _nice_step(ra_span)
    ra_vals = np.arange(
        np.floor((ra0_deg - ra_span / 2.0) / ra_step) * ra_step,
        ra0_deg + ra_span / 2.0 + ra_step, ra_step,
    )
    ra_samp = np.linspace(ra0_deg - ra_span, ra0_deg + ra_span, 241)
    dec_samp = np.linspace(dec0_deg - fov_deg, dec0_deg + fov_deg, 241)

    for dv in dec_vals:                                  # constant-Dec curves
        l, m = sky_astrometry.radec_to_lm(
            ra_samp, np.full_like(ra_samp, dv),
            ra0_deg=ra0_deg, dec0_deg=dec0_deg,
        )
        l, m = sky_astrometry.sky_to_instrument_lm(
            l, m, dec0_deg=dec0_deg,
        )
        ld = np.rad2deg(l) + dl_off_deg
        md = np.rad2deg(m) + dm_off_deg
        ax.plot(ld, md, **grid_kw)
        ok = (np.abs(ld) < half) & (np.abs(md) < half * 0.98)
        if ok.any():
            i = np.flatnonzero(ok)[0]                    # leftmost in-frame
            ax.text(max(ld[i], -half * 0.99), md[i], f" {dv:+.2f}°",
                    va="bottom", ha="left", **lbl_kw)
    for rv in ra_vals:                                   # constant-RA curves
        l, m = sky_astrometry.radec_to_lm(
            np.full_like(dec_samp, rv), dec_samp,
            ra0_deg=ra0_deg, dec0_deg=dec0_deg,
        )
        l, m = sky_astrometry.sky_to_instrument_lm(
            l, m, dec0_deg=dec0_deg,
        )
        ld = np.rad2deg(l) + dl_off_deg
        md = np.rad2deg(m) + dm_off_deg
        ax.plot(ld, md, **grid_kw)
        ok = (np.abs(ld) < half * 0.98) & (np.abs(md) < half)
        if ok.any():
            i = np.flatnonzero(ok)[np.argmin(md[ok])]    # bottom-most
            h = (rv % 360.0) / 15.0
            hh = int(h)
            mm = (h - hh) * 60.0
            ax.text(ld[i], max(md[i], -half * 0.99), f"{hh:02d}h{mm:04.1f}m",
                    va="bottom", ha="center", rotation=90, **lbl_kw)

    # ---- sidereal (l,m) dump-veto regions -------------------------------
    if veto_rows:
        from matplotlib.patches import Circle  # local; keep top-level light
        for i, v in enumerate(veto_rows):
            try:
                lx = float(np.rad2deg(float(v["l_rad"])))
                my = float(np.rad2deg(float(v["m_rad"])))
                r_deg = float(np.rad2deg(float(v.get("tol_rad", 0.0))))
            except (KeyError, TypeError, ValueError):
                continue
            if abs(lx) > half or abs(my) > half:
                continue  # veto centre outside this FOV
            ax.add_patch(Circle(
                (lx, my), max(r_deg, 0.01),
                fill=False, edgecolor="#ff5050", lw=1.2,
                alpha=0.85, zorder=5,
            ))
            ax.plot([lx], [my], "+", color="#ff5050", ms=6,
                    mew=1.2, zorder=6)
            ax.text(lx, my + max(r_deg, 0.01),
                    f"V{i + 1}", color="#ff5050", fontsize=7,
                    va="bottom", ha="center", zorder=6)

    # ---- NVSS markers (2026-07-18) ---------------------------------------
    # Detections (snr >= NVSS_DETECT_SNR): solid lime circle + name.
    # Bright expected-but-undetected sources (apparent flux would give
    # snr >= threshold at the frame's flux scale): dashed orange circle,
    # so a sensitivity/pointing regression is visible at a glance.
    if nvss_rows:
        k = (noise or {}).get("flux_scale_units_per_mjy")
        for r in nvss_rows:
            try:
                # Instrument-frame coords (the image axes); fall back
                # to true (l, m) for pre-2026-07-19 sidecars.
                lx = float(np.rad2deg(float(
                    r.get("l_img_rad", r["l_rad"]))))
                my = float(np.rad2deg(float(
                    r.get("m_img_rad", r["m_rad"]))))
            except (KeyError, TypeError, ValueError):
                continue
            if abs(lx) > half or abs(my) > half:
                continue
            snr = r.get("snr")
            name = str(r.get("name", "")).replace("NVSS ", "")
            if r.get("detected"):
                ax.plot(
                    [lx], [my], "o", ms=11, mfc="none", mec="#7CFC00",
                    mew=1.1, alpha=0.9, zorder=6,
                )
                ax.text(
                    lx, my + 0.035, f"{name} ({snr:.0f}σ)",
                    color="#7CFC00", fontsize=6.5, va="bottom",
                    ha="center", alpha=0.95, zorder=6,
                )
            elif (
                k is not None and snr is not None
                and float(r.get("flux_app_mjy", 0.0)) * k / sigma
                >= NVSS_DETECT_SNR
            ):
                ax.plot(
                    [lx], [my], "o", ms=11, mfc="none", mec="#ffa040",
                    mew=1.0, ls="", alpha=0.8, zorder=6,
                )
                ax.text(
                    lx, my + 0.035, f"{name} (exp {snr:.0f}σ)",
                    color="#ffa040", fontsize=6.5, va="bottom",
                    ha="center", alpha=0.9, zorder=6,
                )

    utc = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    n_det = sum(1 for r in (nvss_rows or []) if r.get("detected"))
    title = (
        f"{utc} UTC   |   center RA {ra0_deg / 15.0:.4f} h  "
        f"Dec {dec0_deg:+.3f}°   |   {len(used_chgroups)}/16 chgroups"
    )
    if noise is not None and noise.get("sigma_mjy") is not None:
        title += (
            f"\nσ = {noise['sigma_mjy']:.1f} mJy   |   implied per-ant "
            f"SEFD ≈ {noise['sefd_implied_jy'] / 1e3:.1f} kJy "
            f"(ref {noise['ref_sefd_jy'] / 1e3:.0f} kJy → "
            f"{noise['sigma_pred_mjy_at_ref_sefd']:.1f} mJy)   |   "
            f"NVSS: {n_det} detected "
            f"(scale from {noise['n_flux_scale_sources']} src)"
        )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(png_path), facecolor="white")


# ---------------------------------------------------------------------------
# Frame store
# ---------------------------------------------------------------------------


@dataclass
class SkyFrameStore:
    """Disk layout + retention for the sky-monitor frames.

    ``root/frames/YYYYMMDD/sky_<ts>_n<ncg>.{png,npz}`` (UTC days).
    """

    root: Path = field(default_factory=lambda: SKY_MONITOR_ROOT)
    retention_h: float = 48.0

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @staticmethod
    def _day(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")

    def write_frame(
        self,
        image: np.ndarray,
        *,
        ts: float,
        median: float,
        sigma: float,
        used_chgroups: list[int],
        meta: dict[str, Any],
        annotate: Optional[dict[str, Any]] = None,
        uv_grid: Optional[np.ndarray] = None,
    ) -> tuple[Path, Path]:
        """Write the PNG (sigma-stretched greyscale) + NPZ (raw float32
        image + full metadata). Returns ``(png_path, npz_path)``.

        ``annotate`` (2026-06-09): when provided
        (``{ra0_deg, dec0_deg, fov_rad, nvss_rows}``), the PNG becomes
        an annotated figure with an RA/Dec graticule + colorbar
        (:func:`render_annotated_png`) and a ``<stem>.json`` sidecar is
        written carrying the in-FOV NVSS source list for the page-side
        table; falls back to the bare greyscale dump if the figure
        render throws. The NPZ raw image is unchanged either way.
        """
        day_dir = self.frames_dir / self._day(ts)
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sky_{int(ts)}_n{len(used_chgroups)}"
        png_path = day_dir / f"{stem}.png"
        npz_path = day_dir / f"{stem}.npz"

        wrote_png = False
        if annotate is not None:
            try:
                render_annotated_png(
                    png_path, image,
                    median=median, sigma=sigma,
                    ra0_deg=float(annotate["ra0_deg"]),
                    dec0_deg=float(annotate["dec0_deg"]),
                    fov_rad=float(annotate["fov_rad"]),
                    ts=ts, used_chgroups=used_chgroups,
                    veto_rows=annotate.get("veto_rows"),
                    nvss_rows=annotate.get("nvss_rows"),
                    noise=annotate.get("noise"),
                    astrom_dlm_rad=annotate.get("astrom_dlm_rad"),
                )
                wrote_png = True
            except Exception:                              # noqa: BLE001
                LOG.exception(
                    "annotated frame render failed; falling back to "
                    "bare greyscale",
                )
            try:
                sidecar = {
                    "ra0_deg": float(annotate["ra0_deg"]),
                    "dec0_deg": float(annotate["dec0_deg"]),
                    "fov_deg": float(np.rad2deg(float(annotate["fov_rad"]))),
                    "nvss_min_mjy": NVSS_MIN_MJY,
                    "nvss_detect_snr": NVSS_DETECT_SNR,
                    "nvss": annotate.get("nvss_rows", []),
                    "sidereal_vetos": annotate.get("veto_rows", []),
                    "noise": annotate.get("noise"),
                    "astrom_check": annotate.get("astrom_check"),
                }
                (day_dir / f"{stem}.json").write_text(
                    json.dumps(sidecar), encoding="utf-8",
                )
            except OSError:
                LOG.warning("could not write frame sidecar %s.json", stem)
        if not wrote_png:
            img_sigma = (image - median) / sigma
            # origin='lower' so +m (north) is up, matching the cube
            # explorer notebook's imshow convention.
            mpimg.imsave(
                str(png_path),
                img_sigma,
                cmap="gray",
                vmin=PNG_VMIN_SIGMA,
                vmax=PNG_VMAX_SIGMA,
                origin="lower",
            )
        extra: dict[str, Any] = {}
        if uv_grid is not None:
            # Combined (clipped, aligned) UV grid — ~0.5 MB compressed
            # per frame; enables offline flag/weighting experiments and
            # per-cell forensics without re-capturing snapshots.
            extra["uv_grid"] = np.asarray(uv_grid, dtype=np.complex64)
        if annotate is not None and annotate.get("raw_snapshots"):
            # Raw per-chgroup sparse snapshots (~1.3 MB/frame
            # compressed): lets offline analysis re-run the combine
            # with different clip/align/weighting choices.
            for s in annotate["raw_snapshots"]:
                cg = int(s["meta"]["chgroup"])
                extra[f"cg{cg:02d}_vis"] = s["vis"]
                extra[f"cg{cg:02d}_ix_row"] = s["ix_row"]
                extra[f"cg{cg:02d}_ix_col"] = s["ix_col"]
                extra[f"cg{cg:02d}_meta_json"] = np.bytes_(
                    json.dumps(s["meta"]).encode("utf-8"),
                )
        np.savez_compressed(
            npz_path,
            image=image.astype(np.float32),
            median=np.float64(median),
            sigma=np.float64(sigma),
            used_chgroups=np.asarray(used_chgroups, dtype=np.int16),
            meta_json=np.bytes_(json.dumps(meta).encode("utf-8")),
            **extra,
        )
        return png_path, npz_path

    def list_frames(self, *, since_unix: float) -> list[dict[str, Any]]:
        """Frames newer than ``since_unix``, ascending in time.

        Pure directory scan + filename parse; never opens the NPZs.
        """
        out: list[dict[str, Any]] = []
        frames_dir = self.frames_dir
        if not frames_dir.exists():
            return out
        # Only scan day dirs that can contain frames in range.
        first_day = self._day(since_unix)
        for day_dir in sorted(frames_dir.iterdir()):
            if not day_dir.is_dir() or day_dir.name < first_day:
                continue
            for p in day_dir.iterdir():
                m = _FRAME_RE.match(p.name)
                if m is None or not p.name.endswith(".png"):
                    continue
                ts = int(m.group("ts"))
                if ts < since_unix:
                    continue
                out.append({
                    "ts": ts,
                    "n_chgroups": int(m.group("ncg")),
                    "day": day_dir.name,
                    "png": p.name,
                })
        out.sort(key=lambda d: d["ts"])
        return out

    def resolve_png(self, day: str, name: str) -> Optional[Path]:
        """Validated path for serving. Returns None on any funny
        business (path separators, non-matching names)."""
        if not re.fullmatch(r"\d{8}", day) or _FRAME_RE.match(name) is None:
            return None
        if not name.endswith(".png"):
            return None
        p = self.frames_dir / day / name
        return p if p.is_file() else None

    def resolve_sidecar(self, day: str, png_name: str) -> Optional[Path]:
        """Validated path of the ``.json`` sidecar for a frame PNG.
        Returns None when missing (e.g. pre-overlay frames)."""
        png = self.resolve_png(day, png_name)
        if png is None:
            return None
        p = png.with_suffix(".json")
        return p if p.is_file() else None

    def prune(self, *, now: float) -> int:
        """Delete frames (and empty day dirs) older than retention.
        Returns the number of files removed. Cheap: only day dirs at
        or before the cutoff day are scanned.
        """
        cutoff = now - self.retention_h * 3600.0
        cutoff_day = self._day(cutoff)
        n_removed = 0
        frames_dir = self.frames_dir
        if not frames_dir.exists():
            return 0
        for day_dir in sorted(frames_dir.iterdir()):
            if not day_dir.is_dir() or day_dir.name > cutoff_day:
                continue
            for p in list(day_dir.iterdir()):
                m = _FRAME_RE.match(p.name)
                if m is None:
                    continue
                if int(m.group("ts")) < cutoff:
                    try:
                        p.unlink()
                        n_removed += 1
                    except OSError:
                        LOG.warning("sky prune: cannot unlink %s", p)
            try:
                next(day_dir.iterdir())
            except StopIteration:
                try:
                    day_dir.rmdir()
                except OSError:
                    pass
        return n_removed


# ---------------------------------------------------------------------------
# The monitor (ingest → combine → frame)
# ---------------------------------------------------------------------------


class SkyMonitor:
    """Thread-safe ingest + frame builder.

    Frames are built lazily from ingest calls (no background thread):
    whenever a snapshot arrives and ``frame_interval_s`` has elapsed
    since the last frame, we combine every snapshot fresher than
    ``freshness_s`` and write a frame. If the corr fleet stops
    posting, frames simply stop — which is itself the signal the
    monitor exists to surface.

    Args:
        store: frame store (defaults to /dataz root).
        frame_interval_s: target movie cadence (30 s production).
        freshness_s: a chgroup snapshot older than this is excluded
            from new frames (3× the interval: one missed POST is
            tolerated, a dead node ages out).
        min_chgroups: minimum fresh chgroups to bother writing a
            frame. 1 by default — a partial sky is more useful than
            no sky, and the per-frame chgroup count is surfaced in
            the UI.
        oversample: UV zero-padding factor before the iFFT (exact
            band-limited interpolation; see
            :func:`dirty_image_from_uv`). 2 ⇒ 512×512 frames over the
            same FOV. Display-only anti-aliasing; the gridding itself
            is untouched (corr-side, kernel_support=1).
        grid_correct: divide out the nearest-cell (pillbox) gridding
            envelope so edge-of-FOV sources display at true relative
            strength (:func:`pillbox_grid_correction`).
    """

    def __init__(
        self,
        store: SkyFrameStore | None = None,
        sefd_publisher: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        frame_interval_s: float = 30.0,
        freshness_s: float = 90.0,
        min_chgroups: int = 1,
        n_grid: int = 256,
        oversample: int = 2,
        grid_correct: bool = True,
        nvss_enabled: bool = True,
        veto_provider: Optional[Callable[[], list[dict[str, Any]]]] = None,
        armed_mjd_provider: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self.store = store if store is not None else SkyFrameStore()
        # 2026-08-07: optional sink for the per-frame implied SEFD. Injected
        # rather than imported so this module keeps no etcd dependency and
        # stays unit-testable; app.py wires it to ControlStore.put_dict.
        self._sefd_publisher = sefd_publisher
        # 2026-06-14: returns the active sidereal (l,m) dump-veto regions
        # (from C2's /mon/c2/sidereal_vetos) to overlay on each frame.
        self._veto_provider = veto_provider
        # 2026-07-18: returns the fleet capture-arm MJD (etcd
        # /mon/snap/1/armed_mjd) — the absolute time base that converts
        # each snapshot's block_n to its exact capture time. None /
        # failures fall back to corr wall clocks minus the measured
        # export lag (EXPORT_LAG_FALLBACK_S).
        self._armed_mjd_provider = armed_mjd_provider
        self.frame_interval_s = float(frame_interval_s)
        self.freshness_s = float(freshness_s)
        self.min_chgroups = int(min_chgroups)
        self.n_grid = int(n_grid)
        self.oversample = max(1, int(oversample))
        self.grid_correct = bool(grid_correct)
        # NVSS overlay catalog: parsed once on a daemon thread (~10 s
        # for the 260 MB tdat, then npz-cached under the store root);
        # frames built before it lands simply omit the overlay.
        self.nvss_enabled = bool(nvss_enabled)
        self._nvss = sky_astrometry.NvssCatalog(
            min_mjy=NVSS_MIN_MJY, cache_dir=self.store.root,
        )
        if self.nvss_enabled:
            self._nvss.start_loading()

        self._lock = threading.Lock()
        self._latest: dict[int, dict[str, Any]] = {}     # chgroup → snapshot
        self._recv_unix: dict[int, float] = {}           # chgroup → ingest time
        self._last_frame_unix: float = 0.0
        self.n_ingested = 0
        self.n_rejected = 0
        self.n_frames = 0
        # Instrument-static baseline history (2026-07-19): per chgroup,
        # the last STATIC_SUB_MAXLEN snapshot vis vectors. Seeded from
        # the stored frames so a dashboard restart doesn't cost a
        # 5-minute warmup.
        from collections import deque
        self._hist: dict[int, Any] = {}
        self._deque = deque
        # Astrometric self-cal state (rad, image-frame l/m): EMA of the
        # per-frame xcorr residual, applied to predictions + graticule.
        self._astrom_dl_rad: float = 0.0
        self._astrom_dm_rad: float = 0.0
        try:
            self._seed_history_from_store()
        except Exception:                                # noqa: BLE001
            LOG.exception("static-sub history seed failed (cold start)")

    #: etcd key for the static-sky SEFD rollup. Single-instance (the sky
    #: monitor runs only on h23) so there is no cn/half in the path, and it
    #: sits under /mon/sky/ which the influx pusher scans as its own prefix.
    SEFD_MON_KEY = "/mon/sky/sefd"

    def _publish_sefd(self, noise: dict[str, Any], *, ts: float) -> None:
        """Publish the frame's implied SEFD to etcd for the influx pusher.

        Deliberately a no-op unless the frame is actually usable:
        ``static_sub_ready`` false means the instrumental baseline has not
        been subtracted, the image is structure-dominated and the flux scale
        fitted against it is meaningless -- ``sefd_implied_jy`` is None in
        that case. Publishing nothing leaves a gap in Grafana, which is the
        honest rendering; publishing a zero would draw a cliff.

        Never raises: a monitoring sink must not be able to kill the frame
        loop that feeds the sky tab.
        """
        pub = self._sefd_publisher
        if pub is None:
            return
        sefd = noise.get("sefd_implied_jy")
        if sefd is None or not np.isfinite(float(sefd)):
            return
        payload = dict(noise)
        payload["ts_unix"] = float(ts)
        # bool is not useful as an influx field here and the pusher skips it;
        # keep it in the payload for anyone reading etcd directly.
        try:
            pub(self.SEFD_MON_KEY, payload)
        except Exception:                                   # noqa: BLE001
            LOG.warning(
                "sky SEFD publish to %s failed (frame loop continues)",
                self.SEFD_MON_KEY, exc_info=True,
            )

    def _seed_history_from_store(self) -> None:
        """Prime the static-sub history from raw snapshots persisted in
        recent frame NPZs (best-effort, ascending time)."""
        frames_dir = self.store.frames_dir
        if not frames_dir.exists():
            return
        paths: list[Path] = []
        for day_dir in sorted(frames_dir.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            paths.extend(sorted(day_dir.glob("sky_*.npz"), reverse=True))
            if len(paths) >= STATIC_SUB_MAXLEN:
                break
        n_seeded = 0
        for p in reversed(paths[:STATIC_SUB_MAXLEN]):
            try:
                with np.load(p, allow_pickle=False) as z:
                    for cg in range(16):
                        k = f"cg{cg:02d}_vis"
                        if k not in z.files:
                            continue
                        self._push_history(cg, np.asarray(z[k]))
                        n_seeded += 1
            except Exception:                            # noqa: BLE001
                continue
        if n_seeded:
            LOG.info(
                "static-sub history seeded: %d snapshots from %d frames",
                n_seeded, len(paths[:STATIC_SUB_MAXLEN]),
            )

    def _push_history(self, cg: int, vis: np.ndarray) -> None:
        """Append a snapshot's vis to the chgroup history; reset the
        history when the pattern length changes (re-prepare)."""
        dq = self._hist.get(cg)
        if dq is None or (len(dq) and dq[-1].shape != vis.shape):
            dq = self._deque(maxlen=STATIC_SUB_MAXLEN)
            self._hist[cg] = dq
        dq.append(np.asarray(vis, dtype=np.complex64))

    def _static_subtract(
        self, fresh: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[int, int], bool]:
        """Return snapshots with the per-cell temporal-median baseline
        removed (where enough history exists), the per-chgroup history
        depth, and whether ALL fresh chgroups were subtracted."""
        out: list[dict[str, Any]] = []
        n_hist: dict[int, int] = {}
        all_sub = True
        for s in fresh:
            cg = int(s["meta"]["chgroup"])
            dq = self._hist.get(cg)
            n = len(dq) if dq is not None else 0
            n_hist[cg] = n
            if (
                dq is not None
                and n >= STATIC_SUB_MIN_HIST
                and dq[-1].shape == s["vis"].shape
            ):
                baseline = complex_median(np.stack(dq))
                s = dict(s)
                s["vis"] = (s["vis"] - baseline).astype(np.complex64)
            else:
                all_sub = False
            out.append(s)
        return out, n_hist, all_sub

    # -- ingest --------------------------------------------------------

    def ingest(self, body: bytes, *, now: float | None = None) -> dict[str, Any]:
        """Parse + store one snapshot; maybe build a frame.

        Returns a JSON-ready ack:
        ``{ok, chgroup, frame_written: bool, n_fresh}``.
        Raises ``ValueError`` on malformed payloads (route returns 400).
        """
        if now is None:
            now = time.time()
        snap = parse_snapshot_npz(body)
        cg = int(snap["meta"]["chgroup"])
        if not (0 <= cg <= 15):
            self.n_rejected += 1
            raise ValueError(f"chgroup={cg} out of range 0..15")

        frame_written = False
        with self._lock:
            self._latest[cg] = snap
            self._recv_unix[cg] = now
            self._push_history(cg, snap["vis"])
            self.n_ingested += 1
            fresh = self._fresh_snapshots_locked(now)
            due = (now - self._last_frame_unix) >= self.frame_interval_s
            if due and len(fresh) >= self.min_chgroups:
                try:
                    self._build_frame_locked(fresh, now)
                    frame_written = True
                except Exception:                       # noqa: BLE001
                    LOG.exception("sky frame build failed (continuing)")
        return {
            "ok": True,
            "chgroup": cg,
            "frame_written": frame_written,
            "n_fresh": len(fresh),
        }

    def _fresh_snapshots_locked(self, now: float) -> list[dict[str, Any]]:
        return [
            self._latest[cg]
            for cg in sorted(self._latest)
            if now - self._recv_unix[cg] <= self.freshness_s
        ]

    # -- frame build ----------------------------------------------------

    def _build_frame_locked(
        self, fresh: list[dict[str, Any]], now: float,
    ) -> None:
        # ---- data-time alignment (2026-07-18 astrometry fix) -----------
        # The 16 chgroup snapshots are up to ~30 s apart in DATA time;
        # the drift-scan phase center moves 15″/s of RA, so an unaligned
        # combine smears sources into trails and the frame's phase
        # center is ill-defined. Compute each snapshot's data mid-time
        # (capture-arm anchor via etcd; corr-wall-clock fallback), pick
        # the newest as the frame's reference, and translate every other
        # chgroup to it with a UV phase ramp.
        armed_mjd: Optional[float] = None
        if self._armed_mjd_provider is not None:
            try:
                v = self._armed_mjd_provider()
                if v is not None and np.isfinite(float(v)):
                    armed_mjd = float(v)
            except Exception:                           # noqa: BLE001
                LOG.exception("armed_mjd provider failed (fallback)")
        armed_unix = (
            (armed_mjd - _MJD_UNIX_EPOCH) * 86400.0
            if armed_mjd is not None else None
        )
        t_data: dict[int, float] = {}
        dec0_pre = 0.0
        for s in fresh:
            cg = int(s["meta"]["chgroup"])
            t_data[cg] = snapshot_data_mid_unix(
                s["meta"], armed_unix=armed_unix,
            )
            dec0_pre = float(s["meta"].get("dec_deg", dec0_pre) or dec0_pre)
        t_ref = max(t_data.values()) if t_data else now
        align_span_s = (
            (t_ref - min(t_data.values())) if t_data else 0.0
        )
        cos_dec0 = float(np.cos(np.deg2rad(dec0_pre)))
        align_dl_rad = {
            cg: -cos_dec0
            * sky_astrometry.SIDEREAL_RATE_RAD_PER_S
            * (t_ref - t_cg)
            for cg, t_cg in t_data.items()
        }
        # Instrument-static baseline removal (see STATIC_SUB_MIN_HIST).
        fresh_sub, static_hist, static_sub_ready = self._static_subtract(
            fresh,
        )
        uv, used, clip_stats = combine_chgroups_to_uv(
            fresh_sub, n_grid=self.n_grid, align_dl_rad=align_dl_rad,
        )
        if not used:
            return
        # Sigma is estimated on the UNCORRECTED image: grid correction
        # amplifies edge/corner noise (×π/2 .. ×2.47), and folding that
        # into a global MAD would dim phase-center sources in the σ
        # stretch. Estimating first keeps center-source σ values
        # comparable to pre-correction frames; corrected edge sources
        # then display at their true relative strength.
        image = dirty_image_from_uv(uv, oversample=self.oversample)
        median, sigma = robust_sigma(image)
        if self.grid_correct:
            # Correct the median-subtracted signal so the stored median
            # stays valid and σ-units scale exactly by the correction.
            corr = pillbox_grid_correction(image.shape[0])
            image = (image - np.float32(median)) * corr + np.float32(median)

        # Frame metadata: enough to re-derive pixel scale + provenance.
        cell_lambdas = sorted({
            round(float(s["meta"].get("cell_lambda", 0.0)), 9)
            for s in fresh
        })
        dec_degs = sorted({
            round(float(s["meta"].get("dec_deg", 0.0)), 4) for s in fresh
        })

        # ---- Astrometry + NVSS overlay (2026-06-09) -------------------
        # Un-fringestopped vis ⇒ phase center = meridian at obs_dec at
        # the frame's DATA reference time (2026-07-18 fix — was the
        # frame-build wall time, up to ~30 s late = ~20 px of RA):
        # (α₀, δ₀) = (LST(t_ref), dec). FOV = 1/cell_lambda rad.
        annotate: Optional[dict[str, Any]] = None
        astro: dict[str, Any] = {}
        try:
            if cell_lambdas and cell_lambdas[0] > 0:
                fov_rad = 1.0 / float(cell_lambdas[0])
                # Apparent-frame pointing dec: sets the instrument
                # m-axis compression (geometry of the ΔN baselines).
                dec0_apparent = float(dec_degs[0]) if dec_degs else 0.0
                # J2000/ICRS phase center for the catalog + graticule
                # (2026-07-19 fix: without the TETE→ICRS epoch
                # transform, 26 yr of precession put every NVSS
                # prediction ~45 px east + a cos(α)-drifting Dec
                # offset).
                ra0, dec0 = sky_astrometry.phase_center_icrs(
                    t_ref, dec0_apparent,
                )
                n_pix = int(image.shape[0])
                pix_arcsec = np.rad2deg(fov_rad / n_pix) * 3600.0
                window_s = max(
                    float(fresh[0]["meta"].get("window_s") or 0.0),
                    8 * BLOCK_S * 0.999,
                )
                nvss_rows: list[dict[str, Any]] = []
                cat = self._nvss.get() if self.nvss_enabled else None
                if cat is not None:
                    # Select with a margin: the instrument m-axis is
                    # COMPRESSED by cos(lat − dec), so sources with
                    # |m_true| slightly beyond the nominal half-FOV
                    # still land inside the image.
                    sel = sky_astrometry.select_in_fov(
                        cat, ra0_deg=ra0, dec0_deg=dec0,
                        fov_rad=fov_rad * 1.08,
                        max_sources=NVSS_MAX_SOURCES,
                    )
                    r_pix = NVSS_APERTURE_ARCSEC / pix_arcsec
                    for i in range(sel["ra_deg"].size):
                        # True SIN (l, m) → instrument image frame
                        # (m compressed by cos(lat−dec) + w-term;
                        # 2026-07-19 fix — this was the varying-Dec
                        # offset reported on the sky tab).
                        l_img, m_img = sky_astrometry.sky_to_instrument_lm(
                            sel["l_rad"][i], sel["m_rad"][i],
                            dec0_deg=dec0_apparent,
                        )
                        # Astrometric self-cal correction (EMA of the
                        # xcorr residual from previous frames).
                        l_img = float(l_img) + self._astrom_dl_rad
                        m_img = float(m_img) + self._astrom_dm_rad
                        if (abs(float(l_img)) > fov_rad / 2.0
                                or abs(float(m_img)) > fov_rad / 2.0):
                            continue
                        row, col = sky_astrometry.lm_to_pix(
                            l_img, m_img,
                            n_pix=n_pix, fov_rad=fov_rad,
                        )
                        snr, peak, row_pk, col_pk = measure_source_peak(
                            image, row=float(row), col=float(col),
                            median=median, sigma=sigma, radius_pix=r_pix,
                        )
                        theta = float(np.hypot(
                            sel["l_rad"][i], sel["m_rad"][i],
                        ))
                        pb = float(sky_astrometry.pb_resp_power(theta))
                        detected = bool(
                            np.isfinite(snr) and snr >= NVSS_DETECT_SNR
                        )
                        nvss_rows.append({
                            "name": str(sel["name"][i]),
                            "ra_deg": float(sel["ra_deg"][i]),
                            "dec_deg": float(sel["dec_deg"][i]),
                            "flux_mjy": float(sel["flux_mjy"][i]),
                            "l_rad": float(sel["l_rad"][i]),
                            "m_rad": float(sel["m_rad"][i]),
                            "l_img_rad": float(l_img),
                            "m_img_rad": float(m_img),
                            "row": float(row),
                            "col": float(col),
                            "snr": (float(snr) if np.isfinite(snr) else None),
                            "peak": (
                                float(peak) if np.isfinite(peak) else None
                            ),
                            "row_peak": (
                                float(row_pk) if np.isfinite(row_pk) else None
                            ),
                            "col_peak": (
                                float(col_pk) if np.isfinite(col_pk) else None
                            ),
                            "pb": pb,
                            "flux_app_mjy": float(sel["flux_mjy"][i]) * pb,
                            "detected": detected,
                        })
                # Active sidereal (l,m) dump-veto regions (C2 registry).
                veto_rows: list[dict[str, Any]] = []
                if self._veto_provider is not None:
                    try:
                        for v in (self._veto_provider() or []):
                            l_rad = float(v.get("l_rad"))
                            m_rad = float(v.get("m_rad"))
                            row, col = sky_astrometry.lm_to_pix(
                                l_rad, m_rad, n_pix=n_pix, fov_rad=fov_rad,
                            )
                            veto_rows.append({
                                "l_rad": l_rad,
                                "m_rad": m_rad,
                                "tol_rad": float(v.get("tol_rad", 0.0)),
                                "n_hits": int(v.get("n_hits", 0)),
                                "last_hit_unix": float(
                                    v.get("last_hit_unix", 0.0)
                                ),
                                "row": float(row),
                                "col": float(col),
                            })
                    except Exception:                       # noqa: BLE001
                        LOG.exception("veto overlay build failed (skipped)")
                # ---- Noise in physical units + implied SEFD ----------
                # Flux scale from detected NVSS sources (peak image
                # units per apparent mJy), then σ_MAD → mJy and the
                # per-antenna SEFD that would produce that noise.
                if static_sub_ready:
                    k_units_per_mjy, n_flux_src = fit_flux_scale(nvss_rows)
                else:
                    # Without the instrumental-baseline subtraction the
                    # image is structure-dominated; a flux scale fit
                    # against it would be meaningless.
                    k_units_per_mjy, n_flux_src = None, 0
                sigma_pred_mjy = sefd_predicted_sigma_mjy(
                    SEFD_REFERENCE_JY, window_s=window_s,
                )
                sigma_mjy: Optional[float] = None
                sefd_implied_jy: Optional[float] = None
                if k_units_per_mjy is not None:
                    sigma_mjy = float(sigma / k_units_per_mjy)
                    sefd_implied_jy = float(
                        SEFD_REFERENCE_JY * sigma_mjy / sigma_pred_mjy
                    )
                # Per-frame astrometric self-check: cross-correlate the
                # image against a flux-weighted delta map at the
                # predicted NVSS pixels. A significant peak away from
                # (0, 0) means the mapping has drifted (bad cal epoch,
                # clock, pointing...) — this is the check that catches
                # what apertures at wrong positions silently miss.
                astrom_check: Optional[dict[str, Any]] = None
                pix_rad = fov_rad / n_pix
                if len(nvss_rows) >= 5:
                    try:
                        astrom_check = measure_astrometric_offset(
                            image, median=median, sigma=sigma,
                            nvss_rows=nvss_rows,
                        )
                    except Exception:                    # noqa: BLE001
                        LOG.exception("astrometric self-check failed")
                # Self-cal feedback: fold a confident residual into the
                # EMA correction (applied to NEXT frames' predictions;
                # predictions in THIS frame already carry the previous
                # correction, so astrom_check here is the residual).
                if (
                    astrom_check is not None
                    and astrom_check["z"] >= ASTROM_SELFCAL_MIN_Z
                    and astrom_check["drow_px"] is not None
                    and abs(astrom_check["drow_px"]) <= ASTROM_SELFCAL_MAX_PX
                    and abs(astrom_check["dcol_px"]) <= ASTROM_SELFCAL_MAX_PX
                ):
                    cap = ASTROM_SELFCAL_MAX_PX * pix_rad
                    self._astrom_dl_rad = float(np.clip(
                        self._astrom_dl_rad
                        + ASTROM_SELFCAL_GAIN
                        * astrom_check["dcol_px"] * pix_rad,
                        -cap, cap,
                    ))
                    self._astrom_dm_rad = float(np.clip(
                        self._astrom_dm_rad
                        + ASTROM_SELFCAL_GAIN
                        * astrom_check["drow_px"] * pix_rad,
                        -cap, cap,
                    ))
                astrom_applied = {
                    "dl_px": round(self._astrom_dl_rad / pix_rad, 2),
                    "dm_px": round(self._astrom_dm_rad / pix_rad, 2),
                }
                noise = {
                    "static_sub_ready": static_sub_ready,
                    "sigma_mjy": sigma_mjy,
                    "flux_scale_units_per_mjy": k_units_per_mjy,
                    "n_flux_scale_sources": n_flux_src,
                    "window_s": window_s,
                    "sigma_pred_mjy_at_ref_sefd": sigma_pred_mjy,
                    "ref_sefd_jy": SEFD_REFERENCE_JY,
                    "sefd_implied_jy": sefd_implied_jy,
                    "n_ant_assumed": SEFD_N_ANT,
                    "bw_eff_hz": SEFD_BW_HZ,
                }
                self._publish_sefd(noise, ts=now)
                annotate = {
                    "ra0_deg": ra0, "dec0_deg": dec0,
                    "fov_rad": fov_rad, "nvss_rows": nvss_rows,
                    "veto_rows": veto_rows,
                    "noise": noise,
                    "astrom_check": astrom_check,
                    "astrom_dlm_rad": (
                        self._astrom_dl_rad, self._astrom_dm_rad,
                    ),
                    "raw_snapshots": fresh,
                }
                astro = {
                    "ra0_deg": ra0,
                    "dec0_deg": dec0,
                    "radec_epoch": "ICRS/J2000",
                    "dec0_apparent_deg": dec0_apparent,
                    "astrom_check": astrom_check,
                    "astrom_applied": astrom_applied,
                    "lst_h": sky_astrometry.lst_deg(t_ref) / 15.0,
                    "fov_deg": float(np.rad2deg(fov_rad)),
                    "pix_arcsec": pix_arcsec,
                    "nvss_min_mjy": NVSS_MIN_MJY,
                    "nvss_aperture_arcsec": NVSS_APERTURE_ARCSEC,
                    "nvss": nvss_rows,
                    "nvss_loaded": cat is not None,
                    "nvss_detect_snr": NVSS_DETECT_SNR,
                    "sidereal_vetos": veto_rows,
                    "noise": noise,
                    # Data-time provenance (2026-07-18 astrometry fix).
                    "t_data_unix": t_ref,
                    "data_lag_s": float(now - t_ref),
                    "align_span_s": float(align_span_s),
                    "armed_mjd": armed_mjd,
                }
        except Exception:                                  # noqa: BLE001
            LOG.exception("sky astrometry failed (frame without overlay)")
            annotate = None
        meta = {
            "ts": now,
            "used_chgroups": used,
            "n_grid": self.n_grid,
            "oversample": self.oversample,
            "grid_correct": self.grid_correct,
            "n_pix": int(image.shape[0]),
            "cell_lambda": cell_lambdas,
            "dec_deg": dec_degs,
            "median": median,
            "sigma": sigma,
            "png_vmin_sigma": PNG_VMIN_SIGMA,
            "png_vmax_sigma": PNG_VMAX_SIGMA,
            "uv_clip_k": UV_CLIP_K,
            "uv_clipped_frac": (
                float(sum(c["n_clipped"] for c in clip_stats.values()))
                / max(1, sum(c["n_cells"] for c in clip_stats.values()))
            ),
            "static_sub_ready": static_sub_ready,
            "static_sub_min_hist": STATIC_SUB_MIN_HIST,
            "static_sub_hist": {str(k): v for k, v in static_hist.items()},
            "per_chgroup": {
                str(int(s["meta"]["chgroup"])): {
                    "hostname": s["meta"].get("hostname"),
                    "block_n": s["meta"].get("block_n"),
                    "cubes_seen": s["meta"].get("cubes_seen"),
                    "amp_scale": s["meta"].get("amp_scale"),
                    "unix_ts": s["meta"].get("unix_ts"),
                    "t_data_unix": t_data.get(int(s["meta"]["chgroup"])),
                    "n_uv_clipped": clip_stats.get(
                        int(s["meta"]["chgroup"]), {},
                    ).get("n_clipped"),
                }
                for s in fresh
            },
            **astro,
        }
        self.store.write_frame(
            image, ts=now, median=median, sigma=sigma,
            used_chgroups=used, meta=meta, annotate=annotate,
            uv_grid=uv,
        )
        self._last_frame_unix = now
        self.n_frames += 1
        # Retention: piggyback on frame writes (~1 prune per 30 s).
        try:
            self.store.prune(now=now)
        except Exception:                               # noqa: BLE001
            LOG.exception("sky prune failed (continuing)")
        LOG.info(
            "sky frame #%d written: n_chgroups=%d sigma=%.4g",
            self.n_frames, len(used), sigma,
        )

    # -- status ----------------------------------------------------------

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        if now is None:
            now = time.time()
        with self._lock:
            per_cg = {
                str(cg): {
                    "age_s": round(now - self._recv_unix[cg], 1),
                    "hostname": self._latest[cg]["meta"].get("hostname"),
                    "cubes_seen": self._latest[cg]["meta"].get("cubes_seen"),
                }
                for cg in sorted(self._latest)
            }
            return {
                "ok": True,
                "n_ingested": self.n_ingested,
                "n_rejected": self.n_rejected,
                "n_frames": self.n_frames,
                "last_frame_unix": self._last_frame_unix,
                "last_frame_age_s": (
                    round(now - self._last_frame_unix, 1)
                    if self._last_frame_unix else None
                ),
                "chgroups": per_cg,
            }


__all__ = [
    "SKY_SNAPSHOT_VERSION",
    "SKY_MONITOR_ROOT",
    "PNG_VMIN_SIGMA",
    "PNG_VMAX_SIGMA",
    "NVSS_MIN_MJY",
    "NVSS_APERTURE_ARCSEC",
    "BLOCK_S",
    "EXPORT_LAG_FALLBACK_S",
    "NVSS_DETECT_SNR",
    "SEFD_REFERENCE_JY",
    "SkyFrameStore",
    "SkyMonitor",
    "parse_snapshot_npz",
    "combine_chgroups_to_uv",
    "dirty_image_from_uv",
    "fit_flux_scale",
    "measure_source_peak",
    "measure_source_snr",
    "pillbox_grid_correction",
    "render_annotated_png",
    "robust_sigma",
    "sefd_predicted_sigma_mjy",
    "snapshot_data_mid_unix",
]
