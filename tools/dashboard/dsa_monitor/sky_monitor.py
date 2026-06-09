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
from typing import Any, Optional

import numpy as np

# Force Agg before pyplot-adjacent imports (matches plot_render.py).
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg                       # noqa: E402

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
_FRAME_RE = re.compile(r"^sky_(?P<ts>\d+)_n(?P<ncg>\d+)\.(?:png|npz)$")


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
) -> tuple[np.ndarray, list[int]]:
    """Sum the per-chgroup sparse snapshots into one dense UV grid.

    Each chgroup's vis is divided by ``amp_scale²`` (bandpass
    flattening — see module docstring) before scattering. Chgroups
    whose ``n_grid`` doesn't match are skipped with a warning (a corr
    node running a stale config must not corrupt the whole frame).

    Returns ``(uv_grid complex64 (n_grid, n_grid), used_chgroups)``.
    """
    uv = np.zeros((n_grid, n_grid), dtype=np.complex64)
    used: list[int] = []
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
        # np.add.at: pattern cells are unique per chgroup, but += via
        # ufunc.at is safe even if they ever are not.
        np.add.at(uv, (rows, cols), (snap["vis"] * np.float32(w)))
        used.append(cg)
    return uv, sorted(used)


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
    ) -> tuple[Path, Path]:
        """Write the PNG (sigma-stretched greyscale) + NPZ (raw float32
        image + full metadata). Returns ``(png_path, npz_path)``.
        """
        day_dir = self.frames_dir / self._day(ts)
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sky_{int(ts)}_n{len(used_chgroups)}"
        png_path = day_dir / f"{stem}.png"
        npz_path = day_dir / f"{stem}.npz"

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
        np.savez_compressed(
            npz_path,
            image=image.astype(np.float32),
            median=np.float64(median),
            sigma=np.float64(sigma),
            used_chgroups=np.asarray(used_chgroups, dtype=np.int16),
            meta_json=np.bytes_(json.dumps(meta).encode("utf-8")),
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
        *,
        frame_interval_s: float = 30.0,
        freshness_s: float = 90.0,
        min_chgroups: int = 1,
        n_grid: int = 256,
        oversample: int = 2,
        grid_correct: bool = True,
    ) -> None:
        self.store = store if store is not None else SkyFrameStore()
        self.frame_interval_s = float(frame_interval_s)
        self.freshness_s = float(freshness_s)
        self.min_chgroups = int(min_chgroups)
        self.n_grid = int(n_grid)
        self.oversample = max(1, int(oversample))
        self.grid_correct = bool(grid_correct)

        self._lock = threading.Lock()
        self._latest: dict[int, dict[str, Any]] = {}     # chgroup → snapshot
        self._recv_unix: dict[int, float] = {}           # chgroup → ingest time
        self._last_frame_unix: float = 0.0
        self.n_ingested = 0
        self.n_rejected = 0
        self.n_frames = 0

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
        uv, used = combine_chgroups_to_uv(fresh, n_grid=self.n_grid)
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
            "per_chgroup": {
                str(int(s["meta"]["chgroup"])): {
                    "hostname": s["meta"].get("hostname"),
                    "block_n": s["meta"].get("block_n"),
                    "cubes_seen": s["meta"].get("cubes_seen"),
                    "amp_scale": s["meta"].get("amp_scale"),
                    "unix_ts": s["meta"].get("unix_ts"),
                }
                for s in fresh
            },
        }
        self.store.write_frame(
            image, ts=now, median=median, sigma=sigma,
            used_chgroups=used, meta=meta,
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
    "SkyFrameStore",
    "SkyMonitor",
    "parse_snapshot_npz",
    "combine_chgroups_to_uv",
    "dirty_image_from_uv",
    "pillbox_grid_correction",
    "robust_sigma",
]
