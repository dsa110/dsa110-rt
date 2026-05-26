"""4-panel cube-event PNG generator for the C2 archive.

When the cube NPZs for a triggered event land in
``/dataz/dsa110/candidates/<name>/cubes/``, the C2 service hands the
event off to a :class:`PlotWorker`. The worker concatenates the 8
NPZs (one per ``(search_node, gpu_half)``) along DM, then renders
four PNGs into ``Level2/plots/`` (see ``docs/c1c2/C1C2_DESIGN.md``
§3.7):

  1. ``dm_time_<name>.png``        — DM vs time waterfall, max-projected
     over (l, m) in a window around the peak.
  2. ``image_peak_<name>.png``     — image plane at (DM_peak, t_peak).
  3. ``lightcurve_<name>.png``     — time series at the peak (l, m, DM).
  4. ``kernel_snrs_<name>.png``    — bar plot of cluster SNR by kernel_id.

The first three need the NPZ cubes; the fourth (`kernel_snrs`) only
needs the cluster's member list and can be drawn even if the cubes
haven't arrived yet.

NPZ schema (matches the existing ``CubeDumpWriter``):

  * ``cube``   — fp16 array of shape (n_fdm, n_t, n_grid, n_grid)
  * ``manifest`` — a dict pickled into the NPZ that includes at
    minimum ``event_specnum``, ``mjd_start``, ``sample_period_us``,
    ``coarse_dm_idx`` and ``fine_dm_pc_cc`` (the per-row DM values).

If the NPZs are missing keys we degrade gracefully — the plotter
writes a placeholder PNG that says "no cube data".

The worker uses a ``ThreadPoolExecutor(max_workers=2)`` so plot jobs
don't block the receive loop. Matplotlib runs with the Agg backend
(no display required).
"""

from __future__ import annotations

import glob
import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Hard-pin matplotlib to Agg before any pyplot import; matplotlib must
# only be imported lazily because the plotter is loaded by the service
# at import-time on hosts that may not have a display.
os.environ.setdefault("MPLBACKEND", "Agg")

from .stats import ClusterStats  # noqa: E402
from .window import WindowEntry  # noqa: E402


__all__ = [
    "PlotWorker",
    "PlotJob",
    "render_event_plots",
]


_LOG = logging.getLogger("dsart.coinc.plotter")


_CUBE_NPZ_RE = re.compile(
    r"^cube_s(?P<sid>\d+)_g(?P<g>\d+)_(?P<specnum>\d+)\.npz$"
)


@dataclass(frozen=True, slots=True)
class PlotJob:
    """Inputs to a plot job. ``stats`` and ``members`` may be empty;
    the kernel_snrs panel still requires ``stats``."""

    event_name: str
    archive_root: Path
    stats: Optional[ClusterStats] = None
    members: Tuple[WindowEntry, ...] = ()


class PlotWorker:
    """ThreadPoolExecutor-backed dispatcher for plot jobs."""

    def __init__(self, max_workers: int = 2, *, per_event_timeout_s: float = 30.0) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="c2-plot",
        )
        self._per_event_timeout_s = per_event_timeout_s
        self._inflight: dict[str, Future] = {}

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def enqueue(self, job: PlotJob) -> Future:
        """Queue a plot job; returns the future for it.

        If a job for the same event_name is already in flight, the
        existing future is returned (no duplicate work).
        """
        existing = self._inflight.get(job.event_name)
        if existing is not None and not existing.done():
            return existing
        fut = self._pool.submit(self._run, job)
        self._inflight[job.event_name] = fut
        return fut

    def _run(self, job: PlotJob) -> List[Path]:
        try:
            return render_event_plots(job)
        finally:
            self._inflight.pop(job.event_name, None)


def enqueue_event(
    worker: PlotWorker,
    event_name: str,
    archive_root: Path,
    *,
    stats: Optional[ClusterStats] = None,
    members: Sequence[WindowEntry] = (),
) -> Future:
    """Module-level entry point matching the design's contract."""
    return worker.enqueue(
        PlotJob(
            event_name=event_name,
            archive_root=Path(archive_root),
            stats=stats,
            members=tuple(members),
        )
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_event_plots(job: PlotJob) -> List[Path]:
    """Render up to four PNGs for one event; returns the paths written.

    Always returns 4 paths (the panels we attempted), even if some
    cubes were missing — the placeholder PNGs are still useful for
    operators eyeballing the dashboard.
    """
    plots_dir = job.archive_root / job.event_name / "Level2" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    cubes = _load_cubes(job.archive_root / job.event_name / "cubes")
    written.append(
        _render_dm_time(plots_dir, job.event_name, cubes, job.stats),
    )
    written.append(
        _render_image_peak(plots_dir, job.event_name, cubes, job.stats),
    )
    written.append(
        _render_lightcurve(plots_dir, job.event_name, cubes, job.stats),
    )
    written.append(
        _render_kernel_snrs(plots_dir, job.event_name, job.members),
    )
    return written


# ----- NPZ I/O ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CubeChunk:
    """One per-(s, g) cube slice loaded from an NPZ."""

    search_node_id: int
    gpu_half: int
    event_specnum: int
    cube: np.ndarray  # shape (n_fdm, n_t, n_grid, n_grid)
    fine_dm_pc_cc: np.ndarray  # shape (n_fdm,)
    mjd_start: float
    sample_period_us: float


def _load_cubes(cubes_dir: Path) -> List[_CubeChunk]:
    """Discover + load up to 8 ``cube_sX_gY_*.npz`` files.

    Best-effort: anything that fails to parse / load is skipped with a
    warning. Returns an empty list if the directory doesn't exist yet
    (cube uploader hasn't caught up) — render_event_plots then writes
    placeholder PNGs.
    """
    if not cubes_dir.is_dir():
        return []
    chunks: List[_CubeChunk] = []
    for p in sorted(cubes_dir.glob("cube_s*_g*_*.npz")):
        m = _CUBE_NPZ_RE.match(p.name)
        if not m:
            _LOG.warning("plotter: ignoring %s — name doesn't match", p)
            continue
        try:
            data = np.load(p, allow_pickle=True)
            cube = np.asarray(data["cube"])
            manifest = data.get("manifest")
            if manifest is not None:
                if hasattr(manifest, "item"):
                    manifest = manifest.item()
                if not isinstance(manifest, dict):
                    manifest = {}
            else:
                manifest = {}
            fine_dm = manifest.get("fine_dm_pc_cc")
            if fine_dm is None:
                fine_dm = np.arange(cube.shape[0], dtype=np.float64)
            else:
                fine_dm = np.asarray(fine_dm, dtype=np.float64)
            chunk = _CubeChunk(
                search_node_id=int(m.group("sid")),
                gpu_half=int(m.group("g")),
                event_specnum=int(m.group("specnum")),
                cube=cube,
                fine_dm_pc_cc=fine_dm,
                mjd_start=float(manifest.get("mjd_start", 0.0)),
                sample_period_us=float(
                    manifest.get("sample_period_us", 1048.576),
                ),
            )
            chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "plotter: failed to load %s: %s; skipping", p, exc,
            )
            continue
    return chunks


# ----- panel renderers ----------------------------------------------------


def _placeholder(path: Path, title: str, msg: str) -> Path:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(
        0.5, 0.5, msg,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=12, color="#636e72",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _render_dm_time(
    plots_dir: Path,
    event_name: str,
    cubes: Sequence[_CubeChunk],
    stats: Optional[ClusterStats],
) -> Path:
    path = plots_dir / f"dm_time_{event_name}.png"
    if not cubes:
        return _placeholder(
            path, f"DM × time — {event_name}", "no cube data",
        )
    import matplotlib.pyplot as plt
    # Concatenate DM axis across chunks (assumes disjoint slices).
    dms: List[float] = []
    panels: List[np.ndarray] = []
    for c in cubes:
        # cube shape is (n_fdm, n_t, n_grid, n_grid). Max over (l, m).
        if c.cube.ndim != 4:
            continue
        try:
            mp = c.cube.astype(np.float32).max(axis=(2, 3))  # (n_fdm, n_t)
        except (TypeError, ValueError):
            continue
        panels.append(mp)
        dms.extend(c.fine_dm_pc_cc.tolist())
    if not panels:
        return _placeholder(
            path, f"DM × time — {event_name}",
            "cube data malformed",
        )
    img = np.concatenate(panels, axis=0)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    extent = (
        0, img.shape[1],
        min(dms) if dms else 0, max(dms) if dms else img.shape[0],
    )
    ax.imshow(
        img, aspect="auto", origin="lower", extent=extent,
        cmap="viridis",
    )
    ax.set_xlabel("time sample")
    ax.set_ylabel("DM (pc cm⁻³)")
    title = f"DM × time waterfall — {event_name}"
    if stats is not None:
        title += f" — SNR_max={stats.snr_max:.1f}, DM_med={stats.dm_median:.1f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _render_image_peak(
    plots_dir: Path,
    event_name: str,
    cubes: Sequence[_CubeChunk],
    stats: Optional[ClusterStats],
) -> Path:
    path = plots_dir / f"image_peak_{event_name}.png"
    if not cubes:
        return _placeholder(
            path, f"image at peak — {event_name}", "no cube data",
        )
    import matplotlib.pyplot as plt
    # Pick the chunk + (fdm, t) with the brightest peak across all
    # cubes — robust to which (s, g) actually owns the burst.
    best_chunk: Optional[_CubeChunk] = None
    best_score: float = -np.inf
    best_fdm = 0
    best_t = 0
    for c in cubes:
        if c.cube.ndim != 4:
            continue
        try:
            arr = c.cube.astype(np.float32)
        except (TypeError, ValueError):
            continue
        peaks = arr.max(axis=(2, 3))  # (n_fdm, n_t)
        idx = int(np.argmax(peaks))
        score = float(peaks.flat[idx])
        if score > best_score:
            best_score = score
            best_chunk = c
            best_fdm, best_t = np.unravel_index(idx, peaks.shape)
    if best_chunk is None:
        return _placeholder(
            path, f"image at peak — {event_name}", "cube data malformed",
        )
    img = best_chunk.cube[best_fdm, best_t].astype(np.float32)
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.imshow(img, origin="lower", cmap="magma")
    ax.set_xlabel("l (pix)")
    ax.set_ylabel("m (pix)")
    title = f"image at peak — {event_name}"
    if stats is not None:
        title += f" — DM={best_chunk.fine_dm_pc_cc[best_fdm]:.1f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _render_lightcurve(
    plots_dir: Path,
    event_name: str,
    cubes: Sequence[_CubeChunk],
    stats: Optional[ClusterStats],
) -> Path:
    path = plots_dir / f"lightcurve_{event_name}.png"
    if not cubes:
        return _placeholder(
            path, f"lightcurve — {event_name}", "no cube data",
        )
    import matplotlib.pyplot as plt
    # Same peak-pick strategy as image_peak.
    best: Optional[Tuple[np.ndarray, _CubeChunk, int]] = None
    best_score = -np.inf
    for c in cubes:
        if c.cube.ndim != 4:
            continue
        try:
            arr = c.cube.astype(np.float32)
        except (TypeError, ValueError):
            continue
        peaks = arr.max(axis=(2, 3))  # (n_fdm, n_t)
        idx = int(np.argmax(peaks))
        score = float(peaks.flat[idx])
        if score > best_score:
            fdm, t = np.unravel_index(idx, peaks.shape)
            lc = arr[fdm].max(axis=(1, 2))  # max over (l, m) → (n_t,)
            best = (lc, c, int(fdm))
            best_score = score
    if best is None:
        return _placeholder(
            path, f"lightcurve — {event_name}", "cube data malformed",
        )
    lc, c, fdm = best
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(lc, color="#0984e3", lw=1.5)
    ax.set_xlabel("time sample")
    ax.set_ylabel("peak amplitude (image max)")
    title = f"lightcurve — {event_name} — DM={c.fine_dm_pc_cc[fdm]:.1f}"
    if stats is not None:
        title += f", SNR_max={stats.snr_max:.1f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _render_kernel_snrs(
    plots_dir: Path,
    event_name: str,
    members: Sequence[WindowEntry],
) -> Path:
    path = plots_dir / f"kernel_snrs_{event_name}.png"
    if not members:
        return _placeholder(
            path, f"kernel SNR distribution — {event_name}",
            "no cluster members",
        )
    import matplotlib.pyplot as plt
    by_kernel: dict[str, List[float]] = {}
    for m in members:
        by_kernel.setdefault(m.kernel_id, []).append(m.snr)
    labels = sorted(by_kernel.keys())
    maxes = [max(by_kernel[k]) for k in labels]
    counts = [len(by_kernel[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, maxes, color="#fdcb6e", edgecolor="#2d3436")
    for x, c, b in zip(xs, counts, bars):
        ax.text(
            x, b.get_height(), f"n={c}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("max SNR per kernel")
    ax.set_title(f"kernel SNR distribution — {event_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
