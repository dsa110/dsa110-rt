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

  * ``cube``   — fp16 array of shape (n_fdm, n_t, n_grid, n_grid),
    stored uncompressed (``np.savez``, not ``np.savez_compressed``).
  * Either a top-level pickled ``manifest`` dict (older test fixture
    format) **or** individual top-level scalar arrays (production
    writer in ``dsart.dump.cube_dump``: ``mjd_start``,
    ``event_specnum_start``, ``t_det``, …). We accept both.

If the NPZs are missing keys we degrade gracefully — the plotter
writes a placeholder PNG that says "no cube data".

The worker uses a ``ThreadPoolExecutor(max_workers=2)`` so plot jobs
don't block the receive loop. Matplotlib runs with the Agg backend
(no display required).

Performance notes
-----------------

The cubes are ~855 MB each (192 × 34 × 256 × 256 fp16) and there are
8 per event, so naïvely loading + 3×-reducing them through
``astype(float32)`` blows past 20 GB of RAM and takes >5 minutes. The
optimised pipeline:

  1. Memory-maps each ``cube.npy`` from inside its (uncompressed) zip
     by parsing the local file header ourselves —
     ``np.load(path, mmap_mode='r')`` silently ignores ``mmap_mode``
     for ``.npz`` inputs (numpy ≤ 2.4 NpzFile.__getitem__ doesn't
     forward it to ``format.read_array``), so the OS-level mmap has
     to be set up by hand.
  2. Streams a single ``cube.max(axis=(2, 3))`` reduction per cube and
     caches the result on ``_CubeChunk.peak_grid`` (shape
     ``(n_fdm, n_t)``, fp32, ~26 KB). All three render passes read
     this cached array; none of them re-touch the full cube.
  3. For ``image_peak``, after peak-picking we materialise just the
     single ``(n_grid, n_grid)`` image at ``cube[best_fdm, best_t]``
     (~128 KB).
  4. For ``lightcurve``, ``cube[best_fdm].max(axis=(1, 2))`` is
     algebraically equivalent to ``peak_grid[best_fdm]``, so we don't
     touch the cube at all.

Each major step logs its wall-clock cost at INFO level so operators
can correlate slow plot runs with disk-cache state.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import struct
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

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

    t_total = time.perf_counter()
    written: List[Path] = []

    t0 = time.perf_counter()
    cubes = _load_cubes(job.archive_root / job.event_name / "cubes")
    t_load = time.perf_counter() - t0
    _LOG.info(
        "plotter: loaded %d cubes in %.1fs (event=%s)",
        len(cubes), t_load, job.event_name,
    )

    t0 = time.perf_counter()
    _populate_peak_grids(cubes)
    t_reduce = time.perf_counter() - t0
    _LOG.info(
        "plotter: reduced %d cubes in %.1fs (event=%s)",
        len(cubes), t_reduce, job.event_name,
    )

    try:
        t0 = time.perf_counter()
        written.append(
            _render_dm_time(plots_dir, job.event_name, cubes, job.stats),
        )
        _LOG.info(
            "plotter: dm_time rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )

        t0 = time.perf_counter()
        written.append(
            _render_image_peak(plots_dir, job.event_name, cubes, job.stats),
        )
        _LOG.info(
            "plotter: image_peak rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )

        t0 = time.perf_counter()
        written.append(
            _render_lightcurve(plots_dir, job.event_name, cubes, job.stats),
        )
        _LOG.info(
            "plotter: lightcurve rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )

        t0 = time.perf_counter()
        written.append(
            _render_kernel_snrs(plots_dir, job.event_name, job.members),
        )
        _LOG.info(
            "plotter: kernel_snrs rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )
    finally:
        # Drop references to mmap'd cubes so the OS can reclaim VM
        # ranges promptly; the underlying file pages stay in the page
        # cache so a subsequent re-render is fast.
        for c in cubes:
            c.close()

    _LOG.info(
        "plotter: event=%s total=%.1fs (load=%.1fs reduce=%.1fs)",
        job.event_name, time.perf_counter() - t_total, t_load, t_reduce,
    )
    return written


# ----- NPZ I/O ------------------------------------------------------------


@dataclass(slots=True)
class _CubeChunk:
    """One per-(s, g) cube slice loaded from an NPZ.

    ``cube`` is a memory-map of the underlying NPZ (set up via
    :func:`_mmap_npz_array` so we don't hold ~7 GB of fp16 in RSS for
    a single event). ``peak_grid`` is the ``cube.max(axis=(2, 3))``
    reduction, computed exactly once at load time and reused by all
    three cube-touching panel renderers.
    """

    search_node_id: int
    gpu_half: int
    event_specnum: int
    cube: np.ndarray  # shape (n_fdm, n_t, n_grid, n_grid), fp16, mmap'd
    fine_dm_pc_cc: np.ndarray  # shape (n_fdm,)
    mjd_start: float
    sample_period_us: float
    # Populated by _populate_peak_grids after _load_cubes returns.
    peak_grid: Optional[np.ndarray] = None  # (n_fdm, n_t), fp32
    # Keep the NpzFile reference alive so the metadata zip handle stays
    # valid for the lifetime of the chunk; _mmap_npz_array uses its
    # own independent np.memmap so this is just hygiene.
    _npz: Optional[Any] = field(default=None, repr=False)

    def close(self) -> None:
        """Release the NpzFile + mmap'd cube reference."""
        if self._npz is not None:
            try:
                self._npz.close()
            except Exception:  # noqa: BLE001
                pass
            self._npz = None
        # Drop the mmap'd ndarray reference so the OS can unmap the
        # range when the GC eventually fires. We can't .close() a
        # plain ndarray-with-memmap-base, but releasing the strong
        # ref is enough.
        self.cube = _EMPTY_CUBE


# Sentinel placeholder so the dataclass slot stays a real ndarray
# after close() (avoids isinstance / .ndim checks needing to be
# Optional-aware downstream).
_EMPTY_CUBE: np.ndarray = np.empty((0, 0, 0, 0), dtype=np.float16)


def _mmap_npz_array(npz_path: Path, key: str = "cube") -> np.ndarray:
    """Memory-map an *uncompressed* array stored inside an ``.npz`` file.

    ``np.load(path, mmap_mode='r')`` silently ignores ``mmap_mode``
    for npz inputs in numpy ≤ 2.4 (NpzFile.__getitem__ does not
    forward it to ``format.read_array``), so a 855 MB cube ends up
    fully copied into RAM each time. We avoid that by parsing the
    zip's local file header to find the byte offset of the embedded
    ``.npy`` data and ``np.memmap``-ing it directly.

    Only works for ``ZIP_STORED`` entries (i.e. ``np.savez``, not
    ``np.savez_compressed``). The CubeDumpWriter uses ``np.savez``.
    """
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            f"{key}.npy in {npz_path} is compressed "
            f"(zip method={info.compress_type}); cannot mmap. "
            "Re-write with np.savez instead of np.savez_compressed."
        )
    # Local file header: PK\x03\x04 + 26 bytes of fields, then a
    # variable-length filename and 'extra' field, then the file data.
    # The CRC and sizes mirror the central directory entry we already
    # have in `info`, but the local header's extra-field length can
    # differ (zip64, alignment padding), so parse it from the file.
    with open(npz_path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError(
                f"bad zip local header magic at offset "
                f"{info.header_offset} in {npz_path}",
            )
        name_len = struct.unpack("<H", local[26:28])[0]
        extra_len = struct.unpack("<H", local[28:30])[0]
        data_start = info.header_offset + 30 + name_len + extra_len

        # Now decode the .npy header so we know dtype/shape/order.
        fh.seek(data_start)
        magic = fh.read(10)
        if magic[:6] != b"\x93NUMPY":
            raise ValueError(
                f"missing NPY magic at offset {data_start} in {npz_path}"
            )
        major = magic[6]
        if major == 1:
            hl = struct.unpack("<H", magic[8:10])[0]
            npy_prefix = 10 + hl
        else:  # major in (2, 3): 4-byte header length
            hl_ext = fh.read(2)
            hl = struct.unpack("<I", magic[8:10] + hl_ext)[0]
            npy_prefix = 12 + hl
        header_str = fh.read(hl).decode("latin1")

    # ``literal_eval`` is enough for npy headers (per numpy.lib.format).
    header_dict = ast.literal_eval(header_str)
    dtype = np.dtype(header_dict["descr"])
    shape = tuple(header_dict["shape"])
    order = "F" if bool(header_dict.get("fortran_order", False)) else "C"
    data_offset = data_start + npy_prefix

    return np.memmap(
        str(npz_path),
        dtype=dtype,
        shape=shape,
        order=order,
        mode="r",
        offset=data_offset,
    )


def _read_manifest(npz: Any) -> dict:
    """Best-effort read of a pickled top-level ``manifest`` dict.

    Returns ``{}`` if the npz doesn't carry one (production schema
    stores fields as separate top-level scalar arrays instead).
    """
    if "manifest" not in getattr(npz, "files", ()):
        return {}
    try:
        raw = npz["manifest"]
    except Exception:  # noqa: BLE001 — pickled object load may fail
        return {}
    if hasattr(raw, "item"):
        try:
            raw = raw.item()
        except Exception:  # noqa: BLE001
            return {}
    return raw if isinstance(raw, dict) else {}


def _scalar_or(npz: Any, key: str, default: float) -> float:
    """Read a 0-d array out of an NpzFile, falling back to ``default``."""
    if key not in getattr(npz, "files", ()):
        return float(default)
    try:
        return float(np.asarray(npz[key]))
    except (TypeError, ValueError):
        return float(default)


def _load_cubes(cubes_dir: Path) -> List[_CubeChunk]:
    """Discover + mmap up to 8 ``cube_sX_gY_*.npz`` files.

    Best-effort: anything that fails to parse / mmap is skipped with
    a warning. Returns an empty list if the directory doesn't exist
    yet (cube uploader hasn't caught up) — render_event_plots then
    writes placeholder PNGs.

    The returned chunks have ``peak_grid is None``;
    :func:`_populate_peak_grids` runs the (n_fdm, n_t) reduction in a
    second pass so we can time the IO/decoding cost separately from
    the actual reduction.
    """
    if not cubes_dir.is_dir():
        return []
    chunks: List[_CubeChunk] = []
    for p in sorted(cubes_dir.glob("cube_s*_g*_*.npz")):
        m = _CUBE_NPZ_RE.match(p.name)
        if not m:
            _LOG.warning("plotter: ignoring %s — name doesn't match", p)
            continue
        npz = None
        try:
            cube = _mmap_npz_array(p, "cube")
            # Open the NpzFile for the *small* metadata fields only;
            # we never call ``npz['cube']`` (which would force a full
            # in-RAM copy via numpy.lib.format.read_array). allow_pickle
            # is still required for the legacy test fixture, which
            # stores a pickled ``manifest`` dict.
            npz = np.load(p, allow_pickle=True)
            manifest = _read_manifest(npz)

            fine_dm = manifest.get("fine_dm_pc_cc")
            if fine_dm is None:
                fine_dm = np.arange(cube.shape[0], dtype=np.float64)
            else:
                fine_dm = np.asarray(fine_dm, dtype=np.float64)

            mjd_start = (
                float(manifest["mjd_start"])
                if "mjd_start" in manifest
                else _scalar_or(npz, "mjd_start", 0.0)
            )
            sample_period_us = (
                float(manifest["sample_period_us"])
                if "sample_period_us" in manifest
                else _scalar_or(npz, "sample_period_us", 1048.576)
            )

            chunks.append(_CubeChunk(
                search_node_id=int(m.group("sid")),
                gpu_half=int(m.group("g")),
                event_specnum=int(m.group("specnum")),
                cube=cube,
                fine_dm_pc_cc=fine_dm,
                mjd_start=mjd_start,
                sample_period_us=sample_period_us,
                _npz=npz,
            ))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "plotter: failed to load %s: %s; skipping", p, exc,
            )
            if npz is not None:
                try:
                    npz.close()
                except Exception:  # noqa: BLE001
                    pass
            continue
    return chunks


def _populate_peak_grids(cubes: List[_CubeChunk]) -> None:
    """Populate ``peak_grid = cube.max(axis=(2, 3))`` once per chunk.

    Fast path: if the NPZ already carries a ``peak_grid`` array
    (CubeDumpWriter precomputes it as of 2026-05-27), we skip the
    reduction entirely and just up-cast the cached ``(n_fdm, n_t)``
    fp16 array to fp32 (~26 KB per cube). This drops the plotter's
    dominant cost stage from ~10 s/cube to <50 ms/cube on h23.

    Slow path (backwards compatibility with cubes dumped before the
    writer-side precompute landed): we run ``cube.max(axis=(2, 3))``
    on the mmap'd cube, streaming it once through the OS page cache.
    The fp16 reduction result is up-cast to fp32; downstream rendering
    stays fp32 without any further full-cube copies.

    Done in a second pass (rather than inside ``_load_cubes``) so the
    timing log line can attribute disk-IO + reduction cost cleanly.
    """
    n_fast = 0
    n_slow = 0
    for c in cubes:
        # Fast path: writer-side precomputed peak_grid.
        cached = _try_read_cached_peak_grid(c)
        if cached is not None:
            c.peak_grid = cached
            n_fast += 1
            continue
        if c.cube.ndim != 4:
            continue
        try:
            peaks = c.cube.max(axis=(2, 3))
        except (TypeError, ValueError) as exc:
            _LOG.warning(
                "plotter: max-reduce failed on cube s%d_g%d: %s; skipping",
                c.search_node_id, c.gpu_half, exc,
            )
            continue
        c.peak_grid = np.asarray(peaks, dtype=np.float32)
        n_slow += 1
    if n_fast or n_slow:
        _LOG.info(
            "plotter: peak_grid populated fast=%d (cached) slow=%d (computed)",
            n_fast, n_slow,
        )


def _try_read_cached_peak_grid(c: "_CubeChunk") -> Optional[np.ndarray]:
    """Return cached ``peak_grid`` from the NPZ as fp32, or None.

    The cached array is shape ``(n_fdm, n_t)``, dtype fp16 (matches the
    cube). Validation: ndim==2, non-empty, leading axis matches
    ``c.cube.shape[0]``. Anything off → return None so the caller
    falls back to the full reduction.
    """
    if c._npz is None or "peak_grid" not in c._npz.files:
        return None
    try:
        peaks = c._npz["peak_grid"]
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(peaks, np.ndarray) or peaks.ndim != 2 or peaks.size == 0:
        return None
    if c.cube.ndim == 4 and peaks.shape[0] != c.cube.shape[0]:
        _LOG.warning(
            "plotter: cached peak_grid axis-0 (%d) doesn't match cube "
            "n_fdm (%d) for s%d_g%d; ignoring cache",
            peaks.shape[0], c.cube.shape[0],
            c.search_node_id, c.gpu_half,
        )
        return None
    return np.asarray(peaks, dtype=np.float32)


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


def _peak_chunks(
    cubes: Sequence[_CubeChunk],
) -> List[_CubeChunk]:
    """Subset of ``cubes`` whose ``peak_grid`` was populated successfully."""
    return [c for c in cubes if c.peak_grid is not None]


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
    usable = _peak_chunks(cubes)
    if not usable:
        return _placeholder(
            path, f"DM × time — {event_name}", "cube data malformed",
        )
    import matplotlib.pyplot as plt
    # Concatenate DM axis across chunks (assumes disjoint slices).
    # Each peak_grid is already shape (n_fdm, n_t), fp32 — no further
    # cube data is touched here.
    panels = [c.peak_grid for c in usable]
    dms: List[float] = []
    for c in usable:
        dms.extend(c.fine_dm_pc_cc.tolist())
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


def _argmax_across_chunks(
    usable: Sequence[_CubeChunk],
) -> Tuple[_CubeChunk, int, int, float]:
    """Return ``(best_chunk, best_fdm, best_t, best_score)`` from cached
    peak grids — never touches the full cube data."""
    best_chunk = usable[0]
    best_fdm = 0
    best_t = 0
    best_score = -np.inf
    for c in usable:
        pg = c.peak_grid
        assert pg is not None  # filtered by caller
        idx = int(np.argmax(pg))
        score = float(pg.flat[idx])
        if score > best_score:
            best_score = score
            best_chunk = c
            fdm, t = np.unravel_index(idx, pg.shape)
            best_fdm, best_t = int(fdm), int(t)
    return best_chunk, best_fdm, best_t, best_score


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
    usable = _peak_chunks(cubes)
    if not usable:
        return _placeholder(
            path, f"image at peak — {event_name}", "cube data malformed",
        )
    import matplotlib.pyplot as plt
    best_chunk, best_fdm, best_t, _ = _argmax_across_chunks(usable)
    # Materialise *only* the single (n_grid, n_grid) image plane —
    # ~128 KB, vs ~1.7 GB for ``cube.astype(float32)``.
    img = np.asarray(
        best_chunk.cube[best_fdm, best_t], dtype=np.float32,
    )
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
    usable = _peak_chunks(cubes)
    if not usable:
        return _placeholder(
            path, f"lightcurve — {event_name}", "cube data malformed",
        )
    import matplotlib.pyplot as plt
    best_chunk, best_fdm, _, _ = _argmax_across_chunks(usable)
    # ``cube[fdm].max(axis=(1, 2))`` == ``peak_grid[fdm]`` because max
    # commutes with axis-aligned slicing. No cube touch needed —
    # peak_grid is already cached.
    pg = best_chunk.peak_grid
    assert pg is not None
    lc = np.asarray(pg[best_fdm], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(lc, color="#0984e3", lw=1.5)
    ax.set_xlabel("time sample")
    ax.set_ylabel("peak amplitude (image max)")
    title = (
        f"lightcurve — {event_name} — "
        f"DM={best_chunk.fine_dm_pc_cc[best_fdm]:.1f}"
    )
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
