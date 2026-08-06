"""4-panel cube-event PNG generator for the C2 archive.

When the cube NPZs for a triggered event land in
``/dataz/dsa110/candidates/<name>/cubes/``, the C2 service hands the
event off to a :class:`PlotWorker`. The worker concatenates the 8
NPZs (one per ``(search_node, gpu_half)``) along DM, then renders
four PNGs into ``Level2/plots/`` (see ``docs/c1c2/C1C2_DESIGN.md``
§3.7):

  1. ``dm_time_<name>.png``        — ONE DM vs time waterfall built by
     stacking ALL dumped cubes' fine-DM rows in (search_node,
     gpu_half) order (2026-06-10), i.e. contiguous DM coverage over
     the whole searched range, each cube max-projected over (l, m).
     Every fine-DM row is robustly re-normalised (median/MAD over
     time) so rows from different halves — whose image-max baselines
     and Layer-1/2 calibrations differ slightly — share one σ-unit
     colour scale, keeping the dispersed bowtie tails in the
     non-owning halves visible alongside the owning half. Crosshair
     at the detected (t_peak, DM_peak); dashed separators mark the
     half boundaries.
  2. ``image_peak_<name>.png``     — image plane at the detected
     (DM_peak, t_peak), with a reticle at the detected (l, m).
  3. ``lightcurve_<name>.png``     — time series at the detected DM,
     with a vertical reticle at the detected burst time.
  4. ``kernel_snrs_<name>.png``    — bar plot of cluster SNR by kernel_id.

The first three need the NPZ cubes; the fourth (`kernel_snrs`) only
needs the cluster's member list and can be drawn even if the cubes
haven't arrived yet.

Burst selection (why we don't argmax the cube)
----------------------------------------------

The dumped cube is dominated by steady continuum + low-DM RFI, so a
naïve ``cube.argmax`` lands on the brightest *image* pixel — which is
**not** the burst (a real high-DM transient is fainter than the
brightest steady source in any single image plane). We therefore pick
the panel coordinates from the C2 **detection metadata**:

  * live: the cluster's ``WindowEntry`` members (``PlotJob.members``);
  * offline / re-render: the per-event ``Level2/C1_window_<name>.csv``
    rows (same fields), so ``render_event_plots`` can rebuild correct
    plots for any archived event with no live state.

The peak member (max ``snr``) gives the owning ``(search_node, gpu_half)``
cube, the fine-DM trial (``fine_dm_idx``), and the image pixel
(``l_pix, m_pix``).

The burst *time* comes from the cube's sample-0 anchor when the NPZ
carries one (``cube_specnum_start``, dumps from 2026-08-04 on):

    t_in_cube = event_specnum - cube_specnum_start

Both operands count SEARCH samples (one per detector sample), so this
is a plain subtraction — ``sample_period_specnum`` is native specnums
per search sample and must NOT be divided out here; see
:func:`_metadata_t_idx` for the full units contract and the archive
evidence. Older NPZs carry only ``event_specnum_start``, which is the
dump *key* (the trigger specnum), not the cube's first-sample specnum,
so specnum arithmetic can't locate the burst in them — for those we
fall back to the argmax of the DM-sliced light curve.

Matching the detector's boxcar (2026-08-02)
-------------------------------------------

That argmax used to run on the **raw** per-sample series, while the
detector thresholds a *boxcar match-filtered* one of width
``width_samples``. On real events that mismatch is large: for
260801rmep (``width_samples=16``) the raw argmax sits at sample 110 with
6.5 sigma at the detected pixel, while the width-matched argmax sits at
107 with **15.9 sigma** against a detector-reported 19.7. The panels
were therefore being placed several samples off the burst centre and
the re-measured significance understated it by ~2.4x.

We now convolve the DM light curve with the detector's own
``width_samples`` boxcar (normalised ``1/sqrt(w)``, so the result stays
in sigma units) before taking the argmax, and :func:`_render_dm_time`
shows the **smoothed** DM-time cube. ``_BurstCoords`` carries both the
detector's SNR and the re-measured one so every panel title can print
the difference; a large gap means the cube and the C1 row disagree and
the event needs a human.

Set ``DSART_PLOTTER_SMOOTH=0`` to restore the raw-argmax behaviour.

The light curve and the peak search run on the **detected pixel**
``cube[:, fdm, l_pix, m_pix]``, not on ``cube.max(axis=(2, 3))``: the
detector works per pixel, and a max over 65 536 pixels has an extreme-
value distribution whose median/MAD is not a sigma scale for a point
source (on 260801bdga the image-max series re-measures the burst at 7.3
sigma against 12.4 at the detected pixel). The image-max trace is kept
as context on the light-curve panel.

NPZ schema (matches the production ``CubeDumpWriter``)
------------------------------------------------------

  * ``cube`` — fp16 array of shape **(t_det, n_fdm, n_grid, n_grid)**,
    i.e. axis 0 = time, axis 1 = fine-DM trial, axes 2/3 = the (l, m)
    image grid (``cube[t, fdm, l_pix, m_pix]`` — matches the detector
    score tensor ``[T_det, N_fdm, H=l, W=m]``). Stored uncompressed
    (``np.savez``).
  * Either a top-level pickled ``manifest`` dict (older test fixture
    format) **or** individual top-level scalar arrays (production
    writer in ``dsart.dump.cube_dump``: ``mjd_start``,
    ``event_specnum_start``, ``t_det``, …). We accept both.

.. note::

   The writer also stores a ``peak_grid`` array, but it has been
   observed to be **inconsistent** with the stored ``cube`` (its argmax
   lands at low DM while ``cube.argmax`` lands on the burst DM), so the
   plotter no longer trusts it: it recomputes ``cube.max(axis=(2, 3))``
   for the single burst cube.

If the NPZs are missing keys we degrade gracefully — the plotter
writes a placeholder PNG that says "no cube data".

The worker uses a ``ThreadPoolExecutor(max_workers=2)`` so plot jobs
don't block the receive loop. Matplotlib runs with the Agg backend
(no display required).

Performance notes
-----------------

The cubes are ~855 MB each (192 × 34 × 256 × 256 fp16) and there are
8 per event:

  1. Memory-map each ``cube.npy`` from inside its (uncompressed) zip
     by parsing the local file header ourselves —
     ``np.load(path, mmap_mode='r')`` silently ignores ``mmap_mode``
     for ``.npz`` inputs (numpy ≤ 2.4 NpzFile.__getitem__ doesn't
     forward it to ``format.read_array``), so the OS-level mmap has
     to be set up by hand.
  2. Stream a ``cube.max(axis=(2, 3))`` reduction on EVERY cube →
     per-cube ``waterfall`` (shape ``(t_det, n_fdm)``, fp32).
     2026-06-10: this used to touch only the burst cube; the 8-panel
     dm_time figure needs all of them. The reductions stream through
     the mmap (page-cache backed, ~25 MB working set each) so the cost
     is one sequential read of each NPZ (~7 GB/event cold, fast when
     the dump is still in page cache). Per-cube timing is logged.
  3. For ``image_peak`` we materialise the single ``(n_grid, n_grid)``
     plane at ``cube[t_peak, fine_dm_idx]`` (~128 KB).

Each major step logs its wall-clock cost at INFO level so operators
can correlate slow plot runs with disk-cache state.
"""

from __future__ import annotations

import ast
import csv
import logging
import os
import re
import struct
import time
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
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
    "regenerate_recent_events",
]


_LOG = logging.getLogger("dsart.coinc.plotter")


_CUBE_NPZ_RE = re.compile(
    r"^cube_s(?P<sid>\d+)_g(?P<g>\d+)_(?P<specnum>\d+)\.npz$"
)


# --------------------------------------------------------------------------
# Fine-DM axis from the production DM plan
# --------------------------------------------------------------------------
#
# The dumped cube NPZs do NOT carry per-row DM values (CubeDumpWriter.savez
# omits fine_dm_pc_cc), so the waterfall y-axis used to fall back to bare
# fine-DM trial indices (0..33 per half). To label rows with real DMs we
# read the production DM plan and map each (search_node_id, gpu_half) cube
# to the coarse-DM bucket it owns.
#
# Source-of-truth path: the SAME plan the search fleet runs
# (``--dm-plan-path`` in configs/dsart_search_rt.yaml). NOTE: configs/
# dm_plan.npz is the stale v1 plan (690 fine / 16 coarse / 0–3000) and must
# NOT be used here. Override with DSART_DM_PLAN_PATH if the fleet plan moves.
_DEFAULT_DM_PLAN_PATH = (
    "/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz"
)

# Cache so a multi-event regenerate doesn't re-read the plan per event.
_DM_PLAN_CACHE: dict[str, Optional[dict[int, np.ndarray]]] = {}


def _dm_plan_path() -> str:
    return os.environ.get("DSART_DM_PLAN_PATH", _DEFAULT_DM_PLAN_PATH)


def _load_coarse_bucket_dms(
    plan_path: Optional[str] = None,
) -> Optional[dict[int, np.ndarray]]:
    """Return ``{coarse_idx: ascending fine-DM array}`` from the DM plan.

    Reads ``fine_dm`` + ``fine_to_coarse`` from the production plan NPZ and
    groups the fine-DM trials by their owning coarse-DM bucket. Each search
    half owns exactly one coarse bucket, so this is the per-half row→DM map.
    Returns ``None`` (and the caller falls back to row indices) if the plan
    is unreadable or missing the needed keys.
    """
    path = plan_path or _dm_plan_path()
    if path in _DM_PLAN_CACHE:
        return _DM_PLAN_CACHE[path]
    buckets: Optional[dict[int, np.ndarray]] = None
    try:
        with np.load(path) as npz:
            fine_key = next(
                (k for k in ("fine_dm_pc_cm3", "fine_dm") if k in npz), None,
            )
            if fine_key is not None and "fine_to_coarse" in npz:
                fine_dm = np.asarray(npz[fine_key], dtype=np.float64)
                f2c = np.asarray(npz["fine_to_coarse"], dtype=np.int64)
                if fine_dm.shape[0] == f2c.shape[0] and fine_dm.size:
                    buckets = {
                        int(c): np.sort(fine_dm[f2c == c])
                        for c in np.unique(f2c)
                    }
    except (OSError, ValueError, KeyError) as exc:
        _LOG.warning("plotter: DM plan %s unreadable (%s); "
                     "waterfall y-axis falls back to trial index", path, exc)
        buckets = None
    _DM_PLAN_CACHE[path] = buckets
    return buckets


def _owner_coarse_idx(search_node_id: int, gpu_half: int,
                      present_sids: Sequence[int]) -> int:
    """Map a (search_node_id, gpu_half) cube to its coarse-DM bucket.

    Production fan-out: each search node owns 2 consecutive coarse buckets
    (one per GPU half), assigned in ascending node order — n01→{0,1},
    n02→{2,3}, n09→{4,5}, n13→{6,7}. We rank the search nodes actually
    present so the mapping degrades gracefully if a subset dumped.
    """
    ranks = {sid: i for i, sid in enumerate(sorted(set(present_sids)))}
    return ranks.get(int(search_node_id), 0) * 2 + int(gpu_half)


@dataclass(frozen=True, slots=True)
class PlotJob:
    """Inputs to a plot job. ``stats`` and ``members`` may be empty;
    the kernel_snrs panel still requires ``stats``."""

    event_name: str
    archive_root: Path
    stats: Optional[ClusterStats] = None
    members: Tuple[WindowEntry, ...] = ()


def _render_job(job: "PlotJob") -> List[Path]:
    """Module-level plot-render entry point.

    Must be a top-level function (not a bound method) so it is picklable
    for ``ProcessPoolExecutor`` dispatch. Returns the list of PNG paths
    written. Exceptions propagate into the returned Future, where the
    enqueuer's done-callback logs them."""
    return render_event_plots(job)


class PlotWorker:
    """Executor-backed dispatcher for plot jobs.

    Plot rendering touches ~855 MB cubes + matplotlib, both of which hold
    the GIL for long stretches. When run in a ``ThreadPoolExecutor`` this
    serialises against the C2 service's single asyncio event loop and can
    starve the C1 receiver (observed 2026-05-30: a dump storm wedged the
    loop ~34 min, flatlining C1->C2). The default ``use_process_pool=True``
    isolates that work in a separate process so the receiver always drains
    its sockets. ``use_process_pool=False`` keeps the legacy thread pool
    for environments where fork/spawn is unavailable (restricted sandboxes,
    some test rigs)."""

    def __init__(
        self,
        max_workers: int = 2,
        *,
        per_event_timeout_s: float = 30.0,
        use_process_pool: bool = True,
    ) -> None:
        self._use_process_pool = use_process_pool
        if use_process_pool:
            try:
                self._pool: Any = ProcessPoolExecutor(max_workers=max_workers)
            except Exception:  # noqa: BLE001 — fall back to threads
                _LOG.warning(
                    "PlotWorker: ProcessPoolExecutor unavailable; "
                    "falling back to ThreadPoolExecutor", exc_info=True,
                )
                self._use_process_pool = False
                self._pool = ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="c2-plot",
                )
        else:
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
        existing future is returned (no duplicate work). Bookkeeping for
        ``_inflight`` is done via a done-callback (so it works for both
        thread- and process-pool futures, where the worker can't mutate
        the parent's dict)."""
        existing = self._inflight.get(job.event_name)
        if existing is not None and not existing.done():
            return existing
        fut = self._pool.submit(_render_job, job)
        self._inflight[job.event_name] = fut

        def _done(_f: Future, name: str = job.event_name) -> None:
            self._inflight.pop(name, None)
            exc = _f.exception() if not _f.cancelled() else None
            if exc is not None:
                _LOG.error("plot job failed for %s: %r", name, exc)

        fut.add_done_callback(_done)
        return fut


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

    # Resolve the burst from detection metadata (live members or the
    # archived C1-window CSV), then reduce ONLY the burst cube.
    peak, kernels = _resolve_burst(job)
    burst = _select_burst_chunk(cubes, peak)
    if peak is not None:
        _LOG.info(
            "plotter: burst peak s%s_g%s fdm=%d (l,m)=(%d,%d) "
            "DM=%.1f SNR=%.1f (event=%s, source=%s)",
            peak.search_node_id, peak.gpu_half, peak.fine_dm_idx,
            peak.l_pix, peak.m_pix, peak.dm_pc_cc, peak.snr,
            job.event_name, peak.source,
        )
    else:
        _LOG.warning(
            "plotter: no detection metadata for %s — falling back to "
            "cube argmax (may latch onto low-DM RFI)", job.event_name,
        )

    # 2026-06-10: reduce EVERY cube (not just the burst one) so the
    # dm_time figure can show all 8 halves at once — the dispersed
    # bowtie tails in the non-owning halves are part of the burst's
    # signature and operators want them on one page.
    t0 = time.perf_counter()
    waterfalls = _all_waterfalls(cubes)
    t_reduce = time.perf_counter() - t0
    _LOG.info(
        "plotter: reduced %d cube waterfalls in %.1fs (event=%s)",
        len(waterfalls), t_reduce, job.event_name,
    )
    waterfall = None  # burst cube's waterfall (for lightcurve/coords)
    if burst is not None:
        for c, wf in waterfalls:
            if c is burst:
                waterfall = wf
                break

    # Burst coords resolved against the cube we actually have.
    coords = _burst_coords(burst, waterfall, peak)
    if coords is not None and coords.t_idx is not None:
        _LOG.info(
            "plotter: %s peak placement — detector SNR=%.2f vs cube "
            "re-measure %.2f sigma at t=%d (boxcar w=%d); unsmoothed "
            "re-measure %.2f sigma at t=%s; delta(detector - matched)="
            "%+.2f sigma, delta(matched - unsmoothed)=%+.2f sigma",
            job.event_name, coords.snr, coords.snr_measured, coords.t_idx,
            coords.boxcar, coords.snr_measured_raw, coords.t_idx_raw,
            coords.snr_delta,
            coords.snr_measured - coords.snr_measured_raw,
        )
        if coords.boxcar > 1 and coords.t_idx_raw is not None:
            shift = coords.t_idx - coords.t_idx_raw
            if abs(shift) > 0:
                _LOG.info(
                    "plotter: %s width-matching moved the peak %+d "
                    "sample(s) (raw argmax t=%d -> matched t=%d)",
                    job.event_name, shift, coords.t_idx_raw, coords.t_idx,
                )
        if abs(coords.snr_delta) > _SNR_DELTA_WARN_SIGMA:
            _LOG.warning(
                "plotter: %s cube re-measurement is %+.1f sigma off the "
                "detector's SNR (%.2f reported vs %.2f matched at the "
                "detected pixel). Expected causes, in order of how often "
                "they bite: the detector normalises by the Layer-2 "
                "sigma_k EMA over many cubes while this is one cube's "
                "own median/MAD at one pixel; the dumped cube is a "
                "different pipeline stage than the scored one; or the "
                "dump is time-misaligned and this peak is not the "
                "candidate. Treat the panels as indicative until it is "
                "reconciled.",
                job.event_name, -coords.snr_delta, coords.snr,
                coords.snr_measured,
            )

    try:
        t0 = time.perf_counter()
        written.append(
            _render_dm_time(
                plots_dir, job.event_name, waterfalls, burst, coords,
            ),
        )
        _LOG.info(
            "plotter: dm_time rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )

        t0 = time.perf_counter()
        written.append(
            _render_image_peak(plots_dir, job.event_name, burst, coords),
        )
        _LOG.info(
            "plotter: image_peak rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )

        t0 = time.perf_counter()
        written.append(
            _render_lightcurve(
                plots_dir, job.event_name, waterfall, coords, burst,
            ),
        )
        _LOG.info(
            "plotter: lightcurve rendered in %.1fs (event=%s)",
            time.perf_counter() - t0, job.event_name,
        )

        t0 = time.perf_counter()
        written.append(
            _render_kernel_snrs(plots_dir, job.event_name, kernels),
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
    cube: np.ndarray  # shape (t_det, n_fdm, n_grid, n_grid), fp16, mmap'd
    fine_dm_pc_cc: np.ndarray  # shape (n_fdm,)
    mjd_start: float
    sample_period_us: float
    # TRUE sample-0 anchor (dumps from 2026-08-04 on). -1 / 0 / NaN when
    # the NPZ predates it, in which case _metadata_t_idx declines and the
    # width-matched argmax is used instead. See that function.
    #
    # cube_specnum_start is in SEARCH-SAMPLE units — the SAME units as a
    # C1 row's event_specnum — so their difference is directly an index
    # along axis 0. sample_period_specnum (native specnums per search
    # sample, 16 in production) is metadata, NOT a divisor.
    cube_specnum_start: int = -1
    sample_period_specnum: int = 0
    cube_mjd_start: float = float("nan")
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

            # Fine-DM axis: production NPZs don't store the per-trial
            # DM grid, so fall back to the trial *index* along the cube's
            # DM axis (axis 1). Real DM values, when needed, come from a
            # linear fit over the detection metadata (see _fit_dm_axis).
            fine_dm = manifest.get("fine_dm_pc_cc")
            n_fdm = cube.shape[1] if cube.ndim == 4 else cube.shape[0]
            if fine_dm is None:
                fine_dm = np.arange(n_fdm, dtype=np.float64)
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
            # `mjd_start` in a C2-triggered dump mirrors the manifest's
            # `event_specnum_start`, which that path overwrites with the
            # TRIGGER specnum (c2_trigger_listener._build_manifest). The
            # separately-recorded `cube_mjd_start` is the honest MJD of
            # cube sample 0, so prefer it whenever it is finite; NaN is
            # the writer's sentinel for "pre-2026-08-04 dump".
            cube_mjd_start = _scalar_or(npz, "cube_mjd_start", float("nan"))
            if np.isfinite(cube_mjd_start):
                mjd_start = float(cube_mjd_start)

            chunks.append(_CubeChunk(
                search_node_id=int(m.group("sid")),
                gpu_half=int(m.group("g")),
                event_specnum=int(m.group("specnum")),
                cube=cube,
                fine_dm_pc_cc=fine_dm,
                mjd_start=mjd_start,
                sample_period_us=sample_period_us,
                cube_specnum_start=int(
                    _scalar_or(npz, "cube_specnum_start", -1)
                ),
                sample_period_specnum=int(
                    _scalar_or(npz, "sample_period_specnum", 0)
                ),
                cube_mjd_start=float(cube_mjd_start),
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


# ---------------------------------------------------------------------------
# Burst resolution (from detection metadata, not cube argmax)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BurstPeak:
    """Detected peak of the C2 cluster, used to place every panel.

    Sourced from the live cluster members (``WindowEntry``) or the
    archived ``Level2/C1_window_<name>.csv`` rows — never from a raw
    cube argmax (which would latch onto bright steady continuum / RFI).
    """

    search_node_id: int
    gpu_half: int
    fine_dm_idx: int
    l_pix: int
    m_pix: int
    dm_pc_cc: float
    snr: float
    width_samples: int
    kernel_id: str
    source: str  # "members" | "csv"
    #: the candidate's absolute spec num, for _metadata_t_idx. 0 when the
    #: source row did not carry it.
    event_specnum: int = 0


@dataclass(frozen=True, slots=True)
class _BurstCoords:
    """Resolved panel coordinates within the burst cube.

    ``t_idx`` is the cube time sample of the burst (argmax of the
    width-matched DM light curve); ``fdm_idx`` the fine-DM trial row;
    ``(l_pix, m_pix)`` the image pixel. ``from_metadata`` distinguishes
    the trustworthy metadata path from the cube-argmax fallback (flagged
    on the plots).

    ``snr`` is what the **detector** reported. ``snr_measured`` is what
    the plotter re-measures on the dumped cube at ``t_idx`` after
    applying the detector's own boxcar, and ``snr_measured_raw`` the
    same without smoothing — the gap between the last two is the
    smoothing the detector applied and the post-processing used not to.
    ``boxcar`` is the width actually convolved (1 = none).
    """

    t_idx: Optional[int]
    fdm_idx: int
    l_pix: int
    m_pix: int
    dm_pc_cc: float
    snr: float
    width_samples: int
    n_fdm: int
    t_det: int
    from_metadata: bool
    snr_measured: float = float("nan")
    snr_measured_raw: float = float("nan")
    t_idx_raw: Optional[int] = None
    boxcar: int = 1
    #: True when t_idx came from the cube's sample-0 anchor (the
    #: detector's own index) rather than from a re-located peak.
    t_from_anchor: bool = False

    @property
    def snr_delta(self) -> float:
        """Detector SNR minus the width-matched re-measurement."""
        return float(self.snr) - float(self.snr_measured)


# ---------------------------------------------------------------------------
# Detector-matched boxcar smoothing
# ---------------------------------------------------------------------------


#: Log a WARNING when the cube re-measurement is this far from the
#: detector's reported SNR. Sized to catch "the dump doesn't contain the
#: burst" (tens of sigma), not ordinary normalisation differences.
_SNR_DELTA_WARN_SIGMA: float = 5.0


def _smoothing_enabled() -> bool:
    """``DSART_PLOTTER_SMOOTH=0`` restores the pre-2026-08-02 raw argmax."""
    return os.environ.get("DSART_PLOTTER_SMOOTH", "1").strip() not in (
        "0", "false", "False", "no",
    )


def _boxcar_width(peak: Optional[_BurstPeak]) -> int:
    """The detector's boxcar width for this candidate (>= 1).

    ``width_samples`` on the C1 row is the matched-filter width the
    winning kernel used, in cube samples. 0/absent (the no-metadata
    fallback) means "don't smooth".
    """
    if peak is None or not _smoothing_enabled():
        return 1
    return max(1, int(peak.width_samples))


def _boxcar(series: np.ndarray, width: int) -> np.ndarray:
    """Match-filter ``series`` with a ``width``-sample boxcar.

    Normalised by ``1/sqrt(width)`` so that white noise keeps unit
    variance and the output stays directly comparable to the detector's
    sigma units. ``width <= 1`` is a pass-through.
    """
    w = int(width)
    if w <= 1 or series.size == 0:
        return np.asarray(series, dtype=np.float32)
    kernel = np.full(w, 1.0 / float(np.sqrt(w)), dtype=np.float32)
    return np.convolve(
        np.asarray(series, dtype=np.float32), kernel, mode="same",
    ).astype(np.float32, copy=False)


def _robust_z(series: np.ndarray) -> np.ndarray:
    """Median/MAD-normalised series (sigma units), NaN-safe.

    A degenerate MAD (>half the samples identical — sparse synthetic
    cubes, or a mostly-zero row) falls back to the standard deviation
    rather than to zeros: the *scale* stops being a robust sigma, but
    the argmax still lands on the peak, which is what the caller needs.
    Only a genuinely constant series returns zeros.
    """
    x = np.asarray(series, dtype=np.float32)
    if x.size == 0:
        return x
    med = float(np.median(x))
    sigma = float(np.median(np.abs(x - med))) * 1.4826
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.zeros_like(x)
    return (x - med) / sigma


def _metadata_t_idx(
    chunk: Optional[_CubeChunk],
    peak: Optional[_BurstPeak],
) -> Optional[int]:
    """The detector's own in-cube time index, or ``None`` if unknowable.

    ``t = event_specnum - cube_specnum_start`` — a plain subtraction, no
    division.

    Units contract. BOTH operands count **search samples**, one per
    detector sample, so their difference is already an index along the
    cube's time axis:

      * ``cube_specnum_start`` is the retained cube's
        ``slot.specnum_start`` (``dump/c2_trigger_listener.py``
        ``_build_manifest``), which advances by ``cube_cadence_samples``
        per cube. ``services/search_compute.py:1338-1345`` states the
        rule outright: "specnum_start is in SEARCH-SAMPLE units … must
        NOT be divided by cube_sample_period_specnum" — the same trap,
        fixed once before for the MJD clock.
      * ``event_specnum`` is ``specnum_start + t_idx`` where ``t_idx``
        is the detector's in-cube sample index
        (``detector/decoder.py:216``).

    ``sample_period_specnum`` is native-SNAP-specnums per search sample
    (16 at the production op-point). It is emphatically **not** a
    divisor here; it is read only so a mis-scaled anchor can be named in
    the log. Dividing by it — as this function did from its
    introduction (c60556b) until 2026-08-06 — plots every burst at
    ``t_true // 16``, i.e. collapsed towards the cube start, always in
    range so the guard below never fires.

    Archive evidence for the units (Δ = event_specnum − anchor, all well
    inside one cube cadence of 192 samples): 260804xpnp Δ=10,
    260805pabq Δ=118, 260805tldd Δ=182. Under the divisor reading those
    same events would have plotted at t=0, 7 and 11.

    Only usable when the NPZ carries ``cube_specnum_start`` (dumps from
    2026-08-04 on) — before that the only anchor was
    ``event_specnum_start``, which the C2-trigger path overwrote with the
    trigger specnum, making the delta identically 0.

    Returns None on absent/sentinel anchors or an out-of-range result, so
    the caller falls back to the width-matched argmax.
    """
    if chunk is None or peak is None:
        return None
    anchor = getattr(chunk, "cube_specnum_start", None)
    if anchor is None or int(anchor) < 0:
        return None
    # Diagnostics only — see the units contract above. Never a divisor.
    period = int(getattr(chunk, "sample_period_specnum", 0) or 0)
    t = int(peak.event_specnum) - int(anchor)
    t_det = int(chunk.cube.shape[0]) if chunk.cube.ndim == 4 else 0
    if t_det and not (0 <= t < t_det):
        if period > 1 and 0 <= t // period < t_det:
            # The delta is ~period× too large: the writer handed us
            # NATIVE SNAP specnums where the contract says search
            # samples. That is a schema change, not a plotting problem —
            # guessing a divisor is exactly how the original bug got in.
            _LOG.error(
                "plotter: anchor delta %d is outside the cube [0, %d) but "
                "%d/%d = %d is inside — the anchor looks like NATIVE "
                "specnums, i.e. the writer's units contract changed. "
                "Refusing to guess; falling back to the matched argmax.",
                t, t_det, t, period, t // period,
            )
            return None
        _LOG.warning(
            "plotter: metadata t_idx=%d outside the cube [0, %d) — "
            "falling back to the matched argmax", t, t_det,
        )
        return None
    return int(t)


def _pixel_lightcurve(
    chunk: Optional[_CubeChunk],
    fdm: int,
    l_pix: int,
    m_pix: int,
) -> Optional[np.ndarray]:
    """``cube[:, fdm, l_pix, m_pix]`` — the series the detector scored.

    A strided read of ``t_det`` fp16 values out of the mmap (one page
    touch per sample, ~1 MB of I/O), so it is far cheaper than the
    ``max(axis=(2, 3))`` reduction that already ran.
    """
    if chunk is None or chunk.cube.ndim != 4 or not chunk.cube.size:
        return None
    try:
        t_det, n_fdm, n_l, n_m = chunk.cube.shape
        f = int(np.clip(fdm, 0, n_fdm - 1))
        li = int(np.clip(l_pix, 0, n_l - 1))
        mi = int(np.clip(m_pix, 0, n_m - 1))
        return np.asarray(chunk.cube[:, f, li, mi], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - never sink a plot job
        _LOG.warning("plotter: pixel light curve read failed: %s", exc)
        return None


def _read_window_csv_rows(archive_root: Path, event_name: str) -> List[dict]:
    """Read ``Level2/C1_window_<name>.csv`` rows (empty list if absent)."""
    csv_path = (
        archive_root / event_name / "Level2" / f"C1_window_{event_name}.csv"
    )
    if not csv_path.is_file():
        return []
    try:
        with open(csv_path, newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("plotter: failed to read %s: %s", csv_path, exc)
        return []


def _peak_from_members(
    members: Sequence[WindowEntry],
) -> Optional[_BurstPeak]:
    if not members:
        return None
    m = max(members, key=lambda e: float(e.snr))
    return _BurstPeak(
        search_node_id=int(m.search_node_id),
        gpu_half=int(m.gpu_half),
        fine_dm_idx=int(m.fine_dm_idx),
        l_pix=int(m.l_pix),
        m_pix=int(m.m_pix),
        dm_pc_cc=float(m.dm_pc_cc),
        snr=float(m.snr),
        width_samples=int(m.width_samples),
        kernel_id=str(m.kernel_id),
        source="members",
        event_specnum=int(getattr(m, "event_specnum", 0) or 0),
    )


def _peak_from_csv_rows(rows: Sequence[dict]) -> Optional[_BurstPeak]:
    best = None
    best_snr = -np.inf
    for r in rows:
        try:
            snr = float(r["snr"])
        except (KeyError, TypeError, ValueError):
            continue
        if snr > best_snr:
            best_snr = snr
            best = r
    if best is None:
        return None
    try:
        return _BurstPeak(
            search_node_id=int(best["search_node_id"]),
            gpu_half=int(best["gpu_half"]),
            fine_dm_idx=int(best["fine_dm_idx"]),
            l_pix=int(best["l_pix"]),
            m_pix=int(best["m_pix"]),
            dm_pc_cc=float(best["dm_pc_cc"]),
            snr=float(best["snr"]),
            width_samples=int(best["width_samples"]),
            kernel_id=str(best["kernel_id"]),
            source="csv",
            event_specnum=int(float(best.get("event_specnum", 0) or 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.warning("plotter: malformed C1-window peak row: %s", exc)
        return None


def _resolve_burst(
    job: PlotJob,
) -> Tuple[Optional[_BurstPeak], List[Tuple[str, float]]]:
    """Resolve the burst peak + per-kernel SNRs from detection metadata.

    Prefers the live cluster members; falls back to the archived
    ``C1_window`` CSV so offline re-renders work with no live state.
    Returns ``(peak_or_None, [(kernel_id, snr), ...])``.
    """
    if job.members:
        peak = _peak_from_members(job.members)
        kernels = [(str(m.kernel_id), float(m.snr)) for m in job.members]
        return peak, kernels
    rows = _read_window_csv_rows(job.archive_root, job.event_name)
    peak = _peak_from_csv_rows(rows)
    kernels: List[Tuple[str, float]] = []
    for r in rows:
        try:
            kernels.append((str(r["kernel_id"]), float(r["snr"])))
        except (KeyError, TypeError, ValueError):
            continue
    return peak, kernels


def _select_burst_chunk(
    cubes: Sequence[_CubeChunk],
    peak: Optional[_BurstPeak],
) -> Optional[_CubeChunk]:
    """The cube owning the burst (matching the peak's (s, g)).

    Falls back to the first valid 4-D cube when there's no metadata, so
    the cube-argmax path can still produce a (clearly flagged) plot.
    """
    valid = [c for c in cubes if c.cube.ndim == 4 and c.cube.size]
    if not valid:
        return None
    if peak is not None:
        for c in valid:
            if (c.search_node_id == peak.search_node_id
                    and c.gpu_half == peak.gpu_half):
                return c
        _LOG.warning(
            "plotter: burst cube s%d_g%d not found among %d cubes; "
            "using first available",
            peak.search_node_id, peak.gpu_half, len(valid),
        )
    return valid[0]


def _burst_waterfall(chunk: Optional[_CubeChunk]) -> Optional[np.ndarray]:
    """``cube.max(axis=(2, 3))`` → ``(t_det, n_fdm)`` fp32 for one
    cube (one full streaming reduction; ~25 MB working set)."""
    if chunk is None or chunk.cube.ndim != 4 or not chunk.cube.size:
        return None
    try:
        wf = chunk.cube.max(axis=(2, 3))
    except (TypeError, ValueError) as exc:
        _LOG.warning(
            "plotter: waterfall reduce failed on s%d_g%d: %s",
            chunk.search_node_id, chunk.gpu_half, exc,
        )
        return None
    return np.asarray(wf, dtype=np.float32)


def _all_waterfalls(
    cubes: Sequence[_CubeChunk],
) -> List[Tuple[_CubeChunk, np.ndarray]]:
    """Per-cube ``(t_det, n_fdm)`` waterfalls for every loadable cube,
    sorted by ``(search_node_id, gpu_half)`` — i.e. in fine-DM-coverage
    order at the production op-point (n01 g0 owns the lowest trials,
    n13 g1 the highest). Cubes whose reduction fails are skipped.
    """
    out: List[Tuple[_CubeChunk, np.ndarray]] = []
    for c in sorted(cubes, key=lambda c: (c.search_node_id, c.gpu_half)):
        t0 = time.perf_counter()
        wf = _burst_waterfall(c)
        if wf is None:
            continue
        _LOG.info(
            "plotter: waterfall s%d_g%d reduced in %.1fs",
            c.search_node_id, c.gpu_half, time.perf_counter() - t0,
        )
        out.append((c, wf))
    return out


def _burst_coords(
    chunk: Optional[_CubeChunk],
    waterfall: Optional[np.ndarray],
    peak: Optional[_BurstPeak],
) -> Optional[_BurstCoords]:
    """Resolve in-cube panel coordinates.

    cube axes are ``(t_det, n_fdm, l, m)``. With metadata we take the
    DM row + (l, m) from the peak, and the *time* from the cube's
    sample-0 anchor when it carries one; failing that, from the argmax
    of the DM light curve (the stored ``event_specnum_start`` is the
    dump key, not the cube's first-sample specnum, so it cannot serve
    as an anchor). Without metadata at all we fall back to the
    waterfall global argmax (flagged ``from_metadata=False``).

    The DM light curve is the **detected pixel's** series when the cube
    is readable, match-filtered with the detector's own
    ``width_samples`` boxcar — see the module docstring for why both of
    those matter. We also re-measure the significance at the located
    peak, with and without the boxcar, so callers can print the gap
    against the detector's reported SNR.
    """
    # 2026-08-04: if the cube carries the TRUE sample-0 anchor
    # (`cube_specnum_start`), the detector's own time index is directly
    # computable — `event_specnum - anchor`, both in search samples —
    # and we no longer have to relocate the burst at all. Older dumps
    # have only `event_specnum_start`, which the C2-trigger path
    # overwrote with the trigger specnum, so `_metadata_t_idx` returns
    # None for them and we fall back to the width-matched argmax below.
    t_meta = _metadata_t_idx(chunk, peak)

    if waterfall is None or waterfall.ndim != 2 or not waterfall.size:
        if peak is None:
            return None
        # Metadata but no usable cube: still emit DM/(l,m) for context.
        return _BurstCoords(
            t_idx=None, fdm_idx=int(peak.fine_dm_idx),
            l_pix=int(peak.l_pix), m_pix=int(peak.m_pix),
            dm_pc_cc=peak.dm_pc_cc, snr=peak.snr,
            width_samples=peak.width_samples,
            n_fdm=0, t_det=0, from_metadata=True,
        )
    t_det, n_fdm = int(waterfall.shape[0]), int(waterfall.shape[1])
    if peak is not None:
        fdm = int(np.clip(peak.fine_dm_idx, 0, n_fdm - 1))
        l_pix, m_pix = int(peak.l_pix), int(peak.m_pix)
        # Prefer the detected pixel; fall back to the image-max row when
        # the strided cube read fails.
        series = _pixel_lightcurve(chunk, fdm, l_pix, m_pix)
        if series is None:
            series = np.asarray(waterfall[:, fdm], dtype=np.float32)
        z_raw = _robust_z(series)
        box = _boxcar_width(peak)
        # Note the order: smooth FIRST, then take median/MAD of the
        # smoothed series. Normalising first and convolving after would
        # score the burst against sigma_raw/sqrt(w), i.e. against a
        # white-noise assumption; the detector normalises per boxcar
        # width (Layer-2 publishes s_k_unit_d1_b1..b64), so measuring
        # sigma on the smoothed series is the like-for-like comparison.
        # When the two disagree the residual is correlated on the boxcar
        # scale — logged as the whiteness ratio in render_event_plots.
        z_smooth = _robust_z(_boxcar(series, box)) if box > 1 else z_raw
        t_raw = int(np.argmax(z_raw))
        if t_meta is not None:
            # The cube carries a trustworthy sample-0 anchor, so this IS
            # the detector's own time index -- use it directly rather than
            # relocating the burst. This is what the panels were always
            # supposed to show.
            t_idx = int(t_meta)
        else:
            t_idx = int(np.argmax(z_smooth))
        return _BurstCoords(
            t_idx=t_idx, fdm_idx=fdm, l_pix=l_pix, m_pix=m_pix,
            dm_pc_cc=peak.dm_pc_cc, snr=peak.snr,
            width_samples=peak.width_samples,
            n_fdm=n_fdm, t_det=t_det, from_metadata=True,
            snr_measured=float(z_smooth[t_idx]),
            snr_measured_raw=float(z_raw[t_raw]),
            t_idx_raw=t_raw, boxcar=box,
            t_from_anchor=(t_meta is not None),
        )
    # Fallback: global argmax of the waterfall, (l, m) from that plane.
    flat = int(np.argmax(waterfall))
    t_idx, fdm = (int(v) for v in np.unravel_index(flat, waterfall.shape))
    dm_val = float(chunk.fine_dm_pc_cc[fdm]) if (
        chunk is not None and fdm < len(chunk.fine_dm_pc_cc)
    ) else float(fdm)
    l_pix = m_pix = 0
    if chunk is not None:
        try:
            plane = np.asarray(chunk.cube[t_idx, fdm], dtype=np.float32)
            l_pix, m_pix = (
                int(v) for v in np.unravel_index(int(plane.argmax()), plane.shape)
            )
        except Exception:  # noqa: BLE001
            pass
    return _BurstCoords(
        t_idx=t_idx, fdm_idx=fdm, l_pix=l_pix, m_pix=m_pix,
        dm_pc_cc=dm_val, snr=float(waterfall.flat[flat]),
        width_samples=0, n_fdm=n_fdm, t_det=t_det, from_metadata=False,
    )


# ----- panel renderers ----------------------------------------------------


_RETICLE = "#ff2d55"  # high-contrast reticle colour over magma/viridis

# v3.2: tick/colorbar-tick/axis-label/legend font sizes, bumped ~2x from
# the prior ad hoc 7-10pt values so the archived PNGs stay legible once
# the candidate card (and the dashboard) display them at ~800px panel
# width. 13pt read as barely-larger against the old 8-10pt ticks at
# that display width, so this went to 16pt (round-tripped visually via
# font_comparison.png). Font sizes only — no layout/content changes.
_TICK_FONTSIZE = 16
_AXIS_LABEL_FONTSIZE = 18
# v3.4: the embedded ax.set_title strings were still the original ad hoc
# 10pt (or, for kernel_snrs, matplotlib's rc default) — nearly invisible
# next to the v3.2 16-18pt ticks/axis-labels at the ~800px panel width
# the candidate card displays these at. Bumped to 18pt (same size as the
# axis-label tier) so the title reads as the most prominent text on the
# panel. The small sub-annotations (per-kernel "n=" bar counts, and the
# below-axes marker legends / captions — "brightest point of displayed
# data", "detector-reported burst", "red circle = burst position
# reported by the detector") are a distinct, smaller tier: dialed from
# the prior ad hoc 17pt/9pt values down/up to a consistent ~14pt so the
# hierarchy reads title > axis-label/ticks > sub-annotation. Font sizes
# only — no layout/content changes (margins adjusted separately below
# only where a bigger font newly clipped/collided; see v3.4 notes at
# each such spot).
_TITLE_FONTSIZE = 16
_LEGEND_FONTSIZE = 14

# v3.5: the two small panels (lightcurve, kernel_snrs — both drawn at a
# compact ~8x3.6-4.0in figsize, versus dm_time/image_peak's 10x9in) read
# as disproportionately dense at the shared 18pt axis-label size — e.g.
# lightcurve's rotated "peak amplitude (image max)" ylabel visibly
# crowded that panel's short y-axis. Axis labels on those two panels
# only are dialed down a notch; ticks/legend are left alone (already a
# smaller tier). dm_time/image_peak (the big panels) keep
# _AXIS_LABEL_FONTSIZE as-is.
_AXIS_LABEL_FONTSIZE_SMALL = 14

# Shared figure geometry for the dm_time and image_peak panels: both
# figures are 10.0x9.0 in @ dpi 110 (1100x990 px) with the axes box,
# colorbar, and caption/legend line at identical figure fractions, so
# the two PNGs align pixel-for-pixel when the dashboard renders them
# side by side at equal width — by construction, not tuning.
# v3.3: bottom margin raised 0.115->0.19 (height 0.80->0.725, same top
# edge 0.915) to make room below the axes for the bigger v3.2 tick/axis-
# label/legend fonts — at the old 0.115 margin, dm_time's xlabel + its
# fig-level marker legend (both now larger) collided with each other.
# Top edge held fixed so dm_time/image_peak still align pixel-for-pixel.
_PANEL_RECT = (0.10, 0.19, 0.72, 0.725)   # left, bottom, width, height
_CBAR_RECT = (0.845, 0.19, 0.022, 0.725)
_CAPTION_Y = 0.03   # baseline of the marker-legend/caption line
_CAPTION_X = 0.46   # horizontal centre = centre of the panel box

# v3.6: the dm_time/image_peak title is drawn as two visual tiers rather
# than one two-line ax.set_title (see _fit_title_line and the notes at
# each call site). Line 1 is the short event identity at
# _TITLE_FONTSIZE; line 2 is the burst parameters plus the growing
# _snr_note diagnostic, drawn separately so its font size can be fitted
# to the axes width. _SUBTITLE_Y is the line-2 baseline in axes
# fractions (just above the axes top edge); _TITLE_PAD is the
# ax.set_title pad, in points, that lifts line 1 clear of line 2.
_SUBTITLE_FONTSIZE = _LEGEND_FONTSIZE   # line-2 starting (maximum) size
_SUBTITLE_FONTSIZE_MIN = 10             # never shrink below this
_SUBTITLE_Y = 1.018                     # axes fraction, line-2 baseline
_TITLE_PAD = 26.0                       # points, axes top → line-1 bottom


def _fit_title_line(
    fig,
    ax,
    text: str,
    y: float = _SUBTITLE_Y,
    max_fontsize: int = _SUBTITLE_FONTSIZE,
    min_fontsize: int = _SUBTITLE_FONTSIZE_MIN,
):
    """Draw ``text`` centred above ``ax``, shrunk until it fits its width.

    The dm_time/image_peak axes box is pinned (``_PANEL_RECT``) so the
    two panels align pixel-for-pixel, and the diagnostic second title
    line (burst parameters + ``_snr_note``) has grown past what fits at
    ``_TITLE_FONTSIZE``. Rather than tune a font size to today's longest
    string, measure the rendered text against the axes box with the
    figure's own renderer and step down 1pt at a time (floor
    ``min_fontsize``) until it fits. Future additions to ``_snr_note``
    therefore shrink the line instead of silently clipping it.

    ``min_fontsize`` is a legibility floor, so a pathologically long
    string can still exceed the axes width — but the fitted target is
    the axes box (720 px at the panel geometry), while the hard clip
    boundary is the figure edge (1100 px), so hitting the floor looks
    wide rather than truncated.

    Returns the ``Text`` artist (its ``get_fontsize()`` is the size the
    line landed at).
    """
    artist = ax.text(
        0.5, y, text, transform=ax.transAxes, ha="center", va="baseline",
        fontsize=max_fontsize, color="#2e3440",
    )
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:      # backend without a cached renderer
        return artist
    avail = ax.get_window_extent(renderer).width
    size = max_fontsize
    while size > min_fontsize:
        if artist.get_window_extent(renderer).width <= avail:
            break
        size -= 1
        artist.set_fontsize(size)
    return artist


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


def _signed_sigma(x: float) -> str:
    """``+13.6σ`` / ``−12.6σ`` — a proper minus sign, not a hyphen."""
    sign = "+" if x >= 0.0 else "−"
    return f"{sign}{abs(x):.1f}σ"


def _provenance(coords: Optional[_BurstCoords]) -> str:
    """" · re-searched in cube" / " · no detector info — showing
    brightest pixel" — flags the two fallback placements.

    The healthy/default path — the cube carries a trustworthy sample-0
    anchor, so the panel shows the detector's own time index directly —
    prints nothing: that is what every panel is supposed to show, and a
    tag on it would just be noise. The two fallbacks each get a plain-
    English flag: metadata is present but the cube has no anchor, so
    the burst time was relocated by a width-matched search of the
    dumped cube (not the detector's own index); or there is no
    detection metadata at all, so the panel falls back to the cube's
    single brightest pixel, which can just as easily be bright steady
    continuum or RFI as the burst. ("re-searched in cube" is the short
    form of "burst time re-searched in the cube" — the panel titles are
    two fixed-width lines inside a pinned axes box (see ``_PANEL_RECT``)
    and the long form doesn't fit a worst-case (4-digit-DM, wide-boxcar)
    title even at the font-size floor.)
    """
    if coords is None:
        return ""
    if not coords.from_metadata:
        return " · no detector info — showing brightest pixel"
    if coords.t_from_anchor:
        return ""
    return " · re-searched in cube"


def _snr_note(coords: Optional[_BurstCoords]) -> str:
    """" · 6.1σ (archived cube) · Δ = +13.6σ" — the cube
    re-measurement, printed next to the detector's SNR on every panel
    so the two are never silently conflated. ``Δ`` is the detector's
    SNR minus this re-measurement (positive means the detector read
    higher). The boxcar width and the unsmoothed re-measurement behind
    this number are still written to the log (see the "cube
    re-measurement" INFO/WARNING lines in ``render_event_plots``) —
    only the title itself has been trimmed to the plain essentials.
    Omitted entirely when there is no cube to re-measure against.
    """
    if coords is None or not np.isfinite(coords.snr_measured):
        return ""
    return (
        f" · {coords.snr_measured:.1f}σ (archived cube) · "
        f"Δ = {_signed_sigma(coords.snr_delta)}"
    )


def _robust_row_normalise(img: np.ndarray) -> np.ndarray:
    """Per-row (fine-DM) robust z-score of a ``(n_rows, t)`` waterfall.

    Each row is the time series of the per-plane image MAX, so its
    baseline is an extreme-value statistic (≈4–5 σ for a 256×256
    Gaussian plane) whose level/spread varies slightly per fine-DM
    trial and — when stacking cubes from different halves — per
    (search_node, gpu_half) Layer-1/2 calibration. Subtracting the
    per-row median and dividing by 1.4826×MAD puts every row in
    comparable "σ above its own quiescent max" units without letting
    the burst itself bias the scale (median/MAD ignore a few bright
    samples). Rows with zero MAD (constant) fall back to σ=1.
    """
    med = np.nanmedian(img, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(img - med), axis=1, keepdims=True)
    sigma = 1.4826 * mad
    sigma[~np.isfinite(sigma) | (sigma <= 0.0)] = 1.0
    return (img - med) / sigma


def _render_dm_time(
    plots_dir: Path,
    event_name: str,
    waterfalls: Sequence[Tuple[_CubeChunk, np.ndarray]],
    burst: Optional[_CubeChunk],
    coords: Optional[_BurstCoords],
) -> Path:
    """Single stacked DM × time waterfall over ALL dumped cubes.

    2026-06-10 redesign (v2): one panel, with every cube's fine-DM
    rows stacked in (search_node, gpu_half) order — contiguous DM
    coverage over the whole searched range, so the full dispersion
    bowtie (owning half + non-owning tails) reads as one figure.
    Rows are robustly re-normalised (see :func:`_robust_row_normalise`)
    so all halves share one σ colour scale. Dashed lines mark half
    boundaries; the crosshair marks the detected (t_peak, DM_peak).

    2026-08-02: every row is first match-filtered along time with the
    detector's ``width_samples`` boxcar, so the image shows the cube the
    detector actually thresholded rather than the unsmoothed one. A
    ``width_samples=16`` burst is ~4x more visible here than in the raw
    cube. The boxcar is annotated in the title; ``DSART_PLOTTER_SMOOTH=0``
    turns it off.
    """
    path = plots_dir / f"dm_time_{event_name}.png"
    if not waterfalls:
        return _placeholder(path, f"DM × time — {event_name}", "no cube data")
    import matplotlib.pyplot as plt

    # Stack to (n_rows_total, t): waterfalls arrive (t_det, n_fdm)
    # sorted by (sid, g) == ascending DM coverage. Crop to the common
    # time length in case a half dumped a different-geometry cube.
    t_common = min(int(wf.shape[0]) for _, wf in waterfalls)
    blocks = [
        np.asarray(wf[:t_common, :].T, dtype=np.float32)
        for _, wf in waterfalls
    ]
    # Match-filter along time BEFORE the per-row σ normalisation, so the
    # colour scale is σ of the *smoothed* series — directly comparable to
    # the detector's own units and to `coords.snr_measured`.
    box = int(coords.boxcar) if coords is not None else 1
    if box > 1:
        blocks = [
            np.stack([_boxcar(row, box) for row in blk], axis=0)
            for blk in blocks
        ]
    stacked = _robust_row_normalise(np.concatenate(blocks, axis=0))

    # Real DM per stacked row, pulled from the production DM plan and keyed
    # by each cube's owned coarse-DM bucket (the dumped NPZ does not carry
    # per-row DMs). row_dm[i] is None when the plan is unavailable or a
    # bucket's size doesn't match the cube, so the y-axis can fall back.
    bucket_dms = _load_coarse_bucket_dms()
    present_sids = [int(c.search_node_id) for c, _ in waterfalls]

    # Per-half row offsets (for boundary lines, labels, crosshair).
    offsets: List[int] = []
    labels: List[str] = []
    row_dm = np.full(stacked.shape[0], np.nan, dtype=np.float64)
    row0 = 0
    burst_row: Optional[int] = None
    for (chunk, _), blk in zip(waterfalls, blocks):
        offsets.append(row0)
        labels.append(f"s{chunk.search_node_id}g{chunk.gpu_half}")
        n_blk = blk.shape[0]
        dms = None
        if bucket_dms is not None:
            owner = _owner_coarse_idx(
                chunk.search_node_id, chunk.gpu_half, present_sids,
            )
            cand = bucket_dms.get(owner)
            if cand is not None and cand.shape[0] == n_blk:
                dms = cand
        if dms is None:
            # Fall back to whatever the NPZ carried (real DMs if present,
            # else trial indices) so the row is never left blank.
            fb = np.asarray(chunk.fine_dm_pc_cc, dtype=np.float64)
            if fb.shape[0] >= n_blk:
                dms = fb[:n_blk]
        if dms is not None:
            row_dm[row0:row0 + n_blk] = dms
        if burst is not None and chunk is burst and coords is not None:
            burst_row = row0 + int(np.clip(coords.fdm_idx, 0, n_blk - 1))
        row0 += n_blk
    n_rows = row0
    have_dm_axis = bool(np.isfinite(row_dm).any())

    # Colour scale in robust-σ units: floor a little below baseline,
    # ceiling at the brighter of the p99.9 tail and half the true
    # peak so the detected burst can't be normalised off the page.
    finite = stacked[np.isfinite(stacked)]
    if finite.size:
        vmax = float(np.percentile(finite, 99.9))
        vmax = max(vmax, 0.5 * float(finite.max()), 1.0)
    else:
        vmax = 1.0
    vmin = -2.0

    fig = plt.figure(figsize=(10.0, 9.0))
    ax = fig.add_axes(_PANEL_RECT)
    im = ax.imshow(
        stacked, aspect="auto", origin="lower", cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    # v3.2: tick/axis-label/colorbar font sizes bumped ~1.5-2x across this
    # module (dm_time, image_peak, lightcurve, kernel_snrs) so the
    # archived PNGs are legible at the ~800px panel width the candidate
    # card (and the dashboard) display them at. Font sizes only — no
    # layout/content changes.
    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
    cax = fig.add_axes(_CBAR_RECT)
    cb = fig.colorbar(
        im, cax=cax,
        label=(
            "image-max amplitude (robust σ per DM row"
            + (f", boxcar w={box}" if box > 1 else "")
            + ")"
        ),
    )
    cb.ax.tick_params(labelsize=_TICK_FONTSIZE)
    cb.set_label(cb.ax.get_ylabel(), fontsize=_AXIS_LABEL_FONTSIZE)

    # Half boundaries (dashed) so the (search node, gpu half) bands are
    # still legible; the band id is annotated at the right edge.
    next_offsets = offsets[1:] + [n_rows]
    for off in offsets[1:]:
        ax.axhline(off - 0.5, color="w", lw=0.7, ls=":", alpha=0.7)
    for lab, off, nxt in zip(labels, offsets, next_offsets):
        # get_yaxis_transform: x in axes fraction, y in data (row) coords.
        ax.text(
            1.002, (off + nxt) / 2.0, lab,
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=7, color="0.35", clip_on=False,
        )

    if have_dm_axis:
        # Label the y axis with real DMs from the plan. Place ticks at
        # evenly spaced rows and read the DM at each row (rows are in
        # ascending-DM order across stacked halves).
        finite_rows = np.flatnonzero(np.isfinite(row_dm))
        n_ticks = min(12, finite_rows.size)
        tick_rows = np.linspace(
            finite_rows[0], finite_rows[-1], n_ticks,
        ).round().astype(int)
        tick_rows = np.unique(tick_rows)
        ax.set_yticks(tick_rows)
        ax.set_yticklabels(
            [f"{row_dm[r]:.0f}" for r in tick_rows],
            fontsize=_TICK_FONTSIZE,
        )
        ax.set_ylabel("dispersion measure (pc cm⁻³)",
                     fontsize=_AXIS_LABEL_FONTSIZE)
    else:
        band_centres = [
            (off + nxt) / 2.0 for off, nxt in zip(offsets, next_offsets)
        ]
        ax.set_yticks(band_centres)
        ax.set_yticklabels(labels, fontsize=_TICK_FONTSIZE)
        ax.set_ylabel(
            "fine-DM trials, stacked by (search node, gpu half) — "
            "increasing DM ↑",
            fontsize=_AXIS_LABEL_FONTSIZE,
        )
    ax.set_xlabel("time sample (within cube)", fontsize=_AXIS_LABEL_FONTSIZE)

    # Data apex (independent of the detector): global max of the
    # stacked, row-normalised waterfall. When detection is healthy the
    # white × lands on the red + (the C1 candidate); a separation
    # means the detector did NOT report the bowtie centre (e.g. the
    # fp16 boxcar overflow vanishing the apex candidate, 2026-06-10).
    if finite.size:
        apex_row, apex_t = (
            int(v) for v in np.unravel_index(
                int(np.nanargmax(stacked)), stacked.shape,
            )
        )
        ax.plot(
            apex_t, apex_row, marker="x", color="w", ms=12, mew=2.0,
            ls="none", label="brightest point of displayed data",
        )

    title = f"DM × time (waterfall) — {event_name}"
    subtitle = ""
    if coords is not None:
        if coords.t_idx is not None:
            ax.axvline(
                coords.t_idx, color=_RETICLE, lw=1.0, ls="--", alpha=0.9,
            )
        if burst_row is not None:
            ax.axhline(
                burst_row, color=_RETICLE, lw=1.0, ls="--", alpha=0.9,
            )
            if coords.t_idx is not None:
                ax.plot(
                    coords.t_idx, burst_row, marker="+",
                    color=_RETICLE, ms=16, mew=2.0,
                    label="detector-reported burst",
                )
        subtitle = (
            f"DM {coords.dm_pc_cc:.1f} pc cm⁻³ · "
            f"SNR {coords.snr:.1f} (detector)" + _snr_note(coords)
            + _provenance(coords)
        )
    # v3.6: two visual tiers, not one two-line set_title. v3.5 put the
    # event identity on line 1 and the burst parameters on line 2 of a
    # single 16pt title; since then line 2 has grown the _snr_note
    # diagnostic (cube re-measurement, agreement Δ, a fallback-placement
    # flag) and at 16pt it overflowed the axes box at both figure edges.
    # The box cannot grow: the "shared figure geometry" note above _PANEL_RECT
    # pixel-aligns dm_time/image_peak, so the text has to yield. Line 1
    # stays an ax.set_title at _TITLE_FONTSIZE (lifted by _TITLE_PAD to
    # clear line 2); line 2 is drawn by _fit_title_line, which measures
    # the rendered string against the axes width and steps the font size
    # down until it fits — fitted by measurement, not tuned to today's
    # longest string, so a future _snr_note addition shrinks rather than
    # clips. The margin above the axes still holds both tiers.
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad=_TITLE_PAD)
    if subtitle:
        _fit_title_line(fig, ax, subtitle)
    # Legend lives below the axes so it never overlaps the waterfall,
    # title, axis labels, or colorbar. No frame: plain text on the white
    # figure margin. The in-axes × marker is pure white (legible on the
    # waterfall), which would be invisible on the white margin, so the
    # legend uses proxy handles only — a filled "X" with a grey edge for
    # the apex marker; the data markers themselves are untouched.
    from matplotlib.lines import Line2D
    proxy_handles = []
    if finite.size:
        proxy_handles.append(Line2D(
            [], [], linestyle="none", marker="X",
            markerfacecolor="white", markeredgecolor="#555555",
            markeredgewidth=0.8, markersize=9,
            label="brightest point of displayed data",
        ))
    if coords is not None and burst_row is not None \
            and coords.t_idx is not None:
        proxy_handles.append(Line2D(
            [], [], linestyle="none", marker="+",
            color="#ff2d55", markeredgewidth=2, markersize=10,
            label="detector-reported burst",
        ))
    if proxy_handles:
        fig.legend(
            handles=proxy_handles, loc="lower center",
            bbox_to_anchor=(_CAPTION_X, _CAPTION_Y - 0.016),
            ncol=2, fontsize=_LEGEND_FONTSIZE, frameon=False,
            labelcolor="#3b4252",
        )
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _render_image_peak(
    plots_dir: Path,
    event_name: str,
    chunk: Optional[_CubeChunk],
    coords: Optional[_BurstCoords],
) -> Path:
    path = plots_dir / f"image_peak_{event_name}.png"
    if chunk is None or chunk.cube.ndim != 4 or not chunk.cube.size:
        return _placeholder(
            path, f"image at peak — {event_name}", "no cube data",
        )
    if coords is None or coords.t_idx is None:
        return _placeholder(
            path, f"image at peak — {event_name}", "no detection metadata",
        )
    import matplotlib.pyplot as plt
    # cube[t, fdm] → (l, m) image plane (~128 KB). axis 0 = l, axis 1 = m.
    t_idx = int(np.clip(coords.t_idx, 0, chunk.cube.shape[0] - 1))
    fdm = int(np.clip(coords.fdm_idx, 0, chunk.cube.shape[1] - 1))
    # 2026-08-02: for a wide burst, average the plane over the detector's
    # own boxcar centred on the matched peak — same match filter the
    # detector applied, so the panel shows the image it actually
    # triggered on instead of one arbitrary 1.05 ms slice of it.
    box = int(coords.boxcar)
    if box > 1:
        t_det_n = int(chunk.cube.shape[0])
        t0 = max(0, t_idx - box // 2)
        t1 = min(t_det_n, t0 + box)
        t0 = max(0, t1 - box)
        img = np.asarray(
            chunk.cube[t0:t1, fdm], dtype=np.float32,
        ).mean(axis=0)
    else:
        img = np.asarray(chunk.cube[t_idx, fdm], dtype=np.float32)
    fig = plt.figure(figsize=(10.0, 9.0))
    ax = fig.add_axes(_PANEL_RECT)
    im = ax.imshow(img, origin="lower", cmap="magma")
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    cax = fig.add_axes(_CBAR_RECT)
    cb = fig.colorbar(im, cax=cax, label="amplitude")
    cb.ax.tick_params(labelsize=_TICK_FONTSIZE)
    cb.set_label(cb.ax.get_ylabel(), fontsize=_AXIS_LABEL_FONTSIZE)
    # Reticle at the detected (l, m): x = m (cols), y = l (rows).
    ax.axhline(coords.l_pix, color=_RETICLE, lw=0.8, ls="--", alpha=0.8)
    ax.axvline(coords.m_pix, color=_RETICLE, lw=0.8, ls="--", alpha=0.8)
    ax.add_patch(plt.Circle(
        (coords.m_pix, coords.l_pix), radius=8.0,
        fill=False, edgecolor=_RETICLE, lw=1.8,
    ))
    ax.set_xlabel("m (pix)", fontsize=_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("l (pix)", fontsize=_AXIS_LABEL_FONTSIZE)
    tlabel = f"t={t_idx}" + (f", mean of w={box}" if box > 1 else "")
    title = f"image at (DM={coords.dm_pc_cc:.1f}, {tlabel}) — {event_name}"
    subtitle = (
        f"detected pixel (l,m)=({coords.l_pix},{coords.m_pix})"
        + _snr_note(coords) + _provenance(coords)
    )
    # v3.6: two visual tiers — see the matching note in _render_dm_time.
    # Event identity on line 1 at _TITLE_FONTSIZE; burst (l, m) plus the
    # _snr_note diagnostic on line 2, fitted to the pinned axes width by
    # _fit_title_line. Both panels share the helper so the two tiers
    # read the same size side by side whenever the strings are.
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad=_TITLE_PAD)
    _fit_title_line(fig, ax, subtitle)
    # Geometry is shared with _render_dm_time via _PANEL_RECT /
    # _CBAR_RECT / _CAPTION_Y, so both figures align by construction
    # (same 1100x990 PNG, same axes box, caption on the same line as
    # the dm_time marker legend).
    fig.text(
        _CAPTION_X, _CAPTION_Y,
        "red circle = burst position reported by the detector",
        ha="center", va="baseline", fontsize=_LEGEND_FONTSIZE,
        color="#3b4252",
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _render_lightcurve(
    plots_dir: Path,
    event_name: str,
    waterfall: Optional[np.ndarray],
    coords: Optional[_BurstCoords],
    burst: Optional[_CubeChunk] = None,
) -> Path:
    path = plots_dir / f"lightcurve_{event_name}.png"
    if waterfall is None or waterfall.size == 0:
        return _placeholder(path, f"lightcurve — {event_name}", "no cube data")
    if coords is None:
        return _placeholder(
            path, f"lightcurve — {event_name}", "no detection metadata",
        )
    import matplotlib.pyplot as plt
    fdm = int(np.clip(coords.fdm_idx, 0, waterfall.shape[1] - 1))
    # 2026-08-02: primary trace is the DETECTED PIXEL's series — that is
    # what the detector scored. The image-max row is kept as a faint
    # context trace (it is an extreme-value statistic over 65 536 pixels,
    # so its σ scale is not the point-source one).
    pix = _pixel_lightcurve(burst, fdm, coords.l_pix, coords.m_pix)
    imax = _robust_z(np.asarray(waterfall[:, fdm], dtype=np.float32))
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(imax, color="#b2bec3", lw=1.0, alpha=0.9,
            label="image max (context)")
    if pix is not None:
        z_raw = _robust_z(pix)
        ax.plot(z_raw, color="#0984e3", lw=1.2,
                label=f"pixel ({coords.l_pix},{coords.m_pix}) raw")
        if coords.boxcar > 1:
            ax.plot(
                _robust_z(_boxcar(pix, coords.boxcar)),
                color="#d63031", lw=1.8,
                label=f"pixel, boxcar w={coords.boxcar} (detector-matched)",
            )
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    if coords.t_idx is not None:
        ax.axvline(
            coords.t_idx, color=_RETICLE, lw=1.4, ls="--",
            label=f"matched peak t={coords.t_idx}",
        )
    if (coords.t_idx_raw is not None
            and coords.t_idx_raw != coords.t_idx):
        ax.axvline(
            coords.t_idx_raw, color="#636e72", lw=1.0, ls=":",
            label=f"unsmoothed argmax t={coords.t_idx_raw}",
        )
    ax.legend(loc="upper right", fontsize=_LEGEND_FONTSIZE)
    ax.set_xlabel("time sample (within cube)",
                 fontsize=_AXIS_LABEL_FONTSIZE_SMALL)
    ax.set_ylabel("σ above the row's own median",
                 fontsize=_AXIS_LABEL_FONTSIZE_SMALL)
    # v3.7: single-line title — the old second line (DM+SNR) duplicated
    # the dm_time title / card header and its extra line height pushed
    # this panel's axes box out of alignment with its row-mate
    # (kernel_snrs has a one-line title). Provenance marker stays (it is
    # a data-quality warning, not duplicated info).
    title = (f"lightcurve — {event_name}" + _snr_note(coords)
             + _provenance(coords))
    ax.set_title(title, fontsize=_TITLE_FONTSIZE)
    # v3.3: dropped fig.tight_layout() — with the bigger v3.2 ylabel font
    # it was clipping the rotated ylabel's left edge (reproduced in
    # isolation: tight_layout() pins subplot params from a pre-render
    # estimate that undershoots for this rotated/enlarged label, and that
    # pinned layout was NOT being corrected by the savefig-time bbox
    # recompute below). bbox_inches="tight" alone measures the actual
    # rendered artist extents and expands the saved canvas to fit them,
    # which is sufficient (verified: no more clipping) and needs no
    # separate layout call.
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_kernel_snrs(
    plots_dir: Path,
    event_name: str,
    kernels: Sequence[Tuple[str, float]],
) -> Path:
    path = plots_dir / f"kernel_snrs_{event_name}.png"
    if not kernels:
        return _placeholder(
            path, f"kernel SNR distribution — {event_name}",
            "no cluster members",
        )
    import matplotlib.pyplot as plt
    by_kernel: dict[str, List[float]] = {}
    for kid, snr in kernels:
        by_kernel.setdefault(kid, []).append(float(snr))
    labels = sorted(by_kernel.keys())
    maxes = [max(by_kernel[k]) for k in labels]
    counts = [len(by_kernel[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, maxes, color="#fdcb6e", edgecolor="#2d3436")
    # 12% y-headroom so the above-bar "n=" annotations never clip against
    # the axes top border when a bar spans nearly the full height.
    if maxes and max(maxes) > 0:
        ax.set_ylim(0, max(maxes) * 1.12)
    for x, c, b in zip(xs, counts, bars):
        ax.text(
            x, b.get_height(), f"n={c}",
            ha="center", va="bottom", fontsize=_LEGEND_FONTSIZE,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       fontsize=_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
    ax.set_ylabel("max SNR per kernel", fontsize=_AXIS_LABEL_FONTSIZE_SMALL)
    ax.set_title(f"kernel SNR — {event_name}", fontsize=_TITLE_FONTSIZE)
    # v3.3: dropped fig.tight_layout() — see the matching comment in
    # _render_lightcurve; bbox_inches="tight" alone is sufficient and
    # avoids the same clipping hazard for this panel's rotated xtick
    # labels / ylabel.
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Offline re-render (remake plots for archived events with cubes)
# ---------------------------------------------------------------------------


def regenerate_recent_events(
    archive_root: Path,
    *,
    event_names: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """Re-render plots for archived events that have dumped cubes.

    Resolves the burst from each event's ``Level2/C1_window_<name>.csv``
    (no live state needed) and rewrites the 4 PNGs into
    ``Level2/plots/``. When ``event_names`` is omitted, scans
    ``archive_root`` for event dirs containing ``cubes/*.npz``, newest
    first, optionally capped at ``limit``. Returns the event names
    processed.
    """
    archive_root = Path(archive_root)
    if event_names is None:
        candidates: List[Tuple[float, str]] = []
        for d in archive_root.iterdir():
            if not d.is_dir():
                continue
            cubes_dir = d / "cubes"
            if not cubes_dir.is_dir():
                continue
            npzs = list(cubes_dir.glob("cube_s*_g*_*.npz"))
            if not npzs:
                continue
            mtime = max(p.stat().st_mtime for p in npzs)
            candidates.append((mtime, d.name))
        candidates.sort(reverse=True)
        names = [name for _, name in candidates]
        if limit is not None:
            names = names[: int(limit)]
    else:
        names = list(event_names)

    done: List[str] = []
    for name in names:
        try:
            render_event_plots(PlotJob(event_name=name, archive_root=archive_root))
            done.append(name)
            _LOG.info("plotter: regenerated plots for %s", name)
        except Exception:  # noqa: BLE001
            _LOG.exception("plotter: regeneration failed for %s", name)
    return done


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-render C2 cube event plots from the archive.",
    )
    parser.add_argument(
        "--archive-root", default="/dataz/dsa110/candidates",
        help="event archive root (default: %(default)s)",
    )
    parser.add_argument(
        "--event", action="append", dest="events", default=None,
        help="event name(s) to re-render; repeatable. Default: scan for "
             "recent events that have cubes.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="when scanning, cap to the N most recent events with cubes.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="debug logging",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    done = regenerate_recent_events(
        Path(args.archive_root), event_names=args.events, limit=args.limit,
    )
    print(f"regenerated {len(done)} event(s): {', '.join(done) or '(none)'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
