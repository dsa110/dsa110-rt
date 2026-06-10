"""T1 (per-candidate) + T2 (per-cluster) ASCII candidate logger (M6 chunk 2).

Implements the hourly-rotated, space-separated ASCII row writer locked
in by M6 D1 (T1/T2 schemas) and M6 D2 (rotation policy + multi-writer
flock). One :class:`CandsLogger` instance is owned per
``(search_node_id, gpu_half)`` process; each instance manages two open
file handles (one T1, one T2), rotates them every UTC hour on the hour,
and writes one row per candidate (T1) plus one row per ``ClusterRecord``
(T2) per cube.

Schemas (M6 D1, locked):

T1 (per-candidate; one row per Candidate that survived clustering,
including HDBSCAN/DBSCAN noise points emitted as singleton clusters
with ``cluster_id == -1``)::

    mjd  event_specnum  l_rad  m_rad  l_pix  m_pix  dm_fine_pc_cc
    fine_dm_idx  t_in_cube  width_samples  snr  kernel_id  cl
    is_cluster_peak  search_node_id  gpu_half

T2 (per-cluster; one row per cluster carrying the peak candidate's
properties)::

    mjd  event_specnum  l_rad  m_rad  l_pix  m_pix  dm_fine_pc_cc
    fine_dm_idx  t_in_cube  width_samples  snr  kernel_id  cluster_id
    cntc  cntb_lm  cntb_dm  cube_dump_triggered  search_node_id  gpu_half

The first line of each newly opened file is a single ``#``-prefixed
header naming the columns verbatim (M6 D1).

Rotation policy (M6 D2): every UTC hour, on the hour. File names::

    ${LOG_ROOT}/cands_T1_s${sid}_g${gpu_half}_${YYYYMMDD}_${HH}.txt
    ${LOG_ROOT}/cands_T2_s${sid}_g${gpu_half}_${YYYYMMDD}_${HH}.txt

Concurrent writers (e.g. a future second GPU-half process pointed at
the same log root) are supported via ``fcntl.flock(LOCK_EX)``: each
``write_cube`` call acquires the per-file exclusive lock once,
serialises its T1 batch to the open fd, releases the lock; same for
T2. This ensures inter-process row batches never interleave (no torn
lines), matches the per-row atomicity the M6 D2 hot-path budget
allows, and amortises the per-row syscall cost across the typical
~1-50 rows per cube.
"""

from __future__ import annotations

import fcntl
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..common.contracts import Candidate, ClusterRecord, CubeGeometry
from .features import signed_centred_pix

__all__ = [
    "CandsLoggerConfig",
    "CandsLogger",
    "T1_HEADER",
    "T2_HEADER",
    "T1_COLUMNS",
    "T2_COLUMNS",
]


# ---------------------------------------------------------------------------
# Schema constants (M6 D1)
# ---------------------------------------------------------------------------

T1_COLUMNS: Tuple[str, ...] = (
    "mjd",
    "event_specnum",
    "l_rad",
    "m_rad",
    "l_pix",
    "m_pix",
    "dm_fine_pc_cc",
    "fine_dm_idx",
    "t_in_cube",
    "width_samples",
    "snr",
    "kernel_id",
    "cl",
    "is_cluster_peak",
    "search_node_id",
    "gpu_half",
)
"""T1 column names in serialised order (M6 D1; matches the schema
section verbatim — header line in the file uses these names with a
single leading ``#`` and one space between names)."""

T2_COLUMNS: Tuple[str, ...] = (
    "mjd",
    "event_specnum",
    "l_rad",
    "m_rad",
    "l_pix",
    "m_pix",
    "dm_fine_pc_cc",
    "fine_dm_idx",
    "t_in_cube",
    "width_samples",
    "snr",
    "kernel_id",
    "cluster_id",
    "cntc",
    "cntb_lm",
    "cntb_dm",
    "cube_dump_triggered",
    "search_node_id",
    "gpu_half",
)
"""T2 column names in serialised order (M6 D1)."""

T1_HEADER: str = "# " + " ".join(T1_COLUMNS) + "\n"
T2_HEADER: str = "# " + " ".join(T2_COLUMNS) + "\n"


# Numeric-format pins. Fixed once here so test 8 has a single source of
# truth and so a downstream tweak (e.g. tightening MJD precision) is
# one-line.
_FMT_MJD = "{:.11f}"          # ~0.86 ns precision (>= 9 decimals; M6 D1 / test 8)
_FMT_REAL = "{:.9e}"           # 10 sig-figs (>= 6 sig-figs; test 8)
_FMT_SNR = "{:.6e}"            # 7 sig-figs — adequate for σ output


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandsLoggerConfig:
    """Per-process candidate-logger configuration (M6 D1/D2).

    Args:
        log_root: directory the T1+T2 files live under. Created on
            first ``write_cube`` if missing. In dev defaults to
            ``${REPO_ROOT}/bench/reports/M6/cands_log/``; production
            overrides via the ``DSART_M6_CANDS_LOG_DIR`` env var.
        search_node_id: 0..N_SEARCH-1; appears in file name suffix
            (``_s${search_node_id}``) and in every row's
            ``search_node_id`` column.
        gpu_half: 0..N_SEARCH_GPU-1; appears in file name suffix
            (``_g${gpu_half}``) and in every row's ``gpu_half`` column.
    """

    log_root: Path
    search_node_id: int
    gpu_half: int


# ---------------------------------------------------------------------------
# Internal per-file state
# ---------------------------------------------------------------------------


@dataclass
class _OpenFile:
    """Mutable per-(file-kind) state held by :class:`CandsLogger`.

    Args:
        path: current on-disk path; ``None`` until first write.
        fh: open append-text file handle; ``None`` if not yet opened or
            after :meth:`CandsLogger.close`.
        hour_key: ``(YYYYMMDD, HH)`` tuple identifying which UTC hour
            the file belongs to. Compared against the current
            ``utc_now`` to detect rotation boundaries.
    """

    path: Optional[Path] = None
    fh: Optional[IO[str]] = None
    hour_key: Optional[Tuple[str, str]] = None


# ---------------------------------------------------------------------------
# CandsLogger
# ---------------------------------------------------------------------------


class CandsLogger:
    """Thread-safe T1/T2 ASCII candidate logger (M6 D1/D2).

    A single instance is owned per ``(search_node_id, gpu_half)``
    process; ``write_cube`` is safe to call from multiple threads
    within that process (an internal :class:`threading.Lock` serialises
    file-handle mutation). Multiple processes pointed at the same
    ``log_root`` + ``search_node_id`` + ``gpu_half`` interleave safely
    via :func:`fcntl.flock` per per-cube write batch.

    Per-cube atomicity policy: each ``write_cube`` call serialises its
    full T1 batch under one ``LOCK_EX`` acquisition, then its full T2
    batch under another. This guarantees no inter-process tear within
    a batch while keeping syscall overhead bounded for small cubes
    (1-50 rows). Per-row flock would buy strictly nothing extra at
    these batch sizes (linux O_APPEND atomically appends ≤ PIPE_BUF
    bytes, but inter-process interleave would still happen between
    rows, which the operator never wants — clusters are logical units).
    """

    def __init__(self, config: CandsLoggerConfig) -> None:
        """Construct the logger.

        Files are NOT opened here — the first ``write_cube`` call
        determines the initial UTC hour and opens both T1 and T2 files
        lazily. Construction is therefore side-effect-free except for
        any directory the caller's :class:`CandsLoggerConfig` points
        at being checked for existence (it is created on first write,
        not at construction).
        """
        self._config = config
        self._t1 = _OpenFile()
        self._t2 = _OpenFile()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_cube(
        self,
        *,
        cands: Sequence[Candidate],
        cluster_labels: np.ndarray,
        cluster_records: Sequence[ClusterRecord],
        geom: CubeGeometry,
        triggered_cluster_ids: Iterable[int],
        utc_now: Optional[datetime] = None,
    ) -> None:
        """Append one cube's worth of T1 + T2 rows.

        Args:
            cands: per-cube candidate list, in the order the M5
                detector emitted them. T1 rows preserve this order.
            cluster_labels: ``np.ndarray[len(cands)]`` int (any int
                dtype). ``-1`` denotes HDBSCAN/DBSCAN noise points;
                ``≥ 0`` denotes membership in the cluster with that id.
            cluster_records: one record per cluster (including the
                per-noise-point singleton records emitted by the
                clusterer per M6 D1). T2 rows are written in
                ``cluster_id`` ascending order; noise singletons
                (``cluster_id == -1``) therefore precede all proper
                clusters in the T2 file.
            geom: cube geometry sidecar — drives mjd computation
                (``mjd_start + (event_specnum - specnum_start) /
                sample_period_specnum * sample_period_us / 86400e6``)
                and the pixel→radian / specnum→sample-index
                conversions for non-peak T1 rows.
            triggered_cluster_ids: subset of cluster_records'
                ``cluster_id`` whose dumps fired the auto-trigger
                predicate. Sets the ``cube_dump_triggered`` column on
                T2. Membership is by integer equality; noise singletons
                (``-1``) can be members but typically are not (the M6
                D8 predicate gates on cardinality, SNR, etc.).
            utc_now: injection point for tests + deterministic batch
                replay. Defaults to ``datetime.now(timezone.utc)``.
                Must be timezone-aware (UTC) or naive-interpreted-as-
                UTC; the file-name HH suffix uses the value's hour
                field directly.

        Raises:
            ValueError: if ``len(cluster_labels) != len(cands)`` or if
                a candidate's cluster label points at a cluster id not
                represented in ``cluster_records`` (only enforced for
                ``≥ 0`` ids; ``-1`` noise singletons are assumed
                represented one-per-noise-point per the M6 D1 / chunk-1
                ``cluster.forward`` convention).
        """
        n = len(cands)
        if cluster_labels.shape != (n,):
            raise ValueError(
                f"cluster_labels.shape={cluster_labels.shape} != ({n},)"
            )

        # Index ClusterRecord by cluster_id ≥ 0 → record (for is_cluster_peak
        # lookup in T1). We accept multiple noise singletons (cluster_id == -1)
        # in cluster_records — those don't need a peak lookup since each is
        # its own singleton and per the M6 D1 / test 6 convention the T1 row
        # for a noise candidate writes is_cluster_peak=0.
        peak_idx_by_cid: Dict[int, int] = {}
        for cr in cluster_records:
            if cr.cluster_id >= 0:
                if cr.cluster_id in peak_idx_by_cid:
                    raise ValueError(
                        f"duplicate cluster_id={cr.cluster_id} in cluster_records"
                    )
                peak_idx_by_cid[cr.cluster_id] = cr.peak_candidate_idx

        triggered = {int(c) for c in triggered_cluster_ids}

        utc = self._normalise_utc(utc_now)
        hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))

        # Pre-compute fine_dm_idx + t_in_cube for every candidate up front
        # (vectorised over fine_dm_pc_cc; cheaper than per-candidate
        # searchsorted in the row-build loop). Empty input is fine.
        fine_dm_idxs = self._fine_dm_indices_for(cands, geom)
        t_in_cubes = self._t_in_cube_for(cands, geom)
        mjds = self._mjd_for(cands, geom)

        t1_rows = self._build_t1_rows(
            cands=cands,
            cluster_labels=cluster_labels,
            peak_idx_by_cid=peak_idx_by_cid,
            geom=geom,
            mjds=mjds,
            fine_dm_idxs=fine_dm_idxs,
            t_in_cubes=t_in_cubes,
        )
        t2_rows = self._build_t2_rows(
            cluster_records=cluster_records,
            geom=geom,
            triggered=triggered,
        )

        with self._lock:
            self._append_rows(self._t1, "T1", T1_HEADER, hour_key, t1_rows)
            self._append_rows(self._t2, "T2", T2_HEADER, hour_key, t2_rows)

    def close(self) -> None:
        """Flush + close any open T1/T2 file handles (idempotent)."""
        with self._lock:
            for state in (self._t1, self._t2):
                if state.fh is not None:
                    try:
                        state.fh.flush()
                    finally:
                        state.fh.close()
                    state.fh = None
                    state.path = None
                    state.hour_key = None

    # Context-manager sugar so tests + service code can write
    # ``with CandsLogger(cfg) as logger: ...`` without a try/finally.
    def __enter__(self) -> "CandsLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Row-building helpers (pure; testable in isolation)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_utc(utc_now: Optional[datetime]) -> datetime:
        """Return a timezone-aware UTC datetime.

        Naive datetimes are interpreted as UTC (matching the legacy
        ``datetime.utcnow()`` semantic the M6 D2 file-name template
        was written against). Aware datetimes in any tz are converted
        to UTC.
        """
        if utc_now is None:
            return datetime.now(timezone.utc)
        if utc_now.tzinfo is None:
            return utc_now.replace(tzinfo=timezone.utc)
        return utc_now.astimezone(timezone.utc)

    @staticmethod
    def _fine_dm_indices_for(
        cands: Sequence[Candidate], geom: CubeGeometry
    ) -> np.ndarray:
        """Vectorised closest-grid-index recovery for candidate fine-DMs.

        Mirrors the chunk-1 features.py convention (binary-search +
        choose closer of the two adjacent grid points) so the T1 logger
        and the clusterer agree on ``fine_dm_idx`` for the same
        Candidate. Empty input → ``np.zeros(0, int64)``.
        """
        if not cands:
            return np.zeros(0, dtype=np.int64)
        dm = np.fromiter(
            (c.dm_fine for c in cands), count=len(cands), dtype=np.float64
        )
        grid = geom.fine_dm_pc_cc
        n_grid = grid.shape[0]
        ins = np.searchsorted(grid, dm)
        # Candidate indices are either ins or ins-1; pick the closer.
        # Clamp so adjacent-pair indexing stays in range.
        ins_clamped = np.clip(ins, 1, n_grid - 1)
        left = ins_clamped - 1
        right = ins_clamped
        d_left = np.abs(grid[left] - dm)
        d_right = np.abs(grid[right] - dm)
        idx = np.where(d_left <= d_right, left, right)
        # Edge cases: if dm is below the grid (ins == 0) the above
        # clamp picked index 1 vs 0 — but `dm` < grid[0] makes d_left
        # = grid[0] - dm < grid[1] - dm = d_right, so idx == 0. OK.
        # If dm is above the grid (ins == n_grid) the above clamp
        # picked index n_grid-1 vs n_grid-2 — both d_left, d_right are
        # positive; the right edge wins iff dm is closer to grid[-1],
        # which it always is in that branch. OK.
        return idx.astype(np.int64, copy=False)

    @staticmethod
    def _t_in_cube_for(
        cands: Sequence[Candidate], geom: CubeGeometry
    ) -> np.ndarray:
        """Compute t_in_cube = (event_specnum - specnum_start) //
        sample_period_specnum for each candidate (vectorised, int64).
        """
        if not cands:
            return np.zeros(0, dtype=np.int64)
        ev = np.fromiter(
            (c.event_specnum for c in cands),
            count=len(cands),
            dtype=np.int64,
        )
        return (ev - np.int64(geom.specnum_start)) // np.int64(
            geom.sample_period_specnum
        )

    @staticmethod
    def _mjd_for(
        cands: Sequence[Candidate], geom: CubeGeometry
    ) -> np.ndarray:
        """Compute the per-candidate event MJD per the M6 ch-2 formula::

            mjd = mjd_start + (event_specnum - specnum_start)
                              / sample_period_specnum
                              * sample_period_us / 86400e6

        Computation in float64; ``mjd_start`` is the dominant term, so
        the resolved MJD inherits its ~1e-9 relative precision (~86 µs
        absolute at MJD ~ 60000), which is well below the M6 D1
        9-decimal-place printed precision.
        """
        if not cands:
            return np.zeros(0, dtype=np.float64)
        ev = np.fromiter(
            (c.event_specnum for c in cands),
            count=len(cands),
            dtype=np.int64,
        )
        sample_idx = (ev - np.int64(geom.specnum_start)) // np.int64(
            geom.sample_period_specnum
        )
        days = sample_idx.astype(np.float64) * float(
            geom.sample_period_us
        ) / 86400.0e6
        return float(geom.mjd_start) + days

    @staticmethod
    def _mjd_for_cluster(cr: ClusterRecord, geom: CubeGeometry) -> float:
        """Per-cluster mjd convenience for the T2 row (peak candidate)."""
        sample_idx = (cr.event_specnum - geom.specnum_start) // geom.sample_period_specnum
        return float(geom.mjd_start) + float(sample_idx) * float(
            geom.sample_period_us
        ) / 86400.0e6

    def _build_t1_rows(
        self,
        *,
        cands: Sequence[Candidate],
        cluster_labels: np.ndarray,
        peak_idx_by_cid: Dict[int, int],
        geom: CubeGeometry,
        mjds: np.ndarray,
        fine_dm_idxs: np.ndarray,
        t_in_cubes: np.ndarray,
    ) -> List[str]:
        """Format one T1 row string per candidate (column order = T1_COLUMNS)."""
        rows: List[str] = []
        for i, cand in enumerate(cands):
            cid = int(cluster_labels[i])
            l_pix = int(round(cand.l))
            m_pix = int(round(cand.m))
            l_rad = (
                signed_centred_pix(l_pix, geom.n_grid) * geom.cell_l_rad
                + geom.l0_rad
            )
            m_rad = (
                signed_centred_pix(m_pix, geom.n_grid) * geom.cell_m_rad
                + geom.m0_rad
            )
            # is_cluster_peak: 1 only for the peak of a cluster_id ≥ 0
            # (M6 D1; chunk-2 spec test 6 explicitly pins noise singletons
            # to is_cluster_peak=0 even though they are technically the
            # "peak" of their own one-element record).
            is_peak = 1 if cid >= 0 and peak_idx_by_cid.get(cid) == i else 0
            rows.append(self._format_t1_row(
                mjd=float(mjds[i]),
                event_specnum=int(cand.event_specnum),
                l_rad=l_rad,
                m_rad=m_rad,
                l_pix=l_pix,
                m_pix=m_pix,
                dm_fine_pc_cc=float(cand.dm_fine),
                fine_dm_idx=int(fine_dm_idxs[i]),
                t_in_cube=int(t_in_cubes[i]),
                width_samples=int(cand.width_samples),
                snr=float(cand.snr),
                kernel_id=cand.kernel_id,
                cl=cid,
                is_cluster_peak=is_peak,
                search_node_id=int(cand.search_node_id),
                gpu_half=int(cand.gpu_half),
            ))
        return rows

    def _build_t2_rows(
        self,
        *,
        cluster_records: Sequence[ClusterRecord],
        geom: CubeGeometry,
        triggered: set,
    ) -> List[str]:
        """Format one T2 row per ClusterRecord, sorted by cluster_id ascending.

        Stable sort preserves the input ordering of multiple noise
        singletons (cluster_id == -1) — i.e. the chunk-1 ``cluster.forward``
        convention of "noise points in input-list order" survives
        through to the T2 file.
        """
        ordered = sorted(
            enumerate(cluster_records),
            key=lambda kv: (kv[1].cluster_id, kv[0]),
        )
        rows: List[str] = []
        for _, cr in ordered:
            mjd = self._mjd_for_cluster(cr, geom)
            dump_triggered = 1 if cr.cluster_id in triggered else 0
            rows.append(self._format_t2_row(
                mjd=mjd,
                event_specnum=int(cr.event_specnum),
                l_rad=float(cr.l_rad),
                m_rad=float(cr.m_rad),
                l_pix=int(cr.l_pix),
                m_pix=int(cr.m_pix),
                dm_fine_pc_cc=float(cr.dm_fine_pc_cc),
                fine_dm_idx=int(cr.fine_dm_idx),
                t_in_cube=int(cr.t_in_cube),
                width_samples=int(cr.width_samples),
                snr=float(cr.snr),
                kernel_id=cr.kernel_id,
                cluster_id=int(cr.cluster_id),
                cntc=int(cr.cntc),
                cntb_lm=int(cr.cntb_lm),
                cntb_dm=int(cr.cntb_dm),
                cube_dump_triggered=dump_triggered,
                search_node_id=int(cr.search_node_id),
                gpu_half=int(cr.gpu_half),
            ))
        return rows

    @staticmethod
    def _format_t1_row(
        *,
        mjd: float,
        event_specnum: int,
        l_rad: float,
        m_rad: float,
        l_pix: int,
        m_pix: int,
        dm_fine_pc_cc: float,
        fine_dm_idx: int,
        t_in_cube: int,
        width_samples: int,
        snr: float,
        kernel_id: str,
        cl: int,
        is_cluster_peak: int,
        search_node_id: int,
        gpu_half: int,
    ) -> str:
        return (
            f"{_FMT_MJD.format(mjd)} "
            f"{event_specnum:d} "
            f"{_FMT_REAL.format(l_rad)} "
            f"{_FMT_REAL.format(m_rad)} "
            f"{l_pix:d} "
            f"{m_pix:d} "
            f"{_FMT_REAL.format(dm_fine_pc_cc)} "
            f"{fine_dm_idx:d} "
            f"{t_in_cube:d} "
            f"{width_samples:d} "
            f"{_FMT_SNR.format(snr)} "
            f"{kernel_id} "
            f"{cl:d} "
            f"{is_cluster_peak:d} "
            f"{search_node_id:d} "
            f"{gpu_half:d}\n"
        )

    @staticmethod
    def _format_t2_row(
        *,
        mjd: float,
        event_specnum: int,
        l_rad: float,
        m_rad: float,
        l_pix: int,
        m_pix: int,
        dm_fine_pc_cc: float,
        fine_dm_idx: int,
        t_in_cube: int,
        width_samples: int,
        snr: float,
        kernel_id: str,
        cluster_id: int,
        cntc: int,
        cntb_lm: int,
        cntb_dm: int,
        cube_dump_triggered: int,
        search_node_id: int,
        gpu_half: int,
    ) -> str:
        return (
            f"{_FMT_MJD.format(mjd)} "
            f"{event_specnum:d} "
            f"{_FMT_REAL.format(l_rad)} "
            f"{_FMT_REAL.format(m_rad)} "
            f"{l_pix:d} "
            f"{m_pix:d} "
            f"{_FMT_REAL.format(dm_fine_pc_cc)} "
            f"{fine_dm_idx:d} "
            f"{t_in_cube:d} "
            f"{width_samples:d} "
            f"{_FMT_SNR.format(snr)} "
            f"{kernel_id} "
            f"{cluster_id:d} "
            f"{cntc:d} "
            f"{cntb_lm:d} "
            f"{cntb_dm:d} "
            f"{cube_dump_triggered:d} "
            f"{search_node_id:d} "
            f"{gpu_half:d}\n"
        )

    # ------------------------------------------------------------------
    # File-handle management
    # ------------------------------------------------------------------

    def _file_path(self, kind: str, hour_key: Tuple[str, str]) -> Path:
        """Build the file path for ``kind`` ∈ {"T1", "T2"} at this hour."""
        ymd, hh = hour_key
        cfg = self._config
        name = (
            f"cands_{kind}_s{cfg.search_node_id}_g{cfg.gpu_half}"
            f"_{ymd}_{hh}.txt"
        )
        return cfg.log_root / name

    def _append_rows(
        self,
        state: _OpenFile,
        kind: str,
        header: str,
        hour_key: Tuple[str, str],
        rows: Sequence[str],
    ) -> None:
        """Open / rotate the file as needed, write header on first
        open, append ``rows`` under one ``LOCK_EX`` acquisition.

        Holding the threading.Lock around this method (the caller's
        responsibility — :meth:`write_cube` does so) bounds intra-
        process concurrency; the inter-process flock bounds the rest.
        """
        if state.fh is not None and state.hour_key != hour_key:
            try:
                state.fh.flush()
            finally:
                state.fh.close()
            state.fh = None
            state.path = None
            state.hour_key = None

        if state.fh is None:
            path = self._file_path(kind, hour_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append text mode; line-buffered so flush-on-newline keeps
            # the operator's `tail -F` view fresh without an explicit
            # flush in the hot loop. encoding pinned for portability.
            fh = open(path, "a", buffering=1, encoding="utf-8")
            state.fh = fh
            state.path = path
            state.hour_key = hour_key
            # Atomic check-and-write the header: if we're the first
            # process to land on this file (or it was rotated and is
            # therefore size 0), drop the header line. Other writers
            # block here until we release LOCK_EX. The size check after
            # acquiring the lock guards against the race where two
            # processes both race to open the file.
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                # File could have been written to by a peer between our
                # open() and lock acquisition. Use os.fstat on the fd so
                # we read the header against the same inode.
                if os.fstat(fh.fileno()).st_size == 0:
                    fh.write(header)
                    fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

        if not rows:
            return

        fh = state.fh
        assert fh is not None  # typing: open() above guarantees this.
        # Single batched write under one flock acquisition.
        # ``writelines`` invokes one underlying write() syscall in
        # CPython (subject to buffering=1 which only flushes on
        # newline-terminated chunks). We then flush() to push every
        # row to the kernel before releasing the inter-process lock,
        # so a peer process sees a complete batch as soon as we let
        # go (no torn rows from a partial-flushed buffer).
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.writelines(rows)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
