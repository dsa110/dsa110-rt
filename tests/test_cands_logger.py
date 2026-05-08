"""Tests for the M6 chunk-2 T1/T2 ASCII candidate logger.

Covers the nine cases the chunk-2 spec requires:
  1. T1 file gets the correct header on first write.
  2. T2 file gets the correct header on first write.
  3. T1 row ordering preserves Candidate input order; T2 rows are
     ordered by ``cluster_id`` ascending (noise singletons first).
  4. Hourly rotation: a write at HH:59:30 then HH+1:00:01 produces a
     new file with a new header.
  5. Concurrent (multiprocessing) writes against the same log_root +
     sid + gpu_half preserve the union of rows (no torn lines).
  6. ``is_cluster_peak == 1`` for exactly one row per ``cluster_id ≥ 0``
     and 0 for all noise singletons.
  7. ``cube_dump_triggered == 1`` only on T2 rows whose ``cluster_id``
     is in the caller-supplied ``triggered_cluster_ids`` set.
  8. Numeric format: mjd ≥ 9 decimal places; l_rad/m_rad/dm_fine_pc_cc
     printed in scientific notation with ≥ 6 sig-figs; ints have no
     leading zeros / trailing decimals.
  9. Round-trip: 50 written rows reload via ``np.loadtxt(comments='#')``
     with key columns matching.

Tests force ``DSART_TEST=1`` at import time to enable the contracts'
``__post_init__`` invariant checks (matches ``test_contracts.py``).
"""

from __future__ import annotations

import os

# CRITICAL: set DSART_TEST=1 before importing dsart so contracts'
# __post_init__ is active. Mirrors test_contracts.py.
os.environ["DSART_TEST"] = "1"

import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dsart.cluster.cands_logger import (  # noqa: E402
    CandsLogger,
    CandsLoggerConfig,
    T1_COLUMNS,
    T1_HEADER,
    T2_COLUMNS,
    T2_HEADER,
)
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    ClusterRecord,
    CubeGeometry,
)


# ---------------------------------------------------------------------------
# Common fixtures + factories
# ---------------------------------------------------------------------------


SID = 2
GPU_HALF = 1


def _make_geometry(**overrides) -> CubeGeometry:
    base = dict(
        cube_id=0,
        specnum_start=1024,
        sample_period_specnum=16,
        t_det=256,
        n_grid=256,
        n_fdm_in_cube=32,
        sample_period_us=131.072,
        cell_l_rad=1.5e-4,
        cell_m_rad=1.5e-4,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=np.linspace(50.0, 800.0, 32, dtype=np.float64),
        mjd_start=60942.123456789,
    )
    base.update(overrides)
    return CubeGeometry(**base)


def _make_candidate(
    *,
    l_pix: int = 132,
    m_pix: int = 230,
    dm_fine: float | None = None,
    fine_dm_grid: np.ndarray | None = None,
    fine_dm_idx: int = 15,
    event_specnum: int = 2048,
    width_samples: int = 4,
    snr: float = 9.7,
    kernel_id: str = "psf:d3:b16",
    sid: int = SID,
    gpu_half: int = GPU_HALF,
) -> Candidate:
    """Build a Candidate with l/m as float-cast pixel indices, dm_fine
    snapped onto the supplied fine-DM grid (so fine_dm_idx round-trips
    via np.searchsorted/argmin).
    """
    if dm_fine is None:
        if fine_dm_grid is None:
            fine_dm_grid = np.linspace(50.0, 800.0, 32, dtype=np.float64)
        dm_fine = float(fine_dm_grid[fine_dm_idx])
    return Candidate(
        l=float(l_pix),
        m=float(m_pix),
        dm_fine=dm_fine,
        dm_idx=10,
        event_specnum=event_specnum,
        width_samples=width_samples,
        kernel_id=kernel_id,
        snr=snr,
        detector_version="v1.deterministic.20260501",
        flags=int(CandidateFlags.NONE),
        search_node_id=sid,
        gpu_half=gpu_half,
    )


def _make_record(
    *,
    cluster_id: int,
    cands: List[Candidate],
    member_idxs: List[int],
    geom: CubeGeometry,
) -> ClusterRecord:
    """Build a ClusterRecord by picking the highest-snr member as peak
    (matches the chunk-1 ``cluster.forward`` convention)."""
    member_snrs = [cands[i].snr for i in member_idxs]
    peak_within = int(np.argmax(member_snrs))
    peak_idx = member_idxs[peak_within]
    peak = cands[peak_idx]
    l_pix = int(round(peak.l))
    m_pix = int(round(peak.m))
    fine_dm_idx = int(np.argmin(np.abs(geom.fine_dm_pc_cc - peak.dm_fine)))
    t_in_cube = int(
        (peak.event_specnum - geom.specnum_start) // geom.sample_period_specnum
    )
    lm_set = {
        (int(round(cands[i].l)), int(round(cands[i].m))) for i in member_idxs
    }
    dm_set = {
        int(np.argmin(np.abs(geom.fine_dm_pc_cc - cands[i].dm_fine)))
        for i in member_idxs
    }
    return ClusterRecord(
        cluster_id=cluster_id,
        cube_id=geom.cube_id,
        cntc=len(member_idxs),
        cntb_lm=len(lm_set),
        cntb_dm=len(dm_set),
        peak_candidate_idx=peak_idx,
        l_rad=float(l_pix) * geom.cell_l_rad + geom.l0_rad,
        m_rad=float(m_pix) * geom.cell_m_rad + geom.m0_rad,
        l_pix=l_pix,
        m_pix=m_pix,
        dm_fine_pc_cc=float(peak.dm_fine),
        fine_dm_idx=fine_dm_idx,
        t_in_cube=t_in_cube,
        t_seconds=t_in_cube * geom.sample_period_us / 1e6,
        width_samples=peak.width_samples,
        snr=peak.snr,
        kernel_id=peak.kernel_id,
        event_specnum=peak.event_specnum,
        search_node_id=peak.search_node_id,
        gpu_half=peak.gpu_half,
    )


def _t1_path(root: Path, hour_key: tuple[str, str]) -> Path:
    ymd, hh = hour_key
    return root / f"cands_T1_s{SID}_g{GPU_HALF}_{ymd}_{hh}.txt"


def _t2_path(root: Path, hour_key: tuple[str, str]) -> Path:
    ymd, hh = hour_key
    return root / f"cands_T2_s{SID}_g{GPU_HALF}_{ymd}_{hh}.txt"


def _utc(year=2026, month=5, day=7, hour=14, minute=30, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


@pytest.fixture
def log_root(tmp_path: Path) -> Path:
    root = tmp_path / "cands_log"
    root.mkdir()
    return root


@pytest.fixture
def logger(log_root: Path) -> CandsLogger:
    cfg = CandsLoggerConfig(
        log_root=log_root, search_node_id=SID, gpu_half=GPU_HALF
    )
    lg = CandsLogger(cfg)
    yield lg
    lg.close()


# ---------------------------------------------------------------------------
# Test 1+2: header on first write
# ---------------------------------------------------------------------------


def _three_cluster_workload(geom: CubeGeometry):
    """Construct a deterministic 5-candidate workload:
      * cands[0..1] form cluster 0 (peak at idx 1, snr 12.0).
      * cands[2..3] form cluster 1 (peak at idx 3, snr 15.0).
      * cands[4]    is a noise singleton (cluster_id == -1).
    """
    cands = [
        _make_candidate(l_pix=10, m_pix=20, fine_dm_idx=3, snr=8.0,
                        event_specnum=geom.specnum_start + 16),
        _make_candidate(l_pix=11, m_pix=21, fine_dm_idx=4, snr=12.0,
                        event_specnum=geom.specnum_start + 32),
        _make_candidate(l_pix=200, m_pix=100, fine_dm_idx=20, snr=11.0,
                        event_specnum=geom.specnum_start + 48),
        _make_candidate(l_pix=201, m_pix=101, fine_dm_idx=21, snr=15.0,
                        event_specnum=geom.specnum_start + 64),
        _make_candidate(l_pix=50, m_pix=50, fine_dm_idx=10, snr=9.0,
                        event_specnum=geom.specnum_start + 80),
    ]
    labels = np.array([0, 0, 1, 1, -1], dtype=np.int64)
    records = [
        _make_record(cluster_id=0, cands=cands, member_idxs=[0, 1], geom=geom),
        _make_record(cluster_id=1, cands=cands, member_idxs=[2, 3], geom=geom),
        _make_record(cluster_id=-1, cands=cands, member_idxs=[4], geom=geom),
    ]
    return cands, labels, records


def test_t1_header_on_first_write(logger: CandsLogger, log_root: Path) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    utc = _utc()
    logger.write_cube(
        cands=cands,
        cluster_labels=labels,
        cluster_records=records,
        geom=geom,
        triggered_cluster_ids=set(),
        utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    p = _t1_path(log_root, hour_key)
    assert p.exists(), f"T1 file not created: {p}"
    lines = p.read_text().splitlines()
    assert lines, "T1 file is empty"
    assert lines[0] == T1_HEADER.rstrip("\n")
    # Header contents match the schema-locked column names verbatim.
    assert lines[0] == "# " + " ".join(T1_COLUMNS)
    # One header + 5 data rows.
    assert len(lines) == 1 + len(cands)


def test_t2_header_on_first_write(logger: CandsLogger, log_root: Path) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    utc = _utc()
    logger.write_cube(
        cands=cands,
        cluster_labels=labels,
        cluster_records=records,
        geom=geom,
        triggered_cluster_ids=set(),
        utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    p = _t2_path(log_root, hour_key)
    assert p.exists(), f"T2 file not created: {p}"
    lines = p.read_text().splitlines()
    assert lines, "T2 file is empty"
    assert lines[0] == T2_HEADER.rstrip("\n")
    assert lines[0] == "# " + " ".join(T2_COLUMNS)
    # One header + 3 cluster rows.
    assert len(lines) == 1 + len(records)


# ---------------------------------------------------------------------------
# Test 3: row ordering
# ---------------------------------------------------------------------------


def _data_rows(text: str) -> List[List[str]]:
    """Split a file's text into a list-of-list-of-tokens for non-comment lines."""
    out: List[List[str]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        out.append(line.split())
    return out


def test_t1_rows_in_input_order(logger: CandsLogger, log_root: Path) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    utc = _utc()
    logger.write_cube(
        cands=cands,
        cluster_labels=labels,
        cluster_records=records,
        geom=geom,
        triggered_cluster_ids=set(),
        utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    rows = _data_rows(_t1_path(log_root, hour_key).read_text())
    assert len(rows) == len(cands)
    # event_specnum column = T1_COLUMNS.index("event_specnum") = 1
    es_col = T1_COLUMNS.index("event_specnum")
    for written, expected in zip(rows, cands):
        assert int(written[es_col]) == expected.event_specnum, (
            f"T1 row order changed: written={written}, expected={expected}"
        )


def test_t2_rows_sorted_by_cluster_id_ascending(
    logger: CandsLogger, log_root: Path
) -> None:
    """T2 rows are stable-sorted by cluster_id ascending. Noise
    singletons (cluster_id == -1) therefore precede all proper
    clusters (cluster_id ≥ 0).
    """
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    # Reverse the input ordering so the sort is non-trivial.
    reversed_records = list(reversed(records))
    utc = _utc()
    logger.write_cube(
        cands=cands,
        cluster_labels=labels,
        cluster_records=reversed_records,
        geom=geom,
        triggered_cluster_ids=set(),
        utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    rows = _data_rows(_t2_path(log_root, hour_key).read_text())
    cid_col = T2_COLUMNS.index("cluster_id")
    cluster_ids = [int(r[cid_col]) for r in rows]
    assert cluster_ids == sorted(cluster_ids), (
        f"T2 cluster_ids not ascending: {cluster_ids}"
    )
    # First row must be a noise singleton (cluster_id == -1).
    assert cluster_ids[0] == -1


# ---------------------------------------------------------------------------
# Test 4: hourly rotation
# ---------------------------------------------------------------------------


def test_hourly_rotation(logger: CandsLogger, log_root: Path) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    utc_a = _utc(hour=14, minute=59, second=30)
    utc_b = _utc(hour=15, minute=0, second=1)

    logger.write_cube(
        cands=cands, cluster_labels=labels, cluster_records=records,
        geom=geom, triggered_cluster_ids=set(), utc_now=utc_a,
    )
    logger.write_cube(
        cands=cands, cluster_labels=labels, cluster_records=records,
        geom=geom, triggered_cluster_ids=set(), utc_now=utc_b,
    )

    hour_a = (utc_a.strftime("%Y%m%d"), utc_a.strftime("%H"))
    hour_b = (utc_b.strftime("%Y%m%d"), utc_b.strftime("%H"))
    assert hour_a != hour_b

    p_t1_a = _t1_path(log_root, hour_a)
    p_t1_b = _t1_path(log_root, hour_b)
    p_t2_a = _t2_path(log_root, hour_a)
    p_t2_b = _t2_path(log_root, hour_b)
    for p in (p_t1_a, p_t1_b, p_t2_a, p_t2_b):
        assert p.exists(), f"missing rotated file: {p}"

    # Each file has its own header (re-emitted on rotation).
    for p in (p_t1_a, p_t1_b):
        assert p.read_text().splitlines()[0] == T1_HEADER.rstrip("\n")
    for p in (p_t2_a, p_t2_b):
        assert p.read_text().splitlines()[0] == T2_HEADER.rstrip("\n")

    # Each file holds exactly one batch's worth of data rows.
    for p in (p_t1_a, p_t1_b):
        assert len(_data_rows(p.read_text())) == len(cands)
    for p in (p_t2_a, p_t2_b):
        assert len(_data_rows(p.read_text())) == len(records)


# ---------------------------------------------------------------------------
# Test 5: concurrent writes via multiprocessing
# ---------------------------------------------------------------------------


# Module-level worker function for spawn-style multiprocessing.
def _mp_worker(
    log_root_str: str, sid: int, gpu_half: int, n_rows: int, worker_tag: int
) -> None:
    # Each subprocess re-evaluates DSART_TEST at import; ensure it's set
    # before importing dsart inside the subprocess.
    os.environ["DSART_TEST"] = "1"
    from dsart.cluster.cands_logger import CandsLogger as _CL
    from dsart.cluster.cands_logger import CandsLoggerConfig as _CLC
    from dsart.common.contracts import CandidateFlags as _CF
    from dsart.common.contracts import Candidate as _C
    from dsart.common.contracts import ClusterRecord as _CR
    from dsart.common.contracts import CubeGeometry as _CG
    import numpy as _np

    cfg = _CLC(
        log_root=Path(log_root_str), search_node_id=sid, gpu_half=gpu_half
    )
    lg = _CL(cfg)
    geom = _CG(
        cube_id=0,
        specnum_start=1024,
        sample_period_specnum=16,
        t_det=256,
        n_grid=256,
        n_fdm_in_cube=32,
        sample_period_us=131.072,
        cell_l_rad=1.5e-4,
        cell_m_rad=1.5e-4,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=_np.linspace(50.0, 800.0, 32, dtype=_np.float64),
        mjd_start=60942.123456789,
    )
    fixed_utc = datetime(2026, 5, 7, 12, 30, 0, tzinfo=timezone.utc)
    for i in range(n_rows):
        # Encode (worker_tag, i) into event_specnum so we can verify
        # the union after the workers finish.
        ev = 1024 + 16 * (worker_tag * 100_000 + i + 1)
        cand = _C(
            l=10.0,
            m=20.0,
            dm_fine=float(geom.fine_dm_pc_cc[3]),
            dm_idx=10,
            event_specnum=ev,
            width_samples=4,
            kernel_id="psf:d3:b16",
            snr=9.7,
            detector_version="v1.deterministic.20260501",
            flags=int(_CF.NONE),
            search_node_id=sid,
            gpu_half=gpu_half,
        )
        rec = _CR(
            cluster_id=-1,
            cube_id=0,
            cntc=1,
            cntb_lm=1,
            cntb_dm=1,
            peak_candidate_idx=0,
            l_rad=10.0 * geom.cell_l_rad,
            m_rad=20.0 * geom.cell_m_rad,
            l_pix=10,
            m_pix=20,
            dm_fine_pc_cc=float(geom.fine_dm_pc_cc[3]),
            fine_dm_idx=3,
            t_in_cube=int((ev - geom.specnum_start) // geom.sample_period_specnum),
            t_seconds=0.0,
            width_samples=4,
            snr=9.7,
            kernel_id="psf:d3:b16",
            event_specnum=ev,
            search_node_id=sid,
            gpu_half=gpu_half,
        )
        lg.write_cube(
            cands=[cand],
            cluster_labels=_np.array([-1], dtype=_np.int64),
            cluster_records=[rec],
            geom=geom,
            triggered_cluster_ids=set(),
            utc_now=fixed_utc,
        )
    lg.close()


def test_concurrent_writes_no_torn_rows(log_root: Path) -> None:
    n_per_worker = 100
    fixed_utc = _utc(hour=12, minute=30, second=0)
    hour_key = (fixed_utc.strftime("%Y%m%d"), fixed_utc.strftime("%H"))

    ctx = mp.get_context("spawn")
    p1 = ctx.Process(
        target=_mp_worker,
        args=(str(log_root), SID, GPU_HALF, n_per_worker, 0),
    )
    p2 = ctx.Process(
        target=_mp_worker,
        args=(str(log_root), SID, GPU_HALF, n_per_worker, 1),
    )
    p1.start()
    p2.start()
    p1.join(timeout=60)
    p2.join(timeout=60)
    assert p1.exitcode == 0, f"worker 0 failed: exit={p1.exitcode}"
    assert p2.exitcode == 0, f"worker 1 failed: exit={p2.exitcode}"

    p_t1 = _t1_path(log_root, hour_key)
    p_t2 = _t2_path(log_root, hour_key)
    assert p_t1.exists() and p_t2.exists()

    t1_rows = _data_rows(p_t1.read_text())
    t2_rows = _data_rows(p_t2.read_text())
    expected_total = 2 * n_per_worker

    assert len(t1_rows) == expected_total, (
        f"T1 row count {len(t1_rows)} != {expected_total}"
    )
    assert len(t2_rows) == expected_total, (
        f"T2 row count {len(t2_rows)} != {expected_total}"
    )

    # Every row matches the schema column count (no torn lines).
    assert all(len(r) == len(T1_COLUMNS) for r in t1_rows), (
        "torn T1 row detected: column count mismatch"
    )
    assert all(len(r) == len(T2_COLUMNS) for r in t2_rows), (
        "torn T2 row detected: column count mismatch"
    )

    # Header should appear exactly once at the top of each file
    # (the second writer must skip it because the file is non-empty
    # by the time it lands).
    t1_text = p_t1.read_text()
    t2_text = p_t2.read_text()
    assert t1_text.count("# mjd ") == 1, (
        f"T1 header appeared {t1_text.count('# mjd ')} times"
    )
    assert t2_text.count("# mjd ") == 1, (
        f"T2 header appeared {t2_text.count('# mjd ')} times"
    )

    # Union of event_specnum across all rows = the full set of (worker, i)
    # combos each side wrote.
    es_col = T1_COLUMNS.index("event_specnum")
    written_es = sorted(int(r[es_col]) for r in t1_rows)
    expected_es: List[int] = []
    for tag in (0, 1):
        for i in range(n_per_worker):
            expected_es.append(1024 + 16 * (tag * 100_000 + i + 1))
    expected_es.sort()
    assert written_es == expected_es, (
        "T1 event_specnum union mismatch — some rows lost or duplicated"
    )


# ---------------------------------------------------------------------------
# Test 6: is_cluster_peak — exactly one peak per cluster_id ≥ 0; 0 for noise
# ---------------------------------------------------------------------------


def test_is_cluster_peak_invariants(
    logger: CandsLogger, log_root: Path
) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    utc = _utc()
    logger.write_cube(
        cands=cands, cluster_labels=labels, cluster_records=records,
        geom=geom, triggered_cluster_ids=set(), utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    rows = _data_rows(_t1_path(log_root, hour_key).read_text())
    cl_col = T1_COLUMNS.index("cl")
    pk_col = T1_COLUMNS.index("is_cluster_peak")
    es_col = T1_COLUMNS.index("event_specnum")

    peaks_per_cid: dict[int, int] = {}
    for r in rows:
        cl = int(r[cl_col])
        pk = int(r[pk_col])
        peaks_per_cid.setdefault(cl, 0)
        peaks_per_cid[cl] += pk
        if cl == -1:
            assert pk == 0, (
                f"noise singleton (event_specnum={r[es_col]}) has is_cluster_peak={pk}"
            )

    for cid, n_peaks in peaks_per_cid.items():
        if cid >= 0:
            assert n_peaks == 1, (
                f"cluster_id={cid} has {n_peaks} peak rows, expected exactly 1"
            )

    # Identify which row is the peak for each cluster and double-check it
    # is the highest-snr member (matches the ClusterRecord peak_candidate_idx
    # convention).
    snr_col = T1_COLUMNS.index("snr")
    by_cid: dict[int, List[List[str]]] = {}
    for r in rows:
        by_cid.setdefault(int(r[cl_col]), []).append(r)
    for cid, members in by_cid.items():
        if cid < 0:
            continue
        peak_row = next(r for r in members if int(r[pk_col]) == 1)
        assert float(peak_row[snr_col]) == max(
            float(r[snr_col]) for r in members
        )


# ---------------------------------------------------------------------------
# Test 7: cube_dump_triggered semantics
# ---------------------------------------------------------------------------


def test_cube_dump_triggered_only_on_named_clusters(
    logger: CandsLogger, log_root: Path
) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    triggered = {1}  # Only cluster 1 fires the dump.
    utc = _utc()
    logger.write_cube(
        cands=cands, cluster_labels=labels, cluster_records=records,
        geom=geom, triggered_cluster_ids=triggered, utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    rows = _data_rows(_t2_path(log_root, hour_key).read_text())
    cid_col = T2_COLUMNS.index("cluster_id")
    dt_col = T2_COLUMNS.index("cube_dump_triggered")
    for r in rows:
        cid = int(r[cid_col])
        dump_flag = int(r[dt_col])
        if cid in triggered:
            assert dump_flag == 1, f"cluster_id={cid} expected dump=1, got {dump_flag}"
        else:
            assert dump_flag == 0, f"cluster_id={cid} expected dump=0, got {dump_flag}"


# ---------------------------------------------------------------------------
# Test 8: numeric format precision
# ---------------------------------------------------------------------------


_INT_PATTERN = re.compile(r"^-?\d+$")
_REAL_PATTERN = re.compile(r"^-?\d+\.\d+([eE][+-]?\d+)?$")
_MJD_PATTERN = re.compile(r"^-?\d+\.(\d+)$")


def _count_sig_figs(token: str) -> int:
    """Count significant figures in a numeric string formatted as a
    plain decimal or scientific notation. Strips sign + exponent."""
    if "e" in token or "E" in token:
        mantissa = token.split("e")[0].split("E")[0]
    else:
        mantissa = token
    mantissa = mantissa.lstrip("-")
    digits = mantissa.replace(".", "")
    digits = digits.lstrip("0")
    return len(digits)


def test_numeric_format_precision(logger: CandsLogger, log_root: Path) -> None:
    geom = _make_geometry()
    cands, labels, records = _three_cluster_workload(geom)
    utc = _utc()
    logger.write_cube(
        cands=cands, cluster_labels=labels, cluster_records=records,
        geom=geom, triggered_cluster_ids={0, 1}, utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))
    t1_rows = _data_rows(_t1_path(log_root, hour_key).read_text())
    t2_rows = _data_rows(_t2_path(log_root, hour_key).read_text())

    def _check_row(row: List[str], cols: tuple) -> None:
        for token, name in zip(row, cols):
            if name == "mjd":
                m = _MJD_PATTERN.match(token)
                assert m, f"mjd not in plain-decimal form: {token!r}"
                # ≥ 9 decimal places per M6 D1 (test 8).
                assert len(m.group(1)) >= 9, (
                    f"mjd has {len(m.group(1))} decimals (need ≥ 9): {token!r}"
                )
            elif name in ("l_rad", "m_rad", "dm_fine_pc_cc"):
                assert _REAL_PATTERN.match(token), (
                    f"{name} not in real-number form: {token!r}"
                )
                # ≥ 6 sig-figs per M6 D1.
                assert _count_sig_figs(token) >= 6, (
                    f"{name}={token!r} has < 6 sig-figs"
                )
            elif name == "snr":
                assert _REAL_PATTERN.match(token), (
                    f"snr not in real-number form: {token!r}"
                )
            elif name == "kernel_id":
                # No internal whitespace; matches contracts'
                # "k_img:k_dm:k_time" pattern.
                assert ":" in token and " " not in token
            else:
                # All remaining columns are plain integers — no leading
                # zeros (except for the literal "0"), no trailing decimal.
                assert _INT_PATTERN.match(token), (
                    f"{name}={token!r} not a plain integer"
                )
                if token not in ("0", "-0") and not token.startswith("-"):
                    assert not token.startswith("0"), (
                        f"{name}={token!r} has a leading zero"
                    )
                assert not token.endswith("."), (
                    f"{name}={token!r} ends with '.'"
                )

    for r in t1_rows:
        _check_row(r, T1_COLUMNS)
    for r in t2_rows:
        _check_row(r, T2_COLUMNS)


# ---------------------------------------------------------------------------
# Test 9: round-trip via np.loadtxt
# ---------------------------------------------------------------------------


def test_roundtrip_via_loadtxt(logger: CandsLogger, log_root: Path) -> None:
    geom = _make_geometry()
    # Build 50 noise singletons spread across event_specnums.
    n = 50
    cands = []
    records = []
    for i in range(n):
        ev = geom.specnum_start + 16 * (i + 1)
        c = _make_candidate(
            l_pix=10 + i,
            m_pix=20 + (i % 7),
            fine_dm_idx=(i % geom.n_fdm_in_cube),
            event_specnum=ev,
            snr=9.0 + 0.1 * i,
        )
        cands.append(c)
        records.append(
            _make_record(
                cluster_id=-1, cands=cands, member_idxs=[i], geom=geom
            )
        )
    labels = np.full(n, -1, dtype=np.int64)
    utc = _utc()
    logger.write_cube(
        cands=cands, cluster_labels=labels, cluster_records=records,
        geom=geom, triggered_cluster_ids=set(), utc_now=utc,
    )
    hour_key = (utc.strftime("%Y%m%d"), utc.strftime("%H"))

    # Pick a numeric-only subset of T1 columns to load via np.loadtxt
    # (kernel_id is a string and would break the default float parser).
    cols_we_check = ("mjd", "event_specnum", "l_rad", "m_rad", "l_pix",
                     "m_pix", "dm_fine_pc_cc", "fine_dm_idx", "t_in_cube",
                     "width_samples", "snr", "cl", "is_cluster_peak",
                     "search_node_id", "gpu_half")
    use_idxs = tuple(T1_COLUMNS.index(c) for c in cols_we_check)
    arr = np.loadtxt(
        _t1_path(log_root, hour_key),
        comments="#",
        usecols=use_idxs,
    )
    assert arr.shape == (n, len(cols_we_check))

    # Reconstruct expected event_specnum + l_pix + cl + search_node_id
    # columns and compare.
    es_idx = cols_we_check.index("event_specnum")
    lpix_idx = cols_we_check.index("l_pix")
    cl_idx = cols_we_check.index("cl")
    sid_idx = cols_we_check.index("search_node_id")
    np.testing.assert_array_equal(
        arr[:, es_idx].astype(np.int64),
        np.array([c.event_specnum for c in cands], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arr[:, lpix_idx].astype(np.int64),
        np.array([int(round(c.l)) for c in cands], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arr[:, cl_idx].astype(np.int64), np.full(n, -1, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        arr[:, sid_idx].astype(np.int64), np.full(n, SID, dtype=np.int64)
    )

    # Check the SNR column round-trips within the format's precision
    # (1e-6 relative is comfortably above the .6e formatter).
    snr_idx = cols_we_check.index("snr")
    expected_snr = np.array([c.snr for c in cands], dtype=np.float64)
    np.testing.assert_allclose(arr[:, snr_idx], expected_snr, rtol=1e-6)

    # And mjd: should match _mjd_for() to within the .11f format
    # precision (~1e-11 absolute on top of MJD ~ 60000).
    mjd_idx = cols_we_check.index("mjd")
    sample_idx = (
        np.array([c.event_specnum for c in cands], dtype=np.int64)
        - geom.specnum_start
    ) // geom.sample_period_specnum
    expected_mjd = (
        geom.mjd_start
        + sample_idx.astype(np.float64) * geom.sample_period_us / 86400e6
    )
    np.testing.assert_allclose(arr[:, mjd_idx], expected_mjd, atol=1e-9)
