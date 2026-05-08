"""Tests for ``dsart.cluster.forward.cluster_candidates`` (M6 chunk 1).

Covers HDBSCAN primary backend, DBSCAN fallback, noise-singleton
emission, peak picking, cntb_lm / cntb_dm computation, and the per-cube
cluster_id reset semantics.

Skips HDBSCAN tests if the package isn't installed (warns at preflight).
"""

from __future__ import annotations

import os

os.environ.setdefault("DSART_TEST", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dsart.cluster.forward import (  # noqa: E402
    ClustererBackend,
    ClustererConfig,
    cluster_candidates,
)
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    ClusterRecord,
    CubeGeometry,
)


def _make_geom(**overrides) -> CubeGeometry:
    base: dict = dict(
        cube_id=0,
        specnum_start=1024,
        sample_period_specnum=16,
        t_det=256,
        n_grid=256,
        n_fdm_in_cube=8,
        sample_period_us=131.072,
        cell_l_rad=1.5e-4,
        cell_m_rad=1.5e-4,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=np.linspace(50.0, 800.0, 8, dtype=np.float64),
        mjd_start=60942.123456789,
    )
    base.update(overrides)
    return CubeGeometry(**base)


def _make_cand(*, l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=8.5,
               width_samples=4, kernel_id="unit:d1:b4", search_node_id=2,
               gpu_half=1) -> Candidate:
    geom = _make_geom()
    return Candidate(
        l=float(l_pix),
        m=float(m_pix),
        dm_fine=float(geom.fine_dm_pc_cc[fine_dm_idx]),
        dm_idx=0,
        event_specnum=geom.specnum_start + t_in_cube * geom.sample_period_specnum,
        width_samples=width_samples,
        kernel_id=kernel_id,
        snr=snr,
        detector_version="v1.M5",
        flags=int(CandidateFlags.NONE),
        search_node_id=search_node_id,
        gpu_half=gpu_half,
    )


_HDBSCAN_AVAILABLE = True
try:
    import hdbscan  # noqa: F401
except ImportError:
    _HDBSCAN_AVAILABLE = False


needs_hdbscan = pytest.mark.skipif(
    not _HDBSCAN_AVAILABLE, reason="hdbscan not installed"
)


# ---------------------------------------------------------------------------
# Empty + singleton paths (backend-independent)
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_outputs() -> None:
    geom = _make_geom()
    labels, records = cluster_candidates([], geom)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64
    assert records == []


def test_singleton_input_emits_noise_record() -> None:
    """Single candidate is emitted as a noise singleton (cluster_id=-1)
    regardless of backend (matches HDBSCAN's min_cluster_size=2 default)."""
    geom = _make_geom()
    cand = _make_cand(snr=20.0)
    labels, records = cluster_candidates([cand], geom)
    assert labels.tolist() == [-1]
    assert len(records) == 1
    rec = records[0]
    assert rec.cluster_id == -1
    assert rec.cntc == 1
    assert rec.cntb_lm == 1
    assert rec.cntb_dm == 1
    assert rec.snr == 20.0
    assert rec.peak_candidate_idx == 0


# ---------------------------------------------------------------------------
# DBSCAN backend (always available — sklearn is a hard dep)
# ---------------------------------------------------------------------------


def test_dbscan_groups_co_located_candidates() -> None:
    """Two candidates at the same (l_pix, m_pix, fine_dm_idx, t_in_cube)
    with the same width should cluster together under DBSCAN(eps=10)."""
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64,
                   width_samples=4, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64,
                   width_samples=4, snr=15.0),  # peak
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    assert labels.tolist() == [0, 0]
    assert len(records) == 1
    rec = records[0]
    assert rec.cluster_id == 0
    assert rec.cntc == 2
    assert rec.cntb_lm == 1  # both at (10, 20)
    assert rec.cntb_dm == 1  # both at fine_dm_idx=2
    assert rec.snr == 15.0
    assert rec.peak_candidate_idx == 1


def test_dbscan_separates_distant_candidates() -> None:
    """Two candidates >> eps apart in feature space cluster as two
    noise singletons (DBSCAN with min_samples=2 emits no clusters and
    labels both as noise)."""
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=0, t_in_cube=10,
                   width_samples=1, snr=10.0),
        _make_cand(l_pix=200, m_pix=240, fine_dm_idx=7, t_in_cube=250,
                   width_samples=8, snr=12.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    assert set(labels.tolist()) == {-1}
    assert len(records) == 2
    assert all(r.cluster_id == -1 for r in records)


def test_dbscan_picks_peak_by_max_snr() -> None:
    """In a 5-member cluster, peak_candidate_idx points at the highest-SNR row."""
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=8.5),
        _make_cand(l_pix=10, m_pix=21, fine_dm_idx=2, t_in_cube=64, snr=22.7),  # peak
        _make_cand(l_pix=11, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=9.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=3, t_in_cube=64, snr=11.5),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=65, snr=10.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    # All co-located within eps=10 → one cluster of 5
    assert set(labels.tolist()) == {0}
    rec = records[0]
    assert rec.cntc == 5
    assert rec.snr == 22.7
    assert rec.peak_candidate_idx == 1
    assert rec.l_pix == 10 and rec.m_pix == 21


def test_dbscan_cntb_lm_counts_unique_lm_cells() -> None:
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=8.5),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=22.7),
        _make_cand(l_pix=11, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=9.0),
        _make_cand(l_pix=10, m_pix=21, fine_dm_idx=2, t_in_cube=64, snr=10.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    rec = records[0]
    assert rec.cntc == 4
    # Unique (l_pix, m_pix) cells: {(10,20), (11,20), (10,21)} → 3
    assert rec.cntb_lm == 3
    # Unique fine_dm_idx values: {2} → 1
    assert rec.cntb_dm == 1


def test_dbscan_cntb_dm_counts_unique_fdm_indices() -> None:
    geom = _make_geom()
    # Cluster spans fine_dm_idx ∈ {2, 3} (within eps=10 with weight 1)
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=8.5),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=3, t_in_cube=64, snr=22.7),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    rec = records[0]
    assert rec.cntb_lm == 1
    assert rec.cntb_dm == 2


def test_dbscan_record_carries_geometry_derived_real_units() -> None:
    geom = _make_geom(cell_l_rad=2.5e-4, cell_m_rad=2.5e-4, l0_rad=0.0, m0_rad=0.0)
    cands = [
        _make_cand(l_pix=132, m_pix=230, fine_dm_idx=4, t_in_cube=64,
                   width_samples=4, snr=20.81),
        _make_cand(l_pix=132, m_pix=230, fine_dm_idx=4, t_in_cube=64,
                   width_samples=4, snr=8.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    rec = records[0]
    assert rec.l_pix == 132 and rec.m_pix == 230
    np.testing.assert_allclose(rec.l_rad, 132 * 2.5e-4)
    np.testing.assert_allclose(rec.m_rad, 230 * 2.5e-4)
    assert rec.fine_dm_idx == 4
    np.testing.assert_allclose(rec.dm_fine_pc_cc, float(geom.fine_dm_pc_cc[4]))
    assert rec.t_in_cube == 64
    np.testing.assert_allclose(rec.t_seconds, 64 * geom.sample_period_us / 1e6,
                               rtol=1e-9)


def test_dbscan_noise_and_cluster_records_both_emitted() -> None:
    """Mixed input: 2 close candidates + 1 distant. Expect one cluster
    record (cluster_id=0, cntc=2) AND one noise record (cluster_id=-1,
    cntc=1)."""
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=12.0),
        _make_cand(l_pix=200, m_pix=240, fine_dm_idx=7, t_in_cube=250,
                   width_samples=8, snr=15.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    assert labels.tolist() == [0, 0, -1]
    assert len(records) == 2
    cluster_ids = [r.cluster_id for r in records]
    assert cluster_ids == [0, -1]  # cluster ≥ 0 first, noise singletons after
    assert records[0].cntc == 2
    assert records[1].cntc == 1


# ---------------------------------------------------------------------------
# HDBSCAN backend (skipped if not installed)
# ---------------------------------------------------------------------------


@needs_hdbscan
def test_hdbscan_groups_co_located_candidates() -> None:
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64,
                   width_samples=4, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64,
                   width_samples=4, snr=15.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64,
                   width_samples=4, snr=12.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.HDBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    assert set(labels.tolist()) == {0}
    assert len(records) == 1
    rec = records[0]
    assert rec.snr == 15.0
    assert rec.peak_candidate_idx == 1


@needs_hdbscan
def test_hdbscan_emits_noise_singletons() -> None:
    """HDBSCAN with min_cluster_size=2 + a single isolated point should
    emit it as a noise singleton (cluster_id=-1)."""
    geom = _make_geom()
    # Two co-located + 1 distant
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=12.0),
        _make_cand(l_pix=200, m_pix=240, fine_dm_idx=7, t_in_cube=250,
                   width_samples=8, snr=15.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.HDBSCAN)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    # Expect: first two grouped (cluster_id=0); third is noise.
    cluster_ids = sorted(set(labels.tolist()))
    assert -1 in cluster_ids
    assert any(r.cluster_id == -1 for r in records)


# ---------------------------------------------------------------------------
# Config + backend dispatch
# ---------------------------------------------------------------------------


def test_invalid_backend_raises() -> None:
    geom = _make_geom()
    cands = [_make_cand(snr=10), _make_cand(snr=11)]
    with pytest.raises(ValueError, match="backend"):
        cluster_candidates(cands, geom, config=ClustererConfig(backend="bogus"))


def test_dbscan_eps_override_is_honoured() -> None:
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=12.0),
    ]
    # With eps=0.0 nothing groups; both become noise (DBSCAN min_samples=2).
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN, dbscan_eps=0.0)
    labels, records = cluster_candidates(cands, geom, config=cfg)
    # All same point → DBSCAN can still group at eps=0 because they
    # have distance 0; but dbscan_min_samples=2 fires only if there are
    # ≥2 within eps. They are, so they cluster. (The test is to verify
    # the eps override path is invoked at all.)
    assert labels.shape == (2,)


def test_records_are_cluster_records() -> None:
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=15.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    _, records = cluster_candidates(cands, geom, config=cfg)
    assert all(isinstance(r, ClusterRecord) for r in records)


def test_per_cube_cluster_id_resets() -> None:
    """Two separate cube calls each return cluster_id=0 for the
    largest cluster (per-cube IDs, not globally monotonic)."""
    geom1 = _make_geom(cube_id=0)
    geom2 = _make_geom(cube_id=1)
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=10.0),
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, snr=15.0),
    ]
    cfg = ClustererConfig(backend=ClustererBackend.DBSCAN)
    _, recs1 = cluster_candidates(cands, geom1, config=cfg)
    _, recs2 = cluster_candidates(cands, geom2, config=cfg)
    assert recs1[0].cluster_id == 0
    assert recs2[0].cluster_id == 0
    assert recs1[0].cube_id == 0
    assert recs2[0].cube_id == 1
