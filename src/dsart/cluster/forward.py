"""Per-cube clusterer (HDBSCAN primary, DBSCAN fallback) — M6 chunk 1.

Per M6 D4 the primary backend is HDBSCAN with cityblock (= manhattan)
distance, ``min_cluster_size=2`` (matches T2's ``min_samples=2``;
preserves the "≥2 candidates anywhere" cluster behaviour),
``min_samples=1`` (HDBSCAN-specific; controls noise-label
conservatism), and ``cluster_selection_epsilon=10.0`` (matches T2's
``eps=10`` cityblock distance).

Per M6 D5 the fallback is sklearn's ``DBSCAN(eps=10, min_samples=2,
metric="cityblock")`` — invoked only when the chunk-6 throughput bench
proves HDBSCAN p99 > 50 ms at production load. Selection at runtime is
via the ``backend`` argument.

Per M6 D6 the clusterer is per-cube; cluster ids reset at each cube
call. Bursts that straddle a cube boundary will produce two clusters;
re-clustering across cubes is intentionally out of scope (offline T2-
level re-cluster against the T1 ASCII log if needed).

Per M6 D1 the output ``ClusterRecord`` carries the peak (highest-SNR)
candidate's full feature set in BOTH integer-index frame (l_pix, m_pix,
fine_dm_idx, t_in_cube) AND real-unit frame (l_rad, m_rad,
dm_fine_pc_cc, t_seconds). The T1/T2 logger always writes real units.

Noise label convention: HDBSCAN/DBSCAN label ``-1`` denotes points not
assigned to any cluster. We emit each noise point as a singleton
``ClusterRecord`` with ``cluster_id = -1`` and
``cntc = cntb_lm = cntb_dm = 1``. This preserves the operator's ability
to inspect every candidate (the T1 logger writes one row per candidate
regardless of cluster assignment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..common.contracts import Candidate, ClusterRecord, CubeGeometry
from .features import (
    DEFAULT_WEIGHTS,
    FeatureMode,
    candidates_to_features,
    candidates_to_real_coords,
)

__all__ = [
    "ClustererBackend",
    "ClustererConfig",
    "cluster_candidates",
]

_LOG = logging.getLogger(__name__)


# Type alias / namespace for the two supported backends.
class ClustererBackend:
    HDBSCAN: str = "hdbscan"
    DBSCAN: str = "dbscan"


@dataclass(frozen=True, slots=True)
class ClustererConfig:
    """Per-process clusterer configuration (M6 D3/D4/D5).

    Defaults match the M6 D4 lock (HDBSCAN) and the T2-reference
    cityblock hyperparameters. The chunk-6 throughput bench may flip
    ``backend`` to ``"dbscan"`` if HDBSCAN p99 > 50 ms.

    Args:
        backend: ``"hdbscan"`` (default) or ``"dbscan"``.
        feature_mode: ``"int"`` (default; T2-convention) or ``"real"``.
        weights: 5-tuple of column weights; column order is
            ``[log2_width, dm_axis, t_axis, l_axis, m_axis]``.
            Defaults to ``DEFAULT_WEIGHTS``.
        min_cluster_size: HDBSCAN minimum cluster size. Default 2 to
            match T2's ``min_samples=2`` "≥2 candidates anywhere"
            behaviour. (HDBSCAN-only; DBSCAN uses ``dbscan_min_samples``.)
        min_samples: HDBSCAN core-point requirement. Default 1
            (aggressive — labels everything as part of a cluster vs
            noise less aggressively than ``min_cluster_size``).
        cluster_selection_epsilon: HDBSCAN distance threshold (matches
            T2's ``eps=10`` cityblock).
        dbscan_eps: DBSCAN ``eps`` (used only when ``backend == "dbscan"``).
            Defaults to ``cluster_selection_epsilon``.
        dbscan_min_samples: DBSCAN ``min_samples``. Default 2.
        metric: ``"manhattan"`` (HDBSCAN spelling) — passed through.
            DBSCAN uses ``"cityblock"``; we translate internally.
    """

    backend: str = ClustererBackend.HDBSCAN
    feature_mode: str = FeatureMode.INT
    weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS
    min_cluster_size: int = 2
    min_samples: int = 1
    cluster_selection_epsilon: float = 10.0
    dbscan_eps: Optional[float] = None
    dbscan_min_samples: int = 2
    metric: str = "manhattan"


# ---------------------------------------------------------------------------
# HDBSCAN / DBSCAN dispatch
# ---------------------------------------------------------------------------


def _label_with_hdbscan(
    features: np.ndarray, config: ClustererConfig
) -> np.ndarray:
    """Run HDBSCAN; return per-point integer labels (-1 = noise).

    Lazy-imports hdbscan so that the module loads in environments
    without hdbscan installed (tests + DBSCAN-only fallback paths).
    """
    import hdbscan  # type: ignore[import-not-found]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_epsilon=float(config.cluster_selection_epsilon),
        metric=config.metric,
        allow_single_cluster=True,
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(features).astype(np.int64, copy=False)


def _label_with_dbscan(
    features: np.ndarray, config: ClustererConfig
) -> np.ndarray:
    """Run sklearn DBSCAN; return per-point integer labels (-1 = noise)."""
    from sklearn.cluster import DBSCAN  # type: ignore[import-not-found]

    eps = (
        float(config.dbscan_eps)
        if config.dbscan_eps is not None
        else float(config.cluster_selection_epsilon)
    )
    db = DBSCAN(
        eps=eps,
        min_samples=int(config.dbscan_min_samples),
        metric="cityblock",
        n_jobs=1,
    )
    return db.fit_predict(features).astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_candidates(
    cands: Sequence[Candidate],
    geom: CubeGeometry,
    *,
    config: Optional[ClustererConfig] = None,
) -> Tuple[np.ndarray, List[ClusterRecord]]:
    """Cluster the per-cube candidate list and emit one record per cluster
    (plus one singleton record per noise point).

    Args:
        cands: per-cube candidate list (as emitted by the M5 detector +
            cross-kernel merger).
        geom: cube geometry sidecar (for INT→REAL conversion + fine_dm_idx
            recovery).
        config: clusterer config; defaults to ``ClustererConfig()`` (HDBSCAN
            + INT mode + T2-reference weights).

    Returns:
        Tuple ``(labels, records)`` where:
          * ``labels`` — ``np.ndarray[len(cands)]`` int64; the per-candidate
            cluster id (-1 for noise points). Index-aligned with ``cands``
            (consumers like the T1 logger use this to attach a cluster
            label to each input row).
          * ``records`` — ``List[ClusterRecord]``, one per cluster id ≥ 0
            and one per noise point. Cluster-id ≥ 0 records are sorted
            ascending by cluster id; noise singletons follow at the end
            in input-list order.

    Raises:
        ImportError: if ``backend == "hdbscan"`` and ``hdbscan`` is not
            installed.
    """
    cfg = config or ClustererConfig()
    if not cands:
        return np.zeros(0, dtype=np.int64), []

    features = candidates_to_features(
        cands, geom, mode=cfg.feature_mode, weights=cfg.weights
    )
    n_cands = len(cands)
    if n_cands == 1:
        # Singleton fast path — no clustering needed; emit as cluster_id=0
        # if backend would have grouped a single point as its own cluster
        # (HDBSCAN with min_cluster_size=2 would label it -1, so honour
        # the "noise" semantics: emit as a -1 singleton record).
        labels = np.full(1, -1, dtype=np.int64)
    elif cfg.backend == ClustererBackend.HDBSCAN:
        labels = _label_with_hdbscan(features, cfg)
    elif cfg.backend == ClustererBackend.DBSCAN:
        labels = _label_with_dbscan(features, cfg)
    else:
        raise ValueError(
            f"backend={cfg.backend!r}, expected one of "
            f"{ClustererBackend.HDBSCAN!r}, {ClustererBackend.DBSCAN!r}"
        )

    coords = candidates_to_real_coords(cands, geom)

    records: List[ClusterRecord] = []
    unique_labels = sorted(int(c) for c in set(labels.tolist()) if c >= 0)
    for cid in unique_labels:
        member_idxs = np.where(labels == cid)[0]
        records.append(_build_record(cid, member_idxs, cands, coords, geom))

    # Noise points (label == -1): one singleton record per point, in
    # input-list order. cluster_id stays -1 (operator-visible).
    noise_idxs = np.where(labels == -1)[0]
    for idx in noise_idxs:
        records.append(_build_record(-1, np.asarray([idx]), cands, coords, geom))

    return labels, records


def _build_record(
    cluster_id: int,
    member_idxs: np.ndarray,
    cands: Sequence[Candidate],
    coords: List[tuple],
    geom: CubeGeometry,
) -> ClusterRecord:
    """Build one ClusterRecord from a list of member indices.

    Picks the highest-SNR candidate as the peak; computes cntc /
    cntb_lm / cntb_dm from the member set.
    """
    member_snrs = np.asarray([cands[i].snr for i in member_idxs])
    peak_within = int(np.argmax(member_snrs))
    peak_idx = int(member_idxs[peak_within])
    peak_cand = cands[peak_idx]
    (l_rad, m_rad, dm_fine_pc_cc, t_seconds,
     l_pix, m_pix, fine_dm_idx, t_in_cube) = coords[peak_idx]

    lm_set = {
        (coords[i][4], coords[i][5]) for i in member_idxs
    }  # (l_pix, m_pix) tuples
    dm_set = {coords[i][6] for i in member_idxs}  # fine_dm_idx values

    return ClusterRecord(
        cluster_id=int(cluster_id),
        cube_id=int(geom.cube_id),
        cntc=int(member_idxs.shape[0]),
        cntb_lm=int(len(lm_set)),
        cntb_dm=int(len(dm_set)),
        peak_candidate_idx=peak_idx,
        l_rad=float(l_rad),
        m_rad=float(m_rad),
        l_pix=int(l_pix),
        m_pix=int(m_pix),
        dm_fine_pc_cc=float(dm_fine_pc_cc),
        fine_dm_idx=int(fine_dm_idx),
        t_in_cube=int(t_in_cube),
        t_seconds=float(t_seconds),
        width_samples=int(peak_cand.width_samples),
        snr=float(peak_cand.snr),
        kernel_id=peak_cand.kernel_id,
        event_specnum=int(peak_cand.event_specnum),
        search_node_id=int(peak_cand.search_node_id),
        gpu_half=int(peak_cand.gpu_half),
    )
