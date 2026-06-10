"""Candidate → 5D clustering-feature converter (M6 chunk 1, D3 lock).

Per M6 D3 the clusterer accepts EITHER detector-frame integer indices
``(l_pix, m_pix, fine_dm_idx, t_in_cube, log2_width_samples)`` OR
real-unit physical coordinates
``(l_rad, m_rad, dm_fine_pc_cc, t_seconds, log2_width_samples)``,
gated by ``FeatureMode``. Default is ``"int"`` (matches the dsa110-T2
``cluster_heimdall.py`` convention with ``cityblock`` distance + the
T2-reference weights ``(log2_width × 4, idm × 1, itime × 1, l_pix × 1,
m_pix × 1)``).

The 5-feature row layout is ``[log2_width, dm_axis, t_axis, l_axis,
m_axis]`` for both modes — same column ordering, different units. The
weights vector is applied COLUMN-WISE before clustering (cityblock /
manhattan distance is invariant to per-axis scaling, so multiplying
column ``j`` by ``w_j`` produces a metric where axis ``j`` carries
weight ``w_j``).

The T1/T2 ASCII log writer (chunk 2) ALWAYS uses the real-unit form for
output rows (M6 D3); ``FeatureMode`` only affects the in-process
clustering distance.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from ..common.contracts import Candidate, CubeGeometry

__all__ = [
    "FeatureMode",
    "DEFAULT_WEIGHTS",
    "candidates_to_features",
    "candidates_to_real_coords",
]


# Type alias / namespace for the two supported feature modes.
class FeatureMode:
    INT: str = "int"
    REAL: str = "real"


# Per M6 D3: T2-reference weights, columns in order
# [log2_width, dm_axis, t_axis, l_axis, m_axis].
DEFAULT_WEIGHTS: tuple[float, float, float, float, float] = (
    4.0,  # log2_width  (compress / boost width axis)
    1.0,  # dm_axis     (fine_dm_idx in INT mode; dm_fine_pc_cc in REAL mode)
    1.0,  # t_axis      (t_in_cube samples in INT; t_seconds in REAL)
    1.0,  # l_axis      (l_pix in INT; l_rad in REAL)
    1.0,  # m_axis
)


# ---------------------------------------------------------------------------
# Per-Candidate primitives — convert detector-frame fields to physical units.
# ---------------------------------------------------------------------------


def centred_pix_offset(pix: float, n_grid: int) -> float:
    """Signed pixel offset from the image centre: ``pix - n_grid // 2``.

    2026-06-10 (supersedes the short-lived ``signed_centred_pix`` FFT-
    wrap interpretation): the production cube images are CENTRED — the
    sky origin (l, m) = (0, 0) sits at pixel ``n_grid // 2`` on both
    spatial axes, exactly like the corr-side sky-export frames
    (``bench/_corr_fast_replay.pixel_to_lm_radians``: centre pixel
    ``n_grid // 2`` ⇔ (0, 0)). Confirmed live 2026-06-10 with a 10-shot
    (l, m) injection sweep at DM 500: every dumped-cube apex landed at
    ``true_coord / cell + 128`` to sub-pixel accuracy (e.g. l=+0.009 →
    col 200.5 = 0.009/1.24039e-4 + 128; an injection at m=0 reported
    pixel 0 − 128 → the notorious "corner" ±0.0192 rad value).

    The previous raw-irfft2-layout assumption (wrap negative
    coordinates to the top half of the axis) double-corrected: bursts
    in the top half were ALREADY at positive sky coordinates.

    No-op (returns ``pix``) for ``n_grid <= 0`` (test fixtures with no
    geometry).
    """
    p = float(pix)
    n = int(n_grid)
    if n <= 0:
        return p
    return p - float(n // 2)


def _candidate_to_int_indices(
    cand: Candidate,
    geom: CubeGeometry,
) -> tuple[int, int, int, int]:
    """Recover ``(l_pix, m_pix, fine_dm_idx, t_in_cube)`` from a Candidate.

    The M5 detector emits ``Candidate`` records with ``l, m`` as
    float-cast pixel indices and ``dm_fine`` as the resolved fine-DM in
    pc cm⁻³ (see ``detector.decoder.decode_local_max``). The fine-DM
    INDEX is not directly carried — it's recovered by binary-searching
    ``geom.fine_dm_pc_cc``.

    Args:
        cand: detector-emitted candidate.
        geom: cube geometry sidecar.

    Returns:
        Tuple ``(l_pix, m_pix, fine_dm_idx, t_in_cube)``, all int.
        ``fine_dm_idx`` is the closest grid index by absolute difference
        (handles slight float-roundoff between cand.dm_fine and
        geom.fine_dm_pc_cc[idx]).
    """
    l_pix = int(round(cand.l))
    m_pix = int(round(cand.m))
    # Closest fine-DM grid index. searchsorted returns the insertion
    # point; we test both candidates and pick the closer one.
    insert = int(np.searchsorted(geom.fine_dm_pc_cc, cand.dm_fine))
    if insert <= 0:
        fine_dm_idx = 0
    elif insert >= geom.n_fdm_in_cube:
        fine_dm_idx = geom.n_fdm_in_cube - 1
    else:
        left = abs(geom.fine_dm_pc_cc[insert - 1] - cand.dm_fine)
        right = abs(geom.fine_dm_pc_cc[insert] - cand.dm_fine)
        fine_dm_idx = (insert - 1) if left <= right else insert
    t_in_cube = (cand.event_specnum - geom.specnum_start) // geom.sample_period_specnum
    return l_pix, m_pix, fine_dm_idx, int(t_in_cube)


def _log2_width(width_samples: int) -> float:
    """Return ``log2(width_samples)``, with the M5/T2 convention that
    width=1 → 0.0 (single-sample boxcar = log2(1)=0)."""
    if width_samples <= 0:
        raise ValueError(f"width_samples={width_samples} must be > 0")
    return math.log2(float(width_samples))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def candidates_to_features(
    cands: Sequence[Candidate],
    geom: CubeGeometry,
    *,
    mode: str = FeatureMode.INT,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Convert per-cube candidate list to ``[N_cands, 5]`` feature matrix.

    Column layout (both modes):
        col 0 = log2(width_samples)
        col 1 = dm_axis        (fine_dm_idx [INT] | dm_fine_pc_cc [REAL])
        col 2 = t_axis         (t_in_cube  [INT] | t_seconds      [REAL])
        col 3 = l_axis         (l_pix      [INT] | l_rad          [REAL])
        col 4 = m_axis         (m_pix      [INT] | m_rad          [REAL])

    Each column is multiplied by its corresponding weight after unit
    conversion (cityblock / manhattan distance is invariant to per-axis
    scaling, so this is the standard way to inject per-axis weights into
    HDBSCAN / DBSCAN).

    Args:
        cands: per-cube candidate list (must match the order used to
            interpret cluster labels downstream).
        geom: cube geometry sidecar (for INT→REAL conversion in REAL mode
            and for fine_dm_idx + t_in_cube recovery in INT mode).
        mode: ``"int"`` (default) or ``"real"``.
        weights: 5-tuple of column weights. Defaults to
            ``DEFAULT_WEIGHTS`` (= T2-reference weights).

    Returns:
        ``np.ndarray[N_cands, 5]`` float64. Empty array
        ``np.zeros((0, 5), dtype=np.float64)`` if ``cands`` is empty.

    Raises:
        ValueError: if ``mode`` is not in ``{"int", "real"}`` or
            ``weights`` length is not 5.
    """
    if mode not in (FeatureMode.INT, FeatureMode.REAL):
        raise ValueError(
            f"mode={mode!r}, expected one of {FeatureMode.INT!r}, {FeatureMode.REAL!r}"
        )
    if len(weights) != 5:
        raise ValueError(f"weights must be length 5; got len={len(weights)}")
    if not cands:
        return np.zeros((0, 5), dtype=np.float64)

    n = len(cands)
    out = np.empty((n, 5), dtype=np.float64)
    if mode == FeatureMode.INT:
        for i, cand in enumerate(cands):
            l_pix, m_pix, fine_dm_idx, t_in_cube = _candidate_to_int_indices(
                cand, geom
            )
            out[i, 0] = _log2_width(cand.width_samples)
            out[i, 1] = float(fine_dm_idx)
            out[i, 2] = float(t_in_cube)
            out[i, 3] = float(l_pix)
            out[i, 4] = float(m_pix)
    else:  # REAL
        sec_per_specnum = geom.sample_period_us / 1e6 / geom.sample_period_specnum
        for i, cand in enumerate(cands):
            l_pix, m_pix, fine_dm_idx, t_in_cube = _candidate_to_int_indices(
                cand, geom
            )
            out[i, 0] = _log2_width(cand.width_samples)
            out[i, 1] = float(cand.dm_fine)
            # t_seconds = (event_specnum - specnum_start) * sec_per_specnum.
            # Equivalently t_in_cube * sample_period_us / 1e6.
            out[i, 2] = (cand.event_specnum - geom.specnum_start) * sec_per_specnum
            # 2026-06-10 axis fix: Candidate.l is the cube ROW index
            # (H axis) and Candidate.m the COLUMN index (W axis) — but
            # in the validated sky frame (sky_astrometry /
            # _corr_fast_replay: row = m, col = l) the ROW is the
            # m-axis and the COLUMN the l-axis. Confirmed live with the
            # 2026-06-10 (l, m) injection sweep: a fixed injected m
            # pinned the reported "l" while sweeping injected l moved
            # the reported "m". Swap here (and at every l_rad/m_rad
            # conversion site) so physical coordinates are true sky
            # (l, m); raw l_pix/m_pix stay cube (row, col) indices.
            out[i, 3] = (
                centred_pix_offset(m_pix, geom.n_grid) * geom.cell_l_rad
                + geom.l0_rad
            )
            out[i, 4] = (
                centred_pix_offset(l_pix, geom.n_grid) * geom.cell_m_rad
                + geom.m0_rad
            )

    w = np.asarray(weights, dtype=np.float64)
    out *= w[None, :]
    return out


def candidates_to_real_coords(
    cands: Sequence[Candidate],
    geom: CubeGeometry,
) -> List[tuple[float, float, float, float, int, int, int, int]]:
    """Convert candidates to a list of physical-unit + integer-index
    tuples ``(l_rad, m_rad, dm_fine_pc_cc, t_seconds, l_pix, m_pix,
    fine_dm_idx, t_in_cube)``.

    Used by the clusterer to populate ``ClusterRecord`` peak fields and
    by the T1 logger (chunk 2) to write real-unit rows for non-peak
    candidates.

    Args:
        cands: per-cube candidate list.
        geom: cube geometry sidecar.

    Returns:
        List of length ``len(cands)``. Empty list if ``cands`` is empty.
    """
    if not cands:
        return []
    out: List[tuple[float, float, float, float, int, int, int, int]] = []
    sec_per_specnum = geom.sample_period_us / 1e6 / geom.sample_period_specnum
    for cand in cands:
        l_pix, m_pix, fine_dm_idx, t_in_cube = _candidate_to_int_indices(
            cand, geom
        )
        # Axis swap: row (l_pix) = sky m-axis, col (m_pix) = sky l-axis
        # — see the conversion note in candidates_to_features.
        l_rad = (
            centred_pix_offset(m_pix, geom.n_grid) * geom.cell_l_rad
            + geom.l0_rad
        )
        m_rad = (
            centred_pix_offset(l_pix, geom.n_grid) * geom.cell_m_rad
            + geom.m0_rad
        )
        t_seconds = (cand.event_specnum - geom.specnum_start) * sec_per_specnum
        out.append(
            (l_rad, m_rad, float(cand.dm_fine), t_seconds,
             l_pix, m_pix, fine_dm_idx, t_in_cube)
        )
    return out
