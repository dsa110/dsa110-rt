"""Tests for ``dsart.cluster.features`` (M6 chunk 1)."""

from __future__ import annotations

import os

os.environ.setdefault("DSART_TEST", "1")

import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dsart.cluster.features import (  # noqa: E402
    DEFAULT_WEIGHTS,
    FeatureMode,
    candidates_to_features,
    candidates_to_real_coords,
)
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
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
    """Build a Candidate that matches the M5 detector output convention.

    The detector emits ``Candidate.l = float(l_idx)``,
    ``Candidate.m = float(m_idx)``, ``Candidate.dm_fine`` resolved
    through ``DmPlan``; ``event_specnum = specnum_start + t_in_cube``
    (``detector/decoder.py:216``) — both terms in SEARCH-sample units,
    so ``sample_period_specnum`` does NOT appear here. It did until
    2026-08-06, which made this fixture self-consistent with the
    matching divisor bug in ``_candidate_to_int_indices`` and hid it.
    """
    geom = _make_geom()
    return Candidate(
        l=float(l_pix),
        m=float(m_pix),
        dm_fine=float(geom.fine_dm_pc_cc[fine_dm_idx]),
        dm_idx=0,
        event_specnum=geom.specnum_start + t_in_cube,
        width_samples=width_samples,
        kernel_id=kernel_id,
        snr=snr,
        detector_version="v1.M5",
        flags=int(CandidateFlags.NONE),
        search_node_id=search_node_id,
        gpu_half=gpu_half,
    )


def test_empty_candidate_list_returns_zero_rows() -> None:
    geom = _make_geom()
    out = candidates_to_features([], geom)
    assert out.shape == (0, 5)
    assert out.dtype == np.float64


def test_int_mode_columns_match_detector_indices() -> None:
    geom = _make_geom()
    cands = [
        _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, width_samples=4),
        _make_cand(l_pix=11, m_pix=21, fine_dm_idx=3, t_in_cube=65, width_samples=8),
    ]
    out = candidates_to_features(cands, geom, mode=FeatureMode.INT,
                                 weights=(1, 1, 1, 1, 1))
    # Column layout: log2_width, fine_dm_idx, t_in_cube, l_pix, m_pix
    assert out[0].tolist() == [math.log2(4), 2, 64, 10, 20]
    assert out[1].tolist() == [math.log2(8), 3, 65, 11, 21]


def test_real_mode_l_m_in_radians() -> None:
    geom = _make_geom(cell_l_rad=2.0e-4, cell_m_rad=3.0e-4, l0_rad=1.0e-3, m0_rad=2.0e-3)
    cands = [_make_cand(l_pix=5, m_pix=7, fine_dm_idx=1, t_in_cube=10, width_samples=2)]
    out = candidates_to_features(cands, geom, mode=FeatureMode.REAL,
                                 weights=(1, 1, 1, 1, 1))
    # CENTRED image + row/col → (m, l) axis swap (2026-06-10): l_rad
    # comes from the COLUMN index (m_pix) and m_rad from the ROW index
    # (l_pix), both relative to the centre pixel n_grid//2 = 128.
    expected_l = (7 - 128) * 2.0e-4 + 1.0e-3
    expected_m = (5 - 128) * 3.0e-4 + 2.0e-3
    expected_dm = float(geom.fine_dm_pc_cc[1])
    expected_t_s = 10 * geom.sample_period_us / 1e6
    np.testing.assert_allclose(out[0, 3], expected_l, rtol=1e-12)
    np.testing.assert_allclose(out[0, 4], expected_m, rtol=1e-12)
    np.testing.assert_allclose(out[0, 1], expected_dm, rtol=1e-12)
    np.testing.assert_allclose(out[0, 2], expected_t_s, rtol=1e-9)
    np.testing.assert_allclose(out[0, 0], math.log2(2), rtol=1e-12)


def test_real_mode_centred_swapped_axes() -> None:
    """The cube image is CENTRED (sky origin at pixel n_grid//2) and
    the cube ROW axis is sky m while the COLUMN axis is sky l —
    confirmed live 2026-06-10 with a 10-shot (l, m) injection sweep
    (every dumped-cube apex at ``true_coord / cell + 128`` on the
    swapped axes)."""
    geom = _make_geom()  # n_grid=256, cell=1.5e-4, l0=m0=0
    cands = [_make_cand(l_pix=196, m_pix=60, fine_dm_idx=1, t_in_cube=10)]
    out = candidates_to_features(cands, geom, mode=FeatureMode.REAL,
                                 weights=(1, 1, 1, 1, 1))
    # l_rad from the COLUMN (m_pix=60), m_rad from the ROW (l_pix=196).
    np.testing.assert_allclose(out[0, 3], (60 - 128) * 1.5e-4, rtol=1e-12)
    np.testing.assert_allclose(out[0, 4], (196 - 128) * 1.5e-4, rtol=1e-12)
    # INT mode keeps raw cube-layout pixel indices untouched.
    out_int = candidates_to_features(cands, geom, mode=FeatureMode.INT,
                                     weights=(1, 1, 1, 1, 1))
    assert out_int[0, 3] == 196.0

    coords = candidates_to_real_coords(cands, geom)
    l_rad, m_rad = coords[0][0], coords[0][1]
    np.testing.assert_allclose(l_rad, (60 - 128) * 1.5e-4, rtol=1e-12)
    np.testing.assert_allclose(m_rad, (196 - 128) * 1.5e-4, rtol=1e-12)
    assert coords[0][4] == 196  # l_pix stays raw


def test_weights_applied_column_wise() -> None:
    geom = _make_geom()
    cand = _make_cand(l_pix=10, m_pix=20, fine_dm_idx=2, t_in_cube=64, width_samples=4)
    weights = (4.0, 1.0, 1.0, 1.0, 1.0)  # T2-default: log2_width × 4
    out = candidates_to_features([cand], geom, mode=FeatureMode.INT, weights=weights)
    assert out[0, 0] == 4.0 * math.log2(4)
    assert out[0, 1] == 2.0  # weight 1
    assert out[0, 3] == 10.0  # weight 1


def test_default_weights_are_t2_reference() -> None:
    assert DEFAULT_WEIGHTS == (4.0, 1.0, 1.0, 1.0, 1.0)


def test_invalid_mode_raises() -> None:
    geom = _make_geom()
    cand = _make_cand()
    with pytest.raises(ValueError, match="mode"):
        candidates_to_features([cand], geom, mode="bogus")


def test_invalid_weights_length_raises() -> None:
    geom = _make_geom()
    cand = _make_cand()
    with pytest.raises(ValueError, match="weights"):
        candidates_to_features([cand], geom, weights=(1.0, 2.0, 3.0))


def test_zero_width_raises() -> None:
    geom = _make_geom()
    # Synthesise a Candidate-like object that bypasses the dataclass
    # __post_init__ — the inner _log2_width is supposed to reject
    # width_samples <= 0 even if a caller manages to slip it through.
    # Use the public API; Candidate refuses width_samples=0 via its
    # own __post_init__, so we just verify the API rejects via Candidate.
    with pytest.raises(ValueError, match="width_samples"):
        _make_cand(width_samples=0)


def test_candidates_to_real_coords_returns_aligned_tuples() -> None:
    geom = _make_geom(cell_l_rad=2.0e-4, cell_m_rad=3.0e-4)
    cands = [
        _make_cand(l_pix=5, m_pix=7, fine_dm_idx=1, t_in_cube=10, width_samples=2),
        _make_cand(l_pix=99, m_pix=42, fine_dm_idx=4, t_in_cube=200, width_samples=8),
    ]
    coords = candidates_to_real_coords(cands, geom)
    assert len(coords) == 2
    (l_rad, m_rad, dm_pc, t_s, l_pix, m_pix, fdm_idx, t_in_cube) = coords[0]
    assert l_pix == 5 and m_pix == 7 and fdm_idx == 1 and t_in_cube == 10
    # Centred + swapped axes: l from col (m_pix), m from row (l_pix).
    np.testing.assert_allclose(l_rad, (7 - 128) * 2.0e-4)
    np.testing.assert_allclose(m_rad, (5 - 128) * 3.0e-4)
    np.testing.assert_allclose(dm_pc, float(geom.fine_dm_pc_cc[1]))


def test_candidates_to_real_coords_empty() -> None:
    geom = _make_geom()
    assert candidates_to_real_coords([], geom) == []


def test_fine_dm_idx_recovery_picks_closest() -> None:
    geom = _make_geom(fine_dm_pc_cc=np.array([50.0, 100.0, 200.0, 400.0],
                                              dtype=np.float64),
                      n_fdm_in_cube=4)
    # Synthesise a candidate whose dm_fine is mid-way between [100, 200]
    # → closer to 100 (dist 50 vs 50; tie-break to left).
    cand = Candidate(
        l=0.0, m=0.0, dm_fine=150.0, dm_idx=0,
        event_specnum=geom.specnum_start, width_samples=1,
        kernel_id="unit:d1:b1", snr=8.0, detector_version="v1.M5",
        flags=0, search_node_id=0, gpu_half=0,
    )
    out = candidates_to_features([cand], geom, mode=FeatureMode.INT,
                                 weights=(1, 1, 1, 1, 1))
    # log2(1) = 0, fdm_idx = 1 (tied; left-bias), t_in_cube=0, l_pix=0, m_pix=0
    assert out[0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_fine_dm_idx_clamps_above_max() -> None:
    geom = _make_geom(fine_dm_pc_cc=np.array([50.0, 100.0, 200.0],
                                              dtype=np.float64),
                      n_fdm_in_cube=3)
    cand = Candidate(
        l=0.0, m=0.0, dm_fine=999.0, dm_idx=0,
        event_specnum=geom.specnum_start, width_samples=1,
        kernel_id="unit:d1:b1", snr=8.0, detector_version="v1.M5",
        flags=0, search_node_id=0, gpu_half=0,
    )
    out = candidates_to_features([cand], geom, mode=FeatureMode.INT,
                                 weights=(1, 1, 1, 1, 1))
    assert out[0, 1] == 2.0  # last index


def test_fine_dm_idx_clamps_below_min() -> None:
    geom = _make_geom(fine_dm_pc_cc=np.array([50.0, 100.0, 200.0],
                                              dtype=np.float64),
                      n_fdm_in_cube=3)
    cand = Candidate(
        l=0.0, m=0.0, dm_fine=10.0, dm_idx=0,
        event_specnum=geom.specnum_start, width_samples=1,
        kernel_id="unit:d1:b1", snr=8.0, detector_version="v1.M5",
        flags=0, search_node_id=0, gpu_half=0,
    )
    out = candidates_to_features([cand], geom, mode=FeatureMode.INT,
                                 weights=(1, 1, 1, 1, 1))
    assert out[0, 1] == 0.0  # first index
