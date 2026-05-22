"""Tests for the M7.4 CPU-side cube retention ring.

``src/dsart/services/cube_pipeline.py``:

  * ``RetainedCube`` — pinned-host cube tensor + metadata.
  * ``CubeRetentionRing`` — circular buffer of pinned cubes with
    write-pos rotation and newest-first iteration.
  * ``find_cube_for_specnum`` — specnum-window lookup helper used by
    the M7.4 C2 trigger listener.

These tests do NOT exercise the pinned allocator (we run on CPU-only
CI hosts); they pass ``pinned=False`` so the ring falls back to plain
numpy buffers.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.services.cube_pipeline import (  # noqa: E402
    CubeRetentionRing,
    RetainedCube,
    find_cube_for_specnum,
)


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        CubeRetentionRing(depth=0, t_det=4, n_fdm=2, n_grid=4, pinned=False)
    with pytest.raises(ValueError):
        CubeRetentionRing(depth=-1, t_det=4, n_fdm=2, n_grid=4, pinned=False)
    with pytest.raises(ValueError):
        CubeRetentionRing(depth=2, t_det=0, n_fdm=2, n_grid=4, pinned=False)
    with pytest.raises(ValueError):
        CubeRetentionRing(depth=2, t_det=4, n_fdm=0, n_grid=4, pinned=False)
    with pytest.raises(ValueError):
        CubeRetentionRing(depth=2, t_det=4, n_fdm=2, n_grid=0, pinned=False)


def test_empty_ring_lookup_is_none() -> None:
    r = CubeRetentionRing(depth=4, t_det=8, n_fdm=2, n_grid=4, pinned=False)
    assert r.n_committed == 0
    assert r.write_pos == 0
    assert find_cube_for_specnum(r, 0) is None
    assert find_cube_for_specnum(r, 12345) is None
    assert r.snapshot() == []


# ---------------------------------------------------------------------------
# stage_cube semantics
# ---------------------------------------------------------------------------


def _make_payload(t_det: int, n_fdm: int, n_grid: int, value: float) -> np.ndarray:
    arr = np.full(
        (t_det, n_fdm, n_grid, n_grid), value, dtype=np.float16,
    )
    return arr


def test_single_cube_stage_and_lookup() -> None:
    t_det, n_fdm, n_grid, sps = 4, 2, 4, 16
    r = CubeRetentionRing(
        depth=2, t_det=t_det, n_fdm=n_fdm, n_grid=n_grid, pinned=False,
    )
    payload = _make_payload(t_det, n_fdm, n_grid, value=1.0)
    rec = r.stage_cube(
        cube_id=0,
        event_specnum_start=1000,
        mjd_start=58000.0,
        sample_period_specnum=sps,
        sample_period_us=1048.576,
        cube_tensor=payload,
    )
    assert isinstance(rec, RetainedCube)
    assert rec.cube_id == 0
    assert rec.event_specnum_start == 1000
    assert rec.t_det == t_det
    assert rec.n_fdm == n_fdm
    assert rec.n_grid == n_grid
    assert rec.sample_period_specnum == sps
    assert r.n_committed == 1
    # Spec-num window is [1000, 1000 + 4 * 16) = [1000, 1064).
    assert find_cube_for_specnum(r, 1000) is rec
    assert find_cube_for_specnum(r, 1063) is rec
    assert find_cube_for_specnum(r, 1064) is None
    assert find_cube_for_specnum(r, 999) is None


def test_stage_rejects_shape_mismatch() -> None:
    r = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    bad = np.zeros((4, 2, 8, 8), dtype=np.float16)
    with pytest.raises(ValueError):
        r.stage_cube(
            cube_id=0,
            event_specnum_start=0,
            mjd_start=58000.0,
            sample_period_specnum=16,
            sample_period_us=1048.576,
            cube_tensor=bad,
        )


def test_stage_rejects_bad_cube_id() -> None:
    r = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    payload = _make_payload(4, 2, 4, 1.0)
    with pytest.raises(ValueError):
        r.stage_cube(
            cube_id=-1,
            event_specnum_start=0,
            mjd_start=58000.0,
            sample_period_specnum=16,
            sample_period_us=1048.576,
            cube_tensor=payload,
        )


def test_stage_accepts_float32_input_and_downcasts() -> None:
    t_det, n_fdm, n_grid = 4, 2, 4
    r = CubeRetentionRing(
        depth=2, t_det=t_det, n_fdm=n_fdm, n_grid=n_grid, pinned=False,
    )
    payload32 = np.ones((t_det, n_fdm, n_grid, n_grid), dtype=np.float32)
    rec = r.stage_cube(
        cube_id=0,
        event_specnum_start=0,
        mjd_start=58000.0,
        sample_period_specnum=16,
        sample_period_us=1048.576,
        cube_tensor=payload32,
    )
    assert rec.pinned_host_tensor.dtype == np.float16
    assert np.all(rec.pinned_host_tensor == 1.0)


# ---------------------------------------------------------------------------
# Ring rotation + newest-first lookup
# ---------------------------------------------------------------------------


def test_ring_rotation_evicts_oldest() -> None:
    t_det, n_fdm, n_grid, sps = 4, 2, 4, 16
    depth = 3
    r = CubeRetentionRing(
        depth=depth, t_det=t_det, n_fdm=n_fdm, n_grid=n_grid, pinned=False,
    )
    # Stage 5 cubes; only the last 3 should be live.
    for k in range(5):
        r.stage_cube(
            cube_id=k,
            event_specnum_start=1000 + k * 64,  # cubes don't overlap (64 = t_det * sps)
            mjd_start=58000.0 + 1e-6 * k,
            sample_period_specnum=sps,
            sample_period_us=1048.576,
            cube_tensor=_make_payload(t_det, n_fdm, n_grid, float(k)),
        )
    assert r.n_committed == 5
    snap = r.snapshot()
    assert len(snap) == depth
    assert [c.cube_id for c in snap] == [4, 3, 2]
    # Spec-nums in evicted cubes 0/1 → miss.
    assert find_cube_for_specnum(r, 1000) is None
    assert find_cube_for_specnum(r, 1064) is None
    # Spec-num in live cube 2 → hit.
    cube2 = find_cube_for_specnum(r, 1128)
    assert cube2 is not None and cube2.cube_id == 2
    # Spec-num in live cube 4 → hit.
    cube4 = find_cube_for_specnum(r, 1300)
    assert cube4 is not None and cube4.cube_id == 4
    # Spec-num beyond newest cube → miss (too_early classification on
    # the listener side).
    newest = snap[0]
    end_excl = (
        int(newest.event_specnum_start)
        + int(newest.t_det) * int(newest.sample_period_specnum)
    )
    assert find_cube_for_specnum(r, end_excl) is None


def test_newest_wins_on_overlap() -> None:
    """When two ring slots happen to cover the same specnum window
    (only possible under extreme overlap; production cubes advance by
    ``cube_cadence_samples`` < ``t_det``), the freshest hit wins."""
    t_det, n_fdm, n_grid, sps = 4, 2, 4, 16
    r = CubeRetentionRing(
        depth=4, t_det=t_det, n_fdm=n_fdm, n_grid=n_grid, pinned=False,
    )
    # Two cubes with overlapping spec-num windows.
    r.stage_cube(
        cube_id=10,
        event_specnum_start=1000,
        mjd_start=58000.0,
        sample_period_specnum=sps,
        sample_period_us=1048.576,
        cube_tensor=_make_payload(t_det, n_fdm, n_grid, 1.0),
    )
    r.stage_cube(
        cube_id=11,
        event_specnum_start=1016,  # overlaps with cube 10 (window 1000..1064)
        mjd_start=58000.0001,
        sample_period_specnum=sps,
        sample_period_us=1048.576,
        cube_tensor=_make_payload(t_det, n_fdm, n_grid, 2.0),
    )
    hit = find_cube_for_specnum(r, 1020)
    assert hit is not None
    assert hit.cube_id == 11  # newest wins


def test_iter_oldest_first_after_rotation() -> None:
    r = CubeRetentionRing(
        depth=3, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    for k in range(5):
        r.stage_cube(
            cube_id=k,
            event_specnum_start=1000 + k * 100,
            mjd_start=58000.0,
            sample_period_specnum=16,
            sample_period_us=1048.576,
            cube_tensor=_make_payload(4, 2, 4, float(k)),
        )
    oldest_first = [c.cube_id for c in r.iter_oldest_first()]
    newest_first = [c.cube_id for c in r.iter_newest_first()]
    assert oldest_first == [2, 3, 4]
    assert newest_first == [4, 3, 2]


def test_pinned_buffers_are_independent() -> None:
    """Re-staging into the same ring slot must overwrite the buffer
    rather than aliasing the caller's source array."""
    r = CubeRetentionRing(
        depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False,
    )
    src = _make_payload(4, 2, 4, 1.0)
    rec0 = r.stage_cube(
        cube_id=0,
        event_specnum_start=0,
        mjd_start=58000.0,
        sample_period_specnum=16,
        sample_period_us=1048.576,
        cube_tensor=src,
    )
    # Mutate the caller's source; the ring's copy must not change.
    src[0, 0, 0, 0] = 999.0
    assert rec0.pinned_host_tensor[0, 0, 0, 0] == 1.0
