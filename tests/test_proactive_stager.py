"""Unit tests for :mod:`dsart.dump.proactive_stager` (2026-07-21).

Covers the proactive bright-cube staging + claim protocol that converts
``too_late`` C2 dump misses into successful dumps:

  * staging fires at/above the SNR threshold and not below;
  * min-interval + LRU budget eviction;
  * TTL garbage collection;
  * claim-on-``too_late`` renames the staged NPZ into the event dir with
    the event-specnum filename and fires the uploader (mock);
  * live-ring hit path drops the redundant staged copy;
  * repeated / unmatched claims are idempotent no-ops.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from dsart.common.contracts import CubeDumpManifest
from dsart.dump.proactive_stager import (
    ProactiveCubeStager,
    ProactiveStagerConfig,
)

# Production-ish geometry knobs (tiny grid so the NPZ is cheap).
T_DET = 192
SAMPLE_PERIOD_SPECNUM = 16
N_FDM = 4
N_GRID = 8
# Specnum span of one cube. event_specnum and the cube anchor are both
# in SEARCH-sample units (detector/decoder.py:216,
# services/search_compute.py:1338-1345), so a cube covers exactly t_det
# of them. This was T_DET * SAMPLE_PERIOD_SPECNUM, matching the 16x-too-
# wide claim window in the stager, so the pair was self-consistent and
# neither could fail.
WINDOW = T_DET


class _FakeWriter:
    """Stand-in for :class:`CubeDumpWriter` that writes the NPZ
    synchronously to ``manifest.npz_path`` (so ``claim`` has a real file
    to rename). ``accept`` toggles the queue-full path."""

    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.submitted: List[CubeDumpManifest] = []

    def submit(
        self,
        *,
        cube: Any,
        manifest: CubeDumpManifest,
        on_complete=None,
    ) -> bool:
        self.submitted.append(manifest)
        if not self.accept:
            return False
        path = Path(manifest.npz_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(""), cube=np.asarray(cube, dtype=np.float16))
        # np.savez appends .npz when given a str without suffix; normalise.
        produced = path.with_suffix("").with_suffix(".npz")
        if produced != path and produced.exists():
            os.replace(produced, path)
        return True


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _make_stager(
    tmp_path: Path,
    *,
    config: Optional[ProactiveStagerConfig] = None,
    writer: Optional[_FakeWriter] = None,
    clock: Optional[_Clock] = None,
    uploads: Optional[List[Tuple[str, Path]]] = None,
    gpu_half: int = 0,
    search_node_id: int = 3,
) -> ProactiveCubeStager:
    cfg = config or ProactiveStagerConfig()
    up = uploads if uploads is not None else []
    return ProactiveCubeStager(
        cfg,
        dump_root=tmp_path,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        cube_dump_writer=writer or _FakeWriter(),
        upload_fn=lambda name, d: up.append((name, d)),
        time_now_s=clock or _Clock(),
    )


def _stage(
    stager: ProactiveCubeStager,
    *,
    cube_id: int,
    specnum_start: int,
    max_snr: float,
) -> bool:
    cube = np.zeros((T_DET, N_FDM, N_GRID, N_GRID), dtype=np.float16)
    return stager.maybe_stage(
        cube_id=cube_id,
        specnum_start=specnum_start,
        t_det=T_DET,
        sample_period_specnum=SAMPLE_PERIOD_SPECNUM,
        n_fdm_in_cube=N_FDM,
        n_grid=N_GRID,
        mjd_start=60000.0,
        max_snr=max_snr,
        cube_tensor=cube,
    )


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------


def test_stages_at_and_above_threshold(tmp_path: Path) -> None:
    stager = _make_stager(
        tmp_path, config=ProactiveStagerConfig(snr_threshold=50.0)
    )
    assert _stage(stager, cube_id=0, specnum_start=0, max_snr=50.0) is True
    assert stager.n_pending == 1
    assert (tmp_path / "pending_g0_0" / "cube_s3_g0_0.npz").exists()


def test_does_not_stage_below_threshold(tmp_path: Path) -> None:
    stager = _make_stager(
        tmp_path, config=ProactiveStagerConfig(snr_threshold=50.0)
    )
    assert _stage(stager, cube_id=0, specnum_start=0, max_snr=49.9) is False
    assert stager.n_pending == 0
    assert stager.mon["below_thresh"] == 1
    assert not list(tmp_path.glob("pending_*"))


def test_disabled_is_noop(tmp_path: Path) -> None:
    stager = _make_stager(
        tmp_path, config=ProactiveStagerConfig(enabled=False)
    )
    assert _stage(stager, cube_id=0, specnum_start=0, max_snr=999.0) is False
    assert stager.n_pending == 0


def test_submit_dropped_when_writer_queue_full(tmp_path: Path) -> None:
    writer = _FakeWriter(accept=False)
    stager = _make_stager(tmp_path, writer=writer)
    assert _stage(stager, cube_id=0, specnum_start=0, max_snr=80.0) is False
    assert stager.n_pending == 0
    assert stager.mon["submit_dropped"] == 1


# ---------------------------------------------------------------------------
# Rate protection
# ---------------------------------------------------------------------------


def test_min_interval_throttles(tmp_path: Path) -> None:
    clock = _Clock(1000.0)
    stager = _make_stager(
        tmp_path,
        config=ProactiveStagerConfig(min_interval_s=5.0),
        clock=clock,
    )
    assert _stage(stager, cube_id=0, specnum_start=0, max_snr=80.0) is True
    clock.t += 1.0  # < min_interval
    assert _stage(stager, cube_id=1, specnum_start=WINDOW, max_snr=80.0) is False
    assert stager.mon["rate_limited"] == 1
    clock.t += 5.0  # now past min_interval
    assert _stage(stager, cube_id=2, specnum_start=2 * WINDOW, max_snr=80.0) is True


def test_budget_evicts_oldest(tmp_path: Path) -> None:
    clock = _Clock(1000.0)
    stager = _make_stager(
        tmp_path,
        config=ProactiveStagerConfig(max_pending=2, min_interval_s=0.0),
        clock=clock,
    )
    for i in range(3):
        clock.t += 1.0
        assert _stage(
            stager, cube_id=i, specnum_start=i * WINDOW, max_snr=80.0
        ) is True
    # Only the two newest survive; the oldest (specnum 0) dir is gone.
    assert stager.n_pending == 2
    assert stager.mon["evicted"] == 1
    assert not (tmp_path / "pending_g0_0").exists()
    assert (tmp_path / f"pending_g0_{WINDOW}").exists()
    assert (tmp_path / f"pending_g0_{2 * WINDOW}").exists()


# ---------------------------------------------------------------------------
# TTL GC
# ---------------------------------------------------------------------------


def test_ttl_gc_removes_stale(tmp_path: Path) -> None:
    clock = _Clock(1000.0)
    stager = _make_stager(
        tmp_path,
        config=ProactiveStagerConfig(ttl_s=100.0, min_interval_s=0.0),
        clock=clock,
    )
    assert _stage(stager, cube_id=0, specnum_start=0, max_snr=80.0) is True
    assert (tmp_path / "pending_g0_0").exists()
    clock.t += 101.0
    removed = stager.gc()
    assert removed == 1
    assert stager.n_pending == 0
    assert not (tmp_path / "pending_g0_0").exists()


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


def test_claim_renames_and_uploads(tmp_path: Path) -> None:
    uploads: List[Tuple[str, Path]] = []
    stager = _make_stager(tmp_path, uploads=uploads)
    # Cube covers [1000, 1000 + WINDOW).
    _stage(stager, cube_id=7, specnum_start=1000, max_snr=101.8)
    event_specnum = 1000 + WINDOW // 2  # inside the window
    event_dir = stager.claim(
        event_specnum=event_specnum, event_name="260721upyy"
    )
    assert event_dir == tmp_path / "260721upyy"
    final = tmp_path / "260721upyy" / f"cube_s3_g0_{event_specnum}.npz"
    assert final.exists()
    # Provisional file gone; entry forgotten.
    assert not (tmp_path / "pending_g0_1000" / "cube_s3_g0_1000.npz").exists()
    assert stager.n_pending == 0
    assert stager.mon["claimed"] == 1
    # Uploader fired with the event name + dir.
    assert uploads == [("260721upyy", tmp_path / "260721upyy")]
    # NPZ is loadable + correctly named.
    with np.load(final) as npz:
        assert "cube" in npz


def test_claim_miss_for_uncovered_specnum(tmp_path: Path) -> None:
    stager = _make_stager(tmp_path)
    _stage(stager, cube_id=0, specnum_start=1000, max_snr=80.0)
    # Specnum below the staged window -> no match.
    assert stager.claim(event_specnum=10, event_name="ev") is None
    assert stager.mon["claim_miss"] == 1
    # Staged entry untouched.
    assert stager.n_pending == 1


def test_repeated_claim_is_idempotent(tmp_path: Path) -> None:
    uploads: List[Tuple[str, Path]] = []
    stager = _make_stager(tmp_path, uploads=uploads)
    _stage(stager, cube_id=0, specnum_start=1000, max_snr=80.0)
    ev = 1000 + 5
    assert stager.claim(event_specnum=ev, event_name="ev") is not None
    # Second claim: entry already forgotten -> no-op, no double upload.
    assert stager.claim(event_specnum=ev, event_name="ev") is None
    assert stager.mon["claimed"] == 1
    assert len(uploads) == 1


def test_claim_absent_npz_is_graceful(tmp_path: Path) -> None:
    """If the staged NPZ never made it to disk (writer dropped it), the
    claim fails gracefully rather than raising."""
    stager = _make_stager(tmp_path)
    _stage(stager, cube_id=0, specnum_start=1000, max_snr=80.0)
    # Simulate the file never landing.
    (tmp_path / "pending_g0_1000" / "cube_s3_g0_1000.npz").unlink()
    assert stager.claim(event_specnum=1005, event_name="ev") is None
    assert stager.mon["claim_miss"] == 1


# ---------------------------------------------------------------------------
# Live-ring precedence (drop_pending)
# ---------------------------------------------------------------------------


def test_drop_pending_removes_staged_copy(tmp_path: Path) -> None:
    stager = _make_stager(tmp_path)
    _stage(stager, cube_id=0, specnum_start=1000, max_snr=80.0)
    assert (tmp_path / "pending_g0_1000").exists()
    assert stager.drop_pending(1005) is True
    assert stager.n_pending == 0
    assert stager.mon["dropped_live"] == 1
    assert not (tmp_path / "pending_g0_1000").exists()
    # Dropping again is a no-op.
    assert stager.drop_pending(1005) is False


def test_per_half_dir_naming(tmp_path: Path) -> None:
    s0 = _make_stager(tmp_path, gpu_half=0)
    s1 = _make_stager(tmp_path, gpu_half=1)
    _stage(s0, cube_id=0, specnum_start=1000, max_snr=80.0)
    _stage(s1, cube_id=0, specnum_start=1000, max_snr=80.0)
    # Distinct per-half provisional dirs => no cross-half eviction race.
    assert (tmp_path / "pending_g0_1000").exists()
    assert (tmp_path / "pending_g1_1000").exists()
