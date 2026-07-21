"""C2TriggerListener x ProactiveCubeStager wiring (2026-07-21).

Verifies the listener's claim/drop hooks fire in the right branches:

  * ``too_late`` miss -> ``stager.claim`` is called; a successful claim
    bumps the ``rescued`` mon-point;
  * live-ring ``hit`` -> the live dump path runs unchanged AND
    ``stager.drop_pending`` discards the redundant staged copy;
  * with no stager wired the listener behaves exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dsart.coinc.wire import C2_TRIGGER_FLAG_DUMP_CUBE, C2TriggerPacket
from dsart.dump.c2_trigger_listener import (
    C2TriggerListener,
    C2TriggerListenerConfig,
)
from dsart.services.cube_pipeline import CubeRetentionRing


class _FakeStager:
    def __init__(self, claim_dir: Optional[Path] = None) -> None:
        self.claim_calls: List[Tuple[int, str]] = []
        self.drop_calls: List[int] = []
        self._claim_dir = claim_dir

    def claim(self, *, event_specnum: int, event_name: str):
        self.claim_calls.append((int(event_specnum), str(event_name)))
        return self._claim_dir

    def drop_pending(self, event_specnum: int) -> bool:
        self.drop_calls.append(int(event_specnum))
        return True


def _cfg(tmp_path: Path) -> C2TriggerListenerConfig:
    return C2TriggerListenerConfig(
        bind_host="127.0.0.1",
        base_port=12000,
        gpu_half=0,
        search_node_id=1,
        dump_root=tmp_path,
    )


def _packet(event_name: str, event_specnum: int) -> C2TriggerPacket:
    return C2TriggerPacket(
        event_name=event_name,
        event_specnum=event_specnum,
        mjd_target=58000.0,
        trigger_class_id=7,
        flags=C2_TRIGGER_FLAG_DUMP_CUBE,
    )


def _stage(ring: CubeRetentionRing, *, cube_id: int, specnum: int) -> None:
    ring.stage_cube(
        cube_id=cube_id,
        event_specnum_start=specnum,
        mjd_start=58000.0,
        sample_period_specnum=16,
        sample_period_us=1048.576,
        cube_tensor=np.full((4, 2, 4, 4), float(cube_id), dtype=np.float16),
    )


def test_too_late_triggers_claim_and_rescue(tmp_path: Path) -> None:
    ring = CubeRetentionRing(depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False)
    # Oldest retained window starts at 2000; a trigger at 1000 is too_late.
    _stage(ring, cube_id=0, specnum=2000)
    stager = _FakeStager(claim_dir=tmp_path / "260721upyy")
    listener = C2TriggerListener(config=_cfg(tmp_path), ring=ring, stager=stager)
    listener._handle_trigger(_packet("260721upyy", 1000), ("10.0.0.1", 5))
    assert stager.claim_calls == [(1000, "260721upyy")]
    assert stager.drop_calls == []
    mon = listener.mon
    assert mon["too_late"] == 1
    assert mon["rescued"] == 1


def test_too_late_claim_miss_no_rescue(tmp_path: Path) -> None:
    ring = CubeRetentionRing(depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False)
    _stage(ring, cube_id=0, specnum=2000)
    stager = _FakeStager(claim_dir=None)  # nothing staged for it
    listener = C2TriggerListener(config=_cfg(tmp_path), ring=ring, stager=stager)
    listener._handle_trigger(_packet("ev", 1000), ("10.0.0.1", 5))
    assert stager.claim_calls == [(1000, "ev")]
    mon = listener.mon
    assert mon["too_late"] == 1
    assert mon["rescued"] == 0


def test_hit_drops_pending_and_dispatches(tmp_path: Path) -> None:
    ring = CubeRetentionRing(depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False)
    _stage(ring, cube_id=0, specnum=1000)  # window [1000, 1064)
    stager = _FakeStager()
    dispatched: List[int] = []
    listener = C2TriggerListener(
        config=_cfg(tmp_path),
        ring=ring,
        stager=stager,
        dispatcher=lambda r, p, m: (dispatched.append(int(p.event_specnum)), True)[1],
    )
    listener._handle_trigger(_packet("ev", 1010), ("10.0.0.1", 5))
    assert dispatched == [1010]
    assert stager.drop_calls == [1010]
    assert stager.claim_calls == []
    mon = listener.mon
    assert mon["hits"] == 1
    assert mon["dispatched"] == 1


def test_no_stager_is_unchanged(tmp_path: Path) -> None:
    ring = CubeRetentionRing(depth=2, t_det=4, n_fdm=2, n_grid=4, pinned=False)
    _stage(ring, cube_id=0, specnum=2000)
    listener = C2TriggerListener(config=_cfg(tmp_path), ring=ring)
    # Must not raise with no stager wired.
    listener._handle_trigger(_packet("ev", 1000), ("10.0.0.1", 5))
    assert listener.mon["too_late"] == 1
    assert listener.mon["rescued"] == 0
