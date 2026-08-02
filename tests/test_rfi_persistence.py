"""Unit tests for the RFI flag time-persistence latch (M8.1).

CPU-only: the latch is pure elementwise torch, so these run anywhere
(the GPU op-point is exercised by the corr-node DoD, not here).
"""

from __future__ import annotations

import math

import pytest
import torch

from dsart.common.constants import BLOCK_DURATION_S
from dsart.rfi import FlagPersistence, RFIFlagger, seconds_to_cubes
from dsart.rfi.combine import FlagSourceBit
from dsart.rfi.autos import AutoSpectra


SHAPE = (4, 8, 2)  # tiny (ant, ch, pol) cube


def _mask(cells, value=True, shape=SHAPE):
    m = torch.zeros(shape, dtype=torch.bool)
    for c in cells:
        m[c] = value
    return m


# ---------------------------------------------------------------------------
# seconds_to_cubes
# ---------------------------------------------------------------------------


def test_seconds_to_cubes_matches_block_cadence():
    # 30 s / 134.218 ms = 223.5 -> 224 (ceil: the window must COVER 30 s).
    assert seconds_to_cubes(30.0) == 224
    # 900 s / 134.218 ms = 6705.3 -> 6706.
    assert seconds_to_cubes(900.0) == 6706
    assert seconds_to_cubes(0.0) == 0
    assert seconds_to_cubes(-1.0) == 0
    # Exactly one block is one cube.
    assert seconds_to_cubes(BLOCK_DURATION_S) == 1


# ---------------------------------------------------------------------------
# Run-counter path (latch_frac == 1.0)
# ---------------------------------------------------------------------------


def test_latches_only_after_full_window():
    p = FlagPersistence(latch_window_cubes=4, hold_cubes=3)
    hot = _mask([(0, 0, 0)])
    for i in range(3):
        out, stats = p.update(hot)
        assert not out.any(), f"latched early at cube {i}"
        assert int(stats.n_new_latched) == 0
    out, stats = p.update(hot)
    assert bool(out[0, 0, 0])
    assert int(stats.n_new_latched) == 1
    assert int(stats.n_latched) == 1
    # Nothing else latched.
    assert int(out.sum()) == 1


def test_hold_expires_after_exactly_hold_cubes():
    p = FlagPersistence(latch_window_cubes=2, hold_cubes=3)
    hot = _mask([(1, 2, 1)])
    clean = _mask([])
    p.update(hot)
    out, _ = p.update(hot)
    assert bool(out[1, 2, 1])          # latched
    # Detectors let go: held for exactly `hold_cubes` more cubes.
    for i in range(3):
        out, _ = p.update(clean)
        assert bool(out[1, 2, 1]), f"expired early at hold cube {i}"
    out, stats = p.update(clean)
    assert not out.any()
    assert int(stats.n_latched) == 0


def test_single_gap_resets_the_strict_run():
    """frac == 1.0 means *every* cube: one clean cube restarts the run."""
    p = FlagPersistence(latch_window_cubes=4, hold_cubes=5)
    hot = _mask([(0, 0, 0)])
    clean = _mask([])
    for _ in range(3):
        p.update(hot)
    out, _ = p.update(clean)               # gap on cube 4
    assert not out.any()
    for _ in range(3):
        out, _ = p.update(hot)
        assert not out.any()               # run restarted from 0
    out, _ = p.update(hot)
    assert bool(out[0, 0, 0])              # 4 consecutive again -> latched


def test_continuous_rfi_keeps_topping_up_the_hold():
    p = FlagPersistence(latch_window_cubes=2, hold_cubes=2)
    hot = _mask([(0, 1, 0)])
    clean = _mask([])
    for _ in range(20):                    # far longer than hold_cubes
        out, _ = p.update(hot)
    assert bool(out[0, 1, 0])
    # Clock only starts once the detectors stop.
    for _ in range(2):
        out, _ = p.update(clean)
        assert bool(out[0, 1, 0])
    out, _ = p.update(clean)
    assert not out.any()


def test_run_counter_path_allocates_no_ring():
    p = FlagPersistence(latch_window_cubes=224, hold_cubes=6705)
    assert not p.uses_ring
    p.update(_mask([]))
    # 2 int32 tensors over the cube: no [W, ...] ring.
    assert p.state_bytes == 2 * math.prod(SHAPE) * 4


def test_disabled_when_window_or_hold_is_zero():
    for kwargs in ({"latch_window_cubes": 0, "hold_cubes": 10},
                   {"latch_window_cubes": 10, "hold_cubes": 0}):
        p = FlagPersistence(**kwargs)
        assert not p.enabled
        out, stats = p.update(_mask([(0, 0, 0)]))
        assert not out.any()
        assert int(stats.n_latched) == 0
        assert p.state_bytes == 0


@pytest.mark.parametrize("bad", [
    {"latch_window_cubes": -1, "hold_cubes": 1},
    {"latch_window_cubes": 1, "hold_cubes": -1},
    {"latch_window_cubes": 1, "hold_cubes": 1, "latch_frac": 0.0},
    {"latch_window_cubes": 1, "hold_cubes": 1, "latch_frac": 1.5},
])
def test_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        FlagPersistence(**bad)


def test_rejects_non_bool_mask():
    p = FlagPersistence(latch_window_cubes=2, hold_cubes=2)
    with pytest.raises(ValueError):
        p.update(torch.zeros(SHAPE, dtype=torch.uint8))


def test_reset_drops_every_latch():
    p = FlagPersistence(latch_window_cubes=1, hold_cubes=50)
    out, _ = p.update(_mask([(0, 0, 0)]))
    assert out.any()
    p.reset()
    assert p.state_bytes == 0
    out, _ = p.update(_mask([]))
    assert not out.any()


# ---------------------------------------------------------------------------
# Ring path (latch_frac < 1.0)
# ---------------------------------------------------------------------------


def test_ring_path_tolerates_gaps():
    # 3 of the last 4 cubes flagged is enough.
    p = FlagPersistence(latch_window_cubes=4, hold_cubes=2, latch_frac=0.75)
    assert p.uses_ring
    hot = _mask([(0, 0, 0)])
    clean = _mask([])
    seq = [hot, clean, hot, hot]           # 3/4 within the window
    for i, m in enumerate(seq[:-1]):
        out, _ = p.update(m)
        assert not out.any(), f"latched early at cube {i}"
    out, _ = p.update(seq[-1])
    assert bool(out[0, 0, 0])


def test_ring_path_never_latches_on_a_partial_window():
    p = FlagPersistence(latch_window_cubes=5, hold_cubes=2, latch_frac=0.2)
    # 1/5 would already clear the threshold, but the window isn't full.
    out, _ = p.update(_mask([(0, 0, 0)]))
    assert not out.any()


def test_ring_count_slides_out_old_cubes():
    p = FlagPersistence(latch_window_cubes=3, hold_cubes=1, latch_frac=1.0 - 1e-9)
    hot = _mask([(0, 0, 0)])
    clean = _mask([])
    for _ in range(3):
        p.update(hot)                      # latched
    # Three clean cubes push every flagged sample out of the window.
    for _ in range(3):
        p.update(clean)
    out, _ = p.update(clean)
    assert not out.any()


def test_ring_state_bytes_is_the_documented_size():
    p = FlagPersistence(latch_window_cubes=6, hold_cubes=2, latch_frac=0.5)
    p.update(_mask([]))
    cells = math.prod(SHAPE)
    expected = (
        cells * 4            # hold  int32
        + 6 * cells          # ring  uint8 [W, ...]
        + cells * 2          # count int16
    )
    assert p.state_bytes == expected


# ---------------------------------------------------------------------------
# RFIFlagger integration
# ---------------------------------------------------------------------------


#: Single-M autos so `combine` reads s1[M].squeeze(0) directly and the
#: SK MC threshold table is exercised at a real production depth.
_M = 4096


def _autos_for(shape=SHAPE, s1_value=1.0):
    """AutoSpectra sitting exactly at the Gaussian SK expectation.

    ``SK = ((M+1)/(M-1)) * (M*S2/S1^2 - 1)``; with ``S2 = 2*S1^2/M`` this
    is 1.0 to within 1/M, i.e. dead centre of the acceptance band, so no
    detector fires and the only thing that can flag a cell is the
    persistence latch (or flagants.dat).
    """
    n_ant, n_ch, n_pol = shape
    s1 = torch.full((1, n_ant, n_ch, n_pol), s1_value * _M)
    s2 = torch.full((1, n_ant, n_ch, n_pol), 2.0 * s1_value * s1_value * _M)
    return AutoSpectra(s1={_M: s1}, s2={_M: s2})


def test_flagger_without_persistence_is_unchanged():
    f = RFIFlagger(flagants_path=None, warmup_cubes=0, m_values=(_M,))
    assert f.persistence is None
    res = f.flag_block(None, None, autos_override=_autos_for(SHAPE))
    assert res.n_persist_latched == 0
    assert res.n_persist_new == 0
    assert not (res.source_tags & int(FlagSourceBit.PERSISTENCE)).any()


def test_flagger_tags_persisted_cells_and_holds_them():
    """A cell the detectors stop flagging stays in the final mask."""
    f = RFIFlagger(
        flagants_path=None,
        warmup_cubes=0,
        m_values=(_M,),
        run_sum_threshold=False,
        persistence=FlagPersistence(latch_window_cubes=2, hold_cubes=2),
    )
    # Drive the detectors deterministically by monkey-patching the
    # combine step's inputs: easier to make a hot cell via group-outlier
    # than to hand-craft SK statistics, so instead we drive the latch
    # directly and assert combine's plumbing.
    hot = _mask([(0, 0, 0)])
    pers = f.persistence
    assert pers is not None
    pers.update(hot)
    out, _ = pers.update(hot)
    assert bool(out[0, 0, 0])

    # Now a fully clean cube through the real flagger: the latch must
    # still contribute the cell to `mask` and set bit 5 in the tags.
    res = f.flag_block(None, None, autos_override=_autos_for(SHAPE))
    assert bool(res.mask[0, 0, 0])
    assert bool(
        res.source_tags[0, 0, 0] & int(FlagSourceBit.PERSISTENCE)
    )
    assert res.n_persist_latched == 1
    assert res.flag_fraction_total == pytest.approx(
        1.0 / math.prod(SHAPE), rel=1e-6
    )


def test_reset_persistence_clears_the_latch():
    f = RFIFlagger(
        flagants_path=None,
        warmup_cubes=0,
        m_values=(_M,),
        persistence=FlagPersistence(latch_window_cubes=1, hold_cubes=99),
    )
    assert f.persistence is not None
    f.persistence.update(_mask([(0, 0, 0)]))
    f.reset_persistence()
    res = f.flag_block(None, None, autos_override=_autos_for(SHAPE))
    assert res.n_persist_latched == 0


def test_flagants_do_not_feed_the_latch(tmp_path):
    """Statically flagged antennas must not consume latch state."""
    fa = tmp_path / "flagants.dat"
    fa.write_text("0\n")
    f = RFIFlagger(
        flagants_path=fa,
        warmup_cubes=0,
        m_values=(_M,),
        run_sum_threshold=False,
        persistence=FlagPersistence(latch_window_cubes=1, hold_cubes=99),
    )
    autos = _autos_for(SHAPE)
    for _ in range(3):
        res = f.flag_block(None, None, autos_override=autos)
    # ant 0 is flagged by flagants.dat on every cube, but the latch was
    # never fed it, so nothing is held.
    assert res.n_persist_latched == 0
    assert bool(res.mask[0].all())
    assert not (res.source_tags & int(FlagSourceBit.PERSISTENCE)).any()
