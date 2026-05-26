"""Acceptance tests for :mod:`dsart.coarse_dm.stage2_per_dm_fifo`.

Pins the per-coarse-DM FIFO bank that will replace the legacy
uniform-depth :class:`Stage2FIFO` on the corr_fast TX path once
Option A (corr-side stage-2 alignment) is wired in.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.coarse_dm.stage2_per_dm_fifo import PerCoarseDmStage2FIFO


def _slice(value: float, shape=(2,), dtype=torch.float32) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype)


def test_passthrough_for_zero_depth():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[0])
    s = _slice(1.0)
    out = fifo.push(0, s)
    assert out is s
    assert fifo.occupancy(0) == 0


def test_warmup_then_steady_state_depth1():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[1])
    s0 = _slice(0.0); s1 = _slice(1.0); s2 = _slice(2.0)
    # depth=1: first push warms up (no emit), then every push emits the
    # PREVIOUS one (1-sample delay).
    assert fifo.push(0, s0) is None
    out = fifo.push(0, s1)
    assert torch.equal(out, s0)
    out = fifo.push(0, s2)
    assert torch.equal(out, s1)


def test_warmup_then_steady_state_depth3():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[3])
    slices = [_slice(float(i)) for i in range(6)]
    # Warm-up: first 3 pushes don't emit.
    assert fifo.push(0, slices[0]) is None
    assert fifo.push(0, slices[1]) is None
    assert fifo.push(0, slices[2]) is None
    # Then every push emits slice[i - 3].
    assert torch.equal(fifo.push(0, slices[3]), slices[0])
    assert torch.equal(fifo.push(0, slices[4]), slices[1])
    assert torch.equal(fifo.push(0, slices[5]), slices[2])


def test_per_coarse_dm_independence():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[0, 2, 1])
    # c=0 passthrough
    s = _slice(0.5)
    out0 = fifo.push(0, s)
    assert out0 is s
    # c=1 depth=2 warm-up
    s1a = _slice(10.0); s1b = _slice(11.0); s1c = _slice(12.0)
    assert fifo.push(1, s1a) is None
    assert fifo.push(1, s1b) is None
    assert torch.equal(fifo.push(1, s1c), s1a)
    # c=2 depth=1 warm-up
    s2a = _slice(100.0); s2b = _slice(101.0)
    assert fifo.push(2, s2a) is None
    assert torch.equal(fifo.push(2, s2b), s2a)
    # c=0 still passthrough independently
    s_more = _slice(0.75)
    assert fifo.push(0, s_more) is s_more


def test_occupancy_and_len_track_state():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[2, 0, 3])
    assert len(fifo) == 0
    fifo.push(0, _slice(1.0))
    assert fifo.occupancy(0) == 1 and fifo.occupancy(2) == 0
    fifo.push(2, _slice(2.0))
    fifo.push(2, _slice(3.0))
    assert fifo.occupancy(2) == 2
    assert len(fifo) == 3
    # passthrough c=1 doesn't increase occupancy
    fifo.push(1, _slice(4.0))
    assert fifo.occupancy(1) == 0
    assert len(fifo) == 3


def test_rejects_shape_mismatch_within_one_coarse_dm():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[2])
    fifo.push(0, _slice(0.0, shape=(2,)))
    with pytest.raises(ValueError, match="slice shape"):
        fifo.push(0, _slice(0.0, shape=(3,)))


def test_rejects_dtype_mismatch_within_one_coarse_dm():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[2])
    fifo.push(0, _slice(0.0, dtype=torch.float32))
    with pytest.raises(ValueError, match="slice dtype"):
        fifo.push(0, _slice(0.0, dtype=torch.float16))


def test_rejects_out_of_range_coarse_dm():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[1, 1])
    with pytest.raises(ValueError, match="out of range"):
        fifo.push(-1, _slice(0.0))
    with pytest.raises(ValueError, match="out of range"):
        fifo.push(2, _slice(0.0))


def test_rejects_non_tensor_input():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[1])
    with pytest.raises(TypeError, match="torch.Tensor"):
        fifo.push(0, np.zeros(2))


def test_rejects_empty_depths():
    with pytest.raises(ValueError, match="non-empty"):
        PerCoarseDmStage2FIFO(depths_per_coarse_dm=[])


def test_rejects_negative_depth():
    with pytest.raises(ValueError, match=">= 0"):
        PerCoarseDmStage2FIFO(depths_per_coarse_dm=[1, -1, 0])


def test_flush_drains_oldest_first():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[3])
    slices = [_slice(float(i)) for i in range(3)]
    for s in slices:
        fifo.push(0, s)
    drained = fifo.flush(0)
    assert len(drained) == 3
    for d, expected in zip(drained, slices):
        assert torch.equal(d, expected)
    # After flush the coarse-DM is empty.
    assert fifo.occupancy(0) == 0
    # Warming up again works.
    assert fifo.push(0, _slice(99.0)) is None


def test_iter_yields_state_triples():
    fifo = PerCoarseDmStage2FIFO(depths_per_coarse_dm=[2, 0, 3])
    fifo.push(0, _slice(0.0))
    fifo.push(2, _slice(0.0))
    fifo.push(2, _slice(1.0))
    seen = list(fifo)
    assert seen == [(0, 1, 2), (1, 0, 0), (2, 2, 3)]
