"""Tests for :mod:`dsart.dump.voltage_ring` (pure ring + specnum math)."""

from __future__ import annotations

import numpy as np
import pytest

from dsart.common.constants import BLOCK_SAMPLES_SPECNUM
from dsart.dump.voltage_ring import (
    VoltageRing,
    block_n_to_first_specnum,
    specnum_to_block_n,
    window_block_range,
)


def test_specnum_block_mapping_inverse() -> None:
    for block_n in (0, 1, 7, 1000):
        first = block_n_to_first_specnum(block_n)
        assert specnum_to_block_n(first) == block_n
        # Any specnum inside the block maps back to the same block.
        assert specnum_to_block_n(first + BLOCK_SAMPLES_SPECNUM - 1) == block_n


def test_specnum_negative_raises() -> None:
    with pytest.raises(ValueError):
        specnum_to_block_n(-1)


def test_window_block_range_clamped() -> None:
    assert list(window_block_range(10, 2, 3)) == [8, 9, 10, 11, 12, 13]
    # Clamps the low edge at 0.
    assert list(window_block_range(1, 5, 0)) == [0, 1]
    with pytest.raises(ValueError):
        window_block_range(10, -1, 0)


def _block(bpb: int, fill: int) -> np.ndarray:
    return np.full(bpb, fill & 0xFF, dtype=np.uint8)


def test_store_contains_extract() -> None:
    bpb = 8
    ring = VoltageRing(n_blocks=4, bytes_per_block=bpb)
    assert ring.newest_block_n == -1
    assert ring.oldest_block_n == -1
    for b in range(3):
        ring.store(b, _block(bpb, b + 1))
    assert ring.newest_block_n == 2
    assert ring.contains(0) and ring.contains(2)
    assert not ring.contains(3)

    win = ring.extract_window(1, n_pre=1, n_post=1)
    assert win.n_present == 3
    assert [b for b, _ in win.blocks] == [0, 1, 2]
    # Bytes are the stored fill (private copies).
    assert all(arr[0] == b + 1 for b, arr in win.blocks)
    assert win.first_block_n == 0 and win.last_block_n == 2


def test_rolloff_reports_dropped() -> None:
    bpb = 4
    ring = VoltageRing(n_blocks=3, bytes_per_block=bpb)
    for b in range(5):           # 0..4, ring holds 3 → 0,1 roll off
        ring.store(b, _block(bpb, b))
    assert ring.newest_block_n == 4
    assert ring.oldest_block_n == 2
    assert not ring.contains(1)
    assert ring.contains(2) and ring.contains(4)
    win = ring.extract_window(3, n_pre=3, n_post=1)
    present = [b for b, _ in win.blocks]
    assert present == [2, 3, 4]
    assert set(win.dropped) == {0, 1}


def test_copy_block_absent_returns_none() -> None:
    ring = VoltageRing(n_blocks=2, bytes_per_block=4)
    assert ring.copy_block(0) is None
    assert ring.copy_block(-5) is None
    ring.store(0, _block(4, 9))
    out = ring.copy_block(0)
    assert out is not None and out[0] == 9
    # mutating the returned copy must not corrupt the ring
    out[0] = 0
    assert ring.copy_block(0)[0] == 9


def test_store_wrong_size_raises() -> None:
    ring = VoltageRing(n_blocks=2, bytes_per_block=4)
    with pytest.raises(ValueError):
        ring.store(0, np.zeros(3, dtype=np.uint8))


def test_mon_fields() -> None:
    ring = VoltageRing(n_blocks=3, bytes_per_block=10)
    ring.store(0, _block(10, 1))
    m = ring.mon()
    assert m["n_blocks"] == 3
    assert m["bytes_per_block"] == 10
    assert m["n_stored"] == 1
    assert m["ram_bytes"] == 30
