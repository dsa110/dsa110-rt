"""Tests for M7.2 async-TX cube ring (``transport/tx_ring.py``).

Covers:
  (a) single-slot producer / consumer roundtrip
  (b) full-ring backpressure (producer blocks on reserve_slot)
  (c) header metadata round-trip (block_n, specnum, n_dm, flags)
  (d) wait_slot returns None on timeout (no published cubes)
  (e) poison-pill: signal_worker_exit → wait_slot returns None
  (f) shape / dtype mismatch on copy_to_slot raises ValueError
  (g) close is idempotent + unlinks shm (no leaks)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import uuid

import numpy as np
import pytest

from dsart.transport.tx_ring import (
    FLAG_RFI_WARMING_UP,
    CubeShmRing,
    CubeShmRingDims,
    TxRingBackpressureError,
    TxRingClosedError,
)


def _unique_name() -> str:
    return f"dsart-test-tx-{uuid.uuid4().hex[:12]}-pid{os.getpid()}"


def _make_dims(
    n_slots: int = 4,
    shape: tuple[int, ...] = (2, 16, 64),
) -> CubeShmRingDims:
    return CubeShmRingDims(
        n_slots=n_slots, shape=shape, dtype=np.dtype("complex64"),
    )


# ---------------------------------------------------------------------------
# (a) Basic roundtrip
# ---------------------------------------------------------------------------


def test_single_slot_roundtrip_same_process():
    """Producer + consumer in the SAME process exercise the shm + queue
    plumbing without subprocess overhead. Verifies the data path."""
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()

    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    consumer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=False)
    try:
        cube = np.arange(np.prod(dims.shape), dtype=np.complex64).reshape(dims.shape)
        cube.imag = 17.0

        slot_idx = producer.reserve_slot(timeout_s=0.5)
        producer.copy_to_slot(slot_idx, cube)
        producer.publish_slot(
            slot_idx,
            block_n=42, specnum=99,
            n_dm=dims.shape[0], n_fv=dims.shape[1], n_filled=dims.shape[2],
            rfi_warming_up=True,
        )

        meta = consumer.wait_slot(timeout_s=1.0)
        assert meta is not None
        assert meta.slot_idx == slot_idx
        assert meta.block_n == 42
        assert meta.specnum == 99
        assert meta.n_dm == dims.shape[0]
        assert meta.n_fv == dims.shape[1]
        assert meta.n_filled == dims.shape[2]
        assert meta.flags & FLAG_RFI_WARMING_UP
        assert meta.rfi_warming_up is True

        view = consumer.view_slot(meta.slot_idx)
        np.testing.assert_array_equal(view, cube)
        consumer.release_slot(meta.slot_idx)

        assert producer.n_publish == 1
        assert consumer.n_consume == 1
    finally:
        consumer.close()
        producer.close()


def test_multiple_blocks_in_order():
    dims = _make_dims(n_slots=4)
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    consumer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=False)
    try:
        for block in range(20):
            cube = np.full(dims.shape, block + 1j, dtype=np.complex64)
            slot = producer.reserve_slot(timeout_s=0.5)
            producer.copy_to_slot(slot, cube)
            producer.publish_slot(
                slot, block_n=block, specnum=block * 7,
                n_dm=dims.shape[0], n_fv=dims.shape[1], n_filled=dims.shape[2],
            )
            meta = consumer.wait_slot(timeout_s=1.0)
            assert meta is not None
            assert meta.block_n == block
            assert meta.specnum == block * 7
            assert consumer.view_slot(meta.slot_idx)[0, 0, 0] == complex(block, 1)
            consumer.release_slot(meta.slot_idx)
    finally:
        consumer.close()
        producer.close()


# ---------------------------------------------------------------------------
# (b) Backpressure
# ---------------------------------------------------------------------------


def test_backpressure_when_no_consumer():
    """With no consumer draining done_q, the producer fills all slots
    then blocks; reserve_slot must time out raising TxRingBackpressureError."""
    dims = _make_dims(n_slots=2)
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    try:
        # First 2 slots succeed.
        for i in range(2):
            slot = producer.reserve_slot(timeout_s=0.5)
            cube = np.zeros(dims.shape, dtype=np.complex64)
            producer.copy_to_slot(slot, cube)
            producer.publish_slot(
                slot, block_n=i, specnum=i,
                n_dm=dims.shape[0], n_fv=dims.shape[1], n_filled=dims.shape[2],
            )
        # 3rd reserve should time out (no consumer recycling).
        t0 = time.monotonic()
        with pytest.raises(TxRingBackpressureError):
            producer.reserve_slot(timeout_s=0.1)
        elapsed = time.monotonic() - t0
        # Should respect the timeout (not exceed it by much).
        assert 0.08 < elapsed < 0.5
        assert producer.n_backpressure >= 1
    finally:
        producer.close()


# ---------------------------------------------------------------------------
# (c) wait_slot timeout returns None (no torch / no data path)
# ---------------------------------------------------------------------------


def test_wait_slot_timeout_returns_none():
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    consumer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=False)
    try:
        t0 = time.monotonic()
        meta = consumer.wait_slot(timeout_s=0.2)
        assert meta is None
        assert 0.15 < (time.monotonic() - t0) < 0.5
    finally:
        consumer.close()
        producer.close()


# ---------------------------------------------------------------------------
# (d) Poison pill
# ---------------------------------------------------------------------------


def test_poison_pill():
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    consumer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=False)
    try:
        producer.signal_worker_exit()
        meta = consumer.wait_slot(timeout_s=1.0)
        assert meta is None
    finally:
        consumer.close()
        producer.close()


# ---------------------------------------------------------------------------
# (e) Shape / dtype enforcement
# ---------------------------------------------------------------------------


def test_copy_to_slot_shape_mismatch_raises():
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    try:
        slot = producer.reserve_slot(timeout_s=0.5)
        bad = np.zeros((3, 16, 64), dtype=np.complex64)
        with pytest.raises(ValueError, match="shape"):
            producer.copy_to_slot(slot, bad)
    finally:
        producer.close()


def test_copy_to_slot_dtype_mismatch_raises():
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    try:
        slot = producer.reserve_slot(timeout_s=0.5)
        bad = np.zeros(dims.shape, dtype=np.complex128)
        with pytest.raises(ValueError, match="dtype"):
            producer.copy_to_slot(slot, bad)
    finally:
        producer.close()


# ---------------------------------------------------------------------------
# (f) Lifecycle / close idempotency
# ---------------------------------------------------------------------------


def test_close_is_idempotent_and_raises_on_reuse():
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    producer.close()
    producer.close()  # idempotent
    with pytest.raises(TxRingClosedError):
        producer.reserve_slot(timeout_s=0.1)


def test_stats_snapshot_shape():
    dims = _make_dims()
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=8)
    done_q = ctx.Queue(maxsize=8)
    name = _unique_name()
    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    try:
        s = producer.stats()
        assert s["name"] == name
        assert s["n_slots"] == dims.n_slots
        assert s["n_publish"] == 0
        assert s["n_consume"] == 0
        assert s["free_slots"] == dims.n_slots
    finally:
        producer.close()


# ---------------------------------------------------------------------------
# (g) Subprocess roundtrip — the production scenario
# ---------------------------------------------------------------------------


def _subprocess_consumer(
    shm_name: str, dims_shape: tuple[int, ...], n_slots: int,
    ready_q: mp.Queue, done_q: mp.Queue, result_q: mp.Queue,
    n_expected: int,
) -> None:
    """Subprocess: attaches to producer's ring, reads n_expected cubes,
    posts checksums on result_q."""
    from dsart.transport.tx_ring import CubeShmRing, CubeShmRingDims
    dims = CubeShmRingDims(
        n_slots=n_slots, shape=dims_shape, dtype=np.dtype("complex64"),
    )
    ring = CubeShmRing(
        name=shm_name, dims=dims, ready_q=ready_q, done_q=done_q, owner=False,
    )
    checksums: list[tuple[int, complex]] = []
    try:
        for _ in range(n_expected):
            meta = ring.wait_slot(timeout_s=5.0)
            if meta is None:
                result_q.put(("error", "wait_slot timeout", len(checksums)))
                return
            view = ring.view_slot(meta.slot_idx)
            checksums.append((meta.block_n, complex(view.sum())))
            ring.release_slot(meta.slot_idx)
        result_q.put(("ok", checksums, ring.n_consume))
    finally:
        ring.close()


def test_subprocess_consumer_roundtrip():
    """End-to-end: producer in main process, consumer in subprocess.
    Verifies the shm + queue plumbing across process boundaries."""
    dims = _make_dims(n_slots=4, shape=(2, 16, 64))
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue(maxsize=16)
    done_q = ctx.Queue(maxsize=16)
    result_q = ctx.Queue(maxsize=4)
    name = _unique_name()

    producer = CubeShmRing(name, dims, ready_q=ready_q, done_q=done_q, owner=True)
    n_blocks = 32
    consumer_proc = ctx.Process(
        target=_subprocess_consumer,
        args=(name, dims.shape, dims.n_slots,
              ready_q, done_q, result_q, n_blocks),
    )
    consumer_proc.start()
    try:
        expected: list[tuple[int, complex]] = []
        for block in range(n_blocks):
            cube = np.full(dims.shape, block + 1j * block, dtype=np.complex64)
            expected.append((block, complex(cube.sum())))
            slot = producer.reserve_slot(timeout_s=2.0)
            producer.copy_to_slot(slot, cube)
            producer.publish_slot(
                slot, block_n=block, specnum=block,
                n_dm=dims.shape[0], n_fv=dims.shape[1], n_filled=dims.shape[2],
            )
        # Get consumer result
        try:
            status, payload, n_consume = result_q.get(timeout=10.0)
        except Exception:
            consumer_proc.terminate()
            raise AssertionError("subprocess consumer never posted result")
        assert status == "ok", f"consumer error: {payload!r}"
        assert n_consume == n_blocks
        assert payload == expected
    finally:
        producer.signal_worker_exit()
        consumer_proc.join(timeout=5.0)
        if consumer_proc.is_alive():
            consumer_proc.terminate()
            consumer_proc.join(timeout=2.0)
        producer.close()
