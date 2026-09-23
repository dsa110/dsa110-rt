"""CubeShmRing.device_copy_to_slot: GPU -> shm DMA, byte-exact.

The async-TX producer now DMAs each worker's DM slice straight from the
GPU into its shm slot (no host staging + memcpy on the corr pipeline
thread). The worker reads the slot through ``view_slot`` as before, so
the bytes it sees must be exactly the tensor's.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs CUDA", allow_module_level=True)

from dsart.transport.tx_ring import CubeShmRing, CubeShmRingDims  # noqa: E402


def _ring(shape, name):
    ctx = mp.get_context("spawn")
    return CubeShmRing(
        name=name,
        dims=CubeShmRingDims(n_slots=3, shape=shape, dtype=np.dtype("complex64")),
        ready_q=ctx.Queue(), done_q=ctx.Queue(), owner=True,
    )


def test_device_copy_is_byte_exact_and_slot_isolated():
    shape = (2, 128, 2203)
    ring = _ring(shape, f"dsart-test-devcopy-{np.random.randint(1 << 30)}")
    try:
        g = torch.Generator(device="cuda").manual_seed(5)
        a = torch.randn(shape, dtype=torch.complex64, device="cuda", generator=g)
        b = torch.randn(shape, dtype=torch.complex64, device="cuda", generator=g)
        assert ring.device_copy_to_slot(0, a) is True
        assert ring.device_copy_to_slot(2, b) is True
        np.testing.assert_array_equal(ring.view_slot(0), a.cpu().numpy())
        np.testing.assert_array_equal(ring.view_slot(2), b.cpu().numpy())
        # slot 1 untouched
        assert not np.any(ring.view_slot(1))
        # a strided slice of a bigger cube (what transmit passes) works
        big = torch.randn((8,) + shape[1:], dtype=torch.complex64, device="cuda")
        assert ring.device_copy_to_slot(1, big[4:6]) is True
        np.testing.assert_array_equal(ring.view_slot(1), big[4:6].cpu().numpy())
    finally:
        ring.close()          # must unregister before unmapping: no crash


def test_device_copy_rejects_wrong_shape():
    ring = _ring((2, 4, 8), f"dsart-test-devcopy-bad-{np.random.randint(1 << 30)}")
    try:
        with pytest.raises(ValueError, match="shape"):
            ring.device_copy_to_slot(
                0, torch.zeros((2, 4, 9), dtype=torch.complex64, device="cuda"))
    finally:
        ring.close()
