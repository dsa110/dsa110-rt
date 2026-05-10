"""Tests for ``dsart.services.host_pin`` — CUDA host-memory registration.

Exercises:

  * Idempotent registration: re-registering the same buffer is O(1)
    no-op (no error), and ``is_registered`` reports correctly.
  * Non-numpy / non-contiguous inputs are silently skipped.
  * After registration, PyTorch reports ``is_pinned() == True`` on
    ``torch.from_numpy(arr)`` (GPU-only test, marks as skip when no CUDA).
  * H2D throughput improves measurably after registration (GPU only,
    relaxed threshold so flaky CI noise doesn't fail).
  * ``unpack_int4_split`` auto-pins the input buffer when called with
    a CUDA device; the buffer remains usable for subsequent calls
    (idempotency from the kernel's perspective).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DSART_TEST", "1")

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import FADA_BYTES_PER_BLOCK
from dsart.services.host_pin import (
    is_registered,
    maybe_register_host_buffer,
    n_registered,
    unregister_all,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty registry; clean up after itself."""
    unregister_all()
    yield
    unregister_all()


def test_non_numpy_skipped_silently():
    """Bytes / lists / non-contiguous views are silently skipped."""
    assert maybe_register_host_buffer(b"hello") is False
    assert maybe_register_host_buffer([1, 2, 3]) is False
    assert maybe_register_host_buffer(None) is False

    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    sliced = arr[::2]                          # non-contiguous
    assert sliced.flags["C_CONTIGUOUS"] is False
    assert maybe_register_host_buffer(sliced) is False
    assert n_registered() == 0


def test_zero_size_skipped():
    """Zero-byte buffers are skipped (no point pinning nothing)."""
    arr = np.empty((0,), dtype=np.uint8)
    assert maybe_register_host_buffer(arr) is False
    assert n_registered() == 0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="cudaHostRegister requires CUDA runtime + driver")
def test_idempotent_registration():
    """Registering the same buffer twice returns True both times, only
    one entry in the registry."""
    arr = np.zeros(1024 * 1024, dtype=np.uint8)
    assert is_registered(arr) is False
    assert maybe_register_host_buffer(arr) is True
    assert is_registered(arr) is True
    n_after_first = n_registered()
    assert n_after_first == 1

    # Second call: same buffer, same key — no-op.
    assert maybe_register_host_buffer(arr) is True
    assert n_registered() == n_after_first


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="pinned-memory check requires CUDA")
def test_register_makes_torch_see_pinned():
    """After registration, ``torch.from_numpy(arr).is_pinned()`` is True.

    This is the core invariant: PyTorch's H2D fast path keys off this
    flag to skip the pageable bounce buffer.
    """
    arr = np.zeros(2 * 1024 * 1024, dtype=np.uint8)
    t_before = torch.from_numpy(arr)
    assert t_before.is_pinned() is False     # baseline

    assert maybe_register_host_buffer(arr) is True

    t_after = torch.from_numpy(arr)
    assert t_after.is_pinned() is True


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="H2D timing requires CUDA")
def test_h2d_speedup_after_registration():
    """H2D copy is meaningfully faster after registration.

    Threshold is loose (1.5x speedup) because CI hardware varies, but
    the production gain on a 2080 Ti is ~2.6x (66 ms → 25 ms for the
    full 288 MB fada block). Anything < 1.5x suggests host-pin isn't
    actually engaging.
    """
    size = 64 * 1024 * 1024                  # 64 MB — long enough to dominate launch overhead
    arr_pageable = np.zeros(size, dtype=np.uint8)
    arr_pinned = np.zeros(size, dtype=np.uint8)

    def _bench(arr, n_iter=10):
        torch.cuda.synchronize()
        _ = torch.as_tensor(arr, device="cuda"); torch.cuda.synchronize()  # warmup
        ts = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t = torch.as_tensor(arr, device="cuda")
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
            del t
        return float(np.median(ts))

    pageable_ms = _bench(arr_pageable)

    assert maybe_register_host_buffer(arr_pinned) is True
    pinned_ms = _bench(arr_pinned)

    speedup = pageable_ms / pinned_ms
    assert speedup >= 1.5, (
        f"expected ≥1.5x H2D speedup after host-pin; got "
        f"pageable={pageable_ms:.2f} ms, pinned={pinned_ms:.2f} ms, "
        f"speedup={speedup:.2f}x"
    )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="unpack_int4_split GPU path requires CUDA")
def test_unpack_int4_split_auto_pins():
    """``unpack_int4_split`` registers its input on first GPU-targeted call."""
    from dsart.services.slow_corr_kernel import unpack_int4_split

    # Synthetic raw fada bytes (uint8); contents irrelevant for pinning.
    raw = np.zeros(FADA_BYTES_PER_BLOCK, dtype=np.uint8)
    assert is_registered(raw) is False

    real, imag = unpack_int4_split(raw, device="cuda")
    assert real.device.type == "cuda"
    assert imag.device.type == "cuda"
    assert is_registered(raw) is True

    # Second call on the same buffer is a registry hit — no error, still
    # produces correct output.
    real2, imag2 = unpack_int4_split(raw, device="cuda")
    torch.testing.assert_close(real, real2)
    torch.testing.assert_close(imag, imag2)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="GC unregister-on-finalize requires CUDA")
def test_finalizer_unregisters_on_gc():
    """When a registered numpy buffer is GC'd, the kernel's pinning is
    released BEFORE the OS munmaps the backing pages.

    Without this, the CUDA driver would hold a stale registration for
    a virtual-address range whose physical pages have been freed; a
    fresh allocation that lands at the recycled VA then fails
    ``cudaHostRegister`` (or worse, errors during async H2D with
    ``cudaErrorAlreadyMapped``). This is the bug that bit the
    pipelined fast-corr bench when each block allocated a fresh raw
    buffer.
    """
    import gc
    from dsart.services.host_pin import _FINALIZERS

    arr = np.zeros(8 * 1024 * 1024, dtype=np.uint8)   # 8 MB
    addr_size = (int(arr.ctypes.data), int(arr.nbytes))
    assert maybe_register_host_buffer(arr) is True
    assert addr_size in {tuple(k) for k in [addr_size]}  # sanity
    assert is_registered(arr) is True
    assert id(arr) in _FINALIZERS

    # Drop the only reference; GC must run the finalizer.
    del arr
    gc.collect()

    # After finalize, registry no longer holds the entry.
    with _registry_snapshot() as snap:
        assert addr_size not in snap, (
            f"finalizer did not run; registry still has {addr_size}, "
            f"all entries={snap}"
        )


from contextlib import contextmanager


@contextmanager
def _registry_snapshot():
    """Yield a snapshot of the current host-pin registry contents."""
    from dsart.services.host_pin import _REGISTERED, _REGISTRY_LOCK
    with _REGISTRY_LOCK:
        snap = set(_REGISTERED)
    yield snap


def test_unregister_all_clears_registry():
    """``unregister_all`` empties the registry even when CUDA isn't loaded."""
    arr = np.zeros(1024, dtype=np.uint8)
    maybe_register_host_buffer(arr)        # may be 0 or 1 depending on CUDA
    n_before = n_registered()
    unregister_all()
    assert n_registered() == 0
    # Re-register works after unregister.
    if torch.cuda.is_available():
        assert maybe_register_host_buffer(arr) is True
        assert n_registered() == 1
    else:
        # Without CUDA, registry is always empty.
        assert n_before == 0
