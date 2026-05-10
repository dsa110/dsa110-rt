"""CUDA host-memory registration helper for fast PSRDADA → GPU H2D.

Background
==========
A PSRDADA shared-memory ringbuffer page is regular pageable host memory
from the kernel's perspective. When PyTorch does ``torch.as_tensor(arr,
device='cuda')`` on a numpy view of a pageable buffer, the CUDA driver
must first stage the contents through an internal pinned bounce buffer
(one extra CPU memcpy in the way). On the 2080 Ti this caps the H2D
throughput at ~4-5 GB/s — about 66 ms for a 288 MB ``fada`` block.

If the same buffer is registered with ``cudaHostRegister`` once, the
driver knows it's page-locked and can DMA straight from the buffer to
GPU memory at ~12 GB/s (PCIe Gen3 x16 effective), dropping the H2D to
~25 ms — a **2.6× speedup** with zero algorithmic change.

This is exactly what the legacy ``dsaX_bfCorr.cu`` does in
``dada_cuda_dbregister`` (registers all PSRDADA buffer pages once at
service startup). We provide the same primitive here for the python
services (``corr_slow_compute`` and ``corr_fast_compute``).

Usage
=====
The high-level entry-point is :func:`maybe_register_host_buffer`, which
is **idempotent** — call it freely on every PSRDADA page; it
short-circuits after the first registration of a given (addr, size)
pair, so the steady-state cost is a single dict lookup. The expected
call site is inside :func:`dsart.services.slow_corr_kernel.unpack_int4_split`,
which both fast and slow correlator services route their H2D through.

The legacy bfCorr path registers all ``db->buffer[ibuf]`` pages eagerly
at startup via direct ``ipcbuf_t`` access. The python ``psrdada-python``
binding doesn't expose that handle, so we do lazy registration on first
encounter — after one full pass through the PSRDADA ring (typically
4-8 buffer pages) every page is registered and the path is steady-state
fast.

Failure modes are non-fatal: if ``libcudart.so`` can't be loaded (CPU-only
runs, dev environment), or ``cudaHostRegister`` returns an error
(unmapped memory, already registered by something else, device without
unified addressing), we silently fall back to the pageable path so
correctness is preserved. Diagnostics are logged at DEBUG level only.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import weakref
from typing import Final

import numpy as np

LOG = logging.getLogger(__name__)

# Escape hatch: setting ``DSART_DISABLE_HOST_PIN=1`` in the environment
# makes :func:`maybe_register_host_buffer` a no-op. Useful for A/B
# benchmarking against the pageable baseline and for diagnosing
# pinning-related regressions in unfamiliar deployments. Read once at
# import time so the steady-state cost is one boolean check per call.
_DISABLED: Final[bool] = bool(int(os.environ.get("DSART_DISABLE_HOST_PIN", "0")))
if _DISABLED:
    LOG.info("host-pin disabled via DSART_DISABLE_HOST_PIN=1")


# ---------------------------------------------------------------------------
# CUDA runtime binding (lazy, optional)
# ---------------------------------------------------------------------------


_CUDART_LOCK = threading.Lock()
_CUDART: ctypes.CDLL | None = None
_CUDART_PROBED = False

# cudaHostRegister flags: 0 = "default" (legacy bfCorr value).
_CUDA_HOST_REGISTER_DEFAULT: Final[int] = 0

# CUDA error codes we treat as benign (already-registered, etc.) so we
# don't spam logs when PSRDADA recycles a previously-seen page.
_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: Final[int] = 712


def _get_cudart() -> ctypes.CDLL | None:
    """Lazy-load ``libcudart.so`` and bind the host-register signatures.

    Returns ``None`` (cached) if the library can't be loaded; subsequent
    calls are then no-ops at the cost of one boolean check.
    """
    global _CUDART, _CUDART_PROBED
    if _CUDART_PROBED:
        return _CUDART
    with _CUDART_LOCK:
        if _CUDART_PROBED:                 # double-checked locking
            return _CUDART
        try:
            lib = ctypes.CDLL("libcudart.so")
            lib.cudaHostRegister.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
            ]
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            lib.cudaHostUnregister.restype = ctypes.c_int
            _CUDART = lib
        except OSError as exc:
            LOG.debug("libcudart.so not loadable, host-pin disabled: %s", exc)
            _CUDART = None
        _CUDART_PROBED = True
    return _CUDART


# ---------------------------------------------------------------------------
# Idempotent host-pin registry
# ---------------------------------------------------------------------------


_REGISTRY_LOCK = threading.Lock()
_REGISTERED: set[tuple[int, int]] = set()

# Active weakref finalizers, keyed by id(base_buffer). Holding a
# strong reference to the finalizer object keeps it alive until the
# underlying numpy base array is GC'd, at which point the finalizer
# fires and calls cudaHostUnregister so we don't leave the CUDA
# driver tracking pinned regions backed by munmap'd memory.
_FINALIZERS: dict[int, "weakref.finalize"] = {}


def _numpy_base(arr: np.ndarray) -> np.ndarray:
    """Walk ``.base`` chain to find the array that owns the memory."""
    base = arr
    while base.base is not None and isinstance(base.base, np.ndarray):
        base = base.base
    return base


def _make_unregister_finalizer(addr: int, size: int) -> None:
    """Schedule cudaHostUnregister(addr) when the owning python object dies.

    Bound to the numpy base array via :func:`weakref.finalize`. If the
    base array is GC'd, the finalizer:
      1. calls ``cudaHostUnregister(addr)`` to release the page-lock
         BEFORE numpy's underlying ``munmap`` happens,
      2. removes ``(addr, size)`` from the bookkeeping registry so
         subsequent diagnostic counts stay accurate.
    The CUDA driver call is best-effort: if it fails (e.g. the runtime
    is in shutdown), we log at DEBUG only.
    """
    cudart = _get_cudart()
    if cudart is None:
        return

    def _on_finalize() -> None:
        rval = cudart.cudaHostUnregister(ctypes.c_void_p(addr))
        if rval != 0:
            LOG.debug(
                "host-unpin (finalizer): cudaHostUnregister(0x%x) → rval=%d",
                addr, rval,
            )
        with _REGISTRY_LOCK:
            _REGISTERED.discard((addr, size))
            _FINALIZERS.pop(addr, None)

    return _on_finalize


def _buffer_address_and_size(buf: object) -> tuple[int, int] | None:
    """Return (host_address, byte_count) for a numpy/memoryview-like host buffer.

    Returns ``None`` when the buffer doesn't expose a stable host address
    (e.g. zero-copy GPU tensors, non-contiguous numpy views) — the caller
    should silently skip in that case rather than failing.
    """
    if isinstance(buf, np.ndarray):
        if not buf.flags["C_CONTIGUOUS"] and not buf.flags["F_CONTIGUOUS"]:
            return None
        return int(buf.ctypes.data), int(buf.nbytes)
    if isinstance(buf, memoryview):
        if not buf.contiguous:
            return None
        # memoryview → underlying buffer pointer + size via numpy view
        arr = np.asarray(buf)
        if not arr.flags["C_CONTIGUOUS"]:
            return None
        return int(arr.ctypes.data), int(arr.nbytes)
    return None


def maybe_register_host_buffer(buf: object) -> bool:
    """Register a host buffer with CUDA so PyTorch H2D copies hit the pinned path.

    The CUDA driver is treated as the source of truth: every call goes
    through ``cudaHostRegister``. The driver short-circuits on a
    re-pin of the same physical page (returning
    ``cudaErrorHostMemoryAlreadyRegistered=712`` in microseconds), so
    the steady-state cost on the hot path is one ctypes round-trip
    (sub-microsecond on the M2/M3 services). Re-registering correctly
    handles the stale-VA case where a numpy buffer is GC'd and the
    OS hands its virtual-address range to a fresh allocation — the
    new physical page is re-registered, and the registry isn't
    silently lying about a buffer that no longer exists.

    The ``_REGISTERED`` set is kept as a *bookkeeping aid only* — it
    powers :func:`unregister_all` for clean shutdown / test teardown
    and the diagnostic count in :func:`n_registered`. It never
    influences whether the actual register call is made.

    Parameters
    ----------
    buf : np.ndarray | memoryview | object exposing a host pointer
        Host buffer to register. Typically the result of
        ``np.asarray(reader.getNextPage())`` from PSRDADA.

    Returns
    -------
    bool
        True if the buffer is currently page-locked (either freshly
        registered by this call or already pinned). False if
        registration was skipped or failed (no CUDA runtime, non-
        contiguous buffer, or ``cudaHostRegister`` returned an error
        the driver doesn't categorise as "already registered").

    Notes
    -----
    Mirrors ``dada_cuda_dbregister`` in ``dsaX_bfCorr.cu`` (lines 92-121).
    bfCorr eagerly registers all ``db->buffer[ibuf]`` pages at startup
    via direct ``ipcbuf_t`` access; the python ``psrdada-python`` binding
    doesn't expose that handle, so we do lazy on-first-encounter
    registration here. After one full pass through the PSRDADA ring
    (typically 4-8 buffer pages) every page is registered and the H2D
    path is in steady state.
    """
    if _DISABLED:
        return False

    addr_size = _buffer_address_and_size(buf)
    if addr_size is None:
        return False
    addr, size = addr_size
    if size == 0:
        return False

    cudart = _get_cudart()
    if cudart is None:
        return False

    rval = cudart.cudaHostRegister(
        ctypes.c_void_p(addr), ctypes.c_size_t(size),
        ctypes.c_uint(_CUDA_HOST_REGISTER_DEFAULT),
    )
    if rval == 0:
        # Newly registered; record + arm a weakref finalizer on the
        # underlying numpy base so we cudaHostUnregister BEFORE the OS
        # munmaps the memory. Without this, GC'ing a pinned numpy
        # array leaves the CUDA driver tracking a page-locked region
        # whose backing pages have been freed — which manifests later
        # as "resource already mapped" / async copy errors when a
        # fresh allocation lands at the recycled VA and we try to
        # re-register.
        with _REGISTRY_LOCK:
            _REGISTERED.add((addr, size))
            n = len(_REGISTERED)
        if isinstance(buf, np.ndarray):
            base = _numpy_base(buf)
            base_id = id(base)
            with _REGISTRY_LOCK:
                if base_id not in _FINALIZERS:
                    cb = _make_unregister_finalizer(addr, size)
                    if cb is not None:
                        _FINALIZERS[base_id] = weakref.finalize(base, cb)
        LOG.info(
            "host-pinned PSRDADA buffer page: addr=0x%x size=%d MB (now %d page(s) pinned)",
            addr, size // (1024 * 1024), n,
        )
        return True
    if rval == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        # Steady-state hot path. Idempotent re-register.
        with _REGISTRY_LOCK:
            _REGISTERED.add((addr, size))
        return True
    LOG.debug(
        "host-pin: cudaHostRegister(addr=0x%x size=%d) failed rval=%d; "
        "falling back to pageable H2D",
        addr, size, rval,
    )
    return False


def is_registered(buf: object) -> bool:
    """Whether ``buf`` is recorded in the local bookkeeping registry.

    Returns True iff a previous :func:`maybe_register_host_buffer` call
    on this exact (addr, size) returned True. Note that the registry
    is bookkeeping only; the CUDA driver remains the authoritative
    source for pinning state. This helper exists for tests + the
    diagnostic count.
    """
    addr_size = _buffer_address_and_size(buf)
    if addr_size is None:
        return False
    with _REGISTRY_LOCK:
        return addr_size in _REGISTERED


def unregister_all() -> int:
    """Unregister every host buffer recorded in the registry.

    Returns the number of buffers actually unregistered. Best-effort:
    individual ``cudaHostUnregister`` failures are logged at DEBUG and
    counted as skipped. Active weakref finalizers are detached so they
    don't fire later on already-unregistered addresses.

    Intended for clean shutdown / test teardown only — the OS will
    clean up automatically on process exit, and per-buffer finalizers
    handle the per-array case automatically.
    """
    cudart = _get_cudart()
    with _REGISTRY_LOCK:
        # Detach per-buffer finalizers first so they don't double-call
        # cudaHostUnregister on the addresses we're about to free here.
        for fin in _FINALIZERS.values():
            fin.detach()
        _FINALIZERS.clear()
        registered = list(_REGISTERED)
        _REGISTERED.clear()
    if cudart is None:
        return 0
    n = 0
    for addr, _size in registered:
        rval = cudart.cudaHostUnregister(ctypes.c_void_p(addr))
        if rval == 0:
            n += 1
        else:
            LOG.debug("host-unpin: cudaHostUnregister(0x%x) → rval=%d",
                      addr, rval)
    return n


def n_registered() -> int:
    """Number of host buffers currently held in the registry."""
    with _REGISTRY_LOCK:
        return len(_REGISTERED)
