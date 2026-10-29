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

# x2 G0 finding (2026-10-29): today's finalizer is bound to the numpy wrapper the caller
# passes in. For a PSRDADA page that is the per-block ``np.asarray(reader.getNextPage())``
# (its base is the memoryview, where ``_numpy_base`` stops), so a page's pin lives only as long
# as one block's wrapper: when the wrapper dies the page is unregistered and the next visit
# pays ``cudaHostRegister`` again (~30 ms for 288 MB on a corr node); a live wrapper of an
# older visit instead triggers the force-clean re-register. Whether either fires depends on
# CPython id() reuse, because ``_FINALIZERS`` is keyed by id(base) but the finalizer removes
# by address. ``DSART_HOST_PIN_PERSIST=1`` registers each (address, size) of memory numpy does
# not own (a PSRDADA ring page: a stable shm mapping) once for the life of the process and never
# unregisters it; steady state is one set lookup per block. numpy-owned buffers keep the
# finalizer path. Off by default: today's behaviour is unchanged. Measured 2026-10-29 on n04
# (1x legacy, production argv, replayed voltages, 24-page ring, 600 blocks): 30 fresh
# registrations at 30.3 ms each (908 ms), 6 finalizer unregisters and 2 force-cleans.
_PERSIST: Final[bool] = bool(int(os.environ.get("DSART_HOST_PIN_PERSIST", "0")))
if _PERSIST:
    LOG.info("host-pin: persistent page registration (DSART_HOST_PIN_PERSIST=1)")


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
            lib.cudaGetLastError.argtypes = []
            lib.cudaGetLastError.restype = ctypes.c_int
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

#: diagnostic counters (x2 per-block trace): fresh cudaHostRegister calls and their
#: wall time, finalizer-driven unregisters, force-clean re-registrations
_STATS: dict = {"register_new": 0, "register_ms": 0.0, "finalizer_unregister": 0,
                "force_clean": 0, "skipped_ineligible": 0, "cuda_errors_cleared": 0}

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


def _needs_finalizer(base: np.ndarray, base_id: int) -> bool:
    """Whether to arm an unregister finalizer for ``base``.

    numpy-owned memory (``OWNDATA``: freed and its address reused once the
    array dies) needs one unless a LIVE finalizer already guards this very
    object. Today's test, "``id(base)`` not in ``_FINALIZERS``", is fooled
    by the stale entries the historical pop key leaves behind: a new buffer
    that reuses a dead wrapper's id got no finalizer, stayed registered after
    numpy freed it, and the next pinned allocation at that address failed
    with "resource already mapped" (n05, 2026-10-29: torch's own CUDA init).
    Other memory (a PSRDADA page wrapper) keeps today's test unchanged.
    """
    if base.flags["OWNDATA"]:
        fin = _FINALIZERS.get(base_id)
        return fin is None or not fin.alive
    return base_id not in _FINALIZERS


def _make_unregister_finalizer(addr: int, size: int, key: int | None = None,
                               owned: bool = False) -> None:
    """Schedule cudaHostUnregister(addr) when the owning python object dies.

    ``key`` is the ``_FINALIZERS`` key the finalizer was stored under
    (``id(base)``). Today's default removes ``addr`` instead -- a key that is
    never present, so entries are never removed -- and the force-clean test
    below (``base_id not in _FINALIZERS``) has come to depend on those stale
    ids. Only ``DSART_HOST_PIN_PERSIST=1`` removes the right key; the default
    keeps the historical behaviour bit for bit -- except for numpy-owned
    buffers (``owned``), whose entry is always removed by its own key (see
    :func:`_needs_finalizer`).

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
        _STATS["finalizer_unregister"] += 1
        rval = cudart.cudaHostUnregister(ctypes.c_void_p(addr))
        _clear_cuda_error(cudart, rval)
        if rval != 0:
            LOG.debug(
                "host-unpin (finalizer): cudaHostUnregister(0x%x) → rval=%d",
                addr, rval,
            )
        with _REGISTRY_LOCK:
            _REGISTERED.discard((addr, size))
            if key is not None and (_PERSIST or owned):
                # fires while its object is being freed: no live object can hold this id yet
                _FINALIZERS.pop(key, None)
            else:
                _FINALIZERS.pop(addr, None)

    return _on_finalize


def _pin_eligible(buf: object) -> bool:
    """Only memory whose lifetime the caller controls is page-locked.

    Eligible: an owner-less ``memoryview`` exported from C memory (a PSRDADA
    ring page: psrdada's reader hands out a READ-ONLY view of the shm page)
    and any array over it; writable buffers (numpy-owned, a POSIX-shm mmap);
    a read-only view of numpy-owned memory.

    Never: a ``np.memmap`` (file-backed), or read-only memory owned by some
    other Python object -- ``np.frombuffer(f.read(...))`` (a ``bytes`` page:
    the x2 I2 replay driver), a read-only ``mmap`` of a file, a read-only
    view of ``bytes``. Those come and go per block at recycled addresses;
    registering them (~30 ms per 288 MB) churns the driver's registry, and a
    stale registration at a reused address is what CUDA then reports as
    "resource already mapped" at an unrelated later call.
    """
    if isinstance(buf, memoryview):
        return not (buf.readonly and buf.obj is not None)
    if not isinstance(buf, np.ndarray):
        return True
    a = buf
    while True:
        if isinstance(a, np.memmap):
            return False
        if isinstance(a.base, np.ndarray):
            a = a.base
            continue
        break
    if buf.flags["WRITEABLE"]:
        return True
    root = a.base
    if root is None:
        return True                                  # a read-only view of numpy-owned memory
    if isinstance(root, memoryview):
        return root.obj is None                      # owner-less C memory: a PSRDADA page
    return False


def _clear_cuda_error(cudart, rval: int) -> None:
    """After a failed runtime call, consume the error it left: otherwise
    the next ``cudaGetLastError`` in PyTorch (after any kernel launch)
    reports it as that unrelated kernel's failure."""
    if rval != 0:
        _STATS["cuda_errors_cleared"] += 1
        try:
            cudart.cudaGetLastError()
        except Exception:  # noqa: BLE001 -- best effort
            pass


def _buffer_address_and_size(buf: object) -> tuple[int, int] | None:
    """Return (host_address, byte_count) for a numpy/memoryview-like host buffer.

    Returns ``None`` when the buffer doesn't expose a stable host address
    (e.g. zero-copy GPU tensors, non-contiguous numpy views) — the caller
    should silently skip in that case rather than failing.
    """
    if isinstance(buf, np.ndarray):
        if not buf.flags["C_CONTIGUOUS"] and not buf.flags["F_CONTIGUOUS"]:
            return None
        # CORRECTION (2026-05-17, M7.1 5-min soak bisect on n06):
        # psrdada-python's Reader.getNextPage() actually returns a
        # NON-writeable view of the ipcio_t ring page (the underlying
        # mmap on the reader side is PROT_READ-only — only the writer
        # has PROT_WRITE). So the production fada-read path was
        # falling through this branch and skipping pinning, dropping
        # the H2D path onto the pageable code (~4-5 GB/s vs ~12 GB/s
        # pinned, i.e. 60-72 ms vs 25 ms for a 288 MB fada block).
        # This is the 33 ms gap between the standalone bench
        # (108 ms p50, writeable synth buffers) and the orchestrator
        # (141 ms p50, non-writeable PSRDADA pages) measured at the
        # M7.1 x32 op-point.
        #
        # The original "skip if not WRITEABLE" was a band-aid for a
        # different problem: PyTorch's torch.as_tensor() on a
        # non-writeable array used to take a special internal H2D
        # path that did its own cudaHostRegister, conflicting with
        # ours and raising cudaErrorAlreadyMapped on the second
        # iteration. That collision is harmless now — the host_pin
        # registry treats already-mapped pages as success (see the
        # cudaErrorHostMemoryAlreadyRegistered=712 short-circuit in
        # maybe_register_host_buffer). So we let PSRDADA-style
        # non-writeable pages through, pin them, and skip the
        # PyTorch internal-pin path by ensuring the page is already
        # in the driver's pinned set.
        #
        # Bench paths that mint a fresh np.frombuffer(bytes) per
        # iteration are NOT pinned any more (x2 I2, 2026-10-29): those
        # pages come and go at recycled addresses, and on n05 the third
        # one failed its H2D with "invalid argument" (asynchronously:
        # "misaligned address" / "resource already mapped" later on).
        # :func:`_pin_eligible` keeps PSRDADA pages (read-only but
        # owner-less shm views) and refuses those.
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
    if not _pin_eligible(buf):
        _STATS["skipped_ineligible"] += 1
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

    # persistent pinning only for memory numpy does not own (a PSRDADA page: the base chain
    # ends at the reader's memoryview); numpy-owned buffers (warm-up dummy pages, benches) can
    # be freed and their address reused, so they keep today's finalizer path
    if _PERSIST and not (isinstance(buf, np.ndarray) and _numpy_base(buf).flags["OWNDATA"]):
        with _REGISTRY_LOCK:
            if (addr, size) in _REGISTERED:
                return True
        import time as _time
        _t0 = _time.perf_counter()
        rval = cudart.cudaHostRegister(
            ctypes.c_void_p(addr), ctypes.c_size_t(size),
            ctypes.c_uint(_CUDA_HOST_REGISTER_DEFAULT),
        )
        _clear_cuda_error(cudart, rval)        # do not leave the error for torch's next check
        if rval in (0, _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED):
            if rval == 0:
                _STATS["register_new"] += 1
                _STATS["register_ms"] += (_time.perf_counter() - _t0) * 1e3
            with _REGISTRY_LOCK:
                _REGISTERED.add((addr, size))
                n = len(_REGISTERED)
            LOG.info("host-pinned PSRDADA buffer page (persistent): addr=0x%x size=%d MB "
                     "(now %d page(s) pinned)", addr, size // (1024 * 1024), n)
            return True
        LOG.debug("host-pin: cudaHostRegister(addr=0x%x size=%d) failed rval=%d", addr, size, rval)
        return False

    import time as _time
    _t0 = _time.perf_counter()
    rval = cudart.cudaHostRegister(
        ctypes.c_void_p(addr), ctypes.c_size_t(size),
        ctypes.c_uint(_CUDA_HOST_REGISTER_DEFAULT),
    )
    _clear_cuda_error(cudart, rval)
    if rval == 0:
        _STATS["register_new"] += 1
        _STATS["register_ms"] += (_time.perf_counter() - _t0) * 1e3
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
                if _needs_finalizer(base, base_id):
                    cb = _make_unregister_finalizer(addr, size, key=base_id,
                                                    owned=bool(base.flags["OWNDATA"]))
                    if cb is not None:
                        _FINALIZERS[base_id] = weakref.finalize(base, cb)
        LOG.info(
            "host-pinned PSRDADA buffer page: addr=0x%x size=%d MB (now %d page(s) pinned)",
            addr, size // (1024 * 1024), n,
        )
        return True
    if rval == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        # Two cases land here:
        #   (a) Steady-state hot path: PSRDADA hands us back the same
        #       buffer page we already pinned earlier (same numpy base
        #       object). The driver still has the page-lock, the
        #       physical pages are unchanged, and we just refresh our
        #       bookkeeping. This is the production-fast path.
        #   (b) A fresh numpy buffer landed at a recycled VA whose
        #       previous owner's weakref finalizer hasn't fired yet
        #       (Python GC is non-deterministic; pytest + bench loops
        #       that mint a fresh ~288 MB buffer per iteration are the
        #       canonical trigger). The new array's *physical* pages
        #       are different from the old, but cudaHostRegister keys
        #       its bookkeeping on the host VA pointer alone, so the
        #       driver still has the stale pages page-locked. Treating
        #       this as success leaves us silently doing H2D copies
        #       against the wrong physical pages, which manifests
        #       later as cudaErrorInvalidArgument / "resource already
        #       mapped" inside the next torch.as_tensor.
        #
        # We distinguish via the _FINALIZERS dict (keyed by id(base)):
        # case (a) has a live finalizer for this base, case (b) does
        # not. In case (b) we force-unregister the stale registration
        # and re-register, so the driver sees the fresh physical pages.
        force_clean = False
        if isinstance(buf, np.ndarray):
            base = _numpy_base(buf)
            base_id = id(base)
            with _REGISTRY_LOCK:
                force_clean = _needs_finalizer(base, base_id)        # no LIVE finalizer: case (b)
        if force_clean:
            _STATS["force_clean"] += 1
            _t1 = _time.perf_counter()
            _clear_cuda_error(cudart, cudart.cudaHostUnregister(ctypes.c_void_p(addr)))
            with _REGISTRY_LOCK:
                _REGISTERED.discard((addr, size))
            rval = cudart.cudaHostRegister(
                ctypes.c_void_p(addr), ctypes.c_size_t(size),
                ctypes.c_uint(_CUDA_HOST_REGISTER_DEFAULT),
            )
            _clear_cuda_error(cudart, rval)
            _STATS["register_ms"] += (_time.perf_counter() - _t1) * 1e3
            if rval == 0:
                with _REGISTRY_LOCK:
                    _REGISTERED.add((addr, size))
                if isinstance(buf, np.ndarray):
                    base = _numpy_base(buf)
                    base_id = id(base)
                    with _REGISTRY_LOCK:
                        if _needs_finalizer(base, base_id):
                            cb = _make_unregister_finalizer(addr, size, key=base_id,
                                                            owned=bool(base.flags["OWNDATA"]))
                            if cb is not None:
                                _FINALIZERS[base_id] = weakref.finalize(
                                    base, cb,
                                )
                LOG.debug(
                    "host-pin: force-cleaned stale registration at "
                    "addr=0x%x size=%d MB (recycled VA), re-registered",
                    addr, size // (1024 * 1024),
                )
                return True
            LOG.debug(
                "host-pin: force-clean re-register failed rval=%d at "
                "addr=0x%x; falling back to pageable",
                rval, addr,
            )
            return False
        # Case (a): refresh bookkeeping; no driver call needed.
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
        _clear_cuda_error(cudart, rval)
        if rval == 0:
            n += 1
        else:
            LOG.debug("host-unpin: cudaHostUnregister(0x%x) → rval=%d",
                      addr, rval)
    return n


def stats() -> dict:
    """Copy of the diagnostic counters (fresh registrations and their total wall
    time in ms, finalizer unregisters, force-clean re-registrations)."""
    return dict(_STATS)


def n_registered() -> int:
    """Number of host buffers currently held in the registry."""
    with _REGISTRY_LOCK:
        return len(_REGISTERED)
