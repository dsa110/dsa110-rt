"""Python ctypes wrapper for ``recv_ring.c`` — POSIX-shm SPMC ring (M4a chunk 4).

Loads the ``_recv_ring`` C extension (built by setuptools) and exposes a
high-level :class:`RxRing` handle. The C library provides the CONC-1
contract (plan §4.4 lines 1463-1475):

- POSIX-shm segment opened/created by the RX (writer) process; attached
  read-only by compute (reader) processes.
- Atomic write protocol: payload bytes → validity_flags (release) →
  write_seq_per_corr (release). Compute reads with acquire semantics.
- SPMC: 1 writer, N_compute=2 readers. RX never reads read_seq_per_compute.

Huge-page fallback (D-item):
    The plan recommends ``MAP_LOCKED | MAP_HUGETLB | (1 GiB shift)``. This
    wrapper does NOT request huge pages at the ctypes level — that requires
    OS privilege. The underlying C mmap falls back to 4 KiB pages on any
    kernel that rejects hugetlb (e.g. if ``/proc/sys/vm/nr_hugepages == 0``).
    Document the fallback in ``M4a_PLAN_FIXES.md`` as D-item D1.

cudaHostRegister:
    Deferred to chunk 5 (``production_rx_ring.py``). This module performs
    no CUDA calls.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("dsart.transport.recv_ring")

# ---------------------------------------------------------------------------
# Locate and load the shared library
# ---------------------------------------------------------------------------

def _load_recv_ring_lib() -> ctypes.CDLL:
    """Load ``_recv_ring.so`` (built by setuptools next to this file).

    Search order:
    1. ``_recv_ring.so`` alongside this file (editable install / inplace build).
    2. Any ``_recv_ring*.so`` found by ``ctypes.util.find_library``.
    3. Fall back to loading ``librt.so`` and return a stub with a warning.
    """
    here = Path(__file__).parent
    # Glob for the build artifact (e.g. _recv_ring.cpython-311-x86_64-linux-gnu.so)
    candidates = list(here.glob("_recv_ring*.so"))
    if candidates:
        lib_path = str(candidates[0])
    else:
        # Try the undecorated name (editable build with --inplace).
        plain = here / "_recv_ring.so"
        if plain.exists():
            lib_path = str(plain)
        else:
            lib_path = None

    if lib_path is None:
        warnings.warn(
            "_recv_ring.so not found; recv_ring.py operating in stub mode. "
            "Run 'pip install -e .' or 'python setup.py build_ext --inplace' "
            "to build the C extension. "
            "D-item D2: document build pre-requisite in M4a_PLAN_FIXES.md.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None  # type: ignore[return-value]

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as exc:
        warnings.warn(
            f"Could not load {lib_path}: {exc}. "
            "recv_ring.py operating in stub mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None  # type: ignore[return-value]

    _bind_c_api(lib)
    return lib


def _bind_c_api(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype for each C function."""
    # rx_ring_open_or_create
    lib.rx_ring_open_or_create.restype = ctypes.c_void_p
    lib.rx_ring_open_or_create.argtypes = [
        ctypes.c_char_p,   # name
        ctypes.c_uint32,   # n_corr
        ctypes.c_uint32,   # n_coarse_dm
        ctypes.c_uint32,   # t_buf_samples
        ctypes.c_uint32,   # n_filled_per_corr
        ctypes.c_uint32,   # bytes_per_cell
        ctypes.c_int,      # owner
        ctypes.c_char_p,   # errbuf
        ctypes.c_size_t,   # errbuf_len
    ]
    # rx_ring_close
    lib.rx_ring_close.restype = None
    lib.rx_ring_close.argtypes = [ctypes.c_void_p]

    # rx_ring_unlink
    lib.rx_ring_unlink.restype = ctypes.c_int
    lib.rx_ring_unlink.argtypes = [ctypes.c_char_p]

    # rx_ring_write_slot
    lib.rx_ring_write_slot.restype = ctypes.c_int
    lib.rx_ring_write_slot.argtypes = [
        ctypes.c_void_p,   # ring
        ctypes.c_uint32,   # corr
        ctypes.c_uint32,   # dm
        ctypes.c_uint64,   # t_seq
        ctypes.c_void_p,   # payload
        ctypes.c_size_t,   # payload_bytes
        ctypes.c_uint16,   # validity_flags
    ]

    # rx_ring_read_slot
    lib.rx_ring_read_slot.restype = ctypes.c_int
    lib.rx_ring_read_slot.argtypes = [
        ctypes.c_void_p,   # ring
        ctypes.c_uint32,   # corr
        ctypes.c_uint32,   # dm
        ctypes.c_uint64,   # t_seq
        ctypes.c_uint32,   # compute_half
        ctypes.c_void_p,   # out_payload
        ctypes.c_size_t,   # out_payload_bytes
        ctypes.POINTER(ctypes.c_uint16),  # out_validity
    ]

    # rx_ring_update_read_seq
    lib.rx_ring_update_read_seq.restype = None
    lib.rx_ring_update_read_seq.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint64,
    ]

    # rx_ring_get_write_seq
    lib.rx_ring_get_write_seq.restype = ctypes.c_uint64
    lib.rx_ring_get_write_seq.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    # rx_ring_get_overrun_count
    lib.rx_ring_get_overrun_count.restype = ctypes.c_uint64
    lib.rx_ring_get_overrun_count.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    # rx_ring_memset_data
    lib.rx_ring_memset_data.restype = None
    lib.rx_ring_memset_data.argtypes = [ctypes.c_void_p]

    # rx_ring_get_dims
    lib.rx_ring_get_dims.restype = None
    lib.rx_ring_get_dims.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),  # n_corr
        ctypes.POINTER(ctypes.c_uint32),  # n_coarse_dm
        ctypes.POINTER(ctypes.c_uint32),  # t_buf_samples
        ctypes.POINTER(ctypes.c_uint32),  # n_filled_per_corr
        ctypes.POINTER(ctypes.c_uint32),  # bytes_per_cell
        ctypes.POINTER(ctypes.c_uint64),  # slot_stride_bytes
        ctypes.POINTER(ctypes.c_size_t),  # shm_size
    ]


# Lazy-load; None if the .so is missing (stub mode for tests that mock it).
_LIB: Optional[ctypes.CDLL] = None
_LIB_LOADED = False


def _get_lib() -> ctypes.CDLL:
    global _LIB, _LIB_LOADED
    if not _LIB_LOADED:
        _LIB = _load_recv_ring_lib()
        _LIB_LOADED = True
    if _LIB is None:
        raise RuntimeError(
            "_recv_ring.so not loaded. Build the C extension with "
            "'pip install -e .' and ensure gcc is available."
        )
    return _LIB


# ---------------------------------------------------------------------------
# Validity-flag constants (mirrors C header)
# ---------------------------------------------------------------------------

VF_DATA_PRESENT     = 0b000001  # bit0
VF_PATTERN_MISMATCH = 0b000010  # bit1
VF_RESERVED_B2      = 0b000100  # bit2
VF_RESERVED_B3      = 0b001000  # bit3
VF_RX_OVERRUN       = 0b010000  # bit4
VF_RFI_WARMING_UP   = 0b100000  # bit5

BYTES_CINT8_COMPLEX = 2  # 2 × int8
BYTES_CFP16_COMPLEX = 4  # 2 × float16


# ---------------------------------------------------------------------------
# RxRingDims — dimension tuple
# ---------------------------------------------------------------------------

class RxRingDims:
    """Immutable dimension parameters for an :class:`RxRing`."""

    __slots__ = (
        "n_corr",
        "n_coarse_dm",
        "t_buf_samples",
        "n_filled_per_corr",
        "bytes_per_cell",
    )

    def __init__(
        self,
        n_corr: int,
        n_coarse_dm: int,
        t_buf_samples: int,
        n_filled_per_corr: int,
        bytes_per_cell: int = BYTES_CINT8_COMPLEX,
    ) -> None:
        if bytes_per_cell not in (BYTES_CINT8_COMPLEX, BYTES_CFP16_COMPLEX):
            raise ValueError(
                f"bytes_per_cell must be 2 (cint8) or 4 (cfp16); got {bytes_per_cell}"
            )
        self.n_corr = int(n_corr)
        self.n_coarse_dm = int(n_coarse_dm)
        self.t_buf_samples = int(t_buf_samples)
        self.n_filled_per_corr = int(n_filled_per_corr)
        self.bytes_per_cell = int(bytes_per_cell)

    def __repr__(self) -> str:
        return (
            f"RxRingDims(n_corr={self.n_corr}, n_coarse_dm={self.n_coarse_dm}, "
            f"t_buf_samples={self.t_buf_samples}, "
            f"n_filled_per_corr={self.n_filled_per_corr}, "
            f"bytes_per_cell={self.bytes_per_cell})"
        )


# ---------------------------------------------------------------------------
# RxRing — Python handle
# ---------------------------------------------------------------------------


class RxRing:
    """POSIX-shm SPMC receive ring handle.

    Created by the RX (writer) process; attached read-only by compute readers.

    Examples::

        # Writer (RX process):
        ring = RxRing.open_or_create(
            "/dsart_rx_ring_01", RxRingDims(16, 24, 512, 5800)
        )
        ring.write_slot(corr=0, dm=3, t_seq=100, payload=buf, validity=VF_DATA_PRESENT)

        # Reader (compute process, read-only attach):
        ring = RxRing.mmap_attach_readonly("/dsart_rx_ring_01", dims)
        payload, vf = ring.read_slot(corr=0, dm=3, t_seq=100, compute_half=0)
    """

    def __init__(self, _handle: int, dims: RxRingDims, *, owner: bool) -> None:
        self._handle = _handle   # opaque C pointer (as Python int)
        self.dims = dims
        self._owner = owner
        self._closed = False

    @classmethod
    def open_or_create(
        cls,
        name: str,
        dims: RxRingDims,
    ) -> "RxRing":
        """Create (or re-open) a shm segment as the writer/owner.

        Zero-initialises header + data on creation.
        """
        lib = _get_lib()
        errbuf = ctypes.create_string_buffer(256)
        handle = lib.rx_ring_open_or_create(
            name.encode(),
            dims.n_corr,
            dims.n_coarse_dm,
            dims.t_buf_samples,
            dims.n_filled_per_corr,
            dims.bytes_per_cell,
            1,   # owner
            errbuf,
            256,
        )
        if handle == 0 or handle is None:
            raise OSError(
                f"rx_ring_open_or_create failed: {errbuf.value.decode()}"
            )
        return cls(handle, dims, owner=True)

    @classmethod
    def mmap_attach_readonly(
        cls,
        name: str,
        dims: RxRingDims,
    ) -> "RxRing":
        """Attach read-only to an existing shm segment (compute reader)."""
        lib = _get_lib()
        errbuf = ctypes.create_string_buffer(256)
        handle = lib.rx_ring_open_or_create(
            name.encode(),
            dims.n_corr,
            dims.n_coarse_dm,
            dims.t_buf_samples,
            dims.n_filled_per_corr,
            dims.bytes_per_cell,
            0,   # reader
            errbuf,
            256,
        )
        if handle == 0 or handle is None:
            raise OSError(
                f"rx_ring_open_or_create (reader) failed: {errbuf.value.decode()}"
            )
        return cls(handle, dims, owner=False)

    def close(self) -> None:
        """Unmap the shm segment. Does not unlink the name."""
        if not self._closed and self._handle:
            lib = _get_lib()
            lib.rx_ring_close(ctypes.c_void_p(self._handle))
            self._handle = 0
            self._closed = True

    def unlink(self) -> None:
        """Remove the shm segment name (writer only). Call after close()."""
        # We can call shm_unlink even after closing fd.
        pass  # delegate to class method

    @staticmethod
    def unlink_name(name: str) -> int:
        """Remove the shm name from the namespace."""
        lib = _get_lib()
        return lib.rx_ring_unlink(name.encode())

    def __enter__(self) -> "RxRing":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_slot(
        self,
        corr: int,
        dm: int,
        t_seq: int,
        payload: bytes | np.ndarray | None,
        validity_flags: int,
    ) -> None:
        """Write one COO slot to the ring.

        Args:
            corr: correlator index (0..n_corr-1).
            dm: coarse DM index (0..n_coarse_dm-1).
            t_seq: absolute time-sequence number; slot index = t_seq % T_buf.
            payload: raw bytes or numpy array. ``None`` → zero-fill.
            validity_flags: bitmask (see VF_* constants).
        """
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()

        if payload is None:
            payload_ptr = None
            payload_bytes = 0
        elif isinstance(payload, np.ndarray):
            payload_ptr = payload.ctypes.data_as(ctypes.c_void_p)
            payload_bytes = payload.nbytes
        else:
            payload_bytes_array = (ctypes.c_char * len(payload))(*payload)
            payload_ptr = ctypes.cast(payload_bytes_array, ctypes.c_void_p)
            payload_bytes = len(payload)

        ret = lib.rx_ring_write_slot(
            ctypes.c_void_p(self._handle),
            ctypes.c_uint32(corr),
            ctypes.c_uint32(dm),
            ctypes.c_uint64(t_seq),
            payload_ptr,
            ctypes.c_size_t(payload_bytes),
            ctypes.c_uint16(validity_flags),
        )
        if ret != 0:
            raise OSError(f"rx_ring_write_slot failed (ret={ret})")

    def read_slot(
        self,
        corr: int,
        dm: int,
        t_seq: int,
        compute_half: int = 0,
    ) -> tuple[bytes, int]:
        """Read one COO slot from the ring.

        Returns:
            (payload_bytes, validity_flags) tuple.

        Raises:
            OSError: on overrun or invalid parameters (ret=-1).
        """
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()

        payload_size = self.dims.n_filled_per_corr * self.dims.bytes_per_cell
        out_buf = (ctypes.c_char * payload_size)()
        out_vf = ctypes.c_uint16(0)

        ret = lib.rx_ring_read_slot(
            ctypes.c_void_p(self._handle),
            ctypes.c_uint32(corr),
            ctypes.c_uint32(dm),
            ctypes.c_uint64(t_seq),
            ctypes.c_uint32(compute_half),
            ctypes.cast(out_buf, ctypes.c_void_p),
            ctypes.c_size_t(payload_size),
            ctypes.byref(out_vf),
        )
        if ret != 0:
            raise OSError(
                f"rx_ring_read_slot failed (ret={ret}, vf={out_vf.value:#04x})"
            )
        return bytes(out_buf), int(out_vf.value)

    def update_read_seq(self, compute_half: int, new_read_seq: int) -> None:
        """Advance a compute reader's read sequence."""
        if self._closed:
            raise RuntimeError("RxRing is closed")
        # PROT_READ for compute-attached rings (plan §4.4 line 1467) — the
        # header is mapped read-only on the reader side, so an atomic
        # store into read_seq_per_compute[] from the C library would
        # SIGSEGV. The header field is dead state in v1 (plan: RX never
        # reads it), so compute readers track read_seq in their own
        # process memory and skip the C-level update. See M4a D14.
        if not self._owner:
            return
        lib = _get_lib()
        lib.rx_ring_update_read_seq(
            ctypes.c_void_p(self._handle),
            ctypes.c_uint32(compute_half),
            ctypes.c_uint64(new_read_seq),
        )

    def get_write_seq(self, corr: int) -> int:
        """Current write_seq_per_corr[corr] (acquire load)."""
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()
        return int(lib.rx_ring_get_write_seq(
            ctypes.c_void_p(self._handle), ctypes.c_uint32(corr)
        ))

    def get_overrun_count(self, compute_half: int) -> int:
        """Per-compute-half overrun counter (acquire load)."""
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()
        return int(lib.rx_ring_get_overrun_count(
            ctypes.c_void_p(self._handle), ctypes.c_uint32(compute_half)
        ))

    def memset_data(self) -> None:
        """Zero-fill entire data section (cmd: prepare; owner only)."""
        if not self._owner:
            raise PermissionError("memset_data requires owner=True")
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()
        lib.rx_ring_memset_data(ctypes.c_void_p(self._handle))
