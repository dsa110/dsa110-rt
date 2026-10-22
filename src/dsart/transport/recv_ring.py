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
from dataclasses import dataclass
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
    1. ``_recv_ring.cpython-<MAJ><MIN>-*.so`` matching this interpreter's
       ABI tag (e.g. cpython-311 for Python 3.11). This is the artifact
       ``python setup.py build_ext --inplace`` writes for the active env.
    2. Any other ``_recv_ring*.so`` alongside this file (covers
       undecorated `_recv_ring.so` from editable builds).
    3. Fall back to loading ``librt.so`` and return a stub with a warning.

    Why the ABI-tag filter: hosts that have been through multiple
    interpreter upgrades (e.g. legacy cpython-38 builds left behind on
    n01) can end up with two ``_recv_ring*.so`` files in the package
    directory. The default ``glob`` order is filesystem-dependent and
    can return the OLD .so first, in which case the loader picks a
    binary that's out of sync with the current ``recv_ring.c`` symbol
    set — symptomatically: M7.2.9-era code that calls
    ``rx_ring_assemble_validity_block`` would silently fall back to
    the legacy Python loop. Selecting by ABI tag first avoids this.
    """
    import sys as _sys

    here = Path(__file__).parent
    abi_tag = f"cpython-{_sys.version_info.major}{_sys.version_info.minor}"
    abi_matches = sorted(here.glob(f"_recv_ring.{abi_tag}-*.so"))
    other_matches = [
        p for p in sorted(here.glob("_recv_ring*.so"))
        if p not in abi_matches
    ]
    candidates = abi_matches + other_matches
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

    # rx_ring_write_slot — M7.4 v2 signature (adds float scale + offset
    # before the uint16 validity_flags). The C library exposes ONLY the
    # v2 symbol after the M7.4 rebuild; older .so builds will fail at
    # ``RxRing.write_slot`` with a ctypes argtype mismatch (caller-side
    # stale-build detection).
    lib.rx_ring_write_slot.restype = ctypes.c_int
    lib.rx_ring_write_slot.argtypes = [
        ctypes.c_void_p,   # ring
        ctypes.c_uint32,   # corr
        ctypes.c_uint32,   # dm
        ctypes.c_uint64,   # t_seq
        ctypes.c_void_p,   # payload
        ctypes.c_size_t,   # payload_bytes
        ctypes.c_float,    # scale (M7.4)
        ctypes.c_float,    # offset (M7.4)
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

    # rx_ring_assemble_validity_block (M7.2 hot-path optimisation):
    # batched walk of (n_corr × popcount(coarse_dm_mask) × cube_cadence)
    # slot vf bytes; aggregates per-t validity + counters in one C call.
    # Bound only if the .so was rebuilt against the new recv_ring.c
    # (older builds will not export this symbol; the wrapper falls back
    # to a Python loop). Stale-binary detection lives in
    # ``RxRing.assemble_validity_block`` rather than here so the import
    # succeeds for older C builds too.
    if hasattr(lib, "rx_ring_assemble_validity_block"):
        lib.rx_ring_assemble_validity_block.restype = ctypes.c_int
        lib.rx_ring_assemble_validity_block.argtypes = [
            ctypes.c_void_p,                  # ring
            ctypes.c_uint64,                  # specnum_start
            ctypes.c_uint32,                  # cube_cadence_samples
            ctypes.c_uint32,                  # t_det
            ctypes.c_uint32,                  # compute_half
            ctypes.c_uint32,                  # coarse_dm_mask
            ctypes.c_uint32,                  # n_active_dms_per_corr
            ctypes.POINTER(ctypes.c_uint8),   # out_validity_per_t
            ctypes.POINTER(ctypes.c_uint64),  # out_n_overrun
            ctypes.POINTER(ctypes.c_uint64),  # out_n_pattern_mismatch
            ctypes.POINTER(ctypes.c_uint64),  # out_n_no_data_present
        ]

    # rx_ring_assemble_dense_block (M7.4): batched dense scatter +
    # per-(corr, t) scale/offset sidecar capture. Optional symbol so
    # callers with stale .so builds can fall back to the zero-stub
    # path in ProductionRxRingSource (M7.2 behaviour).
    if hasattr(lib, "rx_ring_assemble_dense_block"):
        lib.rx_ring_assemble_dense_block.restype = ctypes.c_int
        lib.rx_ring_assemble_dense_block.argtypes = [
            ctypes.c_void_p,                  # ring
            ctypes.c_uint64,                  # specnum_start
            ctypes.c_uint32,                  # t_det
            ctypes.c_uint32,                  # out_t_stride  (≥ t_det)
            ctypes.c_uint32,                  # n_grid
            ctypes.c_uint32,                  # owned_dm
            ctypes.c_uint32,                  # compute_half
            ctypes.c_uint32,                  # n_active_dms_per_corr
            ctypes.POINTER(ctypes.c_int32),   # n_filled_per_corr [n_corr]
            ctypes.POINTER(ctypes.c_int32),   # linear_lut_strided [n_corr * lut_stride]
            ctypes.c_uint32,                  # lut_stride
            ctypes.POINTER(ctypes.c_int8),    # out_cint8
            ctypes.POINTER(ctypes.c_float),   # out_scale_per_t [n_corr * out_t_stride]
            ctypes.POINTER(ctypes.c_float),   # out_offset_re_per_t [n_corr * out_t_stride]
            ctypes.POINTER(ctypes.c_float),   # out_offset_im_per_t [n_corr * out_t_stride]
            ctypes.POINTER(ctypes.c_uint8),   # out_validity_per_t [t_det]
            ctypes.POINTER(ctypes.c_uint64),  # out_n_overrun
            ctypes.POINTER(ctypes.c_uint64),  # out_n_pattern_mismatch
            ctypes.POINTER(ctypes.c_uint64),  # out_n_no_data_present
        ]

    # rx_ring_assemble_compact_block (M7.4.1 GPU-scatter, 2026-05-27):
    # compact COO wire-payload variant of dense_block. Optional symbol
    # so older .so builds gracefully degrade to the dense path.
    if hasattr(lib, "rx_ring_assemble_compact_block"):
        lib.rx_ring_assemble_compact_block.restype = ctypes.c_int
        lib.rx_ring_assemble_compact_block.argtypes = [
            ctypes.c_void_p,                  # ring
            ctypes.c_uint64,                  # specnum_start
            ctypes.c_uint32,                  # t_det
            ctypes.c_uint32,                  # sidecar_t_stride (>= t_det)
            ctypes.c_uint32,                  # owned_dm
            ctypes.c_uint32,                  # compute_half
            ctypes.c_uint32,                  # n_active_dms_per_corr
            ctypes.POINTER(ctypes.c_int32),   # n_filled_per_corr [n_corr]
            ctypes.POINTER(ctypes.c_int8),    # out_cells_packed [n_corr * t_det * n_filled_max * 2]
            ctypes.c_uint32,                  # n_filled_max
            ctypes.POINTER(ctypes.c_float),   # out_scale_per_t      [n_corr * sidecar_t_stride]
            ctypes.POINTER(ctypes.c_float),   # out_offset_re_per_t  [n_corr * sidecar_t_stride]
            ctypes.POINTER(ctypes.c_float),   # out_offset_im_per_t  [n_corr * sidecar_t_stride]
            ctypes.POINTER(ctypes.c_uint8),   # out_validity_per_t   [t_det]
            ctypes.POINTER(ctypes.c_uint64),  # out_n_overrun
            ctypes.POINTER(ctypes.c_uint64),  # out_n_pattern_mismatch
            ctypes.POINTER(ctypes.c_uint64),  # out_n_no_data_present
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
# Mirrors RX_RING_MAX_CORR in recv_ring.c (16 chgroups x n_sub <= 4).
RX_RING_MAX_CORR = 64


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
        if not 1 <= int(n_corr) <= RX_RING_MAX_CORR:
            raise ValueError(
                f"n_corr must be in [1, {RX_RING_MAX_CORR}]; got {n_corr}"
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
        """Attach as a consumer to an existing shm segment.

        NOTE: despite the historical method name (kept for
        backward-compatibility), the consumer mapping is
        ``PROT_READ|PROT_WRITE`` — the SPMC contract requires the
        consumer to atomically bump
        ``overrun_count_per_compute[half]`` (on every reader-side
        overrun in ``rx_ring_read_slot``) and to update
        ``read_seq_per_compute[half]`` (on every ``release``).
        With a true ``PROT_READ`` mapping the first overrun /
        release would segfault. OS-level isolation between writer
        and consumer is intentionally weak; the SPMC contract is
        enforced by the atomic protocol, not by mmap protection
        bits.
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
        *,
        scale: float = 0.0,
        offset: float = 0.0,
    ) -> None:
        """Write one COO slot to the ring (M7.4 v2 layout).

        Args:
            corr: correlator index (0..n_corr-1).
            dm: coarse DM index (0..n_coarse_dm-1).
            t_seq: absolute time-sequence number; slot index = t_seq % T_buf.
            payload: raw bytes or numpy array. ``None`` → zero-fill.
            validity_flags: bitmask (see VF_* constants).
            scale: per-slot dequant multiplicative scale (f32). M7.4 amend;
                defaults to 0.0 (which the search-side scatter treats as
                "skip dequant contribution"). Production callers
                (recv_epoll.c::commit_slot) pass the per-(cube, dm, t_idx)
                scale from the ProdFrame header.
            offset: per-slot dequant DC offset (f32). Symmetric cint8 path
                always emits 0; we ship one offset on the wire and the
                search-side scatter duplicates into re + im sidecars.
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
            ctypes.c_float(float(scale)),
            ctypes.c_float(float(offset)),
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

    def assemble_validity_block(
        self,
        *,
        specnum_start: int,
        cube_cadence_samples: int,
        t_det: int,
        compute_half: int = 0,
        coarse_dm_mask: int | None = None,
        n_active_dms_per_corr: int | None = None,
    ) -> tuple[np.ndarray, int, int, int]:
        """Batched validity-walk over a cube's worth of ring slots.

        Replaces the per-cube
        ``n_corr × popcount(coarse_dm_mask) × cube_cadence_samples``
        rx_ring_read_slot calls (the M7.2 ProductionRxRingSource hot
        path) with a single C call. Does NOT memcpy payloads — only the
        vf bytes are inspected, which is all the M7.2 consumer needs
        (the bring-up CubePipeline takes its per-chgroup streams from
        a pre-allocated zero-filled cache; see
        ``ProductionRxRingSource._per_chgroup_streams_zero``).

        Args:
            specnum_start: absolute t for slot 0 of the cube's detector
                window.
            cube_cadence_samples: stride between cube emits — reserved
                for future per-cube book-keeping. The walk uses
                ``t_det`` exclusively.
            t_det: detector window length (samples walked). The wseq
                wait in :class:`ProductionRxRingSource._iter` gates on
                ``t_det`` samples having been written past the cube
                boundary, so the walk is guaranteed in-window slots.
            compute_half: 0 or 1, for the per-half overrun-counter bump.
            coarse_dm_mask: bitmask of which coarse-DMs to include
                (bit ``i`` set ⇒ include ``dm=i``). ``None`` ⇒ all
                ``n_coarse_dm`` dims (matches the legacy Python loop).

        Returns:
            Tuple ``(validity_per_t, n_overrun, n_pattern_mismatch,
            n_no_data_present)``:

            * ``validity_per_t``: ``np.bool_`` array of shape
              ``(t_det,)``. Entry ``t`` is ``True`` iff every
              (corr, dm-in-mask) slot at that ``t`` reported
              ``VF_DATA_PRESENT`` and no overrun / pattern mismatch.
              Entries at ``t ≥ t_det`` are always ``True`` (callers
              only read ``[:t_det]``).
            * The three ``int`` counters are deltas for THIS cube; the
              caller folds them into its own running totals.

        Raises:
            RuntimeError: if the ring is closed.
            OSError: if the underlying C call returned non-zero (bad
                args, etc.).
            NotImplementedError: if the loaded ``_recv_ring.so`` does
                not export ``rx_ring_assemble_validity_block`` (stale
                build — rebuild with ``python setup.py build_ext
                --inplace``).
        """
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()
        if not hasattr(lib, "rx_ring_assemble_validity_block"):
            raise NotImplementedError(
                "_recv_ring.so does not export "
                "rx_ring_assemble_validity_block; rebuild the C "
                "extension: `python setup.py build_ext --inplace`"
            )

        if coarse_dm_mask is None:
            coarse_dm_mask = (1 << self.dims.n_coarse_dm) - 1

        # Default n_active_dms_per_corr to popcount(coarse_dm_mask) —
        # the producer wrote each dm bit, so popcount is the per-corr
        # per-sample write-slot count.
        if n_active_dms_per_corr is None:
            n_active_dms_per_corr = bin(int(coarse_dm_mask)).count("1")
        if n_active_dms_per_corr <= 0:
            n_active_dms_per_corr = 1

        # The C function writes ``t_det`` validity bytes (one per
        # detector-window sample). Size the output buffer accordingly.
        out = np.ones(int(t_det), dtype=np.uint8)
        n_over = ctypes.c_uint64(0)
        n_pat = ctypes.c_uint64(0)
        n_nodp = ctypes.c_uint64(0)
        ret = lib.rx_ring_assemble_validity_block(
            ctypes.c_void_p(self._handle),
            ctypes.c_uint64(int(specnum_start)),
            ctypes.c_uint32(int(cube_cadence_samples)),
            ctypes.c_uint32(int(t_det)),
            ctypes.c_uint32(int(compute_half)),
            ctypes.c_uint32(int(coarse_dm_mask)),
            ctypes.c_uint32(int(n_active_dms_per_corr)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(n_over),
            ctypes.byref(n_pat),
            ctypes.byref(n_nodp),
        )
        if ret != 0:
            raise OSError(
                f"rx_ring_assemble_validity_block failed (ret={ret})"
            )
        return (
            out.astype(np.bool_, copy=False),
            int(n_over.value),
            int(n_pat.value),
            int(n_nodp.value),
        )

    def assemble_dense_block(
        self,
        *,
        specnum_start: int,
        t_det: int,
        n_grid: int,
        owned_dm: int,
        n_filled_per_corr: np.ndarray,
        linear_lut_strided: np.ndarray,
        compute_half: int = 0,
        n_active_dms_per_corr: int = 1,
        out_t_stride: int | None = None,
        out_cint8: np.ndarray | None = None,
        out_scale: np.ndarray | None = None,
        out_offset_re: np.ndarray | None = None,
        out_offset_im: np.ndarray | None = None,
        out_validity: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,  # out_cint8                  (n_corr, out_t_stride, 2, n_grid, n_grid)
        np.ndarray,  # out_scale                  (n_corr, out_t_stride)
        np.ndarray,  # out_offset_re              (n_corr, out_t_stride)
        np.ndarray,  # out_offset_im              (n_corr, out_t_stride)
        np.ndarray,  # out_validity               (t_det,) bool
        int,         # n_overrun
        int,         # n_pattern_mismatch
        int,         # n_no_data_present
    ]:
        """Batched dense-scatter walk over a cube's worth of ring slots (M7.4).

        Drop-in replacement for the M7.2 zero-stub
        ``ProductionRxRingSource._per_chgroup_cint8_stack_zero`` path.
        Walks ``(n_corr, owned_dm, t in [0, t_det))`` ring slots and
        scatters each slot's COO cint8 payload into a dense per-(corr, t)
        ``[2, n_grid, n_grid]`` plane via the caller-supplied LUT, AND
        captures the per-slot ``scale`` / ``offset`` sidecar into
        per-(corr, t) arrays for the GPU dequant kernel
        (``fused_dequant_combine_per_fdm_per_t``).

        Args:
            specnum_start: absolute ``t`` for slot 0 of the cube window.
            t_det: detector window length. Same wseq-wait gate as
                :meth:`assemble_validity_block`.
            n_grid: dense-grid edge size.
            owned_dm: coarse-DM index to scatter (each search_compute
                half owns ONE coarse-DM; mirrors
                ``--coarse-dm-owners-half-{0,1}``).
            n_filled_per_corr: ``int32 [n_corr]`` actual N_filled per
                corr's sparsity pattern. Use ``-1`` to mark a corr as
                "intentionally silent" (no scatter; dense stays zero).
            linear_lut_strided: ``int32 [n_corr, lut_stride]`` row-major
                LUT. Entry ``[c, k]`` = ``ix_row[k] * n_grid + ix_col[k]``
                — the flat ``n_grid * n_grid`` target index. Padding
                rows beyond ``n_filled_per_corr[c]`` are ignored.
            compute_half: 0 or 1.
            out_*: caller-allocated output buffers; allocated fresh
                when ``None``. Reusing buffers across cubes avoids
                per-cube allocations on the search hot path.

        Returns:
            ``(out_cint8, out_scale, out_offset_re, out_offset_im,
            out_validity, n_overrun, n_pattern_mismatch,
            n_no_data_present)``.

        Raises:
            RuntimeError: if the ring is closed.
            OSError: if the C call returned non-zero.
            NotImplementedError: if the .so doesn't export
                ``rx_ring_assemble_dense_block`` (rebuild C extensions).
        """
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()
        if not hasattr(lib, "rx_ring_assemble_dense_block"):
            raise NotImplementedError(
                "_recv_ring.so does not export "
                "rx_ring_assemble_dense_block; rebuild the C "
                "extension: `python setup.py build_ext --inplace`"
            )

        n_corr = int(self.dims.n_corr)
        t_det = int(t_det)
        n_grid = int(n_grid)
        # out_t_stride: caller-managed T axis of the dense output. When
        # ``None`` we mirror t_det (smallest valid value); callers that
        # want lookahead beyond the detector window allocate a larger
        # buffer (e.g. T_stream-sized) and pass that explicitly.
        out_t_stride_i = int(out_t_stride) if out_t_stride is not None else t_det
        if out_t_stride_i < t_det:
            raise ValueError(
                f"out_t_stride={out_t_stride_i} must be >= t_det={t_det}"
            )

        n_filled_arr = np.ascontiguousarray(n_filled_per_corr, dtype=np.int32)
        if n_filled_arr.shape != (n_corr,):
            raise ValueError(
                f"n_filled_per_corr.shape={n_filled_arr.shape}; "
                f"expected ({n_corr},)"
            )
        lut = np.ascontiguousarray(linear_lut_strided, dtype=np.int32)
        if lut.ndim != 2 or lut.shape[0] != n_corr:
            raise ValueError(
                f"linear_lut_strided.shape={lut.shape}; "
                f"expected ({n_corr}, lut_stride)"
            )
        lut_stride = int(lut.shape[1])

        cint8_shape = (n_corr, out_t_stride_i, 2, n_grid, n_grid)
        if out_cint8 is None:
            out_cint8 = np.zeros(cint8_shape, dtype=np.int8)
        elif out_cint8.shape != cint8_shape:
            raise ValueError(
                f"out_cint8.shape={out_cint8.shape}; expected {cint8_shape}"
            )
        elif out_cint8.dtype != np.int8:
            raise ValueError(
                f"out_cint8.dtype={out_cint8.dtype}; expected int8"
            )
        if not out_cint8.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "out_cint8 must be C-contiguous (M7.4 scatter writes "
                "in row-major order)"
            )

        def _scratch_or_alloc(arr: np.ndarray | None, shape, dtype):
            if arr is None:
                return np.zeros(shape, dtype=dtype)
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(
                    f"buffer shape/dtype mismatch: got "
                    f"{arr.shape}/{arr.dtype}, expected {shape}/{dtype}"
                )
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "M7.4 scatter output buffers must be C-contiguous"
                )
            return arr

        per_t_shape = (n_corr, out_t_stride_i)
        out_scale     = _scratch_or_alloc(out_scale,     per_t_shape, np.float32)
        out_offset_re = _scratch_or_alloc(out_offset_re, per_t_shape, np.float32)
        out_offset_im = _scratch_or_alloc(out_offset_im, per_t_shape, np.float32)
        out_validity  = _scratch_or_alloc(out_validity,  (t_det,),    np.uint8)

        n_over = ctypes.c_uint64(0)
        n_pat = ctypes.c_uint64(0)
        n_nodp = ctypes.c_uint64(0)

        n_active_safe = int(n_active_dms_per_corr) if n_active_dms_per_corr > 0 else 1
        ret = lib.rx_ring_assemble_dense_block(
            ctypes.c_void_p(self._handle),
            ctypes.c_uint64(int(specnum_start)),
            ctypes.c_uint32(t_det),
            ctypes.c_uint32(out_t_stride_i),
            ctypes.c_uint32(n_grid),
            ctypes.c_uint32(int(owned_dm)),
            ctypes.c_uint32(int(compute_half)),
            ctypes.c_uint32(n_active_safe),
            n_filled_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_uint32(lut_stride),
            out_cint8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offset_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offset_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_validity.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(n_over),
            ctypes.byref(n_pat),
            ctypes.byref(n_nodp),
        )
        if ret != 0:
            raise OSError(
                f"rx_ring_assemble_dense_block failed (ret={ret})"
            )
        return (
            out_cint8,
            out_scale,
            out_offset_re,
            out_offset_im,
            out_validity.astype(np.bool_, copy=False),
            int(n_over.value),
            int(n_pat.value),
            int(n_nodp.value),
        )

    def assemble_compact_block(
        self,
        *,
        specnum_start: int,
        t_det: int,
        owned_dm: int,
        n_filled_per_corr: np.ndarray,
        n_filled_max: int,
        compute_half: int = 0,
        n_active_dms_per_corr: int = 1,
        sidecar_t_stride: int | None = None,
        out_cells_packed: np.ndarray | None = None,
        out_scale: np.ndarray | None = None,
        out_offset_re: np.ndarray | None = None,
        out_offset_im: np.ndarray | None = None,
        out_validity: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,  # out_cells_packed  (n_corr, t_det, n_filled_max * 2)
        np.ndarray,  # out_scale         (n_corr, sidecar_t_stride)
        np.ndarray,  # out_offset_re     (n_corr, sidecar_t_stride)
        np.ndarray,  # out_offset_im     (n_corr, sidecar_t_stride)
        np.ndarray,  # out_validity      (t_det,) bool
        int,         # n_overrun
        int,         # n_pattern_mismatch
        int,         # n_no_data_present
    ]:
        """Compact-payload walker (M7.4.1 GPU-scatter, 2026-05-27).

        Like :meth:`assemble_dense_block` but emits the raw COO wire
        payload (~30 MiB) instead of the dense per-(corr, t) plane
        (~565 MiB). The GPU-side scatter consumes the compact buffer
        + the per-corr LUT (preloaded GPU-side at startup) and writes
        the dense plane directly in device memory. Eliminates the
        CPU-side 565 MiB memset + 565 MiB H2D from the cube hot path.

        Args:
            specnum_start: absolute ``t`` for slot 0 of the cube window.
            t_det: detector window length.
            owned_dm: coarse-DM index to scatter.
            n_filled_per_corr: ``int32 [n_corr]`` actual N_filled per
                corr's sparsity pattern. ``-1`` marks a corr silent.
            n_filled_max: wire-side ``n_filled_per_corr`` from the
                ring header (the leading dim that bounds the slot
                payload bytes). Must equal ``dims.n_filled_per_corr``.
            compute_half: 0 or 1.
            n_active_dms_per_corr: same wseq-wait gate as
                :meth:`assemble_validity_block`.
            out_*: caller-allocated buffers for reuse across cubes.

        Returns:
            ``(cells_packed, scale, offset_re, offset_im, validity,
            n_overrun, n_pattern_mismatch, n_no_data_present)``.
        """
        if self._closed:
            raise RuntimeError("RxRing is closed")
        lib = _get_lib()
        if not hasattr(lib, "rx_ring_assemble_compact_block"):
            raise NotImplementedError(
                "_recv_ring.so does not export "
                "rx_ring_assemble_compact_block; rebuild the C "
                "extension: `python setup.py build_ext --inplace`"
            )

        n_corr = int(self.dims.n_corr)
        t_det = int(t_det)
        n_filled_max = int(n_filled_max)
        if n_filled_max != int(self.dims.n_filled_per_corr):
            raise ValueError(
                f"n_filled_max={n_filled_max} must equal "
                f"dims.n_filled_per_corr={self.dims.n_filled_per_corr}"
            )
        sidecar_t_stride_i = int(sidecar_t_stride) if sidecar_t_stride is not None else t_det
        if sidecar_t_stride_i < t_det:
            raise ValueError(
                f"sidecar_t_stride={sidecar_t_stride_i} must be >= t_det={t_det}"
            )

        n_filled_arr = np.ascontiguousarray(n_filled_per_corr, dtype=np.int32)
        if n_filled_arr.shape != (n_corr,):
            raise ValueError(
                f"n_filled_per_corr.shape={n_filled_arr.shape}; "
                f"expected ({n_corr},)"
            )

        cells_shape = (n_corr, t_det, n_filled_max * 2)
        if out_cells_packed is None:
            out_cells_packed = np.zeros(cells_shape, dtype=np.int8)
        elif (
            out_cells_packed.shape != cells_shape
            or out_cells_packed.dtype != np.int8
            or not out_cells_packed.flags["C_CONTIGUOUS"]
        ):
            raise ValueError(
                f"out_cells_packed must be C-contiguous int8 of shape "
                f"{cells_shape}; got {out_cells_packed.shape}/"
                f"{out_cells_packed.dtype}"
            )

        per_t_shape = (n_corr, sidecar_t_stride_i)

        def _scratch_or_alloc(arr, shape, dtype):
            if arr is None:
                return np.zeros(shape, dtype=dtype)
            if (
                arr.shape != shape
                or arr.dtype != dtype
                or not arr.flags["C_CONTIGUOUS"]
            ):
                raise ValueError(
                    f"buffer shape/dtype mismatch: got {arr.shape}/"
                    f"{arr.dtype}, expected {shape}/{dtype}"
                )
            return arr

        out_scale     = _scratch_or_alloc(out_scale,     per_t_shape, np.float32)
        out_offset_re = _scratch_or_alloc(out_offset_re, per_t_shape, np.float32)
        out_offset_im = _scratch_or_alloc(out_offset_im, per_t_shape, np.float32)
        out_validity  = _scratch_or_alloc(out_validity,  (t_det,),    np.uint8)

        n_over = ctypes.c_uint64(0)
        n_pat = ctypes.c_uint64(0)
        n_nodp = ctypes.c_uint64(0)

        n_active_safe = int(n_active_dms_per_corr) if n_active_dms_per_corr > 0 else 1
        ret = lib.rx_ring_assemble_compact_block(
            ctypes.c_void_p(self._handle),
            ctypes.c_uint64(int(specnum_start)),
            ctypes.c_uint32(t_det),
            ctypes.c_uint32(sidecar_t_stride_i),
            ctypes.c_uint32(int(owned_dm)),
            ctypes.c_uint32(int(compute_half)),
            ctypes.c_uint32(n_active_safe),
            n_filled_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_cells_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.c_uint32(n_filled_max),
            out_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offset_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_offset_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_validity.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(n_over),
            ctypes.byref(n_pat),
            ctypes.byref(n_nodp),
        )
        if ret != 0:
            raise OSError(
                f"rx_ring_assemble_compact_block failed (ret={ret})"
            )
        return (
            out_cells_packed,
            out_scale,
            out_offset_re,
            out_offset_im,
            out_validity.astype(np.bool_, copy=False),
            int(n_over.value),
            int(n_pat.value),
            int(n_nodp.value),
        )

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


# ===========================================================================
# Ring v4 (x2; docs/x2/SPEC.md §6). Nothing above this line changes.
# ===========================================================================
#
# Writer: the RX (``recv_epoll`` v4 mode) or :class:`RxRingV4Writer` (tests,
# replay). Readers: :class:`RxRingV4Reader` maps the segment PROT_READ and
# never writes a byte of it; its overrun / fan-in counters live in the
# reader process. Row index = the frame's ``seq_first + i`` (tier clock,
# = t0_native / t_int_factor), so rows are absolute-time aligned.

RX4_MAX_DM = 16
RX4_MAX_STREAM = 64
RX4_VERSION = 4
RX4_HEADER_BYTES = 135168

RV_PRESENT = 1 << 0
RV_HOLE = 1 << 1
RV_PATTERN_MISMATCH = 1 << 2
RV_MISSING = 1 << 3
RV_LAPPED = 1 << 4
RV_RFI_WARMING_UP = 1 << 5
RV_NOISE_WARMUP = 1 << 6
RV_ROW_GUARD = 1 << 7
RV_NAMES = ("present", "hole", "pattern_mismatch", "missing", "lapped",
            "rfi_warming_up", "noise_warmup", "row_guard")

RX4_COUNTERS = (
    "datagrams", "ok", "bad_length", "bad_magic", "bad_version", "bad_flags",
    "bad_field", "bad_time", "not_owned", "tier_mismatch", "format_mismatch",
    "stream_mismatch", "pattern_mismatch", "v1_frames", "port_mismatch",
    "bytes", "sweeps",
)
RX4_OK = 1
NATIVE_SAMPLE_S = 32.768e-6

_RX4_HDR_DT = np.dtype([
    ("magic", "<u4"), ("version", "<u4"), ("header_bytes", "<u4"), ("n_dm", "<u4"),
    ("n_stream", "<u4"), ("n_sub", "<u4"), ("row_bytes", "<u4"), ("bits_main", "<u4"),
    ("bits_heavy", "<u4"), ("value_kind", "<u4"), ("n_grid", "<u4"), ("quant_mode", "<u4"),
    ("s_main", "<f4"), ("s_heavy", "<f4"), ("check_scales", "<u4"), ("_pad0", "<u4"),
    ("shm_size", "<u8"), ("data_start", "<u8"), ("create_unix_ns", "<u8"),
    ("writer_pid", "<u8"), ("generation", "<u8"), ("hole_timeout_ns", "<u8"),
    ("heartbeat_ns", "<u8"), ("state", "<u8"),
])
_RX4_DM_DT = np.dtype([
    ("dm_global", "<u4"), ("t_int_factor", "<u4"), ("t_buf", "<u4"), ("reorder_rows", "<u4"),
    ("data_off", "<u8"), ("tags_off", "<u8"), ("steps_off", "<u8"), ("_pad", "<u8", (3,)),
])
_RX4_STREAM_DT = np.dtype([
    ("n_values", "<u4"), ("n_heavy", "<u4"), ("bytes_per_sample", "<u4"), ("pid_set", "<u4"),
    ("pattern_id", "<u8"), ("_pad", "<u8"),
])
_RX4_FLOW_FIELDS = (
    "complete_through", "max_end", "last_progress_ns", "datagrams", "rows_written",
    "rows_late", "rows_dup", "rows_hole", "rows_mismatch", "resyncs", "first_seq",
    "started", "gap_since_ns",
)
_RX4_FLOW_DT = np.dtype([(f, "<u8") for f in _RX4_FLOW_FIELDS] + [("_pad", "<u8", (3,))])
_RX4_OFF_COUNTERS, _RX4_OFF_DMS, _RX4_OFF_STREAMS, _RX4_OFF_FLOWS = 512, 768, 1792, 4096
assert _RX4_HDR_DT.itemsize == 128 and _RX4_DM_DT.itemsize == 64
assert _RX4_STREAM_DT.itemsize == 32 and _RX4_FLOW_DT.itemsize == 128


class Rx4Config(ctypes.Structure):
    """Mirror of ``rx4_config_t`` (recv_ring.h)."""

    _fields_ = [
        ("n_dm", ctypes.c_uint32), ("n_stream", ctypes.c_uint32),
        ("n_sub", ctypes.c_uint32), ("row_bytes", ctypes.c_uint32),
        ("bits_main", ctypes.c_uint32), ("bits_heavy", ctypes.c_uint32),
        ("value_kind", ctypes.c_uint32), ("n_grid", ctypes.c_uint32),
        ("quant_mode", ctypes.c_uint32), ("s_main", ctypes.c_float),
        ("s_heavy", ctypes.c_float), ("check_scales", ctypes.c_uint32),
        ("hole_timeout_ns", ctypes.c_uint64),
        ("dm_global", ctypes.c_uint32 * RX4_MAX_DM),
        ("t_int_factor", ctypes.c_uint32 * RX4_MAX_DM),
        ("t_buf", ctypes.c_uint32 * RX4_MAX_DM),
        ("reorder_rows", ctypes.c_uint32 * RX4_MAX_DM),
        ("stream_n_values", ctypes.c_uint32 * RX4_MAX_STREAM),
        ("stream_n_heavy", ctypes.c_uint32 * RX4_MAX_STREAM),
    ]


def bind_rx4_api(lib: ctypes.CDLL) -> None:
    """Bind the rx4_* symbols (present in _recv_ring and _recv_epoll)."""
    if getattr(lib, "_rx4_bound", False):
        return
    vp, u32, u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
    lib.rx4_create.restype = vp
    lib.rx4_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(Rx4Config),
                               ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.rx4_close.restype = None
    lib.rx4_close.argtypes = [vp]
    lib.rx4_unlink.restype = ctypes.c_int
    lib.rx4_unlink.argtypes = [ctypes.c_char_p]
    lib.rx4_base.restype = vp
    lib.rx4_base.argtypes = [vp]
    lib.rx4_size.restype = u64
    lib.rx4_size.argtypes = [vp]
    lib.rx4_ingest.restype = ctypes.c_int
    lib.rx4_ingest.argtypes = [vp, ctypes.c_char_p, ctypes.c_size_t, u64, u32]
    lib.rx4_sweep.restype = None
    lib.rx4_sweep.argtypes = [vp, u64, u64]
    lib.rx4_set_expected_pid.restype = ctypes.c_int
    lib.rx4_set_expected_pid.argtypes = [vp, u32, u64]
    lib.rx4_clear_expected_pid.restype = ctypes.c_int
    lib.rx4_clear_expected_pid.argtypes = [vp, u32]
    lib.rx4_now_ns.restype = u64
    lib.rx4_now_ns.argtypes = []
    lib.rx4_abi.restype = ctypes.c_int
    lib.rx4_abi.argtypes = [ctypes.POINTER(u64), u32]
    lib.rx4_scan.restype = ctypes.c_int
    lib.rx4_scan.argtypes = [vp, u32, vp, vp, vp]
    lib.rx4_window_validity.restype = ctypes.c_int
    lib.rx4_window_validity.argtypes = [vp, u32, u64, u32, vp, vp]
    lib.rx4_copy_window.restype = ctypes.c_int
    lib.rx4_copy_window.argtypes = [vp, u32, u64, u32, vp, vp, vp, vp]
    lib.rx4_recheck.restype = ctypes.c_int
    lib.rx4_recheck.argtypes = [vp, u32, u64, u32, vp, vp]
    lib._rx4_bound = True
    abi = (ctypes.c_uint64 * 32)()
    n = lib.rx4_abi(abi, 32)
    got = list(abi[:n])
    want = [RX4_HEADER_BYTES, _RX4_OFF_COUNTERS, _RX4_OFF_DMS, _RX4_OFF_STREAMS,
            _RX4_OFF_FLOWS, 64, 32, 128, 32, len(RX4_COUNTERS), 112, 120, 72,
            ctypes.sizeof(Rx4Config), 96, 32]
    if got != want:
        raise RuntimeError(f"ring v4 ABI mismatch between C and Python: {got} != {want}")


def _rx4_lib() -> ctypes.CDLL:
    lib = _get_lib()
    if not hasattr(lib, "rx4_create"):
        raise NotImplementedError(
            "_recv_ring.so has no ring v4 symbols; rebuild: python setup.py build_ext --inplace")
    bind_rx4_api(lib)
    return lib


def rx4_now_ns() -> int:
    """CLOCK_MONOTONIC ns (the clock of the ring's progress fields)."""
    return time.monotonic_ns()


@dataclass(frozen=True)
class RingV4Spec:
    """Dimensions + format of one node's v4 ring (validated, JSON-able).

    ``dms`` are the GLOBAL coarse indices owned by the node (both halves),
    ``t_int_factor`` their tier (16 A / 32 B native samples per row) and
    ``t_buf`` the rows kept per stream (retention = t_buf * tif * 32.768 us,
    must be >= ``min_retention_s``)."""

    dms: tuple
    t_int_factor: tuple
    t_buf: tuple
    n_stream: int
    n_sub: int
    bits_main: int
    bits_heavy: int = 16
    value_kind: int = 0
    n_grid: int = 256
    quant_mode: int = 0
    s_main: float = 0.0
    s_heavy: float = 0.0
    check_scales: bool = True
    stream_n_values: tuple = ()
    stream_n_heavy: tuple = ()
    row_bytes: Optional[int] = None
    reorder_rows: int = 64
    hole_timeout_s: float = 0.05
    min_retention_s: float = 20.0

    def __post_init__(self) -> None:
        n = len(self.dms)
        if not 1 <= n <= RX4_MAX_DM:
            raise ValueError(f"{n} owned DMs; ring v4 holds 1..{RX4_MAX_DM}")
        if len(self.t_int_factor) != n or len(self.t_buf) != n:
            raise ValueError("dms, t_int_factor and t_buf must have equal length")
        if len(set(self.dms)) != n:
            raise ValueError("duplicate dm in dms")
        if not 1 <= self.n_stream <= RX4_MAX_STREAM:
            raise ValueError(f"n_stream={self.n_stream} not in [1, {RX4_MAX_STREAM}]")
        for name in ("stream_n_values", "stream_n_heavy"):
            v = getattr(self, name)
            if v and len(v) != self.n_stream:
                raise ValueError(f"{name} must have n_stream entries")
        for i in range(n):
            ret = self.retention_s(i)
            if ret + 1e-9 < self.min_retention_s:
                raise ValueError(
                    f"dm {self.dms[i]}: retention {ret:.2f} s < {self.min_retention_s} s "
                    f"(t_buf={self.t_buf[i]}, t_int_factor={self.t_int_factor[i]})")
            if not 1 <= self.reorder_rows < self.t_buf[i]:
                raise ValueError("reorder_rows must be in [1, t_buf)")
        rb = self.row_bytes_eff
        if rb <= 0 or rb % 8:
            raise ValueError(f"row_bytes={rb} must be a positive multiple of 8")

    def retention_s(self, i: int) -> float:
        return self.t_buf[i] * self.t_int_factor[i] * NATIVE_SAMPLE_S

    def bytes_per_sample(self, s: int) -> int:
        if not self.stream_n_values:
            return 0
        nv, nh = int(self.stream_n_values[s]), int(self.stream_n_heavy[s] if self.stream_n_heavy else 0)
        return nh * self.bits_heavy // 8 + ((nv - nh) * self.bits_main + 7) // 8

    @property
    def row_bytes_eff(self) -> int:
        if self.row_bytes is not None:
            return int(self.row_bytes)
        if not self.stream_n_values:
            raise ValueError("row_bytes or stream_n_values is required")
        bps = max(self.bytes_per_sample(s) for s in range(self.n_stream))
        return (bps + 7) // 8 * 8

    @property
    def size_bytes(self) -> int:
        """Approximate segment size (header + tags + steps + data)."""
        rows = sum(self.n_stream * t for t in self.t_buf)
        return RX4_HEADER_BYTES + rows * (16 + self.row_bytes_eff)

    def to_config(self) -> Rx4Config:
        c = Rx4Config()
        c.n_dm = len(self.dms)
        c.n_stream = self.n_stream
        c.n_sub = self.n_sub
        c.row_bytes = self.row_bytes_eff
        c.bits_main = self.bits_main
        c.bits_heavy = self.bits_heavy
        c.value_kind = self.value_kind
        c.n_grid = self.n_grid
        c.quant_mode = self.quant_mode
        c.s_main = float(self.s_main)
        c.s_heavy = float(self.s_heavy)
        c.check_scales = 1 if self.check_scales else 0
        c.hole_timeout_ns = int(round(self.hole_timeout_s * 1e9))
        for i, (d, f, t) in enumerate(zip(self.dms, self.t_int_factor, self.t_buf)):
            c.dm_global[i] = int(d)
            c.t_int_factor[i] = int(f)
            c.t_buf[i] = int(t)
            c.reorder_rows[i] = int(self.reorder_rows)
        for s in range(self.n_stream):
            if self.stream_n_values:
                c.stream_n_values[s] = int(self.stream_n_values[s])
                c.stream_n_heavy[s] = int(self.stream_n_heavy[s]) if self.stream_n_heavy else 0
        return c

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "RingV4Spec":
        kw = dict(d)
        for k in ("dms", "t_int_factor", "t_buf", "stream_n_values", "stream_n_heavy"):
            if k in kw:
                kw[k] = tuple(int(x) for x in kw[k])
        return cls(**kw)

    @classmethod
    def for_node(cls, plan, value_map, opoint, *, node_index: int, rule=None,
                 retention_s: float = 20.0, cycle_rows_a: int = 384,
                 **kw) -> "RingV4Spec":
        """The spec of search node ``node_index`` (N_SEARCH order) from WS-P's
        ``TieredDmPlan`` + ``ValueMap`` + ``OpPoint``: the DMs of halves
        ``2i, 2i+1`` (owner_half = d % 8), their tier clocks, per-stream
        value counts, and t_buf = retention rounded UP to whole search cycles
        (``cycle_rows_a`` tier-A rows = 201.3 ms; tier B half as many rows)."""
        halves = (2 * node_index, 2 * node_index + 1)
        dms = [int(d) for d in range(len(plan.owner_half)) if int(plan.owner_half[d]) in halves]
        tif = [int(plan.coarse_t_native[d]) for d in dms]
        tbuf = []
        for f in tif:
            cyc = max(1, cycle_rows_a * 16 // f)
            rows = int(np.ceil(retention_s / (f * NATIVE_SAMPLE_S)))
            tbuf.append(int(np.ceil(rows / cyc) * cyc))
        s_main = s_heavy = 0.0
        if rule is not None:
            s_main, s_heavy = float(rule.s_main_f32), float(rule.s_heavy_f32)
        return cls(
            dms=tuple(dms), t_int_factor=tuple(tif), t_buf=tuple(tbuf),
            n_stream=int(value_map.n_streams), n_sub=int(value_map.n_sub),
            bits_main=int(opoint.bits_main), bits_heavy=int(opoint.bits_heavy) or 16,
            value_kind=0 if value_map.kind == "gridded" else 1,
            n_grid=int(value_map.n_grid), s_main=s_main,
            s_heavy=s_heavy, check_scales=rule is not None,
            stream_n_values=tuple(int(x) for x in value_map.n_values),
            stream_n_heavy=tuple(int(x) for x in value_map.n_heavy),
            min_retention_s=retention_s, **kw)


class RxRingV4Writer:
    """Owner handle of a v4 ring driven from Python (tests, replay, benches).
    Production writes come from the C RX thread (``RxEpoll.attach_ring4``)."""

    def __init__(self, handle: int, name: str, spec: RingV4Spec) -> None:
        self._h = handle
        self.name = name
        self.spec = spec
        self._lib = _rx4_lib()

    @classmethod
    def create(cls, name: str, spec: RingV4Spec, *, prefault: bool = False) -> "RxRingV4Writer":
        lib = _rx4_lib()
        err = ctypes.create_string_buffer(512)
        cfg = spec.to_config()
        h = lib.rx4_create(name.encode(), ctypes.byref(cfg), 1 if prefault else 0, err, len(err))
        if not h:
            raise OSError(f"rx4_create({name}) failed: {err.value.decode(errors='replace')}")
        return cls(h, name, spec)

    def ingest(self, datagram: bytes, *, now_ns: Optional[int] = None,
               port_chgroup: Optional[int] = None) -> int:
        """Place one frame-v2 datagram; returns the outcome index
        (``RX4_COUNTERS[rc]``; 1 = ok)."""
        b = bytes(datagram)
        return int(self._lib.rx4_ingest(
            ctypes.c_void_p(self._h), b, len(b),
            rx4_now_ns() if now_ns is None else int(now_ns),
            0xFFFFFFFF if port_chgroup is None else int(port_chgroup)))

    def sweep(self, *, now_ns: Optional[int] = None, stream_mask: int = (1 << 64) - 1) -> None:
        self._lib.rx4_sweep(ctypes.c_void_p(self._h),
                            rx4_now_ns() if now_ns is None else int(now_ns), int(stream_mask))

    def set_expected_pattern_id(self, stream: int, pattern_id: int) -> None:
        if self._lib.rx4_set_expected_pid(ctypes.c_void_p(self._h), int(stream), int(pattern_id)) != 0:
            raise ValueError(f"stream {stream} out of range")

    def clear_expected_pattern_id(self, stream: int) -> None:
        self._lib.rx4_clear_expected_pid(ctypes.c_void_p(self._h), int(stream))

    @property
    def size(self) -> int:
        return int(self._lib.rx4_size(ctypes.c_void_p(self._h)))

    def close(self) -> None:
        if self._h:
            self._lib.rx4_close(ctypes.c_void_p(self._h))
            self._h = 0

    def unlink(self) -> None:
        self._lib.rx4_unlink(self.name.encode())

    def __enter__(self) -> "RxRingV4Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass(frozen=True)
class DmInfo:
    dm: int                 # global coarse index
    local: int              # index inside this ring
    t_int_factor: int       # native samples per row
    t_buf: int              # rows kept per stream
    reorder_rows: int

    @property
    def sample_period_s(self) -> float:
        return self.t_int_factor * NATIVE_SAMPLE_S

    @property
    def retention_s(self) -> float:
        return self.t_buf * self.sample_period_s


@dataclass(frozen=True)
class RingFormat:
    n_stream: int
    n_sub: int
    row_bytes: int
    bits_main: int
    bits_heavy: int
    value_kind: int
    n_grid: int
    quant_mode: int
    s_main: float
    s_heavy: float


@dataclass(frozen=True)
class StreamInfo:
    stream: int
    n_values: int           # 0 = unchecked by the RX
    n_heavy: int
    bytes_per_sample: int
    pattern_id: Optional[int]


@dataclass
class RingWindow:
    """One (dm, [t_first, t_first + n_rows)) window of every stream.

    ``view`` is ``uint8 [n_stream, n_rows, row_bytes]``: zero-copy into the
    ring (read-only, DMA-able) when ``zero_copy`` — rows whose validity lacks
    ``RV_PRESENT`` then hold STALE bytes and the consumer must zero / mask
    them — or a zero-filled copy. ``validity`` is ``uint8 [n_stream,
    n_rows]`` of ``RV_*`` bits. Unpacks as ``view, validity``."""

    dm: int
    t_first: int
    n_rows: int
    view: np.ndarray
    validity: np.ndarray
    status: str             # complete | fan_in | deadline | lapped | timeout | copied
    n_final_streams: int
    waited_s: float
    zero_copy: bool
    steps: Optional[np.ndarray] = None   # f32 [n_stream, n_rows, 2] (copies only)

    def __iter__(self):
        yield self.view
        yield self.validity

    @property
    def usable(self) -> np.ndarray:
        return (self.validity & RV_PRESENT) != 0


@dataclass
class DmSnapshot:
    """Raw rows of one owned DM for the cutout / raw-window archive."""

    dm: int
    t_int_factor: int
    seq0: int                       # ring row of codes[:, 0]
    n_rows: int
    codes: np.ndarray               # uint8 [n_stream, n_rows, row_bytes] (invalid rows zero)
    steps: np.ndarray               # f32 [n_stream, n_rows, 2] (step_main, step_heavy)
    valid: np.ndarray               # uint8 [n_stream, n_rows] RV_* bits
    row0_seq: np.ndarray            # u64 [n_stream] (all = seq0; rows are time-aligned)
    row0_t0_native: np.ndarray      # u64 [n_stream] = seq0 * t_int_factor
    bytes_per_sample: np.ndarray    # u32 [n_stream] (0 = unchecked)
    complete_through: np.ndarray    # u64 [n_stream] at copy time
    coverage: float                 # fraction of streams final through the end
    status: str                     # complete | deadline | too_late


_TAG_WRITING = np.uint64(0xFFFFFFFFFFFFFFFF)
_SEQ_MASK = np.uint64((1 << 56) - 1)


def _classify_tags(tags: np.ndarray, t_first: int, max_end: np.ndarray, t_buf: int) -> np.ndarray:
    """numpy twin of ``rx4_classify`` (recv_ring.c): tags ``u64 [n_stream,
    n_rows]`` of rows ``t_first ..`` -> ``RV_*`` uint8."""
    n = tags.shape[1]
    seq = (np.uint64(t_first) + np.arange(n, dtype=np.uint64))[None, :]
    writing = tags == _TAG_WRITING
    ts = tags & _SEQ_MASK
    fl = (tags >> np.uint64(56)).astype(np.uint8)
    v = np.full(tags.shape, RV_MISSING, dtype=np.uint8)
    match = (ts == seq) & ~writing
    present = match & ((fl & 0x01) != 0)
    hole = match & ~present & ((fl & 0x02) != 0)
    mism = match & ~present & ~hole & ((fl & 0x04) != 0)
    pv = (RV_PRESENT | np.where(fl & 0x08, RV_RFI_WARMING_UP, 0)
          | np.where(fl & 0x10, RV_NOISE_WARMUP, 0)
          | np.where(fl & 0x20, RV_ROW_GUARD, 0)).astype(np.uint8)
    v[present] = pv[present]
    v[hole] = RV_HOLE
    v[mism] = RV_PATTERN_MISMATCH
    v[~writing & (ts > seq) & ((fl & 0x07) != 0)] = RV_LAPPED
    lap_w = writing & (max_end.astype(np.uint64)[:, None] > seq + np.uint64(t_buf))
    v[lap_w] = RV_LAPPED
    return v


class RxRingV4Reader:
    """Read-only consumer of a v4 ring (SPEC §6 API for WS-S).

    ``access="mmap"`` (default, today's consumer model): one ``PROT_READ``
    mapping; zero-copy windows, C helpers with acquire loads.
    ``access="pread"``: NO mapping at all — header, progress, tags, steps and
    rows are ``pread``/``preadv`` from an ``O_RDONLY`` fd into process memory
    (pass pinned host buffers as ``out=``), the seqlock check re-reads the
    tags after the copy. Both paths return identical bytes and validity
    (tested); CP1 times both under full load (ledger E5)."""

    def __init__(self, shm_name: str, fd: int, *, access: str = "mmap", mm=None,
                 mm_rw=None) -> None:
        if access not in ("mmap", "pread"):
            raise ValueError(f"access={access!r}, expected 'mmap' or 'pread'")
        self.shm_name = shm_name
        self.access = access
        self._fd = fd
        self._mm = mm
        self._mm_rw = mm_rw
        self._lib = _rx4_lib()
        static = os.pread(fd, RX4_HEADER_BYTES, 0)
        h = np.frombuffer(static, dtype=_RX4_HDR_DT, count=1)[0]
        if int(h["magic"]) != 0xD5A1107E or int(h["version"]) != RX4_VERSION:
            raise RuntimeError(f"{shm_name}: not a v4 ring (magic/version)")
        self._n_dm = int(h["n_dm"])
        self._data_start = int(h["data_start"])
        self._generation = int(h["generation"])
        self.fmt = RingFormat(
            n_stream=int(h["n_stream"]), n_sub=int(h["n_sub"]), row_bytes=int(h["row_bytes"]),
            bits_main=int(h["bits_main"]), bits_heavy=int(h["bits_heavy"]),
            value_kind=int(h["value_kind"]), n_grid=int(h["n_grid"]),
            quant_mode=int(h["quant_mode"]), s_main=float(h["s_main"]),
            s_heavy=float(h["s_heavy"]))
        self._dmd = np.frombuffer(static, dtype=_RX4_DM_DT, count=RX4_MAX_DM,
                                  offset=_RX4_OFF_DMS).copy()
        self._dms = []
        self._local = {}
        for i in range(self._n_dm):
            d = self._dmd[i]
            info = DmInfo(dm=int(d["dm_global"]), local=i, t_int_factor=int(d["t_int_factor"]),
                          t_buf=int(d["t_buf"]), reorder_rows=int(d["reorder_rows"]))
            self._dms.append(info)
            self._local[info.dm] = i
        ns, rb = self.fmt.n_stream, self.fmt.row_bytes
        self._data = {}
        if access == "mmap":
            self._buf = np.frombuffer(mm, dtype=np.uint8)
            self._base = self._buf.ctypes.data
            self._flows_mm = np.frombuffer(mm, dtype=_RX4_FLOW_DT,
                                           count=RX4_MAX_DM * RX4_MAX_STREAM,
                                           offset=_RX4_OFF_FLOWS).reshape(RX4_MAX_DM, RX4_MAX_STREAM)
            for i, info in enumerate(self._dms):
                off = int(self._dmd[i]["data_off"])
                n = ns * info.t_buf * rb
                src = self._buf
                if mm_rw is not None:
                    src = np.frombuffer(mm_rw, dtype=np.uint8)
                    off -= self._data_start
                arr = src[off:off + n].reshape(ns, info.t_buf, rb)
                arr.flags.writeable = False
                self._data[info.dm] = arr
        self._rc: dict[int, dict] = {d.dm: {"windows": {}, "rows": dict.fromkeys(RV_NAMES, 0),
                                             "overruns": 0, "stall_warnings": 0,
                                             "snapshots": {}} for d in self._dms}
        self._last_stall_warn: dict[int, float] = {}
        self._u64 = ctypes.c_uint64 * RX4_MAX_STREAM

    # -- open / close ----------------------------------------------------------

    @classmethod
    def open(cls, shm_name: str, *, access: str = "mmap", attach_timeout_s: float = 0.0,
             data_writable: bool = False) -> "RxRingV4Reader":
        """Attach read-only (``O_RDONLY``; ``PROT_READ`` in mmap mode). Waits up
        to ``attach_timeout_s`` for the segment to exist and be READY.
        ``data_writable`` (mmap mode) adds a second, read-write mapping of the
        DATA region only, so views can be ``cudaHostRegister``-ed without
        ``cudaHostRegisterReadOnly``; the views stay numpy-read-only and the
        header / tags are never mapped writable."""
        import mmap
        if access not in ("mmap", "pread"):
            raise ValueError(f"access={access!r}, expected 'mmap' or 'pread'")
        path = "/dev/shm/" + shm_name.lstrip("/")
        t_end = time.monotonic() + attach_timeout_s
        while True:
            fd = -1
            try:
                fd = os.open(path, os.O_RDONLY)
                size = os.fstat(fd).st_size
                if size >= RX4_HEADER_BYTES:
                    st = int(np.frombuffer(os.pread(fd, 8, 120), "<u8")[0])
                    if st >= 1:
                        break
            except FileNotFoundError:
                pass
            if fd >= 0:
                os.close(fd)
            if time.monotonic() >= t_end:
                raise FileNotFoundError(f"ring v4 {shm_name} not ready")
            time.sleep(0.05)
        mm = mm_rw = None
        try:
            if access == "mmap":
                mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
                if data_writable:
                    ds = int(np.frombuffer(os.pread(fd, 8, 72), "<u8")[0])
                    fd2 = os.open(path, os.O_RDWR)
                    try:
                        mm_rw = mmap.mmap(fd2, size - ds, flags=mmap.MAP_SHARED,
                                          prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=ds)
                    finally:
                        os.close(fd2)
            return cls(shm_name, fd, access=access, mm=mm, mm_rw=mm_rw)
        except Exception:
            os.close(fd)
            raise

    def close(self) -> None:
        self._data = {}
        for name in ("_buf", "_flows_mm"):
            if hasattr(self, name):
                setattr(self, name, None)
        for m in (self._mm, self._mm_rw):
            if m is not None:
                try:
                    m.close()
                except BufferError:
                    pass
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "RxRingV4Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- low-level reads (the only places that touch the segment) -------------

    def _u64_at(self, off: int) -> int:
        if self.access == "mmap":
            return int(np.frombuffer(self._mm, "<u8", 1, off)[0])
        return int(np.frombuffer(os.pread(self._fd, 8, off), "<u8")[0])

    def _flows(self, i: int) -> np.ndarray:
        ns = self.fmt.n_stream
        if self.access == "mmap":
            return self._flows_mm[i, :ns].copy()
        off = _RX4_OFF_FLOWS + i * RX4_MAX_STREAM * _RX4_FLOW_DT.itemsize
        return np.frombuffer(os.pread(self._fd, ns * _RX4_FLOW_DT.itemsize, off), _RX4_FLOW_DT)

    def _pread_into(self, out: np.ndarray, off: int) -> None:
        mv = memoryview(out).cast("B")
        got = 0
        while got < len(mv):
            n = os.preadv(self._fd, [mv[got:]], off + got)
            if n <= 0:
                raise OSError(f"short pread at {off + got}")
            got += n

    def _pread_runs(self, info: DmInfo, region_off: int, item: int, t_first: int, n_rows: int,
                    out: np.ndarray) -> None:
        """Per-stream [t_first, t_first+n_rows) of a [n_stream][t_buf] x ``item``
        region into ``out [n_stream, n_rows, ...]`` (1-2 preads per stream)."""
        tb = info.t_buf
        i0 = int(t_first) % tb
        n1 = min(n_rows, tb - i0)
        for s in range(self.fmt.n_stream):
            base = region_off + s * tb * item
            self._pread_into(out[s, :n1], base + i0 * item)
            if n1 < n_rows:
                self._pread_into(out[s, n1:], base)

    def _tags(self, info: DmInfo, t_first: int, n_rows: int) -> np.ndarray:
        t = np.empty((self.fmt.n_stream, n_rows), dtype="<u8")
        self._pread_runs(info, int(self._dmd[info.local]["tags_off"]), 8, t_first, n_rows, t)
        return t

    # -- description -----------------------------------------------------------

    def dms(self) -> list[DmInfo]:
        return list(self._dms)

    def dm_info(self, dm: int) -> DmInfo:
        return self._dms[self._loc(dm)]

    def streams(self) -> list[StreamInfo]:
        raw = os.pread(self._fd, RX4_MAX_STREAM * _RX4_STREAM_DT.itemsize, _RX4_OFF_STREAMS)
        sd = np.frombuffer(raw, dtype=_RX4_STREAM_DT)
        return [StreamInfo(stream=s, n_values=int(sd[s]["n_values"]), n_heavy=int(sd[s]["n_heavy"]),
                           bytes_per_sample=int(sd[s]["bytes_per_sample"]),
                           pattern_id=int(sd[s]["pattern_id"]) if int(sd[s]["pid_set"]) else None)
                for s in range(self.fmt.n_stream)]

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def writer_state(self) -> str:
        st = self._u64_at(120)
        return {0: "init", 1: "ready", 2: "closed"}.get(st, f"state{st}")

    def _loc(self, dm: int) -> int:
        try:
            return self._local[int(dm)]
        except KeyError:
            raise KeyError(f"dm {dm} is not owned by ring {self.shm_name} "
                           f"(owned: {sorted(self._local)})") from None

    # -- progress ----------------------------------------------------------------

    def heads(self, dm: int) -> dict:
        """Per-stream ``complete_through`` / ``max_end`` / ``last_progress_ns``."""
        i = self._loc(dm)
        n = self.fmt.n_stream
        if self.access == "mmap":
            ct, me, lp = self._u64(), self._u64(), self._u64()
            if self._lib.rx4_scan(ctypes.c_void_p(self._base), i, ct, me, lp) != 0:
                raise RuntimeError("rx4_scan failed")
            return {"complete_through": np.array(ct[:n], dtype=np.uint64),
                    "max_end": np.array(me[:n], dtype=np.uint64),
                    "last_progress_ns": np.array(lp[:n], dtype=np.uint64)}
        fl = self._flows(i)
        return {"complete_through": fl["complete_through"].astype(np.uint64),
                "max_end": fl["max_end"].astype(np.uint64),
                "last_progress_ns": fl["last_progress_ns"].astype(np.uint64)}

    def flow_stats(self, dm: int) -> dict:
        """Per-stream writer counters of one DM (``datagrams``, ``rows_written``,
        ``rows_hole``, ``rows_late``, ``rows_dup``, ``rows_mismatch``, ``resyncs``,
        ``started``) as int64 arrays."""
        fl = self._flows(self._loc(dm))
        return {f: fl[f].astype(np.int64) for f in _RX4_FLOW_FIELDS}

    def _min_streams(self, min_streams: Optional[int]) -> int:
        n = self.fmt.n_stream
        if min_streams is None:
            return max(1, -(-15 * n // 16))           # ceil(15/16 n) — today's 60/64
        return max(1, min(int(min_streams), n))

    def live_edge(self, dm: int, *, min_streams: Optional[int] = None) -> int:
        """Rows ``< live_edge`` are final on at least ``min_streams`` streams."""
        ct = np.sort(self.heads(dm)["complete_through"].astype(np.int64))
        return int(ct[ct.size - self._min_streams(min_streams)])

    def oldest(self, dm: int) -> int:
        """Oldest row not yet overwritten on any stream (``max(max_end) - t_buf``)."""
        me = self.heads(dm)["max_end"].astype(np.int64)
        return max(0, int(me.max()) - self.dm_info(dm).t_buf)

    def frontier(self, native_end: int, dms=None) -> dict:
        """Per-DM coverage of ``[.., native_end)``: streams final through the
        row holding native sample ``native_end - 1``."""
        out = {}
        for info in (self._dms if dms is None else [self.dm_info(d) for d in dms]):
            seq_end = -(-int(native_end) // info.t_int_factor)
            ct = self.heads(info.dm)["complete_through"].astype(np.int64)
            n_final = int((ct >= seq_end).sum())
            out[info.dm] = {"seq_end": seq_end, "n_final": n_final,
                            "coverage": n_final / ct.size,
                            "complete_through_min": int(ct.min()),
                            "complete_through_max": int(ct.max())}
        return out

    # -- windows -----------------------------------------------------------------

    def _check_rows(self, info: DmInfo, n_rows: int) -> None:
        if not 0 < n_rows <= info.t_buf:
            raise ValueError(f"n_rows={n_rows} not in (0, t_buf={info.t_buf}]")

    def window_validity(self, dm: int, t_first: int, n_rows: int) -> np.ndarray:
        info = self.dm_info(dm)
        self._check_rows(info, n_rows)
        if self.access == "pread":
            me = self.heads(info.dm)["max_end"]
            return _classify_tags(self._tags(info, t_first, n_rows), t_first, me, info.t_buf)
        v = np.zeros((self.fmt.n_stream, n_rows), dtype=np.uint8)
        rc = self._lib.rx4_window_validity(ctypes.c_void_p(self._base), info.local,
                                           int(t_first), int(n_rows), v.ctypes.data, None)
        if rc != 0:
            raise RuntimeError(f"rx4_window_validity rc={rc}")
        return v

    def view(self, dm: int, t_first: int, n_rows: int) -> Optional[np.ndarray]:
        """Zero-copy ``[n_stream, n_rows, row_bytes]`` view (mmap access), or
        None if the window wraps the ring end or access is pread."""
        if self.access != "mmap":
            return None
        info = self.dm_info(dm)
        i0 = int(t_first) % info.t_buf
        if i0 + n_rows > info.t_buf:
            return None
        return self._data[info.dm][:, i0:i0 + n_rows, :]

    def copy_window(self, dm: int, t_first: int, n_rows: int, *,
                    out: Optional[np.ndarray] = None, with_steps: bool = False) -> RingWindow:
        """Seqlock copy into ``out`` (e.g. a pinned host buffer): every
        non-PRESENT (or lapped-during-copy) row is zero-filled. Never blocks
        the writer."""
        info = self.dm_info(dm)
        self._check_rows(info, n_rows)
        ns, rb = self.fmt.n_stream, self.fmt.row_bytes
        shape = (ns, n_rows, rb)
        if out is None:
            out = np.empty(shape, dtype=np.uint8)
        elif out.shape != shape or out.dtype != np.uint8 or not out.flags["C_CONTIGUOUS"]:
            raise ValueError(f"out must be C-contiguous uint8 {shape}")
        steps = np.empty((ns, n_rows, 2), np.float32) if with_steps else None
        if self.access == "mmap":
            v = np.empty((ns, n_rows), dtype=np.uint8)
            rc = self._lib.rx4_copy_window(
                ctypes.c_void_p(self._base), info.local, int(t_first), int(n_rows),
                out.ctypes.data, None if steps is None else steps.ctypes.data, v.ctypes.data, None)
            if rc != 0:
                raise RuntimeError(f"rx4_copy_window rc={rc}")
        else:
            d = self._dmd[info.local]
            me = self.heads(info.dm)["max_end"]
            t1 = self._tags(info, t_first, n_rows)
            v = _classify_tags(t1, t_first, me, info.t_buf)
            self._pread_runs(info, int(d["data_off"]), rb, t_first, n_rows, out)
            if steps is not None:
                self._pread_runs(info, int(d["steps_off"]), 8, t_first, n_rows, steps)
            t2 = self._tags(info, t_first, n_rows)
            v[(t2 != t1) & ((v & RV_PRESENT) != 0)] = RV_LAPPED
            bad = (v & RV_PRESENT) == 0
            out[bad] = 0
            if steps is not None:
                steps[bad] = 0
        self._count_rows(info.dm, v)
        n_final = int((self.heads(info.dm)["complete_through"].astype(np.int64)
                       >= int(t_first) + n_rows).sum())
        return RingWindow(dm=info.dm, t_first=int(t_first), n_rows=int(n_rows), view=out,
                          validity=v, status="copied", n_final_streams=n_final,
                          waited_s=0.0, zero_copy=False, steps=steps)

    def recheck(self, window: RingWindow) -> int:
        """After consuming a zero-copy window: rows overwritten since are
        re-marked ``RV_LAPPED``; returns how many (counted as overruns)."""
        info = self.dm_info(window.dm)
        v = window.validity
        if not v.flags["C_CONTIGUOUS"] or not v.flags["WRITEABLE"]:
            v = window.validity = np.ascontiguousarray(v).copy()
        if self.access == "pread":
            tags = self._tags(info, window.t_first, window.n_rows)
            seq = np.uint64(window.t_first) + np.arange(window.n_rows, dtype=np.uint64)
            ok = ((tags != _TAG_WRITING) & ((tags & _SEQ_MASK) == seq[None, :])
                  & (((tags >> np.uint64(56)) & np.uint64(1)) != 0))
            chg = ((v & RV_PRESENT) != 0) & ~ok
            v[chg] = RV_LAPPED
            n = int(chg.sum())
        else:
            c = ctypes.c_uint64(0)
            rc = self._lib.rx4_recheck(ctypes.c_void_p(self._base), info.local, window.t_first,
                                       window.n_rows, v.ctypes.data, ctypes.byref(c))
            if rc != 0:
                raise RuntimeError(f"rx4_recheck rc={rc}")
            n = int(c.value)
        if n:
            self._rc[info.dm]["overruns"] += 1
            self._rc[info.dm]["rows"]["lapped"] += n
        return n

    def wait_window(self, dm: int, t_first: int, n_rows: int, *,
                    min_streams: Optional[int] = None, grace_s: float = 0.25,
                    deadline_s: float = 1.0, timeout_s: Optional[float] = None,
                    poll_s: float = 0.001, copy: bool = False,
                    out: Optional[np.ndarray] = None,
                    stall_warn_s: float = 3.0) -> RingWindow:
        """Wait for rows ``[t_first, t_first + n_rows)`` of every stream.

        Release rule (per DM fan-in): ``complete`` when every stream is final
        through the window; ``fan_in`` once >= ``min_streams`` (default
        ceil(15/16 n_stream), today's 60/64) are final and ``grace_s`` has
        passed since; ``deadline`` once ``deadline_s`` has passed since the
        FIRST stream became final, whatever the count; ``timeout`` after
        ``timeout_s`` (None = wait forever, with a rate-limited stall WARNING
        naming the behind streams every ``stall_warn_s``); ``lapped`` if the
        window start was already overwritten. Missing streams are flagged in
        ``validity``; ``copy`` (always, with access="pread", into ``out`` if
        given) returns a zero-filled copy instead of a zero-copy view."""
        info = self.dm_info(dm)
        self._check_rows(info, n_rows)
        end = int(t_first) + int(n_rows)
        need = self._min_streams(min_streams)
        t0 = time.monotonic()
        t_need = t_any = None
        status = None
        n_final = 0
        while True:
            hd = self.heads(info.dm)
            ct = hd["complete_through"].astype(np.int64)
            me = hd["max_end"].astype(np.int64)
            n_final = int((ct >= end).sum())
            now = time.monotonic()
            if int(me.max(initial=0)) > int(t_first) + info.t_buf:
                status = "lapped"
            elif n_final == ct.size:
                status = "complete"
            else:
                if n_final >= need:
                    t_need = now if t_need is None else t_need
                    if now - t_need >= grace_s:
                        status = "fan_in"
                if status is None and n_final >= 1:
                    t_any = now if t_any is None else t_any
                    if now - t_any >= deadline_s:
                        status = "deadline"
                if status is None and timeout_s is not None and now - t0 >= timeout_s:
                    status = "timeout"
            if status is not None:
                break
            if now - t0 >= stall_warn_s:
                self._stall_warn(info, ct, end, now - t0, hd["last_progress_ns"])
            time.sleep(poll_s)
        waited = time.monotonic() - t0
        if copy or out is not None or self.access == "pread":
            w = self.copy_window(info.dm, t_first, n_rows, out=out)
            w.status, w.waited_s, w.n_final_streams = status, waited, n_final
        else:
            v = self.window_validity(info.dm, t_first, n_rows)
            view = self.view(info.dm, t_first, n_rows)
            zc = view is not None
            if view is None:                 # wraps: stitched copy (stale rows kept)
                i0 = int(t_first) % info.t_buf
                a = self._data[info.dm][:, i0:, :]
                b = self._data[info.dm][:, :n_rows - a.shape[1], :]
                view = np.concatenate([a, b], axis=1)
            self._count_rows(info.dm, v)
            w = RingWindow(dm=info.dm, t_first=int(t_first), n_rows=int(n_rows), view=view,
                           validity=v, status=status, n_final_streams=n_final,
                           waited_s=waited, zero_copy=zc)
        rc = self._rc[info.dm]
        rc["windows"][status] = rc["windows"].get(status, 0) + 1
        if status == "lapped" or (w.validity & RV_LAPPED).any():
            rc["overruns"] += 1
        return w

    def snapshot(self, native_first: int, native_end: int, *, dms=None,
                 pad_rows: Optional[dict] = None, deadline_s: float = 15.0,
                 poll_s: float = 0.01) -> dict[int, DmSnapshot]:
        """Cutout / raw-window snapshot: for every owned DM (or ``dms``) copy
        the rows covering native samples ``[native_first, native_end)`` plus
        ``pad_rows[dm] = (pre, post)`` rows (e.g. the plan's stage-2 sweep),
        all streams, raw codes + steps + validity. Waits (polling; the writer
        is never blocked) until every stream is final through the end or
        ``deadline_s``, then copies; rows already overwritten make the DM
        ``too_late``."""
        infos = self._dms if dms is None else [self.dm_info(d) for d in dms]
        win = {}
        for info in infos:
            pre, post = (pad_rows or {}).get(info.dm, (0, 0))
            s0 = int(native_first) // info.t_int_factor - int(pre)
            s1 = -(-int(native_end) // info.t_int_factor) + int(post)
            if s1 - s0 > info.t_buf or s1 <= s0 or s0 < 0:
                raise ValueError(f"dm {info.dm}: rows [{s0}, {s1}) do not fit t_buf={info.t_buf}")
            win[info.dm] = (info, s0, s1)
        t_end = time.monotonic() + deadline_s
        while True:
            done = all(int(self.heads(d)["complete_through"].min()) >= s1
                       for d, (_, _, s1) in win.items())
            if done or time.monotonic() >= t_end:
                break
            time.sleep(poll_s)
        bps = np.array([s.bytes_per_sample for s in self.streams()], np.uint32)
        out = {}
        for d, (info, s0, s1) in win.items():
            too_late = self.oldest(d) > s0
            w = self.copy_window(d, s0, s1 - s0, with_steps=True)
            ct = self.heads(d)["complete_through"]
            cov = float((ct.astype(np.int64) >= s1).mean())
            status = "too_late" if too_late else ("complete" if cov == 1.0 else "deadline")
            ns = self.fmt.n_stream
            out[d] = DmSnapshot(
                dm=d, t_int_factor=info.t_int_factor, seq0=s0, n_rows=s1 - s0,
                codes=w.view, steps=w.steps, valid=w.validity,
                row0_seq=np.full(ns, s0, np.uint64),
                row0_t0_native=np.full(ns, s0 * info.t_int_factor, np.uint64),
                bytes_per_sample=bps, complete_through=ct, coverage=cov, status=status)
            sn = self._rc[d]["snapshots"]
            sn[status] = sn.get(status, 0) + 1
        return out

    # -- monitoring --------------------------------------------------------------

    def _count_rows(self, dm: int, v: np.ndarray) -> None:
        rows = self._rc[dm]["rows"]
        for bit, name in enumerate(RV_NAMES):
            rows[name] += int(np.count_nonzero(v & (1 << bit)))

    def _stall_warn(self, info: DmInfo, ct: np.ndarray, end: int, waited: float,
                    last_ns: np.ndarray) -> None:
        now = time.monotonic()
        last = self._last_stall_warn.get(info.dm, 0.0)
        if now - last < 3.0:
            return
        self._last_stall_warn[info.dm] = now
        self._rc[info.dm]["stall_warnings"] += 1
        behind = [int(s) for s in np.nonzero(ct < end)[0]]
        now_ns = rx4_now_ns()
        ages = {s: round((now_ns - int(last_ns[s])) / 1e9, 2) if int(last_ns[s]) else None
                for s in behind[:16]}
        LOG.warning(
            "ring v4 %s dm %d: window ending at row %d waited %.1f s; %d/%d streams "
            "behind %s (last progress age s: %s); writer heartbeat age %.2f s — "
            "check the corr fabric / transport for these streams",
            self.shm_name, info.dm, end, waited, len(behind), ct.size, behind, ages,
            self.heartbeat_age_s())

    def heartbeat_age_s(self) -> float:
        hb = self._u64_at(112)
        return (rx4_now_ns() - hb) / 1e9 if hb else float("inf")

    def counters(self) -> dict:
        """Writer counters (from the segment, read-only) + this reader's own
        per-DM counters (windows by status, rows by validity, overruns)."""
        c = np.frombuffer(os.pread(self._fd, 8 * 32, _RX4_OFF_COUNTERS), "<u8")
        writer = {name: int(c[i]) for i, name in enumerate(RX4_COUNTERS)}
        per_dm = {}
        for info in self._dms:
            fl = self._flows(info.local)
            agg = {f: int(fl[f].sum()) for f in ("datagrams", "rows_written", "rows_late",
                                                  "rows_dup", "rows_hole", "rows_mismatch",
                                                  "resyncs")}
            agg["streams_started"] = int((fl["started"] != 0).sum())
            agg["rows_hole_by_stream"] = fl["rows_hole"].astype(int).tolist()
            per_dm[info.dm] = {"writer": agg, "reader": _deepcopy_counts(self._rc[info.dm])}
        return {"writer": writer, "writer_state": self.writer_state,
                "heartbeat_age_s": self.heartbeat_age_s(), "access": self.access,
                "per_dm": per_dm}

    def stall_report(self, stall_s: float = 1.0) -> dict:
        """Streams without progress for ``stall_s`` (per DM) + writer liveness."""
        now_ns = rx4_now_ns()
        out = {"heartbeat_age_s": self.heartbeat_age_s(), "writer_state": self.writer_state,
               "dms": {}}
        for info in self._dms:
            fl = self._flows(info.local)
            started = fl["started"] != 0
            age = (now_ns - fl["last_progress_ns"].astype(np.int64)) / 1e9
            stalled = np.nonzero(started & (age > stall_s))[0]
            out["dms"][info.dm] = {
                "never_started": [int(s) for s in np.nonzero(~started)[0]],
                "stalled": {int(s): round(float(age[s]), 3) for s in stalled},
            }
        return out


def _deepcopy_counts(d: dict) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}


def unlink_ring_v4(name: str) -> None:
    """Remove a v4 segment name (no error if absent)."""
    try:
        os.unlink("/dev/shm/" + name.lstrip("/"))
    except FileNotFoundError:
        pass
