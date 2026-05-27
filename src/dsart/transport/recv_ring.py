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
