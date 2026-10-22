"""Make ``TRITON_INTERPRET=1`` usable on the search nodes (CPU CI only).

Triton 3.1's interpreter routes every masked load/store through a C++
helper (``triton._C.libtriton.interpreter.load/store``) whose pybind11
bool-mask conversion is broken under numpy >= 2 (measured on n13, numpy
2.4.4: loads return ``other``, stores write nothing, no error).
:func:`install` swaps in an equivalent numpy implementation (gather /
scatter of raw bytes at the lane addresses, masked lanes untouched). It
changes nothing on a GPU run and is only ever called by interpreter tests.
"""
from __future__ import annotations

import ctypes

import numpy as np

__all__ = ["install"]


def _view(lo: int, hi: int) -> np.ndarray:
    return np.ctypeslib.as_array((ctypes.c_uint8 * (hi - lo)).from_address(lo))


class _NumpyMemory:
    @staticmethod
    def load(ptrs, mask, other, dtype):
        dt = np.dtype(dtype)
        ptrs = np.asarray(ptrs, dtype=np.uint64)
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), ptrs.shape)
        out = np.broadcast_to(np.asarray(other, dtype=dt), ptrs.shape).copy()
        if mask.any():
            sel = ptrs[mask].astype(np.int64)
            lo, hi = int(sel.min()), int(sel.max()) + dt.itemsize
            mem = _view(lo, hi)
            idx = (sel - lo)[:, None] + np.arange(dt.itemsize)
            out[mask] = mem[idx].copy().view(dt).reshape(-1)
        return out

    @staticmethod
    def store(ptrs, value, mask):
        ptrs = np.asarray(ptrs, dtype=np.uint64)
        value = np.ascontiguousarray(np.broadcast_to(np.asarray(value), ptrs.shape))
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), ptrs.shape)
        if not mask.any():
            return
        it = value.dtype.itemsize
        sel = ptrs[mask].astype(np.int64)
        lo, hi = int(sel.min()), int(sel.max()) + it
        mem = _view(lo, hi)
        idx = (sel - lo)[:, None] + np.arange(it)
        mem[idx] = value[mask].view(np.uint8).reshape(-1, it)


def install() -> None:
    """Patch the Triton interpreter's memory helper (idempotent)."""
    from triton.runtime import interpreter as ti

    ti._interpreter = _NumpyMemory()
