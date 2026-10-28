"""Compile CUDA C to PTX with libnvrtc through ctypes -- no device, no driver.

CPU nodes (and h23) cannot use cupy's NVRTC helpers (they query the device),
but libnvrtc itself is a pure compiler: CPU tests and ``b8 compile-check``
compile every owner-engine CUDA source for sm_75 with it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys
from typing import Optional, Sequence

__all__ = ["available", "ptx"]


def _lib() -> Optional[ctypes.CDLL]:
    cands = sorted(glob.glob(os.path.join(sys.prefix, "lib", "libnvrtc.so*")))
    cands = [c for c in cands if "builtins" not in c]
    path = cands[0] if cands else ctypes.util.find_library("nvrtc")
    if not path:
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


def available() -> bool:
    return _lib() is not None


def _include_opts() -> list:
    out = []
    for d in (os.path.join(sys.prefix, "include"),
              os.environ.get("CUDA_PATH", "") and os.path.join(os.environ["CUDA_PATH"], "include"),
              "/usr/local/cuda/include"):
        if d and os.path.exists(os.path.join(d, "cuda_fp16.h")):
            out.append("-I" + d)
            break
    return out


def ptx(src: str, *, arch: str = "compute_75", options: Sequence[str] = ("--std=c++14",)) -> str:
    """PTX of ``src``; raises RuntimeError with the NVRTC log on a compile error."""
    lib = _lib()
    if lib is None:
        raise RuntimeError("libnvrtc not found")
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"x2.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram rc={rc}")
    try:
        bo = [o.encode() for o in list(options) + _include_opts()
              + [f"--gpu-architecture={arch}"]]
        arr = (ctypes.c_char_p * len(bo))(*bo)
        rc = lib.nvrtcCompileProgram(prog, len(bo), arr)
        if rc != 0:
            n = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value)
            lib.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC error rc={rc}: {log.value.decode(errors='replace')}")
        n = ctypes.c_size_t()
        lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        lib.nvrtcGetPTX(prog, buf)
        return buf.value.decode(errors="replace")
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
