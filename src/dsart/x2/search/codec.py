"""Owner-side value codec (SPEC §4 row rule, §5 frame-v2 payload) -- K0 reference.

A ring row is one stream sample of one coarse DM. Two row formats:

* ``"cint8"`` (1x, today's wire v1 as stored by the RX ring): ``n_values``
  interleaved ``(int8 re, int8 im)`` pairs, codes summed WITHOUT any scale
  (production ``image/imager_gpu.py`` unit-scale path). Decode = identity.
* ``"frame2"`` (2x, SPEC §5): ``n_heavy`` ``(int8 re, int8 im)`` pairs, then
  the main values as a packed little-endian bit stream, ``b`` bits per
  component, re then im per value, two's complement, padded to a byte.
  Decode (SPEC §4.5) = ``(code + 0.5) * s`` with ``s = s_h`` (heavy) or
  ``s_b`` (main), times the static ``sqrt(n_v)`` and the owner weight
  ``w_v`` (``sqrt(multiplicity)`` for per-baseline values) -- all folded
  into ONE per-value fp32 constant ``c_v`` so the decode is a single
  rounding ``fl32((code + 0.5) * c_v)`` and K1 is pure fp32 adds (no FMA
  contraction can make the GPU and its numpy twin differ).

Everything here is numpy and is the reference the K0 CUDA kernel's twin
(:mod:`dsart.x2.search.kernels_cuda`) is checked against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "RowFormat",
    "pack_frame2_rows",
    "unpack_frame2_rows",
    "unpack_cint8_rows",
    "decode_constants",
    "decode_values",
]


@dataclass(frozen=True)
class RowFormat:
    """Byte layout of one ring row (one stream sample of one DM).

    Attributes:
        kind: ``"cint8"`` (1x) or ``"frame2"`` (2x).
        n_values: values per stream row (heavy + main); for 1x the
            stream's ``n_filled_max`` (rows are padded to it).
        n_heavy: leading int8 values (frame2 only).
        bits: main bits per COMPONENT (4, 5, 6; frame2 only). SPEC §5's
            ``bits_main`` field is per complex value (= 2 * bits).
    """

    kind: str
    n_values: int
    n_heavy: int = 0
    bits: int = 8

    def __post_init__(self) -> None:
        if self.kind not in ("cint8", "frame2"):
            raise ValueError(f"row kind {self.kind!r}")
        if self.kind == "frame2":
            if not 2 <= self.bits <= 8:
                raise ValueError(f"bits={self.bits}")
            if not 0 <= self.n_heavy <= self.n_values:
                raise ValueError(f"n_heavy={self.n_heavy} n_values={self.n_values}")

    @property
    def n_main(self) -> int:
        return self.n_values - (self.n_heavy if self.kind == "frame2" else 0)

    @property
    def row_bytes(self) -> int:
        if self.kind == "cint8":
            return 2 * self.n_values
        return 2 * self.n_heavy + (2 * self.bits * self.n_main + 7) // 8


def _to_twos(code: np.ndarray, bits: int) -> np.ndarray:
    return (np.asarray(code, dtype=np.int64) & ((1 << bits) - 1)).astype(np.uint64)


def pack_frame2_rows(re: np.ndarray, im: np.ndarray, fmt: RowFormat) -> np.ndarray:
    """``re/im [rows, n_values]`` signed codes -> ``uint8 [rows, row_bytes]``.

    Heavy values (the first ``n_heavy``) are int8; main values must lie in
    ``[-2^(b-1), 2^(b-1) - 1]``.
    """
    if fmt.kind != "frame2":
        raise ValueError("pack_frame2_rows needs a frame2 format")
    re = np.asarray(re)
    im = np.asarray(im)
    rows = re.shape[0]
    nh, nm, b = fmt.n_heavy, fmt.n_main, fmt.bits
    lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
    if nm and (re[:, nh:].min(initial=0) < lo or re[:, nh:].max(initial=0) > hi
               or im[:, nh:].min(initial=0) < lo or im[:, nh:].max(initial=0) > hi):
        raise ValueError(f"main codes outside {b}-bit range")
    out = np.zeros((rows, fmt.row_bytes), dtype=np.uint8)
    if nh:
        pair = np.empty((rows, nh, 2), dtype=np.int8)
        pair[..., 0] = re[:, :nh]
        pair[..., 1] = im[:, :nh]
        out[:, :2 * nh] = pair.reshape(rows, 2 * nh).view(np.uint8)
    if nm:
        comp = np.empty((rows, nm, 2), dtype=np.uint64)
        comp[..., 0] = _to_twos(re[:, nh:], b)
        comp[..., 1] = _to_twos(im[:, nh:], b)
        bits = ((comp[..., None] >> np.arange(b, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
        bits = bits.reshape(rows, 2 * nm * b)
        packed = np.packbits(bits, axis=1, bitorder="little")
        out[:, 2 * nh:2 * nh + packed.shape[1]] = packed
    return out


def unpack_frame2_rows(raw: np.ndarray, fmt: RowFormat) -> tuple:
    """``uint8 [rows, row_bytes]`` -> ``(re, im)`` int8 ``[rows, n_values]``."""
    if fmt.kind != "frame2":
        raise ValueError("unpack_frame2_rows needs a frame2 format")
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    rows = raw.shape[0]
    nh, nm, b = fmt.n_heavy, fmt.n_main, fmt.bits
    re = np.empty((rows, fmt.n_values), dtype=np.int8)
    im = np.empty((rows, fmt.n_values), dtype=np.int8)
    if nh:
        pair = raw[:, :2 * nh].view(np.int8).reshape(rows, nh, 2)
        re[:, :nh] = pair[..., 0]
        im[:, :nh] = pair[..., 1]
    if nm:
        nbits = 2 * nm * b
        bits = np.unpackbits(raw[:, 2 * nh:], axis=1, bitorder="little")[:, :nbits]
        bits = bits.reshape(rows, nm, 2, b).astype(np.int64)
        val = (bits << np.arange(b, dtype=np.int64)).sum(axis=-1)
        val = np.where(val >= (1 << (b - 1)), val - (1 << b), val)
        re[:, nh:] = val[..., 0]
        im[:, nh:] = val[..., 1]
    return re, im


def unpack_cint8_rows(raw: np.ndarray, fmt: RowFormat) -> tuple:
    """1x rows (``int8`` or ``uint8 [rows, 2 * n_values]``) -> ``(re, im)``."""
    a = np.ascontiguousarray(raw).view(np.int8).reshape(raw.shape[0], fmt.n_values, 2)
    return a[..., 0].copy(), a[..., 1].copy()


def decode_constants(
    fmt: RowFormat,
    *,
    step_main: float = 1.0,
    step_heavy: float = 1.0,
    n_contrib: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
) -> tuple:
    """Per-value ``(c_v float32 [n_values], offset float32)``.

    1x (``cint8``): ``c_v = 1``, offset 0 -- codes summed raw, exactly
    today's unit-scale combine. 2x (``frame2``): ``c_v = fl32(s * sqrt(n_v)
    * w_v)`` with ``s = step_heavy`` for ``v < n_heavy`` else
    ``step_main``; offset 0.5 (mid-rise reconstruction).
    """
    nv = fmt.n_values
    if fmt.kind == "cint8":
        return np.ones(nv, dtype=np.float32), np.float32(0.0)
    s = np.full(nv, float(step_main), dtype=np.float64)
    s[:fmt.n_heavy] = float(step_heavy)
    if n_contrib is not None:
        s = s * np.sqrt(np.asarray(n_contrib, dtype=np.float64))
    if weight is not None:
        s = s * np.asarray(weight, dtype=np.float64)
    return s.astype(np.float32), np.float32(0.5)


def decode_values(re: np.ndarray, im: np.ndarray, c_v: np.ndarray,
                  offset: float) -> tuple:
    """fp32 decode with ONE rounding per component: ``fl32((code + off) * c_v)``.

    ``code + 0.5`` and the product are formed in float64 (exact: a <=8-bit
    integer plus 0.5 times a float32 has <= 33 significant bits) and
    rounded once to float32 -- the same value the GPU's ``(float)code +
    0.5f`` then ``* c`` produces (both exact before the final rounding).
    """
    c = np.asarray(c_v, dtype=np.float64)
    off = float(offset)
    fr = ((np.asarray(re, dtype=np.float64) + off) * c).astype(np.float32)
    fi = ((np.asarray(im, dtype=np.float64) + off) * c).astype(np.float32)
    return fr, fi
