"""Owner-side value codec (SPEC §4 row rule, §5 frame-v2 payload) -- K0 reference.

A ring row is one stream sample of one coarse DM. Two row formats:

* ``"cint8"`` (1x, today's wire v1 as stored by the RX ring): ``n_values``
  interleaved ``(int8 re, int8 im)`` pairs, codes summed WITHOUT any scale
  (production ``image/imager_gpu.py`` unit-scale path). Decode = identity.
* ``"frame2"`` (2x, SPEC §5 / ``transport/prod_frame2.py``): ``n_heavy``
  ``(int8 re, int8 im)`` pairs, then the main values as a packed
  little-endian bit stream, ``b`` bits per component, re in the low bits,
  two's complement, padded to a byte. Decode (SPEC §4.5, reference
  ``x2/quant.py``) = ``(code + off) * s`` with ``off = 0`` (mid-tread, the
  default) or ``0.5`` (mid-rise, frame flag bit 6), ``s = s_heavy`` /
  ``s_main`` (frame header, sigma_hat units), then times the static owner
  weight ``w_v = f32(sqrt(n_v))`` -- TWO fp32 roundings exactly as SPEC
  §4.5 / ``quant.decode`` + ``quant.natural_weight``:
  ``fl32(fl32((code + off) * s_v) * w_v)`` (``code + off`` is exact). The
  per-value constants are ``c_v = s_v`` (step) and ``w_v``; K1 multiplies
  twice, then adds, with selects between (no FMA contraction can make the
  GPU and its numpy twin differ). The dynamic sigma_hat is NOT applied
  (today's per-row 1/sigma weighting semantics).

Everything here is numpy and is the reference the K0 CUDA kernel's twin
(:mod:`dsart.x2.search.kernels_cuda`) is checked against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "RowFormat",
    "StreamFormats",
    "as_stream_formats",
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
    midrise: bool = False

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


@dataclass(frozen=True)
class StreamFormats:
    """Per-stream row formats of one DM (2x streams carry different value
    and heavy counts); a ring row is padded to the largest stream's bytes."""

    per_stream: tuple

    @property
    def kind(self) -> str:
        return self.per_stream[0].kind

    @property
    def row_bytes(self) -> int:
        return max(f.row_bytes for f in self.per_stream)

    @property
    def n_values(self) -> int:
        return max(f.n_values for f in self.per_stream)

    def __getitem__(self, s: int) -> RowFormat:
        return self.per_stream[s]

    def __len__(self) -> int:
        return len(self.per_stream)


def as_stream_formats(fmt, n_stream: int) -> StreamFormats:
    if isinstance(fmt, StreamFormats):
        if len(fmt) != n_stream:
            raise ValueError(f"{len(fmt)} stream formats for {n_stream} streams")
        return fmt
    return StreamFormats(tuple([fmt] * int(n_stream)))


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
    raw = np.ascontiguousarray(raw, dtype=np.uint8)[:, :fmt.row_bytes]
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
    a = np.ascontiguousarray(raw[:, :fmt.row_bytes]).view(np.int8).reshape(
        raw.shape[0], fmt.n_values, 2)
    return a[..., 0].copy(), a[..., 1].copy()


def decode_constants(
    fmt: RowFormat,
    *,
    step_main: float = 1.0,
    step_heavy: float = 1.0,
    n_contrib: Optional[np.ndarray] = None,
) -> tuple:
    """Per-value ``(c_v float32 [n_values], w_v float32 [n_values], offset)``.

    1x (``cint8``): ``c_v = w_v = 1``, offset 0 -- codes summed raw, exactly
    today's unit-scale combine. 2x (``frame2``): ``c_v = f32(s)`` with
    ``s = step_heavy`` for ``v < n_heavy`` else ``step_main``; ``w_v =
    f32(sqrt(n_v))`` (float64 sqrt, one rounding: ``quant.sqrt_n``) or 1;
    offset 0 (mid-tread) or 0.5 (mid-rise).
    """
    nv = fmt.n_values
    if fmt.kind == "cint8":
        return (np.ones(nv, dtype=np.float32), np.ones(nv, dtype=np.float32),
                np.float32(0.0))
    s = np.full(nv, np.float32(step_main), dtype=np.float32)
    s[:fmt.n_heavy] = np.float32(step_heavy)
    if n_contrib is None:
        w = np.ones(nv, dtype=np.float32)
    else:
        w = np.sqrt(np.asarray(n_contrib, dtype=np.float64)).astype(np.float32)
    return s, w, np.float32(0.5 if fmt.midrise else 0.0)


def decode_values(re: np.ndarray, im: np.ndarray, c_v: np.ndarray,
                  offset: float, w_v: Optional[np.ndarray] = None) -> tuple:
    """SPEC §4.5 decode per component: ``fl32(fl32((code + off) * c_v) * w_v)``.

    ``code + off`` is exact in float32 (an 8-bit integer plus 0); each
    product is one correctly-rounded float32 multiply -- the GPU's two
    ``__fmul_rn`` (no add in between, so nothing to contract).
    """
    c = np.asarray(c_v, dtype=np.float32)
    off = np.float32(offset)
    fr = (np.asarray(re, dtype=np.float32) + off) * c
    fi = (np.asarray(im, dtype=np.float32) + off) * c
    if w_v is not None:
        w = np.asarray(w_v, dtype=np.float32)
        fr = fr * w
        fi = fi * w
    return fr.astype(np.float32), fi.astype(np.float32)
