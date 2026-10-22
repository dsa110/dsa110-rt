"""K0 (ring rows -> time-minor code ring) and K1 (CSR combine), numpy twins.

Stream ring (per owned DM, on the GPU in production): ``int8 [S, V, R, 2]``
-- per stream ``s``, value ``v``, a circular time axis (absolute stream
row ``r`` in slot ``r % R``), ``(re, im)`` codes. Time-minor so one
CSR entry's window is a contiguous run (today's ``transpose_compact_tmajor``
layout ``[g, k, t, 2]``). Codes stay codes ("codes as today"): K1 decodes
on load with the per-value constant ``c_v`` (``codec.decode_constants``).

K1 output: ``uv [F, T, n_act]`` complex64 -- the active half-plane cells
in :class:`~dsart.x2.search.csr.OwnerCsr` order (column-major), already
folded (x 0.5) and checkerboarded (x (-1)^(u+w)). The accumulation is
fp32, one accumulator per component, entries in CSR order; masked
entries add ``+0.0``; decode is ``(float(code) + off) * c_v`` (exact add,
one rounding). The Triton kernel performs the same operations in the same
order, so twin and kernel agree bit for bit. At 1x (``c_v = 1``,
``off = 0``, int8 codes) every partial sum is an integer below 2^24, so
the result equals ``combine_sparse_v4`` (int32 accumulate, ``0.5f *``,
sign, round to half) after the final fp16 rounding.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .codec import RowFormat, as_stream_formats, unpack_cint8_rows, unpack_frame2_rows
from .csr import OwnerCsr

__all__ = [
    "StreamRing",
    "k0_rows_to_ring",
    "k1_twin",
    "v4_emulate",
]


class StreamRing:
    """Host/numpy stream ring of one DM (the CPU path's K0 target)."""

    def __init__(self, n_stream: int, n_values: int, n_slots: int) -> None:
        self.codes = np.zeros((n_stream, n_values, n_slots, 2), dtype=np.int8)
        self.valid = np.zeros((n_stream, n_slots), dtype=bool)
        self.n_slots = int(n_slots)
        self.loaded_hi: Optional[int] = None      # exclusive absolute row
        self.loaded_lo: Optional[int] = None
        self.n_nodata = 0                           # "no data" sentinel values seen

    def slots(self, rows: np.ndarray) -> np.ndarray:
        return np.mod(rows, self.n_slots)


def k0_rows_to_ring(ring: StreamRing, raw: np.ndarray, valid: np.ndarray, *,
                    row0: int, fmt: RowFormat) -> None:
    """Unpack ``raw [S, n, row_bytes]`` (rows ``row0 .. row0 + n - 1``) into
    the stream ring; invalid ``(stream, row)`` (``valid [S, n]`` False) are
    written as zero codes -- the flagged zero-fill of SPEC §6 (0 in DECODED
    space: mid-tread codes, offset 0). Frame-v2 "no data" sentinels
    (code -2^(b-1); -128 for heavy) decode to 0 and are counted."""
    S, n, _ = raw.shape
    if n > ring.n_slots:
        raise ValueError(f"{n} rows > ring of {ring.n_slots}")
    slots = ring.slots(np.arange(row0, row0 + n))
    fmts = as_stream_formats(fmt, S)
    for s in range(S):
        fs = fmts[s]
        unpack = unpack_cint8_rows if fs.kind == "cint8" else unpack_frame2_rows
        nv = fs.n_values
        re, im = unpack(raw[s], fs)                      # [n, nv]
        ok = np.asarray(valid[s], dtype=bool)
        if fs.kind == "frame2":
            # SPEC §4 (Phase 0): code -2^(b-1) = "no data" -> 0 (mid-tread 0);
            # a row with no data at all is invalid
            nd = np.zeros(re.shape, dtype=bool)
            nh = fs.n_heavy
            if nh:
                nd[:, :nh] = (re[:, :nh] == -128) | (im[:, :nh] == -128)
            sent = -(1 << (fs.bits - 1))
            nd[:, nh:] = (re[:, nh:] == sent) | (im[:, nh:] == sent)
            if nd.any():
                ring.n_nodata += int(nd[ok].sum())
                re = np.where(nd, 0, re)
                im = np.where(nd, 0, im)
                ok = ok & ~nd.all(axis=1)
        re = np.where(ok[:, None], re, 0).astype(np.int8)
        im = np.where(ok[:, None], im, 0).astype(np.int8)
        ring.codes[s, :nv, slots, 0] = re
        ring.codes[s, :nv, slots, 1] = im
        ring.valid[s, slots] = ok


def k1_twin(codes: np.ndarray, *, c_v: np.ndarray, offset: float, csr: OwnerCsr,
            shifts: np.ndarray, t_lo: int, n_t: int) -> np.ndarray:
    """K1 twin: ``uv [F, n_t, n_act]`` complex64 for output rows
    ``t_lo .. t_lo + n_t - 1``.

    ``codes`` is the stream ring ``int8 [S, V, R, 2]`` (row ``r`` in slot
    ``r % R``); ``c_v [S, V]`` float32 decode constants.
    """
    S, V, R, _ = codes.shape
    F = shifts.shape[0]
    ptr = csr.cell_ptr.astype(np.int64)
    ne = np.diff(ptr)
    n_act = csr.n_active
    acc_re = np.zeros((F, n_t, n_act), dtype=np.float32)
    acc_im = np.zeros((F, n_t, n_act), dtype=np.float32)
    t = t_lo + np.arange(n_t, dtype=np.int64)
    cv = np.asarray(c_v, dtype=np.float32)
    off = np.float32(offset)
    for j in range(int(ne.max(initial=0))):
        has = ne > j
        e = np.where(has, ptr[:-1] + j, 0)
        s = csr.ent_stream[e].astype(np.int64)
        v = csr.ent_value[e].astype(np.int64)
        cj = csr.ent_conj[e].astype(bool)
        c = cv[s, v]
        for f in range(F):
            rows = t[:, None] - shifts[f, s][None, :].astype(np.int64)     # [n_t, n_act]
            slot = np.mod(rows, R)
            cre = codes[s[None, :], v[None, :], slot, 0].astype(np.float32)
            cim = codes[s[None, :], v[None, :], slot, 1].astype(np.float32)
            vre = (cre + off) * c[None, :]
            vim = (cim + off) * c[None, :]
            vim = np.where(cj[None, :], -vim, vim)
            vre = np.where(has[None, :], vre, np.float32(0.0))
            vim = np.where(has[None, :], vim, np.float32(0.0))
            acc_re[f] += vre
            acc_im[f] += vim
    k = (np.float32(0.5) * csr.sign)[None, None, :]
    return (acc_re * k + 1j * (acc_im * k)).astype(np.complex64)


def v4_emulate(dense: np.ndarray, csr_v4, shifts: np.ndarray, *, t_rows: int,
               t_out_len: int, t_lo: int, n_grid: int, fftshift: bool = True) -> np.ndarray:
    """Literal numpy transcription of the ``sparse_combine_v4`` CUDA kernel
    (int accumulation per stream entry, ``0.5f * (float)(acc + accm)``,
    checkerboard, round to half) -- the production oracle for the 1x
    bit-exactness check, on CPU.

    Args:
        dense: ``int8 [n_sb, n_filled_max, t_pad, 2]`` (the
            ``transpose_compact_tmajor`` buffer).
        csr_v4: a production ``SparseCellCsr`` (built on CPU from torch).
    Returns ``float16 [n_fdm, t_out_len, N, N//2+1, 2]`` with inactive cells 0.
    """
    nh = n_grid // 2 + 1
    F = shifts.shape[0]
    out = np.zeros((F, t_out_len, n_grid, nh, 2), dtype=np.float16)
    uw = csr_v4.cell_uw.numpy().astype(np.int64)
    ptr = csr_v4.cell_ptr.numpy().astype(np.int64)
    eg = csr_v4.ent_g.numpy().astype(np.int64)
    ek = csr_v4.ent_k.numpy().astype(np.int64)
    ekm = csr_v4.ent_km.numpy().astype(np.int64)
    d = dense.astype(np.int64)
    t_out = np.arange(t_out_len, dtype=np.int64)
    for a in range(uw.size):
        u, w = divmod(int(uw[a]), nh)
        acc = np.zeros((F, t_out_len, 2), dtype=np.int64)
        accm = np.zeros((F, t_out_len, 2), dtype=np.int64)
        for e in range(ptr[a], ptr[a + 1]):
            g = eg[e]
            ts = t_out[None, :] + t_lo - shifts[:, g][:, None].astype(np.int64)  # [F, T]
            ok = (ts >= 0) & (ts < t_rows)
            tsc = np.clip(ts, 0, d.shape[2] - 1)
            if ek[e] >= 0:
                acc += np.where(ok[..., None], d[g, ek[e]][tsc], 0)
            if ekm[e] >= 0:
                accm += np.where(ok[..., None], d[g, ekm[e]][tsc], 0)
        o_re = np.float32(0.5) * (acc[..., 0] + accm[..., 0]).astype(np.float32)
        o_im = np.float32(0.5) * (acc[..., 1] - accm[..., 1]).astype(np.float32)
        if fftshift and ((u + w) & 1):
            o_re, o_im = -o_re, -o_im
        out[:, :, u, w, 0] = o_re.astype(np.float16)
        out[:, :, u, w, 1] = o_im.astype(np.float16)
    return out
