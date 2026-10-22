"""Independent numpy reference of the 2x corr chain (x2 WS-C deliverable 2).

    raw fada bytes --unpack--> E[t, ch, pol, ant]          complex128
      --mask--> --cal--> float64 GEMM over core cross-baselines
      V[t_q1, k, row] = sum_{t_int natives, csf channels, pols} conj(E_b) E_a
      --stage 1 (per tier, per coarse DM d, q1 = t_int samples)-->
        x_v,d[t] = sum_{(row, k) -> v} V'[t + s1[k, d], k, row]      (V' = conj if flagged)
      --decimate x2 (tier A) / x4 (tier B), aligned to the absolute clock-->
      --static sky (StaticSkyMean semantics, one state per (d, value))-->
      --row rule (SPEC §4): / sqrt(n), robust EMA sigma, mid-rise codes-->
      --frame v2 (SPEC §5)--> datagram bytes

Everything is float64 numpy, written for clarity, and independent of the
GPU code: gridding is geometric (each (row, channel) source is placed in
its value slot with ``np.add.at`` from the uv geometry, never from the
CSR the kernels use); the stage-1 table is recomputed from the oracle's
formula (:mod:`dsart.x2.corr.plan2x`, pinned to
``docs/timeres_study/snr2x/schemes.py``). Only the *contract* objects --
the value order of each stream and the redundancy classes -- come from
the value map, because they define the wire format.

Sizes are free: tests run with a handful of packets per block and a
subset of antennas/channels; nothing here assumes the production shape.

Conventions (today's, verbatim):

* native sample ``n = packet * 2 + t2``; fast-vis sample ``q = n // t_int``;
* ``V(a, b) = conj(E_b) E_a`` for ``b < a`` (F18), ``u = -(E_a-E_b)/lambda``
  (F20); summed-channel frequencies are band-centre means;
* stage 1 references every summed channel to its sub-band's first summed
  channel (``a1 >= 0``); the output time index is the reference
  channel's arrival time in q1 samples;
* decimated sample ``j`` of a tier with ``D = T_native / q1`` covers q1
  samples ``[j*D, (j+1)*D)`` of the absolute clock (block starts are
  multiples of 4096 natives = 512 q1 = 128 tier-B samples).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import LEGACY_FLUFF_SCALE, NATIVE_SAMPLE_US, SPEED_OF_LIGHT_M_S
from dsart.x2.corr.plan2x import CorrPlan2x

__all__ = [
    "FRAME2_HEADER",
    "MAGIC",
    "Frame2Header",
    "RowRuleState",
    "StaticSkyRef",
    "apply_cal",
    "apply_masks",
    "decimate",
    "decode_row_rule",
    "gemm_core",
    "pack_bits_le",
    "pack_frame2",
    "pack_stream_block",
    "quantise_row_rule",
    "quantise_v1_rows",
    "stage1_values",
    "unpack_bits_le",
    "unpack_frame2",
    "unpack_int4_onwire",
    "value_dest_cells",
    "value_dest_classes",
    "value_dest_from_vm",
]

# ---------------------------------------------------------------------------
# unpack / mask / cal / GEMM
# ---------------------------------------------------------------------------


def unpack_int4_onwire(raw: np.ndarray, *, n_packets: int, nants: int, nchan: int) -> np.ndarray:
    """On-wire ``(n_packets, nants, nchan, 2 t, 2 pol)`` int4 bytes ->
    ``E[n, ch, pol, ant]`` complex128 (``n = packet*2 + t``). Real = low
    nibble, imag = high nibble, both sign-extended, x 0.05 (bfCorr fluff)."""
    b = np.asarray(raw, np.uint8).reshape(n_packets, nants, nchan, 2, 2).view(np.int8)
    re = (b << 4).astype(np.int8) >> 4
    im = b >> 4
    e = (re.astype(np.float64) + 1j * im.astype(np.float64)) * float(LEGACY_FLUFF_SCALE)
    # (pkt, ant, ch, t, pol) -> (pkt, t, ch, pol, ant) -> (n, ch, pol, ant)
    return e.transpose(0, 3, 2, 4, 1).reshape(n_packets * 2, nchan, 2, nants)


def apply_masks(
    E: np.ndarray,
    cube_mask: np.ndarray | None = None,
    time_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Zero flagged voltages. ``cube_mask`` ``(ant, ch, pol)``;
    ``time_mask`` ``(n_acc, ch, pol)`` -- every antenna, ``n_acc`` equal
    accumulations tiling the block's native samples."""
    E = E.copy()
    if cube_mask is not None:
        keep = ~np.asarray(cube_mask, bool).transpose(1, 2, 0)       # (ch, pol, ant)
        E *= keep[None]
    if time_mask is not None:
        tm = np.asarray(time_mask, bool)
        n_acc = tm.shape[0]
        per = E.shape[0] // n_acc
        if per * n_acc != E.shape[0]:
            raise ValueError("time_mask accumulations must tile the block")
        keep = ~np.repeat(tm, per, axis=0)                    # (n, ch, pol)
        E *= keep[..., None]
    return E


def apply_cal(E: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """``E * G`` with ``gains`` ``(ant, ch, pol)`` complex."""
    return E * np.asarray(gains).transpose(1, 2, 0)[None]


def gemm_core(E: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray, *, t_int: int, csf: int) -> np.ndarray:
    """``V[q, k, r] = sum conj(E_b) E_a`` over the ``t_int`` natives of fast
    sample ``q``, the ``csf`` channels of summed channel ``k`` and both
    pols (Stokes I), for the rows ``(a, b)``. float64; ``(n_q, nsum, R)``."""
    n, nch, npol, nants = E.shape
    if n % t_int or nch % csf:
        raise ValueError("t_int must divide the samples and csf the channels")
    n_q, nsum = n // t_int, nch // csf
    ants = np.unique(np.concatenate([rows_a, rows_b]))
    pos = np.searchsorted(ants, np.arange(nants))
    X = E[..., ants].reshape(n_q, t_int, nsum, csf, npol, ants.size)
    X = X.transpose(0, 2, 1, 3, 4, 5).reshape(n_q * nsum, t_int * csf * npol, ants.size)
    G = np.conj(X).transpose(0, 2, 1) @ X                      # [i, j] = sum conj(E_i) E_j
    G = G.reshape(n_q, nsum, ants.size, ants.size)
    return G[:, :, pos[rows_b], pos[rows_a]]


# ---------------------------------------------------------------------------
# value placement (geometric, independent of the CSR) + stage 1
# ---------------------------------------------------------------------------


def value_dest_cells(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    nu_ghz: np.ndarray,
    *,
    cell_lambda: float,
    n_grid: int,
    cell_row: np.ndarray,
    cell_col: np.ndarray,
) -> np.ndarray:
    """Gridded cells, from the geometry alone: the stream-local value index
    of each (row, channel) source, ``(R, n_chan)`` (-1 = off-grid). Rows
    are baselines ``(a, b)``; ``nu_ghz`` the summed-channel frequencies;
    ``cell_row/col`` the stream's cells in wire order (the contract)."""
    lam = SPEED_OF_LIGHT_M_S / (np.asarray(nu_ghz, np.float64) * 1e9)
    du = np.asarray(antpos_e, np.float64)[a] - np.asarray(antpos_e, np.float64)[b]
    dv = np.asarray(antpos_n, np.float64)[a] - np.asarray(antpos_n, np.float64)[b]
    half = n_grid // 2
    col = np.rint(-(du[:, None] / lam[None, :]) / cell_lambda).astype(np.int64) + half
    row = np.rint(-(dv[:, None] / lam[None, :]) / cell_lambda).astype(np.int64) + half
    ok = (row >= 0) & (row < n_grid) & (col >= 0) & (col < n_grid)
    key = np.where(ok, (row << 16) | col, -1)
    vkeys = (np.asarray(cell_row, np.int64) << 16) | np.asarray(cell_col, np.int64)
    srt = np.argsort(vkeys, kind="stable")
    pos = np.clip(np.searchsorted(vkeys[srt], key), 0, max(vkeys.size - 1, 0))
    hit = ok & (vkeys[srt][pos] == key)
    return np.where(hit, srt[pos], -1)


def value_dest_classes(
    bl_class: np.ndarray, bl_conj: np.ndarray, class_id_wire: np.ndarray, n_chan: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-baseline values: every (row, channel) source goes to its
    redundancy class's value, conjugated where the member is reversed."""
    vidx = np.full(int(np.max(bl_class)) + 1, -1, np.int64)
    vidx[np.asarray(class_id_wire, np.int64)] = np.arange(len(class_id_wire))
    dest = np.repeat(vidx[np.asarray(bl_class, np.int64)][:, None], n_chan, axis=1)
    conj = np.repeat(np.asarray(bl_conj, bool)[:, None], n_chan, axis=1)
    return dest, conj


def value_dest_from_vm(vm, stream: int, antpos_e, antpos_n) -> tuple[np.ndarray, np.ndarray]:
    """``(dest, conj)`` ``(R, per)`` for one stream of a WS-P
    :class:`dsart.grid.baseline_grid.ValueMap`; rows are its ``kept_bl``
    (core cross-baselines, xGPU order). Cells come from the geometry, not
    from the map's CSR."""
    from dsart.grid.baseline_grid import kept_core_baselines  # noqa: PLC0415
    st = vm.streams[stream]
    per = vm.nchan_eff // vm.n_sub
    chans = np.arange(st.sub_band * per, (st.sub_band + 1) * per)
    if vm.kind == "gridded":
        mask = np.zeros(antpos_e.shape[0] * (antpos_e.shape[0] + 1) // 2, bool)
        mask[vm.kept_bl] = True
        _, a, b = kept_core_baselines(mask, antpos_e.shape[0])
        nu = build_summed_chgroup_freq_table_GHz(vm.chan_sum_factor)[st.chgroup][chans]
        dest = value_dest_cells(antpos_e, antpos_n, a, b, nu, cell_lambda=vm.cell_lambda,
                                n_grid=vm.n_grid, cell_row=st.cell_row, cell_col=st.cell_col)
        return dest, np.zeros(dest.shape, bool)
    c = vm.classes
    return value_dest_classes(c.bl_class, c.bl_conj, st.class_id, per)


def stage1_values(
    V: np.ndarray,
    dest: np.ndarray,
    conj: np.ndarray,
    *,
    ks: np.ndarray,
    s1: np.ndarray,
    n_values: int,
    t0: int,
    n_out: int,
) -> np.ndarray:
    """Stage 1 for one stream and ONE coarse DM (q1 resolution).

    ``x[t, v] = sum_{(r, k): dest[r, k] = v} V'[t0 + t + s1[k], k, r]`` for
    ``t in [0, n_out)``; ``V' = conj(V)`` where ``conj``. ``ks`` are the
    stream's summed channels, ``s1`` their shifts (q1 units). Placement by
    ``np.add.at`` (order-free, float64)."""
    R = V.shape[2]
    out = np.zeros((n_values, n_out), np.complex128)
    for j, k in enumerate(ks):
        seg = V[t0 + int(s1[j]): t0 + int(s1[j]) + n_out, k, :]      # (n_out, R)
        if seg.shape[0] != n_out:
            raise ValueError("stage-1 window runs past the available samples")
        seg = np.where(conj[:, j][None, :], np.conj(seg), seg)
        d = dest[:, j]
        ok = d >= 0
        np.add.at(out, d[ok], seg[:, ok].T)
    del R
    return out.T                                                      # (n_out, V)


def decimate(x: np.ndarray, D: int) -> np.ndarray:
    """Sum ``D`` consecutive q1 samples (axis 0) -> one stream sample."""
    n = x.shape[0] // D
    return x[: n * D].reshape(n, D, *x.shape[1:]).sum(axis=1)


# ---------------------------------------------------------------------------
# static sky (StaticSkyMean semantics, float64)
# ---------------------------------------------------------------------------


@dataclass
class StaticSkyRef:
    """Strictly causal sliding mean of per-block cell means: a block is
    passed through while fewer than ``warmup_cubes`` blocks have been
    seen, otherwise the mean of the previous ``window_blocks`` blocks'
    per-cell means is subtracted; then this block's mean joins the window
    (exactly ``StaticSkyMean.apply``, one state per key)."""

    window_blocks: int = 8
    warmup_cubes: int = 8
    _ring: dict = field(default_factory=dict, repr=False)
    _seen: dict = field(default_factory=dict, repr=False)

    def apply(self, key, x: np.ndarray) -> np.ndarray:
        ring = self._ring.setdefault(key, [])
        seen = self._seen.get(key, 0)
        if not ring or seen < self.warmup_cubes:
            out = x.copy()
        else:
            out = x - np.mean(ring, axis=0)[None, :]
        ring.append(x.mean(axis=0))
        if len(ring) > self.window_blocks:
            ring.pop(0)
        self._seen[key] = seen + 1
        return out


# ---------------------------------------------------------------------------
# row rule (SPEC §4) and v1 cint8
# ---------------------------------------------------------------------------

S_MAIN = {4: 0.336, 5: 0.191, 6: 0.106}
"""Main step in sigma units per bits-per-component (SPEC §4; Phase 0 confirms)."""


@dataclass
class RowRuleState:
    """Per (stream, coarse DM) robust sigma (EMA), SPEC §4 step 2:
    1.4826 x the LOWER median of ``|re|, |im|`` over every ``stride``-th
    sample of the block, excluding components above ``clip_sigma`` x the
    previous sigma; EMA with ``tau_s``. A block is quantised with the
    previous sigma (its own estimate for the first block)."""

    tau_s: float = 10.0
    block_s: float = 4096 * NATIVE_SAMPLE_US * 1e-6
    clip_sigma: float = 6.0
    stride: int = 4
    sigma: dict = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return 1.0 - math.exp(-self.block_s / self.tau_s)

    def robust(self, y: np.ndarray, prev: float | None) -> float:
        ys = y[:: self.stride]
        comp = np.abs(np.stack([ys.real, ys.imag], axis=-1)).ravel()
        if prev is not None:
            comp = comp[comp <= self.clip_sigma * prev]
        if comp.size == 0:
            return float(prev) if prev is not None else 1.0
        k = (comp.size - 1) // 2
        return float(1.4826 * np.partition(comp, k)[k])

    def step(self, key, y: np.ndarray) -> float:
        """Sigma used to quantise this block, then advance the EMA."""
        prev = self.sigma.get(key)
        est = self.robust(y, prev)
        use = est if prev is None else prev
        self.sigma[key] = est if prev is None else prev + self.alpha * (est - prev)
        return max(use, 1e-30)


def quantise_row_rule(
    x: np.ndarray,
    n: np.ndarray,
    n_heavy: int,
    sigma: float,
    *,
    bits: int,
    s_heavy: float = 0.1,
    s_main: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """SPEC §4 rows ``x[t, v]`` (one stream, one coarse DM) ->
    ``(heavy_codes int8 (t, n_heavy, 2), main_codes int (t, V-n_heavy, 2),
    step_main, step_heavy)``; codes are mid-rise ``floor(y / step)`` clipped
    to the signed range, ``y = x / sqrt(n)``."""
    y = x / np.sqrt(np.asarray(n, np.float64))[None, :]
    s_m = S_MAIN[bits] if s_main is None else s_main
    step_m, step_h = s_m * sigma, s_heavy * sigma
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    ri = np.stack([y.real, y.imag], axis=-1)
    h = np.clip(np.floor(ri[:, :n_heavy] / step_h), -128, 127).astype(np.int8)
    m = np.clip(np.floor(ri[:, n_heavy:] / step_m), lo, hi).astype(np.int16)
    return h, m, step_m, step_h


def decode_row_rule(h, m, n, n_heavy, *, bits, s_heavy=0.1, s_main=None) -> np.ndarray:
    """Owner-side decode in sigma units, re-weighted by sqrt(n) (SPEC §4.5)."""
    s_m = S_MAIN[bits] if s_main is None else s_main
    vh = (h.astype(np.float64) + 0.5) * s_heavy
    vm = (m.astype(np.float64) + 0.5) * s_m
    v = np.concatenate([vh, vm], axis=1)
    return (v[..., 0] + 1j * v[..., 1]) * np.sqrt(np.asarray(n, np.float64))[None, :]


def quantise_v1_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Today's v1 cint8 rule (``TransportTx._encode_rows``), from the
    float32 values: per row ``amax = max |re|,|im|``, ``scale =
    f32(f64(amax)/127)`` (1.0 if amax == 0), ``inv = f32(1/f64(scale))``,
    ``q = clip(rint(f32(x) * inv), -128, 127)``. Returns ``(q (t, V, 2)
    int8, scales (t,) float32)``."""
    a = np.asarray(rows).astype(np.complex64)
    ri = np.stack([a.real, a.imag], axis=-1).astype(np.float32)
    amax = np.abs(ri).reshape(ri.shape[0], -1).max(axis=1) if ri.size else np.zeros(ri.shape[0], np.float32)
    scales = (amax.astype(np.float64) / 127.0).astype(np.float32)
    scales[amax == 0.0] = np.float32(1.0)
    inv = (1.0 / scales.astype(np.float64)).astype(np.float32)
    q = np.clip(np.round(ri * inv[:, None, None]), -128.0, 127.0).astype(np.int8)
    return q, scales


# ---------------------------------------------------------------------------
# bit packing + frame v2 (SPEC §5)
# ---------------------------------------------------------------------------


def pack_bits_le(codes: np.ndarray, bits: int) -> bytes:
    """Little-endian bit stream of ``bits``-bit two's-complement codes in
    the given (flattened) order; padded with zeros to a byte."""
    c = np.asarray(codes, np.int64).ravel() & ((1 << bits) - 1)
    nbits = c.size * bits
    bitarr = ((c[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.uint8).ravel()
    pad = (-nbits) % 8
    if pad:
        bitarr = np.concatenate([bitarr, np.zeros(pad, np.uint8)])
    return np.packbits(bitarr.reshape(-1, 8), axis=1, bitorder="little").tobytes()


def unpack_bits_le(buf: bytes, bits: int, count: int) -> np.ndarray:
    arr = np.unpackbits(np.frombuffer(buf, np.uint8), bitorder="little")[: count * bits]
    vals = (arr.reshape(count, bits).astype(np.int64) << np.arange(bits)[None, :]).sum(1)
    sign = 1 << (bits - 1)
    return np.where(vals & sign, vals - (1 << bits), vals)


MAGIC = 0xD5A1107E
FRAME2_HEADER = struct.Struct("<IHHQQHHBBBBHBBIQIffI8s")
assert FRAME2_HEADER.size == 72


@dataclass(frozen=True)
class Frame2Header:
    flags: int
    seq_first: int
    t0_native: int
    stream_id: int
    dm_idx: int
    k: int
    n_sub: int
    bits_main: int
    bits_heavy: int
    n_grid: int
    t_int_factor: int
    value_kind: int
    n_values: int
    pattern_id: int
    n_heavy: int
    step_main: float
    step_heavy: float
    payload_bytes: int
    version: int = 2
    magic: int = MAGIC


def pack_frame2(h: Frame2Header, payload: bytes) -> bytes:
    if len(payload) != h.payload_bytes:
        raise ValueError("payload_bytes mismatch")
    return FRAME2_HEADER.pack(
        h.magic, h.version, h.flags, h.seq_first, h.t0_native, h.stream_id, h.dm_idx,
        h.k, h.n_sub, h.bits_main, h.bits_heavy, h.n_grid, h.t_int_factor, h.value_kind,
        h.n_values, h.pattern_id & 0xFFFF_FFFF_FFFF_FFFF, h.n_heavy, h.step_main,
        h.step_heavy, h.payload_bytes, b"\0" * 8,
    ) + payload


def unpack_frame2(buf: bytes) -> tuple[Frame2Header, bytes]:
    f = FRAME2_HEADER.unpack_from(buf, 0)
    h = Frame2Header(
        magic=f[0], version=f[1], flags=f[2], seq_first=f[3], t0_native=f[4],
        stream_id=f[5], dm_idx=f[6], k=f[7], n_sub=f[8], bits_main=f[9], bits_heavy=f[10],
        n_grid=f[11], t_int_factor=f[12], value_kind=f[13], n_values=f[14], pattern_id=f[15],
        n_heavy=f[16], step_main=f[17], step_heavy=f[18], payload_bytes=f[19],
    )
    return h, bytes(buf[72: 72 + h.payload_bytes])


FLAG_HEAVY_PRESENT = 1 << 5
MAX_PAYLOAD = 8900


def sample_bytes(n_values: int, n_heavy: int, bits_main: int) -> int:
    """Bytes per sample: heavy cint8 pairs + the byte-padded main stream."""
    return 2 * n_heavy + ((n_values - n_heavy) * bits_main + 7) // 8


def pack_stream_block(
    h_codes: np.ndarray,
    m_codes: np.ndarray,
    *,
    bits_main: int,
    t0_native: int,
    T_native: int,
    stream_id: int,
    dm_idx: int,
    n_sub: int,
    n_grid: int,
    value_kind: int,
    pattern_id: int,
    step_main: float,
    step_heavy: float,
    flags: int = 0,
) -> list[bytes]:
    """Datagrams for one (stream, coarse DM) block of rows: ``k =
    floor(8900 / bytes_per_sample)`` samples each (the last may be
    shorter). ``bits_main`` is per complex value (8/10/12)."""
    n_t, n_heavy = h_codes.shape[0], h_codes.shape[1]
    n_values = n_heavy + m_codes.shape[1]
    b_comp = bits_main // 2
    sb = sample_bytes(n_values, n_heavy, bits_main)
    k = max(1, MAX_PAYLOAD // sb)
    if n_heavy:
        flags |= FLAG_HEAVY_PRESENT
    out = []
    for t_start in range(0, n_t, k):
        t_end = min(n_t, t_start + k)
        parts = []
        for t in range(t_start, t_end):
            parts.append(np.asarray(h_codes[t], np.int8).tobytes())
            parts.append(pack_bits_le(m_codes[t], b_comp))
        payload = b"".join(parts)
        t0 = int(t0_native) + t_start * int(T_native)
        out.append(pack_frame2(Frame2Header(
            flags=flags, seq_first=t0 // int(T_native), t0_native=t0,
            stream_id=stream_id, dm_idx=dm_idx, k=t_end - t_start, n_sub=n_sub,
            bits_main=bits_main, bits_heavy=16 if n_heavy else 0, n_grid=n_grid,
            t_int_factor=T_native, value_kind=value_kind, n_values=n_values,
            pattern_id=pattern_id, n_heavy=n_heavy, step_main=float(step_main),
            step_heavy=float(step_heavy), payload_bytes=len(payload),
        ), payload))
    return out


# ---------------------------------------------------------------------------
# block-level driver (2-block window, one-block latency)
# ---------------------------------------------------------------------------


@dataclass
class Corr2xRef:
    """The whole corr chain for one chgroup, block by block, mirroring the
    GPU orchestrator's framing: block ``k``'s GEMM output joins block
    ``k-1``'s in a 2-block window; stage 1 then emits block ``k-1`` for
    every coarse DM of every tier (the first call emits nothing).

    ``streams[s] = (dest, conj, n_values)`` from :func:`value_dest_cells` /
    :func:`value_dest_classes` / :func:`value_dest_from_vm`."""

    plan: CorrPlan2x
    chgroup: int
    streams: list
    static_sky: StaticSkyRef | None = None
    _prev: np.ndarray | None = field(default=None, repr=False)

    def push(self, V: np.ndarray, *, block_start_q1: int) -> dict | None:
        """``V`` = this block's GEMM output ``(n_q, nsum, R)``; returns
        ``{(d, s): (x_dec (n_dec, V_s) complex, t0_native)}`` for the
        previous block, or None on the first call."""
        prev, self._prev = self._prev, V
        if prev is None:
            return None
        n_q = prev.shape[0]
        joined = np.concatenate([prev, V], axis=0)
        start = block_start_q1 - n_q                                   # absolute q1 of prev[0]
        out = {}
        per = self.plan.per
        for ti, tier in enumerate(self.plan.tiers):
            s1_all = self.plan.stage1_bins(ti)[self.chgroup]           # (nsum, n_c)
            D = tier.decim
            if start % D or n_q % D:
                raise ValueError("blocks must align to the tier decimation")
            for c in range(tier.n_coarse):
                d = tier.d0 + c
                for s, (dest, conj, n_values) in enumerate(self.streams):
                    ks = np.arange(s * per, (s + 1) * per)
                    x = stage1_values(joined, dest, conj, ks=ks, s1=s1_all[ks, c],
                                      n_values=n_values, t0=0, n_out=n_q)
                    xd = decimate(x, D)
                    if self.static_sky is not None:
                        xd = self.static_sky.apply((d, s), xd)
                    out[(d, s)] = (xd, start * tier.q1_native)
        return out
