"""Independent numpy reference of the 2x corr chain (x2 WS-C deliverable 2).

    raw fada bytes --unpack--> E[t, ch, pol, ant]          complex128
      --mask--> --cal--> float64 GEMM over core cross-baselines
      V[t_q1, k, row] = sum_{t_int natives, csf channels, pols} conj(E_b) E_a
      --stage 1 (per tier, per coarse DM d, q1 = t_int samples)-->
        x_v,d[t] = sum_{(row, k) -> v} V'[t + s1[k, d], k, row]      (V' = conj if flagged)
      --decimate x2 (tier A) / x4 (tier B), aligned to the absolute clock-->
      --static sky (StaticSkyMean semantics, one state per (d, value))-->
      --row rule (SPEC §4, WS-T's dsart.x2.quant reference)-->
      --frame v2 (SPEC §5, WS-T's prod_frame2)--> datagram bytes

Everything is float64 numpy, written for clarity, and independent of the
GPU code: gridding is geometric (each (row, channel) source is placed in
its value slot with ``np.add.at`` from the uv geometry, never from the
CSR the kernels use); the stage-1 table is the tiered plan's, which the
tests pin to ``docs/timeres_study/snr2x/schemes.py`` and to an independent
restatement of a1. Only the *contract* objects -- the value order of each
stream and the redundancy classes -- come from the value map, because they
define the wire format; the row rule and the frame codec are WS-T's
references (``dsart.x2.quant``, ``dsart.transport.prod_frame2``), called,
not restated. The GPU stores the fast visibilities as cfp16 (Phase 0:
S/N factor 1 - 2e-8), the reference keeps float64.

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

from dataclasses import dataclass, field

import numpy as np

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import LEGACY_FLUFF_SCALE, SPEED_OF_LIGHT_M_S
from dsart.x2.corr.plan2x import CorrPlan2x

__all__ = [
    "Corr2xRef",
    "StaticSkyRef",
    "apply_cal",
    "apply_masks",
    "decimate",
    "frames_for_block",
    "gemm_core",
    "pack_bits_le",
    "payload_row",
    "quantise_flow",
    "quantise_v1_rows",
    "sample_bytes",
    "stage1_values",
    "unpack_bits_le",
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
    size = int(max(np.max(bl_class), np.max(class_id_wire) if len(class_id_wire) else 0)) + 1
    vidx = np.full(size, -1, np.int64)
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

def quantise_flow(blocks, n_contrib, n_heavy: int, rule) -> list:
    """SPEC §4 on one (stream, coarse DM) flow, block by block, by WS-T's
    reference (:class:`dsart.x2.quant.RowQuantiser`; the row rule is WS-T's,
    this reference does not restate it). ``blocks``: iterable of ``(S, V)``
    complex arrays. Returns ``quant.BlockCodes`` per block."""
    from dsart.x2.quant import RowQuantiser  # noqa: PLC0415
    q = RowQuantiser(rule=rule, n_contrib=np.asarray(n_contrib), n_heavy=int(n_heavy))
    return [q.process_block(np.asarray(x)) for x in blocks]


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


def sample_bytes(n_values: int, n_heavy: int, bits_main: int) -> int:
    """Bytes per stream sample: heavy cint8 pairs + the byte-padded main stream."""
    return 2 * n_heavy + ((n_values - n_heavy) * bits_main + 7) // 8


def payload_row(codes_heavy: np.ndarray, codes_main: np.ndarray, comp_bits: int) -> bytes:
    """One stream sample's payload (SPEC §5): heavy (re, im) int8 pairs, then
    the main codes as one little-endian bit stream, re then im per value."""
    return (np.asarray(codes_heavy, np.int8).tobytes()
            + pack_bits_le(np.asarray(codes_main), comp_bits))


def frames_for_block(codes_heavy, codes_main, **header) -> list:
    """Datagrams of one corr block of one flow, through WS-T's frame-v2
    codec (``dsart.transport.prod_frame2.iter_block_datagrams``)."""
    from dsart.transport import prod_frame2  # noqa: PLC0415
    return list(prod_frame2.iter_block_datagrams(codes_heavy, codes_main, **header))


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
