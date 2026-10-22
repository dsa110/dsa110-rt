"""Corner-turn value maps for the 2x search (docs/x2/SPEC.md §3).

One builder, :func:`build_value_map`, gives for every (chgroup, sub-band)
stream the ordered list of values the corr node sends and the owner (search
half) images, for either value kind:

* ``gridded``  -- a value is ONE uv cell, exactly today's cell assignment:
  the sub-band's cells are :func:`dsart.grid.sparsity_pattern.build_pattern`
  (sub_band=(s, n_sub)) and each (core cross-baseline, summed channel) source
  maps to its nearest cell at that channel's frequency with
  ``FastVisGridder.from_pattern`` arithmetic (F20 negation, lambda-uniform,
  no DEC projection -- DEC enters only ``pattern_id``, as today);
* ``baselines`` -- a value is ONE exact 3-D redundancy class of core
  cross-baselines (vectors agree within ``redundancy_tol_m`` incl. height),
  summed over the sub-band's channels with a per-member conjugation flag;
  the owner places it at the cell nearest the class's mean (E, N) vector at
  the sub-band CENTRE frequency (same cell convention as ``gridded``).

Per value: the static contribution count ``n`` (number of (baseline, summed
channel) sources), the owner cell ``(row, col)`` in today's ``[n_grid,
n_grid]`` single-side layout (``SparsityPattern.ix_row/ix_col`` semantics,
flat index ``row * n_grid + col``), and the heavy flag
(``n >= heavy_frac * N_image``, N_image = sum of n over all values of all
streams). Order within a stream: heavy values first, then main values; each
group in cell order (key ``(row << 16) | col``; ties -- baselines only --
by class id). ``heavy_frac=None`` (1x) has no heavy class, so a 1x gridded
map is ordered exactly like today's patterns.

CSR forms:

* corr side (:meth:`ValueMap.corr_csr`): value <- sources ``(bl, ch, conj)``,
  ``bl`` the xGPU upper-triangle baseline index (0..NBASE-1, b <= a),
  ``ch`` the summed channel within the chgroup, sources in ascending
  ``bl * nchan_eff + ch`` order (a fixed fp32 summation order);
* owner side (:attr:`ValueMap.owner`): the Hermitian half-spectrum today's
  real-FFT combine writes (``tiled_combine_cuda.build_sparse_cell_csr``):
  half cell ``(row, col)``, ``col in [0, n_grid/2]``, flat
  ``row * (n_grid/2 + 1) + col``, receives ``0.5 * sum(weight * (conj ? conj(x)
  : x))`` over its entries, where a value at full-grid cell (row, col) has a
  direct entry at (row, col) if ``col <= n_grid/2`` and a conj entry at the
  mirror ``((n_grid-row) % n_grid, (n_grid-col) % n_grid)`` if that column is
  ``<= n_grid/2``. ``irfft2(G_h, s=(N, N))`` then equals ``Re(ifft2(G))`` of the
  full grid. The 0.5 and the fftshift checkerboard are kernel constants, not
  in the CSR. ``weight = sqrt(n)`` when values travel normalised by
  ``1/sqrt(n)`` (2x), else 1 (1x codes are summed unweighted, as today).

``pattern_id``: 64-bit blake2b over every input (kind, antpos E/N/U, core
mask, chgroup table, csf, n_sub, n_grid, cell_lambda, quantised DEC,
redundancy tolerance, heavy fraction, normalisation); per-stream ids append
the stream index.

Importing this module pulls in torch through ``dsart.grid.__init__``; the
builder itself is numpy-only.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from dsart.common.constants import N_CHGROUP, NANTS, NBASE, NCHAN_PER_CHGROUP, freq_GHz
from dsart.grid.sparsity_pattern import (
    SPEED_OF_LIGHT_M_PER_S,
    build_pattern,
    compute_antpos_hash,
    compute_chgroup_table_hash,
    quantise_dec_deg,
)

__all__ = [
    "CorrCsr",
    "OwnerCsr",
    "RedundancyClasses",
    "StreamValues",
    "VALUE_MAP_SCHEMA_VERSION",
    "ValueMap",
    "build_owner_csr",
    "build_value_map",
    "build_value_map_for_opoint",
    "heavy_threshold",
    "kept_core_baselines",
    "redundancy_classes",
    "value_map_pattern_id",
]

VALUE_MAP_SCHEMA_VERSION = 1
_PERSON = b"dsart-x2-vmap"          # blake2b personalisation (<= 16 bytes)
_KINDS = ("gridded", "baselines")


# ---------------------------------------------------------------------------------------
# Baselines and redundancy
# ---------------------------------------------------------------------------------------
def kept_core_baselines(is_core_baseline_mask: np.ndarray | None, nants: int = NANTS):
    """``(bl, a, b)`` int32 of the core cross-baselines in xGPU order (a outer, b < a).

    Same order and selection as ``sparsity_pattern._per_baseline_uv_meters``.
    """
    a_list, b_list = np.tril_indices(nants)            # row-major: a outer, b = 0..a
    nbase = nants * (nants + 1) // 2
    keep = a_list != b_list
    if is_core_baseline_mask is not None:
        mask = np.asarray(is_core_baseline_mask, dtype=bool)
        if mask.shape != (nbase,):
            raise ValueError(f"is_core_baseline_mask shape {mask.shape} != ({nbase},)")
        keep = keep & mask
    bl = np.flatnonzero(keep).astype(np.int32)
    return bl, a_list[keep].astype(np.int32), b_list[keep].astype(np.int32)


@dataclass(frozen=True)
class RedundancyClasses:
    """Exact 3-D redundancy classes of the kept (core cross) baselines.

    Kept baseline i is xGPU baseline ``kept_bl[i]`` = (ant_a[i], ant_b[i]) with natural
    vector ``pos[a] - pos[b]`` (E, N, U). Class ids follow the first member (lowest
    kept index), whose natural orientation is the class's canonical orientation;
    ``bl_conj[i]`` is True when member i's natural vector is the NEGATIVE of the
    canonical one (its visibility is conjugated before summing).
    """

    tol_m: float
    kept_bl: np.ndarray        # (n_kept,) int32
    ant_a: np.ndarray          # (n_kept,) int32
    ant_b: np.ndarray          # (n_kept,) int32
    bl_class: np.ndarray       # (n_kept,) int32
    bl_conj: np.ndarray        # (n_kept,) bool
    class_ptr: np.ndarray      # (n_class + 1,) int32
    class_members: np.ndarray  # (n_kept,) int32 kept indices, grouped by class, ascending
    class_uvw_m: np.ndarray    # (n_class, 3) float64 mean member vector, canonical orientation
    multiplicity: np.ndarray   # (n_class,) int32
    max_diameter_m: float      # largest within-class member separation (<= tol_m)
    min_gap_m: float           # smallest separation between different classes (> tol_m)

    @property
    def n_kept(self) -> int:
        return int(self.kept_bl.shape[0])

    @property
    def n_class(self) -> int:
        return int(self.multiplicity.shape[0])


def redundancy_classes(antpos_e, antpos_n, antpos_u=None, *, is_core_baseline_mask,
                       tol_m: float = 0.05) -> RedundancyClasses:
    """Group the kept baselines into exact-redundancy classes (single linkage at tol_m,
    3-D Euclidean, either orientation). Raises if a class's diameter exceeds tol_m
    (chaining: the classes would depend on the linkage rule)."""
    e = np.asarray(antpos_e, dtype=np.float64)
    n = np.asarray(antpos_n, dtype=np.float64)
    u = np.zeros_like(e) if antpos_u is None else np.asarray(antpos_u, dtype=np.float64)
    if not (e.shape == n.shape == u.shape) or e.ndim != 1:
        raise ValueError("antpos_e/n/u must be equal-length 1-D arrays")
    bl, a, b = kept_core_baselines(is_core_baseline_mask, e.shape[0])
    V = np.stack([e[a] - e[b], n[a] - n[b], u[a] - u[b]], axis=1)          # (n_kept, 3)
    nk = V.shape[0]
    if nk == 0:
        raise ValueError("no kept baselines")
    if np.any(np.linalg.norm(V, axis=1) <= tol_m):
        raise ValueError("a baseline shorter than tol_m cannot be oriented")
    tol2 = tol_m * tol_m
    nbr_j, nbr_sign, nbr_i = [], [], []
    gap2 = np.inf
    for i0 in range(0, nk, 512):
        Vi = V[i0:i0 + 512]
        for sign in (1, -1):
            d2 = ((Vi[:, None, :] - sign * V[None, :, :]) ** 2).sum(-1)       # (chunk, nk)
            if sign == 1:
                d2[np.arange(Vi.shape[0]), np.arange(i0, i0 + Vi.shape[0])] = np.inf
            ii, jj = np.nonzero(d2 <= tol2)
            nbr_i.append(ii + i0)
            nbr_j.append(jj)
            nbr_sign.append(np.full(ii.shape, sign, dtype=np.int8))
            far = d2[d2 > tol2]
            if far.size:
                gap2 = min(gap2, float(far.min()))
    nbr_i = np.concatenate(nbr_i)
    nbr_j = np.concatenate(nbr_j)
    nbr_sign = np.concatenate(nbr_sign)
    order = np.lexsort((nbr_j, nbr_i))
    nbr_i, nbr_j, nbr_sign = nbr_i[order], nbr_j[order], nbr_sign[order]
    ptr = np.searchsorted(nbr_i, np.arange(nk + 1))
    cls = np.full(nk, -1, dtype=np.int64)
    orient = np.zeros(nk, dtype=np.int8)                 # +1 natural == canonical, -1 conj
    n_class = 0
    for root in range(nk):
        if cls[root] >= 0:
            continue
        cls[root] = n_class
        orient[root] = 1
        stack = [root]
        while stack:
            i = stack.pop()
            for k in range(ptr[i], ptr[i + 1]):
                j, s = int(nbr_j[k]), int(nbr_sign[k])
                oj = orient[i] * s
                if cls[j] < 0:
                    cls[j] = n_class
                    orient[j] = oj
                    stack.append(j)
                elif cls[j] != n_class or orient[j] != oj:
                    raise ValueError(f"inconsistent redundancy linkage at kept baselines {i}, {j}")
        n_class += 1
    members = np.argsort(cls, kind="stable").astype(np.int32)
    mult = np.bincount(cls, minlength=n_class).astype(np.int32)
    class_ptr = np.zeros(n_class + 1, dtype=np.int32)
    class_ptr[1:] = np.cumsum(mult)
    oriented = V * orient[:, None]
    uvw = np.zeros((n_class, 3), dtype=np.float64)
    diam = 0.0
    for c in range(n_class):
        m = members[class_ptr[c]:class_ptr[c + 1]]
        vm = oriented[m]
        uvw[c] = vm.mean(axis=0)
        if m.size > 1:
            diam = max(diam, float(np.sqrt(((vm[:, None, :] - vm[None, :, :]) ** 2).sum(-1).max())))
    if diam > tol_m:
        raise ValueError(f"redundancy class diameter {diam:.4f} m > tol {tol_m} m (linkage chains)")
    return RedundancyClasses(
        tol_m=float(tol_m), kept_bl=bl, ant_a=a, ant_b=b, bl_class=cls.astype(np.int32),
        bl_conj=orient < 0, class_ptr=class_ptr, class_members=members, class_uvw_m=uvw,
        multiplicity=mult, max_diameter_m=diam, min_gap_m=float(np.sqrt(gap2)))


# ---------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamValues:
    """The values of one (chgroup, sub-band) stream, in wire order."""

    stream: int                # g * n_sub + s
    chgroup: int
    sub_band: int
    cell_row: np.ndarray       # (n_values,) uint16, full-grid row (v axis)
    cell_col: np.ndarray       # (n_values,) uint16, full-grid col (u axis)
    n_contrib: np.ndarray      # (n_values,) int32 static count n
    heavy: np.ndarray          # (n_values,) bool, True exactly for [0, n_heavy)
    class_id: np.ndarray       # (n_values,) int32 redundancy class (baselines) / -1 (gridded)
    legacy_pattern_id: int | None   # gridded: today's build_pattern id of this sub-band

    @property
    def n_values(self) -> int:
        return int(self.cell_row.shape[0])

    @property
    def n_heavy(self) -> int:
        return int(np.count_nonzero(self.heavy))

    @property
    def sqrt_n(self) -> np.ndarray:
        return np.sqrt(self.n_contrib.astype(np.float64)).astype(np.float32)

    @property
    def inv_sqrt_n(self) -> np.ndarray:
        """Corr-side normalisation (2x): value = sum(sources) / sqrt(n)."""
        return (1.0 / np.sqrt(self.n_contrib.astype(np.float64))).astype(np.float32)


@dataclass(frozen=True)
class CorrCsr:
    """Corr side, one stream: value v <- sources ``ptr[v]:ptr[v+1]``."""

    stream: int
    ptr: np.ndarray            # (n_values + 1,) int32
    bl: np.ndarray             # (n_src,) int32 xGPU baseline index
    ch: np.ndarray             # (n_src,) int32 summed channel within the chgroup
    conj: np.ndarray           # (n_src,) bool

    @property
    def n_src(self) -> int:
        return int(self.bl.shape[0])


@dataclass(frozen=True)
class OwnerCsr:
    """Owner side: half-spectrum cell <- entries (see module docstring)."""

    n_grid: int
    cell_uw: np.ndarray        # (n_active,) int32, row * (n_grid//2 + 1) + col, ascending
    cell_ptr: np.ndarray       # (n_active + 1,) int32
    ent_stream: np.ndarray     # (n_ent,) int32
    ent_value: np.ndarray      # (n_ent,) int32 index into the stream's values
    ent_conj: np.ndarray       # (n_ent,) uint8: 0 direct, 1 conj (mirror fold)
    ent_weight: np.ndarray     # (n_ent,) float32: sqrt(n) (normalised) or 1

    @property
    def n_half(self) -> int:
        return self.n_grid // 2 + 1

    @property
    def n_active(self) -> int:
        return int(self.cell_uw.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.ent_stream.shape[0])


@dataclass(frozen=True)
class ValueMap:
    """Every stream's values + the owner CSR + provenance. Build with :func:`build_value_map`."""

    kind: str
    n_sub: int
    chan_sum_factor: int
    n_grid: int
    cell_lambda: float
    dec_deg_quant: float
    normalise: bool
    heavy_frac: float | None
    heavy_threshold: int | None    # n >= this is heavy (= ceil(heavy_frac * n_image))
    n_image: int
    redundancy_tol_m: float
    pattern_id: int
    stream_pattern_ids: tuple[int, ...]
    streams: tuple[StreamValues, ...]
    owner: OwnerCsr
    classes: RedundancyClasses | None
    kept_bl: np.ndarray
    n_dropped_offgrid: int         # baselines kind: (class, stream) pairs off the grid
    _inputs: dict = field(repr=False)

    @property
    def n_streams(self) -> int:
        return len(self.streams)

    @property
    def nchan_eff(self) -> int:
        return NCHAN_PER_CHGROUP // self.chan_sum_factor

    @property
    def n_values(self) -> np.ndarray:
        return np.array([s.n_values for s in self.streams], dtype=np.int32)

    @property
    def n_heavy(self) -> np.ndarray:
        return np.array([s.n_heavy for s in self.streams], dtype=np.int32)

    def cell_flat(self, stream: int) -> np.ndarray:
        """Today's LUT form ``row * n_grid + col`` (int32) of a stream's values."""
        s = self.streams[stream]
        return s.cell_row.astype(np.int32) * self.n_grid + s.cell_col.astype(np.int32)

    def corr_csr(self, stream: int) -> CorrCsr:
        """Corr-side CSR of one stream (built on demand; deterministic)."""
        return _corr_csr(self, int(stream))

    def corr_csr_chgroup(self, chgroup: int) -> tuple[CorrCsr, np.ndarray]:
        """The n_sub streams of one chgroup concatenated (one corr kernel launch):
        ``(csr, value_offsets)`` with stream s's values at ``value_offsets[s]:[s+1]``."""
        parts = [self.corr_csr(chgroup * self.n_sub + s) for s in range(self.n_sub)]
        voff = np.zeros(self.n_sub + 1, dtype=np.int32)
        voff[1:] = np.cumsum([p.ptr.shape[0] - 1 for p in parts])
        ptrs, base = [np.zeros(1, dtype=np.int32)], 0
        for p in parts:
            ptrs.append(p.ptr[1:] + base)
            base += p.n_src
        return CorrCsr(stream=-1, ptr=np.concatenate(ptrs).astype(np.int32),
                       bl=np.concatenate([p.bl for p in parts]), ch=np.concatenate([p.ch for p in parts]),
                       conj=np.concatenate([p.conj for p in parts])), voff

    def summary(self) -> dict:
        nv = self.n_values
        nh = self.n_heavy
        nsum = sum(int(s.n_contrib.sum()) for s in self.streams)
        nheavy_w = sum(int(s.n_contrib[s.heavy].sum()) for s in self.streams)
        out = dict(kind=self.kind, n_sub=self.n_sub, csf=self.chan_sum_factor, n_streams=self.n_streams,
                   n_values=[int(nv.min()), float(nv.mean()), int(nv.max())], n_values_total=int(nv.sum()),
                   n_heavy=[int(nh.min()), float(nh.mean()), int(nh.max())],
                   heavy_value_frac=float(nh.sum() / nv.sum()), heavy_weight_frac=nheavy_w / nsum,
                   n_image=self.n_image, heavy_threshold=self.heavy_threshold,
                   owner_active_cells=self.owner.n_active, owner_entries=self.owner.n_entries,
                   pattern_id=f"0x{self.pattern_id:016x}", n_dropped_offgrid=self.n_dropped_offgrid)
        if self.classes is not None:
            c = self.classes
            out.update(n_kept=c.n_kept, n_class=c.n_class, max_multiplicity=int(c.multiplicity.max()),
                       class_max_diameter_m=c.max_diameter_m, class_min_gap_m=c.min_gap_m)
        return out


# ---------------------------------------------------------------------------------------
# pattern_id
# ---------------------------------------------------------------------------------------
def _blake2b_u64(payload: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=_PERSON).digest(), "little")


def _id_payload(*, kind, antpos_e, antpos_n, antpos_u, is_core_baseline_mask, chan_sum_factor, n_sub,
                n_grid, cell_lambda, dec_deg, redundancy_tol_m, heavy_frac, normalise) -> bytes:
    if kind not in _KINDS:
        raise ValueError(f"kind={kind!r}, expected one of {_KINDS}")
    e = np.ascontiguousarray(antpos_e, dtype="<f8")
    n = np.ascontiguousarray(antpos_n, dtype="<f8")
    u = np.zeros_like(e) if antpos_u is None else np.ascontiguousarray(antpos_u, dtype="<f8")
    mask = (np.ones(NBASE, dtype=np.uint8) if is_core_baseline_mask is None
            else np.ascontiguousarray(is_core_baseline_mask, dtype=np.uint8))
    head = np.array([VALUE_MAP_SCHEMA_VERSION, _KINDS.index(kind), chan_sum_factor, n_sub, n_grid,
                     1 if normalise else 0, e.shape[0]], dtype="<i8").tobytes()
    flts = np.array([cell_lambda, quantise_dec_deg(dec_deg), redundancy_tol_m,
                     -1.0 if heavy_frac is None else heavy_frac], dtype="<f8").tobytes()
    return (b"x2vmap" + head + flts + np.uint64(compute_chgroup_table_hash()).tobytes()
            + e.tobytes() + n.tobytes() + u.tobytes() + mask.tobytes())


def value_map_pattern_id(**kw) -> int:
    """The map's ``pattern_id`` from its inputs alone (same keywords as :func:`build_value_map`
    minus the optional ones): lets corr and search agree before either builds."""
    return _blake2b_u64(_id_payload(**kw))


# ---------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------
def _chgroup_nu_GHz(g: int, csf: int) -> np.ndarray:
    """Summed-channel frequencies of chgroup g, exactly as build_pattern computes them."""
    full = np.asarray([freq_GHz(g, ch) for ch in range(NCHAN_PER_CHGROUP)], dtype=np.float64)
    if csf == 1:
        return full
    return full.reshape(NCHAN_PER_CHGROUP // csf, csf).mean(axis=1)


def _cells(du_m, dv_m, nu_GHz, cell_lambda, n_grid):
    """(row, col) int64 of each (baseline, frequency): from_pattern arithmetic, F20."""
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / (np.asarray(nu_GHz, dtype=np.float64) * 1e9)
    u_lam = du_m[:, None] / wavelength_m[None, :]
    v_lam = dv_m[:, None] / wavelength_m[None, :]
    u_lam = -u_lam
    v_lam = -v_lam
    half = n_grid // 2
    col = np.rint(u_lam / cell_lambda).astype(np.int64) + half
    row = np.rint(v_lam / cell_lambda).astype(np.int64) + half
    return row, col


def _keys(row, col):
    return (np.asarray(row).astype(np.uint32) << 16) | np.asarray(col).astype(np.uint32)


def build_owner_csr(streams, n_grid: int, normalise: bool) -> OwnerCsr:
    """Fold every (stream, value) into today's Hermitian half-spectrum layout."""
    n_half = n_grid // 2 + 1
    cells, sts, vals, conjs, wts = [], [], [], [], []
    for st in streams:
        r = st.cell_row.astype(np.int64)
        c = st.cell_col.astype(np.int64)
        v = np.arange(r.shape[0], dtype=np.int64)
        w = st.sqrt_n if normalise else np.ones(r.shape[0], dtype=np.float32)
        d = c <= n_grid // 2
        mr, mc = (n_grid - r) % n_grid, (n_grid - c) % n_grid
        m = mc <= n_grid // 2
        for sel, rr, cc, cj in ((d, r, c, 0), (m, mr, mc, 1)):
            cells.append(rr[sel] * n_half + cc[sel])
            sts.append(np.full(int(sel.sum()), st.stream, dtype=np.int64))
            vals.append(v[sel])
            conjs.append(np.full(int(sel.sum()), cj, dtype=np.int64))
            wts.append(w[sel])
    cell = np.concatenate(cells)
    st_a = np.concatenate(sts)
    val = np.concatenate(vals)
    cj = np.concatenate(conjs)
    wt = np.concatenate(wts).astype(np.float32)
    order = np.lexsort((val, cj, st_a, cell))
    cell, st_a, val, cj, wt = cell[order], st_a[order], val[order], cj[order], wt[order]
    uw, counts = np.unique(cell, return_counts=True)
    ptr = np.zeros(uw.shape[0] + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts)
    return OwnerCsr(n_grid=int(n_grid), cell_uw=uw.astype(np.int32), cell_ptr=ptr,
                    ent_stream=st_a.astype(np.int32), ent_value=val.astype(np.int32),
                    ent_conj=cj.astype(np.uint8), ent_weight=wt)


def heavy_threshold(n_image: int, heavy_frac: float) -> int:
    """Smallest integer n with n >= heavy_frac * n_image, in exact rational arithmetic
    (heavy_frac = 1e-4 is read as 1/10000, not its binary approximation)."""
    frac = Fraction(heavy_frac).limit_denominator(10 ** 12)
    return int(math.ceil(frac * int(n_image)))


def _order_values(keys, class_id, n, threshold):
    """Wire order: heavy first, then main; each in (key, class id) order."""
    base = np.lexsort((class_id, keys))
    if threshold is None:
        return base
    hv = n[base] >= threshold
    return np.concatenate([base[hv], base[~hv]])


def build_value_map(antpos_e, antpos_n, *, antpos_u=None, is_core_baseline_mask, kind: str, n_sub: int,
                    chan_sum_factor: int, dec_deg: float, cell_lambda: float, n_grid: int = 256,
                    heavy_frac: float | None = 1e-4, normalise: bool = True,
                    redundancy_tol_m: float = 0.05) -> ValueMap:
    """Build the value map of every (chgroup, sub-band) stream (SPEC §3).

    ``antpos_*``: ``(NANTS,)`` metres (E, N from the cal blob; U optional, zeros if
    None -- today's gridding ignores U, the redundancy test uses it).
    ``is_core_baseline_mask``: ``(NBASE,)`` bool (station <= 102).
    ``cell_lambda``: explicit, shared by every stream (tee45 in production).
    ``heavy_frac=None`` disables the heavy class (1x); ``normalise=False`` gives the
    1x owner weights (1).
    """
    if kind not in _KINDS:
        raise ValueError(f"kind={kind!r}, expected one of {_KINDS}")
    csf = int(chan_sum_factor)
    n_sub = int(n_sub)
    nchan_eff = NCHAN_PER_CHGROUP // csf
    if csf < 1 or nchan_eff * csf != NCHAN_PER_CHGROUP or n_sub < 1 or nchan_eff % n_sub:
        raise ValueError(f"csf={csf} / n_sub={n_sub} do not tile {NCHAN_PER_CHGROUP} channels")
    if not (np.isfinite(cell_lambda) and cell_lambda > 0):
        raise ValueError(f"cell_lambda={cell_lambda} must be > 0")
    e = np.asarray(antpos_e)
    nn = np.asarray(antpos_n)
    if e.shape != (NANTS,) or nn.shape != (NANTS,):
        raise ValueError(f"antpos shapes must be ({NANTS},)")
    if antpos_u is not None and np.asarray(antpos_u).shape != (NANTS,):
        raise ValueError(f"antpos_u shape must be ({NANTS},)")
    per = nchan_eff // n_sub
    idkw = dict(kind=kind, antpos_e=e, antpos_n=nn, antpos_u=antpos_u,
                is_core_baseline_mask=is_core_baseline_mask, chan_sum_factor=csf, n_sub=n_sub,
                n_grid=int(n_grid), cell_lambda=float(cell_lambda), dec_deg=float(dec_deg),
                redundancy_tol_m=float(redundancy_tol_m), heavy_frac=heavy_frac, normalise=bool(normalise))
    payload = _id_payload(**idkw)
    pattern_id = _blake2b_u64(payload)
    kept_bl, ka, kb = kept_core_baselines(is_core_baseline_mask)
    e64, n64 = e.astype(np.float64), nn.astype(np.float64)
    du_m, dv_m = e64[ka] - e64[kb], n64[ka] - n64[kb]

    # pass 1: per stream (keys in cell order, class ids, n) -- order fixed after N_image
    raw = []          # (g, s, row, col, n, class_id, legacy_id)
    classes = None
    n_drop = 0
    if kind == "gridded":
        antpos_hash = compute_antpos_hash(e, nn)
        table_hash = compute_chgroup_table_hash()
        for g in range(N_CHGROUP):
            row_all, col_all = _cells(du_m, dv_m, _chgroup_nu_GHz(g, csf), cell_lambda, n_grid)
            for s in range(n_sub):
                pat = build_pattern(e, nn, chgroup=g, dec_deg=dec_deg, n_grid=n_grid, kernel_support=1,
                                    chan_sum_factor=csf, cell_lambda=cell_lambda,
                                    is_core_baseline_mask=is_core_baseline_mask, antpos_hash=antpos_hash,
                                    chgroup_table_hash=table_hash,
                                    sub_band=None if n_sub == 1 else (s, n_sub))
                idx = _gridded_source_index(pat, row_all[:, s * per:(s + 1) * per],
                                            col_all[:, s * per:(s + 1) * per], n_grid)
                cnt = np.bincount(idx[idx >= 0], minlength=pat.n_filled).astype(np.int64)
                if np.any(cnt < 1):
                    raise RuntimeError(f"stream ({g},{s}): a pattern cell has no source")
                raw.append((g, s, pat.ix_row.astype(np.int64), pat.ix_col.astype(np.int64), cnt,
                            np.full(pat.n_filled, -1, dtype=np.int64), int(pat.pattern_id)))
    else:
        classes = redundancy_classes(e, nn, antpos_u, is_core_baseline_mask=is_core_baseline_mask,
                                     tol_m=redundancy_tol_m)
        cu, cv = classes.class_uvw_m[:, 0], classes.class_uvw_m[:, 1]
        for g in range(N_CHGROUP):
            nu = _chgroup_nu_GHz(g, csf)
            for s in range(n_sub):
                nu_c = np.array([nu[s * per:(s + 1) * per].mean()])
                row, col = _cells(cu, cv, nu_c, cell_lambda, n_grid)
                row, col = row[:, 0], col[:, 0]
                ok = (row >= 0) & (row < n_grid) & (col >= 0) & (col < n_grid)
                n_drop += int((~ok).sum())
                cid = np.flatnonzero(ok).astype(np.int64)
                raw.append((g, s, row[ok], col[ok], classes.multiplicity[ok].astype(np.int64) * per,
                            cid, None))
    n_image = int(sum(int(r[4].sum()) for r in raw))
    threshold = None if heavy_frac is None else heavy_threshold(n_image, heavy_frac)
    streams = []
    for g, s, row, col, cnt, cid, legacy in raw:
        order = _order_values(_keys(row, col).astype(np.int64), cid, cnt, threshold)
        heavy = (np.zeros(order.shape[0], dtype=bool) if threshold is None else cnt[order] >= threshold)
        streams.append(StreamValues(
            stream=g * n_sub + s, chgroup=g, sub_band=s, cell_row=row[order].astype(np.uint16),
            cell_col=col[order].astype(np.uint16), n_contrib=cnt[order].astype(np.int32), heavy=heavy,
            class_id=cid[order].astype(np.int32), legacy_pattern_id=legacy))
    owner = build_owner_csr(streams, int(n_grid), bool(normalise))
    vm = ValueMap(
        kind=kind, n_sub=n_sub, chan_sum_factor=csf, n_grid=int(n_grid), cell_lambda=float(cell_lambda),
        dec_deg_quant=quantise_dec_deg(dec_deg), normalise=bool(normalise), heavy_frac=heavy_frac,
        heavy_threshold=threshold, n_image=n_image, redundancy_tol_m=float(redundancy_tol_m),
        pattern_id=pattern_id,
        stream_pattern_ids=tuple(_blake2b_u64(payload + b"S" + np.uint16(i).tobytes())
                                 for i in range(len(streams))),
        streams=tuple(streams), owner=owner, classes=classes, kept_bl=kept_bl, n_dropped_offgrid=n_drop,
        _inputs=dict(idkw, du_m=du_m, dv_m=dv_m))
    _validate(vm)
    return vm


def _gridded_source_index(pat, row, col, n_grid) -> np.ndarray:
    """(n_kept, per) cell-order index of each source in ``pat`` (-1 off-grid);
    raises on any in-grid source missing from the pattern (geometry drift)."""
    in_grid = (row >= 0) & (row < n_grid) & (col >= 0) & (col < n_grid)
    pat_keys = _keys(pat.ix_row, pat.ix_col)
    k = _keys(row[in_grid], col[in_grid])
    pos = np.searchsorted(pat_keys, k)
    if pos.size and (int(pos.max()) >= pat_keys.size or not np.all(pat_keys[pos] == k)):
        raise RuntimeError("a source cell is missing from its sub-band pattern (geometry drift)")
    idx = np.full(row.shape, -1, dtype=np.int64)
    idx[in_grid] = pos
    return idx


def _corr_csr(vm: ValueMap, stream: int) -> CorrCsr:
    st = vm.streams[stream]
    g, s = st.chgroup, st.sub_band
    nchan_eff = vm.nchan_eff
    per = nchan_eff // vm.n_sub
    inp = vm._inputs
    kept_bl = vm.kept_bl
    nk = kept_bl.shape[0]
    chans = np.arange(s * per, (s + 1) * per, dtype=np.int64)
    if vm.kind == "gridded":
        row, col = _cells(inp["du_m"], inp["dv_m"], _chgroup_nu_GHz(g, vm.chan_sum_factor)[chans],
                          vm.cell_lambda, vm.n_grid)
        in_grid = (row >= 0) & (row < vm.n_grid) & (col >= 0) & (col < vm.n_grid)
        wire_keys = _keys(st.cell_row, st.cell_col).astype(np.int64)
        if wire_keys.size == 0:
            raise RuntimeError(f"stream {stream}: no values")
        srt = np.argsort(wire_keys, kind="stable")
        k = _keys(row, col).astype(np.int64)
        pos = np.searchsorted(wire_keys[srt], k)
        pos = np.clip(pos, 0, wire_keys.size - 1)
        hit = in_grid & (wire_keys[srt][pos] == k)
        if np.any(in_grid & ~hit):
            raise RuntimeError(f"stream {stream}: in-grid source without a value")
        val = np.where(hit, srt[pos], -1)                              # (nk, per)
        conj = np.zeros((nk, per), dtype=bool)
    else:
        c = vm.classes
        vcls = np.full(c.n_class, -1, dtype=np.int64)
        vcls[st.class_id] = np.arange(st.n_values)
        val = np.repeat(vcls[c.bl_class][:, None], per, axis=1)        # (nk, per)
        conj = np.repeat(c.bl_conj[:, None], per, axis=1)
    flat = kept_bl.astype(np.int64)[:, None] * nchan_eff + chans[None, :]
    keep = val >= 0
    v, f, cj = val[keep], flat[keep], conj[keep]
    order = np.lexsort((f, v))
    v, f, cj = v[order], f[order], cj[order]
    ptr = np.zeros(st.n_values + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(np.bincount(v, minlength=st.n_values))
    if np.any(np.diff(ptr) != st.n_contrib):
        raise RuntimeError(f"stream {stream}: source counts disagree with n_contrib")
    return CorrCsr(stream=stream, ptr=ptr, bl=(f // nchan_eff).astype(np.int32),
                   ch=(f % nchan_eff).astype(np.int32), conj=cj.astype(bool))


def _validate(vm: ValueMap) -> None:
    """Structural invariants of a freshly built map (raises)."""
    tot = 0
    for i, st in enumerate(vm.streams):
        if st.stream != i or st.chgroup != i // vm.n_sub or st.sub_band != i % vm.n_sub:
            raise RuntimeError(f"stream {i}: index fields inconsistent")
        nv = st.n_values
        for name in ("cell_col", "n_contrib", "heavy", "class_id"):
            if getattr(st, name).shape != (nv,):
                raise RuntimeError(f"stream {i}: {name} shape")
        if np.any(st.n_contrib < 1):
            raise RuntimeError(f"stream {i}: a value with no source")
        nh = st.n_heavy
        if np.any(st.heavy[:nh] != True) or np.any(st.heavy[nh:]):  # noqa: E712
            raise RuntimeError(f"stream {i}: heavy values are not first")
        if vm.heavy_threshold is not None and np.any(st.heavy != (st.n_contrib >= vm.heavy_threshold)):
            raise RuntimeError(f"stream {i}: heavy flag != (n >= threshold)")
        keys = _keys(st.cell_row, st.cell_col).astype(np.int64)
        for part in (slice(0, nh), slice(nh, nv)):
            kk, cc = keys[part], st.class_id[part].astype(np.int64)
            if kk.size > 1 and np.any((np.diff(kk) < 0) | ((np.diff(kk) == 0) & (np.diff(cc) <= 0))):
                raise RuntimeError(f"stream {i}: values not in (cell, class) order")
        if vm.kind == "gridded" and np.unique(keys).size != nv:
            raise RuntimeError(f"stream {i}: duplicate gridded cell")
        tot += int(st.n_contrib.sum())
    if tot != vm.n_image:
        raise RuntimeError("n_image != sum of n")
    o = vm.owner
    if o.cell_ptr[-1] != o.n_entries or np.any(np.diff(o.cell_uw) <= 0):
        raise RuntimeError("owner CSR malformed")


def build_value_map_for_opoint(op, antpos_e, antpos_n, *, antpos_u=None, is_core_baseline_mask, dec_deg: float,
                               cell_lambda: float | None = None) -> ValueMap:
    """:func:`build_value_map` with kind / n_sub / csf / heavy / normalisation / tolerance /
    n_grid from an :class:`dsart.x2.opoint.OpPoint` and cell_lambda from its
    ``cell_lambda_mode`` (``tee45``) unless given."""
    if cell_lambda is None:
        if op.cell_lambda_mode != "tee45":
            raise ValueError(f"cell_lambda_mode={op.cell_lambda_mode!r}: pass cell_lambda explicitly")
        from dsart.grid.sparsity_pattern import IMAGE_PIXEL_ARCSEC_TEE, cell_lambda_for_pixel_arcsec
        cell_lambda = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, op.n_grid)
    return build_value_map(antpos_e, antpos_n, antpos_u=antpos_u, is_core_baseline_mask=is_core_baseline_mask,
                           kind=op.value_kind, n_sub=op.n_sub, chan_sum_factor=op.chan_sum_factor,
                           dec_deg=dec_deg, cell_lambda=cell_lambda, n_grid=op.n_grid,
                           heavy_frac=op.heavy_frac, normalise=op.normalise_values,
                           redundancy_tol_m=op.redundancy_tol_m)
