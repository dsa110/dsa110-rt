"""Interim value maps (SPEC §3) for the corr side, until WS-P's
``dsart.grid.baseline_grid`` lands.

A **value** is what one stream ships per (coarse DM, sample): a gridded
uv cell (``kind="cells"``) or one exact-redundancy class of core
cross-baselines summed over the sub-band (``kind="baselines"``). The corr
side needs, per chgroup:

* ``rows`` -- the GEMM output rows: the core cross-baselines (station
  <= 102) in today's xGPU order (``bls = a(a+1)/2 + b``, ``b < a``), with
  their antenna pair; :func:`row_csr_for_values` maps values back onto
  them;
* per stream ``s`` (sub-band ``s`` of ``n_sub``): the ordered values, each
  with its source list ``(row, summed channel k, conj)`` in a FIXED order
  (ascending ``row * nsum + k``, the ``cell_index_map`` flat order), its
  static contribution count ``n`` = number of sources, and its owner
  cell ``(row, col)`` in today's image layout;
* heavy values (``n >= heavy_threshold``) first, then main values in cell
  order (``(row << 16) | col``; ties by class index for baselines).

Geometry is today's, verbatim: ``u = -(E_a - E_b)/lambda``,
``v = -(N_a - N_b)/lambda`` (F20), cell ``rint(u / cell_lambda) +
n_grid//2`` (``build_pattern``); summed-channel frequencies are the
band-centre means. With ``kind="cells"``, ``csf=8``, ``n_sub=4`` and no
heavy split this reproduces ``build_subband_patterns`` cell sets and order
exactly (tested), which is the SPEC §3 1x-compat requirement.

Redundancy classes (interim rule): baseline vectors ``pos_a - pos_b`` in
3-D are binned at ``tol`` (5 cm) after sign canonicalisation; members whose
vector is the negated class representative carry ``conj=True`` (their
visibility is conjugated before summing). WS-P replaces this with the
SPEC's tolerance clustering; the interface stays.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import NCHAN_PER_CHGROUP, SPEED_OF_LIGHT_M_S

__all__ = [
    "CoreRows",
    "StreamValues",
    "ValueMap",
    "build_value_map",
    "core_rows",
    "redundancy_classes",
    "row_csr_for_values",
]


@dataclass(frozen=True)
class CoreRows:
    """Core cross-baselines in xGPU order."""

    bls: np.ndarray        # (R,) legacy baseline index
    a: np.ndarray          # (R,) higher antenna index
    b: np.ndarray          # (R,) lower antenna index
    core_ants: np.ndarray  # (n_core,) sorted antenna indices

    @property
    def n(self) -> int:
        return int(self.bls.shape[0])


def core_rows(is_core_ant: np.ndarray) -> CoreRows:
    is_core_ant = np.asarray(is_core_ant, dtype=bool)
    nants = is_core_ant.shape[0]
    a_l, b_l, k_l = [], [], []
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            if a != b and is_core_ant[a] and is_core_ant[b]:
                a_l.append(a)
                b_l.append(b)
                k_l.append(k)
            k += 1
    return CoreRows(
        bls=np.asarray(k_l, np.int64), a=np.asarray(a_l, np.int64),
        b=np.asarray(b_l, np.int64), core_ants=np.flatnonzero(is_core_ant),
    )


def redundancy_classes(
    rows: CoreRows,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    antpos_u: np.ndarray | None = None,
    *,
    tol_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(class_of_row, conj_of_row, n_class)``. Classes are numbered in
    order of first appearance along the row order; the representative of
    a class is its first row (``conj=False``)."""
    u = np.zeros_like(np.asarray(antpos_e, np.float64)) if antpos_u is None else antpos_u
    pos = np.stack([np.asarray(antpos_e, np.float64), np.asarray(antpos_n, np.float64),
                    np.asarray(u, np.float64)], axis=1)
    d = pos[rows.a] - pos[rows.b]                                   # (R, 3)
    q = np.rint(d / float(tol_m)).astype(np.int64)
    # canonical sign: first nonzero component positive
    first_nz = np.argmax(q != 0, axis=1)
    sgn = np.sign(q[np.arange(q.shape[0]), first_nz])
    sgn[sgn == 0] = 1
    qc = q * sgn[:, None]
    class_of_row = np.empty(rows.n, np.int64)
    rep_sign: dict[tuple, int] = {}
    ids: dict[tuple, int] = {}
    conj = np.zeros(rows.n, bool)
    for r in range(rows.n):
        key = tuple(qc[r])
        if key not in ids:
            ids[key] = len(ids)
            rep_sign[key] = int(sgn[r])
        class_of_row[r] = ids[key]
        conj[r] = int(sgn[r]) != rep_sign[key]
    return class_of_row, conj, len(ids)


@dataclass(frozen=True)
class StreamValues:
    """One stream's ordered values (heavy first)."""

    stream: int
    cell_row: np.ndarray     # (V,) owner cell row (v axis)
    cell_col: np.ndarray     # (V,) owner cell col (u axis)
    n: np.ndarray            # (V,) static contribution count
    src_ptr: np.ndarray      # (V+1,) CSR offsets into src_*
    src_row: np.ndarray      # (S,) GEMM row (core baseline index)
    src_k: np.ndarray        # (S,) summed channel (chgroup-local)
    src_conj: np.ndarray     # (S,) bool
    n_heavy: int
    klass: np.ndarray | None = None   # (V,) class id (baselines kind)

    @property
    def n_values(self) -> int:
        return int(self.n.shape[0])


@dataclass(frozen=True)
class ValueMap:
    kind: str
    chgroup: int
    csf: int
    n_sub: int
    n_grid: int
    cell_lambda: float
    rows: CoreRows
    streams: tuple[StreamValues, ...]
    heavy_threshold: float
    class_of_row: np.ndarray | None = None
    conj_of_row: np.ndarray | None = None
    n_class: int = 0
    pattern_id: int = field(default=0)

    @property
    def nsum(self) -> int:
        return NCHAN_PER_CHGROUP // self.csf

    @property
    def per(self) -> int:
        return self.nsum // self.n_sub

    @property
    def n_values_total(self) -> int:
        return sum(s.n_values for s in self.streams)

    def offsets(self) -> np.ndarray:
        """``(n_sub+1,)`` value offsets of each stream on the concatenated axis."""
        o = np.zeros(self.n_sub + 1, np.int64)
        for s, sv in enumerate(self.streams):
            o[s + 1] = o[s] + sv.n_values
        return o


def _uv_lambda(rows: CoreRows, antpos_e, antpos_n, nu_ghz: np.ndarray):
    lam = SPEED_OF_LIGHT_M_S / (np.asarray(nu_ghz, np.float64) * 1e9)
    du = antpos_e[rows.a].astype(np.float64) - antpos_e[rows.b].astype(np.float64)
    dv = antpos_n[rows.a].astype(np.float64) - antpos_n[rows.b].astype(np.float64)
    return -(du[:, None] / lam[None, :]), -(dv[:, None] / lam[None, :])


def _cells(u, v, cell_lambda, n_grid):
    half = n_grid // 2
    col = np.rint(u / cell_lambda).astype(np.int64) + half
    row = np.rint(v / cell_lambda).astype(np.int64) + half
    ok = (row >= 0) & (row < n_grid) & (col >= 0) & (col < n_grid)
    return row, col, ok


def _pid(*parts) -> int:
    h = hashlib.blake2b(digest_size=8, person=b"dsart-x2-vmap")
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return int.from_bytes(h.digest(), "little")


def build_value_map(
    kind: str,
    *,
    chgroup: int,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    is_core_ant: np.ndarray,
    csf: int,
    n_sub: int,
    n_grid: int,
    cell_lambda: float,
    antpos_u: np.ndarray | None = None,
    heavy_threshold: float = np.inf,
    tol_m: float = 0.05,
) -> ValueMap:
    if kind not in ("cells", "baselines"):
        raise ValueError(f"kind={kind!r}")
    nsum = NCHAN_PER_CHGROUP // csf
    if nsum % n_sub:
        raise ValueError(f"n_sub={n_sub} must divide nsum={nsum}")
    per = nsum // n_sub
    rows = core_rows(is_core_ant)
    nus = build_summed_chgroup_freq_table_GHz(csf)[chgroup]          # (nsum,)
    streams = []
    cls = conj_r = None
    n_class = 0
    if kind == "baselines":
        cls, conj_r, n_class = redundancy_classes(rows, antpos_e, antpos_n, antpos_u, tol_m=tol_m)
    for s in range(n_sub):
        ks = np.arange(s * per, (s + 1) * per)
        if kind == "cells":
            u, v, = _uv_lambda(rows, antpos_e, antpos_n, nus[ks])
            crow, ccol, ok = _cells(u, v, cell_lambda, n_grid)       # (R, per)
            r_idx, k_loc = np.nonzero(ok)
            key = (crow[ok].astype(np.uint64) << np.uint64(16)) | ccol[ok].astype(np.uint64)
            flat = r_idx * nsum + ks[k_loc]
            order = np.lexsort((flat, key))                          # by cell, then flat src
            key_s, flat_s = key[order], flat[order]
            uniq, first, counts = np.unique(key_s, return_index=True, return_counts=True)
            vals_row = (uniq >> np.uint64(16)).astype(np.int64)
            vals_col = (uniq & np.uint64(0xFFFF)).astype(np.int64)
            n = counts.astype(np.int64)
            ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            src_row = flat_s // nsum
            src_k = flat_s % nsum
            src_conj = np.zeros(src_row.shape, bool)
            klass = None
        else:
            # one value per class: sources = members x sub-band channels
            nu_c = float(np.mean(nus[ks]))
            # owner cell of the class representative at the sub-band centre
            rep_rows = np.array([np.flatnonzero(cls == c)[0] for c in range(n_class)])
            sub = CoreRows(rows.bls[rep_rows], rows.a[rep_rows], rows.b[rep_rows], rows.core_ants)
            u, v = _uv_lambda(sub, antpos_e, antpos_n, np.array([nu_c]))
            vals_row, vals_col, ok = _cells(u[:, 0], v[:, 0], cell_lambda, n_grid)
            if not np.all(ok):
                raise ValueError("a redundancy class falls outside the uv grid")
            ptr_l, sr, sk, sc = [0], [], [], []
            for c in range(n_class):
                mem = np.flatnonzero(cls == c)
                flat = (mem[:, None] * nsum + ks[None, :]).reshape(-1)
                order = np.argsort(flat, kind="stable")
                flat = flat[order]
                sr.append(flat // nsum)
                sk.append(flat % nsum)
                sc.append(conj_r[flat // nsum])
                ptr_l.append(ptr_l[-1] + flat.size)
            src_row, src_k, src_conj = np.concatenate(sr), np.concatenate(sk), np.concatenate(sc)
            ptr = np.asarray(ptr_l, np.int64)
            n = np.diff(ptr)
            klass = np.arange(n_class)
        # heavy first, then cell order (class order breaks ties)
        cellkey = (vals_row.astype(np.int64) << 16) | vals_col.astype(np.int64)
        heavy = n >= heavy_threshold
        tie = klass if klass is not None else np.zeros_like(n)
        order = np.lexsort((tie, cellkey, ~heavy))
        new_ptr = [0]
        nr, nk, nc = [], [], []
        for v_ in order:
            a, b = ptr[v_], ptr[v_ + 1]
            nr.append(src_row[a:b])
            nk.append(src_k[a:b])
            nc.append(src_conj[a:b])
            new_ptr.append(new_ptr[-1] + (b - a))
        streams.append(StreamValues(
            stream=chgroup * n_sub + s,
            cell_row=vals_row[order], cell_col=vals_col[order], n=n[order],
            src_ptr=np.asarray(new_ptr, np.int64),
            src_row=np.concatenate(nr) if nr else np.zeros(0, np.int64),
            src_k=np.concatenate(nk) if nk else np.zeros(0, np.int64),
            src_conj=np.concatenate(nc) if nc else np.zeros(0, bool),
            n_heavy=int(heavy.sum()),
            klass=None if klass is None else klass[order],
        ))
    pid = _pid(kind, chgroup, csf, n_sub, n_grid, float(cell_lambda),
               np.asarray(antpos_e, np.float64), np.asarray(antpos_n, np.float64),
               None if antpos_u is None else np.asarray(antpos_u, np.float64),
               np.asarray(is_core_ant, bool), float(heavy_threshold), float(tol_m))
    return ValueMap(kind=kind, chgroup=chgroup, csf=csf, n_sub=n_sub, n_grid=n_grid,
                    cell_lambda=float(cell_lambda), rows=rows, streams=tuple(streams),
                    heavy_threshold=float(heavy_threshold), class_of_row=cls,
                    conj_of_row=conj_r, n_class=n_class, pattern_id=pid)


def row_csr_for_values(vm: ValueMap, *, rows: str = "baselines") -> dict[str, np.ndarray]:
    """Corr-side stage-1 CSR over the concatenated value axis.

    ``rows="baselines"``: sources index the GEMM's core-baseline rows (with
    conj flags). ``rows="classes"``: for ``kind="baselines"`` after the
    GEMM's redundancy-reduce epilogue -- sources index class rows, never
    conjugated, one per (class, channel). Returns int32 arrays ``ptr``
    (V+1), ``row``, ``k``, ``conj`` (uint8) and ``n`` (V,).
    """
    ptr, rr, kk, cc, nn = [0], [], [], [], []
    for sv in vm.streams:
        base = ptr[-1]
        if rows == "classes":
            if vm.kind != "baselines":
                raise ValueError("rows='classes' needs kind='baselines'")
            per_val = []
            for v in range(sv.n_values):
                a, b = sv.src_ptr[v], sv.src_ptr[v + 1]
                ks = np.unique(sv.src_k[a:b])
                per_val.append(ks)
                rr.append(np.full(ks.size, int(sv.klass[v]), np.int64))
                kk.append(ks)
                cc.append(np.zeros(ks.size, np.uint8))
            ptr.extend((base + np.cumsum([p.size for p in per_val])).tolist())
        else:
            rr.append(sv.src_row)
            kk.append(sv.src_k)
            cc.append(sv.src_conj.astype(np.uint8))
            ptr.extend((base + sv.src_ptr[1:]).tolist())
        nn.append(sv.n)
    return {
        "ptr": np.asarray(ptr, np.int64).astype(np.int32),
        "row": np.concatenate(rr).astype(np.int32),
        "k": np.concatenate(kk).astype(np.int32),
        "conj": np.concatenate(cc).astype(np.uint8),
        "n": np.concatenate(nn).astype(np.int64),
    }
