"""Owner-side cell CSR for the K1 combine (SPEC §3 owner form).

The combine writes, for every ACTIVE half-plane cell ``(u, w)``
(``w in [0, N/2]``, today's irfft2 layout, ``lin = u * N + w``), the
Hermitian fold of the gridded sum

    G_h[u, w] = 0.5 * sum_e  (v_e  or  conj(v_e))       (fixed entry order)

times today's fftshift checkerboard ``(-1)^(u+w)``. An entry ``e`` is
``(stream, value, conj)``, ordered ``(stream, conj, value)`` (SPEC §3); a value sitting at full-plane cell ``(u, w)``
contributes directly to half-cell ``(u, w)`` when ``w <= N/2`` and as
``conj`` to the mirror half-cell ``((N-u) % N, (N-w) % N)`` when that
mirror's column is ``<= N/2`` -- exactly the ``k`` / ``km`` pair
``tiled_combine_cuda.build_sparse_cell_csr`` builds from the inverse LUT.
Static per-value scales (row-rule step, ``sqrt(n)``, ``sqrt(mult)``) are
NOT here: they live in the per-value decode constant (``codec``), so K1 is
pure fp32 adds in a fixed order and its numpy twin is bit-exact.

The Hermitian half axis is selectable (``fold``): ``"col"`` is today's
irfft2 layout (``w <= N/2``), ``"row"`` folds along rows (``u <= N/2``).
Active cells are ordered line-major (line = the half-axis coordinate), so
K2 reads a line's cells contiguously; :attr:`OwnerCsr.fft_lines` lists the
lines holding a cell off position 0 along the full axis (they need a K2
FFT) and :attr:`OwnerCsr.const_lines` the lines whose only cell is at
position 0 (its IFFT is that value everywhere -- an exact constant for
K3a). With today's DC-at-centre grid the DSA-110 core spans 112 half
columns but 57 half rows: 2x folds along rows (:func:`choose_fold`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "OwnerCsr",
    "csr_from_owner",
    "csr_from_value_cells",
    "choose_fold",
    "csr_from_lut_1x",
    "csr_matches_v4",
    "expected_image_sigma",
]


@dataclass(frozen=True)
class OwnerCsr:
    """Static owner CSR (build once per pattern / declination).

    Attributes:
        n_grid: image side N.
        n_stream: streams (chgroup x sub-band).
        n_values: ``[n_stream]`` values per stream.
        cell_u, cell_w: ``[n_act]`` int32 half-plane cell coordinates.
        cell_ptr: ``[n_act + 1]`` int32 into the entry arrays.
        ent_stream, ent_value: ``[n_ent]`` int32.
        ent_conj: ``[n_ent]`` int8 (1 = conjugate the value).
        pattern_id: 16-hex id over every input.
    """

    n_grid: int
    n_stream: int
    n_values: np.ndarray
    cell_u: np.ndarray
    cell_w: np.ndarray
    cell_ptr: np.ndarray
    ent_stream: np.ndarray
    ent_value: np.ndarray
    ent_conj: np.ndarray
    pattern_id: str
    fold: str = "col"

    @property
    def n_half(self) -> int:
        return self.n_grid // 2 + 1

    @property
    def n_active(self) -> int:
        return int(self.cell_u.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.ent_stream.shape[0])

    @property
    def max_entries(self) -> int:
        return int(np.diff(self.cell_ptr).max(initial=0))

    @property
    def sign(self) -> np.ndarray:
        """``[n_act]`` float32 fftshift checkerboard ``(-1)^(u+w)``."""
        return np.where(((self.cell_u + self.cell_w) & 1) == 1, -1.0, 1.0).astype(np.float32)

    @property
    def cell_line(self) -> np.ndarray:
        """Half-axis coordinate of each cell (col fold: w; row fold: u)."""
        return self.cell_w if self.fold == "col" else self.cell_u

    @property
    def cell_pos(self) -> np.ndarray:
        """Full-axis coordinate (col fold: u; row fold: w) -- the K2 FFT axis."""
        return self.cell_u if self.fold == "col" else self.cell_w

    @property
    def fft_lines(self) -> np.ndarray:
        """Half-axis lines holding at least one cell off position 0 (K2)."""
        return np.unique(self.cell_line[self.cell_pos != 0]).astype(np.int32)

    @property
    def const_lines(self) -> np.ndarray:
        """Lines whose ONLY cell is at position 0 (exact constants in K3a)."""
        lines = np.unique(self.cell_line[self.cell_pos == 0])
        return np.setdiff1d(lines, self.fft_lines).astype(np.int32)

    def line_ptr(self, lines: np.ndarray) -> np.ndarray:
        """``[len(lines) + 1]`` offsets of each line's cells in the active
        order (cells are sorted line-major)."""
        cl = self.cell_line
        if np.any(np.diff(cl) < 0):
            raise AssertionError("cells not line-sorted")
        lo = np.searchsorted(cl, lines, side="left")
        hi = np.searchsorted(cl, lines, side="right")
        return np.concatenate([lo, hi[-1:]]).astype(np.int32) if len(lines) else np.zeros(1, np.int32)

    def summary(self) -> dict:
        ne = np.diff(self.cell_ptr)
        return dict(n_grid=self.n_grid, n_stream=self.n_stream, n_active=self.n_active,
                    n_entries=self.n_entries, max_entries=int(ne.max(initial=0)),
                    mean_entries=float(ne.mean()) if ne.size else 0.0,
                    fold=self.fold, n_fft_lines=int(self.fft_lines.size),
                    n_const_lines=int(self.const_lines.size),
                    fft_line_range=(int(self.fft_lines.min(initial=-1)),
                                    int(self.fft_lines.max(initial=-1))),
                    pattern_id=self.pattern_id)


def _pid(*parts) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()


def _half_lines(value_cells, N: int, fold: str) -> int:
    u_all, w_all = [], []
    for cells in value_cells:
        c = np.asarray(cells, dtype=np.int64)
        c = c[c >= 0]
        u_all.append(c // N)
        w_all.append(c % N)
    u = np.concatenate(u_all)
    w = np.concatenate(w_all)
    if fold == "row":
        a, b = u, w
    else:
        a, b = w, u
    a_half = np.where(a <= N // 2, a, (N - a) % N)
    b_half = np.where(a <= N // 2, b, (N - b) % N)
    return int(np.unique(a_half[b_half != 0]).size)


def choose_fold(value_cells: Sequence[np.ndarray], n_grid: int) -> str:
    """The Hermitian half axis that minimises the K2 line count. With
    today's DC-at-centre grid the DSA-110 core occupies 112 half-plane
    columns but only 57 half-plane rows, so 2x folds along rows."""
    return "row" if _half_lines(value_cells, n_grid, "row") < \
        _half_lines(value_cells, n_grid, "col") else "col"


def csr_from_value_cells(value_cells: Sequence[np.ndarray], *, n_grid: int,
                         extra_id: Optional[str] = None, fold: str = "col") -> OwnerCsr:
    """Build the owner CSR from per-stream full-plane cell maps.

    Args:
        value_cells: per stream ``s``, an int array ``[n_values_s]`` of the
            full-plane linear cell ``u * N + w`` each value is gridded at
            (``-1`` = value unused). Several values may share a cell
            (per-baseline formats, many-to-one).
        n_grid: N (even).

    Entry order inside a cell: ``(stream, conj, value)`` ascending -- the
    frozen SPEC §3 owner order (for 1x: v4's stream-ascending order).

    ``fold`` picks the Hermitian half axis: ``"col"`` (today's irfft2
    layout, ``w <= N/2``; required for the 1x bit-exactness with v4),
    ``"row"`` (``u <= N/2``) or ``"auto"`` (:func:`choose_fold`). Both give
    the same image ``Re(ifft2(G_full))``.
    """
    N = int(n_grid)
    if N % 2:
        raise ValueError("n_grid must be even")
    if fold == "auto":
        fold = choose_fold(value_cells, N)
    if fold not in ("col", "row"):
        raise ValueError(f"fold={fold!r}")
    us, ws, ss, vs, cs = [], [], [], [], []
    n_values = np.array([len(v) for v in value_cells], dtype=np.int32)
    for s, cells in enumerate(value_cells):
        cells = np.asarray(cells, dtype=np.int64)
        vidx = np.nonzero(cells >= 0)[0]
        if vidx.size == 0:
            continue
        lin = cells[vidx]
        if lin.max() >= N * N:
            raise ValueError(f"stream {s}: cell index {int(lin.max())} >= N^2")
        u = lin // N
        w = lin % N
        mu = (N - u) % N
        mw = (N - w) % N
        if fold == "col":
            d = w <= N // 2                             # direct into (u, w)
            m = mw <= N // 2                            # conj into the mirror
        else:
            d = u <= N // 2
            m = mu <= N // 2
        us.append(u[d]); ws.append(w[d]); ss.append(np.full(int(d.sum()), s)); vs.append(vidx[d])
        cs.append(np.zeros(int(d.sum()), dtype=np.int8))
        us.append(mu[m]); ws.append(mw[m]); ss.append(np.full(int(m.sum()), s)); vs.append(vidx[m])
        cs.append(np.ones(int(m.sum()), dtype=np.int8))
    if not us:
        raise ValueError("no active values")
    u = np.concatenate(us).astype(np.int64)
    w = np.concatenate(ws).astype(np.int64)
    s = np.concatenate(ss).astype(np.int64)
    v = np.concatenate(vs).astype(np.int64)
    c = np.concatenate(cs).astype(np.int64)
    line, pos = (w, u) if fold == "col" else (u, w)
    order = np.lexsort((v, c, s, pos, line))           # line-major cells; entries (stream, conj, value)
    u, w, s, v, c = u[order], w[order], s[order], v[order], c[order]
    line, pos = (w, u) if fold == "col" else (u, w)
    key = line * N + pos                                # unique per cell
    starts = np.concatenate([[0], np.nonzero(np.diff(key))[0] + 1])
    cell_ptr = np.concatenate([starts, [key.size]]).astype(np.int32)
    return OwnerCsr(
        n_grid=N, n_stream=len(value_cells), n_values=n_values,
        cell_u=u[starts].astype(np.int32), cell_w=w[starts].astype(np.int32),
        cell_ptr=cell_ptr, ent_stream=s.astype(np.int32), ent_value=v.astype(np.int32),
        ent_conj=c.astype(np.int8),
        pattern_id=_pid(N, fold, *[np.asarray(x, dtype=np.int64) for x in value_cells],
                        extra_id or ""),
        fold=fold,
    )


def csr_from_owner(owner) -> tuple:
    """WS-P ``baseline_grid.OwnerCsr`` (row-major ``cell_uw``, entries
    ``(stream, conj, value)``, ``ent_weight``) -> ``(OwnerCsr, weight)``:
    the same cells re-ordered column-major for K2 (each cell's entry block
    kept verbatim, so the fp32 accumulation order is the frozen one) and the
    per-entry weights in the new entry order."""
    N = int(owner.n_grid)
    nh = N // 2 + 1
    uw = np.asarray(owner.cell_uw, dtype=np.int64)
    ptr = np.asarray(owner.cell_ptr, dtype=np.int64)
    u, w = uw // nh, uw % nh
    order = np.lexsort((u, w))
    ne = np.diff(ptr)[order]
    new_ptr = np.concatenate([[0], np.cumsum(ne)]).astype(np.int32)
    take = np.concatenate([np.arange(ptr[a], ptr[a + 1]) for a in order]) if order.size \
        else np.zeros(0, np.int64)
    S = int(np.asarray(owner.ent_stream).max(initial=-1)) + 1
    n_values = np.zeros(S, dtype=np.int32)
    es = np.asarray(owner.ent_stream)
    ev = np.asarray(owner.ent_value)
    for s in range(S):
        sel = ev[es == s]
        n_values[s] = int(sel.max(initial=-1)) + 1
    csr = OwnerCsr(
        n_grid=N, n_stream=S, n_values=n_values,
        cell_u=u[order].astype(np.int32), cell_w=w[order].astype(np.int32), cell_ptr=new_ptr,
        ent_stream=es[take].astype(np.int32), ent_value=ev[take].astype(np.int32),
        ent_conj=np.asarray(owner.ent_conj)[take].astype(np.int8),
        pattern_id=_pid(N, uw, ptr, es, ev, np.asarray(owner.ent_conj)),
    )
    return csr, np.asarray(owner.ent_weight, dtype=np.float32)[take]


def csr_from_lut_1x(lut: np.ndarray, n_filled: np.ndarray, *, n_grid: int) -> OwnerCsr:
    """Today's compact LUT (``lut[g, k]`` = full-plane cell of value ``k``
    of stream ``g``; ``n_filled[g]`` valid) -> owner CSR. Same cell set and
    per-cell stream lists as ``build_sparse_cell_csr(build_inverse_lut(..))``
    (checked by :func:`csr_matches_v4`)."""
    lut = np.asarray(lut)
    n_filled = np.asarray(n_filled)
    cells = []
    for g in range(lut.shape[0]):
        row = np.full(lut.shape[1], -1, dtype=np.int64)
        nf = int(n_filled[g])
        row[:nf] = lut[g, :nf]
        bad = (row < 0) | (row >= n_grid * n_grid)
        row[bad] = -1
        cells.append(row)
    return csr_from_value_cells(cells, n_grid=n_grid, extra_id="lut1x")


def csr_matches_v4(csr: OwnerCsr, v4) -> tuple:
    """Compare with a production ``SparseCellCsr`` (torch, any device).

    Returns ``(ok, message)``: same active cells, and per cell the same
    ``(stream, k, km)`` triples as v4's ``ent_g / ent_k / ent_km``.
    """
    N = csr.n_grid
    nh = N // 2 + 1
    uw = v4.cell_uw.detach().cpu().numpy().astype(np.int64)
    ptr = v4.cell_ptr.detach().cpu().numpy().astype(np.int64)
    eg = v4.ent_g.detach().cpu().numpy()
    ek = v4.ent_k.detach().cpu().numpy()
    ekm = v4.ent_km.detach().cpu().numpy()
    want = {}
    for a in range(uw.size):
        u, w = divmod(int(uw[a]), nh)
        trip = []
        for e in range(ptr[a], ptr[a + 1]):
            trip.append((int(eg[e]), int(ek[e]), int(ekm[e])))
        want[(u, w)] = sorted(trip)
    got = {}
    for a in range(csr.n_active):
        u, w = int(csr.cell_u[a]), int(csr.cell_w[a])
        per = {}
        for e in range(csr.cell_ptr[a], csr.cell_ptr[a + 1]):
            g = int(csr.ent_stream[e])
            k, km = per.get(g, (-1, -1))
            if csr.ent_conj[e]:
                km = int(csr.ent_value[e])
            else:
                k = int(csr.ent_value[e])
            per[g] = (k, km)
        got[(u, w)] = sorted((g, k, km) for g, (k, km) in per.items())
    if set(got) != set(want):
        return False, f"active cells differ: {len(got)} vs v4 {len(want)}"
    for key, trip in want.items():
        if got[key] != trip:
            return False, f"cell {key}: {got[key]} vs v4 {trip}"
    return True, f"{len(got)} cells match"


def expected_image_sigma(csr: OwnerCsr, value_sigma: Sequence[np.ndarray]) -> float:
    """Per-pixel image noise std for independent value noise.

    ``value_sigma[s][v]`` = per-component std of decoded value ``v`` of
    stream ``s`` (including its decode constant). Exact for independent,
    equal-re/im-variance values: the irfft2 output is
    ``(1/N^2) [sum over half cells of 2 Re(G_h e^{i theta})]`` with the
    self-mirror columns counted once per conjugate pair (see module doc);
    tested against the float64 reference's exact linear map.
    """
    N = csr.n_grid
    var = np.zeros(csr.n_active)
    for a in range(csr.n_active):
        acc = 0.0
        for e in range(csr.cell_ptr[a], csr.cell_ptr[a + 1]):
            sv = float(value_sigma[csr.ent_stream[e]][csr.ent_value[e]])
            acc += 0.25 * sv * sv
        var[a] = acc
    selfcol = (csr.cell_line == 0) | (csr.cell_line == N // 2)
    factor = np.where(selfcol, 2.0, 4.0)
    return float(np.sqrt(np.sum(factor * var)) / (N * N))
