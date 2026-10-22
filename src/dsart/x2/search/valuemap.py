"""Owner inputs from the frozen WS-P value map (SPEC §3) + frame-v2 steps (SPEC §4/§5).

:func:`dm_inputs_from_value_map` turns a ``grid.baseline_grid.ValueMap``
into the engine's :class:`~dsart.x2.search.engine.DmInputs`:

* the owner CSR re-ordered column-major for K2, each cell's entries kept in
  the frozen ``(stream, conj, value)`` order (``csr.csr_from_owner``);
* per-stream row formats (2x streams differ in value and heavy counts):
  heavy values first as int8 pairs, main values packed at ``bits_main / 2``
  bits per component; 1x = today's cint8 rows;
* the per-value decode step ``c_v = f32(s)`` with ``s = s_heavy`` for
  ``StreamValues.heavy`` values else ``s_main`` (the frame header's
  sigma_hat-unit steps) and the natural weight ``w_v = f32(sqrt(n_v))``,
  applied as two fp32 multiplies (SPEC §4.5) -- the coordinator's
  "per-entry weight table built at pattern load", stored per value because
  a value's direct and mirror entries share it (checked against
  ``ent_weight``); 1x: ``c_v = w_v = 1`` (codes summed raw, today);
* ``value_sigma = sqrt(n_v)``: decoded per-component noise in sigma_hat
  units, for the 2x image scale.

A value map built for a different pattern than the ring's frames is a
PLAN/PATTERN mismatch the engine must refuse (SPEC §7 "fail loudly"):
:func:`check_pattern` compares the frame ``pattern_id``.
"""
from __future__ import annotations

import numpy as np

from .codec import RowFormat, StreamFormats, decode_constants
from .csr import choose_fold, csr_from_owner, csr_from_value_cells

__all__ = ["dm_inputs_from_value_map", "check_pattern"]


def dm_inputs_from_value_map(vmap, *, s_main: float = 1.0, s_heavy: float = 1.0,
                             bits_main: int = 8, midrise: bool = False, fold: str = "auto"):
    """``ValueMap`` -> ``DmInputs`` (one per DM; the map is DM-independent).

    Args:
        s_main, s_heavy: decode steps in sigma_hat units (frame header
            ``s_main`` / ``s_heavy``); ignored for a 1x map.
        bits_main: frame ``bits_main`` (bits per COMPLEX main value: 8/10/12).
        fold: Hermitian half axis of the owner CSR: ``"col"`` = WS-P's
            ``OwnerCsr`` verbatim (today's layout), ``"row"`` = rebuilt from
            the same value cells folded along rows, ``"auto"`` = the axis
            with fewer K2 lines (rows for the DSA-110 core: 57 vs 112).
    """
    from .engine import DmInputs

    cells_all = [np.asarray(sv.cell_row, np.int64) * int(vmap.n_grid)
                 + np.asarray(sv.cell_col, np.int64) for sv in vmap.streams]
    if fold == "auto":
        fold = choose_fold(cells_all, int(vmap.n_grid))
    if fold == "col":
        csr, w_ent = csr_from_owner(vmap.owner)
    else:
        csr = csr_from_value_cells(cells_all, n_grid=int(vmap.n_grid), fold=fold,
                                   extra_id=f"{int(vmap.pattern_id):016x}")
        w_ent = None
    S = len(vmap.streams)
    if csr.n_stream > S:
        raise ValueError(f"owner CSR references {csr.n_stream} streams, map has {S}")
    is_1x = (not vmap.normalise) and vmap.heavy_threshold is None
    V = max(int(sv.n_values) for sv in vmap.streams)
    c_v = np.zeros((S, V), dtype=np.float32)
    w_v = np.zeros((S, V), dtype=np.float32)
    fmts = []
    vsig = []
    for s, sv in enumerate(vmap.streams):
        nv = int(sv.n_values)
        if is_1x:
            f = RowFormat("cint8", n_values=nv)
            c, w, off = decode_constants(f)
            vsig.append(np.ones(nv))
        else:
            nh = int(sv.n_heavy)
            if not np.array_equal(np.asarray(sv.heavy), np.arange(nv) < nh):
                raise ValueError(f"stream {s}: heavy values are not the leading {nh}")
            f = RowFormat("frame2", n_values=nv, n_heavy=nh, bits=int(bits_main) // 2,
                          midrise=bool(midrise))
            n = np.asarray(sv.n_contrib, dtype=np.int64)
            c, w, off = decode_constants(f, step_main=float(np.float32(s_main)),
                                      step_heavy=float(np.float32(s_heavy)),
                                      n_contrib=n if vmap.normalise else None)
            vsig.append(np.sqrt(n.astype(np.float64)) if vmap.normalise else np.ones(nv))
        c_v[s, :nv] = c
        w_v[s, :nv] = w
        fmts.append(f)
    # the per-entry CSR weight must be the per-value static weight
    if w_ent is not None:
        sv_w = np.array([np.sqrt(float(vmap.streams[s].n_contrib[v])) if vmap.normalise
                         else 1.0 for s, v in zip(csr.ent_stream, csr.ent_value)])
        if sv_w.size and np.max(np.abs(sv_w - w_ent) / np.maximum(sv_w, 1.0)) > 1e-6:
            raise ValueError("owner CSR ent_weight is not the per-value sqrt(n)")
    return DmInputs(fmt=StreamFormats(tuple(fmts)), csr=csr, c_v=c_v, offset=float(off),
                    value_sigma=vsig, pattern_id=int(vmap.pattern_id), w_v=w_v)


def check_pattern(vmap, frame_pattern_id: int) -> None:
    if int(frame_pattern_id) != int(vmap.pattern_id):
        raise ValueError(f"PATTERN_MISMATCH: frames {int(frame_pattern_id):016x} vs value map "
                         f"{int(vmap.pattern_id):016x}")
