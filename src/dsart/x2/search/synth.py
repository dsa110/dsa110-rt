"""Synthetic owner inputs (tests, CPU end-to-end runs, the b8 bench).

* :func:`random_value_cells` -- per-stream value -> full-plane cell maps
  clustered around the uv origin (array index N/2, today's convention),
  optionally with shared cells across streams (gridded) or repeated cells
  within a stream (many-to-one, per-baseline formats).
* :func:`make_dm_inputs` -- a :class:`~dsart.x2.search.engine.DmInputs` for
  either row format.
* :class:`SyntheticSky` -- fills a :class:`~dsart.x2.search.ring.MemoryRing`
  with per-value Gaussian noise (unit per-component sigma in decoded units)
  plus dispersed point-source bursts, quantised with the reference row rule
  (2x) or as today's cint8 (1x); keeps the exact decoded values so the
  float64 reference sees the same numbers the engine decodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .codec import RowFormat, decode_constants, pack_frame2_rows
from .csr import csr_from_value_cells
from .geometry import DmSpec, OwnerPlan

__all__ = ["random_value_cells", "make_dm_inputs", "Burst", "SyntheticSky"]


def random_value_cells(n_stream: int, n_values: int, n_grid: int, *, radius: int,
                       seed: int = 0, n_unused: int = 0, repeat_frac: float = 0.0) -> list:
    """Per stream ``[n_values]`` full-plane linear cells within ``radius`` of
    (N/2, N/2); the last ``n_unused`` values are ``-1``."""
    rng = np.random.default_rng(seed)
    h = n_grid // 2
    out = []
    for _ in range(n_stream):
        n_use = n_values - n_unused
        uu = rng.integers(-radius, radius + 1, size=4 * n_use) + h
        ww = rng.integers(-radius, radius + 1, size=4 * n_use) + h
        lin = np.unique(uu * n_grid + ww)
        rng.shuffle(lin)
        lin = lin[:n_use]
        n_use = lin.size
        if repeat_frac > 0:
            n_rep = min(int(repeat_frac * n_use), n_use // 2)
            lin[:n_rep] = lin[n_rep:2 * n_rep]
        cells = np.full(n_values, -1, dtype=np.int64)
        cells[:n_use] = np.sort(lin)
        out.append(cells)
    return out


def make_dm_inputs(cells: Sequence[np.ndarray], *, n_grid: int, fmt: RowFormat,
                   step_main: float = 1.0, step_heavy: float = 1.0,
                   n_contrib: Optional[Sequence[np.ndarray]] = None):
    from .engine import DmInputs

    csr = csr_from_value_cells(cells, n_grid=n_grid)
    S = len(cells)
    V = fmt.n_values
    c_v = np.zeros((S, V), dtype=np.float32)
    off = 0.0
    for s in range(S):
        c, off = decode_constants(fmt, step_main=step_main, step_heavy=step_heavy,
                                  n_contrib=None if n_contrib is None else n_contrib[s])
        c_v[s] = c
    # decoded noise is ~unit per component in sigma_hat units (x sqrt(n))
    vs = [np.ones(V) if n_contrib is None else np.sqrt(np.asarray(n_contrib[s], float))
          for s in range(S)]
    if fmt.kind == "cint8":
        vs = None
    return DmInputs(fmt=fmt, csr=csr, c_v=c_v, offset=float(off), value_sigma=vs)


@dataclass
class Burst:
    d: int
    f: int                  # fine index within DM d
    t: int                  # dedispersed arrival row (tier samples)
    l: int
    m: int
    amp: float              # per-value amplitude in decoded units
    width: int = 1


@dataclass
class SyntheticSky:
    """Generates ring rows for every DM of an owner plan."""

    plan: OwnerPlan
    cells: Dict[int, list]
    formats: Dict[int, RowFormat]
    n_grid: int
    step_main: float = 0.336
    step_heavy: float = 0.1
    noise: float = 1.0
    seed: int = 1
    bursts: List[Burst] = field(default_factory=list)
    code_scale_1x: float = 20.0
    decoded: Dict[int, list] = field(default_factory=dict)      # d -> per stream [T, V] complex

    def rows(self, spec: DmSpec, row0: int, n: int) -> tuple:
        """``(raw uint8 [S, n, row_bytes], decoded per stream [n, V] complex)``."""
        from ..ref.search2x_ref import point_source_values

        fmt = self.formats[spec.d]
        S = spec.n_stream
        V = fmt.n_values
        rng = np.random.default_rng(self.seed * 1000003 + spec.d * 7919 + row0)
        y = (rng.standard_normal((S, n, V)) + 1j * rng.standard_normal((S, n, V))) * self.noise
        for b in self.bursts:
            if b.d != spec.d:
                continue
            for s in range(S):
                vis = point_source_values(self.cells[spec.d][s], self.n_grid, b.l, b.m, b.amp)
                for j in range(b.width):
                    r = b.t - int(spec.shifts[b.f, s]) + j - row0
                    if 0 <= r < n:
                        y[s, r] += vis
        cel = self.cells[spec.d]
        for s in range(S):
            y[s][:, np.asarray(cel[s]) < 0] = 0.0
        raw = np.zeros((S, n, fmt.row_bytes), dtype=np.uint8)
        dec = []
        if fmt.kind == "cint8":
            q = np.clip(np.rint(y * self.code_scale_1x / 3.0), -127, 127)
            for s in range(S):
                re = q[s].real.astype(np.int8)
                im = q[s].imag.astype(np.int8)
                pair = np.stack([re, im], axis=-1).reshape(n, 2 * V)
                raw[s] = pair.view(np.uint8)
                dec.append(re.astype(np.float64) + 1j * im.astype(np.float64))
            return raw, dec
        from ..ref.search2x_ref import quantize_rowrule
        nh = fmt.n_heavy
        for s in range(S):
            re = np.zeros((n, V), dtype=np.int64)
            im = np.zeros((n, V), dtype=np.int64)
            if nh:
                re[:, :nh], im[:, :nh] = quantize_rowrule(y[s][:, :nh], self.step_heavy, 8)
            re[:, nh:], im[:, nh:] = quantize_rowrule(y[s][:, nh:], self.step_main, fmt.bits)
            raw[s] = pack_frame2_rows(re, im, fmt)
            st = np.full(V, self.step_main)
            st[:nh] = self.step_heavy
            dec.append((re + 0.5) * st + 1j * (im + 0.5) * st)
        return raw, dec

    def fill(self, ring, spec: DmSpec, row0: int, n: int) -> None:
        raw, dec = self.rows(spec, row0, n)
        ring.push(spec.d, row0, raw)
        store = self.decoded.setdefault(spec.d, {})
        for i in range(n):
            store[row0 + i] = [dec[s][i] for s in range(spec.n_stream)]

    def decoded_block(self, d: int, row0: int, n: int, S: int, V: int) -> list:
        """Per stream ``[n, V]`` decoded values for rows ``row0 ..`` (zeros where absent)."""
        store = self.decoded.get(d, {})
        out = [np.zeros((n, V), dtype=np.complex128) for _ in range(S)]
        for i in range(n):
            row = store.get(row0 + i)
            if row is None:
                continue
            for s in range(S):
                out[s][i] = row[s]
        return out
