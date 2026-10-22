"""Replay driver: recorded RX rings through the owner engine and production.

Input: a WS-T recording of today's (v3) search RX ring
(``tools/x2/rx_recorder.py`` ``Recording``: ``t_range(dm)``,
``slots(dm, t0, t1)`` -> payload / vf / corr_flags, ``iter_cubes(dm, ...)``
-> what ``RxRing.assemble_compact_block`` would have produced). Duck-typed:
any object with those methods works (tests use a synthetic one).

* :func:`ring_from_recording` -- the recorded rows of one coarse DM as a
  :class:`~dsart.x2.search.ring.MemoryRing` (1x ``cint8`` rows); slots the
  recorder flagged unwritten / torn, or whose vf says overrun / pattern
  mismatch / no data, are invalid (zero-filled) -- the consumer's rules.
* :func:`replay_1x` -- per cycle: the owner engine at today's parameters
  (``EngineConfig.compat_1x``: lagged Layer-1, box filters, 1/256 scale)
  and today's production chain composed on CPU
  (``compat1x.production_cycle_cpu``, the same cube window, the engine's
  applied sigma / s_k so the two compare like for like) -> candidates
  matched within t +- 1, fine DM +- 1, l/m +- 1, dS/N <= 0.1 (plan
  "Correctness 1", 30 s recorded RX slots).

The 2x replay (frame-v2 recordings of ring v4) is the same loop with the
2x engine and a MemoryRing filled from ring-v4 rows (WS-T's v4 recorder).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .codec import RowFormat
from .ring import MemoryRing

__all__ = ["ring_from_recording", "replay_1x"]

F_WRITTEN, F_TORN = 1, 2
VF_DATA_PRESENT, VF_PATTERN_MISMATCH, VF_RX_OVERRUN = 1, 2, 4


def ring_from_recording(rec, dm: int, *, n_values: int, t_buf: Optional[int] = None,
                        d_as: Optional[int] = None) -> MemoryRing:
    a, b = rec.t_range(dm)
    f = rec.slots(dm, a, b)
    payload = np.asarray(f["payload"])                       # [T, n_corr, 2 * nfm] int8
    T, S, pb = payload.shape
    fmt = RowFormat("cint8", n_values=int(n_values))
    cf = np.asarray(f["corr_flags"])
    vf = np.asarray(f["vf"])
    bad = ((cf & F_WRITTEN) == 0) | ((cf & F_TORN) != 0) | ((vf & VF_RX_OVERRUN) != 0) \
        | ((vf & VF_PATTERN_MISMATCH) != 0) | ((vf & VF_DATA_PRESENT) == 0)
    d = int(dm if d_as is None else d_as)
    ring = MemoryRing({d: fmt}, S, t_buf=int(t_buf or T + 8))
    rows = np.zeros((S, T, fmt.row_bytes), np.uint8)
    rows[:, :, :min(pb, fmt.row_bytes)] = payload.transpose(1, 0, 2)[..., :fmt.row_bytes] \
        .view(np.uint8)
    ring.push(d, a, rows, valid=~bad.T)
    return ring


def replay_1x(rec, dm: int, *, plan, lut: np.ndarray, n_filled: np.ndarray, n_grid: int = 256,
              n_cycles: Optional[int] = None, fft_mode: str = "pruned",
              backend: str = "cpu", tol: Optional[dict] = None) -> dict:
    """Run both chains over a recording of coarse DM ``dm``; returns per-cycle
    match statistics and an overall ``ok``."""
    from .compat1x import compare_candidates, production_cycle_cpu
    from .csr import csr_from_lut_1x
    from .engine import DmInputs, EngineConfig, OwnerEngine

    tol = dict(dt=1, ddm=1, dlm=1, dsnr=0.1, **(tol or {}))
    spec = plan.dms[0]
    S, nfm = lut.shape
    ring = ring_from_recording(rec, dm, n_values=nfm, d_as=spec.d)
    csr = csr_from_lut_1x(lut, n_filled, n_grid=n_grid)
    inputs = {spec.d: DmInputs(fmt=RowFormat("cint8", n_values=nfm), csr=csr,
                               c_v=np.ones((S, nfm), np.float32), offset=0.0)}
    eng = OwnerEngine(plan, inputs, ring, EngineConfig.compat_1x(n_grid=n_grid, fft_mode=fft_mode,
                                                                 backend=backend))
    a, b = rec.t_range(dm)
    hop = spec.hop
    pad_l, pad_r = spec.max_pos, spec.max_neg
    hist = eng.states[spec.d].history
    k0 = -(-(a + max(hist, 64) + pad_l) // hop)                  # both windows inside
    k_hi = (b - pad_r) // hop - 1
    if n_cycles is not None:
        k_hi = min(k_hi, k0 + int(n_cycles) - 1)
    cycles, ok = [], True
    for k in range(k0, k_hi + 1):
        r = eng.process_cycle(k).per_dm[spec.d]
        S0 = r.t0 - 64                                           # today's cube start
        raw, _v = ring.wait_window(spec.d, S0 - pad_l, 256 + pad_l + pad_r)
        prod = production_cycle_cpu(cells=raw.view(np.int8), lut=lut, n_filled=n_filled,
                                    shifts=spec.shifts, pad_left=pad_l, t_det=256,
                                    n_grid=n_grid, sigma_inv=1.0 / r.sigma_applied,
                                    s_k=r.s_k_applied, specnum_start=S0, fine_dm=spec.fine_dm)
        cmp = compare_candidates(prod.raw_candidates, r.candidates, dt=tol["dt"],
                                 ddm=tol["ddm"], dlm=tol["dlm"], dsnr=tol["dsnr"],
                                 dm_key=lambda c: c.dm_idx if hasattr(c, "dm_idx")
                                 else c.fdm_local)
        good = not cmp["unmatched_a"]
        ok &= good
        cycles.append(dict(cycle=k, n_prod=len(prod.raw_candidates), n_engine=len(r.candidates),
                           unmatched_prod=len(cmp["unmatched_a"]),
                           extra_engine=len(cmp["unmatched_b"]), max_dsnr=cmp["max_dsnr"],
                           ok=good))
    return dict(dm=int(dm), cycles=cycles, ok=bool(ok and cycles))
