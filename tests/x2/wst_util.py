"""WS-T test helpers (not collected): WS-P value maps + frame generation.

Load from a test with::

    wu = load_wst_util()      # see the 6-line loader in each test module
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from dsart.transport import prod_frame2 as f2
from dsart.x2 import quant as q

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "x2" / "dsa110_antpos.npz"
DEC = 71.63


@functools.lru_cache(maxsize=None)
def antpos():
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    z = np.load(FIX)
    mask = core_baseline_mask_from_station_numbers(z["station"])
    return z["antpos_e"], z["antpos_n"], z["antpos_u"], mask


@functools.lru_cache(maxsize=None)
def opoint(kind: str = "gridded", n_sub: int = 2, bits_main: int = 8):
    from dsart.x2.opoint import opoint_2x
    return opoint_2x(kind, n_sub, bits_main)


@functools.lru_cache(maxsize=None)
def value_map(kind: str = "gridded", n_sub: int = 2, bits_main: int = 8):
    """WS-P's value map for a 2x format (fixture antenna positions, DEC 71.63)."""
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    e, n, u, mask = antpos()
    return build_value_map_for_opoint(opoint(kind, n_sub, bits_main), e, n, antpos_u=u,
                                      is_core_baseline_mask=mask, dec_deg=DEC)


@functools.lru_cache(maxsize=None)
def plan(n_sub: int = 2):
    from dsart.x2.plan import build_tiered_plan_for_opoint
    return build_tiered_plan_for_opoint(opoint("gridded", n_sub, 8))


def stream_blocks(vm, streams, *, rule: q.RowRule, n_samples: int, seed: int = 0,
                  sigma: float = 2.0, n_blocks: int = 2):
    """Row-rule codes for ``n_samples`` noise samples of each stream (after
    ``n_blocks - 1`` warm-up blocks). Returns {stream: BlockCodes}."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in streams:
        sv = vm.streams[s]
        rq = q.RowQuantiser(rule, sv.n_contrib, sv.n_heavy)
        x = None
        for _ in range(n_blocks):
            shape = (n_samples, sv.n_values)
            x = ((rng.normal(0, sigma, shape) + 1j * rng.normal(0, sigma, shape))
                 * np.sqrt(sv.n_contrib)[None, :]).astype(np.complex64)
            bc = rq.process_block(x)
        out[s] = bc
    return out


def datagrams_for(vm, bc, *, stream: int, dm: int, seq_first: int, t_int_factor: int,
                  rule: q.RowRule, k=None, extra_flags: int = 0) -> list[bytes]:
    return list(f2.iter_block_datagrams(
        codes_heavy=bc.codes_heavy, codes_main=bc.codes_main, seq_first=seq_first,
        t_int_factor=t_int_factor, stream_id=stream, dm_idx=dm, n_sub=vm.n_sub,
        bits_main=rule.bits_main, n_grid=vm.n_grid,
        value_kind=0 if vm.kind == "gridded" else 1,
        pattern_id=int(vm.stream_pattern_ids[stream]),
        step_main=float(bc.step_main), step_heavy=float(bc.step_heavy),
        s_main=float(rule.s_main_f32), s_heavy=float(rule.s_heavy_f32),
        midrise=rule.midrise, extra_flags=extra_flags, k=k))


def synthetic_frame(*, stream=0, dm=0, seq_first=0, k=2, n_values=20, n_heavy=2,
                    t_int_factor=16, n_sub=2, bits_main=8, n_grid=256, value_kind=0,
                    pattern_id=7, s_main=0.3534, s_heavy=0.1, fill=None, flags=0,
                    step_main=0.5, step_heavy=0.25) -> bytes:
    """A small valid v2 datagram whose sample i is filled with byte (fill or seq+i)."""
    bps = f2.bytes_per_sample(n_values, n_heavy, bits_main, 16 if n_heavy else 0)
    payload = np.empty((k, bps), np.uint8)
    for i in range(k):
        payload[i] = ((seq_first + i) if fill is None else fill) & 0xFF
    fl = f2.FLAG_QUANTIZED | (f2.FLAG_HEAVY_PRESENT if n_heavy else 0) | flags
    h = f2.Frame2Header(
        seq_first=seq_first, t0_native=seq_first * t_int_factor, stream_id=stream,
        dm_idx=dm, k=k, n_sub=n_sub, bits_main=bits_main,
        bits_heavy=16 if n_heavy else 0, n_grid=n_grid, t_int_factor=t_int_factor,
        value_kind=value_kind, n_values=n_values, pattern_id=pattern_id, n_heavy=n_heavy,
        step_main=step_main, step_heavy=step_heavy, payload_bytes=k * bps,
        s_main=float(np.float32(s_main)), s_heavy=float(np.float32(s_heavy)), flags=fl)
    return f2.pack_frame(h, payload.tobytes())
