"""Fused K3b: ONE Triton kernel for the whole filter bank (CP1 ``--k3b fused``).

v1 (``kernels_triton.k3b_detect``) launches once per filter and re-reads
the lagged image rows for each. This module GENERATES, per filter bank, a
single kernel that walks the centres once and, per centre:

* loads each DISTINCT input row offset the bank needs exactly once
  (``row = c + r``, r over the union of every filter's stage-1 read
  offsets ``d_k - o`` and ``d_k - o - L1``; for b3x 30 distinct rows) and
  forms ``xs = clip(fl32(x16) * sigma_inv[f], +-clip)`` once per row;
* updates every filter's cascade state (the twin's lag states,
  ``filters.FilterTime.lag_sets``) in registers -- the same fp32 operations
  in the same order as ``filters.k3b_twin`` for that filter;
* updates the per-(pixel, 64-centre chunk) maximum over all filters with
  ``better = v > cur  or  (v == cur and k < cur_k)``, which selects the same
  winner as the twin's filter-major strict ``>`` (max value, then lowest
  filter, then earliest centre), so values AND packed ``(t, k)`` equal the
  twin bit for bit;
* stores the raw outputs at the tap pixels.

Per-filter masking keeps the twin's semantics: an input row before the
filter's first needed row (``c_first - a_k``) reads as 0 for that filter.

The generated source is written to a cache dir (Triton's JIT reads source
files) and imported; ``mutate=`` produces deliberately wrong kernels for the
interpreter's negative tests. ``aot_compile`` compiles it for sm_75 without
a GPU (``b8 compile-check``).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["fused_source", "fused_kernel", "k3b_fused", "aot_compile"]

_CACHE: dict = {}


def _plan(bank) -> dict:
    lmax = bank.max_length
    amax = bank.max_a
    offs = set()
    fl = []
    for k, flt in enumerate(bank.filters):
        d, a = flt.d, flt.a
        if not flt.stages:
            offs.add(d)
            fl.append(dict(k=k, d=d, a=a, stages=(), lags=[]))
            continue
        lags = flt.lag_sets()
        L1 = flt.stages[0]
        for o in lags[0]:
            offs.add(d - o)
            offs.add(d - o - L1)
        fl.append(dict(k=k, d=d, a=a, stages=flt.stages, lags=lags))
    return dict(lmax=lmax, amax=amax, offs=sorted(offs), filters=fl)


def _v(r: int) -> str:
    return f"xm{-r}" if r < 0 else f"xp{r}"


def fused_source(bank, *, name: str = "k3b_fused", mutate: Optional[str] = None) -> str:
    """Triton source of the fused kernel for ``bank`` (pure function)."""
    p = _plan(bank)
    L = []
    w = L.append
    w("import triton")
    w("import triton.language as tl")
    w("")
    w("")
    w("@triton.jit")
    w(f"def {name}(ring_ptr, si_ptr, sinv_ptr, best_ptr, tk_ptr, tapidx_ptr, taps_ptr,")
    w("        R, F, P, row0, c_first, n_tap, clip,")
    w("        N_C: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr):")
    w("    pid_p = tl.program_id(0)")
    w("    f = tl.program_id(1)")
    w("    pix = pid_p * BLOCK + tl.arange(0, BLOCK)")
    w("    pm = pix < P")
    w("    si = tl.load(si_ptr + f)")
    w("    FP = F * P")
    w("    base = f * P + pix")
    w("    tap = tl.load(tapidx_ptr + pix, mask=pm, other=-1)")
    for fk in p["filters"]:
        w(f"    sk{fk['k']} = tl.load(sinv_ptr + {fk['k']})")
        for j, os_ in enumerate(fk["lags"]):
            for o in os_:
                w(f"    s{fk['k']}_{j}_{o} = tl.zeros((BLOCK,), dtype=tl.float32)")
    w("    cur = tl.full((BLOCK,), float('-inf'), dtype=tl.float32)")
    w("    ctk = tl.zeros((BLOCK,), dtype=tl.int32)")
    w("    curk = tl.full((BLOCK,), 1 << 20, dtype=tl.int32)")
    w(f"    for it in range(0, N_C + {p['lmax'] - 1}):")
    w(f"        c = c_first - {p['lmax'] - 1} + it")
    for r in p["offs"]:
        v = _v(r)
        w(f"        row_{v} = c + ({r})")
        w(f"        ok_{v} = pm & (row_{v} >= c_first - {p['amax']})")
        w(f"        sl_{v} = (((row_{v} - row0) % R) + R) % R")
        w(f"        {v} = tl.load(ring_ptr + sl_{v} * FP + base, mask=ok_{v}, other=0.0)"
          ".to(tl.float32)")
        w(f"        {v} = tl.minimum(tl.maximum({v} * si, -clip), clip)")
    for fk in p["filters"]:
        k, d, a = fk["k"], fk["d"], fk["a"]

        def xs(r, k=k, a=a):
            return f"tl.where(row_{_v(r)} >= c_first - {a}, {_v(r)}, 0.0)"
        if not fk["stages"]:
            w(f"        score{k} = {xs(d)}")
            continue
        L1 = fk["stages"][0]
        for o in fk["lags"][0]:
            r_in, r_out = d - o, d - o - L1
            if mutate == "offset" and k == len(p["filters"]) - 1 and o == 0:
                # read the nearest OTHER loaded row (always a real change)
                r_in = min((r for r in p["offs"] if r != r_in), key=lambda r: abs(r - r_in))
            w(f"        s{k}_0_{o} = (s{k}_0_{o} + {xs(r_in)}) - {xs(r_out)}")
        for j in range(1, len(fk["stages"])):
            Lj = fk["stages"][j]
            for o in fk["lags"][j]:
                w(f"        s{k}_{j}_{o} = (s{k}_{j}_{o} + s{k}_{j - 1}_{o}) - s{k}_{j - 1}_{o + Lj}")
        w(f"        score{k} = s{k}_{len(fk['stages']) - 1}_0")
    w("        out_c = c >= c_first")
    w("        ci = tl.maximum(c - c_first, 0)")
    w("        chunk = ci // CHUNK")
    w("        tin = ci % CHUNK")
    w("        is_start = out_c & (tin == 0)")
    w("        cur = tl.where(is_start, float('-inf'), cur)")
    w("        ctk = tl.where(is_start, 0, ctk)")
    w("        curk = tl.where(is_start, 1 << 20, curk)")
    for fk in p["filters"]:
        k = fk["k"]
        w(f"        tl.store(taps_ptr + (({k} * F + f) * n_tap + tap) * N_C + ci, score{k},"
          " mask=pm & (tap >= 0) & out_c)")
        w(f"        v{k} = score{k} * sk{k}")
        if mutate == "tie":
            w(f"        b{k} = (v{k} >= cur) & out_c")
        else:
            w(f"        b{k} = ((v{k} > cur) | ((v{k} == cur) & ({k} < curk))) & out_c")
        w(f"        cur = tl.where(b{k}, v{k}, cur)")
        w(f"        ctk = tl.where(b{k}, tin * 256 + {k}, ctk)")
        w(f"        curk = tl.where(b{k}, {k}, curk)")
    w("        is_end = out_c & (tin == CHUNK - 1)")
    w("        tl.store(best_ptr + chunk * FP + base, cur, mask=pm & is_end)")
    w("        tl.store(tk_ptr + chunk * FP + base, ctk, mask=pm & is_end)")
    return "\n".join(L) + "\n"


def _cache_dir() -> Path:
    d = Path(os.environ.get("DSART_X2_KCACHE", Path(tempfile.gettempdir()) / "dsart_x2_kernels"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fused_kernel(bank, *, mutate: Optional[str] = None):
    """The jitted kernel for ``bank`` (generated once per bank / mutation)."""
    src = fused_source(bank, mutate=mutate)
    h = hashlib.blake2b(src.encode(), digest_size=8).hexdigest()
    key = (h, os.environ.get("TRITON_INTERPRET", "0"))
    if key in _CACHE:
        return _CACHE[key]
    path = _cache_dir() / f"dsart_x2_k3b_{h}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(f"dsart_x2_k3b_{h}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[key] = mod.k3b_fused
    return mod.k3b_fused


def k3b_fused(ring, *, ring_row0: int, c_first: int, n_c: int, sigma_inv, s_inv, bank,
              clip: float = 250.0, chunk: int = 64, tap_pixels=None, block: int = 128,
              mutate: Optional[str] = None):
    """Same contract and outputs as ``kernels_triton.k3b_detect`` (one launch)."""
    import torch
    import triton

    R, F, P = ring.shape
    dev = ring.device
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    n_ch = n_c // chunk
    best = torch.empty((n_ch, F, P), dtype=torch.float32, device=dev)
    tk = torch.zeros((n_ch, F, P), dtype=torch.int32, device=dev)
    tapidx = torch.full((P,), -1, dtype=torch.int32, device=dev)
    n_tap = 0
    if tap_pixels is not None:
        tp = torch.as_tensor(np.asarray(tap_pixels), dtype=torch.long, device=dev)
        n_tap = int(tp.numel())
        tapidx[tp] = torch.arange(n_tap, dtype=torch.int32, device=dev)
    K = len(bank)
    taps = torch.zeros((K, F, max(n_tap, 1), n_c), dtype=torch.float32, device=dev)
    sinv = torch.as_tensor(np.asarray(s_inv, dtype=np.float32), device=dev)
    kern = fused_kernel(bank, mutate=mutate)
    kern[(triton.cdiv(P, block), F)](ring, sigma_inv, sinv, best, tk, tapidx, taps,
                                     int(R), int(F), int(P), int(ring_row0), int(c_first),
                                     int(max(n_tap, 1)), float(clip),
                                     N_C=int(n_c), CHUNK=int(chunk), BLOCK=int(block))
    return best, tk, (taps if n_tap else None)


def aot_compile(bank, *, arch: int = 75, n_c: int = 384, block: int = 128) -> None:
    """Compile-only for ``sm_<arch>`` (no GPU, no launch)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = fused_kernel(bank)
    types = dict(ring_ptr="*fp16", si_ptr="*fp32", sinv_ptr="*fp32", best_ptr="*fp32",
                 tk_ptr="*i32", tapidx_ptr="*i32", taps_ptr="*fp32", R="i32", F="i32",
                 P="i32", row0="i32", c_first="i32", n_tap="i32", clip="fp32")
    consts = dict(N_C=n_c, CHUNK=64, BLOCK=block)
    names = fn.arg_names
    sig = {i: types[n] for i, n in enumerate(names) if n not in consts}
    cst = {i: consts[n] for i, n in enumerate(names) if n in consts}
    triton.compile(ASTSource(fn=fn, signature=sig, constants=cst),
                   target=GPUTarget("cuda", arch, 32))
