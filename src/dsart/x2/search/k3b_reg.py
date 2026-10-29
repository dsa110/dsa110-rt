"""K3b with register-resident delay lines (``--k3b reg``), bit-exact vs ``filters.k3b_twin``.

The twin's B-spline cascade keeps one running-sum state per (stage, lag);
by the time-shift identity (every state is +0.0 before the filter's first
centre and every input row before ``c_first - a`` is masked to +0.0) the lag-o
state at step t equals the lag-0 state at step t - o EXACTLY. So each stage
needs ONE running sum plus a delay line of its own past values:

    s1 = (s1 + x(t)) - x(t - L1)            x from the input ring
    s2 = (s2 + s1(t)) - s1(t - L2)          s1 from its delay line
    s3 = (s3 + s2(t)) - s2(t - L3)          s2 from its delay line

-- the same fp32 operations on the same values as the twin's lag states, 6
adds per step for a 3-stage filter instead of 14. All delay lines live in
REGISTERS: the step loop is unrolled by a period ``U`` that every ring size
divides, so ring slots are compile-time register names (no shared memory, no
barriers, no address arithmetic per read). One thread owns one (pixel, fine
DM, part of the bank); the bank is split into parts so each part's rings fit
the register file (``plan_parts``). A thread loads the newest input row of
its part once per step (a small prefetch queue hides the global latency),
scales by the Layer-1 ``sigma_inv`` and clips at 250 sigma once.

Per-filter masks: the filter's x_in is masked before its first step
``st_k = LMAX - L_k`` and its x_out before ``st_k + L1`` (rows before
``c_first - a``), exactly the twin's ``xs_at`` masking. The per-(pixel,
chunk) winner of a part is the 64-bit key of ``k3b_cuda`` (max value, then
lowest filter, then earliest centre); parts are combined by
``k3b_cuda.COMBINE_SRC``. The step loop runs to a multiple of ``U``; steps
past the last centre write nothing.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

__all__ = ["reg_source", "plan_parts", "k3b_reg"]

_KERNELS: dict = {}
# per-call host cost matters (one call per DM-cycle): the bank's parts and
# each (part, options) kernel are resolved once per bank OBJECT (kept alive
# in the value so an id is never reused while cached)
_PARTS: dict = {}
_BY_KEY: dict = {}


def _req(bank, ks) -> dict:
    """Ring requirements of a part: newest offset D, x-ring minimum size and
    per-filter s1 / s2 ring minimum sizes."""
    D = max(bank.filters[k].d for k in ks)
    need_x = 1
    rings = {}
    for k in ks:
        f = bank.filters[k]
        age_in = D - f.d
        need_x = max(need_x, age_in + 1)
        if f.stages:
            need_x = max(need_x, age_in + f.stages[0] + 1)
        if f.order >= 2:
            rings[(k, 1)] = f.stages[1] + 1
        if f.order >= 3:
            rings[(k, 2)] = f.stages[2] + 1
        if f.order > 3:
            raise ValueError(f"{f.name}: > 3 stages not supported")
    return dict(D=D, need_x=need_x, rings=rings)


def _ring_sizes(req: dict, U: int) -> Optional[dict]:
    divs = [d for d in range(1, U + 1) if U % d == 0]

    def fit(n):
        c = [d for d in divs if d >= n]
        return c[0] if c else None
    out = {"x": fit(req["need_x"])}
    for key, n in req["rings"].items():
        out[key] = fit(n)
    if any(v is None for v in out.values()):
        return None
    return out


def _choose_U(req: dict, dq: int, umax: int = 192) -> tuple:
    best = None
    lo = max([req["need_x"]] + list(req["rings"].values()))
    for U in range(lo, umax + 1):
        if U % dq:
            continue
        rs = _ring_sizes(req, U)
        if rs is None:
            continue
        regs = sum(rs.values())
        cost = regs + U // 2                      # registers, and code size (unrolled steps)
        if 64 % U == 0 or U % 64 == 0:
            cost -= 12                            # chunk-aligned: no per-step chunk logic
        if best is None or cost < best[3]:
            best = (U, regs, rs, cost)
    if best is None:
        raise ValueError("no unroll period fits")
    return best[:3]


def plan_parts(bank) -> list:
    """Parts of the bank, one thread each: the widest filters alone, then
    pairs, then everything short together (register budget ~180)."""
    order = sorted(range(len(bank)), key=lambda k: -bank.filters[k].length)
    parts, cur = [], []
    for k in order:
        trial = sorted(cur + [k])
        req = _req(bank, trial)
        try:
            _U, regs, _rs = _choose_U(req, 4)
        except ValueError:
            regs = 10 ** 9
        n3 = sum(1 for kk in trial if bank.filters[kk].order >= 3)
        if cur and (regs > 150 or n3 > 2):
            parts.append(sorted(cur))
            cur = [k]
        else:
            cur = trial
    if cur:
        parts.append(sorted(cur))
    return parts


def reg_source(bank, part: Sequence[int], *, dq: int = 4, name: str = "k3b_reg",
               threads: int = 128, chunk: int = 64, mode: str = "keys") -> tuple:
    """Kernel source for filters ``part``. ``mode="keys"``: per-(pixel, chunk)
    keys for every pixel, and the raw scores of the tap pixels (``tappix`` is
    then the per-pixel tap index, -1 = not a tap: a predicated store in the
    same pass); ``mode="taps"``: raw scores of the listed tap pixels only
    (thread x = tap index; a separate long serial launch, kept for tests)."""
    ks = sorted(part)
    req = _req(bank, ks)
    U, regs, rs = _choose_U(req, dq)
    D = req["D"]
    L0 = max(bank.filters[k].length for k in ks)
    aligned = (64 % U == 0) or (U % 64 == 0)
    # leading zero steps so that the first output centre is an outer-iteration
    # boundary (exact: a filter's states stay +0.0 until its inputs unmask)
    S = (-(L0 - 1)) % (min(U, chunk) if aligned else 1)
    LMAX = L0 + S
    UX = rs["x"]
    st = {k: LMAX - bank.filters[k].length for k in ks}
    single = len(ks) == 1
    mask_end = 0 if single else max(
        st[k] + (bank.filters[k].stages[0] if bank.filters[k].stages else 0) for k in ks)
    L = []
    w = L.append
    w("#include <cuda_fp16.h>")
    w("#define K3B_NEG_INF __int_as_float(0xff800000)")
    w(f"#define LMAX {LMAX}")
    w(f"#define CHUNK {chunk}")
    w("__device__ __forceinline__ unsigned long long k3b_key(float v, int k, int t) {")
    w("    v = __fadd_rn(v, 0.0f);")
    w("    unsigned int u = __float_as_uint(v);")
    w("    unsigned int o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);")
    w("    return ((unsigned long long)o << 32) | (unsigned long long)(((255 - k) << 8) | (255 - t));")
    w("}")
    w(f'extern "C" __global__ void __launch_bounds__({threads}) {name}(')
    w("    const __half* __restrict__ ring, const float* __restrict__ si_,")
    w("    const float* __restrict__ sk_, unsigned long long* __restrict__ keys,")
    w("    const int* __restrict__ tappix, float* __restrict__ taps,")
    w("    int R, int F, int P, long long row0, long long c_first, int n_c, int n_tap, float clip)")
    w("{")
    if mode == "taps":
        w("    const int ti = blockIdx.x * blockDim.x + threadIdx.x;")
        w("    if (ti >= n_tap) return;")
        w("    const int p = tappix[ti];")
    else:
        w("    const int p = blockIdx.x * blockDim.x + threadIdx.x;")
        w("    if (p >= P) return;")
        w("    const int ti = tappix[p];                          /* tap index or -1 */")
    w("    const int f = blockIdx.y;")
    w("    const float si = si_[f];")
    w("    const __half* src = ring + (long long)f * P + p;")
    w("    const int NIT = n_c + LMAX - 1;")
    w("    const int FPi = F * P, RFP = R * F * P;            /* host checks R*F*P < 2^31 */")
    w(f"    int sl = (int)((c_first - (LMAX - 1) + {D} - row0) % R); if (sl < 0) sl += R;")
    w("    int off = sl * FPi;")
    w(f"    float X[{UX}];")
    w(f"    #pragma unroll\n    for (int i = 0; i < {UX}; ++i) X[i] = 0.0f;")
    for k in ks:
        flt = bank.filters[k]
        if mode == "keys":
            w(f"    const float sk{k} = sk_[{k}];")
            w(f"    float cur{k} = K3B_NEG_INF; int ct{k} = 0;")
        for jj in range(flt.order):
            w(f"    float s{k}_{jj} = 0.0f;")
        for jj in (1, 2):
            if (k, jj) in rs:
                w(f"    float R{k}_{jj}[{rs[(k, jj)]}];")
                w(f"    #pragma unroll\n    for (int i = 0; i < {rs[(k, jj)]}; ++i) R{k}_{jj}[i] = 0.0f;")
    w(f"    __half Q[{dq}];")
    w(f"    #pragma unroll\n    for (int i = 0; i < {dq}; ++i) {{ Q[i] = src[off]; "
      "off += FPi; if (off >= RFP) off -= RFP; }")

    def body(masked: bool, ind: str, chunk_static: bool) -> None:
        w(f"{ind}#pragma unroll")
        w(f"{ind}for (int j = 0; j < {U}; ++j) {{")
        w(f"{ind}    const int it = it0 + j;")
        prep = (f"fminf(fmaxf(__fmul_rn(__half2float(Q[j % {dq}]), si), -clip), clip)")
        if single:                     # the row is valid for the part's only filter iff it >= st
            prep = f"(it >= {st[ks[0]]}) ? {prep} : 0.0f"
        w(f"{ind}    const float xs = {prep};")
        w(f"{ind}    Q[j % {dq}] = src[off]; off += FPi; if (off >= RFP) off -= RFP;")
        w(f"{ind}    X[j % {UX}] = xs;")
        for k in ks:
            flt = bank.filters[k]
            a_in = D - flt.d

            def xat(age):
                return f"X[(j + {UX * 64} - {age}) % {UX}]"
            if not flt.stages:
                v = f"(it >= {st[k]}) ? {xat(a_in)} : 0.0f" if masked else xat(a_in)
                w(f"{ind}    const float score{k} = {v};")
                continue
            L1 = flt.stages[0]
            xi = f"((it >= {st[k]}) ? {xat(a_in)} : 0.0f)" if masked else xat(a_in)
            xo = f"((it >= {st[k] + L1}) ? {xat(a_in + L1)} : 0.0f)" if masked else xat(a_in + L1)
            w(f"{ind}    s{k}_0 = __fsub_rn(__fadd_rn(s{k}_0, {xi}), {xo});")
            for jj in (1, 2):
                if jj >= flt.order:
                    break
                Lj = flt.stages[jj]
                n = rs[(k, jj)]
                w(f"{ind}    {{ const float old = R{k}_{jj}[(j + {n * 64} - {Lj}) % {n}];")
                w(f"{ind}      R{k}_{jj}[j % {n}] = s{k}_{jj - 1};")
                w(f"{ind}      s{k}_{jj} = __fsub_rn(__fadd_rn(s{k}_{jj}, s{k}_{jj - 1}), old); }}")
            w(f"{ind}    const float score{k} = s{k}_{flt.order - 1};")
        w(f"{ind}    if ({'' if mode == 'taps' else 'ti >= 0 && '}it >= LMAX - 1 && it < NIT) {{")
        for k in ks:
            w(f"{ind}        taps[(((long long){k} * F + f) * n_tap + ti) * n_c + (it - (LMAX - 1))] "
              f"= score{k};")
        w(f"{ind}    }}")
        if mode == "taps":
            pass
        elif chunk_static:
            # the outer iteration lies inside one chunk (U | 64) or spans whole chunks (64 | U)
            w(f"{ind}    {{ const int tin = (tin0 + j) & (CHUNK - 1);")
            if U % 64 == 0:
                w(f"{ind}      if (j % CHUNK == 0) {{")
                for k in ks:
                    w(f"{ind}          cur{k} = K3B_NEG_INF;")
                w(f"{ind}      }}")
            for k in ks:
                w(f"{ind}      {{ const float v = __fmul_rn(score{k}, sk{k});")
                w(f"{ind}        if (v > cur{k}) {{ cur{k} = v; ct{k} = tin; }} }}")
            if U % 64 == 0:
                w(f"{ind}      if (j % CHUNK == CHUNK - 1) {{")
                w(f"{ind}          const int ch = (it - (LMAX - 1)) / CHUNK;")
                key_store(ind + "          ", "ch")
                w(f"{ind}      }}")
            w(f"{ind}    }}")
        else:
            w(f"{ind}    if (it >= LMAX - 1 && it < NIT) {{")
            w(f"{ind}        const int ci = it - (LMAX - 1);")
            w(f"{ind}        const int tin = ci & (CHUNK - 1);")
            for k in ks:
                w(f"{ind}        if (tin == 0) cur{k} = K3B_NEG_INF;")
                w(f"{ind}        {{ const float v = __fmul_rn(score{k}, sk{k});")
                w(f"{ind}          if (v > cur{k}) {{ cur{k} = v; ct{k} = tin; }} }}")
            w(f"{ind}        if (tin == CHUNK - 1) {{")
            key_store(ind + "            ", "ci / CHUNK")
            w(f"{ind}        }}")
            w(f"{ind}    }}")
        w(f"{ind}}}")

    def key_store(ind: str, ch: str) -> None:
        w(f"{ind}unsigned long long key = k3b_key(cur{ks[0]}, {ks[0]}, ct{ks[0]});")
        for k in ks[1:]:
            w(f"{ind}{{ const unsigned long long o = k3b_key(cur{k}, {k}, ct{k}); "
              "if (o > key) key = o; }")
        w(f"{ind}keys[((long long)({ch}) * F + f) * P + p] = key;")

    chunk_static = aligned and mode == "keys"
    w(f"    for (int it0 = 0; it0 < NIT; it0 += {U}) {{")
    if chunk_static:
        w("        const bool outb = it0 >= LMAX - 1;")
        w("        const int tin0 = (it0 - (LMAX - 1)) & (CHUNK - 1);")
        if U < 64:
            w("        if (outb && tin0 == 0) {")
            for k in ks:
                w(f"            cur{k} = K3B_NEG_INF;")
            w("        }")
    w(f"        if (it0 < {mask_end}) {{")
    if mask_end:
        body(True, "            ", False)              # masked: generic per-step output
    w("        } else {")
    if chunk_static:
        w("            if (outb) {")
        body(False, "                ", True)
        w("            } else {")
        warm = [it0 for it0 in range(0, LMAX - 1, U) if it0 >= mask_end]
        if warm:
            body(False, "                ", False)    # unmasked, still warming up
        w("            }")
        if U < 64:
            w("            if (outb && tin0 + " + str(U) + " == CHUNK) {")
            w("                const int ch = (it0 - (LMAX - 1)) / CHUNK;")
            key_store("                ", "ch")
            w("            }")
    else:
        body(False, "            ", False)
    w("        }")
    w("    }")
    w("}")
    src = "\n".join(L) + "\n"
    meta = dict(name=name, part=ks, U=U, ring_regs=regs, rings=rs, D=D, lmax=LMAX, pad=S,
                threads=threads, dq=dq, aligned=chunk_static, mask_end=mask_end, mode=mode)
    return src, meta


def get_kernel(bank, part, **kw):
    key = (id(bank), tuple(part), tuple(sorted(kw.items())))
    hit = _BY_KEY.get(key)
    if hit is not None and hit[0] is bank:
        return hit[1]
    src, meta = reg_source(bank, part, **kw)
    h = hashlib.blake2b(src.encode(), digest_size=8).hexdigest()
    if h not in _KERNELS:
        import cupy as cp
        _KERNELS[h] = (cp.RawKernel(src, meta["name"], options=("--std=c++14",),
                                    backend="nvrtc"), meta)
    _BY_KEY[key] = (bank, _KERNELS[h])
    return _KERNELS[h]


def k3b_reg(ring, *, ring_row0: int, c_first: int, n_c: int, sigma_inv, s_inv, bank,
            clip: float = 250.0, chunk: int = 64, tap_pixels=None, stream=None,
            parts: Optional[list] = None, gen: Optional[dict] = None, taps_inline: bool = True):
    """Same contract and outputs as ``kernels_triton.k3b_detect``: one keys
    launch per part (which also stores the tap pixels' raw scores), one
    combine. ``taps_inline=False``: the taps come from a separate small launch
    per part over the tap pixels instead -- its 256 threads x F run the whole
    serial recursion on a nearly empty GPU (+0.59 us/image at F 35, n_c 384,
    n13), kept as the cross-check."""
    import cupy as cp
    import torch

    from ...image.tiled_combine_cuda import _ptr
    from .k3b_cuda import _combine

    R, F, P = ring.shape
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    if R * F * P >= 2 ** 31:
        raise ValueError(f"image ring {R}x{F}x{P} too large for 32-bit offsets")
    if parts is None:
        hit = _PARTS.get(id(bank))
        if hit is None or hit[0] is not bank:
            hit = _PARTS[id(bank)] = (bank, plan_parts(bank))
        parts = hit[1]
    n_ch = n_c // chunk
    dev = ring.device
    keys = torch.empty((len(parts), n_ch, F, P), dtype=torch.int64, device=dev)
    best = torch.empty((n_ch, F, P), dtype=torch.float32, device=dev)
    tk = torch.empty((n_ch, F, P), dtype=torch.int32, device=dev)
    n_tap = 0
    tappix = torch.zeros((1,), dtype=torch.int32, device=dev)
    tapidx = torch.full((P,), -1, dtype=torch.int32, device=dev)
    # device tensors pass straight through; a numpy argument is a pageable
    # H2D, which torch completes with a host-side stream sync
    if tap_pixels is not None:
        tappix = (tap_pixels.to(device=dev, dtype=torch.int32)
                  if isinstance(tap_pixels, torch.Tensor)
                  else torch.as_tensor(np.asarray(tap_pixels), dtype=torch.int32, device=dev))
        n_tap = int(tappix.numel())
        tapidx[tappix.long()] = torch.arange(n_tap, dtype=torch.int32, device=dev)
    K = len(bank)
    # every (k, f, tap, centre) is written by the taps kernels: no zero fill
    taps = torch.empty((K, F, max(n_tap, 1), n_c), dtype=torch.float32, device=dev)
    si = torch.as_tensor(sigma_inv, dtype=torch.float32, device=dev).contiguous()
    sk = (s_inv.to(device=dev, dtype=torch.float32) if isinstance(s_inv, torch.Tensor)
          else torch.as_tensor(np.asarray(s_inv, dtype=np.float32), device=dev))
    ring = ring.contiguous()
    st = cp.cuda.ExternalStream((stream or torch.cuda.current_stream()).cuda_stream)
    args_tail = (np.int32(R), np.int32(F), np.int32(P), np.int64(ring_row0), np.int64(c_first),
                 np.int32(n_c), np.int32(max(n_tap, 1)), np.float32(clip))
    with st:
        for i, part in enumerate(parts):
            kern, meta = get_kernel(bank, part, chunk=chunk, **(gen or {}))
            th = meta["threads"]
            kern(((P + th - 1) // th, F), (th,),
                 (_ptr(ring), _ptr(si), _ptr(sk), _ptr(keys[i]), _ptr(tapidx), _ptr(taps))
                 + args_tail)
            if n_tap and not taps_inline:
                tkern, _tmeta = get_kernel(bank, part, chunk=chunk, mode="taps",
                                           name="k3b_reg_taps", **(gen or {}))
                tkern(((n_tap + 127) // 128, F), (128,),
                      (_ptr(ring), _ptr(si), _ptr(sk), _ptr(keys[i]), _ptr(tappix), _ptr(taps))
                      + args_tail)
        n = n_ch * F * P
        _combine()((min(4096, (n + 255) // 256),), (256,),
                   (_ptr(keys), np.int32(len(parts)), np.int64(n), _ptr(best), _ptr(tk)))
    return best, tk, (taps if n_tap else None)
