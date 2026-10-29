"""K3b as generated CUDA kernels (``--k3b cuda``), bit-exact vs ``filters.k3b_twin``.

The detector is a long sequential recursion per (pixel, fine DM): 384 output
centres (tier A) plus each filter's own warm-up from zero state, through
running-sum cascades whose lag states read the input at up to ~30 offsets
spanning ~150 rows (q64). On a 2080 Ti it is latency-bound unless many
independent recursions are in flight per SM, and the input history is what
limits that. So:

* the bank is split into KERNELS by span (``plan_kernels``): the widest
  B-spline alone (q64: 152-row history), everything else together (<= 77
  rows). Each kernel keeps the history of its own span only;
* the history is a shared-memory ring of the RAW fp16 image rows (exact --
  the image ring is fp16), filled ``BATCH`` rows at a time with the global
  loads of the next batch in flight during the current one (one barrier per
  batch). Scaling by the Layer-1 ``sigma_inv`` and the 250-sigma clip
  (``clip(fl32(x16) * si)``, the twin's ``xs``) happen at read time, once per
  distinct offset per step;
* within a kernel the filters are split into thread GROUPS (one warp each:
  every 3-stage B-spline alone, short filters together); a filter starts at
  its OWN first centre ``c_first - (L - 1)`` as the twin does;
* each filter runs the twin's lag-state recursion in the twin's fp32 order
  (``(s + x_in) - x_out`` then the later stages, ``__fadd_rn`` / ``__fsub_rn``)
  with the twin's per-filter input masks (rows before ``c_first - a`` read
  +0.0) and strict ``>`` chunk maximum;
* the per-(pixel, chunk) winner is a 64-bit key ``ord(v) | (255 - k) | (255 -
  t)`` -- max value, then lowest filter, then earliest centre, which is the
  twin's filter-major strict-``>`` order -- combined across groups with a
  shared-memory ``atomicMax``, stored per kernel, and across kernels by
  :data:`COMBINE_SRC`. ``v`` is canonicalised (``-0 -> +0``) so a tie between
  signed zeros goes to the lower filter as in the twin (the value then reads
  +0.0, which compares equal).

Taps (raw scores at the Layer-2 tap pixels) are written by the owning group.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

__all__ = ["cuda_source", "plan_groups", "plan_kernels", "k3b_cuda", "get_kernel",
           "COMBINE_SRC", "cpu_source_check"]

_KERNELS: dict = {}

COMBINE_SRC = r"""
extern "C" __global__ void k3b_combine(const unsigned long long* __restrict__ keys, int n_parts,
                                       long long n, float* __restrict__ best,
                                       int* __restrict__ tk)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned long long key = keys[i];
        for (int j = 1; j < n_parts; ++j) { const unsigned long long o = keys[j * n + i];
                                            if (o > key) key = o; }
        const unsigned int o = (unsigned int)(key >> 32);
        const unsigned int u = (o & 0x80000000u) ? (o & 0x7fffffffu) : ~o;
        const int k = 255 - (int)((key >> 8) & 0xFF), t = 255 - (int)(key & 0xFF);
        best[i] = __uint_as_float(u);
        tk[i] = (t << 8) | k;
    }
}
"""


def _cost(flt) -> int:
    if not flt.stages:
        return 5
    return 2 * sum(len(s) for s in flt.lag_sets()) + 4


def plan_groups(bank, subset: Optional[Sequence[int]] = None, *, max_cost: int = 24) -> list:
    """Filters per thread group (bank order kept inside a group): a filter of
    cost >= 12 (any 3-stage B-spline) is a group of its own; cheaper ones are
    packed in bank order up to ``max_cost``."""
    ks = list(range(len(bank))) if subset is None else list(subset)
    groups, cur, c = [], [], 0
    for k in ks:
        w = _cost(bank.filters[k])
        if w >= 12:
            if cur:
                groups.append(cur)
                cur, c = [], 0
            groups.append([k])
            continue
        if cur and c + w > max_cost:
            groups.append(cur)
            cur, c = [], 0
        cur.append(k)
        c += w
    if cur:
        groups.append(cur)
    return groups


def _reads(flt) -> list:
    if not flt.stages:
        return [flt.d]
    L1 = flt.stages[0]
    out = []
    for o in flt.lag_sets()[0]:
        out += [flt.d - o, flt.d - o - L1]
    return out


def _span(bank, ks) -> int:
    rr = [r for k in ks for r in _reads(bank.filters[k])]
    return max(rr) - min(rr) + 1


def plan_kernels(bank, *, wide: int = 100) -> list:
    """Filter subsets, one kernel each: every filter longer than ``wide``
    samples alone (its history would set everyone's), the rest together."""
    big = [k for k, f in enumerate(bank.filters) if f.length > wide]
    rest = [k for k in range(len(bank)) if k not in big]
    return ([rest] if rest else []) + [[k] for k in big]


def cuda_source(bank, subset: Optional[Sequence[int]] = None, *, pt: int = 32,
                batch: int = 16, unroll: int = 4, ring: str = "f32", maxch: int = 8,
                chunk: int = 64, name: str = "k3b_cuda") -> tuple:
    """``(source, meta)`` of the kernel for filters ``subset`` of ``bank``."""
    if chunk > 256 or len(bank) > 255:
        raise ValueError("chunk <= 256 and <= 255 filters (8-bit packed t, k)")
    if ring not in ("f16", "f32"):
        raise ValueError(f"ring={ring!r}")
    ks_all = list(range(len(bank))) if subset is None else list(subset)
    if chunk % batch:
        raise ValueError(f"batch {batch} must divide chunk {chunk}")
    # the recursion starts LMAX - 1 centres before c_first with LMAX >= every
    # filter's length, rounded so the first output centre is a batch start
    # (earlier steps of a filter only see masked +0.0 inputs: states stay
    # exactly +0.0, so the extra leading steps change nothing)
    LMAX = max(bank.filters[k].length for k in ks_all)
    LMAX += (-(LMAX - 1)) % batch
    groups = plan_groups(bank, ks_all)
    allr = [r for k in ks_all for r in _reads(bank.filters[k])]
    RMIN, RMAX = min(allr), max(allr)
    span = RMAX - RMIN + 1
    RS = span + 2 * batch - 1
    NG = len(groups)
    T = "__half" if ring == "f16" else "float"
    esz = 2 if ring == "f16" else 4
    L = []
    w = L.append
    w("#include <cuda_fp16.h>")
    w(f"#define PT {pt}")
    w(f"#define NG {NG}")
    w(f"#define RS {RS}")
    w(f"#define SPAN {span}")
    w(f"#define RMIN ({RMIN})")
    w(f"#define LMAX {LMAX}")
    w(f"#define BATCH {batch}")
    w(f"#define MAXCH {maxch}")
    w(f"#define CHUNK {chunk}")
    w("#define K3B_NEG_INF __int_as_float(0xff800000)")
    w(f"#define EPT {-(-(batch * pt) // (pt * NG))}")
    w("__device__ __forceinline__ unsigned long long k3b_key(float v, int k, int t) {")
    w("    v = __fadd_rn(v, 0.0f);                       /* -0 -> +0 */")
    w("    unsigned int u = __float_as_uint(v);")
    w("    unsigned int o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);")
    w("    return ((unsigned long long)o << 32) | (unsigned long long)(((255 - k) << 8) | (255 - t));")
    w("}")
    w(f'extern "C" __global__ void __launch_bounds__({pt * NG}) {name}(')
    w("    const __half* __restrict__ ring, const float* __restrict__ si_,")
    w("    const float* __restrict__ sk_, unsigned long long* __restrict__ keys,")
    w("    const int* __restrict__ tapidx, float* __restrict__ taps,")
    w("    int R, int F, int P, long long row0, long long c_first, int n_c, int n_tap, float clip)")
    w("{")
    w(f"    __shared__ {T} xr[RS + SPAN - 1][PT];   /* ring + mirrored head: reads never wrap */")
    w("    __shared__ unsigned long long res[MAXCH][PT];")
    w("    const int px = threadIdx.x, g = threadIdx.y;")
    w("    const int tid = g * PT + px, nth = PT * NG;")
    w("    const int p = blockIdx.x * PT + px;")
    w("    const int f = blockIdx.y;")
    w("    const bool pm = p < P;")
    w("    const float si = si_[f];")
    w("    const long long FP = (long long)F * P;")
    w("    const long long qrow0 = c_first - (LMAX - 1) + RMIN;   /* absolute row of ring index 0 */")
    w("    int slb = (int)((qrow0 - row0) % R); if (slb < 0) slb += R;  /* image slot of q = 0 */")
    w("    const __half* ringf = ring + (long long)f * P;")
    w("    const int NIT = n_c + LMAX - 1;")
    w("    const int QEND = NIT - 1 + SPAN;                     /* ring indices needed: [0, QEND) */")
    w("    const int n_ch = n_c / CHUNK;")
    w("    for (int i = tid; i < MAXCH * PT; i += nth) (&res[0][0])[i] = 0ULL;")
    w("    __half ph[EPT]; int pq[EPT];")

    def pload(q0: str, ind: str) -> None:
        w(f"{ind}#pragma unroll")
        w(f"{ind}for (int e = 0; e < EPT; ++e) {{")
        w(f"{ind}    const int i = tid + e * nth; pq[e] = -1; ph[e] = __ushort_as_half(0);")
        w(f"{ind}    if (i < BATCH * PT) {{")
        w(f"{ind}        const int q = ({q0}) + i / PT, pp = i % PT, pg = blockIdx.x * PT + pp;")
        w(f"{ind}        if (q >= 0 && q < QEND) {{ pq[e] = q; if (pg < P) {{")
        w(f"{ind}            int sl = slb + q; if (sl >= R) sl -= R; if (sl >= R) sl -= R;")
        w(f"{ind}            ph[e] = ringf[(long long)sl * FP + pg]; }} }} }} }}")

    def pstore(ind: str) -> None:
        w(f"{ind}#pragma unroll")
        w(f"{ind}for (int e = 0; e < EPT; ++e) if (pq[e] >= 0)")
        if ring == "f16":
            w(f"{ind}    {{ const int sl = pq[e] % RS, pp = (tid + e * nth) % PT; xr[sl][pp] = ph[e];")
            w(f"{ind}      if (sl < SPAN - 1) xr[sl + RS][pp] = ph[e]; }}")
        else:
            w(f"{ind}    {{ const int sl = pq[e] % RS, pp = (tid + e * nth) % PT;")
            w(f"{ind}      const float v = fminf(fmaxf(__fmul_rn(__half2float(ph[e]), si), -clip), clip);")
            w(f"{ind}      xr[sl][pp] = v; if (sl < SPAN - 1) xr[sl + RS][pp] = v; }}")

    w("    for (int q0 = 0; q0 < SPAN - 1 + BATCH; q0 += BATCH) {")
    pload("q0", "        ")
    pstore("        ")
    w("    }")
    w("    __syncthreads();")
    w("    const int tap = pm ? tapidx[p] : -1;")
    for ks in groups:
        for k in ks:
            flt = bank.filters[k]
            w(f"    const float sk{k} = sk_[{k}];")
            w(f"    float cur{k} = K3B_NEG_INF; int ct{k} = 0;")
            for j, os_ in enumerate(flt.lag_sets()):
                for o in os_:
                    w(f"    float s{k}_{j}_{o} = 0.0f;")
    w("    for (int b0 = 0; b0 < NIT; b0 += BATCH) {")
    w("        const int sb0 = b0 % RS;")
    pload("b0 + SPAN - 1 + BATCH", "        ")
    w("        const bool outb = b0 >= LMAX - 1;                 /* uniform: output batch */")
    w("        const int tin0 = (b0 - (LMAX - 1)) & (CHUNK - 1);")
    w("        const int chb = (b0 - (LMAX - 1)) / CHUNK;")
    for gi, ks in enumerate(groups):
        st_g = min(LMAX - bank.filters[k].length for k in ks)
        w(f"        {'if' if gi == 0 else 'else if'} (g == {gi} && b0 + BATCH > {st_g}) {{")
        for out in (False, True):
            w(f"            {'if (!outb)' if not out else 'else'} {{")
            if out:
                w("                if (tin0 == 0) {")
                for k in ks:
                    w(f"                    cur{k} = K3B_NEG_INF;")
                w("                }")
            w(f"                #pragma unroll {unroll}")
            w("                for (int j = 0; j < BATCH; ++j) {")
            w("                    const int it = b0 + j;")
            w("                    int sb = sb0 + j; if (sb >= RS) sb -= RS;")
            w(f"                    const {T}* rowp = &xr[sb][px];")
            for k in ks:
                flt = bank.filters[k]
                reads = _reads(flt)
                dist = sorted(set(reads))
                ths = {r: LMAX - 1 - flt.a - r for r in dist}
                for r in dist:
                    rr = r - RMIN
                    rd = f"rowp[{rr * pt}]"
                    if ring == "f16":
                        rd = f"fminf(fmaxf(__fmul_rn(__half2float({rd}), si), -clip), clip)"
                    vn = f"x{k}_{'m' if r < 0 else 'p'}{abs(r)}"
                    if out and ths[r] <= LMAX - 1:          # never masked in output batches
                        w(f"                    const float {vn} = {rd};")
                    else:
                        w(f"                    const float {vn} = (it >= {ths[r]}) ? {rd} : 0.0f;")

                def X(r, k=k):
                    return f"x{k}_{'m' if r < 0 else 'p'}{abs(r)}"
                if not flt.stages:
                    w(f"                    const float score{k} = {X(flt.d)};")
                else:
                    L1 = flt.stages[0]
                    ls = flt.lag_sets()
                    for o in ls[0]:
                        w(f"                    s{k}_0_{o} = __fsub_rn(__fadd_rn(s{k}_0_{o}, "
                          f"{X(flt.d - o)}), {X(flt.d - o - L1)});")
                    for jj in range(1, flt.order):
                        Lj = flt.stages[jj]
                        for o in ls[jj]:
                            w(f"                    s{k}_{jj}_{o} = __fsub_rn(__fadd_rn("
                              f"s{k}_{jj}_{o}, s{k}_{jj - 1}_{o}), s{k}_{jj - 1}_{o + Lj});")
                    w(f"                    const float score{k} = s{k}_{flt.order - 1}_0;")
                if out:
                    w(f"                    {{ const float v = __fmul_rn(score{k}, sk{k});")
                    w(f"                      if (v > cur{k}) {{ cur{k} = v; ct{k} = tin0 + j; }} }}")
                    w(f"                    if (tap >= 0) taps[(((long long){k} * F + f) * n_tap + tap) "
                      f"* n_c + (b0 - (LMAX - 1) + j)] = score{k};")
            w("                }")
            if out:
                w("                if (tin0 + BATCH == CHUNK && pm) {")
                for k in ks:
                    w(f"                    atomicMax(&res[chb][px], k3b_key(cur{k}, {k}, ct{k}));")
                w("                }")
            w("            }")
        w("        }")
    pstore("        ")
    w("        __syncthreads();")
    w("    }")
    w("    if (g == 0 && pm)")
    w("        for (int ch = 0; ch < n_ch; ++ch)")
    w("            keys[((long long)ch * F + f) * P + p] = res[ch][px];")
    w("}")
    src = "\n".join(L) + "\n"
    meta = dict(name=name, pt=pt, ng=NG, rs=RS, span=span, groups=groups, maxch=maxch,
                chunk=chunk, lmax=LMAX, subset=ks_all,
                smem=(RS + span - 1) * pt * esz + maxch * pt * 8)
    return src, meta


def get_kernel(bank, subset=None, **kw):
    src, meta = cuda_source(bank, subset, **kw)
    h = hashlib.blake2b(src.encode(), digest_size=8).hexdigest()
    if h not in _KERNELS:
        import cupy as cp
        fn = cp.RawKernel(src, meta["name"], options=("--std=c++14",), backend="nvrtc")
        try:
            fn.preferred_shared_memory_carveout = 100        # all of L1 as shared memory
        except Exception:                                    # noqa: BLE001
            pass
        _KERNELS[h] = (fn, meta)
    return _KERNELS[h]


def _combine():
    if "combine" not in _KERNELS:
        import cupy as cp
        _KERNELS["combine"] = cp.RawKernel(COMBINE_SRC, "k3b_combine", backend="nvrtc")
    return _KERNELS["combine"]


def k3b_cuda(ring, *, ring_row0: int, c_first: int, n_c: int, sigma_inv, s_inv, bank,
             clip: float = 250.0, chunk: int = 64, tap_pixels=None, stream=None,
             gen: Optional[dict] = None, kernels: Optional[list] = None):
    """Same contract and outputs as ``kernels_triton.k3b_detect``; one launch
    per filter subset (``plan_kernels``) + one combine. ``gen`` =
    :func:`cuda_source` options (pt, batch, unroll, ring)."""
    import cupy as cp
    import torch

    from ...image.tiled_combine_cuda import _ptr

    R, F, P = ring.shape
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    subsets = kernels if kernels is not None else plan_kernels(bank)
    n_ch = n_c // chunk
    dev = ring.device
    keys = torch.empty((len(subsets), n_ch, F, P), dtype=torch.int64, device=dev)
    best = torch.empty((n_ch, F, P), dtype=torch.float32, device=dev)
    tk = torch.empty((n_ch, F, P), dtype=torch.int32, device=dev)
    tapidx = torch.full((P,), -1, dtype=torch.int32, device=dev)
    n_tap = 0
    if tap_pixels is not None:
        tp = torch.as_tensor(np.asarray(tap_pixels), dtype=torch.long, device=dev)
        n_tap = int(tp.numel())
        tapidx[tp] = torch.arange(n_tap, dtype=torch.int32, device=dev)
    K = len(bank)
    taps = torch.zeros((K, F, max(n_tap, 1), n_c), dtype=torch.float32, device=dev)
    si = torch.as_tensor(sigma_inv, dtype=torch.float32, device=dev).contiguous()
    sk = torch.as_tensor(np.asarray(s_inv, dtype=np.float32), device=dev)
    ring = ring.contiguous()
    st = cp.cuda.ExternalStream((stream or torch.cuda.current_stream()).cuda_stream)
    with st:
        for i, sub in enumerate(subsets):
            kern, meta = get_kernel(bank, sub, chunk=chunk, **(gen or {}))
            if n_ch > meta["maxch"]:
                raise ValueError(f"{n_ch} chunks > {meta['maxch']}")
            kern(((P + meta["pt"] - 1) // meta["pt"], F), (meta["pt"], meta["ng"]),
                 (_ptr(ring), _ptr(si), _ptr(sk), _ptr(keys[i]), _ptr(tapidx), _ptr(taps),
                  np.int32(R), np.int32(F), np.int32(P), np.int64(ring_row0),
                  np.int64(c_first), np.int32(n_c), np.int32(max(n_tap, 1)),
                  np.float32(clip)))
        n = n_ch * F * P
        _combine()((min(4096, (n + 255) // 256),), (256,),
                   (_ptr(keys), np.int32(len(subsets)), np.int64(n), _ptr(best), _ptr(tk)))
    return best, tk, (taps if n_tap else None)


def cpu_source_check(bank) -> Optional[str]:
    """Source generation sanity for CPU tests (None if fine)."""
    for sub in plan_kernels(bank):
        _src, meta = cuda_source(bank, sub)
        if meta["smem"] > 48 * 1024:
            return f"static shared memory {meta['smem']} B > 48 KiB for {sub}"
    return None
