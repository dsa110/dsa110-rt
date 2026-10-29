"""2x corr stage 1: CSR-driven CUDA kernel (cupy RawModule) + numpy twin.

What it computes, for one chgroup and ONE tier (all of its coarse DMs in
one launch):

    out[d, j, v] = sum_{src in CSR(v), fixed order} sum_{i < D}
                       x_src[q_base + j*D + i + shift[d, k_src]]

where ``x_src`` is the GEMM ring row ``row_src`` at summed channel
``k_src`` (conjugated if flagged), ``shift`` the stage-1 table of the
tier (q1 units; the corr ADVANCES channel k), ``D`` the tier decimation
(2 for tier A, 4 for tier B) and ``q_base`` the absolute q1 sample of the
block being emitted (ring positions are taken modulo ``T_ring``, a power
of two).

Ring layout (shared with :mod:`dsart.x2.corr.gemm2x_cuda`): k-major,
time-minor half2 ``ring[k][row][t mod T_ring]``.

Kernel layout (G0, v5): ONE thread block per value ``v`` covering all
``n_out`` outputs of all ``n_d`` DMs; thread ``tid -> (d = tid % n_d,
tl = tid / n_d)`` owns outputs ``t = tl + o*TT`` (``o < OPT``) in fp32
registers. Sources are staged chunk by chunk into shared memory (one
source per warp, lanes over its window of ``W = OPT*TT*D + spread`` q1
samples from ``q_base + smin[k]``; the conj flag is folded in by flipping
the im sign bit, exact), with each source's per-DM window offset cached in
shared memory; every input sample is read from global memory once per
block and reused by all DMs. Accumulation order is fixed: sources in CSR
order, then ``i = 0..D-1`` (``__fadd_rn``) -- the twin reproduces it in
float32 and is bit-exact. Measured on an RTX 2080 Ti (gridded n_sub 2,
tier A 32 DMs): 16.7 ms per block, vs 40-46 ms for the v3 tile kernel.

The CSR comes from WS-P's value map (:func:`stage1_csr`): rows are the
GEMM ring rows (kept core baselines, or redundancy classes when the GEMM
ran its class-reduce epilogue), values are in wire order (heavy first),
and the per-value source order is the map's (ascending ``bl*nchan+ch``),
or ascending channel for class rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger("dsart.x2.corr.stage1_cuda")

__all__ = [
    "CUDA_SOURCE",
    "OPT_CHOICES",
    "Stage1Csr",
    "Stage1Geometry",
    "compile_check_nvcc",
    "geometry_for",
    "stage1_csr",
    "stage1_tier_cuda",
    "stage1_twin",
]

#: outputs per thread the kernel is instantiated for (``stage1_tier_o<OPT>``)
OPT_CHOICES = (1, 2, 4, 8, 16, 32)

CUDA_SOURCE = r"""
#include <cuda_fp16.h>

/* ring   : half2 bits [C][R][T] (k-major, time-minor, absolute q1 mod T; T = T_mask + 1)
   out    : float2 [n_d][n_out][V]
   blockDim.x = n_d * TT; thread tid -> (d = tid % n_d, tl = tid / n_d), outputs t = tl + o*TT */
template <int OPT>
__device__ __forceinline__ void stage1_body(
    const unsigned int* __restrict__ ring,
    const int* __restrict__ ptr, const int* __restrict__ src_row, const int* __restrict__ src_k,
    const unsigned char* __restrict__ src_conj,
    const int* __restrict__ shifts, const int* __restrict__ smin,
    float2* __restrict__ out,
    int C, int R, int T_mask, int n_d, int n_out, int V, long long q_base,
    int D, int TT, int W, int chunk)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned int* win = reinterpret_cast<unsigned int*>(smem);     /* [chunk][W] half2 bits */
    int* soff = reinterpret_cast<int*>(win + chunk * W);           /* [chunk][n_d] */
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarp = blockDim.x >> 5;
    const int d = tid % n_d, tl = tid / n_d;
    const int v = blockIdx.x;
    const bool active = tl < TT;
    float acc_re[OPT], acc_im[OPT];
#pragma unroll
    for (int o = 0; o < OPT; ++o) { acc_re[o] = 0.f; acc_im[o] = 0.f; }
    const int s0 = ptr[v], s1 = ptr[v + 1];
    for (int c0 = s0; c0 < s1; c0 += chunk) {
        const int nc = min(chunk, s1 - c0);
        __syncthreads();
        for (int si = warp; si < nc; si += nwarp) {
            const int s = c0 + si;
            const int k = src_k[s];
            const unsigned int* base = ring + ((long long)k * R + src_row[s]) * (long long)(T_mask + 1);
            const int sm = smin[k];
            const int start = (int)((q_base + sm) & (long long)T_mask);
            const unsigned int flip = src_conj[s] ? 0x80000000u : 0u;
#pragma unroll 4
            for (int w = lane; w < W; w += 32) win[si * W + w] = base[(start + w) & T_mask] ^ flip;
            for (int dd = lane; dd < n_d; dd += 32) soff[si * n_d + dd] = shifts[dd * C + k] - sm;
        }
        __syncthreads();
        if (active) {
            for (int si = 0; si < nc; ++si) {
                const unsigned int* wp = win + si * W + soff[si * n_d + d] + tl * D;
#pragma unroll
                for (int o = 0; o < OPT; ++o) {
                    const unsigned int* q = wp + o * TT * D;
#pragma unroll 4
                    for (int i = 0; i < D; ++i) {
                        const unsigned int u = q[i];
                        const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&u));
                        acc_re[o] = __fadd_rn(acc_re[o], x.x);
                        acc_im[o] = __fadd_rn(acc_im[o], x.y);
                    }
                }
            }
        }
    }
    if (active) {
#pragma unroll
        for (int o = 0; o < OPT; ++o) {
            const int t = tl + o * TT;
            if (t < n_out) out[((long long)d * n_out + t) * V + v] = make_float2(acc_re[o], acc_im[o]);
        }
    }
}

#define STAGE1_INST(OPT)                                                                       \
extern "C" __global__ void __launch_bounds__(1024) stage1_tier_o##OPT(                         \
    const unsigned int* __restrict__ ring, const int* __restrict__ ptr,                        \
    const int* __restrict__ src_row, const int* __restrict__ src_k,                            \
    const unsigned char* __restrict__ src_conj, const int* __restrict__ shifts,                \
    const int* __restrict__ smin, float2* __restrict__ out, int C, int R, int T_mask, int n_d, \
    int n_out, int V, long long q_base, int D, int TT, int W, int chunk)                       \
{                                                                                              \
    stage1_body<OPT>(ring, ptr, src_row, src_k, src_conj, shifts, smin, out, C, R, T_mask,    \
                     n_d, n_out, V, q_base, D, TT, W, chunk);                                  \
}
STAGE1_INST(1)
STAGE1_INST(2)
STAGE1_INST(4)
STAGE1_INST(8)
STAGE1_INST(16)
STAGE1_INST(32)
"""


@dataclass(frozen=True)
class Stage1Csr:
    """Concatenated stage-1 CSR of one chgroup's streams."""

    ptr: np.ndarray            # (V+1,) int32
    row: np.ndarray            # (S,) int32 ring row
    k: np.ndarray              # (S,) int32 summed channel
    conj: np.ndarray           # (S,) uint8
    value_offsets: np.ndarray  # (n_sub+1,) int32 stream boundaries on the value axis
    n_contrib: np.ndarray      # (V,) int32 static count n (for /sqrt(n))

    @property
    def n_values(self) -> int:
        return int(self.ptr.shape[0] - 1)


def stage1_csr(vm, chgroup: int, *, ring_rows: str = "baselines") -> Stage1Csr:
    """From WS-P's :class:`dsart.grid.baseline_grid.ValueMap`.

    ``ring_rows="baselines"``: rows are kept core baselines (index into
    ``vm.kept_bl``), sources and order exactly ``vm.corr_csr_chgroup``.
    ``ring_rows="classes"`` (baselines kind, after the GEMM class reduce):
    one source per (class, channel of the sub-band), ascending channel,
    never conjugated (the reduce already applied ``bl_conj``)."""
    csr, voff = vm.corr_csr_chgroup(int(chgroup))
    n = np.concatenate([vm.streams[chgroup * vm.n_sub + s].n_contrib for s in range(vm.n_sub)])
    if ring_rows == "baselines":
        kept = np.asarray(vm.kept_bl, np.int64)
        row = np.searchsorted(kept, csr.bl.astype(np.int64))
        if not np.array_equal(kept[row], csr.bl):
            raise ValueError("CSR baseline outside the kept core set")
        return Stage1Csr(ptr=csr.ptr.astype(np.int32), row=row.astype(np.int32),
                         k=csr.ch.astype(np.int32), conj=csr.conj.astype(np.uint8),
                         value_offsets=voff.astype(np.int32), n_contrib=n.astype(np.int32))
    if ring_rows != "classes" or vm.kind != "baselines":
        raise ValueError("ring_rows='classes' needs a baselines value map")
    cls_of_value = np.concatenate([vm.streams[chgroup * vm.n_sub + s].class_id for s in range(vm.n_sub)])
    ptr = [0]
    rows, ks = [], []
    for v in range(cls_of_value.size):
        ch = np.unique(csr.ch[csr.ptr[v]:csr.ptr[v + 1]])
        rows.append(np.full(ch.size, cls_of_value[v], np.int64))
        ks.append(ch)
        ptr.append(ptr[-1] + ch.size)
    row = np.concatenate(rows)
    return Stage1Csr(ptr=np.asarray(ptr, np.int32), row=row.astype(np.int32),
                     k=np.concatenate(ks).astype(np.int32), conj=np.zeros(row.size, np.uint8),
                     value_offsets=voff.astype(np.int32), n_contrib=n.astype(np.int32))


@dataclass(frozen=True)
class Stage1Geometry:
    """Launch geometry of one tier (one block per value)."""

    D: int
    TT: int                    # threads per DM
    OPT: int                   # outputs per thread (t = tl + o*TT)
    n_out: int
    W: int                     # staged window per source, q1 samples
    chunk: int                 # sources per staging round
    smin: np.ndarray           # (C,) int32
    spread: int
    threads: int

    @property
    def smem_bytes(self) -> int:
        return self.chunk * (4 * self.W + 4 * (self.threads // self.TT))


def geometry_for(shifts: np.ndarray, *, D: int, n_out: int, smem_bytes: int = 64 * 1024,
                 max_threads: int = 1024) -> Stage1Geometry:
    """``shifts`` (n_d, C) int; ``n_out`` decimated outputs per block. ``OPT`` = the smallest
    instantiated outputs-per-thread with ``n_d * ceil(n_out / OPT) <= max_threads``; the
    source chunk fills ``smem_bytes`` (64 KB = the sm_75 opt-in maximum)."""
    n_d = int(shifts.shape[0])
    if n_d < 1 or n_d > max_threads:
        raise ValueError(f"n_d={n_d} out of range")
    opt = next((o for o in OPT_CHOICES if n_d * -(-n_out // o) <= max_threads), None)
    if opt is None:
        raise ValueError(f"n_out={n_out} with n_d={n_d} needs more than {OPT_CHOICES[-1]} outputs/thread")
    TT = -(-n_out // opt)
    smin = shifts.min(axis=0).astype(np.int32)
    spread = int((shifts.max(axis=0) - smin).max()) if shifts.size else 0
    W = opt * TT * int(D) + spread
    chunk = max(1, smem_bytes // (4 * W + 4 * n_d))
    return Stage1Geometry(D=int(D), TT=int(TT), OPT=int(opt), n_out=int(n_out), W=int(W),
                          chunk=int(chunk), smin=smin, spread=spread, threads=n_d * TT)


def stage1_twin(
    ring: np.ndarray,
    csr: Stage1Csr,
    shifts: np.ndarray,
    geo: Stage1Geometry,
    *,
    q_base: int,
    n_out: int,
) -> np.ndarray:
    """Numpy twin of ``stage1_tier``: same per-value blocks, staging windows (with the ring
    wrap), chunking and float32 add order. ``ring`` is k-major: complex ``(C, R, T_ring)``
    holding fp16-representable values, or float16 ``(C, R, T_ring, 2)``. Returns ``(n_d, n_out,
    V)`` complex64, bit-identical to the kernel."""
    if ring.dtype == np.float16:
        rre = ring[..., 0].astype(np.float32)
        rim = ring[..., 1].astype(np.float32)
    else:
        rre = ring.real.astype(np.float16).astype(np.float32)
        rim = ring.imag.astype(np.float16).astype(np.float32)
    C, R, T_ring = rre.shape
    if T_ring & (T_ring - 1):
        raise ValueError(f"T_ring={T_ring} must be a power of two")
    if n_out > geo.OPT * geo.TT:
        raise ValueError("geometry covers fewer outputs than n_out")
    n_d = shifts.shape[0]
    D, W, chunk = geo.D, geo.W, geo.chunk
    smin = geo.smin.astype(np.int64)
    V = csr.n_values
    out_re = np.zeros((n_d, n_out, V), np.float32)
    out_im = np.zeros((n_d, n_out, V), np.float32)
    t = np.arange(n_out)
    for v in range(V):
        s0, s1 = int(csr.ptr[v]), int(csr.ptr[v + 1])
        acc_re = np.zeros((n_d, n_out), np.float32)
        acc_im = np.zeros((n_d, n_out), np.float32)
        for c0 in range(s0, s1, chunk):
            nc = min(chunk, s1 - c0)
            srcs = np.arange(c0, c0 + nc)
            kk = csr.k[srcs].astype(np.int64)
            rr = csr.row[srcs].astype(np.int64)
            # win[si, w] = ring[k, row, (q_base + smin[k] + w) & (T_ring - 1)], conj folded in
            tr = (q_base + smin[kk][:, None] + np.arange(W)[None, :]) & (T_ring - 1)
            win_re = rre[kk[:, None], rr[:, None], tr]
            win_im = rim[kk[:, None], rr[:, None], tr]
            cj = csr.conj[srcs].astype(bool)
            win_im[cj] = -win_im[cj]
            for si in range(nc):
                k = kk[si]
                off = t[None, :] * D + shifts[:, k][:, None] - smin[k]            # (n_d, n_out)
                for i in range(D):
                    acc_re = acc_re + win_re[si][off + i]
                    acc_im = acc_im + win_im[si][off + i]
        out_re[:, :, v] = acc_re
        out_im[:, :, v] = acc_im
    return out_re + 1j * out_im


# ---------------------------------------------------------------------------
# GPU launcher
# ---------------------------------------------------------------------------

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        import cupy as cp  # noqa: PLC0415
        LOG.info("compiling stage1_tier via NVRTC")
        _MOD = cp.RawModule(code=CUDA_SOURCE, backend="nvrtc", options=("--std=c++14",))
    return _MOD


def _cp(t):
    import cupy as cp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))


def stage1_tier_cuda(ring, csr_dev: dict, shifts_dev, smin_dev, out, geo: Stage1Geometry, *,
                     q_base: int, n_out: int) -> None:
    """Launch on the current torch stream. ``ring`` fp16 torch k-major ``(C, R, T_ring, 2)``
    (``T_ring`` a power of two); ``csr_dev`` torch tensors ``ptr,row,k,conj``; ``shifts_dev``
    int32 ``(n_d, C)``; ``smin_dev`` int32 ``(C,)``; ``out`` float32 ``(n_d, n_out, V, 2)``."""
    import cupy as cp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    C, R, T_ring, two = ring.shape
    assert two == 2 and ring.dtype == torch.float16 and ring.is_contiguous()
    if T_ring & (T_ring - 1):
        raise ValueError(f"T_ring={T_ring} must be a power of two")
    n_d = int(shifts_dev.shape[0])
    V = int(csr_dev["ptr"].shape[0] - 1)
    assert tuple(out.shape) == (n_d, n_out, V, 2) and out.dtype == torch.float32
    if n_out > geo.OPT * geo.TT:
        raise ValueError("geometry covers fewer outputs than n_out")
    kern = _module().get_function(f"stage1_tier_o{geo.OPT}")
    if geo.smem_bytes > 48 * 1024:
        kern.max_dynamic_shared_size_bytes = geo.smem_bytes
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        kern(
            (V,), (geo.threads,),
            (_cp(ring.view(torch.int32)), _cp(csr_dev["ptr"]), _cp(csr_dev["row"]), _cp(csr_dev["k"]),
             _cp(csr_dev["conj"]), _cp(shifts_dev), _cp(smin_dev), _cp(out),
             np.int32(C), np.int32(R), np.int32(T_ring - 1), np.int32(n_d), np.int32(n_out), np.int32(V),
             np.int64(q_base), np.int32(geo.D), np.int32(geo.TT), np.int32(geo.W),
             np.int32(geo.chunk)),
            shared_mem=geo.smem_bytes,
        )


def compile_check_nvcc(source: str = CUDA_SOURCE, *, arch: str = "sm_75",
                       nvcc: str = "/usr/local/cuda/bin/nvcc") -> str:
    """O4 compile-only check without a GPU: ``nvcc -ptx`` the kernel source.
    Returns the PTX text; raises with nvcc's log on failure."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "k.cu")
        ptx = os.path.join(td, "k.ptx")
        with open(src, "w") as fh:
            fh.write(source)
        r = subprocess.run([nvcc, "-ptx", f"-arch={arch}", "--std=c++14", "-o", ptx, src],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stderr}")
        with open(ptx) as fh:
            return fh.read()
