"""2x corr stage 1 (x2 WS-C deliverable 3): CSR-driven CUDA kernel
(cupy RawModule, the ``tiled_combine_cuda`` v3 pattern) + numpy twin.

What it computes, for one chgroup and ONE tier (all of its coarse DMs in
one launch):

    out[d, j, v] = sum_{src in CSR(v), fixed order} sum_{i < D}
                       x_src[q_base + j*D + i + shift[d, k_src]]

where ``x_src`` is the GEMM ring row ``row_src`` at summed channel
``k_src`` (conjugated if flagged), ``shift`` the stage-1 table of the
tier (q1 units; the corr ADVANCES channel k), ``D`` the tier decimation
(2 for tier A, 4 for tier B) and ``q_base`` the absolute q1 sample of the
block being emitted (ring positions are taken modulo ``T_ring``).

Kernel layout (v3 pattern): one thread block per (value ``v``, tile of
``TT`` output samples); ``blockDim = n_d * TT`` threads, thread ``(d, tl)``
owns output ``(d, tile*TT + tl)`` in two fp32 registers. Sources are
staged chunk by chunk into shared memory, each as one window of ``W =
TT*D + spread`` q1 samples starting at ``q_tile0 + smin[k]`` (``smin`` /
``spread`` = min / max-min of the tier's shifts per channel), so every
input sample a tile needs is read from global memory once per tile and
reused by all ``n_d`` DMs. Accumulation order is fixed: sources in CSR
order, then ``i = 0..D-1`` -- the twin reproduces it in float32 and is
bit-exact.

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
    "Stage1Csr",
    "Stage1Geometry",
    "compile_check_nvcc",
    "geometry_for",
    "stage1_csr",
    "stage1_tier_cuda",
    "stage1_twin",
]

CUDA_SOURCE = r"""
#include <cuda_fp16.h>

/* One block = (value v, tile of TT decimated output samples), all n_d DMs.
   blockDim.x = n_d * TT; thread tid -> (d = tid / TT, tl = tid % TT).
   ring   : half2 [R][C][T_ring]   (re, im), time-minor, absolute q1 mod T_ring
   out    : float2 [n_d][n_out][V] */
extern "C" __global__ void stage1_tier(
    const __half2*       __restrict__ ring,
    const int*           __restrict__ ptr,      /* [V+1] */
    const int*           __restrict__ src_row,  /* [S]   */
    const int*           __restrict__ src_k,    /* [S]   */
    const unsigned char* __restrict__ src_conj, /* [S]   */
    const int*           __restrict__ shifts,   /* [n_d][C] q1 units */
    const int*           __restrict__ smin,     /* [C]   min over d */
    float2*              __restrict__ out,
    int C, int T_ring, int n_d, int n_out, int V,
    long long q_base, int D, int TT, int W, int chunk)
{
    extern __shared__ __half2 win[];            /* [chunk][W] */
    const int v    = blockIdx.x;
    const int tile = blockIdx.y;
    const int tid  = threadIdx.x;
    const int d    = tid / TT;
    const int tl   = tid - d * TT;
    const int t_out = tile * TT + tl;
    const bool active = (d < n_d) && (t_out < n_out);
    const long long q_tile0 = q_base + (long long)tile * TT * D;
    float acc_re = 0.f, acc_im = 0.f;
    const int s0 = ptr[v], s1 = ptr[v + 1];
    for (int c0 = s0; c0 < s1; c0 += chunk) {
        const int nc = min(chunk, s1 - c0);
        __syncthreads();
        for (int i = tid; i < nc * W; i += blockDim.x) {
            const int si = i / W;
            const int w  = i - si * W;
            const int s  = c0 + si;
            const int k  = src_k[s];
            long long q  = q_tile0 + smin[k] + w;
            long long tr = q % T_ring;
            if (tr < 0) tr += T_ring;
            win[i] = ring[((long long)src_row[s] * C + k) * T_ring + tr];
        }
        __syncthreads();
        if (active) {
            for (int si = 0; si < nc; ++si) {
                const int s   = c0 + si;
                const int k   = src_k[s];
                const int off = si * W + tl * D + shifts[d * C + k] - smin[k];
                const float sg = src_conj[s] ? -1.f : 1.f;
                for (int i = 0; i < D; ++i) {
                    const float2 x = __half22float2(win[off + i]);
                    acc_re = __fadd_rn(acc_re, x.x);
                    acc_im = __fadd_rn(acc_im, sg * x.y);
                }
            }
        }
    }
    if (active)
        out[((long long)d * n_out + t_out) * V + v] = make_float2(acc_re, acc_im);
}
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
    """Launch geometry of one tier."""

    D: int
    TT: int
    W: int
    chunk: int
    smin: np.ndarray           # (C,) int32
    spread: int
    threads: int

    @property
    def smem_bytes(self) -> int:
        return 4 * self.chunk * self.W


def geometry_for(shifts: np.ndarray, *, D: int, TT: int | None = None,
                 smem_bytes: int = 48 * 1024) -> Stage1Geometry:
    """``shifts`` (n_d, C) int; picks ``TT`` so ``n_d*TT <= 1024`` and sizes
    the source chunk to the shared-memory budget."""
    n_d = int(shifts.shape[0])
    if TT is None:
        TT = max(1, min(64, 1024 // max(1, n_d)))
    if n_d * TT > 1024:
        raise ValueError(f"n_d({n_d}) * TT({TT}) > 1024 threads")
    smin = shifts.min(axis=0).astype(np.int32)
    spread = int((shifts.max(axis=0) - smin).max()) if shifts.size else 0
    W = TT * D + spread
    chunk = max(1, smem_bytes // (4 * W))
    return Stage1Geometry(D=int(D), TT=int(TT), W=int(W), chunk=int(chunk), smin=smin,
                          spread=spread, threads=n_d * TT)


def stage1_twin(
    ring: np.ndarray,
    csr: Stage1Csr,
    shifts: np.ndarray,
    geo: Stage1Geometry,
    *,
    q_base: int,
    n_out: int,
) -> np.ndarray:
    """Numpy twin of ``stage1_tier``: same tiles, same staging windows (with
    the ring wrap), same chunking and the same float32 add order. ``ring``
    is complex-valued ``(R, C, T_ring)`` holding fp16-representable values
    (or a float16 ``(R, C, T_ring, 2)`` array). Returns ``(n_d, n_out, V)``
    complex64, bit-identical to the kernel."""
    if ring.dtype == np.float16:
        rre = ring[..., 0].astype(np.float32)
        rim = ring[..., 1].astype(np.float32)
    else:
        rre = ring.real.astype(np.float16).astype(np.float32)
        rim = ring.imag.astype(np.float16).astype(np.float32)
    R, C, T_ring = rre.shape
    n_d = shifts.shape[0]
    D, TT, W, chunk = geo.D, geo.TT, geo.W, geo.chunk
    smin = geo.smin.astype(np.int64)
    V = csr.n_values
    out_re = np.zeros((n_d, n_out, V), np.float32)
    out_im = np.zeros((n_d, n_out, V), np.float32)
    n_tiles = -(-n_out // TT)
    tl = np.arange(TT)
    for v in range(V):
        s0, s1 = int(csr.ptr[v]), int(csr.ptr[v + 1])
        for tile in range(n_tiles):
            q_tile0 = q_base + tile * TT * D
            acc_re = np.zeros((n_d, TT), np.float32)
            acc_im = np.zeros((n_d, TT), np.float32)
            for c0 in range(s0, s1, chunk):
                nc = min(chunk, s1 - c0)
                # stage: win[si, w] = ring[row, k, (q_tile0 + smin[k] + w) mod T_ring]
                srcs = np.arange(c0, c0 + nc)
                kk = csr.k[srcs].astype(np.int64)
                rr = csr.row[srcs].astype(np.int64)
                tr = (q_tile0 + smin[kk][:, None] + np.arange(W)[None, :]) % T_ring
                win_re = rre[rr[:, None], kk[:, None], tr]
                win_im = rim[rr[:, None], kk[:, None], tr]
                for si in range(nc):
                    k = kk[si]
                    sg = np.float32(-1.0) if csr.conj[c0 + si] else np.float32(1.0)
                    off = tl[None, :] * D + shifts[:, k][:, None] - smin[k]    # (n_d, TT)
                    for i in range(D):
                        acc_re = acc_re + win_re[si][off + i]
                        acc_im = acc_im + sg * win_im[si][off + i]
            t_out = tile * TT + tl
            ok = t_out < n_out
            out_re[:, t_out[ok], v] = acc_re[:, ok]
            out_im[:, t_out[ok], v] = acc_im[:, ok]
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
    """Launch on the current torch stream. ``ring`` fp16 torch
    ``(R, C, T_ring, 2)``; ``csr_dev`` torch tensors ``ptr,row,k,conj``;
    ``shifts_dev`` int32 ``(n_d, C)``; ``smin_dev`` int32 ``(C,)``; ``out``
    float32 ``(n_d, n_out, V, 2)``."""
    import cupy as cp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    R, C, T_ring, two = ring.shape
    assert two == 2 and ring.dtype == torch.float16 and ring.is_contiguous()
    n_d = int(shifts_dev.shape[0])
    V = int(csr_dev["ptr"].shape[0] - 1)
    assert tuple(out.shape) == (n_d, n_out, V, 2) and out.dtype == torch.float32
    kern = _module().get_function("stage1_tier")
    n_tiles = -(-n_out // geo.TT)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        kern(
            (V, n_tiles), (geo.threads,),
            (_cp(ring), _cp(csr_dev["ptr"]), _cp(csr_dev["row"]), _cp(csr_dev["k"]),
             _cp(csr_dev["conj"]), _cp(shifts_dev), _cp(smin_dev), _cp(out),
             np.int32(C), np.int32(T_ring), np.int32(n_d), np.int32(n_out), np.int32(V),
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
