"""2x fast GEMM (x2 WS-C deliverable 3): Triton kernel + CPU twin.

One launch correlates one block (or an fv-chunk of it) at the 2x op-point
(t_int 8 natives = 262.144 us, csf 2 -> 192 summed channels):

* **core cross-baselines only** -- the core antennas are gathered through
  an index table padded to whole 16-antenna tiles, and only upper-triangle
  tile pairs are launched (21 of 36 at 82 antennas); autos and outrigger
  baselines are never formed;
* **cal + RFI mask on operand load** -- the per-(channel, pol, antenna)
  complex gain, with the RFI cube mask folded in as zeros
  (:func:`compact_cal`), multiplies each voltage as it is loaded, in fp32,
  rounded once to fp16 for the tensor-core dot; the array-burst time mask
  (``HAS_TMASK``, rare) multiplies per 2.097 ms accumulation. No separate
  cal / zero-fill pass over the 1.2 GB of voltages;
* **K = csf x 2 pols x t_int = 32** read straight from today's unpack
  layout ``(NCHAN, 2t, 2p, NPKT, NANTS)`` -- no permute copy -- so Stokes
  I and the channel sum fall out of the one reduction;
* **cfp16 time-minor output** -- ``V(a,b) = conj(E_b) E_a`` stored as
  (re, im) fp16 at ``ring[row, k, t mod T_ring]`` with ``t`` the absolute
  q1 sample: stage 1 reads every source's time series contiguously and
  the 2-block window is an index wrap, never a copy;
* **unique-baseline reduce (optional)** -- ``pair_dst`` sends each pair to
  its ring row (every baseline for gridded cells; singleton classes) or to
  an fp32 scratch row (members of redundant classes); :func:`reduce_classes`
  then sums each class's members in a FIXED order (ascending row) with
  their conjugation flags and writes the class row. No atomics.

:func:`gemm2x_reference` is the CPU twin (same fp32 cal-on-load and fp16
operand rounding, float64 accumulation); ``tests/x2/test_corr2x_gemm.py``
runs the kernel itself under ``TRITON_INTERPRET=1`` against it.

NB: no ``from __future__ import annotations`` -- Triton must see real
``tl.constexpr`` annotation objects (the interpreter needs it).
"""
import numpy as np
import torch
import triton
import triton.language as tl

__all__ = [
    "Gemm2xTables",
    "build_gemm_tables",
    "compact_cal",
    "gemm2x_into_ring",
    "gemm2x_reference",
    "reduce_classes",
    "ring_write_reference",
]

BLOCK = 16


class Gemm2xTables:
    """Static index tables for one corr node (built once at start)."""

    def __init__(self, *, core_ants, n_core, pair_dst, tile_pairs, n_ring_rows,
                 n_scratch_rows, class_ptr, class_src, class_conj, class_row,
                 row_slot_m, row_slot_n):
        self.core_ants = core_ants        # int32 (NC_PAD,)
        self.n_core = int(n_core)
        self.pair_dst = pair_dst          # int32 (NC_PAD, NC_PAD): -1 drop; < n_ring_rows ring; else scratch
        self.tile_pairs = tile_pairs      # int32 (n_pairs, 2)
        self.n_ring_rows = int(n_ring_rows)
        self.n_scratch_rows = int(n_scratch_rows)
        self.class_ptr = class_ptr        # int32 (n_multi+1,)
        self.class_src = class_src        # int32 scratch rows (fixed order)
        self.class_conj = class_conj      # uint8
        self.class_row = class_row        # int32 ring row of each multi-member class
        self.row_slot_m = row_slot_m      # int64 (n_core_rows,) slot of b per core row
        self.row_slot_n = row_slot_n      # int64 slot of a

    @property
    def nc_pad(self) -> int:
        return int(self.core_ants.shape[0])

    def to(self, device):
        out = Gemm2xTables.__new__(Gemm2xTables)
        out.__dict__.update({k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in self.__dict__.items()})
        return out


def build_gemm_tables(vm, *, rows: str = "baselines") -> Gemm2xTables:
    """From a :class:`~dsart.x2.corr.value_map.ValueMap`.

    ``rows="baselines"``: ring row = core-baseline row (gridded cells, or
    per-baseline values without the reduce). ``rows="classes"`` (baselines
    kind): ring row = class id; singleton classes are stored directly,
    members of multi-member classes go to scratch rows that
    :func:`reduce_classes` sums in ascending-row order."""
    cr = vm.rows
    core = np.asarray(cr.core_ants, np.int64)
    nc = core.size
    nc_pad = -(-nc // BLOCK) * BLOCK
    core_ants = np.zeros(nc_pad, np.int32)
    core_ants[:nc] = core
    slot = {int(a): i for i, a in enumerate(core)}
    sm = np.array([slot[int(b)] for b in cr.b], np.int64)
    sn = np.array([slot[int(a)] for a in cr.a], np.int64)
    pair_dst = np.full((nc_pad, nc_pad), -1, np.int32)
    if rows == "baselines":
        pair_dst[sm, sn] = np.arange(cr.n)
        n_ring, n_scr = cr.n, 0
        cptr = np.zeros(1, np.int32)
        csrc = np.zeros(0, np.int32)
        cconj = np.zeros(0, np.uint8)
        crow = np.zeros(0, np.int32)
    elif rows == "classes":
        if vm.kind != "baselines":
            raise ValueError("rows='classes' needs a baselines value map")
        cls, conj = vm.class_of_row, vm.conj_of_row
        counts = np.bincount(cls, minlength=vm.n_class)
        multi = np.flatnonzero(counts > 1)
        n_ring, n_scr = vm.n_class, int(counts[multi].sum())
        cptr_l, csrc_l, cconj_l = [0], [], []
        scr_of_row = np.full(cr.n, -1, np.int64)
        nxt = 0
        for c in multi:
            mem = np.flatnonzero(cls == c)             # ascending row = the fixed order
            for r in mem:
                scr_of_row[r] = nxt
                csrc_l.append(nxt)
                cconj_l.append(int(conj[r]))
                nxt += 1
            cptr_l.append(cptr_l[-1] + mem.size)
        single = counts[cls] == 1
        pair_dst[sm[single], sn[single]] = cls[single]
        pair_dst[sm[~single], sn[~single]] = n_ring + scr_of_row[~single]
        cptr = np.asarray(cptr_l, np.int32)
        csrc = np.asarray(csrc_l, np.int32)
        cconj = np.asarray(cconj_l, np.uint8)
        crow = multi.astype(np.int32)
    else:
        raise ValueError(f"rows={rows!r}")
    nt = nc_pad // BLOCK
    tp = np.array([(i, j) for i in range(nt) for j in range(i, nt)], np.int32)
    return Gemm2xTables(
        core_ants=torch.from_numpy(core_ants), n_core=nc,
        pair_dst=torch.from_numpy(pair_dst), tile_pairs=torch.from_numpy(tp),
        n_ring_rows=n_ring, n_scratch_rows=n_scr,
        class_ptr=torch.from_numpy(cptr), class_src=torch.from_numpy(csrc),
        class_conj=torch.from_numpy(cconj), class_row=torch.from_numpy(crow),
        row_slot_m=torch.from_numpy(sm), row_slot_n=torch.from_numpy(sn),
    )


def compact_cal(cal_real, cal_imag, cube_mask=None):
    """Today's broadcast cal tensors ``(NCHAN, 1, NPOL, 1, NANTS)`` -> fp32
    ``(NCHAN, NPOL, NANTS)`` with the RFI cube mask ``(ANT, CH, POL)``
    folded in as zeros (the GEMM applies both on operand load)."""
    cr = cal_real[:, 0, :, 0, :].float()
    ci = cal_imag[:, 0, :, 0, :].float()
    if cube_mask is not None:
        keep = (~cube_mask).permute(1, 2, 0).to(device=cr.device, dtype=cr.dtype)
        cr = cr * keep
        ci = ci * keep
    return cr.contiguous(), ci.contiguous()


@triton.jit
def _gemm2x_kernel(
    re_ptr, im_ptr,                          # fp16 (NCHAN, 2, NVP, NPKT, NANTS)
    calr_ptr, cali_ptr,                      # fp32 (NCHAN, NVP, NANTS)
    tmask_ptr,                               # fp32 keep (n_acc, NCHAN, NVP) if HAS_TMASK
    core_ptr, dst_ptr, tp_ptr,               # int32 tables
    ring_ptr,                                # fp16 (n_ring_rows, NSUM, T_ring, 2)
    scr_ptr,                                 # fp32 (n_scratch_rows, NSUM, n_fv_scr, 2)
    fv0, n_core, n_ring_rows, t_ring0, T_ring, n_fv_scr, pk_per_acc, nants, nchan,
    s_ch, s_t, s_p, s_pk, s_ant,
    NSUM: tl.constexpr, NVP: tl.constexpr, CSF: tl.constexpr, PPFV: tl.constexpr,
    NC_PAD: tl.constexpr, HAS_TMASK: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_fv = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_p = tl.program_id(2)
    tm = tl.load(tp_ptr + pid_p * 2)
    tn = tl.load(tp_ptr + pid_p * 2 + 1)
    offs_m = tm * BLOCK + tl.arange(0, BLOCK)
    offs_n = tn * BLOCK + tl.arange(0, BLOCK)
    ok_m = offs_m < n_core
    ok_n = offs_n < n_core
    ant_m = tl.load(core_ptr + offs_m, mask=ok_m, other=0)
    ant_n = tl.load(core_ptr + offs_n, mask=ok_n, other=0)

    K: tl.constexpr = CSF * NVP * 2 * PPFV
    kk = tl.arange(0, K)
    ci = kk // (NVP * 2 * PPFV)
    p = (kk // (2 * PPFV)) % NVP
    t2 = (kk // PPFV) % 2
    pk = kk % PPFV
    ch = pid_k * CSF + ci
    pkt = (fv0 + pid_fv) * PPFV + pk
    koff = ch * s_ch + t2 * s_t + p * s_p + pkt * s_pk
    gk = (ch * NVP + p) * nants

    lm = ok_m[:, None] & (kk[None, :] >= 0)
    ln = (kk[:, None] >= 0) & ok_n[None, :]
    am = koff[None, :] + ant_m[:, None] * s_ant
    an = koff[:, None] + ant_n[None, :] * s_ant
    vr_m = tl.load(re_ptr + am, mask=lm, other=0.0).to(tl.float32)
    vi_m = tl.load(im_ptr + am, mask=lm, other=0.0).to(tl.float32)
    vr_n = tl.load(re_ptr + an, mask=ln, other=0.0).to(tl.float32)
    vi_n = tl.load(im_ptr + an, mask=ln, other=0.0).to(tl.float32)
    gr_m = tl.load(calr_ptr + gk[None, :] + ant_m[:, None], mask=lm, other=0.0)
    gi_m = tl.load(cali_ptr + gk[None, :] + ant_m[:, None], mask=lm, other=0.0)
    gr_n = tl.load(calr_ptr + gk[:, None] + ant_n[None, :], mask=ln, other=0.0)
    gi_n = tl.load(cali_ptr + gk[:, None] + ant_n[None, :], mask=ln, other=0.0)
    if HAS_TMASK:
        keep = tl.load(tmask_ptr + ((pkt // pk_per_acc) * nchan + ch) * NVP + p)
        gr_m = gr_m * keep[None, :]
        gi_m = gi_m * keep[None, :]
        gr_n = gr_n * keep[:, None]
        gi_n = gi_n * keep[:, None]
    a_r = (vr_m * gr_m - vi_m * gi_m).to(tl.float16)
    a_i = (vr_m * gi_m + vi_m * gr_m).to(tl.float16)
    b_r = (vr_n * gr_n - vi_n * gi_n).to(tl.float16)
    b_i = (vr_n * gi_n + vi_n * gr_n).to(tl.float16)

    acc_re = tl.dot(a_r, b_r, out_dtype=tl.float32)
    acc_re += tl.dot(a_i, b_i, out_dtype=tl.float32)
    acc_im = tl.dot(a_r, b_i, out_dtype=tl.float32)
    acc_im -= tl.dot(a_i, b_r, out_dtype=tl.float32)

    dst = tl.load(dst_ptr + offs_m[:, None] * NC_PAD + offs_n[None, :])
    to_ring = (dst >= 0) & (dst < n_ring_rows)
    to_scr = dst >= n_ring_rows
    t_ring = (t_ring0 + pid_fv) % T_ring
    roff = ((dst * NSUM + pid_k) * T_ring + t_ring) * 2
    tl.store(ring_ptr + roff, acc_re.to(tl.float16), mask=to_ring)
    tl.store(ring_ptr + roff + 1, acc_im.to(tl.float16), mask=to_ring)
    soff = (((dst - n_ring_rows) * NSUM + pid_k) * n_fv_scr + pid_fv) * 2
    tl.store(scr_ptr + soff, acc_re, mask=to_scr)
    tl.store(scr_ptr + soff + 1, acc_im, mask=to_scr)


def gemm2x_into_ring(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    cal_r: torch.Tensor,
    cal_i: torch.Tensor,
    tables: Gemm2xTables,
    ring: torch.Tensor,
    *,
    fv0: int,
    n_fv: int,
    t_ring0: int,
    csf: int,
    t_int: int,
    scratch: torch.Tensor | None = None,
    time_keep: torch.Tensor | None = None,
    num_warps: int = 4,
) -> None:
    """Correlate fast samples ``[fv0, fv0 + n_fv)`` of the block into the
    ring at ``(t_ring0 + i) % T_ring``. ``ring`` is fp16 ``(n_ring_rows,
    NSUM, T_ring, 2)``; ``scratch`` fp32 ``(n_scratch_rows, NSUM, n_fv, 2)``
    when the tables route pairs there; ``time_keep`` fp32 ``(n_acc, NCHAN,
    NPOL)`` 0/1 (array burst) or None."""
    nchan, two, nvp, npkt, nants = real_v.shape
    assert two == 2 and real_v.dtype == torch.float16 and real_v.is_contiguous()
    assert imag_v.shape == real_v.shape and imag_v.stride() == real_v.stride()
    ppfv = t_int // 2
    nsum = nchan // csf
    assert ring.dtype == torch.float16 and ring.shape[1] == nsum and ring.shape[3] == 2
    assert ring.is_contiguous() and ring.shape[0] == tables.n_ring_rows
    T_ring = int(ring.shape[2])
    if tables.n_scratch_rows:
        assert scratch is not None and scratch.dtype == torch.float32
        assert tuple(scratch.shape) == (tables.n_scratch_rows, nsum, n_fv, 2)
    else:
        scratch = ring.new_empty((1,), dtype=torch.float32)
    has_t = time_keep is not None
    tk = time_keep.contiguous() if has_t else cal_r.new_ones((1,))
    pk_per_acc = npkt // tk.shape[0] if has_t else 1
    s = real_v.stride()
    grid = (n_fv, nsum, int(tables.tile_pairs.shape[0]))
    _gemm2x_kernel[grid](
        real_v, imag_v, cal_r, cal_i, tk,
        tables.core_ants, tables.pair_dst, tables.tile_pairs,
        ring, scratch,
        fv0, tables.n_core, tables.n_ring_rows, t_ring0 % T_ring, T_ring, n_fv,
        pk_per_acc, nants, nchan,
        s[0], s[1], s[2], s[3], s[4],
        NSUM=nsum, NVP=nvp, CSF=csf, PPFV=ppfv, NC_PAD=tables.nc_pad,
        HAS_TMASK=has_t, BLOCK=BLOCK, num_warps=num_warps,
    )


@triton.jit
def _reduce_kernel(
    scr_ptr, ring_ptr, cptr_ptr, csrc_ptr, cconj_ptr, crow_ptr,
    t_ring0, T_ring, n_fv,
    NSUM: tl.constexpr, BLOCK_T: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.program_id(1)
    pt = tl.program_id(2)
    t = pt * BLOCK_T + tl.arange(0, BLOCK_T)
    mt = t < n_fv
    s0 = tl.load(cptr_ptr + c)
    s1 = tl.load(cptr_ptr + c + 1)
    acc_r = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for s in range(s0, s1):
        r = tl.load(csrc_ptr + s)
        sg = 1.0 - 2.0 * tl.load(cconj_ptr + s).to(tl.float32)
        off = ((r * NSUM + k) * n_fv + t) * 2
        acc_r += tl.load(scr_ptr + off, mask=mt, other=0.0)
        acc_i += sg * tl.load(scr_ptr + off + 1, mask=mt, other=0.0)
    row = tl.load(crow_ptr + c)
    roff = ((row * NSUM + k) * T_ring + (t_ring0 + t) % T_ring) * 2
    tl.store(ring_ptr + roff, acc_r.to(tl.float16), mask=mt)
    tl.store(ring_ptr + roff + 1, acc_i.to(tl.float16), mask=mt)


def reduce_classes(scratch, ring, tables: Gemm2xTables, *, t_ring0: int, n_fv: int,
                   block_t: int = 64) -> None:
    """Sum each redundant class's member rows (fixed order, conj applied)
    from ``scratch`` into its ring row at ``(t_ring0 + i) % T_ring``."""
    n_multi = int(tables.class_row.shape[0])
    if n_multi == 0:
        return
    nsum, T_ring = int(ring.shape[1]), int(ring.shape[2])
    grid = (n_multi, nsum, triton.cdiv(n_fv, block_t))
    _reduce_kernel[grid](
        scratch, ring, tables.class_ptr, tables.class_src, tables.class_conj, tables.class_row,
        t_ring0 % T_ring, T_ring, n_fv, NSUM=nsum, BLOCK_T=block_t,
    )


# ---------------------------------------------------------------------------
# CPU twin
# ---------------------------------------------------------------------------


def gemm2x_reference(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    cal_r: torch.Tensor,
    cal_i: torch.Tensor,
    tables: Gemm2xTables,
    *,
    fv0: int,
    n_fv: int,
    csf: int,
    t_int: int,
    time_keep: torch.Tensor | None = None,
    accum: torch.dtype = torch.float64,
) -> torch.Tensor:
    """``(n_fv, NSUM, n_core_rows)`` complex visibilities of the core rows
    (``tables.row_slot_m/n``), with the kernel's operand arithmetic: fp32
    cal-on-load, fp16 rounding of the calibrated operands, accumulation in
    ``accum`` (float64 by default: products of fp16 values are exact)."""
    nchan, _, nvp, npkt, nants = real_v.shape
    ppfv = t_int // 2
    nsum = nchan // csf
    p0, p1 = fv0 * ppfv, (fv0 + n_fv) * ppfv
    ants = tables.core_ants[: tables.n_core].long()
    vr = real_v[:, :, :, p0:p1][..., ants].float()          # (C, 2, P, pk, A)
    vi = imag_v[:, :, :, p0:p1][..., ants].float()
    gr = cal_r[:, :, ants].float()[:, None, :, None, :]       # (C, 1, P, 1, A)
    gi = cal_i[:, :, ants].float()[:, None, :, None, :]
    if time_keep is not None:
        acc_of_pkt = torch.arange(p0, p1, device=real_v.device) // (npkt // time_keep.shape[0])
        tk = time_keep.float()[acc_of_pkt]                    # (pk, C, P)
        tk = tk.permute(1, 2, 0)[:, None, :, :, None]         # (C, 1, P, pk, 1)
        gr = gr * tk
        gi = gi * tk
    xr = (vr * gr - vi * gi).half().to(accum)
    xi = (vr * gi + vi * gr).half().to(accum)
    # (C, 2, P, n_fv, ppfv, A) -> (n_fv, nsum, K, A) with K = (csf, P, 2t, ppfv)
    def arrange(x):
        x = x.reshape(nsum, csf, 2, nvp, n_fv, ppfv, x.shape[-1])
        return x.permute(4, 0, 1, 3, 2, 5, 6).reshape(n_fv, nsum, csf * nvp * 2 * ppfv, x.shape[-1])
    xr, xi = arrange(xr), arrange(xi)
    E = torch.complex(xr, xi)                                 # (n_fv, nsum, K, A)
    G = torch.einsum("fkti,fktj->fkij", E.conj(), E)          # [i, j] = sum conj(E_i) E_j
    sm = tables.row_slot_m.to(real_v.device)
    sn = tables.row_slot_n.to(real_v.device)
    return G[:, :, sm, sn]


def ring_write_reference(vis_rows: torch.Tensor, tables: Gemm2xTables, class_of_row=None,
                         conj_of_row=None) -> torch.Tensor:
    """What the kernel (+ reduce) leaves in the ring rows, as complex
    ``(n_fv, NSUM, n_ring_rows)``: core rows directly, or class sums in
    ascending-row order with conj (``class_of_row`` given)."""
    if class_of_row is None:
        return vis_rows
    cls = torch.as_tensor(class_of_row, device=vis_rows.device)
    cj = torch.as_tensor(conj_of_row, device=vis_rows.device)
    v = torch.where(cj[None, None, :], vis_rows.conj(), vis_rows)
    out = torch.zeros(vis_rows.shape[:2] + (tables.n_ring_rows,), dtype=vis_rows.dtype,
                      device=vis_rows.device)
    for r in range(v.shape[2]):                               # ascending row = fixed order
        out[:, :, int(cls[r])] += v[:, :, r]
    return out
