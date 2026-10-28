"""x2 WS-C deliverable 3: the CSR-driven stage-1 CUDA kernel.

* numpy twin (same tiles, staging windows with the ring wrap, chunking and
  float32 add order) == the float64 reference ``stage1_values`` +
  ``decimate`` to fp32 rounding, for both decimations and conjugated
  sources;
* twin output is invariant to the source chunk size (the add order is the
  CSR order, not the staging order) -- bit for bit;
* WS-P's value-map CSR drives it (rows = kept baselines or classes);
* the CUDA source compiles for sm_75 without a GPU (nvcc -ptx, O4);
* GPU (CP1): kernel == twin, torch.equal.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import torch

from dsart.x2.corr.stage1_cuda import (
    CUDA_SOURCE,
    Stage1Csr,
    compile_check_nvcc,
    geometry_for,
    stage1_twin,
)
from dsart.x2.ref import corr2x_ref as ref


def _problem(seed=0, R=9, C=8, n_sub=2, n_d=3, T_ring=40, conj_frac=0.3):
    rng = np.random.default_rng(seed)
    per = C // n_sub
    # fp16-representable ring values
    ring = (rng.normal(size=(R, C, T_ring)) + 1j * rng.normal(size=(R, C, T_ring)))
    ring = (ring.real.astype(np.float16).astype(np.float64)
            + 1j * ring.imag.astype(np.float16).astype(np.float64))
    streams, ptr, rows, ks, cj, voff = [], [0], [], [], [], [0]
    for s in range(n_sub):
        nv = 5 + s
        dest = rng.integers(-1, nv, size=(R, per))
        dest[0, :nv % per] = np.arange(nv % per)
        dest[:nv, 0] = np.arange(nv)                       # every value has a source
        conj = rng.random((R, per)) < conj_frac
        streams.append((dest, conj, nv))
        for v in range(nv):
            r, kl = np.nonzero(dest == v)
            order = np.argsort(r * C + s * per + kl, kind="stable")
            rows.append(r[order])
            ks.append(s * per + kl[order])
            cj.append(conj[r, kl][order])
            ptr.append(ptr[-1] + r.size)
        voff.append(voff[-1] + nv)
    csr = Stage1Csr(ptr=np.asarray(ptr, np.int32), row=np.concatenate(rows).astype(np.int32),
                    k=np.concatenate(ks).astype(np.int32), conj=np.concatenate(cj).astype(np.uint8),
                    value_offsets=np.asarray(voff, np.int32),
                    n_contrib=np.diff(np.asarray(ptr)).astype(np.int32))
    shifts = rng.integers(0, 7, size=(n_d, C)).astype(np.int32)
    shifts[:, ::per] = 0                                   # sub-band reference channels
    return ring, csr, shifts, streams, per


@pytest.mark.parametrize("D", [2, 4])
@pytest.mark.parametrize("q_base", [0, 27])                 # 27: the window wraps the ring
def test_twin_equals_reference(D, q_base) -> None:
    ring, csr, shifts, streams, per = _problem()
    R, C, T_ring = ring.shape
    n_out = 3
    geo = geometry_for(shifts, D=D, TT=2)
    got = stage1_twin(ring, csr, shifts, geo, q_base=q_base, n_out=n_out)
    # reference on the unwrapped series starting at q_base
    idx = (q_base + np.arange(n_out * D + int(shifts.max()) + 1)) % T_ring
    Vt = ring[:, :, idx].transpose(2, 1, 0)                  # (t, C, R)
    for d in range(shifts.shape[0]):
        for s, (dest, conj, nv) in enumerate(streams):
            ks = np.arange(s * per, (s + 1) * per)
            x = ref.stage1_values(Vt, dest, conj, ks=ks, s1=shifts[d, ks], n_values=nv,
                                  t0=0, n_out=n_out * D)
            want = ref.decimate(x, D)
            lo = csr.value_offsets[s]
            np.testing.assert_allclose(got[d, :, lo:lo + nv], want, rtol=1e-5, atol=1e-5)


def test_twin_chunk_invariant_bit_exact() -> None:
    ring, csr, shifts, _, _ = _problem(seed=3)
    g1 = geometry_for(shifts, D=2, TT=4)
    a = stage1_twin(ring, csr, shifts, g1, q_base=5, n_out=6)
    g2 = type(g1)(D=g1.D, TT=g1.TT, W=g1.W, chunk=1, smin=g1.smin, spread=g1.spread,
                  threads=g1.threads)
    b = stage1_twin(ring, csr, shifts, g2, q_base=5, n_out=6)
    assert np.array_equal(a, b)


def test_geometry_bounds() -> None:
    sh = np.random.default_rng(0).integers(0, 130, size=(32, 192)).astype(np.int32)
    g = geometry_for(sh, D=2)
    assert g.threads <= 1024 and g.smem_bytes <= 48 * 1024
    assert g.W == g.TT * 2 + g.spread and g.chunk >= 1
    assert np.all(sh - g.smin[None, :] <= g.spread)


def test_csr_from_value_map() -> None:
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.corr.stage1_cuda import stage1_csr
    from dsart.x2.opoint import opoint_2x
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    z = np.load(os.path.join(root, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
    mask = core_baseline_mask_from_station_numbers(z["station"])
    vm = build_value_map_for_opoint(opoint_2x("baselines", 2), z["antpos_e"], z["antpos_n"],
                                    is_core_baseline_mask=mask, dec_deg=71.63)
    g = 5
    a = stage1_csr(vm, g, ring_rows="baselines")
    b = stage1_csr(vm, g, ring_rows="classes")
    assert a.n_values == b.n_values == int(vm.n_values[2 * g:2 * g + 2].sum())
    assert np.array_equal(np.diff(a.ptr), a.n_contrib)
    assert a.row.max() < vm.kept_bl.size and b.row.max() < vm.classes.n_class
    # class rows: one source per (class, channel); n stays the member x channel count
    per = vm.nchan_eff // 2
    assert np.all(np.diff(b.ptr) == per)
    assert np.array_equal(a.n_contrib, b.n_contrib)
    assert not b.conj.any() and a.conj.any()


@pytest.mark.skipif(not os.path.exists("/usr/local/cuda/bin/nvcc") and not shutil.which("nvcc"),
                    reason="no nvcc on this host")
def test_cuda_source_compiles_sm75() -> None:
    nvcc = "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else shutil.which("nvcc")
    ptx = compile_check_nvcc(CUDA_SOURCE, nvcc=nvcc)
    assert ".entry stage1_tier" in ptx


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("D", [2, 4])
def test_kernel_equals_twin_gpu(D) -> None:
    from dsart.x2.corr.stage1_cuda import stage1_tier_cuda
    ring, csr, shifts, _, _ = _problem(seed=7, R=40, C=16, n_d=12, T_ring=96)
    geo = geometry_for(shifts, D=D)
    n_out = 17
    want = stage1_twin(ring, csr, shifts, geo, q_base=70, n_out=n_out)
    dev = torch.device("cuda")
    r16 = torch.from_numpy(np.stack([ring.real, ring.imag], -1).astype(np.float16)).to(dev)
    cd = {k: torch.from_numpy(getattr(csr, k)).to(dev) for k in ("ptr", "row", "k", "conj")}
    out = torch.empty((shifts.shape[0], n_out, csr.n_values, 2), dtype=torch.float32, device=dev)
    stage1_tier_cuda(r16, cd, torch.from_numpy(shifts).to(dev),
                     torch.from_numpy(geo.smin).to(dev), out, geo, q_base=70, n_out=n_out)
    got = out.cpu().numpy()
    assert np.array_equal(got[..., 0], want.real.astype(np.float32))
    assert np.array_equal(got[..., 1], want.imag.astype(np.float32))
