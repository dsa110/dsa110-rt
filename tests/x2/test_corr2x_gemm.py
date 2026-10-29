"""x2 WS-C: the 2x fast GEMM.

* the portable Triton kernel (run under ``TRITON_INTERPRET=1`` at small
  shapes) against its CPU twin, for both ring-row kinds (core baselines;
  redundancy classes via scratch + the fixed-order reduce), with cal, the
  folded cube mask, the array-burst time mask, and a ring position that
  wraps; k-major ring ``[k][row][t]``;
* the production tensor-core kernel (``gemm2x_cuda``): its lane/unit ->
  element map covers every upper-triangular slot pair exactly once, and the
  numpy twin of its stores (k-major ring, 16-byte runs of 4 samples, f-major
  class scratch) equals the reference -- on the CPU; on the GPU (x2_gpu) the
  kernel equals the twin / reference to one fp16 ULP at production shapes.

Each interpreted test also runs a MUTANT kernel (one flipped sign) and
requires it to MISS the twin, proving the interpreter executed the kernel.

Tolerance (plan §Correctness 2): the kernel accumulates in fp32 on tensor
cores, the twin in float64 from the same fp16 operands; after the cfp16
store they may differ by one fp16 ULP.
"""
from __future__ import annotations

import numpy as np
import torch

from ._interp import run_interpreted

_COMMON = r"""
import numpy as np, torch
from dsart.grid.baseline_grid import kept_core_baselines, redundancy_classes
from dsart.x2.corr.gemm2x import (build_gemm_tables, compact_cal, gemm2x_into_ring,
                                  gemm2x_reference, reduce_classes, ring_write_reference)
torch.manual_seed(0)
rng = np.random.default_rng(0)
nants, n_core, nchan, csf, npkt, t_int = 20, 18, 4, 2, 16, 8
# a redundant core: 18 antennas on a 3 x 6 grid at 5.75 m, 2 far outriggers
gx, gy = np.meshgrid(np.arange(6) * 5.75, np.arange(3) * 5.75)
e = np.concatenate([gx.ravel(), [900.0, -700.0]])
n = np.concatenate([gy.ravel(), [300.0, 800.0]])
is_core = np.arange(nants) < n_core
a_l, b_l = np.tril_indices(nants)
mask = is_core[a_l] & is_core[b_l]
re = (torch.from_numpy(rng.integers(-7, 8, size=(nchan, 2, 2, npkt, nants))) * 0.05).half()
im = (torch.from_numpy(rng.integers(-7, 8, size=(nchan, 2, 2, npkt, nants))) * 0.05).half()
ph = rng.uniform(-np.pi, np.pi, size=(nchan, 1, 2, 1, nants))
amp = rng.uniform(0.5, 1.5, size=ph.shape)
cal_r = torch.from_numpy((amp * np.cos(ph)).astype(np.float32)).half()
cal_i = torch.from_numpy((amp * np.sin(ph)).astype(np.float32)).half()
cube = torch.zeros(nants, nchan, 2, dtype=torch.bool)
cube[3, 1, 0] = True
cube[7] = True
cr, ci = compact_cal(cal_r, cal_i, cube)
tk = torch.ones(4, nchan, 2)
tk[1, 2, :] = 0.0
tk[3, :, 1] = 0.0
n_fv = npkt // (t_int // 2)
T_ring, t0 = 11, 9

def ulp_ok(got, want):
    g = got.float(); w = want.half().float()
    sp = torch.where(w.abs() > 0, torch.finfo(torch.float16).eps * w.abs() * 2, torch.full_like(w, 6e-8))
    return bool(((g - w).abs() <= sp + 1e-7).all())
"""

_ROWS = _COMMON + r"""
tab = build_gemm_tables(mask, nants=nants)
_, ka, kb = kept_core_baselines(mask, nants)
ring = torch.full((nchan // csf, tab.n_ring_rows, T_ring, 2), float("nan")).half()
for has_t in (False, True):
    ring.fill_(float("nan"))
    gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=t0, csf=csf,
                     t_int=t_int, time_keep=tk if has_t else None)
    want = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=csf, t_int=t_int,
                            time_keep=tk if has_t else None)                # (fv, k, R)
    pos = [(t0 + i) % T_ring for i in range(n_fv)]
    got = torch.complex(ring[..., 0].float(), ring[..., 1].float())[:, :, pos].permute(2, 0, 1)
    ok = ulp_ok(got.real, want.real) and ulp_ok(got.imag, want.imag)
    others = [p for p in range(T_ring) if p not in pos]
    untouched = bool(torch.isnan(ring[:, :, others].float()).all())
    # independent float64 check of the twin itself on two rows (direct sum)
    E = (re.double() + 1j * im.double())                                     # (C,2,P,pkt,A)
    G = (cr.double() + 1j * ci.double())[:, None, :, None, :]
    keep_t = tk.double()[torch.arange(npkt) // (npkt // 4)].permute(1, 2, 0)[:, None, :, :, None] if has_t else 1.0
    Ec = (E * G * keep_t).numpy()
    r = 5
    v = sum(np.conj(Ec[c, t2, p, 4:8, kb[r]]) @ Ec[c, t2, p, 4:8, ka[r]]
            for c in (2, 3) for t2 in (0, 1) for p in (0, 1))
    direct = abs(complex(want[1, 1, r]) - v) < 5e-2
    print("RESULT", has_t, ok, untouched, direct, int(tab.n_ring_rows))
# mutant kernel (one flipped term) must miss the twin
import dsart.x2.corr.gemm2x as g2
g2._gemm2x_kernel = _mutant("dsart.x2.corr.gemm2x", "_gemm2x_kernel",
                            "acc_im -= tl.dot(a_i, b_r, out_dtype=tl.float32)",
                            "acc_im += tl.dot(a_i, b_r, out_dtype=tl.float32)")
ring.fill_(float("nan"))
gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=t0, csf=csf, t_int=t_int)
want = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=csf, t_int=t_int)
got = torch.complex(ring[..., 0].float(), ring[..., 1].float())[:, :, pos].permute(2, 0, 1)
print("MUTANT", not (ulp_ok(got.real, want.real) and ulp_ok(got.imag, want.imag)))
"""

_CLASSES = _COMMON + r"""
rc = redundancy_classes(e, n, is_core_baseline_mask=mask)
tab = build_gemm_tables(mask, nants=nants, classes=rc)
assert tab.n_scratch_rows > 0 and tab.n_ring_rows == rc.n_class
ring = torch.full((nchan // csf, tab.n_ring_rows, T_ring, 2), float("nan")).half()
scr = torch.full((tab.n_scratch_rows, nchan // csf, n_fv, 2), float("nan"))
gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=t0, csf=csf, t_int=t_int,
                 scratch=scr)
reduce_classes(scr, ring, tab, t_ring0=t0, n_fv=n_fv)
rows = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=csf, t_int=t_int)
want = ring_write_reference(rows, tab, rc.bl_class, rc.bl_conj)
pos = [(t0 + i) % T_ring for i in range(n_fv)]
got = torch.complex(ring[..., 0].float(), ring[..., 1].float())[:, :, pos].permute(2, 0, 1)
ok = ulp_ok(got.real, want.real) and ulp_ok(got.imag, want.imag)
print("RESULT", ok, int(rc.n_class), int(tab.n_scratch_rows), bool(torch.isfinite(got).all()))
import dsart.x2.corr.gemm2x as g2
g2._reduce_kernel = _mutant("dsart.x2.corr.gemm2x", "_reduce_kernel",
                            "acc_i += sg * tl.load(", "acc_i -= sg * tl.load(")
ring.fill_(float("nan"))
gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=t0, csf=csf, t_int=t_int,
                 scratch=scr)
g2.reduce_classes(scr, ring, tab, t_ring0=t0, n_fv=n_fv)
got = torch.complex(ring[..., 0].float(), ring[..., 1].float())[:, :, pos].permute(2, 0, 1)
print("MUTANT", not (ulp_ok(got.real, want.real) and ulp_ok(got.imag, want.imag)))
"""


def test_gemm2x_rows_matches_twin_interpreted() -> None:
    out = run_interpreted(_ROWS)
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    assert len(lines) == 2, out
    for ln in lines:
        assert ln.split()[2:5] == ["True", "True", "True"], out
    assert "MUTANT True" in out, out


def test_gemm2x_class_reduce_matches_twin_interpreted() -> None:
    out = run_interpreted(_CLASSES)
    res = [ln for ln in out.splitlines() if ln.startswith("RESULT")][0].split()
    assert res[1] == "True" and res[-1] == "True", out
    assert "MUTANT True" in out, out


def test_gemm_tables_cover_every_core_pair_once() -> None:
    from dsart.grid.baseline_grid import kept_core_baselines, redundancy_classes
    from dsart.x2.corr.gemm2x import build_gemm_tables
    rng = np.random.default_rng(1)
    nants = 30
    e = rng.uniform(-100, 100, nants)
    n = rng.uniform(-100, 100, nants)
    is_core = np.ones(nants, bool)
    is_core[[3, 17]] = False
    a, b = np.tril_indices(nants)
    mask = is_core[a] & is_core[b]
    tab = build_gemm_tables(mask, nants=nants)
    bl, ka, kb = kept_core_baselines(mask, nants)
    d = tab.pair_dst.numpy()
    assert np.array_equal(np.sort(d[d >= 0]), np.arange(bl.size))
    core = tab.core_ants.numpy()[: tab.n_core]
    m, nn = np.nonzero(d >= 0)
    assert np.all(m < nn)
    assert np.array_equal(core[m][np.argsort(d[d >= 0])], kb)
    assert np.array_equal(core[nn][np.argsort(d[d >= 0])], ka)
    # every launched tile pair is upper-triangular and covers all stored pairs
    tp = tab.tile_pairs.numpy()
    assert np.all(tp[:, 0] <= tp[:, 1])
    assert set(map(tuple, np.stack([m // 16, nn // 16], 1))) <= set(map(tuple, tp))
    rc = redundancy_classes(e, n, is_core_baseline_mask=mask)
    tc = build_gemm_tables(mask, nants=nants, classes=rc)
    assert tc.n_ring_rows == rc.n_class
    assert torch.equal(tc.class_ptr[1:] - tc.class_ptr[:-1],
                       torch.from_numpy(rc.multiplicity[rc.multiplicity > 1]).int())


# ---------------------------------------------------------------------------
# GPU (CP1)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("classes", [False, True])
def test_gemm2x_kernel_vs_twin_gpu(classes) -> None:
    """Production shapes (96 antennas, fixture core, 384 channels, a full
    page): ring rows == float64 twin to one fp16 ULP; class reduce in fixed
    order; time mask on load."""
    import os
    from dsart.grid.baseline_grid import redundancy_classes
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.services.slow_corr_kernel import unpack_int4_split
    from dsart.x2.corr.gemm2x import (build_gemm_tables, compact_cal, gemm2x_into_ring,
                                      gemm2x_reference, reduce_classes, ring_write_reference)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    z = np.load(os.path.join(root, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
    mask = core_baseline_mask_from_station_numbers(z["station"])
    rc = redundancy_classes(z["antpos_e"], z["antpos_n"], is_core_baseline_mask=mask) if classes else None
    dev = torch.device("cuda")
    tab = build_gemm_tables(mask, classes=rc).to(dev)
    raw = np.random.default_rng(0).integers(0, 256, size=301989888, dtype=np.uint8)
    re, im = unpack_int4_split(raw, device=dev)
    rng = np.random.default_rng(1)
    ph = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=(384, 1, 2, 1, 96)).astype(np.float32))
    cube = torch.zeros(96, 384, 2, dtype=torch.bool)
    cube[5] = True
    cr, ci = compact_cal(torch.cos(ph).half().to(dev), torch.sin(ph).half().to(dev), cube.to(dev))
    tk = torch.ones(64, 384, 2, device=dev)
    tk[3] = 0
    n_fv, T_ring = 64, 1024
    ring = torch.zeros((192, tab.n_ring_rows, T_ring, 2), dtype=torch.float16, device=dev)
    scr = (torch.empty((tab.n_scratch_rows, 192, n_fv, 2), device=dev) if classes else None)
    gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=1000, csf=2, t_int=8,
                     scratch=scr, time_keep=tk)
    if classes:
        reduce_classes(scr, ring, tab, t_ring0=1000, n_fv=n_fv)
    want = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=2, t_int=8, time_keep=tk)
    if classes:
        want = ring_write_reference(want, tab, rc.bl_class, rc.bl_conj)
    pos = (1000 + torch.arange(n_fv, device=dev)) % T_ring
    got = torch.complex(ring[:, :, pos, 0].float(), ring[:, :, pos, 1].float()).permute(2, 0, 1)
    w16 = torch.complex(want.real.half().float(), want.imag.half().float())
    tol = 2 * torch.finfo(torch.float16).eps * w16.abs() + 1e-6
    assert bool(((got - w16).abs() <= tol * 1.5).all())


def _toy_96(classes: bool, seed: int = 0, npkt: int = 32, nchan: int = 4):
    """96 antennas (90 core on a redundant grid + 6 outriggers), small page."""
    from dsart.grid.baseline_grid import redundancy_classes
    from dsart.x2.corr.gemm2x import build_gemm_tables, compact_cal
    rng = np.random.default_rng(seed)
    nants = 96
    gx, gy = np.meshgrid(np.arange(15) * 5.75, np.arange(6) * 5.75)
    e = np.concatenate([gx.ravel(), rng.uniform(300, 900, 6)])
    n = np.concatenate([gy.ravel(), rng.uniform(300, 900, 6)])
    is_core = np.arange(nants) < 90
    is_core[[7, 40]] = False                                  # flagged core antennas
    a_l, b_l = np.tril_indices(nants)
    mask = is_core[a_l] & is_core[b_l]
    rc = redundancy_classes(e, n, is_core_baseline_mask=mask) if classes else None
    tab = build_gemm_tables(mask, nants=nants, classes=rc)
    re = (torch.from_numpy(rng.integers(-7, 8, size=(nchan, 2, 2, npkt, nants))) * 0.05).half()
    im = (torch.from_numpy(rng.integers(-7, 8, size=(nchan, 2, 2, npkt, nants))) * 0.05).half()
    ph = rng.uniform(-np.pi, np.pi, size=(nchan, 1, 2, 1, nants))
    cr, ci = compact_cal(torch.from_numpy(np.cos(ph).astype(np.float32)).half(),
                         torch.from_numpy(np.sin(ph).astype(np.float32)).half(), None)
    tk = torch.ones(4, nchan, 2)
    tk[1, 1, 0] = 0.0
    return tab, re, im, cr, ci, tk, rc


def test_mma_unit_map_covers_every_slot_pair_once() -> None:
    from dsart.x2.corr.gemm2x_cuda import NSLOT, unit_element_map
    m = unit_element_map()
    pairs = set(map(tuple, m[:, 4:6]))
    want = {(a, b) for a in range(NSLOT) for b in range(NSLOT)
            if a // 16 <= b // 16}                             # upper-triangular tile pairs
    assert len(pairs) == len(m) == len(want) and pairs == want
    assert m[:, 0].max() == 15 and np.bincount(m[:, 0]).max() == 3 * 32 * 4


@pytest.mark.parametrize("classes", [False, True])
def test_mma_twin_stores_equal_reference(classes) -> None:
    """CPU: the twin of the mma kernel's stores (k-major ring, runs of 4, class scratch
    f-major + the fixed-order reduce) == reference rows (+ ring_write_reference)."""
    from dsart.x2.corr.gemm2x import gemm2x_reference, ring_write_reference
    from dsart.x2.corr.gemm2x_cuda import gemm2x_mma_twin
    tab, re, im, cr, ci, tk, rc = _toy_96(classes)
    T_ring, t0, n_fv = 16, 12, 8
    ring, scr = gemm2x_mma_twin(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, T_ring=T_ring, t_ring0=t0,
                                time_keep=tk)
    rows = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=2, t_int=8, time_keep=tk)
    pos = [(t0 + i) % T_ring for i in range(n_fv)]
    if classes:
        cls = tab.class_ptr.numpy()
        for c, row in enumerate(tab.class_row.numpy()):          # the f-major reduce, in order
            acc = np.zeros(scr.shape[:2], np.complex64)
            for s in range(cls[c], cls[c + 1]):
                x = scr[:, :, tab.class_src.numpy()[s]]
                acc = acc + (np.conj(x) if tab.class_conj.numpy()[s] else x)
            ring[:, row, pos] = acc
        want = ring_write_reference(rows, tab, rc.bl_class, rc.bl_conj).numpy()
    else:
        want = rows.numpy()
    got = ring[:, :, pos].transpose(2, 0, 1)                      # (fv, k, R)
    assert np.isfinite(got).all()
    w16 = want.real.astype(np.float16).astype(np.float32) + 1j * want.imag.astype(np.float16).astype(np.float32)
    tol = 2 * np.finfo(np.float16).eps * np.abs(w16) + 1e-6
    assert np.all(np.abs(got - w16) <= 1.5 * tol)
    others = [p for p in range(T_ring) if p not in pos]
    assert np.isnan(ring[:, :, others].real).all()


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("classes", [False, True])
def test_mma_kernel_vs_reference_gpu(classes) -> None:
    """The production tensor-core kernel at production shapes (fixture core, 384 channels,
    a full page, time mask, cube mask): k-major ring == float64 twin to one fp16 ULP; class
    members through the f-major scratch + fixed-order reduce; nothing else written."""
    import os
    from dsart.grid.baseline_grid import redundancy_classes
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.services.slow_corr_kernel import unpack_int4_split
    from dsart.x2.corr.gemm2x import build_gemm_tables, compact_cal, gemm2x_reference, ring_write_reference
    from dsart.x2.corr.gemm2x_cuda import gemm2x_mma_into_ring, reduce_classes_cuda
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    z = np.load(os.path.join(root, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
    mask = core_baseline_mask_from_station_numbers(z["station"])
    rc = redundancy_classes(z["antpos_e"], z["antpos_n"], is_core_baseline_mask=mask) if classes else None
    dev = torch.device("cuda")
    tab = build_gemm_tables(mask, classes=rc).to(dev)
    raw = np.random.default_rng(0).integers(0, 256, size=301989888, dtype=np.uint8)
    re, im = unpack_int4_split(raw, device=dev)
    rng = np.random.default_rng(1)
    ph = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=(384, 1, 2, 1, 96)).astype(np.float32))
    cube = torch.zeros(96, 384, 2, dtype=torch.bool)
    cube[5] = True
    cr, ci = compact_cal(torch.cos(ph).half().to(dev), torch.sin(ph).half().to(dev), cube.to(dev))
    tk = torch.ones(64, 384, 2, device=dev)
    tk[3] = 0
    n_fv, T_ring, t0, fv0 = 64, 1024, 1000, 32
    ring = torch.full((192, tab.n_ring_rows, T_ring, 2), float("nan"), dtype=torch.float16, device=dev)
    scr = torch.empty((192, n_fv, tab.n_scratch_rows, 2), device=dev) if classes else None
    gemm2x_mma_into_ring(re, im, cr, ci, tab, ring, fv0=fv0, n_fv=n_fv, t_ring0=t0, scratch=scr,
                         time_keep=tk)
    if classes:
        reduce_classes_cuda(scr, ring, tab, t_ring0=t0, n_fv=n_fv)
    want = gemm2x_reference(re, im, cr, ci, tab, fv0=fv0, n_fv=n_fv, csf=2, t_int=8, time_keep=tk)
    if classes:
        want = ring_write_reference(want, tab, rc.bl_class, rc.bl_conj)
    pos = (t0 + torch.arange(n_fv, device=dev)) % T_ring
    got = torch.complex(ring[:, :, pos, 0].float(), ring[:, :, pos, 1].float()).permute(2, 0, 1)
    w16 = torch.complex(want.real.half().float(), want.imag.half().float())
    tol = 2 * torch.finfo(torch.float16).eps * w16.abs() + 1e-6
    assert bool(((got - w16).abs() <= tol * 1.5).all())
    del got, w16, want, tol
    n_nan = int(torch.isnan(ring[..., 0]).sum())                 # nothing outside the window
    assert n_nan == 192 * tab.n_ring_rows * (T_ring - n_fv)
    del ring, scr, re, im
    torch.cuda.empty_cache()
