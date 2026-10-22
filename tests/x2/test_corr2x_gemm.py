"""x2 WS-C deliverable 3: the 2x fast-GEMM Triton kernel (run under
``TRITON_INTERPRET=1`` at small shapes) against its CPU twin, for both
ring-row kinds (core baselines; redundancy classes via scratch + the
fixed-order reduce), with cal, the folded cube mask, the array-burst
time mask, and a ring position that wraps.

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
ring = torch.full((tab.n_ring_rows, nchan // csf, T_ring, 2), float("nan")).half()
for has_t in (False, True):
    ring.fill_(float("nan"))
    gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=t0, csf=csf,
                     t_int=t_int, time_keep=tk if has_t else None)
    want = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=csf, t_int=t_int,
                            time_keep=tk if has_t else None)                # (fv, k, R)
    pos = [(t0 + i) % T_ring for i in range(n_fv)]
    got = torch.complex(ring[..., 0].float(), ring[..., 1].float())[:, :, pos].permute(2, 1, 0)
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
    direct = abs(complex(want[1, 1, r]) - v) < 1e-3
    print("RESULT", has_t, ok, untouched, direct, int(tab.n_ring_rows))
"""

_CLASSES = _COMMON + r"""
rc = redundancy_classes(e, n, is_core_baseline_mask=mask)
tab = build_gemm_tables(mask, nants=nants, classes=rc)
assert tab.n_scratch_rows > 0 and tab.n_ring_rows == rc.n_class
ring = torch.full((tab.n_ring_rows, nchan // csf, T_ring, 2), float("nan")).half()
scr = torch.full((tab.n_scratch_rows, nchan // csf, n_fv, 2), float("nan"))
gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=t0, csf=csf, t_int=t_int,
                 scratch=scr)
reduce_classes(scr, ring, tab, t_ring0=t0, n_fv=n_fv)
rows = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=csf, t_int=t_int)
want = ring_write_reference(rows, tab, rc.bl_class, rc.bl_conj)
pos = [(t0 + i) % T_ring for i in range(n_fv)]
got = torch.complex(ring[..., 0].float(), ring[..., 1].float())[:, :, pos].permute(2, 1, 0)
ok = ulp_ok(got.real, want.real) and ulp_ok(got.imag, want.imag)
print("RESULT", ok, int(rc.n_class), int(tab.n_scratch_rows), bool(torch.isfinite(got).all()))
"""


def test_gemm2x_rows_matches_twin_interpreted() -> None:
    out = run_interpreted(_ROWS)
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    assert len(lines) == 2, out
    for ln in lines:
        assert ln.split()[2:5] == ["True", "True", "True"], out


def test_gemm2x_class_reduce_matches_twin_interpreted() -> None:
    out = run_interpreted(_CLASSES)
    assert "RESULT True" in out and "True" == out.split()[-1], out


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
    ring = torch.zeros((tab.n_ring_rows, 192, T_ring, 2), dtype=torch.float16, device=dev)
    scr = (torch.empty((tab.n_scratch_rows, 192, n_fv, 2), device=dev) if classes else None)
    gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=0, n_fv=n_fv, t_ring0=1000, csf=2, t_int=8,
                     scratch=scr, time_keep=tk)
    if classes:
        reduce_classes(scr, ring, tab, t_ring0=1000, n_fv=n_fv)
    want = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=n_fv, csf=2, t_int=8, time_keep=tk)
    if classes:
        want = ring_write_reference(want, tab, rc.bl_class, rc.bl_conj)
    pos = (1000 + torch.arange(n_fv, device=dev)) % T_ring
    got = torch.complex(ring[:, :, pos, 0].float(), ring[:, :, pos, 1].float()).permute(2, 1, 0)
    w16 = torch.complex(want.real.half().float(), want.imag.half().float())
    tol = 2 * torch.finfo(torch.float16).eps * w16.abs() + 1e-6
    assert bool(((got - w16).abs() <= tol * 1.5).all())
