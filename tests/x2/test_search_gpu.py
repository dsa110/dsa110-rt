"""CP1 GPU tests of the owner engine (``x2_gpu``; never run in CPU CI).

Run in the CP1 search track from the node test copy (the runner's
``search.kernels`` step drives ``bench/timeres/b8_owner2x.py``; these tests
are the per-kernel exactness checks behind it):

    pytest -m x2_gpu tests/x2/test_search_gpu.py

Each compares a GPU kernel with its CPU twin (bit-exact where the twin is
exact: K0, K1, K3b; fp32 tolerance for the FFT pair) or with production
(K1 1x CSR vs ``combine_sparse_v4``; K3b box vs ``multi_boxcar_argmax``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _gpu(fn):
    fn = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (CP1)")(fn)
    return pytest.mark.x2_gpu(fn)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "x2" / "dsa110_antpos.npz"


def _map(fid="2x-gridded-ns2-cint4"):
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.opoint import opoint_from_id
    from dsart.x2.search.valuemap import dm_inputs_from_value_map

    z = np.load(FIX)
    op = opoint_from_id(fid)
    vm = build_value_map_for_opoint(op, z["antpos_e"], z["antpos_n"],
                                    is_core_baseline_mask=core_baseline_mask_from_station_numbers(
                                        z["station"]), dec_deg=71.63)
    s_main = {8: 0.3534, 10: 0.1932, 12: 0.1055}[int(op.bits_main)]
    s_heavy = {2: 0.0773, 4: 0.0547}.get(int(op.n_sub), 0.1)
    return op, vm, dm_inputs_from_value_map(vm, s_main=s_main, s_heavy=s_heavy,
                                            bits_main=op.bits_main)


@_gpu
@pytest.mark.parametrize("tiled", [False, True])
def test_gpu_k0_unpack_equals_twin(tiled):
    from dsart.x2.search.codec import as_stream_formats, pack_frame2_rows
    from dsart.x2.search.combine import StreamRing, k0_rows_to_ring
    from dsart.x2.search.kernels_cuda import get_module
    from dsart.image.tiled_combine_cuda import _ptr

    _op, _vm, inp = _map()
    fm = as_stream_formats(inp.fmt, inp.csr.n_stream)
    rng = np.random.default_rng(0)
    S, n, V = len(fm), 40, int(inp.c_v.shape[1])
    raw = np.zeros((S, n, fm.row_bytes), np.uint8)
    for s in range(S):
        f = fm[s]
        lo = -(1 << (f.bits - 1))
        re = rng.integers(lo, -lo, size=(n, f.n_values))
        im = rng.integers(lo, -lo, size=(n, f.n_values))
        re[:, :f.n_heavy] = rng.integers(-128, 128, size=(n, f.n_heavy))
        raw[s, :, :f.row_bytes] = pack_frame2_rows(re, im, f)
    valid = np.ones((S, n), bool)
    valid[3, 5] = False
    ring = StreamRing(S, V, 64)
    k0_rows_to_ring(ring, raw, valid, row0=100, fmt=fm)
    dev = torch.device("cuda")
    codes = torch.zeros((S, V, 64, 2), dtype=torch.int8, device=dev)
    vout = torch.zeros((S, 64), dtype=torch.uint8, device=dev)
    nod = torch.zeros(S, dtype=torch.int64, device=dev)
    t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
    draw, dval = t(raw, torch.uint8), t(valid.astype(np.uint8), torch.uint8)
    nv = t([f.n_values for f in fm.per_stream], torch.int32)
    nh = t([f.n_heavy for f in fm.per_stream], torch.int32)
    bits = t([f.bits for f in fm.per_stream], torch.int32)
    if tiled:                                         # rows 100..139 wrap the 64-slot ring
        from dsart.x2.search.kernels_cuda import launch_k0_tile
        launch_k0_tile(draw, dval, nv, nh, bits, codes, vout, nod, S=S, n=n,
                       row_bytes=fm.row_bytes, V=V, R=64, row0=100,
                       stream=torch.cuda.current_stream())
    else:
        get_module().get_function("k0_unpack_frame2")(
            (n, S), (128,), (_ptr(draw), _ptr(dval), _ptr(nv), _ptr(nh), _ptr(bits), _ptr(codes),
                             _ptr(vout), _ptr(nod), np.int32(S), np.int32(n),
                             np.int32(fm.row_bytes), np.int32(V), np.int32(64), np.int64(100)))
    torch.cuda.synchronize()
    assert np.array_equal(codes.cpu().numpy(), ring.codes)
    assert np.array_equal(vout.cpu().numpy().astype(bool), ring.valid)


@_gpu
def test_gpu_k1_triton_equals_twin_on_the_2x_map():
    from dsart.x2.search.combine import k1_twin
    from dsart.x2.search.kernels_triton import csr_tensors, k1_combine

    _op, _vm, inp = _map()
    rng = np.random.default_rng(1)
    S, V = inp.c_v.shape
    codes = rng.integers(-7, 8, size=(S, V, 128, 2)).astype(np.int8)
    sh = rng.integers(0, 100, size=(3, S)).astype(np.int32)
    dev = torch.device("cuda")
    got = k1_combine(torch.from_numpy(codes).to(dev), torch.as_tensor(inp.c_v, device=dev),
                     csr_tensors(inp.csr, dev), torch.from_numpy(sh).to(dev), t_lo=200,
                     n_t=4, w_v=torch.as_tensor(inp.weights(), device=dev)).cpu().numpy()
    want = k1_twin(codes, c_v=inp.c_v, offset=0.0, csr=inp.csr, shifts=sh, t_lo=200, n_t=4,
                   w_v=inp.w_v)
    assert np.array_equal(got[..., 0], want.real) and np.array_equal(got[..., 1], want.imag)


@_gpu
@pytest.mark.parametrize("fid", ["2x-gridded-ns2-cint4", "2x-baselines-ns4-cint5"])
def test_gpu_k1_staged_equals_twin_on_the_2x_map(fid):
    """``--k1 staged`` (CUDA smem windows) == k1_twin bit for bit, with
    per-stream absolute shifts far apart and a ~50-row stage-2 sweep."""
    from dsart.x2.search.combine import k1_twin
    from dsart.x2.search.k1_staged import k1_staged, staged_geometry
    from dsart.x2.search.kernels_triton import csr_tensors, k1_combine

    _op, _vm, inp = _map(fid)
    rng = np.random.default_rng(2)
    S, V = inp.c_v.shape
    R = 512
    codes = rng.integers(-7, 8, size=(S, V, R, 2)).astype(np.int8)
    F = 18
    base = rng.integers(0, 300, size=S)
    sh = (base[None, :] + rng.integers(0, 50, size=(F, S))).astype(np.int32)
    dev = torch.device("cuda")
    ct = csr_tensors(inp.csr, dev)
    g = staged_geometry(sh, n_t=64)
    ct["smax"] = torch.as_tensor(g.smax, dtype=torch.int32, device=dev)
    args = (torch.from_numpy(codes).to(dev), torch.as_tensor(inp.c_v, device=dev), ct,
            torch.from_numpy(sh).to(dev))
    wv = torch.as_tensor(inp.weights(), device=dev)
    got = k1_staged(*args, t_lo=700, n_t=64, w_v=wv, geom=g).cpu().numpy()
    tri = k1_combine(*args, t_lo=700, n_t=64, w_v=wv).cpu().numpy()
    assert np.array_equal(got, tri)
    want = k1_twin(codes, c_v=inp.c_v, offset=0.0, csr=inp.csr, shifts=sh[:3], t_lo=700,
                   n_t=64, w_v=inp.w_v)
    assert np.array_equal(got[:3, ..., 0], want.real) and np.array_equal(got[:3, ..., 1],
                                                                        want.imag)


@_gpu
def test_gpu_k1_1x_bit_exact_vs_combine_sparse_v4(tmp_path):
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    p = subprocess.run([sys.executable, str(repo / "bench/timeres/b8_owner2x.py"), "kernels",
                        "--fft", "", "--k3b", "", "--k1-formats", "", "--check-1x-csr",
                        "--out", str(tmp_path)], capture_output=True, text=True, timeout=1800)
    import json
    res = json.loads((tmp_path / "b8_kernels.json").read_text())["results"]
    chk = [r for r in res if r["check"] == "k1_1x_csr_vs_v4"]
    assert chk and chk[0]["status"] == "ok", (chk, p.stderr[-2000:])


@_gpu
@pytest.mark.parametrize("mode", ["warp", "fused", "fused_v1"])
@pytest.mark.parametrize("fold", ["col", "row"])
def test_gpu_fft_pair_matches_twin_and_float64(fold, mode):
    from dsart.x2.search.csr import csr_from_value_cells
    from dsart.x2.search.fft import FftPlan, image_from_uv_twin

    _op, vm, _inp = _map()
    cells = [sv.cell_row.astype(np.int64) * 256 + sv.cell_col for sv in vm.streams]
    csr = csr_from_value_cells(cells, n_grid=256, fold=fold)
    plan = FftPlan.build(csr)
    rng = np.random.default_rng(2)
    B = 8
    uv = (rng.standard_normal((B, csr.n_active))
          + 1j * rng.standard_normal((B, csr.n_active))).astype(np.complex64)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "timeres"))
    from b8_owner2x import _warp_runner

    class _Inp:
        pass
    run = _warp_runner(torch.from_numpy(uv).cuda(), plan, _Inp(), torch.device("cuda"), mode=mode)
    got = run().float().cpu().numpy()                         # fp16 ring, 1/N^2 pixel scale
    twin = image_from_uv_twin(uv, plan, np.ones((256, 256)), out_dtype=None)
    sig = float(np.std(twin))
    assert np.max(np.abs(got - twin)) / sig <= 2e-3


@_gpu
@pytest.mark.parametrize("bank", ["box", "b3x"])
def test_gpu_k3b_triton_equals_twin(bank):
    from dsart.x2.search.filters import FilterBank, k3b_twin
    from dsart.x2.search.kernels_triton import k3b_detect

    fb = FilterBank.b3x() if bank == "b3x" else FilterBank.build()
    rng = np.random.default_rng(3)
    R, F, P = 128 + fb.history(), 2, 4096
    ring = rng.standard_normal((R, F, P)).astype(np.float16)
    ring[50, 1, 99] = 400.0
    si = np.array([1.1, 0.7], np.float32)
    sk = (1 / (fb.seeds * 1.05)).astype(np.float32)
    taps = np.array([5, 99, 4000])
    c_first = 3 * R + fb.max_d + fb.max_a
    b0, t0, p0 = k3b_twin(ring, ring_row0=0, c_first=c_first, n_c=128, sigma_inv=si, s_inv=sk,
                          bank=fb, tap_pixels=taps)
    b1, t1, p1 = k3b_detect(torch.from_numpy(ring).cuda(), ring_row0=0, c_first=c_first,
                            n_c=128, sigma_inv=torch.from_numpy(si).cuda(), s_inv=sk, bank=fb,
                            tap_pixels=taps)
    assert np.array_equal(b1.cpu().numpy(), b0) and np.array_equal(t1.cpu().numpy(), t0)
    assert np.array_equal(p1.cpu().numpy(), p0)


@_gpu
@pytest.mark.parametrize("bank", ["box", "b3x"])
def test_gpu_k3b_fused_equals_twin_and_v1(bank):
    """``--k3b fused`` (one generated kernel) == twin == v1, bit for bit."""
    from dsart.x2.search.filters import FilterBank, k3b_twin
    from dsart.x2.search.k3b_fused import k3b_fused
    from dsart.x2.search.kernels_triton import k3b_detect

    rng = np.random.default_rng(4)
    fb = FilterBank.build() if bank == "box" else FilterBank.b3x()
    F, P, n_c = 3, 4096, 384
    R = n_c + fb.history() + 5
    ring = rng.standard_normal((R, F, P)).astype(np.float16)
    ring[50, 1, 77] = 600.0
    c_first = 3 * R + 2 + fb.max_d + fb.max_a
    si = rng.uniform(0.8, 1.2, F).astype(np.float32)
    sk = (1 / fb.seeds).astype(np.float32)
    taps = np.array([0, 77, 4095])
    dev = torch.device("cuda")
    kw = dict(ring_row0=0, c_first=c_first, n_c=n_c, s_inv=sk, bank=fb, tap_pixels=taps)
    b0, t0, p0 = k3b_twin(ring, sigma_inv=si, **kw)
    rt = torch.from_numpy(ring).to(dev)
    st = torch.from_numpy(si).to(dev)
    for fn in (k3b_fused, k3b_detect):
        b1, t1, p1 = fn(rt, sigma_inv=st, **kw)
        assert np.array_equal(b1.cpu().numpy(), b0) and np.array_equal(t1.cpu().numpy(), t0)
        assert np.array_equal(p1.cpu().numpy(), p0)


@_gpu
@pytest.mark.parametrize("k1,k3b,fft", [("triton", "v1", "warp"), ("staged", "fused", "warp"),
                                        ("staged", "reg", "fused")])
def test_gpu_engine_equals_cpu_engine_on_a_synthetic_ring(k1, k3b, fft):
    """Same synthetic 2x ring through backend='cpu' and backend='gpu'."""
    from dsart.x2.search.codec import RowFormat
    from dsart.x2.search.engine import EngineConfig, OwnerEngine
    from dsart.x2.search.geometry import owner_plan_from_tiers
    from dsart.x2.search.ring import MemoryRing
    from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells

    N, V = 256, 60
    fa = np.array([50.0, 53.0, 56.0])
    plan = owner_plan_from_tiers([dict(name="A", t_native=16, coarse=np.array([53.0]), fine=fa,
                                       f2c=np.zeros(3, int))], half=0, n_sub=1, csf=2,
                                 n_halves=1)
    spec = plan.dms[0]
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells = random_value_cells(16, V, N, radius=40, seed=1, n_unused=2)
    ring = MemoryRing({0: fmt}, 16, t_buf=4096)
    SyntheticSky(plan=plan, cells={0: cells}, formats={0: fmt}, n_grid=N,
                 bursts=[Burst(d=0, f=1, t=384 + 100, l=100, m=150, amp=1.5)]).fill(
        ring, spec, 0, 3 * 384)
    inp = make_dm_inputs(cells, n_grid=N, fmt=fmt, step_main=0.3534, step_heavy=0.1)
    out = {}
    for backend, fmode in (("cpu", "pruned"), ("gpu", fft)):
        eng = OwnerEngine(plan, {0: inp}, ring, EngineConfig(n_grid=N, backend=backend,
                                                             fft_mode=fmode, layer2_n_burnin=0,
                                                             k1_impl=k1, k3b_impl=k3b))
        out[backend] = eng.process_cycle(1, keep_maps=True).per_dm[0]
    a, b = out["cpu"], out["gpu"]
    assert np.allclose(a.sigma_applied, b.sigma_applied, rtol=1e-3)
    ca = {(c.t_tier, c.fdm_local, c.l, c.m) for c in a.candidates}
    cb = {(c.t_tier, c.fdm_local, c.l, c.m) for c in b.candidates}
    assert ca == cb and (484, 1, 100, 150) in ca


def test_cp1_gpu_list_is_marked():
    """CPU check: every GPU test here carries the x2_gpu mark (CPU CI
    deselects them; the CP1 runner selects them)."""
    import sys
    mod = sys.modules[__name__]
    names = [n for n in dir(mod) if n.startswith("test_gpu_")]
    assert len(names) == 10                                  # 20 cases with parametrize
    for n in names:
        marks = {m.name for m in getattr(getattr(mod, n), "pytestmark", [])}
        assert "x2_gpu" in marks, n


@_gpu
def test_gpu_layer1_kernel_matches_torch_reference():
    """kernels_cuda.l1_clipstd == noise_norm.layer1.sigma_clipped_std_batched:
    same lower median (exact), same clip, sums in another order (rtol 1e-5)."""
    from dsart.noise_norm.layer1 import sigma_clipped_std_batched
    from dsart.x2.search.kernels_cuda import launch_l1

    rng = np.random.default_rng(9)
    R, F, P, M = 40, 5, 3000, 10000
    img = (rng.standard_normal((R, F, P)) * rng.uniform(0.5, 2.0, (1, F, 1))).astype(np.float16)
    img[3, 1, 10:400] = 80.0                                  # outliers clipped away
    img[7, 2, :] = np.float16(np.inf)                         # non-finite dropped
    lt = rng.integers(0, 32, size=(F, M)).astype(np.int32)
    lp = rng.integers(0, P, size=(F, M)).astype(np.int32)
    lt[4, :] = 7
    lp[4, :] = rng.integers(0, P, M)                          # row 4: all samples non-finite
    img[(17 + 7) % R, 4, :] = np.float16(np.inf)
    t0 = 17
    smp = img[np.mod(t0 + lt, R), np.arange(F)[:, None], lp].astype(np.float32)
    want = sigma_clipped_std_batched(torch.from_numpy(smp)).numpy()
    dev = torch.device("cuda")
    out = torch.empty(F, dtype=torch.float32, device=dev)
    launch_l1(torch.from_numpy(img).to(dev), torch.from_numpy(lt).to(dev),
              torch.from_numpy(lp).to(dev), out, F=F, P=P, R_img=R, t0=t0, M=M,
              stream=torch.cuda.current_stream())
    got = out.cpu().numpy()
    assert want[4] == 0.0 and got[4] == 0.0
    assert np.allclose(got, want, rtol=1e-5, atol=0), (got, want)


@_gpu
def test_gpu_clipstd_rows_matches_torch_reference():
    """kernels_cuda.clipstd_rows (Layer-2, 1e5 samples per row) == the torch
    sigma_clipped_std_batched (rtol 1e-5)."""
    from dsart.noise_norm.layer1 import sigma_clipped_std_batched
    from dsart.x2.search.kernels_cuda import launch_clipstd_rows

    rng = np.random.default_rng(10)
    x = (rng.standard_normal((8, 100000)) * rng.uniform(0.5, 300.0, (8, 1))).astype(np.float32)
    x[2, :500] = 1e6
    x[5, 7] = np.nan
    want = sigma_clipped_std_batched(torch.from_numpy(x.copy())).numpy()
    xd = torch.from_numpy(x).cuda()
    out = torch.empty(8, dtype=torch.float32, device="cuda")
    launch_clipstd_rows(xd, out, stream=torch.cuda.current_stream())
    assert np.allclose(out.cpu().numpy(), want, rtol=1e-5, atol=0), (out.cpu().numpy(), want)
