#!/usr/bin/env python
"""B8 -- the 2x owner engine at CP1 (WS-S). Contract: docs/x2/CP1_RUNBOOK.md,
tools/x2/cp_runner_plans.py (the command lines are the interface).

Subcommands (exit 0 = every gate passed, 1 = a gate failed, 2 = error):

  compile-check                 h23 CPU (CUDA_VISIBLE_DEVICES=""): every module parses;
                                CUDA K0/K2/K3a compile with NVRTC (compile only); Triton
                                K1/K3b compile for sm_75 (no launch); CPU twins and the
                                TRITON_INTERPRET runs agree with their references.
  kernels --fft cufft,warp --k3b box,b3x[,v1,fused] --k1 triton,staged --k1-formats all
          --check-1x-csr
                                GPU: us/image per kernel (K0, H2D, K1 per variant, FFT per
                                variant, Layer-1, K3b per bank x implementation, top-k) per
                                format; every (K1, FFT, K3b) chain total vs the 4.8 us/image
                                gate and the worst half's duty vs the 0.6 aim / 0.75 gate
                                (the best combination per format gates); K1 variants and K3b
                                implementations bit-exact vs their twins; K1 1x CSR vs
                                combine_sparse_v4; FFT |d| <= 2e-3 sigma vs float64; K3b box
                                vs production fp16 boxcars |dS/N| <= 0.02 (+fp16 ulp).
  compat-1x                     GPU: the engine at today's parameters vs production taps
                                (K1 == v4 bit for bit, image vs fp16 cuFFT+prescale,
                                detector vs multi_boxcar_argmax).
  engine --gpus 0,1 --input synthetic-ring|rx-v4 --opoint <id> --minutes M
         [--shm-name /ring --plan p.npz --halves 2j,2j+1 --out DIR]   (WS-T fleet-load form)
         [--variants staged:reg[,triton:v1] --ring-access pread|mmap]
                                one owner half per GPU, real-time paced synthetic ring
                                (in-process; no shm) or ring v4 (RingV4Source); each k1:k3b
                                variant for M / n_variants minutes: per-stage us/image (CUDA
                                events), chain total vs 4.8, GPU duty <= 0.75 (aim 0.6),
                                bounded lag, no lag-recovery seek, host <= 30 ms/cycle;
                                exit 0 if some variant passes on every half. rx-v4 waits for
                                the ring (attach) and the first data, stops when data stop.
  teardown                      remove this bench's /dev/shm/dsart-x2-* and temp files.

All JSON goes under --out (a directory). Runs only from the node test copy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

EXIT_OK, EXIT_GATE, EXIT_ERR = 0, 1, 2
CYCLE_S = 6144 * 32.768e-6
SHM_PREFIX = "/dev/shm/dsart-x2-"
FIXTURE = REPO / "tests" / "fixtures" / "x2" / "dsa110_antpos.npz"
DEC_DEG = 71.63


def _write(out: Path, name: str, obj: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)
    return p


def _status(results: list) -> int:
    return EXIT_GATE if any(r.get("status") == "fail" for r in results) else EXIT_OK


# ---------------------------------------------------------------------------
# compile-check (h23 CPU)
# ---------------------------------------------------------------------------


def _py_sources() -> list:
    fs = sorted((REPO / "src/dsart/x2/search").glob("*.py")) + \
        sorted((REPO / "src/dsart/x2/ref").glob("*.py")) + [Path(__file__)]
    return fs


def check_python() -> list:
    res = []
    for f in _py_sources():
        try:
            compile(f.read_text(), str(f), "exec")
            res.append(dict(check="py_compile", file=str(f.relative_to(REPO)), status="ok"))
        except SyntaxError as e:
            res.append(dict(check="py_compile", file=str(f.relative_to(REPO)), status="fail",
                            error=repr(e)))
    return res


def check_nvrtc() -> dict:
    """Every owner-engine CUDA source compiled for compute_75 with libnvrtc
    (no device needed: ``dsart.x2.search.nvrtc``)."""
    from dsart.x2.search import nvrtc
    if not nvrtc.available():
        return dict(check="cuda_nvrtc", status="skip", reason="libnvrtc not found")
    from dsart.x2.search.filters import FilterBank
    from dsart.x2.search.k1_staged import k1_source
    from dsart.x2.search.k3b_cuda import cuda_source
    from dsart.x2.search.kernels_cuda import CUDA_SOURCE
    srcs = [(CUDA_SOURCE, ("k0_unpack_frame2", "k0_unpack_cint8", "k2_colfft_256",
                           "k3a_rowc2r_256")),
            (k1_source(24), ("k1_staged",))]
    from dsart.x2.search.k3b_reg import plan_parts, reg_source
    for bank_name, fb in (("b3x", FilterBank.b3x()), ("box", FilterBank.build())):
        src, meta = cuda_source(fb, name=f"k3b_cuda_{bank_name}")
        srcs.append((src, (meta["name"],)))
        for i, part in enumerate(plan_parts(fb)):
            src, meta = reg_source(fb, part, name=f"k3b_reg_{bank_name}_{i}")
            srcs.append((src, (meta["name"],)))
    try:
        names = []
        for src, want in srcs:
            txt = nvrtc.ptx(src)
            names += [n for n in want if f".entry {n}" in txt]
        return dict(check="cuda_nvrtc", status="ok", arch="compute_75", kernels=names)
    except Exception as e:                                       # noqa: BLE001
        return dict(check="cuda_nvrtc", status="fail", error=repr(e)[:2000])


def check_triton_compile() -> list:
    if os.environ.get("TRITON_INTERPRET") == "1":
        return [dict(check="triton_compile", status="skip", reason="TRITON_INTERPRET set")]
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except Exception as e:                                       # noqa: BLE001
        return [dict(check="triton_compile", status="skip", reason=f"triton unavailable: {e!r}")]
    from dsart.x2.search import kernels_triton as kt

    def comp(fn, types: dict, consts: dict, label: str) -> dict:
        names = fn.arg_names
        sig = {i: types[n] for i, n in enumerate(names) if n not in consts}
        cst = {i: consts[n] for i, n in enumerate(names) if n in consts}
        try:
            triton.compile(ASTSource(fn=fn, signature=sig, constants=cst),
                           target=GPUTarget("cuda", 75, 32))
            return dict(check="triton_compile", kernel=label, status="ok")
        except Exception as e:                                   # noqa: BLE001
            return dict(check="triton_compile", kernel=label, status="fail",
                        error=repr(e)[:2000])

    out = []
    k1 = dict(codes_ptr="*i8", cv_ptr="*fp32", wv_ptr="*fp32", ptr_ptr="*i32", es_ptr="*i32", ev_ptr="*i32",
              ec_ptr="*i32", hs_ptr="*fp32", sh_ptr="*i32", out_ptr="*fp32", n_act="i32",
              n_t="i32", t_lo="i32", R="i32", V="i32", S="i32", off="fp32")
    out.append(comp(kt._k1_kernel, k1, dict(MAX_NE=64, BLOCK_C=64, BLOCK_T=16), "K1"))
    k3 = dict(ring_ptr="*fp16", si_ptr="*fp32", best_ptr="*fp32", tk_ptr="*i32",
              tapidx_ptr="*i32", taps_ptr="*fp32", R="i32", F="i32", P="i32", row0="i32",
              c_first="i32", n_c="i32", s_inv_k="fp32", clip="fp32", k_id="i32", first="i32",
              D="i32", A="i32", L1="i32", L2="i32", L3="i32", n_tap="i32")
    for order, ln in ((0, 1), (1, 64), (2, 3), (3, 149)):
        out.append(comp(kt._k3b_kernel, k3, dict(LEN=ln, N_C=384, ORDER=order, CHUNK=64,
                                                  BLOCK=128), f"K3b order {order}"))
    # fused K3b (one generated kernel per bank): tier A n_c 384, tier B 192
    from dsart.x2.search.filters import FilterBank
    from dsart.x2.search.k3b_fused import aot_compile
    for bank_name, fb in (("b3x", FilterBank.b3x()), ("box", FilterBank.build())):
        for n_c in (384, 192):
            try:
                aot_compile(fb, arch=75, n_c=n_c)
                out.append(dict(check="triton_compile", kernel=f"K3b fused {bank_name} n_c {n_c}",
                                status="ok"))
            except Exception as e:                               # noqa: BLE001
                out.append(dict(check="triton_compile", kernel=f"K3b fused {bank_name} n_c {n_c}",
                                status="fail", error=repr(e)[:2000]))
    return out


def check_twins() -> list:
    """Small CPU twin checks (the full suite runs in pre.cpu_tests)."""
    res = []
    try:
        import numpy as np
        __import__("torch")
    except Exception as e:                                       # noqa: BLE001
        return [dict(check="twins", status="skip", reason=f"numpy/torch unavailable: {e!r}")]
    try:
        from dsart.x2.ref import search2x_ref as ref
        from dsart.x2.search.codec import RowFormat, pack_frame2_rows, unpack_frame2_rows
        from dsart.x2.search.csr import csr_from_value_cells
        from dsart.x2.search.fft import FftPlan, image_from_uv_twin
        from dsart.x2.search.filters import FilterBank, k3b_twin
        from dsart.x2.search.kernels_cuda import k0_twin_bitgather

        rng = np.random.default_rng(0)
        fmt = RowFormat("frame2", n_values=29, n_heavy=3, bits=4)
        re = rng.integers(-7, 8, size=(5, 29))
        im = rng.integers(-7, 8, size=(5, 29))
        raw = pack_frame2_rows(re, im, fmt)
        ok = all(np.array_equal(a, b) for a, b in zip(unpack_frame2_rows(raw, fmt), (re, im)))
        g = k0_twin_bitgather(raw, fmt)
        ok &= np.array_equal(g[0], re) and np.array_equal(g[1], im)
        res.append(dict(check="twin_k0", status="ok" if ok else "fail"))
        N = 64
        cells = [rng.integers(0, N * N, size=50) for _ in range(3)]
        csr = csr_from_value_cells(cells, n_grid=N)
        Gf = np.zeros((2, N, N), complex)
        for c in cells:
            Gf[:, c // N, c % N] += rng.standard_normal((2, c.size))
        u, w = csr.cell_u, csr.cell_w
        uv = (0.5 * (Gf[:, u, w] + np.conj(Gf[:, (-u) % N, (-w) % N]))).astype(np.complex64)
        img = image_from_uv_twin(uv, FftPlan.build(csr), np.ones((N, N)), out_dtype=None)
        want = np.fft.ifft2(Gf).real
        d = float(np.max(np.abs(img - want)) / np.abs(want).max())
        res.append(dict(check="twin_fft", status="ok" if d < 2e-6 else "fail", rel_err=d))
        fb = FilterBank.b3x()
        R, F, P = 64 + fb.history(), 1, 16
        ring = rng.standard_normal((R, F, P)).astype(np.float16)
        best, _tk, _taps = k3b_twin(ring, ring_row0=0, c_first=fb.max_d + fb.max_a, n_c=64,
                                    sigma_inv=np.ones(1, np.float32),
                                    s_inv=(1 / fb.seeds).astype(np.float32), bank=fb)
        xn = ring.astype(np.float64).reshape(R, F, 4, 4)
        sc = ref.ref_filter_scores(xn, 0, ref.ref_filter_bank((), "b3x"),
                                   np.arange(fb.max_d + fb.max_a, fb.max_d + fb.max_a + 64))
        mx = (sc / fb.seeds[:, None, None, None, None]).max(axis=(0, 1)).reshape(1, F, P)
        d = float(np.max(np.abs(best - mx)))
        res.append(dict(check="twin_k3b", status="ok" if d < 1e-3 else "fail", max_abs=d))
    except Exception as e:                                       # noqa: BLE001
        res.append(dict(check="twins", status="fail", error=traceback.format_exc()[-2000:],
                        exc=repr(e)))
    # Triton interpreter (subprocess: the env var must precede the import)
    try:
        __import__("triton")
        env = dict(os.environ, TRITON_INTERPRET="1")
        p = subprocess.run([sys.executable, __file__, "_selftest-interp"], env=env,
                           capture_output=True, text=True, timeout=600)
        res.append(dict(check="triton_interpret", status="ok" if p.returncode == 0 else "fail",
                        tail=(p.stdout + p.stderr)[-1500:]))
    except ImportError:
        res.append(dict(check="triton_interpret", status="skip", reason="triton unavailable"))
    return res


def selftest_interp() -> int:
    import numpy as np
    import torch

    from dsart.x2.search.triton_interp import install
    install()
    from dsart.x2.search.combine import k1_twin
    from dsart.x2.search.csr import csr_from_value_cells
    from dsart.x2.search.filters import FilterBank, k3b_twin
    from dsart.x2.search.kernels_triton import csr_tensors, k1_combine, k3b_detect

    rng = np.random.default_rng(1)
    cells = [rng.choice(256, size=10, replace=False).astype(np.int64) for _ in range(3)]
    csr = csr_from_value_cells(cells, n_grid=16)
    codes = rng.integers(-7, 8, size=(3, 10, 8, 2)).astype(np.int8)
    cv = rng.uniform(0.1, 2, size=(3, 10)).astype(np.float32)
    wv = np.sqrt(rng.integers(1, 900, size=(3, 10)).astype(np.float64)).astype(np.float32)
    sh = rng.integers(0, 3, size=(2, 3)).astype(np.int32)
    want = k1_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=sh, t_lo=5, n_t=3, w_v=wv)
    got = k1_combine(torch.from_numpy(codes), torch.from_numpy(cv), csr_tensors(csr, "cpu"),
                     torch.from_numpy(sh), t_lo=5, n_t=3, block_c=16, block_t=4,
                     w_v=torch.from_numpy(wv)).numpy()
    ok = np.array_equal(got[..., 0], want.real) and np.array_equal(got[..., 1], want.imag)
    fb = FilterBank.build((1, 2, 4))
    ring = rng.standard_normal((70, 1, 8)).astype(np.float16)
    b0, t0, _ = k3b_twin(ring, ring_row0=0, c_first=4, n_c=64, sigma_inv=np.ones(1, np.float32),
                         s_inv=np.ones(3, np.float32), bank=fb)
    b1, t1, _ = k3b_detect(torch.from_numpy(ring), ring_row0=0, c_first=4, n_c=64,
                           sigma_inv=torch.ones(1), s_inv=np.ones(3), bank=fb, block=8)
    ok &= np.array_equal(b1.numpy(), b0) and np.array_equal(t1.numpy(), t0)
    from dsart.x2.search.k3b_fused import k3b_fused
    b2, t2, _ = k3b_fused(torch.from_numpy(ring), ring_row0=0, c_first=4, n_c=64,
                          sigma_inv=torch.ones(1), s_inv=np.ones(3), bank=fb, block=8)
    ok &= np.array_equal(b2.numpy(), b0) and np.array_equal(t2.numpy(), t0)
    print("interp K1/K3b v1/K3b fused bit-exact:", ok)
    return 0 if ok else 1


def cmd_compile_check(a) -> int:
    res = check_python()
    res.append(check_nvrtc())
    res += check_triton_compile()
    res += check_twins()
    _write(Path(a.out), "b8_compile_check.json", dict(results=res))
    for r in res:
        if r.get("status") != "ok":
            print(json.dumps({k: v for k, v in r.items() if k != "tail"})[:400])
    return _status(res)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _need_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("this subcommand needs a GPU (CP1 only)")


def _value_map(opoint_id: str, antpos_npz: str = "", dec_deg: float = DEC_DEG):
    """WS-P value map of ``opoint_id`` (the same antpos + DEC as WS-T's TX,
    so the pattern ids match the ring's)."""
    import numpy as np
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.opoint import opoint_from_id

    op = opoint_from_id(opoint_id)
    z = np.load(antpos_npz or FIXTURE)
    vm = build_value_map_for_opoint(op, z["antpos_e"], z["antpos_n"],
                                    is_core_baseline_mask=core_baseline_mask_from_station_numbers(
                                        z["station"]), dec_deg=float(dec_deg))
    return op, vm


def _steps(op) -> tuple:
    """Phase 0 decode steps (sigma_hat units): s4 0.3534 / s5 0.1932; heavy
    0.0773 (n_sub 2) / 0.0547 (n_sub 4)."""
    s_main = {8: 0.3534, 10: 0.1932, 12: 0.1055}.get(int(op.bits_main), 1.0)
    s_heavy = {2: 0.0773, 4: 0.0547}.get(int(op.n_sub), 0.1)
    return s_main, s_heavy


def _cuda_ms(fn, reps: int = 5) -> float:
    import torch
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


FORMATS_2X = ("2x-gridded-ns2-cint4", "2x-gridded-ns4-cint5", "2x-gridded-ns4-cint4",
              "2x-baselines-ns4-cint5", "2x-baselines-ns2-cint4", "2x-baselines-ns4-cint4")
K1_VARIANTS = ("triton", "staged")
K3B_IMPLS = ("v1", "fused", "cuda", "reg")
CHAIN_GATE_US = 4.8          # us/image, all-in compute chain (ledger E1)
DUTY_AIM, DUTY_GATE = 0.6, 0.75
STAGES = ("k0", "k1", "fft", "l1", "k3b", "topk", "l2")   # compute stream (h2d overlaps on copy)


def _plan_for(op, plan_path: str = ""):
    from dsart.x2.plan import TieredDmPlan, build_tiered_plan_for_opoint
    if plan_path:
        return TieredDmPlan.from_npz(plan_path, verify=True)
    return build_tiered_plan_for_opoint(op)


def _half_loads(plan) -> dict:
    """Per owner half: images per cycle (sum n_fdm x hop) and its DMs."""
    from dsart.x2.search.geometry import owner_plan_from_tiered
    out = {}
    for h in range(int(plan.n_halves)):
        op_ = owner_plan_from_tiered(plan, half=h)
        out[h] = dict(images_per_cycle=int(sum(s.n_fdm * s.hop for s in op_.dms)),
                      n_dm=len(op_.dms), oplan=op_)
    return out


def _split_k3b(arg: str) -> tuple:
    """``--k3b`` tokens: banks (box, b3x) and implementations (v1, fused)."""
    toks = [t for t in arg.split(",") if t]
    banks = [t for t in toks if t in ("box", "b3x")] or ["b3x"]
    impls = [t for t in toks if t in K3B_IMPLS] or list(K3B_IMPLS)
    bad = [t for t in toks if t not in ("box", "b3x") + K3B_IMPLS]
    if bad:
        raise SystemExit(f"--k3b: unknown {bad} (banks box,b3x; implementations v1,fused)")
    return banks, impls


def _chain_records(fid: str, us: dict, img_per_s: float) -> list:
    """Every (K1, FFT, K3b) combination of one format: all-in us/image vs
    the 4.8 gate and the worst half's GPU duty vs the 0.6 aim / 0.75 gate."""
    out = []
    fixed = us.get("k0", 0.0) + us.get("l1", 0.0) + us.get("topk", 0.0)
    for k1v, k1us in us.get("k1", {}).items():
        for fv, fus in us.get("fft", {}).items():
            for kv, kus in us.get("k3b", {}).items():
                tot = fixed + k1us + fus + kus
                duty = tot * img_per_s * 1e-6
                out.append(dict(check="chain", format=fid, k1=k1v, fft=fv, k3b=kv,
                                us_per_image=dict(k0=us.get("k0"), k1=k1us, fft=fus,
                                                  l1=us.get("l1"), k3b=kus, topk=us.get("topk")),
                                h2d_us_per_image=us.get("h2d"), total_us=tot,
                                gate_us=CHAIN_GATE_US, images_per_s_worst_half=img_per_s,
                                gpu_duty=duty, duty_aim_met=duty <= DUTY_AIM,
                                status="ok" if (tot <= CHAIN_GATE_US and duty <= DUTY_GATE)
                                else "fail"))
    return out


def cmd_kernels(a) -> int:
    _need_gpu()
    import numpy as np
    import torch

    from dsart.x2.search.combine import k1_twin
    from dsart.x2.search.fft import FftPlan, image_from_uv_torch, image_from_uv_twin
    from dsart.x2.search.filters import FilterBank, k3b_twin
    from dsart.x2.search.k1_staged import k1_staged, staged_geometry
    from dsart.x2.search.k3b_cuda import k3b_cuda
    from dsart.x2.search.k3b_fused import k3b_fused
    from dsart.x2.search.k3b_reg import k3b_reg
    from dsart.x2.search.kernels_triton import csr_tensors, k1_combine, k3b_detect
    from dsart.x2.search.valuemap import dm_inputs_from_value_map

    out = Path(a.out)
    res = []
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    fmts = list(FORMATS_2X) if a.k1_formats == "all" else [f for f in a.k1_formats.split(",") if f]
    k1_vars = [v for v in a.k1.split(",") if v]
    banks, impls = _split_k3b(a.k3b)
    fft_vars = [v for v in a.fft.split(",") if v]
    hop, n_t = 384, 64
    # ---- format-independent: K3b (bank x implementation), Layer-1, top-k -------------
    common = {"k3b": {}}
    for bank_name in banks:
        fb = FilterBank.b3x() if bank_name == "b3x" else FilterBank.build()
        F, P = 18, 65536
        R = hop + fb.history()
        ring = torch.randn((R, F, P), device=dev).to(torch.float16)
        si = torch.ones(F, device=dev)
        small = ring[:, :2, :4096].contiguous()
        b0, t0, _ = k3b_twin(small.cpu().numpy(), ring_row0=0, c_first=fb.max_d + fb.max_a,
                             n_c=64, sigma_inv=np.ones(2, np.float32),
                             s_inv=(1 / fb.seeds).astype(np.float32), bank=fb)
        for impl in impls:
            fn = dict(v1=k3b_detect, fused=k3b_fused, cuda=k3b_cuda, reg=k3b_reg)[impl]
            try:
                ms = _cuda_ms(lambda: fn(ring, ring_row0=0, c_first=fb.max_d + fb.max_a,
                                         n_c=hop, sigma_inv=si, s_inv=1 / fb.seeds, bank=fb),
                              reps=2)
                us = 1e3 * ms / (F * hop)
                b1, t1, _ = fn(small, ring_row0=0, c_first=fb.max_d + fb.max_a, n_c=64,
                               sigma_inv=si[:2], s_inv=1 / fb.seeds, bank=fb)
                exact = bool(np.array_equal(b1.cpu().numpy(), b0) and
                             np.array_equal(t1.cpu().numpy(), t0))
                entry = dict(check="k3b", bank=bank_name, impl=impl, us_per_image=us,
                             bit_exact_vs_twin=exact, status="ok" if exact else "fail")
                if bank_name == "box" and impl == "v1":
                    entry.update(_k3b_compat_vs_production(fb, dev))
                    if entry.get("compat_status") == "fail":
                        entry["status"] = "fail"
                if bank_name == "b3x":
                    common["k3b"][impl] = us
                res.append(entry)
            except Exception as e:                               # noqa: BLE001
                res.append(dict(check="k3b", bank=bank_name, impl=impl, status="fail",
                                error=repr(e)[:1000]))
        ring = small = None
    try:
        F = 18
        from dsart.x2.search.kernels_cuda import launch_l1
        img = torch.randn((hop + 150, F, 65536), device=dev).to(torch.float16)
        lt = torch.randint(0, hop, (F, 10000), dtype=torch.int32, device=dev)
        lp = torch.randint(0, 65536, (F, 10000), dtype=torch.int32, device=dev)
        sig = torch.empty(F, device=dev)
        ms = _cuda_ms(lambda: launch_l1(img, lt, lp, sig, F=F, P=65536, R_img=hop + 150, t0=0,
                                        M=10000, stream=torch.cuda.current_stream()))
        common["l1"] = 1e3 * ms / (F * hop)
        del img
        best = torch.randn((hop // 64, F, 65536), device=dev)
        n_top = 224 * hop // 192
        ms = _cuda_ms(lambda: torch.topk(best.view(-1), n_top))
        common["topk"] = 1e3 * ms / (F * hop)
        res.append(dict(check="l1_topk", l1_us_per_image=common["l1"],
                        topk_us_per_image=common["topk"], status="ok"))
    except Exception as e:                                       # noqa: BLE001
        res.append(dict(check="l1_topk", status="fail", error=repr(e)[:1000]))
    # ---- per format: K0, H2D, K1 variants, FFT variants, chain ------------------------
    summary = {}
    for fid in fmts:
        us = dict(k1={}, fft={}, k3b=dict(common["k3b"]), l1=common.get("l1", 0.0),
                  topk=common.get("topk", 0.0))
        try:
            op, vm = _value_map(fid)
            sm, sh = _steps(op)
            inp = dm_inputs_from_value_map(vm, s_main=sm, s_heavy=sh, bits_main=op.bits_main)
            loads = _half_loads(_plan_for(op))
            wh = max(loads, key=lambda h: loads[h]["images_per_cycle"])
            img_per_s = loads[wh]["images_per_cycle"] / CYCLE_S
            spec = max(loads[wh]["oplan"].dms, key=lambda s: s.n_fdm * s.hop)
            F, S = int(spec.n_fdm), int(spec.n_stream)
            fm = inp.fmt
            V = int(inp.c_v.shape[1])
            # K0 + H2D: one cycle of rows of this DM
            from dsart.x2.search.codec import as_stream_formats, pack_frame2_rows
            fms = as_stream_formats(fm, S)
            rb = fms.row_bytes
            raw = np.zeros((S, spec.hop, rb), np.uint8)
            for s_ in range(S):
                f_ = fms[s_]
                lo = -(1 << (f_.bits - 1)) + 1
                re = rng.integers(lo, -lo + 1, size=(spec.hop, f_.n_values))
                im = rng.integers(lo, -lo + 1, size=(spec.hop, f_.n_values))
                raw[s_, :, :f_.row_bytes] = pack_frame2_rows(re, im, f_)
            hraw = torch.from_numpy(raw).pin_memory()
            draw = hraw.to(dev)
            dval = torch.ones((S, spec.hop), dtype=torch.uint8, device=dev)
            R_s = spec.hop + 400
            codes0 = torch.zeros((S, V, R_s, 2), dtype=torch.int8, device=dev)
            vout = torch.zeros((S, R_s), dtype=torch.uint8, device=dev)
            nod = torch.zeros((S,), dtype=torch.int64, device=dev)
            nv = torch.as_tensor([f_.n_values for f_ in fms.per_stream], dtype=torch.int32,
                                 device=dev)
            nh = torch.as_tensor([f_.n_heavy for f_ in fms.per_stream], dtype=torch.int32,
                                 device=dev)
            bits = torch.as_tensor([f_.bits for f_ in fms.per_stream], dtype=torch.int32,
                                   device=dev)
            from dsart.x2.search.kernels_cuda import launch_k0_tile

            def run_k0():
                launch_k0_tile(draw, dval, nv, nh, bits, codes0, vout, nod, S=S, n=spec.hop,
                               row_bytes=rb, V=V, R=R_s, row0=0,
                               stream=torch.cuda.current_stream())
            us["k0"] = 1e3 * _cuda_ms(run_k0) / (F * spec.hop)
            us["h2d"] = 1e3 * _cuda_ms(lambda: hraw.to(dev, non_blocking=True)) / (F * spec.hop)
            codes0 = None
            # K1 variants on the plan's real shifts
            R = 2400
            codes = torch.from_numpy(rng.integers(-7, 8, size=(S, V, R, 2)).astype(np.int8)).to(dev)
            shifts = torch.as_tensor(spec.shifts, dtype=torch.int32, device=dev)
            ct = csr_tensors(inp.csr, dev)
            geom = staged_geometry(spec.shifts, n_t=n_t)
            ct["smax"] = torch.as_tensor(geom.smax, dtype=torch.int32, device=dev)
            cv = torch.as_tensor(inp.c_v, device=dev)
            wv = torch.as_tensor(inp.weights(), device=dev)
            tw = k1_twin(codes.cpu().numpy(), c_v=inp.c_v, offset=0.0, csr=inp.csr,
                         shifts=spec.shifts[:2], t_lo=2000, n_t=2, w_v=inp.w_v)
            # staged K1 as the engine runs it: passes of <= 24 fine DMs
            # (EngineConfig.k1_fdm_per_pass), each with its own window geometry
            bnd = np.linspace(0, F, -(-F // 24) + 1).astype(int)
            k1p = []
            for f0, f1 in zip(bnd[:-1], bnd[1:]):
                shp = np.ascontiguousarray(spec.shifts[int(f0):int(f1)])
                gp = staged_geometry(shp, n_t=n_t)
                cp_ = dict(ct)
                cp_["smax"] = torch.as_tensor(gp.smax, dtype=torch.int32, device=dev)
                k1p.append((int(f0), int(f1), torch.as_tensor(shp, dtype=torch.int32, device=dev),
                            gp, cp_))
            uv_out = torch.empty((F, n_t, inp.csr.n_active, 2), dtype=torch.float32, device=dev)

            def k1_passes():
                for f0, f1, shp, gp, cp_ in k1p:
                    k1_staged(codes, cv, cp_, shp, t_lo=2000, n_t=n_t, w_v=wv, geom=gp,
                              out=uv_out[f0:f1])
            for v in k1_vars:
                if v == "staged":
                    fn = lambda sh_, nt_: k1_staged(codes, cv, ct, sh_, t_lo=2000,  # noqa: E731
                                                    n_t=nt_, w_v=wv, geom=geom)
                else:
                    fn = lambda sh_, nt_: k1_combine(codes, cv, ct, sh_, t_lo=2000,  # noqa: E731
                                                     n_t=nt_, w_v=wv)
                ms = _cuda_ms(k1_passes if v == "staged" else (lambda: fn(shifts, n_t)))
                sub_ = fn(shifts[:2].contiguous(), 2).cpu().numpy()
                exact = bool(np.array_equal(sub_[..., 0], tw.real)
                             and np.array_equal(sub_[..., 1], tw.imag))
                us["k1"][v] = 1e3 * ms / (F * n_t)
                res.append(dict(check="k1", format=fid, variant=v, us_per_image=us["k1"][v],
                                bit_exact_vs_twin=exact, n_fdm=F, spread=geom.spread,
                                win_w=geom.win_w, entry_chunk=geom.ec,
                                csr=inp.csr.summary(), status="ok" if exact else "fail"))
            codes = None
            # FFT variants on this format's map
            plan = FftPlan.build(inp.csr)
            B = F * 64                                 # the engine's batch: F x rows_per_batch
            uvn = (rng.standard_normal((B, inp.csr.n_active))
                   + 1j * rng.standard_normal((B, inp.csr.n_active))).astype(np.complex64)
            G = np.zeros((2, 256, 129), complex)
            G[:, plan.cell_pos, plan.cell_line] = uvn[:2]
            exact = np.fft.irfft2(G, s=(256, 256))
            if plan.fold == "row":
                exact = np.swapaxes(exact, 1, 2)
            twin = image_from_uv_twin(uvn[:2], plan, np.ones((256, 256)), out_dtype=None)
            res.append(dict(check="fft_twin_vs_float64", format=fid, status="ok" if np.max(
                np.abs(twin - exact)) <= 1e-5 * np.abs(exact).max() else "fail"))
            sig = float(np.std(exact))
            uvt = torch.from_numpy(uvn).to(dev)
            mk = torch.ones((256, 256), device=dev)
            for v in fft_vars:
                if v == "cufft":
                    fn = lambda: image_from_uv_torch(uvt, plan, mk, mode="pruned")  # noqa: E731
                elif v == "dense":
                    fn = lambda: image_from_uv_torch(uvt, plan, mk, mode="irfft2")  # noqa: E731
                elif v in ("warp", "fused", "fused_v1", "fused_v2"):
                    fn = _warp_runner(uvt, plan, inp, dev, mode=v)
                else:
                    continue
                ms = _cuda_ms(fn)
                img = fn()[:2].float().cpu().numpy().reshape(2, 256, 256)
                d = float(np.max(np.abs(img - exact)) / sig)
                us["fft"][v] = 1e3 * ms / B
                res.append(dict(check="fft", format=fid, variant=v, us_per_image=us["fft"][v],
                                max_abs_sigma=d, fold=plan.fold,
                                status="ok" if d <= 2e-3 else "fail"))
            chains = _chain_records(fid, us, img_per_s)
            res += chains
            if chains:
                best = min(chains, key=lambda c: c["total_us"])
                summary[fid] = dict(best={k: best[k] for k in ("k1", "fft", "k3b", "total_us",
                                                                "gpu_duty", "status")},
                                    combos={f"{c['k1']}/{c['fft']}/{c['k3b']}":
                                            dict(total_us=round(c["total_us"], 3),
                                                 gpu_duty=round(c["gpu_duty"], 3))
                                            for c in chains},
                                    worst_half=int(wh), images_per_s=img_per_s)
                res.append(dict(check="chain_best", format=fid, **summary[fid]["best"],
                                duty_aim_met=best["gpu_duty"] <= DUTY_AIM))
        except Exception as e:                                   # noqa: BLE001
            res.append(dict(check="format", format=fid, status="fail",
                            error=traceback.format_exc()[-1500:], exc=repr(e)))
    # ---- 1x CSR vs combine_sparse_v4 ------------------------------------------------
    if a.check_1x_csr:
        res.append(_check_1x_csr(dev, rng))
    # the gate: per format, the BEST combination must pass (CP1 selects it); every
    # exactness check must pass; slower combinations are reported, not failed
    gated = [r for r in res if r["check"] not in ("chain",)]
    _write(out, "b8_kernels.json", dict(results=res, summary=summary, gate_us=CHAIN_GATE_US,
                                        duty_aim=DUTY_AIM, duty_gate=DUTY_GATE))
    for fid, s in summary.items():
        print(json.dumps(dict(format=fid, **s["best"])))
    return _status(gated)


def _check_1x_csr(dev, rng) -> dict:
    import numpy as np
    import torch

    from dsart.x2.search.csr import csr_from_lut_1x
    from dsart.x2.search.kernels_triton import csr_tensors, k1_combine
    try:
        from dsart.image.tiled_combine_cuda import (build_inverse_lut, build_sparse_cell_csr,
                                                    combine_sparse_v4, pad_t)
        op, vm = _value_map("1x")
        S = len(vm.streams)
        nfm = max(sv.n_values for sv in vm.streams)
        lut = np.zeros((S, nfm), np.int32)
        nf = np.zeros(S, np.int32)
        for s, sv in enumerate(vm.streams):
            lut[s, :sv.n_values] = sv.cell_row.astype(np.int64) * 256 + sv.cell_col
            nf[s] = sv.n_values
        t_rows, t_det, F = 300, 256, 25
        cells = rng.integers(-127, 128, size=(S, t_rows, nfm, 2)).astype(np.int8)
        dense = torch.zeros((S, nfm, pad_t(t_rows), 2), dtype=torch.int8, device=dev)
        dense[:, :, :t_rows] = torch.from_numpy(cells.transpose(0, 2, 1, 3).copy()).to(dev)
        shifts = rng.integers(-20, 20, size=(F, S)).astype(np.int32)
        csr_v4 = build_sparse_cell_csr(build_inverse_lut(torch.from_numpy(lut).to(dev),
                                                         torch.from_numpy(nf).to(dev),
                                                         n_grid=256), n_grid=256)
        outv4 = torch.zeros((F, t_det, 256, 129), dtype=torch.complex32, device=dev)
        sh_t = torch.from_numpy(shifts).to(dev)
        combine_sparse_v4(dense, csr_v4, sh_t, t_rows=t_rows, t_out_len=t_det, t_lo=0,
                          out=outv4, shift_min=int(shifts.min()), shift_max=int(shifts.max()))
        csr = csr_from_lut_1x(lut, nf, n_grid=256)
        ring = torch.zeros((S, nfm, 512, 2), dtype=torch.int8, device=dev)
        ring[:, :, :t_rows] = dense[:, :, :t_rows]
        uv = k1_combine(ring, torch.ones((S, nfm), device=dev), csr_tensors(csr, dev),
                        sh_t, t_lo=0, n_t=t_det)
        mine = torch.zeros((F, t_det, 256, 129, 2), dtype=torch.float16, device=dev)
        cu = torch.as_tensor(csr.cell_u, device=dev).long()
        cw = torch.as_tensor(csr.cell_w, device=dev).long()
        mine[:, :, cu, cw] = uv.to(torch.float16)
        eq = bool(torch.equal(torch.view_as_real(outv4), mine))
        return dict(check="k1_1x_csr_vs_v4", status="ok" if eq else "fail", equal=eq)
    except Exception as e:                                       # noqa: BLE001
        return dict(check="k1_1x_csr_vs_v4", status="fail", error=repr(e)[:1000])


def _warp_runner(uvt, plan, inp, dev, mode: str = "warp"):
    import numpy as np
    import torch

    from dsart.x2.search.fft import twiddles
    from dsart.x2.search.kernels_cuda import get_module
    from dsart.image.tiled_combine_cuda import _ptr
    import cupy as cp

    N = 256
    B = uvt.shape[0]
    sel = np.nonzero(plan.cell_line_slot >= 0)[0]
    order = sel[np.lexsort((plan.cell_pos[sel], plan.cell_line_slot[sel]))]
    cnt = np.bincount(plan.cell_line_slot[order], minlength=plan.fft_lines.size)
    lp = torch.as_tensor(np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32), device=dev)
    li = torch.as_tensor(order.astype(np.int32), device=dev)
    lpos = torch.as_tensor(plan.cell_pos[order].astype(np.int32), device=dev)
    fl = torch.as_tensor(plan.fft_lines.astype(np.int32), device=dev)
    csel = np.nonzero(plan.cell_const_slot >= 0)[0]
    cci = torch.as_tensor(csel.astype(np.int32), device=dev)
    ccl = torch.as_tensor(plan.cell_line[csel].astype(np.int32), device=dev)
    tw = torch.as_tensor(twiddles(N).view(np.float32).reshape(-1, 2), device=dev)
    pix = torch.full((N, N), 1.0 / (N * N), device=dev)
    uv2 = torch.view_as_real(uvt).contiguous()
    n_fl = int(fl.numel())
    Y = torch.empty((B, n_fl, N, 2), device=dev)
    ring = torch.empty((B, 1, N * N), dtype=torch.float16, device=dev)
    mod = get_module()
    k2, k3 = mod.get_function("k2_colfft_256"), mod.get_function("k3a_rowc2r_256")
    A = int(uvt.shape[1])

    tr = int(plan.fold == "row")
    lines = dict(line_ptr=lp, line_idx=li, line_pos=lpos, fl=fl, cc_idx=cci, cc_line=ccl)

    from dsart.x2.search.kernels_cuda import k23_tables, launch_k23
    tables = k23_tables(lines, n_fl, dev, pix=pix)

    def run_fused(kname="k23v2p_256"):
        launch_k23(uv2, tables, tw, pix, ring, B=B, n_act=A, n_t=B, F=1, t0=0, R_img=B,
                   transpose=tr, stream=torch.cuda.current_stream(), kname=kname)
        return ring.view(B, N, N)

    if mode == "fused":
        return run_fused
    if mode == "fused_v1":
        return lambda: run_fused("k23_fused_256")
    if mode == "fused_v2":
        return lambda: run_fused("k23v2_256")

    def run():
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            k2(((B * n_fl + 3) // 4,), (128,), (_ptr(uv2), _ptr(lp), _ptr(li), _ptr(lpos), _ptr(tw),
                                                _ptr(Y), np.int32(B), np.int32(A),
                                                np.int32(n_fl)), shared_mem=4 * N * 8)
            k3(((B * N + 3) // 4,), (128,), (_ptr(Y), _ptr(uv2), _ptr(fl), _ptr(cci), _ptr(ccl),
                                             _ptr(tw), _ptr(pix), _ptr(ring), np.int32(B),
                                             np.int32(A), np.int32(n_fl), np.int32(int(cci.numel())),
                                             np.int32(B), np.int32(1), np.int64(0), np.int32(B),
                                             np.int32(int(plan.fold == "row"))),
               shared_mem=4 * (N // 2 + 1) * 8)
        return ring.view(B, N, N)
    return run


def _k3b_compat_vs_production(fb, dev) -> dict:
    """Box bank vs today's fp16 cumsum + multi_boxcar_argmax on one cube."""
    import torch

    from dsart.detector.triton_boxcar import multi_boxcar_argmax_triton
    from dsart.x2.search.kernels_triton import k3b_detect

    T, F, P = 256, 4, 65536
    g = torch.Generator(device=dev).manual_seed(0)
    cube = torch.randn((T, F, P), device=dev, generator=g).to(torch.float16)
    cube[100, 1, 1234] = 40.0
    maxw = 64
    cs = torch.zeros((T + maxw, F, P), dtype=torch.float16, device=dev)
    torch.cumsum(cube, dim=0, dtype=torch.float16, out=cs.narrow(0, 1 + maxw // 2, T))
    cs.narrow(0, 1 + maxw // 2 + T, maxw - 1 - maxw // 2).copy_(cs.narrow(0, maxw // 2 + T, 1))
    widths = [f.width for f in fb.filters]
    w_t = torch.tensor(widths, dtype=torch.int32, device=dev)
    off_t = torch.tensor([maxw // 2 - w // 2 for w in widths], dtype=torch.int32, device=dev)
    s_inv = torch.tensor(1 / fb.seeds, dtype=torch.float32, device=dev)
    mx, _win = multi_boxcar_argmax_triton(cs.view(T + maxw, F, P, 1), widths=w_t, offsets=off_t,
                                          sigma_inv=s_inv, n_out=T - maxw, t_base=maxw // 2)
    prod = mx.view(T - maxw, F, P).float().view(3, 64, F, P).amax(dim=1)
    best, _tk, _ = k3b_detect(cube.contiguous(), ring_row0=0, c_first=maxw // 2, n_c=T - maxw,
                              sigma_inv=torch.ones(F, device=dev), s_inv=1 / fb.seeds, bank=fb)
    d = (best - prod).abs()
    # production's boxcar is a difference of two fp16 running sums: each carries
    # up to half an fp16 ulp of its magnitude (|cumsum| reaches ~40 over 256
    # rows: ulp 2^-5), so the tolerance includes one ulp of the window's largest
    # |cumsum| (x max s_inv = 1, b1) besides 0.02 and the stored max's ulp
    cs_max = cs.float().abs().amax(dim=0)                                  # [F, P]
    tol = 0.02 + 2.0 * torch.abs(prod) * 2.0 ** -11 + 2.0 ** -10 * cs_max[None]
    worst = float((d - tol).max())
    return dict(compat_max_abs=float(d.max()), compat_status="ok" if worst <= 0 else "fail",
                compat_note="tolerance 0.02 + 1 fp16 ulp of the stored max + 1 fp16 ulp of "
                            "the largest |cumsum| (production's fp16 running sums)")


class _FakeRecording:
    """WS-T ``Recording`` duck type over an in-memory 1x payload."""

    def __init__(self, payload, t0):
        from dsart.x2.search.replay import F_WRITTEN, VF_DATA_PRESENT
        import numpy as np
        self.p, self.t0 = payload, int(t0)
        T, C, _ = payload.shape
        self.cf = np.full((T, C), F_WRITTEN, np.uint8)
        self.vf = np.full((T, C), VF_DATA_PRESENT, np.uint16)

    def t_range(self, dm):
        return self.t0, self.t0 + self.p.shape[0]

    def slots(self, dm, a, b):
        i, j = a - self.t0, b - self.t0
        return dict(payload=self.p[i:j], vf=self.vf[i:j], corr_flags=self.cf[i:j])


def cmd_compat_1x(a) -> int:
    """(1) kernel taps: K1 1x CSR vs combine_sparse_v4 bit for bit, K3b box vs
    production fp16 boxcars; (2) today's production chain composed on CPU
    (``compat1x.production_cycle_cpu``) vs the owner engine on the GPU in
    1x-compat (``EngineConfig.compat_1x``) cycle by cycle on a synthetic 1x
    recording made with the REAL 1x value map: every production candidate
    matched within t / fine DM / l / m +- 1 and dS/N <= 0.1."""
    _need_gpu()
    import numpy as np

    from dsart.x2.search.codec import RowFormat
    from dsart.x2.search.geometry import owner_plan_1x
    from dsart.x2.search.replay import replay_1x
    from dsart.x2.search.ring import MemoryRing
    from dsart.x2.search.synth import Burst, SyntheticSky

    res = []
    a2 = argparse.Namespace(out=a.out, fft="cufft,warp,fused", k3b="box,v1,fused,reg", k1="triton",
                            k1_formats="", check_1x_csr=True)
    rc = cmd_kernels(a2)
    res.append(dict(check="kernels_taps", status="ok" if rc == 0 else "fail"))
    try:
        op, vm = _value_map("1x")
        S = len(vm.streams)
        nfm = max(int(sv.n_values) for sv in vm.streams)
        cells = []
        for sv in vm.streams:
            c = np.full(nfm, -1, np.int64)
            c[:sv.n_values] = sv.cell_row.astype(np.int64) * 256 + sv.cell_col
            cells.append(c)
        lut = np.stack([np.where(c >= 0, c, 0) for c in cells]).astype(np.int32)
        nf = np.array([(c >= 0).sum() for c in cells], np.int32)
        fine = 500.0 + 2.0 * (np.arange(25) - 12)
        plan = owner_plan_1x(coarse_dm=np.array([500.0]), fine_dm=fine,
                             fine_to_coarse=np.zeros(25, int), owned=[0], n_sub=max(1, S // 16),
                             csf=8)
        spec = plan.dms[0]
        fmt = RowFormat("cint8", n_values=nfm)
        n_cyc = int(a.cycles)
        t0, T = 100, (n_cyc + 5) * spec.hop + spec.max_pos + spec.max_neg
        tmp = MemoryRing({0: fmt}, S, t_buf=T + 8)
        bursts = [Burst(d=0, f=f, t=t, l=l, m=m, amp=amp) for f, t, l, m, amp in
                  ((3, 4 * spec.hop + 40, 100, 140, 0.4), (12, 5 * spec.hop + 90, 60, 200, 0.8),
                   (20, 6 * spec.hop + 15, 180, 70, 1.6))]
        sky = SyntheticSky(plan=plan, cells={0: cells}, formats={0: fmt}, n_grid=256, seed=5,
                           bursts=bursts)
        sky.fill(tmp, spec, t0, T)
        raw, _ = tmp.wait_window(0, t0, T)
        payload = np.ascontiguousarray(raw.transpose(1, 0, 2)).view(np.int8)
        out = replay_1x(_FakeRecording(payload, t0), 0, plan=plan, lut=lut, n_filled=nf,
                        n_grid=256, n_cycles=n_cyc, fft_mode=a.fft_mode, backend="gpu",
                        cfg_kw=dict(k1_impl="triton", k3b_impl=a.k3b))
        n_prod = sum(c["n_prod"] for c in out["cycles"])
        res.append(dict(check="production_vs_engine_1x", status="ok" if (out["ok"] and n_prod)
                        else "fail", n_cycles=len(out["cycles"]), n_prod_candidates=n_prod,
                        fft_mode=a.fft_mode, k3b=a.k3b, cycles=out["cycles"]))
    except Exception as e:                                       # noqa: BLE001
        res.append(dict(check="production_vs_engine_1x", status="fail",
                        error=traceback.format_exc()[-2000:], exc=repr(e)))
    _write(Path(a.out), "b8_compat_1x.json", dict(results=res))
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k not in ("cycles", "error")}))
    return _status(res)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class ClockedRing:
    """Synthetic ring for the engine bench: rows become available at the
    real-time rate; payload drawn from a pre-generated pool (no per-cycle
    generation cost, no shm)."""

    def __init__(self, plan, formats, pool_rows: int = 4096, seed: int = 0):
        import numpy as np
        rng = np.random.default_rng(seed)
        self.plan = plan
        self.fmt = formats
        self.pool = {}
        for spec in plan.dms:
            f = formats[spec.d]
            self.pool[spec.d] = rng.integers(0, 256, size=(spec.n_stream, pool_rows, f.row_bytes),
                                             dtype=np.uint8)
        self.t0 = time.monotonic()
        self.start_cycle = 2
        self.n_wait = 0

    def dms(self):
        return sorted(self.pool)

    def row_format(self, d):
        return self.fmt[d]

    def write_pos(self, d):
        spec = self.plan.by_d(d)
        cyc = self.start_cycle + (time.monotonic() - self.t0) / CYCLE_S
        return int(cyc * spec.hop)

    @property
    def row_bytes(self):
        return max(int(p.shape[2]) for p in self.pool.values())

    def wait_window(self, d, t_first, n_rows, timeout_s=None):
        return self.wait_window_into(d, t_first, n_rows, out=None, timeout_s=timeout_s)

    def wait_window_into(self, d, t_first, n_rows, *, out=None, timeout_s=None):
        """Like RingV4Source: rows copied straight into ``out`` (the engine's
        pinned staging) -- one host copy, as in production."""
        import numpy as np
        while self.write_pos(d) < t_first + n_rows:
            self.n_wait += 1
            time.sleep(0.0005)
        pool = self.pool[d]
        S, n_pool, rb = pool.shape
        if out is None:
            out = np.empty((S, n_rows, self.row_bytes), np.uint8)
        i0 = int(t_first) % n_pool
        n1 = min(n_rows, n_pool - i0)
        out[:, :n1, :rb] = pool[:, i0:i0 + n1]
        if n1 < n_rows:
            out[:, n1:, :rb] = pool[:, :n_rows - n1]
        return out, np.ones((S, n_rows), bool)

    def counters(self):
        return dict(n_wait=self.n_wait)


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _variants(a) -> list:
    """``--variants k1:k3b,...`` (default: staged:reg)."""
    out = []
    for tok in [t for t in a.variants.split(",") if t]:
        k1, _, k3b = tok.partition(":")
        if k1 not in K1_VARIANTS or k3b not in K3B_IMPLS:
            raise SystemExit(f"--variants: {tok!r} is not <{'|'.join(K1_VARIANTS)}>:"
                             f"<{'|'.join(K3B_IMPLS)}>")
        out.append((k1, k3b))
    return out


def _stage_us(stage_ms: dict, images: int) -> dict:
    return {k: (1e3 * v / images if images else 0.0) for k, v in stage_ms.items()}


def run_half(a, half: int, k1: str, k3b: str, seconds: float, part: Path) -> dict:
    """One owner half on this process's GPU for ``seconds`` of data."""
    from dsart.x2.search.engine import EngineConfig, OwnerEngine
    from dsart.x2.search.geometry import owner_plan_from_tiered
    from dsart.x2.search.ring import RingV4Source
    from dsart.x2.search.service import CycleScheduler
    from dsart.x2.search.valuemap import dm_inputs_from_value_map

    op, vm = _value_map(a.opoint, a.antpos_npz, a.dec_deg)
    plan = _plan_for(op, a.plan)
    oplan = owner_plan_from_tiered(plan, half=half)
    sm, sh = _steps(op)
    inp = dm_inputs_from_value_map(vm, s_main=sm, s_heavy=sh, bits_main=op.bits_main)
    inputs = {s.d: inp for s in oplan.dms}
    head = dict(half=half, opoint=a.opoint, k1=k1, k3b=k3b, input=a.input,
                plan_id=str(getattr(plan, "plan_id", "")), n_dm=len(oplan.dms),
                dms=oplan.summary()["dms"], images_per_cycle=int(
                    sum(s.n_fdm * s.hop for s in oplan.dms)))
    head["images_per_s"] = head["images_per_cycle"] / CYCLE_S
    if a.input == "synthetic-ring":
        ring = ClockedRing(oplan, {s.d: inp.fmt for s in oplan.dms})
    else:
        ring = RingV4Source.open(a.shm_name, owned=[s.d for s in oplan.dms],
                                 access=a.ring_access, attach_timeout_s=a.attach_timeout_s)
        ring.validate(oplan, inputs, vmap=vm)
        head["ring_access"] = ring.access
    cfg = EngineConfig(backend="gpu" if not a.cpu else "cpu", fft_mode=a.fft_mode,
                       families=("b3x",) if a.bank == "b3x" else ("b",), n_grid=256,
                       wait_timeout_s=5.0, k1_impl=k1, k3b_impl=k3b)
    eng = OwnerEngine(oplan, inputs, ring, cfg)
    sched = CycleScheduler(oplan, ring, max_lag_cycles=8)
    head["engine_status"] = eng.status()
    rec, stage_ms, images = [], {}, 0
    t_first = None
    stop = "minutes"
    t_part = time.monotonic()
    first_timeout = a.first_data_timeout_s
    while True:
        try:
            k = sched.next_cycle(timeout_s=first_timeout if t_first is None else a.idle_stop_s)
        except TimeoutError:
            stop = "no_data" if t_first is None else "idle"
            break
        t_a = time.monotonic()
        t_first = t_first or t_a
        r = eng.process_cycle(k)
        wall = 1e3 * (time.monotonic() - t_a)
        by = eng._gpu.pop_gpu_ms(by_stage=True) if eng._gpu is not None else {}
        cyc_stage = {}
        for dct in by.values():
            for kk, v in dct.items():
                cyc_stage[kk] = cyc_stage.get(kk, 0.0) + v
        for kk, v in cyc_stage.items():
            stage_ms[kk] = stage_ms.get(kk, 0.0) + v
        images += head["images_per_cycle"]
        gpu = float(sum(v for kk, v in cyc_stage.items() if kk != "h2d"))
        lag = max(sched.lag_ms(k).values(), default=0.0)
        rec.append(dict(cycle=k, wall_ms=wall, gpu_ms=gpu, lag_ms=lag,
                        n_cand=len(r.candidates), phase_ms=dict(r.phase_ms)))
        now = time.monotonic()
        if now - t_part >= 15.0:
            _atomic_json(part, dict(head, final=False, **_half_summary(rec, stage_ms, images,
                                                                        sched, ring)))
            t_part = now
        if now - t_first >= seconds:
            break
    out = dict(head, final=True, stop=stop, **_half_summary(rec, stage_ms, images, sched, ring))
    _atomic_json(part, out)
    if hasattr(ring, "close"):
        ring.close()
    return out


def _half_summary(rec, stage_ms, images, sched, ring) -> dict:
    import numpy as np
    gpu = np.array([x["gpu_ms"] for x in rec]) if rec else np.zeros(0)
    wall = np.array([x["wall_ms"] for x in rec]) if rec else np.zeros(0)
    lag = np.array([x["lag_ms"] for x in rec]) if rec else np.zeros(0)
    st_us = _stage_us(stage_ms, images)
    chain = float(sum(v for k, v in st_us.items() if k != "h2d"))
    duty = float(gpu.mean() / (1e3 * CYCLE_S)) if gpu.size else 0.0
    host = float(np.percentile(np.maximum(wall - gpu, 0), 99)) if wall.size else 0.0
    phase = {}
    for x in rec:
        for kk, v in x.get("phase_ms", {}).items():
            phase.setdefault(kk, []).append(v)
    phase = {kk: dict(mean=float(np.mean(v)), p99=float(np.percentile(v, 99)))
             for kk, v in phase.items()}
    fan = {}
    if hasattr(ring, "fan_in"):
        fan = {int(d): ring.fan_in(d) for d in ring.dms()}
    return dict(n_cycles=len(rec), gpu_duty=duty, host_phase_ms=phase,
                wall_ms_p99=float(np.percentile(wall, 99)) if wall.size else 0.0,
                gpu_ms_mean=float(gpu.mean()) if gpu.size else 0.0,
                gpu_duty_p99=float(np.percentile(gpu, 99) / (1e3 * CYCLE_S)) if gpu.size else 0.0,
                host_ms_p99=host, lag_ms_max=float(lag.max(initial=0)),
                lag_ms_last=float(lag[-1]) if lag.size else 0.0,
                lag_recovery_seeks=int(sched.n_seeks), cycles_skipped=int(sched.n_skipped),
                stage_us_per_image=st_us, chain_us_per_image=chain, fan_in=fan,
                cycles=rec[-20:])


def _engine_gate(h: dict) -> dict:
    # real time: every cycle's wall time (GPU + the host work it does not hide)
    # within 90% of the cycle at p99, no lag-recovery seek, no growing lag; the
    # host's share is reported (host_ms_p99) but gated only through the wall
    wall_p99 = h.get("wall_ms_p99", float("inf"))
    ok = (h.get("n_cycles", 0) > 0 and h["gpu_duty"] <= DUTY_GATE
          and wall_p99 <= 0.9 * 1e3 * CYCLE_S
          and h["lag_ms_last"] <= 2.0 * 1e3 * CYCLE_S and h["lag_recovery_seeks"] == 0
          and h["chain_us_per_image"] <= CHAIN_GATE_US)
    return dict(check="engine", half=h["half"], k1=h["k1"], k3b=h["k3b"],
                plan_id=h.get("plan_id"), n_dm=h.get("n_dm"),
                images_per_s=h.get("images_per_s"), gpu_ms_mean=h.get("gpu_ms_mean"),
                wall_ms_p99=h.get("wall_ms_p99"), host_phase_ms=h.get("host_phase_ms"),
                n_cycles=h.get("n_cycles", 0), stop=h.get("stop"), gpu_duty=h["gpu_duty"],
                duty_aim_met=h["gpu_duty"] <= DUTY_AIM, host_ms_p99=h["host_ms_p99"],
                lag_ms_max=h["lag_ms_max"], lag_recovery_seeks=h["lag_recovery_seeks"],
                stage_us_per_image=h["stage_us_per_image"],
                chain_us_per_image=h["chain_us_per_image"], gate_us=CHAIN_GATE_US,
                status="ok" if ok else "fail")


def cmd_engine(a) -> int:
    """One process per (half, GPU) per variant; variants run one after
    another for minutes / n_variants each. ``--out`` (a directory) gets
    ``b8_engine.json`` (rewritten after every variant and at the end; the
    only ``*.json`` at its top level) and ``halves/`` (per-half progress)."""
    out = Path(a.out)
    if a.half is not None:                                       # child
        k1, _, k3b = a.variants.partition(":")
        run_half(a, int(a.half), k1, k3b, float(a.seconds),
                 out / "halves" / f"{k1}_{k3b}_half{a.half}.json")
        return EXIT_OK
    halves = ([int(h) for h in a.halves.split(",") if h != ""] if a.halves else
              [a.half_base + i for i in range(len([g for g in a.gpus.split(",") if g != ""]))])
    gpus = [g for g in a.gpus.split(",") if g != ""]
    if len(gpus) < len(halves):
        raise SystemExit(f"{len(halves)} halves need {len(halves)} GPUs (--gpus {a.gpus})")
    variants = _variants(a)
    seconds = 60.0 * a.minutes / max(1, len(variants))
    summary = dict(opoint=a.opoint, input=a.input, halves=halves, minutes=a.minutes,
                   variants=[f"{k1}:{k3b}" for k1, k3b in variants], gate_us=CHAIN_GATE_US,
                   duty_aim=DUTY_AIM, duty_gate=DUTY_GATE, results=[], final=False)
    _atomic_json(out / "b8_engine.json", summary)
    no_data = False
    for k1, k3b in variants:
        if no_data:
            summary["results"] += [dict(check="engine", half=h, k1=k1, k3b=k3b, status="skip",
                                        reason="no data in an earlier variant") for h in halves]
            continue
        procs = []
        for h, g in zip(halves, gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                       DSART_X2_CP=os.environ.get("DSART_X2_CP", "b8"))
            cmd = [sys.executable, __file__, "engine", "--half", str(h), "--variants",
                   f"{k1}:{k3b}", "--seconds", str(seconds)] + _child_args(a)
            procs.append(subprocess.Popen(cmd, env=env))
        rcs = [p.wait() for p in procs]
        for h, rc in zip(halves, rcs):
            p = out / "halves" / f"{k1}_{k3b}_half{h}.json"
            if not p.exists():
                summary["results"].append(dict(check="engine", half=h, k1=k1, k3b=k3b,
                                               status="fail", rc=rc))
                continue
            hj = json.loads(p.read_text())
            if hj.get("stop") == "no_data":
                no_data = True
            summary["results"].append(_engine_gate(hj))
        _atomic_json(out / "b8_engine.json", summary)
    res = summary["results"]
    by_var = {}
    for r in res:
        by_var.setdefault(f"{r['k1']}:{r['k3b']}", []).append(r["status"])
    passing = [v for v, st in by_var.items() if st and all(s == "ok" for s in st)]
    summary.update(final=True, passing_variants=passing,
                   status="ok" if passing else "fail")
    _atomic_json(out / "b8_engine.json", summary)
    print(json.dumps(dict(passing_variants=passing, out=str(out / "b8_engine.json"))))
    return EXIT_OK if passing else EXIT_GATE


def _child_args(a) -> list:
    args = ["--opoint", a.opoint, "--input", a.input, "--out", a.out, "--fft-mode", a.fft_mode,
            "--bank", a.bank, "--ring-access", a.ring_access,
            "--attach-timeout-s", str(a.attach_timeout_s),
            "--first-data-timeout-s", str(a.first_data_timeout_s),
            "--idle-stop-s", str(a.idle_stop_s), "--dec-deg", str(a.dec_deg)]
    for flag, val in (("--shm-name", a.shm_name), ("--plan", a.plan),
                      ("--antpos-npz", a.antpos_npz)):
        if val:
            args += [flag, val]
    return args + (["--cpu"] if a.cpu else [])


def cmd_teardown(a) -> int:
    import glob
    removed = []
    for p in glob.glob(SHM_PREFIX + "*"):
        try:
            if os.stat(p).st_uid == os.getuid():
                os.remove(p)
                removed.append(p)
        except OSError:
            pass
    print(json.dumps(dict(removed=removed)))
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile-check")
    c.add_argument("--out", default="b8_out")
    k = sub.add_parser("kernels")
    k.add_argument("--fft", default="cufft,warp,fused_v1,fused_v2,fused",
                   help="cufft, dense, warp, fused (k23v2p_256), fused_v2 (k23v2_256), "
                        "fused_v1 (k23_fused_256)")
    k.add_argument("--k3b", default="box,b3x,v1,fused,cuda,reg",
                   help="filter banks (box, b3x) and implementations (v1, fused, cuda, reg)")
    k.add_argument("--k1", default="triton,staged", help="K1 variants (triton, staged)")
    k.add_argument("--k1-formats", dest="k1_formats", default="all")
    k.add_argument("--check-1x-csr", dest="check_1x_csr", action="store_true")
    k.add_argument("--out", default="b8_out")
    x = sub.add_parser("compat-1x")
    x.add_argument("--out", default="b8_out")
    x.add_argument("--cycles", type=int, default=4)
    x.add_argument("--fft-mode", dest="fft_mode", default="fused", choices=("warp", "fused", "pruned"))
    x.add_argument("--k3b", default="reg", choices=("v1", "fused", "cuda", "reg"))
    e = sub.add_parser("engine")
    e.add_argument("--gpus", default="0,1")
    e.add_argument("--halves", default="",
                   help="owner halves, one per GPU (WS-T fleet-load: 2j,2j+1); default "
                        "--half-base + GPU index")
    e.add_argument("--input", default=None, choices=("synthetic-ring", "rx-v4"),
                   help="default: rx-v4 when --shm-name is given, else synthetic-ring")
    e.add_argument("--shm-name", "--shm", dest="shm_name", default="", help="ring v4 (WS-T)")
    e.add_argument("--plan", default="", help="TieredDmPlan npz (default: built for --opoint)")
    e.add_argument("--opoint", default="2x-gridded-ns2-cint4")
    e.add_argument("--minutes", type=float, default=10.0, help="total, split over --variants")
    e.add_argument("--out", default="b8_out", help="directory; b8_engine.json is the result")
    e.add_argument("--variants", default="staged:reg",
                   help="k1:k3b pairs, run one after another (minutes split between them); "
                        "the default is the G0 pick (staged K1, register K3b, fused FFT)")
    e.add_argument("--ring-access", dest="ring_access", default="pread",
                   choices=("pread", "mmap"))
    e.add_argument("--attach-timeout-s", dest="attach_timeout_s", type=float, default=300.0)
    e.add_argument("--first-data-timeout-s", dest="first_data_timeout_s", type=float,
                   default=600.0)
    e.add_argument("--idle-stop-s", dest="idle_stop_s", type=float, default=10.0)
    e.add_argument("--antpos-npz", dest="antpos_npz", default="")
    e.add_argument("--dec-deg", dest="dec_deg", type=float, default=DEC_DEG)
    e.add_argument("--half", default=None, help=argparse.SUPPRESS)
    e.add_argument("--seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    e.add_argument("--half-base", dest="half_base", type=int, default=0)
    e.add_argument("--fft-mode", dest="fft_mode", default="fused")
    e.add_argument("--bank", default="b3x", choices=("b3x", "box"))
    e.add_argument("--cpu", action="store_true", help="CPU twins (plumbing dry run only)")
    sub.add_parser("teardown")
    sub.add_parser("_selftest-interp")
    a = ap.parse_args(argv)
    if a.cmd == "engine" and a.input is None:
        a.input = "rx-v4" if a.shm_name else "synthetic-ring"
    try:
        if a.cmd == "compile-check":
            return cmd_compile_check(a)
        if a.cmd == "kernels":
            return cmd_kernels(a)
        if a.cmd == "compat-1x":
            return cmd_compat_1x(a)
        if a.cmd == "engine":
            return cmd_engine(a)
        if a.cmd == "teardown":
            return cmd_teardown(a)
        if a.cmd == "_selftest-interp":
            return selftest_interp()
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        try:
            _write(Path(getattr(a, "out", "b8_out")), f"b8_{a.cmd}_error.json",
                   dict(error=traceback.format_exc()))
        except Exception:                                        # noqa: BLE001
            pass
        return EXIT_ERR
    return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
