#!/usr/bin/env python
"""B8 -- the 2x owner engine at CP1 (WS-S). Contract: docs/x2/CP1_RUNBOOK.md,
tools/x2/cp_runner_plans.py (the command lines are the interface).

Subcommands (exit 0 = every gate passed, 1 = a gate failed, 2 = error):

  compile-check                 h23 CPU (CUDA_VISIBLE_DEVICES=""): every module parses;
                                CUDA K0/K2/K3a compile with NVRTC (compile only); Triton
                                K1/K3b compile for sm_75 (no launch); CPU twins and the
                                TRITON_INTERPRET runs agree with their references.
  kernels --fft cufft,warp --k3b box,b3x --k1-formats all --check-1x-csr
                                GPU: us/image per stage and variant; K1 1x CSR bit-exact
                                vs combine_sparse_v4; FFT |d| <= 2e-3 sigma vs float64;
                                K3b == twin (bit) and box vs production fp16 boxcars
                                |dS/N| <= 0.02 (+fp16 ulp); chain total vs 4.8 us/image (E1).
  compat-1x                     GPU: the engine at today's parameters vs production taps
                                (K1 == v4 bit for bit, image vs fp16 cuFFT+prescale,
                                detector vs multi_boxcar_argmax).
  engine --gpus 0,1 --input synthetic-ring|rx-v4 --opoint <id> --minutes M
                                one owner half per GPU, real-time paced synthetic ring
                                (in-process; no shm) or ring v4: GPU duty <= 0.75 (aim
                                0.6), bounded lag, zero ring overruns, host <= 30 ms/cycle.
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
    try:
        import cupy
        from cupy.cuda import compiler
    except Exception as e:                                       # noqa: BLE001
        return dict(check="cuda_nvrtc", status="skip", reason=f"cupy unavailable: {e!r}")
    from dsart.x2.search.kernels_cuda import CUDA_SOURCE
    opts = ["--std=c++14"]
    try:
        cuda_path = cupy.cuda.get_cuda_path()
        if cuda_path:
            opts.append("-I" + os.path.join(cuda_path, "include"))
    except Exception:                                            # noqa: BLE001
        pass
    try:
        ptx = compiler.compile_using_nvrtc(CUDA_SOURCE, options=tuple(opts), arch="compute_75")
        names = [n for n in ("k0_unpack_frame2", "k0_unpack_cint8", "k2_colfft_256",
                             "k3a_rowc2r_256") if n in (ptx if isinstance(ptx, str) else str(ptx))]
        return dict(check="cuda_nvrtc", status="ok", arch="compute_75", kernels=names)
    except Exception as e:                                       # noqa: BLE001
        msg = repr(e)
        st = "fail" if ("error" in msg and "CUDA" not in msg[:40]) else "skip"
        return dict(check="cuda_nvrtc", status=st, error=msg[:2000])


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
    k1 = dict(codes_ptr="*i8", cv_ptr="*fp32", ptr_ptr="*i32", es_ptr="*i32", ev_ptr="*i32",
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
    sh = rng.integers(0, 3, size=(2, 3)).astype(np.int32)
    want = k1_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=sh, t_lo=5, n_t=3)
    got = k1_combine(torch.from_numpy(codes), torch.from_numpy(cv), csr_tensors(csr, "cpu"),
                     torch.from_numpy(sh), t_lo=5, n_t=3, block_c=16, block_t=4).numpy()
    ok = np.array_equal(got[..., 0], want.real) and np.array_equal(got[..., 1], want.imag)
    fb = FilterBank.build((1, 2, 4))
    ring = rng.standard_normal((70, 1, 8)).astype(np.float16)
    b0, t0, _ = k3b_twin(ring, ring_row0=0, c_first=4, n_c=64, sigma_inv=np.ones(1, np.float32),
                         s_inv=np.ones(3, np.float32), bank=fb)
    b1, t1, _ = k3b_detect(torch.from_numpy(ring), ring_row0=0, c_first=4, n_c=64,
                           sigma_inv=torch.ones(1), s_inv=np.ones(3), bank=fb, block=8)
    ok &= np.array_equal(b1.numpy(), b0) and np.array_equal(t1.numpy(), t0)
    print("interp K1/K3b bit-exact:", ok)
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


def _value_map(opoint_id: str):
    import numpy as np
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.opoint import opoint_from_id

    op = opoint_from_id(opoint_id)
    z = np.load(FIXTURE)
    vm = build_value_map_for_opoint(op, z["antpos_e"], z["antpos_n"],
                                    is_core_baseline_mask=core_baseline_mask_from_station_numbers(
                                        z["station"]), dec_deg=DEC_DEG)
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


def cmd_kernels(a) -> int:
    _need_gpu()
    import numpy as np
    import torch

    from dsart.x2.search.csr import csr_from_lut_1x
    from dsart.x2.search.fft import FftPlan, image_from_uv_torch, image_from_uv_twin
    from dsart.x2.search.filters import FilterBank, k3b_twin
    from dsart.x2.search.kernels_triton import csr_tensors, k1_combine, k3b_detect
    from dsart.x2.search.combine import k1_twin
    from dsart.x2.search.valuemap import dm_inputs_from_value_map

    out = Path(a.out)
    res = []
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    fmts = ["2x-gridded-ns2-cint4", "2x-gridded-ns4-cint5", "2x-gridded-ns4-cint4",
            "2x-baselines-ns4-cint5", "2x-baselines-ns2-cint4", "2x-baselines-ns4-cint4"]
    if a.k1_formats not in ("all", ""):
        fmts = a.k1_formats.split(",")
    stage_us = {}
    # ---- K1 per format --------------------------------------------------------------
    for fid in fmts:
        try:
            op, vm = _value_map(fid)
            sm, sh = _steps(op)
            inp = dm_inputs_from_value_map(vm, s_main=sm, s_heavy=sh, bits_main=op.bits_main)
            S, V = inp.c_v.shape
            R = 2400
            codes = torch.from_numpy(rng.integers(-7, 8, size=(S, V, R, 2)).astype(np.int8)).to(dev)
            F, n_t = 18, 64
            shifts = torch.from_numpy(rng.integers(0, 1700, size=(F, S)).astype(np.int32)).to(dev)
            ct = csr_tensors(inp.csr, dev)
            cv = torch.as_tensor(inp.c_v, device=dev)
            ms = _cuda_ms(lambda: k1_combine(codes, cv, ct, shifts, t_lo=2000, n_t=n_t))
            # bit check on a slice vs the twin
            sub = k1_combine(codes, cv, ct, shifts[:2], t_lo=2000, n_t=2).cpu().numpy()
            tw = k1_twin(codes.cpu().numpy(), c_v=inp.c_v, offset=0.0, csr=inp.csr,
                         shifts=shifts[:2].cpu().numpy(), t_lo=2000, n_t=2)
            exact = bool(np.array_equal(sub[..., 0], tw.real) and np.array_equal(sub[..., 1],
                                                                                   tw.imag))
            us = 1e3 * ms / (F * n_t)
            stage_us.setdefault("k1", {})[fid] = us
            res.append(dict(check="k1", format=fid, us_per_image=us, bit_exact_vs_twin=exact,
                            csr=inp.csr.summary(), status="ok" if exact else "fail"))
        except Exception as e:                                   # noqa: BLE001
            res.append(dict(check="k1", format=fid, status="fail", error=repr(e)[:1000]))
    # ---- 1x CSR vs combine_sparse_v4 ------------------------------------------------
    if a.check_1x_csr:
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
            res.append(dict(check="k1_1x_csr_vs_v4", status="ok" if eq else "fail", equal=eq))
        except Exception as e:                                   # noqa: BLE001
            res.append(dict(check="k1_1x_csr_vs_v4", status="fail", error=repr(e)[:1000]))
    # ---- FFT variants -----------------------------------------------------------------
    try:
        op, vm = _value_map("2x-gridded-ns2-cint4")
        sm, sh = _steps(op)
        inp = dm_inputs_from_value_map(vm, s_main=sm, s_heavy=sh, bits_main=8)
        plan = FftPlan.build(inp.csr)
        B = 64
        uvn = (rng.standard_normal((B, inp.csr.n_active))
               + 1j * rng.standard_normal((B, inp.csr.n_active))).astype(np.complex64)
        G = np.zeros((2, 256, 129), complex)                      # [pos, line]: line = half axis
        G[:, plan.cell_pos, plan.cell_line] = uvn[:2]
        exact = np.fft.irfft2(G, s=(256, 256))
        if plan.fold == "row":
            exact = np.swapaxes(exact, 1, 2)
        twin = image_from_uv_twin(uvn[:2], plan, np.ones((256, 256)), out_dtype=None)
        res.append(dict(check="fft_twin_vs_float64", status="ok" if np.max(np.abs(
            twin - exact)) <= 1e-5 * np.abs(exact).max() else "fail"))
        sig = float(np.std(exact))
        uvt = torch.from_numpy(uvn).to(dev)
        mk = torch.ones((256, 256), device=dev)
        for v in a.fft.split(","):
            if v == "cufft":
                fn = lambda: image_from_uv_torch(uvt, plan, mk, mode="pruned")  # noqa: E731
            elif v == "dense":
                fn = lambda: image_from_uv_torch(uvt, plan, mk, mode="irfft2")  # noqa: E731
            elif v == "warp":
                fn = _warp_runner(uvt, plan, inp, dev)
            else:
                continue
            ms = _cuda_ms(fn)
            img = fn()[:2].float().cpu().numpy().reshape(2, 256, 256)
            d = float(np.max(np.abs(img - exact)) / sig)
            stage_us.setdefault("fft", {})[v] = 1e3 * ms / B
            res.append(dict(check="fft", variant=v, us_per_image=1e3 * ms / B, max_abs_sigma=d,
                            status="ok" if d <= 2e-3 else "fail"))
    except Exception as e:                                       # noqa: BLE001
        res.append(dict(check="fft", status="fail", error=traceback.format_exc()[-1500:],
                        exc=repr(e)))
    # ---- K3b ------------------------------------------------------------------------------
    for bank_name in a.k3b.split(","):
        try:
            fb = FilterBank.b3x() if bank_name == "b3x" else FilterBank.build()
            F, P, hop = 18, 65536, 384
            R = hop + fb.history()
            ring = torch.randn((R, F, P), device=dev).to(torch.float16)
            si = torch.ones(F, device=dev)
            ms = _cuda_ms(lambda: k3b_detect(ring, ring_row0=0, c_first=fb.max_d + fb.max_a,
                                             n_c=hop, sigma_inv=si, s_inv=1 / fb.seeds,
                                             bank=fb), reps=2)
            us = 1e3 * ms / (F * hop)
            stage_us.setdefault("k3b", {})[bank_name] = us
            small = ring[:, :2, :4096].contiguous()
            b1, t1, _ = k3b_detect(small, ring_row0=0, c_first=fb.max_d + fb.max_a, n_c=64,
                                   sigma_inv=si[:2], s_inv=1 / fb.seeds, bank=fb)
            b0, t0, _ = k3b_twin(small.cpu().numpy(), ring_row0=0, c_first=fb.max_d + fb.max_a,
                                 n_c=64, sigma_inv=np.ones(2, np.float32),
                                 s_inv=(1 / fb.seeds).astype(np.float32), bank=fb)
            exact = bool(np.array_equal(b1.cpu().numpy(), b0) and
                         np.array_equal(t1.cpu().numpy(), t0))
            entry = dict(check="k3b", bank=bank_name, us_per_image=us, bit_exact_vs_twin=exact,
                         status="ok" if exact else "fail")
            if bank_name == "box":
                entry.update(_k3b_compat_vs_production(fb, dev))
                if entry.get("compat_status") == "fail":
                    entry["status"] = "fail"
            res.append(entry)
        except Exception as e:                                   # noqa: BLE001
            res.append(dict(check="k3b", bank=bank_name, status="fail", error=repr(e)[:1000]))
    chain = {k: min(v.values()) for k, v in stage_us.items() if v}
    total = sum(chain.values()) + 0.05 + 0.03 + 0.18             # K0, Layer-1, top-k/launch
    res.append(dict(check="chain", us_per_image_best=chain, total_us=total,
                    status="ok" if total <= 4.8 else "fail",
                    note="E1 escalation if > 4.8 us/image after the fallbacks"))
    _write(out, "b8_kernels.json", dict(results=res))
    return _status(res)


def _warp_runner(uvt, plan, inp, dev):
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
    cube = (torch.randn((T, F, P), device=dev) * 1.0).to(torch.float16)
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
    tol = 0.02 + 2.0 * torch.abs(prod) * 2.0 ** -11
    worst = float((d - tol).max())
    return dict(compat_max_abs=float(d.max()), compat_status="ok" if worst <= 0 else "fail",
                compat_note="tolerance 0.02 + 1 fp16 ulp of the stored max")


def cmd_compat_1x(a) -> int:
    _need_gpu()
    res = []
    # the kernels-level taps (K1 1x CSR vs v4, K3b box vs production) are run here too
    a2 = argparse.Namespace(out=a.out, fft="cufft,warp", k3b="box", k1_formats="",
                            check_1x_csr=True)
    rc = cmd_kernels(a2)
    res.append(dict(check="kernels_taps", status="ok" if rc == 0 else "fail"))
    _write(Path(a.out), "b8_compat_1x.json", dict(results=res, note=(
        "full CubePipeline-vs-engine comparison on recorded RX taps runs once WS-T's "
        "recordings exist (I1/I2); the taps here cover K1, FFT and K3b")))
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

    def wait_window(self, d, t_first, n_rows, timeout_s=None):
        import numpy as np
        while self.write_pos(d) < t_first + n_rows:
            self.n_wait += 1
            time.sleep(0.0005)
        pool = self.pool[d]
        idx = np.mod(np.arange(t_first, t_first + n_rows), pool.shape[1])
        return pool[:, idx], np.ones((pool.shape[0], n_rows), bool)

    def counters(self):
        return dict(n_wait=self.n_wait)


def run_half(a, half: int) -> dict:
    import numpy as np

    from dsart.x2.plan import build_tiered_plan_for_opoint
    from dsart.x2.search.engine import EngineConfig, OwnerEngine
    from dsart.x2.search.geometry import owner_plan_from_tiered
    from dsart.x2.search.valuemap import dm_inputs_from_value_map

    op, vm = _value_map(a.opoint)
    plan = build_tiered_plan_for_opoint(op)
    oplan = owner_plan_from_tiered(plan, half=half)
    sm, sh = _steps(op)
    inp = dm_inputs_from_value_map(vm, s_main=sm, s_heavy=sh, bits_main=op.bits_main)
    inputs = {s.d: inp for s in oplan.dms}
    if a.input == "synthetic-ring":
        ring = ClockedRing(oplan, {s.d: inp.fmt for s in oplan.dms})
    else:
        try:
            from dsart.transport.recv_ring import open_ring_v4    # WS-T (frozen SPEC §6)
        except Exception as e:                                   # noqa: BLE001
            raise RuntimeError(f"rx-v4 consumer API not available: {e!r}")
        ring = open_ring_v4(a.shm, owned=[s.d for s in oplan.dms])
    cfg = EngineConfig(backend="gpu" if not a.cpu else "cpu", fft_mode=a.fft_mode,
                       families=("b3x",) if a.bank == "b3x" else ("b",),
                       n_grid=256, wait_timeout_s=30.0)
    eng = OwnerEngine(oplan, inputs, ring, cfg)
    t_end = time.monotonic() + 60.0 * a.minutes
    k = 2 if a.input == "synthetic-ring" else None
    rec = []
    while time.monotonic() < t_end:
        if k is None:
            k = int(min(ring.write_pos(s.d) // s.hop for s in oplan.dms)) - 1
        t_a = time.monotonic()
        r = eng.process_cycle(k)
        wall = 1e3 * (time.monotonic() - t_a)
        gms = eng._gpu.pop_gpu_ms() if eng._gpu is not None else {}
        lag = max((ring.write_pos(s.d) - (k + 1) * s.hop) * s.t_native * 32.768e-3
                  for s in oplan.dms)
        rec.append(dict(cycle=k, wall_ms=wall, gpu_ms=float(sum(gms.values())), lag_ms=lag,
                        n_cand=len(r.candidates)))
        k += 1
    gpu = np.array([x["gpu_ms"] for x in rec])
    wall = np.array([x["wall_ms"] for x in rec])
    lag = np.array([x["lag_ms"] for x in rec])
    duty = float(gpu.mean() / (1e3 * CYCLE_S)) if gpu.size else 0.0
    host = float(np.percentile(np.maximum(wall - gpu, 0), 99)) if wall.size else 0.0
    return dict(half=half, opoint=a.opoint, dms=oplan.summary()["dms"], n_cycles=len(rec),
                gpu_duty=duty, host_ms_p99=host, lag_ms_max=float(lag.max(initial=0)),
                lag_ms_last=float(lag[-1]) if lag.size else 0.0,
                status=dict(engine=eng.status()), cycles=rec[-50:])


def cmd_engine(a) -> int:
    out = Path(a.out)
    if a.half is not None:
        res = run_half(a, int(a.half))
        _write(out, f"b8_engine_half{a.half}.json", res)
        return EXIT_OK
    gpus = [g for g in a.gpus.split(",") if g != ""]
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), DSART_X2_CP=os.environ.get(
            "DSART_X2_CP", "b8"))
        cmd = [sys.executable, __file__, "engine", "--half", str(a.half_base + i), "--opoint",
               a.opoint, "--input", a.input, "--minutes", str(a.minutes), "--out", a.out,
               "--fft-mode", a.fft_mode, "--bank", a.bank] + (["--cpu"] if a.cpu else [])
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    res = []
    for i, _g in enumerate(gpus):
        p = out / f"b8_engine_half{a.half_base + i}.json"
        if not p.exists():
            res.append(dict(check="engine", half=a.half_base + i, status="fail", rc=rcs[i]))
            continue
        h = json.loads(p.read_text())
        gate = (h["gpu_duty"] <= 0.75 and h["host_ms_p99"] <= 30.0
                and h["lag_ms_last"] <= 2.0 * 1e3 * CYCLE_S)
        res.append(dict(check="engine", half=h["half"], gpu_duty=h["gpu_duty"],
                        duty_aim_met=h["gpu_duty"] <= 0.6, host_ms_p99=h["host_ms_p99"],
                        lag_ms_max=h["lag_ms_max"], status="ok" if gate else "fail"))
    _write(out, "b8_engine.json", dict(results=res))
    return _status(res)


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
    k.add_argument("--fft", default="cufft,warp")
    k.add_argument("--k3b", default="box,b3x")
    k.add_argument("--k1-formats", dest="k1_formats", default="all")
    k.add_argument("--check-1x-csr", dest="check_1x_csr", action="store_true")
    k.add_argument("--out", default="b8_out")
    x = sub.add_parser("compat-1x")
    x.add_argument("--out", default="b8_out")
    e = sub.add_parser("engine")
    e.add_argument("--gpus", default="0,1")
    e.add_argument("--input", default="synthetic-ring", choices=("synthetic-ring", "rx-v4"))
    e.add_argument("--opoint", default="2x-gridded-ns2-cint4")
    e.add_argument("--minutes", type=float, default=10.0)
    e.add_argument("--out", default="b8_out")
    e.add_argument("--half", default=None)
    e.add_argument("--half-base", dest="half_base", type=int, default=0)
    e.add_argument("--shm", default="")
    e.add_argument("--fft-mode", dest="fft_mode", default="warp")
    e.add_argument("--bank", default="b3x", choices=("b3x", "box"))
    e.add_argument("--cpu", action="store_true", help="CPU twins (plumbing dry run only)")
    sub.add_parser("teardown")
    sub.add_parser("_selftest-interp")
    a = ap.parse_args(argv)
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
