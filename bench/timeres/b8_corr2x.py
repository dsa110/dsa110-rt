#!/usr/bin/env python3
"""B8 - the corr track of CP1 (x2 WS-C); the command lines in
``tools/x2/cp_runner_plans.py`` are the contract (docs/x2/CP1_RUNBOOK.md):

  compile-check                          O4, CPU only (h23): nvcc -ptx of the CUDA stage-1 kernel for
                                         sm_75; Triton AOT compile (sm_75) of every Triton kernel the
                                         corr launches, when this interpreter has triton (h23's has
                                         not: then the node CPU tests carry it, test_corr2x_compile.py)
  profile-1x --out D                     today's 1x chain (production corr_fast args from
                                         configs/dsart_pipeline_rt.yaml, the node's cal blob and DM
                                         plan) on synthetic pages: per-stage StageTimer table and block
                                         wall time, legacy loop and the pipelined orchestrator
  golden-capture --blocks N --out D      the legacy loop over N deterministic pages (RFI lines,
                                         static-sky warm-up, an array-burst block, an injection block)
                                         -> per-block, per-stage hashes (golden.json)
  kernels --maps M --n-sub S [--check-ref] --out D
                                         2x GEMM (+ class reduce), CSR stage 1 per tier, static sky +
                                         row-rule packer, for every (map, n_sub); peak memory;
                                         exactness vs the CPU twins
  compat-1x --golden G --out D           PipelinedCorrFast over the same pages: every stage hash ==
                                         golden (torch.equal per stage, byte-equal v1 datagrams)
  live --opoint F --minutes M --tx drain --reader-identity corr_fast --out D
                                         corr_fast 2x on the live fada of this node (production
                                         corr_fast STOPPED by the runner; capture, merge, corr_slow and
                                         voltage_retention running), TX to a drain; block-time and
                                         buffer-overrun gates
  teardown                               remove /dev/shm/dsart-x2-* and /tmp/dsart-x2-*

Exit codes: 0 every gate passed, 1 a gate failed (named in the JSON), 2 error. JSON goes under
``--out``. Test shm / files are named ``dsart-x2-*``. Budget: 134.218 ms per block (design 57
nominal / 99 worst; escalation E2 at GEMM + stage 1 > 55 ms).

No module-level torch import: ``compile-check`` runs on h23's torch-free interpreter.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

EXIT_OK, EXIT_GATE, EXIT_ERR = 0, 1, 2
BLOCK_MS = 134.217728
CORR_NODES = (3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22)
MAP_NAMES = {"gridded": "gridded", "baselines": "baselines"}
PROD_CORR_FAST_RE = r"dsart[.]services[.]corr_fast_integration"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(out_dir: str | None, name: str, payload: dict) -> str | None:
    payload = dict(payload)
    payload.setdefault("host", socket.gethostname())
    payload.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    text = json.dumps(payload, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(text if len(text) < 4000 else text[:4000] + "\n...", flush=True)
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    return path


def _finish(out_dir, name, payload: dict, gates: dict) -> int:
    failed = sorted(k for k, ok in gates.items() if not ok)
    payload.update(gates=gates, failed_gates=failed, passed=not failed)
    _write(out_dir, name, payload)
    return EXIT_GATE if failed else EXIT_OK


def _digest(t) -> str:
    """blake2b-16 of a tensor/array's VALUES (-0.0 folded to +0.0, so
    ``torch.equal``-equal tensors hash equal), dtype and shape included."""
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.is_floating_point() or t.is_complex():
            t = t + 0
        a = t.contiguous().cpu().numpy()
    else:
        a = np.ascontiguousarray(t)
        if a.dtype.kind in "fc":
            a = a + 0
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{a.dtype}{a.shape}".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _cn_chgroup() -> tuple[int, int]:
    m = re.search(r"(\d+)", socket.gethostname().split(".")[0][-3:])
    cn = int(m.group(1)) if m else 0
    return cn, (CORR_NODES.index(cn) if cn in CORR_NODES else 0)


def _prod_routine() -> dict:
    import yaml
    with open(os.path.join(ROOT, "configs", "dsart_pipeline_rt.yaml")) as fh:
        doc = yaml.safe_load(fh)
    for r in doc["routines"]:
        if r.get("name") == "corr_fast":
            return r
    raise RuntimeError("corr_fast routine not found in dsart_pipeline_rt.yaml")


_VALUELESS = {"--rfi-persist", "--sliding-window", "--inject-watch", "--cal-pol-swap",
              "--rfi-disabled", "--sumthr-disabled", "--static-sky-disabled", "--pipeliner-3s",
              "--no-stage-timing"}


def _prod_argv(*, dec: float, drop: tuple = (), set_: dict | None = None) -> tuple[list[str], dict]:
    """Production corr_fast argv (the yaml routine, tokens substituted), with
    options in ``drop`` removed and ``set_`` options replaced/added."""
    r = _prod_routine()
    cn, chgroup = _cn_chgroup()
    toks = r["args"].split()
    toks = toks[toks.index("dsart.services.corr_fast_integration") + 1:]
    toks = [t.replace("CHGROUP", str(chgroup)).replace("CUSTOMDEC", str(dec)) for t in toks]
    toks = [t if t != "CN" else str(cn) for t in toks]
    opts: list[tuple[str, str | None]] = []
    i = 0
    while i < len(toks):
        k = toks[i]
        if k in _VALUELESS or i + 1 >= len(toks) or toks[i + 1].startswith("--"):
            opts.append((k, None))
            i += 1
        else:
            opts.append((k, toks[i + 1]))
            i += 2
    set_ = dict(set_ or {})
    out: list[str] = []
    for k, v in opts:
        if any(k == d or (d.endswith("*") and k.startswith(d[:-1])) for d in drop) or k in set_:
            continue
        out += [k] if v is None else [k, v]
    for k, v in set_.items():
        out += [k] if v is None else [k, str(v)]
    return out, dict(r.get("env") or {})


def _capture_cfg(argv: list[str]):
    """Run corr_fast's own CLI parsing with ``run`` stubbed: returns the
    exact (cfg, kwargs) production would start with."""
    import dsart.services.corr_fast_integration as cfi
    cap = {}
    real = cfi.run

    def stub(fada_key, output_dir, device, cfg, **kw):
        cap.update(fada_key=fada_key, output_dir=output_dir, device=device, cfg=cfg, kw=kw)
        return {}

    cfi.run = stub
    try:
        rc = cfi.main(argv)
    finally:
        cfi.run = real
    if rc != 0 or "cfg" not in cap:
        raise RuntimeError(f"corr_fast CLI rejected {argv} (rc={rc})")
    return cap


def _set_gpu_env(env: dict) -> None:
    """Production's GPU pinning (CUDA_VISIBLE_DEVICES=1, device cuda:0)
    unless the caller pinned one already; before torch is imported."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ and "CUDA_VISIBLE_DEVICES" in env:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env["CUDA_VISIBLE_DEVICES"])


# ---------------------------------------------------------------------------
# compile-check (O4, CPU)
# ---------------------------------------------------------------------------


def _triton_specs():
    """(module, kernel, signature, [constants...]) of every Triton kernel the
    corr launches in 1x-compat and 2x."""
    i32, f16, f32 = "i32", "*fp16", "*fp32"
    g = dict(re_ptr=f16, im_ptr=f16, calr_ptr=f32, cali_ptr=f32, tmask_ptr=f32, core_ptr="*i32",
             dst_ptr="*i32", tp_ptr="*i32", ring_ptr=f16, scr_ptr=f32, fv0=i32, n_core=i32,
             n_ring_rows=i32, t_ring0=i32, T_ring=i32, n_fv_scr=i32, pk_per_acc=i32, nants=i32,
             nchan=i32, s_ch=i32, s_t=i32, s_p=i32, s_pk=i32, s_ant=i32)
    gc = dict(NSUM=192, NVP=2, CSF=2, PPFV=4, NC_PAD=96, BLOCK=16)
    r = dict(scr_ptr=f32, ring_ptr=f16, cptr_ptr="*i32", csrc_ptr="*i32", cconj_ptr="*u8",
             crow_ptr="*i32", t_ring0=i32, T_ring=i32, n_fv=i32)
    p = dict(codes_ptr="*i8", out_ptr="*u8", n_comp=i32, n_bytes=i32)
    c1 = dict(R_ptr=f16, I_ptr=f16, out_re_ptr=f32, out_im_ptr=f32, n_fv=i32, A_stride_b=i32,
              A_stride_k=i32, A_stride_n=i32, OUT_stride_fv=i32, OUT_stride_bls=i32, OUT_stride_c=i32)
    c1c = dict(NCHAN=384, NCHAN_EFF=48, NANTS=96, NBASE=4656, NVP=2, CSF=8, K=32, BLOCK_M=16, BLOCK_N=16)
    d = dict(vis_re_ptr=f32, vis_im_ptr=f32, shifts_ptr="*i32", csr_offs_ptr="*i32", csr_b_ptr="*i32",
             csr_c_ptr="*i32", out_re_ptr=f32, out_im_ptr=f32, n_dm=i32, t_dedisp=i32, n_filled=i32,
             vis_stride_b=i32, vis_stride_c=i32, vis_stride_t=i32, sh_stride_c=i32, sh_stride_dm=i32,
             out_stride_dm=i32, out_stride_t=i32, out_stride_g=i32)
    return [
        ("dsart.x2.corr.gemm2x", "_gemm2x_kernel", g, [dict(gc, HAS_TMASK=False), dict(gc, HAS_TMASK=True)]),
        ("dsart.x2.corr.gemm2x", "_reduce_kernel", r, [dict(NSUM=192, BLOCK_T=64)]),
        ("dsart.x2.corr.packer2x", "_pack_bits_kernel", p, [dict(B=b, BLOCK=256) for b in (4, 5, 6)]),
        ("dsart.services.triton_corr_fused", "_fused_corr_post_no_cat_kernel", c1, [c1c]),
        ("dsart.services.triton_dedisp", "_fused_dedisp_kernel", d,
         [dict(NCHAN_EFF=48, NBASE=4656, BLOCK_T=128)]),
    ]


def compile_triton_kernels(arch: int = 75) -> dict:
    """AOT-compile every corr Triton kernel for ``sm_<arch>`` (no GPU needed)."""
    import importlib
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    res = {}
    for modname, kname, sig, consts in _triton_specs():
        fn = getattr(importlib.import_module(modname), kname)
        for c in consts:
            key = f"{kname}{json.dumps(c, sort_keys=True)}"
            try:
                k = triton.compile(ASTSource(fn=fn, signature=sig, constants=c),
                                   target=GPUTarget("cuda", arch, 32))
                res[key] = {"ok": True, "cubin_bytes": len(k.asm.get("cubin", b""))}
            except Exception as exc:  # noqa: BLE001
                res[key] = {"ok": False, "error": repr(exc)[:400]}
    return res


def cmd_compile_check(args) -> int:
    from dsart.x2.corr.stage1_cuda import CUDA_SOURCE, compile_check_nvcc
    out: dict = {"cuda": {}, "triton": {}}
    gates: dict = {}
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if os.path.exists(nvcc):
        try:
            ptx = compile_check_nvcc(CUDA_SOURCE, nvcc=nvcc, arch="sm_75")
            out["cuda"]["stage1_tier"] = {"ok": ".entry stage1_tier" in ptx, "ptx_bytes": len(ptx)}
        except Exception as exc:  # noqa: BLE001
            out["cuda"]["stage1_tier"] = {"ok": False, "error": str(exc)[-2000:]}
        gates["cuda_compiles"] = bool(out["cuda"]["stage1_tier"]["ok"])
    else:
        out["cuda"]["stage1_tier"] = {"ok": False, "error": f"no nvcc ({nvcc})"}
        gates["cuda_compiles"] = False
    import importlib.util
    missing = [m for m in ("triton", "torch") if importlib.util.find_spec(m) is None]
    have_triton = not missing
    if missing:
        out["triton"] = {"deferred": f"{sys.executable} has no {'/'.join(missing)}; the AOT sm_75 "
                                     "compile runs in tests/x2/test_corr2x_compile.py (pre.cpu_tests)"}
    if have_triton:
        res = compile_triton_kernels()
        out["triton"] = res
        gates["triton_compiles"] = all(v["ok"] for v in res.values())
    return _finish(args.out, "compile_check.json", out, gates)


# ---------------------------------------------------------------------------
# synthetic production-like pages (profile / golden / compat)
# ---------------------------------------------------------------------------


class Pages:
    """Deterministic fada pages: 8 base pages of int4 noise (sum of two
    uniform [-3, 3] draws per component), cycled, plus events by block:
    narrowband RFI (blocks 20-27, channels 100-103 at full scale on every
    antenna), an array-common burst (block 40, packets 600-663 x3 on every
    antenna and channel) and an injection (block 50, via OnlineInjector)."""

    NPKT, NANTS, NCHAN = 2048, 96, 384
    RFI_BLOCKS = range(20, 28)
    BURST_BLOCK = 40
    INJECT_BLOCK = 50

    def __init__(self, seed: int = 20261022):
        self.seed = seed
        self._base: dict[int, np.ndarray] = {}

    def _components(self, j: int):
        rng = np.random.default_rng(self.seed + j)
        shape = (self.NPKT, self.NANTS, self.NCHAN, 2, 2)
        re = rng.integers(-3, 4, size=shape, dtype=np.int8) + rng.integers(-3, 4, size=shape, dtype=np.int8)
        im = rng.integers(-3, 4, size=shape, dtype=np.int8) + rng.integers(-3, 4, size=shape, dtype=np.int8)
        return re, im

    @staticmethod
    def _pack(re, im) -> np.ndarray:
        return ((re.astype(np.uint8) & 0x0F) | ((im.astype(np.uint8) & 0x0F) << 4)).reshape(-1)

    def page(self, block: int) -> np.ndarray:
        j = block % 8
        if block in self.RFI_BLOCKS or block == self.BURST_BLOCK:
            re, im = self._components(j)
            if block in self.RFI_BLOCKS:
                re[:, :, 100:104] = 7
                im[:, :, 100:104] = -7
            if block == self.BURST_BLOCK:
                sl = slice(600, 664)
                re[sl] = np.clip(re[sl].astype(np.int16) * 3, -7, 7).astype(np.int8)
                im[sl] = np.clip(im[sl].astype(np.int16) * 3, -7, 7).astype(np.int8)
            return self._pack(re, im)
        if j not in self._base:
            self._base[j] = self._pack(*self._components(j))
        return self._base[j]

    def events(self) -> dict:
        return {"rfi_blocks": list(self.RFI_BLOCKS), "burst_block": self.BURST_BLOCK,
                "inject_block": self.INJECT_BLOCK, "seed": self.seed}


def _build_1x_ctx(dev, *, dec: float, recorder, inject: bool):
    """Production 1x context (corr_fast's own CLI parse of the yaml args) with
    a recording TX and today's per-coarse-DM stage-2 FIFO."""
    import dsart.services.corr_fast_integration as cfi
    from dsart.common.constants import NATIVE_SAMPLE_US
    argv, env = _prod_argv(dec=dec, drop=("--transport-tx-*", "--sky-export-*", "--rfi-mon-*",
                                          "--inject-watch", "--ready-sentinel-path", "--output-dir",
                                          "--max-blocks"))
    cap = _capture_cfg(argv)
    cfg = cap["cfg"]
    if inject:
        from dsart.inject.online import InjectionConfig
        cfg.inject_configs = (InjectionConfig(
            inj_id="x2_cp1_golden", dm_pc_cm3=400.0, l_rad=0.002, m_rad=-0.001,
            fluence_jy_ms=2.0e4, width_samples=32, profile="gaussian",
            apply_at_specnum=Pages.INJECT_BLOCK * Pages.NPKT),)
    e, n, core = cfi.load_antpos_from_cal_blob(cfg.cal_path)
    st = cfi.load_station_numbers_for_cal(cfg.cal_path)
    fifo = cfi._Stage2InterChgroupShiftAdapter(
        chgroup=int(cfg.chgroup), coarse_dm_pc_cm3=_plan_dm(cap, cfg),
        t_int_corr_us=float(cfg.t_int_fast_native * NATIVE_SAMPLE_US))
    ctx = cfi.build_context(cfg, device=dev, antpos_e=e, antpos_n=n, is_core_baseline_mask=core,
                            station_numbers=st, stage2_fifo=fifo, transport_tx=recorder,
                            dm_plan=cap["kw"].get("dm_plan"))
    ctx.stage_timer = cfi.StageTimer(device=dev)
    return ctx, cfg, argv


def _plan_dm(cap, cfg) -> np.ndarray:
    plan = cap["kw"].get("dm_plan")
    if plan is None:
        from dsart.coarse_dm.dm_plan import load_dm_plan
        plan = load_dm_plan(str(cfg.dm_plan_path))
    return np.asarray(plan.dm_pc_cc, dtype=np.float64)


class _TapTx:
    """Recording TX: hashes every cube and its v1 cint8 payload per stream."""

    def __init__(self, taps: dict, streams):
        self.taps, self.streams = taps, streams

    def transmit(self, cubes_for_tx, *, block_n, rfi_warming_up):
        from dsart.x2.corr.packer2x import pack_v1
        for i, c in enumerate(cubes_for_tx):
            self.taps.setdefault(block_n, {})[f"fifo{i}"] = _digest(c)
            for s, (lo, hi) in enumerate(self.streams or [(0, c.shape[-1])]):
                q, sc = pack_v1(c[..., lo:hi])
                self.taps[block_n][f"payload{i}_s{s}"] = _digest(q) + _digest(sc)
        return len(cubes_for_tx)


class _Delegate:
    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, k):
        return getattr(self._inner, k)


def _install_taps(ctx, taps: dict, orch=None):
    """Per-block hashes at the same stage boundaries in both loops: unpack,
    RFI mask, post-cal voltages, fast vis (B, C, T), the stage-2 input (after
    static sky); TX cubes + v1 payloads come from :class:`_TapTx`."""
    import dsart.services.corr_fast_integration as cfi
    import dsart.x2.corr.pipeline1x as p1
    cur = {"b": None}

    def put(name, t):
        taps.setdefault(cur["b"], {})[name] = _digest(t)

    real_cal = cfi.apply_cal_split

    def cal(*a):
        r, i = real_cal(*a)
        put("cal_re", r)
        put("cal_im", i)
        return r, i

    cfi.apply_cal_split = cal
    p1.apply_cal_split = cal
    if ctx.rfi_flagger is not None:
        class _Flag(_Delegate):
            def flag_block(self, re, im, **kw):
                put("unpack_re", re)
                res = self._inner.flag_block(re, im, **kw)
                put("rfi_mask", res.mask)
                return res
        ctx.rfi_flagger = _Flag(ctx.rfi_flagger)

    class _Fifo(_Delegate):
        def push(self, x, *, block_n):
            cur["b"] = block_n
            put("sky", x)
            return self._inner.push(x, block_n=block_n)
    ctx.stage2_fifo = _Fifo(ctx.stage2_fifo)
    m = ctx.multi_dm_coarse_dm
    real_dd = m.dedisperse_from_vis

    def dd(vis, *, block_n):
        cur["b"] = block_n
        put("vis_re", vis.real.permute(1, 2, 0))
        put("vis_im", vis.imag.permute(1, 2, 0))
        return real_dd(vis, block_n=block_n)
    m.dedisperse_from_vis = dd
    if orch is not None:
        real_issue = orch._issue

        def issue(st):
            cur["b"] = st.block_n                 # page k issues block k-1
            return real_issue(st)
        orch._issue = issue

        def tap(name, block_n, re, im):
            cur["b"] = block_n
            put(name + "_re", re)
            put(name + "_im", im)
        orch.tap = tap
    return cur


def _run_blocks(mode: str, n_blocks: int, dev, dec: float, taps: dict | None):
    """Legacy loop or PipelinedCorrFast over the Pages; returns timings."""
    import torch
    import dsart.services.corr_fast_integration as cfi
    from dsart.x2.corr.pipeline1x import PipelinedCorrFast
    pages = Pages()
    streams = None
    recorder = _TapTx(taps if taps is not None else {}, None)
    ctx, cfg, argv = _build_1x_ctx(dev, dec=dec, recorder=recorder, inject=True)
    lay = ctx.multi_dm_coarse_dm.subband_layout
    if lay is not None:
        streams = [(int(lay.offsets[s]), int(lay.offsets[s + 1])) for s in range(lay.n_sub)]
    recorder.streams = streams
    orch = PipelinedCorrFast(ctx) if mode == "pipelined" else None
    if orch is not None:
        orch.warmup(4)
    else:
        cfi._warmup_pipeline_jit(ctx, expected_fada_bytes=pages.page(0).size, n_blocks=4)
    cur = _install_taps(ctx, taps, orch) if taps is not None else None
    walls = []
    for b in range(1, n_blocks + 1):
        if cur is not None and orch is None:
            cur["b"] = b
        t0 = time.perf_counter()
        if orch is None:
            cfi.process_block(pages.page(b), ctx=ctx, block_n=b)
        else:
            orch.push(pages.page(b), block_n=b)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    if orch is not None:
        orch.flush()
    ctx.stage_timer.poll()
    w = sorted(walls[4:]) or sorted(walls)
    return {"mode": mode, "n_blocks": n_blocks, "argv": argv,
            "wall_ms_mean": float(np.mean(w)), "wall_ms_p50": float(w[len(w) // 2]),
            "wall_ms_max": float(w[-1]), "stage_ms": ctx.stage_timer.summary()["stage_ms"]}


def cmd_profile_1x(args) -> int:
    _set_gpu_env(_prod_routine().get("env") or {})
    import torch
    dev = torch.device(args.device)
    out = {"legacy": _run_blocks("legacy", args.blocks, dev, args.dec, None)}
    torch.cuda.empty_cache()
    out["pipelined"] = _run_blocks("pipelined", args.blocks, dev, args.dec, None)
    return _finish(args.out, "profile_1x.json", out, {})


def cmd_golden_capture(args) -> int:
    _set_gpu_env(_prod_routine().get("env") or {})
    import torch
    taps: dict = {}
    res = _run_blocks("legacy", args.blocks, torch.device(args.device), args.dec, taps)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "golden.json"), "w") as fh:
        json.dump({"taps": {str(k): v for k, v in taps.items() if k is not None},
                   "pages": Pages().events(), "n_blocks": args.blocks, "dec": args.dec,
                   "argv": res["argv"]}, fh)
    gates = {"blocks_ge_64": args.blocks >= 64,
             "every_block_tapped": all(str(b) in {str(k) for k in taps} for b in range(1, args.blocks + 1))}
    return _finish(args.out, "golden_capture.json", res, gates)


def cmd_compat_1x(args) -> int:
    _set_gpu_env(_prod_routine().get("env") or {})
    import torch
    with open(os.path.join(args.golden, "golden.json")) as fh:
        gold = json.load(fh)
    taps: dict = {}
    res = _run_blocks("pipelined", int(gold["n_blocks"]), torch.device(args.device),
                      float(gold["dec"]), taps)
    mism = []
    n_cmp = 0
    for b, stages in gold["taps"].items():
        mine = taps.get(int(b), {})
        for k, v in stages.items():
            n_cmp += 1
            if mine.get(k) != v:
                mism.append(f"block {b} {k}")
    res.update(n_compared=n_cmp, mismatches=mism[:200], n_mismatches=len(mism))
    gates = {"stages_equal": not mism, "blocks_ge_64": int(gold["n_blocks"]) >= 64, "compared": n_cmp > 0}
    return _finish(args.out, "compat_1x.json", res, gates)


# ---------------------------------------------------------------------------
# kernels (2x)
# ---------------------------------------------------------------------------


def _antpos_for(args):
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    if args.antpos == "fixture":
        z = np.load(os.path.join(ROOT, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
        return (z["antpos_e"].astype(np.float32), z["antpos_n"].astype(np.float32),
                core_baseline_mask_from_station_numbers(z["station"]))
    import dsart.services.corr_fast_integration as cfi
    return cfi.load_antpos_from_cal_blob(args.antpos)


def _setup_2x(fmt, antpos, chgroup):
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.x2.corr.plan2x import CorrPlan2x
    from dsart.x2.corr.stage1_cuda import geometry_for, stage1_csr
    from dsart.x2.opoint import opoint_from_id
    from dsart.x2.plan import build_tiered_plan_for_opoint
    e, n, mask = antpos
    op = opoint_from_id(fmt)
    vm = build_value_map_for_opoint(op, e, n, is_core_baseline_mask=mask, dec_deg=71.63)
    plan = CorrPlan2x.from_tiered_plan(build_tiered_plan_for_opoint(op))
    csr = stage1_csr(vm, chgroup, ring_rows="classes" if op.value_kind == "baselines" else "baselines")
    geos = []
    for ti, t in enumerate(plan.tiers):
        sh = np.ascontiguousarray(plan.stage1_bins(ti)[chgroup].T.astype(np.int32))
        geos.append((t, sh, geometry_for(sh, D=t.decim)))
    return op, vm, plan, csr, geos, mask


def describe(fmt: str, antpos, chgroup: int) -> dict:
    """Shapes, memory and launch geometry of one format (CPU, no timing)."""
    from dsart.x2.corr.gemm2x import build_gemm_tables
    op, vm, plan, csr, geos, mask = _setup_2x(fmt, antpos, chgroup)
    tab = build_gemm_tables(mask, classes=vm.classes if op.value_kind == "baselines" else None)
    nsum = vm.nchan_eff
    out = dict(fmt=fmt, chgroup=chgroup, values=int(csr.n_values), sources=int(csr.row.size),
               ring_rows=tab.n_ring_rows, scratch_rows=tab.n_scratch_rows,
               ring_gib=round(tab.n_ring_rows * nsum * 1024 * 4 / 2 ** 30, 3),
               tile_pairs=int(tab.tile_pairs.shape[0]), plan_id=f"0x{plan.plan_id:016x}",
               pattern_id=f"0x{vm.pattern_id:016x}", tiers=[])
    for t, sh, g in geos:
        n_out = 512 // t.decim
        out["tiers"].append(dict(
            tier=t.name, n_d=t.n_coarse, D=t.decim, TT=g.TT, W=g.W, chunk=g.chunk,
            spread=g.spread, smem_kib=round(g.smem_bytes / 1024, 1), threads=g.threads,
            grid=[int(csr.n_values), -(-n_out // g.TT)],
            out_gib=round(t.n_coarse * n_out * csr.n_values * 8 / 2 ** 30, 3),
            smem_reads_G=round(int(csr.row.size) * t.n_coarse * 512 / 1e9, 2)))
    return out


def _kernels_one(fmt, antpos, chgroup, dev, reps, check_ref):
    import torch
    from common import cuda_time_ms
    from dsart.services.slow_corr_kernel import unpack_int4_split
    from dsart.x2.corr.gemm2x import (build_gemm_tables, gemm2x_into_ring, gemm2x_reference,
                                      reduce_classes, ring_write_reference)
    from dsart.x2.corr.packer2x import PackerRule, RowRulePacker
    from dsart.x2.corr.stage1_cuda import Stage1Csr, stage1_tier_cuda, stage1_twin
    from dsart.x2.corr.static_sky2x import BatchedStaticSky
    op, vm, plan, csr, geos, mask = _setup_2x(fmt, antpos, chgroup)
    classes = op.value_kind == "baselines"
    tab = build_gemm_tables(mask, classes=vm.classes if classes else None).to(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    raw = Pages().page(1)
    re, im = unpack_int4_split(raw, device=dev)
    rng = np.random.default_rng(3)
    ph = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=(384, 2, 96)).astype(np.float32)).to(dev)
    cr, ci = torch.cos(ph).half().float(), torch.sin(ph).half().float()
    nsum = vm.nchan_eff
    ring = torch.zeros((tab.n_ring_rows, nsum, 1024, 2), dtype=torch.float16, device=dev)
    chunk = 128
    scr = torch.empty((tab.n_scratch_rows, nsum, chunk, 2), device=dev) if classes else None

    def gemm(q0=512):
        for f0 in range(0, 512, chunk):
            gemm2x_into_ring(re, im, cr, ci, tab, ring, fv0=f0, n_fv=chunk, t_ring0=q0 + f0,
                             csf=2, t_int=8, scratch=scr)
            if scr is not None:
                reduce_classes(scr, ring, tab, t_ring0=q0 + f0, n_fv=chunk)

    res = {"fmt": fmt, "gemm_ms": cuda_time_ms(gemm, reps=reps)}
    gemm(0)
    gemm(512)                                     # two blocks in the ring for stage 1
    gates = {}
    if check_ref:
        want = gemm2x_reference(re, im, cr, ci, tab, fv0=0, n_fv=16, csf=2, t_int=8)
        if classes:
            want = ring_write_reference(want, tab, vm.classes.bl_class, vm.classes.bl_conj)
        got = torch.complex(ring[:, :, 512:528, 0].float(), ring[:, :, 512:528, 1].float()).permute(2, 1, 0)
        w16 = torch.complex(want.real.half().float(), want.imag.half().float())
        tol = 2 * torch.finfo(torch.float16).eps * w16.abs() + 1e-6
        gates["gemm_within_1ulp"] = bool(((got - w16).abs() <= 1.5 * tol).all())
    csr_dev = {k: torch.from_numpy(np.ascontiguousarray(getattr(csr, k))).to(dev)
               for k in ("ptr", "row", "k", "conj")}
    n_heavy = [int(vm.streams[chgroup * vm.n_sub + s].n_heavy) for s in range(vm.n_sub)]
    t1 = t2 = 0.0
    ring_np = ring.cpu().numpy() if check_ref else None
    for t, sh, g in geos:
        n_out = 512 // t.decim
        out = torch.empty((t.n_coarse, n_out, csr.n_values, 2), device=dev)
        shd, smd = torch.from_numpy(sh).to(dev), torch.from_numpy(g.smin).to(dev)
        ms = cuda_time_ms(lambda: stage1_tier_cuda(ring, csr_dev, shd, smd, out, g, q_base=0,
                                                   n_out=n_out), reps=reps)
        res[f"stage1_{t.name}_ms"] = ms
        t1 += ms
        x = torch.view_as_complex(out)
        rule = PackerRule(comp_bits=op.bits_main // 2)
        ss = BatchedStaticSky(n_d=t.n_coarse, n_values=csr.n_values, device=dev)
        pk = RowRulePacker(n_d=t.n_coarse, value_offsets=csr.value_offsets, n_contrib=csr.n_contrib,
                           n_heavy=n_heavy, rule=rule, device=dev)
        ms2 = cuda_time_ms(lambda: pk.pack(ss.apply_(x.clone())), reps=reps)
        res[f"skypack_{t.name}_ms"] = ms2
        t2 += ms2
        if check_ref:
            nv = 6
            sub = Stage1Csr(ptr=csr.ptr[:nv + 1], row=csr.row[:csr.ptr[nv]], k=csr.k[:csr.ptr[nv]],
                            conj=csr.conj[:csr.ptr[nv]], value_offsets=np.array([0, nv]),
                            n_contrib=csr.n_contrib[:nv])
            twin = stage1_twin(ring_np, sub, sh, g, q_base=0, n_out=n_out)
            gates[f"stage1_{t.name}_bit_exact"] = bool(np.array_equal(
                x[:, :, :nv].cpu().numpy(), twin.astype(np.complex64)))
            pc = RowRulePacker(n_d=t.n_coarse, value_offsets=csr.value_offsets,
                               n_contrib=csr.n_contrib, n_heavy=n_heavy, rule=rule)
            a = pc.pack(x.cpu())
            b = RowRulePacker(n_d=t.n_coarse, value_offsets=csr.value_offsets, n_contrib=csr.n_contrib,
                              n_heavy=n_heavy, rule=rule, device=dev).pack(x)
            gates[f"packer_{t.name}_gpu_eq_cpu"] = all(torch.equal(ra["payload"], rb["payload"].cpu())
                                                       for ra, rb in zip(a, b))
    res.update(stage1_ms=t1, skypack_ms=t2, gemm_plus_stage1_ms=res["gemm_ms"] + t1,
               corr_2x_kernels_ms=res["gemm_ms"] + t1 + t2,
               frac_of_block=(res["gemm_ms"] + t1 + t2) / BLOCK_MS,
               peak_gib=torch.cuda.max_memory_allocated(dev) / 2 ** 30)
    gates["gemm_plus_stage1_le_55ms"] = res["gemm_plus_stage1_ms"] <= 55.0
    return res, gates


def cmd_kernels(args) -> int:
    _set_gpu_env(_prod_routine().get("env") or {})
    import torch
    dev = torch.device(args.device)
    antpos = _antpos_for(args)
    _, chgroup = _cn_chgroup()
    chgroup = args.chgroup if args.chgroup is not None else chgroup
    rows, gates = [], {}
    for kind in args.maps.split(","):
        for ns in args.n_sub.split(","):
            fmt = f"2x-{MAP_NAMES[kind]}-ns{int(ns)}-cint{args.cint}"
            try:
                r, g = _kernels_one(fmt, antpos, chgroup, dev, args.reps, args.check_ref)
                r["describe"] = describe(fmt, antpos, chgroup)
                rows.append(r)
                gates.update({f"{fmt}:{k}": v for k, v in g.items()})
            except torch.cuda.OutOfMemoryError:
                rows.append({"fmt": fmt, "error": "OOM"})
                gates[f"{fmt}:fits_memory"] = False
            torch.cuda.empty_cache()
    return _finish(args.out, "kernels_2x.json", {"chgroup": chgroup, "results": rows}, gates)


# ---------------------------------------------------------------------------
# live 2x on the node's fada
# ---------------------------------------------------------------------------


def _dbmetric(key: str) -> dict:
    for b in ("dada_dbmetric", "/usr/local/bin/dada_dbmetric"):
        try:
            p = subprocess.run([b, "-k", key], capture_output=True, text=True, timeout=2.0)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        for line in (p.stdout + p.stderr).splitlines():
            s = line.strip()
            if not s or s.startswith(("ipc_alloc:", "ipcsync_get:", "ipcbuf_connect:")):
                continue
            kv = dict(re.findall(r"(\w+)=(\d+)", s))
            if "nbufs" in kv and "nfull" in kv:
                return {"nbufs": int(kv["nbufs"]), "nfull": int(kv["nfull"])}
            parts = [x for x in re.split(r"[,\s]+", s) if x]
            if len(parts) >= 2 and all(x.isdigit() for x in parts[:2]):
                return {"nbufs": int(parts[0]), "nfull": int(parts[1])}
    return {}


def _production_corr_fast_pids() -> list[int]:
    """corr_fast processes without this run's DSART_X2_CP marker."""
    me = os.environ.get("DSART_X2_CP")
    out = subprocess.run(["pgrep", "-f", PROD_CORR_FAST_RE], capture_output=True, text=True).stdout
    pids = []
    for p in out.split():
        if int(p) == os.getpid():
            continue
        try:
            env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        if me and f"DSART_X2_CP={me}".encode() in env:
            continue
        pids.append(int(p))
    return pids


def cmd_live(args) -> int:
    if args.reader_identity != "corr_fast" or args.tx != "drain":
        print("live: only --tx drain --reader-identity corr_fast are supported", file=sys.stderr)
        return EXIT_ERR
    prod = _production_corr_fast_pids()
    if prod:
        print(f"live: production corr_fast still running (pids {prod}); refusing", file=sys.stderr)
        return EXIT_ERR
    from dsart.x2.opoint import opoint_from_id
    op = opoint_from_id(args.opoint)
    n_blocks = int(args.minutes * 60e3 / BLOCK_MS)
    out_blocks = os.path.join(args.out, "live_blocks")
    argv, env = _prod_argv(
        dec=args.dec,
        drop=("--transport-tx-*", "--sky-export-*", "--rfi-mon-*", "--inject-watch", "--stage2-mode",
              "--dm-plan-path", "--ready-sentinel-path", "--output-dir", "--max-blocks"),
        set_={"--t-int-fast-native": op.t_int_fast_native, "--chan-sum-factor": op.chan_sum_factor,
              "--n-sub": op.n_sub, "--opoint": op.opoint_id, "--max-blocks": n_blocks,
              "--output-dir": out_blocks, "--blocks-output-mode": "none",
              "--ready-sentinel-path": "/tmp/dsart-x2-corr-fast-live.ready"})
    _set_gpu_env(env)
    import dsart.services.corr_fast_integration as cfi
    samples: dict = {"fada": [], "eada": [], "dada": []}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            for k in samples:
                m = _dbmetric(k)
                if m:
                    samples[k].append((time.time(), m["nfull"], m["nbufs"]))
            stop.wait(2.0)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    cap = {}
    real = cfi.run

    def run_and_keep(*a, **kw):
        cap["summary"] = real(*a, **kw)
        return cap["summary"]

    cfi.run = run_and_keep
    try:
        rc = cfi.main(argv)
    finally:
        cfi.run = real
        stop.set()
        th.join(timeout=5)
    s = cap.get("summary") or {}
    over = {k: (max((nf for _, nf, _ in v), default=None), (v[0][2] if v else None)) for k, v in samples.items()}
    res = {"opoint": op.opoint_id, "argv": argv, "rc": rc, "summary": s,
           "buffers_max_full": {k: {"max_nfull": a, "nbufs": b} for k, (a, b) in over.items()},
           "n_buffer_samples": {k: len(v) for k, v in samples.items()}}
    if rc != 0 or not s:
        _write(args.out, "live_2x.json", res)
        return EXIT_ERR
    mean, p99 = float(s.get("ms_per_block_mean", np.nan)), float(s.get("ms_per_block_p99", np.nan))
    gates = {
        "mean_le_85pct": mean <= 0.85 * BLOCK_MS,
        "p99_le_100pct": p99 <= BLOCK_MS,
        "no_dropped_blocks": int(s.get("n_dropped", 1)) == 0,
        "ran_requested_blocks": int(s.get("n_blocks_in", 0)) >= n_blocks,
    }
    for k in ("fada", "eada"):
        mx, nb = over[k]
        gates[f"{k}_never_full"] = mx is not None and nb is not None and mx < nb
    return _finish(args.out, "live_2x.json", res, gates)


def cmd_teardown(args) -> int:
    removed = []
    for pat in ("/dev/shm/dsart-x2-*", "/tmp/dsart-x2-*"):
        for p in glob.glob(pat):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.unlink(p)
                removed.append(p)
            except OSError:
                pass
    _write(getattr(args, "out", None), "teardown.json", {"removed": removed})
    return EXIT_OK


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gpu=True):
        p.add_argument("--out", default=None)
        if gpu:
            p.add_argument("--device", default="cuda:0",
                           help="after production's CUDA_VISIBLE_DEVICES pin (physical GPU 1)")
            p.add_argument("--dec", type=float, default=71.63, help="pointing DEC (deg)")
        return p

    common(sub.add_parser("compile-check"), gpu=False)
    p = common(sub.add_parser("profile-1x"))
    p.add_argument("--blocks", type=int, default=64)
    p = common(sub.add_parser("golden-capture"))
    p.add_argument("--blocks", type=int, default=64)
    p = common(sub.add_parser("kernels"))
    p.add_argument("--maps", default="gridded,baselines")
    p.add_argument("--n-sub", default="2,4")
    p.add_argument("--cint", type=int, default=4)
    p.add_argument("--check-ref", action="store_true")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--antpos", default="fixture", help="'fixture' or a cal blob path")
    p.add_argument("--chgroup", type=int, default=None)
    p = common(sub.add_parser("compat-1x"))
    p.add_argument("--golden", required=True)
    p = common(sub.add_parser("live"))
    p.add_argument("--opoint", required=True)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--tx", default="drain")
    p.add_argument("--reader-identity", default="corr_fast")
    common(sub.add_parser("teardown"), gpu=False)
    args = ap.parse_args(argv)
    fn = {"compile-check": cmd_compile_check, "profile-1x": cmd_profile_1x,
          "golden-capture": cmd_golden_capture, "kernels": cmd_kernels, "compat-1x": cmd_compat_1x,
          "live": cmd_live, "teardown": cmd_teardown}[args.cmd]
    if args.cmd not in ("compile-check", "teardown"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # bench/timeres/common.py
    try:
        return fn(args)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        _write(getattr(args, "out", None), f"{args.cmd}_error.json",
               {"error": traceback.format_exc()[-4000:]})
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
