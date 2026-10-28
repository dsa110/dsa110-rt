"""O4 compile-only checks of every corr kernel, on the CPU (no GPU):

* the CUDA stage-1 kernel and the tensor-core GEMM (``mma.sync``) through
  ``nvcc -ptx -arch=sm_75``;
* every Triton kernel the corr launches (2x GEMM with/without the time
  mask, class reduce, bit packer at 4/5/6 bits, the row-rule row-stats and
  codes kernels, and today's fused GEMM and stage-1 kernels) AOT-compiled to
  an sm_75 cubin; the row-rule kernels without any FMA in their PTX and with
  the magic-constant rint intact;
* the CP1 bench's ``compile-check`` subcommand end to end (exit code and
  JSON), as the runner calls it. (``teardown`` is not exercised here: it
  removes every ``dsart-x2-*`` file on the host, other workstreams' too.)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH = os.path.join(ROOT, "bench", "timeres", "b8_corr2x.py")
_NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def test_triton_kernels_compile_for_sm75() -> None:
    pytest.importorskip("triton")
    sys.path.insert(0, os.path.dirname(BENCH))
    import b8_corr2x
    res = b8_corr2x.compile_triton_kernels(75)
    bad = {k: v for k, v in res.items() if not v["ok"]}
    assert not bad, bad
    assert len(res) == 12 and all(v["cubin_bytes"] > 0 for v in res.values())
    # the row-rule kernels compile without FMA contraction, and the codes
    # kernel keeps its (q + 1.5*2^23) - 1.5*2^23 rint (not folded away)
    nofma = {k: v for k, v in res.items() if not v["fp_fusion"]}
    assert len(nofma) == 4 and all(v["ptx_fma"] == 0 for v in nofma.values()), nofma
    for k, v in nofma.items():
        if k.startswith("_codes_kernel"):
            ptx = v["ptx"].upper()
            assert "0F4B400000" in ptx and "FMA." not in ptx, k


@pytest.mark.skipif(not os.path.exists(_NVCC), reason="no nvcc on this host")
def test_bench_compile_check_cli(tmp_path) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, BENCH, "compile-check", "--out", str(tmp_path)],
                       capture_output=True, text=True, env=env, timeout=900)
    out = json.loads((tmp_path / "compile_check.json").read_text())
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert out["passed"] and out["gates"]["cuda_compiles"]
    assert out["gates"].get("triton_compiles", True)


@pytest.mark.skipif(not os.path.exists(_NVCC), reason="no nvcc on this host")
def test_cuda_gemm_compiles_sm75() -> None:
    from dsart.x2.corr.gemm2x_cuda import CUDA_SOURCE
    from dsart.x2.corr.stage1_cuda import compile_check_nvcc
    ptx = compile_check_nvcc(CUDA_SOURCE, nvcc=_NVCC)
    assert ".entry gemm2x_mma" in ptx and ".entry reduce_fmajor" in ptx
    assert "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32" in ptx
