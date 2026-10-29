"""CPU dry run of the CP1 owner bench (bench/timeres/b8_owner2x.py):
``compile-check`` (python sources, NVRTC compile-only, Triton compile-only
for sm_75, CPU twins, TRITON_INTERPRET self-test) and ``teardown``; GPU
subcommands refuse cleanly (exit 2) without a GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
B8 = REPO / "bench" / "timeres" / "b8_owner2x.py"


def _run(args, tmp_path, timeout=840):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("TRITON_INTERPRET", None)
    return subprocess.run([sys.executable, str(B8)] + args, cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_compile_check_passes_on_cpu(tmp_path):
    p = _run(["compile-check", "--out", str(tmp_path)], tmp_path)
    res = json.loads((tmp_path / "b8_compile_check.json").read_text())["results"]
    bad = [r for r in res if r["status"] == "fail"]
    assert p.returncode == 0 and not bad, (p.returncode, bad, p.stdout[-2000:], p.stderr[-2000:])
    names = {r["check"] for r in res}
    assert {"py_compile", "cuda_nvrtc", "triton_compile", "twin_k0", "twin_fft",
            "twin_k3b"} <= names
    st = {r["check"]: r["status"] for r in res if r["check"] in ("cuda_nvrtc", "triton_interpret")}
    print(st)
    nv = next(r for r in res if r["check"] == "cuda_nvrtc")
    from dsart.x2.search import nvrtc
    if nvrtc.available():                                     # NVRTC needs no GPU: must really compile
        assert nv["status"] == "ok", nv
        assert {"k0_unpack_frame2", "k2_colfft_256", "k3a_rowc2r_256", "k1_staged",
                "k3b_cuda_b3x", "k3b_cuda_box", "k3b_reg_b3x_0", "k3b_reg_b3x_2",
                "k3b_reg_box_0"} <= set(
            nv["kernels"]), nv
    tc = {r.get("kernel"): r["status"] for r in res if r["check"] == "triton_compile"}
    if tc.get("K1") == "ok":                       # Triton present: every variant compiled
        fused = {k: v for k, v in tc.items() if k and k.startswith("K3b fused")}
        assert len(fused) == 4 and set(fused.values()) == {"ok"}, tc


def test_gpu_subcommands_refuse_without_gpu_and_teardown_is_clean(tmp_path):
    p = _run(["kernels", "--fft", "cufft", "--k3b", "box", "--k1-formats", "",
              "--out", str(tmp_path)], tmp_path, timeout=300)
    assert p.returncode == 2 and "needs a GPU" in (p.stdout + p.stderr)
    t = _run(["teardown"], tmp_path, timeout=60)
    assert t.returncode == 0
