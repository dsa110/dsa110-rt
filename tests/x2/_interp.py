"""Run a snippet under ``TRITON_INTERPRET=1`` on the CPU, in a subprocess.

Why a subprocess: Triton picks interpreter vs compiler at ``@triton.jit``
decoration time from the environment, so the flag must be set before any
kernel module is imported -- which pytest's collection may already have
done in this process.

Why an overlay: the node env ships numpy 2.4, and triton 3.1's C++
interpreter load/store silently returns garbage against the numpy 2 ABI
(probed 2026-10-22 on n13: a ``tl.load``/``tl.store`` copy kernel leaves
its output untouched). With numpy 1.26 first on ``PYTHONPATH`` the
interpreter is exact. Create the overlay once per test node::

    /home/ubuntu/miniforge3/envs/dsa110-rt/bin/python -m pip install \\
        --no-deps --target /home/ubuntu/x2_scratch/_np126 numpy==1.26.4

(``X2_NP1_OVERLAY`` overrides the path.) If no working interpreter is
found the calling test is skipped with the reason, never silently passed.

WS-S's numpy shim (``dsart.x2.search.triton_interp.install``) is tried as a
third option once it is on the path. Whatever is used must pass a probe with
masked and unmasked loads/stores and a dot, and every interpreted test also
runs a MUTANT kernel (``_mutant``: one source edit) and asserts that its
check FAILS -- so a pass can never come from an interpreter that silently
did nothing.

Every snippet is prefixed with :data:`PREAMBLE`, which teaches the
interpreter string annotations: kernel modules written with ``from
__future__ import annotations`` (``triton_dedisp``, ``triton_corr_fused``)
carry ``"tl.constexpr"`` strings, which the compiler accepts (substring
test) but triton 3.1's interpreter only recognises as the exact string
``"constexpr"``.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys

import pytest

OVERLAY = os.environ.get("X2_NP1_OVERLAY", "/home/ubuntu/x2_scratch/_np126")

PREAMBLE = r"""
import triton.runtime.jit as _tj
_tj_norm = _tj._normalize_ty
def _tj_norm2(ty):
    r = _tj_norm(ty)
    return "constexpr" if isinstance(r, str) and r.endswith("constexpr") else r
_tj._normalize_ty = _tj_norm2


def _mutant(modname, attr, old, new):
    # A copy of ``modname`` with ONE source edit (``old`` -> ``new``, must occur
    # exactly once), re-executed so its kernels are re-decorated; returns its
    # ``attr``. Running a mutant proves the interpreter really executes the
    # kernel: the check that passes on the original must fail on the mutant.
    import importlib, sys, types
    m = importlib.import_module(modname)
    src = open(m.__file__).read()
    assert src.count(old) == 1, (modname, old, src.count(old))
    name = modname + "_x2mutant"
    mod = types.ModuleType(name)
    mod.__file__ = m.__file__
    sys.modules[name] = mod
    exec(compile(src.replace(old, new), m.__file__, "exec"), mod.__dict__)
    return getattr(mod, attr)
"""

_SHIM = """
from dsart.x2.search.triton_interp import install as _x2_install
_x2_install()
"""

#: The probe exercises unmasked AND masked loads/stores and a dot: a broken
#: interpreter (numpy-2 ABI: loads return ``other``, stores write nothing)
#: cannot pass it.
_PROBE = r"""
import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, y_ptr, z_ptr, N: tl.constexpr):
    o = tl.arange(0, N)
    a = tl.load(x_ptr + o[:, None] * N + o[None, :])
    tl.store(y_ptr + o[:, None] * N + o[None, :], tl.dot(a, a, out_dtype=tl.float32))
    m = o < N - 3
    b = tl.load(x_ptr + o, mask=m, other=-7.0)
    tl.store(z_ptr + o, b.to(tl.float32) + 1.0, mask=o >= 2)
x = torch.randn(16, 16).half()
y = torch.zeros(16, 16)
z = torch.full((16,), 5.0)
k[(1,)](x, y, z, 16)
err = (y.double() - x.double() @ x.double()).abs().max().item()
want = x.reshape(-1)[:16].float() + 1.0
want[13:] = -6.0
want[:2] = 5.0
ok = err < 1e-3 and torch.allclose(z, want, atol=1e-3)
print("PROBE", "ok" if ok else f"bad {err} {z.tolist()}")
"""


def _env(overlay: bool) -> dict:
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    if overlay and os.path.isdir(OVERLAY):
        env["PYTHONPATH"] = OVERLAY + os.pathsep + env.get("PYTHONPATH", "")
    return env


@functools.lru_cache(maxsize=1)
def _working_env() -> tuple[dict | None, str, str]:
    """``(env, preamble, why_not)``: plain, numpy<2 overlay, or WS-S's shim."""
    notes = []
    for overlay, shim in ((False, False), (True, False), (False, True)):
        pre = PREAMBLE + (_SHIM if shim else "")
        try:
            r = subprocess.run(
                [sys.executable, "-c", pre + _PROBE], env=_env(overlay),
                capture_output=True, text=True, timeout=300,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"overlay={overlay} shim={shim}: {exc!r}")
            continue
        if "PROBE ok" in r.stdout:
            return _env(overlay), pre, ""
        notes.append(f"overlay={overlay} shim={shim}: {(r.stdout + r.stderr).strip()[-160:]}")
    return None, "", "; ".join(notes)


def run_interpreted(code: str, *, timeout: int = 900) -> str:
    """Run ``code`` under the Triton interpreter; return its stdout.
    Skips the calling test if no working interpreter is available and
    fails it (with stderr) if the snippet itself errors."""
    env, pre, why = _working_env()
    if env is None:
        pytest.skip(f"no working Triton interpreter (numpy<2 overlay {OVERLAY} or WS-S shim): {why}")
    r = subprocess.run(
        [sys.executable, "-c", pre + code], env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise AssertionError(f"interpreted snippet failed:\n{r.stdout}\n{r.stderr[-4000:]}")
    return r.stdout
