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
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys

import pytest

OVERLAY = os.environ.get("X2_NP1_OVERLAY", "/home/ubuntu/x2_scratch/_np126")

_PROBE = r"""
import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, y_ptr, N: tl.constexpr):
    o = tl.arange(0, N)
    a = tl.load(x_ptr + o[:, None] * N + o[None, :])
    tl.store(y_ptr + o[:, None] * N + o[None, :], tl.dot(a, a, out_dtype=tl.float32))
x = torch.randn(16, 16).half()
y = torch.zeros(16, 16)
k[(1,)](x, y, 16)
err = (y.double() - x.double() @ x.double()).abs().max().item()
print("PROBE", "ok" if err < 1e-3 else f"bad {err}")
"""


def _env(overlay: bool) -> dict:
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    if overlay and os.path.isdir(OVERLAY):
        env["PYTHONPATH"] = OVERLAY + os.pathsep + env.get("PYTHONPATH", "")
    return env


@functools.lru_cache(maxsize=1)
def _working_env() -> tuple[dict | None, str]:
    notes = []
    for overlay in (False, True):
        try:
            r = subprocess.run(
                [sys.executable, "-c", _PROBE], env=_env(overlay),
                capture_output=True, text=True, timeout=300,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"overlay={overlay}: {exc!r}")
            continue
        if "PROBE ok" in r.stdout:
            return _env(overlay), ""
        notes.append(f"overlay={overlay}: {(r.stdout + r.stderr).strip()[-200:]}")
    return None, "; ".join(notes)


def run_interpreted(code: str, *, timeout: int = 900) -> str:
    """Run ``code`` under the Triton interpreter; return its stdout.
    Skips the calling test if no working interpreter is available and
    fails it (with stderr) if the snippet itself errors."""
    env, why = _working_env()
    if env is None:
        pytest.skip(f"no working Triton interpreter (numpy<2 overlay {OVERLAY}?): {why}")
    r = subprocess.run(
        [sys.executable, "-c", code], env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise AssertionError(f"interpreted snippet failed:\n{r.stdout}\n{r.stderr[-4000:]}")
    return r.stdout
