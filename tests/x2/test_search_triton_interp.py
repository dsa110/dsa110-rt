"""Triton kernels of the owner engine under TRITON_INTERPRET=1 (CPU tensors).

K1 (CSR combine) and K3b (detector) must equal their numpy twins BIT FOR
BIT -- the same fp32 operations in the same order. Tiny shapes only.
"""
import os

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from dsart.x2.search.triton_interp import install  # noqa: E402

install()       # Triton 3.1's interpreter memory helper is broken under numpy 2

from dsart.x2.search.combine import k1_twin  # noqa: E402
from dsart.x2.search.csr import csr_from_value_cells  # noqa: E402
from dsart.x2.search.filters import FilterBank, k3b_twin  # noqa: E402
from dsart.x2.search.kernels_triton import csr_tensors, k1_combine, k3b_detect  # noqa: E402


@pytest.mark.parametrize("fold", ["col", "row"])
def test_k1_triton_equals_twin_bit_for_bit(fold):
    rng = np.random.default_rng(0)
    N, S, V, R = 16, 4, 12, 16
    cells = []
    for _ in range(S):
        c = rng.choice(N * N, size=V, replace=False)
        c[:3] = c[3:6]                                     # many-to-one
        c[-2:] = -1
        cells.append(c.astype(np.int64))
    csr = csr_from_value_cells(cells, n_grid=N, fold=fold)
    codes = rng.integers(-7, 8, size=(S, V, R, 2)).astype(np.int8)
    cv = rng.uniform(0.05, 9.0, size=(S, V)).astype(np.float32)
    wv = np.sqrt(rng.integers(1, 3000, size=(S, V)).astype(np.float64)).astype(np.float32)
    shifts = rng.integers(0, 6, size=(2, S)).astype(np.int32)
    want = k1_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=shifts, t_lo=20, n_t=5, w_v=wv)
    got = k1_combine(torch.from_numpy(codes), torch.from_numpy(cv), csr_tensors(csr, "cpu"),
                     torch.from_numpy(shifts), t_lo=20, n_t=5, block_c=16, block_t=4,
                     w_v=torch.from_numpy(wv)).numpy()
    assert np.array_equal(got[..., 0], want.real) and np.array_equal(got[..., 1], want.imag)


@pytest.mark.parametrize("bank", ["box", "b3x"])
def test_k3b_triton_equals_twin_bit_for_bit(bank):
    rng = np.random.default_rng(1)
    fb = FilterBank.build((1, 2, 4, 8)) if bank == "box" else FilterBank.b3x()
    F, P, n_c = 2, 20, 64
    hist = fb.history()
    R = n_c + hist
    ring = rng.standard_normal((R, F, P)).astype(np.float16)
    ring[5, 1, 3] = 400.0                                     # clip at 250
    c_first = 7 * R + 3 + fb.max_a                            # arbitrary absolute rows
    si = np.array([0.9, 1.7], dtype=np.float32)
    sk = (1.0 / (fb.seeds * 1.1)).astype(np.float32)
    taps = np.array([2, 11, 19])
    b0, t0, p0 = k3b_twin(ring, ring_row0=0, c_first=c_first, n_c=n_c, sigma_inv=si,
                          s_inv=sk, bank=fb, tap_pixels=taps)
    b1, t1, p1 = k3b_detect(torch.from_numpy(ring), ring_row0=0, c_first=c_first, n_c=n_c,
                            sigma_inv=torch.from_numpy(si), s_inv=sk, bank=fb,
                            tap_pixels=taps, block=16)
    assert np.array_equal(b1.numpy(), b0)
    assert np.array_equal(t1.numpy(), t0)
    assert np.array_equal(p1.numpy(), p0)


@triton.jit
def _plus_one(x_ptr, out_ptr, n, ADD: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, x + ADD, mask=m)


@pytest.mark.parametrize("add,ok", [(1.0, True), (2.0, False)])
def test_interpreter_is_not_vacuous(add, ok):
    """The shimmed interpreter really executes kernels: a deliberately
    broken kernel (adds 2) must FAIL the comparison a correct one passes."""
    x = torch.arange(10, dtype=torch.float32)
    out = torch.full((16,), -5.0)
    _plus_one[(1,)](x, out, 10, ADD=add, B=16)
    assert np.array_equal(out[:10].numpy(), (x + 1.0).numpy()) == ok
    assert np.all(out[10:].numpy() == -5.0)                  # masked lanes untouched


@pytest.mark.parametrize("bank", ["box", "b3x", "t"])
def test_k3b_fused_equals_twin_bit_for_bit(bank):
    """ONE fused kernel for the whole bank == the per-filter twin (values,
    packed (t, k) incl. tie order, taps)."""
    from dsart.x2.search.k3b_fused import k3b_fused

    rng = np.random.default_rng(5)
    fb = {"box": FilterBank.build((1, 2, 4, 8)), "b3x": FilterBank.b3x(),
          "t": FilterBank.build((1, 2, 3), ("t",))}[bank]
    F, P, n_c = 2, 20, 64
    R = n_c + fb.history() + 3
    ring = rng.standard_normal((R, F, P)).astype(np.float16)
    ring[7, 0, 4] = 500.0
    c_first = 5 * R + 1 + fb.max_d + fb.max_a
    si = np.array([0.8, 1.3], dtype=np.float32)
    sk = (1.0 / (fb.seeds * 1.07)).astype(np.float32)
    taps = np.array([0, 9, 19])
    b0, t0, p0 = k3b_twin(ring, ring_row0=0, c_first=c_first, n_c=n_c, sigma_inv=si, s_inv=sk,
                          bank=fb, tap_pixels=taps)
    b1, t1, p1 = k3b_fused(torch.from_numpy(ring), ring_row0=0, c_first=c_first, n_c=n_c,
                           sigma_inv=torch.from_numpy(si), s_inv=sk, bank=fb, tap_pixels=taps,
                           block=16)
    assert np.array_equal(b1.numpy(), b0)
    assert np.array_equal(t1.numpy(), t0)
    assert np.array_equal(p1.numpy(), p0)


@pytest.mark.parametrize("mutate", ["offset", "tie"])
def test_k3b_fused_mutants_are_caught(mutate):
    """Deliberately wrong fused kernels (a shifted read; >= instead of the
    tie rule) must NOT match the twin -- the comparison is not vacuous."""
    from dsart.x2.search.k3b_fused import k3b_fused

    rng = np.random.default_rng(6)
    fb = FilterBank.b3x()
    F, P, n_c = 1, 16, 64
    R = n_c + fb.history() + 1
    ring = (rng.standard_normal((R, F, P)) if mutate == "offset"
            else np.zeros((R, F, P))).astype(np.float16)
    c_first = R + fb.max_d + fb.max_a
    si = np.ones(1, np.float32)
    sk = np.ones(len(fb), np.float32)
    b0, t0, _ = k3b_twin(ring, ring_row0=0, c_first=c_first, n_c=n_c, sigma_inv=si, s_inv=sk,
                         bank=fb)
    b1, t1, _ = k3b_fused(torch.from_numpy(ring), ring_row0=0, c_first=c_first, n_c=n_c,
                          sigma_inv=torch.from_numpy(si), s_inv=sk, bank=fb, block=16,
                          mutate=mutate)
    assert not (np.array_equal(b1.numpy(), b0) and np.array_equal(t1.numpy(), t0))
