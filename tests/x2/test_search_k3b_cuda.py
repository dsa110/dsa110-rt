"""K3b as a generated CUDA kernel (x2/search/k3b_cuda.py).

CPU: source generation (groups, shared memory under the static limit).
GPU (x2_gpu): bit-exact vs ``filters.k3b_twin`` -- values, packed (t, k)
incl. ties between signed zeros at masked pixels, taps -- for the b3x, box
and triangle banks, tier-A and tier-B cycle lengths, a pixel count that is
not a multiple of the tile, a nonzero ring origin and clipped inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsart.x2.search.filters import FilterBank, k3b_twin
from dsart.x2.search.k3b_cuda import cpu_source_check, cuda_source, plan_groups

torch = pytest.importorskip("torch")


def _gpu(fn):
    fn = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (CP1)")(fn)
    return pytest.mark.x2_gpu(fn)


BANKS = {"b3x": FilterBank.b3x, "box": FilterBank.build,
         "t": lambda: FilterBank.build((1, 2, 3, 8), ("t",))}


@pytest.mark.parametrize("bank", list(BANKS))
def test_source_and_groups(bank):
    fb = BANKS[bank]()
    g = plan_groups(fb)
    assert sorted(k for grp in g for k in grp) == list(range(len(fb)))
    assert cpu_source_check(fb) is None
    src, meta = cuda_source(fb)
    assert "k3b_cuda" in src and meta["ng"] == len(g)
    from dsart.x2.search import nvrtc
    if nvrtc.available():                                # compiles for sm_75 without a GPU
        assert ".entry k3b_cuda" in nvrtc.ptx(src)
    if bank == "b3x":
        assert g == [[0, 1, 2], [3], [4], [5], [6], [7]]


@pytest.mark.parametrize("bank", list(BANKS))
def test_reg_source_compiles(bank):
    from dsart.x2.search import nvrtc
    from dsart.x2.search.k3b_reg import plan_parts, reg_source
    fb = BANKS[bank]()
    parts = plan_parts(fb)
    assert sorted(k for p in parts for k in p) == list(range(len(fb)))
    for part in parts:
        src, meta = reg_source(fb, part)
        assert meta["U"] % meta["dq"] == 0
        if nvrtc.available():
            assert ".entry k3b_reg" in nvrtc.ptx(src)


@_gpu
@pytest.mark.parametrize("impl", ["cuda", "reg"])
@pytest.mark.parametrize("bank,n_c", [("b3x", 384), ("b3x", 192), ("box", 384), ("t", 128)])
def test_gpu_k3b_cuda_equals_twin(bank, n_c, impl):
    from dsart.x2.search.k3b_cuda import k3b_cuda
    from dsart.x2.search.k3b_reg import k3b_reg

    rng = np.random.default_rng(11)
    fb = BANKS[bank]()
    F, P = 3, 1000                                       # P not a multiple of the 32-px tile
    R = n_c + fb.history() + 7
    ring = rng.standard_normal((R, F, P)).astype(np.float16)
    ring[40, 1, 77] = 600.0                              # clipped at 250 sigma
    ring[:, :, 900:] = np.float16(-0.0)                  # masked pixels: signed-zero ties
    ring[:, 2, 950:] = np.float16(0.0)
    row0 = 13
    c_first = 5 * R + 3 + fb.max_d + fb.max_a
    si = rng.uniform(0.8, 1.2, F).astype(np.float32)
    sk = (1 / (fb.seeds * 1.03)).astype(np.float32)
    taps = np.array([0, 77, 901, 999])
    kw = dict(ring_row0=row0, c_first=c_first, n_c=n_c, s_inv=sk, bank=fb, tap_pixels=taps)
    b0, t0, p0 = k3b_twin(ring, sigma_inv=si, **kw)
    fn = k3b_cuda if impl == "cuda" else k3b_reg
    b1, t1, p1 = fn(torch.from_numpy(ring).cuda(), sigma_inv=torch.from_numpy(si).cuda(), **kw)
    b1, t1, p1 = b1.cpu().numpy(), t1.cpu().numpy(), p1.cpu().numpy()
    assert np.array_equal(b1, b0), np.argwhere(b1 != b0)[:5]
    assert np.array_equal(t1, t0), np.argwhere(t1 != t0)[:5]
    assert np.array_equal(p1, p0)
