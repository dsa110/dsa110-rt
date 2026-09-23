"""Fused Layer-1 under --pipeline-overlap: the detector input stays unit-σ.

search_compute's overlap loop builds cube N+1 (imager, on the prefetch
stream) BEFORE running cube N's Layer-1. Layer-1 used to recover the
absolute σ with the LATEST σ instead of the one the cube was actually
imaged with, and rewrote the fused mask in place while the next build
was reading it. The per-fdm σ then obeyed an undamped recursion
(log σ_n = log σ_raw + log σ_{n-1} - log σ_{n-2}); live on 2026-09-23 the
detector saw per-fdm σ from 0.14 to 0.61 and flooded C1.

This drives the pipeline in exactly the production order through a
noise-level step (a startup-like transient) and requires every fine-DM
trial of every settled cube to reach the detector at unit σ.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("GPU pipeline test", allow_module_level=True)

from dsart.common.constants import N_CHGROUP  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.fine_dm.combiner import compute_time_shift_search  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import CubeRingSlot  # noqa: E402

N_GRID = 32
T_DET = 48
N_FDM = 6


def _noise_slot(cube_id, table, t_stream, amp, rng):
    streams = {}
    for g in range(N_CHGROUP):
        re = rng.standard_normal((t_stream, N_GRID, N_GRID))
        im = rng.standard_normal((t_stream, N_GRID, N_GRID))
        streams[g] = (amp * (re + 1j * im)).astype(np.complex64)
    return CubeRingSlot(
        cube_id=cube_id,
        specnum_start=cube_id * T_DET,
        per_chgroup_streams=streams,
        time_shift_table=table,
        validity_mask=np.ones((T_DET, N_FDM), dtype=np.bool_),
        n_fdm_in_cube=N_FDM,
        t_det=T_DET,
        n_grid=N_GRID,
    )


def test_fused_layer1_unit_sigma_under_overlap(monkeypatch):
    monkeypatch.setenv("DSART_LAYER1_COVERAGE_CORRECT", "0")
    fine = np.linspace(40.0, 60.0, N_FDM)
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=np.array([0.0]), fine_dm_pc_cm3=fine,
        fine_to_coarse=np.zeros(N_FDM, dtype=np.int64),
        t_int_search_us=524.288,
    )
    t_stream = T_DET + int(table.shifts.max()) + 4
    cfg = CubePipelineConfig(
        n_grid=N_GRID, edge_mask_kernel_support=3, device="cuda",
        cube_dtype=torch.float16, gpu_complex_dtype=torch.complex32,
        image_backend="gpu",
    )
    det = DeterministicDetector(
        threshold_sigma=999.0, detector_version="v1.M5", search_node_id=0,
        gpu_half=0, dtype=torch.float16, device=torch.device("cuda"),
    )
    pipe = CubePipeline(config=cfg, detector=det,
                        layer1_state=Layer1State(N_FDM, n_burnin_cubes=2))
    assert pipe._fuse_layer1_into_imager

    rng = np.random.default_rng(0)
    amps = [1.0] * 6 + [3.0] * 6          # step at cube 6
    slots = [_noise_slot(k, table, t_stream, a, rng) for k, a in enumerate(amps)]
    sig = []
    pending = pipe.prefetch_build(slots[0])
    for k in range(len(slots)):
        nxt = pipe.prefetch_build(slots[k + 1]) if k + 1 < len(slots) else None
        res = pipe.process_prefetched(pending)            # production order
        c = res.cube.float()
        live = c.abs().sum((0, 1)) > 0                    # inside edge mask
        sig.append([float(c[:, f][:, live].std()) for f in range(N_FDM)])
        pending = nxt
    sig = np.array(sig)
    # Past burn-in (cube >= 3) the level must be steady. Measured on
    # n13: old code 0.73..1.57 with an undamped ~6-cube oscillation even
    # at CONSTANT noise; fixed 1.01..1.16. (The plain std here includes
    # the edge-taper cells the sigma-clipped L1 subsample treats
    # differently, hence "near 1", not "== 1".)
    s = sig[3:]
    assert float((s.max(0) / s.min(0)).max()) < 1.25, sig
    assert 0.9 < float(np.median(s)) < 1.2, sig
