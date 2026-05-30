"""Standalone equivalence check for the post-cal injection G-fold.

Asserts: adding a bare geometric injection BEFORE cal and then applying
apply_cal_split is numerically identical to applying apply_cal_split
first and then adding the SAME injection with the cal gain folded into
its phasor (the production placement after the 2026-05-30 fix).

Run on a GPU corr node:
  DSART_CAL_APPLY_COMPILE=0 PYTHONPATH=src python tools/ops/_verify_inject_calfold.py
"""
import numpy as np
import torch

from dsart.common.constants import NANTS, NCHAN_PER_CHGROUP, NPOL
from dsart.inject.online import OnlineInjector, InjectionConfig
from dsart.services.slow_corr_kernel import (
    apply_cal_split, make_cal_broadcast_tensors,
    NPACKETS_PER_BLOCK, NTIMES_PER_PACKET,
)

torch.manual_seed(0)
rng = np.random.default_rng(0)

dev = torch.device("cuda:0")
dt = torch.float16
chgroup = 7

shape = (NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL, NPACKETS_PER_BLOCK, NANTS)

# --- non-trivial per-(ant,ch,pol) complex cal gain (mean|G|=1) ---
mag = rng.uniform(0.3, 2.0, size=(NANTS, NCHAN_PER_CHGROUP, NPOL))
phase = rng.uniform(-np.pi, np.pi, size=(NANTS, NCHAN_PER_CHGROUP, NPOL))
gains_fine = (mag * np.exp(1j * phase)).astype(np.complex64)
gains_fine /= np.abs(gains_fine).mean()                 # mean|G| = 1
cal_real, cal_imag = make_cal_broadcast_tensors(gains_fine, device=dev, dtype=dt)

# reconstruct (NPOL, NANTS, NCHAN) the same way build_context does
cr = cal_real.detach().to("cpu", torch.float32).numpy()[:, 0, :, 0, :]
ci = cal_imag.detach().to("cpu", torch.float32).numpy()[:, 0, :, 0, :]
cal_gain = (np.transpose(cr, (1, 2, 0)) + 1j * np.transpose(ci, (1, 2, 0)))

antpos_e = rng.uniform(-500, 500, NANTS)
antpos_n = rng.uniform(-500, 500, NANTS)

cfg = InjectionConfig(
    inj_id="eq", l_rad=0.02, m_rad=-0.015, dm_pc_cm3=150.0,
    width_samples=16, fluence_jy_ms=5000.0, profile="gaussian",
    apply_at_specnum=0,
)

# common random voltages
V0r = (torch.randn(shape, device=dev) * 4).to(dt)
V0i = (torch.randn(shape, device=dev) * 4).to(dt)

# --- Path A: bare inject (pre-cal), then cal ---
injA = OnlineInjector(antpos_e, antpos_n, chgroup, device=dev, dtype=dt,
                      cal_gain=None)
injA.add_pending(cfg)
Ar, Ai = V0r.clone(), V0i.clone()
injA.apply_block(Ar, Ai, block_specnum_start=0)
Ar, Ai = apply_cal_split(Ar, Ai, cal_real, cal_imag)

# --- Path B: cal, then inject with G folded (production post-cal) ---
injB = OnlineInjector(antpos_e, antpos_n, chgroup, device=dev, dtype=dt,
                      cal_gain=cal_gain)
injB.add_pending(cfg)
Br, Bi = apply_cal_split(V0r.clone(), V0i.clone(), cal_real, cal_imag)
injB.apply_block(Br, Bi, block_specnum_start=0)

dr = (Ar.to(torch.float32) - Br.to(torch.float32)).abs()
di = (Ai.to(torch.float32) - Bi.to(torch.float32)).abs()
scale = Ar.to(torch.float32).abs().mean().item() + 1e-6
print(f"mean|A| = {scale:.4f}")
print(f"max |dr| = {dr.max().item():.4f}  mean = {dr.mean().item():.5f}")
print(f"max |di| = {di.max().item():.4f}  mean = {di.mean().item():.5f}")
rel = max(dr.max().item(), di.max().item()) / scale
print(f"relative max diff = {rel:.4f}")
# fp16 cal multiply + 2x ordering → expect small but nonzero rounding
assert rel < 0.05, f"post-cal G-fold NOT equivalent to pre-cal (rel={rel})"
print("PASS: post-cal inject (G-fold) == pre-cal inject then cal (fp16)")
