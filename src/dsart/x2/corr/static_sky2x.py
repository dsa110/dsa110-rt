"""Batched static-sky subtraction for the 2x corr (x2 WS-C deliverable 3).

``StaticSkyMean`` semantics (corr_fast_integration), one independent state
per (coarse DM, value), all DMs of a tier in one call instead of a Python
loop of ``n_dm`` ``apply`` calls:

* a block is passed through while fewer than ``warmup_cubes`` blocks have
  been seen (or the window is empty), otherwise the mean of the previous
  ``window_blocks`` blocks' per-value means is subtracted -- strictly
  causal, a burst never subtracts itself;
* then this block's per-value mean (over its time samples) joins the
  window; the running sum is re-summed exactly every 1024 blocks;
* 2x (adopted 2026-10-29, with the row rule v1.1): an excised sample
  (exactly 0 + 0i: its voltages were zeroed by the array-burst / RFI mask)
  stays exactly 0 -- the window mean is not subtracted from it. Subtracting
  it turned excised data into a fake -sky value that the row rule counted
  as data. The 1x ``StaticSkyMean`` path is unchanged.

In place on ``x`` ``(n_d, n_t, V)`` complex64, ~5 elementwise/reduction
kernels per tier regardless of ``n_d``. Every DM of a tier advances
together (one block per call), so the counters are scalars. The CPU run
of the same code is the twin; ``tests/x2/test_corr2x_static_sky.py``
checks it against ``StaticSkyMean`` per DM and against the float64
reference.
"""
from __future__ import annotations

import torch

__all__ = ["BatchedStaticSky"]


class BatchedStaticSky:
    RESUM_EVERY = 1024

    def __init__(self, *, n_d: int, n_values: int, window_blocks: int = 8,
                 warmup_cubes: int = 8, device="cpu") -> None:
        if window_blocks < 1 or warmup_cubes < 0:
            raise ValueError("window_blocks >= 1 and warmup_cubes >= 0")
        self.window_blocks = int(window_blocks)
        self.warmup_cubes = int(warmup_cubes)
        self.ring = torch.zeros((self.window_blocks, n_d, n_values), dtype=torch.complex64,
                                device=device)
        self.sum = torch.zeros((n_d, n_values), dtype=torch.complex64, device=device)
        self.count = 0
        self.head = 0
        self.seen = 0

    def reset(self) -> None:
        self.ring.zero_()
        self.sum.zero_()
        self.count = self.head = self.seen = 0

    def running_mean(self) -> torch.Tensor | None:
        return None if self.count == 0 else self.sum / self.count

    def running_mean_for(self, dm_slot: int = 0) -> torch.Tensor | None:
        """``(V,)`` window mean of one DM -- the ``StaticSkyMean`` accessor
        :class:`dsart.services.sky_export.SkySnapshotExporter` reads."""
        return None if self.count == 0 else self.sum[int(dm_slot)] / self.count

    def apply_(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract the past-window mean from ``x`` in place and push its mean."""
        if x.dim() != 3 or x.shape[0] != self.sum.shape[0] or x.shape[2] != self.sum.shape[1]:
            raise ValueError(f"x shape {tuple(x.shape)} != (n_d, n_t, {tuple(self.sum.shape)[1]})")
        mean = x.mean(dim=1)                                        # (n_d, V)
        if self.count > 0 and self.seen >= self.warmup_cubes:
            sky = (self.sum / self.count).unsqueeze(1)
            x.sub_(torch.where(x != 0, sky, torch.zeros_like(sky)))  # excised stays 0
        if self.count == self.window_blocks:
            self.sum.sub_(self.ring[self.head])
        else:
            self.count += 1
        self.ring[self.head].copy_(mean)
        self.sum.add_(mean)
        self.head = (self.head + 1) % self.window_blocks
        self.seen += 1
        if self.seen % self.RESUM_EVERY == 0:
            self.sum.copy_(self.ring[: self.count].sum(dim=0))
        return x
