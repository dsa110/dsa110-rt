"""Layer-2 per-conv-output empirical σ_k EMA (plan §3.6.10 lines 1015-1035).

Tracks one σ scalar per kernel triple `k` in the K=128 detector bank,
EMA-smoothed across cubes with time constant ``τ_s`` (default 30 s).
The EMA is updated on each cube as a side effect of
``DeterministicDetector.forward()``; the resulting per-kernel ``s_k`` is
the divisor used to convert raw conv-bank scores to SNR units.

**Interior-only EMA accumulation** per plan §3.6.12 / §3.6.10 line 1018:
under the ``T_det = 2 blocks = 512`` re-pin, ~25% of every cube
(the first 64 + last 64 samples) has partial-width matched-filter
context (the boxcar reads off the cube edge into zero-padding), which
biases σ low if included. Restrict the σ-clipped std estimator to the
``[t_lo:t_hi] = [n_kernel_max_t//2 : T_det − n_kernel_max_t//2]`` slice
of each cube's per-kernel score.

  * **Burn-in** (cubes 0 … N_burnin-1, default 30): Welford running mean
    s_k ← (count · s_k + σ_cube) / (count + 1). During burn-in,
    ``Layer2State.is_warming_up`` returns True so the Detector can set
    ``flags.bit3 = noise_warmup`` on every emitted Candidate.

  * **EMA** (cubes N_burnin+): s_k ← γ · σ_cube + (1 − γ) · s_k where
    γ = 1 − exp(−cube_cadence_s / τ_s) ≈ 0.00447 at default config.

Score is computed for the full cube (boundary samples included; the
canonical-zone emit gate masks them out — plan §1030-1032 / §4.4
``filter_to_canonical``); only the EMA estimator is interior-only.

The state is a small per-kernel float32 tensor (~512 B for K=128); it
lives on the same device as the kernel bank's image-kernel buffers (so
it's part of the Module state via ``register_buffer`` in
``forward.py``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..common.constants import (
    CUBE_CADENCE_S_DEFAULT,
    N_KERNEL_MAX_T_DEFAULT,
    NOISE_LAYER2_N_BURNIN_DEFAULT,
    NOISE_LAYER2_TAU_S_DEFAULT,
    NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
    NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
)
from .layer1 import sigma_clipped_std

__all__ = [
    "Layer2State",
    "layer2_interior_sigma",
]


def layer2_interior_sigma(
    scores: torch.Tensor,
    *,
    n_kernel_max_t: int = N_KERNEL_MAX_T_DEFAULT,
    n_sigma: float = NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
    n_iterations: int = NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """Compute one interior σ_k per kernel triple from one cube's
    per-kernel score tensor.

    Args:
        scores: ``[K, T_det, N_fdm, H, W]`` per-kernel score tensor (the
            output of ``DeterministicDetector._compute_per_kernel_scores``,
            BEFORE division by s_k).
        n_kernel_max_t: widest detector time-kernel boxcar width. The
            interior slice is ``scores[:, t_lo:t_hi, :, :, :]`` with
            ``t_lo = n_kernel_max_t // 2``,
            ``t_hi = T_det − n_kernel_max_t // 2``.
        n_sigma / n_iterations: σ-clipped std parameters, forwarded to
            ``sigma_clipped_std``.

    Returns:
        ``[K] float32`` per-kernel interior σ_k for this cube.
    """
    if scores.dim() != 5:
        raise ValueError(
            f"scores.dim()={scores.dim()}, expected 5 [K, T_det, N_fdm, H, W]"
        )
    K, T_det = scores.shape[0], scores.shape[1]  # noqa: N806
    if T_det <= n_kernel_max_t:
        # Cube too small to have an interior; fall back to the full cube.
        # (This path is exercised by the small unit-test cubes; production
        # T_det = 512 ≫ n_kernel_max_t = 128 so this branch is inactive.)
        t_lo, t_hi = 0, T_det
    else:
        t_lo, t_hi = n_kernel_max_t // 2, T_det - n_kernel_max_t // 2

    out = torch.empty((K,), dtype=torch.float32, device=scores.device)
    for k in range(K):
        out[k] = sigma_clipped_std(
            scores[k, t_lo:t_hi, :, :, :],
            n_sigma=n_sigma,
            n_iterations=n_iterations,
            max_samples=max_samples,
            rng_seed=int(k),
        )
    return out


class Layer2State:
    """Stateful per-kernel σ_k tracker (Welford burn-in → EMA).

    Owned by ``DeterministicDetector`` (registered as a buffer in
    ``forward.py`` so .to(device) / .cuda() / state_dict round-trip).
    Updated by ``forward()`` on each cube; queried by the Detector to
    divide raw scores into SNR units.
    """

    def __init__(
        self,
        n_kernels: int,
        *,
        cube_cadence_s: float = CUBE_CADENCE_S_DEFAULT,
        tau_s: float = NOISE_LAYER2_TAU_S_DEFAULT,
        n_burnin: int = NOISE_LAYER2_N_BURNIN_DEFAULT,
        n_kernel_max_t: int = N_KERNEL_MAX_T_DEFAULT,
        n_sigma: float = NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
        n_iterations: int = NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
        sigma_max_samples: Optional[int] = None,
        sigma_floor: float = 0.0,
        sigma_max_ratio: float = 0.0,
        clamp_escape_cubes: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if n_kernels < 1:
            raise ValueError(f"n_kernels={n_kernels}, expected ≥ 1")
        if tau_s <= 0:
            raise ValueError(f"tau_s={tau_s}, expected > 0")
        if cube_cadence_s <= 0:
            raise ValueError(f"cube_cadence_s={cube_cadence_s}, expected > 0")
        if n_burnin < 0:
            raise ValueError(f"n_burnin={n_burnin}, expected ≥ 0")
        if n_kernel_max_t < 1:
            raise ValueError(f"n_kernel_max_t={n_kernel_max_t}, expected ≥ 1")
        if sigma_floor < 0:
            raise ValueError(f"sigma_floor={sigma_floor}, expected ≥ 0")
        if sigma_max_ratio < 0:
            raise ValueError(f"sigma_max_ratio={sigma_max_ratio}, expected ≥ 0")
        if clamp_escape_cubes < 0:
            raise ValueError(
                f"clamp_escape_cubes={clamp_escape_cubes}, expected ≥ 0"
            )

        self.n_kernels = int(n_kernels)
        self.cube_cadence_s = float(cube_cadence_s)
        self.tau_s = float(tau_s)
        self.n_burnin = int(n_burnin)
        self.n_kernel_max_t = int(n_kernel_max_t)
        self.n_sigma = float(n_sigma)
        self.n_iterations = int(n_iterations)
        self.sigma_max_samples = (
            int(sigma_max_samples) if sigma_max_samples is not None else None
        )
        # M7.4 hardening: prevent σ_k from being depressed by sparse-valid
        # cubes (low-amplitude masked cells biasing σ-clip downwards).
        # σ_k is clamped to ``sigma_floor`` from below after every update
        # so SNR = score / σ_k is bounded above by ``score / sigma_floor``
        # (no unbounded inflation when σ_k learning gets unlucky on one
        # node). Default 0.0 preserves legacy behaviour bit-for-bit.
        self.sigma_floor = float(sigma_floor)
        # 2026-06-07 hardening (T1): cap how far a single cube can lift
        # σ_k. Without an upper bound, a transient high-energy cube
        # (bright RFI, an over-fluence calibration probe, a merger
        # glitch) inflates σ_k and the τ_s≈30 s relaxation suppresses
        # all detections on that half for ~30 s — observed 2026-06-07
        # as the "candidate counter freeze" failure mode after a
        # dump-storm + manual cal-probe burst. When ``sigma_max_ratio
        # > 0`` and the post-EMA σ_k would exceed
        # ``sigma_max_ratio × s_k_prev``, the update for that kernel is
        # rejected (σ_k stays at the previous value) and
        # ``n_clamped_high`` is incremented. ``0.0`` disables the
        # clamp (preserves bit-for-bit legacy behaviour).
        self.sigma_max_ratio = float(sigma_max_ratio)
        # 2026-06-09 (clamp escape hatch): the T1 clamp alone can
        # DEADLOCK — if the live noise legitimately rises above
        # ``sigma_max_ratio × s_k``, every subsequent update is
        # rejected and s_k can never follow (observed on the live
        # fleet as ``n_clamped_high`` incrementing on every cube for
        # hours, a stuck-low σ_k for ≥ 2 kernels, an SNR-inflated
        # junk-candidate floor on the affected kernels, and probe
        # candidates normalised against the wrong divisor). When
        # ``clamp_escape_cubes > 0`` and a kernel's update has been
        # clamped for that many CONSECUTIVE cubes, the next update is
        # accepted as a REBASELINE: s_k jumps directly to that cube's
        # σ_this (the EMA blend would take ~1/gamma escapes ≈ hours
        # to converge otherwise). The streak requirement preserves the
        # clamp's original purpose — a transient (probe / RFI burst /
        # dump storm) spanning < clamp_escape_cubes cubes still never
        # lifts σ_k. ``0`` disables the hatch (legacy T1 behaviour).
        self.clamp_escape_cubes = int(clamp_escape_cubes)
        self.gamma = 1.0 - math.exp(-self.cube_cadence_s / self.tau_s)

        # State: per-kernel running mean / EMA value. Initialised to 1.0
        # (a sensible no-op divisor before any cube has been seen — but
        # callers should not query before update_and_query has been
        # called at least once; the burn-in flag protects against a
        # bench accidentally consuming the noise_warmup state).
        self._s_k = torch.ones(
            (self.n_kernels,), dtype=dtype, device=device,
        )
        self._cube_count = 0
        # T1 mon-points: per-kernel count of cubes whose σ_k update was
        # clamped from above by ``sigma_max_ratio``. Surfaced via
        # ``n_clamped_high`` (scalar sum) and ``per_kernel_clamped_high``
        # (per-kernel tensor) so operators can spot a single-kernel
        # pathology vs. a fleet-wide noise event.
        self._n_clamped_high_total = 0
        self._per_kernel_clamped_high = torch.zeros(
            (self.n_kernels,), dtype=torch.int64, device=device,
        )
        # Clamp-escape state: per-kernel CONSECUTIVE-clamped-cube
        # streak (reset to 0 whenever a kernel's update is accepted)
        # and rebaseline counters.
        self._per_kernel_clamp_streak = torch.zeros(
            (self.n_kernels,), dtype=torch.int64, device=device,
        )
        self._n_clamp_escapes_total = 0
        self._per_kernel_clamp_escapes = torch.zeros(
            (self.n_kernels,), dtype=torch.int64, device=device,
        )

    @property
    def s_k(self) -> torch.Tensor:
        """Current per-kernel σ_k tensor (cloned to prevent mutation)."""
        return self._s_k.detach().clone()

    @property
    def cube_count(self) -> int:
        return self._cube_count

    @property
    def is_warming_up(self) -> bool:
        """Whether the EMA is in burn-in. Detector sets
        ``flags.bit3 = noise_warmup`` while True."""
        return self._cube_count < self.n_burnin

    @property
    def n_clamped_high(self) -> int:
        """Total cube×kernel updates that were clamped from above by
        ``sigma_max_ratio``. Aggregated across all kernels and cubes
        since the last :meth:`reset`. Surfaces the inflation-protection
        path firing rate."""
        return int(self._n_clamped_high_total)

    @property
    def per_kernel_clamped_high(self) -> torch.Tensor:
        """Per-kernel cumulative count of clamped-from-above updates
        (clone). Useful for distinguishing a single-kernel hot spot
        from a fleet-wide noise event."""
        return self._per_kernel_clamped_high.detach().clone()

    @property
    def n_clamp_escapes(self) -> int:
        """Total clamp-escape rebaselines (kernel was clamped for
        ``clamp_escape_cubes`` consecutive cubes, then σ_k jumped to
        the live estimate). A non-zero value means the noise level
        genuinely moved past the ``sigma_max_ratio`` ceiling."""
        return int(self._n_clamp_escapes_total)

    @property
    def per_kernel_clamp_escapes(self) -> torch.Tensor:
        """Per-kernel cumulative clamp-escape rebaseline count (clone)."""
        return self._per_kernel_clamp_escapes.detach().clone()

    @property
    def per_kernel_clamp_streak(self) -> torch.Tensor:
        """Per-kernel CURRENT consecutive-clamped-cube streak (clone).
        A streak pinned near ``clamp_escape_cubes`` (or growing without
        bound when the hatch is disabled) is the live signature of the
        clamp deadlock."""
        return self._per_kernel_clamp_streak.detach().clone()

    def reset(self) -> None:
        """Clear state; reset cube_count. ``cmd: start --resume=false``."""
        self._s_k.fill_(1.0)
        self._cube_count = 0
        self._n_clamped_high_total = 0
        self._per_kernel_clamped_high.zero_()
        self._per_kernel_clamp_streak.zero_()
        self._n_clamp_escapes_total = 0
        self._per_kernel_clamp_escapes.zero_()

    def update_and_query(
        self,
        scores: Optional[torch.Tensor] = None,
        *,
        per_kernel_sigma: Optional[torch.Tensor] = None,
        valid: bool = True,
    ) -> Tuple[torch.Tensor, bool]:
        """Update the per-kernel σ_k from one cube's score tensor and
        return ``(s_k_for_this_cube, is_warming_up)``.

        Pass exactly one of ``scores`` (full per-kernel score tensor;
        ``layer2_interior_sigma`` is run internally) or
        ``per_kernel_sigma`` (precomputed per-kernel σ tensor; lets the
        Detector forward path skip a recomputation when it already has
        the per-kernel interior σ from a fused kernel).

        ``valid``: if False (cube was warmup-flagged or RFI'd), the EMA
        is NOT updated for this cube — but the current ``s_k`` is still
        returned (so the Detector divides by the *previous* cube's s_k
        and emits whatever exceeds threshold; the dropped cube's σ
        contribution is never folded in). Per plan §319.

        Returns:
            ``(s_k, is_warming_up)`` — ``s_k`` is the value of
            ``self._s_k`` *after* this update (i.e., the divisor the
            caller should use for this cube). ``is_warming_up`` is
            True if cube_count < n_burnin (post-update).
        """
        if (scores is None) == (per_kernel_sigma is None):
            raise ValueError(
                "pass exactly one of scores= or per_kernel_sigma="
            )
        if not valid:
            # Skip update; just return current state.
            return self._s_k.detach().clone(), self.is_warming_up

        if scores is not None:
            sigma_this = layer2_interior_sigma(
                scores,
                n_kernel_max_t=self.n_kernel_max_t,
                n_sigma=self.n_sigma,
                n_iterations=self.n_iterations,
                max_samples=self.sigma_max_samples,
            ).to(self._s_k.dtype).to(self._s_k.device)
        else:
            assert per_kernel_sigma is not None
            sigma_this = per_kernel_sigma.to(self._s_k.dtype).to(self._s_k.device)
        if sigma_this.shape != (self.n_kernels,):
            raise ValueError(
                f"per-kernel sigma shape {tuple(sigma_this.shape)} != "
                f"(n_kernels={self.n_kernels},)"
            )
        # Replace any zeros (degenerate kernel — would divide-by-zero
        # downstream) with the previous EMA value, so a single bad
        # cube doesn't poison the divisor.
        zero_mask = sigma_this == 0
        if torch.any(zero_mask):
            sigma_this = torch.where(zero_mask, self._s_k, sigma_this)

        # T1 (2026-06-07): per-kernel upper-bound clamp on the σ_this
        # contribution. When ``sigma_max_ratio > 0`` and the candidate
        # σ_this for a given kernel exceeds ``sigma_max_ratio ×
        # s_k_prev``, that kernel's update is REJECTED (σ_this is
        # replaced by the prior s_k so the blend below is a no-op for
        # that kernel) and the per-kernel clamp counter is bumped. We
        # apply this BEFORE the burn-in / EMA blend so a single bad
        # cube can never lift σ_k beyond the configured factor —
        # bounding the freeze duration after an anomalous cube to a
        # single sample (vs. ~τ_s≈30 s recovery in the unclamped path).
        if self.sigma_max_ratio > 0.0 and self._cube_count > 0:
            ceiling = self.sigma_max_ratio * self._s_k
            high_mask = sigma_this > ceiling
            if torch.any(high_mask):
                # 2026-06-09 escape hatch: kernels that have been
                # clamped for ``clamp_escape_cubes`` CONSECUTIVE cubes
                # rebaseline — σ_k jumps to this cube's estimate —
                # instead of being clamped forever (deadlock; see
                # __init__ docs). Transients shorter than the streak
                # threshold are still fully clamped.
                if self.clamp_escape_cubes > 0:
                    escape_mask = high_mask & (
                        self._per_kernel_clamp_streak
                        >= self.clamp_escape_cubes
                    )
                else:
                    escape_mask = torch.zeros_like(high_mask)
                clamp_mask = high_mask & ~escape_mask
                if torch.any(escape_mask):
                    # Rebaseline: write σ_this into s_k directly. The
                    # EMA blend below then leaves these kernels at
                    # σ_this exactly (gamma·x + (1-gamma)·x = x).
                    self._s_k = torch.where(
                        escape_mask, sigma_this, self._s_k,
                    )
                    self._per_kernel_clamp_escapes += escape_mask.to(
                        self._per_kernel_clamp_escapes.dtype
                    )
                    self._n_clamp_escapes_total += int(
                        escape_mask.sum().item()
                    )
                if torch.any(clamp_mask):
                    # Replace the offending kernels' σ_this with the
                    # prior s_k so the EMA blend is an identity for
                    # them.
                    sigma_this = torch.where(
                        clamp_mask, self._s_k, sigma_this,
                    )
                    self._per_kernel_clamped_high += clamp_mask.to(
                        self._per_kernel_clamped_high.dtype
                    )
                    self._n_clamped_high_total += int(
                        clamp_mask.sum().item()
                    )
                # Streak bookkeeping: clamped kernels extend their
                # streak; accepted kernels (escaped or under the
                # ceiling) reset to zero.
                self._per_kernel_clamp_streak = torch.where(
                    clamp_mask,
                    self._per_kernel_clamp_streak + 1,
                    torch.zeros_like(self._per_kernel_clamp_streak),
                )
            else:
                self._per_kernel_clamp_streak.zero_()

        if self._cube_count < self.n_burnin:
            count = float(self._cube_count)
            self._s_k = (count * self._s_k + sigma_this) / (count + 1.0)
        else:
            self._s_k = self.gamma * sigma_this + (1.0 - self.gamma) * self._s_k

        # σ_k floor: prevents the EMA from being depressed by an
        # unlucky low-amplitude (masked / RFI'd / non-stationary)
        # cube. Applied AFTER the blend so steady-state σ_k > floor
        # is unchanged; only protects the divisor in pathological
        # cases. Disabled by default (``sigma_floor=0.0``).
        if self.sigma_floor > 0.0:
            self._s_k = torch.clamp(self._s_k, min=self.sigma_floor)

        self._cube_count += 1
        return self._s_k.detach().clone(), self.is_warming_up
