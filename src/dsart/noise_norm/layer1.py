"""Layer-1 σ-clipped per-cube global scalar (plan §3.6.9 line 984-1011).

For an input ``cube[T_det, N_fdm, H, W] float32``, compute one σ
scalar per fine_DM trial via 3 iterations of 3σ clipping over the
``[T_det, H, W]`` slab. Output is ``[N_fdm] float32``.

The 5-cube cold-start burn-in (plan §997-1011) is implemented by the
stateful ``Layer1State`` — the bench passes one cube at a time and
``Layer1State.update_and_query()`` returns the per-fdm σ tensor that
should be used to normalise that cube. The first 5 cubes return the
median of all per-cube σs seen so far (so cube 0 returns its own σ,
cube 1 returns median of cubes 0-1, etc.); from cube 6 onward the
current cube's σ is returned directly (no history lookup).

Burn-in cubes are emitted to the detector normally per plan §1011 —
they are NOT warmup-flagged. ``flags.bit3 = noise_warmup`` is reserved
for the Layer-2 burn-in (``layer2.py``).

Implementation notes:

  - sigma_clipped_std uses the median (not the mean) as the clip
    centre per plan §988-993; this is robust to outliers in the cube.
  - NaN cells (set by upstream edge-mask) are excluded throughout.
    The reduction is `nansum`/`nanmedian`-aware so an edge-masked cube
    surface (the §3.6.5 G11 image-plane envelope) doesn't leak into
    the σ estimate.
  - The cube the bench passes in is already in linear units (the
    imager's real Stokes-I dirty image); no log / sqrt prelude.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch

from ..common.constants import (
    NOISE_LAYER1_N_BURNIN_CUBES_DEFAULT,
    NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
    NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
)

__all__ = [
    "Layer1State",
    "layer1_global_scalar",
    "sigma_clipped_std",
    "sigma_clipped_std_batched",
]


def sigma_clipped_std_batched(
    samples: torch.Tensor,
    *,
    n_sigma: float = NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
    n_iterations: int = NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
) -> torch.Tensor:
    """Vectorised σ-clipped std over the last axis of a 2-D tensor.

    Computes one ``sigma_clipped_std`` per row in a single batched
    pipeline (no per-row Python loop, no per-row ``.item()`` syncs).
    NaN-aware: any non-finite input cell is dropped from the estimator
    (consistent with the scalar :func:`sigma_clipped_std`).

    Args:
        samples: ``[B, M]`` tensor of pre-subsampled cells. ``B`` rows
            (independent estimators), ``M`` cells per row. Any float
            dtype; internally upcast to fp32 for the accumulation.
        n_sigma: clip threshold in σ units. Default 3.0.
        n_iterations: number of clip iterations. Default 3.

    Returns:
        ``[B] float32`` per-row σ-clipped standard deviation, on the
        same device as ``samples``. Rows that are entirely
        non-finite (or empty) return 0.0.
    """
    if samples.dim() != 2:
        raise ValueError(
            f"samples.dim()={samples.dim()}, expected 2 [B, M]"
        )
    B, M = samples.shape
    if B == 0 or M == 0:
        return torch.zeros((B,), dtype=torch.float32, device=samples.device)

    s = samples if samples.dtype == torch.float32 else samples.to(torch.float32)

    finite = torch.isfinite(s)
    nan_value = torch.tensor(float("nan"), dtype=torch.float32, device=s.device)
    s_active = torch.where(finite, s, nan_value)

    # Initial median + σ over the finite cells.
    medians = torch.nanmedian(s_active, dim=1).values  # [B] fp32
    diff_sq = (s_active - medians[:, None]) ** 2       # NaN propagates
    count = finite.sum(dim=1).to(torch.float32).clamp(min=1.0)
    # nansum of (NaN)^2 still drops the NaN cells (NaN squared is NaN).
    sigmas = torch.sqrt(torch.nansum(diff_sq, dim=1) / count)

    active = finite
    for _ in range(int(n_iterations)):
        lo = medians - n_sigma * sigmas
        hi = medians + n_sigma * sigmas
        # NaN comparisons short-circuit to False so we don't need to
        # AND with ``finite`` again, but the explicit AND with
        # ``active`` keeps once-rejected cells out for good (the
        # scalar form's iteration shrinks the working set monotonically).
        in_band = (s >= lo[:, None]) & (s <= hi[:, None])
        active = active & in_band
        s_clipped = torch.where(active, s, nan_value)
        medians = torch.nanmedian(s_clipped, dim=1).values
        diff_sq = (s_clipped - medians[:, None]) ** 2
        count = active.sum(dim=1).to(torch.float32).clamp(min=1.0)
        sigmas = torch.sqrt(torch.nansum(diff_sq, dim=1) / count)

    # Replace any NaN (all-NaN row) with 0.0 to match the scalar form's
    # all-NaN-returns-zero contract.
    sigmas = torch.where(
        torch.isnan(sigmas),
        torch.zeros((), dtype=torch.float32, device=s.device),
        sigmas,
    )
    return sigmas


def sigma_clipped_std(
    x: torch.Tensor,
    *,
    n_sigma: float = NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
    n_iterations: int = NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
    max_samples: Optional[int] = None,
    rng_seed: int = 0,
) -> float:
    """Compute the σ-clipped robust standard deviation of ``x``.

    Plan §3.6.9 line 985-993. Three iterations (default) of 3σ
    (default) clipping around the running median; final σ is the
    sqrt of the mean squared deviation from the *clipped* sample's
    median.

    NaN cells are excluded throughout (the upstream edge-mask sets
    cells outside the §3.6.5 G11 envelope to NaN; including them
    would inflate σ).

    Args:
        x: tensor of arbitrary shape; flattened internally.
        n_sigma: clip threshold in σ units. Default 3.0.
        n_iterations: number of clip iterations. Default 3.
        max_samples: optional upper bound on the number of cells used
            for the median + clipped-std estimate. When the flattened
            input has more than ``max_samples`` cells, a deterministic
            uniform random subsample of size ``max_samples`` is drawn
            (Generator(rng_seed); same seed → same subsample). Used
            by the chunk-8 streaming detector to bound the working
            set of ``torch.median`` (which sorts O(N) cells +
            allocates O(N) workspace) at production geometry — the
            per-kernel score has 2.05 M cells fp32 (~8 GiB if 2 GiB
            input × 4× sort workspace) which doesn't fit on an 11 GiB
            2080 Ti; ``max_samples=1_000_000`` caps the median
            workspace at ~16 MiB at the cost of σ̂ standard error
            ≈ σ / √(2 N_samples) ≈ 7e-4 σ — well below the EMA's
            cube-to-cube noise floor.
            Default None preserves chunk-1 behaviour (full input).
        rng_seed: seed for the subsample RNG. Same value → same
            subsample for a given input shape.

    Returns:
        ``float`` σ-clipped std. Returns 0.0 on an all-NaN or
        empty input (rather than raising — the caller can decide
        whether 0 means "edge-masked" or "degenerate cube").
    """
    if x.numel() == 0:
        return 0.0

    flat = x.reshape(-1)

    if max_samples is not None and flat.numel() > int(max_samples):
        gen = torch.Generator(device=flat.device)
        gen.manual_seed(int(rng_seed))
        idx = torch.randint(
            0, flat.numel(), (int(max_samples),),
            device=flat.device, generator=gen,
        )
        flat = flat[idx]
    # Dispatch through the batched primitive so the scalar path and
    # the batched-per-fdm path share one σ-clip implementation.
    sigmas = sigma_clipped_std_batched(
        flat.unsqueeze(0),
        n_sigma=n_sigma,
        n_iterations=n_iterations,
    )
    return float(sigmas[0].item())


def layer1_global_scalar(
    cube: torch.Tensor,
    *,
    n_sigma: float = NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
    n_iterations: int = NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
    max_samples: Optional[int] = None,
    rng_seed: int = 0,
) -> torch.Tensor:
    """Compute the per-fine_DM σ-clipped robust std for one cube.

    Reduces over ``[T_det, H, W]`` for each fine_DM trial; returns
    ``[N_fdm] float32``. Each fdm slice is independent — the noise
    floor varies with DM after dedispersion and we do *not* want to
    pool across DMs.

    Args:
        cube: ``[T_det, N_fdm, H, W]`` cube tensor (real-valued).
        max_samples: optional upper bound on the number of cells used
            for the per-fdm σ estimate; passes through to
            ``sigma_clipped_std``. At production geometry (T_det=256,
            N_grid=256) each fdm slab has 256³ = 16.8 M cells; a
            ``max_samples=1_000_000`` cap brings ``torch.median`` per
            iter from ~25 ms to ~1 ms with σ̂ standard error
            ≈ σ / √(2 × 1e6) ≈ 7e-4 σ. Default ``None`` preserves the
            chunk-1 behaviour (full slab).
        rng_seed: seed for the subsample RNG; the per-fdm call uses
            ``rng_seed + fdm`` so each fdm gets a distinct subsample.

    Returns:
        ``[N_fdm] float32`` per-fine_DM σ scalars. Suitable for
        broadcasting against the cube as
        ``cube_normalised = cube / sigma_layer1[None, :, None, None]``.
    """
    if cube.dim() != 4:
        raise ValueError(
            f"cube.dim()={cube.dim()}, expected 4 [T_det, N_fdm, H, W]"
        )
    T_det, n_fdm, H, W = cube.shape  # noqa: N806
    n_per_fdm = int(T_det) * int(H) * int(W)
    device = cube.device

    if (
        max_samples is not None
        and int(max_samples) > 0
        and n_per_fdm > int(max_samples)
    ):
        # Per-fdm independent subsamples with deterministic seeds
        # (matching the pre-batched ``rng_seed + fdm`` convention). The
        # subsamples are drawn via fancy indexing into the [T,F,H,W]
        # cube to avoid the ~1 GiB ``permute().contiguous()`` copy at
        # production geometry.
        m = int(max_samples)
        flat_idx = torch.empty((n_fdm, m), dtype=torch.int64, device=device)
        gen = torch.Generator(device=device)
        for fdm in range(n_fdm):
            gen.manual_seed(int(rng_seed) + int(fdm))
            flat_idx[fdm] = torch.randint(
                0, n_per_fdm, (m,), device=device, generator=gen,
            )
        samples = _gather_samples_per_fdm(
            cube, flat_idx, T_det=T_det, H=H, W=W,
        )
    else:
        # Full-slab batched path: reshape to [F, T*H*W] with a single
        # contiguous copy (small cubes in tests / cf32-audit benches).
        samples = (
            cube.permute(1, 0, 2, 3).contiguous().view(n_fdm, n_per_fdm)
        )
    return sigma_clipped_std_batched(
        samples,
        n_sigma=n_sigma,
        n_iterations=n_iterations,
    )


def _gather_samples_per_fdm(
    cube: torch.Tensor,
    flat_idx: torch.Tensor,
    *,
    T_det: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Gather subsamples for each fdm trial from a [T, F, H, W] cube.

    ``flat_idx`` has shape ``[N_fdm, M]`` with each entry in
    ``[0, T_det * H * W)`` (a flat (t, h, w) index into the fdm slab).
    Returns ``[N_fdm, M]`` with the same dtype as ``cube``. Uses fancy
    indexing so the cube's contiguous [T, F, H, W] layout is not
    materially copied.
    """
    n_fdm = flat_idx.shape[0]
    hw = int(H) * int(W)
    t_idx = (flat_idx // hw).to(torch.int64)
    rem = flat_idx % hw
    h_idx = (rem // int(W)).to(torch.int64)
    w_idx = (rem % int(W)).to(torch.int64)
    fdm_idx = torch.arange(
        n_fdm, dtype=torch.int64, device=cube.device
    ).unsqueeze(1).expand_as(flat_idx)
    return cube[t_idx, fdm_idx, h_idx, w_idx]


class Layer1State:
    """Stateful Layer-1 σ tracker with the 5-cube cold-start burn-in.

    Plan §3.6.9 lines 997-1011. State per fine_DM is a fixed-size
    scalar ring of length ``n_burnin_cubes`` (default 5); each cube
    pushes one σ scalar. Returns:

        cube_count < n_burnin_cubes : per-fdm median of the
                                      ring's values seen so far
        cube_count >= n_burnin_cubes : current cube's σ directly

    Memory footprint per GPU: ``N_fdm × n_burnin_cubes × 4 B`` ≈ 2 KB
    at default ops. State is host-RAM (numpy); no GPU sync per cube.

    Per plan §1013, state does NOT persist across ``cmd: stop`` /
    ``cmd: start`` cycles within a single resumed run if the gap is
    < ``5 × cube_cadence``; the bench resets via ``Layer1State.reset()``.
    """

    def __init__(
        self,
        n_fdm: int,
        *,
        n_burnin_cubes: int = NOISE_LAYER1_N_BURNIN_CUBES_DEFAULT,
        n_sigma: float = NOISE_SIGMA_CLIP_NSIGMA_DEFAULT,
        n_iterations: int = NOISE_SIGMA_CLIP_N_ITERATIONS_DEFAULT,
        max_samples: Optional[int] = None,
        rng_seed: int = 0,
    ) -> None:
        if n_fdm < 1:
            raise ValueError(f"n_fdm={n_fdm}, expected ≥ 1")
        if n_burnin_cubes < 1:
            raise ValueError(
                f"n_burnin_cubes={n_burnin_cubes}, expected ≥ 1"
            )
        self.n_fdm = int(n_fdm)
        self.n_burnin_cubes = int(n_burnin_cubes)
        self.n_sigma = float(n_sigma)
        self.n_iterations = int(n_iterations)
        self.max_samples = (
            int(max_samples) if max_samples is not None else None
        )
        self.rng_seed = int(rng_seed)
        # Per-fdm sigma history: deque of length ≤ n_burnin_cubes.
        self._history: list[Deque[float]] = [
            deque(maxlen=self.n_burnin_cubes) for _ in range(self.n_fdm)
        ]
        self._cube_count = 0
        # Cached subsample-index tensor (built lazily on first cube,
        # keyed by cube shape + device). Re-used across cubes so we
        # pay the randint generation cost exactly once per pipeline
        # lifetime, not per cube.
        self._subsample_idx: Optional[torch.Tensor] = None
        self._subsample_key: Optional[Tuple[int, int, int, int, str]] = None

    @property
    def cube_count(self) -> int:
        return self._cube_count

    @property
    def is_warming_up(self) -> bool:
        """Whether the Layer-1 burn-in is still active. NOTE: per plan
        §1011, Layer-1 burn-in does NOT set the warmup flag — only
        Layer-2 burn-in does. This predicate is exposed for telemetry /
        bench introspection only."""
        return self._cube_count < self.n_burnin_cubes

    def reset(self) -> None:
        """Clear all history; reset cube_count. Called on
        ``cmd: start --resume=false`` or after a stale-state restart."""
        for d in self._history:
            d.clear()
        self._cube_count = 0

    def _layer1_sigma_cached(self, cube: torch.Tensor) -> torch.Tensor:
        """Compute per-fdm Layer-1 σ on ``cube`` using a cached
        subsample-index tensor.

        The cached indices match the deterministic ``rng_seed + fdm``
        scheme used by the prior per-fdm Python loop, so successive
        cubes at the same geometry see the same subsample positions
        (which is statistically equivalent to fresh-per-cube
        subsampling for the σ-clip estimator and avoids ~3-5 ms of
        per-cube ``randint`` launches at production geometry).
        """
        if cube.dim() != 4:
            raise ValueError(
                f"cube.dim()={cube.dim()}, expected 4 [T_det, N_fdm, H, W]"
            )
        T_det, n_fdm, H, W = cube.shape
        if n_fdm != self.n_fdm:
            raise ValueError(
                f"cube N_fdm={n_fdm} != Layer1State.n_fdm={self.n_fdm}"
            )
        device = cube.device
        n_per_fdm = int(T_det) * int(H) * int(W)
        key = (
            int(T_det),
            int(self.n_fdm),
            int(H),
            int(W),
            str(device),
        )
        needs_subsample = (
            self.max_samples is not None
            and int(self.max_samples) > 0
            and n_per_fdm > int(self.max_samples)
        )
        if needs_subsample:
            if (
                self._subsample_idx is None
                or self._subsample_key != key
            ):
                m = int(self.max_samples)
                flat_idx = torch.empty(
                    (self.n_fdm, m), dtype=torch.int64, device=device,
                )
                gen = torch.Generator(device=device)
                for fdm in range(self.n_fdm):
                    gen.manual_seed(int(self.rng_seed) + int(fdm))
                    flat_idx[fdm] = torch.randint(
                        0, n_per_fdm, (m,),
                        device=device, generator=gen,
                    )
                self._subsample_idx = flat_idx
                self._subsample_key = key
            samples = _gather_samples_per_fdm(
                cube, self._subsample_idx,
                T_det=int(T_det), H=int(H), W=int(W),
            )
        else:
            samples = (
                cube.permute(1, 0, 2, 3)
                .contiguous()
                .view(self.n_fdm, n_per_fdm)
            )
        return sigma_clipped_std_batched(
            samples,
            n_sigma=self.n_sigma,
            n_iterations=self.n_iterations,
        )

    def update_and_query(
        self,
        cube: Optional[torch.Tensor] = None,
        *,
        per_fdm_sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Push this cube's per-fdm σ into the ring and return the per-fdm
        Layer-1 σ to use for normalising this cube.

        Pass exactly one of ``cube`` (cube tensor; layer1_global_scalar
        is run internally) or ``per_fdm_sigma`` (precomputed per-fdm σ
        tensor; lets benches feed σ from a separate computation path).

        Returns:
            ``[N_fdm] float32`` per-fine_DM Layer-1 σ to use for this
            cube. During burn-in (cube_count < n_burnin_cubes) this is
            the median of the 5 most recent per-cube σs; from cube 5+
            this is the current cube's σ unchanged.
        """
        if (cube is None) == (per_fdm_sigma is None):
            raise ValueError(
                "pass exactly one of cube= or per_fdm_sigma="
            )
        if cube is not None:
            sigma_this = self._layer1_sigma_cached(cube)
        else:
            assert per_fdm_sigma is not None
            sigma_this = per_fdm_sigma.to(torch.float32)
        if sigma_this.shape != (self.n_fdm,):
            raise ValueError(
                f"per-fdm sigma shape {tuple(sigma_this.shape)} != "
                f"(n_fdm={self.n_fdm},)"
            )

        device = sigma_this.device
        sigma_np = sigma_this.detach().cpu().numpy()
        for fdm in range(self.n_fdm):
            self._history[fdm].append(float(sigma_np[fdm]))

        if self._cube_count < self.n_burnin_cubes:
            # Use the median of the ring's contents seen so far per
            # plan §1007. The ring has cube_count + 1 entries at this
            # point (we just pushed); take the median of those.
            out_np = np.empty(self.n_fdm, dtype=np.float32)
            for fdm in range(self.n_fdm):
                out_np[fdm] = float(np.median(list(self._history[fdm])))
            self._cube_count += 1
            return torch.from_numpy(out_np).to(device)

        # Cube 5+ : just return the current cube's σ directly.
        self._cube_count += 1
        return sigma_this
