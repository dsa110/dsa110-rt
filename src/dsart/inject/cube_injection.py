"""Cube-level injection harness for the search-node detector unit-test path
(plan §8 line 2329 + M5 D1 / D8 / D12 / F10 / F11 locks).

This is the **post-imaging** peer of ``inject/online.py`` (M3, voltage-domain).
It synthesises a known signal **directly into a real-valued image cube** of
shape ``[T_det, N_fdm, N_grid, N_grid] float32``, bypassing every upstream
stage (no corr / RFI / cal / GEMM / gridder / dedisp / transport / RX-ring /
fine-DM combiner / imager). The cube is fed straight into
``Detector.forward()`` for unit-testing the detector + decoder + merger +
emitter independently of any upstream correctness — this is the
**primary detector correctness gate** per plan §8 line 2329 and the
single most important M5 development driver.

Locked semantics (M5_PLAN_FIXES.md):

  - **D1**: cube is real-valued ``float32``; bypasses every upstream stage.
    The cube *is* the dirty image the imager would have produced.
  - **D8**: thermal-noise background is iid Gaussian with σ = 1 per cell
    (so the analytic FAR ``0.5·erfc(θ/√2)`` applies cell-for-cell), and
    injection amplitudes are then injected at the requested SNR directly
    on top.
  - **D12**: v1 ships **boxcar (flat-top) profiles only** of duration
    ``width_samples``, peak amplitude ``snr / sqrt(width_samples)`` per
    cell. A width-`W` matched boxcar applied to a flat-top of duration `W`
    and peak `A` over unit-σ noise gives recovered SNR = ``A · √W = snr``
    exactly. Profile values other than ``"boxcar"`` raise
    ``NotImplementedError`` in v1.
  - **F10**: ``CubeInjectionConfig.l_pix`` / ``m_pix`` are integer
    cube-pixel indices into the ``[N_grid, N_grid]`` spatial axes
    (convention: ``(N_grid//2, N_grid//2)`` is the phase centre);
    distinct from the sky-cosine ``(l, m)`` used in §3.6.5 / §4.4.
    A classmethod ``from_lm_radians`` converts when the bench wires a
    voltage-fixture cross-check.
  - **F11**: the cube is already in post-Layer-1-normalised units (σ=1
    by construction), so the ``sigma_layer1`` argument passed to
    ``Detector.forward()`` is unity per ``(t, fdm)``.

The dataclass is deliberately small and serialisable so the operator-facing
``bench/cube_injection_detector.py`` (Chunk 5) can sweep ``snr ∈ {6, 8, 10,
12, 15} × width_samples ∈ {2, 4, 8, 16, 32, 64, 128}`` from a yaml manifest.
The serialiser is symmetric with ``inject/online.py::InjectionConfig``
where the fields overlap (``snr ↔ fluence`` is the only intentional
divergence — the cube-level injector works in SNR units directly because
the cube has σ=1, while the voltage-level injector works in physical
fluence units that pass through the fluence → SNR mapping of the imager).

Reference shape conventions (mirrors plan §3.6.10 / §3.6.11):

    cube[t_in_cube, fine_dm_idx, l_pix, m_pix] : float32
        T_det        — cube_axis 0, 0..T_det
        N_fdm        — cube_axis 1, 0..N_fine_DM
        N_grid       — cube_axes 2 and 3, square; (N_grid//2, N_grid//2)
                       == phase centre

The validity_mask returned alongside the cube is a ``[T_det, N_fdm] bool``
tensor of all-True (the synthetic cube is fully valid; no warmup or
corr-side dropout to model). The Detector consumes this mask; the
cube_injection bench passes it through unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Final, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "PROFILE_FAMILIES",
    "CubeInjectionConfig",
    "synthesise_cube",
    "add_injection",
    "iter_snr_width_grid",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


PROFILE_FAMILIES: Final[Tuple[str, ...]] = ("boxcar",)
"""v1 supported profile families per D12. ``"gaussian"`` / ``"scattered"``
families are deferred to v2 when the voltage-domain injector library
schema is finalised. v1 sticks to ``"boxcar"`` because it makes the
matched-filter exact-recovery math (recovered SNR = injected SNR for a
width-matched filter) directly visible in the operator-facing recovery
heatmap.
"""

_CUBE_DTYPE: Final[torch.dtype] = torch.float32
"""Cube dtype per D1 / D8 — float32 real (post-imager dirty image, σ=1
thermal noise per cell). The Detector consumes a fp16 view of this when
called with ``dtype=torch.float16`` per §3.6.11; the cube-injection
harness keeps the master copy at fp32 so the chunk-2 fp16-vs-fp32 parity
test (plan §8 line 2320) has a fp32 reference."""


# ---------------------------------------------------------------------------
# CubeInjectionConfig — F1 / F10 / D8 / D12
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubeInjectionConfig:
    """Single-injection config for the cube-level (post-imager) injector.

    Schema mirrors plan §8 line 2329 ``(l, m, fine_dm_idx, t_in_cube, snr,
    profile, width_samples)`` with the F10 disambiguation that ``l`` / ``m``
    are renamed to ``l_pix`` / ``m_pix`` (integer cube-pixel indices) for
    clarity vs the sky-cosine ``(l, m)`` used elsewhere.

    Args:
        l_pix: integer pixel index along the cube's first spatial axis
            (``cube.shape[2]``). Must satisfy ``0 ≤ l_pix < N_grid``.
            Convention: ``l_pix == N_grid // 2`` is the phase centre.
        m_pix: integer pixel index along the cube's second spatial axis
            (``cube.shape[3]``). Same range / centre convention as l_pix.
        fine_dm_idx: integer index into the cube's fine-DM axis
            (``cube.shape[1]``). The injection is centred at exactly one
            fine-DM trial — the cube-injection bench validates that the
            detector's K_dm boxcar correctly suppresses adjacent trials.
        t_in_cube: integer sample offset of the *peak* of the injected
            pulse from cube start (``cube.shape[0]``). The boxcar profile
            spans ``[t_in_cube - width_samples//2, t_in_cube +
            width_samples//2)`` (right-exclusive); for even widths the
            centre is biased left by half a sample (matches
            ``boxcar_via_cumsum``'s centring convention in
            ``forward.py``). Must satisfy
            ``width_samples//2 ≤ t_in_cube < T_det - width_samples//2``
            so the entire pulse lands inside the cube (the bench wires
            outside-edge cases via separate fixtures).
        snr: per-cell signal-to-noise ratio of the injection AFTER a
            width-matched boxcar filter is applied — i.e. the recovered
            SNR a perfect detector should report. Per D12, the boxcar
            profile's per-cell amplitude is ``snr / sqrt(width_samples)``,
            so a width-W matched filter recovers ``A · √W = snr``
            exactly.
        profile: profile family name. v1 supports only ``"boxcar"`` per
            D12; other values raise NotImplementedError at synthesis
            time.
        width_samples: width of the boxcar profile in cube-time samples
            (``t_int_search_us`` units; default 524.288 µs/sample).
            Per plan §8 line 2329, the bench sweeps
            ``{2, 4, 8, 16, 32, 64, 128}``. The K_time bank covers
            exactly these widths plus K_time=1 (single-sample). Must be
            > 0.
    """

    l_pix: int
    m_pix: int
    fine_dm_idx: int
    t_in_cube: int
    snr: float
    width_samples: int
    profile: str = "boxcar"

    def __post_init__(self) -> None:
        if not isinstance(self.l_pix, int) or self.l_pix < 0:
            raise ValueError(f"l_pix={self.l_pix!r}, expected non-negative int")
        if not isinstance(self.m_pix, int) or self.m_pix < 0:
            raise ValueError(f"m_pix={self.m_pix!r}, expected non-negative int")
        if not isinstance(self.fine_dm_idx, int) or self.fine_dm_idx < 0:
            raise ValueError(
                f"fine_dm_idx={self.fine_dm_idx!r}, expected non-negative int"
            )
        if not isinstance(self.t_in_cube, int) or self.t_in_cube < 0:
            raise ValueError(
                f"t_in_cube={self.t_in_cube!r}, expected non-negative int"
            )
        if not isinstance(self.width_samples, int) or self.width_samples < 1:
            raise ValueError(
                f"width_samples={self.width_samples!r}, expected positive int"
            )
        if not (math.isfinite(self.snr) and self.snr > 0):
            raise ValueError(f"snr={self.snr!r}, expected finite > 0")
        if self.profile not in PROFILE_FAMILIES:
            raise NotImplementedError(
                f"profile={self.profile!r} not in v1-supported "
                f"PROFILE_FAMILIES={PROFILE_FAMILIES}; v2 will add gaussian / "
                f"scattered. See M5_PLAN_FIXES.md D12."
            )

    @classmethod
    def from_lm_radians(
        cls,
        *,
        l_rad: float,
        m_rad: float,
        n_grid: int,
        pixel_scale_rad: float,
        fine_dm_idx: int,
        t_in_cube: int,
        snr: float,
        width_samples: int,
        profile: str = "boxcar",
    ) -> "CubeInjectionConfig":
        """Construct from sky-cosine ``(l, m)`` instead of pixel indices.

        Used by the Chunk-7 voltage-fixture cross-check that needs to
        translate a known burst's sky position to its cube-pixel
        landing. Uses the standard plan §3.6.5 / §3.6.11 mapping:

            l_pix = round(l_rad / pixel_scale_rad) + n_grid // 2
            m_pix = round(m_rad / pixel_scale_rad) + n_grid // 2

        with ``(N_grid // 2, N_grid // 2)`` == phase centre.

        Raises ValueError if the resulting pixel falls outside
        ``[0, n_grid)``.
        """
        if pixel_scale_rad <= 0:
            raise ValueError(
                f"pixel_scale_rad={pixel_scale_rad}, expected > 0"
            )
        l_pix = int(round(l_rad / pixel_scale_rad)) + n_grid // 2
        m_pix = int(round(m_rad / pixel_scale_rad)) + n_grid // 2
        if not (0 <= l_pix < n_grid):
            raise ValueError(
                f"l_pix={l_pix} from l_rad={l_rad}/pixel_scale={pixel_scale_rad}"
                f" falls outside [0, {n_grid})"
            )
        if not (0 <= m_pix < n_grid):
            raise ValueError(
                f"m_pix={m_pix} from m_rad={m_rad}/pixel_scale={pixel_scale_rad}"
                f" falls outside [0, {n_grid})"
            )
        return cls(
            l_pix=l_pix,
            m_pix=m_pix,
            fine_dm_idx=fine_dm_idx,
            t_in_cube=t_in_cube,
            snr=snr,
            width_samples=width_samples,
            profile=profile,
        )

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Cube synthesis
# ---------------------------------------------------------------------------


def _cube_pulse_window(
    t_in_cube: int,
    width_samples: int,
    t_det: int,
) -> Tuple[int, int]:
    """Return ``(t_lo, t_hi)`` (right-exclusive) for a centred boxcar of
    ``width_samples`` peaking at ``t_in_cube``, clipped to ``[0, t_det)``.

    Centring convention matches ``forward.py::boxcar_via_cumsum``: for
    even widths the centre is biased left by half a sample, so

        t_lo = t_in_cube - width // 2
        t_hi = t_in_cube + width - width // 2

    For odd ``width_samples`` the pulse is symmetric around ``t_in_cube``;
    for even widths the right half is one sample wider.
    """
    pad_left = width_samples // 2
    pad_right = width_samples - pad_left
    t_lo = max(0, t_in_cube - pad_left)
    t_hi = min(t_det, t_in_cube + pad_right)
    return (t_lo, t_hi)


def add_injection(
    cube: torch.Tensor,
    config: CubeInjectionConfig,
) -> torch.Tensor:
    """Add a single injection to ``cube`` in-place; return ``cube``.

    Raises ValueError if ``config`` indices fall outside the cube's
    actual dims.

    Per D12: amplitude per cell is ``config.snr / sqrt(config.width_samples)``;
    a width-matched boxcar over unit-σ background then recovers
    SNR = ``amplitude · √width = config.snr`` exactly.
    """
    if cube.dim() != 4:
        raise ValueError(
            f"cube.dim()={cube.dim()}, expected 4 [T_det, N_fdm, N_grid, N_grid]"
        )
    if cube.dtype != _CUBE_DTYPE:
        raise TypeError(
            f"cube.dtype={cube.dtype}, expected {_CUBE_DTYPE} (D8 lock)"
        )
    t_det, n_fdm, h, w = cube.shape
    if h != w:
        raise ValueError(f"cube spatial axes must be square; got H={h}, W={w}")
    if not (0 <= config.l_pix < h):
        raise ValueError(
            f"config.l_pix={config.l_pix} outside cube spatial axis [0, {h})"
        )
    if not (0 <= config.m_pix < w):
        raise ValueError(
            f"config.m_pix={config.m_pix} outside cube spatial axis [0, {w})"
        )
    if not (0 <= config.fine_dm_idx < n_fdm):
        raise ValueError(
            f"config.fine_dm_idx={config.fine_dm_idx} outside fine-DM axis "
            f"[0, {n_fdm})"
        )
    if not (0 <= config.t_in_cube < t_det):
        raise ValueError(
            f"config.t_in_cube={config.t_in_cube} outside time axis [0, {t_det})"
        )

    if config.profile != "boxcar":  # pragma: no cover  (D12 NotImplementedError)
        raise NotImplementedError(
            f"profile={config.profile!r}; v1 supports only 'boxcar' "
            f"(see D12 / PROFILE_FAMILIES)"
        )

    t_lo, t_hi = _cube_pulse_window(config.t_in_cube, config.width_samples, t_det)
    if t_hi <= t_lo:
        # Pulse is fully outside the cube; no-op.
        return cube

    amp = float(config.snr) / math.sqrt(float(config.width_samples))
    cube[t_lo:t_hi, config.fine_dm_idx, config.l_pix, config.m_pix] += amp
    return cube


def synthesise_cube(
    *,
    t_det: int,
    n_fdm: int,
    n_grid: int,
    injections: Tuple[CubeInjectionConfig, ...] = (),
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one synthetic detector cube + accompanying validity_mask
    + sigma_layer1 tensor, with zero or more injections added.

    Shape contract (D8 / F11):
        cube           : [T_det, N_fdm, N_grid, N_grid]  float32 (σ=1 noise)
        validity_mask  : [T_det, N_fdm]                  bool    (all True)
        sigma_layer1   : [T_det, N_fdm]                  float32 (all 1.0)

    The returned ``sigma_layer1`` tensor is unity per ``(t, fdm)`` per
    F11: the cube is already in post-Layer-1-normalised units (σ=1 by
    construction), so the per-cube σ-clipped scalar from
    ``noise_norm/layer1.py`` collapses to 1.0. Detector consumers just
    pass this through.

    The cube and accompanying tensors are constructed on CPU by default;
    pass ``device=torch.device("cuda:0")`` to materialise directly on GPU
    (used by the Chunk-5 throughput bench to skip a copy).

    Args:
        t_det: cube time depth (samples).
        n_fdm: fine-DM axis depth.
        n_grid: cube spatial side length (cube is square).
        injections: tuple of ``CubeInjectionConfig`` to add on top of
            the noise. Empty tuple → noise-only cube (used by the
            cube_injection bench's FAR sub-check, plan §8 line 2329).
        rng: optional ``numpy.random.Generator`` for reproducibility;
            defaults to a fresh generator (non-deterministic).
        noise_std: standard deviation of the iid Gaussian thermal-noise
            background. Defaults to 1.0 per D8; tests pin ``noise_std=0``
            for noise-free recovery checks.
        device: optional torch device.

    Returns:
        ``(cube, validity_mask, sigma_layer1)`` triple, ready to feed
        into ``Detector.forward(cube, validity_mask, sigma_layer1)``.
    """
    if t_det < 1 or n_fdm < 1 or n_grid < 1:
        raise ValueError(
            f"t_det / n_fdm / n_grid must all be ≥ 1; got "
            f"({t_det}, {n_fdm}, {n_grid})"
        )
    if noise_std < 0:
        raise ValueError(f"noise_std={noise_std}, expected ≥ 0")

    if rng is None:
        rng = np.random.default_rng()

    if noise_std == 0.0:
        cube_np = np.zeros((t_det, n_fdm, n_grid, n_grid), dtype=np.float32)
    else:
        cube_np = rng.standard_normal(
            (t_det, n_fdm, n_grid, n_grid), dtype=np.float32
        ).astype(np.float32, copy=False)
        if noise_std != 1.0:
            cube_np *= float(noise_std)

    cube = torch.from_numpy(cube_np)
    if device is not None:
        cube = cube.to(device)

    for inj in injections:
        add_injection(cube, inj)

    validity_mask = torch.ones((t_det, n_fdm), dtype=torch.bool, device=cube.device)
    sigma_layer1 = torch.ones((t_det, n_fdm), dtype=torch.float32, device=cube.device)
    return cube, validity_mask, sigma_layer1


# ---------------------------------------------------------------------------
# Bench helper: SNR × width sweep
# ---------------------------------------------------------------------------


def iter_snr_width_grid(
    *,
    snrs: Tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 15.0),
    widths: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
    l_pix: int,
    m_pix: int,
    fine_dm_idx: int,
    t_in_cube: int,
    profile: str = "boxcar",
) -> Tuple[CubeInjectionConfig, ...]:
    """Return the canonical ``snr × width`` sweep grid that the
    Chunk-5 ``bench/cube_injection_detector.py`` rasterises into the
    operator's recovery-rate heatmap (plan §8 line 2329).

    Defaults are the plan-pinned grid; benches can override either tuple.
    All cells share ``(l_pix, m_pix, fine_dm_idx, t_in_cube, profile)``
    so the heatmap dimensions are exactly ``len(snrs) × len(widths)``.
    """
    return tuple(
        CubeInjectionConfig(
            l_pix=l_pix,
            m_pix=m_pix,
            fine_dm_idx=fine_dm_idx,
            t_in_cube=t_in_cube,
            snr=float(snr),
            width_samples=int(w),
            profile=profile,
        )
        for snr in snrs
        for w in widths
    )
