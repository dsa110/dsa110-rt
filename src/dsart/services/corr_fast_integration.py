"""Fast correlator FULL integration service (M3 chunk 4; plan §4.2 step 1-7).

Where the chunk-2b spine (:mod:`dsart.services.corr_fast_compute`) wires
**unpack → cal → GEMM → Stokes I** end-to-end, this module assembles the
PRODUCTION pipeline:

  1. ``unpack_int4_split``  (M2 D15 / D16)
  2. ``RFIFlagger.flag_block`` → ``FlagBlockResult`` (M3 chunk 3c)
  3. **Voltage zero-fill** of flagged ``(ant, ch, pol)`` cells   (chunk 4)
     — drops flagged channels OUT of the GEMM so the resulting fast vis
     have zero contribution from RFI-contaminated voltages, instead of
     contaminating downstream cells. The zeroed voltage tensor is then
     fed forward; the fast-vis tile already reflects the F18 sign
     convention so no further correction is needed.
  4. ``apply_cal_split`` (with F21 DEC-phase cal; M3 chunk 1)
  5. ``FastCorrKernel.compute_split`` (M3 chunk 2a)
  6. ``stokes_i_pol_sum``                                          (chunk 2a)
  7. ``FastVisGridder.compute``                                    (chunk 3a)
  8. **Static-sky EMA subtraction** of the sparse-COO cube         (chunk 4)
  9. *(pluggable)* coarse-DM dedispersion              (chunk 3b lands here)
 10. *(pluggable)* stage-2 FIFO push                  (chunk 3b lands here)
 11. *(pluggable)* transport TX                          (M4a / chunk 8)

Chunks 9-11 are gated on incoming work and are exposed as Protocol-
shaped hooks (``CoarseDMStage``, ``Stage2FifoStage``, ``TransportTxStage``)
so chunk 3b's coarse-DM module + chunk 8's transport TX can plug in
without touching this orchestrator (Class-A boundary; see
``PARALLEL_AGENTS.md`` §3).

# Why a separate module from corr_fast_compute

``corr_fast_compute`` (chunk 2b) is the **spine** that proves chunks
1-2a compose end-to-end. It writes raw fast-vis tiles to disk for
inspection — it intentionally does not include the RFI flagger,
gridder, or static-sky stages. ``corr_fast_integration`` is the
PRODUCTION service shell that adds those + the pluggable hooks. The
two services share the per-block compute scaffolding but differ in
their pipeline graph + on-disk artefacts.

Per PARALLEL_AGENTS.md §4: this service runs on h01 GPU 0 with
``DSART_BUFFER_KEY_PREFIX=m3`` (which maps the legacy ``fada`` key to
``fa3a`` per the buffer-key prefix rule).

CLI:

    python -m dsart.services.corr_fast_integration \\
        [--fada-key fada]
        [--device auto]
        [--max-blocks N]
        [--t-int-fast-native 8]
        [--obs-dec-deg 53.848986]
        [--apply-cal /path/to/beamformer_weights_*.dat]
        [--cal-mode phase_only|full]
        [--cal-pol-swap]
        [--flagants /path/to/flagants.dat]
        [--n-grid 256]
        [--kernel-support 1]
        [--static-sky-alpha 0.001]
        [--static-sky-disabled]
        [--output-dir /tmp/dsart-fast-grid]
        [--blocks-output-mode full|first_tile_only|none]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from dsart.cal.cal_loader import (
    CalMode,
    FastCorrCalTensors,
    load_cal_with_dec_phase,
)
from dsart.common.config_loader import load
from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NPOL,
    PHI_LAT_OVRO_DEG,
    T_INT_FAST_NATIVE,
)
from dsart.grid import (
    FastVisGridder,
    SparsityPattern,
    build_pattern,
)
from dsart.rfi import (
    FlagBlockResult,
    RFIFlagger,
)
from dsart.services.corr_fast_kernel import (
    FastCorrKernel,
    stokes_i_pol_sum,
)
from dsart.services.slow_corr_kernel import (
    apply_cal_split,
    unpack_int4_split,
)


LOG = logging.getLogger("corr_fast_integration")

DEFAULT_CONFIG_PATH = Path("/home/ubuntu/proj/dsa110-rt/configs/config_corr.yaml")


# ---------------------------------------------------------------------------
# Pluggable stage protocols
# ---------------------------------------------------------------------------


class CoarseDMStage(Protocol):
    """Coarse-DM dedispersion hook (chunk 3b plug-in surface).

    Chunk 3b's ``CoarseDedisperser.dedisperse_cube`` will satisfy this
    protocol once it lands. Until then, a no-op stub is wired by
    ``IntegrationContext.coarse_dm = NoOpCoarseDM()``.
    """

    def dedisperse(
        self,
        gridded: torch.Tensor,
        *,
        block_n: int,
        chgroup: int,
    ) -> torch.Tensor:
        """Return dedispersed cube of shape ``(N_DM, n_fast_vis, N_filled)``.

        For the no-op stub, returns the input wrapped in a
        ``[1, ...]`` axis (single trivial DM trial).
        """
        ...


class Stage2FifoStage(Protocol):
    """Stage-2 FIFO push hook (chunk 3b plug-in surface)."""

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        """Push ``dedispersed`` into the FIFO; return any cubes that
        were evicted by this push (ready for transport TX).
        """
        ...


class TransportTxStage(Protocol):
    """Transport TX hook (chunk 8 plug-in surface)."""

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        """Transmit ``cubes_for_tx``; return count of cubes actually
        sent (== ``len(cubes_for_tx)`` for the production case;
        ``0`` for the no-op stub).
        """
        ...


# ---------------------------------------------------------------------------
# No-op stubs (Chunk 4 placeholder; chunk 3b / chunk 8 swap these out)
# ---------------------------------------------------------------------------


@dataclass
class NoOpCoarseDM:
    """Identity dedisperser: emits the gridded cube untouched on a
    single ``DM=0`` trial axis. Used until chunk 3b plugs in.

    Production replacement: ``dsart.coarse_dm.CoarseDedisperser``
    (Class A; chunk 3b owns it).
    """

    def dedisperse(
        self,
        gridded: torch.Tensor,
        *,
        block_n: int,
        chgroup: int,
    ) -> torch.Tensor:
        return gridded.unsqueeze(0)


@dataclass
class NoOpStage2Fifo:
    """Identity FIFO: every push immediately evicts the same cube
    (depth = 0). Used until chunk 3b plugs in.

    Production replacement: ``dsart.coarse_dm.Stage2Fifo`` (Class A;
    chunk 3b owns it).
    """

    def push(
        self,
        dedispersed: torch.Tensor,
        *,
        block_n: int,
    ) -> list[torch.Tensor]:
        return [dedispersed]


@dataclass
class NoOpTransportTx:
    """Drop-on-the-floor TX. Used until chunk 8 plugs in.

    Production replacement: ``dsart.transport.tx.TransportTx``
    (Class A; chunk 8 owns it).
    """

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        return 0


# ---------------------------------------------------------------------------
# Static-sky EMA subtraction
# ---------------------------------------------------------------------------


@dataclass
class StaticSkyEMA:
    """Per-cell exponential moving average of the gridded cube.

    Subtracts the running mean from each new cube before forwarding,
    then updates the running mean with the (post-subtraction) cube
    blended in at rate ``alpha``.

    The EMA is computed in **complex** to preserve the F20-aligned
    visibility phase; the subtraction is therefore complex too.
    Continuum sources at fixed sky positions decorrelate to a
    near-constant per-cell value across cubes (modulo Earth-rotation
    fringe winding, which the F21 DEC-phase cal already partially
    de-rotates), so the EMA learns + subtracts them.

    Args:
        alpha: EMA smoothing factor in (0, 1]. The EMA half-life is
            ``ln(0.5) / ln(1-alpha) ≈ 0.69 / alpha`` cubes. Default
            ``0.001`` → ~700-cube half-life (~7 s at 134 ms cube
            cadence).
        warmup_cubes: number of cubes at the start during which we
            BUILD the EMA but do NOT subtract (so the first few
            cubes are not artificially zeroed by the cold EMA).
    """

    alpha: float = 0.001
    warmup_cubes: int = 8

    _running_mean: torch.Tensor | None = field(
        default=None, init=False, repr=False,
    )
    _cubes_seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(
                f"StaticSkyEMA.alpha={self.alpha}, expected (0, 1]"
            )
        if self.warmup_cubes < 0:
            raise ValueError(
                f"StaticSkyEMA.warmup_cubes={self.warmup_cubes}, "
                f"expected >= 0"
            )

    @property
    def cubes_seen(self) -> int:
        return self._cubes_seen

    @property
    def in_warmup(self) -> bool:
        return self._cubes_seen < self.warmup_cubes

    def reset(self) -> None:
        self._running_mean = None
        self._cubes_seen = 0

    def apply(self, gridded: torch.Tensor) -> torch.Tensor:
        """Subtract the running mean from ``gridded`` and update the EMA.

        ``gridded`` is expected to be ``(n_fast_vis, N_filled)`` complex64
        (the output of :meth:`FastVisGridder.compute`). The EMA is
        kept at ``(N_filled,)`` complex64 — averaged over the
        ``n_fast_vis`` axis on the way in, broadcast back on the way
        out.
        """
        if not gridded.is_complex():
            raise TypeError(
                f"StaticSkyEMA.apply: gridded must be complex; got "
                f"{gridded.dtype}"
            )
        if gridded.ndim != 2:
            raise ValueError(
                f"StaticSkyEMA.apply: gridded must be 2D "
                f"(n_fast_vis, N_filled); got "
                f"{tuple(gridded.shape)}"
            )

        per_cell_mean = gridded.mean(dim=0)                              # (N_filled,)

        if self._running_mean is None:
            self._running_mean = per_cell_mean.clone().detach()
            out = gridded.clone()                                        # cold start: pass through
        elif self.in_warmup:
            out = gridded.clone()                                        # build EMA, don't subtract
            self._running_mean = (
                (1.0 - self.alpha) * self._running_mean
                + self.alpha * per_cell_mean
            )
        else:
            out = gridded - self._running_mean.unsqueeze(0)              # subtract, then update
            self._running_mean = (
                (1.0 - self.alpha) * self._running_mean
                + self.alpha * per_cell_mean
            )

        self._cubes_seen += 1
        return out


# ---------------------------------------------------------------------------
# RFI mask → voltage-cube zero-fill
# ---------------------------------------------------------------------------


def apply_rfi_mask_to_voltages(
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    rfi_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero ``(ant, ch, pol)`` cells flagged by the RFI flagger.

    Args:
        real_v, imag_v: voltage tensors in M2 GEMM layout
            ``(NCHAN, NTIMES_PER_PACKET, NPOL, NPACKETS_PER_BLOCK,
            NANTS)`` (any float dtype).
        rfi_mask: bool tensor ``(NANTS, NCHAN, NPOL)`` from
            :class:`FlagBlockResult.mask`. **True == flagged**.

    Returns:
        Pair of voltage tensors with flagged cells set to zero
        in-place. The mask is broadcast to the GEMM layout via
        ``(NCHAN, 1, NPOL, 1, NANTS)``.

    Notes
    -----
    Zero-filling at the voltage level (rather than masking the
    visibilities) means the GEMM never sees the bad samples — so any
    baseline that touches a flagged antenna also has zero contribution
    on those (ch, pol) cells, which is the physically correct
    treatment. The cost is one ``mul_`` over the voltage cube: ~2 GB
    of fp16, ~150 µs on a 2080 Ti.
    """
    if rfi_mask.dtype != torch.bool:
        raise TypeError(
            f"rfi_mask must be bool; got {rfi_mask.dtype}"
        )
    n_ant_m, n_ch_m, n_pol_m = rfi_mask.shape
    n_ch_v, _ntp_v, n_pol_v, _np_v, n_ant_v = real_v.shape
    if (n_ant_m, n_ch_m, n_pol_m) != (n_ant_v, n_ch_v, n_pol_v):
        raise ValueError(
            f"rfi_mask shape {tuple(rfi_mask.shape)} != voltage layout "
            f"(NANTS, NCHAN, NPOL) = ({n_ant_v}, {n_ch_v}, {n_pol_v})"
        )

    # Mask reshape → (NCHAN, 1, NPOL, 1, NANTS) — broadcast against
    # voltage GEMM layout (NCHAN, NTIMES_PER_PACKET, NPOL,
    # NPACKETS_PER_BLOCK, NANTS).
    mask_bcast = (
        rfi_mask.permute(1, 2, 0)                                        # (NCHAN, NPOL, NANTS)
        .unsqueeze(1).unsqueeze(3)                                       # (NCHAN, 1, NPOL, 1, NANTS)
        .to(real_v.device)
    )
    keep = (~mask_bcast).to(real_v.dtype)                                # 0/1 fp{16,32}
    real_v.mul_(keep)
    imag_v.mul_(keep)
    return real_v, imag_v


# ---------------------------------------------------------------------------
# Per-block integration context (held across the run loop)
# ---------------------------------------------------------------------------


@dataclass
class FastIntegrationConfig:
    """All knobs for one fast-corr integration run. Held by reference;
    mutate fields between calls only when not in flight.

    Args:
        chgroup: corr-node chgroup index (0-15).
        obs_dec_rad: observing-source declination (rad). Drives F21
            cal-loader and the gridder's pattern lookup.
        n_grid: side length of the gridded cube. Default 256.
        kernel_support: gridding kernel half-width (cells). Default
            1 (3x3 pillbox).
        t_int_fast_native: fast-corr integration depth in NATIVE
            samples per fast-vis tile. Default 8 (262.144 µs); burst
            test override is 32 (1048.576 µs / 4× cadence).
        cal_path / cal_mode / cal_pol_swap: F21 cal options.
        flagants_path: optional flagants.dat path.
        rfi_enabled: kill-switch for the whole RFI flagger stage
            (default True). Useful for synth-data tests where the RFI
            flagger would mistake injected signals for narrowband CW.
        static_sky_alpha / static_sky_warmup_cubes / static_sky_disabled:
            EMA controls. ``static_sky_disabled=True`` is a kill-
            switch (useful for tests + the chunk-5 0319 continuum
            bench where the brightest source IS the static sky).
        rfi_mask_voltage_zero_fill: route RFI mask through the
            voltage zero-fill (default True). Set to False for tests
            that want to verify the RFI flagger fired without
            actually altering the GEMM result.
    """

    chgroup: int
    obs_dec_rad: float
    n_grid: int = 256
    kernel_support: int = 1
    t_int_fast_native: int = T_INT_FAST_NATIVE
    cal_path: Path | None = None
    cal_mode: str = CalMode.PHASE_ONLY
    cal_pol_swap: bool = False
    flagants_path: Path | None = None
    rfi_enabled: bool = True
    rfi_mask_voltage_zero_fill: bool = True
    static_sky_alpha: float = 0.001
    static_sky_warmup_cubes: int = 8
    static_sky_disabled: bool = False


@dataclass
class IntegrationContext:
    """Mutable per-run state: kernel, cal tensors, RFI flagger,
    gridder, static-sky EMA, pluggable stages.

    Constructed once at service startup (after the first fada header
    is read so the device is committed); held for the lifetime of the
    run.
    """

    cfg: FastIntegrationConfig
    device: torch.device
    voltage_dtype: torch.dtype

    kernel: FastCorrKernel
    cal: FastCorrCalTensors | None
    rfi_flagger: RFIFlagger | None
    gridder: FastVisGridder
    static_sky: StaticSkyEMA | None

    coarse_dm: CoarseDMStage = field(default_factory=NoOpCoarseDM)
    stage2_fifo: Stage2FifoStage = field(default_factory=NoOpStage2Fifo)
    transport_tx: TransportTxStage = field(default_factory=NoOpTransportTx)


@dataclass
class IntegrationOutput:
    """Per-block output of :func:`process_block`.

    Args:
        gridded_minus_sky: ``(n_fast_vis, N_filled)`` complex64 — the
            gridded cube AFTER static-sky subtraction. None if the
            block was dropped (malformed / empty).
        rfi: :class:`FlagBlockResult` from the RFI flagger this
            block, or ``None`` if RFI was disabled.
        n_tx: number of cubes the transport stage actually sent this
            block.
        block_n: 1-indexed block counter (mirrors fada page sequence).
    """

    gridded_minus_sky: torch.Tensor | None
    rfi: FlagBlockResult | None
    n_tx: int
    block_n: int


# ---------------------------------------------------------------------------
# Per-block compute (pure; the orchestrator + tests both call this)
# ---------------------------------------------------------------------------


def process_block(
    raw: np.ndarray,
    *,
    ctx: IntegrationContext,
    block_n: int,
) -> IntegrationOutput:
    """Run the chunk-4 pipeline on ONE fada-block-worth of raw bytes.

    This is the entire fast-corr integration graph in one function.
    The PSRDADA loop in :func:`run` calls this per block; tests
    construct a synthetic ``raw`` byte array and call this directly.

    Pipeline (mirrors module docstring step list):

        1. unpack int4 → (real_v, imag_v) GEMM-layout
        2. RFI flag_block(real, imag) → FlagBlockResult (if cfg.rfi_enabled)
        3. Apply RFI mask to voltages (if cfg.rfi_mask_voltage_zero_fill)
        4. apply_cal_split (if ctx.cal is not None)
        5. FastCorrKernel.compute_split → (n_fv, NBASE, NCHAN, 2)
        6. stokes_i_pol_sum → (n_fv, NBASE, NCHAN)
        7. FastVisGridder.compute → (n_fv, N_filled) complex64
        8. StaticSkyEMA.apply (if cfg.static_sky_disabled is False)
        9. coarse_dm.dedisperse → (N_DM, n_fv, N_filled)
       10. stage2_fifo.push → list[evictees]
       11. transport_tx.transmit → n_sent

    Returns
    -------
    :class:`IntegrationOutput`
        Per-block diagnostic + output bundle.
    """
    # 1. Unpack
    real_v, imag_v = unpack_int4_split(
        raw, device=ctx.device, out_dtype=ctx.voltage_dtype,
    )

    # 2. RFI flag (before cal — flags should be on raw data + DC pol
    # so that any cal-induced dynamic range shifts can't mask them).
    rfi_result: FlagBlockResult | None = None
    if ctx.rfi_flagger is not None and ctx.cfg.rfi_enabled:
        rfi_result = ctx.rfi_flagger.flag_block(real_v, imag_v)

        # 3. Voltage zero-fill of flagged cells (full-cube; bandpass
        # outliers and SK are on per-cube SCALAR statistics so the
        # mask is constant-in-time across the cube — we apply it to
        # all (NTIMES, NPACKETS) samples uniformly).
        if ctx.cfg.rfi_mask_voltage_zero_fill and rfi_result.mask.any():
            real_v, imag_v = apply_rfi_mask_to_voltages(
                real_v, imag_v, rfi_result.mask,
            )

    # 4. Cal apply with F21 DEC-phase fold
    if ctx.cal is not None:
        real_v_c, imag_v_c = apply_cal_split(
            real_v, imag_v, ctx.cal.cal_real, ctx.cal.cal_imag,
        )
        del real_v, imag_v
        real_v, imag_v = real_v_c, imag_v_c

    # 5. + 6. Fast-corr GEMM + Stokes I
    vis_2pol = ctx.kernel.compute_split(real_v, imag_v)
    del real_v, imag_v
    vis_stokes_i = stokes_i_pol_sum(vis_2pol)                            # (n_fv, NBASE, NCHAN)
    del vis_2pol

    # 7. Gridder (sparse-COO)
    gridded = ctx.gridder.compute(vis_stokes_i)                          # (n_fv, N_filled) complex64
    del vis_stokes_i

    # 8. Static-sky EMA subtraction
    if ctx.static_sky is not None and not ctx.cfg.static_sky_disabled:
        gridded_minus_sky = ctx.static_sky.apply(gridded)
    else:
        gridded_minus_sky = gridded

    # 9. Coarse-DM dedispersion (no-op stub today)
    dedispersed = ctx.coarse_dm.dedisperse(
        gridded_minus_sky, block_n=block_n, chgroup=ctx.cfg.chgroup,
    )

    # 10. Stage-2 FIFO push (no-op stub today)
    cubes_for_tx = ctx.stage2_fifo.push(dedispersed, block_n=block_n)

    # 11. Transport TX (no-op stub today)
    rfi_warmup_flag = bool(rfi_result.warmup) if rfi_result is not None else False
    n_tx = ctx.transport_tx.transmit(
        cubes_for_tx, block_n=block_n, rfi_warming_up=rfi_warmup_flag,
    )

    return IntegrationOutput(
        gridded_minus_sky=gridded_minus_sky,
        rfi=rfi_result,
        n_tx=n_tx,
        block_n=block_n,
    )


# ---------------------------------------------------------------------------
# Construction helpers (also called by tests + benches directly)
# ---------------------------------------------------------------------------


def _key_to_int(key_str: str) -> int:
    if len(key_str) != 4:
        raise ValueError(f"buffer key must be 4 chars, got {key_str!r}")
    return int(f"0x{key_str}", 16)


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_fada_buffer_size(cfg_path: Path) -> int:
    cfg = load(cfg_path)
    return int(cfg["buffers"]["fada"]["bytes_per_block"])


class _StopRequested(Exception):
    """SIGTERM / SIGINT → request main-loop stop."""


def _install_signals(state: dict[str, Any]) -> None:
    def handle(signum, _frame):
        LOG.info("received signal %d, requesting stop", signum)
        state["stop"] = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def _build_cal(
    cfg: FastIntegrationConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FastCorrCalTensors | None:
    """Load + log the F21 cal tensors. Returns None if no cal_path set."""
    if cfg.cal_path is None:
        return None
    out = load_cal_with_dec_phase(
        cfg.cal_path,
        chgroup=cfg.chgroup,
        obs_dec_rad=cfg.obs_dec_rad,
        cal_mode=cfg.cal_mode,
        pol_swap=cfg.cal_pol_swap,
        device=device,
        dtype=dtype,
    )
    LOG.info(
        "cal: %s mode=%s pol_swap=%s n_flagged=%d mag_p50=%.3g mag_p99=%.3g",
        out.info["cal_path"], out.info["cal_mode"], out.info["pol_swap"],
        out.info["n_flagged"],
        out.info.get("cal_mag_p50", float("nan")),
        out.info.get("cal_mag_p99", float("nan")),
    )
    LOG.info(
        "F21 DEC-phase: chgroup=%d obs_dec_deg=%.4f phi_lat_deg=%.3f "
        "(delta_dec_deg=%+.4f)",
        cfg.chgroup, out.info["obs_dec_deg"], PHI_LAT_OVRO_DEG,
        out.info["obs_dec_deg"] - PHI_LAT_OVRO_DEG,
    )
    return out


def _build_gridder(
    cfg: FastIntegrationConfig,
    *,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    is_core_baseline_mask: np.ndarray | None,
    device: torch.device,
) -> tuple[SparsityPattern, FastVisGridder]:
    pattern = build_pattern(
        antpos_e, antpos_n,
        chgroup=cfg.chgroup,
        dec_deg=math.degrees(cfg.obs_dec_rad),
        n_grid=cfg.n_grid,
        kernel_support=cfg.kernel_support,
        is_core_baseline_mask=is_core_baseline_mask,
    )
    LOG.info(
        "sparsity pattern: chgroup=%d obs_dec_deg=%.4f n_grid=%d "
        "kernel_support=%d → n_filled=%d (id=%s)",
        cfg.chgroup, math.degrees(cfg.obs_dec_rad), cfg.n_grid,
        cfg.kernel_support, pattern.n_filled, pattern.pattern_id[:16],
    )
    gridder = FastVisGridder.from_pattern(
        pattern, antpos_e, antpos_n,
        is_core_baseline_mask=is_core_baseline_mask,
        device=device,
    )
    return pattern, gridder


def build_context(
    cfg: FastIntegrationConfig,
    *,
    device: torch.device,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    is_core_baseline_mask: np.ndarray | None = None,
    coarse_dm: CoarseDMStage | None = None,
    stage2_fifo: Stage2FifoStage | None = None,
    transport_tx: TransportTxStage | None = None,
) -> IntegrationContext:
    """One-shot context builder. Tests call this directly with synthetic
    antpos arrays; the service shell calls this after reading the fada
    header to commit the device.

    The ``coarse_dm`` / ``stage2_fifo`` / ``transport_tx`` parameters
    default to no-op stubs (chunk 4 placeholder); chunk 3b / chunk 8
    will pass real implementations in.
    """
    voltage_dtype: torch.dtype = (
        torch.float32 if (
            cfg.cal_path is not None and cfg.cal_mode == CalMode.FULL
        ) else torch.float16
    )

    kernel = FastCorrKernel(
        device=device,
        t_int_fast_native=cfg.t_int_fast_native,
    )
    LOG.info(
        "FastCorrKernel ready: t_int_fast_native=%d (%.3f µs cadence) "
        "→ n_fast_vis_per_block=%d",
        cfg.t_int_fast_native,
        cfg.t_int_fast_native * NATIVE_SAMPLE_US,
        kernel.n_fast_vis_per_full_block,
    )

    cal = _build_cal(cfg, device=device, dtype=voltage_dtype)

    rfi_flagger: RFIFlagger | None = None
    if cfg.rfi_enabled:
        rfi_flagger = RFIFlagger(
            flagants_path=cfg.flagants_path,
            device=device,
        )
        LOG.info(
            "RFIFlagger ready: warmup_cubes=%d sk_far=%.3g "
            "bandpass_k=%.2f group_k=%.2f",
            rfi_flagger.warmup_cubes, rfi_flagger._sk_far,
            rfi_flagger._bandpass_k, rfi_flagger._group_k,
        )
    else:
        LOG.info("RFIFlagger DISABLED (cfg.rfi_enabled=False)")

    _pattern, gridder = _build_gridder(
        cfg,
        antpos_e=antpos_e, antpos_n=antpos_n,
        is_core_baseline_mask=is_core_baseline_mask,
        device=device,
    )

    static_sky: StaticSkyEMA | None = None
    if not cfg.static_sky_disabled:
        static_sky = StaticSkyEMA(
            alpha=cfg.static_sky_alpha,
            warmup_cubes=cfg.static_sky_warmup_cubes,
        )
        LOG.info(
            "StaticSkyEMA ready: alpha=%.4g warmup_cubes=%d",
            static_sky.alpha, static_sky.warmup_cubes,
        )
    else:
        LOG.info("StaticSkyEMA DISABLED (cfg.static_sky_disabled=True)")

    return IntegrationContext(
        cfg=cfg,
        device=device,
        voltage_dtype=voltage_dtype,
        kernel=kernel,
        cal=cal,
        rfi_flagger=rfi_flagger,
        gridder=gridder,
        static_sky=static_sky,
        coarse_dm=coarse_dm if coarse_dm is not None else NoOpCoarseDM(),
        stage2_fifo=(
            stage2_fifo if stage2_fifo is not None else NoOpStage2Fifo()
        ),
        transport_tx=(
            transport_tx if transport_tx is not None else NoOpTransportTx()
        ),
    )


# ---------------------------------------------------------------------------
# Antpos / core-baseline accessor (reads the same source the gridder
# pattern build uses; centralised here so the service + tests + benches
# don't drift).
# ---------------------------------------------------------------------------


def load_antpos_from_cal_blob(
    cal_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load antenna positions + the core-baseline mask from a cal blob.

    The legacy ``beamformer_weights_*.dat`` blob carries (E, N) antpos
    in its header — the gridder needs the same arrays the cal was
    derived against, so reading both from the same blob guarantees
    the antpos hash on the SparsityPattern matches.

    Returns
    -------
    (antpos_e, antpos_n, is_core_baseline_mask) : tuple
        - antpos_e, antpos_n : (NANTS,) float32 arrays
        - is_core_baseline_mask : (NBASE,) bool — True for cross-
          baselines where both antennas are in the 82-ant core, False
          for outriggers or autos. Mirrors plan §3 line 452 + the
          test_sparsity_pattern.py ``_core_baseline_mask`` helper.
    """
    from dsart.cal.bf_weights import load_bf_weights

    bf = load_bf_weights(cal_path)
    return (
        np.asarray(bf.antpos_e, dtype=np.float32),
        np.asarray(bf.antpos_n, dtype=np.float32),
        _build_core_baseline_mask(n_core=82),
    )


def _build_core_baseline_mask(*, n_core: int = 82) -> np.ndarray:
    """``(NBASE,) bool`` mask: True iff both antennas are in [0, n_core).

    Mirrors plan §3 line 452 + ``test_sparsity_pattern.py``'s helper
    of the same name. The DSA-110 antenna ordering puts the 82 core
    antennas at indices [0, 82) and the 14 outriggers at [82, 96);
    the gridder excludes outrigger-touching baselines.
    """
    nbase = NANTS * (NANTS + 1) // 2
    mask = np.zeros(nbase, dtype=bool)
    k = 0
    for a in range(NANTS):
        for b in range(a + 1):
            mask[k] = (a < n_core) and (b < n_core)
            k += 1
    return mask


# ---------------------------------------------------------------------------
# Service shell (PSRDADA fada → in-memory pipeline → on-disk artefacts)
# ---------------------------------------------------------------------------


def _serialise_block(
    output_dir: Path,
    block_n: int,
    *,
    out: IntegrationOutput,
    blocks_output_mode: str,
) -> None:
    """Write per-block artefacts (gridded cube + meta json) to disk.

    Modes:
      * ``full`` — torch.save the entire ``(n_fv, N_filled)`` tensor.
      * ``first_tile_only`` — only fast-vis tile 0.
      * ``none`` — meta json only (gridded cube discarded).
    """
    if out.gridded_minus_sky is None:
        return

    block_dir = output_dir / f"block_{block_n:06d}"
    block_dir.mkdir(exist_ok=True)

    if blocks_output_mode == "first_tile_only":
        out_tensor = out.gridded_minus_sky[0:1].cpu().contiguous()
        torch.save(out_tensor, block_dir / "gridded_minus_sky.pt")
    elif blocks_output_mode == "full":
        out_tensor = out.gridded_minus_sky.cpu().contiguous()
        torch.save(out_tensor, block_dir / "gridded_minus_sky.pt")
    elif blocks_output_mode == "none":
        out_tensor = None
    else:
        raise ValueError(
            f"blocks_output_mode={blocks_output_mode!r} (expected "
            f"full | first_tile_only | none)"
        )

    meta = {
        "block_n": block_n,
        "n_fast_vis_total": int(out.gridded_minus_sky.shape[0]),
        "n_filled": int(out.gridded_minus_sky.shape[1]),
        "n_fast_vis_written": (
            int(out_tensor.shape[0]) if out_tensor is not None else 0
        ),
        "rfi_flag_fraction_total": (
            float(out.rfi.flag_fraction_total) if out.rfi is not None else None
        ),
        "rfi_warmup": (
            bool(out.rfi.warmup) if out.rfi is not None else None
        ),
        "n_tx": out.n_tx,
        "blocks_output_mode": blocks_output_mode,
    }
    (block_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def run(
    fada_key: int,
    output_dir: Path,
    device: torch.device,
    cfg: FastIntegrationConfig,
    *,
    max_blocks: int | None = None,
    expected_fada_bytes: int = FADA_BYTES_PER_BLOCK,
    blocks_output_mode: str = "first_tile_only",
    coarse_dm: CoarseDMStage | None = None,
    stage2_fifo: Stage2FifoStage | None = None,
    transport_tx: TransportTxStage | None = None,
) -> dict[str, Any]:
    """Connect to PSRDADA fada, run the integration pipeline per block,
    optionally serialise per-block artefacts.

    Returns a summary dict including:
        {n_blocks_in, n_blocks_processed, n_dropped, elapsed_s,
         ms_per_block_p50, ms_per_block_p99, output_dir,
         t_int_fast_native, n_fast_vis_per_block,
         rfi_flag_fraction_p50, rfi_flag_fraction_p99,
         n_tx_total, n_grid, n_filled}
    """
    from psrdada import Reader

    if blocks_output_mode not in ("full", "first_tile_only", "none"):
        raise ValueError(
            f"blocks_output_mode must be full|first_tile_only|none, "
            f"got {blocks_output_mode!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {"stop": False}
    _install_signals(state)

    LOG.info(
        "connecting fada=0x%x device=%s chgroup=%d obs_dec_deg=%.4f",
        fada_key, device, cfg.chgroup, math.degrees(cfg.obs_dec_rad),
    )
    LOG.info("output: %s mode=%s", output_dir, blocks_output_mode)
    reader = Reader(fada_key)

    n_in = 0
    n_processed = 0
    n_drop = 0
    n_tx_total = 0

    try:
        fada_header = reader.getHeader()
        LOG.info("fada header: %d keys (UTC_START=%s)",
                 len(fada_header), fada_header.get("UTC_START", "?"))

        if cfg.cal_path is None:
            raise RuntimeError(
                "corr_fast_integration.run() requires --apply-cal: the "
                "antpos arrays + core-baseline mask are sourced from "
                "the cal blob (load_antpos_from_cal_blob) so the "
                "gridder pattern matches the cal."
            )
        antpos_e, antpos_n, core_mask = load_antpos_from_cal_blob(cfg.cal_path)
        ctx = build_context(
            cfg, device=device,
            antpos_e=antpos_e, antpos_n=antpos_n,
            is_core_baseline_mask=core_mask,
            coarse_dm=coarse_dm,
            stage2_fifo=stage2_fifo,
            transport_tx=transport_tx,
        )

        per_block_ms: list[float] = []
        per_block_flag_frac: list[float] = []
        t_start = time.monotonic()

        while not state["stop"]:
            try:
                page = reader.getNextPage()
            except StopIteration:
                LOG.info("fada reader StopIteration (EOD)")
                break
            if reader.isEndOfData:
                LOG.info("fada EOD flag set; draining final block")
            n_in += 1

            t_block_start = time.monotonic()

            page_arr = np.asarray(page)
            if page_arr.nbytes != expected_fada_bytes:
                LOG.error(
                    "fada block #%d wrong size: got=%d expected=%d; skipping",
                    n_in, page_arr.nbytes, expected_fada_bytes,
                )
                reader.markCleared()
                n_drop += 1
                if max_blocks is not None and n_in >= max_blocks:
                    break
                continue

            out = process_block(page_arr, ctx=ctx, block_n=n_in)
            n_tx_total += out.n_tx

            if blocks_output_mode != "none":
                _serialise_block(
                    output_dir, n_in,
                    out=out, blocks_output_mode=blocks_output_mode,
                )

            n_processed += 1
            if out.rfi is not None:
                per_block_flag_frac.append(float(out.rfi.flag_fraction_total))
            reader.markCleared()
            del out

            t_block_end = time.monotonic()
            per_block_ms.append((t_block_end - t_block_start) * 1000.0)

            if n_in % 16 == 0:
                LOG.info(
                    "processed n_in=%d n_processed=%d n_drop=%d n_tx=%d "
                    "last_block=%.1fms",
                    n_in, n_processed, n_drop, n_tx_total, per_block_ms[-1],
                )

            if max_blocks is not None and n_in >= max_blocks:
                LOG.info("hit --max-blocks=%d; stopping", max_blocks)
                break
            if reader.isEndOfData:
                LOG.info("fada EOD; loop done")
                break

        elapsed = time.monotonic() - t_start
        ms_p50 = float(np.median(per_block_ms)) if per_block_ms else float("nan")
        ms_p99 = float(np.percentile(per_block_ms, 99)) if per_block_ms else float("nan")
        ff_p50 = float(np.median(per_block_flag_frac)) if per_block_flag_frac else float("nan")
        ff_p99 = float(np.percentile(per_block_flag_frac, 99)) if per_block_flag_frac else float("nan")
        LOG.info(
            "summary: n_in=%d n_processed=%d n_drop=%d n_tx=%d elapsed=%.1fs "
            "p50=%.1fms p99=%.1fms ff_p50=%.3f ff_p99=%.3f",
            n_in, n_processed, n_drop, n_tx_total, elapsed,
            ms_p50, ms_p99, ff_p50, ff_p99,
        )
        return {
            "n_blocks_in": n_in,
            "n_blocks_processed": n_processed,
            "n_dropped": n_drop,
            "n_tx_total": n_tx_total,
            "elapsed_s": elapsed,
            "ms_per_block_p50": ms_p50,
            "ms_per_block_p99": ms_p99,
            "rfi_flag_fraction_p50": ff_p50,
            "rfi_flag_fraction_p99": ff_p99,
            "output_dir": str(output_dir),
            "t_int_fast_native": cfg.t_int_fast_native,
            "n_fast_vis_per_block": ctx.kernel.n_fast_vis_per_full_block,
            "n_grid": cfg.n_grid,
            "n_filled": int(ctx.gridder.pattern.n_filled),
        }
    finally:
        try:
            reader.disconnect()
        except Exception:
            LOG.exception("reader.disconnect failed (non-fatal)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fada-key", default="fada",
                   help="4-char fada buffer key (default: fada). "
                        "PARALLEL_AGENTS.md §4.1 maps via "
                        "$DSART_BUFFER_KEY_PREFIX so e.g. m3 → 'fa3a'.")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cuda:N / cpu (default: auto). "
                        "PARALLEL_AGENTS.md §4.2 pins M3 to GPU 0 on h01.")
    p.add_argument("--config", type=Path,
                   default=Path("configs/config_corr.yaml"),
                   help="path to config_corr.yaml for fada buffer-size validation")
    p.add_argument("--max-blocks", type=int, default=1,
                   help="stop after N fada blocks (default: 1 for "
                        "smoke runs; production: omit for unlimited)")
    p.add_argument("--t-int-fast-native", type=int, default=T_INT_FAST_NATIVE,
                   help="fast-corr integration depth in NATIVE samples per "
                        "fast-vis tile. Default: %d (= %d µs cadence). "
                        "Burst-test override: 32 (= 1048.576 µs, 4× cadence)."
                        % (T_INT_FAST_NATIVE, int(T_INT_FAST_NATIVE * NATIVE_SAMPLE_US)))
    p.add_argument("--obs-dec-deg", type=float, required=True,
                   help="observing source declination (deg) — F21 + gridder "
                        "pattern lookup")
    p.add_argument("--chgroup", type=int, default=0,
                   help="corr-node chgroup index 0..15")
    p.add_argument("--apply-cal", type=Path, default=None,
                   help="legacy beamformer_weights_*.dat blob path "
                        "(F21 fold lives in cal_loader)")
    p.add_argument("--cal-mode", default=CalMode.PHASE_ONLY,
                   choices=(CalMode.PHASE_ONLY, CalMode.FULL),
                   help="phase_only = divide by |G| (default; fp16 stable). "
                        "full = preserve gain magnitude (routes through fp32).")
    p.add_argument("--cal-pol-swap", action="store_true",
                   help="swap cal pol axis")
    p.add_argument("--flagants", type=Path, default=None,
                   help="path to legacy flagants.dat (default: no static flagants)")
    p.add_argument("--rfi-disabled", action="store_true",
                   help="disable the RFI flagger entirely (e.g. for "
                        "synth-data tests with injected narrowband signals "
                        "that would trigger the SK detector)")
    p.add_argument("--n-grid", type=int, default=256,
                   help="grid side length (default: 256)")
    p.add_argument("--kernel-support", type=int, default=1,
                   help="gridding kernel half-width in cells (default: 1 → 3x3)")
    p.add_argument("--static-sky-alpha", type=float, default=0.001,
                   help="static-sky EMA smoothing factor (default: 0.001 → "
                        "~700-cube half-life)")
    p.add_argument("--static-sky-warmup-cubes", type=int, default=8,
                   help="cubes during which to BUILD the EMA but not subtract")
    p.add_argument("--static-sky-disabled", action="store_true",
                   help="disable static-sky subtraction entirely (useful for "
                        "the 0319 continuum bench where the brightest source "
                        "IS the static sky)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/tmp/dsart-fast-grid"),
                   help="per-block output dir (default: %(default)s)")
    p.add_argument("--blocks-output-mode", default="first_tile_only",
                   choices=("full", "first_tile_only", "none"),
                   help="full: write the entire (n_fv, N_filled) tensor. "
                        "first_tile_only: write only fast-vis tile 0. "
                        "none: meta json only.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.config.exists():
        fada_b = _load_fada_buffer_size(args.config)
        if fada_b != FADA_BYTES_PER_BLOCK:
            LOG.error("config %s fada bytes_per_block=%d != %d",
                      args.config, fada_b, FADA_BYTES_PER_BLOCK)
            return 2
    else:
        LOG.warning("config %s not found; skipping buffer-size validation", args.config)

    fada_int = _key_to_int(args.fada_key)
    device = _pick_device(args.device)

    if args.apply_cal is not None and not args.apply_cal.is_file():
        LOG.error("--apply-cal path %s not found", args.apply_cal)
        return 2

    cfg = FastIntegrationConfig(
        chgroup=args.chgroup,
        obs_dec_rad=math.radians(args.obs_dec_deg),
        n_grid=args.n_grid,
        kernel_support=args.kernel_support,
        t_int_fast_native=args.t_int_fast_native,
        cal_path=args.apply_cal,
        cal_mode=args.cal_mode,
        cal_pol_swap=args.cal_pol_swap,
        flagants_path=args.flagants,
        rfi_enabled=not args.rfi_disabled,
        static_sky_alpha=args.static_sky_alpha,
        static_sky_warmup_cubes=args.static_sky_warmup_cubes,
        static_sky_disabled=args.static_sky_disabled,
    )

    try:
        run(
            fada_int,
            args.output_dir,
            device,
            cfg,
            max_blocks=args.max_blocks,
            blocks_output_mode=args.blocks_output_mode,
        )
    except _StopRequested:
        LOG.info("clean stop")
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt; clean stop")
    except Exception:
        LOG.exception("fatal error in run()")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
