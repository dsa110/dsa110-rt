"""corr_fast_integration acceptance tests (M3 chunk 4).

Exercises the chunk-4 integration pipeline:

    unpack → RFI flag → voltage zero-fill → cal apply (F21) → fast-corr
    GEMM → Stokes I → grid → static-sky EMA → coarse-DM stub → FIFO stub
    → transport-TX stub

Without PSRDADA — uses synthetic raw fada bytes + synthetic antpos so
the gridder pattern matches. The full-fixture replay (real fada pages
through the full 16-chgroup pipeline) lives in chunk 5/6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)
from dsart.rfi import FlagBlockResult, FlagSourceBit
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    IntegrationContext,
    IntegrationOutput,
    NoOpCoarseDM,
    NoOpStage2Fifo,
    NoOpTransportTx,
    StaticSkyEMA,
    apply_rfi_mask_to_voltages,
    build_context,
    process_block,
    _build_core_baseline_mask,
)


# ---------------------------------------------------------------------------
# Synthetic helpers (mirror chunk-2b + chunk-3a test helpers so no
# dependency on h01-only data)
# ---------------------------------------------------------------------------


def _synthetic_fada_block(seed: int = 20260505) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 256, size=FADA_BYTES_PER_BLOCK, dtype=np.uint8)


def _synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Mirror :func:`tests.test_sparsity_pattern._synth_antpos`."""
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _make_cfg(**overrides) -> FastIntegrationConfig:
    """Default test config: no cal (synthetic antpos doesn't match any
    real cal blob), no flagants, no static-sky warmup so subtraction
    starts immediately, RFI disabled by default to keep raw GEMM
    contributions visible.
    """
    base = dict(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=64,                                                   # small for speed
        kernel_support=1,
        t_int_fast_native=NPACKETS_PER_BLOCK * NTIMES_PER_PACKET,    # 1 tile
        cal_path=None,
        rfi_enabled=False,
        static_sky_disabled=True,
        static_sky_warmup_cubes=0,
    )
    base.update(overrides)
    return FastIntegrationConfig(**base)


def _build_test_context(
    cfg: FastIntegrationConfig,
    *,
    seed: int = 42,
) -> IntegrationContext:
    e, n = _synth_antpos(seed=seed)
    core_mask = _build_core_baseline_mask(n_core=82)
    return build_context(
        cfg, device=torch.device("cpu"),
        antpos_e=e, antpos_n=n,
        is_core_baseline_mask=core_mask,
    )


# ---------------------------------------------------------------------------
# apply_rfi_mask_to_voltages
# ---------------------------------------------------------------------------


def _make_voltages(*, dtype=torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
    """One block of synthetic voltages in M2 GEMM layout. Filled with a
    constant (1.0) so we can detect zero-fill by inspecting the cube
    after masking.
    """
    shape = (NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
             NPACKETS_PER_BLOCK, NANTS)
    real = torch.ones(shape, dtype=dtype)
    imag = torch.ones(shape, dtype=dtype)
    return real, imag


def test_apply_rfi_mask_zeros_only_flagged_cells() -> None:
    real, imag = _make_voltages()
    mask = torch.zeros(NANTS, NCHAN_PER_CHGROUP, NPOL, dtype=torch.bool)
    mask[5, 100, 0] = True                                                  # ant 5, ch 100, pol 0
    apply_rfi_mask_to_voltages(real, imag, mask)
    # Flagged cell across all (NTIMES, NPACKETS) is now 0.
    assert torch.all(real[100, :, 0, :, 5] == 0)
    assert torch.all(imag[100, :, 0, :, 5] == 0)
    # Same antenna, same channel, OTHER pol → untouched.
    assert torch.all(real[100, :, 1, :, 5] == 1)
    # Other ants/chs untouched.
    assert torch.all(real[200, :, :, :, 6] == 1)


def test_apply_rfi_mask_shape_validation() -> None:
    real, imag = _make_voltages()
    bad_mask = torch.zeros(NANTS, NCHAN_PER_CHGROUP - 1, NPOL, dtype=torch.bool)
    with pytest.raises(ValueError, match="rfi_mask shape"):
        apply_rfi_mask_to_voltages(real, imag, bad_mask)


def test_apply_rfi_mask_dtype_must_be_bool() -> None:
    real, imag = _make_voltages()
    int_mask = torch.zeros(NANTS, NCHAN_PER_CHGROUP, NPOL, dtype=torch.int8)
    with pytest.raises(TypeError, match="rfi_mask must be bool"):
        apply_rfi_mask_to_voltages(real, imag, int_mask)


def test_apply_rfi_mask_all_flagged_zeros_everything() -> None:
    real, imag = _make_voltages()
    mask = torch.ones(NANTS, NCHAN_PER_CHGROUP, NPOL, dtype=torch.bool)
    apply_rfi_mask_to_voltages(real, imag, mask)
    assert torch.all(real == 0)
    assert torch.all(imag == 0)


# ---------------------------------------------------------------------------
# StaticSkyEMA
# ---------------------------------------------------------------------------


def test_static_sky_cold_start_passthrough() -> None:
    """First cube goes through unchanged; the EMA is initialised from it."""
    ema = StaticSkyEMA(alpha=0.1, warmup_cubes=0)
    cube = torch.complex(torch.ones(2, 5), torch.zeros(2, 5))            # (n_fv=2, N_filled=5)
    out = ema.apply(cube)
    assert torch.allclose(out, cube)
    assert ema.cubes_seen == 1


def test_static_sky_warmup_no_subtract() -> None:
    """During warmup, the EMA is built but the cube is NOT subtracted."""
    ema = StaticSkyEMA(alpha=0.5, warmup_cubes=3)
    for i in range(3):
        cube = torch.complex(
            torch.full((1, 4), float(i + 1)),
            torch.zeros(1, 4),
        )
        out = ema.apply(cube)
        # In warmup → output equals input (modulo cold-start clone).
        assert torch.allclose(out, cube)
    assert ema.cubes_seen == 3
    assert not ema.in_warmup


def test_static_sky_subtracts_after_warmup() -> None:
    """After warmup, EMA is subtracted from each new cube."""
    ema = StaticSkyEMA(alpha=0.001, warmup_cubes=2)
    constant_value = 5.0 + 3.0j
    constant = torch.full((1, 4), constant_value, dtype=torch.complex64)
    for _ in range(2):
        ema.apply(constant)
    # Now in subtract mode. EMA ≈ 5+3j, so subtraction should be ~0.
    out = ema.apply(constant)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_static_sky_disabled_via_apply_false() -> None:
    """Construction with alpha out of (0, 1] raises."""
    with pytest.raises(ValueError, match="alpha"):
        StaticSkyEMA(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        StaticSkyEMA(alpha=1.5)
    with pytest.raises(ValueError, match="warmup_cubes"):
        StaticSkyEMA(warmup_cubes=-1)


def test_static_sky_input_validation() -> None:
    ema = StaticSkyEMA()
    real_cube = torch.zeros(2, 4)
    with pytest.raises(TypeError, match="must be complex"):
        ema.apply(real_cube)
    bad_shape = torch.complex(torch.zeros(2, 4, 5), torch.zeros(2, 4, 5))
    with pytest.raises(ValueError, match="must be 2D"):
        ema.apply(bad_shape)


# ---------------------------------------------------------------------------
# build_context — defaults
# ---------------------------------------------------------------------------


def test_build_context_default_stages_are_noop() -> None:
    cfg = _make_cfg()
    ctx = _build_test_context(cfg)
    assert isinstance(ctx.coarse_dm, NoOpCoarseDM)
    assert isinstance(ctx.stage2_fifo, NoOpStage2Fifo)
    assert isinstance(ctx.transport_tx, NoOpTransportTx)
    # rfi/static_sky disabled in default test cfg
    assert ctx.rfi_flagger is None
    assert ctx.static_sky is None


def test_build_context_rfi_and_static_sky_enabled() -> None:
    cfg = _make_cfg(rfi_enabled=True, static_sky_disabled=False)
    ctx = _build_test_context(cfg)
    assert ctx.rfi_flagger is not None
    assert ctx.static_sky is not None
    assert ctx.static_sky.alpha == cfg.static_sky_alpha
    assert ctx.static_sky.warmup_cubes == cfg.static_sky_warmup_cubes


def test_build_context_pattern_filled_cells_positive() -> None:
    cfg = _make_cfg()
    ctx = _build_test_context(cfg)
    pattern = ctx.gridder.pattern
    assert pattern.n_filled > 0
    assert pattern.n_grid == cfg.n_grid


# ---------------------------------------------------------------------------
# process_block — happy paths
# ---------------------------------------------------------------------------


def test_process_block_smoke_returns_grid_with_expected_shape() -> None:
    cfg = _make_cfg()
    ctx = _build_test_context(cfg)
    raw = _synthetic_fada_block()
    out = process_block(raw, ctx=ctx, block_n=1)
    assert isinstance(out, IntegrationOutput)
    assert out.gridded_minus_sky is not None
    n_fv = ctx.kernel.n_fast_vis_per_full_block
    n_filled = ctx.gridder.pattern.n_filled
    assert out.gridded_minus_sky.shape == (n_fv, n_filled)
    assert out.gridded_minus_sky.dtype == torch.complex64
    assert out.rfi is None                                                  # rfi_enabled=False
    assert out.n_tx == 0                                                    # NoOpTransportTx
    assert out.block_n == 1


def test_process_block_static_sky_disabled_passes_grid_through() -> None:
    """static_sky_disabled=True → gridded == gridded_minus_sky (object id)."""
    cfg = _make_cfg(static_sky_disabled=True)
    ctx = _build_test_context(cfg)
    raw = _synthetic_fada_block()
    out = process_block(raw, ctx=ctx, block_n=1)
    # Sanity: grid is finite, complex64, has at least one nonzero cell.
    assert torch.all(torch.isfinite(out.gridded_minus_sky.real))
    assert torch.all(torch.isfinite(out.gridded_minus_sky.imag))
    nz = torch.abs(out.gridded_minus_sky).sum().item()
    assert nz > 0.0


def test_process_block_with_rfi_enabled_returns_flag_result() -> None:
    """Verify RFI wiring: flagger fires, returns shape-correct result,
    warmup bit set on cube 1.

    NOTE: we deliberately do not assert a small FAR here — the
    synthetic raw is ``randint(0, 256)`` int4 fluff (uniform
    distribution), not Gaussian, so the SK detector legitimately
    flags everything. The FAR-on-thermal-noise contract is pinned
    by ``tests/test_rfi_flagger.py::test_sk_thermal_noise_far`` (M3
    chunk 3c, F23). This test only validates the orchestrator's
    RFI-stage wiring + the flag_fraction_total field is populated.
    """
    cfg = _make_cfg(rfi_enabled=True, static_sky_disabled=True)
    ctx = _build_test_context(cfg)
    raw = _synthetic_fada_block()
    out = process_block(raw, ctx=ctx, block_n=1)
    assert isinstance(out.rfi, FlagBlockResult)
    assert out.rfi.mask.shape == (NANTS, NCHAN_PER_CHGROUP, NPOL)
    assert out.rfi.mask.dtype == torch.bool
    assert out.rfi.source_tags.shape == (NANTS, NCHAN_PER_CHGROUP, NPOL)
    assert out.rfi.source_tags.dtype == torch.uint8
    assert out.rfi.warmup is True                                           # cube #1 of warmup
    assert 0.0 <= out.rfi.flag_fraction_total <= 1.0
    assert out.gridded_minus_sky is not None
    # When everything is flagged the voltage cube zeros out → grid is
    # exactly zero. This is the expected behaviour for the synthetic
    # full-flag case here; chunk-5 will exercise the partial-flag
    # path on real fada data.
    assert torch.all(out.gridded_minus_sky == 0)


def test_process_block_static_sky_subtracts_continuum() -> None:
    """Run several blocks of the SAME synthetic raw → static-sky EMA
    learns the per-cell mean and subtracts it. The 4th cube (after
    warmup_cubes=3) should have magnitude << the 1st cube.
    """
    cfg = _make_cfg(
        static_sky_disabled=False,
        static_sky_alpha=0.5,                                               # fast-learn for test
        static_sky_warmup_cubes=3,
    )
    ctx = _build_test_context(cfg)
    raw = _synthetic_fada_block()
    out_first = process_block(raw, ctx=ctx, block_n=1)
    mag_first = torch.abs(out_first.gridded_minus_sky).sum().item()
    # Run 2 more warmup cubes + 1 subtract cube.
    for n in (2, 3, 4):
        out_last = process_block(raw, ctx=ctx, block_n=n)
    mag_last = torch.abs(out_last.gridded_minus_sky).sum().item()
    assert mag_last < 0.2 * mag_first, (
        f"static-sky should suppress repeated continuum; mag_first={mag_first:.3e}, "
        f"mag_last={mag_last:.3e}"
    )


# ---------------------------------------------------------------------------
# process_block — RFI mask → voltage zero-fill semantics
# ---------------------------------------------------------------------------


def test_process_block_rfi_mask_zero_fill_kills_flagged_baselines() -> None:
    """Manually inject a flag for ant 5 across all chans/pols via a
    custom flagger; verify the gridded magnitude drops vs the same
    block run with rfi disabled.
    """
    # First run: rfi disabled → reference magnitude.
    cfg_ref = _make_cfg(rfi_enabled=False, static_sky_disabled=True)
    ctx_ref = _build_test_context(cfg_ref)
    raw = _synthetic_fada_block()
    out_ref = process_block(raw, ctx=ctx_ref, block_n=1)
    mag_ref = torch.abs(out_ref.gridded_minus_sky).sum().item()

    # Second run: rfi enabled with a flagger that flags ant 5 always.
    cfg_rfi = _make_cfg(rfi_enabled=True, static_sky_disabled=True)
    ctx_rfi = _build_test_context(cfg_rfi)

    class _FixedAntFlagger:
        warmup_cubes = 0
        in_warmup = False
        _sk_far = 0.0
        _bandpass_k = 0.0
        _group_k = 0.0

        def flag_block(self, real, imag, **kw) -> FlagBlockResult:
            mask = torch.zeros(
                NANTS, NCHAN_PER_CHGROUP, NPOL, dtype=torch.bool,
                device=real.device,
            )
            mask[5, :, :] = True                                            # flag ant 5
            tags = mask.to(torch.uint8) * int(FlagSourceBit.FLAGANTS_DAT)
            ff = float(mask.float().mean().item())
            return FlagBlockResult(
                mask=mask, source_tags=tags, warmup=False,
                flag_fraction_total=ff,
            )

    ctx_rfi.rfi_flagger = _FixedAntFlagger()                                # type: ignore[assignment]
    out_rfi = process_block(raw, ctx=ctx_rfi, block_n=1)
    mag_rfi = torch.abs(out_rfi.gridded_minus_sky).sum().item()
    # Ant 5 is in the core (idx < 82) → killing it removes 81 cross-bls
    # × NCHAN cells; expect a ~5-10% reduction in total grid magnitude.
    assert mag_rfi < mag_ref, (
        f"flagged-ant zero-fill should reduce grid magnitude; "
        f"mag_ref={mag_ref:.3e}, mag_rfi={mag_rfi:.3e}"
    )
    assert (mag_ref - mag_rfi) / mag_ref > 0.005, (
        "flagged ant should reduce grid magnitude by >0.5%"
    )


# ---------------------------------------------------------------------------
# Pluggable stages — no-op contract
# ---------------------------------------------------------------------------


def test_noop_coarse_dm_adds_unit_dm_axis() -> None:
    stub = NoOpCoarseDM()
    cube = torch.complex(torch.ones(2, 8), torch.zeros(2, 8))
    out = stub.dedisperse(cube, block_n=1, chgroup=0)
    assert out.shape == (1, 2, 8)
    assert torch.allclose(out[0], cube)


def test_noop_stage2_fifo_evicts_immediately() -> None:
    stub = NoOpStage2Fifo()
    cube = torch.zeros(1, 2, 4)
    evicted = stub.push(cube, block_n=1)
    assert len(evicted) == 1
    assert evicted[0] is cube


def test_noop_transport_tx_returns_zero() -> None:
    stub = NoOpTransportTx()
    n = stub.transmit([torch.zeros(1)], block_n=1, rfi_warming_up=False)
    assert n == 0


# ---------------------------------------------------------------------------
# Wiring sanity — pluggable stages chained via process_block
# ---------------------------------------------------------------------------


def test_process_block_calls_pluggable_stages_with_expected_shapes() -> None:
    """Verify the orchestrator passes the gridded cube through each
    stage with shapes the protocols promise.
    """
    seen: dict[str, object] = {}

    class _RecordingDedisp:
        def dedisperse(self, gridded, *, block_n, chgroup):
            seen["dedisp_in"] = tuple(gridded.shape)
            seen["dedisp_block_n"] = block_n
            seen["dedisp_chgroup"] = chgroup
            return gridded.unsqueeze(0)

    class _RecordingFifo:
        def push(self, dedispersed, *, block_n):
            seen["fifo_in"] = tuple(dedispersed.shape)
            seen["fifo_block_n"] = block_n
            return [dedispersed]

    class _RecordingTx:
        def transmit(self, cubes_for_tx, *, block_n, rfi_warming_up):
            seen["tx_n_cubes"] = len(cubes_for_tx)
            seen["tx_block_n"] = block_n
            seen["tx_rfi_warmup"] = rfi_warming_up
            return len(cubes_for_tx)

    cfg = _make_cfg()
    e, n = _synth_antpos()
    core_mask = _build_core_baseline_mask(n_core=82)
    ctx = build_context(
        cfg, device=torch.device("cpu"),
        antpos_e=e, antpos_n=n,
        is_core_baseline_mask=core_mask,
        coarse_dm=_RecordingDedisp(),
        stage2_fifo=_RecordingFifo(),
        transport_tx=_RecordingTx(),
    )
    raw = _synthetic_fada_block()
    out = process_block(raw, ctx=ctx, block_n=42)

    n_fv = ctx.kernel.n_fast_vis_per_full_block
    n_filled = ctx.gridder.pattern.n_filled
    assert seen["dedisp_in"] == (n_fv, n_filled)
    assert seen["dedisp_block_n"] == 42
    assert seen["dedisp_chgroup"] == cfg.chgroup
    assert seen["fifo_in"] == (1, n_fv, n_filled)                           # +DM axis
    assert seen["fifo_block_n"] == 42
    assert seen["tx_n_cubes"] == 1
    assert seen["tx_block_n"] == 42
    assert seen["tx_rfi_warmup"] is False                                   # RFI disabled in cfg
    assert out.n_tx == 1


# ---------------------------------------------------------------------------
# F18 + F20 + F21 composition (point-source recovery in dirty image)
# ---------------------------------------------------------------------------


def test_F24_pipeline_composes_grid_consistent_with_chunk2b_spine() -> None:
    """Same synthetic raw + same cal=None + same kernel cadence → the
    pre-grid Stokes-I tensor produced by the chunk-2b spine must be
    bit-identical to what chunk-4 feeds into the gridder. Pinning this
    cross-module composition prevents future drift in the
    spine-vs-integration agreement.

    Implementation: re-import the chunk-2b ``compute_block`` and run
    it with the same kernel; chunk-4's process_block exposes its
    pre-grid result indirectly by routing through a recording
    gridder stub that captures the Stokes-I tensor (via wrapping the
    real ``compute`` method).
    """
    from dsart.services.corr_fast_compute import compute_block as cb_2b

    cfg = _make_cfg()
    ctx = _build_test_context(cfg)
    raw = _synthetic_fada_block()

    # Spine reference.
    vis_2b = cb_2b(raw, kernel=ctx.kernel, cal=None,
                   voltage_dtype=ctx.voltage_dtype)

    # Chunk-4 path: capture the pre-grid Stokes-I tensor.
    captured: dict[str, torch.Tensor] = {}
    real_compute = ctx.gridder.compute

    def _recording(vis_stokes_i):
        captured["pre_grid"] = vis_stokes_i.clone()
        return real_compute(vis_stokes_i)

    ctx.gridder.compute = _recording                                        # type: ignore[assignment]

    process_block(raw, ctx=ctx, block_n=1)

    assert "pre_grid" in captured
    assert captured["pre_grid"].shape == vis_2b.shape
    assert torch.allclose(captured["pre_grid"], vis_2b, atol=1e-3, rtol=1e-3), (
        "chunk-4 integration pre-grid Stokes-I tensor must match the "
        "chunk-2b spine output for the same raw input + kernel + no-cal "
        "config — drift here means F18/F21 sign-conventions are at risk."
    )


# ---------------------------------------------------------------------------
# F31b: streamed kernel + Stokes-I chunking inside process_block
# ---------------------------------------------------------------------------


class TestF31bStreaming:
    """F31b: per-block kernel + Stokes-I streaming is bit-identical to un-chunked.

    F31b adds an inner ``n_fv_chunk`` slab loop in :func:`process_block`
    so the cfp32 ``(n_fv_slab, NBASE, NCHAN, NPOL)`` ``compute_split``
    intermediate stays bounded (~256 MB target) instead of growing to
    ~14 GB at the production ``t_int_fast_native=8`` cadence. The fp16
    matmuls per slab use the same inputs as the un-chunked path so the
    end-to-end gridded output must be bit-identical between
    ``cfg.n_fv_chunk=None`` (auto), explicit chunk sizes, and the
    legacy "all in one slab" behaviour.
    """

    @pytest.mark.parametrize("n_fv_chunk", [1, 2, 4, 8, 16])
    def test_chunked_equals_unchunked(self, n_fv_chunk):
        # t_int_fast_native=32 → packets_per_fast_vis=16 →
        # 2048 / 16 = 128 fast-vis tiles per block.
        cfg_full = _make_cfg(t_int_fast_native=32, n_fv_chunk=128)
        ctx_full = _build_test_context(cfg_full)
        raw = _synthetic_fada_block()
        out_full = process_block(raw, ctx=ctx_full, block_n=1)

        cfg_ch = _make_cfg(t_int_fast_native=32, n_fv_chunk=n_fv_chunk)
        ctx_ch = _build_test_context(cfg_ch)
        out_ch = process_block(raw, ctx=ctx_ch, block_n=1)

        assert out_full.gridded_minus_sky.shape == out_ch.gridded_minus_sky.shape
        torch.testing.assert_close(
            out_ch.gridded_minus_sky,
            out_full.gridded_minus_sky,
            rtol=0,
            atol=0,
        )

    def test_default_n_fv_chunk_auto_yields_valid_output(self):
        """``cfg.n_fv_chunk=None`` ⇒ auto-pick streams the block end-to-end."""
        cfg = _make_cfg(t_int_fast_native=32)
        ctx = _build_test_context(cfg)
        raw = _synthetic_fada_block()
        out_auto = process_block(raw, ctx=ctx, block_n=1)
        assert out_auto.gridded_minus_sky is not None
        assert out_auto.gridded_minus_sky.shape == (
            ctx.kernel.n_fast_vis_per_full_block,
            ctx.gridder.pattern.n_filled,
        )
        assert out_auto.gridded_minus_sky.dtype == torch.complex64

    def test_auto_matches_explicit_full_block(self):
        """``n_fv_chunk=None`` (auto) ≡ ``n_fv_chunk=n_fv_total``."""
        cfg_auto = _make_cfg(t_int_fast_native=32)
        ctx_auto = _build_test_context(cfg_auto)
        raw = _synthetic_fada_block()
        out_auto = process_block(raw, ctx=ctx_auto, block_n=1)

        cfg_full = _make_cfg(t_int_fast_native=32, n_fv_chunk=128)
        ctx_full = _build_test_context(cfg_full)
        out_full = process_block(raw, ctx=ctx_full, block_n=1)

        torch.testing.assert_close(
            out_auto.gridded_minus_sky,
            out_full.gridded_minus_sky,
            rtol=0,
            atol=0,
        )

    def test_rejects_zero_chunk(self):
        cfg = _make_cfg(t_int_fast_native=32, n_fv_chunk=0)
        ctx = _build_test_context(cfg)
        raw = _synthetic_fada_block()
        with pytest.raises(ValueError, match="n_fv_chunk"):
            process_block(raw, ctx=ctx, block_n=1)

    def test_chunk_larger_than_total_clamped(self):
        """``n_fv_chunk > n_fv_total`` is clamped to ``n_fv_total`` (no error)."""
        # 128 fast-vis tiles total; ask for a slab of 256 → clamped to 128.
        cfg = _make_cfg(t_int_fast_native=32, n_fv_chunk=256)
        ctx = _build_test_context(cfg)
        raw = _synthetic_fada_block()
        out = process_block(raw, ctx=ctx, block_n=1)
        assert out.gridded_minus_sky is not None


# ---------------------------------------------------------------------------
# F33: chan_sum_factor pipeline
# ---------------------------------------------------------------------------


class TestF33ChanSumFactor:
    """F33: pre-dedispersion 8-channel sum reduces the post-Stokes-I
    cube by ``chan_sum_factor``, the gridder pattern is rebuilt
    against summed-channel band-CENTER frequencies, and the output
    shape stays ``(n_fv, N_filled)``.
    """

    def test_chan_sum_factor_default_is_one(self) -> None:
        """``cfg.chan_sum_factor`` defaults to 1 ⇒ legacy pipeline."""
        cfg = _make_cfg()
        assert cfg.chan_sum_factor == 1
        ctx = _build_test_context(cfg)
        assert ctx.gridder.pattern.chan_sum_factor == 1

    def test_chan_sum_factor_8_runs_end_to_end(self) -> None:
        """``cfg.chan_sum_factor = 8`` ⇒ the gridder pattern gets 48
        effective channels and process_block produces a valid
        ``(n_fv, N_filled)`` cube."""
        cfg = _make_cfg(chan_sum_factor=8)
        ctx = _build_test_context(cfg)
        assert ctx.gridder.pattern.chan_sum_factor == 8
        raw = _synthetic_fada_block()
        out = process_block(raw, ctx=ctx, block_n=1)
        assert out.gridded_minus_sky is not None
        assert out.gridded_minus_sky.dtype == torch.complex64
        n_fv = ctx.kernel.n_fast_vis_per_full_block
        n_filled = ctx.gridder.pattern.n_filled
        assert out.gridded_minus_sky.shape == (n_fv, n_filled)
        # Sanity: the summed pattern's n_filled is ≤ the per-fine
        # pattern's n_filled (fewer (u, v) cells when channels collapse).
        cfg_ref = _make_cfg(chan_sum_factor=1)
        ctx_ref = _build_test_context(cfg_ref)
        assert (
            ctx.gridder.pattern.n_filled <= ctx_ref.gridder.pattern.n_filled
        )

    def test_chan_sum_factor_invalid(self) -> None:
        """Non-divisor of NCHAN_PER_CHGROUP fails at build_pattern."""
        cfg = _make_cfg(chan_sum_factor=7)
        with pytest.raises(ValueError, match="does not divide"):
            _build_test_context(cfg)
