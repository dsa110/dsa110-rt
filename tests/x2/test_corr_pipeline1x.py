"""x2 WS-C deliverable 1: the pipelined 1x-compat orchestrator is
bit-identical to a sequential loop over ``process_block``.

Same pattern as ``TestRTPhase4Pipeliner`` (pipelined == sequential), but
compared with ``torch.equal`` -- on the CPU every reduction is
deterministic, so anything short of identity is a bug.

What is exercised, per block, on a production-shaped configuration
(t_int 32, csf 8, F34 sliding window, multi-DM stage 1, n_sub 1 and 4,
static sky with warm-up, the per-coarse-DM stage-2 FIFO, a recording TX):

* block 1: empty RFI mask, cold-start stage 1 (zeros), RFI warm-up flag;
* block 2: antenna + channel flags -> mask folded into the cal gains;
* block 3: array-burst time mask fired -> today's voltage pass;
* block 4: channel flags across all antennas.

Equality is asserted on every ``IntegrationOutput.gridded_minus_sky``,
on every cube handed to ``transport_tx.transmit`` (with its block_n and
rfi_warming_up flag) and on ``n_tx``.

CPU tests use 256-packet pages (``tests/x2/_small_blocks.py`` swaps the
size-checked unpack and cal for shape-agnostic versions in BOTH loops; the
GPU tests use real 2048-packet pages). GEMM results are memoised on the
exact (sign-of-zero-normalised) input bytes. A cache hit
can only return the output of a byte-identical input, so the memo never
masks a difference -- it only stops the second pipeline and the second
``n_sub`` from paying for GEMMs the first already did.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

import dsart.services.corr_fast_integration as cfi
from dsart.cal.cal_loader import FastCorrCalTensors
from dsart.coarse_dm.dm_plan import (
    DMPlan,
    build_summed_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
)
from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.rfi import FlagBlockResult, FlagSourceBit
from dsart.x2.corr.pipeline1x import PipelinedCorrFast

CHGROUP = 3
T_INT = 32
CSF = 8
N_BLOCKS = 4
#: CPU tests use 256-packet pages (16 fast samples of 1.05 ms); every DM's
#: in-chgroup stage-1 shift (<= 14 bins at DM 450, chgroup 3) fits the
#: 2-block window, and DM 30's stage-2 shift (~14 samples) makes the
#: per-coarse-DM FIFO emit views of the PREVIOUS block's cube.
NPKT_CPU = 256
DMS = np.array([0.0, 30.0, 250.0, 450.0])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _plan() -> DMPlan:
    freqs = build_summed_chgroup_freq_table_GHz(CSF)
    return DMPlan(
        dm_pc_cc=DMS.copy(),
        n_fine_per_coarse=1,
        t_int_fast_us=T_INT * NATIVE_SAMPLE_US,
        chgroup_freqs_GHz=freqs,
        _delay_native_samples_table=compute_delay_native_samples_table(DMS, freqs),
        chan_sum_factor=CSF,
    )


def _blocks(n_packets: int = 2048) -> list[np.ndarray]:
    return [
        np.random.default_rng(20261022 + i).integers(
            0, 256, size=FADA_BYTES_PER_BLOCK * n_packets // 2048, dtype=np.uint8,
        )
        for i in range(N_BLOCKS)
    ]


_GEMM_MEMO: dict[tuple, torch.Tensor] = {}


def _digest(t: torch.Tensor) -> str:
    canon = (t + 0.0).contiguous()               # -0.0 -> +0.0
    return hashlib.blake2b(memoryview(canon.numpy()).cast("B"), digest_size=16).hexdigest()


def _memoise_gemm(kernel) -> None:
    orig = kernel.compute_split

    def compute_split(real_v, imag_v, **kw):
        key = (_digest(real_v), _digest(imag_v), tuple(sorted(kw.items())))
        hit = _GEMM_MEMO.get(key)
        if hit is None:
            hit = orig(real_v, imag_v, **kw)
            _GEMM_MEMO[key] = hit
        return hit.clone()

    kernel.compute_split = compute_split


class _ScheduledFlagger:
    """Duck-typed RFIFlagger: a fixed mask schedule by call number."""

    warmup_cubes = 1
    _sk_far = 0.0
    _bandpass_k = 0.0
    _group_k = 0.0

    def __init__(self) -> None:
        self.n = 0

    def reset_warmup(self) -> None:
        self.n = 0

    def flag_block(self, real, imag, **kw) -> FlagBlockResult:
        self.n += 1
        mask = torch.zeros(NANTS, NCHAN_PER_CHGROUP, NPOL, dtype=torch.bool)
        ab = None
        if self.n in (2, 3):
            mask[5] = True
            mask[30, 100:140, 0] = True
        if self.n == 3:
            ab = "fire"
        if self.n == 4:
            mask[:, 200:211, :] = True
        tags = mask.to(torch.uint8) * int(FlagSourceBit.FLAGANTS_DAT)
        return FlagBlockResult(
            mask=mask, source_tags=tags, warmup=self.n == 1,
            flag_fraction_total=float(mask.float().mean()), array_burst=ab,
        )


def _fake_ab_time_mask(ctx, rfi_result):
    if rfi_result.array_burst != "fire":
        return None
    m = torch.zeros(64, NCHAN_PER_CHGROUP, NPOL, dtype=torch.bool)
    m[17:19, :, :] = True
    m[40, 50:90, 1] = True
    return m


class _RecordingTx:
    def __init__(self) -> None:
        self.log: list[tuple[int, bool, list[torch.Tensor]]] = []

    def transmit(self, cubes_for_tx, *, block_n, rfi_warming_up) -> int:
        self.log.append(
            (int(block_n), bool(rfi_warming_up), [c.detach().clone() for c in cubes_for_tx])
        )
        return 7 * len(cubes_for_tx)


def _cal_tensors(dtype=torch.float32) -> FastCorrCalTensors:
    rng = np.random.default_rng(7)
    phase = rng.uniform(-np.pi, np.pi, size=(NCHAN_PER_CHGROUP, 1, NPOL, 1, NANTS))
    amp = rng.uniform(0.5, 1.5, size=phase.shape)
    g = amp * np.exp(1j * phase)
    return FastCorrCalTensors(
        cal_real=torch.from_numpy(g.real.astype(np.float32)).to(dtype),
        cal_imag=torch.from_numpy(g.imag.astype(np.float32)).to(dtype),
        raw_bf_weights=None,
        obs_dec_rad=math.radians(71.6),
        chgroup=CHGROUP,
        cal_mode="full",
        pol_swap=False,
        info={},
    )


def _ctx(n_sub: int, device: torch.device = torch.device("cpu")) -> cfi.IntegrationContext:
    e, n = _synth_antpos()
    core = cfi._build_core_baseline_mask(n_core=82)
    cfg = cfi.FastIntegrationConfig(
        chgroup=CHGROUP,
        obs_dec_rad=math.radians(71.6),
        n_grid=64,
        kernel_support=1,
        t_int_fast_native=T_INT,
        cal_path=None,
        rfi_enabled=False,
        rfi_array_burst_mode="off",
        static_sky_disabled=False,
        static_sky_window_s=0.27,           # 2 blocks
        static_sky_warmup_cubes=2,
        chan_sum_factor=CSF,
        n_sub=n_sub,
        sliding_window=True,
        cell_lambda_mode="common",
    )
    ctx = cfi.build_context(
        cfg, device=device, antpos_e=e, antpos_n=n,
        is_core_baseline_mask=core, dm_plan=_plan(),
    )
    cfg.rfi_enabled = True
    ctx.rfi_flagger = _ScheduledFlagger()
    if device.type == "cpu":
        # fp32 voltages: the CPU fp16 GEMM is unusably slow on one core
        ctx.cal = _cal_tensors(torch.float32)
        ctx.voltage_dtype = torch.float32
        _memoise_gemm(ctx.kernel)
    else:
        # production dtype: fp16 voltages, phase_only-style fp16 gains
        c = _cal_tensors(torch.float16)
        ctx.cal = FastCorrCalTensors(
            cal_real=c.cal_real.to(device), cal_imag=c.cal_imag.to(device),
            raw_bf_weights=None, obs_dec_rad=c.obs_dec_rad, chgroup=CHGROUP,
            cal_mode="phase_only", pol_swap=False, info={},
        )
    ctx.stage2_fifo = cfi._Stage2InterChgroupShiftAdapter(
        chgroup=CHGROUP, coarse_dm_pc_cm3=DMS.copy(), t_int_corr_us=T_INT * NATIVE_SAMPLE_US,
    )
    ctx.transport_tx = _RecordingTx()
    ctx.stage_timer = cfi.StageTimer(device=device)
    return ctx


def _run_legacy(n_sub: int, blocks, device):
    ctx = _ctx(n_sub, device)
    outs = [cfi.process_block(b, ctx=ctx, block_n=i + 1) for i, b in enumerate(blocks)]
    return outs, ctx


def _run_pipelined(n_sub: int, blocks, device, **kw):
    ctx = _ctx(n_sub, device)
    orch = PipelinedCorrFast(ctx, **kw)
    released: list[int] = []
    outs = []
    for i, b in enumerate(blocks):
        outs += orch.push(b, block_n=i + 1, release=lambda i=i: released.append(i + 1))
    outs += orch.flush()
    assert released == list(range(1, len(blocks) + 1))
    return outs, ctx


def _assert_identical(a, ctx_a, b, ctx_b) -> None:
    assert [o.block_n for o in a] == [o.block_n for o in b] == list(range(1, N_BLOCKS + 1))
    for oa, ob in zip(a, b):
        ga, gb = oa.gridded_minus_sky.cpu(), ob.gridded_minus_sky.cpu()
        assert ga.shape == gb.shape and ga.dtype == gb.dtype
        assert torch.equal(ga, gb), f"block {oa.block_n}: gridded_minus_sky differs"
        assert oa.n_tx == ob.n_tx
        assert oa.rfi.flag_fraction_total == ob.rfi.flag_fraction_total
        assert oa.rfi.warmup == ob.rfi.warmup
    la, lb = ctx_a.transport_tx.log, ctx_b.transport_tx.log
    assert [(n, w, len(c)) for n, w, c in la] == [(n, w, len(c)) for n, w, c in lb]
    assert len(la) == N_BLOCKS
    for (n, _, ca), (_, _, cb) in zip(la, lb):
        for x, y in zip(ca, cb):
            assert torch.equal(x.cpu(), y.cpu()), f"TX cube of block {n} differs"


@pytest.fixture(autouse=True)
def _ab_patch(monkeypatch):
    monkeypatch.setattr(cfi, "_array_burst_time_mask", _fake_ab_time_mask)


@pytest.fixture
def small(monkeypatch):
    """CPU tests: 256-packet pages through the same (patched) unpack and cal
    in both loops (x2 rule: short CPU tests; full size is CP1)."""
    from ._small_blocks import patch_small_blocks
    return patch_small_blocks(monkeypatch, NPKT_CPU)


@pytest.mark.parametrize("n_sub", [1, 4])
def test_pipelined_equals_process_block_cpu(n_sub: int, small) -> None:
    blocks = _blocks(NPKT_CPU)
    dev = torch.device("cpu")
    legacy, ctx_l = _run_legacy(n_sub, blocks, dev)
    piped, ctx_p = _run_pipelined(n_sub, blocks, dev, expected_fada_bytes=small)
    _assert_identical(legacy, ctx_l, piped, ctx_p)
    # the mask schedule really exercised all three masking routes
    assert ctx_p.rfi_flagger.n == N_BLOCKS
    assert any(bool(o.gridded_minus_sky.abs().sum() > 0) for o in piped[1:])
    # stage timer harvested every block on the CPU (synchronous stamps)
    s = ctx_p.stage_timer.summary()
    assert s["stage_blocks"] == N_BLOCKS
    assert {"unpack", "rfi", "zero_fill", "cal", "gemm", "stage1", "static_sky",
            "stage2", "tx"} <= set(s["stage_ms"])


def test_pipelined_no_defer_and_no_fold_cpu(small) -> None:
    """Undeferred TX and the unfolded mask route give the same bits too."""
    blocks = _blocks(NPKT_CPU)
    dev = torch.device("cpu")
    legacy, ctx_l = _run_legacy(1, blocks, dev)
    piped, ctx_p = _run_pipelined(1, blocks, dev, defer_tx=False, fold_mask_into_cal=False,
                                  expected_fada_bytes=small)
    _assert_identical(legacy, ctx_l, piped, ctx_p)


def test_output_lag_and_release_order_cpu(small) -> None:
    """push(k) returns block k-2 (deferred TX) and releases page k."""
    blocks = _blocks(NPKT_CPU)[:3]
    ctx = _ctx(1, torch.device("cpu"))
    orch = PipelinedCorrFast(ctx, expected_fada_bytes=small)
    got = []
    for i, b in enumerate(blocks):
        got.append([o.block_n for o in orch.push(b, block_n=i + 1)])
    assert got == [[], [], [1]]
    assert [o.block_n for o in orch.flush()] == [2, 3]


def test_stage_timer_never_blocks_and_bounds_pending() -> None:
    t = cfi.StageTimer(device="cpu", window=4, max_pending=2)
    for k in range(10):
        t.begin_block(k)
        t.mark("a")
        t.mark("b")
        t.end_block()
    s = t.summary()
    assert s["stage_blocks"] == 10 and set(s["stage_ms"]) == {"a", "b", "block_gpu"}
    assert all(v["p99"] >= v["p50"] >= 0.0 for v in s["stage_ms"].values())
    t.mark("ignored")                 # no open block: no-op
    assert t.summary()["stage_blocks"] == 10


# ---------------------------------------------------------------------------
# GPU (CP1): the same identity on the production kernels
# ---------------------------------------------------------------------------


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("n_sub", [1, 4])
def test_pipelined_equals_process_block_gpu(n_sub: int) -> None:
    """Production kernels (fp16, Triton GEMM into the ring with new store
    strides, n_fv-row stage 1, async H2D, deferred TX): torch.equal."""
    from dsart.services import host_pin
    blocks = _blocks()
    dev = torch.device("cuda")
    legacy, ctx_l = _run_legacy(n_sub, blocks, dev)
    leg = [(o.block_n, o.gridded_minus_sky.cpu()) for o in legacy]
    log_l = [(n, w, [c.cpu() for c in cs]) for n, w, cs in ctx_l.transport_tx.log]
    del legacy, ctx_l
    torch.cuda.empty_cache()
    piped, ctx_p = _run_pipelined(n_sub, blocks, dev)
    for (n, g), o in zip(leg, piped):
        assert o.block_n == n and torch.equal(g, o.gridded_minus_sky.cpu()), n
    for (n, w, ca), (n2, w2, cb) in zip(log_l, ctx_p.transport_tx.log):
        assert (n, w) == (n2, w2)
        assert all(torch.equal(x, y.cpu()) for x, y in zip(ca, cb)), n
    host_pin.unregister_all()


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
def test_pipelined_overlaps_h2d_gpu() -> None:
    """Timing smoke (not a gate): the stage timer publishes an h2d span and
    the per-block GPU span excludes it (the copy runs on its own stream)."""
    blocks = _blocks()
    ctx = _ctx(4, torch.device("cuda"))
    orch = PipelinedCorrFast(ctx)
    for i, b in enumerate(blocks):
        orch.push(b, block_n=i + 1)
    orch.flush()
    ctx.stage_timer.poll()
    s = ctx.stage_timer.summary()["stage_ms"]
    assert "h2d" in s and "block_gpu" in s
