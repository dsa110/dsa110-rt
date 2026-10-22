"""x2 WS-C deliverable 4: the 2x orchestrator (CPU path) against the
numpy reference, end to end from raw fada pages.

512-packet pages (``tests/x2/_small_blocks.py``; production antenna
fixture, station-number core mask, WS-P value map and tiered plan) carry
int4 noise on six core antennas and zeros elsewhere, so the float64
reference only has to correlate those six. Chgroup 2 keeps the tier-B
stage-1 shifts inside the 128-sample (q1) block.
Checked for both value kinds (per-baseline through the GEMM's class
reduce), on the first/last coarse DM of each tier:

* stage-1 + decimation + static-sky output == reference (fp16 ring
  rounding: <= 2e-3 of the rms);
* the TX hand-off carries one payload row per (DM, tier sample) of the
  packer's layout (row-rule bit-equality itself: ``test_corr2x_packer``);
* TX hand-off framing: emitted block, t0_native, tier sample, stream ids;
* the RFI cube mask (folded into the gains) and the array-burst time mask
  (applied on operand load) zero exactly what the reference zeroes;
* the stage-2 FIFO is never touched in 2x.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

import dsart.services.corr_fast_integration as cfi
from dsart.cal.cal_loader import FastCorrCalTensors
from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import FADA_BYTES_PER_BLOCK, NANTS, NCHAN_PER_CHGROUP
from dsart.rfi import FlagBlockResult, FlagSourceBit
from dsart.x2.corr.pipeline2x import Corr2xPipeline, RecordingTx2x, injection_active
from dsart.x2.corr.plan2x import proposed_corr_plan
from dsart.x2.ref import corr2x_ref as ref

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G = 2
ANTS = [3, 10, 22, 40, 55, 71]                       # core antennas carrying noise
BLOCKS = [5, 6, 7]
NPKT = 512                                           # CPU pages; GPU uses 2048


def _fixture():
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    z = np.load(os.path.join(ROOT, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
    e = z["antpos_e"].astype(np.float32)
    n = z["antpos_n"].astype(np.float32)
    mask = core_baseline_mask_from_station_numbers(z["station"])
    return e, n, mask, z["station"]


def _pages(npkt):
    rng = np.random.default_rng(42)
    pages = []
    for _ in BLOCKS:
        raw = np.zeros((npkt, NANTS, NCHAN_PER_CHGROUP, 2, 2), np.uint8)
        raw[:, ANTS] = rng.integers(0, 256, size=(npkt, len(ANTS), NCHAN_PER_CHGROUP, 2, 2))
        pages.append(raw.reshape(-1))
    assert pages[0].size == FADA_BYTES_PER_BLOCK * npkt // 2048
    return pages


class _Flagger:
    warmup_cubes = 0
    _sk_far = _bandpass_k = _group_k = 0.0

    def __init__(self):
        self.n = 0

    def reset_warmup(self):
        self.n = 0

    def flag_block(self, real, imag, **kw):
        self.n += 1
        mask = torch.zeros(NANTS, NCHAN_PER_CHGROUP, NPOL_, dtype=torch.bool)
        mask[ANTS[1], 40:80, :] = True
        ab = "fire" if self.n == 2 else None
        return FlagBlockResult(mask=mask, source_tags=mask.to(torch.uint8) * int(FlagSourceBit.SK),
                               warmup=False, flag_fraction_total=float(mask.float().mean()),
                               array_burst=ab)


NPOL_ = 2


def _ab(ctx, rfi):
    if rfi.array_burst != "fire":
        return None
    m = torch.zeros(64, NCHAN_PER_CHGROUP, 2, dtype=torch.bool)
    m[10:12] = True
    return m


def _gains():
    rng = np.random.default_rng(7)
    ph = rng.uniform(-np.pi, np.pi, size=(NANTS, NCHAN_PER_CHGROUP, 2))
    return np.exp(1j * ph)


def _build(kind: str, monkeypatch, *, device="cpu", npkt=NPKT, value_subset=24, dms=(0, 31, 32)):
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.x2.opoint import opoint_2x
    from ._small_blocks import patch_small_blocks
    if device == "cpu":
        page_bytes = patch_small_blocks(monkeypatch, npkt)
    else:
        page_bytes = FADA_BYTES_PER_BLOCK
    nq = npkt * 2 // 8
    monkeypatch.setattr(cfi, "_array_burst_time_mask", _ab)
    e, n, mask, _ = _fixture()
    op = opoint_2x(kind, 2)
    vm = build_value_map_for_opoint(op, e, n, is_core_baseline_mask=mask, dec_deg=71.63)
    cfg = cfi.FastIntegrationConfig(
        chgroup=G, obs_dec_rad=math.radians(71.63), n_grid=256, t_int_fast_native=8,
        cal_path=None, rfi_enabled=False, rfi_array_burst_mode="off", static_sky_disabled=True,
        chan_sum_factor=2, n_sub=2, cell_lambda_mode="tee45",
    )
    ctx = cfi.build_context(cfg, device=torch.device(device), antpos_e=e, antpos_n=n,
                            is_core_baseline_mask=mask)
    cfg.rfi_enabled = True
    ctx.rfi_flagger = _Flagger()
    gn = _gains()
    gt = np.ascontiguousarray(gn.transpose(1, 2, 0))[:, None, :, None, :]
    ctx.cal = FastCorrCalTensors(
        cal_real=torch.from_numpy(gt.real.astype(np.float32)).half().to(device),
        cal_imag=torch.from_numpy(gt.imag.astype(np.float32)).half().to(device),
        raw_bf_weights=None, obs_dec_rad=cfg.obs_dec_rad, chgroup=G, cal_mode="phase_only",
        pol_swap=False, info={},
    )
    sentinel = object()
    ctx.stage2_fifo = sentinel
    plan = proposed_corr_plan(2)
    dms = list(dms)
    for d in dms:                                                   # every DM fits the window
        ti, c = plan.tier_of(d)
        assert int(plan.stage1_bins(ti)[G][:, c].max()) < nq - 4
    tx = RecordingTx2x()
    orch = Corr2xPipeline(ctx, plan=plan, vm=vm, is_core_baseline_mask=mask, bits_main=8,
                          tx2x=tx, static_sky_window_blocks=2, static_sky_warmup=0,
                          value_subset=value_subset, dm_subset=dms, expected_fada_bytes=page_bytes,
                          fv_chunk=min(nq, 128))
    return ctx, orch, tx, vm, plan, dms, e, n, mask, gn, sentinel


@pytest.mark.parametrize("kind", ["gridded", "baselines"])
def test_corr2x_pipeline_cpu_equals_reference(kind, monkeypatch) -> None:
    _run_and_check(kind, monkeypatch, device="cpu", npkt=NPKT, value_subset=24, dms=(0, 31, 32))


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("kind", ["gridded", "baselines"])
def test_corr2x_pipeline_gpu_equals_reference(kind, monkeypatch) -> None:
    """Production pages and kernels, every value, first/last DM of each tier."""
    _run_and_check(kind, monkeypatch, device="cuda", npkt=2048, value_subset=None,
                   dms=(0, 31, 32, 46))


def _run_and_check(kind, monkeypatch, *, device, npkt, value_subset, dms) -> None:
    NQ = npkt * 2 // 8
    ctx, orch, tx, vm, plan, dms, e, n, mask, gn, sentinel = _build(
        kind, monkeypatch, device=device, npkt=npkt, value_subset=value_subset, dms=dms)
    pages = _pages(npkt)
    outs = []
    for b, p in zip(BLOCKS, pages):
        outs += orch.push(p, block_n=b)
    outs += orch.flush()
    assert [o.block_n for o in outs] == BLOCKS
    assert ctx.stage2_fifo is sentinel                       # stage-2 FIFO off in 2x
    assert [r["block_n"] for r in tx.log] == [5, 5, 6, 6]    # tiers A, B of blocks 5 and 6
    assert [r["tier"] for r in tx.log] == ["A", "B", "A", "B"]
    NPKT = npkt  # noqa: N806

    # ---- reference: unpack/mask/cal/GEMM on the six antennas ------------------------
    from dsart.grid.baseline_grid import kept_core_baselines
    _, ka, kb = kept_core_baselines(mask, NANTS)
    sel = np.flatnonzero(np.isin(ka, ANTS) & np.isin(kb, ANTS))
    loc = {a: i for i, a in enumerate(ANTS)}
    la = np.array([loc[a] for a in ka[sel]])
    lb = np.array([loc[b] for b in kb[sel]])
    Vs = []
    for i, p in enumerate(pages):
        raw = p.reshape(NPKT, NANTS, NCHAN_PER_CHGROUP, 2, 2)[:, ANTS]
        E = ref.unpack_int4_onwire(raw.reshape(-1), n_packets=NPKT, nants=len(ANTS),
                                   nchan=NCHAN_PER_CHGROUP)
        cube = np.zeros((len(ANTS), NCHAN_PER_CHGROUP, 2), bool)
        cube[1, 40:80, :] = True
        tm = None
        if i == 1:
            tm = np.zeros((64, NCHAN_PER_CHGROUP, 2), bool)
            tm[10:12] = True
        E = ref.apply_masks(E, cube, tm)
        # gains as the GPU sees them: fp16-rounded
        g16 = (gn.real.astype(np.float16).astype(np.float64)
               + 1j * gn.imag.astype(np.float16).astype(np.float64))[ANTS]
        E = ref.apply_cal(E, g16)
        Vs.append(ref.gemm_core(E, la, lb, t_int=8, csf=2))
    per = vm.nchan_eff // 2
    for rec in tx.log:
        blk = rec["block_n"]
        i = BLOCKS.index(blk)
        joined = np.concatenate([Vs[i], Vs[i + 1]], axis=0)
        assert rec["t0_native"] == blk * NQ * 8
        assert rec["stream_ids"] == [G * 2, G * 2 + 1]
        ti = 0 if rec["tier"] == "A" else 1
        tier = plan.tiers[ti]
        assert rec["T_native"] == tier.T_native
        for si, pk in enumerate(rec["packets"]):
            assert pk["payload"].shape == (len(rec["dms"]), NQ // tier.decim,
                                           orch.tiers[ti].packer.bytes_per_sample(si))
    # stage-1 output (after static sky) is carried in the outputs
    sky = {}
    for o in outs:
        for ts, q_emit, pk, x in getattr(o, "x2", []):
            for j, d in enumerate(ts.dms):
                for s in range(2):
                    lo, hi = orch.voff[s], orch.voff[s + 1]
                    st = vm.streams[G * 2 + s]
                    if kind == "gridded":
                        _, aa, bb = kept_core_baselines(mask, NANTS)
                        nu = build_summed_chgroup_freq_table_GHz(2)[G][s * per:(s + 1) * per]
                        dest = ref.value_dest_cells(e, n, aa[sel], bb[sel], nu,
                                                    cell_lambda=vm.cell_lambda, n_grid=256,
                                                    cell_row=st.cell_row[orch.value_sel[s]],
                                                    cell_col=st.cell_col[orch.value_sel[s]])
                        conj = np.zeros(dest.shape, bool)
                    else:
                        c = vm.classes
                        dest, conj = ref.value_dest_classes(c.bl_class[sel], c.bl_conj[sel],
                                                            st.class_id[orch.value_sel[s]], per)
                    blk = q_emit // NQ
                    i = BLOCKS.index(blk)
                    joined = np.concatenate([Vs[i], Vs[i + 1]], axis=0)
                    ks = np.arange(s * per, (s + 1) * per)
                    ti = 0 if ts.name == "A" else 1
                    c_loc = d - plan.tiers[ti].d0
                    xs = ref.stage1_values(joined, dest, conj, ks=ks,
                                           s1=plan.stage1_bins(ti)[G][ks, c_loc],
                                           n_values=hi - lo, t0=0, n_out=NQ)
                    xd = ref.decimate(xs, ts.D)
                    key = (int(d), s)
                    rs = sky.setdefault(key, ref.StaticSkyRef(window_blocks=2, warmup_cubes=0))
                    want = rs.apply(0, xd)
                    got = x[j, :, lo:hi].cpu().numpy().astype(np.complex128)
                    rms = np.sqrt(np.mean(np.abs(want) ** 2)) + 1e-12
                    assert np.abs(got - want).max() <= 2e-3 * rms * 10, (kind, d, s, blk)
                    assert np.sqrt(np.mean(np.abs(got - want) ** 2)) <= 2e-3 * rms
    # the time mask zeroed the flagged accumulations of block 6 (acc 10, 11)
    q_acc = NQ // 64
    assert np.abs(Vs[1][10 * q_acc:12 * q_acc]).max() == 0.0
    assert np.abs(Vs[1][12 * q_acc:14 * q_acc]).max() > 0.0


def test_sky_export_hooks(monkeypatch) -> None:
    """The exporter's duck types: owner cells per value, tier-A slot-0 mean
    x 32/16 (today's natives per sample)."""
    ctx, orch, tx, vm, plan, dms, e, n, mask, gn, _ = _build("baselines", monkeypatch)
    rows, cols = orch.sky_cells()
    assert rows.dtype == np.uint16 and rows.size == orch.csr.n_values
    st = vm.streams[G * 2]
    assert np.array_equal(rows[: orch.voff[1]], st.cell_row[orch.value_sel[0]])
    assert orch.sky_static.running_mean_for(0) is None
    x = torch.ones((len(orch.tiers[0].dms), 4, orch.csr.n_values), dtype=torch.complex64)
    orch.tiers[0].static_sky.apply_(x)
    assert torch.allclose(orch.sky_static.running_mean_for(0), torch.full((orch.csr.n_values,), 2.0 + 0j))


def test_injection_window_detection() -> None:
    class A:
        def __init__(self, lo, hi):
            self.first_native_in_window, self.last_native_in_window = lo, hi

    class Inj:
        pending = {"a": A(5 * 4096 + 100, 5 * 4096 + 200)}

    assert injection_active(Inj(), 5) and not injection_active(Inj(), 4)
    assert not injection_active(Inj(), 6) and not injection_active(None, 5)
