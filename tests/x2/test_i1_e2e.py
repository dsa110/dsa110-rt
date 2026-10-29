"""x2 I1: the whole 2x chain on the CPU with synthetic bursts, physics asserted.

    exact-integer-delay point-source bursts (snr_model style) at DM 300 (tier A)
    and DM 2000 (tier B), widths 2 / 16 native, on-axis and off-axis
      -> corr: stage 1 (corr plan) + decimation, value-domain noise, static sky,
         row-rule packer, FramedTx2x -> frame v2 -> RxRingV4Writer (no sockets)
      -> search: RingV4Source(RxRingV4Reader) -> OwnerEngine (CPU) ->
         OwnerService + C1CyclePool -> C1 rows
      -> C2 test instance (temp dirs) -> CutoutRequest -> CutoutProducer on all
         8 halves (ring snapshot + re-imaging) -> Level3 v2 products -> C3 -> plots

for the gridded n_sub 2 cint4 format here and the per-baseline n_sub 4 cint5 format
in tests/x2/test_i1_e2e_baselines.py (one format per file: each fits the harness's
15 min on one core), plus the same DM-300 bursts through the 1x op-point (today's
wire, 1x-compat engine). The chain driver is ``i1_util.run_2x_chain``.

Asserted per burst: detected by the owner half of its coarse DM; fine DM within
one trial of the nearest; time (centre of the detecting filter's window -- C1
``event_specnum`` is the filter centre c, the window [c - L//2, c - L//2 + L) --
vs the pulse centre at nu_bot_proc) within +-1 T_ext; (l, m) within one pixel; S/N
within +-5% of the prediction (snr_model's exact-integer-shift model at the
injected arrival phase, with the exact imaging response of this value map at the
detected pixel, x the Phase 0 wire factor). On-axis bursts are found one pixel off:
the edge mask drops the phase-centre pixel (``compute_edge_mask(drop_dc)``); the
report carries what that costs (``best_over_true_pixel``). Plus: Level3 v2
products complete, the event never deleted, C3 decides, plots written, the raw
archive re-images (float64 reference) to the cutout.

Reductions (cost only, documented in tests/x2/i1_util.py): value-domain noise;
separable stage-1 signal (checked against corr2x_ref.stage1_values below); each
owner half searches only the fed coarse DMs; the other halves answer cutouts from
a noise ring; vectorised K0/K1/K3b CPU kernels (checked against the WS-S twins
below); one search cycle per DM (Layer-2 burn-in 0); C2 cutouts of 64 rows and
+-20 ms raw windows. Per-burst results go to $DSART_X2_I1_REPORT
(default /tmp/dsart-x2-i1/report.json).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from . import i1_util as U

# ---------------------------------------------------------------------------
# the replacements and the separable signal against their references
# ---------------------------------------------------------------------------


def test_fast_kernels_equal_the_ws_s_twins() -> None:
    from dsart.x2.search.codec import RowFormat, pack_frame2_rows, unpack_frame2_rows
    from dsart.x2.search.combine import k1_twin
    from dsart.x2.search.filters import FilterBank, k3b_twin

    rng = np.random.default_rng(7)
    # K0 unpack, 4- and 5-bit
    for b in (4, 5):
        fmt = RowFormat("frame2", n_values=301, n_heavy=17, bits=b)
        re = rng.integers(-(1 << (b - 1)), 1 << (b - 1), size=(9, 301))
        im = rng.integers(-(1 << (b - 1)), 1 << (b - 1), size=(9, 301))
        re[:, :17] = rng.integers(-128, 128, size=(9, 17))
        im[:, :17] = rng.integers(-128, 128, size=(9, 17))
        raw = pack_frame2_rows(re, im, fmt)
        for a, c in zip(U.unpack_frame2_rows_fast(raw, fmt), unpack_frame2_rows(raw, fmt)):
            np.testing.assert_array_equal(a, c)
    # K1 on the production owner CSR (gridded ns2), two rows of two fines
    f = U.get_format(U.FORMATS[0])
    inp = f.inp
    S, V = inp.c_v.shape
    R = 16
    codes = rng.integers(-7, 8, size=(S, V, R, 2)).astype(np.int8)
    shifts = rng.integers(0, 9, size=(2, S)).astype(np.int32)
    want = k1_twin(codes, c_v=inp.c_v, offset=inp.offset, csr=inp.csr, shifts=shifts, t_lo=20,
                   n_t=2, w_v=inp.w_v)
    got = U.k1_fast(codes, c_v=inp.c_v, offset=inp.offset, csr=inp.csr, shifts=shifts, t_lo=20,
                    n_t=2, w_v=inp.w_v)
    scale = float(np.abs(want).max())
    assert np.abs(got - want).max() <= 2e-6 * scale, np.abs(got - want).max() / scale
    # K3b: b3x and today's boxcars on a small random ring with a bright spike
    for bank in (FilterBank.b3x(), FilterBank.build((1, 2, 4, 8, 16, 32, 64), ("b",))):
        R, F, P = 400, 3, 50
        ring = rng.normal(size=(R, F, P)).astype(np.float16)
        ring[210:213, 1, 17] += np.float16(30.0)
        kw = dict(ring_row0=0, c_first=200, n_c=128, sigma_inv=np.array([1.0, 0.9, 1.1]),
                  s_inv=1.0 / bank.seeds, bank=bank, clip=250.0, chunk=64,
                  tap_pixels=np.array([3, 17, 40]))
        bw, tw, pw = k3b_twin(ring, **kw)
        bf, tf, pf = U.k3b_fast(ring, pix_block=16, **kw)
        # S/N units: the twin's fp32 running sums and the fp32 prefix sums
        # differ by summation order only (~1e-5 of ||h|| for q64)
        np.testing.assert_allclose(bf, bw, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(pf / bank.seeds[:, None, None, None],
                                   pw / bank.seeds[:, None, None, None], rtol=1e-5, atol=1e-3)
        assert np.mean(tf == tw) > 0.995                      # fp32 near-ties only


def test_separable_stage1_equals_the_corr_reference() -> None:
    """stage1_signal == corr2x_ref.stage1_values + decimate on dense fast-vis
    (both tiers, off-axis, a 400-baseline subset of one stream)."""
    from dsart.x2.ref import corr2x_ref as cref

    f = U.get_format(U.FORMATS[0])
    rows = np.sort(np.random.default_rng(3).choice(f.dest[0].shape[0], 400, replace=False))
    for dm, sid in ((300.0, 9), (2000.0, 26)):
        d = f.coarse_of_dm(dm)
        tier = f.tier(d)
        D = tier.T_native // U.Q1_NATIVE
        b = U.Burst("x", dm, 16, 4_000_000 + 7, 37, -23, 1.0)
        B = U.stream_source_sums(f, sid, b, rows=rows)
        j0, X = U.stage1_signal(f, sid, d, b, B, 1.0)
        g, s = divmod(sid, f.n_sub)
        arr = U.native_arrivals(b)[g]
        c0 = s * f.per * f.csf
        chans = np.arange(c0, c0 + f.per * f.csf)
        q_lo = int(arr[chans].min()) // 8 - 4
        q_lo -= q_lo % D
        nq = (int(arr[chans].max()) + 16) // 8 + 4 - q_lo
        nq += (-nq) % D
        s1 = np.asarray(f.cplan.stage1_global[g, s * f.per:(s + 1) * f.per, d])
        q_lo -= int(s1.max()) + D
        q_lo -= q_lo % D
        nq += int(s1.max()) + D
        nq += (-nq) % D
        # dense V[q, k, r] (one stream's summed channels) from the native pulses
        from dsart.common.constants import SPEED_OF_LIGHT_M_S
        _, sm = U.oracle()
        e, n = np.asarray(f.e, float), np.asarray(f.n, float)
        a_, b_ = f.ant_a[rows], f.ant_b[rows]
        l_, m_ = b.lm_rad(f.cell_rad)
        proj = (e[a_] - e[b_]) * l_ + (n[a_] - n[b_]) * m_
        Vd = np.zeros((nq, f.per, rows.size), np.complex128)
        for ci, c in enumerate(chans):
            vis = np.exp(2j * np.pi * sm.NU[g, c] * 1e9 / SPEED_OF_LIGHT_M_S * proj)
            for nn in range(int(arr[c]), int(arr[c]) + b.width):
                Vd[nn // 8 - q_lo, c // f.csf - s * f.per] += vis
        dest = f.dest[sid][rows]
        conj = f.conj[sid][rows]
        n_out = nq - int(s1.max()) - D
        n_out -= n_out % D
        x = cref.stage1_values(Vd, dest, conj, ks=np.arange(f.per), s1=s1, n_values=f.nv(sid),
                               t0=0, n_out=n_out)
        xd = cref.decimate(x, D)
        jr0 = q_lo // D
        full = np.zeros_like(xd)
        a = max(j0, jr0)
        z = min(j0 + X.shape[0], jr0 + xd.shape[0])
        full[a - jr0:z - jr0] = X[a - j0:z - j0]
        assert z > a and np.abs(xd).max() > 1.0
        np.testing.assert_allclose(full, xd, atol=1e-9 * np.abs(xd).max())


def test_predictor_reproduces_snr_model() -> None:
    """My exact-arrival predictor, with rho = 1 and today's boxcars averaged
    over arrival phase, equals snr_model.snr_binned on the same histogram
    (exact integer arrivals, no smear box); and the plan's applied shifts are
    snr_model's Scheme.applied_us (checked inside snr_model_reference)."""
    f = U.get_format(U.FORMATS[0])
    for dm, w in ((300.0, 2), (300.0, 16), (2000.0, 2), (2000.0, 16)):
        b = U.Burst("x", dm, w, 4_000_000, 0, 0, 1.0)
        fi = f.nearest_fine(dm)
        ref = U.snr_model_reference(f, b, fi)
        mine = U.phase_avg_ladder(f, b, fi)
        assert abs(mine / ref["eta_phase_avg"] - 1.0) < 0.02, (dm, w, mine, ref)


# ---------------------------------------------------------------------------
# the 2x chain (gridded n_sub 2 cint4 here; per-baseline n_sub 4 cint5 in
# tests/x2/test_i1_e2e_baselines.py -- one file each fits the harness's 15 min)
# and the 1x-compat leg
# ---------------------------------------------------------------------------


def test_i1_2x_chain_gridded_ns2_cint4(tmp_path, monkeypatch) -> None:
    U.run_2x_chain("2x-gridded-ns2-cint4", tmp_path, monkeypatch)


def test_i1_1x_compat(monkeypatch) -> None:
    """The same DM-300 bursts through today's op-point (see ``run_1x_leg``)."""
    U.run_1x_leg(monkeypatch)


# ---------------------------------------------------------------------------
# real data: 250924mptq (GPU mode only)
# ---------------------------------------------------------------------------

VOLT_ROOT = Path(os.environ.get("DSART_X2_I1_VOLTAGES", "/home/ubuntu/data/voltages/250924mptq"))


@pytest.mark.x2_gpu
def test_i1_real_burst_250924mptq_gpu() -> None:
    """The real burst through the 2x chain AND today's 1x chain, on the same
    voltage pages, cal (the solution live at the burst), DEC steering (the
    burst's DEC), RFI (off) and static-sky warm-up (2 blocks):

    * 2x: Corr2xPipeline (GPU) -> frame v2 -> ring v4 -> GPU owner engine;
    * 1x: corr_fast's production context (multi-DM stage 1, deployed v2 plan,
      per-coarse-DM stage-2 FIFO) under PipelinedCorrFast -> today's v1 wire
      -> the owner engine in 1x-compat mode (production search arithmetic);

    each engine warmed over the cycle before the burst (Layer-1 / Layer-2).
    Asserted: both detect it within one fine trial of DM 404.688 and within 2
    samples of the expected arrival; the 2x (l, m) equals the 1x (l, m) within
    one pixel (the plan's criterion: same conventions -- today's (l, m)
    orientation, fftshift, edge mask, 45" cell). Reported: both S/N, their ratio
    vs the snr_model 2x/1x ratio, and the geometric (TNS) position. NOT feasible
    on CPU; ``node_cpu_tests.sh -g 0 -e DSART_X2_I1_VOLTAGES=<dump>`` with
    ``voltages/*_sb<NN>_data.out`` + ``cals/beamformer_weights_sb<NN>*.dat``."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA-only (GPU mode)")
    staged = U.real_inputs(VOLT_ROOT)
    if staged is None:
        pytest.skip(f"250924mptq voltages + 16 cal files not staged under {VOLT_ROOT}")
    vfiles, cals = staged
    replay = U.replay_module()
    dev = torch.device("cuda:0")
    x2 = U.run_real_2x(vfiles, cals, replay, device=dev)
    U.write_report("real_250924mptq/2x", x2)
    x1 = U.run_real_1x(vfiles, cals, replay, device=dev)
    U.write_report("real_250924mptq/1x", x1)
    n_bot = U.real_n_bot()
    geo = U.real_geometry(x2["cell_rad"])
    cmp = dict(
        n_bot=n_bot, geometry_tns=geo,
        d_pix_2x_minus_1x=[x2["pixel"][0] - x1["pixel"][0], x2["pixel"][1] - x1["pixel"][1]]
        if "pixel" in x2 and "pixel" in x1 else None,
        dl_px_2x_minus_1x=(x2.get("l_rad", np.nan) - x1.get("l_rad", np.nan)) / x2["cell_rad"],
        dm_px_2x_minus_1x=(x2.get("m_rad", np.nan) - x1.get("m_rad", np.nan)) / x2["cell_rad"],
        snr_ratio_2x_1x=x2.get("snr", np.nan) / x1.get("snr", np.nan),
        model_ratio_2x_1x=U.model_ratio_2x_1x(U.REAL_DM, (262.144, 524.288, 1048.576, 2097.152,
                                                           4194.304)))
    if "snr" in x1:
        w1 = max(1, int(x1["width"]))                   # 1x boxcar width (1.048576 ms samples)
        rho = x1["snr"] / math.sqrt(w1)
        for _ in range(20):                             # true per-sample S/N under today's wire
            rho = x1["snr"] / math.sqrt(w1) / float(U.today_compression(np.array([rho]))[0])
        cmp["1x_per_sample_snr_true_est"] = rho
        cmp["1x_wire_response_est"] = float(U.today_compression(np.array([rho]))[0])
    U.write_report("real_250924mptq/compare", cmp)
    for leg, r in (("2x", x2), ("1x", x1)):
        assert "snr" in r, (leg, r)
        assert abs(r["fine"] - r["f0"]) <= 1, (leg, r)
        assert abs(r["event"] - n_bot / r["ext_native"]) <= 2.0, (leg, r, n_bot)
    assert x2["cell_rad"] == pytest.approx(x1["cell_rad"], rel=1e-12)
    assert abs(cmp["dl_px_2x_minus_1x"]) <= 1.0 + 1e-6 and abs(cmp["dm_px_2x_minus_1x"]) <= 1.0 + 1e-6, cmp
