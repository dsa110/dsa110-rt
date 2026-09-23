"""End-to-end: a dispersed burst through the REAL corr + search code.

For every one of the 16 chgroups a delta-function burst is laid into
``vis_stokes_i`` at its true (quantised) arrival time per summed
channel. It then goes through, unmodified:

  corr   Stage1MultiDMCoarseDM  (production fused Triton dedisp+grid)
         + the corr stage-2 delay  (compute_stage2_shifts, as the
           inter-chgroup FIFO applies it)
  search compute_time_shift_search (the per-stream residual shift)

and the streams are summed per fine-DM trial. Because the source sits
at the phase centre, the image peak is the sum over cells, so the
recovered fraction is ``max_(f,t) sum / (number of mapped sources)``
without needing the imager -- which is tested bit-exactly elsewhere.

This is the check that the three new pieces (sub-band stage-1
reference, sub-band patterns, sub-band shift table) compose correctly:
a sign or reference error anywhere shows up as n_sub=4 recovering LESS
than n_sub=1. Skipped without CUDA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip("e2e sub-band test runs the Triton stage-1 kernel",
                allow_module_level=True)
pytest.importorskip("triton")

from dsart.coarse_dm.dm_plan import (  # noqa: E402
    DMPlan,
    build_summed_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
    subband_ref_freqs_table_GHz,
)
from dsart.coarse_dm.stage2_shifts import compute_stage2_shifts  # noqa: E402
from dsart.common.constants import (  # noqa: E402
    K_DM_MS_GHZ2_PC,
    N_CHGROUP,
    NANTS,
    NATIVE_SAMPLE_US,
    NBASE,
)
from dsart.fine_dm.combiner import compute_time_shift_search  # noqa: E402
from dsart.grid.kernel import FastVisGridder  # noqa: E402
from dsart.grid.sparsity_pattern import (  # noqa: E402
    IMAGE_PIXEL_ARCSEC_TEE,
    cell_lambda_for_pixel_arcsec,
)
from dsart.grid.subband import (  # noqa: E402
    build_subband_layout,
    build_subband_patterns,
)
from dsart.services.corr_fast_integration import (  # noqa: E402
    Stage1MultiDMCoarseDM,
)

CSF = 8
NCH = 384 // CSF
T_US = 32 * NATIVE_SAMPLE_US
N_GRID = 256
N_FV = 160
DEV = "cuda:0"
COARSE = np.array([174.47, 327.13, 487.23, 654.79, 829.79, 1012.23,
                   1202.13, 1399.47])
HALF_BUCKET = np.array([71.5, 76.6, 79.4, 82.8, 87.3, 91.6, 94.9, 98.6])


def _setup():
    rng = np.random.default_rng(7)
    e = rng.uniform(-150, 150, NANTS)
    n = rng.uniform(-150, 150, NANTS)
    freqs = build_summed_chgroup_freq_table_GHz(CSF)
    plan = DMPlan(
        dm_pc_cc=COARSE.copy(), n_fine_per_coarse=25, t_int_fast_us=T_US,
        chgroup_freqs_GHz=freqs,
        _delay_native_samples_table=compute_delay_native_samples_table(
            COARSE, freqs),
        chan_sum_factor=CSF,
    )
    cell = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, N_GRID)
    per = {}
    for g in range(N_CHGROUP):
        per[g] = {}
        whole = build_subband_patterns(
            e, n, chgroup=g, n_sub=1, dec_deg=71.63, n_grid=N_GRID,
            kernel_support=1, chan_sum_factor=CSF, cell_lambda=cell,
            is_core_baseline_mask=None,
        )
        # production passes the whole-chgroup gridder in both modes
        per[g]["grid"] = FastVisGridder.from_pattern(whole[0], e, n, device=DEV)
        per[g][1] = build_subband_layout(whole, per[g]["grid"], device=DEV)
        four = build_subband_patterns(
            e, n, chgroup=g, n_sub=4, dec_deg=71.63, n_grid=N_GRID,
            kernel_support=1, chan_sum_factor=CSF, cell_lambda=cell,
            is_core_baseline_mask=None,
        )
        per[g][4] = build_subband_layout(four, per[g]["grid"], device=DEV)
    return plan, freqs, per


WIDTHS = (1, 2, 3, 4, 6, 8)


def _boxcar_snr(x):
    """The detector's statistic: max over boxcar widths of sum / sqrt(w).

    For a burst of total fluence F this is F when it lands in one sample
    and F/sqrt(w) when smeared over w, i.e. exactly the S/N the matched
    filter recovers in white noise.
    """
    c = np.concatenate([[0.0], np.cumsum(x)])
    return max(float(np.max(c[w:] - c[:-w])) / np.sqrt(w) for w in WIDTHS)


def _run(plan, freqs, per, c, dm_true):
    """Recovered fraction for n_sub = 1 and 4 at one true DM."""
    fine = np.linspace(COARSE[c] - HALF_BUCKET[c], COARSE[c] + HALF_BUCKET[c], 25)
    f2c = np.zeros(25, dtype=np.int64)
    kw = dict(coarse_dm_pc_cm3=np.array([COARSE[c]]), fine_dm_pc_cm3=fine,
              fine_to_coarse=f2c, t_int_search_us=T_US,
              merge_coarse_rounding=True, t_int_corr_us=T_US)
    sh = {1: compute_time_shift_search(**kw).shifts,
          4: compute_time_shift_search(
              **kw, n_sub=4,
              nu_subband_ref_GHz=subband_ref_freqs_table_GHz(CSF, 4)).shifts}
    top = float(freqs[0, 0])
    out = {}
    for n_sub in (1, 4):
        series = []            # (stream column, abs start, series)
        ideal = 0
        for g in range(N_CHGROUP):
            lay = per[g][n_sub]
            nu = freqs[g]
            # absolute arrival sample of each summed channel at dm_true
            arr = np.rint(1e3 * K_DM_MS_GHZ2_PC * dm_true
                          * (1 / nu ** 2 - 1 / top ** 2) / T_US).astype(int)
            origin = int(arr.min()) - 8          # local sample 0, absolute
            vis = torch.zeros((N_FV, NBASE, NCH), dtype=torch.complex64,
                              device=DEV)
            for ch in range(NCH):
                vis[arr[ch] - origin, :, ch] = 1.0
            st1 = Stage1MultiDMCoarseDM(
                plan=plan, gridder=per[g]["grid"],
                chgroup=g, dm_indices=np.array([c]), sliding_window=False,
                subband_layout=None if n_sub == 1 else lay,
            )
            cube = st1.dedisperse_from_vis(vis, block_n=0)[0]      # (T, cells)
            s2 = int(compute_stage2_shifts(
                chgroup=g, coarse_dm_pc_cm3=np.array([COARSE[c]]),
                t_int_corr_us=T_US).shifts_samples[0])
            cim = lay.cell_index_map
            ideal += int((cim < lay.n_filled_total).sum().item())
            for s in range(n_sub):
                sl = lay.cell_slice(s)
                ts = cube[:, sl].real.sum(dim=1).double().cpu().numpy()
                series.append((g * n_sub + s, origin + s2, ts))
        lo = min(start + int(sh[n_sub][:, col].min()) for col, start, _ in series)
        hi = max(start + int(sh[n_sub][:, col].max()) + len(ts)
                 for col, start, ts in series)
        best = 0.0
        for f in range(25):
            acc = np.zeros(hi - lo)
            for col, start, ts in series:
                off = start + int(sh[n_sub][f, col]) - lo
                acc[off:off + len(ts)] += ts
            best = max(best, _boxcar_snr(acc))
        out[n_sub] = best / ideal
    return out


def test_n_sub_4_recovers_more_and_n_sub_1_is_sane():
    plan, freqs, per = _setup()
    rows = []
    for c in (0, 3, 7):
        # a uniform sweep across the whole bucket, edge to edge
        for dm in np.linspace(COARSE[c] - 0.98 * HALF_BUCKET[c],
                              COARSE[c] + 0.98 * HALF_BUCKET[c], 13):
            r = _run(plan, freqs, per, c, dm)
            rows.append((c, dm, r[1], r[4]))
    print("\ncoarse  dm_true   d/half   n_sub=1  n_sub=4   gain")
    for c, dm, a, b in rows:
        print("  %d   %8.2f   %+.2f    %.4f   %.4f   x%.3f"
              % (c, dm, (dm - COARSE[c]) / HALF_BUCKET[c], a, b, b / a))
    a_mean = np.mean([a for _, _, a, _ in rows])
    b_mean = np.mean([b for _, _, _, b in rows])
    print("mean recovered: n_sub=1 %.4f  n_sub=4 %.4f  -> x%.3f"
          % (a_mean, b_mean, b_mean / a_mean))
    for c, dm, a, b in rows:
        # each sub-band rounds on its own, so at small |d| the rint luck
        # can go either way by ~1%; beyond that sub-banding must not lose
        assert b >= a - 0.03, (c, dm, a, b)
        if abs(dm - COARSE[c]) > 0.6 * HALF_BUCKET[c]:
            assert b > a, (c, dm, a, b)
    # the offline model predicts x1.06-1.09 averaged over the bucket at
    # W ~ 1 ms (dm-smearing-subband-architecture); require a clear gain
    assert b_mean / a_mean > 1.04
