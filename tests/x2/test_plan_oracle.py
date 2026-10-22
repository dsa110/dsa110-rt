"""TieredDmPlan == the snr2x oracle, bit for bit (docs/x2/SPEC.md §1-2).

The oracle is ``docs/timeres_study/snr2x/schemes.py::proposed(q1=262.144, n_sub)``
(+ ``snr_model.Scheme`` for the integer shift tables, ``model/physics.py`` for
``gen_dm_list`` and F_CTR). ``snr_model`` binds ``dsart`` through an importlib
hack only when ``dsart`` is not yet imported, so this module imports ``dsart``
FIRST and then asserts that both the package and the oracle files come from
the checkout under test (on the node harness: the scratch copy).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import dsart

REPO = Path(__file__).resolve().parents[2]
SNR2X = REPO / "docs" / "timeres_study" / "snr2x"
MODEL = REPO / "docs" / "timeres_study" / "model"


def _under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _load_oracle():
    assert _under(dsart.__file__, REPO / "src"), f"dsart imported from {dsart.__file__}, not {REPO}/src"
    added = [str(SNR2X), str(MODEL)]
    for p in added:
        sys.path.insert(0, p)
    try:
        spec = importlib.util.spec_from_file_location("x2_oracle_schemes", SNR2X / "schemes.py")
        schemes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schemes)
    finally:
        for p in added:
            sys.path.remove(p)
    snr_model, physics = sys.modules["snr_model"], sys.modules["physics"]
    for mod, where in ((schemes, SNR2X), (snr_model, SNR2X), (physics, MODEL)):
        assert _under(mod.__file__, where), f"{mod.__name__} loaded from {mod.__file__}"
    # the oracle must have used THIS dsart (not rebound another checkout)
    assert sys.modules["dsart"] is dsart
    assert _under(sys.modules["dsart.coarse_dm.dm_plan"].__file__, REPO / "src")
    return schemes, snr_model, physics


schemes, snr_model, physics = _load_oracle()

from dsart.coarse_dm.dm_plan import (  # noqa: E402
    DMPlan,
    build_summed_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
)
from dsart.x2.opoint import opoint_2x  # noqa: E402
from dsart.x2.plan import (  # noqa: E402
    F_CTR_MHZ,
    build_tiered_plan,
    build_tiered_plan_for_opoint,
    gen_dm_list,
    summed_freqs_GHz,
)


@pytest.fixture(scope="module", params=[2, 4])
def pair(request):
    n_sub = request.param
    plan = build_tiered_plan_for_opoint(opoint_2x(n_sub=n_sub), git_sha="test", build_utc_ns=0)
    return n_sub, plan, schemes.proposed(q1=262.144, n_sub=n_sub)


def test_band_centre_and_gen_dm_list_are_the_oracles():
    assert F_CTR_MHZ == schemes.F_CTR == 1405.0152587890625
    for args in ((50.0, 1600.0, 524.288, physics.DNU_NATIVE_MHZ * 2, 3072),
                 (1600.0, 3000.0, 1048.576, physics.DNU_NATIVE_MHZ * 2, 3072),
                 (100.0, 1500.0, 1048.576, physics.DNU_NATIVE_MHZ * 8, 768)):
        for tol in (1.25, 1.4):
            a = gen_dm_list(*args, F_CTR_MHZ, tol=tol)
            b = physics.gen_dm_list(*args, schemes.F_CTR, tol=tol)
            assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("csf", [1, 2, 8])
def test_summed_frequencies_are_the_production_table(csf):
    assert np.array_equal(summed_freqs_GHz(csf), build_summed_chgroup_freq_table_GHz(csf))


def test_tiers_equal_oracle(pair):
    n_sub, plan, s = pair
    assert (s.csf, s.n_sub) == (plan.chan_sum_factor, plan.n_sub)
    assert len(s.tiers) == plan.n_tiers == 2
    for i, t in enumerate(s.tiers):
        mine = plan.tier_dict(i)
        for k in ("fine", "coarse", "f2c"):
            assert np.array_equal(np.asarray(t[k]), mine[k]), (i, k)
        assert t["q1"] == mine["q1"] and t["q2"] == mine["q2"]
    # global concatenation, plan rev 4 numbers
    assert plan.n_fine == 526 + 232
    assert [int(x) for x in np.diff(plan.tier_coarse_offsets)] == ([32, 15] if n_sub == 2 else [16, 8])


def test_stage1_bins_equal_oracle(pair):
    _, plan, s = pair
    for i, t in enumerate(s.tiers):
        c0 = int(plan.tier_coarse_offsets[i])
        for c in range(len(t["coarse"])):
            ref = s.stage1_bins(i, c)
            assert np.array_equal(ref, plan.stage1_bins[:, :, c0 + c]), (i, c)


def test_stage2_bins_equal_oracle(pair):
    _, plan, s = pair
    for i, t in enumerate(s.tiers):
        f0 = int(plan.tier_fine_offsets[i])
        for f in range(len(t["fine"])):
            ref = s.stage2_bins(i, f).reshape(-1)        # (16, n_sub) -> stream g*n_sub+s
            assert np.array_equal(ref, plan.stage2_bins[f0 + f]), (i, f)


def test_applied_delays_equal_oracle(pair):
    """The full applied per-native-channel delay (stage 2 - stage 1) of every trial."""
    _, plan, s = pair
    k = np.arange(384) // plan.chan_sum_factor
    sb = k // (plan.nchan_eff // plan.n_sub)
    for f in range(0, plan.n_fine, 7):
        i = int(plan.fine_tier[f])
        d = int(plan.fine_to_coarse[f])
        q1 = int(plan.tier_q1_native[i]) * 32.768
        q2 = int(plan.tier_t_native[i]) * 32.768
        s2 = plan.stage2_bins[f].reshape(16, plan.n_sub)
        mine = s2[:, sb] * q2 - plan.stage1_bins[:, :, d][:, k] * q1
        ref = s.applied_us(i, f - int(plan.tier_fine_offsets[i]))
        assert np.array_equal(mine, ref), f


def test_stage1_has_delay_bins_per_subband_semantics(pair):
    """Same two roundings as production DMPlan.delay_bins_per_subband at the 2x fast sample."""
    n_sub, plan, _ = pair
    freqs = build_summed_chgroup_freq_table_GHz(plan.chan_sum_factor)
    for i in range(plan.n_tiers):
        c = slice(int(plan.tier_coarse_offsets[i]), int(plan.tier_coarse_offsets[i + 1]))
        coarse = np.array(plan.coarse_dm[c])
        dp = DMPlan(dm_pc_cc=coarse, n_fine_per_coarse=1,
                    t_int_fast_us=int(plan.tier_q1_native[i]) * 32.768, chgroup_freqs_GHz=freqs,
                    _delay_native_samples_table=compute_delay_native_samples_table(coarse, freqs),
                    chan_sum_factor=plan.chan_sum_factor)
        for g in range(16):
            assert np.array_equal(dp.delay_bins_per_subband(g, n_sub), plan.stage1_bins[g, :, c]), (i, g)


def test_subband_reference_is_the_production_rule(pair):
    from dsart.coarse_dm.dm_plan import subband_ref_freqs_table_GHz
    n_sub, plan, s = pair
    assert np.array_equal(plan.subband_ref_GHz, subband_ref_freqs_table_GHz(plan.chan_sum_factor, n_sub))
    assert np.array_equal(plan.stage1_coeff_us, s.a1)
    assert np.array_equal(plan.stage2_coeff_us, s.a2)


def test_builder_parameters_are_the_opoint_defaults():
    a = build_tiered_plan(n_sub=2, git_sha="x", build_utc_ns=0)
    b = build_tiered_plan_for_opoint(opoint_2x(n_sub=2), git_sha="x", build_utc_ns=0)
    for name in ("fine_dm", "coarse_dm", "stage1_bins", "stage2_bins"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
