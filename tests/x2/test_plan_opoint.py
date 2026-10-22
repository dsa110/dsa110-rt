"""OpPoint presets and the 2x format factory (docs/x2/SPEC.md §1)."""
from __future__ import annotations

import dataclasses

import pytest

from dsart.common.constants import NATIVE_SAMPLE_US
from dsart.x2.opoint import (
    FORMATS_2X,
    OPPOINT_1X,
    OPPOINT_2X,
    STAGE2_CORR_FIFO_MERGED,
    STAGE2_SEARCH_SINGLE_RINT,
    OpPoint,
    TierSpec,
    opoint_2x,
    opoint_from_id,
)


def test_1x_is_todays_search():
    op = OPPOINT_1X
    assert (op.t_int_fast_native, op.chan_sum_factor, op.n_sub, op.stage1_q_native) == (32, 8, 4, 32)
    assert op.stage2_mode == STAGE2_CORR_FIFO_MERGED
    assert (op.value_kind, op.bits_main, op.bits_heavy) == ("gridded", 16, 0)
    assert op.t_ext_us == 1048.576 and op.plan_version == 2 and op.tiers == ()
    assert op.heavy_frac is None and op.normalise_values is False
    assert op.nchan_eff == 48 and op.n_streams == 64 and op.opoint_id == "1x"


def test_2x_matches_schemes_proposed():
    op = opoint_2x()
    assert (op.t_int_fast_native, op.chan_sum_factor, op.stage1_q_native) == (8, 2, 8)
    assert op.stage2_mode == STAGE2_SEARCH_SINGLE_RINT
    assert op.t_fast_us == 262.144 and op.stage1_q_us == 262.144 and op.t_ext_us == 524.288
    assert [(t.name, t.dm_lo, t.dm_hi, t.t_native) for t in op.tiers] == [
        ("A", 50.0, 1600.0, 16), ("B", 1600.0, 3000.0, 32)]
    # tier samples are the oracle's literals exactly (power-of-two multiples of 32.768)
    assert op.tiers[0].t_us == 524.288 and op.tiers[1].t_us == 1048.576
    assert op.coarse_density_t_ref_native * NATIVE_SAMPLE_US == 1048.576
    assert (op.dm_tol, op.dm_ti_us, op.coarse_density) == (1.25, 40.0, 0.0206)
    assert op.bits_heavy == 16 and op.heavy_frac == 3e-5 and op.normalise_values
    assert op.redundancy_tol_m == 0.05 and op.n_grid == 256 and op.cell_lambda_mode == "tee45"
    assert op.n_halves == 8 and op.nchan_eff == 192 and op.n_streams == 32
    assert OPPOINT_2X is opoint_2x


def test_format_table_rows_and_ids_round_trip():
    assert sorted(FORMATS_2X) == sorted([
        "2x-gridded-ns2-cint4", "2x-gridded-ns4-cint5", "2x-gridded-ns4-cint4",
        "2x-baselines-ns4-cint5", "2x-baselines-ns2-cint4", "2x-baselines-ns4-cint4"])
    for oid, op in FORMATS_2X.items():
        assert op.opoint_id == oid
        assert opoint_from_id(oid) == op
    assert opoint_from_id("1x") is OPPOINT_1X
    assert opoint_from_id("2x-baselines-ns2-cint6") == opoint_2x("baselines", 2, 12)


@pytest.mark.parametrize("kw", [dict(n_sub=3), dict(bits_main=16), dict(value_kind="cells")])
def test_factory_rejects_bad_formats(kw):
    with pytest.raises(ValueError):
        opoint_2x(**kw)


def test_validation_rejects_inconsistent_points():
    op = opoint_2x()
    with pytest.raises(ValueError):
        dataclasses.replace(op, tiers=(TierSpec("A", 50.0, 1500.0, 16), TierSpec("B", 1600.0, 3000.0, 32)))
    with pytest.raises(ValueError):
        dataclasses.replace(op, heavy_frac=None)            # bits_heavy 16 without a heavy rule
    with pytest.raises(ValueError):
        dataclasses.replace(OPPOINT_1X, tiers=op.tiers)     # v2 plan takes its grid from the file
    with pytest.raises(ValueError):
        dataclasses.replace(op, chan_sum_factor=5)
    with pytest.raises(ValueError):
        dataclasses.replace(op, n_sub=7)
    with pytest.raises(ValueError):
        TierSpec("X", 100.0, 50.0, 16)
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.n_sub = 4  # type: ignore[misc]
    assert isinstance(op, OpPoint)
