"""Shape, dtype, and invariant tests for ``dsart.common.contracts``.

Plan §8 M1 DoD line 2141: validates ``__post_init__`` asserts on each
contract dataclass under ``DSART_TEST=1``. Asserts are designed to
short-circuit when ``DSART_TEST`` is unset (production hot path), so
this file forces ``DSART_TEST=1`` at module import time before
importing ``dsart`` modules.

Test policy:
  - Build a minimal "happy-path" factory for each dataclass.
  - For each enforced invariant, perturb a single field and confirm
    a ``ValueError`` / ``TypeError`` is raised.
  - Round-trip ``DmPlan`` through ``to_npz`` / ``from_npz``.
"""

from __future__ import annotations

import os

# CRITICAL: Set DSART_TEST=1 BEFORE importing dsart (it's evaluated at
# import time in config_loader). Tests assume asserts are active.
os.environ["DSART_TEST"] = "1"

import json  # noqa: E402
from dataclasses import fields  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dsart.common import config_loader  # noqa: E402
from dsart.common.constants import (  # noqa: E402
    DM_PLAN_METADATA_VERSION,
    NCHAN_PER_CHGROUP,
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    NU_BOT_PROC_GHZ,
    NU_TOP_PROC_GHZ,
    BW_PROC_MHZ,
    N_CHAN_PROC_NATIVE,
    TRIGGER_OPERATOR_SEARCH_NODE_ID,
    VOLTAGES_SHAPE,
)
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    DmPlan,
    SparseCOOPayload,
    TriggerAck,
    TriggerPacket,
    Voltages,
)


def test_dsart_test_is_active() -> None:
    """Sanity: DSART_TEST=1 was honoured by config_loader."""
    assert config_loader.DSART_TEST is True


# ---------------------------------------------------------------------------
# Voltages
# ---------------------------------------------------------------------------


def _make_voltages(
    *, dtype: str = "int8", shape: tuple = VOLTAGES_SHAPE
) -> Voltages:
    return Voltages(
        tensor=np.zeros(shape, dtype=dtype),
        specnum0=1_234_567,
        utc_block_start_ns=1_872_345_677_000_000_000,
    )


def test_voltages_happy_int8() -> None:
    _make_voltages(dtype="int8")


def test_voltages_happy_float16() -> None:
    _make_voltages(dtype="float16")


def test_voltages_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        _make_voltages(shape=(2048, 96, 384, 2))


def test_voltages_rejects_bad_dtype() -> None:
    with pytest.raises(TypeError, match="dtype"):
        _make_voltages(dtype="float32")


def test_voltages_rejects_negative_specnum0() -> None:
    with pytest.raises(ValueError, match="specnum0"):
        Voltages(
            tensor=np.zeros(VOLTAGES_SHAPE, dtype="int8"),
            specnum0=-1,
            utc_block_start_ns=0,
        )


def test_voltages_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError, match="ndarray"):
        Voltages(
            tensor=[0] * 100,  # type: ignore[arg-type]
            specnum0=0,
            utc_block_start_ns=0,
        )


# ---------------------------------------------------------------------------
# SparseCOOPayload
# ---------------------------------------------------------------------------


def _make_sparse(
    *,
    n_filled: int = 5800,
    bits_per_cell: int = 8,
    chgroup: int = 0,
    n_grid: int = 256,
) -> SparseCOOPayload:
    dtype = "int8" if bits_per_cell == 8 else "float16"
    return SparseCOOPayload(
        values=np.zeros((n_filled, 2), dtype=dtype),
        bits_per_cell=bits_per_cell,
        chgroup=chgroup,
        dm_idx=12,
        specnum=1_234_567,
        utc_block_start_ns=1_872_345_677_000_000_000,
        n_grid=n_grid,
        n_filled=n_filled,
        pattern_id=0xDEADBEEFCAFEBABE,
        t_int=16,
        scale=0.0123,
        offset=-0.5,
    )


def test_sparse_happy_cint8() -> None:
    p = _make_sparse(bits_per_cell=8)
    assert p.values.dtype.name == "int8"


def test_sparse_happy_cfp16() -> None:
    p = _make_sparse(bits_per_cell=16)
    assert p.values.dtype.name == "float16"


def test_sparse_rejects_bits_per_cell_32() -> None:
    with pytest.raises(ValueError, match="bits_per_cell"):
        _make_sparse(bits_per_cell=32)


def test_sparse_rejects_dtype_mismatch() -> None:
    with pytest.raises(TypeError, match="values.dtype"):
        SparseCOOPayload(
            values=np.zeros((10, 2), dtype="float16"),
            bits_per_cell=8,  # claims cint8 but dtype is float16
            chgroup=0,
            dm_idx=0,
            specnum=0,
            utc_block_start_ns=0,
            n_grid=256,
            n_filled=10,
            pattern_id=0,
            t_int=1,
            scale=1.0,
            offset=0.0,
        )


def test_sparse_rejects_n_grid_not_pow2() -> None:
    with pytest.raises(ValueError, match="n_grid"):
        _make_sparse(n_grid=300)


def test_sparse_rejects_chgroup_out_of_range() -> None:
    with pytest.raises(ValueError, match="chgroup"):
        _make_sparse(chgroup=N_CHGROUP)


def test_sparse_rejects_values_shape_1d() -> None:
    with pytest.raises(ValueError, match="\\[N_filled, 2\\]"):
        SparseCOOPayload(
            values=np.zeros(100, dtype="int8"),
            bits_per_cell=8,
            chgroup=0,
            dm_idx=0,
            specnum=0,
            utc_block_start_ns=0,
            n_grid=256,
            n_filled=100,
            pattern_id=0,
            t_int=1,
            scale=1.0,
            offset=0.0,
        )


def test_sparse_rejects_n_filled_inconsistent() -> None:
    with pytest.raises(ValueError, match="n_filled"):
        SparseCOOPayload(
            values=np.zeros((100, 2), dtype="int8"),
            bits_per_cell=8,
            chgroup=0,
            dm_idx=0,
            specnum=0,
            utc_block_start_ns=0,
            n_grid=256,
            n_filled=99,
            pattern_id=0,
            t_int=1,
            scale=1.0,
            offset=0.0,
        )


# ---------------------------------------------------------------------------
# Candidate + CandidateFlags
# ---------------------------------------------------------------------------


def _make_candidate(**overrides: object) -> Candidate:
    base = dict(
        l=0.012,
        m=-0.034,
        dm_fine=524.6,
        dm_idx=87,
        event_specnum=12_345_678,
        width_samples=4,
        kernel_id="psf:d3:b16",
        snr=9.7,
        detector_version="v1.deterministic.20260501",
        flags=int(CandidateFlags.NONE),
        search_node_id=2,
        gpu_half=1,
    )
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


def test_candidate_happy() -> None:
    _make_candidate()


def test_candidate_kernel_id_psf_shift_lm_ok() -> None:
    _make_candidate(kernel_id="psf_shift_lm:d5:b32")


def test_candidate_rejects_bad_kernel_id_format() -> None:
    with pytest.raises(ValueError, match="kernel_id"):
        _make_candidate(kernel_id="psf:d3")


def test_candidate_rejects_bad_kernel_id_image() -> None:
    with pytest.raises(ValueError, match="image token"):
        _make_candidate(kernel_id="ringed:d3:b16")


def test_candidate_rejects_bad_kernel_id_dm() -> None:
    with pytest.raises(ValueError, match="dm token"):
        _make_candidate(kernel_id="psf:d2:b16")


def test_candidate_rejects_bad_kernel_id_time() -> None:
    with pytest.raises(ValueError, match="time token"):
        _make_candidate(kernel_id="psf:d3:b3")


def test_candidate_rejects_search_node_4() -> None:
    with pytest.raises(ValueError, match="search_node_id"):
        _make_candidate(search_node_id=N_SEARCH)


def test_candidate_rejects_gpu_half_2() -> None:
    with pytest.raises(ValueError, match="gpu_half"):
        _make_candidate(gpu_half=N_SEARCH_GPU)


def test_candidate_flags_bitmask_combine() -> None:
    f = CandidateFlags.NOISE_WARMUP | CandidateFlags.HALO_DROPPED
    c = _make_candidate(flags=int(f))
    assert c.flags & int(CandidateFlags.NOISE_WARMUP)
    assert c.flags & int(CandidateFlags.HALO_DROPPED)
    assert not (c.flags & int(CandidateFlags.RFI_WARMING_UP))


def test_candidate_frozen() -> None:
    c = _make_candidate()
    with pytest.raises((AttributeError, TypeError)):
        c.snr = 100.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TriggerPacket
# ---------------------------------------------------------------------------


def _make_trigger(**overrides: object) -> TriggerPacket:
    base = dict(
        trigger_id="s2-g1-000123",
        search_node_id=2,
        emit_utc_ns=1_872_345_678_901_234_567,
        event_specnum=12_345_678,
        event_utc_ns=1_872_345_677_000_000_000,
        l=0.012,
        m=-0.034,
        dm_fine=524.6,
        dm_idx=87,
        fine_dm_trial=87,
        width_samples=4,
        kernel_id="psf:d3:b16",
        snr=9.7,
        actions={"voltage_dump": True, "filterbank": True, "n_beams": 1},
        priority="normal",
        src_name="auto_20260430_142511_b3",
    )
    base.update(overrides)
    return TriggerPacket(**base)  # type: ignore[arg-type]


def test_trigger_happy() -> None:
    _make_trigger()


def test_trigger_operator_search_node_ok() -> None:
    _make_trigger(
        search_node_id=TRIGGER_OPERATOR_SEARCH_NODE_ID,
        trigger_id="op-1872345678901234567",
        priority="high",
    )


def test_trigger_rejects_bad_priority() -> None:
    with pytest.raises(ValueError, match="priority"):
        _make_trigger(priority="urgent")


def test_trigger_rejects_search_node_5() -> None:
    with pytest.raises(ValueError, match="search_node_id"):
        _make_trigger(search_node_id=5)


def test_trigger_rejects_empty_trigger_id() -> None:
    with pytest.raises(ValueError, match="trigger_id"):
        _make_trigger(trigger_id="")


def test_trigger_n_pre_blocks_none_ok() -> None:
    p = _make_trigger(n_pre_blocks=None, n_post_blocks=None)
    assert p.n_pre_blocks is None and p.n_post_blocks is None


# ---------------------------------------------------------------------------
# TriggerAck
# ---------------------------------------------------------------------------


def test_trigger_ack_accepted_happy() -> None:
    TriggerAck(
        trigger_id="s2-g1-000123",
        stage="accepted",
        ack_utc_ns=1_872_345_678_901_234_567,
        accepted=True,
        queue_depth=3,
    )


def test_trigger_ack_accepted_dup() -> None:
    TriggerAck(
        trigger_id="s2-g1-000123",
        stage="accepted",
        ack_utc_ns=1_872_345_678_901_234_567,
        accepted=False,
        reason="dup",
        dup_of="s1-g0-000099",
    )


def test_trigger_ack_completed_happy() -> None:
    TriggerAck(
        trigger_id="s2-g1-000123",
        stage="completed",
        ack_utc_ns=1_872_345_678_901_234_567,
        voltage_dump_path="/home/ubuntu/data/fl_12345678.out",
        filterbank_paths=("/home/ubuntu/data/auto_20260430_142511_b3_b0.fil",),
        dump_completion_utc_ns=1_872_345_679_201_234_567,
        dump_duration_ms=312,
    )


def test_trigger_ack_rejects_bad_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        TriggerAck(
            trigger_id="s2-g1-000123",
            stage="other",
            ack_utc_ns=0,
        )


def test_trigger_ack_accepted_requires_accepted_flag() -> None:
    with pytest.raises(ValueError, match="accepted"):
        TriggerAck(
            trigger_id="s2-g1-000123",
            stage="accepted",
            ack_utc_ns=0,
        )


def test_trigger_ack_rejected_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        TriggerAck(
            trigger_id="s2-g1-000123",
            stage="accepted",
            ack_utc_ns=0,
            accepted=False,
            reason=None,
        )


def test_trigger_ack_dup_requires_dup_of() -> None:
    with pytest.raises(ValueError, match="dup_of"):
        TriggerAck(
            trigger_id="s2-g1-000123",
            stage="accepted",
            ack_utc_ns=0,
            accepted=False,
            reason="dup",
        )


def test_trigger_ack_completed_requires_dump_fields() -> None:
    with pytest.raises(ValueError, match="dump_completion_utc_ns"):
        TriggerAck(
            trigger_id="s2-g1-000123",
            stage="completed",
            ack_utc_ns=0,
        )


def test_trigger_ack_filterbank_paths_must_be_tuple() -> None:
    with pytest.raises(TypeError, match="filterbank_paths"):
        TriggerAck(
            trigger_id="s2-g1-000123",
            stage="completed",
            ack_utc_ns=0,
            voltage_dump_path="/x",
            filterbank_paths=["/x"],  # type: ignore[arg-type]
            dump_completion_utc_ns=0,
            dump_duration_ms=0,
        )


# ---------------------------------------------------------------------------
# DmPlan
# ---------------------------------------------------------------------------


def _make_minimal_dm_plan(*, n_fine: int = 8, n_coarse: int = 3) -> DmPlan:
    fine_dm = np.linspace(0.0, 100.0, n_fine, dtype="float64")
    coarse_dm = np.linspace(0.0, 100.0, n_coarse, dtype="float64")
    fine_to_coarse = np.zeros(n_fine, dtype="int32")
    fine_offsets_idx = np.array([0, n_fine // 2, n_fine, n_fine], dtype="int32")
    fine_offsets_flat = np.zeros(n_fine, dtype="float64")
    return DmPlan(
        dm_min=0.0,
        dm_max=100.0,
        tol=1.5,
        fine_dm=fine_dm,
        coarse_dm=coarse_dm,
        fine_to_coarse=fine_to_coarse,
        fine_offsets_idx=fine_offsets_idx,
        fine_offsets_flat=fine_offsets_flat,
        time_shift_corr_stage1=np.zeros(
            (N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse), dtype="int32"
        ),
        time_shift_corr_stage2=np.zeros((N_CHGROUP, n_coarse), dtype="int32"),
        time_shift_search=np.zeros((n_fine, N_CHGROUP), dtype="int32"),
        dm_idx_range_canonical=np.zeros((N_SEARCH, 2), dtype="int32"),
        dm_idx_range_consumed=np.zeros((N_SEARCH, 2), dtype="int32"),
        dm_idx_range_canonical_per_gpu=np.zeros(
            (N_SEARCH, N_SEARCH_GPU, 2), dtype="int32"
        ),
        dm_idx_range_consumed_per_gpu=np.zeros(
            (N_SEARCH, N_SEARCH_GPU, 2), dtype="int32"
        ),
        dm_overlap_coarse=2,
        metadata={
            "band_top_GHz": NU_TOP_PROC_GHZ,
            "band_bot_GHz": NU_BOT_PROC_GHZ,
            "BW_MHz": BW_PROC_MHZ,
            "N_chan_proc_native": N_CHAN_PROC_NATIVE,
            "t_int_fast_us": 32.768,
            "t_int_search_us": 524.288,
            "tol": 1.5,
            "build_utc_ns": 1_872_345_677_000_000_000,
            "git_sha": "deadbeef",
            "version": DM_PLAN_METADATA_VERSION,
        },
    )


def _plan_kwargs(plan: DmPlan) -> dict:
    """Extract ``DmPlan`` fields as a kwargs dict (slots=True removes __dict__)."""
    return {f.name: getattr(plan, f.name) for f in fields(plan)}


def test_dm_plan_happy() -> None:
    p = _make_minimal_dm_plan()
    assert p.fine_dm.shape == (8,)
    assert p.time_shift_corr_stage1.shape == (N_CHGROUP, NCHAN_PER_CHGROUP, 3)


def test_dm_plan_rejects_bad_dtype() -> None:
    plan = _make_minimal_dm_plan()
    kwargs = _plan_kwargs(plan)
    kwargs["time_shift_corr_stage1"] = plan.time_shift_corr_stage1.astype("int64")
    with pytest.raises(TypeError, match="time_shift_corr_stage1"):
        DmPlan(**kwargs)


def test_dm_plan_rejects_non_monotone_fine_dm() -> None:
    plan = _make_minimal_dm_plan()
    bad = plan.fine_dm.copy()
    bad[3] = bad[2]
    kwargs = _plan_kwargs(plan)
    kwargs["fine_dm"] = bad
    with pytest.raises(ValueError, match="fine_dm.*increasing"):
        DmPlan(**kwargs)


def test_dm_plan_rejects_missing_metadata_key() -> None:
    plan = _make_minimal_dm_plan()
    bad_md = dict(plan.metadata)
    bad_md.pop("git_sha")
    kwargs = _plan_kwargs(plan)
    kwargs["metadata"] = bad_md
    with pytest.raises(ValueError, match="git_sha"):
        DmPlan(**kwargs)


def test_dm_plan_rejects_metadata_version_mismatch() -> None:
    plan = _make_minimal_dm_plan()
    bad_md = dict(plan.metadata)
    bad_md["version"] = 999
    kwargs = _plan_kwargs(plan)
    kwargs["metadata"] = bad_md
    with pytest.raises(ValueError, match="version"):
        DmPlan(**kwargs)


def test_dm_plan_npz_round_trip(tmp_path: Path) -> None:
    plan = _make_minimal_dm_plan()
    out = tmp_path / "dm_plan.npz"
    plan.to_npz(str(out))
    loaded = DmPlan.from_npz(str(out))
    np.testing.assert_array_equal(loaded.fine_dm, plan.fine_dm)
    np.testing.assert_array_equal(loaded.coarse_dm, plan.coarse_dm)
    np.testing.assert_array_equal(
        loaded.time_shift_corr_stage1, plan.time_shift_corr_stage1
    )
    assert loaded.dm_overlap_coarse == plan.dm_overlap_coarse
    assert loaded.metadata["git_sha"] == plan.metadata["git_sha"]
    assert loaded.metadata["version"] == DM_PLAN_METADATA_VERSION


def test_dm_plan_metadata_json_serialisable() -> None:
    plan = _make_minimal_dm_plan()
    s = json.dumps(plan.metadata)
    assert json.loads(s) == plan.metadata


# ---------------------------------------------------------------------------
# Test that DSART_TEST=0 disables asserts (for hot-path correctness)
# ---------------------------------------------------------------------------


def test_dsart_test_disabled_skips_asserts(monkeypatch: pytest.MonkeyPatch) -> None:
    """When DSART_TEST is unset, __post_init__ is a no-op.

    We can't unset DSART_TEST mid-process because it's evaluated at
    import time. Instead, monkeypatch the module-level flag and verify
    constructors that would otherwise raise now succeed.
    """
    from dsart.common import contracts as contracts_mod

    monkeypatch.setattr(contracts_mod, "DSART_TEST", False)
    Voltages(
        tensor=np.zeros((10,), dtype="float64"),  # wrong everything
        specnum0=-1,  # negative
        utc_block_start_ns=-1,
    )
