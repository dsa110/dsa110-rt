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
    VOLTAGES_SHAPE,
)
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    ClusterRecord,
    CubeDumpManifest,
    CubeGeometry,
    DmPlan,
    SparseCOOPayload,
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
    bits_per_cell: int = 16,
    chgroup: int = 0,
    n_grid: int = 256,
) -> SparseCOOPayload:
    dtype = "int8" if bits_per_cell == 16 else "float16"
    return SparseCOOPayload(
        values=np.zeros((n_filled, 2), dtype=dtype),
        bits_per_cell=bits_per_cell,
        chgroup=chgroup,
        dm_idx=12,
        specnum=1_234_567,
        n_grid=n_grid,
        n_filled=n_filled,
        pattern_id=0xDEADBEEFCAFEBABE,
        t_int=16,
        scale=0.0123,
        offset=-0.5,
    )


def test_sparse_happy_cint8() -> None:
    p = _make_sparse(bits_per_cell=16)
    assert p.values.dtype.name == "int8"


def test_sparse_happy_cfp16() -> None:
    p = _make_sparse(bits_per_cell=32)
    assert p.values.dtype.name == "float16"


def test_sparse_rejects_bits_per_cell_8() -> None:
    # 8 bits-per-COMPLEX-cell is invalid (would be 4-bit components which we
    # don't support); valid values are {16, 32} per F2 (revised at hardening).
    with pytest.raises(ValueError, match="bits_per_cell"):
        _make_sparse(bits_per_cell=8)


def test_sparse_rejects_dtype_mismatch() -> None:
    with pytest.raises(TypeError, match="values.dtype"):
        SparseCOOPayload(
            values=np.zeros((10, 2), dtype="float16"),
            bits_per_cell=16,  # claims cint8 but dtype is float16
            chgroup=0,
            dm_idx=0,
            specnum=0,
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
            bits_per_cell=16,
            chgroup=0,
            dm_idx=0,
            specnum=0,
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
            bits_per_cell=16,
            chgroup=0,
            dm_idx=0,
            specnum=0,
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
# TriggerPacket / TriggerAck contract tests — RETIRED in M6 chunk-9 sweep
# (the dataclasses were dropped from src/dsart/common/contracts.py;
# voltage-trigger handoff is now operator-mediated through legacy
# dsa110-xengine; see plan §M6 / §M-defer for the deferred reactivation
# path that will re-introduce wire-form trigger contracts under whatever
# transport is chosen).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CubeGeometry — M6 chunk 1
# ---------------------------------------------------------------------------


def _make_cube_geometry(**overrides: object) -> CubeGeometry:
    base: dict = dict(
        cube_id=0,
        specnum_start=1024,
        sample_period_specnum=16,
        t_det=256,
        n_grid=256,
        n_fdm_in_cube=32,
        sample_period_us=131.072,
        cell_l_rad=1.5e-4,
        cell_m_rad=1.5e-4,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=np.linspace(50.0, 800.0, 32, dtype=np.float64),
        mjd_start=60942.123456789,
    )
    base.update(overrides)
    return CubeGeometry(**base)  # type: ignore[arg-type]


def test_cube_geometry_happy() -> None:
    geom = _make_cube_geometry()
    assert geom.cube_id == 0
    assert geom.fine_dm_pc_cc.shape == (32,)
    assert geom.fine_dm_pc_cc.dtype.name == "float64"


@pytest.mark.parametrize(
    "field, value, exc",
    [
        ("cube_id", -1, ValueError),
        ("specnum_start", -1, ValueError),
        ("sample_period_specnum", 0, ValueError),
        ("t_det", 0, ValueError),
        ("n_grid", 100, ValueError),  # not power of two
        ("n_grid", 0, ValueError),
        ("n_fdm_in_cube", 0, ValueError),
        ("sample_period_us", 0.0, ValueError),
        ("cell_l_rad", 0.0, ValueError),
        ("cell_m_rad", -1e-4, ValueError),
        ("mjd_start", float("nan"), ValueError),
    ],
)
def test_cube_geometry_rejects_bad_scalar(field, value, exc) -> None:
    with pytest.raises(exc):
        _make_cube_geometry(**{field: value})


def test_cube_geometry_rejects_fine_dm_dtype_mismatch() -> None:
    bad = np.linspace(50.0, 800.0, 32, dtype=np.float32)
    with pytest.raises(TypeError, match="fine_dm_pc_cc.dtype"):
        _make_cube_geometry(fine_dm_pc_cc=bad)


def test_cube_geometry_rejects_fine_dm_shape_mismatch() -> None:
    bad = np.linspace(50.0, 800.0, 31, dtype=np.float64)
    with pytest.raises(ValueError, match="fine_dm_pc_cc.shape"):
        _make_cube_geometry(fine_dm_pc_cc=bad)


def test_cube_geometry_rejects_fine_dm_wrong_type() -> None:
    with pytest.raises(TypeError, match="fine_dm_pc_cc must be np.ndarray"):
        _make_cube_geometry(fine_dm_pc_cc=[50.0] * 32)  # type: ignore[arg-type]


def test_cube_geometry_is_frozen() -> None:
    geom = _make_cube_geometry()
    with pytest.raises((AttributeError, Exception)):
        geom.cube_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ClusterRecord — M6 chunk 1
# ---------------------------------------------------------------------------


def _make_cluster_record(**overrides: object) -> ClusterRecord:
    base: dict = dict(
        cluster_id=0,
        cube_id=0,
        cntc=3,
        cntb_lm=2,
        cntb_dm=2,
        peak_candidate_idx=1,
        l_rad=1.5e-4 * 132,
        m_rad=1.5e-4 * 230,
        l_pix=132,
        m_pix=230,
        dm_fine_pc_cc=397.42,
        fine_dm_idx=15,
        t_in_cube=64,
        t_seconds=64 * 131.072e-6,
        width_samples=4,
        snr=20.81,
        kernel_id="unit:d1:b4",
        event_specnum=2048,
        search_node_id=2,
        gpu_half=1,
    )
    base.update(overrides)
    return ClusterRecord(**base)  # type: ignore[arg-type]


def test_cluster_record_happy() -> None:
    cr = _make_cluster_record()
    assert cr.cluster_id == 0
    assert cr.cntc == 3
    assert cr.cntb_lm == 2
    assert cr.cntb_dm == 2


def test_cluster_record_noise_label_ok() -> None:
    cr = _make_cluster_record(cluster_id=-1, cntc=1, cntb_lm=1, cntb_dm=1)
    assert cr.cluster_id == -1


@pytest.mark.parametrize(
    "field, value, exc, match",
    [
        ("cluster_id", -2, ValueError, "cluster_id"),
        ("cube_id", -1, ValueError, "cube_id"),
        ("cntc", 0, ValueError, "cntc"),
        ("cntb_lm", 0, ValueError, "cntb_lm"),
        ("cntb_lm", 5, ValueError, "cntb_lm"),  # > cntc=3
        ("cntb_dm", 0, ValueError, "cntb_dm"),
        ("cntb_dm", 5, ValueError, "cntb_dm"),
        ("peak_candidate_idx", -1, ValueError, "peak_candidate_idx"),
        ("l_pix", -1, ValueError, "l_pix"),
        ("m_pix", -1, ValueError, "m_pix"),
        ("dm_fine_pc_cc", -1.0, ValueError, "dm_fine_pc_cc"),
        ("fine_dm_idx", -1, ValueError, "fine_dm_idx"),
        ("t_in_cube", -1, ValueError, "t_in_cube"),
        ("width_samples", 0, ValueError, "width_samples"),
        ("event_specnum", -1, ValueError, "event_specnum"),
        ("kernel_id", "bogus", ValueError, "kernel_id"),
        ("search_node_id", -1, ValueError, "search_node_id"),
        ("search_node_id", N_SEARCH, ValueError, "search_node_id"),
        ("gpu_half", -1, ValueError, "gpu_half"),
        ("gpu_half", N_SEARCH_GPU, ValueError, "gpu_half"),
    ],
)
def test_cluster_record_rejects_bad_field(field, value, exc, match) -> None:
    with pytest.raises(exc, match=match):
        _make_cluster_record(**{field: value})


def test_cluster_record_is_frozen() -> None:
    cr = _make_cluster_record()
    with pytest.raises((AttributeError, Exception)):
        cr.cluster_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CubeDumpManifest — M6 chunk 3
# ---------------------------------------------------------------------------


def _make_cube_dump_manifest(**overrides: object) -> CubeDumpManifest:
    cluster_record = _make_cluster_record()
    base: dict = dict(
        cube_id=0,
        event_specnum_start=1024,
        mjd_start=60942.123456789,
        t_det=256,
        n_fdm_in_cube=32,
        n_grid=256,
        trigger_source="auto",
        cluster_record=cluster_record,
        npz_path="/tmp/dump_s2_g1_1024.npz",
        search_node_id=2,
        gpu_half=1,
    )
    base.update(overrides)
    return CubeDumpManifest(**base)  # type: ignore[arg-type]


def test_cube_dump_manifest_auto_happy() -> None:
    m = _make_cube_dump_manifest()
    assert m.trigger_source == "auto"
    assert m.cluster_record is not None


def test_cube_dump_manifest_udp_happy() -> None:
    m = _make_cube_dump_manifest(trigger_source="udp", cluster_record=None)
    assert m.trigger_source == "udp"
    assert m.cluster_record is None


def test_cube_dump_manifest_auto_requires_cluster_record() -> None:
    with pytest.raises(ValueError, match="trigger_source='auto'"):
        _make_cube_dump_manifest(trigger_source="auto", cluster_record=None)


def test_cube_dump_manifest_udp_rejects_cluster_record() -> None:
    cr = _make_cluster_record()
    with pytest.raises(ValueError, match="trigger_source='udp'"):
        _make_cube_dump_manifest(trigger_source="udp", cluster_record=cr)


@pytest.mark.parametrize(
    "field, value, exc, match",
    [
        ("cube_id", -1, ValueError, "cube_id"),
        ("event_specnum_start", -1, ValueError, "event_specnum_start"),
        ("mjd_start", float("nan"), ValueError, "mjd_start"),
        ("t_det", 0, ValueError, "t_det"),
        ("n_fdm_in_cube", 0, ValueError, "n_fdm_in_cube"),
        ("n_grid", 100, ValueError, "n_grid"),
        ("trigger_source", "manual", ValueError, "trigger_source"),
        ("npz_path", "", ValueError, "npz_path"),
        ("search_node_id", -1, ValueError, "search_node_id"),
        ("gpu_half", -1, ValueError, "gpu_half"),
    ],
)
def test_cube_dump_manifest_rejects_bad_field(field, value, exc, match) -> None:
    with pytest.raises(exc, match=match):
        _make_cube_dump_manifest(**{field: value})


def test_cube_dump_manifest_is_frozen() -> None:
    m = _make_cube_dump_manifest()
    with pytest.raises((AttributeError, Exception)):
        m.cube_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cross-module units invariant: the in-cube time index (2026-08-06)
# ---------------------------------------------------------------------------
#
# Three independent consumers recover "how far into the cube was this
# candidate" from the SAME two numbers -- the cube's sample-0 anchor and
# the C1 row's event_specnum, both in SEARCH-sample units:
#
#   coinc/plotter._metadata_t_idx      -> the plotted time index
#   cluster/features._candidate_to_int_indices -> t_in_cube
#   coinc/wire.C1BatchHeader.candidate_mjd     -> the reported event MJD
#
# All three had independently acquired a `// sample_period_specnum`,
# which is native SNAP specnums per search sample and belongs only to
# native-specnum conversion (services/coincidencer.search_to_snap_specnum;
# see services/search_compute.py:1338-1345). This test plants ONE offset
# and requires all three to report it, so the next person who adds a
# divisor to any one of them breaks a contract test rather than a night
# of observing.


def test_in_cube_offset_agrees_across_plotter_features_and_wire() -> None:
    import dataclasses

    from dsart.cluster.features import _candidate_to_int_indices
    from dsart.coinc import wire
    from dsart.coinc.plotter import _BurstPeak, _CubeChunk

    PLANTED_T = 118          # ground truth, in search samples
    ANCHOR = 82_556_992      # cube sample-0 specnum (search samples)
    PERIOD_SPECNUM = 16      # native specnums per search sample
    PERIOD_US = 1048.576     # search-sample period
    T_DET = 192
    MJD_START = 60781.5

    event_specnum = ANCHOR + PLANTED_T

    # --- plotter -----------------------------------------------------
    chunk = _CubeChunk(
        search_node_id=2, gpu_half=1, event_specnum=event_specnum,
        cube=np.zeros((T_DET, 1, 1, 1), dtype=np.float16),
        fine_dm_pc_cc=np.asarray([500.0], dtype=np.float64),
        mjd_start=MJD_START, sample_period_us=PERIOD_US,
        cube_specnum_start=ANCHOR, sample_period_specnum=PERIOD_SPECNUM,
        cube_mjd_start=MJD_START,
    )
    peak = _BurstPeak(
        search_node_id=2, gpu_half=1, fine_dm_idx=0, l_pix=0, m_pix=0,
        dm_pc_cc=500.0, snr=12.0, width_samples=4,
        kernel_id="psf:d3:b16", source="members",
        event_specnum=event_specnum,
    )
    from dsart.coinc.plotter import _metadata_t_idx
    plotter_t = _metadata_t_idx(chunk, peak)

    # --- legacy clusterer --------------------------------------------
    geom = _make_cube_geometry(
        specnum_start=ANCHOR,
        sample_period_specnum=PERIOD_SPECNUM,
        sample_period_us=PERIOD_US,
        t_det=T_DET,
        mjd_start=MJD_START,
    )
    cand = _make_candidate(
        event_specnum=event_specnum,
        dm_fine=float(geom.fine_dm_pc_cc[0]),
    )
    _, _, _, features_t = _candidate_to_int_indices(cand, geom)

    # --- wire (implied offset, back out of the MJD) -------------------
    header = wire.build_header(
        cube_id=7, event_specnum_start=ANCHOR, mjd_start=MJD_START,
        sample_period_specnum=PERIOD_SPECNUM, sample_period_us=PERIOD_US,
        n_grid=256, n_fdm_in_cube=34, search_node_id=1, gpu_half=0,
        n_candidates=0,
    )
    wire_t = (
        (header.candidate_mjd(event_specnum) - MJD_START) * 86400.0
        / (PERIOD_US * 1e-6)
    )

    assert plotter_t == PLANTED_T
    assert features_t == PLANTED_T
    # Loose absolute tolerance: an MJD near 60781 has ~1 µs of float64
    # granularity, i.e. ~1e-3 of a 1048.576 µs search sample. The bug
    # this guards against is a factor of 16, not a rounding.
    assert wire_t == pytest.approx(PLANTED_T, abs=1e-3)
    # And none of them is the divided-by-period answer.
    assert PLANTED_T // PERIOD_SPECNUM != PLANTED_T
    assert dataclasses.is_dataclass(chunk)  # (chunk kept alive to here)
