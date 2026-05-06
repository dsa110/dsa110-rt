"""Tests for ``dsart.transport.captured_npz``: the M3 → M5
captured-NPZ loader that closes F6 (M3 chunk-8 hand-off schema).

The fixture-builder helper :func:`_write_synthetic_fixture` is the
canonical way to assemble a captured run on disk in unit tests; the
bench-level test (``test_voltage_fixture_search_bench.py``) imports
it directly so the two layers test the identical fixture shape.

Optionally we also smoke-test against the real h01 fixtures at
``/home/ubuntu/data/m5_fixtures/{0319, 250924mptq}/``; those tests
``skip()`` if the directories don't exist (so the test suite is
portable on any machine).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.transport.captured_npz import (
    CAPTURED_SCHEMA_VERSION,
    CapturedChgroup,
    CapturedManifest,
    T2Truth,
    load_captured_run,
    stack_dense_streams,
)


# ---------------------------------------------------------------------------
# Fixture builder (shared with bench-level test)
# ---------------------------------------------------------------------------


def _write_synthetic_fixture(
    out_dir: Path,
    *,
    n_chgroups_present: int = 4,
    n_fv_total: int = 3,
    n_grid: int = 8,
    n_filled: int = 6,
    run_id: str = "testrun",
    is_burst: bool = True,
    chgroup_indices: Sequence[int] | None = None,
) -> None:
    """Build a small but schema-faithful captured fixture on disk.

    Mirrors the F26 sparse-COO + provenance schema produced by the
    M3 ``bench/m3_emit_m5_fixtures.py`` writer. Each chgroup carries
    a deterministic vis cube (chgroup-index-tagged so cross-chgroup
    bugs are easy to spot in tests) at a stable sparsity pattern.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if chgroup_indices is None:
        chgroup_indices = list(range(n_chgroups_present))
    else:
        chgroup_indices = list(chgroup_indices)
        assert len(chgroup_indices) == n_chgroups_present

    rng = np.random.default_rng(seed=42)
    # Stable sparsity pattern across all chgroups. Cells are unique
    # (ix_row, ix_col) pairs in [0, n_grid).
    flat_idx = rng.choice(n_grid * n_grid, size=n_filled, replace=False)
    ix_row = (flat_idx // n_grid).astype(np.uint16)
    ix_col = (flat_idx %  n_grid).astype(np.uint16)
    pattern_id = np.uint64(0xCAFEF00DDEADBEEF)
    n_baselines = 4656
    n_ant = 96
    antpos_e = rng.standard_normal(n_ant).astype(np.float32)
    antpos_n = rng.standard_normal(n_ant).astype(np.float32)
    is_core_baseline_mask = rng.random(n_baselines) > 0.5

    src_truth = (
        {
            "src_name": "FRB_test",
            "ra_deg": 307.776667,
            "dec_deg": 53.848986,
            "mjd_trigger": 60942.172498,
            "dm_pc_cc": 404.688,
            "t2_snr": 30.0,
        } if is_burst else {
            "src_name": "",
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "mjd_trigger": float("nan"),
            "dm_pc_cc": float("nan"),
            "t2_snr": float("nan"),
        }
    )

    per_chgroup_meta = []
    for chg_idx in chgroup_indices:
        # vis_cube_sparse: complex64 [N_DM=1, n_fv_total, n_filled].
        # Tag values with chg_idx so load + stack tests can verify
        # slot identity.
        vis = np.zeros((1, n_fv_total, n_filled), dtype=np.complex64)
        for t in range(n_fv_total):
            for k in range(n_filled):
                vis[0, t, k] = (
                    np.complex64(complex(chg_idx + 0.1 * t,
                                         0.01 * k - 0.001 * chg_idx))
                )

        npz_path = out_dir / f"chgroup{chg_idx:02d}.npz"
        np.savez(
            npz_path,
            vis_cube_sparse=vis,
            ix_row=ix_row,
            ix_col=ix_col,
            pattern_id=pattern_id,
            n_grid=np.int32(n_grid),
            n_filled=np.int32(n_filled),
            dec_deg_quant=np.float32(53.85),
            kernel_support=np.int32(7),
            antpos_hash=np.uint64(0xAA00 + chg_idx),
            chgroup_table_hash=np.uint64(0xBB00),
            antpos_e=antpos_e,
            antpos_n=antpos_n,
            is_core_baseline_mask=is_core_baseline_mask,
            chgroup=np.int32(chg_idx),
            t_int_fast_native=np.int32(64),
            t_int_fast_us=np.float64(2097.152),
            n_fv_total=np.int32(n_fv_total),
            n_blocks_processed=np.int32(8),
            cell_lambda=np.float32(31.5),
            phi_lat_ovro_deg=np.float32(37.234),
            obs_dec_deg=np.float32(53.85),
            src_kind=np.array("burst" if is_burst else "continuum"),
            src_name=np.array(src_truth["src_name"]),
            src_ra_deg=np.float64(src_truth["ra_deg"]),
            src_dec_deg=np.float64(src_truth["dec_deg"]),
            src_mjd_trigger=np.float64(src_truth["mjd_trigger"]),
            src_dm_pc_cc=np.float64(src_truth["dm_pc_cc"]),
            src_t2_snr=np.float64(src_truth["t2_snr"]),
            run_id=np.array(run_id),
            cal_path=np.array(f"/fake/cal/{chg_idx:02d}.dat"),
            voltage_path=np.array(f"/fake/voltages/sb{chg_idx:02d}.out"),
            git_sha=np.array("0" * 40),
            utc_iso=np.array("2026-05-06T12:00:00Z"),
        )
        per_chgroup_meta.append({
            "chgroup": chg_idx,
            "voltage_path": f"/fake/voltages/sb{chg_idx:02d}.out",
            "cal_path": f"/fake/cal/{chg_idx:02d}.dat",
            "n_fv_total": n_fv_total,
            "n_filled": n_filled,
            "n_blocks_processed": 8,
            "cell_lambda": 31.5,
            "pattern_id_hex": hex(int(pattern_id)),
            "n_grid": n_grid,
            "out_path": str(npz_path),
            "out_bytes": npz_path.stat().st_size,
            "elapsed_s": 0.5,
        })

    manifest = {
        "milestone": "M3",
        "purpose": "M5 voltage-fixture-search inputs (test fixture)",
        "run_id": run_id,
        "src_kind": "burst" if is_burst else "continuum",
        "src_name": src_truth["src_name"],
        "src_truth": src_truth,
        "obs_dec_deg": 53.85,
        "t_int_fast_native": 64,
        "t_int_fast_us": 2097.152,
        "n_chgroups": n_chgroups_present,
        "chgroups": chgroup_indices,
        "per_chgroup": per_chgroup_meta,
        "git_sha": "0" * 40,
        "utc_iso": "2026-05-06T12:00:00Z",
        "n_baselines": n_baselines,
        "phi_lat_ovro_deg": 37.234,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Loader: synthetic fixtures
# ---------------------------------------------------------------------------


def test_schema_version_is_int() -> None:
    """``CAPTURED_SCHEMA_VERSION`` is an int and can be bumped."""
    assert isinstance(CAPTURED_SCHEMA_VERSION, int)
    assert CAPTURED_SCHEMA_VERSION >= 1


def test_load_synthetic_burst_fixture(tmp_path) -> None:
    """Load a 4-chgroup burst-tagged synthetic fixture; verify all
    schema fields + per-chgroup invariants hold.
    """
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=4, n_fv_total=3, n_grid=8, n_filled=6,
        run_id="burst_test", is_burst=True,
    )
    chgroups, manifest = load_captured_run(tmp_path)

    assert isinstance(manifest, CapturedManifest)
    assert manifest.run_id == "burst_test"
    assert manifest.src_kind == "burst"
    assert manifest.is_burst
    assert manifest.src_truth.dm_pc_cc == 404.688
    assert manifest.src_truth.t2_snr == 30.0
    assert manifest.src_truth.is_burst

    assert len(chgroups) == 4
    assert set(chgroups.keys()) == {0, 1, 2, 3}
    for chg_idx, g in chgroups.items():
        assert isinstance(g, CapturedChgroup)
        assert g.chgroup == chg_idx
        assert g.n_grid == 8
        assert g.n_filled == 6
        assert g.n_fv_total == 3
        assert g.kernel_support == 7
        assert g.vis_cube_sparse.dtype == np.complex64
        assert g.vis_cube_sparse.shape == (1, 3, 6)
        assert g.ix_row.dtype == np.uint16
        assert g.ix_col.dtype == np.uint16


def test_load_continuum_fixture_has_nan_truth(tmp_path) -> None:
    """Continuum fixture has NaN T2 truth; ``is_burst`` is False."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=2, n_fv_total=2, n_grid=4, n_filled=3,
        run_id="cont_test", is_burst=False,
    )
    _, manifest = load_captured_run(tmp_path)
    assert manifest.src_kind == "continuum"
    assert not manifest.is_burst
    assert np.isnan(manifest.src_truth.dm_pc_cc)
    assert not manifest.src_truth.is_burst


def test_load_with_chgroups_subset(tmp_path) -> None:
    """``chgroups_subset`` skips the unselected chgroups."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=4, n_fv_total=2, n_grid=4, n_filled=3,
    )
    chgroups, _ = load_captured_run(tmp_path, chgroups_subset=[0, 2])
    assert set(chgroups.keys()) == {0, 2}


def test_load_with_chgroups_subset_validates_membership(tmp_path) -> None:
    """``chgroups_subset`` containing an absent chgroup raises."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=2, n_fv_total=2, n_grid=4, n_filled=3,
    )
    with pytest.raises(ValueError, match="not present"):
        load_captured_run(tmp_path, chgroups_subset=[0, 99])


def test_load_missing_manifest(tmp_path) -> None:
    """Missing manifest.json raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_captured_run(tmp_path)


def test_load_missing_chgroup_npz(tmp_path) -> None:
    """Manifest references chgroupNN.npz that doesn't exist on disk."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=2, n_fv_total=2, n_grid=4, n_filled=3,
    )
    (tmp_path / "chgroup01.npz").unlink()
    with pytest.raises(FileNotFoundError, match="chgroup01"):
        load_captured_run(tmp_path)


def test_load_truncated_manifest_raises(tmp_path) -> None:
    """Manifest missing a required key produces an actionable error."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=2, n_fv_total=2, n_grid=4, n_filled=3,
    )
    mfst_path = tmp_path / "manifest.json"
    mfst = json.loads(mfst_path.read_text(encoding="utf-8"))
    del mfst["src_truth"]
    mfst_path.write_text(json.dumps(mfst), encoding="utf-8")
    with pytest.raises(ValueError, match="src_truth"):
        load_captured_run(tmp_path)


def test_load_npz_missing_field_raises(tmp_path) -> None:
    """A chgroup NPZ missing a required field produces an actionable error."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=1, n_fv_total=2, n_grid=4, n_filled=3,
    )
    # Rewrite chgroup00.npz without the pattern_id field.
    npz_path = tmp_path / "chgroup00.npz"
    with np.load(npz_path, allow_pickle=False) as nz:
        kept = {k: nz[k] for k in nz.files if k != "pattern_id"}
    np.savez(npz_path, **kept)
    with pytest.raises(ValueError, match="pattern_id"):
        load_captured_run(tmp_path)


def test_chgroup_filename_must_match_chgroup_scalar(tmp_path) -> None:
    """The on-disk ``chgroupNN.npz`` filename must match the scalar
    ``chgroup`` field inside it (production sanity check)."""
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=1, n_fv_total=2, n_grid=4, n_filled=3,
    )
    # Tamper: rewrite chgroup00.npz with chgroup=99 inside.
    src_path = tmp_path / "chgroup00.npz"
    with np.load(src_path, allow_pickle=False) as nz:
        kept = {k: nz[k] for k in nz.files}
    kept["chgroup"] = np.int32(99)
    np.savez(src_path, **kept)
    with pytest.raises(ValueError, match="filename chgroup="):
        load_captured_run(tmp_path)


# ---------------------------------------------------------------------------
# Sparse → dense scatter
# ---------------------------------------------------------------------------


def test_scatter_dense_round_trip(tmp_path) -> None:
    """Sparse-COO → dense scatter places the vis_cube_sparse values
    at exactly the (ix_row, ix_col) cells. All other cells are zero.
    """
    _write_synthetic_fixture(
        tmp_path, n_chgroups_present=1, n_fv_total=2, n_grid=4, n_filled=3,
    )
    chgroups, _ = load_captured_run(tmp_path)
    g = chgroups[0]
    dense = g.scatter_dense()
    assert dense.shape == (1, 2, 4, 4)
    assert dense.dtype == np.complex64
    # Every sparse cell is preserved at the expected (row, col).
    for k in range(g.n_filled):
        r = int(g.ix_row[k]); c = int(g.ix_col[k])
        for t in range(g.n_fv_total):
            assert dense[0, t, r, c] == g.vis_cube_sparse[0, t, k], (
                f"mismatch at chg=0 t={t} k={k} (r={r}, c={c})"
            )
    # Sum of |dense| equals sum of |vis_cube_sparse| (no spurious cells).
    assert np.isclose(
        np.abs(dense).sum(), np.abs(g.vis_cube_sparse).sum(),
    )


def test_stack_dense_streams_shape_and_valid_mask(tmp_path) -> None:
    """``stack_dense_streams`` produces ``[16, n_fv, N, N]`` with
    valid_mask True only at the present-chgroup slots.
    """
    _write_synthetic_fixture(
        tmp_path,
        n_chgroups_present=4,
        n_fv_total=3, n_grid=8, n_filled=6,
        chgroup_indices=[0, 2, 5, 15],
    )
    chgroups, _ = load_captured_run(tmp_path)
    streams, valid_mask = stack_dense_streams(chgroups)
    assert streams.shape == (16, 3, 8, 8)
    assert streams.dtype == np.complex64
    expected_valid = [False] * 16
    for i in (0, 2, 5, 15):
        expected_valid[i] = True
    assert valid_mask == expected_valid
    # Zero-filled slots are exactly zero.
    for i, v in enumerate(valid_mask):
        if not v:
            assert np.all(streams[i] == 0.0), f"slot {i} should be zero"
        else:
            assert np.any(streams[i] != 0.0), f"slot {i} should be populated"


def test_stack_dense_streams_fill_missing_false_raises(tmp_path) -> None:
    """``fill_missing=False`` rejects fixtures with absent slots."""
    _write_synthetic_fixture(
        tmp_path,
        n_chgroups_present=2,
        n_fv_total=2, n_grid=4, n_filled=3,
        chgroup_indices=[0, 5],
    )
    chgroups, _ = load_captured_run(tmp_path)
    with pytest.raises(ValueError, match="fill_missing"):
        stack_dense_streams(chgroups, fill_missing=False)


def test_t2_truth_is_burst_property() -> None:
    burst = T2Truth("FRB", 307.78, 53.85, 60942.17, 404.688, 30.0)
    cont = T2Truth("", float("nan"), float("nan"),
                   float("nan"), float("nan"), float("nan"))
    assert burst.is_burst
    assert not cont.is_burst


# ---------------------------------------------------------------------------
# h01-only smoke tests against the real M3 fixtures
# ---------------------------------------------------------------------------


_H01_FIXTURE_ROOT = Path("/home/ubuntu/data/m5_fixtures")


def _h01_run_dir(run_id: str) -> Path:
    return _H01_FIXTURE_ROOT / run_id


@pytest.mark.skipif(
    not _h01_run_dir("0319").is_dir(),
    reason="h01-only smoke test (M3 fixture absent on this host)",
)
def test_h01_load_0319_continuum() -> None:
    """0319 continuum fixture: 15 chgroups (sb12 missing), NaN T2 truth."""
    chgroups, manifest = load_captured_run(_h01_run_dir("0319"))
    assert manifest.run_id == "0319"
    assert manifest.src_kind == "continuum"
    assert not manifest.is_burst
    # Known sb12 data gap.
    assert 12 not in manifest.chgroups
    assert len(chgroups) == 15
    g = chgroups[0]
    assert g.n_grid == 256
    assert g.n_fv_total == 15
    assert g.vis_cube_sparse.shape[0] == 1  # N_DM=1


@pytest.mark.skipif(
    not _h01_run_dir("250924mptq").is_dir(),
    reason="h01-only smoke test (M3 burst fixture absent on this host)",
)
def test_h01_load_250924mptq_burst() -> None:
    """250924mptq burst fixture: 16 chgroups, full T2 truth."""
    chgroups, manifest = load_captured_run(_h01_run_dir("250924mptq"))
    assert manifest.run_id == "250924mptq"
    assert manifest.src_kind == "burst"
    assert manifest.is_burst
    assert manifest.src_truth.dm_pc_cc == pytest.approx(404.688, rel=1e-6)
    assert manifest.src_truth.t2_snr == pytest.approx(30.0)
    assert len(chgroups) == 16
    assert all(c in chgroups for c in range(16))
    g = chgroups[0]
    assert g.n_grid == 256
    assert g.n_fv_total == 512
    assert g.vis_cube_sparse.shape[0] == 1  # N_DM=1
    # Sample one chgroup's scatter to confirm dense materialisation works.
    dense = g.scatter_dense()
    assert dense.shape == (1, 512, 256, 256)
    assert dense.dtype == np.complex64
