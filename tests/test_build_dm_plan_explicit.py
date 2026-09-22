"""``tools/build_dm_plan_explicit.py`` -- explicit-grid DM plan builder.

Added 2026-09-22 with the DM-scheme rework (see
``_inspect/sensitivity/DM_SCHEME_FINAL.md``).  The production builder
``build_dm_plan.py`` derives its grid from a Levin ``tol``, which cannot
express the two things the rework needs: graded bucket widths and
dithered coarse-DM offsets.  This builder takes the grid as input and
back-solves the ``tol`` the grid corresponds to (the contract requires
``tol > 0``, and it is provenance only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "build_dm_plan_explicit.py"

# the deployed grid, so a silent change to the builder is caught here
OFFSETS = "-4.50,6.75,1.50,-0.75,0.00,4.50,-2.25,9.00"
COARSE = [198.28, 320.36, 436.50, 566.19, 709.44, 867.00, 1023.86, 1209.28]
STEPS = [3.105, 3.415, 3.725, 4.036, 4.346, 4.657, 4.967, 5.278]


@pytest.fixture(scope="module")
def plan_npz(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dmplan") / "plan.npz"
    env = dict(os.environ)
    env["DSART_CONFIG_DIR"] = str(REPO / "configs")
    env["PYTHONPATH"] = str(REPO / "src")
    r = subprocess.run(
        [sys.executable, str(TOOL), "--out", str(out),
         "--dm-min", "150", "--dm-max", "1290", "--beta", "0.10",
         f"--coarse-offsets={OFFSETS}"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    return out


def test_grid_matches_the_deployed_plan(plan_npz: Path) -> None:
    z = np.load(plan_npz, allow_pickle=True)
    assert np.allclose(z["coarse_dm"], COARSE, atol=5e-3)
    fine = np.asarray(z["fine_dm"], dtype=float)
    assert fine.size == 8 * 34
    for b, want in enumerate(STEPS):
        step = np.diff(fine[b * 34:(b + 1) * 34])
        assert np.allclose(step, want, atol=5e-3), (b, step[:3], want)
    # the graded ramp: each bucket wider than the last, mildly
    widths = np.array([fine[(b + 1) * 34 - 1] - fine[b * 34] for b in range(8)])
    assert np.all(np.diff(widths) > 0)
    assert widths[-1] / widths[0] == pytest.approx(1.70, abs=0.02)


def test_coverage_and_tol(plan_npz: Path) -> None:
    z = np.load(plan_npz, allow_pickle=True)
    fine = np.asarray(z["fine_dm"], dtype=float)
    assert fine.min() == pytest.approx(150.0, abs=2.0)
    assert fine.max() == pytest.approx(1290.0, abs=6.0)
    # tol is back-solved from the WORST fine step; contract needs > 0
    tol = float(z["tol"])
    assert 1.0 < tol < 4.0
    assert tol == pytest.approx(1.2376, abs=2e-3)


def test_loads_as_a_valid_dmplan(plan_npz: Path) -> None:
    """The real schema gate -- ``DmPlan.__post_init__`` checks the v2 contract."""
    from dsart.common.contracts import DmPlan

    os.environ.setdefault("DSART_TEST", "1")
    plan = DmPlan.from_npz(str(plan_npz))
    assert plan.coarse_dm.shape == (8,)
    assert plan.fine_dm.shape == (272,)
    # BOT convention: the last chgroup's search shift is the zero point
    assert np.all(np.asarray(plan.time_shift_search)[:, 15] == 0)
    assert np.all(np.asarray(plan.time_shift_corr_stage1) >= 0)
    assert np.all(np.asarray(plan.time_shift_corr_stage2) >= 0)
    assert int(plan.dm_overlap_coarse) == 0
    # one coarse DM per GPU half, canonical range is the singleton (i, i)
    rng = np.asarray(plan.dm_idx_range_canonical_per_gpu)
    assert rng.shape == (4, 2, 2)
    assert np.all(rng[..., 0] == rng[..., 1])
    assert sorted(rng[..., 0].ravel().tolist()) == list(range(8))


def test_fine_to_coarse_is_the_structural_block_map(plan_npz: Path) -> None:
    """Not a nearest-coarse search.

    The coarse DMs are dithered off their bucket centres, so a
    nearest-coarse assignment would let trials at a bucket boundary
    migrate into the neighbouring bucket and be dedispersed against a
    coarse DM the stage-2 delays were not built for.
    """
    z = np.load(plan_npz, allow_pickle=True)
    f2c = np.asarray(z["fine_to_coarse"], dtype=int)
    assert np.array_equal(f2c, np.arange(8 * 34) // 34)


def test_metadata_flags_the_stale_stored_shift_tables(plan_npz: Path) -> None:
    """Item 4 of the audit, made machine-readable.

    The stored ``time_shift_*`` tables are BOT-referenced per the v2
    contract; the runtime recomputes them TOP-referenced.  The stored
    values are therefore NOT what runs.  Deleting them would break the
    schema, so instead the divergence is recorded explicitly.
    """
    z = np.load(plan_npz, allow_pickle=True)
    md = json.loads(str(z["metadata"]))
    assert md["stored_shift_tables_reference"] == "nu_chgroup_bot (v2 contract)"
    assert "RECOMPUTED at runtime" in md["runtime_shift_tables"]
    assert "NOT what runs" in md["runtime_shift_tables"]


def test_build_is_reproducible(plan_npz: Path, tmp_path: Path) -> None:
    """Only the build timestamp may differ between two identical builds."""
    out = tmp_path / "again.npz"
    env = dict(os.environ)
    env["DSART_CONFIG_DIR"] = str(REPO / "configs")
    env["PYTHONPATH"] = str(REPO / "src")
    r = subprocess.run(
        [sys.executable, str(TOOL), "--out", str(out),
         "--dm-min", "150", "--dm-max", "1290", "--beta", "0.10",
         f"--coarse-offsets={OFFSETS}"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    a, b = np.load(plan_npz, allow_pickle=True), np.load(out, allow_pickle=True)
    assert set(a.files) == set(b.files)
    for k in sorted(a.files):
        if k == "metadata":
            ja, jb = json.loads(str(a[k])), json.loads(str(b[k]))
            ja.pop("build_utc_ns"), jb.pop("build_utc_ns")
            assert ja == jb
        elif a[k].dtype.kind in "OU":
            assert str(a[k]) == str(b[k]), k
        else:
            assert np.array_equal(a[k], b[k]), k
