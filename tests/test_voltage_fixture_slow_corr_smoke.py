"""tests/test_voltage_fixture_slow_corr_smoke.py — M2 Chunk 6 Phase D smoke harness.

Exercises the full Chunk-6 path on a deterministic synthetic continuum
fixture (no operator-supplied data needed). Pipeline:

    bench/voltage_fixture_slow_corr.py --synthesize ...
        ↳ dada_db -k fada / -k bada (lifecycle managed by orchestrator)
        ↳ corr_slow_compute (consumes fada, produces bada)
        ↳ replay_voltage_dump (synth point source @ (l, m))
        ↳ in-process bada capture → bada_capture.bin

    tools/viz/corr_imager_dedisperser_check.py --bada bada_capture.bin
        ↳ load + grid + iFFT
        ↳ slow_corr_check.png + report.html + observed_peaks.json

Asserts:
  * Pipeline overall_rc == 0.
  * Bada capture file is the right size for the requested n_blocks.
  * The viz tool's rank-1 image-plane peak lands within ~1 grid cell of
    the input synthetic source's (l, m).

Per F9 in M2_PLAN_FIXES, this is the M2 self-test for the operator-sign-off
path (operator approval on real fixtures is recorded out of band per D11).

Marks:
  * `gpu` — requires CUDA + psrdada + dada_db CLI on PATH.
  * `slow` — takes ~3-5 min end-to-end (most of that is corr cold start
    + paced replay).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BADA_BYTES_PER_INTEGRATION,
    NBASE,
    NCHAN_PER_CHGROUP,
    BADA_NPOL,
)


def _has_dada_db() -> bool:
    return shutil.which("dada_db") is not None


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_psrdada_python() -> bool:
    try:
        import psrdada  # noqa: F401
        return True
    except ImportError:
        return False


smoke_required = pytest.mark.skipif(
    not (_has_dada_db() and _has_cuda() and _has_psrdada_python()),
    reason="needs dada_db on PATH + CUDA + psrdada-python "
           "(h01-only end-to-end smoke)",
)


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("voltage_fixture_smoke")
    return d


@smoke_required
@pytest.mark.slow
def test_synth_continuum_pipeline_end_to_end(work_dir: Path) -> None:
    """Smoke: synth point-source → corr_slow → bada → viz dirty image.

    The synthetic source sits at (l, m) = (+0.05, +0.04) rad. The 2D
    synthetic 12×8 antenna grid has 5.5 m × 3.5 m physical aperture →
    fringe spacing ~0.04 rad at 1.5 GHz → the source's (l, m) lands
    well within a single dirty-beam main lobe.
    """
    n_blocks = 5
    src_l = 0.05
    src_m = 0.04
    src_amp = 5.0

    # Step 1: orchestrator runs synth → corr → bada capture.
    cmd = [
        sys.executable, "-m", "bench.voltage_fixture_slow_corr",
        "--synthesize",
        "--synth-thermal-sigma", "0.5",
        f"--synth-source", f"{src_l},{src_m},{src_amp}",
        "--n-blocks", str(n_blocks),
        "--rate", "fast",
        "--skip-meridian",
        "--work-dir", str(work_dir),
        "--timeout-s", "180",
        "--device", "cuda",
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=env, cwd=str(REPO_ROOT), timeout=600)
    print("orchestrator stdout:", proc.stdout[-2000:])
    print("orchestrator stderr:", proc.stderr[-2000:])
    assert proc.returncode == 0, (
        f"orchestrator failed rc={proc.returncode}; "
        f"see {work_dir}/summary.json"
    )

    summary_path = work_dir / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["overall_rc"] == 0, f"summary.json: {summary}"

    bada_path = Path(summary["bada_capture"])
    assert bada_path.is_file(), f"missing {bada_path}"
    expected_bytes = n_blocks * BADA_BYTES_PER_INTEGRATION
    assert bada_path.stat().st_size == expected_bytes, (
        f"bada capture size {bada_path.stat().st_size} != "
        f"expected {expected_bytes} ({n_blocks} blocks × "
        f"{BADA_BYTES_PER_INTEGRATION} B)"
    )

    # Step 2: viz tool produces dirty image + report.
    viz_dir = work_dir / "viz_out"
    viz_cmd = [
        sys.executable, "-m", "tools.viz.corr_imager_dedisperser_check",
        "--mode", "continuum",
        "--check", "slow_corr",
        "--bada", str(bada_path),
        "--out", str(viz_dir),
        "--n-grid", "256",
        "--fov-rad", "0.5",
    ]
    viz_proc = subprocess.run(viz_cmd, capture_output=True, text=True,
                              env=env, cwd=str(REPO_ROOT), timeout=120)
    print("viz stdout:", viz_proc.stdout[-2000:])
    print("viz stderr:", viz_proc.stderr[-2000:])
    assert viz_proc.returncode == 0, f"viz tool failed rc={viz_proc.returncode}"

    assert (viz_dir / "slow_corr_check.png").is_file()
    assert (viz_dir / "report.html").is_file()
    obs_json_path = viz_dir / "observed_peaks.json"
    assert obs_json_path.is_file()

    obs = json.loads(obs_json_path.read_text())
    assert obs["peaks"], "no peaks reported by viz tool"
    rank1 = obs["peaks"][0]

    # Tolerance: 1 grid cell = fov_rad / n_grid = 0.5/256 ≈ 0.002 rad.
    # Allow 4 cells (~0.008 rad) for the imager's discretization.
    fov, n_grid = 0.5, 256
    cell = fov / n_grid
    tol = 4 * cell

    print(f"observed rank-1 peak: l={rank1['l_rad']:+.5f} m={rank1['m_rad']:+.5f} "
          f"flux={rank1['flux']:.3g} SNR={rank1['snr']:.2f}")
    print(f"expected source: l={src_l:+.5f} m={src_m:+.5f} (tol={tol:.5f})")

    assert abs(rank1["l_rad"] - src_l) < tol, (
        f"rank-1 peak l={rank1['l_rad']:+.5f} != expected {src_l:+.5f} "
        f"(diff={rank1['l_rad'] - src_l:+.5f}, tol={tol:.5f})"
    )
    assert abs(rank1["m_rad"] - src_m) < tol, (
        f"rank-1 peak m={rank1['m_rad']:+.5f} != expected {src_m:+.5f} "
        f"(diff={rank1['m_rad'] - src_m:+.5f}, tol={tol:.5f})"
    )
    # Sanity: SNR is at least a few sigma above noise.
    assert rank1["snr"] >= 5.0, (
        f"rank-1 peak SNR={rank1['snr']:.1f} < 5.0 — image-plane source "
        f"too weak relative to noise"
    )
