"""Tests for the M6 chunk-8 operator visualisation tools.

Covers the three deliverables from the chunk-8 spec:

  1. ``tools.viz.m6_t1_t2_inspector``: synthesise a small T1+T2 log
     pair, run the inspector via ``main()``, assert the PNG + MD land
     and the MD references key counts (T1/T2 row totals).

  2. ``tools.viz.m6_cube_dump_verifier``: synthesise a tiny NPZ that
     matches the writer's schema (per ``dsart.dump.cube_dump.
     CubeDumpWriter._write_one``), run the verifier, assert the
     per-cube PNG + summary MD land and the MD mentions the cube_id.

  3. Operator-approval marker template: assert that the canonical
     marker file exists, that it carries TEMPLATE markers, and that
     ``tools/dod/M6.sh``'s operator_approval logic treats a TEMPLATE
     file as not-yet-approved (extracted by reading the script's
     source — running M6.sh requires the conda env / etcd / status
     JSON envelope which is not in scope for this unit test).

The viz tools do not import ``dsart``; they only depend on numpy +
matplotlib so the tests have no torch / GPU prerequisites.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# Match the convention used by tests/test_search_detector_check_viz.py:
# the viz tools live at <REPO_ROOT>/tools/viz, importable as
# ``tools.viz.m6_*`` once REPO_ROOT is on sys.path. PYTHONPATH wiring
# done by the M6.sh harness already covers this in the integration
# path; do it here too for direct-pytest invocations.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")


# ---------------------------------------------------------------------------
# Helpers — synthesise inspector inputs / verifier inputs
# ---------------------------------------------------------------------------


def _t1_header_line() -> str:
    from tools.viz.m6_t1_t2_inspector import T1_COLUMNS

    return "# " + " ".join(T1_COLUMNS) + "\n"


def _t2_header_line() -> str:
    from tools.viz.m6_t1_t2_inspector import T2_COLUMNS

    return "# " + " ".join(T2_COLUMNS) + "\n"


def _t1_row(
    *,
    mjd: float,
    event_specnum: int,
    l_rad: float,
    m_rad: float,
    l_pix: int,
    m_pix: int,
    dm_fine_pc_cc: float,
    fine_dm_idx: int,
    t_in_cube: int,
    width_samples: int,
    snr: float,
    kernel_id: str = "psf:d3:b16",
    cl: int = 0,
    is_cluster_peak: int = 0,
    sid: int = 0,
    g: int = 0,
) -> str:
    return (
        f"{mjd:.11f} {event_specnum:d} {l_rad:.9e} {m_rad:.9e} "
        f"{l_pix:d} {m_pix:d} {dm_fine_pc_cc:.9e} {fine_dm_idx:d} "
        f"{t_in_cube:d} {width_samples:d} {snr:.6e} {kernel_id} "
        f"{cl:d} {is_cluster_peak:d} {sid:d} {g:d}\n"
    )


def _t2_row(
    *,
    mjd: float,
    event_specnum: int,
    l_rad: float,
    m_rad: float,
    l_pix: int,
    m_pix: int,
    dm_fine_pc_cc: float,
    fine_dm_idx: int,
    t_in_cube: int,
    width_samples: int,
    snr: float,
    kernel_id: str = "psf:d3:b16",
    cluster_id: int = 0,
    cntc: int = 1,
    cntb_lm: int = 1,
    cntb_dm: int = 1,
    cube_dump_triggered: int = 0,
    sid: int = 0,
    g: int = 0,
) -> str:
    return (
        f"{mjd:.11f} {event_specnum:d} {l_rad:.9e} {m_rad:.9e} "
        f"{l_pix:d} {m_pix:d} {dm_fine_pc_cc:.9e} {fine_dm_idx:d} "
        f"{t_in_cube:d} {width_samples:d} {snr:.6e} {kernel_id} "
        f"{cluster_id:d} {cntc:d} {cntb_lm:d} {cntb_dm:d} "
        f"{cube_dump_triggered:d} {sid:d} {g:d}\n"
    )


def _make_t1_log(path: Path, rows: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_t1_header_line() + "".join(rows))


def _make_t2_log(path: Path, rows: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_t2_header_line() + "".join(rows))


def _make_cube_dump_npz(
    path: Path,
    *,
    t_det: int = 4,
    n_fdm: int = 3,
    n_grid: int = 8,
    sid: int = 0,
    gpu_half: int = 0,
    event_specnum_start: int = 1024,
    mjd_start: float = 60942.123456789,
    cluster_record: dict | None = None,
    trigger_source: str = "auto",
    peak_value: float = 12.5,
    peak_t: int = 1,
    peak_f: int = 2,
    peak_l: int = 3,
    peak_m: int = 5,
) -> None:
    """Build a NPZ matching the ``CubeDumpWriter._write_one`` schema.

    The cube tensor is a flat zero array with a single bright pixel at
    ``[peak_t, peak_f, peak_l, peak_m]`` so the verifier's argmax has a
    deterministic global peak to land on.
    """
    cube = np.zeros((t_det, n_fdm, n_grid, n_grid), dtype=np.float16)
    cube[peak_t, peak_f, peak_l, peak_m] = np.float16(peak_value)
    if cluster_record is None:
        cluster_record = {
            "cluster_id": 0,
            "cube_id": event_specnum_start,
            "cntc": 3,
            "cntb_lm": 2,
            "cntb_dm": 2,
            "peak_candidate_idx": 1,
            "l_rad": 1.5e-4 * peak_l,
            "m_rad": 1.5e-4 * peak_m,
            "l_pix": peak_l,
            "m_pix": peak_m,
            "dm_fine_pc_cc": 397.42,
            "fine_dm_idx": peak_f,
            "t_in_cube": peak_t,
            "t_seconds": peak_t * 131.072e-6,
            "width_samples": 4,
            "snr": 20.81,
            "kernel_id": "psf:d3:b16",
            "event_specnum": event_specnum_start + peak_t * 16,
            "search_node_id": sid,
            "gpu_half": gpu_half,
        }
    record_str = (
        json.dumps(cluster_record)
        if trigger_source == "auto"
        else json.dumps(None)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        cube=cube,
        mjd_start=np.asarray(mjd_start, dtype="float64"),
        event_specnum_start=np.asarray(event_specnum_start, dtype="int64"),
        t_det=np.asarray(t_det, dtype="int32"),
        n_fdm_in_cube=np.asarray(n_fdm, dtype="int32"),
        n_grid=np.asarray(n_grid, dtype="int32"),
        cluster_record=np.asarray(record_str, dtype="U"),
        trigger_source=np.asarray(trigger_source, dtype="U"),
        search_node_id=np.asarray(sid, dtype="int32"),
        gpu_half=np.asarray(gpu_half, dtype="int32"),
    )


# ---------------------------------------------------------------------------
# Test 1: T1/T2 inspector
# ---------------------------------------------------------------------------


def test_t1_t2_inspector_emits_png_and_md(tmp_path: Path) -> None:
    from tools.viz.m6_t1_t2_inspector import main as inspector_main

    log_root = tmp_path / "cands_log"
    log_root.mkdir()
    report_dir = tmp_path / "viz"

    sid = 0
    g = 0
    t1_path = log_root / f"cands_T1_s{sid}_g{g}_20260507_18.txt"
    t2_path = log_root / f"cands_T2_s{sid}_g{g}_20260507_18.txt"

    base_mjd = 60942.0
    cell = 1.5e-4
    t1_rows = [
        _t1_row(
            mjd=base_mjd + i * 1e-7,
            event_specnum=2048 + 16 * i,
            l_rad=cell * (10 + i), m_rad=cell * (20 + i),
            l_pix=10 + i, m_pix=20 + i,
            dm_fine_pc_cc=200.0 + 5 * i, fine_dm_idx=3 + i,
            t_in_cube=10 + i, width_samples=2 + (i % 4),
            snr=8.0 + 0.5 * i,
            cl=(i % 2), is_cluster_peak=(1 if i in (1, 3) else 0),
            sid=sid, g=g,
        )
        for i in range(6)
    ]
    t2_rows = [
        _t2_row(
            mjd=base_mjd + i * 1e-6,
            event_specnum=2048 + 32 * i,
            l_rad=cell * (50 + 7 * i), m_rad=cell * (50 + 5 * i),
            l_pix=50 + 7 * i, m_pix=50 + 5 * i,
            dm_fine_pc_cc=300.0 + 25 * i, fine_dm_idx=10 + i,
            t_in_cube=20 + 2 * i, width_samples=4 + i,
            snr=12.0 + 1.5 * i,
            cluster_id=i, cntc=2 + i, cntb_lm=1 + i, cntb_dm=1 + i,
            cube_dump_triggered=(1 if i == 0 else 0),
            sid=sid, g=g,
        )
        for i in range(3)
    ]
    _make_t1_log(t1_path, t1_rows)
    _make_t2_log(t2_path, t2_rows)

    rc = inspector_main([
        "--log-root", str(log_root),
        "--report-dir", str(report_dir),
        "--since", str(base_mjd - 1.0),
        "--until", str(base_mjd + 1.0),
        "--search-node-id", str(sid),
        "--gpu-half", str(g),
    ])
    assert rc == 0

    png = report_dir / "m6_t1_t2_inspector.png"
    md = report_dir / "m6_t1_t2_inspector.md"
    assert png.exists() and png.stat().st_size > 0, png
    assert md.exists() and md.stat().st_size > 0, md

    md_text = md.read_text()
    # Counts table mentions the row totals from our synthetic logs.
    assert f"| T1 (per-candidate) | {len(t1_rows)} |" in md_text, md_text
    assert f"| T2 (per-cluster)   | {len(t2_rows)} |" in md_text, md_text
    # The image filename and the canonical M6 chunk-8 title are in the body.
    assert "m6_t1_t2_inspector.png" in md_text
    assert "M6 T1/T2 candidate-log inspector" in md_text


def test_t1_t2_inspector_empty_logs_dont_crash(tmp_path: Path) -> None:
    """Empty / missing log roots must produce a clean exit + empty report."""
    from tools.viz.m6_t1_t2_inspector import main as inspector_main

    empty_root = tmp_path / "no_logs"
    empty_root.mkdir()
    report_dir = tmp_path / "viz"

    rc = inspector_main([
        "--log-root", str(empty_root),
        "--report-dir", str(report_dir),
    ])
    assert rc == 0
    md = report_dir / "m6_t1_t2_inspector.md"
    assert md.exists()
    md_text = md.read_text()
    assert "T1 files matched: 0" in md_text
    assert "T2 files matched: 0" in md_text


def test_t1_t2_inspector_subprocess_invocation(tmp_path: Path) -> None:
    """Verify the ``python -m tools.viz.m6_t1_t2_inspector`` entry path."""
    log_root = tmp_path / "logs"
    log_root.mkdir()
    report_dir = tmp_path / "viz"

    sid = 1
    g = 0
    t1_path = log_root / f"cands_T1_s{sid}_g{g}_20260507_18.txt"
    cell = 1.5e-4
    rows = [_t1_row(
        mjd=60942.0, event_specnum=2048,
        l_rad=cell * 10, m_rad=cell * 20,
        l_pix=10, m_pix=20,
        dm_fine_pc_cc=200.0, fine_dm_idx=3,
        t_in_cube=10, width_samples=4, snr=9.0,
        cl=0, is_cluster_peak=1, sid=sid, g=g,
    )]
    _make_t1_log(t1_path, rows)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'src'}"
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    cmd = [
        sys.executable, "-m", "tools.viz.m6_t1_t2_inspector",
        "--log-root", str(log_root),
        "--report-dir", str(report_dir),
        "--search-node-id", str(sid),
        "--gpu-half", str(g),
    ]
    res = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert (report_dir / "m6_t1_t2_inspector.png").exists()
    assert (report_dir / "m6_t1_t2_inspector.md").exists()


# ---------------------------------------------------------------------------
# Test 2: cube-dump verifier
# ---------------------------------------------------------------------------


def test_cube_dump_verifier_emits_per_cube_png_and_md(tmp_path: Path) -> None:
    from tools.viz.m6_cube_dump_verifier import main as verifier_main

    dump_root = tmp_path / "cube_dump"
    dump_root.mkdir()
    report_dir = tmp_path / "viz"

    # Two NPZ dumps: one auto (carries cluster_record), one udp (None).
    auto_npz = dump_root / "cube_s0_g0_1024.npz"
    udp_npz = dump_root / "cube_s0_g0_2048.npz"
    _make_cube_dump_npz(
        auto_npz, sid=0, gpu_half=0, event_specnum_start=1024,
        trigger_source="auto", peak_value=12.5, peak_t=1, peak_f=2,
    )
    _make_cube_dump_npz(
        udp_npz, sid=0, gpu_half=0, event_specnum_start=2048,
        trigger_source="udp", peak_value=7.0, peak_t=0, peak_f=0,
        cluster_record={},  # ignored when trigger_source='udp'
    )

    rc = verifier_main([
        "--dump-root", str(dump_root),
        "--report-dir", str(report_dir),
        "--max-cubes", "8",
    ])
    assert rc == 0

    md = report_dir / "m6_cube_dump_verifier.md"
    assert md.exists() and md.stat().st_size > 0
    md_text = md.read_text()
    # The auto dump's cube_id surrogates the cluster_record's cube_id (we
    # set cube_id=event_specnum_start=1024 in the synthesizer); the udp
    # dump falls back to event_specnum_start (2048) since no record.
    assert "cube_id = 1024" in md_text, md_text
    assert "cube_id = 2048" in md_text, md_text
    # Per-cube PNGs landed on disk.
    assert (report_dir / "m6_cube_dump_1024.png").exists()
    assert (report_dir / "m6_cube_dump_2048.png").exists()
    # Trigger-source column reflects both flavours.
    assert "auto" in md_text
    assert "udp" in md_text
    # Manifest summary mentions both NPZ basenames.
    assert "cube_s0_g0_1024.npz" in md_text
    assert "cube_s0_g0_2048.npz" in md_text


def test_cube_dump_verifier_max_cubes_caps_inspection(tmp_path: Path) -> None:
    """``--max-cubes`` must clip the inspected NPZ count.

    The summary MD must still mention every NPZ file that was *matched*
    (n_total) but inspect only the first --max-cubes by sorted name.
    """
    from tools.viz.m6_cube_dump_verifier import main as verifier_main

    dump_root = tmp_path / "cube_dump"
    dump_root.mkdir()
    report_dir = tmp_path / "viz"

    cube_ids = [1000, 2000, 3000]
    for spec_start in cube_ids:
        _make_cube_dump_npz(
            dump_root / f"cube_s0_g0_{spec_start}.npz",
            event_specnum_start=spec_start,
            trigger_source="auto",
        )

    rc = verifier_main([
        "--dump-root", str(dump_root),
        "--report-dir", str(report_dir),
        "--max-cubes", "2",
    ])
    assert rc == 0

    md_text = (report_dir / "m6_cube_dump_verifier.md").read_text()
    assert "npz files matched: 3" in md_text
    assert "npz files inspected: 2" in md_text
    # First two by sorted name should have PNGs; the third should not.
    assert (report_dir / "m6_cube_dump_1000.png").exists()
    assert (report_dir / "m6_cube_dump_2000.png").exists()
    assert not (report_dir / "m6_cube_dump_3000.png").exists()


def test_cube_dump_verifier_empty_root_doesnt_crash(tmp_path: Path) -> None:
    from tools.viz.m6_cube_dump_verifier import main as verifier_main

    empty = tmp_path / "no_dumps"
    empty.mkdir()
    report_dir = tmp_path / "viz"
    rc = verifier_main([
        "--dump-root", str(empty),
        "--report-dir", str(report_dir),
        "--max-cubes", "8",
    ])
    assert rc == 0
    md = report_dir / "m6_cube_dump_verifier.md"
    assert md.exists()
    text = md.read_text()
    assert "npz files matched: 0" in text


# ---------------------------------------------------------------------------
# Test 3: operator approval marker template
# ---------------------------------------------------------------------------


APPROVAL_PATH = REPO_ROOT / "bench/reports/M6/m_operator_approved.yaml"
M6_DOD_SH = REPO_ROOT / "tools/dod/M6.sh"


def test_operator_approval_marker_template_present() -> None:
    """The chunk-8 deliverable: marker exists at the canonical path."""
    assert APPROVAL_PATH.exists(), (
        f"missing operator-approval marker at {APPROVAL_PATH}; chunk-8 "
        "deliverable is required for the M6 DoD harness."
    )


def test_operator_approval_marker_carries_template_values() -> None:
    """Marker must remain a TEMPLATE — i.e. NOT a real approval.

    Per the M6 chunk-8 spec the file is a placeholder; the operator
    edits it out-of-band to mark M6 approved.
    """
    text = APPROVAL_PATH.read_text()
    assert "TEMPLATE" in text, (
        "operator-approval marker should still carry TEMPLATE placeholders; "
        "real approvals are a separate operator commit."
    )
    # The three documented template fields all have TEMPLATE values.
    assert re.search(r"^approved_by:\s*TEMPLATE\s*$", text, re.MULTILINE), text
    assert re.search(
        r"^approved_at_utc:\s*TEMPLATE\s*$", text, re.MULTILINE
    ), text
    assert re.search(
        r"^approved_git_sha:\s*TEMPLATE\s*$", text, re.MULTILINE
    ), text


def test_operator_approval_step_treats_template_as_warn() -> None:
    """``tools/dod/M6.sh`` must WARN when the marker carries TEMPLATE.

    We grep the script source for the operator_approval logic rather
    than running M6.sh — running M6.sh requires the conda env, the
    M0/M1/M5 status JSONs, and the M6_preflight gate which are not
    available in unit-test scope.
    """
    src = M6_DOD_SH.read_text()
    # The TEMPLATE check must reside inside the operator_approval STEP
    # block. Match a sentinel substring covering the new behavior.
    assert "STEP=\"operator_approval\"" in src
    assert "grep -q 'TEMPLATE'" in src, (
        "M6.sh operator_approval step is expected to grep for TEMPLATE "
        "and treat such markers as not-yet-approved."
    )
    # The WARN message must reference the TEMPLATE state.
    assert re.search(
        r"warn .*TEMPLATE.* not yet approved", src
    ), "M6.sh should emit a TEMPLATE-related WARN message for template markers"
