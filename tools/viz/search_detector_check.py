#!/usr/bin/env python3
"""tools/viz/search_detector_check.py — operator-facing search-detector
viz CLI (plan §4.7 + §8 lines 1868-1887).

Two modes:

  * ``--mode cube_injection`` (M5 chunk 5; this commit): consumes the
    per-injection NDJSON log produced by ``bench/cube_injection_detector.py``
    and renders:

      - ``recovery_heatmap.png``                — (snr, width) recovery
        fraction + recovered/injected SNR ratio.
      - ``score_per_kernel_<snr>_<width>.png``  — per-cell 128-cell
        post-Layer-2 detector score map (one PNG per sweep cell).
      - ``noise_only_far.png``                  — empirical-vs-analytic
        Gaussian-tail FAR.
      - ``candidates.html``                     — flat candidates table.
      - ``report.html``                         — master self-contained
        report (NO PASS/FAIL banner per plan §4.7).

  * ``--mode burst`` (M5 chunk 7 — voltage-fixture-driven; deferred until
    the M3 sub-agent emits captured per-chgroup transport-TX .npz). The
    code path raises ``NotImplementedError`` with a pointer to chunk 7.

Per plan §8 line 1887: the operator inspects ``report.html`` in a
browser and signs off in a one-line reply. NO automated PASS/FAIL banner
gates milestone closure; numerical PASS is necessary but not sufficient.

CLI:

  python -m tools.viz.search_detector_check \\
      --mode cube_injection                  \\
      --injection-log <path/injection_log.ndjson>  \\
      [--noise-only-log <path/noise_only_log.ndjson>] \\
      [--summary <path/summary.json>]        \\
      --out <out_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.viz.search_helpers import (  # noqa: E402
    CandidateRow,
    FarSample,
    FigureEntry,
    KernelScoreEntry,
    RecoveryCell,
    gaussian_tail_far,
    n_eff_per_cube_per_kernel,
    render_candidates_table_html,
    render_far_curve_png,
    render_recovery_heatmap_png,
    render_score_per_kernel_png,
    stitch_search_html_report,
)


def _load_ndjson(path: Path) -> List[dict]:
    """Load an NDJSON file (one JSON object per non-empty line)."""
    if not path.is_file():
        return []
    records: List[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _safe_get_int_token(token: str) -> int:
    """Parse e.g. ``"d3"`` → ``3``; defensive against malformed input."""
    if not token or len(token) < 2:
        return 1
    try:
        return int(token[1:])
    except ValueError:
        return 1


def _build_score_per_kernel_entries(
    score_map: dict,
) -> List[KernelScoreEntry]:
    """Convert ``{kernel_id: snr}`` dict to a list of
    ``KernelScoreEntry`` records by parsing each kernel_id."""
    entries: List[KernelScoreEntry] = []
    for kid, snr_value in score_map.items():
        parts = kid.split(":")
        if len(parts) != 3:
            continue
        img_tok, dm_tok, time_tok = parts
        entries.append(
            KernelScoreEntry(
                image_token=img_tok,
                dm_token=dm_tok,
                time_token=time_tok,
                k_dm_width=_safe_get_int_token(dm_tok),
                k_time_width=_safe_get_int_token(time_tok),
                snr=float(snr_value),
            )
        )
    return entries


def _render_cube_injection_report(
    *,
    out_dir: Path,
    injection_records: Sequence[dict],
    noise_only_records: Sequence[dict],
    summary: Optional[dict],
) -> Path:
    """Render the cube_injection-mode report.

    Reads the injection records emitted by
    ``bench/cube_injection_detector.py`` and produces:
      - recovery_heatmap.png + per-cell score heatmaps + noise_only FAR
      - candidates.html
      - report.html (master)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: List[FigureEntry] = []

    # ---- Recovery heatmap ----
    recovery_cells: List[RecoveryCell] = []
    for rec in injection_records:
        if rec.get("kind") != "injection":
            continue
        injected = rec.get("injected", {})
        ratios = (
            [s / float(injected.get("snr", 1.0)) for s in rec.get("recovered_snrs", [])]
            if injected.get("snr")
            else []
        )
        snr_ratio_mean = (
            sum(ratios) / len(ratios) if ratios else float("nan")
        )
        recovery_cells.append(
            RecoveryCell(
                injected_snr=float(injected.get("snr", float("nan"))),
                width_samples=int(injected.get("width_samples", 1)),
                n_injected=int(rec.get("n_trials", 0)),
                n_recovered=int(rec.get("n_recovered", 0)),
                snr_ratio_mean=float(snr_ratio_mean),
            )
        )

    if recovery_cells:
        png_path = out_dir / "recovery_heatmap.png"
        render_recovery_heatmap_png(
            recovery_cells, out_path=png_path,
            title="Cube-injection recovery (snr × width)",
        )
        full_recovery = sum(
            1 for c in recovery_cells if c.recovery_fraction >= 0.95
        )
        figures.append(FigureEntry(
            png_filename="recovery_heatmap.png",
            caption="Recovery heatmap (snr × width)",
            observed=f"{full_recovery}/{len(recovery_cells)} cells "
                     f"recover ≥95%",
            expected="≥95% recovery above ~8σ at any width; smooth "
                     "fall-off below threshold",
        ))

    # ---- Per-cell score-per-kernel heatmaps ----
    cells_with_scores = 0
    for rec in injection_records:
        if rec.get("kind") != "injection":
            continue
        injected = rec.get("injected", {})
        score_map = rec.get("score_per_kernel_at_match") or {}
        if not score_map:
            continue
        entries = _build_score_per_kernel_entries(score_map)
        if not entries:
            continue
        snr = float(injected.get("snr", 0.0))
        width = int(injected.get("width_samples", 0))
        png_name = f"score_per_kernel_snr{snr:g}_w{width}.png"
        render_score_per_kernel_png(
            entries,
            out_path=out_dir / png_name,
            title=f"Per-kernel SNR (injected snr={snr:g}, width={width})",
        )
        cells_with_scores += 1

    if cells_with_scores:
        figures.append(FigureEntry(
            png_filename="score_per_kernel_*.png",
            caption=f"Per-kernel score maps "
                    f"({cells_with_scores} cells)",
            observed=f"{cells_with_scores} per-cell heatmaps written",
            expected="matched kernel triple has highest score per cell",
        ))

    # ---- Noise-only FAR plot ----
    far_samples: List[FarSample] = []
    if summary and "far" in summary and summary.get("config"):
        cfg = summary.get("config", {})
        t_det = int(cfg.get("T_det", 512))
        n_fdm = int(cfg.get("N_fdm", 8))
        n_grid = int(cfg.get("N_grid", 64))
        # Average analytic over a representative kernel; v1 image is delta
        # so K_img = 1, mean K_dm × K_time over the bank ≈ (sum widths) /
        # bank_size.
        avg_k_dm = (1 + 3 + 5 + 7) / 4.0
        avg_k_time = (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) / 8.0
        for sample in summary["far"]:
            theta = float(sample["theta"])
            empirical = float(sample["empirical_per_cube_per_kernel"])
            n_eff = n_eff_per_cube_per_kernel(
                t_det=t_det, n_fdm=n_fdm, n_grid=n_grid,
                k_img_volume=1,
                k_dm_width=int(avg_k_dm),
                k_time_width=int(avg_k_time),
            )
            analytic = gaussian_tail_far(theta) * n_eff
            far_samples.append(FarSample(
                theta=theta,
                empirical_per_cube_per_kernel=empirical,
                analytic_per_cube_per_kernel=analytic,
                n_cubes=int(sample.get("n_cubes", 0)),
                n_kernels=int(sample.get("n_kernels", 128)),
            ))

    if far_samples:
        render_far_curve_png(
            far_samples, out_path=out_dir / "noise_only_far.png",
            title="Noise-only FAR (empirical vs analytic Gaussian tail)",
        )
        # Pick the θ=8 row for the criterion summary.
        target_theta = 8.0
        target = next(
            (s for s in far_samples if abs(s.theta - target_theta) < 1e-6),
            None,
        )
        if target is not None and target.analytic_per_cube_per_kernel > 0:
            ratio = (
                target.empirical_per_cube_per_kernel
                / target.analytic_per_cube_per_kernel
            )
            obs_str = (
                f"empirical={target.empirical_per_cube_per_kernel:.3e}, "
                f"analytic={target.analytic_per_cube_per_kernel:.3e}, "
                f"ratio={ratio:.2f}"
            )
        else:
            obs_str = "n/a (no θ=8 sample)"
        figures.append(FigureEntry(
            png_filename="noise_only_far.png",
            caption="Noise-only FAR (per cube per kernel)",
            observed=obs_str,
            expected="ratio ∈ [0.5, 2.0] at θ=8 per plan §8 line 2329",
        ))

    # ---- Candidates HTML table ----
    rows: List[CandidateRow] = []
    rank = 0
    for rec in injection_records:
        if rec.get("kind") != "injection":
            continue
        injected = rec.get("injected", {})
        snrs = rec.get("recovered_snrs") or []
        kernel_id = rec.get("matched_kernel_id") or ""
        if not snrs:
            continue
        rank += 1
        rows.append(CandidateRow(
            rank=rank,
            observed={
                "kernel_id": kernel_id,
                "snr": max(snrs),
                "l": injected.get("l_pix", 0),
                "m": injected.get("m_pix", 0),
                "dm_idx": injected.get("fine_dm_idx", 0),
                "event_specnum": injected.get("t_in_cube", 0),
                "width_samples": injected.get("width_samples", 0),
            },
            injected={
                "l_pix": injected.get("l_pix", 0),
                "m_pix": injected.get("m_pix", 0),
                "fine_dm_idx": injected.get("fine_dm_idx", 0),
                "t_in_cube": injected.get("t_in_cube", 0),
            },
        ))
    if noise_only_records:
        for rec in noise_only_records:
            for snr in (rec.get("candidate_snrs") or []):
                if float(snr) < 8.0:
                    continue
                rank += 1
                rows.append(CandidateRow(
                    rank=rank,
                    observed={
                        "kernel_id": "?",
                        "snr": float(snr),
                        "l": 0, "m": 0, "dm_idx": 0,
                        "event_specnum": int(rec.get("cube_id", 0)),
                        "width_samples": 0,
                    },
                    injected=None,
                ))

    candidates_html_filename: Optional[str] = None
    if rows:
        candidates_path = out_dir / "candidates.html"
        render_candidates_table_html(
            rows, out_path=candidates_path,
            title="Cube-injection candidates",
        )
        candidates_html_filename = "candidates.html"

    # ---- Master report.html ----
    run_id = (summary or {}).get("config", {}).get("seed")
    header_meta = {
        "Run": f"cube_injection (seed={run_id})" if run_id else "cube_injection",
        "Mode": "cube_injection",
        "Generated UTC ns": str(time.time_ns()),
        "Generated UTC": datetime.now(tz=timezone.utc).isoformat(),
        "Tool": "search_detector_check",
        "n_cells": len(recovery_cells),
        "n_noise_cubes": len(noise_only_records),
    }
    report_path = stitch_search_html_report(
        out_dir=out_dir,
        title="Search detector check (cube_injection)",
        header_meta=header_meta,
        figures=figures,
        candidates_html_filename=candidates_html_filename,
    )

    # Self-contained report.txt summary for grep-able CI aggregation
    # (per plan §8 line 1889; observed metrics, NOT a PASS/FAIL).
    summary_lines = [
        "tool=search_detector_check mode=cube_injection",
        f"n_cells={len(recovery_cells)} n_noise_cubes={len(noise_only_records)}",
    ]
    for c in recovery_cells:
        summary_lines.append(
            f"cell snr={c.injected_snr:g} width={c.width_samples} "
            f"recovered={c.n_recovered}/{c.n_injected} "
            f"snr_ratio={c.snr_ratio_mean:.3f}"
        )
    for s in far_samples:
        summary_lines.append(
            f"far theta={s.theta:g} "
            f"empirical={s.empirical_per_cube_per_kernel:.3e} "
            f"analytic={s.analytic_per_cube_per_kernel:.3e}"
        )
    (out_dir / "report.txt").write_text("\n".join(summary_lines) + "\n")

    return report_path


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode", choices=("cube_injection", "burst"), required=True,
    )
    ap.add_argument(
        "--injection-log",
        help="Path to injection_log.ndjson "
             "(required for --mode cube_injection)",
    )
    ap.add_argument(
        "--noise-only-log",
        help="Path to noise_only_log.ndjson (optional; FAR plot is "
             "skipped if absent)",
    )
    ap.add_argument(
        "--summary",
        help="Path to summary.json emitted by the bench (optional; "
             "FAR analytic and run header read from here when "
             "supplied)",
    )
    ap.add_argument(
        "--voltage-run-id",
        help="Voltage fixture run-id (for --mode burst, M5 chunk 7).",
    )
    ap.add_argument(
        "--include-dumps", action="store_true",
        help="Include voltage-dump round-trip + .fil viz (M6 only).",
    )
    ap.add_argument("--out", required=True, help="Output report directory.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()

    if args.mode == "burst":
        # Plan §8 line 2330 — needs M3-emitted captured per-chgroup
        # transport-TX .npz set; deferred to M5 chunk 7.
        raise NotImplementedError(
            "--mode burst is M5 chunk 7 (voltage-fixture-driven). "
            "Requires M3 sub-agent's captured .npz set; the code path "
            "lands when chunk 7 implements bench/voltage_fixture_search.py."
        )

    if not args.injection_log:
        ap.error("--mode cube_injection requires --injection-log")
    injection_path = Path(args.injection_log)
    if not injection_path.is_file():
        ap.error(f"--injection-log {injection_path} not found")
    noise_only_path = Path(args.noise_only_log) if args.noise_only_log else None
    summary_path = Path(args.summary) if args.summary else None

    injection_records = _load_ndjson(injection_path)
    noise_only_records = (
        _load_ndjson(noise_only_path) if noise_only_path else []
    )
    summary: Optional[dict] = None
    if summary_path is not None and summary_path.is_file():
        summary = json.loads(summary_path.read_text())

    report_path = _render_cube_injection_report(
        out_dir=out_dir,
        injection_records=injection_records,
        noise_only_records=noise_only_records,
        summary=summary,
    )
    print(f"[search_detector_check] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
