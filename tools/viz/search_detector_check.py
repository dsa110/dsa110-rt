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

  * ``--mode burst`` (M5 chunk 7 — voltage-fixture-driven): consumes
    the ``detector.json`` produced by ``bench/captured_burst_detector.py``
    (or ``bench/voltage_fixture_search.py --mode captured --detector-sweep``;
    same schema in either case) and renders:

      - ``butterfly.png``       — SNR(K_time, fine_dm) heatmap at the
        recovered burst (l, m). The matched-filter "butterfly" pattern
        — the SNR ridge that traces the burst's K_time × DM signature
        and pinches at K_time matched to the burst's intrinsic width.
      - ``dm_curve.png``        — SNR vs fine_dm overlaid for each
        K_time kernel; locates the labelled DM as a peak in K_time = 1
        and identifies which K_time best matches the burst.
      - ``per_kernel_snr.png``  — bar chart of per-kernel max SNR
        (across the cube, not constrained to the burst location);
        operator inspects this to confirm the burst saturates the
        K_time bank coherently (~√K_time scaling).
      - ``candidates.html``     — flat table of all post-merge
        Candidates with kernel_id, l, m, fine_dm, t_in_cube, SNR,
        width.
      - ``report.html``         — master report with the burst-match
        summary box + off-burst contamination table + figures + links.

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
    (out_dir / "report.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return report_path


def _import_matplotlib():
    """Lazy matplotlib import (mirrors search_helpers._import_matplotlib).

    The viz tool defers matplotlib until the renderer actually runs so
    that ``--help`` and the cube_injection-only code path don't pay
    the import cost in environments where the operator does not need
    figure rendering.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433
    return plt


def _render_burst_butterfly_png(
    *,
    dm_curves: List[dict],
    out_path: Path,
    title: str,
    threshold_sigma: float,
    burst_dm_pc_cc: Optional[float] = None,
    labelled_dm_pc_cc: Optional[float] = None,
) -> None:
    """Render the matched-filter butterfly: SNR heatmap on the
    (K_time, fine_dm) grid at the recovered burst (l, m).

    The "butterfly" name comes from the canonical pulsar-search
    diagnostic where a true broadband burst lights up a diamond-
    shaped ridge in (K_time, fine_dm): it pinches at K_time matched
    to the burst's intrinsic width, and broadens at K_time >> width
    where the boxcar integrates noise across mismatched DM trials.

    Vertical guides mark the recovered burst DM (white dashed) and
    the labelled DM (cyan dashed) for direct visual comparison.
    """
    plt = _import_matplotlib()
    if not dm_curves:
        return
    k_time_widths = [int(c["k_time_width"]) for c in dm_curves]
    n_k = len(k_time_widths)
    n_fdm = len(dm_curves[0]["curve"])
    dm_values = [float(p["dm_pc_cc"]) for p in dm_curves[0]["curve"]]
    grid = [[0.0] * n_fdm for _ in range(n_k)]
    for k, kc in enumerate(dm_curves):
        for f, p in enumerate(kc["curve"]):
            grid[k][f] = float(p["snr"])
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    score_max = max(max(row) for row in grid) if grid else 0.0
    vmin = 0.0
    vmax = max(threshold_sigma * 1.5, score_max)
    im = ax.imshow(
        grid, origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
        aspect="auto",
        extent=(dm_values[0], dm_values[-1], -0.5, n_k - 0.5),
    )
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([f"K_time={k}" for k in k_time_widths])
    ax.set_xlabel("fine DM (pc/cc)")
    ax.set_ylabel("K_time boxcar width (samples)")
    ax.set_title(title)
    if burst_dm_pc_cc is not None:
        ax.axvline(
            burst_dm_pc_cc, color="white", linestyle="--", linewidth=1.2,
            label=f"recovered DM = {burst_dm_pc_cc:.2f}",
        )
    if labelled_dm_pc_cc is not None:
        ax.axvline(
            labelled_dm_pc_cc, color="cyan", linestyle="--", linewidth=1.2,
            label=f"labelled DM = {labelled_dm_pc_cc:.2f}",
        )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.colorbar(im, ax=ax, label="SNR (σ)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_burst_dm_curve_png(
    *,
    dm_curves: List[dict],
    out_path: Path,
    title: str,
    threshold_sigma: float,
    burst_dm_pc_cc: Optional[float] = None,
    labelled_dm_pc_cc: Optional[float] = None,
) -> None:
    """Render SNR vs fine_dm overlaid for each K_time kernel.

    A true broadband burst peaks at the labelled DM in K_time=1
    (which is the un-integrated cube max-over-time) and stays
    consistent in DM across K_time values that are close to the
    intrinsic burst width. K_time >> width tends to migrate the peak
    DM away from the labelled value (boxcar smearing); the operator
    inspects this curve for the K_time at which the peak DM stops
    moving.
    """
    plt = _import_matplotlib()
    if not dm_curves:
        return
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    for kc in dm_curves:
        k_time = int(kc["k_time_width"])
        dms = [float(p["dm_pc_cc"]) for p in kc["curve"]]
        snrs = [float(p["snr"]) for p in kc["curve"]]
        ax.plot(dms, snrs, marker="o", markersize=2.5,
                label=f"K_time={k_time}")
    ax.axhline(
        threshold_sigma, color="grey", linestyle=":", linewidth=1.0,
        label=f"threshold = {threshold_sigma:g} σ",
    )
    if labelled_dm_pc_cc is not None:
        ax.axvline(
            labelled_dm_pc_cc, color="cyan", linestyle="--", linewidth=1.2,
            label=f"labelled DM = {labelled_dm_pc_cc:.2f}",
        )
    if burst_dm_pc_cc is not None:
        ax.axvline(
            burst_dm_pc_cc, color="black", linestyle="--", linewidth=1.0,
            label=f"recovered DM = {burst_dm_pc_cc:.2f}",
        )
    ax.set_xlabel("fine DM (pc/cc)")
    ax.set_ylabel("SNR (σ)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_per_kernel_snr_png(
    *,
    per_kernel_stats: List[dict],
    out_path: Path,
    title: str,
    threshold_sigma: float,
) -> None:
    """Bar chart of per-kernel max SNR across the entire cube.

    The K_time bank is expected to scale roughly as √K_time for an
    ideal broadband burst with width matched to one of the K_time
    bins; off-burst persistent sources / RFI in the field of view
    integrate coherently across larger K_time and can dominate the
    max for K_time >> burst_width. The operator inspects this curve
    against the per-kernel positions (annotated above each bar) to
    distinguish burst kernels from off-burst contaminators.
    """
    plt = _import_matplotlib()
    if not per_kernel_stats:
        return
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    kernel_ids = [s["kernel_id"] for s in per_kernel_stats]
    snrs = [float(s["snr_max"]) for s in per_kernel_stats]
    bars = ax.bar(range(len(kernel_ids)), snrs, color="steelblue")
    ax.axhline(
        threshold_sigma, color="grey", linestyle=":",
        label=f"threshold = {threshold_sigma:g} σ",
    )
    for i, s in enumerate(per_kernel_stats):
        pos = s["snr_max_pos"]
        ax.text(
            i, snrs[i] + 0.5,
            f"(l={int(pos['l'])}, m={int(pos['m'])}, "
            f"fdm={int(pos['fdm'])}, t={int(pos['t'])})",
            ha="center", va="bottom", fontsize=6, rotation=0,
        )
    ax.set_xticks(range(len(kernel_ids)))
    ax.set_xticklabels(kernel_ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("max SNR (σ)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_burst_candidates_table_html(
    *,
    candidates: List[dict],
    burst_match: Optional[dict],
    out_path: Path,
    title: str,
) -> None:
    """Burst-mode candidates table.

    Each row is one Candidate; the burst-matched row is highlighted in
    pale green; off-burst candidates are highlighted in pale yellow.
    Rendered with full kernel/DM/position context per plan §8 line 1883.
    """
    burst_kernel = (
        burst_match.get("matched_kernel_id") if burst_match else None
    )
    burst_dm = (
        float(burst_match.get("dm_fine_pc_cc"))
        if burst_match and burst_match.get("dm_fine_pc_cc") is not None
        else None
    )
    burst_lm = (
        (int(burst_match.get("l_pix")), int(burst_match.get("m_pix")))
        if burst_match else None
    )
    rows: List[str] = []
    for i, c in enumerate(candidates):
        kid = c.get("kernel_id", "")
        l_pix = int(c.get("l_pix", 0))
        m_pix = int(c.get("m_pix", 0))
        is_burst_match = (
            burst_lm is not None
            and abs(l_pix - burst_lm[0]) <= 5
            and abs(m_pix - burst_lm[1]) <= 5
        )
        cls = " class='burst-match'" if is_burst_match else " class='off-burst'"
        rows.append(
            f"<tr{cls}>"
            f"<td>{i + 1}</td>"
            f"<td>{kid}</td>"
            f"<td>{float(c.get('snr', 0.0)):.2f}</td>"
            f"<td>{l_pix}</td>"
            f"<td>{m_pix}</td>"
            f"<td>{int(c.get('dm_idx', 0))}</td>"
            f"<td>{float(c.get('dm_fine_pc_cc', 0.0)):.3f}</td>"
            f"<td>{int(c.get('t_in_cube', 0))}</td>"
            f"<td>{int(c.get('width_samples', 0))}</td>"
            f"</tr>"
        )
    body = "\n".join(rows) if rows else (
        "<tr><td colspan=9>(no candidates)</td></tr>"
    )
    head = (
        "<tr><th>#</th><th>kernel_id</th><th>SNR (σ)</th>"
        "<th>l_pix</th><th>m_pix</th><th>dm_idx</th>"
        "<th>fine_dm (pc/cc)</th><th>t_in_cube</th><th>width</th></tr>"
    )
    html = (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:1100px;margin:1em auto;}"
        "table{border-collapse:collapse;width:100%;font-size:90%;}"
        "th,td{border:1px solid #bbb;padding:3px 6px;text-align:right;}"
        "th{background:#eee;}"
        "tr.burst-match{background:#dff5d8;}"
        "tr.off-burst{background:#fff7d6;}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p><span style='background:#dff5d8;padding:2px 6px;'>"
        "green = within ±5 pix of recovered burst (l, m)</span> &nbsp; "
        "<span style='background:#fff7d6;padding:2px 6px;'>"
        "yellow = off-burst high-SNR detection (likely persistent "
        "source / RFI in field of view)</span></p>"
        f"<table>{head}{body}</table>"
        "</body></html>\n"
    )
    out_path.write_text(html, encoding="utf-8")


def _render_burst_report(
    *,
    out_dir: Path,
    detector_record: dict,
    voltage_run_id: str,
) -> Path:
    """Stitch the burst-mode report.html from the detector.json record.

    The detector.json is the canonical artifact (output of
    ``bench/captured_burst_detector.py`` or
    ``bench/voltage_fixture_search.py --mode captured --detector-sweep``);
    this renderer is decoupled from the bench so the operator can re-
    inspect a previous run without re-running the GPU pipeline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config = detector_record.get("config", {}) or {}
    manifest = detector_record.get("manifest", {}) or {}
    src_truth = manifest.get("src_truth", {}) or {}
    burst_match = detector_record.get("burst_match")
    rfi_contamination = detector_record.get("rfi_contamination", []) or []
    per_kernel_stats = detector_record.get("per_kernel_stats", []) or []
    candidates = detector_record.get("candidates", []) or []
    dm_curves = detector_record.get("dm_curves_at_top_candidate_lm", []) or []
    threshold_sigma = float(config.get("threshold_sigma", 8.0))
    burst_dm = (
        float(burst_match["dm_fine_pc_cc"])
        if burst_match and burst_match.get("dm_fine_pc_cc") is not None
        else None
    )
    labelled_dm = src_truth.get("dm_pc_cc")
    labelled_dm = float(labelled_dm) if labelled_dm is not None else None

    butterfly_png = "butterfly.png"
    dm_curve_png = "dm_curve.png"
    per_kernel_snr_png = "per_kernel_snr.png"
    candidates_html = "candidates.html"

    _render_burst_butterfly_png(
        dm_curves=dm_curves, out_path=out_dir / butterfly_png,
        title=(
            f"Matched-filter butterfly: SNR(K_time, fine_dm) at "
            f"recovered (l, m)"
        ),
        threshold_sigma=threshold_sigma,
        burst_dm_pc_cc=burst_dm, labelled_dm_pc_cc=labelled_dm,
    )
    _render_burst_dm_curve_png(
        dm_curves=dm_curves, out_path=out_dir / dm_curve_png,
        title="DM curve overlay (one line per K_time kernel)",
        threshold_sigma=threshold_sigma,
        burst_dm_pc_cc=burst_dm, labelled_dm_pc_cc=labelled_dm,
    )
    _render_per_kernel_snr_png(
        per_kernel_stats=per_kernel_stats,
        out_path=out_dir / per_kernel_snr_png,
        title="Per-kernel max SNR (annotated with (l, m, fdm, t) of max-cell)",
        threshold_sigma=threshold_sigma,
    )
    _render_burst_candidates_table_html(
        candidates=candidates, burst_match=burst_match,
        out_path=out_dir / candidates_html,
        title=f"Candidates — voltage_run_id={voltage_run_id}",
    )

    burst_summary_html = ""
    if burst_match is not None:
        bm = burst_match
        burst_summary_html = (
            "<section style='background:#dff5d8;padding:8px 12px;"
            "border:1px solid #88c466;margin:1em 0;'>"
            "<h2 style='margin-top:0;'>Burst recovered ✓</h2>"
            "<table style='width:auto;'>"
            f"<tr><td>matched kernel</td><td><b>{bm['matched_kernel_id']}</b>"
            "</td></tr>"
            f"<tr><td>matched SNR</td><td><b>{float(bm['matched_snr']):.2f} σ</b>"
            "</td></tr>"
            f"<tr><td>K_time=1 baseline (b1)</td><td>"
            f"{float(bm['b1_snr']):.2f} σ</td></tr>"
            f"<tr><td>matched-filter SNR boost</td><td>"
            f"{float(bm['matched_filter_snr_boost']):.2f}×</td></tr>"
            f"<tr><td>recovered (l_pix, m_pix)</td><td>"
            f"({int(bm['l_pix'])}, {int(bm['m_pix'])})</td></tr>"
            f"<tr><td>recovered fine DM</td><td>"
            f"{float(bm['dm_fine_pc_cc']):.3f} pc/cc "
            f"(fdm_idx={int(bm['dm_idx'])})</td></tr>"
            f"<tr><td>labelled DM</td><td>"
            f"{float(bm['labelled_dm_pc_cc']):.3f} pc/cc</td></tr>"
            f"<tr><td>DM residual</td><td>"
            f"{float(bm['dm_residual_frac']) * 100:+.2f}%</td></tr>"
            f"<tr><td>DM consistency check</td><td>"
            f"{'PASS (within ±2% labelled DM)' if bm.get('dm_consistent') else 'WARN (outside ±2% labelled DM)'}"
            "</td></tr>"
            "</table>"
            "</section>"
        )
    else:
        burst_summary_html = (
            "<section style='background:#f7d8d8;padding:8px 12px;"
            "border:1px solid #c46666;margin:1em 0;'>"
            "<h2 style='margin-top:0;'>No burst match</h2>"
            "<p>The detector emitted candidates but none were within "
            "the spatial-consistency window of the K_time=1 max-SNR "
            "position. See the candidates table below for the raw "
            "emit stream; the per-kernel SNR plot identifies the "
            "off-burst contaminators.</p>"
            "</section>"
        )

    rfi_html = ""
    if rfi_contamination:
        rfi_rows = []
        for rc in rfi_contamination:
            rfi_rows.append(
                f"<tr><td>{rc['kernel_id']}</td>"
                f"<td>{float(rc['snr']):.2f}</td>"
                f"<td>{int(rc['l_pix'])}</td>"
                f"<td>{int(rc['m_pix'])}</td>"
                f"<td>{float(rc['dm_fine_pc_cc']):.3f}</td>"
                f"<td>{int(rc['t_in_cube'])}</td></tr>"
            )
        rfi_html = (
            "<section><h2>Off-burst high-SNR detections "
            f"({len(rfi_contamination)})</h2>"
            "<p style='font-size:90%;'>Likely persistent sources or "
            "RFI in the field of view. Plan §4.2 expects the corr-"
            "side static-sky-subtract IIR to remove these before the "
            "M3 wire payload, so a populated table here means the "
            "captured fixture preserves them.</p>"
            "<table style='width:auto;'>"
            "<tr><th>kernel_id</th><th>SNR</th><th>l_pix</th>"
            "<th>m_pix</th><th>fine_dm</th><th>t_in_cube</th></tr>"
            + "\n".join(rfi_rows) + "</table></section>"
        )

    figures = [
        FigureEntry(
            png_filename=butterfly_png,
            caption="Matched-filter butterfly (SNR vs K_time × fine_dm)",
        ),
        FigureEntry(
            png_filename=dm_curve_png,
            caption="DM curve overlay (per K_time kernel)",
        ),
        FigureEntry(
            png_filename=per_kernel_snr_png,
            caption="Per-kernel max SNR across the cube",
        ),
    ]

    header_meta: dict = {
        "Run": voltage_run_id,
        "Mode": "burst",
        "Generated": datetime.now(timezone.utc).isoformat(),
        "Tool": "search_detector_check",
        "Captured dir": detector_record.get("captured_dir", "?"),
        "Manifest run_id": manifest.get("run_id", "?"),
        "src_kind": manifest.get("src_kind", "?"),
        "Labelled DM (pc/cc)": (
            f"{float(labelled_dm):.3f}" if labelled_dm is not None else "—"
        ),
        "Labelled T2 SNR": (
            f"{float(src_truth.get('t2_snr')):.2f}"
            if src_truth.get("t2_snr") is not None else "—"
        ),
        "T_det": int(config.get("detector_t_det", 0)) or "?",
        "N_fdm": int(config.get("n_fdm", 0)) or "?",
        "N_grid": int(config.get("n_grid", 0)) or "?",
        "threshold_sigma": f"{threshold_sigma:g}",
        "n_candidates_post_merge": int(
            detector_record.get("n_candidates_post_merge") or 0
        ),
        "imager_ms": (
            f"{float(detector_record.get('timings_ms', {}).get('imager_total', 0.0)):.1f}"
        ),
        "detector_ms": (
            f"{float(detector_record.get('timings_ms', {}).get('detector_total', 0.0)):.1f}"
        ),
    }

    report_path = stitch_search_html_report(
        out_dir=out_dir,
        title=f"Search-detector burst report — {voltage_run_id}",
        header_meta=header_meta,
        figures=figures,
        candidates_html_filename=candidates_html,
        extra_links=(),
    )

    # Splice the burst-summary box + RFI table into the report.html
    # AFTER stitch_search_html_report writes it (the stitcher
    # produces a generic header; burst-mode wants the green/red
    # PASS/no-match summary box + off-burst table inserted between
    # the header and the figures).
    html = report_path.read_text(encoding="utf-8")
    html = html.replace(
        "</table>", "</table>" + burst_summary_html + rfi_html, 1,
    )
    report_path.write_text(html, encoding="utf-8")
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
        "--detector-json",
        help="Path to detector.json from "
             "bench/captured_burst_detector.py or "
             "bench/voltage_fixture_search.py --mode captured "
             "--detector-sweep "
             "(required for --mode burst).",
    )
    ap.add_argument(
        "--include-dumps", action="store_true",
        help="Include voltage-dump round-trip + .fil viz (M6 only).",
    )
    ap.add_argument("--out", required=True, help="Output report directory.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()

    if args.mode == "burst":
        if not args.detector_json:
            ap.error("--mode burst requires --detector-json")
        det_path = Path(args.detector_json)
        if not det_path.is_file():
            ap.error(f"--detector-json {det_path} not found")
        detector_record = json.loads(det_path.read_text())
        voltage_run_id = (
            args.voltage_run_id
            or detector_record.get("manifest", {}).get("run_id")
            or "captured-burst"
        )
        report_path = _render_burst_report(
            out_dir=out_dir,
            detector_record=detector_record,
            voltage_run_id=str(voltage_run_id),
        )
        print(f"[search_detector_check] wrote {report_path}")
        return 0

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
