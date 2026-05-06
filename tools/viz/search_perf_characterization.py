#!/usr/bin/env python3
"""tools/viz/search_perf_characterization.py — M5 Chunk 6c-γ
perf-vs-quality characterization report.

Consumes one or more bank-masked runs of:

  * ``bench/cube_injection_detector.py``  (recovery at θ=8σ per bank-mask)
  * ``bench/search_node_throughput.py``   (cubes/s + per-stage percentiles)

and renders a single self-contained HTML report that lets the operator
pick a Pareto-optimal kernel-bank configuration.

CLI (all paths point to a per-run output directory containing the bench's
``summary.json``; the bench-level ``--bank-mask`` selects the config):

  python -m tools.viz.search_perf_characterization                  \\
      --recovery-summary  path/full/summary.json                    \\
      --recovery-summary  path/k_dm_d1/summary.json                 \\
      --recovery-summary  path/k_img_unit/summary.json              \\
      --recovery-summary  path/k_img_unit_k_dm_d1/summary.json      \\
      --throughput-summary path/full/summary.json                   \\
      --throughput-summary path/k_dm_d1/summary.json                \\
      --throughput-summary path/k_img_unit/summary.json             \\
      --throughput-summary path/k_img_unit_k_dm_d1/summary.json     \\
      [--target-cubes-per-s 8.0]                                    \\
      [--mid-snr 8.0]                                               \\
      [--mid-width 32]                                              \\
      --out  bench/reports/<UTC>/perf_characterization/M5/

Outputs (under ``--out``):

  * ``pareto.png``                — cubes/s × recovery-at-(mid_snr, mid_width)
                                    scatter; one point per bank-mask config.
  * ``recovery_grid.png``         — small-multiples grid, one panel per
                                    bank-mask, (snr × width) recovery heatmap.
  * ``throughput_stages.png``     — per-mask p50 stacked bars by stage
                                    (build_cube, layer1_norm, detector_forward,
                                    emitter_dispatch).
  * ``report.html``               — master self-contained report (NO
                                    PASS/FAIL banner per plan §4.7).
  * ``pareto.json``               — machine-readable joined-Pareto table.

Per Chunk 6c framing (operator decision): N_grid=256 is the default; the
128 case is a perf-headroom measurement, not a target. K_time stays full
across all configs (matched-filter SNR-critical axis).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    """Parsed cube_injection summary.json (one bank-mask config)."""
    label: str
    bank_mask: Optional[str]
    n_kernels: int
    cells: Tuple[dict, ...]
    t_det: int
    n_fdm: int
    n_grid: int
    detector_threshold_sigma: float
    far_samples: Tuple[dict, ...]
    source_path: str

    def cell(self, snr: float, width: int) -> Optional[dict]:
        """Lookup a cell by ``(injected_snr, injected_width)``; return None
        if not present (the operator may pick a mid-point that isn't on
        the bench's grid)."""
        for c in self.cells:
            inj = c.get("injected", {})
            if (
                math.isclose(float(inj.get("snr", -1)), float(snr))
                and int(inj.get("width_samples", -1)) == int(width)
            ):
                return c
        return None


@dataclass(frozen=True, slots=True)
class ThroughputSummary:
    """Parsed search_node_throughput summary.json (one (n_grid,
    bank-mask) config)."""
    label: str
    bank_mask: Optional[str]
    n_kernels: int
    t_det: int
    n_fdm: int
    n_grid: int
    achieved_cubes_per_s: float
    n_cubes_processed: int
    percentiles_ms: Dict[str, Dict[str, float]]
    device: str
    cube_dtype: str
    threshold_sigma: float
    source_path: str

    @property
    def total_p50_ms(self) -> float:
        return float(self.percentiles_ms.get(
            "total_pipeline", {}).get("p50", 0.0))

    @property
    def total_p99_ms(self) -> float:
        return float(self.percentiles_ms.get(
            "total_pipeline", {}).get("p99", 0.0))


@dataclass(slots=True)
class ParetoPoint:
    """One joined Pareto-table row. Either axis may be missing if the
    operator only ran one of the two benches for a given config."""
    label: str
    bank_mask: Optional[str]
    n_kernels: int
    n_grid: int
    cubes_per_s: Optional[float] = None
    total_p50_ms: Optional[float] = None
    total_p99_ms: Optional[float] = None
    detector_p50_ms: Optional[float] = None
    build_cube_p50_ms: Optional[float] = None
    layer1_norm_p50_ms: Optional[float] = None
    emitter_dispatch_p50_ms: Optional[float] = None
    recovery_at_mid: Optional[float] = None
    snr_ratio_at_mid: Optional[float] = None
    recovery_summary_path: Optional[str] = None
    throughput_summary_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _label_from_bank_mask(bank_mask: Optional[str]) -> str:
    """Render a bank-mask string as a compact axis label.

    Examples:
      None                            → "full"
      "*"                             → "full"
      "k_img=unit"                    → "k_img=unit"
      "k_img=unit;k_dm=d1"            → "k_img=unit;k_dm=d1"
    """
    if bank_mask is None or not bank_mask.strip() or bank_mask.strip() == "*":
        return "full"
    return bank_mask.strip()


def _load_recovery_summary(path: Path) -> RecoverySummary:
    data = json.loads(path.read_text())
    cfg = data.get("config", {})
    bank_mask = cfg.get("bank_mask")
    resolved = cfg.get("bank_mask_resolved", {})
    n_kernels = int(resolved.get("n_kernels", 128))
    return RecoverySummary(
        label=_label_from_bank_mask(bank_mask),
        bank_mask=bank_mask,
        n_kernels=n_kernels,
        cells=tuple(data.get("cells", [])),
        t_det=int(cfg.get("T_det", 0)),
        n_fdm=int(cfg.get("N_fdm", 0)),
        n_grid=int(cfg.get("N_grid", 0)),
        detector_threshold_sigma=float(
            cfg.get("detector_threshold_sigma", 8.0)
        ),
        far_samples=tuple(data.get("far", [])),
        source_path=str(path.resolve()),
    )


def _load_throughput_summary(path: Path) -> ThroughputSummary:
    data = json.loads(path.read_text())
    cfg = data.get("config", {})
    bank_mask = cfg.get("bank_mask")
    resolved = cfg.get("bank_mask_resolved", {})
    n_kernels = int(resolved.get("n_kernels", 128))
    return ThroughputSummary(
        label=_label_from_bank_mask(bank_mask),
        bank_mask=bank_mask,
        n_kernels=n_kernels,
        t_det=int(cfg.get("t_det", 0)),
        n_fdm=int(cfg.get("n_fdm", 0)),
        n_grid=int(cfg.get("n_grid", 0)),
        achieved_cubes_per_s=float(data.get("achieved_cubes_per_s", 0.0)),
        n_cubes_processed=int(data.get("n_cubes_processed", 0)),
        percentiles_ms=dict(data.get("percentiles_ms", {})),
        device=str(cfg.get("device", "")),
        cube_dtype=str(cfg.get("cube_dtype", "")),
        threshold_sigma=float(cfg.get("threshold_sigma", 8.0)),
        source_path=str(path.resolve()),
    )


# ---------------------------------------------------------------------------
# Pareto join
# ---------------------------------------------------------------------------


def _join_key(bank_mask: Optional[str], n_grid: int) -> str:
    """Composite key for joining recovery + throughput points by
    ``(n_grid, bank_mask)``."""
    return f"N{n_grid}|{_label_from_bank_mask(bank_mask)}"


def join_pareto(
    recoveries: Sequence[RecoverySummary],
    throughputs: Sequence[ThroughputSummary],
    *,
    mid_snr: float,
    mid_width: int,
) -> List[ParetoPoint]:
    """Join recovery + throughput summaries by (n_grid, bank_mask).

    The recovery axis comes from the cube-level cube_injection bench
    (which doesn't sweep N_grid in any meaningful way — N_grid there is
    just the cube spatial side) so we match recovery to throughput by
    *bank_mask only*. If the same bank-mask is run at multiple N_grid
    values for throughput, the same recovery point is shared.

    A throughput point with no matching recovery summary is still
    rendered (recovery_at_mid stays None); same for the converse.
    """
    points: Dict[str, ParetoPoint] = {}

    # Build by-bank-mask recovery index (last-write-wins across multiple
    # cube_injection runs; the cube-level recovery doesn't depend on
    # N_grid in any meaningful way).
    recovery_by_mask: Dict[str, RecoverySummary] = {}
    for r in recoveries:
        recovery_by_mask[r.label] = r

    # Seed points from throughput first (these carry n_grid).
    for th in throughputs:
        key = _join_key(th.bank_mask, th.n_grid)
        pct = th.percentiles_ms
        pp = ParetoPoint(
            label=th.label,
            bank_mask=th.bank_mask,
            n_kernels=th.n_kernels,
            n_grid=th.n_grid,
            cubes_per_s=th.achieved_cubes_per_s,
            total_p50_ms=float(pct.get("total_pipeline", {}).get("p50", 0.0)),
            total_p99_ms=float(pct.get("total_pipeline", {}).get("p99", 0.0)),
            detector_p50_ms=float(
                pct.get("detector_forward", {}).get("p50", 0.0)
            ),
            build_cube_p50_ms=float(
                pct.get("build_cube", {}).get("p50", 0.0)
            ),
            layer1_norm_p50_ms=float(
                pct.get("layer1_norm", {}).get("p50", 0.0)
            ),
            emitter_dispatch_p50_ms=float(
                pct.get("emitter_dispatch", {}).get("p50", 0.0)
            ),
            throughput_summary_path=th.source_path,
        )
        rec = recovery_by_mask.get(th.label)
        if rec is not None:
            cell = rec.cell(mid_snr, mid_width)
            if cell is not None:
                pp.recovery_at_mid = float(cell.get("recovery_fraction", 0.0))
                ratio = cell.get("snr_ratio_mean")
                pp.snr_ratio_at_mid = (
                    float(ratio) if ratio is not None else None
                )
            pp.recovery_summary_path = rec.source_path
        points[key] = pp

    # Add recovery-only points (no matching throughput run).
    seen_masks = {p.label for p in points.values()}
    for rec in recoveries:
        if rec.label in seen_masks:
            continue
        key = _join_key(rec.bank_mask, rec.n_grid)
        cell = rec.cell(mid_snr, mid_width)
        recovery = (
            float(cell.get("recovery_fraction", 0.0))
            if cell is not None else None
        )
        ratio = cell.get("snr_ratio_mean") if cell is not None else None
        pp = ParetoPoint(
            label=rec.label,
            bank_mask=rec.bank_mask,
            n_kernels=rec.n_kernels,
            n_grid=rec.n_grid,
            recovery_at_mid=recovery,
            snr_ratio_at_mid=float(ratio) if ratio is not None else None,
            recovery_summary_path=rec.source_path,
        )
        points[key] = pp

    return list(points.values())


# ---------------------------------------------------------------------------
# Headline finding
# ---------------------------------------------------------------------------


def cheapest_viable_at_target(
    points: Sequence[ParetoPoint],
    *,
    target_cubes_per_s: float,
    n_grid_filter: Optional[int] = None,
    min_recovery: float = 0.5,
) -> Optional[ParetoPoint]:
    """Find the smallest n_kernels point that hits the target rate AND
    keeps recovery ≥ min_recovery at the mid cell.

    Returns None if no point qualifies. Only considers points with both
    axes present.
    """
    candidates: List[ParetoPoint] = []
    for p in points:
        if p.cubes_per_s is None or p.recovery_at_mid is None:
            continue
        if n_grid_filter is not None and p.n_grid != n_grid_filter:
            continue
        if p.cubes_per_s < target_cubes_per_s:
            continue
        if p.recovery_at_mid < min_recovery:
            continue
        candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.n_kernels, -p.cubes_per_s))
    return candidates[0]


def render_headline(
    points: Sequence[ParetoPoint],
    *,
    target_cubes_per_s: float,
    mid_snr: float,
    mid_width: int,
) -> str:
    """Single-paragraph operator headline. Neutral language (no
    PASS/FAIL); the operator decides which config to deploy."""
    cheapest_256 = cheapest_viable_at_target(
        points, target_cubes_per_s=target_cubes_per_s, n_grid_filter=256,
    )
    cheapest_any = cheapest_viable_at_target(
        points, target_cubes_per_s=target_cubes_per_s, n_grid_filter=None,
    )
    if cheapest_256 is not None:
        return (
            f"At N_grid=256, the cheapest viable bank reaching "
            f"≥ {target_cubes_per_s:.1f} cubes/s while preserving recovery "
            f"≥ 0.5 at (snr={mid_snr:.0f}, width={mid_width}) is "
            f"<b>{escape(cheapest_256.label)}</b> "
            f"({cheapest_256.n_kernels} kernels; "
            f"{cheapest_256.cubes_per_s:.2f} cubes/s; "
            f"recovery={cheapest_256.recovery_at_mid:.2f})."
        )
    if cheapest_any is not None:
        return (
            f"No N_grid=256 bank reaches "
            f"≥ {target_cubes_per_s:.1f} cubes/s. The cheapest viable "
            f"point sits at N_grid={cheapest_any.n_grid}: "
            f"<b>{escape(cheapest_any.label)}</b> "
            f"({cheapest_any.n_kernels} kernels; "
            f"{cheapest_any.cubes_per_s:.2f} cubes/s; "
            f"recovery={cheapest_any.recovery_at_mid:.2f}). Reaching "
            f"the target on a 256 grid requires the GPU port "
            f"(combiner+imager+detector on cuda)."
        )
    best_rate = max(
        (p.cubes_per_s for p in points if p.cubes_per_s is not None),
        default=0.0,
    )
    return (
        f"No measured bank-mask + N_grid combination hits "
        f"≥ {target_cubes_per_s:.1f} cubes/s "
        f"(best observed: {best_rate:.2f}). The GPU port "
        f"(combiner+imager+detector on cuda; cuFFT plan cache; fused "
        f"boxcar) is required regardless of bank shrinkage."
    )


# ---------------------------------------------------------------------------
# Plot rendering (matplotlib lazy import)
# ---------------------------------------------------------------------------


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for tools/viz/search_perf_*; "
            "install via the dsa110-rt conda env"
        ) from exc


def render_pareto_png(
    out_path: Path,
    points: Sequence[ParetoPoint],
    *,
    target_cubes_per_s: float,
    mid_snr: float,
    mid_width: int,
) -> None:
    """Cubes/s × recovery-at-(mid_snr, mid_width) scatter."""
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted: List[ParetoPoint] = [
        p for p in points
        if p.cubes_per_s is not None and p.recovery_at_mid is not None
    ]
    if not plotted:
        ax.text(
            0.5, 0.5,
            "No joined points to plot\n"
            "(need both --recovery-summary AND --throughput-summary "
            "for at least one bank_mask config).",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        seen_n_grid: set[int] = set()
        for p in plotted:
            color = "tab:blue" if p.n_grid >= 256 else "tab:orange"
            marker = "o" if p.n_grid >= 256 else "s"
            label = (
                f"N_grid={p.n_grid}"
                if p.n_grid not in seen_n_grid else None
            )
            seen_n_grid.add(p.n_grid)
            ax.scatter(
                [p.cubes_per_s], [p.recovery_at_mid],
                s=80 + 1.2 * p.n_kernels, c=color, marker=marker,
                edgecolors="black", linewidths=0.6, alpha=0.85,
                label=label,
            )
            ax.annotate(
                f"{p.label}\n[{p.n_kernels} kernels]",
                (p.cubes_per_s, p.recovery_at_mid),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8,
            )
        ax.axvline(
            target_cubes_per_s, linestyle="--", color="grey", alpha=0.6,
            label=f"target={target_cubes_per_s:.1f} cubes/s",
        )
        ax.axhline(0.5, linestyle=":", color="grey", alpha=0.4)
        ax.set_xscale("log")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("achieved cubes/s (log scale)")
    ax.set_ylabel(
        f"recovery fraction at (snr={mid_snr:.0f}, width={mid_width})"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "M5 Chunk 6c — perf vs. quality Pareto\n"
        "(point area ∝ n_kernels)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _build_recovery_grid_arr(rec: RecoverySummary) -> Tuple[
    "List[float]", "List[int]", "List[List[float]]"
]:
    """Extract a (snrs, widths, recovery[snr][width]) grid from a
    recovery summary."""
    snrs: List[float] = sorted({
        float(c.get("injected", {}).get("snr", 0.0))
        for c in rec.cells
    })
    widths: List[int] = sorted({
        int(c.get("injected", {}).get("width_samples", 0))
        for c in rec.cells
    })
    grid: List[List[float]] = [
        [float("nan")] * len(widths) for _ in snrs
    ]
    for c in rec.cells:
        inj = c.get("injected", {})
        try:
            si = snrs.index(float(inj.get("snr", 0.0)))
            wi = widths.index(int(inj.get("width_samples", 0)))
        except ValueError:
            continue
        grid[si][wi] = float(c.get("recovery_fraction", float("nan")))
    return snrs, widths, grid


def render_recovery_grid_png(
    out_path: Path,
    recoveries: Sequence[RecoverySummary],
) -> None:
    """Small-multiples grid: one panel per bank-mask, recovery heatmap
    over (snr, width)."""
    plt = _import_matplotlib()
    if not recoveries:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "No recovery summaries provided.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return
    n = len(recoveries)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 4.0 * rows),
        squeeze=False,
    )
    for i, rec in enumerate(recoveries):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        snrs, widths, grid = _build_recovery_grid_arr(rec)
        if not grid or not grid[0]:
            ax.text(
                0.5, 0.5, "no cells", ha="center", va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        im = ax.imshow(
            grid, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
            aspect="auto",
        )
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in widths], fontsize=8)
        ax.set_yticks(range(len(snrs)))
        ax.set_yticklabels([f"{s:g}" for s in snrs], fontsize=8)
        ax.set_xlabel("width_samples")
        ax.set_ylabel("injected SNR")
        ax.set_title(
            f"{rec.label}  ({rec.n_kernels} kernels)",
            fontsize=10,
        )
        for si in range(len(snrs)):
            for wi in range(len(widths)):
                v = grid[si][wi]
                if not math.isnan(v):
                    ax.text(
                        wi, si, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v < 0.5 else "black",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Hide leftover axes if grid isn't full.
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_axis_off()
    fig.suptitle(
        "Recovery fraction by (injected SNR, width) per bank-mask",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_throughput_stages_png(
    out_path: Path,
    throughputs: Sequence[ThroughputSummary],
) -> None:
    """Per-mask p50 stacked bars by stage."""
    plt = _import_matplotlib()
    if not throughputs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "No throughput summaries provided.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return
    labels = [f"{t.label}\nN={t.n_grid}" for t in throughputs]
    stages = (
        "build_cube",
        "layer1_norm",
        "detector_forward",
        "emitter_dispatch",
    )
    stage_p50 = {
        s: [
            float(t.percentiles_ms.get(s, {}).get("p50", 0.0))
            for t in throughputs
        ]
        for s in stages
    }
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(throughputs)), 5))
    x = list(range(len(throughputs)))
    bottom = [0.0] * len(throughputs)
    colors = {
        "build_cube": "tab:blue",
        "layer1_norm": "tab:green",
        "detector_forward": "tab:orange",
        "emitter_dispatch": "tab:red",
    }
    for s in stages:
        vals = stage_p50[s]
        ax.bar(x, vals, bottom=bottom, label=s, color=colors[s])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("p50 stage latency (ms)")
    ax.set_title(
        "Per-stage p50 latency by bank-mask (cumulative)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, t in enumerate(throughputs):
        ax.text(
            x[i], bottom[i], f"{t.achieved_cubes_per_s:.2f}/s",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report HTML stitcher
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>M5 Chunk 6c — perf-vs-quality characterization</title>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; margin: 1.5em;
         color: #222; max-width: 1200px; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.2em; margin-top: 1.5em;
        border-bottom: 1px solid #aaa; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td.label {{ text-align: left; font-family: monospace; }}
  .headline {{ background: #fff7d0; border-left: 4px solid #b58900;
               padding: 0.6em 0.8em; margin: 0.6em 0; }}
  .meta {{ color: #666; font-size: 0.85em; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd;
         margin: 0.5em 0; }}
  .disclaimer {{ color: #555; font-size: 0.85em; font-style: italic;
                 margin-top: 1em; }}
</style>
</head>
<body>

<h1>M5 Chunk 6c — perf-vs-quality characterization</h1>
<div class="meta">{generated_utc} · target = {target_cubes_per_s:.2f} cubes/s
· mid cell = (snr={mid_snr:.0f}, width={mid_width})</div>

<div class="headline">{headline_html}</div>

<h2>Pareto: cubes/s vs. recovery</h2>
<img src="pareto.png" alt="Pareto plot">

<h2>Joined Pareto table</h2>
{pareto_table_html}

<h2>Per-mask recovery heatmaps</h2>
<img src="recovery_grid.png" alt="Recovery grid">

<h2>Per-mask throughput stage breakdown</h2>
<img src="throughput_stages.png" alt="Throughput stage breakdown">

<p class="disclaimer">
No PASS/FAIL banner per plan §4.7. The operator inspects this report
and chooses a kernel-bank configuration. Recovery fractions come from
``bench/cube_injection_detector.py`` runs at the listed bank-mask;
throughput numbers come from ``bench/search_node_throughput.py`` runs
at the listed (n_grid, bank-mask). Per Chunk 6c framing, N_grid=256 is
the operator's preferred grid; N_grid=128 measurements quantify the
perf headroom if 256 ever proves infeasible.
</p>

</body>
</html>
"""


def _format_optional(v: Optional[float], fmt: str = "{:.3f}") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return fmt.format(v)


def _render_pareto_table_html(points: Sequence[ParetoPoint]) -> str:
    rows = sorted(
        points,
        key=lambda p: (
            -(p.cubes_per_s or 0.0),
            p.n_kernels,
            p.label,
        ),
    )
    head = (
        "<tr>"
        "<th>label (bank-mask)</th>"
        "<th>N_grid</th>"
        "<th>n_kernels</th>"
        "<th>cubes/s</th>"
        "<th>total p50 (ms)</th>"
        "<th>total p99 (ms)</th>"
        "<th>detector p50 (ms)</th>"
        "<th>build_cube p50 (ms)</th>"
        "<th>recovery@mid</th>"
        "<th>SNR ratio@mid</th>"
        "</tr>"
    )
    body_rows: List[str] = []
    for p in rows:
        body_rows.append(
            "<tr>"
            f"<td class='label'>{escape(p.label)}</td>"
            f"<td>{p.n_grid}</td>"
            f"<td>{p.n_kernels}</td>"
            f"<td>{_format_optional(p.cubes_per_s, '{:.2f}')}</td>"
            f"<td>{_format_optional(p.total_p50_ms, '{:.2f}')}</td>"
            f"<td>{_format_optional(p.total_p99_ms, '{:.2f}')}</td>"
            f"<td>{_format_optional(p.detector_p50_ms, '{:.2f}')}</td>"
            f"<td>{_format_optional(p.build_cube_p50_ms, '{:.2f}')}</td>"
            f"<td>{_format_optional(p.recovery_at_mid, '{:.2f}')}</td>"
            f"<td>{_format_optional(p.snr_ratio_at_mid, '{:.2f}')}</td>"
            "</tr>"
        )
    return "<table>" + head + "".join(body_rows) + "</table>"


def render_report_html(
    out_dir: Path,
    points: Sequence[ParetoPoint],
    *,
    target_cubes_per_s: float,
    mid_snr: float,
    mid_width: int,
) -> None:
    headline_html = render_headline(
        points,
        target_cubes_per_s=target_cubes_per_s,
        mid_snr=mid_snr,
        mid_width=mid_width,
    )
    pareto_table_html = _render_pareto_table_html(points)
    html = _HTML_TEMPLATE.format(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        target_cubes_per_s=target_cubes_per_s,
        mid_snr=mid_snr,
        mid_width=mid_width,
        headline_html=headline_html,
        pareto_table_html=pareto_table_html,
    )
    (out_dir / "report.html").write_text(html)


def write_pareto_json(out_dir: Path, points: Sequence[ParetoPoint]) -> None:
    out: List[dict] = []
    for p in sorted(
        points,
        key=lambda q: (-(q.cubes_per_s or 0.0), q.n_kernels, q.label),
    ):
        out.append({
            "label": p.label,
            "bank_mask": p.bank_mask,
            "n_kernels": p.n_kernels,
            "n_grid": p.n_grid,
            "cubes_per_s": p.cubes_per_s,
            "total_p50_ms": p.total_p50_ms,
            "total_p99_ms": p.total_p99_ms,
            "detector_p50_ms": p.detector_p50_ms,
            "build_cube_p50_ms": p.build_cube_p50_ms,
            "layer1_norm_p50_ms": p.layer1_norm_p50_ms,
            "emitter_dispatch_p50_ms": p.emitter_dispatch_p50_ms,
            "recovery_at_mid": p.recovery_at_mid,
            "snr_ratio_at_mid": p.snr_ratio_at_mid,
            "recovery_summary_path": p.recovery_summary_path,
            "throughput_summary_path": p.throughput_summary_path,
        })
    (out_dir / "pareto.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M5 Chunk 6c-γ perf-vs-quality characterization report. "
            "Joins recovery (cube_injection) + throughput "
            "(search_node_throughput) summary.json files into a "
            "single Pareto report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recovery-summary", action="append", type=Path, default=[],
        help="Path to a cube_injection_detector summary.json. Repeatable.",
    )
    parser.add_argument(
        "--throughput-summary", action="append", type=Path, default=[],
        help="Path to a search_node_throughput summary.json. Repeatable.",
    )
    parser.add_argument(
        "--target-cubes-per-s", type=float, default=8.0,
        help="Operator throughput target (default 8.0 per plan §8 line 2318).",
    )
    parser.add_argument(
        "--mid-snr", type=float, default=8.0,
        help="Mid-cell SNR for the Pareto y-axis (default 8.0).",
    )
    parser.add_argument(
        "--mid-width", type=int, default=32,
        help="Mid-cell width_samples for the Pareto y-axis (default 32).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory for the report assets.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    recoveries = [_load_recovery_summary(p) for p in args.recovery_summary]
    throughputs = [
        _load_throughput_summary(p) for p in args.throughput_summary
    ]
    if not recoveries and not throughputs:
        parser.error(
            "must supply at least one --recovery-summary or "
            "--throughput-summary path"
        )

    points = join_pareto(
        recoveries, throughputs,
        mid_snr=float(args.mid_snr),
        mid_width=int(args.mid_width),
    )

    render_pareto_png(
        out_dir / "pareto.png", points,
        target_cubes_per_s=float(args.target_cubes_per_s),
        mid_snr=float(args.mid_snr),
        mid_width=int(args.mid_width),
    )
    render_recovery_grid_png(
        out_dir / "recovery_grid.png", recoveries,
    )
    render_throughput_stages_png(
        out_dir / "throughput_stages.png", throughputs,
    )
    render_report_html(
        out_dir, points,
        target_cubes_per_s=float(args.target_cubes_per_s),
        mid_snr=float(args.mid_snr),
        mid_width=int(args.mid_width),
    )
    write_pareto_json(out_dir, points)
    print(
        f"wrote {len(points)} Pareto points to {out_dir}/report.html "
        f"({len(recoveries)} recovery + {len(throughputs)} throughput)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
