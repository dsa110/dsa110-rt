"""tools/viz/search_helpers.py — M5-owned shared helpers for the
search-detector viz tools (plan §4.7 + §8 line 2329; M5 PARALLEL_AGENTS.md
§3 Class C carve-out).

``tools/viz/common.py`` is M3-owned (M2 hardening retired). Per
PARALLEL_AGENTS.md §3 + M5_PLAN_FIXES.md D5, M5 does NOT edit it; instead
all M5-only viz primitives live here.

Contents:

  * ``stitch_search_html_report`` — master HTML stitcher (figures +
    captions + per-criterion observed-vs-expected tables); NO PASS/FAIL
    banner per plan §4.7.
  * ``render_recovery_heatmap_png`` — 2D ``(injected_snr, width_samples)``
    recovery heatmap (cube_injection mode).
  * ``render_score_per_kernel_png`` — 4×4 grid (one tile per
    image-token), each tile a (k_dm × k_time) heatmap of detector score.
  * ``render_far_curve_png`` — empirical-vs-analytic Gaussian-tail FAR
    curve.
  * ``render_candidates_table_html`` — Candidate listing as a
    self-contained HTML table; called by both modes.
  * Pure-Python helpers (no torch dep) so the viz tool runs in any
    matplotlib-only env on h01 / dev laptop.

The dual-mode CLI (``tools/viz/search_detector_check.py``) imports from
here exclusively; it never imports ``tools/viz/common.py`` directly
(per F8/D5; corr-side helpers there are wrong-shape for the search side
anyway).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def gaussian_tail_far(theta: float) -> float:
    """One-sided Gaussian tail probability ``0.5 · erfc(θ/√2)``.

    Per plan §8 line 2329 + F3 in M5_PLAN_FIXES.md, this is the
    analytic per-cell false-alarm probability for an iid unit-σ
    Gaussian background; the cube_injection bench's FAR sub-check
    compares this against the empirical rate over 30 s of synthetic
    noise-only cubes.
    """
    return 0.5 * math.erfc(float(theta) / math.sqrt(2.0))


def n_eff_per_cube_per_kernel(
    *,
    t_det: int,
    n_fdm: int,
    n_grid: int,
    k_img_volume: int,
    k_dm_width: int,
    k_time_width: int,
) -> float:
    """Per-cube effective-number-of-cells for one kernel triple.

    Per M5_PLAN_FIXES.md F3 footnote:
        ``N_eff = (T_det × N_fdm × N_grid²) / (K_img · K_dm · K_time)``.

    Used to convert ``gaussian_tail_far(θ)`` into an expected emit-rate
    per cube per kernel.
    """
    if k_img_volume < 1 or k_dm_width < 1 or k_time_width < 1:
        raise ValueError(
            f"kernel widths must be ≥ 1; got "
            f"img={k_img_volume}, dm={k_dm_width}, time={k_time_width}"
        )
    n_cells = float(t_det) * float(n_fdm) * float(n_grid) * float(n_grid)
    return n_cells / (
        float(k_img_volume) * float(k_dm_width) * float(k_time_width)
    )


# ---------------------------------------------------------------------------
# Matplotlib bootstrapping (Agg backend; hold the import lazy so the rest
# of the module remains importable in environments without matplotlib).
# ---------------------------------------------------------------------------


def _import_matplotlib():
    try:
        import matplotlib  # noqa: E402
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        return plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for tools/viz/search_*; "
            "install via the dsa110-rt conda env"
        ) from exc


# ---------------------------------------------------------------------------
# Recovery heatmap (cube_injection mode)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoveryCell:
    """One ``(injected_snr, width_samples)`` cell of the recovery sweep."""

    injected_snr: float
    width_samples: int
    n_injected: int
    n_recovered: int
    snr_ratio_mean: float  # mean(recovered_snr / injected_snr); NaN if 0 recovered

    @property
    def recovery_fraction(self) -> float:
        return (self.n_recovered / self.n_injected) if self.n_injected > 0 else 0.0


def render_recovery_heatmap_png(
    cells: Sequence[RecoveryCell],
    *,
    out_path: Path,
    title: str = "Recovery vs (injected_snr, width)",
    figsize: Tuple[float, float] = (8.0, 5.5),
) -> None:
    """Render a 2D recovery heatmap. Colour = recovery_fraction; each
    cell annotated with ``f/n  r=<snr_ratio_mean>`` so the operator can
    eyeball both fraction-recovered and amplitude bias in one glance.
    """
    plt = _import_matplotlib()

    snrs = sorted({c.injected_snr for c in cells})
    widths = sorted({c.width_samples for c in cells})
    grid = [[float("nan")] * len(widths) for _ in snrs]
    annot = [[""] * len(widths) for _ in snrs]
    for c in cells:
        i = snrs.index(c.injected_snr)
        j = widths.index(c.width_samples)
        grid[i][j] = c.recovery_fraction
        ratio = c.snr_ratio_mean
        ratio_str = f"r={ratio:.2f}" if math.isfinite(ratio) else "r=—"
        annot[i][j] = f"{c.n_recovered}/{c.n_injected}\n{ratio_str}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        grid, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto",
    )
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(len(snrs)))
    ax.set_yticklabels([f"{s:g}" for s in snrs])
    ax.set_xlabel("injected width (samples)")
    ax.set_ylabel("injected SNR (σ)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="fraction recovered")
    for i, _row in enumerate(grid):
        for j, _ in enumerate(_row):
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-injection score-per-kernel heatmap (4×4 grid: image × time, each
# tile shows k_dm rows). 128 kernel triples = 4 image × 4 dm × 8 time.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KernelScoreEntry:
    image_token: str
    dm_token: str
    time_token: str
    k_dm_width: int
    k_time_width: int
    snr: float


def render_score_per_kernel_png(
    entries: Sequence[KernelScoreEntry],
    *,
    out_path: Path,
    title: str,
    figsize: Tuple[float, float] = (10.0, 9.0),
    threshold_sigma: float = 8.0,
) -> None:
    """Render the 128-cell post-Layer-2 score heatmap.

    Layout: outer grid is 4 image tokens (one row per token); inner grid
    per row is dm × time (4 × 8 in v1). Plan §8 line 1882: confirms the
    matched kernel triple wins.
    """
    plt = _import_matplotlib()

    image_tokens = sorted({e.image_token for e in entries})
    dm_tokens = sorted({e.dm_token for e in entries})
    time_tokens = sorted({e.time_token for e in entries})
    n_img = len(image_tokens)
    n_dm = len(dm_tokens)
    n_time = len(time_tokens)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_img, 1, figsize=figsize, squeeze=False, sharex=True,
    )

    score_max = max((e.snr for e in entries), default=0.0)
    vmin = 0.0
    vmax = max(threshold_sigma * 1.5, score_max)

    for i_img, img_tok in enumerate(image_tokens):
        ax = axes[i_img][0]
        grid = [[float("nan")] * n_time for _ in range(n_dm)]
        for e in entries:
            if e.image_token != img_tok:
                continue
            i_dm = dm_tokens.index(e.dm_token)
            i_t = time_tokens.index(e.time_token)
            grid[i_dm][i_t] = e.snr
        im = ax.imshow(
            grid, origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(range(n_time))
        ax.set_xticklabels(time_tokens)
        ax.set_yticks(range(n_dm))
        ax.set_yticklabels(dm_tokens)
        ax.set_ylabel(f"img={img_tok}\nk_dm")
        for i_dm in range(n_dm):
            for i_t in range(n_time):
                v = grid[i_dm][i_t]
                if math.isnan(v):
                    continue
                color = "white" if v < (vmax * 0.6) else "black"
                ax.text(i_t, i_dm, f"{v:.1f}", ha="center", va="center",
                        color=color, fontsize=6)
        fig.colorbar(im, ax=ax, label="SNR (σ)")

    axes[-1][0].set_xlabel("k_time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Empirical-vs-analytic FAR curve (cube_injection noise-only sub-bench)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FarSample:
    """One ``θ`` bin of the noise-only FAR sub-check.

    ``empirical_per_cube_per_kernel`` is the per-(cube, kernel) emit
    rate above ``theta`` averaged over the noise-only cubes. The
    bench is responsible for normalising correctly.
    """

    theta: float
    empirical_per_cube_per_kernel: float
    analytic_per_cube_per_kernel: float
    n_cubes: int
    n_kernels: int


def render_far_curve_png(
    samples: Sequence[FarSample],
    *,
    out_path: Path,
    title: str = "Noise-only FAR (empirical vs analytic Gaussian tail)",
    figsize: Tuple[float, float] = (7.5, 5.5),
) -> None:
    """Plot empirical and analytic FAR vs threshold ``θ`` on log-y."""
    plt = _import_matplotlib()

    samples = sorted(samples, key=lambda s: s.theta)
    thetas = [s.theta for s in samples]
    emp = [max(s.empirical_per_cube_per_kernel, 1e-30) for s in samples]
    ana = [max(s.analytic_per_cube_per_kernel, 1e-30) for s in samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(thetas, ana, "k--", label="analytic 0.5·erfc(θ/√2)·N_eff")
    ax.semilogy(thetas, emp, "ro-", label="empirical (per cube per kernel)")
    ax.set_xlabel("threshold θ (σ)")
    ax.set_ylabel("expected events per cube per kernel")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Candidates HTML table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One Candidate row for the operator-facing HTML table.

    ``observed`` is the recovered Candidate; ``injected`` is the matching
    injection (or None for noise-only / unmatched candidates).
    """

    rank: int
    observed: Mapping[str, Any]
    injected: Optional[Mapping[str, Any]] = None


def render_candidates_table_html(
    rows: Sequence[CandidateRow],
    *,
    out_path: Path,
    title: str = "Recovered candidates",
) -> None:
    """Render a self-contained ``candidates.html`` (per plan §8 line
    1875/1883). When an injection is supplied, deltas are reported
    side-by-side; otherwise a "(noise-only)" row is emitted.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body_rows: list[str] = []
    for row in rows:
        obs = row.observed
        inj = row.injected
        cells = [
            f"<td>{row.rank}</td>",
            f"<td>{obs.get('kernel_id', '')}</td>",
            f"<td>{float(obs.get('snr', float('nan'))):.2f}</td>",
            f"<td>{int(obs.get('l', 0))}</td>",
            f"<td>{int(obs.get('m', 0))}</td>",
            f"<td>{int(obs.get('dm_idx', 0))}</td>",
            f"<td>{int(obs.get('event_specnum', 0))}</td>",
            f"<td>{int(obs.get('width_samples', 0))}</td>",
        ]
        if inj is None:
            cells.append("<td colspan=4>(noise-only / unmatched)</td>")
        else:
            d_l = int(obs.get("l", 0)) - int(inj.get("l_pix", 0))
            d_m = int(obs.get("m", 0)) - int(inj.get("m_pix", 0))
            d_dm = int(obs.get("dm_idx", 0)) - int(inj.get("fine_dm_idx", 0))
            d_t = int(obs.get("event_specnum", 0)) - int(inj.get("t_in_cube", 0))
            cells.append(f"<td>{d_l:+d}</td>")
            cells.append(f"<td>{d_m:+d}</td>")
            cells.append(f"<td>{d_dm:+d}</td>")
            cells.append(f"<td>{d_t:+d}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    head = (
        "<tr><th>#</th><th>kernel_id</th><th>SNR (σ)</th><th>l_pix</th>"
        "<th>m_pix</th><th>dm_idx</th><th>event_specnum</th>"
        "<th>width</th><th>Δl</th><th>Δm</th><th>Δdm</th><th>Δt</th></tr>"
    )
    rows_html = "\n".join(body_rows) if body_rows else (
        "<tr><td colspan=12>(no candidates)</td></tr>"
    )

    html = (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:1100px;margin:1em auto;}"
        "table{border-collapse:collapse;width:100%;font-size:90%;}"
        "th,td{border:1px solid #bbb;padding:3px 6px;text-align:right;}"
        "th{background:#eee;}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f"<table>{head}{rows_html}</table>"
        "</body></html>\n"
    )
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Master HTML stitcher (NO PASS/FAIL banner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FigureEntry:
    """One figure for the master report.

    ``observed`` and ``expected`` may both be None (figure-only entry,
    no metric row in the table).
    """

    png_filename: str
    caption: str
    observed: Optional[str] = None
    expected: Optional[str] = None


def stitch_search_html_report(
    *,
    out_dir: Path,
    title: str,
    header_meta: Mapping[str, Any],
    figures: Sequence[FigureEntry],
    candidates_html_filename: Optional[str] = None,
    extra_links: Iterable[Tuple[str, str]] = (),
) -> Path:
    """Write a self-contained ``report.html`` master report.

    Per plan §4.7 + §8 line 1887: NO PASS/FAIL banner. Header line
    format ``Run: <run_id_or_inj> | Mode: <mode> | Generated: <UTC ns> | Tool: search_detector_check``.

    Args:
        out_dir: report directory; PNGs are referenced relative.
        title: master report ``<h1>`` and ``<title>``.
        header_meta: dict of metadata key→value rendered into the
            header table (Run, Mode, Generated UTC ns, Tool, etc.).
        figures: sequence of ``FigureEntry``; each renders a ``<h2>``
            + ``<img>`` + caption + (when supplied) observed/expected/
            delta row.
        candidates_html_filename: optional filename of a sibling
            ``candidates.html`` to link from the master report.
        extra_links: optional ``(label, href)`` tuples rendered in a
            "Links" section; href is relative to ``out_dir``.

    Returns the path to the generated ``report.html``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in header_meta.items()
    )

    fig_blocks: list[str] = []
    metric_rows: list[str] = []
    for f_entry in figures:
        fig_blocks.append(
            f"<section>\n"
            f"  <h2>{f_entry.caption}</h2>\n"
            f"  <img src='{f_entry.png_filename}' "
            f"alt='{f_entry.png_filename}'>\n"
            f"</section>"
        )
        if f_entry.observed is not None or f_entry.expected is not None:
            obs_s = "—" if f_entry.observed is None else f_entry.observed
            exp_s = "—" if f_entry.expected is None else f_entry.expected
            metric_rows.append(
                f"<tr><td>{f_entry.caption}</td><td>{obs_s}</td>"
                f"<td>{exp_s}</td></tr>"
            )

    metrics_html = ""
    if metric_rows:
        metrics_html = (
            "<h2>Per-criterion metrics</h2>\n"
            "<table><tr><th>criterion</th><th>observed</th><th>expected</th></tr>\n"
            + "\n".join(metric_rows) + "</table>"
        )

    links_html = ""
    if candidates_html_filename or extra_links:
        items: list[str] = []
        if candidates_html_filename:
            items.append(
                f"<li><a href='{candidates_html_filename}'>"
                f"Candidates (HTML table)</a></li>"
            )
        for label, href in extra_links:
            items.append(f"<li><a href='{href}'>{label}</a></li>")
        links_html = "<h2>Links</h2><ul>" + "".join(items) + "</ul>"

    html = (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:1000px;margin:1em auto;}"
        "table{border-collapse:collapse;margin:1em 0;}"
        "th,td{border:1px solid #ccc;padding:4px 8px;}"
        "th{background:#eee;}"
        ".meta td:first-child{font-weight:bold;}"
        "img{max-width:100%;border:1px solid #ccc;}"
        ".note{background:#fffbe6;padding:8px;border-left:4px solid #fc0;}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p class='note'><strong>No PASS/FAIL banner</strong> — per plan §4.7,"
        f" this report is for operator inspection only. The operator opens"
        f" this in a browser and signs off in a one-line reply.</p>"
        f"<h2>Header</h2>"
        f"<table class='meta'>{meta_rows}</table>"
        f"{''.join(fig_blocks)}"
        f"{metrics_html}"
        f"{links_html}"
        "</body></html>\n"
    )

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
