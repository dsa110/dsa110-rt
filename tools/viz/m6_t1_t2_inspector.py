#!/usr/bin/env python3
"""tools/viz/m6_t1_t2_inspector.py — operator-facing T1/T2 ASCII log
inspector (M6 chunk 8 — operator-approval gate).

Loads one or more T1 (per-candidate) and T2 (per-cluster) ASCII log
files written by ``dsart.cluster.cands_logger.CandsLogger`` (M6 chunk 2)
and emits a 2x2 PNG summary plot + a markdown report for operator
inspection.

Schema (locked at M6 D1; see ``src/dsart/cluster/cands_logger.py`` for
the canonical column lists):

  * T1 (per-candidate)::

        mjd  event_specnum  l_rad  m_rad  l_pix  m_pix  dm_fine_pc_cc
        fine_dm_idx  t_in_cube  width_samples  snr  kernel_id  cl
        is_cluster_peak  search_node_id  gpu_half

  * T2 (per-cluster)::

        mjd  event_specnum  l_rad  m_rad  l_pix  m_pix  dm_fine_pc_cc
        fine_dm_idx  t_in_cube  width_samples  snr  kernel_id  cluster_id
        cntc  cntb_lm  cntb_dm  cube_dump_triggered  search_node_id
        gpu_half

Files are discovered by name pattern in ``--log-root``:

    cands_T1_s${sid}_g${g}_${YYYYMMDD}_${HH}.txt
    cands_T2_s${sid}_g${g}_${YYYYMMDD}_${HH}.txt

CLI::

    python -m tools.viz.m6_t1_t2_inspector \\
        --log-root bench/reports/M6/cands_log \\
        --report-dir bench/reports/M6/viz \\
        --since 60942.0 --until 60942.5 \\
        --search-node-id 0 --gpu-half 0

Outputs::

    ${report_dir}/m6_t1_t2_inspector.png   — 2x2 summary plot
    ${report_dir}/m6_t1_t2_inspector.md    — markdown report

Per the M6 chunk-8 spec the report carries NO PASS/FAIL banner — the
operator inspects figures + tables and signs off out-of-band by editing
``bench/reports/M6/m_operator_approved.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Mirrors ``dsart.cluster.cands_logger.T1_COLUMNS`` /  ``T2_COLUMNS``.
# Duplicated here so the inspector does not import torch via the dsart
# package on hosts that lack it (the worktree host runs no GPU code).
T1_COLUMNS: Tuple[str, ...] = (
    "mjd",
    "event_specnum",
    "l_rad",
    "m_rad",
    "l_pix",
    "m_pix",
    "dm_fine_pc_cc",
    "fine_dm_idx",
    "t_in_cube",
    "width_samples",
    "snr",
    "kernel_id",
    "cl",
    "is_cluster_peak",
    "search_node_id",
    "gpu_half",
)

T2_COLUMNS: Tuple[str, ...] = (
    "mjd",
    "event_specnum",
    "l_rad",
    "m_rad",
    "l_pix",
    "m_pix",
    "dm_fine_pc_cc",
    "fine_dm_idx",
    "t_in_cube",
    "width_samples",
    "snr",
    "kernel_id",
    "cluster_id",
    "cntc",
    "cntb_lm",
    "cntb_dm",
    "cube_dump_triggered",
    "search_node_id",
    "gpu_half",
)


# ---------------------------------------------------------------------------
# Loaded-table containers
# ---------------------------------------------------------------------------


@dataclass
class T1Table:
    """In-memory T1 table after loading + filtering."""

    mjd: np.ndarray              # float64 [N]
    event_specnum: np.ndarray    # int64   [N]
    l_rad: np.ndarray            # float64 [N]
    m_rad: np.ndarray            # float64 [N]
    dm_fine_pc_cc: np.ndarray    # float64 [N]
    t_in_cube: np.ndarray        # int64   [N]
    width_samples: np.ndarray    # int64   [N]
    snr: np.ndarray              # float64 [N]
    cl: np.ndarray               # int64   [N]
    is_cluster_peak: np.ndarray  # int64   [N]
    search_node_id: np.ndarray   # int64   [N]
    gpu_half: np.ndarray         # int64   [N]
    n_files: int = 0

    @property
    def n_rows(self) -> int:
        return int(self.mjd.shape[0])

    @property
    def n_noise(self) -> int:
        return int(np.sum(self.cl < 0))

    @property
    def n_peak(self) -> int:
        return int(np.sum(self.is_cluster_peak == 1))


@dataclass
class T2Table:
    """In-memory T2 table after loading + filtering."""

    mjd: np.ndarray              # float64 [N]
    event_specnum: np.ndarray    # int64   [N]
    l_rad: np.ndarray            # float64 [N]
    m_rad: np.ndarray            # float64 [N]
    dm_fine_pc_cc: np.ndarray    # float64 [N]
    t_in_cube: np.ndarray        # int64   [N]
    width_samples: np.ndarray    # int64   [N]
    snr: np.ndarray              # float64 [N]
    cluster_id: np.ndarray       # int64   [N]
    cntc: np.ndarray             # int64   [N]
    cntb_lm: np.ndarray          # int64   [N]
    cntb_dm: np.ndarray          # int64   [N]
    cube_dump_triggered: np.ndarray  # int64 [N]
    search_node_id: np.ndarray   # int64   [N]
    gpu_half: np.ndarray         # int64   [N]
    n_files: int = 0

    @property
    def n_rows(self) -> int:
        return int(self.mjd.shape[0])

    @property
    def n_noise(self) -> int:
        return int(np.sum(self.cluster_id < 0))

    @property
    def n_triggered(self) -> int:
        return int(np.sum(self.cube_dump_triggered == 1))


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------


def discover_log_files(
    log_root: Path,
    *,
    kind: str,
    search_node_id: Optional[int] = None,
    gpu_half: Optional[int] = None,
) -> List[Path]:
    """Return sorted T1 (or T2) log files matching the optional filters.

    ``kind`` must be ``"T1"`` or ``"T2"``. Filenames are matched per the
    ``cands_logger`` writer convention::

        cands_${kind}_s${sid}_g${g}_${YYYYMMDD}_${HH}.txt

    Files outside ``log_root`` (e.g. nested subdirs) are not crawled —
    the writer drops them straight into the supplied ``log_root``.
    """
    if kind not in ("T1", "T2"):
        raise ValueError(f"kind={kind!r} must be 'T1' or 'T2'")
    if not log_root.exists():
        return []
    sid_token = "*" if search_node_id is None else str(search_node_id)
    g_token = "*" if gpu_half is None else str(gpu_half)
    pattern = f"cands_{kind}_s{sid_token}_g{g_token}_*.txt"
    return sorted(log_root.glob(pattern))


def _parse_columns(
    rows: List[List[str]], columns: Sequence[str]
) -> dict[str, np.ndarray]:
    """Parse string-token rows into per-column numpy arrays.

    String columns (kernel_id) are kept as ``np.ndarray[str]``; integer
    columns (event_specnum, *_pix, *_idx, *_samples, cl/cluster_id,
    flags, sid, g) are int64; the rest are float64. The clusterer
    writer emits all numerics as space-separated tokens — see
    ``cands_logger._format_t1_row`` / ``_format_t2_row``.
    """
    int_cols = {
        "event_specnum",
        "l_pix",
        "m_pix",
        "fine_dm_idx",
        "t_in_cube",
        "width_samples",
        "cl",
        "is_cluster_peak",
        "cluster_id",
        "cntc",
        "cntb_lm",
        "cntb_dm",
        "cube_dump_triggered",
        "search_node_id",
        "gpu_half",
    }
    str_cols = {"kernel_id"}
    n_rows = len(rows)
    n_cols = len(columns)
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(columns):
        col = [row[j] for row in rows] if n_rows else []
        if name in int_cols:
            out[name] = np.asarray(col, dtype=np.int64) if col else np.zeros(0, dtype=np.int64)
        elif name in str_cols:
            out[name] = np.asarray(col, dtype=object) if col else np.zeros(0, dtype=object)
        else:
            out[name] = np.asarray(col, dtype=np.float64) if col else np.zeros(0, dtype=np.float64)
    if n_rows and any(len(r) != n_cols for r in rows):
        raise ValueError(
            f"row width mismatch: expected {n_cols} cols, got "
            f"{set(len(r) for r in rows)}"
        )
    return out


def _read_rows(path: Path) -> List[List[str]]:
    """Yield non-comment, non-blank rows split on whitespace."""
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def load_t1_logs(paths: Sequence[Path]) -> T1Table:
    """Load + concatenate one or more T1 log files (no MJD filtering)."""
    all_rows: List[List[str]] = []
    for p in paths:
        all_rows.extend(_read_rows(p))
    parsed = _parse_columns(all_rows, T1_COLUMNS)
    return T1Table(
        mjd=parsed["mjd"],
        event_specnum=parsed["event_specnum"],
        l_rad=parsed["l_rad"],
        m_rad=parsed["m_rad"],
        dm_fine_pc_cc=parsed["dm_fine_pc_cc"],
        t_in_cube=parsed["t_in_cube"],
        width_samples=parsed["width_samples"],
        snr=parsed["snr"],
        cl=parsed["cl"],
        is_cluster_peak=parsed["is_cluster_peak"],
        search_node_id=parsed["search_node_id"],
        gpu_half=parsed["gpu_half"],
        n_files=len(paths),
    )


def load_t2_logs(paths: Sequence[Path]) -> T2Table:
    """Load + concatenate one or more T2 log files (no MJD filtering)."""
    all_rows: List[List[str]] = []
    for p in paths:
        all_rows.extend(_read_rows(p))
    parsed = _parse_columns(all_rows, T2_COLUMNS)
    return T2Table(
        mjd=parsed["mjd"],
        event_specnum=parsed["event_specnum"],
        l_rad=parsed["l_rad"],
        m_rad=parsed["m_rad"],
        dm_fine_pc_cc=parsed["dm_fine_pc_cc"],
        t_in_cube=parsed["t_in_cube"],
        width_samples=parsed["width_samples"],
        snr=parsed["snr"],
        cluster_id=parsed["cluster_id"],
        cntc=parsed["cntc"],
        cntb_lm=parsed["cntb_lm"],
        cntb_dm=parsed["cntb_dm"],
        cube_dump_triggered=parsed["cube_dump_triggered"],
        search_node_id=parsed["search_node_id"],
        gpu_half=parsed["gpu_half"],
        n_files=len(paths),
    )


def filter_t1(
    table: T1Table,
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> T1Table:
    """Apply the inclusive [since, until] MJD window filter."""
    if table.n_rows == 0:
        return table
    keep = np.ones(table.n_rows, dtype=bool)
    if since is not None:
        keep &= table.mjd >= since
    if until is not None:
        keep &= table.mjd <= until
    if keep.all():
        return table
    return T1Table(
        mjd=table.mjd[keep],
        event_specnum=table.event_specnum[keep],
        l_rad=table.l_rad[keep],
        m_rad=table.m_rad[keep],
        dm_fine_pc_cc=table.dm_fine_pc_cc[keep],
        t_in_cube=table.t_in_cube[keep],
        width_samples=table.width_samples[keep],
        snr=table.snr[keep],
        cl=table.cl[keep],
        is_cluster_peak=table.is_cluster_peak[keep],
        search_node_id=table.search_node_id[keep],
        gpu_half=table.gpu_half[keep],
        n_files=table.n_files,
    )


def filter_t2(
    table: T2Table,
    *,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> T2Table:
    """Apply the inclusive [since, until] MJD window filter."""
    if table.n_rows == 0:
        return table
    keep = np.ones(table.n_rows, dtype=bool)
    if since is not None:
        keep &= table.mjd >= since
    if until is not None:
        keep &= table.mjd <= until
    if keep.all():
        return table
    return T2Table(
        mjd=table.mjd[keep],
        event_specnum=table.event_specnum[keep],
        l_rad=table.l_rad[keep],
        m_rad=table.m_rad[keep],
        dm_fine_pc_cc=table.dm_fine_pc_cc[keep],
        t_in_cube=table.t_in_cube[keep],
        width_samples=table.width_samples[keep],
        snr=table.snr[keep],
        cluster_id=table.cluster_id[keep],
        cntc=table.cntc[keep],
        cntb_lm=table.cntb_lm[keep],
        cntb_dm=table.cntb_dm[keep],
        cube_dump_triggered=table.cube_dump_triggered[keep],
        search_node_id=table.search_node_id[keep],
        gpu_half=table.gpu_half[keep],
        n_files=table.n_files,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render_summary_png(
    t1: T1Table, t2: T2Table, *, out_path: Path, title: str
) -> None:
    """Render the 2x2 operator-summary PNG.

    Panels:
      (a) SNR histogram for T1 candidates (log-Y).
      (b) DM vs time scatter for T2 clusters (color = SNR, marker size
          ∝ cntc).
      (c) Width-vs-SNR scatter for T2 clusters (logx).
      (d) Sky map (l vs m, T2 clusters, color = SNR).

    Empty data is rendered with an explanatory annotation rather than
    crashing — operators routinely run the inspector against partial
    log roots.
    """
    try:
        import matplotlib  # noqa: E402
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required to render PNGs") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=11)

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # --- (a) T1 SNR histogram (log-Y) ---
    ax_a.set_title("(a) T1 SNR histogram")
    ax_a.set_xlabel("SNR (sigma)")
    ax_a.set_ylabel("N (log)")
    ax_a.set_yscale("log")
    if t1.n_rows > 0:
        snr = np.asarray(t1.snr, dtype=np.float64)
        snr_max = float(np.max(snr))
        snr_min = float(np.min(snr))
        if snr_max == snr_min:
            bins = np.array([snr_min - 0.5, snr_min + 0.5])
        else:
            bins = np.linspace(snr_min, snr_max, 40)
        ax_a.hist(snr, bins=bins, color="#3a78c2", edgecolor="black",
                  linewidth=0.4, log=True)
        ax_a.axvline(float(np.median(snr)), color="orange",
                     linestyle="--", lw=1.0, label=f"median={np.median(snr):.2f}")
        ax_a.legend(fontsize=8, loc="upper right")
    else:
        ax_a.text(0.5, 0.5, "no T1 rows", ha="center", va="center",
                  transform=ax_a.transAxes, color="#888")

    # --- (b) DM vs time scatter (T2) ---
    ax_b.set_title("(b) T2 cluster DM vs time")
    ax_b.set_xlabel("t_in_cube (samples)")
    ax_b.set_ylabel("dm_fine_pc_cc")
    if t2.n_rows > 0:
        sizes = 6 + 2.0 * np.clip(t2.cntc.astype(np.float64), 1, 30)
        sc = ax_b.scatter(
            t2.t_in_cube, t2.dm_fine_pc_cc, c=t2.snr, cmap="viridis",
            s=sizes, edgecolor="black", linewidth=0.3, alpha=0.85,
        )
        fig.colorbar(sc, ax=ax_b, label="SNR")
    else:
        ax_b.text(0.5, 0.5, "no T2 rows", ha="center", va="center",
                  transform=ax_b.transAxes, color="#888")

    # --- (c) Width vs SNR (T2) ---
    ax_c.set_title("(c) T2 cluster width vs SNR")
    ax_c.set_xlabel("width_samples (log)")
    ax_c.set_ylabel("SNR (sigma)")
    if t2.n_rows > 0:
        # Ensure positive widths for log-x; the contract guards
        # width_samples > 0 but defend against a rogue 0 anyway.
        w = np.maximum(t2.width_samples.astype(np.float64), 1.0)
        ax_c.set_xscale("log")
        ax_c.scatter(w, t2.snr, c=t2.snr, cmap="viridis",
                     s=22, edgecolor="black", linewidth=0.3, alpha=0.85)
    else:
        ax_c.text(0.5, 0.5, "no T2 rows", ha="center", va="center",
                  transform=ax_c.transAxes, color="#888")

    # --- (d) Sky map (T2) ---
    ax_d.set_title("(d) T2 cluster sky map")
    ax_d.set_xlabel("l_rad")
    ax_d.set_ylabel("m_rad")
    if t2.n_rows > 0:
        sc2 = ax_d.scatter(
            t2.l_rad, t2.m_rad, c=t2.snr, cmap="plasma",
            s=24, edgecolor="black", linewidth=0.3, alpha=0.85,
        )
        fig.colorbar(sc2, ax=ax_d, label="SNR")
        ax_d.set_aspect("equal", adjustable="datalim")
    else:
        ax_d.text(0.5, 0.5, "no T2 rows", ha="center", va="center",
                  transform=ax_d.transAxes, color="#888")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_window(since: Optional[float], until: Optional[float]) -> str:
    if since is None and until is None:
        return "(unbounded)"
    lo = "-inf" if since is None else f"{since:.6f}"
    hi = "+inf" if until is None else f"{until:.6f}"
    return f"[{lo}, {hi}]"


def _fmt_filter(
    search_node_id: Optional[int], gpu_half: Optional[int]
) -> str:
    sid = "*" if search_node_id is None else str(search_node_id)
    g = "*" if gpu_half is None else str(gpu_half)
    return f"sid={sid}, gpu_half={g}"


def _fmt_or_dash(arr: np.ndarray, fmt: str) -> str:
    if arr.size == 0:
        return "-"
    return fmt.format(float(arr[0]))


def render_markdown_report(
    *,
    t1: T1Table,
    t2: T2Table,
    log_root: Path,
    report_dir: Path,
    image_filename: str,
    out_path: Path,
    since: Optional[float],
    until: Optional[float],
    search_node_id: Optional[int],
    gpu_half: Optional[int],
    t1_paths: Sequence[Path],
    t2_paths: Sequence[Path],
) -> None:
    """Render ``m6_t1_t2_inspector.md`` with a summary table + image link.

    Per the M6 chunk-8 spec the report is operator-facing only — no
    PASS/FAIL banner; the ``bench/reports/M6/m_operator_approved.yaml``
    marker carries the gate.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if t1.n_rows > 0:
        snr_min_t1 = float(np.min(t1.snr))
        snr_max_t1 = float(np.max(t1.snr))
        snr_med_t1 = float(np.median(t1.snr))
    else:
        snr_min_t1 = snr_max_t1 = snr_med_t1 = float("nan")

    if t2.n_rows > 0:
        snr_min_t2 = float(np.min(t2.snr))
        snr_max_t2 = float(np.max(t2.snr))
        snr_med_t2 = float(np.median(t2.snr))
    else:
        snr_min_t2 = snr_max_t2 = snr_med_t2 = float("nan")

    md_lines: List[str] = []
    md_lines.append("# M6 T1/T2 candidate-log inspector")
    md_lines.append("")
    md_lines.append(
        "Operator-facing summary of the M6 chunk-2 ASCII candidate logs. "
        "No PASS/FAIL banner — sign off out-of-band by editing "
        "`bench/reports/M6/m_operator_approved.yaml`."
    )
    md_lines.append("")
    md_lines.append("## Run metadata")
    md_lines.append("")
    md_lines.append(f"- log_root: `{log_root}`")
    md_lines.append(f"- report_dir: `{report_dir}`")
    md_lines.append(f"- mjd window: {_fmt_window(since, until)}")
    md_lines.append(f"- filter: {_fmt_filter(search_node_id, gpu_half)}")
    md_lines.append(f"- T1 files matched: {len(t1_paths)}")
    md_lines.append(f"- T2 files matched: {len(t2_paths)}")
    md_lines.append("")
    md_lines.append("## Counts")
    md_lines.append("")
    md_lines.append("| kind | rows | noise (cl<0) | peak/triggered |")
    md_lines.append("| ---- | ---: | ---: | ---: |")
    md_lines.append(
        f"| T1 (per-candidate) | {t1.n_rows} | {t1.n_noise} | "
        f"{t1.n_peak} (is_cluster_peak=1) |"
    )
    md_lines.append(
        f"| T2 (per-cluster)   | {t2.n_rows} | {t2.n_noise} | "
        f"{t2.n_triggered} (cube_dump_triggered=1) |"
    )
    md_lines.append("")
    md_lines.append("## SNR summary")
    md_lines.append("")
    md_lines.append("| kind | min | median | max |")
    md_lines.append("| ---- | ---: | ---: | ---: |")
    md_lines.append(
        f"| T1 | {snr_min_t1:.3f} | {snr_med_t1:.3f} | {snr_max_t1:.3f} |"
    )
    md_lines.append(
        f"| T2 | {snr_min_t2:.3f} | {snr_med_t2:.3f} | {snr_max_t2:.3f} |"
    )
    md_lines.append("")
    md_lines.append("## Summary plot")
    md_lines.append("")
    md_lines.append(f"![T1/T2 inspector]({image_filename})")
    md_lines.append("")
    md_lines.append(
        "Panels: (a) T1 SNR histogram (log-Y), (b) T2 DM vs time scatter "
        "(color = SNR, marker size ∝ cntc), (c) T2 width vs SNR (log-x), "
        "(d) T2 sky map (color = SNR)."
    )
    md_lines.append("")
    md_lines.append("## Files inspected")
    md_lines.append("")
    if t1_paths:
        md_lines.append("### T1")
        for p in t1_paths:
            md_lines.append(f"- `{p}`")
        md_lines.append("")
    if t2_paths:
        md_lines.append("### T2")
        for p in t2_paths:
            md_lines.append(f"- `{p}`")
        md_lines.append("")
    if not (t1_paths or t2_paths):
        md_lines.append("(no log files matched the filter — empty report)")
        md_lines.append("")

    out_path.write_text("\n".join(md_lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="m6_t1_t2_inspector",
        description=(
            "Operator-facing inspector for M6 chunk-2 T1/T2 ASCII "
            "candidate logs."
        ),
    )
    p.add_argument(
        "--log-root", required=True, type=Path,
        help="Directory holding cands_T{1,2}_s*_g*_YYYYMMDD_HH.txt files.",
    )
    p.add_argument(
        "--report-dir", required=True, type=Path,
        help="Output directory for the PNG + MD report.",
    )
    p.add_argument(
        "--since", type=float, default=None,
        help="MJD lower bound (inclusive). Default: unbounded.",
    )
    p.add_argument(
        "--until", type=float, default=None,
        help="MJD upper bound (inclusive). Default: unbounded.",
    )
    p.add_argument(
        "--search-node-id", type=int, default=None,
        help="Filter to a single search_node_id (matches the file's _s suffix).",
    )
    p.add_argument(
        "--gpu-half", type=int, default=None,
        help="Filter to a single gpu_half (matches the file's _g suffix).",
    )
    p.add_argument(
        "--title", type=str, default=None,
        help="Optional override for the figure suptitle.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    log_root: Path = args.log_root
    report_dir: Path = args.report_dir

    t1_paths = discover_log_files(
        log_root, kind="T1",
        search_node_id=args.search_node_id, gpu_half=args.gpu_half,
    )
    t2_paths = discover_log_files(
        log_root, kind="T2",
        search_node_id=args.search_node_id, gpu_half=args.gpu_half,
    )

    if not t1_paths and not t2_paths:
        msg = (
            f"[m6_t1_t2_inspector] no T1/T2 logs found under {log_root!s} "
            f"({_fmt_filter(args.search_node_id, args.gpu_half)}); "
            "still emitting an empty operator report."
        )
        print(msg)

    t1 = filter_t1(load_t1_logs(t1_paths), since=args.since, until=args.until)
    t2 = filter_t2(load_t2_logs(t2_paths), since=args.since, until=args.until)

    report_dir.mkdir(parents=True, exist_ok=True)
    png_path = report_dir / "m6_t1_t2_inspector.png"
    md_path = report_dir / "m6_t1_t2_inspector.md"

    title = args.title or (
        f"M6 T1/T2 inspector — {log_root}\n"
        f"window={_fmt_window(args.since, args.until)}; "
        f"filter={_fmt_filter(args.search_node_id, args.gpu_half)}; "
        f"T1 rows={t1.n_rows}; T2 rows={t2.n_rows}"
    )
    render_summary_png(t1, t2, out_path=png_path, title=title)
    render_markdown_report(
        t1=t1, t2=t2,
        log_root=log_root, report_dir=report_dir,
        image_filename=png_path.name,
        out_path=md_path,
        since=args.since, until=args.until,
        search_node_id=args.search_node_id, gpu_half=args.gpu_half,
        t1_paths=t1_paths, t2_paths=t2_paths,
    )
    print(f"[m6_t1_t2_inspector] wrote {png_path}")
    print(f"[m6_t1_t2_inspector] wrote {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
