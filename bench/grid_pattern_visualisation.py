#!/usr/bin/env python3
"""bench/grid_pattern_visualisation.py — sparsity-pattern fill-fraction sweep.

Plan §3 line 305 prediction: single-side fill fraction is **monotone
non-increasing in `n_grid`** (as the grid grows, the same baseline ×
channel population covers a smaller fraction of cells; the absolute
``N_filled`` count grows only slowly because each (bls, ch) hits one
cell at the K=1 pillbox kernel).

This bench builds :func:`dsart.grid.sparsity_pattern.build_pattern`
patterns at::

    chgroup ∈ {0, 8, 15}
    n_grid  ∈ {128, 256, 384}
    dec_deg ∈ {30, 45, 53.85}

(= 27 combos), against the on-disk 250924mptq antpos
(``/home/ubuntu/data/voltages/250924mptq/cals/beamformer_weights_sb00_*.dat``;
the antpos is the only field of the cal blob this bench reads). Falls
back to the 0319bbb antpos if the 250924mptq cals aren't on this host.

Reports
=======

* Per-combo fill fractions in a small CSV +
  ``bench/reports/<UTC>/<run_id>/M3-grid-pattern/index.html``.
* PNG plot of fill fraction vs ``n_grid`` for each (chgroup, dec) line.
* PNG plot of the filled-cell footprint at ``(chgroup=0,
  dec=53.85, n_grid=256)`` for visual inspection.
* A monotonicity check: assert ``fill_frac(n_grid=128) ≥
  fill_frac(n_grid=256) ≥ fill_frac(n_grid=384)`` for each
  ``(chgroup, dec)`` line. Exit 0 on pass; exit 1 with a printed
  failure list on any violation.

Usage
=====

::

    python -m bench.grid_pattern_visualisation \\
        [--cal-blob /path/to/beamformer_weights_sb00_*.dat] \\
        [--out-dir bench/reports/<UTC>/<run_id>/M3-grid-pattern]

References
==========

* Plan §3 line 305 — fill-fraction monotonicity prediction.
* :mod:`dsart.grid.sparsity_pattern` — module under bench.
* :mod:`dsart.cal.bf_weights` — antpos loader.
* ``PARALLEL_AGENTS.md`` §5 — voltage-fixture / cal-blob conventions
  on h01.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.cal.bf_weights import load_bf_weights                     # noqa: E402
from dsart.common.constants import NANTS                              # noqa: E402
from dsart.grid.sparsity_pattern import build_pattern                 # noqa: E402


# ---------------------------------------------------------------------------
# Antpos loader (cal blob → antpos arrays only)
# ---------------------------------------------------------------------------


def _resolve_antpos_path(explicit: str | None) -> Path:
    """Find a beamformer_weights_*.dat blob to read antpos from.

    Preference order: explicit CLI flag > 250924mptq cals/ > 0319 cals/.
    Errors out (exit 2) if none can be found — the bench needs at
    least one real antpos to produce meaningful fill fractions.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            print(f"ERROR: --cal-blob {p} not found", file=sys.stderr)
            sys.exit(2)
        return p
    candidates: list[str] = []
    for cals_dir, glob_pat in [
        (Path("/home/ubuntu/data/voltages/250924mptq/cals"),
         "beamformer_weights_sb00*.dat"),
        (Path("/home/ubuntu/data/voltages/0319/cals"),
         "beamformer_weights_sb00_*.dat"),
    ]:
        if cals_dir.is_dir():
            for blob in sorted(cals_dir.glob(glob_pat)):
                candidates.append(str(blob))
    for c in candidates:
        if Path(c).is_file():
            return Path(c)
    print(
        "ERROR: no beamformer_weights_*.dat blob found; "
        "pass --cal-blob explicitly.",
        file=sys.stderr,
    )
    sys.exit(2)


def _core_baseline_mask(
    antpos_e: np.ndarray, antpos_n: np.ndarray, n_core: int = 82,
) -> np.ndarray:
    """``(NBASE,) bool`` mask: True iff both ants are core.

    Selects the ``n_core`` smallest-radius antennas as the core (per
    F27 in ``M3_PLAN_FIXES.md``). Production reads the actual
    ``is_core`` array from etcd ``/cnf/corr_setup_96`` per plan §3
    line 446; this radius-based fallback is what the bench uses when
    antpos comes from a cal blob.

    The cal-blob antpos is **not** sorted by radius — e.g. ant index
    48 is an outrigger at r ≈ 1008 m AND ant index 83 is a core ant
    at r ≈ 423 m. The earlier positional helper leaked outrigger
    baselines into the core image and dropped real core baselines,
    leaving stray fills in the outer uv-plane of the footprint plot.
    """
    from dsart.grid import core_baseline_mask_from_antpos
    return core_baseline_mask_from_antpos(
        antpos_e, antpos_n, n_core=n_core,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


CHGROUPS_DEFAULT: tuple[int, ...] = (0, 8, 15)
N_GRIDS_DEFAULT: tuple[int, ...] = (128, 256, 384)                    # 384 is non-pow-of-2
DECS_DEFAULT: tuple[float, ...] = (30.0, 45.0, 53.85)


def run_sweep(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    *,
    chgroups: Iterable[int] = CHGROUPS_DEFAULT,
    n_grids: Iterable[int] = N_GRIDS_DEFAULT,
    decs: Iterable[float] = DECS_DEFAULT,
    is_core_baseline_mask: np.ndarray | None = None,
) -> list[dict]:
    """Build patterns at every (chgroup, n_grid, dec) and return one row each."""
    rows: list[dict] = []
    for chgroup in chgroups:
        for n_grid in n_grids:
            for dec in decs:
                pat = build_pattern(
                    antpos_e, antpos_n,
                    chgroup=chgroup,
                    dec_deg=dec,
                    n_grid=n_grid,
                    kernel_support=1,
                    is_core_baseline_mask=is_core_baseline_mask,
                )
                rows.append({
                    "chgroup": chgroup,
                    "n_grid": n_grid,
                    "dec_deg": dec,
                    "n_filled": pat.n_filled,
                    "fill_frac_pct": 100.0 * pat.n_filled / (n_grid * n_grid),
                    "skipped": "",
                    "_pattern": pat,
                })
    return rows


def check_monotonicity(rows: list[dict]) -> list[str]:
    """Verify fill_frac is monotone non-increasing in n_grid per (chgroup, dec).

    Returns a list of violation strings; empty list ⇒ all clear.
    """
    violations: list[str] = []
    by_pair: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["fill_frac_pct"] < 0:                                    # skipped row
            continue
        by_pair.setdefault((r["chgroup"], r["dec_deg"]), []).append(r)
    for (chg, dec), group in by_pair.items():
        group_sorted = sorted(group, key=lambda r: r["n_grid"])
        for prev, cur in zip(group_sorted[:-1], group_sorted[1:]):
            if cur["fill_frac_pct"] > prev["fill_frac_pct"] + 1e-9:
                violations.append(
                    f"chgroup={chg} dec={dec} n_grid {prev['n_grid']}→"
                    f"{cur['n_grid']}: fill {prev['fill_frac_pct']:.3f}% "
                    f"→ {cur['fill_frac_pct']:.3f}% (NOT monotone non-increasing)"
                )
    return violations


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chgroup", "n_grid", "dec_deg", "n_filled",
                "fill_frac_pct", "skipped",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})


def _write_pngs(rows: list[dict], out_dir: Path) -> list[Path]:
    """Write PNG plots; if matplotlib unavailable, log + skip.

    Returns the list of written PNG paths (empty if plotting was
    skipped).
    """
    written: list[Path] = []
    try:
        import matplotlib                                              # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping PNG output.")
        return written

    valid = [r for r in rows if r["fill_frac_pct"] >= 0]
    if not valid:
        return written

    # Plot 1: fill fraction vs n_grid, one line per (chgroup, dec).
    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    by_pair: dict[tuple, list[dict]] = {}
    for r in valid:
        by_pair.setdefault((r["chgroup"], r["dec_deg"]), []).append(r)
    for (chg, dec), group in sorted(by_pair.items()):
        gs = sorted(group, key=lambda r: r["n_grid"])
        xs = [r["n_grid"] for r in gs]
        ys = [r["fill_frac_pct"] for r in gs]
        ax.plot(xs, ys, marker="o",
                label=f"chgroup={chg}, dec={dec}°")
    ax.set_xlabel("n_grid (cells per axis)")
    ax.set_ylabel("fill fraction (%)")
    ax.set_title(
        "Sparsity-pattern fill fraction vs n_grid\n"
        "(plan §3 line 305 — monotone non-increasing prediction)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "fill_fraction_vs_ngrid.png"
    fig.savefig(p1)
    plt.close(fig)
    written.append(p1)

    # Plot 2: footprint of one representative pattern.
    rep_rows = [r for r in valid
                if r["chgroup"] == 0 and r["dec_deg"] == 53.85
                and r["n_grid"] == 256]
    if rep_rows:
        rep = rep_rows[0]
        pat = rep["_pattern"]
        ng = rep["n_grid"]
        img = np.zeros((ng, ng), dtype=np.float32)
        img[pat.ix_row.astype(int), pat.ix_col.astype(int)] = 1.0
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        ax.imshow(
            img, origin="lower", cmap="viridis", interpolation="nearest",
        )
        ax.set_xlabel("col (u-axis)")
        ax.set_ylabel("row (v-axis)")
        ax.set_title(
            f"chgroup=0, dec=53.85°, n_grid={ng}\n"
            f"n_filled={pat.n_filled} ({rep['fill_frac_pct']:.3f}% fill); "
            f"K=1 pillbox"
        )
        fig.tight_layout()
        p2 = out_dir / "footprint_chgroup0_dec53p85_ngrid256.png"
        fig.savefig(p2)
        plt.close(fig)
        written.append(p2)

    return written


def _write_html(
    rows: list[dict],
    pngs: list[Path],
    csv_path: Path,
    monotonicity_violations: list[str],
    *,
    antpos_source: Path,
    out_html: Path,
) -> None:
    """Self-contained HTML report (no JS) with the table + inline PNGs."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(out_html.parent).as_posix()
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>M3 grid-pattern visualisation</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:980px;margin:2em auto;}",
        "table{border-collapse:collapse;margin:1em 0;}",
        "th,td{border:1px solid #aaa;padding:4px 8px;text-align:right;}",
        "th{background:#eef;}",
        "tr:nth-child(even){background:#f7f7f7;}",
        ".skipped{color:#888;font-style:italic;}",
        ".pass{color:#080;font-weight:bold;}",
        ".fail{color:#c00;font-weight:bold;}",
        "img{max-width:100%;}",
        "</style>",
        "</head><body>",
        "<h1>M3 grid-pattern visualisation</h1>",
        f"<p>Antpos source: <code>{antpos_source}</code></p>",
        f"<p>Plan §3 line 305 monotonicity check: ",
    ]
    if monotonicity_violations:
        parts.append("<span class='fail'>FAIL</span></p>")
        parts.append("<ul>")
        for v in monotonicity_violations:
            parts.append(f"<li>{v}</li>")
        parts.append("</ul>")
    else:
        parts.append("<span class='pass'>PASS</span></p>")

    parts.append("<h2>Fill fractions</h2>")
    parts.append("<table>")
    parts.append(
        "<tr><th>chgroup</th><th>n_grid</th><th>dec_deg</th>"
        "<th>n_filled</th><th>fill (%)</th></tr>"
    )
    for r in rows:
        cls = " class='skipped'" if r.get("skipped") else ""
        nfill = r["n_filled"] if r["n_filled"] >= 0 else "—"
        ffrac = (
            f"{r['fill_frac_pct']:.3f}" if r["fill_frac_pct"] >= 0 else "—"
        )
        notes = (f" <em>({r['skipped']})</em>" if r.get("skipped") else "")
        parts.append(
            f"<tr{cls}><td>{r['chgroup']}</td><td>{r['n_grid']}</td>"
            f"<td>{r['dec_deg']}</td><td>{nfill}</td>"
            f"<td>{ffrac}{notes}</td></tr>"
        )
    parts.append("</table>")
    parts.append(f"<p>CSV: <a href='{rel(csv_path)}'>{csv_path.name}</a></p>")

    if pngs:
        parts.append("<h2>Plots</h2>")
        for p in pngs:
            parts.append(f"<figure><img src='{rel(p)}'/></figure>")

    parts.append("</body></html>")
    out_html.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cal-blob", type=str, default=None,
        help="Path to a beamformer_weights_*.dat blob (antpos source).",
    )
    ap.add_argument(
        "--out-dir", type=str, default=None,
        help=(
            "Output directory. Default: "
            "bench/reports/<UTC>/<run_id>/M3-grid-pattern/"
        ),
    )
    ap.add_argument(
        "--run-id", type=str, default="grid-pattern-bench",
        help="Run-id used in the default --out-dir path.",
    )
    ap.add_argument(
        "--no-mask", action="store_true",
        help=(
            "Skip the 82-ant core mask (use the full 96-ant antpos). "
            "Default: apply the core mask per plan §3 line 452."
        ),
    )
    args = ap.parse_args()

    # Load antpos
    cal_path = _resolve_antpos_path(args.cal_blob)
    bf = load_bf_weights(cal_path)
    print(
        f"loaded antpos from {cal_path} "
        f"({bf.antpos_e.size} ants; max E={bf.antpos_e.max():.1f} m, "
        f"max N={bf.antpos_n.max():.1f} m)"
    )

    if args.no_mask:
        mask = None
    else:
        mask = _core_baseline_mask(bf.antpos_e, bf.antpos_n, n_core=82)
    if mask is not None:
        # Echo which antennas were classified as core for transparency.
        radii = np.hypot(bf.antpos_e, bf.antpos_n)
        sorted_idx = np.argsort(radii, kind="stable")
        core_ants = sorted_idx[:82]
        outrigger_ants = sorted_idx[82:]
        r_core_max = float(radii[core_ants].max())
        r_outrigger_min = float(radii[outrigger_ants].min())
        print(
            f"core mask: {int(mask.sum())} of {mask.size} baselines kept "
            f"(82-ant core by smallest-radius selection; "
            f"core max r={r_core_max:.1f} m, outrigger min r={r_outrigger_min:.1f} m)"
        )
        # Surface any positionally-surprising classifications.
        positional_surprises = sorted(
            [int(a) for a in core_ants if a >= 82] +
            [int(a) for a in outrigger_ants if a < 82]
        )
        if positional_surprises:
            print(
                f"  positional-vs-radius surprise ants: "
                f"{positional_surprises} (radius-based mask differs from "
                f"the legacy 'first 82 are core' helper)"
            )

    # Resolve out-dir
    if args.out_dir is None:
        utc = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        out_dir = (
            REPO_ROOT / "bench" / "reports" / utc / args.run_id
            / "M3-grid-pattern"
        )
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing report to {out_dir}")

    # Sweep
    rows = run_sweep(
        bf.antpos_e, bf.antpos_n,
        is_core_baseline_mask=mask,
    )

    # Monotonicity check
    violations = check_monotonicity(rows)
    if violations:
        print("MONOTONICITY VIOLATIONS:")
        for v in violations:
            print(f"  {v}")
    else:
        print("monotonicity check: PASS (fill_frac non-increasing in n_grid"
              " for every (chgroup, dec))")

    # Reports
    csv_path = out_dir / "fill_fractions.csv"
    _write_csv(rows, csv_path)
    pngs = _write_pngs(rows, out_dir)
    html = out_dir / "index.html"
    _write_html(
        rows, pngs, csv_path, violations,
        antpos_source=cal_path, out_html=html,
    )
    print(f"wrote: {csv_path}")
    print(f"wrote: {html}")
    for p in pngs:
        print(f"wrote: {p}")

    # Print a small fill-fraction table to stdout for the parent
    # M3 agent's eyes.
    print()
    print(f"{'chgroup':>7} {'n_grid':>6} {'dec':>6} {'n_filled':>10} "
          f"{'fill_%':>7} note")
    for r in rows:
        nfill = r["n_filled"] if r["n_filled"] >= 0 else -1
        ffrac = r["fill_frac_pct"] if r["fill_frac_pct"] >= 0 else -1.0
        print(
            f"{r['chgroup']:>7d} {r['n_grid']:>6d} {r['dec_deg']:>6.2f} "
            f"{nfill:>10d} {ffrac:>7.3f} {r.get('skipped', '')}"
        )

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
