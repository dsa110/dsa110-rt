#!/usr/bin/env python3
"""bench/preflight/inject_dm_width_grid.py — end-to-end DM x width injection
correctness grid (Phase C).

Loops a small grid of cold-plasma-dispersed, finite-width point-source
injections through the FULL real pipeline:

    pure-noise voltages
      -> OnlineInjector (voltage-domain, dispersed, finite width)
      -> corr-fast (cross-correlate + integrate + coarse-DM grid)   [Stage 1]
      -> corr->search bridge (densify + quantise -> CubeRingSlot)    [Stage 2]
      -> CubePipeline (fine dedispersion + imager)                   [Stage 2]
      -> DeterministicDetector (time-domain boxcars)                 [Stage 2]
      -> C1 candidate metadata + notebook-viewable NPZ cube

and, for each cell, compares the recovered candidate against injected truth
(position, fine-DM, boxcar width, peak time, SNR) and counts noise-only
false positives.

The single-cell driver lives in
``bench.preflight._inject_search_driver.run_search_driver`` (Stage 2). This
script is just the grid orchestration + the automated correctness scorecard;
it does not re-implement any pipeline physics.

Grid (default):
    DM    = 150, 500, 1000, 2500 pc/cm^3   (coarse-DM owners 0, 1, 2, 7)
    width = 1, 4, 16 ms                      (~1/4/16 search samples -> b1/b4/b16)

t_int_search = 1048.576 us, so width_ms ~= width_samples; the nearest
power-of-two boxcar (b1/b2/.../b64) is the expected detector kernel.

Run (on a CUDA host, e.g. n01):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:bench:. \\
      python -m bench.preflight.inject_dm_width_grid \\
        --out-dir /tmp/inject_grid \\
        --fluence-jy-ms 20000 \\
        --l-rad 0.004 --m-rad -0.002

Outputs (under ``--out-dir``):
    grid_summary.csv     one row per (DM, width) cell: injected vs recovered
    grid_summary.json    same, machine-readable, plus per-cell paths
    <cell>/...           per-cell C1 csv/ndjson + NPZ cube (from the driver)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger("inject_dm_width_grid")

# Production search sample period (us). DM-plan metadata t_int_search_us.
T_INT_SEARCH_US: float = 1048.576

# Detector time-domain boxcar widths (search samples).
BOXCARS = (1, 2, 4, 8, 16, 32, 64)

# DM -> coarse-DM owner index, from the v2 DM plan coarse centroids
# [258.7, 576.4, 894.5, 1213.2, 1532.9, 1853.8, 2176.0, 2499.9].
# Each search GPU half owns one coarse index and searches its 34 fine DMs.
DEFAULT_DM_OWNERS: tuple[tuple[float, int], ...] = (
    (150.0, 0),
    (500.0, 1),
    (1000.0, 2),
    (2500.0, 7),
)

DEFAULT_WIDTHS_MS: tuple[float, ...] = (1.0, 4.0, 16.0)


def width_ms_to_samples(width_ms: float) -> float:
    """Search-sample width for a given millisecond width."""
    return width_ms * 1000.0 / T_INT_SEARCH_US


def nearest_boxcar(width_samples: float) -> int:
    """Nearest power-of-two detector boxcar to a (fractional) sample width."""
    return min(BOXCARS, key=lambda b: abs(math.log2(b) - math.log2(max(width_samples, 1e-9))))


def n_blocks_for_dm(dm_pc_cm3: float) -> int:
    """Corr-fast block count needed so the dedispersion window fits.

    ``resolved_hi = n_blocks * 128`` tiles; high-DM bursts sweep a large
    dispersive delay that pushes the window start negative at n_blocks=8
    (verified: DM=500 fits in 8; DM=2500 needs ~20). Scales ~linearly with DM.
    """
    if dm_pc_cm3 <= 700.0:
        return 8
    if dm_pc_cm3 <= 1400.0:
        return 12
    return 20


@dataclass
class CellResult:
    """One (DM, width) grid cell: injected truth + recovered candidate."""

    # --- injected truth ---
    dm_inject_pc_cm3: float
    owner_idx: int
    width_ms: float
    width_samples_inject: float
    expected_boxcar: int
    l_rad_inject: float
    m_rad_inject: float
    fluence_jy_ms: float

    # --- recovered (best-matching candidate), None if nothing matched ---
    recovered: bool
    dm_recovered_pc_cm3: Optional[float] = None
    fine_dm_idx_recovered: Optional[int] = None
    width_samples_recovered: Optional[int] = None
    snr_recovered: Optional[float] = None
    l_pix_recovered: Optional[int] = None
    m_pix_recovered: Optional[int] = None
    event_specnum_recovered: Optional[int] = None

    # --- truth-vs-recovered deltas / expectations ---
    peak_row_truth: Optional[int] = None
    peak_col_truth: Optional[int] = None
    expected_t_in_cube: Optional[int] = None
    pixel_offset_cells: Optional[float] = None
    n_candidates: int = 0
    noise_only_fp_count: Optional[int] = None

    # --- artefacts ---
    npz_path: Optional[str] = None
    candidates_csv: Optional[str] = None

    # --- per-cell pass/fail ---
    passed: bool = False
    notes: str = ""


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/key getter (driver returns a dict or object)."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _score_cell(
    res: Any,
    *,
    dm_target: float,
    owner_idx: int,
    width_ms: float,
    l_rad: float,
    m_rad: float,
    fluence: float,
    pixel_tol_cells: float,
    snr_min: float,
) -> CellResult:
    """Build a CellResult by matching the best candidate to injected truth."""
    w_samp = width_ms_to_samples(width_ms)
    exp_box = nearest_boxcar(w_samp)

    diag = _get(res, "diag", default={}) or {}
    peak_row = _get(diag, "cpu_combine_peak_row", "peak_row")
    peak_col = _get(diag, "cpu_combine_peak_col", "peak_col")
    exp_t = _get(diag, "expected_t_in_cube", "expected_t")

    cell = CellResult(
        dm_inject_pc_cm3=float(dm_target),
        owner_idx=int(owner_idx),
        width_ms=float(width_ms),
        width_samples_inject=float(w_samp),
        expected_boxcar=int(exp_box),
        l_rad_inject=float(l_rad),
        m_rad_inject=float(m_rad),
        fluence_jy_ms=float(fluence),
        recovered=False,
        peak_row_truth=None if peak_row is None else int(peak_row),
        peak_col_truth=None if peak_col is None else int(peak_col),
        expected_t_in_cube=None if exp_t is None else int(exp_t),
        noise_only_fp_count=_get(res, "noise_only_fp_count"),
        npz_path=_get(res, "npz_path"),
        candidates_csv=_get(res, "candidates_csv"),
    )

    cands = _get(res, "candidates", default=[]) or []
    cell.n_candidates = len(cands)
    if not cands:
        cell.notes = "no candidates emitted"
        return cell

    # Truth pixel for matching: prefer the CPU-combine diagnostic peak
    # (ground truth pre-detector); fall back to image center for a
    # zenith injection.
    truth_row = cell.peak_row_truth
    truth_col = cell.peak_col_truth

    def cand_field(c: Any, *names: str) -> Any:
        return _get(c, *names)

    def pixel_offset(c: Any) -> float:
        if truth_row is None or truth_col is None:
            return 0.0
        # detector cand.l = grid row, cand.m = grid col (Stage-2 caveat)
        r = cand_field(c, "l", "l_pix", "row")
        col = cand_field(c, "m", "m_pix", "col")
        if r is None or col is None:
            return float("inf")
        return math.hypot(float(r) - truth_row, float(col) - truth_col)

    # Best candidate = highest SNR among those within the pixel window;
    # if none within window, take the global highest-SNR for reporting.
    in_window = [c for c in cands if pixel_offset(c) <= pixel_tol_cells]
    pool = in_window if in_window else cands
    best = max(pool, key=lambda c: float(cand_field(c, "snr") or 0.0))

    cell.recovered = bool(in_window)
    cell.snr_recovered = float(cand_field(best, "snr") or 0.0)
    cell.dm_recovered_pc_cm3 = _maybe_float(cand_field(best, "dm_fine", "dm_pc_cc"))
    cell.fine_dm_idx_recovered = _maybe_int(cand_field(best, "fine_dm_idx"))
    cell.width_samples_recovered = _maybe_int(cand_field(best, "width_samples"))
    cell.l_pix_recovered = _maybe_int(cand_field(best, "l", "l_pix", "row"))
    cell.m_pix_recovered = _maybe_int(cand_field(best, "m", "m_pix", "col"))
    cell.event_specnum_recovered = _maybe_int(cand_field(best, "event_specnum"))
    cell.pixel_offset_cells = pixel_offset(best)

    # Per-cell pass: a candidate within the pixel window, DM within ~1
    # fine step (15% tol), boxcar within one octave, SNR over floor.
    dm_ok = (
        cell.dm_recovered_pc_cm3 is not None
        and abs(cell.dm_recovered_pc_cm3 - dm_target) <= 0.15 * dm_target + 1.0
    )
    box_ok = (
        cell.width_samples_recovered is not None
        and abs(math.log2(max(cell.width_samples_recovered, 1)) - math.log2(exp_box)) <= 1.001
    )
    snr_ok = cell.snr_recovered is not None and cell.snr_recovered >= snr_min
    cell.passed = bool(cell.recovered and dm_ok and box_ok and snr_ok)
    notes = []
    if not cell.recovered:
        notes.append("no candidate within pixel window")
    if not dm_ok:
        notes.append("DM out of tol")
    if not box_ok:
        notes.append("boxcar >1 octave off")
    if not snr_ok:
        notes.append("SNR below floor")
    cell.notes = "; ".join(notes) if notes else "ok"
    return cell


def _maybe_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _maybe_int(x: Any) -> Optional[int]:
    try:
        return None if x is None else int(round(float(x)))
    except (TypeError, ValueError):
        return None


def run_grid(args: argparse.Namespace) -> int:
    # Imported lazily so --help works without torch/cuda.
    from bench.preflight._inject_search_driver import run_search_driver

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_owners = DEFAULT_DM_OWNERS
    if args.dms:
        want = {int(round(d)) for d in args.dms}
        dm_owners = tuple(t for t in DEFAULT_DM_OWNERS if int(round(t[0])) in want)
        if not dm_owners:
            raise SystemExit(f"--dms {args.dms} matched none of {DEFAULT_DM_OWNERS}")
    widths = tuple(args.widths_ms) if args.widths_ms else DEFAULT_WIDTHS_MS

    cells: list[CellResult] = []
    for dm_target, owner_idx in dm_owners:
        for width_ms in widths:
            cell_tag = f"dm{int(dm_target)}_w{width_ms:g}ms"
            cell_dir = out_dir / cell_tag
            n_blocks = int(args.n_blocks) if args.n_blocks else n_blocks_for_dm(dm_target)
            LOG.info("=== cell %s (owner %d, n_blocks=%d) ===", cell_tag, owner_idx, n_blocks)
            try:
                res = run_search_driver(
                    owner_idx=int(owner_idx),
                    dm_pc_cm3=None,
                    dm_target=float(dm_target),
                    l_rad=float(args.l_rad),
                    m_rad=float(args.m_rad),
                    width_ms=float(width_ms),
                    fluence_jy_ms=float(args.fluence_jy_ms),
                    t_det=int(args.t_det),
                    n_grid=int(args.n_grid),
                    n_fdm=None,
                    n_blocks=int(n_blocks),
                    chan_sum_factor=int(args.chan_sum_factor),
                    device=str(args.device),
                    n_burnin=int(args.n_burnin),
                    threshold_sigma=float(args.threshold_sigma),
                    out_dir=str(cell_dir),
                    reuse_corr=False,
                    audit_fp32=True,
                    zero_dm_filter=True,
                    run_noise_only=bool(args.noise_only),
                )
            except Exception as exc:  # noqa: BLE001 - record + continue grid
                LOG.exception("cell %s FAILED to run", cell_tag)
                cell = CellResult(
                    dm_inject_pc_cm3=float(dm_target), owner_idx=int(owner_idx),
                    width_ms=float(width_ms),
                    width_samples_inject=width_ms_to_samples(width_ms),
                    expected_boxcar=nearest_boxcar(width_ms_to_samples(width_ms)),
                    l_rad_inject=float(args.l_rad), m_rad_inject=float(args.m_rad),
                    fluence_jy_ms=float(args.fluence_jy_ms),
                    recovered=False, notes=f"driver raised: {exc!r}",
                )
                cells.append(cell)
                continue

            cell = _score_cell(
                res,
                dm_target=float(dm_target),
                owner_idx=int(owner_idx),
                width_ms=float(width_ms),
                l_rad=float(args.l_rad),
                m_rad=float(args.m_rad),
                fluence=float(args.fluence_jy_ms),
                pixel_tol_cells=float(args.pixel_tol_cells),
                snr_min=float(args.snr_min),
            )
            cells.append(cell)
            LOG.info(
                "cell %s: recovered=%s dm=%s box=%s snr=%s notes=%s",
                cell_tag, cell.recovered, cell.dm_recovered_pc_cm3,
                cell.width_samples_recovered, cell.snr_recovered, cell.notes,
            )

    _write_summary(out_dir, cells)
    _print_scorecard(cells)

    n_pass = sum(1 for c in cells if c.passed)
    LOG.info("grid complete: %d/%d cells passed", n_pass, len(cells))
    return 0 if n_pass == len(cells) else 1


def _write_summary(out_dir: Path, cells: list[CellResult]) -> None:
    rows = [asdict(c) for c in cells]
    (out_dir / "grid_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )
    if rows:
        with (out_dir / "grid_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    LOG.info("wrote %s", out_dir / "grid_summary.csv")


def _print_scorecard(cells: list[CellResult]) -> None:
    print("\n=== DM x width injection correctness scorecard ===")
    hdr = (
        f"{'DM':>6} {'w_ms':>5} {'box':>4} {'rec_box':>7} "
        f"{'dm_rec':>8} {'snr':>8} {'dpix':>5} {'fp':>3} {'pass':>5}  notes"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        print(
            f"{c.dm_inject_pc_cm3:>6.0f} {c.width_ms:>5.1f} "
            f"{c.expected_boxcar:>4d} "
            f"{('-' if c.width_samples_recovered is None else c.width_samples_recovered):>7} "
            f"{('-' if c.dm_recovered_pc_cm3 is None else f'{c.dm_recovered_pc_cm3:.0f}'):>8} "
            f"{('-' if c.snr_recovered is None else f'{c.snr_recovered:.1f}'):>8} "
            f"{('-' if c.pixel_offset_cells is None else f'{c.pixel_offset_cells:.1f}'):>5} "
            f"{('-' if c.noise_only_fp_count is None else c.noise_only_fp_count):>3} "
            f"{'PASS' if c.passed else 'FAIL':>5}  {c.notes}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="/tmp/inject_grid")
    p.add_argument("--fluence-jy-ms", type=float, default=20000.0)
    p.add_argument("--l-rad", type=float, default=0.004)
    p.add_argument("--m-rad", type=float, default=-0.002)
    p.add_argument("--widths-ms", type=float, nargs="*", default=list(DEFAULT_WIDTHS_MS))
    p.add_argument("--dms", type=float, nargs="*", default=None,
                   help="subset of DMs to run (default: 150 500 1000 2500)")
    p.add_argument("--t-det", type=int, default=192)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=0,
                   help="corr-fast blocks; 0 (default) => auto-scale with DM")
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--n-burnin", type=int, default=8)
    p.add_argument("--threshold-sigma", type=float, default=8.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pixel-tol-cells", type=float, default=4.0)
    p.add_argument("--snr-min", type=float, default=12.0)
    p.add_argument("--no-noise-only", dest="noise_only", action="store_false")
    p.set_defaults(noise_only=True)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    return run_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
