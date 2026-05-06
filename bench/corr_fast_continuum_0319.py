"""bench/corr_fast_continuum_0319.py — M3 chunk 5: 0319+415 continuum imager.

Drives the chunk-4 :mod:`dsart.services.corr_fast_integration` pipeline
across all 16 chgroups (sb00..sb15, minus sb12 — the M2 0319 fixture
data gap, see ``PARALLEL_AGENTS.md`` §5) on the real 0319+415
voltage-dump fixture, summing per-chgroup dirty images into a 16-chgroup
combined image and reporting the peak vs the predicted ``(l, m)`` of
0319+415.

CLI::

    python -m bench.corr_fast_continuum_0319 \\
        --voltage-root /home/ubuntu/data/voltages/0319 \\
        --n-blocks 4 \\
        --t-int-fast-native 4096 \\
        --n-grid 256 \\
        --report-dir bench/reports/<UTC>/M3-continuum-0319/

Per-chgroup, per-block compute time on h01 GPU 0:
* unpack int4    ~25 ms
* cal apply       ~3 ms
* fast-corr GEMM  ~10 ms
* Stokes I + grid ~5 ms
* iFFT + image    ~3 ms
Total ~50 ms per block; ~50 * 4 * 15 = 3 s of GPU time at 4 blocks/sb.

Outputs (under ``--report-dir``):

* ``report.html``                          — narrative + dirty-image links
* ``dirty_image_combined.png``             — 16-chgroup-summed dirty image
                                              with predicted (l, m) marked
* ``per_chgroup/dirty_image_chgroup<N>.png`` — per-chgroup dirty image
* ``summary.json``                         — JSON summary (PASS/FAIL gate)

PASS criterion: ``peak_offset_cells <= 4`` (peak within 4 cells of
predicted ``(l, m)`` of 0319+415).

Per the brief, dirty images are summed PIXEL-WISE across chgroups (each
chgroup has a slightly different per-pixel ``(l, m)`` scale due to the
lambda-uniform gridder's per-chgroup ``cell_lambda``; chunk 7 will
properly resample onto a common (l, m) grid). The combined image's peak
is reported in chgroup-0's (l, m) frame for the offset comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from dsart.common.constants import (
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
)
from dsart.services.slow_corr_kernel import NPACKETS_PER_BLOCK
from dsart.services.corr_fast_kernel import NTIMES_PER_PACKET

# bench helper
from bench._corr_fast_replay import (
    ReplayDefaults,
    accumulate_chgroup_grids,
    compute_chgroup_cell_lambda,
    dirty_image_from_dense_grid,
    lm_to_pixel,
    replay_chgroup,
    sparse_to_dense_grid,
)
# Reuse the M2 _compute_expected_lm helper (per brief §7).
from bench.run_0319_pipeline import (
    _compute_expected_lm,
    PHI_LAT_OVRO_DEG,
)


LOG = logging.getLogger("bench.corr_fast_continuum_0319")


# 0319+415 default source coords (mirrors run_0319_pipeline defaults).
SRC_RA_DEG_DEFAULT = 49.9506667
SRC_DEC_DEG_DEFAULT = 41.51169444
SRC_MJD_DEFAULT = 61108.99867338988

ALL_SBS_DEFAULT = [f"{n:02d}" for n in range(16) if n != 12]


@dataclass
class ChgroupResult:
    sb: str
    chgroup: int
    n_blocks_processed: int
    n_fast_vis_total: int
    n_filled: int
    cell_lambda: float
    image_path: str
    peak_pixel_lm: tuple[float, float]
    peak_offset_cells: int
    peak_value: float
    elapsed_s: float


def _parse_t2_json(path: Path) -> dict:
    with path.open() as f:
        doc = json.load(f)
    first = next(iter(doc.values()))
    return {
        "ra_deg": float(first["ra"]),
        "dec_deg": float(first["dec"]),
        "mjd": float(first["mjds"]),
        "specnum": int(first["specnum"]),
    }


def _save_image_png(
    image: np.ndarray,
    *,
    title: str,
    out_path: Path,
    expected_lm_pixel: tuple[int, int] | None = None,
    peak_lm_pixel: tuple[int, int] | None = None,
    cell_lambda: float | None = None,
) -> None:
    n_grid = image.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    extent_lm = None
    if cell_lambda is not None:
        half_lm = (n_grid // 2) / (n_grid * cell_lambda)
        extent_lm = (-half_lm, half_lm, -half_lm, half_lm)
    im = ax.imshow(
        image, origin="lower", cmap="viridis",
        extent=extent_lm,
    )
    ax.set_title(title)
    if extent_lm is None:
        ax.set_xlabel("col (l-axis pixel)")
        ax.set_ylabel("row (m-axis pixel)")
    else:
        ax.set_xlabel("l (rad)")
        ax.set_ylabel("m (rad)")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if expected_lm_pixel is not None and extent_lm is None:
        # Pixel-coord overlay
        ax.plot(
            expected_lm_pixel[1], expected_lm_pixel[0],
            "rx", markersize=14, markeredgewidth=2, label="predicted",
        )
    elif expected_lm_pixel is not None and cell_lambda is not None:
        half = n_grid // 2
        pixel_size_lm = 1.0 / (n_grid * cell_lambda)
        l_pred = (expected_lm_pixel[1] - half) * pixel_size_lm
        m_pred = (expected_lm_pixel[0] - half) * pixel_size_lm
        ax.plot(l_pred, m_pred, "rx", markersize=14, markeredgewidth=2,
                label="predicted")

    if peak_lm_pixel is not None and extent_lm is None:
        ax.plot(
            peak_lm_pixel[1], peak_lm_pixel[0],
            "y+", markersize=12, markeredgewidth=2, label="peak",
        )
    elif peak_lm_pixel is not None and cell_lambda is not None:
        half = n_grid // 2
        pixel_size_lm = 1.0 / (n_grid * cell_lambda)
        l_meas = (peak_lm_pixel[1] - half) * pixel_size_lm
        m_meas = (peak_lm_pixel[0] - half) * pixel_size_lm
        ax.plot(l_meas, m_meas, "y+", markersize=12, markeredgewidth=2,
                label="peak")

    if expected_lm_pixel is not None or peak_lm_pixel is not None:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _process_one_sb(
    *,
    sb: str,
    voltage_path: Path,
    cal_path: Path,
    cfg_template: ReplayDefaults,
    src: dict,
    n_blocks: int,
    device: torch.device,
    per_chgroup_dir: Path,
    expected_lm_rad: tuple[float, float],
) -> ChgroupResult:
    """Replay one sb's voltages and produce a per-chgroup dirty image + peak."""
    chgroup = int(sb)
    cfg = cfg_template.to_cfg(cal_path=cal_path)
    cfg.chgroup = chgroup                                            # FastIntegrationConfig is mutable

    t0 = time.monotonic()
    LOG.info("=== sb%s (chgroup=%d) ===", sb, chgroup)
    ctx, outputs = replay_chgroup(
        voltage_path, cal_path=cal_path, cfg=cfg,
        max_blocks=n_blocks, device=device,
    )
    if not outputs:
        raise RuntimeError(f"sb{sb}: no blocks processed")

    pattern = ctx.gridder.pattern
    n_filled = pattern.n_filled
    grids = accumulate_chgroup_grids(outputs, n_filled=n_filled)     # (n_fv_total, n_filled)

    # SUM across the time axis (continuum) — single combined sparse-COO frame.
    sum_sparse = grids.sum(dim=0, keepdim=True)                      # (1, n_filled)

    # Sparse → dense grid → dirty image.
    dense = sparse_to_dense_grid(
        sum_sparse, pattern.ix_row, pattern.ix_col, n_grid=cfg.n_grid,
    )                                                                # (1, N, N)
    img = dirty_image_from_dense_grid(dense)[0]                      # (N, N) float32
    img_np = img.cpu().numpy()

    from dsart.services.corr_fast_integration import load_antpos_from_cal_blob
    ap_e, ap_n, core_mask = load_antpos_from_cal_blob(cal_path)
    cell_lambda = compute_chgroup_cell_lambda(
        ap_e, ap_n, chgroup=chgroup, n_grid=cfg.n_grid,
        is_core_baseline_mask=core_mask,
    )

    # Predicted (l, m) → pixel
    half = cfg.n_grid // 2
    pred_row, pred_col = lm_to_pixel(
        expected_lm_rad[0], expected_lm_rad[1],
        n_grid=cfg.n_grid, cell_lambda=cell_lambda,
    )

    # Peak in the dirty image (interior — drop 8-pixel border to avoid the
    # well-known iFFT edge wraparound ridge — same convention as
    # tools/viz/common.find_image_peaks).
    edge_pad = 8
    interior = img_np[edge_pad:-edge_pad, edge_pad:-edge_pad]
    peak_flat = int(np.argmax(interior))
    peak_row = peak_flat // interior.shape[1] + edge_pad
    peak_col = peak_flat % interior.shape[1] + edge_pad
    peak_value = float(interior.flat[peak_flat])

    peak_offset = max(
        abs(int(peak_row) - int(pred_row)),
        abs(int(peak_col) - int(pred_col)),
    )

    LOG.info(
        "  sb%s: peak(row,col)=(%d,%d) predicted=(%d,%d) offset=%d cells "
        "value=%.3g cell_lambda=%.3g",
        sb, peak_row, peak_col, int(pred_row), int(pred_col),
        peak_offset, peak_value, cell_lambda,
    )

    image_path = per_chgroup_dir / f"dirty_image_chgroup{chgroup:02d}.png"
    _save_image_png(
        img_np,
        title=f"chgroup={chgroup} (sb{sb}); n_blocks={n_blocks}; "
              f"peak_offset={peak_offset} cells",
        out_path=image_path,
        expected_lm_pixel=(int(pred_row), int(pred_col)),
        peak_lm_pixel=(peak_row, peak_col),
        cell_lambda=cell_lambda,
    )

    elapsed = time.monotonic() - t0
    n_fv_total = grids.shape[0]
    return ChgroupResult(
        sb=sb,
        chgroup=chgroup,
        n_blocks_processed=len(outputs),
        n_fast_vis_total=int(n_fv_total),
        n_filled=int(n_filled),
        cell_lambda=float(cell_lambda),
        image_path=str(image_path),
        peak_pixel_lm=(float(peak_row), float(peak_col)),
        peak_offset_cells=int(peak_offset),
        peak_value=float(peak_value),
        elapsed_s=float(elapsed),
    ), img_np


def _write_report_html(
    out_path: Path,
    *,
    summary: dict,
    chgroup_results: list[ChgroupResult],
    combined_image_path: str,
) -> None:
    rows = "\n".join(
        f"<tr><td>{r.sb}</td><td>{r.chgroup}</td>"
        f"<td>{r.n_blocks_processed}</td>"
        f"<td>{r.n_filled}</td>"
        f"<td>{r.peak_pixel_lm[0]:.0f},{r.peak_pixel_lm[1]:.0f}</td>"
        f"<td>{r.peak_offset_cells}</td>"
        f"<td>{r.peak_value:.3g}</td>"
        f"<td>{r.elapsed_s:.1f}s</td>"
        f"<td><a href='{Path(r.image_path).name}'>per-chgroup img</a></td></tr>"
        for r in chgroup_results
    )
    pass_class = "pass" if summary["passed"] else "fail"
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>M3 chunk 5: 0319+415 continuum imager</title>
<style>
body {{ font-family: system-ui,sans-serif; max-width: 1100px; margin: 1em auto; padding: 0 1em }}
table {{ border-collapse: collapse; width: 100% }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 14px; text-align: left }}
th {{ background: #eee }}
.pass {{ color: green; font-weight: bold }}
.fail {{ color: red; font-weight: bold }}
img {{ max-width: 800px; display: block; margin: 1em 0 }}
.summary {{ background: #f4f4f4; padding: 1em; border-radius: 6px }}
</style>
<h1>M3 chunk 5: 0319+415 continuum imager</h1>
<div class='summary'>
<p><strong>UTC:</strong> {summary['utc_iso']}</p>
<p><strong>git SHA:</strong> {summary['git_sha']}</p>
<p><strong>Host:</strong> {summary['host']}</p>
<p><strong>n_blocks_per_chgroup:</strong> {summary['n_blocks_per_chgroup']}</p>
<p><strong>t_int_fast_native:</strong> {summary['t_int_fast_native']} (= {summary['t_int_fast_native']*NATIVE_SAMPLE_US:.3f} µs cadence)</p>
<p><strong>n_grid:</strong> {summary['n_grid']}</p>
<p><strong>Source:</strong> 0319+415 (RA={summary['src_ra_deg']:.4f} deg, Dec={summary['src_dec_deg']:.4f} deg)</p>
<p><strong>Predicted (l, m):</strong> ({summary['expected_lm']['l_rad']:.4g}, {summary['expected_lm']['m_rad']:.4g}) rad</p>
<p><strong>Combined peak (row, col):</strong> ({summary['combined_peak_row']}, {summary['combined_peak_col']})</p>
<p><strong>Combined peak offset:</strong> {summary['combined_peak_offset_cells']} cells (PASS gate ≤ 4)</p>
<p><strong>Result:</strong> <span class='{pass_class}'>{summary['stage'].upper()}</span></p>
</div>
<h2>16-chgroup-summed dirty image</h2>
<img src="{Path(combined_image_path).name}" />
<h2>Per-chgroup dirty images</h2>
<table>
<tr><th>sb</th><th>chgroup</th><th>n_blocks</th><th>n_filled</th>
<th>peak (row, col)</th><th>offset (cells)</th><th>peak val</th>
<th>elapsed</th><th>image</th></tr>
{rows}
</table>
<p>Note: dirty images are summed pixel-wise; each chgroup has its own
``cell_lambda`` so summed-pixel ``(l, m)`` is approximate (chunk 7 will
properly resample onto a common (l, m) grid).</p>
"""
    out_path.write_text(html)


def _short_git_sha(repo_root: Path) -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--voltage-root", type=Path,
                   default=Path("/home/ubuntu/data/voltages/0319"),
                   help="root containing voltages/ + cals/")
    p.add_argument("--n-blocks", type=int, default=4,
                   help="fada blocks per chgroup to integrate")
    p.add_argument("--t-int-fast-native", type=int, default=4096,
                   help="fast-corr integration depth (native samples per "
                        "fast-vis tile). Default 4096 = one fast-vis tile "
                        "per block (continuum).")
    p.add_argument("--n-grid", type=int, default=256,
                   help="image-plane grid side length")
    p.add_argument("--report-dir", type=Path, required=True,
                   help="output dir for HTML + PNGs + summary.json")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cuda:0 / cpu")
    p.add_argument("--src-json", type=Path, default=None,
                   help="optional T2_*.json override for source coords")
    p.add_argument("--src-ra-deg", type=float, default=SRC_RA_DEG_DEFAULT)
    p.add_argument("--src-dec-deg", type=float, default=SRC_DEC_DEG_DEFAULT)
    p.add_argument("--src-mjd", type=float, default=SRC_MJD_DEFAULT)
    p.add_argument("--cal-mode", default="phase_only",
                   choices=("phase_only", "full"))
    p.add_argument("--cal-pol-swap", action="store_true")
    p.add_argument("--sbs", default=",".join(ALL_SBS_DEFAULT),
                   help="comma-separated sb ids; default = 00..15 minus 12 "
                        "(known 0319 fixture data gap)")
    p.add_argument("--peak-offset-pass-cells", type=int, default=4,
                   help="PASS gate on max(|peak.row - pred.row|, "
                        "|peak.col - pred.col|) in the COMBINED image.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    per_chgroup_dir = args.report_dir / "per_chgroup"
    per_chgroup_dir.mkdir(parents=True, exist_ok=True)

    # Source meta (T2_0319bbb.json takes precedence over the CLI defaults).
    if args.src_json is not None:
        src = _parse_t2_json(args.src_json)
    else:
        # Try the canonical T2_*.json next to the voltages dir
        t2_default = args.voltage_root / "voltages" / "T2_0319bbb.json"
        if t2_default.is_file():
            src = _parse_t2_json(t2_default)
            LOG.info("loaded source coords from %s", t2_default)
        else:
            src = {
                "ra_deg": args.src_ra_deg,
                "dec_deg": args.src_dec_deg,
                "mjd": args.src_mjd,
                "specnum": 0,
            }

    LOG.info(
        "0319+415 source: RA=%.4f deg Dec=%.4f deg MJD=%.6f",
        src["ra_deg"], src["dec_deg"], src["mjd"],
    )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    LOG.info("device=%s", device)

    # Phase center for the predicted-(l, m) overlay: the F21 cal phases
    # at obs_dec = src_dec, so the source should land at (l, m) ≈ (0, 0)
    # at the meridian. The actual 0319 dump was made at HA=0 by definition
    # of the trigger; LST(refmjd) - src_ra_deg should be ≈ 0 (mod 360).
    expected_l, expected_m, ha_src_deg = _compute_expected_lm(
        src_ra_deg=src["ra_deg"],
        src_dec_deg=src["dec_deg"],
        refmjd=src["mjd"],
        phase_center_dec_deg=src["dec_deg"],                          # F21 phases here
    )
    LOG.info(
        "predicted (l, m) for 0319+415: (%.4g, %.4g) rad; HA_src=%.4f deg",
        expected_l, expected_m, ha_src_deg,
    )

    voltage_dir = args.voltage_root / "voltages"
    cals_dir = args.voltage_root / "cals"
    if not voltage_dir.is_dir():
        LOG.error("voltage dir not found: %s", voltage_dir)
        return 2
    if not cals_dir.is_dir():
        LOG.error("cals dir not found: %s", cals_dir)
        return 2

    sbs = [s.strip() for s in args.sbs.split(",") if s.strip()]
    LOG.info("processing sbs: %s", sbs)

    chgroup_results: list[ChgroupResult] = []
    combined_image: np.ndarray | None = None
    n_grid = args.n_grid

    for sb in sbs:
        voltage_path = voltage_dir / f"0319bbb_sb{sb}_data.out"
        cal_path = cals_dir / f"beamformer_weights_sb{sb}_0319+415.dat"
        if not voltage_path.is_file():
            LOG.warning("sb%s: %s not found; SKIPPING (data gap)", sb, voltage_path)
            continue
        if not cal_path.is_file():
            LOG.warning("sb%s: %s not found; SKIPPING (no cal)", sb, cal_path)
            continue

        cfg_template = ReplayDefaults(
            chgroup=int(sb),
            obs_dec_deg=src["dec_deg"],
            n_grid=n_grid,
            kernel_support=1,
            t_int_fast_native=args.t_int_fast_native,
            cal_mode=args.cal_mode,
            cal_pol_swap=args.cal_pol_swap,
            rfi_enabled=False,                                       # brief: continuum bench keeps RFI off
            static_sky_disabled=True,                                # brief: 0319 IS the static sky
        )

        try:
            result, img_np = _process_one_sb(
                sb=sb, voltage_path=voltage_path, cal_path=cal_path,
                cfg_template=cfg_template, src=src,
                n_blocks=args.n_blocks, device=device,
                per_chgroup_dir=per_chgroup_dir,
                expected_lm_rad=(expected_l, expected_m),
            )
        except Exception:
            LOG.exception("sb%s: FAILED", sb)
            continue
        chgroup_results.append(result)
        if combined_image is None:
            combined_image = img_np.copy()
        else:
            combined_image += img_np

    if combined_image is None or not chgroup_results:
        LOG.error("no chgroup results; aborting")
        return 3

    # Combined image peak — interpret in chgroup-0's (l, m) frame. The
    # cell_lambda is monotone in chgroup index (lower freq → larger
    # wavelength → smaller (u, v) → smaller cell_lambda → larger
    # pixel_size_lm), so chgroup-0's frame is the "tightest" (smallest
    # pixel_size_lm) and using it gives a CONSERVATIVE upper bound on
    # the offset-in-cells comparison. Per chunk 7, this is the
    # frame all per-chgroup images will be properly resampled to.
    chg0_cell_lambda = next(
        (r.cell_lambda for r in chgroup_results if r.chgroup == 0),
        chgroup_results[0].cell_lambda,
    )
    half = n_grid // 2
    pred_row, pred_col = lm_to_pixel(
        expected_l, expected_m,
        n_grid=n_grid, cell_lambda=chg0_cell_lambda,
    )

    edge_pad = 8
    interior_combined = combined_image[edge_pad:-edge_pad, edge_pad:-edge_pad]
    peak_flat = int(np.argmax(interior_combined))
    peak_row_combined = peak_flat // interior_combined.shape[1] + edge_pad
    peak_col_combined = peak_flat % interior_combined.shape[1] + edge_pad
    peak_value_combined = float(interior_combined.flat[peak_flat])

    combined_offset = max(
        abs(int(peak_row_combined) - int(pred_row)),
        abs(int(peak_col_combined) - int(pred_col)),
    )

    combined_image_path = args.report_dir / "dirty_image_combined.png"
    _save_image_png(
        combined_image,
        title=f"0319+415 — 16-chgroup-summed dirty image\n"
              f"n_blocks={args.n_blocks}/chgroup; peak_offset={combined_offset} cells",
        out_path=combined_image_path,
        expected_lm_pixel=(int(pred_row), int(pred_col)),
        peak_lm_pixel=(int(peak_row_combined), int(peak_col_combined)),
        cell_lambda=float(chg0_cell_lambda),
    )

    passed = combined_offset <= args.peak_offset_pass_cells
    repo_root = Path(__file__).resolve().parents[1]
    summary = {
        "milestone": "M3",
        "chunk": "chunk_5_voltage_fixture_continuum",
        "stage": "PASS" if passed else "FAIL",
        "passed": passed,
        "host": socket.gethostname(),
        "utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _short_git_sha(repo_root),
        "n_blocks_per_chgroup": args.n_blocks,
        "t_int_fast_native": args.t_int_fast_native,
        "n_grid": n_grid,
        "src_ra_deg": float(src["ra_deg"]),
        "src_dec_deg": float(src["dec_deg"]),
        "src_mjd": float(src["mjd"]),
        "expected_lm": {
            "l_rad": float(expected_l),
            "m_rad": float(expected_m),
            "ha_src_deg": float(ha_src_deg),
            "phase_center_mode": "source_dec (F21)",
        },
        "combined_peak_row": int(peak_row_combined),
        "combined_peak_col": int(peak_col_combined),
        "combined_peak_offset_cells": int(combined_offset),
        "combined_peak_value": float(peak_value_combined),
        "predicted_pixel_row": int(pred_row),
        "predicted_pixel_col": int(pred_col),
        "chg0_cell_lambda": float(chg0_cell_lambda),
        "n_chgroups_processed": len(chgroup_results),
        "n_chgroups_skipped": len(sbs) - len(chgroup_results),
        "per_chgroup": [asdict(r) for r in chgroup_results],
        "peak_offset_pass_gate_cells": args.peak_offset_pass_cells,
    }
    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote %s", summary_path)

    _write_report_html(
        args.report_dir / "report.html",
        summary=summary, chgroup_results=chgroup_results,
        combined_image_path=str(combined_image_path),
    )
    LOG.info("wrote %s", args.report_dir / "report.html")

    print(json.dumps({
        "stage": summary["stage"],
        "combined_peak_offset_cells": summary["combined_peak_offset_cells"],
        "combined_peak_value": summary["combined_peak_value"],
        "n_chgroups_processed": summary["n_chgroups_processed"],
        "report_dir": str(args.report_dir),
    }, indent=2))

    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
