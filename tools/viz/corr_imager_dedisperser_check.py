"""tools/viz/corr_imager_dedisperser_check.py — M2/M3 voltage-fixture viz CLI.

Plan §8 line 2172 (M2 continuum slow-corr check) + §8 lines 2197-2206
(M3 fast-corr continuum + burst dedispersion checks).

M2-supported surface (per F9 in M2_PLAN_FIXES.md — minimal):

    python -m tools.viz.corr_imager_dedisperser_check \\
        --mode continuum \\
        --check slow_corr \\
        --uvh5 <path> \\
        --voltage-run-id <id> \\
        --out bench/reports/<UTC>/<id>/M2/

OR (synthetic / smoke-test mode, no UVH5 / no meridian_fringestop):

    python -m tools.viz.corr_imager_dedisperser_check \\
        --mode continuum \\
        --check slow_corr \\
        --bada <bada_capture.bin> \\
        --out bench/reports/<UTC>/<id>/M2/

Outputs (per plan §8 line 2173):
  * `slow_corr_check.png` — UVH5 / bada-derived dirty image with
    manifest's `continuum_sources` overplotted at expected (l, m) when
    supplied; otherwise reports the brightest 5 image-plane peaks for
    operator inspection.
  * `report.html` — peak-vs-expected table + metadata. NO PASS/FAIL banner
    (operator inspects + signs off via D11 marker file).

Future (deferred — F9):
  * `--mode burst` (dedispersion sweep + filterbank pixel)
  * `--check fast_corr` (fast-vis grid + static-sky-subtract)
  * `--chgroup all` (16-chgroup stage-2 alignment)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.viz.common import (  # noqa: E402
    dirty_image_from_grid,
    expected_sources_from_manifest,
    find_image_peaks,
    grid_uv_natural,
    load_bada_capture,
    load_uvh5,
    render_dirty_image_png,
    render_report_html,
)


FIXTURE_ROOT_DEFAULT = "/home/ubuntu/data/voltage_fixtures"


def _maybe_load_manifest(run_id: str | None) -> dict | None:
    if not run_id:
        return None
    root = Path(os.environ.get("DSART_VOLTAGE_FIXTURE_ROOT", FIXTURE_ROOT_DEFAULT))
    manifest_path = root / run_id / "manifest.yaml"
    if not manifest_path.is_file():
        print(f"WARNING: manifest {manifest_path} not found; "
              f"skipping expected-source overlay", file=sys.stderr)
        return None
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required to load fixture manifests") from exc
    return yaml.safe_load(manifest_path.read_text())


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=("continuum",), default="continuum",
                    help="Image mode (M2 supports continuum only; "
                         "burst/etc. defer to M3)")
    ap.add_argument("--check", choices=("slow_corr",), default="slow_corr",
                    help="Pipeline check (M2 supports slow_corr only)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--uvh5", help="UVH5 file produced by meridian_fringestop")
    src.add_argument("--bada", help="Raw bada_capture.bin (synth/smoke mode)")
    ap.add_argument("--voltage-run-id", help="Manifest run-id for overlay")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--n-grid", type=int, default=256, help="Image grid size")
    ap.add_argument("--fov-rad", type=float, default=0.5,
                    help="Field of view in radians")
    ap.add_argument("--pol", type=int, default=0,
                    help="Pol index (0=XX, 1=YY)")
    ap.add_argument("--n-top", type=int, default=5,
                    help="Number of image-plane peaks to report")
    args = ap.parse_args(argv)

    if args.mode != "continuum" or args.check != "slow_corr":
        raise SystemExit(
            f"--mode {args.mode!r} --check {args.check!r} not supported in M2; "
            f"see plan §8 lines 2197-2206 (M3) and F9 in M2_PLAN_FIXES."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load visibility cube --------------------------------------------
    if args.uvh5:
        cube = load_uvh5(args.uvh5)
        source_kind = "UVH5 (operator-facing)"
        source_path = args.uvh5
    else:
        cube = load_bada_capture(args.bada,
                                  source_name=Path(args.bada).stem)
        source_kind = "bada_capture (synth/smoke)"
        source_path = args.bada

    print(f"[viz] loaded {source_kind} from {source_path}: "
          f"vis shape {cube.vis.shape}, freqs[0]={cube.freqs_Hz[0]/1e9:.3f} GHz, "
          f"freqs[-1]={cube.freqs_Hz[-1]/1e9:.3f} GHz, "
          f"|uvw|.max()={float((cube.uvw_m**2).sum(axis=1).max()**0.5):.2f} m",
          flush=True)

    # ---- Load manifest (optional) ---------------------------------------
    manifest = _maybe_load_manifest(args.voltage_run_id)
    expected = expected_sources_from_manifest(manifest)
    if expected:
        print(f"[viz] manifest expected sources: "
              f"{[(n, f'l={l:+.4f}', f'm={m:+.4f}') for n, l, m in expected]}",
              flush=True)

    # ---- Grid + iFFT ----------------------------------------------------
    print(f"[viz] gridding (n_grid={args.n_grid}, fov_rad={args.fov_rad}, "
          f"pol={args.pol}) ...", flush=True)
    grid, weight = grid_uv_natural(
        cube.vis, cube.uvw_m, cube.freqs_Hz,
        n_grid=args.n_grid, fov_rad=args.fov_rad, pol=args.pol,
    )
    n_filled = int((weight > 0).sum())
    print(f"[viz] grid: {n_filled} cells filled "
          f"of {args.n_grid * args.n_grid} "
          f"({100 * n_filled / (args.n_grid ** 2):.1f}%)", flush=True)

    image = dirty_image_from_grid(grid)
    peaks = find_image_peaks(image, fov_rad=args.fov_rad, n_top=args.n_top)

    print(f"[viz] top {args.n_top} peaks:", flush=True)
    for p in peaks:
        print(f"  rank {p.rank}: (l={p.l_rad:+.5f}, m={p.m_rad:+.5f}) "
              f"flux={p.flux_image_units:.3g} SNR={p.snr_image_plane:.1f}",
              flush=True)

    # ---- Render PNG + HTML ----------------------------------------------
    png_path = out_dir / "slow_corr_check.png"
    title = (
        f"Slow corr continuum check — source={cube.source_name} "
        f"pol={args.pol} fov={args.fov_rad:.3f} rad"
    )
    render_dirty_image_png(
        image, args.fov_rad,
        title=title, out_path=png_path,
        peaks=peaks, expected_sources=expected,
    )
    print(f"[viz] wrote {png_path}", flush=True)

    metadata = {
        "tool": "tools.viz.corr_imager_dedisperser_check",
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "check": args.check,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "voltage_run_id": args.voltage_run_id or "(none)",
        "n_grid": args.n_grid,
        "fov_rad": args.fov_rad,
        "pol": args.pol,
        "vis_shape": str(tuple(cube.vis.shape)),
        "freqs_GHz_lohi": f"{cube.freqs_Hz[-1]/1e9:.4f}–{cube.freqs_Hz[0]/1e9:.4f}",
        "n_grid_cells_filled": n_filled,
        "n_grid_cells_total": args.n_grid * args.n_grid,
    }

    report_path = render_report_html(
        out_dir, title=title, metadata=metadata,
        peaks=peaks, expected_sources=expected,
    )
    print(f"[viz] wrote {report_path}", flush=True)

    # Side-channel JSON for programmatic consumers (DoD orchestrator).
    obs_json = out_dir / "observed_peaks.json"
    obs_json.write_text(json.dumps({
        "metadata": metadata,
        "peaks": [
            {
                "rank": p.rank,
                "l_rad": p.l_rad,
                "m_rad": p.m_rad,
                "flux": p.flux_image_units,
                "snr": p.snr_image_plane,
            }
            for p in peaks
        ],
        "expected": [
            {"name": n, "l_rad": l, "m_rad": m}
            for n, l, m in expected
        ],
    }, indent=2))
    print(f"[viz] wrote {obs_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
