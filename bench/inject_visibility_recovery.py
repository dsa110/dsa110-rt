#!/usr/bin/env python3
"""bench/inject_visibility_recovery.py — M3 chunk-3d operator-inspect bench.

The chunk-3d acceptance proof: the voltage-domain online injector
(``src/dsart/inject/online.py``) produces a per-(ant, ch, t, pol)
contribution which, after a per-baseline ``V_ij = conj(E_i) · E_j``
GEMM (F18, with i = lower antenna, j = higher) + natural-weighted
gridding (M2's ``tools/viz/common.py::grid_uv_natural``) + iFFT2,
recovers a peak at the injected ``(l, m)`` within ≤ 1 grid cell at
``N_grid = 256``.

This is **deliberately scoped** to the injector + the per-baseline
GEMM + the gridder + the iFFT — no PSRDADA, no
``corr_fast_compute`` service, no transport. Bench independence is
the chunk-3d isolation discipline (PARALLEL_AGENTS.md §1) — the
parent M3 agent is still building the spine; this bench gates the
injector on the spine's *expected* downstream behaviour without
depending on the spine itself.

Outputs: an HTML report + dirty-image PNG written to
``bench/reports/<UTC>/<run_id>/M3-injector/``. The report carries
the injected ``(l, m, DM, fluence, width)``, the recovered peak
position, the |Δ| in grid cells, and a small metadata table for
operator inspection. Per the M2 acceptance pattern (D11), no
PASS/FAIL banner is rendered — operator approval is out of band.

Usage::

    python -m bench.inject_visibility_recovery \\
        [--l 0.05] [--m 0.0] [--dm 200] [--width 8] [--fluence 10] \\
        [--n-grid 256] [--fov-rad 0.5] \\
        [--noise-sigma 1e-3] [--seed 42] \\
        [--out-dir bench/reports/<UTC>/<run_id>/M3-injector]

Run-time on a CPU-only laptop: ~30 s (the GEMM is the bulk; uses a
single block at ``NPACKETS_PER_BLOCK = 2048`` micro-time samples
× 96 ants × 384 chans).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))                            # for tools.viz

import torch  # noqa: E402

from dsart.common.constants import (  # noqa: E402
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    SPEED_OF_LIGHT_M_S,
    freq_GHz,
)
from dsart.inject.online import (  # noqa: E402
    InjectionConfig,
    OnlineInjector,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    upper_tri_indices,
)
from tools.viz.common import (  # noqa: E402
    dirty_image_from_grid,
    find_image_peaks,
    grid_uv_natural,
)


# ---------------------------------------------------------------------------
# Synthetic geometry (DSA-110-like 96-ant 2D layout)
# ---------------------------------------------------------------------------


def _antpos_synth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic 96-ant layout (~150 m E-W, ~50 m N-S, U=0)."""
    rng = np.random.default_rng(0xCAFE)
    e = np.linspace(-75.0, +75.0, NANTS) + rng.normal(0, 0.5, NANTS)
    n = np.linspace(-25.0, +25.0, NANTS) + rng.normal(0, 0.3, NANTS)
    u = np.zeros(NANTS, dtype=np.float64)
    return e, n, u


# ---------------------------------------------------------------------------
# Bench core
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    """Run the bench and return a result-summary dict.

    Pipeline:
      1. Build a noise voltage tensor in M2's GEMM layout
         ``(NCHAN, NTIMES, NPOL, NPACKETS, NANTS) fp32``.
      2. Inject the source via :class:`OnlineInjector` at the centre
         of the block.
      3. Time-integrate over the full block (sum over (NTIMES,
         NPACKETS) at each (ch, pol, ant)) → ``(NCHAN, NPOL, NANTS)``
         per-ant voltage.
      4. Per-baseline ``V_ij = conj(E_i) · E_j`` for i = lower, j =
         higher (F18) using ``upper_tri_indices``.
      5. Grid via ``grid_uv_natural`` (its built-in ``-(u, v)``
         negation matches the F20 fix; the iFFT then recovers
         TMS-canonical ``(+l, +m)`` axes).
      6. ``Re(iFFT2(grid))`` dirty image; locate brightest peak.
      7. Compare peak ``(l, m)`` to injected ``(l, m)`` — the bench's
         success criterion is ``|Δ_l|, |Δ_m| ≤ cell_size`` at
         ``N_grid = 256``.

    Returns
    -------
    dict
        Result summary (also rendered as HTML / written to disk by
        :func:`render_report`).
    """
    print(f"[inject_visibility_recovery] starting on "
          f"{torch.cuda.is_available() and 'CUDA' or 'CPU'}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Build noise voltage ----
    rng = np.random.default_rng(args.seed)
    e_ant, n_ant, u_ant = _antpos_synth()
    shape = (
        NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
        NPACKETS_PER_BLOCK, NANTS,
    )
    noise_real = (rng.normal(0, args.noise_sigma, shape)
                  ).astype(np.float32)
    noise_imag = (rng.normal(0, args.noise_sigma, shape)
                  ).astype(np.float32)
    real_v = torch.from_numpy(noise_real).to(device)
    imag_v = torch.from_numpy(noise_imag).to(device)
    print(f"[inject_visibility_recovery] noise voltage: shape={shape} "
          f"sigma={args.noise_sigma:.3e}", flush=True)

    # ---- 2. Inject source ----
    injector = OnlineInjector(
        antpos_e=e_ant, antpos_n=n_ant, chgroup=args.chgroup,
        device=device, dtype=torch.float32, antpos_u=u_ant,
    )
    cfg = InjectionConfig(
        inj_id=f"bench_{uuid.uuid4().hex[:8]}",
        l_rad=args.l, m_rad=args.m,
        dm_pc_cm3=args.dm,
        fluence_jy_ms=args.fluence,
        width_samples=args.width,
        profile=args.profile,
        apply_at_specnum=args.apply_at_specnum,
    )
    injector.add_pending(cfg)
    log = injector.apply_block(real_v, imag_v, block_specnum_start=0)
    if not log["active_inj_ids"]:
        raise RuntimeError(
            f"injector deposited no contribution; log={log}"
        )
    print(f"[inject_visibility_recovery] injection log: {log}", flush=True)

    # ---- 3. Time-integrate per-(ant, ch, pol) ----
    # Sum over (NTIMES, NPACKETS) → (NCHAN, NPOL, NANTS).
    # Equivalent to a continuum integration of the per-ant voltage.
    e_ant_ch_pol = (
        real_v.sum(dim=(1, 3)) + 1j * imag_v.sum(dim=(1, 3))
    )                                                          # (NCHAN, NPOL, NANTS)
    e_ant_ch_pol = e_ant_ch_pol.cpu().numpy().astype(np.complex128)
    del real_v, imag_v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 4. Per-baseline V_ij = conj(E_i) · E_j ----
    # i = lower antenna (b_idx), j = higher (a_idx). NOTE F18 + the
    # slow_corr_kernel's `vis_real = V_real_b[..., b_idx, a_idx]` swap
    # — pinned in tests/test_slow_corr_synth.py
    # ::test_kernel_sign_convention_v_ant1_conj_ant2.
    a_idx, b_idx = upper_tri_indices(NANTS)
    nbase = a_idx.size
    # Take pol 0 only (single-pol injection; the integrated single-pol
    # value is what the gridder + iFFT need).
    e_lo = e_ant_ch_pol[:, 0, b_idx]                           # (NCHAN, NBASE)
    e_hi = e_ant_ch_pol[:, 0, a_idx]
    vis_per_chan = np.conj(e_lo) * e_hi                        # (NCHAN, NBASE)
    vis = vis_per_chan.transpose(1, 0)                         # (NBASE, NCHAN)
    vis = vis[..., None]                                       # (NBASE, NCHAN, 1) for grid_uv_natural
    print(f"[inject_visibility_recovery] vis: shape={vis.shape} "
          f"|max|={float(np.abs(vis).max()):.3e}", flush=True)

    # ---- 5. UVW + grid ----
    antpos_3d = np.zeros((NANTS, 3), dtype=np.float64)
    antpos_3d[:, 0] = e_ant
    antpos_3d[:, 1] = n_ant
    antpos_3d[:, 2] = u_ant
    uvw_m = antpos_3d[a_idx] - antpos_3d[b_idx]                # (NBASE, 3)
    freqs_Hz = np.array(
        [freq_GHz(args.chgroup, ch) * 1.0e9
         for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    grid, weight = grid_uv_natural(
        vis=vis.astype(np.complex64),
        uvw_m=uvw_m,
        freqs_Hz=freqs_Hz,
        n_grid=args.n_grid,
        fov_rad=args.fov_rad,
        pol=0,
        drop_autos=True,
    )
    print(f"[inject_visibility_recovery] grid: shape={grid.shape} "
          f"n_filled={int((weight > 0).sum())}", flush=True)

    # ---- 6. Dirty image + peak ----
    image = dirty_image_from_grid(grid)
    peaks = find_image_peaks(image, fov_rad=args.fov_rad, n_top=5,
                             edge_pad=4)

    # ---- 7. Compare ----
    cell_rad = args.fov_rad / args.n_grid
    dl = peaks[0].l_rad - args.l
    dm = peaks[0].m_rad - args.m
    n_cells_off = math.sqrt(dl * dl + dm * dm) / cell_rad
    print(
        f"[inject_visibility_recovery] injected (l, m) = "
        f"({args.l:+.5f}, {args.m:+.5f}); "
        f"recovered peak  = ({peaks[0].l_rad:+.5f}, {peaks[0].m_rad:+.5f}); "
        f"|Δ| = {n_cells_off:.2f} cells (1 cell = {cell_rad:.5f} rad)",
        flush=True,
    )

    return {
        "config": vars(args),
        "injection_cfg": {
            "inj_id": cfg.inj_id, "l_rad": cfg.l_rad, "m_rad": cfg.m_rad,
            "dm_pc_cm3": cfg.dm_pc_cm3, "fluence_jy_ms": cfg.fluence_jy_ms,
            "width_samples": cfg.width_samples, "profile": cfg.profile,
            "apply_at_specnum": cfg.apply_at_specnum,
        },
        "injection_log": log,
        "image_shape": list(image.shape),
        "n_filled": int((weight > 0).sum()),
        "peaks": [
            {"rank": p.rank, "l_rad": p.l_rad, "m_rad": p.m_rad,
             "flux": p.flux_image_units, "snr": p.snr_image_plane}
            for p in peaks
        ],
        "delta_l_rad": dl,
        "delta_m_rad": dm,
        "delta_n_cells": n_cells_off,
        "cell_rad": cell_rad,
        "image": image,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(result: dict, out_dir: Path) -> Path:
    """Write the operator-inspect HTML report + dirty-image PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image = result.pop("image")

    # PNG (rendered via tools.viz.common helpers — they handle the
    # matplotlib fallback gracefully if matplotlib isn't installed).
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Fall back: skip PNG, write the array as npy for offline inspect.
        np.save(out_dir / "dirty_image.npy", image)
        png_filename = None
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        fov = float(result["config"]["fov_rad"])
        half = fov / 2.0
        extent = (-half, +half, -half, +half)
        vmax = float(np.percentile(np.abs(image), 99.5))
        im = ax.imshow(image, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="equal")
        fig.colorbar(im, ax=ax, label="image flux (a.u.)")
        ax.set_xlabel("l (rad, east+)")
        ax.set_ylabel("m (rad, north+)")
        ax.set_title("M3 chunk-3d injector visibility recovery")

        # Injected and recovered marks.
        cfg = result["injection_cfg"]
        ax.plot(cfg["l_rad"], cfg["m_rad"], "o", mec="lime", mfc="none",
                markersize=14, mew=2, label=(
                    f"injected (l, m) = ({cfg['l_rad']:+.4f}, "
                    f"{cfg['m_rad']:+.4f})"
                ))
        for p in result["peaks"][:1]:
            ax.plot(p["l_rad"], p["m_rad"], "+", color="black",
                    markersize=14, mew=2, label=(
                        f"peak (l, m) = ({p['l_rad']:+.4f}, "
                        f"{p['m_rad']:+.4f})"
                    ))
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        png_path = out_dir / "dirty_image.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        png_filename = png_path.name

    # HTML report.
    cfg = result["injection_cfg"]
    rows_peaks = "\n".join(
        f"<tr><td>{p['rank']}</td><td>{p['l_rad']:+.5f}</td>"
        f"<td>{p['m_rad']:+.5f}</td><td>{p['flux']:.3g}</td>"
        f"<td>{p['snr']:.1f}</td></tr>"
        for p in result["peaks"]
    )
    rows_meta = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in {
            **result["config"],
            "n_filled": result["n_filled"],
            "image_shape": result["image_shape"],
            "delta_l_rad": f"{result['delta_l_rad']:+.5f}",
            "delta_m_rad": f"{result['delta_m_rad']:+.5f}",
            "delta_n_cells": f"{result['delta_n_cells']:.3f}",
            "cell_rad": f"{result['cell_rad']:.5f}",
        }.items()
    )
    image_html = (
        f'<img src="{png_filename}" alt="dirty image">'
        if png_filename else
        "<p>(matplotlib unavailable; dirty image saved as "
        "<code>dirty_image.npy</code>)</p>"
    )
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>M3 chunk-3d injector visibility recovery</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 1em auto; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
  th {{ background: #eee; }}
  img {{ max-width: 100%; }}
  .note {{ background: #fffbe6; padding: 8px; border-left: 4px solid #fc0; }}
</style></head>
<body>
<h1>M3 chunk-3d — injector visibility recovery</h1>
<p class="note"><strong>No PASS/FAIL banner</strong> — operator
inspects the dirty image + peak table and approves out of band
(D11 from M2 carryover). Success criterion: ‖peak − injected‖ ≤ 1
grid cell.</p>

<h2>Injected source</h2>
<table>
<tr><th>field</th><th>value</th></tr>
<tr><td>inj_id</td><td>{cfg['inj_id']}</td></tr>
<tr><td>l_rad</td><td>{cfg['l_rad']:+.5f}</td></tr>
<tr><td>m_rad</td><td>{cfg['m_rad']:+.5f}</td></tr>
<tr><td>dm_pc_cm3</td><td>{cfg['dm_pc_cm3']}</td></tr>
<tr><td>fluence_jy_ms</td><td>{cfg['fluence_jy_ms']}</td></tr>
<tr><td>width_samples</td><td>{cfg['width_samples']}</td></tr>
<tr><td>profile</td><td>{cfg['profile']}</td></tr>
<tr><td>apply_at_specnum</td><td>{cfg['apply_at_specnum']}</td></tr>
</table>

<h2>Dirty image</h2>
{image_html}

<h2>Top peaks</h2>
<table>
<tr><th>rank</th><th>l (rad)</th><th>m (rad)</th><th>flux</th><th>SNR</th></tr>
{rows_peaks}
</table>

<h2>Metadata</h2>
<table>{rows_meta}</table>

</body></html>
"""
    html_path = out_dir / "report.html"
    html_path.write_text(html)
    # Result JSON (drop the image, which is too big for JSON).
    json_path = out_dir / "result.json"
    json_payload = {k: v for k, v in result.items() if k != "image"}
    json_path.write_text(json.dumps(json_payload, indent=2, default=str))
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l", type=float, default=0.05,
                   help="Injected source l direction-cosine (rad).")
    p.add_argument("--m", type=float, default=0.0,
                   help="Injected source m direction-cosine (rad).")
    p.add_argument("--dm", type=float, default=200.0,
                   help="Injected DM (pc / cm³).")
    p.add_argument("--width", type=int, default=8, dest="width",
                   help="Injected pulse FWHM in NATIVE samples.")
    p.add_argument("--fluence", type=float, default=10.0, dest="fluence",
                   help="Injected fluence (Jy · ms).")
    p.add_argument("--profile", type=str, default="gaussian",
                   choices=("gaussian", "boxcar"))
    p.add_argument("--chgroup", type=int, default=0,
                   help="Corr-node chgroup index 0..15.")
    p.add_argument("--apply-at-specnum", type=int,
                   default=NPACKETS_PER_BLOCK // 2,
                   help="SNAP specnum of injection peak (default: block centre).")
    p.add_argument("--noise-sigma", type=float, default=1.0e-3,
                   help="Per-component voltage noise σ.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-grid", type=int, default=256, dest="n_grid")
    p.add_argument("--fov-rad", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def _default_out_dir() -> Path:
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_id = f"chunk3d_{uuid.uuid4().hex[:6]}"
    return REPO_ROOT / "bench" / "reports" / utc / run_id / "M3-injector"


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir()
    result = run(args)
    html_path = render_report(result, out_dir)
    print(f"[inject_visibility_recovery] report → {html_path}", flush=True)
    print(
        f"[inject_visibility_recovery] |Δ| = "
        f"{result['delta_n_cells']:.3f} grid cells "
        f"(1 cell = {result['cell_rad']:.5f} rad)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
