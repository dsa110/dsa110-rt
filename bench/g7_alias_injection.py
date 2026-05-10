#!/usr/bin/env python3
"""bench/g7_alias_injection.py — G7 anti-aliasing kernel sanity bench.

Plan §4.2 line 1351 (G7) introduces the K∈{1,3,5} Gaussian gridding
kernel. K=1 is the legacy nearest-cell pillbox; K>1 spreads each
``(bls, ch)`` Stokes-I sample over a K×K Gaussian-weighted neighborhood
with σ = (K−1)/4 cells (so the per-tap weights sum to 1.0).

Why a bench
===========

Off-axis sources (sources outside ~half the primary beam) are the
canonical alias-injection target: the nearest-cell pillbox imager
spreads their amplitude into the conjugate ``(−l, −m)`` peak as well
as the true ``(+l, +m)`` peak, contaminating the search.

This bench:

1. Loads the 0319 cal blob's antpos arrays
   (``/home/ubuntu/data/voltages/0319/cals/beamformer_weights_sb00_0319+415.dat``;
   falls back to 250924mptq cals if 0319 not present on this host).
2. Synthesises a single-source Stokes-I visibility tensor for a CW
   point source at off-axis ``(l, m) = (0.05, 0.0)`` rad — well
   outside the ~0.025 rad primary-beam HWHM at 1.5 GHz with the 4.5 m
   antennas.
3. Builds the gridder for K ∈ {1, 3, 5} against the same antpos +
   chgroup + dec.
4. Grids the vis, scatters to the dense ``(N_grid, N_grid)`` grid,
   and dirty-images it (``Re(fftshift(iFFT2(ifftshift(grid))))``).
5. Records the peak amplitude at the predicted source pixel and at
   the conjugate ``(−l, −m)`` pixel; reports the suppression ratio
   between the two as a function of K.

Acceptance criterion (per the M3 production-readiness review):

* K=3 must reduce conjugate-peak amplitude by ≥ 3× vs K=1.
* K=5 must reduce conjugate-peak amplitude by ≥ 10× vs K=1.

If either threshold is missed, exit 1 and print which K failed.

Outputs
=======

* ``/tmp/dsart-g7-bench/report.json`` — per-K source/conjugate peak
  amplitudes, the suppression ratios, and a PASS/FAIL boolean.
* ``/tmp/dsart-g7-bench/triptych.png`` — 3-panel dirty-image
  comparison K=1 / K=3 / K=5 with a marker at the predicted source
  position and a circle at the conjugate.

Usage
=====

::

    python -m bench.g7_alias_injection \\
        [--cal-blob /path/to/beamformer_weights_*.dat] \\
        [--chgroup 0] [--obs-dec-deg 41.5074]              \\
        [--n-grid 256] [--n-fast-vis 4]                     \\
        [--source-l 0.05] [--source-m 0.0]                  \\
        [--out-dir /tmp/dsart-g7-bench]

References
==========

* Plan §4.2 line 1351 — G7 anti-aliasing kernel spec.
* :mod:`dsart.grid.sparsity_pattern` (gaussian_kernel_weights,
  build_pattern with K∈{1,3,5}).
* :mod:`dsart.grid.kernel` (FastVisGridder K² scatter).
* :mod:`bench._corr_fast_replay`
  (sparse_to_dense_grid / dirty_image_from_dense_grid /
  compute_chgroup_cell_lambda / pixel_to_lm_radians).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.cal.bf_weights import load_bf_weights                       # noqa: E402
from dsart.common.constants import (                                   # noqa: E402
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    PHI_LAT_OVRO_DEG,
    SPEED_OF_LIGHT_M_S,
    freq_GHz,
)
from dsart.grid.kernel import FastVisGridder                           # noqa: E402
from dsart.grid.sparsity_pattern import (                              # noqa: E402
    build_pattern,
    core_baseline_mask_from_station_numbers,
    core_baseline_mask_from_antpos,
)


_DEFAULT_CALS = [
    Path("/home/ubuntu/data/voltages/0319/cals"),
    Path("/home/ubuntu/data/voltages/250924mptq/cals"),
]


def _resolve_cal_blob(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            print(f"ERROR: --cal-blob {p} not found", file=sys.stderr)
            sys.exit(2)
        return p
    for cals_dir in _DEFAULT_CALS:
        if cals_dir.is_dir():
            for blob in sorted(cals_dir.glob("beamformer_weights_sb00*.dat")):
                if blob.is_file():
                    return blob
    print(
        "ERROR: no beamformer_weights_sb00*.dat found in 0319/cals or "
        "250924mptq/cals; pass --cal-blob",
        file=sys.stderr,
    )
    sys.exit(2)


def _resolve_core_mask(cal_blob: Path, antpos_e, antpos_n) -> np.ndarray:
    """Mirror corr_fast_integration.load_antpos_from_cal_blob's F32 path.

    Prefer station-number mask from the sibling cal yaml; fall back to
    radius-based on synthetic antpos.
    """
    yaml_candidates = sorted(cal_blob.parent.glob("beamformer_weights_*.yaml"))
    if yaml_candidates:
        import yaml as _yaml
        with open(yaml_candidates[0], "r") as f:
            ydoc = _yaml.safe_load(f)
        antenna_order = ydoc["cal_solutions"]["antenna_order"]
        return core_baseline_mask_from_station_numbers(antenna_order)
    return core_baseline_mask_from_antpos(antpos_e, antpos_n, n_core=82)


def _kept_baseline_uv_metres(
    antpos_e: np.ndarray, antpos_n: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (a_kept, b_kept, du_m, dv_m) for cross + core baselines.

    Iteration order matches :func:`upper_tri_indices` (the production
    F18 baseline ordering): for ``a in 0..NANTS-1`` and ``b in 0..a``.
    Autos (``a == b``) and outrigger-touching baselines (mask=False)
    are dropped — same convention as the gridder's pattern build, so
    the returned ``du_m, dv_m`` align row-for-row with the kept-
    baseline dimension of any vis tensor we build below.
    """
    nants = antpos_e.shape[0]
    nbase = nants * (nants + 1) // 2
    a_list = np.empty(nbase, dtype=np.int64)
    b_list = np.empty(nbase, dtype=np.int64)
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            a_list[k] = a
            b_list[k] = b
            k += 1
    is_cross = a_list != b_list
    keep = is_cross & np.asarray(mask, dtype=bool)
    a_kept = a_list[keep]
    b_kept = b_list[keep]
    du_m = (
        antpos_e[a_kept].astype(np.float64)
        - antpos_e[b_kept].astype(np.float64)
    )
    dv_m = (
        antpos_n[a_kept].astype(np.float64)
        - antpos_n[b_kept].astype(np.float64)
    )
    return a_kept, b_kept, du_m, dv_m


def synth_vis_for_point_source(
    antpos_e: np.ndarray, antpos_n: np.ndarray, core_mask: np.ndarray,
    *,
    chgroup: int,
    n_fast_vis: int,
    source_l: float, source_m: float,
    amplitude: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Synthesise ``(n_fv, NBASE, NCHAN)`` Stokes-I vis for one CW source.

    The visibility model follows the F20 sign convention as it appears
    BEFORE the gridder applies its negation:

    .. math::

        V(u_\\lambda, v_\\lambda)
            = A \\cdot \\exp\\!\\big(+\\,2\\pi i\\,(u_\\lambda\\,l + v_\\lambda\\,m)\\big)

    with ``(u_lam, v_lam) = (du_m, dv_m) / lambda_ch`` — i.e. raw
    geometric u, v in λ-units. After the gridder negates ``(u, v)``
    and the dirty-image step does ``iFFT2``, a CW source at ``(l, m)``
    lands at the corresponding ``(+l, +m)`` pixel (same convention as
    :mod:`tools.viz.common.grid_uv_natural`).

    Outrigger / auto baselines that the gridder discards via the
    ``cell_index_map`` sentinel are still populated here (so the
    output is full-(NBASE, NCHAN) shape) but with zeros, so they
    contribute nothing to the gridded image.
    """
    a_kept, b_kept, du_m, dv_m = _kept_baseline_uv_metres(
        antpos_e, antpos_n, core_mask,
    )

    nu_GHz = np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    wavelength_m = SPEED_OF_LIGHT_M_S / (nu_GHz * 1e9)                # (NCHAN,)
    u_lam = du_m[:, None] / wavelength_m[None, :]                     # (Nkept, NCHAN)
    v_lam = dv_m[:, None] / wavelength_m[None, :]
    phase = 2.0 * math.pi * (u_lam * source_l + v_lam * source_m)
    v_kept = amplitude * (np.cos(phase) + 1j * np.sin(phase))         # (Nkept, NCHAN)

    vis = np.zeros((NBASE, NCHAN_PER_CHGROUP), dtype=np.complex64)
    bls_idx = (a_kept * (a_kept + 1) // 2) + b_kept
    vis[bls_idx, :] = v_kept.astype(np.complex64)

    vis_4d = np.broadcast_to(
        vis[None, :, :], (n_fast_vis, NBASE, NCHAN_PER_CHGROUP),
    ).copy()
    return torch.from_numpy(vis_4d).to(device)


def grid_and_image(
    vis: torch.Tensor,
    *,
    antpos_e: np.ndarray, antpos_n: np.ndarray, core_mask: np.ndarray,
    chgroup: int, dec_deg: float, n_grid: int, kernel_support: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Build pattern + gridder for K, grid + dirty-image the vis.

    Returns:
        (dirty_image, cell_lambda)
        - dirty_image: ``(n_grid, n_grid) float32`` tile-0 dirty image
          (sum the rest of the n_fv axis if you want a noisier version;
          for a CW point source they're all identical).
        - cell_lambda: float, the gridder's per-cell λ-extent (used
          downstream to convert pixel coords to (l, m) rad).
    """
    pattern = build_pattern(
        antpos_e, antpos_n,
        chgroup=chgroup, dec_deg=dec_deg, n_grid=n_grid,
        kernel_support=kernel_support, is_core_baseline_mask=core_mask,
    )
    gridder = FastVisGridder.from_pattern(
        pattern, antpos_e, antpos_n,
        is_core_baseline_mask=core_mask, device=device,
    )
    sparse = gridder.compute(vis)                                      # (n_fv, N_filled) cfp32

    # Scatter sparse → dense (N_grid, N_grid) and IFFT.
    n_filled = pattern.n_filled
    ix_row = pattern.ix_row.astype(np.int64)
    ix_col = pattern.ix_col.astype(np.int64)
    dense = torch.zeros(
        (vis.shape[0], n_grid, n_grid),
        dtype=torch.complex64, device=device,
    )
    flat = dense.view(vis.shape[0], n_grid * n_grid)
    flat_idx = torch.from_numpy(
        (ix_row * n_grid + ix_col).astype(np.int64),
    ).to(device)
    flat.scatter_(1, flat_idx.unsqueeze(0).expand(vis.shape[0], -1), sparse)
    dense = flat.view(vis.shape[0], n_grid, n_grid)

    grid_shifted = torch.fft.ifftshift(dense, dim=(-2, -1))
    img_complex = torch.fft.ifft2(grid_shifted, dim=(-2, -1))
    img = torch.fft.fftshift(img_complex, dim=(-2, -1)).real

    # cell_lambda is recoverable from pattern: max_baseline_lambda * 2 / n_grid.
    # We need it for the (l, m) → pixel mapping. Re-compute exactly the
    # same way build_pattern did.
    from dsart.grid.sparsity_pattern import _per_baseline_uv_meters
    du_m, dv_m = _per_baseline_uv_meters(
        antpos_e, antpos_n, is_core_baseline_mask=core_mask,
    )
    nu_top = freq_GHz(chgroup, 0)
    wl_top = SPEED_OF_LIGHT_M_S / (nu_top * 1e9)
    u_lam = -du_m / wl_top                                             # F20 negation
    v_lam = -dv_m / wl_top
    # ``max_baseline_lambda`` scans all NCHAN — but the smallest
    # wavelength (chgroup top) gives the largest |u| / |v|. So the top
    # channel sets cell_lambda; this matches build_pattern.
    nu_full = np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    wl_full = SPEED_OF_LIGHT_M_S / (nu_full * 1e9)
    u_lam_full = -du_m[:, None] / wl_full[None, :]
    v_lam_full = -dv_m[:, None] / wl_full[None, :]
    max_baseline_lambda = float(
        np.max(np.maximum(np.abs(u_lam_full), np.abs(v_lam_full)))
    )
    cell_lambda = max_baseline_lambda * 2.0 / n_grid

    return img[0].to("cpu"), cell_lambda


def lm_to_pixel(
    l_rad: float, m_rad: float, *, n_grid: int, cell_lambda: float,
) -> tuple[int, int]:
    """Convert (l, m) ∈ rad → (ix_row, ix_col) on the (n_grid, n_grid) image."""
    half = n_grid // 2
    pixel_size_lm = 1.0 / (n_grid * cell_lambda)
    ix_col = int(round(l_rad / pixel_size_lm)) + half
    ix_row = int(round(m_rad / pixel_size_lm)) + half
    return ix_row, ix_col


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cal-blob", default=None,
                   help="path to beamformer_weights_*.dat (default: 0319 cals)")
    p.add_argument("--chgroup", type=int, default=0)
    p.add_argument("--obs-dec-deg", type=float, default=PHI_LAT_OVRO_DEG)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--n-fast-vis", type=int, default=4)
    p.add_argument("--source-l", type=float, default=0.05,
                   help="source l (rad) — default 0.05 = ~0.5× PB at 1.4 GHz")
    p.add_argument("--source-m", type=float, default=0.0)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/dsart-g7-bench"))
    p.add_argument("--device", default="cpu",
                   help="cpu / cuda / cuda:0 (default: cpu — bench is small)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cal_blob = _resolve_cal_blob(args.cal_blob)
    print(f"[g7-bench] cal blob: {cal_blob}")
    bf = load_bf_weights(cal_blob)
    antpos_e = np.asarray(bf.antpos_e, dtype=np.float32)
    antpos_n = np.asarray(bf.antpos_n, dtype=np.float32)
    core_mask = _resolve_core_mask(cal_blob, antpos_e, antpos_n)
    n_core_baselines = int(core_mask.sum())
    print(f"[g7-bench] NANTS={NANTS} N_core_baselines_kept={n_core_baselines}")

    print(
        f"[g7-bench] synth source at (l, m) = ({args.source_l:+.4f}, "
        f"{args.source_m:+.4f}) rad; chgroup={args.chgroup} "
        f"obs_dec={args.obs_dec_deg:.4f} deg"
    )
    vis = synth_vis_for_point_source(
        antpos_e, antpos_n, core_mask,
        chgroup=args.chgroup, n_fast_vis=args.n_fast_vis,
        source_l=args.source_l, source_m=args.source_m,
        device=device,
    )

    results: dict[int, dict] = {}
    images: dict[int, np.ndarray] = {}
    for K in (1, 3, 5):
        print(f"[g7-bench] ---- K={K} ----")
        img, cell_lambda = grid_and_image(
            vis,
            antpos_e=antpos_e, antpos_n=antpos_n, core_mask=core_mask,
            chgroup=args.chgroup, dec_deg=args.obs_dec_deg,
            n_grid=args.n_grid, kernel_support=K, device=device,
        )
        img_np = img.numpy()
        images[K] = img_np

        # Source pixel
        src_row, src_col = lm_to_pixel(
            args.source_l, args.source_m,
            n_grid=args.n_grid, cell_lambda=cell_lambda,
        )
        # Conjugate pixel
        conj_row, conj_col = lm_to_pixel(
            -args.source_l, -args.source_m,
            n_grid=args.n_grid, cell_lambda=cell_lambda,
        )

        # Robustly read peaks: 3x3 max-filter around the predicted pixel
        # (the image cell may shift by 1 due to fp rounding in
        # cell_lambda).
        def _peak_in_window(arr, r, c, w=2):
            r0, r1 = max(0, r - w), min(arr.shape[0], r + w + 1)
            c0, c1 = max(0, c - w), min(arr.shape[1], c + w + 1)
            window = arr[r0:r1, c0:c1]
            return float(np.abs(window).max())

        src_peak = _peak_in_window(img_np, src_row, src_col)
        conj_peak = _peak_in_window(img_np, conj_row, conj_col)
        global_peak = float(np.abs(img_np).max())
        results[K] = {
            "kernel_support": K,
            "cell_lambda": float(cell_lambda),
            "source_pixel": [int(src_row), int(src_col)],
            "conj_pixel": [int(conj_row), int(conj_col)],
            "source_peak": src_peak,
            "conj_peak": conj_peak,
            "global_peak": global_peak,
            "src_to_conj_ratio": (
                float("inf") if conj_peak == 0 else src_peak / conj_peak
            ),
        }
        print(
            f"[g7-bench] K={K}: src_peak={src_peak:.4g} "
            f"conj_peak={conj_peak:.4g} src/conj={results[K]['src_to_conj_ratio']:.3g} "
            f"cell_lambda={cell_lambda:.3f}"
        )

    # Suppression ratios (lower conj_peak = better).
    K1_conj = results[1]["conj_peak"]
    K3_conj = results[3]["conj_peak"]
    K5_conj = results[5]["conj_peak"]
    suppress_K3 = (K1_conj / K3_conj) if K3_conj > 0 else float("inf")
    suppress_K5 = (K1_conj / K5_conj) if K5_conj > 0 else float("inf")
    pass_K3 = suppress_K3 >= 3.0
    pass_K5 = suppress_K5 >= 10.0

    print(
        f"[g7-bench] alias suppression vs K=1: "
        f"K=3 → {suppress_K3:.2f}× (need ≥3×) {'PASS' if pass_K3 else 'FAIL'}; "
        f"K=5 → {suppress_K5:.2f}× (need ≥10×) {'PASS' if pass_K5 else 'FAIL'}"
    )

    report = {
        "cal_blob": str(cal_blob),
        "chgroup": int(args.chgroup),
        "obs_dec_deg": float(args.obs_dec_deg),
        "n_grid": int(args.n_grid),
        "n_fast_vis": int(args.n_fast_vis),
        "source_l": float(args.source_l),
        "source_m": float(args.source_m),
        "n_core_baselines_kept": int(n_core_baselines),
        "per_K": {str(K): r for K, r in results.items()},
        "alias_suppression": {
            "K3_vs_K1": float(suppress_K3),
            "K5_vs_K1": float(suppress_K5),
            "K3_threshold": 3.0,
            "K5_threshold": 10.0,
            "pass_K3": pass_K3,
            "pass_K5": pass_K5,
        },
        "pass": pass_K3 and pass_K5,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[g7-bench] report → {args.out_dir / 'report.json'}")

    # Triptych PNG (best-effort; matplotlib may not be installed in
    # all envs — fall back to no PNG rather than failing the bench).
    try:
        import matplotlib                                              # noqa: E402
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt                                 # noqa: E402
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, K in zip(axes, (1, 3, 5)):
            arr = images[K]
            vmax = float(np.abs(arr).max())
            ax.imshow(arr, origin="lower", cmap="viridis",
                      vmin=-vmax, vmax=+vmax)
            r, c = results[K]["source_pixel"]
            cr, cc = results[K]["conj_pixel"]
            ax.plot(c, r, "wx", ms=12, mew=2, label="source")
            ax.plot(cc, cr, "ro", mfc="none", ms=14, mew=2, label="conj")
            ax.set_title(
                f"K={K}\nsrc={results[K]['source_peak']:.2g} "
                f"conj={results[K]['conj_peak']:.2g}"
            )
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"G7 alias-injection: source at "
            f"(l, m) = ({args.source_l:+.3f}, {args.source_m:+.3f}) rad "
            f"chgroup={args.chgroup}"
        )
        fig.tight_layout()
        png_path = args.out_dir / "triptych.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"[g7-bench] triptych → {png_path}")
    except ImportError:
        print(
            f"[g7-bench] matplotlib not available; skipping triptych PNG"
        )

    return 0 if (pass_K3 and pass_K5) else 1


if __name__ == "__main__":
    sys.exit(main())
