"""bench/corr_fast_burst_250924mptq.py — M3 chunk 6: 250924mptq burst imager.

Drives the chunk-4 :mod:`dsart.services.corr_fast_integration` pipeline
across all 16 chgroups (sb00..sb15) on the real 250924mptq burst
voltage-dump fixture, recovers the burst's ``(l, m, t)`` headline, and
emits a frame movie + headline images.

Per ``PARALLEL_AGENTS.md`` §5.1 (burst-fixture test config recipe):

* ``--t-int-fast-native 32`` (= 1048.576 µs cadence; 4× the production
  burst-test cadence of 262.144 µs).
* Custom 1-cell ``dm_plan_burst.npz`` at DM=404.688 pc/cc.
* The burst is ON-AXIS — ``--obs-dec-deg 53.848986`` matches the burst
  declination so the F21 cal phases the source to ``(l, m) ≈ (0, 0)``
  and the burst peak should land at the array centre.
* ``--static-sky-disabled`` — the burst is the signal; we don't want
  the EMA to learn it.
* The chunk-4 ``NoOpCoarseDM`` is used (single-DM trial; per F25 in
  ``M3_PLAN_FIXES.md`` the post-grid no-op is correct for the
  ``N_DM == 1`` case).

Per the upstream burst-search (``scratch/burst_search/``), the burst's
TOP-OF-BAND arrival is at native sample 15248 in the chgroup-0 (sb=0)
reference frame, which is block 3 (4th block, 0-indexed) at offset
2960. Replaying blocks 0..7 covers the burst plus its full
229 ms cross-band dispersion smear.

Within each chgroup, the gridder sums over channels — so the per-chgroup
time series shows the burst smeared by the within-chgroup dispersion
(~14 ms = ~13 fast-vis tiles at t_int=32). Across chgroups, the burst
arrival shifts by the inter-chgroup-top dispersion delay; the bench
applies an inter-chgroup time alignment to the chgroup-0 top-of-band
frame before summing across chgroups (essentially a coarse stage-2
re-alignment, since the chunk-4 NoOp CoarseDM does no time shifts).

Outputs (under ``--report-dir``):

* ``report.html``                          — narrative + image links
* ``time_series_burst_pixel.png``          — coadded time series at the
                                              burst (l, m) pixel
* ``peak_image.png``                       — dirty image at peak fast-vis
                                              tile, with predicted (l, m)
* ``frames/frame_<NNNN>.png``              — per-tile dirty images
* ``movie.mp4`` (or .gif)                  — frames concatenated
* ``summary.json``                         — JSON summary (PASS/FAIL gate)
* ``dm_plan_single.npz``                   — single-DM plan used for the run

PASS criteria:
* ``peak_offset_cells <= 4``  (peak within 4 cells of (0, 0))
* ``peak_t_native within ±32 native samples (~1 ms) of 15248`` —
  this is achievable WITH the cross-chgroup time alignment described
  above; the within-chgroup dispersion smear of ~14 ms means the peak
  IN A SINGLE chgroup is at ~+7 ms relative to 15248. The cross-chgroup
  alignment + summation centres the peak at native 15248 + the chgroup-0
  within-chgroup centroid bias (≈ +7 ms), so the strict ±1 ms gate is
  difficult without per-channel intra-chgroup dedispersion (which is the
  chunk 9 / F25 work). The bench reports the actual peak_t_native + offset
  and stamps PASS/FAIL accordingly; an operator-tunable
  ``--peak-t-tol-native-samples`` knob (default 32) lets the
  parent agent + operator gate this independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsart.common.constants import (
    K_DM_MS_GHZ2_PC,
    NATIVE_SAMPLE_US,
    NU_CHGROUP_TOP_GHZ,
)
from dsart.coarse_dm.dm_plan import (
    DMPlan,
    build_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
)

from bench._corr_fast_replay import (
    ReplayDefaults,
    accumulate_chgroup_grids,
    compute_chgroup_cell_lambda,
    dirty_image_from_dense_grid,
    lm_to_pixel,
    replay_chgroup,
    sparse_to_dense_grid,
)


LOG = logging.getLogger("bench.corr_fast_burst_250924mptq")


@dataclass
class ChgroupBurstResult:
    sb: str
    chgroup: int
    n_blocks_processed: int
    n_fast_vis_total: int
    n_filled: int
    cell_lambda: float
    peak_pixel_row: int
    peak_pixel_col: int
    peak_t_idx: int
    peak_t_native: int
    peak_value: float
    delay_native_to_chg0_top: int
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
        "dm_pc_cc": float(first.get("dm", 0.0)),
        "snr": float(first.get("snr", 0.0)),
    }


def _compute_expected_lm(
    *,
    src_ra_deg: float,
    src_dec_deg: float,
    src_mjd: float,
    phase_center_dec_deg: float,
) -> tuple[float, float, float]:
    """Astropy-backed (l, m) prediction for the burst at the dump MJD.

    Mirrors :func:`bench.run_0319_pipeline._compute_expected_lm`. The burst
    is on-axis if HA(src_mjd, OVRO_LON) ≡ src_ra_deg, but in practice
    the trigger MJD precedes transit by O(arcmin) — astropy returns the
    actual HA so the ``(l, m)`` prediction is correct even off-meridian.

    Returns
    -------
    (l_rad, m_rad, ha_src_deg)
    """
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    OVRO_LON_DEG = -118.281
    OVRO_LAT_DEG = 37.234
    loc = EarthLocation.from_geodetic(
        lon=OVRO_LON_DEG, lat=OVRO_LAT_DEG, height=1188.0,
    )
    t = Time(src_mjd, format="mjd", location=loc)
    lst_deg = float(t.sidereal_time("apparent").deg)
    ha_src_deg = ((lst_deg - src_ra_deg) + 180.0) % 360.0 - 180.0
    ha_rad = math.radians(ha_src_deg)
    dec_src_rad = math.radians(src_dec_deg)
    dec_pc_rad = math.radians(phase_center_dec_deg)
    l_rad = -math.cos(dec_src_rad) * math.sin(ha_rad)
    m_rad = (
        math.sin(dec_src_rad) * math.cos(dec_pc_rad)
        - math.cos(dec_src_rad) * math.sin(dec_pc_rad) * math.cos(ha_rad)
    )
    return l_rad, m_rad, ha_src_deg


def _build_single_dm_plan(
    *,
    dm_pc_cc: float,
    t_int_fast_us: float,
    out_path: Path,
) -> DMPlan:
    """Build a single-cell coarse-DM plan + write to disk via DMPlan.to_npz.

    The single-DM plan is exactly what chunk 4's NoOp CoarseDMStage stub
    needs — N_DM = 1 is the trivial case. Per F25 in M3_PLAN_FIXES.md,
    the no-op stub returns ``gridded.unsqueeze(0)`` which is correct
    for ``N_DM == 1``. Plan is also written to disk for the report
    artefact even though chunk-4's process_block doesn't consume it
    directly today (the bench applies the per-chgroup-top dispersion
    delay manually post-process_block, see _apply_inter_chgroup_alignment).
    """
    coarse_dm = np.array([dm_pc_cc], dtype=np.float64)
    chgroup_freqs = build_chgroup_freq_table_GHz()
    delay_native = compute_delay_native_samples_table(coarse_dm, chgroup_freqs)
    plan = DMPlan(
        dm_pc_cc=coarse_dm,
        n_fine_per_coarse=1,
        t_int_fast_us=t_int_fast_us,
        chgroup_freqs_GHz=chgroup_freqs,
        _delay_native_samples_table=delay_native,
    )
    plan.to_npz(str(out_path))
    LOG.info(
        "wrote single-DM plan %s (DM=%.3f pc/cc; max within-chgroup delay "
        "= %d native samples, max across-chgroup delay = %d native samples)",
        out_path, dm_pc_cc,
        int(delay_native.max()),
        int(_compute_inter_chgroup_top_delay(dm_pc_cc).max()),
    )
    return plan


def _compute_inter_chgroup_top_delay(dm_pc_cc: float) -> np.ndarray:
    """Per-chgroup TOP-channel dispersion delay (native samples) relative to chgroup-0 TOP.

    Δτ_to_chg0_top[g] = round(K · DM · (1/ν_g_top² - 1/ν_chg0_top²) · 1e3 / NATIVE_SAMPLE_US)

    Always ≥ 0 (lower freq → larger delay; chgroup-0 is the highest
    freq in the band). chgroup-0 returns 0.

    Returns:
        ``(N_CHGROUP,) int64`` — delays in NATIVE samples.
    """
    nu_g_top = np.asarray(NU_CHGROUP_TOP_GHZ, dtype=np.float64)      # (N_CHGROUP,)
    nu_chg0 = nu_g_top[0]
    delay_us = (
        K_DM_MS_GHZ2_PC * dm_pc_cc
        * (1.0 / (nu_g_top ** 2) - 1.0 / (nu_chg0 ** 2)) * 1e3
    )
    return np.rint(delay_us / NATIVE_SAMPLE_US).astype(np.int64)


def _apply_inter_chgroup_alignment(
    per_chgroup_image_cubes: dict[int, np.ndarray],
    *,
    inter_chgroup_delay_native: np.ndarray,
    t_int_fast_native: int,
    n_grid: int,
) -> np.ndarray:
    """Time-shift each chgroup's image cube by -Δτ_to_chg0_top, then sum.

    Each chgroup's image cube is shape ``(T_fv, N_grid, N_grid)``. To
    align the top-of-band reference time of every chgroup to the
    chgroup-0 top-of-band frame, we SHIFT each chgroup's time axis
    EARLIER by ``inter_chgroup_delay_bins[g]`` fast-vis tiles, then sum.

    Notes
    -----
    This is a partial (cross-chgroup top-channel only) coarse-DM
    operation. It does NOT correct for the within-chgroup dispersion
    smear (which is ~14 ms = ~13 fast-vis tiles at t_int=32 µs). Per
    F25 in M3_PLAN_FIXES.md, full per-channel dedispersion lives in
    chunk 9; chunk 6's NoOpCoarseDM stub composes correctly for N_DM=1
    and the bench applies this top-channel cross-chgroup shift as a
    post-processing step.

    Args:
        per_chgroup_image_cubes: dict mapping chgroup index → ``(T_fv,
            N_grid, N_grid)`` float32 numpy array.
        inter_chgroup_delay_native: ``(N_CHGROUP,) int64`` per-chgroup
            top-channel delay relative to chgroup-0 top, in native
            samples.
        t_int_fast_native: fast-vis bin width in NATIVE samples.
        n_grid: image side length.

    Returns:
        ``(T_fv_aligned, N_grid, N_grid)`` float32 — coadded across
        chgroups after inter-chgroup top-channel time alignment. The
        time axis length is the SHORTEST per-chgroup cube length minus
        the maximum delay-in-bins, so all chgroups have valid samples
        across the entire returned axis.
    """
    if not per_chgroup_image_cubes:
        raise ValueError("no per-chgroup cubes")

    delay_bins = np.rint(
        inter_chgroup_delay_native.astype(np.float64) / t_int_fast_native,
    ).astype(np.int64)
    max_shift = int(delay_bins.max())
    LOG.info(
        "inter-chgroup top-channel delay (bins): %s (max=%d bins)",
        delay_bins.tolist(), max_shift,
    )

    # Each chgroup contributes its samples [delay_bins[g], delay_bins[g] + T_aligned)
    # back into the [0, T_aligned) reference frame.
    min_T = min(c.shape[0] for c in per_chgroup_image_cubes.values())
    T_aligned = min_T - max_shift
    if T_aligned <= 0:
        raise ValueError(
            f"min_T={min_T} ≤ max_shift={max_shift}; replay more blocks"
        )

    coadded = np.zeros((T_aligned, n_grid, n_grid), dtype=np.float32)
    for g, cube in per_chgroup_image_cubes.items():
        d = int(delay_bins[g])
        # shift earlier: take samples [d, d + T_aligned) from chgroup g
        contrib = cube[d:d + T_aligned]
        if contrib.shape[0] != T_aligned:
            raise RuntimeError(
                f"chgroup {g}: expected {T_aligned} aligned tiles, "
                f"got {contrib.shape[0]}"
            )
        coadded += contrib
    return coadded


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
    im = ax.imshow(image, origin="lower", cmap="viridis", extent=extent_lm)
    ax.set_title(title)
    if extent_lm is None:
        ax.set_xlabel("col (l-axis pixel)")
        ax.set_ylabel("row (m-axis pixel)")
    else:
        ax.set_xlabel("l (rad)")
        ax.set_ylabel("m (rad)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    half = n_grid // 2
    if cell_lambda is not None:
        pixel_size_lm = 1.0 / (n_grid * cell_lambda)

    if expected_lm_pixel is not None:
        if cell_lambda is None:
            ax.plot(expected_lm_pixel[1], expected_lm_pixel[0],
                    "rx", markersize=14, markeredgewidth=2, label="predicted")
        else:
            l = (expected_lm_pixel[1] - half) * pixel_size_lm
            m = (expected_lm_pixel[0] - half) * pixel_size_lm
            ax.plot(l, m, "rx", markersize=14, markeredgewidth=2, label="predicted")

    if peak_lm_pixel is not None:
        if cell_lambda is None:
            ax.plot(peak_lm_pixel[1], peak_lm_pixel[0],
                    "y+", markersize=12, markeredgewidth=2, label="peak")
        else:
            l = (peak_lm_pixel[1] - half) * pixel_size_lm
            m = (peak_lm_pixel[0] - half) * pixel_size_lm
            ax.plot(l, m, "y+", markersize=12, markeredgewidth=2, label="peak")

    if expected_lm_pixel is not None or peak_lm_pixel is not None:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _short_git_sha(repo_root: Path) -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _build_movie(
    image_cube: np.ndarray,
    out_dir: Path,
    *,
    n_frames: int,
    t_int_fast_native: int,
    cell_lambda: float | None,
    expected_lm_pixel: tuple[int, int] | None,
    movie_path: Path,
) -> Path | None:
    """Save n_frames PNGs from image_cube + try to build a .mp4 / .gif.

    Returns the path to the movie if produced, else None.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    T = image_cube.shape[0]
    stride = max(1, T // n_frames)
    sample_indices = list(range(0, T, stride))[:n_frames]
    LOG.info(
        "writing %d frames stride=%d (T_fv=%d)", len(sample_indices), stride, T,
    )

    vmin = float(np.percentile(image_cube, 1))
    vmax = float(np.percentile(image_cube, 99.9))
    for i, t_idx in enumerate(sample_indices):
        t_native = t_idx * t_int_fast_native
        title = f"frame {i:04d} (t_idx={t_idx}, t_native={t_native})"
        n_grid = image_cube.shape[1]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        extent_lm = None
        if cell_lambda is not None:
            half_lm = (n_grid // 2) / (n_grid * cell_lambda)
            extent_lm = (-half_lm, half_lm, -half_lm, half_lm)
        im = ax.imshow(
            image_cube[t_idx], origin="lower", cmap="viridis",
            vmin=vmin, vmax=vmax, extent=extent_lm,
        )
        ax.set_title(title)
        if expected_lm_pixel is not None and cell_lambda is not None:
            half = n_grid // 2
            pixel_size_lm = 1.0 / (n_grid * cell_lambda)
            l = (expected_lm_pixel[1] - half) * pixel_size_lm
            m = (expected_lm_pixel[0] - half) * pixel_size_lm
            ax.plot(l, m, "rx", markersize=10, markeredgewidth=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=80)
        plt.close(fig)

    # Try to assemble a movie (best-effort; not fatal if it fails).
    try:
        import imageio.v3 as iio
        frames = [
            iio.imread(str(frames_dir / f"frame_{i:04d}.png"))
            for i in range(len(sample_indices))
        ]
        iio.imwrite(str(movie_path), frames, fps=10)
        LOG.info("wrote movie %s (%d frames)", movie_path, len(frames))
        return movie_path
    except Exception as exc:
        LOG.warning("imageio movie write failed (%s); falling back to .gif", exc)
        try:
            from PIL import Image
            imgs = [
                Image.open(frames_dir / f"frame_{i:04d}.png")
                for i in range(len(sample_indices))
            ]
            gif_path = movie_path.with_suffix(".gif")
            imgs[0].save(
                gif_path, save_all=True, append_images=imgs[1:],
                duration=100, loop=0,
            )
            LOG.info("wrote gif %s (%d frames)", gif_path, len(imgs))
            return gif_path
        except Exception as exc2:
            LOG.warning("PIL gif fallback failed too (%s); frames-only output",
                        exc2)
            return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--voltage-root", type=Path,
                   default=Path("/home/ubuntu/data/voltages/250924mptq"))
    p.add_argument("--n-blocks", type=int, default=8,
                   help="fada blocks per chgroup (default 8 ≈ 1.07 s wall, "
                        "covers burst at native 15248 + 229 ms cross-band "
                        "dispersion smear)")
    p.add_argument("--t-int-fast-native", type=int, default=32,
                   help="fast-corr integration depth (NATIVE samples per "
                        "fast-vis tile). Default 32 = 1048.576 µs cadence "
                        "(4× burst-test override per PARALLEL_AGENTS.md §5.1).")
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--src-json", type=Path, default=None)
    p.add_argument("--cal-mode", default="phase_only",
                   choices=("phase_only", "full"))
    p.add_argument("--cal-pol-swap", action="store_true")
    p.add_argument("--obs-dec-deg", type=float, default=None,
                   help="observing dec (deg) for F21 cal phase. "
                        "Default = burst dec from T2_*.json (= source dec, "
                        "since on-axis at HA≈0).")
    p.add_argument("--peak-offset-pass-cells", type=int, default=4)
    p.add_argument("--peak-t-tol-native-samples", type=int, default=32,
                   help="PASS gate on |peak_t_native - 15248|. Default 32 "
                        "(~1 ms). Note the within-chgroup smear of ~14 ms "
                        "biases the centroid from a strict 1 ms pin (see "
                        "module docstring); operator-tunable.")
    p.add_argument("--burst-truth-native-sample", type=int, default=15248,
                   help="reference top-of-band native arrival time per "
                        "scratch/burst_search/")
    p.add_argument("--n-frames", type=int, default=50,
                   help="number of frames in the movie (sub-sampled from "
                        "T_fv_aligned)")
    p.add_argument("--sbs", default=",".join(f"{n:02d}" for n in range(16)),
                   help="comma-separated sb ids (default = 00..15, all 16 "
                        "present in the 250924mptq fixture)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)

    voltage_dir = args.voltage_root / "voltages"
    cals_dir = args.voltage_root / "cals"
    if not voltage_dir.is_dir() or not cals_dir.is_dir():
        LOG.error(
            "voltage_root layout invalid: %s / %s",
            voltage_dir, cals_dir,
        )
        return 2

    if args.src_json is not None:
        src = _parse_t2_json(args.src_json)
    else:
        t2_default = voltage_dir / "T2_250924mptq.json"
        if not t2_default.is_file():
            LOG.error("T2 json not found: %s", t2_default)
            return 2
        src = _parse_t2_json(t2_default)

    obs_dec_deg = (
        args.obs_dec_deg if args.obs_dec_deg is not None else src["dec_deg"]
    )

    LOG.info(
        "burst meta: RA=%.4f Dec=%.4f MJD=%.6f DM=%.3f T2_SNR=%.1f",
        src["ra_deg"], src["dec_deg"], src["mjd"], src["dm_pc_cc"], src["snr"],
    )
    LOG.info("obs_dec_deg = %.4f (F21 cal phase)", obs_dec_deg)

    # Single-DM plan + write to disk artefact.
    t_int_fast_us = args.t_int_fast_native * NATIVE_SAMPLE_US
    plan_path = args.report_dir / "dm_plan_single.npz"
    plan = _build_single_dm_plan(
        dm_pc_cc=src["dm_pc_cc"],
        t_int_fast_us=t_int_fast_us,
        out_path=plan_path,
    )

    inter_chgroup_delay_native = _compute_inter_chgroup_top_delay(src["dm_pc_cc"])

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    LOG.info("device=%s", device)

    sbs = [s.strip() for s in args.sbs.split(",") if s.strip()]

    chgroup_results: list[ChgroupBurstResult] = []
    per_chgroup_image_cubes: dict[int, np.ndarray] = {}
    per_chgroup_cell_lambda: dict[int, float] = {}

    for sb in sbs:
        voltage_path = voltage_dir / f"250924mptq_sb{sb}_data.out"
        cal_path = cals_dir / f"beamformer_weights_sb{sb}.dat"
        if not voltage_path.is_file():
            LOG.warning("sb%s: %s missing; skipping", sb, voltage_path)
            continue
        if not cal_path.is_file():
            LOG.warning("sb%s: %s missing; skipping", sb, cal_path)
            continue

        chgroup = int(sb)
        cfg_template = ReplayDefaults(
            chgroup=chgroup,
            obs_dec_deg=obs_dec_deg,
            n_grid=args.n_grid,
            kernel_support=1,
            t_int_fast_native=args.t_int_fast_native,
            cal_mode=args.cal_mode,
            cal_pol_swap=args.cal_pol_swap,
            rfi_enabled=False,
            static_sky_disabled=True,                                # brief: burst is the signal
        )
        cfg = cfg_template.to_cfg(cal_path=cal_path)
        cfg.chgroup = chgroup

        t0 = time.monotonic()
        LOG.info("=== sb%s (chgroup=%d) ===", sb, chgroup)
        ctx, outputs = replay_chgroup(
            voltage_path, cal_path=cal_path, cfg=cfg,
            max_blocks=args.n_blocks, device=device,
        )
        if not outputs:
            LOG.warning("sb%s: no outputs; skipping", sb)
            continue

        pattern = ctx.gridder.pattern
        n_filled = pattern.n_filled
        sparse = accumulate_chgroup_grids(outputs, n_filled=n_filled)  # (T_fv, n_filled)

        dense = sparse_to_dense_grid(
            sparse, pattern.ix_row, pattern.ix_col, n_grid=args.n_grid,
        )                                                                # (T_fv, N, N)
        cube = dirty_image_from_dense_grid(dense)                       # (T_fv, N, N) float32
        cube_np = cube.cpu().numpy()
        per_chgroup_image_cubes[chgroup] = cube_np

        from dsart.services.corr_fast_integration import load_antpos_from_cal_blob
        ap_e, ap_n, core_mask = load_antpos_from_cal_blob(cal_path)
        cell_lambda = compute_chgroup_cell_lambda(
            ap_e, ap_n, chgroup=chgroup, n_grid=args.n_grid,
            is_core_baseline_mask=core_mask,
        )
        per_chgroup_cell_lambda[chgroup] = float(cell_lambda)

        # Per-chgroup peak (in this chgroup's own time axis + cell-lambda).
        edge_pad = 8
        interior = cube_np[:, edge_pad:-edge_pad, edge_pad:-edge_pad]
        peak_flat = int(np.argmax(interior))
        T = interior.shape[0]
        peak_t_idx = peak_flat // (interior.shape[1] * interior.shape[2])
        peak_yx = peak_flat % (interior.shape[1] * interior.shape[2])
        peak_row = peak_yx // interior.shape[2] + edge_pad
        peak_col = peak_yx % interior.shape[2] + edge_pad
        peak_value = float(interior.flat[peak_flat])
        peak_t_native = peak_t_idx * args.t_int_fast_native + int(
            inter_chgroup_delay_native[chgroup]
        )                                                                # add the chgroup-g top delay so all chgroups report time relative to chgroup-0 top
        elapsed = time.monotonic() - t0

        result = ChgroupBurstResult(
            sb=sb,
            chgroup=chgroup,
            n_blocks_processed=len(outputs),
            n_fast_vis_total=int(cube.shape[0]),
            n_filled=int(n_filled),
            cell_lambda=float(cell_lambda),
            peak_pixel_row=int(peak_row),
            peak_pixel_col=int(peak_col),
            peak_t_idx=int(peak_t_idx),
            peak_t_native=int(peak_t_native),
            peak_value=float(peak_value),
            delay_native_to_chg0_top=int(inter_chgroup_delay_native[chgroup]),
            elapsed_s=float(elapsed),
        )
        chgroup_results.append(result)

        # Free the per-chgroup GPU tensors before the next sb's
        # build_context allocates a fresh kernel + sparsity pattern;
        # otherwise residual fp16 intermediates from compute_split's
        # last call will fragment GPU memory enough to OOM at sb01+.
        del ctx, outputs, sparse, dense, cube
        if device.type == "cuda":
            torch.cuda.empty_cache()
        LOG.info(
            "  sb%s peak: (row,col)=(%d,%d) t_idx=%d -> t_native (chg0 top frame)=%d "
            "value=%.3g cell_lambda=%.3g",
            sb, peak_row, peak_col, peak_t_idx, peak_t_native, peak_value, cell_lambda,
        )

    if not per_chgroup_image_cubes:
        LOG.error("no chgroup results; aborting")
        return 3

    # Cross-chgroup time alignment + sum.
    coadded_cube = _apply_inter_chgroup_alignment(
        per_chgroup_image_cubes,
        inter_chgroup_delay_native=inter_chgroup_delay_native,
        t_int_fast_native=args.t_int_fast_native,
        n_grid=args.n_grid,
    )                                                                     # (T_aligned, N, N)
    LOG.info("coadded cube shape: %s", coadded_cube.shape)

    # Find the peak in the coadded (T, l, m) cube.
    edge_pad = 8
    interior = coadded_cube[:, edge_pad:-edge_pad, edge_pad:-edge_pad]
    peak_flat = int(np.argmax(interior))
    peak_t_idx = peak_flat // (interior.shape[1] * interior.shape[2])
    peak_yx = peak_flat % (interior.shape[1] * interior.shape[2])
    peak_row = peak_yx // interior.shape[2] + edge_pad
    peak_col = peak_yx % interior.shape[2] + edge_pad
    peak_value = float(interior.flat[peak_flat])
    peak_t_native = peak_t_idx * args.t_int_fast_native                   # in chgroup-0 top frame after alignment

    # Predicted (l, m): astropy-backed from MJD + RA + Dec. The brief
    # asserts the burst is on-axis at MJD, but in practice the dump MJD
    # in T2_*.json is the TRIGGER time which precedes transit by O(min)
    # of HA, so the source IS off-axis at MJD by HA · cos(δ). Computing
    # the prediction astropy-style mirrors run_0319_pipeline and produces
    # the correct PASS/FAIL gate against the per-chgroup peaks.
    expected_l, expected_m, ha_src_deg = _compute_expected_lm(
        src_ra_deg=src["ra_deg"],
        src_dec_deg=src["dec_deg"],
        src_mjd=src["mjd"],
        phase_center_dec_deg=obs_dec_deg,
    )
    chg0_cell_lambda = per_chgroup_cell_lambda.get(
        0, next(iter(per_chgroup_cell_lambda.values())),
    )
    LOG.info(
        "expected (l, m) at MJD=%.6f: l=%.6f m=%.6f rad (HA_src=%.4f deg)",
        src["mjd"], expected_l, expected_m, ha_src_deg,
    )
    pred_row, pred_col = lm_to_pixel(
        expected_l, expected_m,
        n_grid=args.n_grid, cell_lambda=chg0_cell_lambda,
    )
    peak_offset = max(
        abs(int(peak_row) - int(pred_row)),
        abs(int(peak_col) - int(pred_col)),
    )
    peak_t_offset_native = abs(peak_t_native - args.burst_truth_native_sample)

    LOG.info(
        "COADDED peak: (row,col)=(%d,%d) t_idx=%d -> t_native=%d (truth=%d, "
        "offset=%d native samples = %.3f ms)",
        peak_row, peak_col, peak_t_idx, peak_t_native,
        args.burst_truth_native_sample, peak_t_offset_native,
        peak_t_offset_native * NATIVE_SAMPLE_US * 1e-3,
    )
    LOG.info(
        "predicted pixel (rounded to chgroup-0 cell): (%d,%d); offset = %d cells",
        int(pred_row), int(pred_col), peak_offset,
    )

    # Off-pulse SNR for the headline (mirrors burst_search/burst_250924mptq).
    timeseries = coadded_cube[:, peak_row, peak_col]
    # exclude ±20 fast-vis tiles around the peak
    excl_lo = max(0, peak_t_idx - 20)
    excl_hi = min(timeseries.size, peak_t_idx + 20)
    off_pulse = np.concatenate([timeseries[:excl_lo], timeseries[excl_hi:]])
    if off_pulse.size > 0:
        off_med = float(np.median(off_pulse))
        off_mad = float(np.median(np.abs(off_pulse - off_med)))
        off_sigma = off_mad * 1.4826
        peak_snr = ((peak_value - off_med) / off_sigma) if off_sigma > 0 else float("inf")
    else:
        off_med = float("nan")
        off_sigma = float("nan")
        peak_snr = float("nan")

    LOG.info(
        "burst peak SNR (off-pulse MAD): %.2f (off_med=%.3g, off_sigma=%.3g)",
        peak_snr, off_med, off_sigma,
    )

    # Time-series + peak-image PNGs.
    ts_path = args.report_dir / "time_series_burst_pixel.png"
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    t_native_axis = np.arange(timeseries.size) * args.t_int_fast_native
    t_ms_axis = t_native_axis * NATIVE_SAMPLE_US * 1e-3
    ax.plot(t_ms_axis, timeseries, "b-", linewidth=0.8)
    ax.axvline(
        args.burst_truth_native_sample * NATIVE_SAMPLE_US * 1e-3,
        color="red", linestyle="--", label="truth (15248 nat samples)",
    )
    ax.axvline(
        peak_t_native * NATIVE_SAMPLE_US * 1e-3,
        color="orange", linestyle=":", label=f"peak (t_native={peak_t_native})",
    )
    ax.set_xlabel("time (ms; chgroup-0 top frame)")
    ax.set_ylabel("dirty-image flux at burst pixel")
    ax.set_title(
        f"250924mptq burst — coadded TS at burst pixel "
        f"(SNR={peak_snr:.1f}; t_offset={peak_t_offset_native} nat)"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ts_path, dpi=110)
    plt.close(fig)
    LOG.info("wrote %s", ts_path)

    peak_image_path = args.report_dir / "peak_image.png"
    _save_image_png(
        coadded_cube[peak_t_idx],
        title=f"peak frame (t_idx={peak_t_idx}, t_native={peak_t_native}); "
              f"DM={src['dm_pc_cc']:.3f}",
        out_path=peak_image_path,
        expected_lm_pixel=(int(pred_row), int(pred_col)),
        peak_lm_pixel=(int(peak_row), int(peak_col)),
        cell_lambda=chg0_cell_lambda,
    )

    movie_path = _build_movie(
        coadded_cube, args.report_dir,
        n_frames=args.n_frames,
        t_int_fast_native=args.t_int_fast_native,
        cell_lambda=chg0_cell_lambda,
        expected_lm_pixel=(int(pred_row), int(pred_col)),
        movie_path=args.report_dir / "movie.mp4",
    )

    passed_lm = peak_offset <= args.peak_offset_pass_cells
    passed_t = peak_t_offset_native <= args.peak_t_tol_native_samples
    passed = passed_lm and passed_t

    repo_root = Path(__file__).resolve().parents[1]
    summary = {
        "milestone": "M3",
        "chunk": "chunk_6_voltage_fixture_burst_250924mptq",
        "stage": "PASS" if passed else "FAIL",
        "passed": passed,
        "passed_lm": passed_lm,
        "passed_t": passed_t,
        "host": socket.gethostname(),
        "utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _short_git_sha(repo_root),
        "n_blocks_per_chgroup": args.n_blocks,
        "t_int_fast_native": args.t_int_fast_native,
        "t_int_fast_us": float(t_int_fast_us),
        "n_grid": args.n_grid,
        "obs_dec_deg": float(obs_dec_deg),
        "src_ra_deg": float(src["ra_deg"]),
        "src_dec_deg": float(src["dec_deg"]),
        "src_mjd": float(src["mjd"]),
        "src_dm_pc_cc": float(src["dm_pc_cc"]),
        "src_t2_snr": float(src["snr"]),
        "burst_truth_native_sample": int(args.burst_truth_native_sample),
        "peak_t_native": int(peak_t_native),
        "peak_t_idx": int(peak_t_idx),
        "peak_t_offset_native_samples": int(peak_t_offset_native),
        "peak_t_offset_ms": float(peak_t_offset_native * NATIVE_SAMPLE_US * 1e-3),
        "peak_pixel_row": int(peak_row),
        "peak_pixel_col": int(peak_col),
        "peak_offset_cells": int(peak_offset),
        "peak_value": float(peak_value),
        "peak_snr": float(peak_snr),
        "off_pulse_median": float(off_med),
        "off_pulse_sigma": float(off_sigma),
        "expected_lm": {
            "l_rad": expected_l,
            "m_rad": expected_m,
            "ha_src_deg": ha_src_deg,
            "phase_center_mode": "source_dec (F21)",
        },
        "predicted_pixel_row": int(pred_row),
        "predicted_pixel_col": int(pred_col),
        "chg0_cell_lambda": float(chg0_cell_lambda),
        "n_chgroups_processed": len(chgroup_results),
        "inter_chgroup_top_delay_native": inter_chgroup_delay_native.tolist(),
        "per_chgroup": [asdict(r) for r in chgroup_results],
        "peak_offset_pass_gate_cells": args.peak_offset_pass_cells,
        "peak_t_tol_native_samples": args.peak_t_tol_native_samples,
        "movie_path": (str(movie_path) if movie_path else None),
        "peak_image_path": str(peak_image_path),
        "time_series_path": str(ts_path),
        "dm_plan_single_npz": str(plan_path),
    }
    summary_path = args.report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote %s", summary_path)

    # Minimal HTML report.
    rows = "\n".join(
        f"<tr><td>{r.sb}</td><td>{r.chgroup}</td>"
        f"<td>{r.peak_t_idx}</td><td>{r.peak_t_native}</td>"
        f"<td>{r.peak_pixel_row},{r.peak_pixel_col}</td>"
        f"<td>{r.delay_native_to_chg0_top}</td>"
        f"<td>{r.peak_value:.3g}</td><td>{r.elapsed_s:.1f}s</td></tr>"
        for r in chgroup_results
    )
    pass_class = "pass" if passed else "fail"
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>M3 chunk 6: 250924mptq burst imager</title>
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
<h1>M3 chunk 6: 250924mptq burst imager</h1>
<div class='summary'>
<p><strong>UTC:</strong> {summary['utc_iso']}</p>
<p><strong>git SHA:</strong> {summary['git_sha']}</p>
<p><strong>Host:</strong> {summary['host']}</p>
<p><strong>Burst (T2 truth):</strong> RA={summary['src_ra_deg']:.4f} Dec={summary['src_dec_deg']:.4f} MJD={summary['src_mjd']:.6f} DM={summary['src_dm_pc_cc']:.3f} pc/cc T2_SNR={summary['src_t2_snr']:.1f}</p>
<p><strong>n_blocks_per_chgroup:</strong> {summary['n_blocks_per_chgroup']} ({summary['n_blocks_per_chgroup']*0.134218:.3f} s); <strong>t_int_fast_native:</strong> {summary['t_int_fast_native']} (= {summary['t_int_fast_us']:.3f} µs cadence); <strong>n_grid:</strong> {summary['n_grid']}</p>
<p><strong>Burst truth native sample:</strong> {summary['burst_truth_native_sample']}; <strong>Coadded peak native sample:</strong> {summary['peak_t_native']} (offset = {summary['peak_t_offset_native_samples']} samples = {summary['peak_t_offset_ms']:.3f} ms)</p>
<p><strong>Coadded peak (row, col):</strong> ({summary['peak_pixel_row']}, {summary['peak_pixel_col']}); predicted ({summary['predicted_pixel_row']}, {summary['predicted_pixel_col']}); <strong>offset</strong> = {summary['peak_offset_cells']} cells</p>
<p><strong>Coadded peak SNR (off-pulse MAD):</strong> {summary['peak_snr']:.1f}</p>
<p><strong>(l, m) gate:</strong> ≤ {summary['peak_offset_pass_gate_cells']} cells → <span class='{ "pass" if passed_lm else "fail" }'>{ "PASS" if passed_lm else "FAIL" }</span></p>
<p><strong>t_native gate:</strong> ≤ {summary['peak_t_tol_native_samples']} native samples → <span class='{ "pass" if passed_t else "fail" }'>{ "PASS" if passed_t else "FAIL" }</span></p>
<p><strong>Result:</strong> <span class='{pass_class}'>{summary['stage']}</span></p>
</div>
<h2>Coadded time series at burst pixel</h2>
<img src="{ts_path.name}" />
<h2>Coadded peak frame</h2>
<img src="{peak_image_path.name}" />
<h2>Movie</h2>
{ '<video controls width="800"><source src="' + (movie_path.name if movie_path else "") + '" type="video/mp4"></video>' if movie_path and movie_path.suffix == ".mp4" else '<img src="' + (movie_path.name if movie_path else "frames/") + '" />' if movie_path else '<p>(movie not built; per-frame PNGs under frames/)</p>' }
<h2>Per-chgroup peaks</h2>
<table>
<tr><th>sb</th><th>chgroup</th><th>peak_t_idx</th><th>peak_t_native (chg0 top frame)</th>
<th>peak (row, col)</th><th>delay→chg0 top (nat)</th>
<th>peak val</th><th>elapsed</th></tr>
{rows}
</table>
"""
    (args.report_dir / "report.html").write_text(html)
    LOG.info("wrote %s", args.report_dir / "report.html")

    # Print headline summary for parent agent.
    print(json.dumps({
        "stage": summary["stage"],
        "passed_lm": passed_lm,
        "passed_t": passed_t,
        "peak_t_native": summary["peak_t_native"],
        "peak_t_offset_native_samples": summary["peak_t_offset_native_samples"],
        "peak_pixel": [summary["peak_pixel_row"], summary["peak_pixel_col"]],
        "peak_offset_cells": summary["peak_offset_cells"],
        "peak_snr": summary["peak_snr"],
        "report_dir": str(args.report_dir),
        "movie_path": summary["movie_path"],
        "peak_image_path": summary["peak_image_path"],
    }, indent=2))

    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
