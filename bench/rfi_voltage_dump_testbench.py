#!/usr/bin/env python3
"""bench/rfi_voltage_dump_testbench.py — M7.6 RFI testbench on captured voltages.

Runs the production RFI flagger (:class:`dsart.rfi.RFIFlagger`) on a
SNAP voltage dump and emits PNG diagnostics showing, for one sub-band:

* the **input statistic** each detector consumes (SK statistic per M,
  bandpass auto-power, per-ant in-band power, sum-threshold input mask)
* the **flag mask** each detector ends up emitting
* per-detector flag-fraction time-series across the dump

Intended for the M7.6 RFI tuning campaign: operators sweep the CLI
knobs (``--sk-far``, ``--bandpass-k``, ``--group-k``, ``--sumthr-*``,
``--m-values``, ``--flagants``) against real captured RFI conditions
and visually verify that the flags fire where they should.

Voltage-dump format (per sub-band, e.g. ``~/data/voltages/0319/
voltages/0319bbb_sb06_data.out``):

* raw int4-complex bytes laid out as the fada page
  ``[NPACKETS_PER_BLOCK=2048, NANTS=96, NCHAN_PER_CHGROUP=384,
  NTIMES_PER_PACKET=2, NPOL=2]`` = 301,989,888 bytes per cube
* no header; verified n01:~/data/voltages — file size is an exact
  multiple of the per-cube byte count.

Pipeline (per cube):

    raw bytes -> unpack_int4_split -> (real, imag) fp16 GEMM layout
              -> compute_autos (S1, S2 per M)
              -> RFIFlagger.flag_block (warmup-aware)
              + per-detector helpers (SK / bandpass / group / sumthr /
                flagants) re-invoked to capture each detector's
                input statistic and individual mask for plotting

Each cube is 4096 native time samples = 134.218 ms of voltages. A 15-
cube file (4,529,848,320 bytes) is ~2 s of voltage data. The default
``--warmup-cubes`` is ``n_cubes // 2``; only the second half of cubes
contributes to plots (matching the cold-start convention used in
production).

CLI knobs exposed (all wired into ``RFIFlagger`` / ``compute_autos``
exactly as the live ``corr_fast_compute`` service uses them):

* ``--sk-far``                — SK two-sided per-(M, cell) FAR
* ``--bandpass-k``            — bandpass-outlier MAD-σ threshold
* ``--group-k``               — group-outlier MAD-σ threshold
* ``--sumthr-max-m``          — sum-threshold max window (power of 2)
* ``--sumthr-eta``            — sum-threshold shape parameter
* ``--m-values 64,256,...``   — SK accumulation depths
* ``--flagants <path>``       — static legacy flagants.dat overlay
* ``--rfi-disabled``          — disable entire flagger (sanity check)
* ``--sumthr-disabled``       — disable sumthr post-pass only
* ``--warmup-cubes``          — cold-start window length

Outputs to ``--out-dir``:

* ``sk_diagnostic.png``            — SK per-M (ant × ch) + flag-frac map
* ``bandpass_diagnostic.png``      — mean S1_4096 spectrum, sample
                                     ant traces with median/threshold,
                                     bandpass-outlier mask fraction
* ``group_diagnostic.png``         — per-ant in-band power bars vs pop
                                     median + threshold, per-cube
                                     flagged-ant matrix
* ``sumthr_diagnostic.png``        — base (SK ∪ bandpass) mask vs
                                     dilated mask (added cells)
* ``source_tags_diagnostic.png``   — per-detector flag fraction maps
                                     + final OR-fold mask
* ``flag_fraction_timeseries.png`` — per-detector flag fraction per cube
* ``summary.txt``                  — text summary
* ``per_cube_flagfrac.csv``        — machine-readable per-cube counts

Run:

    python -m bench.rfi_voltage_dump_testbench \\
        --voltage-file ~/data/voltages/0319/voltages/0319bbb_sb06_data.out \\
        --out-dir /tmp/rfi-bench-0319-sb06 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    RFI_BANDPASS_WARMUP_CUBES_DEFAULT,
)
from dsart.rfi import (  # noqa: E402
    DEFAULT_BANDPASS_K,
    DEFAULT_ETA,
    DEFAULT_GROUP_K,
    DEFAULT_M_VALUES,
    DEFAULT_MAX_M,
    DEFAULT_SK_FAR,
    FlagSourceBit,
    RFIFlagger,
    bandpass_outlier_mask,
    compute_autos,
    compute_sk,
    group_outlier_mask,
    sk_combined_mask,
    sk_thresholds,
    sum_threshold_1d,
)
from dsart.rfi.bandpass_outlier import MAD_TO_SIGMA  # noqa: E402
from dsart.rfi.combine import _flagants_to_cube  # noqa: E402
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
    unpack_int4_split,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bytes per voltage cube on disk = fada page size:
# NPACKETS * NANTS * NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL
CUBE_BYTES: int = (
    NPACKETS_PER_BLOCK * NANTS * NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL
)
assert CUBE_BYTES == 301_989_888, CUBE_BYTES

LOG = logging.getLogger("rfi-testbench")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RFI flagger testbench on captured SNAP voltage dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--voltage-file", type=Path, required=True,
        help="Path to a single-sub-band voltage dump (int4 fada layout)."
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for PNGs / summary.txt (created if missing).",
    )
    p.add_argument(
        "--cubes", type=int, default=None,
        help="Number of cubes to process from the start of the file. "
        "Default: all cubes that fit (file_size // CUBE_BYTES).",
    )
    p.add_argument(
        "--warmup-cubes", type=int, default=None,
        help="Cold-start warmup window. Default: cubes // 2 (bandpass-"
        "outlier bypassed for the first half of cubes; plots show only "
        "the second half).",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="torch device: 'cpu', 'cuda', 'cuda:N', or 'auto' (cuda:0 "
        "if available, else cpu).",
    )

    p.add_argument("--sk-far", type=float, default=DEFAULT_SK_FAR,
                   help="SK two-sided per-(M, cell) false-alarm rate.")
    p.add_argument("--bandpass-k", type=float, default=DEFAULT_BANDPASS_K,
                   help="Bandpass-outlier MAD-sigma threshold.")
    p.add_argument("--group-k", type=float, default=DEFAULT_GROUP_K,
                   help="Group-outlier MAD-sigma threshold (across ants).")
    p.add_argument("--sumthr-max-m", type=int, default=DEFAULT_MAX_M,
                   help="Sum-threshold maximum dilation window (power of 2).")
    p.add_argument("--sumthr-eta", type=float, default=DEFAULT_ETA,
                   help="Sum-threshold shape parameter eta (>= 1).")
    p.add_argument(
        "--m-values", type=str,
        default=",".join(str(m) for m in DEFAULT_M_VALUES),
        help="Comma-separated SK accumulation depths (divisors of 4096).",
    )
    p.add_argument("--flagants", type=Path, default=None,
                   help="Optional legacy flagants.dat path.")
    p.add_argument("--rfi-disabled", action="store_true",
                   help="Disable the entire flagger (mask = all-False).")
    p.add_argument("--sumthr-disabled", action="store_true",
                   help="Disable the SumThreshold post-pass only.")

    p.add_argument("--n-ant-traces", type=int, default=6,
                   help="Number of antennas to overlay in the bandpass "
                        "detail panel. Picks a mix of typical + outlier ants.")
    p.add_argument(
        "--repr-cube", type=str, default="last",
        choices=("first-post-warmup", "last", "median-flagfrac"),
        help="Which post-warmup cube to use for the per-cube detail "
        "panels (bandpass traces, sumthr base/dilated masks).",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip PNG generation (CSV / summary.txt still written). "
        "Useful for quick smoke tests on headless hosts.",
    )

    args = p.parse_args(argv)

    # Resolve --device=auto.
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Parse --m-values.
    args.m_values_parsed = tuple(int(s) for s in args.m_values.split(","))
    if not args.m_values_parsed:
        p.error("--m-values must be non-empty")
    return args


# ---------------------------------------------------------------------------
# Per-cube diagnostics container
# ---------------------------------------------------------------------------


@dataclass
class CubeDiagnostics:
    """Per-cube diagnostics captured for the post-warmup plots.

    All arrays kept on CPU as numpy. Per-cube cost: ~2 MB.
    """

    cube_idx: int
    warmup: bool
    # SK: dict[M] -> mean SK statistic over n_acc, shape (NANTS, NCHAN, NPOL),
    # float32.
    sk_mean: dict[int, np.ndarray] = field(default_factory=dict)
    # SK: dict[M] -> OR-of-n_acc mask, shape (NANTS, NCHAN, NPOL), bool.
    sk_mask_per_m: dict[int, np.ndarray] = field(default_factory=dict)
    # SK union mask across all M's, bool.
    sk_mask: np.ndarray | None = None
    # Bandpass: S1_4096 squeezed to (NANTS, NCHAN, NPOL), float32.
    s1_full: np.ndarray | None = None
    # Per-(ant, pol) median bandpass, shape (NANTS, NPOL), float32.
    bp_median: np.ndarray | None = None
    # Per-(ant, pol) MAD-sigma, shape (NANTS, NPOL), float32.
    bp_sigma: np.ndarray | None = None
    # Bandpass-outlier mask, bool.
    bp_mask: np.ndarray | None = None
    # Group: per-(ant, pol) in-band power mean, shape (NANTS, NPOL), float32.
    grp_ant_mean: np.ndarray | None = None
    # Group: pop median + sigma per pol, each shape (NPOL,), float32.
    grp_pop_median: np.ndarray | None = None
    grp_pop_sigma: np.ndarray | None = None
    # Group-outlier mask, bool.
    grp_mask: np.ndarray | None = None
    # Sum-threshold input (SK ∪ bandpass), bool.
    sumthr_input: np.ndarray | None = None
    # Sum-threshold dilated mask (the "added cells"), bool.
    sumthr_added: np.ndarray | None = None
    # flagants overlay mask, bool.
    fa_mask: np.ndarray | None = None
    # Final OR-folded mask, bool.
    final_mask: np.ndarray | None = None
    # Per-cell uint8 source-tag bitfield.
    source_tags: np.ndarray | None = None

    # Per-detector flag fractions.
    frac_sk: float = 0.0
    frac_bp: float = 0.0
    frac_grp: float = 0.0
    frac_sumthr_added: float = 0.0
    frac_fa: float = 0.0
    frac_final: float = 0.0


# ---------------------------------------------------------------------------
# Voltage-dump reader
# ---------------------------------------------------------------------------


def read_cube_bytes(fh, cube_idx: int) -> bytes:
    """Read one 288 MB cube's worth of int4 bytes from a voltage dump."""
    fh.seek(cube_idx * CUBE_BYTES)
    buf = fh.read(CUBE_BYTES)
    if len(buf) != CUBE_BYTES:
        raise IOError(
            f"short read at cube {cube_idx}: got {len(buf)} bytes, expected "
            f"{CUBE_BYTES}"
        )
    return buf


# ---------------------------------------------------------------------------
# Per-detector helpers — replicate combine.RFIFlagger.flag_block step-by-step
# so we can capture each detector's *input statistic* alongside its mask
# (the production flagger throws the intermediates away).
# ---------------------------------------------------------------------------


def _bandpass_median_mad(
    s1_full: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-derive the per-(ant, pol) median and MAD-sigma the
    bandpass-outlier detector uses (so we can plot them alongside the
    spectrum). Mirrors :func:`dsart.rfi.bandpass_outlier.bandpass_outlier_mask`
    exactly.
    """
    med = torch.median(s1_full, dim=1, keepdim=False).values  # (NANTS, NPOL)
    abs_dev = (s1_full - med.unsqueeze(1)).abs()
    mad = torch.median(abs_dev, dim=1, keepdim=False).values  # (NANTS, NPOL)
    sigma = MAD_TO_SIGMA * mad
    sigma = torch.where(sigma > eps, sigma, torch.full_like(sigma, float("nan")))
    return med, sigma


def _group_pop_median_mad(
    s1_full: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-derive the per-pol population median + MAD-sigma the group-
    outlier detector uses, along with the per-(ant, pol) in-band power.
    """
    ant_pol_mean = s1_full.mean(dim=1)                    # (NANTS, NPOL)
    pop_median = torch.median(ant_pol_mean, dim=0).values  # (NPOL,)
    abs_dev = (ant_pol_mean - pop_median.unsqueeze(0)).abs()
    pop_mad = torch.median(abs_dev, dim=0).values          # (NPOL,)
    pop_sigma = MAD_TO_SIGMA * pop_mad
    pop_sigma = torch.where(
        pop_sigma > eps, pop_sigma, torch.full_like(pop_sigma, float("nan")),
    )
    return ant_pol_mean, pop_median, pop_sigma


def _sk_mean_per_m(
    s1_per_m: dict[int, torch.Tensor],
    s2_per_m: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Per-M, average the SK statistic over the leading n_acc axis."""
    out: dict[int, torch.Tensor] = {}
    for m in s1_per_m:
        sk = compute_sk(s1_per_m[m], s2_per_m[m], m)  # (n_acc, NANTS, NCHAN, NPOL)
        out[m] = sk.mean(dim=0)                       # (NANTS, NCHAN, NPOL)
    return out


def _sk_mask_per_m(
    s1_per_m: dict[int, torch.Tensor],
    s2_per_m: dict[int, torch.Tensor],
    *,
    far: float,
) -> dict[int, torch.Tensor]:
    """Per-M, OR-of-n_acc SK mask (NANTS, NCHAN, NPOL) bool."""
    out: dict[int, torch.Tensor] = {}
    for m in s1_per_m:
        low, high = sk_thresholds(m, far=far)
        sk = compute_sk(s1_per_m[m], s2_per_m[m], m)
        mask = (sk < low) | (sk > high)
        out[m] = mask.any(dim=0)
    return out


# ---------------------------------------------------------------------------
# Single-cube flagging + diagnostics capture
# ---------------------------------------------------------------------------


def process_cube(
    raw_bytes: bytes,
    *,
    flagger: RFIFlagger,
    device: torch.device,
    sk_far: float,
    bandpass_k: float,
    group_k: float,
    sumthr_max_m: int,
    sumthr_eta: float,
    sumthr_enabled: bool,
    m_values: tuple[int, ...],
    cube_idx: int,
    capture_diagnostics: bool,
) -> CubeDiagnostics:
    """Run one cube through the production RFI flagger and capture
    detector-level intermediates.

    The flagger's stateful warmup counter is advanced regardless of
    whether we capture diagnostics — this keeps the warmup window
    aligned with the cube index.
    """
    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    t0 = time.perf_counter()
    real, imag = unpack_int4_split(raw_arr, device=device)
    t_unpack = time.perf_counter() - t0

    t0 = time.perf_counter()
    autos = compute_autos(
        real, imag, m_values=m_values,
        n_packets=NPACKETS_PER_BLOCK, n_times_per_packet=NTIMES_PER_PACKET,
    )
    t_autos = time.perf_counter() - t0

    # Drop fluffed voltages now — autos has everything we need.
    del real, imag

    # Use the production flagger to advance the warmup state and produce
    # the OR-folded mask + warmup flag. Reuse our pre-computed autos.
    t0 = time.perf_counter()
    result = flagger.flag_block(
        real=None, imag=None, autos_override=autos,
        n_packets=NPACKETS_PER_BLOCK, n_times_per_packet=NTIMES_PER_PACKET,
    )
    t_flag = time.perf_counter() - t0

    diag = CubeDiagnostics(cube_idx=cube_idx, warmup=result.warmup)
    diag.frac_final = result.flag_fraction_total

    # ------------------------------------------------------------------
    # Re-derive each detector's mask on the same autos the flagger just
    # consumed. We do this on every cube (including warmup) so the
    # flag-fraction time-series has a complete per-detector breakdown.
    # Cheap relative to compute_autos itself; we'd be paying ~4× this in
    # the production flagger anyway.
    # ------------------------------------------------------------------
    full_m = max(m_values)
    s1_full_t = autos.s1[full_m].squeeze(0)                # (NANTS, NCHAN, NPOL)

    # SK union mask across all M's; same call combine.py uses.
    sk_union = sk_combined_mask(autos.s1, autos.s2, far=sk_far)

    # Bandpass-outlier: bypassed during warmup, exactly mirroring
    # RFIFlagger.flag_block.
    if result.warmup:
        bp_mask = torch.zeros_like(sk_union)
    else:
        bp_mask = bandpass_outlier_mask(s1_full_t, k=bandpass_k)

    grp_mask = group_outlier_mask(s1_full_t, k=group_k)

    base = sk_union | bp_mask
    if sumthr_enabled:
        base_t = base.permute(0, 2, 1).contiguous()
        dilated_t = sum_threshold_1d(
            base_t, max_m=sumthr_max_m, eta=sumthr_eta,
        )
        dilated = dilated_t.permute(0, 2, 1).contiguous()
        sumthr_added = dilated & ~base
    else:
        sumthr_added = torch.zeros_like(base)

    n_ant_a, n_ch_a, n_pol_a = sk_union.shape
    fa_mask = _flagants_to_cube(
        flagger.flagants_mask,
        n_ant=n_ant_a, n_ch=n_ch_a, n_pol=n_pol_a,
    )

    final = sk_union | bp_mask | grp_mask | sumthr_added | fa_mask

    # Sanity-check: our recomputed `final` should match what the
    # production flagger emitted, byte-for-byte.
    if not torch.equal(final, result.mask):
        raise RuntimeError(
            f"cube {cube_idx}: recomputed final mask does not match "
            f"flagger.flag_block output (xor="
            f"{int((final ^ result.mask).sum().item())} cells); "
            "testbench / combine.py drift"
        )

    diag.frac_sk = float(sk_union.float().mean().item())
    diag.frac_bp = float(bp_mask.float().mean().item())
    diag.frac_grp = float(grp_mask.float().mean().item())
    diag.frac_sumthr_added = float(sumthr_added.float().mean().item())
    diag.frac_fa = float(fa_mask.float().mean().item())

    if not capture_diagnostics:
        LOG.debug(
            "cube %3d  warmup=%d  unpack=%.0fms  autos=%.0fms  flag=%.0fms  "
            "frac sk=%.3g bp=%.3g grp=%.3g st+=%.3g fa=%.3g final=%.3g  "
            "(no plot capture)",
            cube_idx, int(result.warmup),
            t_unpack * 1e3, t_autos * 1e3, t_flag * 1e3,
            diag.frac_sk, diag.frac_bp, diag.frac_grp,
            diag.frac_sumthr_added, diag.frac_fa, diag.frac_final,
        )
        return diag

    # ------------------------------------------------------------------
    # Post-warmup: also capture the *input statistics* each detector
    # consumes (SK per M, bandpass median/MAD, group ant-mean / pop
    # stats, base+dilated masks) so the PNG plotters can show "what the
    # detector saw" alongside "what it ended up flagging".
    # ------------------------------------------------------------------
    sk_mean = _sk_mean_per_m(autos.s1, autos.s2)
    sk_masks_per_m = _sk_mask_per_m(autos.s1, autos.s2, far=sk_far)
    bp_med, bp_sigma = _bandpass_median_mad(s1_full_t)
    grp_ant_pol, grp_pop_median, grp_pop_sigma = _group_pop_median_mad(s1_full_t)

    tags = torch.zeros_like(final, dtype=torch.uint8)
    tags |= sk_union.to(torch.uint8) * int(FlagSourceBit.SK)
    tags |= bp_mask.to(torch.uint8) * int(FlagSourceBit.BANDPASS_OUTLIER)
    tags |= grp_mask.to(torch.uint8) * int(FlagSourceBit.GROUP_OUTLIER)
    tags |= sumthr_added.to(torch.uint8) * int(FlagSourceBit.SUM_THRESHOLD)
    tags |= fa_mask.to(torch.uint8) * int(FlagSourceBit.FLAGANTS_DAT)

    def _np_bool(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _np_float(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().to(torch.float32).numpy()

    diag.sk_mean = {m: _np_float(v) for m, v in sk_mean.items()}
    diag.sk_mask_per_m = {m: _np_bool(v) for m, v in sk_masks_per_m.items()}
    diag.sk_mask = _np_bool(sk_union)
    diag.s1_full = _np_float(s1_full_t)
    diag.bp_median = _np_float(bp_med)
    diag.bp_sigma = _np_float(bp_sigma)
    diag.bp_mask = _np_bool(bp_mask)
    diag.grp_ant_mean = _np_float(grp_ant_pol)
    diag.grp_pop_median = _np_float(grp_pop_median)
    diag.grp_pop_sigma = _np_float(grp_pop_sigma)
    diag.grp_mask = _np_bool(grp_mask)
    diag.sumthr_input = _np_bool(base)
    diag.sumthr_added = _np_bool(sumthr_added)
    diag.fa_mask = _np_bool(fa_mask)
    diag.final_mask = _np_bool(final)
    diag.source_tags = tags.detach().cpu().numpy()

    LOG.debug(
        "cube %3d  warmup=%d  unpack=%.0fms  autos=%.0fms  flag=%.0fms  "
        "frac sk=%.3g bp=%.3g grp=%.3g st+=%.3g fa=%.3g final=%.3g",
        cube_idx, int(result.warmup),
        t_unpack * 1e3, t_autos * 1e3, t_flag * 1e3,
        diag.frac_sk, diag.frac_bp, diag.frac_grp,
        diag.frac_sumthr_added, diag.frac_fa, diag.frac_final,
    )
    return diag


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def plot_sk_diagnostic(
    diags: list[CubeDiagnostics],
    *,
    m_values: tuple[int, ...],
    sk_far: float,
    out_path: Path,
):
    _, plt = _import_matplotlib()
    n_m = len(m_values)
    fig, axes = plt.subplots(
        3, n_m, figsize=(4.5 * n_m, 11),
        gridspec_kw={"hspace": 0.45, "wspace": 0.3},
    )
    if n_m == 1:
        axes = axes[:, np.newaxis]

    # Pre-compute thresholds for the titles.
    thresholds = {m: sk_thresholds(int(m), sk_far) for m in m_values}

    # Per-M SK stats averaged over post-warmup cubes; we visualise pol 0 and
    # pol 1 separately (rows 0, 1), plus the mask fraction across cubes
    # (row 2, both pols OR'd).
    for col, m in enumerate(m_values):
        sk_stack = np.stack([d.sk_mean[m] for d in diags], axis=0)  # (Nc, ant, ch, pol)
        sk_avg = sk_stack.mean(axis=0)                              # (ant, ch, pol)

        # Symmetric diverging colormap centered at 1 with the SK threshold
        # range as the colour-saturation half-width.
        low, high = thresholds[m]
        sk_lo_dev = max(abs(1.0 - low), 1e-3)
        sk_hi_dev = max(abs(high - 1.0), 1e-3)
        half = max(sk_lo_dev, sk_hi_dev) * 2.0  # show ±2× threshold span
        vmin, vmax = 1.0 - half, 1.0 + half

        for pol in range(NPOL):
            ax = axes[pol, col]
            im = ax.imshow(
                sk_avg[..., pol], aspect="auto", origin="lower",
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
                extent=[0, sk_avg.shape[1], 0, sk_avg.shape[0]],
                interpolation="nearest",
            )
            ax.set_title(
                f"SK (mean over cubes,n_acc)  M={m}  pol={pol}\n"
                f"thresholds [{low:.3f}, {high:.3f}]",
                fontsize=9,
            )
            ax.set_xlabel("channel")
            ax.set_ylabel("ant")
            plt.colorbar(im, ax=ax, shrink=0.85)

        # Flag fraction across cubes for this M, both pols OR'd.
        sk_mask_stack = np.stack(
            [d.sk_mask_per_m[m].astype(np.float32) for d in diags], axis=0,
        )                                                     # (Nc, ant, ch, pol)
        sk_frac = sk_mask_stack.mean(axis=(0, 3))
        ax = axes[2, col]
        im2 = ax.imshow(
            sk_frac, aspect="auto", origin="lower",
            cmap="magma", vmin=0.0, vmax=1.0,
            extent=[0, sk_frac.shape[1], 0, sk_frac.shape[0]],
            interpolation="nearest",
        )
        ax.set_title(
            f"SK flag-fraction (cubes·pols)  M={m}\n"
            f"mean cell-frac flagged = {float(sk_frac.mean()):.3g}",
            fontsize=9,
        )
        ax.set_xlabel("channel")
        ax.set_ylabel("ant")
        plt.colorbar(im2, ax=ax, shrink=0.85)

    fig.suptitle(
        f"SK detector ({len(diags)} post-warmup cubes, FAR={sk_far:g})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def _pick_ant_traces(
    diags: list[CubeDiagnostics], n: int,
) -> list[int]:
    """Pick a representative set of ants for the bandpass detail panel:
    median ant + the most-frequently-flagged ants by bandpass-outlier.
    """
    bp_frac_per_ant = np.stack(
        [d.bp_mask.mean(axis=(1, 2)) for d in diags], axis=0,
    ).mean(axis=0)  # (NANTS,)
    order = np.argsort(-bp_frac_per_ant)   # descending
    top_flagged = list(order[:max(0, n - 1)])
    # Add a "typical" ant (median bp_frac).
    typical = int(np.argsort(bp_frac_per_ant)[bp_frac_per_ant.size // 2])
    picks = list(dict.fromkeys([typical] + top_flagged))[:n]
    return picks


def plot_bandpass_diagnostic(
    diags: list[CubeDiagnostics],
    *,
    bandpass_k: float,
    repr_cube_idx_local: int,
    n_ant_traces: int,
    out_path: Path,
):
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(
        3, NPOL, figsize=(7 * NPOL, 13),
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
    )

    s1_stack = np.stack([d.s1_full for d in diags], axis=0)  # (Nc, ant, ch, pol)
    s1_avg = s1_stack.mean(axis=0)                            # (ant, ch, pol)
    mask_stack = np.stack([d.bp_mask.astype(np.float32) for d in diags], axis=0)
    mask_frac = mask_stack.mean(axis=0)                       # (ant, ch, pol)

    for pol in range(NPOL):
        ax = axes[0, pol]
        # log10 power for visual range; clamp to a small positive floor.
        s1_log = np.log10(np.maximum(s1_avg[..., pol], 1e-6))
        im = ax.imshow(
            s1_log, aspect="auto", origin="lower", cmap="viridis",
            extent=[0, s1_log.shape[1], 0, s1_log.shape[0]],
            interpolation="nearest",
        )
        ax.set_title(
            f"mean log10(S1_4096) (over {len(diags)} cubes)  pol={pol}",
            fontsize=10,
        )
        ax.set_xlabel("channel")
        ax.set_ylabel("ant")
        plt.colorbar(im, ax=ax, shrink=0.85, label="log10(power)")

    # Row 1: per-ant traces from the representative cube, with median +
    # ±k·σ overlaid and bandpass-flagged channels marked.
    repr_diag = diags[repr_cube_idx_local]
    ant_picks = _pick_ant_traces(diags, n_ant_traces)
    for pol in range(NPOL):
        ax = axes[1, pol]
        n_chan = repr_diag.s1_full.shape[1]
        for ant in ant_picks:
            spec = repr_diag.s1_full[ant, :, pol]
            med = repr_diag.bp_median[ant, pol]
            sigma = repr_diag.bp_sigma[ant, pol]
            (line,) = ax.plot(
                np.arange(n_chan), spec, lw=0.8, alpha=0.8, label=f"ant {ant}",
            )
            colour = line.get_color()
            ax.axhline(med, color=colour, lw=0.6, alpha=0.4, linestyle=":")
            if np.isfinite(sigma):
                ax.axhline(med + bandpass_k * sigma, color=colour,
                           lw=0.6, alpha=0.3, linestyle="--")
                ax.axhline(med - bandpass_k * sigma, color=colour,
                           lw=0.6, alpha=0.3, linestyle="--")
            flagged_ch = np.where(repr_diag.bp_mask[ant, :, pol])[0]
            if flagged_ch.size:
                ax.scatter(
                    flagged_ch, spec[flagged_ch], marker="x",
                    color=colour, s=18, zorder=5,
                )
        ax.set_title(
            f"bandpass detail (cube #{repr_diag.cube_idx}) pol={pol}\n"
            f"dotted = per-ant median   dashed = median ± {bandpass_k:.1f}·σ_MAD"
            f"   × = bandpass-flagged ch",
            fontsize=9,
        )
        ax.set_xlabel("channel")
        ax.set_ylabel("power (scaled)")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_yscale("log")

    # Row 2: bandpass-flag fraction per cell.
    for pol in range(NPOL):
        ax = axes[2, pol]
        im = ax.imshow(
            mask_frac[..., pol], aspect="auto", origin="lower",
            cmap="magma", vmin=0.0, vmax=1.0,
            extent=[0, mask_frac.shape[1], 0, mask_frac.shape[0]],
            interpolation="nearest",
        )
        ax.set_title(
            f"bandpass flag fraction  pol={pol}  (cell-mean={float(mask_frac[..., pol].mean()):.3g})",
            fontsize=10,
        )
        ax.set_xlabel("channel")
        ax.set_ylabel("ant")
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"Bandpass-outlier detector ({len(diags)} post-warmup cubes, k={bandpass_k:g})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def plot_group_diagnostic(
    diags: list[CubeDiagnostics],
    *,
    group_k: float,
    out_path: Path,
):
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        gridspec_kw={"hspace": 0.45, "wspace": 0.3,
                     "height_ratios": [1.0, 1.0]},
    )

    # Mean per-(ant, pol) in-band power across post-warmup cubes.
    ant_means = np.stack([d.grp_ant_mean for d in diags], axis=0)  # (Nc, ant, pol)
    pop_medians = np.stack([d.grp_pop_median for d in diags], axis=0)  # (Nc, pol)
    pop_sigmas = np.stack([d.grp_pop_sigma for d in diags], axis=0)    # (Nc, pol)
    # Use the median across cubes for a stable threshold line.
    pop_median_avg = np.nanmedian(pop_medians, axis=0)             # (pol,)
    pop_sigma_avg = np.nanmedian(pop_sigmas, axis=0)               # (pol,)
    # Per-(ant, pol) "ever-flagged" fraction across cubes.
    grp_mask_stack = np.stack(
        [d.grp_mask.astype(np.float32) for d in diags], axis=0,
    )
    flagged_frac_ant_pol = grp_mask_stack.mean(axis=(0, 2))        # (ant, pol)

    n_ant = ant_means.shape[1]
    for pol in range(NPOL):
        ax = axes[0, pol]
        median_per_ant = np.nanmedian(ant_means[:, :, pol], axis=0)
        # Plot post-warmup-mean per-ant in-band power as a bar; colour bars
        # by ever-flagged fraction (red = always flagged).
        flag_frac = flagged_frac_ant_pol[:, pol]
        colors = []
        for f in flag_frac:
            if f >= 0.5:
                colors.append("#b71c1c")     # mostly flagged
            elif f > 0.0:
                colors.append("#f57c00")     # occasionally flagged
            else:
                colors.append("#1976d2")     # never flagged

        ax.bar(np.arange(n_ant), median_per_ant, color=colors)
        # Threshold band.
        mu = float(pop_median_avg[pol])
        sigma = float(pop_sigma_avg[pol]) if np.isfinite(pop_sigma_avg[pol]) else 0.0
        ax.axhline(mu, color="k", lw=1.0, label=f"pop median = {mu:.3g}")
        if sigma > 0:
            ax.axhspan(mu - group_k * sigma, mu + group_k * sigma,
                       color="green", alpha=0.10,
                       label=f"median ± {group_k:.1f}·σ_MAD")
            ax.axhline(mu + group_k * sigma, color="green",
                       lw=0.8, linestyle="--")
            ax.axhline(mu - group_k * sigma, color="green",
                       lw=0.8, linestyle="--")
        ax.set_title(
            f"per-ant in-band power (median over cubes) pol={pol}\n"
            f"red bars = group-flagged ≥ 50% of cubes, "
            f"orange = sometimes, blue = never",
            fontsize=9,
        )
        ax.set_xlabel("ant")
        ax.set_ylabel("mean S1 per ch (scaled)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_yscale("log")

        # Per-cube × per-ant flagged matrix.
        ax2 = axes[1, pol]
        cube_idxs = [d.cube_idx for d in diags]
        flagged_matrix = np.stack(
            [d.grp_mask[:, 0, pol].astype(np.uint8) for d in diags], axis=0,
        )  # (Nc, NANTS); group_mask is broadcast across ch so ch=0 is fine
        im = ax2.imshow(
            flagged_matrix.T, aspect="auto", origin="lower",
            cmap="Reds", vmin=0.0, vmax=1.0,
            extent=[cube_idxs[0] - 0.5, cube_idxs[-1] + 0.5, 0, n_ant],
            interpolation="nearest",
        )
        ax2.set_title(
            f"per-cube flagged-ant matrix  pol={pol}  "
            f"(total flagged frac = {float(flagged_matrix.mean()):.3g})",
            fontsize=9,
        )
        ax2.set_xlabel("cube idx")
        ax2.set_ylabel("ant")
        plt.colorbar(im, ax=ax2, shrink=0.85)

    fig.suptitle(
        f"Group-outlier detector ({len(diags)} post-warmup cubes, k={group_k:g})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def plot_sumthr_diagnostic(
    diags: list[CubeDiagnostics],
    *,
    sumthr_max_m: int,
    sumthr_eta: float,
    repr_cube_idx_local: int,
    out_path: Path,
):
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(
        3, NPOL, figsize=(7 * NPOL, 11),
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
    )

    # Stack base (sk ∪ bp) and added (post-dilation) masks across all cubes.
    base_stack = np.stack(
        [d.sumthr_input.astype(np.float32) for d in diags], axis=0,
    )                                                                  # (Nc, ant, ch, pol)
    added_stack = np.stack(
        [d.sumthr_added.astype(np.float32) for d in diags], axis=0,
    )
    base_frac = base_stack.mean(axis=0)                                # (ant, ch, pol)
    added_frac = added_stack.mean(axis=0)

    repr_diag = diags[repr_cube_idx_local]
    for pol in range(NPOL):
        ax0 = axes[0, pol]
        im = ax0.imshow(
            base_frac[..., pol], aspect="auto", origin="lower",
            cmap="magma", vmin=0.0, vmax=1.0,
            extent=[0, base_frac.shape[1], 0, base_frac.shape[0]],
            interpolation="nearest",
        )
        ax0.set_title(
            f"base mask (SK ∪ bandpass) flag fraction  pol={pol}\n"
            f"cell-mean = {float(base_frac[..., pol].mean()):.3g}",
            fontsize=9,
        )
        ax0.set_xlabel("channel")
        ax0.set_ylabel("ant")
        plt.colorbar(im, ax=ax0, shrink=0.85)

        ax1 = axes[1, pol]
        im = ax1.imshow(
            added_frac[..., pol], aspect="auto", origin="lower",
            cmap="Greens", vmin=0.0, vmax=1.0,
            extent=[0, added_frac.shape[1], 0, added_frac.shape[0]],
            interpolation="nearest",
        )
        ax1.set_title(
            f"cells *added* by SumThreshold dilation  pol={pol}\n"
            f"cell-mean added = {float(added_frac[..., pol].mean()):.3g}",
            fontsize=9,
        )
        ax1.set_xlabel("channel")
        ax1.set_ylabel("ant")
        plt.colorbar(im, ax=ax1, shrink=0.85)

        # Per-cube detail: base vs. base+added composite.
        ax2 = axes[2, pol]
        base_img = repr_diag.sumthr_input[..., pol].astype(np.float32)
        added_img = repr_diag.sumthr_added[..., pol].astype(np.float32)
        composite = np.zeros((*base_img.shape, 3), dtype=np.float32)
        composite[..., 0] = added_img        # red = added by sumthr
        composite[..., 2] = base_img         # blue = base (sk ∪ bp)
        ax2.imshow(
            composite, aspect="auto", origin="lower",
            extent=[0, base_img.shape[1], 0, base_img.shape[0]],
            interpolation="nearest",
        )
        ax2.set_title(
            f"cube #{repr_diag.cube_idx} detail  pol={pol}\n"
            f"blue = base (SK∪bp), red = added by sumthr",
            fontsize=9,
        )
        ax2.set_xlabel("channel")
        ax2.set_ylabel("ant")

    fig.suptitle(
        f"SumThreshold post-pass ({len(diags)} post-warmup cubes; "
        f"max_m={sumthr_max_m}, eta={sumthr_eta:g})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def plot_source_tags_diagnostic(
    diags: list[CubeDiagnostics],
    *,
    out_path: Path,
):
    _, plt = _import_matplotlib()
    detectors = [
        ("SK",                  "sk_mask",       "Reds"),
        ("Bandpass-outlier",    "bp_mask",       "Oranges"),
        ("Group-outlier",       "grp_mask",      "Purples"),
        ("SumThreshold added",  "sumthr_added",  "Greens"),
        ("flagants.dat",        "fa_mask",       "Greys"),
        ("OR-fold (final)",     "final_mask",    "magma"),
    ]
    n_rows = len(detectors)
    fig, axes = plt.subplots(
        n_rows, NPOL, figsize=(7 * NPOL, 3.2 * n_rows),
        gridspec_kw={"hspace": 0.55, "wspace": 0.25},
    )

    for r, (name, attr, cmap) in enumerate(detectors):
        stack = np.stack(
            [getattr(d, attr).astype(np.float32) for d in diags], axis=0,
        )                                                       # (Nc, ant, ch, pol)
        for pol in range(NPOL):
            ax = axes[r, pol]
            frac = stack[..., pol].mean(axis=0)                 # (ant, ch)
            im = ax.imshow(
                frac, aspect="auto", origin="lower",
                cmap=cmap, vmin=0.0, vmax=1.0,
                extent=[0, frac.shape[1], 0, frac.shape[0]],
                interpolation="nearest",
            )
            ax.set_title(
                f"{name}  pol={pol}  (cell-frac={float(frac.mean()):.3g})",
                fontsize=10,
            )
            ax.set_xlabel("channel")
            ax.set_ylabel("ant")
            plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"Per-detector flag-fraction maps ({len(diags)} post-warmup cubes)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def plot_flag_fraction_timeseries(
    all_diags: list[CubeDiagnostics],
    *,
    warmup_cubes: int,
    out_path: Path,
):
    _, plt = _import_matplotlib()
    cube_idx = np.array([d.cube_idx for d in all_diags])
    fracs = {
        "SK":         np.array([d.frac_sk for d in all_diags]),
        "Bandpass":   np.array([d.frac_bp for d in all_diags]),
        "Group":      np.array([d.frac_grp for d in all_diags]),
        "SumThr+":    np.array([d.frac_sumthr_added for d in all_diags]),
        "flagants":   np.array([d.frac_fa for d in all_diags]),
        "Final OR":   np.array([d.frac_final for d in all_diags]),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    style = {
        "SK":         {"color": "#e53935", "lw": 1.5, "marker": "o"},
        "Bandpass":   {"color": "#fb8c00", "lw": 1.2, "marker": "s"},
        "Group":      {"color": "#8e24aa", "lw": 1.2, "marker": "^"},
        "SumThr+":    {"color": "#43a047", "lw": 1.0, "marker": "v"},
        "flagants":   {"color": "#616161", "lw": 1.0, "marker": "D"},
        "Final OR":   {"color": "#1976d2", "lw": 2.2, "marker": None},
    }
    for name, vals in fracs.items():
        ax.plot(cube_idx, vals, label=name, **style[name])
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("cube idx")
    ax.set_ylabel("flag fraction (cells flagged / total)")
    ax.axvspan(-0.5, warmup_cubes - 0.5, color="grey", alpha=0.15,
               label="warmup window")
    ax.axvline(warmup_cubes - 0.5, color="k", lw=1.0, linestyle=":")
    ax.set_title(
        f"Per-detector flag fraction per cube ({len(all_diags)} cubes, "
        f"warmup={warmup_cubes})",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Summary / CSV
# ---------------------------------------------------------------------------


def write_per_cube_csv(
    all_diags: list[CubeDiagnostics],
    *,
    out_path: Path,
):
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "cube_idx", "warmup",
            "frac_sk", "frac_bandpass", "frac_group",
            "frac_sumthr_added", "frac_flagants", "frac_final",
        ])
        for d in all_diags:
            w.writerow([
                d.cube_idx, int(d.warmup),
                f"{d.frac_sk:.6g}", f"{d.frac_bp:.6g}", f"{d.frac_grp:.6g}",
                f"{d.frac_sumthr_added:.6g}", f"{d.frac_fa:.6g}",
                f"{d.frac_final:.6g}",
            ])
    LOG.info("wrote %s", out_path)


def write_summary(
    *,
    args: argparse.Namespace,
    n_cubes: int,
    warmup_cubes: int,
    post_warmup_diags: list[CubeDiagnostics],
    voltage_file: Path,
    out_path: Path,
):
    def _mean(attr: str) -> float:
        if not post_warmup_diags:
            return float("nan")
        return float(np.mean([getattr(d, attr) for d in post_warmup_diags]))

    lines = [
        "DSA-110 RFI testbench — voltage-dump run",
        "=" * 56,
        f"voltage file       : {voltage_file}",
        f"file size          : {voltage_file.stat().st_size:,} bytes",
        f"cubes processed    : {n_cubes}",
        f"warmup cubes       : {warmup_cubes}  (post-warmup={n_cubes - warmup_cubes})",
        f"device             : {args.device}",
        "",
        "Configuration",
        "-" * 56,
        f"sk_far             : {args.sk_far}",
        f"bandpass_k         : {args.bandpass_k}",
        f"group_k            : {args.group_k}",
        f"sumthr_max_m       : {args.sumthr_max_m}",
        f"sumthr_eta         : {args.sumthr_eta}",
        f"sumthr enabled     : {not args.sumthr_disabled}",
        f"m_values           : {args.m_values_parsed}",
        f"flagants           : {args.flagants}",
        f"rfi enabled        : {not args.rfi_disabled}",
        "",
        "Per-detector flag fractions (mean over post-warmup cubes)",
        "-" * 56,
        f"  SK                : {_mean('frac_sk'):.4g}",
        f"  Bandpass-outlier  : {_mean('frac_bp'):.4g}",
        f"  Group-outlier     : {_mean('frac_grp'):.4g}",
        f"  SumThr (added)    : {_mean('frac_sumthr_added'):.4g}",
        f"  flagants.dat      : {_mean('frac_fa'):.4g}",
        f"  Final OR (total)  : {_mean('frac_final'):.4g}",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pick_repr_cube(
    diags: list[CubeDiagnostics],
    choice: str,
) -> int:
    """Pick which post-warmup cube to use for the per-cube detail panels."""
    if choice == "first-post-warmup":
        return 0
    if choice == "last":
        return len(diags) - 1
    if choice == "median-flagfrac":
        order = np.argsort([d.frac_final for d in diags])
        return int(order[len(order) // 2])
    raise ValueError(f"unknown repr-cube choice {choice!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.voltage_file.exists():
        LOG.error("voltage file does not exist: %s", args.voltage_file)
        return 1
    file_size = args.voltage_file.stat().st_size
    n_cubes_in_file = file_size // CUBE_BYTES
    if n_cubes_in_file < 2:
        LOG.error(
            "voltage file %s only has %d cubes (< 2); need at least 2 "
            "for warmup + analysis",
            args.voltage_file, n_cubes_in_file,
        )
        return 1
    if file_size % CUBE_BYTES != 0:
        LOG.warning(
            "voltage file size %d is not an exact multiple of CUBE_BYTES=%d "
            "(remainder = %d); trailing bytes ignored",
            file_size, CUBE_BYTES, file_size % CUBE_BYTES,
        )

    n_cubes = args.cubes if args.cubes is not None else n_cubes_in_file
    n_cubes = min(n_cubes, n_cubes_in_file)
    warmup_cubes = (
        args.warmup_cubes if args.warmup_cubes is not None else n_cubes // 2
    )
    if not 0 <= warmup_cubes < n_cubes:
        LOG.error(
            "warmup_cubes=%d must satisfy 0 <= warmup_cubes < n_cubes=%d",
            warmup_cubes, n_cubes,
        )
        return 1
    if n_cubes - warmup_cubes < 1:
        LOG.error("post-warmup cube count must be >= 1")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    LOG.info("device=%s  cubes=%d  warmup=%d  file=%s",
             device, n_cubes, warmup_cubes, args.voltage_file)

    flagger = RFIFlagger(
        flagants_path=args.flagants,
        device=device,
        sk_far=args.sk_far,
        bandpass_k=args.bandpass_k,
        group_k=args.group_k,
        sum_threshold_max_m=args.sumthr_max_m,
        sum_threshold_eta=args.sumthr_eta,
        m_values=args.m_values_parsed,
        warmup_cubes=warmup_cubes,
        run_sum_threshold=not args.sumthr_disabled,
    )

    all_diags: list[CubeDiagnostics] = []
    post_warmup_diags: list[CubeDiagnostics] = []

    t_total0 = time.perf_counter()
    with args.voltage_file.open("rb") as fh:
        for cube_idx in range(n_cubes):
            buf = read_cube_bytes(fh, cube_idx)
            capture = cube_idx >= warmup_cubes and not args.rfi_disabled
            diag = process_cube(
                buf,
                flagger=flagger,
                device=device,
                sk_far=args.sk_far,
                bandpass_k=args.bandpass_k,
                group_k=args.group_k,
                sumthr_max_m=args.sumthr_max_m,
                sumthr_eta=args.sumthr_eta,
                sumthr_enabled=not args.sumthr_disabled,
                m_values=args.m_values_parsed,
                cube_idx=cube_idx,
                capture_diagnostics=capture,
            )
            all_diags.append(diag)
            if capture:
                post_warmup_diags.append(diag)
            LOG.info(
                "cube %3d/%d  warmup=%d  frac_final=%.3g",
                cube_idx, n_cubes - 1, int(diag.warmup), diag.frac_final,
            )

    t_total = time.perf_counter() - t_total0
    LOG.info(
        "processed %d cubes in %.2f s  (%.0f ms/cube)",
        n_cubes, t_total, t_total / max(n_cubes, 1) * 1e3,
    )

    if not post_warmup_diags:
        LOG.warning(
            "no post-warmup cubes captured (rfi_disabled=%s) — only CSV/summary "
            "will be written",
            args.rfi_disabled,
        )

    csv_path = args.out_dir / "per_cube_flagfrac.csv"
    write_per_cube_csv(all_diags, out_path=csv_path)

    summary_path = args.out_dir / "summary.txt"
    write_summary(
        args=args, n_cubes=n_cubes, warmup_cubes=warmup_cubes,
        post_warmup_diags=post_warmup_diags,
        voltage_file=args.voltage_file, out_path=summary_path,
    )

    if not args.no_plot and post_warmup_diags:
        repr_idx_local = _pick_repr_cube(post_warmup_diags, args.repr_cube)
        plot_sk_diagnostic(
            post_warmup_diags,
            m_values=args.m_values_parsed,
            sk_far=args.sk_far,
            out_path=args.out_dir / "sk_diagnostic.png",
        )
        plot_bandpass_diagnostic(
            post_warmup_diags,
            bandpass_k=args.bandpass_k,
            repr_cube_idx_local=repr_idx_local,
            n_ant_traces=args.n_ant_traces,
            out_path=args.out_dir / "bandpass_diagnostic.png",
        )
        plot_group_diagnostic(
            post_warmup_diags,
            group_k=args.group_k,
            out_path=args.out_dir / "group_diagnostic.png",
        )
        plot_sumthr_diagnostic(
            post_warmup_diags,
            sumthr_max_m=args.sumthr_max_m,
            sumthr_eta=args.sumthr_eta,
            repr_cube_idx_local=repr_idx_local,
            out_path=args.out_dir / "sumthr_diagnostic.png",
        )
        plot_source_tags_diagnostic(
            post_warmup_diags,
            out_path=args.out_dir / "source_tags_diagnostic.png",
        )
        plot_flag_fraction_timeseries(
            all_diags,
            warmup_cubes=warmup_cubes,
            out_path=args.out_dir / "flag_fraction_timeseries.png",
        )

    # Manifest for downstream tooling.
    manifest = {
        "voltage_file": str(args.voltage_file),
        "n_cubes": n_cubes,
        "warmup_cubes": warmup_cubes,
        "device": str(device),
        "config": {
            "sk_far": args.sk_far,
            "bandpass_k": args.bandpass_k,
            "group_k": args.group_k,
            "sumthr_max_m": args.sumthr_max_m,
            "sumthr_eta": args.sumthr_eta,
            "sumthr_enabled": not args.sumthr_disabled,
            "m_values": list(args.m_values_parsed),
            "flagants": str(args.flagants) if args.flagants else None,
            "rfi_enabled": not args.rfi_disabled,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    LOG.info("done; outputs in %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
