#!/usr/bin/env python3
"""Monte-Carlo simulation of FRB S/N loss vs DM given the production DM
plan (`configs/dm_plan.npz`).

Model (per user request, 2026-05-28):

  * The source is a **delta-function pulse** at intrinsic DM = DM_true.
  * Two broadening mechanisms set the *observed* width at the bottom of
    the band:

        w_chan    : intra-channel dispersion smearing at DM = DM_true,
                    evaluated at the bottom of the processed band
                    (worst case across the band — bottom channel
                    dominates).

        w_DMmiss  : inter-channel dispersion smearing across the full
                    processed band introduced by the residual DM
                    mismatch ΔDM = |DM_true − DM_trial^*|, where
                    DM_trial^* is the closest *fine* DM trial in the
                    plan.

  * The ideal width (matched DM) is

        w_ideal   = w_chan

  * The measured width is the quadrature sum

        w_meas    = sqrt( w_chan**2 + w_DMmiss**2 )

  * For a top-hat-broadened delta in white noise, the matched-filter
    S/N scales as 1/sqrt(N_samples_in_box) ∝ 1/sqrt(w).  The DM-plan
    S/N loss is therefore

        η = S/N_meas / S/N_ideal = sqrt(w_ideal / w_meas)

    We report the loss in two ways:

        loss_dB        = 10 * log10( η )         (negative — dB lost)
        loss_pct       = (1 - η) * 100           (positive — % lost)

Dispersion constant (matches src/dsart/common/constants.K_DM_MS_GHZ2_PC):

    K = 4.148808  ms · GHz**2 · pc**-1 · cm**3

Intra-channel smearing at frequency ν with channel width Δν (incoherent
dedispersion):

    w_chan(ν, DM) = | d τ / d ν | · Δν = 2 K DM Δν / ν**3       (eq. 1)

Inter-channel smearing across the full processed band given the residual
DM mismatch ΔDM:

    w_DMmiss(ΔDM) = K · ΔDM · ( 1 / ν_bot**2  −  1 / ν_top**2 )  (eq. 2)

For the M7.6 fleet, the production search runs after the corr-side
``--chan-sum-factor 8`` channel-sum, so the *effective* channel width
that drives w_chan is 8 × the native channel width.  We report both
the native and the chan-sum-8 cases so the assumption is visible.

Outputs (default):

    figs/dm_plan_snr_loss.pdf  — multi-panel summary plot
    figs/dm_plan_snr_loss.txt  — text report with percentiles

Usage::

    python tools/sim/dm_plan_snr_loss.py \\
        --plan configs/dm_plan.npz \\
        --n-samples 200000 \\
        --out-dir docs/overview/figs

The simulation is deterministic given ``--seed`` (default 20260528).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------- consts

K_DM_MS_GHZ2_PC: float = 4.148808
"""Standard pulsar-astronomy dispersion constant.  Matches
``src/dsart/common/constants.K_DM_MS_GHZ2_PC``."""

# Processed-band edges and native channel width (from the dm_plan.npz
# metadata blob).  These match
# src/dsart/common/constants:NU_TOP_PROC_GHZ / NU_BOT_PROC_GHZ /
# DELTA_NU_CH_GHZ.
NU_TOP_PROC_GHZ: float = 1.498750
NU_BOT_PROC_GHZ: float = 1.311280517578125
DELTA_NU_CH_NATIVE_GHZ: float = 3.0517578125e-5  # = BW_conf / N_chan_conf
BW_PROC_MHZ: float = (NU_TOP_PROC_GHZ - NU_BOT_PROC_GHZ) * 1000.0

# Production runs --chan-sum-factor 8 on corr side before incoherent
# dedispersion downstream of corr_fast.  Both Δν values are shown in
# the figure so the choice is auditable.
CHAN_SUM_FACTOR_PROD: int = 8

# Search-side sample period (μs).  Drives the *floor* on observed width
# (one sample is the smallest box we can match-filter against), but the
# user-specified width formula explicitly does NOT include t_int_search
# in the quadrature sum.  Quoted for context only.
T_INT_SEARCH_US: float = 524.288


# ----------------------------------------------------------------- functions


def smear_intra_channel_ms(
    nu_ghz: float, dm_pc_cc: float, dnu_ghz: float
) -> float:
    """Eq. (1).  Intra-channel dispersion smearing in ms."""
    return 2.0 * K_DM_MS_GHZ2_PC * dm_pc_cc * dnu_ghz / (nu_ghz**3)


def smear_inter_channel_ms(
    ddm_pc_cc: float | np.ndarray,
    nu_bot_ghz: float = NU_BOT_PROC_GHZ,
    nu_top_ghz: float = NU_TOP_PROC_GHZ,
) -> float | np.ndarray:
    """Eq. (2).  DM-mismatch smearing across the full band in ms."""
    return (
        K_DM_MS_GHZ2_PC
        * ddm_pc_cc
        * (1.0 / nu_bot_ghz**2 - 1.0 / nu_top_ghz**2)
    )


def load_plan(path: Path) -> dict:
    """Read the locked plan; return only the fields we use."""
    d = np.load(path, allow_pickle=True)
    md = d["metadata"]
    if md.dtype == object:
        meta = md.item()
        if isinstance(meta, str):
            meta = json.loads(meta)
    else:
        meta = {}
    return {
        "fine_dm": np.asarray(d["fine_dm"], dtype=np.float64),
        "coarse_dm": np.asarray(d["coarse_dm"], dtype=np.float64),
        "dm_min": float(d["dm_min"]),
        "dm_max": float(d["dm_max"]),
        "tol": float(d["tol"]),
        "metadata": meta,
    }


def closest_trial(true_dm: np.ndarray, trial_dm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised nearest-trial lookup.

    Returns
    -------
    idx       : (N,) int — index of closest fine trial
    delta_dm  : (N,) float — DM_true − fine_dm[idx]  (signed; |·| is the
                 mismatch driving eq. 2).
    """
    sorted_trials = np.sort(trial_dm)
    # searchsorted gives insertion points; closest is one of the two
    # neighbours.
    ins = np.searchsorted(sorted_trials, true_dm)
    ins = np.clip(ins, 1, len(sorted_trials) - 1)
    left = sorted_trials[ins - 1]
    right = sorted_trials[ins]
    take_right = (right - true_dm) < (true_dm - left)
    closest = np.where(take_right, right, left)
    # Recover indices in the *original* (unsorted) trial array.
    # The plan's fine_dm is already monotone increasing so the sort is
    # an identity; but be safe.
    order = np.argsort(trial_dm)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(trial_dm))
    closest_idx_sorted = np.where(take_right, ins, ins - 1)
    closest_idx = inv[closest_idx_sorted]
    return closest_idx.astype(np.int64), (true_dm - closest)


def run_mc(
    n_samples: int,
    plan: dict,
    chan_sum: int,
    seed: int,
) -> dict:
    """Monte Carlo loop.

    Draws ``n_samples`` true DMs uniformly on [dm_min, dm_max], finds
    the closest fine-DM trial, computes w_ideal and w_meas at the
    *bottom of the band* (worst-case channel), and returns arrays of
    per-sample widths and S/N-loss in dB / %.
    """
    rng = np.random.default_rng(seed)
    true_dm = rng.uniform(plan["dm_min"], plan["dm_max"], size=n_samples)
    _, ddm = closest_trial(true_dm, plan["fine_dm"])
    abs_ddm = np.abs(ddm)

    dnu = DELTA_NU_CH_NATIVE_GHZ * chan_sum
    w_chan = smear_intra_channel_ms(NU_BOT_PROC_GHZ, true_dm, dnu)
    w_miss = smear_inter_channel_ms(abs_ddm)

    w_ideal = w_chan
    w_meas = np.sqrt(w_chan**2 + w_miss**2)

    # avoid divide-by-zero at DM = 0 by setting eta = 1 wherever both
    # widths are 0 (the limiting matched filter is at the sample
    # period — but that's a floor, not part of this model).
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(w_meas > 0, np.sqrt(w_ideal / w_meas), 1.0)
    eta = np.where(np.isfinite(eta), eta, 1.0)
    eta = np.clip(eta, 0.0, 1.0)

    loss_db = 10.0 * np.log10(np.clip(eta, 1e-12, 1.0))
    loss_pct = (1.0 - eta) * 100.0

    return {
        "true_dm": true_dm,
        "abs_ddm": abs_ddm,
        "w_chan_ms": w_chan,
        "w_miss_ms": w_miss,
        "w_ideal_ms": w_ideal,
        "w_meas_ms": w_meas,
        "eta": eta,
        "loss_db": loss_db,
        "loss_pct": loss_pct,
        "chan_sum": chan_sum,
        "delta_nu_ghz": dnu,
    }


def bin_percentiles(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 40,
    worst_percentile: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin median + 16/84 + ``worst_percentile`` of ``y`` vs ``x``
    (DM_true).

    For positive-valued losses (per cent) the "worst" tail is the upper
    percentile (e.g. 99); for the negative-valued log loss (dB) the
    "worst" tail is the lower percentile (pass ``worst_percentile=1.0``).
    """
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    med = np.full(n_bins, np.nan)
    p16 = np.full(n_bins, np.nan)
    p84 = np.full(n_bins, np.nan)
    worst = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        med[b] = np.median(y[sel])
        p16[b] = np.percentile(y[sel], 16)
        p84[b] = np.percentile(y[sel], 84)
        worst[b] = np.percentile(y[sel], worst_percentile)
    return centres, med, p16, p84, worst


def make_plot(
    res_native: dict,
    res_prod: dict,
    plan: dict,
    out_path: Path,
) -> None:
    """Five-panel summary figure.

    Panels:
        (a) fine-DM trial spacing vs DM_true (the plan we inherited)
        (b) ΔDM = DM_true − closest trial — histogram + per-bin spread
        (c) widths (w_chan vs w_miss) vs DM_true
        (d) S/N loss in dB vs DM_true (median + p16-p84 + worst-case)
        (e) S/N loss in % vs DM_true
    """
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.0))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    ax_e, ax_f = axes[2, 0], axes[2, 1]

    fine_dm = plan["fine_dm"]
    spacing = np.diff(fine_dm)

    # ----- (a) DM-trial spacing
    ax_a.step(fine_dm[1:], spacing, where="post", lw=0.9, color="C0")
    ax_a.set_xlabel("Fine DM trial (pc cm$^{-3}$)")
    ax_a.set_ylabel("DM spacing $\\Delta$DM (pc cm$^{-3}$)")
    ax_a.set_title(
        "(a) Production fine-DM plan spacing"
        f"  (N = {len(fine_dm)} trials, tol = {plan['tol']:.2f})"
    )
    ax_a.set_yscale("log")
    ax_a.grid(True, which="both", alpha=0.3)

    # ----- (b) absolute mismatch histogram
    for res, label, col in [
        (res_native, "native $\\Delta\\nu_{ch}$ = 30.5 kHz", "C2"),
        (res_prod, "chan-sum 8: $\\Delta\\nu_{ch}$ = 244 kHz", "C3"),
    ]:
        ax_b.hist(
            res["abs_ddm"],
            bins=80,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=col,
            label=label,
        )
    ax_b.set_xlabel("$|\\Delta\\mathrm{DM}|$ to nearest trial (pc cm$^{-3}$)")
    ax_b.set_ylabel("Probability density")
    ax_b.set_title("(b) Mismatch distribution (uniform DM$_\\mathrm{true}$ draws)")
    ax_b.grid(True, alpha=0.3)

    # ----- (c) widths
    centres_c, med_chan, *_ = bin_percentiles(
        res_prod["true_dm"], res_prod["w_chan_ms"], n_bins=40
    )
    centres_m, med_miss, p16_miss, p84_miss, _ = bin_percentiles(
        res_prod["true_dm"], res_prod["w_miss_ms"], n_bins=40
    )
    ax_c.plot(
        centres_c, med_chan, color="C1", lw=2.0,
        label="$w_\\mathrm{chan}$ at bottom of band\n  (prod: $\\Delta\\nu = 244$ kHz)",
    )
    ax_c.plot(
        centres_m, med_miss, color="C0", lw=2.0,
        label="$w_\\mathrm{DM\\,miss}$ median across band",
    )
    ax_c.fill_between(centres_m, p16_miss, p84_miss, color="C0", alpha=0.25)
    ax_c.set_xlabel("DM$_\\mathrm{true}$ (pc cm$^{-3}$)")
    ax_c.set_ylabel("Pulse width (ms)")
    ax_c.set_title("(c) Broadening contributions (prod chan-sum 8)")
    ax_c.legend(loc="upper left", fontsize=9)
    ax_c.set_yscale("log")
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.axhline(
        T_INT_SEARCH_US * 1e-3, color="grey", ls=":", alpha=0.7,
        label="t_int_search = 524 μs",
    )

    # ----- (d) S/N loss in dB.  For dB, "worst" is most negative → p1.
    for res, label, col in [
        (res_native, "$\\Delta\\nu_{ch}$ = 30.5 kHz (native)", "C2"),
        (res_prod, "$\\Delta\\nu_{ch}$ = 244 kHz (chan-sum 8)", "C3"),
    ]:
        c, med, p16, p84, w1 = bin_percentiles(
            res["true_dm"], res["loss_db"], n_bins=40, worst_percentile=1.0
        )
        ax_d.plot(c, med, color=col, lw=2.0, label=f"median, {label}")
        ax_d.fill_between(c, p16, p84, color=col, alpha=0.20)
        ax_d.plot(c, w1, color=col, lw=1.0, ls="--",
                  label=f"worst 1 %, {label}")
    ax_d.axhline(0.0, color="k", lw=0.8)
    # tol=1.5 sensitivity target: 1.5x adjacent trials means ~33% S/N
    # loss in the trial-spacing-only ideal (no chan smear).  Plot the
    # design target as a dashed line for reference: at 33% loss,
    # 10·log10(0.667) ≈ -1.76 dB.
    ax_d.axhline(
        10.0 * np.log10(1.0 / 1.5),
        color="grey", ls=":", lw=1.0,
        label="design target tol=1.5 ($-1.76$ dB)",
    )
    ax_d.set_xlabel("DM$_\\mathrm{true}$ (pc cm$^{-3}$)")
    ax_d.set_ylabel("S/N loss = $10\\,\\log_{10}(\\eta)$  [dB]")
    ax_d.set_title("(d) Matched-filter S/N loss vs DM$_\\mathrm{true}$")
    ax_d.legend(loc="lower left", fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # ----- (e) S/N loss in % .  For pct, "worst" is largest → p99.
    for res, label, col in [
        (res_native, "native $\\Delta\\nu$", "C2"),
        (res_prod, "chan-sum 8", "C3"),
    ]:
        c, med, p16, p84, w99 = bin_percentiles(
            res["true_dm"], res["loss_pct"], n_bins=40, worst_percentile=99.0
        )
        ax_e.plot(c, med, color=col, lw=2.0, label=f"median  ({label})")
        ax_e.fill_between(c, p16, p84, color=col, alpha=0.20)
        ax_e.plot(
            c, w99, color=col, lw=1.0, ls="--",
            label=f"worst (p99)  ({label})",
        )
    ax_e.set_xlabel("DM$_\\mathrm{true}$ (pc cm$^{-3}$)")
    ax_e.set_ylabel("S/N loss (per cent)")
    ax_e.set_title("(e) Matched-filter S/N loss [per cent]")
    ax_e.legend(loc="upper left", fontsize=8)
    ax_e.grid(True, alpha=0.3)

    # ----- (f) text summary
    ax_f.axis("off")
    lines = ["Monte-Carlo S/N loss summary", "-" * 40]
    for label, res in [
        ("native (Δν = 30.5 kHz)", res_native),
        ("production (chan-sum 8, Δν = 244 kHz)", res_prod),
    ]:
        lines += [
            f"\n{label}:",
            f"  N samples            : {len(res['true_dm']):,}",
            f"  |ΔDM| median         : {np.median(res['abs_ddm']):8.3f} pc cm⁻³",
            f"  |ΔDM| p99            : {np.percentile(res['abs_ddm'], 99):8.3f} pc cm⁻³",
            f"  Loss median          : {np.median(res['loss_db']):8.3f} dB"
            f"  ({np.median(res['loss_pct']):5.2f} %)",
            f"  Loss worst 10 %      : {np.percentile(res['loss_db'], 10):8.3f} dB"
            f"  ({np.percentile(res['loss_pct'], 90):5.2f} %)",
            f"  Loss worst  1 %      : {np.percentile(res['loss_db'], 1):8.3f} dB"
            f"  ({np.percentile(res['loss_pct'], 99):5.2f} %)",
        ]
    lines += [
        "",
        "Design target (DM tol = 1.5):  -1.76 dB  (33.3 %)",
        "Source model: δ-fn pulse, uniform DM$_\\mathrm{true}$ ∈ [0, 3000] pc cm⁻³",
        "Width formula: w_meas = sqrt(w_chan² + w_DM_miss²)",
        "             w_ideal = w_chan",
        "Bottom-of-band channel used for w_chan.",
    ]
    ax_f.text(
        0.02, 0.98, "\n".join(lines),
        family="monospace", fontsize=9, va="top", transform=ax_f.transAxes,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.pdf')}")


def write_text_report(
    res_native: dict, res_prod: dict, plan: dict, out_path: Path
) -> None:
    """ASCII text report (also reproduced in the LaTeX doc)."""
    fine_dm = plan["fine_dm"]
    spacing = np.diff(fine_dm)
    lines = []
    lines.append("DSA-110 dsa110-rt — DM-plan Monte-Carlo S/N-loss report")
    lines.append("=" * 64)
    lines.append("")
    lines.append("Inherited fine-DM plan (configs/dm_plan.npz):")
    lines.append(f"  dm_min          = {plan['dm_min']:.3f} pc cm⁻³")
    lines.append(f"  dm_max          = {plan['dm_max']:.3f} pc cm⁻³")
    lines.append(f"  Levin tolerance = {plan['tol']:.3f}")
    lines.append(f"  N_fine          = {len(fine_dm)} trials")
    lines.append(
        f"  spacing range   = {spacing.min():.4f} .. {spacing.max():.3f} pc cm⁻³"
    )
    lines.append(f"  N_coarse        = {len(plan['coarse_dm'])}")
    lines.append("")
    for label, res in [
        ("Channel-sum factor 1 (native Δν = 30.5 kHz)", res_native),
        ("Channel-sum factor 8 (production Δν = 244 kHz)", res_prod),
    ]:
        lines.append(label)
        lines.append("-" * len(label))
        # For positive-valued series (|ΔDM|, widths, loss %): worst is
        # the high-percentile tail.  For loss in dB (≤0): worst is the
        # low-percentile tail (most negative).
        for name, arr, unit, lo_is_worst in [
            ("|ΔDM|       ", res["abs_ddm"],   "pc cm⁻³", False),
            ("w_chan      ", res["w_chan_ms"], "ms",      False),
            ("w_DM_miss   ", res["w_miss_ms"], "ms",      False),
            ("w_meas      ", res["w_meas_ms"], "ms",      False),
            ("loss (dB)   ", res["loss_db"],   "dB",      True),
            ("loss (%)    ", res["loss_pct"],  "%",       False),
        ]:
            if lo_is_worst:
                p90_q, p99_q = 10.0, 1.0
            else:
                p90_q, p99_q = 90.0, 99.0
            lines.append(
                f"  {name}  median={np.median(arr):8.3f}  "
                f"p90={np.percentile(arr, p90_q):8.3f}  "
                f"p99={np.percentile(arr, p99_q):8.3f}   {unit}"
            )
        lines.append("")
    lines.append("Notes:")
    lines.append(" * loss = 1 − sqrt(w_ideal / w_meas), where")
    lines.append("       w_ideal = w_chan(DM_true, bottom-of-band)")
    lines.append("       w_meas  = sqrt(w_chan² + w_DM_miss²)")
    lines.append(" * w_chan uses the bottom-of-band channel (worst case).")
    lines.append(" * The intrinsic FRB width is taken as a delta-function.")
    lines.append(" * The matched-filter floor at one t_int_search sample")
    lines.append(f"   (~{T_INT_SEARCH_US:.1f} μs) is NOT folded in.")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="configs/dm_plan.npz", type=Path)
    p.add_argument("--n-samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--out-dir", type=Path, default=Path("docs/overview/figs"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_plan(args.plan)
    print(
        f"Loaded DM plan: dm∈[{plan['dm_min']:.1f},{plan['dm_max']:.1f}], "
        f"N_fine={len(plan['fine_dm'])}, N_coarse={len(plan['coarse_dm'])}, "
        f"tol={plan['tol']:.2f}"
    )

    res_native = run_mc(args.n_samples, plan, chan_sum=1, seed=args.seed)
    res_prod = run_mc(
        args.n_samples, plan, chan_sum=CHAN_SUM_FACTOR_PROD, seed=args.seed
    )

    fig_path = args.out_dir / "dm_plan_snr_loss.png"
    txt_path = args.out_dir / "dm_plan_snr_loss.txt"
    make_plot(res_native, res_prod, plan, fig_path)
    write_text_report(res_native, res_prod, plan, txt_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
