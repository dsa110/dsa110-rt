#!/usr/bin/env python3
"""bench/coarse_dm_recovery.py — coarse-DM dedispersion recovery sweep.

Operator-review bench for the M3 chunk-3b
:func:`dsart.coarse_dm.coarse_dedisp` primitive.

What it does
============

1. Synthesises a per-chgroup image cube
   ``[T_fast, NCHAN_PER_CHGROUP, N_grid, N_grid] complex64`` containing
   a single bright pixel at ``(l_idx, m_idx)`` over a range of times,
   dispersed at DM = ``--truth-dm`` pc/cc (default 405) per the
   chgroup's local frequency grid + Convention A reference (chgroup
   TOP). Background is zero (clean operator-review burst).
2. Runs :func:`dsart.coarse_dm.coarse_dedisp` at the matching DM trial.
   Reports peak time bin + spatial location + amplitude relative to
   the predicted ``NCHAN × |amp|²`` truth.
3. Sweeps coarse-DM trials around the truth (``--dm-sweep N``
   trials, default 9, log-spaced × 0.5..2.0 of truth) and plots the
   dedispersed peak amplitude vs DM. Expected shape: triangle peaked
   at the truth DM.

This is an OPERATOR-REVIEW bench (no PASS / FAIL). Reports go to
``bench/reports/<UTC>/<run_id>/M3-coarse-dm/`` per
PARALLEL_AGENTS.md §4.3 layout convention. The report is HTML +
inline PNG + CSV, no JS.

Usage
=====

::

    python -m bench.coarse_dm_recovery \\
        [--truth-dm 405.0]
        [--dm-sweep 9]
        [--n-grid 64]
        [--n-chan 24]
        [--t-fast 96]
        [--chgroup 0]
        [--t-int-fast-us 262.144]
        [--out-dir bench/reports/<UTC>/<run_id>/M3-coarse-dm/]

References
==========

* Plan §3.6.2 (DEDISP architecture) — incoherent-dedispersion shape
  vs DM expected to be triangular.
* :mod:`dsart.coarse_dm` — modules under bench.
* PARALLEL_AGENTS.md §4.3 — bench report layout convention.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.coarse_dm import (                                            # noqa: E402
    DMPlan,
    build_chgroup_freq_table_GHz,
    coarse_dedisp,
    compute_delay_native_samples_table,
)
from dsart.common.constants import (                                     # noqa: E402
    K_DM_MS_GHZ2_PC,
    NATIVE_SAMPLE_US,
    T_INT_FAST_US_DEFAULT,
    freq_GHz,
)


# ---------------------------------------------------------------------------
# Synthetic burst cube (mirrors tests/test_coarse_dm.py::_make_burst_cube
# but exposed here without the test-helper indirection)
# ---------------------------------------------------------------------------


def make_burst_cube(
    *,
    t_fast: int,
    n_chan_test: int,
    n_grid: int,
    chgroup: int,
    dm_pc_cc: float,
    t_peak_top_bin: int,
    l_idx: int,
    m_idx: int,
    t_int_fast_us: float,
    amplitude: complex = 1.0 + 0.0j,
) -> tuple[torch.Tensor, np.ndarray]:
    """Place a single hot pixel in every channel at the dispersed time.

    Returns
    -------
    cube : (T_fast, n_chan_test, n_grid, n_grid) complex64
    delay_bins : (n_chan_test,) int64 — the per-channel forward shift
        (Convention A; relative to chgroup-top channel).
    """
    cube = torch.zeros(
        (t_fast, n_chan_test, n_grid, n_grid), dtype=torch.complex64,
    )
    nu_ch = np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(n_chan_test)],
        dtype=np.float64,
    )
    nu_top = nu_ch[0]
    delay_us = K_DM_MS_GHZ2_PC * dm_pc_cc * (
        1.0 / nu_ch ** 2 - 1.0 / nu_top ** 2
    ) * 1e3
    delay_bins = np.rint(delay_us / t_int_fast_us).astype(np.int64)
    for ch in range(n_chan_test):
        t_peak_ch = t_peak_top_bin + int(delay_bins[ch])
        if 0 <= t_peak_ch < t_fast:
            cube[t_peak_ch, ch, l_idx, m_idx] = amplitude
    return cube, delay_bins


# ---------------------------------------------------------------------------
# Bench panels
# ---------------------------------------------------------------------------


def panel_recovery(
    *,
    cube: torch.Tensor,
    plan: DMPlan,
    chgroup: int,
    truth_dm_idx: int,
    t_peak_top_bin: int,
    l_idx: int,
    m_idx: int,
) -> dict:
    """Single-DM-trial recovery: how close is the peak to the predicted (t, l, m)?"""
    out = coarse_dedisp(
        cube, plan, chgroup=chgroup,
        dm_indices=torch.as_tensor([truth_dm_idx], dtype=torch.int64),
        output_dtype=torch.float32,
    )                                                                # (T_dedisp, 1, ng, ng)
    out_np = out.numpy()[:, 0]                                       # (T_dedisp, ng, ng)
    flat_argmax = int(out_np.reshape(-1).argmax())
    t_dedisp = out_np.shape[0]
    ng = out_np.shape[1]
    t_recovered = flat_argmax // (ng * ng)
    spatial = flat_argmax % (ng * ng)
    l_rec = spatial // ng
    m_rec = spatial % ng
    peak_amp = float(out_np[t_recovered, l_rec, m_rec])
    return {
        "t_recovered": t_recovered,
        "t_truth": t_peak_top_bin,
        "t_err": abs(t_recovered - t_peak_top_bin),
        "l_recovered": int(l_rec),
        "m_recovered": int(m_rec),
        "l_truth": int(l_idx),
        "m_truth": int(m_idx),
        "peak_amplitude": peak_amp,
        "T_dedisp": int(t_dedisp),
    }


def panel_dm_sweep(
    *,
    cube: torch.Tensor,
    plan: DMPlan,
    chgroup: int,
    l_idx: int,
    m_idx: int,
    t_peak_top_bin: int,
) -> list[dict]:
    """Walk all DM trials in ``plan``; for each, record dedispersed
    peak amplitude at the burst pixel.
    """
    rows: list[dict] = []
    for k in range(plan.n_coarse):
        out = coarse_dedisp(
            cube, plan, chgroup=chgroup,
            dm_indices=torch.as_tensor([k], dtype=torch.int64),
            output_dtype=torch.float32,
        )                                                            # (T_dedisp, 1, ng, ng)
        # Peak over time at the burst pixel.
        peak = float(out[:, 0, l_idx, m_idx].max().item())
        # Also peak over (t, l, m) — useful to detect off-DM peak that
        # walks off the burst pixel.
        peak_anywhere = float(out[:, 0, :, :].max().item())
        rows.append({
            "dm_idx": k,
            "dm_pc_cc": float(plan.dm_pc_cc[k]),
            "peak_at_burst_pixel": peak,
            "peak_anywhere": peak_anywhere,
        })
    return rows


# ---------------------------------------------------------------------------
# Reports — CSV + PNG + HTML
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_png(
    sweep: list[dict],
    *,
    truth_dm_pc_cc: float,
    n_chan_test: int,
    out_dir: Path,
) -> Path | None:
    """Plot dedispersed peak amplitude vs DM trial."""
    try:
        import matplotlib                                              # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping PNG output.")
        return None

    if not sweep:
        return None
    dms = [r["dm_pc_cc"] for r in sweep]
    pks_pixel = [r["peak_at_burst_pixel"] for r in sweep]
    pks_any = [r["peak_anywhere"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    ax.plot(dms, pks_pixel, marker="o", label="peak at burst pixel")
    ax.plot(dms, pks_any, marker="s", linestyle="--", alpha=0.6,
            label="peak anywhere in image")
    ax.axvline(truth_dm_pc_cc, color="r", linestyle=":",
               label=f"truth DM = {truth_dm_pc_cc:.1f}")
    ax.set_xlabel("trial DM (pc / cm³)")
    ax.set_ylabel("dedispersed peak amplitude")
    ax.set_title(
        "Coarse-DM dedispersion: peak amplitude vs DM trial\n"
        f"(triangle-shape expected; flat-top width grows with intra-"
        f"chgroup smear; n_chan = {n_chan_test})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "dm_sweep_peak_vs_dm.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_html(
    *,
    recovery: dict,
    sweep: list[dict],
    csv_path: Path,
    png_path: Path | None,
    config: dict,
    out_html: Path,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(out_html.parent).as_posix()

    # Truth peak amplitude (n_chan × |amp|²) for context.
    truth_peak = config["n_chan_test"] * abs(config["amplitude"]) ** 2

    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>M3 coarse-DM recovery bench</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:980px;margin:2em auto;}",
        "table{border-collapse:collapse;margin:1em 0;}",
        "th,td{border:1px solid #aaa;padding:4px 8px;text-align:right;}",
        "th{background:#eef;}",
        "tr:nth-child(even){background:#f7f7f7;}",
        ".pass{color:#080;font-weight:bold;}",
        ".fail{color:#c00;font-weight:bold;}",
        ".note{color:#555;}",
        "img{max-width:100%;}",
        "code{background:#eef;padding:1px 4px;border-radius:3px;}",
        "</style>",
        "</head><body>",
        "<h1>M3 coarse-DM recovery bench</h1>",
        "<p class='note'>Operator-review only — no PASS / FAIL gate.</p>",
        "<h2>Configuration</h2>",
        "<table>",
    ]
    for k in (
        "truth_dm_pc_cc",
        "n_grid",
        "n_chan_test",
        "t_fast",
        "chgroup",
        "t_int_fast_us",
        "t_peak_top_bin",
        "l_idx",
        "m_idx",
        "amplitude",
        "n_dm_sweep",
    ):
        parts.append(f"<tr><th>{k}</th><td>{config[k]}</td></tr>")
    parts.append("</table>")

    # Recovery panel
    t_match = recovery["t_err"] <= 1
    lm_match = (
        recovery["l_recovered"] == recovery["l_truth"]
        and recovery["m_recovered"] == recovery["m_truth"]
    )
    amp_frac = (
        recovery["peak_amplitude"] / truth_peak if truth_peak > 0 else 0.0
    )
    parts.append("<h2>Single-DM-trial recovery</h2>")
    parts.append(
        f"<p>Time-bin recovery: t_recovered = "
        f"<code>{recovery['t_recovered']}</code>, "
        f"t_truth = <code>{recovery['t_truth']}</code> → "
        f"|Δt| = <code>{recovery['t_err']}</code> bins "
        f"(<span class='{'pass' if t_match else 'fail'}'>"
        f"{'within ≤ 1 native sample' if t_match else '>1 sample off'}"
        f"</span>).</p>"
    )
    parts.append(
        f"<p>Spatial recovery: (l, m) = "
        f"<code>({recovery['l_recovered']}, {recovery['m_recovered']})</code>, "
        f"truth = <code>({recovery['l_truth']}, {recovery['m_truth']})</code> "
        f"(<span class='{'pass' if lm_match else 'fail'}'>"
        f"{'EXACT' if lm_match else 'MISMATCH'}</span>).</p>"
    )
    parts.append(
        f"<p>Peak amplitude: "
        f"<code>{recovery['peak_amplitude']:.2f}</code> "
        f"(truth = <code>{truth_peak:.2f}</code>, "
        f"<code>{amp_frac * 100:.1f}%</code>).</p>"
    )
    parts.append(
        f"<p>T_dedisp = <code>{recovery['T_dedisp']}</code> bins "
        f"(= T_fast - max bin shift over the trial subset).</p>"
    )

    # DM sweep panel
    parts.append("<h2>DM sweep (peak vs trial DM)</h2>")
    if png_path is not None:
        parts.append(f"<figure><img src='{rel(png_path)}'/></figure>")
    parts.append("<table><tr><th>dm_idx</th><th>dm_pc_cc</th>")
    parts.append("<th>peak @ burst pixel</th><th>peak anywhere</th></tr>")
    for r in sweep:
        parts.append(
            f"<tr><td>{r['dm_idx']}</td>"
            f"<td>{r['dm_pc_cc']:.3f}</td>"
            f"<td>{r['peak_at_burst_pixel']:.2f}</td>"
            f"<td>{r['peak_anywhere']:.2f}</td></tr>"
        )
    parts.append("</table>")
    parts.append(f"<p>CSV: <a href='{rel(csv_path)}'>{csv_path.name}</a></p>")

    parts.append("</body></html>")
    out_html.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-dm", type=float, default=405.0)
    ap.add_argument("--dm-sweep", type=int, default=9)
    ap.add_argument("--n-grid", type=int, default=64)
    ap.add_argument("--n-chan", type=int, default=24)
    ap.add_argument("--t-fast", type=int, default=96)
    ap.add_argument("--chgroup", type=int, default=0)
    ap.add_argument("--t-int-fast-us", type=float, default=T_INT_FAST_US_DEFAULT)
    ap.add_argument("--t-peak-top-bin", type=int, default=4)
    ap.add_argument("--l-idx", type=int, default=11)
    ap.add_argument("--m-idx", type=int, default=20)
    ap.add_argument("--amplitude", type=float, default=2.0,
                    help="Real amplitude (imag = 0) of each per-channel pixel")
    ap.add_argument(
        "--out-dir", type=str, default=None,
        help=(
            "Output dir. Default: "
            "bench/reports/<UTC>/<run_id>/M3-coarse-dm/"
        ),
    )
    ap.add_argument("--run-id", type=str, default="coarse-dm-recovery")
    args = ap.parse_args()

    # Out dir
    if args.out_dir is None:
        utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_dir = (
            REPO_ROOT / "bench" / "reports" / utc / args.run_id / "M3-coarse-dm"
        )
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing reports to {out_dir}")

    # Build a DM plan with the requested sweep around the truth DM.
    if args.dm_sweep <= 1:
        dms = np.asarray([args.truth_dm], dtype=np.float64)
    else:
        # Symmetric log-sweep × 0.5..2 of truth.
        ratios = np.geomspace(0.5, 2.0, args.dm_sweep)
        dms = np.round(args.truth_dm * ratios, decimals=3)
        # Insert the truth DM if it's not already in the sweep (within rounding).
        if not np.any(np.isclose(dms, args.truth_dm)):
            dms = np.sort(np.concatenate([dms, [args.truth_dm]]))
        # Strip duplicates
        dms = np.unique(dms)
    chgroup_freqs = build_chgroup_freq_table_GHz()
    delay_table = compute_delay_native_samples_table(dms, chgroup_freqs)
    plan = DMPlan(
        dm_pc_cc=dms,
        n_fine_per_coarse=1,
        t_int_fast_us=args.t_int_fast_us,
        chgroup_freqs_GHz=chgroup_freqs,
        _delay_native_samples_table=delay_table,
    )
    truth_dm_idx = int(np.argmin(np.abs(plan.dm_pc_cc - args.truth_dm)))
    print(
        f"DM sweep: {plan.n_coarse} trials at "
        f"{[f'{d:.1f}' for d in plan.dm_pc_cc]}; "
        f"truth_dm_idx = {truth_dm_idx} (= {plan.dm_pc_cc[truth_dm_idx]:.1f})"
    )

    # Build the synthetic burst cube at the truth DM.
    cube, delay_bins = make_burst_cube(
        t_fast=args.t_fast,
        n_chan_test=args.n_chan,
        n_grid=args.n_grid,
        chgroup=args.chgroup,
        dm_pc_cc=args.truth_dm,
        t_peak_top_bin=args.t_peak_top_bin,
        l_idx=args.l_idx,
        m_idx=args.m_idx,
        t_int_fast_us=args.t_int_fast_us,
        amplitude=args.amplitude + 0.0j,
    )
    print(
        f"cube: shape={tuple(cube.shape)} dtype={cube.dtype}; "
        f"max delay across channels = {int(delay_bins.max())} bins "
        f"(= {int(delay_bins.max()) * args.t_int_fast_us:.0f} µs = "
        f"{int(delay_bins.max()) * args.t_int_fast_us / NATIVE_SAMPLE_US:.1f} "
        f"native samples)"
    )

    # Panel 1: recovery at the truth DM.
    recovery = panel_recovery(
        cube=cube,
        plan=plan,
        chgroup=args.chgroup,
        truth_dm_idx=truth_dm_idx,
        t_peak_top_bin=args.t_peak_top_bin,
        l_idx=args.l_idx,
        m_idx=args.m_idx,
    )
    print(
        f"recovery: t_recovered = {recovery['t_recovered']} "
        f"(t_truth = {recovery['t_truth']}, |Δt| = {recovery['t_err']}); "
        f"(l, m) = ({recovery['l_recovered']}, {recovery['m_recovered']}) "
        f"(truth ({recovery['l_truth']}, {recovery['m_truth']})); "
        f"peak amp = {recovery['peak_amplitude']:.2f}"
    )

    # Panel 2: DM sweep.
    sweep = panel_dm_sweep(
        cube=cube,
        plan=plan,
        chgroup=args.chgroup,
        l_idx=args.l_idx,
        m_idx=args.m_idx,
        t_peak_top_bin=args.t_peak_top_bin,
    )
    pks_str = ", ".join(
        f"{r['peak_at_burst_pixel']:.1f}" for r in sweep
    )
    print(f"dm-sweep: peak at burst pixel = [{pks_str}]")

    # Reports
    csv_path = out_dir / "dm_sweep.csv"
    _write_csv(
        sweep, csv_path,
        fieldnames=["dm_idx", "dm_pc_cc", "peak_at_burst_pixel", "peak_anywhere"],
    )
    png_path = _write_png(
        sweep,
        truth_dm_pc_cc=args.truth_dm,
        n_chan_test=args.n_chan,
        out_dir=out_dir,
    )
    config = {
        "truth_dm_pc_cc": args.truth_dm,
        "n_grid": args.n_grid,
        "n_chan_test": args.n_chan,
        "t_fast": args.t_fast,
        "chgroup": args.chgroup,
        "t_int_fast_us": args.t_int_fast_us,
        "t_peak_top_bin": args.t_peak_top_bin,
        "l_idx": args.l_idx,
        "m_idx": args.m_idx,
        "amplitude": args.amplitude,
        "n_dm_sweep": plan.n_coarse,
    }
    html_path = out_dir / "index.html"
    _write_html(
        recovery=recovery,
        sweep=sweep,
        csv_path=csv_path,
        png_path=png_path,
        config=config,
        out_html=html_path,
    )

    print(f"wrote: {html_path}")
    if png_path is not None:
        print(f"wrote: {png_path}")
    print(f"wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
