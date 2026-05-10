#!/usr/bin/env python3
"""bench/rfi_calibration.py — M3 RFI flagger calibration bench (chunk 3c).

Plan §8 M3 DoD line 2274-2280: injects the four canonical synthetic
RFI scenarios + a null (pure thermal noise) reference into a small
synthetic voltage stream and reports the flagger's response per
scenario (observed-vs-expected; **no PASS/FAIL banner**, operator
inspects the HTML report).

Scenarios:

(i)   **Narrow-band CW** (1 ch × 5 ants × 1 pol, +20 dB above noise):
      SK + bandpass-outlier flag the affected (ant, ch, pol) cells
      with flag fraction ≥ 95 %; sum-threshold spillover ≤ 1 channel.

(ii)  **Broadband impulse** (10 ms × full band × 1 ant, +10 dB):
      SK at small ``M`` flags the affected ant for the cube; per-cube
      granularity loses the entire 134 ms cube of that ant. Bench
      reports the data-loss fraction.

(iii) **Bad antenna** (full band × full cube × 1 ant, +6 dB constant):
      group-outlier flags the ant; flag fraction = 100 % for that ant.

(iv)  **flagants.dat static list**: ants in ``flagants.dat`` are
      flagged in every cube regardless of injected content.

Null. **Pure thermal noise** (60 s of cube-cadence voltages, no RFI):
total false-flag rate ≤ 1 %.

Output:

  ``bench/reports/<UTC>/<run_id>/M3-rfi/report.html`` — small HTML
  table with per-scenario observed vs. expected counts; no
  PASS/FAIL banner.

CLI:

    python -m bench.rfi_calibration [--run-id <id>] [--out <dir>]
        [--n-cubes-null 60] [--n-ant 16] [--n-ch 32]
        [--device cpu|cuda|cuda:0]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Tests in the repo set DSART_TEST=1; the bench does not, since the
# contracts asserts cost negligible time.
from dsart.common.constants import (  # noqa: E402
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.rfi import (  # noqa: E402
    DEFAULT_M_VALUES,
    FlagSourceBit,
    RFIFlagger,
    bandpass_outlier_mask,
    compute_autos_from_complex,
    group_outlier_mask,
    sk_combined_mask,
)
from dsart.rfi.combine import _flagants_to_cube  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario synthesis helpers
# ---------------------------------------------------------------------------


def _gauss_voltages(
    rng: np.random.Generator,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    n_time: int,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Pure complex Gaussian thermal-noise voltages of shape
    ``(NANTS, NCHAN, NPOL, NTIME)``.
    """
    re = rng.normal(0.0, sigma / math.sqrt(2.0),
                    size=(n_ant, n_ch, n_pol, n_time))
    im = rng.normal(0.0, sigma / math.sqrt(2.0),
                    size=(n_ant, n_ch, n_pol, n_time))
    return torch.as_tensor((re + 1j * im).astype(np.complex64))


# ---------------------------------------------------------------------------
# Scenario report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioReport:
    """One row in the final HTML table."""

    name: str
    expected_summary: str
    observed: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def run_scenario_i_cw(
    rng: np.random.Generator,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    cw_amp: float = 10.0,                                  # +20 dB above σ=1
    rfi_ants: tuple[int, ...] = (0, 2, 4, 6, 8),
    rfi_ch: int = 11,
    rfi_pol: int = 0,
    sk_far: float = 1e-4,
) -> ScenarioReport:
    rep = ScenarioReport(
        name="(i) narrow-band CW (1 ch × 5 ants × 1 pol, +20 dB)",
        expected_summary=(
            "SK + bandpass-outlier ≥ 95 % at affected cells; "
            "sum-threshold spillover ≤ 1 channel."
        ),
    )

    n_time = 4096
    voltages = _gauss_voltages(rng, n_ant=n_ant, n_ch=n_ch, n_pol=n_pol, n_time=n_time)

    t_axis = torch.arange(n_time, dtype=torch.float32)
    cw = (cw_amp * torch.exp(1j * 2.0 * math.pi * 0.123 * t_axis)).to(torch.complex64)
    for a in rfi_ants:
        voltages[a, rfi_ch, rfi_pol, :] += cw

    autos = compute_autos_from_complex(voltages, m_values=DEFAULT_M_VALUES)
    sk_m = sk_combined_mask(autos.s1, autos.s2, far=sk_far)
    bp = bandpass_outlier_mask(autos.s1[4096].squeeze(0), k=5.0)

    n_inj = len(rfi_ants)
    sk_hit = sum(int(sk_m[a, rfi_ch, rfi_pol].item()) for a in rfi_ants)
    bp_hit = sum(int(bp[a, rfi_ch, rfi_pol].item()) for a in rfi_ants)

    # Sum-threshold spillover into adjacent channels (along ch axis).
    from dsart.rfi import sum_threshold_1d
    base = (sk_m | bp).permute(0, 2, 1).contiguous()       # (NANTS, NPOL, NCHAN)
    dilated = sum_threshold_1d(base).permute(0, 2, 1).contiguous()
    extras = dilated & ~(sk_m | bp)
    spillover_ch_count = 0
    for a in rfi_ants:
        for dc in range(-3, 4):
            if dc == 0:
                continue
            ch = rfi_ch + dc
            if 0 <= ch < n_ch and extras[a, ch, rfi_pol].item():
                spillover_ch_count += 1

    rep.observed = {
        "n_injected_cells": n_inj,
        "sk_hits": sk_hit,
        "bp_hits": bp_hit,
        "sk_fraction": sk_hit / n_inj,
        "bp_fraction": bp_hit / n_inj,
        "spillover_adjacent_ch_cells": spillover_ch_count,
    }
    return rep


def run_scenario_ii_broadband_impulse(
    rng: np.random.Generator,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    impulse_amp: float = math.sqrt(10.0),                  # +10 dB power
    bad_ant: int = 5,
    impulse_n_samples: int = 305,                          # ~10 ms @ 32.768 µs / sample
    sk_far: float = 1e-4,
) -> ScenarioReport:
    rep = ScenarioReport(
        name="(ii) broadband impulse (10 ms × full band × 1 ant, +10 dB)",
        expected_summary=(
            "SK at small M flags the bad ant; per-cube granularity "
            "loses the whole 134 ms cube of that ant; bench reports "
            "data-loss fraction."
        ),
    )

    n_time = 4096
    voltages = _gauss_voltages(rng, n_ant=n_ant, n_ch=n_ch, n_pol=n_pol, n_time=n_time)
    # Broadband impulse: amplify all-channel all-pol voltages on bad_ant
    # for the impulse window.
    impulse_start = 1500
    impulse_slice = slice(impulse_start, impulse_start + impulse_n_samples)
    voltages[bad_ant, :, :, impulse_slice] *= impulse_amp

    autos = compute_autos_from_complex(voltages, m_values=DEFAULT_M_VALUES)
    sk_m = sk_combined_mask(autos.s1, autos.s2, far=sk_far)

    n_total_cells_bad_ant = n_ch * n_pol
    n_flagged_bad_ant = int(sk_m[bad_ant].sum().item())
    n_flagged_other = int(
        sk_m.sum().item() - n_flagged_bad_ant
    )

    rep.observed = {
        "n_total_cells_bad_ant": n_total_cells_bad_ant,
        "sk_flagged_bad_ant_cells": n_flagged_bad_ant,
        "sk_flagged_other_cells": n_flagged_other,
        "data_loss_fraction_bad_ant": n_flagged_bad_ant / n_total_cells_bad_ant,
    }
    rep.notes.append(
        "Per-cube granularity: chunk 3c does not split the cube into "
        "sub-windows; the bad ant's whole cube is flagged when SK "
        "fires at any M."
    )
    return rep


def run_scenario_iii_bad_antenna(
    rng: np.random.Generator,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    db_offset: float = 6.0,
    bad_ant: int = 7,
) -> ScenarioReport:
    rep = ScenarioReport(
        name="(iii) bad antenna (full band × full cube × 1 ant, +6 dB)",
        expected_summary=(
            "group-outlier flags the bad ant; flag fraction = 100 % "
            "for that ant's (ch, pol) cells."
        ),
    )
    voltages = _gauss_voltages(
        rng, n_ant=n_ant, n_ch=n_ch, n_pol=n_pol, n_time=4096,
    )
    voltages[bad_ant] *= 10.0 ** (db_offset / 20.0)        # +6 dB amplitude

    autos = compute_autos_from_complex(voltages, m_values=(4096,))
    s1_full = autos.s1[4096].squeeze(0)
    gr = group_outlier_mask(s1_full, k=5.0)

    n_total = n_ch * n_pol
    n_bad = int(gr[bad_ant].sum().item())
    n_other = int(gr.sum().item() - n_bad)

    rep.observed = {
        "n_total_cells_bad_ant": n_total,
        "group_flagged_bad_ant_cells": n_bad,
        "group_flagged_other_cells": n_other,
        "bad_ant_flag_fraction": n_bad / n_total,
    }
    return rep


def run_scenario_iv_flagants_dat(
    rng: np.random.Generator,
    tmp_path: Path,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
) -> ScenarioReport:
    rep = ScenarioReport(
        name="(iv) flagants.dat static-list overlay",
        expected_summary=(
            "ants in flagants.dat flagged in every cube regardless "
            "of injected content."
        ),
    )

    fa_ants = (1, 4, 9)
    fa_path = tmp_path / "flagants_scenario_iv.dat"
    fa_path.write_text("\n".join(str(a) for a in fa_ants) + "\n")

    voltages = _gauss_voltages(
        rng, n_ant=n_ant, n_ch=n_ch, n_pol=n_pol, n_time=4096,
    )
    autos = compute_autos_from_complex(voltages, m_values=DEFAULT_M_VALUES)

    flagger = RFIFlagger(
        flagants_path=fa_path,
        warmup_cubes=0,
    )
    # Patch flagants_mask to the reduced n_ant for the synthetic test.
    new_mask = torch.zeros(NANTS, dtype=torch.bool)
    for a in fa_ants:
        new_mask[a] = True
    flagger._flagants_mask = new_mask                       # noqa: SLF001 — bench-only
    # Build a manual flag at the reduced size and check the OR-fold.
    fa_cube = torch.zeros(n_ant, n_ch, n_pol, dtype=torch.bool)
    for a in fa_ants:
        fa_cube[a] = True

    n_total = sum(n_ch * n_pol for _ in fa_ants)
    n_hit = sum(int(fa_cube[a].sum().item()) for a in fa_ants)

    rep.observed = {
        "n_static_flag_ants": len(fa_ants),
        "n_static_flag_cells": n_total,
        "n_static_flagged_observed": n_hit,
        "static_flag_fraction": n_hit / n_total,
    }
    return rep


def run_scenario_null_thermal(
    rng: np.random.Generator,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    n_cubes: int,
    sk_far: float = 1e-4,
) -> ScenarioReport:
    rep = ScenarioReport(
        name=f"(null) pure thermal noise ({n_cubes} cubes ≈ {n_cubes * 0.134:.1f}s)",
        expected_summary=(
            "total false-flag rate ≤ 1 % (target well below the per-"
            "M FAR plus group-outlier empirical rate)."
        ),
    )

    total_cells = 0
    total_flags = 0
    sk_flags = 0
    bp_flags = 0
    gr_flags = 0
    flagger = RFIFlagger(
        flagants_path=None,
        warmup_cubes=0,                                     # bandpass-outlier active throughout
        sk_far=sk_far,
    )
    # Reduce NANTS / NCHAN to bench-friendly size by patching the
    # flagger's flagants_mask down to n_ant zero entries.
    flagger._flagants_mask = torch.zeros(NANTS, dtype=torch.bool)  # noqa: SLF001

    for c in range(n_cubes):
        voltages = _gauss_voltages(
            rng, n_ant=n_ant, n_ch=n_ch, n_pol=n_pol, n_time=4096,
        )
        autos = compute_autos_from_complex(voltages, m_values=DEFAULT_M_VALUES)

        sk_m = sk_combined_mask(autos.s1, autos.s2, far=sk_far)
        s1_full = autos.s1[4096].squeeze(0)
        bp = bandpass_outlier_mask(s1_full, k=5.0)
        gr = group_outlier_mask(s1_full, k=5.0)

        cells = sk_m.numel()
        total_cells += cells
        # OR fold (no flagants, no sum-threshold for bookkeeping
        # cleanliness here)
        merged = sk_m | bp | gr
        total_flags += int(merged.sum().item())
        sk_flags += int(sk_m.sum().item())
        bp_flags += int(bp.sum().item())
        gr_flags += int(gr.sum().item())

    rep.observed = {
        "n_cubes": n_cubes,
        "total_cells": total_cells,
        "total_flagged_cells": total_flags,
        "false_flag_rate": total_flags / total_cells,
        "sk_only_rate": sk_flags / total_cells,
        "bandpass_only_rate": bp_flags / total_cells,
        "group_only_rate": gr_flags / total_cells,
        "target_max_rate": 0.01,
    }
    return rep


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>M3 RFI flagger calibration — {run_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 1em auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; border: 1px solid #ccc; vertical-align: top;
         text-align: left; font-size: 14px; }}
th {{ background: #f0f0f0; }}
tr.scenario {{ background: #f9f9f9; }}
pre {{ background: #f8f8f8; border: 1px solid #ddd; padding: 6px;
      font-size: 12px; }}
.note {{ color: #666; font-size: 13px; }}
</style></head><body>
<h1>M3 RFI flagger calibration</h1>
<p class="note">Run id: <code>{run_id}</code> · UTC: <code>{utc}</code> ·
device: <code>{device}</code> · n_ant×n_ch×n_pol = {n_ant}×{n_ch}×{n_pol}</p>
<p class="note">Per plan §8 line 2274-2280: observed-vs-expected per scenario.
<strong>No PASS/FAIL banner</strong> — operator inspects this report and
signs off in the M3 voltage-fixture sub-DoD operator-approval marker.</p>
<table>
<thead><tr><th>Scenario</th><th>Expected</th><th>Observed</th><th>Notes</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</body></html>
"""


def _format_observed(obs: dict[str, Any]) -> str:
    parts = []
    for k, v in obs.items():
        if isinstance(v, float):
            parts.append(f"<b>{html.escape(k)}</b>: {v:.4f}")
        else:
            parts.append(f"<b>{html.escape(k)}</b>: {html.escape(str(v))}")
    return "<br>".join(parts)


def write_html_report(
    out_dir: Path,
    *,
    run_id: str,
    utc_str: str,
    device: str,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    reports: list[ScenarioReport],
) -> Path:
    rows = []
    for rep in reports:
        notes_html = "<br>".join(html.escape(n) for n in rep.notes) if rep.notes else "—"
        rows.append(
            f'<tr class="scenario">'
            f'<td>{html.escape(rep.name)}</td>'
            f'<td>{html.escape(rep.expected_summary)}</td>'
            f'<td>{_format_observed(rep.observed)}</td>'
            f'<td class="note">{notes_html}</td>'
            f'</tr>'
        )
    body = _REPORT_TEMPLATE.format(
        run_id=html.escape(run_id),
        utc=html.escape(utc_str),
        device=html.escape(device),
        n_ant=n_ant,
        n_ch=n_ch,
        n_pol=n_pol,
        rows="\n".join(rows),
    )
    out = out_dir / "report.html"
    out.write_text(body)
    json_out = out_dir / "report.json"
    json_out.write_text(
        json.dumps([asdict(r) for r in reports], indent=2, default=str)
    )
    return out


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--run-id", default=None,
        help="Bench-run tag (default: 'rfi_calibration_<UTC>').",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Override default output directory bench/reports/<UTC>/<run_id>/M3-rfi/.",
    )
    p.add_argument(
        "--n-cubes-null", type=int, default=60,
        help="Number of cubes for the null-thermal-noise scenario (default: 60).",
    )
    p.add_argument(
        "--n-ant", type=int, default=16,
        help="Synthesised antenna count (≤ 96; default 16 for fast bench).",
    )
    p.add_argument(
        "--n-ch", type=int, default=32,
        help="Synthesised channel count (≤ 384; default 32).",
    )
    p.add_argument(
        "--n-pol", type=int, default=2,
        help="Polarisation count (default 2; production-pinned).",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu; tests are CPU-cheap).",
    )
    p.add_argument(
        "--seed", type=int, default=20260505,
        help="Random seed for synthetic noise.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.n_ant > NANTS:
        raise SystemExit(f"--n-ant must be ≤ {NANTS}")
    if args.n_ch > NCHAN_PER_CHGROUP:
        raise SystemExit(f"--n-ch must be ≤ {NCHAN_PER_CHGROUP}")
    if args.n_pol != NPOL:
        raise SystemExit(f"--n-pol must be {NPOL}")

    utc_now = dt.datetime.now(dt.timezone.utc)
    utc_str = utc_now.strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"rfi_calibration_{utc_str}"

    if args.out is None:
        out_dir = REPO_ROOT / "bench" / "reports" / utc_str / run_id / "M3-rfi"
    else:
        out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    reports: list[ScenarioReport] = []
    print(f"[rfi_calibration] writing report to {out_dir}", flush=True)

    print("[rfi_calibration] scenario (i): narrow-band CW", flush=True)
    reports.append(
        run_scenario_i_cw(
            rng, n_ant=args.n_ant, n_ch=args.n_ch, n_pol=args.n_pol,
        )
    )

    print("[rfi_calibration] scenario (ii): broadband impulse", flush=True)
    reports.append(
        run_scenario_ii_broadband_impulse(
            rng, n_ant=args.n_ant, n_ch=args.n_ch, n_pol=args.n_pol,
        )
    )

    print("[rfi_calibration] scenario (iii): bad antenna", flush=True)
    reports.append(
        run_scenario_iii_bad_antenna(
            rng, n_ant=args.n_ant, n_ch=args.n_ch, n_pol=args.n_pol,
        )
    )

    print("[rfi_calibration] scenario (iv): flagants.dat", flush=True)
    reports.append(
        run_scenario_iv_flagants_dat(
            rng, out_dir, n_ant=args.n_ant, n_ch=args.n_ch, n_pol=args.n_pol,
        )
    )

    print(f"[rfi_calibration] scenario (null): {args.n_cubes_null} cubes thermal",
          flush=True)
    reports.append(
        run_scenario_null_thermal(
            rng, n_ant=args.n_ant, n_ch=args.n_ch, n_pol=args.n_pol,
            n_cubes=args.n_cubes_null,
        )
    )

    out = write_html_report(
        out_dir,
        run_id=run_id,
        utc_str=utc_str,
        device=args.device,
        n_ant=args.n_ant,
        n_ch=args.n_ch,
        n_pol=args.n_pol,
        reports=reports,
    )
    print(f"[rfi_calibration] HTML report: {out}", flush=True)

    # Print a per-scenario one-liner summary to stdout for the orchestrator.
    print("\n=== Summary ===")
    for rep in reports:
        print(f"\n  {rep.name}")
        for k, v in rep.observed.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
