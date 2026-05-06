#!/usr/bin/env python3
"""bench/rfi_warmup.py — M3 RFI flagger cold-start warmup bench (chunk 3c).

Plan §8 M3 DoD line 2280: cold-start the flagger, confirm
``flags.bit4 = rfi_warming_up`` is set in the transport header for the
first ``5·τ_B`` cubes (where ``τ_B = 30 s`` is the bandpass-outlier
MAD warmup window) and clears thereafter; confirm SK + group-outlier
remain active during the window while bandpass-outlier is bypassed.

For chunk 3c this is a smaller, scoped test that exercises the
warmup state machine in :class:`dsart.rfi.combine.RFIFlagger` against
the chunk-3c :class:`dsart.rfi.combine.MockTransportHeader`. The
parent M3 agent will integrate the same state machine into the live
``corr_fast_compute`` service and re-run a full-shape variant against
the production transport header.

CLI:

    python -m bench.rfi_warmup [--warmup-cubes 8] [--n-extra-cubes 8]
        [--out <dir>]

Output:

    bench/reports/<UTC>/<run_id>/M3-rfi-warmup/
      report.html      — summary table
      header_flags.csv — per-cube (cube_idx, header.flags, warmup,
                         flag_fraction_total, source_tags_summary)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
    RFI_BANDPASS_TAU_S,
)
from dsart.rfi import (  # noqa: E402
    DEFAULT_M_VALUES,
    FlagSourceBit,
    MockTransportHeader,
    RFIFlagger,
    compute_autos_from_complex,
)


# ---------------------------------------------------------------------------
# Synthetic cube
# ---------------------------------------------------------------------------


def _gauss_voltages(
    rng: np.random.Generator,
    *,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    n_time: int = 4096,
    sigma: float = 1.0,
) -> torch.Tensor:
    re = rng.normal(0.0, sigma / math.sqrt(2.0),
                    size=(n_ant, n_ch, n_pol, n_time))
    im = rng.normal(0.0, sigma / math.sqrt(2.0),
                    size=(n_ant, n_ch, n_pol, n_time))
    return torch.as_tensor((re + 1j * im).astype(np.complex64))


@dataclass
class CubeRecord:
    cube_idx: int
    warmup: bool
    header_flags: int
    flag_fraction_total: float
    sk_flag_count: int
    bp_flag_count: int
    gr_flag_count: int


# ---------------------------------------------------------------------------
# Run the warmup state machine
# ---------------------------------------------------------------------------


def run_warmup_sweep(
    *,
    warmup_cubes: int,
    n_extra_cubes: int,
    n_ant: int,
    n_ch: int,
    n_pol: int,
    seed: int,
) -> tuple[list[CubeRecord], int]:
    """Drive ``warmup_cubes + n_extra_cubes`` cubes through the
    flagger; return per-cube records and the warmup-cube count
    actually configured.

    Bandpass-outlier should fire in zero cells during the warmup
    window (SK and group-outlier still active); after the window
    bandpass-outlier is reactivated.
    """
    flagger = RFIFlagger(
        flagants_path=None,
        warmup_cubes=warmup_cubes,
    )
    # Patch the flagants_mask down to the n_ant scenario size.
    flagger._flagants_mask = torch.zeros(NANTS, dtype=torch.bool)  # noqa: SLF001

    rng = np.random.default_rng(seed)
    records: list[CubeRecord] = []
    total_cubes = warmup_cubes + n_extra_cubes
    for c in range(total_cubes):
        voltages = _gauss_voltages(
            rng, n_ant=n_ant, n_ch=n_ch, n_pol=n_pol,
        )
        autos = compute_autos_from_complex(voltages, m_values=DEFAULT_M_VALUES)
        header = MockTransportHeader()
        result = flagger.flag_block(
            real=None, imag=None,
            autos_override=autos,
            update_header=header,
        )
        # Per-source counts.
        tags = result.source_tags
        sk_n = int(((tags & int(FlagSourceBit.SK)) != 0).sum().item())
        bp_n = int(((tags & int(FlagSourceBit.BANDPASS_OUTLIER)) != 0).sum().item())
        gr_n = int(((tags & int(FlagSourceBit.GROUP_OUTLIER)) != 0).sum().item())
        records.append(
            CubeRecord(
                cube_idx=c,
                warmup=result.warmup,
                header_flags=int(header.flags),
                flag_fraction_total=result.flag_fraction_total,
                sk_flag_count=sk_n,
                bp_flag_count=bp_n,
                gr_flag_count=gr_n,
            )
        )
    return records, warmup_cubes


# ---------------------------------------------------------------------------
# CSV / HTML report
# ---------------------------------------------------------------------------


def write_csv(out_dir: Path, records: list[CubeRecord]) -> Path:
    p = out_dir / "header_flags.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cube_idx", "warmup", "header_flags",
            "flag_fraction_total",
            "sk_flag_count", "bp_flag_count", "gr_flag_count",
        ])
        for r in records:
            w.writerow([
                r.cube_idx, int(r.warmup), r.header_flags,
                f"{r.flag_fraction_total:.6f}",
                r.sk_flag_count, r.bp_flag_count, r.gr_flag_count,
            ])
    return p


def write_html(
    out_dir: Path,
    *,
    run_id: str,
    utc_str: str,
    warmup_cubes: int,
    records: list[CubeRecord],
    n_ant: int,
    n_ch: int,
    n_pol: int,
) -> Path:
    rows = []
    for r in records:
        rows.append(
            f"<tr>"
            f"<td>{r.cube_idx}</td>"
            f"<td>{'<b>warmup</b>' if r.warmup else 'steady'}</td>"
            f"<td>0x{r.header_flags:02x}</td>"
            f"<td>{r.flag_fraction_total:.4f}</td>"
            f"<td>{r.sk_flag_count}</td>"
            f"<td>{r.bp_flag_count}</td>"
            f"<td>{r.gr_flag_count}</td>"
            f"</tr>"
        )

    expected = textwrap.dedent(f"""\
        Cubes 0..{warmup_cubes - 1}: warmup=1, header.bit4 set
        (0x10), bandpass-outlier bypassed (bp_flag_count = 0); SK +
        group-outlier active.

        Cubes {warmup_cubes}..{len(records) - 1}: warmup=0, header.bit4
        clear (0x00), bandpass-outlier active.
    """).strip()

    body = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en"><head>
        <meta charset="utf-8">
        <title>M3 RFI warmup — {html.escape(run_id)}</title>
        <style>
        body {{ font-family: system-ui, sans-serif; max-width: 1100px;
                margin: 1em auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 6px 10px; border: 1px solid #ccc;
                  font-size: 13px; text-align: left; }}
        th {{ background: #f0f0f0; }}
        .note {{ color: #666; font-size: 13px; }}
        </style></head><body>
        <h1>M3 RFI flagger — cold-start warmup</h1>
        <p class="note">Run id: <code>{html.escape(run_id)}</code> · UTC:
        <code>{html.escape(utc_str)}</code> ·
        warmup_cubes = {warmup_cubes} · τ<sub>B</sub> production default
        = {RFI_BANDPASS_TAU_S:.0f} s · synthesised n_ant×n_ch×n_pol =
        {n_ant}×{n_ch}×{n_pol}</p>

        <h2>Expected behaviour</h2>
        <pre>{html.escape(expected)}</pre>

        <h2>Observed per-cube</h2>
        <table>
          <thead><tr>
            <th>cube_idx</th><th>state</th><th>header.flags</th>
            <th>flag_frac_total</th>
            <th>SK flags</th><th>BP flags</th><th>GR flags</th>
          </tr></thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        <p class="note">No PASS/FAIL banner — operator inspects the
        warmup → steady transition row.</p>
        </body></html>
    """)
    p = out_dir / "report.html"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--warmup-cubes", type=int, default=8,
        help=(
            "Number of cubes during which to assert rfi_warming_up "
            "and bypass bandpass-outlier. Production default in "
            "common/constants.RFI_BANDPASS_WARMUP_CUBES_DEFAULT is "
            "≈ 1118 (5·30s/134ms); the bench uses a small value for "
            "fast inspection (default 8)."
        ),
    )
    p.add_argument(
        "--n-extra-cubes", type=int, default=8,
        help="Cubes to run after the warmup window clears (default 8).",
    )
    p.add_argument("--n-ant", type=int, default=8)
    p.add_argument("--n-ch", type=int, default=16)
    p.add_argument("--n-pol", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260505)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-id", default=None)
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
    run_id = args.run_id or f"rfi_warmup_{utc_str}"
    out_dir = (args.out or
               REPO_ROOT / "bench" / "reports" / utc_str / run_id / "M3-rfi-warmup")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rfi_warmup] writing report to {out_dir}", flush=True)
    records, warmup_cubes = run_warmup_sweep(
        warmup_cubes=args.warmup_cubes,
        n_extra_cubes=args.n_extra_cubes,
        n_ant=args.n_ant,
        n_ch=args.n_ch,
        n_pol=args.n_pol,
        seed=args.seed,
    )
    csv_path = write_csv(out_dir, records)
    html_path = write_html(
        out_dir,
        run_id=run_id, utc_str=utc_str,
        warmup_cubes=warmup_cubes, records=records,
        n_ant=args.n_ant, n_ch=args.n_ch, n_pol=args.n_pol,
    )
    print(f"[rfi_warmup] CSV : {csv_path}", flush=True)
    print(f"[rfi_warmup] HTML: {html_path}", flush=True)

    # Per-cube one-liner for stdout / orchestrator
    print("\ncube_idx | warmup | flags  | bp_count | sk_count | gr_count")
    print("-" * 60)
    for r in records:
        print(
            f"{r.cube_idx:8d} | {int(r.warmup):6d} | 0x{r.header_flags:02x}   | "
            f"{r.bp_flag_count:8d} | {r.sk_flag_count:8d} | {r.gr_flag_count:8d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
