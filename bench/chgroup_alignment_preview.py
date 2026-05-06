"""bench/chgroup_alignment_preview.py — M3 chunk 7 (16-chgroup alignment).

Demonstrates that the corr-node fast-vis path preserves time alignment
across the 16 chgroups WITHOUT any explicit stage-2 alignment yet.
Stage-2 ``time_shift_corr_stage2`` (chunk 9 / plan §4.2 step 6) does
the BAND-DEPENDENT inter-chgroup alignment after coarse-DM dedispersion;
this preview shows that BEFORE stage 2, the fast-vis tiles produced by
each chgroup are clock-aligned per-block — i.e. tile ``k`` of chgroup
``g₁`` and tile ``k`` of chgroup ``g₂`` correspond to the SAME wall-
clock window of native samples (within rounding to ``t_int_fast_native``).

This is the per-block intra-cube alignment that the chunk-4 service
inherits FOR FREE from the synchronous fada → unpack → kernel pipeline:
all 16 chgroups consume their fada blocks at the same native cadence,
so tile boundaries land on the same native-sample multiples.

The bench plays back synthetic voltage blocks with a single high-amplitude
**impulse** packet at a known native-sample location, runs each chgroup
through ``corr_fast_integration.process_block``, captures the gridded-
cube power per fast-vis tile, and asserts that the impulse appears at
the SAME fast-vis tile index across all 16 chgroups (modulo at most
±1 tile of phase-wrap rounding).

This is a PREVIEW bench — it does NOT exercise the production stage-2
``time_shift_corr_stage2`` (which would compensate for the band-dependent
geometric delays + dispersion residuals). Stage-2 is owned by chunk 9.

CLI:

    python -m bench.chgroup_alignment_preview \
        [--report-dir bench/reports/<UTC>/M3-chgroup-alignment]
        [--n-grid 32]
        [--t-int-fast-native 8]
        [--impulse-packet 1000]
        [--chgroups 16]

Default knobs are chosen for fast CPU runtime (~30 s on h01) since
this is a synthetic-data demonstration; the production grid + cadence
are exercised by chunks 5-6 on real fixtures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    FADA_BYTES_PER_BLOCK,
    N_CHGROUP,
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.corr_fast_integration import (  # noqa: E402
    FastIntegrationConfig,
    _build_core_baseline_mask,
    build_context,
    process_block,
)
from dsart.services.slow_corr_kernel import (  # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)


LOG = logging.getLogger("chgroup_alignment_preview")


# ---------------------------------------------------------------------------
# Synthetic voltage helpers
# ---------------------------------------------------------------------------


def _synth_antpos_for_chgroup(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic DSA-110-like antpos. Same RNG seed across chgroups
    because the antpos is shared (one physical array; 16 chgroups
    correspond to 16 frequency bands of the SAME antennas, NOT to 16
    different physical arrays).
    """
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _synth_block_with_impulse(
    *,
    impulse_packet: int,
    seed: int = 20260506,
) -> np.ndarray:
    """Generate a synthetic fada block of raw int4 bytes with a single
    high-amplitude impulse packet.

    Args:
        impulse_packet: index into ``[0, NPACKETS_PER_BLOCK)`` where
            the impulse is placed. All bytes for that packet are set
            to 0x77 (= int4 +7 real, +7 imag — the largest legal int4
            magnitude after the legacy fluff scale).
        seed: RNG seed for the background noise floor.

    Returns:
        ``(FADA_BYTES_PER_BLOCK,) uint8`` raw bytes.
    """
    if not (0 <= impulse_packet < NPACKETS_PER_BLOCK):
        raise ValueError(
            f"impulse_packet={impulse_packet}, expected 0..{NPACKETS_PER_BLOCK - 1}"
        )
    rng = np.random.default_rng(seed)
    # Background floor: small int4 magnitudes via byte values around
    # 0x11 (= int4 +1 real, +1 imag). Deterministic across chgroups so
    # the only difference between chgroups is the per-chgroup gridder
    # geometry (which is irrelevant for the impulse-detection pin —
    # autos always show the impulse).
    raw = rng.integers(0, 32, size=FADA_BYTES_PER_BLOCK, dtype=np.uint8)

    # Impulse: bytes for the impulse packet → 0x77.
    # The fada layout is _FADA_VOLT_SHAPE = (NPACKETS_PER_BLOCK, NANTS,
    # NCHAN, NTIMES_PER_PACKET, NPOL); 1 byte per (ant, ch, t, p) = 1
    # complex int4 sample. The packet stride = NANTS * NCHAN *
    # NTIMES_PER_PACKET * NPOL bytes.
    packet_stride = NANTS * NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL
    impulse_offset = impulse_packet * packet_stride
    raw[impulse_offset:impulse_offset + packet_stride] = 0x77
    return raw


def _expected_fast_vis_tile(
    impulse_packet: int, t_int_fast_native: int,
) -> int:
    """Predict which fast-vis tile contains the impulse.

    The fast-vis tile width is ``t_int_fast_native`` NATIVE samples
    (= ``t_int_fast_native / NTIMES_PER_PACKET`` packets). Tile ``k``
    spans native samples ``[k * t_int_fast_native, (k+1) * t_int_fast_native)``,
    which is packets ``[k * t_int_fast_native / NTIMES_PER_PACKET,
    (k+1) * t_int_fast_native / NTIMES_PER_PACKET)``.
    """
    samples_per_tile = t_int_fast_native                                  # native samples
    packets_per_tile = samples_per_tile // NTIMES_PER_PACKET
    return impulse_packet // packets_per_tile


# ---------------------------------------------------------------------------
# Per-chgroup pipeline driver
# ---------------------------------------------------------------------------


def _run_one_chgroup(
    *,
    chgroup: int,
    raw: np.ndarray,
    n_grid: int,
    t_int_fast_native: int,
    obs_dec_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Run one synthetic block through the chunk-4 pipeline for ``chgroup``.

    Returns the per-fast-vis-tile total power
    ``vis_power[t] = sum |gridded[t, c]|²`` (real, length ``n_fast_vis``).
    The per-tile total power is what we use for impulse detection — the
    impulse packet contributes a large autocorrelation that grids into
    every cell, so the per-tile sum is dominated by the tile containing
    the impulse.
    """
    cfg = FastIntegrationConfig(
        chgroup=chgroup,
        obs_dec_rad=math.radians(obs_dec_deg),
        n_grid=n_grid,
        kernel_support=1,
        t_int_fast_native=t_int_fast_native,
        cal_path=None,
        rfi_enabled=False,                                                # synthetic uniform noise → SK fires; bypass
        static_sky_disabled=True,
    )
    e, n = _synth_antpos_for_chgroup(seed=42)                             # SAME antpos across chgroups
    core_mask = _build_core_baseline_mask(n_core=82)
    ctx = build_context(
        cfg, device=device,
        antpos_e=e, antpos_n=n,
        is_core_baseline_mask=core_mask,
    )
    out = process_block(raw, ctx=ctx, block_n=1)
    if out.gridded_minus_sky is None:
        raise RuntimeError(
            f"chgroup={chgroup} returned None gridded_minus_sky"
        )
    grid = out.gridded_minus_sky                                          # (n_fast_vis, N_filled) c64
    vis_power = (grid.real ** 2 + grid.imag ** 2).sum(dim=1)              # (n_fast_vis,) f32
    return vis_power.cpu()


# ---------------------------------------------------------------------------
# Plotting + report
# ---------------------------------------------------------------------------


def _save_alignment_heatmap(
    power_per_chgroup: torch.Tensor,                                      # (N_CHGROUP, n_fast_vis)
    *,
    expected_tile: int,
    out_path: Path,
) -> None:
    """Save the alignment heatmap PNG.

    Each row is a chgroup; columns are fast-vis tiles. The expected
    impulse tile is overlaid in red.
    """
    import matplotlib.pyplot as plt
    n_chg, n_fv = power_per_chgroup.shape
    # Per-row z-score so we can compare relative power across chgroups
    # (absolute power varies because gridder N_filled differs per
    # chgroup geometry).
    arr = power_per_chgroup.numpy().astype(np.float64)
    arr_norm = arr / (arr.max(axis=1, keepdims=True) + 1e-30)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        arr_norm, aspect="auto",
        cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
    )
    ax.axvline(
        expected_tile + 0.5, color="red", linestyle=":",
        label=f"expected tile = {expected_tile}",
    )
    ax.set_xlabel("fast-vis tile index")
    ax.set_ylabel("chgroup")
    ax.set_title(
        f"M3 chunk 7: 16-chgroup alignment preview\n"
        f"per-row-normalised total grid power; impulse should peak at "
        f"the red-dashed tile across all chgroups"
    )
    ax.set_yticks(range(0, n_chg))
    fig.colorbar(im, ax=ax, label="power (per-row max-normalised)")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _save_report_html(
    *,
    out_path: Path,
    summary: dict,
    heatmap_rel_path: str,
) -> None:
    rows = "\n".join(
        f"<tr><td>{g}</td><td>{summary['peak_tile_per_chgroup'][g]}</td>"
        f"<td>{summary['peak_offset_per_chgroup'][g]:+d}</td></tr>"
        for g in range(summary["n_chgroup"])
    )
    pass_str = "PASS" if summary["max_abs_offset_tiles"] <= 1 else "FAIL"
    pass_color = "#2a7" if summary["max_abs_offset_tiles"] <= 1 else "#c33"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>M3 chunk 7 — chgroup alignment preview</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; }} td, th {{ padding: 0.3em 0.8em; border-bottom: 1px solid #ddd; }}
.pass {{ color: {pass_color}; font-weight: bold; }}
img {{ max-width: 100%; }}
</style></head><body>
<h1>M3 chunk 7 — 16-chgroup alignment preview</h1>
<p><span class="pass">{pass_str}</span> — max absolute offset {summary['max_abs_offset_tiles']} tile(s) across {summary['n_chgroup']} chgroups (criterion ≤ 1).</p>
<h2>Configuration</h2>
<ul>
<li>Impulse packet index: <code>{summary['impulse_packet']}</code> (= native sample {summary['impulse_native_sample']})</li>
<li>Fast-vis tile width: <code>{summary['t_int_fast_native']}</code> native samples (= {summary['t_int_fast_us']:.3f} µs)</li>
<li>Predicted impulse tile: <code>{summary['expected_tile']}</code></li>
<li>Grid size: <code>{summary['n_grid']}×{summary['n_grid']}</code></li>
</ul>
<h2>Alignment heatmap</h2>
<img src="{heatmap_rel_path}" alt="alignment heatmap">
<h2>Per-chgroup peak tile</h2>
<table><thead><tr><th>chgroup</th><th>peak tile</th><th>offset from expected</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Notes</h2>
<p>This is a PREVIEW bench — it does NOT exercise stage-2 inter-chgroup alignment
(<code>time_shift_corr_stage2</code>). Production stage-2 alignment lands in chunk 9 along with
the multi-DM-trial integration (per F25 in <code>M3_PLAN_FIXES.md</code>).</p>
</body></html>
"""
    out_path.write_text(html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--report-dir", type=Path, default=None,
                   help="output dir; default = bench/reports/<UTC>/M3-chgroup-alignment/")
    p.add_argument("--n-grid", type=int, default=32,
                   help="grid side length (default: 32 for fast CPU runtime)")
    p.add_argument("--t-int-fast-native", type=int, default=8,
                   help="fast-vis tile width in native samples (default: 8 = 262.144 µs)")
    p.add_argument("--impulse-packet", type=int, default=1000,
                   help="packet index where the impulse is placed (default: 1000 ~ mid-block)")
    p.add_argument("--obs-dec-deg", type=float, default=53.85,
                   help="observing dec for the F21 cal phase (default: 53.85 ~ burst-test source)")
    p.add_argument("--chgroups", type=int, default=N_CHGROUP,
                   help=f"number of chgroups to test (default: {N_CHGROUP})")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / cpu (default: auto — cuda when available; "
                        "16-chgroup CPU run takes ~20 min, GPU ~30 s)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    LOG.info("device = %s", device)
    if args.report_dir is None:
        utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.report_dir = (
            REPO_ROOT / "bench" / "reports" / utc / "M3-chgroup-alignment"
        )
    args.report_dir.mkdir(parents=True, exist_ok=True)

    expected_tile = _expected_fast_vis_tile(
        args.impulse_packet, args.t_int_fast_native,
    )
    LOG.info(
        "chgroup-alignment preview: %d chgroups, n_grid=%d, "
        "t_int_fast_native=%d, impulse_packet=%d (expected tile=%d)",
        args.chgroups, args.n_grid, args.t_int_fast_native,
        args.impulse_packet, expected_tile,
    )

    raw = _synth_block_with_impulse(impulse_packet=args.impulse_packet)
    LOG.info("synth block size: %d bytes (%d expected)", raw.nbytes, FADA_BYTES_PER_BLOCK)

    power_rows: list[torch.Tensor] = []
    peak_tile_per_chgroup: list[int] = []
    for chg in range(args.chgroups):
        LOG.info("running chgroup %d/%d ...", chg, args.chgroups - 1)
        vis_power = _run_one_chgroup(
            chgroup=chg, raw=raw,
            n_grid=args.n_grid, t_int_fast_native=args.t_int_fast_native,
            obs_dec_deg=args.obs_dec_deg, device=device,
        )
        power_rows.append(vis_power)
        peak_tile = int(torch.argmax(vis_power).item())
        peak_tile_per_chgroup.append(peak_tile)
        LOG.info("  chgroup %d peak tile = %d (expected %d, offset %+d)",
                 chg, peak_tile, expected_tile, peak_tile - expected_tile)

    power_per_chgroup = torch.stack(power_rows, dim=0)                    # (N_CHGROUP, n_fast_vis)
    offsets = [t - expected_tile for t in peak_tile_per_chgroup]
    max_abs_offset = max(abs(o) for o in offsets)

    summary = {
        "n_chgroup": args.chgroups,
        "n_grid": args.n_grid,
        "t_int_fast_native": args.t_int_fast_native,
        "t_int_fast_us": args.t_int_fast_native * 32.768,
        "impulse_packet": args.impulse_packet,
        "impulse_native_sample": args.impulse_packet * NTIMES_PER_PACKET,
        "expected_tile": expected_tile,
        "peak_tile_per_chgroup": peak_tile_per_chgroup,
        "peak_offset_per_chgroup": offsets,
        "max_abs_offset_tiles": max_abs_offset,
    }

    heatmap_path = args.report_dir / "chgroup_alignment_heatmap.png"
    _save_alignment_heatmap(
        power_per_chgroup,
        expected_tile=expected_tile, out_path=heatmap_path,
    )
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _save_report_html(
        out_path=args.report_dir / "report.html",
        summary=summary,
        heatmap_rel_path=heatmap_path.name,
    )

    if max_abs_offset > 1:
        LOG.error(
            "CHGROUP ALIGNMENT FAIL: max absolute tile offset = %d "
            "(criterion ≤ 1)", max_abs_offset,
        )
        return 1
    LOG.info(
        "CHGROUP ALIGNMENT PASS: max absolute tile offset = %d "
        "(criterion ≤ 1) across %d chgroups",
        max_abs_offset, args.chgroups,
    )
    LOG.info("wrote %s", args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
