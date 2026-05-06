"""bench/m3_emit_m5_fixtures.py — emit M5 fixture .npz files.

Replays the M3 voltage-fixture-replay path (chunks 5+6) per chgroup and
captures the **post-grid sparse-COO cubes + SparsityPattern UV table +
truth metadata** to per-chgroup ``.npz`` files. M5 (search-side) consumes
these as the canonical inputs for ``bench/voltage_fixture_search.py`` —
the operator gate at plan §8 line 2330.

Per ``PARALLEL_AGENTS.md`` §1: "M3 → M5 coupling point is
``bench/voltage_fixture_search.py`` ... M5 develops up to that point
against ``cube_injection_detector.py`` and only fires the voltage-fixture
gate once M3 has emitted a captured transport-TX ``.npz`` set."

Per plan §8 lines 1841-1843 (M3 voltage-fixture sub-DoDs):

* M3 fast-corr — continuum imager check (``run_id=0319``): replay → full
  ``corr_fast_compute`` → transport-TX captured to ``.npz`` per chgroup.
* M3 fast-corr — burst dedispersion check (``run_id=250924mptq``):
  replay → full ``corr_fast_compute`` → transport-TX ``.npz`` per chgroup.
* M3 fast-corr — 16-chgroup stage-2 alignment preview (burst fixture, all
  16 chgroups): same as above but with all chgroups stitched into the
  manifest.

Output layout (canonical M5 fixture root)::

    /home/ubuntu/data/m5_fixtures/<run_id>/
        chgroup00.npz
        chgroup01.npz
        ...
        chgroup15.npz
        manifest.json

Each ``chgroup<NN>.npz`` follows the F26 transport-TX **sparse-COO**
contract (one of the two shapes ``TransportTx._transmit_one_cube``
auto-detects)::

    vis_cube_sparse:    complex64 (N_DM, n_fv_total, N_filled)

plus the SparsityPattern fields (``ix_row, ix_col, pattern_id, n_grid,
n_filled, dec_deg_quant, kernel_support, antpos_hash,
chgroup_table_hash``) so M5's receiver can verify ``pattern_id`` locally
+ scatter back to the dense grid; plus antpos arrays and the cal_path
hash for full provenance + truth metadata extracted from ``T2_*.json``.

Both fixtures land at the chunks-5/6 single-DM operating point — i.e.
``N_DM == 1`` here. M5 can lift the leading axis and treat the cube as
2D ``(n_fv_total, N_filled)`` for legacy consumers.

Default operating point (matches chunks 5/6 + PARALLEL_AGENTS.md §5.1):

* Continuum (0319): ``--t-int-fast-native 4096`` (~134 ms cadence —
  4 fast-vis tiles per fada block, 16 blocks total). Phase center =
  source dec (F21).
* Burst (250924mptq): ``--t-int-fast-native 64`` (= 2097.152 µs;
  consumer-GPU memory budget per F31). Phase center = source dec.
  Replays 8 fada blocks per chgroup (covers the burst + dispersion smear).

Run::

    # 0319 continuum (fast)
    python -m bench.m3_emit_m5_fixtures --run-id 0319

    # 250924mptq burst (~25 min on h01 GPU 0)
    python -m bench.m3_emit_m5_fixtures --run-id 250924mptq
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from dsart.common.constants import (
    NATIVE_SAMPLE_US,
    NBASE,
    PHI_LAT_OVRO_DEG,
)
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    NoOpCoarseDM,
    load_antpos_from_cal_blob,
)
from bench._corr_fast_replay import (
    accumulate_chgroup_grids,
    replay_chgroup,
)


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixture catalogue
# ---------------------------------------------------------------------------


_FIXTURES = {
    "0319": {
        "kind": "continuum",
        "src_name": "0319+415",
        "voltage_root": Path("/home/ubuntu/data/voltages/0319"),
        "default_t_int_fast_native": 4096,
        "default_max_blocks": 16,
        # 0319 fixture has 15 chgroups (sb12 missing); replay only what's there
        "chgroups": [g for g in range(16) if g != 12],
    },
    "250924mptq": {
        "kind": "burst",
        "src_name": "FRB 250924mptq",
        "voltage_root": Path("/home/ubuntu/data/voltages/250924mptq"),
        "default_t_int_fast_native": 64,   # F31 budget on h01 GPU 0
        "default_max_blocks": 8,
        "chgroups": list(range(16)),
    },
}


# ---------------------------------------------------------------------------
# T2_*.json → truth metadata
# ---------------------------------------------------------------------------


def _parse_t2_json(path: Path) -> dict:
    """Read ``T2_<run_id>.json`` → ``{src_name, ra_deg, dec_deg, mjd_trigger,
    dm_pc_cc, t2_snr}``. ``dm_pc_cc`` is NaN for continuum fixtures (T2
    file may be absent or omit the field)."""
    if not path.is_file():
        return {
            "src_name": "",
            "ra_deg": float("nan"),
            "dec_deg": float("nan"),
            "mjd_trigger": float("nan"),
            "dm_pc_cc": float("nan"),
            "t2_snr": float("nan"),
        }
    with path.open("r") as fh:
        raw = json.load(fh)
    return {
        "src_name": str(raw.get("src_name", "")),
        "ra_deg": float(raw["ra"]) if "ra" in raw else float(raw.get("ra_deg", float("nan"))),
        "dec_deg": float(raw["dec"]) if "dec" in raw else float(raw.get("dec_deg", float("nan"))),
        "mjd_trigger": float(raw["mjds"]) if "mjds" in raw else float(raw.get("mjd_trigger", float("nan"))),
        "dm_pc_cc": float(raw.get("dm", raw.get("dm_pc_cc", float("nan")))),
        "t2_snr": float(raw.get("snr", raw.get("t2_snr", float("nan")))),
    }


# ---------------------------------------------------------------------------
# Per-chgroup capture
# ---------------------------------------------------------------------------


def _emit_chgroup_npz(
    *,
    run_id: str,
    fixture_meta: dict,
    chgroup: int,
    t_int_fast_native: int,
    max_blocks: int,
    obs_dec_deg: float,
    src_truth: dict,
    out_dir: Path,
    device: torch.device,
    git_sha: str,
) -> dict:
    """Replay one chgroup → write ``<out_dir>/chgroup<NN>.npz``.

    Returns a small per-chgroup record for the top-level manifest.
    """
    voltage_root = fixture_meta["voltage_root"]
    voltage_dir = voltage_root / "voltages"
    cals_dir = voltage_root / "cals"
    sb_str = f"{chgroup:02d}"
    voltage_files = list(voltage_dir.glob(f"*sb{sb_str}_data.out"))
    cal_files = list(cals_dir.glob(f"beamformer_weights_sb{sb_str}*.dat"))
    if not voltage_files or not cal_files:
        raise FileNotFoundError(
            f"chgroup={chgroup}: voltage or cal not found in {voltage_root}"
        )
    voltage_path = voltage_files[0]
    cal_path = cal_files[0]

    LOG.info(
        "chgroup %d: replaying %s (%d blocks, t_int_fast_native=%d)",
        chgroup, voltage_path.name, max_blocks, t_int_fast_native,
    )
    t0 = time.perf_counter()
    cfg = FastIntegrationConfig(
        chgroup=chgroup,
        t_int_fast_native=t_int_fast_native,
        cal_path=cal_path,
        cal_mode="phase",
        observing_dec_deg=obs_dec_deg,
        rfi_disabled=True,
        static_sky_disabled=(fixture_meta["kind"] == "burst"),
    )
    ctx, outputs = replay_chgroup(
        voltage_path=voltage_path,
        cal_path=cal_path,
        cfg=cfg,
        max_blocks=max_blocks,
        device=device,
    )
    if not outputs:
        raise RuntimeError(f"chgroup={chgroup}: no blocks produced")

    pattern = ctx.gridder.pattern
    n_filled = int(pattern.n_filled)
    cube_2d = accumulate_chgroup_grids(outputs, n_filled=n_filled).cpu().numpy()
    if cube_2d.dtype != np.complex64:
        cube_2d = cube_2d.astype(np.complex64)
    n_fv_total, _n_f = cube_2d.shape
    cube_3d = cube_2d.reshape(1, n_fv_total, n_filled)  # (N_DM=1, n_fv, N_filled)

    antpos_e, antpos_n, core_mask = load_antpos_from_cal_blob(cal_path)
    cell_lambda = float(ctx.gridder.cell_lambda)

    out_path = out_dir / f"chgroup{chgroup:02d}.npz"
    np.savez_compressed(
        out_path,
        # ----- value channel (the transport-TX payload) -----
        vis_cube_sparse=cube_3d,                # (N_DM=1, n_fv_total, N_filled) complex64
        # ----- SparsityPattern (M5 verifies pattern_id locally) -----
        ix_row=pattern.ix_row.astype(np.uint16),
        ix_col=pattern.ix_col.astype(np.uint16),
        pattern_id=np.uint64(pattern.pattern_id),
        n_grid=np.int32(pattern.n_grid),
        n_filled=np.int32(pattern.n_filled),
        dec_deg_quant=np.float32(pattern.dec_deg_quant),
        kernel_support=np.int32(pattern.kernel_support),
        antpos_hash=np.uint64(pattern.antpos_hash),
        chgroup_table_hash=np.uint64(pattern.chgroup_table_hash),
        # ----- antpos + core mask (so M5 can re-derive locally) -----
        antpos_e=antpos_e.astype(np.float32),
        antpos_n=antpos_n.astype(np.float32),
        is_core_baseline_mask=core_mask.astype(np.bool_),
        # ----- cadence + chgroup -----
        chgroup=np.int32(chgroup),
        t_int_fast_native=np.int32(t_int_fast_native),
        t_int_fast_us=np.float64(t_int_fast_native * NATIVE_SAMPLE_US),
        n_fv_total=np.int32(n_fv_total),
        n_blocks_processed=np.int32(len(outputs)),
        cell_lambda=np.float32(cell_lambda),
        phi_lat_ovro_deg=np.float32(PHI_LAT_OVRO_DEG),
        obs_dec_deg=np.float32(obs_dec_deg),
        # ----- truth metadata (T2_*.json, may be NaN for continuum) -----
        src_kind=fixture_meta["kind"],
        src_name=fixture_meta["src_name"],
        src_ra_deg=np.float64(src_truth["ra_deg"]),
        src_dec_deg=np.float64(src_truth["dec_deg"]),
        src_mjd_trigger=np.float64(src_truth["mjd_trigger"]),
        src_dm_pc_cc=np.float64(src_truth["dm_pc_cc"]),
        src_t2_snr=np.float64(src_truth["t2_snr"]),
        # ----- provenance -----
        run_id=run_id,
        cal_path=str(cal_path),
        voltage_path=str(voltage_path),
        git_sha=git_sha,
        utc_iso=datetime.now(timezone.utc).strftime("%FT%TZ"),
    )

    elapsed_s = time.perf_counter() - t0
    bytes_on_disk = out_path.stat().st_size
    LOG.info(
        "chgroup %d wrote %s (%.1f MiB, n_fv=%d, N_filled=%d, %.1f s)",
        chgroup, out_path, bytes_on_disk / (1 << 20),
        n_fv_total, n_filled, elapsed_s,
    )

    # Free GPU between chgroups (F31 mitigation)
    del ctx, outputs, cube_2d, cube_3d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "chgroup": chgroup,
        "voltage_path": str(voltage_path),
        "cal_path": str(cal_path),
        "n_fv_total": int(n_fv_total),
        "n_filled": int(n_filled),
        "n_blocks_processed": int(len(outputs)),
        "cell_lambda": float(cell_lambda),
        "pattern_id_hex": f"0x{int(pattern.pattern_id):016x}",
        "n_grid": int(pattern.n_grid),
        "out_path": str(out_path),
        "out_bytes": int(bytes_on_disk),
        "elapsed_s": float(elapsed_s),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--run-id", choices=sorted(_FIXTURES.keys()), required=True,
        help="Voltage-fixture run id (canonical key under /home/ubuntu/data/voltages/).",
    )
    p.add_argument(
        "--out-root", type=Path,
        default=Path("/home/ubuntu/data/m5_fixtures"),
        help="Output root; per-chgroup files land at <out_root>/<run_id>/chgroupNN.npz.",
    )
    p.add_argument(
        "--t-int-fast-native", type=int, default=None,
        help="Fast-vis cadence in NATIVE samples (32.768 µs each). "
             "Defaults to fixture-specific value.",
    )
    p.add_argument(
        "--max-blocks", type=int, default=None,
        help="Number of fada blocks per chgroup. Defaults to fixture-specific value.",
    )
    p.add_argument(
        "--obs-dec-deg", type=float, default=None,
        help="Observing declination (degrees) for F21 cal-DEC phase fold. "
             "Defaults to source dec from T2_*.json.",
    )
    p.add_argument(
        "--chgroups", type=str, default=None,
        help="Comma-separated chgroup indices to process (default = all in fixture).",
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
    )
    p.add_argument(
        "--log-level", default="INFO",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    fixture = _FIXTURES[args.run_id]
    voltage_root = fixture["voltage_root"]
    if not voltage_root.is_dir():
        LOG.error("voltage_root does not exist: %s", voltage_root)
        return 2

    t_int_fast_native = args.t_int_fast_native or fixture["default_t_int_fast_native"]
    max_blocks = args.max_blocks or fixture["default_max_blocks"]

    src_truth = _parse_t2_json(voltage_root / "voltages" / f"T2_{args.run_id}.json")
    obs_dec_deg = (
        args.obs_dec_deg
        if args.obs_dec_deg is not None
        else (src_truth["dec_deg"] if not np.isnan(src_truth["dec_deg"]) else PHI_LAT_OVRO_DEG)
    )

    chgroups = (
        [int(g) for g in args.chgroups.split(",")]
        if args.chgroups is not None
        else fixture["chgroups"]
    )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )

    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    git_sha = _git_sha(repo_root)

    LOG.info(
        "M5 fixture emit: run_id=%s kind=%s chgroups=%s "
        "t_int_fast_native=%d max_blocks=%d obs_dec=%.4f device=%s",
        args.run_id, fixture["kind"], chgroups, t_int_fast_native,
        max_blocks, obs_dec_deg, device,
    )
    LOG.info(
        "truth: src=%s ra=%.4f dec=%.4f mjd=%.6f dm=%.3f t2_snr=%.1f",
        src_truth["src_name"] or fixture["src_name"],
        src_truth["ra_deg"], src_truth["dec_deg"],
        src_truth["mjd_trigger"], src_truth["dm_pc_cc"],
        src_truth["t2_snr"],
    )

    per_chgroup_records: list[dict] = []
    for chgroup in chgroups:
        try:
            rec = _emit_chgroup_npz(
                run_id=args.run_id,
                fixture_meta=fixture,
                chgroup=chgroup,
                t_int_fast_native=t_int_fast_native,
                max_blocks=max_blocks,
                obs_dec_deg=obs_dec_deg,
                src_truth=src_truth,
                out_dir=out_dir,
                device=device,
                git_sha=git_sha,
            )
        except FileNotFoundError as exc:
            LOG.warning("chgroup %d skipped: %s", chgroup, exc)
            continue
        per_chgroup_records.append(rec)

    manifest = {
        "milestone": "M3",
        "purpose": "M5 voltage-fixture-search inputs (M3 → M5 coupling, "
                   "PARALLEL_AGENTS.md §1)",
        "run_id": args.run_id,
        "src_kind": fixture["kind"],
        "src_name": fixture["src_name"],
        "src_truth": src_truth,
        "obs_dec_deg": float(obs_dec_deg),
        "t_int_fast_native": int(t_int_fast_native),
        "t_int_fast_us": float(t_int_fast_native * NATIVE_SAMPLE_US),
        "n_chgroups": len(per_chgroup_records),
        "chgroups": [r["chgroup"] for r in per_chgroup_records],
        "per_chgroup": per_chgroup_records,
        "git_sha": git_sha,
        "utc_iso": datetime.now(timezone.utc).strftime("%FT%TZ"),
        "phi_lat_ovro_deg": float(PHI_LAT_OVRO_DEG),
        "n_baselines": int(NBASE),
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    LOG.info(
        "wrote manifest %s (chgroups=%d total bytes=%.1f MiB)",
        manifest_path, len(per_chgroup_records),
        sum(r["out_bytes"] for r in per_chgroup_records) / (1 << 20),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
