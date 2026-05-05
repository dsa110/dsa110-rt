"""bench/run_0319_pipeline.py — drive the slow correlator + meridian
fringestop across all 0319+415 sb voltage dumps (D17 acceptance run).

For each sb in {00..15} \\ {12}:
  1. Build a manifest fixture dir (symlinks the .out file as
     ``fl_0319bbb_chgroup0.out`` + writes a ``manifest.yaml`` matching
     ``tests/fixtures/voltage_fixture_manifest.schema.json``).
  2. Build a per-sb dsamfs param yaml with the correct
     ``ch0: {<hostname>: 1024 + sb*384}`` — meridian_fringestop reads
     ``socket.gethostname()`` to look up the entry.
  3. Run ``bench/voltage_fixture_slow_corr.py`` with:
        * ``--apply-cal <cals/beamformer_weights_sb<NN>...dat>``
        * ``--meridian-param <param.yaml>``
        * ``--meridian-pt-dec-deg 41.51169444`` (D17 wrapper)
        * ``--uvh5-out <work-dir>/0319_sb<NN>.uvh5``
  4. Each sb produces one UVH5 file. Concatenation is a follow-up step
     left to the user / a small post-processing script.

Inputs (defaults match the 2026-05-05 transfer):
  * voltage dumps: ``/home/ubuntu/data/voltages/0319/voltages/0319bbb_sb<NN>_data.out``
  * cal blobs:     ``/home/ubuntu/data/voltages/0319/cals/beamformer_weights_sb<NN>_0319+415.dat``
  * source meta:   ``/home/ubuntu/data/voltages/0319/voltages/T2_0319bbb.json``
  * cal yaml:      ``/home/ubuntu/data/voltages/0319/cals/beamformer_weights_0319+415.yaml``

This script is RUN on h01 (where the data lives + the dsa110-rt env
exists). It dispatches the per-sb sub-runs sequentially (no
parallelism — they share the same ``fada``/``bada`` ring keys).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


# --- defaults ------------------------------------------------------------

DEFAULT_VOLTAGE_ROOT = Path("/home/ubuntu/data/voltages/0319")
DEFAULT_WORK_ROOT = Path("/home/ubuntu/data/vikram/0319_uvh5")

#: chgroup-0 system channel offset (matches CHGROUP_CH0[0] in
#: ``dsart.common.constants``). sb<N>'s ch0 = 1024 + N * 384.
CH0_BASE = 1024
NCHAN_PER_CHGROUP = 384

# 0319+415 source coordinates (from T2_0319bbb.json transferred 2026-05-05).
SRC_RA_DEG_DEFAULT = 49.9506667
SRC_DEC_DEG_DEFAULT = 41.51169444
SRC_MJD_DEFAULT = 61108.99867338988
SRC_SPECNUM_DEFAULT = 21094410

#: All sbs in the transferred dataset (sb12 is the missing h18).
ALL_SBS_DEFAULT = [
    f"{n:02d}" for n in range(16) if n != 12
]


# --- helpers -------------------------------------------------------------


def _mjd_to_utc_iso(mjd: float) -> str:
    """MJD → ISO-8601 UTC timestamp (matches DSA-110 fada UTC_START format)."""
    unix = (mjd - 40587.0) * 86400.0
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime(
        "%Y-%m-%d-%H:%M:%S",
    )


def _read_cal_yaml(yaml_path: Path) -> dict[str, object]:
    """Load the cal_solutions yaml (gives us antenna_order, pol_order, ...)."""
    with yaml_path.open() as f:
        cnf = yaml.safe_load(f)
    if "cal_solutions" not in cnf:
        raise SystemExit(f"{yaml_path}: missing top-level 'cal_solutions' key")
    return cnf["cal_solutions"]


def _build_manifest(
    *,
    work_dir: Path,
    voltage_root: Path,
    cals_dir: Path,
    sb: str,
    src: dict[str, object],
) -> tuple[Path, Path]:
    """Set up <work_dir>/0319_sb<NN>/{manifest.yaml, fl_0319bbb_chgroup0.out}.

    Returns (fixture_dir, manifest_path).
    """
    fixture_dir = work_dir / f"0319_sb{sb}"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    voltage_src = voltage_root / f"0319bbb_sb{sb}_data.out"
    if not voltage_src.is_file():
        raise SystemExit(f"voltage file not found: {voltage_src}")

    voltage_link = fixture_dir / "fl_0319bbb_chgroup0.out"
    if voltage_link.exists() or voltage_link.is_symlink():
        voltage_link.unlink()
    voltage_link.symlink_to(voltage_src)

    cal_src = cals_dir / f"beamformer_weights_sb{sb}_0319+415.dat"
    if not cal_src.is_file():
        raise SystemExit(f"cal blob not found: {cal_src}")

    cal_link = fixture_dir / "antennas.out"
    if cal_link.exists() or cal_link.is_symlink():
        cal_link.unlink()
    cal_link.symlink_to(cal_src)

    flagants_path = fixture_dir / "flagants.dat"
    flagants_path.write_text("")  # empty per user instruction (no flagging)

    manifest = {
        "fixture_kind": "continuum",
        "utc_start_iso": _mjd_to_utc_iso(float(src["mjd"])),
        "dec_deg": float(src["dec_deg"]),
        "utc_start_specnum": int(src["specnum"]),
        "n_blocks": 15,
        "continuum_sources": [
            {"name": "0319+415"},
        ],
        "cal_paths": {
            "antennas_out": "antennas.out",
            "flagants_dat": "flagants.dat",
        },
        # M2-extra metadata (additionalProperties: true)
        "src_ra_deg": float(src["ra_deg"]),
        "src_mjd": float(src["mjd"]),
        "voltage_link": str(voltage_link),
        "cal_link": str(cal_link),
        "sb": sb,
    }
    manifest_path = fixture_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return fixture_dir, manifest_path


def _build_meridian_param(
    *,
    fixture_dir: Path,
    sb: str,
    cal_yaml: dict[str, object],
    src: dict[str, object],
    hostname: str,
    nint: int,
    nfreq_int: int,
) -> Path:
    """Write a per-sb dsamfs param yaml. ch0 dict has a single key matching
    the running hostname so ``socket.gethostname()`` lookup succeeds.
    """
    sb_int = int(sb)
    ch0 = CH0_BASE + sb_int * NCHAN_PER_CHGROUP

    antenna_order = [int(a) for a in cal_yaml["antenna_order"]]
    if len(antenna_order) != 96:
        raise SystemExit(
            f"cal yaml antenna_order has {len(antenna_order)} entries, expected 96"
        )

    # dsamfs/utils.py::parse_params expects antenna_order: dict {idx: ant_id}.
    ant_od = {i: a for i, a in enumerate(antenna_order)}

    # `filelength_minutes` MUST be small enough that
    # `max_frames_per_file = ceil(filelength_minutes*60/(tsamp*nint)) = 1`.
    # dsamfs/io.py:343 has a latent shape-bug on the second loop iteration
    # (it tries to count_nonzero on the previous iteration's freq-averaged
    # `data` and then reshape with the un-averaged nfreq_int factor). With
    # max_frames=1, the inner while loop exits cleanly after writing the
    # single integration we have data for, never triggering the bug.
    tsamp_s = 0.134217728
    one_frame_window_s = tsamp_s * nint                          # = 2.013 s for nint=15
    filelength_minutes = 0.5 * one_frame_window_s / 60.0          # half-window safety

    param = {
        "test": False,
        "key_string": "bada",
        "nant": 96,
        "bw_GHz": 0.250,
        "nchan": 8192,
        "f0_GHz": 1.53,
        "chan_ascending": False,
        "npol": 2,
        "samples_per_frame": 1,
        "samples_per_frame_out": 1,
        "nint": nint,
        "fringestop": True,
        # pt_dec is overridden by the wrapper's monkey-patch on
        # get_pointing_declination; this value is just a placeholder.
        "pt_dec": 0.7245,  # ≈ 41.51 deg, ignored by wrapper but kept for sanity
        "tsamp": tsamp_s,
        "nfreq_int": nfreq_int,
        "antenna_order": ant_od,
        "nchan_spw": NCHAN_PER_CHGROUP,
        "ch0": {hostname: ch0},
        "filelength_minutes": filelength_minutes,
        "outrigger_delays": {},
        "refmjd": float(src["mjd"]),
    }
    param_path = fixture_dir / "meridian_param.yaml"
    param_path.write_text(yaml.safe_dump(param, sort_keys=False))
    return param_path


def _run_one_sb(
    *,
    sb: str,
    fixture_dir: Path,
    cal_dat: Path,
    meridian_param: Path,
    pt_dec_deg: float,
    uvh5_out: Path,
    n_blocks: int,
    dsart_python: str,
    casa38_python: str,
    fada_key: str,
    bada_key: str,
    cal_mode: str,
    cal_pol_swap: bool,
    timeout_s: float,
) -> int:
    """Dispatch one sb through voltage_fixture_slow_corr.py."""
    work_dir = fixture_dir / "_runlogs"
    work_dir.mkdir(exist_ok=True)

    cmd = [
        dsart_python, "-m", "bench.voltage_fixture_slow_corr",
        "--run-id", fixture_dir.name,
        "--chgroups", "0",
        "--rate", "fast",  # not native; we want this to finish quickly
        "--n-blocks", str(n_blocks),
        "--fada-key", fada_key,
        "--bada-key", bada_key,
        "--apply-cal", str(cal_dat),
        "--cal-mode", cal_mode,
        "--meridian-param", str(meridian_param),
        "--meridian-pt-dec-deg", str(pt_dec_deg),
        "--uvh5-out", str(uvh5_out),
        "--casa38-python", casa38_python,
        "--dsart-python", dsart_python,
        "--work-dir", str(work_dir),
        "--timeout-s", str(timeout_s),
        "--device", "auto",
        "--corr-log-level", "INFO",
    ]
    if cal_pol_swap:
        cmd.append("--cal-pol-swap")

    print(f"\n[run_0319] === sb{sb} ({uvh5_out.name}) ===", flush=True)
    print("[run_0319] cmd:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["DSART_VOLTAGE_FIXTURE_ROOT"] = str(fixture_dir.parent)
    proc = subprocess.run(cmd, env=env, timeout=timeout_s + 60)
    return proc.returncode


# --- main ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voltage-root", type=Path, default=DEFAULT_VOLTAGE_ROOT,
                   help="root dir containing voltages/ + cals/ + json")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_ROOT,
                   help="output dir for UVH5 + per-sb fixtures + logs")
    p.add_argument("--sbs", default=",".join(ALL_SBS_DEFAULT),
                   help="comma-separated sb ids to process (default: 00..15 minus 12)")
    p.add_argument("--n-blocks", type=int, default=15,
                   help="how many fada blocks per sb (default: 15 = full file)")
    p.add_argument("--nint", type=int, default=15,
                   help="dsamfs nint (input samples per output integration; "
                        "default 15 → 1 integration per fada-block batch)")
    p.add_argument("--nfreq-int", type=int, default=8,
                   help="dsamfs frequency averaging factor (default 8)")
    p.add_argument("--cal-mode", default="phase", choices=("full", "phase"),
                   help="phase = divide by |G| first (default; fp16 path); "
                        "full = preserve gain magnitude (fp32 path)")
    p.add_argument("--cal-pol-swap", action="store_true")
    p.add_argument("--src-ra-deg", type=float, default=SRC_RA_DEG_DEFAULT)
    p.add_argument("--src-dec-deg", type=float, default=SRC_DEC_DEG_DEFAULT)
    p.add_argument("--src-mjd", type=float, default=SRC_MJD_DEFAULT)
    p.add_argument("--src-specnum", type=int, default=SRC_SPECNUM_DEFAULT)
    p.add_argument("--src-json", type=Path, default=None,
                   help="optional T2_<src>.json to override --src-* values")
    p.add_argument("--cal-yaml", type=Path, default=None,
                   help="path to cals/beamformer_weights_<src>.yaml; "
                        "default = <voltage-root>/cals/beamformer_weights_0319+415.yaml")
    p.add_argument("--hostname", default=None,
                   help="override socket.gethostname() (default: detect)")
    p.add_argument("--dsart-python",
                   default="/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python")
    p.add_argument("--casa38-python",
                   default="/home/ubuntu/anaconda3/envs/casa38/bin/python")
    p.add_argument("--fada-key", default="fada")
    p.add_argument("--bada-key", default="bada")
    p.add_argument("--timeout-s", type=float, default=900.0,
                   help="per-sb timeout (default 15 min — 15 fada blocks at fast rate)")
    p.add_argument("--dry-run", action="store_true",
                   help="set up fixture dirs + param files but don't dispatch")
    args = p.parse_args(argv)

    voltage_root = args.voltage_root
    voltage_dir = voltage_root / "voltages"
    cals_dir = voltage_root / "cals"
    if not voltage_dir.is_dir():
        raise SystemExit(f"voltage dir not found: {voltage_dir}")
    if not cals_dir.is_dir():
        raise SystemExit(f"cals dir not found: {cals_dir}")

    cal_yaml_path = args.cal_yaml or (cals_dir / "beamformer_weights_0319+415.yaml")
    if not cal_yaml_path.is_file():
        raise SystemExit(f"cal yaml not found: {cal_yaml_path}")
    cal_yaml = _read_cal_yaml(cal_yaml_path)

    src = {
        "ra_deg": args.src_ra_deg,
        "dec_deg": args.src_dec_deg,
        "mjd": args.src_mjd,
        "specnum": args.src_specnum,
    }
    if args.src_json is not None:
        with args.src_json.open() as f:
            src_doc = json.load(f)
        # Take the first key's values.
        first_src = next(iter(src_doc.values()))
        src["ra_deg"] = float(first_src.get("ra", src["ra_deg"]))
        src["dec_deg"] = float(first_src.get("dec", src["dec_deg"]))
        src["mjd"] = float(first_src.get("mjds", src["mjd"]))
        src["specnum"] = int(first_src.get("specnum", src["specnum"]))

    hostname = args.hostname or socket.gethostname()
    print(f"[run_0319] hostname={hostname}")
    print(f"[run_0319] source: ra={src['ra_deg']} dec={src['dec_deg']} "
          f"mjd={src['mjd']} specnum={src['specnum']}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    sbs = [s.strip() for s in args.sbs.split(",") if s.strip()]
    print(f"[run_0319] processing sbs: {sbs}")

    summary: list[dict[str, object]] = []
    overall_rc = 0
    for sb in sbs:
        cal_dat = cals_dir / f"beamformer_weights_sb{sb}_0319+415.dat"
        if not cal_dat.is_file():
            print(f"[run_0319] WARN: cal for sb{sb} not found ({cal_dat}); skipping")
            summary.append({"sb": sb, "rc": -1, "reason": "no_cal_blob"})
            overall_rc = max(overall_rc, 7)
            continue

        fixture_dir, _ = _build_manifest(
            work_dir=args.work_dir, voltage_root=voltage_dir, cals_dir=cals_dir,
            sb=sb, src=src,
        )
        param_path = _build_meridian_param(
            fixture_dir=fixture_dir, sb=sb, cal_yaml=cal_yaml, src=src,
            hostname=hostname, nint=args.nint, nfreq_int=args.nfreq_int,
        )
        uvh5_out = args.work_dir / f"0319_sb{sb}.uvh5"

        if args.dry_run:
            print(f"[run_0319] DRY-RUN sb{sb}: fixture={fixture_dir} "
                  f"param={param_path} uvh5={uvh5_out}")
            summary.append({"sb": sb, "rc": 0, "reason": "dry_run",
                            "fixture": str(fixture_dir),
                            "uvh5": str(uvh5_out)})
            continue

        t0 = time.monotonic()
        rc = _run_one_sb(
            sb=sb, fixture_dir=fixture_dir, cal_dat=cal_dat,
            meridian_param=param_path, pt_dec_deg=src["dec_deg"],
            uvh5_out=uvh5_out, n_blocks=args.n_blocks,
            dsart_python=args.dsart_python, casa38_python=args.casa38_python,
            fada_key=args.fada_key, bada_key=args.bada_key,
            cal_mode=args.cal_mode, cal_pol_swap=args.cal_pol_swap,
            timeout_s=args.timeout_s,
        )
        elapsed = time.monotonic() - t0
        summary.append({
            "sb": sb, "rc": rc, "elapsed_s": elapsed,
            "fixture": str(fixture_dir),
            "uvh5": str(uvh5_out) if rc == 0 else None,
        })
        if rc != 0:
            print(f"[run_0319] sb{sb} FAILED rc={rc}", file=sys.stderr)
            overall_rc = max(overall_rc, 8)

    summary_path = args.work_dir / "run_0319_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[run_0319] summary written to {summary_path}")
    print(json.dumps(summary, indent=2))
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
