#!/usr/bin/env python3
"""Build per-DEC fringestopping-table cache (M2, plan §8 line 2164).

Walks a DEC grid (default -30..+85, step 0.25°) and calls the LEGACY
`dsamfs.fringestopping.generate_fringestopping_table` once per DEC,
writing one `.npz` cache file per DEC under `--output-dir`.

Run-environment: this script imports from `dsamfs` (dsa110-meridian-fs)
and indirectly `dsacalib`, neither of which is installed in the
`dsa110-rt` conda env. Run from `casa38`:

    /home/ubuntu/anaconda3/envs/casa38/bin/python \\
        tools/build_fstable_cache.py [args]

Per user constraint (M2_PLAN_FIXES.md D14, 2026-05-04): this tool is a
*caller* of legacy `dsamfs` code; it does NOT modify any casa38 install
or repo. The cache files it produces are passive on-disk artifacts; how
`meridian_fringestop` adopts them (e.g. via an operator-side dsamfs
patch) is out-of-scope for the M2 revamp.

Filename scheme (D6 in M2_PLAN_FIXES.md):

    fringestopping_table_dec_{deg:+08.4f}deg_{nant}ant_refmjd{refmjd:.6f}.npz

Uses 4 decimal degrees of precision so the 0.25° grid does not collide
(legacy `.1f` rounded 0.25° apart values to the same filename); embeds
`refmjd` so cache invalidation on `refmjd` drift is a `glob` filter.

The full DEC grid (461 points) takes ~30 s/file ≈ 4 hours; this is a
one-time operator setup. The DoD-time integration test (M2.sh) uses a
narrow grid (`--dec-min 25.0 --dec-max 25.25 --dec-step 0.25` → 2 files)
to exercise the full path quickly.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger("build_fstable_cache")

CASA38_PY = "/home/ubuntu/anaconda3/envs/casa38/bin/python"

DEFAULT_DEC_MIN_DEG = -30.0
DEFAULT_DEC_MAX_DEG = 85.0
DEFAULT_DEC_STEP_DEG = 0.25
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/data/fstables")

DEFAULT_CORR_SETUP_YAML = Path("/home/ubuntu/proj/dsa110-shell/dsa110-cnf/corr_setup.yaml")
DEFAULT_CONFIG_MFS_YAML = Path("/home/ubuntu/proj/dsa110-shell/dsa110-cnf/config_mfs.yaml")


def _check_env() -> None:
    """Warn (not fail) if running outside casa38; the import will fail anyway."""
    if "casa38" not in sys.executable and "casa38" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        LOG.warning(
            "running outside casa38 env (sys.executable=%s); imports below "
            "will likely fail. Re-invoke as: %s tools/build_fstable_cache.py ...",
            sys.executable,
            CASA38_PY,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _collect_params(corr_setup_yaml: Path, config_mfs_yaml: Path) -> dict[str, Any]:
    """Read corr/mfs yamls directly, bypassing dsamfs.utils.parse_params.

    parse_params reads pt_dec from etcd via dsa_store.DsaStore(); we want a
    deterministic, etcd-free build host. The fields we need are static across
    DEC, so reading the yamls directly is sufficient.
    """
    corr = _load_yaml(corr_setup_yaml)
    mfs = _load_yaml(config_mfs_yaml)

    nant = int(corr["nant"])
    ant_od = corr["antenna_order"]
    antenna_order = [int(v) for v in list(ant_od.values())]
    if len(antenna_order) != nant:
        raise ValueError(
            f"corr_setup nant={nant} disagrees with len(antenna_order)={len(antenna_order)}"
        )

    return {
        "nant": nant,
        "antenna_order": antenna_order,
        "nint": int(mfs["nint"]),
        "tsamp": float(corr["tsamp"]),
        "outrigger_delays": dict(mfs["outrigger_delays"]),
        "refmjd": float(mfs["refmjd"]),
        "_corr_yaml": str(corr_setup_yaml),
        "_mfs_yaml": str(config_mfs_yaml),
    }


def _collect_params_etcd() -> dict[str, Any]:
    """Read the same fields from live etcd (``/cnf/corr`` + ``/cnf/fringe``).

    This is the authoritative source: ``meridian_fringestop``'s
    ``load_visibility_model`` asserts the cached table's ``antenna_order``,
    ``nint``, ``tsamp``, ``outrigger_delays`` and ``refmjd`` all match what
    ``parse_params`` derives from etcd -- so building the cache from etcd
    guarantees a cache *hit* at runtime. The static yamls can drift from etcd;
    prefer ``--from-etcd`` for the production bank.
    """
    from dsautils.dsa_store import DsaStore

    store = DsaStore()
    corr = store.get_dict("/cnf/corr")
    fringe = store.get_dict("/cnf/fringe")
    if not isinstance(corr, dict) or not isinstance(fringe, dict):
        raise RuntimeError("/cnf/corr or /cnf/fringe missing/empty in etcd")

    ant_od = corr["antenna_order"]
    antenna_order = [int(v) for v in list(ant_od.values())]
    nant = int(corr.get("nant", len(antenna_order)))
    if len(antenna_order) != nant:
        raise ValueError(
            f"/cnf/corr nant={nant} disagrees with len(antenna_order)={len(antenna_order)}"
        )

    return {
        "nant": nant,
        "antenna_order": antenna_order,
        "nint": int(fringe["nint"]),
        "tsamp": float(corr["tsamp"]),
        "outrigger_delays": {str(k): v for k, v in dict(fringe["outrigger_delays"]).items()},
        "refmjd": float(fringe["refmjd"]),
        "_corr_yaml": "etcd:/cnf/corr",
        "_mfs_yaml": "etcd:/cnf/fringe",
    }


def _baseline_geometry(antenna_order: list[int], refmjd: float):
    """Compute baseline names + ITRF lengths once. Independent of pt_dec."""
    from dsamfs.utils import baseline_uvw

    bname, blen, _uvw = baseline_uvw(
        antenna_order=antenna_order,
        pt_dec=0.0,
        refmjd=refmjd,
        casa_order=False,
    )
    return bname, blen


def _dec_grid(dec_min_deg: float, dec_max_deg: float, dec_step_deg: float) -> list[float]:
    """Inclusive [dec_min, dec_max] grid at step dec_step. Returns degrees."""
    if dec_step_deg <= 0.0:
        raise ValueError(f"dec_step_deg must be > 0, got {dec_step_deg!r}")
    n = int(math.floor((dec_max_deg - dec_min_deg) / dec_step_deg + 1e-9)) + 1
    return [dec_min_deg + i * dec_step_deg for i in range(n)]


def _fstable_filename(dec_deg: float, nant: int, refmjd: float) -> str:
    """File name pinned per D6 in M2_PLAN_FIXES.md."""
    return f"fringestopping_table_dec_{dec_deg:+08.4f}deg_{nant}ant_refmjd{refmjd:.6f}.npz"


def _build_one_table(
    dec_deg: float,
    blen,
    bname,
    params: dict[str, Any],
    output_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> tuple[str, str]:
    """Build one DEC. Returns (status, path). Status in {"built","skipped","failed"}."""
    import numpy as np

    nant = params["nant"]
    refmjd = params["refmjd"]
    outname = output_dir / _fstable_filename(dec_deg, nant, refmjd)

    if outname.exists() and not force:
        return ("skipped", str(outname))

    if dry_run:
        return ("dry_run", str(outname))

    try:
        from dsamfs.fringestopping import generate_fringestopping_table

        dec_rad = float(np.deg2rad(dec_deg))
        generate_fringestopping_table(
            blen=blen,
            pt_dec=dec_rad,
            nint=params["nint"],
            tsamp=params["tsamp"],
            antenna_order=params["antenna_order"],
            outrigger_delays=params["outrigger_delays"],
            bname=bname,
            mjd0=refmjd,
            outname=str(outname),
        )
    except Exception:
        LOG.exception("dec=%+.4f° table build failed", dec_deg)
        return ("failed", str(outname))

    return ("built", str(outname))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dec-min", type=float, default=DEFAULT_DEC_MIN_DEG, metavar="DEG")
    p.add_argument("--dec-max", type=float, default=DEFAULT_DEC_MAX_DEG, metavar="DEG")
    p.add_argument("--dec-step", type=float, default=DEFAULT_DEC_STEP_DEG, metavar="DEG")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--corr-setup-yaml", type=Path, default=DEFAULT_CORR_SETUP_YAML,
                   help="legacy dsa110-cnf/corr_setup.yaml (provides antenna_order, tsamp, nant)")
    p.add_argument("--config-mfs-yaml", type=Path, default=DEFAULT_CONFIG_MFS_YAML,
                   help="legacy dsa110-cnf/config_mfs.yaml (provides nint, outrigger_delays, refmjd)")
    p.add_argument("--from-etcd", action="store_true",
                   help="source params from live etcd (/cnf/corr + /cnf/fringe) instead of the "
                        "static yamls. RECOMMENDED for the production bank: it guarantees the "
                        "cached table matches meridian_fringestop's runtime asserts (cache hit).")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing tables; default is to skip files that exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print the DEC grid and filenames but do not call the table builder")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _check_env()

    if args.from_etcd:
        params = _collect_params_etcd()
    else:
        if not args.corr_setup_yaml.exists():
            LOG.error("corr_setup yaml not found: %s", args.corr_setup_yaml)
            return 2
        if not args.config_mfs_yaml.exists():
            LOG.error("config_mfs yaml not found: %s", args.config_mfs_yaml)
            return 2
        params = _collect_params(args.corr_setup_yaml, args.config_mfs_yaml)
    LOG.info("param source: corr=%s fringe=%s", params["_corr_yaml"], params["_mfs_yaml"])
    LOG.info("params: nant=%d nint=%d tsamp=%.6fs refmjd=%.6f outriggers=%d",
             params["nant"], params["nint"], params["tsamp"],
             params["refmjd"], len(params["outrigger_delays"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("output_dir=%s force=%s dry_run=%s", args.output_dir, args.force, args.dry_run)

    grid = _dec_grid(args.dec_min, args.dec_max, args.dec_step)
    LOG.info("DEC grid: %d points, %.4f° to %.4f° step %.4f°",
             len(grid), grid[0], grid[-1], args.dec_step)

    if args.dry_run:
        for dec_deg in grid[:5]:
            print(f"  would build: {args.output_dir / _fstable_filename(dec_deg, params['nant'], params['refmjd'])}")
        if len(grid) > 5:
            print(f"  ... ({len(grid) - 5} more)")
        return 0

    bname, blen = _baseline_geometry(params["antenna_order"], params["refmjd"])
    LOG.info("baseline geometry: nbl=%d (auto-corrs included)", len(bname))

    counts = {"built": 0, "skipped": 0, "failed": 0, "dry_run": 0}
    t_start = time.monotonic()
    for i, dec_deg in enumerate(grid):
        status, path = _build_one_table(
            dec_deg, blen, bname, params, args.output_dir,
            force=args.force, dry_run=args.dry_run,
        )
        counts[status] += 1
        if status == "built":
            elapsed = time.monotonic() - t_start
            avg = elapsed / max(1, counts["built"])
            remaining = (len(grid) - i - 1) * avg
            LOG.info("[%4d/%d] dec=%+.4f° BUILT %s (avg=%.1fs/file, eta=%.0fs)",
                     i + 1, len(grid), dec_deg, Path(path).name, avg, remaining)
        elif status == "failed":
            LOG.error("[%4d/%d] dec=%+.4f° FAILED %s", i + 1, len(grid), dec_deg, path)
        elif status == "skipped":
            LOG.debug("[%4d/%d] dec=%+.4f° skipped (exists)", i + 1, len(grid), dec_deg)

    elapsed = time.monotonic() - t_start
    LOG.info("done in %.1fs: built=%d skipped=%d failed=%d (total grid=%d)",
             elapsed, counts["built"], counts["skipped"], counts["failed"], len(grid))
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
