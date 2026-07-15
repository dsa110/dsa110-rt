"""Archive correlator (slow-vis) HDF5 files for a candidate.

The corr fleet's slow-vis path writes ~5-min UVH5 files to
``/dataz/dsa110/operations/correlator/`` as
``YYYY-MM-DDTHH:MM:SS_sb<NN>.hdf5`` (16 subbands per timestamp). A cron
job (``dsa110-T3/scripts/delete_level1_data.py``, daily 04:00 UTC)
deletes them after 3.5 days — so any candidate that needs offline
calibration / localization must capture its window before then.

This module mirrors the legacy T3 behaviour
(``dsaT3/data_manager.py::link_hdf5_files``): **hard-link** every
correlator file whose filename timestamp falls within ±``hours`` of the
candidate's ``t_peak_mjd`` into ``<cand>/calibration/``. Hard links are
instant, cost no space (same /dataz filesystem), and keep the data
alive when the cron unlinks the originals. A ``hdf5_manifest.json``
records the window, file list, and completeness.

Completeness follows T3: expect ``2*hours * 12`` timestamps × 16
subbands (5-min cadence); below 15/16 of that is reported (and exits
nonzero) but whatever was found is still linked — partial data beats
none.

Standalone CLI for now (run manually per event); designed to be called
from the C3 KEEP path later::

    python -m dsart.coinc.cal_hdf5_archive --event 260715twmx
    python -m dsart.coinc.cal_hdf5_archive --event 260715twmx \\
        --hours 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("dsart.coinc.cal_hdf5_archive")

DEFAULT_CANDIDATES_ROOT = Path("/dataz/dsa110/candidates")
DEFAULT_CORRELATOR_DIR = Path("/dataz/dsa110/operations/correlator")
DEFAULT_DEST_SUBDIR = "calibration"
DEFAULT_HOURS_EACH_SIDE = 2.0

N_SUBBANDS = 16
FILE_CADENCE_MIN = 5.0
#: T3 completeness threshold (data_manager.link_hdf5_files): report
#: failure below 15/16 of the expected file count.
COMPLETENESS_FRACTION = 15.0 / 16.0

_MJD_UNIX_EPOCH = 40587.0

#: ``2026-07-15T03:57:52_sb00.hdf5``
_HDF5_NAME_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_sb(?P<sb>\d{2})\.hdf5$")


def _mjd_to_unix(mjd: float) -> float:
    return (mjd - _MJD_UNIX_EPOCH) * 86400.0


def _parse_hdf5_time(name: str) -> Optional[Tuple[float, int]]:
    """(unix_time, subband) from a correlator filename, else None."""
    m = _HDF5_NAME_RE.match(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group("ts"), "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), int(m.group("sb"))
    except ValueError:
        return None


def read_t_peak_mjd(ev_dir: Path, name: str) -> float:
    """Candidate peak time from ``Level3/<name>.json`` (c2.t_peak_mjd)."""
    l3 = ev_dir / "Level3" / f"{name}.json"
    with l3.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    mjd = ((doc or {}).get("c2") or {}).get("t_peak_mjd")
    if not mjd:
        raise ValueError(f"no c2.t_peak_mjd in {l3}")
    return float(mjd)


def select_files(correlator_dir: Path, t_center_unix: float,
                 hours_each_side: float) -> List[Path]:
    """Correlator files whose filename time is within the window.

    Window edges follow T3's semantics on the START stamp of each
    ~5-min file: a file *starting* up to one cadence before the window
    opens still overlaps it, so the lower edge is padded by one file
    length.
    """
    lo = t_center_unix - hours_each_side * 3600.0 - FILE_CADENCE_MIN * 60.0
    hi = t_center_unix + hours_each_side * 3600.0
    out: List[Path] = []
    for p in correlator_dir.iterdir():
        parsed = _parse_hdf5_time(p.name)
        if parsed is None:
            continue
        t, _sb = parsed
        if lo <= t <= hi:
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def archive_event(
    name: str,
    *,
    candidates_root: Path = DEFAULT_CANDIDATES_ROOT,
    correlator_dir: Path = DEFAULT_CORRELATOR_DIR,
    dest_subdir: str = DEFAULT_DEST_SUBDIR,
    hours_each_side: float = DEFAULT_HOURS_EACH_SIDE,
    t_peak_mjd: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Hard-link the candidate's correlator-HDF5 window into
    ``<cand>/<dest_subdir>/``. Returns a JSON-ready report (also written
    to ``<dest>/hdf5_manifest.json`` unless dry_run)."""
    ev_dir = candidates_root / name
    if not ev_dir.is_dir():
        raise FileNotFoundError(f"no candidate dir {ev_dir}")
    if t_peak_mjd is None:
        t_peak_mjd = read_t_peak_mjd(ev_dir, name)
    t_center = _mjd_to_unix(t_peak_mjd)

    files = select_files(correlator_dir, t_center, hours_each_side)
    expected = int(round(
        2.0 * hours_each_side * 60.0 / FILE_CADENCE_MIN)) * N_SUBBANDS
    complete = len(files) >= expected * COMPLETENESS_FRACTION

    dest = ev_dir / dest_subdir
    n_linked = n_existing = n_copied = 0
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dest / src.name
        if dry_run:
            continue
        if dst.exists():
            n_existing += 1
            continue
        try:
            os.link(src, dst)                      # hard link (T3 style)
            n_linked += 1
        except OSError as exc:
            LOG.warning("hard link failed for %s (%s); copying", src, exc)
            shutil.copy2(src, dst)                 # cross-device fallback
            n_copied += 1

    report: Dict[str, Any] = {
        "event_name": name,
        "t_peak_mjd": t_peak_mjd,
        "t_center_utc": datetime.fromtimestamp(
            t_center, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hours_each_side": hours_each_side,
        "correlator_dir": str(correlator_dir),
        "dest": str(dest),
        "n_files": len(files),
        "n_expected": expected,
        "complete": complete,
        "n_linked": n_linked,
        "n_already_present": n_existing,
        "n_copied": n_copied,
        "dry_run": dry_run,
        "archived_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "files": [p.name for p in files],
    }
    if not dry_run:
        try:
            with (dest / "hdf5_manifest.json").open("w") as fh:
                json.dump(report, fh, indent=1)
        except OSError as exc:
            LOG.warning("manifest write failed: %s", exc)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--event", required=True, help="candidate name")
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS_EACH_SIDE,
                    help="window half-width in hours [2.0]")
    ap.add_argument("--candidates-root", type=Path,
                    default=DEFAULT_CANDIDATES_ROOT)
    ap.add_argument("--correlator-dir", type=Path,
                    default=DEFAULT_CORRELATOR_DIR)
    ap.add_argument("--dest-subdir", default=DEFAULT_DEST_SUBDIR)
    ap.add_argument("--mjd", type=float, default=None,
                    help="override t_peak_mjd (else from Level3 json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    rep = archive_event(
        args.event,
        candidates_root=args.candidates_root,
        correlator_dir=args.correlator_dir,
        dest_subdir=args.dest_subdir,
        hours_each_side=args.hours,
        t_peak_mjd=args.mjd,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: v for k, v in rep.items() if k != "files"},
                     indent=1))
    if not rep["complete"]:
        print(f"WARNING: incomplete window: {rep['n_files']}/"
              f"{rep['n_expected']} files (threshold "
              f"{COMPLETENESS_FRACTION:.3f})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
