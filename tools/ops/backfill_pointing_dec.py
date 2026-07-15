#!/usr/bin/env python3
"""Backfill ``c2.pointing_dec_deg`` into historical Level3 event JSONs.

The coincidencer stamps the array pointing declination into every NEW
event's ``Level3/<name>.json`` (``c2.pointing_dec_deg`` +
``c2.pointing_dec_meta``) at archive time. Events archived before that
fix carry no pointing-dec record, so their absolute RA/Dec cannot be
recovered from the (pointing-relative) ``l_median``/``m_median`` image
offsets. When the operator can attest that the pointing was CONSTANT
over a known MJD window, this script stamps that value in with clearly
labeled manual provenance::

    "pointing_dec_meta": {
        "etcd_key": null,
        "read_unix": <time.time() at run>,
        "source": "manual_backfill",
        "note": "<operator-supplied note>"
    }

The dashboard renders these with provenance "manual" (see
``tools/dashboard/dsa_monitor/event_astrometry.py``).

Safety:

  * DRY RUN by default — pass ``--apply`` to write anything.
  * Consistency guard (both modes): every in-window event that carries a
    ``filterbank/filterbank.json`` with a finite ``dec_deg`` (stamped
    live by C3/bbproc at processing time) must agree with ``--dec-deg``
    to < 1e-6 deg, else the script prints the offenders and aborts
    nonzero BEFORE any write.
  * Every modified JSON is first copied to ``--backup-dir/<name>.json``;
    an already-existing backup is never overwritten (the event is
    skipped with a warning instead).
  * Writes are atomic (tmp file + os.replace) and preserve the
    ``json.dumps(indent=2, sort_keys=True)`` formatting used by
    ``dsart.coinc.archive.EventArchiveWriter.write_l3_metadata``.
  * Only events whose ``c2.pointing_dec_deg`` is absent or null are
    touched; a second ``--apply`` run is a no-op.

Usage::

    tools/ops/backfill_pointing_dec.py \
        --root /dataz/dsa110/candidates \
        --dec-deg 16.273406015527343 \
        --mjd-min 61230.0 --mjd-max 61236.75 \
        --backup-dir /home/ubuntu/vishnu/backfills/2026-07-15-pointing-dec \
        --note "constant pointing 2026-07-09..15, operator-confirmed" \
        [--apply]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: |filterbank dec - --dec-deg| tolerance (deg) for the consistency guard.
FILTERBANK_TOL_DEG: float = 1e-6


def _as_finite_float(v: Any) -> Optional[float]:
    """float(v) if it is a finite number, else None (bool excluded)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _atomic_write(path: Path, body: str) -> None:
    """tmpfile + os.replace, mirroring dsart.coinc.archive._atomic_write."""
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _filterbank_dec(event_dir: Path) -> Optional[float]:
    """Finite dec_deg from filterbank/filterbank.json, else None."""
    doc = _load_json(event_dir / "filterbank" / "filterbank.json")
    if not isinstance(doc, dict):
        return None
    return _as_finite_float(doc.get("dec_deg"))


def _scan(
    root: Path, mjd_min: float, mjd_max: float,
) -> Tuple[List[Tuple[str, Path, float, Dict[str, Any]]],
           List[Tuple[str, Path, float]],
           Dict[str, int]]:
    """Walk the archive once.

    Returns ``(eligible, in_window, skipped)`` where ``eligible`` is
    ``[(name, l3_path, t_peak_mjd, parsed_doc)]`` (pointing dec absent
    or null), ``in_window`` is every event inside the MJD window
    regardless of stamp state (consistency-guard population), and
    ``skipped`` counts the silently-skipped by reason.
    """
    eligible: List[Tuple[str, Path, float, Dict[str, Any]]] = []
    in_window: List[Tuple[str, Path, float]] = []
    skipped: Dict[str, int] = {
        "no_level3_json": 0,
        "unparseable_json": 0,
        "legacy_or_no_c2": 0,
        "no_finite_t_peak_mjd": 0,
        "out_of_window": 0,
        "already_stamped": 0,
    }
    for ev_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = ev_dir.name
        l3 = ev_dir / "Level3" / f"{name}.json"
        if not l3.is_file():
            skipped["no_level3_json"] += 1
            continue
        doc = _load_json(l3)
        if not isinstance(doc, dict):
            skipped["unparseable_json"] += 1
            continue
        c2 = doc.get("c2")
        if not isinstance(c2, dict):
            skipped["legacy_or_no_c2"] += 1
            continue
        mjd = _as_finite_float(c2.get("t_peak_mjd"))
        if mjd is None:
            skipped["no_finite_t_peak_mjd"] += 1
            continue
        if not mjd_min <= mjd <= mjd_max:
            skipped["out_of_window"] += 1
            continue
        in_window.append((name, ev_dir, mjd))
        if c2.get("pointing_dec_deg") is not None:
            skipped["already_stamped"] += 1
            continue
        eligible.append((name, l3, mjd, doc))
    return eligible, in_window, skipped


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=Path,
                   default=Path("/dataz/dsa110/candidates"),
                   help="candidate archive root (default: %(default)s)")
    p.add_argument("--dec-deg", type=float, required=True,
                   help="pointing declination (deg) to stamp")
    p.add_argument("--mjd-min", type=float, required=True,
                   help="inclusive lower bound on c2.t_peak_mjd")
    p.add_argument("--mjd-max", type=float, required=True,
                   help="inclusive upper bound on c2.t_peak_mjd")
    p.add_argument("--backup-dir", type=Path, required=True,
                   help="directory receiving a copy of every original "
                        "Level3 json before it is modified")
    p.add_argument("--note", type=str, required=True,
                   help="operator note recorded in pointing_dec_meta")
    p.add_argument("--apply", action="store_true",
                   help="actually write; default is a dry run")
    args = p.parse_args(argv)

    if not (math.isfinite(args.dec_deg) and -90.0 <= args.dec_deg <= 90.0):
        print(f"[ERROR] --dec-deg {args.dec_deg!r} is not a declination "
              f"in [-90, 90]", file=sys.stderr)
        return 2
    if args.mjd_min > args.mjd_max:
        print("[ERROR] --mjd-min > --mjd-max", file=sys.stderr)
        return 2
    if not args.root.is_dir():
        print(f"[ERROR] --root {args.root} is not a directory",
              file=sys.stderr)
        return 2

    prefix = "" if args.apply else "DRY-RUN "
    eligible, in_window, skipped = _scan(
        args.root, args.mjd_min, args.mjd_max)

    # ---- consistency guard (both modes, before any write) --------------
    n_fb_checked = 0
    offenders: List[Tuple[str, float]] = []
    for name, ev_dir, _mjd in in_window:
        fb_dec = _filterbank_dec(ev_dir)
        if fb_dec is None:
            continue
        n_fb_checked += 1
        if abs(fb_dec - args.dec_deg) >= FILTERBANK_TOL_DEG:
            offenders.append((name, fb_dec))
    if offenders:
        print(f"[ERROR] filterbank dec mismatch vs --dec-deg="
              f"{args.dec_deg!r} (tol {FILTERBANK_TOL_DEG}); "
              f"aborting with NO writes:", file=sys.stderr)
        for name, fb_dec in offenders:
            print(f"  {name}: filterbank dec_deg={fb_dec!r}",
                  file=sys.stderr)
        return 1

    # ---- stamp (or narrate) ---------------------------------------------
    run_unix = time.time()
    n_stamped = 0
    n_backup_refused = 0
    for name, l3, mjd, doc in eligible:
        action = "stamped" if args.apply else "would-stamp"
        if args.apply:
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            backup = args.backup_dir / f"{name}.json"
            if backup.exists():
                print(f"{name}  t_peak_mjd={mjd:.6f}  "
                      f"SKIPPED (backup already exists: {backup})")
                n_backup_refused += 1
                continue
            shutil.copy2(l3, backup)
            doc["c2"]["pointing_dec_deg"] = args.dec_deg
            doc["c2"]["pointing_dec_meta"] = {
                "etcd_key": None,
                "read_unix": run_unix,
                "source": "manual_backfill",
                "note": args.note,
            }
            body = json.dumps(doc, indent=2, sort_keys=True, default=str)
            _atomic_write(l3, body + "\n")
        n_stamped += 1
        print(f"{prefix}{name}  t_peak_mjd={mjd:.6f}  {action}")

    # ---- summary ----------------------------------------------------------
    verb = "stamped" if args.apply else "would-stamp"
    print(f"{prefix}summary:")
    print(f"{prefix}  eligible: {len(eligible)}")
    print(f"{prefix}  {verb}: {n_stamped}")
    if n_backup_refused:
        print(f"{prefix}  skipped (backup exists): {n_backup_refused}")
    for reason, n in skipped.items():
        print(f"{prefix}  skipped ({reason}): {n}")
    print(f"{prefix}  filterbank-consistency-checked: {n_fb_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
