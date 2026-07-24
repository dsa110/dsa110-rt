#!/usr/bin/env python3
"""Archive old DSA-110 beamformer-weight files out of a hot weights dir.

`beamformer_weights/generated/` (and `applied/`) accumulate ~17 files
per calibration solution (1 yaml manifest + 16 sb .dat). `generated/`
had grown to ~99k files, degrading /dataz directory operations (a plain
listing took 30-60 s+), which hung the SEFD page and slows the
calibration service that writes+lists it.

This moves OLD files into monthly gzip tarballs under a sibling
`archive/` dir, keeping LIVE:
  (a) any file whose solution timestamp is newer than --keep-days;
  (b) the newest --keep-per-source solutions for every source
      (so `latest_descriptor` and re-applying the current weights never
       break, even for sources last calibrated long ago, e.g. the
       high-dec +71.6 calibrators last done 2025-09); and
  (c) any file whose name contains a --force-keep substring
      (e.g. the currently-applied set's timestamp).

Each monthly tarball is written to a temp file, gzip-verified, and its
member count checked to equal the number of files added, BEFORE any
original is removed. Idempotent (already-archived months skip; a re-run
only touches files still present). --dry-run (default) previews.

Restore a month:
    tar xzf archive/<subdir>_YYYY-MM.tar.gz -C <subdir>/

Intended monthly cron (keeps both dirs from re-bloating):
    archive_bfweights.py --source-dir generated --apply
    archive_bfweights.py --source-dir applied   --apply \\
        --force-keep "$(current applied ISOT)"
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import tarfile
import tempfile
import time
from collections import defaultdict

BF_ROOT = "/dataz/dsa110/operations/beamformer_weights"

_ISO = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
# source = token(s) between the sbNN_/prefix and the ISO timestamp, or ''
_SRC = re.compile(r"^beamformer_weights_(?:sb\d+_)?(.*?)_?\d{4}-\d{2}-\d{2}T")


def _parse(name):
    """(source, isot) for a bf weight file, or (None, None) if unrecognised."""
    if not name.startswith("beamformer_weights_"):
        return None, None
    m = _ISO.search(name)
    if not m:
        return None, None
    sm = _SRC.match(name)
    src = sm.group(1) if sm else ""
    if src.startswith("sb"):
        src = ""
    return src, m.group(1)


def plan(names, keep_days, keep_per_source, today, force_keep=()):
    """Return (keep_set, archive_by_month{YYYY-MM: [names]})."""
    cutoff = today - datetime.timedelta(days=keep_days)
    sols = defaultdict(set)
    parsed = []
    for n in names:
        src, isot = _parse(n)
        if isot is None:
            continue          # unrecognised → never touched (implicitly kept)
        parsed.append((n, src, isot))
        sols[src].add(isot)
    newest = set()
    for src, isots in sols.items():
        for isot in sorted(isots, reverse=True)[:keep_per_source]:
            newest.add((src, isot))
    keep, arch = set(), defaultdict(list)
    for n, src, isot in parsed:
        d = datetime.date.fromisoformat(isot[:10])
        if (d >= cutoff or (src, isot) in newest
                or any(fk in n for fk in force_keep)):
            keep.add(n)
        else:
            arch[isot[:7]].append(n)
    return keep, arch


def archive_month(month, names, src_dir, archive_dir, prefix, *, apply, log):
    """tar.gz the given files (verify) then remove originals."""
    os.makedirs(archive_dir, exist_ok=True)
    present = [n for n in names if os.path.exists(os.path.join(src_dir, n))]
    if not present:
        log(f"  {month}: already archived (0 present) — skip")
        return 0, 0
    final = os.path.join(archive_dir, f"{prefix}_{month}.tar.gz")
    if os.path.exists(final):
        final = os.path.join(
            archive_dir, f"{prefix}_{month}.{int(time.time())}.tar.gz")
    if not apply:
        log(f"  {month}: WOULD archive {len(present)} files -> "
            f"{os.path.basename(final)} then remove originals")
        return len(present), 0
    fd, tmp = tempfile.mkstemp(dir=archive_dir, suffix=".tar.gz.tmp")
    os.close(fd)
    t0 = time.time()
    with tarfile.open(tmp, "w:gz") as tar:
        for n in present:
            tar.add(os.path.join(src_dir, n), arcname=n)
    with tarfile.open(tmp, "r:gz") as tar:      # verify: gzip + member count
        members = tar.getnames()
    if len(members) != len(present):
        os.unlink(tmp)
        raise RuntimeError(
            f"{month}: tar members {len(members)} != {len(present)} — abort")
    os.rename(tmp, final)
    removed = 0
    for n in present:                            # only now remove originals
        try:
            os.unlink(os.path.join(src_dir, n))
            removed += 1
        except OSError as exc:
            log(f"  {month}: WARN could not remove {n}: {exc}")
    log(f"  {month}: archived {len(present)} -> {os.path.basename(final)} "
        f"({os.path.getsize(final)/1e6:.1f} MB), removed {removed} "
        f"[{time.time()-t0:.1f}s]")
    return len(present), removed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default="generated",
                    help="subdir of beamformer_weights/ to prune "
                         "(generated | applied). Default generated.")
    ap.add_argument("--keep-days", type=int, default=90)
    ap.add_argument("--keep-per-source", type=int, default=3)
    ap.add_argument("--force-keep", action="append", default=[],
                    help="always keep files whose name contains this "
                         "substring (repeatable; e.g. the live applied ISOT)")
    ap.add_argument("--apply", action="store_true",
                    help="create tarballs + remove originals "
                         "(default: dry-run)")
    ap.add_argument("--only-month", default=None,
                    help="restrict to one YYYY-MM (pilot)")
    ap.add_argument("--listing", default=None,
                    help="pre-captured `ls -U` (avoids a slow live listing)")
    args = ap.parse_args()

    src_dir = os.path.join(BF_ROOT, args.source_dir)
    archive_dir = os.path.join(BF_ROOT, "archive")
    prefix = args.source_dir

    def log(m):
        print(m, flush=True)

    if args.listing:
        names = [l.strip() for l in open(args.listing) if l.strip()]
    else:
        log(f"scanning {src_dir} (may be slow if the dir is still large) ...")
        names = [e.name for e in os.scandir(src_dir)]
    log(f"{len(names)} entries in {args.source_dir}/")

    keep, arch = plan(names, args.keep_days, args.keep_per_source,
                      datetime.date.today(), force_keep=tuple(args.force_keep))
    n_arch = sum(len(v) for v in arch.values())
    log(f"keep-days={args.keep_days} keep-per-source={args.keep_per_source}"
        + (f" force-keep={args.force_keep}" if args.force_keep else ""))
    log(f"KEEP live={len(keep)}  ARCHIVE={n_arch}  months={len(arch)}"
        + ("  [DRY-RUN]" if not args.apply else ""))

    months = sorted(arch)
    if args.only_month:
        months = [m for m in months if m == args.only_month]
    tot_a = tot_r = 0
    for m in months:
        a, r = archive_month(m, arch[m], src_dir, archive_dir, prefix,
                             apply=args.apply, log=log)
        tot_a += a
        tot_r += r
    log(f"DONE: {'archived' if args.apply else 'would archive'} {tot_a}"
        + (f", removed {tot_r}" if args.apply else "")
        + f"; live dir would be ~{len(keep)} files")


if __name__ == "__main__":
    sys.exit(main())
