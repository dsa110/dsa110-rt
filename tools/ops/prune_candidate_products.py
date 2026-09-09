#!/usr/bin/env python3
"""Prune bulk data products (cubes, voltages) from aged-out KEEP candidates.

A C3 KEEP event keeps everything it staged: the search cubes (8 x ~1.1 GiB, one
per search-node GPU half) and, if a voltage dump fired, ~103 GiB of raw voltage
fragments under ``Level2/voltages``.  Nothing ever reclaims them -- the only
retention that exists is the corr-side ``operations/correlator`` sweeper and the
search-node staging cap, neither of which reaches into the candidate archive.
At ~1.4 TiB/day of archive growth that is the dominant consumer of /dataz.

This tool deletes *only* the two bulk products, and only for events that are

  * older than ``--older-than-days`` (default 14),
  * newer than ``--newer-than``     (default 2026-05-01; the older archive is
    hand-curated and deliberately out of scope),
  * not protected by label, source name, or an explicit keep marker.

Everything that makes an event reviewable afterwards is left alone:
``C3_decision.json``, ``Level3/*.json``, the ``Level2/*.csv`` candidate rows,
the four ``Level2/plots`` diagnostics, and ``filterbank/`` (the dedispersed
product *derived* from the voltages, ~0.5 GiB vs ~103 GiB raw).  ``calibration/``
is also left alone: those files are hardlinks into ``operations/correlator``
while it still holds the original, so deleting them here frees nothing and
would destroy the sole copy once the Level1 cron ages the original out.

Protection is deliberately generous, because the failure is asymmetric -- a
wrongly-kept cube costs 1.1 GiB, a wrongly-deleted one is gone:

  * ``--protect-labels`` (default ``FRB,PULSAR``).  An event is protected if
    *any* user's most recent classification is in this set -- union, not
    majority, so one person's FRB call outvotes three RFI calls.  PULSAR is
    protected by default because the B1933+16 detections are the array's
    verification and polarisation-calibration set, not spam.
  * ``--protect-named-sources`` (default on).  Any event a human gave a source
    name to in the annotation UI is protected regardless of label.
  * ``--include-unclassified`` (default OFF).  An event nobody has looked at is
    not the same as an event judged uninteresting.  Unclassified events also
    have no ``filterbank/``, so their cube is the only data that exists.

To get the strictly literal "everything not labelled FRB" rule:

    --protect-labels FRB --no-protect-named-sources --include-unclassified

Dry-run is the default; ``--execute`` is required to unlink anything, and an
audit manifest naming every deleted path is written *before* the first unlink.

Examples
--------
    # what would go, default policy
    ./prune_candidate_products.py

    # do it
    ./prune_candidate_products.py --execute

    # include the unreviewed backlog too
    ./prune_candidate_products.py --include-unclassified --execute
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

GiB = 1 << 30
TiB = 1 << 40

DEFAULT_ROOT = "/dataz/dsa110/candidates"
DEFAULT_DB = os.path.expanduser("~/.dsa_monitor/annotations.db")
DEFAULT_REPORT_DIR = "/dataz/dsa110/dr_archive/prune"

#: Products eligible for deletion, relative to the event directory.  Order is
#: cosmetic (report ordering only).
DEFAULT_PRODUCTS = ("cubes", "voltages", "Level2/voltages")

#: Never delete these, even if a caller passes them via --products.  Belt and
#: braces against a typo turning this into an archive shredder.
NEVER_DELETE = frozenset(
    {
        "",
        ".",
        "..",
        "/",
        "C3_decision.json",
        "Level3",
        "calibration",
        "filterbank",
        "Level2",  # the *directory*; Level2/voltages specifically is fine
        "Level2/plots",
    }
)

#: An event directory name: YYMMDD + four lowercase letters (260715twmx).
EVENT_RE = re.compile(r"^(\d{6})[a-z]{4}$")

#: Dropped into an emptied product directory so a later reader can tell
#: "pruned to reclaim space" from "never staged" -- the two look identical
#: otherwise, and the difference matters when chasing dump failures.
MARKER_NAME = ".pruned.json"

#: Refuse to run if the annotation DB yields fewer than this many protected
#: events overall.  A missing/truncated/wrong-path DB makes every event look
#: unclassified, which combined with --include-unclassified would delete the
#: entire archive.  Two FRBs exist as of 2026-08; require at least one.
MIN_PROTECTED_SANITY = 1


# --------------------------------------------------------------------------- #
# annotation database
# --------------------------------------------------------------------------- #


@dataclass
class Annotations:
    """Resolved human annotations, keyed by event name."""

    #: event -> set of the most recent non-NULL label of each user
    labels: Dict[str, Set[str]] = field(default_factory=dict)
    #: event -> most recent non-NULL source name, if any
    source_names: Dict[str, str] = field(default_factory=dict)

    def label_key(self, event: str) -> str:
        ls = self.labels.get(event)
        return ",".join(sorted(ls)) if ls else "<UNCLASSIFIED>"


def load_annotations(db_path: str) -> Annotations:
    """Read the append-only annotation store and collapse it to last-click-wins.

    The store keeps every click, so "the current label" is the newest row per
    (event, user) -- and a NULL label is an explicit "I cleared mine", which
    must not fall back to that user's earlier opinion.
    """
    if not os.path.exists(db_path):
        raise SystemExit(
            "annotation DB not found: %s\n"
            "Refusing to run: without labels every event looks unclassified." % db_path
        )
    uri = "file:%s?mode=ro" % db_path
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:  # pragma: no cover - environment dependent
        raise SystemExit("cannot open annotation DB %s: %s" % (db_path, exc))

    ann = Annotations()
    try:
        # ORDER BY ts_utc so the last write per (event, user) wins.  ISO-8601
        # UTC strings sort lexicographically, which is why they are stored that
        # way; id is the tiebreak for same-millisecond double clicks.
        per_user: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
        for event, user, label in conn.execute(
            "SELECT event, user, label FROM classifications ORDER BY ts_utc, id"
        ):
            per_user[event][user] = label
        for event, users in per_user.items():
            live = {lbl for lbl in users.values() if lbl}
            if live:
                ann.labels[event] = live

        for event, name in conn.execute(
            "SELECT event, source_name FROM source_names ORDER BY ts_utc, id"
        ):
            if name:
                ann.source_names[event] = name
            else:
                ann.source_names.pop(event, None)  # explicit clear
    finally:
        conn.close()
    return ann


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    name: str
    path: str
    date: dt.date
    label_key: str
    source_name: Optional[str]
    #: relative product dir -> (n_files, bytes)
    products: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    protected_by: List[str] = field(default_factory=list)

    @property
    def n_bytes(self) -> int:
        return sum(b for _, b in self.products.values())

    @property
    def n_files(self) -> int:
        return sum(n for n, _ in self.products.values())


def parse_event_date(name: str) -> Optional[dt.date]:
    """Event names carry their UTC date as YYMMDD; 2000-relative."""
    m = EVENT_RE.match(name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%y%m%d").date()
    except ValueError:
        return None


def measure_product(path: str) -> Tuple[int, int]:
    """Return (n_files, apparent_bytes) for one product dir, non-recursive-safe.

    Walks, but refuses to cross a symlink -- a symlinked product dir would
    otherwise let deletion escape the candidate tree.
    """
    n = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        # don't descend into symlinked subdirs
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if fn == MARKER_NAME:
                continue
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            n += 1
            total += st.st_size
    return n, total


def keep_marker_present(event_path: str, markers: Sequence[str]) -> Optional[str]:
    for m in markers:
        if os.path.exists(os.path.join(event_path, m)):
            return m
    return None


def select_events(
    root: str,
    ann: Annotations,
    *,
    cutoff_old: dt.date,
    cutoff_new: dt.date,
    protect_labels: Set[str],
    protect_named: bool,
    include_unclassified: bool,
    products: Sequence[str],
    keep_markers: Sequence[str],
    exclude: Set[str],
) -> Tuple[List[Event], List[Event], Dict[str, int]]:
    """Partition the archive into (targets, protected, counters).

    ``targets`` are events whose products should go; ``protected`` are in-window
    events spared, each carrying the reason(s) in ``protected_by``.  Events
    outside the date window are not returned at all -- they are not decisions.
    """
    counters: Dict[str, int] = defaultdict(int)
    targets: List[Event] = []
    protected: List[Event] = []

    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        raise SystemExit("cannot list candidate root %s: %s" % (root, exc))

    for name in names:
        path = os.path.join(root, name)
        date = parse_event_date(name)
        if date is None or not os.path.isdir(path) or os.path.islink(path):
            counters["skipped_not_an_event"] += 1
            continue
        counters["events_total"] += 1

        if not (cutoff_new < date < cutoff_old):
            counters["out_of_window"] += 1
            continue
        counters["in_window"] += 1

        ev = Event(
            name=name,
            path=path,
            date=date,
            label_key=ann.label_key(name),
            source_name=ann.source_names.get(name),
        )

        labels = ann.labels.get(name, set())
        if labels & protect_labels:
            ev.protected_by.append("label=%s" % ",".join(sorted(labels & protect_labels)))
        if protect_named and ev.source_name:
            ev.protected_by.append("source_name=%s" % ev.source_name)
        if not labels and not include_unclassified:
            ev.protected_by.append("unclassified")
        marker = keep_marker_present(path, keep_markers)
        if marker:
            ev.protected_by.append("marker=%s" % marker)
        if name in exclude:
            ev.protected_by.append("excluded")

        # Cross-check the name-derived date against the directory mtime.  If the
        # directory was touched recently something may still be writing to it,
        # so leave it be regardless of what the name says.
        try:
            mtime = dt.datetime.utcfromtimestamp(os.stat(path).st_mtime).date()
        except OSError:
            mtime = date
        if mtime >= cutoff_old:
            ev.protected_by.append("mtime=%s newer than cutoff" % mtime.isoformat())

        for rel in products:
            p = os.path.join(path, rel)
            if not os.path.isdir(p) or os.path.islink(p):
                continue
            n, b = measure_product(p)
            if n:
                ev.products[rel] = (n, b)

        if ev.protected_by:
            protected.append(ev)
            counters["protected"] += 1
        elif not ev.products:
            counters["nothing_to_prune"] += 1
        else:
            targets.append(ev)
            counters["targeted"] += 1

    return targets, protected, dict(counters)


# --------------------------------------------------------------------------- #
# deletion
# --------------------------------------------------------------------------- #


def validate_products(products: Iterable[str], root: str) -> List[str]:
    out = []
    for rel in products:
        rel = rel.strip().strip("/")
        if rel in NEVER_DELETE or rel.startswith(("/", "..")) or ".." in rel.split("/"):
            raise SystemExit("refusing to treat %r as a deletable product" % rel)
        out.append(rel)
    if not out:
        raise SystemExit("no products selected")
    return out


def write_marker(product_path: str, event: str, rel: str, n: int, nbytes: int, stamp: str) -> None:
    payload = {
        "pruned_utc": stamp,
        "event": event,
        "product": rel,
        "n_files_deleted": n,
        "bytes_deleted": nbytes,
        "tool": os.path.basename(__file__),
        "note": "Bulk product reclaimed to free /dataz. Absence of files here "
        "does NOT mean the dump failed.",
    }
    try:
        with open(os.path.join(product_path, MARKER_NAME), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        print("  warn: could not write marker in %s: %s" % (product_path, exc), file=sys.stderr)


def delete_products(targets: List[Event], root: str, stamp: str) -> Tuple[int, int, int]:
    """Unlink the selected products. Returns (n_files, n_bytes, n_errors)."""
    real_root = os.path.realpath(root)
    n_files = n_bytes = n_err = 0
    for ev in targets:
        for rel, (n, b) in sorted(ev.products.items()):
            p = os.path.join(ev.path, rel)
            # Final guard: the resolved path must still live under the archive
            # root and still be a real directory.
            if not os.path.realpath(p).startswith(real_root + os.sep):
                print("  ERROR escaped root, skipping: %s" % p, file=sys.stderr)
                n_err += 1
                continue
            if not os.path.isdir(p) or os.path.islink(p):
                continue
            try:
                for entry in os.listdir(p):
                    ep = os.path.join(p, entry)
                    if os.path.islink(ep) or os.path.isfile(ep):
                        os.unlink(ep)
                    elif os.path.isdir(ep):
                        shutil.rmtree(ep)
            except OSError as exc:
                print("  ERROR %s: %s" % (p, exc), file=sys.stderr)
                n_err += 1
                continue
            n_files += n
            n_bytes += b
            write_marker(p, ev.name, rel, n, b, stamp)
    return n_files, n_bytes, n_err


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def summarise(targets: List[Event], protected: List[Event], products: Sequence[str]) -> None:
    by_label: Dict[str, List[Event]] = defaultdict(list)
    for ev in targets:
        by_label[ev.label_key].append(ev)

    print("\n=== TO DELETE, by classification ===")
    hdr = "%-20s %5s %8s" % ("label", "n_ev", "files")
    for rel in products:
        hdr += " %13s" % rel
    print(hdr + " %10s" % "total")
    grand = defaultdict(int)
    for key in sorted(by_label, key=lambda k: -sum(e.n_bytes for e in by_label[k])):
        evs = by_label[key]
        line = "%-20s %5d %8d" % (key, len(evs), sum(e.n_files for e in evs))
        for rel in products:
            b = sum(e.products.get(rel, (0, 0))[1] for e in evs)
            grand[rel] += b
            line += " %12.1fG" % (b / GiB)
        line += " %9.1fG" % (sum(e.n_bytes for e in evs) / GiB)
        print(line)
    total = sum(e.n_bytes for e in targets)
    line = "%-20s %5d %8d" % ("TOTAL", len(targets), sum(e.n_files for e in targets))
    for rel in products:
        line += " %12.1fG" % (grand[rel] / GiB)
    print("-" * len(hdr))
    print(line + " %9.1fG" % (total / GiB))
    print("\nreclaim: %.2f TiB from %d events" % (total / TiB, len(targets)))

    reasons: Dict[str, List[Event]] = defaultdict(list)
    for ev in protected:
        for r in ev.protected_by:
            reasons[r.split("=")[0]].append(ev)
    print("\n=== PROTECTED (in window, spared) ===")
    for r in sorted(reasons, key=lambda k: -len(reasons[k])):
        evs = reasons[r]
        held = sum(e.n_bytes for e in evs)
        print("  %-18s %4d events  %8.1f GiB held" % (r, len(evs), held / GiB))
    # Name the individually interesting ones -- a bare count hides whether the
    # thing you cared about was actually spared.
    named = [e for e in protected if any(p.startswith(("label", "source_name")) for p in e.protected_by)]
    if named:
        print("\n  by label / source name:")
        for ev in sorted(named, key=lambda e: e.name):
            print(
                "    %-14s %-14s %8.1f GiB  <- %s"
                % (ev.name, ev.label_key, ev.n_bytes / GiB, "; ".join(ev.protected_by))
            )


def write_manifest(
    report_dir: str, stamp: str, args: argparse.Namespace, targets: List[Event], protected: List[Event]
) -> str:
    os.makedirs(report_dir, mode=0o700, exist_ok=True)
    path = os.path.join(report_dir, "prune_%s.json" % stamp)
    payload = {
        "stamp_utc": stamp,
        "executed": bool(args.execute),
        "argv": sys.argv,
        "policy": {
            "root": args.root,
            "older_than_days": args.older_than_days,
            "newer_than": args.newer_than,
            "protect_labels": sorted(args.protect_labels),
            "protect_named_sources": args.protect_named_sources,
            "include_unclassified": args.include_unclassified,
            "products": list(args.products),
        },
        "targets": [
            {
                "event": e.name,
                "date": e.date.isoformat(),
                "label": e.label_key,
                "bytes": e.n_bytes,
                "products": {k: {"n_files": v[0], "bytes": v[1]} for k, v in e.products.items()},
                "paths": sorted(os.path.join(e.path, r) for r in e.products),
            }
            for e in sorted(targets, key=lambda x: x.name)
        ],
        "protected": [
            {
                "event": e.name,
                "date": e.date.isoformat(),
                "label": e.label_key,
                "source_name": e.source_name,
                "bytes_held": e.n_bytes,
                "reasons": e.protected_by,
            }
            for e in sorted(protected, key=lambda x: x.name)
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.chmod(path, 0o600)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default=DEFAULT_ROOT, help="candidate archive root")
    p.add_argument("--db", default=DEFAULT_DB, help="human annotation sqlite DB")
    p.add_argument(
        "--older-than-days", type=int, default=14, metavar="N",
        help="only events strictly older than N days (default 14)",
    )
    p.add_argument(
        "--newer-than", default="2026-05-01", metavar="YYYY-MM-DD",
        help="only events strictly newer than this date (default 2026-05-01)",
    )
    p.add_argument(
        "--protect-labels", default="FRB,PULSAR", metavar="A,B",
        help="never prune an event any user currently labels with one of these "
             "(default FRB,PULSAR)",
    )
    p.add_argument(
        "--no-protect-named-sources", dest="protect_named_sources",
        action="store_false", default=True,
        help="also prune events a human gave a source name to",
    )
    p.add_argument(
        "--include-unclassified", action="store_true",
        help="also prune events nobody has classified (default: spare them)",
    )
    p.add_argument(
        "--products", default=",".join(DEFAULT_PRODUCTS), metavar="A,B",
        help="event-relative product dirs to empty (default %s)" % ",".join(DEFAULT_PRODUCTS),
    )
    p.add_argument(
        "--keep-marker", default="KEEP,KEEP_VOLTAGES,.keep", metavar="A,B",
        help="filenames in an event dir that veto pruning",
    )
    p.add_argument("--exclude-events", default="", metavar="A,B", help="explicit events to spare")
    p.add_argument(
        "--exclude-events-file", metavar="PATH",
        help="file of event names to spare, one per line (# comments ok)",
    )
    p.add_argument(
        "--max-delete-tib", type=float, default=10.0, metavar="X",
        help="abort if the selection exceeds this much data (default 10)",
    )
    p.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="where to write the manifest")
    p.add_argument("--execute", action="store_true", help="actually delete (default: dry run)")
    args = p.parse_args(argv)

    args.protect_labels = {s.strip().upper() for s in args.protect_labels.split(",") if s.strip()}
    args.products = validate_products(args.products.split(","), args.root)
    args.keep_marker = [s.strip() for s in args.keep_marker.split(",") if s.strip()]

    excl = {s.strip() for s in args.exclude_events.split(",") if s.strip()}
    if args.exclude_events_file:
        with open(args.exclude_events_file) as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    excl.add(line)
    args.exclude = excl
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    now = dt.datetime.utcnow()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    cutoff_old = (now - dt.timedelta(days=args.older_than_days)).date()
    cutoff_new = dt.datetime.strptime(args.newer_than, "%Y-%m-%d").date()
    if cutoff_new >= cutoff_old:
        raise SystemExit(
            "empty window: --newer-than %s is not before the age cutoff %s"
            % (cutoff_new, cutoff_old)
        )

    ann = load_annotations(args.db)
    n_prot_rows = sum(1 for ls in ann.labels.values() if ls & args.protect_labels)
    if n_prot_rows < MIN_PROTECTED_SANITY:
        raise SystemExit(
            "annotation DB %s has %d events labelled %s -- expected at least %d.\n"
            "Refusing to run: this looks like the wrong or a truncated database."
            % (args.db, n_prot_rows, sorted(args.protect_labels), MIN_PROTECTED_SANITY)
        )

    print("root            : %s" % args.root)
    print("annotations     : %s (%d classified events, %d protected-label)"
          % (args.db, len(ann.labels), n_prot_rows))
    print("window          : %s < event date < %s  (older than %d days)"
          % (cutoff_new, cutoff_old, args.older_than_days))
    print("products        : %s" % ", ".join(args.products))
    print("protect labels  : %s" % ", ".join(sorted(args.protect_labels)))
    print("protect named   : %s" % args.protect_named_sources)
    print("unclassified    : %s" % ("PRUNE" if args.include_unclassified else "spare"))
    print("mode            : %s" % ("EXECUTE" if args.execute else "dry run"))

    targets, protected, counters = select_events(
        args.root,
        ann,
        cutoff_old=cutoff_old,
        cutoff_new=cutoff_new,
        protect_labels=args.protect_labels,
        protect_named=args.protect_named_sources,
        include_unclassified=args.include_unclassified,
        products=args.products,
        keep_markers=args.keep_marker,
        exclude=args.exclude,
    )
    print("\ncounters: %s" % json.dumps(counters, sort_keys=True))
    summarise(targets, protected, args.products)

    total = sum(e.n_bytes for e in targets)
    if total / TiB > args.max_delete_tib:
        raise SystemExit(
            "\nABORT: selection is %.2f TiB, over the --max-delete-tib %.2f cap."
            % (total / TiB, args.max_delete_tib)
        )

    # Manifest first: if deletion dies halfway we still know what was in scope.
    manifest = write_manifest(args.report_dir, stamp, args, targets, protected)
    print("\nmanifest: %s" % manifest)

    if not args.execute:
        print("\nDRY RUN -- nothing deleted. Re-run with --execute.")
        return 0
    if not targets:
        print("\nnothing to do.")
        return 0

    print("\ndeleting ...")
    n_files, n_bytes, n_err = delete_products(targets, args.root, stamp)
    print("deleted %d files, %.2f TiB, %d errors" % (n_files, n_bytes / TiB, n_err))
    try:
        st = os.statvfs(args.root)
        print("%s now %.2f TiB free" % (args.root, st.f_bavail * st.f_frsize / TiB))
    except OSError:
        pass
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
