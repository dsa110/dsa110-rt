#!/usr/bin/env python3
"""R12 (DM-time ridge) shadow report: what C3 did vs what R12 would have done.

Reads ``<archive_root>/<ev>/C3_decision.json`` (C3 writes it for KEEPs and
REJECTs alike) for events decided since ``--since`` (UTC) and prints, per
event: C3 action + rules, injection flag, C2 trigger class / S/N / DM /
width, R12's verdict with its line frequency, fitted bandwidth B, f and
significance, and the dashboard label (current majority label, read-only).
The summary is the go/no-go evidence for ``c3.r12.mode: reject``:
R12 must never fire on an injection or on anything labelled FRB / PULSAR.

    python tools/x2/r12_shadow_report.py --since "2026-11-12 08:00"
"""

from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict


def labels(db: str) -> dict:
    p = os.path.expanduser(db)
    if not os.path.exists(p):
        return {}
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    rows = con.execute("""SELECT c.event, c.label FROM classifications c
      JOIN (SELECT event, user, MAX(id) AS mid FROM classifications GROUP BY event, user) t ON c.id = t.mid
      WHERE c.label IS NOT NULL""").fetchall()
    per = defaultdict(Counter)
    for e, l in rows:
        per[e][l] += 1
    return {e: c.most_common(1)[0][0] for e, c in per.items()}


def level3(root: str, rejected: str, ev: str) -> dict:
    for r in (root, rejected):
        try:
            return json.load(open(f"{r}/{ev}/Level3/{ev}.json"))
        except (OSError, ValueError):
            continue
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help='UTC, "YYYY-mm-dd HH:MM"')
    ap.add_argument("--archive-root", default="/dataz/dsa110/candidates")
    ap.add_argument("--rejected-root", default="/dataz/dsa110/candidates_rejected")
    ap.add_argument("--labels-db", default="~/.dsa_monitor/annotations.db")
    ap.add_argument("--json", action="store_true", help="one JSON object per event instead of the table")
    a = ap.parse_args(argv)
    since = calendar.timegm(time.strptime(a.since, "%Y-%m-%d %H:%M"))
    lab = labels(a.labels_db)
    rows = []
    for p in glob.glob(f"{a.archive_root}/*/C3_decision.json"):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        if float(d.get("decided_at_unix") or 0) < since or "r12" not in d:
            continue
        ev = d.get("event_name") or os.path.basename(os.path.dirname(p))
        l3 = level3(a.archive_root, a.rejected_root, ev)
        rows.append((float(d["decided_at_unix"]), ev, d, l3, lab.get(ev, "")))
    rows.sort()
    summ = Counter()
    for t, ev, d, l3, label in rows:
        r = d["r12"]
        f = r.get("features") or {}
        c2 = l3.get("c2") or {}
        tag = ("n/a" if not r.get("ok") else ("REJECT" if r.get("acted") else "FLAG")
               if r.get("would_reject") else "pass")
        inj = bool(d.get("is_injection"))
        summ[(d.get("action"), "inj" if inj else "sky", tag)] += 1
        if label:
            summ[("label", label, tag)] += 1
        if a.json:
            print(json.dumps({"event": ev, "decided_at_unix": t, "c3": d.get("action"), "rules": d.get("rules_fired"),
                              "injection": inj, "label": label, "r12": r, "c2": c2,
                              "trigger": (l3.get("trigger") or {}).get("class")}))
            continue
        print(f"{time.strftime('%m-%d %H:%M', time.gmtime(t))} {ev} C3 {d.get('action', '?'):6s} "
              f"{'INJ ' if inj else '    '}{(l3.get('trigger') or {}).get('class', '')[:20]:20s} "
              f"S/N {float(c2.get('snr_max') or 0):6.1f} DM {float(c2.get('dm_median') or 0):7.1f} "
              f"w {float(c2.get('width_median_ms') or 0):5.2f} | R12 {tag:6s} "
              f"{f.get('freq_mhz', 0):7.1f} MHz B {f.get('bw_fit_mhz', 0):6.2f} f {f.get('f_fit', 0):5.2f} "
              f"{f.get('fit_sigma', 0):5.0f}sig | {label}")
    if not a.json:
        print(f"\n{len(rows)} events with R12 since {a.since} UTC")
        for k, v in sorted(summ.items(), key=lambda kv: str(kv[0])):
            print("  ", k, v)
        bad_inj = summ[("KEEP", "inj", "FLAG")] + summ[("KEEP", "inj", "REJECT")]
        bad_lab = sum(v for k, v in summ.items() if k[0] == "label" and k[1] in ("FRB", "PULSAR")
                      and k[2] in ("FLAG", "REJECT"))
        print(f"SAFETY: R12 on injections {bad_inj} (must be 0); on events labelled FRB/PULSAR {bad_lab} (must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
