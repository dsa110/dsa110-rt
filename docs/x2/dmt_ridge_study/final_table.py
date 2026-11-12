"""Final per-class table for the proposed rule (B <= 6 MHz, f >= 0.45, sig >= 8) + flagged unlabelled sky."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_eval as E  # noqa: E402


def flag(f):
    return bool(f.get("r_ok")) and (f.get("r_bw_fit_mhz") or 1e9) <= 6 and (f.get("r_f_fit") or 0) >= 0.45 \
        and (f.get("r_fit_sigma") or 0) >= 8


tab = defaultdict(lambda: [0, 0])
for ev, cls, src, f in E.items:
    tab[(cls, src)][0] += flag(f)
    tab[(cls, src)][1] += 1
for k in sorted(tab):
    print(f"{k[0]:16s} {k[1]:8s} flagged {tab[k][0]}/{tab[k][1]}")
print("\nunlabelled 2x sky (pix): flagged events by C3 action")
by = defaultdict(list)
for ev, cls, src, f in E.items:
    if cls in ("other", "narrow") and src == "cut-pix":
        by[(flag(f), E.c3(ev).get("action", "?"))].append(f"{ev}@{f.get('r_freq_mhz')}")
for k, v in sorted(by.items()):
    print(" flag" if k[0] else " pass", k[1], len(v), v)
syn = [s for s in E.syn if s.get("r_ok")]
print("\nsynthetic flag rate:", {f"{lo}-{hi}": round(sum(flag(s) for s in syn if lo <= s['occ_true_mhz'] < hi) /
                                                    max(1, sum(1 for s in syn if lo <= s['occ_true_mhz'] < hi)), 2)
                                  for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 400)]})
