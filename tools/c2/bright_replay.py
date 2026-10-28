#!/usr/bin/env python3
"""C2 bright-class replay: would 2x's linear bright response fire more dumps? (x2 D13 / ledger S3)

Phase 0 (docs/x2/phase0/SUMMARY.md, "Bright response") found that today's
per-row max-abs wire COMPRESSES bright signals -- observed/true = 0.97 at 35
sigma per sample, 0.93 at 50, 0.80 at 100, 0.49 at 250, saturating near 150
-- while the 2x wire is linear to the clip. So a source that reads 100 sigma
today reads ~145 at 2x. 3b43118 (2026-07-22) showed what a brighter bright
end can do: an all-night bright_frb_extragalactic storm from RFI.

This replays recorded C2 cluster rows (``c2_YYYYMMDD_HH.csv``, schema 1)
three ways, each with the per-class holdoff simulated on the rows' own
clock:

* ``1x``         today's criteria file, rows as recorded (sanity baseline);
* ``1x+lin``     today's criteria, S/N de-compressed (isolates the S/N effect);
* ``2x+lin``     the 2x criteria file (ms widths, DM to 3100), S/N
                 de-compressed, widths in 2x samples.

De-compression: per-sample S/N = snr_max / sqrt(w) is inverted through
today's curve (``rowrule_bright.json`` mc ``today|random``) to the true
per-sample S/N, mapped through the 2x curve (``g_ns2|midtread4|guard1.5|
random``), then scaled back by sqrt(w). w = the cluster's ``width_min``
(narrowest member -> largest per-sample S/N -> UPPER bound on the 2x
brightening) and ``width_median`` (LOWER bound) are both reported. The
schema-1 CSV has no peak member, so the >= 50 sigma peak-member gates fall
back to the medians (same as any pre-schema-2 replay).

Read-only: point it at a COPY of the CSVs. Pure Python (no torch); fine on
h23's CPU::

    python tools/c2/bright_replay.py --csv /tmp/c2copy/*.csv \\
        --criteria-1x configs/c2_trigger_criteria.yaml \\
        --criteria-2x configs/opoints/2x/c2_trigger_criteria.yaml \\
        --curves docs/x2/phase0/results/rowrule_bright.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DUMP_CLASSES = ("bright_frb_extragalactic", "faint_frb_extragalactic",
                "bright_galactic")
P1, P2 = 1048.576, 524.288


class Curve:
    """observed = rho * L(rho) on a tabulated grid (monotone in rho)."""

    def __init__(self, rho: Sequence[float], L: Sequence[float]) -> None:
        self.rho = np.asarray(rho, float)
        self.obs = self.rho * np.asarray(L, float)

    def forward(self, rho: float) -> float:
        return float(np.interp(rho, self.rho, self.obs))

    def inverse(self, obs: float) -> float:
        """Smallest rho with forward(rho) >= obs; beyond saturation -> max."""
        o = np.maximum.accumulate(self.obs)
        if obs <= o[0]:
            return float(obs)
        if obs >= o[-1]:
            return float(self.rho[-1])
        return float(np.interp(obs, o, self.rho))


def load_curves(path: Path, today_key: str = "today|random",
                x2_key: str = "g_ns2|midtread4|guard1.5|random") -> Tuple[Curve, Curve]:
    d = json.loads(Path(path).read_text())
    mc = d["mc"]

    def _c(key: str) -> Curve:
        pts = sorted((float(k), float(v)) for k, v in mc[key].items())
        return Curve([p[0] for p in pts], [p[1] for p in pts])

    return _c(today_key), _c(x2_key)


#: Below this per-sample S/N today's wire is linear to < 1.5% and the two
#: Monte-Carlo curves differ only by their sampling noise; leave it alone.
MIN_PER_SAMPLE: float = 20.0


def decompress(snr: float, width: float, today: Curve, x2: Curve,
               min_per_sample: float = MIN_PER_SAMPLE) -> float:
    """Today's observed S/N -> the 2x observation of the same source."""
    w = max(1.0, float(width))
    per = float(snr) / math.sqrt(w)
    if per < float(min_per_sample):
        return float(snr)
    rho = today.inverse(per)
    return x2.forward(rho) * math.sqrt(w)


def read_rows(paths: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        # a directory = its hourly files (not the rolling c2_last24h.csv)
        files.extend(sorted(p.glob("c2_2*_??.csv")) if p.is_dir() else [p])
    for p in files:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    seen = set()
    out = []
    for r in rows:   # the rolling 24 h file duplicates hourly rows
        key = tuple(r.get(k) for k in ("mjd_peak", "snr_max", "dm_median",
                                       "n_events", "trigger_class"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda r: float(r["mjd_peak"]))
    return out


def _f(r: dict, k: str, default: float = float("nan")) -> float:
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return default


def row_stats(r: dict, *, period: float = P1, width_factor: float = 1.0,
              snr_scale: float = 1.0):
    from dsart.coinc.stats import ClusterStats
    snr = _f(r, "snr_max") * snr_scale
    return ClusterStats(
        n_events=int(_f(r, "n_events", 1)), n_search_nodes=int(_f(r, "n_search_nodes", 1)),
        n_gpu_halves=int(_f(r, "n_gpu_halves", 1)), snr_max=snr,
        snr_sum=_f(r, "snr_sum") * snr_scale, snr_mean=_f(r, "snr_mean") * snr_scale,
        dm_min=_f(r, "dm_min"), dm_max=_f(r, "dm_max"), dm_median=_f(r, "dm_median"),
        dm_iqr=_f(r, "dm_iqr", 0.0), l_median=_f(r, "l_median", 0.0),
        m_median=_f(r, "m_median", 0.0), lm_diag_rad=_f(r, "lm_diag_rad", 0.0),
        width_min=int(round(_f(r, "width_min", 1) * width_factor)),
        width_max=int(round(_f(r, "width_max", 1) * width_factor)),
        width_median=_f(r, "width_median", 1) * width_factor,
        t_start_mjd=_f(r, "t_start_mjd"), t_end_mjd=_f(r, "t_end_mjd"),
        t_peak_mjd=_f(r, "mjd_peak"),
        kernel_ids_distinct=tuple(str(r.get("kernel_ids_distinct", "")).split(";")),
        peak_event_specnum=0, peak_sample_period_us=period,
        gal_dm_max_los=_f(r, "gal_dm_max_los_pc_cc"),
        dm_galactic_fraction=_f(r, "dm_galactic_fraction"),
        dm_peak=float("nan"), width_peak=0,
    )


def replay(rows: Sequence[dict], criteria_path: Path, *, opoint: Optional[str],
           mode: str, today: Optional[Curve] = None, x2: Optional[Curve] = None,
           width_key: str = "width_min") -> Dict[str, object]:
    """Evaluate every row with holdoff on the rows' own clock."""
    from dsart.coinc.criteria import CriteriaEvaluator
    clock = {"t": 0.0}
    ev = CriteriaEvaluator(criteria_path, now=lambda: clock["t"], opoint=opoint)
    classes: Counter = Counter()
    changed: List[dict] = []
    bright: Counter = Counter()
    for r in rows:
        clock["t"] = _f(r, "mjd_peak") * 86400.0
        snr = _f(r, "snr_max")
        scale = 1.0
        if mode in ("1x+lin", "2x+lin"):
            w = _f(r, width_key, 1.0)
            scale = decompress(snr, w, today, x2) / snr if snr > 0 else 1.0
        if mode == "2x+lin":
            st = row_stats(r, period=P2, width_factor=2.0, snr_scale=scale)
        else:
            st = row_stats(r, period=P1, snr_scale=scale)
        for thr in (35.0, 50.0, 100.0):
            if st.snr_max >= thr:
                bright[f"rows_snr_ge_{int(thr)}"] += 1
        tc = ev.evaluate(st)
        name = tc.name if tc is not None else "none"
        if tc is not None and tc.action == "log_only" and name != "log_only":
            name = name                      # e.g. bright_pulsar
        classes[name] += 1
        if name != r.get("trigger_class") and name in DUMP_CLASSES:
            changed.append({"mjd": r["mjd_peak"], "snr_1x": snr,
                            "snr_now": st.snr_max, "dm": r.get("dm_median"),
                            "width_median": r.get("width_median"),
                            "recorded": r.get("trigger_class"), "replayed": name})
    return {"classes": dict(classes), "changed_to_dump": changed,
            "bright": dict(bright)}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", nargs="+", type=Path, required=True)
    ap.add_argument("--criteria-1x", type=Path, required=True)
    ap.add_argument("--criteria-2x", type=Path, required=True)
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = read_rows(args.csv)
    if not rows:
        print("no rows")
        return 1
    span_d = (float(rows[-1]["mjd_peak"]) - float(rows[0]["mjd_peak"])) or 1.0
    today, x2 = load_curves(args.curves)
    res: Dict[str, object] = {"n_rows": len(rows), "span_days": span_d,
                              "recorded": dict(Counter(r.get("trigger_class") for r in rows))}
    runs = {
        "1x": dict(path=args.criteria_1x, opoint=None, mode="1x"),
        "1x+lin[w_min]": dict(path=args.criteria_1x, opoint=None, mode="1x+lin",
                              width_key="width_min"),
        "1x+lin[w_med]": dict(path=args.criteria_1x, opoint=None, mode="1x+lin",
                              width_key="width_median"),
        "2x+lin[w_min]": dict(path=args.criteria_2x, opoint="2x", mode="2x+lin",
                              width_key="width_min"),
        "2x+lin[w_med]": dict(path=args.criteria_2x, opoint="2x", mode="2x+lin",
                              width_key="width_median"),
    }
    summary = {}
    for name, kw in runs.items():
        r = replay(rows, kw.pop("path"), today=today, x2=x2, **kw)
        per_day = {k: round(v / span_d, 2) for k, v in r["classes"].items()
                   if k in DUMP_CLASSES}
        summary[name] = {"dump_class_rows_per_day": per_day,
                         "dump_rows_per_day": round(sum(
                             v for k, v in r["classes"].items() if k in DUMP_CLASSES) / span_d, 2),
                         "n_changed_to_dump": len(r["changed_to_dump"]),
                         "bright_rows_per_day": {k: round(v / span_d, 2)
                                                 for k, v in r["bright"].items()},
                         "changed_examples": r["changed_to_dump"][:10]}
    res["runs"] = summary
    txt = json.dumps(res, indent=1, default=str)
    if args.out:
        args.out.write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
