#!/usr/bin/env python3
"""Analyse an injection campaign: recovery fraction and S/N linearity.

Joins three sources:

  1. the campaign's own fired log (what was requested),
  2. ``/mon/dsart/inject/matches/<inj_id>`` (what the search actually
     recovered), and
  3. the candidate archive (whether it became a KEEP event, i.e. whether
     an operator would ever see it).

Three things worth knowing before reading the output:

* A NON-DETECTION here means "no match record", which is the union of
  "the search never triggered on it" and "it triggered but the matcher
  did not associate it". The matcher keys on DM/time/position proximity,
  so a badly mis-recovered burst can look like a non-detection. The
  observed-DM scatter column is there to expose that case.

* observed_width is the detector's chosen BOXCAR width, not the injected
  pulse width. They are different quantities and will not agree; the
  matched-filter picks the boxcar closest to the smeared width, so at
  high DM a narrow injection is legitimately recovered at a wider boxcar.

* Recovered S/N is expected to be BELOW requested at high DM even in a
  healthy pipeline, because the requested value is derived from a K fitted
  in one DM band and applied through the coarse+fine dedisperser path of
  another. Deviation from the 1:1 line is only a fault if it is
  non-monotonic or if it flattens (saturation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#: The detector clips its sigma-normalised input at +-250 sigma
#: (cube_pipeline detector_input_clip_sigma), so a recovered S/N at or
#: near this is amplitude-blind rather than a measurement.
DETECTOR_CLIP_SIGMA = 250.0


def load_fired(path: str) -> List[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_matches(host: str = "lxd110h20.pro.pvt") -> Dict[str, dict]:
    import etcd3
    c = etcd3.client(host=host, port=2379, timeout=10)
    out = {}
    for kv, meta in c.get_prefix("/mon/dsart/inject/matches/"):
        try:
            d = json.loads(kv.decode())
        except Exception:                                     # noqa: BLE001
            continue
        iid = d.get("inj_id") or meta.key.decode().rsplit("/", 1)[-1]
        out[iid] = d
    return out


def load_archive_actions(root: str = "/dataz/dsa110/candidates") -> List[dict]:
    """Injection-labelled events and their C3 action, for the KEEP check."""
    import glob
    out = []
    for d in glob.glob(os.path.join(root, "2608*")):
        try:
            j = json.load(open(os.path.join(d, "C3_decision.json")))
        except Exception:                                     # noqa: BLE001
            continue
        if not j.get("is_injection"):
            continue
        m = j.get("metrics") or {}
        out.append({"event": os.path.basename(d), "action": j.get("action"),
                    "rules": j.get("rules_fired") or [],
                    "mjd": m.get("mjd"), "snr_c1": m.get("snr_c1"),
                    "dm_c1": m.get("dm_c1")})
    return out


def join(fired: List[dict], matches: Dict[str, dict]) -> List[dict]:
    rows = []
    for f in fired:
        m = matches.get(f["inj_id"])
        best = (m or {}).get("best") or {}
        rows.append({
            **{k: f.get(k) for k in
               ("inj_id", "tag", "dm", "width", "fluence", "want_snr", "K_used",
                "fired_at", "l", "m")},
            "detected": bool(best),
            "obs_snr": best.get("observed_snr"),
            "obs_dm": best.get("observed_dm_pc_cm3"),
            "obs_width": best.get("observed_width_samples"),
            "obs_node": best.get("observed_search_node_id"),
            "obs_half": best.get("observed_gpu_half"),
        })
    return rows


def _fmt(x: Optional[float], w: int = 8, p: int = 2) -> str:
    return ("%*.*f" % (w, p, x)) if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def report(rows: List[dict]) -> None:
    grid = [r for r in rows if r["tag"] == "grid"]
    print("\n=== RECOVERY FRACTION ===")
    print("  overall: %d/%d = %.1f%%"
          % (sum(r["detected"] for r in grid), len(grid),
             100 * sum(r["detected"] for r in grid) / max(1, len(grid))))

    for key, label in (("want_snr", "requested S/N"), ("dm", "DM"),
                       ("width", "injected width")):
        print("\n  by %s:" % label)
        vals = sorted({r[key] for r in grid})
        print("    %-10s %6s %6s %8s   %s" % (label, "n", "det", "frac", ""))
        for v in vals:
            s = [r for r in grid if r[key] == v]
            d = sum(r["detected"] for r in s)
            bar = "#" * int(round(24 * d / max(1, len(s))))
            print("    %-10s %6d %6d %7.0f%%   %s"
                  % (v, len(s), d, 100 * d / max(1, len(s)), bar))

    print("\n  recovery fraction, DM x requested S/N:")
    dms = sorted({r["dm"] for r in grid})
    snrs = sorted({r["want_snr"] for r in grid})
    print("    %-8s %s" % ("DM\\SNR", " ".join("%7.0f" % s for s in snrs)))
    for dm in dms:
        cells = []
        for s in snrs:
            sub = [r for r in grid if r["dm"] == dm and r["want_snr"] == s]
            cells.append("%6.0f%%" % (100 * sum(x["detected"] for x in sub)
                                      / max(1, len(sub))) if sub else "      -")
        print("    %-8.0f %s" % (dm, " ".join(cells)))

    print("\n=== RECOVERED vs REQUESTED S/N ===")
    det = [r for r in grid if r["detected"] and r["obs_snr"]]
    print("  %-6s %-7s %8s %9s %8s %8s"
          % ("DM", "width", "want", "obs(med)", "ratio", "n"))
    for dm in dms:
        for w in sorted({r["width"] for r in grid}):
            for s in snrs:
                sub = [r for r in det if r["dm"] == dm and r["width"] == w
                       and r["want_snr"] == s]
                if not sub:
                    continue
                o = np.median([r["obs_snr"] for r in sub])
                print("  %-6.0f %-7d %8.1f %9.2f %8.2f %8d"
                      % (dm, w, s, o, o / s, len(sub)))

    railed = [r for r in det if r["obs_snr"] >= 0.95 * DETECTOR_CLIP_SIGMA]
    print("\n  probes at/near the %.0f sigma detector clip: %d"
          % (DETECTOR_CLIP_SIGMA, len(railed)))
    for r in railed[:6]:
        print("    %s want=%.0f obs=%.1f" % (r["inj_id"], r["want_snr"], r["obs_snr"]))

    print("\n=== SATURATION CHECK (is the ratio flat with requested S/N?) ===")
    print("  A healthy linear regime keeps obs/want roughly constant.")
    print("  %-8s %9s %9s %7s" % ("want", "med ratio", "p10", "n"))
    for s in snrs:
        sub = [r["obs_snr"] / r["want_snr"] for r in det if r["want_snr"] == s]
        if sub:
            print("  %-8.0f %9.3f %9.3f %7d"
                  % (s, float(np.median(sub)), float(np.percentile(sub, 10)),
                     len(sub)))

    blind = [r for r in rows if str(r["tag"]).startswith("blind")]
    if blind:
        print("\n=== BLIND SPOT: sigma_k suppression after a bright probe ===")
        print("  Same DM and position throughout. A faint probe fired soon")
        print("  after the bright one competes with its own predecessor's")
        print("  inflation of the Layer-2 sigma_k EMA at that (DM, pixel).")
        print("  %-18s %-6s %8s %9s %8s" % ("tag", "DM", "want", "obs", "det"))
        for r in sorted(blind, key=lambda x: (x["dm"], x["fired_at"])):
            print("  %-18s %-6.0f %8.1f %9s %8s"
                  % (r["tag"], r["dm"], r["want_snr"],
                     _fmt(r["obs_snr"], 9), "Y" if r["detected"] else "N"))


def figures(rows: List[dict], out: str) -> None:
    grid = [r for r in rows if r["tag"] == "grid"]
    det = [r for r in grid if r["detected"] and r["obs_snr"]]
    dms = sorted({r["dm"] for r in grid})
    snrs = sorted({r["want_snr"] for r in grid})
    widths = sorted({r["width"] for r in grid})

    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for w in widths:
            fr = [100 * np.mean([r["detected"] for r in grid
                                 if r["want_snr"] == s and r["width"] == w] or [0])
                  for s in snrs]
            ax.plot(snrs, fr, "-o", label="width %d" % w)
        ax.set_xlabel("requested S/N"); ax.set_ylabel("recovered fraction (%)")
        ax.set_ylim(-3, 103); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("Recovery vs requested S/N", fontsize=10)

        ax = axes[1]
        for dm in dms:
            fr = [100 * np.mean([r["detected"] for r in grid
                                 if r["want_snr"] == s and r["dm"] == dm] or [0])
                  for s in snrs]
            ax.plot(snrs, fr, "-o", label="DM %.0f" % dm)
        ax.set_xlabel("requested S/N"); ax.set_ylabel("recovered fraction (%)")
        ax.set_ylim(-3, 103); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        ax.set_title("Recovery vs requested S/N, by DM", fontsize=10)
        fig.suptitle("Injection recovery", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for dm in dms:
            s = [r for r in det if r["dm"] == dm]
            if s:
                ax.plot([r["want_snr"] for r in s], [r["obs_snr"] for r in s],
                        "o", ms=5, alpha=.7, label="DM %.0f" % dm)
        lim = max([r["want_snr"] for r in grid] + [1])
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
        ax.axhline(DETECTOR_CLIP_SIGMA, color="r", ls=":", lw=1,
                   label="detector clip 250")
        ax.set_xlabel("requested S/N"); ax.set_ylabel("recovered S/N")
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        ax.set_title("Linearity — deviation is only a fault if it flattens",
                     fontsize=10)

        ax = axes[1]
        for w in widths:
            s = [r for r in det if r["width"] == w]
            if s:
                ax.plot([r["want_snr"] for r in s],
                        [r["obs_snr"] / r["want_snr"] for r in s],
                        "o", ms=5, alpha=.7, label="width %d" % w)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("requested S/N"); ax.set_ylabel("recovered / requested")
        ax.set_ylim(0, 2); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("Ratio — flat means linear, falling means saturating",
                     fontsize=10)
        fig.suptitle("Recovered vs injected S/N", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)

        blind = [r for r in rows if str(r["tag"]).startswith("blind")]
        if blind:
            fig, ax = plt.subplots(figsize=(9, 5))
            for dm in sorted({r["dm"] for r in blind}):
                s = sorted([r for r in blind if r["dm"] == dm],
                           key=lambda x: x["fired_at"])
                if not s:
                    continue
                t0 = s[0]["fired_at"]
                xs = [(r["fired_at"] - t0) for r in s[1:]]
                ys = [(r["obs_snr"] / r["want_snr"]) if r["obs_snr"] else 0.0
                      for r in s[1:]]
                ax.plot(xs, ys, "-o", label="DM %.0f" % dm)
            ax.axhline(1.0, color="k", ls="--", lw=1, label="unsuppressed")
            ax.set_xlabel("seconds after the bright (60 sigma) probe")
            ax.set_ylabel("faint-probe recovered / requested")
            ax.set_ylim(0, 1.6); ax.grid(alpha=.3); ax.legend(fontsize=8)
            ax.set_title("Blind spot: sigma_k recovery at the same (DM, pixel)",
                         fontweight="bold")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("\nwrote %s" % out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fired", required=True)
    ap.add_argument("--out", default="inject_campaign.pdf")
    a = ap.parse_args()
    fired = load_fired(a.fired)
    matches = load_matches()
    rows = join(fired, matches)
    print("fired=%d  matched=%d" % (len(rows), sum(r["detected"] for r in rows)))
    report(rows)
    figures(rows, a.out)
    json.dump(rows, open(a.out.replace(".pdf", "_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
