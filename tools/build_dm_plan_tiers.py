#!/usr/bin/env python3
"""Build a tiered DM plan v3 (docs/x2/SPEC.md §2) for a 2x op-point.

    python tools/build_dm_plan_tiers.py --opoint 2x-gridded-ns2-cint4 --out /path/dm_plan_2x_ns2.npz
    python tools/build_dm_plan_tiers.py --n-sub 4 --out plan.npz --verify

Every table comes from ``dsart.x2.plan.build_tiered_plan`` (oracle: snr2x
``schemes.proposed(q1=262.144, n_sub)``); the file round-trips through
``TieredDmPlan.from_npz`` and is re-verified against a rebuild from its own
stored inputs before the tool exits 0. Prints a JSON summary (tiers, coarse
counts, fines per coarse DM, per-half ownership). Pure numpy: runs on h23.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.x2.opoint import FORMATS_2X, opoint_2x, opoint_from_id  # noqa: E402
from dsart.x2.plan import TieredDmPlan, build_tiered_plan_for_opoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--opoint", help=f"2x op-point id, e.g. one of {sorted(FORMATS_2X)}")
    g.add_argument("--n-sub", type=int, choices=(2, 4), help="n_sub (value kind / bits do not change the plan)")
    ap.add_argument("--out", required=True, type=Path, help="output .npz")
    ap.add_argument("--git-sha", default=None, help="override the recorded git sha (default: this checkout)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.opoint:
        op = opoint_from_id(args.opoint)
    else:
        op = opoint_2x(n_sub=args.n_sub or 2)
    plan = build_tiered_plan_for_opoint(op, git_sha=args.git_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_npz(args.out)
    back = TieredDmPlan.from_npz(args.out, verify=True)
    summary = dict(out=str(args.out), opoint=op.opoint_id, git_sha=back.metadata["git_sha"],
                   build_utc=back.metadata["build_utc"], **back.summary())
    if not args.quiet:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
