#!/usr/bin/env python3
"""Push dsart-rt pipeline configs into etcd.

M7 chunk 0. The dsart-rt orchestrator (`services/dsart_rt.py`) reads
its per-instance config from etcd keys:

  /cnf/pipeline_rt   -> consumed by `dsart_rt.py -in pipeline_rt`
  /cnf/search_rt     -> consumed by `dsart_rt.py -in search_rt`

This script is the analog of legacy `dsa110-cnf/push_to_etcd.py` — it
reads the YAML files in `configs/` (which ARE the version-controlled
source of truth) and pushes their parsed dict content into etcd under
the corresponding key. Run on h23 (or any host with dsautils +
network access to etcdv3service.pro.pvt).

The orchestrator does NOT re-read /cnf/<instance>_rt on every verb —
it loads the config once per `start` verb. So pushing a config update
takes effect at the next `start` cycle; no rolling-restart needed.

Usage::

    python tools/ops/push_dsart_to_etcd.py [--instance pipeline_rt|search_rt|all]
                                           [--config-dir configs]
                                           [--dry-run]

The default is `--instance all`; both YAMLs are pushed.

Op-point sets (x2, docs/x2/SPEC.md §10)
---------------------------------------
With ``--opoint <selection>`` (``1x``, ``1x-legacy`` or a 2x op-point id
such as ``2x-gridded-ns2-cint4``, see ``dsart.x2.opoint.FORMATS_2X``) the
script NEVER touches the live keys. It writes the selection's set to

  /cnf/opoints/<sel>/pipeline_rt, /cnf/opoints/<sel>/search_rt, /cnf/opoints/<sel>/meta

and is a DRY RUN unless ``--write`` is given. The source is the config
set's own file (``configs/opoints/<set>/<file>``) or, for 1x and 1x-legacy,
the base file when it has none. A 2x selection needs ``--plan <v3 npz>``
(built by ``tools/build_dm_plan_tiers.py --opoint <id>``): the plan must be
THE plan of that op-point (plan_id equal to a rebuild), and the 2x template's
``${var}`` values (n_sub, n_streams, plan path, per-half owned coarse DMs, ...)
are rendered from the op-point and the plan; ``--plan-node-path`` is the
plan's path on the nodes if it differs from ``--plan``.
``--seed-from-live`` (1x, 1x-legacy only) takes the live /cnf/pipeline_rt
and /cnf/search_rt instead — an exact snapshot of what production runs,
drift included. Each config is stamped ``opoint: <op-point id>`` and
``opoint_selection: <sel>`` (+ ``plan_id`` / ``plan_path`` for 2x). Every
run prints a STRUCTURAL diff against what is in /cnf/opoints/<sel>/*
(``--diff-live`` also against the live keys = what a web-UI Start with this
selection would change) and validates the set (stamp, explicit time units,
cross-set file references, X2TODO placeholders / x2_todo entries).
``--write`` refuses a set with problems, and a set with placeholders unless
``--allow-placeholders``. The web-UI Start control is the only writer of the
live keys for op-point switches.

    python tools/ops/push_dsart_to_etcd.py --opoint 2x-gridded-ns2-cint4 \\
        --plan /home/ubuntu/data/dm_plans/dm_plan_2x_ns2.npz            # diff + validate
    python tools/ops/push_dsart_to_etcd.py --opoint 2x-gridded-ns2-cint4 \\
        --plan /home/ubuntu/data/dm_plans/dm_plan_2x_ns2.npz --write    # store the set
    python tools/ops/push_dsart_to_etcd.py --opoint 1x-legacy --seed-from-live --write
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


_INSTANCE_TO_FILENAME = {
    "pipeline_rt": "dsart_pipeline_rt.yaml",
    "search_rt": "dsart_search_rt.yaml",
}

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"

#: Exit codes of the op-point mode.
RC_OK, RC_USAGE, RC_INVALID, RC_PLACEHOLDERS, RC_VERIFY = 0, 2, 3, 4, 5


def _push_one(store: Any, instance: str, yaml_path: Path,
              dry_run: bool) -> int:
    if not yaml_path.is_file():
        print(f"[ERROR] missing config: {yaml_path}", file=sys.stderr)
        return 2
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        print(f"[ERROR] {yaml_path} did not parse to a dict", file=sys.stderr)
        return 3
    key = f"/cnf/{instance}"
    n_buf = len(cfg.get("buffers") or [])
    n_rt = len(cfg.get("routines") or [])
    print(f"  {key} <- {yaml_path}  ({n_buf} buffers, {n_rt} routines)")
    if dry_run:
        print(json.dumps(cfg, indent=2)[:1200])
        return 0
    store.put_dict(key, cfg)
    return 0


# ---------------------------------------------------------------------------
# op-point mode
# ---------------------------------------------------------------------------


def _x2_modules():
    """Import the op-point helpers, preferring THIS checkout's src/ over any
    other dsart on PYTHONPATH (the h23 live checkout may predate dsart.x2).
    The CLI imports nothing from dsart before this point."""
    if ("dsart" not in sys.modules
            and (_REPO_SRC / "dsart" / "x2" / "opoint_runtime.py").is_file()
            and str(_REPO_SRC) not in sys.path):
        sys.path.insert(0, str(_REPO_SRC))
    from dsart.x2 import config_diff, opoint_runtime

    return opoint_runtime, config_diff


def _git_sha(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"git_sha": None, "git_dirty": None}
    try:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path),
                            capture_output=True, text=True, timeout=5)
        if cp.returncode == 0:
            out["git_sha"] = cp.stdout.strip()
        cp = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=str(path),
                            capture_output=True, text=True, timeout=5)
        if cp.returncode == 0:
            out["git_dirty"] = bool(cp.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def push_opoint(
    store: Any,
    selection: str,
    *,
    instances: list[str],
    config_dir: Path,
    plan_path: str | None = None,
    plan_node_path: str | None = None,
    write: bool = False,
    seed_from_live: bool = False,
    allow_placeholders: bool = False,
    diff_live: bool = False,
    out: Callable[[str], None] = print,
    now_utc: Callable[[], str] | None = None,
) -> int:
    """Diff, validate and (with ``write``) store selection ``selection``'s set.

    Writes only ``/cnf/opoints/<sel>/*`` keys (asserted before every put).
    Returns one of the ``RC_*`` codes.
    """
    opr, cdiff = _x2_modules()
    try:
        sel = opr.validate_selection(selection)
    except opr.OpPointError as exc:
        out(f"[ERROR] {exc}")
        return RC_USAGE
    op = opr.op_of(sel)
    if seed_from_live and sel not in opr.BASE_FALLBACK:
        out(f"[ERROR] --seed-from-live is only for {sorted(opr.BASE_FALLBACK)}, not {sel}")
        return RC_USAGE
    plan = None
    if op.plan_version == 3:
        if not plan_path:
            out(f"[ERROR] {sel} needs --plan <v3 plan npz> "
                f"(tools/build_dm_plan_tiers.py --opoint {sel} --out <npz>)")
            return RC_USAGE
        from dsart.x2.plan import TieredDmPlan

        try:
            plan = TieredDmPlan.from_npz(plan_path, verify=True)
        except Exception as exc:  # noqa: BLE001 - any load/verify failure
            out(f"[ERROR] --plan {plan_path}: {exc}")
            return RC_INVALID
        bad = opr.check_plan_for(sel, plan)
        if bad:
            out(f"[ERROR] --plan {plan_path}: " + "; ".join(bad))
            return RC_INVALID
    elif plan_path:
        out(f"[ERROR] {sel} takes no --plan (its DM plan is in the routine args)")
        return RC_USAGE
    try:
        ctx = opr.render_context(sel, plan=plan, plan_path=plan_node_path or plan_path)
    except opr.OpPointError as exc:
        out(f"[ERROR] {exc}")
        return RC_INVALID

    mode = "WRITE" if write else "DRY RUN (nothing is written; add --write)"
    out(f"op-point {sel} (id {op.opoint_id}, config set {opr.config_set_of(sel)}"
        + (f", plan {ctx['plan_id']} at {ctx['plan_path']}" if plan is not None else "")
        + f"): {mode}")
    rendered: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    problems: list[str] = []
    placeholders: list[tuple[str, str, str]] = []

    for inst in instances:
        if seed_from_live:
            src_desc = f"etcd {opr.LIVE_KEYS[inst]}"
            cfg = store.get_dict(opr.LIVE_KEYS[inst])
            have = opr.rendered_selection(cfg)
            if have not in (None, sel):
                out(f"[ERROR] {src_desc} is rendered for {have!r}, not {sel!r}")
                return RC_INVALID
        else:
            try:
                path = opr.config_path(inst, sel, config_dir)
            except FileNotFoundError as exc:
                out(f"[ERROR] {exc}")
                return RC_INVALID
            src_desc = str(path)
            cfg = opr.load_yaml(path)
        if not isinstance(cfg, dict):
            out(f"[ERROR] {src_desc} is not a mapping ({type(cfg).__name__})")
            return RC_INVALID
        try:
            new = opr.render(cfg, sel, ctx, instance=inst)
        except opr.OpPointError as exc:
            out(f"[ERROR] {src_desc}: {exc}")
            return RC_INVALID
        rep = opr.validate_config(new, sel, inst)
        problems += rep.problems
        placeholders += rep.placeholders
        rendered[inst] = new
        sources[inst] = src_desc

        key = opr.set_key(sel, inst)
        cur = store.get_dict(key)
        entries = cdiff.structural_diff(cur, new)
        s = cdiff.summarize(entries)
        n_rt = len(new.get("routines") or [])
        out("")
        out(f"== {key} <- {src_desc}  ({n_rt} routines, sha256 "
            f"{cdiff.canonical_sha256(new)[:16]})")
        if cur is None:
            out(f"   (key absent: the whole config would be added, {s['added']} leaves)")
        elif not entries:
            out("   (no change)")
        else:
            out(f"   vs current: +{s['added']} -{s['removed']} ~{s['changed']}")
            for line in cdiff.format_diff(entries):
                out(line)
        if diff_live:
            live = store.get_dict(opr.LIVE_KEYS[inst])
            le = cdiff.structural_diff(live, new)
            ls = cdiff.summarize(le)
            out(f"   vs LIVE {opr.LIVE_KEYS[inst]} (what a Start with {sel} "
                f"would change): +{ls['added']} -{ls['removed']} ~{ls['changed']}")
            for line in cdiff.format_diff(le):
                out("  " + line)

    # Every time-unit parameter explicit with this op-point's value
    # (the code defaults are 2x; opoint_runtime.time_units_for).
    problems += opr.check_time_units(rendered, sel)
    notes: list[str] = []
    for cfg in rendered.values():
        notes += opr.recalibration_notes(cfg)
    if not seed_from_live:
        # The C2 criteria file is not stored in etcd (C2 reads it on h23), but
        # it is part of the set: its placeholders gate the push as well.
        try:
            crit_path = opr.config_path("c2_criteria", sel, config_dir)
            crit = opr.load_yaml(crit_path)
            rep = opr.validate_config(crit, sel, "c2_criteria", rendered=False)
            problems += rep.problems
            placeholders += rep.placeholders
            notes += opr.recalibration_notes(crit)
        except FileNotFoundError as exc:
            problems.append(str(exc))
    for n in notes:
        out(f"  note (recalibrate at CP2, not blocking): {n}")

    out("")
    out(f"validation: {len(problems)} problem(s), {len(placeholders)} placeholder(s)")
    for p in problems:
        out(f"  PROBLEM {p}")
    by_inst: dict[str, int] = {}
    for inst, _path, _tok in placeholders:
        by_inst[inst] = by_inst.get(inst, 0) + 1
    for inst, n in sorted(by_inst.items()):
        toks = sorted({t for i, _p, t in placeholders if i == inst})
        out(f"  {inst}: {n} placeholder(s): " + ", ".join(toks[:20])
            + (" ..." if len(toks) > 20 else ""))

    if not write:
        out("dry run: nothing written.")
        return RC_INVALID if problems else RC_OK
    if problems:
        out("[REFUSED] the set has problems; nothing written.")
        return RC_INVALID
    if placeholders and not allow_placeholders:
        out("[REFUSED] the set still has placeholders; nothing written "
            "(--allow-placeholders stores it anyway; the Start control will still refuse it).")
        return RC_PLACEHOLDERS

    stamp_utc = (now_utc or (lambda: _dt.datetime.now(_dt.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ")))()
    meta_key = opr.meta_key(sel)
    meta = store.get_dict(meta_key)
    meta = dict(meta) if isinstance(meta, dict) else {}
    files = dict(meta.get("files") or {})
    for inst, cfg in rendered.items():
        files[inst] = {"source": sources[inst], "sha256": cdiff.canonical_sha256(cfg),
                       "pushed_utc": stamp_utc}
    meta.update({
        "opoint": op.opoint_id,
        "selection": sel,
        "config_set": opr.config_set_of(sel),
        "plan_id": ctx.get("plan_id"),
        "plan_path": ctx.get("plan_path"),
        "pushed_utc": stamp_utc,
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "source": "live" if seed_from_live else "yaml",
        "files": files,
        "placeholders": len(placeholders),
        **_git_sha(config_dir),
    })
    writes = [(opr.set_key(sel, inst), cfg) for inst, cfg in rendered.items()]
    writes.append((meta_key, meta))
    for key, payload in writes:
        if not key.startswith(opr.SETS_PREFIX + "/") or opr.is_live_key(key):
            raise AssertionError(f"refusing to write non-op-point key {key}")
    for key, payload in writes:
        store.put_dict(key, payload)
        out(f"  wrote {key}")
    for inst, cfg in rendered.items():
        back = store.get_dict(opr.set_key(sel, inst))
        if cdiff.canonical_sha256(back) != cdiff.canonical_sha256(cfg):
            out(f"[ERROR] read-back of {opr.set_key(sel, inst)} differs from what was written")
            return RC_VERIFY
    out(f"op-point {sel} stored ({len(rendered)} config(s) + meta); live keys untouched.")
    return RC_OK


def main(argv: list[str] | None = None, *, store: Any = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="all",
                   choices=("all", "pipeline_rt", "search_rt"))
    p.add_argument("--config-dir", default="configs",
                   help="directory holding dsart_*_rt.yaml (default: configs)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + print but do not write to etcd")
    g = p.add_argument_group("op-point sets (x2; never writes live keys)")
    g.add_argument("--opoint", default=None, metavar="SELECTION",
                   help="1x, 1x-legacy or a 2x op-point id (e.g. 2x-gridded-ns2-cint4): store "
                        "its set under /cnf/opoints/<sel>/ (dry run unless --write)")
    g.add_argument("--plan", default=None,
                   help="with a 2x --opoint: the v3 DM plan npz (tools/build_dm_plan_tiers.py)")
    g.add_argument("--plan-node-path", default=None,
                   help="the plan's path on the nodes, if it differs from --plan")
    g.add_argument("--write", action="store_true",
                   help="with --opoint: actually write /cnf/opoints/<sel>/*")
    g.add_argument("--seed-from-live", action="store_true",
                   help="with --opoint 1x|1x-legacy: snapshot the live /cnf/pipeline_rt + /cnf/search_rt")
    g.add_argument("--allow-placeholders", action="store_true",
                   help="with --opoint --write: store a set that still has X2TODO placeholders")
    g.add_argument("--diff-live", action="store_true",
                   help="with --opoint: also diff against the live keys")
    args = p.parse_args(argv)

    cfg_dir = Path(args.config_dir)
    if not cfg_dir.is_dir() and not args.seed_from_live:
        print(f"[ERROR] --config-dir {cfg_dir} not found", file=sys.stderr)
        return 2
    if args.instance == "all":
        instances = list(_INSTANCE_TO_FILENAME)
    else:
        instances = [args.instance]

    if args.opoint is None:
        if (args.write or args.seed_from_live or args.allow_placeholders or args.diff_live
                or args.plan or args.plan_node_path):
            print("[ERROR] --write/--plan/--seed-from-live/--allow-placeholders/--diff-live "
                  "need --opoint", file=sys.stderr)
            return 2
        if store is None:
            from dsautils.dsa_store import DsaStore
            store = DsaStore()
        print(f"pushing {len(instances)} instance(s) {'(dry-run)' if args.dry_run else ''}")
        rc_total = 0
        for inst in instances:
            rc = _push_one(store, inst, cfg_dir / _INSTANCE_TO_FILENAME[inst],
                           args.dry_run)
            rc_total = max(rc_total, rc)
        return rc_total

    if args.write and args.dry_run:
        print("[ERROR] --write and --dry-run are exclusive", file=sys.stderr)
        return 2
    if store is None:
        from dsautils.dsa_store import DsaStore
        store = DsaStore()
    return push_opoint(
        store, args.opoint, instances=instances, config_dir=cfg_dir,
        plan_path=args.plan, plan_node_path=args.plan_node_path,
        write=args.write, seed_from_live=args.seed_from_live,
        allow_placeholders=args.allow_placeholders, diff_live=args.diff_live,
    )


if __name__ == "__main__":
    sys.exit(main())
