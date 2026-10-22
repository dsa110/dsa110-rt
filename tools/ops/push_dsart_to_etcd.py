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
With ``--opoint {1x,1x-legacy,2x}`` the script NEVER touches the live
keys. It writes the op-point's set to

  /cnf/opoints/<op>/pipeline_rt, /cnf/opoints/<op>/search_rt, /cnf/opoints/<op>/meta

and is a DRY RUN unless ``--write`` is given. The source is the op-point's
own file (``configs/opoints/<op>/<file>``) or, for 1x and 1x-legacy, the
base file when it has none. ``--seed-from-live`` (1x, 1x-legacy only) takes
the live /cnf/pipeline_rt and /cnf/search_rt instead — an exact snapshot of
what production runs, drift included. Each config is stamped
``opoint: <op>``. Every run prints a STRUCTURAL diff against what is in
/cnf/opoints/<op>/* (``--diff-live`` also against the live keys = what a
web-UI Start with this op-point would change) and validates the set
(stamp, cross-op-point file references, X2TODO placeholders / x2_todo
entries). ``--write`` refuses a set with problems, and a set with
placeholders unless ``--allow-placeholders``. The web-UI Start control is
the only writer of the live keys for op-point switches.

    python tools/ops/push_dsart_to_etcd.py --opoint 2x              # diff + validate
    python tools/ops/push_dsart_to_etcd.py --opoint 2x --write      # store the set
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
    opoint: str,
    *,
    instances: list[str],
    config_dir: Path,
    write: bool = False,
    seed_from_live: bool = False,
    allow_placeholders: bool = False,
    diff_live: bool = False,
    out: Callable[[str], None] = print,
    now_utc: Callable[[], str] | None = None,
) -> int:
    """Diff, validate and (with ``write``) store op-point ``opoint``'s set.

    Writes only ``/cnf/opoints/<op>/*`` keys (asserted before every put).
    Returns one of the ``RC_*`` codes.
    """
    opr, cdiff = _x2_modules()
    try:
        opr.validate_name(opoint)
    except opr.OpPointError as exc:
        out(f"[ERROR] {exc}")
        return RC_USAGE
    if seed_from_live and opoint not in opr.BASE_FALLBACK:
        out(f"[ERROR] --seed-from-live is only for {sorted(opr.BASE_FALLBACK)}, not {opoint}")
        return RC_USAGE

    mode = "WRITE" if write else "DRY RUN (nothing is written; add --write)"
    out(f"op-point {opoint}: {mode}")
    stamped: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    problems: list[str] = []
    placeholders: list[tuple[str, str, str]] = []

    for inst in instances:
        if seed_from_live:
            src_desc = f"etcd {opr.LIVE_KEYS[inst]}"
            cfg = store.get_dict(opr.LIVE_KEYS[inst])
        else:
            try:
                path = opr.config_path(inst, opoint, config_dir)
            except FileNotFoundError as exc:
                out(f"[ERROR] {exc}")
                return RC_INVALID
            src_desc = str(path)
            cfg = opr.load_yaml(path)
        if not isinstance(cfg, dict):
            out(f"[ERROR] {src_desc} is not a mapping ({type(cfg).__name__})")
            return RC_INVALID
        try:
            new = opr.stamp(cfg, opoint)
        except opr.OpPointError as exc:
            out(f"[ERROR] {src_desc}: {exc}")
            return RC_INVALID
        rep = opr.validate_config(new, opoint, inst)
        problems += rep.problems
        placeholders += rep.placeholders
        stamped[inst] = new
        sources[inst] = src_desc

        key = opr.set_key(opoint, inst)
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
            out(f"   vs LIVE {opr.LIVE_KEYS[inst]} (what a Start with op-point {opoint} "
                f"would change): +{ls['added']} -{ls['removed']} ~{ls['changed']}")
            for line in cdiff.format_diff(le):
                out("  " + line)

    if not seed_from_live:
        # The C2 criteria file is not stored in etcd (C2 reads it on h23), but
        # it is part of the set: its placeholders gate the push as well.
        try:
            crit_path = opr.config_path("c2_criteria", opoint, config_dir)
            rep = opr.validate_config(opr.load_yaml(crit_path), opoint, "c2_criteria")
            problems += rep.problems
            placeholders += rep.placeholders
        except FileNotFoundError as exc:
            problems.append(str(exc))

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
    meta_key = opr.meta_key(opoint)
    meta = store.get_dict(meta_key)
    meta = dict(meta) if isinstance(meta, dict) else {}
    files = dict(meta.get("files") or {})
    for inst, cfg in stamped.items():
        files[inst] = {"source": sources[inst], "sha256": cdiff.canonical_sha256(cfg),
                       "pushed_utc": stamp_utc}
    meta.update({
        "opoint": opoint,
        "pushed_utc": stamp_utc,
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "source": "live" if seed_from_live else "yaml",
        "files": files,
        "placeholders": len(placeholders),
        **_git_sha(config_dir),
    })
    writes = [(opr.set_key(opoint, inst), cfg) for inst, cfg in stamped.items()]
    writes.append((meta_key, meta))
    for key, payload in writes:
        if not key.startswith(opr.SETS_PREFIX + "/") or opr.is_live_key(key):
            raise AssertionError(f"refusing to write non-op-point key {key}")
    for key, payload in writes:
        store.put_dict(key, payload)
        out(f"  wrote {key}")
    for inst, cfg in stamped.items():
        back = store.get_dict(opr.set_key(opoint, inst))
        if cdiff.canonical_sha256(back) != cdiff.canonical_sha256(cfg):
            out(f"[ERROR] read-back of {opr.set_key(opoint, inst)} differs from what was written")
            return RC_VERIFY
    out(f"op-point {opoint} stored ({len(stamped)} config(s) + meta); live keys untouched.")
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
    g.add_argument("--opoint", default=None, choices=("1x", "1x-legacy", "2x"),
                   help="store this op-point's set under /cnf/opoints/<op>/ (dry run unless --write)")
    g.add_argument("--write", action="store_true",
                   help="with --opoint: actually write /cnf/opoints/<op>/*")
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
        if args.write or args.seed_from_live or args.allow_placeholders or args.diff_live:
            print("[ERROR] --write/--seed-from-live/--allow-placeholders/--diff-live "
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
        write=args.write, seed_from_live=args.seed_from_live,
        allow_placeholders=args.allow_placeholders, diff_live=args.diff_live,
    )


if __name__ == "__main__":
    sys.exit(main())
