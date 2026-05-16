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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from dsautils.dsa_store import DsaStore


_INSTANCE_TO_FILENAME = {
    "pipeline_rt": "dsart_pipeline_rt.yaml",
    "search_rt": "dsart_search_rt.yaml",
}


def _push_one(store: DsaStore, instance: str, yaml_path: Path,
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="all",
                   choices=("all", "pipeline_rt", "search_rt"))
    p.add_argument("--config-dir", default="configs",
                   help="directory holding dsart_*_rt.yaml (default: configs)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + print but do not write to etcd")
    args = p.parse_args(argv)

    cfg_dir = Path(args.config_dir)
    if not cfg_dir.is_dir():
        print(f"[ERROR] --config-dir {cfg_dir} not found", file=sys.stderr)
        return 2

    store = DsaStore()
    if args.instance == "all":
        instances = list(_INSTANCE_TO_FILENAME)
    else:
        instances = [args.instance]

    print(f"pushing {len(instances)} instance(s) {'(dry-run)' if args.dry_run else ''}")
    rc_total = 0
    for inst in instances:
        rc = _push_one(store, inst, cfg_dir / _INSTANCE_TO_FILENAME[inst],
                       args.dry_run)
        rc_total = max(rc_total, rc)
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
