#!/usr/bin/env python3
"""Push configs/corr_setup_96.yaml into etcd at /cnf/corr_setup_96 via dsautils."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml

ETCD_KEY = "/cnf/corr_setup_96"


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here


def _default_corr_yaml() -> Path:
    return _repo_root() / "configs" / "corr_setup_96.yaml"


def _dsa_endpoint_yaml_from_file(cfg_path: Path) -> Path:
    """dsautils.DsaStore expects dsautils-style YAML: endpoints: ['host:port', ...]."""
    raw = yaml.safe_load(cfg_path.read_text())
    eps = raw.get("endpoints")
    if not isinstance(eps, list) or not eps:
        raise ValueError(f"{cfg_path}: expected non-empty list 'endpoints'")
    host_ports: list[str] = []
    for ep in eps:
        if isinstance(ep, str) and "://" in ep:
            u = urlparse(ep)
            if not u.hostname:
                raise ValueError(f"Bad endpoint URL {ep!r}")
            port = u.port or 2379
            host_ports.append(f"{u.hostname}:{port}")
        elif isinstance(ep, str) and ":" in ep:
            host_ports.append(ep)
        else:
            raise ValueError(f"Unsupported endpoint entry {ep!r}")
    fd, tmpp = tempfile.mkstemp(prefix="dsastore-", suffix=".yml")
    os.close(fd)
    p = Path(tmpp)
    p.write_text(yaml.safe_dump({"endpoints": host_ports}, sort_keys=False))
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--etcd-endpoints",
        type=Path,
        default=_repo_root() / "configs" / "etcd_endpoints.yaml",
        help="YAML with http(s)://host:port or host:port list under endpoints:",
    )
    ap.add_argument(
        "--corr-yaml",
        type=Path,
        default=_default_corr_yaml(),
        help="Source corr_setup YAML to publish",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print dict only; do not write etcd")
    ns = ap.parse_args(argv)

    # Leading provenance comment lines are YAML comments; safe_load skips them.
    data = yaml.safe_load(ns.corr_yaml.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{ns.corr_yaml}: root must be a mapping")

    if ns.dry_run:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    tmp_ep: Path | None = None
    try:
        try:
            from dsautils.dsa_store import DsaStore
        except ImportError as exc:
            raise SystemExit(
                "dsautils (dsa110-pyutils) is required for live etcd push; "
                "install it in the dsa110-rt conda env."
            ) from exc

        tmp_ep = _dsa_endpoint_yaml_from_file(ns.etcd_endpoints)
        store = DsaStore(str(tmp_ep))
        store.put_dict(ETCD_KEY, data)
    finally:
        if tmp_ep is not None:
            tmp_ep.unlink(missing_ok=True)

    print(f"seed_corr_setup_96: put_dict {ETCD_KEY} OK ({len(data)} top-level keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
