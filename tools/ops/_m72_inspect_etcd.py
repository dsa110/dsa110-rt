"""Inspect /cmd + /mon + /mon/service for one orchestrator instance."""
import json
import sys

from dsautils.dsa_store import DsaStore


def main():
    if len(sys.argv) < 3:
        print("usage: _m72_inspect_etcd.py <instance> <cn>", file=sys.stderr)
        sys.exit(2)
    instance = sys.argv[1]
    cn = sys.argv[2]
    s = DsaStore()
    for key in (
        f"/cmd/{instance}/{cn}",
        f"/mon/service/{instance}/{cn}",
        f"/mon/{instance}/{cn}",
    ):
        d = s.get_dict(key)
        print(f"--- {key} ---")
        try:
            print(json.dumps(d, indent=2))
        except TypeError:
            print(d)
        print()


if __name__ == "__main__":
    main()
