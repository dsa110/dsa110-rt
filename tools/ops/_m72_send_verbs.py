"""Send start/stop verbs to one or more dsart-rt instances via etcd.

Usage::

  python _m72_send_verbs.py search-start          # /cmd/search_rt/1 = start
  python _m72_send_verbs.py corr-start            # /cmd/corr_rt/<cn> = start (16)
  python _m72_send_verbs.py search-stop           # /cmd/search_rt/1 = stop
  python _m72_send_verbs.py corr-stop             # /cmd/corr_rt/<cn> = stop (16)
"""
import sys

from dsautils.dsa_store import DsaStore

CORR_CNS = (3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22)
SEARCH_CN = 1
OBS_DEC = 53.85


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    verb = sys.argv[1]
    s = DsaStore()
    if verb == "search-start":
        s.put_dict(f"/cmd/search_rt/{SEARCH_CN}",
                   {"cmd": "start", "val": None})
        print(f"sent start -> /cmd/search_rt/{SEARCH_CN}")
    elif verb == "search-stop":
        s.put_dict(f"/cmd/search_rt/{SEARCH_CN}",
                   {"cmd": "stop", "val": None})
        print(f"sent stop -> /cmd/search_rt/{SEARCH_CN}")
    elif verb == "corr-start":
        for cn in CORR_CNS:
            s.put_dict(f"/cmd/corr_rt/{cn}",
                       {"cmd": "start", "val": OBS_DEC})
            print(f"sent start val={OBS_DEC} -> /cmd/corr_rt/{cn}")
    elif verb == "corr-stop":
        for cn in CORR_CNS:
            s.put_dict(f"/cmd/corr_rt/{cn}",
                       {"cmd": "stop", "val": None})
            print(f"sent stop -> /cmd/corr_rt/{cn}")
    else:
        print(f"unknown verb {verb!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
